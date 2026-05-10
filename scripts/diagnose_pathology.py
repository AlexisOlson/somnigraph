"""Per-feature culprit analysis for stubborn pathology cases.

For each flagged memory, builds the audit's content-residual query, reruns the
reranker, and uses LightGBM's `pred_contrib=True` (SHAP-style) to compare
per-feature contributions between the buried target and the reranker's top-1.

The largest negative `(target_contrib - top1_contrib)` deltas point at the
features that pushed the model toward the wrong choice. Aggregating across
several pathologies surfaces the dominant remaining bias channel.

Usage:
    uv run scripts/diagnose_pathology.py [--top-n 8] [--mid <id> ...]
"""
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "lightgbm",
#   "fastembed",
#   "numpy",
#   "tiktoken",
#   "sqlite-vec",
# ]
# ///

import argparse
import json
import math
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from memory.db import get_db
from memory.embeddings import embed_text
from memory.fts import sanitize_fts_query
from memory.vectors import serialize_f32
from memory.constants import (
    BM25_SUMMARY_WT, BM25_THEMES_WT, DEFAULT_DECAY_RATE, PPR_DAMPING,
)
from memory.reranker import (
    FEATURE_NAMES, _load_model, _load_memory_meta, _compute_proximity,
)


# Diagnostic-only: replicates rerank()'s feature build so we can call
# model.predict(features, pred_contrib=True). If this drifts from reranker.py,
# update both. The audit (audit_reranker_pathology.py) and this share an audit
# of the same logic, so a drift would surface there first.
def build_features(
    db, query, candidate_list, fts_ranked, vec_ranked, theme_ranked,
    fts_scores, vec_distances, theme_overlap_map, ppr_scores,
    feedback_raw, hebb_data, memory_meta,
):
    nan = float("nan")
    pool_size = len(candidate_list)
    abs_best_fts = abs(min(fts_scores.values())) if fts_scores else 0.0
    max_vec = max(vec_distances.values()) if vec_distances else 0.0

    # Hebbian PMI (mirrors reranker.py)
    hebbian_pmi_map = {}
    if hebb_data and hebb_data.get("total_queries", 0) >= 5:
        hebb_mem_freq = hebb_data["mem_freq"]
        hebb_total = hebb_data["total_queries"]
        hebb_mem_count = {mid: len(qs) for mid, qs in hebb_mem_freq.items()}

        def _best_rank(mid):
            ranks = []
            if mid in fts_ranked:
                ranks.append(fts_ranked[mid])
            if mid in vec_ranked:
                ranks.append(vec_ranked[mid])
            return min(ranks) if ranks else 9999

        seed_ids = sorted(candidate_list, key=_best_rank)[:5]
        for cand in candidate_list:
            if cand in seed_ids:
                continue
            total_pmi = 0.0
            for seed in seed_ids:
                if seed not in hebb_mem_count or cand not in hebb_mem_count:
                    continue
                joint = len(hebb_mem_freq.get(seed, set()) &
                           hebb_mem_freq.get(cand, set()))
                if joint < 2:
                    continue
                p_s = hebb_mem_count[seed] / hebb_total
                p_c = hebb_mem_count[cand] / hebb_total
                p_j = joint / hebb_total
                if p_s * p_c == 0:
                    continue
                pmi = math.log2(p_j / (p_s * p_c))
                if pmi > 0:
                    total_pmi += pmi
            if total_pmi > 0:
                hebbian_pmi_map[cand] = total_pmi

    query_terms = [t for t in query.lower().split() if len(t) > 1]
    q_len = len(query_terms)

    idf_stats = memory_meta.get("__idf_stats__", {})
    total_docs = idf_stats.get("total_docs", 1)
    term_doc_freq = idf_stats.get("term_doc_freq", {})
    query_idf_var = 0.0
    if query_terms and term_doc_freq:
        idfs = []
        for term in query_terms:
            df = term_doc_freq.get(term, 0)
            idfs.append(math.log((total_docs + 1) / (df + 1)))
        if len(idfs) > 1:
            idf_mean = sum(idfs) / len(idfs)
            query_idf_var = sum((x - idf_mean) ** 2 for x in idfs) / len(idfs)

    session_retrievals = memory_meta.get("__session_retrievals__", {})
    session_recency_map = {}
    for sess_id, events in session_retrievals.items():
        qpositions = [i for i, (q, _) in enumerate(events) if q == query]
        if not qpositions:
            continue
        qpos = qpositions[-1]
        for j in range(qpos - 1, -1, -1):
            _, mid = events[j]
            ago = qpos - j
            if mid not in session_recency_map or ago < session_recency_map[mid]:
                session_recency_map[mid] = ago

    now_ts = time.time()
    fb_lambda = 0.01

    n = len(candidate_list)
    features = np.zeros((n, 31), dtype=np.float32)

    for i, mid in enumerate(candidate_list):
        features[i, 0] = fts_ranked.get(mid, nan)
        features[i, 1] = vec_ranked.get(mid, nan)
        features[i, 2] = theme_ranked.get(mid, nan)
        features[i, 3] = ppr_scores.get(mid, 0.0)
        features[i, 4] = fts_scores.get(mid, nan)
        features[i, 5] = vec_distances.get(mid, nan)
        features[i, 6] = theme_overlap_map.get(mid, 0)

        fb = feedback_raw.get(mid)
        if fb and fb["count"] > 0:
            features[i, 7] = fb["utilities"][-1]
            features[i, 8] = sum(fb["utilities"]) / fb["count"]
            features[i, 9] = fb["count"]
        else:
            features[i, 7] = nan
            features[i, 8] = nan
            features[i, 9] = 0

        features[i, 10] = hebbian_pmi_map.get(mid, 0.0)

        m = memory_meta.get(mid)
        if m:
            features[i, 11] = m["category"]
            features[i, 12] = m["priority"]
            features[i, 13] = m["age_days"]
            features[i, 14] = m["token_count"]
            features[i, 15] = m["edge_count"]
            features[i, 16] = m["theme_count"]
            features[i, 17] = m["confidence"]
        else:
            features[i, 11] = 1; features[i, 12] = 5; features[i, 13] = 0.0
            features[i, 14] = 200; features[i, 15] = 0; features[i, 16] = 0
            features[i, 17] = 0.5

        if m and query_terms:
            content_set = set(m.get("content_tokens", []))
            matched = sum(1 for t in query_terms if t in content_set)
            features[i, 18] = matched / len(query_terms)
        else:
            features[i, 18] = 0.0

        if m and len(query_terms) > 1:
            features[i, 19] = _compute_proximity(query_terms, m.get("content_tokens", []))
        else:
            features[i, 19] = 0.0

        features[i, 20] = query_idf_var

        if m:
            features[i, 21] = m.get("burstiness", 0.0)
            features[i, 22] = m.get("betweenness", 0.0)
            features[i, 23] = m.get("diversity_score", 0.5)
        else:
            features[i, 21] = 0.0; features[i, 22] = 0.0; features[i, 23] = 0.5

        if m:
            fb_ts = m.get("fb_timestamps", [])
            if fb_ts:
                ws = 0.0; wsum = 0.0
                for ts, util in fb_ts:
                    age_fb = (now_ts - ts) / 86400.0
                    w = math.exp(-fb_lambda * age_fb)
                    ws += w * util
                    wsum += w
                features[i, 24] = ws / wsum if wsum > 0 else nan
            else:
                features[i, 24] = nan
        else:
            features[i, 24] = nan

        features[i, 25] = session_recency_map.get(mid, nan)
        features[i, 26] = q_len
        features[i, 27] = pool_size
        if mid in fts_scores and abs_best_fts > 0:
            features[i, 28] = abs(fts_scores[mid]) / abs_best_fts
        else:
            features[i, 28] = nan
        if mid in vec_distances and max_vec > 0:
            features[i, 29] = vec_distances[mid] / max_vec
        else:
            features[i, 29] = nan
        features[i, 30] = m.get("decay_rate", DEFAULT_DECAY_RATE) if m else DEFAULT_DECAY_RATE

    return features


def build_channels(db, query, rowid_map):
    """Mirrors audit_reranker_pathology.build_channels."""
    qe = embed_text(query)
    vec_results = db.execute(
        "SELECT rowid, distance FROM memory_vec WHERE embedding MATCH ? AND k = 200 ORDER BY distance",
        (serialize_f32(qe),),
    ).fetchall()
    vec_ranked, vec_distances = {}, {}
    for rank, row in enumerate(vec_results):
        mid = rowid_map.get(row["rowid"])
        if mid:
            vec_ranked[mid] = rank
            vec_distances[mid] = row["distance"]

    fts_ranked, fts_scores = {}, {}
    fts_q = sanitize_fts_query(query)
    try:
        fts_results = db.execute(
            f"SELECT rowid, bm25(memory_fts, {BM25_SUMMARY_WT}, {BM25_THEMES_WT}) as rank "
            "FROM memory_fts WHERE memory_fts MATCH ? ORDER BY rank LIMIT 150",
            (fts_q,),
        ).fetchall()
        for rank, row in enumerate(fts_results):
            mid = rowid_map.get(row["rowid"])
            if mid:
                fts_ranked[mid] = rank
                fts_scores[mid] = row["rank"]
    except sqlite3.OperationalError:
        pass

    query_tokens = set(query.lower().split())
    theme_ranked, theme_overlap_map = {}, {}
    if query_tokens:
        rows = db.execute(
            "SELECT id, themes FROM memories WHERE status='active' "
            "AND themes IS NOT NULL AND themes != '[]'"
        ).fetchall()
        overlaps = []
        for r in rows:
            try:
                themes = json.loads(r["themes"]) if r["themes"] else []
            except Exception:
                continue
            tt = set()
            for t in themes:
                tt.update(str(t).lower().replace("-", " ").split())
            o = len(query_tokens & tt)
            if o > 0:
                overlaps.append((r["id"], o))
                theme_overlap_map[r["id"]] = o
        overlaps.sort(key=lambda x: (-x[1], x[0]))
        for rank, (mid, _) in enumerate(overlaps):
            theme_ranked[mid] = rank

    return vec_ranked, vec_distances, fts_ranked, fts_scores, theme_ranked, theme_overlap_map


def build_feedback_and_hebb(db, all_ids):
    fb_raw = {}
    if all_ids:
        ph = ",".join("?" * len(all_ids))
        rows = db.execute(
            f"SELECT memory_id, context FROM memory_events "
            f"WHERE memory_id IN ({ph}) AND event_type='feedback' ORDER BY created_at ASC",
            list(all_ids),
        ).fetchall()
        for r in rows:
            try:
                ctx = json.loads(r["context"]) if r["context"] else {}
                if "utility" in ctx:
                    fb_raw.setdefault(r["memory_id"], {"utilities": [], "count": 0})
                    fb_raw[r["memory_id"]]["utilities"].append(ctx["utility"])
                    fb_raw[r["memory_id"]]["count"] += 1
            except Exception:
                pass

    lookback = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    hebb_rows = db.execute(
        "SELECT query, memory_id FROM memory_events WHERE event_type='retrieved' "
        "AND query IS NOT NULL AND query != '' AND created_at > ?",
        (lookback,),
    ).fetchall()
    mem_freq, qs = {}, set()
    for r in hebb_rows:
        mem_freq.setdefault(r["memory_id"], set()).add(r["query"])
        qs.add(r["query"])
    return fb_raw, {"mem_freq": mem_freq, "total_queries": len(qs)}


def make_content_residual_query(row, min_tokens=3, residual_tokens=10000):
    """Mirrors audit's content-residual mode (no token cap on length)."""
    tok_re = re.compile(r"[A-Za-z0-9_]+")
    summary_tokens = {t.lower() for t in tok_re.findall(row["summary"] or "")}
    content_tokens = tok_re.findall(row["content"] or "")
    residual = [t for t in content_tokens if t.lower() not in summary_tokens]
    if len(residual) < min_tokens:
        return None
    return " ".join(residual[:residual_tokens])


def find_pathologies(db, model, rowid_map, memory_meta, top_n):
    """Re-run audit pass and return top-N highest-gap pathologies."""
    rows = db.execute(
        "SELECT id, summary, content, themes, category FROM memories "
        "WHERE status='active' ORDER BY created_at DESC"
    ).fetchall()
    pathologies = []
    print(f"Scanning {len(rows)} memories for top-{top_n} pathologies...", flush=True)
    for i, row in enumerate(rows):
        mid = row["id"]
        query = make_content_residual_query(row)
        if not query:
            continue
        v_r, v_d, f_r, f_s, t_r, t_o = build_channels(db, query, rowid_map)
        all_ids = set(v_r) | set(f_r) | set(t_r)
        if mid not in all_ids:
            continue

        ranks = []
        if mid in f_r: ranks.append(("fts", f_r[mid]))
        if mid in v_r: ranks.append(("vec", v_r[mid]))
        if mid in t_r: ranks.append(("theme", t_r[mid]))
        best_chan, best_rank = min(ranks, key=lambda x: x[1])

        fb_raw, hebb_data = build_feedback_and_hebb(db, all_ids)

        candidate_list = sorted(all_ids)
        features = build_features(
            db, query, candidate_list, f_r, v_r, t_r, f_s, v_d, t_o,
            ppr_scores={}, feedback_raw=fb_raw, hebb_data=hebb_data,
            memory_meta=memory_meta,
        )
        preds = model.predict(features)
        scored = sorted(zip(candidate_list, preds), key=lambda x: -x[1])
        sorted_ids = [m for m, _ in scored]
        if mid not in sorted_ids:
            continue
        rerank_rank = sorted_ids.index(mid)
        gap = rerank_rank - best_rank
        if gap > 10:
            pathologies.append({
                "mid": mid,
                "summary": (row["summary"] or "")[:80],
                "category": row["category"],
                "query": query,
                "candidate_list": candidate_list,
                "features": features,
                "preds": preds,
                "sorted_ids": sorted_ids,
                "rerank_rank": rerank_rank,
                "best_chan": best_chan,
                "best_rank": best_rank,
                "channels": ranks,
                "gap": gap,
            })
        if (i + 1) % 50 == 0:
            print(f"  scanned {i+1}/{len(rows)} — {len(pathologies)} flagged", flush=True)

    pathologies.sort(key=lambda p: -p["gap"])
    return pathologies[:top_n]


def diagnose_one(p, model, memory_meta):
    """Emit feature-contribution comparison for one pathology."""
    feats = p["features"]
    candidate_list = p["candidate_list"]
    sorted_ids = p["sorted_ids"]

    target_idx = candidate_list.index(p["mid"])
    top1_mid = sorted_ids[0]
    top1_idx = candidate_list.index(top1_mid)

    # SHAP-style contributions: shape (n, n_features+1), last col is bias.
    contribs = model.predict(feats, pred_contrib=True)
    target_contrib = contribs[target_idx, :-1]
    top1_contrib = contribs[top1_idx, :-1]
    delta = target_contrib - top1_contrib  # negative = hurt target relative to top-1

    target_pred = float(contribs[target_idx, :].sum())
    top1_pred = float(contribs[top1_idx, :].sum())
    pred_gap = top1_pred - target_pred

    top1_meta = memory_meta.get(top1_mid, {})
    top1_summary = (top1_meta.get("summary") or "")[:60]

    print()
    print("=" * 88)
    print(f"  TARGET   {p['mid'][:8]}  rerank_rank={p['rerank_rank']}  "
          f"best_chan={p['best_chan']}={p['best_rank']}  "
          f"channels={[f'{n}={r}' for n,r in p['channels']]}")
    print(f"  SUMMARY  {p['summary']}")
    print(f"  TOP-1    {top1_mid[:8]}  pred={top1_pred:.4f}  "
          f"target_pred={target_pred:.4f}  gap={pred_gap:.4f}")
    print(f"  TOP-1 SUMMARY  {top1_summary}")
    print()
    print(f"  {'feature':<22}  {'target_val':>12}  {'top1_val':>12}  "
          f"{'tgt_contrib':>12}  {'top1_contrib':>12}  {'delta':>10}")
    print(f"  {'-'*22}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*10}")

    rows = []
    for fi in range(len(FEATURE_NAMES)):
        rows.append({
            "name": FEATURE_NAMES[fi],
            "tgt_val": float(feats[target_idx, fi]),
            "top1_val": float(feats[top1_idx, fi]),
            "tgt_c": float(target_contrib[fi]),
            "top1_c": float(top1_contrib[fi]),
            "delta": float(delta[fi]),
        })
    rows.sort(key=lambda r: r["delta"])  # most-hurting first

    def fmt(v):
        if isinstance(v, float) and (math.isnan(v)):
            return "      nan"
        return f"{v:>12.4f}" if abs(v) < 1e6 else f"{v:>12.2e}"

    for r in rows[:10]:
        print(f"  {r['name']:<22}  {fmt(r['tgt_val'])}  {fmt(r['top1_val'])}  "
              f"{fmt(r['tgt_c'])}  {fmt(r['top1_c'])}  {r['delta']:>10.4f}")
    print(f"  {'... (top 10 negative deltas shown above; sums match prediction gap)':<88}")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=8,
                        help="Number of top pathologies to diagnose (default: 8)")
    parser.add_argument("--mid", action="append", default=None,
                        help="Diagnose specific memory ID(s) instead of top-N audit")
    args = parser.parse_args()

    if not _load_model():
        print("Reranker model not loaded.")
        return
    model = _load_model()
    db = get_db()
    memory_meta = _load_memory_meta(db)
    rowid_map = {r["rowid"]: r["memory_id"]
                 for r in db.execute("SELECT rowid, memory_id FROM memory_rowid_map").fetchall()}

    if args.mid:
        # User-specified IDs: build pathology records for each.
        pathologies = []
        for mid_arg in args.mid:
            row = db.execute(
                "SELECT id, summary, content, themes, category FROM memories WHERE id LIKE ?",
                (mid_arg + "%",),
            ).fetchone()
            if not row:
                print(f"  No memory matches {mid_arg}")
                continue
            query = make_content_residual_query(row)
            if not query:
                print(f"  {row['id'][:8]} has insufficient content-residual tokens")
                continue
            v_r, v_d, f_r, f_s, t_r, t_o = build_channels(db, query, rowid_map)
            all_ids = set(v_r) | set(f_r) | set(t_r)
            if row["id"] not in all_ids:
                print(f"  {row['id'][:8]} not in candidate pool for its own residual query")
                continue
            ranks = []
            if row["id"] in f_r: ranks.append(("fts", f_r[row["id"]]))
            if row["id"] in v_r: ranks.append(("vec", v_r[row["id"]]))
            if row["id"] in t_r: ranks.append(("theme", t_r[row["id"]]))
            best_chan, best_rank = min(ranks, key=lambda x: x[1])
            fb_raw, hebb_data = build_feedback_and_hebb(db, all_ids)
            candidate_list = sorted(all_ids)
            features = build_features(
                db, query, candidate_list, f_r, v_r, t_r, f_s, v_d, t_o,
                ppr_scores={}, feedback_raw=fb_raw, hebb_data=hebb_data,
                memory_meta=memory_meta,
            )
            preds = model.predict(features)
            scored = sorted(zip(candidate_list, preds), key=lambda x: -x[1])
            sorted_ids = [m for m, _ in scored]
            rerank_rank = sorted_ids.index(row["id"])
            pathologies.append({
                "mid": row["id"],
                "summary": (row["summary"] or "")[:80],
                "category": row["category"],
                "query": query,
                "candidate_list": candidate_list,
                "features": features,
                "preds": preds,
                "sorted_ids": sorted_ids,
                "rerank_rank": rerank_rank,
                "best_chan": best_chan,
                "best_rank": best_rank,
                "channels": ranks,
                "gap": rerank_rank - best_rank,
            })
    else:
        pathologies = find_pathologies(db, model, rowid_map, memory_meta, args.top_n)

    print()
    print(f"Diagnosing {len(pathologies)} pathologies.")

    all_rows = []
    for p in pathologies:
        rows = diagnose_one(p, model, memory_meta)
        all_rows.append(rows)

    # Aggregate: mean delta per feature across all cases.
    print()
    print("=" * 88)
    print(f"  AGGREGATE across {len(all_rows)} cases — mean (target - top1) contribution")
    print(f"  Negative = feature consistently hurt target relative to top-1.")
    print("=" * 88)
    if all_rows:
        agg = {f: [] for f in FEATURE_NAMES}
        for rows in all_rows:
            for r in rows:
                agg[r["name"]].append(r["delta"])
        summary = sorted(
            ((name, sum(vals) / len(vals), min(vals), max(vals)) for name, vals in agg.items()),
            key=lambda x: x[1],
        )
        print(f"  {'feature':<22}  {'mean_delta':>12}  {'min_delta':>12}  {'max_delta':>12}")
        print(f"  {'-'*22}  {'-'*12}  {'-'*12}  {'-'*12}")
        for name, mean, mn, mx in summary[:15]:
            print(f"  {name:<22}  {mean:>12.4f}  {mn:>12.4f}  {mx:>12.4f}")


if __name__ == "__main__":
    main()
