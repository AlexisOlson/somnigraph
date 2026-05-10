"""Audit the reranker for pathological demotions.

For each memory, use its summary as the query and compare:
  best_channel_rank = min(fts_rank, vec_rank, theme_rank)  # the floor
  reranker_rank     = position in reranker output

A memory should rank at least as well as its strongest channel suggests.
When reranker_rank >> best_channel_rank, the reranker is overruling the
retrieval signal — flag and report.

Usage:
    uv run scripts/audit_reranker_pathology.py [--limit N] [--threshold N]

Outputs a sorted report: worst pathologies first.
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
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from memory.db import get_db
from memory.vectors import serialize_f32
from memory.embeddings import embed_text
from memory.fts import sanitize_fts_query
from memory.constants import BM25_SUMMARY_WT, BM25_THEMES_WT, DATA_DIR
from memory.reranker import rerank, _load_model, _load_memory_meta

PATHOLOGY_AUDITS_DIR = DATA_DIR / "pathology_audits"


def build_channels(db, query: str, rowid_map: dict):
    """Replicate impl_recall's channel-build step. Returns the inputs rerank() needs."""
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


def build_feedback_and_hebb(db, all_ids: set):
    fb_raw = {}
    if all_ids:
        ph = ",".join("?" * len(all_ids))
        fb_rows = db.execute(
            f"SELECT memory_id, context FROM memory_events "
            f"WHERE memory_id IN ({ph}) AND event_type='feedback' ORDER BY created_at ASC",
            list(all_ids),
        ).fetchall()
        for r in fb_rows:
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
    hebb_mem_freq, hebb_qs = {}, set()
    for r in hebb_rows:
        hebb_mem_freq.setdefault(r["memory_id"], set()).add(r["query"])
        hebb_qs.add(r["query"])
    hebb_data = {"mem_freq": hebb_mem_freq, "total_queries": len(hebb_qs)}
    return fb_raw, hebb_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0,
                        help="Audit only first N memories (0=all active)")
    parser.add_argument("--threshold", type=int, default=10,
                        help="Flag when reranker_rank - best_channel_rank > threshold")
    parser.add_argument("--top", type=int, default=30,
                        help="Show top N pathologies in report")
    parser.add_argument("--query-from", choices=["summary", "themes", "content-residual"],
                        default="summary",
                        help="Use memory.summary (default), theme tokens, or "
                             "content-residual (content tokens minus summary tokens) as the "
                             "audit query. content-residual is the strongest OOD test since "
                             "synthetic GT only ever saw summary text.")
    parser.add_argument("--output", type=str, nargs="?", default=None, const="",
                        help="Write pathology JSON (used by adversarial probe "
                             "selection). Pass `--output` with no value to "
                             "auto-name to "
                             "data/pathology_audits/audit_<timestamp>.json, or "
                             "`--output PATH` to write to a specific path. "
                             "Default (flag absent): skip JSON write.")
    args = parser.parse_args()

    if not _load_model():
        print("Reranker model not loaded — nothing to audit.")
        return

    db = get_db()
    rowid_map = {r["rowid"]: r["memory_id"]
                 for r in db.execute("SELECT rowid, memory_id FROM memory_rowid_map").fetchall()}

    rows = db.execute(
        "SELECT id, summary, content, themes, category FROM memories WHERE status='active' "
        "ORDER BY created_at DESC"
    ).fetchall()
    if args.limit:
        rows = rows[: args.limit]

    print(f"Auditing {len(rows)} memories. Query source: {args.query_from}. "
          f"Threshold: rerank_rank - best_channel_rank > {args.threshold}")
    print()

    pathologies = []  # (gap, mid, summary_snippet, best_chan, rerank_rank, channels_used)
    flagged = 0
    self_miss = 0  # memory not in candidate pool when querying with own summary
    t0 = time.time()

    import re
    skipped_thin = 0
    MIN_RESIDUAL_TOKENS = 3
    tok_re = re.compile(r"[A-Za-z0-9_]+")

    for i, row in enumerate(rows):
        mid = row["id"]
        if args.query_from == "themes":
            try:
                themes = json.loads(row["themes"]) if row["themes"] else []
            except Exception:
                themes = []
            query = " ".join(str(t).replace("-", " ") for t in themes)
        elif args.query_from == "content-residual":
            summary_tokens = {t.lower() for t in tok_re.findall(row["summary"] or "")}
            content_tokens = tok_re.findall(row["content"] or "")
            residual = [t for t in content_tokens if t.lower() not in summary_tokens]
            if len(residual) < MIN_RESIDUAL_TOKENS:
                skipped_thin += 1
                continue
            query = " ".join(residual)
        else:
            query = row["summary"] or ""
        if not query.strip():
            continue

        vec_ranked, vec_distances, fts_ranked, fts_scores, theme_ranked, theme_overlap_map = \
            build_channels(db, query, rowid_map)

        all_ids = set(vec_ranked) | set(fts_ranked) | set(theme_ranked)
        if mid not in all_ids:
            self_miss += 1
            continue

        # Compute best (lowest) channel rank for the source memory
        ranks = []
        if mid in fts_ranked:
            ranks.append(("fts", fts_ranked[mid]))
        if mid in vec_ranked:
            ranks.append(("vec", vec_ranked[mid]))
        if mid in theme_ranked:
            ranks.append(("theme", theme_ranked[mid]))
        best_chan_name, best_chan_rank = min(ranks, key=lambda x: x[1])

        fb_raw, hebb_data = build_feedback_and_hebb(db, all_ids)

        result = rerank(db, query, fts_ranked, vec_ranked, fts_scores, vec_distances,
                        theme_ranked, theme_overlap_map, fb_raw, hebb_data, ppr_cache={})
        if not result:
            continue
        sorted_ids, _ = result
        if mid not in sorted_ids:
            self_miss += 1
            continue
        rerank_rank = sorted_ids.index(mid)

        gap = rerank_rank - best_chan_rank
        if gap > args.threshold:
            pathologies.append({
                "gap": gap,
                "memory_id": mid,
                "category": row["category"],
                "summary": query[:80],
                "best_chan_name": best_chan_name,
                "best_chan_rank": best_chan_rank,
                "rerank_rank": rerank_rank,
                "channels": [f"{n}={r}" for n, r in ranks],
            })
            flagged += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(rows)} processed — flagged={flagged}, self_miss={self_miss} "
                  f"({elapsed:.1f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"\nAudit complete in {elapsed:.1f}s")
    print(f"  {len(rows)} memories audited (query source: {args.query_from})")
    if args.query_from == "content-residual":
        print(f"  {skipped_thin} skipped (residual < {MIN_RESIDUAL_TOKENS} tokens)")
    print(f"  {self_miss} self-misses (target not in candidate pool)")
    print(f"  {flagged} pathologies flagged (gap > {args.threshold})")
    print()

    if not pathologies:
        return

    pathologies.sort(key=lambda p: -p["gap"])

    print(f"=== Top {min(args.top, len(pathologies))} pathologies ===\n")
    print(f"{'gap':>5}  {'id':<10} {'cat':<10} {'best_chan':<12} {'rerank':>7}  channels                    summary")
    for p in pathologies[: args.top]:
        chans = " ".join(p["channels"])
        best_chan = f"{p['best_chan_name']}={p['best_chan_rank']}"
        print(f"{p['gap']:>5}  {p['memory_id'][:8]:<10} {p['category']:<10} {best_chan:<12} "
              f"{p['rerank_rank']:>7}  {chans:<28}  {p['summary']}")

    # JSON output for adversarial probe selection (written before the
    # histogram so a console-encoding failure on the histogram doesn't lose
    # the audit data)
    if args.output is not None:
        if args.output == "":
            PATHOLOGY_AUDITS_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out_path = PATHOLOGY_AUDITS_DIR / f"audit_{ts}.json"
        else:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "audit_timestamp": datetime.now(timezone.utc).isoformat(),
            "query_source": args.query_from,
            "threshold": args.threshold,
            "memories_audited": len(rows),
            "self_misses": self_miss,
            "pathologies": pathologies,
        }, indent=2, sort_keys=True))
        print(f"\nPathology JSON written to {out_path}")

    # Histogram of gaps (ASCII-only so it renders on cp1252 consoles)
    print(f"\n=== Gap distribution ===")
    buckets = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 250), (250, 1000), (1000, 999999)]
    for lo, hi in buckets:
        n = sum(1 for p in pathologies if lo < p["gap"] <= hi)
        bar = "#" * min(n, 60)
        hi_label = str(hi) if hi < 999999 else "inf"
        print(f"  {lo:>4}-{hi_label:<5}  {n:>4}  {bar}")


if __name__ == "__main__":
    main()
