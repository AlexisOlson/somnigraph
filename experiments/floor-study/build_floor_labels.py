#!/usr/bin/env python3
"""Reconstruct use/ignore labels for the proactive-injection floor study.

Arc step 3 / autonomous-experiments #2. Deterministic replay over the event
log — NO embedding, NO paid API, NO writes to the source store.

PROVENANCE (read this before trusting any number):
  The artifact assumed a local-fastembed store so RRF could be recomputed for
  free. The actual copied production store is OpenAI-1536-dim. Recomputing RRF
  would require OpenAI query embeddings (a paid call, forbidden by the study's
  cost rail) and fastembed cannot substitute (384-dim, incompatible vector
  space; db.py hard-fails on the mismatch). So per the artifact's Step-2
  fallback ("if recall_meta already stored per-candidate scores, use them"),
  this script uses the *stored* per-candidate scores from recall_meta.

  Those stored scores are the pipeline's fused score at decision time. Their
  meaning depends on which scoring path was live:
    - Reranker era (LightGBM live): stored score = reranker output. CONTAMINATED
      for this study. Detectable: reranker outputs run large (top_score up to
      2.2); RRF-fusion tops out ~0.27.
    - RRF-fallback era (reranker_model.txt absent, formula path): stored score =
      rrf_fuse (RRF fusion + UCB exploration) + hebbian + PPR-expansion. This is
      the pipeline's *no-reranker, no-LLM* fused score — the same cheap path the
      hint layer's floor would gate on. This is the clean population.
  We classify each turn's path by date (reranker outage began 2026-04-07) and
  cross-check by score magnitude. Primary study population = RRF-fallback era.

JOIN (matches scripts/select_real_pathology_targets.py):
  A feedback event is joined to the recall_meta with the *same query string*,
  nearest in time, within 60s (DEFAULT_MATCH_WINDOW_SECONDS).

LABELS (artifact Step 1):
  Per (turn, surfaced/kept candidate): used = a feedback event joined to that
  turn references the candidate with utility > 0; else ignored. Candidates
  surfaced but never rated are ignored (utility 0).

Usage:
  SOMNIGRAPH_DATA_DIR=<copy>  python build_floor_labels.py [--emit-beyond]
"""
import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone

MATCH_WINDOW_SECONDS = 60.0           # matches select_real_pathology_targets.py
RERANKER_OUTAGE_CUT = "2026-04-08T00:00:00Z"   # formula fallback began 2026-04-07
RERANKER_SCORE_SIGNATURE = 0.5        # stored top_score above this = reranker output


def parse_iso(s: str) -> float:
    """ISO8601 (with trailing Z) -> epoch seconds, timezone-aware."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).timestamp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("SOMNIGRAPH_DATA_DIR"),
                    help="Store copy dir (contains memory.db). Default: $SOMNIGRAPH_DATA_DIR")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "floor_labels.json"))
    ap.add_argument("--emit-beyond", action="store_true",
                    help="Also write floor_labels_beyond.jsonl (large, not committed)")
    args = ap.parse_args()

    if not args.data_dir:
        sys.exit("ERROR: set SOMNIGRAPH_DATA_DIR or pass --data-dir")
    db_path = os.path.join(args.data_dir, "memory.db")
    if not os.path.exists(db_path):
        sys.exit(f"ERROR: no memory.db at {db_path}")

    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row

    cut_epoch = parse_iso(RERANKER_OUTAGE_CUT)

    # ---- Load recall_meta turns ----
    # group by query for the join; keep per-turn detail keyed by event id
    turns = {}                       # event_id -> turn dict
    by_query = defaultdict(list)     # query -> [(epoch, event_id)]
    prefix_to_full = {}              # 8-char -> set(full ids)  (collision audit)
    for r in db.execute("SELECT id FROM memories"):
        p = r["id"][:8]
        prefix_to_full.setdefault(p, set()).add(r["id"])
    prefix_collisions = sum(1 for p, s in prefix_to_full.items() if len(s) > 1)

    last_reranker_sig_date = None
    for r in db.execute(
        "SELECT id, query, created_at, context FROM memory_events "
        "WHERE event_type='recall_meta' AND query IS NOT NULL AND query != ''"
    ):
        try:
            ctx = json.loads(r["context"]) if r["context"] else {}
        except (json.JSONDecodeError, TypeError):
            continue
        kept = ctx.get("kept", []) or []
        beyond = ctx.get("beyond_limit", []) or []
        top_score = ctx.get("top_score")
        epoch = parse_iso(r["created_at"])
        era = "reranker" if epoch < cut_epoch else "rrf"
        # magnitude cross-check: a stored top_score in reranker range overrides
        sig = top_score is not None and top_score > RERANKER_SCORE_SIGNATURE
        if sig:
            era_mag = "reranker"
            if last_reranker_sig_date is None or r["created_at"] > last_reranker_sig_date:
                last_reranker_sig_date = r["created_at"]
        else:
            era_mag = "rrf"
        # path used for the study: reranker if EITHER signal says so (conservative)
        path = "reranker" if (era == "reranker" or era_mag == "reranker") else "rrf"
        turns[r["id"]] = {
            "turn_id": r["id"],
            "created_at": r["created_at"],
            "query": r["query"],
            "top_score": top_score,
            "limit": ctx.get("limit"),
            "era_date": era,
            "era_mag": era_mag,
            "path": path,
            "kept": [(str(m), float(s)) for m, s in kept],
            "beyond": [(str(m), float(s)) for m, s in beyond],
            "used_mids": {},          # mid8 -> max utility (filled from feedback)
            "matched_fb": 0,
        }
        by_query[r["query"]].append((epoch, r["id"]))

    for q in by_query:
        by_query[q].sort()

    # ---- Join feedback -> nearest same-query recall_meta within window ----
    stats = {
        "recall_metas_loaded": len(turns),
        "feedback_considered": 0,
        "feedback_no_query": 0,
        "feedback_no_query_match": 0,
        "feedback_outside_window": 0,
        "feedback_matched": 0,
        "feedback_window_collisions": 0,   # >1 same-query turn within window
    }
    for r in db.execute(
        "SELECT memory_id, query, created_at, context FROM memory_events "
        "WHERE event_type='feedback'"
    ):
        stats["feedback_considered"] += 1
        q = r["query"]
        if not q:
            stats["feedback_no_query"] += 1
            continue
        cands = by_query.get(q)
        if not cands:
            stats["feedback_no_query_match"] += 1
            continue
        try:
            ctx = json.loads(r["context"]) if r["context"] else {}
        except (json.JSONDecodeError, TypeError):
            ctx = {}
        util = ctx.get("utility")
        fb_ts = parse_iso(r["created_at"])
        # nearest same-query turn
        best = min(cands, key=lambda c: abs(c[0] - fb_ts))
        if abs(best[0] - fb_ts) > MATCH_WINDOW_SECONDS:
            stats["feedback_outside_window"] += 1
            continue
        within = sum(1 for c in cands if abs(c[0] - fb_ts) <= MATCH_WINDOW_SECONDS)
        if within > 1:
            stats["feedback_window_collisions"] += 1
        stats["feedback_matched"] += 1
        turn = turns[best[1]]
        turn["matched_fb"] += 1
        if util is not None:
            mid8 = r["memory_id"][:8]
            prev = turn["used_mids"].get(mid8)
            if prev is None or util > prev:
                turn["used_mids"][mid8] = util

    # ---- Emit kept label rows (primary population) ----
    labels_kept = []
    beyond_rows_f = None
    if args.emit_beyond:
        beyond_path = os.path.join(os.path.dirname(args.out), "floor_labels_beyond.jsonl")
        beyond_rows_f = open(beyond_path, "w", encoding="utf-8")

    # censoring / base-rate accumulators, split by path
    agg = {p: {"kept_n": 0, "kept_used": 0, "kept_rated": 0, "kept_rated_zero": 0,
               "beyond_n": 0, "beyond_used": 0,
               "turns": 0, "turns_with_any_used": 0}
           for p in ("rrf", "reranker")}

    for t in turns.values():
        p = t["path"]
        agg[p]["turns"] += 1
        used_this_turn = 0
        for rank, (mid8, score) in enumerate(t["kept"]):
            util = t["used_mids"].get(mid8)
            rated = util is not None
            used = rated and util > 0
            if used:
                used_this_turn += 1
            labels_kept.append({
                "turn_id": t["turn_id"],
                "created_at": t["created_at"],
                "path": p,
                "memory_id": mid8,
                "rank": rank,
                "rrf_score": score,
                "rated": rated,
                "utility": util,
                "used": used,
            })
            agg[p]["kept_n"] += 1
            agg[p]["kept_used"] += int(used)
            agg[p]["kept_rated"] += int(rated)
            agg[p]["kept_rated_zero"] += int(rated and not used)
        if used_this_turn:
            agg[p]["turns_with_any_used"] += 1
        for rank, (mid8, score) in enumerate(t["beyond"]):
            util = t["used_mids"].get(mid8)
            used = util is not None and util > 0
            agg[p]["beyond_n"] += 1
            agg[p]["beyond_used"] += int(used)
            if beyond_rows_f is not None:
                beyond_rows_f.write(json.dumps({
                    "turn_id": t["turn_id"], "created_at": t["created_at"],
                    "path": p, "memory_id": mid8,
                    "rank": len(t["kept"]) + rank, "rrf_score": score, "used": used,
                }) + "\n")
    if beyond_rows_f is not None:
        beyond_rows_f.close()

    def rate(a, b):
        return (a / b) if b else None

    base_rates = {}
    for p in ("rrf", "reranker"):
        a = agg[p]
        base_rates[p] = {
            "turns": a["turns"],
            "kept_candidates": a["kept_n"],
            "kept_used": a["kept_used"],
            "kept_base_rate": rate(a["kept_used"], a["kept_n"]),
            "kept_rated": a["kept_rated"],
            "kept_rated_zero": a["kept_rated_zero"],
            "kept_unrated": a["kept_n"] - a["kept_rated"],
            "explicit_base_rate": rate(a["kept_used"], a["kept_rated"]),
            "turns_with_any_used": a["turns_with_any_used"],
            "beyond_candidates": a["beyond_n"],
            "beyond_used": a["beyond_used"],
            "beyond_used_rate": rate(a["beyond_used"], a["beyond_n"]),
        }

    out = {
        "meta": {
            "script": os.path.basename(__file__),
            "data_dir": args.data_dir,
            "db": db_path,
            "match_window_seconds": MATCH_WINDOW_SECONDS,
            "reranker_outage_cut": RERANKER_OUTAGE_CUT,
            "reranker_score_signature": RERANKER_SCORE_SIGNATURE,
            "last_reranker_signature_date": last_reranker_sig_date,
            "prefix_collisions_8char": prefix_collisions,
            "join_stats": stats,
            "base_rates": base_rates,
            "score_provenance": (
                "STORED per-candidate scores from recall_meta (not recomputed). "
                "RRF-era stored score = rrf_fuse(RRF+UCB) + hebbian + PPR-expansion "
                "(pipeline no-reranker path). Reranker-era = LightGBM output "
                "(excluded from primary). See module docstring."
            ),
            "label_definition": "used = joined feedback with utility > 0; else ignored (incl. never rated).",
        },
        "labels_kept": labels_kept,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # ---- console summary ----
    print("=== JOIN STATS ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    orphan = stats["feedback_no_query"] + stats["feedback_no_query_match"] + stats["feedback_outside_window"]
    print(f"  orphaned (no match/window): {orphan}  "
          f"({rate(orphan, stats['feedback_considered']):.3f} of feedback)")
    print("=== PROVENANCE ===")
    print(f"  last reranker-signature (top_score>{RERANKER_SCORE_SIGNATURE}) date: {last_reranker_sig_date}")
    print(f"  prefix_collisions_8char: {prefix_collisions}")
    print("=== BASE RATES ===")
    for p in ("rrf", "reranker"):
        b = base_rates[p]
        print(f"  [{p}] turns={b['turns']} kept={b['kept_candidates']} "
              f"used={b['kept_used']} base_rate={b['kept_base_rate']}")
        print(f"       beyond={b['beyond_candidates']} beyond_used={b['beyond_used']} "
              f"beyond_used_rate={b['beyond_used_rate']}")
    print(f"\nwrote {args.out}  ({len(labels_kept)} kept rows)")


if __name__ == "__main__":
    main()
