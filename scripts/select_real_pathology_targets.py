"""Adversarial target selection from real-recall pathologies.

A "real hard recall" is a feedback row where:
  - The memory was rated >= utility_threshold (genuinely useful)
  - The retrieval ranked the memory at position >= rank_threshold in the
    `recall_meta.kept` list (the model didn't think it was the strongest match)
  - The query is a real production query (reason does not start with "[probe")
  - The memory is still active (joined against `memories`)

These are real production failures: the rater determined the memory was useful
despite the model burying it. Mining these is the right adversarial training
signal — not the FTS-handicapped residual audit shape (which produces queries
the bundled crafter can't naturally reproduce — see V5+2 group analysis).

Contract mirrors `scripts/select_pathology_targets.py:select_pathology_targets`
so `probe_recall.run_probe`'s `--mix` orchestration can swap selectors without
other changes.

`recall_meta.kept` stores `(mid[:8], score)` 8-char prefixes (see
`memory/tools.py:788`). Feedback events store full `memory_id`. We match by
prefix; collision risk at 478 active memories is ~2.6e-5 — negligible.
"""

from __future__ import annotations

import bisect
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from memory.db import get_db


DEFAULT_UTILITY_THRESHOLD = 0.7
DEFAULT_RANK_THRESHOLD = 5  # 0-indexed; rank >= 5 = 6th position or later
DEFAULT_MATCH_WINDOW_SECONDS = 60
MAX_PATHOLOGY_PINS = 8  # matches audit-based selector cap


def _parse_iso(ts: str) -> float:
    """Parse '%Y-%m-%dT%H:%M:%fZ' (UTC) to epoch seconds.

    The DB writes timestamps via SQLite's strftime('%Y-%m-%dT%H:%M:%fZ','now'),
    which is UTC. Naive datetime.strptime + .timestamp() would interpret the
    parsed datetime as local time — wrong, even though the offset cancels in
    pairwise window checks. Use timezone-aware parsing so the returned epoch
    is what it claims to be.

    Returns 0.0 on parse failure (caller detects via the dropped match).
    """
    if not ts:
        return 0.0
    from datetime import datetime, timezone
    s = ts.rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return 0.0


def mine_real_pathologies(
    db,
    *,
    utility_threshold: float = DEFAULT_UTILITY_THRESHOLD,
    rank_threshold: int = DEFAULT_RANK_THRESHOLD,
    match_window_seconds: float = DEFAULT_MATCH_WINDOW_SECONDS,
) -> tuple[list[dict], dict]:
    """Mine real-recall pathologies from memory_events.

    Returns (rows, stats) where each row is:
      {memory_id, query, utility, rank, kept_size, fb_created_at, recall_created_at}
    """
    stats = {
        "recall_metas_loaded": 0,
        "recall_metas_with_kept": 0,
        "feedback_considered": 0,
        "feedback_no_query_match": 0,
        "feedback_outside_window": 0,
        "feedback_memory_not_in_kept": 0,
        "feedback_below_rank_threshold": 0,
        "feedback_inactive_memory": 0,
        "hard_real": 0,
    }

    # Active memory IDs (pathology mining ignores deleted/superseded memories;
    # consistent with the inactive-memory filter in build_gt_from_feedback.py).
    active_ids = {
        r["id"]
        for r in db.execute(
            "SELECT id FROM memories WHERE status = 'active'"
        )
    }

    # Build per-query list of (epoch_ts, prefix_to_rank, kept_size).
    # `kept` is [[prefix, score], ...] in rank order; index = rank.
    metas_by_query: dict[str, list[tuple[float, dict, int]]] = defaultdict(list)
    for r in db.execute(
        "SELECT query, created_at, context FROM memory_events "
        "WHERE event_type = 'recall_meta' AND query IS NOT NULL "
        "AND context IS NOT NULL"
    ):
        stats["recall_metas_loaded"] += 1
        try:
            ctx = json.loads(r["context"])
        except (TypeError, ValueError):
            continue
        kept = ctx.get("kept") or []
        if not kept:
            continue
        stats["recall_metas_with_kept"] += 1
        prefix_to_rank = {entry[0]: i for i, entry in enumerate(kept)}
        ts = _parse_iso(r["created_at"])
        metas_by_query[r["query"]].append((ts, prefix_to_rank, len(kept)))

    # Sort each query's metas by epoch for bisect-based closest-match lookup
    sorted_metas: dict[str, tuple[list[float], list[tuple[dict, int]]]] = {}
    for q, entries in metas_by_query.items():
        entries.sort(key=lambda x: x[0])
        sorted_metas[q] = (
            [e[0] for e in entries],
            [(e[1], e[2]) for e in entries],
        )

    rows: list[dict] = []
    for r in db.execute(
        "SELECT memory_id, query, context, created_at FROM memory_events "
        "WHERE event_type = 'feedback' AND query IS NOT NULL "
        "AND context IS NOT NULL"
    ):
        try:
            ctx = json.loads(r["context"])
        except (TypeError, ValueError):
            continue
        reason = ctx.get("reason", "") or ""
        if reason.startswith("[probe"):
            continue
        utility = ctx.get("utility")
        if utility is None or utility < utility_threshold:
            continue
        stats["feedback_considered"] += 1

        if r["memory_id"] not in active_ids:
            stats["feedback_inactive_memory"] += 1
            continue

        meta_for_q = sorted_metas.get(r["query"])
        if meta_for_q is None:
            stats["feedback_no_query_match"] += 1
            continue

        fb_ts = _parse_iso(r["created_at"])
        timestamps, payloads = meta_for_q
        # Closest timestamp via bisect (timestamps are sorted ascending).
        idx = bisect.bisect_left(timestamps, fb_ts)
        candidates_idx = []
        if idx < len(timestamps):
            candidates_idx.append(idx)
        if idx > 0:
            candidates_idx.append(idx - 1)
        if not candidates_idx:
            stats["feedback_no_query_match"] += 1
            continue
        best_idx = min(candidates_idx, key=lambda i: abs(timestamps[i] - fb_ts))
        if abs(timestamps[best_idx] - fb_ts) > match_window_seconds:
            stats["feedback_outside_window"] += 1
            continue

        prefix_to_rank, kept_size = payloads[best_idx]
        prefix = r["memory_id"][:8]
        rank = prefix_to_rank.get(prefix)
        if rank is None:
            stats["feedback_memory_not_in_kept"] += 1
            continue
        if rank < rank_threshold:
            stats["feedback_below_rank_threshold"] += 1
            continue

        stats["hard_real"] += 1
        rows.append({
            "memory_id": r["memory_id"],
            "query": r["query"],
            "utility": float(utility),
            "rank": int(rank),
            "kept_size": int(kept_size),
            "fb_created_at": r["created_at"],
            "recall_created_at": None,  # could surface via reverse lookup; not needed
        })

    return rows, stats


def aggregate_per_memory(rows: list[dict]) -> dict[str, dict]:
    """Aggregate hard-real rows per memory_id.

    Returns {mid: {worst_rank, max_utility, severity, count, sample_query}}
    where severity = max(utility * rank) across this memory's hard recalls.
    Severity drives the selection weight; sample_query is the worst single
    instance (for the bundled crafter as a hard-mode prompt example).
    """
    by_mem: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_mem[row["memory_id"]].append(row)

    agg = {}
    for mid, instances in by_mem.items():
        instances.sort(key=lambda x: -(x["utility"] * x["rank"]))
        worst = instances[0]
        agg[mid] = {
            "worst_rank": worst["rank"],
            "max_utility": worst["utility"],
            "severity": worst["utility"] * worst["rank"],
            "count": len(instances),
            "sample_query": worst["query"],
        }
    return agg


def select_real_pathology_targets(
    num_groups: int,
    *,
    memory_content: dict,
    exclude: set[str] | None = None,
    content_chars: int = 600,
    utility_threshold: float = DEFAULT_UTILITY_THRESHOLD,
    rank_threshold: int = DEFAULT_RANK_THRESHOLD,
    match_window_seconds: float = DEFAULT_MATCH_WINDOW_SECONDS,
) -> tuple[list[dict], dict]:
    """Select up to `num_groups` real-recall pathology memories as probe targets.

    Mirrors `select_pathology_targets.select_pathology_targets` so
    `probe_recall.run_probe`'s `--mix` orchestration is selector-agnostic.

    Args:
        num_groups: budget — return at most this many target entries
        memory_content: {mid: {summary, themes, category}} from probe_recall
        exclude: memory IDs to skip (already picked by another selector)
        content_chars: per-target content snippet length
        utility_threshold/rank_threshold/match_window_seconds: mining filters

    Returns:
        (target_list, info) where target_list mirrors the shape produced by
        probe_recall.select_targets's eligible-list pass; info reports
        provenance for logging.
    """
    db = get_db()
    try:
        rows, mine_stats = mine_real_pathologies(
            db,
            utility_threshold=utility_threshold,
            rank_threshold=rank_threshold,
            match_window_seconds=match_window_seconds,
        )
        agg = aggregate_per_memory(rows)

        # Per-memory pin count (probe_target events) — same source as the
        # audit-based selector, with the same 8-pin cap.
        pathology_mids = set(agg.keys())
        pin_counts: Counter = Counter()
        if pathology_mids:
            for r in db.execute(
                "SELECT memory_id, COUNT(*) AS n FROM memory_events "
                "WHERE event_type = 'probe_target' GROUP BY memory_id"
            ):
                if r["memory_id"] in pathology_mids:
                    pin_counts[r["memory_id"]] = r["n"]

        info = {
            "source": "real-recall",
            "utility_threshold": utility_threshold,
            "rank_threshold": rank_threshold,
            "match_window_seconds": match_window_seconds,
            "hard_real_rows": mine_stats["hard_real"],
            "unique_pathology_memories": len(agg),
            "skipped_at_cap": 0,
            "skipped_excluded": 0,
            "skipped_inactive": 0,  # active_ids filter already applied in mining
            "selected": 0,
            "mine_stats": mine_stats,
        }

        if not agg:
            return [], info

        excluded = exclude or set()
        eligible: list[tuple[str, dict]] = []
        for mid, summary in agg.items():
            if mid in excluded:
                info["skipped_excluded"] += 1
                continue
            if pin_counts.get(mid, 0) >= MAX_PATHOLOGY_PINS:
                info["skipped_at_cap"] += 1
                continue
            if mid not in memory_content:
                # Active per memories table but absent from probe_recall's
                # memory_content map — possible if probe_recall filtered
                # further (e.g., min summary length). Skip rather than crash.
                info["skipped_inactive"] += 1
                continue
            eligible.append((mid, summary))

        if not eligible:
            return [], info

        # Load full content for eligible candidates only
        candidates: list[dict] = []
        for mid, summary in eligible:
            row = db.execute(
                "SELECT content FROM memories WHERE id = ?", (mid,)
            ).fetchone()
            if row is None:
                continue
            candidates.append({
                "id": mid,
                "themes": memory_content[mid].get("themes", "[]"),
                "summary": memory_content[mid].get("summary", ""),
                "category": memory_content[mid].get("category", ""),
                "content": row["content"][:content_chars],
                "_pathology_severity": summary["severity"],
                "_pathology_worst_rank": summary["worst_rank"],
                "_pathology_max_utility": summary["max_utility"],
                "_pathology_count": summary["count"],
                "_pathology_sample_query": summary["sample_query"],
                "_pathology_pin_count": pin_counts.get(mid, 0),
            })

        if not candidates:
            return [], info

        # Efraimidis-Spirakis weighted sampling without replacement.
        # weight = severity (utility * worst_rank), softened by 1/(1+pins) so
        # memories already partway pinned cede priority. Mirrors the audit
        # selector's gap-based weighting shape; only the weight source differs.
        def _weight(c: dict) -> float:
            sev = max(c["_pathology_severity"], 0.01)
            pins = c["_pathology_pin_count"]
            return sev / (1.0 + pins)

        keyed = []
        for c in candidates:
            w = max(_weight(c), 1e-9)
            u = random.random()
            keyed.append((u ** (1.0 / w), c))
        keyed.sort(key=lambda x: x[0], reverse=True)
        selected = [c for _, c in keyed[:num_groups]]
        info["selected"] = len(selected)
        return selected, info
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Inspection / standalone CLI
# ---------------------------------------------------------------------------


def _print_distribution(rows: list[dict], rank_threshold: int) -> None:
    """Print rank distribution and per-memory aggregates for inspection."""
    if not rows:
        print("\nNo hard-real pathology rows found.")
        return

    print(f"\nRank distribution among hard real (utility-weighted, "
          f"threshold rank >= {rank_threshold}):")
    bins = [(rank_threshold, rank_threshold + 2),
            (rank_threshold + 2, rank_threshold + 5),
            (rank_threshold + 5, rank_threshold + 10),
            (rank_threshold + 10, 999)]
    for lo, hi in bins:
        n = sum(1 for x in rows if lo <= x["rank"] < hi)
        bar = "#" * min(n, 60)
        hi_label = str(hi) if hi < 999 else "+"
        print(f"  rank {lo:>2}-{hi_label:<4} : {n:>4} {bar}")

    agg = aggregate_per_memory(rows)
    print(f"\nUnique memories with hard-real recalls: {len(agg)}")

    multi = [(mid, s) for mid, s in agg.items() if s["count"] >= 2]
    multi.sort(key=lambda x: -x[1]["count"])
    if multi:
        print(f"\nMemories with multiple hard-real recalls (top 15):")
        for mid, s in multi[:15]:
            q_short = s["sample_query"][:60]
            print(f"  {mid[:8]} count={s['count']:>2} "
                  f"worst_rank={s['worst_rank']:>2} "
                  f"util={s['max_utility']:.2f}  q='{q_short}'")

    print(f"\nTop 20 by severity (utility * worst_rank):")
    by_sev = sorted(agg.items(), key=lambda x: -x[1]["severity"])[:20]
    for mid, s in by_sev:
        q_short = s["sample_query"][:60]
        print(f"  sev={s['severity']:>5.2f} util={s['max_utility']:.2f} "
              f"rank={s['worst_rank']:>2} {mid[:8]}  q='{q_short}'")


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Mine real-recall pathologies (high-utility memories that "
                    "ranked low in production retrieval) for adversarial probing."
    )
    parser.add_argument("--utility", type=float, default=DEFAULT_UTILITY_THRESHOLD,
                        help=f"Minimum utility rating (default: {DEFAULT_UTILITY_THRESHOLD})")
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK_THRESHOLD,
                        help=f"Minimum 0-indexed rank (default: {DEFAULT_RANK_THRESHOLD})")
    parser.add_argument("--window", type=float, default=DEFAULT_MATCH_WINDOW_SECONDS,
                        help=f"Max seconds between feedback and recall_meta "
                             f"(default: {DEFAULT_MATCH_WINDOW_SECONDS})")
    args = parser.parse_args()

    if "SOMNIGRAPH_DATA_DIR" not in os.environ:
        os.environ["SOMNIGRAPH_DATA_DIR"] = os.path.expanduser("~/.claude/data")
    if "SOMNIGRAPH_EMBEDDING_BACKEND" not in os.environ:
        os.environ["SOMNIGRAPH_EMBEDDING_BACKEND"] = "fastembed"

    db = get_db()
    try:
        print(f"Mining real-recall pathologies "
              f"(utility >= {args.utility}, rank >= {args.rank}, "
              f"match window <= {args.window:.0f}s)...")
        rows, stats = mine_real_pathologies(
            db,
            utility_threshold=args.utility,
            rank_threshold=args.rank,
            match_window_seconds=args.window,
        )
    finally:
        db.close()

    print(f"\nMining stats:")
    for k, v in stats.items():
        print(f"  {k:<35} {v}")
    _print_distribution(rows, args.rank)


if __name__ == "__main__":
    _main()
