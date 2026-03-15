"""Utility calibration study: does feedback correlate with independent relevance?

Compares live feedback scores (from memory_events) against LLM-judged ground
truth relevance (from gt_snapshot.json). The key question: is the feedback loop
learning what's actually relevant, or just reinforcing retrieval habits?

Output: correlation stats, scatter data, and a breakdown of the four quadrants
(high-feedback/high-GT, high-feedback/low-GT, low-feedback/high-GT, neither).

Usage:
    python scripts/utility_calibration.py
    python scripts/utility_calibration.py --plot       # save scatter plot
    python scripts/utility_calibration.py --verbose    # show quadrant examples
"""

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path.home() / ".somnigraph"
if not DATA_DIR.exists():
    DATA_DIR = Path.home() / ".claude" / "data"

DB_PATH = DATA_DIR / "memory.db"
GT_PATH = DATA_DIR / "tuning_studies" / "gt_snapshot.json"


def load_feedback(db_path: Path) -> dict[str, list[float]]:
    """Load all feedback utility scores per memory from memory_events."""
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT memory_id, context FROM memory_events WHERE event_type = 'feedback'"
    ).fetchall()
    db.close()

    feedback: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        mid = row["memory_id"]
        try:
            ctx = json.loads(row["context"])
            util = ctx.get("utility")
            if util is not None:
                feedback[mid].append(float(util))
        except (json.JSONDecodeError, TypeError):
            continue
    return dict(feedback)


def load_ground_truth(gt_path: Path) -> dict[str, list[float]]:
    """Load GT relevance scores per memory (aggregated across all queries)."""
    with open(gt_path) as f:
        gt = json.load(f)

    gt_scores: dict[str, list[float]] = defaultdict(list)
    for query, candidates in gt.items():
        for mid, score in candidates.items():
            gt_scores[mid].append(float(score))
    return dict(gt_scores)


def compute_stats(pairs: list[tuple[float, float]]) -> dict:
    """Compute correlation and summary stats for (feedback, gt) pairs."""
    n = len(pairs)
    if n < 3:
        return {"n": n, "error": "insufficient data"}

    fb = [p[0] for p in pairs]
    gt = [p[1] for p in pairs]

    fb_mean = sum(fb) / n
    gt_mean = sum(gt) / n

    # Pearson
    cov = sum((f - fb_mean) * (g - gt_mean) for f, g in pairs) / n
    fb_std = (sum((f - fb_mean) ** 2 for f in fb) / n) ** 0.5
    gt_std = (sum((g - gt_mean) ** 2 for g in gt) / n) ** 0.5
    pearson = cov / (fb_std * gt_std) if fb_std > 0 and gt_std > 0 else 0.0

    # Spearman (rank correlation)
    def rankdata(values):
        indexed = sorted(enumerate(values), key=lambda x: x[1])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(indexed):
            j = i
            while j < len(indexed) - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = rank
            i = j + 1
        return ranks

    fb_ranks = rankdata(fb)
    gt_ranks = rankdata(gt)
    rank_pairs = list(zip(fb_ranks, gt_ranks))
    rank_fb_mean = sum(r[0] for r in rank_pairs) / n
    rank_gt_mean = sum(r[1] for r in rank_pairs) / n
    rank_cov = sum((r[0] - rank_fb_mean) * (r[1] - rank_gt_mean) for r in rank_pairs) / n
    rank_fb_std = (sum((r[0] - rank_fb_mean) ** 2 for r in rank_pairs) / n) ** 0.5
    rank_gt_std = (sum((r[1] - rank_gt_mean) ** 2 for r in rank_pairs) / n) ** 0.5
    spearman = rank_cov / (rank_fb_std * rank_gt_std) if rank_fb_std > 0 and rank_gt_std > 0 else 0.0

    return {
        "n": n,
        "pearson": round(pearson, 4),
        "spearman": round(spearman, 4),
        "feedback_mean": round(fb_mean, 4),
        "gt_mean": round(gt_mean, 4),
        "feedback_std": round(fb_std, 4),
        "gt_std": round(gt_std, 4),
    }


def quadrant_analysis(
    pairs: list[tuple[float, float, str]],
    fb_threshold: float = 0.5,
    gt_threshold: float = 0.5,
) -> dict:
    """Classify memories into four quadrants based on feedback vs GT scores."""
    quadrants = {
        "high_fb_high_gt": [],  # Validated: feedback agrees with relevance
        "high_fb_low_gt": [],   # Self-reinforcing: feedback inflated
        "low_fb_high_gt": [],   # Coverage gap: relevant but poorly rated
        "low_fb_low_gt": [],    # Correctly ignored
    }
    for fb, gt, mid in pairs:
        if fb >= fb_threshold and gt >= gt_threshold:
            quadrants["high_fb_high_gt"].append((fb, gt, mid))
        elif fb >= fb_threshold and gt < gt_threshold:
            quadrants["high_fb_low_gt"].append((fb, gt, mid))
        elif fb < fb_threshold and gt >= gt_threshold:
            quadrants["low_fb_high_gt"].append((fb, gt, mid))
        else:
            quadrants["low_fb_low_gt"].append((fb, gt, mid))
    return quadrants


def main():
    parser = argparse.ArgumentParser(description="Utility calibration study")
    parser.add_argument("--db", type=str, default=str(DB_PATH))
    parser.add_argument("--gt", type=str, default=str(GT_PATH))
    parser.add_argument("--plot", action="store_true", help="Save scatter plot")
    parser.add_argument("--verbose", action="store_true", help="Show quadrant examples")
    parser.add_argument("--fb-threshold", type=float, default=0.5,
                        help="Feedback threshold for quadrant analysis (default: 0.5)")
    parser.add_argument("--gt-threshold", type=float, default=0.5,
                        help="GT threshold for quadrant analysis (default: 0.5)")
    args = parser.parse_args()

    db_path = Path(args.db)
    gt_path = Path(args.gt)

    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}")
        sys.exit(1)
    if not gt_path.exists():
        print(f"ERROR: Ground truth not found: {gt_path}")
        sys.exit(1)

    # Load data
    print("Loading feedback from memory_events...")
    feedback = load_feedback(db_path)
    print(f"  {len(feedback)} memories with feedback ({sum(len(v) for v in feedback.values())} total events)")

    print("Loading ground truth...")
    gt_scores = load_ground_truth(gt_path)
    print(f"  {len(gt_scores)} memories in GT ({sum(len(v) for v in gt_scores.values())} query-memory pairs)")

    # Find overlap
    overlap_mids = set(feedback.keys()) & set(gt_scores.keys())
    print(f"\nOverlap: {len(overlap_mids)} memories have both feedback and GT scores")
    print(f"  Feedback only: {len(feedback) - len(overlap_mids)}")
    print(f"  GT only: {len(gt_scores) - len(overlap_mids)}")

    if not overlap_mids:
        print("ERROR: No overlapping memories. Cannot compute correlation.")
        sys.exit(1)

    # Build paired data: mean feedback vs mean GT per memory
    pairs_with_ids = []
    for mid in overlap_mids:
        fb_mean = sum(feedback[mid]) / len(feedback[mid])
        gt_mean = sum(gt_scores[mid]) / len(gt_scores[mid])
        pairs_with_ids.append((fb_mean, gt_mean, mid))

    pairs = [(fb, gt) for fb, gt, _ in pairs_with_ids]

    # Correlation
    print("\n" + "=" * 60)
    print("CORRELATION: Mean Feedback vs Mean GT Relevance (per memory)")
    print("=" * 60)
    stats = compute_stats(pairs)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Also compute per-event correlation (where the same memory appears
    # in a specific GT query AND has feedback from that query)
    print("\n" + "=" * 60)
    print("PER-QUERY CORRELATION")
    print("=" * 60)
    # Load raw GT for per-query matching
    with open(gt_path) as f:
        gt_raw = json.load(f)

    # Load feedback with query info
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT memory_id, query, context FROM memory_events WHERE event_type = 'feedback'"
    ).fetchall()
    db.close()

    # Build query -> memory -> feedback utility
    query_feedback: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        try:
            ctx = json.loads(row["context"])
            util = ctx.get("utility")
            if util is not None and row["query"]:
                query_feedback[row["query"]][row["memory_id"]].append(float(util))
        except (json.JSONDecodeError, TypeError):
            continue

    # Match: GT queries that also have feedback events
    per_query_pairs = []
    matched_queries = 0
    for gt_query, gt_candidates in gt_raw.items():
        # Try exact match and substring match on query
        fb_for_query = query_feedback.get(gt_query, {})
        if not fb_for_query:
            continue
        matched_queries += 1
        for mid, gt_score in gt_candidates.items():
            if mid in fb_for_query:
                fb_mean = sum(fb_for_query[mid]) / len(fb_for_query[mid])
                per_query_pairs.append((fb_mean, float(gt_score)))

    print(f"  Matched queries: {matched_queries}")
    print(f"  Matched (query, memory) pairs: {len(per_query_pairs)}")
    if per_query_pairs:
        pq_stats = compute_stats(per_query_pairs)
        for k, v in pq_stats.items():
            print(f"  {k}: {v}")

    # Quadrant analysis
    print("\n" + "=" * 60)
    print(f"QUADRANT ANALYSIS (fb >= {args.fb_threshold}, gt >= {args.gt_threshold})")
    print("=" * 60)
    quadrants = quadrant_analysis(pairs_with_ids, args.fb_threshold, args.gt_threshold)
    total = len(pairs_with_ids)
    for name, items in quadrants.items():
        pct = len(items) / total * 100 if total else 0
        label = {
            "high_fb_high_gt": "Validated (feedback agrees with relevance)",
            "high_fb_low_gt": "Self-reinforcing (feedback inflated)",
            "low_fb_high_gt": "Coverage gap (relevant but underrated)",
            "low_fb_low_gt": "Correctly filtered",
        }[name]
        print(f"  {name}: {len(items)} ({pct:.1f}%) -- {label}")

    if args.verbose:
        print("\n--- Self-reinforcing examples (high feedback, low GT) ---")
        sorted_inflated = sorted(quadrants["high_fb_low_gt"], key=lambda x: x[0] - x[1], reverse=True)
        for fb, gt, mid in sorted_inflated[:5]:
            print(f"  {mid[:12]}  fb={fb:.2f}  gt={gt:.2f}  gap={fb-gt:+.2f}")

        print("\n--- Coverage gap examples (low feedback, high GT) ---")
        sorted_gap = sorted(quadrants["low_fb_high_gt"], key=lambda x: x[1] - x[0], reverse=True)
        for fb, gt, mid in sorted_gap[:5]:
            print(f"  {mid[:12]}  fb={fb:.2f}  gt={gt:.2f}  gap={fb-gt:+.2f}")

    # Never-surfaced analysis
    print("\n" + "=" * 60)
    print("NEVER-SURFACED MEMORIES")
    print("=" * 60)
    gt_only = set(gt_scores.keys()) - set(feedback.keys())
    gt_only_relevant = []
    for mid in gt_only:
        mean_gt = sum(gt_scores[mid]) / len(gt_scores[mid])
        if mean_gt >= args.gt_threshold:
            gt_only_relevant.append((mean_gt, mid))
    gt_only_relevant.sort(reverse=True)
    print(f"  Memories in GT but never given feedback: {len(gt_only)}")
    print(f"  Of those, with mean GT >= {args.gt_threshold}: {len(gt_only_relevant)}")
    if gt_only_relevant:
        print(f"  Top 5 never-surfaced relevant memories:")
        for gt_val, mid in gt_only_relevant[:5]:
            print(f"    {mid[:12]}  mean_gt={gt_val:.2f}  (appears in {len(gt_scores[mid])} queries)")

    # Feedback-only analysis (memories with feedback but not in GT candidate sets)
    fb_only = set(feedback.keys()) - set(gt_scores.keys())
    print(f"\n  Memories with feedback but not in any GT candidate set: {len(fb_only)}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    r = stats.get("spearman", 0)
    if r > 0.6:
        verdict = "Strong correlation -- feedback is tracking relevance well"
    elif r > 0.3:
        verdict = "Moderate correlation -- feedback has signal but also noise"
    elif r > 0.1:
        verdict = "Weak correlation -- feedback may be tracking habits more than relevance"
    else:
        verdict = "No meaningful correlation -- feedback is not tracking relevance"
    print(f"  Spearman r = {r:.4f}: {verdict}")

    inflated_pct = len(quadrants["high_fb_low_gt"]) / total * 100 if total else 0
    gap_pct = len(quadrants["low_fb_high_gt"]) / total * 100 if total else 0
    print(f"  Self-reinforcement risk: {inflated_pct:.1f}% of memories have inflated feedback")
    print(f"  Coverage gap: {gap_pct:.1f}% of memories are underrated despite GT relevance")
    if gt_only_relevant:
        print(f"  Never-surfaced relevant: {len(gt_only_relevant)} memories the retriever may be missing")

    # Plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("\nWARNING: matplotlib not installed. Skipping plot.")
            return

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        fb_vals = [p[0] for p in pairs]
        gt_vals = [p[1] for p in pairs]
        ax.scatter(fb_vals, gt_vals, alpha=0.3, s=15)
        ax.set_xlabel("Mean Feedback Utility")
        ax.set_ylabel("Mean GT Relevance")
        ax.set_title(f"Utility Calibration (n={len(pairs)}, Spearman r={stats.get('spearman', 0):.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x")
        ax.axhline(args.gt_threshold, color="r", alpha=0.2, linestyle=":")
        ax.axvline(args.fb_threshold, color="r", alpha=0.2, linestyle=":")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.set_aspect("equal")

        plot_path = DATA_DIR / "tuning_studies" / "utility_calibration.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved: {plot_path}")


if __name__ == "__main__":
    main()
