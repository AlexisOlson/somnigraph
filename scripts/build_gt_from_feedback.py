# /// script
# requires-python = ">=3.11"
# dependencies = ["sqlite-vec>=0.1.6"]
# ///
"""
Build ground-truth labels from retrieval feedback events.

Extracts (query, memory_id, utility) triples from the memory_events table
and writes a GT JSON compatible with tune_gt.py / train_reranker.py.

For each query, the mean utility across feedback events is used as the
relevance label. Queries with fewer than --min-memories rated memories
are dropped (too sparse to learn from).

Usage:
  uv run build_gt_from_feedback.py                    # Default output
  uv run build_gt_from_feedback.py --min-memories 3   # Require 3+ rated mems
  uv run build_gt_from_feedback.py --output path.json # Custom output path
"""

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from memory.constants import DATA_DIR
from memory.db import DB_PATH


def build_gt(db_path: Path, min_memories: int = 3) -> dict[str, dict[str, float]]:
    """Build GT dict from feedback events.

    Returns {query_text: {memory_id: mean_utility}}.
    """
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row

    # Collect all feedback events with query context
    rows = db.execute("""
        SELECT query, memory_id, context FROM memory_events
        WHERE event_type = 'feedback'
          AND context IS NOT NULL
          AND query IS NOT NULL
          AND query != ''
        ORDER BY created_at
    """).fetchall()

    # Aggregate: (query, memory_id) -> [utility scores]
    utilities: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        try:
            ctx = json.loads(r["context"])
            if "utility" not in ctx:
                continue
            utilities[r["query"]][r["memory_id"]].append(ctx["utility"])
        except (json.JSONDecodeError, TypeError):
            continue

    # Build GT: mean utility per (query, memory)
    gt = {}
    skipped = 0
    for query, mems in utilities.items():
        if len(mems) < min_memories:
            skipped += 1
            continue
        gt[query] = {
            mid: sum(scores) / len(scores)
            for mid, scores in mems.items()
        }

    db.close()

    print(f"Feedback events parsed: {sum(len(m) for m in utilities.values())}")
    print(f"Queries with feedback: {len(utilities)}")
    print(f"Queries after min_memories={min_memories} filter: {len(gt)}")
    print(f"Queries skipped (too sparse): {skipped}")

    # Distribution stats
    all_labels = [v for mems in gt.values() for v in mems.values()]
    if all_labels:
        import statistics
        print(f"\nLabel distribution:")
        print(f"  Count: {len(all_labels)}")
        print(f"  Mean:  {statistics.mean(all_labels):.3f}")
        print(f"  Stdev: {statistics.stdev(all_labels):.3f}")
        print(f"  Min:   {min(all_labels):.3f}")
        print(f"  Max:   {max(all_labels):.3f}")

        # Bin distribution
        bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
        labels_arr = sorted(all_labels)
        for i in range(len(bins) - 1):
            count = sum(1 for v in labels_arr if bins[i] <= v < bins[i+1])
            print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {count}")

    return gt


def main():
    parser = argparse.ArgumentParser(description="Build GT from feedback events")
    parser.add_argument("--min-memories", type=int, default=3,
                        help="Minimum rated memories per query (default: 3)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: DATA_DIR/tuning_studies/gt_feedback.json)")
    parser.add_argument("--db", type=str, default=str(DB_PATH),
                        help="Path to memory.db")
    args = parser.parse_args()

    output = Path(args.output) if args.output else DATA_DIR / "tuning_studies" / "gt_feedback.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    gt = build_gt(Path(args.db), min_memories=args.min_memories)

    with open(output, "w") as f:
        json.dump(gt, f, indent=2)
    print(f"\nGT written to {output}")
    print(f"Use with: uv run scripts/train_reranker.py --gt {output}")


if __name__ == "__main__":
    main()
