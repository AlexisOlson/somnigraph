"""Read the shadow-mode write instrumentation distribution.

impl_remember() logs one 'write_shadow' event per write attempt (measurement
only — it gates nothing). Each event carries the nearest same-category neighbor
similarity in the similarity_score column plus the top-3 neighbors in context.
This script buckets those similarities by 0.05 and splits by outcome so a future
session can set Write Guard thresholds from the observed production distribution
instead of a guessed cutoff (STEWARDSHIP P2 step 1).

Usage:
    uv run python scripts/write_shadow_histogram.py
    SOMNIGRAPH_DATA_DIR=/path/to/data uv run python scripts/write_shadow_histogram.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memory.db import get_db  # noqa: E402

BUCKET = 0.05
OUTCOMES = ["inserted", "superseded", "dedup_rejected"]


def main():
    db = get_db()
    rows = db.execute(
        "SELECT similarity_score, context FROM memory_events "
        "WHERE event_type = 'write_shadow' ORDER BY id"
    ).fetchall()
    db.close()

    if not rows:
        print("No write_shadow events yet. The distribution accumulates as "
              "remember() is called in production.")
        return

    # bucket_index -> outcome -> count. Bucket by nearest-neighbor similarity;
    # events with no same-category neighbor (similarity_score IS NULL) counted
    # separately — they are structurally un-dedupable (first-of-category writes).
    hist = defaultdict(lambda: defaultdict(int))
    no_neighbor = defaultdict(int)
    totals = defaultdict(int)

    for r in rows:
        ctx = json.loads(r["context"]) if r["context"] else {}
        outcome = ctx.get("outcome", "unknown")
        totals[outcome] += 1
        sim = r["similarity_score"]
        if sim is None:
            no_neighbor[outcome] += 1
            continue
        # Clamp into [0, 1] for bucketing; cosine sim can dip slightly negative.
        sim = max(0.0, min(1.0, sim))
        idx = min(int(sim / BUCKET), int(1.0 / BUCKET) - 1)
        hist[idx][outcome] += 1

    n = len(rows)
    print(f"write_shadow events: {n}")
    print("outcomes:", ", ".join(f"{o}={totals.get(o, 0)}" for o in OUTCOMES) +
          (f", other={sum(v for k, v in totals.items() if k not in OUTCOMES)}"
           if any(k not in OUTCOMES for k in totals) else ""))
    print()

    header = f"{'sim range':>13} | " + " | ".join(f"{o:>14}" for o in OUTCOMES) + " |   total"
    print(header)
    print("-" * len(header))
    max_idx = int(1.0 / BUCKET)
    for idx in range(max_idx):
        lo, hi = idx * BUCKET, (idx + 1) * BUCKET
        counts = [hist[idx].get(o, 0) for o in OUTCOMES]
        row_total = sum(counts)
        if row_total == 0:
            continue
        cells = " | ".join(f"{c:>14}" for c in counts)
        print(f"{lo:>5.2f}-{hi:<5.2f}  | {cells} | {row_total:>7}")

    nn_total = sum(no_neighbor.values())
    if nn_total:
        cells = " | ".join(f"{no_neighbor.get(o, 0):>14}" for o in OUTCOMES)
        print(f"{'no neighbor':>13} | {cells} | {nn_total:>7}")


if __name__ == "__main__":
    main()
