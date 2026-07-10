#!/usr/bin/env python3
"""Undormant counterfactual arm -- experiments/sleep-bench/gen_undormant_arm.py

Forks a slept arm and reactivates its dormant memories. This is a pure status
flip: `process_dormancy` never touches vec/FTS rows (verified on the
whole-corpus conv0-c1_rem 2026-07-10: memory_vec 6,271/6,271 and memory_fts
6,271/6,271 rows present with 5,737 dormant), so reactivation exactly restores
the dormanted details to retrieval alongside everything sleep added.

The point is to decompose REM's entangled effect (2026-07-10 finding: judged
c2-frozen recall is CI-clear negative and dormancy is the suspected eroder):

    [undormant - c1_nrem]  = pure summary ADDITION (the H7 additive regime, at scale)
    [c1_rem   - undormant] = the price of DORMANCY alone

LLM-free; safe anywhere. The new arm scores with retrieve.py like any other.

Usage:
    uv run experiments/sleep-bench/gen_undormant_arm.py \
        --src ~/.somnigraph/benchmark/sleepbench-full/conv0-c1_rem \
        --dst ~/.somnigraph/benchmark/sleepbench-full/conv0-c1_rem_undormant
"""

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fork import fork_store  # carries the live-store isolation guard


def main() -> None:
    ap = argparse.ArgumentParser(description="Fork an arm and reactivate its dormant memories")
    ap.add_argument("--src", required=True, help="slept arm to fork (e.g. conv0-c1_rem)")
    ap.add_argument("--dst", required=True, help="new arm dir (must not exist)")
    args = ap.parse_args()

    dst = fork_store(Path(args.src).expanduser(), Path(args.dst).expanduser())

    db = sqlite3.connect(str(dst / "memory.db"))
    try:
        cur = db.execute("UPDATE memories SET status = 'active' WHERE status = 'dormant'")
        db.commit()
        n = cur.rowcount
        counts = dict(db.execute("SELECT status, count(*) FROM memories GROUP BY status"))
    finally:
        db.close()

    print(f"[undormant] reactivated {n} dormant memories in {dst}")
    print(f"[undormant] status counts now: {counts}")


if __name__ == "__main__":
    main()
