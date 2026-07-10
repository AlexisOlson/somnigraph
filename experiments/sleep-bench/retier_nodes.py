#!/usr/bin/env python3
"""One-off consolidation-node retier -- experiments/sleep-bench/retier_nodes.py

Applies the `94aad00` priority band (floor 6) to consolidation nodes generated
under the old max-member rule, per plan-reranker-2x2.md D1 pre-req. Refuses to
touch the live store via the standard isolation guard.

Usage:
    uv run experiments/sleep-bench/retier_nodes.py --data-dir <arm dir>
"""

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from precheck import assert_isolated_data_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Floor consolidation-node priority at 6")
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    assert_isolated_data_dir(data_dir)

    db = sqlite3.connect(str(data_dir / "memory.db"))
    try:
        cur = db.execute(
            "UPDATE memories SET base_priority = 6 "
            "WHERE source = 'consolidate' AND base_priority < 6"
        )
        db.commit()
        print(f"[retier] {cur.rowcount} node(s) retiered to p6 in {data_dir}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
