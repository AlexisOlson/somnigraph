"""
Store-agnostic fork/reset primitive — experiments/sleep-bench/fork.py (§3.3).

Replaces the sleep-fork's hand-rolled `cp memory.db` + hardcoded-snapshot-path
dance (which nearly broke on `snapshot_production_db`'s hardcoded live path)
with a small primitive: shutil.copytree the whole isolated SOMNIGRAPH_DATA_DIR
(memory.db + WAL/SHM sidecars + tuning_studies/ + taxonomy.json + any personal
data JSON) so every stage-arm in Phase 1+ is a byte-identical fork of a frozen
baseline, reproducible by construction.

Reuses precheck.py's isolation guard for both endpoints of every fork/reset --
a fork or reset can never target the live store.

Usage:
    uv run experiments/sleep-bench/fork.py fork <src_dir> <dst_dir>
    uv run experiments/sleep-bench/fork.py reset <data_dir>

Deferred: this module is intentionally minimal for Phase 0 (precheck doesn't
need forking -- it runs the pipeline once on a fresh ingest). Phase 1's
per-stage arms (frozen vs NREM-only vs NREM+REM vs full cycle, each its own
fork) are what actually exercise this.
"""

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from precheck import assert_isolated_data_dir  # reuse the guard, don't duplicate it


def scrub_access_counts(data_dir: Path) -> None:
    """Reset retrieval-access state so sleep sees an honest never-recalled store.

    Scoring passes call the recall path against arm stores, which bumps
    recall_count and last_accessed as a side effect. REM's dormancy criteria
    read exactly those columns (recall_count + reflect_count < threshold, plus
    last_accessed age), so a previously-scored fork looks "recently used" and
    dormancy silently under-fires (H11's 7% vs the honest 97.5% on a fresh
    whole-corpus store, 2026-07-10). last_accessed resets to created_at — the
    same convention REM applies when last_accessed is missing.
    """
    assert_isolated_data_dir(data_dir)
    db_path = data_dir / "memory.db"
    if not db_path.exists():
        raise FileNotFoundError(f"no memory.db under {data_dir}")
    db = sqlite3.connect(db_path)
    try:
        cur = db.execute(
            "UPDATE memories SET recall_count = 0, reflect_count = 0, "
            "startup_count = 0, access_count = 0, use_count = 0, "
            "last_accessed = created_at"
        )
        db.commit()
        print(f"[scrub] reset access counters on {cur.rowcount} rows in {db_path}")
    finally:
        db.close()


def fork_store(src_dir: Path, dst_dir: Path, scrub: bool = False) -> Path:
    """copytree an isolated data dir into a fresh one. Returns dst_dir."""
    src_dir = src_dir.resolve()
    dst_dir = dst_dir.resolve()
    assert_isolated_data_dir(src_dir)
    assert_isolated_data_dir(dst_dir)
    if not src_dir.exists():
        raise FileNotFoundError(f"fork source does not exist: {src_dir}")
    if dst_dir.exists():
        raise FileExistsError(
            f"fork destination already exists: {dst_dir} "
            "(remove it first, or pick a new name)"
        )
    shutil.copytree(src_dir, dst_dir)
    print(f"[fork] {src_dir} -> {dst_dir}")
    if scrub:
        scrub_access_counts(dst_dir)
    return dst_dir


def reset_store(data_dir: Path) -> None:
    """rmtree an isolated data dir. Caller is responsible for re-ingesting."""
    data_dir = data_dir.resolve()
    assert_isolated_data_dir(data_dir)
    if data_dir.exists():
        shutil.rmtree(data_dir)
        print(f"[reset] removed {data_dir}")
    else:
        print(f"[reset] nothing to remove at {data_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fork or reset an isolated sleep-bench store"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fork = sub.add_parser("fork", help="copytree src -> dst (both must be isolated)")
    p_fork.add_argument("src", type=str)
    p_fork.add_argument("dst", type=str)
    p_fork.add_argument(
        "--scrub-access-counts", action="store_true",
        help="after copying, zero recall/reflect/access counters and reset "
             "last_accessed to created_at on the DESTINATION, so a previously "
             "scored source doesn't shield rows from REM dormancy",
    )

    p_reset = sub.add_parser("reset", help="rmtree an isolated store dir")
    p_reset.add_argument("data_dir", type=str)

    args = parser.parse_args()
    if args.cmd == "fork":
        fork_store(
            Path(args.src).expanduser(), Path(args.dst).expanduser(),
            scrub=args.scrub_access_counts,
        )
    elif args.cmd == "reset":
        reset_store(Path(args.data_dir).expanduser())


if __name__ == "__main__":
    main()
