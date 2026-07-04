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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from precheck import assert_isolated_data_dir  # reuse the guard, don't duplicate it


def fork_store(src_dir: Path, dst_dir: Path) -> Path:
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

    p_reset = sub.add_parser("reset", help="rmtree an isolated store dir")
    p_reset.add_argument("data_dir", type=str)

    args = parser.parse_args()
    if args.cmd == "fork":
        fork_store(Path(args.src).expanduser(), Path(args.dst).expanduser())
    elif args.cmd == "reset":
        reset_store(Path(args.data_dir).expanduser())


if __name__ == "__main__":
    main()
