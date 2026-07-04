#!/usr/bin/env python3
"""Store-state deltas across an iterated-sleep chain -- experiments/sleep-bench/iter_state.py

Given an ORDERED list of snapshot data-dirs (frozen, c1_nrem, c1_rem,
c1_consol, c2, ...), print count_store_state for each and the delta vs the
previous snapshot, and flag the PLATEAU point -- the first cycle boundary where
a full cycle stops changing the store (no new edges, no new/removed memories,
no new summaries, no archives). That is the H11 stop rule: iterate until the
per-cycle diff is ~0, or the pass cap.

Reuses precheck.count_store_state / diff_snapshots verbatim -- pure sqlite
reads, no LLM, safe under the abstinence hold.

Usage (dirs, in order):
    uv run experiments/sleep-bench/iter_state.py \
        ~/.somnigraph/benchmark/sleepbench-arms/conv0-frozen \
        ~/.somnigraph/benchmark/sleepbench-arms/conv0-c1 \
        ~/.somnigraph/benchmark/sleepbench-arms/conv0-c2

Or by label under a common arms dir:
    uv run experiments/sleep-bench/iter_state.py --arms-dir \
        ~/.somnigraph/benchmark/sleepbench-arms --conv 0 --labels frozen c1 c2 c3
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from precheck import count_store_state, diff_snapshots

# Counters whose net-zero delta across a full cycle defines "plateau".
PLATEAU_KEYS = ["active_memories", "edges_total", "active_summaries",
                "deleted_memories", "dormant_memories"]
# Extra context printed but not part of the plateau test.
CONTEXT_KEYS = ["total_memories", "topics_assigned", "edges_entity",
                "edges_derived_from"]


def resolve_entries(args) -> list:
    if args.arms_dir:
        base = Path(args.arms_dir).expanduser()
        return [(lab, base / f"conv{args.conv}-{lab}") for lab in args.labels]
    return [(Path(d).name, Path(d).expanduser()) for d in args.dirs]


def main() -> None:
    ap = argparse.ArgumentParser(description="H11 store-state / plateau tracker")
    ap.add_argument("dirs", nargs="*", help="ordered snapshot data-dirs")
    ap.add_argument("--arms-dir", default=None,
                    help="resolve --labels as <arms-dir>/conv<conv>-<label>")
    ap.add_argument("--conv", type=int, default=0)
    ap.add_argument("--labels", nargs="+", default=None)
    args = ap.parse_args()

    if args.arms_dir and not args.labels:
        ap.error("--arms-dir requires --labels")
    if not args.arms_dir and not args.dirs:
        ap.error("give snapshot dirs positionally, or --arms-dir + --labels")

    entries = resolve_entries(args)
    prev = None
    prev_label = None
    plateau_at = None
    for label, d in entries:
        db = d / "memory.db"
        if not db.exists():
            print(f"\n[{label}] MISSING {db} -- stopping (not yet generated).")
            break
        snap = count_store_state(db)
        print(f"\n[{label}]  ({d})")
        for k in PLATEAU_KEYS + CONTEXT_KEYS:
            print(f"    {k:<18} {snap[k]}")
        if prev is not None:
            delta = diff_snapshots(prev, snap)
            dstr = ", ".join(f"{k}{delta[k]:+d}" for k in PLATEAU_KEYS)
            print(f"    d({label} - {prev_label}): {dstr}")
            changed = any(delta.get(k, 0) != 0 for k in PLATEAU_KEYS)
            if not changed and plateau_at is None:
                plateau_at = label
                print(f"    >>> PLATEAU: '{label}' changed nothing vs '{prev_label}' "
                      f"-- stop rule met.")
        prev, prev_label = snap, label

    print()
    if plateau_at:
        print(f"[iter_state] plateau reached at '{plateau_at}': further cycles are inert.")
    else:
        print("[iter_state] no plateau across the provided snapshots -- keep iterating "
              "(or you have not reached the pass cap).")


if __name__ == "__main__":
    main()
