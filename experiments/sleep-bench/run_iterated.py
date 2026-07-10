#!/usr/bin/env python3
"""Run the combined + iterated sleep cycle (H11) -- experiments/sleep-bench/run_iterated.py

*** RUN IN TERMINAL *** -- this drives the real LLM sleep stages (Sonnet NREM +
opus REM + opus consolidate) in sequence, once per pass. Minutes per stage; not
a background job.

Forks a frozen baseline, then loops the FULL cycle
    sleep_nrem.py deep  ->  sleep_rem.py  ->  sleep_consolidate.py -v
on ONE evolving store, forking a byte-identical SNAPSHOT after each stage of
pass 1 (c1_nrem, c1_rem, c1 == c1_consol) and at each cycle boundary thereafter
(c2, c3, ...). Stops when a full cycle leaves the store unchanged (plateau) or
--max-passes is hit. The snapshots are the arms retrieve.py then scores.

Isolation: every fork/reset goes through precheck's guard -- it can never touch
the live store (hard no-sleep until ~2026-07-15). SUMMARY_MIN_MEMBERS is pinned
(default 5, the shipped floor) so REM is NOT inert -- this REQUIRES the branch's
member-count floor; on main, REM makes 0 LoCoMo summaries and the "combined"
arm silently degenerates to NREM-only. It is set in os.environ, which
build_subprocess_env copies into every sleep subprocess.

Usage:
    uv run experiments/sleep-bench/run_iterated.py \
        --frozen ~/.somnigraph/benchmark/sleepbench-arms/conv0-frozen \
        --arms-dir ~/.somnigraph/benchmark/sleepbench-arms --conv 0 \
        --max-passes 4 [--min-members 5] [--force]

Then score (LLM-free, safe to run in-session):
    for each arm: retrieve.py --data-dir <arm> --label <label> --conv 0 --scorers rrf formula --fresh
    analyze_iterated.py ;  iter_state.py --arms-dir <arms> --conv 0 --labels frozen c1 c2 ...
    judged bracket on c1_rem+ arms: judge_summary_coverage.py + impute_rescore.py
"""

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from precheck import (assert_isolated_data_dir, count_store_state,
                      diff_snapshots, run_sleep_script)
from fork import fork_store, reset_store

# Same plateau test as iter_state.py -- a cycle that moves none of these is inert.
PLATEAU_KEYS = ["active_memories", "edges_total", "active_summaries",
                "deleted_memories", "dormant_memories"]


def snapshot(work: Path, arms: Path, conv: int, label: str) -> Path:
    """Fork the evolving work store into a frozen arm snapshot conv<c>-<label>."""
    dst = arms / f"conv{conv}-{label}"
    if dst.exists():
        reset_store(dst)          # asserts isolation internally
    fork_store(work, dst)         # asserts isolation internally
    return dst


def state(work: Path) -> dict:
    return count_store_state(work / "memory.db")


def fmt_state(snap: dict, delta: dict | None = None) -> str:
    if delta is None:
        return ", ".join(f"{k}={snap[k]}" for k in PLATEAU_KEYS)
    return ", ".join(f"{k}={snap[k]} (d{delta[k]:+d})" for k in PLATEAU_KEYS)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="H11 iterated full-cycle runner (TERMINAL / LLM)")
    ap.add_argument("--frozen", required=True, help="frozen baseline store to fork from")
    ap.add_argument("--arms-dir", required=True, help="dir where snapshot arms are written")
    ap.add_argument("--conv", type=int, default=0)
    ap.add_argument("--work-label", default="iter",
                    help="name of the evolving working store: conv<c>-<work-label>")
    ap.add_argument("--max-passes", type=int, default=4)
    ap.add_argument("--min-members", type=int, default=5,
                    help="SOMNIGRAPH_SUMMARY_MIN_MEMBERS (shipped floor = 5)")
    ap.add_argument("--nrem-mode", default="deep", choices=["standard", "deep"])
    ap.add_argument("--force", action="store_true",
                    help="reset the working store if it already exists")
    args = ap.parse_args()

    frozen = Path(args.frozen).expanduser().resolve()
    arms = Path(args.arms_dir).expanduser().resolve()
    assert_isolated_data_dir(frozen)
    if not (frozen / "memory.db").exists():
        sys.exit(f"[run_iterated] frozen store has no memory.db: {frozen}")

    work = arms / f"conv{args.conv}-{args.work_label}"
    assert_isolated_data_dir(work)
    if work.exists():
        if not args.force:
            sys.exit(f"[run_iterated] working store exists: {work} (use --force to reset)")
        reset_store(work)
    fork_store(frozen, work)

    # Pin the summary floor BEFORE any sleep subprocess -- build_subprocess_env
    # does os.environ.copy(), so this propagates. Without it REM is inert.
    os.environ["SOMNIGRAPH_SUMMARY_MIN_MEMBERS"] = str(args.min_members)
    print(f"\n[run_iterated] SUMMARY_MIN_MEMBERS={args.min_members}; "
          f"max_passes={args.max_passes}; nrem={args.nrem_mode}")

    prev_cycle = state(work)  # == frozen state
    print(f"[state] frozen: {fmt_state(prev_cycle)}")
    created: list[Path] = []

    for p in range(1, args.max_passes + 1):
        print(f"\n========== PASS {p} ==========")
        run_sleep_script("sleep_nrem.py", [args.nrem_mode], work)
        if p == 1:
            created.append(snapshot(work, arms, args.conv, "c1_nrem"))
        run_sleep_script("sleep_rem.py", [], work)
        if p == 1:
            created.append(snapshot(work, arms, args.conv, "c1_rem"))
        run_sleep_script("sleep_consolidate.py", ["-v"], work)

        cyc = state(work)
        delta = diff_snapshots(prev_cycle, cyc)
        created.append(snapshot(work, arms, args.conv, f"c{p}"))  # cycle boundary
        if p == 1:
            # alias so the plan's c1_consol naming also resolves for analysis
            created.append(snapshot(work, arms, args.conv, "c1_consol"))
        print(f"[state] after cycle {p}: {fmt_state(cyc, delta)}")
        if all(delta.get(k, 0) == 0 for k in PLATEAU_KEYS):
            print(f"\n>>> PLATEAU after pass {p}: full cycle changed nothing. Stopping.")
            break
        prev_cycle = cyc

    labels = sorted({"frozen"} | {d.name.replace(f"conv{args.conv}-", "")
                                  for d in created})
    print("\n[run_iterated] arms created:")
    for d in created:
        print(f"    {d.name}")
    print("\n[run_iterated] next (LLM-free, safe in-session):")
    print(f"    score each arm: retrieve.py --data-dir <arm> --label <label> "
          f"--conv {args.conv} --scorers rrf formula --fresh")
    print(f"    then:           analyze_iterated.py --labels {' '.join(labels)}")
    print(f"    plateau check:  iter_state.py --arms-dir {arms} --conv {args.conv} "
          f"--labels {' '.join(labels)}")
    print("    judged bracket on c1_rem+ arms: judge_summary_coverage.py + impute_rescore.py")


if __name__ == "__main__":
    main()
