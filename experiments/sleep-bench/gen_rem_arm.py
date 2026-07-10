#!/usr/bin/env python3
"""Generate the H7 REM arm (leaf summaries only) -- experiments/sleep-bench/gen_rem_arm.py

The EXPENSIVE, run-once half of the H7 REM floor sweep (proposal
docs/proposals/sleep-pipeline-optimization.md §4 Stage C; findings-phase0.md
Finding 1 -- REM is inert on LoCoMo because the old absolute token floor blocks
every summary).

WHAT THIS DOES:
  1. fork.py the frozen arm into a fresh REM arm (byte-identical copy).
  2. Run REM *leaf summary generation only* on the fork, at the most-open
     member floor on the sweep ladder (default 3). This is monotone: a summary
     that exists at floor=3 is byte-identical to the one that would exist at
     floor=10 -- the floor decides WHETHER to summarize, never HOW -- so one
     generation pass at the lowest floor produces the superset the sweep then
     subsets by member count (sweep_rem_floor.py). Cuts the LLM cost from Nx
     (one pass per rung) to 1x and removes cross-rung LLM nondeterminism.

WHY LEAF-ONLY: we call detect_clusters -> score_clusters -> generate_summaries
directly rather than sleep_rem.main(), which also runs parent-summary
generation (a second, summary-of-summaries granularity variable -- the deferred
A/B) and maintenance (stub-deletion / dormancy -- noise orthogonal to the
floor). Isolating the leaf step means every generated summary's generated_from
is a pure detail-member list, so len(generated_from) == the true cluster member
count the sweep keys on.

The floor itself is set via SOMNIGRAPH_SUMMARY_MIN_MEMBERS, read by sleep_rem at
import (constant SUMMARY_MIN_MEMBERS) -- so this script sets the env BEFORE
importing sleep_rem.

RUN IN TERMINAL: this is the LLM pass (~one `claude -p --model opus` call per
qualifying cluster). Over 5 minutes; not a background job.

Usage:
    uv run experiments/sleep-bench/gen_rem_arm.py \
        --frozen ~/.somnigraph/benchmark/sleepbench-arms/conv0-frozen \
        --rem    ~/.somnigraph/benchmark/sleepbench-arms/conv0-rem \
        --min-members 3 [--force]
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(HERE))

from precheck import assert_isolated_data_dir  # isolation guard -- never touch live
from fork import fork_store, reset_store
import store as store_mod

# The sweep ladder, high -> low. gen at the lowest rung; report coverage of each.
LADDER = [25, 15, 10, 5, 3]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the H7 REM (leaf-only) arm")
    ap.add_argument("--frozen", required=True, help="frozen arm (fork source)")
    ap.add_argument("--rem", required=True, help="REM arm to create (fork dest)")
    ap.add_argument("--min-members", type=int, default=3,
                    help="member floor for generation = lowest sweep rung (default 3)")
    ap.add_argument("--force", action="store_true",
                    help="reset --rem if it already exists")
    args = ap.parse_args()

    frozen = Path(args.frozen).expanduser().resolve()
    rem = Path(args.rem).expanduser().resolve()
    assert_isolated_data_dir(frozen)
    assert_isolated_data_dir(rem)
    if not (frozen / "memory.db").exists():
        raise SystemExit(f"no memory.db at frozen source: {frozen}")

    if rem.exists():
        if args.force:
            reset_store(rem)
        else:
            raise SystemExit(f"{rem} already exists -- pass --force to replace it")
    fork_store(frozen, rem)

    # Env MUST be set before importing sleep_rem: SUMMARY_MIN_MEMBERS is read at
    # module import, and DATA_DIR/get_db resolve from SOMNIGRAPH_DATA_DIR.
    os.environ["SOMNIGRAPH_DATA_DIR"] = str(rem)
    os.environ["SOMNIGRAPH_SUMMARY_MIN_MEMBERS"] = str(args.min_members)
    key = store_mod.resolve_openai_key()
    if not key:
        raise SystemExit(
            "no OpenAI key found -- summary vectors need embeddings. Set "
            "OPENAI_API_KEY or place openai_api_key in the store/.claude/data dir."
        )
    os.environ["OPENAI_API_KEY"] = key

    import sleep_rem
    if sleep_rem.SUMMARY_MIN_MEMBERS != args.min_members:
        raise SystemExit(
            f"floor mis-set: sleep_rem.SUMMARY_MIN_MEMBERS="
            f"{sleep_rem.SUMMARY_MIN_MEMBERS} != --min-members {args.min_members}. "
            "The env must be set before importing sleep_rem."
        )

    print(f"[gen] REM arm: {rem}")
    print(f"[gen] member floor (SUMMARY_MIN_MEMBERS) = {sleep_rem.SUMMARY_MIN_MEMBERS} "
          f"(= {sleep_rem.SUMMARY_MIN_MEMBERS / sleep_rem.RECALL_LIMIT_REF:.2g}x recall limit)")

    db = sleep_rem.get_db()

    print("[gen] Step 1: detect clusters")
    clusters = sleep_rem.detect_clusters(db)
    sizes = sorted((len(c["memory_ids"]) for c in clusters), reverse=True)
    print(f"[gen]   {len(clusters)} clusters; member-counts: {sizes}")
    for t in LADDER:
        print(f"[gen]     clusters >= {t} members: {sum(1 for s in sizes if s >= t)}")

    if not clusters:
        raise SystemExit("[gen] no clusters detected -- nothing to summarize")

    print(f"[gen] Step 2: score clusters (member floor = {sleep_rem.SUMMARY_MIN_MEMBERS})")
    work = sleep_rem.score_clusters(db, clusters)
    print(f"[gen]   {len(work)} clusters clear the floor -> candidates for summary")

    print("[gen] Step 3: generate LEAF summaries (LLM; parent + maintenance SKIPPED)")
    gen, refr = sleep_rem.generate_summaries(db, work)
    print(f"[gen]   generated={gen} refreshed={refr}")
    db.close()

    # Report the member-count distribution of what actually landed -- this is
    # the coverage the sweep will subset. A rung with 0 summaries is a dead rung.
    conn = sqlite3.connect(str(rem / "memory.db"))
    try:
        layer = {r[0]: r[1] for r in conn.execute("SELECT id, layer FROM memories").fetchall()}
        counts = []
        for gen_from, in conn.execute(
            "SELECT generated_from FROM memories "
            "WHERE layer='summary' AND status='active'"
        ).fetchall():
            if not gen_from:
                continue
            members = json.loads(gen_from)
            if members and not any(layer.get(m) == "summary" for m in members):
                counts.append(len(members))
    finally:
        conn.close()
    counts.sort(reverse=True)

    print(f"\n[gen] {len(counts)} active LEAF summaries; member-counts: {counts}")
    print("[gen] sweep-rung coverage (summaries active at each floor):")
    for t in LADDER:
        print(f"[gen]     floor >= {t}: {sum(1 for c in counts if c >= t)} summaries")
    print(f"[gen]     floor = none (control): 0 summaries  (== frozen)")
    print(f"\n[gen] DONE. Next: uv run experiments/sleep-bench/sweep_rem_floor.py "
          f"--rem-dir {rem} --conv <N>")


if __name__ == "__main__":
    main()
