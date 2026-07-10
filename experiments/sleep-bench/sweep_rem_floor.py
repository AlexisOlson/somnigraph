#!/usr/bin/env python3
"""H7 REM floor sweep -- experiments/sleep-bench/sweep_rem_floor.py

The CHEAP, rerunnable half of the H7 sweep. Given the REM arm produced by
gen_rem_arm.py (all leaf summaries generated at the lowest floor), walk the
member-count ladder and, at each rung, activate exactly the leaf summaries whose
cluster has >= threshold members, then score retrieval with the real recall path
(retrieve.py, rrf scorer). No LLM, no re-embedding -- just a status toggle over
already-generated, already-embedded nodes + a local rescore.

WHY rrf ONLY: this is REM-on-frozen with NO NREM edges, so PPR has nothing to
traverse and formula would be bit-identical to rrf (the H3 frozen check). rrf
also declares wants_expansion=False, so retrieve.py's edge-embedding health
guard doesn't apply. Summaries are new content/index nodes -- exactly the rrf
(content) channel. (derived_from edges from generation are ignored by rrf.)

MONOTONICITY: lowering the floor only ADDS summaries; a summary's content is
identical across rungs. So the only thing that varies rung-to-rung is which
summaries are active -- the curve is clean by construction.

CONTROL: the "none" rung (threshold = infinity) activates zero summaries, so the
store == frozen for retrieval. Every rung's delta is measured against it
(analyze_rem_floor.py); it should also match the true frozen__rrf.jsonl from H3.

THE ARTIFACT WE EXPECT (raw-first, per the session decision): summaries are NOT
evidence turns, so retrieve.py's gt_scores only credit evidence_ids. A summary
that out-ranks its own evidence turn (pushing it below rank 10) scores a MISS on
recall@10 even though the retrieval is arguably better -- the same uncreditable-
survivor class that made NREM merge look recall-negative. analyze_rem_floor.py
surfaces this; crediting it is the deferred coverage judge (retrieve.py's
documented seam), not built here.

RUN IN TERMINAL alongside gen (it drives one retrieve.py subprocess per rung;
the whole sweep can exceed 5 min). Touches only the isolated REM arm.

Usage:
    uv run experiments/sleep-bench/sweep_rem_floor.py \
        --rem-dir ~/.somnigraph/benchmark/sleepbench-arms/conv0-rem --conv 0
"""

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from precheck import assert_isolated_data_dir

NONE = 10 ** 9  # sentinel threshold for the zero-summary control rung
DEFAULT_LADDER = [NONE, 25, 15, 10, 5, 3]
RESULTS_DIR = HERE / "results"


def label_for(threshold: int) -> str:
    return "rem_none" if threshold >= NONE else f"rem_m{threshold}"


def leaf_summary_counts(db_path: Path) -> dict[str, int]:
    """{summary_id: member_count} for LEAF summaries only.

    A summary is a leaf iff every id in its generated_from is a detail memory
    (not itself a summary). member_count = len(generated_from) -- the exact
    cluster member list preserved at generation (sleep_rem source_ids).
    """
    conn = sqlite3.connect(str(db_path))
    try:
        layer = {r[0]: r[1] for r in conn.execute("SELECT id, layer FROM memories").fetchall()}
        out: dict[str, int] = {}
        for sid, gen_from in conn.execute(
            "SELECT id, generated_from FROM memories WHERE layer='summary'"
        ).fetchall():
            if not gen_from:
                continue
            members = json.loads(gen_from)
            if not members:
                continue
            if any(layer.get(m) == "summary" for m in members):
                continue  # parent summary -- kept OFF for the whole leaf sweep
            out[sid] = len(members)
        return out
    finally:
        conn.close()


def set_active_summaries(db_path: Path, active_ids: set[str]) -> None:
    """Flip every summary to inactive, then activate exactly `active_ids`.

    Only toggles between active<->inactive (never touches 'deleted'), so it is
    fully reversible and idempotent across rungs. Detail memories are untouched.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "UPDATE memories SET status='inactive' "
            "WHERE layer='summary' AND status IN ('active','inactive')"
        )
        if active_ids:
            ids = list(active_ids)
            ph = ",".join("?" * len(ids))
            conn.execute(
                f"UPDATE memories SET status='active' WHERE id IN ({ph})", ids
            )
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="H7 REM floor sweep (toggle + rescore)")
    ap.add_argument("--rem-dir", required=True, help="REM arm from gen_rem_arm.py")
    ap.add_argument("--conv", type=int, required=True, help="LoCoMo conversation index")
    ap.add_argument("--thresholds", type=int, nargs="+", default=None,
                    help=f"member-count ladder (default {DEFAULT_LADDER[1:]} + none control)")
    ap.add_argument("--recall-limit", type=int, default=20)
    ap.add_argument("--recall-budget", type=int, default=5000)
    args = ap.parse_args()

    rem = Path(args.rem_dir).expanduser().resolve()
    assert_isolated_data_dir(rem)
    db_path = rem / "memory.db"
    if not db_path.exists():
        raise SystemExit(f"no memory.db at {rem} -- run gen_rem_arm.py first")

    thresholds = args.thresholds if args.thresholds else DEFAULT_LADDER
    # Ensure the none-control is present and thresholds run high -> low.
    thresholds = sorted(set(thresholds + [NONE]), reverse=True)

    leaf = leaf_summary_counts(db_path)
    if not leaf:
        raise SystemExit(
            "[sweep] no leaf summaries in this store -- did gen_rem_arm.py run? "
            "(REM is inert if the member floor was never lowered.)"
        )
    print(f"[sweep] {len(leaf)} leaf summaries; member-counts: "
          f"{sorted(leaf.values(), reverse=True)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for t in thresholds:
        active = {sid for sid, cnt in leaf.items() if cnt >= t}
        label = label_for(t)
        set_active_summaries(db_path, active)
        print(f"\n[sweep] === floor >= {('none' if t >= NONE else t)}: "
              f"{len(active)} summaries active  (label={label}) ===")
        cmd = [
            sys.executable, str(HERE / "retrieve.py"),
            "--data-dir", str(rem), "--label", label, "--conv", str(args.conv),
            "--scorers", "rrf",
            "--recall-limit", str(args.recall_limit),
            "--recall-budget", str(args.recall_budget),
            "--fresh",
        ]
        subprocess.run(cmd, check=True)
        manifest.append({"threshold": (None if t >= NONE else t),
                         "label": label, "n_summaries_active": len(active)})

    # Leave the store in the fully-generated state (all leaf summaries active).
    set_active_summaries(db_path, set(leaf))

    manifest_path = RESULTS_DIR / "rem_floor_sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n[sweep] DONE. Manifest -> {manifest_path}")
    print(f"[sweep] Restored all {len(leaf)} leaf summaries to active in {rem}.")
    print("[sweep] Next: uv run experiments/sleep-bench/analyze_rem_floor.py")


if __name__ == "__main__":
    main()
