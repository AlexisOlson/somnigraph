#!/usr/bin/env python3
"""Analyze the H7 REM floor sweep -- experiments/sleep-bench/analyze_rem_floor.py

Reads the per-rung Stage-A JSONL that sweep_rem_floor.py produced (rem_none +
rem_m{t}, rrf scorer) and reports the summary-count -> retrieval curve:

  - Per-rung metric means, ordered along the member-count ladder.
  - Paired delta vs the CONTROL rung (rem_none = zero summaries == frozen),
    bootstrapped CI over query_id (same stats as analyze_h3 / the sleep-fork).
  - n_summaries_active per rung (from the sweep manifest) beside each row, so
    the curve reads as "K summaries -> what retrieval change."

Pure JSONL math -- touches no store, safe under the abstinence hold, reruns
instantly. Reuses analyze_h3's cell/delta helpers and the sleep-fork's
paired_bootstrap_ci verbatim.

INSTRUMENT CHECK: rem_none must be ~0 vs the true frozen__rrf.jsonl (if present
from H3) on every metric -- zero summaries active == frozen. Reported first; a
non-zero there means the toggle leaked and the rest is suspect.

READING THE CURVE (raw-first caveat): recall metrics here CREDIT ONLY evidence
turns (retrieve.py gt_scores). A summary that out-ranks its evidence turn scores
as a recall LOSS even when it's a better answer -- the uncreditable-survivor
artifact (cf. H3 merge). So a negative recall delta is NOT yet "summaries hurt";
it is the signal to run the coverage decomposition / judge. NDCG/MRR moving with
no recall gain = summaries displacing evidence without covering it.

Usage:
    uv run experiments/sleep-bench/analyze_rem_floor.py
    uv run experiments/sleep-bench/analyze_rem_floor.py --frozen-label frozen
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sleep-fork"))

from analyze_h3 import load_cell, paired_deltas, mean, fmt_ci  # reuse, don't duplicate
from analyze_sleep_fork import paired_bootstrap_ci

METRICS = [
    "ndcg_at_budget", "r10", "mrr",
    "recall_any_10", "recall_all_10", "recall_any_effk",
]


def load_manifest(results_dir: Path) -> dict[str, int]:
    """label -> n_summaries_active, from sweep_rem_floor.py's manifest."""
    path = results_dir / "rem_floor_sweep_manifest.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    return {r["label"]: r["n_summaries_active"] for r in rows}


def discover_rungs(results_dir: Path) -> list[str]:
    """Ordered rung labels present as rrf JSONL: rem_none first, then m-desc."""
    labels = []
    for p in results_dir.glob("rem_*__rrf.jsonl"):
        labels.append(p.name[: -len("__rrf.jsonl")])
    def key(lbl: str):
        if lbl == "rem_none":
            return (0, 0)  # control first
        try:
            return (1, -int(lbl.split("_m")[1]))  # m25, m15, ... m3  (high -> low)
        except (IndexError, ValueError):
            return (2, 0)
    return sorted(set(labels), key=key)


def main() -> None:
    ap = argparse.ArgumentParser(description="H7 REM floor-sweep analyzer")
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    ap.add_argument("--frozen-label", default="frozen",
                    help="H3 frozen rrf JSONL, for the rem_none==frozen instrument check")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rdir = Path(args.results_dir).expanduser()
    rungs = discover_rungs(rdir)
    if "rem_none" not in rungs:
        raise SystemExit(
            "no rem_none control rung found -- run sweep_rem_floor.py first "
            f"(looked in {rdir})."
        )
    manifest = load_manifest(rdir)

    cells = {lbl: load_cell(rdir, lbl, "rrf") for lbl in rungs}
    control = cells["rem_none"]

    # Common query set across all rungs (paired deltas need every query present).
    common = set(control)
    for c in cells.values():
        common &= set(c)
    common = sorted(common)
    print(f"[h7] rungs: {rungs}")
    print(f"[h7] {len(common)} query_ids common to all rungs, "
          f"n_boot={args.n_boot}, seed={args.seed}\n")

    # --- Instrument check: rem_none vs true frozen (H3) should be ~0 ---
    try:
        frozen = load_cell(rdir, args.frozen_label, "rrf")
        fcommon = sorted(set(frozen) & set(control))
        print("=== INSTRUMENT CHECK: rem_none - frozen (expect ~0: zero summaries == frozen) ===")
        for m in METRICS:
            deltas = paired_deltas(control, frozen, fcommon, m)
            print(f"  {m:<18} {fmt_ci(paired_bootstrap_ci(deltas, args.n_boot, args.seed))}")
        print()
    except FileNotFoundError:
        print(f"[h7] (no {args.frozen_label}__rrf.jsonl -- skipping rem_none==frozen check)\n")

    # --- Per-rung means, along the ladder ---
    print("=== Per-rung means (rrf) ===")
    hdr = f"{'rung':<10} {'n_summ':>7} " + " ".join(f"{m[:11]:>11}" for m in METRICS)
    print(hdr)
    print("-" * len(hdr))
    for lbl in rungs:
        n = manifest.get(lbl, "?")
        vals = " ".join(
            f"{mean(cells[lbl][q].get(m) for q in common):>11.4f}" for m in METRICS
        )
        print(f"{lbl:<10} {str(n):>7} {vals}")

    # --- Paired delta vs the rem_none control ---
    print("\n=== Delta vs rem_none control (rem_m{t} - rem_none), paired bootstrap ===")
    print("  (* = 95% CI excludes 0; negative recall may be the uncreditable-summary artifact)")
    for lbl in rungs:
        if lbl == "rem_none":
            continue
        n = manifest.get(lbl, "?")
        print(f"\n  --- {lbl}  ({n} summaries active) ---")
        for m in METRICS:
            deltas = paired_deltas(cells[lbl], control, common, m)
            print(f"    {m:<18} {fmt_ci(paired_bootstrap_ci(deltas, args.n_boot, args.seed))}")

    print("\n[h7] Verdict logic: a rung is a WIN if recall_any/r@10 rises CI-clear "
          "with no NDCG/MRR collapse. A recall DROP with summaries present is the "
          "signal to run the coverage decomposition (summary out-ranking its "
          "evidence turn), not proof that summaries hurt.")


if __name__ == "__main__":
    main()
