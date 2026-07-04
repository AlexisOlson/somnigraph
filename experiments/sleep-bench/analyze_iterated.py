#!/usr/bin/env python3
"""Analyze the combined + iterated sleep cycle (H11) -- experiments/sleep-bench/analyze_iterated.py

Generalizes analyze_h3.py from the 2-arm {frozen, nrem} decomposition to an
ORDERED chain of N labelled snapshots produced by retrieve.py -- e.g.
    frozen  c1_nrem  c1_rem  c1_consol  c2  c3  c4
-- and reports, per scorer (rrf = content/index effect, formula = + graph/PPR):

  * Per-cell metric means (rows = metric, cols = label), one table per scorer.
  * BASELINE deltas   label - frozen          (total effect at each point).
  * ADJACENT deltas   label[k] - label[k-1]   (the per-stage / per-pass gain --
    the "does REM help ON TOP of edges" and "does pass 2 compound" contrasts).
  * GRAPH effect      formula - rrf per label  (edge contribution; should grow
    as NREM accumulates edges across passes).

Every delta is a paired bootstrap CI over the query_ids common to ALL included
cells (reusing the sleep-fork's verified paired_bootstrap_ci via analyze_h3 --
one source of truth for the stats).

Pure JSONL math -- touches no store, safe under the abstinence hold, reruns
instantly. Labels whose (label, scorer) JSONL is missing are dropped with a
warning, so you can analyze after pass 2 before passes 3-4 exist.

IMPORTANT -- the credit model (H7): any summary-bearing snapshot (c1_rem
onward) is UNDER-credited by these raw evidence-id metrics, because REM
summaries out-rank evidence turns without being evidence turns (H7's sign
flip). Read raw rrf/formula deltas on summary arms as a LOWER bound; the judged
bracket (impute_rescore.py + judge_summary_coverage.py) is the real credit.
This script reports the raw substrate; it does not re-credit.

Usage:
    uv run experiments/sleep-bench/analyze_iterated.py
    uv run experiments/sleep-bench/analyze_iterated.py \
        --labels frozen c1_nrem c1_rem c1_consol c2 --scorers rrf formula
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_h3 import (  # reuse verbatim -- one source of truth for the stats
    METRICS, fmt_ci, load_cell, mean, paired_bootstrap_ci, paired_deltas,
)

DEFAULT_RESULTS = HERE / "results"
DEFAULT_LABELS = ["frozen", "c1_nrem", "c1_rem", "c1_consol", "c2", "c3", "c4"]


def cell_exists(results_dir: Path, label: str, scorer: str) -> bool:
    return (results_dir / f"{label}__{scorer}.jsonl").exists()


def main() -> None:
    ap = argparse.ArgumentParser(description="H11 combined+iterated cycle analyzer")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--labels", nargs="+", default=DEFAULT_LABELS,
                    help="ordered snapshot labels, baseline first")
    ap.add_argument("--scorers", nargs="+", default=["rrf", "formula"])
    ap.add_argument("--baseline", default="frozen",
                    help="label used as the fixed baseline for the total-effect deltas")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)  # fixed seed = sleep-fork parity
    args = ap.parse_args()

    rdir = Path(args.results_dir).expanduser()

    # Keep only labels present for EVERY requested scorer (paired math needs both).
    labels = []
    for lab in args.labels:
        missing = [sc for sc in args.scorers if not cell_exists(rdir, lab, sc)]
        if missing:
            print(f"[skip] label '{lab}' missing scorer(s) {missing} -- dropped")
            continue
        labels.append(lab)
    if len(labels) < 2:
        print(f"ERROR: need >=2 labels present for all scorers. Have: {labels}")
        sys.exit(1)
    if args.baseline not in labels:
        print(f"ERROR: baseline '{args.baseline}' not among present labels {labels}")
        sys.exit(1)

    cells = {(lab, sc): load_cell(rdir, lab, sc)
             for lab in labels for sc in args.scorers}

    # Common query set across ALL cells -- paired deltas need every query everywhere.
    common = None
    for c in cells.values():
        common = set(c) if common is None else (common & set(c))
    common = sorted(common or [])
    if not common:
        print("ERROR: no query_id common to all cells.")
        sys.exit(1)
    print(f"[h11] {len(labels)} labels x {len(args.scorers)} scorers; "
          f"{len(common)} query_ids common to all cells; "
          f"n_boot={args.n_boot}, seed={args.seed}")
    print(f"[h11] chain: {' -> '.join(labels)}   baseline={args.baseline}\n")

    # ---- Per-cell means, one table per scorer ----
    for sc in args.scorers:
        print(f"=== Per-cell means [scorer={sc}] (rows=metric, cols=label) ===")
        hdr = f"{'metric':<18}" + "".join(f"{lab:>12}" for lab in labels)
        print(hdr)
        print("-" * len(hdr))
        for m in METRICS:
            cellvals = []
            for lab in labels:
                mv = mean(cells[(lab, sc)][q].get(m) for q in common)
                cellvals.append(f"{mv:>12.4f}" if mv is not None else f"{'--':>12}")
            print(f"{m:<18}" + "".join(cellvals))
        print()

    def report(title: str, cell_b: tuple, cell_a: tuple) -> None:
        print(f"--- {title} ---")
        for m in METRICS:
            deltas = paired_deltas(cells[cell_b], cells[cell_a], common, m)
            ci = paired_bootstrap_ci(deltas, args.n_boot, args.seed)
            print(f"    {m:<18} {fmt_ci(ci)}")
        print()

    # ---- BASELINE deltas: each non-baseline label - baseline, per scorer ----
    print(f"=== BASELINE deltas: label - {args.baseline}  "
          f"(total effect; * = CI excludes 0) ===\n")
    for sc in args.scorers:
        for lab in labels:
            if lab == args.baseline:
                continue
            report(f"[{sc}] {lab} - {args.baseline}", (lab, sc), (args.baseline, sc))

    # ---- ADJACENT deltas: label[k] - label[k-1], per scorer ----
    print("=== ADJACENT deltas: label[k] - label[k-1]  (per-stage / per-pass gain) ===\n")
    for sc in args.scorers:
        for prev, cur in zip(labels, labels[1:]):
            report(f"[{sc}] {cur} - {prev}", (cur, sc), (prev, sc))

    # ---- GRAPH effect: formula - rrf per label (needs both scorers) ----
    if "rrf" in args.scorers and "formula" in args.scorers:
        print("=== GRAPH effect: formula - rrf per label  (edge/PPR contribution) ===\n")
        for lab in labels:
            report(f"formula - rrf @ {lab}", (lab, "formula"), (lab, "rrf"))

    print("[h11] Reading guide: BASELINE-under-formula rising across c1 -> c2 -> ... "
          "with the GRAPH effect growing = edges compounding across passes. REM arms "
          "(c1_rem+) raw deltas are a LOWER bound (H7 credit model) -- confirm with the "
          "judged bracket. A DROP at c1_consol or a later cycle = consolidate's archival "
          "cost (H3) outrunning its canonicalization: a real finding, not a bug.")


if __name__ == "__main__":
    main()
