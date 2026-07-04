#!/usr/bin/env python3
"""Analyze Step 1 (H3 — NREM/graph) — experiments/sleep-bench/analyze_h3.py

Reads the four Stage-A per-query JSONL files produced by retrieve.py for the
2x1 grid {frozen, nrem} x {rrf, formula} and reports the decomposition H3 tests:

  - Per-cell metric means (the 2x2 of arm x scorer).
  - GRAPH contribution, per arm:  Delta(formula - rrf), paired over query_id.
    On frozen (0 edges) this must be ~0 (PPR has nothing to traverse); on nrem
    it is the edge contribution PPR reads from memory_edges.
  - SLEEP contribution, per scorer:  Delta(nrem - frozen), paired over query_id.
    Under rrf (graph-blind) this moves only via NREM's dup-merge changing the
    index; under formula it additionally moves via the new edges.
  - The H3 headline is the CONTRAST: sleep helps under formula but ~not under
    rrf  <=>  the graph carries retrieval signal. Reported as both per-scorer
    sleep deltas; their difference is the edge-attributable sleep gain.

Pure JSONL math -- touches no store, so it is safe to run under the abstinence
hold and reruns instantly. Reuses experiments/sleep-fork/analyze_sleep_fork.py's
verified paired_bootstrap_ci (same stats as the sleep-fork).

Usage:
    uv run experiments/sleep-bench/analyze_h3.py
    uv run experiments/sleep-bench/analyze_h3.py --frozen-label frozen --nrem-label nrem
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sleep-fork"))
from analyze_sleep_fork import paired_bootstrap_ci  # reuse the sleep-fork's stats verbatim

DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"

# Metrics to decompose (keys under each JSONL row's "metrics" dict).
METRICS = [
    "ndcg_at_budget", "r10", "mrr",
    "recall_any_10", "recall_all_10", "recall_any_effk",
]


def load_cell(results_dir: Path, label: str, scorer: str) -> dict:
    """Return {query_id: metrics_dict} for one (label, scorer) JSONL file."""
    path = results_dir / f"{label}__{scorer}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"missing Stage-A output: {path}\n"
            f"  run retrieve.py for --label {label} --scorers {scorer} first."
        )
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["query_id"]] = row["metrics"]
    return out


def mean(vals) -> float | None:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def paired_deltas(cell_b: dict, cell_a: dict, common: list, metric: str) -> list:
    """cell_b - cell_a per query (skip a query if either side lacks the metric)."""
    deltas = []
    for q in common:
        vb = cell_b[q].get(metric)
        va = cell_a[q].get(metric)
        if vb is None or va is None:
            continue
        deltas.append(vb - va)
    return deltas


def fmt_ci(ci: dict) -> str:
    if ci["mean"] is None:
        return "n=0"
    star = "" if (ci["ci_low"] <= 0 <= ci["ci_high"]) else "  *"  # * = CI excludes 0
    return (f"{ci['mean']:+.4f}  [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]  "
            f"n={ci['n']} pos={ci['frac_positive']:.2f}{star}")


def main() -> None:
    ap = argparse.ArgumentParser(description="H3 (NREM/graph) decomposition analyzer")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--frozen-label", default="frozen")
    ap.add_argument("--nrem-label", default="nrem")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)  # fixed seed = sleep-fork parity
    args = ap.parse_args()

    rdir = Path(args.results_dir).expanduser()
    cells = {
        ("frozen", "rrf"): load_cell(rdir, args.frozen_label, "rrf"),
        ("frozen", "formula"): load_cell(rdir, args.frozen_label, "formula"),
        ("nrem", "rrf"): load_cell(rdir, args.nrem_label, "rrf"),
        ("nrem", "formula"): load_cell(rdir, args.nrem_label, "formula"),
    }

    # Common query set across all four cells -- paired deltas require every query
    # present in every cell (a query missing from one arm can't be paired).
    common = set(cells[("frozen", "rrf")])
    for c in cells.values():
        common &= set(c)
    common = sorted(common)
    if not common:
        print("ERROR: no query_id common to all four cells.")
        sys.exit(1)
    print(f"[h3] {len(common)} query_ids common to all 4 cells "
          f"(frozen/nrem x rrf/formula), n_boot={args.n_boot}, seed={args.seed}\n")

    # --- 2x2 of per-cell means ---
    print("=== Per-cell means (arm x scorer) ===")
    header = f"{'metric':<18} {'frozen.rrf':>11} {'frozen.form':>12} {'nrem.rrf':>11} {'nrem.form':>11}"
    print(header)
    print("-" * len(header))
    for m in METRICS:
        row = [mean(cells[(arm, sc)][q].get(m) for q in common)
               for arm, sc in (("frozen", "rrf"), ("frozen", "formula"),
                               ("nrem", "rrf"), ("nrem", "formula"))]
        cells_str = "  ".join(f"{v:>10.4f}" if v is not None else f"{'--':>10}" for v in row)
        print(f"{m:<18} {cells_str}")

    def report(title, cell_b, cell_a):
        print(f"\n=== {title} ===")
        print("  (* = 95% bootstrap CI excludes 0)")
        for m in METRICS:
            deltas = paired_deltas(cells[cell_b], cells[cell_a], common, m)
            ci = paired_bootstrap_ci(deltas, args.n_boot, args.seed)
            print(f"  {m:<18} {fmt_ci(ci)}")

    # --- GRAPH contribution: formula - rrf, per arm ---
    report("GRAPH  Delta(formula - rrf) on FROZEN  [expect ~0: 0 edges]",
           ("frozen", "formula"), ("frozen", "rrf"))
    report("GRAPH  Delta(formula - rrf) on NREM    [the edge contribution]",
           ("nrem", "formula"), ("nrem", "rrf"))

    # --- SLEEP contribution: nrem - frozen, per scorer ---
    report("SLEEP  Delta(nrem - frozen) under RRF     [content/index effect only]",
           ("nrem", "rrf"), ("frozen", "rrf"))
    report("SLEEP  Delta(nrem - frozen) under FORMULA  [content + graph effect]",
           ("nrem", "formula"), ("frozen", "formula"))

    print("\n[h3] Verdict logic: H3 holds if SLEEP-under-FORMULA shows a positive "
          "CI-clear delta while SLEEP-under-RRF is ~null -- i.e. sleep's benefit "
          "is carried by the graph (edges), visible only to the PPR-aware scorer. "
          "The GRAPH-on-NREM block is the same signal read the other way.")


if __name__ == "__main__":
    main()
