"""Run 5 feature-set variations × 5 seeds and produce a comparison table.

Usage:
    python scripts/locomo_bench/cv_variations.py
"""

import subprocess
import sys
import re
from collections import defaultdict

SEEDS = [42, 123, 456, 789, 0]
N_ESTIMATORS = 300

INTERSECTION = [
    "fts_rank", "vec_rank", "fts_bm25", "vec_dist", "query_coverage",
    "query_length", "rank_agreement", "score_percentile", "has_temporal_expr",
    "topk_session_frac", "mmr_redundancy", "unique_token_frac",
    "phase1_rrf_score", "graph_synthetic_score", "graph_coref_hits",
]

CONFIG_K = [f for f in INTERSECTION if f != "score_percentile"] + ["token_count"]

VARIATIONS = {
    # P: K + score_percentile (keep both token_count and score_percentile)
    "P_K_plus_sp": CONFIG_K + ["score_percentile"],
    # Q: K + centroid_distance
    "Q_K_plus_cd": CONFIG_K + ["centroid_distance"],
    # R: K swap has_temporal_expr for centroid_distance
    "R_K_hte_to_cd": [f for f in CONFIG_K if f != "has_temporal_expr"] + ["centroid_distance"],
    # S: K swap has_temporal_expr for entity_density
    "S_K_hte_to_ed": [f for f in CONFIG_K if f != "has_temporal_expr"] + ["entity_density"],
    # T: K minus has_temporal_expr (trim weakest)
    "T_K_trim_hte": [f for f in CONFIG_K if f != "has_temporal_expr"],
}

METRICS = ["MRR", "NDCG@10", "R@1", "R@3", "R@5", "R@10", "R@20"]

SCRIPT = "scripts/locomo_bench/train_locomo_reranker.py"


def parse_seed_results(output: str) -> dict[str, dict[str, float]]:
    """Parse per-seed metrics from train script output."""
    results = {}
    for line in output.splitlines():
        line = line.strip()
        # Match lines like: seed_42          1977   0.584    0.617   43.7%   68.8%   76.7%   85.0%   91.2%
        m = re.match(
            r"(seed_\d+)\s+\d+\s+"
            r"([\d.]+)\s+([\d.]+)\s+"
            r"([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%",
            line,
        )
        if m:
            seed_name = m.group(1)
            results[seed_name] = {
                "MRR": float(m.group(2)),
                "NDCG@10": float(m.group(3)),
                "R@1": float(m.group(4)),
                "R@3": float(m.group(5)),
                "R@5": float(m.group(6)),
                "R@10": float(m.group(7)),
                "R@20": float(m.group(8)),
            }
    return results


def run_variation(name: str, features: list[str]) -> dict[str, dict[str, float]]:
    """Run one variation across all seeds."""
    cmd = [
        sys.executable, SCRIPT,
        "--train-only",
        "--n-estimators", str(N_ESTIMATORS),
        "--random-seeds", *[str(s) for s in SEEDS],
        "--feature-names", *features,
    ]
    print(f"\n{'='*70}")
    print(f"  {name} ({len(features)} features)")
    print(f"{'='*70}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")
        return {}
    print(result.stdout)
    return parse_seed_results(result.stdout)


def main():
    all_variation_results = {}

    for name, features in VARIATIONS.items():
        seed_results = run_variation(name, features)
        all_variation_results[name] = seed_results

    # Summary table
    print(f"\n{'='*90}")
    print(f"  SUMMARY: Mean across {len(SEEDS)} seeds")
    print(f"{'='*90}")
    header = f"{'Config':<22} {'N':>3}  {'MRR':>6}  {'NDCG':>6}  {'R@1':>6}  {'R@3':>6}  {'R@5':>6}  {'R@10':>6}  {'R@20':>6}"
    print(header)
    print("-" * len(header))

    for name, seed_results in all_variation_results.items():
        if not seed_results:
            continue
        n_feat = len(VARIATIONS[name])
        means = {}
        for metric in METRICS:
            vals = [sr[metric] for sr in seed_results.values()]
            means[metric] = sum(vals) / len(vals)

        r_metrics = ["R@1", "R@3", "R@5", "R@10", "R@20"]
        fmt_parts = [f"{name:<22} {n_feat:>3}"]
        fmt_parts.append(f"  {means['MRR']:>6.3f}")
        fmt_parts.append(f"  {means['NDCG@10']:>6.3f}")
        for m in r_metrics:
            fmt_parts.append(f"  {means[m]:>5.1f}%")
        print("".join(fmt_parts))

    # Best-seed table
    print(f"\n{'='*90}")
    print(f"  BEST SEED per variation (by R@10)")
    print(f"{'='*90}")
    print(header)
    print("-" * len(header))

    for name, seed_results in all_variation_results.items():
        if not seed_results:
            continue
        n_feat = len(VARIATIONS[name])
        best_seed = max(seed_results, key=lambda s: seed_results[s]["R@10"])
        sr = seed_results[best_seed]
        r_metrics = ["R@1", "R@3", "R@5", "R@10", "R@20"]
        fmt_parts = [f"{name:<22} {n_feat:>3}"]
        fmt_parts.append(f"  {sr['MRR']:>6.3f}")
        fmt_parts.append(f"  {sr['NDCG@10']:>6.3f}")
        for m in r_metrics:
            fmt_parts.append(f"  {sr[m]:>5.1f}%")
        fmt_parts.append(f"  ({best_seed})")
        print("".join(fmt_parts))

    # Variance table
    print(f"\n{'='*60}")
    print(f"  R@10 VARIANCE across seeds")
    print(f"{'='*60}")
    print(f"{'Config':<22} {'Min':>6}  {'Max':>6}  {'Spread':>6}")
    print("-" * 46)
    for name, seed_results in all_variation_results.items():
        if not seed_results:
            continue
        vals = [sr["R@10"] for sr in seed_results.values()]
        print(f"{name:<22} {min(vals):>5.1f}%  {max(vals):>5.1f}%  {max(vals)-min(vals):>5.1f}pp")


if __name__ == "__main__":
    main()
