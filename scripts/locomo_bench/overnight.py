"""Overnight experiment runner for LoCoMo reranker.

Runs a full sequence: forward stepwise → train → eval baseline → eval expanded.
Logs everything to timestamped files. Parse results in the morning.

Usage:
    uv run scripts/locomo_bench/overnight.py
    uv run scripts/locomo_bench/overnight.py --skip-selection --features vec_rank query_coverage fts_rank ...
"""

import argparse
import ast
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

LOG_DIR = Path.home() / ".somnigraph" / "benchmark" / "overnight"
LOG_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")


def log_path(name: str) -> Path:
    return LOG_DIR / f"{TIMESTAMP}_{name}.log"


SUMMARY_PATH = LOG_DIR / f"{TIMESTAMP}_summary.log"


def _append_summary(text: str):
    """Append a line to the summary log, flushing immediately."""
    with open(SUMMARY_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n")
        f.flush()


def run_step(name: str, cmd: list[str], parse_fn=None):
    """Run a command, log output, optionally parse result."""
    path = log_path(name)
    cmd_str = " ".join(cmd)
    print(f"\n{'='*70}")
    print(f"  STEP: {name}")
    print(f"  CMD:  {cmd_str}")
    print(f"  LOG:  {path}")
    print(f"{'='*70}\n")

    _append_summary(f"\n--- {name} ---")
    _append_summary(f"CMD: {cmd_str}")
    _append_summary(f"LOG: {path}")
    _append_summary(f"Started: {datetime.now().strftime('%H:%M:%S')}")

    t0 = time.time()
    with open(path, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        result_lines = []
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()
            result_lines.append(line)
        proc.wait()

    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
    summary = f"[{status}] {elapsed:.0f}s"
    print(f"\n{summary}")
    _append_summary(summary)

    # Capture tail of output (last table or key result) in summary
    tail = [l.rstrip() for l in result_lines[-20:] if l.strip()]
    if tail:
        _append_summary("Output tail:")
        for line in tail:
            _append_summary(f"  {line}")

    if proc.returncode != 0:
        print(f"ABORTING — see {path}")
        sys.exit(1)

    if parse_fn:
        return parse_fn(result_lines)
    return None


def parse_forward_features(lines: list[str]) -> list[str]:
    """Extract feature names from forward stepwise output."""
    for line in reversed(lines):
        m = re.search(r"Final set \(\d+\): (\[.*\])", line)
        if m:
            features = ast.literal_eval(m.group(1))
            print(f"  → Selected features: {features}")
            return features
    print("ERROR: Could not parse forward stepwise output")
    sys.exit(1)


def parse_eval_summary(lines: list[str]) -> str:
    """Extract the summary table from eval output."""
    table_lines = []
    capture = False
    for line in lines:
        if "Category" in line and "MRR" in line:
            capture = True
        if capture:
            table_lines.append(line.rstrip())
            if "OVERALL" in line:
                # Grab one more line for the (excludes adversarial) note
                capture = "maybe_one_more"
            elif capture == "maybe_one_more":
                table_lines.append(line.rstrip())
                break
    return "\n".join(table_lines)


def main():
    parser = argparse.ArgumentParser(description="Overnight LoCoMo reranker experiments")
    parser.add_argument("--skip-selection", action="store_true",
                        help="Skip forward stepwise, use --features directly")
    parser.add_argument("--features", nargs="+",
                        help="Feature names to use (with --skip-selection)")
    parser.add_argument("--n-estimators-select", type=int, default=300,
                        help="n_estimators for forward stepwise (default: 300)")
    parser.add_argument("--n-estimators-train", type=int, default=300,
                        help="n_estimators for final model (default: 300)")
    parser.add_argument("--conversations", nargs="+", type=int,
                        default=list(range(10)),
                        help="Conversation indices for eval (default: 0-9)")
    args = parser.parse_args()

    conv_args = [str(c) for c in args.conversations]

    print(f"Overnight run started: {TIMESTAMP}")
    print(f"Logs: {LOG_DIR}")
    print(f"Conversations: {args.conversations}")

    # Write config to summary
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(f"Overnight run: {TIMESTAMP}\n")
        f.write(f"Conversations: {args.conversations}\n")
        f.write(f"n_estimators: select={args.n_estimators_select}, train={args.n_estimators_train}\n")

    # --- Step 1: Forward stepwise feature selection ---
    if args.skip_selection:
        if not args.features:
            print("ERROR: --skip-selection requires --features")
            sys.exit(1)
        features = args.features
        print(f"Skipping selection, using: {features}")
    else:
        features = run_step(
            "01_forward_stepwise",
            ["uv", "run", "scripts/locomo_bench/train_locomo_reranker.py",
             "--two-phase", "--select", "forward",
             "--n-estimators", str(args.n_estimators_select)],
            parse_fn=parse_forward_features,
        )

    _append_summary(f"\nForward features ({len(features)}): {features}")

    # --- Step 1b: Backward elimination for comparison ---
    if args.skip_selection:
        backward_features = features  # no separate backward set
        _append_summary("Backward: skipped (--skip-selection)")
    else:
        backward_features = run_step(
            "01b_backward_elimination",
            ["uv", "run", "scripts/locomo_bench/train_locomo_reranker.py",
             "--two-phase", "--select", "backward",
             "--n-estimators", str(args.n_estimators_select)],
            parse_fn=parse_forward_features,  # same output format
        )
        _append_summary(f"Backward features ({len(backward_features)}): {backward_features}")

    # --- Run all feature sets ---
    old_features = [
        "fts_rank", "vec_rank", "vec_dist", "theme_overlap",
        "query_coverage", "speaker_match", "query_length", "token_count",
        "has_temporal_expr", "entity_bridge_count", "theme_complementarity",
        "centroid_distance", "entity_fts_rank", "sub_query_hit_count",
        "seed_keyword_overlap",
    ]

    feature_sets = [("forward", features)]
    if set(backward_features) != set(features):
        feature_sets.append(("backward", backward_features))
    else:
        _append_summary("Backward == forward, skipping duplicate eval")
    feature_sets.append(("old15", old_features))

    step = 2
    for set_name, feat_list in feature_sets:
        _append_summary(f"\n=== Evaluating: {set_name} ({len(feat_list)} features) ===")

        # Train and save model
        run_step(
            f"{step:02d}_train_{set_name}",
            ["uv", "run", "scripts/locomo_bench/train_locomo_reranker.py",
             "--two-phase", "--train-only", "--save-model",
             "--n-estimators", str(args.n_estimators_train),
             "--feature-names"] + feat_list,
        )
        step += 1

        # Eval baseline (no expansion)
        run_step(
            f"{step:02d}_eval_{set_name}_baseline",
            ["uv", "run", "scripts/locomo_bench/eval_retrieval.py",
             "--dataset", "locomo", "--configs", "locomo_reranker",
             "--conversations"] + conv_args,
        )
        step += 1

        # Eval with expansion
        run_step(
            f"{step:02d}_eval_{set_name}_expanded",
            ["uv", "run", "scripts/locomo_bench/eval_retrieval.py",
             "--dataset", "locomo", "--configs", "locomo_reranker",
             "--conversations"] + conv_args + ["--expand-all"],
        )
        step += 1

    # Restore best model (forward) as deployed
    run_step(
        f"{step:02d}_restore_forward_model",
        ["uv", "run", "scripts/locomo_bench/train_locomo_reranker.py",
         "--two-phase", "--train-only", "--save-model",
         "--n-estimators", str(args.n_estimators_train),
         "--feature-names"] + features,
    )

    # --- Done ---
    print(f"\n{'='*70}")
    print(f"  ALL STEPS COMPLETE")
    print(f"  Summary: {LOG_DIR / f'{TIMESTAMP}_summary.log'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
