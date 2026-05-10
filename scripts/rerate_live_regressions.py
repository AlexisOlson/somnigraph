"""Re-rate live worst-regression queries via LLM judge to detect GT noise.

Motivation: V5+1b → V5+5b retrains showed persistent live worst-regression
queries (e.g. "claude.ai capacity two-tier KB chat attachments") whose
drill-down inspection (`scripts/drill_query_scores.py`) revealed the reranker
was returning the right memories at top-N — the GT labels themselves were
the problem. This script automates that diagnostic at scale: pick the bottom-N
live queries by reranker NDCG, re-rate the candidate pool with `claude -p`
using the same rubric as `judge_ground_truth.py`, and emit a side-by-side
disagreement report.

The output answers: of the live worst-regressions, how many are GT noise
vs real model bugs?

NDCG here is unweighted NDCG@10 against the existing GT — no token budget,
no production-RRF comparison. The diagnostic is about disagreement *between
GT and a fresh rater*, not about reranker-vs-RRF deltas; the simpler metric
keeps the script self-contained and lets it run from cached features
without rebuilding `full_data` (the load is minutes-expensive).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import lightgbm as lgb
import numpy as np

from memory.constants import DATA_DIR
from memory.db import get_db
from judge_ground_truth import JUDGE_SYSTEM, build_judge_prompt


def _call_claude(prompt: str, model: str) -> str | None:
    """Call the claude CLI and return stripped stdout, or None on failure.

    Reimplemented locally because `judge_ground_truth.py:_call_claude`
    hardcodes `claude.cmd` on Windows; this resolves whatever `claude`
    binary is on PATH (claude.exe on Windows installs from .local/bin).
    """
    claude_bin = shutil.which("claude")
    if claude_bin is None:
        print("    ERROR: 'claude' not found on PATH", flush=True)
        return None
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)

    err = ""
    for attempt in range(3):
        try:
            result = subprocess.run(
                [claude_bin, "-p", "--model", model],
                input=prompt,
                capture_output=True,
                timeout=300,
                env=env,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode == 0:
                break
            err = result.stderr.strip()
        except subprocess.TimeoutExpired:
            err = "timeout"
        if attempt < 2:
            time.sleep(2 ** attempt)
    else:
        print(f"    WARNING: claude -p failed after 3 attempts: {err}", flush=True)
        return None

    text = result.stdout.strip()
    if not text:
        return None

    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    return text


FEATURES_PATH = DATA_DIR / "tuning_studies" / "reranker_features.pkl"
MODEL_PATH = DATA_DIR / "tuning_studies" / "reranker_model.txt"
GT_PATH = DATA_DIR / "tuning_studies" / "gt_feedback.json"
SAMPLE_WEIGHTS_PATH = DATA_DIR / "tuning_studies" / "gt_feedback_sample_weights.json"
DEFAULT_OUTPUT_DIR = DATA_DIR / "tuning_studies" / "rerate_audits"

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_BOTTOM_N = 20
DEFAULT_TOP_K = 5
CONTENT_CHARS = 600


def ndcg_at_k(ranked_mids: list[str], gt: dict[str, float], k: int = 10) -> float:
    """Standard NDCG@k. No token budget — see module docstring."""
    if not gt:
        return 0.0
    dcg = 0.0
    for i, mid in enumerate(ranked_mids[:k]):
        rel = gt.get(mid, 0.0)
        dcg += rel / math.log2(i + 2)
    ideal_rels = sorted(gt.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


def load_live_queries(sample_weights_path: Path) -> set[str]:
    sw = json.loads(sample_weights_path.read_text())
    return {q for q, v in sw["weights"].items() if v.get("mode") == "live"}


def per_query_predictions(
    booster: lgb.Booster,
    feature_data: dict,
    live_queries: set[str],
    gt: dict[str, dict[str, float]],
    top_k: int,
) -> list[dict]:
    """Return per-live-query: query, ndcg@10, top-K predictions with GT labels."""
    feature_matrix = feature_data["features"]
    memory_ids = feature_data["memory_ids"]
    gt_labels = feature_data["labels"]

    rows_by_query: dict[str, list[int]] = {}
    for i, (qtext, _mid) in enumerate(memory_ids):
        rows_by_query.setdefault(qtext, []).append(i)

    out: list[dict] = []
    for qtext, rows in rows_by_query.items():
        if qtext not in live_queries:
            continue
        gt_for_q = gt.get(qtext, {})
        if not gt_for_q:
            continue

        X = feature_matrix[rows]
        scores = booster.predict(X)
        order = np.argsort(-scores)
        ranked_mids = [memory_ids[rows[i]][1] for i in order]

        ndcg = ndcg_at_k(ranked_mids, gt_for_q, k=10)

        top_k_entries = []
        for k in range(min(top_k, len(order))):
            row_idx = rows[order[k]]
            mid = memory_ids[row_idx][1]
            top_k_entries.append({
                "rank": k + 1,
                "score": float(scores[order[k]]),
                "mid": mid,
                "gt_label": float(gt_labels[row_idx]),
            })

        # Positive-GT memories outside top-K (worth including in re-rating set)
        top_k_mids = {e["mid"] for e in top_k_entries}
        outside_positives = []
        for row_idx in rows:
            mid = memory_ids[row_idx][1]
            if mid in top_k_mids:
                continue
            label = float(gt_labels[row_idx])
            if label >= 0.3:  # threshold for "claimed positive in GT"
                rank_in_pred = next(
                    (k + 1 for k in range(len(order)) if rows[order[k]] == row_idx),
                    None,
                )
                outside_positives.append({
                    "rank": rank_in_pred,
                    "mid": mid,
                    "gt_label": label,
                })

        out.append({
            "query": qtext,
            "ndcg_at_10": ndcg,
            "n_positives": sum(1 for r in rows if gt_labels[r] >= 0.3),
            "top_k": top_k_entries,
            "outside_positives": outside_positives,
        })
    return out


def fetch_memory_content(mids: list[str]) -> dict[str, dict]:
    """Pull summary/themes/category/content for a set of memory IDs."""
    if not mids:
        return {}
    db = get_db()
    try:
        placeholders = ",".join("?" * len(mids))
        out = {}
        for r in db.execute(
            f"SELECT id, summary, themes, category, content FROM memories WHERE id IN ({placeholders})",
            mids,
        ):
            out[r["id"]] = {
                "id": r["id"],
                "summary": r["summary"] or "",
                "themes": r["themes"] or "[]",
                "category": r["category"] or "",
                "content": (r["content"] or "")[:CONTENT_CHARS],
            }
        return out
    finally:
        db.close()


def rerate_query(
    query: str,
    candidate_mids: list[str],
    content_map: dict[str, dict],
    model: str,
) -> dict[str, float] | None:
    """Call claude -p judge on the candidate pool. Returns {mid: rater_score}."""
    candidates = [content_map[mid] for mid in candidate_mids if mid in content_map]
    if not candidates:
        return None
    prompt = JUDGE_SYSTEM + "\n\n" + build_judge_prompt(query, candidates, "")
    text = _call_claude(prompt, model)
    if not text:
        return None
    try:
        grouped = json.loads(text)
    except json.JSONDecodeError:
        print(f"    WARNING: parse failure on rater output for: {query[:60]}", flush=True)
        return None
    GROUP_SIZE = 10
    out: dict[str, float] = {}
    for gi, start in enumerate(range(0, len(candidates), GROUP_SIZE), 1):
        group = candidates[start:start + GROUP_SIZE]
        scores = grouped.get(str(gi))
        if not isinstance(scores, list) or len(scores) != len(group):
            continue
        for c, s in zip(group, scores):
            if isinstance(s, (int, float)):
                out[c["id"]] = round(float(s), 4)
    return out


def classify_disagreement(
    top_k: list[dict],
    rater_scores: dict[str, float],
    delta_threshold: float = 0.3,
) -> dict:
    """Classify per-query pattern of GT-vs-rater agreement on top-K predictions."""
    rows = []
    for entry in top_k:
        rater = rater_scores.get(entry["mid"])
        rows.append({
            "rank": entry["rank"],
            "mid": entry["mid"],
            "score": entry["score"],
            "gt": entry["gt_label"],
            "rater": rater,
            "delta": (rater - entry["gt_label"]) if rater is not None else None,
        })

    deltas = [r["delta"] for r in rows if r["delta"] is not None]
    if not deltas:
        return {"verdict": "no-rater-data", "rows": rows}

    n_rater_higher = sum(1 for d in deltas if d > delta_threshold)
    n_rater_lower = sum(1 for d in deltas if d < -delta_threshold)
    mean_delta = sum(deltas) / len(deltas)

    # Classification:
    #  - "gt-under-rates": rater consistently scores model's top-K higher than GT
    #    (suggests GT under-labels; the regression is GT noise)
    #  - "gt-over-rates": rater consistently scores them lower (GT may be over-broad,
    #    pulling marginal items into "positive" territory; model is correctly demoting)
    #  - "model-bug": rater agrees with GT — the model is genuinely picking weak memories
    #  - "mixed": no clear pattern
    if n_rater_higher >= 2 and n_rater_lower == 0 and mean_delta > 0.15:
        verdict = "gt-under-rates"
    elif n_rater_lower >= 2 and n_rater_higher == 0 and mean_delta < -0.15:
        verdict = "gt-over-rates"
    elif n_rater_higher == 0 and n_rater_lower == 0:
        verdict = "agreement"
    else:
        verdict = "mixed"

    return {
        "verdict": verdict,
        "mean_delta": mean_delta,
        "n_rater_higher": n_rater_higher,
        "n_rater_lower": n_rater_lower,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bottom-n", type=int, default=DEFAULT_BOTTOM_N,
                        help=f"How many bottom-NDCG live queries to re-rate (default: {DEFAULT_BOTTOM_N})")
    parser.add_argument("--min-positives", type=int, default=2,
                        help="Skip queries with fewer than this many GT positives "
                             "(label >= 0.3). Zero-positive queries are trivially "
                             "NDCG=0 — they're labeling-completeness questions, not "
                             "ordering questions, and answer differently. Default: 2.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"Top-K predictions to surface per query (default: {DEFAULT_TOP_K})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Rater model (default: {DEFAULT_MODEL})")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: rerate_audits/<timestamp>.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip the rater call; print bottom-N selection only")
    parser.add_argument("--delta-threshold", type=float, default=0.3,
                        help="Per-row |rater - gt| threshold to count as a disagreement (default: 0.3)")
    args = parser.parse_args()

    print(f"Loading model from {MODEL_PATH}...", flush=True)
    booster = lgb.Booster(model_file=str(MODEL_PATH))

    print(f"Loading features from {FEATURES_PATH}...", flush=True)
    with open(FEATURES_PATH, "rb") as f:
        feature_data = pickle.load(f)

    print(f"Loading GT from {GT_PATH}...", flush=True)
    gt = json.loads(GT_PATH.read_text())

    print(f"Loading sample weights from {SAMPLE_WEIGHTS_PATH}...", flush=True)
    live_queries = load_live_queries(SAMPLE_WEIGHTS_PATH)
    print(f"  Live queries in sidecar: {len(live_queries)}", flush=True)

    print("Computing per-query NDCG@10...", flush=True)
    per_q = per_query_predictions(booster, feature_data, live_queries, gt, args.top_k)
    print(f"  Live queries with predictions + GT: {len(per_q)}", flush=True)

    pos_filtered = [q for q in per_q if q["n_positives"] >= args.min_positives]
    print(f"  After --min-positives={args.min_positives}: {len(pos_filtered)} eligible", flush=True)

    pos_filtered.sort(key=lambda x: x["ndcg_at_10"])
    bottom = pos_filtered[:args.bottom_n]

    print(f"\nBottom-{args.bottom_n} live queries by reranker NDCG@10:")
    print(f"{'NDCG':<8} {'#pos':<6} {'top1_gt':<8} query")
    for entry in bottom:
        top1 = entry["top_k"][0]["gt_label"] if entry["top_k"] else 0.0
        print(f"{entry['ndcg_at_10']:.4f}  {entry['n_positives']:<5}  {top1:.2f}      {entry['query'][:80]}")

    if args.dry_run:
        print("\n[dry-run] Skipping rater call.")
        return

    # Build content map for all candidates we'll need
    all_mids = set()
    for entry in bottom:
        all_mids.update(e["mid"] for e in entry["top_k"])
        all_mids.update(e["mid"] for e in entry["outside_positives"])
    content_map = fetch_memory_content(list(all_mids))
    print(f"\nFetched content for {len(content_map)} memories", flush=True)

    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        output_path = Path(args.output)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"rerate_{ts}.json"

    audit_rows = []
    verdict_counts: dict[str, int] = {}
    print(f"\nRe-rating with {args.model}...", flush=True)
    for i, entry in enumerate(bottom, 1):
        candidate_mids = [e["mid"] for e in entry["top_k"]] + [e["mid"] for e in entry["outside_positives"]]
        # Dedup while preserving order
        seen = set()
        candidate_mids = [m for m in candidate_mids if not (m in seen or seen.add(m))]

        t0 = time.time()
        rater_scores = rerate_query(entry["query"], candidate_mids, content_map, args.model)
        elapsed = time.time() - t0

        if rater_scores is None:
            verdict_counts["rater-failure"] = verdict_counts.get("rater-failure", 0) + 1
            audit_rows.append({**entry, "rater_scores": None, "verdict": "rater-failure"})
            print(f"  [{i}/{len(bottom)}] ({elapsed:.1f}s) FAILED: {entry['query'][:60]}", flush=True)
            continue

        cls = classify_disagreement(entry["top_k"], rater_scores, args.delta_threshold)
        verdict = cls["verdict"]
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

        audit_rows.append({
            "query": entry["query"],
            "ndcg_at_10": entry["ndcg_at_10"],
            "n_positives": entry["n_positives"],
            "top_k": entry["top_k"],
            "outside_positives": entry["outside_positives"],
            "rater_scores": rater_scores,
            "classification": cls,
        })

        print(f"  [{i}/{len(bottom)}] ({elapsed:.1f}s) verdict={verdict:<14} "
              f"mean_delta={cls.get('mean_delta', float('nan')):+.2f}  {entry['query'][:60]}",
              flush=True)

        # Checkpoint after each query
        with open(output_path, "w") as f:
            json.dump({
                "metadata": {
                    "model": args.model,
                    "bottom_n": args.bottom_n,
                    "top_k": args.top_k,
                    "delta_threshold": args.delta_threshold,
                    "verdict_counts": verdict_counts,
                    "completed": i,
                    "total": len(bottom),
                },
                "audit_rows": audit_rows,
            }, f, indent=2)

    print(f"\nVerdict summary:")
    for v, n in sorted(verdict_counts.items(), key=lambda x: -x[1]):
        print(f"  {v:<18} {n}")
    print(f"\nWritten: {output_path}")


if __name__ == "__main__":
    main()
