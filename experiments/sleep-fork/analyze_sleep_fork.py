#!/usr/bin/env python3
"""Sleep counterfactual fork — paired delta + attribution analysis.

Joins the frozen-copy baseline eval against the slept-copy eval on the identical
GT query set, computes paired per-query deltas for NDCG@5k / R@10 / MRR with a
bootstrap 95% CI, and attributes the delta to concrete sleep actions by diffing
the two memory stores.

Both evals were produced by scripts/locomo_bench/eval_retrieval.py with
--configs formula (the learned reranker is on formula fallback — reranker_model.txt
absent since 2026-04-07). So this is a sleep-vs-FORMULA measurement, not
sleep-vs-reranker. See findings-sleep-fork.md.

Metrics:
  - NDCG@5k : taken from each record's `ndcg` field (token-budget NDCG at the
              budget the eval ran with; each side uses its own store's token_map,
              so a GT memory archived by sleep correctly becomes unretrievable).
  - R@10    : |relevant ∩ top-10 retrieved| / |relevant|, relevant = GT score>=0.5.
  - MRR     : 1 / rank of first relevant in the retrieved list (0 if none in list).

Usage:
  uv run python analyze_sleep_fork.py \
    --baseline <A>/eval_baseline.jsonl --slept <B>/eval_slept.jsonl \
    --gt <A>/tuning_studies/gt_calibrated.json \
    --db-a <A>/memory.db --db-b <B>/memory.db \
    --out <BASE>/sleep_fork_results.json [--seed 1729] [--n-boot 10000]
"""
import argparse
import json
import math
import sqlite3
from pathlib import Path

import numpy as np

REL_THRESHOLD = 0.5  # matches eval_retrieval.py relevant_count definition


def load_records(path, config="formula"):
    """query -> record, keeping only the requested config."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("config") != config:
                continue
            out[r["query"]] = r
    return out


def recall_at_k(retrieved, rel_ids, k):
    if not rel_ids:
        return None
    topk = retrieved[:k]
    hit = len(rel_ids & set(topk))
    return hit / len(rel_ids)


def mrr(retrieved, rel_ids):
    if not rel_ids:
        return None
    for i, mid in enumerate(retrieved):
        if mid in rel_ids:
            return 1.0 / (i + 1)
    return 0.0


def paired_bootstrap_ci(deltas, n_boot, seed, alpha=0.05):
    """Bootstrap CI on the mean of paired deltas. deltas is a 1-D array."""
    deltas = np.asarray([d for d in deltas if d is not None], dtype=float)
    n = len(deltas)
    if n == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = deltas[idx].mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return {
        "n": int(n),
        "mean": float(deltas.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "frac_positive": float((deltas > 0).mean()),
        "frac_negative": float((deltas < 0).mean()),
        "frac_zero": float((deltas == 0).mean()),
    }


def summarize(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0, "mean": None}
    return {"n": len(vals), "mean": float(np.mean(vals))}


def db_counts(db_path):
    """Attribution snapshot of one store."""
    c = sqlite3.connect(str(db_path))
    c.row_factory = sqlite3.Row
    out = {}
    status = dict(c.execute(
        "SELECT status, COUNT(*) FROM memories GROUP BY status").fetchall())
    out["memories_by_status"] = {k: v for k, v in status.items()}
    out["memories_total"] = c.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    out["memories_source_sleep"] = c.execute(
        "SELECT COUNT(*) FROM memories WHERE source='sleep'").fetchone()[0]
    out["edges_total"] = c.execute("SELECT COUNT(*) FROM memory_edges").fetchone()[0]
    out["edges_by_type"] = {
        r[0]: r[1] for r in c.execute(
            "SELECT edge_type, COUNT(*) FROM memory_edges GROUP BY edge_type "
            "ORDER BY 2 DESC").fetchall()}
    out["edges_by_creator"] = {
        (r[0] or "?"): r[1] for r in c.execute(
            "SELECT created_by, COUNT(*) FROM memory_edges GROUP BY created_by "
            "ORDER BY 2 DESC").fetchall()}
    try:
        out["sleep_log_rows"] = c.execute("SELECT COUNT(*) FROM sleep_log").fetchone()[0]
    except sqlite3.OperationalError:
        out["sleep_log_rows"] = None
    c.close()
    return out


def db_attribution(db_a, db_b):
    a = db_counts(db_a)
    b = db_counts(db_b)

    def statusdelta(k):
        return b["memories_by_status"].get(k, 0) - a["memories_by_status"].get(k, 0)

    diff = {
        "active_delta": statusdelta("active"),
        "dormant_delta": statusdelta("dormant"),
        "superseded_delta": statusdelta("superseded"),
        "deleted_delta": statusdelta("deleted"),
        "total_delta": b["memories_total"] - a["memories_total"],
        "source_sleep_delta": b["memories_source_sleep"] - a["memories_source_sleep"],
        "edges_total_delta": b["edges_total"] - a["edges_total"],
        "sleep_log_rows_delta": (
            (b["sleep_log_rows"] or 0) - (a["sleep_log_rows"] or 0)
            if a["sleep_log_rows"] is not None else None),
    }
    return {"frozen_A": a, "slept_B": b, "diff_B_minus_A": diff}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--slept", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--db-a", required=True)
    ap.add_argument("--db-b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="formula")
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--reranker-loaded", action="store_true",
                    help="set only if the learned reranker actually loaded (it did not)")
    args = ap.parse_args()

    base = load_records(args.baseline, args.config)
    slept = load_records(args.slept, args.config)
    gt = json.loads(Path(args.gt).read_text())

    queries = [q for q in base if q in slept and q in gt]
    missing_base = [q for q in gt if q not in base]
    missing_slept = [q for q in gt if q not in slept]

    metrics = {"ndcg": {"b": [], "s": [], "d": []},
               "r10": {"b": [], "s": [], "d": []},
               "mrr": {"b": [], "s": [], "d": []}}
    per_query = []

    for q in queries:
        rb, rs = base[q], slept[q]
        rel_ids = {mid for mid, sc in gt[q].items() if sc >= REL_THRESHOLD}

        ndcg_b, ndcg_s = rb.get("ndcg"), rs.get("ndcg")
        r10_b = recall_at_k(rb.get("retrieved", []), rel_ids, 10)
        r10_s = recall_at_k(rs.get("retrieved", []), rel_ids, 10)
        mrr_b = mrr(rb.get("retrieved", []), rel_ids)
        mrr_s = mrr(rs.get("retrieved", []), rel_ids)

        metrics["ndcg"]["b"].append(ndcg_b)
        metrics["ndcg"]["s"].append(ndcg_s)
        metrics["ndcg"]["d"].append(
            None if ndcg_b is None or ndcg_s is None else ndcg_s - ndcg_b)
        metrics["r10"]["b"].append(r10_b)
        metrics["r10"]["s"].append(r10_s)
        metrics["r10"]["d"].append(
            None if r10_b is None or r10_s is None else r10_s - r10_b)
        metrics["mrr"]["b"].append(mrr_b)
        metrics["mrr"]["s"].append(mrr_s)
        metrics["mrr"]["d"].append(
            None if mrr_b is None or mrr_s is None else mrr_s - mrr_b)

        per_query.append({
            "query": q, "n_relevant": len(rel_ids),
            "ndcg_b": ndcg_b, "ndcg_s": ndcg_s,
            "r10_b": r10_b, "r10_s": r10_s,
            "mrr_b": mrr_b, "mrr_s": mrr_s,
        })

    results = {
        "config": args.config,
        "reranker_loaded": bool(args.reranker_loaded),
        "scorer_note": (
            "FORMULA scoring on BOTH sides (reranker_model.txt absent since "
            "2026-04-07). This is a sleep-vs-formula delta, not sleep-vs-reranker."),
        "gt_file": args.gt,
        "n_queries_joined": len(queries),
        "n_gt_queries": len(gt),
        "queries_missing_from_baseline": len(missing_base),
        "queries_missing_from_slept": len(missing_slept),
        "seed": args.seed,
        "n_boot": args.n_boot,
        "baseline": {
            "ndcg@5k": summarize(metrics["ndcg"]["b"]),
            "r@10": summarize(metrics["r10"]["b"]),
            "mrr": summarize(metrics["mrr"]["b"]),
        },
        "slept": {
            "ndcg@5k": summarize(metrics["ndcg"]["s"]),
            "r@10": summarize(metrics["r10"]["s"]),
            "mrr": summarize(metrics["mrr"]["s"]),
        },
        "paired_delta": {
            "ndcg@5k": paired_bootstrap_ci(metrics["ndcg"]["d"], args.n_boot, args.seed),
            "r@10": paired_bootstrap_ci(metrics["r10"]["d"], args.n_boot, args.seed + 1),
            "mrr": paired_bootstrap_ci(metrics["mrr"]["d"], args.n_boot, args.seed + 2),
        },
        "sleep_attribution": db_attribution(args.db_a, args.db_b),
    }

    Path(args.out).write_text(json.dumps(results, indent=2))

    # console summary
    def fmt(ci):
        if ci["mean"] is None:
            return "n/a"
        return (f"{ci['mean']:+.4f}  95% CI [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]  "
                f"(n={ci['n']}, +{ci['frac_positive']:.0%}/-{ci['frac_negative']:.0%})")

    print("=" * 72)
    print(f"SLEEP FORK — paired delta (slept − frozen), config={args.config}")
    print(f"  scorer: FORMULA on both sides (reranker fallback); joined {len(queries)} queries")
    print("-" * 72)
    for m, label in [("ndcg@5k", "NDCG@5k"), ("r@10", "R@10"), ("mrr", "MRR")]:
        b = results["baseline"][m]["mean"]
        s = results["slept"][m]["mean"]
        bs = f"{b:.4f}" if b is not None else "n/a"
        ss = f"{s:.4f}" if s is not None else "n/a"
        print(f"  {label:8s}  frozen={bs}  slept={ss}   delta={fmt(results['paired_delta'][m])}")
    print("-" * 72)
    d = results["sleep_attribution"]["diff_B_minus_A"]
    print(f"  sleep actions (B−A): active {d['active_delta']:+d}, dormant {d['dormant_delta']:+d}, "
          f"superseded {d['superseded_delta']:+d}, deleted {d['deleted_delta']:+d}")
    print(f"                       total mem {d['total_delta']:+d}, source=sleep {d['source_sleep_delta']:+d}, "
          f"edges {d['edges_total_delta']:+d}")
    print(f"  results written to {args.out}")
    print("=" * 72)


if __name__ == "__main__":
    main()
