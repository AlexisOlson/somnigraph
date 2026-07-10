#!/usr/bin/env python3
"""Train driver for the H10 pre/post-sleep reranker 2x2 (WP4).

Trains ONE pointwise LightGBM reranker on ONE store (frozen -> R_pre, or the
slept store c2 -> R_post), on the convs-1-9 training queries supplied by WP1's
label bridge (`gt_2x2.json`, shape `{query_text: {memory_id: grade}}`). Both
models share the identical 24-feature schema (WP2 `SCHEMA_24`); on the frozen
store the four sleep-created features are all-zero columns and LightGBM cannot
split on them (gain == 0) -- that asymmetry IS the experiment (plan sec D2).

One store per invocation (the plan's run sequence trains R_pre and R_post in two
separate terminal jobs). Because `tune_gt` binds DATA_DIR / MEMORY_DB at import,
pointing at a store requires `ingest_baseline._point_at_fresh_store(store)`
followed by `importlib.reload(tune_gt)` -- the module-cache trap documented in
plan sec D5 and demonstrated by `bench_features.py`'s selftest. This is done
once, up front, before any `load_tuning_data` call.

Hyperparameters replicate the production V6 recipe verbatim from
`scripts/train_reranker.py` `train_and_evaluate()` (pointwise LGBMRegressor, NOT
lambdarank): objective=regression, metric=rmse, num_leaves=31,
learning_rate=0.1, n_estimators=500, min_child_samples=20, subsample=0.8,
colsample_bytree=0.8, random_state=42, GroupKFold(5) by query, early_stopping(50)
on a val eval_set per fold, final model fit on all rows. Added on top for the
plan's determinism requirement (sec Risks "LightGBM seeds fixed"):
deterministic=True + force_row_wise=True (documented deviation -- see the module
docstring's "Deviations" note in the final report; these do not change the
learning recipe, only pin float reduction order).

CV metric mirrors `train_reranker.evaluate_reranker_ranking`: per fold, fit on
the train queries, predict on the held-out queries' FULL candidate pool, rank by
predicted score, and score NDCG@5k / graded-recall@5k / recall@10 / MRR with the
same `tune_gt.compute_*` helpers at the 5000-token budget.

Artifacts (into --out-dir, default results-full):
  {name}.txt              booster save_model (the V6 bundle format)
  {name}__features.json   the mask list; order == model input column order
  {name}__importances.json  {feature: gain}
  {name}__cv.json         per-fold + mean metrics + config + n queries/candidates

All store access is read-only apart from `load_tuning_data`'s benign sidecar
caches (tune_gt_embeddings.pkl / tune_gt_ppr_cache.pkl / tune_gt_data.pkl written
into the store dir; memory.db itself is only SELECTed).

Usage (smoke -- your own testing only):
    uv run --extra tuning experiments/sleep-bench/train_2x2.py \
        --gt <scratchpad>/gt_smoke.json \
        --store ~/.somnigraph/benchmark/sleepbench-full/conv0-frozen \
        --name R_pre_smoke --max-queries 150

Full-scale invocations are Alexis's terminal jobs (printed at the end of a
successful smoke).
"""

import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import lightgbm as lgb  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from bench_features import (  # noqa: E402
    ALL_FEATURE_NAMES_35,
    SCHEMA_24,
    augment_memory_meta,
    extract_features_35,
    select_schema,
)

DEFAULT_LOCOMO = Path.home() / ".claude" / "repos" / "locomo" / "data" / "locomo10.json"
DEFAULT_OUT_DIR = HERE / "results-full"
DEFAULT_KEY_FILE = Path(r"C:\Users\Alexis\.claude\data\openai_api_key")
SEED = 42

# V6 parity -- copied verbatim from train_reranker.train_and_evaluate()'s
# lgb_params (scripts/train_reranker.py). deterministic/force_row_wise added for
# reproducibility (plan Risks: "LightGBM seeds fixed"); they pin the float
# reduction order without altering the learning recipe.
LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 31,
    "learning_rate": 0.1,
    "n_estimators": 500,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
    "random_state": SEED,
    "deterministic": True,
    "force_row_wise": True,
}


# ---------------------------------------------------------------------------
# Store pointing (the module-cache trap; reuse bench_features' proven sequence)
# ---------------------------------------------------------------------------


def _point_and_reload(store: Path, key_file: Path):
    """Point env at `store`, reset module caches, reload tune_gt, return it.

    Mirrors bench_features._extract_all_on_store's opening exactly.
    """
    from ingest_baseline import _point_at_fresh_store

    _point_at_fresh_store(store)
    if key_file.exists():
        os.environ["OPENAI_API_KEY"] = key_file.read_text().strip()

    import tune_gt
    importlib.reload(tune_gt)  # re-bind DATA_DIR / MEMORY_DB from mutated modules
    return tune_gt


# ---------------------------------------------------------------------------
# Feature-mask resolution (WP6 hook)
# ---------------------------------------------------------------------------


def _resolve_mask(spec: str | None) -> list[str]:
    """--feature-mask is optional. None -> SCHEMA_24. A JSON list literal (starts
    with '[') is parsed inline; anything else is treated as a path to a JSON file
    containing a list of feature names. Every name must be in ALL_FEATURE_NAMES_35.
    """
    if not spec:
        return list(SCHEMA_24)
    spec = spec.strip()
    if spec.startswith("["):
        names = json.loads(spec)
    else:
        names = json.loads(Path(spec).expanduser().read_text(encoding="utf-8"))
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ValueError("--feature-mask must resolve to a JSON list of strings")
    unknown = [n for n in names if n not in ALL_FEATURE_NAMES_35]
    if unknown:
        raise ValueError(f"--feature-mask names not in the 35-name superset: {unknown}")
    return list(names)


# ---------------------------------------------------------------------------
# GT truncation for smokes (limits OpenAI embedding cost to --max-queries)
# ---------------------------------------------------------------------------


def _maybe_truncate_gt(gt_path: Path, max_queries: int | None, scratch: Path) -> Path:
    """If max_queries is set, write a truncated copy (first N queries in file
    order) into `scratch` and return its path; else return gt_path unchanged.
    Deterministic: json preserves file/insertion order, so the first N are stable.
    """
    if not max_queries:
        return gt_path
    full = json.loads(gt_path.read_text(encoding="utf-8"))
    keys = list(full.keys())[:max_queries]
    trunc = {k: full[k] for k in keys}
    scratch.mkdir(parents=True, exist_ok=True)
    out = scratch / f"{gt_path.stem}.trunc{max_queries}.json"
    out.write_text(json.dumps(trunc, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[train2x2] truncated GT {len(full)} -> {len(trunc)} queries -> {out}")
    return out


# ---------------------------------------------------------------------------
# Extraction: build the (masked) feature matrix + labels + query groups
# ---------------------------------------------------------------------------


def _extract_matrix(tune_gt, gt_path: Path, mask: list[str]):
    """Point-and-extract already done by caller. Runs load_tuning_data, augments
    meta with the 4 sleep features, extracts 35 features per query, column-selects
    to `mask`. Returns a dict with the stacked matrix, labels, query_ids (group),
    memory_ids [(qtext, mid)], and the per-query ranked-eval scaffolding.
    """
    from memory.constants import PPR_DAMPING
    from memory.db import get_db
    from train_reranker import PRODUCTION_PARAMS

    db = get_db()
    meta = augment_memory_meta(db)

    full_data, ground_truth, _ = tune_gt.load_tuning_data(
        str(gt_path), ppr_dampings=[PPR_DAMPING]
    )
    params = dict(PRODUCTION_PARAMS)

    queries = full_data["gt_queries"]
    feats_all, labels_all, qids_all, mids_all = [], [], [], []
    kept_queries = []
    for qi, qtext in enumerate(queries):
        gt_for_q = ground_truth.get(qtext, {})
        if not gt_for_q:
            continue
        feats35, labels, mids = extract_features_35(
            qtext, params, full_data["search_data"],
            full_data["feedback_raw"], full_data["hebb_data"],
            meta, gt_for_q,
        )
        if len(mids) == 0:
            continue
        sub, _ = select_schema(feats35, ALL_FEATURE_NAMES_35, mask)
        qidx = len(kept_queries)
        kept_queries.append(qtext)
        feats_all.append(sub.astype(np.float32))
        labels_all.append(labels.astype(np.float32))
        qids_all.extend([qidx] * len(mids))
        mids_all.extend([(qtext, m) for m in mids])

    X = np.concatenate(feats_all, axis=0)
    y = np.concatenate(labels_all, axis=0)
    groups = np.array(qids_all, dtype=np.int32)

    return {
        "X": X,
        "y": y,
        "groups": groups,
        "memory_ids": mids_all,
        "queries": kept_queries,
        "ground_truth": ground_truth,
        "token_map": full_data["token_map"],
    }


# ---------------------------------------------------------------------------
# Ranking metrics per fold (mirror train_reranker.evaluate_reranker_ranking)
# ---------------------------------------------------------------------------


def _compute_mrr(ranked_results: dict, ground_truth: dict, threshold: float = 0.5) -> float:
    """Mean reciprocal rank of the first relevant (GT >= threshold) memory."""
    rr_sum = 0.0
    n = 0
    for qtext, ranked in ranked_results.items():
        gt = ground_truth.get(qtext)
        if not gt:
            continue
        n += 1
        relevant = {mid for mid, s in gt.items() if s >= threshold}
        if not relevant:
            continue
        for rank, mid in enumerate(ranked, start=1):
            if mid in relevant:
                rr_sum += 1.0 / rank
                break
    return rr_sum / n if n else 0.0


def _rank_and_score(tune_gt, preds, val_idx, groups, memory_ids, queries,
                    ground_truth, token_map):
    """Group val-fold predictions by query, rank, and score the four metrics."""
    ranked_results = {}
    for qi in sorted(set(groups[val_idx].tolist())):
        qtext = queries[qi]
        mask = groups[val_idx] == qi
        rows = val_idx[mask]
        q_preds = preds[mask]
        q_mids = [memory_ids[r][1] for r in rows]
        order = np.argsort(-q_preds)
        ranked_results[qtext] = [q_mids[i] for i in order]

    ndcg = tune_gt.compute_ndcg(ranked_results, token_map, ground_truth, budget=5000)
    graded = tune_gt.compute_graded_recall(ranked_results, token_map, ground_truth, budget=5000)
    r10 = tune_gt.compute_recall_at_k(ranked_results, ground_truth, k=10, threshold=0.5)
    mrr = _compute_mrr(ranked_results, ground_truth)
    return {
        "ndcg_5k": float(ndcg),
        "graded_recall_5k": float(graded),
        "recall_10": float(r10),
        "mrr": float(mrr),
    }


# ---------------------------------------------------------------------------
# Train + CV
# ---------------------------------------------------------------------------


def train_and_cv(tune_gt, data: dict, n_folds: int):
    """GroupKFold CV (ranking metrics + RMSE) then fit the final model on all rows."""
    X, y, groups = data["X"], data["y"], data["groups"]
    memory_ids = data["memory_ids"]
    queries = data["queries"]
    ground_truth = data["ground_truth"]
    token_map = data["token_map"]

    n_queries = len(queries)
    n_folds_eff = min(n_folds, n_queries)
    if n_folds_eff < 2:
        raise SystemExit(f"[train2x2] need >=2 query-groups for CV, have {n_queries}")

    print(f"\n[train2x2] CV: {len(y)} rows, {X.shape[1]} features, "
          f"{n_queries} queries, {n_folds_eff} folds")

    gkf = GroupKFold(n_splits=n_folds_eff)
    fold_metrics = []
    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(
            X[train_idx], y[train_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        preds = np.clip(model.predict(X[val_idx]), 0, 1)
        rmse = float(np.sqrt(np.mean((preds - y[val_idx]) ** 2)))
        rank_m = _rank_and_score(tune_gt, preds, val_idx, groups, memory_ids,
                                 queries, ground_truth, token_map)
        n_val_q = len(set(groups[val_idx].tolist()))
        fm = {
            "fold": fold_i,
            "n_val_queries": n_val_q,
            "n_val_rows": int(len(val_idx)),
            "best_iter": int(model.best_iteration_ or LGB_PARAMS["n_estimators"]),
            "rmse": rmse,
            **rank_m,
        }
        fold_metrics.append(fm)
        print(f"  Fold {fold_i}: NDCG@5k={fm['ndcg_5k']:.4f} "
              f"gRecall@5k={fm['graded_recall_5k']:.4f} R@10={fm['recall_10']:.4f} "
              f"MRR={fm['mrr']:.4f} RMSE={fm['rmse']:.4f} "
              f"(val_q={n_val_q}, best_iter={fm['best_iter']})")

    keys = ["ndcg_5k", "graded_recall_5k", "recall_10", "mrr", "rmse"]
    mean = {f"mean_{k}": float(np.mean([f[k] for f in fold_metrics])) for k in keys}
    print("\n[train2x2] MEAN across folds:")
    for k in keys:
        print(f"    {k:18s} {mean['mean_' + k]:.4f}")

    print("\n[train2x2] fitting final model on all rows...")
    final = lgb.LGBMRegressor(**LGB_PARAMS)
    final.fit(X, y)
    return fold_metrics, mean, final


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------


def _save_artifacts(out_dir: Path, name: str, final_model, mask: list[str],
                    fold_metrics: list[dict], mean: dict, data: dict,
                    config: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    booster = final_model.booster_

    model_path = out_dir / f"{name}.txt"
    booster.save_model(str(model_path))

    feats_path = out_dir / f"{name}__features.json"
    feats_path.write_text(json.dumps(mask, indent=2), encoding="utf-8")

    # Gain importance, aligned to the model's input column order == mask order.
    gains = booster.feature_importance(importance_type="gain")
    importances = {mask[i]: float(gains[i]) for i in range(len(mask))}
    imp_path = out_dir / f"{name}__importances.json"
    imp_path.write_text(json.dumps(importances, indent=2, sort_keys=True), encoding="utf-8")

    cv = {
        "name": name,
        "config": config,
        "n_queries": len(data["queries"]),
        "n_candidates": int(len(data["y"])),
        "n_features": len(mask),
        "features": mask,
        "fold_metrics": fold_metrics,
        **mean,
    }
    cv_path = out_dir / f"{name}__cv.json"
    cv_path.write_text(json.dumps(cv, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\n[train2x2] saved:")
    for p in (model_path, feats_path, imp_path, cv_path):
        print(f"    {p}")
    return importances


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gt", required=True,
                    help="GT JSON in {qtext: {mid: label}} shape (WP1 gt_2x2.json)")
    ap.add_argument("--store", required=True, help="store data dir (frozen or c2)")
    ap.add_argument("--name", required=True, help="artifact stem (R_pre | R_post | arbitrary)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--feature-mask", default=None,
                    help="optional JSON list (inline '[...]' or file path) of feature "
                         "names; default SCHEMA_24 (WP6 hook)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED,
                    help="fixed seed (default 42; propagated to LightGBM)")
    ap.add_argument("--locomo-data", default=str(DEFAULT_LOCOMO))
    ap.add_argument("--key-file", default=str(DEFAULT_KEY_FILE))
    ap.add_argument("--max-queries", type=int, default=None,
                    help="smoke only: cap training queries (truncates the GT copy)")
    args = ap.parse_args()

    # Propagate a non-default seed into the shared param dict.
    if args.seed != SEED:
        LGB_PARAMS["random_state"] = args.seed

    store = Path(args.store).expanduser().resolve()
    gt_path = Path(args.gt).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    key_file = Path(args.key_file).expanduser()
    mask = _resolve_mask(args.feature_mask)

    scratch = Path(
        r"C:\Users\Alexis\AppData\Local\Temp\claude"
        r"\C--Users-Alexis-repos-somnigraph"
        r"\d2cafcdd-0ff6-4bb5-a74a-acd677d71759\scratchpad"
    )
    gt_used = _maybe_truncate_gt(gt_path, args.max_queries, scratch)

    print(f"[train2x2] store={store}")
    print(f"[train2x2] name={args.name}  features={len(mask)}  folds={args.folds}  seed={args.seed}")
    print(f"[train2x2] mask={mask}")

    tune_gt = _point_and_reload(store, key_file)
    data = _extract_matrix(tune_gt, gt_used, mask)

    fold_metrics, mean, final = train_and_cv(tune_gt, data, args.folds)

    config = {
        "store": str(store),
        "gt": str(gt_path),
        "gt_used": str(gt_used),
        "max_queries": args.max_queries,
        "seed": args.seed,
        "folds": args.folds,
        "lgb_params": LGB_PARAMS,
        "schema_source": "SCHEMA_24" if args.feature_mask is None else str(args.feature_mask),
    }
    importances = _save_artifacts(out_dir, args.name, final, mask,
                                  fold_metrics, mean, data, config)

    # Sleep-feature importance check (structural on frozen).
    sleep_feats = [f for f in ("is_summary", "is_consolidation_node",
                               "consolidated_member", "member_count") if f in mask]
    print("\n[train2x2] sleep-feature gain importances:")
    for f in sleep_feats:
        print(f"    {f:22s} gain={importances.get(f, 0.0):.4f}")
    all_zero = all(importances.get(f, 0.0) == 0.0 for f in sleep_feats)
    print(f"[train2x2] sleep features all-zero-gain: {all_zero} "
          f"(expected True on frozen, may be False on c2)")


if __name__ == "__main__":
    main()
