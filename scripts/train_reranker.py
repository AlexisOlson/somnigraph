# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "sqlite-vec>=0.1.6",
#     "openai>=2.0.0",
#     "numpy>=1.26",
#     "scikit-learn>=1.0",
#     "lightgbm>=4.0",
# ]
# ///
"""
Train a learned re-ranker for memory retrieval.

Extracts per-(query, candidate) features from the existing retrieval pipeline,
trains a LightGBM pointwise regressor on GT relevance labels, and evaluates
via 5-fold cross-validation grouped by query.

Usage:
  uv run train_reranker.py                    # Extract features, train, evaluate
  uv run train_reranker.py --extract-only     # Just build feature matrix
  uv run train_reranker.py --train-only       # Train from saved features
  uv run train_reranker.py --compare          # Compare reranker vs production
"""

import argparse
import json
import math
import pickle
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold

# Suppress sklearn feature-name warnings — we use numpy arrays throughout,
# but LightGBM's eval_set validation warns about missing feature names.
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from tune_gt import (
    PRODUCTION_PARAMS,
    load_tuning_data,
    score_and_rank,
    evaluate_trial,
    compute_ndcg,
    compute_graded_recall,
    compute_recall_at_k,
    get_db,
)

from memory.constants import DATA_DIR
FEATURES_PATH = DATA_DIR / "tuning_studies" / "reranker_features.pkl"
MODEL_PATH = DATA_DIR / "tuning_studies" / "reranker_model.pkl"
GT_PATH = DATA_DIR / "tuning_studies" / "gt_calibrated.json"

# Category encoding
CATEGORY_MAP = {
    "episodic": 0, "semantic": 1, "procedural": 2,
    "reflection": 3, "entity": 4, "meta": 5,
}

# Feature names (order matters — matches extraction)
# No derived scores (fts_score, vec_score, etc.) — the model learns its own
# transform of raw ranks. No binary in_fts/in_vec — redundant with rank != -1.
FEATURE_NAMES = [
    # Retrieval signals (query-dependent)
    "fts_rank",       # 0: FTS rank position (-1 if not retrieved)
    "vec_rank",       # 1: Vec rank position (-1 if not retrieved)
    "theme_rank",     # 2: Theme rank position (-1 if not retrieved)
    "ppr_score",      # 3: PPR expansion score (raw, from graph walk)
    "fts_bm25",       # 4: raw BM25 score from FTS (0 if not retrieved)
    "vec_dist",       # 5: raw cosine distance from vec search (0 if not retrieved)
    "theme_overlap",  # 6: raw token overlap count from theme channel (0 if none)

    # Feedback signals (query-independent, raw — no ewma_alpha dependency)
    "fb_last",        # 7: most recent utility score (-1 if no feedback)
    "fb_mean",        # 8: mean of all utility scores (-1 if no feedback)
    "fb_count",       # 9: feedback event count

    # Graph signals
    "hebbian_pmi",    # 10: PMI co-retrieval score

    # Memory metadata
    "category",       # 11: ordinal encoded
    "priority",       # 12: base_priority (1-10)
    "age_days",       # 13: days since created_at
    "token_count",    # 14: token count
    "edge_count",     # 15: number of edges
    "theme_count",    # 16: number of themes
    "confidence",     # 17: confidence score
]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def load_memory_metadata(db) -> dict:
    """Load per-memory metadata from DB for feature extraction."""
    rows = db.execute("""
        SELECT id, category, base_priority, created_at, token_count,
               confidence, themes
        FROM memories WHERE status = 'active'
    """).fetchall()

    meta = {}
    now = time.time()
    for r in rows:
        mid = r["id"]
        # Age in days
        try:
            from datetime import datetime, timezone
            created = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - created).total_seconds() / 86400
        except Exception:
            age_days = 0.0

        # Theme count
        theme_count = 0
        if r["themes"]:
            try:
                theme_count = len(json.loads(r["themes"]))
            except Exception:
                pass

        meta[mid] = {
            "category": CATEGORY_MAP.get(r["category"], 1),
            "priority": r["base_priority"] or 5,
            "age_days": age_days,
            "token_count": r["token_count"] or 200,
            "confidence": r["confidence"] if r["confidence"] is not None else 0.5,
            "theme_count": theme_count,
        }

    # Edge counts
    edge_rows = db.execute("""
        SELECT source_id, target_id FROM memory_edges
    """).fetchall()
    edge_counts = defaultdict(int)
    for r in edge_rows:
        edge_counts[r["source_id"]] += 1
        edge_counts[r["target_id"]] += 1
    for mid in meta:
        meta[mid]["edge_count"] = edge_counts.get(mid, 0)

    return meta


def extract_features_for_query(
    qtext: str,
    params: dict,
    search_data: dict,
    feedback_raw: dict,
    hebb_data: dict,
    memory_meta: dict,
    ground_truth_for_query: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract feature matrix and labels for one query.

    Only uses params for: ppr_damping (PPR cache key — a retrieval constant).
    All scoring weights are learned by the model, not baked into features.

    Returns (features, labels, memory_ids) where:
      features: (n_candidates, n_features) array
      labels: (n_candidates,) array of GT relevance
      memory_ids: list of memory IDs (for debugging)
    """
    ppr_damping = params.get("ppr_damping", 0.216)

    # --- Retrieval results ---
    fts_ids = search_data["fts_results"].get(qtext, [])
    vec_ids = search_data["vec_results"].get(qtext, [])
    themes_map = search_data["themes_map"]

    fts_ranked = {mid: rank for rank, mid in enumerate(fts_ids)}
    vec_ranked = {mid: rank for rank, mid in enumerate(vec_ids)}

    # Raw scores from precomputed searches
    fts_score_map = search_data.get("fts_scores", {}).get(qtext, {})
    vec_score_map = search_data.get("vec_scores", {}).get(qtext, {})

    # Theme channel
    query_tokens = set(qtext.lower().split())
    scored = []
    theme_overlap_map = {}
    for mid, mem_themes in themes_map.items():
        if not mem_themes:
            continue
        overlap = len(mem_themes & query_tokens)
        if overlap > 0:
            scored.append((mid, overlap))
            theme_overlap_map[mid] = overlap
    scored.sort(key=lambda x: (-x[1], x[0]))
    theme_ranked = {mid: rank for rank, (mid, _) in enumerate(scored)}

    # --- Candidate pool (all memories seen by any channel + PPR) ---
    candidate_ids = set(fts_ranked) | set(vec_ranked) | set(theme_ranked)

    # PPR scores (raw, from graph walk — no boost coefficient)
    ppr_scores = {}
    ppr_cache = search_data.get("ppr_cache")
    if ppr_cache is not None:
        d_key = round(ppr_damping, 3)
        ppr_scores = ppr_cache.get((d_key, qtext), {})
        candidate_ids |= set(ppr_scores.keys())

    # --- Hebbian PMI (raw PMI, no scaling — model learns the weight) ---
    # Seeds = top-5 by best available rank (min of fts/vec rank, lower is better)
    hebbian_pmi_map = {}
    if hebb_data and hebb_data["total_queries"] >= 5:
        hebb_mem_freq = hebb_data["mem_freq"]
        hebb_total = hebb_data["total_queries"]
        hebb_mem_count = {mid: len(qs) for mid, qs in hebb_mem_freq.items()}

        def _best_rank(mid):
            ranks = []
            if mid in fts_ranked:
                ranks.append(fts_ranked[mid])
            if mid in vec_ranked:
                ranks.append(vec_ranked[mid])
            return min(ranks) if ranks else 9999

        seed_ids = sorted(candidate_ids, key=_best_rank)[:5]

        for candidate in candidate_ids:
            if candidate in seed_ids:
                continue
            total_pmi = 0.0
            for seed in seed_ids:
                if seed not in hebb_mem_count or candidate not in hebb_mem_count:
                    continue
                joint = len(hebb_mem_freq.get(seed, set()) &
                           hebb_mem_freq.get(candidate, set()))
                if joint < 2:
                    continue
                p_s = hebb_mem_count[seed] / hebb_total
                p_c = hebb_mem_count[candidate] / hebb_total
                p_j = joint / hebb_total
                if p_s * p_c == 0:
                    continue
                pmi = math.log2(p_j / (p_s * p_c))
                if pmi > 0:
                    total_pmi += pmi
            if total_pmi > 0:
                hebbian_pmi_map[candidate] = total_pmi

    # --- Build feature matrix ---
    n_features = len(FEATURE_NAMES)
    candidate_list = sorted(candidate_ids)
    n_candidates = len(candidate_list)

    features = np.zeros((n_candidates, n_features), dtype=np.float32)
    labels = np.zeros(n_candidates, dtype=np.float32)

    for i, mid in enumerate(candidate_list):
        # Retrieval signals
        fts_r = fts_ranked.get(mid, -1)
        vec_r = vec_ranked.get(mid, -1)
        theme_r = theme_ranked.get(mid, -1)

        features[i, 0] = fts_r                                              # fts_rank
        features[i, 1] = vec_r                                              # vec_rank
        features[i, 2] = theme_r                                            # theme_rank
        features[i, 3] = ppr_scores.get(mid, 0.0)                           # ppr_score
        features[i, 4] = fts_score_map.get(mid, 0.0)                        # fts_bm25
        features[i, 5] = vec_score_map.get(mid, 0.0)                        # vec_dist
        features[i, 6] = theme_overlap_map.get(mid, 0)                      # theme_overlap

        # Feedback signals (raw — no ewma_alpha or Beta prior dependency)
        fb = feedback_raw.get(mid)
        if fb and fb["count"] > 0:
            features[i, 7] = fb["utilities"][-1]                              # fb_last
            features[i, 8] = sum(fb["utilities"]) / fb["count"]               # fb_mean
            features[i, 9] = fb["count"]                                      # fb_count
        else:
            features[i, 7] = -1.0                                             # fb_last (sentinel)
            features[i, 8] = -1.0                                             # fb_mean (sentinel)
            features[i, 9] = 0                                                # fb_count

        # Graph signals
        features[i, 10] = hebbian_pmi_map.get(mid, 0.0)                     # hebbian_pmi

        # Memory metadata
        meta = memory_meta.get(mid)
        if meta:
            features[i, 11] = meta["category"]                               # category
            features[i, 12] = meta["priority"]                                # priority
            features[i, 13] = meta["age_days"]                                # age_days
            features[i, 14] = meta["token_count"]                             # token_count
            features[i, 15] = meta["edge_count"]                              # edge_count
            features[i, 16] = meta["theme_count"]                             # theme_count
            features[i, 17] = meta["confidence"]                              # confidence
        else:
            features[i, 11] = 1   # semantic default
            features[i, 12] = 5
            features[i, 13] = 0.0
            features[i, 14] = 200
            features[i, 15] = 0
            features[i, 16] = 0
            features[i, 17] = 0.5

        # Label
        labels[i] = ground_truth_for_query.get(mid, 0.0)

    return features, labels, candidate_list


def extract_all_features(full_data: dict, ground_truth: dict,
                         memory_meta: dict, gt_only: bool = False,
                         neg_ratio: float = 0,
                         neg_strategy: str = "random",
                         neg_top_k: int = 50) -> dict:
    """Extract features for all GT queries.

    Args:
      gt_only: If True, filter candidates per query.
      neg_ratio: Negatives per positive (e.g., 2.0). 0 = no filtering.
      neg_strategy: "random", "topk" (keep top-K ranked per channel),
                    or "hard" (MSE-scored false positives).
      neg_top_k: For "topk" strategy, keep non-GT candidates ranked
                 in top-K by any channel.

    Returns dict with keys:
      features: (N, n_features) array
      labels: (N,) array
      query_ids: (N,) array of query indices (for GroupKFold)
      memory_ids: list of (query, mid) pairs
      feature_names: list of feature names
    """
    params = dict(PRODUCTION_PARAMS)
    search_data = full_data["search_data"]
    feedback_raw = full_data["feedback_raw"]
    hebb_data = full_data["hebb_data"]

    rng = np.random.RandomState(42)

    all_features = []
    all_labels = []
    all_query_ids = []
    all_mids = []

    # For "hard" strategy: train MSE on full data first, then mine false positives
    mse_model_for_mining = None
    if gt_only and neg_strategy == "hard":
        print("  Training MSE model for hard negative mining...")
        # Quick extraction of full features for MSE training
        full_feats = extract_all_features(full_data, ground_truth, memory_meta)
        mse_params = {
            "objective": "regression", "metric": "rmse",
            "num_leaves": 31, "learning_rate": 0.1, "n_estimators": 300,
            "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8,
            "verbose": -1, "random_state": 42,
        }
        mse_model_for_mining = lgb.LGBMRegressor(**mse_params)
        mse_model_for_mining.fit(full_feats["features"], full_feats["labels"])
        print("  MSE model ready for mining.")

    queries = full_data["gt_queries"]
    for qi, qtext in enumerate(queries):
        gt_for_q = ground_truth.get(qtext, {})
        if not gt_for_q:
            continue

        feats, labs, mids = extract_features_for_query(
            qtext, params, search_data, feedback_raw, hebb_data,
            memory_meta, gt_for_q,
        )

        if gt_only:
            gt_set = set(gt_for_q.keys())
            gt_indices = [i for i, mid in enumerate(mids) if mid in gt_set]
            neg_indices = [i for i, mid in enumerate(mids) if mid not in gt_set]

            if not gt_indices:
                continue

            keep = list(gt_indices)

            if neg_indices:
                if neg_strategy == "topk":
                    # Keep negatives that were ranked in top-K by any channel
                    fts_ids = search_data["fts_results"].get(qtext, [])
                    vec_ids = search_data["vec_results"].get(qtext, [])
                    top_fts = set(fts_ids[:neg_top_k])
                    top_vec = set(vec_ids[:neg_top_k])
                    top_set = top_fts | top_vec
                    hard_neg = [i for i in neg_indices if mids[i] in top_set]
                    # If we have more than neg_ratio * gt_count, subsample
                    n_want = int(len(gt_indices) * neg_ratio) if neg_ratio > 0 else len(hard_neg)
                    if len(hard_neg) > n_want:
                        hard_neg = list(rng.choice(hard_neg, size=n_want, replace=False))
                    keep.extend(hard_neg)

                elif neg_strategy == "hard" and mse_model_for_mining is not None:
                    # Score all negatives with MSE, take highest-scored
                    neg_feats = feats[neg_indices]
                    neg_scores = mse_model_for_mining.predict(neg_feats)
                    n_want = int(len(gt_indices) * neg_ratio) if neg_ratio > 0 else len(neg_indices)
                    n_want = min(n_want, len(neg_indices))
                    # Take top-scored negatives (hardest false positives)
                    top_neg_idx = np.argsort(-neg_scores)[:n_want]
                    keep.extend([neg_indices[i] for i in top_neg_idx])

                elif neg_ratio > 0:  # random
                    n_neg = min(int(len(gt_indices) * neg_ratio), len(neg_indices))
                    sampled_neg = rng.choice(neg_indices, size=n_neg, replace=False)
                    keep.extend(sampled_neg)

                keep.sort()

            feats = feats[keep]
            labs = labs[keep]
            mids = [mids[i] for i in keep]
            labs[labs < 0.1] = 0.0

        all_features.append(feats)
        all_labels.append(labs)
        all_query_ids.extend([qi] * len(mids))
        all_mids.extend([(qtext, mid) for mid in mids])

        if (qi + 1) % 50 == 0:
            n_pos = sum(1 for l in labs if l > 0)
            print(f"  {qi + 1}/{len(queries)} queries, "
                  f"{len(mids)} candidates, {n_pos} positive")

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    query_ids = np.array(all_query_ids, dtype=np.int32)

    n_pos = np.sum(labels > 0)
    print(f"\nFeature extraction complete:")
    print(f"  Total samples: {len(labels)}")
    print(f"  Positive (GT > 0): {n_pos} ({100*n_pos/len(labels):.1f}%)")
    print(f"  Queries: {len(set(query_ids))}")
    print(f"  Features: {features.shape[1]}")

    return {
        "features": features,
        "labels": labels,
        "query_ids": query_ids,
        "memory_ids": all_mids,
        "feature_names": FEATURE_NAMES,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_and_evaluate(feature_data: dict, n_folds: int = 5,
                       n_estimators: int = 500) -> dict:
    """Train LightGBM with GroupKFold CV. Returns results dict."""
    X = feature_data["features"]
    y = feature_data["labels"]
    groups = feature_data["query_ids"]

    print(f"\n{'='*70}")
    print("TRAINING: LightGBM Pointwise Regressor")
    print(f"{'='*70}")
    print(f"  Samples: {len(y)}, Features: {X.shape[1]}, Folds: {n_folds}")
    print(f"  n_estimators: {n_estimators}")

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "n_estimators": n_estimators,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "random_state": 42,
    }

    gkf = GroupKFold(n_splits=n_folds)
    fold_results = []
    oof_predictions = np.zeros(len(y))
    importances = np.zeros(X.shape[1])

    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        preds = model.predict(X_val)
        preds = np.clip(preds, 0, 1)
        oof_predictions[val_idx] = preds
        importances += model.feature_importances_

        rmse = np.sqrt(np.mean((preds - y_val) ** 2))
        fold_results.append({"fold": fold_i, "rmse": rmse, "n_val": len(val_idx),
                             "best_iter": model.best_iteration_})

        n_val_pos = np.sum(y_val > 0)
        print(f"  Fold {fold_i}: RMSE={rmse:.4f} "
              f"best_iter={model.best_iteration_} "
              f"val_samples={len(val_idx)} val_pos={n_val_pos}")

    importances /= n_folds

    mean_rmse = np.mean([r["rmse"] for r in fold_results])
    print(f"\n  Mean RMSE: {mean_rmse:.4f}")

    # Feature importance
    print(f"\n  Feature importance (gain):")
    imp_order = np.argsort(importances)[::-1]
    for idx in imp_order:
        print(f"    {FEATURE_NAMES[idx]:20s}: {importances[idx]:8.1f}")

    # Train final model on all data
    print(f"\nTraining final model on all data...")
    final_model = lgb.LGBMRegressor(**lgb_params)
    final_model.fit(X, y)

    return {
        "fold_results": fold_results,
        "oof_predictions": oof_predictions,
        "importances": importances,
        "final_model": final_model,
        "mean_rmse": mean_rmse,
    }


def discretize_labels(y, n_levels: int = 10):
    """Convert continuous GT (0-1) to integer relevance levels.

    When n_levels >= 100, uses direct scaling: round(score * n_levels).
    This preserves nearly all granularity of continuous scores.

    When n_levels < 100, uses quantile-based binning on nonzero scores
    to create balanced classes, with 0 reserved for irrelevant (score == 0).
    """
    result = np.zeros(len(y), dtype=np.int32)
    nonzero_mask = y > 0
    if nonzero_mask.sum() == 0:
        return result

    if n_levels >= 100:
        # Direct scaling — preserves granularity, no quantile compression
        result[nonzero_mask] = np.clip(
            np.round(y[nonzero_mask] * n_levels).astype(np.int32),
            1, n_levels  # floor at 1 so positives never map to 0 (irrelevant)
        )
    else:
        # Quantile bins on nonzero scores
        nonzero_scores = y[nonzero_mask]
        # n_levels - 1 quantile bins for positives, level 0 = irrelevant
        quantiles = np.linspace(0, 100, n_levels)
        bin_edges = np.percentile(nonzero_scores, quantiles)
        # Deduplicate edges (ties in scores)
        bin_edges = np.unique(bin_edges)
        # digitize: level 1..n for positives
        result[nonzero_mask] = np.digitize(nonzero_scores, bin_edges[:-1])

    return result


def build_groups(query_ids):
    """Build group array (number of candidates per query) from sorted query_ids."""
    groups = []
    current = query_ids[0]
    count = 0
    for qi in query_ids:
        if qi == current:
            count += 1
        else:
            groups.append(count)
            current = qi
            count = 1
    groups.append(count)
    return groups


def compute_label_gains(y, y_discrete):
    """Compute label_gain mapping: for each integer level, the mean GT score.

    This lets LambdaRank/rank_xendcg weight swaps by actual relevance
    difference rather than treating all adjacent levels equally.
    """
    gains = []
    for level in range(y_discrete.max() + 1):
        mask = y_discrete == level
        if mask.sum() > 0:
            gains.append(float(np.mean(y[mask])))
        else:
            gains.append(0.0)
    return gains


def train_lambdarank(feature_data: dict, n_folds: int = 5,
                     objective: str = "lambdarank",
                     n_levels: int = 10) -> dict:
    """Train LightGBM ranker with GroupKFold CV.

    Args:
      objective: "lambdarank" or "rank_xendcg"
      n_levels: Number of quantile bins for discretization
    """
    X = feature_data["features"]
    y = feature_data["labels"]
    query_ids = feature_data["query_ids"]

    y_train_labels = discretize_labels(y, n_levels=n_levels)
    label_gains = compute_label_gains(y, y_train_labels)
    n_unique = len(set(y_train_labels))
    label_dist = {i: int(np.sum(y_train_labels == i))
                  for i in sorted(set(y_train_labels))}
    label_desc = f"{n_unique} levels: {label_dist}"

    print(f"\n{'='*70}")
    print(f"TRAINING: LightGBM {objective}")
    print(f"{'='*70}")
    print(f"  Samples: {len(y)}, Features: {X.shape[1]}, Folds: {n_folds}")
    print(f"  Labels: {label_desc}")

    lgb_params = {
        "objective": objective,
        "metric": "ndcg",
        "eval_at": [5, 10],
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_samples": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "label_gain": ",".join(f"{g:.4f}" for g in label_gains),
        "verbose": -1,
        "random_state": 42,
    }
    if objective == "lambdarank":
        lgb_params["lambdarank_truncation_level"] = 15
        lgb_params["lambdarank_norm"] = True

    print(f"  Label gains: {[f'{g:.3f}' for g in label_gains]}")

    gkf = GroupKFold(n_splits=n_folds)
    fold_results = []
    importances = np.zeros(X.shape[1])

    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X, y, query_ids)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_tr = y_train_labels[train_idx]
        y_va = y_train_labels[val_idx]
        train_qids = query_ids[train_idx]
        val_qids = query_ids[val_idx]

        train_groups = build_groups(train_qids)
        val_groups = build_groups(val_qids)

        model = lgb.LGBMRanker(**lgb_params)
        model.fit(
            X_train, y_tr,
            group=train_groups,
            eval_set=[(X_val, y_va)],
            eval_group=[val_groups],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        importances += model.feature_importances_

        fold_results.append({"fold": fold_i, "n_val": len(val_idx),
                             "best_iter": model.best_iteration_})

        n_val_pos = np.sum(y_va > 0)
        print(f"  Fold {fold_i}: best_iter={model.best_iteration_} "
              f"val_samples={len(val_idx)} val_pos={n_val_pos}")

    importances /= n_folds

    print(f"\n  Feature importance (gain):")
    imp_order = np.argsort(importances)[::-1]
    for idx in imp_order:
        print(f"    {FEATURE_NAMES[idx]:20s}: {importances[idx]:8.1f}")

    # Train final model on all data
    print(f"\nTraining final model on all data...")
    all_groups = build_groups(query_ids)
    final_model = lgb.LGBMRanker(**lgb_params)
    final_model.fit(X, y_train_labels, group=all_groups)

    return {
        "fold_results": fold_results,
        "importances": importances,
        "final_model": final_model,
    }


# ---------------------------------------------------------------------------
# Ranking evaluation (NDCG/Recall using reranker predictions)
# ---------------------------------------------------------------------------


def evaluate_reranker_ranking(feature_data: dict, full_data: dict,
                              ground_truth: dict, n_folds: int = 5,
                              use_lambdarank: bool = False,
                              train_feature_data: dict = None,
                              lr_objective: str = "lambdarank",
                              lr_n_levels: int = 10) -> dict:
    """Evaluate reranker ranking quality via CV, comparing to production.

    For each fold: train on 4 folds, predict on held-out fold, rank candidates
    by predicted score, compute NDCG and Recall at 5k token budget.

    When use_lambdarank=True and train_feature_data is provided, training uses
    the GT-only feature set but evaluation predicts on the full candidate pool.
    """
    # Full candidate pool for evaluation
    X_full = feature_data["features"]
    y_full = feature_data["labels"]
    groups_full = feature_data["query_ids"]
    memory_ids = feature_data["memory_ids"]  # list of (query, mid)
    token_map = full_data["token_map"]

    queries = full_data["gt_queries"]
    params = dict(PRODUCTION_PARAMS)

    # Training data (may be GT-only subset for LambdaRank)
    if use_lambdarank and train_feature_data is not None:
        X_tr = train_feature_data["features"]
        y_tr = train_feature_data["labels"]
        groups_tr = train_feature_data["query_ids"]
    else:
        X_tr = X_full
        y_tr = y_full
        groups_tr = groups_full

    if use_lambdarank:
        y_tr_labels = discretize_labels(y_tr, n_levels=lr_n_levels)
        label_gains = compute_label_gains(y_tr, y_tr_labels)

        lgb_params = {
            "objective": lr_objective,
            "metric": "ndcg",
            "eval_at": [5, 10],
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "min_child_samples": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "label_gain": ",".join(f"{g:.4f}" for g in label_gains),
            "verbose": -1,
            "random_state": 42,
        }
        if lr_objective == "lambdarank":
            lgb_params["lambdarank_truncation_level"] = 15
            lgb_params["lambdarank_norm"] = True
    else:
        lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 31,
            "learning_rate": 0.1,
            "n_estimators": 500,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
            "random_state": 42,
        }

    # Split by query — use training data's query IDs for fold assignment
    gkf = GroupKFold(n_splits=n_folds)

    label = "LambdaRank (GT-only train)" if use_lambdarank else "Reranker"
    print(f"\n{'='*70}")
    print(f"RANKING EVALUATION: {label} vs Production RRF")
    print(f"{'='*70}")

    all_fold_metrics = []

    # Build query-to-fold mapping from training splits
    for fold_i, (train_idx_tr, val_idx_tr) in enumerate(gkf.split(X_tr, y_tr, groups_tr)):
        val_query_ids_set = set(groups_tr[val_idx_tr])

        if use_lambdarank:
            # Train on GT-only training fold
            X_train = X_tr[train_idx_tr]
            y_train_lr = y_tr_labels[train_idx_tr]
            train_qids = groups_tr[train_idx_tr]
            train_groups = build_groups(train_qids)

            # Validation on GT-only val fold (for early stopping)
            X_val_tr = X_tr[val_idx_tr]
            y_val_lr = y_tr_labels[val_idx_tr]
            val_qids = groups_tr[val_idx_tr]
            val_groups = build_groups(val_qids)

            model = lgb.LGBMRanker(**lgb_params)
            model.fit(
                X_train, y_train_lr,
                group=train_groups,
                eval_set=[(X_val_tr, y_val_lr)],
                eval_group=[val_groups],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        else:
            X_train = X_tr[train_idx_tr]
            y_train = y_tr[train_idx_tr]
            X_val_tr = X_tr[val_idx_tr]
            y_val = y_tr[val_idx_tr]

            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val_tr, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )

        # Predict on FULL candidate pool for val queries
        val_mask_full = np.isin(groups_full, list(val_query_ids_set))
        val_idx_full = np.where(val_mask_full)[0]
        X_val_full = X_full[val_idx_full]
        val_groups_full = groups_full[val_idx_full]

        preds = model.predict(X_val_full)
        if not use_lambdarank:
            preds = np.clip(preds, 0, 1)

        # Group predictions by query
        reranker_ranked_results = {}
        production_ranked_results = {}

        for qi in sorted(val_query_ids_set):
            qtext = queries[qi]
            mask = val_groups_full == qi
            q_preds = preds[mask]
            q_mids = [memory_ids[idx][1] for idx in val_idx_full[mask]]

            # Reranker ranking: sort by predicted score
            order = np.argsort(-q_preds)
            reranker_ranked_results[qtext] = [q_mids[i] for i in order]

            # Production ranking
            ranked = score_and_rank(qtext, params, full_data["search_data"],
                                    full_data["feedback_raw"], full_data["hebb_data"])
            production_ranked_results[qtext] = ranked

        # Compute metrics
        reranker_ndcg = compute_ndcg(reranker_ranked_results, token_map,
                                     ground_truth, budget=5000)
        reranker_recall = compute_graded_recall(reranker_ranked_results, token_map,
                                                ground_truth, budget=5000)

        prod_ndcg = compute_ndcg(production_ranked_results, token_map,
                                 ground_truth, budget=5000)
        prod_recall = compute_graded_recall(production_ranked_results, token_map,
                                            ground_truth, budget=5000)

        fold_metrics = {
            "fold": fold_i,
            "n_queries": len(val_query_ids_set),
            "reranker_ndcg": reranker_ndcg,
            "reranker_recall": reranker_recall,
            "production_ndcg": prod_ndcg,
            "production_recall": prod_recall,
        }
        all_fold_metrics.append(fold_metrics)

        print(f"  Fold {fold_i} ({len(val_query_ids_set)} queries):")
        print(f"    Reranker:   NDCG={reranker_ndcg:.4f}  Recall={reranker_recall:.4f}")
        print(f"    Production: NDCG={prod_ndcg:.4f}  Recall={prod_recall:.4f}")
        print(f"    Delta:      NDCG={reranker_ndcg-prod_ndcg:+.4f}  "
              f"Recall={reranker_recall-prod_recall:+.4f}")

    # Summary
    mean_reranker_ndcg = np.mean([m["reranker_ndcg"] for m in all_fold_metrics])
    mean_reranker_recall = np.mean([m["reranker_recall"] for m in all_fold_metrics])
    mean_prod_ndcg = np.mean([m["production_ndcg"] for m in all_fold_metrics])
    mean_prod_recall = np.mean([m["production_recall"] for m in all_fold_metrics])

    print(f"\n  {'='*50}")
    print(f"  MEAN ACROSS {n_folds} FOLDS:")
    print(f"  {'='*50}")
    print(f"  Reranker:   NDCG={mean_reranker_ndcg:.4f}  Recall={mean_reranker_recall:.4f}")
    print(f"  Production: NDCG={mean_prod_ndcg:.4f}  Recall={mean_prod_recall:.4f}")
    print(f"  Delta:      NDCG={mean_reranker_ndcg-mean_prod_ndcg:+.4f}  "
          f"Recall={mean_reranker_recall-mean_prod_recall:+.4f}")

    return {
        "fold_metrics": all_fold_metrics,
        "mean_reranker_ndcg": mean_reranker_ndcg,
        "mean_reranker_recall": mean_reranker_recall,
        "mean_prod_ndcg": mean_prod_ndcg,
        "mean_prod_recall": mean_prod_recall,
    }


# ---------------------------------------------------------------------------
# Per-query comparison
# ---------------------------------------------------------------------------


def per_query_comparison(feature_data: dict, full_data: dict,
                         ground_truth: dict) -> None:
    """Train on all data, compare per-query rankings, analyze regressions."""
    X = feature_data["features"]
    y = feature_data["labels"]
    groups = feature_data["query_ids"]
    memory_ids = feature_data["memory_ids"]
    token_map = full_data["token_map"]
    queries = full_data["gt_queries"]
    params = dict(PRODUCTION_PARAMS)

    print(f"\n{'='*70}")
    print("PER-QUERY ANALYSIS (trained on all data — for diagnostics only)")
    print(f"{'='*70}")

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "n_estimators": 500,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "random_state": 42,
    }
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X, y)
    preds = np.clip(model.predict(X), 0, 1)

    # Per-query NDCG comparison
    deltas = []
    for qi in sorted(set(groups)):
        qtext = queries[qi]
        gt = ground_truth.get(qtext, {})
        if not gt:
            continue

        mask = groups == qi
        q_preds = preds[mask]
        q_mids = [memory_ids[idx][1] for idx in np.where(mask)[0]]

        # Reranker
        order = np.argsort(-q_preds)
        reranker_ranked = {qtext: [q_mids[i] for i in order]}

        # Production
        prod_ranked = {qtext: score_and_rank(qtext, params, full_data["search_data"],
                                             full_data["feedback_raw"],
                                             full_data["hebb_data"])}

        r_ndcg = compute_ndcg(reranker_ranked, token_map, ground_truth, budget=5000)
        p_ndcg = compute_ndcg(prod_ranked, token_map, ground_truth, budget=5000)
        deltas.append({"query": qtext[:60], "reranker": r_ndcg, "production": p_ndcg,
                       "delta": r_ndcg - p_ndcg, "n_gt": len(gt)})

    deltas.sort(key=lambda x: x["delta"])

    improved = sum(1 for d in deltas if d["delta"] > 0.001)
    regressed = sum(1 for d in deltas if d["delta"] < -0.001)
    unchanged = len(deltas) - improved - regressed

    print(f"  Improved: {improved}, Regressed: {regressed}, Unchanged: {unchanged}")

    if regressed > 0:
        print(f"\n  Worst regressions:")
        for d in deltas[:min(10, regressed)]:
            print(f"    {d['delta']:+.4f}  R={d['reranker']:.4f} P={d['production']:.4f}  "
                  f"GT={d['n_gt']}  {d['query']}")

    if improved > 0:
        print(f"\n  Best improvements:")
        for d in deltas[-min(10, improved):][::-1]:
            print(f"    {d['delta']:+.4f}  R={d['reranker']:.4f} P={d['production']:.4f}  "
                  f"GT={d['n_gt']}  {d['query']}")


# ---------------------------------------------------------------------------
# Ablation: drop feedback features
# ---------------------------------------------------------------------------


def ablation_no_feedback(feature_data: dict, n_folds: int = 5) -> None:
    """Train without feedback features to quantify their contribution."""
    X = feature_data["features"]
    y = feature_data["labels"]
    groups = feature_data["query_ids"]

    # Feedback features: indices 7-9
    feedback_indices = [7, 8, 9]
    keep_indices = [i for i in range(X.shape[1]) if i not in feedback_indices]
    X_no_fb = X[:, keep_indices]

    print(f"\n{'='*70}")
    print(f"ABLATION: No Feedback Features ({X_no_fb.shape[1]} features)")
    print(f"{'='*70}")

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "n_estimators": 500,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "random_state": 42,
    }

    gkf = GroupKFold(n_splits=n_folds)
    rmses = []

    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X_no_fb, y, groups)):
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_no_fb[train_idx], y[train_idx],
            eval_set=[(X_no_fb[val_idx], y[val_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        preds = np.clip(model.predict(X_no_fb[val_idx]), 0, 1)
        rmse = np.sqrt(np.mean((preds - y[val_idx]) ** 2))
        rmses.append(rmse)
        print(f"  Fold {fold_i}: RMSE={rmse:.4f}")

    print(f"  Mean RMSE (no feedback): {np.mean(rmses):.4f}")


def evaluate_two_stage(feature_data: dict, full_data: dict,
                       ground_truth: dict, n_folds: int = 5,
                       n_levels: int = 20) -> dict:
    """Two-stage: MSE predicts scores → LambdaRank reranks with MSE score as extra feature."""
    X = feature_data["features"]
    y = feature_data["labels"]
    groups = feature_data["query_ids"]
    memory_ids = feature_data["memory_ids"]
    token_map = full_data["token_map"]
    queries = full_data["gt_queries"]
    params = dict(PRODUCTION_PARAMS)

    mse_params = {
        "objective": "regression", "metric": "rmse",
        "num_leaves": 31, "learning_rate": 0.1, "n_estimators": 500,
        "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8,
        "verbose": -1, "random_state": 42,
    }

    y_discrete = discretize_labels(y, n_levels=n_levels)
    label_gains = compute_label_gains(y, y_discrete)

    lr_params = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [5, 10],
        "num_leaves": 31, "learning_rate": 0.05, "n_estimators": 500,
        "min_child_samples": 5, "subsample": 0.8, "colsample_bytree": 0.8,
        "lambdarank_truncation_level": 15, "lambdarank_norm": True,
        "label_gain": ",".join(f"{g:.4f}" for g in label_gains),
        "verbose": -1, "random_state": 42,
    }

    gkf = GroupKFold(n_splits=n_folds)

    print(f"\n{'='*70}")
    print("RANKING EVALUATION: Two-Stage (MSE -> LambdaRank) vs Production RRF")
    print(f"{'='*70}")

    all_fold_metrics = []

    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        val_groups = groups[val_idx]

        # Stage 1: Train MSE, get predictions for train+val
        mse_model = lgb.LGBMRegressor(**mse_params)
        mse_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        mse_preds_train = np.clip(mse_model.predict(X_train), 0, 1)
        mse_preds_val = np.clip(mse_model.predict(X_val), 0, 1)

        # Stage 2: Augment features with MSE predictions
        X_train_aug = np.column_stack([X_train, mse_preds_train])
        X_val_aug = np.column_stack([X_val, mse_preds_val])

        train_qids = groups[train_idx]
        val_qids = groups[val_idx]
        train_groups = build_groups(train_qids)
        val_groups_lr = build_groups(val_qids)

        y_train_d = y_discrete[train_idx]
        y_val_d = y_discrete[val_idx]

        lr_model = lgb.LGBMRanker(**lr_params)
        lr_model.fit(
            X_train_aug, y_train_d,
            group=train_groups,
            eval_set=[(X_val_aug, y_val_d)],
            eval_group=[val_groups_lr],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        preds = lr_model.predict(X_val_aug)

        # Group predictions by query
        val_query_ids = sorted(set(val_groups))
        reranker_ranked_results = {}
        production_ranked_results = {}

        for qi in val_query_ids:
            qtext = queries[qi]
            mask = val_groups == qi
            q_preds = preds[mask]
            q_mids = [memory_ids[idx][1] for idx in val_idx[mask]]

            order = np.argsort(-q_preds)
            reranker_ranked_results[qtext] = [q_mids[i] for i in order]

            ranked = score_and_rank(qtext, params, full_data["search_data"],
                                    full_data["feedback_raw"], full_data["hebb_data"])
            production_ranked_results[qtext] = ranked

        reranker_ndcg = compute_ndcg(reranker_ranked_results, token_map,
                                     ground_truth, budget=5000)
        reranker_recall = compute_graded_recall(reranker_ranked_results, token_map,
                                                ground_truth, budget=5000)
        prod_ndcg = compute_ndcg(production_ranked_results, token_map,
                                 ground_truth, budget=5000)
        prod_recall = compute_graded_recall(production_ranked_results, token_map,
                                            ground_truth, budget=5000)

        fold_metrics = {
            "fold": fold_i, "n_queries": len(val_query_ids),
            "reranker_ndcg": reranker_ndcg, "reranker_recall": reranker_recall,
            "production_ndcg": prod_ndcg, "production_recall": prod_recall,
        }
        all_fold_metrics.append(fold_metrics)

        print(f"  Fold {fold_i} ({len(val_query_ids)} queries):")
        print(f"    Reranker:   NDCG={reranker_ndcg:.4f}  Recall={reranker_recall:.4f}")
        print(f"    Production: NDCG={prod_ndcg:.4f}  Recall={prod_recall:.4f}")
        print(f"    Delta:      NDCG={reranker_ndcg-prod_ndcg:+.4f}  "
              f"Recall={reranker_recall-prod_recall:+.4f}")

    mean_r_ndcg = np.mean([m["reranker_ndcg"] for m in all_fold_metrics])
    mean_r_recall = np.mean([m["reranker_recall"] for m in all_fold_metrics])
    mean_p_ndcg = np.mean([m["production_ndcg"] for m in all_fold_metrics])
    mean_p_recall = np.mean([m["production_recall"] for m in all_fold_metrics])

    print(f"\n  {'='*50}")
    print(f"  MEAN ACROSS {n_folds} FOLDS:")
    print(f"  {'='*50}")
    print(f"  Reranker:   NDCG={mean_r_ndcg:.4f}  Recall={mean_r_recall:.4f}")
    print(f"  Production: NDCG={mean_p_ndcg:.4f}  Recall={mean_p_recall:.4f}")
    print(f"  Delta:      NDCG={mean_r_ndcg-mean_p_ndcg:+.4f}  "
          f"Recall={mean_r_recall-mean_p_recall:+.4f}")

    return {"mean_reranker_ndcg": mean_r_ndcg, "mean_reranker_recall": mean_r_recall}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train learned re-ranker")
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract features, don't train")
    parser.add_argument("--train-only", action="store_true",
                        help="Train from saved features")
    parser.add_argument("--compare", action="store_true",
                        help="Run full ranking comparison")
    parser.add_argument("--ablation", action="store_true",
                        help="Run feedback feature ablation")
    parser.add_argument("--lambdarank", action="store_true",
                        help="Train LambdaRank on GT-only candidates")
    parser.add_argument("--lr-objective", type=str, default="lambdarank",
                        choices=["lambdarank", "rank_xendcg"],
                        help="Ranking objective (default: lambdarank)")
    parser.add_argument("--lr-levels", type=int, default=10,
                        help="Number of quantile bins for lambdarank (default: 10)")
    parser.add_argument("--neg-ratio", type=float, default=2.0,
                        help="Negative sampling ratio for GT-only mode (default: 2.0, 0=all)")
    parser.add_argument("--neg-strategy", type=str, default="random",
                        choices=["random", "topk", "hard"],
                        help="Negative sampling strategy (default: random)")
    parser.add_argument("--neg-top-k", type=int, default=50,
                        help="For topk strategy: keep negs ranked in top-K per channel")
    parser.add_argument("--two-stage", action="store_true",
                        help="Two-stage: MSE predictions as feature for LambdaRank")
    parser.add_argument("--n-estimators", type=int, default=500,
                        help="Max boosting rounds (default: 500)")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument("--gt", type=str, default=str(GT_PATH),
                        help="Path to ground truth JSON")
    args = parser.parse_args()

    gt_path = args.gt

    if args.train_only:
        # Load saved features
        if not FEATURES_PATH.exists():
            print(f"ERROR: No saved features at {FEATURES_PATH}")
            print("Run without --train-only first to extract features.")
            sys.exit(1)
        print(f"Loading features from {FEATURES_PATH}...")
        with open(FEATURES_PATH, "rb") as f:
            feature_data = f.read()
        feature_data = pickle.loads(feature_data)

        # Still need full_data for ranking evaluation
        full_data, ground_truth, _ = load_tuning_data(gt_path)
    else:
        # Load data and extract features
        full_data, ground_truth, _ = load_tuning_data(gt_path)

        print("\nLoading memory metadata...")
        db = get_db()
        memory_meta = load_memory_metadata(db)
        print(f"  {len(memory_meta)} active memories with metadata")

        print("\nExtracting features (full candidate pool)...")
        t0 = time.time()
        feature_data = extract_all_features(full_data, ground_truth, memory_meta)
        print(f"  Extraction time: {time.time() - t0:.1f}s")

        # Sanity checks
        X = feature_data["features"]
        print(f"\nSanity checks:")
        print(f"  NaN count: {np.isnan(X).sum()}")
        print(f"  Inf count: {np.isinf(X).sum()}")
        for fi, fname in enumerate(FEATURE_NAMES):
            col = X[:, fi]
            print(f"  {fname:20s}: min={col.min():10.4f}  max={col.max():10.4f}  "
                  f"mean={col.mean():10.4f}  std={col.std():10.4f}")

        # Save features
        FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FEATURES_PATH, "wb") as f:
            pickle.dump(feature_data, f)
        print(f"\nFeatures saved to {FEATURES_PATH}")

        if args.extract_only:
            return

    if args.lambdarank:
        if args.neg_ratio > 0 and not args.train_only:
            # GT + sampled negatives for training
            print(f"\nExtracting features (GT + {args.neg_ratio}x "
                  f"{args.neg_strategy} negatives)...")
            lr_train_data = extract_all_features(
                full_data, ground_truth, memory_meta,
                gt_only=True, neg_ratio=args.neg_ratio,
                neg_strategy=args.neg_strategy,
                neg_top_k=args.neg_top_k)
        else:
            # Use full candidate pool for training
            lr_train_data = feature_data

        # Train ranker
        train_results = train_lambdarank(lr_train_data, n_folds=args.folds,
                                         objective=args.lr_objective,
                                         n_levels=args.lr_levels)

        # Save model
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                "model": train_results["final_model"],
                "feature_names": FEATURE_NAMES,
                "params": PRODUCTION_PARAMS,
                "train_results": {
                    "importances": train_results["importances"].tolist(),
                },
            }, f)
        print(f"\nModel saved to {MODEL_PATH}")

        # Ranking evaluation: train on lr_train_data, predict on full pool
        ranking_results = evaluate_reranker_ranking(
            feature_data, full_data, ground_truth, n_folds=args.folds,
            use_lambdarank=True, train_feature_data=lr_train_data,
            lr_objective=args.lr_objective, lr_n_levels=args.lr_levels)

    else:
        # Train pointwise regressor
        train_results = train_and_evaluate(feature_data, n_folds=args.folds,
                                           n_estimators=args.n_estimators)

        # Save model
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                "model": train_results["final_model"],
                "feature_names": FEATURE_NAMES,
                "params": PRODUCTION_PARAMS,
                "train_results": {
                    "mean_rmse": train_results["mean_rmse"],
                    "importances": train_results["importances"].tolist(),
                },
            }, f)
        print(f"\nModel saved to {MODEL_PATH}")

        # Ranking comparison
        ranking_results = evaluate_reranker_ranking(
            feature_data, full_data, ground_truth, n_folds=args.folds)

        # Per-query analysis
        per_query_comparison(feature_data, full_data, ground_truth)

    # Two-stage evaluation
    if args.two_stage:
        evaluate_two_stage(feature_data, full_data, ground_truth,
                          n_folds=args.folds, n_levels=args.lr_levels)

    # Ablation
    if args.ablation:
        ablation_no_feedback(feature_data, n_folds=args.folds)


if __name__ == "__main__":
    main()
