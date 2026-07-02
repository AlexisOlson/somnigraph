# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "sqlite-vec>=0.1.6",
#     "openai>=2.0.0",
#     "tiktoken>=0.7.0",
#     "mcp[cli]>=1.2.0",
#     "numpy>=1.26",
#     "scikit-learn>=1.0",
#     "lightgbm>=4.0",
#     "fastembed>=0.4.0",
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
import gc
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

from memory.constants import CATEGORY_DECAY_RATES, DATA_DIR, DEFAULT_DECAY_RATE, PPR_DAMPING
FEATURES_PATH = DATA_DIR / "tuning_studies" / "reranker_features.pkl"
MODEL_PATH = DATA_DIR / "tuning_studies" / "reranker_model.pkl"
GT_PATH = DATA_DIR / "tuning_studies" / "gt_calibrated.json"
RESULTS_JSON_PATH = DATA_DIR / "tuning_studies" / "reranker_results.json"

# Live MCP server loads booster from .txt + feature names from .json (constants.py)
from memory.constants import MODEL_PATH as MCP_MODEL_PATH, MODEL_FEATURES_PATH


def _export_for_mcp(final_model):
    """Export trained model to the format the live reranker loads."""
    MCP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_model.booster_.save_model(str(MCP_MODEL_PATH))
    with open(MODEL_FEATURES_PATH, "w") as f:
        json.dump(FEATURE_NAMES, f)
    print(f"MCP model exported to {MCP_MODEL_PATH}")
    print(f"MCP features exported to {MODEL_FEATURES_PATH}")

# Category encoding
CATEGORY_MAP = {
    "episodic": 0, "semantic": 1, "procedural": 2,
    "reflection": 3, "entity": 4, "meta": 5,
}

# Feature names (order matters — matches extraction and live reranker)
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
    "fb_last",        # 7: most recent utility score (NaN if no feedback)
    "fb_mean",        # 8: mean of all utility scores (NaN if no feedback)
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

    # Extended features (Tier 1)
    "query_coverage",   # 18: fraction of query terms found in content
    "proximity",        # 19: inverse min-span of query terms in content
    "query_idf_var",    # 20: IDF variance across query terms
    "burstiness",       # 21: retrieval event burstiness (CoV^2 - 1)

    # Extended features (Tier 2)
    "betweenness",      # 22: betweenness centrality in memory graph
    "diversity_score",  # 23: 1 - mean cosine sim to neighbors
    "fb_time_weighted", # 24: exponentially time-weighted feedback mean
    "session_recency",  # 25: queries ago since last co-retrieval in session

    # Extended features (Tier 3 — query-level and normalization)
    "query_length",         # 26: number of query terms (complexity proxy)
    "candidate_pool_size",  # 27: total candidates after PPR expansion
    "fts_bm25_norm",        # 28: per-query normalized BM25 score (0-1)
    "vec_dist_norm",        # 29: per-query normalized vector distance (0-1)
    "decay_rate",           # 30: memory decay rate (0 = permanent)
]


# ---------------------------------------------------------------------------
# Sample-weights sidecar (Phase 2)
# ---------------------------------------------------------------------------


def load_sample_weights_sidecar(
    path: str | None,
    overrides: list[str] | None = None,
) -> tuple[dict[str, float], dict[str, str], dict[str, str], dict]:
    """Load per-query weights + modes + pinned targets from the build_gt sidecar.

    Args:
      path: Sidecar JSON written by build_gt_from_feedback.py. None disables
        weighting (all queries default weight=1.0, mode="live").
      overrides: List of "MODE=VALUE" strings to override the sidecar's
        weight schedule at train time without re-emitting the sidecar.

    Returns (query_weights, query_modes, query_pinned_targets, metadata):
      query_weights:        {qtext: weight} — every query in the sidecar.
      query_modes:          {qtext: mode}   — same keys.
      query_pinned_targets: {qtext: memory_id} — only queries that have a
        canonical pinned target (probe-event or synthetic-anchor sources;
        live queries deliberately absent so per-(q,m) boost only fires
        where there's a single confident answer).
      metadata:             schedule + holdout list + counts + schema_version
                            (empty dict if no path).

    Queries absent from the sidecar fall back to weight=1.0 / mode="live"
    at the call site — this function returns only what's in the file.
    """
    if not path:
        return {}, {}, {}, {}

    with open(path) as f:
        sidecar = json.load(f)

    metadata = dict(sidecar.get("metadata", {}))
    weights_block = sidecar.get("weights", {})

    # Parse overrides ("synthetic-self-anchor=0", "probe-mild=1.4", ...) and
    # apply them to the loaded schedule. We re-derive weights from the schedule
    # so an override propagates to every query of the affected mode.
    schedule = dict(metadata.get("weight_schedule", {}))
    parsed_overrides: dict[str, float] = {}
    if overrides:
        for spec in overrides:
            if "=" not in spec:
                raise ValueError(
                    f"--override-weight expects MODE=VALUE, got {spec!r}"
                )
            mode_key, val = spec.split("=", 1)
            parsed_overrides[mode_key.strip()] = float(val.strip())
        schedule.update(parsed_overrides)
        metadata["weight_schedule"] = schedule
        metadata["overrides_applied"] = parsed_overrides

    query_weights: dict[str, float] = {}
    query_modes: dict[str, str] = {}
    query_pinned_targets: dict[str, str] = {}
    for q, info in weights_block.items():
        mode = info.get("mode", "live")
        # If an override touched this mode, use the override; otherwise use
        # the per-query weight as written. This matters when the user asks
        # "make synthetic 0 without re-emitting the sidecar" — every
        # synthetic-self-anchor row goes to 0 even though its sidecar weight
        # was 0.5.
        if mode in parsed_overrides:
            weight = float(parsed_overrides[mode])
        else:
            weight = float(info.get("weight", 1.0))
        query_weights[q] = weight
        query_modes[q] = mode
        # schema v2: pinned_target_id may be null — only record real strings.
        pinned = info.get("pinned_target_id")
        if pinned:
            query_pinned_targets[q] = str(pinned)

    return query_weights, query_modes, query_pinned_targets, metadata


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def load_memory_metadata(db) -> dict:
    """Load per-memory metadata from DB for feature extraction.

    Uses the reranker module's _load_memory_meta() which precomputes all
    26 feature metadata (including betweenness, diversity, burstiness, etc.).
    """
    from memory.reranker import _load_memory_meta
    return _load_memory_meta(db)


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

    pool_size = len(candidate_ids)

    # Per-query magnitude for normalized score features.
    # SQLite FTS5 BM25 is negative (more-negative = better match), so we
    # normalize by the absolute value of the strongest match in the pool.
    abs_best_fts_score = abs(min(fts_score_map.values())) if fts_score_map else 0.0
    max_vec_score = max(vec_score_map.values()) if vec_score_map else 0.0

    # --- Query-level features ---
    query_terms = [t for t in qtext.lower().split() if len(t) > 1]
    q_len = len(query_terms)

    idf_stats = memory_meta.get("__idf_stats__", {})
    total_docs = idf_stats.get("total_docs", 1)
    term_doc_freq = idf_stats.get("term_doc_freq", {})
    query_idf_var = 0.0
    if query_terms and term_doc_freq:
        idfs = []
        for term in query_terms:
            df = term_doc_freq.get(term, 0)
            idf = math.log((total_docs + 1) / (df + 1))
            idfs.append(idf)
        if len(idfs) > 1:
            idf_mean = sum(idfs) / len(idfs)
            query_idf_var = sum((x - idf_mean)**2 for x in idfs) / len(idfs)

    # Session recency
    session_retrievals = memory_meta.get("__session_retrievals__", {})
    session_recency_map = {}
    for sess_id, events in session_retrievals.items():
        query_positions = [i for i, (q, _) in enumerate(events) if q == qtext]
        if not query_positions:
            continue
        qpos = query_positions[-1]
        for j in range(qpos - 1, -1, -1):
            _, mid = events[j]
            queries_ago = qpos - j
            if mid not in session_recency_map or queries_ago < session_recency_map[mid]:
                session_recency_map[mid] = queries_ago

    now_ts = time.time()
    fb_lambda = 0.01

    # --- Build feature matrix ---
    from memory.reranker import _compute_proximity

    n_features = len(FEATURE_NAMES)
    candidate_list = sorted(candidate_ids)
    n_candidates = len(candidate_list)

    features = np.zeros((n_candidates, n_features), dtype=np.float32)
    labels = np.zeros(n_candidates, dtype=np.float32)

    # Missing-encoding policy (see docs/architecture.md "What didn't work"):
    # any feature whose missing value would masquerade as a real measurement
    # uses NaN so LightGBM learns an explicit missing-branch split. Features
    # whose 0 has a legitimate meaning (no overlap, no PMI, zero count) keep 0.
    nan = float("nan")

    for i, mid in enumerate(candidate_list):
        # Channel ranks: real values 0-199 (smaller=better). -1 used to read
        # as "better than rank 0" — bug-shape sentinel, NaN-ified 2026-05-08.
        features[i, 0] = fts_ranked.get(mid, nan)                           # fts_rank
        features[i, 1] = vec_ranked.get(mid, nan)                           # vec_rank
        features[i, 2] = theme_ranked.get(mid, nan)                         # theme_rank
        features[i, 3] = ppr_scores.get(mid, 0.0)                           # ppr_score (0 legitimate)
        # Raw channel scores: BM25 is negative (more-negative=better),
        # vec_dist is non-negative (smaller=closer). Default 0 misrepresented
        # both — fts as "no match" and vec as "perfect match." NaN now.
        features[i, 4] = fts_score_map.get(mid, nan)                        # fts_bm25
        features[i, 5] = vec_score_map.get(mid, nan)                        # vec_dist
        features[i, 6] = theme_overlap_map.get(mid, 0)                      # theme_overlap (0 legitimate)

        # Feedback signals — NaN for missing so LightGBM learns a separate
        # "missing" branch instead of treating no-feedback as worst-case.
        fb = feedback_raw.get(mid)
        if fb and fb["count"] > 0:
            features[i, 7] = fb["utilities"][-1]                              # fb_last
            features[i, 8] = sum(fb["utilities"]) / fb["count"]               # fb_mean
            features[i, 9] = fb["count"]                                      # fb_count
        else:
            features[i, 7] = float("nan")                                     # fb_last (missing)
            features[i, 8] = float("nan")                                     # fb_mean (missing)
            features[i, 9] = 0                                                # fb_count

        # Graph signals
        features[i, 10] = hebbian_pmi_map.get(mid, 0.0)                     # hebbian_pmi

        # Memory metadata. memory_meta is active-only; a missing entry means a
        # non-active candidate (pending/superseded). Post-hygiene the candidate
        # pool is active-only so this branch is dead, but NaN-encode it per the
        # codified policy and to keep feature parity with reranker.py — the old
        # masquerading defaults (category=1, priority=5, ...) read as a real
        # mid-priority semantic memory. Feature parity between live and training
        # extraction is a hard invariant (see experiments.md § Feature parity).
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
            features[i, 11] = nan   # missing memory (non-active candidate)
            features[i, 12] = nan
            features[i, 13] = nan
            features[i, 14] = nan
            features[i, 15] = nan
            features[i, 16] = nan
            features[i, 17] = nan

        # Extended features (Tier 1). Missing memory → NaN; memory present with no
        # usable query terms keeps the legitimate 0.0 (coverage/proximity genuinely
        # zero, not missing).
        if meta is None:
            features[i, 18] = nan                                            # query_coverage
        elif query_terms:
            content_set = set(meta.get("content_tokens", []))
            matched = sum(1 for t in query_terms if t in content_set)
            features[i, 18] = matched / len(query_terms)                     # query_coverage
        else:
            features[i, 18] = 0.0

        if meta is None:
            features[i, 19] = nan                                            # proximity
        elif len(query_terms) > 1:
            features[i, 19] = _compute_proximity(query_terms, meta.get("content_tokens", []))
        else:
            features[i, 19] = 0.0

        features[i, 20] = query_idf_var                                     # query_idf_var

        if meta:
            features[i, 21] = meta.get("burstiness", 0.0)                   # burstiness
        else:
            features[i, 21] = nan

        # Extended features (Tier 2)
        if meta:
            features[i, 22] = meta.get("betweenness", 0.0)                  # betweenness
            features[i, 23] = meta.get("diversity_score", 0.5)              # diversity_score
        else:
            features[i, 22] = nan
            features[i, 23] = nan

        if meta:
            fb_ts = meta.get("fb_timestamps", [])
            if fb_ts:
                weighted_sum = 0.0
                weight_sum = 0.0
                for ts, util in fb_ts:
                    age_fb = (now_ts - ts) / 86400.0
                    w = math.exp(-fb_lambda * age_fb)
                    weighted_sum += w * util
                    weight_sum += w
                features[i, 24] = weighted_sum / weight_sum if weight_sum > 0 else float("nan")
            else:
                features[i, 24] = float("nan")
        else:
            features[i, 24] = float("nan")

        # NaN for missing — real session_recency is queries_ago >= 1, so -1 reads
        # as "more recent than any real co-retrieval" and routes self-reinforcement
        # bias the same way the fb-NaN sentinel did before 2026-05-08.
        features[i, 25] = session_recency_map.get(mid, float("nan"))         # session_recency

        # Extended features (Tier 3)
        features[i, 26] = q_len                                               # query_length
        features[i, 27] = pool_size                                            # candidate_pool_size
        # Normalized channel scores: real values 0-1. Per-candidate missing
        # used to default 0, which reads as "weakest match" for FTS and
        # "closest" (wrong direction!) for vec. NaN now.
        if mid in fts_score_map and abs_best_fts_score > 0:
            features[i, 28] = abs(fts_score_map[mid]) / abs_best_fts_score
        else:
            features[i, 28] = nan                                            # fts_bm25_norm
        if mid in vec_score_map and max_vec_score > 0:
            features[i, 29] = vec_score_map[mid] / max_vec_score
        else:
            features[i, 29] = nan                                            # vec_dist_norm
        features[i, 30] = meta.get("decay_rate", DEFAULT_DECAY_RATE) if meta else nan  # decay_rate (NaN for missing memory)

        # Label
        labels[i] = ground_truth_for_query.get(mid, 0.0)

    return features, labels, candidate_list


def extract_all_features(full_data: dict, ground_truth: dict,
                         memory_meta: dict, gt_only: bool = False,
                         neg_ratio: float = 0,
                         neg_strategy: str = "random",
                         neg_top_k: int = 50,
                         query_weights: dict[str, float] | None = None,
                         query_modes: dict[str, str] | None = None,
                         query_pinned_targets: dict[str, str] | None = None,
                         pinned_boost: float = 1.0) -> dict:
    """Extract features for all GT queries.

    Args:
      gt_only: If True, filter candidates per query.
      neg_ratio: Negatives per positive (e.g., 2.0). 0 = no filtering.
      neg_strategy: "random", "topk" (keep top-K ranked per channel),
                    or "hard" (MSE-scored false positives).
      neg_top_k: For "topk" strategy, keep non-GT candidates ranked
                 in top-K by any channel.
      query_weights: Optional {qtext: weight} from the sidecar. Queries
        absent get the default weight 1.0.
      query_modes: Optional {qtext: mode} from the sidecar. Queries absent
        get mode "live". Used downstream for per-mode evaluation metrics.
      query_pinned_targets: Optional {qtext: memory_id} from sidecar v2.
        Rows whose memory_id matches the pinned target get their per-row
        weight multiplied by pinned_boost (Phase 3a per-(q,m) weighting).
      pinned_boost: Multiplier applied to pinned-target rows. 1.0 = no-op
        (legacy uniform-per-query weighting); >1.0 emphasizes the
        highest-confidence label per probe/synthetic query.

    Returns dict with keys:
      features: (N, n_features) array
      labels: (N,) array
      query_ids: (N,) array of query indices (for GroupKFold)
      memory_ids: list of (query, mid) pairs
      feature_names: list of feature names
      weights: (N,) array of per-row sample weights (defaults to 1.0)
      modes_per_row: list[str] of length N
      query_modes: dict {qtext: mode} (broadcast-source for modes_per_row)
    """
    params = dict(PRODUCTION_PARAMS)
    search_data = full_data["search_data"]
    feedback_raw = full_data["feedback_raw"]
    hebb_data = full_data["hebb_data"]

    qw = query_weights or {}
    qm = query_modes or {}
    qp = query_pinned_targets or {}

    rng = np.random.RandomState(42)

    all_features = []
    all_labels = []
    all_query_ids = []
    all_mids = []
    all_weights = []
    all_modes = []

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
        # Broadcast the query's weight + mode to every row. Queries absent
        # from the sidecar default to 1.0 / "live" — same as pre-Phase-2
        # uniform weighting. Phase 3a: rows whose memory_id matches the
        # query's pinned target get the boost multiplier on top.
        q_weight = qw.get(qtext, 1.0)
        q_mode = qm.get(qtext, "live")
        pinned_mid = qp.get(qtext)
        if pinned_mid and pinned_boost != 1.0:
            row_weights = [
                q_weight * pinned_boost if mid == pinned_mid else q_weight
                for mid in mids
            ]
        else:
            row_weights = [q_weight] * len(mids)
        all_weights.extend(row_weights)
        all_modes.extend([q_mode] * len(mids))

        if (qi + 1) % 50 == 0:
            n_pos = sum(1 for l in labs if l > 0)
            print(f"  {qi + 1}/{len(queries)} queries, "
                  f"{len(mids)} candidates, {n_pos} positive")

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    query_ids = np.array(all_query_ids, dtype=np.int32)
    weights = np.array(all_weights, dtype=np.float32)

    n_pos = np.sum(labels > 0)
    print(f"\nFeature extraction complete:")
    print(f"  Total samples: {len(labels)}")
    print(f"  Positive (GT > 0): {n_pos} ({100*n_pos/len(labels):.1f}%)")
    print(f"  Queries: {len(set(query_ids))}")
    print(f"  Features: {features.shape[1]}")

    if qw:
        # Distribution of per-row weights — quick sanity that the schedule
        # actually attached to the data we're about to train on.
        unique_w, counts = np.unique(weights, return_counts=True)
        print(f"  Sample-weight distribution:")
        for w, n in zip(unique_w, counts):
            print(f"    weight={w:.3f}  rows={n}")
        n_zero_rows = int(np.sum(weights == 0.0))
        print(f"  Held-out rows (weight=0): {n_zero_rows}")

    return {
        "features": features,
        "labels": labels,
        "query_ids": query_ids,
        "memory_ids": all_mids,
        "feature_names": FEATURE_NAMES,
        "weights": weights,
        "modes_per_row": all_modes,
        "query_modes": dict(qm),
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
    weights = feature_data.get("weights")
    modes_per_row = feature_data.get("modes_per_row")
    if weights is None:
        weights = np.ones(len(y), dtype=np.float32)
    if modes_per_row is None:
        modes_per_row = ["live"] * len(y)

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

    # Per-mode RMSE accumulators across folds. Held-out modes (weight=0) still
    # get RMSE reported because that's the generalization signal.
    mode_rmse_acc: dict[str, list[float]] = defaultdict(list)
    mode_n_acc: dict[str, int] = defaultdict(int)

    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train, w_val = weights[train_idx], weights[val_idx]

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[w_val],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        preds = model.predict(X_val)
        preds = np.clip(preds, 0, 1)
        oof_predictions[val_idx] = preds
        importances += model.feature_importances_

        # Aggregate RMSE uses unweighted mean — reporting model-side error
        # over actual rows. Per-mode weighting kicks in via the breakdown.
        rmse = np.sqrt(np.mean((preds - y_val) ** 2))
        fold_results.append({"fold": fold_i, "rmse": float(rmse),
                             "n_val": len(val_idx),
                             "best_iter": model.best_iteration_})

        # Per-mode RMSE within this fold.
        val_modes = [modes_per_row[i] for i in val_idx]
        val_modes_arr = np.array(val_modes)
        for mode in np.unique(val_modes_arr):
            mask = val_modes_arr == mode
            if mask.sum() == 0:
                continue
            rmse_m = float(np.sqrt(np.mean((preds[mask] - y_val[mask]) ** 2)))
            mode_rmse_acc[mode].append(rmse_m)
            mode_n_acc[mode] += int(mask.sum())

        n_val_pos = np.sum(y_val > 0)
        print(f"  Fold {fold_i}: RMSE={rmse:.4f} "
              f"best_iter={model.best_iteration_} "
              f"val_samples={len(val_idx)} val_pos={n_val_pos}")

        # Free this fold's array copies before the next fold allocates its
        # own — fancy indexing copies, and at millions of rows the train
        # copy alone is ~460MB. Two folds' worth held simultaneously OOM'd
        # the V6 retrain (2026-07-01); the RHS of the next fold's slice
        # assignment is evaluated while these are still referenced.
        del X_train, X_val, y_train, y_val, w_train, w_val, model, preds
        gc.collect()

    importances /= n_folds

    mean_rmse = np.mean([r["rmse"] for r in fold_results])
    print(f"\n  Mean RMSE: {mean_rmse:.4f}")

    # Per-mode RMSE summary (mean across folds, total row count).
    per_mode_rmse: dict[str, dict] = {}
    if any(len(rs) > 0 for rs in mode_rmse_acc.values()):
        print(f"\n  Per-mode RMSE (mean across folds):")
        for mode in sorted(mode_rmse_acc.keys()):
            rs = mode_rmse_acc[mode]
            mean_r = float(np.mean(rs))
            per_mode_rmse[mode] = {
                "rmse": mean_r,
                "n_rows": mode_n_acc[mode],
                "n_folds_seen": len(rs),
            }
            print(f"    {mode:25s} n_rows={mode_n_acc[mode]:6d} "
                  f"folds={len(rs):2d} RMSE={mean_r:.4f}")

    # Feature importance
    print(f"\n  Feature importance (gain):")
    imp_order = np.argsort(importances)[::-1]
    for idx in imp_order:
        print(f"    {FEATURE_NAMES[idx]:20s}: {importances[idx]:8.1f}")

    # Train final model on all data
    print(f"\nTraining final model on all data...")
    final_model = lgb.LGBMRegressor(**lgb_params)
    final_model.fit(X, y, sample_weight=weights)

    return {
        "fold_results": fold_results,
        "oof_predictions": oof_predictions,
        "importances": importances,
        "final_model": final_model,
        "mean_rmse": float(mean_rmse),
        "per_mode_rmse": per_mode_rmse,
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
    weights = feature_data.get("weights")
    if weights is None:
        weights = np.ones(len(y), dtype=np.float32)

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
        w_train = weights[train_idx]
        w_val = weights[val_idx]

        train_groups = build_groups(train_qids)
        val_groups = build_groups(val_qids)

        model = lgb.LGBMRanker(**lgb_params)
        model.fit(
            X_train, y_tr,
            group=train_groups,
            sample_weight=w_train,
            eval_set=[(X_val, y_va)],
            eval_group=[val_groups],
            eval_sample_weight=[w_val],
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
    final_model.fit(X, y_train_labels, group=all_groups, sample_weight=weights)

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
    query_modes_map: dict[str, str] = feature_data.get("query_modes", {}) or {}

    queries = full_data["gt_queries"]
    params = dict(PRODUCTION_PARAMS)

    # Training data (may be GT-only subset for LambdaRank)
    if use_lambdarank and train_feature_data is not None:
        X_tr = train_feature_data["features"]
        y_tr = train_feature_data["labels"]
        groups_tr = train_feature_data["query_ids"]
        weights_tr = train_feature_data.get("weights")
    else:
        X_tr = X_full
        y_tr = y_full
        groups_tr = groups_full
        weights_tr = feature_data.get("weights")
    if weights_tr is None:
        weights_tr = np.ones(len(y_tr), dtype=np.float32)

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
    # Per-mode aggregation across folds — every val query partitioned by its
    # mode, then NDCG@5k and R@10 computed within each partition.
    per_mode_fold_metrics: dict[str, list[dict]] = defaultdict(list)

    # Build query-to-fold mapping from training splits
    for fold_i, (train_idx_tr, val_idx_tr) in enumerate(gkf.split(X_tr, y_tr, groups_tr)):
        val_query_ids_set = set(groups_tr[val_idx_tr])

        if use_lambdarank:
            # Train on GT-only training fold
            X_train = X_tr[train_idx_tr]
            y_train_lr = y_tr_labels[train_idx_tr]
            train_qids = groups_tr[train_idx_tr]
            train_groups = build_groups(train_qids)
            w_train = weights_tr[train_idx_tr]

            # Validation on GT-only val fold (for early stopping)
            X_val_tr = X_tr[val_idx_tr]
            y_val_lr = y_tr_labels[val_idx_tr]
            val_qids = groups_tr[val_idx_tr]
            val_groups = build_groups(val_qids)
            w_val = weights_tr[val_idx_tr]

            model = lgb.LGBMRanker(**lgb_params)
            model.fit(
                X_train, y_train_lr,
                group=train_groups,
                sample_weight=w_train,
                eval_set=[(X_val_tr, y_val_lr)],
                eval_group=[val_groups],
                eval_sample_weight=[w_val],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        else:
            X_train = X_tr[train_idx_tr]
            y_train = y_tr[train_idx_tr]
            X_val_tr = X_tr[val_idx_tr]
            y_val = y_tr[val_idx_tr]
            w_train = weights_tr[train_idx_tr]
            w_val = weights_tr[val_idx_tr]

            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(
                X_train, y_train,
                sample_weight=w_train,
                eval_set=[(X_val_tr, y_val)],
                eval_sample_weight=[w_val],
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

        # Partition val queries by mode and compute per-mode NDCG@5k + R@10.
        # Held-out modes (e.g. probe-hard, weight=0) still appear here —
        # that's the whole point of holdout, the metric is the signal.
        #
        # Phase-1 miss caveat (added 2026-05-09 after V5+1b): a query with
        # at least one positive-label memory where no positive memory survives
        # to the ranked list mechanically scores NDCG=0 / R@10=0 — the
        # candidate-pool retrieval missed the target before the reranker saw
        # anything. As probe modes (especially `hard`) inject more
        # vocabulary-stripped queries, the miss rate grows and drags the
        # headline mean down independently of model quality. We compute both
        # the miss-inclusive metric (apples-to-apples cross-retrain when GT
        # is fixed) and the non-miss-only metric (model quality on rankable
        # queries). Compare miss-inclusive across iterations on identical GT;
        # use non-miss when the holdout composition is shifting.
        def _is_phase1_miss(ranked_ids: list[str], gt: dict[str, float]) -> bool:
            relevant = {mid for mid, score in gt.items() if score >= 0.5}
            if not relevant:
                return False  # no positive labels — query has no "miss" semantics
            return not relevant.intersection(ranked_ids)

        mode_buckets: dict[str, dict] = defaultdict(dict)
        for qtext, ranked in reranker_ranked_results.items():
            mode = query_modes_map.get(qtext, "live")
            mode_buckets[mode][qtext] = ranked
        for mode, sub_ranked in mode_buckets.items():
            sub_gt = {q: ground_truth[q] for q in sub_ranked if q in ground_truth}
            if not sub_gt:
                continue
            ndcg_m = compute_ndcg(sub_ranked, token_map, sub_gt, budget=5000)
            recall_m = compute_recall_at_k(sub_ranked, sub_gt, k=10, threshold=0.5)

            # Non-miss subset: drop queries where retrieval surfaced no
            # positive-label memory at all.
            non_miss_ranked = {q: r for q, r in sub_ranked.items()
                               if not _is_phase1_miss(r, sub_gt[q])}
            non_miss_gt = {q: sub_gt[q] for q in non_miss_ranked}
            n_miss = len(sub_ranked) - len(non_miss_ranked)
            if non_miss_gt:
                ndcg_nm = compute_ndcg(non_miss_ranked, token_map, non_miss_gt, budget=5000)
                recall_nm = compute_recall_at_k(non_miss_ranked, non_miss_gt, k=10, threshold=0.5)
            else:
                ndcg_nm = float("nan")
                recall_nm = float("nan")

            per_mode_fold_metrics[mode].append({
                "fold": fold_i,
                "n_queries": len(sub_ranked),
                "n_phase1_miss": n_miss,
                "ndcg_5k": float(ndcg_m),
                "recall_10": float(recall_m),
                "ndcg_5k_non_miss": float(ndcg_nm),
                "recall_10_non_miss": float(recall_nm),
            })

        print(f"  Fold {fold_i} ({len(val_query_ids_set)} queries):")
        print(f"    Reranker:   NDCG={reranker_ndcg:.4f}  Recall={reranker_recall:.4f}")
        print(f"    Production: NDCG={prod_ndcg:.4f}  Recall={prod_recall:.4f}")
        print(f"    Delta:      NDCG={reranker_ndcg-prod_ndcg:+.4f}  "
              f"Recall={reranker_recall-prod_recall:+.4f}")

        # Same per-fold memory hygiene as train_and_evaluate — these copies
        # are fold-sized and the next iteration re-allocates before rebinding.
        del X_train, X_val_tr, X_val_full, w_train, w_val, model, preds
        gc.collect()

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

    # Per-mode summary across folds.
    per_mode_summary: dict[str, dict] = {}
    if per_mode_fold_metrics:
        # Identify holdout modes via the (already loaded) sidecar metadata if
        # available; otherwise fall back to tagging probe-hard. The label is
        # informational — the metric is computed identically for all modes.
        sidecar_meta = feature_data.get("sample_weights_metadata") or {}
        holdout_modes = set(sidecar_meta.get("holdout_modes") or ["probe-hard"])
        print(f"\n  {'='*50}")
        print(f"  PER-MODE RERANKER METRICS (mean across folds):")
        print(f"  {'='*50}")
        for mode in sorted(per_mode_fold_metrics.keys()):
            rows = per_mode_fold_metrics[mode]
            ndcg_m = float(np.mean([r["ndcg_5k"] for r in rows]))
            recall_m = float(np.mean([r["recall_10"] for r in rows]))
            # Non-miss aggregates: filter NaN folds (modes where every val
            # query was a Phase-1 miss in that fold; rare, but possible at
            # tiny mode sizes).
            nm_ndcg_vals = [r["ndcg_5k_non_miss"] for r in rows
                            if not math.isnan(r["ndcg_5k_non_miss"])]
            nm_recall_vals = [r["recall_10_non_miss"] for r in rows
                              if not math.isnan(r["recall_10_non_miss"])]
            ndcg_nm = float(np.mean(nm_ndcg_vals)) if nm_ndcg_vals else float("nan")
            recall_nm = float(np.mean(nm_recall_vals)) if nm_recall_vals else float("nan")
            n_q = sum(r["n_queries"] for r in rows)
            n_miss = sum(r["n_phase1_miss"] for r in rows)
            miss_pct = (n_miss / n_q * 100.0) if n_q else 0.0
            holdout = mode in holdout_modes
            per_mode_summary[mode] = {
                "n_queries": n_q,
                "n_phase1_miss": n_miss,
                "miss_pct": miss_pct,
                "ndcg_5k": ndcg_m,
                "recall_10": recall_m,
                "ndcg_5k_non_miss": ndcg_nm,
                "recall_10_non_miss": recall_nm,
                "n_folds_seen": len(rows),
                "holdout": holdout,
            }
            tag = "  [HOLDOUT]" if holdout else ""
            ndcg_nm_str = f"{ndcg_nm:.4f}" if not math.isnan(ndcg_nm) else "  n/a "
            recall_nm_str = f"{recall_nm:.4f}" if not math.isnan(recall_nm) else "  n/a "
            print(f"    {mode:25s} n={n_q:5d} folds={len(rows):2d} "
                  f"NDCG={ndcg_m:.4f}  R@10={recall_m:.4f}  |  "
                  f"non-miss NDCG={ndcg_nm_str}  R@10={recall_nm_str}  "
                  f"(miss {n_miss}/{n_q}={miss_pct:.1f}%){tag}")

    return {
        "fold_metrics": all_fold_metrics,
        "mean_reranker_ndcg": float(mean_reranker_ndcg),
        "mean_reranker_recall": float(mean_reranker_recall),
        "mean_prod_ndcg": float(mean_prod_ndcg),
        "mean_prod_recall": float(mean_prod_recall),
        "per_mode_metrics": per_mode_summary,
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
    weights = feature_data.get("weights")
    if weights is None:
        weights = np.ones(len(y), dtype=np.float32)

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
            sample_weight=weights[train_idx],
            eval_set=[(X_no_fb[val_idx], y[val_idx])],
            eval_sample_weight=[weights[val_idx]],
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
    weights = feature_data.get("weights")
    if weights is None:
        weights = np.ones(len(y), dtype=np.float32)

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
        w_train, w_val = weights[train_idx], weights[val_idx]
        val_groups = groups[val_idx]

        # Stage 1: Train MSE, get predictions for train+val
        mse_model = lgb.LGBMRegressor(**mse_params)
        mse_model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[w_val],
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
            sample_weight=w_train,
            eval_set=[(X_val_aug, y_val_d)],
            eval_group=[val_groups_lr],
            eval_sample_weight=[w_val],
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
# Results JSON
# ---------------------------------------------------------------------------


def _write_results_json(path: str, train_results: dict, ranking_results: dict,
                        sample_weights_metadata: dict) -> None:
    """Persist headline metrics + per-mode breakdown alongside the model."""
    out = {
        "mean_rmse": train_results.get("mean_rmse"),
        "per_mode_rmse": train_results.get("per_mode_rmse", {}),
        "ranking": {
            "mean_reranker_ndcg": ranking_results.get("mean_reranker_ndcg"),
            "mean_reranker_recall": ranking_results.get("mean_reranker_recall"),
            "mean_prod_ndcg": ranking_results.get("mean_prod_ndcg"),
            "mean_prod_recall": ranking_results.get("mean_prod_recall"),
            "per_mode_metrics": ranking_results.get("per_mode_metrics", {}),
        },
        "sample_weights_metadata": sample_weights_metadata or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"Results JSON written to {path}")


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
    parser.add_argument("--vec-input-overrides", type=str, default=None,
                        help="Sidecar JSON of synthetic (query → context) overrides "
                             "from build_gt_from_feedback.py. Lets training mirror "
                             "recall's (query, context) asymmetry for synthetic anchors. "
                             "Real recall_meta entries from the DB take precedence.")
    parser.add_argument("--sample-weights", type=str, default=None,
                        help="Sample-weights sidecar JSON from "
                             "build_gt_from_feedback.py. Enables per-row sample "
                             "weighting + per-mode evaluation metrics. If absent, "
                             "all rows get weight=1.0 / mode='live' (legacy behavior).")
    parser.add_argument("--override-weight", action="append", default=None,
                        help="Override schedule weight at train time without "
                             "re-emitting the sidecar. Format: MODE=VALUE "
                             "(e.g. --override-weight synthetic-self-anchor=0). "
                             "Repeatable.")
    parser.add_argument("--pinned-boost", type=float, default=5.0,
                        help="Per-(q,m) multiplier on rows where memory_id "
                             "matches the query's pinned_target_id (sidecar "
                             "schema v2). Default 5.0 (locked by V3 sweep, "
                             "2026-05-09); set to 1.0 to disable per-(q,m) "
                             "boosting (legacy uniform-per-query weighting). "
                             "Higher values emphasize the highest-confidence "
                             "label per probe/synthetic query but risk "
                             "over-pinning.")
    parser.add_argument("--results-json", type=str, default=str(RESULTS_JSON_PATH),
                        help=f"Path for the results JSON (default: {RESULTS_JSON_PATH})")
    args = parser.parse_args()

    gt_path = args.gt
    vec_overrides_path = args.vec_input_overrides

    # Load sample-weights sidecar if provided. This must happen before
    # extract_all_features so per-row weights/modes attach during extraction.
    (
        query_weights,
        query_modes,
        query_pinned_targets,
        sample_weights_metadata,
    ) = load_sample_weights_sidecar(
        args.sample_weights, overrides=args.override_weight,
    )
    if args.sample_weights:
        print(f"\nSample-weights sidecar loaded: {args.sample_weights}")
        sched = sample_weights_metadata.get("weight_schedule", {})
        print(f"  Schema version: {sample_weights_metadata.get('schema_version', 1)}")
        print(f"  Schedule (effective): {sched}")
        print(f"  Holdout modes: {sample_weights_metadata.get('holdout_modes', [])}")
        print(f"  Queries in sidecar: {len(query_weights)}")
        print(f"  Queries with pinned_target_id: {len(query_pinned_targets)}")
        sidecar_default_boost = sample_weights_metadata.get("pinned_boost_default")
        print(f"  Pinned boost (CLI):     {args.pinned_boost}")
        if sidecar_default_boost is not None:
            print(f"  Pinned boost (sidecar): {sidecar_default_boost} (informational)")
        if sample_weights_metadata.get("overrides_applied"):
            print(f"  Overrides applied: {sample_weights_metadata['overrides_applied']}")
    sample_weights_metadata = dict(sample_weights_metadata)
    sample_weights_metadata["pinned_boost_applied"] = float(args.pinned_boost)

    if args.train_only:
        # Load saved features
        if not FEATURES_PATH.exists():
            print(f"ERROR: No saved features at {FEATURES_PATH}")
            print("Run without --train-only first to extract features.")
            sys.exit(1)
        print(f"Loading features from {FEATURES_PATH}...")
        # pickle.load streams from the file — read()+loads held the raw
        # bytes AND the reconstructed objects simultaneously (~2x peak).
        with open(FEATURES_PATH, "rb") as f:
            feature_data = pickle.load(f)

        # Still need full_data for ranking evaluation
        full_data, ground_truth, _ = load_tuning_data(
            gt_path, vec_overrides_path, ppr_dampings=[PPR_DAMPING],
        )

        # If a sidecar was supplied, re-derive per-row weights/modes from the
        # cached (qtext, mid) pairs so --train-only respects the new schedule
        # without forcing a fresh extraction. Queries absent from the sidecar
        # default to weight=1.0 / mode="live", same as the no-sidecar case.
        # Phase 3a: apply per-(q,m) pinned boost on the cached path too —
        # the cache already has the (qtext, mid) pairs, so this is just an
        # extra multiplier per row.
        if args.sample_weights:
            n_rows = len(feature_data["memory_ids"])
            new_weights = np.ones(n_rows, dtype=np.float32)
            new_modes = ["live"] * n_rows
            boost = float(args.pinned_boost)
            for i, (qtext, mid) in enumerate(feature_data["memory_ids"]):
                w = float(query_weights.get(qtext, 1.0))
                pinned = query_pinned_targets.get(qtext)
                if pinned and mid == pinned and boost != 1.0:
                    w *= boost
                new_weights[i] = w
                if qtext in query_modes:
                    new_modes[i] = query_modes[qtext]
            feature_data["weights"] = new_weights
            feature_data["modes_per_row"] = new_modes
            feature_data["query_modes"] = dict(query_modes)
            feature_data["sample_weights_metadata"] = sample_weights_metadata
    else:
        # Load data and extract features
        full_data, ground_truth, _ = load_tuning_data(
            gt_path, vec_overrides_path, ppr_dampings=[PPR_DAMPING],
        )

        print("\nLoading memory metadata...")
        db = get_db()
        memory_meta = load_memory_metadata(db)
        n_mems = sum(1 for k in memory_meta if not k.startswith("__"))
        print(f"  {n_mems} active memories with metadata")

        print("\nExtracting features (full candidate pool)...")
        t0 = time.time()
        feature_data = extract_all_features(
            full_data, ground_truth, memory_meta,
            query_weights=query_weights, query_modes=query_modes,
            query_pinned_targets=query_pinned_targets,
            pinned_boost=float(args.pinned_boost),
        )
        feature_data["sample_weights_metadata"] = sample_weights_metadata
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
                neg_top_k=args.neg_top_k,
                query_weights=query_weights, query_modes=query_modes,
                query_pinned_targets=query_pinned_targets,
                pinned_boost=float(args.pinned_boost))
            lr_train_data["sample_weights_metadata"] = sample_weights_metadata
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
        _export_for_mcp(train_results["final_model"])

        # Ranking evaluation: train on lr_train_data, predict on full pool
        ranking_results = evaluate_reranker_ranking(
            feature_data, full_data, ground_truth, n_folds=args.folds,
            use_lambdarank=True, train_feature_data=lr_train_data,
            lr_objective=args.lr_objective, lr_n_levels=args.lr_levels)

        _write_results_json(args.results_json, train_results, ranking_results,
                            sample_weights_metadata)

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
                    "per_mode_rmse": train_results.get("per_mode_rmse", {}),
                },
            }, f)
        print(f"\nModel saved to {MODEL_PATH}")
        _export_for_mcp(train_results["final_model"])

        # Ranking comparison
        ranking_results = evaluate_reranker_ranking(
            feature_data, full_data, ground_truth, n_folds=args.folds)

        # Per-query analysis
        per_query_comparison(feature_data, full_data, ground_truth)

        # Results JSON sidecar — captures the per-mode metrics so they can be
        # diffed across runs (held-out hard NDCG/R@10 is the trustworthy
        # generalization signal).
        _write_results_json(args.results_json, train_results, ranking_results,
                            sample_weights_metadata)

    # Two-stage evaluation
    if args.two_stage:
        evaluate_two_stage(feature_data, full_data, ground_truth,
                          n_folds=args.folds, n_levels=args.lr_levels)

    # Ablation
    if args.ablation:
        ablation_no_feedback(feature_data, n_folds=args.folds)


if __name__ == "__main__":
    main()
