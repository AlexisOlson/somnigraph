"""Reranker — learned scoring for memory retrieval.

Loads a trained LightGBM model and extracts the same features used during
training (from train_reranker.py) on-the-fly during impl_recall().

The reranker replaces the hand-tuned RRF + UCB + Hebbian + PPR + theme boost
formula with a single model.predict() call. Feature extraction reuses the
same raw signals the formula consumed — ranks, PPR scores, feedback, metadata —
but lets the model learn the weighting instead of specifying it by hand.

Fallback: if no model file exists, the formula path runs unchanged.
"""

import json
import logging
import math
import pickle
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from memory.constants import DATA_DIR, MODEL_PATH, HEBBIAN_MIN_JOINT

logger = logging.getLogger("claude-memory")

# Category encoding (must match train_reranker.py)
CATEGORY_MAP = {
    "episodic": 0, "semantic": 1, "procedural": 2,
    "reflection": 3, "entity": 4, "meta": 5,
}

# Feature names (order must match train_reranker.py exactly)
FEATURE_NAMES = [
    "fts_rank", "vec_rank", "theme_rank", "ppr_score",
    "fts_bm25", "vec_dist", "theme_overlap",
    "fb_last", "fb_mean", "fb_count",
    "hebbian_pmi",
    "category", "priority", "age_days", "token_count",
    "edge_count", "theme_count", "confidence",
]

# Module-level model cache
_model_cache: dict = {"model": None, "loaded": False}


def load_model(path: Path | None = None) -> dict | None:
    """Load pickled reranker model. Returns model dict or None.

    The model dict contains:
      - "model": trained LGBMRegressor or LGBMRanker
      - "feature_names": list of feature names
      - "params": production params used during training
      - "train_results": dict with importances, etc.
    """
    model_path = path or MODEL_PATH
    if not model_path.exists():
        logger.info("No reranker model at %s — using formula scoring", model_path)
        return None

    try:
        with open(model_path, "rb") as f:
            model_dict = pickle.load(f)
        # Validate it has the expected structure
        if "model" not in model_dict:
            logger.warning("Reranker model file missing 'model' key")
            return None
        logger.info("Reranker model loaded from %s", model_path)
        _model_cache["model"] = model_dict
        _model_cache["loaded"] = True
        return model_dict
    except Exception as e:
        logger.warning("Failed to load reranker model: %s", e)
        return None


def get_model() -> dict | None:
    """Get cached model, loading on first call."""
    if not _model_cache["loaded"]:
        load_model()
    return _model_cache["model"]


def extract_live_features(
    db: sqlite3.Connection,
    vec_ranked: dict[str, int],
    fts_ranked: dict[str, int],
    fts_scores: dict[str, float],
    vec_scores: dict[str, float],
    all_ids: set[str],
    ppr_scores: dict[str, float],
    feedback_map: dict[str, dict],
    themes_map: dict[str, str],
    query: str,
) -> tuple[np.ndarray, list[str]]:
    """Build (N, 18) feature matrix from live retrieval data.

    Produces identical features to train_reranker.py:extract_features_for_query().

    Args:
        db: Database connection for metadata queries
        vec_ranked: memory_id -> 0-based rank from vector search
        fts_ranked: memory_id -> 0-based rank from FTS search
        fts_scores: memory_id -> raw BM25 score (negative, lower = better match)
        vec_scores: memory_id -> raw cosine distance
        all_ids: Union of all candidate memory IDs
        ppr_scores: memory_id -> raw PPR score from graph walk
        feedback_map: memory_id -> {"count": N, "ewma": F} (from rrf_fuse)
        themes_map: memory_id -> JSON themes string
        query: The FTS query text (for theme overlap computation)

    Returns:
        (features, candidate_ids) where features is (N, 18) float32 array
    """
    if not all_ids:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32), []

    # --- Theme channel (rank by query-token overlap) ---
    # Iterate all memories in themes_map (not just all_ids) to match training,
    # which scans the entire active corpus for theme overlap.
    query_tokens = set(query.lower().split())
    theme_scored = []
    theme_overlap_map: dict[str, int] = {}
    for mid, raw_themes in themes_map.items():
        if not raw_themes:
            continue
        try:
            mem_themes = set(json.loads(raw_themes))
        except (json.JSONDecodeError, TypeError):
            continue
        mem_themes_lower = {t.lower() for t in mem_themes}
        overlap = len(mem_themes_lower & query_tokens)
        if overlap > 0:
            theme_scored.append((mid, overlap))
            theme_overlap_map[mid] = overlap

    theme_scored.sort(key=lambda x: (-x[1], x[0]))
    theme_ranked = {mid: rank for rank, (mid, _) in enumerate(theme_scored)}

    # --- Expand candidate pool with PPR and theme candidates ---
    candidate_ids_set = set(all_ids) | set(ppr_scores.keys()) | set(theme_ranked.keys())

    # --- Hebbian PMI (raw, no scaling) ---
    hebbian_pmi_map = _compute_hebbian_pmi(db, candidate_ids_set, fts_ranked, vec_ranked)

    # --- Memory metadata (batch query) ---
    metadata = _load_candidate_metadata(db, candidate_ids_set)

    # --- Raw feedback (need individual utilities, not just EWMA) ---
    feedback_raw = _load_raw_feedback(db, candidate_ids_set)

    # --- Build feature matrix ---
    candidate_list = sorted(candidate_ids_set)
    n = len(candidate_list)
    n_features = len(FEATURE_NAMES)
    features = np.zeros((n, n_features), dtype=np.float32)

    for i, mid in enumerate(candidate_list):
        # Retrieval signals
        features[i, 0] = fts_ranked.get(mid, -1)          # fts_rank
        features[i, 1] = vec_ranked.get(mid, -1)          # vec_rank
        features[i, 2] = theme_ranked.get(mid, -1)        # theme_rank
        features[i, 3] = ppr_scores.get(mid, 0.0)         # ppr_score
        features[i, 4] = fts_scores.get(mid, 0.0)         # fts_bm25
        features[i, 5] = vec_scores.get(mid, 0.0)         # vec_dist
        features[i, 6] = theme_overlap_map.get(mid, 0)    # theme_overlap

        # Feedback signals (raw, no EWMA dependency)
        fb = feedback_raw.get(mid)
        if fb and fb["count"] > 0:
            features[i, 7] = fb["utilities"][-1]           # fb_last
            features[i, 8] = sum(fb["utilities"]) / fb["count"]  # fb_mean
            features[i, 9] = fb["count"]                   # fb_count
        else:
            features[i, 7] = -1.0                          # fb_last (sentinel)
            features[i, 8] = -1.0                          # fb_mean (sentinel)
            features[i, 9] = 0                             # fb_count

        # Graph signals
        features[i, 10] = hebbian_pmi_map.get(mid, 0.0)   # hebbian_pmi

        # Memory metadata
        meta = metadata.get(mid)
        if meta:
            features[i, 11] = meta["category"]
            features[i, 12] = meta["priority"]
            features[i, 13] = meta["age_days"]
            features[i, 14] = meta["token_count"]
            features[i, 15] = meta["edge_count"]
            features[i, 16] = meta["theme_count"]
            features[i, 17] = meta["confidence"]
        else:
            features[i, 11] = 1    # semantic default
            features[i, 12] = 5
            features[i, 13] = 0.0
            features[i, 14] = 200
            features[i, 15] = 0
            features[i, 16] = 0
            features[i, 17] = 0.5

    return features, candidate_list


def rerank(model_dict: dict, features: np.ndarray, candidate_ids: list[str]) -> dict[str, float]:
    """Run model.predict() and return scores keyed by memory ID.

    Returns dict[memory_id, predicted_score] sorted by score descending.
    """
    model = model_dict["model"]
    # Use booster directly to avoid sklearn wrapper's feature-name validation
    # (the training script saves numpy arrays, but sklearn expects DataFrames)
    try:
        preds = model.booster_.predict(features)
    except (AttributeError, TypeError):
        preds = model.predict(features)
    # Clip for regressor (ranker scores are unbounded but ordering is what matters)
    if hasattr(model, '_objective_type') and 'regression' in str(getattr(model, '_objective_type', '')):
        preds = np.clip(preds, 0, 1)

    return {mid: float(score) for mid, score in zip(candidate_ids, preds)}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_hebbian_pmi(
    db: sqlite3.Connection,
    candidate_ids: set[str],
    fts_ranked: dict[str, int],
    vec_ranked: dict[str, int],
) -> dict[str, float]:
    """Compute raw Hebbian PMI for candidates (no scaling — model learns weight).

    Seeds = top-5 by best available rank (min of fts/vec rank).
    Matches train_reranker.py logic exactly.
    """
    # Get co-retrieval data from event log (last 30 days)
    all_candidate_list = list(candidate_ids)
    if not all_candidate_list:
        return {}

    hebb_ph = ",".join("?" * len(all_candidate_list))
    lookback = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    hebb_rows = db.execute(f"""
        SELECT query, memory_id FROM memory_events
        WHERE event_type = 'retrieved'
          AND memory_id IN ({hebb_ph})
          AND created_at > ?
    """, all_candidate_list + [lookback]).fetchall()

    # Build query->memories and memory->queries maps
    hebb_query_mems: dict[str, set[str]] = {}
    hebb_mem_freq: dict[str, set[str]] = {}
    for r in hebb_rows:
        q, mid = r["query"], r["memory_id"]
        if q not in hebb_query_mems:
            hebb_query_mems[q] = set()
        hebb_query_mems[q].add(mid)
        if mid not in hebb_mem_freq:
            hebb_mem_freq[mid] = set()
        hebb_mem_freq[mid].add(q)

    total_queries = len(hebb_query_mems)
    if total_queries < 5:
        return {}

    hebb_mem_count = {mid: len(qs) for mid, qs in hebb_mem_freq.items()}

    # Seeds = top-5 by best available rank
    def _best_rank(mid: str) -> int:
        ranks = []
        if mid in fts_ranked:
            ranks.append(fts_ranked[mid])
        if mid in vec_ranked:
            ranks.append(vec_ranked[mid])
        return min(ranks) if ranks else 9999

    seed_ids = sorted(candidate_ids, key=_best_rank)[:5]
    seed_set = set(seed_ids)

    hebbian_pmi_map: dict[str, float] = {}
    for candidate in candidate_ids:
        if candidate in seed_set:
            continue
        total_pmi = 0.0
        for seed in seed_ids:
            if seed not in hebb_mem_count or candidate not in hebb_mem_count:
                continue
            joint = len(hebb_mem_freq.get(seed, set()) &
                       hebb_mem_freq.get(candidate, set()))
            if joint < HEBBIAN_MIN_JOINT:
                continue
            p_s = hebb_mem_count[seed] / total_queries
            p_c = hebb_mem_count[candidate] / total_queries
            p_j = joint / total_queries
            if p_s * p_c == 0:
                continue
            pmi = math.log2(p_j / (p_s * p_c))
            if pmi > 0:
                total_pmi += pmi
        if total_pmi > 0:
            hebbian_pmi_map[candidate] = total_pmi

    return hebbian_pmi_map


def _load_candidate_metadata(
    db: sqlite3.Connection,
    candidate_ids: set[str],
) -> dict[str, dict]:
    """Load per-memory metadata for feature extraction."""
    if not candidate_ids:
        return {}

    ph = ",".join("?" * len(candidate_ids))
    id_list = list(candidate_ids)

    rows = db.execute(f"""
        SELECT id, category, base_priority, created_at, token_count,
               confidence, themes
        FROM memories WHERE id IN ({ph})
    """, id_list).fetchall()

    now_utc = datetime.now(timezone.utc)
    meta: dict[str, dict] = {}
    for r in rows:
        mid = r["id"]
        # Age in days
        try:
            created = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
            age_days = (now_utc - created).total_seconds() / 86400
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
    edge_rows = db.execute(f"""
        SELECT source_id, target_id FROM memory_edges
        WHERE source_id IN ({ph}) OR target_id IN ({ph})
    """, id_list + id_list).fetchall()

    edge_counts: dict[str, int] = {}
    for r in edge_rows:
        s, t = r["source_id"], r["target_id"]
        if s in candidate_ids:
            edge_counts[s] = edge_counts.get(s, 0) + 1
        if t in candidate_ids:
            edge_counts[t] = edge_counts.get(t, 0) + 1

    for mid in meta:
        meta[mid]["edge_count"] = edge_counts.get(mid, 0)

    return meta


def _load_raw_feedback(
    db: sqlite3.Connection,
    candidate_ids: set[str],
) -> dict[str, dict]:
    """Load raw feedback utility sequences (not EWMA — model learns its own weighting).

    Returns dict[memory_id, {"utilities": [float, ...], "count": int}].
    Matches train_reranker.py's feedback_raw format.
    """
    if not candidate_ids:
        return {}

    ph = ",".join("?" * len(candidate_ids))
    id_list = list(candidate_ids)

    fb_rows = db.execute(f"""
        SELECT memory_id, context FROM memory_events
        WHERE memory_id IN ({ph})
          AND event_type = 'feedback'
          AND context IS NOT NULL
        ORDER BY created_at ASC
    """, id_list).fetchall()

    feedback_raw: dict[str, dict] = {}
    for r in fb_rows:
        try:
            ctx = json.loads(r["context"])
            if "utility" not in ctx:
                continue
            mid = r["memory_id"]
            if mid not in feedback_raw:
                feedback_raw[mid] = {"utilities": [], "count": 0}
            feedback_raw[mid]["utilities"].append(ctx["utility"])
            feedback_raw[mid]["count"] += 1
        except (json.JSONDecodeError, TypeError):
            continue

    return feedback_raw
