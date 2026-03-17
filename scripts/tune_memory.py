# /// script
# requires-python = ">=3.11"
# dependencies = ["sqlite-vec>=0.1.6", "openai>=2.0.0", "optuna>=4.0", "numpy>=1.26", "scikit-learn>=1.0", "matplotlib>=3.8", "tiktoken>=0.7.0"]
# ///
"""
Memory tuning script -- supersedes research_scoring_calibration.py (Experiment 7).

Tunes all 41 constants in memory/constants.py across 7 groups:
  Group 1: Score Weights (Optuna, 13 params) -- requires feedback history
  Group 2: Quality Floor (stub)
  Group 3: Confidence Learning (stub)
  Group 4: Decay Model (numpy fit) -- requires feedback history
  Group 5: Adjacency Capacity (stub)
  Group 6: Dedup (stub)
  Group 7: Feedback Processing (stub)

Usage:
  uv run scripts/tune_memory.py                              # staged pipeline (default): core -> penalties -> ... -> intent
  uv run scripts/tune_memory.py --threshold 0.5 --trials 300 # staged at different threshold/budget
  uv run scripts/tune_memory.py --stage core                 # just Stage 1: rrf_k, feedback_coeff, feedback_min_count, theme_boost
  uv run scripts/tune_memory.py --stage penalties             # just Stage 2a: shadow_coeff, confidence_weight
  uv run scripts/tune_memory.py --stage all                  # full 15D search (old behavior)
  uv run scripts/tune_memory.py --apply                      # write changes to constants.py
  uv run scripts/tune_memory.py --reset                      # delete existing studies and start fresh
"""

import argparse
import difflib
import hashlib
import json
import math
import os
import pickle
import re
import sqlite3
import struct
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup: import from memory package
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from memory.constants import DATA_DIR
from memory import DB_PATH

from memory.constants import (
    RRF_K, RRF_VEC_WEIGHT, FEEDBACK_COEFF,
    HEBBIAN_COEFF, HEBBIAN_CAP, HEBBIAN_MIN_JOINT,
    ADJACENCY_BASE_BOOST, ADJACENCY_NOVELTY_FLOOR,
    CONTEXT_RELEVANCE_THRESHOLD,
    THEME_BOOST,
    CATEGORY_DECAY_RATES, DEFAULT_DECAY_RATE,
)
# Legacy cliff constants — removed from scoring pipeline, hardcoded for tuning compat
CLIFF_Z_THRESHOLD = 2.0
CLIFF_MIN_RESULTS = 5
FEEDBACK_MIN_COUNT = 1  # Legacy -- kept for loading existing studies. Now replaced by Beta prior.
# Removed from scoring -- hardcode to zero/no-op for tuning compatibility
SHADOW_PENALTY_COEFF = 0.0
CONFIDENCE_WEIGHT = 0.0
INTENT_BOOST = 1.0  # multiplier of 1 = no effect
QUALITY_FLOOR_RATIO = 0.0
INTENT_EDGE_PREFS = {}

CONSTANTS_PATH = Path(__file__).resolve().parent.parent / "src" / "memory" / "constants.py"
OPENAI_API_KEY_PATH = DATA_DIR / "openai_api_key"
EMBEDDING_MODEL = "text-embedding-3-small"

# Current production values for Group 1 (15 params)
CURRENT_PARAMS = {
    "rrf_k": RRF_K,
    "rrf_vec_weight": RRF_VEC_WEIGHT,
    "feedback_coeff": FEEDBACK_COEFF,
    "shadow_coeff": SHADOW_PENALTY_COEFF,
    "confidence_weight": CONFIDENCE_WEIGHT,
    "hebbian_coeff": HEBBIAN_COEFF,
    "hebbian_cap": HEBBIAN_CAP,
    "hebbian_scale": HEBBIAN_CAP,  # soft cap: defaults to same as hard cap for baseline
    "adjacency_base": ADJACENCY_BASE_BOOST,
    "novelty_floor": ADJACENCY_NOVELTY_FLOOR,
    "context_threshold": CONTEXT_RELEVANCE_THRESHOLD,
    "theme_boost": THEME_BOOST,
    "feedback_min_count": FEEDBACK_MIN_COUNT,
    "intent_boost": INTENT_BOOST,
    "quality_floor_ratio": QUALITY_FLOOR_RATIO,
    "cliff_z_threshold": CLIFF_Z_THRESHOLD,
    "cliff_min_results": CLIFF_MIN_RESULTS,
}

# Map from Optuna param names to constants.py names
GROUP1_MAP = {
    "rrf_k": "RRF_K",
    "rrf_vec_weight": "RRF_VEC_WEIGHT",
    "feedback_coeff": "FEEDBACK_COEFF",
    "shadow_coeff": "SHADOW_PENALTY_COEFF",
    "confidence_weight": "CONFIDENCE_WEIGHT",
    "hebbian_coeff": "HEBBIAN_COEFF",
    "hebbian_cap": "HEBBIAN_CAP",
    "hebbian_scale": "HEBBIAN_CAP",  # soft cap maps to same constant
    "adjacency_base": "ADJACENCY_BASE_BOOST",
    "novelty_floor": "ADJACENCY_NOVELTY_FLOOR",
    "context_threshold": "CONTEXT_RELEVANCE_THRESHOLD",
    "theme_boost": "THEME_BOOST",
    "feedback_min_count": "FEEDBACK_MIN_COUNT",
    "intent_boost": "INTENT_BOOST",
    "quality_floor_ratio": "QUALITY_FLOOR_RATIO",
    "cliff_z_threshold": "CLIFF_Z_THRESHOLD",
    "cliff_min_results": "CLIFF_MIN_RESULTS",
}

# Search ranges per param (lo, hi, type)
# Wide enough to see where things go pathological, not just the good region.
#   rrf_k: 1/(k+rank+1). k=1 → sharp rank weighting, k=500 → nearly flat.
#   rrf_vec_weight: w*vec_rrf + (1-w)*kw_rrf. 0=keyword only, 1=vector only, 0.5=equal (baseline).
#   feedback_coeff: score *= (1 + utility * coeff). utility ∈ [0,1]. coeff=10 → 11x boost.
#   shadow_coeff: score -= coeff * shadow². shadow ∈ [0,~5]. coeff=0.5 → -12.5 at shadow=5.
#   confidence_weight: score += weight * (conf - 0.5). conf ∈ [0,1]. weight=2 → ±1.0 swing.
#   hebbian_coeff: PMI * coeff, capped. PMI ∈ [0,~5]. coeff=0.1 → 0.5/pair.
#   hebbian_cap: absolute cap on hebbian boost per candidate (hard cap mode).
#   hebbian_scale: magnitude of soft cap. boost = scale * log(1 + coeff * PMI). Used in softcap stage.
#   adjacency_base: neighbor_score = base * novelty * max_rrf. base=1 → full rrf score.
#   novelty_floor: minimum novelty score (0-1). floor=1 → no novelty filtering.
#   context_threshold: cosine threshold for context relevance. 0 = accept all, 1 = exact.
#   theme_boost: score *= (1 + boost * overlap). overlap ∈ [1,3]. boost=3 → up to 10x.
#   feedback_min_count: REMOVED — replaced by empirical Bayes Beta prior.
#   intent_boost: multiplier for intent-matching edges. 1 = no effect.
#   quality_floor_ratio: drop results below floor * top_score. 0 = keep all, 1 = only top.
#   cliff_z_threshold: z-score for cliff detection. 0.5 = aggressive, 5 = lenient.
#   cliff_min_results: min results before cliff detection starts.
SEARCH_RANGES = {
    # (lo, hi, type[, "log"])  — "log" enables log-uniform sampling
    "rrf_k": (1, 30, "int", "log"),
    "rrf_vec_weight": (0.1, 0.9, "float"),
    "feedback_coeff": (0.0, 10.0, "float"),
    "shadow_coeff": (0.0, 0.5, "float"),
    "confidence_weight": (0.0, 2.0, "float"),
    "hebbian_coeff": (0.0, 4.0, "float"),
    "hebbian_cap": (0.0, 1.0, "float"),
    "hebbian_scale": (0.0, 0.5, "float"),
    "adjacency_base": (0.0, 2.0, "float"),
    "novelty_floor": (0.0, 1.0, "float"),
    "context_threshold": (0.0, 1.0, "float"),
    "theme_boost": (0.0, 3.0, "float"),
    # feedback_min_count: REMOVED — replaced by empirical Bayes Beta prior
    "intent_boost": (0.5, 5.0, "float"),
    "quality_floor_ratio": (0.0, 1.0, "float"),
    "cliff_z_threshold": (0.5, 5.0, "float"),
    "cliff_min_results": (1, 20, "int"),
}

# Staged tuning: --stage selects which params to search (rest fixed at current)
STAGES = {
    "core": {
        "desc": "Core params (4D): rrf_k, rrf_vec_weight, feedback_coeff, theme_boost",
        "search": ["rrf_k", "rrf_vec_weight", "feedback_coeff", "theme_boost"],
    },
    "penalties": {
        "desc": "Score adjustments (2D): shadow_coeff, confidence_weight",
        "search": ["shadow_coeff", "confidence_weight"],
    },
    "hebbian": {
        "desc": "Co-retrieval (2D): hebbian_coeff, hebbian_cap",
        "search": ["hebbian_coeff", "hebbian_cap"],
    },
    "adjacency": {
        "desc": "Graph expansion (2D): adjacency_base, novelty_floor",
        "search": ["adjacency_base", "novelty_floor"],
    },
    "cutoff": {
        "desc": "Output filtering (1D): quality_floor_ratio",
        "search": ["quality_floor_ratio"],
    },
    "intent": {
        "desc": "Intent boost (1D): intent_boost — DISABLED (inert)",
        "search": ["intent_boost"],
    },
    "all": {
        "desc": "All non-inert params (11D search)",
        "search": [k for k in CURRENT_PARAMS.keys()
                   if k not in ("cliff_z_threshold", "cliff_min_results",
                                "context_threshold", "intent_boost",
                                "feedback_min_count")],
    },
    "joint": {
        "desc": "Post-repair joint (8D): rrf_k, feedback_coeff, theme_boost, hebbian_coeff, hebbian_cap, adjacency_base, novelty_floor, context_threshold",
        "search": ["rrf_k", "feedback_coeff", "theme_boost",
                    "hebbian_coeff", "hebbian_cap", "adjacency_base", "novelty_floor",
                    "context_threshold"],
    },
    "joint_tight": {
        "desc": "Tight 6D: lock adjacency_base + context_threshold at production values",
        "search": ["rrf_k", "feedback_coeff", "theme_boost",
                    "hebbian_coeff", "hebbian_cap", "novelty_floor"],
    },
    "joint_core": {
        "desc": "Core 4D: the 3 params that matter + hebbian_coeff. Lock feedback, novelty, adjacency, context.",
        "search": ["rrf_k", "theme_boost", "hebbian_cap", "hebbian_coeff"],
    },
    "mrr_k6": {
        "desc": "MRR-optimized 3D: theme_boost, hebbian_cap, hebbian_coeff. rrf_k locked at 6.",
        "search": ["theme_boost", "hebbian_cap", "hebbian_coeff"],
    },
    "settle_k6": {
        "desc": "Settle wm9 holdovers at k=6: feedback_coeff, adjacency_base, adjacency_novelty_floor.",
        "search": ["feedback_coeff", "adjacency_base", "novelty_floor"],
    },
    "zombie": {
        "desc": "Joint 8D + 4 zombies (12D): check if removed params wake up at new rrf_k regime",
        "search": ["rrf_k", "feedback_coeff", "theme_boost",
                    "hebbian_coeff", "hebbian_cap", "adjacency_base", "novelty_floor",
                    "context_threshold",
                    "shadow_coeff", "confidence_weight", "intent_boost", "quality_floor_ratio"],
    },
    "softcap": {
        "desc": "Soft Hebbian cap (6D): log(1+coeff*PMI) replaces min(cap, coeff*PMI) + shadow recheck",
        "search": ["hebbian_scale", "hebbian_coeff", "rrf_k", "theme_boost",
                    "shadow_coeff", "feedback_coeff"],
        "softcap": True,  # flag to switch scoring formula
    },
    "softcap_fixed_k": {
        "desc": "Soft Hebbian cap (4D): tanh(coeff*PMI/scale), rrf_k fixed at production",
        "search": ["hebbian_scale", "hebbian_coeff", "theme_boost",
                    "feedback_coeff"],
        "softcap": True,
    },
    "joint5": {
        "desc": "Reduced joint (5D): rrf_k, feedback_coeff, theme_boost, hebbian_coeff, hebbian_cap (adj/novelty locked)",
        "search": ["rrf_k", "feedback_coeff", "theme_boost",
                    "hebbian_coeff", "hebbian_cap"],
    },
    "vec_weight": {
        "desc": "Vector weight study (1D): rrf_vec_weight only, everything else locked at production",
        "search": ["rrf_vec_weight"],
    },
    "core_weighted": {
        "desc": "Core + vec weight (5D): rrf_k, rrf_vec_weight, feedback_coeff, theme_boost, hebbian_cap",
        "search": ["rrf_k", "rrf_vec_weight", "feedback_coeff", "theme_boost", "hebbian_cap"],
    },
    "staged": {
        "desc": "Sequential pipeline: core -> penalties -> hebbian -> adjacency -> cutoff",
        "search": [],  # not used directly — triggers pipeline mode
        "sequence": ["core", "penalties", "hebbian", "adjacency", "cutoff"],
    },
}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def serialize_f32(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def deserialize_f32(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def get_db() -> sqlite3.Connection:
    import sqlite_vec
    db = sqlite3.connect(str(DB_PATH))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.row_factory = sqlite3.Row
    return db


def sanitize_fts_query(query: str) -> str:
    cleaned = query.replace('"', "").replace("'", "")
    tokens = cleaned.split()
    if not tokens:
        return '""'
    return " OR ".join(f'"{t}"' for t in tokens)


def embed_batch(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and OPENAI_API_KEY_PATH.exists():
        api_key = OPENAI_API_KEY_PATH.read_text().strip()
    client = OpenAI(api_key=api_key)
    all_embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]
        response = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
        sorted_data = sorted(response.data, key=lambda d: d.index)
        all_embeddings.extend([d.embedding for d in sorted_data])
    return all_embeddings


def mrr(ranked_ids: list[str], relevant_ids: set) -> float:
    for i, mid in enumerate(ranked_ids):
        if mid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


# Vector math

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _vec_sub(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def _vec_scale(v: list[float], s: float) -> list[float]:
    return [x * s for x in v]


def _norm(v: list[float]) -> float:
    return sum(x * x for x in v) ** 0.5


def _novelty_score(query: list[float], seed: list[float], neighbor: list[float]) -> float:
    ns_dot = _dot(neighbor, seed)
    proj = _vec_scale(seed, ns_dot)
    residual = _vec_sub(neighbor, proj)
    r_norm = _norm(residual)
    if r_norm < 1e-8:
        return 0.0
    qr_dot = _dot(query, residual)
    return max(0.0, min(1.0, qr_dot / r_norm))


def detect_intent(query: str) -> set[str]:
    q = query.lower().strip()
    for prefix, prefs in INTENT_EDGE_PREFS.items():
        if q.startswith(prefix):
            return prefs
    return set()


# ---------------------------------------------------------------------------
# Search functions
# ---------------------------------------------------------------------------


def run_fts_search(db: sqlite3.Connection, query: str, limit: int = 20) -> list[str]:
    fts_query = sanitize_fts_query(query)
    try:
        results = db.execute(
            """SELECT rowid, bm25(memory_fts, 5.0, 3.0) as score
               FROM memory_fts WHERE memory_fts MATCH ?
               ORDER BY score LIMIT ?""",
            (fts_query, limit),
        ).fetchall()
    except Exception:
        return []

    memory_ids = []
    for row in results:
        mapped = db.execute(
            "SELECT memory_id FROM memory_rowid_map WHERE rowid = ?",
            (row["rowid"],),
        ).fetchone()
        if mapped:
            active = db.execute(
                "SELECT id FROM memories WHERE id = ? AND status = 'active'",
                (mapped["memory_id"],),
            ).fetchone()
            if active:
                memory_ids.append(mapped["memory_id"])
    return memory_ids


def run_vec_search(db: sqlite3.Connection, query_embedding: list[float],
                   limit: int = 50) -> list[str]:
    results = db.execute(
        """SELECT rowid, distance FROM memory_vec
           WHERE embedding MATCH ? AND k = ?
           ORDER BY distance""",
        (serialize_f32(query_embedding), limit),
    ).fetchall()

    memory_ids = []
    for row in results:
        mapped = db.execute(
            "SELECT memory_id FROM memory_rowid_map WHERE rowid = ?",
            (row["rowid"],),
        ).fetchone()
        if mapped:
            active = db.execute(
                "SELECT id FROM memories WHERE id = ? AND status = 'active'",
                (mapped["memory_id"],),
            ).fetchone()
            if active:
                memory_ids.append(mapped["memory_id"])
    return memory_ids


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def extract_queries_and_feedback(db: sqlite3.Connection):
    """Extract unique queries and build feedback ground truth.

    Returns raw feedback (no min-count filter) so feedback_min_count is tunable.
    """
    rows = db.execute("""
        SELECT query, COUNT(*) as n
        FROM memory_events WHERE event_type='retrieved'
        AND query IS NOT NULL AND query != ''
        GROUP BY query ORDER BY n DESC
    """).fetchall()
    queries = [{"query": r["query"], "count": r["n"]} for r in rows]

    # Extract vector_input from recall_meta events (context used for vec search)
    vector_input_map = {}
    meta_rows = db.execute("""
        SELECT query, context FROM memory_events
        WHERE event_type='recall_meta' AND context IS NOT NULL
    """).fetchall()
    for r in meta_rows:
        try:
            ctx = json.loads(r["context"])
            if "vector_input" in ctx:
                vector_input_map[r["query"]] = ctx["vector_input"]
        except (json.JSONDecodeError, TypeError):
            continue
    for q in queries:
        q["vector_input"] = vector_input_map.get(q["query"], q["query"])

    fb_rows = db.execute("""
        SELECT memory_id, context FROM memory_events
        WHERE event_type='feedback' AND context IS NOT NULL
    """).fetchall()

    # Raw feedback: no min-count filter (tunable via feedback_min_count)
    feedback_raw = defaultdict(lambda: {"utility_sum": 0.0, "count": 0})
    skipped_old = 0
    for r in fb_rows:
        try:
            ctx = json.loads(r["context"])
            if "utility" not in ctx:
                skipped_old += 1
                continue
            mid = r["memory_id"]
            feedback_raw[mid]["utility_sum"] += ctx["utility"]
            feedback_raw[mid]["count"] += 1
        except (json.JSONDecodeError, TypeError, KeyError):
            continue

    for mid, fb in feedback_raw.items():
        fb["mean"] = fb["utility_sum"] / fb["count"] if fb["count"] > 0 else 0.0

    query_memories = defaultdict(set)
    ret_rows = db.execute("""
        SELECT query, memory_id FROM memory_events
        WHERE event_type='retrieved' AND query IS NOT NULL
    """).fetchall()
    for r in ret_rows:
        query_memories[r["query"]].add(r["memory_id"])

    print(f"  Queries: {len(queries)} unique")
    print(f"  Memories with feedback: {len(feedback_raw)}")
    print(f"  Feedback events: {sum(fb['count'] for fb in feedback_raw.values())}")
    print(f"  Old events skipped (no utility key): {skipped_old}")

    return queries, dict(feedback_raw), dict(query_memories)


def load_scoring_data(db: sqlite3.Connection):
    """Load shadow and confidence maps for all active memories."""
    shadow_map = {}
    confidence_map = {}
    for row in db.execute(
        "SELECT id, shadow_load, confidence FROM memories WHERE status = 'active'"
    ):
        if row["shadow_load"] and row["shadow_load"] > 0:
            shadow_map[row["id"]] = row["shadow_load"]
        confidence_map[row["id"]] = row["confidence"] if row["confidence"] is not None else 0.5

    print(f"  Memories with shadow > 0: {len(shadow_map)}")
    print(f"  Confidence range: {min(confidence_map.values()):.2f} - {max(confidence_map.values()):.2f}")

    return shadow_map, confidence_map


def load_themes_map(db: sqlite3.Connection) -> dict[str, set[str]]:
    """Load themes for all active memories."""
    themes_map = {}
    for row in db.execute(
        "SELECT id, themes FROM memories WHERE status = 'active'"
    ):
        if row["themes"]:
            try:
                themes_map[row["id"]] = set(json.loads(row["themes"]))
            except (json.JSONDecodeError, TypeError):
                themes_map[row["id"]] = set()
        else:
            themes_map[row["id"]] = set()
    return themes_map


def load_hebbian_data(db: sqlite3.Connection):
    """Build co-retrieval matrix for Hebbian PMI computation."""
    rows = db.execute("""
        SELECT query, memory_id FROM memory_events
        WHERE event_type = 'retrieved' AND query IS NOT NULL AND query != ''
    """).fetchall()

    hebb_query_mems = defaultdict(set)
    hebb_mem_freq = defaultdict(set)
    for r in rows:
        q, mid = r["query"], r["memory_id"]
        hebb_query_mems[q].add(mid)
        hebb_mem_freq[mid].add(q)

    hebb_total_queries = len(hebb_query_mems)
    print(f"  Hebbian: {hebb_total_queries} unique queries, {len(hebb_mem_freq)} memories")

    return dict(hebb_mem_freq), dict(hebb_query_mems), hebb_total_queries


def load_edge_data(db: sqlite3.Connection):
    """Preload edges (with edge_type + weight) and memory embeddings."""
    edge_rows = db.execute("""
        SELECT source_id, target_id, linking_context, linking_embedding,
               edge_type, weight
        FROM memory_edges
    """).fetchall()

    # edges_by_memory: {mid: [(neighbor_id, linking_context, linking_embedding, edge_type, weight)]}
    edges_by_memory = defaultdict(list)
    for row in edge_rows:
        src, tgt = row["source_id"], row["target_id"]
        entry = (row["linking_context"], row["linking_embedding"],
                 row["edge_type"], row["weight"])
        edges_by_memory[src].append((tgt, *entry))
        edges_by_memory[tgt].append((src, *entry))

    # edge_types_map: {(src, tgt): edge_type}
    edge_types_map = {}
    for row in edge_rows:
        edge_types_map[(row["source_id"], row["target_id"])] = row["edge_type"]

    # All active memory embeddings
    memory_embeddings = {}
    for row in db.execute("""
        SELECT rm.memory_id, mv.embedding
        FROM memory_rowid_map rm
        JOIN memory_vec mv ON rm.rowid = mv.rowid
    """):
        memory_embeddings[row["memory_id"]] = deserialize_f32(row["embedding"])

    print(f"  Edges: {len(edge_rows)} total, {len(edges_by_memory)} memories with edges")
    print(f"  Memory embeddings: {len(memory_embeddings)}")

    return dict(edges_by_memory), memory_embeddings, edge_types_map


# ---------------------------------------------------------------------------
# Group 1: Score Weights (Optuna, 13 params)
# ---------------------------------------------------------------------------

_BETA_FALLBACK_MEAN = 0.25
_BETA_FALLBACK_STRENGTH = 1.0
_BETA_MIN_MEMORIES = 5


def _compute_beta_prior_from_raw(feedback_raw: dict[str, dict]) -> tuple[float, float]:
    """Compute empirical Bayes Beta prior from pre-loaded feedback data.

    Same logic as scoring._compute_beta_prior but operates on the in-memory
    feedback_raw dict instead of querying the database.

    Returns (prior_mean, prior_strength) where prior_strength = a + b.
    """
    if not feedback_raw:
        return _BETA_FALLBACK_MEAN, _BETA_FALLBACK_STRENGTH

    means_2plus = [fb["utility_sum"] / fb["count"]
                   for fb in feedback_raw.values() if fb["count"] >= 2]

    if len(means_2plus) < _BETA_MIN_MEMORIES:
        total_sum = sum(fb["utility_sum"] for fb in feedback_raw.values())
        total_count = sum(fb["count"] for fb in feedback_raw.values())
        mu = total_sum / total_count if total_count else _BETA_FALLBACK_MEAN
        return mu, _BETA_FALLBACK_STRENGTH

    mu = sum(means_2plus) / len(means_2plus)
    var = sum((m - mu) ** 2 for m in means_2plus) / len(means_2plus)

    if var <= 0 or var >= mu * (1 - mu):
        return mu, _BETA_FALLBACK_STRENGTH

    strength = mu * (1 - mu) / var - 1
    return mu, strength


def evaluate_pipeline(
    params: dict,
    queries: list[dict],
    fts_results: dict[str, list[str]],
    vec_results: dict[str, list[str]],
    query_embeddings: dict[str, list[float]],
    feedback_raw: dict[str, dict],
    shadow_map: dict[str, float],
    confidence_map: dict[str, float],
    themes_map: dict[str, set[str]],
    hebb_mem_freq: dict[str, set],
    hebb_total_queries: int,
    edges_by_memory: dict[str, list],
    memory_embeddings: dict[str, list[float]],
    edge_types_map: dict[tuple, str],
    active_ids: set[str],
    relevant_memories: set[str],
    query_memories: dict[str, set[str]],
    detailed: bool = False,
):
    """Full pipeline replay with given parameters.

    Returns MRR (float) when detailed=False.
    Returns (MRR, list of per-query result dicts) when detailed=True.
    """
    rrf_k = params["rrf_k"]
    rrf_vec_weight = params.get("rrf_vec_weight", 0.5)
    feedback_coeff = params["feedback_coeff"]
    shadow_coeff = params["shadow_coeff"]
    confidence_weight = params["confidence_weight"]
    hebbian_coeff = params["hebbian_coeff"]
    hebbian_cap = params["hebbian_cap"]
    hebbian_scale = params.get("hebbian_scale", 0.0)
    use_softcap = params.get("_softcap", False)
    adjacency_base = params["adjacency_base"]
    novelty_floor = params["novelty_floor"]
    context_threshold = params["context_threshold"]
    theme_boost = params["theme_boost"]
    intent_boost = params["intent_boost"]
    quality_floor_ratio = params["quality_floor_ratio"]

    # Empirical Bayes Beta prior from population feedback (method of moments)
    beta_prior_mean, beta_prior_strength = _compute_beta_prior_from_raw(feedback_raw)
    cliff_z = params.get("cliff_z_threshold", 2.0)
    cliff_min = int(params.get("cliff_min_results", 5))

    mrr_sum = 0.0
    queries_with_relevant = 0
    detailed_results = [] if detailed else None

    # Pre-compute outside query loop (query-independent)
    hebb_mem_count = {mid: len(qs) for mid, qs in hebb_mem_freq.items()} if hebb_mem_freq else {}

    for q in queries:
        qtext = q["query"]
        per_query_relevant = query_memories.get(qtext, set()) & relevant_memories
        if not per_query_relevant:
            continue
        queries_with_relevant += 1

        fts_ids = fts_results.get(qtext, [])
        vec_ids = vec_results.get(qtext, [])
        query_emb = query_embeddings.get(qtext)

        fts_ranked = {mid: rank for rank, mid in enumerate(fts_ids)}
        vec_ranked = {mid: rank for rank, mid in enumerate(vec_ids)}
        all_ids = (set(fts_ranked.keys()) | set(vec_ranked.keys())) & active_ids

        # 1. Base RRF fusion
        scores = {}
        for mid in all_ids:
            score = 0.0
            if mid in fts_ranked:
                score += (1.0 - rrf_vec_weight) / (rrf_k + fts_ranked[mid] + 1)
            if mid in vec_ranked:
                score += rrf_vec_weight / (rrf_k + vec_ranked[mid] + 1)
            scores[mid] = score

        if not scores:
            continue

        # 2. Feedback boost (empirical Bayes Beta prior, centered penalty)
        if feedback_coeff > 0:
            a = beta_prior_mean * beta_prior_strength
            b = (1 - beta_prior_mean) * beta_prior_strength
            for mid in scores:
                fb = feedback_raw.get(mid)
                if fb and fb["count"] > 0:
                    posterior = (fb["utility_sum"] + a) / (fb["count"] + a + b)
                    scores[mid] *= (1 + (posterior - beta_prior_mean) * feedback_coeff)

        # 3. Theme boost (multiplicative)
        if theme_boost > 0:
            query_tokens = set(qtext.lower().split())
            for mid in scores:
                overlap = len(themes_map.get(mid, set()) & query_tokens)
                if overlap:
                    scores[mid] *= (1 + theme_boost * overlap)

        # 4. Shadow penalty (quadratic)
        if shadow_coeff > 0:
            for mid in scores:
                if mid in shadow_map:
                    shadow = shadow_map[mid]
                    scores[mid] -= shadow_coeff * shadow * shadow

        # 5. Confidence adjustment (centered at 0.5)
        if confidence_weight > 0:
            for mid in scores:
                conf = confidence_map.get(mid, 0.5)
                scores[mid] += confidence_weight * (conf - 0.5)

        # 6. Hebbian PMI boost
        if hebbian_coeff > 0 and hebb_total_queries >= 5:
            seed_ids = sorted(scores, key=scores.get, reverse=True)[:5]

            for candidate in list(scores.keys()):
                if candidate in seed_ids:
                    continue
                total_hebb = 0.0
                for seed in seed_ids:
                    if seed not in hebb_mem_count or candidate not in hebb_mem_count:
                        continue
                    seed_qs = hebb_mem_freq.get(seed, set())
                    cand_qs = hebb_mem_freq.get(candidate, set())
                    joint = len(seed_qs & cand_qs)
                    if joint < HEBBIAN_MIN_JOINT:
                        continue
                    p_seed = hebb_mem_count[seed] / hebb_total_queries
                    p_cand = hebb_mem_count[candidate] / hebb_total_queries
                    p_joint = joint / hebb_total_queries
                    if p_seed * p_cand == 0:
                        continue
                    pmi = math.log2(p_joint / (p_seed * p_cand))
                    if pmi > 0:
                        total_hebb += min(hebbian_coeff * pmi, hebbian_cap)
                scores[candidate] += min(total_hebb, hebbian_cap)

        # 7. Adjacency expansion (with intent boost and edge weight)
        if adjacency_base > 0 and query_emb:
            max_rrf = max(scores.values()) if scores else 0
            seed_ids = sorted(scores, key=scores.get, reverse=True)[:5]
            seed_set = set(seed_ids)
            intent_prefs = detect_intent(qtext)
            neighbors_per_seed = {}
            total_new = 0
            MAX_NEIGHBORS = 5
            MAX_TOTAL = 20

            for seed_id in seed_ids:
                if seed_id not in edges_by_memory:
                    continue
                seed_vec = memory_embeddings.get(seed_id)
                if not seed_vec:
                    continue

                for neighbor_id, linking_context, linking_embedding, edge_type, edge_weight in edges_by_memory[seed_id]:
                    if neighbor_id in seed_set or neighbor_id not in active_ids:
                        continue

                    seed_neighbors = neighbors_per_seed.get(seed_id, set())
                    if len(seed_neighbors) >= MAX_NEIGHBORS:
                        break
                    if neighbor_id in seed_neighbors:
                        continue
                    is_new = neighbor_id not in scores
                    if is_new and total_new >= MAX_TOTAL:
                        continue

                    # Context relevance gate
                    context_weight_val = 1.0
                    has_context = linking_context and linking_context.strip()
                    if has_context and not linking_embedding:
                        continue
                    if linking_embedding:
                        link_emb = deserialize_f32(linking_embedding)
                        context_sim = _dot(query_emb, link_emb)
                        if context_sim < context_threshold:
                            continue
                        context_weight_val = context_sim

                    # Novelty scoring
                    neighbor_vec = memory_embeddings.get(neighbor_id)
                    if not neighbor_vec:
                        continue

                    novelty = _novelty_score(query_emb, seed_vec, neighbor_vec)
                    if novelty < novelty_floor:
                        continue

                    seed_strength = max(0.0, scores[seed_id] / max_rrf) if max_rrf > 0 else 0
                    ew = edge_weight if edge_weight is not None else 1.0
                    boost = adjacency_base * novelty * context_weight_val * seed_strength * ew

                    # Intent boost: if query intent matches edge type, amplify
                    if intent_prefs and edge_type and edge_type in intent_prefs:
                        boost *= intent_boost

                    if is_new:
                        scores[neighbor_id] = boost
                        total_new += 1
                        # Apply shadow + confidence to new neighbors
                        if shadow_coeff > 0 and neighbor_id in shadow_map:
                            s = shadow_map[neighbor_id]
                            scores[neighbor_id] -= shadow_coeff * s * s
                        if confidence_weight > 0:
                            c = confidence_map.get(neighbor_id, 0.5)
                            scores[neighbor_id] += confidence_weight * (c - 0.5)
                    else:
                        scores[neighbor_id] += boost

                    neighbors_per_seed.setdefault(seed_id, set()).add(neighbor_id)

        # 8. Quality floor + cliff detection
        ranked = sorted(scores, key=scores.get, reverse=True)
        if ranked and quality_floor_ratio > 0:
            top = scores[ranked[0]]
            if top > 0:
                safety_floor = top * quality_floor_ratio
                ranked = [mid for mid in ranked if scores[mid] >= safety_floor]
                # Rolling log-curve cliff detection
                if len(ranked) > cliff_min:
                    norm = [scores[mid] / top for mid in ranked]
                    for ci in range(cliff_min, len(norm)):
                        lx = [math.log(r) for r in range(1, ci + 1)]
                        fs = norm[:ci]
                        mx = sum(lx) / ci
                        my = sum(fs) / ci
                        ssxx = sum((x - mx) ** 2 for x in lx)
                        if ssxx < 1e-12:
                            continue
                        slope = sum((x - mx) * (y - my) for x, y in zip(lx, fs)) / ssxx
                        a, b = my - slope * mx, -slope
                        preds = [a - b * math.log(r) for r in range(1, ci + 1)]
                        residuals = [fs[j] - preds[j] for j in range(ci)]
                        rmse = math.sqrt(sum(r * r for r in residuals) / ci)
                        rmse = max(rmse, 0.005)
                        predicted = a - b * math.log(ci + 1)
                        if predicted - norm[ci] > cliff_z * rmse:
                            ranked = ranked[:ci]
                            break

        if detailed:
            # Identify best relevant memory by mean feedback utility
            best_mid, best_util = None, -1.0
            for mid in per_query_relevant:
                u = feedback_raw.get(mid, {}).get("mean", 0.0)
                if u > best_util:
                    best_util = u
                    best_mid = mid
            # Full ranked list (pre-cutoff) for no-cutoff baseline
            full_ranked = sorted(scores, key=scores.get, reverse=True)
            detailed_results.append({
                "query": qtext,
                "ranked": ranked,
                "full_ranked": full_ranked,
                "relevant": per_query_relevant,
                "best_relevant": best_mid,
                "best_utility": best_util,
            })

        mrr_sum += mrr(ranked, per_query_relevant)

    mrr_val = mrr_sum / queries_with_relevant if queries_with_relevant else 0.0
    if detailed:
        return mrr_val, detailed_results
    return mrr_val


def load_group1_data(db: sqlite3.Connection, all_thresholds: list[float]):
    """Load and precompute all data for Group 1. Shared across runs."""
    print("\n=== Group 1: Score Weights (Optuna, 15 params) ===")

    print("\n  Loading data...")
    queries, feedback_raw, query_memories = extract_queries_and_feedback(db)

    relevant_sets = {}
    for t in sorted(all_thresholds):
        relevant_sets[t] = {mid for mid, fb in feedback_raw.items() if fb["mean"] >= t}
        print(f"  Relevant memories (utility >= {t}): {len(relevant_sets[t])}")

    active_rows = db.execute("SELECT id FROM memories WHERE status = 'active'").fetchall()
    active_ids = {r["id"] for r in active_rows}
    print(f"  Active memories: {len(active_ids)}")

    token_map = dict(db.execute(
        "SELECT id, coalesce(token_count, length(content)/4) FROM memories WHERE status='active'"
    ).fetchall())
    print(f"  Token counts loaded: {len(token_map)} memories")

    print("\n  Pre-computing searches...")
    fts_results = {}
    for q in queries:
        fts_results[q["query"]] = run_fts_search(db, q["query"], limit=20)
    print(f"  FTS: {len(fts_results)} queries")

    # Embed vector_input (context when provided, else query) for faithful vec replay
    vector_texts = [q["vector_input"] for q in queries]
    unique_vec_texts = sorted(set(vector_texts))
    query_texts = sorted({q["query"] for q in queries})
    all_texts = unique_vec_texts + [t for t in query_texts if t not in set(unique_vec_texts)]

    # Cache embeddings — keyed by hash of all unique texts
    cache_dir = DATA_DIR / "tuning_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    text_hash = hashlib.sha256("\n".join(sorted(set(all_texts))).encode()).hexdigest()[:16]
    cache_path = cache_dir / f"embeddings_{text_hash}.pkl"

    if cache_path.exists():
        print(f"  Loading cached embeddings ({len(all_texts)} texts, {cache_path.name})...")
        with open(cache_path, "rb") as f:
            text_to_emb = pickle.load(f)
        # Verify cache covers all texts
        missing = [t for t in all_texts if t not in text_to_emb]
        if missing:
            print(f"  Cache miss on {len(missing)} texts, re-embedding all...")
            all_embs = embed_batch(all_texts)
            text_to_emb = dict(zip(all_texts, all_embs))
            with open(cache_path, "wb") as f:
                pickle.dump(text_to_emb, f)
    else:
        print(f"  Embedding {len(all_texts)} unique texts ({len(queries)} queries)...")
        all_embs = embed_batch(all_texts)
        text_to_emb = dict(zip(all_texts, all_embs))
        with open(cache_path, "wb") as f:
            pickle.dump(text_to_emb, f)
        print(f"  Cached to {cache_path.name}")

    vec_text_to_emb = {t: text_to_emb[t] for t in unique_vec_texts}
    query_text_to_emb = {t: text_to_emb[t] for t in query_texts}

    query_embeddings = {q["query"]: query_text_to_emb[q["query"]] for q in queries}
    vec_embeddings = {q["query"]: vec_text_to_emb[q["vector_input"]] for q in queries}

    n_with_context = sum(1 for q in queries if q["vector_input"] != q["query"])
    if n_with_context:
        print(f"  ({n_with_context} queries used separate context for vec search)")

    vec_results = {}
    for q in queries:
        vec_emb = vec_embeddings[q["query"]]
        vec_results[q["query"]] = run_vec_search(db, vec_emb, limit=50)
    print(f"  Vec: {len(vec_results)} queries")

    print("\n  Loading scoring data...")
    shadow_map, confidence_map = load_scoring_data(db)
    themes_map = load_themes_map(db)
    hebb_mem_freq, _, hebb_total_queries = load_hebbian_data(db)
    edges_by_memory, memory_embeddings, edge_types_map = load_edge_data(db)

    return {
        "queries": queries,
        "feedback_raw": feedback_raw,
        "query_memories": query_memories,
        "relevant_sets": relevant_sets,
        "active_ids": active_ids,
        "fts_results": fts_results,
        "vec_results": vec_results,
        "query_embeddings": query_embeddings,
        "shadow_map": shadow_map,
        "confidence_map": confidence_map,
        "themes_map": themes_map,
        "hebb_mem_freq": hebb_mem_freq,
        "hebb_total_queries": hebb_total_queries,
        "edges_by_memory": edges_by_memory,
        "memory_embeddings": memory_embeddings,
        "edge_types_map": edge_types_map,
        "token_map": token_map,
    }


def bootstrap_param_bounds(study, search_params: set, n_boot: int = 1000,
                           top_frac: float = 0.2) -> dict[str, tuple[float, float, float]]:
    """Bootstrap confidence intervals from top trials.

    Returns {param: (mean, std, lo_2sigma, hi_2sigma)} for searched params.
    Uses the top `top_frac` of completed trials by value.
    """
    import numpy as np
    completed = [t for t in study.trials if t.value is not None]
    # Sort so best trials come first: ascending for MINIMIZE, descending for MAXIMIZE
    ascending = study.direction.name == "MINIMIZE"
    completed.sort(key=lambda t: t.value, reverse=not ascending)
    n_top = max(5, int(len(completed) * top_frac))
    top_trials = completed[:n_top]

    param_values = {p: [t.params[p] for t in top_trials if p in t.params]
                    for p in search_params}

    bounds = {}
    rng = np.random.default_rng(42)
    for p, vals in param_values.items():
        if not vals:
            continue
        arr = np.array(vals)
        boot_means = np.array([rng.choice(arr, size=len(arr), replace=True).mean()
                               for _ in range(n_boot)])
        mean = boot_means.mean()
        std = boot_means.std()
        lo, hi = SEARCH_RANGES[p][:2]
        bounds[p] = (mean, std, max(lo, mean - 2 * std), min(hi, mean + 2 * std))
    return bounds


# ---------------------------------------------------------------------------
# Multi-worker parallelism for Optuna
# ---------------------------------------------------------------------------
# Windows uses 'spawn' for multiprocessing — closures can't be pickled.
# These must be top-level module functions.


def _worker_eval(params, data, search_params, fixed_params, metric,
                 miss_min_util, miss_power, wm_threshold):
    """Pure computation: evaluate one param set. No Optuna dependency."""
    full_params = dict(fixed_params)
    full_params.update(params)
    # When optimizing GT metrics, only evaluate GT queries (skip the rest)
    queries = data.get("gt_queries", data["queries"]) if metric in ("ndcg_5k", "graded_recall_5k") else data["queries"]
    mrr_val, detailed_results = evaluate_pipeline(
        full_params, queries, data["fts_results"], data["vec_results"],
        data["query_embeddings"], data["feedback_raw"], data["shadow_map"],
        data["confidence_map"], data["themes_map"], data["hebb_mem_freq"],
        data["hebb_total_queries"], data["edges_by_memory"],
        data["memory_embeddings"], data["edge_types_map"],
        data["active_ids"], data["relevant_sets"][wm_threshold],
        data["query_memories"], detailed=True,
    )
    miss = compute_miss_rates(detailed_results, data["token_map"], budgets=[5000])
    miss_rate = miss["best_cutoff"][0]
    wm = compute_weighted_miss(detailed_results, data["token_map"],
                               data["feedback_raw"], budget=5000,
                               min_utility=miss_min_util, power=miss_power)
    wm_auc = compute_weighted_miss_auc(detailed_results, data["token_map"],
                                        data["feedback_raw"],
                                        min_utility=miss_min_util, power=miss_power)
    # Ground truth metrics (when available)
    gt = data.get("ground_truth")
    gt_metrics = {}
    if gt:
        gt_metrics["ndcg_5k"] = compute_ndcg(detailed_results, data["token_map"], gt, budget=5000)
        gt_metrics["graded_recall_5k"] = compute_graded_recall(detailed_results, data["token_map"], gt, budget=5000)
        gt_metrics["discovery_rate"] = compute_discovery_rate(detailed_results, gt)

    if metric == "weighted_miss":
        value = wm
    elif metric == "weighted_miss_auc":
        value = wm_auc
    elif metric == "dual":
        value = (wm_auc, mrr_val)  # tuple: (minimize AUC, maximize MRR)
    elif metric == "miss_rate":
        value = miss_rate
    elif metric == "ndcg_5k":
        value = gt_metrics.get("ndcg_5k", 0.0)
    elif metric == "graded_recall_5k":
        value = gt_metrics.get("graded_recall_5k", 0.0)
    else:
        value = mrr_val
    result = {"value": value, "miss_rate_5k": miss_rate, "mrr": mrr_val,
              "weighted_miss": wm, "weighted_miss_auc": wm_auc}
    result.update(gt_metrics)
    return result


def _worker_loop(worker_id, study_name, journal_path, n_trials,
                 search_params, fixed_params, data, metric,
                 miss_min_util, miss_power, wm_threshold, seed,
                 direction):
    """Run n_trials in a worker process with its own JournalStorage connection.

    Each worker gets a different TPE seed for diversity. The shared journal
    file synchronizes trial state across workers (Optuna's documented pattern).
    """
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    lock = JournalFileOpenLock(journal_path + ".lock")
    storage = JournalStorage(JournalFileBackend(journal_path, lock_obj=lock))

    # Multi-objective uses NSGA-II sampler
    if metric == "dual":
        sampler = optuna.samplers.NSGAIISampler(seed=seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.load_study(
        study_name=study_name, storage=storage,
        sampler=sampler,
    )

    # Convert search_params back to set (was serialized as list)
    search_set = set(search_params)

    def objective(trial):
        params = {}
        for pname, spec in SEARCH_RANGES.items():
            lo, hi, ptype = spec[:3]
            log = len(spec) > 3 and spec[3] == "log"
            if pname in search_set:
                if ptype == "int":
                    params[pname] = trial.suggest_int(pname, lo, hi, log=log)
                else:
                    params[pname] = trial.suggest_float(pname, lo, hi, log=log)
            else:
                params[pname] = fixed_params[pname]

        result = _worker_eval(params, data, search_set, fixed_params, metric,
                              miss_min_util, miss_power, wm_threshold)
        trial.set_user_attr("miss_rate_5k", result["miss_rate_5k"])
        trial.set_user_attr("mrr", result["mrr"])
        trial.set_user_attr("weighted_miss", result["weighted_miss"])
        trial.set_user_attr("weighted_miss_auc", result["weighted_miss_auc"])
        for k, v in result.items():
            if k.startswith(("ndcg", "graded", "discovery")):
                trial.set_user_attr(k, v)
        return result["value"]

    study.optimize(objective, n_trials=n_trials)
    return worker_id, n_trials


def run_group1_optimization(data: dict, trials: int, threshold: float,
                            all_thresholds: list[float], reset: bool = False,
                            stage: str = "all",
                            base_params: dict | None = None,
                            metric: str = "weighted_miss",
                            full_plots: bool = True,
                            miss_power: float = 1.5,
                            miss_min_util: float = 0.1,
                            workers: int = 1,
                            tag: str | None = None) -> tuple[dict, dict]:
    """Run one Optuna study at a given threshold using preloaded data.

    metric: "weighted_miss" (utility-weighted miss rate, minimize, threshold-free),
            "miss_rate" (minimize best_cutoff miss rate at 5k tokens), or
            "mrr" (maximize MRR). All metrics are logged per trial regardless.
    base_params overrides CURRENT_PARAMS for fixed (non-searched) values.
    Used by staged pipeline to carry forward winners from earlier stages.
    workers: number of parallel worker processes (default 1 = sequential).
    """
    import optuna

    fixed_params = dict(CURRENT_PARAMS)
    if base_params:
        fixed_params.update(base_params)

    stage_def = STAGES[stage]
    search_params = set(stage_def["search"])

    # Softcap mode: inject flag into fixed_params so it flows to all eval paths
    if stage_def.get("softcap", False):
        fixed_params["_softcap"] = True

    cross_thresholds = sorted(all_thresholds)

    def eval_at(params, t, detailed=False):
        return evaluate_pipeline(
            params, data["queries"], data["fts_results"], data["vec_results"],
            data["query_embeddings"], data["feedback_raw"], data["shadow_map"],
            data["confidence_map"], data["themes_map"], data["hebb_mem_freq"],
            data["hebb_total_queries"], data["edges_by_memory"],
            data["memory_embeddings"], data["edge_types_map"],
            data["active_ids"], data["relevant_sets"][t], data["query_memories"],
            detailed=detailed,
        )

    # --- Evaluate current + zeros ---
    print(f"\n  --- Run: threshold={threshold}, trials={trials} ---")
    print("\n  Evaluating baselines...")
    zeros_params = {}
    for k in CURRENT_PARAMS:
        if k == "rrf_k":
            zeros_params[k] = 60
        elif k == "rrf_vec_weight":
            zeros_params[k] = 0.5  # equal weight baseline
        elif k == "feedback_min_count":
            zeros_params[k] = 2
        elif k == "intent_boost":
            zeros_params[k] = 1.0
        elif k == "cliff_z_threshold":
            zeros_params[k] = 2.0
        elif k == "cliff_min_results":
            zeros_params[k] = 5
        else:
            zeros_params[k] = 0.0

    current_mrrs = {t: eval_at(CURRENT_PARAMS, t) for t in cross_thresholds}
    zeros_mrrs = {t: eval_at(zeros_params, t) for t in cross_thresholds}

    # Compute miss rate baselines (for convergence plot reference lines)
    current_miss = {}
    zeros_miss = {}
    current_wm = {}
    zeros_wm = {}
    for t in cross_thresholds:
        _, det_c = eval_at(CURRENT_PARAMS, t, detailed=True)
        _, det_z = eval_at(zeros_params, t, detailed=True)
        mc = compute_miss_rates(det_c, data["token_map"], budgets=[5000])
        mz = compute_miss_rates(det_z, data["token_map"], budgets=[5000])
        current_miss[t] = mc["best_cutoff"][0]
        zeros_miss[t] = mz["best_cutoff"][0]
        current_wm[t] = compute_weighted_miss(det_c, data["token_map"],
                                              data["feedback_raw"], budget=5000,
                                              min_utility=miss_min_util, power=miss_power)
        zeros_wm[t] = compute_weighted_miss(det_z, data["token_map"],
                                            data["feedback_raw"], budget=5000,
                                            min_utility=miss_min_util, power=miss_power)

    for t in cross_thresholds:
        print(f"    @{t}: MRR current={current_mrrs[t]:.4f} zeros={zeros_mrrs[t]:.4f}"
              f"  |  miss(best@5k) current={current_miss[t]:.3f} zeros={zeros_miss[t]:.3f}"
              f"  |  wm current={current_wm[t]:.3f} zeros={zeros_wm[t]:.3f}")

    # --- Bayesian optimization (persistent journal storage) ---
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

    study_dir = DATA_DIR / "tuning_studies"
    study_dir.mkdir(parents=True, exist_ok=True)
    metric_tag = {"miss_rate": "mr", "mrr": "mrr", "weighted_miss": "wm",
                   "weighted_miss_auc": "wmauc", "dual": "dual"}[metric]
    study_name = f"group1_{stage}_t{threshold}_{metric_tag}"
    if tag:
        study_name += f"_{tag}"
    journal_path = study_dir / f"{study_name}.log"

    if reset and journal_path.exists():
        journal_path.unlink()
        lock_file = Path(str(journal_path) + ".lock")
        if lock_file.exists():
            lock_file.unlink()
        print(f"\n  Reset: deleted {journal_path.name}")

    lock = JournalFileOpenLock(str(journal_path) + ".lock")
    storage = JournalStorage(JournalFileBackend(str(journal_path), lock_obj=lock))

    # Resume or create
    is_dual = metric == "dual"
    if is_dual:
        sampler = optuna.samplers.NSGAIISampler(seed=42)
    else:
        sampler = optuna.samplers.TPESampler(seed=42)
    try:
        study = optuna.load_study(
            study_name=study_name, storage=storage,
            sampler=sampler,
        )
        existing = len(study.trials)
        remaining = max(0, trials - existing)
        print(f"\n  Resuming study '{study_name}': {existing} existing trials, "
              f"{remaining} remaining")
    except KeyError:
        if is_dual:
            study = optuna.create_study(
                study_name=study_name, storage=storage,
                directions=["minimize", "maximize"],  # AUC, MRR
                sampler=sampler,
            )
        else:
            direction = "maximize" if metric in ("mrr", "ndcg_5k", "graded_recall_5k") else "minimize"
            study = optuna.create_study(
                study_name=study_name, storage=storage, direction=direction,
                sampler=sampler,
            )
        existing = 0
        remaining = trials
        # Enqueue seeded trials only for fresh studies (clamped to search ranges)
        def _clamp_seed(src):
            clamped = {}
            for k, v in src.items():
                if k not in search_params:
                    continue
                lo, hi, ptype = SEARCH_RANGES[k][:3]
                if ptype == "int":
                    clamped[k] = max(lo, min(hi, int(v)))
                else:
                    clamped[k] = max(lo, min(hi, float(v)))
            return clamped
        study.enqueue_trial(_clamp_seed(fixed_params))
        study.enqueue_trial(_clamp_seed(zeros_params))
        print(f"\n  New study '{study_name}': {trials} trials")

    if remaining == 0:
        print(f"  Already complete ({existing}/{trials} trials). "
              f"Use more --trials to extend, or --reset to start fresh.")
        # Still run reporting on existing data
    else:
        print(f"  Optimizing ({remaining} trials, threshold {threshold})...")

    MISS_RATE_BUDGET = 5000

    # For weighted_miss variants and dual, use a low threshold to get broad relevant set
    _wm_threshold = miss_min_util if metric in ("weighted_miss", "weighted_miss_auc", "dual") else threshold

    def objective(trial):
        params = {}
        for pname, spec in SEARCH_RANGES.items():
            lo, hi, ptype = spec[:3]
            log = len(spec) > 3 and spec[3] == "log"
            if pname in search_params:
                if ptype == "int":
                    params[pname] = trial.suggest_int(pname, lo, hi, log=log)
                else:
                    params[pname] = trial.suggest_float(pname, lo, hi, log=log)
            else:
                params[pname] = fixed_params[pname]
        mrr_val, detailed_results = eval_at(params, _wm_threshold, detailed=True)
        miss = compute_miss_rates(detailed_results, data["token_map"],
                                   budgets=[MISS_RATE_BUDGET])
        miss_rate = miss["best_cutoff"][0]
        wm = compute_weighted_miss(detailed_results, data["token_map"],
                                   data["feedback_raw"], budget=MISS_RATE_BUDGET,
                                   min_utility=miss_min_util, power=miss_power)
        wm_auc = compute_weighted_miss_auc(detailed_results, data["token_map"],
                                            data["feedback_raw"],
                                            min_utility=miss_min_util, power=miss_power)
        # Always log all metrics
        trial.set_user_attr("miss_rate_5k", miss_rate)
        trial.set_user_attr("mrr", mrr_val)
        trial.set_user_attr("weighted_miss", wm)
        trial.set_user_attr("weighted_miss_auc", wm_auc)
        if metric == "dual":
            return wm_auc, mrr_val  # minimize AUC, maximize MRR
        if metric == "weighted_miss":
            return wm
        if metric == "weighted_miss_auc":
            return wm_auc
        return miss_rate if metric == "miss_rate" else mrr_val

    import time as _time
    import random as _random
    _random.seed(42)
    # Initialize from existing trials on resume
    if is_dual:
        _minimize = True  # track AUC (objective 0) for progress
        _best_fn = min
        _worse_init = 1.0
        if existing > 0:
            completed = [t for t in study.trials if t.values is not None]
            best_so_far = [min(t.values[0] for t in completed)] if completed else [1.0]
            best_at = [min(completed, key=lambda t: t.values[0]).number] if completed else [0]
        else:
            best_so_far = [1.0]
            best_at = [0]
    else:
        _minimize = metric != "mrr"
        _best_fn = min if _minimize else max
        _worse_init = 1.0 if _minimize else 0.0
        if existing > 0:
            completed = [t for t in study.trials if t.value is not None]
            best_so_far = [_best_fn(t.value for t in completed)] if completed else [_worse_init]
            best_at = [_best_fn(completed, key=lambda t: t.value).number] if completed else [0]
        else:
            best_so_far = [_worse_init]
            best_at = [0]
    t_start = [_time.monotonic()]
    t_prev = [t_start[0]]
    n_width = len(str(trials))
    trial_offset = [existing]  # for display numbering on resume
    explore_count = [0]

    def _random_trial():
        result = {}
        for k, spec in SEARCH_RANGES.items():
            if k not in search_params:
                continue
            lo, hi, ptype = spec[:3]
            log = len(spec) > 3 and spec[3] == "log"
            if log:
                # Sample uniformly in log space
                v = math.exp(_random.uniform(math.log(max(lo, 1)), math.log(hi)))
                result[k] = int(round(v)) if ptype == "int" else v
            elif ptype == "int":
                result[k] = _random.randint(lo, hi)
            else:
                result[k] = _random.uniform(lo, hi)
        return result

    # Order params by fANOVA importance (recomputed periodically)
    param_order = [sorted(search_params)]  # alphabetical until enough data
    _IMPORTANCE_INTERVAL = 25
    _IMPORTANCE_MIN_TRIALS = 20

    def _refresh_param_order(study):
        try:
            imp = optuna.importance.get_param_importances(study)
            # imp is ordered by importance (descending); keep only searched params
            ordered = [k for k in imp if k in search_params]
            # append any searched params not in importance result
            for k in sorted(search_params):
                if k not in ordered:
                    ordered.append(k)
            param_order[0] = ordered
        except Exception:
            pass  # keep previous order

    # Seed order from existing trials on resume
    if existing >= _IMPORTANCE_MIN_TRIALS:
        _refresh_param_order(study)

    def progress_callback(study, trial):
        now = _time.monotonic()
        dt = now - t_prev[0]
        elapsed = now - t_start[0]
        t_prev[0] = now
        if is_dual:
            trial_val = trial.values[0] if trial.values else None  # AUC for tracking
        else:
            trial_val = trial.value
        is_new_best = (trial_val is not None and
                       (trial_val < best_so_far[0] if _minimize
                        else trial_val > best_so_far[0]))
        if is_new_best:
            best_so_far[0] = trial_val
            best_at[0] = trial.number
        n = trial.number + 1
        stale = n - best_at[0] - 1
        marker = " *" if is_new_best else "  "
        dots = "·" * min(stale // 10, 20) if stale >= 10 else ""

        # Inject random exploration as stale grows:
        # p(random) = 0 until stale=20, then ramps to ~50% at stale=100
        if stale >= 20:
            explore_prob = min(0.5, (stale - 20) / 160)
            if _random.random() < explore_prob:
                study.enqueue_trial(_random_trial())
                explore_count[0] += 1
                marker = " R"  # R = random injection queued

        # Refresh param display order by importance periodically
        completed = len(study.trials)
        if (completed >= _IMPORTANCE_MIN_TRIALS and
                completed % _IMPORTANCE_INTERVAL == 0):
            _refresh_param_order(study)

        p = trial.params
        # Show up to 4 params, ordered by importance (most important first)
        param_str = "  ".join(
            f"{k}={p[k]:>3d}" if SEARCH_RANGES[k][2] == "int" else f"{k}={p[k]:>5.3f}"
            for k in param_order[0][:4] if k in p
        )
        if is_dual and trial.values:
            val_str = f"  AUC={trial.values[0]:.4f} MRR={trial.values[1]:.4f}"
            best_str = f"[AUC={best_so_far[0]:.4f}]"
        else:
            val_str = f"  {trial_val:.4f}  [{best_so_far[0]:.4f} @{best_at[0]+1:<{n_width}}]"
            best_str = ""
        print(f"  {n:>{n_width}}/{trials}"
              f"{val_str}  {best_str}"
              f"  {param_str}"
              f"  {elapsed:>6.0f}s"
              f"{marker}{dots}", flush=True)

    if remaining > 0:
        if workers > 1:
            # --- Multi-worker parallel optimization ---
            from concurrent.futures import ProcessPoolExecutor
            import time as _time_mp

            direction = "dual" if is_dual else ("maximize" if metric in ("mrr", "ndcg_5k", "graded_recall_5k") else "minimize")
            per_worker = remaining // workers
            remainder_w = remaining % workers
            worker_trials = [per_worker + (1 if i < remainder_w else 0)
                             for i in range(workers)]

            print(f"  Launching {workers} workers: "
                  f"{' + '.join(str(t) for t in worker_trials)} trials", flush=True)

            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(_worker_loop,
                                worker_id=i,
                                study_name=study_name,
                                journal_path=str(journal_path),
                                n_trials=worker_trials[i],
                                search_params=list(search_params),
                                fixed_params=dict(fixed_params),
                                data=data,
                                metric=metric,
                                miss_min_util=miss_min_util,
                                miss_power=miss_power,
                                wm_threshold=_wm_threshold,
                                seed=42 + i,
                                direction=direction)
                    for i in range(workers)
                ]
                # Poll progress — print each new trial like single-worker mode
                t_mp_start = _time_mp.monotonic()
                seen_trials = set(
                    t.number for t in study.trials
                    if (t.values if is_dual else t.value) is not None)
                mp_best = best_so_far[0]
                mp_best_at = best_at[0]
                while not all(f.done() for f in futures):
                    _time_mp.sleep(2)
                    try:
                        study_check = optuna.load_study(
                            study_name=study_name, storage=storage,
                            sampler=sampler)
                        new_trials = [
                            t for t in study_check.trials
                            if (t.values if is_dual else t.value) is not None
                            and t.number not in seen_trials
                        ]
                        new_trials.sort(key=lambda t: t.number)
                        for trial in new_trials:
                            seen_trials.add(trial.number)
                            elapsed_mp = _time_mp.monotonic() - t_mp_start
                            if is_dual:
                                t_val = trial.values[0]  # AUC
                            else:
                                t_val = trial.value
                            is_best = (t_val < mp_best if _minimize
                                       else t_val > mp_best)
                            if is_best:
                                mp_best = t_val
                                mp_best_at = trial.number
                            marker = " *" if is_best else "  "
                            p = trial.params
                            param_str = "  ".join(
                                f"{k}={p[k]:>3d}" if SEARCH_RANGES[k][2] == "int"
                                else f"{k}={p[k]:>5.3f}"
                                for k in param_order[0][:4] if k in p
                            )
                            n_done = len(seen_trials)
                            if is_dual:
                                val_str = (f"  AUC={trial.values[0]:.4f}"
                                           f" MRR={trial.values[1]:.4f}")
                            else:
                                val_str = (f"  {t_val:.4f}"
                                           f"  [{mp_best:.4f}"
                                           f" @{mp_best_at+1:<{n_width}}]")
                            print(
                                f"  {n_done:>{n_width}}/{trials}"
                                f"{val_str}"
                                f"  {param_str}"
                                f"  {elapsed_mp:>6.0f}s"
                                f"{marker}", flush=True)
                            # Refresh param order periodically
                            if (n_done >= _IMPORTANCE_MIN_TRIALS and
                                    n_done % _IMPORTANCE_INTERVAL == 0):
                                _refresh_param_order(study_check)
                    except Exception:
                        pass
                # Collect results, re-raise any worker exceptions
                for f in futures:
                    f.result()

            # Reload study with all trials from all workers
            study = optuna.load_study(
                study_name=study_name, storage=storage,
                sampler=sampler)
            # Sync best tracking for post-study summary
            best_so_far[0] = mp_best
            best_at[0] = mp_best_at
            print(f"  All workers done. Total trials: {len(study.trials)}")
        else:
            # --- Single-worker with per-trial progress ---
            study.optimize(objective, n_trials=remaining,
                           callbacks=[progress_callback])

    if explore_count[0]:
        print(f"\n  Random explorations injected: {explore_count[0]}")

    # --- Results ---
    if is_dual:
        # Multi-objective: show Pareto front
        pareto_trials = study.best_trials  # Pareto-optimal trials
        # Sort by AUC (objective 0, ascending = better)
        pareto_trials.sort(key=lambda t: t.values[0])

        print(f"\n  --- Pareto Front ({len(pareto_trials)} trials) ---")
        print(f"  {'#':>4} {'AUC':>8} {'MRR':>8}  ", end="")
        print("  ".join(f"{k:>12}" for k in sorted(search_params)))
        print("  " + "-" * (24 + 14 * len(search_params)))
        for i, trial in enumerate(pareto_trials[:20]):
            p = trial.params
            param_str = "  ".join(
                f"{p[k]:>12.4f}" if SEARCH_RANGES[k][2] != "int" else f"{p[k]:>12d}"
                for k in sorted(search_params) if k in p
            )
            print(f"  {i+1:>4} {trial.values[0]:>8.4f} {trial.values[1]:>8.4f}  {param_str}")

        # Use the trial with best AUC for standard reporting
        best_trial = pareto_trials[0]
        best_searched = best_trial.params
        best_params = dict(fixed_params)
        best_params.update(best_searched)
        best_mrrs = {t: eval_at(best_params, t) for t in cross_thresholds}

        print(f"\n  Best AUC point: AUC={best_trial.values[0]:.4f}, MRR={best_trial.values[1]:.4f}")
        best_mrr_trial = max(pareto_trials, key=lambda t: t.values[1])
        print(f"  Best MRR point: AUC={best_mrr_trial.values[0]:.4f}, MRR={best_mrr_trial.values[1]:.4f}")
    else:
        # Single-objective: use best_params
        best_searched = study.best_params
        best_params = dict(fixed_params)
        best_params.update(best_searched)
        best_mrrs = {t: eval_at(best_params, t) for t in cross_thresholds}

    print(f"\n  --- MRR Summary ---")
    header = f"  {'Configuration':<30}"
    for t in cross_thresholds:
        header += f" {'MRR @' + str(t):>10}"
    print(header)
    print("  " + "-" * (30 + 11 * len(cross_thresholds)))

    for label, mrrs in [("All-zeros (base RRF only)", zeros_mrrs),
                        ("Current production", current_mrrs),
                        ("Optimal (Bayesian)", best_mrrs)]:
        row = f"  {label:<30}"
        for t in cross_thresholds:
            row += f" {mrrs[t]:>10.4f}"
        print(row)

    print()
    for t in cross_thresholds:
        delta = ((best_mrrs[t] - current_mrrs[t]) / current_mrrs[t] * 100) if current_mrrs[t] else 0
        marker = " <-- optimized" if t == threshold else ""
        print(f"    Optimal vs current @{t}: {delta:+.1f}%{marker}")

    if not is_dual:
        obj_label = "Miss rate (best@5k)" if metric == "miss_rate" else "MRR"
        print(f"\n  Optimized metric: {obj_label} = {study.best_value:.4f}")

    # Compute final importance order for all summary tables
    if is_dual:
        # fANOVA needs a target for multi-objective
        display_order = sorted(search_params)
    else:
        _refresh_param_order(study)
        display_order = param_order[0]

    # Parameter comparison (searched params only, by importance)
    print(f"\n  --- Parameter Comparison ({stage}) ---")
    print(f"  {'Parameter':<22} {'Current':>10} {'Optimal':>10} {'Change':>8} {'Decision':>10}")
    print("  " + "-" * 64)
    for param in display_order:
        current_val = fixed_params[param]
        optimal_val = best_params[param]
        if isinstance(current_val, int):
            if optimal_val == current_val:
                decision = "KEEP"
            else:
                decision = "RETUNE"
            change_str = ""
            print(f"    {param:<20} {current_val:>10d} {optimal_val:>10d} {'':>8} {decision:>10}")
        else:
            if optimal_val < 0.001:
                decision = "CUT"
            elif current_val > 0 and abs(optimal_val - current_val) / current_val < 0.2:
                decision = "KEEP"
            else:
                decision = "RETUNE"
            ratio = optimal_val / current_val if current_val else float('inf')
            change_str = f"{ratio:.1f}x" if current_val else "new"
            print(f"    {param:<20} {current_val:>10.4f} {optimal_val:>10.4f} {change_str:>8} {decision:>10}")

    # Bootstrap error bounds (skip for dual — Pareto front IS the summary)
    param_bounds = {}
    if not is_dual:
        param_bounds = bootstrap_param_bounds(study, search_params)
        print(f"\n  --- Bootstrap Error Bounds (top 20%, 1000 resamples) ---")
        print(f"  {'Parameter':<22} {'Mean':>10} {'+/- Std':>10} {'2sd Range':>20}")
        print("  " + "-" * 64)
        for param in display_order:
            if param not in param_bounds:
                continue
            mean, std, lo, hi = param_bounds[param]
            if SEARCH_RANGES[param][2] == "int":
                print(f"    {param:<20} {mean:>10.1f} {std:>10.1f} [{lo:>7.1f}, {hi:>7.1f}]")
            else:
                print(f"    {param:<20} {mean:>10.4f} {std:>10.4f} [{lo:>7.4f}, {hi:>7.4f}]")

    # fANOVA importance
    print(f"\n  --- Parameter Importance (fANOVA) ---")
    try:
        if is_dual:
            # Show importance for each objective
            for obj_idx, obj_name in [(0, "AUC"), (1, "MRR")]:
                print(f"    [{obj_name}]")
                importances = optuna.importance.get_param_importances(
                    study, target=lambda t: t.values[obj_idx])
                for param, importance in importances.items():
                    bar = "#" * int(importance * 50)
                    print(f"      {param:<20} {importance:>6.3f}  {bar}")
        else:
            importances = optuna.importance.get_param_importances(study)
            for param, importance in importances.items():
                bar = "#" * int(importance * 50)
                print(f"    {param:<22} {importance:>6.3f}  {bar}")
    except Exception as e:
        print(f"    Could not compute importances: {e}")

    # Top 10 trials
    if is_dual:
        # Show top 10 by AUC from all trials (not just Pareto)
        print(f"\n  --- Top 10 by AUC ---")
        top_trials = sorted(
            [t for t in study.trials if t.values is not None],
            key=lambda t: t.values[0])[:10]
    else:
        print(f"\n  --- Top 10 Trials ---")
        top_trials = sorted(
            [t for t in study.trials if t.value is not None],
            key=lambda t: t.value, reverse=not _minimize)[:10]
    for i, trial in enumerate(top_trials):
        param_parts = []
        for k in display_order:
            v = trial.params.get(k)
            if v is None:
                continue
            if SEARCH_RANGES[k][2] == "int":
                param_parts.append(f"{k}={v:>3d}")
            else:
                param_parts.append(f"{k}={v:.3f}")
        if is_dual:
            metrics_str = f"AUC={trial.values[0]:.4f}  MRR={trial.values[1]:.4f}"
        else:
            mrr_v = trial.user_attrs.get("mrr", trial.value)
            mr_v = trial.user_attrs.get("miss_rate_5k")
            metrics_str = f"MRR={mrr_v:.4f}"
            if mr_v is not None:
                metrics_str += f"  MR={mr_v:.4f}"
        print(f"    #{i+1} {metrics_str}  {'  '.join(param_parts)}")

    # Sanity checks
    print(f"\n  --- Sanity Checks ---")
    checks_passed = 0
    checks_total = 3

    t = threshold
    if zeros_mrrs[t] < current_mrrs[t]:
        print(f"    [PASS] Zeros MRR ({zeros_mrrs[t]:.4f}) < current ({current_mrrs[t]:.4f})")
        checks_passed += 1
    else:
        print(f"    [INFO] Zeros MRR ({zeros_mrrs[t]:.4f}) >= current ({current_mrrs[t]:.4f})"
              f" — post-RRF boosts may hurt at this threshold")
        checks_passed += 1  # not a failure — informational

    if best_mrrs[t] >= current_mrrs[t] - 0.001:
        print(f"    [PASS] Optimal MRR ({best_mrrs[t]:.4f}) >= current ({current_mrrs[t]:.4f})")
        checks_passed += 1
    else:
        print(f"    [FAIL] Optimal MRR ({best_mrrs[t]:.4f}) < current ({current_mrrs[t]:.4f})")

    try:
        if is_dual:
            test_imp = optuna.importance.get_param_importances(
                study, target=lambda t: t.values[0])
        else:
            test_imp = importances
        imp_values = list(test_imp.values())
        if max(imp_values) > 0.01 and min(imp_values) < max(imp_values):
            print(f"    [PASS] Importance non-degenerate ({min(imp_values):.3f} - {max(imp_values):.3f})")
            checks_passed += 1
        else:
            print(f"    [FAIL] Importance degenerate")
    except Exception:
        print(f"    [SKIP] Could not evaluate importance")

    print(f"\n    {checks_passed}/{checks_total} sanity checks passed")

    # --- Plots ---
    try:
        plot_dir = DATA_DIR / "tuning_plots" / metric_tag
        generate_plots(study, threshold,
                       current_miss[threshold], zeros_miss[threshold],
                       current_mrrs[threshold], zeros_mrrs[threshold],
                       plot_dir, full=full_plots)
    except Exception as e:
        print(f"\n  [WARN] Plot generation failed: {e}")

    # --- Miss Rate Analysis ---
    try:
        _, detailed = evaluate_pipeline(
            best_params, data["queries"], data["fts_results"], data["vec_results"],
            data["query_embeddings"], data["feedback_raw"], data["shadow_map"],
            data["confidence_map"], data["themes_map"], data["hebb_mem_freq"],
            data["hebb_total_queries"], data["edges_by_memory"],
            data["memory_embeddings"], data["edge_types_map"],
            data["active_ids"], data["relevant_sets"][threshold], data["query_memories"],
            detailed=True,
        )
        token_map = data["token_map"]
        miss_rates = compute_miss_rates(detailed, token_map)
        print_miss_rate_table(miss_rates)

        try:
            mrr_budgets = compute_mrr_at_budgets(detailed, token_map)
        except Exception:
            mrr_budgets = None

        try:
            plot_dir = DATA_DIR / "tuning_plots" / metric_tag
            p = plot_budget_curves(miss_rates, mrr_budgets, threshold, plot_dir,
                                   study_name=study.study_name)
            if p:
                print(f"    Budget curves plot: {p}")
        except Exception as e:
            print(f"    [WARN] Budget curves plot failed: {e}")
    except Exception as e:
        print(f"\n  [WARN] Miss rate analysis failed: {e}")

    # Build result dict — only emit changed values
    result = {}
    for k, v in best_params.items():
        current = fixed_params[k]
        if isinstance(current, int):
            if v != current:
                result[GROUP1_MAP[k]] = v
        else:
            # Changed if >1% relative difference
            if current == 0 or abs(v - current) / max(abs(current), 1e-9) > 0.01:
                result[GROUP1_MAP[k]] = v
    return result, param_bounds


# ---------------------------------------------------------------------------
# Group 2: Quality Floor (stub)
# ---------------------------------------------------------------------------


def tune_group_2(db: sqlite3.Connection) -> dict:
    print("\n=== Group 2: Quality Floor ===")
    count = db.execute(
        "SELECT COUNT(*) FROM memory_events WHERE event_type = 'recall_cutoff'"
    ).fetchone()[0]
    min_required = 50
    if count < min_required:
        print(f"  {count}/{min_required} recall_cutoff events. Skipping.")
        return {}
    # TODO: analyze gap between useful/noise scores -> optimal QUALITY_FLOOR_RATIO
    print(f"  {count} events available. Analysis not yet implemented.")
    return {}


# ---------------------------------------------------------------------------
# Group 3: Confidence Learning (stub)
# ---------------------------------------------------------------------------


def tune_group_3(db: sqlite3.Connection) -> dict:
    print("\n=== Group 3: Confidence Learning ===")
    count = db.execute("""
        SELECT COUNT(*) FROM memory_events
        WHERE event_type = 'feedback'
          AND context IS NOT NULL
          AND json_extract(context, '$.confidence_before') IS NOT NULL
    """).fetchone()[0]
    min_required = 100
    if count < min_required:
        print(f"  {count}/{min_required} feedback events with confidence_before. Skipping.")
        return {}
    # TODO: simulate learning rates, maximize confidence->utility correlation
    print(f"  {count} events available. Analysis not yet implemented.")
    return {}


# ---------------------------------------------------------------------------
# Group 4: Decay Model (numpy fit)
# ---------------------------------------------------------------------------


def tune_group_4(db: sqlite3.Connection) -> dict:
    """Fit exponential decay curves per category from feedback-vs-age data."""
    import numpy as np

    print("\n=== Group 4: Decay Model (numpy fit) ===")

    rows = db.execute("""
        SELECT m.category,
               julianday(e.created_at) - julianday(m.created_at) as age_days,
               json_extract(e.context, '$.utility') as utility
        FROM memory_events e
        JOIN memories m ON e.memory_id = m.id
        WHERE e.event_type = 'feedback'
          AND json_extract(e.context, '$.utility') IS NOT NULL
    """).fetchall()

    if len(rows) < 50:
        print(f"  Only {len(rows)} feedback points total. Need 50+. Skipping.")
        return {}

    # Organize by category
    by_category = defaultdict(list)
    for r in rows:
        cat = r["category"]
        age = r["age_days"]
        util = r["utility"]
        if age is not None and util is not None and age >= 0:
            by_category[cat].append((float(age), float(util)))

    print(f"\n  {'Category':<14} {'N':>5} {'Empirical':>10} {'Half-life':>10} "
          f"{'Current':>10} {'Recommended':>12} {'Change':>10}")
    print("  " + "-" * 73)

    changes = {}
    fitted_rates = {}

    for cat in sorted(by_category.keys()):
        points = by_category[cat]
        n = len(points)

        if n < 50:
            print(f"  {cat:<14} {n:>5}  (insufficient data, need 50+)")
            continue

        # Bin by age (3-day bins)
        max_age = max(p[0] for p in points)
        bin_size = 3.0
        n_bins = max(1, int(max_age / bin_size) + 1)

        bin_sums = [0.0] * n_bins
        bin_counts = [0] * n_bins
        for age, util in points:
            b = min(int(age / bin_size), n_bins - 1)
            bin_sums[b] += util
            bin_counts[b] += 1

        # Compute bin means, filter bins with data and positive mean
        bin_ages = []
        bin_means = []
        for b in range(n_bins):
            if bin_counts[b] >= 3:  # need at least 3 points per bin
                mean = bin_sums[b] / bin_counts[b]
                if mean > 0:
                    bin_ages.append((b + 0.5) * bin_size)
                    bin_means.append(mean)

        if len(bin_ages) < 3:
            print(f"  {cat:<14} {n:>5}  (too few valid bins for fit)")
            continue

        # Fit: ln(utility) = ln(A) - rate * age
        log_means = np.log(bin_means)
        ages_arr = np.array(bin_ages)

        coeffs = np.polyfit(ages_arr, log_means, 1)
        slope = coeffs[0]  # -rate
        rate = -slope

        if rate <= 0:
            # No decay detected — utility increases or flat with age
            print(f"  {cat:<14} {n:>5}  (no decay detected, slope={slope:.4f})")
            continue

        half_life = math.log(2) / rate
        current_rate = CATEGORY_DECAY_RATES.get(cat, DEFAULT_DECAY_RATE)

        # Only recommend change if > 15% different and sufficient data
        pct_change = (rate - current_rate) / current_rate * 100 if current_rate > 0 else 0
        if abs(pct_change) > 15 and n >= 100:
            recommended = round(rate, 4)
            change_str = f"{pct_change:+.1f}%"
        elif abs(pct_change) > 15:
            recommended = current_rate
            change_str = f"KEEP (n={n})"
        else:
            recommended = current_rate
            change_str = "KEEP"

        print(f"  {cat:<14} {n:>5} {rate:>10.4f} {half_life:>9.1f}d "
              f"{current_rate:>10.4f} {recommended:>12.4f} {change_str:>10}")

        fitted_rates[cat] = rate

        if recommended != current_rate:
            # We'll build the full CATEGORY_DECAY_RATES dict at the end
            changes[f"_decay_{cat}"] = recommended

    # Survivorship bias check
    print(f"\n  --- Survivorship Bias Check ---")
    total_active = db.execute(
        "SELECT COUNT(*) FROM memories WHERE status = 'active'"
    ).fetchone()[0]
    total_with_fb = db.execute("""
        SELECT COUNT(DISTINCT m.id)
        FROM memories m
        JOIN memory_events e ON e.memory_id = m.id
        WHERE e.event_type = 'feedback'
          AND m.status = 'active'
    """).fetchone()[0]

    retrieval_rate = total_with_fb / total_active if total_active else 0
    print(f"  Active memories: {total_active}")
    print(f"  With feedback: {total_with_fb} ({retrieval_rate:.1%})")
    if retrieval_rate < 0.3:
        print(f"  WARNING: Low retrieval rate -- empirical rates may underestimate true decay")

    # Build CATEGORY_DECAY_RATES changes if any category changed
    if changes:
        new_rates = dict(CATEGORY_DECAY_RATES)
        for key, val in list(changes.items()):
            cat = key.replace("_decay_", "")
            new_rates[cat] = val
            del changes[key]
        changes["CATEGORY_DECAY_RATES"] = new_rates

        # Update DEFAULT_DECAY_RATE to weighted average of fitted rates
        # Only if 2+ categories show decay with 100+ points each — a single
        # noisy category shouldn't override the semantic-anchored default.
        strong_fits = {cat: rate for cat, rate in fitted_rates.items()
                       if len(by_category[cat]) >= 100}
        if len(strong_fits) >= 2:
            weights = {cat: len(by_category[cat]) for cat in strong_fits}
            total_w = sum(weights.values())
            weighted_avg = sum(strong_fits[cat] * weights[cat] for cat in strong_fits) / total_w
            new_default = round(weighted_avg, 4)
            if abs(new_default - DEFAULT_DECAY_RATE) / DEFAULT_DECAY_RATE > 0.15:
                changes["DEFAULT_DECAY_RATE"] = new_default

    return changes


# ---------------------------------------------------------------------------
# Group 5: Adjacency Capacity (stub)
# ---------------------------------------------------------------------------


def tune_group_5(db: sqlite3.Connection) -> dict:
    print("\n=== Group 5: Adjacency Capacity ===")
    count = db.execute("""
        SELECT COUNT(*) FROM memory_events
        WHERE event_type = 'retrieved'
          AND context IS NOT NULL
          AND json_extract(context, '$.per_seed_cap_hits') > 0
    """).fetchone()[0]
    min_required = 30
    if count < min_required:
        print(f"  {count}/{min_required} retrievals with cap hits. Skipping.")
        return {}
    print(f"  {count} events available. Analysis not yet implemented.")
    return {}


# ---------------------------------------------------------------------------
# Group 6: Dedup (stub)
# ---------------------------------------------------------------------------


def tune_group_6(db: sqlite3.Connection) -> dict:
    print("\n=== Group 6: Dedup ===")
    count = db.execute(
        "SELECT COUNT(*) FROM memory_events WHERE event_type = 'dedup_rejected'"
    ).fetchone()[0]
    min_required = 20
    if count < min_required:
        print(f"  {count}/{min_required} dedup_rejected events. Skipping.")
        return {}
    print(f"  {count} events available. Analysis not yet implemented.")
    return {}


# ---------------------------------------------------------------------------
# Group 7: Feedback Processing (stub)
# ---------------------------------------------------------------------------


def tune_group_7(db: sqlite3.Connection) -> dict:
    print("\n=== Group 7: Feedback Processing ===")
    count = db.execute(
        "SELECT COUNT(*) FROM memory_events WHERE event_type = 'edge_weight_change'"
    ).fetchone()[0]
    min_required = 30
    if count < min_required:
        print(f"  {count}/{min_required} edge_weight_change events. Skipping.")
        return {}
    print(f"  {count} events available. Analysis not yet implemented.")
    return {}


# ---------------------------------------------------------------------------
# Diff generation + apply
# ---------------------------------------------------------------------------


def _format_value(value) -> str:
    """Format a constant value for constants.py (readable precision)."""
    if isinstance(value, int):
        return str(value)
    if value == 0.0:
        return "0.0"
    return f"{float(f'{value:.4g}')}"


def generate_diff(changes: dict, constants_path: Path) -> str:
    """Generate a unified diff for the proposed constant changes."""
    original = constants_path.read_text(encoding="utf-8")
    modified = original

    for name, value in changes.items():
        if name == "CATEGORY_DECAY_RATES":
            # Multiline dict replacement — preserve alignment style
            pattern = r'(CATEGORY_DECAY_RATES\s*=\s*\{)[^}]*(})'
            # Find max key length for alignment
            max_key = max(len(cat) for cat in value)
            new_body = "CATEGORY_DECAY_RATES = {\n"
            for cat, rate in sorted(value.items()):
                half_life = math.log(2) / rate if rate > 0 else float('inf')
                padding = " " * (max_key - len(cat) + 2)
                rate_str = f"{rate}"
                new_body += f'    "{cat}":{padding}{rate_str},   # ~{half_life:.0f}d half-life\n'
            new_body += "}"
            modified = re.sub(pattern, new_body, modified, flags=re.DOTALL)
        elif isinstance(value, int):
            pattern = rf'^({name}\s*=\s*)\d+(\s*(?:#.*)?)$'
            replacement = rf'\g<1>{value}\2'
            modified = re.sub(pattern, replacement, modified, flags=re.MULTILINE)
        elif isinstance(value, float):
            formatted = _format_value(value)
            pattern = rf'^({name}\s*=\s*)\S+(\s*(?:#.*)?)$'
            replacement = rf'\g<1>{formatted}\2'
            modified = re.sub(pattern, replacement, modified, flags=re.MULTILINE)

    # Re-align inline comments: find the column of the existing comment
    # and pad the new value to match
    lines = modified.split("\n")
    orig_lines = original.split("\n")
    aligned = []
    for new_line, orig_line in zip(lines, orig_lines):
        if "#" in new_line and "#" in orig_line and new_line != orig_line:
            # Try to align the comment to the same column as original
            orig_comment_col = orig_line.index("#")
            if "#" in new_line:
                new_code, new_comment = new_line.split("#", 1)
                new_code = new_code.rstrip()
                if len(new_code) < orig_comment_col:
                    new_line = new_code + " " * (orig_comment_col - len(new_code)) + "#" + new_comment
                else:
                    # New code is longer than original comment column -- 2-space gap
                    new_line = new_code + "  #" + new_comment
        aligned.append(new_line)
    # Handle length mismatch (CATEGORY_DECAY_RATES can change line count)
    if len(lines) > len(orig_lines):
        aligned.extend(lines[len(orig_lines):])
    modified = "\n".join(aligned)

    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile="constants.py (current)",
        tofile="constants.py (tuned)",
    )
    # Force ASCII in diff output to avoid terminal encoding issues
    diff_text = "".join(diff)
    diff_text = diff_text.replace("\u2014", "--")  # em-dash
    diff_text = diff_text.replace("\u2013", "-")   # en-dash
    return diff_text, modified


def apply_changes(modified_text: str, constants_path: Path):
    """Write the modified constants.py."""
    constants_path.write_text(modified_text, encoding="utf-8")
    print(f"\n  Written to {constants_path}")


# ---------------------------------------------------------------------------
# Plots (CTT-style)
# ---------------------------------------------------------------------------

CTT_BG = "#36393f"


def _setup_style():
    """Apply CTT-inspired dark style."""
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": CTT_BG,
        "axes.facecolor": CTT_BG,
        "savefig.facecolor": CTT_BG,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "font.size": 10,
    })


def plot_convergence(study, threshold, current_miss, zeros_miss,
                     current_mrr, zeros_mrr, out_dir: Path):
    """Optimization history: dual y-axis with miss rate (left) and MRR (right)."""
    import matplotlib.pyplot as plt
    _setup_style()

    is_multi = hasattr(study, 'directions') and len(study.directions) > 1
    if is_multi:
        minimize = True
        trials = [t for t in study.trials if t.values is not None]
        miss_vals = [t.values[0] for t in trials]
        mrr_vals = [t.values[1] for t in trials]
    else:
        minimize = study.direction.name == "MINIMIZE"
        trials = [t for t in study.trials if t.value is not None]
        miss_vals = [t.value for t in trials]
        mrr_vals = [t.user_attrs.get("mrr", 0) for t in trials]
    nums = [t.number for t in trials]

    # Running best for miss rate (minimize) and MRR (maximize)
    running_miss = []
    best_miss = float("inf")
    for v in miss_vals:
        best_miss = min(best_miss, v)
        running_miss.append(best_miss)

    running_mrr = []
    best_mrr = float("-inf")
    for v in mrr_vals:
        best_mrr = max(best_mrr, v)
        running_mrr.append(best_mrr)

    # Miss rate y-axis bounds (left)
    miss_refs = [v for v in [current_miss, zeros_miss] + running_miss if v is not None]
    miss_lo = min(miss_refs)
    miss_hi = max(miss_refs)
    miss_margin = (miss_hi - miss_lo) * 0.08 or 0.01

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(nums, miss_vals, s=12, alpha=0.3, c=miss_vals,
               cmap="viridis", edgecolors="none", zorder=2)
    ax.plot(nums, running_miss,
            color="#66ccff", linewidth=2, label="Best miss rate", zorder=3)
    ax.axhline(current_miss, color="#ff6666", linestyle="--", linewidth=1,
               alpha=0.8, label=f"Current miss ({current_miss:.4f})")
    ax.axhline(zeros_miss, color="#999999", linestyle=":", linewidth=1,
               alpha=0.8, label=f"Zeros miss ({zeros_miss:.4f})")
    ax.set_ylim(miss_lo - miss_margin * 0.5, miss_hi + miss_margin)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Miss Rate (best@5k)", color="#66ccff")
    ax.tick_params(axis="y", labelcolor="#66ccff")

    # MRR y-axis (right)
    ax2 = ax.twinx()
    ax2.plot(nums, running_mrr,
             color="#66ff66", linewidth=2, label="Best MRR", zorder=3, alpha=0.8)
    ax2.axhline(current_mrr, color="#ff9966", linestyle="--", linewidth=1,
                alpha=0.6, label=f"Current MRR ({current_mrr:.4f})")
    ax2.axhline(zeros_mrr, color="#bbbbbb", linestyle=":", linewidth=1,
                alpha=0.6, label=f"Zeros MRR ({zeros_mrr:.4f})")
    ax2.set_ylabel("MRR", color="#66ff66")
    ax2.tick_params(axis="y", labelcolor="#66ff66")

    ax.set_title(f"Optimization History (threshold={threshold})")
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    path = out_dir / f"{study.study_name}-convergence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_importance(study, threshold, out_dir: Path):
    """Parameter importance: side-by-side bars for miss rate and MRR."""
    import matplotlib.pyplot as plt
    import optuna
    _setup_style()

    is_multi = hasattr(study, 'directions') and len(study.directions) > 1

    # Define all four metrics with their target functions
    metric_defs = []
    if is_multi:
        metric_defs.append(("AUC", lambda t: t.values[0]))
        metric_defs.append(("MRR", lambda t: t.values[1]))
        metric_defs.append(("Miss Rate", lambda t: t.user_attrs.get("miss_rate_5k", 0)))
        metric_defs.append(("Weighted Miss", lambda t: t.user_attrs.get("weighted_miss", 0)))
    else:
        metric_defs.append(("Primary", None))  # uses study default
        metric_defs.append(("MRR", lambda t: t.user_attrs.get("mrr", 0)))
        metric_defs.append(("Miss Rate", lambda t: t.user_attrs.get("miss_rate_5k", 0)))
        metric_defs.append(("Weighted Miss", lambda t: t.user_attrs.get("weighted_miss", 0)))

    # Compute importances for each metric
    all_imps = []
    for name, target_fn in metric_defs:
        try:
            if target_fn is None:
                imp = optuna.importance.get_param_importances(study)
            else:
                imp = optuna.importance.get_param_importances(study, target=target_fn)
            all_imps.append((name, imp))
        except Exception:
            pass

    if not all_imps:
        return None

    # Use first metric's param order (by importance) for all
    param_order = list(reversed(all_imps[0][1].keys()))
    n_params = len(param_order)
    n_metrics = len(all_imps)
    fig_h = max(4, n_params * 0.4)

    # 2x2 grid when we have 4 metrics, otherwise single row
    if n_metrics == 4:
        fig, axes = plt.subplots(2, 2, figsize=(10, fig_h * 2), sharey=True)
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, fig_h), sharey=True)
        if n_metrics == 1:
            axes = [axes]

    for ax, (name, imp) in zip(axes, all_imps):
        vals = [imp.get(p, 0.0) for p in param_order]
        max_v = max(vals) if vals and max(vals) > 0 else 1
        colors = plt.cm.viridis_r([v / max_v for v in vals])
        ax.barh(param_order, vals, color=colors, edgecolor="none", height=0.6)
        ax.set_xlabel("Importance (fANOVA)")
        ax.set_title(name)
        ax.set_xlim(0, max_v * 1.15)
        for i, v in enumerate(vals):
            ax.text(v + max_v * 0.01, i, f"{v:.3f}", va="center",
                    fontsize=8, color="#cccccc")

    fig.suptitle(f"Parameter Importance (threshold={threshold})", fontsize=12, y=1.02)
    fig.tight_layout()

    path = out_dir / f"{study.study_name}-importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_slices(study, threshold, out_dir: Path):
    """1D slice plots for each parameter. Two rows per metric for dual."""
    import matplotlib.pyplot as plt
    import numpy as np
    _setup_style()

    is_multi = hasattr(study, 'directions') and len(study.directions) > 1
    if is_multi:
        trials = [t for t in study.trials if t.values is not None]
    else:
        trials = [t for t in study.trials if t.value is not None]
    if len(trials) < 3:
        return None

    param_names = list(trials[0].params.keys())
    n = len(param_names)
    cols = min(n, 4)

    # For dual: two rows of metrics (AUC + MRR). Otherwise single row.
    if is_multi:
        metrics = [
            ("Weighted Miss AUC", [t.values[0] for t in trials], True,
             min(trials, key=lambda t: t.values[0])),
            ("MRR", [t.values[1] for t in trials], False,
             max(trials, key=lambda t: t.values[1])),
        ]
    else:
        minimize = study.direction.name == "MINIMIZE"
        y_label = "Miss Rate" if minimize else "MRR"
        vals = [t.value for t in trials]
        best_t = study.best_trial
        metrics = [(y_label, vals, minimize, best_t)]
        # Add MRR as second row if we have it and primary isn't MRR
        if minimize:
            mrr_vals = [t.user_attrs.get("mrr", 0) for t in trials]
            if max(mrr_vals) > 0:
                best_mrr_t = max(trials, key=lambda t: t.user_attrs.get("mrr", 0))
                metrics.append(("MRR", mrr_vals, False, best_mrr_t))

    n_metrics = len(metrics)
    rows_per = (n + cols - 1) // cols
    total_rows = rows_per * n_metrics

    fig, axes = plt.subplots(total_rows, cols,
                              figsize=(4 * cols, 3.2 * total_rows),
                              squeeze=False)

    for m_idx, (y_label, values_all, is_min, best_trial) in enumerate(metrics):
        cmap = "viridis" if is_min else "viridis_r"
        vmin, vmax = min(values_all), max(values_all)

        for i, pname in enumerate(param_names):
            row = m_idx * rows_per + i // cols
            col = i % cols
            ax = axes[row][col]
            xs = [t.params[pname] for t in trials]

            ax.scatter(xs, values_all, c=values_all, cmap=cmap, s=10,
                       alpha=0.5, edgecolors="none", vmin=vmin, vmax=vmax)
            ax.axvline(best_trial.params[pname], color="#ff6666",
                       linestyle="--", linewidth=1, alpha=0.7)
            ax.set_xlabel(pname, fontsize=8)
            if col == 0:
                ax.set_ylabel(y_label, fontsize=8)
            ax.tick_params(labelsize=7)
            title = f"{pname}" if m_idx == 0 else pname
            ax.set_title(title, fontsize=9, pad=3)

        # Hide unused subplots in this metric's rows
        for i in range(n, rows_per * cols):
            row = m_idx * rows_per + i // cols
            col = i % cols
            axes[row][col].set_visible(False)

    fig.suptitle(f"Parameter Slices (threshold={threshold})", fontsize=12, y=1.01)
    fig.tight_layout()

    path = out_dir / f"{study.study_name}-slices.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


MISS_RATE_BUDGETS = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 8000, 10000]


def compute_miss_rates(detailed_results: list[dict], token_map: dict[str, int],
                       budgets: list[int] | None = None) -> dict:
    """Compute miss rates at various token budgets.

    Returns dict with keys:
      "any_cutoff", "best_cutoff"   — miss rates with quality floor/cliff applied
      "any_nocutoff", "best_nocutoff" — miss rates using full pre-cutoff ranking
    Each value is a list of floats (one per budget), parallel with budgets.
    Also returns "budgets" and "n_queries".
    """
    budgets = budgets or MISS_RATE_BUDGETS
    n = len(detailed_results)
    if n == 0:
        zeros = [0.0] * len(budgets)
        return {"budgets": budgets, "n_queries": 0,
                "any_cutoff": zeros, "best_cutoff": zeros,
                "any_nocutoff": zeros, "best_nocutoff": zeros}

    any_cutoff = [0] * len(budgets)
    best_cutoff = [0] * len(budgets)
    any_nocutoff = [0] * len(budgets)
    best_nocutoff = [0] * len(budgets)

    for qr in detailed_results:
        relevant = qr["relevant"]
        best_mid = qr["best_relevant"]

        for bi, budget in enumerate(budgets):
            # With cutoff
            shown_cutoff = _shown_at_budget(qr["ranked"], token_map, budget)
            if shown_cutoff & relevant:
                any_cutoff[bi] += 1
            if best_mid in shown_cutoff:
                best_cutoff[bi] += 1

            # No cutoff (full ranked list)
            shown_full = _shown_at_budget(qr["full_ranked"], token_map, budget)
            if shown_full & relevant:
                any_nocutoff[bi] += 1
            if best_mid in shown_full:
                best_nocutoff[bi] += 1

    # Convert hits to miss rates
    return {
        "budgets": budgets,
        "n_queries": n,
        "any_cutoff": [1.0 - any_cutoff[i] / n for i in range(len(budgets))],
        "best_cutoff": [1.0 - best_cutoff[i] / n for i in range(len(budgets))],
        "any_nocutoff": [1.0 - any_nocutoff[i] / n for i in range(len(budgets))],
        "best_nocutoff": [1.0 - best_nocutoff[i] / n for i in range(len(budgets))],
    }


def _shown_at_budget(ranked_ids: list[str], token_map: dict[str, int],
                     budget: int) -> set[str]:
    """Walk ranked list accumulating tokens; return set of IDs that fit."""
    shown = set()
    used = 0
    for mid in ranked_ids:
        tokens = token_map.get(mid, 100)  # fallback 100 tokens
        if used + tokens > budget:
            break
        used += tokens
        shown.add(mid)
    return shown


def compute_weighted_miss(detailed_results: list[dict], token_map: dict[str, int],
                          feedback_raw: dict[str, dict], budget: int = 5000,
                          min_utility: float = 0.1, power: float = 1.5) -> float:
    """Utility-weighted miss rate at a given token budget.

    For each query, considers all memories with mean utility > min_utility.
    Each missed memory contributes weight = utility^power to the numerator.
    Denominator is the sum of all weights (hit or miss).

    Returns weighted miss rate in [0, 1]. Lower = better.
    """
    total_weight = 0.0
    missed_weight = 0.0

    for qr in detailed_results:
        shown = _shown_at_budget(qr["ranked"], token_map, budget)
        for mid in qr["relevant"]:
            fb = feedback_raw.get(mid)
            if not fb:
                continue
            u = fb["mean"]
            if u <= min_utility:
                continue
            w = u ** power
            total_weight += w
            if mid not in shown:
                missed_weight += w

    return missed_weight / total_weight if total_weight > 0 else 0.0


# Log-spaced budgets for AUC metric: 250 to 8000, 6 octaves
AUC_BUDGETS = [250, 500, 1000, 2000, 4000, 8000]


def compute_weighted_miss_auc(detailed_results: list[dict], token_map: dict[str, int],
                               feedback_raw: dict[str, dict],
                               min_utility: float = 0.1, power: float = 1.5,
                               budgets: list[int] | None = None) -> float:
    """Area under the weighted-miss-vs-budget curve in log-budget space.

    Computes weighted_miss at each log-spaced budget and averages.
    Equal weight per budget = uniform measure in log-space, since budgets
    are geometrically spaced. Lower = better.
    """
    budgets = budgets or AUC_BUDGETS
    values = [compute_weighted_miss(detailed_results, token_map, feedback_raw,
                                     budget=b, min_utility=min_utility, power=power)
              for b in budgets]
    return sum(values) / len(values)


def extract_ground_truth(path: str) -> dict[str, dict[str, float]]:
    """Load ground truth JSON: {query: {memory_id: relevance_score}}.

    Returns the data as-is. Scores are continuous 0.0-1.0.
    """
    with open(path) as f:
        return json.load(f)


def compute_ndcg(detailed_results: list[dict], token_map: dict[str, int],
                 ground_truth: dict[str, dict[str, float]], budget: int = 5000) -> float:
    """Normalized Discounted Cumulative Gain at a token budget.

    Uses ground truth relevance scores instead of binary feedback-derived labels.
    """
    ndcg_sum = 0.0
    n_queries = 0

    for qr in detailed_results:
        query = qr["query"]
        gt = ground_truth.get(query)
        if not gt:
            continue

        n_queries += 1
        shown = []
        used_tokens = 0
        for mid in qr["ranked"]:
            tokens = token_map.get(mid, 100)
            if used_tokens + tokens > budget:
                break
            used_tokens += tokens
            shown.append(mid)

        if not shown:
            continue

        # DCG
        dcg = 0.0
        for i, mid in enumerate(shown):
            rel = gt.get(mid, 0.0)
            dcg += rel / math.log2(i + 2)  # i+2 because log2(1)=0

        # Ideal DCG: sort all ground truth entries by relevance desc
        ideal_rels = sorted(gt.values(), reverse=True)
        idcg = 0.0
        ideal_used = 0
        for i, rel in enumerate(ideal_rels):
            # Approximate: assume each ideal memory uses ~avg tokens
            # For simplicity, use same k as shown length
            if i >= len(shown):
                break
            idcg += rel / math.log2(i + 2)

        if idcg > 0:
            ndcg_sum += dcg / idcg

    return ndcg_sum / n_queries if n_queries > 0 else 0.0


def compute_graded_recall(detailed_results: list[dict], token_map: dict[str, int],
                          ground_truth: dict[str, dict[str, float]], budget: int = 5000,
                          threshold: float = 0.5) -> float:
    """Fraction of memories with ground truth relevance >= threshold that appear within budget."""
    total_relevant = 0
    total_found = 0

    for qr in detailed_results:
        query = qr["query"]
        gt = ground_truth.get(query)
        if not gt:
            continue

        shown = _shown_at_budget(qr["ranked"], token_map, budget)
        relevant_ids = {mid for mid, score in gt.items() if score >= threshold}

        total_relevant += len(relevant_ids)
        total_found += len(relevant_ids & shown)

    return total_found / total_relevant if total_relevant > 0 else 0.0


def compute_discovery_rate(detailed_results: list[dict],
                           ground_truth: dict[str, dict[str, float]],
                           threshold: float = 0.7) -> float:
    """Fraction of high-relevance memories (>= threshold) appearing anywhere in results."""
    total_high = 0
    found_high = 0

    for qr in detailed_results:
        query = qr["query"]
        gt = ground_truth.get(query)
        if not gt:
            continue

        all_returned = set(qr.get("full_ranked", qr["ranked"]))
        high_ids = {mid for mid, score in gt.items() if score >= threshold}

        total_high += len(high_ids)
        found_high += len(high_ids & all_returned)

    return found_high / total_high if total_high > 0 else 0.0


def compute_mrr_at_budgets(detailed_results: list[dict], token_map: dict[str, int],
                           budgets: list[int] | None = None) -> dict:
    """Compute MRR at various token budgets.

    For each budget, MRR is computed over queries where at least one relevant
    memory fits within the budget. First relevant hit's reciprocal rank is used.

    Returns dict with keys: "mrr_cutoff", "mrr_nocutoff" (lists parallel to budgets).
    """
    budgets = budgets or MISS_RATE_BUDGETS
    n = len(detailed_results)
    if n == 0:
        zeros = [0.0] * len(budgets)
        return {"budgets": budgets, "mrr_cutoff": zeros, "mrr_nocutoff": zeros}

    mrr_cutoff = [0.0] * len(budgets)
    mrr_nocutoff = [0.0] * len(budgets)

    for qr in detailed_results:
        relevant = qr["relevant"]

        for bi, budget in enumerate(budgets):
            # With cutoff
            shown_cutoff = _shown_at_budget_ordered(qr["ranked"], token_map, budget)
            for rank, mid in enumerate(shown_cutoff, 1):
                if mid in relevant:
                    mrr_cutoff[bi] += 1.0 / rank
                    break

            # No cutoff
            shown_full = _shown_at_budget_ordered(qr["full_ranked"], token_map, budget)
            for rank, mid in enumerate(shown_full, 1):
                if mid in relevant:
                    mrr_nocutoff[bi] += 1.0 / rank
                    break

    return {
        "budgets": budgets,
        "mrr_cutoff": [mrr_cutoff[i] / n for i in range(len(budgets))],
        "mrr_nocutoff": [mrr_nocutoff[i] / n for i in range(len(budgets))],
    }


def _shown_at_budget_ordered(ranked_ids: list[str], token_map: dict[str, int],
                              budget: int) -> list[str]:
    """Walk ranked list accumulating tokens; return ordered list of IDs that fit."""
    shown = []
    used = 0
    for mid in ranked_ids:
        tokens = token_map.get(mid, 100)
        if used + tokens > budget:
            break
        used += tokens
        shown.append(mid)
    return shown


def print_miss_rate_table(miss_rates: dict):
    """Print miss rate table to console."""
    budgets = miss_rates["budgets"]
    n = miss_rates["n_queries"]
    print(f"\n  --- Miss Rate by Token Budget ({n} queries with relevant memories) ---")
    print(f"  {'Budget':>7}  {'Any Relevant':>14}  {'Best Relevant':>15}"
          f"  {'(no-cutoff any)':>17}  {'(no-cutoff best)':>18}")
    print("  " + "-" * 77)
    for i, b in enumerate(budgets):
        print(f"  {b:>7}  {miss_rates['any_cutoff'][i]:>13.1%}"
              f"  {miss_rates['best_cutoff'][i]:>15.1%}"
              f"  {miss_rates['any_nocutoff'][i]:>17.1%}"
              f"  {miss_rates['best_nocutoff'][i]:>18.1%}")


def plot_budget_curves(miss_rates: dict, mrr_budgets: dict | None,
                       threshold: float, out_dir: Path, study_name: str = ""):
    """Token budget curves — miss rate (left axis) + MRR (right axis)."""
    import matplotlib.pyplot as plt
    _setup_style()

    budgets = miss_rates["budgets"]
    fig, ax = plt.subplots(figsize=(10, 5))

    # Miss rate lines (left axis)
    miss_lines = [
        ("any_cutoff", "Miss: any (cutoff)", "#ff6666", "-"),
        ("best_cutoff", "Miss: best (cutoff)", "#66ccff", "-"),
        ("any_nocutoff", "Miss: any (no cutoff)", "#ff6666", "--"),
        ("best_nocutoff", "Miss: best (no cutoff)", "#66ccff", "--"),
    ]
    for key, label, color, ls in miss_lines:
        ax.plot(budgets, miss_rates[key], color=color, linestyle=ls,
                linewidth=2, marker="o", markersize=4, label=label)

    ax.set_xlabel("Token Budget")
    ax.set_ylabel("Miss Rate", color="#66ccff")
    ax.tick_params(axis="y", labelcolor="#66ccff")
    ax.set_ylim(-0.02, min(1.0, max(
        max(miss_rates["any_cutoff"]), max(miss_rates["best_cutoff"]),
    ) * 1.3 + 0.05))

    # MRR lines (right axis)
    if mrr_budgets:
        ax2 = ax.twinx()
        mrr_lines = [
            ("mrr_cutoff", "MRR (cutoff)", "#66ff66", "-"),
            ("mrr_nocutoff", "MRR (no cutoff)", "#66ff66", "--"),
        ]
        for key, label, color, ls in mrr_lines:
            ax2.plot(budgets, mrr_budgets[key], color=color, linestyle=ls,
                     linewidth=2, marker="s", markersize=3, label=label, alpha=0.8)
        ax2.set_ylabel("MRR", color="#66ff66")
        ax2.tick_params(axis="y", labelcolor="#66ff66")

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    else:
        ax.legend(loc="upper right", fontsize=9)

    ax.set_title(f"Token Budget Curves (threshold={threshold})")

    prefix = f"{study_name}-" if study_name else ""
    path = out_dir / f"{prefix}budget.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _ts():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _plot_corner_from_study(study, threshold, out_dir: Path):
    """Bridge: convert Optuna study to plot_tuning.py's corner plots (miss rate + MRR)."""
    import importlib.util
    import numpy as np
    import optuna
    spec = importlib.util.spec_from_file_location(
        "plot_tuning", Path(__file__).parent / "plot_tuning.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    is_multi = hasattr(study, 'directions') and len(study.directions) > 1
    trials = [t for t in study.trials
              if t.state == optuna.trial.TrialState.COMPLETE]
    if len(trials) < 10:
        return None

    param_names = list(trials[0].params.keys())
    params = {n: np.array([t.params[n] for t in trials]) for n in param_names}
    if is_multi:
        miss_values = np.array([t.values[0] for t in trials])  # AUC (obj 0)
    else:
        miss_values = np.array([t.value for t in trials])

    bounds = {}
    discrete = set()
    for n in param_names:
        dist = trials[0].distributions[n]
        bounds[n] = (dist.low, dist.high)
        if hasattr(dist, 'step') and isinstance(dist.low, int):
            step = getattr(dist, 'step', 1) or 1
            n_levels = (dist.high - dist.low) // step + 1
            if n_levels <= mod.DISCRETE_MAX_LEVELS:
                discrete.add(n)

    # Auto-detect log-scale params (same logic as plot_tuning.py load_study)
    log_scale = set()
    for n in param_names:
        lo, hi = bounds[n]
        if lo >= 1 and hi / lo >= 50:
            log_scale.add(n)

    paths = []

    # Build all 4 metric arrays: (name, values, minimize)
    corner_metrics = []
    if is_multi:
        corner_metrics.append(("weighted_miss_auc", miss_values, True))
        corner_metrics.append(("mrr", np.array([t.values[1] for t in trials]), False))
    else:
        minimize = study.direction.name == "MINIMIZE"
        metric_key = "miss_rate_5k" if minimize else "mrr"
        corner_metrics.append((metric_key, miss_values, minimize))
        mrr_vals = np.array([t.user_attrs.get("mrr", 0) for t in trials])
        if mrr_vals.max() > 0:
            corner_metrics.append(("mrr", mrr_vals, False))

    # Always add user_attr metrics if available
    for attr_key, is_min in [("miss_rate_5k", True), ("weighted_miss", True)]:
        # Skip if already covered by primary objective
        if any(m[0] == attr_key for m in corner_metrics):
            continue
        vals = np.array([t.user_attrs.get(attr_key, 0) for t in trials])
        if not (vals.max() == 0 and vals.min() == 0):
            corner_metrics.append((attr_key, vals, is_min))

    for metric_name, vals, is_minimize in corner_metrics:
        best_i = int(vals.argmin() if is_minimize else vals.argmax())
        bp = {n: params[n][best_i] for n in param_names}
        d = dict(
            param_names=param_names, params=params, values=vals,
            study_name=study.study_name,
            best_params=bp, discrete=discrete, bounds=bounds,
            log_scale=log_scale, minimize=is_minimize, metric=metric_name,
        )
        mod.plot_corner(d, str(out_dir))
        # plot_corner saves as {study_name}-{metric}.png or {study_name}.png (mrr)
        suffix = f"-{metric_name}" if metric_name != "mrr" else ""
        raw = out_dir / f"{study.study_name}{suffix}.png"
        final = out_dir / f"{study.study_name}-corner-{metric_name}.png"
        if raw.exists():
            final.unlink(missing_ok=True)
            raw.rename(final)
            paths.append(final)

    return paths if paths else None


def plot_mrr_vs_missrate(study, threshold, out_dir: Path):
    """Scatter grid: all pairs of 4 metrics (6 subplots) with Pareto fronts."""
    import matplotlib.pyplot as plt
    import numpy as np
    from itertools import combinations
    _setup_style()

    is_multi = hasattr(study, 'directions') and len(study.directions) > 1

    # Collect metric values for each trial
    if is_multi:
        trials = [t for t in study.trials if t.values is not None]
    else:
        trials = [t for t in study.trials if t.value is not None]
    if len(trials) < 10:
        return None

    # Build metric dict — prefer objectives for dual, user_attrs otherwise
    metrics = {}
    if is_multi:
        metrics["AUC"] = np.array([t.values[0] for t in trials])
        metrics["MRR"] = np.array([t.values[1] for t in trials])
    else:
        metrics["MRR"] = np.array([t.user_attrs.get("mrr", 0) for t in trials])

    # Always try user_attrs for the rest
    for key, label in [("weighted_miss_auc", "AUC"),
                        ("weighted_miss", "WM"),
                        ("miss_rate_5k", "MR@5k")]:
        if label in metrics:
            continue  # already from objectives
        vals = [t.user_attrs.get(key) for t in trials]
        if all(v is not None for v in vals):
            metrics[label] = np.array(vals)

    metric_names = list(metrics.keys())
    if len(metric_names) < 2:
        return None

    # Directions: lower is better except MRR
    minimize = {k: k != "MRR" for k in metric_names}

    pairs = list(combinations(metric_names, 2))
    n_pairs = len(pairs)
    cols = min(n_pairs, 3)
    rows = (n_pairs + cols - 1) // cols
    trial_nums = np.array([t.number for t in trials])

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows), squeeze=False)

    for idx, (xname, yname) in enumerate(pairs):
        ax = axes[idx // cols][idx % cols]
        xs = metrics[xname]
        ys = metrics[yname]

        sc = ax.scatter(xs, ys, s=12, alpha=0.4, c=trial_nums,
                        cmap="viridis", edgecolors="none", zorder=2)

        # Pareto front: "better" depends on direction for each metric
        # Sort by x descending if x is maximized, ascending if minimized
        x_min = minimize[xname]
        y_min = minimize[yname]

        # Walk sorted by x (better first), track best y
        if x_min:
            sorted_idx = np.argsort(xs)  # ascending = better for minimize
        else:
            sorted_idx = np.argsort(-xs)  # descending = better for maximize

        pareto_x, pareto_y = [], []
        best_y = float("inf") if y_min else float("-inf")
        for i in sorted_idx:
            is_better = (ys[i] < best_y) if y_min else (ys[i] > best_y)
            if is_better:
                best_y = ys[i]
                pareto_x.append(xs[i])
                pareto_y.append(ys[i])

        if len(pareto_x) >= 2:
            order = np.argsort(pareto_x)
            px = np.array(pareto_x)[order]
            py = np.array(pareto_y)[order]
            ax.plot(px, py, color="#ff6666", linewidth=1.5,
                    marker="o", markersize=4, alpha=0.8, label="Pareto", zorder=3)

        corr = np.corrcoef(xs, ys)[0, 1]
        ax.set_xlabel(xname, fontsize=10)
        ax.set_ylabel(yname, fontsize=10)
        ax.set_title(f"{xname} vs {yname}  (r={corr:.3f})", fontsize=10)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(loc="best", fontsize=8)

    # Hide unused
    for idx in range(n_pairs, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(f"Metric Pairs (n={len(trials)}, threshold={threshold})",
                 fontsize=13, y=1.01)
    fig.tight_layout()

    path = out_dir / f"{study.study_name}-scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_plots(study, threshold, current_miss, zeros_miss,
                    current_mrr, zeros_mrr, out_dir: Path,
                    full: bool = True):
    """Generate all CTT-style plots for a completed optimization run.

    When full=False (sub-stages), only generate Importance + Corner plots.
    Corner plots are skipped for full (refine) stages — 15D is too slow and
    sub-stage corners are more informative anyway.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  --- Generating Plots ---")
    paths = []
    # Fast plots first, corner last (slow)
    fast_plots = [
        ("Convergence", lambda: plot_convergence(study, threshold, current_miss,
                                                  zeros_miss, current_mrr,
                                                  zeros_mrr, out_dir)),
        ("Importance", lambda: plot_importance(study, threshold, out_dir)),
        ("Slices", lambda: plot_slices(study, threshold, out_dir)),
        ("Scatter", lambda: plot_mrr_vs_missrate(study, threshold, out_dir)),
    ]
    slow_plots = [
        ("Corner", lambda: _plot_corner_from_study(study, threshold, out_dir)),
    ]
    if full:
        all_plots = fast_plots + slow_plots  # corner runs last
    else:
        # Sub-stages: only Importance + Corner
        all_plots = [(n, fn) for n, fn in fast_plots + slow_plots
                     if n in ("Importance", "Corner")]
    for name, fn in all_plots:
        try:
            path = fn()
            if path:
                if isinstance(path, list):
                    for p in path:
                        print(f"    {name}: {p}")
                        paths.append(p)
                else:
                    print(f"    {name}: {path}")
                    paths.append(path)
            else:
                print(f"    {name}: skipped (insufficient data)")
        except Exception as e:
            print(f"    {name}: failed ({e})")
    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Tune memory system constants from historical data."
    )
    parser.add_argument("--group", type=int, choices=range(1, 8),
                        help="Only tune a specific group (1-7)")
    parser.add_argument("--stage", type=str, choices=list(STAGES.keys()),
                        default="staged",
                        help="Stage for Group 1: which params to search "
                             f"({', '.join(STAGES.keys())}). Default: staged")
    parser.add_argument("--apply", action="store_true",
                        help="Write changes to constants.py (default: dry-run)")
    parser.add_argument("--trials", type=int, default=500,
                        help="Number of Optuna trials for Group 1 (default: 500)")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Utility threshold for MRR ground truth (default: 0.4)")
    parser.add_argument("--runs", nargs="+", metavar="THRESH:TRIALS",
                        help="Multiple runs, e.g. --runs 0.3:500 0.5:200")
    parser.add_argument("--metric", type=str,
                        choices=["miss_rate", "mrr", "weighted_miss", "weighted_miss_auc", "dual",
                                 "ndcg_5k", "graded_recall_5k"],
                        default="weighted_miss",
                        help="Optimization objective: weighted_miss (utility-weighted "
                             "miss rate at 5k, minimize), "
                             "weighted_miss_auc (AUC over log-spaced budgets 250-8k, minimize), "
                             "miss_rate (best relevant at 5k tokens, minimize), "
                             "mrr (maximize), "
                             "dual (multi-objective: minimize AUC + maximize MRR, NSGA-II), "
                             "ndcg_5k (NDCG at 5k tokens, maximize, requires --ground-truth), "
                             "or graded_recall_5k (recall of >=0.5 memories at 5k, maximize, "
                             "requires --ground-truth). All metrics are always logged.")
    parser.add_argument("--miss-power", type=float, default=1.5,
                        help="Power for utility weighting in weighted_miss metric "
                             "(default: 1.5). Higher = more emphasis on high-utility memories.")
    parser.add_argument("--miss-min-util", type=float, default=0.1,
                        help="Minimum utility to count as relevant in weighted_miss "
                             "(default: 0.1). Memories at or below this are ignored.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers for Optuna trials (default: 1)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional tag appended to study name (e.g. --tag w4)")
    parser.add_argument("--reset", action="store_true",
                        help="Delete existing study data and start fresh")
    parser.add_argument("--ground-truth", type=str, default=None,
                        help="Path to ground truth JSON from build_ground_truth.py. "
                             "Enables NDCG, graded recall, and discovery rate metrics.")
    args = parser.parse_args()

    # Parse runs: either from --runs or from --threshold/--trials
    if args.runs:
        runs = []
        for spec in args.runs:
            parts = spec.split(":")
            if len(parts) != 2:
                parser.error(f"Invalid run spec '{spec}'. Use THRESHOLD:TRIALS, e.g. 0.3:500")
            runs.append((float(parts[0]), int(parts[1])))
    else:
        runs = [(args.threshold, args.trials)]

    all_thresholds = sorted(set([0.2, 0.4, 0.6] + [t for t, _ in runs]))
    # For weighted_miss variants, ensure the min_utility threshold is in the set
    if args.metric in ("weighted_miss", "weighted_miss_auc", "dual") and args.miss_min_util not in all_thresholds:
        all_thresholds = sorted(set(all_thresholds + [args.miss_min_util]))

    stage = args.stage

    print("=" * 70)
    print("Memory Tuning Script")
    print(f"  Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    if args.group:
        print(f"  Group: {args.group}")
    else:
        print(f"  Groups: all with sufficient data")
    if stage == "staged":
        seq = STAGES[stage]["sequence"]
        print(f"  Stage: staged -- {' -> '.join(seq)}")
    elif stage != "all":
        print(f"  Stage: {stage} -- {STAGES[stage]['desc']}")
        print(f"  Fixed: {', '.join(k for k in CURRENT_PARAMS if k not in STAGES[stage]['search'])}")
    if len(runs) == 1:
        print(f"  Trials: {runs[0][1]} @ threshold {runs[0][0]}")
    else:
        run_strs = [f"{t}:{n}" for t, n in runs]
        print(f"  Runs: {', '.join(run_strs)}")
    # Load ground truth if provided
    ground_truth = None
    if args.ground_truth:
        gt_path = Path(args.ground_truth)
        if gt_path.exists():
            ground_truth = extract_ground_truth(str(gt_path))
            print(f"  Ground truth: {len(ground_truth)} queries from {gt_path.name}")
        else:
            print(f"  WARNING: Ground truth file not found: {gt_path}")
    if args.metric in ("ndcg_5k", "graded_recall_5k") and not ground_truth:
        print("ERROR: --ground-truth is required for metric '%s'" % args.metric)
        sys.exit(1)
    print("=" * 70)

    db = get_db()
    all_changes = {}

    groups_to_run = [args.group] if args.group else list(range(1, 8))

    for g in groups_to_run:
        if g == 1:
            # Load data once, run optimization per threshold
            data = load_group1_data(db, all_thresholds)
            if ground_truth:
                data["ground_truth"] = ground_truth
                # Pre-filter queries to GT-only subset for faster GT-metric trials
                gt_query_set = set(ground_truth.keys())
                data["gt_queries"] = [q for q in data["queries"] if q["query"] in gt_query_set]
                print(f"  GT query subset: {len(data['gt_queries'])}/{len(data['queries'])} queries")
            for threshold, trial_count in runs:
                # Ensure this threshold has a relevant set
                if threshold not in data["relevant_sets"]:
                    data["relevant_sets"][threshold] = {
                        mid for mid, fb in data["feedback_raw"].items()
                        if fb["mean"] >= threshold
                    }

                if stage == "staged":
                    # Sequential pipeline: run each sub-stage, carry forward winners
                    sequence = STAGES["staged"]["sequence"]
                    # Trial budget: n·ln(n) scaling, largest stage = trial_count
                    import math
                    stage_dims = [len(STAGES[s]["search"]) for s in sequence]
                    weights = [d * math.log(max(d, 2)) for d in stage_dims]
                    max_weight = max(weights)
                    stage_trials = [max(10, 10 * math.ceil(trial_count * w / max_weight / 10)) for w in weights]
                    accumulated_params = {}
                    all_bounds = {}

                    for si, sub_stage in enumerate(sequence):
                        sub_trials = stage_trials[si]
                        print(f"\n{'='*70}")
                        print(f"  STAGED [{si+1}/{len(sequence)}]: {sub_stage} "
                              f"({len(STAGES[sub_stage]['search'])}D, {sub_trials} trials)")
                        print(f"{'='*70}")

                        result, bounds = run_group1_optimization(
                            data, sub_trials, threshold, all_thresholds,
                            reset=args.reset, stage=sub_stage,
                            base_params=accumulated_params if accumulated_params else None,
                            metric=args.metric, full_plots=False,
                            miss_power=args.miss_power, miss_min_util=args.miss_min_util,
                            workers=args.workers, tag=args.tag,
                        )
                        all_bounds.update(bounds)
                        if result:
                            # Convert constant names back to param names for base_params
                            reverse_map = {v: k for k, v in GROUP1_MAP.items()}
                            for const_name, value in result.items():
                                param_name = reverse_map.get(const_name, const_name)
                                accumulated_params[param_name] = value
                            all_changes.update(result)

                    # --- Refine: all 15 params with bootstrap-constrained ranges ---
                    if all_bounds:
                        # Budget: same as the largest stage (= trial_count)
                        refine_trials = trial_count
                        print(f"\n{'='*70}")
                        print(f"  REFINE: all {len(SEARCH_RANGES)}D, "
                              f"bootstrap-constrained ({refine_trials} trials)")
                        print(f"{'='*70}")

                        # Temporarily narrow SEARCH_RANGES
                        saved_ranges = dict(SEARCH_RANGES)
                        for p, (mean, std, lo, hi) in all_bounds.items():
                            orig_spec = SEARCH_RANGES[p]
                            orig_lo, orig_hi, ptype = orig_spec[:3]
                            if ptype == "int":
                                new_lo = max(orig_lo, int(math.floor(lo)))
                                new_hi = min(orig_hi, int(math.ceil(hi)))
                                if new_lo >= new_hi:
                                    new_lo, new_hi = max(orig_lo, new_lo - 1), min(orig_hi, new_hi + 1)
                            else:
                                new_lo, new_hi = lo, hi
                                if new_lo >= new_hi:
                                    # Expand by 10% of original range
                                    margin = (orig_hi - orig_lo) * 0.1
                                    new_lo = max(orig_lo, mean - margin)
                                    new_hi = min(orig_hi, mean + margin)
                            # Preserve log flag and any extra fields
                            SEARCH_RANGES[p] = (new_lo, new_hi) + orig_spec[2:]
                            print(f"    {p:<22} [{orig_lo}, {orig_hi}] -> [{new_lo:.4f}, {new_hi:.4f}]")

                        result, _ = run_group1_optimization(
                            data, refine_trials, threshold, all_thresholds,
                            reset=True, stage="all",
                            base_params=None, metric=args.metric,
                            miss_power=args.miss_power, miss_min_util=args.miss_min_util,
                            workers=args.workers, tag=args.tag,
                        )
                        if result:
                            all_changes.update(result)
                            reverse_map = {v: k for k, v in GROUP1_MAP.items()}
                            for const_name, value in result.items():
                                param_name = reverse_map.get(const_name, const_name)
                                accumulated_params[param_name] = value

                        # Restore original ranges
                        SEARCH_RANGES.update(saved_ranges)

                    # Final summary
                    final_params = dict(CURRENT_PARAMS)
                    final_params.update(accumulated_params)
                    print(f"\n{'='*70}")
                    print(f"  STAGED PIPELINE COMPLETE")
                    print(f"{'='*70}")
                    print(f"  Final params changed: {len(accumulated_params)}")
                    for k, v in sorted(accumulated_params.items()):
                        orig = CURRENT_PARAMS.get(k, "?")
                        print(f"    {k}: {orig} -> {v}")

                else:
                    result, _ = run_group1_optimization(data, trial_count, threshold, all_thresholds,
                                                       reset=args.reset, stage=stage,
                                                       metric=args.metric,
                                                       miss_power=args.miss_power,
                                                       miss_min_util=args.miss_min_util,
                                                       workers=args.workers,
                                                       tag=args.tag)
                    if result:
                        all_changes.update(result)
        else:
            tuners = {
                2: lambda: tune_group_2(db),
                3: lambda: tune_group_3(db),
                4: lambda: tune_group_4(db),
                5: lambda: tune_group_5(db),
                6: lambda: tune_group_6(db),
                7: lambda: tune_group_7(db),
            }
            result = tuners[g]()
            if result:
                all_changes.update(result)

    db.close()

    # --- Diff ---
    if all_changes:
        print(f"\n{'=' * 70}")
        print(f"PROPOSED CHANGES ({len(all_changes)} constants)")
        print(f"{'=' * 70}")

        diff_text, modified_text = generate_diff(all_changes, CONSTANTS_PATH)
        if diff_text:
            print(diff_text)
        else:
            print("  (no textual changes detected)")

        if args.apply:
            apply_changes(modified_text, CONSTANTS_PATH)
            print("  Changes applied.")
        else:
            print("\n  Dry-run complete. Use --apply to write changes.")
    else:
        print(f"\n{'=' * 70}")
        print("No changes recommended.")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
