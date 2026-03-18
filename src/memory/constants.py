"""Tuning constants for the memory system.

Organized in three tiers:
- Core: wm1 study validated (RRF, feedback, adjacency, Hebbian)
- Secondary: Tuned occasionally (decay rates, shadow, confidence)
- Fixed: Structural (embedding model, thresholds, sleep limits)
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Data directory — configurable via env var, defaults to ~/.somnigraph/
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.environ.get("SOMNIGRAPH_DATA_DIR",
                Path.home() / ".somnigraph"))

# Reranker model path — learned scoring replaces the hand-tuned formula when present
MODEL_PATH = DATA_DIR / "tuning_studies" / "reranker_model.pkl"

# ---------------------------------------------------------------------------
# Core — wm15 study (2026-03-09), NSGA-II dual AUC+MRR, 500 trials, k=6 fixed
# ---------------------------------------------------------------------------

RRF_K = 6                    # wm13 best. Confirmed across wm12 (6D) and wm13 (4D). Was 14 (wm9).
RRF_VEC_WEIGHT = 0.5        # Weight for vector vs keyword: score = w*vec + (1-w)*kw. 0.5 = equal (baseline).
FEEDBACK_COEFF = 0.0         # Deprecated — replaced by UCB exploration bonus. Was 5.15 (wm9).
UCB_COEFF = 1.0              # Exploration bonus: score *= (1 + UCB_COEFF * sqrt(posterior_var)). Tune this.
ADJACENCY_BASE_BOOST = 0.33  # wm9: flat landscape, restored original. Was 0.93 (wm7).
ADJACENCY_NOVELTY_FLOOR = 0.67  # wm9 bootstrap mean. Was 0.05 (wm7).
HEBBIAN_COEFF = 3.0          # wm9 bootstrap mean (rounded). Was 1.06 (wm7).
HEBBIAN_CAP = 0.21           # wm15 Pareto #4. Was 0.07 (wm9). Old "degrades above 0.1" was wrong at k=6.
HEBBIAN_MIN_JOINT = 2        # min co-retrieval count before PMI contributes
CONTEXT_RELEVANCE_THRESHOLD = 0.5  # wm9: flat landscape, kept. Min cos(query, linking_emb) for edge to fire.

# ---------------------------------------------------------------------------
# Secondary — tune occasionally
# ---------------------------------------------------------------------------

# Per-category decay rates: decay_rate = ln(2) / half_life_days
# Higher = faster decay. 0 = timeless. NULL = use category default.
CATEGORY_DECAY_RATES = {
    "meta":        0.004,   # ~173 day half-life — foundational, rarely changes
    "reflection":  0.006,   # ~116 day half-life — insights age slowly
    "semantic":    0.008,   # ~87 day half-life  — anchored near current 90d global
    "procedural":  0.012,   # ~58 day half-life  — conventions shift; pinned patterns bypass decay
    "episodic":    0.023,   # ~30 day half-life  — session events fade fastest
}
DEFAULT_DECAY_RATE = 0.008  # semantic default, matches current behavior
DECAY_FLOOR = 0.35

# REMOVED from scoring — kept as metadata columns
# SHADOW_PENALTY_COEFF = 0.005
# CONFIDENCE_WEIGHT = 0.04

# REMOVED: QUALITY_FLOOR_RATIO (was 0.0, dead code)
# REMOVED: CLIFF_Z_THRESHOLD, CLIFF_MIN_RESULTS — cliff detection replaced by agent-specified limit parameter.
# See docs/roadmap.md § "Can cutoff history calibrate the cliff detector?" for the analysis.

# Confidence learning rates (Group 3)
CONF_USEFUL_THRESHOLD = 0.5    # utility >= this triggers growth
CONF_USELESS_THRESHOLD = 0.1   # utility < this triggers decay
CONF_GROWTH_RATE = 0.08        # asymptotic growth coefficient
CONF_DECAY_RATE = 0.07         # multiplicative decay (conf *= 1 - rate)
CONF_DURABILITY_NUDGE = 0.05   # durability signal scaling factor
CONF_DEFAULT = 0.5             # initial confidence for new/null memories
CONF_MIN_DELTA = 0.001         # minimum |delta| to apply update

# Adjacency expansion capacity (Group 5)
ADJACENCY_SEED_COUNT = 5       # Top-scoring memories to expand from
MAX_NEIGHBORS_PER_SEED = 5     # Max neighbors per seed (legacy adjacency only)
MAX_EXPANSION_TOTAL = 20       # Absolute cap on new neighbors

# PPR expansion (replaces adjacency BFS) — Phase 17, wm19 (906 trials, 4D)
PPR_DAMPING = 0.775            # wm19. Was 0.5. Higher = more graph walk, less teleport.
PPR_BOOST_COEFF = 2.0          # wm19. Was 0.33. At search ceiling — true optimum may be higher.
PPR_MIN_SCORE = 0.007          # wm19. Was 0.001. Stricter filter reduces noise. (Formula path only.)
PPR_RERANKER_SEEDS = 30        # Reranker uses more seeds than the formula path for better graph coverage.
PPR_MAX_ITER = 50              # convergence limit (not tuned)
PPR_CONVERGENCE_TOL = 1e-6     # early stopping (not tuned)

# DEPRECATED by PPR: ADJACENCY_BASE_BOOST, ADJACENCY_NOVELTY_FLOOR, CONTEXT_RELEVANCE_THRESHOLD
THEME_BOOST = 0.19             # wm19. Was 0.97 (wm15). PPR absorbs much of theme boost's prior role.
# REMOVED: FEEDBACK_MIN_COUNT — replaced by empirical Bayes Beta prior (scoring._compute_beta_prior).
# Beta(a,b) fitted from population feedback via method of moments. a+b (~11) replaces the old alpha=1 shrinkage.

# Feedback processing (Group 7)
DECAY_DURABILITY_SCALE = 0.2   # Max % change to decay_rate per durability signal
EDGE_WEIGHT_POS_STEP = 0.02    # Edge weight increase per useful retrieval
EDGE_WEIGHT_NEG_STEP = 0.01    # Edge weight decrease per useless retrieval
THEME_REFINE_THRESHOLD = 0.6   # Min utility to add query terms as themes
MAX_THEMES = 12                # Max themes per memory
MAX_NEW_TERMS = 3              # Max theme terms added per feedback cycle

# Feedback-time enrichment thresholds (empirically calibrated 2026-03-07)
CO_UTILITY_THRESHOLD = 0.7    # Min utility for co-utility edge creation between co-retrieved memories
BRIDGE_THEME_THRESHOLD = 0.8  # Min utility for adding bridge themes (query terms not in content)
BRIDGE_TERM_MIN_LEN = 4       # Min length for bridge theme terms (filters noise like "next", "plan")

# Disappointed-recall detection (empirically calibrated 2026-03-07)
DISAPPOINTED_RECALL_MAX_UTIL = 0.2   # All memories at or below this = disappointed
DISAPPOINTED_RETRIEVAL_SCORE = 0.25  # top_score above this = retrieval failure; below = coverage gap

# REMOVED: Intent-aware query routing (0% fire rate, 10/10 reviewers agree remove)

# ---------------------------------------------------------------------------
# Fixed — structural
# ---------------------------------------------------------------------------

EMBEDDING_BACKEND = os.environ.get("SOMNIGRAPH_EMBEDDING_BACKEND", "openai")

_BACKEND_CONFIG = {
    "openai": {"model": "text-embedding-3-small", "dim": 1536},
    "fastembed": {"model": "BAAI/bge-small-en-v1.5", "dim": 384},
}

if EMBEDDING_BACKEND not in _BACKEND_CONFIG:
    raise ValueError(f"Unknown embedding backend: {EMBEDDING_BACKEND}. Must be one of: {list(_BACKEND_CONFIG)}")

EMBEDDING_MODEL = _BACKEND_CONFIG[EMBEDDING_BACKEND]["model"]
EMBEDDING_DIM = _BACKEND_CONFIG[EMBEDDING_BACKEND]["dim"]

DEDUP_THRESHOLD = 0.1  # cosine distance; similarity > 0.9
PENDING_STALE_DAYS = 14

# Sleep skill limits
MAX_EDGES_PER_CYCLE = 5
MAX_PRIORITY_BOOST_PER_CYCLE = 2
