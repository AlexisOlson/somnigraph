"""Tuning constants for the memory system.

Organized in three tiers:
- Core: wm38 study (2026-03-16), 500q GT, blended ratio=0.5
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
# Core — wm38 study (2026-03-16), 500q GT, blended ratio=0.5, tight refinement
# ---------------------------------------------------------------------------

RRF_K = 6                    # wm13 best. Legacy — used as PPR seed k. Scoring uses K_FTS/K_VEC.
K_FTS = 8.002                # wm38. Was 6.593 (wm37). 12D tight, +79bp blended.
K_VEC = 6.845                # wm38. Was 5.637 (wm37).
RRF_VEC_WEIGHT = 0.505       # wm38. Was 0.532 (wm37).
FEEDBACK_COEFF = 0.0         # Deprecated — replaced by UCB exploration bonus. Was 2.15 (wm27).
UCB_COEFF = 0.840            # wm38. Was 0.805 (wm37).
ADJACENCY_BASE_BOOST = 0.33  # DEPRECATED by PPR.
ADJACENCY_NOVELTY_FLOOR = 0.67  # DEPRECATED by PPR.
HEBBIAN_COEFF = 0.001746     # wm36. 2D NDCG tune, top-10 cluster 0.00174±0.00004. Was 0.0016 (wm35).
HEBBIAN_CAP = 0.275          # wm38. Was 0.285 (wm37).
HEBBIAN_MIN_JOINT = 2        # min co-retrieval count before PMI contributes
CONTEXT_RELEVANCE_THRESHOLD = 0.5  # DEPRECATED by PPR.

W_THEME = 0.116              # wm38. Was 0.087 (wm37). Theme channel weight.
K_THEME = 4.924              # wm38. Was 3.327 (wm37). Softer theme fusion.
EWMA_ALPHA = 0.431           # wm38. Was 0.375 (wm37). More recency in feedback.
BM25_SUMMARY_WT = 13.278     # wm38. Was 15.033 (wm37). Less summary dominance.
BM25_THEMES_WT = 5.731       # wm38. Was 6.337 (wm37).

# ---------------------------------------------------------------------------
# Secondary — tune occasionally
# ---------------------------------------------------------------------------

# Per-category decay rates: decay_rate = ln(2) / half_life_days
# Higher = faster decay. 0 = timeless. NULL = use category default.
CATEGORY_DECAY_RATES = {
    "entity":      0.0,     # timeless — refreshed during sleep, not decayed
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

# PPR expansion (replaces adjacency BFS) — wm38 (2026-03-16), tight refinement
PPR_DAMPING = 0.216            # wm38. Was 0.227 (wm37).
PPR_BOOST_COEFF = 1.591        # wm38. Was 1.807 (wm37).
PPR_MIN_SCORE = 0.0            # wm36. Was 0.0002. Near-zero; no signal in 0-0.0005 range.
PPR_RERANKER_SEEDS = 30        # Reranker uses more seeds than the formula path for better graph coverage.
PPR_MAX_ITER = 50              # convergence limit (not tuned)
PPR_CONVERGENCE_TOL = 1e-6     # early stopping (not tuned)

# DEPRECATED by PPR: ADJACENCY_BASE_BOOST, ADJACENCY_NOVELTY_FLOOR, CONTEXT_RELEVANCE_THRESHOLD
THEME_BOOST = 0.0              # DEPRECATED — replaced by W_THEME/K_THEME (third RRF channel).
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

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

DEDUP_THRESHOLD = 0.1  # cosine distance; similarity > 0.9
PENDING_STALE_DAYS = 14

# Sleep skill limits
MAX_EDGES_PER_CYCLE = 5
MAX_PRIORITY_BOOST_PER_CYCLE = 2
