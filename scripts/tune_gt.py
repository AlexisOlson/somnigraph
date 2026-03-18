# /// script
# requires-python = ">=3.11"
# dependencies = ["sqlite-vec>=0.1.6", "openai>=2.0.0", "numpy>=1.26", "optuna>=4.0", "matplotlib>=3.8", "scikit-learn>=1.0", "scipy>=1.10"]
# ///
"""
Tune memory scoring params against hand-judged ground truth.

Loads real memory DB, pre-computes FTS+vec searches for GT queries,
then runs Optuna TPE over 10 scoring params optimizing NDCG@5k.

Usage:
  uv run tune_gt.py --trials 500 --jobs 6
  uv run tune_gt.py --trials 500 --jobs 6 --tag wm24 --adaptive
  uv run tune_gt.py --gt ~/Downloads/ground_truth.json --trials 500  # custom GT

Default GT: ~/.claude/data/tuning_studies/gt_snapshot.json
Snapshot with: copy ground_truth.json gt_snapshot.json
"""

import argparse
import json
import math
import pickle
import sqlite3
import struct
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup: import from memory package
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from memory.constants import (
    UCB_COEFF,
    HEBBIAN_COEFF, HEBBIAN_CAP, HEBBIAN_MIN_JOINT,
    PPR_DAMPING, PPR_BOOST_COEFF, PPR_MIN_SCORE, PPR_RERANKER_SEEDS,
    RRF_VEC_WEIGHT, RRF_K,
)

# Production-only constants (not in somnigraph constants.py -- the reranker
# makes these irrelevant for live scoring, but tuning tools still need them).
K_FTS = 8.002
K_VEC = 6.845
W_THEME = 0.116
K_THEME = 4.924
BM25_SUMMARY_WT = 13.278
BM25_THEMES_WT = 5.731
from memory.embeddings import embed_batch
from memory.fts import sanitize_fts_query
from memory.scoring import personalized_pagerank

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

from memory.constants import DATA_DIR
from memory.db import DB_PATH as MEMORY_DB

# ---------------------------------------------------------------------------
# Production params
# ---------------------------------------------------------------------------

PRODUCTION_PARAMS = {
    "k_fts": 8.002,
    "k_vec": 6.845,
    "w_vec": 0.505,
    "ucb_coeff": 0.840,
    "w_theme": 0.116,
    "k_theme": 4.924,
    "ppr_boost": 1.591,
    "ppr_damping": 0.216,
    "ppr_min_score": 0.0,
    "hebbian_coeff": 0.001746,
    "hebbian_cap": 0.275,
    "ewma_alpha": 0.431,
    "bm25_summary_wt": 13.278,
    "bm25_themes_wt": 5.731,
}

SEARCH_RANGES = {
    "k_fts": (7.0, 9.0, "float"),      # wm38 peak ~8
    "k_vec": (6.0, 7.0, "float"),      # wm38 peak ~6.85
    "w_vec": (0.49, 0.52, "float"),    # wm38 peak ~0.505
    "ucb_coeff": (0.75, 1.0, "float"), # wm38 peak ~0.84
    "w_theme": (0.075, 0.15, "float"), # wm38 peak ~0.116
    "k_theme": (4.0, 6.0, "float"),    # wm38 peak ~4.92
    "ppr_boost": (1.5, 1.75, "float"), # wm38 peak ~1.59
    "ppr_damping": (0.20, 0.24, "float"),   # wm38 peak ~0.216
    "ppr_min_score": (0.0, 0.0005, "float"),
    "hebbian_coeff": (0.0015, 0.002, "float"),
    "hebbian_cap": (0.25, 0.325, "float"),  # wm38 peak ~0.275
    "ewma_alpha": (0.375, 0.45, "float"),   # wm38 peak ~0.431
    "bm25_summary_wt": (12.0, 15.0, "float"), # wm38 peak ~13.3
    "bm25_themes_wt": (5.0, 6.0, "float"),  # wm38 peak ~5.73
}

PPR_MAX_SEEDS = PPR_RERANKER_SEEDS  # shared with live scoring
PPR_DAMPING_GRID = sorted(set(
    [round(0.10 + i * 0.05, 2) for i in range(17)]    # 0.10..0.90 coarse
    + [round(0.20 + i * 0.01, 2) for i in range(31)]  # 0.20..0.50 fine
    + [round(0.18 + i * 0.002, 3) for i in range(51)]  # 0.180..0.280 milli
    + [round(0.200 + i * 0.001, 3) for i in range(41)] # 0.200..0.240 micro
))

# Short param names for display
_SHORT_NAMES = {
    "k_fts": "kf", "k_vec": "kv", "w_vec": "wv",
    "ucb_coeff": "ucb", "w_theme": "wt", "k_theme": "kt",
    "ppr_boost": "pprb", "ppr_damping": "damp", "ppr_min_score": "pmin",
    "hebbian_coeff": "heb", "hebbian_cap": "hcap",
    "ewma_alpha": "ewa",
    "bm25_summary_wt": "bsw", "bm25_themes_wt": "btw",
}

# BM25 weight grid for FTS cache
BM25_SUMMARY_GRID = [round(1.0 + i * 1.0, 1) for i in range(10)]   # 1.0..10.0
BM25_THEMES_GRID = [round(0.5 + i * 0.5, 1) for i in range(16)]    # 0.5..8.0

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def get_db() -> sqlite3.Connection:
    import sqlite_vec  # noqa: F401
    db = sqlite3.connect(str(MEMORY_DB))
    db.enable_load_extension(True)
    db.load_extension(sqlite_vec.loadable_path())
    db.enable_load_extension(False)
    db.row_factory = sqlite3.Row
    return db


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(db: sqlite3.Connection, gt: dict[str, dict[str, float]]):
    """Load everything needed for scoring from real memory DB."""

    gt_queries = list(gt.keys())
    print(f"  GT queries: {len(gt_queries)}")

    # Active memories
    active_ids = set()
    for row in db.execute("SELECT id FROM memories WHERE status='active'"):
        active_ids.add(row["id"])
    print(f"  Active memories: {len(active_ids)}")

    # Token map
    token_map = dict(db.execute(
        "SELECT id, coalesce(token_count, length(content)/4) FROM memories WHERE status='active'"
    ).fetchall())

    # Themes map (lowercased for case-insensitive matching against query tokens)
    themes_map: dict[str, set[str]] = {}
    for row in db.execute("SELECT id, themes FROM memories WHERE status='active'"):
        if row["themes"]:
            try:
                themes_map[row["id"]] = {t.lower() for t in json.loads(row["themes"])}
            except (json.JSONDecodeError, TypeError):
                themes_map[row["id"]] = set()
        else:
            themes_map[row["id"]] = set()

    # Feedback — store raw utility sequence per memory for EWMA at variable alpha
    fb_rows = db.execute("""
        SELECT memory_id, context FROM memory_events
        WHERE event_type='feedback' AND context IS NOT NULL
        ORDER BY created_at
    """).fetchall()
    feedback_raw: dict[str, dict] = defaultdict(lambda: {"utilities": [], "count": 0})
    for r in fb_rows:
        try:
            ctx = json.loads(r["context"])
            if "utility" not in ctx:
                continue
            mid = r["memory_id"]
            feedback_raw[mid]["utilities"].append(ctx["utility"])
            feedback_raw[mid]["count"] += 1
        except (json.JSONDecodeError, TypeError):
            continue
    feedback_raw = dict(feedback_raw)
    print(f"  Memories with feedback: {len(feedback_raw)}")

    # Hebbian co-retrieval data
    hebb_rows = db.execute("""
        SELECT query, memory_id FROM memory_events
        WHERE event_type='retrieved' AND query IS NOT NULL AND query != ''
    """).fetchall()
    hebb_query_mems: dict[str, set[str]] = defaultdict(set)
    hebb_mem_freq: dict[str, set[str]] = defaultdict(set)
    for r in hebb_rows:
        q, mid = r["query"], r["memory_id"]
        hebb_query_mems[q].add(mid)
        hebb_mem_freq[mid].add(q)
    hebb_total_queries = len(hebb_query_mems)
    hebb_data = {
        "mem_freq": dict(hebb_mem_freq),
        "total_queries": hebb_total_queries,
    }
    print(f"  Hebbian: {hebb_total_queries} queries, {len(hebb_mem_freq)} memories")

    # Edges (for PPR) — exclude contradiction-flagged edges
    edge_rows = db.execute("""
        SELECT source_id, target_id, weight, flags FROM memory_edges
    """).fetchall()
    ppr_adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
    n_contradiction = 0
    for r in edge_rows:
        if r["flags"] and "contradiction" in r["flags"]:
            n_contradiction += 1
            continue
        s, t, w = r["source_id"], r["target_id"], r["weight"] or 1.0
        if s in active_ids and t in active_ids:
            ppr_adj[s].append((t, w))
            ppr_adj[t].append((s, w))
    ppr_adj = dict(ppr_adj)
    print(f"  Edges: {len(edge_rows)} raw ({n_contradiction} contradiction excluded), "
          f"{len(ppr_adj)} memories with edges")

    # Memory embeddings (for PPR subgraph)
    mem_embs: dict[str, list[float]] = {}
    emb_rows = db.execute("""
        SELECT m.id, v.embedding
        FROM memories m
        JOIN memory_rowid_map rm ON rm.memory_id = m.id
        JOIN memory_vec v ON v.rowid = rm.rowid
        WHERE m.status = 'active'
    """).fetchall()
    for r in emb_rows:
        blob = r["embedding"]
        dim = len(blob) // 4
        mem_embs[r["id"]] = list(struct.unpack(f"{dim}f", blob))
    print(f"  Memory embeddings: {len(mem_embs)}")

    # Vector input map (context used for vec search, may differ from query text)
    vector_input_map: dict[str, str] = {}
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

    return {
        "gt_queries": gt_queries,
        "active_ids": active_ids,
        "token_map": token_map,
        "themes_map": themes_map,
        "feedback_raw": feedback_raw,
        "hebb_data": hebb_data,
        "ppr_adj": ppr_adj,
        "memory_embeddings": mem_embs,
        "vector_input_map": vector_input_map,
    }


# ---------------------------------------------------------------------------
# Pre-compute searches
# ---------------------------------------------------------------------------


def precompute_searches(db: sqlite3.Connection, gt_queries: list[str],
                        vector_input_map: dict[str, str],
                        embed_cache_path: Path) -> dict:
    """Pre-compute FTS + vec search results for all GT queries."""

    # Determine vec search texts
    vec_texts = []
    for q in gt_queries:
        vec_texts.append(vector_input_map.get(q, q))

    # Embed queries (with cache)
    all_texts = list(set(vec_texts))
    if embed_cache_path.exists():
        with open(embed_cache_path, "rb") as f:
            embed_cache = pickle.load(f)
        print(f"  Embedding cache loaded: {len(embed_cache)} entries")
    else:
        embed_cache = {}

    missing = [t for t in all_texts if t not in embed_cache]
    if missing:
        print(f"  Embedding {len(missing)} new texts...")
        new_embs = embed_batch(missing)
        for text, emb in zip(missing, new_embs):
            embed_cache[text] = emb
        with open(embed_cache_path, "wb") as f:
            pickle.dump(embed_cache, f)
        print(f"  Cache updated: {len(embed_cache)} entries")

    q_embeddings = {}
    for q, vt in zip(gt_queries, vec_texts):
        q_embeddings[q] = embed_cache[vt]

    # FTS search (production BM25 weights — live queries used for non-production weights)
    print(f"  Running FTS searches (BM25 weights: {BM25_SUMMARY_WT}, {BM25_THEMES_WT})...")
    fts_results: dict[str, list[str]] = {}
    fts_scores: dict[str, dict[str, float]] = {}
    for q in gt_queries:
        sanitized = sanitize_fts_query(q)
        if not sanitized:
            fts_results[q] = []
            continue
        try:
            rows = db.execute(f"""
                SELECT rm.memory_id, bm25(memory_fts, {BM25_SUMMARY_WT}, {BM25_THEMES_WT}) as score
                FROM memory_fts
                JOIN memory_rowid_map rm ON rm.rowid = memory_fts.rowid
                WHERE memory_fts MATCH ?
                ORDER BY score
                LIMIT 200
            """, (sanitized,)).fetchall()
            fts_results[q] = [r["memory_id"] for r in rows]
            fts_scores[q] = {r["memory_id"]: r["score"] for r in rows}
        except Exception:
            fts_results[q] = []
            fts_scores[q] = {}

    # Vec search
    print("  Running vec searches...")
    vec_results: dict[str, list[str]] = {}
    vec_scores: dict[str, dict[str, float]] = {}
    for q in gt_queries:
        emb = q_embeddings[q]
        blob = struct.pack(f"{len(emb)}f", *emb)
        rows = db.execute("""
            SELECT rm.memory_id, distance
            FROM memory_vec
            JOIN memory_rowid_map rm ON rm.rowid = memory_vec.rowid
            WHERE embedding MATCH ?
            AND k = 200
            ORDER BY distance
        """, (blob,)).fetchall()
        vec_results[q] = [r["memory_id"] for r in rows]
        vec_scores[q] = {r["memory_id"]: r["distance"] for r in rows}

    print(f"  FTS: {sum(len(v) for v in fts_results.values())} total hits")
    print(f"  Vec: {sum(len(v) for v in vec_results.values())} total hits")

    return {
        "fts_results": fts_results,
        "vec_results": vec_results,
        "fts_scores": fts_scores,
        "vec_scores": vec_scores,
        "q_embeddings": q_embeddings,
    }


# ---------------------------------------------------------------------------
# PPR cache
# ---------------------------------------------------------------------------


def precompute_ppr_cache(gt_queries: list[str], search_data: dict,
                         ppr_adj: dict, cache_path: Path) -> dict:
    """Pre-compute PPR scores for all GT queries across damping grid."""

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"  PPR cache loaded: {len(cache)} entries")
        # Check for missing grid entries and fill incrementally
        missing_dampings = set()
        if cache and gt_queries:
            sample_q = gt_queries[0]
            for d in PPR_DAMPING_GRID:
                if (round(d, 3), sample_q) not in cache:
                    missing_dampings.add(d)
        if not missing_dampings:
            return cache
        print(f"  Extending PPR cache with {len(missing_dampings)} new damping values...")
    else:
        missing_dampings = None  # compute all

    n_dampings = len(missing_dampings) if missing_dampings else len(PPR_DAMPING_GRID)
    print(f"  Computing PPR cache ({len(gt_queries)} queries x {n_dampings} dampings)...")
    fts_results = search_data["fts_results"]
    vec_results = search_data["vec_results"]

    if not cache_path.exists():
        cache: dict[tuple, dict] = {}

    for qi, q in enumerate(gt_queries):
        fts_ids = fts_results.get(q, [])
        vec_ids = vec_results.get(q, [])
        fts_ranked = {mid: rank for rank, mid in enumerate(fts_ids)}
        vec_ranked = {mid: rank for rank, mid in enumerate(vec_ids)}
        text_ids = set(fts_ranked.keys()) | set(vec_ranked.keys())

        # Base RRF scores for seed selection (symmetric RRF_K)
        base_scores = {}
        for mid in text_ids:
            score = 0.0
            if mid in fts_ranked:
                score += (1 - RRF_VEC_WEIGHT) / (RRF_K + fts_ranked[mid] + 1)
            if mid in vec_ranked:
                score += RRF_VEC_WEIGHT / (RRF_K + vec_ranked[mid] + 1)
            base_scores[mid] = score

        if not base_scores:
            continue

        top_seeds = sorted(base_scores, key=base_scores.get, reverse=True)[:PPR_MAX_SEEDS]
        seed_weights = {sid: base_scores[sid] for sid in top_seeds}
        seed_set = set(seed_weights)

        # 2-hop subgraph
        hop1 = set()
        for sid in seed_set:
            for neighbor, _w in ppr_adj.get(sid, []):
                hop1.add(neighbor)
        hop1 -= seed_set
        subgraph_nodes = seed_set | hop1
        for nid in hop1:
            for neighbor, _w in ppr_adj.get(nid, []):
                subgraph_nodes.add(neighbor)
        sub_adj: dict[str, list[tuple[str, float]]] = {}
        for node in subgraph_nodes:
            if node in ppr_adj:
                sub_adj[node] = [(n, w) for n, w in ppr_adj[node] if n in subgraph_nodes]

        if not sub_adj:
            continue

        dampings_to_compute = sorted(missing_dampings) if missing_dampings else PPR_DAMPING_GRID
        for damping in dampings_to_compute:
            raw_ppr = personalized_pagerank(sub_adj, seed_weights, damping=damping)
            ppr_scores = {mid: ps for mid, ps in raw_ppr.items()
                          if mid not in seed_set and ps > 0}
            cache[(round(damping, 3), q)] = ppr_scores

        if (qi + 1) % 20 == 0:
            print(f"    {qi + 1}/{len(gt_queries)} queries done")

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"  PPR cache saved: {len(cache)} entries")
    return cache


# ---------------------------------------------------------------------------
# Scoring pipeline (from bench_locomo_ablation.py)
# ---------------------------------------------------------------------------


def _ewma(utilities: list[float], alpha: float) -> float:
    """Compute EWMA over a sequence of utility values."""
    val = utilities[0]
    for u in utilities[1:]:
        val = alpha * u + (1 - alpha) * val
    return val


def _compute_beta_prior(feedback_raw: dict, alpha: float) -> tuple[float, float]:
    if not feedback_raw:
        return 0.25, 1.0
    means_2plus = [_ewma(fb["utilities"], alpha)
                   for fb in feedback_raw.values() if fb["count"] >= 2]
    if len(means_2plus) < 5:
        all_utils = [u for fb in feedback_raw.values() for u in fb["utilities"]]
        mu = sum(all_utils) / len(all_utils) if all_utils else 0.25
        return mu, 1.0
    mu = sum(means_2plus) / len(means_2plus)
    var = sum((m - mu) ** 2 for m in means_2plus) / len(means_2plus)
    if var <= 0 or var >= mu * (1 - mu):
        return mu, 1.0
    return mu, mu * (1 - mu) / var - 1


# Thread-local DB connections for live BM25 queries
_tls = threading.local()


def _fts_query_live(qtext: str, summary_wt: float, themes_wt: float) -> list[str]:
    """Run FTS query with custom BM25 weights using thread-local DB connection."""
    if not hasattr(_tls, "db"):
        _tls.db = sqlite3.connect(str(MEMORY_DB), check_same_thread=False)
        _tls.db.row_factory = sqlite3.Row
    sanitized = sanitize_fts_query(qtext)
    if not sanitized:
        return []
    try:
        rows = _tls.db.execute(f"""
            SELECT rm.memory_id, bm25(memory_fts, {summary_wt}, {themes_wt}) as score
            FROM memory_fts
            JOIN memory_rowid_map rm ON rm.rowid = memory_fts.rowid
            WHERE memory_fts MATCH ?
            ORDER BY score
            LIMIT 200
        """, (sanitized,)).fetchall()
        return [r["memory_id"] for r in rows]
    except Exception:
        return []


def _score_and_rank_reranker(
    qtext: str,
    params: dict,
    search_data: dict,
    feedback_raw: dict | None,
    hebb_data: dict | None,
    reranker: dict,
) -> list[str]:
    """Score candidates using the learned reranker model.

    reranker dict must contain:
      "model": trained LGBMRegressor (or any .predict() compatible model)
      "memory_meta": dict[str, dict] from load_memory_metadata()
    """
    import numpy as np

    model = reranker["model"]
    memory_meta = reranker["memory_meta"]

    # Import the feature extraction function (avoid circular import)
    from train_reranker import extract_features_for_query

    # Use empty GT — we don't need labels, just features
    feats, _, mids = extract_features_for_query(
        qtext, params, search_data, feedback_raw or {},
        hebb_data or {"mem_freq": {}, "total_queries": 0},
        memory_meta, {},
    )

    if len(mids) == 0:
        return []

    preds = np.clip(model.predict(feats), 0, 1)
    order = np.argsort(-preds)
    return [mids[i] for i in order]


def score_and_rank(
    qtext: str,
    params: dict,
    search_data: dict,
    feedback_raw: dict | None = None,
    hebb_data: dict | None = None,
) -> list[str]:
    """Full scoring pipeline: 2-channel RRF + PPR + reranking.

    If search_data contains a "reranker_model" key, uses the learned model
    instead of the hand-tuned formula. The model scores each candidate and
    the result is sorted by predicted score.
    """
    # --- Learned reranker path ---
    reranker = search_data.get("reranker_model")
    if reranker is not None:
        return _score_and_rank_reranker(qtext, params, search_data,
                                        feedback_raw, hebb_data, reranker)

    k_fts = params["k_fts"]
    k_vec = params["k_vec"]
    w_vec = params["w_vec"]
    ucb_coeff = params.get("ucb_coeff", 0.0)
    w_theme = params.get("w_theme", 0.0)
    k_theme = params.get("k_theme", 5.0)
    hebbian_coeff = params.get("hebbian_coeff", 0.0)
    hebbian_cap = params.get("hebbian_cap", 0.0)
    ppr_boost_coeff = params.get("ppr_boost", 0.0)
    ppr_min = params.get("ppr_min_score", 0.0)
    ppr_damping = params.get("ppr_damping", 0.5)

    # FTS results — live query if BM25 weights differ from cached, else use cache
    bm25_sw = params.get("bm25_summary_wt", BM25_SUMMARY_WT)
    bm25_tw = params.get("bm25_themes_wt", BM25_THEMES_WT)
    if bm25_sw != BM25_SUMMARY_WT or bm25_tw != BM25_THEMES_WT:
        fts_ids = _fts_query_live(qtext, bm25_sw, bm25_tw)
    else:
        fts_ids = search_data["fts_results"].get(qtext, [])
    vec_ids = search_data["vec_results"].get(qtext, [])
    themes_map = search_data["themes_map"]

    fts_ranked = {mid: rank for rank, mid in enumerate(fts_ids)}
    vec_ranked = {mid: rank for rank, mid in enumerate(vec_ids)}

    # Theme channel: rank all memories by query-token overlap
    theme_ranked = {}  # memory_id -> rank (0-based)
    if w_theme > 0:
        query_tokens = set(qtext.lower().split())
        scored = []
        for mid, mem_themes in themes_map.items():
            if not mem_themes:
                continue
            overlap = len(mem_themes & query_tokens)
            if overlap > 0:
                scored.append((mid, overlap))
        scored.sort(key=lambda x: (-x[1], x[0]))
        theme_ranked = {mid: rank for rank, (mid, _) in enumerate(scored)}

    # Three-channel RRF
    text_ids = set(fts_ranked.keys()) | set(vec_ranked.keys()) | set(theme_ranked.keys())
    scores = {}
    for mid in text_ids:
        score = 0.0
        if mid in fts_ranked:
            score += (1.0 - w_vec) / (k_fts + fts_ranked[mid] + 1)
        if mid in vec_ranked:
            score += w_vec / (k_vec + vec_ranked[mid] + 1)
        if mid in theme_ranked:
            score += w_theme / (k_theme + theme_ranked[mid] + 1)
        scores[mid] = score

    if not scores:
        return []

    # PPR expansion
    if ppr_boost_coeff > 0:
        ppr_scores = {}
        ppr_cache = search_data.get("ppr_cache")
        if ppr_cache is not None:
            d_key = round(ppr_damping, 3)
            ppr_scores = ppr_cache.get((d_key, qtext), {})

        for mid, ps in ppr_scores.items():
            if ps < ppr_min:
                continue
            if mid in scores:
                scores[mid] += ppr_boost_coeff * ps
            else:
                scores[mid] = ppr_boost_coeff * ps

    # UCB exploration bonus (replaces feedback boost)
    ewma_alpha = params.get("ewma_alpha", 0.3)
    if ucb_coeff > 0 and feedback_raw is not None:
        beta_mean, beta_strength = _compute_beta_prior(feedback_raw, ewma_alpha)
        a = beta_mean * beta_strength
        b = (1 - beta_mean) * beta_strength
        for mid in scores:
            fb = feedback_raw.get(mid)
            if fb and fb["count"] > 0:
                ewma_val = _ewma(fb["utilities"], ewma_alpha)
                effective_n = min(fb["count"], 1.0 / (2 * ewma_alpha - ewma_alpha ** 2))
                a_post = ewma_val * effective_n + a
                b_post = (1 - ewma_val) * effective_n + b
            else:
                # No feedback — maximum uncertainty (prior only)
                a_post = a
                b_post = b
            ab = a_post + b_post
            posterior_var = (a_post * b_post) / (ab * ab * (ab + 1))
            scores[mid] *= (1 + ucb_coeff * math.sqrt(posterior_var))

    # Theme boost — removed (replaced by third RRF channel above)

    # Hebbian PMI
    if hebbian_coeff > 0 and hebb_data and hebb_data["total_queries"] >= 5:
        hebb_mem_freq = hebb_data["mem_freq"]
        hebb_total = hebb_data["total_queries"]
        hebb_mem_count = {mid: len(qs) for mid, qs in hebb_mem_freq.items()}
        seed_ids = sorted(scores, key=scores.get, reverse=True)[:5]

        for candidate in list(scores.keys()):
            if candidate in seed_ids:
                continue
            total_hebb = 0.0
            for seed in seed_ids:
                if seed not in hebb_mem_count or candidate not in hebb_mem_count:
                    continue
                joint = len(hebb_mem_freq.get(seed, set()) & hebb_mem_freq.get(candidate, set()))
                if joint < HEBBIAN_MIN_JOINT:
                    continue
                p_s = hebb_mem_count[seed] / hebb_total
                p_c = hebb_mem_count[candidate] / hebb_total
                p_j = joint / hebb_total
                if p_s * p_c == 0:
                    continue
                pmi = math.log2(p_j / (p_s * p_c))
                if pmi > 0:
                    total_hebb += min(hebbian_coeff * pmi, hebbian_cap)
            scores[candidate] += min(total_hebb, hebbian_cap)

    return sorted(scores, key=scores.get, reverse=True)


# ---------------------------------------------------------------------------
# NDCG computation
# ---------------------------------------------------------------------------


def compute_ndcg(ranked_results: dict[str, list[str]], token_map: dict[str, int],
                 ground_truth: dict[str, dict[str, float]], budget: int = 5000) -> float:
    """NDCG at token budget across all GT queries."""
    ndcg_sum = 0.0
    n_queries = 0

    for qtext, ranked in ranked_results.items():
        gt = ground_truth.get(qtext)
        if not gt:
            continue
        n_queries += 1

        # Determine which memories fit in budget
        shown = []
        used_tokens = 0
        for mid in ranked:
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
            dcg += rel / math.log2(i + 2)

        # Ideal DCG: greedily pack highest-relevance memories into same token budget
        # Sort by relevance (desc), then pack by token cost
        ideal_candidates = sorted(gt.items(), key=lambda x: x[1], reverse=True)
        ideal_rels = []
        ideal_tokens_used = 0
        for mid_ideal, rel_ideal in ideal_candidates:
            t_cost = token_map.get(mid_ideal, 100)
            if ideal_tokens_used + t_cost > budget:
                continue  # skip this one, try smaller ones
            ideal_tokens_used += t_cost
            ideal_rels.append(rel_ideal)
        idcg = 0.0
        for i, rel in enumerate(ideal_rels):
            idcg += rel / math.log2(i + 2)

        if idcg > 0:
            ndcg_sum += dcg / idcg

    return ndcg_sum / n_queries if n_queries > 0 else 0.0


def compute_graded_recall(ranked_results: dict[str, list[str]], token_map: dict[str, int],
                          ground_truth: dict[str, dict[str, float]], budget: int = 5000,
                          threshold: float = 0.5) -> float:
    """Relevance-weighted recall: did the ranker find the *best* relevant memories?

    For each query, count N relevant memories in the budget, then score:
      sum(GT scores of those N) / sum(top-N GT scores)

    A ranker that surfaces the N most relevant scores 1.0 regardless of how
    many relevant memories exist beyond the budget.  A ranker that surfaces
    N low-relevance memories when high-relevance ones exist scores < 1.0.
    """
    total_score = 0.0
    total_ideal = 0.0

    for qtext, ranked in ranked_results.items():
        gt = ground_truth.get(qtext)
        if not gt:
            continue

        # What the ranker actually retrieved within budget
        shown = set()
        used_tokens = 0
        for mid in ranked:
            tokens = token_map.get(mid, 100)
            if used_tokens + tokens > budget:
                break
            used_tokens += tokens
            shown.add(mid)

        # Relevant memories that were shown
        found_relevant = {mid for mid in shown if gt.get(mid, 0) >= threshold}
        n = len(found_relevant)
        if n == 0:
            continue

        # Actual: sum of GT scores for the N relevant memories retrieved
        actual = sum(gt[mid] for mid in found_relevant)

        # Ideal: sum of top-N GT scores (best possible N relevant memories)
        all_relevant_scores = sorted(
            (score for mid, score in gt.items() if score >= threshold),
            reverse=True,
        )
        ideal = sum(all_relevant_scores[:n])

        total_score += actual
        total_ideal += ideal

    return total_score / total_ideal if total_ideal > 0 else 0.0


def compute_recall_at_k(ranked_results: dict[str, list[str]],
                        ground_truth: dict[str, dict[str, float]],
                        k: int = 10, threshold: float = 0.5) -> float:
    """Classic recall@k: fraction of relevant memories in the top k positions."""
    total_relevant = 0
    total_found = 0

    for qtext, ranked in ranked_results.items():
        gt = ground_truth.get(qtext)
        if not gt:
            continue
        shown = set(ranked[:k])
        relevant_ids = {mid for mid, score in gt.items() if score >= threshold}
        total_relevant += len(relevant_ids)
        total_found += len(relevant_ids & shown)

    return total_found / total_relevant if total_relevant > 0 else 0.0


# ---------------------------------------------------------------------------
# Trial evaluation
# ---------------------------------------------------------------------------


def evaluate_trial(params: dict, data: dict, ground_truth: dict,
                   metric: str = "ndcg_5k") -> dict:
    """Evaluate one param set across all GT queries. Returns metric dict."""

    ranked_results = {}
    for qtext in data["gt_queries"]:
        ranked = score_and_rank(qtext, params, data["search_data"],
                                data["feedback_raw"], data["hebb_data"])
        ranked_results[qtext] = ranked

    ndcg = compute_ndcg(ranked_results, data["token_map"], ground_truth, budget=5000)
    recall = compute_graded_recall(ranked_results, data["token_map"], ground_truth, budget=5000)
    recall_10 = compute_recall_at_k(ranked_results, ground_truth, k=10)

    if metric == "ndcg_5k":
        value = ndcg
    elif metric == "graded_recall_5k":
        value = recall
    elif metric == "dual":
        value = (ndcg, recall)  # tuple for NSGA-II
    elif metric == "dual_recall":
        value = (recall, ndcg)  # recall-primary dual
    elif isinstance(metric, tuple) and metric[0] == "blended":
        # Blended: ratio * (NDCG / baseline_N) + (Recall / baseline_R)
        # Normalizes by baseline so ratio=1.0 means equal weight on relative improvement
        ratio, bl_n, bl_r = metric[1], metric[2], metric[3]
        value = ratio * (ndcg / bl_n) + (recall / bl_r)
    else:
        value = ndcg

    return {"value": value, "ndcg_5k": ndcg, "graded_recall_5k": recall,
            "recall_at_10": recall_10}


# ---------------------------------------------------------------------------
# Optuna workers
# ---------------------------------------------------------------------------


def _run_trial(params_fixed: dict, searchable: list[str], data: dict,
               ground_truth: dict, metric: str, trial,
               active_ranges: dict | None = None):
    """Single Optuna trial. Returns float or tuple for dual."""
    ranges = active_ranges or SEARCH_RANGES
    params = dict(params_fixed)
    for pname in searchable:
        spec = ranges[pname]
        if spec[2] == "int":
            log = len(spec) > 3 and spec[3] == "log"
            params[pname] = trial.suggest_int(pname, spec[0], spec[1], log=log)
        else:
            params[pname] = trial.suggest_float(pname, spec[0], spec[1])

    result = evaluate_trial(params, data, ground_truth, metric)
    trial.set_user_attr("ndcg_5k", result["ndcg_5k"])
    trial.set_user_attr("graded_recall_5k", result["graded_recall_5k"])
    trial.set_user_attr("recall_at_10", result["recall_at_10"])
    return result["value"]


def _mp_worker(worker_id, study_name, rdb_url, n_trials, searchable, fixed,
               data_cache_path, gt_path, metric, seed,
               active_ranges=None):
    """Worker process: loads data from disk cache, runs trials."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    with open(data_cache_path, "rb") as f:
        data = pickle.load(f)

    with open(gt_path) as f:
        ground_truth = json.load(f)

    storage = optuna.storages.RDBStorage(
        url=rdb_url,
        engine_kwargs={"connect_args": {"timeout": 120}},
    )
    if metric in ("dual", "dual_recall"):
        sampler = optuna.samplers.NSGAIISampler(seed=seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.load_study(
        study_name=study_name, storage=storage,
        sampler=sampler,
    )

    def objective(trial):
        return _run_trial(dict(fixed), list(searchable), data, ground_truth,
                          metric, trial, active_ranges=active_ranges)

    # Retry on storage errors (SQLite locking with multiple workers)
    trials_remaining = n_trials
    max_retries = 5
    for attempt in range(max_retries):
        # Snapshot trial count before this attempt (shared study, so count all)
        is_dual = metric in ("dual", "dual_recall")
        n_before = len([t for t in study.trials
                        if (t.values is not None if is_dual else t.value is not None)])
        try:
            study.optimize(objective, n_trials=trials_remaining, catch=(Exception,))
            break  # completed successfully
        except Exception:
            # Count how many new trials this attempt added
            study = optuna.load_study(
                study_name=study_name, storage=storage,
                sampler=sampler,
            )
            n_after = len([t for t in study.trials
                           if (t.values is not None if is_dual else t.value is not None)])
            completed_this_attempt = max(0, n_after - n_before)
            trials_remaining -= completed_this_attempt
            if trials_remaining <= 0 or attempt == max_retries - 1:
                break
            time.sleep(1 + attempt)  # brief backoff


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _fmt_time(secs):
    if secs < 60:
        return f"{secs:.0f}s"
    m, s = divmod(int(secs), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _print_trial(trial, searchable, n_completed, n_trials, best_val, best_at, t_start, dual=False, metric=""):
    now = time.monotonic()
    # For dual, use first objective as the tracking value
    val = trial.values[0] if dual and trial.values else trial.value
    if val is None:
        return
    is_best = val > best_val[0]
    if is_best:
        best_val[0] = val
        best_at[0] = trial.number
    n = n_completed[0]
    stale = trial.number - best_at[0]
    marker = " *" if is_best else ""
    dots = "." * min(stale // 10, 20) if stale >= 10 else ""
    elapsed = now - t_start
    s_per_it = elapsed / n if n > 0 else 0
    remaining = (n_trials - n) * s_per_it
    pct = 100 * n / n_trials
    n_width = len(str(n_trials))
    parts = []
    for k in searchable:
        v = trial.params.get(k)
        if v is None:
            continue
        s = _SHORT_NAMES.get(k, k)
        parts.append(f"{s}={v}" if isinstance(v, int) else f"{s}={v:.3f}")
    diff = val - best_val[0]
    _is_blended = isinstance(metric, tuple) and metric[0] == "blended"
    if _is_blended:
        bl_score = metric[1] + 1.0
        diff_bps = diff * 10000
        diff_str = f" {diff_bps:+.0f}bp" if not is_best else ""
        best_bps = (best_val[0] - bl_score) * 10000
        best_str = f"[{best_bps:+.0f}bp @{best_at[0]+1}]"
    else:
        diff_str = f" {diff:+.4f}" if not is_best else ""
        best_str = f"[{best_val[0]:.4f} @{best_at[0]+1}]"
    if dual and trial.values:
        if metric == "dual_recall":
            val_str = f"R={trial.values[0]:.4f} N={trial.values[1]:.4f}"
        else:
            val_str = f"N={trial.values[0]:.4f} R={trial.values[1]:.4f}"
    elif _is_blended:
        n_val = trial.user_attrs.get("ndcg_5k", 0)
        r_val = trial.user_attrs.get("graded_recall_5k", 0)
        bps = (val - bl_score) * 10000
        val_str = f"N={n_val*100:.2f}% R={r_val*100:.2f}% {bps:+.0f}bp"
    else:
        val_str = f"{val:.4f}"
    print(f"  {n:>{n_width}}/{n_trials} {pct:>4.0f}%"
          f"  {val_str}{diff_str}  {best_str}"
          f"  {' '.join(parts)}"
          f"  {_fmt_time(elapsed)} {s_per_it:.1f}s/it ETA {_fmt_time(remaining)}"
          f"{marker}{dots}", flush=True)


# ---------------------------------------------------------------------------
# Main tuning function
# ---------------------------------------------------------------------------


def load_tuning_data(gt_path: str) -> tuple[dict, dict, str]:
    """Load and pre-compute all data needed for tuning.

    Returns (full_data, ground_truth, data_cache_path).
    """
    with open(gt_path) as f:
        ground_truth = json.load(f)
    print(f"\nGround truth: {len(ground_truth)} queries from {Path(gt_path).name}")

    print("\nLoading data...")
    db = get_db()
    data = load_data(db, ground_truth)

    embed_cache_path = DATA_DIR / "tune_gt_embeddings.pkl"
    print("\nPre-computing searches...")
    search_results = precompute_searches(db, data["gt_queries"],
                                          data["vector_input_map"],
                                          embed_cache_path)
    search_data = {
        **search_results,
        "themes_map": data["themes_map"],
    }

    ppr_cache_path = DATA_DIR / "tuning_studies" / "tune_gt_ppr_cache.pkl"
    ppr_cache_path.parent.mkdir(parents=True, exist_ok=True)
    ppr_cache = precompute_ppr_cache(data["gt_queries"], search_data,
                                      data["ppr_adj"], ppr_cache_path)
    search_data["ppr_cache"] = ppr_cache

    full_data = {
        "gt_queries": data["gt_queries"],
        "token_map": data["token_map"],
        "feedback_raw": data["feedback_raw"],
        "hebb_data": data["hebb_data"],
        "search_data": search_data,
    }

    data_cache_path = DATA_DIR / "tuning_studies" / "tune_gt_data.pkl"
    data_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_cache_path, "wb") as f:
        pickle.dump(full_data, f)

    return full_data, ground_truth, str(data_cache_path)


def run_tuning(gt_path: str, n_trials: int = 500, n_jobs: int = 1,
               tag: str = "", metric: str = "ndcg_5k",
               fix: list[str] | None = None,
               set_params: dict[str, float] | None = None,
               active_ranges: dict | None = None,
               preloaded: tuple | None = None):
    """Main entry: load data, pre-compute, tune, plot.

    Args:
        active_ranges: Per-phase search ranges (defaults to SEARCH_RANGES).
        preloaded: (full_data, ground_truth, data_cache_path) to skip loading.

    Returns the completed Optuna study.
    """
    import optuna
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    ranges = active_ranges or SEARCH_RANGES

    if preloaded:
        full_data, ground_truth, data_cache_path = preloaded
    else:
        full_data, ground_truth, data_cache_path = load_tuning_data(gt_path)

    # Evaluate production baseline
    print("\nProduction baseline...")
    baseline = evaluate_trial(dict(PRODUCTION_PARAMS), full_data, ground_truth, "ndcg_5k")
    print(f"  NDCG@5k:          {baseline['ndcg_5k']:.4f}")
    print(f"  Graded Recall@5k: {baseline['graded_recall_5k']:.4f}")

    # Resolve blended metric
    if isinstance(metric, tuple) and metric[0] == "blended_pending":
        ratio = metric[1]
        metric = ("blended", ratio, baseline["ndcg_5k"], baseline["graded_recall_5k"])
        baseline = evaluate_trial(dict(PRODUCTION_PARAMS), full_data, ground_truth, metric)
        print(f"  Blended (ratio={ratio}): {baseline['value']:.4f}")

    # Setup Optuna
    searchable = [k for k in ranges if k not in (fix or [])]
    fixed = {k: v for k, v in PRODUCTION_PARAMS.items() if k not in searchable}
    # Override fixed values with --set params
    if set_params:
        for k, v in set_params.items():
            if k in fixed:
                fixed[k] = v
            elif k in SEARCH_RANGES:
                # Force a searchable param to be fixed at a specific value
                if k in searchable:
                    searchable.remove(k)
                fixed[k] = v

    _metric_tag = f"blended_r{metric[1]:.2f}" if isinstance(metric, tuple) and metric[0] == "blended" else str(metric)
    study_name = f"gt_{_metric_tag}_{len(searchable)}D"
    data_cache_path = str(data_cache_path)  # ensure string for pickle
    if tag:
        study_name = f"{tag}_{study_name}"

    storage_dir = DATA_DIR / "tuning_studies"
    storage_dir.mkdir(parents=True, exist_ok=True)
    rdb_path = storage_dir / "tune_gt.db"
    rdb_url = f"sqlite:///{rdb_path}"

    storage = optuna.storages.RDBStorage(
        url=rdb_url,
        engine_kwargs={"connect_args": {"timeout": 120}},
    )
    is_dual = metric in ("dual", "dual_recall")
    if is_dual:
        from optuna.samplers import NSGAIISampler
        study = optuna.create_study(
            study_name=study_name,
            directions=["maximize", "maximize"],
            sampler=NSGAIISampler(),
            storage=storage,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=TPESampler(n_startup_trials=max(20, len(searchable) * 5)),
            storage=storage,
            load_if_exists=True,
        )

    print(f"\n{'=' * 70}")
    print(f"Tuning {study_name} -- {len(searchable)}D, {n_trials} trials, {n_jobs} workers")
    print(f"  Search: {searchable}")
    print(f"  Fixed:  {', '.join(f'{k}={v}' for k, v in fixed.items())}")
    _dual_desc = "maximize both (recall primary)" if metric == "dual_recall" else "maximize both"
    print(f"  Metric: {metric} ({_dual_desc if is_dual else 'maximize'})")
    if is_dual:
        print(f"  Baseline: NDCG={baseline['ndcg_5k']:.4f}, Recall={baseline['graded_recall_5k']:.4f}")
    else:
        print(f"  Baseline: {baseline[metric]:.4f}")
    print(f"{'=' * 70}\n")

    if n_jobs > 1:
        per_worker = n_trials // n_jobs
        remainder = n_trials % n_jobs
        worker_trials = [per_worker + (1 if i < remainder else 0) for i in range(n_jobs)]
        print(f"  Workers: {' + '.join(str(t) for t in worker_trials)} trials\n")

        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = [
                pool.submit(_mp_worker,
                            worker_id=i,
                            study_name=study_name,
                            rdb_url=rdb_url,
                            n_trials=worker_trials[i],
                            searchable=searchable,
                            fixed=fixed,
                            data_cache_path=data_cache_path,
                            gt_path=gt_path,
                            metric=metric,
                            seed=42 + i,
                            active_ranges=ranges if ranges is not SEARCH_RANGES else None)
                for i in range(n_jobs)
            ]

            t_start = time.monotonic()
            seen = set(t.number for t in study.trials
                       if (t.values is not None if is_dual else t.value is not None))
            best_val = [0.0]
            best_at = [0]
            n_completed = [0]

            while not all(f.done() for f in futures):
                time.sleep(2)
                try:
                    check = optuna.load_study(study_name=study_name, storage=storage)
                    new_trials = [t for t in check.trials
                                  if (t.values is not None if is_dual else t.value is not None)
                                  and t.number not in seen]
                    new_trials.sort(key=lambda t: t.number)
                    for trial in new_trials:
                        seen.add(trial.number)
                        n_completed[0] += 1
                        _print_trial(trial, searchable, n_completed, n_trials,
                                     best_val, best_at, t_start, dual=is_dual, metric=metric)
                except Exception:
                    pass

            # Catch remaining
            check = optuna.load_study(study_name=study_name, storage=storage)
            for trial in check.trials:
                has_val = trial.values is not None if is_dual else trial.value is not None
                if has_val and trial.number not in seen:
                    seen.add(trial.number)
                    n_completed[0] += 1
                    _print_trial(trial, searchable, n_completed, n_trials,
                                 best_val, best_at, t_start, dual=is_dual, metric=metric)

            for f in futures:
                exc = f.exception()
                if exc:
                    print(f"  Worker error: {type(exc).__name__}: {exc}")

        study = optuna.load_study(study_name=study_name, storage=storage)

    else:
        # Single process
        best_val = [0.0]
        best_at = [0]
        n_completed = [0]
        t_start = time.monotonic()

        def objective(trial):
            return _run_trial(dict(fixed), list(searchable), full_data,
                              ground_truth, metric, trial,
                              active_ranges=ranges if ranges is not SEARCH_RANGES else None)

        def _log_trial(study, trial):
            has_val = trial.values is not None if is_dual else trial.value is not None
            if not has_val:
                return
            n_completed[0] += 1
            _print_trial(trial, searchable, n_completed, n_trials,
                         best_val, best_at, t_start, dual=is_dual, metric=metric)

        study.optimize(objective, n_trials=n_trials,
                       show_progress_bar=True, catch=(Exception,),
                       callbacks=[_log_trial])

    # Results
    print(f"\n{'=' * 70}")
    if is_dual:
        pareto = study.best_trials
        print(f"Pareto front: {len(pareto)} solutions")
        print(f"  Baseline: NDCG={baseline['ndcg_5k']:.4f}, Recall={baseline['graded_recall_5k']:.4f}")
        _obj1, _obj2 = ("Recall", "NDCG") if metric == "dual_recall" else ("NDCG", "Recall")
        print(f"\n  {'#':>3}  {_obj1:>7}  {_obj2:>7}  Params")
        print(f"  {'-'*3}  {'-'*7}  {'-'*7}  {'-'*50}")
        for pi, trial in enumerate(sorted(pareto, key=lambda t: t.values[0], reverse=True)):
            ndcg_v, recall_v = trial.values
            parts = []
            for k in searchable:
                v = trial.params.get(k)
                if v is None:
                    continue
                s = _SHORT_NAMES.get(k, k)
                parts.append(f"{s}={v}" if isinstance(v, int) else f"{s}={v:.3f}")
            print(f"  {pi+1:>3}  {ndcg_v:>7.4f}  {recall_v:>7.4f}  {' '.join(parts)}")
    else:
        print(f"Best {metric}: {study.best_value:.4f}  (baseline: {baseline[metric]:.4f},"
              f" delta: {study.best_value - baseline[metric]:+.4f})")
        print(f"Best params:")
        for k, v in study.best_params.items():
            prod = PRODUCTION_PARAMS.get(k)
            delta = ""
            if prod is not None:
                if isinstance(v, int):
                    delta = f"  (was {prod})"
                else:
                    delta = f"  (was {prod:.4f})"
            s = _SHORT_NAMES.get(k, k)
            if isinstance(v, int):
                print(f"  {s:>6} = {v}{delta}")
            else:
                print(f"  {s:>6} = {v:.4f}{delta}")

        # Side metrics
        best_t = study.best_trial
        for attr in ["ndcg_5k", "graded_recall_5k"]:
            v = best_t.user_attrs.get(attr)
            if v is not None:
                bl = baseline.get(attr, 0)
                print(f"  {attr}: {v:.4f} (baseline: {bl:.4f}, delta: {v - bl:+.4f})")

    # Plots
    plot_dir = storage_dir / "plots" / study_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nPlots -> {plot_dir}")

    try:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from plot_optuna_charts import plot_all
        if is_dual:
            if metric == "dual_recall":
                _OBJ_LABELS = {0: "recall", 1: "ndcg"}
            else:
                _OBJ_LABELS = {0: "ndcg", 1: "recall"}
            for oi, label in _OBJ_LABELS.items():
                print(f"\n  --- {label.upper()} plots ---")
                plot_all(study, plot_dir, rdb_path=rdb_path,
                         study_name=study_name,
                         objective_idx=oi, suffix=f"_{label}",
                         recall_primary=(metric == "dual_recall"))
        else:
            plot_all(study, plot_dir, rdb_path=rdb_path, study_name=study_name)
    except Exception as e:
        print(f"  Plots failed: {e}")

    if is_dual:
        # Pareto front scatter plot
        try:
            import matplotlib.pyplot as plt
            completed = [t for t in study.trials if t.values is not None]
            if metric == "dual_recall":
                ndcgs = [t.values[1] for t in completed]
                recalls = [t.values[0] for t in completed]
                pareto_ndcgs = [t.values[1] for t in study.best_trials]
                pareto_recalls = [t.values[0] for t in study.best_trials]
            else:
                ndcgs = [t.values[0] for t in completed]
                recalls = [t.values[1] for t in completed]
                pareto_ndcgs = [t.values[0] for t in study.best_trials]
                pareto_recalls = [t.values[1] for t in study.best_trials]

            fig, ax = plt.subplots(figsize=(8, 6), facecolor="#3a3a3a")
            ax.set_facecolor("#2a2a2a")
            ax.scatter(ndcgs, recalls, c="#4a8a8a", s=10, alpha=0.5, label="Trials")
            ax.scatter(pareto_ndcgs, pareto_recalls, c="#ff4500", s=50,
                       edgecolors="white", linewidth=1, zorder=3, label="Pareto")
            ax.scatter([baseline["ndcg_5k"]], [baseline["graded_recall_5k"]],
                       c="#e8d44d", s=80, marker="*", zorder=4, label="Production")
            ax.set_xlabel("NDCG@5k", color="white", fontsize=10)
            ax.set_ylabel("Graded Recall@5k", color="white", fontsize=10)
            ax.set_title("Dual Objective Pareto Front", color="white", fontsize=12)
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("white")
            ax.legend(facecolor="#2a2a2a", edgecolor="#555555", labelcolor="white")
            fig.savefig(plot_dir / "pareto.png", dpi=150, facecolor="#3a3a3a", bbox_inches="tight")
            plt.close(fig)
            print(f"  pareto -> pareto.png")
        except Exception as e:
            print(f"  Pareto plot failed: {e}")

    print(f"{'=' * 70}")
    if is_dual:
        # Return the Pareto solution with best primary objective
        best_t = max(study.best_trials, key=lambda t: t.values[0])
        return study, {**fixed, **best_t.params}
    return study, {**fixed, **study.best_params}


# ---------------------------------------------------------------------------
# Adaptive phase-based tuning
# ---------------------------------------------------------------------------

HARD_LIMITS = {
    "k_fts": (0.5, 30.0, "float"),
    "k_vec": (0.5, 30.0, "float"),
    "w_vec": (0.0, 1.0, "float"),
    "ucb_coeff": (0.0, 5.0, "float"),
    "w_theme": (0.0, 2.0, "float"),
    "k_theme": (0.5, 30.0, "float"),
    "ppr_boost": (0.0, 8.0, "float"),
    "ppr_damping": (0.01, 0.99, "float"),
    "ppr_min_score": (0.0, 1.0, "float"),
    "hebbian_coeff": (-0.1, 0.5, "float"),
    "hebbian_cap": (0.0, 0.5, "float"),
    "ewma_alpha": (0.01, 0.99, "float"),
    "bm25_summary_wt": (0.5, 20.0, "float"),
    "bm25_themes_wt": (0.1, 20.0, "float"),
}

# Thresholds for adaptive decisions
_FREEZE_CUM_IMPORTANCE = 0.10   # freeze bottom params contributing < this fraction
_FREEZE_IMP_CEILING = 0.05      # never freeze above this regardless of cumulative
_FREEZE_CI_FRAC = 0.15          # freeze if CI < this fraction of range
_BOUNDARY_MARGIN = 0.10         # expand if mean within this fraction of edge
_EXPAND_FACTOR = 0.50           # expand edge by this fraction of range
_UNFREEZE_WIDEN = 0.15          # widen bootstrap CI by this factor on unfreeze
_CONVERGENCE_THRESHOLD = 0.0003 # stop if improvement < this for patience phases
_CONVERGENCE_PATIENCE = 3       # number of stale phases before stopping
_NARROW_FLOOR = 0.50             # max reduction per phase (range keeps ≥50% of width)
_REGRESSION_THRESHOLD = -0.002  # unfreeze recent freezes if regression exceeds this
_MIN_TRIALS_PER_DIM = 15        # absolute floor for trials per dimension
_MIN_RANGE_FRAC = 0.02          # auto-freeze if range < this fraction of hard limits
_WARMUP_BUDGET_MULT = 1.25      # warmup: extra exploration in full space
_PHASE1_BUDGET_MULT = 1.5       # first adaptive phase: most uncertain

# Static check: SEARCH_RANGES must be within HARD_LIMITS
for _p, _spec in SEARCH_RANGES.items():
    if _p not in HARD_LIMITS:
        raise RuntimeError(f"SEARCH_RANGES param '{_p}' missing from HARD_LIMITS")
    _hard = HARD_LIMITS[_p]
    if _spec[0] < _hard[0] or _spec[1] > _hard[1]:
        raise RuntimeError(
            f"SEARCH_RANGES['{_p}'] = [{_spec[0]}, {_spec[1]}] exceeds "
            f"HARD_LIMITS [{_hard[0]}, {_hard[1]}]"
        )


def _pooled_ci(p: str, all_studies: list, top_frac: float = 0.2,
               n_boot: int = 1000) -> tuple[float, float] | None:
    """Bootstrap CI for a param from all phases where it was searchable.

    Returns (ci_lo, ci_hi) or None if insufficient data.
    """
    import numpy as np

    # Collect (param_value, objective) from all trials that searched this param
    # Handle both single-objective (t.value) and multi-objective (t.values) studies
    pairs = []
    for study in all_studies:
        for t in study.trials:
            obj = None
            if t.values is not None:
                obj = t.values[0]
            elif t.value is not None:
                obj = t.value
            if obj is not None and p in t.params:
                pairs.append((t.params[p], obj))
    if len(pairs) < 10:
        return None

    pairs.sort(key=lambda x: x[1], reverse=True)
    n_top = max(5, int(len(pairs) * top_frac))
    top = pairs[:n_top]
    arr = np.array([v for v, _ in top])
    obj = np.array([o for _, o in top])

    rng = np.random.default_rng(42)
    boot_optima = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(arr), size=len(arr))
        best_idx = idx[obj[idx].argmax()]
        boot_optima[b] = arr[best_idx]
    return (float(np.percentile(boot_optima, 2.5)),
            float(np.percentile(boot_optima, 97.5)))


def _unfreeze_range(p: str, info: dict, current_range: tuple | None = None,
                    all_studies: list | None = None) -> tuple:
    """Compute the search range for an unfrozen parameter.

    For float params: CI-widened by _UNFREEZE_WIDEN. If all_studies is provided,
    uses pooled bootstrap CI across all phases; otherwise falls back to stored
    freeze-time CI. Merged with last-known range.
    For int params: frozen_value ± 2 (generous, allows correcting early locks).
    Always clamped to HARD_LIMITS.
    """
    orig_spec = SEARCH_RANGES[p]
    hard = HARD_LIMITS[p]
    ptype = orig_spec[2]

    if ptype == "int":
        val = int(round(info["value"]))
        lo = max(hard[0], val - 2)
        hi = min(hard[1], val + 2)
        if hi <= lo:
            hi = lo + 1
    else:
        # Prefer pooled CI from all phases; fall back to freeze-time CI
        pooled = _pooled_ci(p, all_studies) if all_studies else None
        if pooled is not None:
            ci_lo, ci_hi = pooled
        else:
            ci_lo = info.get("ci_lo", info["value"])
            ci_hi = info.get("ci_hi", info["value"])
        ci_width = ci_hi - ci_lo
        widen = ci_width * _UNFREEZE_WIDEN / 2
        lo = max(hard[0], ci_lo - widen)
        hi = min(hard[1], ci_hi + widen)
        # Floor: if CI-based range is pathologically narrow, fall back to
        # at least 25% of last-known range (matches narrowing floor)
        if current_range is not None:
            prior_width = current_range[1] - current_range[0]
            min_width = prior_width * 0.25
            if (hi - lo) < min_width:
                center = (lo + hi) / 2
                lo = max(hard[0], center - min_width / 2)
                hi = min(hard[1], lo + min_width)
                lo = max(hard[0], hi - min_width)

    return (lo, hi) + orig_spec[2:]


def bootstrap_param_bounds(study, searchable: list[str],
                           active_ranges: dict,
                           n_boot: int = 1000,
                           top_frac: float = 0.2) -> dict:
    """Bootstrap confidence intervals from top trials.

    For float params: bootstraps the *optimum location* — resamples top trials
    and records the param value of the best trial in each sample. The CI on
    this is naturally wider than a CI on the mean, reflecting actual uncertainty
    about where the peak is.

    Returns (center, std, ci_lo, ci_hi) for each param.
    For int params: mode of top trials, CI brackets top-3 frequent values.
    """
    import numpy as np
    from collections import Counter

    # For multi-objective, use first objective (NDCG)
    has_multi = hasattr(study, "directions") and len(study.directions) > 1
    if has_multi:
        completed = [t for t in study.trials if t.values is not None]
        completed.sort(key=lambda t: t.values[0], reverse=True)
    else:
        completed = [t for t in study.trials if t.value is not None]
        completed.sort(key=lambda t: t.value, reverse=True)
    if not completed:
        return {}

    n_top = max(5, int(len(completed) * top_frac))
    top_trials = completed[:n_top]

    # Build paired (param_value, objective) per param — must stay aligned
    # when some top trials lack a param (e.g., seeded from prior phase)
    def _get_obj(t):
        return t.values[0] if has_multi else t.value

    param_pairs = {p: [(t.params[p], _get_obj(t))
                       for t in top_trials if p in t.params]
                   for p in searchable}

    bounds = {}
    rng = np.random.default_rng(42)
    for p, pairs in param_pairs.items():
        if not pairs:
            continue
        lo, hi = active_ranges[p][:2]
        ptype = active_ranges[p][2] if len(active_ranges[p]) > 2 else "float"
        vals = [v for v, _ in pairs]

        if ptype == "int":
            # Top-3 frequent values + intermediates for integers
            counts = Counter(int(v) for v in vals)
            top3 = [v for v, _ in counts.most_common(3)]
            mode_val = top3[0]
            if top3:
                range_lo = min(top3)
                range_hi = max(top3)
                ci_lo_val = float(max(lo, range_lo))
                ci_hi_val = float(min(hi, range_hi))
            else:
                ci_lo_val, ci_hi_val = float(mode_val), float(mode_val)
            arr = np.array(vals, dtype=float)
            std = float(arr.std())
            bounds[p] = (float(mode_val), std, ci_lo_val, ci_hi_val)
        else:
            # Bootstrap the optimum location: resample, take param value of best
            arr = np.array(vals, dtype=float)
            obj_arr = np.array([o for _, o in pairs])
            boot_optima = np.empty(n_boot)
            for b in range(n_boot):
                idx = rng.integers(0, len(arr), size=len(arr))
                best_idx = idx[obj_arr[idx].argmax()]
                boot_optima[b] = arr[best_idx]
            center = float(np.median(boot_optima))
            ci_lo_val = float(np.percentile(boot_optima, 2.5))
            ci_hi_val = float(np.percentile(boot_optima, 97.5))
            std = float(boot_optima.std())
            bounds[p] = (center, std, max(lo, ci_lo_val), min(hi, ci_hi_val))
    return bounds


def check_multimodality(values: list[float], ptype: str) -> bool:
    """Test for multimodality using GMM BIC comparison + mode separation.

    Returns True only if BIC strongly favors 2 components AND the two
    modes are well-separated (distance > 2σ of the narrower component)
    AND both components have meaningful weight (>15%).

    TPE concentrates trials in promising regions, creating apparent
    clusters that aren't true modes — the weight and separation checks
    filter these out.
    """
    if ptype == "int" and len(set(values)) < 5:
        return False
    if len(values) < 30:
        return False
    import numpy as np
    from sklearn.mixture import GaussianMixture

    X = np.array(values, dtype=float).reshape(-1, 1)
    gm1 = GaussianMixture(n_components=1, random_state=42).fit(X)
    gm2 = GaussianMixture(n_components=2, random_state=42).fit(X)

    # BIC must strongly favor 2 components
    if gm2.bic(X) >= gm1.bic(X) - 10:
        return False

    # Both components must have meaningful weight (not just outlier cluster)
    min_weight = min(gm2.weights_)
    if min_weight < 0.15:
        return False

    # Modes must be well-separated: distance between means > 2σ of narrower
    means = gm2.means_.flatten()
    stds = np.sqrt(gm2.covariances_.flatten())
    separation = abs(means[0] - means[1])
    min_std = min(stds)
    if min_std <= 0:
        return False
    return separation > 2 * min_std


def detect_boundary_peak(mean: float, lo: float, hi: float,
                         margin: float = _BOUNDARY_MARGIN) -> str | None:
    """Check if bootstrap mean is near a bound edge."""
    rng = hi - lo
    if rng <= 0:
        return None
    if (mean - lo) / rng < margin:
        return "lo"
    if (hi - mean) / rng < margin:
        return "hi"
    return None


def int_param_dominant(study, param: str, metric: str,
                       top_frac: float = 0.2) -> tuple[int | None, bool]:
    """Check if one integer value is significantly better among top trials.

    Uses only the top fraction of trials (by objective), finds the top 3
    most frequent integer values among them, and tests whether the best
    is statistically dominant over the runner-up.

    Returns (best_value, is_dominant).
    """
    import numpy as np
    from collections import Counter

    is_dual = metric in ("dual", "dual_recall")
    completed = [t for t in study.trials
                 if (t.values is not None if is_dual else t.value is not None)
                 and param in t.params]
    if len(completed) < 10:
        return None, False

    # Sort by objective, take top fraction
    if is_dual:
        completed.sort(key=lambda t: t.values[0], reverse=True)
    else:
        completed.sort(key=lambda t: t.value, reverse=True)
    n_top = max(10, int(len(completed) * top_frac))
    top_trials = completed[:n_top]

    # Find top 3 most frequent values in top trials
    counts = Counter(int(t.params[param]) for t in top_trials)
    top3_vals = {v for v, _ in counts.most_common(3)}

    if len(top3_vals) < 2:
        return counts.most_common(1)[0][0], True

    # Group top trials by these values only, compare mean objective
    groups: dict[int, list[float]] = {}
    for t in top_trials:
        v = int(t.params[param])
        if v in top3_vals:
            obj = t.values[0] if is_dual else t.value
            groups.setdefault(v, []).append(obj)

    ranked = sorted(groups.items(), key=lambda kv: np.mean(kv[1]), reverse=True)
    best_val, best_scores = ranked[0]
    _, runner_scores = ranked[1]

    best_mean = np.mean(best_scores)
    runner_mean = np.mean(runner_scores)

    # Pooled standard error of the difference
    se_best = np.std(best_scores, ddof=1) / np.sqrt(len(best_scores)) if len(best_scores) > 1 else 0
    se_runner = np.std(runner_scores, ddof=1) / np.sqrt(len(runner_scores)) if len(runner_scores) > 1 else 0
    se_diff = np.sqrt(se_best**2 + se_runner**2)

    # Dominant if best mean > runner-up mean by > 1 SE
    if se_diff > 0:
        dominant = (best_mean - runner_mean) / se_diff > 2.0
    else:
        dominant = best_mean > runner_mean

    return best_val, dominant


def analyze_phase(study, active_ranges: dict, searchable: list[str],
                  metric: str) -> dict:
    """Run all analysis sub-steps on a completed phase.

    Returns {param: {importance, mean, std, ci_lo, ci_hi, multimodal, boundary}}.
    """
    import optuna

    # Importance
    is_dual = metric in ("dual", "dual_recall")
    try:
        if is_dual:
            importances = optuna.importance.get_param_importances(
                study, target=lambda t: t.values[0])
        else:
            importances = optuna.importance.get_param_importances(study)
    except Exception:
        importances = {}

    # Bootstrap
    bootstrap = bootstrap_param_bounds(study, searchable, active_ranges)

    # Per-param analysis
    has_multi = hasattr(study, "directions") and len(study.directions) > 1
    if has_multi:
        completed = [t for t in study.trials if t.values is not None]
    else:
        completed = [t for t in study.trials if t.value is not None]

    result = {}
    for p in searchable:
        vals = [t.params[p] for t in completed if p in t.params]
        spec = active_ranges[p]
        lo, hi, ptype = spec[:3]
        bs = bootstrap.get(p, (0.0, 0.0, lo, hi))
        mean, std, ci_lo, ci_hi = bs

        multimodal = check_multimodality(vals, ptype) if vals else False
        boundary = detect_boundary_peak(mean, lo, hi)

        # For int params, check if one value is statistically dominant
        int_best, int_dominant = (None, False)
        if ptype == "int":
            int_best, int_dominant = int_param_dominant(study, p, metric)

        result[p] = {
            "importance": importances.get(p, 0.0),
            "mean": mean,  # mode for int params
            "std": std,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "multimodal": multimodal,
            "boundary": boundary,
            "int_best": int_best,
            "int_dominant": int_dominant,
        }

    return result


def compute_next_phase(analysis: dict, active_ranges: dict,
                       frozen_params: dict, phase_num: int,
                       unfreeze_interval: int,
                       manually_fixed: set,
                       phase_improved: bool = True,
                       all_studies: list | None = None) -> tuple:
    """Decide freeze/narrow/expand/unfreeze for each param.

    Returns (new_ranges, new_frozen, new_searchable, actions).
    actions is a list of (param, action_str) for logging.
    """
    new_ranges = {}
    new_frozen = dict(frozen_params)
    new_searchable = []
    actions = []

    # Check for unfreezing
    for p, info in list(new_frozen.items()):
        if p in manually_fixed:
            continue  # never unfreeze manually fixed params
        frozen_at_phase = info["phase"]
        if phase_num - frozen_at_phase >= unfreeze_interval:
            spec = _unfreeze_range(p, info, active_ranges.get(p), all_studies=all_studies)
            lo, hi = spec[:2]
            new_ranges[p] = spec
            new_searchable.append(p)
            del new_frozen[p]
            s = _SHORT_NAMES.get(p, p)
            actions.append((p, "unfreeze", f"{s} [{lo:.3g}, {hi:.3g}]",
                            f"was {info['value']:.4g}, frozen since p{frozen_at_phase}"))

    # Landscape stability check: block freezing if any important param is wide
    # (i.e., the landscape is still shifting — freezing others is premature)
    _UNSTABLE_IMP = 0.10   # importance threshold for "significant" param
    _UNSTABLE_CI = 0.50    # CI > this fraction of range = "wide"
    landscape_unstable = False
    for p, info in analysis.items():
        if p in new_frozen or p in manually_fixed:
            continue
        spec = active_ranges[p]
        rng_w = spec[1] - spec[0]
        ci_w = info["ci_hi"] - info["ci_lo"]
        if (info["importance"] > _UNSTABLE_IMP
                and rng_w > 0 and ci_w / rng_w > _UNSTABLE_CI):
            landscape_unstable = True
            break

    # Max importance across active params (for normalizing narrowing floor)
    max_imp = max((info["importance"] for p, info in analysis.items()
                   if p not in new_frozen and p not in manually_fixed), default=1.0)

    # Cumulative importance: freeze bottom params contributing < 10% of total
    active_imps = {p: info["importance"] for p, info in analysis.items()
                   if p not in new_frozen and p not in manually_fixed}
    total_imp = sum(active_imps.values())
    sorted_by_imp = sorted(active_imps.items(), key=lambda x: x[1])
    cum = 0.0
    freeze_eligible = set()
    for p_imp, imp_val in sorted_by_imp:
        cum += imp_val
        if total_imp > 0 and cum / total_imp < _FREEZE_CUM_IMPORTANCE:
            freeze_eligible.add(p_imp)
        else:
            break

    # Process currently active params
    for p, info in analysis.items():
        if p in new_frozen or p in manually_fixed:
            continue
        if p in [a[0] for a in actions if a[1] == "unfreeze"]:
            continue  # already handled in unfreeze

        spec = active_ranges[p]
        lo, hi, ptype = spec[:3]
        rng_width = hi - lo
        importance = info["importance"]
        ci_width = info["ci_hi"] - info["ci_lo"]
        s = _SHORT_NAMES.get(p, p)

        # Freeze: low importance, unimodal, tight CI (float params only;
        # int params use dominance test below)
        # Blocked when landscape is unstable (important params still wide)
        # Blocked in phase 1 — first adaptive pass should only narrow, not commit
        # Allow multimodal freeze when importance is negligible (< 1%)
        multimodal_ok = not info["multimodal"] or importance < 0.01
        if (phase_num >= 2
                and not landscape_unstable
                and ptype != "int"
                and p in freeze_eligible and importance < _FREEZE_IMP_CEILING
                and multimodal_ok
                and rng_width > 0
                and ci_width / rng_width < _FREEZE_CI_FRAC):
            freeze_val = info["mean"]
            new_frozen[p] = {
                "value": freeze_val,
                "phase": phase_num,
                "ci_lo": info["ci_lo"],
                "ci_hi": info["ci_hi"],
            }
            ci_pct = f"{100 * ci_width / rng_width:.0f}%" if rng_width > 0 else "?"
            actions.append((p, "freeze", f"{s} = {freeze_val:.4g}",
                            f"imp={importance:.3f}, CI={ci_pct} of range"))
            continue

        # Expand: peak near boundary
        hard = HARD_LIMITS[p]
        new_lo, new_hi = lo, hi
        expanded_lo, expanded_hi = False, False
        if info["boundary"] == "lo":
            expand = rng_width * _EXPAND_FACTOR
            candidate_lo = max(hard[0], lo - expand)
            if ptype == "int":
                candidate_lo = int(candidate_lo)
            if candidate_lo < lo:
                new_lo = candidate_lo
                expanded_lo = True
                actions.append((p, "expand", f"{s} lo: {lo:.3g} -> {new_lo:.3g}",
                                f"mean={info['mean']:.3g} near lower bound"))
            # else: already at hard limit, no-op
        elif info["boundary"] == "hi":
            expand = rng_width * _EXPAND_FACTOR
            candidate_hi = min(hard[1], hi + expand)
            if ptype == "int":
                candidate_hi = int(math.ceil(candidate_hi))
            if candidate_hi > hi:
                new_hi = candidate_hi
                expanded_hi = True
                actions.append((p, "expand", f"{s} hi: {hi:.3g} -> {new_hi:.3g}",
                                f"mean={info['mean']:.3g} near upper bound"))
            # else: already at hard limit, no-op

        # Integer params: dominance-based decisions (independent of multimodality)
        if ptype == "int":
            mode_val = int(round(info["mean"]))  # bootstrap returns mode for ints
            best_val = info.get("int_best", mode_val)
            dominant = info.get("int_dominant", False)
            freeze_val = best_val if best_val is not None else mode_val
            # Can't narrow further: range is already min width (2 adjacent ints)
            at_min_range = (hi - lo) <= 1
            if (phase_num >= 2 and not landscape_unstable
                    and dominant and ((p in freeze_eligible and importance < _FREEZE_IMP_CEILING) or at_min_range)):
                # Dominant + (low importance OR can't narrow further) → freeze
                reason = f"imp={importance:.3f}, dominant (>2SE)"
                if at_min_range:
                    reason += ", range=[{},{}] minimal".format(int(lo), int(hi))
                new_frozen[p] = {
                    "value": freeze_val,
                    "phase": phase_num,
                    "ci_lo": info["ci_lo"],
                    "ci_hi": info["ci_hi"],
                }
                actions.append((p, "freeze", f"{s} = {freeze_val}",
                                reason))
                continue
            elif phase_improved:
                # Start from CI bounds, expand around best if CI is too narrow
                int_ci_lo = int(math.floor(info["ci_lo"]))
                int_ci_hi = int(math.ceil(info["ci_hi"]))
                # Floor: keep at least 50% of current int range
                int_rng = hi - lo
                min_span = max(2, int(math.ceil(int_rng * _NARROW_FLOOR)))
                ci_span = int_ci_hi - int_ci_lo
                if ci_span >= min_span:
                    # CI is wide enough — use it directly
                    narrow_lo = max(hard[0], int_ci_lo)
                    narrow_hi = min(hard[1], int_ci_hi)
                else:
                    # CI too narrow — expand around best to meet floor
                    half = min_span / 2
                    narrow_lo = max(hard[0], int(math.floor(freeze_val - half)))
                    narrow_hi = min(hard[1], int(math.ceil(freeze_val + half)))
                    # Re-clamp to maintain span if one side hit hard limit
                    if narrow_hi - narrow_lo < min_span:
                        if narrow_lo == hard[0]:
                            narrow_hi = min(hard[1], narrow_lo + min_span)
                        else:
                            narrow_lo = max(hard[0], narrow_hi - min_span)
                # Preserve expand on the side that was expanded
                new_lo = min(new_lo, narrow_lo) if expanded_lo else narrow_lo
                new_hi = max(new_hi, narrow_hi) if expanded_hi else narrow_hi
                if new_hi <= new_lo:
                    new_hi = min(hard[1], new_lo + 1)
                    if new_hi <= new_lo:
                        new_lo = max(hard[0], new_hi - 1)
                reason = f"imp={importance:.3f}, mode={mode_val}"
                if dominant:
                    reason += ", dominant"
                else:
                    reason += ", not yet dominant"
                actions.append((p, "narrow", f"{s} [{new_lo}, {new_hi}]",
                                reason))
                new_ranges[p] = (new_lo, new_hi) + spec[2:]
                new_searchable.append(p)
                continue
            else:
                # Phase didn't improve — keep int range (but preserve any expand)
                actions.append((p, "keep", f"{s} [{int(new_lo)}, {int(new_hi)}]",
                                f"imp={importance:.3f}, phase didn't improve"))
                new_ranges[p] = (new_lo, new_hi) + spec[2:]
                new_searchable.append(p)
                continue

        # Float params: narrow based on CI (only if phase improved)
        # Track narrowing reason for action log (logged after floor adjustment)
        _narrow_reason = None
        if phase_improved:
            if not info["multimodal"] and ci_width < rng_width * 0.8:
                new_lo = new_lo if expanded_lo else max(new_lo, info["ci_lo"])
                new_hi = new_hi if expanded_hi else min(new_hi, info["ci_hi"])
                if not any(p == a[0] and a[1] == "expand" for a in actions):
                    _narrow_reason = f"imp={importance:.3f}, CI={ci_width:.3g}"
            elif info["multimodal"]:
                # Multimodal: still narrow, but use mean ± 2σ (wider than CI)
                wide_lo = info["mean"] - 2 * info["std"]
                wide_hi = info["mean"] + 2 * info["std"]
                wide_width = wide_hi - wide_lo
                if wide_width < rng_width * 0.8:
                    new_lo = new_lo if expanded_lo else max(new_lo, wide_lo)
                    new_hi = new_hi if expanded_hi else min(new_hi, wide_hi)
                    if not any(p == a[0] and a[1] == "expand" for a in actions):
                        _narrow_reason = f"multimodal, imp={importance:.3f}, ±2σ"
                else:
                    if not any(p == a[0] for a in actions):
                        actions.append((p, "keep", f"{s} [{lo:.3g}, {hi:.3g}]",
                                        f"multimodal, imp={importance:.3f}, 2σ>80%"))
        else:
            # Phase didn't improve — keep float range (but preserve any expand)
            if not any(p == a[0] for a in actions):
                actions.append((p, "keep", f"{s} [{new_lo:.3g}, {new_hi:.3g}]",
                                f"imp={importance:.3f}, phase didn't improve"))

        # Floor: non-int ranges keep ≥50% of width per phase
        _was_floored = False
        if ptype != "int":
            new_width = new_hi - new_lo
            min_width = rng_width * _NARROW_FLOOR
            if new_width < min_width:
                _was_floored = True
                center = (new_lo + new_hi) / 2
                hard = HARD_LIMITS[p]
                new_lo = max(hard[0], center - min_width / 2)
                new_hi = min(hard[1], new_lo + min_width)
                new_lo = max(hard[0], new_hi - min_width)  # re-clamp if hi hit limit
        else:
            new_lo, new_hi = int(math.floor(new_lo)), int(math.ceil(new_hi))
            if new_hi <= new_lo:
                new_hi = min(int(hard[1]), new_lo + 1)

        # Log narrow action AFTER floor adjustment so display matches actual ranges
        if _narrow_reason is not None:
            floored = " (floored)" if _was_floored else ""
            actions.append((p, "narrow", f"{s} [{new_lo:.3g}, {new_hi:.3g}]",
                            _narrow_reason + floored))

        # Auto-freeze if range collapsed below L_min (TuRBO-inspired)
        if ptype != "int":
            hard_w = hard[1] - hard[0] if hard else 0
            final_w = new_hi - new_lo
            if hard_w > 0 and final_w / hard_w < _MIN_RANGE_FRAC:
                freeze_val = (new_lo + new_hi) / 2
                new_frozen[p] = {"value": freeze_val, "phase": phase_num,
                                 "ci_lo": new_lo, "ci_hi": new_hi}
                actions.append((p, "freeze", f"{s} = {freeze_val:.4g}",
                                f"range collapsed to {100*final_w/hard_w:.1f}% of hard limits"))
                continue

        new_ranges[p] = (new_lo, new_hi) + spec[2:]
        new_searchable.append(p)

    # Sanity check: no searchable param should have a point range
    for p in new_searchable:
        lo, hi = new_ranges[p][0], new_ranges[p][1]
        if lo == hi:
            s = _SHORT_NAMES.get(p, p)
            raise RuntimeError(
                f"BUG: {s} is searchable but has point range [{lo}, {hi}]. "
                f"Should be frozen or have a nonzero-width range."
            )

    return new_ranges, new_frozen, new_searchable, actions


def seed_study(new_study, prev_study, new_searchable: list[str],
               new_ranges: dict, n_seeds: int = 10):
    """Enqueue top trials from prev study into new study."""
    has_multi = hasattr(prev_study, "directions") and len(prev_study.directions) > 1
    if has_multi:
        completed = [t for t in prev_study.trials if t.values is not None]
        completed.sort(key=lambda t: t.values[0], reverse=True)
    else:
        completed = [t for t in prev_study.trials if t.value is not None]
        completed.sort(key=lambda t: t.value, reverse=True)

    seeded = 0
    for trial in completed[:n_seeds]:
        params = {}
        valid = True
        for p in new_searchable:
            if p not in trial.params:
                valid = False
                break
            v = trial.params[p]
            spec = new_ranges[p]
            lo, hi, ptype = spec[:3]
            # Clamp to new range
            if ptype == "int":
                v = max(lo, min(hi, int(round(v))))
            else:
                v = max(lo, min(hi, v))
            params[p] = v
        if valid:
            new_study.enqueue_trial(params)
            seeded += 1
    return seeded


def _get_best_metric(study, metric) -> float:
    """Extract best metric value from a study."""
    is_dual = metric in ("dual", "dual_recall")
    if is_dual:
        if not study.best_trials:
            return 0.0
        return max(t.values[0] for t in study.best_trials)
    else:
        return study.best_value if study.best_value is not None else 0.0


def _find_latest_run(tag: str, metric: str) -> str | None:
    """Find the most recent adaptive run prefix matching a tag in the study DB."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    rdb_path = DATA_DIR / "tuning_studies" / "tune_gt.db"
    if not rdb_path.exists():
        return None
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{rdb_path}",
        engine_kwargs={"connect_args": {"timeout": 30}},
    )
    try:
        summaries = storage.get_all_studies()
    except Exception:
        return None

    # Look for study names like "{tag}_{timestamp}_p0_{metric}_..."
    import re
    pattern = re.compile(rf"^({re.escape(tag)}_\d{{4}}_\d{{4}})_p0_")
    matches = []
    for s in summaries:
        m = pattern.match(s.study_name)
        if m:
            matches.append(m.group(1))

    if not matches:
        return None
    # Sort lexicographically — timestamp format ensures latest is last
    matches.sort()
    return matches[-1]


def run_adaptive(gt_path: str, total_trials: int = 1000, n_jobs: int = 1,
                 tag: str = "", metric: str = "ndcg_5k",
                 fix: list[str] | None = None,
                 set_params: dict[str, float] | None = None,
                 max_phases: int = 6, unfreeze_interval: int = 3,
                 resume: bool = False,
                 preloaded: tuple | None = None,
                 skip_plots: bool = False):
    """Adaptive phase-based tuning. Each phase is a separate Optuna study.

    Args:
        preloaded: (full_data, ground_truth, data_cache_path) to skip loading.
        skip_plots: Skip plot phase and chart generation (for CV mode).

    Returns:
        (best_params, best_ndcg) — best complete param set and its NDCG score.
    """

    # Run prefix: timestamp for uniqueness, or find latest matching prefix on resume
    if resume:
        run_prefix = _find_latest_run(tag or "adaptive", metric)
        if not run_prefix:
            print(f"  No previous adaptive run found for tag '{tag}'. Starting fresh.")
            resume = False
            run_prefix = f"{tag}_{time.strftime('%m%d_%H%M')}" if tag else time.strftime("%m%d_%H%M")
        else:
            print(f"  Resuming run: {run_prefix}")
    else:
        run_prefix = f"{tag}_{time.strftime('%m%d_%H%M')}" if tag else time.strftime("%m%d_%H%M")

    # Load data once
    if preloaded:
        full_data, ground_truth, data_cache_path = preloaded
    else:
        full_data, ground_truth, data_cache_path = load_tuning_data(gt_path)

    # Production baseline (use ndcg_5k for baseline computation — metric may be blended)
    print("\nProduction baseline...")
    baseline = evaluate_trial(dict(PRODUCTION_PARAMS), full_data, ground_truth, "ndcg_5k")
    print(f"  NDCG@5k:          {baseline['ndcg_5k']:.4f}")
    print(f"  Graded Recall@5k: {baseline['graded_recall_5k']:.4f}")

    # Resolve blended metric now that we have baselines
    if isinstance(metric, tuple) and metric[0] == "blended_pending":
        ratio = metric[1]
        metric = ("blended", ratio, baseline["ndcg_5k"], baseline["graded_recall_5k"])
        print(f"  Blended (ratio={ratio}): N*{ratio}/{baseline['ndcg_5k']:.4f} + R/{baseline['graded_recall_5k']:.4f}")
        # Re-evaluate baseline with blended metric
        baseline = evaluate_trial(dict(PRODUCTION_PARAMS), full_data, ground_truth, metric)
        print(f"  Baseline blended:  {baseline['value']:.4f}")

    manually_fixed = set(fix or [])
    if set_params:
        manually_fixed |= set(set_params.keys())
    initial_searchable = [k for k in SEARCH_RANGES if k not in manually_fixed]
    n_searchable = len(initial_searchable)

    # Budget allocation — all trials go to adaptive phases.
    # Each phase gets trials_per_dim * n_active trials (dimension-scaled).
    # Warmup counts as 1 phase at full dimensionality.
    available = total_trials
    trials_per_dim = max(_MIN_TRIALS_PER_DIM,
                         available // (max_phases * n_searchable))
    min_phase_trials = max(30, 2 * trials_per_dim)
    warmup_trials = int(max(0.20 * total_trials,
                            trials_per_dim * n_searchable) * _WARMUP_BUDGET_MULT)
    remaining = total_trials - warmup_trials
    max_adaptive = max_phases - 1

    # Track state across phases
    frozen_params = {}  # {param: {value, phase, ci_lo, ci_hi}}
    current_ranges = dict(SEARCH_RANGES)
    phase_history = []  # [{phase, study_name, n_trials, best_metric, n_active, n_frozen, actions}]
    param_trajectory = {p: [] for p in SEARCH_RANGES}  # {param: [action_per_phase]}
    all_completed_trials = []  # [(values, complete_params)] across all phases (dual only)
    global_best = 0.0
    stale_phases = 0
    unfreeze_cycles = 0
    _MAX_UNFREEZE_CYCLES = 3

    print(f"\n{'=' * 70}")
    print(f"ADAPTIVE TUNING: {total_trials} total trials, up to {max_phases} phases")
    print(f"  Trials/dim: {trials_per_dim} (min phase: {min_phase_trials})")
    print(f"  Warm-up: {warmup_trials} trials ({n_searchable}D)")
    print(f"  Remaining: {remaining} trials across up to {max_adaptive} phases")
    print(f"  Unfreeze interval: every {unfreeze_interval} phases")
    if manually_fixed:
        print(f"  Manually fixed: {', '.join(manually_fixed)}")
    print(f"{'=' * 70}")

    prev_study = None
    all_studies = []  # all phase studies, for pooled CI on unfreeze
    trials_used = 0
    recently_frozen = set()  # params frozen in the most recent decision round
    skip_analysis = False    # skip analysis after unfreeze (no data for unfrozen params)
    last_phase_regressed = False  # warmup never counts as regression
    global_best_params = {}  # best complete param set across all phases
    global_best_study = None  # study that produced global best (for bootstrap CI)
    global_best_searchable = []  # searchable params in best phase
    global_best_ranges = {}  # ranges in best phase

    phase = 0
    _max_phases_hard_cap = max_phases + 2  # allow extra phases after late unfreeze
    while phase < max_phases:
        if trials_used >= total_trials:
            print(f"\n  Budget exhausted after phase {phase - 1}.")
            break

        if phase == 0:
            # Warm-up phase: full ranges
            phase_trials = min(warmup_trials, total_trials - trials_used)
            phase_tag = f"{run_prefix}_p0"
            phase_ranges = dict(SEARCH_RANGES)
            phase_searchable = list(initial_searchable)
        elif skip_analysis:
            # After unfreeze: run with current ranges, no analysis (no data yet)
            skip_analysis = False
            actions = []
            recently_frozen = set()
            phase_searchable = [p for p in current_ranges
                                if p not in frozen_params and p not in manually_fixed]
            phase_ranges = {p: current_ranges[p] for p in phase_searchable}

            print(f"\n{'~' * 70}")
            print(f"Phase {phase}: exploration after unfreeze (skipping analysis)")
            print(f"  Active: {len(phase_searchable)}D, Frozen: {len(frozen_params)}")

            # Budget: dimension-scaled
            n_active = len(phase_searchable)
            mult = _PHASE1_BUDGET_MULT if phase == 1 else 1.0
            phase_budget = max(min_phase_trials, int(trials_per_dim * n_active * mult))
            phase_trials = min(phase_budget, total_trials - trials_used)
            phase_tag = f"{run_prefix}_p{phase}"
        else:
            # Adaptive phase: analyze previous, adjust bounds
            print(f"\n{'~' * 70}")
            print(f"Phase {phase} analysis:")
            analysis = analyze_phase(prev_study, current_ranges,
                                     [p for p in current_ranges if p not in frozen_params
                                      and p not in manually_fixed],
                                     metric)

            # Print diagnostic table
            def _fv(v, w=10):
                """Format a value to fixed width, adapting precision."""
                if isinstance(v, int) or (isinstance(v, float) and v == int(v) and abs(v) < 1000):
                    return f"{int(v):{w}d}"
                if abs(v) >= 100:
                    return f"{v:{w}.2f}"
                if abs(v) >= 1:
                    return f"{v:{w}.4f}"
                return f"{v:{w}.4f}"

            max_imp = max((a["importance"] for a in analysis.values()), default=1.0)
            bar_width = 20
            W = 10  # column width for numeric values

            print(f"\n  {'Param':>6}  {'Imp':>5}  {'':>{bar_width}}  {'Mean':>{W}}  {'CI lo':>{W}}  {'CI hi':>{W}}  {'CI%':>4}  {'Bi':>2}  {'Edge':>4}")
            print(f"  {'-'*6}  {'-'*5}  {'-'*bar_width}  {'-'*W}  {'-'*W}  {'-'*W}  {'-'*4}  {'-'*2}  {'-'*4}")
            for p in sorted(analysis, key=lambda k: analysis[k]["importance"], reverse=True):
                a = analysis[p]
                s = _SHORT_NAMES.get(p, p)
                spec = current_ranges[p]
                rng_w = spec[1] - spec[0]
                ci_w = a["ci_hi"] - a["ci_lo"]
                ci_pct = f"{100 * ci_w / rng_w:.0f}%" if rng_w > 0 else "?"
                multi = "*" if a["multimodal"] else ""
                bdry = a["boundary"] or ""
                n_blocks = int(bar_width * a["importance"] / max_imp) if max_imp > 0 else 0
                bar = "#" * n_blocks + "." * (bar_width - n_blocks)
                print(f"  {s:>6}  {a['importance']:>5.3f}  {bar}  {_fv(a['mean'])}"
                      f"  {_fv(a['ci_lo'])}  {_fv(a['ci_hi'])}"
                      f"  {ci_pct:>4}  {multi:>2}  {bdry:>4}")
            if frozen_params:
                frozen_strs = []
                for p, info in frozen_params.items():
                    s = _SHORT_NAMES.get(p, p)
                    frozen_strs.append(f"{s}={info['value']:.3g}")
                print(f"\n  Frozen: {', '.join(frozen_strs)}")

            # Compute next phase
            prev_frozen = set(frozen_params.keys())
            new_ranges, frozen_params, phase_searchable, actions = compute_next_phase(
                analysis, current_ranges, frozen_params, phase,
                unfreeze_interval, manually_fixed,
                phase_improved=not last_phase_regressed,
                all_studies=all_studies)
            recently_frozen = set(frozen_params.keys()) - prev_frozen

            if not phase_searchable:
                # Evaluate the all-frozen config before stopping
                frozen_config = dict(PRODUCTION_PARAMS)
                if set_params:
                    frozen_config.update(set_params)
                frozen_config.update({p: info["value"] for p, info in frozen_params.items()})
                result = evaluate_trial(frozen_config, full_data, ground_truth, metric)
                score = result["value"]
                print(f"  All params frozen — verification score: {score:.4f}")
                if score > global_best:
                    global_best = score
                    candidate = {**frozen_config}
                    for p in SEARCH_RANGES:
                        if p not in candidate:
                            if set_params and p in set_params:
                                candidate[p] = set_params[p]
                            else:
                                candidate[p] = PRODUCTION_PARAMS.get(p, 0)
                    global_best_params = candidate
                break

            # Sanity: frozen and searchable must not overlap
            _overlap = set(frozen_params.keys()) & set(phase_searchable)
            if _overlap:
                raise RuntimeError(
                    f"BUG: params both frozen and searchable: {_overlap}")
            # Sanity: every non-fixed param must be frozen or searchable
            _all_params = set(SEARCH_RANGES.keys()) - manually_fixed
            _accounted = set(frozen_params.keys()) | set(phase_searchable)
            _missing = _all_params - _accounted
            if _missing:
                raise RuntimeError(
                    f"BUG: params neither frozen nor searchable: {_missing}")

            # Print decisions (aligned columns)
            print(f"\n  Decisions:")
            if actions:
                max_verb = max(len(a[1]) for a in actions)
                max_detail = max(len(a[2]) for a in actions)
                for _, verb, detail, reason in actions:
                    print(f"    {verb:<{max_verb}}  {detail:<{max_detail}}  ({reason})")
            print(f"\n  Active: {len(phase_searchable)}D, Frozen: {len(frozen_params)}")

            # Merge: keep ranges for all params (frozen retain last-known)
            current_ranges.update(new_ranges)
            phase_ranges = new_ranges

            # Budget: dimension-scaled
            n_active = len(phase_searchable)
            mult = _PHASE1_BUDGET_MULT if phase == 1 else 1.0
            phase_budget = max(min_phase_trials, int(trials_per_dim * n_active * mult))
            phase_trials = min(phase_budget, total_trials - trials_used)
            phase_tag = f"{run_prefix}_p{phase}"

        # Guard: skip phase if remaining budget is too small for meaningful analysis
        if phase_trials <= 0:
            print(f"\n  No budget remaining for phase {phase} — stopping.")
            break
        if phase > 0 and phase_trials < 10:
            print(f"\n  Phase {phase} budget ({phase_trials}) too small for "
                  f"reliable analysis — stopping.")
            break

        # Build fixed params for this phase
        phase_fixed = {k: v for k, v in PRODUCTION_PARAMS.items()
                       if k not in phase_searchable}
        if set_params:
            for k, v in set_params.items():
                phase_fixed[k] = v
        for p, info in frozen_params.items():
            phase_fixed[p] = info["value"]

        # Sanity: fixed and searchable must not overlap
        _fix_search_overlap = set(phase_fixed.keys()) & set(phase_searchable)
        if _fix_search_overlap:
            raise RuntimeError(
                f"BUG: params both fixed and searchable: {_fix_search_overlap}")

        print(f"\n--- Phase {phase}: {phase_trials} trials, {len(phase_searchable)}D ---")

        # Seed new study from previous
        import optuna
        from optuna.samplers import TPESampler
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        storage_dir = DATA_DIR / "tuning_studies"
        storage_dir.mkdir(parents=True, exist_ok=True)
        rdb_path = storage_dir / "tune_gt.db"
        rdb_url = f"sqlite:///{rdb_path}"

        is_dual = metric in ("dual", "dual_recall")
        _metric_tag = f"blended_r{metric[1]:.2f}" if isinstance(metric, tuple) and metric[0] == "blended" else str(metric)
        study_name = f"{phase_tag}_{_metric_tag}_{len(phase_searchable)}D"

        storage = optuna.storages.RDBStorage(
            url=rdb_url,
            engine_kwargs={"connect_args": {"timeout": 120}},
        )
        if is_dual:
            new_study = optuna.create_study(
                study_name=study_name,
                directions=["maximize", "maximize"],
                sampler=optuna.samplers.NSGAIISampler(),
                storage=storage,
                load_if_exists=True,
            )
        else:
            new_study = optuna.create_study(
                study_name=study_name,
                direction="maximize",
                sampler=TPESampler(n_startup_trials=max(20, len(phase_searchable) * 5)),
                storage=storage,
                load_if_exists=True,
            )

        if prev_study is not None:
            n_seeded = seed_study(new_study, prev_study, phase_searchable,
                                  phase_ranges)
            print(f"  Seeded {n_seeded} trials from previous phase")

        # Run the phase using run_tuning's core logic
        # We duplicate the worker launch here to avoid run_tuning's data loading
        # and baseline printing on every phase
        if n_jobs > 1:
            per_worker = phase_trials // n_jobs
            remainder = phase_trials % n_jobs
            worker_trials = [per_worker + (1 if i < remainder else 0)
                             for i in range(n_jobs)]

            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                futures = [
                    pool.submit(_mp_worker,
                                worker_id=i,
                                study_name=study_name,
                                rdb_url=rdb_url,
                                n_trials=worker_trials[i],
                                searchable=phase_searchable,
                                fixed=phase_fixed,
                                data_cache_path=data_cache_path,
                                gt_path=gt_path,
                                metric=metric,
                                seed=42 + phase * 100 + i,
                                active_ranges=phase_ranges)
                    for i in range(n_jobs)
                ]

                t_start = time.monotonic()
                seen = set(t.number for t in new_study.trials
                           if (t.values is not None if is_dual else t.value is not None))
                best_val = [0.0]
                best_at = [0]
                n_completed = [0]

                while not all(f.done() for f in futures):
                    time.sleep(2)
                    try:
                        check = optuna.load_study(study_name=study_name, storage=storage)
                        new_trials = [t for t in check.trials
                                      if (t.values is not None if is_dual else t.value is not None)
                                      and t.number not in seen]
                        new_trials.sort(key=lambda t: t.number)
                        for trial in new_trials:
                            seen.add(trial.number)
                            n_completed[0] += 1
                            _print_trial(trial, phase_searchable, n_completed,
                                         phase_trials, best_val, best_at, t_start,
                                         dual=is_dual, metric=metric)
                    except Exception:
                        pass

                # Catch remaining
                check = optuna.load_study(study_name=study_name, storage=storage)
                for trial in check.trials:
                    has_val = trial.values is not None if is_dual else trial.value is not None
                    if has_val and trial.number not in seen:
                        seen.add(trial.number)
                        n_completed[0] += 1
                        _print_trial(trial, phase_searchable, n_completed,
                                     phase_trials, best_val, best_at, t_start,
                                     dual=is_dual, metric=metric)

                for f in futures:
                    if f.exception():
                        print(f"  Worker error: {f.exception()}")

            new_study = optuna.load_study(study_name=study_name, storage=storage)
        else:
            best_val = [0.0]
            best_at = [0]
            n_completed = [0]
            t_start = time.monotonic()

            def objective(trial, _fixed=phase_fixed, _search=phase_searchable,
                          _ranges=phase_ranges):
                return _run_trial(dict(_fixed), list(_search), full_data,
                                  ground_truth, metric, trial,
                                  active_ranges=_ranges)

            def _log_trial(study, trial):
                has_val = trial.values is not None if is_dual else trial.value is not None
                if not has_val:
                    return
                n_completed[0] += 1
                _print_trial(trial, phase_searchable, n_completed,
                             phase_trials, best_val, best_at, t_start,
                             dual=is_dual, metric=metric)

            new_study.optimize(objective, n_trials=phase_trials,
                               catch=(Exception,), callbacks=[_log_trial])

        # Phase results
        phase_best = _get_best_metric(new_study, metric)
        improvement = phase_best - global_best

        # Accumulate completed trials with complete param sets (for cross-phase Pareto)
        if is_dual:
            for t in new_study.trials:
                if t.values is not None:
                    complete = dict(phase_fixed)
                    complete.update(t.params)
                    all_completed_trials.append((t.values, complete))

        # Track global best params (complete set including frozen)
        if phase_best >= global_best:
            if is_dual:
                # Find actual best-primary-objective trial (not just Pareto front)
                completed = [t for t in new_study.trials if t.values is not None]
                best_t = max(completed, key=lambda t: t.values[0]) if completed else None
                candidate = {**{p: info["value"] for p, info in frozen_params.items()},
                             **best_t.params} if best_t else None
            elif not is_dual and new_study.best_params:
                candidate = {**{p: info["value"] for p, info in frozen_params.items()},
                             **new_study.best_params}
            else:
                candidate = None
            if candidate:
                # Fill fixed/missing from set_params then production
                for p in SEARCH_RANGES:
                    if p not in candidate:
                        if set_params and p in set_params:
                            candidate[p] = set_params[p]
                        else:
                            candidate[p] = PRODUCTION_PARAMS.get(p, 0)
                global_best_params = candidate
                global_best_study = new_study
                global_best_searchable = list(phase_searchable)
                global_best_ranges = dict(phase_ranges) if phase_ranges else dict(current_ranges)

        if phase_best > global_best + _CONVERGENCE_THRESHOLD:
            last_phase_regressed = False
            global_best = phase_best
            stale_phases = 0
        else:
            last_phase_regressed = phase_best < global_best - _CONVERGENCE_THRESHOLD
            if phase_best > global_best:
                global_best = phase_best
            stale_phases += 1

        trials_used += phase_trials
        remaining = total_trials - trials_used
        all_studies.append(new_study)

        # Record trajectory
        for p in SEARCH_RANGES:
            if p in manually_fixed:
                param_trajectory[p].append("fixed")
            elif p in frozen_params:
                param_trajectory[p].append(f"frozen({frozen_params[p]['value']:.3g})")
            elif p in phase_ranges:
                lo, hi = phase_ranges[p][:2]
                param_trajectory[p].append(f"[{lo:.3g},{hi:.3g}]")
            else:
                param_trajectory[p].append("--")

        # Collect completed trial count for this phase
        n_complete = len([t for t in new_study.trials
                          if (hasattr(t, 'values') and t.values is not None)
                          or (not is_dual and t.value is not None)])

        phase_history.append({
            "phase": phase,
            "study_name": study_name,
            "n_trials": phase_trials,
            "n_complete": n_complete,
            "best_metric": phase_best,
            "improvement": improvement,
            "n_active": len(phase_searchable),
            "n_frozen": len(frozen_params),
            "actions": actions if phase > 0 else [],
        })

        prev_study = new_study

        # Check for regression after freezing — unfreeze recent freezes
        if (improvement < _REGRESSION_THRESHOLD
                and recently_frozen and phase > 0):
            unfroze = []
            for p in list(recently_frozen):
                if p in frozen_params and p not in manually_fixed:
                    current_ranges[p] = _unfreeze_range(p, frozen_params[p],
                                                         current_ranges.get(p),
                                                         all_studies=all_studies)
                    del frozen_params[p]
                    unfroze.append(_SHORT_NAMES.get(p, p))
            if unfroze:
                print(f"\n  Regression ({improvement:+.4f}) after freezing "
                      f"{', '.join(unfroze)} — unfreezing them.")
                stale_phases = max(0, stale_phases - 1)
                skip_analysis = True  # no data for unfrozen params yet

        # Check convergence — must verify with all params unlocked
        if stale_phases >= _CONVERGENCE_PATIENCE:
            if frozen_params and unfreeze_cycles < _MAX_UNFREEZE_CYCLES:
                unfreeze_cycles += 1
                print(f"\n  Stale for {_CONVERGENCE_PATIENCE} phases with "
                      f"{len(frozen_params)} frozen params — unfreezing all "
                      f"for convergence check (cycle {unfreeze_cycles}/{_MAX_UNFREEZE_CYCLES}).")
                # Restore ranges for frozen params
                for p, info in frozen_params.items():
                    if p in manually_fixed:
                        continue
                    current_ranges[p] = _unfreeze_range(p, info, current_ranges.get(p),
                                                         all_studies=all_studies)
                n_unfrozen = len(frozen_params)
                frozen_params.clear()
                stale_phases = 0  # full patience for convergence check
                skip_analysis = True  # next phase: explore first, analyze after
                # Boost budget: unfreeze adds dimensions, needs more trials
                n_active_after = len([p for p in SEARCH_RANGES
                                      if p not in manually_fixed])
                min_unfreeze_trials = trials_per_dim * n_active_after
                remaining = max(remaining, min_unfreeze_trials)
                total_trials = trials_used + remaining
                print(f"  Budget boosted: {min_unfreeze_trials} min trials "
                      f"for {n_active_after}D convergence check.")
            else:
                if unfreeze_cycles >= _MAX_UNFREEZE_CYCLES and frozen_params:
                    print(f"\n  Converged: unfreeze cycle cap ({_MAX_UNFREEZE_CYCLES}) "
                          f"reached with {len(frozen_params)} frozen params.")
                else:
                    print(f"\n  Converged: no improvement > {_CONVERGENCE_THRESHOLD} "
                          f"for {_CONVERGENCE_PATIENCE} phases (all params unlocked).")
                break

        # Extend max_phases if unfreeze triggered near the end
        if skip_analysis and phase >= max_phases - 1 and max_phases < _max_phases_hard_cap:
            max_phases = min(max_phases + 1, _max_phases_hard_cap)
            print(f"  Extended max_phases to {max_phases} for unfreeze verification.")

        phase += 1

    # Plot phase (before summary so its results are included)
    plot_dir = None
    is_dual = metric in ("dual", "dual_recall")
    if prev_study and not skip_plots:
        storage_dir = DATA_DIR / "tuning_studies"
        plot_dir = storage_dir / "plots" / run_prefix
        plot_dir.mkdir(parents=True, exist_ok=True)

    plot_phase_trials = 0
    if prev_study and global_best_study and global_best_params and not skip_plots:
        _PLOT_TOP_N = 6   # top params to actually plot

        try:
            # Use global_best_study for all charts (no separate plot phase)
            plot_study = global_best_study
            plot_study_name = plot_study.study_name

            # Generate plots — top N by importance for slice/rank/contour
            print(f"\n  Plots -> {plot_dir}")
            sys.path.insert(0, str(REPO_ROOT / "scripts"))
            _rdb = storage_dir / "tune_gt.db"

            if is_dual:
                from plot_optuna_charts import (
                    plot_history_dual, plot_edf_dual,
                    plot_importance_dual, plot_metrics,
                    plot_slice, plot_contour_only,
                    plot_range_evolution, extract_all_run_data,
                )
                if metric == "dual_recall":
                    _OBJ_LABELS = {0: "recall", 1: "ndcg"}
                else:
                    _OBJ_LABELS = {0: "ndcg", 1: "recall"}

                # Load all-run data for background scatter
                _bg_data = {}
                for oi in _OBJ_LABELS:
                    try:
                        _bg_data[oi] = extract_all_run_data(
                            _rdb, run_prefix, objective_idx=oi)
                    except Exception:
                        _bg_data[oi] = None

                # --- Combined charts (objective-independent) ---
                _recall_primary = metric == "dual_recall"
                def _run_combined():
                    plot_history_dual(_rdb, run_prefix,
                                     plot_dir / "history.png",
                                     recall_primary=_recall_primary)
                    print("  history.png")
                    plot_edf_dual(_rdb, run_prefix,
                                 plot_dir / "edf.png",
                                 recall_primary=_recall_primary)
                    print("  edf.png")
                    plot_importance_dual(plot_study,
                                        plot_dir / "importance.png",
                                        recall_primary=_recall_primary)
                    print("  importance.png")
                    plot_metrics(plot_study, plot_dir / "metrics.png",
                                objective_idx=0,
                                bg_data=_bg_data.get(0),
                                recall_primary=_recall_primary)
                    print("  metrics.png")
                    plot_range_evolution(
                        param_trajectory, HARD_LIMITS, _SHORT_NAMES,
                        global_best_params,
                        plot_dir / "range_evolution.png")
                    print("  range_evolution.png")

                # --- Per-objective charts (slice all, rank top6, contour top6) ---
                def _run_per_obj(oi, label):
                    bg = _bg_data.get(oi)
                    plot_slice(plot_study, plot_dir / f"slice_{label}.png",
                              objective_idx=oi, bg_data=bg)
                    print(f"  slice_{label}.png")
                    plot_slice(plot_study, plot_dir / f"slice_{label}_wide.png",
                              objective_idx=oi, bg_data=bg,
                              hard_limits=HARD_LIMITS)
                    print(f"  slice_{label}_wide.png")
                    plot_contour_only(
                        plot_study, plot_dir, rdb_path=_rdb,
                        study_name=plot_study_name,
                        top_n=_PLOT_TOP_N, objective_idx=oi,
                        suffix=f"_{label}")

                # Get top params for rank/contour
                from plot_optuna_charts import _get_top_params
                try:
                    _plot_top_params = _get_top_params(
                        plot_study, _PLOT_TOP_N, objective_idx=0)
                    print(f"  Top {len(_plot_top_params)} params: "
                          f"{', '.join(_plot_top_params)}")
                except Exception:
                    _plot_top_params = None

                if n_jobs == 1:
                    _run_combined()
                    for oi, label in _OBJ_LABELS.items():
                        _run_per_obj(oi, label)
                else:
                    from concurrent.futures import ThreadPoolExecutor as _PlotPool
                    max_w = min(n_jobs, 3)  # combined + ndcg + recall
                    with _PlotPool(max_workers=max_w) as ppool:
                        pfuts = {
                            ppool.submit(_run_combined): "combined",
                        }
                        for oi, label in _OBJ_LABELS.items():
                            pfuts[ppool.submit(_run_per_obj, oi, label)] = \
                                f"{label} per-obj"
                        for pf in pfuts:
                            try:
                                pf.result()
                            except Exception as pe:
                                print(f"  {pfuts[pf]} failed: {pe}")
            elif isinstance(metric, tuple) and metric[0] == "blended":
                from plot_optuna_charts import (
                    plot_importance_blended, plot_metrics, plot_slice,
                    plot_contour_only, plot_range_evolution,
                    extract_all_run_data,
                )

                # Load merged data from ALL phases for slice/metrics plots
                _all_data = extract_all_run_data(_rdb, run_prefix)
                _all_ndcg = extract_all_run_data(
                    _rdb, run_prefix,
                    objective_idx=lambda t: t.user_attrs.get("ndcg_5k", 0))
                _all_recall = extract_all_run_data(
                    _rdb, run_prefix,
                    objective_idx=lambda t: t.user_attrs.get("graded_recall_5k", 0))

                # Compute union of all actually-searched bounds across phases
                _union_limits = {}
                for _p, _spec in SEARCH_RANGES.items():
                    _vals = _all_data["params"].get(_p, [])
                    if _vals:
                        _lo, _hi = min(_vals), max(_vals)
                        _margin = (_hi - _lo) * 0.05 if _hi > _lo else 0.01
                        _union_limits[_p] = (_lo - _margin, _hi + _margin, _spec[2])
                    else:
                        _union_limits[_p] = _spec

                _blended_plots = [
                    ("importance.png",
                     lambda: plot_importance_blended(plot_study, plot_dir / "importance.png")),
                    ("slice_blended.png",
                     lambda: plot_slice(plot_study, plot_dir / "slice_blended.png",
                                        data_override=_all_data,
                                        hard_limits=_union_limits)),
                    ("slice_ndcg.png",
                     lambda: plot_slice(plot_study, plot_dir / "slice_ndcg.png",
                                        data_override=_all_ndcg,
                                        hard_limits=_union_limits)),
                    ("slice_recall.png",
                     lambda: plot_slice(plot_study, plot_dir / "slice_recall.png",
                                        data_override=_all_recall,
                                        hard_limits=_union_limits)),
                    ("metrics.png",
                     lambda: plot_metrics(plot_study, plot_dir / "metrics.png",
                                          primary_label="Blended", skip_attrs=set(),
                                          data_override=_all_data)),
                    ("range_evolution.png",
                     lambda: plot_range_evolution(
                         param_trajectory, HARD_LIMITS, _SHORT_NAMES,
                         global_best_params,
                         plot_dir / "range_evolution.png")),
                    ("contour.png",
                     lambda: plot_contour_only(
                         plot_study, plot_dir, rdb_path=_rdb,
                         study_name=plot_study_name,
                         top_n=_PLOT_TOP_N)),
                ]
                for _pname, _pfn in _blended_plots:
                    try:
                        _pfn()
                        print(f"  {_pname}")
                    except Exception as _pe:
                        print(f"  {_pname} FAILED: {_pe}")
            else:
                from plot_optuna_charts import plot_all
                plot_all(plot_study, plot_dir,
                         rdb_path=_rdb,
                         study_name=plot_study_name,
                         top_n=_PLOT_TOP_N)

        except Exception as e:
            import traceback
            print(f"\n  Plot phase failed: {e}")
            traceback.print_exc()
            # Fall back to static plots from best study
            if plot_dir:
                print(f"\n  Plots (fallback) -> {plot_dir}")
                try:
                    sys.path.insert(0, str(REPO_ROOT / "scripts"))
                    from plot_optuna_charts import plot_all
                    plot_study = global_best_study or prev_study
                    plot_all(plot_study, plot_dir,
                             rdb_path=storage_dir / "tune_gt.db",
                             study_name=plot_study.study_name,
                             top_n=6)
                except Exception as e2:
                    print(f"    Fallback plots failed: {e2}")

    # Non-dual: multi-phase history/EDF + range evolution
    if plot_dir and len(phase_history) > 0 and not is_dual:
        try:
            sys.path.insert(0, str(REPO_ROOT / "scripts"))
            from plot_optuna_charts import (plot_history_multiphase,
                                            plot_edf_multiphase,
                                            plot_range_evolution)
            _rdb = storage_dir / "tune_gt.db"
            plot_range_evolution(param_trajectory, HARD_LIMITS, _SHORT_NAMES,
                                global_best_params,
                                plot_dir / "range_evolution.png")
            print(f"  range_evolution.png")
            plot_history_multiphase(
                _rdb, run_prefix,
                plot_dir / "history.png")
            print(f"  history.png (all phases)")
            plot_edf_multiphase(
                _rdb, run_prefix,
                plot_dir / "edf.png")
            print(f"  edf.png (all phases)")
        except Exception as e:
            import traceback
            print(f"  Multi-phase charts failed: {e}")
            traceback.print_exc()

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"ADAPTIVE TUNING SUMMARY")
    print(f"{'=' * 70}")
    n_adaptive = len(phase_history)
    total_all_trials = trials_used
    print(f"  {n_adaptive} phases, "
          f"{total_all_trials} trials, best "
          f"{'Blended' if isinstance(metric, tuple) and metric[0] == 'blended' else 'Recall' if metric == 'dual_recall' else 'NDCG'} {global_best:.4f}")

    # Phase timeline
    print(f"\n  Phase Timeline:")
    print(f"  {'':>3}  {'Trials':>7}  {'Dims':>4}  {'Best':>7}  {'vs prev':>7}  Notes")
    print(f"  {'':>3}  {'------':>7}  {'----':>4}  {'------':>7}  {'------':>7}  -----")
    for ph in phase_history:
        n_ok = ph.get('n_complete', ph['n_trials'])
        trials_str = f"{n_ok}/{ph['n_trials']}" if n_ok < ph['n_trials'] else str(ph['n_trials'])
        dims = f"{ph['n_active']}D"
        if ph['n_frozen']:
            dims += f"+{ph['n_frozen']}f"
        imp = ph['improvement']
        phase_label = f"p{ph['phase']}" if isinstance(ph['phase'], int) else ph['phase']
        imp_str = f"{imp:+.4f}" if ph['phase'] != 0 else ""
        # Summarize notable actions
        notes = []
        if ph['phase'] == "plot":
            notes.append("focused exploration")
        for _, verb, detail, _ in ph.get("actions", []):
            if verb in ("freeze", "expand", "unfreeze"):
                notes.append(f"{verb} {detail.split()[0]}")
        notes_str = ", ".join(notes[:4])
        if len(notes) > 4:
            notes_str += f" +{len(notes)-4} more"
        print(f"  {phase_label:<4} {trials_str:>7}  {dims:>4}  {ph['best_metric']:>.4f}"
              f"  {imp_str:>7}  {notes_str}")

    # Per-param summary: what settled where and how confidently
    print(f"\n  Parameter Summary:")
    print(f"  {'':>6}  {'Prod':>8}  {'Best':>8}  {'Status':<30}  Journey")
    print(f"  {'':>6}  {'----':>8}  {'----':>8}  {'------':<30}  -------")
    for p in SEARCH_RANGES:
        s = _SHORT_NAMES.get(p, p)
        traj = param_trajectory[p]
        prod = PRODUCTION_PARAMS.get(p)
        ptype = SEARCH_RANGES[p][2]

        # Determine final state
        state = traj[-1]
        n_frozen_phases = sum(1 for t in traj if t.startswith("frozen"))
        n_narrow = 0
        n_expand = 0
        for i in range(1, len(traj)):
            prev_t, cur_t = traj[i-1], traj[i]
            if prev_t.startswith("[") and cur_t.startswith("["):
                try:
                    prev_lo, prev_hi = (float(x) for x in prev_t.strip("[]").split(","))
                    cur_lo, cur_hi = (float(x) for x in cur_t.strip("[]").split(","))
                    if (cur_hi - cur_lo) < (prev_hi - prev_lo) * 0.95:
                        n_narrow += 1
                    elif (cur_hi - cur_lo) > (prev_hi - prev_lo) * 1.05:
                        n_expand += 1
                except ValueError:
                    pass

        # Status description
        if state == "fixed":
            status = "fixed (manual)"
        elif state.startswith("frozen"):
            val = state[7:-1]
            status = f"converged -> {val}"
        elif state.startswith("["):
            try:
                lo, hi = state.strip("[]").split(",")
                status = f"searching [{lo}, {hi}]"
            except ValueError:
                status = f"searching {state}"
        else:
            status = state

        # Journey summary
        journey_parts = []
        if n_narrow: journey_parts.append(f"{n_narrow}x narrowed")
        if n_expand: journey_parts.append(f"{n_expand}x expanded")
        if n_frozen_phases: journey_parts.append(f"{n_frozen_phases}x frozen")
        journey = ", ".join(journey_parts) if journey_parts else "unchanged"

        # Format values
        if ptype == "int":
            prod_str = str(prod) if prod is not None else "?"
            best_str = ""  # will be in best params section
        else:
            prod_str = f"{prod:.4f}" if prod is not None else "?"
            best_str = ""

        print(f"  {s:>6}  {prod_str:>8}  {'':>8}  {status:<30}  {journey}")

    # Trajectory detail (column-aligned with bracket/comma alignment)
    all_trajs = {}
    for p in SEARCH_RANGES:
        traj = param_trajectory[p]
        compressed = [traj[0]]
        for t in traj[1:]:
            compressed.append("." if t == compressed[-1] else t)
        all_trajs[p] = compressed
    n_phases = max(len(t) for t in all_trajs.values())

    # For each column, find max width of integer/decimal parts of numbers
    # so decimal points align within [lo, hi] entries
    # Each number has int_part.dec_part; we align on the '.'
    def _num_parts(s):
        """Split a number string into (int_width, dec_width) for alignment.
        int_width includes space for sign (always reserved).
        """
        s = s.strip()
        raw = s.lstrip('-')
        # +1 for sign space always
        if '.' in raw:
            ip, dp = raw.split('.', 1)
            return len(ip) + 1, len(dp)  # +1 for sign
        return len(raw) + 1, -1  # -1 means no decimal

    # Per-column: max int/dec widths for lo and hi parts
    col_lo_int = [0] * n_phases
    col_lo_dec = [0] * n_phases
    col_hi_int = [0] * n_phases
    col_hi_dec = [0] * n_phases
    col_other_w = [0] * n_phases
    for p in SEARCH_RANGES:
        traj = all_trajs[p]
        for i, entry in enumerate(traj):
            if i >= n_phases:
                break
            if entry.startswith("[") and "," in entry:
                lo_part, hi_part = entry[1:-1].split(",", 1)
                li, ld = _num_parts(lo_part)
                hi_, hd = _num_parts(hi_part)
                col_lo_int[i] = max(col_lo_int[i], li)
                col_lo_dec[i] = max(col_lo_dec[i], ld)
                col_hi_int[i] = max(col_hi_int[i], hi_)
                col_hi_dec[i] = max(col_hi_dec[i], hd)
            else:
                col_other_w[i] = max(col_other_w[i], len(entry))

    DIM = "\033[2m"
    RST = "\033[0m"

    def _align_num(s, max_int, max_dec):
        """Align a number string on its decimal point.

        Always reserves space for sign. Pads decimals with dimmed trailing
        zeroes so decimal points and digits align.
        """
        s = s.strip()
        # Parse sign
        if s.startswith("-"):
            sign = "-"
            s = s[1:]
        else:
            sign = " "

        if '.' in s:
            ip, dp = s.split('.', 1)
        elif max_dec > 0:
            ip, dp = s, ""
        else:
            return f"{sign}{s:>{max_int - 1}}"

        # Pad decimals: real digits left-aligned, trailing zeroes dimmed
        if max_dec > 0:
            real_dec = dp.rstrip('0') if dp else ""
            n_trailing = max_dec - len(dp)
            n_zero_in_dp = len(dp) - len(real_dec) if dp else 0
            if n_trailing > 0:
                # Original had fewer decimals than max — pad with dimmed zeroes
                dec_str = f"{dp}{DIM}{'0' * n_trailing}{RST}"
            elif n_zero_in_dp > 0:
                # Original has trailing zeroes — dim them
                dec_str = f"{real_dec}{DIM}{'0' * n_zero_in_dp}{RST}"
            else:
                dec_str = dp
            return f"{sign}{ip:>{max_int - 1}}.{dec_str}"
        else:
            return f"{sign}{ip:>{max_int - 1}}"

    # Total column width
    col_widths = []
    for i in range(n_phases):
        lo_w = col_lo_int[i] + (col_lo_dec[i] + 1 if col_lo_dec[i] > 0 else 0)
        hi_w = col_hi_int[i] + (col_hi_dec[i] + 1 if col_hi_dec[i] > 0 else 0)
        bracket_w = lo_w + hi_w + 4 if (lo_w + hi_w) > 0 else 0  # [lo, hi]
        col_widths.append(max(bracket_w, col_other_w[i], 3))

    def _fmt_entry(entry, i):
        w = col_widths[i]
        if entry.startswith("[") and "," in entry:
            lo_part, hi_part = entry[1:-1].split(",", 1)
            lo_aligned = _align_num(lo_part, col_lo_int[i], col_lo_dec[i])
            hi_aligned = _align_num(hi_part, col_hi_int[i], col_hi_dec[i])
            return f"[{lo_aligned}, {hi_aligned}]"
        return f"{entry:<{w}}"

    # Print in chunks of 5 columns
    max_cols_per_row = 5
    print(f"\n  Range Evolution:")
    for chunk_start in range(0, n_phases, max_cols_per_row):
        chunk_end = min(chunk_start + max_cols_per_row, n_phases)
        chunk = range(chunk_start, chunk_end)
        hdr_parts = [f"p{i:<{col_widths[i]-1}}" for i in chunk]
        if chunk_start > 0:
            print()  # blank line between chunks
        print(f"    {'':>6}  {'  '.join(hdr_parts)}")
        for p in SEARCH_RANGES:
            s = _SHORT_NAMES.get(p, p)
            traj = all_trajs[p]
            parts = []
            for i in chunk:
                entry = traj[i] if i < len(traj) else ""
                formatted = _fmt_entry(entry, i)
                parts.append(f"{formatted:<{col_widths[i]}}")
            print(f"    {s:>6}  {'  '.join(parts)}")

    # Best params (tracked across all phases, not just final)
    best_params = dict(global_best_params)
    if set_params:
        best_params.update(set_params)
    if prev_study and best_params:
        # Bootstrap CIs from the phase that produced the global best
        ci_study = global_best_study or prev_study
        ci_searchable = global_best_searchable or [k for k in SEARCH_RANGES
                            if k not in frozen_params and k not in manually_fixed]
        ci_ranges = global_best_ranges or {k: current_ranges.get(k, SEARCH_RANGES[k])
                        for k in ci_searchable}
        boot = {}
        if ci_searchable:
            boot = bootstrap_param_bounds(ci_study, ci_searchable, ci_ranges)

        print(f"\n  Best params:")
        for k in SEARCH_RANGES:
            v = best_params.get(k)
            if v is None:
                continue
            s = _SHORT_NAMES.get(k, k)
            prod = PRODUCTION_PARAMS.get(k)
            ptype = SEARCH_RANGES[k][2]
            was = f"  (was {prod})" if prod is not None and ptype == "int" else \
                  f"  (was {prod:.4f})" if prod is not None else ""
            if ptype == "int":
                line = f"    {s:>6} = {v}{was}"
            else:
                ci = boot.get(k)
                if ci:
                    _, _, ci_lo, ci_hi = ci
                    line = f"    {s:>6} = {v:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]{was}"
                else:
                    frozen_tag = "  (frozen)" if k in frozen_params else ""
                    line = f"    {s:>6} = {v:.4f}{was}{frozen_tag}"
            print(line)

    # Pareto front across all phases (dual metric only)
    if is_dual and all_completed_trials:
        _obj1, _obj2 = ("Recall", "NDCG") if metric == "dual_recall" else ("NDCG", "Recall")

        # Compute cross-phase Pareto front
        # A trial is Pareto-optimal if no other trial dominates it on both objectives
        pareto_indices = []
        for i, (vi, _) in enumerate(all_completed_trials):
            dominated = False
            for j, (vj, _) in enumerate(all_completed_trials):
                if i == j:
                    continue
                if vj[0] >= vi[0] and vj[1] >= vi[1] and (vj[0] > vi[0] or vj[1] > vi[1]):
                    dominated = True
                    break
            if not dominated:
                pareto_indices.append(i)

        # Sort by primary objective descending
        pareto_indices.sort(key=lambda i: all_completed_trials[i][0][0], reverse=True)

        # Identify which params vary across Pareto solutions
        pareto_params = [all_completed_trials[i][1] for i in pareto_indices]
        varying = [k for k in SEARCH_RANGES if k not in manually_fixed
                   and len(set(p.get(k) for p in pareto_params)) > 1]

        print(f"\n  Pareto front: {len(pareto_indices)} solutions (across {len(all_completed_trials)} trials)")
        print(f"    Baseline: NDCG={baseline['ndcg_5k']:.4f}, Recall={baseline['graded_recall_5k']:.4f}")
        print(f"\n    {'#':>3}  {_obj1:>7}  {_obj2:>7}  Params")
        print(f"    {'-'*3}  {'-'*7}  {'-'*7}  {'-'*50}")
        for pi, idx in enumerate(pareto_indices):
            vals, params = all_completed_trials[idx]
            parts = []
            for k in varying:
                v = params.get(k)
                if v is None:
                    continue
                s = _SHORT_NAMES.get(k, k)
                parts.append(f"{s}={v:.3f}")
            print(f"    {pi+1:>3}  {vals[0]:>7.4f}  {vals[1]:>7.4f}  {' '.join(parts)}")

    # Compare best against baseline
    if prev_study:
        print(f"\n  vs Baseline:")
        is_blended = isinstance(metric, tuple) and metric[0] == "blended"
        if is_blended:
            if best_params:
                final_eval = evaluate_trial(best_params, full_data, ground_truth, metric)
                bl_n, bl_r = baseline['ndcg_5k'], baseline['graded_recall_5k']
                fn, fr = final_eval['ndcg_5k'], final_eval['graded_recall_5k']
                bl_score = metric[1] + 1.0
                bps = (final_eval['value'] - bl_score) * 10000
                print(f"    Baseline:  N={bl_n*100:.2f}%  R={bl_r*100:.2f}%")
                print(f"    Best:      N={fn*100:.2f}%  R={fr*100:.2f}%  ({bps:+.0f}bp blended)")
                print(f"    Delta:     N={fn - bl_n:+.4f}  R={fr - bl_r:+.4f}")
        elif metric == "dual_recall":
            print(f"    Baseline Recall: {baseline['graded_recall_5k']:.4f}")
            print(f"    Best Recall:     {global_best:.4f}  ({global_best - baseline['graded_recall_5k']:+.4f})")
            if best_params:
                final_eval = evaluate_trial(best_params, full_data, ground_truth, metric)
                print(f"    Final NDCG:      {final_eval['ndcg_5k']:.4f}  "
                      f"({final_eval['ndcg_5k'] - baseline['ndcg_5k']:+.4f})")
        else:
            print(f"    Baseline NDCG:  {baseline['ndcg_5k']:.4f}")
            print(f"    Best NDCG:      {global_best:.4f}  ({global_best - baseline['ndcg_5k']:+.4f})")
            if best_params:
                final_eval = evaluate_trial(best_params, full_data, ground_truth, metric)
                print(f"    Final Recall:   {final_eval['graded_recall_5k']:.4f}  "
                      f"({final_eval['graded_recall_5k'] - baseline['graded_recall_5k']:+.4f})")

    if plot_dir:
        print(f"\n  Plots -> {plot_dir}")

    print(f"{'=' * 70}")

    return best_params, global_best


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def run_cv(gt_path: str, n_folds: int = 5, n_trials: int = 500,
           n_jobs: int = 1, tag: str = "", metric: str = "ndcg_5k",
           fix: list[str] | None = None,
           set_params: dict[str, float] | None = None,
           max_phases: int = 6, unfreeze_interval: int = 3,
           start_fold: int = 0):
    """K-fold cross-validation using adaptive tuning per fold."""
    import random

    # Load all data once
    full_data, ground_truth, data_cache_path = load_tuning_data(gt_path)

    all_queries = list(ground_truth.keys())
    print(f"\nCross-validation: {n_folds}-fold, {len(all_queries)} queries, metric={metric}")

    # Deterministic shuffle
    rng = random.Random(42)
    rng.shuffle(all_queries)

    # Build folds
    folds = [[] for _ in range(n_folds)]
    for i, q in enumerate(all_queries):
        folds[i % n_folds].append(q)
    print(f"  Fold sizes: {[len(f) for f in folds]}")

    # Production baseline on full set
    baseline = evaluate_trial(dict(PRODUCTION_PARAMS), full_data, ground_truth, metric)
    print(f"\nFull-set production baseline:")
    print(f"  NDCG@5k:          {baseline['ndcg_5k']:.4f}")
    print(f"  Graded Recall@5k: {baseline['graded_recall_5k']:.4f}")

    searchable = [k for k in SEARCH_RANGES if k not in (fix or [])]

    # Run each fold
    fold_results = []

    for fi in range(start_fold, n_folds):
        test_queries = folds[fi]
        train_queries = [q for j in range(n_folds) for q in folds[j] if j != fi]
        train_gt = {q: ground_truth[q] for q in train_queries}
        test_gt = {q: ground_truth[q] for q in test_queries}

        print(f"\n{'#' * 70}")
        print(f"# FOLD {fi+1}/{n_folds}: train={len(train_queries)}, test={len(test_queries)}")
        print(f"{'#' * 70}")

        # Build fold-specific data bundle
        train_data = {
            "gt_queries": train_queries,
            "token_map": full_data["token_map"],
            "feedback_raw": full_data["feedback_raw"],
            "hebb_data": full_data["hebb_data"],
            "search_data": full_data["search_data"],
        }

        # Save for mp workers
        fold_cache_path = DATA_DIR / "tuning_studies" / f"tune_gt_cv_fold{fi}.pkl"
        with open(fold_cache_path, "wb") as f:
            pickle.dump(train_data, f)
        train_gt_path = DATA_DIR / "tuning_studies" / f"tune_gt_cv_fold{fi}_gt.json"
        with open(train_gt_path, "w") as f:
            json.dump(train_gt, f)

        # Run adaptive tuning on train fold
        fold_tag = f"{tag}_cv{fi}" if tag else f"cv{fi}"
        fold_preloaded = (train_data, train_gt, str(fold_cache_path))

        best_params, fold_best_val = run_adaptive(
            gt_path=str(train_gt_path),
            total_trials=n_trials,
            n_jobs=n_jobs,
            tag=fold_tag,
            metric=metric,
            fix=fix,
            set_params=set_params,
            max_phases=max_phases,
            unfreeze_interval=unfreeze_interval,
            preloaded=fold_preloaded,
            skip_plots=True,
        )

        if not best_params:
            print(f"  No completed trials in fold {fi+1}")
            continue

        # Evaluate on held-out test set
        test_data = {
            "gt_queries": test_queries,
            "token_map": full_data["token_map"],
            "feedback_raw": full_data["feedback_raw"],
            "hebb_data": full_data["hebb_data"],
            "search_data": full_data["search_data"],
        }

        tuned_test = evaluate_trial(best_params, test_data, test_gt, metric)
        prod_test = evaluate_trial(dict(PRODUCTION_PARAMS), test_data, test_gt, metric)

        fold_results.append({
            "fold": fi,
            "train_best": fold_best_val,
            "test_ndcg": tuned_test["ndcg_5k"],
            "test_recall": tuned_test["graded_recall_5k"],
            "prod_ndcg": prod_test["ndcg_5k"],
            "prod_recall": prod_test["graded_recall_5k"],
            "best_params": best_params,
        })

        print(f"\n  FOLD {fi+1} HELD-OUT RESULTS:")
        _train_label = "Recall" if metric == "dual_recall" else "NDCG"
        print(f"    Train best {_train_label}: {fold_best_val:.4f}")
        print(f"    Test  NDCG@5k:  tuned={tuned_test['ndcg_5k']:.4f}  prod={prod_test['ndcg_5k']:.4f}"
              f"  delta={tuned_test['ndcg_5k'] - prod_test['ndcg_5k']:+.4f}")
        print(f"    Test  Recall:   tuned={tuned_test['graded_recall_5k']:.4f}  prod={prod_test['graded_recall_5k']:.4f}"
              f"  delta={tuned_test['graded_recall_5k'] - prod_test['graded_recall_5k']:+.4f}")

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"CROSS-VALIDATION SUMMARY ({n_folds}-fold, metric={metric})")
    print(f"{'=' * 70}")
    print(f"{'Fold':>6}  {'Train':>8}  {'Test NDCG':>10}  {'Prod NDCG':>10}  {'Delta':>8}  {'Test Recall':>12}  {'k_fts':>5}  {'k_vec':>5}  {'wt':>5}  {'damp':>5}")
    print(f"{'-'*6}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*12}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}")

    for r in fold_results:
        bp = r["best_params"]
        print(f"  {r['fold']+1:>4}  {r['train_best']:>8.4f}  {r['test_ndcg']:>10.4f}  {r['prod_ndcg']:>10.4f}"
              f"  {r['test_ndcg'] - r['prod_ndcg']:>+8.4f}  {r['test_recall']:>12.4f}"
              f"  {bp.get('k_fts', ''):>5.1f}  {bp.get('k_vec', ''):>5.1f}"
              f"  {bp.get('w_theme', 0):>5.2f}  {bp.get('ppr_damping', 0):>5.2f}")

    n_results = len(fold_results)
    if n_results > 0:
        mean_test_ndcg = sum(r["test_ndcg"] for r in fold_results) / n_results
        mean_prod_ndcg = sum(r["prod_ndcg"] for r in fold_results) / n_results
        mean_test_recall = sum(r["test_recall"] for r in fold_results) / n_results
        std_delta = (sum((r["test_ndcg"] - r["prod_ndcg"] - (mean_test_ndcg - mean_prod_ndcg))**2
                         for r in fold_results) / n_results) ** 0.5

        print(f"{'-'*6}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*12}")
        print(f"  Mean  {'':>8}  {mean_test_ndcg:>10.4f}  {mean_prod_ndcg:>10.4f}"
              f"  {mean_test_ndcg - mean_prod_ndcg:>+8.4f}  {mean_test_recall:>12.4f}")
        print(f"  Std(delta): {std_delta:.4f}")

        print(f"\n  Generalization: mean test delta = {mean_test_ndcg - mean_prod_ndcg:+.4f}"
              f" (train was ~{sum(r['train_best'] for r in fold_results)/n_results:.4f})")

    # Check param consistency
    print(f"\n  Param stability across folds:")
    for k in searchable:
        vals = [r["best_params"].get(k) for r in fold_results if k in r["best_params"]]
        if vals:
            mn = sum(vals) / len(vals)
            sd = (sum((v - mn)**2 for v in vals) / len(vals)) ** 0.5
            s = _SHORT_NAMES.get(k, k)
            prod = PRODUCTION_PARAMS.get(k, "?")
            if isinstance(vals[0], int):
                print(f"    {s:>6}: {' '.join(f'{v:>3}' for v in vals)}  mean={mn:.1f} std={sd:.1f}  (prod={prod})")
            else:
                print(f"    {s:>6}: {' '.join(f'{v:.2f}' for v in vals)}  mean={mn:.3f} std={sd:.3f}  (prod={prod})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Tune memory scoring against ground truth")
    _default_gt = str(DATA_DIR / "tuning_studies" / "gt_calibrated.json")
    parser.add_argument("--gt", default=_default_gt,
                        help=f"Path to ground_truth.json (default: {_default_gt})")
    parser.add_argument("--trials", type=int, default=500,
                        help="Number of Optuna trials (default: 500)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--tag", type=str, default="",
                        help="Study name tag prefix")
    parser.add_argument("--metric", type=str, default="ndcg_5k",
                        help="Optimization metric: ndcg_5k, graded_recall_5k, dual, dual_recall (default: ndcg_5k)")
    parser.add_argument("--ratio", type=float, default=None,
                        help="Blend ratio: score = NDCG * ratio + Recall. "
                             "Single-objective TPE. Overrides --metric to 'blended'.")
    parser.add_argument("--fix", nargs="*", default=None,
                        help="Params to hold at production values")
    parser.add_argument("--search", nargs="*", default=None,
                        help="Only search these params (fix everything else)")
    parser.add_argument("--set", nargs="*", default=None, dest="set_params",
                        help="Override param values: param=value (e.g. --set hebbian_coeff=0 hebbian_cap=0)")
    parser.add_argument("--cv", type=int, default=0,
                        help="Run K-fold cross-validation (e.g. --cv 5)")
    parser.add_argument("--adaptive", action="store_true",
                        help="Enable adaptive phase-based tuning")
    parser.add_argument("--max-phases", type=int, default=6,
                        help="Maximum adaptive phases (default: 10)")
    parser.add_argument("--unfreeze-interval", type=int, default=3,
                        help="Re-evaluate frozen params every N phases (default: 3)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume adaptive run using --tag as study name prefix")
    parser.add_argument("--start-fold", type=int, default=0,
                        help="Skip to this fold (0-indexed) in CV mode")
    args = parser.parse_args()

    # --search: invert to --fix (fix everything NOT in --search)
    if args.search is not None:
        if args.fix is not None:
            parser.error("Cannot use --fix and --search together")
        args.fix = [k for k in SEARCH_RANGES if k not in args.search]

    # --ratio: blended metric with baseline normalization
    # Marker tuple — resolved inside run_adaptive/run_study after baseline is computed
    if args.ratio is not None:
        args.metric = ("blended_pending", args.ratio)

    # Parse --set param=value pairs
    set_params = None
    if args.set_params:
        set_params = {}
        for s in args.set_params:
            k, v = s.split("=", 1)
            set_params[k] = int(v) if SEARCH_RANGES.get(k, (0, 0, "float"))[2] == "int" else float(v)

    if args.cv > 1:
        run_cv(
            gt_path=args.gt,
            n_folds=args.cv,
            n_trials=args.trials,
            n_jobs=args.jobs,
            tag=args.tag,
            metric=args.metric,
            fix=args.fix,
            set_params=set_params,
            max_phases=args.max_phases,
            unfreeze_interval=args.unfreeze_interval,
            start_fold=args.start_fold,
        )
    elif args.adaptive:
        run_adaptive(
            gt_path=args.gt,
            total_trials=args.trials,
            n_jobs=args.jobs,
            tag=args.tag,
            metric=args.metric,
            fix=args.fix,
            set_params=set_params,
            max_phases=args.max_phases,
            unfreeze_interval=args.unfreeze_interval,
            resume=args.resume,
        )
    else:
        run_tuning(
            gt_path=args.gt,
            n_trials=args.trials,
            n_jobs=args.jobs,
            tag=args.tag,
            metric=args.metric,
            fix=args.fix,
            set_params=set_params,
        )


if __name__ == "__main__":
    main()
