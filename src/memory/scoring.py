"""Scoring pipeline — RRF fusion, feedback, Hebbian, PPR expansion, quality floor.

Each function takes explicit parameters and modifies rrf_scores in-place (mutating dict).
Secondary data needed for logging is returned.
"""

import json
import math
import sqlite3
from datetime import datetime, timedelta, timezone

from memory.constants import (
    RRF_K, RRF_VEC_WEIGHT, FEEDBACK_COEFF,
    HEBBIAN_COEFF, HEBBIAN_CAP, HEBBIAN_MIN_JOINT,
    ADJACENCY_BASE_BOOST, ADJACENCY_NOVELTY_FLOOR,
    CONTEXT_RELEVANCE_THRESHOLD,
    CLIFF_Z_THRESHOLD, CLIFF_MIN_RESULTS,
    THEME_BOOST,
    ADJACENCY_SEED_COUNT, MAX_NEIGHBORS_PER_SEED, MAX_EXPANSION_TOTAL,
    PPR_DAMPING, PPR_BOOST_COEFF, PPR_MIN_SCORE, PPR_MAX_ITER, PPR_CONVERGENCE_TOL,
)
from memory.vectors import deserialize_f32, _dot, _novelty_score

# Cached Beta prior (recomputed when feedback count changes)
_beta_cache: dict = {"n_events": -1, "prior_mean": 0.25, "prior_strength": 1.0}

_BETA_FALLBACK_MEAN = 0.25
_BETA_FALLBACK_STRENGTH = 1.0
_BETA_MIN_MEMORIES = 5  # need at least this many memories with 2+ events for fitting


def _compute_beta_prior(db: sqlite3.Connection) -> tuple[float, float]:
    """Compute empirical Bayes Beta prior from all feedback data.

    Returns (prior_mean, prior_strength) where prior_strength = a + b.
    Uses method of moments on per-memory mean utilities.
    Caches result and recomputes when feedback event count changes.
    """
    n_events = db.execute(
        "SELECT COUNT(*) FROM memory_events WHERE event_type = 'feedback'"
    ).fetchone()[0]

    if n_events == _beta_cache["n_events"]:
        return _beta_cache["prior_mean"], _beta_cache["prior_strength"]

    if n_events == 0:
        _beta_cache.update(n_events=0, prior_mean=_BETA_FALLBACK_MEAN,
                           prior_strength=_BETA_FALLBACK_STRENGTH)
        return _BETA_FALLBACK_MEAN, _BETA_FALLBACK_STRENGTH

    # Per-memory aggregates (only memories with 2+ events for variance estimate)
    rows = db.execute("""
        SELECT memory_id,
               SUM(json_extract(context, '$.utility')) as usum,
               COUNT(*) as cnt
        FROM memory_events
        WHERE event_type = 'feedback'
        GROUP BY memory_id
        HAVING cnt >= 2
    """).fetchall()

    if len(rows) < _BETA_MIN_MEMORIES:
        # Not enough data — use global mean with fallback strength
        global_row = db.execute("""
            SELECT SUM(json_extract(context, '$.utility')),
                   COUNT(*)
            FROM memory_events WHERE event_type = 'feedback'
        """).fetchone()
        mu = global_row[0] / global_row[1] if global_row[1] else _BETA_FALLBACK_MEAN
        _beta_cache.update(n_events=n_events, prior_mean=mu,
                           prior_strength=_BETA_FALLBACK_STRENGTH)
        return mu, _BETA_FALLBACK_STRENGTH

    means = [r[1] / r[2] for r in rows]
    mu = sum(means) / len(means)
    var = sum((m - mu) ** 2 for m in means) / len(means)

    if var <= 0 or var >= mu * (1 - mu):
        # Variance too high for valid Beta — use empirical mean, fallback strength
        _beta_cache.update(n_events=n_events, prior_mean=mu,
                           prior_strength=_BETA_FALLBACK_STRENGTH)
        return mu, _BETA_FALLBACK_STRENGTH

    strength = mu * (1 - mu) / var - 1  # a + b
    _beta_cache.update(n_events=n_events, prior_mean=mu, prior_strength=strength)
    return mu, strength


def rrf_fuse(
    db: sqlite3.Connection,
    vec_ranked: dict[str, int],
    fts_ranked: dict[str, int],
    all_ids: set[str],
    boost_themes_list: list[str],
) -> tuple[dict[str, float], dict, dict]:
    """RRF fusion + feedback boost + theme boost.

    Returns (rrf_scores, feedback_map, themes_map).
    """
    # Batch-fetch feedback + themes for all candidates
    feedback_map = {}   # memory_id -> {"count": N, "ewma": F}
    themes_map = {}
    if all_ids:
        placeholders = ",".join("?" * len(all_ids))
        id_list = list(all_ids)

        # Feedback from event log (EWMA utility)
        fb_rows = db.execute(f"""
            SELECT memory_id, context FROM memory_events
            WHERE memory_id IN ({placeholders})
              AND event_type = 'feedback'
            ORDER BY created_at ASC
        """, id_list).fetchall()
        _EWMA_ALPHA = 0.3
        for r in fb_rows:
            mid_fb = r["memory_id"]
            try:
                ctx = json.loads(r["context"]) if r["context"] else {}
            except (json.JSONDecodeError, TypeError):
                ctx = {}
            utility = ctx.get("utility", 0.0)
            if mid_fb not in feedback_map:
                feedback_map[mid_fb] = {"count": 0, "ewma": utility}
            else:
                prev = feedback_map[mid_fb]["ewma"]
                feedback_map[mid_fb]["ewma"] = _EWMA_ALPHA * utility + (1 - _EWMA_ALPHA) * prev
            feedback_map[mid_fb]["count"] += 1

        # Themes
        theme_rows = db.execute(f"""
            SELECT id, themes FROM memories WHERE id IN ({placeholders})
        """, id_list).fetchall()
        themes_map = {r["id"]: r["themes"] for r in theme_rows}

    # Empirical Bayes prior from population feedback (Beta method of moments)
    prior_mean, prior_strength = _compute_beta_prior(db)

    rrf_scores = {}
    for mid in all_ids:
        score = 0.0
        if mid in vec_ranked:
            score += RRF_VEC_WEIGHT / (RRF_K + vec_ranked[mid] + 1)
        if mid in fts_ranked:
            score += (1.0 - RRF_VEC_WEIGHT) / (RRF_K + fts_ranked[mid] + 1)

        # Feedback boost (empirical Bayes — Beta prior, centered penalty)
        if mid in feedback_map:
            fb = feedback_map[mid]
            a = prior_mean * prior_strength
            b = (1 - prior_mean) * prior_strength
            # Blend EWMA with Beta prior — more observations = more weight on EWMA
            effective_n = fb["count"]
            posterior_mean = (fb["ewma"] * effective_n + a) / (effective_n + a + b)
            score *= (1 + (posterior_mean - prior_mean) * FEEDBACK_COEFF)

        # Theme boost (multiplicative)
        if boost_themes_list and mid in themes_map and themes_map[mid]:
            try:
                mem_themes = set(json.loads(themes_map[mid]))
                overlap = len(mem_themes & set(boost_themes_list))
                if overlap:
                    score *= (1 + THEME_BOOST * overlap)
            except (json.JSONDecodeError, TypeError):
                pass

        rrf_scores[mid] = score

    return rrf_scores, feedback_map, themes_map


def apply_hebbian(
    db: sqlite3.Connection,
    rrf_scores: dict[str, float],
) -> None:
    """Apply Hebbian co-retrieval boost (PMI-based). Modifies rrf_scores in-place."""
    if not rrf_scores:
        return

    seed_ids_hebb = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:5]
    # Build per-query memory sets from retrieval history for seeds + candidates
    all_candidate_ids = list(rrf_scores.keys())
    hebb_ph = ",".join("?" * len(all_candidate_ids))
    lookback = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    hebb_rows = db.execute(f"""
        SELECT query, memory_id FROM memory_events
        WHERE event_type = 'retrieved'
          AND memory_id IN ({hebb_ph})
          AND created_at > ?
    """, all_candidate_ids + [lookback]).fetchall()

    # query -> set of memory_ids
    hebb_query_mems = {}
    hebb_mem_freq = {}  # memory_id -> set of unique queries
    for r in hebb_rows:
        q, mid = r["query"], r["memory_id"]
        if q not in hebb_query_mems:
            hebb_query_mems[q] = set()
        hebb_query_mems[q].add(mid)
        if mid not in hebb_mem_freq:
            hebb_mem_freq[mid] = set()
        hebb_mem_freq[mid].add(q)

    hebb_total_queries = len(hebb_query_mems)

    if hebb_total_queries >= 5:  # need enough queries for PMI to be meaningful
        # Convert freq sets to counts
        hebb_mem_count = {mid: len(qs) for mid, qs in hebb_mem_freq.items()}

        for candidate in all_candidate_ids:
            if candidate in seed_ids_hebb:
                continue
            total_boost = 0.0
            for seed in seed_ids_hebb:
                if seed not in hebb_mem_count or candidate not in hebb_mem_count:
                    continue
                # Count joint queries
                seed_qs = hebb_mem_freq.get(seed, set())
                cand_qs = hebb_mem_freq.get(candidate, set())
                joint = len(seed_qs & cand_qs)
                if joint < HEBBIAN_MIN_JOINT:
                    continue
                # PMI = log2(P(A,B) / (P(A) * P(B)))
                p_seed = hebb_mem_count[seed] / hebb_total_queries
                p_cand = hebb_mem_count[candidate] / hebb_total_queries
                p_joint = joint / hebb_total_queries
                pmi = math.log2(p_joint / (p_seed * p_cand))
                if pmi > 0:
                    total_boost += min(HEBBIAN_COEFF * pmi, HEBBIAN_CAP)
            rrf_scores[candidate] += min(total_boost, HEBBIAN_CAP)


def personalized_pagerank(
    adj: dict[str, list[tuple[str, float]]],
    seeds: dict[str, float],
    damping: float = PPR_DAMPING,
    max_iter: int = PPR_MAX_ITER,
    tol: float = PPR_CONVERGENCE_TOL,
) -> dict[str, float]:
    """Power-iteration PPR over a weighted adjacency list.

    At each step, for each node with score > 0, distribute
    damping * score * edge_weight / weighted_out_degree to neighbors,
    then add (1 - damping) * normalized_seed_weight.
    Returns all nodes with score > 0.
    """
    if not seeds or not adj:
        return {}

    # Normalize seed weights to sum to 1
    total_seed = sum(seeds.values())
    if total_seed <= 0:
        return {}
    norm_seeds = {k: v / total_seed for k, v in seeds.items()}

    # Precompute weighted out-degree per node
    out_degree: dict[str, float] = {}
    for node, neighbors in adj.items():
        out_degree[node] = sum(w for _, w in neighbors)

    # Initialize scores from seeds
    scores = dict(norm_seeds)

    for _ in range(max_iter):
        new_scores: dict[str, float] = {}

        # Distribute scores along edges
        for node, score in scores.items():
            if node not in adj:
                continue
            deg = out_degree.get(node, 0.0)
            if deg <= 0:
                continue
            for neighbor, weight in adj[node]:
                contrib = damping * score * weight / deg
                new_scores[neighbor] = new_scores.get(neighbor, 0.0) + contrib

        # Add teleport (restart) component
        for node, seed_w in norm_seeds.items():
            new_scores[node] = new_scores.get(node, 0.0) + (1 - damping) * seed_w

        # Check convergence (L1 delta)
        delta = 0.0
        all_keys = set(scores) | set(new_scores)
        for k in all_keys:
            delta += abs(new_scores.get(k, 0.0) - scores.get(k, 0.0))

        scores = new_scores
        if delta < tol:
            break

    return {k: v for k, v in scores.items() if v > 0}


def expand_via_ppr(
    db: sqlite3.Connection,
    rrf_scores: dict[str, float],
    query_embedding: list[float],
    query: str,
    exclude_set: set[str],
) -> tuple[int, int, int, bool]:
    """PPR-based graph expansion. Modifies rrf_scores in-place.

    Returns (expansion_new, expansion_boosted, 0, False) for compatibility.
    """
    pre_expansion_ids = set(rrf_scores.keys())

    expansion_new = 0
    expansion_boosted = 0
    if not rrf_scores:
        return 0, 0, 0, False

    # 1. Seeds = top ADJACENCY_SEED_COUNT by RRF score
    seed_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:ADJACENCY_SEED_COUNT]
    if not seed_ids:
        return 0, 0, 0, False
    seed_set = set(seed_ids)

    # 2. Query 1-hop edges from seeds
    ph = ",".join("?" * len(seed_ids))
    hop1_rows = db.execute(f"""
        SELECT source_id, target_id, weight
        FROM memory_edges
        WHERE source_id IN ({ph}) OR target_id IN ({ph})
    """, seed_ids + seed_ids).fetchall()

    # Collect hop-1 neighbor IDs
    hop1_neighbors = set()
    for edge in hop1_rows:
        hop1_neighbors.add(edge["source_id"])
        hop1_neighbors.add(edge["target_id"])
    hop1_neighbors -= seed_set

    # 3. Query edges among hop-1 neighbors (2-hop subgraph)
    hop2_rows = []
    if hop1_neighbors:
        n_list = list(hop1_neighbors)
        n_ph = ",".join("?" * len(n_list))
        hop2_rows = db.execute(f"""
            SELECT source_id, target_id, weight
            FROM memory_edges
            WHERE source_id IN ({n_ph}) OR target_id IN ({n_ph})
        """, n_list + n_list).fetchall()

    # 4. Build adjacency dict (both directions per edge)
    adj: dict[str, list[tuple[str, float]]] = {}
    for edge in list(hop1_rows) + list(hop2_rows):
        src, tgt = edge["source_id"], edge["target_id"]
        w = edge["weight"] if edge["weight"] is not None else 1.0
        adj.setdefault(src, []).append((tgt, w))
        adj.setdefault(tgt, []).append((src, w))

    # 5. Seed weights from RRF scores
    seed_weights = {sid: rrf_scores[sid] for sid in seed_ids}

    # 6. Run PPR
    ppr_scores = personalized_pagerank(adj, seed_weights)

    # 7. Apply PPR scores to rrf_scores
    total_new = 0
    for mid, ps in ppr_scores.items():
        if mid in seed_set or mid in exclude_set:
            continue
        if ps < PPR_MIN_SCORE:
            continue
        is_new = mid not in rrf_scores
        if is_new and total_new >= MAX_EXPANSION_TOTAL:
            continue
        if is_new:
            rrf_scores[mid] = PPR_BOOST_COEFF * ps
            total_new += 1
        else:
            rrf_scores[mid] += PPR_BOOST_COEFF * ps
            expansion_boosted += 1
    expansion_new = total_new

    # 8. Filter non-active expansion neighbors
    new_neighbor_ids = [mid for mid in rrf_scores if mid not in pre_expansion_ids]
    if new_neighbor_ids:
        n_ph = ",".join("?" * len(new_neighbor_ids))
        active_rows = db.execute(
            f"SELECT id FROM memories WHERE id IN ({n_ph}) AND status = 'active'",
            new_neighbor_ids,
        ).fetchall()
        active_neighbor_ids = {row["id"] for row in active_rows}
        for mid in new_neighbor_ids:
            if mid not in active_neighbor_ids:
                del rrf_scores[mid]

    return expansion_new, expansion_boosted, 0, False


def _expand_adjacency_legacy(
    db: sqlite3.Connection,
    rrf_scores: dict[str, float],
    query_embedding: list[float],
    query: str,
    exclude_set: set[str],
) -> tuple[int, int, int, bool]:
    """Novelty-scored BFS adjacency expansion (superseded by PPR in wm19).

    Preserved for research comparison and tuning tool compatibility.
    Returns (expansion_new, expansion_boosted, per_seed_cap_hits, total_cap_hit).
    """
    pre_expansion_ids = set(rrf_scores.keys())

    expansion_new = 0
    expansion_boosted = 0
    per_seed_cap_hits = 0
    total_cap_hit = False
    if not rrf_scores:
        return expansion_new, expansion_boosted, per_seed_cap_hits, total_cap_hit

    max_rrf = max(rrf_scores.values())
    seed_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:ADJACENCY_SEED_COUNT]

    if not seed_ids:
        return expansion_new, expansion_boosted, per_seed_cap_hits, total_cap_hit

    # Fetch seed embeddings
    seed_embeddings = {}  # memory_id -> list[float]
    for sid in seed_ids:
        row = db.execute(
            "SELECT rm.rowid FROM memory_rowid_map rm WHERE rm.memory_id = ?", (sid,)
        ).fetchone()
        if row:
            vec_row = db.execute(
                "SELECT embedding FROM memory_vec WHERE rowid = ?", (row["rowid"],)
            ).fetchone()
            if vec_row:
                seed_embeddings[sid] = deserialize_f32(vec_row["embedding"])

    # Fetch all edges from seeds (no edge_type filter)
    ph = ",".join("?" * len(seed_ids))
    edge_rows = db.execute(f"""
        SELECT source_id, target_id, flags, linking_context, linking_embedding,
               edge_type, weight
        FROM memory_edges
        WHERE source_id IN ({ph}) OR target_id IN ({ph})
    """, seed_ids + seed_ids).fetchall()

    seed_set = set(seed_ids)
    neighbors_per_seed = {}  # seed_id -> set of neighbor_ids
    total_new = 0

    for edge in edge_rows:
        src, tgt = edge["source_id"], edge["target_id"]
        if src in seed_set:
            seed_id, neighbor_id = src, tgt
        elif tgt in seed_set:
            seed_id, neighbor_id = tgt, src
        else:
            continue
        if neighbor_id in exclude_set or neighbor_id in seed_set:
            continue

        # Cap per-seed (dedup) and total
        seed_neighbors = neighbors_per_seed.get(seed_id, set())
        if len(seed_neighbors) >= MAX_NEIGHBORS_PER_SEED:
            per_seed_cap_hits += 1
            continue
        if neighbor_id in seed_neighbors:
            continue
        is_new = neighbor_id not in rrf_scores
        if is_new and total_new >= MAX_EXPANSION_TOTAL:
            total_cap_hit = True
            continue

        # Context relevance gate — skip edges with context but no embedding
        context_weight = 1.0
        has_context = edge["linking_context"] and edge["linking_context"].strip()
        if has_context and not edge["linking_embedding"]:
            continue  # embedding not yet computed — skip
        if edge["linking_embedding"]:
            link_emb = deserialize_f32(edge["linking_embedding"])
            context_sim = _dot(query_embedding, link_emb)
            if context_sim < CONTEXT_RELEVANCE_THRESHOLD:
                continue
            context_weight = context_sim

        # Novelty scoring
        seed_vec = seed_embeddings.get(seed_id)
        if not seed_vec:
            continue

        n_rowid = db.execute(
            "SELECT rowid FROM memory_rowid_map WHERE memory_id = ?",
            (neighbor_id,)
        ).fetchone()
        if not n_rowid:
            continue
        n_vec_row = db.execute(
            "SELECT embedding FROM memory_vec WHERE rowid = ?",
            (n_rowid["rowid"],)
        ).fetchone()
        if not n_vec_row:
            continue
        neighbor_vec = deserialize_f32(n_vec_row["embedding"])

        novelty = _novelty_score(query_embedding, seed_vec, neighbor_vec)
        if novelty < ADJACENCY_NOVELTY_FLOOR:
            continue

        seed_strength = max(0.0, rrf_scores[seed_id] / max_rrf) if max_rrf > 0 else 0
        edge_weight = edge["weight"] if edge["weight"] is not None else 1.0
        boost = ADJACENCY_BASE_BOOST * novelty * context_weight * seed_strength * edge_weight

        if is_new:
            rrf_scores[neighbor_id] = boost
            total_new += 1
        else:
            rrf_scores[neighbor_id] += boost
            expansion_boosted += 1
        neighbors_per_seed.setdefault(seed_id, set()).add(neighbor_id)
    expansion_new = total_new

    # Filter non-active expansion neighbors
    new_neighbor_ids = [mid for mid in rrf_scores if mid not in pre_expansion_ids]
    if new_neighbor_ids:
        n_ph = ",".join("?" * len(new_neighbor_ids))
        active_rows = db.execute(
            f"SELECT id FROM memories WHERE id IN ({n_ph}) AND status = 'active'",
            new_neighbor_ids,
        ).fetchall()
        active_neighbor_ids = {row["id"] for row in active_rows}
        for mid in new_neighbor_ids:
            if mid not in active_neighbor_ids:
                del rrf_scores[mid]

    return expansion_new, expansion_boosted, per_seed_cap_hits, total_cap_hit


def _fit_log_curve(ranks: list[int], scores: list[float]) -> tuple[float, float]:
    """Fit s = a - b·ln(rank) via OLS. Returns (a, b)."""
    n = len(ranks)
    lx = [math.log(r) for r in ranks]
    mx = sum(lx) / n
    my = sum(scores) / n
    ssxx = sum((x - mx) ** 2 for x in lx)
    if ssxx < 1e-12:
        return my, 0.0
    ssxy = sum((x - mx) * (y - my) for x, y in zip(lx, scores))
    slope = ssxy / ssxx
    return my - slope * mx, -slope


def apply_quality_floor(
    rrf_scores: dict[str, float],
) -> tuple[list[str], list[str], float, float]:
    """Adaptive cliff detection with safety floor.

    Fits a log curve (s = a - b·ln(rank)) to the score sequence using a
    rolling window. At each position i >= CLIFF_MIN_RESULTS, fits to
    positions [1..i], predicts i+1, and cuts when the actual score falls
    more than CLIFF_Z_THRESHOLD standard deviations below the prediction.

    Returns (sorted_ids, dropped_ids, top_score, floor_val).
    """
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    if not sorted_ids:
        return sorted_ids, [], 0.0, 0.0

    top_score = rrf_scores[sorted_ids[0]]
    if top_score <= 0:
        return sorted_ids, [], top_score, 0.0

    safety_floor = 0
    above_floor = [mid for mid in sorted_ids if rrf_scores[mid] >= safety_floor]
    dropped_ids = [mid for mid in sorted_ids if rrf_scores[mid] < safety_floor]

    if len(above_floor) <= CLIFF_MIN_RESULTS:
        return above_floor, dropped_ids, top_score, safety_floor

    # Normalize scores for curve fitting
    norm_scores = [rrf_scores[mid] / top_score for mid in above_floor]

    # Rolling log-curve cliff detection
    cliff_pos = len(above_floor)  # default: keep all above floor
    for i in range(CLIFF_MIN_RESULTS, len(norm_scores)):
        fit_ranks = list(range(1, i + 1))
        fit_scores = norm_scores[:i]
        a, b = _fit_log_curve(fit_ranks, fit_scores)

        # RMSE on training data
        preds = [a - b * math.log(r) for r in fit_ranks]
        residuals = [fit_scores[j] - preds[j] for j in range(len(fit_scores))]
        rmse = math.sqrt(sum(r * r for r in residuals) / len(residuals))
        rmse = max(rmse, 0.005)  # floor to avoid false positives on perfect fits

        # Predict next position
        predicted = a - b * math.log(i + 1)
        actual = norm_scores[i]
        deviation = predicted - actual  # positive = dropped more than expected

        if deviation > CLIFF_Z_THRESHOLD * rmse:
            cliff_pos = i
            break

    kept_ids = above_floor[:cliff_pos]
    cliff_dropped = above_floor[cliff_pos:]
    dropped_ids = cliff_dropped + dropped_ids

    # floor_val reflects the effective cutoff (cliff or safety, whichever was higher)
    if kept_ids:
        floor_val = rrf_scores[kept_ids[-1]]
    else:
        floor_val = safety_floor

    return kept_ids, dropped_ids, top_score, floor_val
