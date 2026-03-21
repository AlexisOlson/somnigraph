"""Reranker — learned scoring for memory retrieval.

Loads a trained LightGBM model and precomputes per-memory metadata.
Provides rerank() as a drop-in replacement for the RRF+UCB+Hebbian+PPR pipeline.
Falls back to None if model not found (caller should use legacy pipeline).

26 features (0-17 base, 18-25 extended):
  0-6:   retrieval signals (fts_rank, vec_rank, theme_rank, ppr_score, fts_bm25, vec_dist, theme_overlap)
  7-9:   feedback signals (fb_last, fb_mean, fb_count)
  10:    graph signal (hebbian_pmi)
  11-17: memory metadata (category, priority, age_days, token_count, edge_count, theme_count, confidence)
  18-20: query-content features (query_coverage, proximity, query_idf_var)
  21:    temporal (burstiness)
  22-23: graph structure (betweenness, diversity_score)
  24:    feedback temporal (fb_time_weighted)
  25:    session (session_recency)
"""

import json
import logging
import math
import pickle
import struct
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from memory.constants import MODEL_PATH, PPR_DAMPING

logger = logging.getLogger("claude-memory")

# Category encoding (must match train_reranker.py)
CATEGORY_MAP = {
    "episodic": 0, "semantic": 1, "procedural": 2,
    "reflection": 3, "entity": 4, "meta": 5,
}

# Feature names (order must match train_reranker.py exactly)
FEATURE_NAMES = [
    # Retrieval signals (query-dependent)
    "fts_rank", "vec_rank", "theme_rank", "ppr_score",
    "fts_bm25", "vec_dist", "theme_overlap",
    # Feedback signals
    "fb_last", "fb_mean", "fb_count",
    # Graph signals
    "hebbian_pmi",
    # Memory metadata
    "category", "priority", "age_days", "token_count",
    "edge_count", "theme_count", "confidence",
    # Extended features (Tier 1)
    "query_coverage", "proximity", "query_idf_var", "burstiness",
    # Extended features (Tier 2)
    "betweenness", "diversity_score", "fb_time_weighted", "session_recency",
]

# Cache for model + metadata
_cache = {
    "model": None,
    "feature_names": None,
    "memory_meta": None,
    "meta_mem_count": -1,  # refresh when active memory count changes
    "loaded": False,
    "failed": False,
}


def _load_model():
    """Load model from pkl. Returns Booster or None.

    Extracts the underlying LightGBM Booster from the sklearn wrapper
    so we can predict without sklearn dependency.
    """
    if _cache["failed"]:
        return None
    if _cache["model"] is not None:
        return _cache["model"]

    if not MODEL_PATH.exists():
        logger.info("No reranker model at %s — using formula scoring", MODEL_PATH)
        _cache["failed"] = True
        return None

    try:
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        sklearn_model = data["model"]
        # Extract the underlying Booster for dependency-free prediction
        booster = sklearn_model.booster_
        _cache["model"] = booster
        _cache["feature_names"] = data.get("feature_names", [])
        _cache["loaded"] = True
        logger.info("Reranker model loaded from %s (%d features)",
                     MODEL_PATH, len(_cache["feature_names"]))
        return booster
    except Exception as e:
        logger.warning("Failed to load reranker model: %s", e)
        _cache["failed"] = True
        return None


def _load_memory_meta(db):
    """Precompute per-memory metadata. Cached until memory count changes."""
    count = db.execute(
        "SELECT COUNT(*) FROM memories WHERE status = 'active'"
    ).fetchone()[0]

    if count == _cache["meta_mem_count"] and _cache["memory_meta"] is not None:
        return _cache["memory_meta"]

    now_dt = datetime.now(timezone.utc)
    rows = db.execute("""
        SELECT id, category, base_priority, created_at, token_count,
               confidence, themes, content, summary
        FROM memories WHERE status = 'active'
    """).fetchall()

    meta = {}
    for r in rows:
        mid = r["id"]
        try:
            created = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
            age_days = (now_dt - created).total_seconds() / 86400
        except Exception:
            age_days = 0.0

        theme_count = 0
        if r["themes"]:
            try:
                theme_count = len(json.loads(r["themes"]))
            except Exception:
                pass

        content_text = ""
        if r["summary"]:
            content_text = r["summary"]
        if r["content"]:
            content_text = content_text + " " + r["content"] if content_text else r["content"]

        meta[mid] = {
            "category": CATEGORY_MAP.get(r["category"], 1),
            "priority": r["base_priority"] or 5,
            "age_days": age_days,
            "token_count": r["token_count"] or 200,
            "confidence": r["confidence"] if r["confidence"] is not None else 0.5,
            "theme_count": theme_count,
            "content_text": content_text.lower(),
            "content_tokens": content_text.lower().split(),
        }

    # Edge counts + adjacency
    edge_rows = db.execute("SELECT source_id, target_id FROM memory_edges").fetchall()
    edge_counts = defaultdict(int)
    adjacency = defaultdict(set)
    for r in edge_rows:
        s, t = r["source_id"], r["target_id"]
        edge_counts[s] += 1
        edge_counts[t] += 1
        if s in meta and t in meta:
            adjacency[s].add(t)
            adjacency[t].add(s)
    for mid in meta:
        meta[mid]["edge_count"] = edge_counts.get(mid, 0)

    # Burstiness
    event_rows = db.execute("""
        SELECT memory_id, created_at FROM memory_events
        WHERE event_type IN ('retrieved', 'feedback')
        ORDER BY memory_id, created_at
    """).fetchall()
    events_by_mem = defaultdict(list)
    for r in event_rows:
        try:
            t = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
            events_by_mem[r["memory_id"]].append(t.timestamp())
        except Exception:
            pass

    for mid in meta:
        timestamps = events_by_mem.get(mid, [])
        if len(timestamps) < 3:
            meta[mid]["burstiness"] = 0.0
            continue
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        mu = sum(intervals) / len(intervals)
        if mu <= 0:
            meta[mid]["burstiness"] = 0.0
            continue
        var = sum((x - mu)**2 for x in intervals) / len(intervals)
        meta[mid]["burstiness"] = (var / (mu * mu)) - 1.0

    # Feedback timestamps
    fb_ts_rows = db.execute("""
        SELECT memory_id, context, created_at FROM memory_events
        WHERE event_type = 'feedback' AND context IS NOT NULL
        ORDER BY memory_id, created_at
    """).fetchall()
    fb_timestamps = defaultdict(list)
    for r in fb_ts_rows:
        try:
            ctx = json.loads(r["context"])
            if "utility" not in ctx:
                continue
            t = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
            fb_timestamps[r["memory_id"]].append((t.timestamp(), ctx["utility"]))
        except Exception:
            pass
    for mid in meta:
        meta[mid]["fb_timestamps"] = fb_timestamps.get(mid, [])

    # Betweenness centrality (Brandes)
    active_ids = list(meta.keys())
    betweenness = {mid: 0.0 for mid in active_ids}
    if adjacency:
        for source in active_ids:
            if source not in adjacency:
                continue
            dist = {source: 0}
            pred = defaultdict(list)
            sigma = defaultdict(float)
            sigma[source] = 1.0
            queue = [source]
            order = []
            head = 0
            while head < len(queue):
                v = queue[head]
                head += 1
                order.append(v)
                for w in adjacency.get(v, set()):
                    if w not in dist:
                        dist[w] = dist[v] + 1
                        queue.append(w)
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)
            delta = defaultdict(float)
            for w in reversed(order):
                if w == source:
                    continue
                for v in pred[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                betweenness[w] += delta[w]

        n = len(active_ids)
        if n > 2:
            norm = (n - 1) * (n - 2) / 2.0
            for mid in betweenness:
                betweenness[mid] /= norm

    for mid in meta:
        meta[mid]["betweenness"] = betweenness.get(mid, 0.0)

    # Diversity score
    mids_with_neighbors = set()
    for mid, neighbors in adjacency.items():
        mids_with_neighbors.add(mid)
        mids_with_neighbors.update(neighbors)

    embeddings = {}
    if mids_with_neighbors:
        emb_rows = db.execute("""
            SELECT m.id, v.embedding
            FROM memories m
            JOIN memory_rowid_map rm ON rm.memory_id = m.id
            JOIN memory_vec v ON v.rowid = rm.rowid
            WHERE m.status = 'active'
        """).fetchall()
        for r in emb_rows:
            if r["id"] in mids_with_neighbors:
                blob = r["embedding"]
                dim = len(blob) // 4
                embeddings[r["id"]] = struct.unpack(f"{dim}f", blob)

    for mid in meta:
        neighbors = adjacency.get(mid, set())
        if not neighbors or mid not in embeddings:
            meta[mid]["diversity_score"] = 0.5
            continue
        emb_a = embeddings[mid]
        sims = []
        for nid in neighbors:
            if nid in embeddings:
                emb_b = embeddings[nid]
                dot = sum(a * b for a, b in zip(emb_a, emb_b))
                sims.append(dot)
        meta[mid]["diversity_score"] = 1.0 - (sum(sims) / len(sims)) if sims else 0.5

    # IDF stats
    total_docs = len(meta)
    term_doc_freq = defaultdict(int)
    for mid, m in meta.items():
        for term in set(m["content_tokens"]):
            term_doc_freq[term] += 1
    meta["__idf_stats__"] = {"total_docs": total_docs, "term_doc_freq": dict(term_doc_freq)}

    # Session retrievals
    session_rows = db.execute("""
        SELECT memory_id, session_id, query, created_at FROM memory_events
        WHERE event_type = 'retrieved' AND session_id IS NOT NULL
        ORDER BY created_at
    """).fetchall()
    session_retrievals = defaultdict(list)
    for r in session_rows:
        session_retrievals[r["session_id"]].append((r["query"], r["memory_id"]))
    meta["__session_retrievals__"] = dict(session_retrievals)

    _cache["memory_meta"] = meta
    _cache["meta_mem_count"] = count
    return meta


def _compute_proximity(query_terms, content_tokens):
    """Inverse min-span window of query terms in content."""
    if not content_tokens or not query_terms:
        return 0.0
    term_positions = defaultdict(list)
    query_set = set(query_terms)
    for pos, token in enumerate(content_tokens):
        if token in query_set:
            term_positions[token].append(pos)
    if len(term_positions) < len(query_set):
        return 0.0
    events = []
    for term, positions in term_positions.items():
        for pos in positions:
            events.append((pos, term))
    events.sort()
    n_terms = len(query_set)
    term_count = defaultdict(int)
    unique_count = 0
    min_span = len(content_tokens)
    left = 0
    for right in range(len(events)):
        pos_r, term_r = events[right]
        if term_count[term_r] == 0:
            unique_count += 1
        term_count[term_r] += 1
        while unique_count == n_terms:
            pos_l, term_l = events[left]
            span = pos_r - pos_l + 1
            if span < min_span:
                min_span = span
            term_count[term_l] -= 1
            if term_count[term_l] == 0:
                unique_count -= 1
            left += 1
    return n_terms / min_span


def rerank(
    db,
    query: str,
    fts_ranked: dict[str, int],
    vec_ranked: dict[str, int],
    fts_scores: dict[str, float],
    vec_distances: dict[str, float],
    theme_ranked: dict[str, int],
    theme_overlap_map: dict[str, int],
    feedback_raw: dict,
    hebb_data: dict,
    ppr_cache: dict,
) -> tuple[list[str], dict[str, float]] | None:
    """Score candidates using learned reranker.

    Returns (sorted_ids, scores_dict) or None if reranker unavailable
    (caller should fall back to legacy pipeline).
    """
    model = _load_model()
    if model is None:
        return None

    memory_meta = _load_memory_meta(db)

    # Build candidate pool
    candidate_ids = set(fts_ranked) | set(vec_ranked) | set(theme_ranked)

    # PPR scores
    ppr_scores = {}
    if ppr_cache:
        d_key = round(PPR_DAMPING, 3)
        ppr_scores = ppr_cache.get((d_key, query), {})
        candidate_ids |= set(ppr_scores.keys())

    if not candidate_ids:
        return []

    # Hebbian PMI
    hebbian_pmi_map = {}
    if hebb_data and hebb_data.get("total_queries", 0) >= 5:
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

    # Query-level features
    query_terms = [t for t in query.lower().split() if len(t) > 1]

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
        query_positions = [i for i, (q, _) in enumerate(events) if q == query]
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

    # Build feature matrix (26 features, matching FEATURE_NAMES order)
    candidate_list = sorted(candidate_ids)
    n = len(candidate_list)
    # Use list-of-lists to avoid numpy dependency at prediction time
    features = [[0.0] * 26 for _ in range(n)]

    for i, mid in enumerate(candidate_list):
        f = features[i]
        f[0] = fts_ranked.get(mid, -1)                     # fts_rank
        f[1] = vec_ranked.get(mid, -1)                      # vec_rank
        f[2] = theme_ranked.get(mid, -1)                    # theme_rank
        f[3] = ppr_scores.get(mid, 0.0)                     # ppr_score
        f[4] = fts_scores.get(mid, 0.0)                     # fts_bm25
        f[5] = vec_distances.get(mid, 0.0)                  # vec_dist
        f[6] = theme_overlap_map.get(mid, 0)                # theme_overlap

        # Feedback
        fb = feedback_raw.get(mid)
        if fb and fb["count"] > 0:
            f[7] = fb["utilities"][-1]                       # fb_last
            f[8] = sum(fb["utilities"]) / fb["count"]        # fb_mean
            f[9] = fb["count"]                               # fb_count
        else:
            f[7] = -1.0
            f[8] = -1.0
            f[9] = 0

        f[10] = hebbian_pmi_map.get(mid, 0.0)              # hebbian_pmi

        m = memory_meta.get(mid)
        if m:
            f[11] = m["category"]                            # category
            f[12] = m["priority"]                            # priority
            f[13] = m["age_days"]                            # age_days
            f[14] = m["token_count"]                         # token_count
            f[15] = m["edge_count"]                          # edge_count
            f[16] = m["theme_count"]                         # theme_count
            f[17] = m["confidence"]                          # confidence
        else:
            f[11] = 1; f[12] = 5; f[13] = 0.0; f[14] = 200
            f[15] = 0; f[16] = 0; f[17] = 0.5

        # Tier 1
        if m and query_terms:
            content_set = set(m.get("content_tokens", []))
            matched = sum(1 for t in query_terms if t in content_set)
            f[18] = matched / len(query_terms)               # query_coverage
        else:
            f[18] = 0.0

        if m and len(query_terms) > 1:
            f[19] = _compute_proximity(query_terms, m.get("content_tokens", []))
        else:
            f[19] = 0.0

        f[20] = query_idf_var                                # query_idf_var

        if m:
            f[21] = m.get("burstiness", 0.0)                # burstiness
        else:
            f[21] = 0.0

        # Tier 2
        if m:
            f[22] = m.get("betweenness", 0.0)               # betweenness
            f[23] = m.get("diversity_score", 0.5)            # diversity_score
        else:
            f[22] = 0.0; f[23] = 0.5

        if m:
            fb_ts = m.get("fb_timestamps", [])
            if fb_ts:
                weighted_sum = 0.0
                weight_sum = 0.0
                for ts, util in fb_ts:
                    age_fb = (now_ts - ts) / 86400.0
                    w = math.exp(-fb_lambda * age_fb)
                    weighted_sum += w * util
                    weight_sum += w
                f[24] = weighted_sum / weight_sum if weight_sum > 0 else -1.0
            else:
                f[24] = -1.0
        else:
            f[24] = -1.0

        f[25] = session_recency_map.get(mid, -1)            # session_recency

    # Predict using Booster (list-of-lists input, no numpy needed)
    preds = model.predict(features)

    # Sort by predicted score descending
    scored = sorted(zip(candidate_list, preds), key=lambda x: -x[1])
    scores_dict = {mid: float(s) for mid, s in scored}
    return [mid for mid, _ in scored], scores_dict


def invalidate_cache():
    """Force metadata refresh on next recall (call after remember/forget)."""
    _cache["meta_mem_count"] = -1
    _cache["memory_meta"] = None
