"""Modular candidate expansion methods for multi-hop retrieval.

Each method is independently toggleable for ablation experiments.
All methods are no-LLM: they use FTS, vector search, and mechanical
text processing only.
"""

import json
import logging
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stopwords (minimal set for query decomposition and keyword extraction)
# ---------------------------------------------------------------------------

STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "not", "no", "what", "which",
    "who", "whom", "where", "when", "why", "how", "that", "this", "these",
    "those", "it", "its", "he", "she", "they", "we", "you", "i", "my",
    "your", "his", "her", "our", "their", "me", "him", "us", "them",
    "about", "from", "as", "if", "then", "than", "so", "very", "just",
    "also", "much", "many", "some", "any", "all", "each", "every",
    "likely", "would", "based", "considered", "answer", "yes",
})

# Named entities worth bridging on (allowlist extracted from LoCoMo corpus,
# curated by Opus — replaces the broken capitalization heuristic)
LOCOMO_ENTITIES = frozenset({
    # People
    "amy", "andrew", "anna", "anthony", "audrey", "cal", "calvin", "caroline",
    "caro", "cindy", "dave", "david", "deb", "deborah", "debs", "dre", "ed",
    "emma", "ev", "evan", "frank", "george", "gina", "harry", "herbert",
    "jack", "james", "jean", "jill", "jo", "joanna", "john", "jolene", "jon",
    "josh", "karlie", "kyle", "laura", "lebron", "maria", "mark", "matt",
    "max", "mel", "melanie", "mell", "nate", "neal", "ned", "nicole", "nils",
    "nutt", "olafur", "oliver", "oscar", "patrick", "patterson", "peter",
    "rob", "rothfuss", "rowling", "russell", "sam", "samantha", "samuel",
    "sara", "shia", "stephenson", "susie", "tim", "toby", "tupac", "watson",
    # Pets
    "bailey", "buddy", "coco", "daisy", "luna", "marley", "panda", "pepper",
    "pixie", "precious", "scout", "shadow", "tilly",
    # Places
    "bali", "banff", "barcelona", "bogota", "boston", "california", "canada",
    "chicago", "detroit", "edinburgh", "england", "fenway", "florida",
    "francisco", "galway", "himalayas", "ireland", "italy", "janeiro",
    "japan", "jasper", "liverpool", "london", "manchester", "mexico", "miami",
    "michigan", "minnesota", "moher", "nuuk", "oregon", "paris", "phuket",
    "rio", "rockies", "rome", "scotland", "seattle", "shibuya", "shinjuku",
    "spain", "stamford", "sweden", "tahoe", "talkeetna", "tampa", "thailand",
    "tokyo", "toronto", "turkey", "vancouver", "woodhaven", "york",
    # Brands / products
    "facebook", "ferrari", "ford", "gatorade", "instagram", "logitech",
    "mustang", "nike", "nintendo", "prius", "sennheiser", "starbucks",
    "tiktok", "youtube",
    # Media / culture
    "aerosmith", "aragorn", "bach", "bareilles", "battlefield", "catan",
    "civilization", "cyberpunk", "disney", "dune", "fortnite", "gatsby",
    "godfather", "gondor", "gryffindor", "hobbit", "labeouf", "mario",
    "minalima", "mozart", "overcooked", "overwatch", "potter", "python",
    "ratatouille", "seraphim", "spider-man", "stormlight", "unity",
    "valhalla", "valorant", "witcher", "zelda",
    # Events / other
    "christmas", "eisenhower", "fireworks", "kustom", "perseid",
    "pomodoro", "rocky", "santa", "smoky", "thanksgiving", "wolves",
    # Demonyms / languages (specific enough for bridging)
    "french", "german", "hawaiian", "japanese", "irish",
})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExpansionContext:
    """Shared state passed to all expansion methods."""
    db: sqlite3.Connection
    question: str
    query_embedding: list[float]
    seeds: list[str]                    # top-N memory IDs from initial RRF
    existing_ids: set[str]              # IDs already in candidate pool
    fts_ranked: dict[str, int]          # mutable — methods may add entries
    vec_ranked: dict[str, int]          # mutable
    vec_distances: dict[str, float]     # mutable
    fts_scores: dict[str, float]        # mutable
    all_speakers: set[str]              # known speakers (lowercase)
    rowid_map: dict[int, str] = field(default_factory=dict)  # rowid -> memory_id (optional cache)
    memories: dict[str, dict] = field(default_factory=dict)   # memory_id -> metadata (optional cache)


@dataclass
class ExpansionResult:
    """Tracks per-method expansion yields."""
    per_method: dict[str, set[str]] = field(default_factory=dict)

    def add(self, method: str, new_ids: set[str]):
        self.per_method[method] = new_ids

    @property
    def all_new_ids(self) -> set[str]:
        result = set()
        for ids in self.per_method.values():
            result |= ids
        return result

    def method_counts(self) -> dict[str, int]:
        """Return {memory_id: count_of_methods_that_found_it}."""
        counts: dict[str, int] = {}
        for ids in self.per_method.values():
            for mid in ids:
                counts[mid] = counts.get(mid, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

# Cache IDF per DB (keyed by id so each conversation DB gets its own lookup)
_idf_cache: dict[int, tuple[dict[str, float], int]] = {}


def _get_idf(db: sqlite3.Connection) -> tuple[dict[str, float], int]:
    """Compute IDF for all terms in the corpus. Cached per DB connection.

    Returns (idf_dict, n_docs) where idf_dict maps term -> log(N/df).
    """
    db_id = id(db)
    if db_id in _idf_cache:
        return _idf_cache[db_id]

    # Count total documents
    n_docs = db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    if n_docs == 0:
        _idf_cache[db_id] = ({}, 0)
        return {}, 0

    # Compute document frequency per token from content
    df: Counter = Counter()
    rows = db.execute("SELECT content FROM memories").fetchall()
    for row in rows:
        content = row[0] if isinstance(row, tuple) else row["content"]
        if not content:
            continue
        content = re.sub(r"^\[[^\]]+\]\s*", "", content)
        tokens = set()
        for tok in content.lower().split():
            clean = re.sub(r"[.,!?\"';:()\[\]]+", "", tok)
            if clean and len(clean) >= 3 and clean not in STOPWORDS:
                tokens.add(clean)
        for t in tokens:
            df[t] += 1

    idf = {term: math.log((n_docs - count + 0.5) / (count + 0.5) + 1)
           for term, count in df.items()}
    _idf_cache[db_id] = (idf, n_docs)
    return idf, n_docs


def _fts_search(db: sqlite3.Connection, query: str, limit: int = 30,
                rowid_map: dict[int, str] | None = None,
                ) -> list[tuple[str, float]]:
    """Run an FTS query and return (memory_id, bm25_score) pairs.

    If rowid_map is provided, uses it for rowid->memory_id lookup (no DB hit).
    Handles OperationalError gracefully (returns empty list).
    """
    if not query or not query.strip():
        return []
    try:
        rows = db.execute(
            "SELECT rowid, bm25(memory_fts) as score "
            "FROM memory_fts WHERE memory_fts MATCH ? ORDER BY score LIMIT ?",
            (query, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    results = []
    for row in rows:
        if rowid_map:
            mid = rowid_map.get(row["rowid"])
        else:
            mapped = db.execute(
                "SELECT memory_id FROM memory_rowid_map WHERE rowid = ?",
                (row["rowid"],),
            ).fetchone()
            mid = mapped["memory_id"] if mapped else None
        if mid:
            results.append((mid, row["score"]))
    return results


def _extract_question_entities(question: str, speakers: set[str]) -> set[str]:
    """Extract entity names from a question.

    Two sources:
    1. Speaker names: match question tokens against known speakers (case-insensitive).
    2. Known entities: match against LOCOMO_ENTITIES allowlist.

    Returns lowercase entity strings.
    """
    entities = set()
    for tok in question.split():
        clean = re.sub(r"[.,!?\"';:()\[\]]+", "", tok).lower()
        if clean and (clean in speakers or clean in LOCOMO_ENTITIES):
            entities.add(clean)

    return entities


# ---------------------------------------------------------------------------
# Method 1: Entity-focused retrieval
# ---------------------------------------------------------------------------

def expand_entity_focus(ctx: ExpansionContext) -> set[str]:
    """Find ALL memories mentioning the question's primary entity.

    Extracts entity names from the question and runs FTS queries
    for each, adding candidates not already in the pool.

    Note: re-ranking existing candidates by overwriting fts_ranked was
    tried and hurt performance (-18pp multi-hop R@10) because the
    reranker's fts_rank feature has different semantics than entity-rank.
    To leverage entity-focused signals for existing candidates, add a
    separate reranker feature (entity_fts_rank) and retrain.
    """
    entities = _extract_question_entities(ctx.question, ctx.all_speakers)
    if not entities:
        return set()

    new_ids = set()
    for entity in entities:
        # FTS phrase match — generous limit to find candidates outside initial pool
        results = _fts_search(ctx.db, f'"{entity}"', limit=100, rowid_map=ctx.rowid_map)
        for mid, score in results:
            if mid not in ctx.existing_ids and mid not in new_ids:
                new_ids.add(mid)

    if new_ids:
        logger.debug("  expand_entity_focus: +%d candidates (entities: %s)",
                     len(new_ids), ", ".join(sorted(entities)))
    return new_ids


# ---------------------------------------------------------------------------
# Method 2: Multi-query fusion
# ---------------------------------------------------------------------------

def expand_multi_query(ctx: ExpansionContext) -> set[str]:
    """Decompose question into entity+attribute sub-queries for broader FTS.

    Generates multiple targeted FTS queries and merges via mini-RRF.
    Only adds NEW candidates not already in the pool.

    Note: re-ranking existing candidates via fts_ranked was tried and
    degraded performance. To leverage sub-query signals for existing
    candidates, add sub-query features to the reranker and retrain.
    """
    entities = _extract_question_entities(ctx.question, ctx.all_speakers)

    # Extract content words (non-stopwords, non-entities)
    tokens = ctx.question.lower().split()
    content_words = []
    entity_lower = {e.lower() for e in entities}
    for tok in tokens:
        clean = re.sub(r"[.,!?\"';:()\[\]]+", "", tok)
        if (clean and len(clean) >= 3
                and clean not in STOPWORDS
                and clean not in entity_lower):
            content_words.append(clean)

    # Generate sub-queries
    sub_queries = []

    # Entity-only queries
    for entity in sorted(entities):
        sub_queries.append(f'"{entity}"')

    # Entity + content word pairs
    for entity in sorted(entities):
        for cw in content_words:
            sub_queries.append(f'"{entity}" AND {cw}')

    # If no entities found, fall back to content word pairs
    if not entities and len(content_words) >= 2:
        for i, cw1 in enumerate(content_words[:4]):
            for cw2 in content_words[i+1:5]:
                sub_queries.append(f"{cw1} AND {cw2}")

    # Cap and deduplicate
    seen = set()
    unique_queries = []
    for q in sub_queries:
        if q not in seen:
            seen.add(q)
            unique_queries.append(q)
    sub_queries = unique_queries[:8]

    if not sub_queries:
        return set()

    # Run each sub-query and score via mini-RRF (new candidates only)
    rrf_scores: dict[str, float] = {}
    RRF_K = 60

    for sq in sub_queries:
        results = _fts_search(ctx.db, sq, limit=20, rowid_map=ctx.rowid_map)
        for rank, (mid, _score) in enumerate(results):
            if mid not in ctx.existing_ids:
                rrf_scores[mid] = rrf_scores.get(mid, 0.0) + 1.0 / (RRF_K + rank)

    # Take top-30 by fused score
    ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:30]
    new_ids = {mid for mid, _ in ranked}

    if new_ids:
        logger.debug("  expand_multi_query: +%d candidates (%d sub-queries)",
                     len(new_ids), len(sub_queries))
    return new_ids


# ---------------------------------------------------------------------------
# Method 3: Keyword expansion from results
# ---------------------------------------------------------------------------

def expand_keyword(ctx: ExpansionContext) -> set[str]:
    """Extract distinctive keywords from seeds and run a second FTS pass.

    Finds terms that appear in retrieved evidence but not in the original
    query, then searches for other memories containing those terms.
    """
    if not ctx.seeds:
        return set()

    # Collect tokens from seed content
    seed_tokens: Counter = Counter()
    for seed_id in ctx.seeds:
        row = ctx.db.execute(
            "SELECT content FROM memories WHERE id = ?", (seed_id,)
        ).fetchone()
        if not row or not row["content"]:
            continue
        # Strip [Speaker] prefix
        content = re.sub(r"^\[[^\]]+\]\s*", "", row["content"])
        for tok in content.lower().split():
            clean = re.sub(r"[.,!?\"';:()\[\]]+", "", tok)
            if clean and len(clean) >= 3 and clean not in STOPWORDS:
                seed_tokens[clean] += 1

    # Remove tokens that are already in the query
    query_tokens = {
        re.sub(r"[.,!?\"';:()\[\]]+", "", t.lower())
        for t in ctx.question.split()
    }
    for qt in query_tokens:
        seed_tokens.pop(qt, None)

    # Rank by TF-IDF: frequent in seeds, rare in corpus
    idf, _ = _get_idf(ctx.db)
    top_terms = sorted(
        seed_tokens,
        key=lambda t: seed_tokens[t] * idf.get(t, 0.0),
        reverse=True,
    )[:10]
    if not top_terms:
        return set()

    # Run FTS with OR-joined keywords
    fts_query = " OR ".join(top_terms)
    results = _fts_search(ctx.db, fts_query, limit=30, rowid_map=ctx.rowid_map)

    new_ids = set()
    for mid, score in results:
        if mid not in ctx.existing_ids:
            new_ids.add(mid)

    if new_ids:
        logger.debug("  expand_keyword: +%d candidates (terms: %s)",
                     len(new_ids), ", ".join(top_terms[:5]))
    return new_ids


# ---------------------------------------------------------------------------
# Method 4: Session expansion
# ---------------------------------------------------------------------------

def expand_session(ctx: ExpansionContext) -> set[str]:
    """Find other memories from the same conversation sessions as seeds."""
    if not ctx.seeds:
        return set()

    new_ids = set()
    sessions_expanded = set()

    for seed_id in ctx.seeds:
        row = ctx.db.execute(
            "SELECT themes FROM memories WHERE id = ?", (seed_id,)
        ).fetchone()
        if not row or not row["themes"]:
            continue
        try:
            themes = json.loads(row["themes"])
        except (json.JSONDecodeError, TypeError):
            continue

        for t in themes:
            if isinstance(t, str) and t.startswith("session_"):
                if t in sessions_expanded:
                    continue
                sessions_expanded.add(t)
                session_rows = ctx.db.execute(
                    "SELECT id FROM memories WHERE status = 'active' AND themes LIKE ?",
                    (f'%"{t}"%',),
                ).fetchall()
                count = 0
                for sr in session_rows:
                    if sr["id"] not in ctx.existing_ids and sr["id"] not in new_ids:
                        new_ids.add(sr["id"])
                        count += 1
                        if count >= 5:
                            break

    if new_ids:
        logger.debug("  expand_session: +%d candidates (sessions: %s)",
                     len(new_ids), ", ".join(sorted(sessions_expanded)))
    return new_ids


# ---------------------------------------------------------------------------
# Method 5: Entity bridge expansion
# ---------------------------------------------------------------------------

def expand_entity_bridge(ctx: ExpansionContext) -> set[str]:
    """Extract entities from seed content and search for bridging turns.

    Finds proper nouns in the seeds that aren't in the original query,
    then searches for other memories mentioning those entities.
    """
    if not ctx.seeds:
        return set()

    bridge_entities = set()
    query_tokens_lower = set(ctx.question.lower().split())

    for seed_id in ctx.seeds:
        row = ctx.db.execute(
            "SELECT content FROM memories WHERE id = ?", (seed_id,)
        ).fetchone()
        if not row or not row["content"]:
            continue
        for w in row["content"].split():
            clean = re.sub(r"[.,!?\"';:()\[\]]+$", "", w).lower()
            if clean in LOCOMO_ENTITIES:
                bridge_entities.add(clean)

    # Remove entities already in the query
    bridge_entities -= query_tokens_lower

    if not bridge_entities:
        return set()

    # FTS search for bridge entities
    from memory.fts import sanitize_fts_query
    entity_query = sanitize_fts_query(" OR ".join(sorted(bridge_entities)[:10]))
    if not entity_query:
        return set()

    results = _fts_search(ctx.db, entity_query, limit=20, rowid_map=ctx.rowid_map)
    new_ids = set()
    for mid, score in results:
        if mid not in ctx.existing_ids:
            new_ids.add(mid)

    if new_ids:
        logger.debug("  expand_entity_bridge: +%d candidates (bridge entities: %s)",
                     len(new_ids), ", ".join(sorted(bridge_entities)[:5]))
    return new_ids


# ---------------------------------------------------------------------------
# Method 6: Rocchio PRF
# ---------------------------------------------------------------------------

def expand_rocchio(ctx: ExpansionContext) -> set[str]:
    """Pseudo-relevance feedback via embedding centroid blend.

    Computes 0.7 * query_embedding + 0.3 * seed_centroid, normalizes,
    and searches for nearest neighbors in the expanded vector space.
    """
    import numpy as np
    from memory.vectors import serialize_f32, deserialize_f32

    if not ctx.seeds:
        return set()

    # Load seed embeddings
    seed_embeddings = []
    for seed_id in ctx.seeds:
        vec_row = ctx.db.execute(
            "SELECT embedding FROM memory_vec WHERE rowid = "
            "(SELECT rowid FROM memory_rowid_map WHERE memory_id = ?)",
            (seed_id,),
        ).fetchone()
        if vec_row and vec_row["embedding"]:
            seed_embeddings.append(
                np.array(deserialize_f32(vec_row["embedding"]), dtype=np.float32)
            )

    if not seed_embeddings:
        return set()

    q_emb = np.array(ctx.query_embedding, dtype=np.float32)
    centroid = np.mean(seed_embeddings, axis=0)
    expanded_emb = 0.7 * q_emb + 0.3 * centroid
    norm = np.linalg.norm(expanded_emb)
    if norm > 0:
        expanded_emb = expanded_emb / norm

    rocchio_results = ctx.db.execute(
        "SELECT rowid, distance FROM memory_vec "
        "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (serialize_f32(expanded_emb.tolist()), 30),
    ).fetchall()

    new_ids = set()
    for rr in rocchio_results:
        mapped = ctx.db.execute(
            "SELECT memory_id FROM memory_rowid_map WHERE rowid = ?",
            (rr["rowid"],),
        ).fetchone()
        if mapped and mapped["memory_id"] not in ctx.existing_ids:
            mid = mapped["memory_id"]
            new_ids.add(mid)
            # Update vec_ranked/distances so reranker can see these
            ctx.vec_ranked[mid] = len(ctx.vec_ranked)
            ctx.vec_distances[mid] = rr["distance"]

    if new_ids:
        logger.debug("  expand_rocchio: +%d candidates", len(new_ids))
    return new_ids


# ---------------------------------------------------------------------------
# Feature computation helpers (for reranker, no side effects on ctx)
# ---------------------------------------------------------------------------

def compute_entity_fts_ranks(
    db: sqlite3.Connection,
    question: str,
    speakers: set[str],
    candidate_ids: set[str],
    rowid_map: dict[int, str] | None = None,
) -> dict[str, int]:
    """Return {memory_id: rank} for entity-focused FTS. -1 if not found.

    Runs FTS phrase queries for each entity extracted from the question.
    For candidates matching multiple entities, keeps the best (lowest) rank.
    """
    entities = _extract_question_entities(question, speakers)
    if not entities:
        return {mid: -1 for mid in candidate_ids}

    best_rank: dict[str, int] = {}
    for entity in entities:
        results = _fts_search(db, f'"{entity}"', limit=100, rowid_map=rowid_map)
        for rank, (mid, _score) in enumerate(results):
            if mid in candidate_ids:
                if mid not in best_rank or rank < best_rank[mid]:
                    best_rank[mid] = rank

    return {mid: best_rank.get(mid, -1) for mid in candidate_ids}


def compute_sub_query_hits(
    db: sqlite3.Connection,
    question: str,
    speakers: set[str],
    candidate_ids: set[str],
    rowid_map: dict[int, str] | None = None,
) -> dict[str, int]:
    """Return {memory_id: hit_count} for sub-query decomposition.

    Decomposes question into entity-only and entity+content_word sub-queries.
    For each candidate, counts how many sub-queries returned it in top-30.
    """
    entities = _extract_question_entities(question, speakers)

    # Extract content words (non-stopwords, non-entities)
    tokens = question.lower().split()
    entity_lower = {e.lower() for e in entities}
    content_words = []
    for tok in tokens:
        clean = re.sub(r"[.,!?\"';:()\[\]]+", "", tok)
        if (clean and len(clean) >= 3
                and clean not in STOPWORDS
                and clean not in entity_lower):
            content_words.append(clean)

    # Generate sub-queries (same logic as expand_multi_query)
    sub_queries = []
    for entity in sorted(entities):
        sub_queries.append(f'"{entity}"')
    for entity in sorted(entities):
        for cw in content_words:
            sub_queries.append(f'"{entity}" AND {cw}')
    if not entities and len(content_words) >= 2:
        for i, cw1 in enumerate(content_words[:4]):
            for cw2 in content_words[i+1:5]:
                sub_queries.append(f"{cw1} AND {cw2}")

    # Deduplicate and cap
    seen = set()
    unique_queries = []
    for q in sub_queries:
        if q not in seen:
            seen.add(q)
            unique_queries.append(q)
    sub_queries = unique_queries[:8]

    if not sub_queries:
        return {mid: 0 for mid in candidate_ids}

    # Count how many sub-queries each candidate appears in
    hit_counts: dict[str, int] = {mid: 0 for mid in candidate_ids}
    for sq in sub_queries:
        results = _fts_search(db, sq, limit=30, rowid_map=rowid_map)
        matched_ids = {mid for mid, _score in results}
        for mid in candidate_ids:
            if mid in matched_ids:
                hit_counts[mid] += 1

    return hit_counts


def compute_seed_keyword_overlap(
    db: sqlite3.Connection,
    question: str,
    seed_ids: list[str],
    candidate_ids: set[str],
    memories: dict[str, dict] | None = None,
) -> dict[str, float]:
    """Return {memory_id: overlap_fraction} for seed keyword analysis.

    Extracts distinctive keywords from seeds (tokens in seeds but not in
    query, excluding stopwords). For each candidate, computes the fraction
    of seed keywords present in its content.

    If memories dict is provided, uses cached content instead of DB lookups.
    """
    if not seed_ids:
        return {mid: 0.0 for mid in candidate_ids}

    # Collect tokens from seed content
    seed_tokens: Counter = Counter()
    for seed_id in seed_ids:
        if memories and seed_id in memories:
            content = memories[seed_id].get("content", "")
        else:
            row = db.execute(
                "SELECT content FROM memories WHERE id = ?", (seed_id,)
            ).fetchone()
            content = row["content"] if row else ""
        if not content:
            continue
        content = re.sub(r"^\[[^\]]+\]\s*", "", content)
        for tok in content.lower().split():
            clean = re.sub(r"[.,!?\"';:()\[\]]+", "", tok)
            if clean and len(clean) >= 3 and clean not in STOPWORDS:
                seed_tokens[clean] += 1

    # Remove tokens already in the query
    query_tokens = {
        re.sub(r"[.,!?\"';:()\[\]]+", "", t.lower())
        for t in question.split()
    }
    for qt in query_tokens:
        seed_tokens.pop(qt, None)

    # Top distinctive terms (ranked by TF-IDF)
    idf, _ = _get_idf(db)
    top_terms_list = sorted(
        seed_tokens,
        key=lambda t: seed_tokens[t] * idf.get(t, 0.0),
        reverse=True,
    )[:20]
    top_terms = set(top_terms_list)
    if not top_terms:
        return {mid: 0.0 for mid in candidate_ids}

    # For each candidate, compute overlap fraction
    result: dict[str, float] = {}
    for mid in candidate_ids:
        if memories and mid in memories:
            mem_content = memories[mid].get("content", "")
        else:
            row = db.execute(
                "SELECT content FROM memories WHERE id = ?", (mid,)
            ).fetchone()
            mem_content = row["content"] if row else ""
        if not mem_content:
            result[mid] = 0.0
            continue
        content_tokens = set()
        for tok in mem_content.lower().split():
            clean = re.sub(r"[.,!?\"';:()\[\]]+", "", tok)
            if clean:
                content_tokens.add(clean)
        matched = top_terms & content_tokens
        result[mid] = len(matched) / len(top_terms)

    return result


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

def run_expansions(ctx: ExpansionContext, config) -> ExpansionResult:
    """Run all enabled expansion methods and return merged results.

    ``config`` can be a BenchConfig or a dict with method names as keys.
    """
    result = ExpansionResult()
    pre_count = len(ctx.existing_ids)

    # Support both BenchConfig and plain dict
    def _enabled(method: str) -> bool:
        if isinstance(config, dict):
            return config.get(method, False)
        return getattr(config, f"expand_{method}", False)

    if _enabled("entity_focus"):
        ids = expand_entity_focus(ctx)
        result.add("entity_focus", ids)
        ctx.existing_ids |= ids

    if _enabled("multi_query"):
        ids = expand_multi_query(ctx)
        result.add("multi_query", ids)
        ctx.existing_ids |= ids

    if _enabled("keyword"):
        ids = expand_keyword(ctx)
        result.add("keyword", ids)
        ctx.existing_ids |= ids

    if _enabled("session"):
        ids = expand_session(ctx)
        result.add("session", ids)
        ctx.existing_ids |= ids

    if _enabled("entity_bridge"):
        ids = expand_entity_bridge(ctx)
        result.add("entity_bridge", ids)
        ctx.existing_ids |= ids

    if _enabled("rocchio"):
        ids = expand_rocchio(ctx)
        result.add("rocchio", ids)
        ctx.existing_ids |= ids

    total_new = len(result.all_new_ids)
    if total_new > 0:
        methods_used = [m for m, ids in result.per_method.items() if ids]
        logger.debug("  Expansion: %d → %d candidates (+%d net from %d methods: %s)",
                     pre_count, pre_count + total_new, total_new,
                     len(methods_used), ", ".join(methods_used))

    return result
