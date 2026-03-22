# /// script
# requires-python = ">=3.11"
# dependencies = ["sqlite-vec>=0.1.6", "openai>=2.0.0", "numpy>=1.26", "lightgbm>=4.0,<4.7", "scikit-learn>=1.0"]
# ///
"""
Train a LoCoMo-specific LightGBM reranker and run ablation experiments.

Extracts 25 features designed for fresh LoCoMo DBs (no feedback, no edges,
no Hebbian — those are all dead on benchmark data). Groups A-E (17 features) are
v2 baseline; Group F (5 features) adds temporal, entity density, and inter-passage
features; Group G (3 features) adds embedding-based set diversity (MMR, unique tokens,
centroid distance). Uses leave-one-conversation-out CV (10-fold, one per conversation).

Usage:
  uv run scripts/locomo_bench/train_locomo_reranker.py --ablation
  uv run scripts/locomo_bench/train_locomo_reranker.py --features A B
  uv run scripts/locomo_bench/train_locomo_reranker.py --extract-only
  uv run scripts/locomo_bench/train_locomo_reranker.py --train-only --ablation
  uv run scripts/locomo_bench/train_locomo_reranker.py --save-model
"""

import argparse
import json
import math
import pickle
import re
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import warnings

import lightgbm as lgb
import numpy as np

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

# Setup: add src/ to path for memory package imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from memory.constants import BM25_SUMMARY_WT, BM25_THEMES_WT, DATA_DIR
from memory.reranker import _compute_proximity
from memory.vectors import deserialize_f32

BENCHMARK_DIR = DATA_DIR / "benchmark"

# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

_ENTITY_STOPWORDS = {
    "i'm", "i've", "i'll", "i'd", "it", "it's", "its", "we", "he", "she",
    "they", "what", "how", "the", "this", "that", "your", "my", "do",
    "can't", "don't", "won't", "have", "keep", "take", "thanks", "well",
}


def _extract_entities(text: str, speakers: set[str]) -> set[str]:
    """Extract likely proper nouns, excluding speakers and stopwords."""
    text = re.sub(r'^\[[^\]]+\]\s*', '', text)  # strip [Speaker] prefix
    entities = set()
    for i, w in enumerate(text.split()):
        if i == 0:
            continue
        clean = re.sub(r'[.,!?"\';:()]+$', '', w)
        if clean and len(clean) >= 2 and clean[0].isupper() and not clean.isupper():
            lower = clean.lower()
            if lower not in _ENTITY_STOPWORDS and lower not in speakers:
                entities.add(lower)
    return entities


_TEMPORAL_PATTERN = re.compile(
    r'\b(?:'
    r'(?:January|February|March|April|May|June|July|August|September|October|November|December'
    r'|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b'
    r'|20\d{2}'
    r'|yesterday|today|tomorrow'
    r'|(?:last|next|this)\s+(?:week|month|year|summer|winter|spring|fall)'
    r'|\d{1,2}/\d{1,2}(?:/\d{2,4})?'
    r'|\d{1,2}\s+(?:days?|weeks?|months?|years?)\s+ago'
    r')',
    re.IGNORECASE,
)


def _count_temporal_exprs(text: str) -> int:
    """Count temporal expressions in text (dates, relative time, years)."""
    return len(_TEMPORAL_PATTERN.findall(text))
FEATURES_PATH = BENCHMARK_DIR / "locomo_features.pkl"
MODEL_PATH = BENCHMARK_DIR / "locomo_reranker_model.pkl"

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

# Group A: Core retrieval signals (0-4)
# Group B: Query-content text features (5-8)
# Group C: Cross-result features (9-11)
# Group D: Metadata (12-14)
# Group E: Entity & session features (15-16)

FEATURE_NAMES = [
    # Group A
    "fts_rank",             # 0: FTS rank (-1 if not retrieved)
    "vec_rank",             # 1: Vec rank (-1 if not retrieved)
    "fts_bm25",             # 2: Raw BM25 score
    "vec_dist",             # 3: Raw cosine distance
    "theme_overlap",        # 4: Query token overlap with themes
    # Group B
    "query_coverage",       # 5: matched_query_terms / total_query_terms
    "proximity",            # 6: Inverse min-span of query terms in content
    "speaker_match",        # 7: Does query mention this turn's speaker?
    "query_length",         # 8: Number of query tokens
    # Group C
    "rank_agreement",       # 9: abs(fts_rank - vec_rank), 9999 if one missing
    "neighbor_density",     # 10: Candidates within ±2 dia_id positions
    "score_percentile",     # 11: Percentile of bare RRF score within query
    # Group D
    "token_count",          # 12: Turn length
    "age_days",             # 13: Temporal position in conversation
    "theme_count",          # 14: Number of themes
    # Group E
    "entity_overlap",       # 15: Shared named entities between query and candidate
    "session_cooccurrence", # 16: Other top-20 RRF candidates in same session
    # Group F: Inter-passage and structural features
    "has_temporal_expr",    # 17: Count of temporal expressions in content
    "entity_density",       # 18: Named entities per token (entity-richness)
    "topk_session_frac",    # 19: Fraction of top-10 RRF sharing this session (2nd pass)
    "entity_bridge_count",  # 20: Entities shared with other top-10, not query (2nd pass)
    "theme_complementarity",# 21: Query coverage not already in top-10 (2nd pass)
    # Group G: Set diversity features (2nd pass, embedding-based)
    "mmr_redundancy",       # 22: Max embedding similarity to other top-10 candidates
    "unique_token_frac",    # 23: Fraction of tokens unique to this candidate vs top-10
    "centroid_distance",    # 24: Embedding distance from centroid of top-10
]

NUM_FEATURES = len(FEATURE_NAMES)  # 25

FEATURE_GROUPS = {
    "A": list(range(0, 5)),
    "B": list(range(5, 9)),
    "C": list(range(9, 12)),
    "D": list(range(12, 15)),
    "E": list(range(15, 17)),
    "F": list(range(17, 22)),
    "G": list(range(22, 25)),
}

_ALL_INDICES = list(range(NUM_FEATURES))

ABLATION_CONFIGS = {
    "A_only": ["A"],
    "B_only": ["B"],
    "C_only": ["C"],
    "D_only": ["D"],
    "E_only": ["E"],
    "all": ["A", "B", "C", "D", "E"],
    "no_A": ["B", "C", "D", "E"],
    "no_B": ["A", "C", "D", "E"],
    "no_C": ["A", "B", "D", "E"],
    "no_D": ["A", "B", "C", "E"],
    "no_E": ["A", "B", "C", "D"],
    # Individual C feature drops
    "no_C9": [i for i in _ALL_INDICES if i != 9],   # drop rank_agreement
    "no_C10": [i for i in _ALL_INDICES if i != 10],  # drop neighbor_density
    "no_C11": [i for i in _ALL_INDICES if i != 11],  # drop score_percentile
    # Dead feature pruning (proximity=6, age_days=13, theme_count=14)
    "pruned": [i for i in _ALL_INDICES if i not in (6, 13, 14)],
    "no_C_pruned": [i for i in _ALL_INDICES if i not in (6, 9, 10, 11, 13, 14)],
    # Group F ablations
    "F_only": ["F"],
    "G_only": ["G"],
    "no_F": ["A", "B", "C", "D", "E", "G"],
    "no_G": ["A", "B", "C", "D", "E", "F"],
    "no_FG": ["A", "B", "C", "D", "E"],
    "no_C_pruned_F": [i for i in _ALL_INDICES if i not in (6, 9, 10, 11, 13, 14, 22, 23, 24)],  # best v2 + F
    "no_C_pruned_G": [i for i in _ALL_INDICES if i not in (6, 9, 10, 11, 13, 14, 17, 18, 19, 20, 21)],  # best v2 + G
    "no_C_pruned_FG": [i for i in _ALL_INDICES if i not in (6, 9, 10, 11, 13, 14)],  # best v2 + F + G
    "all_v3": ["A", "B", "C", "D", "E", "F", "G"],
}

RRF_K = 60  # Standard RRF constant


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_conversation_data(conv_idx: int) -> dict | None:
    """Load pre-ingested DB and metadata for one conversation."""
    from locomo_bench.config import BenchConfig
    from locomo_bench.ingest import extract_questions, extract_turns, load_locomo
    from locomo_bench.run import setup_isolated_db

    bench_config = BenchConfig()
    run_dir = bench_config.base_dir / f"retrieval_{conv_idx}"

    if not (run_dir / "memory.db").exists():
        print(f"  Conv {conv_idx}: no DB at {run_dir}, skipping")
        return None

    dia_map_path = run_dir / "dia_map.json"
    if not dia_map_path.exists():
        print(f"  Conv {conv_idx}: no dia_map.json, skipping")
        return None

    # Load dia_map
    with open(dia_map_path) as f:
        raw = json.load(f)
    dia_map = {}
    for k, v in raw.items():
        parts = k.split(":", 1)
        if len(parts) == 2:
            dia_map[(int(parts[0]), parts[1])] = v

    # Setup isolated DB
    setup_isolated_db(run_dir)

    # Load questions for this conversation
    data = load_locomo(bench_config.locomo_data)
    all_questions = extract_questions(data)
    conv_questions = [q for q in all_questions if q["conv_id"] == conv_idx]

    # Load memory metadata from DB
    from memory.db import get_db
    db = get_db()

    rows = db.execute("""
        SELECT m.id, m.content, m.themes, m.token_count, m.created_at, m.summary,
               v.embedding
        FROM memories m
        LEFT JOIN memory_vec v ON v.rowid = m.rowid
        WHERE m.status = 'active'
    """).fetchall()

    # First pass: collect all speakers for entity extraction
    all_speakers = set()
    parsed_rows = []
    for r in rows:
        content = r["content"] or ""
        sp_match = re.match(r"^\[([^\]]+)\]", content)
        speaker = sp_match.group(1).lower() if sp_match else ""
        if speaker:
            all_speakers.add(speaker)
        parsed_rows.append((r, content, speaker))

    memories = {}
    for r, content, speaker in parsed_rows:
        mid = r["id"]
        themes_list = []
        if r["themes"]:
            try:
                themes_list = json.loads(r["themes"])
            except json.JSONDecodeError:
                pass

        try:
            from datetime import datetime, timezone
            created = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - created).total_seconds() / 86400
        except Exception:
            age_days = 0.0

        # Build content text for proximity/coverage
        content_lower = content.lower()
        content_tokens = content_lower.split()

        # Theme tokens for overlap
        theme_tokens = set()
        for t in themes_list:
            theme_tokens.update(str(t).lower().replace("-", " ").split())

        # Parse session from themes (e.g. "session_3")
        session = None
        for t in themes_list:
            if isinstance(t, str) and t.startswith("session_"):
                session = t
                break

        token_count = r["token_count"] or len(content.split())
        entities = _extract_entities(content, all_speakers)
        # Parse embedding
        emb = None
        if r["embedding"]:
            try:
                emb = np.array(deserialize_f32(r["embedding"]), dtype=np.float32)
            except Exception:
                pass
        memories[mid] = {
            "content": content,
            "content_lower": content_lower,
            "content_tokens": content_tokens,
            "speaker": speaker,
            "themes_list": themes_list,
            "theme_tokens": theme_tokens,
            "theme_count": len(themes_list),
            "token_count": token_count,
            "age_days": age_days,
            "created_at": r["created_at"],
            "entities": entities,
            "session": session,
            "embedding": emb,
            # Group F pre-computed
            "temporal_expr_count": _count_temporal_exprs(content),
            "entity_density": len(entities) / max(token_count, 1),
        }

    db.close()

    # Build dia_id ordinal mapping for neighbor_density
    # Sort memories by created_at to get temporal ordering
    sorted_mids = sorted(memories.keys(), key=lambda m: memories[m]["created_at"])
    ordinal_map = {mid: i for i, mid in enumerate(sorted_mids)}
    for mid in memories:
        memories[mid]["ordinal"] = ordinal_map[mid]

    return {
        "conv_idx": conv_idx,
        "run_dir": run_dir,
        "dia_map": dia_map,
        "questions": conv_questions,
        "memories": memories,
        "ordinal_map": ordinal_map,
    }


# ---------------------------------------------------------------------------
# Search precomputation
# ---------------------------------------------------------------------------


def run_searches(conv_data: dict) -> dict:
    """Run FTS and vec searches for all questions in a conversation."""
    from memory.db import get_db
    from memory.embeddings import embed_text
    from memory.fts import sanitize_fts_query
    from memory.vectors import serialize_f32

    from locomo_bench.run import setup_isolated_db

    # Re-setup to point at correct DB
    setup_isolated_db(conv_data["run_dir"])

    db = get_db()
    questions = conv_data["questions"]

    fts_results = {}   # query -> [memory_ids ordered by rank]
    fts_scores = {}    # query -> {memory_id: bm25_score}
    vec_results = {}   # query -> [memory_ids ordered by rank]
    vec_scores = {}    # query -> {memory_id: cosine_distance}

    for qi, q in enumerate(questions):
        qtext = q["question"]

        # --- FTS search ---
        fts_query = sanitize_fts_query(qtext)
        fts_ids = []
        fts_sc = {}
        try:
            fts_rows = db.execute(
                f"""
                SELECT rowid, bm25(memory_fts, {BM25_SUMMARY_WT}, {BM25_THEMES_WT}) as rank
                FROM memory_fts WHERE memory_fts MATCH ?
                ORDER BY rank LIMIT 200
                """,
                (fts_query,),
            ).fetchall()
            for row in fts_rows:
                mapped = db.execute(
                    "SELECT memory_id FROM memory_rowid_map WHERE rowid = ?",
                    (row["rowid"],),
                ).fetchone()
                if mapped:
                    fts_ids.append(mapped["memory_id"])
                    fts_sc[mapped["memory_id"]] = row["rank"]
        except sqlite3.OperationalError:
            pass

        fts_results[qtext] = fts_ids
        fts_scores[qtext] = fts_sc

        # --- Vec search ---
        query_embedding = embed_text(qtext)
        vec_rows = db.execute(
            """
            SELECT rowid, distance FROM memory_vec
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
            """,
            (serialize_f32(query_embedding), 200),
        ).fetchall()

        vec_ids = []
        vec_sc = {}
        for row in vec_rows:
            mapped = db.execute(
                "SELECT memory_id FROM memory_rowid_map WHERE rowid = ?",
                (row["rowid"],),
            ).fetchone()
            if mapped:
                vec_ids.append(mapped["memory_id"])
                vec_sc[mapped["memory_id"]] = row["distance"]

        vec_results[qtext] = vec_ids
        vec_scores[qtext] = vec_sc

        if (qi + 1) % 50 == 0:
            print(f"    Searched {qi + 1}/{len(questions)} queries")

    db.close()

    return {
        "fts_results": fts_results,
        "fts_scores": fts_scores,
        "vec_results": vec_results,
        "vec_scores": vec_scores,
    }


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_features(conv_data: dict, search_data: dict) -> dict:
    """Extract features for all (query, candidate) pairs in a conversation.

    Returns dict with features, labels, query_ids, memory_ids arrays.
    """
    questions = conv_data["questions"]
    memories = conv_data["memories"]
    dia_map = conv_data["dia_map"]
    ordinal_map = conv_data["ordinal_map"]

    all_features = []
    all_labels = []
    all_query_ids = []
    all_mids = []

    for qi, q in enumerate(questions):
        qtext = q["question"]

        # Evidence set (memory IDs that are correct answers)
        evidence_mids = set()
        for eid in q.get("evidence", []):
            key = (q["conv_id"], eid)
            if key in dia_map:
                evidence_mids.add(dia_map[key])

        if not evidence_mids:
            continue

        # Build candidate pool from search results
        fts_ids = search_data["fts_results"].get(qtext, [])
        vec_ids = search_data["vec_results"].get(qtext, [])
        fts_ranked = {mid: rank for rank, mid in enumerate(fts_ids)}
        vec_ranked = {mid: rank for rank, mid in enumerate(vec_ids)}

        # Theme overlap
        query_tokens = set(qtext.lower().split())
        theme_ranked = {}
        theme_overlap_map = {}
        scored_themes = []
        for mid, mem in memories.items():
            overlap = len(query_tokens & mem["theme_tokens"])
            if overlap > 0:
                scored_themes.append((mid, overlap))
                theme_overlap_map[mid] = overlap
        scored_themes.sort(key=lambda x: (-x[1], x[0]))
        for rank, (mid, _) in enumerate(scored_themes):
            theme_ranked[mid] = rank

        candidate_ids = set(fts_ranked) | set(vec_ranked) | set(theme_ranked)

        # Also include evidence memories (so we can label them even if not retrieved)
        candidate_ids |= evidence_mids

        if not candidate_ids:
            continue

        # Compute bare RRF scores for score_percentile
        rrf_scores = {}
        for mid in candidate_ids:
            score = 0.0
            if mid in fts_ranked:
                score += 1.0 / (RRF_K + fts_ranked[mid])
            if mid in vec_ranked:
                score += 1.0 / (RRF_K + vec_ranked[mid])
            rrf_scores[mid] = score

        # Score percentile
        all_rrf = sorted(rrf_scores.values())
        n_rrf = len(all_rrf)

        # Neighbor density: count candidates within ±2 ordinal positions
        candidate_ordinals = {}
        for mid in candidate_ids:
            if mid in memories:
                candidate_ordinals[mid] = memories[mid]["ordinal"]
        ordinal_set = set(candidate_ordinals.values())

        # Query-level features
        query_terms = [t for t in qtext.lower().split() if len(t) > 1]
        query_length = len(query_terms)

        # Speaker detection: check if any speaker name appears in query
        query_lower = qtext.lower()

        # Entity overlap: extract query entities
        # Build all_speakers from memories (already computed during load)
        all_speakers_set = {mem["speaker"] for mem in memories.values() if mem["speaker"]}
        query_entities = _extract_entities(qtext, all_speakers_set)

        # Session co-occurrence: get top-20 RRF candidates' sessions
        top20_rrf = sorted(candidate_ids, key=lambda m: rrf_scores.get(m, 0), reverse=True)[:20]
        top20_set = set(top20_rrf)
        session_counts = defaultdict(int)
        for mid in top20_rrf:
            mem = memories.get(mid)
            if mem and mem.get("session"):
                session_counts[mem["session"]] += 1

        # Build feature matrix
        candidate_list = sorted(candidate_ids)
        n = len(candidate_list)
        features = np.zeros((n, NUM_FEATURES), dtype=np.float32)
        labels = np.zeros(n, dtype=np.float32)

        for i, mid in enumerate(candidate_list):
            mem = memories.get(mid)
            if mem is None:
                continue

            fts_r = fts_ranked.get(mid, -1)
            vec_r = vec_ranked.get(mid, -1)

            # Group A: Core retrieval signals
            features[i, 0] = fts_r                                    # fts_rank
            features[i, 1] = vec_r                                    # vec_rank
            features[i, 2] = search_data["fts_scores"].get(
                qtext, {}).get(mid, 0.0)                              # fts_bm25
            features[i, 3] = search_data["vec_scores"].get(
                qtext, {}).get(mid, 0.0)                              # vec_dist
            features[i, 4] = theme_overlap_map.get(mid, 0)            # theme_overlap

            # Group B: Query-content text features
            if query_terms:
                content_set = set(mem["content_tokens"])
                matched = sum(1 for t in query_terms if t in content_set)
                features[i, 5] = matched / len(query_terms)           # query_coverage
            if len(query_terms) > 1:
                features[i, 6] = _compute_proximity(
                    query_terms, mem["content_tokens"])                # proximity

            # Speaker match
            if mem["speaker"] and mem["speaker"] in query_lower:
                features[i, 7] = 1.0                                  # speaker_match

            features[i, 8] = query_length                             # query_length

            # Group C: Cross-result features
            if fts_r >= 0 and vec_r >= 0:
                features[i, 9] = abs(fts_r - vec_r)                   # rank_agreement
            else:
                features[i, 9] = 9999                                 # rank_agreement (missing)

            # Neighbor density: count candidates within ±2 ordinal positions
            my_ord = candidate_ordinals.get(mid, -999)
            if my_ord >= 0:
                neighbors = sum(
                    1 for o in ordinal_set
                    if o != my_ord and abs(o - my_ord) <= 2
                )
                features[i, 10] = neighbors                           # neighbor_density

            # Score percentile
            rrf_s = rrf_scores.get(mid, 0.0)
            if n_rrf > 1:
                # Count how many scores are <= this one
                rank_in_sorted = sum(1 for s in all_rrf if s <= rrf_s)
                features[i, 11] = rank_in_sorted / n_rrf             # score_percentile
            else:
                features[i, 11] = 0.5

            # Group D: Metadata
            features[i, 12] = mem["token_count"]                      # token_count
            features[i, 13] = mem["age_days"]                         # age_days
            features[i, 14] = mem["theme_count"]                      # theme_count

            # Group E: Entity & session features
            features[i, 15] = len(query_entities & mem.get("entities", set()))  # entity_overlap
            mem_session = mem.get("session")
            if mem_session and mid in top20_set:
                features[i, 16] = session_counts.get(mem_session, 1) - 1  # session_cooccurrence

            # Group F: First-pass features (independent per candidate)
            features[i, 17] = mem.get("temporal_expr_count", 0)           # has_temporal_expr
            features[i, 18] = mem.get("entity_density", 0.0)             # entity_density

            # Label: binary
            labels[i] = 1.0 if mid in evidence_mids else 0.0

        # Group F: Second-pass features (inter-passage, based on top-10 RRF)
        top10_rrf = sorted(candidate_list,
                           key=lambda m: rrf_scores.get(m, 0), reverse=True)[:10]
        top10_set = set(top10_rrf)

        # Collect top-10 sessions, entities, and covered query terms
        top10_sessions = defaultdict(int)
        top10_entities = set()
        top10_query_coverage = set()
        for mid in top10_rrf:
            mem = memories.get(mid)
            if not mem:
                continue
            if mem.get("session"):
                top10_sessions[mem["session"]] += 1
            top10_entities |= mem.get("entities", set())
            content_set = set(mem["content_tokens"])
            top10_query_coverage |= (set(query_terms) & content_set)

        for i, mid in enumerate(candidate_list):
            mem = memories.get(mid)
            if mem is None:
                continue

            # topk_session_frac: fraction of top-10 sharing this session
            mem_session = mem.get("session")
            if mem_session and len(top10_rrf) > 0:
                # Count others in top-10 with same session (exclude self)
                same = top10_sessions.get(mem_session, 0)
                if mid in top10_set:
                    same = max(same - 1, 0)
                features[i, 19] = same / len(top10_rrf)                   # topk_session_frac

            # entity_bridge_count: entities shared with other top-10, NOT with query
            mem_entities = mem.get("entities", set())
            shared_with_topk = mem_entities & (top10_entities - query_entities)
            features[i, 20] = len(shared_with_topk)                       # entity_bridge_count

            # theme_complementarity: query terms this memory covers that top-10 don't
            if query_terms:
                content_set = set(mem["content_tokens"])
                my_coverage = set(query_terms) & content_set
                novel = my_coverage - top10_query_coverage
                features[i, 21] = len(novel) / len(query_terms)           # theme_complementarity

        # Group G: Set diversity features (embedding-based, 2nd pass)
        # Collect top-10 embeddings and token sets
        top10_embeddings = []
        top10_all_tokens = set()
        for mid in top10_rrf:
            mem = memories.get(mid)
            if not mem:
                continue
            if mem.get("embedding") is not None:
                top10_embeddings.append(mem["embedding"])
            top10_all_tokens |= set(mem["content_tokens"])

        # Compute centroid of top-10 embeddings
        centroid = None
        if top10_embeddings:
            centroid = np.mean(top10_embeddings, axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-9  # normalize

        for i, mid in enumerate(candidate_list):
            mem = memories.get(mid)
            if mem is None or mem.get("embedding") is None:
                continue
            emb = mem["embedding"]
            emb_norm = np.linalg.norm(emb)
            if emb_norm < 1e-9:
                continue

            # mmr_redundancy: max cosine similarity to other top-10 candidates
            max_sim = 0.0
            for other_emb in top10_embeddings:
                sim = float(np.dot(emb, other_emb) / (emb_norm * (np.linalg.norm(other_emb) + 1e-9)))
                if sim > max_sim:
                    max_sim = sim
            features[i, 22] = max_sim                                     # mmr_redundancy

            # unique_token_frac: fraction of this candidate's tokens not in other top-10
            mem_tokens = set(mem["content_tokens"])
            if mem_tokens:
                unique = mem_tokens - top10_all_tokens
                # If this candidate IS in top-10, exclude self from the comparison
                if mid in top10_set:
                    # Recompute: other top-10 tokens minus this candidate
                    other_tokens = set()
                    for other_mid in top10_rrf:
                        if other_mid != mid:
                            other_mem = memories.get(other_mid)
                            if other_mem:
                                other_tokens |= set(other_mem["content_tokens"])
                    unique = mem_tokens - other_tokens
                features[i, 23] = len(unique) / len(mem_tokens)           # unique_token_frac

            # centroid_distance: cosine distance from centroid of top-10
            if centroid is not None:
                cos_sim = float(np.dot(emb, centroid) / (emb_norm + 1e-9))
                features[i, 24] = 1.0 - cos_sim                          # centroid_distance

        all_features.append(features)
        all_labels.append(labels)
        all_query_ids.extend([qi] * n)
        all_mids.extend([(qtext, mid) for mid in candidate_list])

    if not all_features:
        return {"features": np.zeros((0, NUM_FEATURES)), "labels": np.zeros(0),
                "query_ids": np.zeros(0, dtype=np.int32), "memory_ids": []}

    features = np.concatenate(all_features, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    query_ids = np.array(all_query_ids, dtype=np.int32)

    return {
        "features": features,
        "labels": labels_arr,
        "query_ids": query_ids,
        "memory_ids": all_mids,
        "feature_names": FEATURE_NAMES,
    }


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


def compute_metrics(ranked_mids: list[str], evidence_mids: set[str]) -> dict:
    """Compute R@k, MRR, and NDCG@10 for a single query."""
    first_hit = None
    for i, mid in enumerate(ranked_mids):
        if mid in evidence_mids:
            first_hit = i + 1
            break

    metrics = {"mrr": 1.0 / first_hit if first_hit else 0.0}
    for k in [1, 3, 5, 10, 20]:
        top_k = set(ranked_mids[:k])
        metrics[f"r@{k}"] = 1.0 if (top_k & evidence_mids) else 0.0

    # NDCG@10
    dcg = 0.0
    for i, mid in enumerate(ranked_mids[:10]):
        if mid in evidence_mids:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because rank is 1-indexed
    # Ideal: all evidence in top positions
    ideal_hits = min(len(evidence_mids), 10)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    metrics["ndcg@10"] = dcg / idcg if idcg > 0 else 0.0

    return metrics


def evaluate_ranking(
    conv_data: dict,
    search_data: dict,
    feature_data: dict,
    model: lgb.LGBMRegressor | None,
    feature_indices: list[int],
) -> list[dict]:
    """Evaluate a model (or bare RRF) on one conversation's questions.

    Returns per-question metrics dicts.
    """
    questions = conv_data["questions"]
    dia_map = conv_data["dia_map"]
    memories = conv_data["memories"]

    # Group feature data by query
    query_groups = defaultdict(list)
    for idx, (qtext, mid) in enumerate(feature_data["memory_ids"]):
        query_groups[qtext].append(idx)

    results = []
    for q in questions:
        qtext = q["question"]
        evidence_mids = set()
        for eid in q.get("evidence", []):
            key = (q["conv_id"], eid)
            if key in dia_map:
                evidence_mids.add(dia_map[key])

        if not evidence_mids:
            continue

        indices = query_groups.get(qtext, [])
        if not indices:
            continue

        mids_for_q = [feature_data["memory_ids"][i][1] for i in indices]

        if model is not None:
            X = feature_data["features"][indices][:, feature_indices]
            scores = model.predict(X)
            scored = sorted(zip(mids_for_q, scores), key=lambda x: -x[1])
            ranked = [mid for mid, _ in scored]
        else:
            # Bare RRF ranking
            fts_ids = search_data["fts_results"].get(qtext, [])
            vec_ids = search_data["vec_results"].get(qtext, [])
            fts_ranked = {mid: rank for rank, mid in enumerate(fts_ids)}
            vec_ranked = {mid: rank for rank, mid in enumerate(vec_ids)}
            rrf = {}
            for mid in set(mids_for_q):
                score = 0.0
                if mid in fts_ranked:
                    score += 1.0 / (RRF_K + fts_ranked[mid])
                if mid in vec_ranked:
                    score += 1.0 / (RRF_K + vec_ranked[mid])
                rrf[mid] = score
            ranked = sorted(set(mids_for_q), key=lambda m: -rrf.get(m, 0))

        metrics = compute_metrics(ranked, evidence_mids)
        metrics["question"] = qtext
        metrics["category"] = q["category"]
        results.append(metrics)

    return results


def train_cv(
    all_conv_features: list[dict],
    all_conv_data: list[dict],
    all_search_data: list[dict],
    feature_indices: list[int],
    n_estimators: int = 500,
) -> tuple[list[dict], np.ndarray]:
    """Leave-one-conversation-out CV. Returns (per-question metrics, importances)."""
    n_convs = len(all_conv_features)
    all_metrics = []
    importances = np.zeros(len(feature_indices))

    for test_idx in range(n_convs):
        # Train on all other conversations
        train_X_parts = []
        train_y_parts = []
        for i in range(n_convs):
            if i == test_idx:
                continue
            fd = all_conv_features[i]
            if len(fd["labels"]) == 0:
                continue
            train_X_parts.append(fd["features"][:, feature_indices])
            train_y_parts.append(fd["labels"])

        if not train_X_parts:
            continue

        X_train = np.concatenate(train_X_parts)
        y_train = np.concatenate(train_y_parts)

        # Hold out 10% of training for early stopping
        n_train = len(y_train)
        n_val = max(1, n_train // 10)
        rng = np.random.RandomState(42 + test_idx)
        val_mask = np.zeros(n_train, dtype=bool)
        val_mask[rng.choice(n_train, size=n_val, replace=False)] = True

        model = lgb.LGBMRegressor(
            num_leaves=31, learning_rate=0.1, n_estimators=n_estimators,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            verbose=-1, random_state=42,
        )
        model.fit(
            X_train[~val_mask], y_train[~val_mask],
            eval_set=[(X_train[val_mask], y_train[val_mask])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        importances += model.feature_importances_

        # Evaluate on test conversation
        test_metrics = evaluate_ranking(
            all_conv_data[test_idx],
            all_search_data[test_idx],
            all_conv_features[test_idx],
            model,
            feature_indices,
        )
        for m in test_metrics:
            m["test_conv"] = test_idx
        all_metrics.extend(test_metrics)

    importances /= max(n_convs, 1)
    return all_metrics, importances


def train_full_model(
    all_conv_features: list[dict],
    feature_indices: list[int],
    n_estimators: int = 500,
) -> lgb.LGBMRegressor:
    """Train on all data, return final model."""
    X_parts = []
    y_parts = []
    for fd in all_conv_features:
        if len(fd["labels"]) == 0:
            continue
        X_parts.append(fd["features"][:, feature_indices])
        y_parts.append(fd["labels"])

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)

    model = lgb.LGBMRegressor(
        num_leaves=31, learning_rate=0.1, n_estimators=n_estimators,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        verbose=-1, random_state=42,
    )
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Bare RRF baseline
# ---------------------------------------------------------------------------


def evaluate_bare_rrf(
    all_conv_data: list[dict],
    all_search_data: list[dict],
    all_conv_features: list[dict],
) -> list[dict]:
    """Evaluate bare RRF (no model) across all conversations."""
    all_metrics = []
    for i in range(len(all_conv_data)):
        metrics = evaluate_ranking(
            all_conv_data[i], all_search_data[i],
            all_conv_features[i], None, [],
        )
        for m in metrics:
            m["test_conv"] = i
        all_metrics.extend(metrics)
    return all_metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def summarize_metrics(metrics: list[dict], label: str) -> dict:
    """Aggregate per-question metrics into summary."""
    if not metrics:
        return {}
    n = len(metrics)
    summary = {
        "label": label,
        "n": n,
        "mrr": sum(m["mrr"] for m in metrics) / n,
        "ndcg@10": sum(m.get("ndcg@10", 0) for m in metrics) / n,
    }
    for k in [1, 3, 5, 10, 20]:
        key = f"r@{k}"
        summary[key] = sum(m.get(key, 0) for m in metrics) / n
    return summary


def print_results_table(results: list[dict]):
    """Print comparison table of ablation results."""
    if not results:
        return

    header = f"{'Config':<15} {'N':>5} {'MRR':>7} {'NDCG@10':>8} {'R@1':>7} {'R@3':>7} {'R@5':>7} {'R@10':>7} {'R@20':>7}"
    print(f"\n{'=' * 83}")
    print("LoCoMo Reranker Ablation Results (Leave-One-Conv-Out CV)")
    print(f"{'=' * 83}")
    print(header)
    print("-" * 83)

    for r in results:
        row = f"{r['label']:<15} {r['n']:>5} {r['mrr']:>7.3f} {r.get('ndcg@10', 0):>8.3f}"
        for k in [1, 3, 5, 10, 20]:
            row += f" {r.get(f'r@{k}', 0):>7.1%}"
        print(row)

    # Delta from bare RRF (first row)
    if len(results) > 1:
        base = results[0]
        print("-" * 83)
        print("Delta vs bare RRF:")
        for r in results[1:]:
            row = f"  {r['label']:<13}"
            row += f" {'':>5} {r['mrr'] - base['mrr']:>+7.3f} {r.get('ndcg@10', 0) - base.get('ndcg@10', 0):>+8.3f}"
            for k in [1, 3, 5, 10, 20]:
                delta = r.get(f"r@{k}", 0) - base.get(f"r@{k}", 0)
                row += f" {delta:>+7.1%}"
            print(row)


def _feature_group_label(feature_idx: int) -> str:
    """Return the group letter for a feature index."""
    for group, indices in FEATURE_GROUPS.items():
        if feature_idx in indices:
            return group
    return "?"


def print_importance_table(importances: np.ndarray, feature_indices: list[int]):
    """Print feature importance from full model."""
    print(f"\n{'=' * 55}")
    print("Feature Importance (gain, full model)")
    print(f"{'=' * 55}")

    names = [FEATURE_NAMES[i] for i in feature_indices]
    order = np.argsort(importances)[::-1]
    for idx in order:
        group = _feature_group_label(feature_indices[idx])
        print(f"  [{group}] {names[idx]:20s}: {importances[idx]:8.1f}")


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

# Dead features (always excluded from selection)
_DEAD_FEATURES = {6, 13, 14}  # proximity, age_days, theme_count


def _cv_score(all_conv_features, all_conv_data, all_search_data,
              feature_indices, n_estimators=500):
    """Quick CV score (NDCG@10) for a feature set."""
    if not feature_indices:
        return 0.0
    metrics, _ = train_cv(all_conv_features, all_conv_data, all_search_data,
                          sorted(feature_indices), n_estimators)
    if not metrics:
        return 0.0
    return np.mean([m["ndcg@10"] for m in metrics])


def select_correlation_filter(all_conv_features, threshold=0.85):
    """Remove features highly correlated with a better feature (by univariate gain).

    Returns list of surviving feature indices.
    """
    # Concatenate all features
    X = np.concatenate([fd["features"] for fd in all_conv_features], axis=0)
    y = np.concatenate([fd["labels"] for fd in all_conv_features], axis=0)

    candidates = [i for i in range(NUM_FEATURES) if i not in _DEAD_FEATURES]

    # Compute pairwise correlations
    X_sub = X[:, candidates]
    corr = np.corrcoef(X_sub.T)

    # Compute univariate importance (variance of feature weighted by label)
    importances = {}
    for ci, fi in enumerate(candidates):
        pos_mask = y > 0
        if pos_mask.sum() > 0 and (~pos_mask).sum() > 0:
            # Simple: difference in means between positive and negative
            importances[fi] = abs(X[pos_mask, fi].mean() - X[~pos_mask, fi].mean())
        else:
            importances[fi] = 0.0

    # Greedily remove correlated features (keep the more important one)
    removed = set()
    for ci in range(len(candidates)):
        if candidates[ci] in removed:
            continue
        for cj in range(ci + 1, len(candidates)):
            if candidates[cj] in removed:
                continue
            if abs(corr[ci, cj]) > threshold:
                fi, fj = candidates[ci], candidates[cj]
                drop = fj if importances.get(fi, 0) >= importances.get(fj, 0) else fi
                removed.add(drop)

    survivors = [i for i in candidates if i not in removed]

    print(f"\n{'=' * 60}")
    print(f"Correlation Filter (threshold={threshold})")
    print(f"{'=' * 60}")
    print(f"Candidates: {len(candidates)} (excluding {len(_DEAD_FEATURES)} dead)")
    print(f"Removed: {len(removed)} correlated features")
    for r in sorted(removed):
        # Find what it was correlated with
        ri = candidates.index(r)
        correlated_with = []
        for ci, fi in enumerate(candidates):
            if fi != r and fi not in removed and abs(corr[ri, ci]) > threshold:
                correlated_with.append(f"{FEATURE_NAMES[fi]} (r={corr[ri, ci]:.2f})")
        print(f"  Dropped {FEATURE_NAMES[r]:25s} correlated with: {', '.join(correlated_with)}")
    print(f"Survivors ({len(survivors)}): {[FEATURE_NAMES[i] for i in survivors]}")
    return survivors


def select_forward_stepwise(all_conv_features, all_conv_data, all_search_data,
                            candidates=None, n_estimators=500, seed=None):
    """Greedy forward selection: add the feature that most improves CV MRR.

    If seed is provided (list of feature indices), start with those features
    already selected and build from there.
    """
    if candidates is None:
        candidates = [i for i in range(NUM_FEATURES) if i not in _DEAD_FEATURES]

    selected = list(seed) if seed else []
    remaining = [i for i in candidates if i not in selected]
    best_score = 0.0
    history = []

    print(f"\n{'=' * 60}")
    print("Forward Stepwise Selection")
    print(f"{'=' * 60}")
    if selected:
        print(f"Seed: {[FEATURE_NAMES[i] for i in selected]}")
        best_score = _cv_score(all_conv_features, all_conv_data, all_search_data,
                               selected, n_estimators)
        print(f"Seed NDCG@10: {best_score:.4f}")
    print(f"Candidate pool: {len(remaining)} features")

    step = 0
    while remaining:
        step += 1
        best_feat = None
        best_new_score = best_score

        for feat in remaining:
            trial = selected + [feat]
            score = _cv_score(all_conv_features, all_conv_data, all_search_data,
                              trial, n_estimators)
            if score > best_new_score:
                best_new_score = score
                best_feat = feat

        if best_feat is None:
            print(f"  Step {step}: no improvement, stopping")
            break

        delta = best_new_score - best_score
        selected.append(best_feat)
        remaining.remove(best_feat)
        best_score = best_new_score
        history.append((best_feat, best_score, delta))
        print(f"  Step {step}: +{FEATURE_NAMES[best_feat]:25s} NDCG@10={best_score:.4f} (+{delta:.4f})")

    print(f"\nFinal set ({len(selected)}): {[FEATURE_NAMES[i] for i in selected]}")
    print(f"Final NDCG@10: {best_score:.4f}")
    return selected, history


def select_backward_elimination(all_conv_features, all_conv_data, all_search_data,
                                candidates=None, n_estimators=500):
    """Greedy backward elimination: remove the feature whose removal hurts least."""
    if candidates is None:
        candidates = [i for i in range(NUM_FEATURES) if i not in _DEAD_FEATURES]

    selected = list(candidates)
    best_score = _cv_score(all_conv_features, all_conv_data, all_search_data,
                           selected, n_estimators)
    history = []

    print(f"\n{'=' * 60}")
    print("Backward Elimination")
    print(f"{'=' * 60}")
    print(f"Starting with {len(selected)} features, NDCG@10={best_score:.4f}")

    step = 0
    while len(selected) > 1:
        step += 1
        best_drop = None
        best_new_score = -1.0

        for feat in selected:
            trial = [f for f in selected if f != feat]
            score = _cv_score(all_conv_features, all_conv_data, all_search_data,
                              trial, n_estimators)
            if score > best_new_score:
                best_new_score = score
                best_drop = feat

        if best_new_score < best_score:
            print(f"  Step {step}: any removal hurts, stopping")
            break

        delta = best_new_score - best_score
        selected.remove(best_drop)
        best_score = best_new_score
        history.append((best_drop, best_score, delta))
        print(f"  Step {step}: -{FEATURE_NAMES[best_drop]:25s} NDCG@10={best_score:.4f} ({delta:+.4f})")

    print(f"\nFinal set ({len(selected)}): {[FEATURE_NAMES[i] for i in selected]}")
    print(f"Final NDCG@10: {best_score:.4f}")
    return selected, history


def select_permutation_importance(all_conv_features, all_conv_data, all_search_data,
                                  candidates=None, n_estimators=500, n_repeats=5):
    """Rank features by permutation importance (CV MRR drop when shuffled)."""
    if candidates is None:
        candidates = [i for i in range(NUM_FEATURES) if i not in _DEAD_FEATURES]

    # Train with all candidates
    baseline_score = _cv_score(all_conv_features, all_conv_data, all_search_data,
                               candidates, n_estimators)

    print(f"\n{'=' * 60}")
    print("Permutation Importance")
    print(f"{'=' * 60}")
    print(f"Baseline NDCG@10 (all {len(candidates)} features): {baseline_score:.4f}")
    print(f"Shuffling each feature {n_repeats}x...")

    importances = {}
    for feat in candidates:
        drops = []
        for rep in range(n_repeats):
            # Shuffle this feature across all conversations
            shuffled = []
            for fd in all_conv_features:
                fd_copy = {
                    "features": fd["features"].copy(),
                    "labels": fd["labels"],
                    "query_ids": fd["query_ids"],
                    "memory_ids": fd["memory_ids"],
                    "feature_names": fd["feature_names"],
                }
                rng = np.random.RandomState(42 + rep)
                rng.shuffle(fd_copy["features"][:, feat])
                shuffled.append(fd_copy)

            score = _cv_score(shuffled, all_conv_data, all_search_data,
                              candidates, n_estimators)
            drops.append(baseline_score - score)

        mean_drop = np.mean(drops)
        std_drop = np.std(drops)
        importances[feat] = (mean_drop, std_drop)

    # Sort by importance (biggest drop = most important)
    ranked = sorted(importances.items(), key=lambda x: -x[1][0])
    print(f"\n{'Feature':30s} {'NDCG Drop':>10s} {'Std':>8s} {'Signal?':>8s}")
    print("-" * 60)
    for feat, (drop, std) in ranked:
        signal = "YES" if drop > std else ("maybe" if drop > 0 else "no")
        print(f"  {FEATURE_NAMES[feat]:28s} {drop:>+10.4f} {std:>8.4f} {signal:>8s}")

    return ranked


def select_mrmr(all_conv_features, top_k=15):
    """Minimum Redundancy Maximum Relevance feature selection."""
    X = np.concatenate([fd["features"] for fd in all_conv_features], axis=0)
    y = np.concatenate([fd["labels"] for fd in all_conv_features], axis=0)

    candidates = [i for i in range(NUM_FEATURES) if i not in _DEAD_FEATURES]

    # Relevance: mutual information approximation (F-statistic as proxy)
    relevance = {}
    for fi in candidates:
        pos_mask = y > 0
        if pos_mask.sum() > 0 and (~pos_mask).sum() > 0:
            pos_mean = X[pos_mask, fi].mean()
            neg_mean = X[~pos_mask, fi].mean()
            pooled_var = X[:, fi].var() + 1e-9
            relevance[fi] = abs(pos_mean - neg_mean) / np.sqrt(pooled_var)
        else:
            relevance[fi] = 0.0

    # Greedy mRMR selection
    selected = []
    remaining = list(candidates)

    print(f"\n{'=' * 60}")
    print(f"mRMR Selection (top {top_k})")
    print(f"{'=' * 60}")

    for step in range(min(top_k, len(candidates))):
        best_feat = None
        best_score = -float("inf")

        for feat in remaining:
            rel = relevance[feat]
            # Redundancy: mean absolute correlation with already selected
            if selected:
                red = np.mean([abs(np.corrcoef(X[:, feat], X[:, s])[0, 1])
                               for s in selected])
            else:
                red = 0.0
            score = rel - red
            if score > best_score:
                best_score = score
                best_feat = feat

        if best_feat is None:
            break

        selected.append(best_feat)
        remaining.remove(best_feat)
        print(f"  {step+1:2d}. {FEATURE_NAMES[best_feat]:25s} "
              f"rel={relevance[best_feat]:.3f} mrmr={best_score:.3f}")

    print(f"\nSelected ({len(selected)}): {[FEATURE_NAMES[i] for i in selected]}")
    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train LoCoMo-specific LightGBM reranker with ablation")
    parser.add_argument("--conversations", type=int, nargs="+",
                        default=list(range(10)),
                        help="Conversation indices (default: 0-9)")
    parser.add_argument("--features", nargs="+",
                        choices=list(FEATURE_GROUPS.keys()),
                        help="Feature groups to use (default: all)")
    parser.add_argument("--config", type=str,
                        choices=list(ABLATION_CONFIGS.keys()),
                        help="Named ablation config (overrides --features)")
    parser.add_argument("--ablation", action="store_true",
                        help="Run full ablation matrix")
    parser.add_argument("--extract-only", action="store_true",
                        help="Extract features without training")
    parser.add_argument("--train-only", action="store_true",
                        help="Train from saved features")
    parser.add_argument("--save-model", action="store_true",
                        help="Save final model trained on all data")
    parser.add_argument("--n-estimators", type=int, default=500,
                        help="Max boosting rounds (default: 500)")
    parser.add_argument("--select", type=str, nargs="?", const="all",
                        choices=["all", "corr", "forward", "backward", "permutation", "mrmr"],
                        help="Run feature selection: all (runs all methods), corr (correlation filter), "
                             "forward (stepwise forward), backward (stepwise backward), "
                             "permutation (permutation importance), mrmr (min-redundancy max-relevance)")
    parser.add_argument("--seed-features", type=str, nargs="+",
                        help="Seed features for forward stepwise (by name)")
    parser.add_argument("--exclude", type=str, nargs="+",
                        help="Features to exclude from selection (by name)")

    args = parser.parse_args()

    t_start = time.time()

    # Determine feature indices
    if args.config:
        config_val = ABLATION_CONFIGS[args.config]
        if config_val and isinstance(config_val[0], str):
            feature_indices = []
            for g in config_val:
                feature_indices.extend(FEATURE_GROUPS[g])
            feature_indices.sort()
        else:
            feature_indices = sorted(config_val)
    elif args.features:
        feature_indices = []
        for g in args.features:
            feature_indices.extend(FEATURE_GROUPS[g])
        feature_indices.sort()
    else:
        feature_indices = list(range(NUM_FEATURES))  # all

    if args.train_only:
        # Load saved features
        if not FEATURES_PATH.exists():
            print(f"No saved features at {FEATURES_PATH}. Run --extract-only first.")
            sys.exit(1)
        print(f"Loading features from {FEATURES_PATH}...")
        with open(FEATURES_PATH, "rb") as f:
            saved = pickle.load(f)
        all_conv_features = saved["conv_features"]
        all_conv_data = saved["conv_data"]
        all_search_data = saved["search_data"]
    else:
        # Extract features for all conversations
        print(f"Loading and extracting features for conversations: {args.conversations}")

        # Ensure API key is available before embedding
        import os
        if not os.environ.get("OPENAI_API_KEY"):
            for key_path in [
                DATA_DIR / "openai_api_key",
                Path.home() / ".claude" / "secrets" / "openai_api_key",
                Path.home() / ".claude" / "data" / "openai_api_key",
            ]:
                if key_path.exists():
                    os.environ["OPENAI_API_KEY"] = key_path.read_text().strip()
                    break

        # Warm embedding cache
        from locomo_bench.config import BenchConfig
        from locomo_bench.ingest import extract_questions, load_locomo
        from eval_retrieval import warm_embed_cache

        bench_config = BenchConfig()
        data = load_locomo(bench_config.locomo_data)
        all_questions = extract_questions(data)
        query_texts = [q["question"] for q in all_questions
                       if q["conv_id"] in args.conversations]
        warm_embed_cache(query_texts)

        all_conv_data = []
        all_search_data = []
        all_conv_features = []

        for conv_idx in args.conversations:
            print(f"\n--- Conversation {conv_idx} ---")

            conv_data = load_conversation_data(conv_idx)
            if conv_data is None:
                continue

            print(f"  {len(conv_data['memories'])} memories, "
                  f"{len(conv_data['questions'])} questions")

            search_data = run_searches(conv_data)
            print(f"  Searches complete")

            feature_data = extract_features(conv_data, search_data)
            n_pos = np.sum(feature_data["labels"] > 0)
            print(f"  Features: {feature_data['features'].shape[0]} pairs, "
                  f"{n_pos} positive ({100*n_pos/max(len(feature_data['labels']),1):.1f}%)")

            all_conv_data.append(conv_data)
            all_search_data.append(search_data)
            all_conv_features.append(feature_data)

        # Save features
        print(f"\nSaving features to {FEATURES_PATH}...")
        FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FEATURES_PATH, "wb") as f:
            pickle.dump({
                "conv_features": all_conv_features,
                "conv_data": all_conv_data,
                "search_data": all_search_data,
            }, f)

        if args.extract_only:
            total_pairs = sum(len(fd["labels"]) for fd in all_conv_features)
            total_pos = sum(np.sum(fd["labels"] > 0) for fd in all_conv_features)
            print(f"\nExtraction complete: {total_pairs} pairs, "
                  f"{total_pos} positive ({100*total_pos/max(total_pairs,1):.1f}%)")
            print(f"Saved to {FEATURES_PATH}")
            return

    print(f"\nLoaded {len(all_conv_features)} conversations")
    total_pairs = sum(len(fd["labels"]) for fd in all_conv_features)
    total_pos = sum(int(np.sum(fd["labels"] > 0)) for fd in all_conv_features)
    print(f"Total: {total_pairs} pairs, {total_pos} positive "
          f"({100*total_pos/max(total_pairs,1):.1f}%)")

    # Feature selection mode
    if args.select:
        if args.exclude:
            for name in args.exclude:
                if name not in FEATURE_NAMES:
                    print(f"ERROR: unknown feature '{name}'. Available: {FEATURE_NAMES}")
                    return
                _DEAD_FEATURES.add(FEATURE_NAMES.index(name))
            print(f"Excluding: {args.exclude}")

        methods = (["corr", "forward", "backward", "permutation", "mrmr"]
                   if args.select == "all" else [args.select])

        for method in methods:
            if method == "corr":
                select_correlation_filter(all_conv_features)
            elif method == "forward":
                seed_indices = None
                if args.seed_features:
                    seed_indices = []
                    for name in args.seed_features:
                        if name not in FEATURE_NAMES:
                            print(f"ERROR: unknown feature '{name}'. Available: {FEATURE_NAMES}")
                            return
                        seed_indices.append(FEATURE_NAMES.index(name))
                select_forward_stepwise(all_conv_features, all_conv_data,
                                        all_search_data, n_estimators=args.n_estimators,
                                        seed=seed_indices)
            elif method == "backward":
                select_backward_elimination(all_conv_features, all_conv_data,
                                            all_search_data, n_estimators=args.n_estimators)
            elif method == "permutation":
                select_permutation_importance(all_conv_features, all_conv_data,
                                              all_search_data, n_estimators=args.n_estimators)
            elif method == "mrmr":
                select_mrmr(all_conv_features)

        elapsed = time.time() - t_start
        print(f"\nFeature selection completed in {elapsed:.1f}s")
        return

    all_results = []

    # Bare RRF baseline
    print("\nEvaluating bare RRF baseline...")
    bare_metrics = evaluate_bare_rrf(all_conv_data, all_search_data, all_conv_features)
    bare_summary = summarize_metrics(bare_metrics, "bare_rrf")
    all_results.append(bare_summary)

    if args.ablation:
        # Run all ablation configs
        for config_name, config_val in ABLATION_CONFIGS.items():
            # Support both group name lists and raw index lists
            if config_val and isinstance(config_val[0], str):
                indices = []
                for g in config_val:
                    indices.extend(FEATURE_GROUPS[g])
                indices.sort()
            else:
                indices = sorted(config_val)

            print(f"\nTraining: {config_name} (features: {[FEATURE_NAMES[i] for i in indices]})")
            metrics, importances = train_cv(
                all_conv_features, all_conv_data, all_search_data,
                indices, args.n_estimators,
            )
            summary = summarize_metrics(metrics, config_name)
            all_results.append(summary)

        # Feature importance from full model
        all_indices = list(range(NUM_FEATURES))
        print("\nTraining full model for feature importance...")
        _, full_importances = train_cv(
            all_conv_features, all_conv_data, all_search_data,
            all_indices, args.n_estimators,
        )
        print_importance_table(full_importances, all_indices)

    else:
        # Single training run with specified features
        print(f"\nTraining with features: {[FEATURE_NAMES[i] for i in feature_indices]}")
        metrics, importances = train_cv(
            all_conv_features, all_conv_data, all_search_data,
            feature_indices, args.n_estimators,
        )
        summary = summarize_metrics(metrics, "reranker")
        all_results.append(summary)
        print_importance_table(importances, feature_indices)

    # Print comparison table
    print_results_table(all_results)

    # Save model if requested
    if args.save_model:
        print(f"\nTraining final model on all data...")
        model = train_full_model(all_conv_features, feature_indices, args.n_estimators)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                "model": model,
                "feature_names": [FEATURE_NAMES[i] for i in feature_indices],
                "feature_indices": feature_indices,
            }, f)
        print(f"Model saved to {MODEL_PATH}")

    elapsed = time.time() - t_start
    print(f"\nCompleted in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
