# /// script
# requires-python = ">=3.11"
# dependencies = ["sqlite-vec>=0.1.6", "openai>=2.0.0", "numpy>=1.26", "tiktoken", "lightgbm>=4.0,<4.7"]
# ///
"""
Unified retrieval quality evaluation.

Runs the same impl_recall pipeline with different scoring configs and datasets,
measuring R@k/MRR (LoCoMo) or NDCG@k (production GT). Outputs per-question
JSONL and a summary table.

Scoring configs:
  bare     — pure FTS+vector+RRF, no boosts (THEME_BOOST=0, UCB=0, Hebbian=0, PPR=0)
  formula  — all formula constants at production values
  reranker — learned model replaces formula

Datasets:
  locomo   — LoCoMo conv 0 (or other convs), evidence-based R@k/MRR
  production — production GT queries, graded relevance NDCG@k

Usage:
  # LoCoMo bare vs formula (confirm features are inert)
  uv run scripts/locomo_bench/eval_retrieval.py --dataset locomo --configs bare formula

  # Production feature contribution
  uv run scripts/locomo_bench/eval_retrieval.py --dataset production --configs bare formula reranker

  # LoCoMo enriched ingestion
  uv run scripts/locomo_bench/eval_retrieval.py --dataset locomo --configs bare --enrich

  # Custom recall limit sweep
  uv run scripts/locomo_bench/eval_retrieval.py --dataset locomo --configs bare --recall-limits 10 20 50
"""

import argparse
import json
import logging
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

# Setup: add src/ to path for memory package imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoCoMo reranker integration
# ---------------------------------------------------------------------------

_locomo_model = None
_locomo_feature_indices = None
_locomo_feature_names = None

_expansion_flags: dict[str, bool] = {}
_n_seeds: int = 10
_graph_resolve: bool = False
_synthetic_coverage: dict | None = None  # {qkey: {synth_id: [covered_turn_ids]}}

# Per-conversation cache for reranker (avoids reloading metadata per query)
_conv_cache: dict = {}  # keys: memories, all_speakers, ordinal_map, syn_to_turns, db_path


def enable_expansion(**kwargs):
    """Enable specific expansion methods for two-phase reranking."""
    _expansion_flags.update(kwargs)


def _resolve_synthetic_nodes(db, candidate_list, preds, memories, all_speakers,
                             syn_to_turns=None):
    """Replace synthetic nodes with their source turns via EXTRACTED_FROM edges.

    Uses pre-cached syn_to_turns mapping (built once per conversation) instead of
    per-query DB lookups. The memories dict already has is_synthetic flags and
    metadata for all turns (loaded at conversation start).

    Returns (resolved_candidate_list, resolved_preds, newly_added_turn_ids).
    """
    # Identify synthetic candidates from pre-cached flag
    synthetic_ids = set()
    for mid in candidate_list:
        mem = memories.get(mid)
        if mem and mem.get("is_synthetic", False):
            synthetic_ids.add(mid)

    if not synthetic_ids:
        return candidate_list, preds, set()

    # Build score map
    score_map = dict(zip(candidate_list, preds))

    # Resolve synthetic → source turns using pre-cached mapping
    existing_turns = set(candidate_list) - synthetic_ids
    new_turn_ids = set()

    if syn_to_turns is None:
        syn_to_turns = {}

    for syn_id in synthetic_ids:
        syn_score = score_map[syn_id]
        for turn_id in syn_to_turns.get(syn_id, set()):
            if turn_id in existing_turns:
                score_map[turn_id] = max(score_map.get(turn_id, 0), syn_score)
            elif turn_id in memories:
                score_map[turn_id] = max(score_map.get(turn_id, 0), syn_score)
                new_turn_ids.add(turn_id)
                existing_turns.add(turn_id)

    # Remove synthetic nodes, keep only turns
    for syn_id in synthetic_ids:
        score_map.pop(syn_id, None)

    resolved_list = sorted(score_map.keys())
    resolved_preds = [score_map[mid] for mid in resolved_list]
    return resolved_list, resolved_preds, new_turn_ids


def _install_locomo_reranker():
    """Load LoCoMo reranker model and monkey-patch memory.reranker.rerank."""
    import pickle

    import memory.reranker

    global _locomo_model, _locomo_feature_indices, _locomo_feature_names

    benchmark_dir = Path.home() / ".somnigraph" / "benchmark"
    model_path = benchmark_dir / "locomo_reranker_model.pkl"
    if not model_path.exists():
        logger.error("No LoCoMo reranker model at %s. "
                      "Run train_locomo_reranker.py --save-model first.", model_path)
        # Fall back to disabling reranker
        memory.reranker._cache = {"model": None, "feature_names": None, "memory_meta": None, "meta_mem_count": -1, "loaded": True, "failed": True}
        return

    if _locomo_model is None:
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        sklearn_model = data["model"]
        # Extract underlying Booster to avoid scikit-learn dependency
        _locomo_model = sklearn_model.booster_
        _locomo_feature_indices = data["feature_indices"]
        _locomo_feature_names = data["feature_names"]
        logger.info("Loaded LoCoMo reranker (%d features)", len(_locomo_feature_names))

    # Monkey-patch: make the production reranker return predictions from our model
    # by installing a custom rerank function
    _original_rerank = memory.reranker.rerank

    def _locomo_rerank(db, query, fts_ranked, vec_ranked, fts_scores,
                       vec_distances, theme_ranked, theme_overlap_map,
                       feedback_raw, hebb_data, ppr_cache):
        """LoCoMo reranker: extract 17 features and predict."""
        logger.debug("_locomo_rerank: query=%s candidates=%d",
                     query[:50], len(set(fts_ranked) | set(vec_ranked) | set(theme_ranked)))
        from locomo_bench.train_locomo_reranker import (
            FEATURE_NAMES as LOCOMO_FEATURES,
        )

        candidate_ids = set(fts_ranked) | set(vec_ranked) | set(theme_ranked)
        if not candidate_ids:
            return [], {}

        try:
            return _locomo_rerank_inner(
                db, query, fts_ranked, vec_ranked, fts_scores,
                vec_distances, theme_ranked, theme_overlap_map,
                candidate_ids)
        except Exception as e:
            logger.error("_locomo_rerank FAILED: %s", e, exc_info=True)
            return None  # fall back to RRF

    def _load_conv_cache(db):
        """Load all per-conversation metadata once; reuse across queries."""
        from locomo_bench.train_locomo_reranker import (
            _extract_entities, _count_temporal_exprs,
        )
        import numpy as _np
        import re as _re
        from collections import defaultdict as _ddict

        rows = db.execute("""
            SELECT m.id, m.content, m.themes, m.token_count, m.created_at,
                   m.source, v.embedding
            FROM memories m
            LEFT JOIN memory_vec v ON v.rowid = m.rowid
            WHERE m.status = 'active'
        """).fetchall()

        # First pass: collect speakers
        all_speakers = set()
        parsed_rows = []
        for r in rows:
            content = r["content"] or ""
            sp_match = _re.match(r"^\[([^\]]+)\]", content)
            speaker = sp_match.group(1).lower() if sp_match else ""
            if speaker:
                all_speakers.add(speaker)
            parsed_rows.append((r, content, speaker))

        memories = {}
        from memory.vectors import deserialize_f32 as _deser_init
        from datetime import datetime, timezone
        for r, content, speaker in parsed_rows:
            themes_list = []
            if r["themes"]:
                try:
                    themes_list = json.loads(r["themes"])
                except json.JSONDecodeError:
                    pass
            try:
                created = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
                age_days = (datetime.now(timezone.utc) - created).total_seconds() / 86400
            except Exception:
                age_days = 0.0

            session = None
            for t in themes_list:
                if isinstance(t, str) and t.startswith("session_"):
                    session = t
                    break

            emb = None
            if r["embedding"]:
                try:
                    emb = _np.array(_deser_init(r["embedding"]), dtype=_np.float32)
                except Exception:
                    pass
            memories[r["id"]] = {
                "content": content,
                "content_tokens": content.lower().split(),
                "speaker": speaker,
                "theme_tokens": {tok for t in themes_list for tok in str(t).lower().replace("-", " ").split()},
                "theme_count": len(themes_list),
                "token_count": r["token_count"] or len(content.split()),
                "age_days": age_days,
                "entities": _extract_entities(content, all_speakers),
                "session": session,
                "embedding": emb,
                "is_synthetic": r["source"] == "extraction",
            }

        # Graph data
        _graph_edges = _ddict(int)
        _graph_syn_targets = _ddict(set)
        _graph_coref_nbrs = _ddict(set)
        _syn_ids = set()

        try:
            _edge_rows = db.execute("""
                SELECT source_id, target_id, linking_context
                FROM memory_edges
            """).fetchall()
            for _er in _edge_rows:
                _src, _tgt, _ctx = _er["source_id"], _er["target_id"], _er["linking_context"] or ""
                _graph_edges[_src] += 1
                _graph_edges[_tgt] += 1
                if _ctx.startswith("extracted_from:"):
                    _graph_syn_targets[_tgt].add(_src)
                    _syn_ids.add(_src)
                elif _ctx.startswith("claim_coref:"):
                    _graph_coref_nbrs[_src].add(_tgt)
                    _graph_coref_nbrs[_tgt].add(_src)
        except Exception:
            pass

        _syn_to_turns = _ddict(set)
        for turn_id, syn_set in _graph_syn_targets.items():
            for syn_id in syn_set:
                _syn_to_turns[syn_id].add(turn_id)

        for _mid in memories:
            memories[_mid]["graph_edge_count"] = _graph_edges.get(_mid, 0)
            memories[_mid]["graph_synthetic_ids"] = _graph_syn_targets.get(_mid, set())
            memories[_mid]["graph_coref_neighbors"] = _graph_coref_nbrs.get(_mid, set())

        sorted_mids = sorted(memories.keys(), key=lambda m: memories[m]["age_days"], reverse=True)
        ordinal_map = {mid: i for i, mid in enumerate(sorted_mids)}

        # Build rowid -> memory_id map for FTS/vec lookups
        rowid_map = {}
        for r in db.execute("SELECT rowid, memory_id FROM memory_rowid_map").fetchall():
            rowid_map[r["rowid"]] = r["memory_id"]

        return {
            "memories": memories,
            "all_speakers": all_speakers,
            "ordinal_map": ordinal_map,
            "syn_to_turns": dict(_syn_to_turns),
            "rowid_map": rowid_map,
        }

    def _locomo_rerank_inner(db, query, fts_ranked, vec_ranked, fts_scores,
                              vec_distances, theme_ranked, theme_overlap_map,
                              candidate_ids):
        from memory.reranker import _compute_proximity
        from locomo_bench.train_locomo_reranker import (
            _extract_entities, NUM_FEATURES, _count_temporal_exprs,
        )
        import numpy as _np
        RRF_K = 60

        # Use per-conversation cache (loaded once, reused across queries)
        global _conv_cache
        import memory.db as _db_mod
        db_path = str(_db_mod.DB_PATH)
        if _conv_cache.get("db_path") != db_path:
            logger.info("  Loading conv cache for %s (%d memories in DB)",
                        db_path, db.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
            _conv_cache = _load_conv_cache(db)
            _conv_cache["db_path"] = db_path

        memories = _conv_cache["memories"]
        all_speakers = _conv_cache["all_speakers"]
        ordinal_map = _conv_cache["ordinal_map"]
        _syn_to_turns = _conv_cache["syn_to_turns"]
        _rowid_map = _conv_cache["rowid_map"]

        query_tokens = set(query.lower().split())
        query_terms = [t for t in query.lower().split() if len(t) > 1]
        query_lower = query.lower()
        query_entities = _extract_entities(query, all_speakers)

        # RRF scores for all candidates (including synthetic — needed for graph_synthetic_score)
        rrf_scores = {}
        for mid in candidate_ids:
            score = 0.0
            if mid in fts_ranked:
                score += 1.0 / (RRF_K + fts_ranked[mid])
            if mid in vec_ranked:
                score += 1.0 / (RRF_K + vec_ranked[mid])
            rrf_scores[mid] = score

        # Filter synthetic nodes from candidate pool (they bridged in Phase 1, now drop)
        # Their RRF scores remain available for graph_synthetic_score computation
        if _graph_resolve:
            candidate_ids = {mid for mid in candidate_ids
                             if not memories.get(mid, {}).get("is_synthetic", False)}

        all_rrf = sorted(rrf_scores.values())
        n_rrf = len(all_rrf)

        # Session co-occurrence: top-20 RRF candidates (synthetic already removed)
        top20_rrf = sorted(candidate_ids, key=lambda m: rrf_scores.get(m, 0), reverse=True)[:20]
        top20_set = set(top20_rrf)
        from collections import defaultdict as _defaultdict
        session_counts = _defaultdict(int)
        for mid in top20_rrf:
            mem = memories.get(mid)
            if mem and mem.get("session"):
                session_counts[mem["session"]] += 1

        candidate_ordinals = {mid: ordinal_map.get(mid, -999) for mid in candidate_ids}
        ordinal_set = set(candidate_ordinals.values())

        candidate_list = sorted(candidate_ids)
        features = _np.zeros((len(candidate_list), NUM_FEATURES), dtype=_np.float32)

        for i, mid in enumerate(candidate_list):
            mem = memories.get(mid)
            if not mem:
                continue
            fts_r = fts_ranked.get(mid, -1)
            vec_r = vec_ranked.get(mid, -1)

            features[i, 0] = fts_r
            features[i, 1] = vec_r
            features[i, 2] = fts_scores.get(mid, 0.0)
            features[i, 3] = vec_distances.get(mid, 0.0)
            features[i, 4] = theme_overlap_map.get(mid, 0)

            if query_terms:
                content_set = set(mem["content_tokens"])
                matched = sum(1 for t in query_terms if t in content_set)
                features[i, 5] = matched / len(query_terms)
            if len(query_terms) > 1:
                features[i, 6] = _compute_proximity(query_terms, mem["content_tokens"])
            if mem["speaker"] and mem["speaker"] in query_lower:
                features[i, 7] = 1.0
            features[i, 8] = len(query_terms)

            if fts_r >= 0 and vec_r >= 0:
                features[i, 9] = abs(fts_r - vec_r)
            else:
                features[i, 9] = 9999

            my_ord = candidate_ordinals.get(mid, -999)
            if my_ord >= 0:
                features[i, 10] = sum(1 for o in ordinal_set if o != my_ord and abs(o - my_ord) <= 2)

            rrf_s = rrf_scores.get(mid, 0.0)
            if n_rrf > 1:
                features[i, 11] = sum(1 for s in all_rrf if s <= rrf_s) / n_rrf

            features[i, 12] = mem["token_count"]
            features[i, 13] = mem["age_days"]
            features[i, 14] = mem["theme_count"]

            # Group E: Entity & session features
            features[i, 15] = len(query_entities & mem.get("entities", set()))
            mem_session = mem.get("session")
            if mem_session and mid in top20_set:
                features[i, 16] = session_counts.get(mem_session, 1) - 1

            # Group F: has_temporal_expr (17), entity_density (18)
            features[i, 17] = _count_temporal_exprs(mem["content"])
            features[i, 18] = len(mem.get("entities", set())) / max(mem["token_count"], 1)

        # Second pass: features that depend on top-10 RRF
        top10_rrf = sorted(candidate_ids, key=lambda m: rrf_scores.get(m, 0), reverse=True)[:10]
        top10_set = set(top10_rrf)

        # theme_complementarity (21): query coverage not already in top-10
        top10_query_coverage = set()
        for t10_mid in top10_rrf:
            t10_mem = memories.get(t10_mid)
            if t10_mem:
                top10_query_coverage |= set(query_terms) & set(t10_mem["content_tokens"])

        # centroid_distance (24): use cached embeddings from memories dict
        top10_embs = []
        candidate_embs = {}
        for mid in candidate_list:
            mem = memories.get(mid)
            emb = mem.get("embedding") if mem else None
            if emb is not None:
                candidate_embs[mid] = emb
                if mid in top10_set:
                    top10_embs.append(emb)

        centroid = None
        if top10_embs:
            centroid = _np.mean(top10_embs, axis=0)
            centroid_norm = float(_np.linalg.norm(centroid)) + 1e-9

        for i, mid in enumerate(candidate_list):
            mem = memories.get(mid)
            if not mem:
                continue

            # theme_complementarity (21)
            if query_terms:
                content_set = set(mem["content_tokens"])
                my_coverage = set(query_terms) & content_set
                novel = my_coverage - top10_query_coverage
                features[i, 21] = len(novel) / len(query_terms)

            # centroid_distance (24)
            emb = candidate_embs.get(mid)
            if emb is not None and centroid is not None:
                emb_norm = float(_np.linalg.norm(emb)) + 1e-9
                cos_sim = float(_np.dot(emb, centroid) / (emb_norm * centroid_norm))
                features[i, 24] = 1.0 - cos_sim

        # Group H: Phase sentinels for phase 1
        for i, mid in enumerate(candidate_list):
            features[i, 25] = -1    # entity_fts_rank sentinel
            features[i, 26] = -1    # sub_query_hit_count sentinel
            features[i, 27] = 0.0   # seed_keyword_overlap
            features[i, 28] = 0     # phase = 0
            features[i, 29] = 0     # expansion_method_count
            features[i, 30] = rrf_scores.get(mid, 0.0)  # phase1_rrf_score
            features[i, 31] = 0     # is_seed

            # Group I: Graph features
            features[i, 32] = mem.get("graph_edge_count", 0)
            syn_ids = mem.get("graph_synthetic_ids", set())
            if syn_ids:
                features[i, 33] = max(rrf_scores.get(sid, 0.0) for sid in syn_ids)
            coref_nbrs = mem.get("graph_coref_neighbors", set())
            if coref_nbrs:
                features[i, 34] = len(coref_nbrs & candidate_ids)
            features[i, 35] = 0     # is_graph_resolved (set to 1 after resolution)

        # --- Phase 1 prediction ---
        X = features[:, _locomo_feature_indices]
        preds = _locomo_model.predict(X)

        # (Graph resolution already done pre-feature-extraction above)

        # If no expansion flags, return phase 1 results directly
        if not any(_expansion_flags.values()):
            scored = sorted(zip(candidate_list, preds), key=lambda x: -x[1])
            scores_dict = {mid: float(s) for mid, s in scored}
            sorted_ids = [mid for mid, _ in scored]
            return sorted_ids, scores_dict

        # --- Phase 2: expand using phase 1 top-K as seeds ---
        from locomo_bench.expansion import (
            ExpansionContext, run_expansions,
            compute_entity_fts_ranks, compute_sub_query_hits,
            compute_seed_keyword_overlap,
        )
        from memory.embeddings import embed_text as _embed
        from memory.fts import sanitize_fts_query as _sanitize_fts2
        from memory.constants import BM25_SUMMARY_WT as _BM25_S, BM25_THEMES_WT as _BM25_T
        from memory.vectors import deserialize_f32 as _deser

        phase1_scored = sorted(zip(candidate_list, preds), key=lambda x: -x[1])
        top_seeds = [mid for mid, _ in phase1_scored[:_n_seeds]]
        top_seeds_set = set(top_seeds)

        # Copies to avoid corrupting phase 1 data
        exp_fts_ranked = dict(fts_ranked)
        exp_vec_ranked = dict(vec_ranked)
        exp_vec_distances = dict(vec_distances)
        exp_fts_scores = dict(fts_scores)
        q_emb = _np.array(_embed(query), dtype=_np.float32)

        exp_ctx = ExpansionContext(
            db=db,
            question=query,
            query_embedding=q_emb.tolist(),
            seeds=list(top_seeds),
            existing_ids=set(candidate_ids),
            fts_ranked=exp_fts_ranked,
            vec_ranked=exp_vec_ranked,
            vec_distances=exp_vec_distances,
            fts_scores=exp_fts_scores,
            all_speakers=all_speakers,
            rowid_map=_rowid_map,
            memories=memories,
        )
        exp_result = run_expansions(exp_ctx, _expansion_flags)
        exp_method_counts = exp_result.method_counts()
        expanded_ids = candidate_ids | exp_result.all_new_ids

        # Load metadata for new candidates
        new_ids = expanded_ids - set(memories.keys())
        if new_ids:
            for new_id in new_ids:
                r = db.execute("""
                    SELECT m.id, m.content, m.themes, m.token_count, m.created_at,
                           v.embedding
                    FROM memories m
                    LEFT JOIN memory_vec v ON v.rowid = m.rowid
                    WHERE m.id = ? AND m.status = 'active'
                """, (new_id,)).fetchone()
                if not r:
                    expanded_ids.discard(new_id)
                    continue
                content = r["content"] or ""
                sp_match = _re.match(r"^\[([^\]]+)\]", content)
                speaker = sp_match.group(1).lower() if sp_match else ""
                themes_list = []
                if r["themes"]:
                    try:
                        themes_list = json.loads(r["themes"])
                    except json.JSONDecodeError:
                        pass
                try:
                    from datetime import datetime as _dt2, timezone as _tz2
                    created = _dt2.fromisoformat(r["created_at"].replace("Z", "+00:00"))
                    age_days = (_dt2.now(_tz2.utc) - created).total_seconds() / 86400
                except Exception:
                    age_days = 0.0
                session = None
                for t in themes_list:
                    if isinstance(t, str) and t.startswith("session_"):
                        session = t
                        break
                token_count = r["token_count"] or len(content.split())
                entities = _extract_entities(content, all_speakers)
                emb = None
                if r["embedding"]:
                    try:
                        emb = _np.array(_deser(r["embedding"]), dtype=_np.float32)
                    except Exception:
                        pass
                memories[new_id] = {
                    "content": content,
                    "content_tokens": content.lower().split(),
                    "speaker": speaker,
                    "theme_tokens": {tok for t in themes_list for tok in str(t).lower().replace("-", " ").split()},
                    "theme_count": len(themes_list),
                    "token_count": token_count,
                    "age_days": age_days,
                    "entities": entities,
                    "session": session,
                    "embedding": emb,
                }

            sorted_mids = sorted(memories.keys(), key=lambda m: memories[m]["age_days"], reverse=True)
            ordinal_map = {mid: i for i, mid in enumerate(sorted_mids)}

        # Recompute theme_overlap for expanded candidates not in original map
        for mid in expanded_ids:
            if mid not in theme_overlap_map:
                mem = memories.get(mid)
                if mem:
                    overlap = len(query_tokens & mem["theme_tokens"])
                    if overlap > 0:
                        theme_overlap_map[mid] = overlap

        # Retrieval scores for expanded candidates
        max_vec_rank = max(exp_vec_ranked.values()) if exp_vec_ranked else 0
        max_fts_rank = max(exp_fts_ranked.values()) if exp_fts_ranked else 0
        q_norm = float(_np.linalg.norm(q_emb))

        for new_id in (expanded_ids - set(exp_fts_ranked.keys()) - set(exp_vec_ranked.keys())):
            mem = memories.get(new_id)
            if not mem:
                continue
            if new_id not in exp_vec_distances and mem.get("embedding") is not None:
                emb = mem["embedding"]
                emb_norm = float(_np.linalg.norm(emb))
                if emb_norm > 0 and q_norm > 0:
                    cos_dist = 1.0 - float(_np.dot(q_emb, emb) / (q_norm * emb_norm))
                    exp_vec_distances[new_id] = cos_dist
                    max_vec_rank += 1
                    exp_vec_ranked[new_id] = max_vec_rank
            if new_id not in exp_fts_ranked:
                _fts_q2 = _sanitize_fts2(query)
                if _fts_q2:
                    try:
                        rowid_row = db.execute(
                            "SELECT rowid FROM memory_rowid_map WHERE memory_id = ?",
                            (new_id,),
                        ).fetchone()
                        if rowid_row:
                            fts_check = db.execute(
                                f"SELECT bm25(memory_fts, {_BM25_S}, {_BM25_T}) as score "
                                f"FROM memory_fts WHERE memory_fts MATCH ? AND rowid = ?",
                                (_fts_q2, rowid_row["rowid"]),
                            ).fetchone()
                            if fts_check:
                                exp_fts_scores[new_id] = fts_check["score"]
                                max_fts_rank += 1
                                exp_fts_ranked[new_id] = max_fts_rank
                    except Exception:
                        pass

        # Expanded RRF scores
        exp_rrf_scores = {}
        for mid in expanded_ids:
            score = 0.0
            if mid in exp_fts_ranked:
                score += 1.0 / (RRF_K + exp_fts_ranked[mid])
            if mid in exp_vec_ranked:
                score += 1.0 / (RRF_K + exp_vec_ranked[mid])
            exp_rrf_scores[mid] = score

        # Compute Group H features
        entity_fts_ranks = compute_entity_fts_ranks(db, query, all_speakers, expanded_ids, rowid_map=_rowid_map)
        sub_query_hits = compute_sub_query_hits(db, query, all_speakers, expanded_ids, rowid_map=_rowid_map)
        seed_keyword_overlaps = compute_seed_keyword_overlap(db, query, top_seeds, expanded_ids, memories=memories)

        # Build phase 2 feature matrix
        exp_candidate_list = sorted(expanded_ids)
        n2 = len(exp_candidate_list)
        features2 = _np.zeros((n2, NUM_FEATURES), dtype=_np.float32)

        exp_all_rrf = sorted(exp_rrf_scores.values())
        n_exp_rrf = len(exp_all_rrf)
        exp_top20 = sorted(expanded_ids, key=lambda m: exp_rrf_scores.get(m, 0), reverse=True)[:20]
        exp_top20_set = set(exp_top20)
        exp_session_counts = _defaultdict(int)
        for mid in exp_top20:
            mem = memories.get(mid)
            if mem and mem.get("session"):
                exp_session_counts[mem["session"]] += 1

        exp_candidate_ordinals = {mid: ordinal_map.get(mid, -999) for mid in expanded_ids}
        exp_ordinal_set = set(exp_candidate_ordinals.values())

        for i, mid in enumerate(exp_candidate_list):
            mem = memories.get(mid)
            if not mem:
                continue
            fts_r = exp_fts_ranked.get(mid, -1)
            vec_r = exp_vec_ranked.get(mid, -1)

            features2[i, 0] = fts_r
            features2[i, 1] = vec_r
            features2[i, 2] = exp_fts_scores.get(mid, 0.0)
            features2[i, 3] = exp_vec_distances.get(mid, 0.0)
            features2[i, 4] = theme_overlap_map.get(mid, 0)
            if query_terms:
                content_set = set(mem["content_tokens"])
                matched = sum(1 for t in query_terms if t in content_set)
                features2[i, 5] = matched / len(query_terms)
            if len(query_terms) > 1:
                features2[i, 6] = _compute_proximity(query_terms, mem["content_tokens"])
            if mem["speaker"] and mem["speaker"] in query_lower:
                features2[i, 7] = 1.0
            features2[i, 8] = len(query_terms)
            if fts_r >= 0 and vec_r >= 0:
                features2[i, 9] = abs(fts_r - vec_r)
            else:
                features2[i, 9] = 9999
            my_ord = exp_candidate_ordinals.get(mid, -999)
            if my_ord >= 0:
                features2[i, 10] = sum(1 for o in exp_ordinal_set if o != my_ord and abs(o - my_ord) <= 2)
            rrf_s = exp_rrf_scores.get(mid, 0.0)
            if n_exp_rrf > 1:
                features2[i, 11] = sum(1 for s in exp_all_rrf if s <= rrf_s) / n_exp_rrf
            features2[i, 12] = mem["token_count"]
            features2[i, 13] = mem["age_days"]
            features2[i, 14] = mem["theme_count"]
            features2[i, 15] = len(query_entities & mem.get("entities", set()))
            mem_session = mem.get("session")
            if mem_session and mid in exp_top20_set:
                features2[i, 16] = exp_session_counts.get(mem_session, 1) - 1
            features2[i, 17] = _count_temporal_exprs(mem["content"])
            features2[i, 18] = len(mem.get("entities", set())) / max(mem["token_count"], 1)

            # Group H: real values
            features2[i, 25] = entity_fts_ranks.get(mid, -1)
            features2[i, 26] = sub_query_hits.get(mid, 0)
            features2[i, 27] = seed_keyword_overlaps.get(mid, 0.0)
            features2[i, 28] = 1
            features2[i, 29] = exp_method_counts.get(mid, 0)
            features2[i, 30] = rrf_scores.get(mid, 0.0)  # phase1_rrf_score (0 if new)
            features2[i, 31] = 1.0 if mid in top_seeds_set else 0.0

            # Group I: Graph features
            features2[i, 32] = mem.get("graph_edge_count", 0)
            syn_ids = mem.get("graph_synthetic_ids", set())
            if syn_ids:
                features2[i, 33] = max(exp_rrf_scores.get(sid, 0.0) for sid in syn_ids)
            coref_nbrs = mem.get("graph_coref_neighbors", set())
            if coref_nbrs:
                features2[i, 34] = len(coref_nbrs & expanded_ids)
            features2[i, 35] = 0  # is_graph_resolved (TODO: track resolution)

        # Group F second-pass (inter-passage based on expanded top-10)
        exp_top10 = sorted(exp_candidate_list, key=lambda m: exp_rrf_scores.get(m, 0), reverse=True)[:10]
        exp_top10_set = set(exp_top10)
        exp_top10_sessions = _defaultdict(int)
        exp_top10_entities = set()
        exp_top10_query_coverage = set()
        for mid in exp_top10:
            mem = memories.get(mid)
            if not mem:
                continue
            if mem.get("session"):
                exp_top10_sessions[mem["session"]] += 1
            exp_top10_entities |= mem.get("entities", set())
            content_set = set(mem["content_tokens"])
            exp_top10_query_coverage |= (set(query_terms) & content_set)

        # Load embeddings for Group G computation
        exp_top10_embeddings = []
        candidate_embs2 = {}
        for mid in exp_candidate_list:
            mem = memories.get(mid)
            if not mem:
                continue
            emb = mem.get("embedding")
            if emb is None:
                # Try loading from DB
                vec_row = db.execute(
                    "SELECT v.embedding FROM memory_vec v "
                    "JOIN memory_rowid_map m ON v.rowid = m.rowid "
                    "WHERE m.memory_id = ?", (mid,)
                ).fetchone()
                if vec_row and vec_row["embedding"]:
                    try:
                        emb = _np.array(_deser(vec_row["embedding"]), dtype=_np.float32)
                    except Exception:
                        pass
            if emb is not None:
                candidate_embs2[mid] = emb
                if mid in exp_top10_set:
                    exp_top10_embeddings.append(emb)

        exp_top10_all_tokens = set()
        for mid in exp_top10:
            mem = memories.get(mid)
            if mem:
                exp_top10_all_tokens |= set(mem["content_tokens"])

        exp_centroid = None
        if exp_top10_embeddings:
            exp_centroid = _np.mean(exp_top10_embeddings, axis=0)
            exp_centroid_norm = float(_np.linalg.norm(exp_centroid)) + 1e-9

        for i, mid in enumerate(exp_candidate_list):
            mem = memories.get(mid)
            if not mem:
                continue

            mem_session = mem.get("session")
            if mem_session and len(exp_top10) > 0:
                same = exp_top10_sessions.get(mem_session, 0)
                if mid in exp_top10_set:
                    same = max(same - 1, 0)
                features2[i, 19] = same / len(exp_top10)
            mem_entities = mem.get("entities", set())
            shared_with_topk = mem_entities & (exp_top10_entities - query_entities)
            features2[i, 20] = len(shared_with_topk)
            if query_terms:
                content_set = set(mem["content_tokens"])
                my_coverage = set(query_terms) & content_set
                novel = my_coverage - exp_top10_query_coverage
                features2[i, 21] = len(novel) / len(query_terms)

            emb = candidate_embs2.get(mid)
            if emb is not None and exp_centroid is not None:
                emb_norm = float(_np.linalg.norm(emb)) + 1e-9
                cos_sim = float(_np.dot(emb, exp_centroid) / (emb_norm * exp_centroid_norm))
                features2[i, 24] = 1.0 - cos_sim

            # mmr_redundancy and unique_token_frac
            if emb is not None:
                max_sim = 0.0
                for other_emb in exp_top10_embeddings:
                    sim = float(_np.dot(emb, other_emb) / ((float(_np.linalg.norm(emb)) + 1e-9) * (float(_np.linalg.norm(other_emb)) + 1e-9)))
                    if sim > max_sim:
                        max_sim = sim
                features2[i, 22] = max_sim

            mem_tokens = set(mem["content_tokens"])
            if mem_tokens:
                if mid in exp_top10_set:
                    other_tokens = set()
                    for other_mid in exp_top10:
                        if other_mid != mid:
                            other_mem = memories.get(other_mid)
                            if other_mem:
                                other_tokens |= set(other_mem["content_tokens"])
                    unique = mem_tokens - other_tokens
                else:
                    unique = mem_tokens - exp_top10_all_tokens
                features2[i, 23] = len(unique) / len(mem_tokens)

        # Phase 2 prediction
        X2 = features2[:, _locomo_feature_indices]
        preds2 = _locomo_model.predict(X2)

        # Resolve any synthetic nodes that expansion brought in
        if _graph_resolve:
            exp_candidate_list, preds2, _ = _resolve_synthetic_nodes(
                db, exp_candidate_list, preds2, memories, all_speakers,
                syn_to_turns=_syn_to_turns)

        scored = sorted(zip(exp_candidate_list, preds2), key=lambda x: -x[1])
        scores_dict = {mid: float(s) for mid, s in scored}
        sorted_ids = [mid for mid, _ in scored]

        return sorted_ids, scores_dict

    memory.reranker.rerank = _locomo_rerank
    # Also patch in tools.py where rerank is imported as reranker_rerank
    import memory.tools
    memory.tools.reranker_rerank = _locomo_rerank
    # Make the reranker appear "loaded" so impl_recall uses it
    memory.reranker._cache = {"model": True, "feature_names": None, "memory_meta": None, "meta_mem_count": -1, "loaded": True, "failed": False}


# ---------------------------------------------------------------------------
# Scoring configs
# ---------------------------------------------------------------------------

CONFIGS = {
    "bare": {
        "W_THEME": 0,
        "UCB_COEFF": 0,
        "HEBBIAN_COEFF": 0,
        "PPR_BOOST_COEFF": 0,
        "reranker": False,
    },
    "formula": {
        # All formula constants at production values (no patches needed)
        "reranker": False,
    },
    "reranker": {
        # Learned model replaces formula
        "reranker": True,
    },
    # Individual feature ablations for Phase 4
    "+theme": {
        "UCB_COEFF": 0,
        "HEBBIAN_COEFF": 0,
        "PPR_BOOST_COEFF": 0,
        "reranker": False,
    },
    "+ucb": {
        "HEBBIAN_COEFF": 0,
        "PPR_BOOST_COEFF": 0,
        "reranker": False,
    },
    "+hebbian": {
        "PPR_BOOST_COEFF": 0,
        "reranker": False,
    },
    "locomo_reranker": {
        "W_THEME": 0,
        "UCB_COEFF": 0,
        "HEBBIAN_COEFF": 0,
        "PPR_BOOST_COEFF": 0,
        "reranker": False,  # disable production reranker
        "locomo_model": True,  # use LoCoMo-specific reranker
    },
}


def apply_scoring_config(config_name: str):
    """Monkey-patch scoring constants and reranker availability for a config."""
    import memory.constants
    import memory.reranker
    import memory.scoring

    config = CONFIGS[config_name]

    _PATCHABLE = (
        "W_THEME", "K_THEME", "UCB_COEFF", "HEBBIAN_COEFF",
        "HEBBIAN_CAP", "PPR_BOOST_COEFF", "EWMA_ALPHA",
        "K_FTS", "K_VEC", "RRF_VEC_WEIGHT", "BM25_SUMMARY_WT", "BM25_THEMES_WT",
    )

    # Save originals on first call (for restoration)
    if not hasattr(apply_scoring_config, "_originals"):
        apply_scoring_config._originals = {
            attr: getattr(memory.constants, attr)
            for attr in _PATCHABLE
            if hasattr(memory.constants, attr)
        }

    # Restore originals first
    for attr, val in apply_scoring_config._originals.items():
        setattr(memory.constants, attr, val)
        if hasattr(memory.scoring, attr):
            setattr(memory.scoring, attr, val)

    # Apply patches
    for attr in _PATCHABLE:
        if attr in config:
            setattr(memory.constants, attr, config[attr])
            if hasattr(memory.scoring, attr):
                setattr(memory.scoring, attr, config[attr])

    # Handle HEBBIAN_CAP — zero it out when HEBBIAN_COEFF is 0
    if config.get("HEBBIAN_COEFF", None) == 0:
        memory.constants.HEBBIAN_CAP = 0
        if hasattr(memory.scoring, "HEBBIAN_CAP"):
            memory.scoring.HEBBIAN_CAP = 0

    # Control reranker availability
    if config.get("locomo_model", False):
        # LoCoMo-specific reranker: monkey-patch memory.reranker.rerank
        _install_locomo_reranker()
    elif config.get("reranker", False):
        # Force-load production model
        memory.reranker._cache = {"model": None, "feature_names": None, "memory_meta": None, "meta_mem_count": -1, "loaded": False, "failed": False}
        memory.reranker._load_model()
    else:
        # Disable reranker
        memory.reranker._cache = {"model": None, "feature_names": None, "memory_meta": None, "meta_mem_count": -1, "loaded": True, "failed": True}


# ---------------------------------------------------------------------------
# LoCoMo evaluation
# ---------------------------------------------------------------------------


def eval_locomo(
    config_name: str,
    conv_idx: int,
    recall_limit: int,
    recall_budget: int,
    enrich: bool,
) -> list[dict]:
    """Run LoCoMo retrieval eval with the given scoring config."""
    from locomo_bench.config import BenchConfig
    from locomo_bench.ingest import extract_questions, extract_turns, load_locomo
    from locomo_bench.run import setup_isolated_db

    bench_config = BenchConfig()
    data = load_locomo(bench_config.locomo_data)
    all_turns = extract_turns(data)
    all_questions = extract_questions(data)

    conv_turns = [t for t in all_turns if t["conv_id"] == conv_idx]
    conv_questions = [q for q in all_questions if q["conv_id"] == conv_idx]

    logger.info("LoCoMo conv %d: %d turns, %d questions", conv_idx,
                len(conv_turns), len(conv_questions))

    # Setup isolated DB
    suffix = "_enriched" if enrich else ""
    run_dir = bench_config.base_dir / f"retrieval_{conv_idx}{suffix}"
    setup_isolated_db(run_dir)

    # Apply scoring config AFTER setup_isolated_db (which resets reranker cache)
    apply_scoring_config(config_name)

    # Load or reuse ingested data
    dia_map_path = run_dir / "dia_map.json"
    from memory.db import get_db
    db = get_db()
    count = db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    db.close()

    if count > 0 and dia_map_path.exists():
        logger.info("Reusing DB: %d memories (config=%s)", count, config_name)
        dia_map = _load_dia_map(dia_map_path)
    else:
        from locomo_bench.ingest import ingest_conversation
        dia_map = ingest_conversation(
            conv_turns,
            embed_cache_path=bench_config.embed_cache,
            enrich=enrich,
        )
        with open(dia_map_path, "w") as f:
            json.dump({f"{k[0]}:{k[1]}": v for k, v in dia_map.items()}, f)

    # Auto-build graph if --graph-resolve and no synthetic nodes yet
    if _graph_resolve:
        db = get_db()
        syn_count = db.execute(
            "SELECT COUNT(*) FROM memories WHERE source = 'extraction'"
        ).fetchone()[0]
        db.close()
        if syn_count == 0:
            from locomo_bench.build_graph import build_graph, load_dia_map as _load_graph_dia_map
            logger.info("Building graph nodes for conv %d...", conv_idx)
            graph_dia_map = _load_graph_dia_map(run_dir)
            build_graph(
                conv_idx, graph_dia_map,
                embed_cache_path=bench_config.embed_cache,
            )

    # Run evaluation
    from memory.tools import impl_recall
    records = []
    t_start = time.time()

    for qi, q in enumerate(conv_questions):
        # Map evidence dia_ids to short memory IDs
        evidence_short = set()
        evidence_full_to_short = {}
        for eid in q.get("evidence", []):
            key = (q["conv_id"], eid)
            if key in dia_map:
                full_id = dia_map[key]
                short_id = full_id[:8]
                evidence_short.add(short_id)
                evidence_full_to_short[full_id] = short_id

        if not evidence_short:
            continue

        # Build synthetic coverage map for this question
        synth_covers = {}
        if _synthetic_coverage is not None:
            qkey = f"{conv_idx}:{qi}"
            cov = _synthetic_coverage.get(qkey, {})
            for synth_full_id, covered_turn_ids in cov.items():
                if covered_turn_ids:
                    short_synth = synth_full_id[:8]
                    covered_short = set()
                    for tid in covered_turn_ids:
                        if tid in evidence_full_to_short:
                            covered_short.add(evidence_full_to_short[tid])
                    if covered_short:
                        synth_covers[short_synth] = covered_short

        result = impl_recall(
            query=q["question"],
            context=q["question"],
            budget=recall_budget,
            limit=recall_limit,
            internal=True,
        )
        retrieved = _parse_ids(result)

        # Score
        scores = _score_locomo(retrieved, evidence_short, recall_limit,
                               synth_covers=synth_covers)
        records.append({
            "config": config_name,
            "dataset": "locomo",
            "conv_id": conv_idx,
            "category": q["category"],
            "question": q["question"],
            "answer": str(q["answer"]),
            "evidence": q["evidence"],
            **scores,
        })

        if (qi + 1) % 50 == 0:
            elapsed = time.time() - t_start
            logger.info("  %d/%d questions (%.1f q/s)", qi + 1,
                        len(conv_questions), (qi + 1) / elapsed)

    return records


def _score_locomo(
    retrieved: list[str],
    evidence_short: set[str],
    limit: int,
    synth_covers: dict[str, set[str]] | None = None,
) -> dict:
    """Score a single LoCoMo question.

    If synth_covers is provided, a synthetic node at rank k counts as recalling
    the GT evidence turns it covers (per LLM-judged coverage table).
    """
    first_hit = None
    for i, rid in enumerate(retrieved):
        is_hit = rid in evidence_short
        if not is_hit and synth_covers and rid in synth_covers:
            # Synthetic covers some evidence turns for this question
            is_hit = True
        if is_hit and first_hit is None:
            first_hit = i + 1

    scores = {
        "mrr": 1.0 / first_hit if first_hit else 0.0,
        "first_hit_rank": first_hit,
        "retrieved_count": len(retrieved),
        "evidence_count": len(evidence_short),
    }

    for k in [1, 3, 5, 10, 20, 50]:
        if k <= limit:
            top_k_set = set(retrieved[:k])
            hit = bool(top_k_set & evidence_short)
            if not hit and synth_covers:
                hit = bool(top_k_set & synth_covers.keys())
            scores[f"r@{k}"] = hit

    return scores


# ---------------------------------------------------------------------------
# Production GT evaluation
# ---------------------------------------------------------------------------


def eval_production(
    config_name: str,
    gt_path: Path,
    db_path: Path,
    recall_limit: int,
    recall_budget: int,
) -> list[dict]:
    """Run production GT eval with the given scoring config."""
    import memory.constants
    import memory.db
    import memory.reranker

    # Load GT
    with open(gt_path) as f:
        ground_truth = json.load(f)

    logger.info("Production GT: %d queries (config=%s)", len(ground_truth), config_name)

    # Point at production snapshot DB
    memory.db.DB_PATH = db_path

    # Ensure API key is available
    if not os.environ.get("OPENAI_API_KEY"):
        for key_path in [
            Path.home() / ".somnigraph" / "openai_api_key",
            Path.home() / ".claude" / "secrets" / "openai_api_key",
            Path.home() / ".claude" / "data" / "openai_api_key",
        ]:
            if key_path.exists():
                os.environ["OPENAI_API_KEY"] = key_path.read_text().strip()
                break

    # Point MODEL_PATH at real model location (not benchmark dir)
    for model_path in [
        Path.home() / ".somnigraph" / "tuning_studies" / "reranker_model.pkl",
        Path.home() / ".claude" / "data" / "tuning_studies" / "reranker_model.pkl",
    ]:
        if model_path.exists():
            memory.constants.MODEL_PATH = model_path
            memory.reranker.MODEL_PATH = model_path
            break

    # Reset module state
    import memory.embeddings
    memory.embeddings._openai_client = None

    # Apply scoring config
    apply_scoring_config(config_name)

    # Build token map for NDCG computation
    from memory.db import get_db
    db = get_db()
    token_rows = db.execute(
        "SELECT id, token_count FROM memories WHERE status = 'active'"
    ).fetchall()
    token_map = {r["id"]: r["token_count"] or 100 for r in token_rows}
    db.close()

    # Run evaluation
    from memory.tools import impl_recall
    records = []
    ranked_results = {}
    t_start = time.time()

    queries = list(ground_truth.keys())
    for qi, query in enumerate(queries):
        result = impl_recall(
            query=query,
            context=query,
            budget=recall_budget,
            limit=recall_limit,
            internal=True,
        )
        retrieved = _parse_ids_full(result)

        ranked_results[query] = retrieved
        gt_scores = ground_truth[query]

        # Per-query NDCG at token budget
        ndcg = _ndcg_at_budget(retrieved, gt_scores, token_map, recall_budget)

        records.append({
            "config": config_name,
            "dataset": "production",
            "query": query,
            "ndcg": ndcg,
            "retrieved_count": len(retrieved),
            "relevant_count": sum(1 for s in gt_scores.values() if s >= 0.5),
        })

        if (qi + 1) % 20 == 0:
            elapsed = time.time() - t_start
            logger.info("  %d/%d queries (%.1f q/s)", qi + 1,
                        len(queries), (qi + 1) / elapsed)

    # Reset DB connection
    memory.db._db_connection = None

    return records


def _ndcg_at_budget(
    ranked: list[str],
    gt_scores: dict[str, float],
    token_map: dict[str, int],
    budget: int,
) -> float:
    """NDCG at token budget for a single query."""
    # DCG: pack ranked results into budget
    shown = []
    used_tokens = 0
    for mid in ranked:
        tokens = token_map.get(mid, 100)
        if used_tokens + tokens > budget:
            break
        used_tokens += tokens
        shown.append(mid)

    if not shown:
        return 0.0

    dcg = 0.0
    for i, mid in enumerate(shown):
        rel = gt_scores.get(mid, 0.0)
        dcg += rel / math.log2(i + 2)

    # IDCG: greedy packing of highest-relevance memories
    ideal_candidates = sorted(gt_scores.items(), key=lambda x: x[1], reverse=True)
    ideal_rels = []
    ideal_tokens_used = 0
    for mid_ideal, rel_ideal in ideal_candidates:
        t_cost = token_map.get(mid_ideal, 100)
        if ideal_tokens_used + t_cost > budget:
            continue
        ideal_tokens_used += t_cost
        ideal_rels.append(rel_ideal)

    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# ID parsing
# ---------------------------------------------------------------------------


def _parse_ids(recall_output: str) -> list[str]:
    """Extract short (8-char) memory IDs from impl_recall output."""
    import re
    for line in recall_output.split("\n"):
        if line.startswith("recall_feedback IDs:"):
            return line.split(":")[1].strip().split()
    ids = []
    for line in recall_output.split("\n"):
        match = re.search(r"ID:\s+([a-f0-9]{8})", line)
        if match:
            ids.append(match.group(1))
    return ids


def _parse_ids_full(recall_output: str) -> list[str]:
    """Extract full memory IDs from impl_recall output."""
    import re
    ids = []
    for line in recall_output.split("\n"):
        match = re.search(r"ID:\s+([a-f0-9]{8})", line)
        if match:
            short_id = match.group(1)
            # Try to find full ID from the recall_feedback line
            pass

    # For production GT, we need full IDs. Parse from output format.
    # The recall output contains lines like "ID: abcd1234  [...]"
    # The recall_feedback line has short IDs, but GT uses full UUIDs.
    # We need to map short → full via DB.
    short_ids = _parse_ids(recall_output)
    if not short_ids:
        return []

    from memory.db import get_db
    db = get_db()
    full_ids = []
    for short in short_ids:
        row = db.execute(
            "SELECT id FROM memories WHERE id LIKE ?", (short + "%",)
        ).fetchone()
        if row:
            full_ids.append(row["id"])
    return full_ids


def _load_dia_map(path: Path) -> dict:
    """Load dia_map from JSON."""
    with open(path) as f:
        raw = json.load(f)
    result = {}
    for k, v in raw.items():
        parts = k.split(":", 1)
        if len(parts) == 2:
            result[(int(parts[0]), parts[1])] = v
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report_locomo(records: list[dict], configs: list[str], recall_limit: int):
    """Print LoCoMo comparison table."""
    from locomo_bench.config import CATEGORY_NAMES

    k_values = [k for k in [1, 3, 5, 10, 20, 50] if k <= recall_limit]

    print(f"\n{'=' * 90}")
    print(f"LoCoMo Retrieval Comparison (limit={recall_limit})")
    print(f"{'=' * 90}")

    for config_name in configs:
        config_records = [r for r in records if r["config"] == config_name]
        if not config_records:
            continue

        print(f"\n--- {config_name} ---")
        print(f"{'Category':<15} {'N':>5} {'MRR':>7}"
              + "".join(f" {'R@'+str(k):>7}" for k in k_values))
        print("-" * 70)

        cat_metrics = defaultdict(lambda: {"count": 0, "mrr_sum": 0.0,
                                           **{f"r@{k}": 0 for k in k_values}})
        for r in config_records:
            cat = r["category"]
            cat_metrics[cat]["count"] += 1
            cat_metrics[cat]["mrr_sum"] += r["mrr"]
            for k in k_values:
                if r.get(f"r@{k}"):
                    cat_metrics[cat][f"r@{k}"] += 1

        total_n = total_mrr = 0
        total_recalls = defaultdict(int)

        ADVERSARIAL_CAT = 5  # Excluded from OVERALL (no ground truth answer)

        for cat in sorted(cat_metrics):
            m = cat_metrics[cat]
            n = m["count"]
            if n == 0:
                continue
            name = CATEGORY_NAMES.get(cat, f"cat_{cat}")
            mrr = m["mrr_sum"] / n
            row = f"{name:<15} {n:>5} {mrr:>7.3f}"
            for k in k_values:
                r_at_k = m[f"r@{k}"] / n
                row += f" {r_at_k:>7.1%}"
            print(row)
            if cat != ADVERSARIAL_CAT:
                total_n += n
                total_mrr += m["mrr_sum"]
                for k in k_values:
                    total_recalls[k] += m[f"r@{k}"]

        if total_n:
            print("-" * 70)
            row = f"{'OVERALL':<15} {total_n:>5} {total_mrr/total_n:>7.3f}"
            for k in k_values:
                row += f" {total_recalls[k]/total_n:>7.1%}"
            print(row)
            print("(excludes adversarial)")

    # Comparison summary
    if len(configs) > 1:
        print(f"\n{'=' * 90}")
        print("Config Comparison (OVERALL, excludes adversarial)")
        print(f"{'Config':<15} {'N':>5} {'MRR':>7}"
              + "".join(f" {'R@'+str(k):>7}" for k in k_values))
        print("-" * 70)

        for config_name in configs:
            config_records = [r for r in records
                              if r["config"] == config_name and r["category"] != ADVERSARIAL_CAT]
            if not config_records:
                continue
            n = len(config_records)
            mrr = sum(r["mrr"] for r in config_records) / n
            row = f"{config_name:<15} {n:>5} {mrr:>7.3f}"
            for k in k_values:
                r_at_k = sum(1 for r in config_records if r.get(f"r@{k}")) / n
                row += f" {r_at_k:>7.1%}"
            print(row)


def report_production(records: list[dict], configs: list[str]):
    """Print production GT comparison table."""
    print(f"\n{'=' * 70}")
    print("Production GT — NDCG Comparison")
    print(f"{'=' * 70}")
    print(f"{'Config':<15} {'N':>5} {'NDCG':>8} {'mean_ret':>10}")
    print("-" * 50)

    for config_name in configs:
        config_records = [r for r in records if r["config"] == config_name]
        if not config_records:
            continue
        n = len(config_records)
        ndcg = sum(r["ndcg"] for r in config_records) / n
        mean_ret = sum(r["retrieved_count"] for r in config_records) / n
        print(f"{config_name:<15} {n:>5} {ndcg:>8.4f} {mean_ret:>10.1f}")


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def snapshot_production_db(dest: Path):
    """Copy production memory.db to benchmark directory."""
    src = Path.home() / ".claude" / "data" / "memory.db"
    if not src.exists():
        logger.error("Production DB not found at %s", src)
        sys.exit(1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("Snapshot already exists at %s, skipping copy", dest)
        return
    logger.info("Snapshotting %s → %s (%.1f MB)", src, dest,
                src.stat().st_size / 1e6)
    shutil.copy2(str(src), str(dest))
    logger.info("Snapshot complete")


# ---------------------------------------------------------------------------
# Embedding cache — batch-embed all queries once, serve from cache thereafter
# ---------------------------------------------------------------------------

_embed_cache: dict[str, list[float]] = {}


def warm_embed_cache(texts: list[str]):
    """Batch-embed all texts and install a monkey-patch so impl_recall
    reads from cache instead of making per-query API calls."""
    import memory.embeddings

    # Deduplicate while preserving order
    unseen = []
    for t in texts:
        if t not in _embed_cache and t not in unseen:
            unseen.append(t)

    if unseen:
        logger.info("Batch-embedding %d queries...", len(unseen))
        # embed_batch handles chunking internally (100 per API call)
        embeddings = memory.embeddings.embed_batch(unseen)
        for text, emb in zip(unseen, embeddings):
            _embed_cache[text] = emb
        logger.info("Embedding cache warmed (%d total entries)", len(_embed_cache))

    # Monkey-patch embed_text to use cache (fallback to real call for cache misses)
    _original_embed = memory.embeddings.embed_text

    def _cached_embed(text: str) -> list[float]:
        if text in _embed_cache:
            return _embed_cache[text]
        result = _original_embed(text)
        _embed_cache[text] = result
        return result

    memory.embeddings.embed_text = _cached_embed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Unified retrieval quality evaluation")
    parser.add_argument("--dataset", required=True, choices=["locomo", "production"],
                        help="Dataset to evaluate on")
    parser.add_argument("--configs", nargs="+", default=["bare", "formula"],
                        choices=list(CONFIGS.keys()),
                        help="Scoring configs to compare")

    # LoCoMo options
    parser.add_argument("--conversations", type=int, nargs="+", default=[0],
                        help="LoCoMo conversation indices")
    parser.add_argument("--enrich", action="store_true",
                        help="LLM-enrich turns at ingest")

    # Production options
    parser.add_argument("--gt-path", type=str,
                        default=str(Path.home() / ".claude" / "data" / "ground_truth.json"),
                        help="Path to ground truth JSON")
    parser.add_argument("--snapshot", action="store_true",
                        help="Snapshot production DB before eval")

    # Shared options
    parser.add_argument("--recall-limit", type=int, default=20,
                        help="Max results from impl_recall")
    parser.add_argument("--recall-limits", type=int, nargs="+",
                        help="Run sweep over multiple limits (overrides --recall-limit)")
    parser.add_argument("--recall-budget", type=int, default=10000,
                        help="Token budget for impl_recall")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path")
    parser.add_argument("--expand-entity-focus", action="store_true",
                        help="Entity-focused FTS expansion")
    parser.add_argument("--expand-multi-query", action="store_true",
                        help="Multi-query decomposition expansion")
    parser.add_argument("--expand-keyword", action="store_true",
                        help="Keyword expansion from seeds")
    parser.add_argument("--expand-session", action="store_true",
                        help="Session co-occurrence expansion")
    parser.add_argument("--expand-entity-bridge", action="store_true",
                        help="Entity bridge expansion")
    parser.add_argument("--expand-rocchio", action="store_true",
                        help="Rocchio PRF vector centroid blend")
    parser.add_argument("--expand-all", action="store_true",
                        help="Enable all expansion methods")
    parser.add_argument("--n-seeds", type=int, default=10,
                        help="Number of phase 1 seeds for phase 2 expansion (default: 10)")
    parser.add_argument("--graph-resolve", action="store_true",
                        help="Resolve synthetic graph nodes to source turns after Phase 1")
    parser.add_argument("--synthetic-coverage", type=str, default=None,
                        help="Path to synthetic_coverage.json for L5b scoring")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    limits = args.recall_limits or [args.recall_limit]

    # Enable expansion methods
    expand_methods = {
        "entity_focus": args.expand_all or args.expand_entity_focus,
        "multi_query": args.expand_all or args.expand_multi_query,
        "keyword": args.expand_all or args.expand_keyword,
        "session": args.expand_all or args.expand_session,
        "entity_bridge": args.expand_all or args.expand_entity_bridge,
        "rocchio": args.expand_all or args.expand_rocchio,
    }
    active_methods = [m for m, v in expand_methods.items() if v]
    if active_methods:
        enable_expansion(**expand_methods)
        logger.info("Expansion methods enabled: %s", ", ".join(active_methods))

    global _n_seeds, _graph_resolve, _synthetic_coverage
    _n_seeds = args.n_seeds
    _graph_resolve = args.graph_resolve

    if args.synthetic_coverage:
        cov_path = Path(args.synthetic_coverage)
        if not cov_path.exists():
            logger.error("Synthetic coverage file not found: %s", cov_path)
            sys.exit(1)
        _synthetic_coverage = json.loads(cov_path.read_text())
        logger.info("Loaded synthetic coverage: %d questions", len(_synthetic_coverage))

    benchmark_dir = Path.home() / ".somnigraph" / "benchmark"
    snapshot_path = benchmark_dir / "production_snapshot.db"

    # Snapshot if requested
    if args.dataset == "production" and args.snapshot:
        snapshot_production_db(snapshot_path)

    # Ensure API key is available before embedding
    if not os.environ.get("OPENAI_API_KEY"):
        for key_path in [
            Path.home() / ".somnigraph" / "openai_api_key",
            Path.home() / ".claude" / "secrets" / "openai_api_key",
            Path.home() / ".claude" / "data" / "openai_api_key",
        ]:
            if key_path.exists():
                os.environ["OPENAI_API_KEY"] = key_path.read_text().strip()
                break

    # Pre-embed all queries once (avoid per-query API calls across configs)
    if args.dataset == "production":
        with open(args.gt_path) as f:
            gt_queries = list(json.load(f).keys())
        warm_embed_cache(gt_queries)
    elif args.dataset == "locomo":
        from locomo_bench.ingest import extract_questions, load_locomo
        from locomo_bench.config import BenchConfig
        bench_config = BenchConfig()
        data = load_locomo(bench_config.locomo_data)
        all_questions = extract_questions(data)
        locomo_queries = [q["question"] for q in all_questions
                          if q["conv_id"] in args.conversations]
        warm_embed_cache(locomo_queries)

    all_records = []

    # Open JSONL for incremental writes
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = benchmark_dir / f"eval_{args.dataset}_{'_'.join(args.configs)}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = open(out_path, "w")

    for limit in limits:
        for config_name in args.configs:
            logger.info("=== Config: %s, limit: %d ===", config_name, limit)

            if args.dataset == "locomo":
                for conv_idx in args.conversations:
                    records = eval_locomo(
                        config_name, conv_idx, limit, args.recall_budget,
                        args.enrich,
                    )
                    all_records.extend(records)
                    for r in records:
                        out_file.write(json.dumps(r, default=str) + "\n")
                    out_file.flush()
            else:
                # Production
                db_path = snapshot_path
                if not db_path.exists():
                    logger.error("No production snapshot. Run with --snapshot first.")
                    sys.exit(1)
                records = eval_production(
                    config_name, Path(args.gt_path), db_path,
                    limit, args.recall_budget,
                )
                all_records.extend(records)
                for r in records:
                    out_file.write(json.dumps(r, default=str) + "\n")
                out_file.flush()

    out_file.close()
    logger.info("Results saved to %s (%d records)", out_path, len(all_records))

    # Report
    if args.dataset == "locomo":
        report_locomo(all_records, args.configs, limits[-1])
    else:
        report_production(all_records, args.configs)


if __name__ == "__main__":
    main()
