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
        """LoCoMo reranker: extract 15 features and predict."""
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

    def _locomo_rerank_inner(db, query, fts_ranked, vec_ranked, fts_scores,
                              vec_distances, theme_ranked, theme_overlap_map,
                              candidate_ids):
        from memory.reranker import _compute_proximity
        RRF_K = 60

        # Load memory metadata
        rows = db.execute("""
            SELECT id, content, themes, token_count, created_at
            FROM memories WHERE status = 'active'
        """).fetchall()
        memories = {}
        for r in rows:
            content = r["content"] or ""
            themes_list = []
            if r["themes"]:
                try:
                    themes_list = json.loads(r["themes"])
                except json.JSONDecodeError:
                    pass
            import re as _re
            speaker = ""
            sp_match = _re.match(r"^\[([^\]]+)\]", content)
            if sp_match:
                speaker = sp_match.group(1).lower()
            try:
                from datetime import datetime, timezone
                created = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
                age_days = (datetime.now(timezone.utc) - created).total_seconds() / 86400
            except Exception:
                age_days = 0.0
            memories[r["id"]] = {
                "content_tokens": content.lower().split(),
                "speaker": speaker,
                "theme_tokens": {tok for t in themes_list for tok in str(t).lower().replace("-", " ").split()},
                "theme_count": len(themes_list),
                "token_count": r["token_count"] or len(content.split()),
                "age_days": age_days,
            }

        # Ordinal map for neighbor density
        sorted_mids = sorted(memories.keys(), key=lambda m: memories[m]["age_days"], reverse=True)
        ordinal_map = {mid: i for i, mid in enumerate(sorted_mids)}

        query_tokens = set(query.lower().split())
        query_terms = [t for t in query.lower().split() if len(t) > 1]
        query_lower = query.lower()

        # RRF scores for percentile
        rrf_scores = {}
        for mid in candidate_ids:
            score = 0.0
            if mid in fts_ranked:
                score += 1.0 / (RRF_K + fts_ranked[mid])
            if mid in vec_ranked:
                score += 1.0 / (RRF_K + vec_ranked[mid])
            rrf_scores[mid] = score
        all_rrf = sorted(rrf_scores.values())
        n_rrf = len(all_rrf)

        candidate_ordinals = {mid: ordinal_map.get(mid, -999) for mid in candidate_ids}
        ordinal_set = set(candidate_ordinals.values())

        candidate_list = sorted(candidate_ids)
        import numpy as _np
        features = _np.zeros((len(candidate_list), 15), dtype=_np.float32)

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

        # Predict with LoCoMo model (select feature columns)
        X = features[:, _locomo_feature_indices]
        preds = _locomo_model.predict(X)

        scored = sorted(zip(candidate_list, preds), key=lambda x: -x[1])
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

    # Run evaluation
    from memory.tools import impl_recall
    records = []
    t_start = time.time()

    for qi, q in enumerate(conv_questions):
        # Map evidence dia_ids to short memory IDs
        evidence_short = set()
        for eid in q.get("evidence", []):
            key = (q["conv_id"], eid)
            if key in dia_map:
                evidence_short.add(dia_map[key][:8])

        if not evidence_short:
            continue

        result = impl_recall(
            query=q["question"],
            context=q["question"],
            budget=recall_budget,
            limit=recall_limit,
            internal=True,
        )
        retrieved = _parse_ids(result)

        # Score
        scores = _score_locomo(retrieved, evidence_short, recall_limit)
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
) -> dict:
    """Score a single LoCoMo question."""
    first_hit = None
    for i, rid in enumerate(retrieved):
        if rid in evidence_short:
            if first_hit is None:
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
            scores[f"r@{k}"] = bool(top_k_set & evidence_short)

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
                total_recalls[k] += m[f"r@{k}"]
            print(row)
            total_n += n
            total_mrr += m["mrr_sum"]

        if total_n:
            print("-" * 70)
            row = f"{'OVERALL':<15} {total_n:>5} {total_mrr/total_n:>7.3f}"
            for k in k_values:
                row += f" {total_recalls[k]/total_n:>7.1%}"
            print(row)

    # Comparison summary
    if len(configs) > 1:
        print(f"\n{'=' * 90}")
        print("Config Comparison (OVERALL)")
        print(f"{'Config':<15} {'N':>5} {'MRR':>7}"
              + "".join(f" {'R@'+str(k):>7}" for k in k_values))
        print("-" * 70)

        for config_name in configs:
            config_records = [r for r in records if r["config"] == config_name]
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

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    limits = args.recall_limits or [args.recall_limit]

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

    # Report
    if args.dataset == "locomo":
        report_locomo(all_records, args.configs, limits[-1])
    else:
        report_production(all_records, args.configs)

    # Save JSONL
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = benchmark_dir / f"eval_{args.dataset}_{'_'.join(args.configs)}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, default=str) + "\n")
    logger.info("Results saved to %s (%d records)", out_path, len(all_records))


if __name__ == "__main__":
    main()
