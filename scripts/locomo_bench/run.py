# /// script
# requires-python = ">=3.11"
# dependencies = ["sqlite-vec>=0.1.6", "openai>=2.0.0", "numpy>=1.26"]
# ///
"""
LoCoMo end-to-end QA benchmark for Somnigraph.

Two modes:
  --prompt-mode core       Faithful CORE replication (GPT-4.1, their exact prompts)
  --prompt-mode somnigraph Our optimized prompts from prior work

Ingests LoCoMo conversations via impl_remember, retrieves via impl_recall,
generates answers via LLM, and judges correctness. Uses Somnigraph's full
scoring pipeline (reranker, RRF, Hebbian, PPR, feedback loop).

Usage:
  # CORE replication (conversation 0 only, skip adversarial, GPT-4.1)
  uv run scripts/locomo_bench/run.py --conversations 0 --skip-adversarial

  # Full Somnigraph run (all conversations, all categories)
  uv run scripts/locomo_bench/run.py --prompt-mode somnigraph --reader-model claude-haiku-4-5-20251001

  # Quick test
  uv run scripts/locomo_bench/run.py --conversations 0 --limit 10
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Setup: add src/ and scripts/ to path for memory and locomo_bench imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from locomo_bench.config import CATEGORY_NAMES, BenchConfig
from locomo_bench.corrected_gt import (
    get_corrected_answer,
    get_error_type,
    load_corrections,
)
from locomo_bench.evaluate import (
    generate_answer,
    judge_answer,
    recall_memories,
    token_f1,
)
from locomo_bench.ingest import (
    extract_questions,
    extract_turns,
    ingest_conversation,
    load_locomo,
)
from locomo_bench.report import load_results, print_report, write_summary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DB isolation
# ---------------------------------------------------------------------------


def setup_isolated_db(conv_dir: Path):
    """Patch memory module globals to use an isolated DB directory.

    Preserves access to the OpenAI API key from the real DATA_DIR,
    and resets module-level caches that depend on the old path.
    """
    import memory.constants
    import memory.db
    import memory.embeddings
    import memory.reranker

    conv_dir.mkdir(parents=True, exist_ok=True)

    # Ensure OpenAI API key is available via env var before patching DATA_DIR.
    # The embeddings module reads the key file from DATA_DIR on first use,
    # but our patched dir won't have it.
    if not os.environ.get("OPENAI_API_KEY"):
        # Check multiple known locations
        for key_path in [
            memory.constants.DATA_DIR / "openai_api_key",
            Path.home() / ".claude" / "secrets" / "openai_api_key",
            Path.home() / ".claude" / "data" / "openai_api_key",
        ]:
            if key_path.exists():
                os.environ["OPENAI_API_KEY"] = key_path.read_text().strip()
                break

    memory.constants.DATA_DIR = conv_dir
    memory.db.DB_PATH = conv_dir / "memory.db"
    memory.reranker._cache = {"model": None, "feature_names": None, "memory_meta": None, "meta_mem_count": -1, "loaded": False, "failed": False}
    # Point MODEL_PATH at the real model location (not the benchmark dir).
    # Must patch both constants and reranker since reranker imports at module load.
    real_model = Path.home() / ".somnigraph" / "tuning_studies" / "reranker_model.pkl"
    if not real_model.exists():
        real_model = Path.home() / ".claude" / "data" / "tuning_studies" / "reranker_model.pkl"
    if real_model.exists():
        memory.constants.MODEL_PATH = real_model
        memory.reranker.MODEL_PATH = real_model
    # Reset embeddings client so it picks up the env var if needed
    memory.embeddings._openai_client = None


# ---------------------------------------------------------------------------
# Single-conversation pipeline
# ---------------------------------------------------------------------------


def run_conversation(
    conv_idx: int,
    conversations: list[dict],
    all_turns: list[dict],
    all_questions: list[dict],
    config: BenchConfig,
    run_dir: Path,
    question_limit: int = 0,
    results_path: Path | None = None,
    done_keys: set | None = None,
    corrections: dict | None = None,
) -> list[dict]:
    """Run the full pipeline for one conversation."""
    conv_dir = run_dir / f"conv_{conv_idx}"

    # Filter turns and questions for this conversation
    conv_turns = [t for t in all_turns if t["conv_id"] == conv_idx]
    conv_questions = [q for q in all_questions if q["conv_id"] == conv_idx]

    if config.skip_adversarial:
        conv_questions = [q for q in conv_questions if q["category"] != 5]

    if question_limit:
        conv_questions = conv_questions[:question_limit]

    logger.info("Conversation %d: %d turns, %d questions",
                conv_idx, len(conv_turns), len(conv_questions))

    # Phase 1: Ingest
    setup_isolated_db(conv_dir)

    # Check if DB already has memories (skip-ingest for resume)
    from memory.db import get_db
    db = get_db()
    try:
        count = db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    except Exception:
        count = 0
    finally:
        db.close()

    if count > 0 and config.resume:
        logger.info("  Reusing existing DB (%d memories)", count)
        # Rebuild dia_map from DB
        dia_map = _rebuild_dia_map(conv_dir)
    else:
        dia_map = ingest_conversation(conv_turns)
        logger.info("  Ingestion complete: %d memories", len(dia_map))

    # Phase 2: Evaluate each question
    results = []
    _done = done_keys or set()
    for q_idx, q in enumerate(conv_questions):
        if (q["conv_id"], q["question"]) in _done:
            continue

        try:
            result = _evaluate_question(q, dia_map, config, q_idx, corrections)
            results.append(result)

            label = result.get("judge_label", "?")
            cat_name = CATEGORY_NAMES.get(q["category"], "?")
            gt_short = str(q["answer"])[:30]
            gen_short = result.get("generated_answer", "")[:30]
            logger.info(
                "  [%d/%d] %s (%s) GT: %s | Gen: %s",
                q_idx + 1, len(conv_questions), label, cat_name,
                gt_short, gen_short,
            )

        except Exception as e:
            logger.error("  Error on question %d: %s", q_idx, e)
            result = {
                "conv_id": conv_idx,
                "category": q["category"],
                "question": q["question"],
                "ground_truth": str(q["answer"]),
                "generated_answer": "",
                "judge_label": "ERROR",
                "judge_reasoning": str(e),
                "judge_method": "error",
                "f1": 0.0,
                "match_ratio": 0.0,
                "retrieval_hit": False,
                "retrieved_count": 0,
            }
            results.append(result)

        # Write incrementally
        if results_path:
            with open(results_path, "a") as f:
                f.write(json.dumps(result) + "\n")

        if (q_idx + 1) % 20 == 0:
            correct = sum(1 for r in results if r.get("judge_label") == "CORRECT")
            logger.info("  Progress: %d/%d | accuracy so far: %.1f%%",
                        q_idx + 1, len(conv_questions),
                        100 * correct / len(results))

    return results


def _evaluate_question(
    q: dict,
    dia_map: dict,
    config: BenchConfig,
    seed: int,
    corrections: dict | None = None,
) -> dict:
    """Evaluate a single question: retrieve -> generate -> judge.

    If corrections are provided and this question has a corrected GT,
    judges against both original and corrected GT. The corrected judgment
    is stored in corrected_* fields; original in judge_*.
    """
    question = q["question"]
    gold = str(q["answer"])
    category = q["category"]
    conv_idx = q["conv_id"]
    qa_idx = q.get("qa_idx", -1)

    # Look up corrected GT
    corrected_gold = None
    gt_error_type = None
    if corrections and qa_idx >= 0:
        corrected_gold = get_corrected_answer(corrections, conv_idx, qa_idx)
        gt_error_type = get_error_type(corrections, conv_idx, qa_idx)

    # Retrieve
    memory_ids, context = recall_memories(question, config)

    # Check if evidence was retrieved
    evidence_mids = set()
    for eid in q.get("evidence", []):
        key = (conv_idx, eid)
        if key in dia_map:
            full_id = dia_map[key]
            short_id = full_id[:8]
            evidence_mids.add(short_id)
    retrieval_hit = bool(evidence_mids & set(memory_ids[:config.reader_top_k]))

    # Generate answer (always uses original GT — only matters for somnigraph
    # mode adversarial multiple-choice formatting)
    generated = generate_answer(
        question, context, category, gold, config, seed=seed,
    )

    # Token F1 against original GT
    f1 = token_f1(generated, gold)

    # Build base result
    result = {
        "conv_id": conv_idx,
        "qa_idx": qa_idx,
        "category": category,
        "question": question,
        "ground_truth": gold,
        "generated_answer": generated,
        "f1": round(f1, 4),
        "retrieval_hit": retrieval_hit,
        "retrieved_count": len(memory_ids),
    }

    # Add corrected GT metadata
    if corrected_gold:
        result["corrected_gt"] = corrected_gold
        result["gt_error_type"] = gt_error_type
        result["corrected_f1"] = round(token_f1(generated, corrected_gold), 4)

    # Judge (skip if --no-judge)
    if config.no_judge:
        result.update({
            "judge_label": "PENDING",
            "judge_reasoning": "",
            "judge_method": "skipped",
            "match_ratio": 0.0,
        })
        if corrected_gold:
            result.update({
                "corrected_label": "PENDING",
                "corrected_reasoning": "",
            })
        return result

    # Judge against original GT
    judgment = judge_answer(question, gold, generated, category, config)
    result.update({
        "judge_label": judgment["label"],
        "judge_reasoning": judgment["reasoning"],
        "judge_method": judgment.get("method", ""),
        "match_ratio": round(judgment.get("match_ratio", 0.0), 4),
    })

    # Judge against corrected GT (separate LLM call, only for affected questions)
    if corrected_gold:
        corrected_judgment = judge_answer(
            question, corrected_gold, generated, category, config,
        )
        result.update({
            "corrected_label": corrected_judgment["label"],
            "corrected_reasoning": corrected_judgment["reasoning"],
        })

    return result


def _rebuild_dia_map(conv_dir: Path) -> dict:
    """Rebuild (conv_id, dia_id) -> memory_id map from existing DB."""
    import sqlite3
    db_path = conv_dir / "memory.db"
    if not db_path.exists():
        return {}

    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    dia_map = {}

    # Parse dia_id from content (format: "[Speaker] text") and themes
    for row in db.execute("SELECT id, content, themes FROM memories WHERE status = 'active'"):
        # We need a way to map back to dia_id. Since we don't store it directly,
        # we use the memory ID as-is. Evidence matching uses short IDs.
        # This is a limitation of the resume path.
        pass

    db.close()
    return dia_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="LoCoMo end-to-end QA benchmark for Somnigraph")
    parser.add_argument("--conversations", type=int, nargs="+",
                        help="Conversation indices (default: all 10)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--run-dir", type=str,
                        help="Resume into existing run directory (implies --resume)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max questions per conversation (0=all)")

    # Model options
    parser.add_argument("--reader-model", type=str, default="gpt-4.1",
                        help="Model for answer generation (default: gpt-4.1)")
    parser.add_argument("--judge-model", type=str, default="gpt-4.1",
                        help="Model for judging (default: gpt-4.1)")
    parser.add_argument("--prompt-mode", type=str, default="core",
                        choices=["core", "somnigraph"],
                        help="Prompt set: core (CORE replication) or somnigraph (our prompts)")

    # Retrieval options
    parser.add_argument("--recall-limit", type=int, default=20,
                        help="Max results from impl_recall")
    parser.add_argument("--reader-top-k", type=int, default=20,
                        help="Top-k retrieved memories shown to reader")

    # Category options
    parser.add_argument("--skip-adversarial", action="store_true",
                        help="Skip category 5 (match CORE methodology)")

    # Reranker options
    parser.add_argument("--locomo-reranker", action="store_true",
                        help="Use LoCoMo-specific reranker instead of production")

    # Ablation options
    parser.add_argument("--feedback-loop", action="store_true",
                        help="Two-pass: run, feed scores back, re-run")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip judging (reader + retrieval only, batch judge later)")
    parser.add_argument("--no-corrected-gt", action="store_true",
                        help="Skip corrected GT judging (original GT only)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    config = BenchConfig(
        model_reader=args.reader_model,
        model_judge=args.judge_model,
        prompt_mode=args.prompt_mode,
        recall_limit=args.recall_limit,
        reader_top_k=args.reader_top_k,
        skip_adversarial=args.skip_adversarial,
        use_feedback_loop=args.feedback_loop,
        resume=args.resume,
        no_judge=args.no_judge,
        use_corrected_gt=not args.no_corrected_gt,
    )

    # Install LoCoMo reranker if requested
    if args.locomo_reranker:
        from locomo_bench.eval_retrieval import _install_locomo_reranker
        _install_locomo_reranker()

    # Create or reuse run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
        config.resume = True
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = config.base_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load GT corrections
    corrections = None
    if config.use_corrected_gt:
        try:
            corrections = load_corrections()
            logger.info("Loaded %d GT corrections (will report both original and corrected)",
                        len(corrections))
        except FileNotFoundError:
            logger.warning("Corrected GT file not found, using original GT only")

    # Save config
    config_dict = {
        "model_reader": config.model_reader,
        "model_judge": config.model_judge,
        "prompt_mode": config.prompt_mode,
        "recall_limit": config.recall_limit,
        "reader_top_k": config.reader_top_k,
        "skip_adversarial": config.skip_adversarial,
        "use_feedback_loop": config.use_feedback_loop,
        "use_corrected_gt": config.use_corrected_gt,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info("Run directory: %s", run_dir)
    logger.info("Config: %s", json.dumps(config_dict))

    # Load dataset
    conversations = load_locomo(config.locomo_data)
    all_turns = extract_turns(conversations)
    all_questions = extract_questions(conversations)

    conv_indices = args.conversations or list(range(len(conversations)))
    logger.info("Conversations: %s", conv_indices)
    logger.info("Total turns: %d, Total questions: %d",
                len(all_turns), len(all_questions))

    # Category breakdown
    from collections import defaultdict
    cat_counts = defaultdict(int)
    for q in all_questions:
        if q["conv_id"] in conv_indices:
            cat_counts[q["category"]] += 1
    for cat in sorted(cat_counts):
        logger.info("  Cat %d (%s): %d questions",
                    cat, CATEGORY_NAMES.get(cat, "?"), cat_counts[cat])

    # Output files
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"

    # Resume support
    done_keys = set()
    if args.resume and results_path.exists():
        with open(results_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done_keys.add((r["conv_id"], r["question"]))
        logger.info("Resuming: %d questions already done", len(done_keys))

    all_results = []
    t_start = time.time()

    for conv_idx in conv_indices:
        logger.info("\n%s Conversation %d %s",
                    "=" * 20, conv_idx, "=" * 20)

        results = run_conversation(
            conv_idx, conversations, all_turns, all_questions,
            config, run_dir,
            question_limit=args.limit,
            results_path=results_path,
            done_keys=done_keys,
            corrections=corrections,
        )

        all_results.extend(results)

        # Per-conversation summary
        correct = sum(1 for r in results if r.get("judge_label") == "CORRECT")
        total = len(results)
        logger.info("  Conv %d: %d/%d correct (%.1f%%)",
                    conv_idx, correct, total,
                    100 * correct / total if total else 0)

    # Final report
    elapsed = time.time() - t_start
    logger.info("\nCompleted in %.1f minutes", elapsed / 60)

    # Load all results (including resumed ones)
    if results_path.exists():
        all_results = load_results(results_path)

    print_report(all_results)
    write_summary(all_results, summary_path)
    logger.info("Results: %s", results_path)
    logger.info("Summary: %s", summary_path)


if __name__ == "__main__":
    main()
