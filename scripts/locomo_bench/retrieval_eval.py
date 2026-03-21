# /// script
# requires-python = ">=3.11"
# dependencies = ["sqlite-vec>=0.1.6", "openai>=2.0.0", "numpy>=1.26", "tiktoken"]
# ///
"""
LoCoMo retrieval-only evaluation.

Ingests conversations via the somnigraph write path, then runs each question
through impl_recall and checks if the evidence turns appear in the retrieved set.
No LLM reader or judge — pure retrieval quality.

Usage:
  uv run scripts/locomo_bench/retrieval_eval.py                    # conv 0, default
  uv run scripts/locomo_bench/retrieval_eval.py --conversations 0 1 2
  uv run scripts/locomo_bench/retrieval_eval.py --recall-limit 50
  uv run scripts/locomo_bench/retrieval_eval.py --skip-ingest      # reuse existing DB
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from locomo_bench.config import CATEGORY_NAMES, BenchConfig
from locomo_bench.ingest import extract_questions, extract_turns, load_locomo
from locomo_bench.run import setup_isolated_db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retrieval scoring
# ---------------------------------------------------------------------------


def recall_and_score(
    question: str,
    evidence_short_ids: set[str],
    limit: int,
    budget: int,
) -> dict:
    """Run impl_recall and score against evidence."""
    from memory.tools import impl_recall

    result = impl_recall(
        query=question,
        context=question,
        budget=budget,
        limit=limit,
        internal=True,
    )

    # Parse memory IDs from recall output
    retrieved = _parse_ids(result)

    # Score at various k
    scores = {}
    first_hit = None
    for i, rid in enumerate(retrieved):
        if rid in evidence_short_ids:
            if first_hit is None:
                first_hit = i + 1
            break

    scores["mrr"] = 1.0 / first_hit if first_hit else 0.0
    scores["first_hit_rank"] = first_hit

    for k in [1, 3, 5, 10, 20, 50]:
        if k <= limit:
            top_k_set = set(retrieved[:k])
            scores[f"r@{k}"] = bool(top_k_set & evidence_short_ids)

    scores["retrieved_count"] = len(retrieved)
    scores["evidence_count"] = len(evidence_short_ids)
    return scores


def _parse_ids(recall_output: str) -> list[str]:
    """Extract short memory IDs from impl_recall output."""
    import re
    for line in recall_output.split("\n"):
        if line.startswith("recall_feedback IDs:"):
            return line.split(":")[1].strip().split()
    # Fallback
    ids = []
    for line in recall_output.split("\n"):
        match = re.search(r"ID:\s+([a-f0-9]{8})", line)
        if match:
            ids.append(match.group(1))
    return ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="LoCoMo retrieval-only eval")
    parser.add_argument("--conversations", type=int, nargs="+", default=[0])
    parser.add_argument("--recall-limit", type=int, default=20)
    parser.add_argument("--recall-budget", type=int, default=10000)
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--enrich", action="store_true",
                        help="LLM-enrich turns at ingest (extracts topics/facts/entities)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    config = BenchConfig()

    # Load dataset
    data = load_locomo(config.locomo_data)
    all_turns = extract_turns(data)
    all_questions = extract_questions(data)

    for conv_idx in args.conversations:
        conv_turns = [t for t in all_turns if t["conv_id"] == conv_idx]
        conv_questions = [q for q in all_questions if q["conv_id"] == conv_idx]

        logger.info("=== Conversation %d: %d turns, %d questions ===",
                    conv_idx, len(conv_turns), len(conv_questions))

        # Setup isolated DB
        suffix = "_enriched" if args.enrich else ""
        run_dir = config.base_dir / f"retrieval_{conv_idx}{suffix}"
        setup_isolated_db(run_dir)

        # Ingest
        if args.skip_ingest:
            from memory.db import get_db
            db = get_db()
            count = db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            db.close()
            logger.info("Reusing DB: %d memories", count)
            dia_map = _rebuild_dia_map_from_db(run_dir)
        else:
            from locomo_bench.ingest import ingest_conversation
            dia_map = ingest_conversation(conv_turns,
                                          embed_cache_path=config.embed_cache,
                                          enrich=args.enrich)
            # Save dia_map for --skip-ingest reuse
            with open(run_dir / "dia_map.json", "w") as f:
                json.dump({f"{k[0]}:{k[1]}": v for k, v in dia_map.items()}, f)

        logger.info("dia_map: %d entries", len(dia_map))

        # Evaluate retrieval
        cat_metrics = defaultdict(lambda: {
            "count": 0, "mrr_sum": 0.0,
            **{f"r@{k}": 0 for k in [1, 3, 5, 10, 20, 50]},
        })

        output_records = []
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

            scores = recall_and_score(
                q["question"], evidence_short,
                limit=args.recall_limit,
                budget=args.recall_budget,
            )

            cat = q["category"]
            cat_metrics[cat]["count"] += 1
            cat_metrics[cat]["mrr_sum"] += scores["mrr"]
            for k in [1, 3, 5, 10, 20, 50]:
                key = f"r@{k}"
                if key in scores and scores[key]:
                    cat_metrics[cat][key] += 1

            record = {
                "conv_id": conv_idx,
                "category": cat,
                "question": q["question"],
                "ground_truth": str(q["answer"]),
                "evidence": q["evidence"],
                **scores,
            }
            output_records.append(record)

            if (qi + 1) % 20 == 0:
                elapsed = time.time() - t_start
                logger.info("  %d/%d questions (%.1f q/s)",
                            qi + 1, len(conv_questions), (qi + 1) / elapsed)

        # Report
        print(f"\n{'=' * 80}")
        print(f"Retrieval Results — Conversation {conv_idx} (limit={args.recall_limit})")
        print(f"{'=' * 80}")
        print(f"{'Category':<15} {'N':>5} {'MRR':>7}"
              + "".join(f" {'R@'+str(k):>7}" for k in [1, 3, 5, 10, 20, 50]
                        if k <= args.recall_limit))
        print("-" * 80)

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
            for k in [1, 3, 5, 10, 20, 50]:
                key = f"r@{k}"
                if k <= args.recall_limit:
                    r = m[key] / n
                    row += f" {r:>7.1%}"
                    total_recalls[k] += m[key]
            print(row)
            total_n += n
            total_mrr += m["mrr_sum"]

        if total_n:
            print("-" * 80)
            row = f"{'OVERALL':<15} {total_n:>5} {total_mrr/total_n:>7.3f}"
            for k in [1, 3, 5, 10, 20, 50]:
                if k <= args.recall_limit:
                    row += f" {total_recalls[k]/total_n:>7.1%}"
            print(row)

        # Adversarial note
        if 5 in cat_metrics and cat_metrics[5]["count"] > 0:
            m5 = cat_metrics[5]
            r10 = m5.get("r@10", 0) / m5["count"]
            print(f"\nNote: Adversarial R@10 = {r10:.1%} "
                  "(lower is better — these have no real answer)")

        print()

        # Save JSONL output
        if args.output:
            out_path = Path(args.output)
        else:
            out_path = run_dir / "retrieval_results.jsonl"
        with open(out_path, "w") as f:
            for r in output_records:
                f.write(json.dumps(r, default=str) + "\n")
        logger.info("Results saved to %s", out_path)


def _rebuild_dia_map_from_db(run_dir: Path) -> dict:
    """Load dia_map from saved JSON."""
    map_path = run_dir / "dia_map.json"
    if map_path.exists():
        with open(map_path) as f:
            raw = json.load(f)
        # Keys stored as "conv_id:dia_id", need (int, str) tuples
        result = {}
        for k, v in raw.items():
            parts = k.split(":", 1)
            if len(parts) == 2:
                result[(int(parts[0]), parts[1])] = v
        return result
    return {}


if __name__ == "__main__":
    main()
