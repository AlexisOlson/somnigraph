"""Judge which GT evidence turns each synthetic node covers.

For each question, collects all GT evidence turns and all synthetic nodes
extracted from those turns, then asks an LLM which synthetics contain
equivalent information for answering the question.

Output: JSON mapping (question_key -> {synthetic_id -> [covered_turn_ids]})

Usage:
    python scripts/locomo_bench/judge_synthetic_coverage.py [--resume]
"""

import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

BENCHMARK_DIR = Path.home() / ".somnigraph" / "benchmark"
OUTPUT_PATH = BENCHMARK_DIR / "synthetic_coverage.json"
BATCH_SIZE = 10
MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """\
You are evaluating whether synthetic claim/segment nodes contain equivalent \
information to original dialogue turns, in the context of answering a specific question.

For each synthetic node, determine which evidence turns (if any) it covers — \
meaning it contains the specific information from that turn needed to answer \
the question. A synthetic covers a turn if a reader seeing only the synthetic \
(not the original turn) would have the same evidence for answering the question.

Be strict: surface topic overlap is not enough. The synthetic must carry the \
specific facts, details, or statements from the turn that are relevant to the question."""

QUESTION_TEMPLATE = """\
--- Question {q_label} (category: {category}) ---
Question: {question}
Answer: {answer}

Evidence turns:
{evidence_block}

Synthetic nodes:
{synthetic_block}
"""

BATCH_FOOTER = """
For each synthetic in each question, list which evidence turn IDs it covers.
If a synthetic covers no evidence turns for that question, list "none".

Respond in exactly this format:
Q1/S1: E2, E3
Q1/S2: none
Q2/S1: E1
..."""

CATEGORIES = {1: "single-hop", 2: "temporal", 3: "multi-hop", 4: "open-domain", 5: "adversarial"}


def load_conv_data(conv_idx: int):
    """Load questions, dia_map, synthetic nodes, and their edges for one conversation."""
    conv_dir = BENCHMARK_DIR / f"retrieval_{conv_idx}"
    dia_map = json.loads((conv_dir / "dia_map.json").read_text())

    db = sqlite3.connect(str(conv_dir / "memory.db"))
    db.row_factory = sqlite3.Row

    memories = {}
    for row in db.execute("SELECT id, content, source FROM memories"):
        memories[row["id"]] = {
            "content": row["content"],
            "is_synthetic": row["source"] == "extraction",
        }

    syn_to_turns = defaultdict(set)
    for row in db.execute(
        "SELECT source_id, target_id, linking_context FROM memory_edges "
        "WHERE linking_context LIKE 'extracted_from:%'"
    ):
        syn_to_turns[row["source_id"]].add(row["target_id"])

    db.close()

    corrected = json.loads((BENCHMARK_DIR / "locomo10_corrected.json").read_text())
    questions = corrected[conv_idx]["qa"]

    return questions, dia_map, memories, syn_to_turns


def build_question_block(question, dia_map, memories, syn_to_turns, conv_idx, q_label):
    """Build prompt block for one question.

    Returns (block_str, evidence_labels, synthetic_labels) or None.
    """
    gt_turns = {}
    for eid in question.get("evidence", []):
        key = f"{conv_idx}:{eid}"
        if key in dia_map:
            gt_turns[eid] = dia_map[key]

    if not gt_turns:
        return None

    gt_turn_ids = set(gt_turns.values())
    relevant_syns = {}
    for syn_id, turn_ids in syn_to_turns.items():
        if turn_ids & gt_turn_ids:
            relevant_syns[syn_id] = turn_ids & gt_turn_ids

    if not relevant_syns:
        return None

    evidence_labels = {}
    evidence_lines = []
    for i, (dia_id, mem_id) in enumerate(gt_turns.items(), 1):
        label = f"E{i}"
        evidence_labels[label] = mem_id
        content = memories.get(mem_id, {}).get("content", "[missing]")
        if len(content) > 500:
            content = content[:500] + "..."
        evidence_lines.append(f"  {label} ({dia_id}): {content}")

    synthetic_labels = {}
    synthetic_lines = []
    for i, (syn_id, _) in enumerate(relevant_syns.items(), 1):
        label = f"S{i}"
        synthetic_labels[label] = syn_id
        content = memories.get(syn_id, {}).get("content", "[missing]")
        if len(content) > 500:
            content = content[:500] + "..."
        synthetic_lines.append(f"  {label}: {content}")

    category = CATEGORIES.get(question.get("category", 0), "unknown")
    block = QUESTION_TEMPLATE.format(
        q_label=q_label,
        category=category,
        question=question["question"],
        answer=str(question.get("answer", "")),
        evidence_block="\n".join(evidence_lines),
        synthetic_block="\n".join(synthetic_lines),
    )

    return block, evidence_labels, synthetic_labels


def parse_batch_response(response: str, question_map: dict):
    """Parse batched response with Q1/S1: E2 format.

    Returns: {qkey: {synthetic_id: [covered_turn_ids]}}
    """
    results = {}
    for line in response.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        m = re.match(r"Q(\d+)/S(\d+)\s*:\s*(.*)", line, re.IGNORECASE)
        if not m:
            continue

        q_label = f"Q{m.group(1)}"
        s_label = f"S{m.group(2)}"
        covered_str = m.group(3).strip().lower()

        if q_label not in question_map:
            continue
        qinfo = question_map[q_label]
        qkey = qinfo["qkey"]

        if qkey not in results:
            results[qkey] = {}

        syn_id = qinfo["synthetic_labels"].get(s_label)
        if not syn_id:
            continue

        if covered_str == "none" or not covered_str:
            results[qkey][syn_id] = []
            continue

        covered_turns = []
        for token in re.findall(r"E\d+", covered_str, re.IGNORECASE):
            token = token.upper()
            turn_id = qinfo["evidence_labels"].get(token)
            if turn_id:
                covered_turns.append(turn_id)
        results[qkey][syn_id] = covered_turns

    return results


def call_claude(system_prompt: str, user_prompt: str, model: str) -> str:
    """Call Claude via claude -p subprocess."""
    prompt = ""
    if system_prompt:
        prompt += system_prompt + "\n\n"
    prompt += user_prompt

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)

    claude_cmd = "claude.cmd" if sys.platform == "win32" else "claude"

    for attempt in range(3):
        try:
            result = subprocess.run(
                [claude_cmd, "-p", "--model", model],
                input=prompt,
                capture_output=True,
                timeout=300,
                env=env,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode == 0:
                return result.stdout.strip()
            err = result.stderr.strip()
        except subprocess.TimeoutExpired:
            err = "timeout"

        if attempt < 2:
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Claude call failed after 3 attempts: {err}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    parser.add_argument("--conversations", type=int, nargs="+",
                        default=list(range(10)))
    parser.add_argument("--dry-run", action="store_true",
                        help="Count pairs without calling LLM")
    args = parser.parse_args()

    # Load existing results if resuming
    existing = {}
    if args.resume and OUTPUT_PATH.exists():
        existing = json.loads(OUTPUT_PATH.read_text())
        print(f"Loaded {len(existing)} existing judgments")

    results = dict(existing)

    total_pairs = 0
    total_questions = 0
    skipped = 0
    judged = 0
    batch_count = 0

    for conv_idx in args.conversations:
        questions, dia_map, memories, syn_to_turns = load_conv_data(conv_idx)
        print(f"\nConv {conv_idx}: {len(questions)} questions, {len(syn_to_turns)} synthetics")

        # Collect all questions needing judgment for this conversation
        pending = []
        for qa_idx, q in enumerate(questions):
            qkey = f"{conv_idx}:{qa_idx}"
            total_questions += 1

            if qkey in results:
                skipped += 1
                continue

            batch_data = build_question_block(q, dia_map, memories, syn_to_turns, conv_idx, "")
            if batch_data is None:
                results[qkey] = {}
                continue

            pending.append((qa_idx, q, batch_data))

        if args.dry_run:
            for _, _, (_, _, syn_labels) in pending:
                total_pairs += len(syn_labels)
            continue

        # Process in batches
        for batch_start in range(0, len(pending), BATCH_SIZE):
            batch = pending[batch_start:batch_start + BATCH_SIZE]

            question_map = {}
            prompt_blocks = []

            for i, (qa_idx, q, _) in enumerate(batch, 1):
                q_label = f"Q{i}"
                # Rebuild with correct label
                block_data = build_question_block(q, dia_map, memories, syn_to_turns, conv_idx, q_label)
                if block_data is None:
                    continue
                block, evidence_labels, synthetic_labels = block_data
                qkey = f"{conv_idx}:{qa_idx}"
                question_map[q_label] = {
                    "evidence_labels": evidence_labels,
                    "synthetic_labels": synthetic_labels,
                    "qkey": qkey,
                }
                prompt_blocks.append(block)

            if not prompt_blocks:
                continue

            user_prompt = "\n".join(prompt_blocks) + BATCH_FOOTER
            response = call_claude(SYSTEM_PROMPT, user_prompt, MODEL)
            batch_results = parse_batch_response(response, question_map)

            # Store results, including empty for questions not in response
            for q_label, qinfo in question_map.items():
                qkey = qinfo["qkey"]
                if qkey in batch_results:
                    results[qkey] = batch_results[qkey]
                else:
                    results[qkey] = {}

            judged += len(question_map)
            batch_count += 1

            OUTPUT_PATH.write_text(json.dumps(results, indent=2))
            print(f"  {judged} judged ({batch_count} batches), saved checkpoint")

    if args.dry_run:
        print(f"\nDry run: {total_questions} questions, {total_pairs} (question, synthetic) pairs")
        return

    # Final save
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))

    # Stats
    total_covered = 0
    total_not_covered = 0
    for qkey, coverage in results.items():
        for syn_id, turns in coverage.items():
            if turns:
                total_covered += 1
            else:
                total_not_covered += 1

    print(f"\nDone. {judged} judged, {skipped} skipped (resumed), {batch_count} batches")
    print(f"Coverage: {total_covered} synthetics cover evidence, {total_not_covered} do not")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
