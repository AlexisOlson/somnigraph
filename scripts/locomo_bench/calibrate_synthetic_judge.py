"""Calibrate synthetic coverage judging: Opus grades a sample, then compare Haiku against it.

Step 1: Sample 1 question per category per conversation (up to 50 total).
Step 2: Opus judges in 2 batched calls (odd vs even conversations).
Step 3: Haiku judges the same questions individually.
Step 4: Compare agreement.

Usage:
    python scripts/locomo_bench/calibrate_synthetic_judge.py --step opus
    python scripts/locomo_bench/calibrate_synthetic_judge.py --step haiku
    python scripts/locomo_bench/calibrate_synthetic_judge.py --step compare
"""

import json
import os
import random
import re
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

BENCHMARK_DIR = Path.home() / ".somnigraph" / "benchmark"
CALIBRATION_DIR = BENCHMARK_DIR / "calibration"
SAMPLE_PATH = CALIBRATION_DIR / "sample.json"
OPUS_PATH = CALIBRATION_DIR / "opus_judgments.json"
HAIKU_PATH = CALIBRATION_DIR / "haiku_judgments.json"
SONNET_PATH = CALIBRATION_DIR / "sonnet_judgments.json"

CATEGORIES = {1: "single-hop", 2: "temporal", 3: "multi-hop", 4: "open-domain", 5: "adversarial"}

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


def load_conv_data(conv_idx: int):
    """Load questions, dia_map, memories, syn_to_turns for one conversation."""
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


def build_question_block(q, dia_map, memories, syn_to_turns, conv_idx, q_label):
    """Build prompt block for one question. Returns (block_str, evidence_labels, synthetic_labels) or None."""
    gt_turns = {}
    for eid in q.get("evidence", []):
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

    block = QUESTION_TEMPLATE.format(
        q_label=q_label,
        category=CATEGORIES.get(q["category"], "unknown"),
        question=q["question"],
        answer=str(q.get("answer", "")),
        evidence_block="\n".join(evidence_lines),
        synthetic_block="\n".join(synthetic_lines),
    )

    return block, evidence_labels, synthetic_labels


def sample_questions():
    """Sample 1 question per category per conversation where synthetics exist."""
    random.seed(42)
    samples = []

    for conv_idx in range(10):
        questions, dia_map, memories, syn_to_turns = load_conv_data(conv_idx)

        # Group by category
        by_cat = defaultdict(list)
        for qa_idx, q in enumerate(questions):
            cat = q.get("category", 0)
            # Check if this question has relevant synthetics
            gt_turns = {}
            for eid in q.get("evidence", []):
                key = f"{conv_idx}:{eid}"
                if key in dia_map:
                    gt_turns[eid] = dia_map[key]
            if not gt_turns:
                continue
            gt_turn_ids = set(gt_turns.values())
            has_syn = any(turn_ids & gt_turn_ids for turn_ids in syn_to_turns.values())
            if has_syn:
                by_cat[cat].append(qa_idx)

        for cat in sorted(by_cat.keys()):
            chosen = random.choice(by_cat[cat])
            samples.append({
                "conv_idx": conv_idx,
                "qa_idx": chosen,
                "category": cat,
                "question": questions[chosen]["question"],
            })

    print(f"Sampled {len(samples)} questions across {len(set(s['conv_idx'] for s in samples))} conversations")
    for cat_id, cat_name in sorted(CATEGORIES.items()):
        count = sum(1 for s in samples if s["category"] == cat_id)
        print(f"  {cat_name}: {count}")

    return samples


def parse_batch_response(response: str, question_map: dict):
    """Parse batched response with Q1/S1: E2 format.

    question_map: {q_label: {"evidence_labels": {...}, "synthetic_labels": {...}, "qkey": str}}
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


def run_opus(samples):
    """Run Opus on sampled questions in 2 batched calls (even/odd conversations)."""
    # Split into even/odd
    even_samples = [s for s in samples if s["conv_idx"] % 2 == 0]
    odd_samples = [s for s in samples if s["conv_idx"] % 2 == 1]

    all_results = {}

    for batch_name, batch_samples in [("even", even_samples), ("odd", odd_samples)]:
        print(f"\n{'='*60}")
        print(f"  Opus batch: {batch_name} conversations ({len(batch_samples)} questions)")
        print(f"{'='*60}")

        question_map = {}  # q_label -> {evidence_labels, synthetic_labels, qkey}
        prompt_blocks = []

        for i, sample in enumerate(batch_samples, 1):
            conv_idx = sample["conv_idx"]
            qa_idx = sample["qa_idx"]
            questions, dia_map, memories, syn_to_turns = load_conv_data(conv_idx)
            q = questions[qa_idx]

            q_label = f"Q{i}"
            result = build_question_block(q, dia_map, memories, syn_to_turns, conv_idx, q_label)
            if result is None:
                continue

            block, evidence_labels, synthetic_labels = result
            qkey = f"{conv_idx}:{qa_idx}"
            question_map[q_label] = {
                "evidence_labels": evidence_labels,
                "synthetic_labels": synthetic_labels,
                "qkey": qkey,
            }
            prompt_blocks.append(block)

        user_prompt = "\n".join(prompt_blocks) + BATCH_FOOTER
        print(f"  Prompt: {len(user_prompt)} chars, {len(question_map)} questions")

        response = call_claude(SYSTEM_PROMPT, user_prompt, "claude-opus-4-6")
        print(f"  Response: {len(response)} chars")
        print(f"  Raw response:\n{response}\n")

        batch_results = parse_batch_response(response, question_map)
        all_results.update(batch_results)

    return all_results


def run_batched(samples, batch_size: int, model: str = "claude-haiku-4-5-20251001"):
    """Run a model on sampled questions in batches of batch_size."""
    all_results = {}

    # Preload conv data (avoid redundant loads)
    conv_cache = {}
    for sample in samples:
        ci = sample["conv_idx"]
        if ci not in conv_cache:
            conv_cache[ci] = load_conv_data(ci)

    # Process in batches
    for batch_start in range(0, len(samples), batch_size):
        batch = samples[batch_start:batch_start + batch_size]

        question_map = {}
        prompt_blocks = []

        for i, sample in enumerate(batch, 1):
            conv_idx = sample["conv_idx"]
            qa_idx = sample["qa_idx"]
            questions, dia_map, memories, syn_to_turns = conv_cache[conv_idx]
            q = questions[qa_idx]

            q_label = f"Q{i}"
            result = build_question_block(q, dia_map, memories, syn_to_turns, conv_idx, q_label)
            if result is None:
                continue

            block, evidence_labels, synthetic_labels = result
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
        response = call_claude(SYSTEM_PROMPT, user_prompt, model)
        batch_results = parse_batch_response(response, question_map)
        all_results.update(batch_results)

        print(f"  Batch {batch_start//batch_size + 1}: {len(batch_results)} questions parsed")

    return all_results


def compare(opus_results, haiku_results):
    """Compare Opus vs Haiku judgments."""
    agree = 0
    disagree = 0
    opus_stricter = 0
    haiku_stricter = 0
    details = []

    for qkey in opus_results:
        if qkey not in haiku_results:
            continue

        opus_coverage = opus_results[qkey]
        haiku_coverage = haiku_results[qkey]

        all_syns = set(opus_coverage.keys()) | set(haiku_coverage.keys())
        for syn_id in all_syns:
            opus_turns = set(opus_coverage.get(syn_id, []))
            haiku_turns = set(haiku_coverage.get(syn_id, []))

            if opus_turns == haiku_turns:
                agree += 1
            else:
                disagree += 1
                if opus_turns < haiku_turns:
                    opus_stricter += 1
                elif haiku_turns < opus_turns:
                    haiku_stricter += 1
                else:
                    # Different but not subset
                    pass
                details.append({
                    "qkey": qkey,
                    "syn_id": syn_id[:12],
                    "opus": sorted(opus_turns),
                    "haiku": sorted(haiku_turns),
                })

    total = agree + disagree
    print(f"\nAgreement: {agree}/{total} ({100*agree/total:.1f}%)" if total else "No comparisons")
    print(f"Disagreements: {disagree}")
    print(f"  Opus stricter: {opus_stricter}")
    print(f"  Haiku stricter: {haiku_stricter}")
    print(f"  Different (neither subset): {disagree - opus_stricter - haiku_stricter}")

    if details:
        print(f"\nDisagreement details (first 10):")
        for d in details[:10]:
            print(f"  {d['qkey']} syn={d['syn_id']}: opus={d['opus']} haiku={d['haiku']}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, choices=["sample", "opus", "haiku", "sonnet", "compare"])
    args = parser.parse_args()

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

    if args.step == "sample":
        samples = sample_questions()
        SAMPLE_PATH.write_text(json.dumps(samples, indent=2))
        print(f"Saved to {SAMPLE_PATH}")

    elif args.step == "opus":
        if not SAMPLE_PATH.exists():
            samples = sample_questions()
            SAMPLE_PATH.write_text(json.dumps(samples, indent=2))
        else:
            samples = json.loads(SAMPLE_PATH.read_text())
            print(f"Loaded {len(samples)} samples")

        results = run_opus(samples)
        OPUS_PATH.write_text(json.dumps(results, indent=2))
        print(f"\nSaved {len(results)} Opus judgments to {OPUS_PATH}")

    elif args.step == "haiku":
        samples = json.loads(SAMPLE_PATH.read_text())
        print(f"Loaded {len(samples)} samples")

        batch_sizes = [5, 10, 20, 50]
        all_haiku = {}
        for bs in batch_sizes:
            print(f"\n--- Haiku batch_size={bs} ---")
            results = run_batched(samples, batch_size=bs)
            all_haiku[str(bs)] = results
            print(f"  {len(results)} questions judged")

        HAIKU_PATH.write_text(json.dumps(all_haiku, indent=2))
        print(f"\nSaved Haiku judgments ({len(batch_sizes)} batch sizes) to {HAIKU_PATH}")

    elif args.step == "sonnet":
        samples = json.loads(SAMPLE_PATH.read_text())
        print(f"Loaded {len(samples)} samples")

        print(f"\n--- Sonnet batch_size=10 ---")
        results = run_batched(samples, batch_size=10, model="claude-sonnet-4-6")
        SONNET_PATH.write_text(json.dumps(results, indent=2))
        print(f"\nSaved {len(results)} Sonnet judgments to {SONNET_PATH}")

    elif args.step == "compare":
        opus = json.loads(OPUS_PATH.read_text())

        if HAIKU_PATH.exists():
            haiku_all = json.loads(HAIKU_PATH.read_text())
            for bs, haiku in sorted(haiku_all.items(), key=lambda x: int(x[0])):
                print(f"\n{'='*50}")
                print(f"  Haiku batch_size={bs} vs Opus")
                print(f"{'='*50}")
                compare(opus, haiku)

        if SONNET_PATH.exists():
            sonnet = json.loads(SONNET_PATH.read_text())
            print(f"\n{'='*50}")
            print(f"  Sonnet batch_size=10 vs Opus")
            print(f"{'='*50}")
            compare(opus, sonnet)


if __name__ == "__main__":
    main()
