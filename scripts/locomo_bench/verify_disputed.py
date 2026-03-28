"""Re-judge the cases where Haiku was stricter than Opus, to verify Opus ground truth."""

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
CALIBRATION_DIR = BENCHMARK_DIR / "calibration"

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

BATCH_FOOTER = """
For each synthetic in each question, list which evidence turn IDs it covers.
If a synthetic covers no evidence turns for that question, list "none".

Respond in exactly this format:
Q1/S1: E2, E3
Q1/S2: none
Q2/S1: E1
..."""


def main():
    opus = json.loads((CALIBRATION_DIR / "opus_judgments.json").read_text())
    haiku_all = json.loads((CALIBRATION_DIR / "haiku_judgments.json").read_text())
    haiku = haiku_all["5"]
    corrected = json.loads((BENCHMARK_DIR / "locomo10_corrected.json").read_text())

    # Find Haiku-stricter cases
    cases = []
    for qkey in opus:
        if qkey not in haiku:
            continue
        for syn_id in set(opus[qkey].keys()) | set(haiku[qkey].keys()):
            o = set(opus[qkey].get(syn_id, []))
            h = set(haiku[qkey].get(syn_id, []))
            if o - h:
                cases.append({"qkey": qkey, "syn_id": syn_id})

    by_question = defaultdict(list)
    for c in cases:
        by_question[c["qkey"]].append(c["syn_id"])

    # Build prompt
    blocks = []
    q_metadata = {}

    q_num = 0
    for qkey, syn_ids in by_question.items():
        conv_idx, qa_idx = int(qkey.split(":")[0]), int(qkey.split(":")[1])
        q = corrected[conv_idx]["qa"][qa_idx]
        cat = CATEGORIES.get(q.get("category", 0), "unknown")

        conv_dir = BENCHMARK_DIR / f"retrieval_{conv_idx}"
        dia_map = json.loads((conv_dir / "dia_map.json").read_text())
        db = sqlite3.connect(str(conv_dir / "memory.db"))
        db.row_factory = sqlite3.Row
        memories = {}
        for row in db.execute("SELECT id, content, source FROM memories"):
            memories[row["id"]] = row["content"]

        syn_to_turns = defaultdict(set)
        for row in db.execute(
            "SELECT source_id, target_id, linking_context FROM memory_edges "
            "WHERE linking_context LIKE 'extracted_from:%'"
        ):
            syn_to_turns[row["source_id"]].add(row["target_id"])
        db.close()

        gt_turns = {}
        for eid in q.get("evidence", []):
            key = f"{conv_idx}:{eid}"
            if key in dia_map:
                gt_turns[eid] = dia_map[key]

        if not gt_turns:
            continue

        q_num += 1
        q_label = f"Q{q_num}"

        evidence_labels = {}
        evidence_lines = []
        for i, (dia_id, mem_id) in enumerate(gt_turns.items(), 1):
            label = f"E{i}"
            evidence_labels[label] = mem_id
            content = memories.get(mem_id, "[missing]")
            if len(content) > 500:
                content = content[:500] + "..."
            evidence_lines.append(f"  {label} ({dia_id}): {content}")

        synthetic_labels = {}
        synthetic_lines = []
        for i, syn_id in enumerate(syn_ids, 1):
            label = f"S{i}"
            synthetic_labels[label] = syn_id
            content = memories.get(syn_id, "[missing]")
            if len(content) > 500:
                content = content[:500] + "..."
            synthetic_lines.append(f"  {label}: {content}")

        q_metadata[q_label] = {
            "evidence_labels": evidence_labels,
            "synthetic_labels": synthetic_labels,
            "qkey": qkey,
        }

        block = (
            f"--- Question {q_label} (category: {cat}) ---\n"
            f"Question: {q['question']}\n"
            f"Answer: {q.get('answer', '')}\n\n"
            f"Evidence turns:\n" + "\n".join(evidence_lines) + "\n\n"
            f"Synthetic nodes:\n" + "\n".join(synthetic_lines) + "\n"
        )
        blocks.append(block)

    prompt = "\n".join(blocks) + BATCH_FOOTER
    full_prompt = SYSTEM_PROMPT + "\n\n" + prompt

    print(f"Prompt: {len(full_prompt)} chars, {q_num} questions, {len(cases)} synthetics")
    print()

    # Call Opus
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    claude_cmd = "claude.cmd" if sys.platform == "win32" else "claude"

    result = subprocess.run(
        [claude_cmd, "-p", "--model", "claude-opus-4-6"],
        input=full_prompt,
        capture_output=True,
        timeout=300,
        env=env,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return

    response = result.stdout.strip()
    print(f"Response:\n{response}\n")

    # Parse and compare with original Opus
    print("\n" + "=" * 60)
    print("  Verification: Original Opus vs Re-judged Opus")
    print("=" * 60)

    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"Q(\d+)/S(\d+)\s*:\s*(.*)", line, re.IGNORECASE)
        if not m:
            continue

        q_label = f"Q{m.group(1)}"
        s_label = f"S{m.group(2)}"
        covered_str = m.group(3).strip().lower()

        if q_label not in q_metadata:
            continue
        qinfo = q_metadata[q_label]
        qkey = qinfo["qkey"]
        syn_id = qinfo["synthetic_labels"].get(s_label)
        if not syn_id:
            continue

        # Parse new coverage
        if covered_str == "none" or not covered_str:
            new_turns = set()
        else:
            new_turns = set()
            for token in re.findall(r"E\d+", covered_str, re.IGNORECASE):
                turn_id = qinfo["evidence_labels"].get(token.upper())
                if turn_id:
                    new_turns.add(turn_id)

        orig_turns = set(opus[qkey].get(syn_id, []))
        haiku_turns = set(haiku.get(qkey, {}).get(syn_id, []))

        status = "CONSISTENT" if new_turns == orig_turns else "FLIPPED"
        if new_turns == haiku_turns and status == "FLIPPED":
            status = "FLIPPED → agrees with Haiku"

        print(f"  {qkey} syn={syn_id[:12]}: {status}")
        if status != "CONSISTENT":
            print(f"    Original Opus: {sorted(orig_turns)}")
            print(f"    Re-judged:     {sorted(new_turns)}")
            print(f"    Haiku:         {sorted(haiku_turns)}")


if __name__ == "__main__":
    main()
