# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
LoCoMo LLM judge (batched): scores reader answers as CORRECT/WRONG.

Sends questions in batches to a single LLM call instead of per-question
subprocess calls. Cat 5 and abstentions scored mechanically.

Supports corrected GT: when --corrected-gt is passed, questions with
known GT errors are judged against both original and corrected answers.
Report shows both original-GT and corrected-GT accuracy.

Usage:
  python scripts/locomo_bench/judge_batch.py --input <run_dir>/results.jsonl
  python scripts/locomo_bench/judge_batch.py --input <run_dir>/results.jsonl --corrected-gt
  python scripts/locomo_bench/judge_batch.py --input <run_dir>/results.jsonl --model claude-opus-4-6 --batch-size 200
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}

DEFAULT_MODEL = "claude-opus-4-6"
DEFAULT_BATCH_SIZE = 200

ADVERSARIAL_PHRASES = [
    "no information available", "not mentioned", "unanswerable",
    "cannot determine", "not answerable", "cannot be determined",
    "no mention", "not enough information", "not discussed",
    "not specified", "not addressed", "not indicated",
    "no evidence", "not covered", "not stated",
    "cannot be answered", "no relevant information",
    "not provided", "not include",
]

BATCH_PROMPT = """You are a judge grading answers to questions. For each numbered item, respond with ONLY the number and CORRECT or WRONG.

Grading rules:
- Be generous. Same core meaning = CORRECT. Synonyms, paraphrases, rewordings all count.
- Date format differences don't matter. "May 7" = "7 May 2023" = "May 7th, 2023".
- Relative dates resolving to approximately the correct date = CORRECT.
- Partial list answers: if the key parts are covered, CORRECT.
- WRONG only if factually incorrect, completely unrelated, or misses the core point.

Respond in exactly this format, one per line:
1. CORRECT
2. WRONG
3. CORRECT
...

Here are the items to judge:

"""


def score_mechanical(record: dict) -> dict | None:
    """Return mechanical judgment if applicable, else None (needs LLM)."""
    gen_lower = record["generated_answer"].lower()

    if record["category"] == 5:
        abstained = any(p in gen_lower for p in ADVERSARIAL_PHRASES)
        return {
            "label": "CORRECT" if abstained else "WRONG",
            "reasoning": "mechanical: adversarial " + ("abstention" if abstained else "specific answer"),
        }

    if any(p in gen_lower for p in ADVERSARIAL_PHRASES):
        return {"label": "WRONG", "reasoning": "mechanical: abstained on answerable question"}

    return None


def format_batch_item(idx: int, record: dict, gold_override: str | None = None) -> str:
    """Format one item for the batch prompt."""
    cat_name = CATEGORY_NAMES.get(record["category"], "unknown")
    gold = gold_override or record["ground_truth"]
    return (
        f"{idx}. [{cat_name}]\n"
        f"   Q: {record['question']}\n"
        f"   Gold: {gold}\n"
        f"   Answer: {record['generated_answer']}\n"
    )


def judge_batch(items: list[tuple[int, dict]], model: str,
                gold_overrides: dict[int, str] | None = None) -> dict[int, str]:
    """Send a batch to the LLM, return {idx: label} mapping."""
    prompt = BATCH_PROMPT
    for idx, record in items:
        override = gold_overrides.get(idx) if gold_overrides else None
        prompt += format_batch_item(idx, record, gold_override=override) + "\n"

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
                break
            err = result.stderr.strip()
        except subprocess.TimeoutExpired:
            err = "timeout"

        if attempt < 2:
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"claude -p failed after 3 attempts: {err}")

    output = result.stdout.strip()
    return parse_batch_response(output, [idx for idx, _ in items])


def parse_batch_response(output: str, expected_ids: list[int]) -> dict[int, str]:
    """Parse numbered CORRECT/WRONG lines from LLM response."""
    results = {}
    for line in output.splitlines():
        line = line.strip()
        m = re.match(r"(\d+)\.\s*(CORRECT|WRONG)", line, re.IGNORECASE)
        if m:
            idx = int(m.group(1))
            label = m.group(2).upper()
            results[idx] = label

    for idx in expected_ids:
        if idx not in results:
            results[idx] = "UNKNOWN"

    return results


def token_f1(pred: str, gold: str) -> float:
    """Compute token-level F1 between prediction and gold."""
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = sum(1 for t in pred_tokens if t in gold_tokens)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def run_judging(records: list[dict], model: str, batch_size: int,
                out_file, gold_overrides: dict | None = None,
                label_key: str = "judge_label",
                reasoning_key: str = "judge_reasoning",
                f1_key: str | None = None,
                gt_key: str | None = None):
    """Run mechanical + batched LLM judging, write results to out_file.

    Returns (correct_count, total_count).
    """
    mechanical = []
    needs_llm = []
    for r in records:
        judgment = score_mechanical(r)
        if judgment:
            mechanical.append((r, judgment))
        else:
            needs_llm.append(r)

    print(f"Mechanical: {len(mechanical)} (cat5={sum(1 for r, _ in mechanical if r['category'] == 5)}, "
          f"abstentions={sum(1 for r, _ in mechanical if r['category'] != 5)})")
    print(f"Need LLM: {len(needs_llm)} in {(len(needs_llm) + batch_size - 1) // batch_size} batches")

    correct_count = 0
    total_count = 0

    # Write mechanical results
    for r, judgment in mechanical:
        r[label_key] = judgment["label"]
        r[reasoning_key] = judgment["reasoning"]
        if f1_key and gt_key and gt_key in r:
            r[f1_key] = round(token_f1(r["generated_answer"], r[gt_key]), 4)
        out_file.write(json.dumps(r) + "\n")
        if judgment["label"] == "CORRECT":
            correct_count += 1
        total_count += 1

    out_file.flush()
    if mechanical:
        print(f"  Wrote {len(mechanical)} mechanical results")

    # Batched LLM judging
    t_start = time.time()
    batches = [needs_llm[i:i + batch_size] for i in range(0, len(needs_llm), batch_size)]

    for batch_idx, batch in enumerate(batches):
        items = [(i + 1, r) for i, r in enumerate(batch)]

        # Build gold overrides for this batch
        batch_overrides = None
        if gold_overrides:
            batch_overrides = {}
            for i, r in enumerate(batch):
                key = (r["conv_id"], r.get("qa_idx", -1))
                if key in gold_overrides:
                    batch_overrides[i + 1] = gold_overrides[key]

        try:
            labels = judge_batch(items, model, gold_overrides=batch_overrides)
        except Exception as e:
            print(f"  BATCH {batch_idx + 1} ERROR: {e}")
            labels = {i + 1: "ERROR" for i in range(len(batch))}

        unknowns = sum(1 for v in labels.values() if v == "UNKNOWN")
        batch_correct = 0

        for i, r in enumerate(batch):
            label = labels.get(i + 1, "UNKNOWN")
            r[label_key] = label
            r[reasoning_key] = "batch"
            if f1_key and gt_key and gt_key in r:
                r[f1_key] = round(token_f1(r["generated_answer"], r[gt_key]), 4)
            out_file.write(json.dumps(r) + "\n")
            if label == "CORRECT":
                correct_count += 1
                batch_correct += 1
            total_count += 1

        out_file.flush()
        elapsed = time.time() - t_start
        acc = correct_count / total_count if total_count else 0
        print(
            f"  Batch {batch_idx + 1}/{len(batches)} "
            f"({len(batch)} items) "
            f"{batch_correct}/{len(batch)} correct "
            f"| {unknowns} unknown "
            f"| Running acc={acc:.1%} "
            f"| {elapsed:.0f}s elapsed"
        )

    return correct_count, total_count


def main():
    parser = argparse.ArgumentParser(description="LoCoMo LLM judge (batched)")
    parser.add_argument("--input", type=str, required=True,
                        help="Reader results JSONL to judge")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path (default: <input_dir>/judge_<model>.jsonl)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Judge model")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Questions per LLM call")
    parser.add_argument("--limit", type=int, default=0, help="Max questions (0=all)")
    parser.add_argument("--corrected-gt", action="store_true",
                        help="Also judge against corrected GT (reports both)")

    args = parser.parse_args()

    reader_path = Path(args.input)
    if not reader_path.exists():
        print(f"ERROR: Reader output not found: {reader_path}")
        sys.exit(1)

    # Output path
    model_short = args.model.replace("claude-", "").replace("-4-6", "").replace("-4-5", "")
    if args.output:
        judge_output = Path(args.output)
    else:
        judge_output = reader_path.parent / f"judge_{model_short}.jsonl"

    records = [json.loads(l) for l in open(reader_path) if l.strip()]
    if args.limit:
        records = records[:args.limit]
    print(f"Loaded {len(records)} reader results")
    print(f"Output: {judge_output}")

    # Load corrected GT if requested
    corrections = None
    corrected_overrides = None
    if args.corrected_gt:
        from locomo_bench.corrected_gt import load_corrections, get_corrected_answer
        try:
            corrections = load_corrections()
            # Build override map: (conv_id, qa_idx) -> corrected_answer
            corrected_overrides = {}
            for (conv_idx, qa_idx), entry in corrections.items():
                corrected_overrides[(conv_idx, qa_idx)] = entry["correct_answer"]
            # Tag records with corrected GT
            for r in records:
                key = (r["conv_id"], r.get("qa_idx", -1))
                if key in corrected_overrides:
                    r["corrected_gt"] = corrected_overrides[key]
            n_affected = sum(1 for r in records if "corrected_gt" in r)
            print(f"Corrected GT: {len(corrected_overrides)} corrections, {n_affected} affect this run")
        except FileNotFoundError:
            print("WARNING: Corrected GT file not found, judging with original GT only")
            args.corrected_gt = False

    # Resume support
    done_keys = set()
    if args.resume and judge_output.exists():
        for line in open(judge_output):
            r = json.loads(line)
            done_keys.add((r["conv_id"], r["question"]))
        print(f"Resuming: {len(done_keys)} already judged")

    remaining = [r for r in records if (r["conv_id"], r["question"]) not in done_keys]
    print(f"Questions to judge: {len(remaining)}")

    if not remaining:
        print("Nothing to do.")
        print_report(judge_output, corrected=args.corrected_gt)
        return

    mode = "a" if args.resume else "w"
    out = open(judge_output, mode)

    # Pass 1: judge against original GT
    print(f"\n--- Original GT judging ({args.model}) ---")
    run_judging(remaining, args.model, args.batch_size, out)

    # Pass 2: judge corrected GT questions against corrected answers
    if args.corrected_gt and corrected_overrides:
        corrected_questions = [r for r in remaining if "corrected_gt" in r]
        if corrected_questions:
            print(f"\n--- Corrected GT judging ({len(corrected_questions)} questions) ---")
            # Seek back and rewrite — instead, we already have the records in memory.
            # Just do a second pass on the corrected questions and update in place.
            # We need to re-judge only the corrected ones with swapped gold.
            corrected_needs_llm = []
            for r in corrected_questions:
                mech = score_mechanical(r)
                if mech:
                    r["corrected_label"] = mech["label"]
                    r["corrected_f1"] = round(token_f1(r["generated_answer"], r["corrected_gt"]), 4)
                else:
                    corrected_needs_llm.append(r)

            # Batch judge corrected questions
            if corrected_needs_llm:
                batches = [corrected_needs_llm[i:i + args.batch_size]
                           for i in range(0, len(corrected_needs_llm), args.batch_size)]
                for batch_idx, batch in enumerate(batches):
                    items = [(i + 1, r) for i, r in enumerate(batch)]
                    batch_overrides = {}
                    for i, r in enumerate(batch):
                        batch_overrides[i + 1] = r["corrected_gt"]
                    try:
                        labels = judge_batch(items, args.model, gold_overrides=batch_overrides)
                    except Exception as e:
                        print(f"  Corrected BATCH {batch_idx + 1} ERROR: {e}")
                        labels = {i + 1: "ERROR" for i in range(len(batch))}

                    for i, r in enumerate(batch):
                        r["corrected_label"] = labels.get(i + 1, "UNKNOWN")
                        r["corrected_f1"] = round(token_f1(r["generated_answer"], r["corrected_gt"]), 4)

                    print(f"  Corrected batch {batch_idx + 1}/{len(batches)} done")

    out.close()

    # Rewrite output with all fields (original + corrected labels)
    with open(judge_output, "w") as f:
        for r in records:
            if (r["conv_id"], r["question"]) in done_keys:
                continue
            f.write(json.dumps(r) + "\n")

    # Also append resumed records from existing file
    if args.resume and done_keys:
        # The resumed records are already in the file from the previous run
        pass

    print(f"\nResults saved to {judge_output}")
    print_report(judge_output, corrected=args.corrected_gt)


def print_report(path: Path, corrected: bool = False):
    """Print aggregate judge report."""
    records = [json.loads(l) for l in open(path) if l.strip()]

    for gt_label, label_key, title in [
        ("Original GT", "judge_label", "Original GT"),
        ("Corrected GT", "corrected_label", "Corrected GT"),
    ]:
        if label_key == "corrected_label" and not corrected:
            continue

        cat_metrics = defaultdict(lambda: {"count": 0, "correct": 0, "ret_hit": 0, "unknown": 0})

        for r in records:
            cat = r["category"]
            label = r.get(label_key)
            if label is None:
                # For corrected_label, use judge_label if no correction applies
                label = r.get("judge_label", "UNKNOWN")
            cat_metrics[cat]["count"] += 1
            if label == "CORRECT":
                cat_metrics[cat]["correct"] += 1
            if label in ("UNKNOWN", "ERROR"):
                cat_metrics[cat]["unknown"] += 1
            cat_metrics[cat]["ret_hit"] += int(r.get("retrieval_hit", False))

        print(f"\n{'=' * 60}")
        print(f"LoCoMo End-to-End QA Results — {title}")
        print(f"{'=' * 60}")
        print(f"{'Category':<20} {'N':>5} {'Acc':>7} {'RetHit':>7} {'Unk':>5}")
        print("-" * 50)

        total_n = total_correct = 0
        main_n = main_correct = 0

        for cat in sorted(cat_metrics):
            m = cat_metrics[cat]
            n = m["count"]
            name = CATEGORY_NAMES.get(cat, f"cat_{cat}")
            acc = m["correct"] / n if n else 0
            ret = m["ret_hit"] / n if n else 0
            print(f"{name:<20} {n:>5} {acc:>7.1%} {ret:>7.1%} {m['unknown']:>5}")
            total_n += n
            total_correct += m["correct"]
            if cat != 5:
                main_n += n
                main_correct += m["correct"]

        if total_n:
            print("-" * 50)
            print(f"{'ALL (1-5)':<20} {total_n:>5} {total_correct / total_n:>7.1%}")
        if main_n:
            print(f"{'OVERALL (1-4)':<20} {main_n:>5} {main_correct / main_n:>7.1%}")

        print()


if __name__ == "__main__":
    main()
