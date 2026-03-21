"""Standalone ground truth judge — no dependencies beyond stdlib + claude CLI.

Reads pre-exported candidate sets and uses claude -p to grade relevance.
Designed to run on a machine with only the claude CLI installed.

Usage:
    python judge_ground_truth.py                          # run all queries
    python judge_ground_truth.py --resume                 # skip already-judged
    python judge_ground_truth.py --max-queries 10         # limit queries
    python judge_ground_truth.py --dry-run                # show stats only
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# LLM judge prompt (identical to build_ground_truth.py)
# ---------------------------------------------------------------------------

GROUP_SIZE = 10  # candidates per scoring group

JUDGE_SYSTEM = """You are evaluating memory relevance for a personal AI memory retrieval system.

Given a query and candidate memories organized into small groups, score each memory's
relevance on a continuous 0.0-1.0 scale.

## Calibration Anchors

- **0.0**: Completely irrelevant — no topical overlap whatsoever
- **0.1-0.2**: Shares a keyword but discusses an unrelated topic (e.g., "training" in ML context vs sports context)
- **0.2-0.3**: Same broad domain but different sub-topic (e.g., query about chess openings, memory about chess rating systems)
- **0.3-0.5**: Useful background — related context that provides supporting information but doesn't directly address the query
- **0.5-0.7**: Directly addresses part of the query — contains information the searcher would want to see
- **0.7-0.9**: Directly and substantially addresses the query — this is what the searcher was looking for
- **0.9-1.0**: Essential — this memory is exactly what the query targets, contains the specific answer or information sought

## Scoring Guidelines

- Use the full scale. Most memories should score below 0.5.
- A memory that merely mentions the same topic is NOT automatically relevant — it must address the query's intent.
- Break ties with as much precision as you like (e.g., 0.35 vs 0.38).
- Consider the query's likely intent: what would someone searching for this actually want to find?

## Response Format

Each group header lists its member IDs in order. Return a JSON object where keys are
group numbers (as strings) and values are arrays of scores in the SAME order as the IDs
in that group's header. Each array MUST have exactly as many scores as IDs in that group.

Respond with ONLY the JSON object. No other text.
Example for 2 groups of 3:
{"1": [0.85, 0.0, 0.42], "2": [0.0, 0.65, 0.12]}"""


def build_judge_prompt(query: str, candidates: list[dict], vector_input: str = "") -> str:
    parts = [f"## Query (keyword search)\n{query}\n"]
    if vector_input:
        parts.append(f"\n## Context (what was actually being looked for)\n{vector_input}\n")

    # Organize candidates into groups
    groups = []
    for i in range(0, len(candidates), GROUP_SIZE):
        groups.append(candidates[i:i + GROUP_SIZE])

    parts.append(f"\n## Candidate Memories ({len(candidates)} total, {len(groups)} groups)\n")
    for gi, group in enumerate(groups, 1):
        ids = [c["id"][:12] for c in group]
        parts.append(f"### Group {gi} (IDs: {', '.join(ids)})\n")
        for c in group:
            themes = c.get("themes", "[]")
            parts.append(
                f"**[{c['id'][:12]}]** ({c['category']}) {c.get('summary', '(no summary)')}\n"
                f"Themes: {themes}\n"
                f"{c['content']}\n"
            )
    return "\n".join(parts)


def _call_claude(prompt: str, model: str) -> str | None:
    """Call claude -p and return stripped stdout, or None on failure."""
    claude_cmd = "claude.cmd" if sys.platform == "win32" else "claude"
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)

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
        print(f"    WARNING: claude -p failed after 3 attempts: {err}")
        return None

    text = result.stdout.strip()
    if not text:
        return None

    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    return text


def judge_relevance(
    query: str,
    candidates: list[dict],
    model: str,
    vector_input: str = "",
    batch_size: int = 50,
) -> tuple[dict[str, float], dict[str, float]]:
    """Returns (relevant_scores, all_scores) where relevant has score >= 0.1."""
    if not candidates:
        return {}, {}

    if len(candidates) > batch_size:
        merged_relevant = {}
        merged_all = {}
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            batch_rel, batch_all = judge_relevance(query, batch, model, vector_input, batch_size=len(batch) + 1)
            merged_relevant.update(batch_rel)
            merged_all.update(batch_all)
        return merged_relevant, merged_all

    prompt = JUDGE_SYSTEM + "\n\n" + build_judge_prompt(query, candidates, vector_input)

    text = _call_claude(prompt, model)
    if not text:
        print(f"    WARNING: Empty response for query: {query[:60]}")
        return {}, {}

    try:
        grouped_scores = json.loads(text)
    except json.JSONDecodeError:
        print(f"    WARNING: Failed to parse judge response for query: {query[:60]}")
        print(f"    Response: {text[:200]}")
        return {}, {}

    # Build ordered ID list matching the groups
    groups = []
    for i in range(0, len(candidates), GROUP_SIZE):
        groups.append(candidates[i:i + GROUP_SIZE])

    resolved = {}
    all_scores = {}
    for gi, group in enumerate(groups, 1):
        key = str(gi)
        score_array = grouped_scores.get(key)
        if not isinstance(score_array, list):
            print(f"    WARNING: Missing or invalid group {key} for query: {query[:60]}")
            continue
        if len(score_array) != len(group):
            print(f"    WARNING: Group {key} length mismatch: expected {len(group)}, got {len(score_array)}"
                  f" for query: {query[:60]}")
            # Still use what we can if lengths match partially
            if len(score_array) > len(group):
                score_array = score_array[:len(group)]
            # If fewer scores than candidates, skip this group entirely
            else:
                continue
        for c, score in zip(group, score_array):
            if isinstance(score, (int, float)):
                full_id = c["id"]
                all_scores[full_id] = round(float(score), 4)
                if score >= 0.1:
                    resolved[full_id] = round(float(score), 4)

    return resolved, all_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Judge memory relevance from pre-exported candidates")
    ts_dir = str(Path.home() / ".claude" / "data" / "tuning_studies")
    parser.add_argument("--candidates", type=str,
                        default=f"{ts_dir}/gt_candidates.json",
                        help="Path to exported candidates JSON")
    parser.add_argument("--output", type=str,
                        default=f"{ts_dir}/gt_v2.json",
                        help="Output path")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6",
                        help="Model for LLM judge (default: claude-sonnet-4-6)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output, skip already-judged queries")
    parser.add_argument("--max-queries", type=int, default=0,
                        help="Limit number of queries to process (0 = all)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Max candidates per judge call (default: 50)")
    parser.add_argument("--queries", type=str,
                        default=f"{ts_dir}/gt_v2_queries.json",
                        help="Path to JSON query set to judge (filters candidates)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (default: 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show stats without calling LLM")
    args = parser.parse_args()

    # Load candidates
    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"ERROR: Candidates file not found: {candidates_path}")
        sys.exit(1)

    with open(candidates_path) as f:
        all_candidates = json.load(f)
    print(f"Loaded {len(all_candidates)} queries from {candidates_path}")

    # Filter to specific query set if provided
    if args.queries:
        queries_path = Path(args.queries)
        if not queries_path.exists():
            print(f"ERROR: Queries file not found: {queries_path}")
            sys.exit(1)
        with open(queries_path) as f:
            queries_data = json.load(f)
        query_set = set(queries_data.keys()) if isinstance(queries_data, dict) else set(queries_data)
        before = len(all_candidates)
        all_candidates = {q: data for q, data in all_candidates.items() if q in query_set}
        print(f"Filtered to {len(all_candidates)}/{before} queries from {queries_path}")

    # Load existing results if resuming
    output_path = Path(args.output)
    ground_truth = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            ground_truth = json.load(f)
        print(f"Resuming: {len(ground_truth)} queries already judged")

    # Filter to pending queries
    pending = [(q, data) for q, data in all_candidates.items() if q not in ground_truth]
    if args.max_queries > 0:
        pending = pending[:args.max_queries]
    print(f"Queries to process: {len(pending)}")

    if args.dry_run:
        total_cands = sum(len(data["candidates"]) for _, data in pending)
        avg = total_cands / len(pending) if pending else 0
        print(f"Total candidates: {total_cands}")
        print(f"Avg candidates per query: {avg:.0f}")
        est_tokens = total_cands * 200  # rough: 200 tokens/memory
        print(f"Estimated input tokens: {est_tokens:,.0f}")
        return

    total_relevant = 0
    total_candidates = 0
    start_time = time.time()

    # Full scores file (includes < 0.1 and candidate IDs)
    full_scores_path = output_path.with_suffix(".full.json")
    full_scores = {}
    if args.resume and full_scores_path.exists():
        with open(full_scores_path) as f:
            full_scores = json.load(f)

    def _judge_one(item):
        """Judge a single query. Returns (query, relevant, full_entry, n_candidates)."""
        query, data = item
        candidates = data["candidates"]
        vector_input = data.get("vector_input", "")
        if not candidates:
            return query, {}, {"candidate_ids": [], "scores": {}}, 0
        relevant, all_scored = judge_relevance(query, candidates, args.model,
                                               vector_input, batch_size=args.batch_size)
        candidate_ids = [c["id"] for c in candidates]
        full_entry = {"candidate_ids": candidate_ids, "scores": all_scored}
        return query, relevant, full_entry, len(candidates)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_judge_one, item): item[0] for item in pending}
        for future in as_completed(futures):
            completed += 1
            query, relevant, full_entry, n_cands = future.result()
            ground_truth[query] = relevant
            full_scores[query] = full_entry
            total_relevant += len(relevant)
            total_candidates += n_cands

            elapsed = time.time() - start_time
            rate = (completed / elapsed * 60) if elapsed > 0 else 0
            n_rejected = len(full_entry["scores"]) - len(relevant)
            print(f"  [{completed}/{len(pending)}] ({rate:.1f}/min) "
                  f"{n_cands} cands -> {len(relevant)} rel, {n_rejected} rej  "
                  f"{query[:60]}")

            # Save checkpoint after every query
            with open(output_path, "w") as f:
                json.dump(ground_truth, f, indent=2)
            with open(full_scores_path, "w") as f:
                json.dump(full_scores, f, indent=2)

    # Final save
    with open(output_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    with open(full_scores_path, "w") as f:
        json.dump(full_scores, f, indent=2)

    elapsed = time.time() - start_time
    n_judged = len(ground_truth)
    print(f"\nDone. {n_judged} queries judged in {elapsed:.0f}s")
    print(f"Total candidates evaluated: {total_candidates}")
    print(f"Total relevant memories found: {total_relevant}")
    if n_judged:
        print(f"Avg relevant per query: {total_relevant / n_judged:.1f}")
    print(f"Output: {output_path}")
    print(f"Full scores (with rejections): {full_scores_path}")


if __name__ == "__main__":
    main()
