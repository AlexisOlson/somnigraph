"""Standalone ground truth judge -- no dependencies beyond stdlib + claude CLI.

Reads pre-exported candidate sets and uses claude -p to grade relevance.
Designed to run on a machine with only the claude CLI installed.

Usage:
    python judge_ground_truth.py                          # run all queries
    python judge_ground_truth.py --resume                 # skip already-judged
    python judge_ground_truth.py --max-queries 10         # limit queries
    python judge_ground_truth.py --dry-run                # show stats only

Migration notes (production -> somnigraph):
- Kept standalone (no memory package imports) for portability.
- Identical judge prompt and scoring logic as build_ground_truth.py.
- The handoff zip (ground_truth_handoff.zip) includes a copy of this script
  that can run independently of the somnigraph repo.
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

JUDGE_SYSTEM = """You are evaluating memory relevance for a personal AI memory retrieval system.

Given a query (the search used to recall memories) and a list of candidate memories,
score each memory's relevance to the query on a continuous 0.0-1.0 scale.

## Calibration Anchors

Use these reference points to anchor your scores:

- **0.0**: Completely irrelevant -- no topical overlap whatsoever
- **0.1-0.2**: Shares a keyword but discusses an unrelated topic (e.g., "training" in ML context vs sports context)
- **0.2-0.3**: Same broad domain but different sub-topic (e.g., query about chess openings, memory about chess rating systems)
- **0.3-0.5**: Useful background -- related context that provides supporting information but doesn't directly address the query
- **0.5-0.7**: Directly addresses part of the query -- contains information the searcher would want to see
- **0.7-0.9**: Directly and substantially addresses the query -- this is what the searcher was looking for
- **0.9-1.0**: Essential -- this memory is exactly what the query targets, contains the specific answer or information sought

## Scoring Guidelines

- Use the full scale. Most memories should score below 0.5.
- A memory that merely mentions the same topic is NOT automatically relevant -- it must address the query's intent.
- Break ties with as much precision as you like (e.g., 0.35 vs 0.38).
- Consider the query's likely intent: what would someone searching for this actually want to find?

IMPORTANT: Only include memories scoring >= 0.1. Omit irrelevant ones entirely.

Respond with ONLY a JSON object mapping memory IDs to scores. No other text.
Example: {"abc123": 0.85, "def456": 0.42}"""


def build_judge_prompt(query: str, candidates: list[dict], vector_input: str = "") -> str:
    parts = [f"## Query (keyword search)\n{query}\n"]
    if vector_input:
        parts.append(f"\n## Context (what was actually being looked for)\n{vector_input}\n")
    parts.append(f"\n## Candidate Memories ({len(candidates)} total)\n")
    for c in candidates:
        themes = c.get("themes", "[]")
        parts.append(
            f"### [{c['id'][:12]}] ({c['category']}) {c.get('summary', '(no summary)')}\n"
            f"Themes: {themes}\n"
            f"{c['content']}\n"
        )
    return "\n".join(parts)


def judge_relevance(
    query: str,
    candidates: list[dict],
    model: str,
    vector_input: str = "",
    batch_size: int = 50,
) -> dict[str, float]:
    if not candidates:
        return {}

    if len(candidates) > batch_size:
        all_scores = {}
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            batch_scores = judge_relevance(query, batch, model, vector_input, batch_size=len(batch) + 1)
            all_scores.update(batch_scores)
        return all_scores

    prompt = JUDGE_SYSTEM + "\n\n" + build_judge_prompt(query, candidates, vector_input)

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
        return {}

    text = result.stdout.strip()
    if not text:
        print(f"    WARNING: Empty response for query: {query[:60]}")
        return {}

    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        scores = json.loads(text)
    except json.JSONDecodeError:
        print(f"    WARNING: Failed to parse judge response for query: {query[:60]}")
        print(f"    Response: {text[:200]}")
        return {}

    # Resolve short IDs back to full IDs
    id_map = {}
    for c in candidates:
        id_map[c["id"][:12]] = c["id"]
        id_map[c["id"]] = c["id"]

    resolved = {}
    for kid, score in scores.items():
        full_id = id_map.get(kid)
        if full_id and isinstance(score, (int, float)) and score >= 0.1:
            resolved[full_id] = round(float(score), 4)

    return resolved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Judge memory relevance from pre-exported candidates")
    parser.add_argument("--candidates", type=str, default="candidates.json",
                        help="Path to exported candidates JSON (default: candidates.json)")
    parser.add_argument("--output", type=str, default="ground_truth.json",
                        help="Output path (default: ground_truth.json)")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6",
                        help="Model for LLM judge (default: claude-sonnet-4-6)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output, skip already-judged queries")
    parser.add_argument("--max-queries", type=int, default=0,
                        help="Limit number of queries to process (0 = all)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Max candidates per judge call (default: 50)")
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

    for i, (query, data) in enumerate(pending):
        elapsed = time.time() - start_time
        rate = (i / elapsed * 60) if elapsed > 0 and i > 0 else 0

        candidates = data["candidates"]
        vector_input = data.get("vector_input", "")
        total_candidates += len(candidates)

        print(f"  [{i+1}/{len(pending)}] ({rate:.1f}/min) {query[:70]}")

        if not candidates:
            ground_truth[query] = {}
            continue

        scores = judge_relevance(query, candidates, args.model, vector_input, batch_size=args.batch_size)
        ground_truth[query] = scores
        total_relevant += len(scores)

        print(f"    {len(candidates)} candidates -> {len(scores)} relevant")

        # Save checkpoint every 10 queries
        if (i + 1) % 10 == 0:
            with open(output_path, "w") as f:
                json.dump(ground_truth, f, indent=2)
            print(f"    [saved checkpoint: {len(ground_truth)} queries]")

    # Final save
    with open(output_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    elapsed = time.time() - start_time
    n_judged = len(ground_truth)
    print(f"\nDone. {n_judged} queries judged in {elapsed:.0f}s")
    print(f"Total candidates evaluated: {total_candidates}")
    print(f"Total relevant memories found: {total_relevant}")
    if n_judged:
        print(f"Avg relevant per query: {total_relevant / n_judged:.1f}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
