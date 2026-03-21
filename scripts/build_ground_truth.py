# /// script
# requires-python = ">=3.11"
# dependencies = ["openai", "sqlite-vec", "numpy", "lightgbm"]
# ///
"""Build unbiased ground truth for memory retrieval evaluation.

For each historical query, retrieves a deep candidate set (union of vector + keyword
results at large budget), then asks an LLM to grade relevance on a continuous 0-1 scale.

Output: JSON file mapping {query -> {memory_id: relevance_score, ...}}
Only memories judged >= 0.1 relevance are included.

Usage:
    uv run scripts/build_ground_truth.py --dry-run
    uv run scripts/build_ground_truth.py --export-candidates out.json
    uv run scripts/build_ground_truth.py --resume --max-queries 200

Migration notes (production → somnigraph):
- DB_PATH, get_db(), serialize_f32() imported from memory package (were local)
- get_embedding() replaced by embed_text() from memory.embeddings
- OpenAI client init handled by memory.embeddings (env var or DATA_DIR/openai_api_key)
- OUTPUT_PATH uses DATA_DIR instead of hardcoded ~/.claude/data/
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup — add src/ to path for memory package imports
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memory import DB_PATH
from memory.constants import DATA_DIR
from memory.db import get_db
from memory.embeddings import embed_text
from memory.vectors import serialize_f32

OUTPUT_PATH = DATA_DIR / "ground_truth.json"


# ---------------------------------------------------------------------------
# Candidate retrieval
# ---------------------------------------------------------------------------

def retrieve_candidates(
    db: sqlite3.Connection,
    query: str,
    query_embedding: list[float],
    top_k: int = 200,
) -> list[dict]:
    """Deep candidate retrieval: union of vector + keyword top_k results.

    Returns list of {id, summary, content, category, themes, source} dicts.
    """
    # Vector search
    vec_blob = serialize_f32(query_embedding)
    vec_rows = db.execute("""
        SELECT rm.memory_id, distance
        FROM memory_vec v
        JOIN memory_rowid_map rm ON rm.rowid = v.rowid
        WHERE v.embedding MATCH ? AND k = ?
    """, (vec_blob, top_k)).fetchall()
    vec_ids = {r["memory_id"] for r in vec_rows}

    # Keyword search (BM25 via FTS5)
    # Escape special characters for FTS5
    safe_query = query.replace('"', '""')
    tokens = safe_query.split()
    # Use OR matching for broad recall
    fts_query = " OR ".join(f'"{t}"' for t in tokens if len(t) > 1)
    fts_ids = set()
    if fts_query:
        try:
            fts_rows = db.execute(f"""
                SELECT rm.memory_id
                FROM memory_fts f
                JOIN memory_rowid_map rm ON rm.rowid = f.rowid
                WHERE memory_fts MATCH ?
                ORDER BY bm25(memory_fts, 5.0, 3.0)
                LIMIT ?
            """, (fts_query, top_k)).fetchall()
            fts_ids = {r["memory_id"] for r in fts_rows}
        except sqlite3.OperationalError:
            pass  # malformed FTS query

    # Union
    all_ids = vec_ids | fts_ids
    if not all_ids:
        return []

    # Fetch memory content
    ph = ",".join("?" * len(all_ids))
    id_list = list(all_ids)
    rows = db.execute(f"""
        SELECT id, summary, content, category, themes, source
        FROM memories
        WHERE id IN ({ph}) AND status = 'active'
    """, id_list).fetchall()

    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """You are evaluating memory relevance for a personal AI memory retrieval system.

Given a query (the search used to recall memories) and a list of candidate memories,
score each memory's relevance to the query on a continuous 0.0-1.0 scale.

## Calibration Anchors

Use these reference points to anchor your scores:

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

IMPORTANT: Only include memories scoring >= 0.1. Omit irrelevant ones entirely.

Respond with ONLY a JSON object mapping memory IDs to scores. No other text.
Example: {"abc123": 0.85, "def456": 0.42}"""


def build_judge_prompt(query: str, candidates: list[dict], vector_input: str = "") -> str:
    """Build the user prompt for the LLM judge."""
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
    """Ask LLM (via claude -p) to grade relevance of candidates to query. Returns {id: score}."""
    if not candidates:
        return {}

    if len(candidates) > batch_size:
        all_scores = {}
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            batch_scores = judge_relevance(query, batch, model, vector_input, batch_size=len(batch) + 1)  # prevent re-splitting
            all_scores.update(batch_scores)
        return all_scores

    prompt = JUDGE_SYSTEM + "\n\n" + build_judge_prompt(query, candidates, vector_input)

    import subprocess

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

    # Handle markdown code blocks
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

def extract_queries(db: sqlite3.Connection) -> list[dict]:
    """Extract unique queries that have retrieval history, with vector_input context."""
    rows = db.execute("""
        SELECT query, COUNT(*) as n
        FROM memory_events WHERE event_type='retrieved'
        AND query IS NOT NULL AND query != ''
        GROUP BY query ORDER BY n DESC
    """).fetchall()
    queries = [{"query": r["query"], "count": r["n"]} for r in rows]

    # Attach vector_input (context used for vector search) from recall_meta events
    meta_rows = db.execute("""
        SELECT query, context FROM memory_events
        WHERE event_type='recall_meta' AND context IS NOT NULL
        AND context LIKE '%vector_input%'
    """).fetchall()
    vector_inputs = {}
    for r in meta_rows:
        try:
            ctx = json.loads(r["context"])
            if "vector_input" in ctx:
                vector_inputs[r["query"]] = ctx["vector_input"]
        except (json.JSONDecodeError, TypeError):
            continue

    for q in queries:
        q["vector_input"] = vector_inputs.get(q["query"], "")

    return queries


def dedup_queries(
    queries: list[dict],
    threshold: float = 0.75,
) -> list[dict]:
    """Deduplicate near-identical keyword queries using containment similarity.

    Queries like "corrections calibration patterns gotchas working preferences memory"
    and "corrections calibration patterns gotchas working preferences claudie" are
    superset variants of a shared core. They retrieve nearly identical candidate sets,
    so judging both is redundant.

    Uses asymmetric containment: if the smaller token set is >= threshold contained
    in the larger, they cluster together. This catches superset variants that Jaccard
    misses (e.g., 6 shared tokens + 3 unique = Jaccard 0.67, but containment 1.0).

    Keeps the highest-count representative from each cluster.
    """
    if not queries:
        return queries

    # Precompute token sets
    tokenized = [(q, set(q["query"].lower().split())) for q in queries]

    # Greedy clustering: assign each query to the first cluster it matches
    clusters: list[list[dict]] = []
    cluster_tokens: list[set[str]] = []

    for q, tokens in tokenized:
        if not tokens:
            clusters.append([q])
            cluster_tokens.append(tokens)
            continue

        matched = False
        for ci, ct in enumerate(cluster_tokens):
            if not ct:
                continue
            smaller = min(len(tokens), len(ct))
            if smaller == 0:
                continue
            intersection = len(tokens & ct)
            # Containment: fraction of the smaller set that overlaps
            containment = intersection / smaller
            if containment >= threshold:
                clusters[ci].append(q)
                matched = True
                break

        if not matched:
            clusters.append([q])
            cluster_tokens.append(tokens)

    # Pick representative: highest retrieval count per cluster
    representatives = []
    removed = 0
    for cluster in clusters:
        cluster.sort(key=lambda q: q["count"], reverse=True)
        representatives.append(cluster[0])
        removed += len(cluster) - 1

    representatives.sort(key=lambda q: q["count"], reverse=True)

    if removed > 0:
        print(f"Query dedup: {len(queries)} -> {len(representatives)} "
              f"({removed} near-duplicates removed, containment >= {threshold})")

    return representatives


def main():
    parser = argparse.ArgumentParser(description="Build unbiased ground truth for memory eval")
    parser.add_argument("--budget", type=int, default=100,
                        help="Top-k per retrieval path (default 100, yielding up to 200 candidates)")
    parser.add_argument("--model", type=str, default="claude-opus-4-6",
                        help="Model for LLM judge (default: claude-opus-4-6)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show candidate counts without calling LLM")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file, skip already-judged queries")
    parser.add_argument("--max-queries", type=int, default=0,
                        help="Limit number of queries to process (0 = all)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH),
                        help=f"Output path (default: {OUTPUT_PATH})")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Max candidates per judge call (reduces context effects, default: 50)")
    parser.add_argument("--query-index", type=int, default=0,
                        help="Start from this query index (0-based, sorted by retrieval count desc)")
    parser.add_argument("--export-candidates", type=str, default="",
                        help="Export candidate sets to JSON (no LLM calls). For offline judging.")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Skip query deduplication (judge all unique query strings)")
    parser.add_argument("--dedup-threshold", type=float, default=0.75,
                        help="Containment similarity threshold for query dedup (default: 0.75)")
    parser.add_argument("--queries", type=str, default="",
                        help="Path to JSON file with query allowlist. Accepts list of "
                             "{query, vector_input} dicts or list of strings.")
    args = parser.parse_args()

    db = get_db()

    if args.queries:
        # Load query allowlist from file
        with open(args.queries) as f:
            query_data = json.load(f)
        if isinstance(query_data, list):
            if query_data and isinstance(query_data[0], str):
                queries = [{"query": q, "count": 0, "vector_input": ""} for q in query_data]
            else:
                queries = query_data
        elif isinstance(query_data, dict):
            # Export-candidates format: {query: {vector_input, candidates}}
            queries = [{"query": q, "count": 0, "vector_input": d.get("vector_input", "")}
                       for q, d in query_data.items()]
        print(f"Loaded {len(queries)} queries from {args.queries}")
    else:
        queries = extract_queries(db)
        print(f"Found {len(queries)} unique queries")

    if not args.no_dedup and not args.queries:
        queries = dedup_queries(queries, threshold=args.dedup_threshold)

    # --- Export candidates mode ---
    if args.export_candidates:
        export = {}
        total_candidates = 0
        for i, q in enumerate(queries):
            query = q["query"]
            emb = embed_text(query)
            candidates = retrieve_candidates(db, query, emb, top_k=args.budget)
            total_candidates += len(candidates)
            export[query] = {
                "vector_input": q.get("vector_input", ""),
                "candidates": candidates,
            }
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i+1}/{len(queries)}] {query[:60]:60s} -> {len(candidates)} candidates")
        export_path = Path(args.export_candidates)
        with open(export_path, "w") as f:
            json.dump(export, f)
        avg = total_candidates / len(queries) if queries else 0
        print(f"\nExported {len(export)} queries to {export_path}")
        print(f"Avg candidates per query: {avg:.0f}")
        print(f"File size: {export_path.stat().st_size / 1024 / 1024:.1f} MB")
        return

    # Load existing results if resuming
    output_path = Path(args.output)
    ground_truth = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            ground_truth = json.load(f)
        print(f"Resuming: {len(ground_truth)} queries already judged")

    # Filter to queries not yet judged, apply index offset
    pending = [q for q in queries if q["query"] not in ground_truth]
    if args.query_index > 0:
        pending = pending[args.query_index:]
    if args.max_queries > 0:
        pending = pending[:args.max_queries]
    print(f"Queries to process: {len(pending)}")

    if args.dry_run:
        # Just show candidate counts
        total_candidates = 0
        for i, q in enumerate(pending[:20]):
            emb = embed_text(q["query"])
            candidates = retrieve_candidates(db, q["query"], emb, top_k=args.budget)
            total_candidates += len(candidates)
            print(f"  [{i+1}] {q['query'][:60]:60s} -> {len(candidates)} candidates")
        avg = total_candidates / min(len(pending), 20) if pending else 0
        print(f"\nAvg candidates per query: {avg:.0f}")
        est_tokens = avg * 200 * len(pending)  # rough estimate: 200 tokens/memory
        print(f"Estimated input tokens: {est_tokens:,.0f}")
        return

    total_relevant = 0
    total_candidates = 0
    start_time = time.time()

    for i, q in enumerate(pending):
        query = q["query"]
        elapsed = time.time() - start_time
        rate = (i / elapsed * 60) if elapsed > 0 and i > 0 else 0

        print(f"  [{i+1}/{len(pending)}] ({rate:.1f}/min) {query[:70]}")

        # Get embedding
        emb = embed_text(query)

        # Deep retrieval
        candidates = retrieve_candidates(db, query, emb, top_k=args.budget)
        total_candidates += len(candidates)

        if not candidates:
            ground_truth[query] = {}
            continue

        # LLM judge
        scores = judge_relevance(query, candidates, args.model, q.get("vector_input", ""), batch_size=args.batch_size)
        ground_truth[query] = scores
        total_relevant += len(scores)

        print(f"    {len(candidates)} candidates -> {len(scores)} relevant")

        # Save periodically (every 10 queries)
        if (i + 1) % 10 == 0:
            with open(output_path, "w") as f:
                json.dump(ground_truth, f, indent=2)
            print(f"    [saved checkpoint: {len(ground_truth)} queries]")

    # Final save
    with open(output_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nDone. {len(ground_truth)} queries judged in {elapsed:.0f}s")
    print(f"Total candidates evaluated: {total_candidates}")
    print(f"Total relevant memories found: {total_relevant}")
    print(f"Avg relevant per query: {total_relevant / len(ground_truth):.1f}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
