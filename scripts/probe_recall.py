# /// script
# requires-python = ">=3.11"
# dependencies = ["sqlite-vec>=0.1.6", "openai>=2.0.0", "tiktoken>=0.7.0"]
# ///
"""
Retrieval probe -- generates recall queries targeting under-covered memories,
runs them through the real recall pipeline, and has a subagent rate the results.

Produces the same retrieval + feedback events that tune_memory.py consumes.
Designed to run standalone (outside Claude Code), like sleep scripts.

Usage:
    uv run scripts/probe_recall.py                # defaults: 30 queries, opus
    uv run scripts/probe_recall.py --queries 50   # more queries
    uv run scripts/probe_recall.py --model sonnet # cheaper/faster
    uv run scripts/probe_recall.py --budget 6000  # see deeper into ranked list
    uv run scripts/probe_recall.py --dry-run      # show plan, don't execute
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memory.constants import DATA_DIR
from memory import DB_PATH
from memory.tools import impl_recall, impl_recall_feedback

LOG_DIR = DATA_DIR / "sleep_logs"
PROGRESS_LOG = DATA_DIR / "sleep_progress.log"

_SUBPROCESS_CWD = DATA_DIR / "subprocess_workspace"
EMPTY_MCP_CONFIG = json.dumps({"mcpServers": {}})

_log_file = None


def _init_log():
    global _log_file
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_file = LOG_DIR / f"probe_{ts}.log"
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_LOG.write_text("", encoding="utf-8")
    return _log_file


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}\n"
    print(msg.encode("ascii", "replace").decode("ascii"), flush=True)
    if _log_file:
        with open(_log_file, "a", encoding="utf-8") as f:
            f.write(line)
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(line)


def log_detail(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    if _log_file:
        with open(_log_file, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    import sqlite_vec
    db = sqlite3.connect(str(DB_PATH))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.row_factory = sqlite3.Row
    return db


# ---------------------------------------------------------------------------
# Coverage analysis
# ---------------------------------------------------------------------------

def _compute_memory_quality(db: sqlite3.Connection) -> dict[str, float]:
    """Compute per-memory coverage quality from non-probe feedback.

    Returns dict of {memory_id: quality} where:
      0.0 = never retrieved, or only by probe
      0.1 = retrieved but no real feedback
      avg(utility) = from non-probe feedback events
    """
    # Which memories have any retrieval
    retrieved = set()
    for r in db.execute(
        'SELECT DISTINCT memory_id FROM memory_events WHERE event_type = "retrieved"'
    ):
        retrieved.add(r["memory_id"])

    # Collect non-probe feedback utilities per memory
    real_feedback = defaultdict(list)
    for r in db.execute(
        "SELECT memory_id, json_extract(context, '$.utility') as utility, "
        "json_extract(context, '$.reason') as reason "
        "FROM memory_events WHERE event_type = 'feedback'"
    ):
        reason = r["reason"] or ""
        if reason.startswith("[probe"):
            continue
        utility = r["utility"]
        if utility is not None:
            real_feedback[r["memory_id"]].append(float(utility))

    quality = {}
    for mid in retrieved:
        if mid in real_feedback:
            quality[mid] = sum(real_feedback[mid]) / len(real_feedback[mid])
        else:
            quality[mid] = 0.1  # retrieved, never rated by a real session
    return quality


UNDERSERVED_THRESHOLD = 0.5


def find_coverage_gaps(db: sqlite3.Connection, min_mems: int, coverage_threshold: float):
    """Find themes with low coverage quality, return sorted by gap size."""
    theme_memories = defaultdict(set)
    memory_themes = defaultdict(set)
    memory_content = {}

    for row in db.execute(
        "SELECT id, themes, content, summary, category FROM memories WHERE status = 'active'"
    ):
        memory_content[row["id"]] = {
            "summary": row["summary"] or "",
            "category": row["category"] or "",
            "themes": row["themes"] or "[]",
        }
        if row["themes"]:
            try:
                for t in json.loads(row["themes"]):
                    theme_memories[t].add(row["id"])
                    memory_themes[row["id"]].add(t)
            except (json.JSONDecodeError, TypeError):
                pass

    # Per-memory quality from real (non-probe) feedback
    mem_quality = _compute_memory_quality(db)

    # Score each theme by mean quality; collect underserved members
    all_themes = []
    for theme, mids in theme_memories.items():
        if len(mids) < min_mems:
            continue
        qualities = [mem_quality.get(mid, 0.0) for mid in mids]
        coverage = sum(qualities) / len(qualities)
        underserved = [mid for mid in mids
                       if mem_quality.get(mid, 0.0) < UNDERSERVED_THRESHOLD]
        all_themes.append({
            "theme": theme,
            "total": len(mids),
            "coverage": coverage,
            "underserved_ids": underserved,
        })

    # Primary: themes below threshold
    gaps = [t for t in all_themes if t["coverage"] < coverage_threshold]

    # Fallback: if nothing below threshold, take bottom 20 by coverage
    if not gaps:
        all_themes.sort(key=lambda t: t["coverage"])
        gaps = [t for t in all_themes[:20] if t["underserved_ids"]]

    gaps.sort(key=lambda g: len(g["underserved_ids"]), reverse=True)
    return gaps, memory_content, memory_themes, mem_quality


def select_targets(gaps, memory_content, mem_quality: dict, num_queries: int,
                   batch_size: int, content_chars: int):
    """Select target memories from coverage gaps, prioritizing least-probed.

    Returns list of target groups, each with batch_size memories to craft queries for.
    """
    # Deduplicate: a memory might appear in multiple theme gaps
    all_underserved = {}
    for gap in gaps:
        for mid in gap["underserved_ids"]:
            if mid in memory_content:
                if mid not in all_underserved:
                    all_underserved[mid] = {
                        "id": mid,
                        "themes": list(set(
                            t for g in gaps
                            if mid in g["underserved_ids"]
                            for t in [g["theme"]]
                        )),
                        **memory_content[mid],
                    }

    if not all_underserved:
        return []

    # Per-memory probe vs real feedback counts
    db = get_db()
    probe_hits = Counter()
    real_hits = Counter()
    for r in db.execute(
        "SELECT memory_id, json_extract(context, '$.reason') as reason "
        "FROM memory_events WHERE event_type = 'feedback'"
    ):
        mid = r["memory_id"]
        if mid not in all_underserved:
            continue
        reason = r["reason"] or ""
        if reason.startswith("[probe"):
            probe_hits[mid] += 1
        else:
            real_hits[mid] += 1

    # Filter out memories where probe feedback already exceeds 2x real feedback
    MAX_PROBE_RATIO = 2.0
    eligible = {}
    skipped = 0
    for mid, entry in all_underserved.items():
        p = probe_hits.get(mid, 0)
        r = real_hits.get(mid, 0)
        if p > MAX_PROBE_RATIO * max(r, 1):
            skipped += 1
            continue
        eligible[mid] = entry

    if skipped:
        log(f"  Skipped {skipped} memories (probe/real ratio > {MAX_PROBE_RATIO})")

    if not eligible:
        log("  All underserved memories already saturated with probe feedback.")
        db.close()
        return []

    # Load full content for eligible memories (up to content_chars)
    for mid in eligible:
        row = db.execute(
            "SELECT content FROM memories WHERE id = ?", (mid,)
        ).fetchone()
        if row:
            eligible[mid]["content"] = row["content"][:content_chars]
    db.close()

    # Sort by (probe/real ratio, has any signal) -- truly dark memories first,
    # then by ascending ratio within memories that have some signal
    candidates = list(eligible.values())
    random.shuffle(candidates)  # randomize before stable sort for tie-breaking
    def _sort_key(c):
        mid = c["id"]
        p = probe_hits.get(mid, 0)
        r = real_hits.get(mid, 0)
        has_signal = 0 if (p == 0 and r == 0) else 1
        ratio = p / max(r, 1)
        return (has_signal, ratio)
    candidates.sort(key=_sort_key)
    selected = candidates

    # Group into batches for query crafting
    groups = []
    num_groups = min(len(selected) // batch_size + 1, int(num_queries * 0.7))

    for i in range(num_groups):
        start = i * batch_size
        end = start + batch_size
        group = selected[start:end]
        if group:
            groups.append(group)

    return groups


# ---------------------------------------------------------------------------
# LLM subprocess
# ---------------------------------------------------------------------------

def _ensure_subprocess_workspace():
    _SUBPROCESS_CWD.mkdir(parents=True, exist_ok=True)
    git_dir = _SUBPROCESS_CWD / ".git"
    if not git_dir.exists():
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n")


def call_llm(prompt: str, model: str, system: str = "", timeout: int = 240) -> str:
    """Call claude -p with the specified model for JSON-only output."""
    _ensure_subprocess_workspace()
    env = dict(os.environ)
    env.pop("CLAUDECODE", None)

    kwargs = {}
    if sys.platform == "win32":
        CREATE_NO_WINDOW = 0x08000000
        kwargs["creationflags"] = CREATE_NO_WINDOW

    claude_bin = shutil.which("claude") or "claude"
    cmd = [
        claude_bin, "-p",
        "--model", model,
        "--no-session-persistence",
        "--strict-mcp-config",
        "--mcp-config", EMPTY_MCP_CONFIG,
        "--system-prompt", system or (
            "You are a JSON-only output tool. Respond with the requested "
            "JSON. No markdown fences, no commentary, no tool calls."
        ),
        "--tools", "",
        "--disable-slash-commands",
        "--setting-sources", "local",
    ]

    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
            cwd=str(_SUBPROCESS_CWD),
            env=env,
            **kwargs,
        )
        if result.returncode != 0:
            log(f"  WARNING: {model} returned {result.returncode}: {result.stderr[:200]}")
            return ""
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        log(f"  WARNING: {model} timed out ({timeout}s)")
        return ""


def parse_json_response(raw: str) -> any:
    """Parse JSON from LLM output, stripping markdown fences if present."""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Query crafting
# ---------------------------------------------------------------------------

CRAFT_SYSTEM = (
    "You generate realistic memory recall queries for a personal memory system. "
    "You are a JSON-only output tool. Return ONLY a JSON array, no markdown fences."
)

CRAFT_TEMPLATE = """Given these memories from an AI's personal memory system, craft {n_queries} realistic recall queries that someone would use to find them.

IMPORTANT query style rules:
- "query" is keyword-heavy (5-15 tokens), NOT natural language. Specific nouns, proper names, technical terms that appear literally in the memory content/summary/themes.
- Mix lengths: some short (2-3 tokens), some medium (5-8), some long (10-15)
- Each query should target 1-3 of the provided memories (not all of them)
- Think: "what FTS5 tokens would match THIS specific memory?"

CONTEXT field (optional, for vector search):
- "context" is a longer natural-language description (1-2 sentences) of what you're looking for.
- Use context when the keywords are ambiguous or shared across domains (e.g. "training" could mean ML training or sports training), OR when a semantic description would reach memories the keywords might miss.
- Leave context as "" when the keywords are already specific enough (proper nouns, unique technical terms).
- Roughly half the queries should have context, half should not.

Memories:
{memories}

Return a JSON array of objects:
[{{"query": "keyword string here", "context": "optional longer description or empty string", "target_ids": ["id1", "id2"], "intent": "brief description of what the searcher wants"}}]"""


def craft_queries(group: list[dict], n_queries: int, model: str, timeout: int) -> list[dict]:
    """Have LLM craft recall queries targeting a group of memories."""
    mem_texts = []
    for m in group:
        mem_texts.append(
            f"ID: {m['id'][:8]}\n"
            f"Category: {m['category']}\n"
            f"Themes: {m['themes']}\n"
            f"Summary: {m['summary']}\n"
            f"Content: {m['content']}"
        )

    prompt = CRAFT_TEMPLATE.format(
        n_queries=n_queries,
        memories="\n---\n".join(mem_texts),
    )

    raw = call_llm(prompt, model, CRAFT_SYSTEM, timeout)
    parsed = parse_json_response(raw)
    if isinstance(parsed, list):
        return parsed
    log(f"  WARNING: failed to parse craft response: {(raw or '')[:200]}")
    return []


# ---------------------------------------------------------------------------
# Result rating
# ---------------------------------------------------------------------------

RATE_SYSTEM = (
    "You rate memory recall results for utility. "
    "You are a JSON-only output tool. Return ONLY a JSON object, no markdown fences."
)

RATE_TEMPLATE = """A memory system ran this recall query:
Query: "{query}"
Intent: {intent}
Target memories (what the query was designed to find): {target_ids}

Here are the results returned (in rank order):

{results}

Rate each returned memory's utility for this query's intent.
Scale: 0.0 = completely irrelevant, 0.3 = marginal, 0.6 = useful, 0.9 = critical hit.

Also provide:
- cutoff_rank: position (1-indexed) of the last useful result. 0 if nothing in top positions was useful.
- For memories that are target_ids, rate based on how well they match the query intent (should be high if they actually appeared).
- For non-target memories, rate honestly -- they may be relevant even though they weren't the intended targets.

Return JSON:
{{"feedback": {{"mem_id": utility_score, ...}}, "cutoff_rank": N, "reason": "brief explanation"}}"""


def rate_results(query: str, intent: str, target_ids: list[str],
                 recall_output: str, model: str, timeout: int) -> dict:
    """Have LLM rate the recall results."""
    prompt = RATE_TEMPLATE.format(
        query=query,
        intent=intent,
        target_ids=json.dumps(target_ids),
        results=recall_output,
    )

    raw = call_llm(prompt, model, RATE_SYSTEM, timeout)
    parsed = parse_json_response(raw)
    if isinstance(parsed, dict) and "feedback" in parsed:
        return parsed
    log(f"  WARNING: failed to parse rate response: {(raw or '')[:200]}")
    return {}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_probe(
    num_queries: int = 30,
    dry_run: bool = False,
    model: str = "opus",
    min_mems: int = 2,
    coverage_threshold: float = 0.5,
    batch_size: int = 5,
    content_chars: int = 600,
    recall_budget: int = 4000,
    recall_limit: int = 15,
    timeout: int = 240,
    extra_queries_path: str = None,
    workers: int = 1,
):
    _init_log()
    log(f"=== Retrieval Probe ===")
    log(f"  queries={num_queries}, model={model}, budget={recall_budget}, "
        f"limit={recall_limit}, batch={batch_size}")
    log(f"  coverage_threshold={coverage_threshold}, content_chars={content_chars}, "
        f"timeout={timeout}s")

    db = get_db()

    # 1. Find coverage gaps
    log("\n--- Finding coverage gaps ---")
    gaps, memory_content, memory_themes, mem_quality = find_coverage_gaps(db, min_mems, coverage_threshold)
    total_underserved = sum(len(g["underserved_ids"]) for g in gaps)
    below_threshold = sum(1 for g in gaps if g["coverage"] < coverage_threshold)
    if below_threshold == len(gaps):
        log(f"  {len(gaps)} themes below {coverage_threshold:.0%} quality, "
            f"{total_underserved} underserved memories")
    else:
        log(f"  {below_threshold} themes below {coverage_threshold:.0%} -- "
            f"fallback: targeting {len(gaps)} lowest-quality themes, "
            f"{total_underserved} underserved memories")

    if not gaps:
        log("  No gaps found. Nothing to probe.")
        return

    # Show top gaps
    for g in gaps[:20]:
        log(f"  {g['theme']:<35} {g['total']:>3} mems, {g['coverage']:>5.1%} quality, "
            f"{len(g['underserved_ids'])} underserved")

    db.close()

    # 2. Select target groups
    log("\n--- Selecting targets ---")
    groups = select_targets(gaps, memory_content, mem_quality, num_queries, batch_size, content_chars)
    log(f"  {len(groups)} target groups ({sum(len(g) for g in groups)} unique memories)")

    if dry_run:
        log("\n--- DRY RUN: would craft queries for these groups ---")
        for i, group in enumerate(groups):
            log(f"\n  Group {i+1}:")
            for m in group:
                log(f"    {m['id'][:8]} [{m['category']}] {m['summary'][:80]}")
        return

    # 3. Craft queries + run + rate (parallel across groups)
    log(f"\n--- Crafting queries and probing (workers={workers}) ---")
    total_queries = 0
    total_feedback = 0

    # Assign query budget per group
    queries_per_group = []
    remaining_budget = num_queries
    for group in groups:
        n = max(1, min(3, remaining_budget))
        queries_per_group.append(n)
        remaining_budget -= n
        if remaining_budget <= 0:
            break
    groups = groups[:len(queries_per_group)]

    def process_group(args):
        idx, group, n_per_group = args
        group_queries = 0
        group_feedback = 0

        log(f"\n  Group {idx+1}/{len(groups)} ({len(group)} target memories)")
        for m in group:
            log_detail(f"    Target: {m['id'][:8]} -- {m['summary'][:80]}")

        queries = craft_queries(group, n_per_group, model, timeout)
        if not queries:
            log(f"    Group {idx+1}: WARNING: no queries crafted, skipping")
            return 0, 0

        for q in queries:
            query_text = q.get("query", "")
            context_text = q.get("context", "")
            intent = q.get("intent", "")
            target_ids = q.get("target_ids", [])
            if not query_text:
                continue

            log(f"\n    [{idx+1}] Query: {query_text}")
            if context_text:
                log(f"    [{idx+1}] Context: {context_text}")
            log(f"    [{idx+1}] Intent: {intent}")
            log(f"    [{idx+1}] Targets: {target_ids}")

            recall_output = impl_recall(
                query=query_text,
                context=context_text,
                budget=recall_budget,
                limit=recall_limit,
                internal=False,
            )
            group_queries += 1

            log_detail(f"    [{idx+1}] Recall output:\n{recall_output}")

            rating = rate_results(query_text, intent, target_ids, recall_output,
                                  model, timeout)
            if not rating or "feedback" not in rating:
                log(f"    [{idx+1}] WARNING: no ratings returned, skipping feedback")
                continue

            feedback = rating["feedback"]
            cutoff = rating.get("cutoff_rank", -1)
            reason = rating.get("reason", "")

            log(f"    [{idx+1}] Cutoff: {cutoff}, Ratings: {len(feedback)}")
            log_detail(f"    [{idx+1}] Reason: {reason}")
            log_detail(f"    [{idx+1}] Feedback: {json.dumps(feedback)}")

            fb_result = impl_recall_feedback(
                feedback=json.dumps(feedback),
                query=query_text,
                reason=f"[probe] {reason}",
                cutoff_rank=cutoff,
            )
            group_feedback += len(feedback)
            log(f"    [{idx+1}] {fb_result}")

        return group_queries, group_feedback

    work_items = [(i, g, n) for i, (g, n) in enumerate(zip(groups, queries_per_group))]

    if workers > 1 and len(work_items) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for q_count, fb_count in pool.map(process_group, work_items):
                total_queries += q_count
                total_feedback += fb_count
    else:
        for item in work_items:
            q_count, fb_count = process_group(item)
            total_queries += q_count
            total_feedback += fb_count

    # 4. Run extra queries (hand-picked, skip craft step)
    if extra_queries_path:
        extra_path = Path(extra_queries_path)
        if extra_path.exists():
            try:
                extra_qs = json.loads(extra_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                log(f"\nWARNING: failed to load extra queries: {e}")
                extra_qs = []

            if extra_qs and not dry_run:
                log(f"\n--- Extra queries ({len(extra_qs)} from {extra_path.name}) ---")
                for q in extra_qs:
                    query_text = q.get("query", "")
                    context_text = q.get("context", "")
                    intent = q.get("intent", "")
                    target_ids = q.get("target_ids", [])
                    if not query_text:
                        continue

                    log(f"\n    Query: {query_text}")
                    if context_text:
                        log(f"    Context: {context_text}")
                    log(f"    Intent: {intent}")

                    recall_output = impl_recall(
                        query=query_text,
                        context=context_text,
                        budget=recall_budget,
                        limit=recall_limit,
                        internal=False,
                    )
                    total_queries += 1

                    log_detail(f"    Recall output:\n{recall_output}")

                    rating = rate_results(query_text, intent, target_ids, recall_output,
                                          model, timeout)
                    if not rating or "feedback" not in rating:
                        log("    WARNING: no ratings returned, skipping feedback")
                        continue

                    feedback = rating["feedback"]
                    cutoff = rating.get("cutoff_rank", -1)
                    reason = rating.get("reason", "")

                    log(f"    Cutoff: {cutoff}, Ratings: {len(feedback)}")
                    log_detail(f"    Reason: {reason}")
                    log_detail(f"    Feedback: {json.dumps(feedback)}")

                    fb_result = impl_recall_feedback(
                        feedback=json.dumps(feedback),
                        query=query_text,
                        reason=f"[probe-extra] {reason}",
                        cutoff_rank=cutoff,
                    )
                    total_feedback += len(feedback)
                    log(f"    {fb_result}")
            elif dry_run and extra_qs:
                log(f"\n--- DRY RUN: {len(extra_qs)} extra queries from {extra_path.name} ---")
                for q in extra_qs:
                    log(f"    {q.get('query', '?')}")
        else:
            log(f"\nWARNING: extra queries file not found: {extra_path}")

    log(f"\n=== Probe complete: {total_queries} queries, {total_feedback} feedback events ===")
    if _log_file:
        log(f"Log: {_log_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Retrieval probe -- generate tuning signal for under-covered memories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--queries", type=int, default=30,
                        help="Number of recall queries to generate")
    parser.add_argument("--model", type=str, default="opus",
                        choices=["opus", "sonnet", "haiku"],
                        help="Model for query crafting and result rating")
    parser.add_argument("--budget", type=int, default=4000,
                        help="Token budget per recall (higher = more results to rate)")
    parser.add_argument("--limit", type=int, default=15,
                        help="Max memories returned per recall")
    parser.add_argument("--coverage", type=float, default=0.5,
                        help="Themes below this retrieval coverage are targeted (0.0-1.0)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Target memories per group for query crafting")
    parser.add_argument("--content-chars", type=int, default=600,
                        help="Max chars of memory content shown to query crafter")
    parser.add_argument("--min-mems", type=int, default=2,
                        help="Min memories a theme needs to be considered a gap")
    parser.add_argument("--timeout", type=int, default=240,
                        help="Subprocess timeout in seconds")
    parser.add_argument("--extra-queries", type=str, default=None,
                        help="JSON file with extra queries to run (skips craft step)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker threads (default 1; >1 risks OOM from Bun subprocesses)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without executing")
    args = parser.parse_args()

    run_probe(
        num_queries=args.queries,
        dry_run=args.dry_run,
        model=args.model,
        min_mems=args.min_mems,
        coverage_threshold=args.coverage,
        batch_size=args.batch_size,
        content_chars=args.content_chars,
        recall_budget=args.budget,
        recall_limit=args.limit,
        timeout=args.timeout,
        extra_queries_path=args.extra_queries,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
