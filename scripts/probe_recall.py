# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "sqlite-vec>=0.1.6",
#     "openai>=2.0.0",
#     "tiktoken>=0.7.0",
#     "mcp[cli]>=1.2.0",
#     "numpy>=1.26",
#     "lightgbm>=4.0",
#     "fastembed>=0.4.0",
# ]
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
import re
import shutil
import subprocess
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# Memory ID prefix as printed by format_memory_full / format_memory_compact.
# Used to extract rank-ordered IDs from impl_recall's formatted output.
_RANK_ID_RE = re.compile(r"ID:\s*([0-9a-f]{8})")


def _parse_ranked_ids(recall_output: str) -> list[str]:
    """Extract memory ID prefixes from a recall output string in rank order."""
    return _RANK_ID_RE.findall(recall_output or "")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memory.constants import DATA_DIR
from memory import DB_PATH
from memory.events import _log_event
from memory.tools import impl_recall, impl_recall_feedback


def _emit_phase1_miss_event(target_id_full: str, query: str, mode: str,
                            returned_ids: list[str]) -> None:
    """Record a recall_miss event for a probe Phase-1 miss so the next
    sleep cycle's repair pass can diagnose theme/coverage fixes.

    Only called for natural/mild probe modes — hard-mode misses are
    designed-difficulty and shouldn't enter the repair queue.
    """
    db = get_db()
    try:
        _log_event(
            db, target_id_full, "recall_miss",
            query=query,
            context={
                "miss_type": "retrieval_failure",
                "source": "probe",
                "mode": mode,
                "returned_ids": returned_ids,
            },
        )
        db.commit()
    finally:
        db.close()


def _emit_probe_target_event(
    target_id_full: str,
    query: str,
    context_text: str,
    mode: str,
    target_rank,
    candidate_pool_size: int,
) -> None:
    """Record a probe_target event with the pinned target ID. Emitted for
    every probe query (hits and misses, all modes) so build_gt can lift
    pinned labels without parsing reason strings.

    target_rank is the 1-indexed rank of the pinned target in the candidate
    pool, or None for a Phase-1 miss.
    """
    db = get_db()
    try:
        _log_event(
            db, target_id_full, "probe_target",
            query=query,
            context={
                "mode": mode,
                "target_rank": target_rank,
                "candidate_pool_size": candidate_pool_size,
                "vector_input": (context_text or None) if context_text != query else None,
                "source": "probe",
            },
        )
        db.commit()
    finally:
        db.close()

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
                   batch_size: int, content_chars: int,
                   exclude: set | None = None):
    """Select target memories from coverage gaps, prioritizing least-pinned.

    Returns list of target groups, each with batch_size memories to craft queries for.

    `exclude` is the set of memory IDs already picked by another selector (e.g.
    the pathology selector in mixed-mode probing) — they're skipped at dedup.
    """
    excluded = exclude or set()
    # Deduplicate: a memory might appear in multiple theme gaps
    all_underserved = {}
    for gap in gaps:
        for mid in gap["underserved_ids"]:
            if mid in excluded:
                continue
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

    # Per-memory probe_target pin count. The probe_target event (Phase 1) is
    # emitted once per probe query, keyed to the pinned target memory — exactly
    # the labeled (q, M, ~1.0) signal we're trying to spread across the
    # population. Earlier versions of this filter counted all probe-feedback
    # rows (target + 14 rated candidates per probe), which conflated "this
    # memory was the target" with "this memory was rated as a non-target
    # candidate" and produced backwards saturation behavior (heavy real-recall
    # memories absorbing more probes than dark ones).
    db = get_db()
    pin_counts = Counter()
    for r in db.execute(
        "SELECT memory_id, COUNT(*) AS n FROM memory_events "
        "WHERE event_type = 'probe_target' GROUP BY memory_id"
    ):
        if r["memory_id"] in all_underserved:
            pin_counts[r["memory_id"]] = r["n"]

    # Cap pins per memory: 4 lines up with one full mixed-mode bundle
    # (1 nat + 2 mild + 1 hard). Past 4, marginal label value drops fast — the
    # next probe shape mostly repeats angles the bundle already covered.
    MAX_PINS_PER_MEMORY = 4
    eligible = {}
    skipped = 0
    for mid, entry in all_underserved.items():
        if pin_counts.get(mid, 0) >= MAX_PINS_PER_MEMORY:
            skipped += 1
            continue
        eligible[mid] = entry

    if skipped:
        log(f"  Skipped {skipped} memories (already pinned >= {MAX_PINS_PER_MEMORY} times)")

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

    # Selection: weighted sampling without replacement (Efraimidis-Spirakis
    # A-Res). Bias toward dark memories (zero pins) but keep stochasticity so
    # the same candidates don't get picked run after run.
    #
    # Weight rationale (now pin-count-based, not ratio-based):
    #   - dark memory (0 pins): 1.0 — never had a labeled (q, M, ~1.0) pair
    #     from a crafted query, maximally useful to pin
    #   - partially pinned (1..MAX-1 pins): 1 / (1 + pins) — soft down-weight
    #     so already-pinned memories don't dominate but still surface in the
    #     tail when the underserved-and-unpinned set runs out
    #
    # The hard ceiling (MAX_PINS_PER_MEMORY filter above) drops memories at
    # the cap; this layer adds smooth bias under the cap.
    candidates = list(eligible.values())

    def _selection_weight(mid: str) -> float:
        return 1.0 / (1.0 + pin_counts.get(mid, 0))

    # Efraimidis-Spirakis: assign each candidate key = U^(1/w). Top-k by key
    # is equivalent to k-without-replacement weighted sampling. Stable for
    # any weight distribution; weights of zero get key 0 (never picked).
    keyed = []
    for c in candidates:
        w = max(_selection_weight(c["id"]), 1e-9)
        u = random.random()
        # u^(1/w) is monotonic in u and weight-biased; high w → key near 1
        keyed.append((u ** (1.0 / w), c))
    keyed.sort(key=lambda x: x[0], reverse=True)
    selected = [c for _, c in keyed]

    # Group into batches for query crafting. With batch_size=1 (default),
    # each group is one memory and num_groups equals the budget.
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

# Three difficulty modes. Mixed cycles 1/3 each across queries so the reranker
# gets supervised exposure to easy, medium, and hard regimes.
#
#   natural — realistic FTS-friendly recall: lifts memory tokens, keyword-heavy.
#             The "happy path" the ranker already wins. Validates that FTS
#             keeps working.
#   mild    — paraphrases obvious FTS tokens, keeps proper nouns and unique
#             identifiers. Forces vec/theme to share load with FTS.
#   hard    — strips distinctive vocabulary; describes role/topic instead of
#             names. Forces the ranker to bridge via vec embedding, theme
#             channel, or graph traversal — but the query must still uniquely
#             identify the target. The goal is a hard recoverable recall task.
#
# Each call targets a single PINNED memory (the highest-priority underserved
# memory in the group). Other memories in the group are passed as context so
# the LLM understands the topic neighborhood, but never as targets. This gives
# us paired data: same target × three modes, so cross-mode rank deltas are
# directly comparable.

# Common style reference shown verbatim in all three templates so the LLM has
# a concrete gradient to anchor against. Without this, mild often degenerates
# into natural and hard sometimes degenerates into mild.
#
# The example uses generic software-engineering vocabulary; do not introduce
# domain-specific or proprietary names here — this script is part of a public
# repo and the prompts ship to LLM subprocesses verbatim.
STYLE_REFERENCE = """STYLE REFERENCE — same memory rendered in all three modes:

  Memory: "After upgrading the http_client library to v5.0, async cancellation
   stopped propagating through nested task groups; workaround is wrapping the
   inner group in shield() until the upstream patch lands."

  natural: "http_client v5.0 async cancellation nested task groups shield workaround"
  mild:    "http_client v5.0 broke cancel propagation in inner task groups, wrap with shield"
  hard:    "the library upgrade that broke cancellation across nested concurrent operations and the temporary lock workaround"

Notice the gradient: natural lifts the memory's tokens directly; mild keeps
the unique identifier (http_client v5.0) but paraphrases verbs and connectors;
hard describes the bug by its effect ("upgrade that broke cancellation") and
the workaround by its mechanism ("lock"), without using the library's name or
the specific API (shield). Crucially, the hard query still carries enough
information that a reader could confirm which memory it points to — it is
hard, not unanswerable.
"""

CRAFT_TEMPLATE_NATURAL = """Generate {n_queries} recall query/queries in NATURAL mode targeting the pinned memory.

TARGET MEMORY (write a query that retrieves THIS specific memory):
{target}

CONTEXT MEMORIES (related, may share themes — DO NOT target these, they are background):
{context_memories}

{style_reference}

NATURAL mode rules:
- Keyword-heavy: lift specific terms directly from the target memory's content/summary/themes.
- Use proper nouns, technical IDs, and distinctive phrases as written.
- Length 5-12 tokens.
- This is the easy regime — FTS should win.

CONTEXT field (always required):
- 1-2 sentences of natural-language description of what you're looking for. Drives vector search.

Return a JSON array of objects (target_ids must be ["{target_id}"]):
[{{"query": "keyword string", "context": "natural-language description", "target_ids": ["{target_id}"], "intent": "brief description of what the searcher wants"}}]"""

CRAFT_TEMPLATE_MILD = """Generate {n_queries} recall query/queries in MILD ADVERSARIAL mode targeting the pinned memory.

TARGET MEMORY (write a query that retrieves THIS specific memory):
{target}

CONTEXT MEMORIES (related, may share themes — DO NOT target these, they are background):
{context_memories}

{style_reference}

MILD ADVERSARIAL mode rules:
- Rewrite AT LEAST HALF of the target memory's content tokens as synonyms or paraphrases (e.g. swap common verbs and connector phrases for synonyms).
- KEEP proper nouns, unique technical IDs, and distinctive identifiers — those are often the only way to disambiguate the topic. Strip common verbs/nouns instead.
- Length 5-12 tokens.
- Goal: SOME lexical anchors remain, but most matching keywords are paraphrased. Forces vec/theme to share load with FTS.

CONTEXT field (always required):
- 1-2 sentences of natural-language description. Drives vector search.

Return a JSON array of objects (target_ids must be ["{target_id}"]):
[{{"query": "paraphrased keyword string", "context": "natural-language description", "target_ids": ["{target_id}"], "intent": "brief description of what the searcher wants"}}]"""

CRAFT_TEMPLATE_HARD = """Generate {n_queries} recall query/queries in HARD ADVERSARIAL mode targeting the pinned memory.

TARGET MEMORY (write a query that retrieves THIS specific memory):
{target}

CONTEXT MEMORIES (related, may share themes — DO NOT target these, they are background):
{context_memories}

{style_reference}

HARD ADVERSARIAL mode rules:
- Use AT MOST ONE distinctive identifier from the target memory (proper noun, code name, or unique technical term). Describe everything else by role, relationship, mechanism, or topic.
  E.g. instead of a person's name, say "the engineer who proposed the rewrite"; instead of specific variable names, say "the macro factors used in the regression"; instead of a project codename, say "the migration tool".
- Length 5-12 tokens.

HARD CONSTRAINT — the query must still be answerable in theory:
- A reader given just the query, the target memory, and the context memories should be able to identify the target as the right answer.
- The query must carry enough specificity (effect, mechanism, role, relationship, distinguishing feature) to point uniquely to the target rather than describing a broad topic any memory could match.
- If you find yourself writing a query so vague it could match dozens of memories, you've stripped too much. Add back one distinguishing detail.
- The goal is a hard recall task — enough semantic signal for vec embedding, theme channel, or graph traversal to bridge — NOT an unrecoverable one.

CONTEXT field (always required):
- 1-2 sentences in your own words describing the topic. Without it, recall is genuinely unrecoverable.

Return a JSON array of objects (target_ids must be ["{target_id}"]):
[{{"query": "topical paraphrased query", "context": "natural-language description", "target_ids": ["{target_id}"], "intent": "brief description of what the searcher wants"}}]"""

CRAFT_TEMPLATES = {
    "natural": CRAFT_TEMPLATE_NATURAL,
    "mild": CRAFT_TEMPLATE_MILD,
    "hard": CRAFT_TEMPLATE_HARD,
}

# Phase 3d (2026-05-09): Bundled-orthogonal craft. A single LLM call produces
# 4 mode-tagged queries per pinned target — 1 natural + 2 mild + 1 hard —
# attacking the same target from genuinely different angles. Replaces the
# 3-separate-calls-per-target path for --mode mixed only; single-mode
# invocations (--mode natural/mild/hard) still use the per-mode path so
# ablations and debugging remain straightforward.
#
# Orthogonality is asymmetric on purpose:
#   - The two MILD queries must attack the target from different angles.
#     Mild is the highest-EV training signal (in-distribution, weight 1.2,
#     paraphrased verbs but distinctive IDs preserved), so this is where
#     LLM laziness hurts most. Lean hard in the prompt.
#   - Cross-mode orthogonality (mild vs natural, mild vs hard) is preferred
#     but secondary — hard is already orthogonal by construction.
#   - Natural orthogonality matters least; FTS-keyword form is what we want.
#
# The angle field per query is logged (not persisted to events) and exists
# so two near-identical mild angles are spottable in the probe log.

STYLE_REFERENCE_BUNDLED = """STYLE REFERENCE -- same memory rendered as 4 orthogonal queries:

  Memory: "After upgrading the http_client library to v5.0, async cancellation
   stopped propagating through nested task groups; workaround is wrapping the
   inner group in shield() until the upstream patch lands."

  natural (FTS-friendly, keyword-heavy):
    query: "http_client v5.0 async cancellation nested task groups shield workaround"
    angle: "literal keywords from the memory"

  mild #1 (mechanism angle -- paraphrase verbs/connectors, keep distinctive IDs):
    query: "http_client v5.0 broke cancel propagation in inner task groups, wrap with shield"
    angle: "describes the cancellation-propagation failure mechanism"

  mild #2 (symptom / debugging angle -- orthogonal to mild #1):
    query: "v5.0 upgrade why do nested task groups silently skip cleanup"
    angle: "describes the user-visible symptom, not the mechanism"

  hard (topical paraphrase, no proper nouns, role-based):
    query: "the library upgrade that broke cancellation across nested concurrent operations"
    angle: "no library or API names; defines target by effect"

The two mild queries are genuinely DIFFERENT angles -- one targets the
mechanism, the other the user-visible symptom. They are NOT paraphrases of
the same sentence. Orthogonality between the two mild queries is the most
important property of the bundle.
"""

CRAFT_TEMPLATE_BUNDLED = """Generate exactly 4 recall queries targeting the pinned memory, one per mode slot in this exact order: natural, mild, mild, hard.

TARGET MEMORY (every query must retrieve THIS specific memory):
{target}

CONTEXT MEMORIES (related, may share themes -- DO NOT target these, they are background):
{context_memories}

{style_reference}

MODE RULES

natural (1 query):
- Keyword-heavy: lift specific terms directly from the target's content/summary/themes.
- Use proper nouns, technical IDs, and distinctive phrases as written.
- Length 5-12 tokens. FTS should win this regime.

mild (2 queries) -- CRITICAL: orthogonality requirement:
- The 2 mild queries MUST attack the target from genuinely different angles. Examples of valid orthogonal pairs:
    * mechanism vs symptom ("why does X break Y" vs "the side effect users see")
    * cause vs workaround ("the upgrade that introduced the bug" vs "the temporary fix until upstream lands")
    * who vs what ("the engineer's argument for the rewrite" vs "the proposal itself")
    * before vs after ("the legacy behavior" vs "what changed in the new version")
- Each query: rewrite AT LEAST HALF of the target's content tokens as synonyms or paraphrases. KEEP proper nouns and unique technical IDs -- those are often the only way to disambiguate.
- Length 5-12 tokens.
- DO NOT produce two paraphrases of the same sentence. If both mild queries reduce to the same angle, you have failed the orthogonality test. Pick a different second angle and try again before responding.

hard (1 query):
- Use AT MOST ONE distinctive identifier. Describe everything else by role, relationship, mechanism, or topic.
- Length 5-12 tokens.
- The query must still uniquely identify the target -- a reader given just the query, target, and context should be able to confirm the target is the right answer. Hard, not unanswerable.

CROSS-MODE ORTHOGONALITY (preferred, secondary to mild-vs-mild):
- Avoid making mild #1 a slightly-paraphrased natural, or hard a slightly-stripped mild #2.
- Hard is naturally orthogonal by construction (vocabulary stripped). Natural orthogonality matters least.

CONTEXT field (always required, per query):
- 1-2 sentences of natural-language description. Drives vector search.

ANGLE field (always required, per query):
- One short phrase (3-10 words) describing this query's angle. Used for QC.
- If your two mild queries have near-identical angles, that is a regression to flag.

Return a JSON array of EXACTLY 4 objects in this order -- natural, mild, mild, hard:
[
  {{"query": "...", "context": "...", "intent": "...", "mode": "natural", "angle": "..."}},
  {{"query": "...", "context": "...", "intent": "...", "mode": "mild",    "angle": "..."}},
  {{"query": "...", "context": "...", "intent": "...", "mode": "mild",    "angle": "..."}},
  {{"query": "...", "context": "...", "intent": "...", "mode": "hard",    "angle": "..."}}
]

The pinned target id is "{target_id}" -- the script enforces target_ids on all 4 queries, you do not need to include it in the output."""


def _fetch_topical_neighbors(target: dict, max_neighbors: int = 3) -> list[dict]:
    """Find memories sharing at least one theme with the target. Used to give
    the craft LLM real topical context (vs the random group-siblings that
    happen to fall in the same select_targets batch).

    Returns up to max_neighbors active memories, excluding the target itself.
    """
    target_themes = target.get("themes") or []
    if isinstance(target_themes, str):
        try:
            target_themes = json.loads(target_themes)
        except (json.JSONDecodeError, TypeError):
            target_themes = []
    if not target_themes:
        return []

    db = get_db()
    try:
        # Score each candidate by overlap count, then return top-K. SQLite
        # doesn't have a clean way to score JSON-array overlap inline, so
        # we filter to candidates whose themes column mentions any of our
        # themes, then count overlaps in Python.
        like_clauses = " OR ".join(["themes LIKE ?"] * len(target_themes))
        like_params = [f'%"{t}"%' for t in target_themes]
        rows = db.execute(
            f"SELECT id, category, themes, summary, content FROM memories "
            f"WHERE status = 'active' AND id != ? AND ({like_clauses})",
            [target["id"]] + like_params,
        ).fetchall()

        target_theme_set = {t.lower() for t in target_themes if isinstance(t, str)}
        scored = []
        for r in rows:
            try:
                cand_themes = json.loads(r["themes"]) if r["themes"] else []
            except (json.JSONDecodeError, TypeError):
                continue
            cand_set = {t.lower() for t in cand_themes if isinstance(t, str)}
            overlap = len(target_theme_set & cand_set)
            if overlap > 0:
                scored.append((overlap, dict(r)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored[:max_neighbors]]
    finally:
        db.close()


def craft_queries(group: list[dict], n_queries: int, model: str, timeout: int,
                  craft_mode: str = "natural") -> list[dict]:
    """Generate recall queries targeting group[0] (the pinned anchor) in the
    given mode. Topical neighbors (memories that actually share themes with
    the target) are looked up and shown as context — the other group members
    are NOT used here because select_targets groups by probe/real ratio, not
    by topic, so group-siblings aren't reliably topical neighbors.
    """
    if not group:
        return []
    template = CRAFT_TEMPLATES.get(craft_mode, CRAFT_TEMPLATE_NATURAL)

    target = group[0]
    target_id_short = target["id"][:8]
    target_text = (
        f"ID: {target_id_short}\n"
        f"Category: {target['category']}\n"
        f"Themes: {target['themes']}\n"
        f"Summary: {target['summary']}\n"
        f"Content: {target['content']}"
    )

    neighbors = _fetch_topical_neighbors(target, max_neighbors=3)
    if neighbors:
        context_text = "\n---\n".join(
            f"ID: {m['id'][:8]} | [{m['category']}] themes={m['themes']} | "
            f"summary: {m['summary']}"
            for m in neighbors
        )
    else:
        context_text = "(no topical neighbors found — target is unique within its themes)"

    prompt = template.format(
        n_queries=n_queries,
        target=target_text,
        context_memories=context_text,
        style_reference=STYLE_REFERENCE,
        target_id=target_id_short,
    )

    raw = call_llm(prompt, model, CRAFT_SYSTEM, timeout)
    parsed = parse_json_response(raw)
    if isinstance(parsed, list):
        # Force pinning — trust the prompt but enforce structurally too.
        for q in parsed:
            q["target_ids"] = [target_id_short]
        return parsed
    log(f"  WARNING: failed to parse craft response ({craft_mode}): {(raw or '')[:200]}")
    return []


_BUNDLED_EXPECTED = Counter({"natural": 1, "mild": 2, "hard": 1})


def craft_queries_bundled(group: list[dict], model: str, timeout: int) -> list[dict]:
    """Single LLM call producing 4 mode-tagged queries against group[0]:
    1 natural + 2 mild + 1 hard, attacking the same pinned target from
    different angles (Phase 3d, 2026-05-09).

    Returns a list of 4 dicts (each with `query`, `context`, `intent`,
    `mode`, `angle`, `target_ids`, and `_mode` for downstream-compat) on
    success. Returns [] if both attempts fail to parse or produce the
    wrong mode distribution -- caller falls back to the per-mode path.
    """
    if not group:
        return []
    target = group[0]
    target_id_short = target["id"][:8]
    target_text = (
        f"ID: {target_id_short}\n"
        f"Category: {target['category']}\n"
        f"Themes: {target['themes']}\n"
        f"Summary: {target['summary']}\n"
        f"Content: {target['content']}"
    )

    neighbors = _fetch_topical_neighbors(target, max_neighbors=3)
    if neighbors:
        context_text = "\n---\n".join(
            f"ID: {m['id'][:8]} | [{m['category']}] themes={m['themes']} | "
            f"summary: {m['summary']}"
            for m in neighbors
        )
    else:
        context_text = "(no topical neighbors found -- target is unique within its themes)"

    prompt = CRAFT_TEMPLATE_BUNDLED.format(
        target=target_text,
        context_memories=context_text,
        style_reference=STYLE_REFERENCE_BUNDLED,
        target_id=target_id_short,
    )

    for attempt in range(2):
        raw = call_llm(prompt, model, CRAFT_SYSTEM, timeout)
        parsed = parse_json_response(raw)
        if not isinstance(parsed, list) or len(parsed) != 4:
            got_len = len(parsed) if isinstance(parsed, list) else "n/a"
            log(f"  WARNING: bundled craft attempt {attempt+1} for {target_id_short}: "
                f"expected list of 4, got {type(parsed).__name__} of len {got_len}; "
                f"head={(raw or '')[:120]!r}")
            continue

        got = Counter(
            q.get("mode", "") for q in parsed if isinstance(q, dict)
        )
        if got != _BUNDLED_EXPECTED:
            log(f"  WARNING: bundled craft attempt {attempt+1} for {target_id_short}: "
                f"bad mode distribution {dict(got)}, expected {dict(_BUNDLED_EXPECTED)}")
            continue

        for q in parsed:
            angle = q.get("angle", "")
            log_detail(f"    bundled [{q['mode']}] angle: {angle}")
            q["target_ids"] = [target_id_short]
            q["_mode"] = q["mode"]
        return parsed

    log(f"  WARNING: bundled craft for {target_id_short} failed twice; "
        "falling back to per-mode path")
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

Here are the results returned (in rank order):

{results}

## Task 1 — utility ratings

Rate each returned memory's utility for this query's intent.
Scale: 0.0 = completely irrelevant, 0.3 = marginal, 0.6 = useful, 0.9 = critical hit.

Also provide:
- cutoff_rank: position (1-indexed) of the last useful result. 0 if nothing in top positions was useful.
- Rate every memory honestly against the stated intent. Do not assume any particular result was the "intended" answer — judge by relevance only.

## Task 2 — pairwise relationships (opportunistic)

Since these memories are already in context, surface any clear pairwise relationships among the ones you rated highly (>=0.6 utility). Skip pairs that are merely topically near; only flag relationships that would be useful for retrieval and reasoning.

Types:
- "related" — they cover the same topic or directly reference each other
- "supports" — one elaborates on, evidences, or refines the other
- "contradicts" — they say incompatible things about the same topic
- "supersedes" — one is a newer/correct version that replaces the other

Cap at 5 relationships. Return an empty array if nothing clear stands out — quality matters more than recall here.

## Output format

Return JSON:
{{"feedback": {{"mem_id": utility_score, ...}}, "cutoff_rank": N, "reason": "brief explanation",
  "relationships": [{{"a": "id1_short", "b": "id2_short", "type": "related|supports|contradicts|supersedes", "note": "brief reason"}}]}}"""


def rate_results(query: str, intent: str, target_ids: list[str],
                 recall_output: str, model: str, timeout: int) -> dict:
    """Have LLM rate the recall results.

    `target_ids` is kept in the signature so callers can post-hoc compute
    target-row rating distribution (a low-cost honesty signal — see
    Phase 3b). It is intentionally NOT shown to the rater: revealing the
    target biases neighbor labels ("less relevant than the target") and
    contaminates the gradient. The pinned 1.0 label still flows through
    via the probe_target event + GT dedup logic.
    """
    prompt = RATE_TEMPLATE.format(
        query=query,
        intent=intent,
        results=recall_output,
    )

    raw = call_llm(prompt, model, RATE_SYSTEM, timeout)
    parsed = parse_json_response(raw)
    if isinstance(parsed, dict) and "feedback" in parsed:
        return parsed
    raw = raw or ""
    log(f"  WARNING: failed to parse rate response (len={len(raw)}): "
        f"head={raw[:120]!r} tail={raw[-120:]!r}")
    return {}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

_VALID_MODES = ("mixed", "natural", "mild", "hard")


def _modes_for_group(n: int, mode: str, query_offset: int) -> list[str]:
    """Assign craft modes for n queries within a group (per-mode path only).

    As of Phase 3d (2026-05-09), --mode mixed routes through the bundled
    craft path (`craft_queries_bundled`) and does not call this function;
    the cycle below is only used as a fallback when the bundled call fails
    twice in a row, and for explicit single-mode invocations (where it
    just repeats the requested mode n times).

    The 50% mild / 25% natural / 25% hard cycle is preserved as the
    fallback distribution: mild is the highest-EV training signal
    (in-distribution, weight 1.2, distinctive IDs preserved), natural is a
    sanity floor (FTS should win), hard is held out and structurally
    bounded by candidate-pool blindness.
    """
    if mode == "mixed":
        cycle = ("mild", "natural", "mild", "hard")
        return [cycle[(query_offset + i) % len(cycle)] for i in range(n)]
    return [mode] * n


def run_probe(
    num_queries: int = 30,
    dry_run: bool = False,
    model: str = "opus",
    min_mems: int = 2,
    coverage_threshold: float = 0.5,
    batch_size: int = 1,
    content_chars: int = 600,
    recall_budget: int = 4000,
    recall_limit: int = 15,
    timeout: int = 240,
    extra_queries_path: str = None,
    workers: int = 1,
    mode: str = "mixed",
    mix: float = 0.0,
    adversarial_source: str = "real",
    adversarial_rank_threshold: int = 5,
):
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
    if not 0.0 <= mix <= 1.0:
        raise ValueError(f"mix must be in [0.0, 1.0], got {mix!r}")
    if adversarial_source not in ("real", "audit"):
        raise ValueError(
            f"adversarial_source must be 'real' or 'audit', got {adversarial_source!r}"
        )

    _init_log()
    log(f"=== Retrieval Probe ===")
    log(f"  queries={num_queries}, model={model}, budget={recall_budget}, "
        f"limit={recall_limit}, batch={batch_size}, mode={mode}")
    log(f"  coverage_threshold={coverage_threshold}, content_chars={content_chars}, "
        f"timeout={timeout}s")

    db = get_db()

    # 1. Find coverage gaps
    log("\n--- Finding coverage gaps ---")
    gaps, memory_content, memory_themes, mem_quality = find_coverage_gaps(db, min_mems, coverage_threshold)
    # Two units worth knowing: the (memory, theme) pair count drives priority
    # ordering (themes with more underserved members sort first), but it
    # double-counts memories that live in multiple underserved themes. The
    # unique memory count is what saturation is bounded by.
    pair_count = sum(len(g["underserved_ids"]) for g in gaps)
    unique_underserved = {mid for g in gaps for mid in g["underserved_ids"]}
    below_threshold = sum(1 for g in gaps if g["coverage"] < coverage_threshold)
    if below_threshold == len(gaps):
        log(f"  {len(gaps)} themes below {coverage_threshold:.0%} quality; "
            f"{len(unique_underserved)} unique underserved memories "
            f"({pair_count} memory-theme pairs)")
    else:
        log(f"  {below_threshold} themes below {coverage_threshold:.0%} -- "
            f"fallback: targeting {len(gaps)} lowest-quality themes, "
            f"{len(unique_underserved)} unique underserved memories "
            f"({pair_count} memory-theme pairs)")

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

    # Adversarial-first split: when mix > 0, the pathology selector picks
    # first (it's the scarcer signal — typically 30-80 entries), and the
    # coverage-fill selector backfills any unfilled adversarial budget. A
    # memory picked by the pathology selector is excluded from the coverage
    # pass so the two sources never produce duplicate targets in one run.
    adv_groups: list = []
    adv_info: dict = {}
    if mix > 0.0:
        # Translate query budget → group budget. Mixed mode = 4 queries/group;
        # single-mode batch=1 averages ~2 queries/group across the 1-3 cycle.
        per_group = 4 if mode == "mixed" else 2
        adv_groups_wanted = int(num_queries * mix) // per_group

        if adversarial_source == "real":
            from select_real_pathology_targets import select_real_pathology_targets
            adv_targets, adv_info = select_real_pathology_targets(
                adv_groups_wanted,
                memory_content=memory_content,
                content_chars=content_chars,
                rank_threshold=adversarial_rank_threshold,
            )
            log(f"  Adversarial (real-recall): {adv_info['selected']}/{adv_groups_wanted} "
                f"groups from {adv_info['hard_real_rows']} hard-real rows / "
                f"{adv_info['unique_pathology_memories']} unique memories "
                f"(util>={adv_info['utility_threshold']}, rank>={adv_info['rank_threshold']}; "
                f"skipped {adv_info['skipped_at_cap']} at-cap, "
                f"{adv_info['skipped_excluded']} excluded, "
                f"{adv_info['skipped_inactive']} inactive)")
        else:
            from select_pathology_targets import select_pathology_targets
            adv_targets, adv_info = select_pathology_targets(
                adv_groups_wanted,
                memory_content=memory_content,
                content_chars=content_chars,
            )
            if adv_info["audit_path"]:
                log(f"  Adversarial (audit): {adv_info['selected']}/{adv_groups_wanted} groups "
                    f"from {adv_info['audit_path']} ({adv_info['audit_pathologies']} pathologies; "
                    f"skipped {adv_info['skipped_at_cap']} at-cap, "
                    f"{adv_info['skipped_excluded']} excluded, "
                    f"{adv_info['skipped_inactive']} inactive)")
            else:
                log(f"  Adversarial (audit): 0 groups (no pathology audit found in "
                    f"data/pathology_audits/; coverage-fill will absorb the full budget)")
        adv_groups = [[t] for t in adv_targets]

    adv_mids = {g[0]["id"] for g in adv_groups}
    cov_query_budget = num_queries - (len(adv_groups) * (4 if mode == "mixed" else 2))
    cov_groups = select_targets(
        gaps, memory_content, mem_quality,
        max(cov_query_budget, 0), batch_size, content_chars,
        exclude=adv_mids,
    )

    groups = adv_groups + cov_groups
    if mix > 0.0:
        log(f"  Mixed selection: {len(adv_groups)} adversarial + {len(cov_groups)} coverage = "
            f"{len(groups)} groups")

    # Assign query budget per group up-front so dry-run can report the same
    # plan that would actually execute.
    #
    # Phase 3d (2026-05-09): mixed mode consumes exactly 4 queries per group
    # (1 natural + 2 mild + 1 hard, produced by a single bundled LLM call).
    # Single-mode invocations keep the historical 1-3 budget per group.
    # group_query_offsets is only consulted on the per-mode fallback path;
    # bundled mixed-mode produces its own fixed 1n/2m/1h order.
    queries_per_group = []
    group_query_offsets = []
    remaining_budget = num_queries
    running_offset = 0
    for group in groups:
        if mode == "mixed":
            if remaining_budget < 4:
                break
            n = 4
        else:
            n = max(1, min(3, remaining_budget))
        queries_per_group.append(n)
        group_query_offsets.append(running_offset)
        running_offset += n
        remaining_budget -= n
        if remaining_budget <= 0:
            break
    groups = groups[:len(queries_per_group)]

    if dry_run:
        log("\n--- DRY RUN: would craft queries for these groups ---")
        if mode == "mixed":
            log(f"  {len(groups)} groups x 4 queries = {len(groups) * 4} queries planned "
                f"(bundled craft: 1 natural + 2 mild + 1 hard per group)")
        else:
            log(f"  {len(groups)} groups, {sum(queries_per_group)} queries planned "
                f"(per-mode path, mode={mode})")
        for i, group in enumerate(groups):
            n = queries_per_group[i]
            log(f"\n  Group {i+1} (n={n}):")
            for m in group:
                log(f"    {m['id'][:8]} [{m['category']}] {m['summary'][:80]}")
        return

    # 3. Craft queries + run + rate (parallel across groups)
    log(f"\n--- Crafting queries and probing (workers={workers}) ---")
    total_queries = 0
    total_feedback = 0
    # Phase 3b: collect the LLM's rating of the pinned target row to detect
    # probe-craft drift. The rater no longer sees which result is the target,
    # so this is a clean honesty check — if probes consistently land their
    # target at <0.5 utility, the craft step is generating queries that don't
    # actually point to the target.
    target_row_ratings: list[float] = []
    target_row_missing = 0  # pinned target wasn't in the rated set
    # Phase 3d: aggregate mode counts (across bundled + fallback paths) and
    # track how many groups successfully used the bundled craft.
    total_mode_counts: Counter = Counter()
    bundled_success_groups = 0
    bundled_attempted_groups = 0

    log(f"  {len(groups)} target groups ({sum(len(g) for g in groups)} unique memories)")

    def process_group(args):
        idx, group, n_per_group, q_offset = args
        group_queries = 0
        group_feedback = 0
        group_target_ratings: list[float] = []
        group_target_missing = 0
        group_mode_counts: Counter = Counter()
        group_bundled_attempted = 0
        group_bundled_succeeded = 0

        # group[0] is the pinned target. The remaining group memories are
        # not used (topical neighbors are fetched fresh inside craft_queries).
        log(f"\n  Group {idx+1}/{len(groups)} target: {group[0]['summary'][:80]}")
        log_detail(f"    Target ID: {group[0]['id'][:8]}")

        # Pinned anchor: group[0] (the highest-priority underserved memory in
        # the group, after select_targets' sort). All mode calls for this
        # group target the same memory so cross-mode rank deltas are paired.
        pinned_id_short = group[0]["id"][:8]
        pinned_summary = group[0]["summary"][:80]
        log_detail(f"    Pinned target: {pinned_id_short} -- {pinned_summary}")

        # Phase 3d: mixed mode goes through a single bundled LLM call that
        # produces 4 mode-tagged orthogonal queries. On parse / mode-count
        # failure (after one retry inside craft_queries_bundled), fall back
        # to the legacy per-mode path so the group still contributes.
        # Single-mode invocations always use the per-mode path.
        queries: list[dict] = []
        if mode == "mixed":
            group_bundled_attempted = 1
            queries = craft_queries_bundled(group, model, timeout)
            if queries:
                group_bundled_succeeded = 1
            else:
                log_detail(f"    Group {idx+1}: bundled craft failed, falling back to per-mode")

        if not queries:
            group_modes = _modes_for_group(n_per_group, mode, q_offset)
            for q_mode in group_modes:
                sub = craft_queries(group, 1, model, timeout, craft_mode=q_mode)
                for s in sub:
                    s["_mode"] = q_mode
                    queries.append(s)

        if not queries:
            log(f"    Group {idx+1}: WARNING: no queries crafted, skipping")
            return (0, 0, [], 0, group_mode_counts,
                    group_bundled_attempted, group_bundled_succeeded)

        for q in queries:
            query_text = q.get("query", "")
            context_text = q.get("context", "")
            intent = q.get("intent", "")
            target_ids = q.get("target_ids", [pinned_id_short])
            q_mode = q.get("_mode", "natural")
            if not query_text:
                continue
            group_mode_counts[q_mode] += 1

            log(f"\n    [{idx+1}/{q_mode}] Query: {query_text}")
            if context_text:
                log(f"    [{idx+1}/{q_mode}] Context: {context_text}")
            log(f"    [{idx+1}/{q_mode}] Intent: {intent}")
            log(f"    [{idx+1}/{q_mode}] Targets: {target_ids}")

            recall_output = impl_recall(
                query=query_text,
                context=context_text,
                budget=recall_budget,
                limit=recall_limit,
                internal=False,
            )
            group_queries += 1

            log_detail(f"    [{idx+1}/{q_mode}] Recall output:\n{recall_output}")

            # Extract rank order from recall_output. target_rank is 1-indexed
            # for the pinned target; None if not in candidate pool (Phase 1
            # miss). Distinguishes "ranker buried it" (high rank) from
            # "retrieval channels never surfaced it" (None).
            ranked_ids = _parse_ranked_ids(recall_output)
            target_ranks = {}
            for tid in target_ids:
                if tid in ranked_ids:
                    target_ranks[tid] = ranked_ids.index(tid) + 1
                else:
                    target_ranks[tid] = None

            phase1_misses = [t for t, r in target_ranks.items() if r is None]
            if phase1_misses:
                log(f"    [{idx+1}/{q_mode}] Phase 1 miss: targets not in candidate pool: {phase1_misses}")
                # Queue for next-sleep repair pass — but only for natural/mild.
                # Hard-mode misses are designed-difficulty failures, not real
                # coverage gaps, so we don't want them clogging the repair queue.
                if q_mode in ("natural", "mild"):
                    # The pinned target's full ID lives in group[0]; target_ids
                    # only carries the 8-char prefix.
                    pinned_full = group[0]["id"]
                    _emit_phase1_miss_event(
                        pinned_full, query_text, q_mode, ranked_ids
                    )
                    log(f"    [{idx+1}/{q_mode}] Queued recall_miss for repair (target={pinned_full[:8]})")
            for t, r in target_ranks.items():
                if r is not None:
                    log(f"    [{idx+1}/{q_mode}] Target rank: {t}={r}/{len(ranked_ids)}")

            # Emit probe_target event for the pinned target (independent of
            # LLM rating). Lossless: written for hits, misses, and all modes.
            pinned_full_id = group[0]["id"]
            _emit_probe_target_event(
                target_id_full=pinned_full_id,
                query=query_text,
                context_text=context_text,
                mode=q_mode,
                target_rank=target_ranks.get(pinned_id_short),
                candidate_pool_size=len(ranked_ids),
            )

            rating = rate_results(query_text, intent, target_ids, recall_output,
                                  model, timeout)
            if not rating or "feedback" not in rating:
                log(f"    [{idx+1}/{q_mode}] WARNING: no ratings returned, skipping feedback")
                continue

            feedback = rating["feedback"]
            cutoff = rating.get("cutoff_rank", -1)
            reason = rating.get("reason", "")
            relationships = rating.get("relationships", []) or []

            # Pinned-target honesty signal — only meaningful when the target
            # was actually in the rated set (which excludes Phase-1 misses).
            target_score = feedback.get(pinned_id_short)
            if target_score is None:
                group_target_missing += 1
            else:
                try:
                    group_target_ratings.append(float(target_score))
                except (TypeError, ValueError):
                    group_target_missing += 1

            log(f"    [{idx+1}/{q_mode}] Cutoff: {cutoff}, Ratings: {len(feedback)}")
            log_detail(f"    [{idx+1}/{q_mode}] Reason: {reason}")
            log_detail(f"    [{idx+1}/{q_mode}] Feedback: {json.dumps(feedback)}")

            # Embed mode + per-target rank in the feedback reason so analysis
            # can stratify GT by difficulty later (queryable via SQL on
            # memory_events.context.reason).
            target_rank_str = ",".join(
                f"{t}={r if r is not None else 'miss'}"
                for t, r in target_ranks.items()
            )
            reason_tagged = (
                f"[probe-{q_mode}] target_rank={target_rank_str} | {reason}"
            )

            # Pass the LLM's surfaced relationships through to
            # impl_recall_feedback — it handles edge creation, dedup, and
            # the >=0.6 utility floor uniformly across probe and live
            # recall-feedback callers. Probe vs live provenance is preserved
            # in the [probe-mode] tag inside the reason field.
            fb_result = impl_recall_feedback(
                feedback=json.dumps(feedback),
                query=query_text,
                reason=reason_tagged,
                cutoff_rank=cutoff,
                relationships=json.dumps(relationships),
            )
            group_feedback += len(feedback)
            log(f"    [{idx+1}/{q_mode}] {fb_result}")

        return (group_queries, group_feedback, group_target_ratings,
                group_target_missing, group_mode_counts,
                group_bundled_attempted, group_bundled_succeeded)

    work_items = [(i, g, n, o) for i, (g, n, o) in
                  enumerate(zip(groups, queries_per_group, group_query_offsets))]

    def _accumulate(result):
        nonlocal total_queries, total_feedback, target_row_missing
        nonlocal bundled_success_groups, bundled_attempted_groups
        (q_count, fb_count, tgt_ratings, tgt_missing,
         mode_counts, bnd_attempted, bnd_succeeded) = result
        total_queries += q_count
        total_feedback += fb_count
        target_row_ratings.extend(tgt_ratings)
        target_row_missing += tgt_missing
        total_mode_counts.update(mode_counts)
        bundled_attempted_groups += bnd_attempted
        bundled_success_groups += bnd_succeeded

    if workers > 1 and len(work_items) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(process_group, work_items):
                _accumulate(result)
    else:
        for item in work_items:
            _accumulate(process_group(item))

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

                    # Emit probe_target event for extra-query pinned targets.
                    # target_ids in the extra-queries JSON carries 8-char prefixes;
                    # resolve to a full memory id before logging. Skip if no
                    # target_ids or if the prefix can't be uniquely resolved.
                    if target_ids:
                        prefix = (target_ids[0] or "").strip()
                        if len(prefix) >= 8:
                            db_lookup = get_db()
                            try:
                                row = db_lookup.execute(
                                    "SELECT id FROM memories "
                                    "WHERE substr(id, 1, 8) = ? AND status = 'active'",
                                    (prefix[:8],),
                                ).fetchone()
                            finally:
                                db_lookup.close()
                            if row:
                                ranked_ids_extra = _parse_ranked_ids(recall_output)
                                short = prefix[:8]
                                tr = (ranked_ids_extra.index(short) + 1
                                      if short in ranked_ids_extra else None)
                                if len(target_ids) > 1:
                                    log(f"    WARNING: extra query has {len(target_ids)} targets, "
                                        f"only emitting probe_target for the first ({short})")
                                _emit_probe_target_event(
                                    target_id_full=row["id"],
                                    query=query_text,
                                    context_text=context_text,
                                    mode="extra",
                                    target_rank=tr,
                                    candidate_pool_size=len(ranked_ids_extra),
                                )

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

    # Phase 3d: bundled-craft summary. In mixed mode, each successful bundled
    # call should contribute 1 natural + 2 mild + 1 hard. Fallbacks to the
    # per-mode path use the 50/25/25 cycle, so total mode counts may diverge
    # from the strict 1n/2m/1h-per-target shape when bundled fails.
    if mode == "mixed" and bundled_attempted_groups:
        n_bundled = bundled_success_groups
        n_fallback = bundled_attempted_groups - bundled_success_groups
        log(
            f"Bundled craft: {n_bundled}/{bundled_attempted_groups} targets succeeded "
            f"({n_fallback} fell back to per-mode); produced "
            f"{{natural: {total_mode_counts.get('natural', 0)}, "
            f"mild: {total_mode_counts.get('mild', 0)}, "
            f"hard: {total_mode_counts.get('hard', 0)}}}"
        )

    # Phase 3b: target-row LLM rating distribution. The rater no longer sees
    # the target, so this is a clean check on probe-craft quality. Median
    # >= 0.7 is healthy; < 0.5 is a red flag for craft drift.
    if target_row_ratings:
        rs = sorted(target_row_ratings)
        n = len(rs)
        median = rs[n // 2] if n % 2 else (rs[n // 2 - 1] + rs[n // 2]) / 2
        p10 = rs[max(0, int(n * 0.10) - 1)]
        p90 = rs[min(n - 1, int(n * 0.90))]
        below_05 = sum(1 for v in rs if v < 0.5)
        log(
            f"Target row LLM rating distribution across {n} probes "
            f"(target hidden from rater): median={median:.2f} "
            f"p10={p10:.2f} p90={p90:.2f} "
            f"<0.5={below_05}/{n}"
        )
        if target_row_missing:
            log(f"  Target absent from rated set: {target_row_missing} (Phase-1 miss or rater dropped row)")
    else:
        log("Target row LLM rating distribution: no ratable target rows "
            "(all Phase-1 misses or no probes ran)")

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
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Memories per group. With pinning, only group[0] is targeted, "
                             "so the default is 1 (no wasted candidates). Larger values "
                             "are a holdover from the pre-pinning design.")
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
    parser.add_argument("--mode", type=str, default="mixed",
                        choices=list(_VALID_MODES),
                        help="Query difficulty mode: mixed cycles 50% mild / 25% natural / 25% hard; "
                             "single mode applies it to every query")
    parser.add_argument("--mix", type=float, default=0.0,
                        help="Adversarial-vs-coverage mix. 0.0 = pure coverage-fill (default). "
                             "0.5 = half adversarial + half coverage-fill from the remainder. "
                             "Adversarial source is controlled by --adversarial-source. "
                             "Adversarial picks first, coverage backfills any unfilled "
                             "adversarial budget when the pathology list is shorter than the budget.")
    parser.add_argument("--adversarial-source", type=str, default="real",
                        choices=["real", "audit"],
                        help="Where adversarial targets come from when --mix > 0. "
                             "'real' (default): high-utility memories that ranked low in "
                             "actual production recalls (mined from memory_events). "
                             "'audit': memories the latest audit_reranker_pathology.py JSON "
                             "flags as content-residual or summary-as-query failures "
                             "(kept for sentinel-class regression checks; see V5+2 group "
                             "analysis for why this isn't the primary signal).")
    parser.add_argument("--adversarial-rank-threshold", type=int, default=5,
                        help="Real-recall pathology cutoff: minimum 0-indexed rank in "
                             "recall_meta.kept for a high-utility memory to count as "
                             "buried. Only used with --adversarial-source real. "
                             "Default 5 (6th position or later); lower = more candidates "
                             "(rank>=3: 40 candidates, rank>=5: 16, rank>=7: 5 against "
                             "production at the time of writing). The model never buries "
                             "useful memories past position 10, so the cap supply is "
                             "structurally tight.")
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
        mode=args.mode,
        mix=args.mix,
        adversarial_source=args.adversarial_source,
        adversarial_rank_threshold=args.adversarial_rank_threshold,
    )


if __name__ == "__main__":
    main()
