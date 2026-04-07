# /// script
# requires-python = ">=3.11"
# dependencies = ["sqlite-vec>=0.1.6", "openai>=2.0.0", "tiktoken>=0.7.0", "mcp[cli]>=1.2.0", "lightgbm>=4.0"]
# ///
"""
Claude Memory MCP Server — persistent memory with hybrid search.

MCP wiring layer — tool decorators delegate to memory/ package modules.
12 tools: startup_load, remember, recall, recall_feedback, link, update, forget,
  reflect, review_pending, consolidate, reembed_all, memory_stats

Stack: SQLite + sqlite-vec (vector KNN) + FTS5 (keyword) + OpenAI embeddings API
Transport: stdio (spawned by Claude Code)
Database: ~/.claude/data/memory.db
"""

import logging
import os
import sys

# Ensure DATA_DIR points to production DB regardless of which memory_server.py runs
if "SOMNIGRAPH_DATA_DIR" not in os.environ:
    _default = os.path.join(os.path.expanduser("~"), ".claude", "data")
    if os.path.isdir(_default):
        os.environ["SOMNIGRAPH_DATA_DIR"] = _default

from mcp.server.fastmcp import FastMCP

# Log to file for post-mortem debugging (stderr is discarded by Claude Code)
_log_path = os.path.join(
    os.environ.get("SOMNIGRAPH_DATA_DIR", os.path.join(os.path.expanduser("~"), ".claude", "data")),
    "memory_server.log",
)
logging.basicConfig(
    handlers=[
        logging.FileHandler(_log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("claude-memory")
logger.info("Memory server starting (PID %d)", os.getpid())

# ---------------------------------------------------------------------------
# Re-exports for sleep script compatibility (sleep_nrem.py, sleep_rem.py, sleep.py)
# These scripts do: from memory_server import get_db, embed_text, ...
# ---------------------------------------------------------------------------

from memory.constants import *  # noqa: E402,F403 — all tuning constants
from memory.vectors import (  # noqa: E402
    serialize_f32, deserialize_f32, _dot, _vec_sub, _vec_scale, _norm, _novelty_score,
)
from memory.privacy import _strip_sensitive  # noqa: E402
from memory.embeddings import (  # noqa: E402
    get_openai_client, get_tokenizer, count_tokens,
    embed_text, embed_batch, build_enriched_text,
)
from memory.formatting import (  # noqa: E402
    format_memory_compact, format_memory_full, format_memory_pending,
)
from memory.fts import KNOWN_PHRASES, sanitize_fts_query, _themes_for_fts, update_fts, delete_fts  # noqa: E402
from memory.themes import (  # noqa: E402
    THEME_VARIANTS, CONTENT_THEME_PHRASES, LEARNED_MAPPINGS_PATH,
    _get_all_mappings, add_theme_mapping, normalize_themes,
    normalize_all_themes, split_theme_on_memories,
)
from memory.db import get_db, DB_PATH, _resolve_id  # noqa: E402
from memory.sync import _row_get, _log_event  # noqa: E402
from memory.decay import effective_priority  # noqa: E402
from memory.write import _insert_memory  # noqa: E402
from memory.graph import (  # noqa: E402
    _create_edge, _handle_temporal_evolution, _find_related_memories,
    _check_fast_path, _compute_shadow_load, _source_confidence_modifier,
)
from memory.stats import compute_stats  # noqa: E402

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

from memory.tools import (  # noqa: E402
    impl_startup_load,
    impl_remember,
    impl_recall,
    impl_recall_feedback,
    impl_link,
    impl_update,
    impl_forget,
    impl_reflect,
    impl_review_pending,
    impl_consolidate,
    impl_reembed_all,
    impl_memory_stats,
)

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP("claude-memory")


@mcp.tool()
def startup_load(budget: int = 3000) -> str:
    """Session start briefing. Returns highest-priority active memories within token budget.

    Args:
        budget: Maximum tokens for the briefing (default 3000).
    """
    return impl_startup_load(budget)


@mcp.tool()
def remember(
    content: str,
    category: str = "semantic",
    priority: int = 5,
    themes: str = "[]",
    summary: str = "",
    source: str = "session",
    status: str = "active",
    decay_rate: float = -1,
    confidence: float = -1,
    flags: str = "[]",
) -> str:
    """Store a new memory with embedding and dedup check.

    Args:
        content: The memory content. Embedded as a vector for semantic similarity search —
            write in natural language with full context, relationships, and reasoning.
            Richer context produces better semantic matches.
        category: One of: episodic, semantic, procedural, reflection, meta, entity.
            Entity memories are curated summaries of named entities (people, projects,
            places) that serve as retrieval hubs linked to detailed memories.
        priority: Importance 1-10 (10 = pinned, never decays).
        themes: JSON array of tags, indexed by FTS5 for keyword retrieval.
            Use canonical hyphenated terms, e.g. '["skyrim-modding","identity"]'.
        summary: One-line summary, indexed by FTS5 for keyword matching. Include specific
            names, technical terms, and key identifiers. Auto-generated if empty.
        source: Origin: session, manual, auto, journal, correction, etc.
        status: 'active' (default) or 'pending' (for auto-capture awaiting review).
        decay_rate: Custom decay rate. -1 (default) = use category default. 0 = timeless.
        confidence: Trust level 0.1-0.95. -1 (default) = auto-resolve from source/status.
        flags: JSON array of flags, e.g. '["pinned"]'. "pinned" or "keep" = immune to decay.
    """
    return impl_remember(content, category, priority, themes, summary, source, status, decay_rate, confidence, flags)


@mcp.tool()
def recall(
    query: str,
    context: str = "",
    budget: int = 5000,
    category: str = "",
    limit: int = 5,
    exclude_ids: str = "[]",
    since: str = "",
    before: str = "",
    boost_themes: str = "[]",
    min_priority: int = 0,
    internal: bool = False,
) -> str:
    """Hybrid search: RRF fusion over vector similarity + FTS5 keyword matching.

    Args:
        query: **Feeds FTS5 keyword search only.** Optimize for FTS5 token matching:
            use specific nouns, proper names, and technical terms that appear literally
            in memory content, summaries, or theme tags. For each word, ask: "would
            this token appear in a memory I stored?" Drop words that describe the
            search intent rather than matching stored content.
            Good: `"bread garden Claudie letter seed"`
            Bad: `"recent thing brought gift correspondence"`
            Multi-word phrases like "power bi" are auto-detected from a known list;
            use explicit quotes for others (e.g. '"Dear Bread" letter').
        context: **Feeds vector cosine-distance search only.** Optimize for embedding
            similarity: write a paragraph-length natural-language description of what
            you're looking for. This text is embedded and compared against stored memory
            embeddings (which are built from content + category + themes + summary).
            Richer context = closer match to stored paragraph-length embeddings. A 3-word
            query embeds in a narrow semantic space and misses relevant memories whose
            embeddings were built from full paragraphs. If empty, query is used as
            fallback for vector search (but this is suboptimal for anything beyond
            simple lookups). Use boost_themes aggressively alongside context.
        budget: Maximum tokens for results (default 5000).
        category: Optional filter by category.
        limit: How many results to return (default 5). Set per-query based on intent:
            1-3 focused lookup, 5 standard recall, 8-13 broad exploration.
            Independent of budget — limit caps count, budget caps tokens.
        exclude_ids: JSON array of memory IDs to skip, e.g. '["id1","id2"]'. Use for iterative refinement.
        since: Only memories created on or after this date. ISO date or relative: "7d", "30d", "90d".
        before: Only memories created before this date. ISO date format.
        boost_themes: JSON array of themes to weight higher, e.g. '["chess","calibration"]'.
        min_priority: Only return memories with base_priority >= this value.
        internal: Set true for subagent/background use. Suppresses retrieval event
            logging and feedback ID footer (prevents stop hook from tracking these recalls).
    """
    logger.info("recall() called: query=%r, limit=%d", query[:80], limit)
    return impl_recall(query, context, budget, category, limit, exclude_ids, since, before, boost_themes, min_priority, internal)


@mcp.tool()
def recall_feedback(
    feedback: str = "{}",
    query: str = "",
    reason: str = "",
    cutoff_rank: int = -1,
) -> str:
    """Report utility feedback on recalled memories to improve future retrieval.

    Flat format — each key is a memory ID (8-char prefix or full UUID), value is either:
      - A float 0.0-1.0 (utility only)
      - A list [utility, durability] where durability is -1.0 to 1.0

    Example: {"c2dc0332": 0.9, "38282da2": 0.4, "79ca5bfb": [0.1, -0.8]}

    Utility scale: 0.0 = useless, 0.3 = marginal, 0.6 = useful, 0.9 = critical.
    Durability: -1.0 = stale, 0.0 = neutral, 1.0 = enduring.

    Args:
        feedback: JSON object mapping memory IDs to utility scores.
        query: The recall query that produced these results (for context tracking).
        reason: Short explanation of how these memories were used.
        cutoff_rank: Position of last useful result (1-indexed). 0 = nothing useful.
            -1 (default) = not reporting cutoff. Used to tune quality floor threshold.
    """
    return impl_recall_feedback(feedback, query, reason, cutoff_rank)


@mcp.tool()
def link(
    source_id: str,
    target_id: str,
    linking_context: str,
    flags: str = "[]",
) -> str:
    """Create a relationship edge between two memories that are both in context.

    Use this when you can see a genuine relationship between memories during a session —
    the linking_context you provide (why they relate) is higher quality than what sleep
    would infer from embeddings alone.

    Args:
        source_id: Full or 8-char prefix ID of the source memory.
        target_id: Full or 8-char prefix ID of the target memory.
        linking_context: Why these memories are related (natural language). This is the
            high-value part — be specific about the relationship.
        flags: JSON array of structural flags: "contradiction", "revision", "derivation".
            Empty array [] for contextual/thematic links (the common case).
    """
    return impl_link(source_id, target_id, linking_context, flags)


@mcp.tool()
def update(
    memory_id: str,
    content: str = "",
    summary: str = "",
    themes: str = "",
    category: str = "",
    priority: int = -1,
    flags: str = "",
    decay_rate: float = -999,
    confidence: float = -999,
) -> str:
    """Update an existing memory in-place, preserving edges, feedback stats, and co-retrieval history.

    Only provide fields you want to change — omitted fields are left untouched.
    Re-embeds automatically if content, summary, themes, or category change.

    Args:
        memory_id: Full or prefix ID of the memory to update.
        content: New content (replaces existing). Empty = no change.
        summary: New summary. Empty = no change.
        themes: New themes as JSON array, e.g. '["home-office","furniture"]'. Empty = no change.
        category: New category. Empty = no change.
        priority: New priority 1-10. -1 = no change.
        flags: New flags as JSON array. Empty = no change.
        decay_rate: New decay rate. -999 = no change. -1 = use category default. 0 = timeless.
        confidence: New confidence 0.1-0.95. -999 = no change.
    """
    return impl_update(memory_id, content, summary, themes, category, priority, flags, decay_rate, confidence)


@mcp.tool()
def forget(memory_id: str) -> str:
    """Soft-delete a memory. It won't appear in searches or startup but can be recovered.

    Args:
        memory_id: Full or prefix ID of the memory to delete.
    """
    return impl_forget(memory_id)


@mcp.tool()
def reflect(memory_id: str) -> str:
    """Touch a memory: update last_accessed and increment reflect_count (reheat).

    Use when a memory is referenced in conversation but not through search.

    Args:
        memory_id: Full or prefix ID of the memory to reheat.
    """
    return impl_reflect(memory_id)


@mcp.tool()
def review_pending(
    action: str = "list",
    memory_id: str = "",
    edit_content: str = "",
) -> str:
    """Manage auto-captured memories awaiting review.

    Args:
        action: One of: list, confirm, confirm_all, discard, discard_all.
        memory_id: Required for confirm/discard actions.
        edit_content: Optional new content when confirming (replaces existing).
    """
    return impl_review_pending(action, memory_id, edit_content)


@mcp.tool()
def consolidate(category: str = "") -> str:
    """Lightweight dedup, orphan cleanup, and stale pending cleanup.

    Not the full sleep skill — just finds obvious duplicates, prunes orphaned
    embedding rows, and cleans up old pending memories.

    Args:
        category: Optional category filter. Empty = all categories.
    """
    return impl_consolidate(category)


@mcp.tool()
def reembed_all() -> str:
    """Re-embed all active and pending memories with enriched text.

    Use after changing embedding logic or enrichment strategy. Iterates all
    non-deleted memories, recomputes enriched embeddings, and updates the
    vector table.
    """
    return impl_reembed_all()


@mcp.tool()
def memory_stats() -> str:
    """Diagnostic overview of the memory store. Returns counts, access stats,
    decay distribution, and token budget without loading memory content."""
    logger.info("memory_stats() called")
    return impl_memory_stats()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import atexit
    import threading
    from memory.reranker import warmup as _reranker_warmup

    atexit.register(lambda: logger.info("Memory server exiting (PID %d)", os.getpid()))

    # Eager-load reranker model in background so first recall() doesn't block
    def _warmup_thread():
        try:
            db = get_db()
            _reranker_warmup(db)
        except Exception as e:
            logger.warning("Background reranker warmup failed: %s", e)

    threading.Thread(target=_warmup_thread, daemon=True).start()

    mcp.run(transport="stdio")
