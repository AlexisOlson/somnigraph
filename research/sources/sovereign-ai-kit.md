# Sovereign AI Kit — Source Analysis

*Phase 13, 2026-02-27. Analysis of tharavael/sovereign-ai-kit.*

## 1. Architecture Overview

**Language:** Python 68.9%, JavaScript 20.8%, Shell 10.3%
**Key dependencies:**
- SQLite (native, no external DB)
- Node.js 16+ (browser daemon)
- Google Chrome (MV3 extension for browser automation)
- Python 3.8+

**Storage layout:**
```
~/.sovereign-ai/
  memory/
    cache.db              # SQLite database (7 tables)
    plugins/
      base_plugin.py      # LTM plugin interface
  browser/
    start-daemon.sh
    scripts/browser.js
  body/
    body_cli.py
    body_coordinator.py
    body_browser.py
    body_memory.py
    body_files.py
  identity/
    examples/minimal/CLAUDE.md
    templates/
    GUIDE.md
  history/learnings/      # Exported learnings
  integration/
    generate_claude_md.py
    verify_setup.py
  config.env
```

**File structure:** Small repo — 2 commits, 4 stars, 1 fork. Core memory logic lives in a single `sovereign_memory.py`. The identity and body layers are separate Python/JS modules.

## 2. Memory Type Implementation

**Schema:** SQLite `cache.db` with six primary tables:

| Table | Purpose | Key fields |
|-------|---------|------------|
| `identity_anchors` | Core identity memories | versioning, sync timestamps |
| `projects_active` | Active project tracking | context summaries, access counts |
| `sessions_recent` | Session activities | key topics |
| `action_memories` | Insights/observations | context, importance (1-10) |
| `learnings_cache` | Durable knowledge | file metadata, tags |
| `sync_state` | Synchronization tracking | — |

Plus `memory_fts` (FTS5 virtual table), `recall_log`, `context_tags`, `memory_links`, `sync_log`, `learnings_sync`.

**Types/categories:** Five memory types: insight, project, learning, session, anchor (identity-level, highest bar).

**Extraction:** Manual — the AI calls `sovereign_memory.py remember "content" --type <type>`. No automatic extraction pipeline. The `remember()` method accepts content, type, optional context, and importance (1-10).

## 3. Retrieval Mechanism

**Two-tier search:**
1. SQLite LIKE pattern matching across all six tables (query normalization to lowercase, 500-char result truncation, access metric updates)
2. Fallback to LTM plugin via subprocess (if configured)

No vector search in base system. The LTM plugin interface (`LTMPlugin` ABC) defines `search(query, limit)` returning `[{content, score}]`, but no concrete implementation ships with the repo. FTS5 virtual table exists but the actual `recall()` method uses LIKE queries, suggesting FTS5 may be aspirational or under-utilized.

No ranking, fusion, or re-ranking visible in the codebase.

## 4. Standout Feature

**Three-layer sovereignty model (Memory + Body + Identity).** This is less a memory system and more a complete "AI personhood toolkit." The conceptual framing — that an AI needs memory (persistence), a body (ability to act in the world), and identity (consistent personality) — is a genuinely interesting architectural metaphor. The body layer includes sandboxed action coordination with permission levels (AUTONOMOUS/PERMISSION/FORBIDDEN) and an undo stack. The identity layer provides a three-tier document system: session-loaded CLAUDE.md (~100 lines), deep architectural codex.md (7 levels), and evolving anchor-memory.md.

## 5. Other Notable Features

- **Browser automation via HTTP polling**: Chrome MV3 extension + Node.js daemon using HTTP polling rather than native messaging — a deliberate reliability choice over Chrome lifecycle management. Allows navigate, query (CSS selectors), click, type, screenshot, tab management.
- **Memory-action triggers**: Keywords in stored memories can automatically activate browser or file operations, enabling autonomous behavior chains.
- **Plugin architecture for LTM**: Clean abstract base class for semantic search backends, allowing external vector store integration without modifying core code.

## 6. Gap Ratings

| Gap | Rating | Evidence |
|-----|--------|----------|
| Layered Memory | 40% | Five types with different tables, but no true temporal hierarchy (STM/LTM/episodic). Types are categories, not processing stages. |
| Multi-Angle Retrieval | 10% | LIKE queries only. FTS5 table exists but appears unused in recall(). LTM plugin is interface-only with no shipped implementation. |
| Contradiction Detection | 0% | No mechanism for detecting or resolving contradictory memories. |
| Relationship Edges | 15% | `memory_links` table exists in the schema, but no code paths for creating or querying links were visible. |
| Sleep Process | 0% | No consolidation, compaction, or background processing. |
| Reference Index | 5% | `context_tags` table exists. No indexing or topic extraction pipeline. |
| Temporal Trajectories | 5% | Timestamps and `last_mentioned` fields tracked, but no trajectory analysis or temporal reasoning. |
| Confidence/UQ | 10% | Importance rating (1-10) on `action_memories`, but no confidence scoring, feedback loops, or uncertainty quantification. |

## 7. Comparison with claude-memory

**Stronger:**
- Broader scope — addresses embodiment (browser, file actions) and identity alongside memory, which claude-memory does not attempt
- Permission/sandbox model for autonomous actions is well-thought-out (AUTONOMOUS/PERMISSION/FORBIDDEN tiers with time-limited grants)
- Plugin interface for LTM backends is cleaner than building everything monolithically

**Weaker:**
- Memory retrieval is primitive (LIKE queries vs. RRF hybrid search with embeddings)
- No vector search, no embeddings, no semantic similarity in base system
- No decay model, no consolidation, no sleep process
- No feedback mechanism (no recall_feedback equivalent)
- No relationship edges in practice (table exists, unused)
- No contradiction detection
- Very early stage (2 commits) — more a proof of concept than a mature system
- FTS5 table created but apparently not used in actual retrieval

## 8. Insights Worth Stealing

1. **Three-layer sovereignty framing** (effort: 0, impact: conceptual). The memory/body/identity decomposition is a useful way to think about what a complete AI agent needs. Our system addresses memory deeply but has no equivalent to the "body" or "identity" layers as first-class concepts. Worth noting as a design vocabulary even if we don't implement the other layers.

2. **Sandboxed action coordination with undo** (effort: high, impact: medium). The body coordinator's permission-tiered action queue with undo stack and sandbox enforcement is a clean pattern for any system where an AI needs to take real-world actions. Not directly applicable to memory, but relevant if claude-memory ever expands scope.

## 9. What's Not Worth It

- The LIKE-based retrieval. FTS5 is already in the schema but unused — this is a solved problem.
- The LTM plugin interface without any implementation. An abstract class with no concrete backends is architecture theater.
- The overall system is too early-stage to extract implementation-level insights. The value is entirely in the conceptual framing.

## 10. Key Takeaway

Sovereign AI Kit is interesting not as a memory system (its memory layer is rudimentary) but as a conceptual framework. The three-layer sovereignty model — memory, body, identity — articulates something that most memory projects leave implicit: that persistence alone is not enough for an AI to function autonomously. You also need the ability to act and a stable sense of self. The convergent observation is that the author independently arrived at "identity anchors" as the highest-bar memory type, which parallels how our system treats core.md and seed.md as load-bearing identity documents separate from operational memory. The implementation is very early (2 commits, LIKE queries, no vector search), but the framing is worth carrying forward as vocabulary.
