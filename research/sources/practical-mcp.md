# Practical MCP Memory Servers -- Tier 2 Analysis

*Phase 13, 2026-03-05. Combined analysis of 4 community MCP memory implementations.*

## 1. memory-mcp (yuvalsuede)

### Architecture

Three cleanly separated components: (1) hooks-driven extractor running silently in background, (2) MCP server for mid-session search/retrieval, (3) CLI for human management. All share a single JSON store at `.memory/state.json`.

Written in TypeScript. ~9 source files. Calls Anthropic's Haiku API directly via `fetch()` for extraction, consolidation, and question-answering. Estimated daily cost: $0.05-0.10.

MCP tools: `memory_search`, `memory_related`, `memory_ask` (RAG-style LLM synthesis over top 30 matches), `memory_save`, `memory_recall`, `memory_delete`, `memory_consolidate`, `memory_consciousness` (generates the CLAUDE.md content), `memory_stats`, `memory_init`.

### Two-Tier System

The core innovation. Tier 1 is an auto-generated `CLAUDE.md` file with `<!-- MEMORY:START -->` / `<!-- MEMORY:END -->` markers. This is read automatically by Claude Code on session start -- no MCP call needed, zero latency. Line-budgeted to ~150 lines across 6 sections with per-type allocations (architecture: 25, decision: 25, pattern: 25, gotcha: 20, progress: 30, context: 15). Unused budget from empty sections redistributes to over-budget ones.

Tier 2 is `.memory/state.json` -- unlimited storage, searchable via MCP tools mid-conversation. The claim is that "80% of sessions need only Tier 1" (i.e., CLAUDE.md is sufficient and no MCP calls are needed).

**Consciousness generation** is the bridge: a `generateConsciousness()` method in `store.ts` that:
1. Filters active memories with confidence > 0.3
2. Groups by type
3. Sorts each group by `confidence * (1 + accessCount/10)` (importance score)
4. Renders within per-type line budgets
5. Truncates individual lines to 120 chars
6. Adds overflow indicator pointing to MCP tools

The CLAUDE.md is synced after every extraction, consolidation, and manual save. Existing non-memory content in CLAUDE.md is preserved outside the markers.

### Hooks Integration

Uses three Claude Code hook events:
- **Stop** (after each Claude response): Normal extraction with minimum 3 transcript lines
- **PreCompact** (before context compaction): Lower threshold (1 line) because context is about to be lost
- **SessionEnd**: Extract + always consolidate regardless of thresholds

Hook input is JSON via stdin: `{session_id, transcript_path, cwd, hook_event_name}`. The extractor reads the session JSONL transcript from a cursor position (tracking what's already been processed), summarizes it into human-readable format (USER/CLAUDE/TOOL lines), chunks at 6000 chars with 500-char overlap, and sends each chunk to Haiku.

**Extraction prompt** is context-aware: it includes all existing active memories and instructs Haiku to extract "only NEW or UPDATED memories." The prompt asks for 0-3 memories per extraction typically. Returns JSON array with optional `supersedes_content` field for updates.

**Concurrency safety**: PID-based lock file in `.memory/lock`, atomic writes via `.tmp` + rename, per-session cursors.

### Memory Lifecycle

Six types: architecture, decision, pattern, gotcha, progress, context.

Dedup: Jaccard similarity on tokenized content (stop words removed). Threshold > 0.6 = auto-supersede the older memory.

Confidence decay: Progress memories decay linearly to 0 over 7 days. Context memories decay over 30 days. Architecture/decision/pattern/gotcha are permanent (no decay). Memories below 0.3 confidence are hidden from CLAUDE.md but remain searchable.

Consolidation triggered when: extraction count divisible by 10, or > 80 active memories, or SessionEnd. Groups memories by type, sends groups with 5+ memories to Haiku asking for keep/merge/drop decisions. Archived memories pruned after 14 days.

**Git snapshots**: Optional feature that commits entire project to a hidden `__memory-snapshots` branch after each extraction. Optional push to remote. Provides rollback capability tied to memory extraction events.

### Worth Stealing

1. **Two-tier with line budgets**: The CLAUDE.md generation with per-type line budgets and importance-based sorting is a well-engineered approach to the "what fits in context" problem. The surplus redistribution across types is a nice touch.
2. **Context-aware extraction prompts**: Sending existing memories to the extractor with "only extract NEW or UPDATES" reduces duplication at the source.
3. **Differential hook behavior**: PreCompact getting a lower threshold because "context is about to be lost" shows understanding of the hook lifecycle.
4. **Per-session cursors**: Tracking extraction position per session avoids re-processing already-extracted content.
5. **Supersedes chain**: Both Jaccard-based auto-superseding and explicit supersedes_content matching provide clean memory evolution.

---

## 2. claude-mem (thedotmack)

### Architecture

Not an MCP server -- it's a **Claude Code plugin** (installed via `/plugin marketplace add`). This is a significant architectural distinction: plugins get deeper lifecycle integration than MCP servers, including hook access and context injection.

Written in TypeScript, built to ESM. Components: hooks layer (TypeScript CLI), worker service (Express HTTP API on port 37777 managed by Bun), SQLite database, Chroma vector DB for semantic search, web viewer UI (React at localhost:37777), and an MCP server providing 3 search tools.

The worker service runs as a persistent background process (Bun-managed). Hooks fire and POST to `http://127.0.0.1:{port}/api/sessions/...` -- the worker handles all AI processing asynchronously. This decouples hook execution speed from AI processing time.

**Mode system**: Configurable observation profiles loaded from JSON files in `plugin/modes/`. Default mode is "code" (software development). Supports inheritance via `parent--override` pattern (e.g., `code--ko` for Korean language). Modes define observation types, concepts, and all prompt templates.

### Compression Pipeline

The AI compression uses the **Claude Agent SDK** -- it spawns a separate Claude subprocess that acts as an "observer" of the primary session. This is a unique architectural choice:

1. **PostToolUse hook** fires after each tool operation, sending tool name, input, output, and cwd to the worker
2. Worker enqueues the observation for the SDK agent
3. SDK agent receives the observation as XML (`<observed_from_primary_session>`) and produces structured XML observations:
   ```xml
   <observation>
     <type>bugfix | discovery | implementation | decision | ...</type>
     <title>Short title</title>
     <subtitle>One sentence</subtitle>
     <facts><fact>...</fact></facts>
     <narrative>Full context</narrative>
     <concepts><concept>...</concept></concepts>
     <files_read><file>...</file></files_read>
     <files_modified><file>...</file></files_modified>
   </observation>
   ```
4. Observations are parsed by XML regex, validated against mode's valid types, stored in SQLite with FTS5 for full-text search and Chroma for semantic search.

The agent processes observations as they arrive via an async queue (`SessionQueueProcessor`) with 3-minute idle timeout that kills the subprocess.

### Endless Mode

Described as "biomimetic memory architecture for extended sessions." Available as a beta feature. The key problem it addresses: during long sessions, tool outputs consume massive context. Endless Mode compresses tool outputs to ~500 tokens in real-time, preventing context window exhaustion.

Based on the mode system architecture, Endless Mode would be a different mode profile with different observation types and prompts optimized for continuous operation rather than session-bounded capture.

### Plugin vs MCP

Key differences from MCP-server-based approaches:
- **5 lifecycle hooks**: SessionStart (inject context), UserPromptSubmit (initialize session), PostToolUse (capture observation), Stop (generate summary), SessionEnd (complete session)
- **Context injection**: SessionStart hook returns structured context that gets injected into the conversation. This is fundamentally different from CLAUDE.md reading -- it's programmatic injection with configurable what/how.
- **Progressive disclosure**: Context injected at session start is a compact timeline with observation IDs. Full details are fetched on-demand via MCP search tools. The 3-layer workflow (search -> timeline -> get_observations) claims ~10x token savings.
- **Token economics**: Tracks "discovery tokens" (how many tokens the original tool output was) vs "read tokens" (how many the compressed observation costs). Shows compression savings.

### Context Injection

The `ContextBuilder` service generates context injected at session start:
- Queries observations from SQLite filtered by type and concept
- Queries recent session summaries
- Builds a chronological timeline mixing observations and summaries
- Renders as markdown with configurable sections (header, summaries, timeline, footer)
- Most recent N observations get full details; older ones get compact index entries (ID, time, type, title, read token count)
- Supports worktree awareness (shows observations from both parent repo and worktree)

### Worth Stealing

1. **Worker service decoupling**: Hooks POST to a local HTTP API rather than doing AI work inline. This keeps hooks fast and resilient.
2. **Agent SDK as observer**: Using a separate Claude subprocess to observe and compress the primary session is clever. The primary session stays fast; compression happens asynchronously.
3. **Progressive disclosure with token accounting**: The 3-layer search workflow (index -> timeline -> full details) with explicit token cost per observation is a strong pattern for cost-conscious retrieval.
4. **Privacy tags**: `<private>content</private>` strips content before it reaches the worker/database. Simple user-level control.
5. **Mode inheritance**: `code--ko` pattern allows language/domain customization without duplicating entire mode configs.
6. **Session queue with idle timeout**: The 3-minute idle timeout that kills the SDK subprocess prevents resource waste.

---

## 3. mcp-memory-service (doobidoo)

### Architecture

The most feature-rich implementation by far. Python-based, v10.22.0, 968+ tests. Two servers: MCP server (for Claude Desktop/Code integration) and FastAPI HTTP server (dashboard + REST API + OAuth 2.1).

Storage backends via Strategy Pattern:
- **SQLite-Vec**: Local, 5ms reads, uses sqlite-vec extension for KNN semantic search
- **Cloudflare**: D1 (SQL) + Vectorize (vector index), cloud-only
- **Hybrid** (recommended): Local SQLite-Vec for reads, background Cloudflare sync

Embeddings: ONNX model (sentence-transformers/all-MiniLM-L6-v2) for lightweight local vector generation. Also supports external embedding APIs (vLLM, Ollama, TEI, OpenAI-compatible).

Quality scoring: Multi-tier (local ONNX at 80-150ms / Groq Llama 3 / Gemini 1.5 Flash). Quality scores (0.0-1.0) affect retention policies and search ranking.

12 unified MCP tools (consolidated from 34 in v10.0.0). Document ingestion for PDF, DOCX, TXT, JSON, CSV with intelligent chunking.

### Dream Consolidation

The "dream-inspired consolidation" system (`DreamInspiredConsolidator`) is a 6-phase pipeline:

1. **Relevance scoring** (ExponentialDecayCalculator): `score = base_importance * decay_factor * connection_boost * access_boost * quality_multiplier`. Decay is exponential: `exp(-age_days / retention_period)`. Connection boost: +10% per connection. Access boost: 1.5x if accessed in last day, 1.2x for last week. Quality multiplier: 1.0 to 1.5x based on quality score. Association-based quality boost for well-connected memories.

2. **Semantic clustering** (SemanticClusteringEngine): Groups similar memories. Only runs weekly/monthly/quarterly.

3. **Creative association discovery** (CreativeAssociationEngine): Finds semantic relationships between memories. Stores in graph table with inferred relationship types. Only runs weekly/monthly.

4. **Semantic compression** (SemanticCompressionEngine): Compresses memory clusters. Only runs weekly/monthly/quarterly.

5. **Controlled forgetting** (ControlledForgettingEngine): Quality-based retention policies. High quality (>=0.7): keep up to 365 days inactive. Medium (0.5-0.7): 180 days. Low (<0.5): 30-90 days scaled by quality. Archives to `~/.mcp_memory_archive/` filesystem with JSON backups. Recovery is possible. Protected memories (tagged critical/important) maintain minimum 50% relevance.

6. **Relationship inference** (RelationshipInferenceEngine): Multi-factor analysis classifying relationships as causes, fixes, contradicts, supports, follows, related. Confidence-scored with 0.6 threshold.

**Scheduling**: APScheduler with cron triggers. Phases are horizon-gated:
- Clustering: weekly, monthly, quarterly
- Associations: weekly, monthly
- Compression: weekly, monthly, quarterly
- Forgetting: monthly, quarterly, yearly

Incremental mode: processes oldest-first in configurable batch sizes, tracking `last_consolidated_at` per memory.

### Visualization

The D3.js visualization appears to be in a separate `video/` directory with React components (`VectorSpace3D.tsx`, `Speedometer.tsx`) -- looks like it's a demo/promotional visualization rather than an integrated dashboard feature.

The actual dashboard is a FastAPI-served single-page app (`web/static/index.html`) with REST API endpoints mirroring MCP tools. SSE (Server-Sent Events) for real-time updates.

### Hook Patterns

Extensive hook system in `claude-hooks/`:
- **auto-capture-hook.js** (PostToolUse on Edit/Write/Bash): Reads transcript, runs regex pattern detection for decisions/errors/learnings/implementations, stores via HTTP API. Patterns include confidence scores and minimum content lengths.
- **session-start.js**: Context retrieval and injection
- **session-end.js**: Session wrap-up
- **mid-conversation.js**: Mid-session context updates
- **topic-change.js**: Detects conversation topic shifts
- **permission-request.js**: Handles permission-related context

Utilities: adaptive pattern detector, conversation analyzer, context shift detector, git analyzer, memory scorer, project detector, session tracker, tiered conversation monitor, user override detector (`#remember` / `#skip` markers).

The hook system talks to the server via HTTP API (not MCP), achieving "90% token reduction vs MCP tools."

### Worth Stealing

1. **Multi-factor relevance scoring**: The combination of exponential decay, connection boost, access recency, quality multiplier, and association boost is the most sophisticated relevance model across all implementations reviewed.
2. **Quality-gated retention**: High-quality memories surviving 365 days while low-quality memories get archived after 30-90 days is more nuanced than binary keep/drop.
3. **Horizon-gated consolidation phases**: Not running expensive operations (clustering, compression) on daily cycles makes practical sense. The phase-to-horizon mapping is well-designed.
4. **Controlled forgetting with recovery**: Archiving to filesystem with JSON backups before deletion, plus a `recover_memory()` method, provides a safety net other implementations lack.
5. **Relationship inference engine**: Automatic classification of association types (causes, fixes, contradicts, supports) adds semantic meaning to connections beyond just "related."
6. **Hybrid storage**: Local SQLite-Vec for fast reads + background Cloudflare sync for cloud backup is a pragmatic production pattern.
7. **Pattern-based auto-capture**: Regex patterns for decisions/errors/learnings with confidence scores, combined with `#remember`/`#skip` user overrides, balance automation with user control.

---

## 4. memory-graph

### Architecture

Graph-database-first approach using multiple backend options: Neo4j, FalkorDB, FalkorDB Lite, Memgraph, SQLite fallback, Turso, and a cloud backend. Written in Python with Pydantic models.

Distinguishing characteristic: explicit graph relationships are first-class citizens, not an afterthought. While other implementations store memories as flat records with optional tags/associations, memory-graph stores memories as nodes with typed, directed edges.

MCP tools are designed for any MCP-compliant client (Claude Code, Claude Desktop, ChatGPT Desktop, Cursor, Windsurf, VS Code Copilot, Continue.dev, Cline, Gemini CLI).

Has a separate SDK package (`memorygraphsdk`) for programmatic access.

### Auto-Trigger Patterns

Memory-graph takes the approach of **prompting the agent to store memories** via CLAUDE.md/AGENTS.md instructions rather than implementing hook-based automatic capture. The recommended CLAUDE.md configuration instructs the agent to:

- Use `recall_memories` BEFORE any task
- Automatically store on: git commits, bug fixes, version releases, architecture decisions, pattern discoveries
- Timing modes: `immediate | on-commit | session-end`

This is fundamentally different from the hook-based approaches of the other three implementations. It's simpler (no hook infrastructure) but depends entirely on the agent following instructions consistently.

The `integration/context_capture.py` module provides functions for programmatic capture:
- `capture_task_context()`: Stores task with goals, files, project links
- `capture_command_execution()`: Stores commands with outputs, links to tasks
- `analyze_error_patterns()`: Identifies recurring errors, updates frequency counts
- `track_solution_effectiveness()`: Tracks which solutions worked for which errors

All functions automatically sanitize sensitive data (API keys, tokens, passwords, credentials, emails) via regex patterns before storage.

### Graph Schema

**Memory model** (Pydantic): 13 memory types (task, code_pattern, problem, solution, project, technology, error, fix, command, file_context, workflow, general, conversation). Each memory has title, content, summary, tags, importance (0-1), confidence (0-1), effectiveness (0-1), usage_count, version (optimistic concurrency), and rich context (project_path, files_involved, languages, frameworks, git info, session_id).

**Relationship model**: 7 categories with 33 specific relationship types:
- **Causal**: CAUSES, TRIGGERS, LEADS_TO, PREVENTS, BREAKS
- **Solution**: SOLVES, ADDRESSES, ALTERNATIVE_TO, IMPROVES, REPLACES
- **Context**: OCCURS_IN, APPLIES_TO, WORKS_WITH, REQUIRES, USED_IN
- **Learning**: BUILDS_ON, CONTRADICTS, CONFIRMS, GENERALIZES, SPECIALIZES
- **Similarity**: SIMILAR_TO, VARIANT_OF, RELATED_TO, ANALOGY_TO, OPPOSITE_OF
- **Workflow**: FOLLOWS, DEPENDS_ON, ENABLES, BLOCKS, PARALLEL_TO
- **Quality**: EFFECTIVE_FOR, INEFFECTIVE_FOR, PREFERRED_OVER, DEPRECATED_BY, VALIDATED_BY

Relationships have properties: strength (0-1), confidence (0-1), evidence_count, success_rate, validation_count, counter_evidence_count, and **bi-temporal tracking** (valid_from, valid_until, recorded_at, invalidated_by).

**Multi-tenancy support**: Optional tenant_id, team_id, visibility (private/project/team/public), created_by fields on memory context.

### Session Briefing

`proactive/session_briefing.py` generates automatic session briefings:
- Recent activities (last N days)
- Unresolved problems (problem memories without SOLVES/ADDRESSES relationships)
- Relevant patterns (code_pattern memories sorted by effectiveness)
- Deprecation warnings (memories with DEPRECATED_BY relationships)

Output is structured Pydantic model with `format_as_text(verbosity)` method (minimal/standard/detailed).

### Intelligence Layer

`intelligence/pattern_recognition.py`: PatternRecognizer finds similar problems via keyword matching in graph queries, with attached solutions.

`intelligence/entity_extraction.py`: Extracts entities from memory content.

`intelligence/temporal.py`: Temporal analysis of memory patterns.

`intelligence/context_retrieval.py`: Retrieves relevant context for current work.

### Worth Stealing

1. **Typed relationships with bi-temporal tracking**: 33 relationship types across 7 categories is the richest relationship schema reviewed. The bi-temporal fields (valid_from/valid_until + recorded_at) enable "when did we learn this?" vs "when was it true?" queries.
2. **Causal chain traversal**: The ability to query `[timeout_fix] --CAUSES--> [memory_leak] --SOLVED_BY--> [connection_pooling]` as a chain is genuinely useful for understanding problem evolution.
3. **Deprecation warnings**: Using DEPRECATED_BY relationships to surface warnings at session start is a practical application of graph structure.
4. **Solution effectiveness tracking**: Recording whether solutions actually worked (via SOLVES vs ATTEMPTED_SOLUTION relationships with success_rate) provides empirical data on what fixes what.
5. **Unresolved problem surfacing**: Querying for problem nodes without SOLVES edges to surface active issues at session start is simple but effective.
6. **Content sanitization**: Regex-based scrubbing of API keys, tokens, passwords, and credentials before storage is a security measure the other implementations lack.

---

## 5. Cross-Cutting Patterns

### Hook-Driven Auto-Capture
Three of four implementations (memory-mcp, claude-mem, mcp-memory-service) use Claude Code hooks for automatic capture. memory-graph instead relies on agent instructions. The hook-based approaches differ in their transport:
- memory-mcp: Hooks call extractor.js directly, which calls Haiku API
- claude-mem: Hooks POST to local HTTP worker service
- mcp-memory-service: Hooks POST to local HTTP API

All three parse the session transcript (JSONL) to extract content.

### LLM-Powered Extraction
memory-mcp and claude-mem both use LLMs for compression/extraction. memory-mcp uses Haiku directly; claude-mem spawns a full Claude Agent SDK subprocess as an "observer." mcp-memory-service uses regex pattern detection for auto-capture (no LLM needed) but uses LLMs for quality scoring.

### Consolidation
Three implementations have consolidation: memory-mcp (Haiku-based merge/drop), mcp-memory-service (6-phase dream pipeline with scheduling), and memory-graph (inherent in graph structure via REPLACES/DEPRECATED_BY relationships). claude-mem doesn't have explicit consolidation -- its compression is real-time rather than batch.

### Confidence/Relevance Decay
memory-mcp: Simple linear decay for progress (7 days) and context (30 days) types only.
mcp-memory-service: Exponential decay with multi-factor scoring (connections, access, quality).
memory-graph: No automatic decay, but importance/confidence/effectiveness scores updated via relationships.

### Deduplication
memory-mcp: Jaccard similarity on tokenized content (>0.6 threshold).
mcp-memory-service: Content hash + word overlap detection (>0.8 Jaccard).
claude-mem: Content hash at storage layer.
memory-graph: Relies on graph structure; SIMILAR_TO/VARIANT_OF relationships rather than dedup.

### Context Window Management
memory-mcp: Two-tier (auto-read CLAUDE.md + searchable JSON).
claude-mem: Progressive disclosure (compact timeline at session start + on-demand full details via MCP).
mcp-memory-service: Hook-injected context at session start + MCP tools for search.
memory-graph: Agent prompted to `recall_memories` before tasks.

### Privacy Controls
claude-mem: `<private>` tags strip content before storage.
memory-graph: Content sanitization (regex scrub of secrets/PII).
Others: No explicit privacy controls.

---

## 6. Comparison to claude-memory

Our implementation (`mcp__claude-memory`) compared to features from these four:

### What We Already Have (Parity or Better)
- **MCP-based store/recall/search/forget**: Standard across all implementations
- **Embedding-based semantic search**: We use this; memory-mcp uses keyword/Jaccard only
- **Consolidation**: We have it; memory-mcp and mcp-memory-service also have it
- **Session-based operations** (sleep/reflect/startup): Our sleep cycle is analogous to mcp-memory-service's dream consolidation but more explicit in its session lifecycle

### What We're Missing (Potentially Valuable)

1. **Auto-generated CLAUDE.md tier** (from memory-mcp): Our system relies entirely on MCP tool calls. A line-budgeted CLAUDE.md that's auto-read on session start would eliminate the cold-start problem where the agent needs to call recall before having context. The importance-sorted, per-type budgeted approach is well-tested.

2. **Hook-driven automatic capture** (from all three hook-using implementations): We require explicit `remember` calls. Auto-capture via PostToolUse/Stop/SessionEnd hooks would capture knowledge that the agent doesn't think to store. This is the biggest gap.

3. **Progressive disclosure** (from claude-mem): Our `recall` returns all matches. A compact-index-first approach that returns IDs/titles/token-costs before full content would reduce token waste on irrelevant results.

4. **Typed relationships** (from memory-graph): Our memories are flat with tags. Explicit CAUSES/SOLVES/CONTRADICTS relationships between memories would enable causal chain queries ("what happened after we changed X?") and unresolved-problem surfacing.

5. **Quality-based retention** (from mcp-memory-service): Our forgetting is binary. Tiered retention (365d for high quality, 30d for low) would better manage long-term memory health.

6. **Content sanitization** (from memory-graph): We don't scrub secrets/PII before storage. Since memory persists across sessions, stored API keys or passwords could be a security risk.

7. **Session briefing** (from memory-graph): Automatic generation of "here's what's unresolved, here's recent activity, here are relevant patterns" at session start. Our `startup_load` provides recall but not structured briefing.

### What We Have That Others Don't

1. **Explicit reflect/sleep cycle**: Our session lifecycle (reflect during, sleep at end) with distinct consolidation phases is more cognitively structured than other implementations' batch consolidation.
2. **Recall feedback**: Our ability to mark recalled memories as helpful/unhelpful for relevance tuning.
3. **Memory stats and review_pending**: Administrative introspection tools.
4. **Re-embedding**: Ability to re-embed all memories when embedding model changes.
