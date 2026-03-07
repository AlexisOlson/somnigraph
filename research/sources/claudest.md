# Claudest (claude-memory plugin) -- Source Analysis

*Phase 13, 2026-02-27. Analysis of gupsammy/Claudest.*

## 1. Architecture Overview

**Language:** Python (stdlib-only; no external dependencies at runtime)
**Key dependencies:**
- SQLite with FTS5 (or FTS4 fallback, or LIKE fallback)
- Python 3.7+ standard library only
- pytest + hypothesis (dev only)

**Storage layout:**
```
~/.claude-memory/
  conversations.db          # SQLite WAL-mode, v3 schema
```

**File structure (claude-memory plugin):**
```
plugins/claude-memory/
  .claude-plugin/plugin.json   # Plugin metadata (v0.7.7)
  hooks/
    memory-setup.py            # SessionStart: creates dir, background import
    memory-context.py          # SessionStart: queries + injects recent sessions
    memory-sync.py             # Stop: async sync current session to DB
    import_conversations.py    # Bulk import from JSONL conversation files
    sync_current.py            # Incremental session sync
  commands/                    # Slash commands
  skills/
    recall-conversations/
      SKILL.md                 # Activation triggers + search strategy
      scripts/
        recent_chats.py        # Retrieve N most recent sessions
        search_conversations.py # FTS5/BM25 keyword search
        memory_lib/             # Shared utility package
          db.py                 # Schema, connection, settings, FTS detection
          content.py            # Message text extraction, tool detection
          parsing.py            # JSONL parsing, branch detection via UUID chains
          formatting.py         # Session formatting, time/path utilities
    extract-learnings/
      SKILL.md                 # Learning extraction + placement workflow
```

## 2. Memory Type Implementation

**Schema:** The core memory unit is a *session branch* -- not an individual memory object. The system stores raw conversation messages and retrieves at the branch/session level.

**Tables (v3 schema):**
- `projects` -- directory-derived metadata with unique path
- `sessions` -- one row per conversation UUID, parent relationships, git context
- `branches` -- conversation forks per session, with `leaf_uuid` and `fork_point_uuid`
- `messages` -- all utterances stored once (deduped by UUID), role-validated
- `branch_messages` -- many-to-many junction for branch-to-message mapping
- `import_log` -- file hash tracking for idempotent import
- `messages_fts` / `branches_fts` -- FTS virtual tables with auto-sync triggers

**Types/categories:** No explicit memory types. The system operates at the session/branch level. The `extract-learnings` skill layers a classification on top:
- Layer 0: `~/.claude/CLAUDE.md` (universal behavioral preferences)
- Layer 1: `<repo>/CLAUDE.md` (project-specific knowledge)
- Layer 2: `MEMORY.md` (concise working notes)
- Layer 3: `memory/*.md` (detailed reference material)
- Meta: suggestions for new workflow automations

**Extraction:** Two modes:
1. **Automatic** -- sessions sync to SQLite on Stop hook (async, non-blocking). Messages stored with deduplication.
2. **Manual curation** -- `extract-learnings` skill runs a 4-stage workflow (Gather, Analyze, Propose, Execute) requiring human approval before writing to any layer.

## 3. Retrieval Mechanism

**Recall pipeline:**

1. **Context injection (SessionStart):** `memory-context.py` runs `select_sessions()`:
   - Iterates recent sessions for current project (excludes current session and subagents)
   - Skips sessions with <=1 exchange
   - Collects 2-exchange sessions, stops at first session with >2 exchanges
   - Batch-loads messages in single optimized query
   - Formats as markdown: timeline, modified files (last 10), git commits, user-assistant exchanges

2. **On-demand search (`recall-conversations`):**
   - `search_conversations.py` constructs FTS5 queries
   - Term sanitization: removes quotes, parens, asterisks, FTS keywords (NEAR, AND, OR, NOT)
   - Query format: `"term1" OR "term2" OR "term3"` (quoted terms joined with OR)
   - Ranked by `ORDER BY bm25(branches_fts)`
   - Searches at branch level (aggregated content), not individual messages
   - Configurable result limit (1-10, default 5)
   - Output: markdown (token-efficient) or JSON

3. **FTS cascade:**
   - FTS5 with BM25 ranking (preferred)
   - FTS4 with MATCH + snippet, no BM25 (fallback)
   - LIKE pattern matching on `aggregated_content` (last resort)
   - Detected at startup via `PRAGMA compile_options`

**No vector search, no embeddings, no fusion.** This is the defining architectural choice.

## 4. Standout Feature

**FTS5-only philosophy and the anti-vector argument.**

Claudest is the only system in the survey that explicitly rejects vector/embedding-based retrieval as unnecessary complexity. The argument: "No vector database, no embedding pipeline, no external dependencies. Just SQLite and Python's standard library."

The bet is that for agent-constructed queries, FTS5's BM25 ranking provides sufficient signal. The agent itself constructs discriminative keyword queries (the skill instructions coach: "content-bearing words that discriminate between sessions" and "avoid generic terms like 'discuss' or 'chat'"). This offloads the semantic matching problem to the LLM's query formulation rather than embedding infrastructure.

**What this gains:**
- Zero external dependencies (no API keys, no embedding models, no vector stores)
- Deterministic, inspectable ranking (BM25 scores are reproducible)
- Cross-platform portability (Python stdlib + SQLite ships everywhere)
- No cold-start latency from model loading
- Trivial deployment (install plugin, done)

**What this loses:**
- No semantic similarity -- "authentication" won't match "login" unless both appear in the same branch
- No fuzzy matching for concept-level retrieval
- Relies entirely on the agent formulating good keyword queries
- No multi-angle retrieval (only one search axis: text match)

**Hook integration** is the complementary design choice -- memory is injected transparently. The user doesn't invoke recall; the system pushes relevant context on session start and syncs on session end. This makes the whole system feel invisible.

## 5. Other Notable Features

1. **Branch-aware storage.** Conversation rewinds (where Claude Code lets you fork a conversation) are tracked via a many-to-many `branch_messages` table rather than duplicating messages. Search operates at the branch level with `aggregated_content`, ensuring multi-word queries match within coherent conversation paths.

2. **Extract-learnings as a knowledge ladder.** The 4-layer placement system (CLAUDE.md global -> repo CLAUDE.md -> MEMORY.md -> topic files) with human approval gates creates a quality-filtered path from raw conversation to persistent knowledge. Valid learnings include "discovered commands, non-obvious gotchas, architectural decisions, behavioral corrections, configuration quirks" -- explicit exclusion of "information readable from code, generic best practices, one-off fixes."

3. **Noise filtering (v0.7.1).** Removes teammate coordination messages and `prompt_suggestion` subagent noise from both context injection and search results. This is a pragmatic response to context pollution in real-world team usage.

4. **Security hardening (v0.7.0+).** TOCTOU prevention via `tempfile.mkstemp()` with 0o600 permissions, path traversal blocking via UUID validation and `resolve().relative_to()`, FTS injection prevention through term sanitization. More security attention than most systems in the survey.

5. **Session selection algorithm.** The heuristic of "collect short sessions, stop at first substantive one" is clever for context injection -- it avoids wasting context window on trivial sessions while ensuring the most recent meaningful work is available.

## 6. Gap Ratings

| Gap | Rating | Evidence |
|-----|--------|----------|
| Layered Memory | 40% | Extract-learnings has a 4-tier knowledge hierarchy (CLAUDE.md/MEMORY.md/topic files), but the core memory is flat session storage. No episodic/semantic/procedural distinction. |
| Multi-Angle Retrieval | 15% | Single-axis: FTS5 keyword match only. No vector, no tag-based, no temporal, no graph traversal. Agent query formulation is the only "angle." |
| Contradiction Detection | 0% | No mechanism to detect conflicting information across sessions or learnings. |
| Relationship Edges | 0% | No relationships between memories/sessions. Each branch is independent. |
| Sleep Process | 0% | No background consolidation, no decay, no maintenance cycles. |
| Reference Index | 35% | Extract-learnings creates a de facto reference index in MEMORY.md and topic files, but it's manual and unstructured. No automatic indexing. |
| Temporal Trajectories | 10% | Sessions are timestamped and ordered, but no trajectory tracking, no evolution of concepts over time. |
| Confidence/UQ | 0% | No confidence scoring, no uncertainty quantification, no feedback loop. |

## 7. Comparison with claude-memory

**Stronger:**
- Zero dependencies -- no API keys, no embedding model, no Python packages beyond stdlib. Our system requires OpenAI API for embeddings, sqlite-vec extension, and several pip packages.
- Cross-platform portability -- runs anywhere Python + SQLite exist without configuration.
- Hook-based transparency -- memory injection is completely automatic. Our system requires explicit `startup_load()` and `recall()` calls.
- Branch-aware conversation storage -- tracks conversation forks/rewinds, which we don't model.
- Security hardening -- more deliberate about path traversal, TOCTOU, injection prevention.
- Plugin ecosystem -- memory is one module in a cohesive 8-plugin system (research, coding, thinking, etc.).

**Weaker:**
- No semantic search at all -- cannot find conceptually related content that doesn't share keywords. Our hybrid RRF (vector + FTS5) handles this.
- No memory lifecycle -- no decay, no confidence, no feedback loops, no consolidation. Memories are forever or manually curated.
- No relationship graph -- sessions are isolated islands. Our edge system (with contradiction/revision/derivation flags) enables graph traversal.
- No structured memory types -- everything is a session branch. Our episodic/procedural/semantic/reflection categories enable type-appropriate retrieval.
- No background processing -- no sleep cycles, no automated maintenance. Our NREM/REM pipeline handles consolidation, edge creation, and taxonomy.
- No dimensional feedback -- no way to tell the system which retrievals were useful. Our `recall_feedback()` with confidence gradient tunes future retrieval.
- Retrieval is all-or-nothing -- either the FTS match fires or it doesn't. No scoring fusion, no graceful degradation across retrieval methods.

## 8. Insights Worth Stealing

1. **Session-level search granularity** (effort: low, impact: medium). Searching at the branch/session level rather than individual memory level means multi-word queries are more likely to match coherent contexts. We could add a session-level view to recall that groups related memories by their originating session, providing conversation-coherent context when needed.

2. **Agent-coached query formulation** (effort: low, impact: medium). The SKILL.md for recall-conversations explicitly coaches the agent on what makes a good search query ("content-bearing words that discriminate between sessions"). We could add similar guidance to our recall prompts -- our agents sometimes construct poor queries because they don't know what FTS5 responds well to.

3. **Noise filtering for team contexts** (effort: low, impact: low-medium). Filtering out subagent noise and coordination messages before storage. If we ever encounter context pollution from similar sources, this is a known solution.

4. **Extract-learnings approval gates** (effort: medium, impact: medium). The 4-stage Gather/Analyze/Propose/Execute workflow with explicit human approval before writing to persistent knowledge layers. Our `review_pending()` is similar in spirit but operates at a different granularity (individual memories vs. proposed file edits). The placement decision tree (which layer should this go to?) is worth studying.

## 9. What's Not Worth It

- **Adopting FTS5-only retrieval.** ~~We already have FTS5 as one axis of our RRF hybrid. Dropping vectors would sacrifice semantic matching that our system demonstrably benefits from (e.g., matching "authentication" to "login flow" discussions).~~ **Phase 14 correction (2026-03-01):** This claim was wrong. Experiment 1 showed FTS-only MRR (0.1609) = Vec-only MRR (0.1582) = Fused MRR (0.1607). Vector search adds 0 marginal value over FTS5-only at our scale (~300 memories with curated summary+themes). The "authentication"/"login flow" example was hypothetical, not measured. Claudest's core argument is validated for our use case. We retain vector infrastructure for other purposes (sleep similarity, deduplication) but its retrieval contribution is nil. See [[retrospective-experiments#Experiment 1 Vector Search Marginal Value]].
- **Branch-level storage model.** Our individual-memory granularity is better for the operations we need (decay, confidence, edges, feedback). Session-level storage makes these impossible.
- **Plugin marketplace architecture.** Interesting for distribution but orthogonal to memory quality. Our MCP server approach is more tightly integrated.

## 10. Key Takeaway

Claudest's claude-memory plugin is the strongest argument in the survey for FTS5-only retrieval, and it makes that argument through engineering discipline rather than just ideology: zero dependencies, cross-platform stdlib-only Python, branch-aware deduplication, and security hardening that most memory systems neglect. The anti-vector position is defensible for its use case (agent-constructed keyword queries against recent conversation history) but fundamentally limits the system to lexical matching. The real insight is that the quality of the agent's query formulation matters more than most systems acknowledge -- coaching the agent to construct discriminative queries is cheap and effective regardless of whether you also use vectors. The extract-learnings pipeline, with its layered knowledge hierarchy and human approval gates, is a thoughtful approach to the "raw sessions vs. curated knowledge" problem that complements automated extraction systems like ours.

---

## Phase 14 Experimental Validation (2026-03-01)

Phase 14 ran retrospective experiments against our live data (~300 memories, 124 queries with feedback, 1,101 feedback events). The results directly test Claudest's core claim.

### Experiment 1: Vector Search Marginal Value

| Configuration | MRR |
|---------------|-----|
| FTS-only | 0.1609 |
| Vec-only | 0.1582 |
| RRF fused (current) | 0.1607 |

**Result:** FTS-only MRR matches hybrid RRF. Vector search adds zero marginal retrieval quality at our scale. Claudest's core argument — that FTS5 is sufficient when the LLM constructs keyword queries and the stored text is well-curated — is confirmed for our use case.

**Why it works for us:** Our summary and themes fields are human-curated with discriminative keywords. Our recall instructions coach the agent on query formulation. With curated metadata and coached queries, keyword matching captures the same information as semantic similarity.

**Qualification:** This result is specific to ~300 curated memories. At larger scales or with less human curation, vector search may become necessary. We retain the infrastructure.

### Correction to §9

The original §9 stated: "Adopting FTS5-only retrieval [is not worth it]... our system demonstrably benefits from [vectors]." This was stated without evidence and is experimentally refuted. See corrected text in §9 above.

### See Also

- [[retrospective-experiments]] — Full experimental methodology and results
- [[agent-output-locomo]] — LoCoMo benchmark independently showing simple retrieval beating complex approaches
