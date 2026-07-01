# LangMem — LangChain's LLM-prompted memory + prompt-optimization library for LangGraph agents

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

LangMem is a Python library (`pip install langmem`, MIT, ~1.5k stars) from LangChain. It is **not** a storage engine, not a server, not a Claude Code integration. It is a set of LangGraph-composable primitives that (a) prompt an LLM to extract/consolidate memories into LangGraph's `BaseStore`, and (b) optimize an agent's system prompt from feedback. All persistence, indexing, and search are delegated to `langgraph.store.BaseStore` (`InMemoryStore` for dev, `AsyncPostgresStore` for prod).

### Storage & Schema
No custom storage. Memories are LangGraph `Item`s: `namespace` (tuple), `key` (UUID), `value` (`{"kind": ..., "content": ...}`), `created_at`, `updated_at`, `score`. The base memory schema (`extraction.py:86 Memory`) is a single field: `content: str`. Users can supply a Pydantic `schema=` (e.g. a `UserProfile` "Profile" memory) for structured extraction, but the default is flat text. No themes, priority, decay_rate, valid_from/until, layers, source attribution, or trust — none of Somnigraph's schema richness. `namespace` provides multi-user/multi-scope partitioning (e.g. `("memories", "{langgraph_user_id}")`).

### Memory Types
Conceptually three (docs `conceptual_guide.md`): **semantic** (facts — as "Collection" of records or a single "Profile"), **episodic** (experiences), **procedural** (system prompts, handled by the prompt optimizer, not the store). In code these are just different Pydantic `schemas` passed to the manager plus different namespaces; there is no type-specific storage or retrieval behavior.

### Write Path
Two modes:
- **Hot path** (`knowledge/tools.py`): `create_manage_memory_tool` gives the agent a CRUD tool — `manage_memory(content, action=create|update|delete, id)`. It is a **thin wrapper over `store.put/delete`** (lines 271-337). Zero dedup, zero enrichment, zero quality/salience gating — the LLM decides everything via the tool's instruction string.
- **Background** (`knowledge/extraction.py`): `create_memory_store_manager` / `MemoryStoreManager`. On each conversation it (1) generates search queries (an LLM `query_gen`, or `get_dilated_windows` over recency windows) and semantic-searches the store for candidate existing memories (`ainvoke` L1015-1039); (2) feeds conversation + retrieved existing memories to `MemoryManager`, which calls `create_extractor` (trustcall) to emit parallel insert/update/delete tool calls in one LLM turn (L253-339); (3) optionally runs additional **`phases`**, each a second LLM pass with its own instructions/schema (L1087-1100); (4) diffs against the store and puts/deletes (L1102-1135). Dedup is instruction-only ("Avoid duplicate extractions", L475); `enable_deletes` defaults **False**. There is no embedding-similarity merge, no salience threshold, no algorithmic reconciliation.

### Retrieval
`create_search_memory_tool` → `store.search(namespace, query, filter, limit, offset)` — **pure semantic vector search** delegated to `BaseStore` (`openai:text-embedding-3-small`, 1536d). No BM25/FTS, no RRF fusion, no learned reranker, no graph expansion, no scoring formula. Ranking is whatever cosine similarity the store returns. `filter` allows metadata equality predicates only.

### Consolidation / Processing
No offline batch "sleep." Consolidation is **synchronous, per-conversation, LLM-prompted**: the background manager's search→extract→phase→diff loop. The `phases` list is the nearest analogue to a multi-stage pass but runs inline per turn, not as a scheduled offline job. `reflection.py` adds a **`ReflectionExecutor`** (`Local`/`Remote`) that debounces this work: `submit(payload, after_seconds=N, thread_id=...)` schedules extraction on a `PriorityQueue`, and a later message on the same `thread_id` cancels/reschedules the pending task — so consolidation fires once the conversation goes quiet rather than on every turn.

### Lifecycle Management
No decay, no forgetting, no archival, no versioning/supersession chain. The only lifecycle op is agent- or manager-initiated `update`/`delete` (LLM-mediated). `created_at`/`updated_at` exist on store items but no time-travel query.

### The other half: prompt optimization (RSI)
`prompts/optimization.py` — `create_prompt_optimizer` (and `create_multi_prompt_optimizer`) rewrite an agent's **system prompt** from conversation trajectories + feedback. Three strategies: `gradient` (reflect-critique then apply, 1-3 reflection steps), `metaprompt` (reflection, single LLM call/step), `prompt_memory` (single-shot). This is "procedural memory" as an optimizable artifact — a distinct axis from item retrieval, and arguably LangMem's real differentiator. The evidence file's memory-retrieval feature matrix omits it entirely.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| "Automatically extracts, consolidates, and updates knowledge" | `MemoryStoreManager.ainvoke` search→extract→phase→diff | Real, but consolidation = one LLM tool-call turn; dedup is instruction-only, no algorithm |
| Semantic search over memories | `store.search` in `create_search_memory_tool` | True — vector only, delegated to BaseStore |
| Hybrid / BM25 / graph retrieval | — | **Absent.** No FTS, no fusion, no reranker. `graph_rag.py` is 100% commented-out demo code |
| Graph-RAG / entity resolution | `graph_rag.py` (all 182 lines commented) | **Not implemented.** Aspirational stub only |
| Cross-provider LLM support | `init_chat_model` | Plausible but only Anthropic/OpenAI paths shown/tested |
| Prompt optimization improves agents (RSI) | `create_prompt_optimizer` 3 strategies | Implemented; no benchmark of gains published |
| Benchmark scores (LoCoMo/LME/PersonaMem) | none | **No published numbers** — not comparable to our 85.1 LoCoMo QA |

---

## Relevance to Somnigraph

### What LangMem does that Somnigraph doesn't
- **Prompt optimization / RSI** (`prompts/optimization.py`): learns at the *system-prompt* level from trajectories+feedback. Somnigraph stores `procedural` memories as retrievable items but never rewrites a governing prompt. Different learning axis entirely; nothing in `reranker.py`/`tools.py` touches this.
- **Debounced deferred consolidation** (`reflection.py ReflectionExecutor`): schedule-after-quiet with per-thread cancellation. Somnigraph's sleep (`scripts/sleep_nrem.py`/`sleep_rem.py`) is manually/externally triggered; there is no in-loop debounce that fires consolidation when a session goes idle.
- **Namespace multi-tenancy**: first-class `("memories", "{user_id}")` partitioning. Somnigraph is explicitly single-user.
- **Runtime-configurable structured Profile memories**: a single Pydantic doc updated in place (good for preference state). Somnigraph has no preference-state / profile primitive.

### What Somnigraph does better
Essentially the entire retrieval and lifecycle stack. Somnigraph has hybrid BM25+vector RRF fusion (`fts.py`, `scoring.py`), a 26-feature learned reranker with measured NDCG (`reranker.py`), an explicit feedback loop with r=0.70 GT correlation, PPR graph-conditioned retrieval over typed edges, offline LLM-mediated sleep consolidation with contradiction/evolution classification, and per-category decay. LangMem has **none** of these: retrieval is raw cosine top-k, there is no reranker, no feedback loop, no decay, no working graph, and consolidation is a single per-turn LLM tool-call. Write-path quality gating that the r/AIMemory sweep flagged as the real differentiator is also absent (dedup is a prompt instruction).

---

## Worth Stealing (ranked)

### 1. Debounced deferred consolidation trigger (Low/Medium)
**What**: `ReflectionExecutor.submit(payload, after_seconds=N, thread_id=...)` queues consolidation and cancels/reschedules if the same thread produces more input before the timer fires — so the expensive extraction runs once the conversation goes quiet, not every turn (`reflection.py`, `LocalReflectionExecutor` + `PriorityQueue`).
**Why**: Somnigraph's sleep is a batch script run out-of-band. A lightweight "fire sleep after the session has been idle N seconds / M turns, coalescing rapid activity" trigger would make consolidation feel automatic without per-turn cost, and the cancellation-on-new-activity avoids redundant passes.
**How**: A session-idle hook (the proactive-injection `UserPromptSubmit` hook is a natural host) that debounces a call into the orchestrator; per-session key = thread_id, reset timer on each prompt. No change to sleep internals, just its trigger.

### 2. Dilated recency windows for consolidation candidate search (Low)
**What**: `utils.get_dilated_windows` builds queries over exponentially growing tail windows of the conversation (last 1, 2, 4, 8… messages) and searches each, so both very recent and broader context pull candidate existing memories to reconcile against (`extraction.py:1032`).
**Why**: Somnigraph's sleep pairwise/merge step selects candidates by embedding/theme; a cheap multi-granularity recency query is a complementary way to surface "what existing memories might this new batch supersede or duplicate" without tuning a single window size.
**How**: In the NREM candidate-gathering step, when comparing a fresh memory batch, issue searches at a few dilated windows and union the candidate set before pairwise classification.

### 3. Prompt optimization as procedural-memory learning (Note-only / High)
**What**: Treat the agent's own governing instructions as a learnable artifact updated from feedback (`gradient`/`metaprompt` strategies).
**Why**: Conceptually adjacent to Somnigraph's feedback loop, but operates on the prompt rather than on retrieval ranking. Worth keeping on the radar as a second learning surface; not adoptable without an agent-prompt Somnigraph doesn't own.

---

## Not Useful For Us

### LangGraph `BaseStore` delegation
Somnigraph's whole thesis is a custom SQLite+sqlite-vec+FTS5 stack with a learned reranker. Delegating storage/search to a generic vector store would throw away hybrid retrieval, the reranker, and the graph.

### `graph_rag.py`
Entirely commented-out demonstration code (entity extract → embed → dedup → edge extract). No working graph; nothing to port. Somnigraph's sleep-detected typed edges + PPR are strictly ahead.

### CRUD memory tools
`create_manage_memory_tool` is a raw put/delete wrapper with no gating — Somnigraph's `remember`/`forget` already do more (categorization, decay, enrichment).

---

## Connections

- **Write-path-is-the-differentiator thesis** (Phase 18 sweep: byterover.md, agentmemory.md, ai-memory-comparison.md): LangMem is a *negative* data point — it has essentially no write-path quality discipline (dedup is a prompt string, no salience gate), and correspondingly publishes no benchmark numbers. Consistent with the finding that thin-write-path systems don't top LoCoMo/LME.
- **LLM-mediated consolidation** convergent with memos/mem0-style "LLM decides insert/update/delete" — same pattern as Somnigraph's NREM classification, but LangMem runs it per-turn inline rather than offline with edge typing.
- **Prompt optimization** is a capability none of the other profiled memory systems have; it aligns LangMem more with DSPy/trajectory-optimization than with the memory-retrieval cohort.

---

## Summary Assessment

LangMem is two libraries wearing one name: (1) a thin, LLM-prompted CRUD-and-extract layer over LangGraph's `BaseStore`, and (2) a genuinely distinctive prompt-optimization ("RSI") toolkit. As a *memory-retrieval* system — the axis this corpus cares about — it is deliberately minimal: pure semantic top-k with no fusion, no reranker, no feedback loop, no decay, and no working graph (the graph-RAG file is entirely commented out). Its consolidation is a single per-turn LLM tool-call whose deduplication is an instruction string. On every retrieval and lifecycle dimension Somnigraph is well ahead.

The single most useful thing to take is architectural, not algorithmic: the **debounced `ReflectionExecutor`** — a clean pattern for firing consolidation once a session goes idle, coalescing bursts of activity, which directly addresses Somnigraph's "sleep is manually triggered" gap and dovetails with the proactive-injection hook already on the roadmap. Secondarily, `get_dilated_windows` is a cheap multi-granularity candidate-search trick for the sleep merge step.

What's overhyped: the docs and file names imply graph-RAG and entity resolution that the code does not deliver (`graph_rag.py` is 182 lines of comments). What the evidence file misses in the other direction: its memory-retrieval feature matrix (correctly all-false on hybrid/decay/dedup/graph) never registers that LangMem's actual headline feature is prompt optimization — so the audit undersells the project on its own terms while correctly scoring it near-zero as a retrieval engine. Net: **MAYBE** — one adoptable pattern (debounced consolidation), no core retrieval idea worth adopting.
