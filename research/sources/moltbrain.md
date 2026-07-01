# MoltBrain — Passive session-transcript capture for Claude Code via an observer subprocess, with an optional crypto-paid storage vault

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

MoltBrain (npm `moltbrain`, v9.0.9, AGPL-3.0, formerly "claude-recall") is a **context-persistence / "memory compression" plugin** for Claude Code, OpenClaw, and MoltBook. It is a TypeScript/Bun monorepo (~19 MB) whose job is to watch a coding session, distill it into structured "observations," and re-inject relevant history into future sessions. It is a session-history recorder, not a retrieval-quality research system.

### Storage & Schema
SQLite (via `bun:sqlite`) is the system of record; ChromaDB is an optional vector sidecar. Schema (`migrations/001_initial.sql`, `002`, `003`):
- **`sessions`** — content_session_id, project, timestamps, is_complete, prompt_count.
- **`observations`** — the memory unit: `type`, `title`, `subtitle`, `narrative`, `facts` (JSON array), `concepts` (JSON array), `files_read` (JSON), `files_modified` (JSON), `project`, `prompt_number`, `tokens_used`/`discovery_tokens`, `is_favorite` (mig 002), plus `observation_tags` junction (mig 003). ~17 columns.
- **`observations_fts`** — FTS5 virtual table over title/subtitle/narrative/facts/concepts, kept in sync by insert/update/delete triggers.
- **`summaries`** — session-level rollup with fixed fields: `request`, `investigated`, `learned`, `completed`, `next_steps`, `notes`.
- **`prompts`**, **`pending_messages`** (a work queue), `tags`, `favorites`.
- ChromaDB collection `cm__{project}` mirrors observations/summaries for semantic search (`src/core/vector/VectorSync.ts`). **Disabled on Windows** (MCP stdio subprocess spawns console popups — `this.disabled = process.platform === 'win32'`).

There is **no vector column in SQLite** (no sqlite-vec); vectors live only in Chroma. No `priority`, no `decay_rate`, no `valid_from/valid_until`, no edge table.

### Memory Types
Observation `type` is a free-form, mode-configurable enum (the mode JSON defines `observation_types`, e.g. discovery/decision/investigation/learned/next-steps — see `contrib/modes/research.json`, `src/core/domain/ModeManager.ts`). Types are labels for display/filtering, not distinct storage or retrieval paths. No episodic/semantic/procedural cognitive taxonomy.

### Write Path
This is the system's one genuinely distinctive mechanism. A **sidecar "observer" agent** runs the memory extraction out-of-band:
- Claude Code hooks (PostToolUse, Stop, UserPromptSubmit) feed the primary session's tool calls into a queue (`pending_messages`), handled by `src/interface/handlers/*` and a background worker.
- `src/core/engine/SDKAgent.ts` spawns a **separate Claude subprocess via `@anthropic-ai/claude-agent-sdk`** that is **observer-only** — every tool (Bash/Read/Write/Edit/Grep/Glob/WebFetch/Task/TodoWrite…) is in `disallowedTools`. It cannot act; it only watches.
- Each tool event is rendered as XML (`buildObservationPrompt`, `src/parser/prompts.ts`: `<what_happened>`, `<parameters>`, `<outcome>`), and the observer is prompted to emit an `<observation>` in a fixed XML schema (`buildInitPrompt`).
- The response is parsed (`src/parser/parser.ts`) and written via `storeObservation` (`src/core/storage/observations/store.ts`), then synced to Chroma.

Notably absent: **no deduplication, no salience/quality gating, no enrichment beyond the LLM's own extraction, no entity resolution.** The observer captures whatever it deems noteworthy; there is no filter deciding a memory is too low-value to keep. `cleanup-duplicates.ts` exists as a maintenance CLI but is not part of the write path; the "deduplicate" hits elsewhere in the repo are all UI-level merge of SSE + paginated rows by row ID.

### Retrieval
`src/core/engine/search/SearchOrchestrator.ts` picks one of three strategies — there is **no score fusion**:
- **Filter-only (no query text)** → `SQLiteSearchStrategy` (metadata WHERE clauses).
- **Query text + Chroma available** → `ChromaSearchStrategy` (pure vector similarity from Chroma; Chroma owns the ranking).
- **`HybridSearchStrategy`** (used by findByConcept/findByType/findByFile): SQLite metadata filter to get candidate IDs → `chromaSync.queryChroma()` to rank → `intersectWithRanking()` keeps metadata IDs **in Chroma's rank order** → hydrate from SQLite. This is a filter-then-semantic-order pipeline, **not RRF and not a learned reranker**. The final order is simply Chroma's cosine order restricted to the filtered set.
- If Chroma fails or is unavailable (e.g. Windows), it silently falls back to SQLite FTS/filter with `fellBack: true`.

No relevance score is computed or blended; no BM25×vector fusion; no reranker model; no graph expansion. FTS5 BM25 exists but is the fallback lane, never fused with vectors.

### Consolidation / Processing
"Consolidation" = **session summarization**, not memory-graph consolidation. When a session completes, `src/interface/handlers/summarize.ts` + `buildSummaryPrompt` ask the observer to emit the 5-field summary (request/investigated/learned/completed/next_steps) stored in `summaries`. There is no offline pass that merges duplicate memories, detects contradictions, builds edges, or reclassifies. No sleep cycle. No clustering.

### Lifecycle Management
**None.** No decay, no half-lives, no archival tier, no versioning/supersession, no TTL. Memories are immutable rows that accumulate forever (favorites/tags are the only mutation). The "time-travel" feature is just a timeline query over `created_at_epoch`, not versioned state.

**Crypto/vault layer (optional, external):** the README's "Storage Dapp" and "Virtuals Protocol" sections describe an *optional remote vault* at `app.moltbrain.dev/storage` — per-wallet-scoped, Postgres-backed, paid per API call via **x402 micropayments ($0.01 USDC on Base)**, plus a separate GAME-SDK plugin (`nhevers/Moltbrain-virtuals`) giving Virtuals AI agents persistent storage. This is a hosted commercial add-on, not part of the local memory engine, and has no bearing on retrieval quality.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| "Learns and recalls your project context automatically" | Observer subprocess auto-captures every session; hooks wire it in | Validated — passive capture genuinely works end-to-end |
| "knowledge-graph" (package.json keyword) | No edge table, no entity linking, no graph traversal anywhere in code | **Questionable / overclaim** — there is no graph |
| Dual search modes (semantic + full-text) | ChromaSearchStrategy + SQLiteSearchStrategy both present | Validated |
| "Hybrid" search | filter→Chroma-order→intersect; no fusion/rerank | Plausible-but-thin — it is filter-then-semantic-order, not hybrid ranking |
| Multi-agent shared memory | MoltBook/Virtuals via remote vault | Validated as external feature; not in local engine |
| Crypto-paid storage tier | x402 / USDC-on-Base vault, separate hosted service | Validated but orthogonal to memory quality |
| carsteneu "13/47" score | Feature-checklist tally, **not a benchmark** | Not comparable to any QA/recall metric |

**No LoCoMo, no PERMA, no R@k, no NDCG — this system publishes zero retrieval-quality numbers.**

---

## Relevance to Somnigraph

### What MoltBrain does that Somnigraph doesn't
- **Passive write-time auto-capture via an out-of-band observer subprocess.** Somnigraph captures explicitly (the CLAUDE.md rhythm + `remember()`); MoltBrain's observer-only Claude subprocess distills the primary session's tool stream into structured observations with zero user action. Somnigraph has no analogue in `tools.py` / no write-side auto-ingest at all.
- **Code-anchored provenance as first-class metadata** (`files_read` / `files_modified` per observation, enabling `findByFile`). Somnigraph's schema (`db.py`) has no file-provenance fields; a memory cannot be retrieved by "which memories touched `scoring.py`."
- **A shipped web viewer + SSE live feed** (localhost:37777) and per-project scoping. Somnigraph is CLI/MCP-only.

### What Somnigraph does better
Essentially the entire retrieval-quality stack. Somnigraph has RRF fusion + a 26-feature LightGBM reranker (`reranker.py`) with measured NDCG=0.7958; MoltBrain does single-channel selection with no fusion and no learned ranking. Somnigraph has a graph with typed edges and PPR expansion (`scoring.py`); MoltBrain's "knowledge-graph" is a keyword only. Somnigraph has LLM-mediated sleep consolidation (`sleep_nrem.py`/`sleep_rem.py` — edge creation, contradiction classification, merge/archive); MoltBrain has session summaries only. Somnigraph has an explicit feedback loop with Spearman r=0.70 to GT; MoltBrain has none. Somnigraph has per-category decay; MoltBrain has no lifecycle at all. Somnigraph is benchmarked (85.1% LoCoMo QA); MoltBrain publishes no quality metric.

---

## Worth Stealing (ranked)

### 1. File-provenance fields on the memory unit (Low effort)
**What**: MoltBrain stores `files_read` / `files_modified` per observation and exposes `findByFile`, so memory is retrievable by the code artifact it concerns.
**Why**: Somnigraph is a memory system for a *coding* agent yet cannot answer "what do I know that relates to `reranker.py`?" This is a cheap, high-precision retrieval channel orthogonal to BM25/vector, and file-path co-occurrence could seed Hebbian edges during sleep.
**How**: Add an optional `files` JSON field to the `db.py` memory schema; index it; add a file-filter path in `fts.py`/`tools.py`; optionally feed file overlap as a candidate edge signal in `sleep_nrem.py`. Purely additive.

### 2. Out-of-band observer subprocess for auto-capture (High effort, note-only)
**What**: A separate observer-only Claude subprocess (all tools disallowed) watches the tool stream and emits structured memories without user prompting.
**Why**: Somnigraph's roadmap flags write-path/auto-capture as a gap; this is a concrete architecture for it, and it decouples capture cost from the main session.
**How**: Would require a hook + worker + SDK subprocess layer Somnigraph deliberately doesn't have. **Caveat that guts most of the value:** MoltBrain does this with *no dedup and no quality gate*, so it accumulates noise — the opposite of the Phase 18 finding that write-path *discipline* (grounding, salience filtering) is what the LoCoMo leaders win on. Adopt the pattern only if paired with a salience gate Somnigraph would have to build itself. Hence note-only.

---

## Not Useful For Us

### Crypto storage vault (x402 / USDC-on-Base / Virtuals Protocol)
A hosted commercial add-on for paid per-wallet remote storage and multi-agent monetization. Single-user local Somnigraph has no use for micropayment-gated storage or on-chain agent economies.

### The 5-field session summary schema
`request/investigated/learned/completed/next_steps` is a fine UI rollup but is coarser and less structured than Somnigraph's category/theme/layer model; nothing to import.

### "Hybrid" search
It is filter-then-semantic-order, strictly weaker than Somnigraph's RRF + learned reranker. No mechanism to borrow.

---

## Connections

- **Write-path-quality thesis (see the Phase 18 sweep: `byterover.md`, `agentmemory.md`, `ai-memory-comparison.md`):** MoltBrain is the *negative* example — it has an interesting auto-capture write path but zero write-time quality control, reinforcing that capture without grounding/salience is where these session-recorder tools plateau.
- **Session-transcript recorders as a category:** convergent in spirit with other "context persistence for Claude Code" tools (hook-driven, per-project, timeline UI). Distinguished from *retrieval-quality* systems (Mem0, MemOS, Somnigraph) that publish LoCoMo/PERMA numbers — MoltBrain publishes none.
- **carsteneu evidence file:** its "13/47" is a documentation-feature tally, not a benchmark; do not treat it as comparable to any recall or QA metric.

---

## Summary Assessment

MoltBrain's core contribution is an **ergonomic passive-capture pipeline**: an observer-only Claude subprocess, driven by Claude Code hooks, that turns a live coding session into structured, file-annotated observations and re-surfaces them later — all with a polished web viewer and a per-project store. As a *product* for "never lose your project context," it is coherent and well-engineered. As a *memory-research artifact*, it is thin: single-channel retrieval with no fusion, no learned ranking, no graph (despite the keyword), no consolidation beyond session summaries, no feedback, and no lifecycle. It publishes no retrieval or QA numbers, so there is nothing to compare against Somnigraph's 85.1% LoCoMo.

The single most transferable idea is small and cheap: **file-provenance fields (`files_read`/`files_modified`) as a first-class, indexed retrieval dimension** — a channel Somnigraph lacks and could add to `db.py`/`fts.py` in an afternoon, with a bonus use as a sleep-time edge signal. The observer-subprocess auto-capture is architecturally interesting and maps to a known Somnigraph gap, but MoltBrain implements it without any salience or dedup gating, which is exactly the discipline the corpus keeps showing is what actually moves benchmark numbers — so it is a pattern to note, not to port wholesale.

Overhyped: the "knowledge-graph" label (there is none) and the crypto/Virtuals vault (an orthogonal hosted monetization layer). Missing for our purposes: any evidence the retrieval is good. Verdict for Somnigraph: **MAYBE**, on the strength of the file-provenance idea alone.
