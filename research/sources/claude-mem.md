# claude-mem — Automatic transcript-to-memory capture for Claude Code, with FTS5/Chroma retrieval and heavy distribution polish

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

claude-mem (thedotmack, v13.9.1, Apache-2.0, TypeScript/Bun) is a "memory compression" plugin for Claude Code and several other agent CLIs. Its center of gravity is the **automatic write path** — it passively watches session transcripts via lifecycle hooks and uses an LLM (Claude Agent SDK / Gemini / OpenRouter) to compress agent activity into structured "observations." Retrieval is a comparatively thin FTS5 + optional-Chroma layer. The repo is large (337 `src` files) but most of that mass is distribution: multi-IDE adapters (Cursor, Windsurf, Codex, Gemini, OpenCode, OpenClaw), a hosted Server Beta with auth/rate-limit/Postgres, a React web viewer, telemetry, installers, and 30+ translated READMEs.

### Storage & Schema
Single SQLite DB (`~/.claude-mem/claude-mem.db`, `bun:sqlite`, WAL). Schema (`src/services/sqlite/schema.sql`) is four entity tables plus a work queue and a feedback table:
- `sdk_sessions` — one row per observed session.
- `observations` — the memory unit: `title`, `subtitle`, `narrative`, `text`, `facts` (JSON), `concepts` (JSON), `type`, `files_read/modified` (JSON), `content_hash`, `generated_by_model`. `UNIQUE(memory_session_id, content_hash)`.
- `session_summaries` — one structured summary per session (request/investigated/learned/completed/next_steps).
- `user_prompts` — per-prompt history for UI + LIKE search.
- `pending_messages` — persistent ingestion work queue.
- `observation_feedback(signal_type, ...)` — declared "usage-signal tracking for tier routing," **but has zero readers/writers in `src`** (grep for `observation_feedback`/`signal_type` returns only the schema). It is an unrealized stub.

There are **no** entity/edge/junction tables, no vector column (vectors live in Chroma), no decay/version columns, no priority/salience field.

### Memory Types
A flat `type` enum on observations (default modes: `discovery`, `progress`, `blocker`, `decision`; mode-configurable via `ModeManager`). No episodic/semantic/procedural taxonomy, no priority, no themes. `concepts` (free-text tags the LLM emits) are the only associative metadata.

### Write Path
This is the interesting part. Hooks (SessionStart, UserPromptSubmit, PostToolUse, Stop, SessionEnd) enqueue raw tool events into `pending_messages`; an async worker batches them and calls an LLM with a single-shot extraction prompt (`src/server/generation/providers/shared/prompt-builder.ts`) that asks for `<observation>` XML blocks or a `<skip_summary/>` if nothing is durable. Notable mechanisms:
- **Privacy stripping**: `stripTags()` removes `<private>`, `<claude-mem-context>`, `<system-reminder>` from every event payload before it reaches the LLM; `processGeneratedResponse` additionally discards observations derived entirely from private input (belt-and-suspenders).
- **Dedup**: exact-match only — `computeObservationContentHash` = sha256(session+title+narrative)[:16], inserted `ON CONFLICT DO NOTHING`. No semantic/near-dup detection.
- **Salience gating**: entirely delegated to the LLM's judgment (`<skip_summary/>`). No scoring, no threshold, no quality refinement pass.

### Retrieval
Despite the "hybrid semantic + keyword" README copy, the code is a **single-channel-with-fallback chain, not fusion** (`src/services/worker/SearchManager.ts`):
1. Filter-only SQLite (no query text).
2. If Chroma configured: `chroma_query_documents` with `query_texts:[query]` (so ChromaDB computes embeddings with its **default** model — claude-mem owns neither the embedding model nor the index) → take up to 100 IDs → **apply a 90-day recency filter** (`RECENCY_WINDOW_MS`) → hydrate rows from SQLite by ID. On any Chroma error, fall back to FTS5.
3. Else: FTS5 BM25 (`SessionSearch.searchObservations`, `ORDER BY observations_fts.rank`).

There is **no RRF, no reciprocal-rank fusion, no learned reranker, no cross-encoder** anywhere (grep confirms). Chroma similarity distances are returned but the hydrate step re-orders by recency/date, so even the vector ranking is largely discarded. User-prompt "search" is a raw SQL `LIKE`.

### Consolidation / Processing
None in the biological sense. The closest thing is the **Knowledge Corpus** feature (`src/services/worker/knowledge/CorpusBuilder.ts`, `KnowledgeAgent.ts`): on demand, it runs a search filter, hydrates the matching observations, renders them to a markdown "corpus" file with an LLM-generated system prompt, and estimates tokens. It is a *packaging* step (bundle N observations into a reusable context blob) — no merging, no edge detection, no contradiction handling, no archival.

### Lifecycle Management
None. No decay (the only `decay` hit in `src` is telemetry backoff), no supersession, no contradiction detection, no archival, no versioning, no explicit forget. Memories accumulate; the 90-day retrieval window is the only thing that ages anything out, and only from *results*, not from storage.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| "Hybrid semantic + keyword search" | README; but `SearchManager` is Chroma-OR-FTS5 fallback, recency-filtered, no fusion | **Questionable** — no hybrid fusion exists; two channels never combined |
| "Knowledge graph" (keyword in package.json) | No edge/entity tables; `concepts` are flat tags | **Questionable** — marketing term, no graph |
| Automatic, zero-intervention capture | Hook lifecycle + async worker + LLM extraction, all wired | **Validated** — genuinely automatic; the real strength |
| Privacy via `<private>` tags | `stripTags()` at ingest + post-gen discard | **Validated** — two-layer enforcement |
| Multi-platform (Claude/Codex/Gemini/Cursor/Windsurf/OpenCode/OpenClaw) | Dedicated adapter dirs + installers | **Validated** for the named ones; "Copilot/Hermes/+More" thin |
| Dedup | `UNIQUE(session, content_hash)` exact sha256, `ON CONFLICT DO NOTHING` | **Partial** — exact-match only; evidence file's `dedup:false` is now slightly stale |
| ~79.3k GitHub stars | carsteneu vitals | Not verifiable from clone; implausibly high for the mechanism depth — treat as unverified |

No end-to-end QA benchmark (LoCoMo/PERMA) is run anywhere in the repo. Not comparable to Somnigraph's 85.1% LoCoMo QA.

---

## Relevance to Somnigraph

### What claude-mem does that Somnigraph doesn't
- **Passive, automatic write path.** claude-mem captures memory from the transcript with no agent cooperation: hooks → queue → async LLM compression. Somnigraph's `tools.py:remember()` fires only when the agent *chooses* to call it. This is Somnigraph's single biggest capability gap and directly corroborates the Phase 18 write-path-value thesis (`docs/sessions/2026-06-28-phase18-source-sweep.md`) — except claude-mem automates capture *without* the write-path discipline (no semantic dedup, no salience score, no quality gate) that the thesis says actually wins benchmarks.
- **Privacy filtering at ingest.** `<private>`-tag stripping has no analog in Somnigraph's write path (`events.py`/`tools.py`).
- **Multi-IDE / multi-provider distribution surface.** Somnigraph is single-target (Claude Code MCP). Not a research concern, but a real reach differential.

### What Somnigraph does better
Essentially the entire retrieval and lifecycle stack. Somnigraph has RRF fusion (`scoring.py`), a 26-feature LightGBM reranker with measured NDCG=0.7958 (`reranker.py`), PPR graph expansion over typed edges, an explicit feedback loop with Spearman r=0.70 to ground truth, LLM-mediated sleep consolidation with merge/archive/contradiction handling (`scripts/sleep_nrem.py`, `sleep_rem.py`), and per-category exponential decay. claude-mem has none of these — its "retrieval" is BM25-or-default-embeddings with a hard recency cutoff, and its "consolidation" is on-demand markdown bundling. On every axis this corpus cares about (retrieval quality, feedback, graph, decay), Somnigraph is a generation ahead.

---

## Worth Stealing (ranked)

### 1. Privacy `<private>` tag stripping at the write path (Low)
**What**: Strip user-tagged sensitive spans (and system-reminder/context tags) from content *before* it is embedded/stored, with a second pass that drops any memory derived entirely from private input.
**Why**: Somnigraph ingests whatever the agent hands `remember()`; there is no mechanism to keep a flagged span out of the DB/embedding. Cheap, self-contained, genuinely useful for a personal always-on memory.
**How**: A `strip_private()` in `events.py`/`tools.py` run before `embed_text()`; skip the write if the summary+content are empty after stripping. Mirrors claude-mem's `stripTags` + `processGeneratedResponse` discard.

### 2. Passive transcript-driven capture as the answer to the manual-`remember()` gap (High — note-only)
**What**: A hook that watches the session and asynchronously LLM-compresses activity into candidate memories, instead of relying on the agent to self-report.
**Why**: This is the one architecturally interesting idea and it targets Somnigraph's real weakness. But it is a large surface and cuts against Somnigraph's deliberate agent-driven design; and claude-mem's version lacks exactly the quality gating Somnigraph's own thesis says is decisive. Adopt-worthy only paired with write-path discipline (semantic dedup, salience scoring) — not as-is.
**How**: A `UserPromptSubmit`/`Stop` hook feeding a batched extraction queue, then routing candidates through `remember()`'s existing 0.9-similarity dedup + a salience gate before commit. Convergent-but-larger sibling of the already-designed `docs/proposals/proactive-injection.md`.

---

## Not Useful For Us

- **FTS5-or-Chroma fallback retrieval + 90-day recency cutoff.** Strictly weaker than Somnigraph's RRF+reranker+PPR; the recency cliff would actively destroy long-horizon recall Somnigraph is built for.
- **Knowledge Corpus bundling.** On-demand markdown digest of observations; overlaps nothing Somnigraph needs and is not consolidation.
- **Multi-IDE adapters, web viewer, hosted Server Beta, telemetry, i18n.** Product/distribution scaffolding, orthogonal to a research artifact.
- **`observation_feedback` table.** Declared for "tier routing" but has no code — do not model anything on it.

---

## Connections

- **Corroborates the Phase 18 write-path finding** (`docs/sessions/2026-06-28-phase18-source-sweep.md`, `ai-memory-comparison.md`): claude-mem competes on the *write path* (automatic capture) and is thin on retrieval — same shape as ByteRover (BM25-only) and agentmemory (write-time grounding). But it is the counter-example that sharpens the thesis: it automates capture *without* quality gating, and there is no benchmark to show it pays off.
- **BM25-primary retrieval** echoes ByteRover / MemPalace (`bytrover.md` if present) — retrieval leaders in that group win on write quality, not fusion; claude-mem has neither the write quality nor the fusion.
- **Packaging over an upstream engine**: like several r/AIMemory entries, the "semantic search" is entirely delegated to ChromaDB's default embeddings via its MCP tool — claude-mem owns no vector model or index.

---

## Summary Assessment

claude-mem's genuine contribution is an **automatic, transcript-driven, LLM-compressed write path** wrapped in unusually thorough distribution (multi-IDE, installers, viewer, hosted mode). For a user who wants "memory that just happens" across many agent CLIs, it is a polished product. As a *research* artifact for this corpus it is thin: retrieval is single-channel-with-fallback (no RRF, no reranker, hard 90-day recency cutoff), consolidation is on-demand markdown bundling, and lifecycle management is absent. The evidence file (carsteneu) is refreshingly accurate on all of this — it correctly marks `hybrid`, `decay`, `supersede`, `contradiction`, `clustering`, `codeGraph` false; the only drift is that current code *does* have exact-hash dedup (evidence pinned 13.3.0 marked `dedup:false`), and its `semantic ✅` is fair only as "delegates to Chroma," not as fusion.

The single most important takeaway is negative-space confirmation: claude-mem is what Somnigraph's automatic-capture gap would look like if built *without* the retrieval/feedback/consolidation stack — automation without discipline. The two small things worth carrying over are the `<private>` tag stripping (cheap, adopt) and the passive-capture pattern as a design reference for closing the manual-`remember()` gap (large, note-only, and only if fed through Somnigraph's existing dedup + a salience gate). Nothing here challenges any Somnigraph retrieval or lifecycle decision. Verdict: MAYBE — one revisit-if angle (passive write path) plus one low-effort borrow (privacy stripping).
