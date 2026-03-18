# memv Analysis (Agent Output)

*Generated 2026-03-17 by Opus agent reading GitHub repo*

---

## memv Analysis: Architecture, Comparison, and Insights for Somnigraph

### Repository Overview

**memv** (PyPI: `memvee`) is a structured, temporal memory library for AI agents built by [vstorm-co](https://github.com/vstorm-co). MIT licensed. Python 3.13+, ~4.8K LOC, 215 tests. v0.1.0 released February 2026. 51 stars as of analysis date. The library is explicitly positioned as a "remember what users said" system with ambitions toward procedural/agent memory (deferred to post-v0.2.0).

The project synthesizes ideas from three sources it explicitly credits: **Nemori** (predict-calibrate extraction, episode segmentation), **Graphiti** (bi-temporal validity), and **SimpleMem** (write-time temporal normalization). The claim is that "no single competitor has all four" of its core features (predict-calibrate, write-time temporal normalization, bi-temporal validity, episode segmentation). This is a credible claim based on the source analyses in the Somnigraph research corpus.

### File Structure (Key Paths)

```
src/memv/
  memory/
    memory.py          — Public Memory class (async context manager)
    _api.py            — add_exchange, retrieve, clear_user, knowledge CRUD
    _lifecycle.py      — LifecycleManager (init/open/close, wires components)
    _pipeline.py       — Pipeline (segmentation → episodes → extraction → storage)
    _task_manager.py   — Auto-processing, buffering, background tasks
  processing/
    batch_segmenter.py — LLM-based topic grouping (single call per batch)
    boundary.py        — Legacy incremental boundary detector
    episodes.py        — EpisodeGenerator (messages → third-person narrative)
    extraction.py      — PredictCalibrateExtractor (predict → calibrate → extract gaps)
    episode_merger.py  — Cosine similarity dedup of similar episodes
    prompts.py         — All LLM prompt templates (extraction, prediction, segmentation)
    temporal.py        — Relative→absolute date resolution, backfill logic
  retrieval/
    retriever.py       — Hybrid vector+BM25 with RRF fusion
  storage/sqlite/
    _knowledge.py      — SemanticKnowledge store (bi-temporal fields)
    _episodes.py       — Episode store
    _messages.py       — Message store (append-only)
    _vector_index.py   — sqlite-vec wrapper
    _text_index.py     — FTS5 wrapper
  models.py            — Pydantic models (Message, Episode, SemanticKnowledge, BiTemporalValidity)
  config.py            — MemoryConfig dataclass
  protocols.py         — EmbeddingClient, LLMClient protocols
  embeddings/openai.py — OpenAI text-embedding-3-small adapter
  llm/pydantic_ai.py   — PydanticAI multi-provider adapter
  dashboard/           — Textual TUI for browsing memory state
notes/
  PLAN.md              — Detailed roadmap with competitive analysis
  PROGRESS.md          — Session progress log
  RESOURCES.md         — Reading list
docs/                  — MkDocs site (concepts, API reference, examples)
tests/                 — 215 tests (models, storage, pipeline, retrieval, e2e)
```

---

## 1. Architecture Overview

### Core Components

**Storage layer**: SQLite with sqlite-vec (vector similarity) and FTS5 (BM25 text search). Three stores: `MessageStore` (append-only conversation log), `EpisodeStore` (segmented conversation units with narratives), `KnowledgeStore` (extracted facts with bi-temporal validity). UUIDs stored as TEXT, datetimes as Unix timestamps (INTEGER), embeddings as JSON arrays.

**Write path**: Messages buffer until a threshold is reached (configurable, default 10). The `BatchSegmenter` groups messages into topic-coherent episodes via a single LLM call, with time gaps (default 30 min) as hard segmentation boundaries. `EpisodeGenerator` converts each message group into a titled third-person narrative. `EpisodeMerger` optionally deduplicates semantically similar episodes (cosine threshold 0.9). The `PredictCalibrateExtractor` then runs the predict-calibrate loop against each episode. Extracted knowledge is embedded, deduplicated against existing knowledge (vector similarity threshold 0.8), and stored with vector+text indexing.

**Read path**: `Retriever` does hybrid search: embeds query via sqlite-vec (vector), runs FTS5 (BM25), fuses with RRF (k=60, configurable vector_weight 0-1). Post-fusion temporal filtering via `at_time` and `include_expired` parameters. Results returned as `RetrievalResult` with `to_prompt()` formatting. Embedding cache (LRU with TTL) reduces repeated API calls.

**Temporal model**: Every `SemanticKnowledge` entry carries four temporal fields:
- `valid_at` / `invalid_at` — event time (when the fact was true in the world)
- `created_at` / `expired_at` — transaction time (when the system learned/superseded it)
- `superseded_by` — UUID link to the replacement entry

Contradictions don't delete old facts; they set `expired_at` and link via `superseded_by`. Point-in-time queries use `is_valid_at(event_time)` for event-time filtering and `is_current()` for transaction-time filtering.

**Extraction model**: The predict-calibrate approach (from Nemori/Hindsight paper, arXiv:2508.03341) works in two stages:
1. **Predict**: Given existing knowledge and the episode title, the LLM predicts what the episode should contain
2. **Calibrate**: Compare prediction against the original messages (not the generated narrative — this is a deliberate design decision to avoid LLM-generated corruption). Extract only what was unpredicted.

Each extracted item is classified as `new`, `update`, or `contradiction`, with a `supersedes` index pointing into the numbered existing knowledge list. Confidence threshold is 0.7 (the only code-level quality filter; all other quality enforcement is prompt-level for language agnosticism).

### Schema

```sql
-- Messages (append-only)
CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    role TEXT,
    content TEXT,
    sent_at INTEGER
);

-- Episodes (segmented conversations)
CREATE TABLE episodes (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    title TEXT,
    content TEXT,
    original_messages TEXT,  -- JSON
    start_time INTEGER,
    end_time INTEGER,
    created_at INTEGER
);

-- Knowledge (extracted facts with bi-temporal validity)
CREATE TABLE semantic_knowledge (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    statement TEXT,
    source_episode_id TEXT,
    created_at INTEGER,
    importance_score REAL,
    embedding TEXT,          -- JSON array
    valid_at INTEGER,
    invalid_at INTEGER,
    expired_at INTEGER,
    superseded_by TEXT
);
```

---

## 2. Unique Concepts

**a. Predict-calibrate extraction** — The core innovation. Instead of extracting all facts from a conversation, memv first predicts what the conversation should contain given existing knowledge, then extracts only the gaps. Importance emerges from prediction error, not explicit LLM scoring. Cold start (no existing knowledge) extracts everything; subsequent episodes are filtered through the prediction model. This is the clearest implementation of the Nemori/Hindsight approach in the research corpus.

**b. Bi-temporal validity with supersession chains** — Dual timeline tracking (event time vs. transaction time) on every knowledge entry. Facts are never deleted on contradiction; they're expired and linked to their successor via `superseded_by`. This enables point-in-time queries on both "when was this true?" and "when did we learn this?" axes. The `include_expired=True` flag exposes the full revision history.

**c. Episode-first extraction pipeline** — Messages are segmented into episodes before any extraction happens. The `BatchSegmenter` handles interleaved topics (non-consecutive message groupings) via a single LLM call, which is more efficient than incremental boundary detection. Episodes carry both a third-person narrative (for retrieval context) and the original messages (for extraction ground truth).

**d. Extraction-source discipline** — A deliberate distinction between episode content (LLM-generated narrative, used for display/retrieval) and original messages (raw conversation, used as ground truth for extraction). The code comments explain: "Episode content is for RETRIEVAL (narrative is fine) / Extraction source is ONLY original messages (ground truth)." This prevents LLM hallucinations in the narrative from propagating into the knowledge base.

**e. Write-time temporal normalization** — Relative dates ("last week", "yesterday") are resolved to absolute timestamps at extraction time using the episode's end time as reference. The `temporal.py` module provides regex-based detection of unresolved references and a `backfill_temporal_fields()` function that parses temporal_info strings into `valid_at`/`invalid_at` values. This accounts for "56.7% of temporal reasoning" according to the SimpleMem ablation the authors cite.

**f. Index-based supersession** — Existing knowledge is passed to the extraction prompt as a numbered list (`[0] User works at Google`, `[1] User lives in SF`). The LLM returns a `supersedes: int` index pointing to which existing entry is being replaced. This is more reliable than post-hoc vector similarity matching for contradictions. Pipeline falls back to vector-based matching when the index is null or out of bounds.

**g. Atomization quality via prompt engineering** — All quality enforcement (third-person construction, no pronouns, no relative time, no assistant-sourced facts) is handled at the prompt level rather than code-level regex filters. The rationale: regex filters are English-only and were removed in favor of language-agnostic prompt instructions. Seven knowledge categories are defined in prompts.py: identity/background, preferences, technical details, relationships, goals/plans, beliefs/values, habits/patterns.

**h. Planned `extends` relationship with cascade invalidation** — Not yet implemented but designed (PLAN.md v0.2.0 §7): child facts that extend a parent (e.g., "works on Search team at Google" extends "works at Google") would be cascade-invalidated when the parent is superseded. Addresses a validated gap: the current system misses transitive contradictions.

---

## 3. How memv Addresses Our 7 Known Gaps

| Gap | Assessment |
|-----|-----------|
| 1. Layered Memory | **Partial, different framing.** memv has three layers (messages → episodes → knowledge) but these are pipeline stages, not retrieval-time layers. There's no working memory vs. long-term distinction, no procedural memory, no category-based routing. The roadmap mentions agent/procedural memory as a post-v0.2.0 goal but explicitly defers it. The seven knowledge categories in prompts.py are extraction-time labels, not retrieval-time filters. |
| 2. Multi-Angle Retrieval | **Not addressed.** Retrieval is hybrid (vector + BM25 + RRF) but single-angle — one query, one embedding, one fused result set. No query expansion, no conversation-aware retrieval, no graph traversal, no theme-based boosting. The PLAN.md Ideas Parking Lot explicitly lists "Conversation-aware retrieval" and "Multi-phase chain-of-thought retrieval" as deferred. |
| 3. Contradiction Detection | **Directly addressed — this is a strength.** The predict-calibrate approach inherently surfaces contradictions (extraction classifies items as `new`/`update`/`contradiction`). Index-based `supersedes` links old and new facts explicitly. The bi-temporal model preserves history rather than overwriting. However, cascade invalidation for transitive contradictions is not yet implemented (planned for v0.2.0). |
| 4. Relationship Edges | **Planned but not implemented.** The `extends` relationship (v0.2.0 §7) would create parent-child links between facts, enabling cascade invalidation. The Ideas Parking Lot defers `derives` (inferred knowledge) and full knowledge graph. Currently there are no explicit edges between knowledge entries beyond `superseded_by` chains. |
| 5. Sleep Process | **Not addressed.** No offline consolidation, no scheduled maintenance, no decay. The PLAN.md "Not doing" list explicitly excludes "Background consolidation" as "premature without knowledge growth data." Knowledge persists indefinitely unless manually invalidated or superseded by a contradiction. |
| 6. Reference Index | **Not addressed.** Episode source tracking exists (`source_episode_id` on knowledge), but there's no citation system, no research source tracking, no external reference management. This is expected — memv is a conversation memory library, not a knowledge management system. |
| 7. Temporal Trajectories | **Strongest coverage of any system analyzed.** The bi-temporal model with `superseded_by` chains creates explicit temporal trajectories. Point-in-time queries (`at_time` parameter) let you reconstruct what was known at any moment. `include_expired=True` exposes the full revision history. Write-time temporal normalization ensures temporal information is captured at extraction time rather than lost. This is the most complete temporal model in the research corpus. |

---

## 4. Comparison

### Where memv Contributes

**Bi-temporal model is more principled than Somnigraph's.** Somnigraph has `valid_from`/`valid_until` fields on memories but uses them primarily for consolidation-based temporal handling during sleep. memv's dual-timeline separation (event time vs. transaction time) with explicit supersession chains is architecturally cleaner and enables true point-in-time queries. Somnigraph can answer "what do I know now?" but cannot easily answer "what did I know last Tuesday?" or "show me how this fact evolved over time."

**Predict-calibrate extraction reduces write-path noise.** Somnigraph extracts knowledge through LLM-mediated consolidation during sleep (NREM classification, REM summarization), but the initial write path accepts everything the caller provides via `remember()`. memv's predict-calibrate approach filters at extraction time — only genuinely novel information enters the knowledge base. This is a fundamentally different philosophy: filter early (memv) vs. consolidate later (Somnigraph).

**Supersession chains are more explicit than Somnigraph's contradiction handling.** Somnigraph detects contradictions during sleep consolidation and flags edges with `contradiction=True`, but the link between the old and new fact is implicit (via graph edges). memv's `superseded_by` UUID creates an explicit, traversable chain of fact evolution.

**Episode segmentation provides structure Somnigraph lacks.** Somnigraph stores individual memories as atomic units. memv groups conversations into episodes first, preserving conversational context. This intermediate representation enables the predict-calibrate approach (you need the episode as context for prediction) and provides a natural unit for retrieval display.

**Multi-user isolation is built-in.** All memv queries are scoped by `user_id`. Somnigraph is single-user by design.

### Where Somnigraph Is Stronger

**Retrieval quality is far more sophisticated.** Somnigraph has four retrieval signals (vector, BM25, graph/PPR, theme matching) fused with RRF, a learned LightGBM reranker trained on ground-truth judgments, and a feedback loop (`recall_feedback()`) that shapes future scoring. memv has vector + BM25 + RRF with no learning, no feedback, no graph signal, and no score threshold filtering (it always returns `top_k` results regardless of relevance — score threshold is a planned v0.1.1 feature).

**Feedback loop is unique.** Somnigraph's `recall_feedback()` → EWMA scoring → UCB exploration → reranker training pipeline is the most distinctive feature in the research corpus. memv has no feedback mechanism. The PLAN.md Ideas Parking Lot lists "Feedback loop" and "Retrieval reinforcement" as deferred ideas.

**Offline consolidation adds a dimension memv doesn't have.** Somnigraph's three-phase sleep process (NREM classification → REM summarization → maintenance) performs entity extraction, theme enrichment, summary generation, and contradiction detection offline. memv has no equivalent — knowledge stays exactly as extracted, with no mechanism for refinement, compression, or relationship discovery after the initial write.

**Biological decay is a meaningful information management strategy.** Somnigraph's decay model (biological half-life + reheat on access) naturally surfaces recent/relevant memories and lets unused ones fade. memv has no decay — knowledge persists indefinitely. The `forgetAfter` idea in the parking lot is noted but not implemented.

**Graph structure enables multi-hop reasoning.** Somnigraph's entity-relationship graph with PPR traversal can surface related memories that share entities or themes, even when they're not textually or semantically similar to the query. memv has no graph layer; the planned `extends` relationship is parent-child only, not a general graph.

**Enriched embeddings improve retrieval.** Somnigraph concatenates content + category + themes + summary into the embedding input, broadening the semantic surface for vector search. memv embeds only the knowledge statement text.

**Evaluation methodology is rigorous.** Somnigraph has ground-truth judgments, 5-fold CV, NDCG measurements, utility calibration studies. memv has a LongMemEval benchmark harness built but not yet run ("Full run pending").

### Fundamental Differences

**Write-time filtering vs. read-time consolidation.** memv invests LLM compute at write time (predict-calibrate extraction, episode generation, temporal normalization) to ensure only clean, novel facts enter the knowledge base. Somnigraph accepts raw memories at write time and invests LLM compute in offline consolidation (sleep) to refine, categorize, and relate them. These are opposing philosophies with different tradeoffs: memv's approach is cleaner for the knowledge base but more expensive per conversation turn; Somnigraph's approach is cheaper on the write path but requires a separate consolidation process.

**Library vs. server.** memv is a pip-installable library designed for integration into any agent framework (Pydantic AI, LangGraph, CrewAI, etc.). Somnigraph is an MCP server tightly coupled to Claude Code's usage patterns. This difference shapes every API decision — memv needs to be generic; Somnigraph can be opinionated.

**Multi-user vs. single-user.** memv is designed for multi-tenant applications where privacy isolation between users is a hard requirement. Somnigraph serves one user's memory needs. This drives different schema designs (every memv query includes `user_id`; Somnigraph has no user scoping).

**Fact-centric vs. memory-centric.** memv stores atomic declarative facts ("User works at Anthropic"). Somnigraph stores richer memory objects with content, summary, category, themes, priority, source metadata, and graph edges. memv's facts are lean and composable; Somnigraph's memories carry more context but are heavier.

---

## 5. Insights Worth Stealing

1. **Supersession chains with explicit `superseded_by` links** — HIGH impact, LOW effort. Somnigraph already has contradiction detection during sleep, but the link between old and new memories is implicit (contradiction-flagged graph edges). Adding a `superseded_by` field to the memory schema would enable traversable fact-evolution histories and could improve the sleep consolidation's contradiction handling. This is a schema migration + minor consolidation code change.

2. **Bi-temporal query support (`at_time` parameter on recall)** — MEDIUM impact, MEDIUM effort. Somnigraph has `valid_from`/`valid_until` fields but doesn't expose point-in-time queries through the MCP API. Adding an `at_time` parameter to `recall()` that filters by event time would unlock temporal reasoning without changing the storage layer. The filtering logic from memv's `_passes_temporal_filter()` is straightforward.

3. **Extraction-source discipline (narrative vs. raw messages)** — HIGH impact on extraction quality, LOW effort conceptually. The principle that extraction should work from raw source material rather than LLM-generated summaries is sound. Somnigraph's sleep REM step generates summaries that then inform subsequent processing. Worth auditing whether any consolidation steps extract facts from LLM-generated content rather than from original memory content.

4. **Write-time temporal normalization** — MEDIUM impact, MEDIUM effort. Somnigraph doesn't resolve relative time references at write time. Adding a temporal parsing step (similar to memv's `temporal.py`) during `remember()` or during sleep consolidation could improve temporal field population. The regex patterns and dateutil-based parsing are well-tested and portable.

5. **Episode-level grouping for consolidation context** — LOW impact (Somnigraph already consolidates by category/theme), LOW effort. The idea of grouping related memories into episode-like clusters before consolidation could improve sleep REM summarization quality by providing conversational context. Currently Somnigraph processes memories individually during sleep.

---

## 6. What's Not Worth It

**Predict-calibrate extraction** — Somnigraph's write path is fundamentally different (caller provides memories directly via `remember()`; there's no conversation-to-fact extraction pipeline). Retrofitting predict-calibrate would require building a conversation ingestion layer that doesn't exist and isn't needed — the caller (Claude Code) already decides what to remember.

**Episode segmentation** — Same reasoning. Somnigraph doesn't ingest raw conversations. The sleep consolidation process serves a similar purpose (grouping and refining knowledge) but operates on already-extracted memories.

**Multi-user isolation** — Somnigraph is a personal memory system by design. Adding user scoping would add complexity without benefit.

**Library architecture** — memv's plugin architecture (protocols for storage backends, embedding clients, LLM clients) is designed for a reusable library. Somnigraph is an opinionated server; the tight coupling to specific implementations (SQLite, OpenAI embeddings, Claude for consolidation) is a feature, not a bug.

**Prompt-only quality enforcement** — memv removed code-level extraction validation in favor of prompt-level rules for language agnosticism. Somnigraph's quality controls are in the consolidation pipeline (NREM classification, theme extraction, etc.) and benefit from being code-level where precision matters.

---

## 7. Critical Assessment

### Strengths

**Clean architecture.** The codebase is well-organized with clear separation of concerns. The pipeline stages (messages → episodes → knowledge) form a logical progression. The protocol-based design enables future backend swaps. 215 tests for 4.8K LOC is good coverage.

**Honest roadmap.** The PLAN.md is unusually transparent about what's missing ("table-stakes that memv is still missing"), what was tried and removed (regex filters), and what's explicitly deferred with reasoning. The "Not doing" and "Ideas Parking Lot" sections show disciplined prioritization.

**Strongest temporal model in the corpus.** The bi-temporal validity implementation is the most principled approach to temporal knowledge management seen in the 63+ systems analyzed. The separation of event time from transaction time, combined with supersession chains and point-in-time queries, is genuinely useful and well-implemented.

**Predict-calibrate is validated and well-implemented.** The extraction flow faithfully implements the Nemori/Hindsight approach with practical additions (index-based supersession, cold-start handling, temporal backfill). The extraction-source discipline (raw messages, not narratives) is a smart detail.

### Limitations

**No retrieval learning or feedback.** The retrieval system is static — RRF with fixed k=60 and configurable but hand-tuned vector_weight. No mechanism to learn from retrieval outcomes. No score threshold filtering (every query returns exactly `top_k` results). This is the biggest functional gap relative to Somnigraph.

**No offline processing.** Knowledge stays exactly as extracted. No consolidation, no decay, no relationship discovery, no re-embedding, no maintenance. The system depends entirely on extraction-time quality. If the LLM extracts a poor fact, it persists forever unless manually invalidated.

**Benchmark claims are unvalidated.** The PLAN.md notes the LongMemEval harness is "built, full run pending." The predict-calibrate approach is attributed to Nemori (which has published numbers), but memv's own performance is unmeasured. The claim that it reduces extraction noise is plausible but unquantified.

**LLM dependency on the write path is expensive.** Every conversation requires: (1) batch segmentation LLM call, (2) episode generation LLM call, (3) prediction LLM call, (4) extraction LLM call, and optionally (5) merge verification LLM call. That's 3-5 LLM calls per conversation turn. Somnigraph's write path is zero LLM calls (`remember()` just writes to SQLite); LLM work happens during offline sleep.

**No graph structure limits relational reasoning.** The planned `extends` relationship (v0.2.0) addresses parent-child facts, but there's no entity graph, no theme connections, no general-purpose relationship edges. Retrieval cannot discover connections between facts that aren't textually or semantically similar.

**Cascade invalidation gap is validated but unresolved.** The PLAN.md honestly documents that "User works on Search team at Google" survives when the user moves to Anthropic because the embedding similarity is below the contradiction threshold. The `extends` solution is designed but not implemented.

**v0.1.0 maturity.** The library is alpha-quality with acknowledged missing features (score threshold, Postgres backend, protocol completeness). The "Not doing (yet)" list is longer than the "Completed" list. This is appropriate for an early-stage project but means the system is not yet production-ready by its own standards.
