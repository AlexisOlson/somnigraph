# Memvid — Single-file portable memory *format* (Rust) with regex-triplet entities, RRF-fused lexical/vector RAG, and bitemporal fact-versioning

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

Memvid is a **Rust library crate** (`memvid-core`, Apache-2.0) that packages data, embeddings, search indices, and metadata into a single portable `.mv2` file — "memory without a database." It is a storage/retrieval *engine and file format*, not a conversational-agent memory service. CLI (`memvid-cli`, Node) and Python SDK (`memvid-sdk`) wrap it. ~22.7k LOC in `src/memvid/` alone; large document-ingestion surface (PDF/docx/xlsx/pptx readers, Whisper audio, CLIP visual search, encryption capsules, ACL, PII redaction).

### Storage & Schema
Single-file `.mv2` container (`MV2_SPEC.md`): Header → WAL → append-only Data Segments of "Smart Frames" → Lex/Vec/Time indices → TOC. A **Frame** (`src/types/frame.rs`) has 9 fields: `frame_id`, `uri`, `title`, `created_at`, `encoding`, `payload`, `payload_checksum`, `tags` (free-form `Map<String,String>`), `status` (active/tombstoned). Frames are **immutable & append-only**; deletion is a tombstone (`status`).

Above raw frames sits a **MemoryCard** layer (`src/types/memory_card.rs`) — structured SPO facts. Fields worth noting: `entity`, `slot`, `value`, `kind` (Fact/Preference/Event/Profile/Relationship/Goal), `polarity`, **`event_date` (when it became true) vs `document_date` (when recorded)** — genuine bitemporality — plus `version_key` (defaults to `entity:slot`), `version_relation` (Sets/Updates/Extends/Retracts), provenance (`source_frame_id`, `source_offset`, `engine`, `engine_version`, `confidence`).

### Memory Types
Two axes. Frame-level: MIME/encoding-typed payloads. Card-level: the 6 `MemoryKind` variants above. No episodic/semantic/procedural taxonomy, no priority, no themes, no decay_rate.

### Write Path
`put_bytes()` appends a frame (manual — **no auto-extraction from conversation**). Entity extraction is offline and runs via `TripletExtractor` (`src/triplet/extractor.rs`) → `RulesEngine` (`src/enrich/rules.rs`): **regex patterns** (`ExtractionRule { pattern: Regex, kind, entity, slot, value, polarity }`) that emit MemoryCards. Default `ExtractionMode::Rules`. **LLM extraction is a stub**: `should_run_llm()` gates a block that literally does `let llm_count = if ... { 0 } else { 0 };` (extractor.rs ~121-129) — the only `EnrichmentEngine` impl is `RulesEngine`. Dedup: `deduplicate_cards()` collapses same `entity:slot`. No quality/salience gating, no LLM refinement.

### Retrieval
Two distinct paths — this is the crux, and the README/evidence blur them:

1. **Base `search()`** (`src/memvid/search/mod.rs`) is **Tantivy BM25 lexical only** + optional temporal date-range filters + a lex-only fallback. Vector search exists (`search_vec`, HNSW + ONNX BGE-small 384d, or OpenAI `api_embed`) but is *not* fused into `search()`.
2. **The real hybrid lives in the RAG `ask()` pipeline** (`src/memvid/ask.rs`). `ask()` classifies the question (analytical / aggregation / recency / update), widens `top_k` (analytical → ×5, aggregation → ×3), gathers **candidate lists** (lexical, OR-relaxed query, vector via `vec_search_with_embedding`, corrections), and **fuses with RRF** (`fuse_hits_rrf`, `RRF_K = 60.0`, contribution `1/(60+rank)`). Then a **cosine semantic re-rank** over the query embedding (`apply_semantic_ranking`), `promote_temporal_extremes` for recency/update questions, `diversify_hits_for_aggregation` (one hit per session), and `promote_corrections`. This is a hand-built heuristic router, not a learned model.

`graph_search.rs::hybrid_search` is **misnamed** — it does no fusion. A `QueryPlanner` matches hardcoded keyword patterns ("who lives in", "who works at", possessives) to a `slot`, does an O(1) `entity:slot` MemoryCard lookup, and returns those frames at fixed `score = 1.0` (`graph_score=1.0, vector_score=0.0`); on no match it falls back to lexical. There is no graph traversal, PPR, or score blending.

### Reranker
`src/types/reranker.rs` defines a `Reranker` trait whose doc-comment advertises `CrossEncoderReranker`, `LLMReranker`, `Bm25Reranker` — **none of these exist in the codebase** (only a `MockReranker` in tests). `RerankerKind` enum defaults to `None`. The effective "reranking" is the cosine + heuristic stack inside `ask()`.

### Consolidation / Processing
**None.** No sleep, no offline pairwise relationship detection, no merge/archive cycle, no clustering, no gap analysis. An `enrichment_worker` runs extraction in the background but only emits cards.

### Lifecycle Management
No decay / forgetting (a 1-day-half-life factor in `search/tantivy.rs:220` is a *recency sort boost*, not memory decay). **Versioning is the real lifecycle feature**: `MemoryCard::supersedes()` (memory_card.rs:248) — same `version_key`, `Updates`/`Retracts` relation supersedes when its `event_date` (fallback `document_date`) is newer. Point-in-time reads: `as_of_frame` / `as_of_ts` on `SearchRequest`, plus `timeline()` and session `replay`. Raw frames are never mutated; supersession is expressed at the card layer.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| "+35% SOTA on LoCoMo" | README highlight only; no absolute score, no eval harness in repo | **Unvalidated / not comparable.** Relative to an unnamed baseline; repo's `synthesize_answer()` concatenates top-3 snippets verbatim, so any QA-judge number needs an *external* LLM reader not present here. Not comparable to Somnigraph's absolute 85.1 LoCoMo QA. |
| "+76% multi-hop, +56% temporal vs industry average" | README only | Unvalidated; no numbers, no ablation. |
| "0.025ms P50 / 1,372× throughput" | `benches/*.rs` microbenchmarks | Plausible for in-file vector/lex scan (mmap, SIMD `src/simd.rs`), but this is *retrieval latency*, not QA accuracy. |
| Hybrid BM25+vector search | `ask.rs` RRF fusion (real); base `search()` is lex-only | **Plausible but mislocated** — exists only in the `ask()` RAG path, not the core search API or the misnamed `graph_search::hybrid_search`. |
| Entity extraction / Memory Cards (SPO) | `rules.rs` regex engine, real | **Validated but regex-only** — LLM extraction mode is a no-op stub. |
| Portable single-file, no DB | `MV2_SPEC.md`, real | **Validated.** Genuine engineering; the actual differentiator. |
| Time-travel / point-in-time | `as_of_frame`/`as_of_ts`, `timeline`, `replay` | **Validated.** Real and well-formed. |

---

## Relevance to Somnigraph

### What Memvid does that Somnigraph doesn't
- **Bitemporal fact tracking** — `event_date` (validity time) vs `document_date` (record time) as *separate* fields on each memory card. Somnigraph's `db.py` schema has `valid_from`/`valid_until` (validity interval) but no distinct "when we recorded it" axis, so it can't answer "what did I believe on date T" vs "what was true on date T."
- **Point-in-time retrieval surface** — `as_of_ts` reconstructs the memory state as of a timestamp. Somnigraph stores validity intervals but `tools.py::recall` has no "recall as of date T" parameter.
- **Slot-based supersession at write/query time** (`version_key = entity:slot`, `Updates`/`Retracts`). Somnigraph derives `evolves`/`revision` edges only later, during NREM sleep (`sleep_nrem.py`) — Memvid resolves the newest value synchronously per entity:slot.
- **Portable single-file container** with embedded indices — orthogonal to Somnigraph's SQLite design, not a gap.

### What Somnigraph does better
- **Learned retrieval.** Somnigraph's 26-feature LightGBM reranker (`reranker.py`, NDCG 0.7958) vs Memvid's *nonexistent* concrete reranker (trait + mock) and cosine-plus-heuristics `ask()` stack.
- **Tuned fusion.** RRF `k=14` Bayesian-optimized (`scoring.py`) vs Memvid's fixed `RRF_K=60`.
- **Real graph-conditioned retrieval.** PPR expansion over typed edges (`scoring.py`) vs Memvid's keyword-pattern → O(1) slot lookup with no traversal or blending.
- **Offline LLM consolidation** (NREM/REM sleep) — Memvid has none.
- **Explicit feedback loop** with measured GT correlation (Spearman r=0.70) — Memvid has none.
- **LLM-mediated write path** — Memvid's extraction is regex-only (LLM stub).

---

## Worth Stealing (ranked)

### 1. Bitemporal card fields: event_date vs document_date (Medium)
**What**: Store two timestamps per memory — when the fact *became true* and when it was *recorded* — instead of a single validity interval.
**Why**: Somnigraph's `valid_from/valid_until` conflates these. Bitemporality lets recall answer "what was true as of T" separately from "what I had learned by T," which matters for correcting backdated facts and for honest-accounting audit trails.
**How**: Add a `recorded_at` (or `document_date`) column alongside `valid_from` in `db.py`; `sleep_nrem.py` supersession/`evolves` edge creation compares event-time, not insert-time; optional `as_of` filter in `tools.py::recall`.

### 2. `as_of` point-in-time recall parameter (Low–Medium)
**What**: A `recall(query, as_of=<ts>)` mode that filters to memories whose validity interval contains T and returns the value that was current then.
**Why**: Somnigraph already has the interval columns; the retrieval surface just doesn't expose time-travel. Cheap given existing schema, and useful for reconstructing past belief states.
**How**: Predicate in the `tools.py` recall SQL (`valid_from <= T AND (valid_until IS NULL OR valid_until > T)`); pick per-`version_key` newest-by-event-date. No new storage.

### 3. Question-type-conditioned retrieval widening (Low, note-only)
**What**: `ask()` widens candidate `top_k` and post-processes by detected question class — aggregation → dedup-by-session diversification; recency/update → promote temporal extremes.
**Why**: Somnigraph's recall is one-size-fits-all; aggregation/recency queries could benefit from session-diversification and recency promotion.
**How**: A lightweight classifier in `tools.py::recall` gating a diversify/promote pass. Caveat: these are LoCoMo-shaped heuristics (Goodhart risk) and Somnigraph prefers learned signals — treat as a probe, not a keeper.

---

## Not Useful For Us

- **Single-file `.mv2` container format** — Somnigraph is single-user on SQLite; portability across machines isn't a goal, and SQLite already gives durable local storage.
- **Regex `RulesEngine` extraction** — brittle keyword patterns ("who works at"); Somnigraph's write path is human/LLM-authored, not scraped from documents.
- **Multimodal ingestion (Whisper/CLIP/PDF/xlsx), encryption capsules, ACL, PII** — document-RAG surface irrelevant to a personal Claude memory.
- **Keyword-pattern `QueryPlanner`** — inferior to learned reranking Somnigraph already has.

---

## Connections

- **Convergent RRF fusion**: independently arrives at reciprocal-rank fusion of lexical+vector like Somnigraph's `scoring.py`, but with an untuned `k=60` and no learned second stage — reinforces the Phase-18 finding (see `ai-memory-comparison.md`, `byterover.md`) that LoCoMo leaders win on **write-path structuring**, not retrieval sophistication. Here the structuring is regex SPO cards, not LLM extraction.
- **Bitemporality / supersession** echoes the temporal-edge / supersession patterns in `memv.md` and Graphiti-style systems (see `similar-systems.md`); Memvid's contribution is doing it at the `entity:slot` card layer synchronously rather than via an offline pass.
- **Extractive "answer synthesis"** (snippet concatenation) is the same shortcut flagged in other packaging-over-engine repos — the published QA number depends on an external reader model, mirroring the `mirix.md` / `agentmemory.md` judge-model caveats.

---

## Summary Assessment

Memvid's genuine contribution is **engineering, not retrieval science**: a portable, mmap-friendly single-file container (`.mv2`) that bundles frames, HNSW/BM25 indices, bitemporal SPO cards, encryption, and ACL, with sub-millisecond in-file scans. As a *format* it is polished. As a *memory intelligence* it is thin: extraction is regex-only (the LLM mode is a literal no-op stub), the "graph" is keyword-pattern slot lookup with no traversal, the advertised cross-encoder/LLM rerankers do not exist, and there is no consolidation, decay, or feedback loop.

The single most useful idea for Somnigraph is **bitemporality** — separating "when the fact was true" (`event_date`) from "when it was recorded" (`document_date`), with an `as_of` point-in-time read. Somnigraph already has validity intervals in `db.py`; adding the record-time axis and exposing a time-travel recall parameter is low-cost and improves the honest-accounting/audit story without touching the reranker.

What's overhyped: the "+35% SOTA on LoCoMo" banner. There is **no absolute score, no methodology, and no eval harness in the repo**, and the in-tree `synthesize_answer()` merely concatenates the top-3 retrieved snippets — so any LLM-judged QA number is produced by an external reader Memvid doesn't ship. It is a relative claim against an unnamed baseline and is **not comparable** to Somnigraph's measured 85.1% LoCoMo QA. The carsteneu evidence file's corrections (hybrid=true, entities=true) are directionally right but under-specified: hybrid fusion lives only in the `ask()` RAG path (not the base `search()` API, and *not* the misnamed `graph_search::hybrid_search`), entities are regex-extracted, and — contra the evidence file's "supersede ❌ absent" — supersession genuinely exists at the MemoryCard layer (`version_relation` + `supersedes()`), just not for raw append-only frames.
