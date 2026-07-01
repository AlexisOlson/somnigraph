# Wax — Swift/Metal single-file RAG-memory engine (perf-first, deterministic, offline)

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

Wax (`christopherkarani/Wax`, Swift 6.1, Apache-2.0, ~744★) is a **performance engine** first and a "memory system" second. The center of gravity is `WaxCore`: a single-`.wax`-file container (dual headers, WAL crash recovery, LZ4 frames, TOC) with SQLite FTS5 for text/EAV facts and a Metal-accelerated HNSW vector index. It ships as a Swift library + CLI daemon + MCP server (`WaxMCPServer`, 7473-line test file). The "memory" layer (`Sources/Wax/Orchestrator`, `Broker`, `Enrichment`, `Maintenance`, `UnifiedSearch`) sits on top. Everything is **on-device, offline, deterministic** — no LLM in any hot or background path. CoreML embedders bundled (all-MiniLM-L6-v2, snowflake-arctic-embed-s).

### Storage & Schema
Frame-based container. A memory unit is a `Frame` with role (`document`/`chunk`), parentId, chunkIndex, `searchText`, and a free-form `Metadata` string map (`session_id`, `wax.content.hash`, project, etc.). Separately, a **bitemporal structured store** (`WaxCore/StructuredMemory`) holds EAV facts: `EntityKey`/`PredicateKey`/`FactValue`, with `StructuredMemoryAsOf { systemTimeMs, validTimeMs }` — genuine two-axis time (transaction time + valid time), plus `StructuredEdges`/`StructuredEvidence`. Schema fields per the evidence file (~6) is a fair estimate; the real answer is "varies by tool."

### Memory Types
Two parallel models: (1) unstructured chunked documents (RAG), (2) structured EAV facts with entities/predicates/edges. Session-scoped ephemeral vs broker-managed persistent is a lifecycle distinction, not a typed taxonomy. No episodic/semantic/procedural/reflection categories like Somnigraph.

### Write Path
`MemoryOrchestrator.remember()` (`MemoryOrchestrator.swift:342`): chunk → embed → dedup. **Dedup is real**: content-SHA hash (`ContentHasher`) + `rememberDedupProbe(contentHash, expectedChunkCount, embeddingIdentity)` — if a complete prior copy exists, the write is skipped (`:355`). Enrichment (`Enrichment/`) is a deterministic async pipeline: `KeywordExtractor` (TF + stopwords + technical-identifier heuristics), `EntityExtractor`. **No autoExtract from conversation** — the agent must explicitly call `remember`/`save`/`handoff`. **No quality/salience gate on write** (importance is computed later, at read/maintenance time).

### Retrieval
`UnifiedSearch.swift` (1432 lines). Channels: FTS5 BM25 + Metal HNSW vector. Fusion is **RRF** (`HybridSearch.rrfFusion`, rank-based, k default 60) with **query-adaptive weights** (`AdaptiveFusionConfig`): a `RuleBasedQueryClassifier` buckets the query into factual/semantic/temporal/exploratory by keyword heuristics, then applies preset BM25/vector/temporal weights (factual 0.7/0.3, semantic 0.3/0.7, temporal 0.25/0.25/0.5). Post-fusion there are **two hand-tuned heuristic rerankers**: `semanticMemoryRerank` and `intentAwareRerank` (`:856`). The latter is a large pile of hardcoded scoring rules — entity/year/date-key coverage, quoted-phrase exact match, and **LoCoMo-shaped literals** ("moved to", "public launch is", "person18"/"atlas10" numeric-entity disambiguation, penalties for "allergic"/"peanut"). **No learned reranker, no RRF-k tuning, no feedback loop.**

### Consolidation / Processing
No sleep, no LLM merge, no relationship discovery. "Consolidation" = **deterministic extractive summarization**: `ExtractiveSurrogateGenerator` (`algorithmID "extractive_v1"`) segments text, scores sentences, and selects via **MMR** into hierarchical tiers `SurrogateTiers { full ~100tok, gist ~25tok, micro ~8tok }`. `Maintenance/` runs live-set rewrite/index compaction and surrogate optimization. `session_synthesize`/`handoff` pass cross-session context but do no semantic consolidation.

### Lifecycle Management
`ImportanceScorer` (`RAG/ImportanceScorer.swift`): exponential **decay** on age (168h half-life) + recency (24h half-life) + log-frequency, weighted 0.3/0.3/0.4. But it drives **tier selection** (`SurrogateTierSelector`) and context-budget packing (`FastRAGContextBuilder`), **not eviction** — nothing is forgotten by decay. Versioning is real but **only for structured facts**: `VersionRelation { sets, updates, extends, retracts }` where `updates`/`retracts` supersede, queryable "as-of" via the bitemporal axes. `fact_retract` retracts facts; there is no general memory `forget`/`delete` tool.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| ~6ms p95 hybrid recall, ~9ms cold open | `Resources/docs/benchmarks/` perf results | Plausible; Metal HNSW + single-file is genuinely fast. Latency only. |
| Hybrid text+vector search | `UnifiedSearch.swift`, `HybridSearch.rrfFusion` | Validated in code. |
| Query-adaptive fusion | `AdaptiveFusionConfig` + `RuleBasedQueryClassifier` | Validated; weights are hand-set presets, not learned. |
| "100% on-device, no API keys" | CoreML embedders bundled; no network in code paths | Validated — strongest, most honest claim. |
| Deduplication | `rememberDedupProbe` content-hash + embedding-identity | Validated. |
| Retrieval/QA accuracy | **none reported** — no LoCoMo/recall/NDCG/MRR anywhere | **Unvalidated.** Only latency benchmarks exist. |

---

## Relevance to Somnigraph

### What Wax does that Somnigraph doesn't
- **Bitemporal structured fact store** (`StructuredMemoryAsOf`: system-time + valid-time, as-of queries). Somnigraph's `db.py` has only single-axis `valid_from`/`valid_until`; Wax can answer "what did we believe was true on date X, as recorded by date Y."
- **Write-time deduplication** (content-hash + embedding-identity probe). Somnigraph's `tools.py remember()` relies on the memory server's 0.9-similarity dedupe; Wax's exact-hash + chunk-count probe is cheaper and complementary.
- **Query-adaptive fusion weights** by rule-classified query type. Somnigraph uses a single Bayesian-tuned RRF k=14 for all queries, then leans on the learned reranker.
- **Deterministic extractive tiering under a token budget** — `SurrogateTierSelector` picks full/gist/micro per frame by importance to fit a context budget. Somnigraph has detail/summary/gestalt *layers* but no query-time budget-aware tier selection.
- **Raw performance / Metal HNSW / single-file WAL container** — a whole engineering axis Somnigraph (SQLite + sqlite-vec, single-user) doesn't compete on and doesn't need to.

### What Somnigraph does better
- **Learned retrieval quality.** Somnigraph's 26-feature LightGBM reranker (`reranker.py`, NDCG 0.7958) vs Wax's hand-tuned `intentAwareRerank` heuristic stack. Wax's reranker is brittle and, tellingly, hardcodes LoCoMo-specific literals.
- **Feedback loop.** Somnigraph has explicit per-query utility ratings, EWMA/UCB, Spearman r=0.70 with GT. Wax has **none** — no signal from whether a recall was useful.
- **LLM-mediated sleep consolidation** (`sleep_nrem.py`/`sleep_rem.py`): typed edges, contradiction detection, merge/archive, gap-driven questions. Wax's "consolidation" is extractive MMR compression — no semantic relationship discovery.
- **Graph-conditioned retrieval** (PPR expansion, betweenness feature). Wax has structured EAV edges but no PPR/graph expansion into unstructured recall.
- **Measured end-to-end QA** (85.1% LoCoMo). Wax reports only latency.

---

## Worth Stealing (ranked)

### 1. Bitemporal axis for the fact/temporal layer (High)
**What**: Split time into transaction-time (when recorded) and valid-time (when true), with an explicit "as-of" query (`StructuredMemoryAsOf`).
**Why**: Somnigraph's `valid_from`/`valid_until` conflates these. NREM contradiction/evolves edges would be sharper if "we learned X was wrong on date Y" were distinguishable from "X stopped being true on date Y" — this is exactly the temporal-evolution case the sleep pipeline classifies.
**How**: Add a `recorded_at` alongside `valid_from`/`valid_until` in `db.py`; let `sleep_nrem.py` set `valid_until` (fact expired) vs a new `superseded_at` (belief corrected) distinctly. Mostly schema + one classifier branch; retrieval largely unaffected.

### 2. Importance-driven tier selection under a token budget (Medium)
**What**: At retrieval time, pick each memory's rendering tier (full/gist/micro) from an importance score (age/recency/frequency decay + query specificity boost) to fit a fixed context budget — `SurrogateTierSelector`/`FastRAGContextBuilder`.
**Why**: Somnigraph has detail/summary/gestalt layers but injects a fixed granularity. Budget-aware tier selection would let low-value hits ride along as micro-surrogates instead of being dropped — relevant to the proactive-injection work (`docs/proactive-injection.md`), where compactness is the whole point.
**How**: A selector in `scoring.py`/`tools.py` that, given a token budget and per-result reranker score + decay, emits the coarsest tier that keeps the budget. Reuses existing layer fields.

### 3. Write-time exact-dup probe as a cheap pre-filter (Low)
**What**: Content-hash + chunk-count + embedding-identity probe before the expensive embed/similarity dedupe.
**Why**: Cheap guard against re-remembering identical content (common with auto-capture/pending flows).
**How**: Hash-index check in `tools.py remember()` short-circuiting before the 0.9-similarity path.

---

## Not Useful For Us

### Metal HNSW / single-file container / WAL engineering
Somnigraph is single-user on SQLite + sqlite-vec; the perf ceiling Wax targets isn't a Somnigraph constraint. Pure re-platforming cost, no research payoff.

### The `intentAwareRerank` heuristic stack
A brittle, benchmark-shaped pile of hardcoded rules — the exact hand-tuned-formula regime Somnigraph already retired in favor of the learned reranker. Anti-pattern, not asset.

### Session/handoff MCP surface, markdown sync, WaxRepo TUI, Photo/Video RAG
Product surface for coding-agent workflows, orthogonal to Somnigraph's retrieval-quality research agenda.

---

## Connections
- **Single-file container + WAL + LZ4**: convergent with Memvid's `.mv2` format (the evidence file notes this); a packaging choice, not a memory idea.
- **Extractive MMR surrogates instead of LLM summaries**: shares the "deterministic write-path enrichment beats fancy retrieval" thesis from the Phase 18 sweep (ByteRover BM25-only, agentmemory write-time grounding) — but Wax stops at extractive compression and never adds the semantic-merge step that sweep found the QA leaders win on.
- **Query-adaptive fusion by query class**: same instinct as several systems that route factual→lexical / semantic→dense; Somnigraph's learned reranker subsumes this signal rather than hardcoding it.
- **Bitemporal facts**: the most distinctive contribution; no other profiled system in the corpus implements a true two-axis time model.

---

## Summary Assessment

Wax's core contribution is **engineering, not memory research**: a fast, deterministic, fully-offline single-file RAG engine with Metal-accelerated hybrid search, honest about being on-device. It is very well-built (extensive tests, WAL crash recovery, CI quality gates) and would be a strong choice for someone who wants a local memory store with no cloud dependency. But as a *memory system* it is deliberately shallow: no learned ranking, no feedback loop, no LLM consolidation, no graph expansion, no forgetting. Its "intelligence" is a stack of hand-tuned heuristics.

The single most valuable idea for Somnigraph is the **bitemporal fact model** (transaction-time vs valid-time with as-of queries) — it would make the sleep pipeline's contradiction-vs-evolution distinction crisper at the schema level. Second is **budget-aware tier selection** feeding the proactive-injection compactness goal. Both are additive and modest; neither displaces anything.

What's overhyped, and the sharpest correction to the evidence file: Wax reports **zero retrieval-quality or QA numbers** — the only "benchmarks" are latency (6ms p95). Its shipped `intentAwareRerank` nonetheless hardcodes LoCoMo-specific literals ("moved to", "public launch is", numeric-entity disambiguation like person18/atlas10), i.e. it is tuned to a benchmark it never publishes results on. Any comparison of Wax to Somnigraph's 85.1% LoCoMo QA is category-invalid: there is no comparable number to compare against. Separately, the evidence file's README-only read marked **layeredMemory, decay, supersede, and dedup as absent** — all four exist in code (`SurrogateTiers`, `ImportanceScorer`, `VersionRelation`, `rememberDedupProbe`), just not surfaced in the README. That's the main evidence-vs-code discrepancy.
