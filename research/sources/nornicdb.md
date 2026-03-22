# NornicDB Analysis (Agent Output)

*Generated 2026-03-22 by Opus agent reading local clone*

---

## 1. Architecture Overview

**Repo**: https://github.com/orneryd/NornicDB
**Stars**: ~312 (as of 2026-03-22)
**License**: MIT (with defensive patent non-assertion grant in PATENTS.md)
**Language**: Go (1.26+), ~522K lines of Go across ~1,444 files (520 test files)
**Created**: 2025-12-06, 771 commits to date
**Description**: "The Graph Database That Learns" — a Neo4j-compatible graph database with native vector search, biological memory decay, auto-relationship inference, and GPU acceleration.

**Core design**: NornicDB is a full graph database engine, not a memory server. It implements the Neo4j Bolt protocol and Cypher query language, so existing Neo4j drivers work unchanged. On top of this, it adds AI-native features: HNSW vector search, BM25 full-text search with RRF fusion, three-tier biological decay, automatic relationship discovery (Auto-TLP), cross-encoder reranking, MMR diversity, and a canonical ledger model with temporal validity.

**Storage**: Badger KV (dgraph-io/badger/v4) with async write engine, node/edge caching, and optional disk-backed vector storage. Everything runs in-process — no external database dependencies.

**Module organization** (key packages under `pkg/`):
- `storage/` — Badger-backed node/edge persistence, schema management, temporal indexing, async engine
- `search/` — Hybrid search service (HNSW + BM25 + RRF fusion + cross-encoder + MMR), ~5,400 lines
- `decay/` — Three-tier memory decay with promotion
- `inference/` — Auto-relationship engine (similarity, co-access, temporal, transitive)
- `linkpredict/` — Topological link prediction (Adamic-Adar, Jaccard, etc.) with hybrid scoring
- `temporal/` — Kalman-filtered access pattern tracking, decay integration, session detection
- `mcp/` — MCP server with 6 tools (store, recall, discover, link, task, tasks)
- `cypher/` — Cypher parser and query engine
- `bolt/` — Neo4j Bolt protocol implementation
- `embed/` — Embedding providers (local GGUF, Ollama, OpenAI-compatible)
- `gpu/` — Metal/CUDA/Vulkan acceleration, k-means clustering
- `heimdall/` — Built-in AI assistant with plugin system
- `nornicdb/` — Core database type and lifecycle

---

## 2. Memory Model

### Three-Tier Decay

Defined in `pkg/decay/decay.go`. Three tiers with exponential decay:

| Tier | Half-Life | Lambda (per hour) | Base Importance | Use Case |
|------|-----------|-------------------|----------------|----------|
| Episodic | 7 days | 0.00412 | 0.3 | Chat context, sessions |
| Semantic | 69 days | 0.000418 | 0.6 | Facts, decisions |
| Procedural | 693 days | 0.0000417 | 0.9 | Skills, patterns |

Score formula: `score = RecencyWeight * exp(-lambda * hours) + FrequencyWeight * log(1 + accessCount) / log(101) + ImportanceWeight * importance`

Default weights: Recency 0.4, Frequency 0.3, Importance 0.3.

### Tier Promotion

Memories can be promoted up tiers based on access patterns (`PromoteTier()` in `pkg/decay/decay.go`):
- **Episodic to Semantic**: 10+ accesses AND 3+ days old
- **Semantic to Procedural**: 50+ accesses AND 30+ days old

Promotion is configurable and can be disabled. This is a meaningful feature — Somnigraph has per-memory decay rates but no automatic promotion between tiers.

### Archival

Memories with decay score below threshold (default 0.05) are flagged for archival. Background recalculation runs on a configurable interval (default 1 hour).

### Adaptive Decay via Temporal Integration

`pkg/temporal/decay_integration.go` modifies decay rates based on Kalman-filtered access patterns:
- Frequently accessed nodes: decay multiplier as low as 0.1 (10x slower)
- Rarely accessed nodes: decay multiplier up to 2.0 (2x faster)
- Daily pattern detected: additional slowdown
- Burst access: near-zero decay during active use
- Cold nodes (no access in 2+ weeks): faster decay, archive candidate

This is more sophisticated than Somnigraph's static per-memory decay rate, but operates on a different axis — NornicDB adjusts decay based on access velocity rather than explicit feedback.

---

## 3. Retrieval Pipeline

Implemented in `pkg/search/search.go` (5,436 lines). The pipeline has up to four stages:

### Stage 1a: Vector Search (HNSW)
- HNSW index for approximate nearest neighbor search
- GPU-accelerated brute-force option (Metal/CUDA/Vulkan)
- Optional IVF-PQ compressed index for large datasets
- Optional k-means cluster routing (IVF-HNSW)
- BM25-seeded HNSW insertion ordering (claimed 2.7x faster build)

### Stage 1b: BM25 Full-Text Search
- Custom BM25 implementation (v1 and v2 engines)
- Priority property weighting (content, text, title, name, description)
- Phrase search support

### Stage 2: RRF Fusion
Standard Reciprocal Rank Fusion (`fuseRRF()` in search.go):
```
RRF_score = vectorWeight / (k + vector_rank) + bm25Weight / (k + bm25_rank)
```
Default k=60, equal weights. Adaptive weighting based on query length: short queries (1-2 words) boost BM25 weight to 1.5, long queries (6+ words) boost vector weight to 1.5 (`GetAdaptiveRRFConfig()`).

### Stage 3: Cross-Encoder Reranking (Optional)
Defined in `pkg/search/rerank.go`. Calls an external reranking API (Cohere, HuggingFace TEI, or local model). Top-K candidates (default 100) are re-scored. Fail-open: falls back to original ranking on API error. The `Reranker` interface allows pluggable implementations beyond cross-encoder.

### Stage 4: MMR Diversification (Optional)
Maximal Marginal Relevance in `applyMMR()`. Lambda parameter controls relevance/diversity balance (default 0.7 = 70% relevance, 30% diversity). Uses embedding similarity between selected results to penalize redundancy.

### Caching
LRU search result cache with TTL. Invalidated on index mutations (IndexNode/RemoveNode). Cache key includes query text + all search options.

---

## 4. Relationship Discovery (Auto-TLP)

NornicDB has two complementary relationship inference systems:

### Semantic Inference (`pkg/inference/`)
Four methods, triggered on store and access events:
1. **Embedding similarity** — Nodes above threshold (default 0.82) get edge suggestions
2. **Co-access patterns** — Nodes accessed within 30s window, minimum 3 co-accesses
3. **Temporal proximity** — Same-session nodes (30-minute window) linked
4. **Transitive inference** — A→B and B→C suggests A→C (minimum confidence 0.5)

Additional infrastructure:
- **Cooldown table** — Prevents rapid re-materialization of edges
- **Evidence buffer** — Requires multiple signals before materializing an edge
- **Edge meta store** — Logs edge provenance for audit trails
- **Heimdall QC** — Optional SLM-based quality control on edge creation
- **Kalman adapter** — Smoothed confidence and temporal pattern detection
- **Cluster integration** — GPU k-means for accelerated similarity search at scale

### Topological Link Prediction (`pkg/linkpredict/`)
Five classic graph algorithms:
1. **Common Neighbors** — `|N(u) ∩ N(v)|`
2. **Jaccard Coefficient** — Normalized overlap
3. **Adamic-Adar** — Weights rare connections higher
4. **Resource Allocation** — Inverse degree weighting
5. **Preferential Attachment** — Degree product

**Hybrid scoring** (`HybridScorer`) blends topological and semantic signals with configurable weights (default 0.5/0.5). Ensemble mode runs all 5 topology algorithms with learned weights (Adamic-Adar: 0.3, Resource Allocation: 0.25, Jaccard: 0.2, Preferential Attachment: 0.15, Common Neighbors: 0.1).

Exposed via Neo4j GDS-compatible Cypher procedures (`gds.linkPrediction.*.stream()`).

---

## 5. Consolidation & Maintenance

NornicDB lacks a dedicated "sleep" or offline consolidation phase like Somnigraph's. Instead:

- **Decay recalculation** runs on a background timer (default 1 hour)
- **Temporal tracker** detects access patterns via Kalman filtering and adjusts decay modifiers
- **Auto-relationship inference** fires on store/access events (online, not batched)
- **HNSW maintenance** rebuilds index periodically when deferred mutations accumulate
- **BM25/vector index persistence** debounced saves to disk after mutations

There is no LLM-mediated consolidation (no merge, archive, annotate, rewrite decisions). The system relies on automatic decay + archival threshold + access-pattern-driven decay adjustment.

---

## 6. Comparison to Somnigraph

### What NornicDB has that we don't

**Cross-encoder reranking as pipeline stage.** NornicDB's `Reranker` interface in `pkg/search/rerank.go` supports external cross-encoder APIs (Cohere, HuggingFace TEI) as a post-RRF refinement step. Somnigraph's reranker is a learned LightGBM model using extracted features — different approach, arguably more self-contained, but NornicDB's cross-encoder can capture fine-grained query-document interactions that feature-based models miss.

**MMR diversity as post-reranking step.** `applyMMR()` in search.go re-ranks results to balance relevance with diversity using embedding similarity. Somnigraph tested an MMR-style diversity feature (`max_sim_to_higher`) in the reranker and found it hurt NDCG by 0.3% — a pointwise model can't coordinate which results to keep vs. drop (documented in `docs/experiments.md`). NornicDB applies MMR as a separate post-reranking pass, which avoids the pointwise coordination problem but hasn't been evaluated for quality impact.

**Adaptive RRF weights.** `GetAdaptiveRRFConfig()` adjusts vector vs. BM25 weights based on query length. Short queries boost BM25, long queries boost vector. Somnigraph uses fixed RRF weighting regardless of query characteristics.

**Tier promotion.** Automatic episodic → semantic → procedural promotion based on access count and age thresholds. Somnigraph's decay rates are set at creation time and only adjusted by explicit feedback or sleep decisions.

**Kalman-filtered temporal tracking.** `pkg/temporal/tracker.go` uses Kalman velocity filters to predict access patterns, detect sessions, and adaptively modify decay rates. This is a more principled temporal model than Somnigraph's simple last_accessed timestamp.

**Topological link prediction.** Five classical graph algorithms (Adamic-Adar, Jaccard, etc.) with hybrid topology+semantic scoring. Somnigraph's graph edges are created by explicit user action, LLM decision during sleep, or Hebbian co-retrieval — no topological heuristics.

**GPU acceleration.** Metal/CUDA/Vulkan pathways for vector search and k-means clustering. Somnigraph is CPU-only.

**Neo4j Bolt compatibility.** Any Neo4j driver works against NornicDB. Somnigraph is MCP-only.

**Canonical Ledger Model.** Versioned facts with temporal validity windows, as-of reads, queryable mutation log with receipts (`pkg/storage/badger_temporal_index.go`, `pkg/storage/receipt.go`). Somnigraph has no versioned fact model — memories are current state, not temporal records.

### What we have that they don't

**Learned reranker with ground truth.** LightGBM model trained on 1032 human-judged queries (NDCG=0.7958). NornicDB's cross-encoder calls an external API — there's no ground truth collection, no evaluation framework, no evidence the reranking actually helps on their workload.

**Retrieval feedback loop.** `recall_feedback()` creates a gradient signal. EWMA aggregation, UCB exploration bonus, decay adjustment, edge strengthening, theme enrichment. NornicDB has no mechanism for the user/agent to tell it which results were useful.

**Sleep pipeline with LLM consolidation.** Three-phase offline processing (NREM classification, REM per-memory LLM decisions, archiving). NornicDB decays and archives automatically but makes no intelligent decisions about merging, rewriting, or annotating memories.

**Hebbian PMI edge strengthening.** Co-retrieval patterns strengthen edges via pointwise mutual information. NornicDB has co-access detection but it creates new edges — it doesn't strengthen existing ones based on retrieval co-occurrence.

**Theme normalization and controlled vocabulary.** Canonical theme taxonomy with normalization. NornicDB uses free-form tags without vocabulary control.

**Tuning and evaluation infrastructure.** Ground truth collection, Optuna hyperparameter optimization, probe_recall scripts, feature importance analysis. NornicDB has benchmark comparisons against Neo4j (LDBC) but no retrieval quality evaluation.

**UCB exploration bonus.** Bayesian exploration for under-retrieved memories. NornicDB has no mechanism to surface memories that haven't been tested.

**Enriched embeddings.** Content + category + themes + summary concatenated before embedding. NornicDB embeds raw text content.

### Architectural trade-offs

| Dimension | NornicDB | Somnigraph | Trade-off |
|-----------|----------|-----------|-----------|
| Language | Go (compiled, single binary) | Python (interpreted, MCP server) | Performance vs. ecosystem/iteration speed |
| Storage | Badger KV (in-process) | SQLite + sqlite-vec + FTS5 | Both zero-infrastructure. Badger scales better; SQLite is more inspectable. |
| Vector search | HNSW (in-process, GPU-optional) | sqlite-vec (CPU, exact by default) | NornicDB faster at scale; Somnigraph exact at small scale. |
| BM25 | Custom Go implementation | SQLite FTS5 | Both work. NornicDB's is more configurable; Somnigraph's is battle-tested. |
| RRF fusion | Adaptive weights by query length | Fixed weights | NornicDB adapts; Somnigraph relies on reranker to learn optimal weighting. |
| Reranking | External cross-encoder API | Learned LightGBM (26 features) | NornicDB: potentially higher quality per-pair. Somnigraph: self-contained, evaluated. |
| Decay | Three fixed tiers + Kalman adaptation | Per-memory configurable rate + feedback | NornicDB: biologically motivated structure. Somnigraph: fine-grained per-memory control. |
| Graph | Cypher/Bolt property graph | SQLite edges with typed relationships | NornicDB: full graph database. Somnigraph: lightweight edge table. |
| Consolidation | Continuous (timer-based, no LLM) | Periodic sleep (LLM-mediated decisions) | NornicDB: always current, no LLM cost. Somnigraph: deeper decisions, higher quality. |
| Feedback | None | recall_feedback() with EWMA + UCB | Somnigraph adapts to usage; NornicDB does not. |
| Embeddings | Local (BGE-M3 GGUF, 1024d) or external | Cloud (OpenAI, 1536d) | Privacy vs. quality/convenience |
| Scope | General-purpose graph DB | Claude Code memory system | NornicDB is infrastructure; Somnigraph is application. |

---

## 7. Worth Adopting?

**MMR diversity as post-reranking step**: Somnigraph already tested an MMR-style diversity feature inside the pointwise reranker — it hurt NDCG because a pointwise model can't coordinate selections (see `docs/experiments.md`). NornicDB's approach (MMR as a separate post-reranking pass) avoids this problem. Worth revisiting if redundancy becomes a measured issue, but requires a listwise mechanism, not a feature addition.

**Adaptive RRF weights**: Maybe. The query-length heuristic is simple but principled — short keyword-like queries benefit more from BM25, longer natural language queries from vectors. However, Somnigraph's reranker already learns the optimal blend implicitly through features like `fts_ranked` and `vec_ranked`. The adaptive weights might be redundant if the reranker handles this well. Worth testing.

**Tier promotion**: Interesting conceptually but potentially redundant with Somnigraph's feedback-adjusted decay. Somnigraph already adjusts per-memory decay rates via feedback — automatic promotion would add a parallel mechanism for the same goal.

**Kalman temporal tracking**: The Kalman velocity filter for access patterns is elegant, but Somnigraph's feedback loop captures similar information (which memories are useful) more directly. The access-rate prediction and session detection are novel, but unclear how much they'd help when the user explicitly rates results.

**Cross-encoder reranking**: Not worth adopting directly. Somnigraph's LightGBM reranker is self-contained and evaluated; adding a cross-encoder dependency would require an external service and evaluation infrastructure to verify it helps.

**Topological link prediction**: Worth watching but not adopting. Somnigraph's graph is relatively sparse and edges carry semantic types (support, contradict, evolve). Topological heuristics work best on denser graphs where structural patterns are informative.

---

## 8. Worth Watching

**Scale behavior.** NornicDB is designed for much larger datasets than Somnigraph currently handles. If Somnigraph's memory store grows past ~10K memories, NornicDB's HNSW performance characteristics and k-means cluster routing would be informative references.

**Cross-encoder quality impact.** If NornicDB publishes retrieval quality evaluations (not just throughput benchmarks), the cross-encoder reranking delta would be useful data for whether Somnigraph should consider adding one.

**Adaptive RRF in practice.** The query-length heuristic is simple. If community usage reveals more sophisticated adaptive strategies, that could inform Somnigraph's approach.

**Kalman temporal patterns.** The Kalman filter approach to access pattern detection is novel in this space. If it proves useful for decay adjustment, it could complement Somnigraph's feedback-based approach.

---

## 9. Key Claims

1. **"12x-52x faster than Neo4j"** on LDBC benchmarks. *Evidence*: Published benchmark results in `docs/performance/benchmarks-vs-neo4j.md`. Plausible for an in-process Go database vs. a JVM-based server, but the comparison is hardware-specific (M3 Max) and workload-specific. No independent reproduction.

2. **"Neo4j-compatible, drop-in replacement"** via Bolt + Cypher. *Evidence*: The Bolt protocol and Cypher parser are substantially implemented (52 functions, 950+ APOC procedures). True for basic operations; complex Cypher queries may hit parser gaps (recent commits show ongoing Cypher fixes).

3. **"HNSW build 2.7x faster via BM25-seeded insertion order."** *Evidence*: Referenced in README with a dev.to blog post. The mechanism (reducing traversal waste during construction via better insertion ordering) is sound. Internal benchmark only.

4. **Three-tier biological decay "mimics human cognition."** *Evidence*: The three tiers (7/69/693-day half-lives) are a reasonable model inspired by cognitive science literature, but the specific half-life values are design choices, not empirically derived from cognitive data. The promotion thresholds (10 accesses / 3 days, 50 accesses / 30 days) are also heuristic.

5. **"Auto-TLP: automatic relationship discovery."** *Evidence*: Substantially implemented across `pkg/inference/` and `pkg/linkpredict/`. Four semantic methods + five topological algorithms + hybrid scoring. Whether auto-discovered relationships are actually useful (precision/recall) is not evaluated.

---

## 10. Relevance to Somnigraph

**Medium-high** for architectural comparison. NornicDB and Somnigraph solve overlapping problems (memory decay, hybrid retrieval, graph relationships) from fundamentally different starting points: NornicDB is a general-purpose graph database that added memory features; Somnigraph is a purpose-built memory system for a specific agent. The comparison is instructive for understanding trade-offs between generality and specialization.

**Medium** for borrowable ideas. MMR diversity and adaptive RRF weights are the most actionable. The tier promotion and Kalman temporal tracking are interesting but potentially redundant with Somnigraph's feedback loop.

**Low** for direct code reuse. Different language (Go vs. Python), different storage (Badger vs. SQLite), different architecture (full graph DB vs. MCP memory server).

The most valuable insight: NornicDB's Auto-TLP shows what automatic relationship discovery looks like when you combine classical graph theory (Adamic-Adar, Jaccard) with semantic signals (embedding similarity, co-access). Somnigraph creates edges through human action, LLM judgment, and Hebbian co-retrieval — all empirically grounded. NornicDB's topological heuristics are a different paradigm: structure-based inference that works without any human or LLM involvement. Whether that produces useful edges at Somnigraph's scale (~600 memories, sparse graph) is an open question.
