# flexvec: SQL Vector Retrieval with Programmatic Embedding Modulation -- Analysis

*Generated 2026-03-24 by Opus 4.6 agent reading arXiv:2603.22587v1*

---

## Paper Overview

**Paper**: Damian Delmas (Independent Researcher, Vancouver, BC). "flexvec: SQL Vector Retrieval with Programmatic Embedding Modulation." arXiv:2603.22587v1, March 25, 2026. Preprint. 15 pages (10 main + 5 appendices). Code: https://github.com/damiandelmas/flexvec (MIT).

**Problem addressed**: Retrieval APIs for AI agents expose a fixed search endpoint with its own query language, hiding both the embedding matrix and the score array from the caller. The agent cannot compose filtering, scoring modifications, and post-processing in a single statement. Pre-filtering (before vector scoring) and post-filtering (after) each lose either recall or efficiency when metadata filters are involved.

**Core claim**: By loading the embedding matrix into memory as a numpy array and exposing it alongside the score array as a programmable surface, arithmetic operations (suppression, decay, centroid shift, trajectory blending, MMR diversity) can compose between scoring and selection. A "query materializer" integrates these operations into standard SQL as pseudo-functions, letting an AI agent write a single SQL statement that handles pre-filtering, vector scoring with modulations, and result composition -- without modifying the SQLite engine.

**Scale of evaluation**: Production corpus of 240,000 chunks (AI coding session history across 4,000 sessions) in a single SQLite database, with 128-dimensional Nomic Embed v1.5 embeddings (Matryoshka truncation). Behavioral validation on four BEIR datasets (SciFact 5,183 docs; NFCorpus 3,633; SCIDOCS 25,657; FiQA 57,638) with 30 queries per dataset. Scaling tested up to 1M chunks. No retrieval quality benchmarks (NDCG, MRR, recall) against baselines -- the evaluation is latency, algebraic correctness, and behavioral change (RBO, ILS, centroid similarity).

---

## Architecture / Method

### The Core Insight

Standard vector retrieval computes `scores = M @ q` and returns `top-K`. The score array is discarded after selection. flexvec exposes this array as a mutable surface: after the dot product, operations reshape scores before selection. This is the "Programmatic Embedding Modulation" (PEM) concept.

The key architectural move is separating *candidate selection* (which memories enter scoring) from *scoring* (how they're ranked) from *composition* (how results join back to metadata). These become three explicit SQL phases rather than one opaque API call.

### Three-Phase Pipeline

**Phase 1: SQL pre-filter.** A SQL subquery returns candidate chunk IDs. Standard SQLite filtering on metadata columns (type, project, time range). This determines *what* enters scoring. Example: `SELECT id FROM messages WHERE type = 'assistant' AND length(content) > 300`.

**Phase 2: numpy modulation.** Brute-force cosine similarity via matrix multiply on the pre-filtered candidates, followed by optional modulations that reshape scores. Passes top 500 scored candidates to Phase 3.

**Phase 3: SQL compose.** The outer query joins scored candidates back to database tables for grouping, filtering, and final result construction. Standard SQL over a temporary table.

The agent writes a single SQL statement; a "query materializer" intercepts pseudo-functions (`vec_ops()`, `keyword()`), dispatches each phase to the appropriate engine (SQLite for pre-filter and compose, numpy for scoring, FTS5 for keyword), writes results to temp tables, and rewrites the SQL to reference them before handing it back to SQLite.

### Five Modulations

| Token | Operation | Formula | Cost |
|-------|-----------|---------|------|
| `suppress:X` | Subtract directional similarity to topic X | `scores -= w * (M @ embed(X))` | 1 matmul |
| `decay:N` | Reciprocal decay with N-day half-life | `scores *= 1/(1 + days/N)` | elementwise |
| `centroid:ids` | Shift query toward mean of example embeddings | `q = alpha*q + (1-alpha)*mean(E[ids]); q /= ||q||` | 1 mean |
| `from:A to:B` | Blend directional similarity (trajectory) | `scores = 0.5*sim + 0.5*(M @ (embed(B)-embed(A)))` | 1 matmul |
| `diverse` | MMR iterative selection (lambda=0.7) | `score = lambda*rel - (1-lambda)*max_sim` | k*n pairwise |

Modulations execute in a fixed order regardless of token position: centroid (query shift) -> base similarity -> trajectory -> decay -> suppress -> diverse. They compose sequentially -- each transforms the score array passed from the previous step. Because each modulation changes the scoring function, absolute scores are not comparable across configurations.

### Interface Layers

Three levels of abstraction:
1. **Python API**: Direct numpy operations on the candidate matrix
2. **Token grammar**: `diverse suppress:database sqlite storage decay:7 pool:1000` -- whitespace-delimited tokens parsed by a deterministic parser
3. **SQL composition**: `vec_ops()` and `keyword()` as pseudo-functions in SQL FROM clauses, joined with standard SQL

### Hybrid Retrieval

Keyword (FTS5 BM25) and vector search compose via SQL JOIN -- the intersection requires chunks to match both keyword and semantic criteria. The agent controls the intersection/union logic through standard SQL.

### Schema Discovery

The agent calls `@orient`, a SQL preset that queries `pragma_table_info()` to discover the live schema at runtime. This means schema changes propagate without updating instructions. ~2,200 tokens of static MCP instructions plus runtime schema discovery.

---

## Key Claims & Evidence

### Latency (Table 2, 240K chunks, 128-dim, Intel i9-13900KF, 32GB RAM)

| Operation | Time | Scope |
|-----------|------|-------|
| Matrix multiply (240K x 128) | 5ms | Phase 2 only |
| Scoring + 3 modulations + MMR | 12ms | Phase 2 only |
| Full pipeline (pre-filter + scoring + compose) | 19ms | All phases |
| FTS5 keyword search | 18ms | -- |
| Hybrid (keyword JOIN vec_ops) | 63ms | All phases |

### Scaling (Table 4, warm cache)

| Corpus size | Base matmul | Full pipeline | Memory |
|-------------|-------------|---------------|--------|
| 250K | 5ms | 19ms | 122MB |
| 500K | 7ms | 37ms | 244MB |
| 750K | 15ms | 73ms | 366MB |
| 1M | 17ms | 82ms | 514MB |

Memory scales linearly (128-dim float32 = 512 bytes/chunk). Latency is sublinear below ~500K (cache effects), linear above.

### Pre-filtering Effect (Table 3)

| Pre-filter | Candidates | Phase 2 time |
|------------|-----------|--------------|
| Full corpus | 240,000 | 12ms |
| Non-tool, last 30 days | 61,000 | 12ms |
| Non-tool, last 7 days | 11,000 | 0.6ms |
| Non-tool, last 24 hours | 3,000 | 0.4ms |
| One project, 30 days | 37,000 | 5ms |

Latency drop becomes visible below ~50K candidates.

### Behavioral Validation (Table 5, four BEIR datasets, 30 queries each)

| Modulation | Measured Effect | nDCG@10 retention |
|------------|----------------|-------------------|
| `diverse` | ILS reduction 10-40% across all corpora | 59-93% (Table 6) |
| `suppress:X` | RBO vs baseline 0.19-0.41 | -- |
| `decay:7` | Results 42-50 days more recent on average | -- |
| `centroid:ids` | Centroid similarity +0.05 to +0.12 | -- |
| `from:A to:B` | RBO vs baseline 0.08-0.25 | -- |

### MMR Diversity vs. Relevance Tradeoff (Table 6)

| Corpus | Baseline nDCG@10 | `diverse` nDCG@10 | Retention | ILS reduction |
|--------|------------------|-------------------|-----------|---------------|
| SciFact | 0.60 | 0.56 | 93% | 12% |
| FiQA | 0.41 | 0.33 | 80% | 28% |
| SCIDOCS | 0.18 | 0.15 | 83% | 18% |
| NFCorpus | 0.13 | 0.08 | 59% | 40% |

Tightly clustered corpora pay a higher diversity cost. The low NFCorpus baseline (0.13) reflects reduced dimensionality (128-dim Nomic) against specialized biomedical queries.

### Methodological Strengths

- **Algebraic correctness testing**: 1,840 individual score comparisons across four corpora with zero floating-point mismatches. This is unusually rigorous for a systems paper.
- **Honest scope**: The paper explicitly states this is not a retrieval quality benchmark. The evaluation measures "do the operations do what their formulas say" and "does latency scale," not "does this beat BM25 on BEIR." This is refreshing.
- **Production deployment**: 6 weeks in production, ~6,500 agent queries. The materializer rewrite failure rate is 0% across that usage (though deeply nested SQL may still fail).
- **Pre-filter vs. post-filter**: The case study (Section 5.2) where post-filtering a 1% subset loses all but 1 of 200 top-cosine candidates is a clean demonstration of the pre-filtering advantage.

### Methodological Weaknesses

- **No retrieval quality comparison**: The paper doesn't measure whether PEM modulations actually improve retrieval quality on any standard benchmark. Behavioral validation (RBO, ILS, centroid similarity) shows that modulations *change* rankings but not whether the changes are *better*. The suppress case study (Table 7) is compelling qualitatively but anecdotal.
- **128-dim only**: All results use Matryoshka-truncated 128-dim embeddings. At 1536-dim (Somnigraph's embedding dimension), the matrix multiply would be ~12x more expensive and memory would be ~6.1GB for 1M chunks. The sub-100ms latency claim may not hold.
- **Single-user production corpus**: The 240K chunk corpus is one user's AI coding session history. Generalizability to other corpus types is untested beyond the BEIR behavioral validation.
- **30 queries per dataset for behavioral validation**: Small sample. The paper acknowledges the 30-query subset runs slightly optimistic (SciFact nDCG@10 0.60 on 30 queries vs 0.55 on full 300).
- **Agent error rate acknowledged but unmitigated**: 4% SQL composition error rate (down from 10%). The paper is honest about this but doesn't evaluate whether agent errors systematically bias toward certain types of queries or results.
- **No comparison to reranking**: The paper positions PEM as an alternative to learned rerankers but never compares them. A reranker that takes the same candidate set and applies learned feature combinations could outperform manually-composed modulation tokens.

---

## Relevance to Somnigraph

### What flexvec does that we don't

**1. Explicit dominant-cluster suppression.** The `suppress:X` modulation directly addresses a problem Somnigraph doesn't have a mechanism for: when a large cluster of semantically similar memories dominates results, suppressing the cluster's direction in embedding space surfaces buried content. Somnigraph's production reranker relies on the `diversity_score` feature (rank 24 of 26 by importance) and the memory graph (PPR expansion) to diversify, but neither explicitly suppresses a known dominant direction. The suppress case study (Table 7) -- where baseline returns descriptive/marketing content while suppressed results return actual architecture -- is the kind of failure that could occur in Somnigraph when many related memories share similar themes.

**2. SQL pre-filtering before vector scoring.** Somnigraph's retrieval pipeline in `tools.py` / `impl_recall()` runs FTS5 and sqlite-vec in parallel, then fuses with RRF. The vector channel scores all ~730 memories every time -- there's no metadata pre-filter before the vector search. At current scale this is fine (730 memories, not 240K chunks), but the principle is sound: when the agent knows the category, time range, or other metadata constraints, pre-filtering before scoring is strictly better than post-filtering. sqlite-vec's virtual table approach is post-filter by nature (WHERE clauses apply after the KNN scan).

**3. Agent-composed queries.** The agent writes SQL rather than calling a fixed `recall(query, limit)` API. This gives the agent compositional control over filtering, scoring modifications, and result construction in a single statement. Somnigraph's `recall()` takes a query string, optional context, limit, and boost_themes -- the agent can't express "give me procedural memories from the last week, suppressing chess-related content, diversified."

**4. Trajectory (directional) blending.** The `from:A to:B` modulation applies embedding arithmetic (the vector difference between two concepts) as a scoring bias. This is a creative way to search for content along a conceptual trajectory (e.g., `from:prototype to:production`). Somnigraph has no equivalent.

### What we already do better

**1. Learned reranking vs. manual modulation tokens.** flexvec's modulations are manually specified per-query by the agent. Somnigraph's 26-feature LightGBM reranker automatically combines feedback signals, graph signals, metadata, and query-dependent features into a single score. The reranker learns which signals matter from 1032 ground-truth queries; flexvec's modulations require the agent to know which operations to apply. The reranker is strictly more powerful for *ranking quality* -- it just isn't composable in the way PEM is.

**2. Feedback loop.** flexvec has no feedback mechanism. Queries are stateless. Somnigraph's per-query utility feedback (r=0.70 with GT) and EWMA aggregation with empirical Bayes prior provide the dominant retrieval signal. A memory system without feedback is guessing -- this is Somnigraph's primary architectural differentiator and flexvec doesn't address it.

**3. Graph-conditioned expansion.** Somnigraph's PPR-based graph traversal, Hebbian co-retrieval weighting, and edge-based expansion surface memories that are semantically distant but structurally connected. flexvec operates on embedding similarity only (with modulations to reshape it). The multi-hop failure analysis shows 88% of evidence turns have zero content-word overlap with the query -- embedding modulations alone cannot bridge vocabulary gaps that require graph traversal.

**4. Sleep/consolidation.** flexvec stores raw session history and never transforms it. Somnigraph's NREM edge detection and REM clustering/summarization create structure that doesn't exist in the raw data. Whether this helps retrieval is unmeasured, but the capability exists.

**5. Hybrid retrieval with RRF fusion.** Somnigraph fuses FTS5 BM25 and vector channels via RRF (k=6). flexvec's hybrid approach is SQL JOIN (intersection) -- a chunk must appear in *both* keyword and vector results. This is more restrictive. Union-based fusion (RRF) is generally preferred for recall because a memory relevant by one channel shouldn't be excluded for missing the other.

---

## Worth Stealing (ranked)

### 1. Suppress modulation for dominant-cluster mitigation

**What**: Subtract directional similarity to a named concept from all candidate scores, demoting an entire semantic cluster without removing individual memories.

**Why it matters**: Somnigraph's production database has thematic clusters (chess analysis, home automation, memory system internals) that could dominate retrieval for ambiguous queries. The reranker's `diversity_score` feature is weak (rank 24/26). When the agent knows what it *doesn't* want, there's no mechanism to express that.

**Implementation**: Add an optional `suppress` parameter to `recall()` (string or list of strings). In `tools.py` / `impl_recall()`, embed the suppress terms, compute dot products against all candidate embeddings, and subtract the weighted similarity from scores before reranking. This operates on the vector channel's raw scores before RRF fusion. Alternatively, add it as a reranker feature: `suppress_similarity` = max cosine similarity between the candidate and any suppress term embedding. The reranker approach is more principled (learned weighting) but less transparent to the agent.

**Effort**: Low. One parameter addition, one embedding call, one vector operation. The infrastructure for embedding arbitrary text already exists in `embeddings.py`.

### 2. Pre-filtering before vector search at scale

**What**: When the agent specifies metadata constraints (category, time range, priority threshold), filter candidates *before* vector scoring rather than after.

**Why it matters**: At 730 memories, scoring the full corpus is cheap. At 5,000+ memories (the scaling question in roadmap), pre-filtering becomes relevant. More importantly, post-filtering via sqlite-vec's virtual table can cause pool starvation -- the exact problem flexvec's Section 5.2 demonstrates (1 of 200 candidates survive post-filter).

**Implementation**: When `recall()` receives category, time-range, or other metadata filters, generate the candidate ID set via SQL first, then pass only those IDs to the vector channel. In `fts.py` and `scoring.py`, accept an optional `candidate_ids` parameter that restricts which memories enter scoring. This is a plumbing change, not algorithmic.

**Effort**: Low-Medium. The SQL filtering exists (memories table has category, priority, created_at). Wiring it into the vector channel before scoring requires modifying `impl_recall()` flow in `tools.py`.

### 3. Directional embedding arithmetic for trajectory queries

**What**: Allow queries like "how did X evolve from A to B" by blending the vector difference `embed(B) - embed(A)` into the scoring.

**Why it matters**: Temporal evolution queries ("how did the reranker approach change from formula to learned model") are a known weakness. The `from:A to:B` trajectory modulation is a lightweight way to bias retrieval toward a conceptual direction without requiring event-time metadata or temporal indexes.

**Implementation**: Add an optional `trajectory` parameter to `recall()` as a tuple `(from_concept, to_concept)`. Compute direction vector = `embed(to) - embed(from)`, normalize, blend into vector channel scores. Could be a reranker feature (`trajectory_alignment` = cosine similarity between candidate and direction vector) or a pre-scoring query modification.

**Effort**: Medium. Requires two additional embedding calls and a design decision about where in the pipeline to apply it (pre-RRF vector channel modification vs. reranker feature).

---

## Not Useful For Us

### Agent-written SQL for retrieval

flexvec's central design -- letting the AI agent write SQL -- is architecturally incompatible with Somnigraph's approach. Somnigraph exposes `recall()`, `remember()`, `forget()` as MCP tools with structured parameters. The agent doesn't need to know SQL. This is a deliberate design choice: Somnigraph's CLAUDE.md snippet works because the interface is simple enough that a fresh Claude session can use it correctly within 2-3 sessions. A SQL interface would require extensive schema documentation, increase the error surface (flexvec reports 4% agent SQL error rate even with guidance), and remove the ability to apply learned reranking transparently.

The 4% error rate is particularly concerning. For a personal memory system where every query matters, 1-in-25 queries silently returning wrong results is unacceptable. flexvec's production use case (searching coding session history) is more tolerant of retrieval failure than Somnigraph's (retrieving memories that shape ongoing conversations).

### In-memory embedding matrix

flexvec loads all embeddings into a numpy array at startup for brute-force matmul. At 128-dim this costs 122MB for 250K chunks. Somnigraph uses 1536-dim embeddings (text-embedding-3-small) via sqlite-vec. Loading 730 memories at 1536-dim float32 would cost ~4.5MB -- trivial. But the brute-force approach doesn't compose with sqlite-vec's virtual table interface, which Somnigraph uses for vector search. The benefit (sub-millisecond scoring) is irrelevant at Somnigraph's scale; the cost (abandoning sqlite-vec, loading all embeddings at startup, maintaining a numpy dependency for scoring) isn't worth it.

### Query materializer architecture

The three-phase materializer that intercepts pseudo-functions, dispatches to different engines, and rewrites SQL is an elegant solution to a problem Somnigraph doesn't have. Somnigraph's retrieval pipeline is a Python function (`impl_recall()` in `tools.py`) that orchestrates FTS5 search, vector search, RRF fusion, graph expansion, and reranking in code. The materializer pattern would add complexity without benefit -- the pipeline phases are already explicit in Python, and adding SQL as an intermediary adds a parsing/rewriting layer with its own failure modes.

### Token grammar

The modulation tokens (`diverse suppress:X decay:7 pool:1000`) are clever shorthand for the agent-written-SQL use case. For Somnigraph's structured parameter approach, adding individual parameters (suppress, trajectory, decay_override) to `recall()` is simpler, more discoverable (agents see parameter schemas), and less error-prone than a mini-language embedded in a query string.

### MMR diversity selection

flexvec's `diverse` modulation applies MMR (Maximal Marginal Relevance) iterative selection. Somnigraph doesn't use MMR and the analysis suggests it may not need to. The `diversity_score` reranker feature is weak, but PPR graph expansion and RRF fusion across FTS5 and vector channels already provide structural diversity. MMR's k*n pairwise cost is also the most expensive modulation in flexvec. Adding it to Somnigraph's pipeline after reranking would need to justify itself against the existing diversity mechanisms.

---

## Impact on Implementation Priority

### Minimal impact on current priorities

This paper doesn't change the priority ordering in STEWARDSHIP.md. The main takeaways (suppress, pre-filter, trajectory) are incremental improvements to the retrieval pipeline, not architectural shifts.

**P2 (reranker iteration)**: The suppress concept could become a reranker feature (`suppress_similarity`) in the 31-feature retrain, but this requires a design decision about how the agent specifies what to suppress. More likely a post-retrain enhancement.

**P4 (LoCoMo benchmark)**: The multi-hop failure analysis (88% vocabulary gap) is the binding constraint on retrieval improvements. flexvec's modulations operate *within* the embedding similarity space and cannot bridge vocabulary gaps where query and evidence have zero content-word overlap. HyDE and better expansion methods remain the expected next levers, not score-array modulations.

**Roadmap #10 (prospective indexing)**: The trajectory modulation (`from:A to:B`) is orthogonal to prospective indexing. Prospective indexing addresses the vocabulary gap at *write time* by generating hypothetical queries; trajectory addresses the direction of search at *query time*. Both could help, but prospective indexing attacks the more fundamental problem.

**Roadmap #21 (expansion method ablation)**: flexvec's suppress is conceptually related to the dead rocchio expansion method (0% fire rate in LoCoMo). Rocchio failed because it modified the query vector toward/away from known-relevant/irrelevant documents -- a centroid shift in embedding space. suppress operates on scores (post-dot-product), not on the query vector (pre-dot-product). The distinction matters: suppress doesn't change which candidates are retrieved, only how they're ranked. Rocchio changes which candidates score highly in the first place. The failure of rocchio doesn't predict the failure of suppress.

---

## Connections

### sqlite-vec (Alex Garcia)

flexvec explicitly positions itself relative to sqlite-vec, which Somnigraph uses for vector search. The paper notes that sqlite-vec provides "brute-force KNN inside SQLite via C virtual tables, with filtering scoped to metadata columns and a limited set of comparison operators." flexvec argues for bypassing the virtual table interface in favor of direct numpy matrix operations, which enable score manipulation between scoring and selection. This is the fundamental architectural divergence: sqlite-vec integrates vector search into SQLite's query planner; flexvec keeps vector operations outside SQLite and uses the materializer to bridge them.

For Somnigraph, the sqlite-vec approach is correct at current scale. The paper's argument for brute-force numpy matters at 240K+ chunks where ANN indexes introduce pre/post-filter tradeoffs -- a problem Somnigraph doesn't face at 730 memories.

### HyDE (hypothetical document embeddings)

flexvec's centroid modulation (`centroid:ids`) is a manual version of the insight behind HyDE: shifting the query vector toward document-like regions of embedding space. HyDE automates this by generating a hypothetical document via LLM; centroid requires the agent to supply known-good IDs. For Somnigraph's multi-hop vocabulary gap problem, HyDE's automatic approach is more promising because the agent doesn't know which memory IDs to use as anchors when the query has zero overlap with evidence.

### SPLADE (learned sparse representations)

SPLADE addresses the vocabulary mismatch problem through learned term expansion in the sparse representation. flexvec's suppress modulation operates in the dense space. For Somnigraph's vocabulary gap (88% zero content-word overlap in multi-hop failures), SPLADE-style expansion of the sparse channel would be complementary to any dense-space modulations. Different attack surfaces for the same problem.

### A-Mem (Zettelkasten-inspired memory)

A-Mem's enriched embeddings (embedding the full atomic note including links and context) are the write-time analog of flexvec's query-time modulations. Both try to close the gap between what was stored and what the query asks for. A-Mem does it at write time (richer representations); flexvec does it at query time (richer scoring). Somnigraph's enriched embeddings (content + summary + themes) are closer to A-Mem's approach. The paper confirms that query-time and write-time enrichment are complementary, not competing.

### LoCoMo benchmark

flexvec doesn't evaluate on LoCoMo or any QA benchmark. The behavioral validation (BEIR datasets, 30 queries each) measures ranking change, not answer quality. This makes direct comparison with Somnigraph's 85.1% LoCoMo accuracy impossible. The paper's honest framing of this as a limitation is appropriate.

---

## Summary Assessment

flexvec is a well-executed systems paper that solves a real problem (giving AI agents compositional control over retrieval) within a specific context (large session-history corpora in SQLite). The three-phase pipeline architecture, query materializer, and PEM concept are cleanly designed, and the paper's evaluation is honest about what it does and doesn't measure. The algebraic correctness testing (1,840 score comparisons, zero mismatches) is unusually rigorous.

For Somnigraph, the paper's value is limited. The core design -- agent-written SQL with modulation tokens -- is architecturally incompatible with Somnigraph's MCP tool interface and learned reranker. The modulations that flexvec exposes (suppress, decay, centroid, trajectory, diversity) are interesting query-time operations, but Somnigraph's reranker already handles most of the same concerns through learned feature combinations, and the remaining gap (vocabulary mismatch in multi-hop retrieval) requires write-time solutions (prospective indexing, HyDE) that score-array manipulations cannot address.

The single most useful idea is the **suppress modulation** -- a mechanism for the agent to demote an entire semantic cluster by subtracting directional similarity. This is a cheap, composable operation that could be added to Somnigraph's `recall()` as an optional parameter, either applied pre-reranker to the vector channel or as an additional reranker feature. It addresses a real failure mode (dominant clusters) that the current pipeline handles weakly. Everything else in the paper is either already handled better by existing Somnigraph components (learned reranking > manual modulations, RRF fusion > SQL JOIN intersection, feedback loop > stateless queries) or solves problems that don't exist at Somnigraph's scale (pre-filter optimization at 240K chunks, in-memory embedding matrix).
