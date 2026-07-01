# m_flow - Graph-as-scorer memory: min-cost path propagation up a 4-layer "cone" knowledge graph, no LLM at query time

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Recovery note

The table URL `github.com/FlowElement-ai/m_flow` 404s (org, repo, and `mflow-benchmarks` all resolve to nothing — likely renamed/privated after the audit). Recovered the full source from the **PyPI sdist `mflow-ai==0.3.6`** (`pip download mflow-ai`, extracted from `mflow_ai-0.3.6.tar.gz`, 1099 files, 963 tests). All code paths below are cited from that tree. The `docs.m-flow.ai` / `m-flow.ai` sites are live (200). This is a code-level read, not a README skim.

---

## Architecture

M-flow is a **bio-inspired Graph RAG memory engine** whose thesis is a sharp one: *existing systems build graphs but still retrieve by embedding distance; M-flow makes the graph itself the scoring engine.* It is a full product (FastAPI REST, MCP stdio/sse/http, CLI, Next.js UI, Docker/Helm, multi-tenant auth), pluggable across SQLite/Postgres, LanceDB/Chroma/PGVector/Qdrant/Weaviate/Milvus/Pinecone, and Kuzu/Neo4j/Neptune. Apache-2.0. Author Junting Hua (Chinese + English bilingual codebase — many comments/patterns are CJK-aware).

### Storage & Schema
Relational (SQLAlchemy) + vector + graph, three separate stores, per-user+dataset DB isolation. The memory unit is not a flat record but a **four-layer "cone" graph** (`m_flow/core/domain/models/`):

- **Episode** — a bounded semantic focus (an incident, a decision process). Fields: summary, signature, content, time fields.
- **Facet** — one topical cross-section of an Episode (search_text, anchor_text, facet_type, description, aliases).
- **FacetPoint** — an atomic assertion derived from a Facet (search_text, aliases, embedding).
- **Entity** — a named thing (person/tool/metric) that **bridges across Episodes** (name, type, description).

Edges are typed: `has_facet` (Episode→Facet), `has_point` (Facet→FacetPoint), `involves_entity` (Episode→Entity **and** Facet→Entity), plus `same_entity_as`, `includes_chunk`. Every node level and edges carry their own embedding, indexed into **7 vector collections** (`Episode_summary`, `Facet_search_text`, `Facet_anchor_text`, `FacetPoint_search_text`, `Entity_name`, `Concept_name`, `RelationType_relationship_name` — config.py:38).

### Memory Types
Episodic (the cone) and **Procedural** (v0.3.1+, `m_flow/memory/procedural/`) — step-by-step task knowledge with its own extraction/classifier/governance. No decay/reflection/meta categories. 5 retrieval "modes" (Episodic, Procedural, Triplet Completion, Lexical, Cypher) exposed via `RecallMode`.

### Write Path
Heavy, LLM-driven, well-staged (`m_flow/memory/episodic/episode_builder/`, `write_episodic_memories.py`, 1001 lines):
- **Coreference resolution at ingestion** (`coreference/` module): pronouns (he/she/it/their) resolved to concrete antecedents *before* indexing, so the graph stores names not pronouns. Bilingual (English + 11 Chinese pronoun types with semantic-role analysis).
- Phase 0A three-way parallel: entity-name extraction, facet drafting, facet↔entity matching.
- Facets extracted via structured-output LLM (Instructor or BAML, `STRUCTURED_OUTPUT_FRAMEWORK`), FacetPoints refined, entity descriptions merged/optimized.
- **Entity resolution**: `facet_entity_matcher.py` uses *exact* word-boundary regex matching (entities extracted in EXACT original form) + `same_entity_as` edges. `semantic_merge.py` does within-episode synonym merge but only above a **conservative 0.90 cosine threshold, default-disabled** ("no false merge" prioritized over "full merge").
- **Procedural write-path quality gate** (`governance/worth_storing.py`): a `WorthEvaluator` scores each procedure 0-1 from negative signals (generic-step patterns, <100 chars) and positive signals (safety-critical / org-specific / toolchain / complexity regex, step count ≥5, context completeness), then routes to `index_level ∈ {full, summary_only, none}` at threshold 0.4. This is explicit salience gating on the write path.
- Dedup at KuzuDB adapter entry level; a background worker queue (`mflow_workers/`, Modal) persists nodes/edges async.

### Retrieval (the core contribution)
`m_flow/retrieval/episodic/bundle_search.py` (719 lines) + `bundle_scorer.py` + `adaptive_scoring.py`. Flow:
1. **Preprocess** query (time parsing, hybrid decision when core-word count ≤ 3).
2. **Wide-net multi-granularity vector search**: embed the query once, search *all 7 collections* with `wide_search_top_k=100` — entry points at every layer of the cone simultaneously (config.py:38, `_vector_search`).
3. **Adaptive scoring context** (`adaptive_scoring.py`): per-collection confidence `Conf = f_dist(raw_dist/baseline) × f_gap(top1−top2 gap)`. Each collection has a hand-set distance baseline (Episode_summary 1.06 for long text, FacetPoint 0.50 for short); `f_dist` maps distance/baseline ratio to a [0.1,1.0] quality factor, `f_gap` maps top1/top2 separation to a [0.2,1.0] discriminability factor. These dynamically set the fusion weight `lambda` (semantic vs structural, clipped [0.3,0.95]) and node/edge weights — a *query-adaptive* trust dial rather than a fixed formula.
4. **Two-phase projection** of vector hits into the Kuzu subgraph.
5. **Min-cost path (bundle) scoring** (`compute_episode_bundles`): every Episode's score = the **cheapest evidence path** from any vector hit up to that Episode. Cost accumulates `node_direct_distance + edge_distance + hop_cost(0.05)`; a missing edge embedding costs `edge_miss_cost=0.9`; matching an Episode directly is *penalized* (`direct_episode_penalty=0.3`) to prefer routing through fine-grained detail; a strong facet-direct hit (<0.1) discounts the edge/hop cost. Paths considered: FacetPoint→Facet→Episode, Entity→Facet→Episode, Entity→Episode, Facet→Episode, direct. **Multi-hop reasoning is pure cost arithmetic — no LLM call at retrieval time.** "One strong path is enough."
6. Exact-match bonuses (number/keyword/english), optional time bonus (mentioned-time vs created-at weighting), then rank, assemble output (summary or facet-detail mode).

There is **no learned reranker and no RRF** — fusion is the adaptive-lambda blend plus the graph path arithmetic. "Reranking" appears only as a docstring aspiration.

### Consolidation / Processing
**None** in the sleep/consolidation sense. No offline reprocessing, no gap analysis, no summary-of-summaries pass, no contradiction reconciliation. The `deduplicate_nodes_and_edges` fixes and semantic-merge run at write time only. Grep for decay/consolidate/sleep/rerank/feedback across `m_flow/` returns only unrelated adapter hits.

### Lifecycle Management
**None.** No time decay, no forgetting curve, no supersession/versioning, no trust/confidence-per-source, no time-travel/history query. Lifecycle = explicit `delete`/`prune` MCP tools only. (The retrieval "penalties" are relevance scoring, not knowledge decay.) Procedural memory has `update_usage_stats` / `reconcile_active`, the closest thing to lifecycle, but scoped to procedures.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Graph is the scoring engine, not just what's embedded | `compute_episode_bundles` genuinely propagates cost along typed edges; scoring is min-path, not cosine rank | **Validated** — the code does exactly this; it's a real architectural distinction |
| LoCoMo-10 **81.8%** (best vs Cognee 79.4, Zep 73.4, Mem0 50.4) | README + external `mflow-benchmarks` repo (now 404); gpt-5-mini answer + **gpt-4o-mini judge**, top-k=10, Cat-5 excluded | **Plausible but not comparable to our 85.1.** End-to-end QA (good) but judged by weak gpt-4o-mini; our 85.1 uses an Opus judge (Opus is ~3pp stricter than gpt-4.1-mini). Different judge + Cat-5 exclusion = not apples-to-apples |
| LongMemEval **89%** (temporal 93, multi-session 82) | README; first 100 Qs only, top-k=10 | **Plausible**, small slice (100 Qs), same weak-judge caveat |
| Multi-hop found via graph path with zero keyword overlap | The Entity→Facet→Episode path in `bundle_scorer` supports this mechanically | **Validated as a mechanism**; the "migration→Redis" example is illustrative not measured |
| Multi-tenant, face-aware partitioning | `auth/`, per-user+dataset DB isolation in adapters | **Validated** |
| No decay/contradiction/consolidation | Confirmed by code absence | **Validated (as a limitation)** |

---

## Relevance to Somnigraph

### What m_flow does that Somnigraph doesn't
- **Graph *as the scorer* via min-cost path propagation.** Somnigraph's graph participates through PPR expansion + a betweenness *feature* in `reranker.py`, but final ranking is still RRF+reranker over largely flat candidates. M-flow scores each result by the tightest evidence *chain*, letting a query hit at fine granularity (a FacetPoint) route up to the right Episode with zero surface overlap. This is precisely the mechanism our `docs/multihop-failure-analysis.md` names as the ~88% vocabulary-gap ceiling — reached here structurally instead of via synthetic vocabulary bridges (our L5b approach in `scripts/`).
- **Multi-granularity vector entry points.** 7 collections spanning summary→atomic-fact→entity, all searched simultaneously (`config.py:38`). Somnigraph embeds one enriched vector per memory (`embeddings.py`); it has no sub-memory (fact-level) or super-memory (entity-bridge) index to enter retrieval at.
- **Write-path salience gating.** `worth_storing.py` is a concrete `should_index / index_level ∈ {full, summary_only, none}` gate — exactly the "write-path quality gating" Somnigraph is documented as lacking (STEWARDSHIP / brief). Corroborates the Phase 18 finding (ByteRover/agentmemory) that write-path quality, not retrieval, is what LoCoMo leaders win on.
- **Coreference resolution at ingestion** — pronoun→antecedent before indexing. Somnigraph stores whatever the caller wrote; multi-turn "it/they" references degrade its embeddings.
- **Cross-Episode entity bridging** as a first-class retrieval path (Entity node linked across all Episodes). Somnigraph has typed edges but no persistent entity index spanning memories.

### What Somnigraph does better
- **Learned reranker with measured GT correlation.** 26-feature LightGBM, NDCG=0.7958, Spearman r=0.70. M-flow's adaptive scoring is entirely *hand-tuned* (dozens of magic thresholds in `config.py`: ratio_good 0.5, gap_high 0.15, lambda boosts) — the same kind of formula Somnigraph *replaced* because a learned model beat it +6.17pp.
- **Feedback loop.** Somnigraph has explicit per-query utility+durability ratings, EWMA, UCB exploration, Hebbian PMI. M-flow has *no* feedback signal at all — retrieval never learns from use.
- **Sleep-based offline consolidation.** NREM edge/merge/archive + REM gap analysis. M-flow builds its graph only at write time and never revisits it (no contradiction detection, no re-summarization).
- **Lifecycle / decay.** Somnigraph's per-category exponential decay + reheat vs M-flow's none.
- **Honest benchmark judging.** Our 85.1 uses an Opus judge and documents the LoCoMo judge's 62.8% false-accept rate; M-flow leans on gpt-4o-mini judging without that caveat.

---

## Worth Stealing (ranked)

### 1. Min-cost path propagation up a granularity graph (High)
**What**: Score a memory by the *cheapest evidence chain* connecting a query hit to it, accumulating `vector_distance + edge_distance + hop_penalty`, with the query allowed to enter at sub-memory (fact) or entity granularity — not just at the memory-summary level.
**Why**: This directly attacks the vocabulary-gap ceiling documented in `docs/multihop-failure-analysis.md`. Somnigraph currently patches multi-hop with synthetic vocabulary bridges (L5b); path propagation reaches the same target structurally, and it's an *inference-time* method (no extra LLM), unlike our sleep-built synthetics.
**How**: Would require a sub-memory index (fact/claim nodes) and edges carrying embeddings — a large lift on `db.py` schema and `embeddings.py`. Cheaper first step: prototype a *bounded* version in `scoring.py` — when PPR-expanded neighbors exist, add a "best-path-cost" feature (`neighbor_score + edge_weight + hop_penalty`) to the reranker feature vector, and let LightGBM decide its weight. That tests the signal without the entity/fact-layer rebuild.

### 2. Write-path salience gate with tiered indexing (Medium)
**What**: `WorthEvaluator`-style scorer at `remember()` time returning `{full, summary_only, none}` from cheap signals (generic-boilerplate patterns = negative; novelty/specificity/complexity = positive; length floor).
**Why**: Somnigraph's documented gap. Independent corroboration (Phase 18) that write-path quality is the LoCoMo differentiator. A `summary_only` tier is a nice middle ground — index the gestalt layer but skip detail for low-value memories.
**How**: New `src/memory/worth.py`, called in `tools.py` `impl_remember` before embedding; low-worth memories get `layer=gestalt`-only or `priority` capped. Regex/heuristic first (no LLM), matching M-flow's cheap gate.

### 3. Query-adaptive semantic/structural fusion weight (Medium)
**What**: Instead of a fixed RRF k, compute per-query confidence from (a) distance-to-collection-baseline ratio and (b) top1/top2 gap, and shift the blend toward whichever signal is discriminative *for this query* (`f_dist × f_gap → lambda`).
**Why**: Somnigraph's RRF k=14 is globally Bayesian-optimized but static; a query where vector top-1/top-2 are near-tied should trust vector less. This is a principled, cheap, learnable-adjacent knob.
**How**: Add top1/top2-gap and distance-ratio as *reranker features* in `reranker.py` (let the model learn the mapping M-flow hand-codes) rather than importing the hand-tuned thresholds wholesale.

### 4. Coreference resolution before embedding (Low-Medium)
**What**: Resolve pronouns to antecedents in the memory text prior to enrichment/embedding.
**Why**: Multi-turn captures ("it broke again") embed poorly. Cheap quality win on the write path.
**How**: An optional preprocessing step in `embeddings.py`/`tools.py`; could be a single LLM call during sleep rather than at write time to stay latency-free.

---

## Not Useful For Us

### Multi-store adapter zoo (7 vector DBs, 3 graph DBs, Postgres, auth, Next.js UI, Modal workers)
Somnigraph is deliberately single-user SQLite+sqlite-vec+FTS5. M-flow's pluggable-everything infra is product surface, not research signal.

### Hand-tuned scoring constant thicket
`config.py` has ~40 magic numbers (baselines, ratio/gap thresholds, lambda boosts). Somnigraph explicitly moved *away* from this by learning the reranker; importing the constants would be a regression. Steal the *feature ideas* (idea #3), not the values.

### 5 "retrieval modes" / Cypher mode
Mode proliferation (incl. raw Cypher passthrough) is API surface for a multi-tenant product; irrelevant to a single learned pipeline.

---

## Connections

- **Convergent with the Phase 18 write-path thesis** (see `ai-memory-comparison.md`, ByteRover, agentmemory, `docs/sessions/2026-06-28-phase18-source-sweep.md`): write-path quality, not retrieval cleverness, is what the LoCoMo leaders win on. M-flow's `worth_storing.py` + coreference resolution + entity bridging are all write-path investments — and it's a benchmark leader. Third independent corroboration.
- **Contrast with Somnigraph's own multi-hop work** (`docs/multihop-failure-analysis.md`, L5b synthetic bridges): M-flow reaches the same multi-hop target via inference-time path arithmetic rather than sleep-built synthetic vocabulary — the cleaner mechanism if the sub-memory graph exists.
- **Cone graph ≈ layered memory** seen elsewhere (Somnigraph's detail/summary/gestalt layers, MemPalace verbatim, MIRIX): M-flow's Episode→Facet→FacetPoint is the same hierarchy but, uniquely, makes *all* layers vector-searchable entry points rather than just display tiers.
- **Graphiti/Zep lineage**: same "typed edges + graph retrieval" family, but M-flow's differentiator vs Zep (which it beats 81.8 vs 73.4) is the path-cost scorer replacing embedding-rank-then-graph-context.

---

## Summary Assessment

M-flow's one genuinely novel, code-verified contribution is **making the graph the scoring function**: retrieval enters via a wide multi-granularity vector net, projects into a 4-layer cone graph, and ranks each Episode by the single cheapest evidence path (`vector_dist + edge_dist + hop_cost`) connecting it to the query — multi-hop reasoning as pure cost arithmetic, no LLM at query time, with a deliberate penalty on matching coarse summaries directly so that fine-grained facts and cross-Episode entities do the routing. This is a real answer to the multi-hop vocabulary gap that Somnigraph currently patches with sleep-built synthetic bridges, and it's the single most important idea to take.

The honest caveats: everything downstream of that idea is **hand-tuned formula** (dozens of thresholds M-flow tunes by feel), the exact approach Somnigraph deliberately abandoned when a learned reranker beat it +6.17pp. And M-flow has *none* of Somnigraph's learning machinery — no feedback loop, no learned reranker, no decay, no consolidation. Its benchmark lead (81.8 LoCoMo, 89 LongMemEval) is real end-to-end QA but judged by gpt-4o-mini on a Cat-5-excluded slice, so it is **not directly comparable to our Opus-judged 85.1**; the numbers should not be placed side by side. The durable lessons are three write-path/retrieval mechanisms — path-cost propagation (as a reranker *feature* first), a tiered write-path worth-gate, and query-adaptive fusion weighting — worth prototyping in `scoring.py`/`reranker.py`/a new `worth.py` without importing M-flow's constant thicket. Verdict: **DIVE** on the path-cost idea; the rest is corroboration and cheap borrowings.
