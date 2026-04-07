# mempalace Analysis

*Generated 2026-04-07 by Opus agent reading local clone*

---

## 1. Architecture Overview

**Repo**: https://github.com/milla-jovovich/mempalace
**License**: MIT
**Language**: Python 3.9+, ~13,467 lines across 18 source modules + 4 benchmark scripts + 4 test files.
**Created**: 2026-04-04, 7 commits, 2 contributors (milla-jovovich, bensig).
**Version**: 3.0.0 (pyproject.toml) / 2.0.0 (__init__.py) -- version mismatch.
**Dependencies**: chromadb>=0.4.0, pyyaml>=6.0 (only two runtime deps).
**Description**: "Give your AI a memory -- mine projects and conversations into a searchable palace. No API key required." Claims highest LongMemEval score ever published (96.6% zero-API, 100% with Haiku rerank).

**Stack**:
```
CLI / MCP Server (19 tools)
    |
Mining Pipeline (project files, conversations, general extraction)
    |
Storage: ChromaDB (vector store) + SQLite (knowledge graph)
    |
Retrieval: ChromaDB semantic search + wing/room filtering
    |
Compression: AAAK dialect (symbolic shorthand)
```

**Module organization** (under `mempalace/`):
- `cli.py` (300+ lines) -- CLI entry point with 6 commands (init, mine, search, wake-up, split, compress, status)
- `miner.py` (500 lines) -- Project file ingestion: scan, chunk (800 chars), route to rooms by folder/keyword, store to ChromaDB
- `convo_miner.py` (400 lines) -- Conversation ingestion: exchange-pair chunking, topic detection
- `searcher.py` (130 lines) -- Thin wrapper around `chromadb.collection.query()` with wing/room WHERE filters
- `knowledge_graph.py` (350 lines) -- SQLite temporal triple store (entities, triples with valid_from/valid_to)
- `palace_graph.py` (230 lines) -- Room-based navigation graph built from ChromaDB metadata (tunnels = shared rooms across wings)
- `layers.py` (200 lines) -- 4-layer memory stack: L0 identity (~50 tokens), L1 essential facts (~800 tokens), L2 on-demand, L3 deep search
- `dialect.py` (300+ lines) -- AAAK compression: entity codes, emotion markers, flag signals
- `general_extractor.py` (521 lines) -- Regex-based classification into 5 memory types (decision, preference, milestone, problem, emotional)
- `entity_detector.py` (300+ lines) -- Two-pass entity extraction from file content
- `entity_registry.py` (300+ lines) -- Persistent entity store with Wikipedia lookup for disambiguation
- `mcp_server.py` (500+ lines) -- MCP server with 19 tools (read, write, KG, diary, graph traversal)
- `normalize.py` (200 lines) -- Format normalization for 5 chat export formats (Claude Code JSONL, Claude.ai JSON, ChatGPT JSON, Slack JSON, plain text)
- Supporting: `config.py`, `onboarding.py`, `room_detector_local.py`, `spellcheck.py`, `split_mega_files.py`

**Key design choices**:
1. **Verbatim-first**: Never summarize user content. Store exact words, use structure (wings/rooms) for retrieval precision.
2. **Zero-API default**: Core features work with only ChromaDB default embeddings (all-MiniLM-L6-v2). LLM reranking is optional.
3. **Palace metaphor**: Wing (project/person) > Hall (memory type) > Room (topic) > Closet (summary) > Drawer (verbatim). This hierarchical taxonomy provides the retrieval boost.
4. **Local-first**: Everything runs offline. ChromaDB + SQLite, no cloud dependencies.

## 2. Memory Model

### Memory types and schema

Memories are stored as ChromaDB documents ("drawers") with metadata:
- `wing` -- project or person (e.g., "wing_driftwood", "wing_kai")
- `room` -- topic within a wing (e.g., "auth-migration", "frontend")
- `hall` -- memory type corridor (hall_facts, hall_events, hall_discoveries, hall_preferences, hall_advice, hall_diary)
- `source_file` -- provenance
- `chunk_index` -- position within source
- `added_by` -- agent name
- `filed_at` -- timestamp

The general extractor adds `memory_type` (decision, preference, milestone, problem, emotional) as another classification axis.

### Storage backend

Two parallel stores:
1. **ChromaDB** (PersistentClient) -- vector store for all drawer content. Uses ChromaDB default all-MiniLM-L6-v2 embeddings (384-dim). No explicit ANN index configuration. Benchmarks optionally use bge-large-en-v1.5 (1024-dim) via fastembed.
2. **SQLite** (knowledge_graph.py) -- temporal triple store for entity relationships. Schema: `entities` (id, name, type, properties) and `triples` (subject, predicate, object, valid_from, valid_to, confidence, source_closet). Indexed on subject, object, predicate, and temporal range.

### Write path

**Project mining** (`miner.py`):
1. Load `mempalace.yaml` config (wing name, room definitions)
2. Walk directory, skip common dirs (.git, node_modules, etc.)
3. For each readable file: detect room by folder path > filename > keyword scoring
4. Chunk at 800 chars with 100-char overlap, splitting on paragraph/line boundaries
5. Check if file already mined (metadata lookup)
6. Store each chunk as a drawer in ChromaDB with wing/room metadata

**Conversation mining** (`convo_miner.py`):
1. Normalize format (5 chat export formats supported)
2. Chunk by exchange pair (user turn + AI response = one unit) or by paragraph
3. Detect room via keyword scoring (technical, architecture, planning, decisions, problems)
4. Store drawers with `ingest_mode: convos` metadata

**General extraction** (`general_extractor.py`):
1. Split text into segments (speaker turns or paragraphs)
2. Filter out code-heavy lines
3. Score each segment against 5 regex marker sets (~20 patterns each)
4. Disambiguate (resolved problems become milestones, etc.)
5. Tag with memory_type for room routing

### Deduplication

`tool_check_duplicate()` in the MCP server checks 0.9 cosine similarity against top-5 before adding a new drawer. The miner checks `file_already_mined()` by source_file metadata. No write-path deduplication beyond these checks -- no density adaptation, no merge/link triage.

### Knowledge graph

The `KnowledgeGraph` class provides a separate temporal triple store:
- `add_triple("Max", "child_of", "Alice", valid_from="2015-04-01")`
- `invalidate("Max", "has_issue", "sports_injury", ended="2026-02-15")`
- `query_entity("Max", as_of="2026-01-15")` -- returns only temporally-valid facts
- `timeline("Max")` -- chronological fact history

This is manually populated via MCP tools (kg_add, kg_invalidate) or seeded from entity_detector facts. It is **not** automatically populated from mined content -- the agent must explicitly add facts. The contradiction detection shown in the README ("Soren finished the auth migration" leading to "attribution conflict") relies on this KG being pre-populated with ground truth.

## 3. Retrieval

### Search mechanisms

**Production retrieval** (`searcher.py`): A single call to `chromadb.collection.query()` with optional `where` filter for wing and/or room. Returns top-N by cosine distance. That is the entire retrieval pipeline for the MCP server and CLI.

There is no BM25, no FTS, no hybrid fusion, no reranking, no graph traversal at query time in the production code. The `palace_graph.py` module provides BFS traversal and tunnel detection, exposed via MCP tools (`mempalace_traverse_graph`, `mempalace_find_tunnels`), but these are separate tools -- not integrated into the search path.

**Benchmark retrieval** (`longmemeval_bench.py`): The benchmark scripts implement significantly more sophisticated retrieval that is **not available in the production code**:
- Hybrid scoring: keyword overlap re-ranking (30% distance reduction for keyword matches)
- Temporal date boost (40% distance reduction for sessions near target date)
- Two-pass retrieval for assistant-reference questions
- Person name boosting (40% distance reduction for proper noun matches)
- Quoted phrase extraction (60% distance reduction for exact quoted phrases)
- Preference pattern extraction (16 regex patterns creating synthetic documents)
- Memory/nostalgia pattern matching
- LLM reranking via Claude Haiku/Sonnet (send top-K to LLM for re-ordering)
- Room-based score boosting (20% distance reduction for room matches)
- bge-large-en-v1.5 embeddings (1024-dim, vs. default 384-dim)

This is a critical distinction: the claimed 96.6%-100% scores come from benchmark-only code that users of the MCP server or CLI do not get.

### Ranking/scoring

Production: raw ChromaDB cosine distance (lower = better). No post-processing.

Benchmarks: `fused_dist = dist * (1.0 - boost_weight * overlap)` where overlap can come from keywords, temporal proximity, room matches, person name matches, or quoted phrases. Multiple boosts stack multiplicatively.

### Query processing

No query expansion, no HyDE, no multi-query, no entity extraction from queries. The query text goes directly to ChromaDB `query_texts` parameter.

## 4. Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| "96.6% LongMemEval R@5, zero API calls" | Benchmark script provided, reproducible | **Credible but contextual.** LongMemEval tests retrieval over ~53 sessions per question -- essentially "find which session contains the answer." Storing each session verbatim and querying with a decent embedding model is a strong baseline that many systems overlook. The 96.6% is real but measures a simpler task than multi-hop reasoning. |
| "100% LongMemEval with Haiku rerank" | Benchmark script provided, detailed progression documented | **Credible but overfit.** The 99.4% to 100% step is explicitly acknowledged as "teaching to the test" -- three targeted fixes for three specific questions. The honest accounting in BENCHMARKS.md is commendable. |
| "100% LoCoMo with Sonnet rerank" | Benchmark script provided | **Structurally guaranteed, not a retrieval result.** top-k=50 exceeds the number of sessions per conversation (19-32), so the ground truth is always in the candidate pool. The Sonnet rerank is doing reading comprehension, not retrieval. The honest LoCoMo score is 60.3% at top-10. The benchmarks doc acknowledges this. |
| "92.9% ConvoMem" | Benchmark script provided | **Credible.** ConvoMem is a more meaningful benchmark for this system's approach. |
| "34% retrieval improvement from palace structure" | Results table in README: 60.9% to 94.8% with wing+room filtering | **Plausible but circular.** Wing+room filtering is an oracle selector -- you need to know the right wing and room to query. The improvement measures the value of perfect metadata routing, not the palace's ability to route automatically. |
| "30x compression with AAAK" | AAAK benchmark mode scores 84.2% on LongMemEval (vs. 96.6% raw) | **Compression is real but harmful.** The 12.4pp retrieval drop shows AAAK destroys semantic signal for embedding-based search. The README positions AAAK as lossless, but it is lossy for the retrieval pipeline. |
| "Contradiction detection" | Knowledge graph supports temporal validity checking | **Requires manual pre-population.** The README example shows contradiction detection, but it depends on the KG already containing the correct facts. There is no automated contradiction detection from mined content. |

### Benchmark integrity

The BENCHMARKS.md document is unusually honest for a repo README -- it explicitly calls out overfitting, structural guarantees, and the difference between clean and contaminated results. A held-out 50/450 split exists. The LoCoMo top-k=50 caveat is clearly stated. This level of methodological honesty is rare in the memory system landscape.

## 5. Standout Feature

**The honest benchmarking methodology.** While many memory systems make retrieval claims without reproducible benchmarks or with unstated caveats, MemPalace provides:
- Reproducible benchmark scripts for 4 benchmarks (LongMemEval, LoCoMo, ConvoMem, MemBench)
- Detailed progression from 96.6% to 100% with each improvement explained
- Explicit acknowledgment of overfitting and structural guarantees
- Train/test split for future uncontaminated evaluation
- Comparison tables against published systems

The benchmarking infrastructure (~5,290 lines across 4 benchmark scripts) is larger than much of the core library.

## 6. Gap Ratings (Somnigraph Comparison)

| Gap | Rating | Evidence |
|-----|--------|----------|
| Layered Memory | 30% | 4-layer stack (L0-L3) provides token budgeting for wake-up context. But layers are just metadata filters on the same ChromaDB store -- no actual layered processing, no decay, no consolidation between layers. Somnigraph does not have explicit layers but has richer per-memory lifecycle management. |
| Multi-Angle Retrieval | 10% | Production retrieval is vector-only (single ChromaDB query). No BM25, no FTS, no hybrid fusion. The benchmark scripts have hybrid scoring, but it is not in the product. Somnigraph has BM25 + vector + theme + PPR + learned reranker. |
| Contradiction Detection | 25% | Knowledge graph supports temporal fact invalidation (valid_from/valid_to) and the MCP server has kg_invalidate. But detection is manual -- the agent must call kg_invalidate. No automated contradiction detection from text. Somnigraph detects contradictions during sleep via LLM classification and maintains contradiction edges. |
| Relationship Edges / Graph | 35% | Two graph layers: (1) knowledge graph with typed temporal triples (SQLite), (2) palace graph with room-based tunnels across wings (derived from ChromaDB metadata). Neither is used during retrieval. Somnigraph has coref/contradiction/temporal/Hebbian edges integrated into PPR and the reranker. |
| Sleep Process / Consolidation | 0% | No consolidation, no sleep, no background processing. Content is stored as-is at mine time and never re-processed. Somnigraph has 3-phase sleep (NREM dedup + REM synthesis + archiving). |
| Reference Index | 15% | Source files tracked via metadata. Knowledge graph links facts to source closets. But no citation or provenance tracking beyond simple source_file strings. |
| Temporal Trajectories | 25% | Knowledge graph has valid_from/valid_to for temporal fact validity. Timeline queries are available. But temporal information is not used in retrieval (benchmark temporal boost is not in production). Somnigraph has temporal features in the reranker and temporal edges in the graph. |
| Feedback Loop | 0% | No feedback mechanism. No way for the user or agent to rate retrieval results. No learning from usage patterns. Somnigraph has recall_feedback leading to EWMA + UCB + Hebbian + decay adjustment. |
| Learned Reranker | 5% | Benchmark scripts have LLM-based reranking (Haiku/Sonnet), but this is a prompt-based approach, not a learned model. Not available in production. Somnigraph has a 26-feature LightGBM trained on 1032 ground truth queries (NDCG=0.7958). |

## 7. Relevance to Somnigraph

### What it does better

1. **Benchmark infrastructure**: 4 benchmark runners with reproducible scripts, honest accounting of overfitting, train/test splits. Somnigraph's LoCoMo benchmark is comparable in rigor, but MemPalace covers more benchmarks and the methodological transparency in BENCHMARKS.md is a higher standard for honest reporting.

2. **Zero-API baseline**: Proving that raw verbatim storage + good embeddings scores 96.6% on LongMemEval is a valuable finding. It establishes the "simple baseline" that all memory systems should compare against.

3. **Format normalization**: Supports 5 chat export formats (Claude Code JSONL, Claude.ai JSON, ChatGPT JSON, Slack JSON, plain text) with clean conversion to a common transcript format. Somnigraph ingests memories via agent interaction, not bulk import.

4. **Onboarding flow**: Interactive first-run setup that seeds the entity registry and wing structure. Somnigraph has no onboarding -- it learns from usage.

5. **AAAK compression for context budgeting**: The L0+L1 wake-up context (~170 tokens) is an interesting approach to the "how does the AI know anything at startup" problem. Somnigraph's startup_load serves a similar purpose but at higher token cost.

### What Somnigraph does better

1. **Retrieval quality**: Somnigraph's hybrid retrieval (BM25 + vector + theme + PPR + 26-feature LightGBM reranker) is fundamentally more sophisticated than MemPalace's single ChromaDB query. The gap between MemPalace's production retrieval and its benchmark retrieval reveals this clearly -- all the interesting retrieval work lives in benchmark scripts that users never see.

2. **Consolidation**: Somnigraph has a 3-phase sleep pipeline. MemPalace has none. Memories are never re-evaluated, merged, or pruned.

3. **Feedback loop**: Somnigraph's recall_feedback leading to EWMA + UCB + Hebbian is entirely absent from MemPalace. No learning from usage.

4. **Graph integration into retrieval**: Somnigraph uses PPR and graph features in the reranker. MemPalace's two graph structures (KG and palace graph) are queryable but disconnected from the search pipeline.

5. **Evaluation rigor on the product itself**: Somnigraph evaluates the actual production retrieval pipeline (26-feature reranker, ground truth, LoCoMo end-to-end QA). MemPalace's impressive benchmarks evaluate code that is not in the product.

## 8. Worth Stealing (Ranked)

### 1. Train/test split discipline for benchmarks
**Effort: low. Impact: medium.**
The explicit 50/450 dev/held-out split with clear contamination rules is good methodology. Somnigraph's LoCoMo evaluation should adopt a similar discipline if it does not already have one.

### 2. Honest benchmarking template
**Effort: low. Impact: medium.**
The BENCHMARKS.md structure -- progression with each improvement explained, explicit overfitting acknowledgment, structural guarantee caveats -- is a good template for how Somnigraph documents its own benchmark results.

### 3. Multi-format conversation import
**Effort: medium. Impact: low for Somnigraph.**
The `normalize.py` module cleanly handles 5 chat formats. If Somnigraph ever needs bulk conversation import (e.g., for LoCoMo or PERMA evaluation), this pattern is well-implemented.

### 4. Wing/room metadata for retrieval scoping
**Effort: medium. Impact: low.**
The 34% retrieval boost from filtering to the right wing+room is real (even if the oracle routing inflates the number). Somnigraph's themes serve a similar purpose but less explicitly. Worth considering if theme-based pre-filtering would help narrow search before the reranker runs.

## 9. Not Worth It

1. **AAAK compression**: Drops retrieval from 96.6% to 84.2%. The compression is impressive as a novelty but counterproductive for embedding-based search. Somnigraph should not adopt lossy compression for stored memories.

2. **Palace metaphor as architectural feature**: Wings/halls/rooms are metadata tags on a flat ChromaDB store. The metaphor adds cognitive load without architectural depth. Somnigraph's theme system achieves the same scoping without the metaphor layer.

3. **Knowledge graph as separate manual store**: Requiring the agent to manually add/invalidate facts in a parallel store is fragile. Somnigraph's automatic graph edge creation during sleep is more robust. Adopting a manual KG alongside the automatic graph would duplicate effort.

4. **Entity registry with Wikipedia lookup**: Cute feature (disambiguate "Max" the person vs. "max" the function), but the problem is solvable with simpler context heuristics. The Wikipedia API dependency for entity disambiguation is overkill.

5. **Hooks for auto-save**: The Stop/PreCompact hooks tell the AI to save memories every N messages. This is a workaround for not having a feedback-driven memory system -- if retrieval quality is high, the agent learns what to remember naturally.

## 10. Impact on Implementation Priority

**No change to Somnigraph priorities.** MemPalace operates in a fundamentally different design space:

- MemPalace is a **bulk ingestion system** (mine files, search later). Somnigraph is an **interactive memory system** (remember/recall in real-time, consolidate overnight).
- MemPalace's strongest contribution (benchmark methodology) is already well-covered by Somnigraph's own evaluation infrastructure (LoCoMo benchmark, ground truth pipeline, reranker NDCG).
- MemPalace's retrieval pipeline is too simple to inform Somnigraph's retrieval development. The interesting benchmark-only retrieval code (hybrid scoring, temporal boost, person name boost) is substantially less sophisticated than what Somnigraph already has in production.

The benchmarking honesty is worth noting as a standard to aspire to, but it does not change what Somnigraph should work on next. P2 (reranker iteration) and P4 (LoCoMo ablations) remain the right priorities.

## 11. Connections

- **LongMemEval** (`longmemeval.md`): MemPalace's primary benchmark. The 96.6% raw baseline is a useful reference point for "how well does naive verbatim + embeddings work."
- **LoCoMo** (`locomo.md`): MemPalace's honest 60.3% top-10 LoCoMo score vs. Somnigraph's 95.4% R@10 (Level 5b) demonstrates the retrieval quality gap between single-channel vector search and hybrid multi-channel retrieval with a learned reranker.
- **Mem0** (`mem0-paper.md`): MemPalace explicitly positions against Mem0's extraction approach ("2x Mem0 on ConvoMem"). The finding that raw verbatim outperforms LLM extraction aligns with the structural-memory paper's finding that mixed memory (not pure extraction) works best.
- **cortex-engine** (`cortex-engine.md`): Both are MCP memory servers, but cortex-engine has consolidation, FSRS scheduling, and prediction-error gating -- features MemPalace entirely lacks. Cortex-engine is architecturally more ambitious while MemPalace is better evaluated.
- **AWM** (`awm.md`): AgentWorkingMemory's 10-phase retrieval pipeline is the opposite end of the complexity spectrum from MemPalace's single ChromaDB query.
- **EXIA GHOST** (`exia-ghost.md`): Both achieve strong LoCoMo numbers through different means. EXIA GHOST's 89.94% comes from extraction quality + prompt engineering while MemPalace's comes from verbatim storage + benchmark-specific heuristics.
- **Structural Memory** (`structural-memory.md`): The paper's finding that mixed memory structures outperform any single approach supports MemPalace's verbatim-first thesis (do not throw information away) while also showing why MemPalace's vector-only retrieval is a ceiling.

## 12. Summary Assessment

**Maturity**: Early (7 commits over 3 days, April 2026). The code is clean and well-documented but very young. Version mismatch (3.0.0 in pyproject.toml, 2.0.0 in __init__.py) suggests rapid iteration without consistency checks.

**Contributions**: Two genuine contributions to the field:
1. Establishing that raw verbatim storage + default embeddings is a 96.6% LongMemEval baseline that most memory systems fail to beat -- a useful "is your complexity justified?" benchmark.
2. Setting a high standard for benchmark honesty (explicit overfitting acknowledgment, structural guarantee caveats, train/test splits).

**Weaknesses**:
1. **Benchmark/product gap**: The most impressive results (100% LongMemEval, hybrid scoring, temporal boost, LLM reranking) exist only in benchmark scripts. The product ships single-channel ChromaDB search.
2. **No consolidation or feedback**: Memories are write-once. No sleep, no decay, no learning from usage, no deduplication beyond similarity threshold.
3. **AAAK is counterproductive**: Marketed as a compression breakthrough but degrades retrieval by 12.4pp.
4. **Knowledge graph is disconnected**: Manually populated, not used during search, not integrated with the palace.
5. **Contradiction detection is manual**: The README implies automated detection, but it requires pre-populated ground truth in the KG.

**Bottom line**: A well-benchmarked verbatim storage system that proves simple baselines are undervalued, but whose product retrieval is far simpler than its benchmark retrieval suggests.
