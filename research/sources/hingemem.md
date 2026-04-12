# HingeMem — Boundary Guided Long-Term Memory with Query Adaptive Retrieval

*Generated 2026-04-11 by Opus agent reading arXiv:2604.06845*

---

## Paper Overview

- **Paper**: Yijie Zhong, Yunfan Gao, and Haofen Wang. 2026. "HingeMem: Boundary Guided Long-Term Memory with Query Adaptive Retrieval for Scalable Dialogues." In *Proceedings of the ACM Web Conference 2026 (WWW '26)*, April 13--17, Dubai, UAE. ACM. doi:10.1145/3774904.3792089
- **Authors**: Yijie Zhong (Tongji University), Yunfan Gao (Shanghai Research Institute for Intelligent Autonomous Systems, Tongji University), Haofen Wang (Tongji University, corresponding author)
- **Code**: Not explicitly linked in the paper. Appendix A references open-source baselines but no HingeMem repo URL is provided.
- **Venue**: WWW '26 (peer-reviewed conference paper)

**Problem addressed**: Existing long-term memory systems either (a) use continuous summarization/writing that produces redundancy and loses detail, or (b) build OpenIE-based graphs with fixed Top-K retrieval that can't adapt to query type. Both waste compute and struggle with diverse query categories (single-hop, multi-hop, temporal, open-domain, adversarial).

**Core approach**: Neuroscience-inspired architecture with two components: (1) a "cortex" that performs event segmentation to detect boundaries across four dimensions (person, time, location, topic), and (2) a "hippocampus" that constructs hyperedges connecting element nodes within each boundary-delimited segment. Retrieval uses query analysis to determine *what* to retrieve (which element interfaces) and *how much* (adaptive stopping based on query type classification).

**Key claims**:
- ~5% overall F1 improvement on LoCoMo over baselines, with >10% on multi-hop (Table 1)
- 68% computational cost reduction vs. HippoRAG2 (Figure 4)
- Operates without category-specific question templates
- Robust across model scales from Qwen3-0.6B to Qwen-Flash (128K context)

---

## Architecture

### Storage & Schema

Memory is organized as a hypergraph with two entity types:

**Element nodes** (four types):
- **Person**: `n = (name, mentions, [granularity])` — unique identifier + list of mentions in the segment
- **Time**: Same schema, with granularity field for temporal resolution (year/month/day/minute/approx)
- **Location**: Same schema
- **Topic**: `(label, mentions)` — extracted via topic clustering prompt

**Hyperedges**: A hyperedge `h^i = (P^i, T^i, L^i, C^i, d^i, r^i)` connects a subset of person, time, location, and topic nodes from a single segment, plus a description `d^i` and the boundary reason `r^i`. Each hyperedge corresponds to a boundary-delimited segment of dialogue.

**Salience scores**: Each node gets a salience score based on three dimensions:
- **Frequency**: How often the node appears across memory
- **Centrality**: Degree within the hypergraph structure
- **Diversity**: Co-occurrence across different contexts (cross-hyperedge presence)

### Memory Types

Two tiers of topic classification, applied globally across all hyperedges:

- **Common topics** (`C_common`): Widely documented, recurring across periods. High frequency.
- **Rare topics** (`C_rare`): Niche, localized, sparsely documented. Low frequency.

This distinction drives retrieval strategy — common topics get broad recall, rare topics get precision-focused retrieval. The classification uses a theme clustering prompt (`P_TC`) applied to all hyperedge descriptions (Section 3.2.2).

### Write Path

The write path is a two-stage pipeline (Figure 2, Section 3.2):

**Stage 1 — Boundary Extraction (Cortex)**:
1. Process each session `s_i` through a boundary extraction prompt (`P_BE`)
2. Detect boundaries triggered by changes in person, time, location, or topic
3. Boundary types: `{change_person, person_time, time_change, location_topic_shift, explicit_marker}` (from Appendix A.1)
4. Segment dialogue into boundary-delimited chunks
5. Within each segment, extract element nodes and form hyperedges

**Stage 2 — Memory Construction (Hippocampus)**:
1. Merge new nodes with existing memory by matching on unique identifiers
2. Normalize timestamps to ISO 8601 (e.g., "yesterday" → absolute date)
3. Compute salience scores for all nodes
4. **Deduplication via Jaccard similarity**: Compare node sets of hyperedge pairs; merge when Jaccard > 0.8 (Equation 4). This prevents redundant storage of similar segments.
5. Classify topics into common vs. rare categories

**Statistics on LOCOMO** (Table 4, Appendix B): Avg. 12.8 persons, 59.6 times, 34.6 locations, 81.5 topics, 103.2 hyperedges per conversation.

### Retrieval

Retrieval is a three-stage pipeline (Section 3.3):

**Stage 1 — Retrieval Plan Generation**:
- A query analysis prompt (`P_Q`, Appendix A.2) classifies the query into one of three types:
  - **Recall priority**: Coverage/enumeration/counting questions. Retrieve as many relevant hyperedges as possible.
  - **Precision priority**: Questions seeking a single best piece of evidence. Retrieve few, high-confidence results.
  - **Judgment**: Yes/no/existential decisions. Balanced retrieval.
- The prompt also extracts: query elements (person, time, location, topic), element priority ordering, and relevant names (for node matching)

**Stage 2 — Hyperedge Rerank**:
- Initial candidates: Match query elements to person/time/location/topic nodes → find connected hyperedges → compute initial similarity score `ξ` via embedding similarity
- Reranking incorporates:
  - Node salience values of involved elements (weighted by priority `p` from retrieval plan)
  - Topic penalty term `Ω_T`: Distance from query to rare/common terms, applied via weighted softmax (Equation 7)
  - This boosts rare-topic hyperedges and prevents common-topic overshadowing

**Stage 3 — Adaptive Stop**:
- Different stopping criteria per query type (Figure 3):
  - **Recall priority**: Select hyperedges before the score inflection point (where `ξ^{i-1} - ξ^i < λ_{knee}` and `ξ > max(ξ)/2`). λ_knee = 0.1.
  - **Precision priority**: Select hyperedges where score > 80% of maximum score
  - **Judgment**: Apply softmax, keep hyperedges > 80% of max softmax score
- This replaces fixed Top-K with content-dependent stopping

### Consolidation / Processing

No explicit offline consolidation step. All processing happens at write time (boundary extraction + node merging + deduplication) and at query time (reranking). The Jaccard-based merge at write time (threshold 0.8) is the only compression mechanism.

There is no sleep-like process, no temporal decay, no forgetting mechanism. The memory grows monotonically — deduplication prevents redundancy but nothing is ever removed or down-weighted over time.

### Lifecycle Management

Not addressed. No decay, no archival, no capacity management. The system assumes all memories remain equally accessible indefinitely. For LOCOMO's scale (~294 turns per conversation, ~10 conversations), this is manageable. For production-scale deployment (thousands of sessions), the lack of lifecycle management would be a significant limitation.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| ~5% overall improvement on LoCoMo | Table 1: 61.1 F1 vs. best baseline (LangMem 54.7 F1 cat-format, HippoRAG2 59.2 F1 cat-format). J-score: 75.1 vs. best baseline 70.6 (HippoRAG2). | **Partially supported.** The 5% claim is reasonable for F1 against most baselines. However, the comparison is complicated: HippoRAG2 at 59.2 F1 (cat-format) is only 1.9pp behind. The J-score gap is more convincing (75.1 vs 70.6). The "~20% relative improvement" claim from the abstract appears to refer to specific categories (multi-hop: 53.6 F1 vs baseline range of 38-48). |
| >10% multi-hop improvement | Table 1: Multi-hop F1 = 53.6 vs. HippoRAG2 46.4, LangMem 45.6, Mem0 47.8. | **Supported.** Clear multi-hop gains, especially vs. non-graph methods. The 7.2pp gap over HippoRAG2 is meaningful. |
| 68% compute reduction vs. HippoRAG2 | Figure 4: HippoRAG2's bubble is ~3x larger than HingeMem's. Both at similar F1 (~60). | **Plausible but weakly evidenced.** The "68%" figure appears to come from the question-answering token cost (Figure 4 shows total tokens including memory construction). HippoRAG2 uses unconstrained OpenIE which generates large graph memories. The comparison is fair in that both are graph-based, but the metric is token-cost-focused, not wall-clock time. |
| Operates without category-specific templates | Appendix A.2: Single query analysis prompt handles all types. Other baselines (LOCOMO, LangMem) use different templates per question category. | **Supported and practically important.** The adaptive retrieval replaces hardcoded category routing. In production, you rarely know the question category a priori. |
| Robust across model scales (0.6B to 128K) | Figure 6: HingeMem shows consistent improvement from Qwen3-0.6B through Qwen-Flash. Other methods fluctuate. | **Supported.** The consistency across scales is notable — most baselines show non-monotonic performance as model size changes. |
| Boundary-guided memory > continuous writing | Table 3 ablation: RAG+BM (boundary memory) = 57.4 vs RAG+TM (text memory) = 44.6 overall F1. | **Strongly supported.** The 12.8pp gap is the largest single-component contribution in the ablation. Segmentation matters more than retrieval sophistication. |

---

## Relevance to Somnigraph

### What HingeMem does that Somnigraph doesn't

1. **Event segmentation at write time**: HingeMem applies Event Segmentation Theory to detect boundaries across four explicit dimensions. Somnigraph stores memories as atomic units (one per `remember()` call) without automatic segmentation of conversation streams. The boundary detection is LLM-mediated but structured — it produces typed boundaries, not just summaries.

2. **Hyperedge indexing**: The four-dimensional hypergraph (person × time × location × topic) creates a structured index that supports multi-path retrieval. A query about "Alex's meeting last Thursday" can route through person AND time dimensions simultaneously. Somnigraph's graph has typed edges from sleep consolidation but doesn't have multi-dimensional indexing of this kind.

3. **Query-type-adaptive retrieval depth**: The three-way query classification (recall/precision/judgment) with different stopping criteria per type is a clean design. Somnigraph replaced its cliff detector with a user-specified `limit` parameter — the system doesn't reason about how much context a query needs.

4. **Topic rarity awareness**: The common/rare topic distinction prevents popular topics from drowning out niche information during retrieval. The softmax-weighted rarity penalty (Equation 7) is a principled approach. Somnigraph's reranker learns feature weights but doesn't have an explicit rarity signal.

### What Somnigraph does better

1. **Learned reranker**: Somnigraph's 26-feature LightGBM reranker, trained on 1032 real queries, substantially outperforms HingeMem's embedding-similarity + salience heuristic. HingeMem's reranking formula (Equations 6-7) is hand-tuned with fixed weights — exactly the approach Somnigraph moved away from (and documented as "what didn't work").

2. **Feedback loop**: Somnigraph has explicit retrieval feedback (`recall_feedback`) that feeds back into reranker training and UCB exploration. HingeMem has no feedback mechanism — it cannot learn from its retrieval mistakes.

3. **Offline consolidation**: Somnigraph's three-phase sleep pipeline (NREM pairwise classification, REM gap analysis) performs relationship detection, contradiction resolution, and temporal evolution tracking. HingeMem's "hippocampus" does structural merging at write time but no deeper semantic processing.

4. **Lifecycle management**: Somnigraph has biological decay, dormancy detection, and archival. HingeMem has none — memories persist indefinitely with no capacity management.

5. **Graph sophistication**: Somnigraph's graph has typed edges (supports, contradicts, temporal_evolution, etc.) discovered by LLM during sleep. HingeMem's hypergraph has structural edges (element membership) but no semantic edge types. PPR expansion and betweenness centrality provide richer graph signals than hyperedge co-membership.

6. **Retrieval performance**: Somnigraph's Level 5b achieves R@10=95.4% on LoCoMo. HingeMem doesn't report retrieval metrics — only end-to-end QA F1 (61.1%). Direct comparison isn't possible, but Somnigraph's 85.1% overall QA accuracy (Opus judge) vs. HingeMem's 75.1 J-score suggests Somnigraph has a meaningful QA edge as well. (Caveat: different judges and scoring methodologies make this an approximate comparison.)

---

## Worth Stealing (ranked)

### 1. Query-type-adaptive retrieval depth (medium effort)

**What**: Classify incoming queries into recall/precision/judgment categories and apply different retrieval strategies (how many results, what confidence threshold). This replaces the current flat `limit` parameter with content-aware stopping.

**Why**: Somnigraph currently relies on the caller to set `limit` (default 5). The caller often doesn't know whether they need broad coverage or a single precise answer. HingeMem's ablation shows +2.6pp F1 from adaptive stopping alone (Table 3, row 4 vs 5). The three query types map well to real usage patterns: "what have I said about X" (recall), "what was the decision on Y" (precision), "did I ever mention Z" (judgment).

**How**: Add a lightweight query classification step in `impl_recall()` — either a simple prompt or a small classifier on query features (query_length, question words, etc.). Map to different reranker score thresholds or result counts. The reranker already produces calibrated scores; the missing piece is using them for adaptive cutoff rather than fixed-K. This is adjacent to the cliff detector that was removed — but the key difference is that HingeMem's approach uses query semantics, not score distribution, to decide depth. Could be integrated as a reranker feature (query_type) rather than a post-hoc filter.

### 2. Topic rarity signal for reranker (low effort)

**What**: Compute a rarity score for each memory's themes/topics and use it as a reranker feature. Common topics get slight downweighting; rare topics get preserved.

**Why**: This addresses the "popular topic drowning" problem. In Somnigraph's current system, frequently-discussed topics dominate BM25 and vector retrieval, making it harder to surface niche memories. The reranker could learn the right balance, but it needs the signal. HingeMem's common/rare classification (Section 3.2.2) is simple but effective.

**How**: Compute IDF-like rarity scores over theme frequencies in the memory database. Add as a reranker feature (`theme_rarity` or `topic_idf`). The infrastructure is already there — `query_idf_var` is the top new feature by importance in the 26-feature model. A memory-side IDF signal would complement the existing query-side one. Could be computed during sleep consolidation and cached.

### 3. Boundary-triggered segmentation for conversation ingestion (high effort)

**What**: When ingesting long conversations (e.g., LoCoMo benchmark, or hypothetical auto-capture from chat logs), detect event boundaries across person/time/location/topic dimensions and create structured memory units aligned to segments rather than individual turns.

**Why**: HingeMem's ablation shows boundary-guided memory is worth +12.8pp F1 over plain text memory (Table 3). Somnigraph's current unit is the individual `remember()` call — typically one fact or decision. For batch ingestion of conversation history, event-segmented chunks would be more natural units. The vocabulary gap problem in multi-hop retrieval (documented in `docs/multihop-failure-analysis.md`) could be partially addressed by wider segments that co-locate related turns.

**How**: This would be a new ingestion mode, not a replacement for the current `remember()` API. A boundary extraction prompt (similar to HingeMem's Appendix A.1) would process conversation transcripts and emit structured segments. Each segment becomes a memory with richer context than a single turn. The four-dimensional boundary detection (person/time/location/topic) is a good framework. High effort because it requires a new ingestion pipeline and decisions about how segmented memories interact with the existing graph.

---

## Not Useful For Us

**Hyperedge structure as primary index**: HingeMem's hypergraph is the *only* retrieval index — queries route through element nodes to hyperedges. Somnigraph's hybrid BM25 + vector + graph approach is more flexible. Replacing it with a hypergraph would lose the learned reranker's ability to fuse heterogeneous signals. The hyperedge structure is interesting for batch-ingested conversations but doesn't map to Somnigraph's incremental single-memory-at-a-time storage pattern.

**Salience scoring formula**: The frequency × centrality × diversity formula (Section 3.2.2) is a reasonable heuristic but strictly inferior to a learned reranker. Somnigraph already has graph-derived features (betweenness, edge count) that serve the same purpose with learned weights.

**The specific boundary types**: The five boundary triggers (change_person, person_time, time_change, location_topic_shift, explicit_marker) are tuned for multi-party dialogue. Somnigraph's memories are typically single-user, first-person — the "person" and "location" dimensions are much less relevant. The concept of event segmentation is worth stealing; the specific taxonomy is not.

**No-category-template approach**: HingeMem's single unified prompt is positioned as an advantage over category-specific templates. Somnigraph doesn't use category-specific retrieval templates at all — the reranker handles query diversity through features, not prompts. This is a solved problem for us.

---

## Connections

- **Event Segmentation Theory**: HingeMem cites Zacks & Swallow (2007) and Baldassano et al. (2017) on hippocampal-cortical boundary detection. This is the same neuroscience literature that motivates Somnigraph's sleep pipeline, but applied at write time rather than consolidation time. The two approaches are complementary — segmentation at ingest, relationship discovery during sleep.

- **HippoRAG/HippoRAG2**: HingeMem positions itself against HippoRAG2's unconstrained OpenIE graph construction, which produces large graphs with high retrieval overhead. Somnigraph's graph is similarly constrained (edges typed during sleep, not open-ended extraction), making both systems closer to HingeMem's philosophy of structured-over-exhaustive.

- **LightMem**: Both HingeMem and LightMem use segmentation strategies (HingeMem: event boundaries, LightMem: topic-aware segmentation). HingeMem's boundary detection is more principled (four explicit dimensions vs. LightMem's topic clustering), but LightMem adds temporal decay and offline consolidation that HingeMem lacks.

- **LoCoMo benchmark results**: HingeMem reports 61.1 F1 overall on LoCoMo with GPT-4o (Table 1). For comparison context: Somnigraph's 85.1% accuracy uses Opus judge with corrected GT, and LightMem reports 72.99% (GPT-4o-mini). The different metrics (F1 vs accuracy vs J-score) and judges make direct comparison imprecise, but the ranking appears to be Somnigraph > LightMem > HingeMem > baselines.

- **Somnigraph's cliff detector removal**: HingeMem's adaptive stopping is conceptually related to the cliff detector Somnigraph removed — both try to determine how many results are "enough." The cliff detector failed because score distributions don't predict content sufficiency (R-squared < 0). HingeMem's approach is different in kind: it uses query semantics (type classification) rather than score statistics. This is worth revisiting.

- **Somnigraph's `query_idf_var` feature**: The top new feature in the 26-feature retrain. HingeMem's topic rarity signal (common vs. rare) operates on the memory side rather than the query side. The two are complementary — query IDF variance measures how discriminative the query is, while topic rarity measures how unusual the memory's content is.

---

## Summary Assessment

**Relevance**: Medium. HingeMem addresses a real problem (structured memory construction + adaptive retrieval) but its solutions are mostly simpler versions of things Somnigraph already does. The event segmentation idea is the most transferable contribution.

**Quality**: Solid conference paper with clear ablation study and reasonable experimental design. The LoCoMo evaluation follows the standard protocol. Limitations: no retrieval-only metrics (R@K, MRR), no code release, the "~20% improvement" headline is cherry-picked from multi-hop rather than overall results.

**What to remember**: The query-type-adaptive retrieval depth idea deserves consideration for Somnigraph — it's a cleaner approach to the "how many results" problem than either fixed-K or score-based cliff detection. The topic rarity signal is a quick win for the reranker feature set. The boundary segmentation concept is interesting for any future batch-ingestion pipeline.

**What to skip**: The hypergraph indexing structure, the salience heuristics, and the specific boundary taxonomy are all either redundant with or inferior to Somnigraph's existing approach.

**Overall rating**: Medium relevance, medium quality. A competent system paper with one good idea (adaptive retrieval depth via query classification) and one solid empirical finding (boundary-guided segmentation >> continuous writing, +12.8pp F1). The neuroscience framing is mostly decorative — the actual implementation is "segment by detected changes, index by entity type, classify query, adapt retrieval depth."
