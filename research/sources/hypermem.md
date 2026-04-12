# HyperMem -- Hypergraph Memory for Long-Term Conversations

*Generated 2026-04-11 by Opus agent reading arXiv:2604.08256*

---

## Paper Overview

**Paper**: Yue, Hu, Sheng, Zhou, Zhang, Liu, Guo, Deng. "HyperMem: Hypergraph Memory for Long-Term Conversations." arXiv:2604.08256, April 2026. 16 pages (9 main + 7 appendix).
**Authors**: Juwei Yue, Chuanrui Hu, Jiawei Sheng, Zuyi Zhou, Wenyuan Zhang, Tingwen Liu, Li Guo (Institute of Information Engineering, Chinese Academy of Sciences; University of Chinese Academy of Sciences) and Yafeng Deng (EverMind AI).
**Code**: Claimed forthcoming ("Our source code is about to be released"), not yet available.

**Problem addressed**: Existing chunk-based RAG and graph-based memory systems rely on pairwise relationships, which cannot capture *high-order associations* -- joint dependencies among three or more content elements. When facts about a topic are scattered across temporally distant episodes, pairwise edges fragment the retrieval landscape and miss the holistic coherence needed for multi-hop and temporal reasoning.

**Core approach**: HyperMem organizes memory as a three-level hypergraph: **Topics** (thematic anchors spanning months), **Episodes** (temporally contiguous dialogue segments), and **Facts** (atomic queryable assertions extracted from episodes). Hyperedges -- which connect arbitrary sets of nodes, not just pairs -- bind all episodes sharing a topic and all facts within an episode. This enables coarse-to-fine retrieval: topic-level search narrows the space, episode-level retrieval preserves temporal context, and fact-level retrieval provides precise answers. Embedding propagation across hyperedges enriches node representations with topical context.

**Key claims**:
- 92.73% overall accuracy on LoCoMo (GPT-4o-mini judge), beating HyperGraphRAG (86.49%) by 6.24pp and MIRIX (85.38%) by 7.35pp (Table 1). **Note**: GPT-4o-mini as judge is lenient -- Somnigraph's own testing shows it accepts 62.81% of intentionally vague wrong answers. This inflates all numbers in Table 1 uniformly, so relative comparisons hold but absolute accuracy is overstated.
- 96.08% on single-hop, 93.62% on multi-hop, 89.72% on temporal, 70.83% on open-domain (Table 1).
- Token efficiency: "Episode + Fact" config reaches 89.48% at 2.5x Mem0's tokens; full config reaches 92.73% at 7.5x (Figure 5). Compared to GraphRAG (35.3x tokens for 67.60%) and HyperGraphRAG (26.3x for 86.49%), this is notably efficient.
- Ablation shows episode context is the most critical component (-3.76% without it), especially for temporal reasoning (-5.61%) (Table 2, Figure 3).

---

## Architecture

### Storage & Schema

Three node types forming a hypergraph H = (V^T ∪ V^E ∪ V^F, E^E ∪ E^F):

- **Topic nodes** V^T: `(title, summary)`. Thematic anchors that persist across sessions.
- **Episode nodes** V^E: `(dialogue, title, episode_summary)`. Raw dialogue turns plus a concise subject line and narrative summary.
- **Fact nodes** V^F: `(content, potential, keywords)`. `content` is the atomic assertion, `potential` lists anticipated query patterns, `keywords` are representative terms for BM25 retrieval.

Two hyperedge types:
- **Episode hyperedges** E^E: connect all episode nodes within the same topic to the topic node. Each node has importance weight w^E ∈ [0,1] assigned by the LLM during topic aggregation.
- **Fact hyperedges** E^F: connect all fact nodes belonging to the same episode. Each node has importance weight w^F ∈ [0,1] assigned during fact extraction.

### Memory Types

The hierarchy is **Topic > Episode > Fact**, not a type taxonomy. This is structurally similar to Somnigraph's implicit hierarchy (themes > memories > extracted claims), but HyperMem makes it the explicit storage and retrieval primitive.

No distinction between episodic/semantic/procedural -- all memories are factual assertions grounded in episodes. The `potential` field on fact nodes (anticipated queries) is the closest thing to a semantic enrichment layer.

### Write Path

Three-stage LLM-driven pipeline (Algorithm 1, Section 3.2):

**Stage 1: Episode Detection** (Section 3.2.1). Streaming boundary detection: each incoming dialogue turn is buffered and an LLM evaluates (1) semantic completeness, (2) time gap, (3) linguistic signals (topic transitions, closures). Outputs `should_end` or `should_wait`. On `should_end`, an Episode node is created from the buffer. Prompt template shown in Figure 6 -- notably explicit about special rules (greetings + topic = one episode, ignore pleasantries).

**Stage 2: Topic Aggregation** (Section 3.2.2). Each new episode is compared to historical episodes using lexical + semantic similarity (details in Section 3.3.1). Three cases:
1. No similar episodes exist → create new Topic.
2. Similar episodes found but different topic → create new Topic with those episodes.
3. Similar episodes found, same topic exists → update existing Topic (regenerate title/summary incorporating new episode).

The LLM assigns episode importance weights. Prompt template in Figure 7 -- requires all four criteria (same event/theme, narrative continuity, identity of core subject, temporal tolerance) for aggregation.

**Stage 3: Fact Extraction** (Section 3.2.3). For each topic, the LLM extracts atomic queryable facts from all episodes within that topic. Each fact includes: `content` (the assertion), `potential` (anticipated queries), `keywords` (for BM25), and `importance_weight`. Facts are anchored to source episodes for provenance. Prompt in Figure 8 emphasizes "answerable facts" over narrative context.

### Retrieval

**Offline Index Construction** (Algorithm 2, Section 3.3.1):
1. Build dual BM25 + vector indices for all node types (topic, episode, fact).
2. Compute hyperedge embeddings as weighted aggregation of constituent node embeddings (Equation 2).
3. Propagate: each node's embedding is updated with a weighted sum of its incident hyperedge embeddings (Equation 3), controlled by hyperparameter λ.

**Online Retrieval** (Algorithm 3, Section 3.3.2) -- coarse-to-fine with progressive top-k:

- **Stage 1 (Topic)**: Score all topic nodes via RRF(BM25, cosine). Select top-k^T topics. (Default k^T=10.)
- **Stage 2 (Episode)**: Expand selected topics to constituent episodes via hyperedges. Score episodes via RRF(BM25, cosine) with propagated embeddings. Rerank. Select top-k^E episodes. (Default k^E=10.)
- **Stage 3 (Fact)**: Expand selected episodes to constituent facts via hyperedges. Score facts via RRF(BM25, cosine). Rerank. Select top-k^F facts. (Default k^F=30.)

The reranker is Qwen3-Reranker-4B (a cross-encoder), applied at each stage after RRF. This is a fixed pretrained model, not a learned/fine-tuned reranker.

**Response Generation**: The response context is built from fact `content` fields, augmented with episode `summary` fields for narrative grounding. This avoids injecting raw dialogue transcripts, saving tokens. GPT-4.1-mini generates the final answer with chain-of-thought.

### Consolidation / Processing

None. All processing is online/streaming -- episode detection, topic aggregation, and fact extraction happen as dialogue arrives. There is no offline consolidation, sleep cycle, or scheduled maintenance.

### Lifecycle Management

No decay, forgetting, or archival mechanisms described. Topic nodes persist indefinitely and accumulate episodes. No feedback loop or retrieval-based scoring adjustment.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| 92.73% overall on LoCoMo | Table 1, avg of 3 runs, GPT-4o-mini judge | Judge is lenient (see above). But all baselines use the same judge, so relative ordering is valid. |
| Beats MIRIX (85.38%) by 7.35pp | Table 1. MIRIX result from Wang and Chen 2025, reproduced using GPT-4.1-mini | Cross-model comparison: HyperMem uses GPT-4o-mini judge, MIRIX uses GPT-4.1-mini. Not strictly apples-to-apples. |
| 93.62% on multi-hop | Table 1 | This is the headline result. Multi-hop is where hyperedges should matter most, and the gap over LightRAG (84.04%) is substantial (+9.58pp). |
| Episode context is the most critical component | Ablation Table 2: -3.76% without EC, -0.91% without FC | Convincing. Temporal category drops -5.61% without EC. |
| Hierarchical retrieval (topic→episode) matters for multi-hop | Figure 3: w/o TR&ER degrades multi-hop by -5.68% | Moderate. Fact-only retrieval still achieves 87.9% on multi-hop, so the hierarchy helps but isn't dramatic. |
| Token-efficient: 92.73% at 7.5x Mem0 tokens | Figure 5 | Good efficiency relative to RAG methods (GraphRAG at 35.3x, HyperGraphRAG at 26.3x). But 7.5x Mem0 is still substantial. |
| Embedding propagation helps | λ=0.5 achieves 92.66% vs lower with λ=0 (not shown directly) | λ sensitivity in Figure 4 shows modest impact: 88.79% at λ=0 vs 92.66% at λ=0.5. The +3.87pp gain is meaningful but conflated with the fact that λ=0 also disables hyperedge-enriched representations. |

---

## Relevance to Somnigraph

### What HyperMem does that Somnigraph doesn't

1. **Explicit hierarchical retrieval with progressive narrowing**. Somnigraph retrieves memories in a single phase (BM25 + vector → RRF → reranker). HyperMem's topic→episode→fact pipeline naturally reduces the candidate pool at each stage, which is especially effective for multi-hop where relevant facts share a topic but are temporally scattered. Somnigraph's Level 5b graph-augmented retrieval addresses a similar problem (synthetic vocabulary bridges) but through a flatter mechanism.

2. **Query-anticipating fact extraction**. The `potential` field on fact nodes ("predict potential queries this fact can answer") is a form of proactive retrieval alignment. This addresses the vocabulary gap problem -- if the fact anticipates the query's wording, BM25 and vector search have better signal. Somnigraph's synthetic claims serve a similar function but without explicit query anticipation.

3. **Hypergraph embedding propagation**. Aggregating embeddings across hyperedges (Equations 2-3) gives each node contextual enrichment from its topical neighbors. This is lightweight (no neural network, just weighted averaging) and could improve vector search quality.

4. **Streaming episode boundary detection**. Somnigraph relies on the MCP client to segment memories (each `remember()` call is one memory). HyperMem automatically detects episode boundaries from dialogue flow, which is more appropriate for conversation-based use cases.

### What Somnigraph does better

1. **Learned reranker with feedback loop**. HyperMem uses a fixed pretrained cross-encoder (Qwen3-Reranker-4B). Somnigraph's LightGBM reranker is trained on real user feedback (1032 queries, NDCG=0.7958) and continuously improves via the EWMA feedback loop. This is a fundamental architectural advantage -- the reranker adapts to the user's actual retrieval patterns.

2. **Offline consolidation (sleep)**. HyperMem does all processing online. Somnigraph's NREM/REM sleep pipeline performs pairwise relationship detection, contradiction classification, merge/archive, and gap analysis -- expensive operations that would degrade online latency if done at write time.

3. **Graph edge typing and contradiction handling**. Somnigraph's typed edges (supports, contradicts, evolves, revision, derivation) encode relationship semantics. HyperMem's hyperedges are purely structural (same-topic, same-episode) with no semantic typing. HyperMem has no way to represent that two facts contradict each other.

4. **Decay and lifecycle management**. Somnigraph's biological decay, dormancy detection, and archival prevent unbounded memory growth. HyperMem's topic nodes grow monotonically with no forgetting mechanism.

5. **Explicit feedback**. Per-query utility ratings (0-1 float + durability) with UCB exploration. HyperMem has no feedback mechanism.

6. **Evaluation rigor on multi-hop retrieval**. Somnigraph's Level 5b retrieval evaluation (R@10=95.4%, MRR=0.882) uses separate retrieval and QA metrics. HyperMem reports only end-to-end QA accuracy, making it impossible to separate retrieval quality from reader quality.

---

## Worth Stealing (ranked)

### 1. Query-anticipating fact fields (low-medium effort)
**What**: During extraction (NREM sleep), generate a `potential` field listing plausible queries each extracted claim could answer. Store alongside the claim.
**Why**: Directly addresses the vocabulary gap that causes 88% of multi-hop retrieval failures. If a claim about "Nate's tournament wins" stores potential queries like "how many tournaments has Nate won", the BM25 signal improves dramatically. This is complementary to (and possibly more effective than) HyDE-style synthetic documents.
**How**: Add a `potential_queries` column to synthetic/claim nodes. During NREM extraction, prompt for 2-3 anticipated query phrasings per claim. Index these in FTS5 (high BM25 weight) and include in the embedding.

### 2. Hyperedge embedding propagation (low effort)
**What**: After computing node embeddings, propagate contextual information from co-occurring nodes via weighted averaging: h'_v = h_v + λ * Agg(h_e for e in N(v)).
**Why**: HyperMem shows +3.87pp from λ=0 to λ=0.5. For Somnigraph, this could be applied along graph edges during the sleep pipeline: each memory's embedding is adjusted toward its PPR-connected neighbors. This is much cheaper than re-embedding and could improve vector retrieval without changing the embedding model.
**How**: Post-sleep-pipeline step: for each memory with edges, compute weighted average of neighbor embeddings, blend with α weight, update stored embedding. One-time batch operation.

### 3. Coarse-to-fine retrieval with theme-level pre-filtering (medium effort)
**What**: Before BM25+vector search, first match the query against theme/topic summaries to narrow the candidate pool, then search within matched topics.
**Why**: Somnigraph already has themes on memories. Using themes as a coarse first-pass filter could reduce noise in the candidate pool, especially for queries where the theme is obvious but the vocabulary overlap is poor. HyperMem's topic top-k=10 → episode top-k=10 → fact top-k=30 pipeline shows this can be aggressive without losing recall.
**How**: Add a theme-level BM25+vector index. At recall time, optionally query themes first (top-k=5), then restrict memory search to memories tagged with matched themes. Make this a reranker feature or an optional retrieval mode rather than mandatory.

---

## Not Useful For Us

- **Streaming episode boundary detection**: Somnigraph's MCP-based architecture means the client (Claude Code) decides what constitutes a memory unit. Episode detection is architecturally inappropriate -- we don't process raw dialogue streams.

- **Hypergraph formalism**: The mathematical framework (hyperedges, weighted node sets) is more complex than needed. Somnigraph's typed edges + PPR achieve similar graph-conditioned retrieval with simpler implementation. The key insight (topical grouping) is achievable with Somnigraph's existing theme system.

- **Qwen3-Reranker-4B**: A 4B-parameter cross-encoder is massive overkill for single-user MCP memory. Somnigraph's LightGBM reranker is orders of magnitude faster, smaller, and trained on actual user data. Cross-encoder reranking is the right idea but at the wrong scale for this use case.

- **Online-only processing**: No consolidation means every write operation includes full topic matching and fact extraction. This is fine for a research benchmark but would add unacceptable latency to a live MCP server where `remember()` should return quickly.

---

## Connections

**vs. LightMem** (lightmem.md): Both use topic-level aggregation and fact extraction, but LightMem adds sleep-time consolidation (closer to Somnigraph's offline philosophy). HyperMem's hypergraph structure is more principled than LightMem's flat topic_id grouping. LightMem achieves 72.99% on LoCoMo vs HyperMem's 92.73%, though the models and configs differ.

**vs. Mem0** (mem0-paper.md): Mem0's flat extract-then-update pipeline is the simplest version of what HyperMem elaborates. HyperMem adds the hierarchical structure that Mem0g's knowledge graph attempted but with hyperedges instead of pairwise entity-relationship triples. Mem0 at 66.88% vs HyperMem at 92.73% on the same benchmark.

**vs. HippoRAG** (hipporag.md): Both use graph-based retrieval with embedding propagation. HippoRAG uses PPR on a knowledge graph (entity-relation triples), HyperMem uses hyperedges for group associations. HippoRAG 2 achieves 81.62% on LoCoMo (Table 1) vs HyperMem's 92.73%. The conceptual difference is pairwise vs. high-order associations.

**vs. A-Mem** (a-mem.md): A-Mem's Zettelkasten approach (atomic notes with links) is conceptually closer to Somnigraph's graph than to HyperMem's hierarchy. A-Mem claims LoCoMo rank-1 in its paper but the metrics aren't directly comparable (different judge, different eval setup).

**vs. Somnigraph Level 5b**: Somnigraph's synthetic vocabulary bridges serve a similar function to HyperMem's fact nodes with `potential` fields -- both create intermediate representations that improve retrieval when query-document vocabulary overlap is poor. Somnigraph's R@10=95.4% on LoCoMo retrieval suggests the underlying retrieval mechanism is competitive, but Somnigraph's end-to-end QA (85.1% with Opus judge, ~87.2% corrected GT) is lower -- this gap is attributable to reader quality and judge strictness, not retrieval.

**vs. MemOS** (memos.md): MemOS achieves 75.80% on LoCoMo (Table 1) vs HyperMem's 92.73%. MemOS has stronger lifecycle management (conflict detection, preference extraction) but weaker retrieval. Both are recent CAS-affiliated systems.

**vs. Hindsight** (hindsight-paper.md): Hindsight's four epistemically-distinct networks (experiential, declarative, prospective, reflective) provide richer memory typing than HyperMem's single type hierarchy. Hindsight scores 83.6% on LongMemEval (different benchmark).

---

## Summary Assessment

**Relevance**: Medium-high. The 92.73% LoCoMo claim is the current SOTA among published systems on that benchmark, making this paper important to understand. The query-anticipating fact extraction and embedding propagation ideas are directly applicable to Somnigraph.

**Strength**: The hypergraph formalization of "topical grouping" is clean and the three-level coarse-to-fine retrieval is well-motivated. The multi-hop results (+9.58pp over LightRAG) demonstrate that grouping temporally scattered facts under shared topics genuinely helps. Case studies (Figures 9-12) are persuasive -- especially Figure 10 where HyperMem aggregates all 7 tournament mentions across 10 months via topic hyperedges.

**Weakness**: No code released yet, making reproduction impossible. No feedback loop, no decay, no offline consolidation -- the system is a static index that grows monotonically. The reranker (Qwen3-Reranker-4B) is a large pretrained cross-encoder, not a learned model -- this means HyperMem's ranking quality is model-dependent rather than data-driven. Open-domain performance (70.83%) remains the weakest category, and the paper acknowledges this limitation ("often require external knowledge beyond the conversation history"). The evaluation uses only LoCoMo with GPT-4o-mini judge -- no LongMemEval, no PERMA, no retrieval-only metrics.

**Bottom line**: The core insight -- that topical grouping via hyperedges helps multi-hop retrieval -- is validated by the results. For Somnigraph, the actionable takeaway is not the hypergraph formalism but two specific ideas: (1) query-anticipating fields on extracted facts, and (2) lightweight embedding propagation along graph edges. Both are low-effort additions to the existing sleep pipeline.
