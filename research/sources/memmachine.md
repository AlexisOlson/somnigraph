# MemMachine -- Ground-Truth-Preserving Memory System for Personalized AI Agents

*Generated 2026-04-11 by Opus agent reading arXiv:2604.04853*

---

## Paper Overview

**Paper**: Shu Wang, Edwin Yu, Oscar Love, Tom Zhang, Tom Wong, Steve Scargall, Charles Fan. "MemMachine: A Ground-Truth-Preserving Memory System for Personalized AI Agents." arXiv:2604.04853v1, April 2026. 18 pages.
**Authors**: All from MemVerge, Inc. (enterprise infrastructure company).
**Code**: https://github.com/MemMachine/MemMachine (Apache 2.0). Evaluation scripts at https://github.com/MemMachine/MemMachine/tree/main/evaluation.

**Problem addressed**: LLM-based agents need persistent memory across sessions, but current approaches (Mem0, Zep, MemGPT) rely heavily on per-message LLM extraction, which is expensive, introduces compounding extraction errors, and destroys ground truth by replacing raw episodes with distilled summaries.

**Core approach**: MemMachine stores raw conversational episodes (individual sentences) as ground truth in a two-tier architecture: short-term memory (STM, a sliding window of recent episodes with LLM-generated summaries) and long-term memory (LTM, sentence-level indexed episodes in PostgreSQL + Neo4j + pgvector). LLMs are reserved for three targeted operations only: STM summarization, profile extraction, and optional agent-mode multi-hop reasoning. Retrieval uses "contextualized retrieval" -- expanding nucleus matches to neighboring episodes within a cluster -- followed by cross-encoder reranking and chronological sorting. The system also maintains a profile memory (semantic/user preferences) extracted from episodes.

**Key claims**:
- 0.9169 on LoCoMo with gpt-4.1-mini (Table 10). Strong claim, well-evidenced with per-category breakdowns and ablation.
- 93.0% on LongMemEvalS (Table 12, config C15 with k=100 and GPT-5-mini). Systematic 6-dimension ablation across 12 configurations.
- ~80% fewer input tokens than Mem0 (Table 8). Directly measured: 4.20M vs 19.21M.
- 93.2% accuracy on HotpotQA hard set via Retrieval Agent (Table 3). Agent mode only.
- GPT-5-mini outperforms GPT-5 by +2.6% when co-optimized with prompt design (Section 8.4.6). Surprising and well-documented finding.

---

## Architecture

### Storage & Schema

Three database backends working together:
- **PostgreSQL**: Relational storage for episodes, metadata, profile memory. SQL-based filtering.
- **pgvector**: Vector similarity search on sentence embeddings (OpenAI text-embedding-3-small).
- **Neo4j**: Graph-structured long-term memory. Sentences linked to originating episodes via `EXTRACTED_FROM`-style relational mappings.

Each episode carries: producer (user/agent/system), timestamp, session_id, custom metadata key-value pairs, and a unique identifier. Multi-tenant via `org_id/project_id` pairs with further `user_id`, `agent_id`, `session_id` isolation.

### Memory Types

1. **Short-Term Memory (STM)**: Sliding window of recent episodes. Configurable size. Generates compressed summaries via LLM when content exceeds the window. Always available without retrieval.

2. **Long-Term Memory (LTM)**: Persistent store for episodes that have exited STM. Four-stage indexing pipeline: sentence extraction (NLTK Punkt), metadata augmentation (inherited from parent episode), embedding generation (configurable model), relational mapping (sentences linked to originating episodes in Neo4j).

3. **Profile Memory (Semantic)**: Structured user preferences, facts, and behavioral patterns extracted from conversational data. Stored in SQL (PostgreSQL or SQLite). Supports contradiction-based updates -- new information can override existing profile entries.

No procedural memory type. The paper explicitly acknowledges this gap (Section 3.3) and flags it as future work.

### Write Path

**Ground-truth-preserving design**: Raw messages are stored as-is. No per-message LLM extraction or fact distillation. The system indexes at the sentence level (NLTK Punkt tokenizer, Section 4.4), creating one embedding per sentence rather than per message. This is the central architectural bet -- fine-grained indexing of raw data rather than LLM-distilled summaries.

LLMs are invoked only for:
1. STM overflow summarization (session-level, not per-message)
2. Profile extraction (structured user attributes from conversational data)
3. Agent-mode inference (optional, query-time only)

The paper claims this reduces LLM token usage by ~80% versus Mem0 (Table 8: 4.20M vs 19.21M input tokens on LoCoMo).

### Retrieval

**Staged recall pipeline** (Figure 2):
1. **STM search**: Check recent context first (always available, no retrieval cost).
2. **LTM vector search**: ANN or exact match against sentence embeddings. Matched sentences traced back to originating episodes, duplicates removed.
3. **Contextualization**: The key retrieval innovation. Nucleus episode located via embedding search, then expanded to include 1 preceding + 2 following episodes within the same session to form an "episode cluster." This addresses the embedding dissimilarity problem -- contextually important turns may have embeddings distant from the query but are conversationally adjacent to relevant turns.
4. **Deduplication**: Overlapping episodes between STM and LTM results are merged.
5. **Reranking**: Cross-encoder model (AWS Cohere rerank-v3-5-0) applied to episode clusters. The paper uses "multi-query reranking" in agent mode -- concatenating all queries (original + sub-queries) for reranking, so intermediate chain-of-query facts score well even if not directly referenced in the original query (Section 5.5).
6. **Chronological sort**: Final results ordered by timestamp for narrative coherence.

**Retrieval depth (k)**: The most impactful single parameter. k=30 is the sweet spot for LoCoMo (0.912), with diminishing/negative returns beyond k=50 (0.890). Non-monotonic: k=20 misses multi-hop evidence, k=50+ overwhelms the LLM with distractors (Section 8.4.1).

### Retrieval Agent (v0.3)

An opt-in LLM-orchestrated retrieval pipeline for multi-hop queries (Section 5). Implemented as a composable tool tree:

- **ToolSelectAgent** (root): Single LLM call classifies each query into one of three structural types.
  - **ChainOfQuery**: Multi-hop dependency chains. Up to 3 iterations of (1) retrieve, (2) sufficiency judgment + query rewrite, (3) evidence accumulation. Calibrated confidence scoring with early stopping at >= 0.8.
  - **SplitQuery**: Fan-out queries decomposed into 2-6 independent sub-queries executed concurrently via `asyncio.gather()`. Structural constraints enforce single-fact-lookup per sub-query.
  - **MemMachine**: Direct single-hop retrieval. Leaf node for simple queries.

All three strategies delegate to the same underlying declarative memory search. Prompts tuned via APO (Auto Prompt Optimization) from Agent Lightning [18].

Agent mode results: 93.2% on HotpotQA hard (Table 3), 92.6% on WikiMultiHop with noise. On LoCoMo, agent provides only +0.5pp over baseline MemMachine (90.5% vs 90.2%) -- expected since LoCoMo is predominantly single-hop.

### Consolidation / Processing

**Minimal offline processing**. No sleep cycle, no background consolidation, no edge detection, no LLM-mediated relationship extraction. The system relies on:
- STM summarization (on overflow only)
- Profile updates (when contradicting information appears)
- Sentence-level indexing at ingest time

This is a deliberate design choice -- ground truth preservation means avoiding LLM-mediated transformation of stored data.

### Lifecycle Management

- **No decay**: Episodes persist indefinitely. No forgetting mechanism, no importance scoring, no temporal decay.
- **No feedback loop**: No mechanism for users or the system to rate retrieval quality or adjust future retrieval behavior.
- **Profile updates**: Profile memory supports contradiction-based updates (new info overrides old), but episodic memory is append-only.
- **Multi-tenancy**: Project-based namespace isolation with user/agent/session sub-isolation.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| 0.9169 on LoCoMo (SOTA) | Table 10-11, per-category breakdown, comparison with 6 systems | **Strong**. Uses gpt-4.1-mini as eval-LLM (agent mode). Memory-mode score is 0.9123. Comparison uses published baselines. |
| 93.0% on LongMemEvalS | Table 12-14, 12-config ablation, per-category breakdown | **Strong**. Systematic ablation with pair-wise comparisons. Best config uses k=100 + GPT-5-mini. |
| 93.2% on HotpotQA hard | Table 3-4, per-tool breakdown | **Moderate**. Agent mode only. 500 questions from hard set. No published baseline comparison beyond their own. |
| ~80% fewer tokens than Mem0 | Table 8 | **Strong**. Direct measurement: 4.20M vs 19.21M. Apples-to-apples on LoCoMo. |
| Retrieval-stage > ingestion-stage | Table 13, LongMemEvalS ablation | **Strong**. Cumulative retrieval optimizations (+4.2% depth, +2.0% formatting, +1.8% prompt, +1.4% bias correction) far exceed ingestion (+0.8% chunking). |
| GPT-5-mini > GPT-5 with right prompt | Section 8.4.6, Tables 12-13 | **Notable**. +2.6% advantage persists across retrieval depths. Argues against common practice of reusing prompts across model upgrades. |
| Ground truth > extraction | Architectural argument + token comparison | **Moderate**. Compelling argument but no direct ablation comparing sentence-indexed raw data vs LLM-extracted facts on the same retrieval pipeline. |
| Sentence chunking helps | Table 13, +0.8% (C6 vs C9) | **Weak signal**. Small effect. Most configurations achieve 0.98-1.00 on single-session extraction regardless. |

---

## Relevance to Somnigraph

### What MemMachine does that Somnigraph doesn't

1. **Raw episode preservation as ground truth**. Somnigraph stores LLM-extracted memories (summaries, themes, entities). MemMachine stores the raw sentences and retrieves them directly. This eliminates extraction error and preserves exact wording -- critical for factual/temporal/compliance queries. The tradeoff is storage and retrieval complexity.

2. **Sentence-level indexing**. One embedding per sentence vs Somnigraph's one embedding per memory (which may be a paragraph-length summary). Finer granularity enables more precise vector matching but requires more embeddings and a contextualization step to reassemble conversational context.

3. **Retrieval Agent with query routing**. The ToolSelectAgent classifies queries into structural types (chain, fan-out, direct) and routes to specialized strategies. Somnigraph has no query-type-aware retrieval -- all queries go through the same BM25 + vector + RRF + reranker pipeline.

4. **Multi-query reranking**. When agent mode decomposes a query, all sub-queries plus the original are concatenated for the final reranking pass. This ensures intermediate evidence (needed for chain reasoning but not directly matching the original query) scores well. Somnigraph's reranker only sees the original query.

5. **Configurable embedding backend**. Supports swapping embedding models per deployment. Somnigraph has this (added in the configurable-embedding branch) but MemMachine also supports configurable reranking models.

6. **User-query bias correction**. Prepending "user:" to search queries shifts retrieval toward user messages, which tend to be shorter but more factually dense (+1.4% on LongMemEvalS). Simple and effective.

### What Somnigraph does better

1. **Learned reranker with feedback loop**. Somnigraph's 26-feature LightGBM reranker trained on 1032 real-data queries with explicit utility feedback (per-query r=0.70 with GT). MemMachine uses Cohere rerank-v3-5-0, a generic cross-encoder with no domain adaptation or feedback signal. The paper lists "reinforcement learning integration" as future work (Section 10).

2. **Graph-conditioned retrieval via PPR**. Somnigraph's typed edges (detected during sleep) enable PPR expansion and betweenness centrality as a reranker feature. MemMachine uses Neo4j for relational mapping (sentence -> episode provenance) but does not perform graph traversal for retrieval expansion. Their graph is structural metadata, not a knowledge graph.

3. **Offline consolidation (sleep pipeline)**. Three-phase NREM/REM/orchestrator: relationship detection, contradiction classification, temporal evolution, summary generation. MemMachine has no offline processing beyond ingest-time indexing. Their ground-truth commitment precludes LLM-mediated transformation.

4. **Temporal decay and lifecycle management**. Biological decay, dormancy detection, importance-weighted retention. MemMachine's episodes are permanent and undifferentiated -- no mechanism to surface frequently-relevant memories or retire stale ones.

5. **Explicit feedback signal**. Per-query utility ratings with EWMA aggregation, UCB exploration bonus. MemMachine has no retrieval quality signal at all.

6. **Multi-hop retrieval via synthetic vocabulary bridges**. Somnigraph's Level 5b graph-augmented retrieval uses synthetic claim/segment nodes as Phase 1 RRF bridges, achieving R@10=95.4% and multi-hop R@10=88.8%. MemMachine's Retrieval Agent addresses multi-hop via iterative LLM calls (ChainOfQuery), which is more expensive and bounded by iteration limits.

---

## Worth Stealing (ranked)

### 1. User-query bias correction (trivial)
**What**: Prepend "user:" to search queries to bias retrieval toward user messages.
**Why**: +1.4% on LongMemEvalS. User messages are shorter but factually denser for recall tasks. In Somnigraph's context, this translates to biasing toward the human's statements rather than assistant responses during recall.
**How**: Add "user:" prefix to the query string before embedding. Could also be implemented as a BM25 field boost on producer metadata. Test on LoCoMo evaluation first.

### 2. Multi-query reranking for decomposed queries (medium)
**What**: When a query is decomposed (sub-queries, expansion terms), concatenate all queries for the final reranking pass so intermediate evidence scores well.
**Why**: MemMachine's ChainOfQuery achieves 95.31% recall on multi-hop bridge questions specifically because intermediate facts match sub-queries even when they don't match the original. Somnigraph's keyword expansion already generates multiple query forms but the reranker only sees the original query text for feature computation.
**How**: Modify reranker feature extraction to compute query-candidate similarity against all expansion variants, taking the max or mean. This is a feature engineering change, not an architectural one.

### 3. Query-type routing (large)
**What**: Classify incoming queries into structural types (dependency chain, fan-out, direct) and route to specialized retrieval strategies.
**Why**: Different query types have fundamentally different retrieval needs. Chain queries need iterative retrieval; fan-out queries need parallel sub-retrieval; direct queries need nothing beyond baseline. MemMachine's agent mode adds +2.0pp on HotpotQA and +5.2pp on WikiMultiHop over baseline.
**How**: This would require an LLM classification call per query, which conflicts with Somnigraph's single-user MCP design (latency matters). Could be implemented as an optional mode for complex queries. The ROI is questionable for single-user personal memory where most queries are direct lookups -- MemMachine's own LoCoMo results show only +0.5pp from agent mode.

### 4. Contextualization via episode clustering (medium)
**What**: After finding a nucleus match, expand to include 1 preceding + 2 following turns from the same session.
**Why**: Conversational turns are interdependent. A recommendation turn only makes sense with the preceding question and constraints. Somnigraph's PPR expansion serves a similar purpose but operates on detected relationships rather than temporal adjacency. Adjacency-based expansion is simpler and would complement graph-based expansion.
**How**: Add session_id and turn_order metadata to memories. After initial retrieval, expand each result to include adjacent turns from the same session. This could be a post-retrieval enrichment step before the reranker.

---

## Not Useful For Us

- **Sentence-level chunking of raw episodes**. Somnigraph's memories are already LLM-extracted -- re-indexing at sentence level would require storing raw conversations, which contradicts the current architecture. The ground-truth-preserving design is interesting but represents a fundamentally different bet (storage + retrieval complexity vs extraction error).

- **STM sliding window**. Somnigraph operates as an MCP server for Claude Code, which manages its own conversation context. There is no need for a separate short-term memory layer.

- **Profile memory as a separate subsystem**. Somnigraph already handles user preferences, facts, and patterns through its category system (episodic, procedural, semantic, entity). A dedicated profile extraction pipeline would add complexity without clear benefit for a single-user system.

- **Neo4j dependency**. MemMachine uses Neo4j for graph storage alongside PostgreSQL and pgvector -- three database systems. Somnigraph's SQLite + sqlite-vec architecture is deliberately lightweight. The graph functionality MemMachine gets from Neo4j (sentence-to-episode provenance) is far simpler than what Somnigraph's edge system provides.

- **APO prompt tuning**. The paper reports ~6% improvement from jointly tuning all agent prompts via Agent Lightning APO. This is a good practice but not a transferable technique -- it's benchmark-specific optimization.

---

## Connections

- **mem0-paper.md**: MemMachine positions itself explicitly against Mem0's per-message LLM extraction. Their token comparison (Table 8: 4.20M vs 19.21M) is the most direct published evidence for the cost of Mem0's approach. Mem0 achieves 0.6713 vs MemMachine's 0.8747 on LoCoMo (Table 11, both gpt-4o-mini). The gap is substantial but uses different eval-LLM configurations.

- **hindsight.md**: Hindsight also stores raw facts with provenance and uses cross-encoder reranking. The key difference is Hindsight's disposition-shaped reasoning and mental models, which MemMachine lacks entirely. Both use PostgreSQL + pgvector.

- **lightmem.md**: LightMem takes the opposite approach -- aggressive compression and topic-aware segmentation to minimize tokens. MemMachine stores raw sentences. On LoCoMo, LightMem reports 72.99% (GPT-4o-mini) vs MemMachine's 87.47% (GPT-4o-mini, Table 10). Direct comparison is complicated by different eval frameworks, but the gap suggests that ground-truth preservation + better retrieval beats compression.

- **a-mem.md**: A-Mem's self-organizing memory with Ebbinghaus-inspired activation scoring contrasts with MemMachine's no-decay, no-lifecycle approach. MemMachine's future work explicitly mentions "memory consolidation and forgetting" (Section 10).

- **locomo.md**: MemMachine uses the LoCoMo evaluation framework from Mem0's published code. Their 0.9169 (gpt-4.1-mini, agent mode) is the highest published score on this benchmark. Important context: they exclude 446 adversarial questions (standard practice per Section 7.1).

- **memobase.md**: Memobase is the next-best system on LoCoMo at 0.7578 (Table 11). MemMachine's +9.7pp advantage is the largest gap in the comparison table.

- **perma.md**: MemMachine does not evaluate on PERMA. Their profile memory (preferences, behavioral patterns) should perform reasonably on PERMA's preference-state maintenance tasks, but their lack of cross-domain synthesis mechanisms (no graph traversal, no relational reasoning at retrieval time) would likely hit the same multi-domain wall that other systems face.

- **hypermem.md / cognee.md**: MemMachine's Retrieval Agent tool tree (ToolSelectAgent -> ChainOfQuery/SplitQuery/MemMachine) resembles the query decomposition patterns in these systems but is simpler -- three fixed strategies rather than dynamic planning.

---

## Summary Assessment

**Relevance**: Medium-high. MemMachine is the current LoCoMo SOTA and provides the strongest published evidence that ground-truth preservation + intelligent retrieval can outperform LLM-mediated extraction. The Retrieval Agent's query routing and multi-query reranking are directly applicable ideas.

**Quality**: Strong paper. Systematic ablation on LongMemEvalS (6 dimensions, 12 configurations), comprehensive LoCoMo evaluation with per-category breakdowns, honest discussion of limitations (Section 9.7 lists 5 specific threats to validity). The GPT-5-mini > GPT-5 finding is genuinely surprising and well-documented.

**Threat to Somnigraph**: Moderate. Their 0.9169 on LoCoMo (agent mode) significantly exceeds Somnigraph's 85.1% (Opus judge). However, the comparison is not apples-to-apples: MemMachine uses gpt-4.1-mini as both eval-LLM and judge (via Mem0's evaluation framework), while Somnigraph uses GPT-4.1-mini as reader with Opus as judge. MemMachine's strength (raw episode retrieval) is orthogonal to Somnigraph's (learned reranking + graph expansion + feedback loop). The two systems make fundamentally different bets: MemMachine bets on retrieval quality over raw ground truth; Somnigraph bets on LLM-mediated consolidation creating better representations than raw data. Both approaches have merit, and the benchmarks may not distinguish between them clearly because LoCoMo's questions are predominantly factual recall where ground truth preservation naturally excels.

**Key takeaway**: The user-query bias correction is trivially adoptable. Multi-query reranking is the most impactful transferable idea. The ground-truth-preserving philosophy is interesting but represents a different architectural direction -- worth understanding, not worth copying.
