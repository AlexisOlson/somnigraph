# On the Structural Memory of LLM Agents

*Zeng, Fang, Liu, Meng. University of Glasgow / University of Aberdeen. arXiv:2412.15266, December 2024.*

Code: https://github.com/zengrh3/StructuralMemory

---

## 1. Paper Summary

The paper presents the first systematic comparison of four memory structures — chunks, knowledge triples, atomic facts, and summaries — combined with three retrieval methods — single-step, reranking, and iterative — for LLM-based agents. The study spans six datasets across four tasks: multi-hop QA (HotPotQA, 2WikiMultihopQA, MuSiQue), single-hop QA (NarrativeQA), dialogue understanding (LoCoMo), and reading comprehension (QuALITY). They also introduce "mixed memory," the union of all four structures. The goal is to answer a question no prior work had addressed head-on: which memory structure works best for which task, and how do structure and retrieval method interact?

## 2. Key Concepts and Techniques

### Memory structures

- **Chunks**: Fixed-length text segments (max 1K tokens) from the original document. Preserves context and narrative flow but is coarse-grained.
- **Knowledge triples**: `<head; relation; tail>` extracted by LLM. Captures entity relationships compactly but loses narrative context.
- **Atomic facts**: Minimal self-contained sentences extracted by LLM, with pronoun resolution. Finest granularity — each fact stands alone.
- **Summaries**: LLM-generated condensation of source documents. Most compressed, captures global themes but loses detail.
- **Mixed**: Union of all four. The memory pool contains chunks, triples, atomic facts, and summaries simultaneously.

### Retrieval methods

- **Single-step**: Standard top-K retrieval via text-embedding-3-small. One pass, no refinement.
- **Reranking**: Retrieve top-K, then LLM reranks to select top-R. Adds one LLM call.
- **Iterative**: Retrieve top-T, LLM refines the query, repeat N times, then final top-K retrieval. Multiple LLM calls per query.

### Answer generation

Two variants tested: **Memory-Only** (memories are the context) and **Memory-Doc** (use retrieved memories to locate original documents, then use those as context). Memory-Doc provides richer context but dilutes precision.

### Experimental setup

GPT-4o-mini-128k, text-embedding-3-small, chunk size 1K tokens. Evaluation: EM and F1 for QA tasks, accuracy for QuALITY. Default K=100, R=10, T=50, N=3.

## 3. Key Findings

### Finding 1: Mixed memory delivers balanced performance

Mixed memory consistently achieves the best or near-best results across tasks and retrieval methods. Under iterative retrieval: 82.11% F1 on HotPotQA, 68.15% on 2WikiMultihopQA. The diversity of representation types means at least one structure will match the query's information need.

Mixed memory also shows the strongest noise resilience — when noise documents are added, mixed degrades more slowly than any individual structure.

### Finding 2: Different structures suit different tasks

- **Multi-hop QA**: Knowledge triples and atomic facts excel. Triples capture entity relationships directly; atomic facts preserve individual reasoning steps. Triples achieve 62.06% F1 on 2WikiMultihopQA (iterative), atomic facts achieve 81.29% on HotPotQA (iterative).
- **Long-context / dialogue**: Chunks and summaries win. Chunks preserve narrative flow needed for NarrativeQA (31.63% F1 under reranking). Summaries capture the abstraction needed for LoCoMo (12.04 EM, 44.83 F1 under reranking).
- **Reading comprehension**: Mixed is best (79.50% on QuALITY under iterative). Chunks are strong individually (77.00%).

### Finding 3: Iterative retrieval consistently outperforms

Across nearly all structure/task combinations, iterative retrieval beats single-step and reranking. The LLM-refined query finds better candidates on subsequent passes. Optimal iteration count is 2-3; beyond that, marginal returns diminish and noise can increase.

### Finding 4: Hyperparameter sensitivity

- **K (retrieval depth)**: Chunks are robust across K values; triples, atomic facts, and summaries peak around K=50-100 and degrade at K=200 due to noise.
- **R (reranking depth)**: Smaller R (10-25) outperforms larger R. Reranking is most effective when focused on a small, high-quality candidate set.
- **T (per-iteration retrieval)**: Peaks around T=50; T=75 introduces noise.
- **N (iterations)**: 2-3 is optimal. Gains flatten quickly.

### Finding 5: Memory-Doc vs Memory-Only

Extensive-context tasks (NarrativeQA, QuALITY) benefit from Memory-Doc, which traces memories back to source documents. Precision tasks (HotPotQA, LoCoMo) prefer Memory-Only — the memories themselves are sufficient and adding source documents dilutes focus.

## 4. Comparison to Somnigraph

### Where Somnigraph's memories fall in the taxonomy

Somnigraph's memories are closest to **atomic facts** — self-contained text entries with metadata (category, themes, priority, summary, embeddings). Each memory is a standalone unit of information, not a fixed-length chunk of raw text, not a triple, not a summary. The `remember()` call creates entries that are essentially LLM-authored atomic facts with rich metadata.

However, Somnigraph also has elements of **summaries** (the summary field on each memory, used for embedding) and **knowledge triples** (typed graph edges: support, contradict, evolve, derive — though these are relationships between memories, not the memories themselves). The system does not store raw chunks.

In the paper's framing, Somnigraph is closest to a "mixed" system where the primary store is atomic facts, augmented by summary representations (via the summary field) and graph relationships (via typed edges). It is not, however, a true mixed system — it doesn't store the same information in multiple parallel representations.

### Retrieval comparison

- **Single-step**: Somnigraph's base retrieval (BM25 + vector + theme overlap with RRF fusion) is substantially more sophisticated than the paper's single-step (vector-only top-K). The paper's finding that single-step is weakest doesn't directly apply — Somnigraph's "single-step" is already a multi-signal fusion.
- **Reranking**: Somnigraph's LightGBM reranker (26 features, trained on 1032 queries) is a learned reranker, not an LLM reranker. The paper's reranking uses an LLM to score relevance — more expensive, more flexible, but not trainable on user-specific feedback. Somnigraph's approach is closer in spirit but with measurable quality and zero inference-time LLM cost.
- **Iterative**: Somnigraph has no iterative retrieval. The LoCoMo benchmark pipeline does use query expansion (BM25-damped IDF keywords, synthetic vocabulary bridges via graph), which serves a similar purpose — enriching the query signal — but it's not a multi-turn LLM refinement loop.

### LoCoMo results

The paper reports LoCoMo results with mixed memory + iterative retrieval: EM=12.50, F1=45.25. Somnigraph's Run 1 (L3 retrieval, GPT-4.1-mini reader, 1540 questions excluding adversarial) achieves F1=57.43 overall (single-hop 43.92, temporal 60.45, multi-hop 28.76, open-domain 64.07). This is a comparable metric — both use token-overlap F1 against LoCoMo gold answers.

Somnigraph's F1 advantage (+12.18pp overall) comes from hybrid retrieval + learned reranker, not memory structure diversity. The paper uses vector-only single/reranking/iterative retrieval with GPT-4o-mini; Somnigraph uses BM25 + vector + theme + graph RRF fusion with a 26-feature LightGBM reranker and GPT-4.1-mini reader.

Note: F1 and LLM-as-judge accuracy diverge substantially. Somnigraph's Run 2 (L5b retrieval) scored *higher* on judge accuracy (85.6% vs 85.1%) but *lower* on F1 (24.14 vs 57.43) — the reader generated semantically correct but lexically different answers. F1 alone is unreliable for evaluating answer quality.

## 5. Worth Adopting?

### Parallel multi-structure storage: No

The paper's strongest finding — that mixed memory outperforms any single structure — suggests storing the same information in multiple representations. For Somnigraph, this would mean generating chunks, triples, summaries, and atomic facts for every memory and retrieving across all of them. The cost (4x storage, 4x embedding, increased noise in the candidate pool) is high, and the benefit is uncertain given that Somnigraph already has multi-signal retrieval (BM25 + vector + theme + graph) which serves a similar diversity function without redundant storage.

### Knowledge triples as a retrieval signal: Already partially done

The paper shows triples excel at multi-hop QA via entity relationship capture. Somnigraph's typed graph edges (support, contradict, evolve, derive) and Hebbian co-retrieval edges already serve this role. The graph-augmented retrieval in L5b (synthetic vocabulary bridges, `graph_synthetic_score` feature) is a more sophisticated version of what triples provide. No change needed.

### Iterative query refinement: Worth considering for hard queries

The paper's strongest retrieval finding is that iterative refinement (2-3 rounds) consistently helps. Somnigraph's BM25-damped IDF expansion and synthetic vocabulary bridges are static expansions — they don't use the initial retrieval results to refine the query. An iterative refinement step could help for the remaining multi-hop misses, but at the cost of LLM calls per retrieval. Given that Somnigraph's R@10 is already 95.4% on LoCoMo, the ceiling for improvement is small, and the latency/cost tradeoff is unfavorable for a production system that currently achieves sub-200ms retrieval with zero LLM cost.

### Summary field as a parallel representation: Already done

Somnigraph stores both the full content and a summary for each memory, and embeds the summary. This is effectively a two-representation system (atomic fact + summary), which the paper suggests is beneficial. No change needed.

## 6. Worth Watching

- **Structure-retrieval interaction effects**: The paper's core contribution is showing that structure and retrieval method interact non-trivially. Future work might identify which specific structure-retrieval combinations matter and why, which could inform when to use graph-augmented retrieval vs. direct embedding search.
- **Noise resilience of mixed memory**: The paper shows mixed memory degrades gracefully under noise. As Somnigraph's memory store grows, noise (irrelevant old memories, partially decayed entries) becomes a bigger issue. The finding that structural diversity helps with noise is worth monitoring.
- **Iterative retrieval with learned rerankers**: The paper only tests LLM-based reranking. A hybrid — iterative retrieval with a learned reranker in the loop — is unexplored and could be interesting. Would the reranker's signal improve with each iteration?

## 7. Key Claims

| Claim | Evidence |
|-------|----------|
| Mixed memory outperforms individual structures | **Supported**: Table 1 shows mixed achieves best or second-best across 18 task/retrieval combinations. The pattern is consistent. |
| Iterative retrieval is consistently best | **Supported**: Iterative wins in 14 of 20 structure/task cells in Table 1. The few exceptions are marginal. |
| Different structures suit different tasks | **Supported**: Triples and atomic facts dominate multi-hop; chunks and summaries dominate long-context. Clear and reproducible pattern in Table 1. |
| Mixed memory is most noise-resilient | **Supported**: Figure 8 shows mixed degrading most slowly across 4 datasets as noise documents increase. |
| Reranking is most effective with small R | **Supported**: Figure 5 shows consistent performance peak at R=10-25, declining at R=50-75. |
| 2-3 iteration turns are optimal | **Supported**: Figure 7 shows clear diminishing returns after N=3 across all structures. |

## 8. Relevance to Somnigraph

**Medium**. The paper provides useful empirical confirmation that Somnigraph's design choices are reasonable: atomic-fact-like storage with summary representations, multi-signal retrieval, and learned reranking. The main actionable finding — that iterative query refinement consistently helps — is already partially addressed by Somnigraph's expansion pipeline, and the remaining gap (true iterative refinement) involves unfavorable latency/cost tradeoffs for production use.

The paper's most interesting contribution to Somnigraph's context is the demonstration that no single memory structure dominates all tasks. This validates the intuition behind Somnigraph's graph-augmented approach: different retrieval signals (BM25 for lexical match, vectors for semantic similarity, graph edges for relational reasoning) serve similar diversity purposes as multiple memory structures, without the storage overhead of maintaining parallel representations.

On LoCoMo with comparable F1 metrics, Somnigraph outperforms the paper's best configuration by +12.18pp (57.43 vs 45.25), suggesting retrieval sophistication (hybrid fusion + learned reranking + graph augmentation) matters more than memory structure diversity when the underlying structures are already reasonably expressive.
