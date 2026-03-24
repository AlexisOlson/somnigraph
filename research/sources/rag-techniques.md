# RAG Techniques (NirDiamant) -- Analysis

*Generated 2026-03-22 by Opus agent reading github.com/NirDiamant/RAG_Techniques*

---

## Repository Overview

**Repo**: NirDiamant/RAG_Techniques. ~50K newsletter subscribers. 34 techniques implemented as Jupyter notebooks with LangChain/LlamaIndex, organized by category. Runnable scripts for several. MIT license.

**What it is**: A tutorial cookbook, not a research paper or production system. Each notebook demonstrates one retrieval-augmented generation technique end-to-end with working code. Value is in the breadth -- 34 techniques spanning foundational chunking through graph-based and self-correcting architectures, all in one place with consistent structure.

**What it isn't**: No unified system, no shared evaluation harness, no benchmark comparisons across techniques. Each notebook is standalone. Most have no quantitative evaluation -- they demonstrate the mechanism on toy data and leave measurement to the reader. The two evaluation notebooks (DeepEval, GroUSE) are separate tutorials, not applied across the other techniques.

**Scale**: 34 notebooks, 4 evaluation files, helper utilities. Categories: Foundational (5), Query Enhancement (3), Context Enrichment (6), Advanced Retrieval (7), Iterative/Adaptive (3), Advanced Architecture (5), Evaluation (3), Explainability (1), Special (1).

---

## Techniques Analyzed

Only techniques with relevance to Somnigraph's architecture or current bottlenecks are analyzed below. Foundational techniques (simple RAG, CSV RAG, chunk size selection) and multi-modal techniques are excluded -- they cover ground Somnigraph has already moved past.

### HyDE (Hypothetical Document Embedding)

**Mechanism**: At query time, an LLM generates a hypothetical answer document matching the expected chunk size. This synthetic document is embedded and used as the search vector instead of the query. Retrieval is document-to-document similarity rather than query-to-document.

**Key design**: Single expansion per query. Prompt constrains output to exact chunk size. No multi-variant generation. Per-query LLM call (expensive at scale).

**Evaluation**: Not quantified in notebook. Demonstrated on climate change query.

### HyPE (Hypothetical Prompt Embeddings)

**Mechanism**: At indexing time, an LLM generates multiple hypothetical questions for each chunk ("what questions would this chunk answer?"). Each question is embedded separately. The vector index stores multiple embedding vectors per chunk, one per hypothetical question. At query time, the user's query is embedded normally and matched against the hypothetical question embeddings -- question-to-question matching instead of query-to-document.

**Key design**: All expansion is offline (zero per-query overhead). Chunks are duplicated in the docstore with multiple index pointers. Deduplication at retrieval time (`list(set(context))`). FAISS L2 distance.

**Evaluation**: Claimed up to 42pp improvement in retrieval precision and 45pp in claim recall (per paper abstract). Notebook results are mixed per-question (some 5/5 relevance, some 1/5).

### Proposition Chunking

**Mechanism**: LLM decomposes documents into atomic, self-contained propositions (single facts). Each proposition is rated on four dimensions (accuracy, clarity, completeness, conciseness) with threshold filtering. Only passing propositions are embedded and stored.

**Key design**: Write-time transformation trades context for precision. Quality gate discards propositions below threshold. Best for factual queries; larger chunks better for complex queries requiring context.

### Document Augmentation via Question Generation

**Mechanism**: At indexing time, generate 40 hypothetical questions per document/fragment. Clean, deduplicate, embed alongside originals. The augmented questions serve as bridge tokens between user queries and document content.

**Key design**: Questions stored as separate documents with metadata type="AUGMENTED", linked to parent via metadata. Retrieval returns the parent document regardless of whether the match was original or augmented.

### Contextual Chunk Headers (CCH)

**Mechanism**: Prepend document title and section context to each chunk before embedding. Makes implicit references (pronouns, "this report," "the company") explicit in embedding space.

**Key design**: One LLM call per document (not per chunk) generates the title. Concatenated text is embedded, not just the chunk.

**Evaluation** (KITE benchmark, 50 questions): Average 27.9% improvement across 4 datasets. BVP Cloud dataset saw 142% improvement. Combined with RSE on FinanceBench: 83% vs 19% baseline.

### Relevant Segment Extraction (RSE)

**Mechanism**: Post-retrieval reconstruction of contiguous document segments. After reranking, compute per-chunk values using beta CDF transform + exponential rank decay, subtract an irrelevant-chunk penalty. Solve a constrained maximum-sum subarray problem to find the best contiguous segments, including bridging irrelevant chunks between relevant ones.

**Key design**: Requires no-overlap chunking (enables reconstruction). Beta CDF with a=0.4, b=0.4 transforms relevance scores. Constraints: max segment 20 chunks, max total 30 chunks, min segment value 0.7. Runtime ~5-10ms.

**Evaluation** (KITE benchmark): Average 42.6% improvement. AI Papers: +75.6%. Combined with CCH achieves 83% on FinanceBench.

### Fusion Retrieval

**Mechanism**: Combine FAISS vector similarity with BM25 keyword scoring via normalized linear combination. Vector scores normalized as `1 - (v-min)/(max-min)`, BM25 normalized as `(b-min)/(max-min)`. Combined: `alpha * vec + (1-alpha) * bm25`, default alpha=0.5.

**Key design**: Linear score fusion (not RRF). Simple tokenization (whitespace split). No field weighting on BM25.

### Retrieval with Feedback Loop

**Mechanism**: Store user feedback (relevance 1-5, quality 1-5) in JSONL. On each new query, use LLM to determine which historical feedback is relevant to the current query-document pair. Adjust document scores by `score *= avg_feedback_relevance / 3` (3 = neutral midpoint). Periodically fine-tune the vector index by adding high-quality query-response pairs (relevance >= 4, quality >= 4) as new documents.

**Key design**: Feedback is a multiplicative reranker, not a feature. Periodic index augmentation with synthetic good-answer documents.

### Adaptive Retrieval

**Mechanism**: Classify queries into four types (Factual, Analytical, Opinion, Contextual), route to type-specific retrieval strategies. Factual: enhance query + LLM-rank by relevance. Analytical: decompose into sub-queries, retrieve per sub-query, select diverse results. Opinion: identify viewpoints, retrieve per viewpoint. Contextual: incorporate user context, rank considering context.

**Key design**: All routing and ranking via LLM prompting (not learned). Query classification determines expansion strategy and diversity weighting.

### Self-RAG

**Mechanism**: Six-step sequential pipeline with gating. (1) LLM decides if retrieval is necessary. (2) Retrieve. (3) LLM filters each document as Relevant/Irrelevant. (4) Generate response per relevant context. (5) Rate support level (Fully/Partially/No). (6) Rate utility (1-5), select best. Early exit if no retrieval needed.

**Key design**: Hard retrieval gate (yes/no) with soft relevance filtering. Multi-response generation with hierarchical selection (support > utility). No learning.

### Corrective RAG (CRAG)

**Mechanism**: Score retrieved documents on [0,1] relevance. Route by threshold: >0.7 use as-is, <0.3 web search fallback (LLM rewrites query), 0.3-0.7 hybrid of document + web search. Knowledge refinement extracts bullet points for cleaner assembly.

**Key design**: Three-bin threshold routing with explicit fallback. Query rewriting for failed retrievals.

### Graph RAG (LangChain)

**Mechanism**: Represent documents as knowledge graph. Nodes are text chunks. Edges from semantic similarity + shared extracted concepts (NLP + LLM extraction, lemmatized). Query processing: embed query, retrieve initial documents, then Dijkstra-like priority traversal through graph. At each node, add content to context, check if answer is complete (LLM judge), update neighbor priorities by edge weight.

**Key design**: Explicit completeness checking during traversal -- LLM evaluates "do I have enough to answer?" at each step. Not just scoring; the traversal terminates when the answer is sufficient.

### RAPTOR

**Mechanism**: Build hierarchical tree of document abstractions. Level 0 = original documents. Each subsequent level: embed, cluster (GMM), summarize clusters via LLM. Retrieval starts at highest level, matches query to coarse summaries, recursively retrieves children. Contextual compression on final results.

**Key design**: Multi-level abstraction with tree traversal. Coarse-to-fine retrieval. GMM clustering (not k-means) allows soft assignment.

### Dartboard Retrieval

**Mechanism**: Greedy selection balancing relevance and diversity via log-normal probability scoring. Oversample candidates (k * oversampling_factor), compute query-document and pairwise-document distances, then iteratively select documents that maximize `DIVERSITY_WEIGHT * max_dist_from_selected + RELEVANCE_WEIGHT * query_prob`, normalized via logsumexp. Already-selected documents are masked.

**Key design**: Log-normal kernel converts distances to probabilities. `max_dist_from_selected` ensures each new selection is dissimilar from the current set. Default equal weights.

### Explainable Retrieval

**Mechanism**: After standard retrieval, generate per-document explanations of relevance via LLM. Returns (document, explanation) pairs.

### Iterative Retrieval

**Mechanism**: Multi-round retrieval where each round's results inform the next query. Initial retrieval, then LLM generates follow-up queries based on gaps in the current context, retrieves again, repeats until sufficient.

---

## Key Claims & Evidence

### 1. Write-time expansion outperforms query-time expansion for retrieval quality

HyPE (offline question generation) achieves comparable or better results than HyDE (runtime document generation) with zero per-query cost. Document Augmentation follows the same pattern: generate alternative representations at indexing time. The consistent lesson: amortize LLM expansion cost to write time when possible.

Evidence: HyPE claims 42pp precision improvement. CCH achieves 27.9% average improvement on KITE. Neither has the per-query latency penalty of HyDE.

### 2. Segment coherence matters more than individual-document ranking

RSE's 42.6% improvement (KITE) comes not from better ranking but from reconstructing contiguous segments that restore context destroyed by chunking. Top-k retrieval fragments documents; RSE reassembles them. The improvement is largest on complex documents (AI Papers: +75.6%) where relevant information clusters spatially.

### 3. Feedback is a multiplicative reranker, not a learned feature

The feedback loop notebook multiplies document scores by normalized feedback rather than training on feedback. This matches Somnigraph's empirical finding that feedback has near-zero marginal contribution as a reranker feature while still tracking relevance at r=0.70 per-query. The signal exists but operates through the store (shaping what's available), not through the ranker (shaping what's elevated).

### 4. Query decomposition is the unimplemented solution for multi-hop

Adaptive Retrieval's "analytical" pathway decomposes complex queries into sub-queries, retrieves per sub-query, then selects diverse results. No notebook tests this on multi-hop benchmarks, but the mechanism directly addresses the failure mode where a single query embedding can't simultaneously match all hops. Iterative retrieval extends this further: each round discovers gaps and generates targeted follow-up queries.

### 5. Quality gating at retrieval time has real value

Both Self-RAG and CRAG implement explicit quality gates. Self-RAG gates on whether retrieval is even necessary. CRAG gates on whether retrieved results are sufficient. Neither is learned -- both use LLM-as-judge. The philosophical point: not every query benefits from retrieval, and not every retrieval is good enough to use.

---

## Relevance to Somnigraph

### The multi-hop bottleneck (directly relevant)

Somnigraph's weakest category is multi-hop (~18pp below single-hop). Six expansion methods were neutral. Evidence exists in the candidate pool at rank 100+ but the reranker can't elevate it. Three techniques from this repo speak to this:

**Adaptive query decomposition.** Multi-hop queries like "What did A lead to through B?" require matching both A-B and B-C relationships, but a single embedding can only approximate the centroid of these semantics. Decomposing into sub-queries ("What connects A to B?", "What connects B to C?") and retrieving per sub-query would surface candidates from different regions of the embedding space. This is fundamentally different from candidate expansion (adding more results from the same query) -- it's multiple queries targeting different semantic neighborhoods.

**Iterative retrieval with gap detection.** Rather than one-shot decomposition, iteratively retrieve, identify what's missing from the answer, and generate targeted follow-up queries. This is more expensive but adapts to the actual gaps rather than guessing what sub-queries to generate upfront.

**Graph traversal with completeness checking.** The LangChain GraphRAG notebook's Dijkstra-like traversal includes an LLM completeness check at each node. Somnigraph's PPR traversal has no stopping criterion based on answer sufficiency -- it runs a fixed number of iterations. Adding a lightweight sufficiency check could prevent premature termination on multi-hop chains.

### Prospective indexing / HyPE (directly relevant to roadmap Tier 1 #10)

HyPE is the same core idea as Somnigraph's planned prospective indexing experiment (and Kumiho's approach that eliminated the >6-month accuracy cliff). The implementation details differ: HyPE stores hypothetical questions as separate embedding vectors in the same index (multi-vector per document), while Kumiho and Somnigraph's plan append questions to the enriched text before embedding (single vector per memory, richer text).

The multi-vector approach has a specific advantage for multi-hop: different hypothetical questions per memory can match different hops independently. With single-vector enrichment, the embedding averages across all hypothetical questions, diluting the signal for any individual hop. This is worth testing as an alternative to the planned append-to-enriched-text approach.

### RSE-adjacent segment reconstruction (novel for memory systems)

RSE solves a document-specific problem (chunking fragments context), but the underlying mechanism -- reconstructing coherent segments from scattered relevant pieces -- translates to memory retrieval. Somnigraph's memories aren't document chunks with linear order, but they do have graph structure (theme edges, temporal clustering, Hebbian co-retrieval reinforcement).

A variant: after reranking, identify clusters of highly-connected memories (via edge structure) and present them as coherent groups rather than a flat ranked list. This doesn't improve retrieval rank directly but could improve downstream QA accuracy by giving the answer generator better-organized context. RSE's constrained optimization (max segment length, minimum value threshold) would translate to max cluster size and minimum edge density thresholds.

### What's already covered

**Fusion retrieval**: Somnigraph uses RRF (rank-based), which is more robust than the linear score normalization shown here. Linear normalization is sensitive to score distributions; RRF is not.

**Feedback loop**: Somnigraph's feedback loop is significantly more sophisticated than this notebook's multiplicative adjustment. Empirical Bayes prior, EWMA, UCB exploration -- the notebook's approach is a starting point, not an improvement.

**Reranking**: Somnigraph's 26-feature LightGBM reranker is a learned model. This repo's reranking notebook uses a Cohere cross-encoder as a black box. Different approach entirely.

**Graph RAG**: Somnigraph already has PPR graph traversal, Hebbian edge reinforcement, and edge metadata (contradiction flags, linking context, weights). The LangChain notebook's graph is simpler (concept extraction + similarity edges). Somnigraph's graph is richer.

---

## Worth Stealing (Ranked)

### 1. Multi-vector prospective indexing (HIGH value, MEDIUM effort)

Store 2-4 hypothetical recall queries per memory as separate embedding vectors in sqlite-vec, rather than appending them to the enriched text. At recall time, search both the original embedding and the hypothetical query embeddings, then deduplicate. This preserves the full semantic signal of each hypothetical query rather than averaging them into one embedding.

**Why this over append-to-text**: Somnigraph's current enriched embedding already concatenates content + category + themes + summary. Adding more text to an already-rich embedding may hit diminishing returns (the embedding model's capacity to encode distinct signals in 1536 dimensions has limits). Separate vectors keep the signals separable.

**Trade-off**: Increases vector index size by 3-5x. sqlite-vec KNN cost is O(n), so this directly multiplies retrieval time. At ~730 memories * 4 vectors = ~3000 vectors, still fast. At 5000 memories, might need evaluation.

**Interaction with roadmap**: This is an implementation variant of Tier 1 #10 (prospective indexing). Worth testing both approaches (multi-vector vs. append) since the infrastructure for comparison already exists.

### 2. Query decomposition for multi-hop (HIGH value, HIGH effort)

Before recall, classify the query. If it requires multi-hop reasoning (multiple entities, causal chains, temporal sequences), decompose into sub-queries and retrieve independently for each. Merge results with deduplication and present the combined candidate set to the reranker.

**Why this could work where expansion failed**: The six neutral expansion methods all added more candidates from the same semantic neighborhood. Query decomposition retrieves from different neighborhoods entirely. If a multi-hop query about "A's effect on C through B" retrieves A-related and C-related memories but misses B, no amount of expansion around A or C will surface B. A sub-query targeting B directly will.

**Cost**: Per-query LLM call for classification + decomposition. Could be gated (only for queries the classifier tags as multi-hop) to limit cost.

**Interaction with roadmap**: Not currently on the roadmap. Would be a new Tier 1 experiment. Directly addresses the multi-hop bottleneck identified in expansion experiments.

### 3. Completeness-gated graph traversal (MEDIUM value, MEDIUM effort)

During PPR graph expansion, add a lightweight check: given the current candidate set, is the query answerable? If not, expand further (increase PPR iterations, lower the seed threshold). This converts PPR from a fixed-depth traversal to an adaptive one that terminates when sufficient context is gathered.

**Implementation**: After initial PPR + reranking, send the top-k candidates + query to a fast LLM (Haiku) with the prompt "Can this question be fully answered from these memories? Yes/No/Partially." If Partially or No, re-run PPR with relaxed parameters.

**Why this matters for multi-hop**: Multi-hop answers require context from multiple graph regions. Fixed-depth PPR may terminate before reaching all necessary regions. Completeness checking detects this and triggers deeper exploration.

**Cost**: One additional Haiku call per query when the check triggers. Could cache completeness judgments for similar queries.

### 4. Post-reranker cluster presentation (LOW value, LOW effort)

After reranking, group the top-k results by graph connectivity (shared edges, theme overlap) and present them as labeled clusters rather than a flat list. This doesn't change retrieval quality but may improve downstream QA accuracy by giving the answer generator organized context.

**Implementation**: Simple: connected-component analysis on the subgraph induced by top-k results. Label each component by dominant theme.

### 5. CCH-style embedding verification (LOW value, LOW effort)

Verify that Somnigraph's enriched embeddings (content + category + themes + summary) are actually making implicit references explicit. Test: for memories with pronouns or implicit references in content ("he mentioned," "the project"), compare retrieval quality with and without the enrichment. If enrichment doesn't help on these cases, the concatenation format may need revision (e.g., prepending a one-line context sentence before content, as CCH does with document titles).

---

## Not Useful For Us

### 1. Linear score fusion

The fusion retrieval notebook uses `alpha * vec + (1-alpha) * bm25` with min-max normalization. RRF (which Somnigraph uses) is strictly more robust -- it's rank-based, not score-based, so it's invariant to score distribution differences between channels. No reason to switch.

### 2. Multiplicative feedback reranking

The feedback loop notebook's `score *= feedback / 3` is a crude version of what Somnigraph already does with empirical Bayes priors and EWMA. The periodic index augmentation (adding good answers as documents) is interesting conceptually but wrong for a memory system -- it would pollute the memory store with synthetic documents.

### 3. Self-RAG's retrieval gate

The "should I retrieve?" binary decision doesn't apply. Somnigraph's recall is invoked by the agent explicitly; there's no ambient retrieval to gate. The agent already makes this decision by choosing whether to call recall().

### 4. CRAG's web search fallback

Memory systems don't have a web search fallback. The threshold-based routing concept (good/bad/ambiguous) could theoretically apply to recall confidence, but the `limit` parameter already handles this more naturally -- the agent decides how much context it needs.

### 5. RAPTOR's hierarchical abstraction

Somnigraph already has a three-layer hierarchy (detail/summary/gestalt) via CLS-inspired sleep consolidation. RAPTOR's GMM-based clustering adds nothing -- Somnigraph's clustering is theme-based with LLM judgment during REM, which is more semantically grounded than embedding-space GMM.

### 6. Dartboard's diversity scoring

Diversity is a secondary concern for memory retrieval. The reranker already implicitly captures diversity through features like burstiness and diversity_score. Explicit diversity penalties would risk excluding memories that are similar but independently important (e.g., two related decisions about the same project).

---

## Connections

### To Kumiho (arXiv:2603.17244)

HyPE's offline question generation is the same core mechanism as Kumiho's prospective indexing. Kumiho appends questions to summaries before embedding; HyPE stores them as separate vectors. Kumiho reports eliminating the >6-month accuracy cliff on LoCoMo-Plus (37.5% to 84.4%). HyPE claims 42pp precision improvement. The convergent finding: write-time query anticipation is high-value.

### To Dynamic Cheatsheet (arXiv:2504.07952)

DC's per-query curation is the continuous analog of what Adaptive Retrieval does with query classification. Both recognize that different queries need different retrieval strategies. DC embeds the strategy choice in memory curation; Adaptive Retrieval makes it an explicit routing decision. Somnigraph currently has neither -- all queries go through the same pipeline regardless of type.

### To Ori-Mnemos (v0.5.0)

Ori-Mnemos' recursive sub-question decomposition during retrieval is the learned version of Adaptive Retrieval's analytical pathway. Ori-Mnemos trains Q-values on sub-question effectiveness; the RAG Techniques notebook uses hard-coded LLM prompting. Somnigraph has neither, and the multi-hop bottleneck suggests it should.

### To the expansion experiment results

Six expansion methods were neutral because they all add candidates from the same semantic neighborhood. HyPE/prospective indexing and query decomposition both operate differently: they create new entry points into the embedding space rather than expanding around existing entry points. This is the distinction between "more candidates from the same search" and "more searches."

### To the reranker

The reranker can't elevate evidence at rank 100+ because at that rank, the features available (BM25 rank, vector distance, metadata) provide no signal to distinguish relevant from irrelevant. Query decomposition reduces the problem by ensuring each sub-query's relevant evidence appears at rank 1-20 rather than 100+. Multi-vector indexing does the same by ensuring different hypothetical questions match different aspects of the query directly.

---

## Summary of Key Takeaways

1. **Write-time expansion (HyPE, CCH) consistently outperforms query-time expansion (HyDE)** in both quality and cost. Somnigraph's planned prospective indexing is aligned with this finding. The multi-vector variant is worth testing alongside the append-to-text approach.

2. **Query decomposition is the strongest untested mechanism for multi-hop.** The expansion experiments showed that more candidates from one query don't help. Multiple targeted queries -- each matching one hop -- is the structural solution.

3. **Segment coherence (RSE) is a distinct improvement axis from ranking.** Even with perfect ranking, presenting fragmented context hurts downstream QA. Graph-based clustering of reranked results is the memory-system equivalent.

4. **These notebooks confirm what Somnigraph already does well.** RRF fusion, learned reranking, graph traversal, feedback loops -- all are represented here in simpler forms. The gap is in query-adaptive retrieval (decomposition, gating, iterative follow-up), not in the core pipeline.

5. **No quantitative evidence across techniques.** The repo's biggest limitation: no shared evaluation, no benchmark comparisons, no ablation studies. Each technique is demonstrated but not measured against the others. The evaluation notebooks are separate tutorials, not applied systematically.
