# SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval -- Analysis

*Generated 2026-03-22 by Opus 4.6 agent reading arXiv:2109.10086v1*

---

## Paper Overview

**Paper**: Thibault Formal, Benjamin Piwowarski, Carlos Lassance, Stephane Clinchant (Naver Labs Europe, Sorbonne Universite / LIP6). "SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval." arXiv:2109.10086v1, Sep 2021. 6 pages. Code: https://github.com/naver/splade

**Problem addressed**: Standard bag-of-words retrieval (BM25) suffers from the vocabulary mismatch problem -- relevant documents may not contain the exact terms a query uses. Dense retrieval (BERT bi-encoders) solves this but loses exact-match capability. The original SPLADE model showed that learned sparse representations could combine expansion with exact matching, but its effectiveness and efficiency had room for improvement. BM25 achieves only 0.184 MRR@10 on MS MARCO dev, while dense methods reach 0.330-0.370.

**Core claim**: By switching from sum pooling to max pooling over the MLM prediction head, introducing a document-only expansion variant, and applying knowledge distillation, SPLADE v2 achieves state-of-the-art results on the BEIR zero-shot benchmark (0.500 avg NDCG@10, best on 11 of 14 datasets) while maintaining the efficiency properties of inverted indexes. The key insight is that a BERT MLM head can learn *which vocabulary terms should be activated* for a document or query, producing sparse vectors in the full 30,522-dimensional WordPiece vocabulary that can be scored via dot product and stored in standard inverted indexes.

**Scale**: Evaluated on MS MARCO passage ranking (8.8M passages, 6,980 dev queries, 43 TREC DL 2019 queries), plus 14 BEIR datasets for zero-shot transfer. Compared against BM25, DeepCT, doc2query-T5, SparTerm, COIL-tok, DeepImpact (sparse); ANCE, TCT-ColBERT, TAS-B, RocketQA (dense); and ColBERT (late interaction). Three model variants tested: SPLADE-max, SPLADE-doc, DistilSPLADE-max.

---

## Architecture / Method

### How SPLADE Produces Sparse Representations

SPLADE repurposes BERT's Masked Language Model (MLM) prediction layer as a term importance estimator. For each token in the input sequence, the MLM head produces a distribution over the full vocabulary (|V| = 30,522 WordPiece tokens). Instead of predicting which token was masked, SPLADE interprets these logits as importance weights: "given this input token in context, how important is each vocabulary term?"

The per-token importance for vocabulary term j given input token i:

```
w_ij = transform(h_i)^T E_j + b_j
```

where `h_i` is BERT's contextual embedding for input token i, `E_j` is BERT's input embedding for vocabulary term j, and `transform(.)` is a linear + GeLU + LayerNorm layer (the standard MLM head). This reuses pre-trained MLM weights -- the model already knows which terms are likely given a context.

### Aggregation: Sum vs. Max Pooling

The original SPLADE aggregated per-token importances via sum after a log-saturation:

```
w_j = SUM_i log(1 + ReLU(w_ij))     [SPLADE, Eq. 2]
```

SPLADE v2's key change replaces sum with max:

```
w_j = MAX_i log(1 + ReLU(w_ij))     [SPLADE-max, Eq. 6]
```

**Why max is better**: Sum accumulates evidence from multiple input tokens. If three tokens each weakly predict "database," the sum may be large even though no single token strongly indicates it. Max requires at least one token to strongly predict the expansion term. This produces sharper, more confident expansions. The improvement is +1.8 MRR@10 on MS MARCO dev and +1.9 NDCG@10 on TREC DL 2019.

### Scoring and Retrieval

The query and document each get a sparse vector in R^{30522}. Retrieval is a dot product:

```
s(q, d) = SUM_j w_q_j * w_d_j
```

Because both vectors are sparse (most entries are zero), this is equivalent to an inverted index lookup: for each non-zero query term, look up the posting list and accumulate document weights. No approximate nearest neighbor search needed -- standard IR infrastructure works.

### Document-Only Expansion (SPLADE-doc)

A variant that only expands documents, not queries. The query representation is a simple binary indicator of which terms appear:

```
s(q, d) = SUM_{j in q} w_d_j
```

This is strictly more efficient: document representations are precomputed and indexed offline, and query processing requires no neural inference (just term lookup). The cost is a modest effectiveness drop (-1.8 MRR@10 vs SPLADE-max) while still beating doc2query-T5.

### Distillation (DistilSPLADE-max)

A two-stage distillation process:

1. Train a SPLADE first-stage retriever and a cross-encoder reranker using existing distillation triplets
2. Use the trained SPLADE to mine hard negatives, score them with the cross-encoder, then train a new SPLADE from scratch on these harder examples using Margin-MSE loss

This pushes DistilSPLADE-max to 0.368 MRR@10 (competitive with RocketQA's 0.370) and 0.729 NDCG@10 on TREC DL 2019 (best in class).

### Sparsity Regularization (FLOPS)

The model would naturally produce dense representations (every vocabulary term gets some weight). FLOPS regularization penalizes the expected number of floating-point operations during retrieval:

```
l_FLOPS = SUM_j (mean_batch(w_j))^2
```

This squares the mean activation per term, penalizing terms that are active across many documents (long posting lists). Separate regularization weights for queries (lambda_q) and documents (lambda_d) allow pushing queries to be sparser (faster retrieval) while documents can be denser (better recall).

### Effectiveness-Efficiency Tradeoff

The regularization strength lambda directly controls the sparsity-effectiveness tradeoff:

| Configuration | MRR@10 | Relative FLOPS |
|---------------|--------|----------------|
| DistilSPLADE-max (high FLOPS) | 0.368 | ~4.0 |
| DistilSPLADE-max (low FLOPS) | 0.350 | ~0.3 |
| SPLADE-doc (19 non-zero weights/doc) | 0.296 | Very low |
| SPLADE-doc (200 non-zero weights/doc) | 0.322 | Low |
| BM25 | 0.184 | Baseline |

---

## Key Claims & Evidence

### MS MARCO Dev and TREC DL 2019

| Model | Type | MRR@10 | R@1000 | NDCG@10 (TREC) | R@1000 (TREC) |
|-------|------|--------|--------|-----------------|----------------|
| BM25 | Sparse | 0.184 | 0.853 | 0.506 | 0.745 |
| DeepCT | Sparse | 0.243 | 0.913 | 0.551 | 0.756 |
| doc2query-T5 | Sparse | 0.277 | 0.947 | 0.642 | 0.827 |
| SparTerm | Sparse | 0.279 | 0.925 | -- | -- |
| DeepImpact | Sparse | 0.326 | 0.948 | 0.695 | -- |
| COIL-tok | Sparse | 0.341 | 0.949 | 0.660 | -- |
| SPLADE (original) | Sparse | 0.322 | 0.955 | 0.665 | 0.813 |
| **SPLADE-max** | **Sparse** | **0.340** | **0.965** | **0.684** | **0.851** |
| **SPLADE-doc** | **Sparse** | **0.322** | **0.946** | **0.667** | **0.747** |
| **DistilSPLADE-max** | **Sparse** | **0.368** | **0.979** | **0.729** | **0.865** |
| ANCE | Dense | 0.330 | 0.959 | 0.648 | -- |
| TCT-ColBERT | Dense | 0.359 | 0.970 | 0.719 | 0.760 |
| TAS-B | Dense | 0.347 | 0.978 | 0.717 | 0.843 |
| RocketQA | Dense | 0.370 | 0.979 | -- | -- |

Key observation: DistilSPLADE-max matches or exceeds all dense retrieval baselines while maintaining inverted-index efficiency.

### BEIR Zero-Shot (NDCG@10)

| Dataset | BM25 | ColBERT | TAS-B | SPLADE-max | DistilSPLADE-max |
|---------|------|---------|-------|------------|------------------|
| MS MARCO | 0.228 | 0.425 | 0.408 | 0.402 | **0.433** |
| ArguAna | **0.315** | 0.233 | 0.427 | 0.439 | 0.479 |
| Climate-FEVER | 0.213 | 0.184 | 0.228 | 0.199 | **0.235** |
| DBPedia | 0.273 | 0.392 | 0.384 | 0.366 | **0.435** |
| FEVER | 0.753 | 0.771 | 0.700 | 0.730 | **0.786** |
| FiQA-2018 | 0.236 | 0.317 | 0.300 | 0.287 | **0.336** |
| HotpotQA | **0.603** | 0.593 | 0.584 | 0.636 | **0.684** |
| NQ | 0.329 | 0.524 | 0.463 | 0.469 | 0.521 |
| Quora | 0.789 | **0.854** | 0.835 | 0.835 | 0.838 |
| SciFact | 0.665 | 0.671 | 0.643 | 0.628 | **0.693** |
| TREC-COVID | 0.656 | 0.677 | 0.481 | 0.673 | **0.710** |
| **Avg (all 14)** | 0.440 | 0.455 | 0.435 | 0.460 | **0.500** |
| **Best on dataset** | 2 | 2 | 0 | 0 | **11** |

DistilSPLADE-max dominates the zero-shot benchmark, winning 11 of 14 datasets. BM25 wins on 2 datasets (notably ArguAna), confirming that lexical matching still matters for certain domains.

### Ablation Summary

| Change | MRR@10 Delta | Notes |
|--------|-------------|-------|
| Sum → Max pooling | +1.8 | Consistent across FLOPS range |
| Full model → Doc-only | -1.8 | But more efficient (no query encoder) |
| No distillation → Distillation | +2.8 | Largest single improvement |
| High lambda (sparse) → Low lambda (dense) | +1.8 | Trading efficiency for effectiveness |

### Methodological Strengths

- **Clean separation of contributions**: Max pooling, doc-only, and distillation are independently evaluated, making it clear what each adds.
- **Efficiency-effectiveness tradeoff explicitly characterized**: The FLOPS metric and Figures 1-2 show the full Pareto frontier, not just cherry-picked operating points.
- **Zero-shot generalization tested**: BEIR evaluation across 14 diverse datasets goes beyond in-domain MS MARCO performance, testing whether learned expansions transfer.
- **Reproducibility**: Code is public, models trained on public data (MS MARCO), and the method builds on standard BERT infrastructure.
- **Practical deployment path**: Inverted indexes are standard IR infrastructure -- SPLADE representations slot into existing systems without architectural changes.

### Methodological Weaknesses

- **No ablation of the FLOPS regularizer vs. L1 or top-k**: The paper uses FLOPS throughout but doesn't compare against simpler sparsity mechanisms to show it's necessary.
- **Limited analysis of what expansion terms are learned**: One example is shown (Figure 3 in the original SPLADE paper, not reproduced here), but there's no systematic analysis of expansion quality or failure modes.
- **MS MARCO bias**: The model is trained on MS MARCO and evaluated mainly on MS MARCO + BEIR. MS MARCO has short passages (~60 words) and sparse relevance labels (~1.1 relevant per query). Behavior on long documents or dense annotations is untested.
- **No latency measurements**: FLOPS is a proxy for retrieval efficiency, but actual wall-clock latency numbers are absent. The claim "inverted index efficiency" is argued from architecture, not measured.
- **Short paper (6 pages)**: The analysis is necessarily compressed. The distillation process in particular is underspecified -- the two-stage training pipeline has multiple hyperparameter choices that aren't discussed.
- **Static corpus assumption**: All evaluation is offline on fixed collections. No evaluation of incremental indexing (adding documents after initial index construction).

---

## Relevance to claude-memory

### What SPLADE Does That We Don't

1. **Learned term expansion at the token level**: Somnigraph's FTS5 channel (`fts.py`) is plain BM25 over summary + themes. It finds exact term matches and nothing else. When a query says "database" and a memory's summary says "PostgreSQL," FTS5 misses this unless sleep happened to add "database" as a theme. SPLADE would activate "database" in the PostgreSQL memory's representation automatically, via the MLM head's learned associations. This is the vocabulary mismatch problem that Somnigraph currently patches via sleep-driven theme enrichment and vector search fallback.

2. **Contextualized term weighting**: FTS5 uses BM25 weights (term frequency, inverse document frequency, field weights). These are corpus-level statistics. SPLADE produces per-document, context-dependent weights -- the term "Python" gets a high weight in a memory about scripting but a low weight in a memory about snakes. BM25 gives it the same IDF in both. This distinction is particularly relevant for Somnigraph's corpus of personal memories where terms are heavily polysemous across life domains (e.g., "engine" in chess vs. cars vs. software).

3. **Query expansion**: The full SPLADE model expands queries too. A query "how does the scoring pipeline work" might activate "RRF," "feedback," "reranker" -- terms that appear in relevant memories but not in the query. Somnigraph's FTS5 query construction (`fts.py`) uses known phrases and AND/OR logic but performs no semantic expansion. This is what the expansion-wip branch is attempting at a different level (entity focus, multi-query, keyword expansion), but SPLADE does it in a principled, learned way at the token level.

4. **Unified sparse-and-expansion representation**: SPLADE combines exact matching and expansion in a single model. Somnigraph splits these across two channels (FTS5 for exact, vector for semantic) and fuses with RRF. SPLADE's approach is architecturally cleaner -- one model that does both, one index to maintain.

### What We Already Do Better

1. **Hybrid retrieval with RRF fusion** (implemented): Somnigraph's RRF fusion of FTS5 + vector search is a simple, robust strategy that works from day one with zero training data. SPLADE requires MS MARCO-scale training data (500k+ query-passage pairs) to learn good expansions. For a personal memory system that starts empty, RRF over untrained channels is strictly more practical than a learned sparse model that needs fine-tuning.

2. **The reranker handles signal combination** (implemented): Somnigraph's LightGBM reranker (26 features, +6.17pp NDCG over formula) already learns the scoring function from ground truth data. Adding SPLADE as a first-stage retriever would provide a better candidate pool, but the reranker can already compensate for FTS5's limitations by learning interaction effects between `fts_rank`, `vec_rank`, and metadata features. The question is whether the improvement in first-stage candidates would propagate through the reranker.

3. **Feedback loop shapes retrieval over time** (implemented): SPLADE representations are frozen at index time. Somnigraph's feedback loop (explicit utility scores, EWMA aggregation, empirical Bayes prior) and the reranker's learned weights adapt to what's actually useful. A memory with perfect SPLADE expansion terms but consistently low utility would still get demoted by the feedback signal. SPLADE can't do this.

4. **Sleep-based enrichment does expansion offline** (implemented): Somnigraph's sleep pipeline already performs a form of document expansion -- theme enrichment during REM adds terms to the FTS index that weren't in the original content. This is slower and less principled than SPLADE's learned expansion, but it's free (no model inference at retrieval time) and adaptive (themes evolve with the corpus).

5. **SQLite-native infrastructure** (implemented): FTS5 is a built-in SQLite extension. SPLADE would require either (a) a separate inference server for encoding queries, or (b) precomputed document representations stored in a custom inverted index outside SQLite. Either option adds significant infrastructure complexity to a system that currently needs only `memory.db`.

---

## Worth Stealing (ranked)

### 1. SPLADE Score as a Reranker Feature (Medium Value, Medium Effort)

**What**: Run a pre-trained SPLADE model (e.g., `naver/splade-cocondenser-ensembledistil` from HuggingFace) at retrieval time to produce a SPLADE relevance score for each candidate, and add it as a new feature in the LightGBM reranker alongside `fts_bm25` and `vec_dist`.

**Why it matters**: The expansion-wip experiments showed that evidence exists in the candidate pool at rank 100+ but the reranker can't elevate it. A SPLADE score would provide a third retrieval signal that captures vocabulary expansion -- semantically related terms that BM25 misses and vectors approximate noisily. The BEIR results show SPLADE-max outperforms BM25 by 0.020 average NDCG@10 and matches dense retrieval, suggesting the expansion signal is genuinely complementary. Critically, the lesson from expansion-wip is that new signals must be new features, not overwrites. A `splade_score` feature alongside `fts_bm25` lets the reranker learn the right weighting without breaking existing feature semantics.

**Implementation**: Install `transformers` + a SPLADE checkpoint (~250MB). At `remember()` time, encode each memory through SPLADE-doc to produce a sparse representation, store it (either in a separate table or serialized in a JSON column). At `recall()` time, encode the query, compute dot product against stored representations, add the score as a reranker feature. The document-only variant (SPLADE-doc) means query processing is just a lookup, not a forward pass.

**Effort**: Medium. Requires a new dependency (transformers + model weights), storage for SPLADE representations (~200 non-zero weights per memory, ~1.6KB each), and encoding at write time. The reranker integration is trivial -- just one more feature column. Main risk: the model is trained on MS MARCO (web passages), not personal memories. Transfer to Somnigraph's domain is unvalidated.

### 2. Document Expansion at Write Time via SPLADE-doc (Medium Value, Medium Effort)

**What**: At `remember()` time, run SPLADE-doc to identify the top-k expansion terms for each memory, and append them to the FTS5 index as additional searchable text. This turns SPLADE into a write-time enrichment step that improves FTS5 without changing the retrieval pipeline.

**Why it matters**: This addresses the vocabulary mismatch that FTS5 currently handles only through sleep-driven theme enrichment (which is slow, LLM-dependent, and doesn't run until the next sleep cycle). SPLADE expansion is immediate, deterministic, and learned from a large corpus. A memory about "tuning LightGBM hyperparameters" would get expansion terms like "optimization," "gradient," "boosting," "parameters" added to its FTS representation, making it findable by queries using those terms without waiting for sleep.

**Implementation**: At write time in `impl_remember()`, encode the memory content through SPLADE-doc, extract the top-20 non-zero expansion terms (excluding terms already in the content), and append them to the FTS5 indexed text (e.g., as a new field `expansion_terms` with a weight of 1x, lower than summary's 5x). The expansion terms would be stored in a column for inspection and could be updated during sleep if the model is re-run.

**Effort**: Medium. Same model dependency as #1. The FTS5 schema change is straightforward. Risk: expansion terms from a web-passage model may not be the right terms for personal memories.

### 3. Learned Query Expansion for FTS5 (Low-Medium Value, Low Effort)

**What**: Use a pre-trained SPLADE encoder to expand queries at retrieval time. Before sending the query to FTS5, encode it through SPLADE, extract the top-5 activated expansion terms, and OR-join them with the original query.

**Why it matters**: This is the lowest-effort way to get SPLADE-style expansion into the pipeline. It doesn't require changing the index, storing new representations, or modifying the schema. The expansion terms bridge the vocabulary gap at query time -- a query "how does scoring work" might expand to include "RRF" "reranker" "feedback" "pipeline." The expansion-wip multi-query and keyword expansion methods attempted this via LLM calls; SPLADE does it via a small, fast model.

**Implementation**: In `fts.py`, before building the FTS5 query, optionally run the query through a SPLADE encoder (if available), extract top-5 expansion terms, and add them as OR clauses. The SPLADE encoder is small (~250MB DistilBERT) and fast (~10ms per query on CPU).

**Effort**: Low. No schema changes, no storage changes, no retraining. Just a query preprocessing step. The model can be loaded lazily and cached. Failure mode is graceful -- if SPLADE isn't available, the query proceeds unchanged.

---

## Not Useful For Us

### Full SPLADE as a Replacement for FTS5 + Vector Search

SPLADE is designed as a first-stage retriever for large-scale collections (millions of passages). At Somnigraph's scale (~700 memories), the retrieval problem is fundamentally different. BM25 + vector KNN already examines every candidate. SPLADE's efficiency advantage (inverted index vs. brute-force) doesn't matter when the corpus fits in memory. And replacing two well-understood channels (FTS5, sqlite-vec) with a single learned model would remove the channel independence that makes RRF robust. The architectural complexity (model loading, GPU/CPU inference, custom index) outweighs the benefit at this scale.

### FLOPS Regularization for Index Efficiency

The FLOPS regularizer optimizes for balanced posting lists in an inverted index -- a property that matters when scanning millions of documents. Somnigraph scans hundreds. Index efficiency is not a constraint, and the regularization mechanism has no analog in a system that doesn't use posting lists.

### Distillation Training Pipeline

DistilSPLADE-max's two-stage distillation (train retriever + reranker, mine hard negatives, retrain) produces the best results but requires MS MARCO-scale training infrastructure (4x V100 GPUs, 150k training iterations). This is a research-lab training pipeline, not something applicable to a personal memory system. If we use SPLADE at all, it would be as a pre-trained model applied zero-shot.

### Query-Document Joint Training

SPLADE trains query and document encoders jointly to maximize ranking loss. Somnigraph's queries and memories evolve independently -- the system can't retrain a model every time a new memory is added. The joint training assumption (fixed corpus, known query distribution) doesn't hold for a continuously growing personal memory system.

### Replacing the Reranker with SPLADE

SPLADE is a first-stage retriever, not a reranker. It operates in the same stage as FTS5/vector search (candidate generation), not in the same stage as the LightGBM reranker (candidate scoring). The reranker uses 26 features including feedback, graph signals, and metadata that SPLADE has no access to. These are different layers of the pipeline.

---

## Impact on Implementation Priority

### Retrieval pipeline (architecture.md § The retrieval pipeline) -- Unchanged

SPLADE validates the hybrid retrieval architecture. The paper itself notes that dense and sparse signals are complementary -- "dense retrieval can still benefit from BOW models (e.g. by combining both types of signals)." Somnigraph's RRF fusion of FTS5 + vector search is the right structure. SPLADE would be a better sparse channel than BM25, but the two-channel + reranker architecture remains correct.

### Reranker improvement experiments (roadmap Tier 1 #8) -- Strengthened

The remaining reranker improvement experiment (raw-score features) should consider SPLADE score as a candidate feature. The expansion-wip finding that "new signals must be new features, not overwrites" aligns directly with how SPLADE score would be integrated -- as a third retrieval signal feature alongside `fts_bm25` and `vec_dist`, not as a replacement for either.

### Prospective indexing (roadmap Tier 1 #10) -- Modified

SPLADE's document expansion is a mechanistically different approach to the same problem that prospective indexing addresses: bridging the gap between how information is stored and how it's queried. Kumiho's prospective indexing generates hypothetical future queries; SPLADE generates expansion terms from a pre-trained language model. These could be complementary -- SPLADE handles vocabulary expansion (synonyms, hypernyms), while prospective indexing handles intent expansion (what questions would this memory answer?). The experiment design should test both independently and in combination.

### Expansion-wip branch -- New consideration

The expansion-wip work tested six candidate expansion methods and found all neutral because the bottleneck is ranking, not coverage. SPLADE offers a fundamentally different approach: instead of expanding queries at retrieval time (which inflates the candidate pool without helping the reranker prioritize), SPLADE expands documents at index time (which enriches the FTS representation so that the right candidates score higher in the first place). This is worth testing as an alternative to the query-side expansion strategies that failed. The SPLADE-doc variant in particular -- write-time document expansion with no query-time cost -- fits Somnigraph's architecture naturally.

---

## Connections

### To BM25/FTS5 (Somnigraph's current sparse channel)

SPLADE is a direct successor to BM25 in the neural IR literature. The relationship:

| Dimension | BM25 (FTS5) | SPLADE |
|-----------|-------------|--------|
| **Term weighting** | Corpus statistics (TF, IDF, field weights) | Learned, context-dependent (MLM head) |
| **Expansion** | None (exact match only) | Implicit via vocabulary prediction |
| **Index structure** | Standard inverted index | Standard inverted index (same!) |
| **Training data needed** | None | 500k+ query-passage pairs |
| **Domain transfer** | Perfect (corpus-agnostic) | Good (BEIR zero-shot), but trained on web passages |
| **MRR@10 on MS MARCO** | 0.184 | 0.340-0.368 |
| **Infrastructure** | SQLite FTS5 built-in | PyTorch model + custom index |

The key insight: SPLADE beats BM25 by 2x on MRR@10 while using the same index structure. The improvement comes entirely from better term weighting and expansion, not from a different retrieval paradigm.

### To HippoRAG (hipporag.md)

Both papers improve retrieval beyond naive BM25/vector search, but at different levels:

| Dimension | SPLADE | HippoRAG |
|-----------|--------|----------|
| **What it improves** | Term matching (vocabulary mismatch) | Multi-hop reasoning (connecting passages) |
| **Mechanism** | Learned token expansion via MLM | Knowledge graph + PPR traversal |
| **Index-time cost** | One BERT forward pass per document | LLM call per document (OpenIE) |
| **Query-time cost** | One BERT forward pass (or none for doc-only) | LLM call + PPR computation |
| **Training data** | MS MARCO (public, large-scale) | None (uses pre-trained LLM) |
| **Failure mode** | Misses multi-hop connections | Misses vocabulary-level matches |

For Somnigraph, these address orthogonal weaknesses: SPLADE would improve the FTS5 channel (better term matching), while HippoRAG's PPR improves graph traversal (better multi-hop). They operate at different stages and could coexist.

### To the expansion-wip branch

The expansion-wip experiments (entity focus, multi-query, keyword, session expansion, entity bridge, Rocchio PRF) are all query-side expansion strategies applied at retrieval time. SPLADE's document-only variant (SPLADE-doc) is a document-side expansion applied at index time. The key difference: query expansion inflates the candidate pool without changing candidate scores in the existing channels; document expansion enriches the index so that existing channels produce better rankings. The expansion-wip finding -- "evidence is in the candidate pool at rank 100+ but the reranker can't elevate it" -- suggests the bottleneck is ranking, not coverage. SPLADE-doc addresses this differently: instead of adding more candidates, it makes existing candidates more findable by enriching their term representations.

### To Kumiho (kumiho.md)

Kumiho's prospective indexing and SPLADE's document expansion both enrich memories at write time to bridge the cue-trigger disconnect. Kumiho generates 2-3 hypothetical future queries per memory; SPLADE generates expansion terms from the MLM head. Kumiho's approach is more targeted (it predicts specific queries) but more expensive (LLM call per memory). SPLADE's is more general (it expands the vocabulary) but limited to terms the MLM associates with the context. Both are worth testing for Somnigraph, and they likely complement each other.

### To AgentWorkingMemory (awm.md)

AWM uses a 10-phase retrieval pipeline including BM25 + vectors + cross-encoder reranking + beam-search graph walk, all with local ONNX models. SPLADE's local-model approach (DistilBERT, ~250MB) is architecturally compatible with AWM's all-local philosophy and with Somnigraph's desire to avoid API calls during retrieval. A local SPLADE model for query expansion would add ~10ms latency with no API dependency.

---

## Summary Assessment

SPLADE v2 is a well-executed improvement to a clean idea: repurpose the MLM prediction head for term importance estimation, producing sparse vectors that combine exact matching with learned expansion. The paper is concise (6 pages) and makes three clear contributions (max pooling, doc-only variant, distillation), each independently evaluated. The BEIR results are genuinely impressive -- winning 11 of 14 zero-shot datasets suggests the learned expansions generalize beyond MS MARCO.

**For Somnigraph specifically:**

- **Most actionable idea**: SPLADE-doc as a write-time document expansion for the FTS5 index. This addresses the vocabulary mismatch that sleep-driven theme enrichment currently handles slowly and incompletely, without changing the retrieval pipeline or adding query-time latency. It also offers a principled alternative to the query-side expansion strategies that proved neutral in expansion-wip.

- **Second takeaway**: SPLADE score as a reranker feature is the right integration point. The expansion-wip lesson -- new signals must be new features, not overwrites -- directly applies. Adding `splade_score` alongside `fts_bm25` and `vec_dist` lets the reranker decide how much weight to give the expansion signal per query.

- **Third takeaway**: The gap between BM25 (0.184 MRR@10) and SPLADE (0.340-0.368) on MS MARCO quantifies how much Somnigraph's FTS5 channel leaves on the table. However, this gap is measured on web passages with single-sentence queries. For personal memories with enriched summaries and sleep-added themes, the gap is likely smaller -- the enrichment pipeline already does some of what SPLADE would do.

- **Practical constraint**: SPLADE requires a PyTorch model (~250MB) and per-document inference at write time. This is heavier than Somnigraph's current write path (embedding API call + FTS5 insert). Whether the quality improvement justifies the infrastructure complexity depends on whether the reranker can already compensate for FTS5's limitations using existing features.

**Quality of the work**: Solid. Not a top venue publication (arXiv preprint, later published at SIGIR 2022 follow-up), but the code is public, the results are reproducible, and the BEIR benchmark became a standard evaluation. The SPLADE line of work (v1, v2, SPLADE++) has been widely adopted in the IR community, with the HuggingFace model cards showing significant usage. The paper is honest about tradeoffs (effectiveness vs. efficiency) and provides the data to evaluate them.
