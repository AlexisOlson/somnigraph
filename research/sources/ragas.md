# RAGAS: Automated Evaluation of Retrieval Augmented Generation -- Analysis

*Phase 15, 2026-03-06. Analysis of arXiv:2309.15217 (EACL 2024).*

---

## Paper Overview

**Paper**: Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert. "RAGAS: Automated Evaluation of Retrieval Augmented Generation." EACL 2024. arXiv:2309.15217 (v1 Sep 2023, v2 Apr 2025). CC BY 4.0.

**Problem**: RAG pipelines combine retrieval and generation components, each needing separate assessment. Existing evaluation either requires expensive human annotations or uses crude end-to-end metrics that cannot diagnose where failures occur. The paper identifies three quality dimensions that map to the RAG pipeline stages: is the context relevant? is the answer grounded in context? does the answer address the question?

**Core contribution**: A reference-free evaluation framework with three LLM-as-judge metrics -- Faithfulness, Answer Relevancy, and Context Relevancy -- that together provide component-level diagnostics for RAG systems without ground-truth annotations. Achieves 0.95 accuracy (pairwise agreement with human annotators) on Faithfulness, 0.78 on Answer Relevancy, and 0.70 on Context Relevancy.

**Scale**: The paper itself introduces 3 metrics and a 50-question WikiEval benchmark. The open-source RAGAS library (https://github.com/explodinggradients/ragas, 8k+ stars) has since expanded to 30+ metrics covering RAG, agents/tool-use, NL comparison, and general-purpose evaluation. The library, not the paper, is the more significant artifact.

**Distinction**: RAGAS evaluates *RAG pipelines*, not memory systems. The paper assumes a stateless retrieve-then-generate architecture. Persistent memory introduces temporal dynamics, consolidation, decay, relationship edges, and multi-session context that RAGAS does not address. The analysis below focuses on what transfers and what does not.

---

## Metrics Breakdown

### Paper Metrics (Original Three)

#### 1. Faithfulness (F)

**What it measures**: Whether claims in the generated answer can be inferred from the retrieved context. This is a *groundedness* check -- it catches hallucinations and confabulations that go beyond what the context supports.

**Computation**:
1. Decompose the answer into atomic statements S = {s_1, ..., s_n}
2. For each statement, LLM judges whether it is supported by the context
3. F = |V| / |S| where V is the set of verified (supported) statements

**Signals**: High F means the generation is grounded. Low F means the model is hallucinating or drawing on parametric knowledge. The two-step decompose-then-verify approach achieved 95% agreement with human annotators -- the strongest result in the paper.

**Implementation detail**: Uses gpt-3.5-turbo-16k for both steps. The decomposition step is the key innovation -- it converts free-form text into atomic verifiable claims, making the verification step tractable.

#### 2. Answer Relevancy (AR)

**What it measures**: Whether the generated answer directly addresses the input question. Penalizes incomplete answers and answers containing redundant/tangential information.

**Computation**:
1. Generate n candidate questions from the answer (reverse the QA direction)
2. Embed both the original question and all candidate questions (text-embedding-ada-002)
3. AR = (1/n) * sum(cosine_sim(q, q_i))

**Signals**: Low AR means the answer is off-topic, incomplete, or padded with irrelevant information. The embedding-based approach avoids direct LLM scoring, which the paper found unreliable for relevancy judgments. Achieved 0.78 accuracy.

**Subtle design choice**: By generating questions *from the answer* and comparing to the original question, this metric implicitly checks information completeness -- if the answer omits key aspects of the question, the reverse-generated questions will drift away from the original, lowering the score.

#### 3. Context Relevancy (CR)

**What it measures**: Whether the retrieved context contains only information necessary to answer the question. High CR means tight, focused retrieval; low CR means the retrieved passages contain excessive noise.

**Computation**:
1. LLM extracts sentences from the context that are crucial for answering the question
2. CR = (extracted crucial sentences) / (total sentences in context)

**Signals**: Low CR indicates over-retrieval -- the retriever is pulling in tangentially related but unhelpful passages. This is the weakest metric (0.70 accuracy). The authors note that "ChatGPT often struggles with selecting crucial sentences, especially for longer contexts."

**Limitation**: This metric is sensitive to context length. A 50-sentence context with 5 crucial sentences scores 0.10, even if retrieval was actually good -- it just returned too much. This conflates retrieval quality with retrieval quantity.

### Library Metrics (Post-Paper Expansion)

The RAGAS library has expanded significantly beyond the paper. The most relevant additions for our purposes:

#### 4. Context Precision (CP)

**What it measures**: Whether relevant chunks are ranked higher than irrelevant ones in the retrieved context list.

**Computation**: Mean precision at each rank position, weighted by relevance. CP@K = sum(Precision@k * v_k) / total_relevant_in_top_K, where v_k is binary relevance at rank k.

**Requires**: Ground-truth reference or reference contexts for relevance labeling.

**Key difference from CR**: Context Relevancy asks "is the context focused?", Context Precision asks "are the good chunks at the top?" This is a ranking metric, directly applicable to our RRF scoring.

#### 5. Context Recall

**What it measures**: How many relevant pieces of information were successfully retrieved.

**Computation (LLM-based)**: Decompose the reference answer into claims. For each claim, check if it can be attributed to the retrieved context. Context Recall = (attributed claims) / (total reference claims).

**Requires**: A reference answer (or reference contexts for the non-LLM variant).

**Why it matters**: This directly measures the "needle in haystack" retrieval problem. If a memory exists in our store but `recall()` fails to surface it, Context Recall catches that failure.

#### 6. Context Entities Recall

**What it measures**: Named entity overlap between retrieved context and reference. CER = |entities_in_both| / |entities_in_reference|.

**Why it's relevant**: A lightweight, non-LLM proxy for retrieval quality. Particularly useful for our entity-sparse memory store -- if a recall mentions "Matt" or "Skyrim" and the retrieved memories don't contain those entities, something went wrong.

#### 7. Factual Correctness

**What it measures**: End-to-end accuracy of the generated response against a reference answer. Decomposes both into claims, uses NLI to determine overlap, computes precision/recall/F1.

**Key distinction from Faithfulness**: Faithfulness checks response-vs-context (groundedness). Factual Correctness checks response-vs-reference (accuracy). A response can be perfectly faithful to wrong context and still score 0 on Factual Correctness.

#### 8. Noise Sensitivity

**What it measures**: How often the system makes errors when given irrelevant retrieved documents alongside relevant ones. NS = incorrect_claims / total_claims. Lower is better.

**Why it's relevant**: Directly measures robustness to retrieval noise. If `recall()` returns 10 memories but only 3 are relevant, Noise Sensitivity tells us whether the reader (Opus) was misled by the 7 irrelevant ones. This connects to LongMemEval's finding that ~50% of errors are reading failures.

#### 9. Aspect Critic

**What it measures**: Configurable binary LLM-as-judge metric for arbitrary dimensions (harmfulness, coherence, correctness, conciseness, custom aspects).

**Why it's relevant**: This is the extensible primitive. For memory-specific evaluation dimensions not covered by the standard RAG metrics (e.g., temporal accuracy, preference consistency, staleness detection), Aspect Critic provides a framework for building custom LLM-judge evaluators.

---

## Key Claims and Evidence

### Claim 1: LLM-as-judge metrics can replace human annotation for RAG evaluation

**Evidence**: Faithfulness achieves 0.95 agreement with human annotators on WikiEval (50 questions, 2 annotators, 90-95% inter-annotator agreement). Answer Relevancy: 0.78. Context Relevancy: 0.70. All three outperform naive GPT-scoring (ask GPT to rate 0-10) and GPT-ranking (ask GPT to choose between two).

**Evidence quality**: Moderate. The WikiEval dataset is small (50 questions) and constructed specifically for this evaluation -- ChatGPT-generated questions from Wikipedia, with synthetically degraded variants for lower-quality comparisons. The 0.95 faithfulness agreement is impressive but tested on relatively straightforward factual content. No adversarial evaluation. The comparison baselines (GPT Score, GPT Ranking) are naive -- no comparison against established metrics like BERTScore or QA-specific metrics.

**Our assessment**: The core finding -- that decompose-then-verify outperforms holistic scoring -- is likely robust and generalizable. But the specific accuracy numbers should be treated as upper bounds for clean, short-context, factual QA. Our use case (subjective memories, long conversations, personal context) would likely see lower agreement.

### Claim 2: Component-level metrics are more diagnostic than end-to-end metrics

**Evidence**: The three metrics independently diagnose different failure modes: low F = hallucination (generation problem), low CR = noisy retrieval (retrieval problem), low AR = off-topic generation (either). This decomposition lets practitioners localize and fix issues.

**Evidence quality**: Conceptually sound but not empirically validated in the paper. The paper does not show a case study where the metrics guided debugging. The claim is architectural, not experimental.

**Our assessment**: This is the most transferable insight. The decomposition principle -- separate metrics for separate pipeline stages -- applies directly to our `recall()` pipeline. We need metrics for: (a) did `recall()` surface the right memories? (b) did the reader (Opus) use them correctly? (c) was the final response helpful? LongMemEval already confirmed this decomposition matters: ~50% of errors are reading failures, not retrieval failures.

### Claim 3: Reference-free evaluation enables faster development cycles

**Evidence**: The paper argues that removing the need for ground-truth annotations makes evaluation more practical. No empirical comparison of development velocity with vs. without RAGAS.

**Evidence quality**: Weak as a standalone claim. The practical value is obvious but unquantified. The real question is whether reference-free metrics are *accurate enough* to guide development -- the 0.70 Context Relevancy accuracy suggests the answer is "sometimes not."

**Our assessment**: For our system, the hybrid approach is more practical: reference-free metrics for rapid iteration (Faithfulness, Answer Relevancy), reference-based metrics for regression testing (Context Recall, Factual Correctness with curated test cases).

---

## Applicability to Memory Systems

### What Transfers Well

**1. Faithfulness is directly applicable.** When `recall()` surfaces memories and Opus generates a response, Faithfulness measures whether the response is grounded in those memories. This is the reading-quality signal that LongMemEval identified as responsible for ~50% of errors. No adaptation needed -- the metric works as-is on (recalled_memories, generated_response) pairs.

**2. Context Precision maps to RRF ranking quality.** Our RRF scoring (k=14) produces a ranked list of memories. Context Precision measures whether the relevant ones are ranked high. This directly evaluates our scoring coefficients (FEEDBACK_COEFF=0.35, ADJACENCY_BASE_BOOST=0.08, etc.) and could be used as the objective function for future scoring optimization (like our Experiment 7 Bayesian calibration).

**3. Context Recall maps to retrieval completeness.** Given a question where we know the answer requires memories A, B, and C, Context Recall measures whether `recall()` surfaces all three. This is the retrieval-quality signal.

**4. Noise Sensitivity maps to reading robustness.** Our `recall()` deliberately overshoots (casts a wide net, then ranks), so irrelevant memories in the result set are expected. Noise Sensitivity measures whether this hurts downstream generation quality -- directly relevant to tuning QUALITY_FLOOR_RATIO (currently 0.3).

**5. The decompose-then-verify pattern is general.** The Faithfulness metric's two-step approach (extract claims, verify each) can be adapted for memory-specific checks: extract temporal claims from a response, verify each against memory timestamps. Extract preference claims, verify against stored preferences. The pattern is more valuable than the specific metric.

### What Doesn't Transfer

**1. Context Relevancy (the paper's weakest metric) is a poor fit.** It measures retrieval precision as a sentence ratio, which conflates retrieval quality with retrieval volume. For memory systems, we deliberately retrieve more than strictly necessary (our staged curated recall pulls 6-8k tokens, returns ~2k) because the filtering happens downstream. A memory-specific metric should measure whether *useful* memories were retrieved, not whether *only* useful memories were retrieved.

**2. Answer Relevancy's reverse-generation trick is fragile for personal context.** Generating questions from an answer and comparing embeddings works for factual QA ("What year was X born?") but breaks for personal memory queries ("How do I feel about Y?") where the answer is inherently subjective and multiple valid framings exist. The embedding similarity would be unreliable.

**3. No temporal dimension.** None of RAGAS's metrics account for temporal validity. A response that uses an outdated memory (the user moved from Austin to Denver last month) would score high on Faithfulness (the claim *is* in the context) and high on Factual Correctness (if the reference hasn't been updated). We need a "Temporal Faithfulness" metric that RAGAS doesn't provide.

**4. No memory lifecycle evaluation.** RAGAS evaluates a single retrieve-generate cycle. It has no concept of: Was the right memory written in the first place? Has it decayed appropriately? Were contradictions resolved? Are relationship edges accurate? These are write-path and maintenance-path questions that RAGAS's read-path metrics cannot address.

**5. No multi-turn / multi-session awareness.** RAGAS evaluates independent (query, context, response) triples. Memory systems accumulate context across sessions. Questions like "Did the system correctly integrate information from sessions 3, 7, and 12?" require a fundamentally different evaluation structure. LongMemEval's multi-session reasoning (MR) questions address this; RAGAS does not.

### Comparison to LongMemEval's Framework

LongMemEval provides a complementary evaluation lens through its Value/Key/Query/Reading framework:

| LongMemEval Stage | RAGAS Metric | Gap |
|---|---|---|
| **Value** (what's indexed) | None | RAGAS assumes indexing is given; no metric for index quality |
| **Key** (indexing strategy) | Context Precision (partial) | CP evaluates ranking, not the choice of what to index |
| **Query** (search strategy) | None | RAGAS takes the query as given; no metric for query quality |
| **Reading** (interpretation) | Faithfulness, Noise Sensitivity | Good coverage of the reading stage |

The key gap: RAGAS covers the *reading* stage well but provides almost nothing for the *indexing* and *query* stages. LongMemEval's finding that K=V+fact is the best key strategy cannot be evaluated by any RAGAS metric. For a comprehensive memory evaluation framework, we'd need LongMemEval's component decomposition *plus* RAGAS-style automated metrics at each stage.

---

## Adaptation Proposal for claude-memory

### Tier 1: Direct Adoption (works as-is)

**Memory Faithfulness**: Apply RAGAS Faithfulness to (recalled_memories, opus_response) pairs. No adaptation needed. Measures whether Opus stays grounded in recalled context. Implementation: log `recall()` results and the subsequent response; run Faithfulness offline.

- **Effort**: Low (use RAGAS library directly)
- **Value**: High (catches the ~50% reading-failure error class)
- **Ground truth needed**: None (reference-free)

### Tier 2: Adapted Metrics (RAGAS metrics with memory-specific modifications)

**Memory Retrieval Recall**: Adapt Context Recall for our memory store. Requires curated test cases: (query, expected_memory_ids) pairs. Given a query, does `recall()` surface the expected memories?

- **Ground truth**: Manually curated test set. Start with 50 (query, expected_ids) pairs covering the five LongMemEval abilities (IE, MR, KU, TR, Abstention). Use our 386 existing memories as the corpus.
- **Effort**: Medium (curate test data, adapt metric computation)
- **Value**: High (directly measures retrieval quality)

**Memory Ranking Quality**: Adapt Context Precision to evaluate RRF output ranking. For each test query, label which returned memories are truly relevant. Compute mean average precision.

- **Ground truth**: Same test set as above, but with per-result relevance labels (relevant/not-relevant)
- **Effort**: Medium (piggybacks on the same test set)
- **Value**: High (directly evaluates our RRF scoring, could replace or complement Experiment 7's MRR metric)

**Memory Noise Robustness**: Adapt Noise Sensitivity. Inject known-irrelevant memories into recall results, measure whether downstream generation quality degrades.

- **Ground truth**: Synthetic -- add irrelevant memories to result sets
- **Effort**: Low-medium (synthetic data generation)
- **Value**: Medium (validates QUALITY_FLOOR_RATIO tuning)

### Tier 3: Novel Metrics (not in RAGAS, needed for memory)

**Temporal Faithfulness**: Decompose response into temporal claims, verify each against memory timestamps and `valid_from`/`valid_until` fields. A response that cites an invalidated memory scores 0 on that claim.

- **Computation**: Extract temporal statements from response (LLM), cross-reference with memory temporal metadata, compute claim-level accuracy.
- **Effort**: Medium-high (novel metric, needs prompt engineering)
- **Value**: High for Knowledge Update (KU) queries

**Write-Path Quality**: Evaluate whether `remember()` correctly captured the right information at the right granularity with the right metadata (category, themes, priority, decay_rate). Not a RAGAS-style metric -- more like a unit test suite for the write path.

- **Computation**: For each test memory, verify: correct category? meaningful themes? appropriate priority? duplicate detected?
- **Effort**: Medium (test case curation)
- **Value**: Medium (write path is relatively stable; failures are rare but consequential)

**Consolidation Quality**: Evaluate sleep pipeline outputs. Did NREM clustering produce coherent groups? Did edge inference create accurate relationships? Did contradiction detection catch real contradictions without false positives?

- **Computation**: Pre/post comparison of memory store after each sleep cycle. Manual audit of a sample of changes.
- **Effort**: High (requires sleep cycle logging and manual review)
- **Value**: Medium-high (sleep is the most complex pipeline)

**Abstention Accuracy**: Adapted from LongMemEval's abstention ability. When the memory store does NOT contain relevant information, does the system correctly say "I don't know" instead of confabulating? RAGAS has no abstention metric.

- **Computation**: Curate queries about information never stored. Measure false-positive rate (system fabricates an answer from unrelated memories).
- **Effort**: Medium (test case curation)
- **Value**: High (confabulation is the most dangerous failure mode)

### Ground Truth Generation Strategy

The biggest barrier to adopting RAGAS-style metrics is ground truth. Three approaches, in order of practicality:

1. **Retrospective from recall_feedback**: We already collect `recall_feedback()` scores. These can be reinterpreted as relevance labels -- memories scored >0.5 are "relevant," <0.2 are "not relevant." This gives us a noisy but free relevance signal for every real query.

2. **Curated test set**: 50-100 manually curated (query, expected_memories, expected_response) triples. Built once, updated quarterly. Covers the five LongMemEval abilities. This is the ClaudeMemEval benchmark noted in the index as "Medium" priority.

3. **Synthetic degradation**: Following the WikiEval paper's approach, create degraded variants of real recall results (remove a relevant memory, add irrelevant ones, corrupt temporal metadata) and verify that metrics detect the degradation. Cheap to generate, good for regression testing.

---

## Insights Worth Stealing

Ranked by effort-to-impact ratio:

### 1. Decompose-then-verify as a general evaluation pattern (Impact: High, Effort: Low)

The two-step Faithfulness computation -- extract atomic claims, verify each independently -- is applicable far beyond RAG. Apply it to: temporal claim verification, preference consistency checking, contradiction detection during sleep, even recall_feedback automation (decompose a session's use of recalled memories into claims, auto-score each). This is a *pattern*, not a metric.

### 2. Component-level metrics over end-to-end metrics (Impact: High, Effort: Low)

The principle of separating retrieval quality from reading quality from generation quality is the single most important design insight. We should never evaluate `recall()` quality by looking only at final response quality -- the confound between retrieval failures and reading failures makes the signal uninterpretable. LongMemEval already showed this, but RAGAS provides the specific metric machinery.

### 3. Reverse-question generation for relevancy (Impact: Medium, Effort: Medium)

The Answer Relevancy trick -- generate questions that the answer could plausibly answer, then compare to the original question -- is clever. For memory systems, a variant: given a `recall()` result set, generate the question that result set would best answer. If it diverges from the original query, the retrieval drifted. This could be a diagnostic for query-result alignment.

### 4. Embedding-based comparison over direct LLM scoring (Impact: Medium, Effort: Low)

The paper found that direct LLM scoring (rate 0-10) is unreliable, while embedding-based similarity comparison is more stable. This suggests that our recall_feedback mechanism (which asks Opus to assign utility scores) may be noisier than an embedding-based relevance signal. Worth investigating: compute cosine similarity between query embedding and each returned memory embedding, compare to Opus-assigned utility scores. If they correlate strongly, the cheaper signal could supplement or replace the LLM judgment.

### 5. Aspect Critic as a custom metric factory (Impact: Medium, Effort: Low)

The Aspect Critic pattern -- define a criterion in natural language, get a binary LLM judgment -- is the right abstraction for building memory-specific evaluators quickly. "Does this response correctly reflect the user's most recent preference?" "Does this response use information that was explicitly marked as outdated?" These are one-prompt-away from being automated metrics.

### 6. Synthetic degradation for regression testing (Impact: Medium, Effort: Medium)

The WikiEval approach of creating lower-quality variants (answer without context, incomplete answers, add tangential context) is directly applicable. Create a battery of degraded `recall()` results -- remove the most relevant memory, add 5 irrelevant ones, shuffle the ranking -- and verify that our metrics detect each degradation. This is cheaper than curating ground truth and good for regression.

---

## What's Not Worth It

**1. Context Relevancy as defined in the paper.** The sentence-ratio metric conflates retrieval volume with retrieval quality. Our staged curated recall deliberately over-retrieves, and the Sonnet subagent filters. Penalizing over-retrieval would optimize for the wrong thing. Context Precision (ranking quality) is the right metric for us; Context Relevancy is not.

**2. WikiEval as a benchmark.** 50 Wikipedia questions about post-2022 events. No personal context, no multi-session, no temporal reasoning, no preferences. Completely irrelevant to our use case. We need our own ClaudeMemEval benchmark.

**3. The specific RAGAS library integration for production monitoring.** The RAGAS Python library is designed for RAG pipeline CI/CD -- run metrics on every query, track trends. Our system processes ~10-30 recalls per session, and evaluation latency (LLM judge calls) would dominate. Better to run metrics offline in batch, perhaps during sleep cycles or on session transcripts post-hoc.

**4. Response Relevancy for subjective queries.** The reverse-question-generation trick breaks on queries like "What matters to me right now?" or "How did we handle the Skyrim mod design?" where multiple valid answer framings exist. For our personal-context use case, Faithfulness (is the response grounded?) is more reliable than Relevancy (does the response address the question?), because "addressing the question" is underdetermined for open-ended queries.

**5. The multi-modal and SQL metrics.** The RAGAS library has expanded into multimodal faithfulness, SQL query equivalence, and tool-call accuracy. None of these are relevant to text-based memory retrieval.

---

## Key Takeaway

RAGAS provides a principled decomposition of RAG evaluation into component-level metrics and a practical LLM-as-judge implementation for each. Its most transferable contribution is not any specific metric but the *pattern*: extract atomic claims from generated text, verify each claim independently against a reference (context for groundedness, reference answer for accuracy, temporal metadata for currency). For claude-memory, this pattern enables automated evaluation of the reading stage (where ~50% of our errors live) without requiring ground-truth annotations. The retrieval stage is less well-served -- we need ranking-aware metrics (Context Precision) with curated test data, not the paper's reference-free Context Relevancy. The biggest gap is temporal evaluation: no RAGAS metric accounts for memory staleness, knowledge updates, or temporal reasoning, which are core competencies for a persistent memory system. The practical path forward is a hybrid: RAGAS Faithfulness as a free reading-quality signal, Context Precision/Recall with curated test cases for retrieval quality, and custom Aspect-Critic-style metrics for memory-specific dimensions (temporal faithfulness, abstention accuracy, consolidation quality).

---

## Impact on Implementation Priority

The RAGAS analysis has the following effects on the #1-#14 priority list:

**No reordering needed.** RAGAS is an evaluation framework, not an architecture. It provides measurement tools, not implementation alternatives.

**Reinforcements**:
- **#12 (Reading strategy optimization)**: Strongly reinforced. RAGAS Faithfulness provides the first practical metric for evaluating reading quality -- the error class responsible for ~50% of failures (LongMemEval). This was previously a qualitative observation; RAGAS makes it measurable. Adds: "Apply RAGAS Faithfulness as automated reading-quality metric on session transcripts."
- **#14 (Retrieval feedback loop)**: Reinforced. RAGAS Context Precision could serve as an offline complement to our real-time recall_feedback -- a second signal validating that RRF scoring is producing good rankings. The embedding-based comparison insight (cheaper than LLM scoring) may improve feedback signal quality.
- **#10 (CMA behavioral probes)**: Reinforced. RAGAS's decompose-then-verify pattern provides the evaluation methodology for building automated probes. Aspect Critic provides the implementation primitive for custom binary evaluators (e.g., "did the system use outdated information?").

**New sub-item under "Remaining (not yet analyzed)"**:
- **ClaudeMemEval benchmark design**: Elevated from Medium to Medium-High priority. RAGAS provides the metric framework; what's missing is the test data. A curated set of 50-100 (query, expected_memories, expected_response) triples -- covering IE, MR, KU, TR, and Abstention -- would unlock automated regression testing using adapted RAGAS metrics. This is the highest-leverage evaluation investment.

**Connections to existing priorities**:
- The decompose-then-verify pattern is directly useful for #3 (contradiction detection) -- decompose new memories into claims, verify each against existing memories.
- Context Precision could become the objective function for future RRF scoring optimization (#1), replacing or complementing MRR from Experiment 7.
- Noise Sensitivity evaluation could validate QUALITY_FLOOR_RATIO tuning and inform the quality floor that determines which memories are dropped from recall results.

---

## See Also

- [[agent-output-longmemeval]] -- The complementary benchmark analysis. LongMemEval provides the *what to measure* (five abilities, Value/Key/Query/Reading framework); RAGAS provides the *how to measure* (LLM-as-judge metrics, decompose-then-verify pattern). Together they frame a complete evaluation strategy.
- [[agent-output-memoryagentbench]] -- The four-competency benchmark. MemoryAgentBench's evaluation metrics (SubEM, LLM-judge, Accuracy) are cruder than RAGAS's component-level metrics. But its competency decomposition (AR, TTL, LRU, SF) and the selective-forgetting finding (universally catastrophic) identify evaluation dimensions that RAGAS cannot reach.
- [[phase-8-synthesis]] -- Cross-cutting synthesis of 15 sources, including the "~50% reading failures" finding that RAGAS Faithfulness now makes measurable.
- [[retrospective-experiments]] -- Our Experiment 7 (Bayesian scoring calibration) used MRR as the objective. RAGAS Context Precision is a richer alternative that accounts for the relevance of each result, not just whether the top result is correct.
- [[retrieval-feedback]] -- Design sketch for recall_feedback. RAGAS metrics could serve as offline validation that feedback-driven scoring improvements actually improve retrieval quality, closing the loop between real-time feedback and offline evaluation.
