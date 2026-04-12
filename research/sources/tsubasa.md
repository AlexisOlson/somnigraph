# TSUBASA -- Improving Long-Horizon Personalization via Evolving Memory and Self-Learning

*Generated 2026-04-11 by Opus agent reading arXiv:2604.07894*

---

## Paper Overview

**Paper**: Xinliang Frederick Zhang and Lu Wang (Computer Science and Engineering, University of Michigan, Ann Arbor). "TSUBASA: Improving Long-Horizon Personalization via Evolving Memory and Self-Learning with Context Distillation." arXiv:2604.07894, April 9 2026. ACL 2026. 19 pages (9 main + 10 appendix).

**Authors**: Xinliang Frederick Zhang (also author of PRIME, UL-TRA, Narrative-of-Thought) and Lu Wang (senior author, U. Michigan NLP group).

**Code**: Not mentioned in the paper. No repository URL provided.

**Problem addressed**: Existing memory-augmented personalized LLMs face two bottlenecks: (1) a *train-inference gap* where fine-tuning on raw conversations does not prepare the model for the compressed observation format used at inference; (2) a *quality-efficiency tradeoff* where standard RAG paradigms must retrieve ever-larger context windows to improve quality, incurring prohibitive cost. Current memory systems act as "linear-growing vector databases" that fail at temporal reasoning and reflective tasks (Section 1).

**Core approach**: Two synergistic "wings" -- *dynamic memory writing* (structured observation extraction + LLM-driven memory evolution with ADD/UPDATE/RECONCILE/IGNORE operations) and *internalized memory reading* (self-learning on synthetic QA pairs + teacher-student context distillation to bake user knowledge into model parameters). The name stands for "Two-winged perSonalization Unifying Bi-Directional Autonomous memory Storage and parametric Assimilation."

**Key claims**: (1) Observation-level extraction consistently outperforms raw utterance retrieval. (2) Context distillation bridges the train-inference gap. (3) Evolving memory (vs. static accumulation) improves performance especially at larger model scales. (4) TSUBASA achieves Pareto improvement -- higher quality at lower token cost than RAG baselines. (5) TSUBASA-PRO beats prior SOTA (Memory-R1, Mem0) on LoCoMo by ~50% F1 and on LongMemEval.

---

## Architecture

### Storage & Schema

Linear memory store: a numbered list of observation strings, each a standalone factual sentence. No vector database, no embeddings, no graph structure. Entries are indexed by integer position. The memory manager references entries by index for UPDATE and RECONCILE operations (Figure A6). Retrieval at inference uses unspecified similarity-based RAG with k=3 (TSUBASA) or k=10 (TSUBASA-PRO) retrieved entries.

### Memory Types

Single type: **observations**. Defined as "factual observations about the user's life, background, and habits" (Figure A5). The extraction prompt explicitly excludes meta-observations ("X is supportive", "X appreciates") and targets concrete facts: past events, recurring habits, stated preferences. Each observation follows a 5W1H principle (Who/What, Where, When, Why/How) and must be temporally grounded with absolute dates.

No distinction between episodic, semantic, procedural, or reflective memory. Everything is flattened to factual observations.

### Write Path

Two-stage pipeline (Section 4.1, Figures A5-A7):

**Stage 1 -- Observation Extraction**: An LLM processes each conversation session and extracts major observations about the target speaker. The prompt (Figure A5) emphasizes information density: convert 20 raw utterances into ~7 observations (the paper's compression ratio from Section 6.1). This is not a summarization step -- it extracts discrete factual claims, each self-contained.

**Stage 2 -- Memory Evolution**: A memory manager LLM receives the current memory store + new observations and decides on one of four operations per observation (Figure A6):

- **ADD**: New information not already captured
- **UPDATE**: New info adds detail or recency to an existing entry (referenced by index)
- **RECONCILE**: New info contradicts or changes an existing entry; merged into a "Temporal Narrative" (e.g., "On [Date1], X did Y. On [Date2], X now does Z"). This explicitly preserves both the old and new states with temporal grounding.
- **IGNORE**: Already captured or trivial

The RECONCILE operation is the most interesting design choice. The paper explicitly contrasts it with destructive DELETE (Memory-R1, Mem0): "RECONCILE weaves conflicting information into a coherent, self-contained narrative... to ensure these narratives remain robust, we further augment them with explicit temporal grounding" (Section 4.1). This preserves historical context while recording the evolution, rather than erasing the old state.

The paper also notes that RECONCILE accounts for "the inherently non-stationary nature of human preferences and behaviors" (citing Hughes et al., 2020). This is a direct acknowledgment that preferences change and the old state has informational value.

### Retrieval

Similarity-based retrieval over the observation store at inference time. k=3 for TSUBASA, k=10 for TSUBASA-PRO (Appendix C). No details on embedding model or similarity function. At k=5 (~100 tokens), TSUBASA captures "the majority of relevant context, approaching the ceiling with 20x fewer tokens" than full session history (Section 6.3, Figure 2).

### Consolidation / Processing

The memory evolution step IS the consolidation -- it runs after each session, not in a batch. Each new observation is compared against the full current memory store, and the manager decides whether to add, update, reconcile, or ignore. This is session-time consolidation, not offline.

No separate offline consolidation pass. No sleep-like batch processing. No progressive abstraction layers.

### Lifecycle Management

No decay, no deletion, no dormancy. Memories grow monotonically (IGNORE prevents duplicates but nothing is ever removed). The paper acknowledges the store grows over time but does not discuss scaling or forgetting.

The "evolving memory" variant in Table A2 (which performs memory evolution after each session) slightly underperforms "static accumulation" (which just appends all observations) on some metrics for smaller models, but outperforms at 32B. The paper attributes this to the memory manager's reasoning capability scaling with model size: "the memory manager achieves sufficient model capacity, realizing its full potential only beyond 8B params" (Section 6.1).

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Observation extraction outperforms raw utterance RAG | Table A2: RAG(observation) beats RAG(utterance) on F1 across all 4 model sizes (4B: 31.17 vs 28.53, 8B: 28.10 vs 24.81 [sic -- regression at 8B for observations on Recall], 14B: 31.38 vs 27.69, 32B: 28.64 vs 24.31 [same pattern]). Input length drops from 75 to 57-62 tokens. | **Partially supported.** F1 improvements are consistent but modest (2-4 points). The claim is about information density, which is validated by the token reduction. However, the improvements are not dramatic, and some sub-metrics favor raw utterances. |
| Context distillation bridges the train-inference gap | Table 1: Vanilla Training (naive autoregressive on raw conversations) catastrophically fails (F1 5.76-8.28 across scales). TSUBASA (no grounding) recovers to F1 12-22 (still poor). TSUBASA with RAG grounding reaches F1 31-37. | **Well supported.** The vanilla training failure is dramatic and consistent across all model sizes. The gap between "no grounding" and "with grounding" confirms that parametric knowledge alone is insufficient -- RAG grounding is necessary even after distillation. |
| TSUBASA achieves Pareto improvement over RAG | Figure 2: TSUBASA (evolving mem.) achieves higher F1 than RAG baselines at lower token cost. Best config at ~100 tokens matches or exceeds RAG at 200+ tokens. | **Supported for the specific configurations shown.** The Pareto frontier is clear in the plots. However, the comparison is between TSUBASA (which includes fine-tuning cost) and RAG (inference-only). The "efficiency" metric is inference-time tokens, not total compute including training. |
| Evolving memory outperforms static accumulation | Table A2: Mixed. At 4B, static acc. wins on F1 (32.82 vs 31.38). At 8B, static acc. wins (34.50 vs 33.99). At 14B, evolving wins on F1 (37.63 vs 37.07). At 32B, evolving wins (34.25 vs 32.75). | **Weakly supported.** The crossover occurs between 8B and 14B, suggesting the memory manager needs substantial reasoning capability. For smaller models, the evolution step introduces errors that outweigh the deduplication benefit. Honest of the authors to report this. |
| TSUBASA-PRO beats Memory-R1 and Mem0 on LoCoMo | Table 3: TSUBASA-PRO (Qwen3-14B) overall F1 37.63 vs Memory-R1 GRPO 41.10 F1. Wait -- actually examining Table 3, the comparison uses different base models. TSUBASA-PRO with Qwen3-8B gets F1 33.75, and with Qwen3-14B gets F1 37.63. Memory-R1 numbers are from Yan et al. 2025 (LLaMA-3.1-8B). The "nearly 50%" claim appears to be about F1 improvement over Mem0, not Memory-R1. | **Overstated.** The paper claims "outperforms widely used systems like Mem0 by nearly 50%" -- Mem0 F1 is 22.96 (Table 3, Qwen3-14B row) vs TSUBASA-PRO 37.63, which is ~64% improvement. But vs Memory-R1 the comparison is cross-model-family, which makes it less clean. On LongMemEval (Table A1), TSUBASA-PRO static acc. beats Memory-R1 GRPO on F1 (45.75 vs 46.70) and BLEU-1 (42.10 vs 41.10) but loses on Judge (57.40 vs 57.80). |
| Top-d=5 logit truncation is optimal for KL distillation | Figure 3: Non-monotonic relationship. d=1 is worst (hard labels lose soft-label information). d=5 is sweet spot. d>5 degrades due to long-tail noise. | **Well supported.** Clean experimental result with clear mechanistic explanation. d=1 collapses to cross-entropy; d>10 introduces noise from irrelevant vocabulary tokens. The finding that soft labels at d=5 outperform hard labels (d=1) by ~6 F1 points is the strongest empirical contribution. |
| Scaling saturation at 32B | Section 6.1: TSUBASA significantly outperforms baselines at 4-14B but saturates at 32B (only surpasses RAG(observation) by 0.77 F1). | **Interesting but underexplored.** The authors hypothesize that larger models already possess sufficient reasoning and in-context learning capability, reducing the marginal benefit of parametric memorization. This is plausible but not verified. |

---

## Relevance to Somnigraph

### What TSUBASA does that Somnigraph doesn't

1. **Parametric memory internalization**: TSUBASA's second wing -- fine-tuning the LLM itself via context distillation -- is a fundamentally different approach to memory reading. Instead of retrieving memories and presenting them in context, it bakes user knowledge into model parameters. Somnigraph is purely retrieval-augmented; it never modifies the model. This is architecturally impossible for us (Claude is not fine-tunable by end users), but the finding that parametric + retrieval outperforms either alone is worth noting.

2. **RECONCILE as temporal narrative**: TSUBASA's RECONCILE operation explicitly merges contradicting information into a temporal narrative that preserves both old and new states with dates. This is more structured than Somnigraph's contradiction edge classification (hard/soft/contextual/temporal), which labels the contradiction but doesn't produce a merged narrative. TSUBASA's approach creates a single self-contained entry that tells the evolution story; Somnigraph's approach preserves both entries separately with a classified edge between them.

3. **Information-dense observation extraction**: The 20:7 utterance-to-observation compression ratio, combined with the 5W1H extraction prompt, produces higher information density per token than raw storage. Somnigraph's `remember()` stores whatever the user or auto-capture provides -- there is no systematic extraction step that distills raw conversations into dense factual observations.

4. **Synthetic QA generation for self-training**: TSUBASA generates SWI-based QA pairs from the user's own data for fine-tuning. While Somnigraph can't fine-tune Claude, the concept of generating synthetic questions from stored memories could inform evaluation -- e.g., generating test questions from the memory store to measure retrieval quality without manual GT construction.

### What Somnigraph does better

1. **Retrieval sophistication**: TSUBASA's retrieval is an unspecified similarity search over a flat list. Somnigraph has hybrid BM25 + vector with RRF fusion, a 26-feature LightGBM reranker, PPR-based graph expansion, and explicit feedback loop. The retrieval quality gap is likely enormous -- TSUBASA's k=3 retrieval at inference is the weakest link in their pipeline.

2. **Feedback loop**: TSUBASA has no mechanism for the user or system to indicate whether retrieved memories were useful. Somnigraph's per-query utility ratings with EWMA aggregation and UCB exploration provide a closed-loop signal (per-query r=0.70 with GT). TSUBASA's quality signal comes entirely from the fine-tuning objective, which is offline and batch.

3. **Memory lifecycle**: Somnigraph has per-category exponential decay with configurable half-lives, reheat on access, dormancy detection, and offline consolidation. TSUBASA's memories grow monotonically with no decay or deletion. For a long-running system, this is a critical difference.

4. **Memory type semantics**: Somnigraph's 5 types (episodic, semantic, procedural, reflection, meta) enable type-specific retrieval weighting and lifecycle rules. TSUBASA's single observation type cannot distinguish between a procedural correction ("always use --force flag") and an episodic fact ("user visited Tokyo in March").

5. **Offline consolidation**: Somnigraph's three-phase sleep pipeline (NREM, REM, orchestrator) performs cross-memory synthesis, relationship detection, contradiction classification, and summary generation. TSUBASA's memory evolution is session-time only and operates pairwise (one new observation against the existing store). It cannot perform the multi-memory synthesis or progressive abstraction that batch consolidation enables.

6. **Graph structure**: Somnigraph's PPR-based graph expansion, synthetic claim nodes, and edge-typed relationships (EXTRACTED_FROM, claim_coref, contradiction) provide structural reasoning over the memory store. TSUBASA's flat indexed list has no relational structure.

7. **Contradiction handling depth**: Somnigraph classifies contradictions into graded types (hard, soft, contextual, temporal) and preserves both entries with a typed edge. TSUBASA's RECONCILE merges them into a single entry, which is cleaner for simple cases but loses the granularity of knowing what kind of contradiction occurred and whether it might reverse.

---

## Worth Stealing (ranked)

### 1. RECONCILE as Temporal Narrative (Medium effort)

**What**: When new information contradicts existing memory, TSUBASA merges both into a single temporally-grounded narrative: "On [Date1], X preferred Y. On [Date2], X now prefers Z." This preserves evolution history in a single retrievable entry rather than requiring graph traversal across two entries with a contradiction edge.

**Why**: Somnigraph's contradiction edges correctly identify that two memories conflict, but at retrieval time the reader must interpret both entries plus the edge type to understand the evolution. A merged narrative is immediately interpretable and retrieval-friendly. The paper's emphasis on standalone density ("a retriever must be able to understand a refined_observation without seeing any other context" -- Figure A6) is well-suited to our BM25+vector retrieval, which scores individual memories independently.

**How**: During sleep consolidation (not at write time -- Somnigraph's write path is user-controlled), when processing contradiction edges of type "temporal", generate a merged narrative that tells the evolution story. Store as a new semantic memory superseding both originals. The contradiction edge and original memories remain in the audit trail. This is complementary to the existing contradiction classification, not a replacement.

### 2. 5W1H Observation Extraction Prompt for Sleep Pipeline (Low effort)

**What**: TSUBASA's extraction prompt (Figure A5) enforces standalone, fact-dense observations with temporal grounding. Each observation is a complete sentence with subject, action, and context. The "Temporal Awareness" instruction -- "distinguish between the date of the report and the date of the event" -- is particularly valuable.

**Why**: Somnigraph's REM sleep step extracts claims and segments from memories, but the extraction prompt could benefit from the 5W1H discipline and the explicit temporal grounding instruction. Currently, extracted claims may lack temporal context that makes them useful as vocabulary bridges.

**How**: Incorporate the 5W1H structure and temporal grounding instruction into the claim extraction prompts in the sleep REM pipeline. Low effort -- this is a prompt improvement, not an architectural change.

### 3. Synthetic QA for Retrieval Evaluation (Medium effort)

**What**: TSUBASA generates QA pairs from the user's data for self-training. The filtering pipeline removes uninformative temporal questions, unanswerable questions, and overly long answers (Appendix A).

**Why**: Somnigraph's ground truth construction (via `build_ground_truth.py`) requires manual LLM judging. Generating synthetic QA pairs from the memory store could provide a cheaper, more scalable evaluation signal -- particularly for regression testing after reranker retrains or pipeline changes. The QA pairs would test whether the retrieval pipeline can find the memory that answers the question.

**How**: Given the memory store, prompt an LLM to generate factual questions answerable from specific memories. Use these as automated retrieval evaluation: can `recall(question)` surface the source memory in the top-k? This doesn't replace human GT judging but provides a fast regression check. The filtering criteria from Appendix A (drop temporal, unanswerable, verbose) are directly applicable.

### 4. Information Density as a Retrieval Optimization (Low-Medium effort)

**What**: The paper's central finding that observation-level extraction beats raw utterance retrieval (consistent across all model sizes) suggests that denser, more distilled memory entries are easier to retrieve correctly.

**Why**: Somnigraph stores memories at whatever granularity the user or auto-capture provides. Some memories are dense factual statements; others are verbose with conversational context. The retrieval pipeline (BM25 + vector) may work better on uniformly dense entries because (a) BM25 gets more signal per token and (b) embedding similarity is less diluted by conversational filler.

**How**: During sleep consolidation, identify verbose memories and distill them into denser observation-style entries. This is already partially addressed by the summary generation in sleep, but could be made more aggressive with a 5W1H-style density target.

---

## Not Useful For Us

1. **The parametric fine-tuning wing entirely**: Somnigraph's host model (Claude) is not fine-tunable. The context distillation approach (teacher-student KL divergence with top-d logit truncation) requires LoRA training on the inference model. This is architecturally incompatible. The finding that parametric + retrieval outperforms either alone is interesting but not actionable.

2. **The Qwen3 model family specifics**: All experiments use Qwen3-4B through 32B. The model-specific findings (saturation at 32B, memory manager capability threshold at 8B) are tied to this family and do not transfer to our setting where Claude is the reasoning model.

3. **The self-learning QA synthesis pipeline**: The mechanism for generating synthetic QA pairs for fine-tuning is coupled to the parametric training objective. We could adapt the *evaluation* aspect (Worth Stealing #3) but the training pipeline itself is irrelevant.

4. **The top-d logit truncation finding**: An elegant result (d=5 beats both d=1 and d>10) but only matters for KL-divergence-based distillation, which we cannot do.

5. **TSUBASA-PRO configuration**: The PRO variant (Appendix C) differs only in k=10 retrieval and a "concise generation" system prompt. These are trivial engineering choices, not insights.

---

## Connections

### To Prior Somnigraph Analyses

**Memory-R1 (arXiv:2508.19828)**: TSUBASA explicitly benchmarks against Memory-R1 and claims competitive or superior results (Table 3, Table A1). The key architectural difference: Memory-R1 uses RL to learn operation policies; TSUBASA uses structured prompts for operations and invests its learning budget in parametric distillation instead. Both use ADD/UPDATE/DELETE-style operations, but TSUBASA's RECONCILE (temporal narrative merge) is a more thoughtful design than Memory-R1's DELETE (destructive erasure). Memory-R1's RL-trained operations discover emergent consolidation behaviors; TSUBASA's prompted operations are more explicit but less adaptive.

**Mem0 (arXiv:2504.19413)**: TSUBASA's memory evolution prompt (Figure A6) is structurally similar to Mem0's operation set but with two differences: (1) RECONCILE replaces DELETE, preserving evolution history; (2) temporal grounding is enforced in every operation. The paper frames this as addressing Mem0's weakness of "destructive DELETE operations that erase historical context" (Section 1).

**EvoMemory (arXiv:2511.20857)**: Both TSUBASA and EvoMemory address evolving preferences, but EvoMemory benchmarks test-time adaptation while TSUBASA focuses on parametric internalization. The evolving memory manager designs are similar in spirit (both use UPDATE/RECONCILE-style operations) but different in execution.

**PERMA (arXiv:2603.23231)**: TSUBASA does not evaluate on PERMA, but the multi-domain cross-session preference maintenance that PERMA tests is exactly the scenario where TSUBASA's evolving memory should shine. The RECONCILE operation's temporal narrative format directly addresses PERMA's temporal depth dimension. If TSUBASA were evaluated on PERMA, the multi-domain Turn=1 metric (currently 0.306 SOTA) would be the interesting test -- TSUBASA's flat memory store has no mechanism for cross-domain synthesis beyond what the retriever happens to surface.

**A-Mem (arXiv:2502.12110)**: A-Mem's inter-memory links (+121% multi-hop) address structural relationships that TSUBASA's flat indexed list cannot represent. TSUBASA's RECONCILE partially compensates by merging related information into single entries, but this only handles contradiction-driven merges, not arbitrary relational links.

### Broader Patterns

TSUBASA represents a growing trend of *hybrid parametric-retrieval* approaches to memory, joining PRIME (Zhang et al., 2025) and Personalized Pieces (Tan et al., 2024a) in arguing that retrieval alone is insufficient for deep personalization. The consistent finding across these papers -- that fine-tuning on user data provides benefits that retrieval cannot replicate -- is important context for Somnigraph's purely retrieval-based design. Somnigraph compensates with retrieval sophistication (reranker, graph, feedback), but the parametric gap is real.

The RECONCILE operation, independently designed, converges with Somnigraph's temporal contradiction classification. Both systems recognize that preference evolution is informational, not just noise to be resolved. The difference is in representation: TSUBASA merges into a narrative; Somnigraph preserves separate entries with typed edges. Both are valid; the merged narrative is better for retrieval simplicity, the separate entries are better for analytical depth.

---

## Summary Assessment

TSUBASA is a well-structured paper that cleanly separates the memory writing problem (how to maintain a high-quality memory store) from the memory reading problem (how to internalize memories into model parameters). The writing wing's RECONCILE operation is a genuine contribution -- it is the first memory system in our corpus to explicitly design for preference evolution preservation rather than treating contradictions as errors to be resolved. The reading wing's context distillation with top-d logit truncation is a clean technical contribution but is irrelevant to systems that cannot fine-tune their host model.

**Strengths**: Information density argument is well-supported empirically. RECONCILE is a principled design choice with clear motivation. The Pareto efficiency analysis (Figure 2) is a useful framing. Honest reporting of evolving memory's weakness at small model scales.

**Weaknesses**: Retrieval mechanism is a complete black box -- no embedding model, no similarity function, no retrieval evaluation metrics. The "nearly 50% over Mem0" headline claim is cherry-picked (F1 improvement over the weakest meaningful baseline). Cross-model-family comparisons with Memory-R1 are imprecise. No evaluation of memory store quality independent of end-to-end QA (unlike PERMA's decoupled evaluation). The parametric wing requires per-user fine-tuning, which limits practical deployment.

**For Somnigraph**: The main takeaways are (1) RECONCILE-as-temporal-narrative is worth adapting for our sleep pipeline's handling of temporal contradictions, (2) 5W1H extraction discipline could improve our claim extraction prompts, and (3) synthetic QA generation from the memory store could provide cheap regression testing for retrieval quality. The parametric internalization wing is architecturally incompatible but provides useful context: purely retrieval-based systems (like ours) have a ceiling that parametric approaches can exceed, and we should be aware of that ceiling even if we cannot cross it.

**Relevance**: Medium. The writing wing contributes one stealable idea (RECONCILE narrative) and validates information density as a retrieval optimization lever. The reading wing is inapplicable. The paper is less relevant to Somnigraph than Memory-R1 (which validated RL for memory operations) or PERMA (which introduced the multi-domain evaluation we plan to run), but more relevant than most implementation-focused repos.
