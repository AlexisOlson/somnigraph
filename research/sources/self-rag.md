# Self-RAG: Learning to Retrieve, Generate, and Critique Through Self-Reflection -- Analysis

*Generated 2026-03-22 by Opus 4.6 agent reading arXiv:2310.11511v1*

---

## Paper Overview

**Paper**: Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi (University of Washington, Allen Institute for AI, IBM Research AI). "Self-RAG: Learning to Retrieve, Generate, and Critique Through Self-Reflection." Preprint, arXiv:2310.11511v1, Oct 2023. 30 pages (15 main + 15 appendix). Code: https://selfrag.github.io/ (models on HuggingFace, training/eval code available).

**Problem addressed**: Standard RAG indiscriminately retrieves a fixed number of passages regardless of whether retrieval is necessary, which can introduce irrelevant context that degrades generation quality. Moreover, RAG outputs are not guaranteed to be consistent with retrieved passages since models aren't explicitly trained to leverage and follow facts from retrieved evidence. These two problems -- retrieval timing and output grounding -- are coupled: retrieving when unnecessary adds noise, and generating without critique loses attribution.

**Core claim**: A single LM can learn to decide *when* to retrieve (via a Retrieve token), evaluate *whether* retrieved passages are relevant (IsRel), assess *whether* its own generation is supported by evidence (IsSup), and judge *overall utility* (IsUse) -- all through special "reflection tokens" added to the vocabulary. This internalizes the retrieve-critique loop as next-token prediction, making the model self-correcting without external critic models at inference time and controllable via soft or hard constraints on the reflection token probabilities.

**Scale**: Evaluated on six tasks: PopQA (1,399 queries), TriviaQA-unfiltered (11,313 queries), PubHealth (fact verification), ARC-Challenge (science reasoning), biography generation (FactScore), ASQA (long-form QA with citations). Baselines: Llama2 7B/13B, Alpaca 7B/13B, Llama2-chat 13B, ChatGPT, Ret-ChatGPT, CoVE 65B, SAIL 7B, Toolformer 6B, Llama2-FT 7B, perplexity.ai. Model variants: Self-RAG 7B and 13B (both based on Llama2). Training: 150k instruction-output pairs, Contriever-MS MARCO retriever.

---

## Architecture / Method

### The Reflection Token Framework

Self-RAG's central innovation is encoding retrieval decisions and quality judgments as special tokens in the model's vocabulary. Four token types, each with defined outputs:

| Token Type | Input | Output Values | Function |
|-----------|-------|---------------|----------|
| **Retrieve** | x or (x, y) | {yes, no, continue} | Decides whether to trigger retrieval |
| **IsRel** | (x, d) | {relevant, irrelevant} | Evaluates passage relevance to query |
| **IsSup** | (x, d, y) | {fully supported, partially supported, no support} | Evaluates whether generation is grounded in evidence |
| **IsUse** | (x, y) | {5, 4, 3, 2, 1} | Evaluates overall response utility |

The first three tokens are per-segment (one sentence); IsUse is per-output. The "continue" value for Retrieve allows the model to keep using a previously retrieved passage across multiple segments without re-retrieving.

### Training Pipeline

Training is a two-stage distillation process:

**Stage 1: Train the Critic (C).** GPT-4 is prompted to generate reflection tokens for sampled (input, output) pairs. Type-specific prompts are used (e.g., for Retrieve: "Given an instruction, make a judgment on whether finding some external documents from the web helps to generate a better response."). This produces 4k-20k labeled examples per token type. A Llama2-7B model is fine-tuned on this data to predict reflection tokens, achieving >90% agreement with GPT-4 on Retrieve, IsSup, and IsRel (80% on IsUse, where disagreement concentrates on adjacent scores like 4 vs. 5).

**Stage 2: Train the Generator (M).** The critic C annotates the full training corpus (150k examples) offline:
1. For each segment, C predicts whether retrieval is needed
2. If yes, passages are retrieved and C evaluates IsRel and IsSup for each
3. The best passage (relevant + supported) is selected; if none qualifies, a random passage is included with its (negative) reflection tokens
4. Retrieved passages are masked for loss computation (the model learns to *use* passages, not to *memorize* them)
5. M is trained with standard next-token prediction on the augmented corpus, learning to generate both task output and reflection tokens

The key insight: by generating training data offline with the critic, the approach avoids the expense of RLHF (no reward model during training, no PPO). The reflection tokens become part of the output vocabulary, so M can generate them at inference without needing C.

### Inference Algorithm

At each segment step t:

1. M generates a Retrieve token. If the probability of Retrieve=Yes exceeds threshold delta (default 0.2), retrieval is triggered
2. If retrieving: K passages are retrieved; M processes each in parallel, generating IsRel, the continuation segment, and IsSup for each passage
3. Segments are scored via a weighted combination: `f(y_t, d) = p(y_t | x, d, y<t) + S(Critique)` where `S = sum(w_G * s_G)` for each critique type G
4. Segment-level beam search (beam width B=2) selects the best continuation
5. If not retrieving: M generates normally, with only IsUse evaluation

The critique scores `s_G` are normalized probabilities of the desirable token. For IsRel: `p(Relevant) / (p(Relevant) + p(Irrelevant))`. For IsSup: weighted sum giving Fully=1.0, Partially=0.5, None=0.0. For IsUse: weighted sum mapping {1,2,3,4,5} to {-1, -0.5, 0, 0.5, 1}.

### Controllability at Inference

The weights `w_G` for each critique type are inference-time hyperparameters. This enables test-time customization without retraining:
- **High w_IsSup**: Emphasizes evidence grounding, increases citation precision at the cost of fluency
- **High w_IsUse**: Emphasizes perceived utility, favors more creative/complete responses
- **Retrieval threshold delta**: Controls retrieval frequency; higher delta means less retrieval

This is architecturally significant: the same trained model can be deployed for both factual QA (high IsSup weight, low delta) and creative writing (high IsUse weight, high delta) by adjusting three numbers.

### Adaptive Retrieval vs. Always-Retrieve

A core thesis of the paper: retrieving only when needed outperforms always-retrieving. The model learns to skip retrieval for queries that don't require factual grounding (e.g., "Write a poem about summer"). Retrieval frequency varies by task: ~90% on PopQA (factual questions about rare entities) but much lower on PubHealth (where the claim itself contains the needed information). This is demonstrated empirically -- see the efficiency-accuracy tradeoff analysis below.

---

## Key Claims & Evidence

### Main Results (Table 2, zero-shot evaluation)

| Model | PopQA (acc) | TriviaQA (acc) | PubHealth (acc) | ARC-C (acc) | Bio (FS) | ASQA (em) |
|-------|------------|----------------|-----------------|-------------|----------|-----------|
| **Self-RAG 13B** | **55.8** | **69.3** | **74.5** | **73.1** | 80.2 | **31.7** |
| **Self-RAG 7B** | 54.9 | 66.4 | 72.4 | 67.3 | **81.2** | 30.0 |
| Ret-ChatGPT | 50.8 | 65.7 | 54.7 | 75.3 | -- | 40.7 |
| ChatGPT | 29.3 | 74.3 | 70.1 | 75.3 | 71.8 | 35.3 |
| Ret-Alpaca 13B | 46.1 | 66.9 | 51.1 | 57.6 | 77.7 | 34.8 |
| Llama2-FT 7B | 48.7 | 57.3 | 64.3 | 65.8 | 78.2 | 31.0 |

Self-RAG 7B outperforms ChatGPT on 4 of 6 tasks despite being ~25x smaller. The 7B model occasionally beats 13B on factual precision (Bio FactScore) because smaller models generate shorter, more precisely grounded outputs.

### ASQA Citation Quality

| Model | Precision | Recall |
|-------|-----------|--------|
| **Self-RAG 13B** | **70.3** | **71.3** |
| **Self-RAG 7B** | 66.9 | 67.8 |
| Ret-ChatGPT | 65.1 | 76.6 |
| Ret-Alpaca 7B | 5.5 | 7.2 |
| Ret-Llama2-chat 13B | 19.8 | 36.1 |

Self-RAG surpasses ChatGPT on citation precision. Most retrieval-augmented baselines have catastrophically low citation accuracy (<8%), indicating they don't learn to ground generations in retrieved evidence.

### Ablation Results (7B, 50k training subset)

| Variant | PopQA (acc) | PubHealth (acc) | ASQA (em) |
|---------|------------|-----------------|-----------|
| Self-RAG (50k) | 45.5 | 73.5 | 32.1 |
| No Retriever R | 43.6 | 67.8 | 31.0 |
| No Critic C | 42.6 | 72.0 | 18.1 |
| No retrieval (inference) | 24.7 | 73.0 | -- |
| Hard constraints | 28.3 | 72.6 | -- |
| Retrieve top 1 (always) | 41.8 | 73.1 | 28.6 |
| Remove IsSup | 44.1 | 73.2 | 30.6 |

Key findings:
- **No Critic** causes the largest ASQA drop (-14.0 em), confirming that self-reflection, not just retrieval, drives citation quality
- **Retrieve top 1** (standard RAG) drops PopQA by 3.7 and ASQA by 3.5 vs. adaptive retrieval
- **Removing IsSup** hurts ASQA (-1.5 em) but barely affects factual QA tasks -- IsSup matters most when citation accuracy is measured
- **No retrieval** collapses PopQA (-20.8) but barely affects PubHealth (-0.5), showing the model correctly learns which tasks need retrieval

### Efficiency-Accuracy Tradeoff

Varying the retrieval threshold delta on PopQA and PubHealth shows:
- On PopQA (rare entities), reducing retrieval frequency causes large accuracy drops
- On PubHealth (self-contained claims), accuracy is nearly flat across retrieval frequencies
- The model learns task-appropriate retrieval rates without per-task tuning

### Critic Model Accuracy vs. GPT-4

| Aspect | Llama2-7B Critic |
|--------|-----------------|
| Retrieve | 93.8% |
| IsSup | 93.5% |
| IsRel | 80.2% |
| IsUse | 73.5% |

High agreement on binary/ternary decisions (Retrieve, IsSup), lower on the 5-point IsUse scale where adjacent-score confusion dominates.

### Human Evaluation (50 samples each, PopQA + Bio)

| Metric | Score |
|--------|-------|
| S&P (plausible + supported) PopQA | 92.5% |
| S&P Bio | 70.0% |
| IsRel agreement | 95.0% |
| IsSup agreement | 90.0% |

Human annotators confirm that the model's self-assessed reflection tokens align with their own judgments at high rates.

### Methodological Strengths

- **Clean separation of training and inference costs.** The critic is used offline during data creation, not during inference. This avoids the double-model overhead that plagues RLHF and makes the approach practical for deployment.
- **Controllable inference without retraining.** The weight/threshold mechanism lets users customize behavior (precision vs. fluency, retrieval frequency) at test time. This is genuinely novel compared to RLHF, which bakes preferences into weights.
- **Comprehensive evaluation suite.** Six tasks spanning closed-set, short-form, and long-form generation, with metrics covering accuracy, factuality (FactScore), fluency (MAUVE), and citation quality (precision + recall). Most RAG papers evaluate only on QA accuracy.
- **Honest about limitations.** The paper acknowledges that Self-RAG can still generate unsupported claims, that IsUse agreement is lower than other tokens, and that the approach depends on retriever quality.
- **Practical training data requirements.** 150k instruction-output pairs with 4k-20k critic labels per token type. The critic distillation avoids needing GPT-4 at scale -- only ~20k GPT-4 calls to bootstrap the critic.

### Methodological Weaknesses

- **No temporal or personal memory evaluation.** All benchmarks are static knowledge tasks (Wikipedia QA, fact verification, biography). The paper says nothing about how Self-RAG would handle evolving information, personal context, or multi-session memory -- the domain where Somnigraph operates.
- **Retriever is treated as a black box.** Contriever-MS MARCO is used off-the-shelf with no analysis of retrieval quality or its interaction with the reflection mechanism. If the retriever returns poor candidates, IsRel can filter them out, but the model can't recover information that wasn't retrieved.
- **Segment-level beam search is expensive.** Processing K passages in parallel at each segment step, with beam width B, gives O(K*B) forward passes per segment. At K=5, B=2, this is 10x the cost of standard generation. The paper uses vLLM for efficiency but doesn't report actual latency numbers.
- **Critic training depends on GPT-4 quality.** The critic's ceiling is GPT-4's ability to judge relevance, support, and utility. The 80% IsRel and 73.5% IsUse agreement rates suggest the critic model has non-trivial noise, which propagates into generator training data.
- **No analysis of retrieval decision quality.** The paper shows that adaptive retrieval outperforms always-retrieve, but doesn't analyze *when the model gets the retrieval decision wrong* -- cases where it should have retrieved but didn't, or retrieved unnecessarily. Without this analysis, it's unclear how much headroom exists in the retrieval gating.
- **Single retriever, single corpus.** All experiments use Contriever over Wikipedia. The generalization to other retrieval setups (dense, sparse, hybrid) or non-encyclopedic corpora is untested.

---

## Relevance to claude-memory

### What Self-RAG Does That We Don't

1. **Formalized retrieval gating.** Somnigraph's `recall()` is always triggered by the agent calling the tool -- there's no mechanism for the system itself to decide whether retrieval is necessary. The CLAUDE.md snippet provides heuristic guidance ("recall before reacting"), but the decision is entirely in the agent's hands. Self-RAG formalizes this as a learned decision (the Retrieve token), with a calibrated probability threshold. This is relevant to the `limit` parameter design in `roadmap.md` -- both address "how much retrieval," but Self-RAG also addresses "whether to retrieve at all."

2. **Inline generation critique.** Our feedback loop is post-hoc: the agent uses recalled memories, generates a response, and later provides utility ratings via `recall_feedback()`. Self-RAG's critique happens *during* generation -- the model evaluates passage relevance (IsRel) and output support (IsSup) at each sentence. This is a fundamentally different feedback architecture: real-time self-correction vs. retrospective utility scoring.

3. **Passage-level relevance filtering.** Our reranker (`reranker.py`) scores and ranks candidate memories before returning them to the agent. Self-RAG's IsRel token evaluates relevance *after* retrieval *per passage*, allowing the model to reject irrelevant passages even when the retriever ranked them highly. In Somnigraph, irrelevant passages that rank high are returned to the agent, which must decide to ignore them without structured guidance.

4. **Generation-evidence attribution.** Self-RAG's IsSup token explicitly tracks whether each generated sentence is supported by cited evidence. Somnigraph has no attribution mechanism -- the agent receives memories and generates freely, with no structured signal about which claims are grounded in which memories.

### What We Already Do Better

1. **Feedback accumulates over time.** Self-RAG's reflection tokens are ephemeral -- they evaluate one generation and are discarded. Our feedback loop (`memory_events`, utility scores, EWMA prior, UCB exploration) builds a persistent relevance signal that improves retrieval over weeks and months. Self-RAG has no mechanism for a passage's utility history to reshape future retrieval scoring. This is the difference between a critique system and a learning system.

2. **Multi-channel hybrid retrieval.** Self-RAG uses a single retriever (Contriever). Somnigraph fuses vector search + FTS5 + PPR graph traversal via RRF, then rescores with a 26-feature LightGBM reranker. The reranker's learned interaction effects (`fts_rank` matters more for recent memories) are architecturally unreachable for Self-RAG's single-channel retrieval.

3. **Consolidation and memory evolution.** Self-RAG operates on a static Wikipedia corpus. Somnigraph's sleep pipeline (NREM relationship detection, REM summary generation, consolidation cluster judgment) handles the problems that emerge when memory is persistent and growing: redundancy, contradiction, staleness, vocabulary drift. Self-RAG has no mechanism for any of these.

4. **Memory type differentiation.** All of Self-RAG's passages are treated identically. Somnigraph's five memory types (episodic, semantic, procedural, reflection, meta) with per-type decay rates, priority schemes, and category-aware retrieval enable type-specific handling that flat passage retrieval cannot express.

5. **Agent-controlled retrieval budget.** The `limit` parameter lets the agent specify how many results it wants based on intent (1-3 for fact lookup, 8-13 for broad survey), which outperforms both Self-RAG's fixed beam width and its adaptive threshold (which controls *frequency* of retrieval, not *quantity*). Our design is informed by 846 cutoff events showing the cutoff is a content-level judgment with mean=5.7, std=4.1 -- Self-RAG has no comparable analysis of retrieval quantity.

---

## Worth Stealing (ranked)

### 1. Retrieval Gating Signal for CLAUDE.md Guidance (High Value, Low Effort)

**What**: Self-RAG's Retrieve token formalizes when retrieval is needed vs. unnecessary. While we can't train Claude to generate special tokens, we can translate the Retrieve token's decision logic into more precise CLAUDE.md guidance. Self-RAG's training data reveals that retrieval is unnecessary when queries don't require factual grounding (creative tasks, personal opinions) and necessary for factual claims about specific entities, dates, or technical details.

**Why it matters**: The current snippet says "recall before reacting" -- a heuristic that doesn't distinguish between "tell me about our chess discussion" (needs recall) and "write a commit message for this diff" (doesn't). Self-RAG's 90%+ Retrieve accuracy on mixed-task data shows that the retrieval decision is learnable and has clear patterns.

**Implementation**: Refine the CLAUDE.md snippet's recall guidance with explicit categories: (a) always recall -- references to past work, decisions, corrections, user preferences; (b) consider recalling -- technical questions where stored gotchas might exist; (c) skip recall -- purely generative tasks, code formatting, commit messages. Add the `limit` parameter guidance to each category (1-3 for targeted lookups, 5 for standard, 8-13 for surveys). This is documentation work, not code.

**Effort**: Low. One session to draft and test updated snippet guidance.

### 2. Passage-Specific Relevance Feedback in recall_feedback() (Medium Value, Medium Effort)

**What**: Extend `recall_feedback()` to accept per-memory relevance judgments alongside the overall utility score. Currently the agent rates the entire retrieval set. Self-RAG's IsRel token evaluates each passage independently, which produces finer-grained signal. A per-memory relevance score would tell the reranker which specific memories were useful vs. noise, rather than relying on the coarse signal that all memories in a good retrieval set were good.

**Why it matters**: The utility calibration study showed per-query correlation is strong (r=0.70) but per-memory correlation is weak (r=0.14). The aggregation destroys context. Per-memory ratings would preserve the per-query signal while also building per-memory profiles. Self-RAG's ablation shows that removing IsSup (the generation-evidence link) hurts citation precision -- the per-passage signal carries information that aggregate scores don't.

**Implementation**: `recall_feedback()` already accepts a dict of `{memory_id: utility}`. The infrastructure exists. The change is in the CLAUDE.md guidance: instruct the agent to rate individual memories, not just the set. The reranker could then use per-memory feedback as a feature (currently fb_last/fb_mean are aggregated across all queries). A per-memory-per-query feature would require schema changes to `memory_events` (adding which memory the feedback is *about*, not just which memories were co-retrieved).

**Effort**: Medium. Schema change is small; the training pipeline needs to extract per-memory features; snippet guidance needs updating.

### 3. Support Verification During Sleep (Medium Value, Medium Effort)

**What**: Self-RAG's IsSup token evaluates whether generated text is supported by evidence. Apply this concept during sleep's REM phase: when generating summaries, have the LLM also evaluate whether each claim in the summary is fully supported by the source detail memories. Flag summaries with unsupported claims for regeneration or human review.

**Why it matters**: The "detail loss behind summaries" problem is an open concern (identified by external reviewers, listed in `roadmap.md` as the resolution-fidelity evaluation). Self-RAG's IsSup mechanism provides a concrete approach to measuring this: generate a summary, then evaluate each sentence for support against the source memories. Sentences rated "no support" indicate detail loss.

**Implementation**: During REM summary generation (`sleep_rem.py`), after generating a summary, send each sentence + source memories back to the LLM with the IsSup evaluation prompt. Compute a support score. If below threshold, regenerate with explicit instruction to ground claims. Log support scores in `sleep_log` for the consolidation quality evaluation.

**Effort**: Medium. Requires an additional LLM call per summary (already making one for generation). The IsSup prompt template can be adapted directly from Self-RAG's GPT-4 prompt (Table 11 in appendix).

### 4. Inference-Time Weighting as a recall() Parameter (Low Value, Low Effort)

**What**: Self-RAG's w_G weights let users emphasize different quality aspects at inference time. Somnigraph could expose a similar mechanism: a `mode` parameter on `recall()` that adjusts the reranker's behavior. For example, `mode="precise"` could boost the reranker's weight on fts_rank (exact keyword matching), while `mode="explore"` could boost PPR and lower-ranked candidates.

**Why it matters**: The reranker currently applies the same scoring regardless of query intent. Self-RAG shows that different tasks benefit from different quality tradeoffs (citation precision vs. fluency). Our reranker could similarly benefit from intent-conditioned scoring -- the `limit` parameter handles *quantity*, but not *what kind* of results to prioritize.

**Implementation**: Add a `mode` enum to `recall()`. In `reranker.py`, apply post-prediction score adjustments based on mode (e.g., multiply fts-heavy candidates by 1.2 in precise mode). This is a lightweight intervention that doesn't require retraining -- it's a post-hoc rescoring based on mode.

**Effort**: Low. But value is uncertain -- the reranker already captures intent-dependent patterns through feature interactions. This might add complexity without measurable benefit. Would need A/B testing.

---

## Not Useful For Us

### Fine-Tuning a Generator with Reflection Tokens

Self-RAG's core mechanism -- training a 7B/13B model to generate special tokens as part of its vocabulary -- requires end-to-end fine-tuning of the generator LM. Somnigraph operates as a tool layer around Claude, which is accessed via API. We cannot modify Claude's vocabulary, training data, or decoding process. The reflection token mechanism is architecturally inapplicable to any system that uses an LM as a service rather than hosting its own model.

### Segment-Level Beam Search

Self-RAG's inference algorithm processes K passages in parallel at each sentence, performing segment-level beam search with critique-guided scoring. This is a generation-time algorithm for a model the system controls. Somnigraph provides memories to Claude and receives text back -- we have no control over the generation process between input and output. The beam search mechanism lives in a layer we cannot access.

### Critic Model Distillation Pipeline

The GPT-4 -> Critic -> Generator distillation pipeline is elegant but solves a problem we don't have. Somnigraph doesn't need an internal critic model -- it has explicit feedback from the agent (utility scores) and an external judge (GT relevance from Sonnet). Our learning signal is richer (continuous 0-1 scores with per-query context) and cheaper (no GPT-4 calls for labels) than Self-RAG's binary/ternary critic labels.

### Static Corpus Retrieval Architecture

Self-RAG retrieves from a fixed Wikipedia corpus using a single dense retriever. Somnigraph's memory store is dynamic (memories are added, modified, archived continuously), multi-channel (FTS5 + vector + PPR), and evolving (sleep enriches metadata, feedback reshapes scoring). Self-RAG's retrieval assumptions don't transfer.

### IsUse as a Standalone Utility Signal

Self-RAG's IsUse token provides a 5-point utility judgment independent of retrieved passages. This is conceptually similar to our feedback loop but strictly weaker: it's ephemeral (no persistence), ungrounded (no GT validation), and unlearned (the model predicts it but doesn't update from it). Our EWMA-smoothed utility prior with empirical Bayes centering is a more sophisticated version of the same concept.

---

## Impact on Implementation Priority

### Retrieval gating guidance in CLAUDE.md snippet -- Strengthened

Self-RAG provides empirical evidence that formalizing the retrieval decision (when to retrieve vs. not) improves both accuracy and efficiency. The current snippet's "recall before reacting" heuristic is too coarse. Self-RAG's task-dependent retrieval rates (90% on PopQA, much lower on PubHealth) demonstrate that the optimal retrieval frequency varies dramatically by query type. This strengthens the case for more nuanced recall guidance in the snippet, though the implementation is documentation-level, not architectural. Connects to `docs/claude-md-guide.md` Tier 2 guidance.

### Feedback loop architecture -- Unchanged

Self-RAG's inline critique (IsRel, IsSup, IsUse) is architecturally different from our post-hoc feedback loop. The paper provides no evidence that inline critique is superior to retrospective feedback for persistent memory systems -- Self-RAG evaluates on static benchmarks with no temporal dimension. Our feedback loop's strength (accumulating signal over time, validated at r=0.70 per-query correlation) addresses a problem Self-RAG doesn't attempt. No change to feedback architecture.

### Reranker features -- Unchanged

Self-RAG's passage-level critique tokens (IsRel, IsSup) could conceptually become reranker features, but they require a fine-tuned model we don't have. The reranker already captures relevance through fts_rank, vec_dist, and PPR -- these are structural analogs to IsRel. IsSup (generation support) operates at a layer we can't access (between memory retrieval and Claude's generation). No new reranker features from this paper.

### limit parameter design -- Strengthened

Self-RAG's adaptive retrieval threshold controls *whether* to retrieve, while our `limit` parameter controls *how much* to retrieve. Together they form a complete picture: decide whether to retrieve at all (agent judgment guided by snippet), then decide how many results to return (agent-specified limit with Fibonacci anchors). Self-RAG's ablation showing that "Retrieve top 1" (always retrieve, single passage) underperforms adaptive retrieval by 3.7 on PopQA validates the principle that retrieval quantity should be query-dependent -- the same principle behind `limit`.

### Consolidation quality evaluation -- New consideration

Self-RAG's IsSup mechanism (evaluating whether generated text is supported by evidence) suggests a concrete approach to the open problem of summary fidelity evaluation during sleep. If REM generates a summary from detail memories, IsSup-style evaluation could detect unsupported claims in the summary. This is a small addition to the sleep pipeline that could address the "detail loss behind summaries" concern without requiring a full fidelity benchmark. Connects to `roadmap.md` Tier 3 experiment #20 (resolution-fidelity evaluation).

---

## Connections

### To RAGAS (arXiv:2309.15217)

Both RAGAS and Self-RAG evaluate RAG system quality, but at different points in the pipeline:

| Dimension | Self-RAG | RAGAS |
|-----------|----------|-------|
| **When evaluated** | During generation (inline tokens) | After generation (post-hoc metrics) |
| **Who evaluates** | The generator model itself | External LLM judge |
| **What's evaluated** | Per-passage relevance + per-sentence support + overall utility | Faithfulness + answer relevance + context relevance |
| **Training required** | Yes (fine-tune generator + critic) | No (reference-free, works on any RAG system) |
| **Controllability** | Inference-time weights adjust quality tradeoffs | Diagnostic only (identifies problems, doesn't fix them) |

Self-RAG's IsSup is conceptually identical to RAGAS's faithfulness metric (both measure whether generated claims are supported by evidence). The difference: Self-RAG embeds this evaluation in the generation loop, while RAGAS measures it externally. For Somnigraph, RAGAS's external evaluation approach is more applicable since we can't modify Claude's generation process. Self-RAG's IsSup could inform how we design RAGAS-style evaluation for our retrieval pipeline -- particularly for the consolidation quality evaluation (`roadmap.md` Tier 3 #20).

### To Memory-R1 (arXiv:2508.19828)

Both Self-RAG and Memory-R1 use learned signals to improve memory/retrieval decisions. The comparison illuminates fundamentally different approaches to the same problem:

| Dimension | Self-RAG | Memory-R1 |
|-----------|----------|-----------|
| **Learning signal** | Supervised (GPT-4 distillation) | Reinforcement learning (GRPO) |
| **What's learned** | When to retrieve + passage critique | Store/update/delete policies |
| **Training scale** | 150k examples + 20k critic labels | 152 QA pairs |
| **Model modification** | Fine-tuned vocabulary (reflection tokens) | RL-trained policy over tool calls |
| **Memory operations** | Read-only (retrieval) | Read + write (store/update/delete) |

Memory-R1's RL approach to memory decisions (including *write* operations) is more relevant to Somnigraph than Self-RAG's read-only critique. Memory-R1 learns when to store, update, and delete memories -- operations that map directly to `remember()`, sleep consolidation, and `forget()`. Self-RAG only learns when and how to read. However, Self-RAG's passage-level critique (IsRel, IsSup) provides finer-grained evaluation than Memory-R1's coarse QA reward signal.

### To the Feedback Loop (architecture.md)

Self-RAG's reflection tokens and Somnigraph's feedback loop represent two architecturally distinct approaches to the same problem: using quality signals to improve retrieval over time.

| Dimension | Self-RAG Reflection | Somnigraph Feedback |
|-----------|-------------------|-------------------|
| **Timing** | Inline (during generation) | Post-hoc (after use) |
| **Persistence** | Ephemeral (per-generation) | Persistent (EWMA + Beta prior) |
| **Signal type** | Binary/ternary (IsRel, IsSup) + 5-point (IsUse) | Continuous 0-1 utility |
| **Accumulation** | None (each generation is independent) | Memory-level profiles over time |
| **Validation** | GPT-4 agreement (80-95%) | GT correlation (r=0.70 per-query) |
| **What it shapes** | Current generation's passage selection | Future retrievals' scoring |

Self-RAG optimizes the *current* generation by selecting the best passage + output combination. Somnigraph optimizes *future* retrievals by learning which memories tend to be useful in which contexts. These are complementary, not competing. The ideal system would do both: use reflection-style critique to filter passages during the current generation, then use feedback-style learning to reshape retrieval for future queries. Somnigraph already has the second half; the first half would require changes to how Claude processes retrieved memories, which is outside our control.

### To the Limit Parameter Design (roadmap.md)

Self-RAG's adaptive retrieval threshold and Somnigraph's `limit` parameter both address retrieval quantity, but at different levels:

- **Self-RAG**: Binary gate (retrieve or not) controlled by Retrieve token probability, with threshold delta. Task-dependent but not query-dependent within a task.
- **Somnigraph**: Integer limit (1-13) controlled by the agent per-query, with Fibonacci anchors based on 846 empirical cutoff observations. Query-dependent within and across tasks.

Self-RAG's approach is coarser (yes/no) but automatic (the model decides). Somnigraph's approach is finer-grained (how many) but manual (the agent decides). The paper's analysis of retrieval frequency vs. accuracy validates the principle that retrieval quantity should vary by context -- exactly the principle behind `limit`.

### To Expansion-WIP Work

The current branch tests 6 candidate expansion methods that are all neutral -- evidence exists in candidate pools at rank 100+ but the reranker can't elevate it. Self-RAG's approach to this problem is different: rather than trying to *rank* more candidates higher, it *filters* candidates at generation time via IsRel. This is a post-retrieval filtering approach rather than a re-ranking approach. The implication for expansion-wip: if the reranker can't elevate expanded candidates, a post-retrieval relevance filter (cheaper than IsRel fine-tuning -- even a simple LLM call) might help more than better ranking. However, the key lesson from expansion-wip (overwriting reranker features with semantically different values degrades performance) is orthogonal to Self-RAG's concerns.

---

## Summary Assessment

Self-RAG is a well-executed paper that formalizes an important insight: the decision of *when* to retrieve and the evaluation of *whether* retrieval helped should be part of the generation process, not external to it. The reflection token mechanism is technically clean -- it turns retrieval decisions and quality judgments into next-token prediction, avoiding the costs of RLHF and external critic models at inference time. The controllability via inference-time weights is genuinely useful and under-explored in the literature.

**For Somnigraph specifically:**

- **Strongest takeaway**: The retrieval gating decision ("should I call recall()?") deserves more structured guidance than "recall before reacting." Self-RAG's empirical evidence (90% retrieval rate on rare-entity QA, much lower on self-contained tasks) maps directly to better CLAUDE.md snippet guidance with task-type-specific recall recommendations. This is a documentation improvement, not an architectural change.

- **Second takeaway**: IsSup-style evaluation (is this claim supported by this evidence?) is a concrete, implementable approach to the summary fidelity problem during sleep. Adding a support verification step to REM summary generation would address the "detail loss behind summaries" concern with a well-defined metric.

- **Third takeaway**: Per-passage relevance feedback is a natural extension of `recall_feedback()`. The infrastructure for per-memory utility scores exists; the gap is in the agent's guidance (rate individual memories, not just the set) and the reranker's feature extraction (per-memory-per-query features rather than per-memory aggregates).

- **Limitation for us**: Self-RAG's core mechanism (fine-tuned reflection tokens) is architecturally inapplicable to Somnigraph. We use Claude as a service, not a fine-tuned model. The *concepts* (retrieval gating, passage critique, generation support) translate as design principles, but the *implementation* (special tokens, segment beam search, critic distillation) does not.

**Quality of the work**: Strong preprint from a well-established group (Asai, Hajishirzi at UW/AI2). Comprehensive evaluation across six tasks with multiple metrics, clean ablation studies, human evaluation, and honest acknowledgment of limitations. The code and models are publicly available. The paper was arXived October 2023 and has become influential in the RAG literature (the adaptive retrieval and self-reflection ideas appear in multiple subsequent systems). The main weakness is the lack of temporal or personal memory evaluation -- the paper operates entirely in the static-corpus QA paradigm. For a memory system like Somnigraph, the most transferable insights are at the design principle level, not the implementation level.
