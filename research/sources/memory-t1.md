# Memory-T1: Reinforcement Learning for Temporal Reasoning in Multi-session Agents -- Analysis

*Phase 15, 2026-03-06. Analysis of arXiv:2512.20092 (Dec 2025).*

---

## Paper Overview

**Paper**: Yiming Du, Baojun Wang, Yifan Xiang, Zhaowei Wang, Wenyu Huang, Boyang Xue, Bin Liang, Xingshan Zeng, Fei Mi, Haoli Bai, Lifeng Shang, Jeff Z. Pan, Yuxin Jiang, Kam-Fai Wong (2025). arXiv:2512.20092, December 23 2025. Multi-institutional (Huawei Noah's Ark Lab, CUHK, HKUST, University of Edinburgh). cs.CL.

**Problem addressed**: Existing long-context LLMs degrade as conversation histories expand, especially for time-dependent questions. Multi-session dialogues accumulate temporal context that standard retrieval (lexical or semantic) handles poorly -- a question about "what happened last week" requires temporal reasoning that BM25 and vector search don't natively support. Prior RL approaches (Time-R1) rely on structured metadata and fail on unstructured multi-session dialogues.

**Core claim**: A coarse-to-fine retrieval framework combining temporal filtering, BM25 relevance scoring, and GRPO-trained evidence selection -- trained with a novel multi-level reward (accuracy + evidence grounding + temporal consistency) -- achieves SOTA on Time-Dialog (67.0% with 7B) and generalizes out-of-domain to LoCoMo. The temporal consistency reward, operating at both session and utterance levels, is the distinguishing contribution.

**Scale**: Trained on 4,065 QA examples from Time-Dialog. Models: Qwen2.5-3B/7B-Instruct. Evaluated on Time-Dialog (200 test examples, 11 temporal reasoning subtasks across 3 difficulty levels) and LoCoMo (5 subtasks, out-of-domain). Compared against GPT-4, Qwen2.5-14B, LLaMA-3.1-8B, Time-R1, MemAgent.

**Critical distinction**: Memory-T1 is a **read-time retrieval and reasoning system**, not a memory management system. It performs NO write operations (no ADD/UPDATE/DELETE). The "memory" in the title refers to selecting from existing dialogue sessions, not constructing or maintaining a memory bank. This fundamentally distinguishes it from Memory-R1 and Mem-alpha.

---

## Architecture / Method

### Coarse-to-Fine Retrieval Pipeline

The architecture has two phases, each progressively narrowing the evidence:

**Phase 1 -- Candidate Generation (heuristic)**:

1. **Temporal Filtering**: An LLM predicts the query's target temporal window (t_start, t_end). All sessions with timestamps outside this window are discarded, producing a temporally-filtered set M_temp. The paper does not specify which LLM performs this prediction or the prompt used.

2. **Relevance Filtering**: BM25 ranks the temporally-filtered sessions by lexical relevance to the query. Top-k sessions form the candidate pool C. At k=10, evidence recall reaches ~90% (Figure 3). BM25 parameters are unspecified.

Phase 1 is entirely heuristic -- no learning involved. Its purpose is to reduce the input space from potentially hundreds of sessions to ~10 high-recall candidates.

**Phase 2 -- Fine-grained Selection (learned)**:

The RL agent receives the candidate pool C and produces a structured output:

```
{selected_memory:[session_3,session_16]. answer:19 days.}
```

This is an end-to-end selection-and-generation step: the agent simultaneously chooses which sessions constitute evidence and generates the answer grounded in those sessions. The structured format enables the reward function to separately evaluate evidence quality and answer quality.

### GRPO Training

**Algorithm**: Group Relative Policy Optimization (GRPO), which computes advantages relative to batch-average rewards rather than a learned value function. Same algorithm used by Memory-R1 and Mem-alpha.

**Hyperparameters**:
- Base models: Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct
- Batch size: 32
- Learning rate: 1e-6
- K=8 rollout responses per prompt
- KL coefficient: 0.1
- Maximum sequence length: 16k tokens
- Framework: VERL

### Multi-Level Reward Design

The reward function is the paper's core contribution. Overall reward:

```
R = w_a * R_a + w_g * R_g + w_t * R_t    (if parsing succeeds)
R = -0.5                                   (if parsing fails)
```

Weights: w_a=0.6, w_g=0.2, w_t=0.2. Range: [-1, 1].

| Component | What It Measures | Formula |
|-----------|-----------------|---------|
| R_a (Accuracy) | Answer correctness | Task-specific: EM for options, unit-aware accuracy for timestamps, epsilon-EM for intervals, Hamming accuracy for sequences |
| R_g (Evidence Grounding) | Source identification | Jaccard Index of predicted vs. gold session IDs, scaled to [-1, 1] |
| R_t (Temporal Consistency) | Temporal alignment | alpha * R_s + beta * R_f (alpha=beta=0.5) |

The temporal consistency reward has two sub-components:

**Chronological Proximity (R_s)**: Session-level. Logistic function penalizing temporal distance between session timestamp and query range:
```
R_s = c / (1 + exp(x)) - d
where x = (gap(U, I_Q) - m) / s
c=1.5, d=0.5, m=7 days, s=1
```
A session within 7 days of the query range gets near-maximum reward; sessions further away are progressively penalized.

**Chronological Fidelity (R_f)**: Utterance-level. Scores individual events within selected sessions:
- +1 if event fully within query temporal range
- +0.5 if partially overlapping
- -1 if no temporal overlap

This two-level temporal reward (session proximity + utterance fidelity) is the architectural novelty. It provides dense supervision for temporal reasoning at two granularities, addressing the sparse reward problem that makes temporal RL hard.

### Training Data: Time-Dialog

Extended from the Time dataset (Wei et al., 2025) with augmented temporal annotations:

- **Size**: 4,716 QA examples (4,065 train / 451 val / 200 test)
- **Augmentation method**: Iterative GPT-4 annotation + human verification (>95% accuracy)
- **Added annotations**: Question temporal range (I_Q), evidence grounding (M*), utterance-level event times

**11 Reasoning Subtasks across 3 Difficulty Levels**:
- Category A (basic): Localization, Duration Comparison, Computation, Order Comparison, Extraction
- Category B (intermediate): Explicit Reasoning, Order Reasoning, Relative Reasoning
- Category C (complex): Counterfactual, Co-temporality, Timeline

Annotations serve only as training-time reward signals -- withheld during inference.

---

## Key Claims & Evidence

### Main Results (Table 1)

**Time-Dialog Performance**:

| Method | Category A | Category B | Category C | Overall |
|--------|-----------|-----------|-----------|---------|
| GPT-4 (oracle) | 56.3 | 76.8 | 79.8 | 64.8 |
| Qwen2.5-14B | 59.2 | 60.1 | 66.3 | 60.7 |
| Qwen2.5-7B | 52.8 | 56.4 | 54.2 | 53.3 |
| Time-R1 | -- | -- | -- | 49.4 |
| MemAgent | -- | -- | -- | 49.9 |
| **Memory-T1 (3B)** | 49.5 | 79.5 | 80.3 | **66.9** |
| **Memory-T1 (7B)** | -- | -- | -- | **67.0** |

Key findings:
- Memory-T1 (3B) outperforms Qwen2.5-14B (66.9 vs 60.7) -- a 3B model beating a 14B model through better temporal reasoning policy
- Memory-T1 (7B) matches GPT-4 oracle (67.0 vs 64.8) at a fraction of the size
- Category B and C (intermediate/complex temporal reasoning) show the largest gains: 79.5 and 80.3 vs 56.4 and 54.2 for the base Qwen2.5-7B
- Time-R1 (49.4%) and MemAgent (49.9%) trail by >17 absolute points

### Context Length Robustness (Figure 4)

This is the most compelling evidence. Performance across input lengths:

| Context Length | Qwen2.5-7B Baseline | Memory-T1 | Gap |
|---------------|---------------------|-----------|-----|
| 0-8k | ~70% | ~70% | ~0 |
| 8-16k | ~65% | ~75% | +10 |
| 16-32k | ~55% | ~75% | +20 |
| 32-64k | ~45% | ~70% | +25 |
| 64-128k | ~40% | ~65% | +25 |

Memory-T1 maintains 65-75% F1 across all lengths while baselines collapse by >30 points. The advantage grows with context length -- exactly where temporal filtering matters most, because longer dialogues have more sessions to confuse the model.

### Ablation Results (Table -- 3B model)

| Configuration | Cat A | Cat B | Cat C | Overall |
|--------------|-------|-------|-------|---------|
| Full model | 49.5 | 79.5 | 80.3 | 66.9 |
| w/o R_t (no temporal) | 45.6 | 75.1 | 64.3 | 63.5 (-5.1%) |
| w/o R_s only | 61.1 | 34.8 | 66.3 | 66.3 (-0.9%) |
| w/o R_f only | 50.0 | 56.5 | 63.0 | 64.8 (-3.1%) |
| w/o R_g (no grounding) | 40.9 | 75.3 | 75.9 | 60.8 (-9.1%) |
| R_a only | 43.6 | 57.5 | 59.0 | 51.9 (-22.4%) |

The ablation reveals a nuanced picture:

1. **Evidence grounding (R_g) is the most important auxiliary signal**: Removing it causes -9.1% overall. Without R_g, the agent can produce correct answers without citing correct sources, so the policy never learns to ground its reasoning.

2. **R_a alone is insufficient**: -22.4% overall. Answer-only reward produces a policy that guesses well on simple questions but fails at complex temporal reasoning. This is the sparse-reward problem -- correctness alone is too binary.

3. **The paradoxical R_s removal**: Removing session-level temporal proximity *improves* Category A by +23.4% but *destroys* Category B by -56.2%. This means R_s creates slight overhead for simple tasks (where temporal filtering is unnecessary) but is essential for intermediate reasoning (where session-level temporal context is the primary signal). The synergy between R_s and R_f is load-bearing.

4. **Utterance-level R_f matters more than session-level R_s**: Removing R_f alone (-3.1%) hurts more than removing R_s alone (-0.9%), and R_f's damage is spread across all categories rather than concentrated in one.

### Out-of-Domain Generalization (LoCoMo, Table 3)

| Model | Single-Hop | Multi-Hop | Temporal | Open-Domain | Adversarial | Overall |
|-------|-----------|----------|----------|------------|-----------|---------|
| Qwen2.5-3B (Non-RAG) | 49.8 | 28.7 | 24.5 | 13.5 | 16.6 | 33.5 |
| Memory-T1 (Non-RAG) | 51.2 | 30.2 | 31.5 | 15.8 | 26.0 | 37.7 |
| Qwen2.5-3B (RAG) | 46.0 | 22.0 | 27.3 | 11.4 | 19.5 | 31.9 |
| Memory-T1 (RAG) | 48.9 | 25.8 | 30.7 | 14.6 | 29.8 | 36.7 |

+4.2% overall improvement in the Non-RAG setting. The strongest gains are on Temporal (+7.0) and Adversarial (+9.4) subtasks -- exactly the categories where temporal reasoning policy should help. Notably, Memory-T1 performs better in Non-RAG than RAG mode, suggesting the learned internal temporal reasoning supersedes simple retrieval.

### Noise Robustness (Table 4)

| Noise Level | Overall |
|-------------|---------|
| 0% | 67.0 |
| 5% | 67.0 |
| 10% | 62.0 |
| 20% | 60.0 |

Graceful degradation under temporal label noise. Even at 20% corrupted temporal annotations, performance only drops 7 absolute points. Temporal reasoning subtasks remain above 88.9% at 20% noise, suggesting the model learns robust temporal patterns rather than memorizing label-specific mappings.

### Efficiency (Table 5)

Average latency: 1.26 seconds per query. Retrieval overhead: 0.01s (negligible). The coarse-to-fine approach is computationally efficient because Phase 1 (BM25 + temporal filtering) is fast, and Phase 2 operates on only ~10 candidate sessions.

### Methodological Weaknesses

1. **No variance reporting**: Single-run results. With GRPO rollouts and stochastic generation, variance matters.

2. **Temporal window prediction is a black box**: The LLM-predicted (t_start, t_end) is critical to Phase 1 filtering but the paper never specifies which model performs this, how it is prompted, or what happens when it makes errors. If the temporal window prediction is wrong, ground-truth evidence sessions get filtered out before the RL agent ever sees them.

3. **Oracle gap**: Memory-T1 achieves 67.0% vs. the GPT-4 oracle's 86.2% (estimated from partial data). The 19-point gap suggests substantial room for improvement, possibly in the Phase 1 filtering bottleneck.

4. **No write operations**: The system selects from fixed dialogue sessions. It never constructs, updates, or consolidates memory. This limits its applicability to the "retrospective temporal QA" use case.

5. **Training data scale**: 4,065 training examples is 27x more than Memory-R1's 152. The data efficiency comparison is unclear -- is Memory-T1's advantage from the reward design or simply from more training data?

6. **No explicit limitations section**: The paper lacks a dedicated limitations discussion, which is a methodological weakness in itself.

---

## Comparison with Memory-R1 and Mem-alpha

This is the third independent "RL for memory" paper, but it occupies a fundamentally different position in the design space. The comparison reveals more about the landscape than about any single paper.

### Structural Comparison

| Dimension | Mem-alpha | Memory-R1 | Memory-T1 |
|-----------|-----------|-----------|-----------|
| **Core task** | Memory construction | Memory construction + retrieval | Temporal evidence selection |
| **Write operations** | INSERT, UPDATE, DELETE | ADD, UPDATE, DELETE, NOOP | **None** |
| **Read operations** | BM25 top-k (frozen) | Similarity RAG + learned distillation | BM25 + temporal filter + learned selection |
| **RL algorithm** | GRPO (no KL) | PPO + GRPO | GRPO |
| **Reward signals** | 4 (correctness, format, compression, content) | 1 (exact match) | 3 (accuracy, grounding, temporal) |
| **Reward complexity** | Multi-signal, weighted | Single binary signal | Multi-signal, multi-level |
| **Training data** | 562 instances | 152 QA pairs | 4,065 QA pairs |
| **Model scale** | Qwen3-4B | LLaMA-3.1-8B, Qwen2.5-3B/7B/14B | Qwen2.5-3B/7B |
| **Memory types** | 3 (core, semantic, episodic) | Untyped flat bank | N/A (no memory bank) |
| **Temporal reasoning** | None | None | Core focus |
| **Benchmarks** | MemoryAgentBench | LoCoMo, MSC, LongMemEval | Time-Dialog, LoCoMo |
| **Contradiction handling** | Explicitly excluded | Not addressed | Not addressed |
| **Context length robustness** | 30k->400k+ generalization | Not tested | Stable to 128k |

### What Memory-T1 Adds to the RL-for-Memory Landscape

1. **Temporal specialization**: Memory-R1 and Mem-alpha treat all information as temporally flat. Memory-T1 demonstrates that temporal reasoning is a distinct capability that benefits from dedicated reward signals. The temporal consistency reward (session proximity + utterance fidelity) provides dense supervision that pure accuracy rewards cannot.

2. **Multi-level reward granularity**: Mem-alpha has 4 reward components but all operate at the same level (per-action or per-episode). Memory-R1 has a single binary signal. Memory-T1's reward operates at three levels: answer (accuracy), evidence (grounding), and temporal (session + utterance). This gradient of granularity is the architectural insight.

3. **Read-time focus**: Memory-R1 and Mem-alpha both focus on write-time optimization (building better memories). Memory-T1 focuses exclusively on read-time optimization (selecting better evidence from existing memories). Together, the three papers cover write-only (Mem-alpha), write+read (Memory-R1), and read-only (Memory-T1) -- the full spectrum.

4. **Coarse-to-fine retrieval**: The two-phase pipeline (heuristic narrowing + learned selection) is a practical architecture that separates recall from precision. Phase 1 achieves ~90% evidence recall cheaply; Phase 2 uses learned policy for precision. This decomposition is cleaner than Memory-R1's approach where retrieval and distillation are somewhat entangled.

### Where It Agrees with the Other Two Papers

1. **GRPO works**: All three papers use GRPO successfully. Memory-R1 also tested PPO and found comparable results. GRPO's advantage (no value function needed) makes it the practical choice for memory RL.

2. **Small models beat large baselines with RL**: Memory-T1 (3B) beats Qwen2.5-14B. Memory-R1 (8B) beats all baselines. Mem-alpha (4B) beats GPT-4.1-mini. The pattern is fully replicated: RL-trained small models > untrained large models for memory tasks.

3. **Auxiliary rewards help**: Memory-T1's ablation shows R_a alone gives -22.4% (vs. full model with grounding + temporal rewards). Mem-alpha's ablation shows removing the content validity reward drops performance significantly. Memory-R1's simpler single-reward approach works but achieves lower absolute scores. The evidence favors multi-signal rewards.

4. **Contradiction handling remains the gap nobody fills**: Three papers, three explicit or implicit avoidances. This is now strongly replicated.

### Where It Disagrees

1. **Data efficiency**: Memory-R1 uses 152 examples, Mem-alpha 562, Memory-T1 4,065. Memory-T1 uses 27x more training data than Memory-R1, making data-efficiency comparison impossible. Memory-R1's claim of remarkable data efficiency is unchallenged but also unreplicated.

2. **Write vs. read optimization**: Memory-R1 and Mem-alpha invest in write-time operations. Memory-T1 invests entirely in read-time selection. The implicit claim of Memory-T1 is that given fixed dialogue history, the bottleneck is temporal evidence selection, not memory construction. This is domain-specific -- for temporal QA over dialogue, it is plausible; for long-lived agent memory, write-time operations are clearly needed.

3. **Reward weight distribution**: Memory-T1 gives 60% weight to accuracy and 20% each to grounding and temporal. Mem-alpha gives correctness weight 1.0 with compression at 0.05 and content at 0.1. Memory-R1 uses only accuracy. The question of whether auxiliary rewards are complementary (Memory-T1's view), decorative (Memory-R1's implicit view), or essential (Mem-alpha's ablation), remains unresolved.

---

## Standout Feature

**Multi-level temporal reward with session-utterance granularity.** Among the three RL-for-memory papers, Memory-T1 is the only one that treats temporal reasoning as a first-class reward component. The two-level temporal consistency reward (R_s at session level + R_f at utterance level) provides dense supervision at multiple granularities. This addresses the sparse reward problem that plagues RL for memory: a binary "correct answer" signal is insufficient for learning temporal reasoning, but session proximity and utterance fidelity provide gradient even when the final answer is wrong.

The ablation result where removing R_s paradoxically helps simple tasks but destroys intermediate tasks reveals genuine complexity in temporal reasoning -- a finding that purely accuracy-driven approaches (Memory-R1, Mem-alpha) would never surface. Temporal reasoning is not a uniform capability; it has distinct components that interact nonlinearly.

This is also the only paper that demonstrates robustness to context length scaling. Memory-R1 and Mem-alpha do not test this dimension. Memory-T1's stable 65-75% across 0-128k tokens, while baselines collapse by 30+ points, is the strongest practical evidence for temporal-aware retrieval.

---

## Gap Ratings

Rating scale: 0 = not addressed, 1 = minimal, 2 = partial, 3 = solid, 4 = thorough, 5 = state-of-art.

| Capability | Rating | Notes |
|-----------|--------|-------|
| Multi-Channel Retrieval | 2 | BM25 + temporal filter is two-channel but no vector search, no graph traversal. Coarse-to-fine is architecturally interesting but limited in retrieval modalities. |
| Memory Consolidation / Sleep | 0 | No write operations at all. No consolidation, summarization, or abstraction. |
| Contradiction Handling | 0 | Not addressed. Read-only system cannot resolve contradictions in stored memories. |
| Decay / Staleness | 1 | Temporal filtering implicitly handles "staleness" by focusing on relevant time windows, but there is no explicit decay model. Temporal proximity reward is a form of recency bias. |
| Temporal Modeling | 4 | Core strength. Multi-level temporal reward, temporal window prediction, 11 temporal reasoning subtasks. Missing: event-time vs. storage-time distinction ([[agent-output-tremu]]'s key insight). |
| Graph / Relationship | 0 | No inter-memory relationships. Sessions are independent units. |
| Feedback Loops | 2 | The RL reward is a training-time feedback loop. No inference-time feedback, no recall_feedback equivalent, no online adaptation. |
| Proactive Memory | 0 | Purely reactive -- responds to queries, never proactively surfaces relevant memories. |
| Scalability | 3 | Demonstrated stable performance to 128k tokens. Coarse-to-fine pipeline is efficient (1.26s latency). No testing beyond 128k. |
| Human-in-the-Loop | 0 | Fully automated. No human oversight or curation mechanism. |

---

## Relevance to claude-memory

### What This Paper Does That We Don't

1. **Temporal-aware retrieval**: Our `recall()` uses RRF over vector + BM25 with no explicit temporal reasoning. Memory-T1 demonstrates that temporal filtering (predicting query temporal window and discarding sessions outside it) dramatically improves retrieval for time-dependent queries. Questions like "what did we discuss last week about X" would benefit from a temporal filter before vector/BM25 scoring.

2. **Evidence grounding in retrieval**: The R_g reward trains the agent to cite its sources. Our system retrieves memories but does not systematically verify that responses are grounded in specific retrieved memories. Evidence grounding is related to our existing `recall_feedback` -- if we tracked which recalled memories were actually used in responses, we would have a grounding signal.

3. **Multi-level reward design**: The two-level temporal reward (session + utterance) is an example of how to decompose a sparse reward into dense sub-signals. For our `recall_feedback`, this suggests we could decompose "was this memory useful?" into sub-signals: "was the temporal context right?", "was the factual content used?", "was the category appropriate?"

### What We Already Do Better

1. **Write operations**: Memory-T1 has none. Our full `remember()` pipeline with typed memories, priority levels, source tracking, pending review, and auto-capture is categorically more capable.

2. **Memory lifecycle**: Decay, consolidation, dormancy, confidence gradients -- all absent from Memory-T1. Our system manages memory as a living, evolving store; Memory-T1 treats it as a static archive.

3. **Graph structure**: Our edge graph with novelty-scored adjacency expansion addresses inter-memory relationships. Memory-T1 treats sessions as independent units.

4. **Human curation**: Our `/reflect` + pending review + explicit `recall_feedback` provides human-in-the-loop quality control that Memory-T1 lacks entirely.

5. **Retrieval sophistication**: Our RRF fusion (vector + BM25 + feedback boost + Hebbian PMI + shadow penalty + confidence) is substantially more sophisticated than Memory-T1's BM25 + temporal filter.

### Adaptation Potential

The most transferable insight is **temporal-aware candidate filtering before retrieval scoring**. Our `recall()` could add an optional temporal filter: when the query contains temporal language ("last week", "yesterday", "when we discussed X in January"), predict a temporal window and boost or filter memories by timestamp before RRF scoring. This would be a lightweight heuristic version of Memory-T1's Phase 1, without RL training.

---

## Insights Worth Stealing

### 1. Temporal-Aware Retrieval Filter (High value, Medium effort)

**What**: Before applying RRF scoring, predict whether the query has a temporal dimension. If so, estimate a temporal window and boost memories whose timestamps fall within that window. This is Memory-T1's Phase 1 translated to our architecture.

**How to adapt**: In `recall()`, add a lightweight temporal intent detector (simpler than Memory-T1's LLM-based prediction). If temporal intent is detected, compute a timestamp-based boost factor that multiplies into the RRF score. Memories within the predicted window get boosted; memories far outside get penalized. This interacts well with our existing `_detect_intent()` mechanism, which already maps query prefixes (why/when/who/how) to preferred edge types.

**Effort**: Medium. Requires (a) temporal intent detection (could be rule-based: "last week", "yesterday", "in February", "recently"), (b) temporal window estimation, (c) timestamp-based boost in RRF scoring. No LLM call needed -- rule-based temporal parsing is sufficient for our use case.

**Why it ranks first**: Temporal retrieval is a real gap. Questions like "what did we decide about X last session" currently rely on vector + BM25 matching on content, not on temporal proximity. The timestamp data already exists in our memories -- we just don't use it for retrieval scoring.

### 2. Evidence Grounding as Recall Quality Signal (Medium value, Low effort)

**What**: Memory-T1's R_g reward measures whether the agent cites the correct source sessions. Translated to our context: do the recalled memories actually get used in the response? This is a dimension of `recall_feedback` we could track.

**How to adapt**: When providing `recall_feedback`, distinguish between "this memory was used to construct the response" (high grounding) and "this memory was surfaced but not used" (low grounding). The recall_feedback already captures utility, but explicitly separating "used" from "not used" (rather than a 0-1 float) could sharpen the signal. Alternatively, the stop hook could automatically detect which recalled memory IDs appear in the response text.

**Effort**: Low. Minor enhancement to recall_feedback semantics or hook logic.

### 3. Coarse-to-Fine Retrieval Architecture (Medium value, High effort)

**What**: Separate recall from precision. Use a cheap, high-recall first pass (temporal filter + BM25) to generate candidates, then a more expensive, high-precision second pass (learned selection or reranking) to select final results.

**How to adapt**: This is structurally what our staged curated recall already does (Stage 1: standard recall; Stage 2: Sonnet subagent curates). Memory-T1 validates this architecture from a different angle. The specific insight is that Phase 1 should optimize for recall (don't miss relevant memories) while Phase 2 optimizes for precision (don't include irrelevant ones). Our BM25 field weighting and vector search already serve the recall function; the precision function is served by RRF scoring and budget trimming. The gap is that we don't have a temporal dimension in Phase 1.

**Effort**: High for full implementation. The staged curated recall already covers most of this. The marginal gain from further architectural decomposition is small.

### 4. Multi-Level Reward Decomposition for Feedback (Low-Medium value, Low effort)

**What**: Memory-T1's insight that a single accuracy reward is insufficient (-22.4% when used alone) and that decomposing the reward into sub-signals (accuracy + grounding + temporal) provides denser learning signal.

**How to adapt**: Our `recall_feedback` currently takes a single utility float (0-1) and optional durability (-1 to 1). We could decompose utility into sub-dimensions: temporal_relevance (was the time context right?), factual_relevance (was the content used?), category_relevance (was the memory type appropriate?). This would provide richer signal for scoring optimization. However, this increases friction for the user/agent providing feedback.

**Effort**: Low for schema change. The question is whether richer feedback dimensions are worth the additional friction. For now, the single utility float is probably sufficient -- we can always decompose later if scoring experiments reveal the need.

---

## What's Not Worth It

1. **The RL training pipeline**: Same conclusion as Memory-R1 and Mem-alpha. We don't have a training distribution, and our "memory manager" is already a frontier model. The GRPO infrastructure is irrelevant.

2. **Time-Dialog benchmark specifics**: The 11 temporal reasoning subtasks are specific to conversational temporal QA. Our temporal needs are different -- we need "when did we discuss X" and "has this changed since last time", not "how many days between event A and event B in a dialogue."

3. **The specific temporal reward formulas**: The logistic function for R_s with c=1.5, d=0.5, m=7, s=1 is tuned for dialogue sessions with day-level timestamps. Our memory timestamps span weeks to months, and the temporal distribution is entirely different. The principle (temporal proximity matters) transfers; the specific math does not.

4. **BM25-only candidate generation**: Our RRF fusion already includes BM25 alongside vector search. Memory-T1's BM25-only retrieval is a step backward for us.

5. **The structured output format**: `{selected_memory:[...]. answer:...}` is a training artifact. Our system doesn't need structured evidence citation because the recall results are already separated from the response generation.

6. **Context length robustness testing**: Impressive but irrelevant. Our memories are stored as discrete entries with metadata, not as raw dialogue sessions. The "context length" problem doesn't apply to our architecture in the same way.

---

## Key Takeaway

Memory-T1 is the temporal reasoning specialist in the RL-for-memory trilogy. Where Memory-R1 and Mem-alpha focus on *building* memories through write operations, Memory-T1 focuses on *selecting* evidence through temporal-aware retrieval. Its core contribution -- a multi-level temporal consistency reward operating at both session and utterance granularity -- demonstrates that temporal reasoning is a distinct capability requiring dedicated training signals, not just a side effect of answer-accuracy optimization. The -22.4% drop when using accuracy alone, combined with the paradoxical interaction between session-level and utterance-level temporal rewards across difficulty categories, reveals genuine complexity in temporal reasoning that the other two papers ignore. For claude-memory, the transferable insight is temporal-aware retrieval filtering: our `recall()` has rich scoring (vector + BM25 + feedback + Hebbian + shadow + confidence) but no temporal dimension, and Memory-T1 demonstrates that temporal filtering alone can maintain 65-75% F1 where baselines without it collapse to 40%. Adding a timestamp-based boost to our RRF scoring -- a lightweight heuristic version of Memory-T1's Phase 1 -- would address a real retrieval gap without requiring any RL training.

---

## Impact on Implementation Priority

| Priority | Change | Rationale |
|----------|--------|-----------|
| #1 RRF fusion + enriched keys | **Enriched** | Memory-T1 demonstrates that temporal filtering is a retrieval channel worth adding. A timestamp-based boost in RRF scoring would be a natural extension. Not a new priority -- an enhancement to #1. |
| #2 Relationship edges | Unchanged | Not addressed by Memory-T1. |
| #3 Graded contradiction detection | **Further strengthened** | Three independent RL-for-memory papers now all avoid contradiction handling. The pattern is fully replicated: this is hard, underserved, and differentiating. |
| #4 Decay floor + power-law | Unchanged | Memory-T1's temporal proximity reward is not a decay model but validates that recency matters for retrieval scoring. Our decay model addresses this at the lifecycle level rather than the retrieval level. |
| #5 Sleep skill (consolidation) | Unchanged | Memory-T1 has no consolidation. Write-only vs. read-only specialization means no lessons for batch consolidation. |
| #6 Mutation log | Unchanged | No audit trail in Memory-T1. |
| #7 Confidence scores | Unchanged | Not addressed. |
| #8-11 | Unchanged | Not affected. |
| #12 Reading strategy optimization | **Reinforced** | Memory-T1's entire contribution is read-time optimization. The coarse-to-fine architecture and evidence grounding reward validate that how you read memory matters independently of what you store. Our staged curated recall addresses this; Memory-T1 confirms the architectural direction. |
| #13 Entity resolution | Unchanged | Not addressed. |
| #14 Retrieval feedback loop | **Maintained at "strongly reinforced"** | Memory-T1's grounding reward (R_g) is another variant of task-performance feedback. Three papers now all validate that task-performance signal -- whether as RL reward, grounding metric, or retrieval feedback -- is the most important driver of memory quality. The convergence across three independent papers with different architectures, rewards, and benchmarks makes this the most robustly validated insight in the RL-for-memory literature. |

**New consideration**: Temporal boost in RRF scoring. When a query contains temporal language, boost memories by timestamp proximity. This is a lightweight addition to the existing RRF pipeline, not a new priority item. Implementation: extend `_detect_intent()` to detect temporal queries, compute a timestamp-based boost factor, and incorporate it into the post-RRF scoring alongside feedback, Hebbian, shadow, and confidence components.

---

## Cross-Reference: The RL-for-Memory Trilogy

Three independent papers, three architectures, one converging conclusion:

| Paper | Focus | Operations | Temporal | Reward | Key Insight |
|-------|-------|-----------|----------|--------|-------------|
| [[agent-output-mem-alpha]] | Write policy | INSERT/UPDATE/DELETE | None | 4-signal (correctness dominant) | Compression pressure + content validity |
| [[agent-output-memory-r1]] | Write + read | ADD/UPDATE/DELETE/NOOP | None | Single (EM) | Memory distillation at read time |
| Memory-T1 | Read policy | Selection only | Core focus | 3-signal (multi-level temporal) | Dense temporal reward at two granularities |

**Convergent findings** (now replicated across 3 papers):
1. GRPO is the practical RL algorithm for memory tasks
2. RL-trained small models beat untrained large models
3. Task-performance signal is the dominant driver of memory quality
4. Contradiction handling is universally avoided
5. Multi-signal rewards generally outperform single-signal rewards

**Divergent findings** (interesting open questions):
1. Write-time vs. read-time optimization -- which matters more? Memory-R1 does both; Mem-alpha and Memory-T1 specialize. No head-to-head comparison exists.
2. Data efficiency -- 152 (R1) vs 562 (alpha) vs 4,065 (T1). Is Memory-R1's efficiency real or undertested?
3. Reward weight distribution -- remains unresolved. Each paper uses different weights.

**The gap that remains**: All three papers solve clean, well-defined subtasks (write a memory bank, select temporal evidence, filter candidates). None addresses the integrated long-term memory problem: write + consolidate + decay + contradict + retrieve + ground + update beliefs over time. Our system is the only one attempting the full lifecycle.

---

## See Also

- [[agent-output-memory-r1]] -- Two-agent RL framework for memory construction + distillation
- [[agent-output-mem-alpha]] -- RL-trained memory construction with compression reward
- [[agent-output-tremu]] -- Temporal reasoning framework (event-time vs. storage-time distinction that Memory-T1 lacks)
- [[agent-output-locomo-plus]] -- LoCoMo benchmark with cognitive memory and PPR/graph traversal
- [[agent-output-beam]] -- 10M-token benchmark where temporal reasoning is one of 10 evaluated abilities
- [[agent-output-ragas]] -- Decompose-then-verify evaluation pattern relevant to evidence grounding
