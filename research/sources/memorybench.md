# MemoryBench: A Benchmark for Memory and Continual Learning in LLM Systems -- Analysis

*Phase 15, 2026-03-06. Analysis of arXiv:2510.17281 (2025).*

---

## 1. Paper Overview

**Paper**: Qingyao Ai, Yichen Tang, Changyue Wang, Jianming Long, Weihang Su, Yiqun Liu. Tsinghua University (Department of Computer Science and Technology, Institute for AI). arXiv:2510.17281. v1 October 20 2025, v4 December 12 2025. No confirmed venue yet.

**Problem**: Existing benchmarks for LLM memory systems focus almost exclusively on reading comprehension with long-form inputs -- testing whether systems can retrieve facts from extended contexts. They ignore a critical dimension: whether LLM systems can *learn from accumulated user feedback at service time* to improve their performance over successive interactions. The paper frames this gap in terms of the declarative/procedural memory distinction from cognitive science: current benchmarks test declarative memory (storing and retrieving facts) but never test procedural memory (learning *how to do things better* from performance feedback).

**Core contribution**: MemoryBench, a benchmark and evaluation framework with three modules (Task Provider, User Simulator, Performance Monitor) spanning 11 datasets, 3 domains, 4 task formats, 2 languages, and ~20,000 cases. The central innovation is the **User Simulator** -- an LLM-as-user paradigm that generates realistic feedback (verbose critiques, like/dislike actions, copy behavior) on system outputs, creating procedural memory resources that systems should exploit to improve. The main finding: state-of-the-art memory systems (A-Mem, Mem0, MemoryOS) cannot consistently outperform naive RAG baselines on this broader evaluation, exposing their narrow optimization for long-context reading comprehension.

**Scale**: ~20,000 cases across 11 datasets. 4:1 train/test split. Training portion used for feedback generation; test portion for evaluation. Off-policy and on-policy evaluation protocols.

**Code/Data**: Open-sourced (data, processing scripts, evaluation pipelines, baseline implementations).

---

## 2. Benchmark Design

### The Declarative/Procedural Distinction

This is the paper's organizing principle. Drawing from cognitive science:

- **Declarative memory**: Factual knowledge. Split into semantic (user-independent facts) and episodic (user-dependent facts from conversation history). This is what existing benchmarks test.
- **Procedural memory**: Non-factual knowledge about task execution -- workflow preferences, reward signals, solution strategies. The paper argues user feedback logs are the primary resource for procedural memory in LLM systems.

### Task Format Taxonomy

The paper introduces a 2x2 task format grid based on input/output length, arguing that existing benchmarks cluster entirely in one cell:

| Format | Input | Output | Existing Benchmark Coverage | Example |
|--------|-------|--------|----------------------------|---------|
| **LiSo** | Long (600+ tokens) | Short | Nearly all prior work | Reading comprehension, QA over conversations |
| **SiLo** | Short (<600 tokens) | Long | Rare | Creative writing from short prompts |
| **LiLo** | Long | Long | Very rare | Long-form analysis/synthesis |
| **SiSo** | Short | Short | Moderate | Classification, categorization |

This is a legitimate critique. [[agent-output-beam|BEAM]], [[agent-output-locomo-plus|LoCoMo-Plus]], and [[agent-output-memory-arena|MemoryArena]] all use LiSo-dominated evaluation. Systems optimized for "find the answer in this long context" may not generalize.

### 11 Datasets

| Dataset | Domain | Language | Format | Cases |
|---------|--------|----------|--------|-------|
| Locomo | Open | English | LiSo | 1,986 |
| DialSim-Friends | Open | English | LiSo | 1,000 |
| DialSim-BigBang | Open | English | LiSo | 1,000 |
| DialSim-TheOffice | Open | English | LiSo | 1,000 |
| LexEval-Summarization | Legal | Chinese | LiSo | 1,000 |
| IdeaBench | Academic | English | LiSo | 2,374 |
| LimitGen-Syn | Academic | English | LiSo | 1,000 |
| WritingPrompts | Open | English | LiLo | 2,000 |
| JuDGE | Legal | Chinese | SiLo | 2,505 |
| HelloBench | Academic/Open | English | SiLo/SiSo | 943 |
| WritingBench | Multi-domain | English/Chinese | LiLo/SiLo | 790 |
| NF-Cats | Open | English | SiSo | 2,397 |
| SciTechNews | Academic | English | SiSo | 1,000 |

Three domains (open, legal, academic), two languages (English, Chinese), all four format cells covered. The dataset selection is heterogeneous -- which is the point, but it also means no single dataset deeply tests any one memory competency.

### Three-Module Architecture

1. **Task Provider**: Supplies queries (q), context (c), and evaluation metadata (v). Manages the 4:1 train/test partition and retrieves pre-generated feedback logs for off-policy evaluation.

2. **User Simulator**: The novel contribution. Two paths:
   - *Metric-based path* (for verifiable tasks like Locomo, DialSim): Objective metrics (F1, accuracy) are mapped deterministically to a 1-10 satisfaction scale.
   - *LLM-as-user path* (for open-ended tasks): A strong LLM (Qwen-32B) is prompted with user persona, domain expertise, and evaluation criteria, producing JSON output with reasoning, verbal feedback, and satisfaction score.

   Satisfaction scores are then mapped to *actions* via sigmoid probability functions: P(like), P(dislike), P(copy). Calibrated against empirical user behavior data (5.59% like rate, 0.91% dislike rate globally).

3. **Performance Monitor**: Evaluates test performance using LLM-as-Judge (DeepSeek-V3 or WritingBench-Critic). Min-max or z-score normalization within each dataset before cross-dataset averaging.

### Evaluation Protocols

- **Off-policy**: Run backbone LLM once on all training cases, generate one feedback session per case, accumulate all feedback, then test. Simpler and reproducible, but doesn't capture iterative learning.
- **On-policy**: Randomly sample batches each timestep, collect fresh feedback after each batch, system updates before next step. More realistic but computationally expensive (only feasible for fast baselines).

### RAG Baseline Variants

- **BM25-S / Embed-S**: Whole dialog sessions stored as single documents.
- **BM25-M / Embed-M**: Individual messages stored as separate documents.

This session-vs-message chunking comparison is useful -- it mirrors a real design choice in any memory system.

---

## 3. Key Claims and Evidence

### Claim 1: LLM-as-User simulation produces feedback indistinguishable from human feedback.

**Evidence**: Human annotation study where annotators could not reliably distinguish simulated from human-written feedback. The satisfaction score distribution is reported: score 9 (41.05%), score 8 (32.12%), with minimal low scores. The action probability calibration matches empirical user behavior rates.

**Assessment**: Moderate confidence. The "indistinguishable" claim is strong but the validation methodology is thin in detail -- no inter-annotator agreement numbers are reported in the main text, and the satisfaction distribution being heavily skewed toward high scores (73% at 8-9) suggests the backbone LLM is generally competent at these tasks, which limits how much procedural feedback can teach.

### Claim 2: State-of-the-art memory systems cannot consistently outperform naive RAG baselines.

**Evidence**: Across all domain/format partitions, none of A-Mem, Mem0, or MemoryOS consistently beats BM25 or embedding RAG. On SiLo tasks specifically, vanilla LLM (no memory at all) outperforms all memory-augmented systems.

**Assessment**: High confidence -- this is a clean finding with clear implications. But the explanation matters: these systems were designed for reading comprehension (LiSo) and are being evaluated on generation tasks (SiLo, LiLo, SiSo) for which their architectures are not suited. This is less "memory systems fail" and more "memory systems designed for retrieval fail on generation tasks." The paper acknowledges this somewhat but could be clearer about the distinction.

### Claim 3: Memory systems treat all inputs as declarative memory, failing to exploit procedural memory.

**Evidence**: The paper observes that A-Mem, Mem0, and MemoryOS "simply treat all inputs as declarative memory" -- they store facts but don't extract actionable patterns from feedback about their own performance. When given verbose user feedback ("your summary missed the key legal precedent"), these systems store the feedback as a fact rather than adjusting their approach.

**Assessment**: This is the paper's most important insight. It identifies a genuine architectural gap. However, the benchmark itself doesn't cleanly isolate this -- it bundles procedural memory failure with task-format generalization failure, making it hard to attribute how much of the underperformance comes from each source.

### Claim 4: Efficiency is a major concern for existing memory systems.

**Evidence**: MemoryOS requires 17+ seconds per case for memory construction. Mem0 has highly inconsistent timing across task formats. A-Mem is efficient but ineffective.

**Assessment**: Useful practical data. For any production system, 17 seconds per memory write is prohibitive.

### Claim 5: Performance instability in domain-specific tasks.

**Evidence**: Stepwise experiments on academic and legal domains show "significant fluctuation during the training process" -- systems that improve after some feedback batches regress after others, particularly on specialized domains.

**Assessment**: This suggests feedback filtering is critical -- not all feedback is useful, and domain-specific feedback may be noisier. This connects directly to our quality floor and feedback scoring mechanisms.

---

## 4. Standout Feature

**The User Simulator as procedural memory generator.** This is what distinguishes MemoryBench from every benchmark in our prior survey. All seven previously analyzed benchmarks ([[agent-output-beam|BEAM]], [[agent-output-locomo-plus|LoCoMo-Plus]], [[agent-output-tremu|TReMu]], [[agent-output-memory-arena|MemoryArena]], [[agent-output-amemgym|AMemGym]], [[agent-output-ama-bench|AMA-Bench]], [[agent-output-memory-t1|Memory-T1]]) test whether a system can *retrieve stored information*. MemoryBench tests whether a system can *improve from feedback about its own outputs* -- a fundamentally different capability.

The feedback taxonomy (verbose critiques, action signals, implicit behavior) and the probabilistic action model (calibrated against empirical user behavior rates) create a realistic simulation of how users actually interact with LLM systems. The two-path simulation approach (metric-based for verifiable tasks, LLM-as-user for open-ended tasks) is pragmatic and well-designed.

This is the first benchmark where `recall_feedback()` is directly tested as a capability -- not just as a feature of memory systems, but as a core competency.

However, there is a crucial distinction: MemoryBench's "procedural memory" is about learning from user feedback on *different tasks of the same type*, not about improving recall quality on the same stored knowledge. Our `recall_feedback()` does the latter -- it scores the usefulness of retrieved memories to refine future retrieval ranking. MemoryBench's feedback is about learning writing style preferences, legal reasoning patterns, or summarization quality from user critiques. These are complementary but different.

---

## 5. Competency Coverage Ratings

| Competency | Coverage | Justification |
|-----------|----------|---------------|
| **Information Retrieval** | 45% | RAG baselines test basic retrieval, but the benchmark is not designed to stress-test retrieval quality. The focus is on whether retrieved information (including feedback) improves generation. |
| **Multi-Session Reasoning** | 30% | Off-policy evaluation accumulates feedback across training cases, but there is no explicit test of reasoning across sessions. The on-policy protocol gets closer, but most results are off-policy. |
| **Knowledge Update / Contradiction** | 5% | Not addressed. No tasks test whether systems update beliefs when feedback contradicts prior stored knowledge. The paper never discusses contradiction resolution. |
| **Temporal Reasoning** | 5% | Not addressed. No temporal reasoning tasks. The stepwise training protocol has temporal structure (feedback accumulates over time), but this is not tested as a competency. |
| **Abstention / Confidence** | 0% | Not tested. No tasks where the correct answer is "I don't know" or where confidence calibration matters. |
| **Write-Path Behavior** | 60% | Indirectly tested. If a memory system cannot effectively write/organize feedback into useful memory, it will underperform on test cases. The finding that systems "treat all inputs as declarative memory" is a write-path diagnosis. But write quality is not directly measured -- it is inferred from downstream task performance. |
| **Consolidation Quality** | 15% | The stepwise/incremental training protocol touches on consolidation (how well does accumulated feedback integrate over time?), but consolidation is not measured independently. |
| **Proactive / Contextual Recall** | 20% | Tested indirectly -- systems should proactively retrieve relevant feedback when processing a new task. But the benchmark doesn't isolate proactive recall as a capability. |
| **Relationship / Graph Reasoning** | 0% | Not discussed. No graph-based or relational memory approaches tested. No tasks requiring relationship inference. |
| **Agentic Task Performance** | 25% | The on-policy protocol has agentic characteristics (system acts, receives feedback, updates, acts again). But the tasks themselves are not agentic -- they are standard NLP tasks (QA, summarization, writing). |

**Aggregate**: ~20%. MemoryBench covers a genuinely novel dimension (procedural memory / feedback utilization) that no other benchmark in our survey tests. But it has minimal coverage of most traditional memory competencies. It is complementary to, not competitive with, benchmarks like BEAM, LoCoMo-Plus, and MemoryArena.

---

## 6. Relevance to claude-memory

### Could we run against this?

Partially. The benchmark is designed for end-to-end LLM systems, not standalone memory modules. To evaluate claude-memory against MemoryBench, we would need:

1. **Wrapper**: An LLM system that uses claude-memory as its memory backend, processes MemoryBench tasks, receives simulated feedback, and stores it using `remember()`.
2. **Feedback-to-memory mapping**: The key challenge. MemoryBench provides verbose feedback ("your summary missed X, the legal precedent Y was important"). We would need to decide: store this as-is? Extract actionable patterns? Feed it through `recall_feedback()` somehow?
3. **Retrieval integration**: When processing test cases, the system would call `recall()` to retrieve relevant prior feedback, then condition its response on that feedback.

The adaptation is non-trivial but illuminating. It would test whether our RRF hybrid search, Hebbian PMI boosting, and feedback scoring can surface the *right* prior feedback for a new task.

### What would it reveal?

1. **Feedback utilization gap**: Our `recall_feedback()` scores the utility of retrieved memories for future retrieval ranking. MemoryBench tests whether the *content* of feedback can improve task performance. These are different loops. We have the retrieval-quality loop but may be missing the task-quality loop.

2. **Cross-task transfer**: Can claude-memory retrieve feedback from task A that is relevant to task B? Our edge graph and adjacency expansion might help here -- if feedback on one legal summarization task connects to another via edges, that is cross-task procedural memory.

3. **Write-path quality**: How well does `remember()` capture the actionable content of verbose feedback? If a user says "your summary was too focused on procedural details, I need the substantive legal reasoning," does that become a memory that is useful when retrieved for a similar future task?

4. **Domain stability**: The performance instability finding on academic/legal domains connects to our quality floor and confidence gradient. Would our feedback scoring prevent the oscillation that MemoryBench systems exhibit?

### Adaptation effort

Medium. The main work is building the wrapper and the feedback-to-memory mapping. The benchmark itself is well-structured and open-sourced. Estimated: 2-3 days to build a minimal evaluation harness, 1 week for thorough evaluation across all partitions.

---

## 7. Insights Worth Stealing

### Rank 1: Task format diversity as evaluation dimension (Impact: High, Effort: Low)

The 2x2 task format grid (LiSo/SiLo/LiLo/SiSo) is a clean and revealing evaluation axis. Systems optimized for LiSo (most existing benchmarks) may fail on SiLo or LiLo. If we build our own evaluation suite, we should include all four cells. This is a design principle, not an implementation task.

### Rank 2: User feedback as procedural memory resource (Impact: High, Effort: Medium)

The distinction between declarative and procedural memory is worth internalizing. Our system currently treats all memories as declarative -- facts, decisions, corrections, reflections. The procedural memory concept -- storing *how to do things better* extracted from feedback -- suggests a potential new memory category or handling pathway. Not "user said X was wrong" (which we store as corrections) but "pattern: when summarizing legal documents, user consistently wants substantive reasoning over procedural details."

This connects to our theme normalization and edge graph. A cluster of feedback memories about "legal summarization style" connected by edges could function as procedural memory without a new mechanism.

### Rank 3: Sigmoid action probability model (Impact: Medium, Effort: Low)

The calibrated mapping from satisfaction scores to user actions (like/dislike/copy) using sigmoid functions is a clean, principled approach. If we ever build user simulation for testing, this model is directly reusable. The empirical calibration against real user behavior rates (5.59% like, 0.91% dislike) grounds it in reality.

### Rank 4: Session-vs-message chunking comparison (Impact: Medium, Effort: Low)

The BM25-S vs BM25-M and Embed-S vs Embed-M comparison -- storing whole sessions vs individual messages -- directly maps to our memory granularity decisions. We currently store individual memories (closer to the -M variants). Understanding how session-level vs message-level chunking affects retrieval quality on different task types could inform `remember()` design.

### Rank 5: Feedback filtering for domain stability (Impact: Medium, Effort: Medium)

The performance instability on specialized domains suggests that not all feedback is useful -- some feedback may be noisy, contradictory, or domain-inappropriate. Our quality floor, confidence gradient, and recall_feedback scoring are exactly the mechanisms that should address this. MemoryBench provides a testbed for validating that these mechanisms actually prevent oscillation.

---

## 8. What's Not Worth It

- **Chinese language evaluation**: Two of eleven datasets are Chinese. The language dimension is orthogonal to memory architecture evaluation for our purposes. Skip unless pursuing multilingual deployment.

- **SFT baseline replication**: The supervised fine-tuning baseline (training model weights on feedback) is architecturally irrelevant to us. claude-memory is non-parametric -- we do not modify model weights. The comparison is interesting conceptually (parametric vs non-parametric continual learning) but not actionable.

- **LLM-as-Judge calibration**: Their judge pipeline (DeepSeek-V3, WritingBench-Critic) is specific to their task selection. If we adapt the benchmark, we would use our own evaluation methodology.

- **Full on-policy evaluation**: Computationally expensive (the paper says it takes "days to weeks" for some methods) and the off-policy results tell the same story. Off-policy is sufficient for diagnostic evaluation.

- **Replicating all 11 datasets**: Most value comes from testing across the 4 format cells (LiSo/SiLo/LiLo/SiSo). One representative dataset per cell is sufficient for architectural diagnosis.

---

## 9. Key Takeaway

MemoryBench identifies a genuine blind spot in both memory system design and memory system evaluation: **procedural memory** -- the ability to learn task execution strategies from accumulated user feedback, not just retrieve stored facts. Its central finding -- that A-Mem, Mem0, and MemoryOS "simply treat all inputs as declarative memory" and cannot consistently outperform naive RAG -- is damning but unsurprising once the benchmark expands beyond reading comprehension. For claude-memory, the most important implication is conceptual rather than technical: our `recall_feedback()` optimizes *retrieval quality* (which memories are useful to surface), but we have no mechanism for *procedural memory* -- extracting generalizable task-execution patterns from feedback and applying them to novel tasks. The edge graph and theme clustering could serve this function if feedback memories are connected to each other and to the tasks they inform, but this pathway is not yet built. MemoryBench also validates our instinct that feedback filtering matters: unfiltered feedback causes performance oscillation, which is exactly what our quality floor, confidence gradient, and decay mechanisms are designed to prevent.

---

## 10. Impact on Implementation Priority

**Priority #3 (Temporal reasoning)**: No impact. MemoryBench does not test temporal reasoning.

**Priority #5 (PPR / graph traversal)**: Mild positive. MemoryBench doesn't test graph reasoning, but the procedural memory challenge -- connecting related feedback across tasks -- is a use case for graph-based memory. If feedback on legal summarization task A is connected via edges to legal summarization task B, PPR could surface relevant procedural knowledge. This strengthens the case for PPR as a general mechanism, not just for multi-hop factual queries.

**Priority #8 (Proactive recall)**: Moderate positive. MemoryBench's procedural memory framing implies that good systems should proactively recall relevant feedback when encountering a similar task type, without being explicitly asked. This is proactive recall applied to the feedback domain.

**Priority #11 (Event-time vs storage-time)**: No impact. Not tested.

**Priority #14 (On-policy evaluation)**: Strong positive. MemoryBench's on-policy protocol -- iterative feedback collection and memory updating -- is the most realistic evaluation of continual learning. Our current evaluation is entirely off-policy (retrospective scoring). Building on-policy evaluation capability would let us test whether `recall_feedback()` and the confidence gradient actually improve performance over time.

**New consideration**: The declarative/procedural distinction suggests a potential new priority: **procedural memory extraction** -- a pipeline that takes verbose user feedback and extracts generalizable patterns, storing them as a distinct memory type (or with distinct themes/edges). This would sit between `remember()` and `recall()` -- a feedback-to-pattern extraction step. Not high enough priority to displace existing items, but worth noting as a future direction.

---

## 11. See Also

- [[agent-output-amemgym|AMemGym]] -- Also tests on-policy evaluation (memory system participates in conversation). Complementary: AMemGym tests write/read/utilization decomposition; MemoryBench tests procedural memory from feedback. Both find that existing systems fail at capabilities beyond simple retrieval.
- [[agent-output-memory-arena|MemoryArena]] -- Tests write-path behavior directly. MemoryBench tests it indirectly through downstream task performance. MemoryArena's POMDP formalization complements MemoryBench's continual learning formalization.
- [[agent-output-ama-bench|AMA-Bench]] -- Tests on agentic trajectories. MemoryBench tests on NLP tasks with feedback. Both find memory systems trail simpler baselines. AMA-Bench's needle protocol cleanly separates write vs read failures; MemoryBench bundles them.
- [[agent-output-beam|BEAM]] -- Tests at massive scale (up to 10M tokens) but only LiSo tasks. MemoryBench's task format diversity (SiLo, LiLo, SiSo) fills BEAM's gaps but at much smaller scale.
- [[agent-output-locomo-plus|LoCoMo-Plus]] -- MemoryBench includes Locomo as one of its 11 datasets. On Locomo specifically, existing memory systems "indeed outperformed naive RAG baselines" -- consistent with LoCoMo-Plus's findings. The degradation comes when expanding beyond this familiar territory.
- [[agent-output-a-mem|A-Mem]], [[agent-output-mem0-paper|Mem0]] -- Both tested as baselines and found wanting. A-Mem is efficient but ineffective; Mem0 has inconsistent timing. Their architectural assumption of declarative-only memory is the core limitation MemoryBench exposes.
- [[agent-output-ragas|RAGAS]] -- MemoryBench's LLM-as-Judge approach is in the same family as RAGAS's evaluation methodology, but applied to diverse task types rather than just retrieval faithfulness.
