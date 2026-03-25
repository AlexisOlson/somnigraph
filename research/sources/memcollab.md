# MemCollab: Cross-Agent Memory Collaboration via Contrastive Trajectory Distillation -- Analysis

*Generated 2026-03-24 by Opus 4.6 agent reading arXiv:2603.23234*

---

## Paper Overview

**Paper**: Yurui Chang, Yiran Wu, Qingyun Wu, Lu Lin (2026). Pennsylvania State University + AG2AI. arXiv:2603.23234v1, March 25, 2026. 18 pages (8 main + 10 appendix). No code release mentioned.

**Problem addressed**: Existing agent memory systems are per-agent -- the memory is constructed from one model's trajectories and reused by that same model. In heterogeneous deployments (multiple model sizes or architectures), naively transferring memory from one model to another *degrades* performance because the memory encodes agent-specific biases (stylistic preferences, heuristic shortcuts) alongside the useful reasoning patterns. The paper asks: can a single memory system serve multiple agents with different capabilities?

**Core claim**: By contrasting reasoning trajectories from two different agents on the same task, you can distill "normative reasoning constraints" (what to enforce, what to avoid) that are agent-agnostic. These constraints generalize across model sizes and model families, improving both weaker and stronger agents while reducing inference-time reasoning turns.

**Scale of evaluation**: 4 benchmarks (MATH500, GSM8K, MBPP, HumanEval), 3 backbone models (Qwen2.5-7B-Instruct, Qwen2.5-32B-Instruct, LLaMA-3-8B-Instruct), 6 baselines (vanilla, Buffer of Thoughts, Dynamic Cheatsheet, w/ Memory from same model, w/ Memory from other model, w/ Self-Contrast Memory). 1000 training instances per dataset, 500 evaluation instances, top-3 retrieval with TF-IDF similarity within task-filtered subset. All experiments on 8x NVIDIA RTX 6000 GPUs.

---

## Architecture / Method

### Core Insight

Raw reasoning trajectories are a mixture of two signals: task-relevant reasoning structure (`s`) and agent-specific bias (`b`). Transferring memory directly transfers both. MemCollab separates them by contrasting a correct trajectory against an incorrect one from a different agent on the same problem, producing abstract "reasoning constraints" that capture only the structural invariants. This is essentially contrastive learning applied to LLM reasoning traces rather than to embeddings.

### Two-Stage Pipeline

**Stage 1: Contrast-Derived Memory Construction** (Algorithm 1)

For each training task instance `x`:
1. Generate trajectories from both agents: `tau_w = A_w(x)` (weak), `tau_s = A_s(x)` (strong)
2. Determine which is correct via an indicator function `I(.)` (checks final answer/code execution)
3. Assign preferred `tau+` and unpreferred `tau-` based on correctness (the preferred trajectory can come from *either* agent)
4. Use the stronger model `f_s` to summarize the contrastive pair into up to `K` memory entries

Each memory entry takes the form: `m_k = (enforce i_k; avoid v_k)` where `i_k` is a reasoning invariant from `tau+` and `v_k` is the corresponding violation pattern from `tau-`.

The discrepancy operator is defined as:

```
Delta(tau-, tau+) = {(v_k, i_k)}_{k=1}^{K}
```

The resulting memory bank stores *normative, abstract reasoning constraints* -- not raw trajectories, demonstrations, or solutions.

**Stage 2: Task-Aware Memory Retrieval and Inference** (Algorithm 2)

At inference time:
1. **Task classification**: The agent's base LLM classifies the new query `q` into a task category and subcategory `(c_q, u_q)` (detailed prompts in Appendix G -- the taxonomy is task-domain-specific, e.g., "Prealgebra > Distance & Midpoints")
2. **Filtered retrieval**: Only memories tagged with matching `(c_m, u_m) = (c_q, u_q)` are considered; top-p by TF-IDF similarity within that subset
3. **Memory-guided inference**: The agent reasons conditioned on retrieved constraints: `y = A(q, M_q)`

The retrieval budget is p=3 (top-3 memories), each contributing up to K=3 extracted constraints (so up to 9 constraints per query). Figure 4 shows performance peaks at p=3 and degrades beyond -- too many constraints introduce noise and attention dispersion.

### Design Choices

- **Contrastive over accumulative**: The paper's central argument against systems like Buffer of Thoughts and Dynamic Cheatsheet. Those accumulate from single-agent traces; MemCollab contrasts across agents. The theoretical framing (Appendix D) draws parallels to InfoNCE loss, arguing that contrastive distillation selects for the task-invariant component `s` while suppressing the agent-specific bias `b`.
- **Task-level classification for retrieval**: Rather than fine-grained embedding similarity, MemCollab uses coarse category labels computed by the agent LLM itself. This is a deliberate choice to reduce interference from task-irrelevant constraints. The JSD analysis (Figure 5) validates this -- algebraically related math categories cluster tightly in error-type space, while number theory and counting/probability diverge.
- **Memory entries are constraints, not demonstrations**: The format "When X, enforce Y; avoid Z" is designed for cross-agent transfer. No code snippets, no worked solutions, no agent-specific reasoning chains.

### Acknowledged Limitations

The paper is notably thin on limitations. The only explicit future work mention is extending to multi-agent (>2) settings. Missing from the discussion:

- The training phase requires both agents to solve every problem, doubling compute at construction time
- The approach requires ground-truth verification (answer/code correctness), which limits applicability to domains where outputs are verifiable
- The task taxonomy is hand-designed per domain (7 MATH categories, each with ~8-15 subcategories)
- No analysis of memory bank size growth or maintenance
- No evaluation on conversational or episodic memory tasks -- only stateless reasoning benchmarks

---

## Key Claims & Evidence

### Table 1: Single Model Family (Qwen-2.5)

| Backbone | Method | MATH500 | GSM8K | MBPP | HumanEval | Avg |
|----------|--------|---------|-------|------|-----------|-----|
| Qwen-2.5-7B | Vanilla | 52.2 | 85.4 | 47.9 | 42.7 | 57.1 |
| | Buffer of Thoughts | 45.8 | 86.4 | **57.6** | 62.2 | 63.0 |
| | Dynamic Cheatsheet | 66.0 | 83.8 | 36.7 | 40.7 | 56.8 |
| | w/ Memory (Self) | 60.0 | 86.2 | 50.2 | 42.7 | 59.8 |
| | w/ Memory (Qwen-32B) | 50.6 | 86.6 | 48.6 | 34.1 | 55.0 |
| | w/ Self-Contrast | 58.4 | 85.6 | 52.5 | 73.3 | 67.5 |
| | **MemCollab** | **67.0** | **87.4** | **57.6** | **74.4** | **71.6** |
| Qwen-2.5-32B | Vanilla | 63.8 | 93.0 | 58.0 | 68.3 | 70.8 |
| | Buffer of Thoughts | 68.2 | 93.0 | 63.8 | 84.1 | 77.3 |
| | Dynamic Cheatsheet | 73.4 | 90.8 | 57.6 | 64.6 | 71.6 |
| | w/ Memory (32B) | 69.6 | 93.2 | 60.3 | 79.3 | 75.6 |
| | w/ Self-Contrast | 69.6 | 93.4 | 58.7 | **87.8** | 77.4 |
| | **MemCollab** | **73.8** | **93.6** | **64.3** | 86.6 | **79.6** |

### Table 2: Cross-Family (LLaMA-3-8B + Qwen-2.5-32B)

| Backbone | Method | MATH500 | GSM8K | MBPP | HumanEval | Avg |
|----------|--------|---------|-------|------|-----------|-----|
| LLaMA-3-8B | Vanilla | 27.4 | 73.0 | 37.0 | 29.3 | 41.7 |
| | w/ Memory (32B) | 18.8 | 56.6 | 35.8 | 34.8 | 36.3 |
| | **MemCollab** | **42.4** | **74.4** | **49.8** | **48.8** | **53.9** |
| Qwen-2.5-32B | Vanilla | 63.8 | 93.0 | 58.0 | 68.3 | 70.8 |
| | w/ Memory (32B) | 69.6 | 93.2 | **60.3** | 79.3 | 75.6 |
| | **MemCollab** | **70.6** | **95.2** | **60.3** | **86.6** | **78.2** |

### Table 3: Inference Efficiency (Qwen-2.5-7B)

| Dataset | Vanilla turns | MemCollab turns |
|---------|--------------|-----------------|
| MATH500 | 2.7 | 2.2 |
| GSM8K | 1.8 | 1.6 |
| MBPP | 3.1 | 1.4 |
| HumanEval | 3.3 | 1.5 |

### Key result: naive transfer hurts

The most important finding for Somnigraph: `w/ Memory (Qwen-32B)` applied to the 7B model *degrades* performance vs. vanilla on MATH500 (50.6 vs. 52.2) and HumanEval (34.1 vs. 42.7). Memory built from a different model is worse than no memory, confirming that raw trajectory distillation couples reasoning with agent bias.

### Methodological Strengths

- Clean experimental design with well-chosen baselines (memory-source ablation isolates the contrast effect from the memory effect)
- The self-contrast baseline (contrasting the model's own sampled trajectories) is a valuable control -- it partially works but underperforms cross-agent contrast
- JSD analysis of error-type distributions across task categories (Figure 5) provides empirical grounding for the task-aware retrieval design
- Consistent gains across both model families and both directions (7B benefits, 32B benefits)

### Methodological Weaknesses

- **Retrieval is primitive**: TF-IDF similarity within task-filtered subsets. No embedding retrieval, no reranking. The task classifier is doing most of the heavy lifting -- the "retrieval" is really just classification + top-3 within a small bucket.
- **No statistical significance**: Raw accuracy numbers with no confidence intervals, variance, or significance tests across the 500-instance eval sets.
- **Training data overlap risk**: Training on 1000 instances from the same datasets used for evaluation (different split, but same distribution). The paper doesn't discuss this.
- **LLM-as-Judge for trajectory correctness**: The indicator function `I(.)` uses code execution for code tasks but isn't clearly specified for math tasks. If it's answer-matching, the usual parsing issues apply.
- **The constraint format is never evaluated in isolation**: We don't know whether the "enforce X; avoid Y" format is better than, say, a worked example or a reasoning chain. The paper conflates the contrastive extraction with the constraint format.
- **No memory bank statistics**: How many constraints are generated? How many per task category? What's the overlap/redundancy? This is a memory system paper with no analysis of the memory itself.

---

## Relevance to Somnigraph

### What MemCollab Does That We Don't

**1. Contrastive memory construction from multiple reasoning traces.** Somnigraph stores memories as they're produced by a single Claude instance. There's no mechanism to compare a successful reasoning trace against a failed one and distill the structural difference. The closest analog is the `procedural` category with `source="correction"` -- user corrections capture "you did X wrong, do Y instead" -- but this is reactive and human-mediated, not systematic. MemCollab's approach could be applied at `sleep_rem.py` consolidation time: for memories tagged with correction themes, automatically contrast the correction against the original behavior to produce cleaner procedural constraints.

**2. Task-category-conditioned retrieval.** Somnigraph's retrieval in `fts.py` and `scoring.py` treats all memories as a flat candidate pool. The `category` field (episodic/semantic/procedural/reflection/meta) and `themes` serve a similar filtering purpose, but there's no hierarchical task taxonomy that restricts retrieval to a narrow relevant subset. The reranker in `reranker.py` handles relevance scoring, but the candidate generation phase doesn't pre-filter by task type. For Somnigraph's ~730 memories this is fine -- the search space is small enough that the reranker handles it. At scale, category-conditioned retrieval would matter more.

**3. "Avoid X" as first-class memory content.** Somnigraph memories describe what to do or what happened. The "avoid" half of MemCollab's constraints -- explicitly encoding failure patterns -- is underrepresented. Procedural corrections exist but are typically framed positively ("do Y") rather than as a (do Y, don't do X) pair. The `contradiction` edge flag in the graph is the closest structural analog, but it flags inter-memory conflicts, not intra-pattern dos/don'ts.

### What We Already Do Better

**1. Retrieval sophistication.** MemCollab's retrieval is TF-IDF within a task-classified bucket. Somnigraph's hybrid search (FTS5 BM25 + sqlite-vec embeddings + RRF fusion) followed by a 26-feature LightGBM reranker is orders of magnitude more capable. The reranker's `query_coverage`, `proximity`, and `vec_dist` features alone handle the "is this memory relevant to this query?" question far more precisely than classification + TF-IDF.

**2. Feedback loop.** MemCollab has no feedback mechanism at all. Once memories are constructed, they're static. Somnigraph's explicit per-query utility feedback (r=0.70 GT correlation) means the system learns which memories are actually useful over time. The EWMA aggregation, UCB exploration, and Hebbian co-retrieval in `scoring.py` and `reranker.py` are all absent from MemCollab.

**3. Memory lifecycle and consolidation.** MemCollab's memory bank is constructed offline and never updated. Somnigraph's sleep pipeline (NREM edge detection, REM clustering/summarization/archiving), biological decay, and confidence tracking handle the temporal dynamics of memory that MemCollab entirely ignores. Memories evolve, decay, get promoted or archived -- none of which exists here.

**4. Memory granularity and schema.** Somnigraph's rich schema (priority, decay_rate, confidence, shadow_load, layer, valid_from/valid_until, themes, category) and graph structure (edges with linking context and contradiction/revision flags) represent a much more nuanced model of memory than MemCollab's flat (constraint_text, category, subcategory) entries.

**5. Scale-appropriate design.** MemCollab is designed for stateless reasoning tasks on benchmarks. Somnigraph manages persistent, evolving memory for a long-running agent with ~730 memories and ~1500 edges. The problems are structurally different -- MemCollab never has to handle temporal reasoning, memory updates, contradictions, or multi-session continuity.

---

## Worth Stealing (ranked)

### 1. Contrastive correction distillation in sleep pipeline

**What**: When REM consolidation in `sleep_rem.py` encounters correction-themed procedural memories, contrast the original behavior against the correction to produce a cleaner "enforce X; avoid Y" constraint.

**Why it matters**: Current corrections are human-mediated and inconsistently formatted. Systematic contrastive distillation could produce more transferable procedural memories. This directly addresses the "correction" memory pathway described in the CLAUDE.md snippet -- corrections are already p7 procedural memories, but they preserve the raw phrasing rather than extracting the structural reasoning constraint.

**Implementation**: In `sleep_rem.py`, during step 7 (theme enrichment) or as a new step, identify procedural memories with `source="correction"` themes. For each, use the LLM to extract: (a) what reasoning pattern led to the error, (b) what the correct pattern is, (c) abstract both into a domain-agnostic constraint. Store the refined version alongside or replacing the original. Could reuse the existing `_summarize_cluster()` machinery with a contrastive prompt variant.

**Effort**: Medium. The core logic is a prompt engineering task on top of existing sleep infrastructure. The harder part is evaluating whether the refined constraints are actually more useful than the raw corrections (would need A/B testing via the feedback loop).

### 2. Explicit "anti-patterns" in procedural memories

**What**: Add a structured "avoid" field to procedural memories, pairing what-to-do with what-not-to-do.

**Why it matters**: The multi-hop failure analysis found that 88% of evidence turns have zero content-word overlap with queries. One contributor to this vocabulary gap is that procedural memories describe correct behavior but don't describe the failure modes they're guarding against. An "avoid" field would add vocabulary that matches failure-mode queries, potentially improving FTS5 retrieval in `fts.py` for procedural content.

**Implementation**: Schema change in `db.py` -- either a new `anti_pattern` column or encoding within the existing `content` field as a structured section. The `impl_remember` function in `tools.py` would accept an optional `avoid` parameter. Write-time enrichment in `sleep_rem.py` could populate this for existing corrections. The BM25 index in `fts.py` would naturally pick up the additional vocabulary.

**Effort**: Low-Medium. Schema change is trivial; the value depends on whether the additional vocabulary actually helps retrieval, which would show up in the LoCoMo benchmark or production feedback metrics.

### 3. Task-aware pre-filtering as a reranker feature

**What**: Add a `task_category_match` binary feature to the reranker indicating whether the query and candidate memory belong to the same broad task category.

**Why it matters**: MemCollab's JSD analysis shows that error patterns cluster by task category. The production reranker's `query_coverage` and `proximity` features partially capture this, but an explicit category-match signal could help. This is cheap to compute and could be trained into the existing 26-feature model (bringing it to 27 or 31 with the pending features).

**Implementation**: In `reranker.py`, classify the query into a coarse category (could use the existing `themes` field of candidate memories as proxy). Add as feature in `_compute_features()`. Would need GT annotation of queries by task category for training.

**Effort**: Medium. The feature extraction is simple but building the task taxonomy and annotating training data is the real cost. May not be worth it at Somnigraph's current scale (~730 memories).

---

## Not Useful For Us

**Cross-agent memory sharing.** Somnigraph serves a single Claude Code instance. The core problem MemCollab solves -- filtering agent-specific bias so memory transfers across models -- doesn't apply. If Somnigraph ever served multiple model backends (e.g., Claude + GPT), this would become relevant, but it's not on any current roadmap.

**The task classification taxonomy.** MemCollab's taxonomy is hand-designed for math/code benchmarks (7 MATH categories with ~8-15 subcategories each). Somnigraph's memory spans personal/professional/technical domains with no clean taxonomic structure. The `category` field (episodic/semantic/procedural/reflection/meta) serves a different purpose -- it describes the *type* of memory, not the *domain* it applies to. Building a domain taxonomy for Somnigraph's heterogeneous memory would be over-engineering for ~730 memories.

**TF-IDF retrieval within buckets.** Somnigraph's hybrid FTS5 + vector + RRF + reranker pipeline already far exceeds this. Adopting TF-IDF would be a regression.

**The training-time trajectory generation.** MemCollab requires generating reasoning traces from two models on a training dataset, then contrasting them offline. Somnigraph operates in a single-agent, online setting where memories are created during normal interaction, not from batch processing of training problems.

**The "normative constraint" format exclusively.** The "When X, enforce Y; avoid Z" format is well-suited for procedural reasoning constraints but wouldn't replace Somnigraph's existing memory types (episodic events, semantic knowledge, reflections, meta). It's a format that works for one category of memory, not a universal memory representation.

---

## Impact on Implementation Priority

**Minimal impact on current priorities.** MemCollab addresses a problem (cross-agent memory sharing) that Somnigraph doesn't have, using techniques (batch trajectory generation, offline contrast) that don't fit Somnigraph's online single-agent setting.

**Minor influence on P2 (reranker iteration)**: The idea of a `task_category_match` feature is interesting but low-priority relative to the pending 31-feature retrain and the raw-score features experiment. It would be a candidate for a future feature round, not the current one.

**Minor influence on P4 (LoCoMo benchmark)**: The contrastive correction idea could inform how procedural memories are written for the LoCoMo QA benchmark, but this is indirect. The multi-hop vocabulary gap problem isn't caused by agent-specific bias -- it's caused by fundamental semantic distance between queries and evidence. MemCollab's approach doesn't address this.

**No impact on**: Roadmap Tier 1 #3 (sleep impact measurement), #5 (counterfactual coverage), #10 (prospective indexing), #21 (expansion ablation). The feedback self-reinforcement open problem, contradiction detection, and edge pruning are all orthogonal to this paper's concerns.

The "contrastive correction distillation" idea (Worth Stealing #1) could eventually feed into sleep pipeline improvements, but it's not high enough priority to bump any current roadmap item.

---

## Connections

**Dynamic Cheatsheet** (Suzgun et al., 2025; `dynamic-cheatsheet.md`): MemCollab explicitly compares against DC as a baseline and outperforms it. The key difference: DC accumulates from a single agent's trajectory; MemCollab contrasts across agents. DC's DC-RS variant also uses embedding retrieval (OpenAI text-embedding-3-small, same as Somnigraph), while MemCollab uses TF-IDF -- a surprising regression in retrieval sophistication. Both store reasoning guidance rather than raw experiences, but DC stores strategies/snippets while MemCollab stores constraints.

**Mem0** (`mem0-paper.md`): Both are memory systems for LLM agents, but solving different problems. Mem0's extract-then-update pipeline operates on conversational content; MemCollab operates on reasoning trajectories. Mem0 has no cross-agent concern. The only connection is the shared finding that naive memory transfer doesn't work well -- Mem0's flat memories entangle facts with extraction artifacts, MemCollab's naive transfer entangles reasoning with agent bias. Different instantiations of the same problem.

**A-Mem** (`a-mem.md`): A-Mem's Zettelkasten-inspired enriched embeddings represent an alternative approach to memory abstraction. Both A-Mem and MemCollab abstract away from raw content, but in different directions: A-Mem enriches with linked context (similar to Somnigraph's edge structure), while MemCollab distills down to normative constraints. Somnigraph's approach is closer to A-Mem's (rich, connected, evolving) than to MemCollab's (flat, static, constraint-based).

**HyDE** (`hyde.md`): The contrastive trajectory analysis has a faint structural similarity to HyDE's hypothetical document generation -- both use an LLM to generate an intermediate representation that bridges a gap (HyDE bridges query-document vocabulary gap, MemCollab bridges agent-agent reasoning gap). But the mechanisms are completely different and the connection is superficial.

**LoCoMo** (`locomo.md`): MemCollab doesn't evaluate on LoCoMo or any conversational memory benchmark. Its evaluation is purely on stateless reasoning tasks (MATH, GSM8K, MBPP, HumanEval). This is a significant scope limitation -- it tells us nothing about how contrastive memory construction would work for the episodic/temporal memory that LoCoMo tests and that Somnigraph primarily handles.

---

## Summary Assessment

MemCollab is a clean paper with a well-defined contribution: contrastive distillation of reasoning trajectories produces agent-agnostic memory that transfers across model sizes and families. The experimental design is solid -- the memory-source ablation (same model, other model, self-contrast, cross-contrast) isolates the contribution cleanly, and the finding that naive transfer hurts while contrastive transfer helps is convincing. The task-aware retrieval mechanism is simple but empirically justified by the JSD analysis.

The paper's main weakness is its narrow evaluation scope. All benchmarks are stateless reasoning tasks with verifiable answers. There's no evaluation on conversational memory, temporal reasoning, or any setting where memories evolve over time -- which is Somnigraph's entire operating domain. The memory system itself is never analyzed (how many entries? redundancy? growth characteristics?), and the retrieval mechanism is rudimentary compared to what exists in the literature. The theoretical framing via InfoNCE (Appendix D) is interesting but post-hoc -- the method works via prompting, not by optimizing the InfoNCE objective.

For Somnigraph, the single most useful takeaway is the idea of contrastive correction distillation: when the system has both a failed reasoning attempt and its correction (which happens naturally via user corrections), systematically extracting the structural "enforce/avoid" constraint could produce cleaner procedural memories than storing the raw correction. This is a modest but concrete improvement to the sleep pipeline, not a paradigm shift. The cross-agent sharing problem that motivates the paper is orthogonal to Somnigraph's single-agent architecture.
