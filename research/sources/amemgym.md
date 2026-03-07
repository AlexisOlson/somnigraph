# AMemGym: Interactive Memory Benchmarking for Assistants in Long-Horizon Conversations -- Analysis

*Phase 15, 2026-03-06. Analysis of arXiv:2603.01966 (ICLR 2026).*

---

## 1. Paper Overview

**Paper**: Cheng Jiayang, Dongyu Ru, Lin Qiu, Yiyang Li, Xuezhi Cao, Yangqiu Song, Xunliang Cai. Hong Kong University of Science and Technology + Meitu Inc. ICLR 2026. arXiv:2603.01966, submitted March 2, 2026.

**Problem**: Existing memory benchmarks for conversational AI are *static* and *off-policy*. They construct conversations offline, then replay them to memory systems -- evaluating retrieval on pre-generated traces rather than on the conversations the memory system itself would produce. This creates **reuse bias**: the evaluated system's memory operations (what it stores, when, how it compresses) couple tightly with the conversational dynamics, and those dynamics change when the memory system is swapped. Off-policy evaluation measures how well a system handles *someone else's conversation*; on-policy evaluation measures how well it handles *its own*.

**Core contribution**: Two things. First, **AMemGym**, an interactive environment with simulated users grounded in structured state evolution, enabling on-policy evaluation where the memory system's own behavior shapes the conversation it must remember. Second, a **diagnostic decomposition** that attributes failures to write, read, or utilization stages -- enabling targeted optimization. Third, a **self-evolution** framework where agents iteratively refine their memory policies using environmental feedback.

**Scale**: Base configuration: 20 user profiles, 10 questions each, 200 total questions tested at 11 evaluation points, ~47 turns per user (~943 sessions total), requiring 128K+ context. Extra configuration: 20 profiles with 20 evolution periods, ~82 turns per user (~1,632 sessions), requiring 512K+. 16+ LLMs and multiple memory agent configurations evaluated. User profiles drawn from 100K synthetic personas (Nemotron-Personas, CC BY 4.0).

**Cost**: Structured data generation $0.40/instance. User simulation $0.17/instance (GPT-4.1) or $0.02 (DeepSeek-v3). Evaluation cost ~$13.00/instance (dominated by the evaluated model's inference cost). Fully automated pipeline.

**Code and data**: Not explicitly stated in the paper text, but ICLR 2026 venue and structured benchmark design suggest public release.

---

## 2. Benchmark Design

### On-Policy vs Off-Policy: The Central Distinction

This is the paper's defining contribution and needs careful unpacking.

**Off-policy** (all prior benchmarks): Generate conversation traces offline. Store them. Replay the same trace to every memory system. Evaluate all systems against the same ground truth.

**On-policy** (AMemGym): A simulated user interacts *live* with the memory system under test. The user's utterances expose state information. The memory system's responses, acknowledgments, and behavior influence the conversation flow. Different memory systems produce different conversations and are evaluated on their *own* conversation traces.

**Why this matters**: Table 2 demonstrates empirically that off-policy and on-policy evaluations produce different rankings:
- AWE-(2,4,30): 0.291 on-policy vs 0.253 off-policy (rank drops by 3)
- RAG-(2,4,30): 0.227 on-policy vs 0.241 off-policy (rank rises by 2)
- Gemini-2.5-Flash-Lite: 0.269 on-policy vs 0.204 off-policy (substantial divergence)
- Claude-Sonnet-4: 0.336 vs 0.339 (minimal divergence)

The insight: **for memory agents, the bias is substantial and configuration-dependent**. For native LLMs (which don't actively manage memory), the bias is less pronounced because their conversation traces are more similar to each other. This means off-policy evaluation systematically understates the advantage of better memory systems -- the systems that most actively curate memory are most harmed by reuse bias.

### Structured State Evolution

The benchmark grounds evaluation in a formal state model:

1. **User profiles**: Sampled from Nemotron-Personas (100K synthetic personas spanning ages 18-85, 6 age groups, 9 education levels, 16 occupations). Demographic, geographic, and personality variation.

2. **Canonical state schema**: $\Sigma = \{(s_j, V_j)\}_{j=1}^M$ -- M unique state variables, each with discrete possible values. Extracted from evaluation questions via LLM, then merged/refined to eliminate redundancy.

3. **State evolution**: States progress through $N_p$ periods (10 base, 20 extra). Each transition is prompted by a narrative "life event": $\sigma_{t-1} \xrightarrow{e_t} \sigma_t$. This gives the benchmark a temporal structure -- states genuinely change over time, not just accumulate.

4. **State exposure**: Pre-generated utterances implicitly expose subsets of the current state. A user LLM (GPT-4.1 or DeepSeek-v3) role-plays based on profile + current states + dialogue history, producing coherent turns grounded in the structured data.

5. **Question generation**: For each profile, 10 evaluation questions requiring retrieval and reasoning over 2-3 distinct memory states. Multiple-choice with 4-7 options. Ground-truth answers are validated by a reflection step: an LLM classifier must be able to recover the state variant from the QA pair. Only validated answers are accepted.

### Evaluation Metrics

**Overall Performance ($S_{overall}$)**: Average QA accuracy across all evaluation points.

**Upper Bound ($S_{UB}$)**: Performance when the model is given perfect state access (all relevant state variables provided directly). Tests reasoning without memory confounds.

**Normalized Memory Score ($S_{memory}$)**: Isolates memory from task performance:
$$S_{memory} = \frac{S_{overall} - S_{random}}{S_{UB} - S_{random}}$$

This is clean metric design. By normalizing against both random baseline and reasoning ceiling, it separates memory capability from general model capability. A model with perfect reasoning but terrible memory scores 0. A model with terrible reasoning gets penalized by a low $S_{UB}$, but its memory score can still be high if its memory-limited performance is proportionally close to its ceiling.

**Diagnostic Decomposition**: Three failure categories attributed by probing state knowledge at different points:
- **Write failure**: The system never stored the required state information. Diagnosed by checking whether the state value was accessible at the nearest write position.
- **Read failure**: The system stored the information but cannot retrieve it at evaluation time. Diagnosed by checking stored vs. retrieved state values.
- **Utilization failure**: The system has the information but fails to apply it correctly. Diagnosed when all relevant state values are available to the model but the answer is wrong.

This is conceptually identical to [[agent-output-longmemeval|LongMemEval's]] finding that "50% of errors are reading failures" -- but formalized as a three-way decomposition rather than a post-hoc analysis.

### Memory Approaches Evaluated

| Approach | Write | Read | Key Characteristic |
|----------|-------|------|--------------------|
| **Native LLM** | Raw conversation in context window | Context-window attention | No external memory |
| **RAG** | Raw text indexing in external store | Embedding-based relevance retrieval | No curation at write time |
| **AWE** (Agentic Write External) | LLM-based extraction → external store | Embedding-based relevance retrieval | Curated writes, external retrieval |
| **AWI** (Agentic Write In-Context) | LLM-based compression → in-context buffer | No independent retrieval (in-context) | Curated writes, no retrieval |

AWE variants are parameterized as AWE-(freq, $n_s$, topk): update frequency, minimum short-term messages before writing, and top-k retrieved memories. Default: AWE-(2, 4, 30).

Additionally evaluated: **Mem0-G**, **Nemori**, **A-Mem** as established frameworks (results in Appendix F.4, not fully reported in main text).

### Human Validation

Strong quality assurance:
- State exposure quality: 99.1%, inter-annotator agreement 96.8%
- Conversational state integrity: 99.2%, inter-rater reliability 98.2%
- Ground-truth judgment: 0.92 inter-human agreement, 0.96/0.94 LLM-human agreement

These numbers are impressive. The benchmark's data quality appears genuinely high.

---

## 3. Key Claims and Evidence

### Claim 1: Off-policy evaluation introduces systematic reuse bias for memory agents.

**Evidence**: Table 2 shows ranking inversions between on-policy and off-policy evaluation for memory agents (AWE drops 3 ranks, RAG gains 2). For native LLMs, the bias is smaller (Claude-Sonnet-4: 0.336 vs 0.339) because they don't actively curate memory.

**Evidence quality**: Strong. This is the paper's central empirical claim and it is well-supported by controlled comparisons. The rank inversions demonstrate that off-policy results can be actively misleading for system selection. The theoretical mechanism (memory operations couple with interaction patterns) is intuitive and the data confirms it.

**Implication for claude-memory**: Our system is human-curated (Alexis calls remember/recall), which makes it neither fully on-policy nor off-policy. In an automated setting (hook-driven auto-capture), our system *would* be subject to this bias. Any benchmark evaluation we run using static conversation traces would systematically misestimate our system's performance. This is a genuine methodological concern we should acknowledge.

### Claim 2: All native LLMs lose more than half their upper bound performance as interactions lengthen.

**Evidence**: All tested models achieve $S_{UB} > 0.8$ (strong reasoning with perfect state access), but memory scores drop below 50% of upper bounds in later periods. Some perform "no better than random guessing in later periods." Even Claude-Sonnet-4, the top performer, only achieves 0.336 memory score -- meaning it retains only a third of its theoretical ceiling.

**Evidence quality**: Strong. The $S_{UB}$ control demonstrates that the degradation is a memory problem, not a reasoning problem. The breadth of models tested (16+ including frontier models like GPT-5.2, Claude-Sonnet-4, Gemini-3-Pro-Preview) makes this a robust finding.

**Implication**: Even with expanding context windows, native LLMs cannot reliably remember personal state information across long interactions. External memory remains necessary. This is consistent with [[agent-output-beam|BEAM's]] finding of sharp performance collapse with interaction length.

### Claim 3: Agentic write to external storage (AWE) is the strongest memory approach.

**Evidence**: AWE-(2,4,30) achieves 0.291 memory score, outperforming RAG (0.227), native LLM (0.203), and AWI (0.172). The advantage comes from: (a) curated writes reduce utilization failures (0.074 vs 0.244 for native LLM) by providing relevance-filtered information, and (b) external storage with embedding retrieval provides better recall than in-context compression.

**Evidence quality**: Moderate-strong. The comparison is fair (same backbone model, same environment), but only one backbone (gpt-4.1-mini) is tested for agent configurations. AWE's advantage may vary with model capability.

**Implication**: The optimal architecture is curated writes to external storage + embedding-based retrieval -- which is exactly what claude-memory does. Our `remember()` call with LLM-extracted content stored in SQLite+sqlite-vec is an AWE-style system. This validates our architectural direction.

### Claim 4: Write and utilization failures, not read failures, are the primary bottleneck.

**Evidence**: Mean failure rates across all periods:
| Approach | Write | Read | Utilization |
|----------|-------|------|-------------|
| LLM | .301 | .087 | .244 |
| RAG | .377 | .172 | .067 |
| AWE | .338 | .159 | .074 |
| AWI | .286 | .245 | .122 |

For native LLMs: write (.301) and utilization (.244) dominate, with read surprisingly low (.087). For RAG/AWE: write is the biggest failure mode (.377/.338), suggesting these systems lose information at extraction time.

**Evidence quality**: Strong for the pattern but nuanced. The "write" category here conflates two things: (a) the system never received the information (conversation dynamics), and (b) the system received it but failed to extract/store it. In on-policy evaluation, write failures include cases where the conversation never exposed the relevant state -- which is partly a conversational dynamics issue, not a memory system issue.

**Implication**: This inverts [[agent-output-longmemeval|LongMemEval's]] finding that 50% of errors are reading failures. AMemGym says reads are the *smallest* failure category. The discrepancy likely comes from the evaluation paradigm: LongMemEval uses static traces where information is guaranteed to be present (so write can't fail), while AMemGym's on-policy evaluation includes write failures from never receiving the information. Both are correct within their framing -- but AMemGym's framing is more realistic.

### Claim 5: Self-evolution via environmental feedback improves memory policies.

**Evidence**: Iterative policy refinement improves memory scores from 0.172 to 0.197 (AWI baseline). Write failures drop from 0.293 to 0.263 with complete feedback. Qualitative analysis shows policies evolving from generic ("improve skill tracking") to specific, actionable rules ("teaching approach documentation with logistics schema").

**Evidence quality**: Moderate. The improvement (0.172 to 0.197) is meaningful but modest. Only AWI is tested for self-evolution, and AWI is the weakest baseline -- the gains might be smaller starting from a stronger baseline. The qualitative finding (generic to specific) is more interesting than the quantitative improvement.

**Implication**: This validates our auto-capture correction loop (auto-detecting patterns and adjusting behavior). It also suggests that our memory policies (the CLAUDE.md instructions for what/when/how to remember) could benefit from iterative refinement based on recall_feedback data. The self-evolution framework is essentially "write better prompts based on what worked."

---

## 4. Standout Feature

What makes AMemGym genuinely different from the five benchmarks we have analyzed:

- **LoCoMo/LoCoMo-Plus**: Static traces, off-policy, 10 dialogues, ~300 turns max.
- **LongMemEval**: Static traces, off-policy, needle-in-haystack structure, 500 questions.
- **MemoryAgentBench**: Static traces, off-policy, inject-once-query-many, 146 instances.
- **BEAM**: Static traces, off-policy, 100 coherent conversations up to 10M tokens, 2,000 questions.
- **TReMu**: Static traces, off-policy, temporal reasoning focus, 600 questions.

**AMemGym**: Interactive environment, on-policy, simulated users, configurable scale, diagnostic decomposition, self-evolution loop.

The standout is the **on-policy evaluation paradigm**. All five prior benchmarks share the same fundamental limitation: they evaluate memory systems on conversations those systems did not participate in creating. AMemGym is the first benchmark to close this loop. The ranking inversions in Table 2 prove this matters -- off-policy evaluation is not just theoretically suspect but empirically misleading.

The **diagnostic decomposition** is also novel in its formal three-way attribution (write/read/utilization). LongMemEval identified reading failures informally. AMemGym formalizes and quantifies all three failure modes with a decision-tree attribution methodology.

The **self-evolution loop** makes AMemGym more than a benchmark -- it is an optimization environment. This bridges the gap between evaluation and improvement that no prior benchmark addresses.

---

## 5. Competency Coverage Ratings

| Dimension | Rating | Justification |
|-----------|--------|---------------|
| **Information Retrieval** | 60% | Retrieval is evaluated indirectly through the read failure diagnostic. The benchmark measures whether stored information can be retrieved, but does not test retrieval strategies (multi-hop, fuzzy matching, semantic similarity) in isolation. Retrieval is measured by outcome, not mechanism. |
| **Multi-Session Reasoning** | 55% | Questions require reasoning over 2-3 distinct state variables that may have been exposed across different sessions/periods. However, the reasoning is primarily about *current* state, not about synthesizing information across multiple time points simultaneously. The state evolution creates a multi-session structure, but questions target point-in-time snapshots rather than cross-temporal integration. |
| **Knowledge Update/Contradiction** | 40% | State evolution inherently creates knowledge updates -- a user's job, location, or preferences change over time. Systems must track the *current* value, not a historical one. However, contradictions are not adversarial (no deliberate misinformation), and the benchmark does not test whether systems can detect or resolve conflicting statements. It tests whether they track the latest state, which is a weaker form of update handling. |
| **Temporal Reasoning** | 35% | State evolution across periods provides a temporal dimension, and evaluation at different time points tests temporal tracking. But no temporal *reasoning* is required -- no "when did X change?" or "what was X before Y happened?" questions. The benchmark tests whether memory reflects the current state at each evaluation point, not whether the system can reason about temporal relationships. |
| **Abstention/Confidence** | 15% | Questions are multiple-choice with a valid answer at each evaluation point. There is no "unanswerable" category and no evaluation of confidence calibration. The diagnostic decomposition identifies *why* the system got it wrong, but doesn't test whether it knows what it doesn't know. |
| **Write-Path Behavior** | 70% | This is AMemGym's strongest differentiator among benchmarks. The diagnostic decomposition explicitly measures write failures as a separate category. The AWE/AWI/RAG comparison evaluates different write strategies (curated extraction vs raw indexing vs in-context compression). The self-evolution loop optimizes write policies. No other benchmark directly evaluates write-path behavior. However, "write" is measured by outcome (was the information available later?) rather than by process (was the extraction faithful? was the compression lossy?). |
| **Consolidation Quality** | 10% | The benchmark tests whether information persists across periods, but does not evaluate consolidation processes (merging, summarizing, compressing). AWI's in-context compression is the closest, and it performs worst -- but the evaluation does not diagnose whether compression quality was the cause. |
| **Proactive/Contextual Recall** | 25% | The simulated user does not explicitly request stored information -- questions are posed at evaluation checkpoints, not during natural conversation flow. There is no test of whether the system proactively surfaces relevant memories during conversation. The "utilization" diagnostic partially captures this: can the system apply stored knowledge when asked? But proactive = unprompted, which is not tested. |
| **Relationship/Graph Reasoning** | 10% | Questions require 2-3 state variables, but these are independent attributes of a single user profile, not relationships between entities. No graph traversal, entity linking, or relationship reasoning is tested. |
| **Agentic Task Performance** | 45% | Memory agents actively decide what to store and when -- this is genuinely agentic. The self-evolution loop is agentic meta-learning. But the task is narrowly defined (track personal state, answer MCQs) with no tool use, planning, or autonomous action beyond memory management. |

---

## 6. Relevance to claude-memory

### Could we run against this benchmark?

**Yes, with moderate adaptation.** AMemGym's interactive design means we cannot just replay static traces -- we would need to connect our system to the simulated user environment. This requires:

1. **An adapter layer** that maps the simulated user's utterances to `remember()` calls (or auto-capture via hooks) and evaluation questions to `recall()` calls. Our system currently relies on Alexis to invoke these tools; an automated adapter would need to decide *what* to remember from each turn.

2. **State extraction at write time**: Our `remember()` accepts free-text content with category/priority metadata. The AWE pattern (LLM-based extraction of salient state from conversation turns) is what our auto-capture hooks would need to do. This is a test of write-path quality.

3. **Evaluation harness**: Connect to AMemGym's user simulator, run conversations, evaluate at checkpoint positions. The multiple-choice format simplifies evaluation (no LLM judge needed for answer quality).

### What would it reveal?

1. **Write-path quality**: Our biggest gap. AMemGym would expose whether auto-capture correctly identifies and stores state-changing information. With human curation (Alexis calls `remember()`), write quality depends on Alexis. With hooks, it depends on the extraction prompt. AMemGym would give us a number for each.

2. **On-policy retrieval**: Whether our RRF hybrid retrieval (vector + FTS5) surfaces the right memories when conversation dynamics are shaped by our own system's behavior. This is the bias that Table 2 quantifies.

3. **Write vs read vs utilization breakdown**: The diagnostic decomposition would tell us which stage to invest in. Based on our architecture (enriched embeddings, BM25 field weighting, Hebbian PMI boost), we should have strong read performance. Write quality (extraction completeness) and utilization (can the model use retrieved memories correctly) are less certain.

4. **Configuration sensitivity**: The AWE parameter sweep (freq, n_s, topk) maps to our system's tunable parameters -- how often to auto-capture, how many recent messages to include, how many memories to retrieve. AMemGym would let us optimize these.

### Adaptation needed

- **Auto-capture pipeline**: Our hooks need to function as an AWE write-path -- extracting salient state from each turn and calling `remember()`. This is the main implementation gap.
- **Retrieval adapter**: Map evaluation questions to `recall()` calls, format results for the LLM to answer MCQs.
- **State variable tracking**: AMemGym tests discrete state variables (job, location, preferences with specific values). Our memory store is free-text. We would need to either (a) extract state from memories at read time, or (b) store structured state alongside free-text content.
- **User simulator integration**: Run the AMemGym user simulator against our system. Requires Python integration with their environment.

**Effort estimate**: Medium-high (~1-2 weeks). The main work is building the auto-capture adapter and integrating with the AMemGym environment. The evaluation itself is straightforward (MCQ accuracy + diagnostic probing).

---

## 7. Insights Worth Stealing

Ranked by effort/impact ratio (best first):

### 7.1 Diagnostic Write/Read/Utilization Decomposition (High impact, Low effort)

The three-way failure attribution is the most directly transferable idea. For any evaluation we run -- even informal self-testing -- we should decompose errors into these three categories:
- **Write**: Did `remember()` capture the relevant information?
- **Read**: Did `recall()` retrieve it?
- **Utilization**: Did the model use it correctly?

**Adaptation**: Add diagnostic probing to our recall_feedback workflow. When a recall result is marked low-utility, log whether the issue was: (a) the relevant memory wasn't stored (write), (b) the relevant memory wasn't surfaced (read), or (c) the memory was surfaced but not used effectively (utilization). This requires minimal code changes -- just an optional `failure_stage` field on recall_feedback.

### 7.2 On-Policy Awareness for Benchmark Interpretation (High impact, Zero effort)

Any time we evaluate against a static benchmark (LoCoMo, LongMemEval, BEAM), we should note that the results are off-policy and may not transfer to our actual usage pattern. This is a calibration insight, not an implementation change. AMemGym's Table 2 gives us a rough estimate of the bias magnitude: up to 0.065 memory score difference, with rank inversions possible.

### 7.3 Self-Evolution Policy Refinement Pattern (Medium impact, Medium effort)

The iterative loop (run → get feedback → refine policy → repeat) could be applied to our auto-capture prompts. After collecting recall_feedback data on what was useful vs. not, we could periodically regenerate the extraction prompt used by auto-capture hooks, incorporating patterns from high-utility vs. low-utility captures.

**Adaptation**: Monthly or quarterly review of recall_feedback data to refine the auto-capture extraction prompt. The "complete feedback" variant (showing the system its answers vs. ground truth) could be approximated by showing the extraction prompt examples of what it captured vs. what Alexis actually remembered manually.

### 7.4 Normalized Memory Score as Evaluation Metric (Medium impact, Low effort)

$S_{memory} = (S_{overall} - S_{random}) / (S_{UB} - S_{random})$ cleanly separates memory from reasoning. If we build CMA behavioral probes (#10), we should compute $S_{UB}$ by providing perfect context and measure how much our retrieval degrades from that ceiling.

**Adaptation**: For any evaluation, run a "perfect retrieval" baseline alongside the real system. The gap is the pure memory cost.

### 7.5 State-Evolution Data Generation Pipeline (Low impact, High effort)

The structured data generation (profile → schema → state evolution → utterances) is well-engineered and could be used to generate custom test data for our system. But the effort of implementing the full pipeline is high, and we could use AMemGym's own environment directly.

**Adaptation**: Only worth it if we need test data with specific characteristics not covered by AMemGym (e.g., our domain-specific state variables like project status, preference evolution, procedural corrections).

---

## 8. What's Not Worth It

### User simulator fine-tuning
The paper discusses using GPT-4.1 vs DeepSeek-v3 as user simulators and their cost tradeoffs. This is benchmark infrastructure engineering, not a transferable finding.

### AWI (Agentic Write In-Context) approach
AWI performs worst across the board (0.172 memory score). Its failure mode -- information loss from in-context compression -- is exactly what we've already rejected by using external storage. No insight here.

### Multiple-choice question format
The MCQ evaluation makes the benchmark tractable but limits the types of memory tasks tested. Personality-consistent preferences, nuanced relationship understanding, and open-ended recall cannot be captured in 4-7 option MCQs. For our use case (human-facing assistant with rich conversational context), MCQ evaluation is artificially narrow.

### Cost optimization of data generation
The paper's analysis of $0.40/instance for data synthesis and $0.17/$0.02 for simulation costs is specific to their pipeline and not transferable.

---

## 9. Key Takeaway

AMemGym's most important contribution is not any single finding but a **methodological correction**: off-policy evaluation of memory agents is systematically biased, and the bias is worst for the systems that most actively curate their memory -- exactly the systems we want to build. The diagnostic decomposition (write/read/utilization) is the most directly actionable insight: it provides a principled framework for diagnosing memory failures that we should adopt regardless of whether we run against AMemGym itself. The finding that write failures dominate for external-storage systems (.338 for AWE, .377 for RAG) inverts the conventional wisdom from [[agent-output-longmemeval|LongMemEval]] (where reading failures dominated) and suggests that extraction quality -- *what* we store, not *how* we retrieve it -- is the binding constraint for our architecture. The self-evolution result confirms that memory policies can be iteratively improved through downstream task feedback, which validates both our recall_feedback mechanism (#14) and the broader trajectory toward hook-driven auto-capture with quality feedback.

---

## 10. Impact on Implementation Priority

**No reordering of the #1-#14 priority list.** Specific impacts:

- **#14 (Retrieval feedback loop)**: Strongly reinforced. AMemGym's self-evolution framework is essentially the same idea: use downstream task performance to iteratively improve memory policies. Their improvement from 0.172 to 0.197 via policy refinement maps to our plan of using recall_feedback data to tune extraction and retrieval parameters. The diagnostic decomposition adds a new dimension: feedback could specify *which stage* failed, enabling targeted policy updates.

- **#10 (CMA behavioral probes)**: Enriched with methodology. AMemGym's diagnostic decomposition should be incorporated into our probe design. Every probe question should be attributable to write, read, or utilization failure when the system gets it wrong. AMemGym's normalized memory score ($S_{memory}$) should be adopted as the evaluation metric, computed against a perfect-context $S_{UB}$ baseline.

- **#12 (Reading strategy optimization)**: Nuanced by AMemGym's finding that read failures are the *smallest* failure category for external-storage systems (.159 for AWE). This does not contradict LongMemEval's reading-failure finding (that study provides information perfectly and tests reading), but it reframes the priority: in end-to-end on-policy evaluation, improving read strategy has less headroom than improving write-path quality. Reading strategy remains important but may be less binding than we thought.

- **Auto-capture hooks (not yet numbered)**: AMemGym's results implicitly argue for high-quality auto-capture as a top priority. Write failures dominate for all external-storage systems, and write quality is determined by the extraction process -- which, in our system, is the auto-capture hook or Alexis's manual `remember()` calls. If we move toward automated memory (which the benchmark is designed to evaluate), extraction quality becomes our most important lever. This may warrant elevating auto-capture design to a numbered priority.

- **#3 (Contradiction detection)**: Partially tested but not directly evaluated. AMemGym's state evolution creates implicit knowledge updates (a user's job changes), but the benchmark tests whether the system tracks the latest state, not whether it detects or resolves contradictions. The write/read/utilization decomposition does not include a "stale knowledge" failure mode. Our contradiction detection remains a differentiator that AMemGym does not evaluate.

- **#1 (RRF fusion)**: Validated by AWE's retrieval architecture. AWE uses embedding-based retrieval (single channel) with curated writes, achieving the best scores. Adding BM25 and PPR (our target architecture) should improve read performance further, but AMemGym suggests the read channel is not the binding constraint -- write quality is. Multi-channel RRF remains correct but its impact ceiling may be lower than the write-path improvements that AMemGym prioritizes.

---

## 11. See Also

- [[agent-output-beam]] -- BEAM benchmark analysis (ICLR 2026). 10M-token coherent conversations, 10 memory abilities, 2,000 questions, LIGHT framework. Off-policy. Contradiction Resolution universally catastrophic. Noise filtering as important as retrieval at scale.
- [[agent-output-longmemeval]] -- LongMemEval benchmark analysis (ICLR 2025). 5 abilities, 500 questions, ~1.5M tokens max. "50% of errors are reading failures." Off-policy. AMemGym's diagnostic decomposition formalizes and extends this finding, but inverts the conclusion for on-policy evaluation (write failures dominate instead).
- [[agent-output-memoryagentbench]] -- MemoryAgentBench analysis (ICLR 2026). 4 competencies, 146 instances. Multi-hop selective forgetting catastrophic (max 7%). Off-policy.
- [[agent-output-locomo]] -- LoCoMo benchmark analysis (ACL 2024). Full-context baseline 72.9%, filesystem agent 74%. Off-policy.
- [[agent-output-locomo-plus]] -- LoCoMo-Plus analysis (Feb 2026). Cognitive memory with implicit constraints. 15-26 point drop from factual to cognitive. Off-policy.
- [[agent-output-tremu]] -- TReMu benchmark analysis (ACL Findings 2025). Temporal reasoning benchmark. GPT-4o baseline 30%, framework 78%. Off-policy.

### Cross-benchmark synthesis (updated)

| Dimension | LoCoMo (2024) | LongMemEval (2025) | MAB (2026) | BEAM (2026) | TReMu (2025) | LoCoMo+ (2026) | **AMemGym (2026)** |
|-----------|---------------|-------------------|------------|-------------|--------------|-----------------|-------------------|
| Max scale | 26K tokens | 1.5M tokens | 1.44M tokens | 10M tokens | 16K tokens | ~26K tokens | 128K-512K+ tokens |
| Questions | 5 types | 500 | 146 instances | 2,000 | 600 | diagnostic | 200 (x11 periods) |
| Evaluation | Off-policy | Off-policy | Off-policy | Off-policy | Off-policy | Off-policy | **On-policy** |
| Construction | Machine-human | Needle-haystack | Inject-query | Coherent narrative | Built on LoCoMo | Built on LoCoMo | **Structured state evolution** |
| Ecological validity | Medium | Medium-low | Low | High | Low | Medium | **High** |
| Metric | LLM-judge | LLM-judge | SubEM/accuracy | Nugget partial | Accuracy/F1 | Constraint validity | **MCQ + diagnostic** |
| Write-path eval | No | No | No | No | No | No | **Yes (diagnostic)** |
| Contradiction | Adversarial only | Folded into KU | SF (0-7%) | Near-zero | None | None | **Implicit (state update)** |
| Self-improvement | No | No | No | No | No | No | **Yes (self-evolution)** |
| Unique strength | Full-context baseline | Reading strategy | Agentic competencies | Scale + coherence | Temporal depth | Cognitive memory | **On-policy + write diagnostics** |

The updated gap: AMemGym is the first benchmark to evaluate write-path behavior, breaking the universal "write-blind" pattern across all prior benchmarks. However, it evaluates write quality by outcome (was information available later?), not by process (was the extraction faithful?). A truly comprehensive write-path evaluation -- measuring extraction completeness, compression fidelity, and update correctness independently -- remains unbuilt.
