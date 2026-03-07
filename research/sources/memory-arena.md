# MemoryArena: Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks -- Analysis

*Phase 15, 2026-03-06. Analysis of arXiv:2602.16313 (Feb 2026).*

---

## 1. Paper Overview

**Paper**: Zexue He, Yu Wang, Churan Zhi, Yuanzhe Hu, Tzu-Ping Chen, Lang Yin, Ze Chen, Tong Arthur Wu, Siru Ouyang, Zihan Wang, Jiaxin Pei, Julian McAuley, Yejin Choi, and Alex Pentland. UC San Diego, MIT Media Lab, University of Michigan. arXiv:2602.16313, submitted February 18, 2026. Not yet peer-reviewed at time of analysis.

**Problem**: Existing memory benchmarks either (a) test recall in isolation (LoCoMo, LongMemEval, MemoryAgentBench) through post-hoc single-query retrieval, or (b) evaluate single-session agentic tasks (WebShop, SWE-Bench) that do not require persistent memory across episodes. No benchmark tests whether agents can *acquire* information through environment interaction, *store* it persistently, and *apply* it to solve causally dependent future tasks. This is the fundamental loop of any real deployment: the memory-agent-environment cycle.

**Core contribution**: A benchmark of 736 multi-session task instances across four domains (web shopping, group travel planning, progressive web search, formal reasoning in math and physics), where later subtasks causally depend on information acquired in earlier subtasks. The benchmark is paired with a POMDP (Partially Observable Markov Decision Process) formalization that reframes memory failures as belief drift. Evaluation covers 4 long-context models, 4 external memory systems, and 4 RAG systems.

**Scale**: 736 task instances, 5-16 subtasks per instance, traces ranging from 14k to 122k tokens. Expert-curated formal reasoning tasks from PhD-level academic papers. 12 systems evaluated (4 long-context, 4 memory, 4 RAG).

**Code and data**: Not confirmed at time of analysis. Likely forthcoming given the venue and author affiliations.

---

## 2. Benchmark Design

### Task Domains

| Domain | Tasks | Sessions/Task | Avg Trace | Construction |
|--------|-------|--------------|-----------|-------------|
| Bundled Web Shopping | 150 | 6 (fixed) | 41.5k tokens | WebShop items, compatibility chains |
| Group Travel Planning | 270 | 5-9 (variable) | 40.6k tokens | TravelPlanner base + sequential participants |
| Progressive Web Search | 256 | 2-16 (variable) | 122.4k tokens | BrowseComp-Plus filtered + decomposed |
| Formal Reasoning (Math) | 40 | 2-16 (variable) | 18.1k tokens | PhD-expert curated derivation chains |
| Formal Reasoning (Physics) | 20 | 2-12 (variable) | 14.1k tokens | PhD-expert curated derivation chains |

**Key design property**: Causal dependencies between subtasks. Each later subtask requires information that was only available through earlier subtask execution. This is not just "retrieve a fact mentioned earlier" -- it is "use the product you purchased in session 2 to constrain your purchase in session 5." The agent must act to acquire information, not just read it.

### Dataset Construction

Each domain has a distinct pipeline:

- **Web Shopping**: Category filtering on WebShop (1M+ items), screening rule templates with `dependency_map` and `reject_map` constraints, distractor sampling (2 compatible + 2 incompatible per level), preference injection (budget/rating), manual verification.
- **Travel Planning**: 45 TravelPlanner base instances transformed by adding 5-8 participants with JOIN constraints (shared activities) and RELATION constraints (comparative preferences). Dependency chains up to depth 4.
- **Web Search**: BrowseComp-Plus (830 entries) filtered to 256 tasks that require multi-step information accumulation. Each subquery adds one constraint with strict causal ordering. Human-annotated for semantic coherence and verified solvability.
- **Formal Reasoning**: Senior PhD experts decompose academic paper derivations into ordered lemma/proposition sequences. Each intermediate result becomes a question. Causal consistency enforced -- no statement depends on later information.

### Evaluation Metrics

| Metric | Definition | Scope |
|--------|-----------|-------|
| **Task Success Rate (SR)** | Fraction of tasks fully solved (all subtasks correct) | All domains |
| **Task Progress Score (PS)** | Average fraction of subtasks correctly completed per task | All domains |
| **Soft Progress Score (sPS)** | Partial credit based on constraint satisfaction fraction | Travel Planning only |
| **SR@k** | Success rate at the k-th subtask depth | Decay analysis |

The PS metric is essential here -- SR alone would show near-zero everywhere and be uninformative. sPS was added specifically because Travel Planning drove both SR and PS to zero for all methods, making comparison impossible without partial credit.

### Systems Evaluated

**0D (flat, no structure)**:
- Long-context agents: GPT-5.1-mini, GPT-4.1-mini, Gemini-3-Flash, Claude-Sonnet-4.5
- RAG: BM25, text-embedding-3-small

**1D (flat with consolidation)**:
- Memory: Letta (MemGPT), Mem0, ReasoningBank
- RAG: MemoRAG

**2D (structured/graph)**:
- Memory: Mem0-g (graph variant)
- RAG: GraphRAG

All external memory systems implement two abstract functions: `Retrieve(M, task, actions, observations)` and `Update(M, traces)`. This clean abstraction makes the comparison fair.

---

## 3. Key Claims and Evidence

### Claim 1: Agents near-saturated on memory recall benchmarks fail catastrophically on agentic memory tasks.

**Evidence**: Strong. Mean SR across all methods: Shopping 0.02, Travel 0.00, Search 0.38, Math 0.09, Physics 0.35. Best single-domain SR: text-embedding-3-small at 0.50 on Web Search. Best overall SR: Claude-Sonnet-4.5 and text-embedding-3-small at 0.19-0.23. These same models achieve near-saturated scores on LoCoMo. The gap is massive and consistent across all 12 systems.

**Quality**: High. The comparison is apples-to-apples -- same underlying models, different benchmark design. The performance collapse is real and attributable to the benchmark's causal dependency structure, not to model capability.

### Claim 2: External memory systems often underperform simple long-context approaches.

**Evidence**: Moderate. Long-context average SR: 0.16. Memory agent average SR: 0.13. RAG average SR: 0.20. The pattern varies by domain -- RAG outperforms long-context on Search (0.49 vs 0.44 SR) and Formal Reasoning (Math: 0.11 vs 0.04 SR), but memory agents universally underperform on Shopping (0.00 vs 0.06 SR). The claim is slightly overstated -- RAG systems are competitive or better in domains with long traces. But the headline finding holds: external memory does not reliably help, and often hurts.

**Quality**: Medium-high. The result is confounded by the task agent being fixed (GPT-5.1-mini for all external memory conditions). A jointly-trained or fine-tuned task agent might extract more value from structured memory. The authors acknowledge this as the "training mismatch" bottleneck.

### Claim 3: Performance decays consistently with subtask depth, interpretable as belief drift.

**Evidence**: Strong. SR@k decay curves (Figure 3) show monotonic decline in all domains for all methods. Shopping: ~60% to ~25% (long-context) over 6 subtasks. Search: ~60% to ~20% over 16 subtasks. The POMDP framing is clean: small errors in the agent's implicit state estimate compound across sessions, exactly matching the Partially Observable MDP prediction of belief drift under imperfect observation.

**Quality**: High conceptually, medium empirically. The POMDP formalization is elegant and explanatory, but the paper does not estimate actual belief-state divergence -- it observes SR decay and attributes it to belief drift. This is plausible but not rigorously demonstrated.

### Claim 4: Memory systems add 1.5-2x latency without compensating performance gains.

**Evidence**: Strong. Long-context average: 33.6-81.8 sec/subtask. Memory systems: 114.8-132.8 sec. RAG: 90.2-107.4 sec. The latency premium is real and consistent. No systematic relationship exists between memory complexity (0D vs 2D) and latency, suggesting operational overhead dominates.

**Quality**: High. This is a practical finding that directly impacts deployment decisions.

### Claim 5: When external memory helps, it is specifically in high-token-count scenarios (>120k tokens) where attention saturation degrades long-context performance.

**Evidence**: Moderate. Progressive Web Search (122.4k tokens avg) is the one domain where RAG consistently outperforms long-context. But RAG also outperforms on Formal Reasoning (18.1k tokens), which contradicts the "high-token-count" framing. The more precise claim is: external memory helps when information is *distributed and selectively needed* rather than when traces are simply long.

**Quality**: Medium. The trace-length explanation is partial. The formal reasoning result suggests that selective retrieval of intermediate results (lemmas, propositions) is valuable independent of total context length.

---

## 4. Standout Feature

**What makes this different**: MemoryArena is the first benchmark that tests *agentic memory acquisition and application* rather than *memory recall*. Prior benchmarks inject information into conversation history and test whether the system can retrieve it later. MemoryArena requires the agent to:

1. **Act in an environment** to discover information (browse products, search the web, verify proofs)
2. **Decide what to store** from action traces
3. **Retrieve and apply** stored information to solve dependent future tasks

This is the write-path evaluation that [[agent-output-beam|BEAM]], [[agent-output-longmemeval|LongMemEval]], and [[agent-output-memoryagentbench|MemoryAgentBench]] all lack. The Batch 1 synthesis identified "no benchmark evaluates write-path behavior" as a critical gap. MemoryArena does not directly evaluate write quality, but it *indirectly* evaluates it by making task success dependent on what the agent chose to store. If the agent stores the wrong things (or stores them in a lossy format), later tasks fail. This is the first benchmark where write-path quality is load-bearing rather than invisible.

The POMDP formalization is the second distinguishing feature. By framing memory failures as belief-state estimation errors that accumulate across sessions, the authors provide a theoretical framework for understanding *why* systems fail, not just *that* they fail. This connects memory system evaluation to a well-studied mathematical framework with known optimal solutions, which creates a clear path for improvement: build memory systems that preserve sufficient statistics for belief tracking.

---

## 5. Competency Coverage Ratings

### Information Retrieval — 60%
Tests whether retrieved information enables correct task completion across sessions. Does not isolate retrieval quality from reading quality or reasoning quality. The SR@k decay analysis provides indirect retrieval diagnostics. Strong on testing retrieval *in context* (during agentic execution), weaker on isolating retrieval as an independent capability.

### Multi-Session Reasoning — 90%
The core strength of the benchmark. Every task requires reasoning across 2-16 sessions with causal dependencies. Formal reasoning tasks (math/physics derivation chains) test deep multi-hop reasoning. Travel planning tests constraint propagation across participants. This is the best multi-session reasoning benchmark in the literature.

### Knowledge Update/Contradiction — 15%
Not directly tested. Tasks do not present contradictory information or require updating prior beliefs. The benchmark assumes information acquired in earlier sessions remains valid. The closest approximation is Travel Planning's evolving constraint set, but this is accumulation (adding constraints), not contradiction (revising beliefs).

### Temporal Reasoning — 20%
Subtask ordering is strictly sequential and causally enforced, but the benchmark does not test temporal reasoning as such (no "when did X happen" or "what changed between session 3 and session 7" questions). The SR@k metric captures temporal degradation, but as a diagnostic tool rather than a tested capability. Far weaker than [[agent-output-tremu|TReMu]]'s explicit temporal decomposition.

### Abstention/Confidence — 5%
Not tested. Tasks have deterministic correct answers. There is no mechanism for the agent to express uncertainty or abstain. No unanswerable questions are included. This contrasts with [[agent-output-beam|BEAM]]'s explicit Abstention ability (200 questions).

### Write-Path Behavior — 55%
The most significant partial coverage of write-path behavior in any benchmark to date. Task success depends on what the agent chose to store from prior sessions. However, write quality is evaluated only *indirectly* through downstream task performance -- there is no intrinsic evaluation of what was written, whether it was complete, whether it was well-structured, or whether the right abstractions were chosen. The `Update()` mechanism is a black box evaluated only by its downstream effects. Still, this is the first benchmark where write-path failures are visible in results, which is a major advance.

### Consolidation Quality — 25%
The benchmark compares 0D (no consolidation), 1D (flat consolidation), and 2D (structured consolidation) approaches. The finding that 1D consolidation often *hurts* compared to 0D raw history is an indirect measure of consolidation quality. But there is no direct evaluation of consolidation -- no measurement of compression ratio, information loss, or abstraction quality. The "representation mismatch" finding (compressed memory loses task-critical details) is a consolidation quality observation, but it emerges from failure analysis rather than systematic measurement.

### Proactive/Contextual Recall — 10%
All recall in MemoryArena is query-driven (the task agent asks its memory system for relevant information). There is no evaluation of proactive memory surfacing (the system volunteering information the agent didn't ask for). The `Retrieve()` function is always explicitly invoked.

### Relationship/Graph Reasoning — 30%
Travel Planning tests constraint graphs (participant relationships, shared activities, comparative preferences) up to depth 4. Shopping tests compatibility chains across 6 products. These implicitly require some graph reasoning. Mem0-g and GraphRAG are tested as 2D structured approaches. But graph structure is not the evaluation target -- it is a property of some tested systems. No evaluation of whether graph-based memory representations are more accurate or complete.

### Agentic Task Performance — 95%
This is the benchmark's raison d'etre. Every task requires the agent to take actions in an environment (browse products, search the web, plan itineraries, verify proofs), acquire information from those actions, and use stored information to complete dependent tasks. This is the most realistic agentic memory evaluation available. The only deduction is that the environments are simulated (WebShop, BrowseComp-Plus) rather than fully open-ended.

---

## 6. Relevance to claude-memory

### Could we run against this?

Not directly in its current form. MemoryArena requires:
1. An agentic framework that can execute web browsing, shopping, search, and formal reasoning tasks
2. Integration with WebShop and BrowseComp-Plus environments
3. A task agent (GPT-5.1-mini in the paper) that interfaces with our memory system via `Retrieve()`/`Update()` API

Our system does not have an agentic execution framework -- it is a personal memory MCP server for Claude Code sessions. The benchmark tests a fundamentally different use case: autonomous agent task execution across episodes, not human-AI conversational memory.

### What would it reveal if adapted?

If we adapted the paradigm (not the specific tasks), it would reveal:

1. **Write-path quality under agentic conditions**: Does our `remember()` capture the right information from complex action traces? The benchmark's key finding -- that memory systems lose task-critical details during consolidation -- directly challenges our sleep pipeline's compression approach.

2. **Retrieval under causal dependency**: Can our RRF retrieval find information that is relevant because of *causal connection to the current task* rather than *semantic similarity to the query*? This is where PPR-over-edges (#2) should shine, and where pure vector + BM25 is weakest.

3. **Belief drift in our system**: How does recall accuracy degrade over sequences of dependent operations? Our sleep pipeline (NREM consolidation, edge building) may introduce exactly the "representation mismatch" MemoryArena identifies: compressed memories that lose the precise details needed for downstream reasoning.

### Adaptation needed

A meaningful adaptation would be:
- Design 10-20 multi-step tasks using Claude Code's actual capabilities (file editing, code analysis, research across sessions)
- Make each step depend on information acquired in earlier steps
- Measure whether the memory system preserves sufficient information across `/sleep` cycles
- Track SR@k decay curves to detect belief drift

This is feasible as a manual evaluation protocol and could be formalized into the CMA behavioral probes (#10). The POMDP framing suggests a specific probe: give Claude a multi-step task, run sleep between steps, measure whether the post-consolidation memory preserves the "sufficient statistics" (the minimum information needed for correct execution of the next step).

---

## 7. Insights Worth Stealing

### Rank 1: SR@k decay curves as a diagnostic tool (Low effort, High impact)

Measuring success rate at increasing dependency depth is a simple, general diagnostic that we could apply to any multi-step evaluation. It reveals whether failure is catastrophic (early collapse) or gradual (belief drift), which has different remedies. We could implement SR@k measurement in our CMA probes (#10) by creating dependency-chained test questions.

### Rank 2: POMDP framing for memory failures (Zero effort, High conceptual impact)

The insight that memory failures are belief-state estimation errors that compound across sessions is immediately applicable to how we think about our system. Our sleep pipeline should be evaluated not by "did it preserve information" but by "does it preserve sufficient statistics for belief tracking." This reframes consolidation quality: the goal is not compression efficiency but state-estimation fidelity. This changes nothing in the code but changes how we evaluate whether sleep is working.

### Rank 3: Representation mismatch as a first-class failure mode (Low effort, Medium impact)

The finding that consolidated/compressed memory "may not align well with in-context learning" is a direct warning about our sleep pipeline. When NREM creates summaries, those summaries may be harder for the model to use than the raw memory content, even if they contain the same information. This suggests we should evaluate sleep quality not just by coverage (did we capture everything?) but by usability (does the compressed form work as well in `recall()` prompts?).

### Rank 4: 0D vs 1D vs 2D structure taxonomy (Zero effort, Medium impact)

The paper's clean taxonomy (0D = no structure, 1D = flat consolidation, 2D = graph/tree) maps directly onto our architecture. We are currently 0D for vector search, 1D for BM25 (curated field weighting), and 2D for edges (novelty-scored adjacency expansion). The finding that 2D does not consistently outperform 0D warns against over-engineering the graph component -- the edge graph must provide measurably better retrieval, not just theoretical elegance.

### Rank 5: Task-agent / memory-system training mismatch (Zero effort, High diagnostic value)

The paper identifies that task agents and memory systems are "not jointly optimized," limiting the agent's ability to formulate effective queries and integrate retrieved information. In our system, this manifests as: Claude (the task agent) is not trained to use `recall()` output optimally. The reading strategy optimization (#12) and our curated recall (Sonnet subagent synthesis) both address this mismatch from the reading side. MemoryArena validates that this mismatch is a dominant failure mode, not a minor inefficiency.

### Rank 6: Latency benchmarking as a standard evaluation dimension (Low effort, Low-medium impact)

Including latency alongside accuracy is good practice. Our system should track per-recall latency (already partially logged in events). The finding that memory systems add 1.5-2x latency without performance gains is a cautionary benchmark for any complexity we add.

---

## 8. What's Not Worth It

### Running the actual benchmark

MemoryArena evaluates autonomous agent task execution in simulated environments (WebShop, BrowseComp-Plus). Our system is a personal memory MCP for a coding assistant. The specific tasks (product compatibility, travel itineraries, web search aggregation, proof verification) are irrelevant to our use case. Building the infrastructure to run these tasks would be a significant engineering effort with no direct return.

### The formal reasoning domain specifics

The math and physics derivation chain tasks, while impressive (PhD-expert curated), test a capability (multi-step formal proof verification) that is orthogonal to personal memory. The insights about belief drift and retrieval quality transfer; the specific domain does not.

### Optimizing for PS/sPS metrics

The partial-credit metrics (PS, sPS) were designed for MemoryArena's specific subtask structure. They don't transfer to our evaluation needs. SR@k as a decay diagnostic transfers well; PS as a partial-credit scheme does not.

### Pursuing the 2D graph structure results

MemoryArena's finding that GraphRAG (2D) does not reliably outperform BM25 (0D) could be read as an argument against investing in our edge graph (#2). But this would be a misapplication: MemoryArena's graph evaluation is limited to GraphRAG's specific query-focused summarization approach, not PPR-based traversal. The benchmark does not test the graph traversal mechanisms we plan to implement. The result is informative about GraphRAG specifically, not about graph-based memory in general.

---

## 9. Key Takeaway

MemoryArena is the first benchmark that makes write-path quality load-bearing: if the agent stores the wrong things or stores them in a lossy format, dependent future tasks fail. The central finding is sobering -- models near-saturated on recall benchmarks (LoCoMo, LongMemEval) achieve mean 16% success on agentic memory tasks, and external memory systems often make things worse by introducing representation mismatches between raw history and compressed/structured memory. The POMDP formalization elevates this from an empirical observation to a theoretical framework: memory failures are belief-state estimation errors that compound across sessions. For claude-memory, the most actionable insight is that consolidation quality should be measured by state-estimation fidelity (does the compressed memory preserve sufficient statistics for downstream tasks?) rather than compression ratio or coverage. The benchmark's SR@k decay curves provide a reusable diagnostic pattern for any multi-step evaluation we build.

---

## 10. Impact on Implementation Priority

### Priority #1 (RRF fusion): No change.
MemoryArena shows text-embedding-3-small RAG achieving highest overall SR (0.23), validating semantic retrieval as the foundation. BM25 RAG is competitive (0.19 SR), and the two together should outperform either alone. The benchmark does not test RRF fusion directly but confirms both channels' independent value.

### Priority #2 (Relationship edges + PPR): Slight caution, no reorder.
GraphRAG (2D) does not consistently outperform BM25 (0D) in MemoryArena. This is a data point against the assumption that graph structure automatically improves retrieval. However, GraphRAG's approach (query-focused summarization) is fundamentally different from PPR-based traversal. The result counsels empirical validation of PPR's value before deep investment, but does not argue against the approach. The Travel Planning domain's dependency depth-4 chains are exactly where PPR should provide value -- yet GraphRAG scored 0.00 SR there too. Worth understanding *why* before assuming PPR will do better.

### Priority #3 (Contradiction detection + temporal invalidation): No change.
Not tested by MemoryArena. The benchmark assumes stable information. This remains the "hardest unsolved problem" per [[agent-output-memoryagentbench|MemoryAgentBench]].

### Priority #5 (Sleep/consolidation pipeline): Increased scrutiny, no reorder.
The "representation mismatch" finding is a direct challenge. Our NREM summaries, REM topic synthesis, and gestalt layers all produce compressed representations. MemoryArena demonstrates that compression can lose task-critical details and produce representations that "may not align well with in-context learning." The implication: sleep consolidation needs a quality gate that evaluates whether the compressed representation preserves sufficient downstream-task statistics. This is not currently part of the pipeline.

### Priority #10 (CMA behavioral probes): Reinforced, potential reorder upward.
SR@k decay curves over dependency-chained probes would be a powerful diagnostic. The POMDP framing gives CMA probes a theoretical anchor: test whether memory preserves sufficient belief-state statistics across consolidation cycles. Consider elevating #10 above #9 (multi-angle indexing) based on MemoryArena's demonstration that decay diagnostics reveal failure modes invisible to point-in-time evaluation.

### Priority #12 (Reading strategy optimization): Reinforced.
The task-agent/memory-system training mismatch finding validates that reading quality is a first-class concern. Even perfect retrieval fails if the task agent cannot effectively utilize retrieved information. Our staged curated recall (Sonnet subagent) addresses this, and the MemoryArena result provides empirical backing for continued investment.

### Priority #14 (Retrieval feedback loop): No change.
Not directly tested by MemoryArena. The benchmark evaluates end-to-end task success, not retrieval quality in isolation. But the belief-drift interpretation implies that feedback on retrieval quality (did the retrieved memory actually help?) could be used to detect and correct drift before it compounds.

### New consideration: Write-path quality evaluation
MemoryArena's most novel contribution is making write-path quality visible through downstream task performance. This suggests a new evaluation dimension not currently in our priority list: measure whether `remember()` captures the right information by testing downstream `recall()` success on dependent queries. This could be integrated into #10 (CMA probes) as a "write-then-read" probe pattern.

---

## 11. See Also

- [[agent-output-beam]] -- BEAM benchmark (10M tokens, 10 memory abilities). Complementary: BEAM tests recall quality at extreme scale; MemoryArena tests agentic memory application at moderate scale. Together they cover scale+recall and depth+application.
- [[agent-output-locomo-plus]] -- LoCoMo-Plus benchmark (cognitive memory). MemoryArena's causal dependencies are a different kind of difficulty from LoCoMo-Plus's semantic disconnects. Both expose failures invisible to basic recall tests.
- [[agent-output-tremu]] -- TReMu benchmark (temporal reasoning). MemoryArena has weak temporal coverage; TReMu fills this gap completely. The two are complementary.
- [[agent-output-memoryagentbench]] -- MemoryAgentBench (multi-hop selective forgetting). MemoryArena does not test forgetting or contradiction. MemoryAgentBench's universal catastrophic failure (max 7%) on multi-hop forgetting remains the strongest argument for Priority #3.
- [[agent-output-longmemeval]] -- LongMemEval. The "50% reading failures" finding from LongMemEval is reinforced by MemoryArena's "training mismatch" finding: even when retrieval works, the task agent fails to utilize the information.
- [[agent-output-ragas]] -- RAGAS evaluation framework. The Aspect-Critic pattern could be applied to evaluate MemoryArena-style write-path quality: "Did the memory capture the information needed for the next subtask?"
- [[architecture-decisions]] -- Our architecture. The POMDP framing suggests a new evaluation criterion: "sufficient statistics preservation" through consolidation.
- [[phase-13-synthesis]] -- Community systems. The "write-time quality gates" pattern (4/9 repos) is directly relevant to MemoryArena's implicit write-path evaluation. PsychMem's interference detection and PLTM-Claude's 4-judge jury address the consolidation quality gap MemoryArena exposes.
