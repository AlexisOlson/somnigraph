# MEMTRACK: Evaluating Long-Term Memory and State Tracking in Multi-Platform Dynamic Agent Environments -- Analysis

*Phase 15, 2026-03-06. Analysis of arXiv:2510.01353 (2025).*

---

## 1. Paper Overview

**Paper**: Darshan Deshpande, Varun Gangal, Hersh Mehta, Anand Kannappan, Rebecca Qian, Peng Wang. Patronus AI. arXiv:2510.01353, submitted October 1, 2025.

**Venue**: Accepted to NeurIPS 2025 SEA (Software Engineering for AI) Workshop. Licensed CC BY 4.0.

**Problem**: Existing memory benchmarks evaluate agents in single-platform, conversational settings. Real enterprise workflows involve agents operating across multiple platforms simultaneously — issue trackers, chat, version control, code editors — where information arrives asynchronously, conflicts across sources, and must be tracked through state changes over time. No prior benchmark evaluates memory in this multi-platform, state-tracking regime.

**Core contribution**: MEMTRACK, a benchmark of 47 long-context datapoints simulating realistic software engineering workflows across Linear (issue tracking), Slack (communication), and Gitea (version control). Each datapoint provides a chronologically platform-interleaved timeline with noisy, conflicting, cross-referring information and codebase comprehension requirements. Three metrics — Correctness, Efficiency, Redundancy — capture memory effectiveness beyond simple QA. Experiments with GPT-5 and Gemini-2.5-Pro across three memory configurations (NoMem, Mem0, Zep) reveal that the best model achieves only 60% Correctness, and that adding external memory backends (Mem0, Zep) provides no significant improvement.

**Scale**: 47 curated datapoints. Models: GPT-5, Gemini-2.5-Pro. Memory backends: NoMem (baseline), Mem0, Zep. Platforms: Linear, Slack, Gitea (containerized). Multi-turn sequential questioning with follow-ups.

**Code/data availability**: The MEMTRACK benchmark dataset does not appear to be openly available on GitHub as of March 2026. Patronus AI's GitHub organization hosts other benchmarks (TRAIL, FinanceBench) but not MEMTRACK. The benchmark infrastructure connects to Patronus AI's broader "Generative Simulators" product line for enterprise agent evaluation.

---

## 2. Benchmark Design

### Platform Architecture

MEMTRACK simulates a software development organization using three interconnected platforms:

| Platform | Role | Information Type |
|----------|------|-----------------|
| **Linear** | Issue/project tracking | Task assignments, status changes, priority updates, sprint planning |
| **Slack** | Communication | Discussion threads, decision announcements, informal context, conflicting instructions |
| **Gitea** | Version control | PRs, code changes, commit messages, file-system state |

Agents access these platforms through tool calls, creating a realistic multi-source information retrieval problem. Timeline events are chronologically interleaved across platforms — a Slack discussion may reference a Linear issue that spawned a Git PR, with status updates arriving out of order across all three.

### Four-Stage Pipeline

1. **Task generation**: Scenarios created via three complementary methodologies (see below)
2. **Event injection**: Curated event timelines loaded into containerized Linear, Slack, and Gitea instances
3. **Agent execution**: Sequential question injection with monitoring — agents access platforms through tool calls
4. **Performance evaluation**: LLM-as-judge scoring on Correctness; automated measurement of Efficiency and Redundancy

### Dataset Construction (Three Methodologies)

**Bottom-Up (Open-Source Repositories)**: Real scenarios derived from closed issues/PRs in popular open-source repositories. Actual PR solutions are used, with contextual distractions simulating typical SWE noise. Works backward from known outcomes.

**Top-Down (In-House Experts)**: Former product and engineering team members document real-world workflows, capturing how problems are scoped, communicated, and solved across platforms.

**Hybrid**: Expert-provided high-level ideation combined with LLM-assisted data generation. Annotators prompt LLMs (Claude-4-Sonnet mentioned) with task context, motivation, end goals, and codebase to produce event sequences. Iterative refinement ensures quality.

### Memory Capabilities Taxonomy

The paper explicitly tests three memory capabilities:

| Capability | What It Tests |
|-----------|--------------|
| **Acquisition** | Can the agent extract and store relevant information from multi-platform event streams? |
| **Selection** | Can the agent retrieve the right information from stored memory when answering questions? |
| **Conflict Resolution** | Can the agent resolve contradictions between information from different platforms/timepoints? |

### Evaluation Metrics

| Metric | Definition | Assessment Method |
|--------|-----------|-------------------|
| **Correctness** | Whether the agent produces accurate answers, averaged per-example across questions | LLM-as-judge; brief phrase outputs enable direct and approximate matching |
| **Efficiency** | Resource/time optimization in tool-calling patterns | Automated measurement of tool-call sequences |
| **Redundancy** | Unnecessary repeated tool invocations accessing the same information | Automated measurement of duplicate/redundant calls |

Additional measurements include **cross-platform entropy** (how diversely an agent accesses different platforms) and **tool-calling extent** (total volume and breadth of tool use).

### Multi-Turn Question Structure

Each datapoint involves sequential questions with follow-ups. Initial questions target information from the event timeline; follow-up questions probe whether the agent retained and can apply that information without re-accessing the source platforms. Performance decay on follow-ups is a key diagnostic.

---

## 3. Key Claims and Evidence

### Claim 1: GPT-5 achieves only 60% Correctness — the ceiling is low

**Evidence**: GPT-5+NoMem scored 0.601 Correctness, 0.667 Efficiency, 0.206 Redundancy. This is the best configuration tested.

**Quality**: Strong. 47 datapoints is small but carefully curated. The result is consistent across memory configurations (adding Mem0 or Zep doesn't meaningfully change Correctness). The headline number is robust.

**Implication**: Even frontier models cannot reliably reason over multi-platform event timelines at enterprise scale. This is not a retrieval failure alone — it is a reasoning-over-complex-state failure.

### Claim 2: External memory backends (Mem0, Zep) provide no significant improvement

**Evidence**:

| Configuration | Correctness | Efficiency | Redundancy |
|--------------|-------------|------------|------------|
| GPT-5+NoMem | 0.601 | 0.667 | 0.206 |
| GPT-5+Mem0 | 0.610 | 0.656 | 0.214 |
| GPT-5+Zep | 0.601 | 0.660 | 0.214 |
| Gemini-2.5-Pro+NoMem | 0.144 | 0.638 | 0.237 |
| Gemini-2.5-Pro+Mem0 | 0.118 | 0.658 | 0.240 |
| Gemini-2.5-Pro+Zep | 0.140 | 0.662 | 0.238 |

**Quality**: The null result is striking. Mem0 and Zep add marginal noise to GPT-5 scores and may actually *hurt* Gemini-2.5-Pro (from 0.144 to 0.118 with Mem0). The paper reports that agents "prefer re-accessing information directly from the platforms rather than leveraging the memory component." This is consistent with [[agent-output-amemgym]]'s finding that write failures dominate memory system performance — if agents don't effectively write to memory, memory backends are inert.

**Implication**: The bottleneck is not memory *storage* or *retrieval* quality. It is the agent's ability to (a) decide what to store, (b) decide when to consult memory vs. re-access source platforms, and (c) reason over retrieved results. This validates the "write-path matters more than read-path" thesis from [[agent-output-memory-arena]] and [[agent-output-amemgym]].

### Claim 3: Performance decays on follow-up questions

**Evidence**: The paper reports consistent performance decay across methods on follow-up turns, indicating agents struggle to maintain multi-turn context and cross-platform state.

**Quality**: Moderate. The specific magnitude of decay is not available from the accessible summaries. But the directionality is clear and consistent with BEAM's finding that memory degrades with distance.

### Claim 4: Low tool-call entropy limits performance

**Evidence**: Cross-platform entropy measurements show agents under-diversify their tool use — they tend to access familiar platforms repeatedly rather than casting a net across all information sources.

**Quality**: Moderate. This is an interesting diagnostic but its causal relationship to Correctness is correlational, not established.

### Claim 5: Agents need practice with memory tools

**Evidence**: The paper concludes that agents require training/practice using memory tools to improve large context reasoning and multi-turn question understanding.

**Quality**: Weak as stated (aspirational rather than demonstrated). But directionally consistent with the RL memory literature (Memory-R1, Mem-alpha) which shows that RL-trained memory policies outperform untrained ones by 10-15%.

---

## 4. Standout Feature

**What makes MEMTRACK different from the other 8 benchmarks in our survey**: It is the only benchmark that evaluates memory across **genuinely separate platforms** with **containerized environments and tool-call access**. Every other benchmark — BEAM, LoCoMo-Plus, TReMu, MemoryArena, AMemGym, AMA-Bench, Memory-T1 — operates within a single information channel (usually conversation history or text documents). MEMTRACK forces agents to *navigate* to information across Linear, Slack, and Gitea using tool calls, synthesize state from asynchronous cross-platform events, and handle the noise/contradiction that multi-source information naturally generates.

This is conceptually closer to how real enterprise agents operate than any prior benchmark. The information is not handed to the agent in a prompt — the agent must go get it, from the right platform, at the right time, and reconcile what it finds.

The tradeoff: 47 datapoints is an order of magnitude smaller than BEAM (2,000 questions), MemoryArena (736 tasks), or AMemGym (hundreds of episodes). The benchmark sacrifices statistical power for ecological validity.

---

## 5. Competency Coverage Ratings

| Dimension | Coverage | Justification |
|-----------|----------|---------------|
| **Information Retrieval** | 75% | Core competency. Tests cross-platform retrieval through tool calls. But retrieval is mediated by agent tool-use decisions, not isolated as a retrieval-system metric. |
| **Multi-Session Reasoning** | 55% | Multi-turn within a session (initial questions + follow-ups). Not true multi-session (agent does not persist across separate episodes). Below [[agent-output-memory-arena]] and [[agent-output-amemgym]] on this dimension. |
| **Knowledge Update / Contradiction** | 70% | Explicitly listed as "conflict resolution" capability. Timelines contain conflicting information across platforms. However, no isolated contradiction detection metric — folded into Correctness. |
| **Temporal Reasoning** | 50% | Chronologically interleaved timelines require understanding event ordering. But no dedicated temporal reasoning tasks or metrics (contrast with [[agent-output-tremu]] which isolates temporal abilities). |
| **Abstention / Confidence** | 15% | Not explicitly evaluated. No questions designed to test whether agents correctly abstain when information is unavailable or contradictory. |
| **Write-Path Behavior** | 60% | The Mem0/Zep experiments implicitly evaluate write-path — agents must decide what to store. The key finding (agents don't use memory tools effectively) is a write-path result. But no isolated write-path metrics or diagnostic decomposition. |
| **Consolidation Quality** | 5% | Not tested. The benchmark evaluates within-task performance, not how agents consolidate or compress information over time. |
| **Proactive / Contextual Recall** | 30% | Follow-up questions test whether agents retain context without re-querying. But this is passive retention, not proactive surfacing of relevant context. |
| **Relationship / Graph Reasoning** | 40% | Cross-referencing across platforms requires tracking relationships (issue X references PR Y discussed in Slack thread Z). But no explicit graph structure or multi-hop reasoning evaluation. |
| **Agentic Task Performance** | 80% | The strongest dimension. Agents must navigate real tool interfaces, decide which platform to query, manage tool-call budgets, and synthesize answers from multi-source evidence. This is genuinely agentic evaluation. |

**Aggregate**: ~48%. Strong on agentic execution and cross-platform information synthesis. Weak on consolidation, abstention, and proactive recall. Missing temporal isolation and multi-session persistence.

---

## 6. Relevance to claude-memory

### Could we run against MEMTRACK?

**Not directly, and the effort would be high.** MEMTRACK requires:
1. Containerized Linear, Slack, and Gitea instances with loaded event timelines
2. A tool-call interface layer connecting the agent to these platforms
3. The benchmark's specific 47 datapoints (not publicly released)
4. The four-stage evaluation pipeline

Our system (claude-memory) operates within a single MCP context (Claude Code sessions). We do not have multi-platform tool-call routing. Running MEMTRACK would require either (a) access to the Patronus AI evaluation infrastructure or (b) significant engineering to replicate the containerized platform setup.

### What would it reveal about claude-memory?

Even if adapted, MEMTRACK primarily tests the **agent's** ability to use memory tools, not the **memory system's** quality. The paper's core finding — that Mem0 and Zep provide no improvement — is a comment on agent-side memory utilization, not storage/retrieval quality. claude-memory's differentiators (decay, consolidation, feedback loops, edge graph) operate at a level MEMTRACK does not measure.

However, MEMTRACK would stress-test one important property: **can recall() surface the right information when the query involves cross-referencing multiple event sources?** Our RRF hybrid retrieval would need to handle queries like "what was the decision about issue X discussed in Slack and updated in Linear?" This is multi-hop retrieval with platform-source disambiguation — a realistic challenge for our BM25+vector+edge expansion pipeline.

### Adaptation needed

- **Low-effort proxy**: Extract the "conflict resolution" question subset and adapt as standalone contradiction detection probes. Feed synthetic multi-source event histories into memory, then test recall correctness. This isolates the retrieval dimension without the full platform infrastructure.
- **Medium-effort**: Build a synthetic "multi-source remember + recall" test — ingest events tagged with source platforms, then query for information requiring cross-source synthesis. Tests whether our edge graph and RRF handle source-diversity well.
- **High-effort**: Full platform containerization. Not recommended given the 47-datapoint scale and the fact that the core finding (agents don't use memory tools well) is an agent-side problem, not a memory-system problem.

---

## 7. Insights Worth Stealing

Ranked by effort/impact:

### 1. Cross-platform entropy as a diagnostic metric (Low effort / Medium impact)
Measuring how diversely an agent accesses different information *sources* (not just information *types*) is a transferable idea. In our context, this maps to tracking which memory categories, themes, or time periods a recall() query draws from. If our retrieval consistently pulls from the same cluster/category, that is a diversity problem analogous to low cross-platform entropy. Could be added to recall_meta logging with minimal effort.

### 2. The "agents don't use memory tools" finding as design constraint (Zero effort / High conceptual impact)
This is the paper's most important lesson for us. If we expose claude-memory as passive tools (remember/recall), the agent may under-utilize them — exactly as GPT-5 under-utilizes Mem0 and Zep. Our hook-driven auto-capture (correction/decision/gotcha patterns), startup_load briefing, and recall_feedback enforcement are all defenses against this failure mode. The MEMTRACK result validates our architecture's emphasis on *making memory use automatic* rather than relying on the agent to decide when to use memory.

### 3. Chronologically interleaved multi-source timelines as test data (Medium effort / Medium impact)
The idea of constructing timelines where events from different sources are interleaved chronologically, with deliberate noise and conflicts, is a good pattern for stress-testing our ingestion pipeline. We could construct synthetic NREM sleep inputs with this structure — interleaved events from different "sessions" with contradictions — to test whether our contradiction detection and temporal invalidation hold up.

### 4. Follow-up question decay as diagnostic (Low effort / Low-medium impact)
Testing whether recall performance degrades when the same information is queried in a follow-up turn (without re-accessing the source) maps directly to testing whether our recall_feedback and Hebbian co-retrieval boost are working. If we recall a memory, provide feedback, then query again, does the boosted memory surface faster? This is a micro-benchmark we could build.

### 5. Three-methodology dataset construction (Medium effort / Medium impact)
The bottom-up (from real repository issues), top-down (expert-documented workflows), and hybrid (expert ideation + LLM synthesis) approach is a good template for building our own ClaudeMemEval benchmark (#10 on implementation priority). Bottom-up: mine real session transcripts for memory-relevant episodes. Top-down: manually design failure cases from known gaps. Hybrid: seed with real patterns, use LLM to generate variations.

---

## 8. What's Not Worth It

**Replicating the full platform infrastructure**: 47 datapoints across containerized Linear/Slack/Gitea is high engineering cost for modest statistical power. The benchmark's value is in its conceptual framing, not its scale.

**Chasing the Gemini-2.5-Pro results**: Gemini scores (0.118-0.144 Correctness) are catastrophically low, likely due to tool-calling failures rather than memory issues. These results tell us about Gemini's agentic capabilities, not about memory architecture.

**Treating the Mem0/Zep null result as evidence against memory backends in general**: The null result is about *these agents' ability to use these tools*, not about the fundamental value of external memory. Our system's hooks, startup briefing, and feedback loops are specifically designed to address the agent utilization problem that MEMTRACK exposes.

**Efficiency and Redundancy metrics in isolation**: These measure tool-call optimization, which is model-specific and prompt-sensitive. Not transferable to our system where retrieval is a single recall() call, not a multi-step tool-call sequence.

---

## 9. Key Takeaway

MEMTRACK's lasting contribution is demonstrating that **multi-platform state tracking is a genuinely harder problem than single-context memory**, and that **off-the-shelf memory backends fail not because of retrieval quality but because agents do not use them effectively**. The 60% ceiling for GPT-5 — with or without memory augmentation — is a sobering result that suggests the bottleneck in agent memory is at the *utilization* layer, not the *storage* layer. For claude-memory, this validates our architectural emphasis on automatic memory integration (hooks, startup briefing, forced feedback) rather than passive tool exposure. The benchmark is too small (47 datapoints) and too infrastructure-heavy (containerized platforms) for us to run directly, but its cross-source conflict resolution concept and the diagnostic of follow-up decay are both adaptable to our evaluation work. The most transferable insight: measure whether your memory system is actually *being used*, not just whether it *could work if used*.

---

## 10. Impact on Implementation Priority

| Priority Item | Impact | Notes |
|--------------|--------|-------|
| **#1 RRF fusion** | Slight reinforcement | Cross-platform queries require multi-source synthesis — RRF's multi-channel approach handles this naturally. No change needed. |
| **#2 Relationship edges** | Moderate reinforcement | Cross-platform cross-referencing (issue → PR → Slack thread) is a relationship traversal problem. Our edge graph would help here. |
| **#3 Contradiction detection** | Strong reinforcement | Explicitly tested as "conflict resolution" capability. The finding that agents fail at resolving cross-platform contradictions strengthens the case for our temporal invalidation approach. |
| **#5 Sleep / consolidation** | No direct impact | MEMTRACK tests within-task performance, not consolidation quality. |
| **#10 CMA behavioral probes** | Moderate impact — new test type | Cross-source conflict resolution probes are a new category worth adding to our ClaudeMemEval design. Adapt MEMTRACK's conflict scenarios as synthetic multi-source contradiction tests. |
| **#12 Reading strategy** | Slight reinforcement | Follow-up decay suggests that how retrieved information is presented affects retention across turns. |
| **#14 Retrieval feedback** | Strong validation | The paper's key finding — agents don't use memory tools — is exactly the problem feedback loops solve. Forced recall_feedback ensures memory is evaluated after every use, creating the practice signal that MEMTRACK identifies as missing. |

**No reordering needed.** MEMTRACK reinforces existing priorities (#3 contradiction, #14 feedback) and adds a test-design concept for #10, but does not shift the ordering. The main insight is architectural (make memory use automatic) rather than algorithmic.

---

## 11. See Also

- [[agent-output-beam]] — BEAM benchmark (ICLR 2026). 10M-token coherent conversations, 10 memory abilities. Larger scale (2,000 questions) but single-context. Tests the same models but in conversational rather than multi-platform settings.
- [[agent-output-memory-arena]] — MemoryArena (Feb 2026). 736 agentic multi-session tasks. Shares the "write-path matters" finding. True multi-session persistence testing (MEMTRACK is multi-turn within session). POMDP formalization complementary to MEMTRACK's multi-platform framing.
- [[agent-output-amemgym]] — AMemGym (ICLR 2026). On-policy evaluation with diagnostic decomposition. Confirms write failures dominate. Their `.338` write-failure rate predicts MEMTRACK's null result for Mem0/Zep.
- [[agent-output-ama-bench]] — AMA-Bench. Agentic trajectories with needle protocol. Their graph-hurts finding (-24.6%) is relevant: MEMTRACK similarly finds that adding memory infrastructure does not help when the agent cannot utilize it.
- [[agent-output-tremu]] — TReMu. Temporal reasoning benchmark. Provides the temporal dimension that MEMTRACK lacks — complementary for evaluation coverage.
- [[practical-mcp-comparison]] — Phase 13 comparison of 8 MCP memory servers. The hook-driven auto-capture finding is directly relevant to MEMTRACK's "agents don't use memory tools" conclusion.
- [[agent-output-locomo-plus]] — LoCoMo-Plus. Cognitive memory with semantic gaps. The Level-2 cognitive memory (implicit constraints) tests a dimension MEMTRACK does not reach.
- [[agent-output-memory-t1]] — Memory-T1. RL temporal selection. Their approach of training agents to *use* memory tools is the direct answer to MEMTRACK's identified gap.
