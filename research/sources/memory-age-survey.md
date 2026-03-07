# "Memory in the Age of AI Agents" -- Analysis

*Generated 2026-02-20 by Opus agent reading arXiv:2512.13564*

---

## Paper Overview

**Paper**: Yuyang Hu, Shichun Liu, Yanwei Yue, et al. (47 authors). "Memory in the Age of AI Agents." arXiv:2512.13564, December 2025 (v2: January 2026). CC BY 4.0.

**Problem addressed**: The agent memory field has become increasingly fragmented. Existing works differ substantially in motivations, implementations, and evaluation protocols, while loosely defined terminology has obscured conceptual clarity. No prior survey has cleanly separated agent memory from LLM memory, RAG, and context engineering.

**Core contribution**: A three-dimensional taxonomy (Forms x Functions x Dynamics) that provides a unifying lens across 200+ papers. The survey explicitly delineates agent memory's boundaries against related concepts and compiles the most comprehensive paper list for the field to date.

**Scale**: 200+ papers cataloged across 9 cross-cutting categories (3 forms x 3 functions). Companion GitHub repository (`Shichun-Liu/Agent-Memory-Paper-List`) actively maintained. v2 (January 2026) added several recent works including MemRL, Agentic Memory, and EverMemOS.

---

## Taxonomy Analysis

### Their Framework: Forms x Functions x Dynamics

**Forms** (how memory is stored):
- **Token-level**: Discrete, human-readable units (text, JSON, markdown) in external stores. Subdivided into flat (1D linear logs), planar (2D tree/graph), and hierarchical (3D multi-layer pyramids).
- **Parametric**: Knowledge embedded in model weights via internal editing (ROME, MEMIT) or external adapters (LoRA).
- **Latent**: Continuous representations in hidden states, KV caches, or learned embeddings.

**Functions** (what memory serves):
- **Factual**: Declarative knowledge -- user preferences, environmental facts, entity relationships. Organized as graphs or structured records.
- **Experiential**: Procedural knowledge from task execution -- case-based trajectories, strategy-based workflows, skill-based code, and hybrids.
- **Working**: Active scratchpad managing task-specific context -- input condensation, observation abstraction, state consolidation, hierarchical folding.

**Dynamics** (how memory changes over time):
- **Formation**: Extracting and encoding information via summarization, distillation, or structuring.
- **Evolution**: Consolidation (merging/generalizing), updating (resolving contradictions), forgetting (pruning low-utility entries).
- **Retrieval**: Timing determination, query construction, selection strategies (lexical, semantic, graph, generative, hybrid), post-processing.

### Mapping to Our Framework

| Their Concept | Our Equivalent | Notes |
|---------------|---------------|-------|
| **Token-level form** | Our entire architecture (SQLite + FTS5 + sqlite-vec) | We operate exclusively in the token-level form. This is correct for our use case -- claude-memory is a human-curated system where interpretability and editability are essential. The survey validates that token-level is the dominant form for agent memory systems. |
| **Parametric form** | Not applicable | Model editing (ROME, MEMIT) and adapter-based approaches (LoRA) are irrelevant to our design. We don't modify model weights. |
| **Latent form** | Not applicable | KV cache optimization, learned embeddings as memory -- these address a different problem (efficient inference) rather than persistent cross-session memory. |
| **Factual function** | Our `semantic` category | Direct mapping. User preferences, facts about the world, entity relationships. |
| **Experiential function** | Our `episodic` + `procedural` categories | Their "experiential" is broader than our split. Our distinction between episodic (what happened) and procedural (how to do things) is finer-grained. AWM's parameterized slots, Dynamic Cheatsheet's strategy extraction, and ExpeL's experiential learning all live in their "experiential" bucket. Our split is more actionable for retrieval. |
| **Working function** | Our `startup_load()` + context engineering | Working memory in their framework is the scratchpad for active reasoning -- not persistent. Maps to our startup briefing (what's in context) rather than stored memories. |
| **Formation** | Our `remember()` pipeline | Write path: content arrives, gets embedded, dedup-checked, stored. They call this "formation." |
| **Evolution** | Our `/sleep` + `consolidate()` + graded contradiction detection | Process path: memories are merged, contradictions resolved, layers built. Direct match. |
| **Retrieval** | Our `recall()` with RRF fusion | Read path: hybrid search, ranking, budget-constrained output. Direct match. |
| **Reflection / meta** (ours) | No direct equivalent | They have no category for memories *about* the memory system or for self-reflective insights. Our `reflection` and `meta` types represent a capability axis the survey doesn't address. |

### Key Structural Comparison

**Their 3x3 grid (Forms x Functions)** is orthogonal to our types. They ask "what form does the memory take?" independently of "what function does it serve?" This is a useful framing we haven't used explicitly -- a procedural memory could theoretically be stored as token-level, parametric, or latent. For our system, all memories are token-level, so the forms axis collapses to a single value and their functions axis maps to our types.

**Their Dynamics axis maps cleanly to our W-P-R** (Write-Process-Retrieve from Zhang et al.'s memory survey):
- Formation = Write
- Evolution = Process
- Retrieval = Retrieve

They don't add a new axis we're missing. However, their **sub-decomposition of Dynamics** is more granular than ours in some areas:

| Their Sub-step | Our Treatment | Gap? |
|---------------|---------------|------|
| Timing determination (when to retrieve) | Implicit -- recall is tool-invoked | No gap for human-in-the-loop; would matter for fully autonomous agents |
| Query construction (how to form the retrieval query) | Direct query passthrough | Minor gap -- query rewriting/expansion could improve recall quality (our #12 reading strategy partially addresses) |
| Generative retrieval (synthesize memory rather than look it up) | Not implemented | Interesting but premature for our scale. Worth watching. |
| Post-retrieval processing (re-rank, filter, compress) | RRF fusion + token budgeting | Covered |

### What They Add That We're Missing

**1. The "generative retrieval" concept**: Rather than retrieving stored memories, the system *generates* a context-specific memory representation dynamically. This is distinct from standard retrieval and could address the reading failure problem (our finding that ~50% of errors are reading failures). Instead of returning raw memories and hoping the main agent reads them correctly, a retrieval sub-agent could synthesize a targeted answer. This overlaps with our #12 (reading strategy optimization) but frames it differently.

**2. Explicit attention to the parametric-token synchronization problem**: When a system uses both parametric memory (fine-tuned weights) and token-level memory (external stores), keeping them consistent is an unsolved problem. Not relevant to us now, but important context as models gain built-in memory capabilities.

**3. The flat/planar/hierarchical sub-taxonomy for token-level memory**: Our architecture is currently flat (all memories in one table, no structural hierarchy in storage). Our planned layer column (detail/summary/gestalt) introduces a hierarchical dimension, but without the explicit graph structure they describe as "planar." This validates our planned addition of `memory_edges` (priority #2) as moving from flat to planar, and the sleep skill's layer generation as adding the hierarchical dimension.

---

## New Systems & Papers Identified

### Papers NOT in Our 22-Source Corpus That Look Relevant

| Paper | Year | Why Relevant | Priority |
|-------|------|-------------|----------|
| **Mem-alpha: Learning Memory Construction via RL** (arXiv:2509.25911) | 2025 | RL framework for training agents to manage memory. Features core/episodic/semantic components with multiple tools. Trained on 30k tokens, generalizes to 400k+. Directly addresses "memory automation" -- making write/process/retrieve decisions learnable rather than heuristic. | **High** -- challenges our human-in-the-loop assumption with evidence that learned policies outperform heuristics |
| **Memory-R1: Enhancing LLM Agents to Manage Memories via RL** (arXiv:2508.19828) | 2025 | Two-agent RL system: Memory Manager (ADD/UPDATE/DELETE/NOOP) + Answer Agent (pre-selects relevant entries). Fine-tuned with PPO and GRPO. | **High** -- validates that memory operations benefit from learned policies; the Memory Manager concept is relevant to our auto-capture design |
| **MemRL: Self-Evolving Agents via Runtime RL on Episodic Memory** | 2026 | Runtime RL for memory management. Appears in both experiential and working memory categories. | **Medium** -- newer, less information available, but the runtime learning angle is distinctive |
| **HippoRAG: Neurobiologically Inspired Long-Term Memory** (arXiv:2405.xxxxx) | 2024 | Hippocampus-inspired retrieval architecture. Cited frequently in the survey. | **Medium** -- directly relevant to our CLS-theory-informed design |
| **MemoryAgentBench** (ICLR 2026) | 2026 | New benchmark: four core competencies (accurate retrieval, test-time learning, long-range understanding, selective forgetting). Based on cognitive science memory theories. | **High** -- new benchmark we should evaluate against, especially the "selective forgetting" competency which no existing benchmark tests well |
| **Agentic Memory: Learning Unified Long-Term and Short-Term Memory** | 2026 | Unifying long-term and short-term memory in a single learnable framework. | **Medium** -- addresses the working/long-term boundary we handle with startup_load vs. recall |
| **MAGMA: Multi-Graph Agentic Memory Architecture** | 2026 | Multi-graph structure for memory. Could inform our planned `memory_edges` design. | **Medium** -- graph architecture details |
| **RGMem: Renormalization Group-based Memory Evolution** | 2025 | Physics-inspired approach to memory evolution across scales. | **Low-Medium** -- interesting theoretical framing for multi-layer consolidation |
| **SGMem: Sentence Graph Memory for Long-Term Conversational Agents** | 2025 | Sentence-level graph for conversational memory. | **Low** -- conversational focus, less relevant to our agentic coding use case |
| **MOOM: Maintenance, Organization and Optimization of Memory** | 2025 | Explicitly addresses the maintenance/organization problem. | **Medium** -- if it offers concrete optimization strategies for consolidation |
| **EverMemOS: Self-Organizing Memory Operating System** | 2026 | Memory as OS -- similar framing to MemGPT/Letta but self-organizing. | **Medium** -- may have new patterns for our memory lifecycle |
| **ComoRAG: Cognitive-Inspired Memory-Organized RAG** | 2025 | Cognitive science-grounded RAG with explicit memory organization. | **Low-Medium** -- RAG-focused but the cognitive framing may add |
| **CAM: Constructivist View of Agentic Memory** | 2025 | Constructivist epistemology applied to agent memory. | **Low** -- theoretical framing |
| **In Prospect and Retrospect: Reflective Memory Management** | 2025 | Reflective memory management -- directly relevant to our `/reflect` design. | **Medium** -- could inform reflection pipeline improvements |
| **Forgetful but Faithful: A Cognitive Memory Architecture and Benchmark for Privacy-Aware Generative Agents** (arXiv:2512.12856) | 2025 | Privacy-aware forgetting -- a dimension we haven't addressed. | **Low-Medium** -- not urgent for single-user system but forward-looking |
| **MemEvolve: Meta-Evolution of Agent Memory Systems** | 2025 | Meta-evolution of memory architectures themselves. | **Medium** -- if it addresses how memory *systems* should adapt over time, not just memories |
| **PRINCIPLES: Synthetic Strategy Memory** | 2025 | Extracting strategic principles from experience -- directly maps to our procedural memory consolidation. | **Medium** -- could inform sleep skill's procedure extraction |
| **LEGOMem: Modular Procedural Memory** | 2025 | Modular approach to procedural memory. | **Medium** -- if it solves the structured procedural representation gap we identified |
| **H2R: Hierarchical Hindsight Reflection** | 2025 | Hierarchical reflection -- may inform our layer generation. | **Medium** |

### Specifically for Our Identified Gaps

**Multi-channel retrieval**: HippoRAG and MAGMA both use graph-based retrieval alongside vector/keyword. Already well-covered in our corpus via Zep and Hindsight.

**Consolidation/sleep**: The survey cites CLS theory (which we already use as our foundation) and references several consolidation approaches, but no paper beyond our existing sources offers significantly new consolidation algorithms. The RL-based approaches (Mem-alpha, Memory-R1) represent a genuinely new angle -- using learned policies for consolidation decisions rather than heuristics.

**Temporal reasoning**: The survey mentions temporal decay mechanisms (recency score decreasing hourly by 0.995) and bi-temporal modeling, both of which we've already incorporated from Zep.

**Forgetting/decay**: MemEvolve and the "Forgetful but Faithful" paper address forgetting, but the survey itself notes this remains the most under-explored area -- consistent with our Phase 8 finding.

**Reading strategies**: Not explicitly addressed by the survey. Our finding that ~50% of errors are reading failures remains an insight the survey doesn't engage with deeply.

**Relationship edges/graphs**: MAGMA, SGMem, and the planar memory sub-taxonomy validate our priority #2 (relationship edges). No fundamentally new edge type beyond what Engram and CortexGraph already provided.

---

## Agreements & Disagreements with Our Findings

### Agreements (Strong)

**1. Multi-channel retrieval is non-negotiable**

The survey describes retrieval strategies as "lexical, semantic, graph, generative, or hybrid" and positions hybrid as the mature approach. Their discussion of Zep's triple retrieval, Hindsight's four-way parallel, and multiple systems using RRF fusion fully aligns with our consensus finding (6 sources). **No disagreement.**

**2. Summaries are routing aids, not replacements**

The survey's hierarchical memory sub-taxonomy (flat -> planar -> hierarchical) implicitly supports this -- the hierarchical level serves as a navigation layer to the detail level below. Their distinction between working memory (active scratchpad) and factual memory (persistent store) also implies that compressed representations serve immediate reasoning while full details remain retrievable. **No disagreement.**

**3. Temporal awareness requires explicit modeling**

The survey explicitly discusses bi-temporal modeling and temporal trajectory tracking, citing Zep. Their Evolution dynamics section covers how memories should be updated with temporal metadata. **No disagreement.**

**4. ~50% of errors are reading failures**

The survey does **not** address this finding explicitly. Their Retrieval dynamics section covers timing, query construction, selection, and post-processing -- but "post-processing" focuses on re-ranking and compression, not on whether the consuming agent correctly interprets retrieved memories. This is a gap in their framework. **Neither agrees nor disagrees -- they don't engage with the reading failure problem.**

**5. Mixing memory sources produces interference**

The survey doesn't directly address this AWM finding. Their taxonomy separates factual, experiential, and working memory by function, which could be read as implicit agreement (keep types separate). But they don't discuss negative transfer from mixing quality levels. **Neither agrees nor disagrees explicitly.**

**6. Consolidation is essential (CLS theory)**

Strong agreement. The survey explicitly maps to CLS theory and Complementary Learning Systems. Their Evolution dynamics section treats consolidation as a first-class operation. They cite the same McClelland et al. / Kumaran et al. lineage we build on. **Strong agreement, same theoretical foundation.**

### One Potential Disagreement

**Automation vs. human-in-the-loop**: The survey identifies "memory automation" as a key future direction and highlights Mem-alpha and Memory-R1 as evidence that learned policies for memory operations outperform heuristic approaches. Their framing positions human-curated memory as a transitional stage toward fully automated memory management. This is the survey's strongest implicit challenge to our design, where human review is a core feature, not a workaround.

However, this isn't a direct contradiction of our findings -- it's a different design goal. The survey focuses on fully autonomous multi-agent systems where human review is infeasible. For a single-user system with a persistent relationship, human curation is a genuine differentiator that produces higher-quality memories than automated capture (our Phase 8 consensus, validated by Dynamic Cheatsheet's and Evo-Memory's findings on unsupervised accumulation). The RL-based approaches could eventually augment our auto-capture quality (better detection of what's worth capturing) without replacing human review.

### One Notable Omission

The survey does **not** discuss the "reading strategy" problem as a distinct concern. Their dynamics framework has Formation -> Evolution -> Retrieval, but Retrieval ends at "post-retrieval processing" (re-ranking, filtering). What happens *after* retrieved memories are injected into context -- whether the consuming agent correctly interprets them -- is outside their framework. This remains our most distinctive finding from LongMemEval, and it's a genuine blind spot in the survey.

---

## Benchmarks

### Benchmarks Mentioned in the Survey

| Benchmark | What It Tests | In Our Corpus? |
|-----------|--------------|----------------|
| **LoCoMo** (LOCOMO) | Long-context memory: single-hop, temporal, multi-hop, open-domain | Yes (used in Hindsight eval) |
| **LongMemEval** | Five memory abilities, seven question types | Yes (deep-analyzed, Source 11) |
| **StreamBench** | Streaming data memory with external feedback; quality and efficiency over time | **No** -- new to us |
| **SWE-bench Verified** | Software engineering task completion | Known but not memory-specific |
| **GAIA** | General AI assistant capabilities | Known but not memory-specific |
| **BrowseComp** | Web browsing and research tasks | Known but not memory-specific |
| **XBench** | Complex problem-solving | **No** -- new to us |

### New Benchmark: MemoryAgentBench (ICLR 2026)

Not cited in the survey itself (it postdates v2), but discoverable through the companion paper list. Four core competencies:
1. **Accurate retrieval** -- can the agent find the right memory?
2. **Test-time learning** -- can the agent learn from new information within a session?
3. **Long-range understanding** -- can the agent synthesize across distant memories?
4. **Selective forgetting** -- can the agent correctly ignore outdated information?

This is the first benchmark we've seen that explicitly tests **selective forgetting** as a capability, filling one of our identified evaluation gaps. The "test-time learning" competency also maps to our auto-capture pattern detection.

### New Benchmark: MemAgents Workshop (ICLR 2026)

Not a benchmark per se, but an interdisciplinary workshop bridging three perspectives:
- Memory architectures (episodic, semantic, working, parametric)
- Systems & evaluation (data structures, retrieval pipelines, long-horizon benchmarks)
- Neuroscience-inspired memory (CLS theory, hippocampal-cortical consolidation)

Worth monitoring for accepted papers. The workshop's framing closely mirrors our project's arc.

### Assessment vs. Our Existing Benchmark Knowledge

We previously identified WorM, StructMemEval, LongMemEval, and LOCOMO as our benchmark corpus. The survey adds StreamBench and XBench as new names, plus confirms GAIA/SWE-bench/BrowseComp as relevant (though not memory-specific). The most valuable addition is MemoryAgentBench (ICLR 2026) -- its selective forgetting competency fills a gap no existing benchmark addresses.

---

## Relevance to claude-memory

### What This Survey Adds

**1. The Forms x Functions grid as a positioning tool.** We can now precisely state: "claude-memory operates in the token-level form across factual, experiential, and (via startup_load) working functions, with planned expansion to planar topology via memory_edges." This positions us clearly in the landscape.

**2. The distinction between agent memory, LLM memory, RAG, and context engineering.** The survey's explicit boundary-drawing helps articulate what claude-memory is *not*: it's not RAG (stateless retrieval from a fixed knowledge base), not LLM memory (attention/KV optimization), and not context engineering (single-turn prompt optimization). It's a persistent, evolving, agent-controlled memory system. This framing is useful for explaining the project to others.

**3. The RL-based memory automation frontier.** Mem-alpha and Memory-R1 represent a genuinely new direction not covered in our 22 sources. The idea that memory operations (what to store, when to consolidate, what to forget) could be end-to-end optimized via RL is compelling. For our system, this could mean: better auto-capture heuristics trained on which pending memories get confirmed vs. discarded, or learned retrieval strategies that adapt based on recall_feedback signals. This is a Phase 3+ consideration.

**4. The 200+ paper list as a living reference.** The companion GitHub repo is more comprehensive than our literature survey and is actively maintained. It's a discovery tool for future phases.

**5. Generative retrieval as a concept.** Instead of returning raw memories, synthesize a context-specific answer from them. This could be a more sophisticated version of our #12 (reading strategy optimization) -- instead of formatting memories better for the reading agent, have a retrieval agent that *interprets* memories and returns a synthesis.

### What We're Already Doing Better

**1. Our type taxonomy is finer-grained than theirs.** Their three functional categories (factual/experiential/working) compress too much. Our five types (episodic/semantic/procedural/reflection/meta) separate what they lump together, and our reflection/meta categories represent a self-referential capability their framework doesn't capture at all. The reflection type enables memory *about* memory -- a prerequisite for the kind of metacognitive improvement our system is designed to support.

**2. Our reading failure insight has no counterpart in the survey.** The ~50% reading failure finding (LongMemEval) doesn't appear in their Retrieval dynamics. Their framework stops at "post-retrieval processing" and doesn't analyze what happens after retrieved memories enter the agent's context. This is a genuine blind spot.

**3. Our human-in-the-loop design addresses quality problems their automation frontier hasn't solved.** The survey acknowledges that "model hallucinations during formation corrupt long-term knowledge" creating a "circular dependency." Our pending-with-review design directly addresses this by keeping auto-captured memories out of active retrieval until human-confirmed. Their RL-based automation doesn't solve the hallucination propagation problem -- it just makes the hallucination-prone process faster.

**4. Our consensus on memory source interference is absent from their framework.** AWM's negative transfer finding (offline+online combination hurts performance) should inform how mixed-quality memory systems are designed. The survey's clean separation of forms doesn't address the quality-consistency problem that arises when memories of different provenance are mixed.

**5. Our decay model is more sophisticated than most systems they cite.** The survey mentions a decay factor of 0.995 per hour. Our exponential decay with a floor at 35% of base priority, combined with priority-10 exemptions and access-based reheating, is more nuanced than what most surveyed systems implement. The survey confirms our Phase 8 finding that forgetting is the most under-implemented operation.

**6. We have concrete evaluation tiers; they list benchmarks without an evaluation strategy.** Our four-tier evaluation framework (behavioral probes, context completeness, LongMemEval subset, sleep quality metrics) is more actionable than their benchmark compilation.

---

## Worth Stealing (ranked)

### 1. MemoryAgentBench as evaluation target (Value: High, Effort: Low)
The ICLR 2026 benchmark tests selective forgetting as a core competency -- something no benchmark in our existing corpus does. Once available, run claude-memory against it. Particularly valuable for validating our decay model and consolidate() operations.

### 2. Generative retrieval as an enhancement to recall() (Value: High, Effort: Medium)
Instead of returning formatted memory entries, have a lightweight sub-agent synthesize an answer from retrieved memories before returning to the main context. This could address the reading failure problem more directly than format improvements alone. Implementation: add an optional `synthesize=True` parameter to `recall()` that runs a cheap model over the retrieved memories with the original query as prompt. This is an evolution of our #12 (reading strategy optimization).

### 3. The flat/planar/hierarchical framing for tracking our architecture's evolution (Value: Medium, Effort: Zero)
Use their sub-taxonomy as a progress marker: we're currently flat (all memories in one table), moving to planar (memory_edges, priority #2), and targeting hierarchical (sleep skill's layers, priority #5). This is just vocabulary, no implementation change, but useful for communication and self-assessment.

### 4. Monitor RL-based memory operation papers for auto-capture improvement (Value: Medium, Effort: Low)
Read Mem-alpha and Memory-R1 in detail. The specific insight worth extracting: what features of a memory operation (store/update/forget) predict whether it improves or degrades downstream task performance? This could improve our auto-capture heuristics -- not by replacing human review, but by making the auto-capture patterns more precise about what's worth flagging.

### 5. StreamBench for continuous learning evaluation (Value: Low-Medium, Effort: Low)
The streaming data benchmark tests how agents leverage memory of previous interactions to improve over time. Could be adapted for evaluating whether our memory system actually improves Claude's performance across sessions.

---

## Impact on Implementation Priority

### Changes to the 14-Item Priority List

| # | Current Item | Survey Impact | Change? |
|---|-------------|---------------|---------|
| 1 | RRF fusion + enriched keys | Strongly validated. Survey positions hybrid retrieval as mature consensus. | **No change.** |
| 2 | Relationship edges | Validated. Their flat/planar/hierarchical progression confirms edges are the next structural evolution. MAGMA adds a multi-graph variant to watch. | **No change.** |
| 3 | Graded contradiction detection + temporal invalidation | Validated. Survey's Evolution dynamics covers this. | **No change.** |
| 4 | Decay floor + power-law + dormancy | Validated. Survey confirms forgetting is under-implemented across the field. | **No change.** |
| 5 | Sleep skill | Strongly validated by CLS alignment. Survey's Evolution dynamics + emerging RL-based consolidation are worth noting but don't change the design. | **No change. Note: RL-based consolidation (Mem-alpha) is a future enhancement vector.** |
| 6 | Mutation log + lifecycle states | Not addressed directly by survey. Remains an implementation concern below the survey's abstraction level. | **No change.** |
| 7 | Confidence scores | Validated by Memory-R1's approach. Their RL framework uses outcome-driven signals to update memory quality estimates. | **No change.** |
| 8 | Reference index / memory_stats() | Not addressed by survey. | **No change.** |
| 9 | Multi-angle indexing | Partially addressed by their multi-form discussion. | **No change.** |
| 10 | CMA behavioral probes | Not addressed by survey. Continuum's framework remains the source. | **No change.** |
| 11 | Temporal extraction on remember() | Validated by their Formation dynamics discussion. | **No change.** |
| 12 | Reading strategy optimization | **Enhanced.** Generative retrieval concept from the survey provides a more ambitious framing: instead of just formatting memories better, synthesize an answer from them. Consider renaming to "post-retrieval synthesis." | **Enrich, don't reorder.** |
| 13 | Lightweight entity resolution | Validated by multiple graph-based systems in the survey. | **No change.** |
| 14 | Retrieval feedback loop | Validated by RL-based approaches (learned retrieval policies). | **No change.** |

**Net impact: Zero reordering. One enrichment (item #12).** The survey confirms our existing priorities and ordering rather than disrupting them. This is expected -- our 22-source corpus already covered the substantive architectural territory.

### Potential New Items (not yet prioritized)

| Candidate | Source | Assessment |
|-----------|--------|------------|
| **MemoryAgentBench evaluation** | ICLR 2026 | Add to evaluation strategy when available. Not an implementation priority -- it's an evaluation tool. |
| **Generative retrieval mode for recall()** | Survey concept | Bundle with #12 as an evolution. Don't add as separate item. |
| **RL-based auto-capture tuning** | Mem-alpha, Memory-R1 | Phase 3+ research. Too early to add to the priority list. Would require training infrastructure we don't have. |

---

## Connections

### To Prior Analyses

- **[[agent-output-memory-survey]]** (Zhang et al., ACM TOIS 2025): Our existing memory survey. The Hu et al. survey covers much of the same territory but is more recent (December 2025 vs. 2025 publication) and more comprehensive (200+ papers vs. 27 systems). Zhang et al.'s W-P-R decomposition maps cleanly to Hu et al.'s Dynamics axis. Hu et al. adds the Forms dimension that Zhang et al. doesn't have.

- **[[agent-output-continuum]]**: Continuum's CMA behavioral requirements map to the survey's Forms x Functions grid. Continuum requires persistence (token-level factual memory), selective retention (evolution dynamics), retrieval-driven mutation (evolution dynamics), associative routing (planar memory structure), temporal continuity (temporal formation/evolution), and consolidation (evolution dynamics).

- **[[agent-output-hindsight-paper]]**: The survey cites Hindsight's multi-channel retrieval and temporal narratives. Our deep analysis of Hindsight already extracted the architectural patterns the survey surfaces.

- **[[agent-output-zep-paper]]**: Zep's bi-temporal modeling is cited prominently in the survey's discussion of temporal memory formation and evolution.

- **[[agent-output-mem0-paper]]**: Mem0 is used as a reference framework in the survey. Our analysis of its architectural limitations (thin consolidation, no temporal modeling) remains valid.

- **[[broader-landscape]]**: CLS theory, which is foundational to both our project and the survey, was established in our Phase 7 analysis. The survey adds no new theoretical foundation beyond CLS.

### To the Companion Repository

The `Shichun-Liu/Agent-Memory-Paper-List` GitHub repository is the most comprehensive actively-maintained paper list in the field. It should replace our ad-hoc tracking of new papers. Checking it monthly would surface new relevant work more efficiently than manual searching.

---

## Summary Assessment

### Overall Value: Moderate-High for Positioning, Low for Architecture

This survey is most valuable as a **positioning and framing tool** -- it gives us precise vocabulary for describing claude-memory's place in the landscape, and the companion paper list is an excellent discovery resource. The Forms x Functions x Dynamics taxonomy is clean and useful.

For **architectural guidance**, the survey adds little beyond what our 22-source corpus already covers. The substantive insights (RRF fusion, CLS-based consolidation, temporal modeling, relationship edges, decay/forgetting) were all established in our Phases 6-8. This is not a criticism of the survey -- it's a survey, not a system paper. It organizes the field rather than advancing it.

The **genuinely new contributions** relative to our corpus are:
1. The RL-based memory automation frontier (Mem-alpha, Memory-R1) -- a new direction worth watching
2. The generative retrieval concept -- a reframing of our reading strategy problem
3. MemoryAgentBench -- a new evaluation tool with selective forgetting testing
4. The 200+ paper list as a living discovery resource

The survey **does not contradict** any of our consensus findings. Our architecture, priorities, and theoretical foundation are consistent with the state of the field as this survey describes it. Our human-in-the-loop design, reading failure insight, and five-type taxonomy represent points where we're *ahead* of the survey's coverage.

**Recommended action**: Read Mem-alpha (arXiv:2509.25911) and Memory-R1 (arXiv:2508.19828) as the two highest-value follow-ups. Add MemoryAgentBench to the evaluation strategy when available. Monitor the ICLR 2026 MemAgents workshop for accepted papers. Use the companion paper list for ongoing discovery.
