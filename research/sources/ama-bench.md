# AMA-Bench: Evaluating Long-Horizon Memory for Agentic Applications -- Analysis

*Phase 15, 2026-03-06. Analysis of arXiv:2602.22769 (Feb 2026).*

---

## 1. Paper Overview

**Paper:** "AMA-Bench: Evaluating Long-Horizon Memory for Agentic Applications"
**Authors:** Yujie Zhao, Boqin Yuan, Junbo Huang, Haocheng Yuan, Zhongming Yu, Haozhou Xu, Lanxiang Hu, Abhilash Shankarampeta, Zimeng Huang, Wentao Ni, Yuandong Tian, Jishen Zhao
**Submitted:** February 26, 2026 (v1), March 4, 2026 (v2). arXiv:2602.22769.
**Venue:** Preprint (cs.AI, cs.LG). No confirmed venue yet.
**Code/Data:** GitHub (AMA-Bench/AMA-Hub), HuggingFace (AMA-bench/AMA-bench), Leaderboard at HuggingFace Spaces.
**License:** CC BY 4.0.

**Problem:** Existing memory benchmarks (LoCoMo, LongMemEval, MemoryAgentBench) are fundamentally dialogue-centric: they test memory over multi-turn human-agent conversations with natural language content. Real agentic applications involve machine-generated data streams -- code diffs, JSON state dumps, ASCII tables, environment observations -- that are causally grounded and information-dense. Memory systems optimized for phatic conversation fail catastrophically on these dense, structured trajectories. No benchmark previously tested this.

**Core contribution:** Two things. First, **AMA-Bench**, a benchmark of 2,496 expert-annotated QA pairs across six agentic domains (real-world subset) plus 1,200 programmatically-generated QA pairs with ground-truth verification (synthetic subset). Second, **AMA-Agent**, a memory framework using a Causality Graph and Tool-Augmented Retrieval that achieves 57.22% average accuracy, surpassing the strongest baseline (MemoRAG at 46.06%) by 11.16%.

**Scale:** Real-world trajectories average 57,506 tokens (max 996,826 in Open World Tool QA). Synthetic subset spans five horizon lengths: 8K, 16K, 32K, 64K, 128K tokens (240 samples per interval). Six domains: web navigation, software engineering, text-to-SQL, embodied AI, gaming, open-world tool use. Fifteen baseline systems evaluated.

---

## 2. Benchmark Design

### The Four Memory Capabilities

| # | Capability | Mechanism | Definition |
|---|-----------|-----------|------------|
| A | Recall | Memory Retrieval | Identification of temporal and sequential information |
| B | Causal Inference | Memory Retrieval | Verification of action preconditions and dependency relations between states |
| C | State Updating | Memory Evolution | Tracking updates to states, including explicit observations and hidden states |
| D | State Abstraction | Memory Condensation | Filtering redundant content while extracting precise and condensed key information |

This is a narrow taxonomy compared to BEAM's 10 abilities or LongMemEval's 5 types, but the categories are designed specifically for agentic contexts where state transitions, not conversational facts, are the primary content. The mapping to memory *mechanisms* (retrieval, evolution, condensation) is a useful framing.

### Dataset Construction

**Real-World Subset (2,496 QA pairs):**
- Trajectories sourced from established agent benchmarks: ALFWorld, WebArena, SWE-Bench, Spider 2.0, and others
- Graduate-level annotators with LLM agent research experience
- 12 memory-intensive QA pairs per trajectory, following standardized guidelines
- Cross-reviewer validation with sanity checks
- Each question answered by "explicit and unambiguous evidence within the trajectory"

**Synthetic Subset (1,200 QA pairs):**
- Two programmatic environments: BabyAI (grid-world, avg 563 turns, 30K tokens) and TextWorld (text adventure, avg 57 turns, 32K tokens)
- Ground-truth MDP available: latent state s_t and transition kernel P_phi(s_t, a_{t+1}) are programmatically defined and machine-verifiable
- Two perturbations for realism:
  - **Action Stochasticity (epsilon):** Random noise injected into optimal policy to simulate sub-optimal agents
  - **Observation Verbosity (gamma):** Controllable descriptive granularity in symbolic representations
- QA pairs anchored to backend state variables, enabling automatic verification
- Five horizon lengths (8K-128K tokens, 240 samples each)

**Key design innovation: the Needle Protocol.** A "needle" is the minimal set of trajectory turn IDs containing all evidence necessary to answer a query. Because the synthetic subset is backed by a programmatic environment, needles can be automatically synthesized and verified. This enables precise ablation: you can test whether a memory system succeeds because it retrieved the right turns or despite not retrieving them.

### Evaluation Metrics

- **Primary:** LLM-as-judge accuracy using Qwen3-32B, with human-judge agreement validation reported in appendix
- **Secondary:** F1 score
- Rule-based QA grounding for synthetic subset (leveraging programmatic environment semantics)

### Domain Coverage and Token Ranges

| Domain | Avg Tokens | Max Tokens |
|--------|-----------|-----------|
| Open World Tool QA | 288,651 | 996,826 |
| Web Task Execution | 34,265 | 166,260 |
| Embodied AI | 26,306 | 60,717 |
| Software Engineering | 19,296 | 28,615 |
| Gaming | 14,909 | 33,360 |
| Text-to-SQL | 6,049 | 10,718 |
| **Overall** | **57,506** | — |

The variance is enormous -- Open World Tool QA alone accounts for a 5x increase in average trajectory length. This is realistic (real agent tasks have wildly varying context needs) but means aggregate metrics may be dominated by the longest-trajectory domains.

---

## 3. Key Claims and Evidence

### Claim 1: Memory systems fall short of the long-context baseline.

**Evidence:** GPT 5.2 (400K context window) achieves 72.26% accuracy by simply stuffing the full trajectory into context. The best memory system (AMA-Agent at 57.22%) is 15 points lower. Even on the Qwen3-32B backbone where memory systems are tested, the base model with full context (51.81%) is competitive with or outperforms most memory baselines. Only AMA-Agent and MemoRAG consistently beat the long-context approach on Qwen3-32B.

**Assessment:** Strong evidence. This mirrors the LoCoMo finding (full-context 72.9% vs filesystem agent 74%) but is more damning because the token counts are much larger. The gap between GPT 5.2 (72.26%) and AMA-Agent (57.22%) is substantial -- 15 percentage points. Memory systems are supposed to be the *solution* when context windows are insufficient; here they are the bottleneck even when context windows *are* sufficient.

**Caveat:** The comparison is somewhat unfair -- long-context models are frontier commercial systems (GPT 5.2 at 400K), while memory systems run on Qwen3-32B (32K native context). A fairer comparison is AMA-Agent (57.22%) vs Qwen3-32B long-context (51.81%), where AMA-Agent wins by 5.4 points.

### Claim 2: Memory architecture matters more than model scale.

**Evidence:** Scaling from Qwen3-8B to Qwen3-32B (4x parameters) yields only 0.038 average improvement in the long-context setting. Memory architecture variance across systems reaches 0.45 (from Mem1 at 12.29% to AMA-Agent at 57.22%).

**Assessment:** Convincing. The 12x ratio (0.45 / 0.038) between architecture variance and scale improvement is striking. This aligns with our design philosophy: investing in retrieval quality (RRF, edges, decay) rather than waiting for bigger models.

### Claim 3: Similarity-based retrieval fails on agent trajectories.

**Evidence:** The needle protocol ablation reveals a devastating pattern:

| System | Full Obs + Needle | Constructed Memory + Needle | End-to-End | Drop |
|--------|------------------|---------------------------|-----------|------|
| HippoRAG2 | 0.46 | 0.37 | 0.21 | -43.2% |
| MemoryBank | 0.46 | 0.27 | 0.26 | -41.3% (at construction) |
| Mem1 | 0.46 | 0.29 | 0.20 | -37.0% (at construction) |
| A-Mem | 0.46 | 0.29 | 0.24 | -37.0% (at construction) |

The pattern: systems lose performance at two stages -- (1) during memory construction (compression discards dense state information) and (2) during retrieval (similarity search fails to find relevant machine-generated content). HippoRAG2 survives construction relatively well (0.37) but collapses during retrieval (0.21), while MemoryBank collapses at construction (0.27) but barely degrades further during retrieval (0.26).

**Assessment:** This is the paper's most valuable contribution. The needle protocol cleanly separates *write-path failures* (compression loss) from *read-path failures* (retrieval miss), something no prior benchmark does with this precision. The finding that HippoRAG2 -- the most sophisticated graph-based retrieval system -- loses 43.2% end-to-end despite strong constructed-memory performance is a direct indictment of similarity-based retrieval on machine-generated text.

### Claim 4: AMA-Agent's Causality Graph and Tool-Augmented Retrieval are both essential.

**Evidence (ablation):**

| Configuration | Recall | Causal | State Upd | State Abs | Average |
|--------------|--------|--------|-----------|-----------|---------|
| Full AMA-Agent | 0.62 | 0.61 | 0.53 | 0.47 | 0.57 |
| Without Causality Graph | 0.48 | 0.48 | 0.36 | 0.35 | 0.43 (-24.6%) |
| Without Tool-Aug Retrieval | 0.47 | 0.51 | 0.42 | 0.31 | 0.44 (-22.8%) |

Removing either component produces similar overall degradation (~23-25%), but they hit different capabilities hardest: the Causality Graph is most important for State Updating (-32.1%), while Tool-Augmented Retrieval is most important for State Abstraction (-34.0%).

**Assessment:** Clean ablation with interpretable capability-specific impacts. Both components contribute roughly equally but address different failure modes. The Causality Graph preserves causal structure that compression destroys; the tools provide alternative retrieval paths when similarity search fails.

### Claim 5: Scalability beyond 32K favors AMA-Agent.

**Evidence:** On the synthetic subset, long-context baselines "maintain competitive accuracy at shorter scales" but "degrade significantly beyond 32K." AMA-Agent "exhibits superior scalability, maintaining robust and consistently high accuracy even at 128K." The real-to-synthetic correlation (Figure 9A) shows methods clustering near the diagonal, validating the synthetic subset as a reliable proxy.

**Assessment:** The scalability claim is plausible but harder to verify without the full figure data. The real-to-synthetic correlation is a methodological strength -- it means the synthetic subset can be used for cheap, rapid iteration without real-world trajectory collection.

---

## 4. Standout Feature

**The Needle Protocol** is what distinguishes AMA-Bench from every prior memory benchmark. By leveraging programmatic environments with ground-truth MDPs, the benchmark can automatically identify the minimal evidence set (needle) for each question and test at three granularities:

1. **Full Observation + Needle:** Can the model answer given the original trajectory with the needle turns present? (Tests comprehension)
2. **Constructed Memory + Needle:** Can the model answer given the memory system's representation with the needle turns injected? (Tests write-path / compression quality)
3. **End-to-End:** Can the model answer using only what the memory system retrieves? (Tests full pipeline including retrieval)

This decomposition is strictly more informative than aggregate accuracy. It tells you *where* a system fails: is the problem that your compression discards critical state information (MemoryBank), or that your retrieval can't find it even when it's preserved (HippoRAG2)? No prior benchmark offers this diagnostic precision.

The closest prior work is LongMemEval's error taxonomy (50% reading failures), which was manually constructed. AMA-Bench automates this decomposition at scale.

---

## 5. Competency Coverage Ratings

| Dimension | Coverage | Justification |
|-----------|----------|---------------|
| **Information Retrieval** | 75% | Recall capability directly tests retrieval of temporal and sequential information from agent trajectories. The needle protocol provides precise retrieval diagnostics. Does not test multi-channel fusion strategies or retrieval parameter sensitivity. |
| **Multi-Session Reasoning** | 15% | Focuses on within-episode memory only. Authors explicitly acknowledge cross-task and lifelong scenarios as future work. |
| **Knowledge Update/Contradiction** | 45% | State Updating tests tracking changes to explicit and hidden states, which partially covers knowledge updates. However, no explicit contradiction detection or resolution tasks. "State Updating" is closer to "state tracking" than "belief revision." |
| **Temporal Reasoning** | 40% | Recall capability includes "temporal and sequential information." Causal Inference tests preconditions and dependency chains, which are temporally ordered. But no explicit temporal arithmetic (when did X happen? how long between X and Y?) or temporal comparison tasks. |
| **Abstention/Confidence** | 10% | No explicit abstention tasks. The benchmark tests whether systems produce correct answers, not whether they appropriately refuse when evidence is absent. |
| **Write-Path Behavior** | 70% | The needle protocol's "Constructed Memory + Needle" stage directly tests write-path quality -- whether memory construction preserves critical information. State Abstraction tests compression quality. This is the benchmark's strongest competency. |
| **Consolidation Quality** | 35% | State Abstraction ("filtering redundant content while extracting condensed key information") partially tests consolidation. But there is no explicit test of offline consolidation, summary quality, or evolution tracking. |
| **Proactive/Contextual Recall** | 5% | All evaluation is reactive (answer a question given a trajectory). No tests for unsolicited, proactive memory surfacing. |
| **Relationship/Graph Reasoning** | 55% | Causal Inference directly tests graph-like reasoning (preconditions, dependency chains). The Causality Graph in AMA-Agent uses directed and undirected edges with neighborhood traversal. But the benchmark tests this through QA accuracy, not graph structure quality. |
| **Agentic Task Performance** | 85% | This is the benchmark's raison d'etre. Six real agentic domains, machine-generated trajectories, state transitions, tool use. The highest agentic task coverage of any memory benchmark reviewed. |

**Total weighted average:** ~43.5%. AMA-Bench is deep on agentic retrieval/write-path and shallow on conversational memory competencies (multi-session, abstention, proactive recall). This is by design -- it fills a gap that prior benchmarks don't cover.

---

## 6. Relevance to claude-memory

### Could we run against this?

**Partially, with significant adaptation.** AMA-Bench is designed for agentic memory systems that process interaction trajectories -- sequences of (observation, action, observation) tuples from environments like ALFWorld or WebArena. Our system processes conversational memories from multi-session human-Claude interactions.

Running directly: No. Our `remember()`/`recall()` API expects natural language content with themes, categories, and summaries. Agent trajectories are machine-generated state dumps with causal structure. We would need to:

1. Build an ingestion layer that converts trajectory turns into memories (each state transition becomes a memory with appropriate metadata)
2. Map our retrieval pipeline (vector + FTS5 + planned PPR) against their QA evaluation
3. Handle the scale difference -- 57K avg tokens per trajectory vs our ~386 discrete memories

### What would it reveal?

The **needle protocol** would be the most valuable diagnostic. It would cleanly separate:
- How much information our `remember()` + sleep consolidation pipeline discards (write-path quality)
- How well our RRF retrieval finds the right memories when they exist (read-path quality)

This is exactly the decomposition we need. LongMemEval told us "50% of errors are reading failures" but didn't let us test this mechanistically.

The **Causal Inference** capability would directly test our planned PPR-over-edges (#2). Agent trajectories with causal dependency chains are precisely where graph traversal should outperform flat similarity search. If PPR can follow causal edges to find preconditions several steps back, we should see measurable gains on this dimension.

### Adaptation effort

**Medium-high.** The synthetic subset (BabyAI + TextWorld) is more tractable -- programmatic environments with structured state, 30K tokens average. We could:

1. Treat each trajectory turn as a memory with structured metadata (action, observation, state changes)
2. Map the four capabilities to our eval framework
3. Use the needle protocol to benchmark write-path vs read-path quality

The real-world subset would require processing raw agent logs from six different domain benchmarks, which is substantially more work.

**Verdict:** Not a direct evaluation target, but the needle protocol methodology should be adapted into our CMA behavioral probes (#10). The Causality Graph design is relevant to our PPR implementation (#2).

---

## 7. Insights Worth Stealing

### Tier 1: High impact, low-to-medium effort

1. **Needle protocol methodology for our eval probes (#10).** The idea of identifying minimal evidence sets and testing at three granularities (full context, constructed memory + evidence, end-to-end) is transferable to any memory system. For our CMA probes, we could: store a known set of memories, identify which subset is needed for a query, and measure whether recall() retrieves that subset. This cleanly separates retrieval quality from reading quality. *Effort: low. Impact: high diagnostic value.*

2. **Write-path vs read-path decomposition as standard eval practice.** Every benchmark we've reviewed reports end-to-end accuracy. AMA-Bench shows this conflates two independent failure modes. Our eval framework should always report both: (a) what fraction of relevant memories survive `remember()` + sleep, and (b) what fraction of surviving relevant memories are retrieved by `recall()`. *Effort: low (instrumentation). Impact: high for debugging.*

3. **Tool-augmented retrieval as fallback.** AMA-Agent's retrieval uses similarity search first, then self-evaluates sufficiency, then falls back to graph traversal and keyword search. The self-evaluation step is the key insight: let the model decide whether retrieved context is sufficient before synthesizing a response. Our staged curated recall already has a version of this (Sonnet subagent decides whether to multi-hop), but making sufficiency assessment explicit at retrieval time is worth adopting. *Effort: medium. Impact: medium.*

### Tier 2: Medium impact, medium effort

4. **Causality Graph as edge type.** AMA-Agent's directed causality edges (state A caused state B) and undirected association edges (state A co-occurs with state B) map naturally to our `memory_edges` schema. When we implement PPR (#2), we should support directed causal edges alongside the undirected association edges we currently plan. This is especially relevant for procedural memories where "I tried X, it failed, then Y worked" has causal structure. *Effort: medium (schema change + edge creation logic). Impact: medium for procedural recall.*

5. **Synthetic environment for controlled evaluation.** BabyAI and TextWorld provide ground-truth MDPs for automatic QA generation and verification. For our system, we could build a synthetic conversation generator with known ground-truth (plant specific facts, relationships, contradictions, temporal sequences) to generate machine-verifiable evaluation questions. This would give us a scalable alternative to hand-crafted CMA probes. *Effort: medium-high. Impact: high for iteration speed.*

### Tier 3: Lower priority

6. **Compression diagnostics for sleep pipeline.** The finding that "compression tuned for redundant natural language fails to preserve dense state information" is a warning for our NREM compression. Our sleep pipeline compresses memories during consolidation -- we should add a post-sleep diagnostic that measures information preservation (can the system still answer the same questions after consolidation as before?). *Effort: medium. Impact: depends on consolidation maturity.*

---

## 8. What's Not Worth It

- **Reproducing the full AMA-Bench evaluation.** The benchmark requires running agent trajectories from six domain-specific benchmarks (ALFWorld, WebArena, SWE-Bench, Spider 2.0, etc.), each with its own setup and environment. Our system is conversational memory, not agentic memory. Full reproduction would require building an entirely different ingestion pipeline for marginal benefit.

- **Adopting the Causality Graph architecture wholesale.** AMA-Agent's three-stage graph construction (signal extraction from adjacent turn pairs, edge creation, global integration with embedding) is designed for sequential state-transition trajectories. Our memories are discrete, user-authored records with natural language content, themes, and metadata. The graph structure is different (our edges connect related memories, not sequential states). The *principles* (causal edges, graph traversal, tool-augmented retrieval) are worth stealing; the specific architecture is not.

- **The four-capability taxonomy.** Recall / Causal Inference / State Updating / State Abstraction are useful categories for agentic memory but don't map well to conversational memory evaluation. BEAM's 10-ability taxonomy or LongMemEval's 5-type framework are more relevant to our use case.

- **Per-domain analysis across six agentic domains.** The variance across web navigation, software engineering, text-to-SQL, etc. reflects domain-specific challenges (code vs JSON vs natural language observations) that are irrelevant to our conversational memory system.

---

## 9. Key Takeaway

AMA-Bench is the first benchmark to rigorously test memory systems on machine-generated agentic trajectories rather than natural language conversations, and its central finding is sobering: even the best memory system (AMA-Agent at 57.22%) trails a frontier long-context model (GPT 5.2 at 72.26%) by 15 points, and similarity-based retrieval collapses on dense, structured agent data (HippoRAG2 drops 43.2% end-to-end). The paper's most transferable contribution is the **needle protocol** -- a methodology for cleanly decomposing memory system failures into write-path (compression loss) and read-path (retrieval miss) components. For claude-memory, the direct benchmark applicability is limited (we're conversational, not agentic), but the needle protocol methodology should be adopted into our evaluation probes, and the paper's validation that graph-based causal reasoning and tool-augmented retrieval both independently contribute ~23-25% of system performance reinforces our PPR implementation priority.

---

## 10. Impact on Implementation Priority

**No reordering of the #1-#14 priority list.** Specific impacts:

- **#2 (Relationship edges / PPR traversal):** Reinforced. AMA-Agent's Causality Graph ablation shows -24.6% without causal edges, with the steepest degradation on State Updating (-32.1%). This is the third independent data point (after HippoRAG and LoCoMo-Plus) confirming that graph structure provides information that flat retrieval cannot recover. The causality graph's directed edges are worth considering as an edge type in our schema -- not just undirected "related" associations but directed "caused" / "preceded" / "superseded" edges.

- **#1 (RRF fusion):** Reinforced indirectly. AMA-Agent's Tool-Augmented Retrieval combines embedding similarity, keyword search, and graph neighborhood traversal -- three channels fused by a sufficiency-checking agent. This is architecturally close to our planned vector + FTS5 + PPR three-channel RRF, with the addition of an explicit sufficiency check. The sufficiency self-evaluation step is a design idea worth noting: after retrieval, the agent assesses whether evidence is adequate before answering, and can invoke additional tools if not. Our staged curated recall already embodies a version of this.

- **#10 (CMA behavioral probes):** Enriched. The needle protocol is the most precise evaluation methodology we've encountered for separating write-path from read-path failures. Our CMA probes should adopt this three-level structure: (1) can the system answer when all memories are in context? (2) can it answer when the right memories survive write-path processing? (3) can it answer end-to-end? This would give us the diagnostic precision that LongMemEval's "50% reading failures" finding only approximated.

- **#3 (Contradiction detection):** Tangentially relevant. State Updating tests tracking changes, but AMA-Bench does not test contradiction detection or resolution directly. The finding that compression methods "fail to preserve dense state and causal information" is a warning for our consolidation pipeline -- sleep compression should be validated against a needle-style information preservation test. But this is about compression quality, not contradiction handling per se.

- **#5 (Sleep/consolidation):** Cautionary note added. MemoryBank's -41.3% degradation at the construction stage (before retrieval even happens) shows that aggressive compression can be worse than no compression. Our NREM pipeline should include a post-consolidation diagnostic: pick a random subset of pre-consolidation memories, generate questions they should answer, and verify the post-consolidation representation still answers them.

---

## 11. See Also

- [[agent-output-beam]] -- BEAM benchmark analysis (ICLR 2026). 10 abilities, 10M tokens, 100 conversations. Contradiction Resolution universally catastrophic. Noise filtering critical at scale.
- [[agent-output-longmemeval]] -- LongMemEval benchmark analysis (ICLR 2025). 5 abilities, 500 questions, ~1.5M tokens max. "50% of errors are reading failures."
- [[agent-output-memoryagentbench]] -- MemoryAgentBench analysis (ICLR 2026). 4 competencies, 146 instances, 1.44M tokens max. Multi-hop selective forgetting catastrophic (max 7%).
- [[agent-output-locomo-plus]] -- LoCoMo-Plus analysis (Feb 2026). Cognitive memory, implicit constraints, cue-trigger semantic disconnect.
- [[agent-output-tremu]] -- TReMu analysis. Temporal reasoning. Event-time vs storage-time critical.
- [[agent-output-hipporag]] -- HippoRAG analysis. PPR over entity edges, single-step multi-hop. BFS hurts retrieval.

### Cross-benchmark synthesis (updated)

| Dimension | LoCoMo (2024) | LongMemEval (2025) | MAB (2026) | BEAM (2026) | AMA-Bench (2026) |
|-----------|---------------|-------------------|------------|-------------|------------------|
| Max scale | 26K tokens | 1.5M tokens | 1.44M tokens | 10M tokens | ~1M tokens |
| Domain | Conversations | Conversations | Conversations | Conversations | **Agent trajectories** |
| Capabilities | 5 types | 5 abilities | 4 competencies | 10 abilities | 4 capabilities |
| Key innovation | Full-context baseline | Error taxonomy | Multi-hop forgetting | Scale + noise filtering | **Needle protocol** |
| Content type | Natural language | Natural language | Natural language | Natural language | **Machine-generated** |
| Write-path eval | No | No | Partial | No | **Yes (explicit)** |
| Graph reasoning | No | No | No | No | **Yes (Causal Inference)** |

AMA-Bench fills a genuine gap: it is the only benchmark testing memory on machine-generated agentic content, and the only one with explicit write-path diagnostics via the needle protocol. Its weakness is the narrow focus on in-episode agentic memory -- no multi-session, no contradiction resolution, no abstention, no proactive recall. For claude-memory, the methodology (needle protocol, write-path/read-path decomposition) is more valuable than the benchmark itself.
