# MemoryAgentBench: Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions -- Analysis

*Generated 2026-02-20 by Opus 4.6 agent reading arXiv:2507.05257v2, GitHub repo, HuggingFace dataset, and Emergent Mind summary*

---

## Paper Overview

**Paper**: Yuanzhe Hu\*, Yu Wang\* (equal contribution), Julian McAuley. UC San Diego. ICLR 2026 (Poster). arXiv:2507.05257. v1 submitted July 7 2025, v2 September 26 2025. Accepted January 26 2026.

**Problem addressed**: Existing LLM agent benchmarks focus on reasoning, planning, and execution. Memory -- how agents memorize, update, and retrieve long-term information -- is under-evaluated. Existing memory benchmarks either rely on limited context lengths or test static long-context settings (book-based QA), which do not reflect the interactive, multi-turn nature of memory agents that incrementally accumulate information. Moreover, no existing benchmark covers all four essential memory competencies.

**Core contribution**: A unified benchmark that (1) identifies four core memory competencies, (2) restructures existing long-context datasets into incremental multi-turn interactions that simulate realistic agent processing, (3) introduces two novel datasets (EventQA and FactConsolidation) targeting gaps in existing evaluation, and (4) provides comprehensive evaluation of 15+ memory systems spanning long-context, RAG, structure-augmented RAG, and agentic memory paradigms.

**Scale**: 146 evaluation instances across 4 competency splits (22 AR, 6 TTL, 110 LRU, 8 CR). Context lengths range from 124K to 1.44M tokens. 15+ systems evaluated including GPT-4o, Claude-3.7-Sonnet, Gemini-2.0-Flash, BM25, Contriever, text-embed-3 (small/large), Qwen3-Embedding-4B, RAPTOR, GraphRAG, MemoRAG, HippoRAG-v2, Mem0, Cognee, Zep, Self-RAG, MemGPT, and MIRIX.

---

## Benchmark Design

### Competencies Tested

The paper identifies four core competencies for memory agents:

| # | Competency | Abbreviation | Definition | Key Challenge |
|---|-----------|-------------|-----------|--------------|
| 1 | **Accurate Retrieval** | AR | Extract correct information snippets in response to queries, supporting single- and multi-hop reasoning | Finding needles in growing haystacks of incrementally accumulated information |
| 2 | **Test-Time Learning** | TTL | Acquire new behaviors/skills during deployment without additional parameter training | Generalizing from few-shot demonstrations embedded in conversation history |
| 3 | **Long-Range Understanding** | LRU | Integrate information across extended contexts (100K+ tokens) requiring global comprehension | Holistic understanding vs. snippet extraction -- fundamentally different from retrieval |
| 4 | **Selective Forgetting** | SF / CR | Revise, overwrite, or remove previously stored information when encountering contradictory evidence | Detecting and resolving conflicts between stored facts and newer updates |

The paper explicitly notes that selective forgetting "aligns with goals in model editing and knowledge unlearning tasks" -- framing it as conflict resolution rather than natural decay or time-based forgetting.

### Task Construction

**Methodology**: "Inject once, query multiple times." Existing long-context datasets are segmented into chunks (512 tokens for AR/SF synthesis tasks, 4096 for others) and fed incrementally to the agent. This simulates multi-turn interactions where information accumulates over time.

**Datasets by competency**:

| Competency | Dataset | Source | Avg Context | Metric | Novel? |
|-----------|---------|--------|-------------|--------|--------|
| AR | SH-Doc QA | Existing (RULER/HELMET) | 197K | SubEM | No |
| AR | MH-Doc QA | Existing | 421K | SubEM | No |
| AR | LongMemEval (S*) | Existing (Wu et al. ICLR 2025) | 355K | LLM-judge | No |
| AR | EventQA | New | 534K | Accuracy (MC) | **Yes** |
| TTL | BANKING77, CLINC150, NLU, TREC | Existing (intent classification) | Various | Accuracy | No |
| TTL | Movie Recommendation | Existing | 1.44M | Recall@5 | No |
| LRU | Infinity-Bench Summarization | Existing | 172K | F1-Score | No |
| LRU | Detective QA | Existing | 124K | Accuracy | No |
| SF | FactConsolidation-SH | New (from MQUAKE) | 262K | SubEM | **Yes** |
| SF | FactConsolidation-MH | New (from MQUAKE) | 262K | SubEM | **Yes** |

**EventQA** (novel): Tests narrative temporal reasoning in novel-length text. Agents must predict "what happens next" based on accumulated understanding of event chains. Multiple-choice format.

**FactConsolidation** (novel): Constructs counterfactual edit pairs from the MQUAKE dataset. Original facts and their contradictory rewrites are presented sequentially through the conversation, simulating realistic information updates. Agents are explicitly instructed to prioritize later information when conflicts arise. Available in single-hop (direct factual recall) and multi-hop (inference across multiple updated facts) variants. Context lengths tested: 6K, 32K, 64K, 262K tokens.

### Evaluation Metrics

| Metric | Used For | Definition |
|--------|----------|-----------|
| SubEM (Substring Exact Match) | AR (Doc QA), SF | Ground truth appears as substring of model response |
| Accuracy | AR (EventQA), TTL (classification) | Standard multi-choice accuracy |
| Recall@5 | TTL (Movie Rec) | Target item in top-5 recommendations |
| F1-Score | LRU (Summarization) | Token-level F1 against reference |
| LLM-Judge | AR (LongMemEval) | GPT-4o judge with per-type instructions (borrowed from LongMemEval) |

---

## Key Results

### Overall Performance

| Agent | Overall | AR | TTL | LRU | SF |
|-------|---------|------|------|------|------|
| **Long-Context** | | | | | |
| GPT-4o | 48.8 | ~49 | 87.6 (class) / 15.1 (rec) | 32.2 (sum) / 63.4 (det) | 45 (SH) / 5 (MH) |
| Claude-3.7-Sonnet | 49.6 | — | — | — | — |
| GPT-4.1-mini | 46.9 | — | — | — | — |
| Gemini-2.0-Flash | 42.4 | — | — | — | — |
| GPT-4o-mini (baseline) | 42.3 | 49.2 avg | 48.6 avg | 46.2 avg | 25.0 avg |
| **RAG** | | | | | |
| HippoRAG-v2 | — | 65.1 (best AR) | — | — | — |
| BM25 | — | 60.5 | — | — | — |
| text-embed-3-large | — | 54.6 | — | — | — |
| **Agentic** | | | | | |
| MIRIX | — | — | — | — | 8 (MH, best) |

### Key Findings by Competency

**Accurate Retrieval**: RAG systems outperform long-context models. HippoRAG-v2 achieves 65.1% average, beating GPT-4o-mini's 49.2%. BM25 at 60.5% shows that keyword matching remains highly competitive. Structure-augmented RAG (RAPTOR, GraphRAG) generally *underperforms* simpler RAG methods -- architectural overhead does not translate to retrieval gains.

**Test-Time Learning**: Long-context models dominate. GPT-4o achieves 87.6% on classification tasks by seeing all demonstrations in-context. RAG methods struggle because they retrieve only fragments, losing the holistic pattern needed for in-context learning. Movie recommendation is universally hard (best ~15%).

**Long-Range Understanding**: Long-context models again dominate. RAG is fundamentally unsuited for tasks requiring global comprehension (summarization, detective reasoning). Smaller chunk sizes help AR but hurt LRU -- there is a direct tension between these competencies.

**Selective Forgetting**: **Universal catastrophic failure.** All methods achieve at most 7% accuracy on multi-hop forgetting. Single-hop is better but still weak (45% for long-context). This is the single most important finding of the paper.

### Ablation Studies

**Chunk size** (Figure 2): Smaller chunks improve AR (more focused retrieval) but hurt LRU (lose global coherence). This creates an inherent design tension for systems that need both competencies.

**Top-K retrieval** (Figure 3): Performance improves with more retrieved chunks, but 10 chunks already means ~40K tokens of input -- substantial compute cost.

**Backbone model** (Table 3): RAG methods plateau with stronger backbone LLMs. Architecture limits performance, not model capability. Agentic methods (MIRIX) show dramatic gains with better backbones (+23.2% on EventQA with GPT-4.1-mini), suggesting agentic approaches have more headroom.

**FactConsolidation validation** (Table 4): At 6K tokens, o4-mini achieves 100% SH and 80% MH, confirming the tasks are solvable. At 32K: 61% SH, 14% MH. Performance collapses with scale, proving this is a memory management problem, not a reasoning capability problem.

---

## Selective Forgetting (Deep Dive)

This is the most novel and practically important aspect of the paper. The findings here are stark.

### What "Selective Forgetting" Means in This Benchmark

The paper tests a specific form of forgetting: **conflict resolution** -- replacing outdated information with newer contradictory information. This is NOT tested as:
- Natural decay (information fading over time due to disuse)
- Explicit deletion (user requests removal of specific memories)
- Relevance filtering (deprioritizing information that's no longer contextually useful)

Instead, the agent receives factual statements F1 at time T1, then receives contradictory statement F2 at time T2 > T1, and must answer questions using F2 (the later fact). The agent is explicitly instructed to prioritize later information.

### FactConsolidation Construction

1. Start with counterfactual edit pairs from MQUAKE (a model editing dataset)
2. Each pair contains an original fact and a rewritten contradictory fact
3. Rewritten facts are placed *after* original facts in the conversation sequence
4. **Single-hop (SH)**: "What country created tool AA?" -- requires recalling the most recent answer
5. **Multi-hop (MH)**: "What is the death location of the spouse of X?" -- requires chaining through *multiple* updated facts

### Why Everything Fails

The results are damning:

| Method Type | SF-SH | SF-MH |
|------------|-------|-------|
| Long-context (GPT-4o) | ~45% | ~5% |
| RAG (best) | <40% | <7% |
| Agentic (MIRIX, best) | — | 8% |

**Why multi-hop is catastrophically hard**: Multi-hop forgetting requires the agent to:
1. Detect that Fact A has been superseded by Fact A'
2. Detect that Fact B has been superseded by Fact B'
3. Chain A' and B' together correctly
4. Not chain A with B', or A' with B

Each update is independently hard to track; chaining them compounds errors multiplicatively.

**Why RAG fails at forgetting**: Standard embedding-based retrieval has no concept of information supersession. When a query matches both the old fact and the new fact, both are retrieved with similar scores. The agent has no signal about which is more recent unless the retrieval system explicitly models temporal ordering. BM25 is even worse -- keyword matching treats old and new facts identically.

**Why long-context fails at scale**: At 6K tokens, o4-mini achieves 100% SH. At 32K, 61%. At 262K, much lower. The model can reason about contradictions in short contexts but loses track when the contradictory facts are separated by hundreds of thousands of tokens.

### What "Correct Forgetting" Looks Like

In MemoryAgentBench's framework, correct forgetting is:
- **Detecting** that a newer statement contradicts an older one
- **Resolving** the conflict in favor of the newer statement
- **Answering** questions using only the updated information
- **Chaining** through multiple updated facts without mixing old and new

This is evaluated by exact match against the updated answer. There is no partial credit for acknowledging the conflict, no evaluation of the process by which forgetting occurs.

### Comparison to Our Planned Decay (#4) + Dormancy

Our system's approach to forgetting is fundamentally different from what MemoryAgentBench tests:

| Dimension | MemoryAgentBench | Our Planned System |
|-----------|-----------------|-------------------|
| **Trigger** | Contradictory new information | Time passage + disuse |
| **Mechanism** | Conflict detection + resolution | Power-law decay + dormancy floor |
| **Evaluation** | "Does the agent give the updated answer?" | Implicit: does stale info stop surfacing? |
| **Scope** | Fact-level replacement | Memory-level scoring and lifecycle |
| **Multi-hop** | Chaining through multiple updated facts | Not explicitly addressed |

Our decay model handles *staleness* -- information becomes harder to retrieve over time -- but MemoryAgentBench tests *contradiction* -- information must be *actively replaced* when a newer version arrives. These are complementary, not overlapping.

Our priority #3 (graded contradiction detection + temporal invalidation) is much more directly relevant to what MemoryAgentBench tests. Specifically, the `valid_from`/`valid_until` temporal fields and LLM-based contradiction classification at `remember()` time would directly address the single-hop case. The multi-hop case would additionally require our priority #2 (relationship edges) to propagate invalidation through fact chains.

---

## Relevance to claude-memory

### As an Evaluation Target

**Could we run our system against this benchmark?** In principle, yes, with adaptation. The benchmark expects an agent interface that:
1. Receives text chunks incrementally
2. Processes/stores them in memory
3. Answers questions using stored memory

Our system's `remember()` + `recall()` interface maps to this. The adaptation needed:
- Write a wrapper that calls `remember()` for each incoming chunk (or triggers auto-capture)
- For each question, call `recall()` and format the response
- The wrapper would need to handle chunk-level ingestion, which differs from our session-level storage

**What would be realistic vs. unrealistic?**
- **AR tasks**: Realistic fit. Our hybrid retrieval (vector + FTS5) would be tested directly.
- **TTL tasks**: Poor fit. Our system doesn't do in-context learning from stored memories; it stores facts and procedures, not demonstrations. We would fail at classification tasks because we don't preserve the full context needed for few-shot learning.
- **LRU tasks**: Poor fit for the same reason. Summarization and holistic comprehension require the full context, not memory retrieval. Our system is designed for targeted recall, not global synthesis.
- **SF tasks**: **Highly relevant.** This directly tests our contradiction detection (#3) and temporal invalidation. We could benchmark our system's ability to answer with updated facts.

**Practical recommendation**: Run FactConsolidation (SF split) against our system as a targeted diagnostic. The 8 instances with both SH and MH variants would reveal whether our `valid_from`/`valid_until` + contradiction detection can handle fact supersession. Skip TTL and LRU -- those test fundamentally different capabilities than what our memory system provides.

### Comparison to LongMemEval

| Dimension | LongMemEval | MemoryAgentBench |
|-----------|-------------|-----------------|
| **Venue** | ICLR 2025 | ICLR 2026 |
| **Scale** | 500 QA pairs, 7 question types | 146 instances, 4 competencies |
| **Competencies** | IE, multi-session, knowledge update, temporal, abstention | AR, TTL, LRU, SF |
| **Context** | 115K (S) / 1.5M (M) tokens | 124K - 1.44M tokens |
| **Interaction** | Static chat history | Incremental multi-turn |
| **Novel datasets** | None (all purpose-built) | EventQA, FactConsolidation |
| **Forgetting** | Not tested (acknowledged as gap) | Tested (SF competency) |
| **Knowledge update** | Tested (16% of questions) | Tested (SF overlaps) |
| **Write-path** | Not tested | Not tested (still feeds chunks) |
| **Evaluation framework** | 3-stage (index/retrieve/read) with 4 control points | Per-competency metrics |

**Which is more relevant for our use case?**

LongMemEval remains more relevant for evaluating our core retrieval quality. Its five-ability taxonomy maps more directly to our real usage: information extraction, multi-session reasoning, temporal reasoning, and abstention are all capabilities we need daily. Its unified framework (Value/Key/Query/Reading) provides diagnostic decomposition that MemoryAgentBench lacks.

However, MemoryAgentBench fills the specific gap LongMemEval acknowledged: **forgetting**. LongMemEval's "knowledge update" questions test whether the system returns the latest info, but they don't test multi-hop chaining through contradictions or the scaling behavior of conflict resolution.

**Recommendation**: Use LongMemEval as the primary retrieval/reasoning benchmark. Use MemoryAgentBench's FactConsolidation split as a targeted forgetting diagnostic. Together they provide better coverage than either alone.

### Impact on Implementation Priorities

**Priority #3 (Graded contradiction detection + temporal invalidation)**: **Urgency increased.** MemoryAgentBench's finding that *all* systems fail at multi-hop forgetting means this is an unsolved problem with high diagnostic value. If our system can even partially handle FactConsolidation-MH, that would represent a meaningful advance. The `valid_from`/`valid_until` approach we planned is necessary but probably insufficient for multi-hop -- we also need invalidation propagation through relationship edges (#2).

**Priority #4 (Decay floor + power-law + dormancy)**: **Unchanged but better scoped.** MemoryAgentBench tests contradiction-based forgetting, not decay-based forgetting. These are orthogonal. Our decay system handles information staleness; FactConsolidation tests information supersession. Both are needed, but they solve different problems. We should not confuse good decay with good conflict resolution.

**Priority #5 (Sleep/consolidation)**: **Slightly reinforced.** The consolidation pipeline's contradiction detection step (Step 4 in sleep-skill) is precisely what FactConsolidation tests. If `/sleep` can detect and resolve contradictions during consolidation, our system would perform better on SF tasks. However, the benchmark tests *online* conflict resolution (handling contradictions as they arrive), not *batch* consolidation. We may need both online (at `remember()` time) and batch (at `/sleep` time) contradiction handling.

**Priority #10 (CMA behavioral probes)**: **FactConsolidation-SH could serve as a ready-made behavioral probe for knowledge update.** Instead of designing our own probe from scratch, we could adapt the MQUAKE edit pairs. This would give us a standardized, validated test. The multi-hop variant is harder to adapt but equally valuable if we can get it working.

**Priority #12 (Reading strategy optimization)**: **No direct impact.** MemoryAgentBench doesn't decompose retrieval vs. reading errors like LongMemEval does. The reading strategy findings from LongMemEval remain our primary guide here.

**New consideration**: MemoryAgentBench reveals a **tension between AR and LRU** at the chunk-size level. Smaller chunks help retrieval but hurt holistic understanding. Our system currently operates at the memory-unit level (each `remember()` call is one unit), which sidesteps this tension -- but if we ever move to automatic chunking of longer inputs, this tension will resurface.

---

## Connections

### To Prior Analyses

**[[agent-output-longmemeval|LongMemEval]]** (ICLR 2025): MemoryAgentBench directly incorporates LongMemEval as one of its AR sub-tasks. Performance on the LongMemEval subset within MemoryAgentBench provides a cross-benchmark calibration point. LongMemEval's knowledge-update questions (16% of its benchmark) are a lighter version of what FactConsolidation-SH tests. LongMemEval explicitly acknowledged forgetting/deletion as a gap in its ethics section -- MemoryAgentBench fills that gap.

**LoCoMo** (Maharana et al.): LoCoMo tests conversational recall with update and correction capabilities. Its knowledge-update questions ("I moved to Texas" superseding "I live in New York") are conceptually similar to FactConsolidation-SH. Hindsight achieved 85.67-89.61% on LoCoMo, suggesting that multi-path retrieval with temporal edges can handle single-hop updates. MemoryAgentBench's multi-hop variant goes further than LoCoMo by requiring chained reasoning through multiple updated facts.

**[[agent-output-continuum|Continuum]]** CMA behavioral probes: Continuum defined six behavioral requirements (persistence, selective retention, retrieval-driven mutation, associative routing, temporal continuity, consolidation). MemoryAgentBench's four competencies partially overlap: AR tests persistence + retrieval, TTL tests selective retention, LRU tests consolidation (in the sense of integrated understanding), SF tests a specific form of retrieval-driven mutation. The CMA probes we planned (#10) could borrow FactConsolidation's methodology for the "knowledge update" probe type.

**[[agent-output-hindsight-paper|Hindsight]]**: Hindsight's TEMPR retain/recall primitives and CARA reflect primitive would be interesting to evaluate on MemoryAgentBench. Their consolidation with history audit trail (storing both old and new versions) directly addresses the conflict resolution problem. Hindsight's 83.6-91.4% on LongMemEval and 85.67% on LoCoMo suggest strong single-hop update handling, but their multi-hop forgetting performance is unknown. If Hindsight-style architectures also fail at multi-hop SF, that confirms the problem is fundamental, not architectural.

**[[agent-output-zep-paper|Zep]]**: Zep's bi-temporal modeling (fact validity time vs. storage time) is precisely the mechanism needed for FactConsolidation. When Zep detects a contradictory update, it sets `valid_until` on the old fact and creates a new fact with `valid_from`. At retrieval time, only currently-valid facts are returned. This should handle SF-SH well. SF-MH would require Zep's graph traversal to follow edges through only currently-valid nodes -- unclear if this is implemented.

**[[agent-output-a-mem|A-Mem]]**: A-Mem's memory evolution (updating existing notes when new information arrives) could address SF-SH -- when an update arrives, the existing note is evolved to reflect the new information. But A-Mem performs in-place mutation without audit trail, which means the original fact is lost. For SF-MH, A-Mem would need its link graph to propagate changes, which the current architecture doesn't support.

**[[agent-output-mem0-paper|Mem0]]**: Tested directly in MemoryAgentBench as one of the structure-augmented RAG baselines. Performance details not separately reported in the aggregate tables, but the paper notes Mem0 as part of the evaluated systems. Mem0's InformationContent() gating and DB-level fact conflict handling should give it an edge on SF-SH, but multi-hop chaining through Mem0's flat fact store would be difficult.

### Benchmark Ecosystem Positioning

The three benchmarks now form a complementary evaluation suite:

| Benchmark | Primary Strength | Weakness for Us |
|-----------|-----------------|----------------|
| **LongMemEval** | Decomposed retrieval diagnostics, 5 abilities | No forgetting, chat-only paradigm |
| **LoCoMo** | Conversational recall with updates | Simpler update scenarios, no multi-hop |
| **MemoryAgentBench** | Forgetting competency, multi-system comparison | Coarse metrics, no retrieval/reading decomposition |

For a comprehensive evaluation of claude-memory, we should:
1. Run LongMemEval for retrieval quality diagnostics
2. Run MemoryAgentBench FactConsolidation for forgetting diagnostics
3. Build ClaudeMemEval for our specific agentic coding use case

---

## Limitations of This Analysis

1. **No reviewer feedback accessible.** The OpenReview forum for the ICLR 2026 submission did not expose reviews through the web interface at the time of this analysis. Reviewer critiques about benchmark design, dataset construction methodology, or evaluation metric choices are unknown.

2. **Per-system results are incomplete.** The paper reports aggregate and category-level results, but individual system performance on each sub-task was not fully extracted from the HTML version. The PDF likely contains more detailed tables.

3. **The "selective forgetting" framing may overstate novelty.** LongMemEval's knowledge-update questions and LoCoMo's update/correction tasks test related capabilities. MemoryAgentBench's contribution is the multi-hop variant and the explicit framing as "forgetting," but single-hop conflict resolution was already being tested.

4. **146 instances is small.** Compared to LongMemEval's 500 QA pairs or LoCoMo's 7,512 QA pairs, MemoryAgentBench's 146 instances (only 8 for the key SF competency) provide limited statistical power. Single-run variance on LLM outputs could significantly affect results.

5. **Write-path is still untested.** Like LongMemEval, MemoryAgentBench feeds information to the agent rather than testing what the agent *chooses* to memorize. Our human-in-the-loop curation model (where `remember()` is an explicit action) remains untested by any public benchmark.

---

## Summary Assessment

MemoryAgentBench makes three contributions that matter for our project:

**First**, it establishes selective forgetting / conflict resolution as a first-class memory competency and shows that *every existing system fails catastrophically at multi-hop forgetting*. This validates our investment in contradiction detection (#3) and temporal invalidation, while revealing that single-fact resolution is insufficient -- we need invalidation propagation through relationship chains.

**Second**, it provides ready-made evaluation material. The FactConsolidation dataset (from MQUAKE counterfactual edit pairs) can be directly adapted as a behavioral probe (#10) for our system's knowledge-update capability. The 6K/32K/64K/262K context-length ladder provides a scaling diagnostic.

**Third**, it reveals fundamental architectural tensions. The chunk-size tradeoff between AR and LRU, the backbone-model ceiling for RAG methods, and the dramatic gains from agentic approaches with better models all inform our architecture. The finding that agentic methods (MIRIX) have more headroom than RAG methods with stronger backbones supports our design choice of an agentic memory system over pure RAG.

**What it does NOT provide**: retrieval/reading decomposition (LongMemEval is better for this), realistic agentic coding scenarios (no benchmark tests this), write-path evaluation, or natural decay testing. Our planned decay (#4) and dormancy system addresses a different forgetting problem than what MemoryAgentBench tests.

**Bottom line**: MemoryAgentBench is our best available diagnostic for conflict resolution / knowledge update. LongMemEval remains our best available diagnostic for retrieval quality. Neither replaces the need for ClaudeMemEval -- a benchmark grounded in our actual use case.

---

## Sources

- [arXiv:2507.05257](https://arxiv.org/abs/2507.05257) -- Paper (HTML v2)
- [GitHub: HUST-AI-HYZ/MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench) -- Code repository
- [HuggingFace: ai-hyz/MemoryAgentBench](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench) -- Dataset
- [OpenReview](https://openreview.net/forum?id=DT7JyQC3MR) -- ICLR 2026 submission
- [Emergent Mind](https://www.emergentmind.com/topics/memoryagentbench) -- Summary
