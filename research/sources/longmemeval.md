# LongMemEval: Benchmarking Chat-Based Long-Term Memory -- Analysis

*Generated 2026-02-18 by Opus agent reading 2410.10813v2*

Published as a conference paper at ICLR 2025. Authors: Di Wu (UCLA), Hongwei Wang, Wenhao Yu (Tencent AI Lab Seattle), Yuwei Zhang (UC San Diego), Kai-Wei Chang (UCLA), Dong Yu (Tencent). Code/data at https://github.com/xiaowu0162/LongMemEval.

---

## Paper Overview

LongMemEval is a benchmark for evaluating long-term memory in chat assistants -- not a memory system itself. 500 manually curated questions test five core memory abilities across seven question types, embedded within freely scalable user-assistant chat histories. Two standard configurations: LongMemEval_S (~115k tokens, ~50 sessions) and LongMemEval_M (500 sessions, ~1.5M tokens).

The paper also provides a unified three-stage framework (indexing, retrieval, reading) with four control points (value, key, query, reading strategy) for analyzing any memory system, and proposes optimizations at each stage that collectively yield significant gains.

**Core finding**: Commercial chat assistants (ChatGPT, Coze) show 30-64% accuracy drops compared to offline reading. Long-context LLMs show 30-60% drops on LongMemEval_S. Even with perfect retrieval, suboptimal reading strategy costs up to 10 absolute points.

---

## Architecture / Method (Benchmark Design)

### The Five Core Memory Abilities

| Ability | Abbrev | What It Tests |
|---------|--------|---------------|
| Information Extraction | IE | Recall specific facts from either user or assistant utterances within a single session |
| Multi-Session Reasoning | MR | Synthesize information across 2-6 sessions (aggregation, comparison, counting) |
| Knowledge Updates | KU | Recognize changes in user state over time; answer with the *latest* info, not stale |
| Temporal Reasoning | TR | Reason with both explicit time mentions in text and timestamp metadata |
| Abstention | ABS | Identify questions about information never mentioned; answer "I don't know" |

### The Seven Question Types

| Type | Ability Tested | % of Dataset | Example |
|------|---------------|--------------|---------|
| single-session-user | IE | 14% | "How long is my commute?" (user mentioned 45 min in a session) |
| single-session-assistant | IE | 11% | "What romantic restaurant did you recommend in Rome?" (assistant said Roscioli) |
| single-session-preference | IE (personalization) | 6% | "Suggest accessories for my photography setup" (system must recall Sony user) |
| multi-session | MR | 27% | "How many musical instruments do I own?" (4, across 4 sessions) |
| knowledge-update | KU | 16% | "Where was my most recent family trip?" (Paris, not Hawaii -- updated) |
| temporal-reasoning | TR | 27% | "How many months since my last museum visit with a friend?" (5 months) |
| abstention | ABS | ~6% (30 Qs) | "How many fish in my 30-gallon tank?" ("You never mentioned a 30-gallon tank") |

### Data Construction Pipeline

1. **Attribute ontology**: 164 user attributes across 5 categories (demographics, lifestyle, situational context, life events, belongings). Detailed subcategories (Table 5 in appendix).
2. **Background sampling**: LLM generates a user background paragraph per attribute.
3. **Question construction**: LLM proposes (question, answer) pairs; human experts filter and rewrite (only ~5% yield rate). Answers decomposed into evidence statements with optional timestamps.
4. **Evidence session construction**: Self-chat simulation where user LLM conveys evidence *indirectly* (e.g., asking about car insurance rather than stating "I bought a car"). Human-edited ~70% of sessions.
5. **History compilation**: Needle-in-a-haystack approach. Evidence sessions embedded within padding sessions drawn from ShareGPT (25%), UltraChat (25%), and simulated sessions (50%). Timestamps assigned with evidence sessions as anchors.

### Evaluation

- **QA Accuracy**: GPT-4o judge with prompt-engineered per-type instructions. Meta-evaluation: >97% agreement with human experts.
- **Memory Recall**: Recall@k and NDCG@k using human-annotated answer location labels.
- Separate evaluation prompts for different types (temporal gets off-by-one tolerance, knowledge-update accepts old+new as long as new is present, preference checks rubric satisfaction).

### Unified Memory Framework

Three stages, four control points:

| Stage | Control Point | Design Space |
|-------|--------------|--------------|
| Indexing | **CP1: Value** -- granularity/format | Session, round, summary, facts |
| Indexing | **CP2: Key** -- what to index on | K=V, K=fact, K=keyphrase, K=V+fact (expansion) |
| Retrieval | **CP3: Query** -- how to search | Raw question, time-aware expansion |
| Reading | **CP4: Reading strategy** | Direct, Chain-of-Note, JSON vs NL format |

Nine existing systems mapped to this framework (Table 2): In-context RAG, MemoryBank, LD-Agent, CoN, ChatGPT, Coze, RAPTOR, MemWalker, HippoRAG.

---

## Key Claims & Evidence

### Claim 1: Round-level decomposition is the sweet spot for value granularity.
**Evidence**: Figure 5 -- decomposing sessions into individual rounds significantly improves QA with GPT-4o as reader. Replacing with extracted facts/summaries hurts performance due to information loss, *except* for multi-session reasoning where fact decomposition helps by normalizing format across sessions.

**Our reading**: Validates our choice to store per-round data. But the nuance matters: facts help multi-session but hurt single-session. Suggests a dual-representation approach (store rounds, also extract facts as supplementary keys).

### Claim 2: Key expansion with extracted user facts yields the best retrieval.
**Evidence**: Table 3 -- K=V+fact gives 9.4% higher recall@k and 5.4% higher accuracy over K=V baseline. Rank merging (separate indices, fuse after retrieval) performs *worse* than key merging (concatenate fact to value before embedding). Table 10 confirms rank merging drops recall substantially.

**Our reading**: This is directly relevant to our RRF fusion priority. Their finding that key merging > rank merging challenges the assumption that separate retrieval paths + RRF is always superior. The concatenation approach avoids index explosion and benefits from the embedding model capturing both original and expanded content. However, they only tested basic rank merging -- sophisticated RRF with cross-encoder reranking (as in Hindsight/Zep) might perform differently.

### Claim 3: Time-aware query expansion is essential for temporal reasoning.
**Evidence**: Table 4 -- with GPT-4o for extraction, recall improves 6.8-11.3% on temporal subset. Llama 3.1 8B fails badly (hallucinated time ranges, false positives on non-temporal queries). Examples in Table 11: weak models extract spurious time ranges for questions with no temporal reference.

**Our reading**: Temporal reasoning is 27% of LongMemEval. Our decay model handles staleness but not explicit temporal queries ("What did I do last March?"). This is a gap. Their approach is simple: extract timestamped events at index time, extract time ranges at query time, filter. Requires a strong LLM for query expansion.

### Claim 4: Chain-of-Note + JSON format yields best reading performance.
**Evidence**: Figure 6 -- under oracle retrieval, CoN + JSON yields 0.924 for GPT-4o (vs 0.862 for NL + Direct). Up to 10 absolute points difference. Without CoN, JSON doesn't consistently help; with CoN, JSON consistently helps.

**Our reading**: Reading strategy matters more than expected. Even perfect retrieval doesn't guarantee correct answers. The extract-then-reason decomposition is reminiscent of our summary-first approach.

### Claim 5: Substantial reading errors persist even with perfect retrieval.
**Evidence**: Figure 14 error analysis -- 15-19% of all instances have correct retrieval but wrong generation (40-50% of error instances). Higher with weaker readers.

**Our reading**: This means ~half of all failures are reading failures, not retrieval failures. Memory system design should invest in reading strategy, not just retrieval optimization.

### Commercial System Failures (Table 7)
- **ChatGPT**: Good at single-session IE (1.00 with mini, 0.688 with 4o). Drops sharply on MR/TR. Overwrites information during compression.
- **Coze**: Fails to record indirectly provided information. MR accuracy as low as 0.118 (GPT-3.5) / 0.147 (GPT-4o).
- Key insight: Both use fact extraction as primary representation, confirming that fact-only storage causes information loss.

---

## Relevance to claude-memory

### Direct Applicability: High

LongMemEval tests exactly the capabilities a personal assistant memory needs:

1. **Information Extraction** -- Our core use case. Can the system recall what was discussed?
2. **Multi-Session Reasoning** -- Our cross-session synthesis. "How many times has the user mentioned X?" requires aggregating across memories.
3. **Knowledge Updates** -- Our contradiction detection priority (#3). "User moved from Austin to Seattle" -- does the system return the current state?
4. **Temporal Reasoning** -- Partially addressed by our timestamp indexing, but LongMemEval tests *relative* temporal reasoning ("How many months since...") which requires calculation, not just recency.
5. **Abstention** -- Our confidence scores priority (#7). Knowing what you don't know.

### What LongMemEval Tests That Maps to Our Priorities

| LongMemEval Ability | Our Priority | Status |
|---------------------|-------------|--------|
| Multi-session reasoning | #9 Multi-angle indexing | Not started |
| Knowledge updates | #3 Graded contradiction detection | Not started |
| Temporal reasoning | #4 Decay + power-law | Partially (staleness, not temporal queries) |
| Abstention | #7 Confidence scores | Not started |
| Information extraction | Core retrieval (RRF, hybrid search) | In progress |

### What LongMemEval Does NOT Test

1. **Procedural memory** -- No "how to do X" queries. All questions are episodic/semantic about user state.
2. **Proactive recall** -- System never volunteers information; always responds to explicit questions.
3. **Consolidation quality** -- No test for whether consolidation preserves or destroys information.
4. **Write-path behavior** -- Benchmark feeds sessions directly; doesn't test what the system *chooses* to memorize.
5. **Human-in-the-loop curation** -- No test for user correction/editing of memories.
6. **Contradiction *detection* as a standalone ability** -- Knowledge-update tests whether the system returns the latest info, not whether it explicitly recognizes and flags the contradiction.
7. **Relationship/graph reasoning** -- No "How are X and Y connected?" queries.
8. **Scale beyond personal facts** -- All questions are about one user's personal information. No organizational knowledge, shared context, or multi-user scenarios.
9. **Latency** -- DialSim penalizes slow responses; LongMemEval does not.
10. **Memory deletion/forgetting** -- Acknowledged in ethics section as a gap.

### Adaptation Feasibility

LongMemEval is open-source (MIT license) with configurable history lengths. Could be adapted:
- Feed sessions through our `remember()` pipeline instead of raw storage
- Evaluate recall quality at retrieval stage (they provide annotated evidence locations)
- Test our system on the 500 questions with our own indexing/retrieval/reading pipeline
- The padding session mixture (ShareGPT/UltraChat/simulated) could be replaced with Claude Code-style interactions

**Key limitation for us**: LongMemEval assumes a chat assistant paradigm (multi-turn dialogues). Our system operates on coding sessions, tool outputs, and mixed-mode interactions. The attribute ontology (lifestyle, shopping, travel) doesn't map to our usage patterns (debugging sessions, architecture decisions, calibration patterns). We'd need a parallel benchmark grounded in agentic coding workflows.

---

## Worth Stealing (ranked)

### 1. Fact-Augmented Key Expansion (K=V+fact)
**What**: Extract user facts from each stored round, concatenate with the round text to form the embedding key. Don't replace -- augment.
**Why**: 9.4% recall improvement. Simple. No index explosion. Compatible with our existing pgvector setup.
**How**: At `remember()` time, extract structured facts and prepend to text before embedding. Store both the raw text and the fact-augmented key.
**Priority impact**: Strengthens #1 (RRF) by providing a richer embedding surface. Could partially address #9 (multi-angle indexing) without separate indices.

### 2. Time-Aware Query Expansion
**What**: At index time, extract timestamped events. At query time, use LLM to infer a time range, then filter retrieval to that range before similarity search.
**Why**: 11.3% recall improvement on temporal queries. Temporal reasoning is a known weak point.
**How**: Add `event_timestamps` to memory units. On recall, if query has temporal references, extract time range and add as a WHERE clause.
**Priority impact**: Extends #4 (Decay + power-law) from passive staleness to active temporal reasoning.

### 3. Chain-of-Note Reading Strategy
**What**: Before answering, LLM first extracts relevant information from each retrieved item, then reasons over the extracted notes.
**Why**: Up to 10-point improvement even with perfect retrieval. Decomposes long-context reading into two simpler tasks.
**How**: Could be implemented in the recall pipeline: after retrieval, run a summarization pass before returning to the agent.
**Priority impact**: New priority -- reading strategy optimization. Currently not on the roadmap.

### 4. JSON Structured Format for Retrieved Memories
**What**: Present retrieved memories in structured JSON rather than natural language prose.
**Why**: Consistently helps when combined with CoN. Helps the reader model distinguish memory items from the prompt.
**How**: Already partially doing this (our recall returns structured data). Ensure downstream consumers receive JSON-formatted memories.

### 5. The Unified Framework as an Evaluation Lens
**What**: Decompose any memory system into Value/Key/Query/Reading and evaluate each independently.
**Why**: Enables targeted diagnosis. If retrieval is good but QA is bad, the problem is reading, not indexing.
**How**: Instrument our system to log retrieval results separately from final answers. Measure recall@k and end-to-end QA independently.

### 6. Abstention Testing Methodology
**What**: Take real questions, modify them to have false premises, test whether system correctly refuses.
**Why**: 30 questions from the benchmark specifically test this. Simple to construct for our domain.
**How**: Generate false-premise variants of real recall queries. Measure false positive rate.

---

## Not Useful For Us

1. **The padding session construction pipeline** -- We don't need synthetic chat histories. Our evaluation would use real Claude Code sessions.
2. **The specific attribute ontology** (164 attributes in 5 categories) -- Lifestyle/shopping/travel categories don't map to coding assistant usage.
3. **The LLM-simulated evidence sessions** -- Indirect evidence embedding is a good benchmark design choice but doesn't reflect how our users actually convey information (they're usually direct about what they want stored).
4. **The offline reading baseline** -- Useful for the paper's argument but not actionable for us. We already know long-context isn't a substitute for memory.
5. **Rank merging vs key merging comparison** -- Their rank merging implementation is naive (no cross-encoder reranking, no sophisticated fusion). The negative result doesn't generalize to systems like Hindsight that use full RRF + reranking.

---

## Impact on Implementation Priority

### Priorities Confirmed
- **#1 RRF fusion**: Validated but with nuance. Key expansion (concatenation) outperformed naive rank merging. Our RRF should combine dense+sparse on the *expanded* keys, not fuse separate dense-on-facts + dense-on-text pipelines.
- **#3 Graded contradiction detection**: Knowledge-update questions (16% of benchmark) require exactly this. Systems that overwrite (ChatGPT) or fail to record (Coze) both fail.
- **#7 Confidence scores**: Abstention ability directly requires knowing what you don't know.

### Priorities Strengthened
- **#4 Decay + power-law**: Needs extension beyond staleness decay. Add explicit temporal indexing (timestamped events) and time-aware query filtering.
- **#9 Multi-angle indexing**: Fact extraction as key expansion is a lightweight form of multi-angle indexing that doesn't require separate indices.

### New Priority Suggested
- **Reading strategy optimization**: Not currently on the roadmap. LongMemEval shows that 40-50% of errors are reading failures, not retrieval failures. Chain-of-Note + structured format could yield 5-10% improvement with minimal architectural change.

### Priorities Unchanged
- **#2 Relationship edges**: Not tested by LongMemEval. Still needed for our use case but not validated/invalidated here.
- **#5 Sleep skill**: LongMemEval tests the product of consolidation (can you answer?) but doesn't evaluate the process. Our sleep/consolidation design remains driven by other papers (Generative Agents, Hindsight).
- **#6 Mutation log**: Not relevant to this benchmark.
- **#8 Reference index**: Not tested.

---

## Connections

### To Prior Analyses

**[[agent-output-zep-paper|Zep]]** achieved 71.2% on LongMemEval (per our prior notes). Zep uses bi-temporal indexing and triple retrieval via RRF -- precisely the kind of system LongMemEval was designed to evaluate. The 71.2% maps to somewhere between the paper's reported baselines (ChatGPT at 57.7% with GPT-4o, their optimized system at ~72% with top-10 retrieval on LongMemEval_M with round values + fact key expansion).

**[[agent-output-hindsight|Hindsight]]** achieved 83.6-91.4%. Their MPFP graph traversal + typed edges + RRF fusion + cross-encoder reranking addresses several of LongMemEval's challenges: multi-session reasoning (graph traversal connects related facts), temporal reasoning (temporal edges + bi-temporal indexing), knowledge updates (consolidation with history audit trail).

**[[agent-output-cogmem|CogMem]]** Oberauer FoA/DA/LTM maps to LongMemEval's challenge: information in LTM (full history) must be promoted to FoA (retrieved context) for reasoning. CogMem's summary-first/detail-on-demand is analogous to LongMemEval's finding that round > session > summary but facts help multi-session.

**[[agent-output-a-mem|A-Mem]]** +121% on multi-hop aligns with LongMemEval's multi-session reasoning (27% of questions). A-Mem's association links would help aggregate facts across sessions.

**[[agent-output-mem0-paper|Mem0]]** InformationContent() gating addresses the Coze failure mode: Coze fails to record indirectly provided information. Mem0's LLM classifier decides what to store. LongMemEval's indirect evidence embedding specifically challenges this gating decision.

**[[agent-output-dynamic-cheatsheet|Dynamic Cheatsheet]]** "Summaries fail replacing detail" -- directly confirmed by LongMemEval Table 3: fact-only or summary-only keys underperform V+fact expansion.

### Key Synthesis

LongMemEval's most important contribution for us is the decomposed evaluation framework. By separating retrieval metrics (recall@k, NDCG@k) from end-to-end QA accuracy, it reveals that:

1. Retrieval accounts for ~50-60% of errors
2. Reading accounts for ~40-50% of errors
3. These are largely independent failure modes

This means optimizing retrieval alone has diminishing returns. Our roadmap should balance retrieval improvements (#1 RRF, #2 edges, #4 decay, #9 multi-angle) with reading strategy work (CoN, structured output, confidence calibration).

### Benchmark Adoption Recommendation

**Use LongMemEval as one evaluation axis**, but don't treat it as the primary benchmark for claude-memory. Reasons:

1. It tests personal-chat memory, not agentic-coding memory. High scores on LongMemEval wouldn't guarantee our system works well for its actual use case.
2. It doesn't test write-path decisions, consolidation quality, or proactive recall -- all central to our design.
3. It doesn't test latency, which matters for interactive coding.

**Recommended approach**: Build a parallel "ClaudeMemEval" benchmark grounded in real coding sessions, but borrow LongMemEval's five-ability taxonomy and seven-type structure. Use LongMemEval as a secondary validation -- if our system can't handle knowledge updates and temporal reasoning on LongMemEval, it certainly can't handle them in practice.
