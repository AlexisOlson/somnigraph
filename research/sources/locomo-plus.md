# LoCoMo-Plus: Beyond-Factual Cognitive Memory Evaluation Framework for LLM Agents -- Analysis

*Phase 15, 2026-03-06. Analysis of arXiv:2602.10715 (Feb 2026).*

---

## 1. Paper Overview

**Paper:** "LoCoMo-Plus: Beyond-Factual Cognitive Memory Evaluation Framework for LLM Agents"
**Authors:** Yifei Li, Weidong Guo, Lingling Zhang, Rongman Xu, Muye Huang, Hui Liu, Lijiao Xu, Yu Xu, Jun Liu
**Submitted:** February 11, 2026 (arXiv:2602.10715)
**Venue:** Preprint (cs.CL, cs.AI). 16 pages, 8 figures.
**Code:** https://github.com/xjtuleeyf/Locomo-Plus

**Problem:** Existing conversational memory benchmarks -- including the original LoCoMo -- test only *factual* memory: explicitly stated information with well-defined ground-truth answers. Real conversational memory depends heavily on *implicit constraints* -- user states, goals, values, and causal context that are never directly queried but must still shape agent behavior. Existing evaluation methods compound this gap through two systematic biases: task disclosure (telling the model what is being tested) and surface-form metrics (BLEU/ROUGE/F1 penalizing valid alternative answers).

**Contribution:** LoCoMo-Plus extends the original LoCoMo benchmark with:
1. A taxonomy of **cognitive memory** decomposed into four latent constraint types (causal, state, goal, value)
2. A six-stage construction pipeline that enforces **cue-trigger semantic disconnect** -- trigger queries have low surface similarity to their dependency cues
3. A **constraint-consistency evaluation framework** replacing string-matching metrics with validity-space membership assessment
4. Systematic analysis of **input-side** (task disclosure) and **output-side** (generation length) evaluation biases

**Scale:** The paper does not report exact instance counts, stating it "prioritizes diagnostic coverage over scale" and is "unsuitable for training." Cognitive memory instances are embedded into the original LoCoMo's 10 long dialogue trajectories. This is a diagnostic benchmark, not a training dataset.

---

## 2. Benchmark Design

### What's New vs. Original LoCoMo

The original LoCoMo (Maharana et al., ACL 2024) provided five reasoning types: single-hop, multi-hop, temporal, open-domain, adversarial. All five test **explicit factual recall** -- the answer exists verbatim (or nearly so) in the conversation history, and the question directly references what it wants.

LoCoMo-Plus adds a fundamentally different layer: **Level-2 Cognitive Memory**.

### The Two Levels

| Level | Memory Type | Answer Source | Evaluation | Example |
|-------|------------|---------------|------------|---------|
| Level-1 | Factual | Explicitly stated in history | Single ground-truth, string-matchable | "What's Emily's dog's name?" |
| Level-2 | Cognitive | Implicit constraint inferred from behavior | Valid response space, constraint-consistency | Cousin's diabetes diagnosis → user's changed relationship with food → self-perception shift (query never mentions diabetes) |

### Four Cognitive Memory Subtypes

1. **Causal:** Past events or conditions affect later behavior. A cousin's diabetes diagnosis causes complete dietary changes, later reflected in self-perception shifts *without explicit mention of the original cause*.
2. **State-based:** Physical or emotional states drive subsequent behavior. Workplace criticism causes cautious preventive actions weeks later *without restating the initial anxiety*.
3. **Goal-oriented:** Long-term intentions shape current choices and may be re-evaluated. Saving for a vintage car contrasts with later recognition that happiness comes from camping -- *requires recognizing potential goal evolution*.
4. **Value-based:** Beliefs and values constrain reactions across contexts. Prioritizing team well-being over profit connects to later work-life balance tensions *without explicit contradiction*.

### Cue-Trigger Semantic Disconnect

This is the paper's central design innovation. Traditional benchmarks allow "shortcut solutions" -- the query shares enough keywords with the stored information that simple embedding similarity or BM25 retrieval finds the right chunk. LoCoMo-Plus systematically breaks this:

**Construction pipeline (6 stages):**

1. **Implicit cue generation:** LLMs produce short dialogues conveying memory-relevant information through natural conversation, not explicit facts. Candidate set c0.
2. **Memory-worthy verification:** Human annotators retain only instances that convey "persistent or behaviorally constraining information that cannot be trivially inferred from local context alone." → c1
3. **Cue-trigger query construction:** LLMs generate downstream trigger queries maintaining "low surface-level semantic similarity" while ensuring correct resolution depends on the cue. → (c1, q, t)0
4. **Semantic filtering:** BM25 and MPNet-based scoring remove cases with high lexical or semantic overlap. → (c1, q, t)1
5. **Elicitation validation:** Human annotators verify that responses require "recalling and applying information from the cue through an implicit constraint, rather than relying on surface similarity." → (c1, q, t)2
6. **Insertion into LoCoMo dialogues:** Validated instances embedded into long conversation trajectories with consistent temporal gaps and realistic interference.

The key filtering step (Stage 4) is what makes this benchmark fundamentally different from its predecessors. By explicitly removing cases where retrieval-based shortcut solutions would work, LoCoMo-Plus tests whether systems *understand* conversational context rather than merely *retrieving* it.

### Constraint-Consistency Evaluation

The paper identifies two systematic biases in existing evaluation:

**Input-side bias (task disclosure):** Telling the model "this is a temporal reasoning question" shifts its behavior. The paper shows that temporal and adversarial task scores are disproportionately inflated under task disclosure, meaning reported gains partly reflect "sensitivity to task prompts rather than stable memory behavior." LoCoMo-Plus presents queries "as natural dialogue continuations without explicit task disclosure."

**Output-side bias (generation length):** EM, F1, BLEU, and ROUGE all peak near the ground-truth reference length (5.18 tokens on average) and degrade as outputs become shorter or longer. This means models with different generation styles are penalized or favored "based on length alone, regardless of semantic correctness." LoCoMo-Plus evaluates whether responses satisfy implicit constraints -- correctness is "membership in a valid response space, allowing multiple acceptable realizations."

**LLM-as-judge validation:**
- Human1 vs. Human2 agreement: 0.903
- Human1 vs. LLM Judge: 0.801
- Human2 vs. LLM Judge: 0.820
- Judge stability across backbone models: |delta| ranges from 0.68 (Qwen2.5-14B) to 3.33 (Qwen2.5-7B)

---

## 3. Key Claims and Evidence

### Claim 1: Cognitive memory is substantially harder than factual memory

**Evidence: Strong.** Every model, retrieval method, and memory system shows a consistent performance gap between LoCoMo (factual) and LoCoMo-Plus (cognitive):

| Category | System | LoCoMo | LoCoMo-Plus | Gap |
|----------|--------|--------|-------------|-----|
| Best closed-source | Gemini-2.5-pro | 71.78% | 45.72% | 26.06% |
| Best open-source | Qwen2.5-14B | 63.45% | 44.21% | 19.24% |
| Best memory system | A-Mem (GPT-4o) | 59.64% | 42.44% | 17.20% |
| Best RAG | text-embedding-large (GPT-4o) | 45.32% | 29.77% | 15.55% |

The consistency across architecturally diverse systems (15-26 point drops across the board) makes a strong case that the difficulty is inherent to cognitive memory tasks, not an artifact of any particular approach.

**Assessment:** The universal degradation pattern is convincing. The gap is not just noise -- it is structurally consistent, with larger models showing *larger* absolute gaps (Gemini-2.5-pro drops 26 points vs. Qwen2.5-3B dropping only 11). This suggests that factual memory performance is partially "free" from general capability, while cognitive memory is genuinely harder.

### Claim 2: Existing memory systems don't solve cognitive memory

**Evidence: Moderate.** Memory systems (Mem0, SeCom, A-Mem) all outperform RAG baselines on LoCoMo-Plus, but none close the gap:

| Memory System | LoCoMo-Plus Score | vs. Best Full-Context |
|---------------|-------------------|----------------------|
| A-Mem | 42.44% | -3.28% vs. Gemini-2.5-pro |
| SeCom | 42.63% | -3.09% |
| Mem0 | 41.44% | -4.28% |

All three memory systems use GPT-4o as backbone, which scores 41.94% on LoCoMo-Plus with full context. The memory systems are essentially *matching* full-context performance, not exceeding it. This mirrors the original LoCoMo's full-context paradox -- at these scales, the memory systems add complexity without clear benefit for cognitive tasks.

**Assessment:** The evidence for "memory systems don't help" is somewhat weakened by the lack of testing with stronger backbone models (Gemini-2.5-pro at 45.72% with full context was not tested as a backbone for the memory systems). Still, the pattern is clear: cognitive memory isn't solved by retrieval improvements alone.

### Claim 3: Task disclosure inflates benchmark scores

**Evidence: Moderate-to-strong.** The paper demonstrates distribution shifts between task-disclosed and unified evaluation, particularly in temporal and adversarial categories. However, specific numerical performance changes are presented only in figures (Figure 5), not in tables with exact values.

### Claim 4: String-matching metrics are systematically biased

**Evidence: Strong.** The generation-length analysis (Figure 6) is particularly clean: all four standard metrics (EM, F1, BLEU, ROUGE) show clear inverted-U curves peaking at the ground-truth reference length of 5.18 tokens. This is a structural property of the metrics, not debatable.

### Claim 5: Cognitive memory degrades faster with context length

**Evidence: Moderate.** Figure 7 shows "rapid performance collapse" for cognitive memory as context length increases, contrasting with more stable object memory. However, the specific context lengths and exact performance numbers are not reported in the text.

---

## 4. Standout Feature

**What makes LoCoMo-Plus unique among the three major benchmarks:**

| Dimension | LoCoMo (original) | LongMemEval | MemoryAgentBench | **LoCoMo-Plus** |
|-----------|-------------------|-------------|------------------|-----------------|
| Core innovation | Long conversations | Five memory abilities | Four competencies | **Semantic disconnect** |
| Memory level | Explicit facts | Explicit facts | Explicit facts + behavior | **Implicit constraints** |
| Evaluation | String-matching + LLM judge | LLM judge per type | SubEM + accuracy | **Constraint-consistency** |
| Scale | ~9K tokens | ~115K-1.5M tokens | 124K-1.44M tokens | Diagnostic (extends LoCoMo) |
| Retrieval difficulty | Standard | Needle-in-haystack | Incremental accumulation | **Anti-shortcut filtered** |

The standout is the **deliberate destruction of retrieval shortcuts.** Every other benchmark assumes that if you can find the right chunk, you can answer the question. LoCoMo-Plus attacks this assumption: the right chunk might be found, but understanding its *implications* for the current context is the actual challenge. This is the first benchmark to formally separate "retrieval" from "comprehension of retrieved context."

This maps directly onto our prior finding from [[agent-output-longmemeval]]: "~50% of errors are reading failures, not retrieval failures." LoCoMo-Plus is the first benchmark *designed* to test this reading/comprehension layer specifically.

---

## 5. Competency Coverage Ratings

| Competency | Coverage | Justification |
|------------|----------|---------------|
| **Information Retrieval** | 75% | Inherits all five LoCoMo question types for factual recall. The cognitive memory layer adds implicit retrieval -- finding relevant constraints when the query doesn't lexically match. Does not test retrieval mechanism quality directly (no recall@k). |
| **Multi-Session Reasoning** | 60% | Multi-hop questions from original LoCoMo test cross-session synthesis. Cognitive memory subtypes (especially causal and goal) require implicit cross-session reasoning. But no formal multi-session structure beyond LoCoMo's existing sessions. |
| **Knowledge Update/Contradiction** | 25% | Not a focus. No explicit contradiction detection or knowledge update tasks. The goal-oriented cognitive subtype touches on goal *evolution*, but this is a minor element. MemoryAgentBench's FactConsolidation tests this far more thoroughly. |
| **Temporal Reasoning** | 40% | Inherits LoCoMo's temporal question type. Context-length degradation analysis examines temporal aspects. But cognitive memory subtypes don't specifically test temporal ordering beyond the causal subtype's implicit temporal dependency. |
| **Abstention/Confidence** | 15% | Inherits LoCoMo's adversarial questions (designed to induce hallucination). The task-disclosure analysis reveals that adversarial scores are inflated by explicit task framing. But no formal abstention evaluation -- no "I don't know" ground truth for cognitive memory tasks. |
| **Write-Path Behavior** | 0% | No testing of memory storage, indexing, or write-time processing. The benchmark evaluates read-path only. |
| **Consolidation Quality** | 0% | No sleep, consolidation, or memory management evaluation. |
| **Proactive/Contextual Recall** | 45% | The constraint-consistency framework essentially tests this: can the system *proactively apply* implicit constraints without being explicitly asked? The unified (non-task-disclosed) evaluation format requires the model to recognize when a constraint is relevant. However, this is tested in a QA format, not in genuine proactive recall scenarios. |
| **Relationship/Graph Reasoning** | 20% | Causal and value-based cognitive subtypes require understanding relationships between events/beliefs and behavior. But no explicit graph structure or relationship traversal testing. Paper does not discuss PPR, knowledge graphs, or graph-based retrieval at all. |
| **Agentic Task Performance** | 5% | Not an agentic benchmark. Models receive queries and produce responses; no tool use, no iterative search, no autonomous behavior tested. |

**Aggregate coverage:** 29%. LoCoMo-Plus is narrow but deep -- it tests one thing (implicit constraint comprehension) better than any other benchmark. It does not attempt breadth.

---

## 6. Relevance to claude-memory

### Could We Run Against This?

**Partially, with adaptation.** The benchmark extends LoCoMo's 10 conversations, so the base data is available. The cognitive memory instances and the constraint-consistency evaluation framework require the LoCoMo-Plus code (GitHub repo available).

**Key adaptations needed:**

1. **Ingestion pipeline.** We would need to convert LoCoMo conversations into our memory format -- either by treating each turn as a `remember()` call or by pre-processing conversations into summary+themes memories (closer to how we actually use the system).

2. **Evaluation format.** The constraint-consistency evaluation requires an LLM judge, which we can implement. The non-task-disclosed query format maps naturally to our `recall()` interface.

3. **Scale mismatch.** LoCoMo conversations are ~9K tokens. Our system holds ~386 memories spanning months of real interaction. The benchmark's conversations are far shorter than our actual use case, meaning the benchmark may *underestimate* the difficulty we face in production (more interference, longer temporal gaps, more implicit constraints accumulated).

4. **Memory type mismatch.** Our memories are curated summaries with themes, not raw conversation turns. This curation may *help* with cognitive memory tasks (summaries capture implicit constraints better than verbatim text) or *hurt* (curation might strip context needed for constraint inference).

### What Would It Reveal?

1. **Whether our enriched embeddings handle semantic disconnect.** Our text-embedding-3-small embeddings are built from content+category+themes+summary. If the summary captures the implicit constraint (e.g., "dietary change after cousin's diabetes diagnosis"), retrieval might work even when the query doesn't share surface terms. This is the most interesting test for us.

2. **Whether BM25 field weighting compensates.** Our curated `summary` field (5x weight in BM25) contains human-authored descriptions that may bridge the cue-trigger gap. If the summary says "changed eating habits after family health scare," BM25 on that text might find it for a query about food preferences -- even though verbatim conversation text wouldn't match.

3. **Whether adjacency expansion helps.** If a memory about the cousin's diagnosis is linked (via memory_edges) to a memory about dietary changes, and that is linked to a memory about self-perception, our novelty-scored edge expansion during recall could bridge the gap that pure vector/BM25 search cannot.

4. **Reading vs. retrieval bottleneck.** If we retrieve the right memories but still fail cognitive memory questions, the bottleneck is in how the calling model (Opus) integrates retrieved context -- not in our retrieval system. This would be consistent with LongMemEval's "50% reading failure" finding.

### Verdict

Running against LoCoMo-Plus would be informative for testing our enriched-embedding and adjacency-expansion strategies specifically on semantic-disconnect queries. It would not test write-path, consolidation, decay, or any of the sleep pipeline. Worth doing as a targeted retrieval quality test, but not as a comprehensive system evaluation.

---

## 7. Insights Worth Stealing

**Ranked by effort/impact:**

### 1. Anti-shortcut filtering for our own test suite (effort: low, impact: high)

The BM25+MPNet semantic filtering in Stage 4 is directly applicable to any retrieval test set we build. When creating recall test cases, we should systematically check whether the query shares high lexical/semantic overlap with the target memory. If it does, the test case is trivially solvable by any retrieval system and doesn't test what matters. Filter or flag these.

**Implementation:** For each (query, target_memory) pair in a test set, compute BM25 and cosine similarity. Discard pairs above a threshold. This is a 20-line script.

### 2. Constraint-consistency as evaluation paradigm (effort: medium, impact: high)

Our recall_feedback system currently uses a single utility float (0-1). For evaluating *cognitive* recall quality, we could adopt constraint-consistency: did the retrieved memories contain the information needed to satisfy the implicit constraint, even if the specific answer wasn't in the memory text?

**Implementation:** During feedback, ask not "was this memory useful?" but "did this memory provide a constraint that shaped the response?" This would require modifying how we think about feedback during sessions -- currently it is outcome-focused, but constraint-consistency is input-focused.

### 3. Non-task-disclosed evaluation format (effort: low, impact: medium)

When testing our system, we should present queries as natural continuations, not as "retrieve information about X." Task-disclosed queries inflate performance by ~5-20% (estimated from Figure 5), meaning our own ad-hoc testing likely overestimates recall quality.

**Implementation:** When building test cases, frame queries as a user would naturally ask ("I'm thinking about what to eat tonight" vs. "Recall my dietary restrictions").

### 4. The four cognitive subtypes as a taxonomy for memory quality (effort: low, impact: medium)

When reviewing memories during `/reflect` or `consolidate()`, we could classify memories by which cognitive subtypes they serve: Does this memory capture a *causal* constraint (X happened, which means Y)? A *state* (emotional/physical condition)? A *goal* (intention that may evolve)? A *value* (belief that constrains behavior)?

Memories that serve none of these are purely factual. Memories that serve multiple are high-value. This taxonomy could inform priority scoring.

### 5. Generation-length bias awareness (effort: zero, impact: low-but-good-to-know)

If we ever use string-matching metrics (BLEU, ROUGE, F1) for anything, the paper's finding about length bias is a useful caution. Our LLM-judge approach for recall_feedback sidesteps this, but it is worth knowing.

---

## 8. What's Not Worth It

1. **The dataset itself as our benchmark.** LoCoMo conversations are ~9K tokens -- smaller than a single startup_load. Our system operates at 386+ memories accumulated over months. The scale mismatch means performance on LoCoMo-Plus would not predict performance in our actual use case. Building our own test suite from real recall patterns (as we did in retrospective experiments) remains more informative.

2. **The four cognitive subtypes as a memory schema.** While useful as an evaluation taxonomy (see "Insights Worth Stealing" #4), these subtypes don't map onto our memory categories (episodic, procedural, semantic, reflection, meta). Our categories describe *what kind of thing is stored*; the cognitive subtypes describe *what kind of constraint a memory provides*. These are orthogonal dimensions. Trying to merge them would over-complicate the schema.

3. **Replicating the six-stage construction pipeline.** The pipeline is designed for building benchmark instances at scale with LLM generation + human filtering. We have ~386 real memories, not synthetic ones. Our test cases should come from real recall patterns, not synthetic cue-trigger pairs.

4. **The specific model results.** The leaderboard (Gemini-2.5-pro at 45.72%, GPT-4o at 41.94%, etc.) tells us nothing actionable about our system. These results test backbone model capability, not memory architecture.

5. **Per-subtype performance breakdowns.** The paper does not report results by causal/state/goal/value subtype -- only aggregate LoCoMo-Plus scores. Even if they did, the subtypes are not well-enough defined to be operationalizable in our system.

---

## 9. Key Takeaway

LoCoMo-Plus formalizes what our system has encountered informally: the hardest recall failures happen not because the right memory is missing from retrieval results, but because the *connection* between the query context and the relevant memory requires inference across a semantic gap. A user asking about food preferences needs a memory about a cousin's diabetes diagnosis -- and no amount of embedding similarity between "food preferences" and "cousin diagnosed with diabetes" will bridge that gap through vector search alone. This is the first benchmark to systematically construct and measure this kind of failure, and the universal 15-26 point performance drop across all tested systems (including specialized memory architectures) confirms that cognitive memory is a genuinely unsolved problem. For claude-memory, the most actionable implication is that our edge graph and adjacency expansion -- which can traverse from "food preferences" to "dietary changes" to "family health events" -- may be our strongest asset for cognitive memory tasks, if we activate it in the retrieval path. The paper also validates our investment in curated summaries over verbatim storage: summaries that capture *why* something matters (the implicit constraint) rather than *what* was said (the surface text) are precisely what cognitive memory requires.

---

## 10. Impact on Implementation Priority

**Strengthens the case for:**

- **PPR/graph-based retrieval (#3-ish on priority list):** The paper's core finding -- that semantic disconnect defeats all pure vector/BM25 retrieval approaches -- is the strongest empirical argument yet for activating our edge graph in the retrieval path. Our novelty-scored adjacency expansion was designed for exactly this scenario: traversing from the query's semantic neighborhood to related memories that don't share surface terms. LoCoMo-Plus provides the empirical grounding that this matters.

- **Enriched embeddings (already implemented):** Our decision to embed content+category+themes+summary rather than content alone is validated. The paper shows that surface-form matching systematically fails for cognitive memory. Our summary field -- which captures "why" not just "what" -- is the right strategy.

- **Summary quality in write path:** If summaries are the bridge across semantic disconnect, summary quality during `remember()` becomes even more important. A summary that says "cousin's diabetes diagnosis" is better than one that says "family conversation about health" -- the former captures the causal constraint, the latter does not.

**Neutral on:**

- **Decay model, sleep pipelines, consolidation:** LoCoMo-Plus doesn't test any of these. No impact on those priorities.

- **Hebbian PMI boost, feedback scoring:** These affect *ranking* of retrieved results, not whether the right results are retrieved at all. For cognitive memory, the bottleneck is retrieval coverage (finding the semantically disconnected memory), not ranking precision.

**Minor implications:**

- **Theme normalization:** Well-normalized themes could help bridge semantic gaps. If "dietary-change" and "family-health" are both themes on the relevant memory, and the query context touches either theme, FTS5 on themes could provide the cross-domain bridge. This supports continued investment in theme quality.

---

## See Also

- [[agent-output-locomo]] -- Original LoCoMo analysis, full-context paradox, Letta filesystem evaluation
- [[agent-output-longmemeval]] -- LongMemEval benchmark, "50% reading failures" finding, five memory abilities framework
- [[agent-output-memoryagentbench]] -- MemoryAgentBench, four competencies (AR/TTL/LRU/SF), selective forgetting analysis
- [[retrospective-experiments]] -- Our own retrieval experiments confirming FTS+Vec hybrid value, enriched embedding advantage
- [[agent-output-a-mem]] -- A-Mem memory system (tested in LoCoMo-Plus, 42.44% cognitive memory score)
