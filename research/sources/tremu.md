# TReMu: Towards Neuro-Symbolic Temporal Reasoning for LLM-Agents with Memory in Multi-Session Dialogues -- Analysis

*Phase 15, 2026-03-06. Analysis of arXiv:2502.01630 (ACL Findings 2025).*

---

## 1. Paper Overview

**Paper**: Yubin Ge (University of Illinois Urbana-Champaign), Salvatore Romeo, Jason Cai, Raphael Shu, Monica Sunkara, Yassine Benajiba, Yi Zhang (Amazon Web Services). ACL Findings 2025. arXiv:2502.01630, submitted February 2025.

**Problem addressed**: Temporal reasoning in multi-session dialogues is under-studied. Existing memory benchmarks (LoCoMo, LongMemEval, TimeDial) either do not focus specifically on temporal reasoning or treat it as one dimension among many rather than decomposing it into distinct sub-capabilities. Two characteristics unique to multi-session dialogues make temporal reasoning hard: (1) events are described using *relative* time expressions (e.g., "last Tuesday," "two weeks ago") that require grounding against session timestamps, and (2) temporally related events may be *scattered across sessions* rather than co-located.

**Core contribution**: A benchmark of 600 temporal multiple-choice questions (488 answerable, 112 unanswerable) built on top of LoCoMo dialogues, plus a neuro-symbolic framework (TReMu) that combines time-aware memorization with Python code generation for temporal calculation. The framework raises GPT-4o from 29.83% (standard prompting) to 77.67% accuracy, demonstrating both the difficulty of the task and the effectiveness of structured temporal reasoning.

**Scale**: 600 questions across 3 temporal question types, evaluated on 3 LLMs (GPT-4o, GPT-4o-mini, GPT-3.5-Turbo) against 5 baselines. Dialogues average 600 turns and 16,000 tokens across up to 32 sessions (inherited from LoCoMo).

---

## 2. Benchmark Design

### Temporal Question Types

The benchmark decomposes temporal reasoning into three distinct sub-tasks:

| Type | Count | Definition | Example Pattern |
|------|-------|-----------|----------------|
| **Temporal Anchoring (TA)** | 264 | "When exactly did X happen?" -- resolve a single event's date from relative time expressions | "When did John start his new job?" (mentioned as "last Monday" in session 14) |
| **Temporal Precedence (TP)** | 102 | "Which happened first, X or Y?" -- compare ordering of two events across sessions | "Did John's promotion or Mary's graduation happen first?" |
| **Temporal Interval (TI)** | 234 | "How long between X and Y?" -- compute duration between two cross-session events | "How many weeks passed between the concert and the move?" |

Additionally, 112 **unanswerable questions** test the system's ability to recognize when temporal information is insufficient or the question cannot be grounded in the dialogue. These are evaluated via precision, recall, and F1.

### Dataset Construction Pipeline

Four-step process, all using GPT-4o:

1. **Temporal event extraction** -- extract events from each session with relative time annotations, inferring absolute dates from session timestamps and relative expressions
2. **Cross-session event linking** -- identify related events across sessions by entity overlap
3. **Multiple-choice QA generation** -- generate 3-5 answer options per question, including unanswerable variants
4. **Manual quality control** -- human reviewers verify alignment with design specs, correct answer grounding, and remove unreasonable questions

No inter-annotator agreement metrics are reported, which is a notable methodological gap.

### Evaluation Metrics

- **Accuracy** for overall and per-type performance (answerable questions)
- **Precision, Recall, F1** for unanswerable question detection
- Code execution failure rates tracked separately

### Key Design Distinction: Event Occurrence vs. Event Mention

The benchmark deliberately distinguishes when an event *occurred* from when it was *mentioned*. A user might say "I got promoted last week" in session 20, but the promotion happened the previous week relative to session 20's timestamp. Systems that conflate mention-time with event-time fail systematically. This is the core challenge TReMu's time-aware memorization addresses.

---

## 3. Key Claims and Evidence

### Claim 1: Standard LLMs fail badly at multi-session temporal reasoning.
**Evidence**: GPT-4o achieves only 29.83% with standard prompting. Even CoT only reaches 61.67%. The 30% baseline suggests near-random performance on a multiple-choice task (likely 3-5 options). This is a strong result -- temporal reasoning on dialogues is genuinely hard, not a solved problem.

**Evidence quality**: Strong. The gap between standard prompting and the best method (77.67%) is large enough to be meaningful. Multiple baselines tested.

### Claim 2: Time-aware memorization outperforms holistic session summaries.
**Evidence**: Timeline + CoT (71.50%) substantially outperforms MemoChat + CoT and standard CoT (61.67%). The key mechanism: generating fine-grained memory pieces linked to *inferred date markers* rather than session-level summaries. Grouping simultaneous events by timestep and recording absolute rather than relative dates.

**Evidence quality**: Medium-strong. The ablation shows clear additive value. However, the paper does not evaluate intermediate approaches (e.g., session summaries with timestamps vs. event-level temporal annotations), so we cannot isolate which component of "time-aware memorization" drives the gains.

### Claim 3: Neuro-symbolic reasoning (code generation) outperforms natural language reasoning for temporal calculations.
**Evidence**: TReMu (77.67%) outperforms Timeline + CoT (71.50%). The 6.17-point improvement comes from generating Python code with `datetime` and `dateutil.relativedelta` for temporal calculations rather than reasoning in natural language.

**Evidence quality**: Strong and intuitive. Duration calculations and date arithmetic are exactly the kind of task where symbolic computation should dominate neural approximation. Code execution failure rates are low (2-5% for GPT-4o), so the approach is reliable.

### Claim 4: Temporal Interval is the hardest sub-task.
**Evidence**: Across all methods, TI scores are consistently lowest. Even TReMu only reaches 68.38% on TI vs. 84.47% on TA and 81.37% on TP. Standard prompting on TI: 30.34%.

**Evidence quality**: Strong. Duration computation requires correctly grounding *both* events and performing arithmetic, compounding error sources.

### Claim 5: Unanswerable question detection remains challenging.
**Evidence**: Best F1 is 64.42% (TReMu on GPT-4o). Standard prompting achieves only 20.84% F1.

**Evidence quality**: Medium. The 112 unanswerable questions are a relatively small test set. Still, the pattern -- systems consistently over-answer rather than abstaining -- aligns with broader findings in LongMemEval and elsewhere.

---

## 4. Standout Feature

What distinguishes TReMu from the other three benchmarks we have analyzed:

- **LongMemEval** tests temporal reasoning as one of five abilities (27% of questions). TReMu goes *deep* on temporal reasoning exclusively, decomposing it into three sub-tasks and demonstrating that different temporal operations have different difficulty profiles.
- **MemoryAgentBench** has no temporal reasoning dimension at all. Its "Selective Forgetting" competency deals with knowledge updates but not temporal grounding.
- **LoCoMo** includes temporal questions as one of five reasoning types, but treats temporal reasoning as event ordering without distinguishing anchoring, precedence, and interval computation.

The standout insight is the **event-occurrence vs. event-mention distinction**. No other benchmark we have analyzed explicitly tests whether a system can resolve "last Tuesday" relative to a session's timestamp to an absolute date. This is the exact gap in our system: we store `created_at` timestamps on memories but do not distinguish between when a memory was *stored* and when the event it describes *occurred*.

The neuro-symbolic approach (code generation for temporal calculation) is also novel in the memory benchmark space. Rather than treating temporal reasoning as a pure language task, TReMu demonstrates that offloading date arithmetic to symbolic computation yields large gains. This suggests that temporal reasoning in memory systems may benefit more from structured tooling than from better prompting.

---

## 5. Competency Coverage Ratings

| Dimension | Rating | Justification |
|-----------|--------|---------------|
| **Information Retrieval** | 25% | Questions require retrieving specific events from long dialogues, but retrieval is not the evaluated dimension -- it is assumed (memory is provided to the model). The benchmark tests *reasoning over* retrieved information, not retrieval itself. |
| **Multi-Session Reasoning** | 70% | Temporal Precedence and Temporal Interval explicitly require integrating information from multiple sessions. Cross-session event linking is central to the construction pipeline. However, the reasoning is specifically temporal, not general multi-session synthesis. |
| **Knowledge Update/Contradiction** | 15% | Unanswerable questions test some boundary cases, but the benchmark does not test knowledge updates, contradictions, or supersession. If a user says "I moved to Austin" in session 5 and "I moved to Denver" in session 20, TReMu does not test which location is current. |
| **Temporal Reasoning** | 95% | This is the benchmark's entire focus. Three distinct temporal sub-tasks, event-occurrence vs. mention-time distinction, absolute date grounding from relative expressions. The only gap: it does not test recency queries ("what happened most recently?") or temporal range filters ("what happened in January?"), which are more operational than computational. |
| **Abstention/Confidence** | 30% | 112 unanswerable questions with P/R/F1 evaluation. This is meaningful but only covers temporal un-answerability (insufficient temporal information), not broader abstention (e.g., "I never mentioned that"). |
| **Write-Path Behavior** | 20% | The time-aware memorization component of TReMu is a write-path contribution: how to structure temporal information at storage time. But the benchmark itself does not evaluate write-path quality -- it provides memory and tests reasoning. |
| **Consolidation Quality** | 0% | No consolidation, merging, or summary evaluation. Memories are constructed once and used as-is. |
| **Proactive/Contextual Recall** | 0% | All questions are explicitly asked. No proactive memory surfacing is tested. |
| **Relationship/Graph Reasoning** | 10% | Cross-session event linking identifies entity relationships implicitly, but no graph structure or relationship reasoning is tested. |
| **Agentic Task Performance** | 5% | Purely QA-based evaluation. No tool use, planning, or autonomous action. The code generation is agentic in a narrow sense (LLM writes and executes code), but within a fixed evaluation pipeline. |

---

## 6. Relevance to claude-memory

### Could we run against this benchmark?

**Partially, with adaptation.** The benchmark is built on LoCoMo dialogues, which are publicly available. The 600 temporal QA pairs may or may not be released (the paper does not mention a public release, but ACL Findings papers typically provide code/data). Even without the exact questions, we could reconstruct a similar evaluation:

1. Take LoCoMo dialogues
2. Store them into claude-memory using `remember()` with temporal extraction
3. Query using `recall()` for temporal questions
4. Evaluate accuracy

The main adaptation needed: our system would need to perform temporal *extraction* at write time (priority #11) and temporal *reasoning* at query time. Currently, `recall()` returns memories ranked by relevance but does not perform date arithmetic or temporal ordering. A user asking "how long between X and Y?" would get relevant memories, but the temporal computation would fall to the calling LLM's in-context reasoning -- which TReMu shows achieves only ~62% even with CoT.

### What would it reveal?

Running this benchmark would expose two gaps:

1. **Temporal metadata quality.** Our memories have `created_at` timestamps, but these reflect when the memory was *stored*, not when the described event *occurred*. For accurate temporal anchoring, we need an `event_time` field populated at write time. This is exactly priority #11 (temporal extraction on `remember()`).

2. **Temporal reasoning capability.** Even with perfect temporal metadata, `recall()` has no mechanism to answer "how many weeks between X and Y?" or "which happened first?" These require either (a) a temporal reasoning layer between recall and response, or (b) structured temporal metadata that the calling LLM can compute over. TReMu's code-generation approach suggests option (b) could be implemented as a recall post-processor that generates Python code over temporal fields.

### Adaptation needed

| Component | Current State | Needed for TReMu | Effort |
|-----------|--------------|-------------------|--------|
| Event timestamp | `created_at` only (storage time) | `event_time` field (when event occurred) | Medium -- priority #11 |
| Temporal extraction | None | Parse relative time expressions at `remember()` time | Medium-High |
| Temporal query detection | None | Detect "when", "how long", "before/after" patterns | Low -- regex/LLM classification |
| Temporal computation | Delegated to calling LLM | Code generation or structured datetime operations | Medium |
| Timeline view | None | Chronological memory display for temporal context | Low-Medium |

---

## 7. Insights Worth Stealing

Ranked by effort/impact ratio:

### 1. Event-time vs. storage-time distinction (effort: low-medium, impact: high)

Add an optional `event_time` field to the memory schema. When `remember()` receives content describing a past event ("last Tuesday I started the new job"), extract the event's actual date. This is the single most impactful architectural change TReMu suggests. Without it, temporal queries against our memories will always confuse when something was stored with when it happened.

Implementation: at `remember()` time, if the content contains temporal expressions, use the LLM (already in the write path) to extract an absolute date. Store in a new `event_time` column alongside `created_at`. Default to `created_at` when no distinct event time is detected.

### 2. Temporal query routing (effort: low, impact: medium)

TReMu's three question types map to different computational requirements. We could detect temporal query intent at `recall()` time (already partially implemented with `_detect_intent()` for edge type routing) and route to specialized handling:

- **Anchoring** ("when did X happen?"): filter by event_time, return timestamped results
- **Precedence** ("which happened first?"): retrieve both events, sort by event_time
- **Interval** ("how long between X and Y?"): retrieve both events, compute `event_time` delta

This does not require neuro-symbolic code generation. Simple datetime arithmetic over structured fields would handle most cases.

### 3. Timeline summarization at write time (effort: medium, impact: medium)

TReMu's time-aware memorization groups simultaneous events by timestep. We could add a lightweight version: when storing a batch of memories from the same session, generate a timeline summary that orders events chronologically with absolute dates. This would serve as a supplementary retrieval key for temporal queries.

### 4. Code generation for temporal computation (effort: medium-high, impact: medium)

TReMu's strongest finding: generating Python code with `datetime`/`dateutil` for temporal calculations outperforms natural language CoT by 6+ points. We could implement this as a post-recall processing step: when a temporal query is detected, instead of returning raw memory text, generate a Python snippet that computes the answer from structured temporal fields.

This is the highest-effort suggestion and may be overkill for our current scale. The calling LLM (Opus) can likely handle most temporal arithmetic in-context if given properly structured timestamps. The code generation approach becomes more valuable at scale or with weaker calling models.

### 5. Unanswerable temporal question detection (effort: low, impact: low-medium)

TReMu's 112 unanswerable questions test whether systems can recognize when temporal information is insufficient. We could adopt a similar principle: when a temporal query retrieves memories but the `event_time` fields are null or ambiguous, signal low confidence rather than hallucinating a date.

---

## 8. What's Not Worth It

- **Multiple-choice evaluation format.** The benchmark uses MC questions exclusively. Our system operates in a generative context where the LLM produces free-text answers. Adapting to MC format would require constructing distractor options, which adds overhead without testing realistic usage patterns. If we build a temporal evaluation, open-ended generation with LLM-as-judge (following LongMemEval's approach) is more relevant.

- **Full LoCoMo dialogue ingestion.** TReMu builds on LoCoMo's 600-turn, 16K-token dialogues. Our memory system stores curated facts and events, not raw conversation transcripts. Ingesting raw LoCoMo dialogues would test a workflow we do not use. A more relevant evaluation would test temporal reasoning over our actual memory schema.

- **Neuro-symbolic code generation as a core feature.** While the code generation results are impressive, building a full code-generation-and-execution pipeline into `recall()` is overengineered for our use case. The calling LLM already has tool access and can generate code if needed. What we need is better *input* to the LLM's temporal reasoning (structured timestamps), not a built-in symbolic engine.

- **Cross-session event linking during construction.** TReMu's pipeline links events across sessions by shared entities during benchmark construction. For our system, this is already handled by the edge graph -- `memory_edges` connects related memories. We do not need a separate entity-linking pipeline.

---

## 9. Key Takeaway

TReMu demonstrates that temporal reasoning in multi-session dialogue is a distinct, decomposable capability where current LLMs fail badly (30% baseline) and where structured approaches yield large gains (78% with time-aware memory + code generation). The critical architectural insight for claude-memory is the **event-time vs. storage-time distinction**: our `created_at` field records when a memory was stored, but temporal queries require knowing when the described event *occurred*. This single schema addition -- an `event_time` column populated by temporal extraction at `remember()` time -- would address the deepest gap TReMu exposes. The neuro-symbolic code generation is impressive but secondary; with proper temporal metadata, the calling LLM's in-context arithmetic is likely sufficient for our scale. The benchmark reinforces that priority #11 (temporal extraction) is a prerequisite for priority #3 (temporal invalidation) -- you cannot invalidate stale temporal information if you do not know when events happened.

---

## 10. Impact on Implementation Priority

### Direct impacts:

**Priority #3 (Temporal invalidation)**: TReMu provides empirical evidence that temporal reasoning is genuinely hard (30% baseline) and requires structured metadata (not just better prompts). This strengthens the case for temporal invalidation but also clarifies the prerequisite: you need accurate event timestamps before you can invalidate based on them. Priority #3 should be re-scoped as "temporal invalidation *given* event_time metadata" and sequenced *after* #11.

**Priority #11 (Temporal extraction on remember())**: Elevated from "nice to have" to "architecturally necessary." TReMu's core finding is that the event-occurrence vs. event-mention distinction is what makes temporal reasoning tractable. Without `event_time` extraction at write time, we have no foundation for temporal queries, temporal invalidation, or temporal ordering. This should move up in priority.

### Indirect impacts:

- **Temporal query detection** (part of `_detect_intent()`): Already partially implemented. TReMu's three-type taxonomy (anchoring, precedence, interval) provides a more structured classification scheme we could adopt.
- **Edge graph temporal annotations**: If `memory_edges` encode temporal relationships (X happened before Y, X caused Y), the edge graph becomes a temporal index. TReMu does not explore this, but the connection to our architecture is clear.
- **Recall output formatting**: When temporal queries are detected, `recall()` could format results chronologically with explicit timestamps rather than by relevance score. Low effort, potentially high impact for the calling LLM's ability to reason temporally.

### Recommended priority adjustment:

Move #11 (temporal extraction) to immediately before #3 (temporal invalidation). Implement as:
1. Add `event_time` column to memories table (nullable, defaults to `created_at`)
2. At `remember()` time, when content contains temporal expressions, extract absolute date via LLM
3. In `recall()`, when temporal query is detected, include `event_time` in formatted output
4. Only then implement temporal invalidation (#3), which can now operate on accurate event timestamps

---

## See Also

- [[agent-output-longmemeval]] -- LongMemEval benchmark analysis (temporal reasoning = 27% of questions, tests five abilities)
- [[agent-output-memoryagentbench]] -- MemoryAgentBench analysis (no temporal reasoning dimension, but Selective Forgetting overlaps with temporal invalidation concerns)
- [[agent-output-locomo]] -- LoCoMo benchmark analysis (temporal reasoning as one of five question types, TReMu builds directly on LoCoMo dialogues)
- [[retrospective-experiments]] -- Phase 14 experiments, including vector search assessment relevant to temporal query routing
