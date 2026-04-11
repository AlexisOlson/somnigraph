# WRIT: Write Integrity Test -- Analysis

*Phase 28, 2026-04-11. Analysis of https://github.com/markmhendrickson/writ (Mark Hendrickson).*

---

## 1. Overview

**What it is**: WRIT (Write Integrity Test) is a benchmark for evaluating whether AI memory systems maintain correct, usable, and evolving state over time. It is not a memory system itself -- it tests memory systems. The core thesis: existing benchmarks (LoCoMo, LongMemEval, BEAM, AMB) test whether you can *find* a stored fact, but none test whether the stored fact is *still correct* after agents write to it.

**Repo**: https://github.com/markmhendrickson/writ. MIT license. TypeScript (Node/tsx). ~1,500 lines of harness code + 50 hand-authored JSON scenario files across 10 categories. 4 commits, all from 2026-04-09 (initial scaffold, rename from WORKMEM, comparison section, full harness). Single contributor.

**Companion blog post**: "No AI memory benchmark tests what actually breaks" (markmhendrickson.com). The benchmark is motivated by the observation that retrieval metrics are necessary but not sufficient -- the hard problem in production is keeping stored facts correct across agent writes, corrections, and time.

**Stack**: TypeScript + Vitest. No database, no embeddings, no LLM calls in the core harness. The harness is a pure evaluation framework; memory systems plug in via adapters. An optional LLM judge (gpt-4o-mini via OpenAI API, configurable) handles rubric-based scoring for complex scenarios. Falls back to substring matching when no API key is available.

---

## 2. Architecture

### What WRIT actually evaluates

WRIT does not store or retrieve memories. It defines **scenarios** (multi-session conversation timelines with ground-truth annotations) and evaluates memory systems against them via an **adapter interface**. The evaluation loop:

1. Feed conversation sessions (3-8 per scenario) into the adapter's `processSession()`
2. Send a probe question via `probe()`
3. Inspect internal state via `getHistory()`, `getStateAsOf()`, `getProvenance()`
4. Score the response against ground truth across 9 metrics
5. Attribute failures to one of three layers: state, retrieval, or agent policy

### The adapter interface

```typescript
interface MemoryAdapter {
  processSession(session: Session): Promise<void>;
  probe(prompt: string, options?: ProbeOptions): Promise<ProbeResult>;
  getHistory(factId: string): Promise<FactHistory | null>;
  getStateAsOf(factId: string, timestamp: string): Promise<unknown | null>;
  getProvenance(factId: string): Promise<Provenance | null>;
  getCapabilities(): AdapterCapabilities;
  reset(): Promise<void>;
}
```

The adapter declares its capabilities (history, temporal replay, provenance, abstention). Metrics that require unsupported capabilities score as `null` rather than penalizing the system. This is a fair design choice.

### Three evaluation modes

| Mode | Description | Purpose |
|------|-------------|---------|
| No Memory | Probe only, no prior context | Baseline: what the model invents |
| Native Memory | System uses its own memory | Production behavior |
| Oracle Memory | System receives perfect ground-truth state | Ceiling: isolates model from memory failures |

Comparing modes isolates whether failures are memory-level or model-level. If Native < Oracle, the memory system failed. If Oracle < perfect, the model failed even with correct memory. If Native < No Memory, memory actively harms performance. This decomposition is the benchmark's strongest design choice.

### Built-in adapters

**Baseline**: Naive mutable key-value store. Regex-based fact extraction ("my X is Y"). Overwrites on update. No history, no provenance, no temporal replay. Designed to score well on retrieval and poorly on everything else.

**Neotoma**: HTTP adapter for Neotoma's observation-based memory (the author's own product -- Mark Hendrickson is behind Neotoma). Immutable append-only observations, entity state derived from full observation history, temporal replay via observation timestamps, provenance chain from observation to source to session. Supports history, temporal replay, and provenance. Entity extraction is regex-based in the adapter, not in Neotoma itself.

---

## 3. Scenario Dataset

50 scenarios across 10 categories, 5 per category:

| Category | Tests | Example |
|----------|-------|---------|
| **drift** (5) | Value changes silently over time | Employer changes across 4 sessions |
| **temporal** (5) | Reconstructing state at a past date | "What was my address in March?" |
| **provenance** (5) | Tracing facts to their source | Conflicting user-stated vs. agent-computed values |
| **constraint** (5) | Applying implicit preferences | Dairy allergy inferred from behavior |
| **entity** (5) | Merging, renaming, relationship tracking | Person merge, company rename |
| **forgetting** (5) | Selective non-persistence | Ephemeral instructions vs. durable facts |
| **update** (5) | Conflicting writes, rapid corrections | Revert to previous value |
| **multi_hop** (5) | Combining multiple stored facts | Manager's city derived from two facts |
| **abstention** (5) | Declining when memory is insufficient | Fact never mentioned, retracted fact |
| **work_state** (5) | Ongoing tasks, workflows, checklists | Multi-step workflow with step modifications |

### Scenario quality

The scenarios are well-crafted. Each has 3-8 sessions with realistic timestamps spanning weeks to months. Conversations read naturally, not like test fixtures. The interference annotations (near-duplicates, contradictions, low-salience facts) are present but not yet systematically exploited by the evaluator.

Ground truth is rich: `current_value`, `value_history` (timestamped), `provenance` (source session, message index, agent/user), `eval_rubric` (exact/structured/llm_judge), `constraint_check` (must_contain/must_not_contain). The `memory_events` array tracks each fact's lifecycle: when introduced, updated, retracted, and whether it should persist.

### Memory event types

The taxonomy is practical:
- **explicit**: Clearly stated facts ("My email is...")
- **mutable**: Facts that change over time (employer, address)
- **latent**: Implicit preferences (dairy avoidance inferred from behavior)
- **entity**: People, places, linked objects
- **work_state**: Tasks, workflows, checklists
- **non_memory**: Information that should NOT persist (ephemeral instructions)

The `non_memory` type is novel in the benchmark space. No other benchmark we've analyzed explicitly tests whether a system correctly *forgets* information that was marked or implied as temporary.

---

## 4. Metrics

### Core metrics (9 dimensions)

| Metric | Definition | Novel? |
|--------|-----------|--------|
| Recall Accuracy | Can the system find the stored fact? | No -- standard |
| Update Fidelity | Does it reflect the current value, not a stale one? | **Yes** -- no other benchmark tests this |
| Drift Rate | Fraction of values that changed without explicit correction | **Yes** |
| Detectability | Can the system show when, what, and the previous value? | **Yes** |
| Temporal Accuracy | Correctness of as-of-date state reconstruction | Partial -- LongMemEval has temporal reasoning |
| Provenance Completeness | Can the system trace a fact to its source? | **Yes** |
| Constraint Consistency | Are inferred constraints correctly applied? | No -- BEAM tests this |
| Hallucination Rate | Fraction of responses with no grounding in any known value | No -- standard |
| Abstention Quality | Precision/recall of declining when memory is insufficient | No -- BEAM/LongMemEval test this |

Five of the nine metrics are genuinely novel in the benchmark landscape: update fidelity, drift rate, detectability, temporal accuracy (at the state layer, not query understanding), and provenance completeness. These test the *state layer* that all other benchmarks assume is perfect.

### Failure attribution

Each failing scenario is attributed to one of three layers:
- **State**: Data lost or corrupted (drift, provenance loss)
- **Retrieval**: Data exists but wasn't found
- **Agent Policy**: Data found but misapplied (constraint violations)

This decomposition is more useful than a single accuracy number. A system with 70% recall and 20% provenance completeness has a very different failure profile than one with 70% recall and 90% provenance completeness. The multi-dimensional scorecard is the right idea.

---

## 5. How It Addresses Common Memory System Gaps

Since WRIT is a benchmark, not a memory system, we rate whether it *tests for* each gap:

| Gap | Rating | Notes |
|-----|--------|-------|
| Layered memory | **Not tested** | No scenarios distinguish raw facts from summaries from high-level understanding |
| Multi-angle retrieval | **Indirectly tested** | Recall accuracy measures retrieval effectiveness but doesn't diagnose the mechanism |
| Contradiction detection | **Directly tested** | provenance-003-conflicting-sources tests agent-error propagation; update scenarios test conflicting writes |
| Relationship edges | **Partially tested** | Entity scenarios test merge/rename/relationships, but don't test graph traversal |
| Sleep/consolidation | **Not tested** | Static scenario ingestion; no mechanism for testing post-ingestion consolidation |
| Reference index | **Not tested** | No overview or summary scenarios |
| Temporal trajectories | **Directly tested** | 5 dedicated temporal scenarios + drift scenarios that require temporal awareness |

---

## 6. Comparison with Somnigraph

### What WRIT tests that we should care about

**Update fidelity**: Somnigraph stores memories with `previous_values` support via its contradiction detection system, but we've never measured whether the current value reliably wins over stale values in retrieval. A system that scores high on LoCoMo R@10 could still present stale values when a fact has been updated. WRIT's drift scenarios would test this.

**Provenance**: Somnigraph tracks `source_session` metadata in the sleep pipeline, but it's not a first-class queryable property. WRIT's provenance scenarios would expose whether "where did this fact come from?" questions are answerable.

**Selective forgetting**: Somnigraph has `non_memory` support only via the `forget()` tool -- there's no automatic non-persistence. WRIT's forgetting scenarios test whether ephemeral instructions are correctly dropped.

**Work state tracking**: Somnigraph stores work state as regular memories. WRIT's work_state scenarios (multi-step workflows with incremental modifications) would test whether ordered, mutable structured state survives across sessions.

### Where Somnigraph is stronger (as a system)

- **Retrieval**: Hybrid FTS5 + vec + graph + RRF + learned reranker. WRIT doesn't test retrieval quality in isolation -- it's end-to-end.
- **Consolidation**: Sleep-based memory evolution. WRIT has no mechanism for testing post-ingestion processing.
- **Feedback loop**: Retrieval quality improves over time via feedback. WRIT is a snapshot evaluation.
- **Graph**: Entity relationships, synthetic bridge nodes, coref edges. WRIT's entity scenarios are simple compared to the graph capabilities we've built.
- **Scale**: Somnigraph handles thousands of memories. WRIT scenarios have 3-8 sessions with a handful of facts each.

### Where WRIT exposes gaps we haven't tested

- **Mutable fact tracking**: Do we correctly surface the *latest* value when a fact changes across sessions? Our LoCoMo evaluation doesn't stress this because LoCoMo's corpus is static.
- **Agent-error detection**: provenance-003 (agent miscalculates, propagates error) tests whether the system can distinguish user-stated from agent-inferred facts. Somnigraph doesn't track this distinction.
- **Non-persistence**: We have no `non_memory` category. Everything remembered persists until explicitly forgotten.

---

## 7. Insights Worth Stealing

### 1. Three-mode evaluation (High impact, Low effort)

The no_memory / native_memory / oracle_memory decomposition is the cleanest idea in WRIT. Adding an "oracle" mode to our LoCoMo pipeline would isolate reader failures from retrieval failures definitively. We partially do this by reporting R@10 alongside QA accuracy, but oracle mode would make the decomposition explicit.

**Effort**: 1 session. Feed gold evidence turns directly to the reader, bypass retrieval.

### 2. Update fidelity metric (Medium impact, Low effort)

For memories that have been updated (contradiction detection triggered, `forget` + `remember` cycle), measure whether the current value beats the stale value in retrieval ranking. This is a diagnostic we've never run on production data.

**Effort**: Script against production DB. Find memories with contradictions, query for each, check if current value ranks higher.

### 3. Failure attribution to state/retrieval/policy layers (Medium impact, Low effort)

Our LoCoMo pipeline reports retrieval metrics and QA accuracy separately, but doesn't systematically categorize failures. Adding a "state layer" attribution (data corrupted or lost) vs. "retrieval layer" (data exists but not found) vs. "reader layer" (data found but misinterpreted) would improve our error analysis.

**Effort**: Annotate existing QA errors with this taxonomy. Mostly analysis, minimal code.

### 4. Non-memory type (Low impact, Medium effort)

Explicitly supporting a `non_memory` or `ephemeral` memory type that auto-expires or is never stored. This addresses the over-retention failure mode that Somnigraph doesn't currently test for.

**Effort**: Add a `should_persist: false` field or `ephemeral` category. Would require changes to `remember()` and possibly sleep consolidation.

### 5. Provenance-003 pattern: agent-error propagation (Low-Medium impact, Low effort)

The specific pattern where an agent miscalculates a value and then propagates it is a realistic production failure mode. Worth creating a test case: store a memory from user input, then store a derived-but-wrong value from assistant output. Does the system distinguish source authority?

**Effort**: Scenario design + manual testing. No code changes needed.

---

## 8. What's Not Worth It

1. **Running WRIT against Somnigraph**: The 50 scenarios test 50 specific facts across short timelines. The evaluation is too narrow and too qualitative to produce meaningful aggregate scores. Writing a Somnigraph adapter would take a session, and the diagnostic value would be low compared to LoCoMo (1,540 queries) or PERMA (2,166 preferences).

2. **The regex-based fact extraction in the adapters**: Both the baseline and Neotoma adapters use regex to extract facts from conversation text ("my X is Y", "I work at X"). This is a benchmark shortcut, not an architecture choice. Not relevant to Somnigraph's LLM-based extraction.

3. **The interference annotations**: The scenarios include `interference` arrays (near-duplicates, contradictions, distractors), but the evaluator doesn't use them. They're metadata for future evaluation extensions that don't exist yet.

4. **The Neotoma adapter as an architecture reference**: The adapter is a thin HTTP wrapper around Neotoma's API. It reveals Neotoma's API shape (observation-based, immutable, entity-centric) but not its internal architecture. Neotoma itself isn't open-source.

5. **The LLM judge implementation**: Configurable OpenAI-compatible judge with fallback to substring matching. Standard pattern, nothing novel. Our LoCoMo judge pipeline is more sophisticated (Opus + GPT-4.1-mini calibration, batch processing, vague-answer detection).

---

## 9. Critical Assessment

### Novelty

**High for the thesis, medium for the implementation.** The core insight -- that write integrity is a distinct failure mode from retrieval failure, and that no benchmark tests it -- is genuinely original and correctly argued. The README's benchmark comparison table is honest and well-researched. The gap WRIT identifies is real.

The implementation is competent but early. 50 scenarios is a minimum viable dataset. The evaluator handles the basic cases (exact match, structured match, LLM judge) but doesn't yet exploit all the scenario annotations (interference isn't scored, `non_memory` isn't checked for over-retention in the evaluator code). The `eval_rubric.method` value `"multi_criterion"` appears in provenance-003's ground truth but isn't implemented in the evaluator (it would fall through to exact match).

### Code quality

**Good.** Clean TypeScript, strict mode, well-typed interfaces, comprehensive Vitest test suite (unit tests for evaluator, loader, judge, baseline adapter; integration test for full run; scenario validation test that checks all 50 scenarios). The test coverage is thorough for the scope of the project.

### Maturity

**Very early.** 4 commits, all from the same day (2026-04-09). No CI configuration despite the README mentioning it. No npm publish. No external adopters visible. The Neotoma adapter requires a running Neotoma instance (not publicly available). The benchmark is essentially a proof-of-concept for the blog post.

### Community

**None yet.** Single contributor, 2 days old at time of analysis. No stars, no issues, no PRs. The blog post may drive initial interest, but the benchmark needs more scenarios, more adapters, and external validation to become a standard.

### Conflict of interest

**Present but disclosed.** The Neotoma adapter is a first-class citizen (supports history, temporal replay, provenance) while the baseline intentionally doesn't. This is the same pattern as AMB (builder's product tops the leaderboard). However, WRIT is more honest about it: the baseline is explicitly described as "what most memory systems do" (mutable KV store), and the benchmark is designed to expose why that's insufficient. The benchmark's value proposition is that *most systems would score poorly*, not that Neotoma scores well.

---

## 10. Summary Verdict

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Novel contribution | **Medium-High** | The thesis (write integrity as distinct from retrieval) is genuinely original. Implementation is early. |
| Engineering quality | **Medium** | Clean code, good tests, but evaluator has unimplemented features and the dataset is small. |
| Evaluation fairness | **Medium** | Neotoma is advantaged by design, but the asymmetry is disclosed and structurally motivated. |
| Scenario quality | **High** | Well-crafted, realistic conversations. Good coverage of write-integrity failure modes. |
| Maturity | **Very Low** | 4 commits, 1 day of development, no external users. |
| Relevance to Somnigraph | **Medium** | The thesis matters. The three-mode evaluation and failure attribution are worth stealing. Running the benchmark itself is not worth the effort at 50 scenarios. |
| Research value | **Medium** | Identifies a real gap. Needs 10x more scenarios and external validation to become a citation-worthy benchmark. |

WRIT is a well-argued proof-of-concept for a genuinely underexplored evaluation dimension. The write-integrity thesis is correct and original -- no widely used benchmark tests whether stored facts survive agent writes. The three-mode evaluation (no memory / native / oracle) and three-layer failure attribution (state / retrieval / policy) are clean abstractions worth adopting. The scenario dataset is small but high-quality. The benchmark is too early and too narrow to run against Somnigraph, but the ideas behind it should inform how we think about evaluation beyond retrieval accuracy.
