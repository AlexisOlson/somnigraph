# LLMs as Method Actors: A Model for Prompt Engineering and Architecture -- Analysis

*Generated 2026-03-22 by Opus 4.6 agent reading arXiv:2411.05778v2*

---

## Paper Overview

**Paper**: Colin Doyle (Loyola Law School, Loyola Marymount University). "LLMs as Method Actors: A Model for Prompt Engineering and Architecture." arXiv:2411.05778v2, Nov 2024. 41 pages (13 main + 28 appendix of full prompts). Code: https://github.com/colindoyle0000/llms-as-method-actors (public).

**Problem addressed**: Prompt engineering techniques like Chain-of-Thought treat LLM responses as thoughts and try to mimic human reasoning steps. This mental model is imprecise -- LLMs don't think, they perform. The paper argues that treating LLMs as actors performing from scripts, rather than thinkers reasoning through problems, produces better prompt engineering and architectural decisions. Tested on NYT Connections puzzles, where GPT-4o with vanilla prompting solves 27% and CoT solves 41%.

**Core claim**: The "Method Actor" mental model yields four actionable principles: (1) prompts are scripts providing character, motivation, and stage direction; (2) performance requires preparation through intermediate outputs; (3) tasks should be decomposed until imitation equals authenticity; (4) where imitation fails, compensate with non-LLM methods (deterministic validation). The principles collectively produce an 86% solve rate (GPT-4o) and 87% perfect-solve rate (o1-preview) on Connections puzzles.

**Scale**: 100 NYT Connections puzzles (#331-430). 7 approaches tested: Vanilla, CoT, CoT-Scripted, Actor, Actor-2 (all GPT-4o); Oneshot-o1, Vanilla-o1, Actor-o1 (o1-preview). No other benchmarks. Single-run per puzzle (cost constraint acknowledged). Puzzle #410 retested 4 times across all 7 approaches.

---

## Architecture / Method

### The Mental Model

The paper's contribution is primarily conceptual, not algorithmic. The "Method Actor" analogy maps:

| Acting Concept | LLM Equivalent | Implication for Prompt Design |
|---------------|----------------|------------------------------|
| Actor | LLM | Performs from scripts, does not generate original thought |
| Script | System/user prompt | Sets scene, character, motivation, stage direction |
| Performance | LLM response | Judged by verisimilitude, not truth |
| Hallucination | Staying in character | Faithful to script, not to external reality |
| Rehearsal | Intermediate API calls | Preparation produces better final performance |
| Director | Prompt engineer | Shapes the performance through cues, not instructions |

### Four Principles

**Principle 1: Prompt engineering is playwriting and directing.** Prompts should provide character, motivation, setting, and direction -- not just role assignment. The Actor prompts cast the LLM as an FBI-recruited puzzle solver defusing a bomb in a children's hospital. This dramatic framing reportedly increased the LLM's willingness to exhaust its output context exploring solutions rather than giving up early.

**Principle 2: Performance requires preparation.** Complex outputs need intermediate "performances" -- separate API calls that build context incrementally. The Actor approach uses 5 brainstorming calls before discernment, each exploring a different pattern template. This is essentially Tree-of-Thoughts with a dramatic framing and domain-specific decomposition.

**Principle 3: Decompose until imitation = authenticity.** Identify subtasks where producing a convincing imitation of the output is functionally equivalent to actually doing the task. Brainstorming connections between words is such a task -- an LLM that *appears* to brainstorm connections genuinely produces candidate connections. Discernment (judging which brainstormed connection is correct) is *not* such a task -- appearing to judge well is different from judging well.

**Principle 4: Where imitation fails, use non-LLM methods.** Actor-2 compensates for weak discernment with deterministic logic:
- **Unique-pair validation**: Don't submit a guess until two guesses share no overlapping words (reduces red herrings combinatorially)
- **Mole word filtering**: Insert already-solved words into the candidate list; if the LLM includes a mole word in its guess, the guess is rejected as a likely hallucination. Mathematically filters ~2/3 of random guesses when 2 correct groups are already found

### Prompt Architecture (Actor-2 / GPT-4o)

The architecture is a multi-stage pipeline with 5 phases:

1. **Brainstorm** (5 API calls): Each call applies one of 24 pattern templates to the word list. Templates are drawn from real Connections answer patterns (synonyms, homophones, shared prefixes/suffixes, pop culture, etc.). The prompt includes dramatic framing, examples, and step-by-step instructions specific to the pattern.

2. **Extract**: A separate "notes editor" persona extracts viable guesses from brainstorming notes, reducing context length for downstream processing.

3. **Discern**: The LLM evaluates extracted guesses and selects the strongest. Prompt includes a taxonomy of 20+ connection patterns with examples as calibration reference.

4. **Decide**: A fresh LLM call determines whether the top guess is strong enough to submit.

5. **Evaluate**: After 5 guesses accumulate, the LLM selects the strongest for submission. Deterministic logic then gates actual submission (unique-pair and mole-word checks).

### Actor-o1 Adaptation

For o1-preview, the architecture simplifies: brainstorm + discern collapse into a single call (o1's internal chain-of-thought handles the reasoning steps that GPT-4o needed explicit prompting for). The unique-pair/triplet/quadruplet validation becomes stricter -- Actor-o1 waits for four non-overlapping guesses before submitting when possible.

---

## Key Claims & Evidence

### GPT-4o Results

| Approach | Puzzles Solved | Solved Perfectly |
|----------|---------------|-----------------|
| Vanilla | 27% | 12% |
| CoT | 41% | 20% |
| CoT-Scripted | 56% | 24% |
| Actor | 78% | 41% |
| Actor-2 | 86% | 50% |

### o1-preview Results

| Approach | Puzzles Solved | Solved Perfectly |
|----------|---------------|-----------------|
| Oneshot-o1 | 79% | 72% |
| Vanilla-o1 | 100% | 76% |
| Actor-o1 | 99% | 87% |

### Incorrect Guesses (o1-preview, out of 400 possible)

| Approach | Incorrect Guesses |
|----------|------------------|
| Oneshot-o1 | N/A (one-shot) |
| Vanilla-o1 | 35 |
| Actor-o1 | 19 |

### Ablation-like Comparisons

The paper doesn't provide clean ablations (no systematic removal of individual components). The closest evidence for what matters:

| Comparison | Interpretation |
|-----------|---------------|
| CoT (41%) vs CoT-Scripted (56%) | Curated examples + step instructions add +15pp over generic "step-by-step" |
| CoT-Scripted (56%) vs Actor (78%) | Multi-call architecture + dramatic framing + template cycling add +22pp |
| Actor (78%) vs Actor-2 (86%) | Deterministic validation (unique-pair + mole words) adds +8pp |
| Vanilla-o1 (76% perfect) vs Actor-o1 (87% perfect) | Method Actor architecture adds +11pp perfect-solve even on a strong reasoning model |

### Puzzle #410 Deep Dive (4 iterations)

| Approach | Solve Rate | Bad Guesses |
|----------|-----------|-------------|
| Vanilla | 1/4 | 15 |
| CoT | 4/4 | 7 |
| Actor-2 | 4/4 | 7 |
| Oneshot-o1 | 4/4 | 0 |
| Actor-o1 | 2/4 | 13 |

Actor-o1 fails on this high-red-herring puzzle because the unique-pair heuristic settles on incorrect guesses. The one-shot approach, forced to find a complete non-overlapping solution in one pass, naturally resolves red herrings. This reveals a structural weakness in iterative deterministic validation.

### Methodological Strengths

- **Clearly articulated mental model**: The four principles are concrete enough to apply to other domains, not just Connections puzzles
- **Honest about the weak point**: The paper identifies discernment (not brainstorming) as the LLM's limitation and designs around it rather than prompting harder
- **Puzzle #410 investigation**: Retesting a failure case across all approaches with 4 iterations shows genuine curiosity about failure modes
- **Full prompts published**: The 28-page appendix contains every prompt template, making the work fully reproducible
- **Separation of LLM and non-LLM contributions**: Actor-2's deterministic logic is clearly identified as non-LLM, which is honest about what the prompting vs. engineering contributed

### Methodological Weaknesses

- **Single benchmark, single domain**: All results are on Connections puzzles. No evidence the mental model transfers to other tasks. The paper explicitly acknowledges this as future work but never delivers it.
- **No ablations**: Cannot isolate the contribution of dramatic framing vs. template cycling vs. multi-call architecture vs. deterministic validation. The CoT-Scripted -> Actor jump bundles multiple changes.
- **Single run per puzzle**: Each approach was run once per puzzle (acknowledged cost constraint). With LLM stochasticity, results could shift significantly. The Puzzle #410 retest (4 runs) shows solve rates ranging from 1/4 to 4/4 across approaches.
- **No cost analysis**: The Actor approaches use 10-20+ API calls per puzzle vs. 2 for Vanilla. No token counts, latency measurements, or cost-per-solve reported. This makes the 27% -> 86% improvement look free when it is likely 10-20x more expensive.
- **Confounded improvements**: The dramatic FBI framing, the 24 pattern templates, and the multi-stage architecture are all changed simultaneously. An "Actor without drama" or "Drama without templates" condition would disambiguate.
- **Not peer-reviewed**: arXiv preprint from a law school professor. No formal venue. The work is competent but lacks the rigor expected of a machine learning conference paper.

---

## Relevance to claude-memory

### What Method Actors Does That We Don't

1. **Pattern template cycling for coverage**: The brainstorming phase cycles through 24 domain-specific templates, ensuring the LLM explores diverse connection types rather than settling on the first plausible one. Somnigraph's `sleep_rem.py` prompts use a single framing for each consolidation step. Cycling through multiple framings -- e.g., asking the LLM to classify a memory cluster from different perspectives (temporal, causal, thematic, contradictory) -- could improve sleep quality. This connects to the open question in `roadmap.md` about whether sleep improves retrieval: if sleep prompts are under-exploring the space of possible relationships, the answer might be "not yet."

2. **Explicit imitation-authenticity boundary identification**: Principle 3 (decompose until imitation = authenticity) is a useful lens for evaluating where LLMs can be trusted in our pipeline. Somnigraph uses LLM judgment extensively in sleep (NREM contradiction detection, REM clustering, thematic consolidation) without explicitly asking "is imitation equivalent to authenticity for this subtask?" Contradiction detection's catastrophic F1 (0.025-0.037) suggests the answer is "no" -- it's a subtask where appearing to detect contradictions is very different from actually detecting them.

3. **Deterministic validation layers**: Actor-2's unique-pair and mole-word checks are non-LLM compensations for LLM weaknesses. Our pipeline doesn't have analogous deterministic checks on LLM-generated outputs during sleep. When `sleep_rem.py` generates a summary or detects a contradiction, the output is accepted without validation. Adding deterministic sanity checks (e.g., confirming that a "contradiction" pair actually shares entity overlap, or that a summary preserves key terms from its source memories) would align with Principle 4.

### What We Already Do Better

1. **Actual retrieval system vs. "just use context"**: The Reddit thread that surfaced this paper framed it as "prompt engineering as alternative to RAG." But Method Actors doesn't address retrieval at all -- it operates entirely within a single puzzle's context window. Somnigraph's hybrid retrieval (vector + FTS5 + RRF fusion + reranker) solves a fundamentally different problem: finding relevant information from hundreds of memories that don't fit in context. The "just use context" argument breaks at ~50 memories with current context windows; Somnigraph manages 700+ and growing.

2. **Learning from feedback**: Method Actors has no learning mechanism. Each puzzle is solved independently; success on puzzle #331 teaches nothing about puzzle #332. Somnigraph's feedback loop (per-query r=0.70 with GT, reranker trained on 1032 queries) accumulates signal over time. This is the fundamental difference between a prompt engineering framework and a memory system.

3. **Persistent, evolving knowledge**: The 24 pattern templates in Method Actors are static -- hand-curated by the author from prior puzzles. Somnigraph's sleep pipeline generates, updates, and prunes knowledge automatically. The CLAUDE.md snippet evolves through testing (dogfood test, simulation, fresh-session test). Method Actors would need a human to update templates for each new domain.

4. **Principled scoring**: Method Actors' discernment stage relies on LLM judgment to pick the "strongest" guess, which the paper itself identifies as the weak point. Somnigraph replaces exactly this kind of LLM ranking judgment with a learned LightGBM reranker (+6.17pp NDCG over formula), using features the LLM can't access (co-retrieval PMI, PPR scores, feedback history).

---

## Worth Stealing

### 1. Imitation-Authenticity Audit for Sleep Prompts (Medium Value, Low Effort)

**What**: Systematically evaluate each LLM-mediated step in `sleep_rem.py` through the lens of Principle 3: is imitation equivalent to authenticity for this subtask? Where it isn't, add deterministic validation (Principle 4).

**Why it matters**: Sleep is Somnigraph's most LLM-dependent component, and its impact on retrieval quality is unmeasured (the central open question in `roadmap.md`). If sleep prompts are asking the LLM to do things where "appearing to do it well" diverges from "doing it well" -- and contradiction detection's 0.025-0.037 F1 confirms this for at least one subtask -- then sleep quality may be lower than assumed.

**Implementation**: For each REM step, classify the subtask:
- **Imitation ~ authenticity**: Summarization (generating a plausible summary is a plausible summary), theme extraction (generating plausible themes produces usable themes), edge labeling (generating a plausible relationship description is a usable relationship description)
- **Imitation != authenticity**: Contradiction detection (appearing to find contradictions != finding contradictions), temporal ordering (appearing to order events != correctly ordering events), importance ranking (appearing to judge importance != judging importance correctly)

For the second category, add deterministic checks: entity overlap verification for contradiction candidates, timestamp comparison for temporal claims, term-preservation checks for summaries.

**Effort**: Low. This is a prompt and validation audit, not new infrastructure. Could produce a prioritized list of sleep steps to harden.

### 2. Template Cycling for Sleep Consolidation (Low-Medium Value, Low Effort)

**What**: When `sleep_rem.py` processes a memory cluster, run multiple classification passes with different framing prompts rather than a single prompt. Aggregate results by voting or union.

**Why it matters**: Method Actors' 24 templates ensure coverage of the connection space. A single sleep prompt may systematically miss certain relationship types (e.g., temporal relationships, causal chains, contradictions through intermediate facts). The paper's CoT-Scripted (56%) vs. Actor (78%) gap suggests that diverse prompting perspectives produce meaningful gains even when the underlying model is the same.

**Implementation**: For REM clustering, instead of one "classify this cluster" prompt, use 3-4 framing variants:
- "What temporal relationships exist between these memories?"
- "What causal or dependency chains connect these memories?"
- "Do any of these memories contradict or supersede each other?"
- "What thematic patterns emerge across these memories?"

Union the detected relationships. Cost: 3-4x the LLM calls per cluster during sleep (offline, so latency is not a constraint).

**Effort**: Low. Prompt variants + aggregation logic. Sleep already runs offline, so the additional API calls are acceptable.

---

## Not Useful For Us

### Single-Session Context Assumption
The entire Method Actors framework assumes all relevant information fits in the context window. The paper doesn't address retrieval, persistence, or accumulation of knowledge across sessions. Somnigraph exists precisely because this assumption doesn't hold for long-lived personal memory. A memory system with 700+ memories spanning months of interaction cannot rely on "just prompt better." This isn't a limitation of the paper's scope -- it's a structural mismatch with our use case.

### Dramatic Framing / Emotional Manipulation
The FBI bomb-defusal framing reportedly improved output quality by making the LLM use more of its context window. This is an uncontrolled observation (no ablation isolating the dramatic framing from other architectural changes). Even if real, it's a fragile prompt hack -- model updates can change sensitivity to emotional stimuli. Somnigraph's prompts (CLAUDE.md snippet, sleep pipeline prompts) should optimize for clarity and precision, not emotional manipulation. The finding that "emotional stimuli improve LLM reasoning" (Li et al., 2023, cited by the paper) is not robustly replicated and may not survive the transition to newer models.

### Domain-Specific Template Library
The 24 Connections puzzle templates are the most labor-intensive component and are entirely domain-specific. They encode knowledge of NYT Connections patterns (homophones, shared prefixes, pop culture references, etc.). This approach doesn't generalize -- you'd need a new template library for each domain. For Somnigraph's recall queries, which span arbitrary personal topics, pre-built templates are infeasible. The Dynamic Cheatsheet approach (self-curating, evolving memory) is the correct pattern for domain-agnostic knowledge accumulation.

### Unique-Pair Combinatorial Validation
The deterministic validation in Actor-2 (wait for two non-overlapping groups before submitting) exploits the specific structure of Connections puzzles (exactly 4 groups of exactly 4 words, no overlap). This constraint doesn't exist in memory retrieval. There's no equivalent of "if two retrieval sets don't overlap, they're more likely correct." The principle behind it (compensate for LLM weakness with deterministic checks) transfers; the specific mechanism doesn't.

### Mole Word Hallucination Detection
Inserting known-correct items as traps to detect hallucination is clever for a game with discrete correct/incorrect states. Memory retrieval has no equivalent: we can't insert "known-irrelevant" memories into recall results to test whether the LLM would use them, because the LLM doesn't select from results -- it receives all results and uses them as context. The hallucination detection problem in our context is about sleep-generated summaries or edge labels being unfaithful to source memories, not about the LLM selecting wrong items from a set.

---

## Impact on Implementation Priority

### CLAUDE.md snippet design -- Unchanged, mildly informative
The snippet IS a prompt engineering artifact, and Method Actors' Principle 1 (playwriting and directing) is relevant to how it's written. But the snippet has already been through three validation passes (dogfood, simulation, fresh-session) and the current design is stable. The paper doesn't offer specific techniques that would improve the snippet beyond what was already learned from the Dynamic Cheatsheet analysis and direct testing.

### Sleep pipeline quality -- Strengthened
The imitation-authenticity lens (Principle 3) provides a concrete framework for the unmeasured sleep impact question (`roadmap.md` § "Does sleep improve retrieval?"). If sleep substeps are asking the LLM to do things where imitation diverges from authenticity, that's a specific, testable hypothesis about why sleep might not help as much as expected. This strengthens the case for the sleep impact measurement experiment (Tier 1 #3) and suggests adding deterministic validation to sleep as a prerequisite or parallel effort.

### Recall query formulation -- Unchanged
The paper's template-cycling approach doesn't apply to recall query formulation. Our recall queries come from the agent (Claude Code), which already has sophisticated internal reasoning about what to search for. Adding template-based query reformulation would be the expansion-wip approach (multi-query, keyword, entity-focused variants), which has already shown neutral results -- evidence exists in candidate pools at rank 100+ but the reranker can't elevate it. Method Actors offers no new insight on this ranking bottleneck.

### Contradiction detection -- Unchanged, validated
The paper's Principle 3 independently validates the conclusion already in `roadmap.md`: contradiction detection is a subtask where imitation != authenticity. The paper doesn't offer solutions, but it provides a clean conceptual framework for *why* LLM-based contradiction detection fails. This aligns with the existing assessment that contradiction handling is "research-grade hard" and that `valid_from`/`valid_until` temporal bounds are the practical workaround.

---

## Connections

### Dynamic Cheatsheet (arXiv:2504.07952)

Both papers address how persistent external structure improves LLM performance, but from opposite directions:

| Dimension | Method Actors | Dynamic Cheatsheet |
|-----------|--------------|-------------------|
| **Memory source** | Human-curated templates (static) | Self-curated by LLM (evolving) |
| **Persistence** | Within a single puzzle solve | Across sequential tasks |
| **Learning** | None -- each puzzle independent | Accumulates strategies over time |
| **Retrieval** | None -- templates cycled sequentially | Cosine similarity (DC-RS) or cumulative (DC-Cu) |
| **Domain specificity** | Entirely domain-specific (24 Connections templates) | Domain-agnostic (works across AIME, Game of 24, GPQA) |
| **Feedback** | Correct/incorrect guess feedback within game | Self-assessed output quality, no ground truth |

Dynamic Cheatsheet is the closer analog to Somnigraph's architecture (persistent, self-curating, retrieval-based). Method Actors is its conceptual ancestor -- showing that structured external guidance helps, but requiring human curation that Dynamic Cheatsheet automates. For Somnigraph's CLAUDE.md snippet, the synthesis is: the snippet is a hand-curated "cheatsheet" (Method Actors style) that should eventually be supplemented by self-curated strategies (Dynamic Cheatsheet style) through the feedback loop and sleep consolidation.

### CLAUDE.md Snippet (docs/claude-md-guide.md)

The snippet is Somnigraph's closest existing artifact to Method Actors' "script." Both serve the same function: structured external text that shapes LLM behavior without parameter changes.

| Dimension | CLAUDE.md Snippet | Method Actors Prompts |
|-----------|-------------------|----------------------|
| **Length** | ~30 lines (Tier 1) + ~100 lines (Tier 2) | ~500 words per prompt, 24 templates |
| **Structure** | Workflow steps + scoring guidance + examples | Character + motivation + step-by-step + examples |
| **Testing** | 3 validation passes (dogfood, simulation, fresh-session) | Single experiment, no validation methodology |
| **Dramatic framing** | None (professional, precise) | FBI bomb defusal scenario |
| **Feedback integration** | Scoring guidance shapes utility ratings | None |

The snippet was designed through empirical testing, not through Method Actors' principles, but independently arrived at similar patterns: clear role definition (memory system user), step-by-step workflow, calibration examples (scoring guidance), and concrete anchors (Fibonacci limits). The paper's Principle 2 ("performance requires preparation") maps to the snippet's `startup_load()` guidance -- loading context before acting.

### Sleep Pipeline Prompts (sleep_rem.py)

The paper's principles apply most directly to the sleep pipeline's LLM-mediated steps. Each REM step is an LLM "performance" that could be improved by:
- **Principle 1**: Better scene-setting (current prompts are functional but not carefully crafted for the LLM's role)
- **Principle 2**: Multi-pass processing (current pipeline does single-pass per step)
- **Principle 3**: Identifying which steps have imitation-authenticity gaps (contradiction detection: yes; summarization: no)
- **Principle 4**: Adding deterministic validation where gaps exist

### Generative Agents (arXiv:2304.03442)

Generative Agents' reflection mechanism (periodic synthesis of observations into higher-level insights) is the consolidation analog to Method Actors' brainstorm-then-discern pipeline. Both decompose complex judgment into generation (brainstorm/observe) then evaluation (discern/reflect). The key difference: Generative Agents' reflection is self-initiated and continuous; Method Actors' discernment is externally triggered and bounded. Somnigraph's sleep pipeline combines both patterns -- externally triggered (cron schedule) but self-directed within each run.

### LoCoMo Audit (locomo-audit.md)

The LoCoMo audit found that the LLM judge accepts 62.81% of intentionally vague wrong answers. This is a concrete example of Method Actors' Principle 3 in action: LLM-as-judge is a subtask where imitation (appearing to judge correctly) diverges from authenticity (actually judging correctly). The audit's finding validates adding deterministic validation to LLM judge outputs -- exactly what Principle 4 recommends. Our LoCoMo pipeline (`scripts/locomo_bench/`) already uses dual judging (strict + lenient) as a partial response to this, but could benefit from more deterministic checks.

---

## Summary Assessment

This paper is a prompt engineering framework wrapped in a theatrical metaphor, applied to a single word puzzle benchmark. The core intellectual contribution -- four principles for when and how to decompose LLM tasks -- is sound and practically useful, even if the experimental validation is limited to one domain. The dramatic FBI framing is eye-catching but uncontrolled. The strongest technical contribution is the clean separation between "subtasks where LLM imitation works" (brainstorming) and "subtasks where it doesn't" (discernment), with deterministic compensation for the latter. The 27% to 86% improvement on GPT-4o is impressive but confounds multiple simultaneous changes.

**For Somnigraph specifically:**

- **Strongest takeaway**: Principle 3 (imitation-authenticity decomposition) provides a concrete framework for auditing sleep pipeline prompts. The question "for which sleep substeps is appearing to do the task equivalent to doing the task?" has immediate practical value and connects to the unmeasured sleep impact question.

- **Second takeaway**: The paper does not support "prompt engineering as alternative to RAG." The Method Actors framework operates entirely within a single context window on a bounded problem. For persistent memory across hundreds of sessions with 700+ memories, retrieval is not optional. The Reddit framing that surfaced this paper was misleading.

- **Third takeaway**: Template cycling (exploring a problem from multiple perspectives) is a low-cost improvement for sleep consolidation, where the additional API calls are acceptable and diverse relationship detection is valuable.

- **Limitation for us**: The paper's domain specificity (Connections puzzles only), lack of ablations, single-run design, and absence of cost analysis make it a source of *ideas* rather than *evidence*. The principles are transferable; the specific results are not.

**Quality of the work**: Moderate. Not peer-reviewed (arXiv preprint, no venue). Written by a law professor, not an ML researcher -- this shows in the lack of ablations, cost analysis, and statistical rigor. But the full prompt publication (28-page appendix) enables reproducibility, and the honest identification of discernment as the weak point shows good analytical instinct. The mental model itself is the contribution; the experimental validation is illustrative rather than conclusive.
