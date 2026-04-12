# A-MBER -- Affective Memory Benchmark for Emotion Recognition

*Generated 2026-04-11 by Opus agent reading arXiv:2604.07017*

---

## Paper Overview

**Paper**: Deliang Wen, Ke Sun, Yu Wang (Shanghai Shanmo Shongyang Technology Co., Ltd. / RedBearAI) and Xingyu Wang (China Mobile Communications Group Liaoning Co., Ltd.). "Memory Bear AI: A-MBER: Affective Memory Benchmark for Emotion Recognition." arXiv:2604.07017, April 2026. 29 pages.

**Code**: Not mentioned in the paper. No GitHub link provided. The paper describes a "released evaluation workflow" with Langfuse-based benchmark packaging, but no URL is given.

**Problem addressed**: Existing evaluation resources split into two disconnected tracks: (1) emotion datasets that test local/instantaneous affect (EmotionLines, MELD, RECCON), and (2) long-term memory benchmarks that test factual recall, temporal reasoning, or knowledge updating (LoCoMo, LongMemEval). Neither tests whether a memory system can use remembered interaction history to interpret a user's *present* affective state. A brief, polite response like "I'm fine" may mask disappointment, guardedness, or suppression when read against prior setbacks, recurring triggers, or a longer emotional trajectory. No existing benchmark measures this capability.

**Core approach**: Construct a synthetic benchmark through a staged pipeline (persona specification, long-horizon planning, conversation generation, turn-grounded annotation, question construction, post-generation curation, benchmark unit packaging). All evaluation items are anchored to specific dialogue turns and grounded in observable evidence. Three task families: judgment (infer present affective state), retrieval (identify supporting historical turns), explanation (justify interpretation with grounded reasoning). Layered composition separates core evaluation from diagnostic probes and stress tests.

**Key claims**: (1) Present affective interpretation is a distinct evaluation target from both local emotion recognition and factual memory. (2) Structured memory systems outperform all baselines on this target, with the strongest gains on long-range implicit affect, high memory-dependency items, trajectory-based reasoning, and adversarial conditions. (3) Gold-evidence performance remains below ceiling, indicating that affective interpretation difficulty persists beyond evidence selection. (4) The benchmark's memory-level annotations are not merely descriptive metadata but correspond to real performance differences.

---

## Benchmark Design

### Task Types

Three primary task families, each targeting a different aspect of affective memory:

| Family | Question | Answer format | Scoring |
|--------|----------|---------------|---------|
| **Judgment** | Infer the user's present affective/relational state at the anchor turn | Multiple choice / label | Exact match |
| **Retrieval** | Identify historically relevant supporting turns | Evidence turn IDs | Set match / turn-level F1 |
| **Explanation** | Justify the interpretation by linking anchor turn to history | Open-form text | LLM judge (rubric-based) |
| **Insufficient-evidence** | Recognize when available evidence cannot support a strong conclusion | Label / open-form | Exact match or calibration-sensitive judgment |

The three families work together to distinguish surface correctness from historically grounded reasoning. A model can get judgment right by guessing; retrieval forces it to show its work; explanation forces it to articulate why.

### Dataset Construction

**Scenario**: Teacher/counselor-student interaction across repeated sessions. Single controlled interpersonal scenario chosen for stable role relations, meaningful event accumulation, and natural alignment with affect-sensitive trajectories. The paper acknowledges this limits domain generality but argues it provides cleaner evaluation.

**Construction pipeline** (6 stages):

1. **Persona specification**: Persona pool with associated interaction patterns
2. **Long-horizon planning**: Global outline, session scripts, event plan, emotion arc, question plan -- all specified before any dialogue is generated. Cross-session structure is designed, not emergent
3. **Conversation generation**: Observable dialogue produced from the plan. Includes structured delivery descriptions (vocal properties) alongside text
4. **Turn-grounded annotation**: Affective, relational, and evidential structure annotated at turn level. Gold evidence tied to specific turn IDs
5. **Question generation**: Two-phase -- `hard_core` items first (long-horizon affective), then `everything_else`. Post-generation curation includes deduplication, difficulty filtering, structural pruning
6. **Benchmark unit packaging**: Each unit links anchor turn, context view, gold evidence turns, task specification, and expected output

**Alternative realizations**: The paper also describes agent-to-agent interaction as an auxiliary construction regime (better character differentiation and spontaneity, worse schema stability) and hardware-mediated interaction as a future extension. Single-agent staged generation is preferred for controllability.

### Evaluation Methodology

**Benchmark composition** -- three layers:

- **Core layer**: Implicit emotion judgment, retrieval, explanation. The main evaluation mass
- **Diagnostic layer**: Relation-state judgment, trajectory-based reasoning, multi-hop retrieval. Sub-capability probes
- **Stress-test layer**: Modality-missing, modality-ambiguous, difficult robustness cases

Adversarial items (pseudo-relevant history, insufficient evidence) cut across all layers as horizontal tags rather than forming a separate family. This is a thoughtful design choice -- adversarial cases test calibration within each task type rather than being isolated.

**Memory levels** (0-3):

| Level | Description |
|-------|-------------|
| 0 | Local -- answerable from the anchor turn alone |
| 1 | Near-history helpful -- recent context improves but isn't essential |
| 2 | Longer history useful -- cross-session context materially changes interpretation |
| 3 | Longer history required -- cannot be interpreted without substantial historical reconstruction |

**Reasoning structures**: Direct/single-hop, multi-hop (multiple earlier events must be integrated), trajectory-based (pattern across sessions), conflict/complex (contradictory signals requiring resolution).

**Context conditions** (5 compared systems):

1. No-Memory Baseline (anchor turn + minimal local context)
2. Long-Context Baseline (larger raw history window, no retrieval)
3. Retrieved-Memory Baseline (current input + retrieved subset)
4. Structured Memory System (Red Bear AI memory system -- event memory, relation cues, affective index)
5. Gold-Evidence Condition (oracle supporting turns from annotation)

Two evaluation conditions in the released repository: `session_local` (current session only) and `full_history` (complete cross-session history).

### Scale & Coverage

The paper does not provide exact item counts. It describes the benchmark as organized around multiple scenarios with multiple sessions each, but the total number of benchmark units, conversations, and turns is not explicitly stated. This is a notable gap -- without knowing the scale, it is difficult to assess statistical power of the experimental results.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Memory systematically improves affective interpretation | Table 3: Clear progression from No-Memory (0.34/0.29/0.31) through Long-Context (0.47/0.41/0.44) to Retrieved-Memory (0.58/0.54/0.53) to Structured Memory (0.69/0.66/0.65) across judgment/retrieval/explanation | **Credible.** Monotonic improvement across all three task families is strong internal validation. The progression from raw context to retrieval to structured memory is the right comparison ladder. |
| Gains concentrate on implicit affect, not near-range factual items | Table 4: Implicit affect shows largest delta (No-Memory 0.18 vs Structured 0.65 = +0.47), Near Fact shows smallest (0.49 vs 0.76 = +0.27) | **Credible.** This is exactly what a well-designed affective memory benchmark should show. The benchmark discriminates where it claims to. |
| Higher memory levels show larger structured-memory advantages | Table 5: Level 0 gap is 0.26 (0.52 vs 0.78), Level 3 gap is 0.42 (0.16 vs 0.58) | **Credible.** The widening gap validates that memory-level annotations track real difficulty. |
| Trajectory-based reasoning benefits most from memory | Table 6: Trajectory gap = 0.42 (0.21 vs 0.63), vs Direct/Single-hop gap = 0.28 (0.49 vs 0.77) | **Credible.** Consistent with the benchmark's thesis that reconstructing emotional trajectories requires structured memory. |
| Memory stabilizes interpretation under degraded local signals | Table 7: Under modality-missing, Structured Memory degrades less than baselines (0.73 standard vs 0.64 missing = -0.09, vs No-Memory 0.39 vs 0.30 = -0.09). Under adversarial: 0.73 vs 0.54 = -0.19 | **Partially credible.** The pattern is right but the absolute gap under adversarial conditions is similar across systems. More useful: structured memory under adversarial (0.54) still beats no-memory standard (0.39). |
| Gold-evidence performance stays below ceiling | Table 3: Gold-Evidence achieves 0.81/0.79/0.77 -- well below 1.0 | **Credible and important.** This means the benchmark tests interpretation quality, not just evidence access. Even perfect retrieval doesn't solve the task. |
| Affective memory evaluation is distinct from factual memory evaluation | Table 1 comparison + Section 6.1 argumentation | **Credible but self-serving.** The comparison table with LoCoMo/LongMemEval/LoCoMo-Plus is fair. The gap is real. Whether the gap matters enough to warrant a separate benchmark depends on whether anyone builds systems that optimize for affective interpretation specifically. |

---

## Comparison with Existing Benchmarks

### vs LoCoMo

LoCoMo tests factual recall, temporal reasoning, summarization, and dialogue generation over multi-session conversations. A-MBER targets a different capability: using history to interpret present emotional state, not to answer factual questions. The benchmarks are complementary, not competing. A system could score well on LoCoMo (excellent factual retrieval) and poorly on A-MBER (cannot reconstruct emotional trajectories), or vice versa. LoCoMo's construction pipeline is similar (synthetic multi-session dialogues), but A-MBER's planning-first approach with explicit emotion arcs is more structured.

### vs LongMemEval

LongMemEval focuses on memory operations (information extraction, multi-session reasoning, knowledge update, temporal reasoning, abstention). A-MBER shares the multi-session setting but differs in what it evaluates: affective interpretation rather than memory operations. LongMemEval's notion of difficulty is operational complexity; A-MBER's is interpretive ambiguity.

### vs BEAM

BEAM tests 10 memory abilities at extreme context lengths (up to 10M tokens). Its focus is on whether systems can handle scale. A-MBER is shorter but denser -- difficulty comes from affective event density and cross-session interpretive burden, not raw length.

### vs PERMA

The most interesting comparison. Both target the intersection of memory and personalization. PERMA evaluates preference-state maintenance through event-driven interactions; A-MBER evaluates emotional-state interpretation through interaction history. PERMA's "persona state" formalization parallels A-MBER's "present affective interpretation" -- both ask whether the system has integrated history into a coherent model of the user. Key differences: PERMA tests across 20 domains with cross-domain synthesis; A-MBER tests within a single teacher-student scenario with deeper affective trajectories. PERMA decouples memory fidelity from task performance; A-MBER decouples judgment from retrieval from explanation. PERMA has explicit noise injection; A-MBER has modality degradation and adversarial conditions.

### vs MemoryAgentBench

MemoryAgentBench tests memory-augmented agents on concrete tasks (scheduling, summarization). A-MBER is more interpretive -- the task is understanding, not action. No overlap in what they measure.

### What A-MBER uniquely tests

The trajectory-based reasoning structure is the genuinely novel contribution. No other benchmark explicitly tests whether a system can reconstruct how a user's emotional state *developed* across sessions and use that trajectory to interpret the present moment. This is qualitatively different from recalling that a user mentioned X three sessions ago.

---

## Relevance to Somnigraph

### Could we run against A-MBER?

**Uncertain -- likely no, at present.** The paper describes a released evaluation workflow but provides no code repository URL or data download link. The benchmark packaging uses Langfuse-based instances, which implies structured evaluation but not necessarily a public dataset. The evaluation conditions (`session_local`, `full_history`) suggest a runnable pipeline exists, but without access to the data or the code, running Somnigraph against it would require either (a) the authors releasing the dataset, or (b) reimplementing the construction pipeline.

If the data becomes available, the ingestion path would be straightforward: each benchmark unit has an interaction history (multi-session dialogue), an anchor turn, and task-specific questions. We would ingest the history as memories, then answer judgment/retrieval/explanation questions against the anchor turn. The retrieval task maps directly to our recall pipeline. Judgment and explanation would require a reader model on top.

**Key blocker**: The single teacher-student scenario is narrower than our typical usage (developer-assistant interaction). The affective vocabulary and emotional trajectories may not exercise the parts of our system that matter most (technical memory, procedural knowledge, graph-based cross-referencing).

### What would it reveal?

1. **Retrieval for implicit signals**: Our current retrieval is optimized for content-word overlap and semantic similarity. Affective retrieval requires connecting "I'm fine" to an earlier setback with zero lexical overlap -- exactly the vocabulary gap problem we identified in LoCoMo multi-hop analysis. The graph-based synthetic bridges might help here, but only if the extraction pipeline captures emotional connections, which it currently does not.

2. **Trajectory reconstruction**: Our graph has temporal edges and entity co-reference, but no "emotional arc" representation. A-MBER's trajectory-based questions would test whether our existing temporal ordering + Hebbian co-retrieval can reconstruct affect trajectories implicitly, or whether explicit emotional relationship edges are needed.

3. **Calibration under insufficient evidence**: The insufficient-evidence and adversarial subsets test whether our system can recognize when it doesn't have enough information. Currently, recall returns whatever it finds with scores -- there's no "I don't have enough to answer this" signal. This maps to a real production gap.

4. **Retrieval F1 on affective evidence**: Our reranker is trained on factual relevance judgments. Evidence turns for affective interpretation might have very different feature profiles (high temporal proximity, low content overlap, high relational signal). Running A-MBER retrieval would reveal whether the reranker generalizes to affective relevance or needs domain-specific features.

---

## Worth Stealing (ranked)

### 1. Memory-level annotations as analysis dimensions (low effort)

Each benchmark item tagged with how much it depends on history (Level 0-3) and what reasoning structure it requires (direct, multi-hop, trajectory, conflict). This is a cheap annotation that enables stratified analysis. We could retroactively tag LoCoMo questions with similar levels and analyze whether our reranker's gains concentrate where they should. If our Level 3 performance isn't substantially better than Level 0, the reranker isn't doing its job.

### 2. Layered benchmark composition (medium effort)

Core / diagnostic / stress-test layers with adversarial tags as a cross-cutting dimension. This is better than a flat question list because it explicitly separates "what are we measuring" from "how hard is it" from "what breaks under stress." Applying this to our LoCoMo evaluation would mean categorizing questions into main evaluation (single-hop factual), diagnostic (multi-hop, temporal), and stress-test (vocabulary gap, adversarial distractor) layers.

### 3. The judgment/retrieval/explanation decomposition (medium effort)

Testing the same anchor point across three task families forces a system to be right for the right reasons. We currently evaluate retrieval (R@k, MRR) and QA accuracy separately. Linking them per-question ("for this question, did you retrieve the right evidence AND answer correctly AND explain why?") would be more diagnostic. We already have this data in principle -- the LoCoMo pipeline has both retrieval metrics and QA scores per question. Building the per-question joint analysis is straightforward.

### 4. Adversarial insufficient-evidence items (medium effort)

Questions where the correct answer is "I don't have enough information to conclude this." Our current benchmark evaluation assumes every question has a positive answer. Adding negative/adversarial items to LoCoMo evaluation would test calibration. This connects to the limit parameter design -- a system that always returns memories can't express "I don't know."

### 5. Modality degradation as robustness test (high effort, low priority)

Testing whether interpretation holds up when local signals are weakened. Not directly applicable to our text-only system, but the general principle -- "does the system degrade gracefully when input quality drops" -- could be adapted. Example: corrupting or removing recent context and checking whether longer-range retrieval compensates.

---

## Not Useful For Us

- **The teacher-student scenario specifics**: The particular relational dynamics (counselor-student emotional trajectory) don't transfer to our developer-assistant usage. The framework is interesting; the scenario content is not.

- **Delivery description / modality representation**: A-MBER preserves vocal properties in structured text form (tone, pace, etc.) as a stand-in for audio. This is a text approximation of multimodal data. Not relevant to our pure-text system.

- **The Red Bear AI / Memory Bear system comparison**: The structured memory system evaluated is the authors' own system (Red Bear AI). The comparison is useful for understanding the benchmark's discriminative power but the specific system architecture is not documented in enough detail to learn from.

- **Single-agent vs multi-agent construction analysis** (Table 9): Interesting for benchmark builders, not for memory system builders.

---

## Connections

- **LoCoMo multi-hop failure analysis** (`docs/multihop-failure-analysis.md`): The vocabulary gap we identified (88% of evidence turns have zero content-word overlap with queries) is exactly what A-MBER's implicit affect items test. "I'm fine" has zero overlap with the earlier setback that gives it meaning. Our synthetic vocabulary bridges are designed for this problem.

- **PERMA** (`research/sources/perma.md`): Both benchmarks test whether memory systems can maintain coherent user models across sessions, but through different lenses (preference consistency vs affective interpretation). If we pursue PERMA (Priority 5), A-MBER's trajectory-based reasoning concept should inform how we evaluate temporal evolution of user state.

- **Graph extraction pipeline**: Our v6 extraction produces claims, segments, and entity co-reference edges. It does not produce emotional relationship edges (e.g., "user was disappointed about X in session 3, which connects to guarded response in session 7"). A-MBER's emotion arcs suggest this could be a useful extraction target, though the value is unproven.

- **Hebbian co-retrieval** (`src/memory/scoring.py`): Our Hebbian learning strengthens connections between memories retrieved together. If affective evidence tends to be co-retrieved with factual evidence about the same event, Hebbian learning might implicitly build the emotional connections A-MBER tests. An interesting question: does our existing co-retrieval pattern reconstruct emotional trajectories without explicit affect modeling?

- **Feedback loop and calibration**: The insufficient-evidence items connect to our utility calibration study findings. Our feedback loop tracks whether retrieved memories were useful; A-MBER asks whether the system can recognize when they're *insufficient*. These are related but distinct capabilities.

---

## Summary Assessment

**Relevance**: Medium. A-MBER identifies a genuine gap -- no existing benchmark tests affective memory as a distinct capability from factual memory. The evaluation design is thoughtful (especially the memory-level annotations, layered composition, and judgment/retrieval/explanation decomposition). However, the benchmark is currently unavailable (no public code or data), narrowly scoped (single teacher-student scenario), and targets a capability that is secondary to Somnigraph's primary use case (developer-assistant memory for a coding tool).

**Quality**: The paper is well-structured with clear experimental methodology. The five-system comparison ladder (no-memory through gold-evidence) is exemplary -- it cleanly isolates the contribution of access to history, retrieval quality, and structured memory organization. The results tables are consistent with the paper's claims. The self-acknowledged limitations are honest.

**Weaknesses**: (1) No public code or data release. (2) No reported dataset scale (number of benchmark units, conversations, turns). (3) The only structured memory system tested is the authors' own (Red Bear AI), raising evaluation independence concerns. (4) The teacher-student scenario is narrow and may not generalize. (5) Explanation scoring depends on LLM judge quality, which is acknowledged but not validated (no inter-annotator agreement reported for explanation items).

**For Somnigraph**: The conceptual contributions (memory levels, layered composition, three-family task decomposition) are more valuable than the benchmark itself. These ideas can improve our existing LoCoMo evaluation methodology without requiring us to run against A-MBER. If the dataset becomes public and includes scenarios beyond teacher-student interaction, revisiting would be worthwhile. For now, PERMA remains the more actionable next benchmark -- it tests cross-domain synthesis (our graph's intended strength) rather than affective interpretation (a capability we don't currently optimize for).

**Priority for catalog**: Low-medium. Worth tracking but not worth building an evaluation pipeline for unless the data ships publicly.
