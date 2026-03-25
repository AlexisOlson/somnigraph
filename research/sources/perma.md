# PERMA: Benchmarking Personalized Memory Agents via Event-Driven Preference and Realistic Task Environments -- Analysis

*Generated 2026-03-24 by Opus 4.6 agent reading arXiv:2603.23231*

---

## Paper Overview

**Paper**: Shuochen Liu, Junyi Zhu, Long Shu, Junda Lin, Yuhao Chen, Haotian Zhang, Chao Zhang, Derong Xu, Jia Li, Bo Tang, Zhiyu Li, Feiyu Xiong, Enhong Chen, Tong Xu (University of Science and Technology of China, KU Leuven, MemTensor, City University of Hong Kong, Northeastern University). "PERMA: Benchmarking Personalized Memory Agents via Event-Driven Preference and Realistic Task Environments." arXiv:2603.23231, March 2026. 39 pages. Code: https://github.com/PolarisBoy1/PERMA.

**Problem addressed**: Existing personalization memory benchmarks have three blind spots: (1) they frame preferences as static statements given upfront, rather than signals that emerge through event-driven dialogue; (2) they treat user models as snapshots rather than evolving states that accumulate across sessions; (3) they evaluate end-to-end output quality without decoupling memory quality from generation quality. PERMA fills this gap by constructing temporally ordered interaction histories where preferences emerge gradually across multiple sessions and domains, and evaluating both memory fidelity and task performance separately.

**Core claim**: By modeling personalization as the maintenance of a temporally evolving "persona state" (an integrated representation of preferences and episodic memory derived from event-driven dialogues), PERMA reveals that advanced memory systems outperform raw long-context LLMs and vanilla RAG by linking related interactions into persistent persona states -- but all systems still struggle with temporal depth and cross-domain interference.

**Scale**: 10 synthetic users across 10 countries, 20 domains, ~800 events, 2,166 preference details, 1.8M tokens. Three context tiers: Clean (base), Noisy (5 types of in-session noise injection), Style-aligned Long-context (116k tokens, WildChat-interleaved). Seven memory systems benchmarked (RAG, MemOS, Mem0, Lightmem, Memobase, EverMemOS, Supermemory) plus 10 standalone LLMs. Two evaluation protocols: one-shot MCQ (8 options from 3 binary dimensions) and interactive evaluation via user simulator.

---

## Architecture / Method

### The Core Insight

PERMA reframes personalization evaluation from "can you recall this preference?" to "can you maintain a coherent, evolving persona state across noisy, multi-session, multi-domain interactions?" This is a meaningful shift. Prior benchmarks (LoCoMo, PrefEval, PersonaMem) either give preferences upfront or test isolated recall. PERMA tests whether the system can infer preferences from behavior, integrate them across domains, and apply them correctly when asked -- all under realistic noise conditions.

### Dataset Construction (Two-Stage Pipeline)

**Phase I: Timeline + Dialogue Generation.** Starting from PersonaLens user profiles (grounded in the PRISM dataset), a GPT-4o planner generates domain-specific interaction timelines. Events are typed as either "Emergence" (first interaction in a domain, establishing initial preferences) or "Supplement" (follow-up that refines/modifies/deepens preferences). A dialogue generation agent then produces multi-turn conversations for each event, conditioned on the user profile, prior history, and event description.

**Phase II: Task Insertion.** Evaluation tasks are inserted at three temporal checkpoints:
- **Type 1 (Zero-Memory)**: Before any relevant events occur. Tests parametric bias -- a non-personalized baseline.
- **Type 2 (In-Time)**: After all relevant domain sessions. Tests peak recall.
- **Type 3 (Post-Intervention)**: After intervening sessions on unrelated topics. Tests forgetting robustness and cross-domain interference.

Tasks are further split into Single-Domain (|dom| = 1) and Multi-Domain (|dom| > 1), the latter requiring cross-domain synthesis.

### Noise Injection (Text Variability)

Five types of in-session noise, drawn from a taxonomy of human prompt biases:

| Type | Description |
|------|-------------|
| Omitted Info | Vague references requiring context resolution ("that game") |
| Context Switch | Mid-conversation topic jumps |
| Inconsistent Pref. | User contradicts earlier preferences within the same session |
| Multi-lingual | Code-switching between languages |
| Colloquial Exp. | Slang, informal register |

This is injected per-event during dialogue generation, not post-hoc. The paper validates that 98% of injected noise adheres to the designated type.

### Linguistic Alignment (Style-Aligned Long-Context)

Real user queries sampled from WildChat replace the synthetic user turns, while preserving the underlying intent. WildChat dialogues are also interleaved as irrelevant distractors, expanding context to ~116k tokens. This tests whether systems can handle authentic linguistic diversity rather than exploiting the consistent lexical patterns of synthetic data.

### Evaluation Protocol

**MCQ Evaluation**: 8 options generated by permuting three binary dimensions -- Task Completion (T), Preference Consistency (P), and Informational Confidence (I). This is clever: it separates "did you do the task" from "did you respect preferences" from "were you confident about it."

**Interactive Evaluation**: An LLM-based user simulator provides corrective feedback until the agent's response satisfies all preferences, or 10 turns elapse. Reports Turn=1 (one-shot success rate) and Turn<=2 (success with one correction round), plus task Completion rate. Also tracks User Token (total tokens generated by the simulator), which measures how much clarification the system needed.

**Memory Fidelity**: BERT-F1 between retrieved context and ground-truth dialogues, plus a 1-4 Memory Score from an LLM judge evaluating coverage, accuracy, and noise.

---

## Key Claims & Evidence

### Main Results (Clean, Single-Domain, Table 4)

| System | MCQ Acc. | BERT-F1 | Memory Score | Context Token | Completion | Turn=1 | Turn<=2 |
|--------|----------|---------|-------------|--------------|------------|--------|---------|
| Kimi-K2.5 | **0.882** | - | - | 34078.6 | - | - | - |
| Qwen3-32B | 0.870 | - | - | 34078.6 | - | - | - |
| Gemini2.5-Flash | 0.870 | - | - | 34078.6 | - | - | - |
| RAG (BGE-M3) | 0.702 | **0.859** | 1.89 | 928.8 | 0.830 | 0.461 | 0.797 |
| MemOS | 0.811 | 0.830 | **2.27** | 709.1 | 0.842 | **0.548** | 0.801 |
| Mem0 | 0.686 | 0.781 | 1.91 | 340.1 | 0.797 | 0.475 | 0.775 |
| Lightmem | 0.657 | 0.792 | 1.83 | 297.3 | 0.794 | 0.532 | **0.813** |
| Memobase | 0.733 | 0.781 | 1.86 | 1033.3 | **0.804** | 0.504 | 0.830 |
| EverMemOS | 0.728 | 0.827 | 2.08 | 3230.5 | 0.846 | 0.508 | 0.790 |
| Supermemory | 0.655 | 0.799 | 1.84 | 94.3 | 0.804 | 0.501 | 0.804 |

### Temporal Degradation (Type 2 -> Type 3, Figure 3)

All memory systems show accuracy decline from Type 2 to Type 3:
- RAG: -2.3%
- MemOS: -3.1%
- Mem0: -1.4%
- Lightmem: -4.7%
- Memobase: -5.0%
- Supermemory: -5.0%

Memory Score degradation is more severe: Memobase drops -24.9%, Lightmem -11.0%, MemOS only -5.8%.

### Multi-Domain Performance (Table 6, Clean)

Multi-domain queries cause a pronounced drop for memory systems. Interactive Turn=1 success rates are dramatically lower:
- MemOS: 0.306 (vs 0.548 single-domain)
- RAG: 0.204 (vs 0.461)
- Mem0: 0.280 (vs 0.475)

This is the paper's most important finding: cross-domain synthesis is the hard problem, not single-domain recall.

### Retrieval Depth Analysis (Figure 7)

Expanding retrieval from top-10 to top-20 *hurts* most memory systems on multi-domain tasks but *helps* RAG. Memory systems compress context into structured persona states -- more retrieved items introduce noise. RAG retrieves raw dialogue fragments -- more items improve coverage.

### Style-Aligned Long-Context (Table 8)

At ~116k tokens with real linguistic style, standalone LLMs degrade (GPT-4o-mini fails entirely, Qwen2.5-14B-1M drops from 0.766 to 0.716). Memory systems maintain stable performance, demonstrating their value proposition for very long interaction histories.

### Methodological Strengths

1. **Decoupled evaluation.** Separating memory fidelity (BERT-F1, Memory Score) from task performance (MCQ, interactive) is genuinely useful. Most benchmarks conflate retrieval quality with generation quality.
2. **Temporal probing.** Type 1/2/3 checkpoints and positional probing (10%-100%) provide fine-grained temporal dynamics, not just aggregate scores.
3. **Interactive evaluation.** The user simulator with corrective feedback measures recovery ability, not just first-shot accuracy. Turn=1 vs Turn<=2 distinguishes "bad memory" from "needs one clarification."
4. **Noise as natural phenomenon.** The finding that noise sometimes *helps* memory systems (by emphasizing preferences through internal conflict) is surprising and credible.

### Methodological Weaknesses

1. **Fully synthetic data.** The dataset is LLM-generated end-to-end (GPT-4o for timelines, GPT-4o for dialogues, GPT-4o for noise injection, GPT-4o for style alignment). Despite human validation (97.75% MCQ accuracy, 1.99/2 event coverage), the preferences and dialogues have the homogeneity of machine-generated text. The WildChat style alignment partially addresses this, but the underlying preference structure is synthetic.
2. **All memory systems use GPT-4o-mini as backbone.** This controls for generation quality but means the benchmark doesn't test how memory systems interact with stronger backbone models. MemOS's advantage might partially reflect better compatibility with GPT-4o-mini specifically.
3. **MCQ as primary metric.** 8-way multiple choice is easier than open-ended generation. The paper acknowledges this: "selecting the correct option from predefined candidates is considerably easier than generating a reasoned answer." The interactive evaluation is more informative but only applies to memory systems.
4. **No ablation of the memory systems themselves.** The paper benchmarks systems as black boxes. There's no investigation of *why* MemOS outperforms -- is it the structured representation, the parallel retrieval channels (episodic/explicit/implicit), or something else?
5. **Scale is modest.** 10 users, 20 domains, ~800 events. The long-context variant extends to 116k tokens, but real long-term memory systems operate over much longer horizons.

---

## Relevance to Somnigraph

### What PERMA does that we don't

1. **Preference tracking as a first-class evaluation dimension.** PERMA explicitly evaluates whether the system maintains a coherent *persona state* -- the integrated set of user preferences and how they evolve. Somnigraph's LoCoMo benchmark evaluates factual QA recall, not preference consistency. Somnigraph stores preferences implicitly (as memories with `episodic` or `procedural` categories) but has no mechanism for preference emergence, refinement, or temporal validity tracking beyond the `valid_from`/`valid_until` schema. The reranker in `reranker.py` has no features related to preference priority or contradiction between stored preferences.

2. **Cross-domain synthesis evaluation.** PERMA's multi-domain tasks (e.g., combining Travel + Calendar + Restaurant preferences) directly test whether memory systems can join information across domains. Somnigraph's LoCoMo evaluation uses single-conversation QA; the multi-hop failure analysis (`docs/multihop-failure-analysis.md`) identifies vocabulary gap as the bottleneck but doesn't test cross-domain synthesis specifically. The graph edges in `scoring.py` (PPR traversal) could theoretically bridge domains, but this capability is untested.

3. **Noise robustness as a systematic dimension.** PERMA's five noise types (omitted info, context switch, inconsistent preferences, multilingual, colloquial) are injected systematically and evaluated separately. Somnigraph's write-time processing in `impl_remember()` doesn't explicitly handle noisy or contradictory input -- the LLM caller is expected to extract clean information. The FTS5 channel in `fts.py` would struggle with colloquial/slang input since BM25 matching is lexical.

4. **Interactive evaluation with corrective feedback.** PERMA's user simulator measures recovery ability (Turn=1 vs Turn<=2). Somnigraph's feedback loop measures post-hoc utility (explicit scores after retrieval), but there's no evaluation of whether the system can recover from a bad initial retrieval through follow-up interaction.

5. **Memory Score metric (1-4 LLM judge).** PERMA's EVAL_MEMORY_SCORE prompt evaluates retrieved memory across Coverage, Accuracy, and Noise dimensions independently. Somnigraph's LoCoMo evaluation uses a binary LLM judge for answer correctness -- it doesn't evaluate the quality of the *retrieved context* separately from the *generated answer*.

### What we already do better

1. **Explicit retrieval feedback loop.** PERMA evaluates memory systems as passive read/write stores. None of the 7 benchmarked memory systems incorporate per-query feedback that reshapes future retrieval. Somnigraph's feedback loop (EWMA aggregation, UCB exploration, per-query r=0.70 GT correlation) is the primary architectural differentiator, and PERMA's evaluation framework has no mechanism to measure its benefit. The `feedback_mean` and `ucb_bonus` features in `reranker.py` would be invisible to PERMA's evaluation.

2. **Learned reranker vs. fixed retrieval.** PERMA benchmarks systems with fixed top-k semantic retrieval. Somnigraph's 26-feature LightGBM reranker in `reranker.py` (+6.17pp NDCG over the hand-tuned formula) learns what "relevant" means from real data. The retrieval depth experiment (Figure 7) shows that going from top-10 to top-20 hurts most memory systems -- this is exactly the problem a learned reranker solves by scoring rather than just retrieving.

3. **Graph-conditioned retrieval.** Somnigraph's PPR expansion in `scoring.py`, Hebbian co-retrieval weights, and NREM-detected edges provide multi-hop retrieval paths. PERMA's finding that cross-domain synthesis is the hard problem (Turn=1 drops from 0.548 to 0.306 for MemOS) suggests that graph traversal -- connecting Travel preferences to Calendar events -- is exactly the kind of capability needed. The `betweenness` and `edge_count` features in the reranker provide graph signals that none of the benchmarked systems have.

4. **Sleep-based consolidation.** PERMA observes that MemOS wins partly because it categorizes memory into episodic facts, explicit preferences, and implicit preferences with parallel retrieval. Somnigraph's sleep pipeline (NREM edge detection in `sleep_nrem.py`, REM clustering and summarization in `sleep_rem.py`) performs this kind of organization automatically rather than relying on write-time extraction. The `layer` field (detail/summary/gestalt) in the schema already supports the multi-resolution representation that PERMA's analysis suggests is important.

5. **Biological decay.** PERMA's Type 3 checkpoint measures forgetting, and all systems degrade. Somnigraph's per-memory `decay_rate` and temporal suppression in `scoring.py` are designed for exactly this -- managing the relevance of old information without losing it. The `age_hours` and `hours_since_access` features in the reranker explicitly model temporal dynamics.

---

## Worth Stealing (ranked)

### 1. Cross-domain synthesis evaluation task

**What**: Design evaluation tasks that require integrating preferences from 2-3 different domain areas, similar to PERMA's multi-domain tasks but applied to Somnigraph's real memory corpus.

**Why it matters**: PERMA's most striking finding is the cross-domain performance collapse (Turn=1: 0.548 -> 0.306 for the best system). Somnigraph's PPR graph traversal and edge-based expansion are architecturally positioned to handle this, but we have no measurement. The multi-hop failure analysis found 88% vocabulary gap -- cross-domain queries likely have the same problem but across semantic domains rather than just lexical mismatch.

**Implementation**: Extend the LoCoMo benchmark harness or create a new evaluation set: select 20-30 memories that span multiple themes (e.g., a procedural memory about debugging that connects to a project memory about architecture decisions), construct queries that require synthesizing both. Measure whether PPR in `scoring.py` successfully traverses the cross-domain edges that NREM in `sleep_nrem.py` created. This directly tests whether the graph adds value for the hardest retrieval cases.

**Effort**: Medium (1-2 sessions). Requires manual construction of cross-domain queries against the real corpus.

### 2. Preference consistency tracking in the schema

**What**: Add a preference-evolution tracking mechanism -- when a memory updates or supersedes a prior preference, explicitly link them with a "revision" edge and maintain temporal validity.

**Why it matters**: PERMA shows that preference refinement (Emergence -> Supplement events) is the norm, not the exception. Somnigraph has `valid_from`/`valid_until` columns and "revision" edge types in the schema, but these are rarely populated. The contradiction detection problem (0.025-0.037 F1 universally) is partially a detection problem and partially a *representation* problem -- if we tracked preference evolution explicitly at write time, we wouldn't need to detect contradictions post-hoc.

**Implementation**: During `impl_remember()` in `tools.py`, when the category is `procedural` or the content describes a preference/decision, search for existing memories with overlapping themes and check for supersession. If found, create a "revision" edge and set `valid_until` on the old memory. This is a write-time enrichment, not a retrieval change. The NREM edge detection in `sleep_nrem.py` could also check for revision relationships during its linking pass.

**Effort**: Medium (1-2 sessions). The schema already supports it; the work is in the write-time heuristic.

### 3. Decoupled memory fidelity metric for LoCoMo evaluation

**What**: Evaluate the quality of *retrieved context* separately from answer correctness in the LoCoMo pipeline.

**Why it matters**: PERMA's BERT-F1 + Memory Score decoupling reveals that RAG has the highest BERT-F1 (0.859) but lowest MCQ accuracy among memory systems -- retrieval completeness doesn't guarantee synthesis quality. Conversely, some systems with lower BERT-F1 have higher task completion because their structured representations are easier for the reader LLM to use. Somnigraph's LoCoMo pipeline in `scripts/locomo_bench/` only evaluates final answer correctness. Adding a memory fidelity metric would distinguish retrieval failures from reader failures, directly informing whether to invest in reranker improvements or reader improvements.

**Implementation**: After the retrieval step in the LoCoMo pipeline, compute BERT-F1 between retrieved memories and the ground-truth evidence turns. Add an LLM judge step modeled on PERMA's EVAL_MEMORY_SCORE that evaluates coverage, accuracy, and noise of the retrieved context before the reader generates an answer. This parallels the existing R@10 metric but measures quality rather than just presence.

**Effort**: Low (1 session). The retrieval step already exists; this adds a scoring step.

### 4. Noise robustness testing for the retrieval pipeline

**What**: Systematically test how Somnigraph's retrieval handles noisy input at both write time and query time.

**Why it matters**: PERMA's surprising finding that noise sometimes *helps* memory systems (by emphasizing preferences through internal conflict) suggests that Somnigraph's write-time processing might benefit from adversarial testing. The FTS5 BM25 channel in `fts.py` is purely lexical and would likely fail on colloquial/slang queries. The vector channel may be more robust but this is untested. Understanding noise sensitivity would inform whether BM25-damped IDF keyword expansion (currently used for LoCoMo) needs adaptation for real-world noisy input.

**Implementation**: Create paraphrased and noise-injected variants of existing GT queries (slang, incomplete references, context-switched). Measure per-channel (FTS5 vs vector) retrieval quality degradation. This connects directly to roadmap Tier 2 #15 (paraphrase robustness test) but extends it with PERMA's noise taxonomy.

**Effort**: Medium (1-2 sessions). Query generation is straightforward; the evaluation infrastructure exists.

---

## Not Useful For Us

### MCQ evaluation format

PERMA's 8-way MCQ with binary dimension permutations (Task x Preference x Confidence) is designed for benchmarking many systems at scale. Somnigraph's evaluation needs are different -- we need to measure retrieval quality (R@10, NDCG) and answer quality (open-ended QA with LLM judge), not multiple-choice selection. The MCQ format is explicitly acknowledged by the authors as easier than generation, and our existing LoCoMo pipeline already tests the harder version.

### Synthetic user construction pipeline

PERMA's two-stage pipeline (PersonaLens profiles -> GPT-4o timeline -> GPT-4o dialogue generation) produces controlled, reproducible data. But Somnigraph operates on real interaction data from a single user with organic memory growth over months. The synthetic pipeline's value is in controlled experiments; our value is in real-world signal quality. Building synthetic users would not improve our understanding of production behavior. The LoCoMo benchmark already serves the controlled-experiment role.

### Style-aligned long-context variant

The WildChat-interleaved 116k-token context is designed to stress-test long-context LLMs. Somnigraph's retrieval pipeline operates on a memory store (~730 memories), not raw dialogue context. The context window stress test is irrelevant to a retrieval-based system.

### Interactive user simulator protocol

PERMA's LLM-based user simulator with corrective feedback is useful for evaluating agent-like systems. Somnigraph's MCP-based architecture means the Claude Code instance *is* the user -- the feedback loop already closes naturally through real interaction, not simulated interaction. Building a simulator would be less informative than the existing production feedback data.

---

## Impact on Implementation Priority

### Strengthens existing priorities

- **Roadmap Tier 2 #15 (Paraphrase robustness test)**: PERMA's noise taxonomy (5 types) provides a concrete framework for what "paraphrase" means. The finding that colloquial language and multi-lingual input challenge memory systems suggests this test should include more than just lexical paraphrase -- it should test register shifts. Still Tier 2 priority but now has a clearer design.

- **Roadmap Tier 1 #10 (Prospective indexing)**: PERMA's multi-domain performance collapse confirms that vocabulary gap is the central problem for cross-domain retrieval, not just multi-hop factual recall. Prospective indexing (generating hypothetical future queries at write time) would directly address the cross-domain synthesis failure that PERMA identifies as the hardest problem.

- **Expansion method ablation (#21)**: PERMA shows that expanding retrieval depth from top-10 to top-20 hurts most memory systems because compressed persona states introduce noise. This provides external evidence that Somnigraph's two-phase expansion (retrieve -> expand -> rerank) is architecturally sound -- the reranker prevents expanded candidates from degrading quality. The ablation should specifically test whether the reranker successfully filters expansion noise.

### No change to priorities

- **Sleep impact measurement (#3)**: PERMA doesn't measure consolidation effects, so it provides no new information about sleep's value. Still the central unanswered question.
- **Counterfactual coverage (#5)**: PERMA's decoupled evaluation (BERT-F1 for retrieval, MCQ for task) is philosophically similar but methodologically different. The counterfactual check remains the right approach for measuring selection bias in Somnigraph's retrieval.

### New consideration

- **Cross-domain evaluation task**: Not currently on the roadmap. PERMA's data strongly suggests that cross-domain synthesis should be evaluated. This could be a lightweight addition to the LoCoMo benchmark pipeline or a separate mini-evaluation on the production corpus. Effort is low enough that it doesn't need to be a roadmap item -- it could be incorporated into the next evaluation session.

---

## Connections

### LoCoMo (benchmark)

PERMA explicitly positions itself against LoCoMo in Table 3, noting that LoCoMo lacks dynamic preferences, event-driven preference formation, context noise, implicit preferences, cross-domain reasoning, and interactive evaluation. PERMA does check all these boxes. However, LoCoMo's strength is in factual episodic memory with specific answer validation -- a different evaluation axis. The two benchmarks are complementary rather than competing. Somnigraph's 85.1% on LoCoMo measures factual recall; PERMA would measure preference-state maintenance. Neither alone is sufficient.

### Mem0 (extract-then-update pipeline)

Mem0 performs poorly on PERMA (MCQ 0.686, worst among dedicated memory systems). PERMA's case study (Appendix B) reveals why: Mem0 stores timestamped event summaries without preference abstraction, making it good for "what did the user ask about?" but weak for "what does the user prefer?" This mirrors Somnigraph's finding that Mem0's flat memory representation limits its LoCoMo performance (66.88% vs Somnigraph's 85.1%). The shared lesson: structured representation matters more than storage volume.

### MemOS (memory operating system)

MemOS is the clear winner on PERMA (MCQ 0.811, Memory Score 2.27, best Turn=1 at 0.548). PERMA's case study shows MemOS uses three parallel memory channels: episodic facts, explicit preferences, and implicit preferences. This multi-channel approach resembles Somnigraph's multi-category schema (episodic/semantic/procedural/reflection/meta) and multi-signal retrieval (FTS5 + vector + graph). Somnigraph's `memos.md` analysis should be cross-referenced for architectural comparison.

### Kumiho (graph-native memory, prospective indexing)

PERMA's cross-domain synthesis failure (Turn=1 dropping 44% from single to multi-domain) is exactly the problem Kumiho's graph-native architecture targets. Kumiho's prospective indexing (generating anticipatory queries at write time) would bridge the vocabulary gap that causes cross-domain misses. PERMA provides the empirical evidence for why prospective indexing matters -- the benchmarked systems without graph structure consistently fail at cross-domain synthesis.

### HyDE (hypothetical document embeddings)

PERMA's vocabulary mismatch problem (user says "chill and small" but the memory stores "boutique tech gatherings") is the same semantic gap that HyDE addresses. Somnigraph's multi-hop failure analysis found 88% zero content-word overlap between queries and evidence -- PERMA confirms this extends to preference-style queries too. HyDE or HyDE-like query expansion at retrieval time would help for both benchmarks.

### A-Mem (Zettelkasten-inspired, enriched embeddings)

A-Mem's autonomous linking and note evolution resemble what PERMA suggests is needed for preference tracking: memories that update themselves as new evidence arrives. Somnigraph's NREM edge detection is the closest equivalent but operates on a sleep cycle rather than at write time.

---

## Summary Assessment

PERMA is a well-constructed benchmark that identifies a genuine gap: existing evaluations measure factual recall, not preference-state maintenance. The distinction between "what happened" and "what does the user want" is real and underserved. The temporal probing (Type 1/2/3 checkpoints), noise injection, and decoupled memory/task evaluation are methodologically sound contributions that go beyond what LoCoMo, LongMemEval, or PersonaMem offer.

The paper's most important finding for Somnigraph is the cross-domain synthesis collapse. All memory systems -- including the best performer MemOS -- struggle dramatically when tasks require integrating preferences from multiple domains. This validates Somnigraph's investment in graph-based retrieval (PPR, Hebbian co-retrieval, NREM-detected edges) as architecturally positioned for the hardest retrieval problem, but also highlights that this capability is completely untested. The gap between "we have the architecture for this" and "we've measured that it works" is the actionable takeaway.

The paper's limitations are typical of benchmark papers: fully synthetic data, black-box system evaluation, modest scale. The 10-user, 20-domain design is sufficient for comparative analysis but doesn't stress-test the long-tail scenarios that real systems encounter. The reliance on GPT-4o-mini as a unified backbone means the results reflect that specific model's strengths and weaknesses. Still, as a benchmark contribution it's solid -- the evaluation dimensions (temporal depth, noise robustness, cross-domain synthesis, interactive recovery) are the right ones to measure, and the results provide useful signal about where current memory systems fail.
