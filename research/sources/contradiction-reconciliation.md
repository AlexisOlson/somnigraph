# Explanation Generation for Contradiction Reconciliation with LLMs — Analysis

*Generated 2026-03-24 by Opus 4.6 agent reading arXiv:2603.22735*

---

## Paper Overview

**Citation:** Jason Chan, Zhixue Zhao, Robert Gaizauskas. "Explanation Generation for Contradiction Reconciliation with LLMs." arXiv:2603.22735, March 2026. University of Sheffield, UK. 17 pages (9 main + 8 appendix). Code release planned but not yet available.

**Problem addressed:** Existing NLP work treats contradictions as errors to be resolved by choosing one side (arbitration). Humans don't do this — they hypothesize reconciling explanations that make both statements compatible. No benchmark or evaluation framework exists for this reconciliation capability.

**Core claim:** Reconciliatory explanation generation (REG) is a novel and challenging task where models must generate a natural language explanation that makes two apparently contradictory statements mutually compatible. Most LLMs achieve limited success (best: 40.25% success rate), and extended thinking/reasoning generally hurts coherence rather than helping.

**Scale of evaluation:** 18 models (7 Qwen3 sizes, 2 Tulu-3, 4 Olmo-3, Llama-3.3-70B, 2 DeepSeek-R1-Distill, gpt-5-mini, gpt-5.2). 275 contradiction instances from ChaosNLI-MNLI-C (MultiNLI re-annotated by 100 human annotators). 9 NLI judge models for automatic evaluation. Replicated on Implied NLI (8,000 instances). Three metrics: effectiveness, coherence, success (conjunction of both).

---

## Architecture / Method

### Task formulation

Given premise P and hypothesis H where an NLI judge predicts "contradiction," generate explanation E such that:

1. **Effective:** M_nli(P + E, H) = entailment (the combined premise+explanation entails the hypothesis)
2. **Coherent:** M_nli(P, E) ≠ contradiction AND M_nli(E, P) ≠ contradiction (the explanation doesn't contradict the premise, checked in both directions to mitigate order sensitivity)
3. **Successful:** Both effective AND coherent

The core insight is that both criteria can be evaluated using standard three-way NLI judgments (entail/neutral/contradict), which means LLM judges already trained on NLI can evaluate REG without novel annotation — the evaluation is automatic and scalable.

### Dataset curation

The paper repurposes ChaosNLI, a re-annotation of MultiNLI where 100 annotators labeled each instance. They select 275 instances where the plurality label is "contradiction" (w_con > 0.5), but notably none had unanimous contradiction agreement (w_con = 1.0, mean = 0.62). This means every instance has substantial annotator disagreement about whether it's even a contradiction — the paper leverages this ambiguity as signal rather than noise.

### Evaluation protocol

For each (P, H) pair and each of 18 explanation models, the generated explanation E is evaluated by 9 NLI judge models. Each judge only evaluates instances it initially classified as "contradiction." Metrics are averaged across all judges to produce mean rates. This per-judge filtering is methodologically important — a judge that didn't think P and H contradicted each other in the first place can't fairly evaluate whether E reconciled the contradiction.

### Judge selection

Judges selected from the same 18 models based on accuracy at predicting ChaosNLI-MNLI majority labels. Nine judges selected (accuracy 63-71%): Qwen3-{4B,8B,14B,32B}, Olmo-3-{7B,32B}, Llama-3.1-{8B,70B}, Llama-3.3-70B. Control experiment: randomly shuffled explanations get rejected ~99% of the time, confirming judges discriminate content vs. noise.

---

## Key Claims & Evidence

### Main results (Table 2, averaged across 9 judges)

| Model | Coherence (non-think/think) | Effectiveness (non-think/think) | Success (non-think/think) |
|-------|---------------------------|-------------------------------|--------------------------|
| Qwen3-0.6B | 38.32 / 44.40 | 24.38 / **52.87** | 8.64 / 9.02 |
| Qwen3-4B | **83.94** / 62.53 | 18.61 / 33.25 | 12.45 / 19.97 |
| Qwen3-14B | **85.56** / 73.45 | 20.59 / 25.46 | 14.59 / 17.26 |
| Qwen3-32B | 75.77 / 69.14 | 30.14 / 28.77 | 19.44 / 18.78 |
| Llama-3.3-70B | 76.56 / N/A | 37.28 / N/A | **26.20** / N/A |
| gpt-5-mini | N/A / 71.57 | N/A / 59.73 | N/A / **40.25** |
| gpt-5.2 | N/A / 72.30 | N/A / 49.95 | N/A / 31.56 |

### Key findings

1. **Coherence is easier than effectiveness.** Most models can avoid contradicting the premise (coherence 62-86%), but struggle to make P+E actually entail H (effectiveness 12-60%). Success rates are low because both must hold simultaneously.

2. **Thinking hurts coherence.** Models with chain-of-thought reasoning generally produce *less* coherent explanations. The paper shows a concrete failure mode: the model's thought trace explicitly acknowledges the contradiction ("even though it seems to contradict the premise") but then endorses a near-paraphrase of H as the explanation anyway. The reasoning process surfaces the contradiction without resolving it.

3. **Effectiveness from thinking plateaus at scale.** Mid-sized models (Qwen3-4B, 8B, Olmo-3-7B) gain 11-15pp in effectiveness from thinking. Larger models (Qwen3-32B, Olmo-3-32B) gain much less. The paper interprets this as a capacity ceiling — larger models already retrieve the latent knowledge without thinking, but can't utilize it better.

4. **Higher human disagreement = harder reconciliation.** Spearman r = -0.2257 (p < 0.0005) between w_con (proportion of annotators labeling "contradiction") and model success rate. When more humans think it's a real contradiction, models struggle more to explain it away.

5. **Proprietary models dominate.** gpt-5-mini (40.25%) and gpt-5.2 (31.56%) substantially outperform the best open model, Llama-3.3-70B (26.20%).

6. **Qwen3-0.6B's deceptive effectiveness.** The smallest model achieves 52.87% effectiveness (highest) but only 8.64% success. Manual inspection reveals it simply paraphrases H without adding explanatory content — the NLI judge sees P+paraphrase(H) and predicts entailment, but the paraphrase contradicts P.

### Reconciliation strategies observed (qualitative, Table 3)

- **Unstated context:** Adding temporal or circumstantial context not in either statement (e.g., "as evening falls" explains why vendors have both torches and merchandise)
- **Reinterpretation:** Shifting the meaning of key terms ("success" means overall objective, not the specific action; "semifinals are set" means matchups, not dates)
- **Entity disambiguation:** Interpreting P and H as referring to different entities or events
- **Figure of speech:** Treating one statement as metaphorical

### Methodological strengths

- Clean task formulation that maps to existing NLI evaluation infrastructure
- Multi-judge averaging reduces individual model bias (validated: self-preference bias < 1.06pp per Table 6)
- Randomized-explanation control confirms judges discriminate (shuffled explanations rejected ~99%)
- Replication on INLI dataset shows consistent patterns (proprietary still best, thinking still hurts coherence)

### Methodological weaknesses

- **The dataset is short-form.** Premise-hypothesis pairs are 1-2 sentences. Real contradictions in memory systems span paragraphs or emerge across distant conversations. The paper acknowledges this as a limitation.
- **w_con never reaches 1.0.** All "contradictions" have substantial annotator disagreement (mean 0.62). The task may partly measure the ability to find an explanation that already exists (the non-contradiction annotators saw it), not the ability to generate a novel reconciliation.
- **No taxonomy of reconciliation strategies.** The paper observes several strategies (Table 3) but explicitly declines to categorize them, citing difficulty. This limits actionability.
- **NLI judges have 63-71% accuracy on their own.** Using models that are wrong 29-37% of the time as judges introduces systematic noise. The multi-judge averaging mitigates but doesn't eliminate this.
- **Thinking models tested at different temperatures.** Non-thinking uses temperature=0; thinking uses temperature=0.6, top_p=0.95. This confound means some of the "thinking hurts coherence" finding could be temperature-related (higher temperature = more diverse = more likely to generate contradicting content).

---

## Relevance to Somnigraph

### What this paper does that we don't

**Framing contradiction as reconciliation rather than detection.** Somnigraph's current approach to contradiction (the `contradiction` flag on edges in `graph.py`, NREM edge detection in `sleep_nrem.py`) is purely about *detecting* that two memories contradict each other, then *filtering* contradiction-flagged edges from PPR traversal in `scoring.py`. The paper suggests a third option beyond "detect and flag" or "detect and discard": generate an explanation that reconciles the contradiction, which could become a new memory that links the two contradicting memories with context.

**Automatic evaluation of contradiction handling.** The paper's insight that NLI judgments can evaluate reconciliation quality without human annotation is directly applicable. Somnigraph currently has no way to evaluate whether its contradiction detection (or any future reconciliation) is working correctly. The three-metric framework (effectiveness, coherence, success) could be adapted to evaluate NREM's contradiction edge quality.

**Evidence that simple NLI-as-judge works.** The 9 NLI judges achieve reasonable discrimination (shuffled explanations rejected 99% of the time, self-preference bias < 1.06pp). This validates the NLI cross-encoder approach that's already on Somnigraph's roadmap — the judges here are general-purpose instruct models, not fine-tuned NLI models, which means even a lightweight NLI approach should work for detection.

### What we already do better

**Real contradictions, not curated ones.** Somnigraph's contradictions emerge organically over months of use — a user changes their preference, corrects an earlier belief, or reports a fact that conflicts with a stored memory. These are genuine information evolution events, not curated NLI pairs with 1-2 sentence scope. The paper's short-form pairs don't capture the complexity of contradictions that span weeks of conversation context.

**The valid_from/valid_until schema.** Somnigraph already has temporal validity tracking in `db.py` that handles the most common class of "contradiction" — information that was true at one time and false at another. This addresses the temporal-context reconciliation strategy (the paper's "unstated context" pattern) structurally rather than generatively.

**Contradiction-aware graph traversal.** The PPR contradiction edge filtering in `scoring.py` already prevents the system from co-surfacing contradicting information. The paper doesn't address retrieval at all — it only generates explanations for contradictions already identified.

**Feedback-driven relevance.** The paper evaluates explanation quality via NLI judges with no user feedback. Somnigraph's feedback loop (r=0.70 per-query GT correlation) would provide direct signal on whether a reconciliation explanation was actually useful in context, something no NLI judge can determine.

---

## Worth Stealing (ranked)

### 1. Reconciliation-as-enrichment during sleep

**What:** When NREM detects a contradiction edge between memories A and B, generate a reconciling explanation E and store it as the edge's `context` field, rather than just flagging the edge.

**Why it matters:** Currently, contradiction edges in `graph.py` are binary flags — they record that a contradiction exists but not *why* or *how* the memories might be compatible. The paper shows that even modest LLMs can generate explanations that reconcile apparent contradictions ~20-40% of the time. For a personal memory system where most "contradictions" are temporal (preferences changed) or contextual (different situations), the success rate on real data could be higher than on ChaosNLI.

**Implementation:** In `sleep_nrem.py`, after detecting a potential contradiction edge, call the LLM with the paper's prompt template (Appendix C.2): present both memory contents and ask for an explanation that makes them compatible. Use an NLI check (could be the same LLM acting as judge) to verify effectiveness and coherence. If successful, store E in the edge's `context` field and *don't* set the contradiction flag. If unsuccessful (NLI check fails), set the contradiction flag as currently done. This creates a graduated response: reconciled contradictions become enriched edges, genuine contradictions get flagged.

**Effort:** Medium. The prompting is straightforward (paper provides templates). The NLI evaluation adds one extra LLM call per detected contradiction. Main complexity is integrating with the existing NREM pipeline and deciding how to handle partial success (effective but not coherent, or vice versa).

### 2. NLI-based contradiction evaluation metric

**What:** Use the paper's three-metric framework (effectiveness, coherence, success) to measure NREM's contradiction detection quality on the real corpus.

**Why it matters:** Somnigraph has no measurement of contradiction detection quality. The roadmap notes this explicitly: "Contradiction detection rate on the real corpus: manually annotate ~50 known contradictions, measure NREM's detection rate." The paper's NLI-based evaluation could partially automate this — instead of manual annotation, use an NLI judge to assess whether flagged edges are genuine contradictions and whether unflagged edges between semantically similar memories should have been flagged.

**Implementation:** For each contradiction-flagged edge in `graph.py`, check M_nli(memory_A, memory_B) with an NLI judge. If the judge doesn't predict "contradiction," the edge may be a false positive. Conversely, sample non-contradiction edges between related memories and check for undetected contradictions (false negatives). This gives precision/recall estimates without manual annotation.

**Effort:** Low. Requires one LLM call per edge to evaluate. The NLI prompt is provided in the paper (Appendix C.1). Could run as a diagnostic script rather than integrated into the pipeline.

### 3. Reconciliation strategy as a reranker signal

**What:** When two memories that appear relevant to a query contradict each other, the *type* of reconciliation (temporal change, context-dependent, reinterpretation, genuine conflict) could inform which memory to rank higher.

**Why it matters:** Currently, contradiction-flagged edges simply block PPR traversal. But the paper's strategy taxonomy (temporal context, reinterpretation, entity disambiguation) suggests that different contradiction types should be handled differently. A temporal contradiction ("I prefer X" vs. "I now prefer Y") should surface the newer memory; a contextual contradiction ("X works for project A" vs. "X doesn't work for project B") should surface whichever matches the query context.

**Implementation:** This would require storing the reconciliation type (from idea #1) in the edge metadata. The reranker in `reranker.py` could then use contradiction-type features to differentially weight contradicting memories rather than blocking both. This is speculative — the paper doesn't provide a taxonomy, and building one for personal memory contradictions would be original work.

**Effort:** High. Requires the reconciliation pipeline from #1 plus taxonomy development plus reranker feature engineering.

---

## Not Useful For Us

**The REG task as formulated.** The paper's task — generate a standalone explanation E that reconciles P and H — is designed for evaluation purposes. In a memory system, the goal isn't to produce a separate explanation artifact but to handle contradictions at retrieval time (surfacing the right memory) or storage time (maintaining consistency). Generating a free-text explanation that satisfies NLI criteria is a means to an end, not the end itself.

**The specific dataset (ChaosNLI-MNLI-C).** These are crowd-sourced NLI pairs from diverse genres (fiction, news, letters). They don't represent the types of contradictions that appear in personal memory (preference changes, factual corrections, context-dependent assertions). The finding that r = -0.2257 between annotator disagreement and model success is interesting but tells us nothing about how models would perform on personal-memory contradictions where temporal context is always available.

**The model ranking.** Somnigraph uses whatever LLM the Claude Code session provides (currently Opus). The paper's detailed model comparison across 18 models (Table 2) is irrelevant to model selection because Somnigraph doesn't choose its LLM. The finding that proprietary models dominate is unsurprising and uninformative for our context.

**Extended thinking analysis.** The finding that thinking hurts coherence is interesting for the NLP community but irrelevant to Somnigraph. Sleep pipeline operations already use structured prompts with specific output formats, not open-ended chain-of-thought reasoning. Whether thinking helps or hurts REG doesn't affect how Somnigraph would implement reconciliation.

**The multi-judge evaluation protocol.** Using 9 judges and averaging is appropriate for benchmarking but overkill for a production memory system. A single NLI check (one LLM call) is sufficient for Somnigraph's purposes — the variance reduction from multiple judges isn't worth the cost for a system processing a few contradictions per sleep cycle.

---

## Impact on Implementation Priority

**Strengthens the NLI cross-encoder roadmap item.** The paper validates that off-the-shelf NLI judgments can evaluate contradiction-related tasks with reasonable accuracy (shuffled explanations rejected 99%). This is evidence that the NLI cross-encoder item listed in STEWARDSHIP.md ("NLI cross-encoder for contradiction detection") is a viable approach. The paper's judges are general instruct models, not fine-tuned NLI models — if they can discriminate at this level, a dedicated NLI model (like the DeBERTa-v3 approach from EXIA GHOST in `similar-systems.md`) should do even better.

**Reframes the contradiction detection open problem.** The roadmap currently frames contradiction as a detection problem (0.025-0.037 F1 universally catastrophic). This paper suggests a complementary framing: even when detection is imperfect, *reconciliation* can recover value from detected contradictions rather than just flagging them. This doesn't solve the detection problem but suggests that detection + reconciliation could be more useful than detection alone.

**No impact on current priorities (P2-P4).** The reranker iteration (P2) and LoCoMo benchmark (P4) work are unaffected. Contradiction handling is an open problem, not a priority item — this paper provides evidence and framing for when that problem is eventually addressed, but doesn't make it more urgent than the current roadmap items.

**Informs Tier 3 experiment: "Contradiction detection rate."** The roadmap proposes manually annotating ~50 known contradictions to measure NREM's detection rate. The paper's NLI-based evaluation (Worth Stealing #2) could partially automate this, reducing the manual effort. The experiment is still 1 session of effort but could produce precision/recall estimates rather than just a detection rate.

---

## Connections

### EXIA GHOST (similar-systems.md)

EXIA GHOST claims 92.31% F1 on contradiction detection using a DeBERTa-v3 NLI cross-encoder ensemble. This paper validates the NLI approach from the other direction — showing that NLI judgments can evaluate reconciliation quality. Together, they suggest: use a fine-tuned NLI model for detection (EXIA GHOST's approach), then use a general LLM for reconciliation (this paper's approach). Two complementary pieces of the same pipeline.

### Kumiho (kumiho.md)

Kumiho has formal AGM belief revision correspondence — when a new memory contradicts an existing one, it follows principled contraction/revision rules. This paper's reconciliation framing is the complement: instead of deciding which belief to keep (AGM revision), generate an explanation for why both might be true (reconciliation). Kumiho's `findContradictoryStatements()` would be the detection layer; this paper's approach would be an alternative to Kumiho's discard-one-side revision strategy.

### Mem0 (mem0-paper.md)

Mem0's extract-then-update pipeline handles contradiction by simply overwriting old information. This is the most aggressive arbitration strategy — the opposite of reconciliation. The paper's finding that reconciliation succeeds ~20-40% of the time suggests that always-overwrite loses information that could be preserved through explanation.

### HyDE (hyde.md)

The paper's reconciliation strategies (adding temporal context, reinterpretation, entity disambiguation) are structurally similar to HyDE's hypothetical document generation. Both ask an LLM to generate text that bridges a gap — HyDE bridges query-document vocabulary gap, REG bridges the logical gap between contradicting statements. If Somnigraph implements HyDE for vocabulary-gap retrieval (the multi-hop failure analysis motivation), the same infrastructure could serve reconciliation generation.

---

## Summary Assessment

This paper introduces a clean task formulation and evaluation framework for an underexplored problem. The key methodological contribution — using standard NLI judgments to evaluate reconciliation without human annotation — is sound and validated. The finding that most LLMs achieve limited success (best 40.25%) is an honest negative result that correctly identifies a capability gap. The thinking-hurts-coherence finding is genuinely surprising and well-evidenced.

For Somnigraph specifically, the paper's primary value is conceptual rather than directly implementable. It reframes contradiction from a binary problem (detect and flag/discard) to a graduated one (detect, attempt reconciliation, flag only on failure). This is a better fit for personal memory where most "contradictions" are temporal or contextual rather than genuinely irreconcilable. The practical ideas — reconciliation-as-enrichment during sleep, NLI-based evaluation of contradiction edges — are worth pursuing when the contradiction detection problem rises in priority.

The paper's main limitation for our purposes is scope. One-to-two sentence NLI pairs are a different beast from the multi-paragraph, temporally-evolving contradictions in a personal memory system. The 40.25% success rate (best model, on short-form pairs with inherent annotator disagreement about whether they're even contradictions) should be read as an upper bound on what to expect for harder, real-world contradictions. The most important takeaway is the reconciliation framing itself — not the specific numbers.
