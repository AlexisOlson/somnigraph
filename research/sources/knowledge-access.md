# Knowledge Access Beats Model Size: Memory Augmented Routing for Persistent AI Agents -- Analysis

*Generated 2026-03-24 by Opus 4.6 agent reading arXiv:2603.23013v1*

---

## Paper Overview

**Paper**: Xunzhuo Liu, Bowei He, Xue Liu, Andy Luo, Haichen Zhang, Huamin Chen. (vLLM Semantic Router Project, MBZUAI, McGill University, Mila, AMD, Red Hat.) "Knowledge Access Beats Model Size: Memory Augmented Routing for Persistent AI Agents." arXiv:2603.23013v1, March 2026. 10 pages + appendix. No code link provided.

**Problem addressed**: Production AI agents face a cost-quality tension -- large models (100B+) answer better but cost 10-30x more per token than small models (7-8B). Model routing (sending easy queries to cheap models) and conversational memory (storing facts from prior interactions) have been studied independently, but their interaction is unstudied. The gap matters because the interaction is non-obvious: memory could help routing by grounding the small model, or hurt it by injecting irrelevant context that inflates confidence.

**Core claim**: A 2x2 factorial experiment (memory x routing) reveals a compound effect: routing provides cost savings, memory provides correctness, and together they achieve what neither can alone. The 8B model with memory (30.5% F1) recovers 69% of the full-context 235B quality (43.9%) at 96% lower effective cost. The key finding is that without memory, the small model is "confidently wrong" -- routing keeps nearly all queries on the cheap path regardless, but memory transforms those confident-wrong answers into confident-right ones.

**Scale**: 152 LoCoMo questions (conversation 26, single conversation), 500 LongMemEval questions (~48 sessions, ~121K tokens per haystack). Models: Qwen3-VL-8B-Instruct vs. Qwen3-VL-235B-A22B-Instruct-AWQ. Metrics: token-level F1, BLEU-1, EffCost (input + 4x output) x (P/8B). Baselines: cold 8B, cold 235B, full-context 235B, Mem0/Zep/LangMem published numbers.

---

## Architecture / Method

### The Core Insight

The paper identifies a previously unreported interaction: confidence-based routing and conversational memory address *orthogonal failure modes*. Routing is quality-blind (the small model is confident about user-specific facts it has never seen), while memory is cost-blind (every query gets the same model). The compound effect is that memory makes routing worthwhile by changing *what* the confident answers contain, without substantially changing *whether* the model is confident.

### Three Components

**1. Cross-Model Memory Injection.** After each inference call, the routing layer stores the conversation turn-pair (question + response) as a verbatim record with a timestamp prefix, an embedding (Matryoshka 768-dim), and user metadata. At query time, top-k memories are retrieved and injected as system-context messages. The key design choice: store verbatim turn-pairs, not LLM-generated summaries. Preliminary experiments with summary-based storage *hurt* performance due to hallucinated content creating RAG poisoning. This is cross-model because memories from large-model interactions are available to the small model.

**2. Confidence-Based Routing.** A probe-then-escalate strategy:
1. Send the memory-augmented query to the cheap model (8B) with log-probability output
2. Compute mean log-probability across all N output tokens: `l_bar = (1/N) * sum(log p(t_i))`
3. Normalize to [0,1] confidence: `c = (l_bar - l_min) / |l_min|` where `l_min = -3`
4. If `c >= tau`, accept. Otherwise escalate to the expensive model (235B)

The normalization floor `l_min = -3` maps to `c = 0`; `l_bar = 0` maps to `c = 1`. At `tau = 0.50`, the threshold corresponds to geometric-mean token probability >= 22%. This is zero-shot -- no training, no labeled data, no external classifiers.

**3. Hybrid Retrieval.** Dense cosine search + BM25 keyword matching, fused via configurable strategy (RRF, weighted score, BM25-dominant modes). The paper treats this as a component but evaluates it primarily on LongMemEval. On LongMemEval, hybrid retrieval adds +7.7 F1 over cosine-only (44.2% vs 36.5%).

### Design Choices and Their Justifications

- **Verbatim storage over summaries**: Avoids hallucination cascading. The paper found in preliminary experiments that LLM-generated summaries for storage hurt performance -- a form of RAG poisoning. This motivates storing actual Q/A pairs.
- **Mean log-prob over verbalized confidence**: The paper cites Xiong et al. [29] showing white-box logprob methods outperform verbalized confidence, and that LLMs are systematically overconfident when asked to verbalize uncertainty.
- **No training requirement**: The entire pipeline is zero-shot. The confidence signal is the model's own log-probabilities; the memory store is populated from interaction history.

### Limitations Acknowledged vs. Missed

**Acknowledged**: LoCoMo is a single conversation; LongMemEval uses a stratified 100-question sample; cold-start trajectory (how quickly synergy builds) is uncharacterized; log-prob confidence can be miscalibrated; production-scale validation needed.

**Missed**:
- The 152-question LoCoMo evaluation is on conversation 26 only. The paper never runs all 10 conversations, making the results narrower than they appear. Somnigraph's LoCoMo benchmark uses all 10 conversations (1531 non-adversarial questions).
- No retrieval evaluation at all on LoCoMo -- only end-to-end QA F1. The paper cannot distinguish retrieval failures from reasoning failures.
- The paper stores raw turn-pairs without any consolidation, summarization, or decay. For a persistent agent accumulating memories over months, this creates unbounded storage growth and retrieval noise from stale information.
- The confidence routing threshold `tau = 0.50` is fixed; no analysis of how different thresholds trade off quality vs. cost beyond the single operating point.
- The Mem0 comparison is against published numbers (GPT-4o based, 41.0% F1) rather than a controlled re-run. Different backbone models make the comparison suggestive rather than rigorous.

---

## Key Claims & Evidence

### Main Results (LoCoMo, Table I)

| Condition | S (70) | M (40) | O (12) | T (30) | ALL (152) |
|-----------|--------|--------|--------|--------|-----------|
| Warm (compound) | 46.4 | 20.8 | 25.5 | 10.3 | **30.5** |
| Warm (memory only) | 45.7 | 20.8 | 25.5 | 10.3 | 30.1 |
| Cold compound | 16.7 | 6.2 | 9.6 | 13.1 | 13.0 |
| Cold 8B | 17.5 | 8.8 | 24.7 | 14.1 | 15.4 |
| Cold 235B | 19.8 | 7.6 | 16.1 | 6.4 | 13.7 |
| Full-ctx 235B | 61.9 | 47.0 | 19.1 | 16.1 | 43.9 |
| *Mem0 (GPT-4o)* | -- | *38.1* | *47.7* | *48.9* | *41.0* |
| *Zep* | -- | *35.7* | *49.6* | *42.0* | *36.7* |

### 2x2 Factorial (Table II)

| | F1 (%) | % on 8B | EffCost |
|---|--------|---------|---------|
| No memory, No routing | 15.4 | 100 | 15K |
| No memory, Routing (tau=0.50) | 13.0 (96% on 8B) | 96.1 | ~15K |
| Memory, No routing | 30.1 | 100 | 110K |
| Memory, Routing | **30.5** (100% on 8B) | 100 | **22K** |

The critical finding: with memory, 100% of queries stay on the 8B model. Memory doesn't change routing behavior -- it changes answer *quality*.

### Hybrid Retrieval on LongMemEval (Table IV)

| Retrieval | F1 (%) | BLEU-1 (%) |
|-----------|--------|------------|
| Cosine-only | 36.5 | 34.9 |
| Hybrid (BM25 + cosine) | **44.2** | **42.0** |

Per-type breakdown (Table V) -- gains concentrate in knowledge-update (+26.7), single-session-user (+19.0), temporal-reasoning (+5.7). Declines in single-session-assistant (-3.0) and multi-session (-2.2).

### Cost Projection (Table VI)

| Strategy | F1 | $/query | Personal (7K/mo) | Enterprise (3M/mo) |
|----------|-----|---------|-------------------|---------------------|
| All large (no memory) | 13.7 | 0.325 | $22.75 | $9,750 |
| All small (no memory) | 15.4 | 0.020 | $1.37 | $585 |
| Compound (ours) | 30.5 | 0.025 | $1.72 | $738 |

### Methodological Strengths

- The 2x2 factorial design cleanly isolates the interaction effect. This is the paper's best contribution -- it reveals that the compound is not just additive.
- The finding that memory does not change routing behavior (96% vs 100% on 8B) is a genuine insight about confidence-based routing.
- The LongMemEval evaluation (500 questions, ~48 sessions) is more robust than the LoCoMo evaluation, providing a real test of retrieval at scale.
- The per-category breakdown in Table V reveals where hybrid retrieval helps (entity-specific, fact-update) vs. hurts (multi-session synthesis), which is actionable.

### Methodological Weaknesses

- **Single-conversation LoCoMo is unreliable.** 152 questions from one conversation is a small and potentially unrepresentative sample. Somnigraph's LoCoMo benchmark uses all 10 conversations (1531 questions) for a reason.
- **No retrieval evaluation on LoCoMo.** Without R@k or MRR, the paper cannot attribute performance to retrieval quality vs. model reasoning. Somnigraph separately measures R@10 (88.7%) and end-to-end QA (85.1%).
- **Apples-to-oranges Mem0 comparison.** Mem0's 41.0% uses GPT-4o; this paper's 30.5% uses Qwen3-8B. The comparison is about system design philosophy, not head-to-head retrieval quality.
- **No ablation of memory store contents.** All 214 turns are stored as verbatim Q/A pairs. No comparison against extracted facts, summaries, or structured memories. The paper only tests one memory representation.
- **Temporal questions hurt by memory injection (-3.8 F1).** This is acknowledged but under-analyzed. The paper suggests "structured temporal representations" without exploring any.
- **LongMemEval uses 100-question stratified sample**, not the full 500. The 500-question claim applies to the memory store setup, not the evaluation.

---

## Relevance to Somnigraph

### What This Paper Does That We Don't

**1. Confidence routing / query difficulty estimation.** Somnigraph serves a single Claude Code instance and has no concept of routing queries to different models or estimating whether a query can be answered from memory alone. The paper's log-prob confidence signal is a lightweight proxy for query difficulty. While Somnigraph doesn't need model routing (it always uses Claude), the *confidence signal itself* could inform retrieval strategy: a query where the model is already confident might need fewer memories (lower `limit`), while low-confidence queries might benefit from broader retrieval.

This connects to the `limit` parameter in `recall()` (implemented in `src/memory/tools.py`). Currently the agent sets `limit` based on intent; a confidence signal could automate or calibrate this choice.

**2. Cross-model memory accumulation.** The paper stores responses from both models in the same memory pool, so knowledge from expensive model interactions is available to cheaper paths. In Somnigraph, all memories are created by the same Claude instance, but the principle generalizes: memories created during high-effort reasoning sessions could be tagged or prioritized differently from quick factual stores. This relates to the `source` metadata that could distinguish correction-driven memories from routine stores.

**3. Verbatim storage with empirical justification for avoiding summaries.** The paper found that LLM-generated summaries for storage create RAG poisoning. Somnigraph stores both `content` (full text) and `summary` (compressed) and embeds on enriched text (content + themes + summary). The "detail loss behind summaries" is already flagged as an open problem in `docs/roadmap.md` (Tier 3 experiment #21, resolution-fidelity evaluation). This paper provides additional empirical evidence that summary-based storage can actively hurt.

### What We Already Do Better

**1. Retrieval quality, measured and iterated.** The paper uses basic cosine + BM25 with no learned reranking. Somnigraph's 26-feature LightGBM reranker (`src/memory/reranker.py`) achieves +6.17pp NDCG over the formula baseline, trained on 1032 real-data queries. The LoCoMo-specific 17-feature reranker achieves R@10=88.7%. The paper's hybrid retrieval is Somnigraph's *starting point* (bare RRF), not its ceiling.

**2. Memory lifecycle.** The paper stores raw turn-pairs indefinitely with no consolidation, decay, or pruning. Somnigraph has biological decay (`src/memory/scoring.py`), sleep-based consolidation (NREM edge detection, REM clustering/summarization), temporal validity (`valid_from`/`valid_until`), and archival. For a persistent agent running over months, unbounded raw storage would create retrieval noise that the paper never addresses.

**3. Feedback loop.** The paper has no feedback mechanism. Retrieved memories are used once and their utility is never measured. Somnigraph's per-query explicit feedback (r=0.70 GT correlation) with EWMA aggregation is the primary architectural differentiator, and it shapes every subsequent retrieval via the `feedback_mean` feature in the reranker.

**4. Graph structure.** The paper's memory store is flat -- just embeddings and text. Somnigraph's edge graph with PPR traversal, Hebbian co-retrieval reinforcement, and contradiction/revision flags provides structural relationships that raw cosine + BM25 cannot discover. Multi-hop queries, where the paper's system struggles most (20.8 F1), are exactly where graph traversal helps.

**5. Structured temporal handling.** The paper prepends timestamps to turn-pairs and acknowledges this fails for temporal reasoning (-3.8 F1 on LoCoMo temporal). Somnigraph has `valid_from`/`valid_until` schema fields, temporal features in the reranker, and TReMu-style temporal reasoning is on the research roadmap. The paper correctly identifies that structured temporal representations are needed but doesn't implement any.

**6. Comprehensive LoCoMo evaluation.** Somnigraph evaluates all 10 LoCoMo conversations (1531 non-adversarial questions) with separate retrieval metrics (R@10, MRR) and end-to-end QA (85.1% Opus judge). The paper's 152-question single-conversation evaluation is thin by comparison.

---

## Worth Stealing (ranked)

### 1. Confidence as a retrieval-depth signal

**What**: Use the LLM's output confidence (or a proxy) to modulate how many memories to retrieve, rather than always using a fixed or agent-specified `limit`.

**Why it matters**: The paper shows that confidence is decoupled from correctness for user-specific queries -- the model can be confidently wrong. But the *direction* of the confidence shift when memories are injected (0.38 -> 0.93 in the worked example) is informative. If Somnigraph tracked how much confidence changes after memory injection, it could learn which queries genuinely benefit from more context. This connects to the `limit` parameter design and the open question of prospective vs. retrospective judgment (roadmap #6 follow-up).

**Implementation**: Not directly available in MCP (Claude doesn't expose log-probs). But the concept could be approximated by tracking the agent's *stated* confidence or the pattern of re-recalls (a query that triggers multiple recall attempts with different terms suggests low confidence). This would be a feature in the reranker or a `limit`-adjustment heuristic in `src/memory/tools.py`.

**Effort**: Medium. The proxy signals are noisy, and the MCP interface doesn't expose model internals. More of a research direction than a concrete implementation.

### 2. Temporal failure characterization

**What**: The per-category breakdown reveals that turn-pair memories with timestamp prefixes *hurt* temporal reasoning by -3.8 F1. The model parrots dates from context without performing temporal arithmetic.

**Why it matters**: This is empirical evidence for a failure mode Somnigraph should watch for. Somnigraph's `valid_from`/`valid_until` are database-level fields, not context-injected text, so they avoid the "text date" problem. But the finding reinforces that the roadmap's event-time implementation (Tier 3 #18) should store temporal information as structured metadata, not embedded in memory content. It also suggests that for temporal queries, injecting more memories could make things worse if the model treats timestamps as text.

**Implementation**: During event-time implementation (`docs/roadmap.md` Tier 3 #18), ensure temporal information is exposed as structured fields that the reader prompt can reason about, rather than relying on inline date parsing. Add temporal query detection to the LoCoMo benchmark evaluation to track whether Somnigraph exhibits the same degradation pattern.

**Effort**: Low (the design insight is free; the implementation is already planned).

### 3. Cross-model knowledge principle

**What**: Memories from high-quality interactions (expensive model, careful reasoning, corrected outputs) should be preferentially available to routine retrieval.

**Why it matters**: Somnigraph already has `source` metadata on memories (e.g., `source="correction"` for user corrections). The paper's cross-model principle suggests that memories born from effortful interactions are disproportionately valuable. This could inform the priority or decay_rate defaults: memories with `source="correction"` or tagged as emerging from extended reasoning could receive higher base priority.

**Implementation**: In `src/memory/tools.py` `impl_remember()`, adjust default priority based on source metadata. Corrections already get p7; this is consistent with the paper's finding. No code change needed -- the current defaults already implement this principle for the most important case.

**Effort**: Already done for the primary case (corrections at p7). Low effort to extend if new source tags are added.

### 4. RAG poisoning from summaries

**What**: The paper's preliminary finding that LLM-generated summaries stored as memories hurt performance via hallucination cascading.

**Why it matters**: Somnigraph's REM sleep generates summaries that become retrievable memories. If those summaries hallucinate details, they contaminate the retrieval pool. This connects directly to the "detail loss behind summaries" open problem and the resolution-fidelity evaluation (roadmap Tier 3 #21).

**Implementation**: When implementing the sleep impact measurement (roadmap Tier 1 #3), specifically measure whether REM-generated summary memories ever displace the original detail memories they were summarizing. Track whether summary memories receive lower feedback scores than their source memories. This is a diagnostic, not a code change.

**Effort**: Low (add tracking to existing planned experiment).

---

## Not Useful For Us

**Model routing.** Somnigraph serves a single Claude Code instance via MCP. There is no small/large model distinction, no cost optimization via routing, and no log-probability access. The entire routing layer -- the paper's primary contribution -- is architecturally irrelevant. Somnigraph does not control which model processes the query.

**Raw turn-pair memory format.** Somnigraph's memories are structured (content, summary, themes, category, priority, decay_rate, confidence, edges) with editorial judgment applied at write time and during sleep. Storing raw Q/A turn-pairs would be a regression in memory quality. The paper's choice is appropriate for their minimal-intervention design but incompatible with Somnigraph's enriched memory architecture.

**Zero-shot everything.** The paper's value proposition is that no training is needed -- pure zero-shot confidence routing + memory injection. Somnigraph already trains a reranker on 1032 queries and has 249K feedback events. The "no training" constraint doesn't apply; we have the infrastructure and the data.

**EffCost metric.** The effective cost metric (input + 4x output) x (P/8B) is specific to the model-routing cost optimization. Somnigraph's cost concerns are about token budget for memory injection (controlled by `limit` and `budget` parameters), not about model selection.

**Qwen3-VL-specific findings.** The confidence calibration, routing threshold, and F1 numbers are tied to Qwen3-VL-8B and 235B. These don't transfer to Claude, which has different confidence characteristics, different failure modes on user-specific queries, and no exposed log-probabilities via MCP.

---

## Impact on Implementation Priority

**Minimal impact on current priorities.** The paper's main contribution (memory x routing interaction) is orthogonal to Somnigraph's architecture. The secondary findings reinforce existing priorities rather than creating new ones:

- **Roadmap Tier 3 #18 (event-time implementation)**: The temporal failure finding (-3.8 F1) provides additional motivation for structured temporal metadata over inline timestamps. Priority unchanged -- this was already identified as high-impact.
- **Roadmap Tier 1 #3 (sleep impact measurement)**: The RAG poisoning finding adds a specific diagnostic to track -- do REM summaries displace or degrade their source memories? Priority unchanged, scope slightly expanded.
- **Roadmap Tier 3 #21 (resolution-fidelity evaluation)**: The verbatim-vs-summary finding provides an external data point supporting this experiment's hypothesis. Priority unchanged.
- **Hybrid retrieval validation**: The paper's +7.7 F1 from adding BM25 on LongMemEval (500 questions, ~48 sessions) is independent confirmation that hybrid retrieval matters at scale. Somnigraph already has hybrid retrieval (`src/memory/fts.py` BM25 channel + `src/memory/embeddings.py` vector channel, fused in `src/memory/scoring.py`). This confirms the design but doesn't change any priority.
- **LoCoMo benchmark (P4)**: The paper's 30.5% F1 on 152 LoCoMo questions with an 8B model (vs. Somnigraph's 85.1% on 1531 questions with Opus) is not directly comparable but adds another data point to the comparative benchmarking landscape. The paper's Mem0 comparison (30.5% vs 41.0%) is less informative than Somnigraph's (85.1% vs 66.88%) due to the backbone model difference.

No priority reordering recommended.

---

## Connections

### LoCoMo (benchmark)

Both systems evaluate on LoCoMo. The paper uses only conversation 26 (152 questions); Somnigraph uses all 10 (1531 non-adversarial). The paper's per-category breakdown aligns with Somnigraph's findings: multi-hop is the hardest category, temporal reasoning is challenging. But the paper's 30.5% overall F1 vs. Somnigraph's 85.1% reflects the enormous gap between raw turn-pair injection into an 8B model and a purpose-built retrieval pipeline with learned reranking and Opus as reader/judge.

### HyDE (hypothetical document embeddings)

The paper's confidence routing is conceptually adjacent to HyDE's approach of using LLM generation to improve retrieval. Both leverage the model's generative capacity as a signal for retrieval. The paper uses generation confidence (log-probs) to decide whether to trust the cheap path; HyDE uses generation content (hypothetical documents) to improve search. Neither is directly applicable to Somnigraph's MCP-based architecture, but both suggest that the generative model's internal signals (confidence, hypothetical answers) contain retrieval-relevant information that Somnigraph currently ignores.

### Mem0 (extract-then-update pipeline)

The paper explicitly compares against Mem0 on LoCoMo (30.5% 8B compound vs. 41.0% Mem0 GPT-4o). The comparison highlights different design philosophies: Mem0 extracts structured facts with entity resolution, enabling better multi-hop reasoning (38.1 M vs. 20.8 M). The paper's raw turn-pair storage is simpler but weaker for questions requiring synthesis. Somnigraph takes a middle path -- structured memories with themes and edges, but not entity-resolved facts.

### Kumiho (graph-native memory, prospective indexing)

The paper's temporal failure (-3.8 F1 on temporal questions) is exactly the problem Kumiho addresses with event extraction. Kumiho's structured events with consequences appended to summaries at ingestion time would address the timestamp-parsing failure the paper identifies. This reinforces the connection between Somnigraph's roadmap Tier 3 #18 (event-time) and Kumiho's approach.

### LongMemEval (benchmark)

The paper's LongMemEval evaluation (500 questions, ~48 sessions, ~121K tokens) is the more robust half of the paper's experiments. The per-type breakdown (Table V) provides the best evidence for hybrid retrieval's complementarity: knowledge-update gains +26.7, single-session-user gains +19.0, while multi-session and single-session-assistant decline. This category-specific analysis is more informative than aggregate numbers and could guide Somnigraph's retrieval channel weighting if per-category evaluation were added to the LoCoMo benchmark.

### SPLADE (learned sparse representations)

The paper's BM25 channel is a static keyword matcher. SPLADE provides learned sparse representations that could capture the same complementarity more effectively. The paper doesn't consider learned sparse retrieval, which is a missed opportunity given that the hybrid retrieval ablation (Table V) shows the gains are entity/keyword-specific -- exactly where SPLADE would excel.

---

## Summary Assessment

This paper makes one genuinely novel contribution: the 2x2 factorial design that reveals confidence-based routing is a "silent quality trap" for personalization workloads. Without memory, routing keeps queries on the cheap path (96% at tau=0.50) but the answers are confidently wrong. Memory doesn't primarily change routing behavior -- it changes what the routed answers contain. This insight is well-supported by the experimental design and clearly presented.

The rest of the paper is competent engineering with a thin evaluation. The LoCoMo evaluation (152 questions, single conversation, no retrieval metrics) is too narrow to support strong claims. The LongMemEval evaluation is broader and more convincing, providing useful per-category evidence for hybrid retrieval complementarity. The Mem0 comparison is apples-to-oranges (8B Qwen vs GPT-4o). The memory design (raw turn-pairs, no consolidation, no feedback, no structured metadata) is minimal by design but would not scale to persistent deployments.

For Somnigraph, the paper's practical value is limited. The model routing contribution is architecturally irrelevant (single-model MCP architecture). The hybrid retrieval finding confirms what Somnigraph already implements. The temporal failure characterization (-3.8 F1) and RAG poisoning from summaries are the most transferable insights, but both reinforce existing roadmap items rather than creating new ones. The single most useful takeaway is the empirical evidence that LLM-generated summaries stored as memories can hurt downstream QA -- a concrete data point for the sleep impact measurement (roadmap Tier 1 #3) and resolution-fidelity evaluation (roadmap Tier 3 #21).
