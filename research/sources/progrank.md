# ProGRank: Probe-Gradient Reranking to Defend Dense-Retriever RAG from Corpus Poisoning -- Analysis

*Generated 2026-03-24 by Opus 4.6 agent reading arXiv:2603.22934v1*

---

## Paper Overview

**Paper**: Xiangyu Yin (Chalmers University of Technology), Yi Qi (University of Leeds), Chih-Hong Cheng (Chalmers / Carl von Ossietzky University of Oldenburg). "ProGRank: Probe-Gradient Reranking to Defend Dense-Retriever RAG from Corpus Poisoning." arXiv:2603.22934v1, 24 March 2026. 17 pages + 3 pages supplementary. No public code link provided.

**Problem addressed**: Dense-retriever RAG is vulnerable to corpus poisoning -- adversaries inject or edit passages so they rank in the Top-K for target queries and steer downstream generation. Existing defenses rely on auxiliary detectors, content filtering, LLM-side reasoning, or dedicated rerankers trained for the task. The paper wants a post-hoc, training-free defense that works purely from retriever-derived signals.

**Core claim**: Optimization-driven poisoned passages concentrate their retrievability on a small set of perturbation-sensitive features. By computing probe gradients of the retrieval similarity score with respect to a fixed parameter subset (a LayerNorm module) under stochastic input perturbations, you can derive two instability signals -- representational consistency and dispersion risk -- that, when combined with a score-gated penalty, selectively demote poisoned passages without harming clean retrieval utility.

**Scale of evaluation**: 3 datasets (MS MARCO, Natural Questions, HotpotQA), 3 dense retrievers (Contriever, DPR, BGE), 3 poisoning methods (PoisonedRAG, LIAR-RAG, Joint-GCG), 3 downstream generators (Qwen2.5-7B, Llama-3-8B, Mistral-7B). 100 evaluation queries per dataset, 50 clean passages per query. Both retrieval-stage metrics (Poison Hit Rate, Poison Recall Rate) and end-to-end metrics (ASR, ACC) reported. Ablations over perturbation repeat count R, probe layer L, perturbation type, and retriever backbone. Adaptive evasive attack also tested.

---

## Architecture / Method

### The Core Insight

Optimization-based poisoning crafts passages to maximize retrieval score for a target query. This optimization concentrates the passage's "retrievability" on a few perturbation-sensitive features. Benign passages, which earned their relevance through genuine content overlap, distribute their matching signal more broadly. This difference manifests as instability when you perturb the retriever's computation: poisoned passages show less consistent gradient directions and higher tail-risk deviations.

The key architectural decision is to probe *parameter-space* sensitivity (gradients of a specific model layer), not *representation-space* sensitivity (e.g., output embedding variance under dropout). This gives a finer-grained signal than generic uncertainty estimation.

### Method in Detail

**Step 1: Stochastic perturbation.** For each query-passage pair (q, p), apply R stochastic perturbation operators T_r to obtain perturbed variants p_r = T_r(p). Perturbation types: token dropout (random masking of passage tokens, probability 0.10), encoder dropout (switching the encoder to training mode to activate its native dropout), or their mixture.

**Step 2: Probe gradient extraction.** For each perturbed pair (q, p_r), compute the gradient of the similarity score with respect to a small fixed parameter subset (a LayerNorm module at layer L):

```
g_{r,theta,vartheta}(q, p) = nabla_vartheta s_theta(q, p_r)
```

This yields R gradient vectors -- a "sensitivity signature" for the pair.

**Step 3: Two instability signals.**

*Representational Consistency (Rep)*: Measures directional agreement of the R probe gradients -- essentially a gradient signal-to-noise ratio:

```
Rep = ||mean(g_r)||_2 / sqrt(mean(||g_r||_2^2) + epsilon)    in [0, 1]
```

Higher Rep means the gradients point consistently in the same direction across perturbations. Converted to penalty: P_rep = -log(Rep + epsilon), so low-consistency candidates get large penalties.

*Dispersion Risk (DR)*: Captures lower-tail instability. For each run r, compute the relative deviation from the mean gradient, map it through an exponential kernel (decay rate alpha), then take the tau-quantile across runs. This focuses on worst-case deviations rather than average behavior. Converted to a saturated penalty via:

```
P_dr = C * P_hat_dr / (P_hat_dr + C + epsilon)
```

where C bounds the maximum penalty.

**Step 4: Score-gated penalty fusion.** The penalties are applied only to high-scoring candidates (those near the Top-K decision boundary), using a sigmoid gate centered at a query-specific threshold:

```
w(q, p) = sigmoid(s(q, p) - mu(q))
```

where mu(q) is an upper-tail quantile of the base score distribution. The final defended score:

```
s_defended(q, p) = s(q, p) - w(q, p) * (P_dr(q, p) + P_rep(q, p))
```

This is elegant: low-scoring candidates (which can't enter Top-K anyway) receive negligible corrections, avoiding unnecessary rank perturbation in the tail.

### Design Choices and Justifications

- **LayerNorm probing**: Gradients w.r.t. all parameters would be prohibitively expensive. Probing a single LayerNorm module (typically at layer L=3) gives a compact sensitivity signature. Justified by analogy to Fisher information analysis.
- **Mixed perturbation**: Token dropout and encoder dropout target different failure modes. The ablation shows their combination performs best.
- **Saturating cap C**: Prevents a few extreme outliers from dominating the penalty. Without it, a single very unstable run could overwhelm the base score.
- **Score gate**: Restricts correction to the decision-critical region. The quantile threshold mu(q) uses m = ceil(sqrt(|D|)) as a heuristic for the upper tail, independent of K.
- **R = 20 perturbation repeats**: The ablation shows performance stabilizes around R = 20. This means 20 forward+backward passes per candidate per query.

### Limitations Acknowledged

The authors are explicit that: (1) not all poisoned passages are intrinsically unstable -- they target the "optimization-concentrated" subset; (2) the formal mechanism analysis (Section 3.4) is an abstraction, not a proof; (3) the score gate depends on the base-score distribution, which may not generalize to all retriever architectures.

### Limitations Missed

- **Cost**: 20 forward+backward passes per candidate is brushed over with "practical deployment would instantiate D as a bounded first-stage candidate pool." Table 4 reports 4.73s/query mean latency -- reasonable for web search but not free. No analysis of how cost scales with pool size B.
- **No non-adversarial evaluation**: The entire evaluation is under attack conditions. There is no measurement of whether ProGRank degrades clean retrieval quality when no poison is present. The clean utility table (Table 5) is only measured *with* poisoned corpus -- not on a fully clean corpus.
- **Limited retriever diversity**: All three retrievers are BERT-scale dual encoders. No evaluation on cross-encoders, late-interaction models (ColBERT), or API-based embedders where gradient access is impossible (the surrogate variant is mentioned but not evaluated).
- **100 queries per dataset**: Small evaluation set. Standard deviations in Table 2 are often comparable to the reported means.

---

## Key Claims & Evidence

### Main Retrieval-Stage Results (Fig. 2, averaged over configurations)

ProGRank consistently reduces Poison Hit Rate and Poison Recall Rate across all three datasets at all Top-K values (5-50). The strongest effect is on HotpotQA. On NQ and MS MARCO, Top-50 Poison Hit Rate drops to around 30% (from near 100% undefended).

### Downstream Generation Results (Table 2, Macro Avg at Top-5)

| Dataset | Method | Substr ASR | Judge ASR | Substr ACC |
|---------|--------|-----------|-----------|------------|
| HotpotQA | Baseline | 0.882 | 0.912 | 0.009 |
| | GRADA | 0.055 | 0.081 | 0.021 |
| | GMTP | 0.064 | 0.021 | 0.166 |
| | RAGuard | 0.380 | 0.377 | 0.351 |
| | **ProGRank** | **0.011** | **0.000** | **0.390** |
| NQ | Baseline | 0.678 | 0.756 | 0.165 |
| | GRADA | 0.028 | 0.055 | 0.013 |
| | GMTP | 0.093 | 0.307 | 0.280 |
| | RAGuard | 0.302 | 0.406 | 0.382 |
| | **ProGRank** | **0.000** | **0.000** | 0.382 |
| MS MARCO | Baseline | 0.799 | 0.891 | 0.081 |
| | GRADA | 0.088 | 0.027 | 0.069 |
| | GMTP | 0.116 | 0.093 | 0.220 |
| | RAGuard | 0.376 | 0.385 | **0.450** |
| | **ProGRank** | **0.033** | **0.000** | 0.447 |

ProGRank achieves 0.000 judge-based ASR on all three datasets (macro avg). Accuracy is competitive -- best on HotpotQA, tied or slightly below on NQ/MS MARCO.

### Adaptive Evasive Attack (Table 3, Macro Avg)

| Dataset | Substr ASR | Judge ASR | Substr ACC |
|---------|-----------|-----------|------------|
| HotpotQA | 0.030 | 0.016 | 0.372 |
| NQ | 0.016 | 0.034 | 0.365 |
| MS MARCO | 0.049 | 0.019 | 0.429 |

ProGRank remains effective under adaptive attacks -- ASR rises from 0.000 to 0.016-0.034 (judge-based), which is still far below the undefended baseline. Clean accuracy drops modestly.

### Ablation (Fig. 5, Macro Avg)

Full model is best. Removing P_dr causes a larger degradation than removing P_rep, suggesting dispersion risk is the more important of the two signals. Removing the score gate w(q,p) also hurts substantially -- confirming that focusing correction on the high-score region is important.

### Clean Utility (Table 5)

| Dataset (Metric) | Baseline | GRADA | GMTP | RAGuard | ProGRank |
|------------------|----------|-------|------|---------|----------|
| HotpotQA (F1) | **0.440** | 0.251 | 0.300 | 0.431 | 0.304 |
| NQ (EM) | 0.175 | 0.150 | 0.100 | 0.175 | **0.225** |
| MS MARCO (ROUGE-L) | 0.235 | **0.251** | 0.245 | 0.235 | 0.202 |

This is the most troubling result: ProGRank's clean utility is not uniformly good. On HotpotQA, it drops from 0.440 to 0.304 F1 (-31%). On NQ it improves, on MS MARCO it drops. These are measured *with* the poisoned corpus present, so it's hard to isolate how much is defense benefit vs. collateral damage.

### Methodological Strengths

- **Thorough ablation design**: R, L, perturbation type, retriever backbone, and component ablation all tested.
- **Adaptive attack evaluation**: Most defense papers skip this. ProGRank-aware evasive attack is tested.
- **End-to-end and retrieval-stage metrics**: Reporting both avoids the trap of showing retrieval improvement that doesn't translate downstream.
- **Clean separation of concerns**: Retrieval-stage analysis on 20% queries, downstream on 80% -- no data snooping.

### Methodological Weaknesses

- **No clean-only corpus evaluation**: Every evaluation includes poison. We never learn what ProGRank does to a clean corpus.
- **Small evaluation set**: 100 queries per dataset with high variance (std often ~0.3-0.5 in Table 2).
- **Surrogate variant unvalidated**: The paper mentions a surrogate retriever for black-box settings but never evaluates it. This is the variant most likely to be relevant for production systems using API-based embeddings.
- **Latency reporting incomplete**: Table 4 gives seconds/query but doesn't specify pool size B or hardware consistently.

---

## Relevance to Somnigraph

### What ProGRank does that we don't

**Nothing directly actionable.** ProGRank is a defense against adversarial corpus poisoning -- a threat that is structurally absent from Somnigraph's architecture. Somnigraph is a single-user personal memory system where all memories are authored by one Claude instance for one user. There is no external adversary injecting poisoned passages, no untrusted corpus boundary, no multi-user trust model. The attack surface ProGRank addresses does not exist in our setting.

The *technical machinery* -- gradient-based stability probing of a retriever -- requires gradient access to the embedding model's parameters. Somnigraph uses OpenAI's text-embedding-3-small via API. We cannot compute probe gradients against a black-box embedding service. The surrogate variant (using a local model as proxy) is mentioned but unvalidated.

**Score gating is a legitimately interesting idea**, though not for adversarial defense. The insight that reranking corrections should focus on the decision boundary (high-score candidates) rather than being applied uniformly is sound. In Somnigraph's context, this could mean: apply expensive reranking features only to candidates near the selection cutoff, not to the entire candidate pool. However, with ~730 memories and a typical candidate pool of ~112, the computational benefit would be negligible, and LightGBM prediction is already sub-millisecond.

### What we already do better

**Reranking via learned features.** ProGRank is explicitly "training-free" -- it computes instability penalties without learning from data. Somnigraph's 26-feature LightGBM reranker is trained on 1032 real queries with measured NDCG improvement (+6.17pp). The learned reranker integrates feedback signals, graph signals, metadata, and query-dependent features -- a far richer signal set than two instability metrics derived from gradient probing.

**Feedback loop.** ProGRank has no feedback mechanism. It makes the same decision about every passage every time. Somnigraph's explicit per-query feedback (r=0.70 GT correlation) means that even if a memory were somehow adversarially injected, its poor utility scores would suppress it over time. The feedback loop is an implicit anomaly defense.

**Hybrid retrieval.** ProGRank operates exclusively on dense retrievers (single-channel cosine similarity). Somnigraph's RRF fusion of FTS5 BM25 + vector similarity already provides natural robustness to single-channel manipulation: a passage that games vector similarity but has no keyword overlap would score poorly in the FTS5 channel and receive a weak RRF rank.

**Graph-based quality signals.** Betweenness centrality, PPR, Hebbian co-retrieval, and edge structure in Somnigraph provide contextual quality signals that are entirely absent from ProGRank's pure query-passage paradigm. A memory with no edges, no co-retrieval history, and no graph centrality would already be ranked poorly by the reranker.

---

## Worth Stealing (ranked)

### 1. Score-gated correction as reranker design principle

**What**: Apply reranker corrections preferentially to candidates near the Top-K decision boundary rather than uniformly.

**Why it matters**: During the LoCoMo two-phase expansion pipeline, the reranker scores all candidates from both baseline and expanded pools. Phase 2 candidates that are far below the selection threshold receive full feature computation even though they cannot plausibly enter the final result set. A score gate could skip expensive features for low-scoring candidates.

**Implementation**: In `src/memory/reranker.py`, after computing base RRF scores, apply a soft threshold to skip full feature extraction for candidates below a quantile cutoff. The 26 features vary in cost -- feedback/metadata features are cheap (table lookups), but `proximity` requires sliding-window computation and `betweenness` requires Brandes traversal (cached, but still). For candidates clearly outside the selection boundary, a reduced feature set could suffice. In practice, this is a micro-optimization: the full 26-feature extraction + LightGBM prediction for ~112 candidates takes <50ms. File to modify: `reranker.py` `_extract_features()`.

**Effort**: Low (1 hour). But the benefit is negligible at current scale. This becomes relevant if the candidate pool grows significantly or if expensive new features are added.

### 2. Perturbation-based embedding stability as a write-time quality signal

**What**: At memory write time, compute embedding stability under text perturbation (synonym replacement, sentence reordering) to flag memories whose embeddings are brittle.

**Why it matters**: The enriched embedding degradation problem (flagged by external reviewers) means that embeddings frozen at write time may diverge from evolving themes. A brittleness score could identify memories whose retrieval performance is fragile -- not because of adversarial manipulation but because of poor embedding stability. This connects to the "representation bifurcation" concern in `architecture.md`.

**Implementation**: During `impl_remember()` in `tools.py`, generate 3-5 paraphrases of the summary, embed each, compute cosine similarity variance. High variance = brittle embedding. Store as a metadata field. Could later become a reranker feature (feature #32: `embedding_stability`). The cost is 3-5 extra embedding API calls per memory write -- acceptable for ~10 new memories/day.

**Effort**: Medium (1 session). Requires embedding API calls during write path, schema addition for the stability score, and reranker feature integration.

### 3. Lower-quantile aggregation for tail-risk features

**What**: When aggregating noisy per-observation signals, use a lower quantile rather than the mean to capture worst-case behavior.

**Why it matters**: ProGRank's dispersion risk uses the tau-quantile (tau=0.1) to focus on tail instability rather than average instability. This aggregation strategy could apply to Somnigraph's `feedback_mean` EWMA: rather than tracking the exponentially-weighted mean of feedback scores, also track a lower quantile to identify memories that occasionally fail badly despite good average scores.

**Implementation**: In `src/memory/events.py`, alongside `feedback_mean` EWMA, maintain a `feedback_floor` tracking the rolling 10th percentile of recent feedback. Add as reranker feature #32 in `reranker.py`. The difficulty is that EWMA doesn't naturally produce quantiles -- would need a separate data structure (e.g., rolling window of last N scores).

**Effort**: Medium (1 session). Requires schema change (new column), event tracking modification, and reranker integration.

---

## Not Useful For Us

### Gradient-based probing

The entire core mechanism requires white-box access to the retriever's parameters (computing gradients of a LayerNorm module). Somnigraph uses OpenAI's text-embedding-3-small via API -- we have no gradient access. The surrogate variant (using a local model) is unvalidated in the paper and would require maintaining a separate embedding model solely for stability probing. This is architecturally incompatible.

### Adversarial defense framework

Corpus poisoning is not a threat in a single-user personal memory system where all content is authored by one Claude instance. There is no external data ingestion, no multi-user trust boundary, no web-scraped corpus. The entire problem formulation is inapplicable. If Somnigraph ever ingests external data (e.g., from web search or shared workspaces), this assessment would change -- but that's not on any current roadmap.

### The specific instability signals (Rep and DR)

Even if we could compute them, representational consistency and dispersion risk are designed to distinguish optimization-crafted passages from natural ones. In a corpus where everything is natural (LLM-authored memories from the same instance), these signals would not carry the adversarial discrimination they're designed for. They might capture something about embedding quality, but that's speculative and untested.

### Mixed perturbation strategy

Token dropout + encoder dropout is specific to BERT-scale transformer encoders that you run locally. Not applicable to API-based embedding services.

---

## Impact on Implementation Priority

**Minimal impact on current priorities.** This paper does not motivate any changes to the current priority ordering in STEWARDSHIP.md.

- **P2 (reranker iteration)**: The score-gating idea is interesting but not actionable at current scale. The 31-feature retrain is the next concrete step; ProGRank does not add a new feature candidate that would improve NDCG on the production model.
- **P4 (LoCoMo benchmark)**: ProGRank's evaluation is on different datasets (MS MARCO, NQ, HotpotQA) under adversarial conditions. No overlap with LoCoMo QA evaluation. The expansion method ablation (roadmap #21) and multi-hop vocabulary gap problem are unrelated to adversarial defense.
- **Roadmap #10 (prospective indexing)**: Unrelated. ProGRank addresses poisoned retrieval, not vocabulary gap.
- **Roadmap #15 (paraphrase robustness)**: The perturbation-based stability concept has a tangential connection -- if memories are brittle under paraphrase, that's a retrieval robustness concern. But ProGRank's perturbations are at the model level (dropout, token masking), not the semantic level (paraphrase). The connection is more conceptual than operational.
- **Open problem: feedback self-reinforcement**: ProGRank's instability detection doesn't address this. The feedback loop's self-reinforcement risk is about selection bias and exposure effects, not about adversarial content.

The one long-term connection: if Somnigraph ever incorporates external data sources (web search results, shared documents, RAG over third-party corpora), adversarial defense at the reranker level becomes relevant. At that point, ProGRank's approach of computing instability signals and applying them via score-gated fusion could inform the defense layer design. But that's a speculative future architecture, not a current priority.

---

## Connections

### HyDE (hypothetical document embeddings)

Interesting contrast: HyDE addresses the vocabulary gap (query and document use different words) by generating a hypothetical answer document. ProGRank addresses adversarial manipulation (poisoned document games the similarity). Both operate at the retrieval stage but solve opposite problems. HyDE is far more relevant to Somnigraph's multi-hop failure analysis (88% vocabulary gap) than ProGRank.

### SPLADE (learned sparse representations)

SPLADE's learned sparse representations provide robustness through vocabulary expansion at the term level. ProGRank's perturbation-based robustness works at the embedding level. Both share the intuition that concentrated features are fragile, but SPLADE's approach (distributing the matching signal across more terms) is proactive while ProGRank's (detecting concentration after the fact) is reactive.

### GRADA (Graph-based reranking against adversarial documents)

Directly compared in Table 2. GRADA [37] uses a separate learned reranker with graph structure -- closer to Somnigraph's architecture (graph signals in the reranker). ProGRank outperforms GRADA on ASR but GRADA beats ProGRank on clean utility (MS MARCO). The comparison suggests graph-based reranking is better for clean quality while gradient-based probing is better for adversarial suppression.

### Self-RAG (self-reflective retrieval-augmented generation)

Self-RAG internalizes the retrieve-critique loop, which provides implicit adversarial robustness through generation-side verification. ProGRank's retriever-side defense is complementary but architecturally different. Self-RAG's approach of critiquing retrieved passages before use is more aligned with Somnigraph's existing architecture (feedback on retrieved results).

---

## Summary Assessment

ProGRank is a well-executed paper within its niche. The core idea -- that optimization-driven poisoned passages are detectable through gradient instability under stochastic perturbation -- is clean and the experimental design is unusually thorough for a defense paper (ablations across four dimensions, adaptive attack evaluation, end-to-end metrics alongside retrieval-stage metrics). The score-gating mechanism is a genuinely good design choice that prevents over-correction on low-scoring candidates. The formal mechanism analysis (Section 3.4) provides helpful intuition even if it falls short of a guarantee.

For Somnigraph, the paper's relevance is limited. The adversarial threat model does not apply to a single-user personal memory system, and the core technical mechanism (parameter-space gradient probing) requires white-box access to the embedding model that we do not have. The paper is not poorly done -- it simply addresses a problem we do not have with a technique we cannot use. The most transferable insight is the score-gating principle: when applying corrections to a ranked list, focus them on the decision boundary rather than distributing them uniformly. This is a minor design principle, not a research direction.

The single most important thing for Somnigraph: this paper confirms that reranking is a flexible and powerful stage for injecting diverse signals into retrieval. ProGRank adds instability signals; Somnigraph adds feedback, graph, and metadata signals. The shared architecture -- compute additional signals per candidate, fuse them with the base retrieval score, rerank -- validates the approach even though the signals themselves are entirely different.
