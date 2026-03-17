# Research Roadmap

What we've learned, what's still open, and what's worth pursuing next. Read alongside `architecture.md` (how the system works) and `experiments.md` (how we tested it).

---

## What we learned

Seven findings from 62 source analyses and 20+ tuning studies that would change your first decisions if you were building a memory system from scratch.

### 1. Feedback is the system

The retrieval feedback loop — where recalled memories receive explicit utility scores that reshape future scoring — is the dominant signal. Feature importance analysis (wm15): feedback accounts for 15–20% of AUC, but with a catch: it's only useful because the empirical Bayes Beta prior centers the scores. wm5 tested removal of the prior: `feedback_coeff` collapsed to near zero. Raw feedback is too noisy to help; the prior is the mechanism that makes it useful.

No other surveyed system (Mem0, Zep/Graphiti, HippoRAG, GraphRAG, Generative Agents) has a closed feedback loop. This is the primary architectural differentiator.

### 2. Theme boost was compensating for missing graph traversal

Theme boost had 67% AUC importance and 86% MRR importance in wm15 — the largest single feature. The original explanation: it converts continuous cosine similarity (noisy) to discrete signal (reliable). The revised explanation (post-PPR, wm19): theme boost was compensating for missing graph traversal. When PPR was added (wm19), `THEME_BOOST` collapsed from 0.97 to 0.19. Four independent external reviewers converged on the same conclusion: the high importance measured *compensation effect*, not intrinsic feature value. PPR is the correct implementation of what theme boost was approximating.

### 3. The parameter space has structure

Two distinct basins exist in the 7D parameter space (wm9):

- **Basin A** (high-k ≈ 14): Broad candidate set, post-RRF signals do the ranking work. Depends on a healthy feedback loop.
- **Basin B** (low-k ≈ 3): Sharp RRF ranking, post-RRF signals are refinements. More robust to feedback degradation but less responsive to learning.

This isn't noise or randomness — it's a real choice between feedback-dependent and feedback-robust configurations. Production chose Basin A (k=6 compromise) because the feedback loop is the competitive advantage; better to fix it than route around it.

### 4. Metric choice drives parameter values

AUC, MRR, and miss_rate@5k optimize to different constants from the same data (wm15 NSGA-II). AUC and MRR barely trade off (shallow Pareto front), but MRR trades off harshly with miss_rate (r = −0.517). Optimizing for "find the first good result fast" actively hurts coverage.

This isn't a bug. Different metrics ask different questions. You're choosing what "good retrieval" means, not finding the objectively correct answer.

### 5. Intuitive scoring signals can be counterproductive

Three signals were implemented, tested, and removed:

- **Quality floor** (wm1: optimal ratio = 0.0). Cliff detection handles the case better because it adapts to score distributions; a fixed ratio can't.
- **Shadow penalty** (all reviewers recommended removal). Confuses temporal relevance (is this memory current?) with query relevance (does this memory answer this query?). Now informs dormancy decisions during sleep instead.
- **Confidence weighting** (<0.1% contribution). Correlates with feedback, so multiplying by confidence double-counts the feedback signal.

Each seemed obviously right beforehand. Empirical testing disagreed.

### 6. Contradiction detection is universally catastrophic

0.025–0.037 F1 across all surveyed systems. This isn't a tuning problem — it's a representation problem. Current approaches detect contradictions between adjacent memories (vector neighbors), but distant contradictions and claim-level supersession go undetected. The `valid_from`/`valid_until` schema is a practical workaround, not a solution.

### 7. Offline consolidation is viable but unmeasured

The sleep pipeline (NREM edge detection, REM clustering/summarization, thematic consolidation) is the most novel piece of the system. It runs. It produces reasonable-looking results on manual inspection. We have no metrics proving it improves retrieval. This is the central unanswered question.

---

## External review findings

Four independent reviewers (Gemini 2.5 Pro, Claude Sonnet 4.6, Claude Opus 4.6, ChatGPT Pro) conducted blind reviews of the architecture and experiment documentation. The reviews converged on several findings that the internal perspective had missed.

### Convergent findings (all or most reviewers)

- **Feedback self-reinforcement.** The feedback loop is selection-biased — only surfaced memories get rated — and six downstream signals amplify the same exposure event. The system may optimize for retrieval habits rather than retrieval needs. See `architecture.md` § Feedback loop self-reinforcement.
- **Theme boost as compensation.** The high pre-PPR importance of theme boost (67% AUC) measured compensation for missing graph traversal, not intrinsic feature value. PPR (wm19) collapsed it from 0.97 to 0.19. The "discrete signal conversion" narrative was a post-hoc rationalization.
- **Enriched embedding degradation.** Embeddings frozen at write time diverge from evolving themes and summaries, creating less channel independence than assumed. See `architecture.md` § Representation bifurcation.
- **Sleep canonizing bias.** Sleep's editorial judgment (theme enrichment, summary generation, dormancy) may systematically favor certain memory types over others, but no measurement exists to detect this.
- **Category as wrong decay axis.** Per-category decay rates assume categories map to importance curves, but within-category variance likely exceeds between-category variance. Per-memory decay (already supported via `decay_rate` column) may be the right granularity.
- **Detail loss behind summaries.** Summaries compress for token efficiency but may lose the specific details that made a memory useful. No fidelity metric exists to detect this.

### High-confidence unique findings

- **PPR contradiction traversal** — PPR walked contradiction-flagged edges, actively co-surfacing conflicting information. Fixed in `scoring.py` (this session).
- **Bimodal utility prior** — The global Beta prior assumes unimodal utility. If the distribution is actually bimodal (high-utility vs. noise), the prior smooths over a real structural boundary.
- **Threshold coupling** — Small upstream perturbations (feedback, theme boost) may change PPR seed selection in ways that amplify downstream. Seed-set stability is unmeasured.
- **Vocabulary co-adaptation** — Queries and stored themes may co-evolve vocabulary, so the system works well on habitual phrasings but fails on paraphrases.
- **startup_load as cache** — Preload shapes query formation, creating ambiguity about whether feedback measures recall quality or preload effectiveness.
- **False bridge ratchet** — Sleep-created edges between topically distant memories can only gain weight (through co-retrieval reinforcement), never lose it (no pruning mechanism). Misclassified edges persist indefinitely.
- **Exact-token recall collapse** — When a query exactly matches stored themes/summary tokens, FTS dominates and vector similarity adds nothing. The system degrades to keyword search for its most familiar content.
- **Edge accumulation without pruning** — No mechanism removes or weakens edges that are no longer useful. The graph grows monotonically, and old misclassified edges can permanently distort traversal.

---

## Open questions

Research-grade questions with hypotheses and proposed experiments. Ordered by information value per effort.

### Does sleep improve retrieval?

**What we know:** Sleep runs, produces edges, summaries, archives. Manual inspection looks reasonable.
**What we don't know:** Whether retrieval metrics (AUC, MRR) are better after sleep than before.
**Experiment:** Snapshot GT metrics → run sleep → re-judge same queries → compare deltas.
**Effort:** 1–2 sessions once GT is complete.
**Hypothesis:** NREM edge detection helps (gives adjacency expansion real edges to walk). REM summary generation may hurt (summaries compete with detail memories for token budget). Net effect is positive but small.

### Do the LocoMo-tuned constants transfer to real data?

**What we know:** Current constants optimized on LoCoMo (public benchmark, ~17 testable queries). Real-memory GT has 1,047 queries (~500 judged so far) with different distributions (more personal, more thematic). Real-data tuning (wm24–wm34) is already underway using ~500 judged queries with 5-fold cross-validation.
**What we don't know:** Whether the LoCoMo-derived basin structure persists on real data. Whether constants shift significantly as more GT queries are judged (200 → 1,047).
**Experiment:** Compare wm24+ constants against LoCoMo-era constants. Re-run when GT judging reaches 500+ queries to test stability.
**Effort:** Ongoing (wm24–wm34 in progress).
**Hypothesis:** `RRF_K` and `FEEDBACK_COEFF` will be similar (structural). `THEME_BOOST` may shift (real queries use themes differently). Basin structure may not persist.

### Is the feedback loop healthy? *(Answered)*

**Result:** Yes, but the answer is level-dependent. Per-query correlation is strong (Spearman r = 0.70) — feedback tracks relevance in context. Per-memory correlation is weak (r = 0.14) — aggregating across queries destroys the context-dependent signal. The empirical Bayes prior operates at the per-memory level, meaning it smooths a context-dependent signal in a context-free way. The prior still helps (it regularizes noisy raw feedback), but a context-conditioned prior could preserve more of the r = 0.70 signal. See `experiments.md` § Utility calibration study for full analysis.

**Original hypothesis** (moderate correlation 0.4–0.6): partially confirmed. The per-query r = 0.70 exceeds the hypothesis; the per-memory r = 0.14 falls below it. The hypothesis didn't distinguish levels of aggregation — that distinction turned out to be the finding.

### Can cutoff history calibrate the cliff detector? *(Answered: replace it)*

**What we know:** The cliff detector (`CLIFF_Z_THRESHOLD = 2.0`, log-curve deviation) is far too permissive: it keeps more memories than the rater wants 96.1% of the time, by an average of 7.4 extra memories. The rater cutoff signal (846 events) is independently validated by GT — memories above cutoff have 2.2x the rate of GT-relevant results (51.8% vs 23.1%). The score gap at the rater cutoff is < 0.02 in 83.6% of cases, well below the cliff detector's sensitivity.

**What we tested:** Three approaches to improving the cliff:

**1. Calibrate cliff from score features: negative result.** Tested 10 list-level score features as predictors of the cutoff position. Linear regression: R² = -0.215 (worse than predicting the mean). Per-position logistic regression with 6 features: F1 = 0.721, dominated entirely by rank position (F1 = 0.720 alone). Score-based features add nothing. The cutoff is a content-level judgment that scores can't capture — 30% of relevance variance isn't in the scores (per-query feedback-GT r = 0.70, not 1.0).

**2. Per-memory cutoff penalty: deferred.** Per-memory cutoff rate does correlate with GT relevance (Spearman r = -0.31 at 20+ observations, improving with more data), but current sparsity is the binding constraint (352/1120 memories never retrieved, 217 with 1–2 appearances). Revisit when memories accumulate 50+ observations.

**3. Agent-specified result limit: the right approach.** Analysis of 846 cutoff events shows the cutoff is a content-aware decision that varies by query intent (mean 5.7, std 4.1, range 0–15). The agent already makes this judgment — the cliff is a score-based heuristic trying to approximate a content-level decision. The solution is to let the agent specify what it wants directly.

**Design: replace cliff with `limit` + token budget.**

Add a `limit` parameter to `recall()` — an integer specifying how many results to return. The agent sets it per-query based on intent. Anchors for guidance: **1, 3, 5, 8, 13** (Fibonacci-spaced to match the natural distribution of cutoff positions):

| Anchor | Use case | Cutoff data coverage |
|--------|----------|---------------------|
| 1 | Quick fact lookup | 14.8% of cutoffs fall at 0–1 |
| 3 | Focused retrieval | 38.3% at 0–3 |
| 5 | Standard recall (default) | 56.5% at 0–5 |
| 8 | Moderate exploration | 75.2% at 0–8 |
| 13 | Broad survey | 95.2% at 0–13 |

With oracle anchor selection, MAE = 0.71 (vs 3.46 baseline). Free-form integer is even better — the anchors calibrate judgment, they don't constrain choice.

The scoring pipeline simplifies: score and rank all candidates → take top `limit` → truncate to fit token `budget`. No cliff detection, no Z threshold, no log-curve fitting. Two parameters the agent understands instead of one hidden heuristic it can't influence.

The cliff detector (`apply_quality_floor` in `scoring.py`) becomes dead code. `CLIFF_Z_THRESHOLD` and `CLIFF_MIN_RESULTS` can be removed from constants.

**Interaction with existing parameters:** `limit` and `budget` are independent constraints applied in sequence — limit caps the count, budget caps the total tokens. If limit=8 but only 4 memories fit in the budget, 4 are returned. If limit=3 but the budget could hold 10, 3 are returned. The agent controls precision (limit), the system controls resource use (budget).

**Effort:** 1 session to implement. Add `limit` parameter to `recall()`, wire through to scoring, update CLAUDE.md snippet with anchor guidance, remove cliff detection code.

**What to track after deployment:** Log the agent's chosen limit alongside the cutoff_rank feedback. Over time, this tells us whether the agent is good at predicting how many results it needs — the equivalent of the cutoff calibration study but for prospective rather than retrospective judgment.

### Where does the feedback loop plateau?

**What we know:** Feedback is the dominant signal. More feedback events = better Bayes prior convergence.
**What we don't know:** At what point more feedback stops helping.
**Experiment:** Plot AUC as function of feedback event count per memory. Check for diminishing returns.
**Effort:** 1 session with GT data.
**Hypothesis:** Plateau around 5–8 events per memory, after which the prior has converged.

### What makes a query hard?

**What we know:** Some queries retrieve well, others don't. The 1,047 GT queries span factual, thematic, temporal, and personal types.
**What we don't know:** Whether failure patterns cluster by query type, and what the failure modes are.
**Experiment:** Cluster GT queries by characteristics (length, theme count, temporal reference, cross-domain), measure per-cluster AUC. Identify systematic failure patterns.
**Effort:** 1–2 sessions.
**Hypothesis:** Temporal queries ("what did we decide last month") and cross-domain queries (spanning multiple themes) are hardest. Single-topic factual queries are easiest.

### Is the utility distribution bimodal?

**What we know:** The global Beta prior assumes a unimodal utility distribution. External review flagged that real utility may be bimodal (high-utility memories vs. noise, with little in between).
**What we don't know:** The actual shape of the utility distribution. If bimodal, the single Beta prior smooths over a real structural boundary, pulling high-utility memories down and low-utility memories up.
**Experiment:** Fit mixture models (Beta mixture, Gaussian mixture) to the per-memory utility distribution. Compare BIC against the single Beta. If multi-modal, test per-mode priors.
**Effort:** 1 session.
**Hypothesis:** Bimodal. High-utility memories are systematically different from noise, and the prior should reflect this.

### Is PPR seed selection stable?

**What we know:** PPR uses top-N RRF scores as seeds. Small upstream changes (feedback, theme boost) change the seed set.
**What we don't know:** Whether small perturbations in seed selection cause large changes in PPR output — threshold coupling that amplifies noise.
**Experiment:** Perturb feedback coefficients by ±10%, measure seed-set overlap and PPR score correlation with unperturbed baseline.
**Effort:** 1 session.
**Hypothesis:** Moderate coupling. Seed selection is stable for top-3 but volatile for seeds 4–5, creating a noisy tail in graph expansion.

### Do misclassified edges persist indefinitely?

**What we know:** Sleep creates edges. Edge weight increases through co-retrieval reinforcement (+0.02 per co-useful retrieval). No mechanism removes or prunes edges.
**What we don't know:** The distribution of edge ages and last-update times. Whether old misclassified edges distort traversal.
**Experiment:** Compute edge age distribution, fraction never updated since creation, and correlation between edge age and co-retrieval utility.
**Effort:** 1 session.
**Hypothesis:** Significant fraction of edges are never reinforced after creation. Some actively hurt retrieval by connecting unrelated memories.

### Do memories converge to attractor states?

**What we know:** Memories accumulate feedback, access counts, shadow load, and edge connections over time.
**What we don't know:** Whether individual memories converge to a small number of stable states (always-retrieved, never-retrieved, summary-captured) — attractor dynamics that would make the feedback loop path-dependent.
**Experiment:** Plot per-memory retrieval frequency over time. Cluster trajectories. Look for convergence to fixed states.
**Effort:** 1–2 sessions.
**Hypothesis:** Three attractors exist. Once a memory enters the "never-retrieved" basin, it's very unlikely to escape without external intervention (manual feedback, sleep-driven theme enrichment).

### How does the system degrade at scale?

**What we know:** Tested at ~730 memories. Constants tuned at this scale. Graph has ~1,500 edges.
**What we don't know:** What happens at 2,000? 5,000? Whether Hebbian co-retrieval or PPR graph traversal become important at larger scales.
**Experiment:** Synthetic corpus expansion (duplicate with perturbation, controlled noise), or wait and measure as real corpus grows organically.
**Effort:** 2–3 sessions for synthetic; ongoing for organic.
**Hypothesis:** Hebbian becomes more important at scale (more co-retrieval signal). Adjacency expansion needs multi-hop (PPR) above ~2,000 memories where graph diameter increases.

### Event-time: highest-impact missing feature?

**What we know:** `architecture.md` calls this the highest-impact schema change in backlog. Memories have `created_at` but not `event_time`.
**What we don't know:** Quantified impact on temporal query retrieval.
**Experiment:** Annotate a subset of memories with event_time, add temporal filtering to recall, measure impact on temporal queries from GT.
**Effort:** 2–3 sessions.
**Hypothesis:** Large impact on temporal queries, negligible on everything else.

---

## Proposed experiments

Ordered by information value per effort, with concrete acceptance criteria.

### Tier 1: Unblocked by GT completion (1 session each)

1. **~~Re-tune constants on real GT.~~** *(Superseded by reranker.)* The hand-tuned formula was replaced by a learned LightGBM reranker that achieved +5.7% NDCG@5k over the formula in 5-fold CV. The tuning studies (wm24–wm34) produced the ground truth and evaluation infrastructure that made the reranker possible. Remaining reranker improvement experiments: raw scores as features (fts_bm25, vec_dist), query-dependent features, and LambdaRank (optimizes NDCG directly instead of pointwise MSE). See `architecture.md` § The reranker.

2. **Feedback loop health check.** *(Complete — answered by utility calibration study.)* Per-query Spearman r = 0.70 confirms feedback tracks relevance. See experiment #4 and `experiments.md` § Utility calibration study.

3. **Sleep impact measurement.** Snapshot GT metrics on 100-query subset → run sleep → re-judge → compare. Accept if: delta measured with confidence intervals.

4. **Utility calibration study.** *(Complete.)* Compared 13,396 feedback events against 13,317 GT judgments across 715 overlapping memories. Per-query Spearman r = 0.70 (feedback tracks relevance in context); per-memory r = 0.14 (aggregation destroys context signal). The empirical Bayes prior operates at the weak-correlation level. 16.4% of memories show inflated feedback (context-dependent, not self-reinforcing); 7.0% are coverage gaps. See `experiments.md` § Utility calibration study.

5. **Counterfactual coverage check.** For 20–30 GT queries, exhaustively judge all ~112 candidates (not just those the system retrieved). Measures how many relevant memories the retriever never surfaces — the selection bias that the standard GT can't detect. Accept if: unseen-relevant rate computed and documented.

6. **Replace cliff detector with agent-specified limit.** *(Complete.)* Score features cannot predict the cutoff (R² < 0). The cliff over-delivers 96.1% of the time. Implemented: `limit` parameter on `recall()` (default 5), cliff detection removed from scoring pipeline. Anchors {1, 3, 5, 8, 13} match the natural distribution (MAE = 0.71 with oracle selection vs 3.46 baseline). `recall_meta` events now log `limit` alongside `cutoff_rank` for prospective-vs-retrospective comparison. See roadmap § "Can cutoff history calibrate the cliff detector?"

7. **Reranker improvement: LambdaRank.** Train LGBMRanker with `lambdarank` objective to optimize NDCG directly instead of pointwise MSE. The biggest expected win — the current regressor minimizes squared error on relevance predictions, but ranking quality depends on ordering, not prediction accuracy. Accept if: NDCG@5k improves over pointwise regressor. Training infrastructure already exists (`--lambdarank` flag in `train_reranker.py`).

8. **Reranker improvement: query features.** Add query-dependent features (query token count, query specificity, category distribution of candidates). Currently the model sees each candidate independently; query-level features could help it calibrate predictions across different query types. Accept if: NDCG@5k improves, or negative result documented.

9. **UCB retuning reassessment.** The ESS-corrected UCB exploration bonus (from the limit-parameter branch) needs joint retuning of UCB_COEFF and EWMA_ALPHA. However, if the reranker is the production scoring path, UCB only matters for the formula fallback. Decide whether retuning is worth the effort. Accept if: decision documented with reasoning.

### Tier 2: Deeper investigation (2-3 sessions each)

10. **Query difficulty clustering.** Cluster 1,047 GT queries by type, measure per-cluster metrics, identify failure modes. Accept if: failure patterns documented in `experiments.md`.

11. **Generalization study.** Train/test split on GT (tune constants on half, evaluate on other half). Accept if: constants stable across splits (or instability documented as finding).

12. **Corpus scaling sensitivity.** Evaluate on 25%, 50%, 75%, 100% of memories, plot scaling curve. Accept if: curve plotted, PPR threshold identified.

13. **Counterfactual sleep evaluation.** Fork the DB, run one copy through sleep, keep the other frozen, compare retrieval metrics on the same query set. Isolates sleep's causal effect on retrieval quality. Accept if: delta measured with confidence intervals.

14. **Paraphrase robustness test.** Run GT queries through systematic paraphrasing (different abstraction levels, different vocabulary). Measures vocabulary co-adaptation — does the system only work well for habitual phrasings? Accept if: NDCG delta under paraphrase computed.

15. **Embedding staleness detection.** For each memory, compare theme set at embedding time vs. current. Flag high-drift memories, re-embed them, measure retrieval quality change. Quantifies the representation bifurcation problem. Accept if: drift distribution documented, re-embedding impact measured.

16. **Graph null models.** Compare current graph against similarity-only, edge-shuffled, edge-type-restricted, and temporal-only variants. Tests whether graph structure adds real signal or just correlates with similarity. Accept if: retrieval quality delta per variant measured.

### Tier 3: New capabilities (3+ sessions each)

17. **Event-time implementation + evaluation.** Schema change, LLM extraction at write time, temporal filtering in recall. Accept if: temporal queries measurably improve.

18. **PPR graph traversal.** Replace one-hop adjacency expansion with Personalized PageRank. Accept if: multi-hop queries measurably improve, or honest documentation of why PPR didn't help at current scale.

19. **Contradiction detection research.** The hardest problem. No clear experiment design yet — this is genuinely research-grade. Accept if: any measurable improvement over the 0.025–0.037 F1 baseline across surveyed systems.

20. **Resolution-fidelity evaluation.** Measure not just "was a relevant memory returned" but "was the required level of detail returned." Assesses whether summaries lose the specific details that made a memory useful — the detail-vs-compression tradeoff. Accept if: fidelity metric defined and measured.

21. **startup_load subsystem measurement.** Quantify how much apparent memory competence comes from preload vs. explicit retrieval. Tests the cache-vs-recall ambiguity by comparing sessions with and without preload. Accept if: preload contribution isolated and documented.

---

## Comparative benchmarking

`similar-systems.md` compares features. This section asks: where can we put numbers next to other systems?

### What's measurable now

- **Contradiction detection:** All systems 0.025–0.037 F1. Somnigraph's NREM catches adjacent contradictions only — likely in the same range. No formal measurement yet.
- **Graph traversal:** HippoRAG's BFS→PPR ablation shows Recall@2 drops from 40.9 to 25.4. Somnigraph uses novelty-scored one-hop expansion (better than BFS, likely worse than PPR). No direct comparison.
- **Latency:** Mem0 reports p95 1.44s. Somnigraph's latency is unmeasured but likely comparable (same underlying stack: SQLite + embedding API call).

### What's not measurable (and why)

- **No common personal memory retrieval benchmark.** LoCoMo is closest but tests conversation-level QA, not the memory→retrieval→feedback loop that somnigraph optimizes. Results aren't comparable.
- **No consolidation benchmark exists.** No system measures this — it's an open research problem (see `architecture.md` § Open Problems).
- **Feedback loop can't be compared.** No other system has one.

### Proposed benchmarking experiments

- **LoCoMo eval** with somnigraph's retrieval pipeline, for apples-to-apples comparison with Mem0's reported numbers. Effort: 1–2 sessions.
- **Contradiction detection rate** on the real corpus: manually annotate ~50 known contradictions, measure NREM's detection rate. Effort: 1 session.
- **Latency profiling:** p50/p95/p99 for `recall()` at current corpus size. Effort: trivial (instrument one session).

---

## What we'd tell someone starting from scratch

Not architecture — advice. The wrong turns we can help you skip.

1. **Start with BM25 keyword search.** Add vector only when BM25 misses semantic matches you care about. Hybrid is better, but BM25 alone gets you surprisingly far with zero infrastructure.

2. **Implement feedback from day one.** It's the dominant retrieval signal and needs time to accumulate. A system without feedback is guessing; a system with 50 feedback events per query is learning.

3. **Don't build a quality floor or cliff detector.** Both fixed thresholds and adaptive score-based heuristics fail — the cutoff is a content-level decision that scores can't capture (R² < 0). Let the agent specify how many results it wants. We tested quality floors (wm1, optimal ratio 0.0), cliff detection (over-delivers 96.1%), and score-feature prediction (anti-predictive) before reaching this conclusion.

4. **Contradiction handling is research-grade hard.** Don't assume you'll solve it. Plan for temporal evolution (`valid_from`/`valid_until`) as the practical workaround. Published F1 is catastrophic across all systems.

5. **Measure before you consolidate.** Without before/after retrieval metrics, you can't tell if your consolidation helps or hurts. We have this problem ourselves — it's the central unanswered question.

6. **Your metric choice is a design decision, not an objective fact.** AUC, MRR, and miss_rate disagree about what's optimal. Choose based on your use case. If you need fast first-hit (MRR), you'll sacrifice coverage (miss_rate). Know the tradeoff.

7. **Intuitive signals need empirical validation.** Three "obviously right" scoring components (quality floor, shadow penalty, confidence weighting) turned out to be useless or counterproductive. Test everything.

---

## Infrastructure gaps

What we can't measure yet and would need to build.

- **Continuous quality tracking.** GT is a one-time snapshot. No mechanism to measure retrieval quality drift as corpus grows and feedback accumulates. Need: periodic GT refresh on a stable query subset.
- **A/B testing harness.** Can't compare scoring variants on the same queries in real time. Need: dual-pipeline execution with statistical significance testing.
- **Sleep ablation framework.** Can't isolate which sleep step (NREM edges, REM summaries, consolidation archival) helps or hurts. Need: per-step before/after measurement.
- **Event log analysis toolkit.** 249K events, no query tools beyond raw SQL. Need: utilities for common queries ("which memories are never retrieved?", "feedback distribution by category", "retrieval latency percentiles").
- **Regression test suite.** Changing `scoring.py` has no automated quality check. Need: lightweight metric validation on a fixed query set, runnable as a pre-commit check.

---

*This document describes the research agenda as of March 2026. It should be updated as experiments complete and new questions emerge.*
