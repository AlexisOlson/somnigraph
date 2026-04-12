# Research Roadmap

What we've learned, what's still open, and what's worth pursuing next. Read alongside `architecture.md` (how the system works) and `experiments.md` (how we tested it).

---

## What we learned

Seven findings from 93 source analyses and 20+ tuning studies that would change your first decisions if you were building a memory system from scratch.

### 1. Feedback is the system

The retrieval feedback loop — where recalled memories receive explicit utility scores that reshape future scoring — is the dominant signal. Feature importance analysis (wm15): feedback accounts for 15–20% of AUC, but with a catch: it's only useful because the empirical Bayes Beta prior centers the scores. wm5 tested removal of the prior: `feedback_coeff` collapsed to near zero. Raw feedback is too noisy to help; the prior is the mechanism that makes it useful.

As of v0.5.0, Ori-Mnemos also closes a feedback loop — but via behavioral inference (inferred from citations, edits, re-recalls) rather than explicit grading. No other surveyed system has explicit per-query feedback with measured GT correlation (r=0.70). This remains the primary architectural differentiator, though the category is no longer unique.

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
**Possible mechanism:** Turrigiano homeostatic scaling (Ori-Mnemos v0.5.0) — clamp per-node mean edge weight to a target by scaling all edges touching that node. This prevents accumulation without explicit pruning. See `research/sources/ori-mnemos.md` §9.1.

### Do memories converge to attractor states?

**What we know:** Memories accumulate feedback, access counts, shadow load, and edge connections over time.
**What we don't know:** Whether individual memories converge to a small number of stable states (always-retrieved, never-retrieved, summary-captured) — attractor dynamics that would make the feedback loop path-dependent.
**Experiment:** Plot per-memory retrieval frequency over time. Cluster trajectories. Look for convergence to fixed states.
**Effort:** 1–2 sessions.
**Hypothesis:** Three attractors exist. Once a memory enters the "never-retrieved" basin, it's very unlikely to escape without external intervention (manual feedback, sleep-driven theme enrichment).

### How does the system degrade at scale?

**What we know:** Tested at ~730 memories. Constants tuned at this scale. Graph has ~1,500 edges.
**What we don't know:** What happens at 2,000? 5,000? Whether Hebbian co-retrieval or PPR graph traversal become important at larger scales.
**Theoretical framework:** The Geometry of Forgetting (arXiv:2604.06222) predicts power-law degradation from interference in embedding space, gated by effective dimensionality. At d_eff ~16 (typical for production embeddings), retrieval accuracy degrades as a power law of database size due to angular crowding. Hybrid retrieval (BM25 + reranker) mitigates this since BM25 operates in a different space, but pure vector recall will degrade. The d_eff audit (experiment #22) provides the empirical input for this prediction.
**Experiment:** Synthetic corpus expansion (duplicate with perturbation, controlled noise), or wait and measure as real corpus grows organically.
**Effort:** 2–3 sessions for synthetic; ongoing for organic.
**Hypothesis:** Hebbian becomes more important at scale (more co-retrieval signal). Adjacency expansion needs multi-hop (PPR) above ~2,000 memories where graph diameter increases. Vector recall specifically will degrade per interference theory; BM25 and graph channels should be more robust.

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

1. **~~Re-tune constants on real GT.~~** *(Superseded by reranker.)* The hand-tuned formula was replaced by a learned LightGBM reranker that achieved +5.7% NDCG@5k over the formula in 5-fold CV. The tuning studies (wm24–wm34) produced the ground truth and evaluation infrastructure that made the reranker possible. All three improvement experiments are now complete: LambdaRank (#7, parity), query features (#8, 8 features integrated), raw-score features (fts_bm25_norm, vec_dist_norm added in 31-feature batch). See `architecture.md` § The reranker.

2. **Feedback loop health check.** *(Complete — answered by utility calibration study.)* Per-query Spearman r = 0.70 confirms feedback tracks relevance. See experiment #4 and `experiments.md` § Utility calibration study.

3. **Sleep impact measurement.** Snapshot GT metrics on 100-query subset → run sleep → re-judge → compare. Accept if: delta measured with confidence intervals.

4. **Utility calibration study.** *(Complete.)* Compared 13,396 feedback events against 13,317 GT judgments across 715 overlapping memories. Per-query Spearman r = 0.70 (feedback tracks relevance in context); per-memory r = 0.14 (aggregation destroys context signal). The empirical Bayes prior operates at the weak-correlation level. 16.4% of memories show inflated feedback (context-dependent, not self-reinforcing); 7.0% are coverage gaps. See `experiments.md` § Utility calibration study.

5. **Counterfactual coverage check.** For 20–30 GT queries, exhaustively judge all ~112 candidates (not just those the system retrieved). Measures how many relevant memories the retriever never surfaces — the selection bias that the standard GT can't detect. Accept if: unseen-relevant rate computed and documented.

6. **Replace cliff detector with agent-specified limit.** *(Complete.)* Score features cannot predict the cutoff (R² < 0). The cliff over-delivers 96.1% of the time. Implemented: `limit` parameter on `recall()` (default 5), cliff detection removed from scoring pipeline. Anchors {1, 3, 5, 8, 13} match the natural distribution (MAE = 0.71 with oracle selection vs 3.46 baseline). `recall_meta` events now log `limit` alongside `cutoff_rank` for prospective-vs-retrospective comparison. See roadmap § "Can cutoff history calibrate the cliff detector?"

7. **~~Reranker improvement: LambdaRank.~~** *(Parity, not improvement.)* Initial negative result (-1.5pp) had two fixable causes: GT-only training pool and coarse quantile bins. With full candidate pool (`--neg-ratio 0`) and 100-level direct scaling (`--lr-levels 100`), LambdaRank reaches parity (0.8099 vs pointwise 0.8127 on 500q). Two-stage (MSE->LambdaRank) also ties at 0.8119. At 500-849 queries, the objective doesn't differentiate. Pointwise remains the production model. See `experiments.md` § Improvement experiments.

8. **~~Reranker improvement: query features.~~** *(Complete.)* 8 query-dependent features added (query_coverage, proximity, query_idf_var, burstiness, betweenness, diversity_score, fb_time_weighted, session_recency). Integrated during production unification (2026-03-20). query_idf_var is the top new feature by importance. 5 more features added in 31-feature batch (query_length, candidate_pool_size, fts_bm25_norm, vec_dist_norm, decay_rate), pending retrain.

9. **FSI stability audit before 31-feature retrain.** Train 10–20 LightGBM models with different random seeds on the current 26-feature dataset, compute gain-based importance per feature across runs, flag features with high rank variance. First-mover bias (arXiv:2603.22346) shows that correlated features in gradient boosting create path-dependent importance concentration — whichever feature wins early splits gets a self-reinforcing advantage. Somnigraph's reranker has at least 4 correlated groups: fts_rank/fts_bm25, feedback_mean/ucb_bonus/fb_time_weighted, age_hours/hours_since_access/session_recency, and the pending fts_bm25_norm/vec_dist_norm pair. The R@10 vs. NDCG feature set disagreement documented in STEWARDSHIP (3 features backward-eliminated for NDCG were selected by R@10) may partly be a first-mover artifact rather than genuine metric disagreement — the FSI audit can disambiguate. Accept if: per-feature stability scores computed, correlated groups identified, findings inform whether the 31-feature retrain needs seed averaging or independent ensemble. See `research/sources/first-mover-bias.md`. Effort: low (1–2 hours, no new dependencies).

10. **UCB retuning reassessment.** The ESS-corrected UCB exploration bonus (from the limit-parameter branch) needs joint retuning of UCB_COEFF and EWMA_ALPHA. However, if the reranker is the production scoring path, UCB only matters for the formula fallback. Decide whether retuning is worth the effort. Accept if: decision documented with reasoning.

10. **Prospective indexing.** At write time (or during sleep REM), generate 2–3 hypothetical future recall queries for each memory and append them to the enriched text before embedding. This bridges the cue-trigger semantic disconnect — the reason queries fail when the question uses different language than the stored memory. Kumiho (arXiv:2603.17244) reports this eliminated the >6-month accuracy cliff on LoCoMo-Plus (37.5% → 84.4%), with independent partial reproduction by the benchmark authors. HyperMem (arXiv:2604.08256) independently validates the approach: their `potential` field (anticipated query patterns per fact) is a core feature of the system that achieves 92.73% LoCoMo — convergent evidence from two independent systems. The cost is one LLM call per memory — already the pattern for Somnigraph's REM classification. Accept if: NDCG@5k improves on GT queries, or negative result documented with analysis of why the enriched embedding already captures this. See `research/sources/kumiho.md`, `research/sources/hypermem.md`.

21. **Expansion method ablation (LoCoMo).** 3 of 6 expansion methods are nearly dead: rocchio (0% fire rate), multi_query (2%), entity_focus (4%). Session (100%), keyword (95%), entity_bridge (96%) do all the work. Run `--expand-all` with only the 3 active methods vs all 6 to confirm the dead methods add no signal, then remove them. Also test individual method contribution by dropping one active method at a time. Accept if: per-method R@10 delta measured, dead methods confirmed removable or revived.

22. **Effective dimensionality audit.** Compute PCA participation ratio on the production embedding matrix (text-embedding-3-small, 384-dim): d_eff = (sum lambda_i)^2 / sum(lambda_i^2). The Geometry of Forgetting (arXiv:2604.06222) finds production embeddings concentrate variance in ~16 effective dimensions regardless of nominal count, placing them in the interference-vulnerable regime. If our d_eff is similar, it quantifies how much vector recall degrades with scale and validates that hybrid retrieval (BM25 + reranker) is necessary, not optional. Accept if: d_eff computed and documented. Effort: trivial (one-off script, minutes). See `research/sources/geometry-of-forgetting.md`.

23. **User-query bias correction.** Prepend a producer-role signal (e.g., "user:") to search queries to bias retrieval toward user-generated content. MemMachine (arXiv:2604.04853) reports +1.4% on LongMemEvalS from this alone. In Somnigraph's context, most memories are stored from user-side content, so the signal may be weaker — but it's trivial to test. Accept if: NDCG delta measured on GT queries. Effort: trivial (one-line change + evaluation run). See `research/sources/memmachine.md`.

### Tier 2: Deeper investigation (2-3 sessions each)

11. **Query difficulty clustering.** Cluster 1,047 GT queries by type, measure per-cluster metrics, identify failure modes. Accept if: failure patterns documented in `experiments.md`.

12. **Generalization study.** Train/test split on GT (tune constants on half, evaluate on other half). Accept if: constants stable across splits (or instability documented as finding).

13. **Corpus scaling sensitivity.** Evaluate on 25%, 50%, 75%, 100% of memories, plot scaling curve. Accept if: curve plotted, PPR threshold identified.

14. **Counterfactual sleep evaluation.** Fork the DB, run one copy through sleep, keep the other frozen, compare retrieval metrics on the same query set. Isolates sleep's causal effect on retrieval quality. Accept if: delta measured with confidence intervals.

15. **Paraphrase robustness test.** Run GT queries through systematic paraphrasing (different abstraction levels, different vocabulary). Measures vocabulary co-adaptation — does the system only work well for habitual phrasings? Accept if: NDCG delta under paraphrase computed.

16. **Embedding staleness detection.** For each memory, compare theme set at embedding time vs. current. Flag high-drift memories, re-embed them, measure retrieval quality change. Quantifies the representation bifurcation problem. Accept if: drift distribution documented, re-embedding impact measured.

17. **Graph null models.** Compare current graph against similarity-only, edge-shuffled, edge-type-restricted, and temporal-only variants. Tests whether graph structure adds real signal or just correlates with similarity. Accept if: retrieval quality delta per variant measured.

### Tier 3: New capabilities (3+ sessions each)

18. **Event-time implementation + evaluation.** Schema change, LLM extraction at write time, temporal filtering in recall. Kumiho's event extraction (structured events with consequences appended to summaries at ingestion) provides immediate causal structure without waiting for sleep — worth considering as a write-time enrichment rather than deferring to sleep. Accept if: temporal queries measurably improve.

19. **PPR graph traversal.** Replace one-hop adjacency expansion with Personalized PageRank. Accept if: multi-hop queries measurably improve, or honest documentation of why PPR didn't help at current scale.

20. **Contradiction detection research.** The hardest problem. Three complementary approaches now identified:

    **(a) Write-path: RECONCILE temporal narratives** (TSUBASA, arXiv:2604.07894, ACL 2026). When new info contradicts existing memory, merge into a temporal narrative ("On [Date1], X did Y. On [Date2], X now does Z") instead of destructive DELETE. Preserves preference evolution. Lower bar than detection — the memory manager decides during write, not during retrieval.

    **(b) Read-path: contradiction surfacing** (Rashomon Memory, arXiv:2604.03588). When top-K results are connected by contradiction edges, annotate the response with the structured disagreement rather than forcing resolution. Somnigraph already has contradiction edges from sleep — this is a post-processing step in `impl_recall()`. Lowest effort of the three.

    **(c) Write-path: reconciliation-as-enrichment** (arXiv:2603.22735). Generate reconciliatory explanations during NREM sleep and store as `linking_context` on contradiction edges. Hardest (best model 40.25% success), but right framing for personal memory.

    The NLI cross-encoder roadmap item (for detecting contradictions in the first place) remains a prerequisite for (c). Approaches (a) and (b) can work with existing sleep-detected contradiction edges. Accept if: any measurable improvement over the 0.025–0.037 F1 baseline, or reconciliation/surfacing quality evaluated on manually identified contradiction pairs. See `research/sources/contradiction-reconciliation.md`, `research/sources/tsubasa.md`, `research/sources/rashomon-memory.md`.

21. **Resolution-fidelity evaluation.** Measure not just "was a relevant memory returned" but "was the required level of detail returned." Assesses whether summaries lose the specific details that made a memory useful — the detail-vs-compression tradeoff. Accept if: fidelity metric defined and measured.

22. **startup_load subsystem measurement.** Quantify how much apparent memory competence comes from preload vs. explicit retrieval. Tests the cache-vs-recall ambiguity by comparing sessions with and without preload. Accept if: preload contribution isolated and documented.

---

## Comparative benchmarking

`similar-systems.md` compares features. This section tracks where we have numbers next to other systems.

### LoCoMo end-to-end QA *(Complete)*

Built and run. **85.1% overall accuracy** (Opus judge), beating Mem0 (66.88 J), Mem0g (68.44 J), and full-context baseline (72.90 J). See `docs/locomo-benchmark.md` for full results, per-category breakdown, and reference tables from the Mem0 paper.

Pipeline: ingest LoCoMo conversations → recall with LoCoMo-specific reranker (12 features, forward stepwise) → GPT-4.1-mini reader → LLM judge. Ported from RedPlanet CORE's benchmark harness. All 10 conversations, 1540 non-adversarial questions.

**Known limitations:** LoCoMo's GT has a 6.4% error rate (corrected GT vendored from locomo-audit). The LLM judge is generous — Opus is 3.2pp stricter than GPT-4.1-mini. GPT-5.4-mini (reasoning model) is -6.6pp worse as reader because it computes dates instead of quoting context.

**Remaining:** Rerun with corrected GT, sleep/feedback ablations to isolate contributions.

### PERMA personalization benchmark *(Proposed)*

PERMA (arXiv:2603.23231, March 2026) evaluates personalized memory agents on preference-state maintenance across event-driven, multi-session, multi-domain interactions. 10 synthetic users, 20 domains, 2,166 preference details, 1.8M tokens. Decoupled evaluation: memory fidelity (BERT-F1, Memory Score 1-4) measured separately from task performance (MCQ accuracy, interactive Turn=1/Turn<=2). See `research/sources/perma.md` for full analysis.

**Why it matters:** PERMA's hardest dimension — cross-domain synthesis — is where all benchmarked systems collapse (MemOS Turn=1 drops from 0.548 to 0.306). Somnigraph's graph-based retrieval (PPR, NREM edges, Hebbian co-retrieval) is architecturally positioned for exactly this, but untested. Fresh benchmark (March 2026) with no established SOTA beyond the paper's baselines.

**SOTA targets to beat (memory systems only, Clean single-domain):**

| Metric | Current SOTA | System | Notes |
|--------|-------------|--------|-------|
| MCQ Accuracy | 0.811 | MemOS | LLMs reach 0.882 (Kimi-K2.5) with full context |
| BERT-F1 | 0.859 | RAG (BGE-M3) | Raw retrieval completeness |
| Memory Score | 2.27 / 4.0 | MemOS | LLM-judged coverage + accuracy + noise |
| Turn=1 (single-domain) | 0.548 | MemOS | One-shot interactive success |
| Turn<=2 (single-domain) | 0.830 | Memobase | Success with one correction round |
| Turn=1 (multi-domain) | 0.306 | MemOS | The hard problem — cross-domain synthesis |
| Completion | 0.846 | EverMemOS | Task completion rate |

**SOTA targets (Noisy):**

| Metric | Current SOTA | System |
|--------|-------------|--------|
| MCQ Accuracy | 0.794 | MemOS |
| Turn=1 | 0.524 | MemOS |

**Primary goal: multi-domain Turn=1 (0.306).** This is where every system collapses and where graph-conditioned retrieval should differentiate. A strong result here (0.45+) would be the headline claim. All other metrics are secondary targets — we'd like to beat them all, but cross-domain synthesis is the story.

**Somnigraph advantages for this benchmark:**
- Learned reranker (vs. fixed top-k retrieval used by all benchmarked systems) — directly addresses the finding that expanding from top-10 to top-20 hurts most systems
- PPR graph traversal for cross-domain bridging — no benchmarked system has graph-conditioned retrieval
- Feedback loop — no benchmarked system adapts retrieval based on utility signals
- Sleep-based consolidation — category-aware memory organization that MemOS achieves via write-time extraction

**Pipeline requirements:** Ingest PERMA's event-driven dialogue sessions → build memory store → answer MCQ and interactive evaluation at Type 1/2/3 checkpoints. All memory systems in the paper use GPT-4o-mini as backbone. Code: https://github.com/PolarisBoy1/PERMA.

**Effort:** 2-3 sessions. Session 1: ingest pipeline + MCQ evaluation. Session 2: interactive evaluation + analysis. Session 3: ablations (graph contribution, feedback contribution, sleep contribution).

### AMB LoCoMo cross-evaluation *(Idea)*

AMB (Agent Memory Benchmark, Vectorize.io) is a meta-benchmark harness wrapping 7 datasets including LoCoMo. Their LoCoMo adapter uses Gemini judge/reader with "generous grading" prompts. Hindsight claims 92.0% on LoCoMo via AMB; we score 85.1% (Opus judge). Writing a Somnigraph adapter for AMB and running just the LoCoMo split would give an apples-to-apples comparison under identical generation/judging conditions — isolating retrieval quality from reader/judge differences. The adapter interface (`ingest` + `retrieve`) maps cleanly to our `remember` + `recall`. See `research/sources/amb.md` for full analysis, including conflict-of-interest concerns (Hindsight's adapter is heavily tuned vs. generic baselines).

**Effort:** 1-2 sessions. Adapter + one evaluation run.

### What's measurable but unmeasured

- **Contradiction detection:** All systems 0.025–0.037 F1. Somnigraph's NREM catches adjacent contradictions only — likely in the same range. No formal measurement yet.
- **Latency:** Mem0 reports p95 1.44s. Somnigraph's latency is unmeasured but likely comparable (same underlying stack: SQLite + embedding API call).
- **Consolidation:** No system benchmarks this. It's an open research problem (see `architecture.md` § Open Problems).

### Proposed benchmarking experiments

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
- **Consolidation safety guards.** The sleep pipeline has no circuit breaker, no dry-run mode, no protection for high-confidence memories. As the system matures, automated consolidation needs safeguards: cap the fraction of memories archived per cycle, protect high-confidence/high-access memories from automated deprecation, preview consolidation actions before committing. Kumiho's Dream State (arXiv:2603.17244) implements all three with configurable thresholds. Need: circuit breaker (max archive ratio per sleep run), pinned-memory protection, dry-run mode for sleep pipeline.
- **Snippet and prompt eval loop.** The CLAUDE.md snippet (Tier 1 instructions) and sleep pipeline prompts (NREM/REM) are tested manually. Anthropic's [Skill Creator](https://github.com/anthropics/skills) provides an agentic eval loop (executor/grader/comparator/analyzer agents, train/test split, iterative refinement) that could automate this. Write eval cases like "does a fresh Claude using only the snippet correctly call recall() with dual-input?" or "does NREM correctly classify this memory pair as contradicting?" and iterate prompts to 100% pass rate. Discovered via r/aimemory (u/m3umax uses 36 eval tests for their memory system instructions).

---

*This document describes the research agenda as of March 2026. It should be updated as experiments complete and new questions emerge.*
