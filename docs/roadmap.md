# Research Roadmap

The forward agenda: what's still open and what's worth pursuing next. The settled lessons this doc used to restate now live in `architecture.md` (how the system works) and `experiments.md` (how we tested it) — this roadmap points there rather than keeping a second copy.

---

## What we learned

The settled lessons from 184 source analyses and 20+ tuning studies are canonized where the design and the evidence live. This section is a pointer, not a second copy — one home per fact. If you're building a memory system from scratch, these are the findings that would change your first decisions, each linked to its home:

| Finding | Canonical home |
|---------|----------------|
| **Feedback's dual role** — the dominant signal in the hand-tuned formula (`FEEDBACK_COEFF=5.15`), but near-zero contribution to the *learned reranker's* ranking; its real value is shaping the store over time (per-query GT correlation r=0.70). | `architecture.md` §§ Stage 1: Feedback boost, What feedback's role became; `experiments.md` § Utility calibration study |
| **Theme boost was compensating for missing graph traversal** — PPR (wm19) collapsed it 0.97→0.19; the high pre-PPR importance measured compensation, not intrinsic value. | `architecture.md` § Stage 2: Theme boost |
| **The parameter space has two basins** — feedback-dependent high-k (≈14) vs feedback-robust low-k (≈3); production chose Basin A (k=6) to leverage the feedback loop. | `experiments.md` § The two-basin discovery |
| **Metric choice drives parameter values** — AUC/MRR/miss_rate optimize to different constants; MRR trades off harshly with miss_rate (r=−0.517). You choose what "good retrieval" means. | `experiments.md` §§ Metrics: why AUC over single-threshold, Multi-objective optimization |
| **Intuitive scoring signals can be counterproductive** — quality floor (optimal ratio 0.0), shadow penalty, and confidence weight were all implemented, tested, and removed. | `architecture.md` § What didn't work |
| **Contradiction detection is universally catastrophic** — 0.025–0.037 F1 across all systems; a representation problem, not a tuning one. | `architecture.md` § Open problems |
| **Offline consolidation is viable but unmeasured** — the sleep pipeline runs and looks reasonable, but no metric proves it improves retrieval. The central open question. | `architecture.md` § Open problems; [Open questions](#open-questions) below |

The condensed advice version — the wrong turns you can skip — is [What we'd tell someone starting from scratch](#what-wed-tell-someone-starting-from-scratch) below.

---

## External review findings

Four independent reviewers (Gemini 2.5 Pro, Claude Sonnet 4.6, Claude Opus 4.6, ChatGPT Pro) blind-reviewed the architecture and experiment docs. Their concerns split by disposition.

**Now canonized** (linked to their homes, not duplicated here):

- **Feedback loop self-reinforcement** — selection bias, six-signal amplification, prior shape, `startup_load`/preload contamination, channel entanglement → `architecture.md` § Feedback loop self-reinforcement.
- **Representation bifurcation** — frozen embeddings drifting from sleep-evolved themes and summaries → `architecture.md` § Representation bifurcation.
- **Theme boost as compensation** → `architecture.md` § Stage 2: Theme boost.
- **PPR walking contradiction-flagged edges** — fixed in `scoring.py` (March 2026).
- **Bimodal utility prior, PPR seed-set stability, false-bridge ratchet / edge accumulation without pruning, detail-loss behind summaries** → tracked as [Open questions](#open-questions) and [Proposed experiments](#proposed-experiments) below.

**Still open, not yet tracked elsewhere:**

- **Sleep canonizing bias.** Sleep's editorial judgment (theme enrichment, summary generation, dormancy) may systematically favor certain memory types over others, but no measurement exists to detect it.
- **Category as the wrong decay axis.** Per-category decay assumes categories map to importance curves, but within-category variance likely exceeds between-category variance. Per-memory decay (already supported via the `decay_rate` column) may be the right granularity.
- **Vocabulary co-adaptation / exact-token recall collapse.** Queries and stored themes may co-evolve, so the system excels on habitual phrasings but degrades toward keyword-only search when a query exactly matches stored theme/summary tokens (FTS dominates, vector adds nothing). Partially probed by the paraphrase-robustness experiment below.

---

## Candidates from the external survey

The 2026-06-30 carsteneu code-level survey produced a ranked ledger of external mechanisms worth stealing — [`ideas-considered.md`](ideas-considered.md) (17 adopt-tier, 87 consider-tier, 33 note-only). Those are *inputs* to this agenda, not commitments: an adopt-tier candidate becomes a proposed experiment here, or a priority in `STEWARDSHIP.md`, only once it's ranked and taken up. See the **Tier 1 — Adopt candidates** shortlist there for the current external menu.

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

### Should memories carry source-event provenance?

**What we know:** Memories have `created_at`. Edges between memories carry `linking_context` and retrieval-shaped weights. There is no link from a memory to the session or sleep event that produced it. Sleep-created entries (entity refreshes, theme adjustments) are indistinguishable from session-created ones in their provenance.
**What we don't know:** Whether a `caused_by` field (event ID + type) would enable diagnostics current edges can't — e.g., distinguishing high-utility memories that trace back to feedback-driven sleep runs vs. direct writes, or detecting whether sleep-modified memories drift in feedback distribution.
**Experiment:** Add a lightweight `caused_by` field. Backfill conservatively for session and sleep events going forward (no historical reconstruction). After ~1 month of accumulation: compare feedback distributions across event types.
**Effort:** 1 session for schema + write-path; several months for the diagnostic queries to be interpretable.
**Hypothesis:** Provenance surfaces a real category that edges don't. Sleep-driven modifications have systematically different feedback profiles than direct stores. See `research/sources/connectome.md` — Connectome's `causedBy[]` is the analog at the record level.

### Should the DB itself be branchable?

**What we know:** Counterfactual sleep evaluation (#14) requires forking the DB to compare sleep-on vs sleep-off on the same query set — currently a manual `cp` of the SQLite file. Several other Tier 2 experiments (paraphrase robustness, generalization, embedding staleness, graph null models) would benefit from cheap variant comparison on the same memory set.
**What we don't know:** Whether SQLite write patterns admit cheap branching (copy-on-write at the page level, or versioned tables), and whether the resulting workflow lowers the friction of counterfactual studies enough to change which experiments are routine.
**Experiment:** Prototype a branch operation. Measure cost of (a) creating a branch, (b) running recall + sleep on it, (c) detecting divergence vs. parent. Test on counterfactual sleep evaluation as the first consumer.
**Effort:** 2–3 sessions for prototype; depends on whether SQLite carries it or a wrapper is needed.
**Hypothesis:** Cheap branching turns one-off counterfactual experiments into routine ones (different feedback histories, sleep runs, reranker versions). Above some corpus size, page-level COW becomes necessary, which is a larger lift. See `research/sources/connectome.md` — Chronicle's first-class branchable record store.

### Should sleep produce autobiographical narrative summaries?

**What we know:** NREM produces structured artifacts (theme normalization, edge weights, contradiction classifications, summary refresh). REM does probe recall and dormancy classification. Neither produces narrative — diary-entry-style summaries written by the agent in its own voice over recent activity. That role is currently filled outside the DB (the agent's seed.md and Personal Vault Journal — manual and hand-curated).
**What we don't know:** Whether narrative summaries as memories (`category="reflection"`, `decay_rate=0`, themes derived from content) surface differently than structured extractions on retrieval — particularly on cross-domain or thematic queries that structured memories handle poorly. Risk: hierarchical merge produces increasingly polished but less faithful narrative, the failure mode visible in Connectome's autobiographical strategy.
**Experiment:** Add a sleep mode that generates a narrative summary of recent session activity (e.g., the last 3 sessions' episodic memories) in the agent's voice. Test retrieval on the cross-domain GT queries identified by query difficulty clustering (#11). Hierarchical merge: when N narrative entries accumulate, merge into a longer-period entry. Track fidelity by comparing entries against source memories.
**Effort:** 2 sessions for implementation + initial evaluation; longer-running natural experiment as material accumulates.
**Hypothesis:** Narrative entries fill a different retrieval niche — "how was the project going around date X" type queries — that structured memories handle poorly. Hierarchical merge will show measurable fidelity drift; the question is whether the legibility gain offsets it. See `research/sources/connectome.md` — Connectome's AutobiographicalStrategy is the analog as a context-window strategy.

### Should the system surface recall hints before the agent asks?

Recall is entirely pull-based, which leaves a structural blind spot: the agent can't decide to recall what it doesn't know exists, so relevant memory is missed whenever the agent never thinks to ask. Counterfactual coverage (proposed experiment #5) measures unseen-relevant memories only for queries the agent *did* make; the queries never made are unmeasured. This is the system's main missing capability, not a tuning refinement.

The design is large enough to live on its own: see [`proposals/proactive-injection.md`](proposals/proactive-injection.md). In brief: a `UserPromptSubmit` hook runs cheap RRF-only retrieval each turn and, when a floor is cleared, injects a one-line hint (count + top score + a few topic handles, no snippets) that lets the agent decide whether to pull. A session cooldown suppresses repetition (anti-repetition, top of the exposure distribution) and stochastic Thompson gating over the per-memory Beta feedback model keeps the under-observed tail in play (anti-starvation), together containing the feedback self-reinforcement the design risks amplifying. The core assumption — that a coarse top-of-ranking surface floor carries usable signal even though the cliff detector found the fine-grained in-list cutoff anti-predictable (R² < 0) — is testable offline against the existing feedback logs before any code is written.

**Effort:** 1 session for the offline floor study (data exists, no new collection); 1-2 sessions for the hook delivery layer and write-back path if positive. **Negative result is publishable:** if the binary surface signal is as weak as the cliff cutoff, that establishes score-based gating fails at both granularities.

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

9. **FSI stability audit before 31-feature retrain.** Train 10–20 LightGBM models with different random seeds on the current 26-feature dataset, compute gain-based importance per feature across runs, flag features with high rank variance. First-mover bias (arXiv:2603.22346) shows that correlated features in gradient boosting create path-dependent importance concentration — whichever feature wins early splits gets a self-reinforcing advantage. Somnigraph's reranker has at least 4 correlated groups: fts_rank/fts_bm25, feedback_mean/ucb_bonus/fb_time_weighted, age_hours/hours_since_access/session_recency, and the pending fts_bm25_norm/vec_dist_norm pair. The R@10 vs. NDCG feature set disagreement documented in STEWARDSHIP (3 features backward-eliminated for NDCG were selected by R@10) may partly be a first-mover artifact rather than genuine metric disagreement — the FSI audit can disambiguate. Accept if: per-feature stability scores computed, correlated groups identified, findings inform whether the 31-feature retrain needs seed averaging or independent ensemble. See `research/sources/first-mover-bias.md`. Effort: low (1–2 hours, no new dependencies). **2026-05-08 update: empirical evidence collected via `scripts/audit_reranker_pathology.py` — content-residual audit shows the model splits behavior by query-shape (summary-shape uses channel signals, OOD-shape falls back to age_days + fb_count). SHAP analysis on top offenders confirms fb_count and age_days dominate the demotion signal for cold-start memories even after the NaN sentinel fix. The seed-variance question remains open but the broader finding — feature interactions encode self-reinforcement bias on multiple surfaces — extends what FSI alone would have shown.**

10. **UCB retuning reassessment.** The ESS-corrected UCB exploration bonus (from the limit-parameter branch) needs joint retuning of UCB_COEFF and EWMA_ALPHA. However, if the reranker is the production scoring path, UCB only matters for the formula fallback. Decide whether retuning is worth the effort. Accept if: decision documented with reasoning.

10. **Prospective indexing.** At write time (or during sleep REM), generate 2–3 hypothetical future recall queries for each memory and append them to the enriched text before embedding. This bridges the cue-trigger semantic disconnect — the reason queries fail when the question uses different language than the stored memory. Kumiho (arXiv:2603.17244) reports this eliminated the >6-month accuracy cliff on LoCoMo-Plus (37.5% → 84.4%), with independent partial reproduction by the benchmark authors. HyperMem (arXiv:2604.08256) independently validates the approach: their `potential` field (anticipated query patterns per fact) is a core feature of the system that achieves 92.73% LoCoMo — convergent evidence from two independent systems. The cost is one LLM call per memory — already the pattern for Somnigraph's REM classification. Accept if: NDCG@5k improves on GT queries, or negative result documented with analysis of why the enriched embedding already captures this. See `research/sources/kumiho.md`, `research/sources/hypermem.md`.

21. **Expansion method ablation (LoCoMo).** 3 of 6 expansion methods are nearly dead: rocchio (0% fire rate), multi_query (2%), entity_focus (4%). Session (100%), keyword (95%), entity_bridge (96%) do all the work. Run `--expand-all` with only the 3 active methods vs all 6 to confirm the dead methods add no signal, then remove them. Also test individual method contribution by dropping one active method at a time. Accept if: per-method R@10 delta measured, dead methods confirmed removable or revived.

22. **Effective dimensionality audit.** Compute PCA participation ratio on the production embedding matrix (text-embedding-3-small, 384-dim): d_eff = (sum lambda_i)^2 / sum(lambda_i^2). The Geometry of Forgetting (arXiv:2604.06222) finds production embeddings concentrate variance in ~16 effective dimensions regardless of nominal count, placing them in the interference-vulnerable regime. If our d_eff is similar, it quantifies how much vector recall degrades with scale and validates that hybrid retrieval (BM25 + reranker) is necessary, not optional. Accept if: d_eff computed and documented. Effort: trivial (one-off script, minutes). See `research/sources/geometry-of-forgetting.md`.

23. **User-query bias correction.** Prepend a producer-role signal (e.g., "user:") to search queries to bias retrieval toward user-generated content. MemMachine (arXiv:2604.04853) reports +1.4% on LongMemEvalS from this alone. In Somnigraph's context, most memories are stored from user-side content, so the signal may be weaker — but it's trivial to test. Accept if: NDCG delta measured on GT queries. Effort: trivial (one-line change + evaluation run). See `research/sources/memmachine.md`.

24. **Real-recall queries as bundled-crafter hard-mode prompt examples.** The bundled crafter's `CRAFT_TEMPLATE_BUNDLED` currently teaches the LLM what hard queries look like via synthetic examples (vocabulary stripped, mechanism vs symptom, etc.). The real-recall pathology miner (`scripts/select_real_pathology_targets.py`) now exposes `_pathology_sample_query` per memory — actual production queries that buried the rated-useful target at rank ≥ 5. Replacing or augmenting the synthetic hard-mode examples with these would teach the LLM the real distribution of hard queries it should generate, instead of an a-priori distribution. Mix 2-3 real-recall queries (across different memories, anonymized if needed) into the prompt's hard-mode examples; re-run a small probe; spot-check `angle:` quality on the hard channel. Accept if: hard-mode probe target ranks improve vs synthetic-example baseline, or null result documented. Defer until V5+3 lands so the change is attributable. Effort: low (prompt change + smoke). Source: handoff-v5plus2 bonus #4 (deleted after lift).

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

Where Somnigraph has numbers next to other systems — LoCoMo retrieval and end-to-end QA (done), plus PERMA, AMB, and the measurable-but-unmeasured gaps (proposed) — now lives in [`benchmarks.md`](benchmarks.md). `similar-systems.md` compares features; `benchmarks.md` compares scores; this roadmap tracks only what is *next*.

The forward benchmarking work: the [proposed experiments](benchmarks.md#proposed-benchmarking-experiments) (contradiction-detection rate, latency profiling), the PERMA ingest pipeline, and the LoCoMo ablations (expansion methods #21, sleep, feedback) that would let `experiments.md` document the retrieval arc end to end.

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
