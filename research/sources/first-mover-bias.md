# First-Mover Bias in Gradient Boosting Explanations: Mechanism, Detection, and Resolution — Analysis

*Generated 2026-03-24 by Opus 4.6 agent reading arXiv:2603.22346*

---

## Paper Overview

**Citation:** Drake Caraker, Bryan Arnold, David Rhoads. "First-Mover Bias in Gradient Boosting Explanations: Mechanism, Detection, and Resolution." arXiv:2603.22346, March 2026. 28 pages (main text + appendices). Code: https://github.com/DrakeCaraker/dash-shap

**Problem addressed:** SHAP-based feature importance rankings from gradient-boosted trees are unstable under multicollinearity — retraining the same model with different random seeds can produce substantially different importance rankings without changing predictive performance. The paper identifies a specific mechanism (sequential residual dependency) and proposes both a diagnostic (FSI) and a resolution (DASH).

**Core claim:** When correlated features compete for early splits in gradient boosting, whichever feature is selected first gains a self-reinforcing residual advantage across subsequent trees. This *first-mover bias* concentrates SHAP importance on arbitrary features within correlated groups, and the effect worsens with more trees. The key insight is that *model independence* — training and explaining multiple independent models rather than one sequential model — is sufficient to largely neutralize the bias.

**Scale of evaluation:** Synthetic data (N=5000, P=50, 10 correlated groups, swept rho in {0, 0.5, 0.7, 0.9, 0.95}, 20 repetitions per level); three real-world datasets (Breast Cancer Wisconsin 30 features, Superconductor 81 features, California Housing 8 features); 9 methods compared; 4 metrics (stability, accuracy, equity, RMSE). Pre-specified 11 success criteria, all passed.

---

## Architecture / Method

### The mechanism: sequential residual dependency

The core insight is elegantly simple. In gradient boosting, model f is constructed as f = sum(h_t) where each tree h_t fits the residuals r_t = y - sum(h_s for s < t). If tree h_1 splits on feature j from a correlated group G_l, it partially removes j's signal from the residuals. Since correlated feature k in G_l carries overlapping signal, k's marginal gain for r_2 is also reduced — but j retains a slight advantage from its own partial fit. Over T iterations, this creates a path-dependent concentration of splits on whichever feature happened to be selected first.

The effect is *probabilistic, not deterministic* — with `colsample_bytree` between 0.1-0.5, correlated features aren't always co-present in the same column sample. But whichever feature accumulates slightly more early selections gains a cumulative residual advantage that gets amplified over thousands of trees.

Key prediction: the concentration increases with T (tree count) and rho (within-group correlation). The paper tests this indirectly via the Large Single Model comparison (~15,000 sequential trees, worst stability of any method).

### DASH: Diversified Aggregation of SHAP

A five-stage pipeline:

1. **Population Generation.** Train M=200 XGBoost models with randomly sampled hyperparameters, critically including low `colsample_bytree` (0.1-0.5) to force feature diversity.

2. **Performance Filtering.** Retain models within epsilon of the best validation score. Absolute mode (epsilon=0.08) for synthetic, relative mode (epsilon=0.05, top 5%) for real-world.

3. **Diversity Selection.** Greedy max-min dissimilarity on L2-normalized gain-based importance vectors, selecting K <= 30 models. Alternative: deduplication (remove pairs with Spearman rho > 0.95).

4. **Consensus SHAP.** Compute interventional TreeSHAP for each selected model on a held-out explain set (disjoint from train/val/test), average element-wise:

   Phi_bar = (1/|S|) * sum(Phi^(i) for i in S)

5. **Stability Diagnostics.** Two tools:
   - **Feature Stability Index (FSI):** FSI_j = sigma_bar_j / (I_bar_j + epsilon_0), where sigma_bar_j is the mean (across observations) of std (across models) of SHAP values, and I_bar_j is the consensus global importance. High FSI = unstable attribution.
   - **Importance-Stability (IS) Plot:** Scatter of (I_bar_j, FSI_j) partitioned into four quadrants by median thresholds. Quadrant II (high importance, high FSI) flags collinear cluster members whose individual rankings shouldn't be trusted.

### Design choices and justifications

- **Why XGBoost population, not a single ensemble?** A single large ensemble maximizes sequential dependency — the Large Single Model produces the *worst* stability of any method tested.
- **Why interventional TreeSHAP?** It conditions on marginal rather than conditional feature distributions. Under high correlation this evaluates out-of-distribution feature combinations (a known limitation), but averaging across diverse models mitigates individual-model artifacts.
- **Why gain-based importance for diversity selection?** Faster than SHAP, sufficient for measuring which features each model relies on. SHAP is only computed for the selected K models.

### Acknowledged limitations

- Targets XGBoost + TreeSHAP; extension to other model families requires adapting diversity selection
- Interventional SHAP under high correlation evaluates OOD feature combinations
- Current pipeline averages main effects only, not interaction tensors
- Nonlinear DGP reduces overall stability for all methods; DASH's advantage grows with rho but is smaller at low correlation
- N_reps=20 may be underpowered for the DASH vs. Stochastic Retrain comparison

### Missed limitations

- The paper doesn't discuss LightGBM at all — only XGBoost. While the mechanism is structurally identical (both are sequential gradient boosters), the default hyperparameters differ (LightGBM's leaf-wise growth vs. XGBoost's level-wise growth), and the paper doesn't test whether the effect magnitude transfers.
- No discussion of pointwise models (regression targets) vs. ranking models (LambdaRank). First-mover bias operates on individual trees, which are shared between objectives, but the residual structure differs.

---

## Key Claims & Evidence

### Table 3: Main mechanism experiment (linear DGP, 20 reps per rho)

| rho | Method              | Stability     | DGP Agree | Equity (CV) | RMSE  |
|-----|---------------------|---------------|-----------|-------------|-------|
| 0.0 | Single Best         | .973 +/- .001 | .985      | .168        | .609  |
| 0.0 | Large Single Model  | .953 +/- .003 | .974      | .170        | .771  |
| 0.0 | Stoch. Retrain      | .975 +/- .001 | .987      | .175        | .581  |
| 0.0 | DASH (MaxMin)       | .972 +/- .002 | .985      | .164        | .596  |
| 0.9 | Single Best         | .958 +/- .003 | .978      | .224        | .614  |
| 0.9 | Large Single Model  | .938 +/- .003 | .967      | .262        | .738  |
| 0.9 | Stoch. Retrain      | .977 +/- .002 | .988      | .182        | .577  |
| 0.9 | DASH (MaxMin)       | .977 +/- .001 | .988      | .176        | .594  |
| 0.95| Single Best         | .951 +/- .004 | .975      | .246        | .608  |
| 0.95| Large Single Model  | .925 +/- .003 | .961      | .284        | .733  |
| 0.95| Stoch. Retrain      | .979 +/- .002 | .989      | .170        | .576  |
| 0.95| DASH (MaxMin)       | .977 +/- .001 | .988      | .172        | .590  |

Key patterns:
- LSM stability degrades monotonically with rho (0.953 -> 0.925), supporting the sequential dependency hypothesis.
- DASH and Stochastic Retrain are effectively flat across all rho levels (~0.977).
- DASH and SR achieve *identical* stability (0.977 at rho=0.9), despite differing in every design choice except model independence.

### Table 4: All methods at rho=0.9

Methods partition cleanly into two tiers:
- **Dependent** (Single Best, LSM, Ensemble SHAP): stability 0.938-0.964
- **Independent** (DASH, Stochastic Retrain, Random Selection, Naive Top-N): stability ~0.977

The between-tier gap (~0.01-0.04) dwarfs the within-tier gap (~0.001).

### Table 6: Real-world datasets

| Dataset         | Method               | Stability      | Delta vs SB(M=200) | RMSE          |
|-----------------|---------------------|---------------|---------------------|---------------|
| Breast Cancer   | Single Best (M=200) | .317 +/- .053 | —                   | —             |
| Breast Cancer   | DASH (MaxMin)       | .930 +/- .005 | +0.613              | —             |
| Superconductor  | Single Best          | .830 +/- .020 | —                   | 9.18 +/- 0.11 |
| Superconductor  | DASH (MaxMin)       | .962 +/- .010 | +0.132              | 9.15 +/- 0.08 |
| Calif. Housing  | Single Best          | .967 +/- .010 | —                   | 0.460 +/- .009|
| Calif. Housing  | DASH (MaxMin)       | .982 +/- .005 | +0.015              | 0.450 +/- .005|

The Breast Cancer result is dramatic: 21 feature pairs with |r| > 0.9 create extreme SHAP instability where single-model stability is only 0.317.

### Methodological strengths

- Clean experimental design isolating sequential vs. independent as the primary contrast (DASH vs. LSM: same tree count, same colsample_bytree, different training structure)
- Variance decomposition (data seed vs. model seed) directly quantifying the contribution of model-selection randomness
- Pre-specified success criteria (11 pass/fail tests documented before running)
- Honest handling of the accuracy/equity confound (Section 5, ground-truth caveat)
- The SR equivalence is a genuine theoretical contribution: it shows the operative mechanism is independence, not any particular aggregation strategy

### Methodological weaknesses

- All experiments use XGBoost; no LightGBM, no CatBoost, no random forest (acknowledged for journal version)
- Only linear and one specific nonlinear DGP tested. The nonlinear scope boundary (stability drops from ~0.93-0.98 to ~0.87-0.93) is a real limitation
- N_reps=20 leaves the DASH vs. SR comparison underpowered
- Real-world datasets evaluate stability only, not accuracy (no GT available)
- The "equity" metric presupposes that equal credit within correlated groups is the correct decomposition — this is a design choice, not a ground truth
- Independent researchers (no institutional affiliation) — not disqualifying, but the work hasn't been through peer review yet

---

## Relevance to Somnigraph

### What this paper does that we don't

**1. Diagnosis of correlated feature groups in the reranker.** Somnigraph's 26-feature (31 with pending additions) LightGBM reranker in `src/memory/reranker.py` has known correlated feature groups:

- **Rank/score pairs:** `fts_rank`/`fts_bm25`, `vec_rank`/`vec_dist` — rank is a monotonic transform of the raw score within each query, so they're highly correlated
- **Feedback signals:** `fb_last`/`fb_mean`/`fb_count`/`fb_time_weighted` — all derived from the same feedback event stream, likely correlated
- **Age/access:** `age_days`/`session_recency` — temporal features measuring overlapping concepts
- **Pending normalized features:** `fts_bm25_norm`/`fts_bm25`, `vec_dist_norm`/`vec_dist` — by definition correlated (one is a per-query normalization of the other)

We currently use LightGBM's built-in `feature_importances_` (gain-based) and the feature selection experiments in STEWARDSHIP (forward stepwise, backward elimination) to assess feature value. Neither method accounts for first-mover bias — the feature importance rankings from our single production model may be path-dependent artifacts of whichever feature won early splits during training.

**2. The FSI diagnostic.** We have no way to audit whether our feature importances are trustworthy without retraining. The FSI formula (cross-model SHAP variance / mean importance) could detect which features in the reranker have unstable attributions, flagging the correlated groups above without computing the correlation matrix directly.

**3. The IS Plot quadrant framework.** This would classify each of our 26 features into: robust drivers (trust the ranking), collinear cluster members (treat the group as one signal), confirmed unimportant (safe to remove), or fragile interactions (investigate further). This is directly useful for the pending 31-feature retrain — knowing which features are cluster members vs. independent drivers would inform whether to keep correlated pairs or consolidate them.

### What we already do better

**1. Feature selection via forward stepwise and backward elimination.** The LoCoMo reranker experiments (STEWARDSHIP changelog, 2026-03-22/23) used both forward stepwise and backward elimination with explicit cross-validation metrics (R@10, NDCG@10). This is a *predictive* feature selection approach — it measures whether adding/removing a feature helps ranking quality, not whether SHAP attributes credit correctly. First-mover bias affects *explanation* stability, but our feature selection is based on held-out ranking metrics, which are immune to SHAP instability because we never use SHAP to select features.

**2. We don't use SHAP for feature selection or model decisions.** The reranker pipeline in `scripts/train_reranker.py` selects features based on 5-fold CV NDCG and R@10, not SHAP importance. Feature importance is reported for understanding, not for deciding what to keep. First-mover bias is primarily a problem when SHAP drives feature selection — if you use it only for post-hoc understanding, the worst case is a misleading but harmless narrative.

**3. Multiple retrained models already exist.** The LoCoMo experiments trained dozens of models with different feature subsets, and the production model has been retrained multiple times with evolving feature sets (18 -> 26 -> 31 features). While this isn't systematic model independence in DASH's sense, it provides some implicit protection against acting on a single model's arbitrary feature concentrations.

---

## Worth Stealing (ranked)

### 1. FSI audit of correlated reranker features

**What:** Compute Feature Stability Index across 10-20 retrained production reranker models with different random seeds to identify which feature importances are path-dependent.

**Why it matters:** We have at least 4 known correlated feature groups (rank/score, feedback cluster, temporal, pending normalized scores). The production model's `feature_importances_` reports `query_idf_var` as the top new feature and `proximity` as near-zero — but if these are affected by first-mover bias, the narrative in STEWARDSHIP ("proximity contributes almost nothing") might be wrong. It might be that proximity is in a correlated group where another feature grabbed the first-mover advantage.

**Implementation:** In `scripts/train_reranker.py`, add a `--stability-audit` mode that: (a) trains M=20 models with different random seeds but identical hyperparameters (Stochastic Retrain, the simplest independence method), (b) computes LightGBM's gain-based importance for each, (c) computes FSI = std(importance across models) / mean(importance across models) for each feature, (d) generates an IS Plot. This requires no SHAP computation — gain-based importance is sufficient for the stability diagnostic and is nearly free to compute.

**Effort:** Low. 1-2 hours. The training infrastructure exists; this is a loop around the existing training call plus a few lines of analysis. No new dependencies.

### 2. Correlated feature group consolidation analysis

**What:** Use the FSI results to decide whether correlated feature pairs should be consolidated before the 31-feature retrain.

**Why it matters:** The pending 31-feature model adds `fts_bm25_norm` and `vec_dist_norm`, which are by construction correlated with `fts_bm25` and `vec_dist`. The paper predicts this will worsen first-mover bias within those groups without adding predictive signal (if the normalized version carries the same information as the raw version, the model splits arbitrarily between them). The R@10 vs. NDCG@10 feature set disagreement documented in STEWARDSHIP (2026-03-23 changelog) might partly be a manifestation of first-mover bias: the two metrics select different features from the same correlated group because different random seeds happened to favor different first movers.

**Implementation:** After the FSI audit (#1), examine whether high-FSI feature pairs correspond to the known correlated groups. If so, consider: (a) keeping only the stronger member of each pair for the 31-feature retrain, or (b) using a single combined feature (e.g., the normalized score only, dropping the raw score). This would reduce the feature count but potentially improve both stability and generalization.

**Effort:** Low-Medium. Depends on FSI audit results. If correlations are confirmed, the consolidation decisions are straightforward; the feature extraction code in `reranker.py` would need minor edits.

### 3. Seed-averaged feature importance for documentation

**What:** Report feature importances as the mean +/- std across 10+ random seeds rather than from a single model, for all future documentation of the reranker.

**Why it matters:** The current STEWARDSHIP documentation says things like "query_idf_var is top new feature by importance." If this ranking is unstable across seeds, the documentation is misleading. Reporting the stability band (mean +/- std) alongside the ranking would be honest accounting — Priority 1 in STEWARDSHIP.

**Implementation:** Minor addition to the training script's reporting. No impact on the deployed model — the production model is still a single model trained on all data. The reporting change is purely for documentation accuracy.

**Effort:** Low. Trivially piggybacked on #1.

---

## Not Useful For Us

### Full DASH pipeline

DASH trains M=200 diverse XGBoost models, filters, selects a diverse K=30 subset, and computes consensus SHAP. This is designed for *explanation reliability* — producing stable feature importance rankings. Somnigraph's reranker is a *production scoring model*. We need one model that scores well, not 200 models that explain consistently. The paper explicitly states DASH "is not intended to outperform tuned single models on prediction." Our concern is ranking quality (NDCG, R@10), not explanation stability.

### SHAP computation infrastructure

The paper's SHAP analysis (interventional TreeSHAP, background datasets, explain sets) is substantial infrastructure for a concern we don't have. We use LightGBM's built-in gain importance for post-hoc understanding, not SHAP. Adding TreeSHAP would require the `shap` dependency, a held-out explain set, and meaningful compute per model. The gain-based FSI audit (#1 above) provides the diagnostic value without any SHAP computation.

### Equity metric / within-group credit distribution

DASH's equity metric measures whether correlated features receive proportional credit. For a production model this is irrelevant — we don't care whether `fts_rank` and `fts_bm25` share credit "fairly," only whether the model ranks memories correctly. Equity is a concern for regulatory or scientific explanation contexts, not for a retrieval reranker.

### Nonlinear DGP scope caveat

The paper shows DASH's advantage diminishes under nonlinear data-generating processes (stability ~0.87-0.93 vs. ~0.93-0.98 for linear). Somnigraph's retrieval scoring is almost certainly nonlinear — features interact in complex ways (e.g., feedback_mean matters more for old memories, PPR matters more for multi-hop queries). This means even if we ran full DASH, the stability gains would be in the lower range. The FSI diagnostic remains useful regardless of the DGP.

---

## Impact on Implementation Priority

### Reranker iteration (P2 in STEWARDSHIP)

The most directly affected priority. The pending 31-feature retrain should be preceded (or accompanied) by the FSI stability audit. This is a small addition to the existing workflow, not a new priority:

1. Before the 31-feature retrain, run the stability audit on the current 26-feature model to establish a baseline.
2. Compare the FSI results against the known correlated groups.
3. Decide whether to consolidate correlated pairs before adding the 5 new features.
4. After the 31-feature retrain, re-run the audit to check whether the new features worsened stability.

This doesn't change the priority ordering but adds a quality check to the retrain workflow.

### Honest accounting (P1)

The paper directly supports reframing how we document feature importance. Instead of "query_idf_var is the top new feature," honest accounting requires either: (a) confirming this ranking is stable across seeds, or (b) reporting it as "query_idf_var is the top new feature in the production model, but importance rankings for correlated features may be path-dependent." This is a documentation standard, not a code change.

### LoCoMo benchmark (P4)

The R@10 vs. NDCG@10 feature set disagreement (3 features backward-eliminated for NDCG were selected by R@10 forward — STEWARDSHIP 2026-03-23) may partly reflect first-mover bias rather than genuine metric disagreement. If those 3 features belong to correlated groups, the disagreement could be an artifact of different random seeds favoring different first movers. The FSI audit could clarify this.

### Roadmap experiments

- **Tier 1 #5 (counterfactual coverage check):** Not directly affected. Counterfactual coverage measures retrieval quality, not explanation stability.
- **Feedback self-reinforcement (open problem):** Tangentially related — the six downstream signals that amplify the same exposure event are correlated features in the reranker. If feedback_mean, ucb_bonus, fb_last, fb_count, fb_time_weighted, and hebbian_pmi are all carrying correlated signal, first-mover bias determines which one gets the importance credit. But the self-reinforcement concern is about the feedback *loop*, not about how credit is distributed among correlated features.

---

## Connections

### Reranker methodology (internal)

The paper is most directly relevant to `scripts/train_reranker.py` and the feature selection experiments documented in `docs/experiments.md`. The forward stepwise / backward elimination methodology is immune to first-mover bias because it evaluates features by held-out ranking metrics, not by SHAP importance. But the *interpretation* of feature importance results (which features matter, which are redundant) is vulnerable.

### R@10 vs. NDCG feature disagreement (internal)

The finding that R@10 and NDCG select structurally different features (STEWARDSHIP 2026-03-23) could have a first-mover component. If the different objectives create different residual structures, the first-mover advantage may favor different features within the same correlated group. The FSI audit can disambiguate: if the disagreeing features have high FSI, the "disagreement" is partly noise; if they have low FSI, it's genuine.

### Utility calibration study (internal)

The calibration study (STEWARDSHIP 2026-03-14, docs/experiments.md) found per-query Spearman r=0.70 and per-memory r=0.14. The per-memory weakness means feedback features (`fb_mean`, `fb_last`, etc.) are noisier at the memory level than they appear at the query level. This noise increases the correlation between feedback features (they're all noisy estimates of the same underlying signal), which is exactly the condition where first-mover bias thrives.

### LoCoMo benchmark (P4)

The LoCoMo-specific 17-feature reranker (distinct from the 26-feature production model) was trained with R@10 optimization. If this model also has correlated feature groups (it likely does — `fts_rank`/`vec_rank` etc. are in both models), its feature importances are similarly vulnerable. The "3 of 6 expansion methods dead" finding (rocchio 0%, multi_query 2%, entity_focus 4%) is based on expansion method contribution measured through the reranker's feature importance — if the relevant features are in a correlated group, the 0% might be a first-mover artifact.

### SPLADE (research/sources/splade.md)

SPLADE's learned sparse representations offer an alternative path that sidesteps the correlated-feature problem entirely: instead of separate FTS and vector channels producing correlated rank/score pairs, a single learned sparse representation could replace both. This doesn't address first-mover bias in the reranker directly, but it would reduce the number of correlated feature groups if adopted.

---

## Summary Assessment

This is solid work that identifies a real mechanism (sequential residual dependency) behind a well-known problem (SHAP instability under multicollinearity) and proposes a principled resolution (model independence). The experimental design is clean — particularly the DASH vs. LSM contrast that isolates sequential vs. independent training while controlling tree count and feature restriction — and the SR equivalence is a genuine theoretical contribution. The pre-specified success criteria and honest handling of the accuracy/equity confound reflect research maturity unusual for an independent team's first paper on the topic.

For Somnigraph specifically, the paper's value is diagnostic rather than architectural. We don't need DASH — our feature selection is metric-driven, not SHAP-driven, so first-mover bias doesn't affect our *decisions*. But it does affect our *understanding*: the feature importance narratives in STEWARDSHIP and experiments.md may be partially artifacts of which features won early splits. The FSI stability audit is a low-cost addition to the reranker training workflow that would either confirm or correct these narratives.

The single most important takeaway: before the 31-feature retrain, add 10 lines to `train_reranker.py` that retrain with different random seeds and report importance stability. If the correlated feature groups (rank/score, feedback cluster, temporal) show high FSI, consolidate before adding more correlated features. If they show low FSI, proceed as planned. Either way, report importance as mean +/- std rather than from a single model — that's honest accounting.
