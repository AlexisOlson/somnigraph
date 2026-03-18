# Experiments

How to run, interpret, and learn from tuning studies on the retrieval scoring pipeline. This isn't a log of results — it's methodology for someone who wants to reproduce or extend the work.

## Contents

1. [What we're optimizing](#what-were-optimizing)
2. [The evaluation setup](#the-evaluation-setup)
3. [Metrics: why AUC over single-threshold](#metrics-why-auc-over-single-threshold)
4. [The staged approach](#the-staged-approach)
5. [Reading corner plots](#reading-corner-plots)
6. [The two-basin discovery](#the-two-basin-discovery)
7. [Multi-objective optimization](#multi-objective-optimization)
8. [Running a study](#running-a-study)
9. [Interpreting results](#interpreting-results)

---

## What we're optimizing

The scoring pipeline has 7 tunable parameters that control how memories are ranked after hybrid retrieval:

| Parameter | Range | What it controls |
|-----------|-------|-----------------|
| `RRF_K` | 1–100 | RRF fusion sharpness (low = top ranks dominate, high = soft union) |
| `FEEDBACK_COEFF` | 0–15 | Weight of feedback boost relative to RRF base score |
| `THEME_BOOST` | 0–2 | Multiplicative boost per overlapping theme |
| `HEBBIAN_COEFF` | 0–10 | Co-retrieval PMI boost weight |
| `HEBBIAN_CAP` | 0–0.5 | Maximum Hebbian contribution per memory |
| `ADJACENCY_BASE_BOOST` | 0–1 | Base boost for graph-expanded neighbors |
| `ADJACENCY_NOVELTY_FLOOR` | 0–1 | Minimum novelty score for expansion to fire |

These parameters interact. `RRF_K` controls how spread out candidate scores are, which changes the effective strength of every downstream boost. `FEEDBACK_COEFF` and `THEME_BOOST` can substitute for each other (both push well-known memories up). The parameter space has structure — basins, ridges, and tradeoffs — not a random landscape.

## The evaluation setup

### Ground truth

Evaluation uses historical retrieval events from `memory_events`. Each event records:
- The query text
- Which memories were retrieved
- The utility scores assigned via `recall_feedback()`

A memory is "relevant" for a query if it received utility ≥ 0.5 in feedback. This is imperfect — not every retrieval gets feedback, and utility is subjective — but it's the signal we have. The alternative (manually labeling query-memory pairs) doesn't scale.

**Evaluation set limitation (March 2026 caveat):** The LoCoMo-derived evaluation set has ~17 testable queries. With 7 tunable parameters, this is a regime where optimization can find "structure" in noise — the discretization ceiling isn't just a measurement inconvenience, it means most of the parameter space is observationally equivalent. Two configurations producing the same miss_rate on 17 queries may have very different behavior on 200 queries. Constants derived from LoCoMo should be treated as exploratory hypotheses, not confirmed properties of the scoring system. Real-data tuning is underway (wm24–wm34, using ~500 judged queries from a 1,047-query ground truth set, with 5-fold cross-validation) — the first confirmatory test of whether LoCoMo-derived structure transfers.

**Selection bias caveat:** Ground truth derived from historical retrievals means the relevance label usually exists only when the system already surfaced the memory. Memories that are relevant but were never retrieved — because they scored poorly under the current constants — are structurally underrepresented in the evaluation set. This biases optimization toward the status quo: it rewards configurations that retrieve what the system already retrieves, not configurations that would find what the system currently misses. Counterfactual evaluation (exhaustive relevance judging of the full candidate set for a subset of queries) is needed to bound this bias.

### Trial execution

Each trial:
1. Temporarily overrides the production constants with trial values
2. Re-runs the scoring pipeline on all historical queries
3. Compares scored rankings against ground-truth relevance
4. Returns a metric value

The scoring pipeline is deterministic given constants — same inputs, same outputs. No API calls during trials (embeddings are cached). A 200-trial study takes ~10–15 minutes.

### Embedding cache

Vector operations dominate trial cost. The tuning framework caches all embeddings at study start and reuses them across trials. This makes trials pure computation: SQL queries + arithmetic.

## Metrics: why AUC over single-threshold

### The discretization problem

The first studies used `miss_rate@5k`: the fraction of known-useful memories not retrieved within a 5,000-token budget. This metric has a ceiling problem.

With ~17 testable queries, miss_rate values are ratios of small integers: 0/17, 1/17, 2/17, etc. The metric has only ~3 distinguishable performance levels in the range where good configurations live. Two configurations that are measurably different in ranking quality can produce identical miss_rate scores.

Study wm3 demonstrated this: the optimizer found configurations with miss_rate = 0.2941 (5/17) and couldn't push below it despite hundreds of trials. The metric, not the system, was the binding constraint.

### Weighted miss_rate

An intermediate fix: weight each miss by the memory's feedback utility, so missing a highly-rated memory costs more than missing a marginal one. This partially solves discretization but introduces a new problem — the metric becomes sensitive to the utility distribution, which varies as the system accumulates feedback.

### AUC (area under the threshold curve)

The solution: don't evaluate at a single budget threshold. Instead, sweep across all possible thresholds and compute the area under the retrieval curve. AUC measures ranking quality holistically: a configuration that puts relevant memories at ranks 1, 2, 3 scores higher than one that puts them at ranks 1, 5, 10, even if both retrieve the same memories within a generous budget.

AUC avoids discretization because it's continuous — small ranking improvements produce small AUC improvements. Study wm9, the first to use AUC, immediately found parameter regions that miss_rate couldn't distinguish.

### MRR (mean reciprocal rank)

MRR focuses on the first relevant result: 1/rank of the top relevant memory, averaged across queries. It captures "how quickly do you find something useful?" which matters for startup_load (tight budget, want the best memories first).

MRR and AUC mostly agree (the Pareto front is shallow), but MRR trades off harshly against miss_rate (r = −0.517 in wm15). Optimizing for "find the first good one fast" can mean missing other good ones entirely.

## The staged approach

Early studies tuned all 7 parameters simultaneously. This works but is slow and produces uninterpretable results — you can't tell which parameters matter.

The staged approach:

1. **Stage 1: Core trio** — Tune `RRF_K`, `FEEDBACK_COEFF`, `THEME_BOOST` with others fixed. These three account for >90% of score variance.
2. **Stage 2: Graph parameters** — Fix the core trio at Stage 1 values, tune `HEBBIAN_COEFF`, `HEBBIAN_CAP`, `ADJACENCY_BASE_BOOST`, `ADJACENCY_NOVELTY_FLOOR`.
3. **Stage 3: Joint refinement** — Narrow search around the Stage 1+2 optimum, all 7 free.

Stage 1 identifies the basin. Stage 2 fills in the graph details. Stage 3 polishes. Most of the value comes from Stage 1 — the graph parameters have minor effect (combined <3% importance) when feedback is working.

The risk: staged optimization can miss cross-stage interactions. Study wm2 revealed a `RRF_K × HEBBIAN_COEFF` interaction (r = −0.77) that the staged approach missed. Joint optimization (wm9, wm15) confirmed that the interaction exists but doesn't change the basin — it shifts the optimum within it.

## Reading corner plots

The tuning framework generates CTT-style corner plots (`plot_tuning.py`): scatter matrices showing every parameter pair, colored by trial metric value.

**What to look for:**

- **Basins**: Clusters of high-metric trials in a region. Two separate clusters = two basins (the system has multiple local optima).
- **Ridges**: Lines of high-metric trials along one parameter axis. The metric is insensitive to one parameter while sensitive to the other — a ridge says "this parameter doesn't matter much."
- **Interactions**: Diagonal patterns in parameter-pair plots. If good trials form a line from (low A, high B) to (high A, low B), parameters A and B substitute for each other.
- **Walls**: Sharp metric transitions along one axis. A wall at `FEEDBACK_COEFF < 2` means the system breaks when feedback is too weak.

**Auto-scaling**: `plot_tuning.py` uses log-scale for parameters with wide ranges (RRF_K, coefficients) and linear for bounded parameters (caps, floors). The `--metric` flag selects which metric to visualize.

## The two-basin discovery

Study wm9 revealed two distinct high-performing regions in the parameter space:

**Basin A (high-k, k ≈ 14):** RRF produces a broad candidate set with compressed scores. Post-RRF signals (feedback, themes, Hebbian) do the real ranking work. This basin relies on the feedback loop being healthy.

**Basin B (low-k, k ≈ 3):** RRF produces a sharp ranking where top candidates strongly dominate. Post-RRF signals are refinements, not primary rankers. This basin is more robust to feedback degradation but less responsive to accumulated learning.

Production uses Basin A (k=6, a compromise within the high-k regime). The reasoning: the feedback loop is the system's competitive advantage. Choosing the basin that leverages it most makes the system more different from alternatives, not less. If the feedback loop breaks, that's a bug to fix, not a parameter to route around.

The two-basin structure explains why different studies found different "optimal" parameters — they were finding different basins, not different estimates of the same optimum.

## Multi-objective optimization

Study wm15 used NSGA-II (a multi-objective evolutionary algorithm) to optimize AUC and MRR simultaneously. Instead of a single "best" configuration, this produces a Pareto front: the set of configurations where improving one metric requires worsening the other.

**Findings:**

- **AUC vs. MRR**: Barely trade off. The Pareto front is shallow — you can have 95%+ of both simultaneously. This means the retrieval pipeline doesn't face a fundamental tension between "rank everything well" and "find the first good result fast."
- **AUC vs. miss_rate@5k**: Also mild tradeoff.
- **MRR vs. miss_rate@5k**: Harsh tradeoff (r = −0.517). Optimizing for fast first-hit actively hurts coverage. This makes sense — putting one memory at rank 1 can push other relevant memories past the budget threshold.

**Production selection**: Pareto point #4 (of 5), slightly favoring AUC over MRR. The reasoning: startup_load already handles the "best memories first" case with its own priority ranking. Recall needs good overall ranking more than a perfect first hit.

## Running a study

```bash
# Basic study: 200 trials optimizing AUC
uv run scripts/tune_memory.py --metric auc --n-trials 200

# Staged: core trio only
uv run scripts/tune_memory.py --metric auc --n-trials 100 --stage core

# Multi-objective
uv run scripts/tune_memory.py --metric auc --metric2 mrr --sampler nsga2 --n-trials 300

# Generate corner plots
uv run scripts/plot_tuning.py --study-name <name> --metric auc
```

The framework uses Optuna for Bayesian optimization (TPE sampler by default, NSGA-II for multi-objective). Studies are stored in a SQLite database alongside the memory database, so results persist across sessions.

**Embedding cache**: Built automatically at study start. Subsequent studies reuse the cache if the memory database hasn't changed.

**Bootstrap confidence**: The framework computes bootstrap confidence intervals on the best trial's metric. This catches cases where the "best" trial is an outlier — if the 95% CI is wide, the result is unreliable.

## Interpreting results

### Feature importance

After a study, compute feature importance: the correlation between each parameter and the metric across all trials. High importance means the metric is sensitive to that parameter; low importance means it can be set to any reasonable value.

Typical importance ranking (wm15):
1. `THEME_BOOST` — 67% AUC importance, 86% MRR importance. Dominant.
2. `FEEDBACK_COEFF` — 15–20%. Strong but secondary to themes.
3. `RRF_K` — 10–15%. Basin selection.
4. `ADJACENCY_BASE_BOOST` — 2–5%. Non-removable but minor.
5. `HEBBIAN_*` — <1%. Nearly irrelevant when feedback works.

### What "importance" means and doesn't mean

High importance means the metric is *sensitive* to the parameter. It doesn't mean the parameter is *useful*. A parameter with high importance and optimal value near zero is one the system wants *removed* — its presence helps only at zero, meaning its absence would help equally.

Low importance means the parameter can be set anywhere in its range without much effect. This is actually good news — it means the system is robust to that parameter, and you don't need to tune it carefully.

### When to stop tuning

Diminishing returns set in after ~200 trials for a 7D space. The first 50 trials explore; trials 50–150 exploit the best region; trials 150+ refine within the basin. If the best trial hasn't improved in 50 consecutive trials, additional trials are unlikely to help.

More impactful than additional trials: changing the metric (reveals different aspects of the system), adding evaluation queries (reduces discretization), or fixing bugs in the pipeline (wm5's revelation that centering is load-bearing came from a validation study, not more optimization).

---

## Utility calibration study

The external reviews (March 2026) flagged a central question: does feedback correlate with independently-judged relevance, or does the feedback loop just reinforce retrieval habits? This study answers it using two datasets that share the same memory pool:

- **Feedback scores**: 13,396 utility ratings from live use, stored in `memory_events`
- **Ground truth scores**: 13,317 relevance judgments from a blind LLM judge (Sonnet, no access to feedback history) across 500 queries

### The two-level result

The answer depends on the unit of analysis:

| Level | Spearman r | Pearson r | n | Interpretation |
|-------|-----------|-----------|---|----------------|
| Per-query (same query, same memory) | 0.70 | 0.77 | 3,072 | Strong — feedback tracks relevance in context |
| Per-memory (averaged across queries) | 0.14 | 0.11 | 715 | Weak — aggregation destroys context-dependent signal |

**Per-query**: When a memory is rated during a specific recall and the GT judge evaluates the same query-memory pair, the two scores agree strongly (r = 0.70). Feedback has real signal. The reviewers' concern about self-reinforcement is not confirmed at this level — the loop is learning what's relevant for each query.

**Per-memory**: When you average a memory's feedback across all queries it appeared in and compare against its average GT relevance, correlation collapses (r = 0.14). This is expected: a memory that's essential for one query (fb = 0.95) and irrelevant for ten others (fb = 0.05 each) averages to ~0.13, but its GT profile might be different because the GT candidate sets are drawn differently than live retrieval.

### Why this matters for the empirical Bayes prior

The prior operates at the per-memory level. It computes an EWMA of a memory's feedback history and shrinks it toward a population mean. This is the level where correlation is weak (r = 0.14). The prior is aggregating a context-dependent signal in a context-free way — it treats "this memory scored 0.9 for query A" and "this memory scored 0.1 for query B" as two noisy estimates of the same underlying quality, when they're actually two accurate estimates of different things.

The prior still helps (wm5 showed raw feedback without the prior is worse than no feedback at all), but it's helping by smoothing noise, not by recovering the true per-memory quality. A prior conditioned on query context (themes, category, even a crude cluster) could preserve more of the r = 0.70 signal that currently gets averaged away.

### Quadrant analysis

Classifying the 715 overlapping memories by feedback threshold (0.5) and GT threshold (0.5):

| Quadrant | Count | % | Meaning |
|----------|-------|---|---------|
| High feedback, high GT | 19 | 2.7% | Validated — feedback agrees with relevance |
| High feedback, low GT | 117 | 16.4% | Inflated — feedback overrates these memories |
| Low feedback, high GT | 50 | 7.0% | Coverage gap — relevant but underrated |
| Low feedback, low GT | 529 | 74.0% | Correctly filtered |

The 16.4% "inflated" quadrant looks alarming but is an artifact of per-memory averaging. Manual inspection of the top 10 outliers (gap +0.71 to +0.84) reveals a consistent pattern: every one has exactly 1 feedback event with a high score (0.90–1.00), and the GT judge *agrees* on the matching query (0.95+ for the same query). The low mean GT comes from the memory appearing in 6–30 other GT candidate sets where it's topically irrelevant (0.10–0.12). These are niche memories correctly rated for their specific use case — the Papyrus scripting gotcha, the UNICHAR XML edge case, the Spriggit YAML structure. They score 0.95 when you need them and 0.10 when you don't. The per-memory mean mischaracterizes this as inflation.

The effective self-reinforcement rate in this dataset is near zero. The reviewers' concern was valid as a structural risk, but the empirical evidence doesn't support it — at least not at the current corpus size and feedback density.

The 7.0% coverage gap is also largely an averaging artifact. Manual inspection of the top 10 gap outliers reveals a retrieval precision problem, not an underrating problem: these memories are surfaced for many queries where they're irrelevant (fb = 0.00), accumulating low scores that dilute their per-memory mean, even though they score 0.80–1.00 for matching queries. For example, the ElevenLabs transcription memory has 32 feedback events including three 1.00 scores, but also many 0.00 scores from unrelated Skyrim and modding queries. Its mean fb (0.24) vs mean GT (0.68) looks like a coverage gap, but the per-query feedback is accurate — the system just retrieves it too broadly.

Both quadrants tell the same story: per-memory averaging mischaracterizes context-dependent memories. The feedback loop is well-calibrated per-query. The real problem is upstream — retrieval precision determines which queries a memory gets exposed to, and overexposure to irrelevant queries dilutes per-memory means.

### Never-surfaced memories

11 memories appear in GT candidate sets but have no feedback history at all — the retriever surfaced them as candidates but they were never rated. 5 of these have mean GT relevance >= 0.5, appearing across 6–21 queries each. These are the purest "coverage gap" cases: relevant memories the system knows about but never promotes high enough to get feedback.

### Cutoff signal validation

The `recall_feedback()` accepts a `cutoff_rank` parameter: the position where the rater stopped finding useful results. This signal is logged as a `recall_cutoff` event but doesn't currently feed into scoring. The calibration study data lets us validate whether it's a real signal.

**Above vs. below cutoff per-query GT relevance:**

| Position | Mean GT | GT >= 0.5 | n |
|----------|---------|-----------|---|
| Above cutoff | 0.549 | 51.8% | 1,923 |
| Below cutoff | 0.336 | 23.1% | 1,404 |

The cutoff separates relevant from irrelevant at a 2.2:1 ratio (51.8% vs 23.1% relevant). This is a real signal — it identifies the useful/noise boundary that the GT judge independently confirms.

**Cutoff vs. cliff detection:**

The automated cliff detector (`CLIFF_Z_THRESHOLD = 2.0`, log-curve deviation) is far too permissive compared to the rater cutoff:

| Comparison | Count | % |
|-----------|-------|---|
| Cliff kept MORE than rater wanted | 746 | 96.1% |
| Cliff kept SAME | 30 | 3.9% |
| Cliff kept LESS | 0 | 0.0% |

When the cliff over-delivers, it returns an average of 7.4 extra memories. The score gap at the rater cutoff is < 0.02 in 83.6% of cases — well below the cliff detector's sensitivity. The cliff catches only dramatic score drops; the real useful/noise boundary is much more subtle and requires the rater's contextual judgment.

This has two implications:

1. **The cliff Z threshold could be calibrated from cutoff history.** 846 cutoff events provide a ground truth for where the useful/noise boundary actually falls. Instead of a fixed Z=2.0, the threshold could be fit to minimize disagreement with historical cutoff positions.

2. **Per-memory cutoff statistics could inform scoring.** A memory that consistently falls below cutoff when surfaced is being overretrieved. This is the retrieval precision signal that the quadrant analysis identified as the real problem — and cutoff_rank already measures it.

### Ground truth caveats for this study

The GT judge and the feedback rater measure different things. The GT judge evaluates topical relevance of a memory to a query (would this memory help someone searching for this?). Live feedback evaluates experienced utility in context (did this memory actually help in this session, given what else was surfaced?). Perfect correlation shouldn't be expected even if both signals are accurate — a topically relevant memory might not be useful if three other memories already covered the same ground.

The per-query correlation (r = 0.70) is high enough to confirm that both signals track the same underlying construct. The per-memory divergence (r = 0.14) is a property of the aggregation, not a failure of either signal.

---

## Reranker methodology

The reranker replaces the hand-tuned scoring formula with a learned LightGBM model. This section documents the methodology — feature design, training strategy, and comparison infrastructure.

### Feature extraction

18 features per (query, candidate) pair. The features are raw signals — no derived scores from the formula, no parameter dependencies. This means the model can be retrained without re-tuning any constants.

**Retrieval signals (query-dependent):**
- `fts_rank`, `vec_rank`, `theme_rank` — Position in each retrieval channel's ranking. -1 if the candidate wasn't retrieved by that channel.
- `fts_bm25` — Raw BM25 score from FTS5. 0 if not retrieved.
- `vec_dist` — Raw cosine distance from vector search. 0 if not retrieved.
- `theme_overlap` — Number of query tokens matching the memory's theme tags. 0 if none.

**Feedback signals (query-independent):**
- `fb_last` — Most recent utility score. -1 sentinel if no feedback.
- `fb_mean` — Mean of all utility scores. -1 sentinel if no feedback.
- `fb_count` — Number of feedback events.

These are raw values, not EWMA-smoothed. The formula used EWMA with a tunable alpha parameter; the model learns its own recency weighting from `fb_last` vs `fb_mean`.

**Graph signals:**
- `ppr_score` — Raw Personalized PageRank score from graph walk. Uses RRF-based seed selection.
- `hebbian_pmi` — Raw PMI co-retrieval score. Seeds are top-5 candidates by best channel rank.

**Memory metadata:**
- `category` — Ordinal encoded (episodic=0, semantic=1, procedural=2, reflection=3, entity=4, meta=5).
- `priority` — Base priority (1-10).
- `age_days` — Days since `created_at`.
- `token_count` — Token length of memory content.
- `edge_count` — Number of graph edges touching this memory.
- `theme_count` — Number of themes assigned.
- `confidence` — Confidence score (0.1-0.95).

### Training strategy

**Cross-validation:** 5-fold GroupKFold, grouped by query. This prevents data leakage — a query's candidates never appear in both train and validation. GroupKFold is critical because candidates for the same query share contextual features; standard KFold would overestimate performance.

**Model:** LightGBM pointwise regressor. Default hyperparameters (num_leaves=31, learning_rate=0.1, n_estimators=500, early stopping at 50 rounds). No hyperparameter tuning — the model's advantage comes from expressiveness, not careful tuning.

**Labels:** Continuous GT relevance scores (0-1) from the judged ground truth set. The regressor predicts relevance directly; ranking is by predicted score.

**LambdaRank variant:** Also implemented (`--lambdarank` flag). Discretizes continuous labels into quantile bins, trains LGBMRanker to optimize NDCG directly. Requires GT-only candidate pools for training (with negative sampling) but evaluates on the full candidate pool. This is an improvement experiment, not the current default.

### Comparison infrastructure

Both the formula and the reranker are evaluated on the same data with the same metrics:

- **NDCG@5k:** Token-budget-aware NDCG. Candidates are packed greedily into a 5000-token budget; ideal DCG uses the same budget constraint. This is the primary comparison metric.
- **Graded Recall@5k:** Relevance-weighted recall. For each query, count N relevant memories (GT ≥ 0.5) in the budget, then score: sum(GT scores of those N) / sum(top-N GT scores). A ranker that finds the N most relevant memories scores 1.0, even when more relevant memories exist than the budget holds. *(Revised March 2026 — the original metric used total relevant count as denominator, which penalized rankers when GT had more relevant memories than the budget could hold. See § GT v2.)*

Evaluation is always out-of-fold: the model predicts on held-out queries, never on training data. The formula gets the same query set for a fair comparison.

### The feedback ablation

Dropping all three feedback features (fb_last, fb_mean, fb_count) costs 0.0004 RMSE — within noise. This was confirmed across all 5 folds and in per-query analysis. The implication: feedback is not useful for ranking, even though it correlates with relevance at the per-query level (r=0.70 from the utility calibration study). The correlation exists but the information is redundant with the retrieval signals — if a memory ranks well in FTS and vector search, feedback adds nothing the model doesn't already know.

### Live scoring

Feature extraction during `impl_recall()` must produce identical features to training-time extraction. The key challenge: training extracts features from cached, precomputed search results; live scoring computes them on-the-fly. The `reranker.py` module handles this, using the same raw signals (ranks, PPR scores, feedback sequences, metadata) from live database queries.

### Feature parity verification (March 2026)

Comparing live feature extraction (`reranker.py`) against training-time extraction (`train_reranker.py` / `tune_gt.py`) revealed six mismatches. All were fixed to ensure live scoring produces identical features to training:

| Feature | Training path | Live path (before) | Fix |
|---------|--------------|-------------------|-----|
| PPR seed k-values | Asymmetric K_FTS=8.002, K_VEC=6.845 | Symmetric RRF_K=6 | Both → symmetric RRF_K |
| PPR seed count | 30 | 5 | Both → 30 (PPR_RERANKER_SEEDS) |
| Theme scan scope | All active memories | FTS/vec candidates only | Both → all active |
| Theme case matching | Raw themes vs lowered query | Both lowered | Both → both lowered |
| PPR contradiction edges | Included | Excluded | Both → excluded |
| PPR score threshold | All > 0 | Only > PPR_MIN_SCORE (0.007) | Both → all > 0 |

Retraining with corrected features improved NDCG from +5.7% to +6.8% on v1 GT (500 queries) — the mismatches were actively hurting live performance.

### GT v2: hard negatives and selection bias

GT v1 (`gt_calibrated.json`, 500 queries) derives relevance labels from historical retrievals — only memories the system already surfaced get judged. GT v2 (`gt_v2.json`, 337+ queries in progress) judges the full candidate pool per query, including memories the system never retrieved.

The difference is stark:

| Metric | v1 GT | v2 GT (337q) |
|--------|-------|-------------|
| Candidates judged / query | 26.6 | 52.3 |
| Relevant memories / query (mean) | 2.9 | 10.2 |
| Relevant memories / query (max) | 19 | 146 |
| Queries with >25 relevant | 0% | 8.6% |
| Formula NDCG@5k | 0.74 | 0.73 |

NDCG is stable across GT versions (0.74 → 0.73), confirming that ranking quality is unchanged — v2 is a harder test set, not evidence of a regression.

The original Recall@5k metric (fraction of all relevant memories in the budget) dropped from 0.89 to 0.55 on v2, but this was a metric defect: it penalized the ranker for there being more relevant memories than the budget could hold. The metric was revised to relevance-weighted recall (see § Comparison infrastructure). With the corrected metric, both formula and reranker score ~0.96 — when the system finds relevant memories, it finds the best ones. The metric has near-zero discriminative power between scoring methods because GT relevance scores cluster tightly (std=0.156 on the 0.5–1.0 range).

**Implication:** NDCG is the primary comparison metric. Relevance-weighted recall is a useful sanity check (confirms the system selects high-quality results) but can't differentiate scoring methods at the current GT score resolution.

### Improvement experiments

#### LambdaRank: negative result

**Hypothesis:** Training LGBMRanker with `lambdarank` objective (optimizes NDCG directly) should outperform pointwise MSE regressor.

**Result:** Pointwise wins on both GT sets.

| Model | GT | NDCG@5k | vs Formula |
|-------|-----|---------|------------|
| Pointwise | v1 (500q, selection-biased) | 0.8097 | +6.8% |
| LambdaRank | v1 | 0.7937 | +5.2% |
| Pointwise | v2 (337q, hard negatives) | 0.7831 | +5.1% |
| LambdaRank | v2 | 0.7701 | +3.8% |

The v1→v2 NDCG drop (0.81→0.78) reflects v2's harder evaluation, not a regression. v2 is the honest number.

The gap is consistent: pointwise outperforms by 1.3–1.6 NDCG points across both GT sets. Three likely explanations:

1. **Data volume.** Pointwise trains on the full candidate pool (222k samples). LambdaRank trains on GT-only candidates + 2x random negatives (52k samples). More data outweighs a better objective.
2. **Label discretization.** LambdaRank bins continuous relevance into quantile levels, losing granularity the regressor preserves.
3. **The pointwise objective is already good enough.** At this data scale, accurate relevance prediction produces good rankings. The listwise ordering benefit is marginal.

The `--lambdarank` flag remains in `train_reranker.py` for future experimentation if more GT data changes the balance.

#### Diversity feature (max_sim_to_higher): negative result

**Hypothesis:** An MMR-style diversity signal — for each candidate, the maximum cosine similarity to any higher-ranked candidate — would help the reranker avoid redundant results.

**Result:** High feature importance (rank #6 of 19 by gain), but hurts NDCG.

| Model | v2 NDCG@5k |
|-------|-----------|
| Pointwise (18 features) | 0.7847 |
| Pointwise + max_sim_to_higher | 0.7821 (−0.3%) |

The model uses the feature heavily but learns the wrong thing: it penalizes results similar to higher-ranked candidates, but in a pointwise model there's no coordination about which results are actually selected. Suppressing a relevant memory because it's similar to *another relevant memory that might also be suppressed* creates cascading errors. This is the fundamental mismatch between a listwise signal (diversity) and a pointwise model (independent predictions).

The feature also has practical costs: 175s extraction time (vs 2.4s without) due to O(n²) pairwise similarity on ~750 candidates with 1536-dim vectors. Even if it helped ranking, the 70x latency penalty would be prohibitive for live scoring.

**Takeaway:** Diversity-aware reranking requires a listwise model that coordinates selections, not a pointwise feature hack. This is a genuine architectural limitation, not a tuning problem.

---

*For per-study results, see the tuning studies log. For cross-study mechanism analysis, see architecture.md § Tuning.*
