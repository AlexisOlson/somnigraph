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

**Evaluation set limitation (March 2026 caveat):** The LoCoMo-derived evaluation set has ~17 testable queries. With 7 tunable parameters, this is a regime where optimization can find "structure" in noise — the discretization ceiling isn't just a measurement inconvenience, it means most of the parameter space is observationally equivalent. Two configurations producing the same miss_rate on 17 queries may have very different behavior on 200 queries. Constants derived from LoCoMo should be treated as exploratory hypotheses, not confirmed properties of the scoring system. Real-data tuning is underway (wm24–wm34, using 200 judged queries from a 1,047-query ground truth set, with 5-fold cross-validation) — the first confirmatory test of whether LoCoMo-derived structure transfers.

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

*For per-study results, see the tuning studies log. For cross-study mechanism analysis, see architecture.md § Tuning.*
