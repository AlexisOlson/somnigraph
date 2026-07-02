# Findings — Sleep Counterfactual Fork (DRAFT, for human review)

> **⚠️ FOR REVIEW — autonomously produced.** This is an unsupervised overnight draft
> (Opus, worktree `exp/sleep-fork`, 2026-07-02). The numbers are machine-checked, but
> every *interpretation* below is a claim for `experiments.md` / `architecture.md` and
> must be ratified by a human before merge. Nothing here was merged or pushed. The live
> store (`~/.claude/data`) was never written — all work was on copies under
> `D:\somnigraph-exp\sleepfork-<run>\`.

## Question

*Does sleep improve retrieval?* — the central open question in `architecture.md`
§ Open problems (roadmap #14 counterfactual sleep eval + #3 sleep impact measurement).

## TL;DR

**At steady state, on an already-consolidated store, sleep has no practically
meaningful effect on retrieval.** Consolidation (edges + refreshed summaries) is
statistically inert; the sleep cycle's only near-significant nudge comes from its
self-generated retrieval-probe feedback, and even that is borderline and negligible in
size. Consolidation and probe-feedback contributions cannot be cleanly separated because
both sit at the noise floor. The strongest single fact: **a full `--deep` pass changed
only ~108 edges and did zero merges/dormancy/archival — the store is saturated, so deep
sleep at steady state has nothing to do.** This is what makes the planned two-week
**abstinence window** the experiment that can actually answer the question — it is the
only condition in which consolidation would have real work.

## Scorer — read this first

`reranker_model.txt` was **absent** from the copied store, so retrieval ran on the
**hand-tuned formula fallback on BOTH sides of every arm.** This is a **sleep-vs-formula**
measurement, not sleep-vs-reranker. (The V6 reranker retrain exists on a branch as of
2026-07-02 but was **not deployed** to the store this experiment copied.) The
sleep-vs-reranker re-measurement is a follow-up once V6 lands. The loud fallback warning
fired on every eval, confirming the scorer identity.

## Method

- **Frozen baseline A** and every slept arm are copies of the same live snapshot (taken
  once at setup), so each arm's pre-sleep state is byte-identical to A. Paired per-query
  deltas, bootstrap 95 % CI (10 000 resamples, fixed seed).
- **GT:** `gt_calibrated.json`, 1032 queries, byte-identical labels across all arms
  (each eval pointed `--gt-path` at A's copy).
- **Metrics:** NDCG@5k (5000-token budget), R@10, MRR (rank cutoff 20). NDCG uses each
  store's own `token_map`, so a GT memory that sleep archived would correctly become
  unretrievable (none were).
- **Eval selection mechanism (important):** `eval_retrieval.py`'s production path reads a
  fixed snapshot at `~/.somnigraph/benchmark/production_snapshot.db` and **ignores
  `SOMNIGRAPH_DATA_DIR`** (`snapshot_production_db` hardcodes the *live* DB). Running
  `--snapshot` on two stores would therefore have evaluated the *same* live DB twice and
  produced a fake null. Instead, each arm's `memory.db` was placed at the snapshot path
  and the eval run **without** `--snapshot`. `eval_production` sets `memory.db.DB_PATH`
  to that snapshot, so candidates, feedback, hebbian, PPR, and token_map all read the
  intended store.
- **One eval-script change** (committed): `eval_production` recorded only `ndcg`; it now
  also emits the ranked `retrieved` list + budget/limit so R@10 and MRR are computable.
- `--configs formula` was used (not `reranker`) to guarantee an unambiguous formula
  scorer regardless of the stale on-disk `.pkl`.

## Arms and results (Δ = slept − frozen; formula both sides; n=1032)

Baseline A: **NDCG@5k 0.6494, R@10 0.5403, MRR 0.9226.**

| Arm | NDCG@5k Δ [95% CI] | R@10 Δ [95% CI] | MRR Δ [95% CI] | Store change vs A |
|---|---|---|---|---|
| **Standard-full** (NREM+REM+probe+repair) | **+0.0022 [+0.0006, +0.0038]** | +0.0025 [−0.0002, +0.0051] | **+0.0059 [+0.0003, +0.0117]** | +137 edges net (871 new consol + 31 probe − 765 removed), 15 summaries refreshed, 0 dormancy/merge/archival, 420 probe feedback events |
| **Standard-consolidation-only** (probe stripped) | +0.0013 [−0.0001, +0.0028] | +0.0008 [−0.0018, +0.0032] | +0.0046 [−0.0004, +0.0098] | +106 edges net, same summaries, probe fully removed |
| **Deep-consolidation-only** (`--deep --probes 0`) | −0.0003 [−0.0015, +0.0010] | −0.0009 [−0.0026, +0.0007] | +0.0032 [−0.0018, +0.0083] | +108 edges, 15 summaries refreshed, 0 dormancy/merge/archival |
| **Probe increment** (standard-full − standard-consol-only) | +0.0009 [+0.0000, +0.0018] | +0.0017 [+0.0000, +0.0036] | +0.0013 [−0.0016, +0.0041] | +31 probe-created edges only |

Bold = 95 % CI excludes zero. Everything else is null.

### What this decomposition says

- **Only the full standard cycle is statistically distinguishable from zero** (NDCG +0.0022,
  MRR +0.0059), and even that is *practically* negligible: +0.0022 NDCG on a 0.65 baseline
  is a **0.34 % relative** change.
- **Consolidation alone is null** — both the standard (probe-stripped) and the deep arms
  have all three CIs straddling zero. The deep arm is the cleanest null (NDCG −0.0003,
  CI symmetric about zero).
- **The probe increment is borderline** — NDCG and R@10 CIs touch zero at the lower bound;
  MRR is null. So the full cycle's marginal significance is the *sum of two individually
  non-significant components* (consolidation ≈ +0.0013, probe ≈ +0.0009). Neither can be
  isolated as "the" driver at this effect size.

## NREM-vs-REM attribution

Not separable at this effect size — the consolidation arms are null, so there is no
positive signal to apportion between edge-building (NREM) and summary refresh (REM). What
*is* observable: NREM produced the edge churn (871 new typed edges — mostly `derived_from`
— against 765 removed, net ≈ +106–108 in both standard and deep), REM refreshed 15 cluster
summaries and generated 0 new summary memories, and dormancy/health made **0** changes.
Standard and deep converged on nearly the same footprint (+106 vs +108 edges), which is
itself evidence of saturation (below).

## Two framing points the reviewer should carry forward

**1. Self-fulfillment / circularity of the probe effect.** The probe selects
high-utility memories, issues synthetic queries whose *targets are those memories*, and
logs recall feedback for them. The formula's dominant signal *is* feedback, and GT
relevance correlates with historical utility. So any probe-driven metric gain is partly
**the scorer being handed the answer key**, not necessarily better retrieval. This is a
direct, first-of-its-kind measurement of the **feedback self-reinforcement loop** flagged
in `architecture.md` — and it says the loop's per-cycle contribution is real but tiny and
borderline. It should be written into the self-reinforcement section, not sold as a
retrieval win.

**2. Why deep looks so small = the headline.** A **full deep pass** touched only ~108
edges and did **zero** merges, dormancy, or archival. That is not deep sleep failing — it
is deep sleep finding **nothing to do**, because the live store is deep-slept on a regular
cadence. Steady-state consolidation is inert *because the work was already done on prior
nights.* This is the strongest single argument for the abstinence experiment.

## Pre-registered follow-up: the two-week abstinence window

This result does **not** answer "does consolidation help retrieval?" in general — it
answers it **for the saturated, regularly-slept regime** (answer: not measurably). The
registered next arm, declared here so it reads as planned rather than post-hoc:

> **Abstinence experiment.** Suspend sleep on the live store for ~2 weeks so unconsolidated
> memories accumulate, then fork and run the identical A-vs-slept protocol. This is the
> only condition where consolidation has real work (merges, dormancy, new typed edges at
> volume), and therefore the only condition that can show a consolidation retrieval effect
> if one exists. Re-run against the **deployed V6 reranker** (not the formula) once it lands,
> to also get the sleep-vs-reranker measurement this fork could not.

## Honest caveats

- Sleep's LLM steps are nondeterministic; single fork per condition (no variance repeats —
  wall-clock was spent on the deep arm and the strip decomposition instead, per the
  amended plan). Effect sizes this small could shift within sleep's own run-to-run noise;
  the *qualitative* conclusion (inert consolidation, saturated store) is robust to that.
- Formula scorer, not the reranker (see above).
- The strip that produced the standard-consolidation-only arm removed the probe across all
  three channels it reaches the formula through — feedback events, the 30-day hebbian
  co-retrieval window (`retrieved` events), and edge weights (shared-edge weights restored
  to A; 31 probe-created edges deleted; 359 probe-bumped consolidation-edge weights reset to
  sleep-creation values). Verified: feedback and retrieved counts back to A exactly, 0
  probe-created edges remaining. See `strip_probe.py` / `verify_strip.py`.
- GT relevance is per-(query, memory); sleep-created memories not in GT correctly score 0.
  No re-judging was done (that is a separate interactive experiment).

## Reproduction

```sh
# worktree + two copies of the live store (per-arm copies forked from A-frozen)
git -C ~/repos/somnigraph worktree add -b exp/sleep-fork ~/repos/somnigraph-sleepfork HEAD
# copies on D: A-frozen, B-slept, B2-deep, B-noprobe (each = live ~/.claude/data at setup)

# Home machine: OpenAI 1536d store -> leave SOMNIGRAPH_EMBEDDING_BACKEND unset.
# Windows: export PYTHONUTF8=1 (sleep.py prints box-drawing chars; cp1252 crashes without it).

# Baseline (frozen A). Place A/memory.db at the snapshot path first, run WITHOUT --snapshot.
cp <A>/memory.db ~/.somnigraph/benchmark/production_snapshot.db
SOMNIGRAPH_DATA_DIR=<A> uv run scripts/locomo_bench/eval_retrieval.py \
  --dataset production --configs formula --gt-path <A>/tuning_studies/gt_calibrated.json \
  --recall-budget 5000 --recall-limit 20 --output <A>/eval_baseline.jsonl

# Standard sleep (full cycle, default probe) on B; deep on B2 (--deep --probes 0).
SOMNIGRAPH_DATA_DIR=<B>  PYTHONUTF8=1 uv run scripts/sleep.py
SOMNIGRAPH_DATA_DIR=<B2> PYTHONUTF8=1 uv run scripts/sleep.py --deep --probes 0

# Slept evals: swap the snapshot to each store's memory.db, re-run the eval above
# (same GT file = A's), output to <store>/eval_slept.jsonl.

# Standard-consolidation-only: copy B -> B-noprobe, strip the probe, re-eval.
python experiments/sleep-fork/strip_probe.py      # removes probe across all 3 channels
python experiments/sleep-fork/verify_strip.py     # asserts feedback/retrieved==A, 0 probe edges

# Deltas + attribution (each arm vs A; probe increment = B vs B-noprobe):
uv run python experiments/sleep-fork/analyze_sleep_fork.py \
  --baseline <A>/eval_baseline.jsonl --slept <arm>/eval_slept.jsonl \
  --gt <A>/tuning_studies/gt_calibrated.json --db-a <A>/memory.db --db-b <arm>/memory.db \
  --out sleep_fork_results_<arm>.json
```

Result JSONs: `sleep_fork_results_{standard,standard_consolidation_only,deep_consolidation,probe_increment}.json`.

## Sleep run facts

- Standard cycle (B): 33 m total — NREM 236 s, REM 585 s, probe 903 s (30 queries, opus). 420 feedback events injected.
- Deep cycle (B2): 18 m 47 s — NREM 303 s, REM 650 s, probe skipped. NREM(deep) over all 1165 active memories completed in ~2 classification batches — few *new* pairs to classify because most similar pairs already carry edges (saturation).
- Both: 0 dormancy, 0 merges, 0 archival, 0 new summary memories; 15 summaries refreshed.
