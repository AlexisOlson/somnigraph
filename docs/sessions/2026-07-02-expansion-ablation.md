# 2026-07-02 — LoCoMo expansion-method ablation (roadmap #21)

Measurement-only ablation of the six candidate-expansion methods in the L5b retrieval
pipeline, plus an orchestrator review pass that corrected the framing. Full findings and
tables: [`experiments/expansion-ablation/findings-expansion-ablation.md`](../../experiments/expansion-ablation/findings-expansion-ablation.md).

## Verdict

Roadmap #21 answered — but not the way the "3 of 6 dead methods" framing predicted.
Removability is **not settleable on this benchmark**: at the current 4000 Phase-1 search
limit the candidate pool holds the entire benchmark DB (≤861 memories), so **all six methods
add zero net-new candidates**, and the `--expand-all` lift (+4.9pp R@10, +12.4pp multi-hop) is
entirely the **Phase-2 rerank pass**. Methods stay in code; this expansion machinery is
bench-harness-only, not the production recall path.

## Surprises

- **The cited fire rates were a stale instrument.** `HANDOFF.md`/roadmap #21 list session 100%,
  keyword 95%, entity_bridge 96%, rocchio 0%, etc. Those date from the ≤200 search-limit era
  (pool < DB). The Level 5 change to a 4000 limit made the pool ⊇ the whole benchmark DB and
  silently drove net-new candidate expansion to 0% for every method — and nobody re-measured.
  The naive "confirm the 3 dead methods removable" task was answerable only by discovering that
  *all six* are inert here, for a structural reason the benchmark can't see past.
- **Arm g is the decisive control.** Forcing Phase 2 on with zero expansion methods
  (`SOMNIGRAPH_FORCE_PHASE2`) reproduces `--expand-all` exactly (R@10 95.4%, positional diff 3)
  and diverges from Phase-1-only on 827 records. The mechanism is proven, not inferred.

## The overclaim-correction story (methodological, for future sessions)

The first draft claimed arms b–f were "mathematically identical." They are not. A
`(conv_id, question)`-keyed diff had collapsed duplicate question strings and reported "0
differences"; a **positional** comparison showed arms differ by 1–7 of 1,977 records. The fix
was to measure the noise floor: **three identical `--expand-all` runs differ from each other by
5–9 records**, so between-arm differences are within the pipeline's own non-determinism (likely
vec-ANN k-NN tie ordering in Phase 1). Softened the claim to "indistinguishable within the
noise floor." A future session that sees small diffs in this harness should recall this: 5–9
records of run-to-run drift is normal; only differences above that floor are real.

## Read-only isolation

The eval's schema-init opens each conversation DB in write mode (idempotent `CREATE ... IF NOT
EXISTS`). To keep the canonical benchmark DBs untouched, the 10 per-conversation DBs were copied
to scratch and `base_dir` redirected via a `SOMNIGRAPH_BENCH_DIR` env override. Source DB mtimes
verified unchanged (2026-03-27) after the full sweep. Reference arm `--expand-all` reproduced the
documented L5b numbers exactly (HARD-STOP gate) before any conclusions.

## Files touched

- `scripts/locomo_bench/config.py` — `SOMNIGRAPH_BENCH_DIR` redirect (read-only DB isolation).
- `scripts/locomo_bench/eval_retrieval.py` — env-gated, metric-neutral scaffolding:
  `SOMNIGRAPH_FIRE_STATS` (net-new fire-rate instrumentation) and `SOMNIGRAPH_FORCE_PHASE2`
  (arm-g toggle on the line-492 Phase-2 gate).
- `experiments/expansion-ablation/` — findings + `ablation_results.json`, `fire_stats_full.json`,
  `arm_g_forcephase2.jsonl`.
- `docs/benchmarks.md` (§ Expansion-method ablation), `STEWARDSHIP.md` (P5 + changelog),
  `docs/roadmap.md` (#21), `scripts/locomo_bench/HANDOFF.md` (staleness note).

## Reversibility

All code changes are env-gated and off by default — production behavior is unchanged. No
expansion method was removed. Nothing pushed; the arc merges via `--no-ff` after doc review.

## What's next

Sleep-pass and feedback-loop ablations remain (P5). If a benchmark test of candidate expansion
is genuinely wanted, it needs a pool smaller than the corpus (e.g. a deliberate ~200-limit run).
