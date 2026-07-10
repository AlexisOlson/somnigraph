# 2026-07-10 — Sleep-bench WP6: feature-structure study of the sleep-aware reranker

Third arc of 2026-07-10 (after the whole-corpus run and the H10 2×2). Alexis's protocol:
random-subset sweep → importance-instability map → iterative **registered** hypothesis
batches until no sensible tests remain, explicitly *not* stepwise selection. Ran to
declared exhaustion: 170 c2 masks + a frozen mini-sweep across Stage A, Stage C1/C2, and
four registered batches, every one pre-registered with predictions in
[`experiments/sleep-bench/wp6-prereg.md`](../../experiments/sleep-bench/wp6-prereg.md).
Results: [`experiments/sleep-bench/findings-wp6-feature-structure.md`](../../experiments/sleep-bench/findings-wp6-feature-structure.md).

## What shipped

- **Rig:** `sweep_2x2_features.py` (resume-safe JSONL sweeps, per-fold gc after a
  LightGBM `bad allocation` crash at mask 119, `--full-mask` arbitrary-subset masks,
  seeded `noiseK` control columns, matrix disk cache: 90s → 2s warm start) +
  `analyze_wp6.py` (structure map; excludes full-mask records from toggleable stats).
- **Findings doc** with the headline structure map and deployment implications.
- **Prereg ledger** as the canonical verdict record — 13 hypotheses/predictions scored.

## Headline findings

1. **Flat because redundant, not irrelevant.** The #1-gain feature (vec_rank) costs only
   −0.0106 to remove; nearly every signal has 2–3 encodings. One drop > 0.02 in 170 masks.
2. **The vec channel is the model**: −0.2006 to remove (12× the next effect). FTS −0.0173;
   score magnitudes beyond ranks ~0.0045 total; per-query constants = gain bookkeeping.
3. **Sleep-feature signal is real but masked, and doubly encoded.** +0.006 on the rich
   core vs +0.0123 on a 3-rank minimal core; decomposition: token_count 60%, credit pool
   82%, graph cluster 78% (pool/graph statistically tied) — two parallel ~80% encodings
   of "synthesis row." Neither C1's pool-led nor H10's graph-led read is the structure.
4. **token_count is a generic content signal, not a summary detector** (frozen knockout
   −0.0126 vs c2 −0.0044 — 3× larger where there are no summaries).
5. **Orphaned flags mislead** (pre-registered H-A inverted): full-pool removal −0.0016 <
   token_count alone −0.0044.
6. **Capacity story dead; junk columns cost generically** (~0.0012/col, equal damage to
   rich and weak models) → the ±0.003 column-width confound and the >0.004 reading rule.
7. **theme_rank + proximity are dead on the slept store** (schema22 prune ≥ all_on);
   theme_rank's 472→8.6 gain collapse is store composition (frozen gain stays 109–141).

## Surprises

- **Batch 4's refutation was the most informative result of the arc**: predicted graph
  lowest (alternate-encoding story), got graph tied with pool at ~80% each. The two
  contradictory prior reads (C1 pool-led CV, H10 graph-led importances) turned out to be
  projections of one doubly-encoded signal cluster.
- The frozen store flipped N1 emphatically — token_count's knockout was *3× bigger* where
  the summary-detector story predicted zero.
- 11 all-constant columns moved a score by 0.0031 (colsample dilution) — an instrument
  confound discovered mid-study, quantified with seeded-noise controls (fm_all_noise4),
  and promoted into a reading rule rather than left as noise.
- LightGBM native-heap creep produced a `bad allocation` at mask 119/130 on the 8 GB
  pagefile; per-fold `del model; gc.collect()` fixed it; resume machinery preserved all
  119 completed masks.

## Caveats

- One corpus (whole-store c2), one recipe (V6-parity pointwise LightGBM), coverage+
  evidence labels; effect sizes travel, orderings may not.
- Most single-feature effects live inside the ±0.003–0.004 confound zone and are only
  bounded, not measured.
- These labels can't price redundancy suppression (h12_out ≈ all_on ≠ "H12 useless").
- **N4 remains open**: core_only_post conv0 cell (does core recalibration alone capture
  ≥80% of the ④−② judged gap?) — registered, needs a training run + explicit go.

## Files touched

- `experiments/sleep-bench/sweep_2x2_features.py` (new, evolved through the arc:
  gc fix `87fde6a`, full-mask+noise `b94a842`, matrix cache `e02ea92`)
- `experiments/sleep-bench/analyze_wp6.py` (new)
- `experiments/sleep-bench/wp6-prereg.md` (ledger: Stage A/C registrations + batches 1–4
  + exhaustion declaration)
- `experiments/sleep-bench/findings-wp6-feature-structure.md` (new)
- Data (gitignored): `results-full/wp6_sweep.jsonl` (170 masks),
  `wp6_sweep_frozen.jsonl`, `matrix_cache_*.pkl`, `wp6_structure.json`

## What's next

- N4 on Alexis's go (the one remaining conv0-touching test).
- Deployment prune (schema22) available whenever the bench schema is next touched.
- Pre-fork production sleep-path review booked ~07-14; live fork ~07-15 (sleep, then
  retrain reranker per H10 guidance).
