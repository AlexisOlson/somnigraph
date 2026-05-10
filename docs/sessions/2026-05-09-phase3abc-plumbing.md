# 2026-05-09 — Phase 3a plumbing + probe pipeline tightening (3b + 3c)

Three changes that compound, all wired but trainer ablations + LLM probe deferred to Alexis's terminal (multi-minute, see `feedback_dont_run_long_commands.md`).

## 3a — per-(q,m) weighting

Sample-weights sidecar bumped to schema_version=2, every entry now carries `pinned_target_id` (full memory_id, null when no canonical answer). Wired through `add_probe_target_gt` (set per probe_target event), `_backfill_probe_targets_from_legacy` (set per resolved legacy row), and `add_synthetic_self_anchor_gt` (set to the memory whose summary is the context).

Resolution at sidecar build time: live → null, probe-* → unique probe/backfill mid (warn + null on disagreement), synthetic-self-anchor → the synthetic mid. Sidecar metadata now reports `pinned_boost_default` (2.0), `queries_with_pinned_target`, `pinned_ambiguous_probe`, `pinned_ambiguous_synthetic`.

`train_reranker.py` gained `--pinned-boost FLOAT` (default 2.0); `extract_all_features` accepts `query_pinned_targets` + `pinned_boost` and applies the multiplier per row where `mid == pinned_target_id`. `--train-only` re-derives per-row weights with the boost from cached `(qtext, mid)` pairs, so an ablation sweep across `--pinned-boost {1.0, 2.0, 5.0}` doesn't force feature re-extraction.

`load_sample_weights_sidecar` returns 4-tuple now (added `query_pinned_targets`). `--pinned-boost` value is recorded in `sample_weights_metadata.pinned_boost_applied` so the results JSON tracks which boost each run used.

## 3b — rate rubric (no target reveal)

The `Target memories (what the query was designed to find): {target_ids}` line was leaking the answer to the LLM rater and biasing neighbor labels ("less relevant than the target"). Removed from `RATE_TEMPLATE`; `target_ids` kept in `rate_results` signature for post-hoc target-row stat collection only.

New stat printed at end of `run_probe`: target row LLM rating distribution (median, p10, p90, count<0.5) — a clean honesty signal now that the rater is blind. Pinned 1.0 label still flows via probe_target events + `_apply_probe_pin` dedup (≥0.9 keep, <0.5 overwrite, 0.5–0.9 max-merge), so target labels are unaffected by the rubric change.

## 3c — mode budget shift

`_modes_for_group` cycle changed from `(natural, mild, hard)` 1/3 each to `(mild, natural, mild, hard)` — 50% mild / 25% natural / 25% hard. Mild is the highest-EV training signal (in-distribution, weight 1.2, paraphrased verbs preserving distinctive IDs); natural is a sanity floor; hard is held out and structurally bounded by candidate-pool blindness. Help text and `--mode mixed` description updated.

## Validation results

V1+V2 done: sidecar build against production DB (`SOMNIGRAPH_DATA_DIR=~/.claude/data`) — 1,273 queries (545 live, 84 natural, 89 mild, 87 hard [HOLDOUT], 468 synthetic), **728 with pinned_target_id** (260 probe + 468 synthetic), 545 live unpinned by design, 0 ambiguous in either source. Two consecutive runs produced byte-identical sidecar (sha256 match). Schema-version=2 verified, pinned shape/coverage spot-checked across all 5 modes.

## Deferred to Alexis's terminal

**(a) V3 ablation:** `uv run scripts/train_reranker.py ... --pinned-boost 1.0` (then 2.0, then 5.0). Boost=1.0 should match Phase 2's 4e (NDCG=0.9146) within ±0.005. Boost=2.0 vs 1.0 is the hypothesis test. Boost=5.0 checks for over-pinning (aggregate RMSE jumping >0.01 = destabilizing).

**(b) V4 probe smoke:** `uv run scripts/probe_recall.py --queries 15 --coverage 1.0`. Expect ~8 mild / 4 natural / 3 hard (50/25/25 with rounding), 15 `probe_target` events written, and the new "Target row LLM rating distribution" line at the end with median ≥ 0.7.

**(c) V5 end-to-end:** re-emit GT to capture the new probes, then `--pinned-boost 2.0` retrain.

## Caveats

Default 2.0 is a starting guess, V3 ablation produces the defensible value (lock it into `pinned_boost_default` afterward). 3b will produce different LLM neighbor ratings going forward; backfilled probe rows still carry the contaminated rubric and are frozen in `memory_events`. 3c shifts probe corpus composition over time toward mild-heavy; held-out hard set will grow more slowly (run an explicit `--mode hard` probe occasionally if hard generalization needs faster signal).

## Reversibility

`--pinned-boost 1.0` reverts 3a; restoring the `Target memories: {target_ids}` line reverts 3b; restoring the 3-element cycle reverts 3c. Schema_version=2 is forward-compatible (older trainers ignore the new field).

## Surprises

The V1 backfill produced 0 newly-active probe-pinned pairs — every probe-mode query already had a real-feedback row that the existing `_apply_probe_pin` dedup folded the pin into; the new column adds value for *future* probes and for the V3 boost mechanic, not for retroactive coverage.

## Files touched

`scripts/build_gt_from_feedback.py` (+~80 lines, schema v2 + pinned tracking across three sources), `scripts/train_reranker.py` (+~25 lines, --pinned-boost CLI + per-row apply on both fresh-extract and --train-only paths), `scripts/probe_recall.py` (+~40 lines, mode cycle + rate rubric + target-row stat collection).
