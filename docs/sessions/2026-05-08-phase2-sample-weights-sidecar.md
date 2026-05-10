# 2026-05-08 — Phase 2 of pinned-target labels — sample-weights sidecar + per-mode metrics + held-out hard

`build_gt_from_feedback.py` now emits `gt_feedback_sample_weights.json` alongside the GT: per-query `{weight, mode, source, holdout}` plus a metadata block (schedule, holdout_modes, by_source counts).

## Mode classification and schedule

Mode classification follows live > probe-{hard/mild/natural/extra} > synthetic-self-anchor priority — real usage trumps probe origin trumps synthetic. Default schedule: live=1.0, probe-natural=1.0, probe-mild=1.2 (slight up-weight, hardest non-adversarial discriminator), probe-hard=0.0 (held out), probe-extra=1.0, synthetic-self-anchor=0.5 (May-8 Goodhart mitigation, half-weight not zero — value to be measured by ablation).

Holdout list is structural (`["probe-hard"]`), independent of the schedule weights so a future ablation can train on hard rows without losing the holdout flag. CLI: `--weight-schedule` JSON + per-mode `--weight-{live,natural,mild,hard,extra,synthetic}` overrides + `--no-sample-weights` opt-out + `--sample-weights-output`. Sidecar deterministic (sort_keys=True, byte-identical across runs verified).

## Trainer integration

`train_reranker.py` consumes the sidecar via `--sample-weights PATH` + `--override-weight MODE=VALUE` (repeatable, no re-emission needed). Per-row weights/modes broadcast in `extract_all_features`; `lgb.fit(sample_weight=...)` + `eval_sample_weight=[...]` threaded through `train_and_evaluate`, `train_lambdarank`, `evaluate_reranker_ranking`, `evaluate_two_stage`, and `ablation_no_feedback`. Per-mode RMSE computed inside `train_and_evaluate` and per-mode NDCG@5k + R@10 computed inside `evaluate_reranker_ranking` (every val query partitioned by mode, held-out modes still scored — that's the point).

New `scripts/.../reranker_results.json` captures headline metrics + `per_mode_metrics` block + `sample_weights_metadata` for diff-over-time tracking. `--train-only` re-derives per-row weights from the cached `memory_ids` so the sidecar can change without forcing a fresh extraction.

## Sidecar build results

Sidecar build against the production DB (`SOMNIGRAPH_DATA_DIR=~/.claude/data`): 1,257 queries classified — live 543, probe-natural 79, probe-mild 84, probe-hard 83 (holdout), synthetic-self-anchor 468; 83 zero-weighted rows match the holdout count exactly. Loader smoke + override smoke pass; sidecar idempotent.

Held-out hard NDCG/R@10 baseline + synthetic-ablation deltas pending — those need a `train_reranker.py --sample-weights ...` run, which belongs in Alexis's terminal (multi-minute, see `feedback_dont_run_long_commands.md`).

## Caveats

Aggregate NDCG should move ≤ 0.02 — the value here is *separation* (held-out hard becomes a number future feature work can move), not a higher number. Phase-1-miss probe rows still won't influence the gradient until candidate-pool widening (Phase 3); they're in held-out eval but contribute 0-NDCG until they're surfaced to the reranker. Synthetic 0.5 is a guess; the ablation `--override-weight synthetic-self-anchor=0` produces the defensible number.

## Reversibility

Sidecar is opt-in, `--no-sample-weights` reverts emission, omitting `--sample-weights` reverts training, no DB / event-schema changes.

## Surprises

Between idempotency runs ~5 minutes apart, the live `live`-mode count moved 543→544 because the production DB is mutating in real time — a built-in reminder that "byte-stable" is a property of the *script over fixed input*, not "byte-stable across days."

P2 progress: pinned-target Phase 2 plumbing complete; held-out hard metric baseline + synthetic ablation are the next concrete numbers. Phase 3 (per-(q,m) weighting + candidate-pool widening so Phase-1-miss rows train, not just evaluate) picks up next.
