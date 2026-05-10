# 2026-05-09 — V5+1 200-query mixed probe + retrain; per-memory pin cap; adversarial scaffolding

V5+1 200-query mixed probe + retrain; per-memory pin cap (4 default, 8 for pathology) replaces the old probe/real ratio cap; pathology audit script gains JSON output mode and a new `select_pathology_targets.py` selector wires `--mix FLOAT` adversarial-first orchestration into `probe_recall.py`; `evaluate_reranker_ranking` gains a non-miss NDCG/R@10 variant alongside the existing miss-inclusive metric.

## V5+1 probe results

`--mode mixed --queries 200 --coverage 0.5`: 200 queries / 50 groups planned, **50/50 bundled craft success** (0 fallbacks), exact `{natural: 50, mild: 100, hard: 50}` distribution. Target row LLM rating distribution: **median=0.95, p10=0.95, p90=0.95, 0/193 below 0.5** — tighter than V4's median=0.95/p10=0.60 across 25× the sample. Phase-1 misses: 6/200 (3%).

Asymmetric orthogonality framing in `CRAFT_TEMPLATE_BUNDLED` is biting at scale — spot-check on the printed Group 50 confirmed both mild queries attacking the same target from genuinely different angles (scope-vs-symptom on an ACWI-ex-AI memory, ranks 1/15, 1/15, 3/15, 2/15 across nat/mild#1/mild#2/hard).

## V5+1a re-emit

GT 1284 → 1488 queries (+200 V5+1 + 4 fresh live + 2 new memories' synthetic anchors). All 200 V5+1 probe_target events captured; 195 collisions kept ≥0.9, 5 max-merged, 6 misses added — only 8 net-new (q, mid) pairs (rest of the V5+1 events folded into existing real-feedback labels written in the same probe call). Held-out probe-hard count grew 89 → 139 (statistical power: noise floor ±0.005 → ±0.0042).

## V5+1b retrain at boost=5.0

Aggregate Reranker NDCG=0.8763, vs RRF +0.0849 (V5b: 0.8691, +0.0821) — apples-to-apples win on the larger GT, +0.0072 NDCG. RMSE 0.0275 vs V5b 0.0268 (within noise). **Held-out probe-hard NDCG=0.9026, R@10=0.9540** on 139 queries, vs V5b's 0.9197/0.9842 on 89 — headline NDCG -0.0171, R@10 -0.0302.

**The held-out hard headline drop is composition, not regression.** Math: V5b had ~3 Phase-1 misses / 89 (3.4%); V5+1b has 9/139 (6.5%, V5+1 added 6). Misses score NDCG=0 mechanically. Excluding misses, V5b non-miss mean ≈ 0.9518; V5+1b non-miss mean ≈ 0.9651. **Non-miss-only hard NDCG actually IMPROVED +0.0133.**

Cross-mode R@10 also dropped (mild -0.011, natural -0.013) — those modes aren't held out, so the drop reflects the new V5+1 bundled-orthogonal queries being intrinsically harder to recover at top-10 than V4's per-mode-crafted queries. The asymmetric mild-orthogonality requirement produces queries that sit further from their targets in retrieval space *by design*.

## Non-miss NDCG metric (new)

`evaluate_reranker_ranking` patched (~+30 lines): the per-mode loop now computes both miss-inclusive (`ndcg_5k`, `recall_10`) and non-miss-only (`ndcg_5k_non_miss`, `recall_10_non_miss`) variants. Phase-1 miss defined as "query has at least one positive-label memory but none of those positive memories appear in the ranked candidates." Per-mode summary printout now reads `NDCG=X.XXXX  R@10=X.XXXX  |  non-miss NDCG=X.XXXX  R@10=X.XXXX  (miss N/total=X.X%)`.

## Per-memory pin cap rewrite

Replaced `MAX_PROBE_RATIO = 2.0` heuristic (probe-feedback rows / real-feedback rows > 2.0 → skip) with explicit `MAX_PINS_PER_MEMORY = 4` against `event_type='probe_target'` count. Old logic was backwards: a memory with rich real-recall could absorb 100 probe pins before being skipped, while a dark memory with no real-recall got skipped after 1 — exactly inverting which memories most need probing. New cap counts pins specifically and treats every memory equally. Soft selection weight simplified from `1/(1+probe/real)` to `1/(1+pins)`.

Display fix in `run_probe`: the prior "N underserved memories" headline summed `len(g["underserved_ids"])` across themes, double-counting memories that appeared in multiple underserved themes — produced misleading 5×-inflated counts (V4 reported 2392 against 470 active memories). Now reports `unique_underserved` (deduped) alongside the pre-dedup `pair_count`.

Smoke against production DB at the time of the change: 365 unique underserved, 28 already at cap=4, 337 still eligible — coverage saturation is closer than the old display implied (~6-7 more V5+1-sized passes saturates the underserved tail at 4 pins each, ~1500-2000 cumulative probe queries).

## Adversarial probing scaffolding (5 file changes)

`audit_reranker_pathology.py` gained `--output PATH` (`""` auto-names to `data/pathology_audits/audit_<timestamp>.json`); pathology dicts now carry full `memory_id` (was 8-char prefix); JSON write moved before histogram so cp1252 console encoding can't kill the data; histogram switched to ASCII (was crashing on `█`/`∞`).

New `scripts/select_pathology_targets.py` (~120 lines) loads latest audit JSON, applies `MAX_PATHOLOGY_PINS = 8` cap (gives feature-ceiling memories two bundles before declaring it unfixable), Efraimidis-Spirakis weighted sampling biased by gap size and softened by `1/(1+pins)`. `probe_recall.py` gained `--mix FLOAT` (default 0.0 = pure coverage-fill); the orchestrator picks adversarial-first (the scarcer signal — typically tens, not hundreds), then coverage backfills any unfilled adversarial budget into the same total query count. `select_targets` gained an `exclude` set so a memory picked by adversarial doesn't double-pick in coverage.

Three smokes pass: `--mix 0.0` no behavior change vs prior selector; `--mix 0.5` with no audit JSON falls back to coverage-only with a helpful log line; `--mix 0.5` with a tiny test audit JSON correctly picks the worst pathology first then coverage-backfills.

Per-memory cap layered: pathology-flagged memories cap at 8 (allows 2 bundles of adversarial pins); non-pathology memories stay at 4. Pathology-cap-reached + still-on-list = "feature investigation candidate."

## Design decisions NOT taken

(1) no auto-audit-after-train hook in `train_reranker.py` — keeps retrains fast; the audit (~3-5 min on 470 memories) runs as a manual step. (2) no persistence ledger tracking "audit count this memory has been on the pathology list" — the pin cap=8 already prevents infinite probing of unfixable memories.

## Surprises

Framed three plausible V5+1b outcomes (held-out hard up, flat, or down). The actual outcome was a fourth shape — *headline NDCG down, underlying model quality up*. The Phase-1 miss rate composition shift dragged the miss-inclusive metric below V5b's despite the model genuinely fitting non-miss queries better. Without splitting miss-inclusive vs non-miss, this would have read as "Phase 3d failed" — instead it's "Phase 3d works AND Phase-1 miss rate is now the dominant noise source on the held-out NDCG metric." The non-miss patch was hot-shipped after seeing this.

## Caveats

Phase-1 miss rate accumulating at ~3% per probe pass means the miss-inclusive metric will diverge further from non-miss as more probes land. Cross-retrain comparisons of held-out hard NDCG should now lead with non-miss NDCG and treat miss-inclusive as a candidate-pool diagnostic. The 8 net-new (q, mid) pairs from V5+1's 200 events confirms again that for already-served memories the pin folds into existing labels.

## Reversibility

`--mix 0.0` reverts the orchestrator to pure coverage-fill (default unchanged); replace `MAX_PINS_PER_MEMORY = 4` with the old probe/real ratio block to revert the cap; the non-miss NDCG block is additive (the existing miss-inclusive numbers are unchanged) so removing it just drops the new column from the printout.

## Files touched

`scripts/probe_recall.py` (pin-cap rewrite + display fix + `--mix` CLI + adversarial-first orchestration; ~+80 lines net), `scripts/audit_reranker_pathology.py` (~+30 lines: `--output`, full memory_id, JSON-before-histogram, ASCII histogram), `scripts/select_pathology_targets.py` (~120 lines, new file), `scripts/train_reranker.py` (~+30 lines: non-miss NDCG/R@10 in per-mode evaluation).

## Phase 3 progress

Adversarial scaffolding ready (audit → pathology JSON → selector → mix flag → probe). V5+2 (a 200-query `--mix 0.5` pass after refreshing the audit) is the next concrete experiment — first adversarial probes against memories the V5+1b model currently mis-ranks. Phase 3 priority B (candidate-pool widening so Phase-1-miss rows contribute training gradient) moved up the priority list — at 6.5% of the hard holdout, miss rate is now the dominant headline-NDCG noise source.
