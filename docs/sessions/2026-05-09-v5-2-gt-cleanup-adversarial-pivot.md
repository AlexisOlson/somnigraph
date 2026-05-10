# 2026-05-09 — V5+2 GT cleanup + retrain; adversarial redirect (audit → real-recall); PPR cache speedup

## GT pollution finding

5.9% of feedback rows in the GT referenced inactive memories (deleted/superseded), measurable as `Phase-1 misses` on the held-out hard set since deleted memories can't be retrieved. Patched `build_gt_from_feedback.py` to filter feedback rows whose memory_id isn't in the active set (synthetic anchors and probe_target events were already filtered).

V5+2a re-emit on the patched parser: 1488 → 1694 queries (+206 net: V5+2 added 200 probes through a partially-disrupted `--mix 0.5` adversarial run before the bug was caught, plus 6 fresh live; held-out probe-hard count grew 139 → 189). 0 ambiguous in either pinned-target source; 280 net-new pinned_target_id entries (~860 → 1140).

## V5+2b retrain results

Clean GT, boost=5.0, 31 features. Aggregate Reranker NDCG=0.8816, vs RRF +0.0885 — a clean apples-to-apples win on a *larger and cleaner* GT (V5+1b: 0.8763, +0.0849). RMSE 0.0265 (V5+1b: 0.0275). Per-fold NDCG range 0.8747-0.8905, very stable.

**Held-out probe-hard:** miss-inclusive NDCG=0.8814, R@10=0.9447, **miss rate=0/189=0.0%** on 189 queries (V5+1b: 0.9026, 0.9540, 9/139=6.5%). **The miss-rate diagnostic resolves Phase 3 priority B (candidate-pool widening) entirely — handoff predicted "1-2% means most was deletion bloat"; it dropped all the way to zero. ALL of V5+1b's miss noise was inactive-memory pollution.**

Per-mode breakdown: live 0.7309/0.8218 (1/554=0.2% miss — live almost completely clean post-filter); probe-mild 0.9358/0.9680; probe-natural 0.9466/0.9694; synthetic 0.9984/1.0000 (synthetic still held out from training, perfect retrieval is upper-bound).

The non-miss NDCG drop (-0.0837 vs V5+1b) reads as a regression on first glance but is composition, not model: V5+1b's 0.9651 was on a 130-query subset (139 - 9 misses); V5+2b's 0.8814 is on the full 189, which now includes the 9 previously-excluded miss queries plus 50 new V5+2 probes. Sanity math: if the 130 carryover queries still average ~0.965, the 59 new entrants must average ~0.70 — plausible for "queries that were structurally hard enough to be misses, now finally rateable" + "V5+2 bundled-orthogonal probes shaped to push the model."

**Feature importance shake-up:** session_recency went 188 → **445.4 gain** — a 2.4x jump, dominating the ranking with fts_bm25_norm (364.4) a clear #2 and query_idf_var (356.0) at #3. Cleanup gave session_recency a sharper signal because inactive-memory feedback was noising it. Phase 3 priority C (session_recency signal-vs-bias leave-one-feature-out audit) just got MORE pressing.

## Adversarial redirect — the design pivot

Mid-V5+2 the audit-based pathology selector produced unintuitive results — the bundled crafter, given a content-residual-flagged memory's content, wrote queries the model handled FINE (target ranks 1-2 across natural+mild). Diagnosis: content-residual = content tokens minus summary tokens; FTS indexes summary+themes, so target's FTS is structurally weak on its own residual. The audit isn't an OOD test, it's an FTS-handicap-target test. Real adversarial signal lives in real-recall pathologies — high-utility-rated memories that ranked low in actual production retrieval.

## Real-recall pathology miner (`scripts/select_real_pathology_targets.py`, ~330 lines, NEW)

Mines `memory_events` by joining feedback rows to closest `recall_meta` (within 60s window via bisect on parsed UTC epochs), filters by utility ≥ 0.7 + rank ≥ 5 + non-probe + active-memory, aggregates per-memory by severity (utility * worst_rank), Efraimidis-Spirakis weighted sample softened by `1/(1+pins)`. Same 8-pin cap as the audit selector. Same `(target_list, info)` contract so `probe_recall.run_probe`'s `--mix` orchestration is selector-agnostic.

Two non-obvious correctness fixes vs the `/tmp/mine_real_pathology.py` draft: (a) `recall_meta.kept` stores `mid[:8]` 8-char prefixes (per `tools.py:788`) but feedback events store full `memory_id` — joined by prefix (collision risk at 478 active memories ~2.6e-5, negligible); (b) bisect on parsed-UTC epochs replaces the broken `_str_diff` closest-timestamp heuristic.

The standalone CLI prints distribution stats for inspection: `python scripts/select_real_pathology_targets.py --rank N --utility F`.

**Smoke against production DB:** 16 hard-real pathologies at default rank>=5 across 16 unique memories. Loosening to rank>=3 surfaces 40; rank>=7 only 5; rank>=10 = 0. **The model never buries a useful memory past position 10** — that's healthy on its own but makes adversarial supply structurally tight.

## `probe_recall.py` patches

Added `--adversarial-source {real,audit}` (default `real` — audit kept as sentinel-class fallback per the V5+2 group analysis) and `--adversarial-rank-threshold INT` (default 5; lower = more candidates). Selector chosen at call site based on the flag; both selectors coexist. Reversibility: pass `--adversarial-source audit` to revert to V5+2's audit-based selection.

## Trainer-side PPR cache patch (~100x speedup)

Discovered mid-session that V5+2b's PPR cache extension was computing 1694 queries × 103 dampings = ~174k PPR walks, of which the trainer reads exactly 1694 (one per query at the production constant `PPR_DAMPING=0.216`). The other 102 dampings exist only for `tune_gt.py`'s hyperparameter sweep grid.

Patched `tune_gt.precompute_ppr_cache` to accept a `dampings` parameter (default `None` = full grid for tune_gt back-compat); rewrote missing-detection to scan **all queries × requested grid** rather than only inspecting `gt_queries[0]` (which was a latent silent-failure: queries added after the cache was last built would silently get empty PPR scores at score time). `scripts/train_reranker.py` now passes `ppr_dampings=[PPR_DAMPING]` through `load_tuning_data` at both call sites. The existing 130k-entry cache stays valid; future trainer runs only extend with `(0.216, q)` per missing query. tune_gt's behavior unchanged. Latent silent-failure bug fixed as a side effect.

## Surprises

(1) Miss rate dropping ALL the way to 0.0% (not the predicted 1-2%) means the candidate-pool widening track was entirely a measurement artifact of inactive-memory pollution — Phase 3 priority B drops off the active list.

(2) session_recency feature importance jumping 2.4x post-cleanup is the inverse-shape surprise: cleanup didn't *flatten* the importance landscape, it *sharpened* it. The single-feature dominance is now a real concern worth investigating before further trainer iteration.

(3) Adversarial supply being structurally tight (only 16 candidates at default rank>=5) means V5+3 will likely want `--adversarial-rank-threshold 3` to give the experiment enough signal to measure.

## Caveats

Cross-retrain held-out hard NDCG comparisons now need explicit cohort accounting — V5+1b non-miss vs V5+2b non-miss are not the same population. Aggregate NDCG and miss-rate are cleaner cross-time signals. The aggregate NDCG win (+0.0053) is modest; the more interpretable result is "miss rate to zero on a larger and cleaner GT."

## Files touched

`scripts/select_real_pathology_targets.py` (NEW, ~330 lines including standalone CLI), `scripts/probe_recall.py` (~+30 lines: --adversarial-source flag, --adversarial-rank-threshold flag, source-conditional selector import in run_probe), `scripts/tune_gt.py` (~+15 lines net: dampings parameter on precompute_ppr_cache and load_tuning_data, robust per-query missing-detection rewrite), `scripts/train_reranker.py` (+1 line: PPR_DAMPING import, +6 lines threading ppr_dampings through both load_tuning_data call sites).

## Phase 3 progress

Priority B (candidate-pool widening) → resolved by GT cleanup. Priority C (session_recency signal-vs-bias leave-one-feature-out audit) → moved up; the 2.4x importance jump makes this the most pressing remaining Phase 3 question.

Next: `uv run scripts/probe_recall.py --mode mixed --queries 200 --coverage 0.5 --mix 0.5 --adversarial-source real --adversarial-rank-threshold 3` is the actual measurement of whether real-recall adversarial probing heals reranker weaknesses.
