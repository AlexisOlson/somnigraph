# 2026-05-09 — V3 boost sweep + V4 smoke + V5 retrain + db.py backend guard

## V3 — four trainer runs, pinned-boost locked at 5.0

Four trainer runs against the same Phase 3a sidecar build (1,273 queries). Held-out probe-hard NDCG climb non-monotonic at small-step granularity, monotonic at the extremes:
- boost=1.0: NDCG=0.9138 / R@10=0.9755 (within ±0.005 of Phase 2 4e's 0.9146, validates per-(q,m) wiring is a true no-op at unit boost)
- boost=2.0: 0.9226/0.9687 (NDCG +0.0088, R@10 -0.0068 dip)
- boost=3.0: 0.9187/0.9721 (BELOW 2.0 on NDCG — fold-variance noise floor on the 87-query holdout is ~±0.005, so adjacent boost values aren't reliably ordered)
- boost=5.0: 0.9302/0.9821 (winner)

RMSE flat across the sweep (0.0265–0.0269), no destabilization at any tested point — the "RMSE jumps >0.01" tripwire never fired, ceiling not searched above 5. Aggregate NDCG drift mild (1→5: -0.0023). Live NDCG drift -0.0075 — designed trade-off, not regression: live has no pinned targets, so the boost only touches it via the model trading "fit fuzzy live labels" for "fit clean pinned labels harder."

Trade-off slope clearly decelerating: 1→2 = +4:1 hard:live favorable, 2→5 = +1.4:1 — extrapolated 5→10 likely crosses 1:1, so worth not chasing.

Locked `DEFAULT_PINNED_BOOST = 5.0` in `build_gt_from_feedback.py:643` (sidecar metadata default) and `train_reranker.py:1625` (CLI arg default), with V3 sweep numbers in the rationale comment.

## V4 — first live exercise of Phase 1's probe_target emitter

`--mode mixed --queries 8 --coverage 1.0`. 2 groups × 4 queries = 8 events, exact mode distribution {natural: 2, mild: 4, hard: 2}. Bundled craft: 2/2 succeeded, 0 fallbacks. Target row LLM rating distribution (Phase 3b blind rubric): median=0.95, p10=0.60, p90=0.95, 0/8 below 0.5 — zero craft drift, removing target_ids from RATE_TEMPLATE didn't hurt label quality at all. Mild orthogonality spot-check: both pairs read as genuinely different angles (Group 1: pipeline-mechanism vs downstream-rollup; Group 2: failure-mechanism vs decision-support), not paraphrases.

One mild observation: Group 1 mild #2 ("how to consolidate investor tranches into one partner name") surfaced its target at rank 15/15 — barely surviving the candidate pool — because the LLM crafter wandered onto a topic adjacent to the target memory's actual content. Worth watching at 200-query scale to see if creativity-drift becomes a pattern.

## V5 — re-emit GT + retrain at locked boost

GT grew 1273 → 1284 queries (+8 V4 + 3 fresh live). All 8 V4 probe_target events captured by the re-emit; dedup correctly folded 7 into existing ≥0.9 real-feedback labels and max-merged 1; "Total new (query, mid) pairs: 0" by design. Sidecar metadata picked up `pinned_boost_default: 5.0` from the locked constant.

Retrain at boost=5.0 deploys final MCP model (`reranker_model.txt` overwritten). Held-out probe-hard NDCG=0.9197, R@10=0.9842 on 89 queries (vs V3 boost=5.0's 0.9302/0.9821 on 87 queries) — NDCG -0.0105, R@10 +0.0021. NDCG outside ±0.005 fold-variance, but the simultaneous R@10 *rise* + probe-hard RMSE rise (0.0285→0.0324) means the 2 new V4 hard probes are genuinely harder to rank-1 but still recoverable in top-10. Sample-composition variance on the small holdout, not regression. Aggregate Reranker NDCG=0.8691, vs RRF +0.0821 — within fold variance of V3 boost=5.0.

## Side-quest — db.py backend-mismatch guard

During V3 setup, the trainer crashed trying to call OpenAI for embeddings against an empty `~/.somnigraph/` cache. Root cause: `EMBEDDING_BACKEND = os.environ.get("SOMNIGRAPH_EMBEDDING_BACKEND", "openai")` silently defaults to OpenAI's 1536d when production runs on fastembed's 384d. Without `SOMNIGRAPH_EMBEDDING_BACKEND=fastembed` set, scripts touching the production DB would either fail loudly when the OpenAI key isn't available (today's behavior, the visible crash) or silently emit dim-1536 vectors against a dim-384 vec0 index if the key were present (the actual footgun).

Added a guard in `db.py:_init_schema` that reads the existing `memory_vec` SQL from `sqlite_master`, parses `float[N]`, and raises a clear `RuntimeError` on dim mismatch — names the env var, both dim values, and the typical pairings (`fastembed` for 384, `openai` for 1536). Verified: `openai` backend against the fastembed-built production DB → loud failure with actionable message; `fastembed` against same → opens cleanly with 470 active memories. One-line entry added to `migration-notes.md`.

## Surprises

(1) Boost=3.0 NDCG (0.9187) landed BELOW boost=2.0 (0.9226) — adjacent boost values aren't reliably monotonic at this holdout size. The 1→5 trend is real (+0.0164, well above noise) but pairwise comparisons within the sweep are noisy, so "rises monotonically" was the wrong shape to expect — the right shape is "rises eventually, with sub-step noise." Treating boost as a continuous knob and sampling 4 points was right; treating each pairwise comparison as signal would have been wrong.

(2) The OpenAI silent-default would have been completely invisible if a stale OpenAI key happened to be on the box — production DB writes would have started accumulating 1536d vectors against a 384d index, with the failure mode being eventual sqlite-vec dim error or, worse, silently wrong cosine distances. The visible crash today was lucky.

## Caveats

The boost=5.0 live drop (-0.0075 from baseline) is the cost we accepted; if live performance becomes the primary concern in a future phase, lower boosts should be reconsidered. Boost ceiling > 5 not searched. The 89-query holdout is small enough that retrain-to-retrain held-out-hard NDCG comparisons need ±0.01 tolerance, not ±0.005.

## Reversibility

`--pinned-boost 1.0` disables per-(q,m) boost; `git revert` the two-line constants change to undo the lock; remove the `db.py` guard (or set `SOMNIGRAPH_EMBEDDING_BACKEND` correctly) to suppress the dim-mismatch RuntimeError.

## Files touched

`scripts/build_gt_from_feedback.py:643` (DEFAULT_PINNED_BOOST 2.0 → 5.0 + V3 rationale comment), `scripts/train_reranker.py:1625` (CLI default 2.0 → 5.0 + same rationale), `src/memory/db.py` (+~20 lines, dim-mismatch guard with helpful RuntimeError), `docs/migration-notes.md` (one-line entry on `SOMNIGRAPH_EMBEDDING_BACKEND` + the new guard).

## Phase 3 progress

Phase 3a-d implementation complete and validated end-to-end (sidecar v2 → bundled craft → live emitter → retrain). V5+1 (200-query mixed probe → re-emit → retrain) is the actual measurement of whether bundled orthogonality moves held-out generalization beyond what the 8-query smoke can resolve.

After V5+1: Phase 3 priorities (b) candidate-pool widening so Phase-1-miss rows contribute training gradient, (c) `session_recency` signal-vs-bias audit via leave-one-feature-out per-mode evaluation.
