# 2026-05-09 — Phase 2 trainer runs complete — held-out hard baseline established + May-8 Goodhart hypothesis tested

Re-emitted GT against production DB after a fresh 150-query probe run earlier in the day: 1,271 queries (543 live, 84 natural, 89 mild, 87 hard [HOLDOUT], 468 synthetic).

## Three trainer runs

**4d** — no-sidecar reference: RMSE=0.0260, aggregate reranker NDCG=0.8731 vs production RRF 0.7883, delta +0.0849.

**4e** — default schedule: RMSE=0.0265, +0.0005 vs baseline — well within the ±0.02 sanity bound, schedule steers without destabilizing. **Headline: held-out probe-hard NDCG=0.9146, R@10=0.9739** on 87 queries the model never saw during training.

Per-mode breakdown (4e): live 0.7334/0.8154, probe-natural 0.9497/0.9766, probe-mild 0.9485/0.9707, probe-hard 0.9146/0.9739, synthetic 0.9977/1.0000.

**4f** — synthetic-zero ablation via `--override-weight synthetic-self-anchor=0`: RMSE=0.0324, jump expected — synthetic rows held out from training but still scored.

## Evaluation notes

The ~0.18 NDCG gap between live and probe modes is mostly evaluation-distribution bias, not retrieval quality — probes have pinned single targets plus crafted queries with intent text; live has fuzzy multi-target relevance from short keyword bags (median 8 tokens) rated by the agent in-flight.

The Phase-1-miss caveat applies to held-out hard: rows where retrieval missed the target give 0 contribution regardless of model quality, so the metric measures "given the candidate pool retrieval surfaced for the hard query, does the reranker order it right" rather than end-to-end answerability.

## Goodhart hypothesis result (4f vs 4e): negative

Held-out probe-hard NDCG basically flat (-0.005), but R@10 *dropped* -2.9pp (0.9739 → 0.9448). Live marginal (NDCG +0.009, R@10 -0.001). Synthetic NDCG collapsed -0.086 as expected.

Synthetic GT is contributing real signal at volume (~37% of training rows, all with clean pinned targets — the model uses it as a reliable ranking lesson that generalizes); the structural May-8 mitigations (NaN sentinels for rank/score features + themes-query/summary-context shape match) appear to have already neutralized the original concern. Schedule stays calibrated; synthetic=0.5 is closer to right than synthetic=0.

## Feature importance

Feature importance from 4e: session_recency #1 (188), query_idf_var #2 (171), fts_bm25_norm #3 (166); category/priority/proximity all dropped further as the May-8 work predicted but session_recency continues to dominate even after the sentinel fixes — worth Phase 3 investigation.

## Path decisions

Path A (re-rate live GT with retrospective Opus) explored and rejected: Alexis confirmed production rater pool is nearly uniformly Opus with full session context, so a retrospective judge with bounded context cannot beat in-flight rating with richer context.

Path B (live-shaped replay-and-rate probe) demoted from "label improvement" to "optional eval-tooling" — its remaining value is a freshly-evaluable metric per model iteration, not better signal than what we have.

## Surprises

Confidently predicted synthetic=0 would *raise* held-out hard if Goodhart was real, and instead R@10 dropped 3pp. The May-8 work overestimated synthetic's risk; volume and label clarity matter more than shape concerns once the structural sentinels are in place.

## Caveats

Held-out hard NDCG=0.9146 is a specific kind of generalization (probe-shaped queries against the current candidate pool); it is *not* a proxy for live performance, and the live number (0.7334) is the actual production bottleneck.

## Phase 3 priorities (in EV order)

(a) per-(q,m) up-weighting of pinned target rows within probe queries — currently the highest-confidence label sits at row-weight identical to LLM-rated neighbors, smallest change with measurable impact; (b) candidate-pool widening so Phase-1-miss rows contribute training gradient rather than just appear in eval; (c) investigation of why `session_recency` continues to dominate feature importance after May-8 sentinel fixes — signal vs bias residue is now empirically testable via leave-one-feature-out per-mode evaluation.

P2 progress: held-out hard baseline established as the trustworthy generalization metric over time; aggregate NDCG vs production RRF +0.0845 (4e) is the apples-to-apples Phase 2 win; default schedule (live=natural=extra=1.0, mild=1.2, hard=0.0 holdout, synthetic=0.5) lands as production-ready.
