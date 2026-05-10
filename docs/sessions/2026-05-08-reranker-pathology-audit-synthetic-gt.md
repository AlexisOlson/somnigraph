# 2026-05-08 — Reranker pathology audit + two bug fixes + synthetic GT

Diagnosed why probe Phase-1 misses cluster on under-retrieved memories: live FTS+vec+theme each rank target at #1, but the LightGBM reranker buries the result past rank 200.

## Two latent bugs found

**(1) Missing-feedback sentinel:** `fb_last`/`fb_mean`/`fb_time_weighted` set to `-1.0` for memories without feedback. Real feedback values live in [0,1], so the sentinel reads as "worse than any real value" — the model learned aggressive demotion of cold-start memories. Fix: use `float("nan")` so LightGBM learns a separate missing-branch split.

**(2) `fts_bm25_norm` normalization broken:** SQLite FTS5 BM25 is negative (more-negative = better match), but the code guarded `if max_fts > 0 else 0.0`, so every value was zero. Fix: normalize by `abs(min(...))`.

## Audit infrastructure

Built `scripts/audit_reranker_pathology.py` — for each memory, query with its summary and check whether `rerank_rank - min(channel_ranks) > 10`. Baseline: 89/468 pathologies, 20 in the 250-1000 gap bucket.

NaN-fix-only: 61/468 — *but the catastrophic bucket grew to 37*, because removing the explicit penalty let the model use other features (session_recency, age_days, token_count) to discriminate against cold-start memories more aggressively. Root cause was deeper than the sentinel: the GT lacks generalization anchors.

## Synthetic GT

Memory M only has labels from queries that already retrieved M; if those queries weren't really about M, the model learns "M = low utility everywhere." Added synthetic self-summary GT: `(memory.summary, memory.id, 1.0)` for every active memory, +466 anchors → 1117 total queries. Combined with the bug fixes: 0/468 pathologies on summary-as-query (Goodhart-correlated by construction) and 1/468 on themes-as-query (OOD-flavored). Mean NDCG 0.7958→0.8681 with delta-vs-formula stable at ~+0.10. `fts_bm25_norm` jumped from broken (importance 0.0) to #2 (352.0); `query_coverage` rose into the top 6.

## Surprises

The NaN sentinel was the textbook example of feature-encoding leakage, but the secondary surprise was that fixing it alone made things *worse* in the catastrophic regime — the model had been using the sentinel as a shock-absorber that masked a deeper GT distributional gap. The synthetic-GT pattern doubles as a generalization anchor and as a cheap regression test (audit reuses the same shape).

**Goodhart caveat acknowledged:** audit and synthetic GT now share structure, so audit alone can't certify generalization — probe Phase-1 miss rate over time is the trustworthy signal.

P2 progress: two bugs closed, 31-feature retrain done. FSI stability audit and feedback-self-reinforcement experiments still pending.
