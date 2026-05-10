# 2026-03-19 — Bidirectional sync + LambdaRank revisited + full GT retrain

Production/repo sync: contradiction edge filtering backported to production scoring.py, theme channel unified (query tokens, not just boost_themes), last_accessed fallback forward-ported.

LambdaRank experiment re-run with two fixes: full candidate pool (--neg-ratio 0) and 100-level direct scaling (--lr-levels 100). Result: parity with pointwise (0.8099 vs 0.8127 on 500q), not improvement. Two-stage also ties at 0.8119. Objective doesn't differentiate at this scale.

GT v3 judging completed (489 queries). Full v2+v3 merge (969 queries, zero overlap) and retrained pointwise: NDCG=0.7971 (+6.18pp over formula, models converge at 895-1369 with n_estimators=1500). Model deployed to production.

Fixed reranker.py to use booster directly (avoids sklearn feature-name warnings), suppressed remaining warnings in training script.

### Surprise

Alexis caught the cargo-culted GT-only training pattern ("why can't we assume all unretrieved are 0?") — the infrastructure for full-pool LambdaRank already existed but was bypassed by the default --neg-ratio 2.0.

No priority reorder — P2 work continues with query features and raw-score features as next experiments.
