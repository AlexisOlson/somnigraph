# 2026-03-14 — Utility calibration study + cliff detector replacement design

Ran Tier 1 experiment #4: compared 13,396 feedback events against 13,317 GT judgments. Per-query Spearman r=0.70 (feedback tracks relevance), per-memory r=0.14 (aggregation destroys context). Outlier inspection confirmed zero self-reinforcement — both "inflated" and "coverage gap" quadrants are per-memory averaging artifacts. Cutoff signal validated (2.2:1 GT ratio).

Score features cannot predict cutoff (R-squared < 0, all models worse than mean baseline). Designed replacement: agent-specified `limit` parameter with Fibonacci anchors {1,3,5,8,13}. Per-memory cutoff penalty deferred (signal improves with data, r=-0.31 at 20 obs, but too sparse now).

### Surprise

The score-feature failure was total — not "noisy" but anti-predictive. The cutoff is genuinely content-dependent. This changed the design from calibrating the cliff to replacing it.
