# 2026-05-08 — Goodhart confirmed via content-residual audit; deeper structural issue surfaced

Added `--query-from {summary,themes,content-residual}` to the audit script. content-residual = content tokens minus summary tokens (median 85 tokens per memory, all 468 pass the >=3 threshold), the strongest free OOD test we have because synthetic GT only ever saw summary text.

Result: **239/468 pathologies** (51%, with 36 in the 250-1000 gap bucket) — worse than the original broken baseline (89). Predicted: 5-15. Honest miss.

## SHAP analysis

SHAP analysis on three top offenders (`c149292b`, `a45315bb`, `8260792a`) — all three had vec rank 0/1 but reranker rank 380-433. Dominant features hurting target vs reranker's top-1:
- `age_days` (target 7-14d vs top1 71-72d, delta ≈ -0.10)
- `fb_count` (target 1-4 vs top1 59-107, delta ≈ -0.03 to -0.10)
- `theme_overlap` (identical values but different SHAP contribution — interaction effect: theme_overlap=3 only fires positively when paired with high age/count)

## Structural diagnosis

The model has two modes — "summary-shaped query → trust channels" (taught by synthetic GT) and "anything else → trust age + fb_count" (the legacy bias surface). NaN sentinel + synthetic GT closed one channel of self-reinforcement but the model routes around through fb_count and age_days.

Also discovered the GT pipeline only captures `query` (not `context`) from `recall_feedback` events: real recall accepts `(query, context)` where vec uses context if provided, but feedback events log only the FTS-style query string, so GT is biased toward short keyword bags (median 8 tokens, p25=6 p75=9). Current synthetic GT (full summaries, 30-100 tokens) is shape-mismatched against real GT distribution.

## Open path forward

(a) multi-shape synthetic GT (summary + themes-joined + first-sentence per memory) to defeat shape-memorization, (b) richer GT schema that preserves `(query, context, memory_id, label)` tuples to match production recall structure, (c) feature ablation on `age_days` and `fb_count` to test whether they encode bias more than signal, (d) cross-cycle probe miss-rate as the trustworthy generalization metric.

Work paused at this decision point to hand off to a fresh orchestrator session.
