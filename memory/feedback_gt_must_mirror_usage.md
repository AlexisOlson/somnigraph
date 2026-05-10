---
name: GT must mirror actual usage distribution
description: Synthetic GT and training labels must match production query/context distribution; structural mismatch causes the model to learn shape-memorization, not retrieval semantics
type: feedback
---

GT should represent actual usage as much as possible — otherwise training is distorted.

**Why:** This was the lesson from the 2026-05-08 reranker bias diagnosis. Single-shape synthetic GT (summary-as-query, ~30-100 tokens) produced 0/468 audit pathologies on summary-shape but 239/468 on content-residual — worse than the broken baseline. Real recall feedback queries are short keyword bags (median 8 tokens, p25=6 p75=9). Real recall takes both `query` AND `context` (vec channel uses context if provided), but `memory_events.feedback` rows log only `query`, so GT loses that asymmetry. The model learned "summary-shape → trust channels, anything else → fall back to legacy bias features (age_days, fb_count)."

**How to apply:** When designing GT — synthetic anchors, judge prompts, label aggregation — first characterize the production query distribution (length, shape, query/context split, frequency of repeats). If synthetic data structurally diverges from that distribution, expect the model to overfit the synthetic shape rather than learn invariants. Prefer richer GT schemas (e.g. `(query, context, memory_id, label)` tuples that mirror live recall signature) over patching distributional mismatch with multi-shape variants.
