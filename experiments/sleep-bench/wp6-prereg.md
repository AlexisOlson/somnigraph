# WP6 pre-registration — feature-structure study (random subsets)

**Date registered:** 2026-07-10, before any Stage-A sweep run. Protocol: Alexis's design
(plan-reranker-2x2.md § WP6) — random-subset sweep → importance-instability structure map
→ targeted hypothesis tests. **No stepwise selection at any stage.** All stages on
train-side CV (convs 1–9, c2 store); conv0 remains untouched by selection.

## Feature split (from SCHEMA_24)

- **Channel core, always-on (12):** fts_rank, vec_rank, theme_rank, fts_bm25, vec_dist,
  fts_bm25_norm, vec_dist_norm, query_idf_var, query_length, candidate_pool_size,
  query_coverage, proximity.
- **Toggleable (12):** priority, token_count, theme_count, member_count, is_summary,
  is_consolidation_node, consolidated_member, edge_count, betweenness, diversity_score,
  ppr_score, theme_overlap.

## Pre-stated hypotheses (from the H10 headline run's importances)

**H-A — the credit pool is substitutive.** {member_count, priority, token_count,
theme_count, is_summary} act as one credit pool for "is this a sleep-synthesis row and
how big": masking `member_count` (R_post gain 93.6) redistributes gain onto the others
with little CV-metric loss; masking the entire pool costs CI-clear NDCG. Decision rule:
pool-out vs all-on paired CV delta CI excludes 0 while single-member knockouts do not.

**H-B — theme ↔ graph substitution.** theme_rank's collapse (R_pre 472 → R_post 8.6)
reflects substitution by the graph/summary cluster, not theme-signal death: with graph
toggleables masked, theme_rank (always-on) gain recovers substantially; CV metric loss
from masking graph features is partially absorbed while theme signal is present.
Decision rule: theme_rank mean gain across masks with graph-off is ≥ 3× its gain with
graph-on (instability read), qualitative.

**H-C — the sleep-awareness dividend is graph-led.** Masking the 4 graph toggleables
{edge_count, betweenness, diversity_score, ppr_score} costs more CV NDCG than masking
the 5 credit-pool toggleables; a graph-only-toggleables model retains most of the
full-schema advantage over the all-toggleables-off (core-only) floor. Decision rule:
paired delta(graph-out) > delta(pool-out), bootstrap over shared folds; "most" = ≥ 60%
of (all-on − core-only) retained, stated before seeing numbers.

## Stage-A design commitments

- Store: conv0-c2 only (toggleables are constant/degenerate on frozen). Labels: the
  existing `gt_2x2.json`. Recipe, folds, seed: byte-identical to WP4's R_post run —
  identical GroupKFold(5) assignments across all masks (paired comparisons).
- Masks: the two anchors (all-on, core-only) + N≈128 random masks, each toggleable
  included independently at p=0.5, seed 1234, deduplicated against anchors and repeats.
- Recorded per mask: per-fold + mean CV metrics (NDCG@5k, R@10, MRR, gRecall@5k, RMSE)
  and per-feature gain importances. No final-model fits (CV only).
- Stage B runs only after Stage A completes; Stage C masks are chosen from Stage B's map
  with each test stated before it runs (appended to this file, dated).

## Outcome interpretation guardrails

- CV metrics here rank *feature sets*, not the headline claim — no Stage-A/B/C number
  competes with the conv0 cells.
- Gain importances within one model are unreliable under correlation (the reason this
  study exists); only cross-mask instability patterns are read as structure.
- A hypothesis "refuted" here means "not visible at ~1,757-query train-side CV power" —
  effect sizes travel to the findings doc either way.
