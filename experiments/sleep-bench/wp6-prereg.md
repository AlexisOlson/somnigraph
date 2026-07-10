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

## Stage C — registered 2026-07-10, after Stage B, before running

**C1 (rule-implementation masks, emitted by the analyzer blind to Stage-B values):**
pool_out, graph_out, graph_only, and the 5 single pool knockouts. RAN 2026-07-10.
Verdicts: **H-A refuted, inverted** (pool_out null at −0.0016 while token_count −0.0044✱ /
theme_count −0.0017✱ / priority −0.0014✱ knockouts are CI-clear — a partial pool is worse
than no pool; the binary flags mislead without the continuous size signal). **H-B refuted**
(theme_rank graph-off/graph-on gain ratio 1.63 < 3). **H-C refuted on clause 1**
(graph_out −0.0005 < pool_out −0.0016: the CV-metric dividend is pool-led, not graph-led;
the 85% graph_only retention clause passed but on a ~0.002 denominator).

**C2 (data-driven combos, chosen from Stage-B/C1 structure; predictions stated before
running):**

1. `token_only` = {token_count} — predict ≈ all_on (the toggleable story reduces to
   token_count).
2. `edge_only` = {edge_count} — the other clean single; predict small positive over
   core_only, < token_only.
3. `best_two` = {token_count, edge_count} — tests the strongest regression interaction
   (token×edge coef −0.0015): predict sub-additive, best_two ≈ token_only.
4. `pair_out_token_edge` = all toggleables except {token_count, edge_count} — same
   interaction from the removal side; predict ≈ pool_out (i.e. the other 10 toggleables
   contribute ~nothing without the two workhorses).
5. `knockout_is_consolidation_node` = all except {is_consolidation_node} — the one
   CI-clear negative marginal; predict ≥ all_on (dropping helps; 8 nodes, 2 of them
   pass-2 duplicates — noise-fitting surface).
6. `h12_out` = all except {priority, member_count, consolidated_member,
   is_consolidation_node} — the full H12/consolidation channel as one number; predict
   ≈ all_on. Interpretation guard: these labels reward evidence/coverage retrieval, not
   redundancy suppression — h12_out ≈ all_on must NOT be read as "H12's channel is
   useless," only as "this instrument cannot price it."
7. `pool_only` = the 5 pool toggleables — retention symmetry with graph_only.

Decision rules: paired per-fold deltas vs all_on (and vs core_only for the *_only masks),
same shared folds; CI-clear at 5-fold paired bootstrap = a real effect at this power,
otherwise reported as bounded-by-noise. No further Stage-C rounds without a new dated
registration.

## Iterative batch protocol + narrative hypotheses N1–N6 — registered 2026-07-10

**Protocol change (Alexis):** continue in ~10-mask registered batches until no sensible
tests remain; goal = explore the mask space enough to narrow down or rule out hidden
combos. Each batch: hypotheses + predictions appended here BEFORE running; free
(no-new-training) checks may run immediately after registration.

**N1 — token_count is a summary detector, not a length prior.** Predict: (a) its knockout
on the FROZEN store costs ~0 (no summaries to detect); (b) `is_summary_only` recovers
clearly less than `token_only` on c2 (continuous beats binary).
**N2 — the H-A inversion is orphaned-flag split-stealing.** Predict: `no_size_no_flags`
(drop token_count + is_summary + member_count together) recovers toward pool_out (−0.0016),
i.e. clearly better than knockout_token_count (−0.0044).
**N3 — graph gain is redundant node-type bookkeeping.** Predict (free check on existing
JSONL): graph-feature gains rise materially (~2×) in masks where token_count (and the
pool) are off; they are alternate encodings of "synthesis row," which is why graph_out
costs ~0.
**N4 — the R_post dividend is core-channel recalibration.** Predict: a `core_only_post`
model (12 core features, trained on c2) captures ≥80% of the ④−② judged NDCG gap on the
conv0 cells. THE ONLY TEST THAT TOUCHES CONV0 — runs only on explicit go, as a registered
additional cell.
**N5 — theme_rank's R_pre→R_post collapse is store composition, not substitution.**
Predict: theme_rank gain stays high (>100) under every frozen-store mask.
**N6 — is_consolidation_node's negative marginal is a label desert.** Predict (free
check): the 8 consolidation-node ids carry ≈0 coverage labels in gt_2x2.json; the flag
partitions a tiny label-poor region (overfit surface).

**Batch 1 masks (c2, adds to wp6_sweep.jsonl):** the 7 Stage-C2 masks above +
`no_size_no_flags` (N2) + `is_summary_only` (N1b).
**Batch 1 frozen mini-sweep (SEPARATE out-file wp6_sweep_frozen.jsonl, --store frozen,
--ignore-anchor-mismatch because the in-sweep gate compares against R_post values):**
anchors + `knockout_token_count` + `token_only` (N1a, N5). Manual anchor: frozen all_on
should reproduce R_pre CV (mean NDCG 0.4534).

## Batch 2 — core opened up (Alexis) + capacity control — registered 2026-07-10

**Scope change (Alexis): any feature may be masked, core included.** Driver gains
`--full-mask` (exact feature set) + seeded `noiseK` synthetic columns; full masks are
excluded from the analyzer's toggleable statistics (core-always-on assumption) and read
directly.

**Batch-1 verdict updates (C2 masks + free checks, before this registration):**
N6 DEAD (nodes carry 2.2 labels/node vs summaries 3.5 — no label desert). N3 weak
(graph gains shift only 1.1–1.9× with pool off; is_summary 3.1× is the one clean
substitution read). token_only (0.5519) does NOT recover all_on (0.5550) — N1's "token
is the detector" needs the flags/context story; best_two (0.5535) is SUPER-additive vs
singles, killing the regression's negative token×edge interaction (extrapolation artifact
of p=0.5 masks). knockout_is_consolidation_node (0.5528) < all_on — its negative marginal
was noise, dropping it does not help. h12_out 0.5524, pool_only 0.5528, pair_out 0.5537:
everything lives within ~0.004 of all_on. Emerging suspicion: NDCG creeps with the COUNT
of toggleables (capacity artifact of colsample_bytree) — batch 2's noise control tests
exactly this.

**N7 — capacity artifact.** Predict: if the all_on − core_only lift is tree-diversity
capacity, core+noise12 ≈ all_on; if content, core+noiseK ≤ core_only (noise splits
waste trees). Masks: `fm_core_noise4`, `fm_core_noise12`. Smoke hint: noise hurts.
**N8 — positive control.** `fm_no_vec_rank` (24 minus the top feature): predict the
largest CI-clear drop in the study, calibrating what load-bearing looks like against the
~0.004 toggleable effects.
**N9 — dead-core candidates.** `fm_no_proximity`, `fm_no_theme_rank`,
`fm_no_candidate_pool_size` (per-query constant: gain 472 but can't rank within a query
— if ≈ all_on, its gain is pointwise score-calibration bookkeeping),
`fm_no_query_coverage`. Predict all ≈ all_on except a small query_coverage drop;
any that hold ≈ all_on become schema-drop candidates.
**N10 — channel structure.** `fm_ranks_only` (3 rank cols), `fm_vec_channel`,
`fm_fts_channel` (single-channel floors), `fm_no_norms`, `fm_no_raw` (raw vs normalized
score redundancy — predict both ≈ all_on, one representation suffices).
**Carried from batch 1:** `no_size_no_flags` (N2), `is_summary_only` (N1b), and the
frozen mini-sweep (N1a/N5; separate out-file `wp6_sweep_frozen.jsonl`,
`--ignore-anchor-mismatch` because the gate compares to R_post; manual anchor = R_pre
mean NDCG 0.4534).

## Batch 3 — channel structure + schema pruning — registered 2026-07-10

**Batch-2 verdicts (before this registration):** N7 capacity DEAD (core+noise4 0.5439 /
core+noise12 0.5462, both < core_only ≈ 0.5490 — junk columns cost, don't lift). N8:
no_vec_rank −0.0106 only — the #1-gain feature is 1-point-replaceable (vec_dist/norm
rebuild it); redundancy, not irrelevance, explains the flat landscape. N9: proximity
(0.5551) and theme_rank (0.5556 > all_on!) DEAD on c2 — drop candidates;
candidate_pool_size −0.0012 bookkeeping-confirmed; query_coverage −0.0033 earns its keep.
N10: ranks_only 0.5336 (high floor); vec-alone 0.5224 vs fts-alone 0.3198 (vec is THE
channel); no_norms/no_raw each −0.004 — raw and normalized scores NOT redundant
(prediction wrong). N2 SUPPORTED (no_size_no_flags −0.0018 ≈ pool_out vs token-knockout
−0.0044: orphaned flags mislead). N1b SUPPORTED (is_summary_only 0.5498 < token_only
0.5519). Frozen mini-sweep (N1a/N5) still outstanding.

**Frozen mini-sweep verdicts (2026-07-10):** manual anchor EXACT (all_on frozen = R_pre
CV per-fold to the digit; gate FAIL vs R_post reference is the documented artifact).
**N1a DEAD** — knockout_token_count on frozen costs −0.0126 (predicted ~0): token_count
is a genuine detail-content signal (3× its c2 effect), not a summary detector. token_only
recovers +0.0077 of the +0.0108 frozen toggle lift. **N5 SUPPORTED** — theme_rank gain
109–141 across all frozen masks (vs ~5–8 on c2): the R_pre→R_post collapse is store
composition. **NEW INSTRUMENT CAVEAT:** all_on vs token_only on frozen differ only by 11
all-constant columns yet score 0.0031 apart — constant/junk columns dilute
colsample_bytree draws, so cross-mask deltas at different column counts carry a ~±0.003
composition confound. Reading rule: trust only deltas > ~0.004 or same-width comparisons;
fm_all_noise4 quantifies.

**Batch-3 masks + predictions:**
1. `fm_no_vec_channel` (drop vec_rank+vec_dist+vec_dist_norm): predict the largest drop
   in the study (> 0.011); `fm_no_fts_channel` (drop fts_rank+fts_bm25+fts_bm25_norm):
   predict modest (vec carries the corpus).
2. `fm_ranks_plus_tog` (fts_rank,vec_rank,theme_rank + all 12 toggleables): if
   toggleables lift the minimal core by ≫ +0.006 (their lift on the rich core), their
   signal is real-but-masked by core redundancy. Predict visible extra lift over
   ranks_only (0.5336).
3. `fm_schema22` (24 minus proximity, theme_rank): deployment prune. Predict ≥ all_on.
4. `fm_no_query_constants` (24 minus query_idf_var, query_length, candidate_pool_size —
   per-query constants, combined gain ~1,400): predict ≈ all_on ("gain is bookkeeping"
   generalizes).
5. `fm_no_scores` (24 minus fts_bm25, vec_dist, fts_bm25_norm, vec_dist_norm): predict
   drop > 0.008 (score magnitudes matter beyond rank orderings).
6. `fm_all_noise4` (24 + noise1–4): if noise hurts the rich model ≈ as much as it hurt
   core-only (−0.005), the damage is generic split displacement; if less, weak-signal
   models are the vulnerable ones.

## Outcome interpretation guardrails

- CV metrics here rank *feature sets*, not the headline claim — no Stage-A/B/C number
  competes with the conv0 cells.
- Gain importances within one model are unreliable under correlation (the reason this
  study exists); only cross-mask instability patterns are read as structure.
- A hypothesis "refuted" here means "not visible at ~1,757-query train-side CV power" —
  effect sizes travel to the findings doc either way.
