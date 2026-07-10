# WP6 — Feature-structure study of the sleep-aware reranker (R_post)

**Date:** 2026-07-10. **Protocol:** Alexis's design — random-subset sweep → importance-
instability structure map → iterative *registered* hypothesis batches until no sensible
tests remain. **No stepwise selection at any stage.** All numbers are train-side 5-fold
GroupKFold CV on the whole-corpus c2 store (convs 1–9, 1,757 queries, 638,570 rows,
SCHEMA_24) unless marked frozen; conv0 was never touched by any selection decision.
Canonical registration + verdict ledger: [`wp6-prereg.md`](wp6-prereg.md). Rig:
`sweep_2x2_features.py` (resume-safe JSONL, matrix disk cache, `--full-mask` +
seeded-noise columns), `analyze_wp6.py`.

**Why this study exists:** H10's R_post gain importances (graph-led: diversity 330,
betweenness 186 > member_count 94) are unreliable under feature correlation — gain
redistributes arbitrarily among correlated encodings. The only trustworthy structure
reads are *cross-mask* patterns: what happens to CV metrics and to other features' gains
when a feature set is removed.

## Headline conclusions

1. **The landscape is flat because the features are REDUNDANT, not irrelevant.** Removing
   the #1-gain feature (vec_rank, gain ~2,900) costs only −0.0106 — vec_dist/vec_dist_norm
   rebuild it. Nearly every signal in the schema has 2–3 alternate encodings. 166 masks
   produced exactly one drop larger than 0.02.

2. **The vec channel IS the model.** Removing all three vec columns costs **−0.2006**
   (0.5550 → 0.3544), twelve times the next-largest effect. Removing the FTS channel costs
   −0.0173. Single-channel floors: vec-alone 0.5224, fts-alone 0.3198, three-ranks-only
   0.5336. Score magnitudes beyond rank orderings are worth only ~0.0045 total, and the
   raw-vs-normalized split within that is below instrument power.

3. **The sleep-feature (toggleable) signal is real but masked by core redundancy — and it
   is doubly encoded.** On the rich 12-feature core, all 12 toggleables together add just
   +0.006; on a minimal 3-rank core they add **+0.0123** (2×). Decomposing that lift:
   token_count alone recovers 60%, the 5-feature credit pool 82%, the 4-feature graph
   cluster 78% — pool and graph are statistically tied, i.e. **two independent ~80%
   encodings of the same "synthesis row" signal**. Neither the pool-led read (Stage C1,
   rich core) nor the graph-led read (H10 importances) is the structure; both are
   projections of one redundant cluster.

4. **token_count is a generic content signal, not a summary detector.** Its knockout costs
   −0.0044 on c2 but **−0.0126 on the frozen store** (which has no summaries to detect) —
   3× larger where the summary story predicts ~0. It is the single most consequential
   toggleable on both stores.

5. **Orphaned flags mislead.** Removing the whole credit pool costs less (−0.0016) than
   removing token_count alone (−0.0044): binary flags (is_summary, member_count context)
   left behind without their continuous size signal actively hurt — a *partial* pool is
   worse than *no* pool. (This inverted pre-registered H-A.)

6. **No capacity artifact — and junk columns cost, generically.** core+noise12 (0.5462) <
   core_only (0.5490): extra columns per se don't lift. Four seeded noise columns cost
   −0.0047 on the full model and −0.0051 on core-only — the damage is generic split
   displacement (~0.0012/junk column), not weak-model vulnerability.

7. **theme_rank's R_pre→R_post gain collapse (472 → 8.6) is store composition, not
   substitution.** Its gain stays 109–141 under every frozen-store mask while c2 masks
   hold it at ~5–8; on c2 its *knockout* actually scores above all_on (0.5556). It and
   proximity are dead weight on the slept store.

8. **Per-query-constant gains are bookkeeping.** query_idf_var + query_length +
   candidate_pool_size carry ~1,400 combined gain but their joint removal costs −0.0016
   (sub-threshold). Gain importance on per-query constants measures pointwise score
   calibration, not ranking signal.

## Instrument caveats (read these before the numbers)

- **Column-width confound ≈ ±0.003:** masks with different column counts shift scores via
  colsample_bytree dilution alone (11 all-constant frozen columns moved the score 0.0031;
  4 noise columns ≈ −0.005). **Reading rule: trust only deltas > ~0.004, or same-width /
  large-gap ordering comparisons.** Most single-feature effects in this study live *inside*
  that zone and are reported as bounded-by-noise.
- CV here ranks *feature sets*, not the headline claim — no WP6 number competes with the
  conv0 2×2 cells (see `findings-h10-reranker-2x2.md`).
- "Refuted" means "not visible at 1,757-query train-side CV power."
- These labels reward evidence/coverage retrieval; they cannot price redundancy
  suppression (why h12_out ≈ all_on must not be read as "H12's channel is useless").

## Key numbers

Anchors: **all_on 0.5550**, core_only ≈ 0.5490 (NDCG@5k, mean over 5 shared folds).
Frozen store: all_on 0.4534 (= R_pre CV, reproduced per-fold to the digit).

| Mask (c2) | NDCG@5k | Δ | Read |
|---|---|---|---|
| fm_no_vec_channel | 0.3544 | −0.2006 | vec is the model |
| fm_fts_channel (fts alone) | 0.3198 | −0.2352 | fts floor |
| fm_vec_channel (vec alone) | 0.5224 | −0.0326 | vec floor |
| fm_ranks_only (3 ranks) | 0.5336 | −0.0214 | rank-only floor |
| fm_no_fts_channel | 0.5377 | −0.0173 | fts assist |
| fm_ranks_plus_token | 0.5410 | +0.0074 vs ranks_only | 60% of toggle lift |
| fm_ranks_plus_graph | 0.5432 | +0.0096 vs ranks_only | 78% — tied with pool |
| fm_ranks_plus_pool | 0.5437 | +0.0101 vs ranks_only | 82% |
| fm_ranks_plus_tog (all 12) | 0.5459 | +0.0123 vs ranks_only | full toggle lift, 2× rich-core |
| fm_no_vec_rank | 0.5444 | −0.0106 | #1 feature 1-point-replaceable |
| fm_core_noise4 / noise12 | 0.5439 / 0.5462 | < core_only | capacity story dead |
| knockout_token_count | ~0.5506 | −0.0044✱ | largest single toggleable |
| fm_no_scores (all 4 score cols) | 0.5505 | −0.0045 | ranks nearly suffice |
| fm_all_noise4 | 0.5503 | −0.0047 | noise cost is generic |
| fm_lean19 (drop 5 dead-core) | 0.5515 | −0.0035 | bounded-by-noise |
| fm_no_query_constants | 0.5534 | −0.0016 | gain = bookkeeping |
| pool_out (all 5 credit feats) | ~0.5534 | −0.0016 | less than token alone! |
| fm_schema22 (drop proximity, theme_rank) | 0.5560 | +0.0010 | validated prune |
| fm_no_theme_rank | 0.5556 | +0.0006 | dead on c2 |
| fm_no_proximity | 0.5551 | +0.0001 | dead on c2 |

Frozen-store contrast: knockout_token_count −0.0126 (vs −0.0044 on c2); token_only
recovers +0.0077 of the +0.0108 total frozen toggle lift; theme_rank gain 109–141
everywhere (vs 5–8 on c2).

## Hypothesis scorecard (all pre-registered; details in wp6-prereg.md)

| Hypothesis | Verdict |
|---|---|
| H-A credit pool substitutive | **Refuted, inverted** — orphaned flags mislead (partial pool < no pool) |
| H-B theme↔graph substitution | Refuted (gain ratio 1.63 < 3) |
| H-C dividend graph-led | Refuted on rich core (pool-led); batch 4 shows dual ~80% encoding — neither leads |
| N1 token = summary detector | **Dead** — frozen knockout −0.0126: generic content signal |
| N2 orphaned-flag split-stealing | Supported |
| N3 graph = redundant bookkeeping | Weak (1.1–1.9× gain shifts); batch 4 upgrade: graph is a real parallel encoding |
| N5 theme_rank collapse = composition | Supported |
| N6 consolidation-node label desert | Dead (2.2 labels/node) |
| N7 capacity artifact | Dead (noise < core_only) |
| N8 positive control (no_vec_rank) | Fired, but only −0.0106 → redundancy discovery |
| N10 raw/norm redundant | Wrong as stated; batch 3: group worth ~0.0045, internal split unresolvable |
| Batch-3 no_scores > 0.008 | Refuted (−0.0045) |
| Batch-4 graph lowest | **Refuted** — graph ties pool at ~80% (the informative surprise) |

## Deployment implications

- **Prune proximity and theme_rank** from the benchmark schema (schema22 ≥ all_on). The
  three per-query constants are droppable but not provably free (lean19 −0.0035).
- **Protect the vec channel above all else**; everything else degrades gracefully. A
  reranker missing any single other feature loses ≤ 0.011.
- **Don't ship binary sleep flags without their continuous companions** (orphaned-flags
  effect): is_summary/member_count travel with token_count or not at all.
- Gain-importance rankings on this family of models are **not** feature-value rankings —
  the study's raison d'être, now demonstrated three ways (vec_rank, theme_rank, query
  constants).

## N4 — the conv0 answer (ran 2026-07-10, after exhaustion)

**Confirmed at ~95%: the R_post dividend is core-channel recalibration, not sleep-feature
signal.** `R_core_post` (12 core features, c2-trained, CV 0.5512) scored on the conv0
cells: judged NDCG **0.5925** vs the full R_post's ④ 0.5957 — the 12 sleep features add
+0.0032 [−0.0247, +0.0295], a null; the core model is nominally *better* on judged R@10
(0.7598 vs 0.7432). Captured fraction of the ④−② gap ≈ 95% (pre-registered bar: ≥80%).

Deployment implication: **"retrain the reranker after sleep" (H10's guidance) does not
require sleep features in the schema** — retraining any reasonable core schema on the
slept distribution captures ~95% of the gain. What changed after sleep is the *core
feature distribution* (active pool 6,271 → 541, vec-distance mix, summary-heavy
candidates), and retraining recalibrates to it.

Post-hoc observation (not registered): on the *frozen* store the c2-trained full model
(0.5043) beats the c2-trained core model (0.4433) by 0.06 even though sleep features are
all-zero there — consistent with sleep features acting as train-time **routers** (their
splits separate summary rows from detail rows, keeping detail calibration clean) rather
than eval-time scorers. The registered sanity expectation ("core-on-frozen ≈ ①") was
mis-specified — ① is frozen-trained, so transfer effects apply. Data: `analyze_n4.json`
(cell slots reused; see prereg N4 verdict for the mapping).

## Open / out of scope

- Raw-vs-normalized score structure: below power here; would need a bigger label set.
- All numbers are one corpus (LoCoMo whole-store c2), one recipe (V6-parity pointwise
  LightGBM), coverage+evidence labels. Effect sizes travel, rankings may not.

## Files

- Ledger: `wp6-prereg.md` (every batch registered before running, dated).
- Data: `results-full/wp6_sweep.jsonl` (170 c2 masks), `results-full/wp6_sweep_frozen.jsonl`
  (frozen mini-sweep), matrix caches (gitignored).
- Rig: `sweep_2x2_features.py`, `analyze_wp6.py`.
