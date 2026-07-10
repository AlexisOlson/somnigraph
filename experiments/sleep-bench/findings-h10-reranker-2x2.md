# Findings — H10: the pre/post-sleep reranker 2×2 (whole-corpus)

**Date:** 2026-07-10. **Plan:** [`plan-reranker-2x2.md`](plan-reranker-2x2.md) (v2, executed as specified).
**Question:** does sleep's retrieval value require a reranker that learned sleep's structure —
and is a sleep-aware reranker safe to deploy before sleeping?
**Verdict: YES to both, and stronger than predicted.** The interaction is CI-clear positive
(judged **NDCG +0.072✱ / MRR +0.079✱**), and with a sleep-aware reranker the slept store
**beats** frozen (**+0.091✱ NDCG / +0.077✱ R@10 judged**) — the first net-positive
whole-corpus sleep configuration, reversing the rrf verdict (R@10 −0.186✱). Dormancy's
recall damage, the lone eroder in the whole-corpus findings, is **recoverable in the
ranking layer**: once R_post ranks covering summaries into the top-10, reactivating the
5,737 dormant details adds nothing (undormant − slept ≈ 0 judged). And the sleep-aware
reranker costs nothing on the unslept store ((3)−(1) null) — training on a slept store is
free insurance, not a tradeoff.

## Design (as executed)

- **Stores** (`~/.somnigraph/benchmark/sleepbench-full/`): conv0-frozen (5,882 active, 0
  edges) and conv0-c2 (541 active incl. 396 summaries + 8 consolidation nodes, 13,013
  edges, 5,737 dormant) — the whole-corpus arms. Bonus eval store: conv0-c1_rem_undormant
  (6,271 active — c1_rem with dormancy reverted). c2 nodes retiered to the `94aad00`
  priority band (floor 6) before feature capture.
- **Rerankers:** R_pre (trained on frozen) and R_post (trained on c2), identical 24-feature
  schema (35 extracted minus dead feedback/session, constants, and `age_days`; see plan
  § D2), identical V6-parity pointwise recipe (LightGBM regression, 500-tree final fit,
  deterministic), GroupKFold(5) grouped by query.
- **Labels (train, convs 1–9):** 1,757 queries (24 intra-conv text-collision queries
  dropped), 2,506 evidence labels + 1,372 sonnet-judged summary-coverage labels (61% of
  2,245 pairs; sonnet ratified by the 50-pair cross-judge audit vs opus: binary 49/50 =
  98%). Bridge: `build_2x2_labels.py`.
- **Eval (conv0, 196 GT queries):** query-level hold-out — no eval query or (query,
  summary) pair appears in training. Eval credit uses the **opus** judge table
  (`summary_coverage.json`, 125/229), a different judge run and model than the train
  labels. Paired bootstrap n=10,000, seed 1234; ✱ = 95% CI excludes 0.
- **Judged is the primary credit regime on slept columns** (dormancy makes raw
  evidence-id scoring measure the scorer, not the store — whole-corpus Result 1). Raw is
  reported beside it throughout. Frozen cells: judged ≡ raw.
- Training-side CV (context, not comparable across stores): R_pre NDCG@5k 0.4534,
  R_post 0.5550, both gRecall@5k = 1.0.

## Cells (judged / raw, NDCG@5k · R@10 · MRR)

| cell | store | reranker | judged | raw |
|---|---|---|---|---|
| ① frozen_Rpre | frozen | R_pre | 0.516 · 0.664 · 0.452 | same |
| ② slept_Rpre | c2 | R_pre | 0.535 · 0.687 · 0.470 | 0.244 · 0.383 · 0.172 |
| ③ frozen_Rpost | frozen | R_post | 0.504 · 0.666 · 0.440 | same |
| ④ slept_Rpost | c2 | R_post | **0.596 · 0.743 · 0.536** | 0.398 · 0.607 · 0.314 |
| undormant_Rpre | undormant | R_pre | 0.594 · 0.771 · 0.536 | 0.367 · 0.552 · 0.293 |
| undormant_Rpost | undormant | R_post | 0.570 · 0.729 · 0.513 | 0.407 · 0.559 · 0.337 |
| ref: frozen rrf | frozen | none | 0.437 · 0.627 · 0.364 | same |

Both learned rerankers clear the rrf baseline on frozen (+0.07–0.08 NDCG raw) — the rig
trains real models, not noise.

## Result 1 — the headline interaction is CI-clear

Interaction = ((④−②)) − ((③−①)), judged: **NDCG +0.0721✱ [+0.023, +0.122], MRR +0.0785✱
[+0.023, +0.136]**; R@10 +0.055 and recall_any +0.046 positive but CI-straddling;
recall_all_10 +0.0714 with CI lower bound exactly 0. Raw interaction is enormous and
CI-clear on all six metrics (NDCG +0.165✱, R@10 +0.223✱) but inherits the credit
artifact; judged is the honest number. Reading: **the value of sleeping the store is
conditional on the reranker having learned sleep's structure** — on ranking metrics
decisively, on recall suggestively at n=196.

## Result 2 — ★ sleep + sleep-aware reranker beats the unslept store

④−③ (slept vs frozen, both under R_post), judged: **NDCG +0.0914✱, R@10 +0.0774✱, MRR
+0.0962✱, recall_any/all_10 +0.0765✱** (recall_any@effk +0.020 ns). Under rrf the same
store pair was R@10 **−0.186✱** (whole-corpus headline). Same stores, same opus credit
table — the entire reversal is the ranking layer converting summary coverage into top-10
hits. This is the first configuration in the program where whole-corpus sleep-as-is
(97.5% dormancy included) is a net retrieval **win**.

## Result 3 — naive deploy is null, not the predicted regression

②−① (slept store, frozen-trained reranker), judged: NDCG +0.019, R@10 +0.022, all CIs
straddling zero (recall_any@effk −0.020 ns the only negative lean). Pre-registration
called ② ≤ ① "near-certain given dormancy" — **wrong under judged credit** (raw: −0.27 to
−0.34✱ across the board, as predicted). R_pre's content features (vec_rank, fts, idf)
transfer to summaries well enough to tread water where fixed-weight rrf lost −0.186✱.
Practical reading: deploying a stale reranker onto a freshly slept store doesn't hurt —
it just forfeits the ④ upside (−0.061✱ NDCG vs R_post on the same store).

## Result 4 — the undormant column refutes the recall-ceiling story

Pre-registration expected undormant ≥ slept cells (ceiling lifted). Judged says no:
undormant_Rpost − slept_Rpost ≈ **−0.026 ns** (nothing gained by reactivating 5,737
details under a sleep-aware reranker), and undormant_Rpost − undormant_Rpre = −0.024 ns
(recall_all_10 −0.056✱ — R_post is mildly *mis*calibrated for a store whose details are
all active; it learned c2's distribution). Meanwhile undormant_Rpre (0.594) ≈ slept_Rpost
(0.596): **two equivalent routes to the same judged endpoint** — keep details active with
any decent reranker, or dormant them and rank summaries with a sleep-aware one. The
choice is an efficiency question (541 vs 6,271 active rows = candidate-pool and index
size), not a retrieval-quality question — which materially softens the whole-corpus
"dormancy = the eroder" verdict: dormancy erodes *fixed-weight* retrieval; it is neutral
under a reranker trained past it.

## Result 5 — importances: the sleep-structure gain is real but graph-led

Side-by-side (gain), the features R_pre structurally could not learn (all-constant on
frozen) vs R_post: `diversity_score` 0 → 330, `betweenness` 0 → 186, `edge_count` 0 → 173,
`member_count` 0 → 94, `priority` 0 → 52, `ppr_score` 0 → 18, `is_consolidation_node`
0 → 1.4. So the sleep-awareness dividend is **graph-features-first** (NREM's edges),
summary/node features second. Two notes:

- `is_summary` gain is exactly 0 in R_post — fully aliased by `member_count` (nonzero ⟺
  summary/node, plus cluster-size signal), the pre-registered aliasing caveat landing in
  its strongest form. Read {member_count, priority, is_summary} as one credit pool.
- `theme_rank` collapsed 472 → 8.6 across the two models — theme signal on this corpus is
  substitutable by the graph/summary cluster. WP6's substitution mapping should confirm.
- `priority` at 52.5 gain is the **first evidence the H12 downweight channel is visible
  to a learned scorer** (invisible to rrf/formula by construction).

## Pre-registered prediction scorecard

1. ② ≤ ① judged recall — **refuted** (null; raw-only regression). The surprise of the run.
2. ④ > ② — **confirmed✱** (all judged metrics except recall_any_10 at exactly-0 lower bound).
3. Interaction > 0 — **confirmed✱** (NDCG, MRR; recall metrics positive, CI-straddling).
4. Summary/priority cluster high in R_post, impossible in R_pre — **confirmed in
   direction, wrong in emphasis** (graph features lead; member_count carries the summary
   pool; is_summary itself 0 via aliasing, as the caveat anticipated).
5. Undormant ≥ slept cells — **refuted** (≈0; the ceiling wasn't binding under R_post).

2/5 clean confirms, 1 directional, 2 refuted — the design did its job of letting the
result surprise us.

## Caveats

- n=196 eval queries, one corpus (LoCoMo), one training run per model (deterministic
  recipe, but no seed sweep). Recall-metric CIs are wide at this n.
- Judged credit rests on one opus judge run (eval) and one sonnet run (train labels);
  cross-judge agreement 98% binary on 50 pairs is reassuring, not proof.
- Residual circularity (named in plan § circularity): train and eval share the store's
  *statistics* (both judges see the same summary texts), though never queries or pairs.
- R@10/recall computed over `--recall-limit 20` ranked lists from ~5k-candidate pools —
  consistent across cells (the contrast is fair) but not comparable to full-pool recall.
- The undormant column evaluates R_post out-of-distribution (trained on c2, scored on a
  store with 11.6× more active rows) — its mild negative is expected transfer loss, not a
  clean "reactivation is harmful" claim.
- Feature importances are gain on one model each; substitution structure (theme_rank's
  collapse, the member_count pool) is hypothesis material for WP6, not conclusions.

## Files / reproduction

- Models + CV + importances: `results-full/R_{pre,post}.txt` + sidecars (gitignored).
- Labels: `build_2x2_labels.py --store <frozen> --convs 1-9 --coverage
  results-full/train_coverage_sonnet.json --out results-full/gt_2x2.json`
  (1,757 q / 3,878 labels). **Never swap the two coverage tables** (sonnet = train,
  convs 1–9; opus `summary_coverage.json` = eval, conv0).
- Train: `train_2x2.py --gt ... --store <frozen|c2> --name R_pre|R_post` (~15–40 min each).
- Cells: `score_model_arm.py --data-dir <arm> --model R_*.txt --features
  R_*__features.json --label <frozen|c2|c1_rem_undormant> --conv 0` (LLM-free, ~1 min
  each; `analyze_2x2.py` hard-fails on label/store mismatch since `867ed65`).
- Analysis: `analyze_2x2.py --cell frozen_Rpre=frozen__model-R_pre ... --judged-coverage
  results-full/summary_coverage.json --baseline-raw frozen__rrf --importances ...`
  → `results-full/analyze_2x2_final.json`.

## What this changes next

1. **The dormancy fix gets a cheaper option.** Whole-corpus said "downweight, don't
   status-kill"; H10 adds: *or keep status-death and deploy a sleep-aware reranker* —
   judged-equivalent, and dormancy's 11.6× active-set reduction is a real efficiency win.
   The downweight sweep should now be scored under both rerankers, not rrf.
2. **Retrain-on-sleep becomes deployment guidance:** sleeping a store without retraining
   the reranker forfeits the gain but doesn't regress (judged); retraining unlocks it and
   is free on unslept stores. For the live 07-15 fork: plan a reranker retrain after the
   first real sleep.
3. **WP6 (random-subset feature-structure study)** — now with concrete hypotheses: the
   member_count/priority/is_summary credit pool, theme_rank ↔ graph substitution, and
   whether graph features alone capture the interaction.
4. H12's `priority` channel is confirmed visible to a learned scorer — the
   coverage-proportional downweight (v2) has a measurement instrument.
