# 2026-07-10: H10 — sleep's value requires (and rewards) a sleep-aware reranker

Same-day continuation of the whole-corpus session: the §4.6 reranker 2×2, from plan v2
through built rig, labels, training, cells, and findings in one day. Orchestrator session;
the five work packages were built in parallel by opus agents against pinned interfaces,
then executed jointly (Alexis's terminal for the LLM judge + training; orchestrator for
bridge, cells, analysis). Full writeup:
[`findings-h10-reranker-2x2.md`](../../experiments/sleep-bench/findings-h10-reranker-2x2.md).

## What shipped

1. **Plan v2** ([`plan-reranker-2x2.md`](../../experiments/sleep-bench/plan-reranker-2x2.md)):
   whole-corpus revision — stores {conv0-frozen, conv0-c2} × rerankers {R_pre, R_post},
   query-level hold-out (train convs 1–9 / eval conv0) replacing v1's nine per-conversation
   fork pairs; 24-feature benchmark-valid schema; sonnet train labels gated by a
   cross-judge audit; judged credit primary on slept columns.
2. **The rig, five work packages** (opus build agents, all acceptance-checked):
   `build_2x2_labels.py` (WP1, GT→trainer bridge + `judge_summary_coverage.py --from-gt`
   sourcing mode), `bench_features.py` (WP2, 35-col extraction + SCHEMA_24),
   `score_model_arm.py` (WP3, Stage-A-format model scorer that never mutates
   recall_count), `train_2x2.py` (WP4, V6-parity deterministic recipe + `--feature-mask`
   WP6 hook), `analyze_2x2.py` (WP5, cells + five contrasts + interaction with paired
   bootstrap). Plus `cross_judge_audit.py` (D4 gate) and `retier_nodes.py` (D1 pre-req,
   applied to c2: 1 node).
3. **Cross-judge audit (D4 gate): sonnet vs opus on 50 shared pairs — binary 49/50 = 98%,
   exact 48/50 = 96%**; the one disagreement had sonnet stricter (safe direction). Sonnet
   ratified for train labels.
4. **Labels:** sonnet judged 2,245 (query, summary) pairs over convs 1–9 (Alexis's
   terminal, ~2h; 1,393 covering = 62%); bridge produced 1,757 queries / 3,878 labels
   (2,506 evidence + 1,372 coverage; 24 collision queries dropped, matching the dry-run
   exactly).
5. **Models:** R_pre (frozen) and R_post (c2), ~1M and ~639k training rows. Structural
   asymmetry confirmed at full scale: sleep features exactly zero gain on frozen;
   member_count 93.6 / priority 52.5 / graph features 18–330 gain on c2.
6. **Six cells + analyzer** → `results-full/analyze_2x2_final.json` and the findings doc.
7. **Review hardening the same day** (independent Fable-orchestrated review with opus
   subagents, before training): no blockers in the H10 rig; its one operational-risk
   finding (nothing verified a cell's `--label` matched its `--data-dir` — a mispairing
   would silently corrupt one cell and the headline interaction) was fixed as `867ed65`
   (analyzer asserts stem vs the data_dir stamped in the sibling `.summary.json`;
   verified positive/negative/missing/undormant cases). The same review cleared the
   parked `sleep/nrem-parallel` branch for merge (one FIX-LATER: env-var ValueError
   guard).

## Headline results (judged credit, opus eval table, 196 conv0 queries, ✱ = CI-clear)

- **Interaction (④−②)−(③−①) = +0.072✱ NDCG / +0.079✱ MRR** — sleep's value is
  conditional on a reranker that learned sleep's structure. Raw interaction +0.165✱
  everywhere (credit-inflated).
- **★ ④−③: the slept store BEATS frozen under R_post** (+0.091✱ NDCG / +0.077✱ R@10 /
  +0.096✱ MRR) — the first net-positive whole-corpus sleep configuration, reversing the
  morning's rrf verdict (R@10 −0.186✱) on the same stores and same credit table. The
  entire reversal is the ranking layer.
- **②−① (naive deploy) is judged-null** (+0.02 ns; raw −0.27✱ to −0.34✱) — the
  pre-registered "near-certain" regression did not survive judged credit. A stale
  reranker on a slept store treads water; it forfeits the upside, doesn't regress.
- **Undormant column refutes the recall-ceiling story:** undormant_Rpost − slept_Rpost
  ≈ −0.026 ns — with a sleep-aware reranker, reactivating the 5,737 dormant details buys
  nothing. undormant_Rpre ≈ slept_Rpost (0.594 vs 0.596): two equivalent routes; the
  choice between them is an efficiency question (541 vs 6,271 active rows).
- **③−① null** — R_post costs nothing on the unslept store. Training past sleep is free
  insurance.

## Design decisions (Alexis, in-session)

- **c2 as the slept store** (over c1) — the H12-consolidated end state is what production
  would run.
- **Drop `age_days`** from the schema: sleep rows are stamped `created_at=now` against
  2022–24 backdated details, a ~1,000-day cliff aliasing `is_summary`, and wall-clock
  nondeterministic. **Keep `priority`** — it is H12's only scoring channel.
- **Cross-judge audit before the label run** (gate, not retrospective check).
- **WP6 protocol replaces stepwise selection**: random feature subsets → importance-map →
  substitution/complement hypotheses → pre-stated tests; train-side CV only, conv0
  reserved. Motivated by ordering/pairing effects that make linear forward/backward
  selection misleading.
- Labeling target settled after a leakage discussion: labels are retrieval-relevance
  (evidence + judge-verified summary coverage), never reader answers; three separations
  (disjoint queries, disjoint pairs, different judge models for train vs eval) + raw
  always reported beside judged; residual shared-store-statistics named in the plan.

## Surprises

- **Two of five pre-registered predictions refuted** (naive-deploy regression; undormant
  ≥ slept). Both refutations are *good news for sleep* — the design let the result
  surprise us in the favorable direction.
- **The sleep-awareness dividend is graph-led, not summary-led**: diversity_score/
  betweenness/edge_count (NREM's edges) absorbed more gain than the summary/node
  features. The summary pool is carried by `member_count` alone — `is_summary` gain is
  exactly 0 (fully aliased), the plan's aliasing caveat landing in its strongest form.
- **`theme_rank` collapsed 472 → 8.6** between R_pre and R_post — theme signal is
  substitutable by the graph/summary cluster on this corpus. WP6 hypothesis material.
- **`priority` at 52.5 gain** — first direct evidence H12's downweight channel is visible
  to a learned scorer (invisible to rrf/formula by construction).
- The WP4 build agent's example bridge command pointed at the **opus eval table** instead
  of the sonnet train table — the exact silent failure the plan warned about (conv0 ids
  match nothing in convs 1–9 → evidence-only labels). Caught in review of the agent
  report; the gotcha is now written into the findings repro section.

## Caveats

- n=196 eval queries, one corpus, one training run per model; recall-metric CIs are wide
  (the interaction is CI-clear on ranking metrics only).
- Judged credit: one opus run (eval) + one sonnet run (train); 98% agreement on 50 pairs
  is reassuring, not proof. Residual circularity = shared store statistics.
- Undormant cells evaluate R_post out-of-distribution (trained on 541 active rows, scored
  on 6,271) — its mild negative there is transfer loss, not a reactivation-harm claim.
- `subsample=0.8` is inert in the recipe (no `bagging_freq`) — kept deliberately for
  V6 parity; fold CV metrics are mildly optimistic (early-stop on the reported fold) but
  the shipped artifact is the fixed 500-tree final fit.

## Files touched

- `experiments/sleep-bench/findings-h10-reranker-2x2.md` (new)
- `experiments/sleep-bench/analyze_2x2.py` (store-mismatch guard, `867ed65`)
- Committed earlier today (`a34013e`): the five WP scripts, `cross_judge_audit.py`,
  `retier_nodes.py`, `judge_summary_coverage.py --from-gt`, plan v2
- Local only (gitignored, `results-full/`): `gt_2x2.json`, `train_coverage_sonnet.json`,
  `R_{pre,post}.txt` + sidecars, 6 cell JSONLs, `analyze_2x2_final.json`
- `docs/sessions/2026-07-10-sleep-bench-whole-corpus.md` + this file + STEWARDSHIP entries

## What's next

1. **WP6** — random-subset feature-structure study, now with three concrete hypotheses
   (member_count/priority/is_summary credit pool; theme↔graph substitution; graph-only
   interaction).
2. **Dormancy decision update**: the downweight sweep should be scored under both
   rerankers, and "status-death + sleep-aware reranker" is now a legitimate cheap
   alternative (11.6× smaller active set at judged parity).
3. **Live 07-15 fork guidance**: plan a reranker retrain after the first real sleep;
   deploying stale is safe but forfeits the gain.
4. Pending Alexis: `sleep/nrem-parallel` merge (review-cleared), push decision, pagefile
   bump. Pre-fork adversarial review of the production sleep path (~07-14) still booked.
