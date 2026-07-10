# 2026-07-10: Whole-corpus sleep — summaries replicate, dormancy is the eroder

Covers the whole-corpus experiment set in motion at the end of the 07-09 session: all 10
LoCoMo conversations ingested into ONE store (5,882 memories), one full iterated sleep run
(overnight, Alexis's terminal, `main@2c08f21`), scoring + credit bracket + a fresh opus
coverage judge, and the counterfactual arm that decomposed the net effect exactly.
Full writeup: [`findings-whole-corpus.md`](../../experiments/sleep-bench/findings-whole-corpus.md).

## What shipped

1. **The whole-corpus run itself** — `ingest_baseline.py --convs all` (first real-data
   exercise of the `--convs` path: clean, merged conv-scoped dia_map resolved exactly) +
   `run_iterated.py --max-passes 2 --nrem-mode deep`. Arms in
   `~/.somnigraph/benchmark/sleepbench-full/`: frozen, c1_nrem, c1_rem, c1, c1_consol, c2.
2. **First real-LLM H12 consolidate run** (prior validation was 62/62 mocked checks):
   8 clusters → 8 core-facts nodes, 18 members downweighted (×0.8, once-ever flag held),
   **0 archived, 0 schema rejects** across both passes.
3. **`gen_undormant_arm.py`** (new) — forks c1_rem and reactivates all 5,737 dormant rows
   (pure status flip; verified vec/FTS were never touched: 6,271/6,271 rows in both
   indexes), giving the counterfactual that isolates dormancy's price.
4. **`impute_iterated.py` judged-guard fix** — the guard demanded judge coverage for ALL
   summaries (an H11-ism where all 28 were eligible); corpus-scale only ~229/396 are
   credit-eligible (generated_from ∩ evidence). Now checks the eligible set.
5. **Fresh opus coverage judge run** (Alexis's terminal): 194 queries / 229 pairs /
   20 batches → `results-full/summary_coverage.json` (gitignored, local). This table is
   also H10's **eval** credit basis.
6. **`.gitignore`**: `experiments/sleep-bench/results-full/` — separate results dir because
   the default `results/` holds H11's outputs under identical labels and would have been
   silently overwritten.

## Headline results

- **★ The undormant decomposition (the finding).** Judged, vs frozen: summaries added
  with details kept active = **+0.078✱ NDCG / +0.071✱ R@10 / +0.088✱ MRR** — **H7
  replicates at 15× scale** (conv0 was +0.093✱; n=2 now, fresh REM generation, fresh
  judge, 14× more distractors). Price of dormancy alone = **−0.136 NDCG / −0.256 R@10**
  (point est.). Rows sum exactly to the net (−0.058 / −0.186✱). The additive-summaries
  claim is now the program's most robust positive result; **dormancy alone flips the
  sign of sleep**.
- **Sleep as-configured LOSES on whole-corpus LoCoMo** (judged c2−frozen, rrf): R@10
  −0.186✱, recall_any −0.240✱, NDCG −0.054 null-leaning, MRR dead null. Ranking survives
  11:1 compression; the entire loss is the recall tail — queries whose fact exists only in
  a dormant detail whose summary dropped it. 55% summary fidelity (125/229 judged pairs,
  statistically = H7's 52%) is enough to ADD value, not enough to REPLACE details.
- **Dormancy fired at 97.5%** (5,737/5,882 details) — every covered member trivially met
  the age (>60d, LoCoMo dates 2022–24) and access (<3, fresh store) criteria. Coverage ⇒
  status-death on cold corpora.
- **Cross-conversation blending essentially did not occur** — the whole-corpus design's
  headline worry: 2.9% cross-conv detail edges, 8/388 summaries mixed (all exactly 2
  convs, topically-generic clusters, speakers correctly attributed, parallel-account
  structure), **0/7 consolidation nodes mixed**. Topical coherence isolates conversations
  on its own; synthesis (where blending harm lives) stayed conv-pure.
- **H12 exonerated consolidate**: retrieval-inert under both scorers (c1 ≈ c1_rem
  everywhere), zero archival damage — the erosion H11 attributed to consolidate is gone,
  leaving dormancy as the lone eroder.

## Surprises

- **H11's "31 dormanted" was instrument contamination.** Every H3/H7/H11 scoring pass
  calls `impl_recall`, which writes `recall_count` into the arm store — covered-active
  details in the H11 arm carried counts up to 187, shielding 380/411 of them from
  dormancy's access criterion. Same code, different history: scoring mutates stores, and
  a store scored-then-slept behaves differently from fresh. Queued rig fix: `fork.py
  --scrub-access-counts`.
- **The raw metric collapse (NDCG 0.437 → 0.012) is ~all credit artifact** — impute−raw
  gap +0.60 NDCG✱; for 78–83% of queries a summary generated from the exact evidence sits
  in the top ranks. The raw number measures the scorer, not the store.
- **Provenance's +0.173✱ upper bound was compression mirage** — the judge confirmed only
  55% of provenance-claimed pairs. The bracket did its job: both bounds were wrong in
  opposite directions and the judge collapsed them.
- **Pass-2 re-consolidation gap (new H12 defect):** pass 2 re-synthesized clusters pass 1
  had already consolidated — two identical-member-set node pairs, plus a third
  near-duplicate caught only by write-path 0.9 dedup (the run's single deletion).
  `gather_clusters` never checks the `consolidated_member` flag. Fix candidate: skip
  majority-flagged clusters.
- **NREM at scale is exactly zero-eviction**: c1_nrem rrf results byte-identical to
  frozen with 5,900 edges added — the H3 link-default property holds at 15× scale.

## Caveats

- Single corpus, single run; edge sets nondeterministic across runs (H11 lesson).
- rrf/formula only — H12's member downweight is invisible to both scorers by construction
  (nothing in rank fusion reads `base_priority`); its payoff is H10's question.
- The judge is the credit basis (opus, n=1, strict fact-level prompt); judged ≠ ground
  truth, it is the best point inside a structurally-bounded bracket.
- The dormancy price is a point estimate between two judged columns (no paired CI on that
  difference); sign and scale are unambiguous.
- Dormancy's 97.5% is the cold-archive extreme — a live store's recent (<60d) and used
  (≥3 accesses) memories are structurally shielded; the 07-15 fork shows the realistic
  rate. The extreme is still the right stress test.
- This run used serial NREM (`sleep/nrem-parallel` was parked behind the mid-run merge
  gate — run_iterated launches fresh subprocesses per pass, so a mid-run merge would have
  made pass 2 run different code than pass 1).

## Files touched

- `experiments/sleep-bench/findings-whole-corpus.md` (new)
- `experiments/sleep-bench/gen_undormant_arm.py` (new)
- `experiments/sleep-bench/impute_iterated.py` (judged-guard fix)
- `.gitignore` (+`experiments/sleep-bench/results-full/`)
- `docs/sessions/2026-07-10-sleep-bench-whole-corpus.md` (this file)
- Local only (gitignored): `results-full/` — all cell JSONLs, `summary_coverage.json`
  (opus eval table), `rem_m3__rrf.jsonl` (judge query-source shim = copy of `c2__rrf`)

## What's next

1. **Dormancy → downweight** (the H12 move applied to `process_dormancy`): H7-style
   generate-once sweep w∈{keep-active…status-death}, find the knee.
2. **REM summary prompt adopts H12's core-facts constraints** (numbers/names/dates/
   attribution) — the direct lever on the 55% fidelity rate.
3. **Pass-2 re-consolidation guard** in `gather_clusters`.
4. **`fork.py --scrub-access-counts`** for rig honesty.
5. **H10 reranker 2×2 — in flight the same day** (build session; its own session file
   lands when the 2×2 finishes): all 5 work-package scripts built and verified by opus
   agents, cross-judge audit passed (sonnet/opus binary agreement 49/50 = 98%), sonnet
   train-label judge running. Today's result sharpens the cell-② prediction: naive deploy
   on a slept store should regress recall unless the reranker learns to reach through
   summaries.
