# Findings — whole-corpus sleep (all 10 LoCoMo conversations, one store)

**Date:** 2026-07-09 (run, overnight, Alexis's terminal) / 2026-07-10 (measurement).
**Question:** does the current sleep pipeline (link-default NREM → member-floor REM → H12
consolidate) improve retrieval on a corpus-scale store — and does cross-conversation
clustering blend facts when 10 unrelated conversations share one store?
**Verdict:** **split, and now exactly decomposed.** The summary layer *works* at corpus
scale — judged **+0.078✱ NDCG** with details kept active, replicating H7 at 15× scale —
but REM's dormancy alone costs **−0.136 NDCG / −0.256 R@10**, flipping the net effect
negative (**R@10 −0.186✱**). Dormancy is the lone remaining eroder: H12 exonerated
consolidate (zero archives), and cross-conversation blending essentially did not occur.
55% summary fidelity is enough to **add** value, not enough to **replace** details.

Everything below is rerank-free (rrf = raw `rrf_fuse`; formula = rrf + PPR over
`memory_edges`), 196 conv0 GT queries against the combined store, paired bootstrap
(n_boot=10,000, seed=1234), ✱ = CI-clear. H12's member downweight is **invisible to both
scorers by construction** (nothing in rank fusion reads `base_priority`) — its payoff is
measured by H10's reranker arm, not here.

## Setup

- **Store:** fresh ingest of all 10 LoCoMo conversations into ONE store —
  `ingest_baseline.py --convs all` → 5,882 memories, merged conv-scoped dia_map
  (`"{conv}:{turn}"` keys; `gt.py` resolves them exactly). First real-data exercise of the
  `--convs` path: clean.
- **Run:** `run_iterated.py --max-passes 2 --nrem-mode deep` from `main@2c08f21` —
  i.e. the OLD serial NREM (the `sleep/nrem-parallel` patch was parked, unmerged, per the
  mid-run merge gate) and the NEW H12 consolidate (**first run with a real LLM**; prior
  validation was 62/62 mocked checks). Pass-1 NREM: 17,646 pairs / 57 batches / ~3.5 min
  each. Plateau rule fired at c1_consol (consolidate pass 1 ≈ no-op vs c1); pass 2 ran to
  completion.
- **Arms** (`~/.somnigraph/benchmark/sleepbench-full/`): frozen, c1_nrem, c1_rem, c1,
  c1_consol (byte-alias of c1), c2, plus the post-hoc counterfactual
  **c1_rem_undormant** (fork of c1_rem with all dormant rows reactivated —
  `gen_undormant_arm.py`, new script).
- **Scoring:** `retrieve.py --scorers rrf formula --out-dir results-full/` (separate dir:
  the default `results/` holds H11's raw results under the same labels and would have been
  overwritten). Credit bracket via `impute_iterated.py`; judged middle from a fresh opus
  coverage run (`judge_summary_coverage.py`, 2026-07-10, Alexis's terminal: 194 queries,
  229 pairs, 20 batches).

## State trajectory

| arm | active | edges | active summaries | dormant | deleted |
|---|---|---|---|---|---|
| frozen | 5,882 | 0 | 0 | 0 | 0 |
| c1_nrem | 5,882 | 5,900 | 0 | 0 | 0 |
| c1_rem | 534 | 12,604 | 389 | **5,737** | 0 |
| c1 | 537 | 12,613 | 392 | 5,737 | 0 |
| c2 | 541 | 13,013 | 396 | 5,737 | 1 |

NREM adds structure and touches nothing else (c1_nrem rrf results are **byte-identical**
to frozen — the H3 link-default zero-eviction property reproduces at scale). REM is where
the store transforms: 389 summaries covering 5,755/5,882 details (97.8%), and dormancy
for essentially every covered member. Consolidate adds 3 nodes in pass 1, 5 in pass 2.

## Result 1 — the raw metric collapse is a credit artifact, not content loss

Raw rrf from c1_rem onward: **NDCG 0.437 → 0.012, R@10 0.627 → 0.013**. This is
mechanically forced — ~97% of GT evidence turns are dormant and out of the index — and
almost entirely credit misassignment: the impute−raw gap is **+0.60 NDCG✱**, and for
78–83% of queries a summary generated from the exact evidence sits in the top ranks.
Never report the raw number as the headline; it measures the scorer, not the store.

## Result 2 — the bracket, collapsed by the judge

| c2 vs frozen (rrf) | NDCG | R@10 | MRR | recall_any@effk |
|---|---|---|---|---|
| raw (lower bound) | −0.425✱ | −0.614✱ | −0.353✱ | −0.719✱ |
| **judged (answer)** | **−0.054 [−0.13,+0.02]** | **−0.186✱** | **+0.004 (null)** | **−0.240✱** |
| provenance (upper bound) | +0.173✱ | +0.109✱ | +0.206✱ | +0.066 |

The provenance bound's +0.173✱ was largely **compression mirage**: the judge (strict,
fact-level: "a reader seeing only the summary has the same evidence") confirmed 125/229
provenance-claimed pairs (**55%** — statistically the same as H7's 52%, so REM's summary
fidelity did not degrade from 419 to 5,882 memories). The shape of the judged verdict is
diagnostic: **ranking survives 11:1 compression** (MRR at parity, NDCG null-leaning on 9%
of the nodes) — the entire loss is recall tail, queries whose specific fact exists only in
a dormant detail whose summary dropped it.

## Result 3 — ★ the undormant decomposition (the finding)

`c1_rem_undormant` = c1_rem with its 5,737 dormant rows reactivated (pure status flip —
verified `process_dormancy` never touched vec/FTS: both indexes held 6,271/6,271 rows).
Judged, vs frozen:

| effect | NDCG | R@10 | MRR |
|---|---|---|---|
| **summaries added, details kept** (undormant − frozen) | **+0.078✱** | **+0.071✱** | **+0.088✱** |
| **price of dormancy alone** (c1_rem − undormant, point est.) | −0.136 | −0.256 | −0.088 |
| net, sleep as-configured (c1_rem − frozen) | −0.058 | −0.186✱ | +0.000 |

Rows sum exactly — the decomposition is complete. Two conclusions:

1. **H7 replicates at 15× scale, n=2.** conv0-alone: +0.093✱ judged; whole-corpus:
   +0.078✱ with a fresh REM generation, a fresh judge, and 14× more distractors. The
   additive-summaries claim is now the program's most robust positive result.
2. **Dormancy is the same defect H12 fixed for consolidate, with a bigger price tag:**
   removal executing an unverified coverage claim (`derived_from` edge + age + low access
   ⇒ status-death; coverage never verified). The archive-loss arc's remedy applies
   verbatim — **downweight, don't status-kill** — and the knee is a generate-once
   H7-style sweep away.

## Result 4 — why dormancy fired at 97.5% (and why H11 said 7%)

`process_dormancy` requires: covered by an active summary + `recall+reflect < 3` +
created >60 days ago. On this store all three are trivially true for every covered
member (LoCoMo dates are 2022–2024; a fresh store has zero access counts) → coverage ⇒
dormancy. **H11's "31 dormanted" was instrument contamination:** every H3/H7/H11 scoring
pass calls `impl_recall`, which writes `recall_count` into the arm store — covered-active
details in the H11 arm show counts up to **187**, shielding 380/411 of them via the
access criterion. Same code, different history.

- **Rig lesson:** scoring mutates stores; a store that is scored and *then* slept behaves
  differently from fresh. Queued fix: `fork.py --scrub-access-counts` so sleep experiments
  fork clean by convention.
- **Regime honesty:** this corpus (years old, never accessed) is dormancy's extreme. A
  live store's recent memories (<60d) and used memories (≥3 accesses) are structurally
  shielded — the live 07-15 fork will show the realistic rate, not 97.5%. The extreme is
  still the right stress test: it is exactly the "cold archive" state a long-abandoned
  store reaches.

## Result 5 — cross-conversation blending: essentially none

The headline worry of the whole-corpus design mostly failed to materialize:

- **Detail–detail edges:** 200/6,852 cross-conv (**2.9%**); duplicate-link edges 1.1%.
  (The 1,682 blank-`edge_type` edges are duplicate links — they carry
  `flags=["duplicate"]` with the legacy column empty, by design.)
- **REM summaries:** **380/388 single-conversation.** All 8 mixed summaries span exactly
  2 convs, traced to 8 topically-generic clusters (`pets-and-dogs`, `gaming-hobby`, …).
  Sampled content keeps speakers correctly attributed and structured as parallel
  accounts ("Two pet-loving friend pairs. Audrey… Andrew…"), not blended facts.
- **Consolidation nodes: 0/7 mixed** — all conv-pure.

Topical coherence isolates conversations on its own. For the real use case (one user's
life in one store) this is the desired behavior with the risk bounded: synthesis, where
blending harm lives, stayed conv-pure; only cheap links crossed.

## Result 6 — first real-LLM H12 run: mechanics clean, one gap

- **Clean:** 8 clusters → 8 nodes, 18 members downweighted (×0.8; 9 per pass; once-ever
  flag held — members re-clustered in pass 2 were skipped), **0 archived, 0 schema rejects,
  0 failed clusters** across both passes (the pre-fix pipeline rejected 22/27 in Phase 0).
  Node content is core-facts style with session anchors and quotes preserved.
- **Gap — pass-2 re-consolidation:** pass 2 re-synthesized clusters pass 1 had already
  consolidated. Two identical-member-set node pairs exist (camping: `536b894a`/`217fcdb2`;
  turtles: `b82d8c54`/`d57204b8`), and the run's single deletion was write-path 0.9 dedup
  catching a third near-duplicate turtle node. `gather_clusters` never checks the
  `consolidated_member` flag, so node count grows per pass while members stay protected.
  Fix candidate: skip clusters whose members are majority-flagged.
- Node priorities (p5–p8) all trace to REM's between-pass audit retier moving member
  priorities — consistent with the old max-member rule; the parked `94aad00` priority
  band (floor 6) plus its retier SQL applies post-merge.
- Consolidate is retrieval-inert under both scorers here (c1 ≈ c1_rem everywhere; c2's
  raw R@10 uptick 0.0128→0.0179 is the only trace) — expected: keep+downweight touches
  nothing rrf/formula can see. H12's *absence of archival damage* is the measured win.

## Caveats

- Single corpus (LoCoMo), single run; edge sets are nondeterministic across runs (H11).
- rrf/formula only — no learned scorer; H12's downweight untested by construction (H10).
- The judge is the credit basis (opus, n=1 run, strict prompt); judged ≠ ground truth,
  it is the best point inside a bracket whose bounds are structural.
- The evidence-id GT itself has known noise (H11 audit: tangential pointers,
  answer-not-in-text, speaker-swapped duplicates).
- The dormancy price is a point estimate from two judged columns (no paired CI computed
  for that difference); its sign and scale are unambiguous.
- This run used serial NREM (old code); the parallel/stagger/dedup patch merged after —
  content-neutral except symmetric-pair dedup, which slightly lowers duplicate-edge yield
  on dense corpora going forward.

## Files / reproduction

- Arms: `~/.somnigraph/benchmark/sleepbench-full/conv0-{frozen,c1_nrem,c1_rem,c1,c1_consol,c2,c1_rem_undormant}`
- Raw results + judge table: `experiments/sleep-bench/results-full/` (gitignored, local;
  `rem_m3__rrf.jsonl` there is the judge's query-source shim = copy of `c2__rrf.jsonl`)
- New/changed harness this arc: `gen_undormant_arm.py` (new), `impute_iterated.py`
  (judged-coverage guard now checks the credit-eligible set, not all summaries),
  `.gitignore` (+`results-full/`)
- Judge recipe: `judge_summary_coverage.py --rem-dir <c2 arm> --results-dir results-full
  --model opus` (checkpoint-resumable); judged rescore: `impute_iterated.py --labels
  c1_rem c1 c2 c1_rem_undormant --judged-coverage results-full/summary_coverage.json`

## What this changes next

1. **Dormancy → downweight** (the H12 move, applied to `process_dormancy`): design the
   sweep w∈{keep-active…status-death}, measure the knee; H10's reranker learns the
   query-conditional arbitration.
2. **REM summary prompt** adopts H12's core-facts constraints (preserve numbers / names /
   dates / attribution) — the direct lever on the 55% fidelity rate.
3. **Pass-2 re-consolidation guard** in `gather_clusters` (skip majority-flagged clusters).
4. **`fork.py --scrub-access-counts`** for rig honesty.
5. **H10 reranker 2×2** — unblocked: the judge table doubles as training labels; today's
   result sharpens the cell-② prediction (naive deploy on a slept store should regress
   recall unless the reranker learns to reach through summaries).
