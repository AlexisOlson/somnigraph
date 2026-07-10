# 2026-07-09: H11 closeout, archive-loss audit, H12 consolidation-node

Orchestrator session (main thread delegating to subagents for analysis, judging, recon,
implementation, and commit planning). Covers the H11 iterated-cycle closeout, the
archive-loss audit that reframed its headline, the H12 consolidate rewrite it motivated,
the H10 reranker-2×2 plan, and the whole-corpus direction set at session end.

## What shipped

1. **H11 (combined + iterated cycle) closed out.** c4 re-scored (the prior background
   re-score died mid-arm), full chain decomposition run, provenance bracket computed via a
   new chain adapter (`impute_iterated.py`). Findings:
   [`findings-h11-iterated-cycle.md`](../../experiments/sleep-bench/findings-h11-iterated-cycle.md).
2. **Archive-loss audit** (two-phase, same subagent: mechanical retrievability trace, then
   content judging of all 41 affected (query, archived-id) pairs). Findings:
   [`findings-h11-archive-loss-audit.md`](../../experiments/sleep-bench/findings-h11-archive-loss-audit.md).
3. **H12 consolidation-node pipeline change implemented** (`dba4203`, branch
   `sleep/consolidation-node`): consolidate synthesizes a de-duped core-facts node,
   `consolidated_from` edges preserve originals, members downweighted instead of archived;
   old behavior behind `--legacy-archive`. Spec:
   [`plan-h12-consolidation-node.md`](../../experiments/sleep-bench/plan-h12-consolidation-node.md).
   62/62 scratch-DB execute-path checks passed (mocked LLM judgments, real embeddings).
4. **H10 (pre/post-sleep reranker 2×2) planned**:
   [`plan-reranker-2x2.md`](../../experiments/sleep-bench/plan-reranker-2x2.md). Recon
   settled the trainer base: production `train_reranker.py` (reads `memory_edges`, store
   via `SOMNIGRAPH_DATA_DIR`), not the LoCoMo trainer (synthetic-graph features, hardcoded
   paths).

## Headline results

- **H11b (compounding) refuted.** Cycles 2–4 add nothing on any metric, scorer, or credit
  regime; cycle 2 is the worst single pass (−0.037 NDCG✱ / −0.045 R@10✱ raw). Summaries
  plateau at pass 1 (28), edges saturate (~984), and the edge effect Δ(formula−rrf) stays
  flat as edges triple — all value and all structure arrive in pass 1.
- **Consolidate was the eroder** — archived 22/133 distinct GT-evidence ids (16.5%) by c4,
  active set bled 419→360 monotonically, raw damage tracking archival exactly. Cumulative
  c4−frozen raw −0.075✱ NDCG / −0.126✱ R@10, but the provenance bound is **+0.127✱ /
  +0.105✱** — the bracket straddles zero.
- **Audit verdict: zero facts destroyed.** Content axis 40/41 PRESERVED, 0 LOST. The 23
  raw-damaged pairs decompose into 16 credit artifacts (stand-in retrieved and carries the
  fact; evidence-id scoring blind), 6 ranking failures (fact survives verbatim in a
  stand-in that never surfaced), 1 pre-existing LoCoMo GT gap (answer text never existed in
  any memory). "Sleep erodes the store" is dead as a content claim; it survives only as a
  ranking-precision claim — which H12 targets directly.
- **Guard correction empirically confirmed**: c1_rem Δ(formula−rrf) non-null
  (recall_any_effk +0.0204✱, NDCG −0.0151✱, MRR −0.0236✱) — unembedded REM edges traverse
  `expand_via_ppr`, closing the 07-04 guard question (`1053fbe`).

## Design decisions (Alexis, in-session)

- **No idempotence requirement on sleep passes.** Multi-pass consolidation is legitimate by
  design; the defect is that archive executes an unverified coverage claim. Fix = score the
  claim, not structural no-op gates.
- **`consolidated_into`/`consolidated_from` typed edges** — `superseded_by` is the wrong
  name for containment; reserve it for write-path supersession. Typed claims get typed
  verification and typed credit.
- **De-duped core-facts synthesis, not full verbatim union.** With members kept,
  losslessness is a system property, not a node property; verbatim union is the ranking
  pathology (long mixed node) plus recall-budget waste. Synthesis also enables temporal
  normalization (relative dates resolved to anchors), which verbatim append structurally
  cannot do.
- **07-15 fork runs the new pipeline** — old-pipeline benchmarking is low priority (it
  survives as the `--legacy-archive` flag if ever wanted).
- **Next claim: sleep works on the whole corpus** (all 10 conversations in one store), not
  per-conversation stores. Cross-conversation clustering worry: measure first (conv
  identity recoverable via dia_map; count cross-conv structures, judge synthesis samples
  for blending/attribution errors); links are free (H3), synthesis is where blending risk
  lives; 10 interleaved strangers is a conservative stressor vs one user's life.

## Surprises

- **Consolidate's superseders were already de-facto union nodes** — the merge/rewrite
  pattern appends member turns verbatim, with session-theme anchor parity verified on all 5
  temporal questions. The content half of H12 was accidentally shipped years-of-decisions
  ago; the harm was entirely retrieval-side (focused originals leaving the index).
- **The `derived_from` dormancy trap**: REM's `process_dormancy` treats `derived_from`
  targets of active summaries as dormancy candidates. Mirroring REM's edge type for
  consolidation provenance would have made members retrieval-dead through their own
  provenance edges — recreating exactly the harm H12 removes. Caught by the implementer's
  consumer analysis; the reason `consolidated_from` is its own edge type.
- **The formula scorer never reads `base_priority`** — the spec's assumed "formula boost
  path" doesn't exist. The member downweight is invisible to both bench scorers (rrf,
  formula); only the reranker feature and `effective_priority` consumers see it. H12's
  downweight knee is therefore only measurable under H10's rerankers.
- **REM does dormancy** (31 details at c1_rem, incl. 2 GT-evidence turns) — contaminates
  the "REM on top of edges" contrast; H7's leaf-only rig never triggered it.
- **29% of c4 edges touch ≥1 deleted endpoint** and `expand_via_ppr` loads edges with no
  status filter — deleted nodes absorb PPR mass and consume expansion slots before the
  post-hoc active filter. Vec/FTS hygiene is clean; the gap is edges only. Candidate
  production fix, not yet queued.
- **LoCoMo GT noise surfaced by the audit**: 2 tangential evidence pointers, 1 answer
  never present in any memory text (0:43 "abstract art", image-caption answer), 2
  speaker-swapped question duplicates.
- Subagent harness blocked one agent's markdown writes — findings files must be placed by
  the orchestrator from returned content.

## Caveats

- Everything is n=1 (conv0, 196 queries). Whole-corpus run is the designated replication
  vehicle.
- The archive-loss audit is a single subagent-as-judge pass (no inter-rater check);
  verdicts carry decisive quotes for auditability.
- The formal 232-pair summary-coverage judge run (for exact H11 bracket numbers) was made
  optional — the decision it guarded is answered. Recipe preserved in
  `findings-h11-iterated-cycle.md` § Result 5.
- H12 verified execute-path only (mocked judgments); the end-to-end bench run
  (pre-registered: raw ≥ frozen everywhere; the 6 H11 ranking-failure pairs recover) has
  not run yet.

## Files touched

- `experiments/sleep-bench/findings-h11-iterated-cycle.md` (new)
- `experiments/sleep-bench/findings-h11-archive-loss-audit.md` (new)
- `experiments/sleep-bench/impute_iterated.py` (new)
- `experiments/sleep-bench/plan-reranker-2x2.md` (new)
- `experiments/sleep-bench/plan-h12-consolidation-node.md` (new)
- `experiments/sleep-bench/results/c4__*.{jsonl,summary.json}` (new, gitignored)
- `scripts/sleep_consolidate.py` (+358/−20, commit `dba4203` on `sleep/consolidation-node`)
- `experiments/sleep-bench/plan-consolidate-summary-rewrite.md` (junked — superseded by
  H12's consolidation-node approach)

## What's next

1. Execute the commit/merge plan (H12 tip merge brings h7-rem-floor + iterated-cycle +
   consolidation-node to main in one `--no-ff` merge; source-scan separately).
2. Whole-corpus experiment: rig extension (all-convs ingest + GT), precheck, one cycle
   with the new pipeline (terminal), cross-conv instrumentation + blending judge.
3. H10 reranker 2×2, redesigned around the whole-corpus store (hold-out = per-conversation
   query splits).
4. Deferred/candidate: edge status filter in `expand_via_ppr`; coverage-proportional
   downweight (v2); result-set family collapse.
