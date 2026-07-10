# 2026-07-04 — Sleep-Bench Step 2 (H7): REM summary floor

Second step of the sleep-bench program (after Phase 0 and Step 1/H3, same day).
Full finding: [`experiments/sleep-bench/findings-h7-rem-floor.md`](../../experiments/sleep-bench/findings-h7-rem-floor.md).

## What shipped

- **Member-count summary floor** (`scripts/sleep_rem.py`). Replaced the absolute
  `SUMMARY_MIN_DETAIL_TOKENS=2000` leaf floor — which produced **zero** summaries on
  LoCoMo (Phase 0 Finding 1) — with `SUMMARY_MIN_MEMBERS` (env `SOMNIGRAPH_SUMMARY_MIN_MEMBERS`,
  default `5` = 1× `RECALL_LIMIT_REF=5`). Principle: a summary earns its place only when its
  topic has more members than one recall returns (recall budgets in *items*, not tokens).
  The token constant is demoted to gating parent-summary creation only.
- **Five new experiment scripts** (`experiments/sleep-bench/`): `gen_rem_arm.py` (fork +
  leaf-only REM at the low floor), `sweep_rem_floor.py` (toggle + rescore per rung),
  `analyze_rem_floor.py` (raw curve), `impute_rescore.py` (provenance + judged bracket),
  `judge_summary_coverage.py` (LLM coverage judge, reuses `judge_synthetic_coverage.py`).

## Headline

On LoCoMo conv0 (196 GT queries, `rrf`), REM leaf summaries produce a **CI-clear retrieval
gain: +0.093 NDCG / +0.097 R@10 (judged)** at full summarization. The naive **raw** metric
says the opposite — −0.046 NDCG, CI-clear — because summaries aren't evidence turns and
out-rank them, so evidence-id scoring counts a coverage win as a miss. The **sign is
flipped by the credit model.** Floor `=5` (1×K) is validated (statistically tied with the
nominally-best `=3`; the 3 extra summaries are 4-member clusters that cover ~nothing).

## Surprises

- **The verdict inverts on the credit model.** Raw: summaries hurt, CI-clear. Judged:
  summaries help, CI-clear. Same retrieval, opposite sign. A reviewer reading only the raw
  sweep would have concluded "REM summaries are a regression" and been exactly wrong — the
  same evidence-id artifact class that made H3's merge look bad, but here it flips a *sign*,
  not just a magnitude.
- **Alexis's "impute the credit to the origin components" reframe was cleaner than my
  plan.** I had proposed a binary artifact-vs-real *classification* of displaced queries.
  His reframe turned it into a continuous **bracket** (raw ≤ judged ≤ provenance) where the
  provenance upper bound is free (structural, `generated_from`) and the LLM judge is only
  needed to collapse a gap that spans zero. Better instrument, less code.
- **The judge rejected ~48% of provenance coverage yet the judged number kept ~76% of the
  upside.** The rejected pairs were low-value (summary retrieved but irrelevant to that
  query); the answering pairs survived. Provenance over-credits on *count*, not on *value*.
- **First clear positive sleep-stage result in the program.** H3 (NREM) was a
  recall↔ranking tradeoff; the sleep-fork (consolidation) was inert. REM summary generation
  is the first sleep stage with an evidenced retrieval win.

## Caveats

- n=1 (conv0); `rrf`-only (no reranker); judge is the credit basis (reader-equivalence,
  opus). The turn-down point (floor < 3) is untested and is the only non-cheap direction.
- **Instrument residual:** `rem_none − frozen` is not exactly 0 (r@10 −0.011, CI incl 0) —
  inactive summary rows still perturb `rrf` ranks in the index before the active-filter. It
  does not affect the sweep (all rungs share the identical index); noted, not papered over.

## Files touched

- `scripts/sleep_rem.py` — member-count floor + constants (only production change).
- `experiments/sleep-bench/{gen_rem_arm,sweep_rem_floor,analyze_rem_floor,impute_rescore,judge_summary_coverage}.py` — new.
- `experiments/sleep-bench/findings-h7-rem-floor.md` — new.
- `experiments/sleep-bench/results/` — `rem_{none,m25,m15,m10,m5,m3}__rrf.jsonl`,
  `rem_floor_sweep_manifest.json`, `summary_coverage.json`.

## Reversibility

- The one production change is the floor (member-count, default 5). It **changes live REM
  behavior** when abstinence ends (2026-07-15 fork) — from token(2000) to members(5). Intended
  fix, flagged on the diff. `--legacy`-style opt-out not added; the old token path survives
  only for parent summaries.
- Experiment scripts are additive. `conv0-rem` arm is a disposable fork; the live store and
  the unrelated 2026-07-03 working tree were never touched.
- Nothing committed — the diff (floor + 5 scripts + finding + this file + changelog) awaits
  Alexis's review; `main` remains ~26 commits ahead of origin, unpushed.

## Retrospective (the 3 questions)

1. **Surprised?** Yes — the sign flip (above). The raw sweep, taken at face value, gives the
   wrong answer with full statistical confidence. The value was entirely in refusing to read
   the raw number as the result.
2. **Priority ordering still right?** Yes, no reorder. H7 is a P5 sub-thread. Worth recording
   that REM summary generation is now an *evidenced* retrieval lever (not just a plausible
   one) — relevant if P5/P6 benchmark work wants a sleep-stage that demonstrably moves recall.
3. **What does STEWARDSHIP/docs need that they don't have?** Two newly-surfaced items: (a) the
   **consolidate summary-rewrite TODO is now unblocked** (REM produces summaries — it was
   untestable before); (b) the **K-anchor-for-live-instance** question — `RECALL_LIMIT_REF=5`
   is `impl_recall`'s *default* limit, but real usage passes larger limits, so the floor's
   "1×K" anchor for the live store is unverified and measurable read-only from the event log.

## What's next

1. **conv1–2 replication** — kill the n=1 caveat on the sign-flip + floor result. Cheap
   (fork + one gen + sweep + judge per conv).
2. **Consolidate summary-rewrite** (the deferred H3 TODO, now testable) — fold duplicate-edged
   summary pairs into one canonical node.
3. **Reranker 2×2 (H10)** — can now learn summaries, not just NREM edges.
4. **(Optional) floor<3 turn-down probe** — regeneration + judge top-up; the one non-cheap
   direction, answers "where does over-summarizing finally hurt."
5. **(Live-instance) recall-limit distribution** from the event log — sets the real K for the
   shipped floor; read-only, abstinence-safe.
