# Findings — Sleep-Bench Step 2 (H7: REM summary floor)

**Status: DRAFT, 2026-07-04.** First retrieval measurement of REM summaries on a
slept fork. Isolated benchmark arms only (`~/.somnigraph/benchmark/sleepbench-arms/conv0-{frozen,rem}`);
the live store (`~/.claude/data`, abstinence hold until ~07-15) was never touched.
Companion to [`findings-h3-nrem-graph.md`](findings-h3-nrem-graph.md) (NREM) and
[`findings-phase0.md`](findings-phase0.md) Finding 1 (REM inert on LoCoMo). Depends on
the [sleep-pipeline proposal](../../docs/proposals/sleep-pipeline-optimization.md) §4
Stage C.

## Question

Phase 0 found REM generates **zero** summaries on LoCoMo — `SUMMARY_MIN_DETAIL_TOKENS=2000`
is an absolute token floor that no short-turn cluster clears. H7: unblock REM with a
corpus-invariant floor, then measure **do REM leaf summaries help retrieval, and at what
floor?**

## Two decisions settled before measuring

**1. The floor should be member-count relative to the recall limit, not an absolute
token sum.** The floor's own justification — "skip topics small enough to retrieve
directly" (`sleep_rem.py`) — is about *retrievability*, and recall budgets in **items**
(`impl_recall` default `limit=5`), not tokens. The absolute token floor was a
corpus-fragile proxy that silently disabled REM on short-turn corpora. Shipped:
`SUMMARY_MIN_MEMBERS` (env-configurable, default `5` = 1× the recall limit `RECALL_LIMIT_REF`);
a cluster is summarized only when it has more members than one recall can surface. The
old token constant is demoted to gating parent (summary-of-summaries) creation only.

**2. Generate-once, subset-to-sweep.** Lowering the floor is *monotone* — a summary that
exists at floor=3 is byte-identical to the one at floor=10; the floor decides *whether*,
never *how*. So one leaf-generation pass at the lowest rung (floor=3) produces the whole
summary set, and each sweep rung is a `status` toggle + local rescore. Cuts LLM cost from
Nx to 1x and removes cross-rung nondeterminism. Leaf-only (no parent/maintenance) so every
summary's `generated_from` is a pure detail-member list.

## The instrument (three-regime credit bracket)

conv0-frozen (419 memories) → one leaf-REM pass at floor=3 (taxonomy assigned 419 memories
to **33 topics**; all 33 clusters cleared floor=3 → **33 leaf summaries**, member counts
34…4). 196 scoreable GT queries. Scored `rrf` only: REM-on-frozen has no NREM edges, so
PPR has nothing to traverse (`formula ≡ rrf`), and `rrf` skips the edge-embedding health
guard. Summaries are new content/index nodes — exactly the `rrf` channel. (V6 dropped as
OOD, per H3.)

REM summaries are **not** evidence turns, so `retrieve.py`'s `gt_scores` credit only
evidence ids — a summary that out-ranks its own evidence turn scores as a *miss*. Three
credit regimes bracket the truth:

- **RAW** — credit only evidence ids (summaries credited with nothing). **Lower bound.**
- **JUDGED** — rewrite each retrieved summary into the evidence ids an LLM judge confirms
  it covers ("a reader seeing only the summary has the same evidence"; strict, topic
  overlap insufficient). **The answer.**
- **PROVENANCE** — credit a retrieved summary with *all* its `generated_from` evidence.
  Exact provenance, but a dense ~600-char summary may have compressed away a turn's
  specific fact, so it over-credits. **Upper bound.**

`RAW ≤ JUDGED ≤ PROVENANCE` by construction. Control is `rem_none` (0 summaries active ==
frozen). Instrument check: `rem_none − frozen ≈ 0` (r@10 −0.011, CI includes 0) — small
residual because inactive summary rows still sit in the vec/FTS index and perturb `rrf`
ranks before the active-filter; it does **not** touch the sweep, where every rung shares
the identical index and only the active-set differs.

## Result 1 — the raw sign is wrong: REM summaries help, CI-clear

At full summarization (`rem_m3`, 33 summaries), Δ vs `rem_none`, paired bootstrap (n=196):

| metric | RAW (lower) | **JUDGED (answer)** | PROVENANCE (upper) |
|---|---|---|---|
| NDCG@5k | −0.046 ✱ | **+0.093 ✱** | +0.136 ✱ |
| R@10 | −0.043 ✱ | **+0.097 ✱** | +0.159 ✱ |
| MRR | −0.051 ✱ | **+0.091 ✱** | +0.130 ✱ |
| recall_any@10 | −0.041 ✱ | **+0.097 ✱** | +0.153 ✱ |
| recall_all@10 | −0.046 ✱ | **+0.092 ✱** | +0.158 ✱ |
| recall_any@effk | −0.071 ✱ | **+0.061 ✱** | +0.117 ✱ |

(✱ = 95% bootstrap CI excludes 0.) The naive raw reading ("summaries hurt −0.046, CI-clear")
is a **pure evidence-id scoring artifact** — the summary out-ranks its evidence turn and
raw scoring calls that a miss. Credit the summary for what it actually covers and the same
summaries are **CI-clear helpful on every metric**. The vocabulary bridge, confirmed at the
retrieval layer: a topic's summary is reliably retrieved when that topic's evidence is
relevant.

The judge (opus, 20 batches, reused `judge_synthetic_coverage.py`) confirmed **123 / 236**
provenance pairs as real coverage (~52%); it rejected ~48% as topical-but-not-answering —
exactly the over-credit the upper bound exists to expose. Yet JUDGED lands at **~76% of the
way from RAW to PROVENANCE** on NDCG: the rejected pairs were mostly low-value (summary
retrieved but irrelevant to *that* query); the pairs that answer were largely confirmed.

## Result 2 — the floor: gain is monotone-saturating; default 5 (=1×K) is validated

JUDGED Δ(rung − `rem_none`), NDCG@5k:

| floor | summaries | JUDGED NDCG Δ |
|---|---|---|
| ≥25 | 3 | +0.005 (CI incl 0) |
| ≥15 | 12 | +0.044 ✱ |
| ≥10 | 18 | +0.070 ✱ |
| **≥5 (1×K)** | **30** | **+0.089 ✱** |
| ≥3 | 33 | +0.093 ✱ |

Monotone-increasing as the floor drops, saturating by floor=5. **floor=3 is nominally best
but statistically tied with floor=5**: 3 of 6 metrics are byte-identical, the rest differ
by ~0.004–0.005 deep inside overlapping CIs. The reason is concrete — floor=3 adds exactly
the three 4-member clusters over floor=5, and those smallest clusters cover almost nothing
extra. **No U-turn**: on LoCoMo more summaries never hurt (judged), so the floor's role here
is cost/noise control, not damage control.

**Decision (2026-07-04):** ship `SUMMARY_MIN_MEMBERS=5` (=1×K) as the default — tied on
quality with fewer nodes, more conservative against over-summarizing tiny clusters (the
write-path noise risk on denser personal stores), and a principled anchor rather than a
benchmark-fit number. LoCoMo runs may use `=3` if marginally-max recall is wanted; either
is fine (Alexis).

## Verdict

- **REM leaf summaries improve LoCoMo conv0 retrieval, CI-clear: +0.093 NDCG / +0.097 R@10
  (judged).** The first clear *positive* retrieval result from a sleep stage in this program
  (H3 NREM was a recall↔ranking tradeoff; the sleep-fork consolidation was inert).
- **The naive raw metric is sign-flipped** — evidence-id scoring makes a coverage win look
  like a −0.046 loss. Any future summary/rewrite stage must be scored with coverage credit,
  not raw evidence ids.
- **The member-count floor unblocks REM and `=5` (1×K) is the validated default.**
- **Method contribution:** the three-regime `RAW ≤ JUDGED ≤ PROVENANCE` bracket is a
  reusable instrument — the provenance upper bound is free (structural), and the LLM judge
  is only needed to collapse a sign-ambiguous gap.

## Caveats (honest accounting)

- **n = 1 conversation** (conv0). Magnitudes indicative; conv1–2 replication is the named
  follow-up to kill n=1.
- **`rrf` channel only** — no production reranker, which may rank summaries differently.
- **The judge is the credit basis** — reader-equivalence standard, opus, ~87.5% Opus/Sonnet
  agreement precedent (L5b). Not an oracle.
- **The turn-down point is untested.** The curve is monotone-up through floor=3; whether an
  even-lower floor (1–2, "summarize everything") finally turns down from noise is the one
  informative untested rung, and the only *non-cheap* direction (needs regeneration + a
  judge top-up — the tiny clusters weren't summarized).

## Reproduction

```powershell
$FROZEN = "C:\Users\Alexis\.somnigraph\benchmark\sleepbench-arms\conv0-frozen"
$REM    = "C:\Users\Alexis\.somnigraph\benchmark\sleepbench-arms\conv0-rem"
$env:PYTHONUTF8 = "1"
$env:OPENAI_API_KEY = (Get-Content "C:\Users\Alexis\.claude\data\openai_api_key" -Raw).Trim()

# 1. Generate REM arm — leaf summaries at member floor 3 (LLM, ~10 min)
uv run experiments/sleep-bench/gen_rem_arm.py --frozen $FROZEN --rem $REM --min-members 3 --force
# 2. Sweep the floor — toggle + rescore per rung (rrf, local)
uv run experiments/sleep-bench/sweep_rem_floor.py --rem-dir $REM --conv 0
# 3. Raw curve + provenance bracket
uv run experiments/sleep-bench/analyze_rem_floor.py
uv run experiments/sleep-bench/impute_rescore.py --rem-dir $REM
# 4. Coverage judge (LLM, ~20 opus batches) + the judged answer inside the bracket
uv run experiments/sleep-bench/judge_summary_coverage.py --rem-dir $REM --model opus
uv run experiments/sleep-bench/impute_rescore.py --rem-dir $REM `
    --judged-coverage experiments/sleep-bench/results/summary_coverage.json
```

Artifacts in `experiments/sleep-bench/results/`: `rem_{none,m25,m15,m10,m5,m3}__rrf.jsonl`,
`rem_floor_sweep_manifest.json`, `summary_coverage.json`.

## What this unblocks

REM now produces summaries on LoCoMo, so the deferred **consolidate summary-rewrite** TODO
(fold duplicate-edged summary/gestalt pairs into one canonical node — [H3 finding]
(findings-h3-nrem-graph.md) § Result 4) is finally testable. Sequence: this → consolidate
rewrite → the reranker 2×2 (H10) can now learn summaries, not just NREM edges.
