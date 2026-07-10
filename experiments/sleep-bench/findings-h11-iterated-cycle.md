# Findings — Sleep-Bench Step 3 (H11: combined + iterated cycle)

**Status: DRAFT, 2026-07-09. Judged bracket PENDING (blocked on an opus judge run — see
§ Result 5).** First measurement of the full production cycle (NREM → REM → consolidate) and of
iterating it, on one evolving store. Isolated benchmark arms only
(`~/.somnigraph/benchmark/sleepbench-arms/conv0-{frozen,c1_nrem,c1_rem,c1,c2,c3,c4}`); the live
store (`~/.claude/data`, abstinence hold until ~07-15) was never touched. Companion to
[`findings-h3-nrem-graph.md`](findings-h3-nrem-graph.md) (NREM alone),
[`findings-h7-rem-floor.md`](findings-h7-rem-floor.md) (REM alone), and
[`findings-phase0.md`](findings-phase0.md) (`cap_hit=50` — one pass undershoots steady state).
Pre-registered in [`plan-combined-iterated-cycle.md`](plan-combined-iterated-cycle.md).

## Questions (pre-registered)

- **H11a (stages add up):** on one store, does each stage's contribution survive in the presence
  of the others — in particular, does REM still help *on top of* NREM edges (H7 only measured
  REM-on-frozen)?
- **H11b (iteration compounds):** does a 2nd/3rd/4th full cycle keep improving retrieval, or
  plateau? Pre-registered expectation: pass 2 still creates edges (`cap_hit=50`), so *some* gain;
  open whether consolidate's archival (H3's deletion cost) eventually drags it down.
- **Headline:** `c4 − frozen` = the total steady-state effect of actually sleeping a fresh store,
  rerank-free.

## The run

`run_iterated.py` forked `conv0-frozen` (419-memory pristine ingest) and looped the full
production cycle `sleep_nrem.py deep → sleep_rem.py → sleep_consolidate.py -v` four times on the
same evolving store, with `SOMNIGRAPH_SUMMARY_MIN_MEMBERS=5` (the H7 shipped floor — without it
REM is inert on LoCoMo and "combined" silently degenerates to NREM-only). Snapshots after each
pass-1 stage (`c1_nrem`, `c1_rem`, `c1`) and at each later cycle boundary (`c2`–`c4`). Scoring:
the H3 rerank-free pair (`rrf` = content/index effect; `formula` = +PPR over `memory_edges`;
Δ(formula − rrf) = the edge effect), 196 GT queries, paired bootstrap (n_boot=10000, seed=1234)
via `analyze_iterated.py`.

## Result 1 — the store never plateaus: monotone active-set bleed

| arm | active | edges | summaries | dormant | deleted |
|---|---|---|---|---|---|
| frozen | 419 | 0 | 0 | 0 | 0 |
| c1_nrem | 419 | 351 | 0 | 0 | 0 |
| c1_rem | 416 | 830 | 28 | 31 | 0 |
| c1 | 395 | 860 | 28 | 31 | 21 |
| c2 | 371 | 974 | 28 | 31 | 45 |
| c3 | 364 | 980 | 28 | 31 | 52 |
| c4 | 360 | 984 | 28 | 31 | 56 |

(“active” includes the 28 summaries from c1_rem on: 416 = 419 − 31 dormant + 28.)

Three different asymptotics in one table:

- **Summaries plateau immediately**: pass 1's REM makes 28 leaf summaries; passes 2–4 add zero.
- **Edges plateau by pass 3**: +351 (NREM p1), +479 (REM p1), +30/+114 (p1/p2 consolidate+NREM),
  then +6, +4. The Phase 0 `cap_hit=50` prediction held for pass 2 (real new edges) and exhausted
  by pass 3.
- **The active set never plateaus — it bleeds monotonically, 419 → 360 (−14%).** Two mechanisms:
  - **Production REM runs dormancy** (`sleep_rem.py process_dormancy`): 31 details (7.4% of the
    store) went dormant in pass 1, including **2 GT-evidence turns**. H7 never saw this — its
    `gen_rem_arm.py` rig was leaf-summary-only. So the `c1_rem − c1_nrem` contrast below is
    summaries *plus* dormancy, not summaries alone.
  - **Consolidate archives every pass and never stops**: 21, 24, 7, 4 memories per pass
    (`status='deleted'`). By c4 it has archived **22 of the 133 distinct GT-evidence ids
    (16.5%)** — non-active counts per arm: 0 → 2 (dormancy) → 11 → 21 → 22 → 22. This is H3's
    deletion cost recurring as *slow erosion*: H3's destructive merge deleted evidence in one
    pass; consolidate's archival deletes a few every pass, forever. The run hit `--max-passes 4`
    without ever meeting the plateau stop-rule.

## Result 2 — raw retrieval: every stage is negative or null; nothing compounds upward

Per-cell means (RRF, raw evidence-id credit — a **lower bound** on summary arms, per H7):

| metric | frozen | c1_nrem | c1_rem | c1 | c2 | c3 | c4 |
|---|---|---|---|---|---|---|---|
| NDCG@5k | 0.4441 | 0.4441 | 0.4176 | 0.4068 | 0.3697 | 0.3662 | 0.3694 |
| R@10 | 0.6484 | 0.6484 | 0.5948 | 0.5723 | 0.5276 | 0.5174 | 0.5225 |
| MRR | 0.3693 | 0.3693 | 0.3409 | 0.3271 | 0.2978 | 0.2951 | 0.2989 |

Adjacent (per-stage / per-pass) deltas, RRF (✱ = 95% CI excludes 0):

| contrast | NDCG@5k | R@10 | reading |
|---|---|---|---|
| c1_nrem − frozen | **+0.0000 exactly** | **+0.0000 exactly** | link-default NREM is byte-identical to frozen under RRF — reproduces H3 Result 4's `nomerge` arm; the sanity gate passes |
| c1_rem − c1_nrem | −0.0265 ✱ | −0.0536 ✱ | raw penalty from 28 summaries displacing (H7 sign-flip, under-credited) + 31 dormancy removals |
| c1 − c1_rem (consolidate #1) | −0.0107 | −0.0225 | leans −, CI includes 0; archived 21 (11 GT-evidence non-active by here) |
| c2 − c1 (cycle 2) | −0.0371 ✱ | −0.0446 ✱ | the worst pass: consolidate archived 24 more (GT-evidence non-active 11→21) |
| c3 − c2 (cycle 3) | −0.0035 | −0.0102 | ~null (r10 CI [−0.0255, 0.0000] borderline) |
| c4 − c3 (cycle 4) | +0.0032 | +0.0051 | ~null, leans + |

Cumulative headline, `c4 − frozen`: **RRF −0.0747 ✱ NDCG / −0.1259 ✱ R@10; formula −0.0888 ✱ /
−0.1105 ✱.** All six metrics CI-clear negative under both scorers.

**H11b answered (raw): iteration does not compound gains — it compounds erosion, then saturates.**
The damage tracks the GT-evidence archival curve exactly: big hits while consolidate is eating
evidence turns (passes 1–2), ~flat once the erosion saturates at 22/133 (passes 3–4). Nothing in
passes 2–4 added retrieval value on any metric under either scorer.

## Result 3 — the graph effect is real but does not grow with the edge count

Δ(formula − rrf) per arm (frozen = +0.0000 exactly on all six metrics — instrument still clean):

| arm (edges) | NDCG@5k | MRR | recall_any@effk | recall_any@10 |
|---|---|---|---|---|
| c1_nrem (351) | −0.0139 | −0.0206 ✱ | +0.0051 | −0.0051 |
| c1_rem (830) | −0.0151 ✱ | −0.0236 ✱ | **+0.0204 ✱** | +0.0051 |
| c1 (860) | −0.0127 ✱ | −0.0222 ✱ | +0.0255 (CI [−0.0001, +0.0561]) | +0.0204 |
| c2 (974) | −0.0121 ✱ | −0.0173 ✱ | +0.0204 | +0.0204 |
| c3 (980) | −0.0124 ✱ | −0.0164 ✱ | +0.0204 | +0.0204 |
| c4 (984) | −0.0141 ✱ | −0.0183 ✱ | +0.0153 | +0.0204 (CI [0.0000, +0.0459]) |

Two findings:

- **Empirical confirmation of the guard correction (commit `1053fbe`).** REM's edges carry **no
  `linking_embedding`**, and the retired edge-embedding guard would have *refused* to score these
  arms as a predicted false-null. The measured Δ(formula − rrf) at c1_rem is **non-null**
  (recall_any@effk +0.0204 ✱; NDCG −0.0151 ✱; MRR −0.0236 ✱) and *changes* from c1_nrem when
  REM's 479 unembedded edges land — `expand_via_ppr` traverses by weight and never reads
  `linking_embedding`, exactly as the code-reading found. The refusal premise was wrong;
  measurement agrees.
- **Same recall↔ranking tradeoff shape as H3, and it saturates.** Edges nearly triple
  (351 → 984) across the chain while the graph effect stays flat: ~−0.012 NDCG / ~−0.018 MRR
  cost, ~+0.02 recall_any lean (mostly not CI-clear at n=196). More edges past pass 1 buy
  nothing measurable. One divergence from H3 worth recording: at c1_nrem the recall side is
  *null* (H3's isolated NREM arm had recall_any@10 +0.041 ✱) — different nondeterministic edge
  set, link-default rather than merge, single run each; the direction (ranking cost, recall
  lean) reproduces, the magnitudes don't.

## Result 4 — provenance bracket: the raw sign is (again) not trustworthy

H7 proved raw evidence-id metrics are a **lower bound** on summary-bearing arms (a summary that
out-ranks the evidence turn it covers scores as a miss). `impute_iterated.py` (new; the chain
adapter for `impute_rescore.py`, which is hardcoded to H7's floor-sweep rungs) re-credits each
retrieved summary at its rank to the evidence it was `generated_from` — the **PROVENANCE upper
bound**. Against `frozen` (raw), paired, RRF:

| arm | RAW Δ NDCG (lower) | PROVENANCE Δ NDCG (upper) | PROVENANCE Δ R@10 |
|---|---|---|---|
| c1_rem | −0.0265 ✱ | **+0.1366 ✱** | +0.1314 ✱ |
| c1 | −0.0372 ✱ | +0.1316 ✱ | +0.1178 ✱ |
| c2 | −0.0743 ✱ | +0.1184 ✱ | +0.0948 ✱ |
| c3 | −0.0778 ✱ | +0.1190 ✱ | +0.0948 ✱ |
| c4 | −0.0747 ✱ | **+0.1271 ✱** | +0.1050 ✱ |

(Formula-scorer bracket is the same shape, upper bound slightly higher: c4 +0.1583 ✱ NDCG.)

- **The bracket straddles zero on every summary arm** — the total steady-state sleep effect is
  **sign-ambiguous pending the judge**. H7's precedent (judged landed ~76% of the way from RAW to
  PROVENANCE) suggests positive, but that ratio is an extrapolation from a different summary set;
  it is not a measurement.
- **The evidence-id artifact *grows* with iteration** (impute − raw: +0.163 NDCG at c1_rem →
  +0.202 at c4, RRF): the summaries increasingly out-rank — and provenance-cover — evidence that
  consolidate *archived*. Under full provenance credit, part of the archival erosion is
  recoverable *if* the summaries actually carry the archived facts. That "if" is exactly what the
  judge decides.
- **Under provenance credit, iteration is still flat-to-negative**: imputed means are essentially
  constant from c1_rem to c4 (RRF NDCG 0.5807 → 0.5712). Even in the most generous credit regime,
  passes 2–4 add nothing; all measurable value is pass-1 REM.

## Result 5 — judged bracket: BLOCKED on an opus judge run

The H7 artifact `results/summary_coverage.json` covers **0 of the 28** H11 summaries (verified by
id: H11's REM pass is a fresh LLM generation — same corpus, same floor, different summary rows).
All 28 need judging (short ids): `01ef147f 1f52ce0b 2063331e 225d5965 2918538c 2ca0bfee 306fd987
36fb30a4 3ddeb9a2 433e25f4 475bd581 522820d4 65042419 692457f2 728958df 779512f9 7d5ee26d
7dfb65e5 8dc9f760 9f3c276a a19b694d c498d4fa c90b2f94 cd8550d5 d9c9801d e5b64d60 ec91f339
fb63a5af`. Dry-run: **195 queries, 232 (query, summary) pairs, ~20 opus batches** — same scale as
H7's judge run. LLM judge runs are terminal jobs (not in-session):

```powershell
cd C:\Users\Alexis\repos\somnigraph
$env:PYTHONUTF8 = "1"
# judge_summary_coverage.py hardcodes its output to results\summary_coverage.json --
# park the H7 artifact first so it isn't overwritten, then restore it.
Move-Item experiments\sleep-bench\results\summary_coverage.json experiments\sleep-bench\results\summary_coverage_h7.json
uv run experiments/sleep-bench/judge_summary_coverage.py --rem-dir C:\Users\Alexis\.somnigraph\benchmark\sleepbench-arms\conv0-c1_rem --model opus
Move-Item experiments\sleep-bench\results\summary_coverage.json experiments\sleep-bench\results\summary_coverage_h11.json
Move-Item experiments\sleep-bench\results\summary_coverage_h7.json experiments\sleep-bench\results\summary_coverage.json
# then the judged answer inside the bracket (LLM-free, any session):
uv run experiments/sleep-bench/impute_iterated.py --judged-coverage experiments/sleep-bench/results/summary_coverage_h11.json
```

(The judge sources its query/evidence set from `results/rem_m3__rrf.jsonl` — valid here because
the GT query set and evidence ids are fork-invariant across all conv0 arms. `--rem-dir` points at
`conv0-c1_rem` for the summary content + `generated_from` provenance; c1–c4 carry the identical
28 summaries.)

## Consolidate's hygiene gap: dangling edges feed PPR mass to deleted nodes

Consolidate archives memories but **never touches `memory_edges`**. At c4: **34 edges have both
endpoints deleted** (the integrity warning), and **287 of 984 edges (29%) touch at least one
deleted endpoint**. This is not cosmetic: `scoring.expand_via_ppr` loads edges with **no status
filter**, so deleted nodes participate in the PPR walk, absorb pagerank mass, and can consume
`MAX_EXPANSION_TOTAL` slots — they are only discarded *afterwards* by the active-filter (step 8).
Every iteration makes the graph a little more haunted. Cheap fixes, in preference order: prune
(or down-weight) edges on archival in consolidate, or exclude non-active endpoints when building
the PPR adjacency. (For scale: index hygiene is otherwise clean — 0 deleted rows left in vec/FTS.)

## Verdict

- **H11a (stages add up) — partially, and only via pass-1 REM.** Link-default NREM is exactly
  free under RRF (the H3 flip working as designed) and its edges give the familiar
  ranking-cost/recall-lean tradeoff; REM's summaries are the only stage that can plausibly add
  value, and whether they net positive over the whole cycle is exactly the open judged bracket
  (raw −0.075 ✱ ≤ truth ≤ provenance +0.127 ✱ at c4).
- **H11b (iteration compounds) — refuted.** No metric improves after pass 1 under any credit
  regime. Summaries and edges plateau by pass 2–3; what does *not* plateau is consolidate's
  archival, which monotonically bleeds the active set (419 → 360) and had eaten 16.5% of the GT
  evidence by c4. The pre-registered "archival cost outrunning canonicalization" branch is the
  one that happened.
- **The actionable finding is about consolidate, not NREM/REM:** it re-archives every pass with
  no stopping condition, its archival is H3's deletion cost in slow motion, and it leaves
  dangling edges that pollute PPR. Consolidate needs (a) a convergence/stop condition, (b) the
  same link-don't-delete reconsideration NREM got in H3 (or at minimum evidence-aware
  conservatism), and (c) edge hygiene on archival.
- **The retired edge-embedding guard's premise is now refuted empirically**, not just by
  code-reading: 830+ unembedded REM edges produced a non-null, CI-clear Δ(formula − rrf).

## Caveats (honest accounting)

- **n = 1 conversation, single run per arm.** Sleep stages are LLM-nondeterministic; c1_nrem's
  graph-effect magnitudes already differ from H3's isolated NREM run. Directions, not magnitudes,
  are the transferable claims.
- **Raw metrics are a lower bound on every summary arm (c1_rem onward)** — the H7 sign-flip. The
  cumulative raw negatives are *not* yet evidence that the full cycle hurts; the bracket
  straddles zero. Conversely the provenance bound over-credits (compression). **The judged run is
  load-bearing for the headline.**
- **`rrf`/`formula` only — no learned reranker.** The V6/production interaction is H10's
  question, untouched here.
- **`c1_rem − c1_nrem` conflates summaries with dormancy** (production `sleep_rem.py` dormanted
  31 details, 2 of them GT evidence). A dormancy-off ablation would cost one more LLM pass and
  wasn't run.
- **The c4 arm was re-scored 2026-07-09** after the original background run died mid-arm
  (0-byte JSONL); both scorers re-ran clean (196/196 rows, `[formula fallback]` scorer status as
  designed for the rig).
- The plateau stop-rule never fired — `--max-passes 4` is the horizon. The store-state trend
  (7 → 4 archived in passes 3 → 4) suggests eventual quiescence, but that's extrapolation.

## Reproduction

```powershell
$ARMS = "C:\Users\Alexis\.somnigraph\benchmark\sleepbench-arms"
$env:PYTHONUTF8 = "1"
$env:OPENAI_API_KEY = (Get-Content "C:\Users\Alexis\.claude\data\openai_api_key" -Raw).Trim()

# 1. The 4-pass cycle (LLM, terminal, ~minutes per stage) -- already run:
uv run experiments/sleep-bench/run_iterated.py --frozen "$ARMS\conv0-frozen" --arms-dir $ARMS --conv 0 --max-passes 4

# 2. Score every arm (LLM-free; c4 shown -- repeat per label):
uv run experiments/sleep-bench/retrieve.py --data-dir "$ARMS\conv0-c4" --label c4 --conv 0 --scorers rrf formula --fresh

# 3. Chain decomposition + provenance bracket (LLM-free):
uv run experiments/sleep-bench/analyze_iterated.py --labels frozen c1_nrem c1_rem c1 c2 c3 c4
uv run experiments/sleep-bench/impute_iterated.py

# 4. Judged bracket: see Result 5 (opus judge -> terminal), then impute_iterated.py --judged-coverage.
```

Artifacts in `experiments/sleep-bench/results/`: `{frozen,c1_nrem,c1_rem,c1,c2,c3,c4}__{rrf,formula}.jsonl`
(+ `.summary.json`). New script: `impute_iterated.py`.
