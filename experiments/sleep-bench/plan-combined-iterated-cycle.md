# Plan — Combined + iterated sleep cycle (H11)

**Status:** captured spec, not built/run. Fills the gap between H3/H7 (single isolated stages,
separate forks) and H10 (reranker interaction).
**Why this before H10:** H3 measured NREM edges with **zero summaries**; H7 measured REM
summaries with **zero edges** (`gen_rem_arm.py` forks *frozen*, not the NREM arm). The combined
`NREM+REM` store was **never scored**, and every scored run is a **single pass** — yet Phase 0
proved one NREM pass undershoots steady state (`cap_hit=50`). This experiment measures whether
the two stages add up in one store, and whether iterating the cycle compounds.
**Cost:** heavy but rerank-free — each pass is a real LLM sleep (Sonnet NREM + opus REM + opus
consolidate), sequential, no gen-once shortcut across passes. → your terminal.

## Questions / hypotheses

- **H11a (stages add up):** on one store, `frozen → NREM → REM → consolidate`, does each stage's
  contribution survive in the presence of the others? Specifically, does REM still help *on top
  of* NREM edges (H7 only showed REM-on-frozen)?
- **H11b (iteration compounds):** does a 2nd/3rd full cycle keep improving retrieval, or plateau?
  Pre-registered expectation: pass 2 still creates edges (Phase 0 `cap_hit=50`), so *some* gain;
  open whether it's monotone or the consolidate archival (H3's deletion cost) eventually drags it
  down.
- **Headline number:** `cycle_N − frozen` = the total steady-state effect of actually sleeping a
  fresh store, rerank-free.

## ⚠️ Load-bearing prerequisite

The member-count summary floor (`SUMMARY_MIN_MEMBERS`) lives **only on branch
`sleep-bench/h7-rem-floor`** — on `main`, `scripts/sleep_rem.py` still has the old token floor
(2000) that produces **zero** LoCoMo summaries. **Run this experiment with that branch checked
out (or merged)**, or REM is inert and the "combined" arm degenerates to NREM-only. The judged
rescore rig (`impute_rescore.py`, `judge_summary_coverage.py`) is on that branch too.

## Design

- **Cycle** = `sleep_nrem.py deep` → `sleep_rem.py` → `sleep_consolidate.py` (Phase 0 order).
  Use the **production stages** (real REM, incl. parent-summary generation + maintenance) with
  `SOMNIGRAPH_SUMMARY_MIN_MEMBERS=5` (shipped floor). *Note:* this bundles parent-summaries +
  maintenance, so `post-REM-1` here is **not** directly comparable to H7's isolated leaf-only
  number — we want the real cycle.
- **conv0, n=1** (consistent with H3/H7; convs paused).
- **Snapshots (arms) to score**, each a byte-identical fork frozen at that point via `fork.py`:
  - `frozen` (exists) — baseline.
  - Pass 1, per stage: `c1_nrem`, `c1_rem`, `c1_consol`.
  - Passes 2..N, per cycle boundary: `c2`, `c3`, … (scoring is cheap/LLM-free, so score every
    stage boundary if you want; per-cycle is the minimum informative set after pass 1).
- **Stop rule:** iterate until a full cycle's DB-delta is ~0 (NREM adds no edges — all pairs
  `edge_exists`/cap not hit; REM adds no summaries; consolidate proposes no actions) **or** cap
  at **N=4**. Use `precheck.py`'s `diff_snapshots` for the delta; report the plateau pass.

## Scoring & credit model

- **Raw:** `--scorers rrf formula` on every snapshot. `rrf` = content/index effect; `formula−rrf`
  = the graph/edge (PPR) effect. Paired bootstrap vs `frozen`, same as `analyze_h3.py`.
- **Judged bracket (REQUIRED for any summary-bearing snapshot — `c1_rem` onward):** H7 proved the
  raw metric **sign-flips** on summaries (they out-rank evidence turns and aren't evidence turns,
  so evidence-id scoring counts a coverage win as a miss). Re-credit with `impute_rescore.py` +
  `judge_summary_coverage.py` (the `raw ≤ judged ≤ provenance` bracket). NREM-only snapshots
  (`c1_nrem`) need no judge — edges don't add retrievable items.

## What each contrast buys

| Contrast | Answers | Sanity |
|---|---|---|
| `c1_nrem − frozen` | reproduces H3 | must match the isolated H3 numbers, else the combined harness is off |
| `c1_rem − c1_nrem` | **REM on top of edges** (new) | H7 was REM-on-frozen; this is the real question |
| `c1_consol − c1_rem` | consolidate's net (canonicalize vs archival cost) | watch for H3's deletion penalty |
| `c(k) − c(k−1)` | per-pass iteration gain | expect >0 at pass 2 (cap_hit), unknown after |
| `c_N − frozen` | **total steady-state sleep effect** | the headline |

## Caveats (pre-registered)

- **n=1, conv0; rrf/formula rerank-free.** The reranker interaction is H10's job, not this.
- **Instrument residual:** inactive rows perturb `rrf` ranks pre-active-filter (H7 noted
  `rem_none − frozen` r@10 −0.011, CI incl 0). Shared across arms; watch `frozen−frozen ≈ 0`.
- **Consolidate archives** (merge_into/rewrite supersede losers) — deletion by another name. Per
  H3, the iterated curve **could go down**; that's a real possible finding, not a failure.
- **No cheap shortcut across passes.** Each pass's LLM output depends on prior-pass state, so
  unlike H7 there's no generate-once/subset trick — every pass is a full LLM sleep.

## Exact command recipe (per pass)

Prereq: on branch `sleep-bench/h7-rem-floor` (see above). All on disposable bench forks under
`~/.somnigraph/benchmark/sleepbench-arms/` — never the live store (abstinence-safe). Multi-minute
LLM runs → your terminal.

```powershell
$ARMS   = "C:\Users\Alexis\.somnigraph\benchmark\sleepbench-arms"
$FROZEN = "$ARMS\conv0-frozen"
$env:PYTHONUTF8 = "1"
$env:OPENAI_API_KEY = (Get-Content "C:\Users\Alexis\.claude\data\openai_api_key" -Raw).Trim()
$env:SOMNIGRAPH_SUMMARY_MIN_MEMBERS = "5"   # shipped floor; REM reads it at import

# ---- Pass 1: fork frozen, then run + SNAPSHOT after each stage ----
$C1 = "$ARMS\conv0-c1"
Remove-Item $C1 -Recurse -Force -ErrorAction SilentlyContinue
uv run experiments/sleep-bench/fork.py fork $FROZEN $C1
$env:SOMNIGRAPH_DATA_DIR = $C1

uv run scripts/sleep_nrem.py deep                                    # GATE: "Embedded <N> edge linking contexts"
uv run experiments/sleep-bench/fork.py fork $C1 "$ARMS\conv0-c1_nrem"   # snapshot
uv run scripts/sleep_rem.py
uv run experiments/sleep-bench/fork.py fork $C1 "$ARMS\conv0-c1_rem"    # snapshot
uv run scripts/sleep_consolidate.py -v
uv run experiments/sleep-bench/fork.py fork $C1 "$ARMS\conv0-c1_consol" # snapshot (= end cycle 1)

# ---- Pass 2 (repeat on the SAME evolving store; snapshot at cycle boundary) ----
uv run scripts/sleep_nrem.py deep;  uv run scripts/sleep_rem.py;  uv run scripts/sleep_consolidate.py -v
uv run experiments/sleep-bench/fork.py fork $C1 "$ARMS\conv0-c2"
# ...repeat until diff_snapshots ~ 0 or N=4...

Remove-Item Env:SOMNIGRAPH_DATA_DIR

# ---- Score every snapshot (LLM-free; safe to run in this session) ----
foreach ($a in "frozen","c1_nrem","c1_rem","c1_consol","c2") {
  uv run experiments/sleep-bench/retrieve.py --data-dir "$ARMS\conv0-$a" --label $a --conv 0 --scorers rrf formula --fresh
}

# ---- Judged rescore for summary-bearing arms (c1_rem onward) ----
#   uv run experiments/sleep-bench/judge_summary_coverage.py ...   (per H7 recipe)
#   uv run experiments/sleep-bench/impute_rescore.py ...           (raw<=judged<=provenance bracket)

# ---- Analyze ----
#   generalize analyze_h3.py -> analyze_iterated.py: same paired-bootstrap machinery,
#   N labels vs the frozen baseline + adjacent-pass deltas. (small new script)
```

## What I can build vs what needs your terminal

- **I build (no LLM):** `analyze_iterated.py` (generalize `analyze_h3.py` to N labels + per-pass
  deltas), a small pass-loop wrapper with `diff_snapshots` plateau detection, and the judged-
  rescore glue. Plus I can run all the **scoring/analysis** (rrf/formula are LLM-free).
- **Your terminal (LLM, sequential):** the per-pass `nrem deep → rem → consolidate` runs.

## Provenance

Surfaced 2026-07-04 when Alexis asked "did we test both legs / multiple sleeps?" — no on both.
Fills the hole H3 (`findings-h3-nrem-graph.md`) + H7 (`findings-h7-rem-floor.md`) left; Phase 0
`cap_hit=50` (`findings-phase0.md`) is the direct evidence one pass undershoots. Proposed as **H11**
in the §6 registry of `docs/proposals/sleep-pipeline-optimization.md`.
