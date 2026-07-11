# Plan — pre-fork adversarial review of the production sleep path

**Written:** 2026-07-11. **Runs:** any session before the ~07-15 abstinence fork.
**Scope:** `scripts/sleep.py` (orchestrator), `sleep_nrem.py`, `sleep_rem.py`,
`sleep_consolidate.py`, `experiments/sleep-bench/fork.py` — ~250 KB of pipeline code.

## Why this review exists

Since the last live sleep (2026-07-01, start of the abstinence window), the sleep path
absorbed four substantive changes that have **only ever run on LoCoMo benchmark stores**:

1. NREM duplicate default flipped merge → link, all layers (`--legacy-merge` opt-out)
2. H12 consolidate rewrite: core-facts node synthesis, `consolidated_from` edges,
   members downweighted not archived (`--legacy-archive` opt-out)
3. NREM parallel classification (merged `673b892`) + consolidation-node priority band
4. REM summary floor: absolute token floor → `SUMMARY_MIN_MEMBERS` (default 5)

The 07-15 fork is a one-shot experiment (the accumulation regime took two weeks to
create). A pipeline bug in the slept arm doesn't just cost a rerun — it costs the
experimental condition. The review's job is to make the slept arm run the *intended*
configuration, with failure modes enumerated before, not after.

## Method

Adversarial read per stage, checklist-driven, one review agent per stage is fine
(code reading, not compute). Verification smokes run on a **synthetic scratch DB only**
— **not** a fork of the live store. A live-store dry run would preview the experiment's
outcome before the registered run and invite mid-flight tuning; the blindness is worth
more than the extra confidence. (Decision point D1 below if we disagree.)

Output: findings ranked by severity → fixes on a branch → a frozen **fork runbook**
(exact commands, env vars, flags) appended to this directory before 07-15.

## Per-stage checklists

### A. Orchestrator (`scripts/sleep.py`)

- Stage ordering and what happens on mid-run failure: is a partially-slept store
  detectable and resumable, or does a crash leave arm B in an unlabeled half-state?
- Env propagation to child stages (`SOMNIGRAPH_DATA_DIR`, embedding backend unset on
  this machine = OpenAI 1536d, `PYTHONUTF8=1` — box-drawing chars crash cp1252).
- `--probes` semantics: registered protocol scores with V6, decide probes on/off (D2).
- Event-log writes during sleep: which events land in the fork's DB, and do any of
  them feed the eval scorer (V6 features read feedback/hebbian/session data — the
  07-02 fork needed `strip_probe.py` for exactly this class of contamination).

### B. NREM (`sleep_nrem.py`)

- Link-default flip on **real** data: live duplicates differ from LoCoMo paraphrases
  (edits, partial supersedes, cross-category near-dups). Read the classification
  prompt + link writes with live examples in mind.
- Parallel classification safety: API failure mid-batch, partial DB writes, ordering
  assumptions, `MAX_PARALLEL` (A1 guard fixed `14fd6de` — verify it's the shipped
  form). Memory pressure: pagefile 12/32 GB is registry-set but **inactive until
  reboot** — reboot before the fork run.
- `cap_hit` behavior at accumulation volume: Phase 0 showed one pass undershoots
  steady state at cap 50. Two weeks of live accumulation — will one pass converge,
  and if not, is multi-pass safe (interacts with D3)?
- Summary-duplicate handling: link + defer to consolidate rewrite was left as a TODO
  at the H3 flip — confirm the deferred path can't orphan or double-link.

### C. REM (`sleep_rem.py`)

- **Dormancy is the known eroder** (whole-corpus: −0.256 R@10 alone, 97.5% dormanted
  on a fresh store). Live check said only 10.4% of active memories are even eligible,
  so the blast radius is bounded — but verify the *live* eligibility predicate matches
  the one that produced 10.4% (recall+reflect < 3, access recency), and that dormancy
  respects priority/pinned classes. Decision D3: run dormancy as-shipped or disable.
- `SUMMARY_MIN_MEMBERS=5` on live cluster-size distribution: LoCoMo clusters are
  session-shaped; live clusters may be smaller. If most live clusters are < 5, REM
  produces near-zero summaries and the fork can't measure the one mechanism proven to
  help (+0.078✱ whole-corpus). Measure the live cluster-size histogram *read-only*
  before deciding the floor is right.
- Summary embedding: OpenAI 1536d on this machine — cost estimate, dim-guard in
  `db.py` confirmed active, API-key sourcing under `SOMNIGRAPH_DATA_DIR` override.
- Personal-vault reads (core.md, seed.md, journal): confirm read-only, present, and
  that vault content can't leak into fork-arm summaries in a way that breaks eval.

### D. Consolidate (`sleep_consolidate.py`)

- H12 node synthesis on live clusters: first real-LLM run was clean (0 archived) on
  LoCoMo — live categories/priorities differ. Read the synthesis prompt + writes for
  live-schema assumptions.
- **Known bug, held for this review:** pass 2 re-consolidates already-consolidated
  clusters → duplicate nodes (`gather_clusters` doesn't skip majority-consolidated
  clusters). A single-pass fork run may not trigger it, but post-fork live cadence
  will. Fix it here (small, well-understood) rather than ship a known dup generator.
- Schema-error rejection rate: Phase 0 saw 22/27 actions rejected; verify the H12
  action schema on a scratch smoke and that rejections are logged loudly.
- `--legacy-archive` must be OFF (downweight path) — confirm default, and that the
  downweight magnitude is the shipped value, not a bench override.

### E. Fork mechanics (`fork.py`) + scoring rig

- `--scrub-access-counts` on the **frozen** arm (it carries 9,440 contaminated
  recall_counts; the slept arm gets the same scrub so arms differ only by sleep).
  Confirm scrub covers every count V6 reads.
- Arm layout per the registered runbook in `findings-sleep-fork.md` (A frozen,
  B slept, snapshot-swap eval, same GT throughout), updated for V6 scoring.
- **New cell to consider (D4): post-sleep reranker retrain.** H10's headline is that
  sleep's value *requires* a retrained reranker, and N4 says the retrain needs no
  schema extension. A fork that only scores B with the frozen V6 model re-runs the
  naive-deploy cell (judged-null on bench). Decide whether the protocol grows a
  B+retrain arm and what its GT/recipe is.

## Cross-cutting

- **Config parity table:** enumerate every env var / flag / constant the 07-15 run
  depends on, with the value the bench validated vs the live default. This table *is*
  the runbook's core. (The A1 bug was exactly a bench-env assumption meeting a live
  env — empty string vs unset.)
- Reversibility: arms are copies; live store untouched; nothing in the review or the
  fork writes to `~/.claude/data` — verify every command in the runbook honors
  `SOMNIGRAPH_DATA_DIR`.
- **No sleep on the live store** remains in force throughout the review. Smokes on
  synthetic scratch DBs only.

## Decision points to settle during review

| # | Question | Current lean |
|---|---|---|
| D1 | Smoke on synthetic scratch only, or allow a mechanics-only live-fork dry run? | Synthetic only — preserve blindness |
| D2 | Probes on or off in the fork arms? | Off (`--probes 0`); V6 scoring makes the probe arm's question moot and strip_probe cost a day last time |
| D3 | Dormancy as-shipped or disabled in the slept arm? | As-shipped — the experiment measures the production pipeline, blast radius bounded (10.4%), and H10 says ranking-layer recovery exists; disabling it changes what's being measured |
| D4 | Add a B+retrain arm (post-sleep reranker retrain per H10/N4)? | Yes — it's the program's headline finding and the fork is the only live test of it; needs a GT/recipe decision |
| D5 | Fix the H12 pass-2 dup bug before the fork? | Yes — small, held for this review, post-fork cadence hits it regardless |

## Exit criteria

- Every checklist item has a verdict (pass / fixed / accepted-risk with reasoning).
- Fixes merged; smokes green on scratch.
- Fork runbook frozen in this directory: exact commands, env table, arm layout,
  eval steps, D1–D5 resolved and recorded.
- Reboot done (pagefile active) before the fork run itself.
