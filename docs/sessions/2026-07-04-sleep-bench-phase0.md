# Session 2026-07-04 — Sleep-Bench Phase 0 (+ rig, + 3 production fixes)

Also the compact handoff: everything a resuming instance needs.

## What shipped (all merged to `main`, up to `1adcf1c`, NOT pushed)

- **New experiment program: sleep-bench.** Measure/optimize each sleep stage on a benchmark
  corpus (LoCoMo) as the reproducible complement to the ~2026-07-15 abstinence fork.
  Proposal: [`docs/proposals/sleep-pipeline-optimization.md`](../proposals/sleep-pipeline-optimization.md)
  (6 phases, Fable-designed).
- **Phase 0 (born-saturation precheck):** `experiments/sleep-bench/precheck.py` + `fork.py`.
  Ran on LoCoMo conv 0 → **gate PASS**. Full results + caveats:
  [`experiments/sleep-bench/findings-phase0.md`](../../experiments/sleep-bench/findings-phase0.md).
- **Stage A retrieval rig:** `experiments/sleep-bench/{retrieve,capture,scorers,gt,metrics,store,ingest_baseline}.py`.
  Wraps the real `memory.tools.impl_recall`; V6 + formula scoring with a loud-load guard;
  LongMemEval metrics (recall_any/all, effective_k) alongside NDCG@5k/R@10/MRR; resumable
  per-query JSONL; independent of the old `eval_retrieval.py`. **Smoke-passed** on a fresh
  frozen conv0 store.
- **3 production bug fixes** (commits, all on main): `d798810` reranker override `.pkl`→`.txt`
  (a stale March pkl was shadowing V6 — the benchmark had been silently formula-only, would
  have hit the abstinence fork too); `42546d8` consolidate PEP-723 deps + `KeyError 'annotated'`;
  `b48feaa` `run.py` dia_map matches all statuses (was `active`-only → phantom post-sleep misses).

## Commits (main)

```
1adcf1c exp(sleep-bench): Phase 0 findings
3f9465c exp(sleep-bench): precheck harness + Stage A rig
906da83 docs: sleep pipeline optimization proposal
b48feaa fix(bench): dia_map matches all statuses          (production fix)
42546d8 fix(sleep): consolidate deps + annotated          (production fix)
d798810 fix(bench): reranker override .txt                (production fix)
```

Branches `exp/sleep-bench-phase0` and `fix/benchmark-reranker-path` are fully merged (deletable).

## Surprises

- **REM is 100% inert on LoCoMo** — `SUMMARY_MIN_DETAIL_TOKENS=2000` (sleep_rem.py:671) blocks
  every summary (turns too short). The highest-EV stage is gated off out of the box; the floor
  sweep is promoted from a Phase-3 optimization to a Phase-1 prerequisite.
- **Consolidate rejects 22/27 proposed actions** on schema errors (no merge_content/new_content/
  arc_context) — ~18% throughput, and this **blocks convergence** (re-proposed-and-rejected
  clusters never fix-point).
- **NREM `cap_hit=50`** — one pass provably undershoots steady state; multi-pass convergence is
  a real, measurable phenomenon here (keep `MAX_EDGES_PER_CYCLE` as the engine, don't remove it).

## Caveats / honest accounting

- Phase 0 numbers came from a **crashed-then-manually-completed** run (the `KeyError`, since
  fixed); a clean end-to-end re-run is pending and would write `precheck_results*.json`.
- **Rig env leak (hardening TODO):** `retrieve.py` reads ambient `SOMNIGRAPH_DATA_DIR` for the
  formula MODEL_PATH message; the *store* is read correctly from `--data-dir` (smoke confirmed
  419 active = the fresh store). Fix: pin `SOMNIGRAPH_DATA_DIR=--data-dir` inside `retrieve.py`.
- Schema-rejection being **benchmark-specific is unverified** (short near-dup turns may inflate it).
- `capture.py` audited SAFE (neither scorer reads the mutated `last_accessed`/`recall_count`/
  `access_count`); static analysis, confirmed live by the smoke.

## Stores on disk (isolated, non-live — never confuse with `~/.claude/data`)

- `~/.somnigraph/benchmark/sleepbench-precheck/conv0` — **SLEPT** (NREM+REM done; 388 active;
  edges + topics; **no** dia_map.json). The Phase 0 store.
- `~/.somnigraph/benchmark/sleepbench-arms/conv0-frozen` — **FRESH** (419 active; **has**
  dia_map.json). Rig baseline; **reuse as Step 1 arm A0**.

## What's next (dependency-ordered)

1. **Step 1 — NREM/graph H3** (ready): fork `conv0-frozen` → `sleep_nrem.py deep` on the copy →
   score both arms on **full** GT (drop `--max-queries`) → compare V6 delta vs formula delta
   (H3: edges help V6, not formula). *Terminal: one NREM run (~15 min, Sonnet).*
2. **Step 2 — schema-rejection diagnosis** (code-only, unblocks convergence): read the consolidate
   judge prompt in `scripts/sleep_consolidate.py`; disambiguate prompt-underspec vs
   model-can't-substantiate; fix (tighten schema and/or validate+retry).
3. **Rig-hardening:** pin `SOMNIGRAPH_DATA_DIR` in `retrieve.py`.
4. **Later:** `converge.py` (needs Step 2 to converge; isolated fork, live-consolidate,
   `--consolidate-only` vs `--full-cycle`, snapshot-per-pass, floor-as-percentile, cap kept +
   optional uncapped ceiling arm); REM floor sweep {2000,800,400,200}/percentile; coverage-scoring
   (`coverage_judge.py`, needed once summaries exist); Phase-1/2 headline decomposition.

## Decisions reserved for Alexis

- The consolidate judge-prompt change (it runs on the **live** abstinence fork too — review
  before 07-15). REM floor arm count / changing the shipped 2000 default. Convergence pass budget.

## Constraints (carry forward)

- **No sleep/consolidate on the live store `~/.claude/data` until ~07-15** (abstinence window).
- Sleep/probe/multi-minute runs → **Alexis's terminal** (provide the command, stop).
- Home machine is **OpenAI 1536d** — leave `SOMNIGRAPH_EMBEDDING_BACKEND` unset; `PYTHONUTF8=1`
  for sleep scripts (box-drawing chars crash cp1252).
- The working tree carries an **unrelated 2026-07-03 thread** uncommitted (`docs/roadmap.md`,
  `research/sources/index.md`, `docs/proposals/source-template-improvements.md`,
  `research/sources/arhuman-mnemos.md`) — awaiting Alexis's separate review; **never stage these.**
  The `docs/roadmap.md` link to the sleep-bench proposal is deferred until this thread clears.

## Reversibility

Nothing pushed — `main` can reset to `origin/main` (currently 6 commits behind local). All work
on isolated benchmark stores; the live store was never touched.
