# Pre-fork sleep-path review — findings (2026-07-11)

Executed per [`plan-prefork-review.md`](plan-prefork-review.md): five per-stage
subagents launched in parallel (orchestrator, NREM, REM, consolidate, fork/scoring),
each briefed with file paths + its checklist section only — never a summary of the
code. The main session built the cross-cutting config-parity table and adversarially
synthesized. Findings were approved by Alexis one decision at a time before any fix
landed; all fixes live on `fix/prefork-review` with this document.

Two findings converged from independent agents working blind to each other
(B-1 from live measurement + feature tracing; the `--probes` hazard from
orchestrator + fork agents) — unseeded convergence, high confidence.

## Decisions (all resolved 2026-07-11, pre-registered before the fork)

| # | Question | Resolution |
|---|---|---|
| D1 | Smoke scope | Synthetic scratch only (unchanged). LLM smokes to Alexis's terminal; commands in this doc. |
| D2 | Probes | **Off** — and explicitly: the default is 10, runbook hard-codes `--probes 0`. |
| D3 | Scrub / dormancy | **No scrub, either arm.** The 10.4% premise was invalid (see B-1); dormancy is expected inert (0 eligible) and that expectation is pre-registered + verified post-run. "If things are working as intended, we don't need to scrub" — the shielding IS the intended behavior; the scrub was a bench workaround for capture.py contamination the live store doesn't have. |
| D4 | Retrain arm | **Full symmetric 2×2** — A gets its own retrain (N4: ~95% of the R_post dividend is sleep-independent recalibration; a B-only retrain confounds). Design in the runbook. |
| D5 | H12 pass-2 dup bug | **Fixed on this branch** (see fix 3). |
| N1 | Arm-B stage list | `sleep.py --deep --probes 0` then **one explicit `sleep_consolidate.py` pass**, with a **B-mid checkpoint between them** so consolidate — never run in production — is measured as its own signal (B-full − B-mid). Integration of consolidate into `sleep.py` deferred until this measurement exists. |
| N2 | NREM mode | `--deep` (production habit + 07-02 precedent; skip-if-stable bounds cost). |
| N3 | Stale NREM prompt | **Fixed on this branch.** The stale prompt+link combination never ran on the live pipeline (flip landed 07-04, inside the abstinence window; the last live sleep ran the prompt truthfully with merge default) — there was no live-parity baseline to preserve. Pre-registered caveat: NREM duplicate rates are not comparable to bench runs. |

## BLOCKER findings

### B-1. Scrub × dormancy is bimodal; D3's 10.4% premise was invalid — RESOLVED by D3 (no scrub)

The detail-dormancy predicate (`sleep_rem.py` `process_dormancy`) is a three-way
conjunction: covered by an active summary's `derived_from` edge AND
recall+reflect < 3 AND **`created_at`** age ≥ 60d (not access recency, as the plan
assumed). Measured read-only on the live store 2026-07-11:

- **Unscrubbed: 0 of 1,230 active detail memories eligible.** Every >60d memory has
  ≥3 recalls. Dormancy is inert.
- **Scrubbed: 934 (75.9%) eligible**, rising to **~962 (78.2%)** once the run's own
  summaries extend coverage mid-run — the whole-corpus catastrophe regime (−0.256
  R@10 class) as a protocol artifact.
- The registered "10.4%" was the low-access **marginal** (11.1% at measurement),
  never the conjunction.

Independently, the fork agent enumerated all 31 V6 features: **none reads any
scrubbed column** (`recall_count`, `reflect_count`, `access_count`, `use_count`,
`startup_count`, `last_accessed`). Feedback/hebbian/session features read
`memory_events`, which the scrub doesn't touch and which stays arm-symmetric with
probes off. Formula and RRF fallbacks are also counter-blind. The scrub buys zero
scoring parity; its only consumer is dormancy.

### B-2. The D4 retrain cells could not run — FIXED (fix 1 + 2)

`eval_retrieval.py` hard-overrode MODEL_PATH to the deployed search order
(`~/.somnigraph` then `~/.claude/data`); an arm-local retrained model was never
loaded, no `--model-path` existed, and a load failure warned-and-continued into
silent formula scoring (the 2026-04→07 outage shape — the loud assert existed only
in the bench rig `scorers.py`). `analyze_sleep_fork.py` additionally hardcoded a
false "FORMULA on BOTH sides" scorer note and defaulted `--config formula`
(a reranker record set joined to zero queries).

## HIGH findings

- **H-1. `sleep.py --probes` defaults to 10** — probes write `retrieved`/`feedback`
  events, bump counters, and create/strengthen hebbian edges: the one sleep-side
  channel V6 reads, and edge/counter mutations that strip_probe-class event deletion
  **cannot undo**. Runbook hard-codes `--probes 0` (default intentionally unchanged —
  probes are production feedback generation). Found independently by two agents.
- **H-2. No DATA_DIR guard in `sleep.py`** — FIXED (fix 6): startup banner prints the
  resolved data dir + DB path; refuses the live store without `--allow-live`. The
  production wrapper (`~/.claude/scripts/sleep.py`) now passes `--allow-live`
  itself — it *is* the intentional live entry point; direct repo invocation stays
  experiment-safe. (The wrapper also silently defaults `SOMNIGRAPH_DATA_DIR` to the
  live store, which is why the guard matters: fork-week muscle memory.)
- **H-3. Three silent part-slept modes.** (a) No run sentinel: NREM/REM write zero
  events; a mid-run crash leaves an unlabeled half-state (runbook: per-stage
  verification gates). (b) NREM batch failures returned `[]`, stamped anyway,
  stderr-only, exit 0 — FIXED observationally (fix 7): failures route through
  `log()`, `failed_batches: N/M` in log + report + sleep_log stats; the deeper
  stamp-only-on-success redesign is deferred (changes retry semantics). Capped or
  failed pairs remain unrecoverable by re-pass — pre-registered as single-draw edge
  yield. (c) REM needs three sidecar files in the fork dir or it silently
  cold-starts / half-sleeps: `rem_thin_migration_done` (absence purges all 32
  summaries and regenerates), `openai_api_key` (absence + no env crashes mid-REM
  after Opus mutations began), `taxonomy.json` (absence forks the taxonomy).
  Runbook: whole-dir BASE copy.
- **H-4. NREM edge-embedding backfill is non-fatal** — a credential failure leaves
  context edges with NULL `linking_embedding`, which PPR skips; production has no
  health guard (bench-only, `retrieve.py`). The key fallback file moves with
  `SOMNIGRAPH_DATA_DIR`. Runbook: post-NREM NULL-embedding gate. Backfill is
  self-healing on re-run (unscoped query).
- **H-5. Pagefile inactive until reboot — verified live** (registry 12/32 GB;
  allocated 5,613 MB at review time). The 0xC0000409 commit-crash regime that
  motivated single-worker NREM is still in force. **Reboot before the fork.**
- **H-6. Detail dormancy respects no protected class** (no pinned/keep, priority,
  `decay_rate=0`, or `human_edited` check; the summary path checks `human_edited`,
  so the omission is an oversight). Post-scrub pool would have contained 473 p≥8,
  all 64 never-decay memories (incl. the 8 tracked "Q:" questions), 396 procedural.
  Moot under D3 (dormancy inert); **must be fixed before any future config in which
  dormancy fires**. Deferred to the dormancy redesign (downweight-not-status-death).
- **H-7. Live→BASE copy has no tooling** — `fork.py` correctly refuses the live dir
  as source, so the most dangerous copy is manual, off a hot WAL DB. Runbook:
  server stopped, `sqlite3 .backup`, `PRAGMA integrity_check`, whole-dir sidecars.
  Related: `wal_checkpoint(TRUNCATE)` before every snapshot-swap copy.

## MEDIUM findings

- **M-1. `sleep.py` never invokes consolidate** — resolved by N1 (explicit step +
  B-mid checkpoint). Integration deferred until consolidate has its first measured
  live signal.
- **M-2. H12 pass-2 duplicate nodes** — FIXED (fix 3). `gather_clusters` now skips
  majority-already-consolidated clusters. **Keyed on `last_consolidated` (stamped on
  every member unconditionally), NOT the `consolidated_member` flag** — the flag
  only lands on downweighted members, and protected members (common on live) are
  never downweighted, so the review-proposed flag-majority test would under-fire
  exactly on live data. A cluster re-enters consolidation once genuinely-new members
  (never stamped, or created after their stamp) reach a majority. Residual: a
  consolidated cluster gaining a few new members waits for new-member majority —
  acceptable v1; execute-side Jaccard check is the escalation if it bites.
- **M-3. A1 empty-string env bug class fixed in NREM only** — FIXED (fix 4):
  `SOMNIGRAPH_SUMMARY_MIN_MEMBERS` (`sleep_rem.py`) and
  `SOMNIGRAPH_CONSOLIDATED_MEMBER_WEIGHT` (`sleep_consolidate.py`, crashed at
  import) now use the `14fd6de` `.strip() or default` pattern.
- **M-4. Consolidate's silent rejection classes** — FIXED (fix 8): unrecognized
  action strings (no `else` branch) and unknown-memory-id rejections now log
  `REJECTED` detail lines, so the rejection-rate instrument sees schema drift
  instead of a flatteringly clean first live run.
- **M-5. Scoring-rig hygiene** — eval wall-clock is an arm asymmetry (hebbian
  30-day lookback + time-decay anchor to now(): all cells same day, fixed order);
  `--snapshot` copies the live DB if the snapshot is absent (never pass it);
  registered non-defaults must be explicit (`--recall-budget 5000`, `--gt-path`);
  `train_reranker._export_for_mcp` overwrites `<arm>/tuning_studies/` (score V6
  cells before retraining). All runbook-encoded.
- **M-6. NREM misc** — stale prompt FIXED (N3). `float(confidence)` crash on
  non-numeric LLM junk: accepted (pre-stamp, re-run clean, cost = wall-clock).
  Duplicate links keep stale edit-pairs active at full PPR weight: pre-registered
  expectation, feeds the consolidate signal.
- **M-7. REM taxonomy restructuring is an unreviewed degree of freedom** — 104
  unassigned live memories trigger 3 sonnet batches that may also merge/split/
  reparent topics store-wide; nondeterministic cluster shaping. Pre-registered;
  arm-B reruns won't reproduce exactly. Phase 4 repair also runs unconditionally
  even with `--probes 0` (≤10 organic misses, Sonnet, theme edits) — included in
  the arm as production behavior; its events are not scorer-visible.

## Notable PASS results

- `SUMMARY_MIN_MEMBERS=5` fits live data: 33 topic clusters (sizes 4–98), **31/33
  clear the floor**; 16 produce summary work (1 new + 15 stale refresh). The fork
  will exercise the one proven-positive mechanism (+0.078✱ whole-corpus).
- H12 write path verified live-safe: every REM-interaction escape claim in the code
  checks out (consolidation nodes can't be dormanted / migration-purged / topic-
  collided); `--legacy-archive` defaults off; downweight 0.8 is the shipped value,
  set nowhere else in the repo. Live stores are protected-heavy → expect clusters
  with a node created but few/zero members downweighted (correct, not a bug).
- NREM parallel classification: DB writes main-thread-only after all futures;
  multi-pass idempotent; A1 guard byte-identical to the shipped `14fd6de` form.
- Personal-vault reads: only `core.md`, read-only, audit-prompt-only; audit won't
  trigger (32 summaries < 100). No leak path into fork summaries.
- `fork.py` isolation guard solid on both endpoints; scrub implementation correct
  for its bench purpose (which the fork doesn't share).
- V6 scrub-coverage question answered vacuously: V6 reads none of the scrubbed
  columns (full 31-feature trace in the fork agent's report).

## Fixes on this branch

| # | File(s) | What |
|---|---|---|
| 1 | `scripts/locomo_bench/eval_retrieval.py` | `--model-path` (must exist; repoints features .json beside it); fail-loud scorer-identity assert after config apply; `model_path` + `model_sha256` stamped into every production record |
| 2 | `experiments/sleep-fork/analyze_sleep_fork.py` | Scorer note derived from the records' identity stamps (per-side model path + SHA); loud empty-join error naming the config mismatch; mixed-SHA-within-side error |
| 3 | `scripts/sleep_consolidate.py` | Pass-2 dup skip in `gather_clusters` (majority-`last_consolidated` test; see M-2); `_parse_flags` helper |
| 4 | `scripts/sleep_rem.py`, `scripts/sleep_consolidate.py` | Empty-string env guards (A1 pattern) |
| 5 | `scripts/sleep_nrem.py` | Classification prompt downstream-effects text: duplicate links non-destructively (was: deletes) |
| 6 | `scripts/sleep.py` + `~/.claude/scripts/sleep.py` (wrapper) | Data-dir banner; live-store refusal without `--allow-live`; wrapper asserts the flag as the intentional live entry point |
| 7 | `scripts/sleep_nrem.py` | Batch-failure accounting: failures through `log()`, per-batch zero-result warning, `failed_batches: N/M` in log + report + sleep_log stats |
| 8 | `scripts/sleep_consolidate.py` | `else` branch for unrecognized actions; detail line for unknown-memory-id rejections |

Smokes run (synthetic scratch only): all files compile; empty-string guards parse to
defaults; live-store refusal fires before any work; `--model-path` in help;
analyzer happy-path shows per-side SHAs and wrong-config exits loudly. Incidentally,
the analyzer smoke **reproduced the cp1252 crash** the moment it ran without
`PYTHONUTF8=1` — the runbook line is load-bearing.

## Deferred (post-fork)

- Dormancy redesign: downweight-not-status-death + protected-class guard (H-6).
- Consolidate integration into `sleep.py` (pending the B-full − B-mid signal).
- NREM stamp-only-on-success (retry semantics change).
- NREM summary-duplicate consolidate-rewrite TODO; `--legacy-merge` dangling-edge
  cleanup (both unreachable in the fork protocol).
- Sentinel events / resumable sleep (runbook gates substitute for now).
- Latent REM `compute_health` TypeError on NULL `last_accessed` (live store has 0).

The frozen fork protocol, config-parity table, and exact commands live in
[`runbook-0715.md`](runbook-0715.md).
