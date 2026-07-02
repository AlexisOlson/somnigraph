# Autonomous Experiments: A Ranked Catalog

**Status: planning artifact, nothing launched.** This is a scoping document, not an execution log. It catalogs where unsupervised overnight grinding — iterative runs against an objective metric, with a closed run→measure→adjust→rerun loop and no decision that needs a human mid-flight — can make real research progress on Somnigraph, and it splits those cleanly from the work that must stay interactive.

Read alongside [`roadmap.md`](../roadmap.md) (the numbered open questions and proposed experiments this draws from), [`benchmarks.md`](../benchmarks.md) (§ Proposed benchmarking experiments), [`ideas-considered.md`](../ideas-considered.md) (Tier 1 adopt candidates), and `STEWARDSHIP.md` (the priority list and the write-path research program). Ready-to-launch artifacts for the top candidates live under [`handoff/`](handoff/).

Alexis chooses what runs. This document does not launch anything.

---

## Alignment with the stewardship arc

A five-session stewardship arc is already in motion (planned 2026-07-01; the "close the ledger" session is the reranker-arc branch this catalog was written against). The remaining arc steps map directly onto the grindable/interactive split below:

| Arc step | Model | Nature | This catalog |
|----------|-------|--------|--------------|
| 1. Close the ledger | Opus | claims | done (reranker-arc) |
| 2. Ship the instruments (shadow near-dup log, secret redaction, `memory_meta` fix) | Sonnet | **mutates live `remember()`** | **Interactive** — §Interactive-only #I1 |
| 3. Proactive-injection offline floor study | Opus | offline replay, publishable negative | **Grindable** — #2 (artifact ready) |
| 4. Sleep counterfactual fork | Opus | offline on a DB copy | **Grindable** — #1 (artifact ready) |
| 5. Write-path experiments (prospective indexing + Write Guard gate) | Opus + Sonnet | indexing offline / gate thresholds need measured distro | **Split** — indexing #4 grindable; gate thresholds §I2 interactive |

The arc's own logic already anticipates this catalog's core distinction: the *measurement* runs unsupervised, the *threshold choice and the prose claim* come back for review.

---

## Safety rails (apply to every grindable candidate)

These are non-negotiable and are baked into every handoff artifact. Any experiment that cannot honor all of them is not grindable and belongs in the interactive list.

1. **Never touch the live store.** The production DB and GT live at `~/.claude/data/` (NOT `~/.somnigraph/`, which is a near-empty 143 KB default). Every experiment runs against a **copy**:
   ```sh
   SCRATCH="$HOME/.somnigraph-exp/<exp-name>-<runid>"
   cp -r "$HOME/.claude/data" "$SCRATCH"          # copies memory.db + tuning_studies/ (GT + reranker .txt)
   export SOMNIGRAPH_DATA_DIR="$SCRATCH"
   export SOMNIGRAPH_EMBEDDING_BACKEND=fastembed  # production is 384d fastembed; db.py hard-fails on 384d/1536d mismatch
   ```
   The embedding-backend export is mandatory: scripts default to OpenAI (1536d), the store is fastembed (384d), and `db.py` hard-fails at connect time on mismatch. Getting this wrong is a loud failure, not a silent corruption — but set it anyway.
2. **Explicit cost ceiling per run, stated up front.** fastembed is local (free). OpenAI embeddings for LoCoMo are cached (294 MB warm cache exists). `sleep.py` and `probe_recall.py` call the `claude` CLI (subscription tokens, not metered API $). `run.py` reader/judge use OpenAI `gpt-4.1` (metered $). Each artifact names its ceiling.
3. **Isolated git worktree, feature branch, commit-only.** Never merge, never push. New files only. (This catalog and its artifacts were themselves produced this way, on `plan/autonomous-experiments`.)
4. **Machine-readable artifacts + a findings markdown, with exact repro commands.** JSON metrics + a `findings-<exp>.md` per experiment. Negative results written up with the same care as positive ones — they are equally publishable here (honest accounting is STEWARDSHIP invariant #1).
5. **Stopping criteria defined before launch.** A fixed sweep grid, an N-trials-without-improvement cap, a cost ceiling, or a wall-clock cap — never an open-ended loop.
6. **There is always residual human review.** Named per candidate. At minimum: the prose claim that lands in the docs must be reviewed before merge (honest accounting), and any threshold read off a measured distribution is a human call (shadow-mode philosophy).

---

## Ranking

Ranked by **information value per unsupervised hour** — the return on an overnight slot where no human is watching. Absolute value, readiness (does the infra exist?), safety, and dollar cost all feed the rank. "Ready" = every script/flag/data file verified present (see §Feasibility appendix); "Prep" = a small named piece must be built first.

| # | Candidate | Info value | Unsup. hours | $ cost | Ready? | Shape |
|---|-----------|-----------|--------------|--------|--------|-------|
| **1** | **Sleep counterfactual fork** (roadmap #14+#3) | **Very high** — the central unanswered question | 4–10h (sleep-bound) | ~0 (claude CLI) | **Ready** | Long grind → session |
| **2** | **Proactive floor study** (proposal; arc step 3) | **High** — publishable either sign | 2–4h | 0 | Ready (write replay script) | Long grind → session |
| **3** | **Objective one-offs bundle** (roadmap #22, #23, edge-age, bimodal, PPR-perturb, latency) | Medium each, **very high per hour** | 2–4h total | 0 | Mostly ready (small scripts) | Sequence → session |
| **4** | **Prospective indexing** (roadmap #10; arc step 5a) | **High** — Kumiho/HyperMem precedent | 3–6h | <$1 (Haiku or claude CLI) | Ready | Grind → session |
| **5** | **LoCoMo expansion ablation** (roadmap #21) | Medium — closes STEWARDSHIP P5 remainder | 2–4h | ~0 (cache warm) | **Ready** (flags verified) | Sweep → session/workflow |
| **6** | **FSI seed-stability audit** (roadmap #9) | Medium — first-mover-bias check | 1–3h | 0 | LoCoMo ready; prod needs `--seed` (Prep) | Fan-out → **workflow** |
| **7** | **LoCoMo sleep + feedback-loop ablations** (roadmap #21 siblings) | Medium | 3–6h | $5–20 (reader/judge) | Partial (`--feedback-loop` ready; sleep needs assembly) | Sweep → session |
| **8** | **Graph null models** (roadmap #17) | Medium — is graph real signal? | 4–8h | 0 | Prep (variant builders) | Sweep → session |
| **9** | **Query difficulty clustering / paraphrase robustness** (roadmap #11, #15) | Medium | 4–8h | <$2 (paraphrase LLM) | Prep | Judged sweep → session |
| **10** | **LongMemEval-S retrieval harness** (ideas Tier-1 #16) | High (fills zero external LME numbers) but **build-first** | 6–12h | 0 (zero-LLM metrics) | **Prep (build adapter)** — higher unsupervised risk | Build+run → session |

---

## Grindable candidates (detailed)

### 1. Sleep counterfactual fork — *the central unanswered question*

**Objective metric.** Retrieval quality (NDCG@5k, R@10, MRR) on the fixed GT query set, measured on a slept copy vs. a frozen copy of the same store. Delta with a bootstrap confidence interval.

**Why it's the top pick.** "Does sleep improve retrieval?" is flagged in `architecture.md` § Open problems and `roadmap.md` as *the* central open question, and no system in the 184-source corpus has this measurement. The methodology is a clean A/B on identical inputs, so the causal claim is unusually strong for one overnight run.

**Loop design.**
1. Copy `~/.claude/data` → `$SCRATCH_A` and `$SCRATCH_B` (two independent copies).
2. Baseline: `eval_retrieval.py --dataset production --configs reranker` on `$SCRATCH_A` (frozen). Record NDCG/R@10/MRR per query.
3. `uv run scripts/sleep.py` with `SOMNIGRAPH_DATA_DIR=$SCRATCH_B` (full NREM+REM cycle mutates the copy only).
4. Re-eval `$SCRATCH_B` with the identical config and GT set. GT relevance labels are per-(query, memory) and stay valid; sleep-created memories that aren't in GT score 0 (this is the fair, conservative scoring — it does not reward sleep for inventing unlabeled hits).
5. Paired delta per query + bootstrap CI. Break down by NREM effect (edges → adjacency expansion) vs. REM effect (summaries competing for budget), matching the roadmap hypothesis.

**Stopping criteria.** Single deterministic pass — one sleep cycle, one before/after eval. No loop. Optional: repeat the fork 3× with different sleep RNG seeds to bound sleep's own nondeterminism (the LLM calls vary), capped at 3.

**Cost / wall-clock.** ~0 metered $ (sleep uses the `claude` CLI; eval uses local fastembed against the warm store). Wall-clock 4–10h, dominated by sleep's LLM calls over the full active set. **Ceiling:** wall-clock cap 12h; if sleep hasn't finished, checkpoint and report partial.

**What could go wrong unsupervised.** (a) Sleep archives/merges aggressively and the copy diverges in ways that make the delta hard to attribute — mitigated by logging every sleep action (archive/merge/edit counts) to the findings file. (b) `sleep.py` shells out to the `claude` CLI, which can stall on auth or rate limits — the wall-clock cap catches this. (c) A slept copy is still a copy; the live store is never touched.

**Residual human review.** The *interpretation* — "sleep helps by +X NDCG, hurts temporal by −Y" — is a claim for `architecture.md`/`experiments.md` and must be reviewed before merge. Whether a positive result should change the live sleep cadence is a stewardship decision, not an autonomous one.

**Expected information value.** Very high. Resolves an open question that has blocked confident statements about the whole consolidation subsystem. A null result is equally valuable and publishable ("consolidation runs, looks reasonable, does not move retrieval metrics").

**Artifact:** [`handoff/sleep-fork-session.md`](handoff/sleep-fork-session.md).

---

### 2. Proactive-injection offline floor study — *publishable either sign*

**Objective metric.** F-beta (β>1, recall-weighted) of a candidate score floor at the binary surface/skip decision, scored against use/ignore labels reconstructed from `recall_meta` + feedback events. Precision/recall curve over the floor sweep; the accept test is whether *any* floor beats both always-inject and never-inject.

**Loop design (deterministic floor only).**
1. Copy store to `$SCRATCH`. Extract, per historical turn, the RRF-only candidate set and whether each surfaced memory was used (utility>0) or ignored (from the ~13k feedback events + `recall_meta`).
2. Sweep the floor over the observed RRF-score range. For each floor, compute what it would surface and score against the use/ignore labels.
3. Emit the P/R curve, the F-beta-optimal operating point, and the two baselines (always/never).

**Stopping criteria.** Fixed floor grid (e.g. 50 points across the score range). No loop.

**The off-policy rail (critical).** A **deterministic** floor is clean replay — the logged feedback was collected under the current pull-only policy, and a fixed floor is a deterministic function of the logged scores, so direct replay is valid. A **stochastic** (Thompson / temperature) gate is *not* clean replay: candidate stochastic policies must be scored by importance-weighted off-policy estimation at higher variance. **The autonomous run does the deterministic floor only.** Stochastic gating is explicitly deferred to an interactive follow-up (§I3) — this matches the proposal's own sequencing ("tune the deterministic floor first, then layer stochasticity in a second pass").

**Cost / wall-clock.** 0 metered $ (RRF-only, fastembed local, replay over existing logs). 2–4h, mostly the replay script run. **Ceiling:** wall-clock 4h.

**What could go wrong unsupervised.** (a) Raw RRF scores may be miscalibrated for the binary decision; the study can *fit* a calibration on the surface/skip labels but must not reuse the repo's GT isotonic calibration (that operates on judge scores, not live retrieval scores) — the artifact flags this. (b) Label reconstruction is the fiddly part; getting the feedback↔recall_meta join window right (the real-pathology miner uses a 60s window) is the main correctness risk — the artifact pins the window and logs join statistics.

**Residual human review.** The F-beta operating point and the β weight encode a miss-vs-noise value judgment that is Alexis's call. Whether a positive result graduates to the `UserPromptSubmit` hook build is a stewardship decision.

**Expected information value.** High. A positive result validates the system's main missing capability (proactive recall) before any hook code is written; a negative result ("binary surface signal as weak as the cliff cutoff, R²≈0") is itself publishable and closes the question — score-based gating fails at both granularities.

**Artifact:** [`handoff/floor-study-session.md`](handoff/floor-study-session.md).

---

### 3. Objective one-offs bundle — *highest information value per hour*

Six small, deterministic, zero-API measurements, each closing a roadmap open question. Individually modest; together the best ratio in the catalog because the marginal cost is minutes and none needs a human mid-run. Run as a sequence in one session; each writes its own JSON + a paragraph in a shared `findings-oneoffs.md`.

| Sub | Question | Method | Roadmap | Ready? |
|-----|----------|--------|---------|--------|
| a | **Effective dimensionality** | PCA participation ratio `d_eff = (Σλ)²/Σλ²` over the 384-d embedding matrix | #22 | script (minutes) |
| b | **User-query bias** | Prepend `user:` to GT queries, re-run `eval_retrieval.py --dataset production`, measure NDCG delta | #23 | Ready (flags exist) |
| c | **Edge-age distribution** | Age + last-update + never-reinforced fraction over the edge table; correlate age vs co-retrieval utility | "Do misclassified edges persist?" | script (minutes) |
| d | **Bimodal utility fit** | Fit Beta-mixture / Gaussian-mixture to per-memory utility, BIC vs single Beta | "Is the utility distribution bimodal?" | extend `utility_calibration.py` |
| e | **PPR seed-set perturbation** | Perturb feedback coeff ±10%, measure top-N seed-set overlap + PPR score correlation | "Is PPR seed selection stable?" | script |
| f | **Latency profiling** | p50/p95/p99 of `recall()` at current corpus size | benchmarks.md § Proposed | instrument one run |

**Loop design.** No loop — a fixed battery. Each sub is a run-once measurement.

**Stopping criteria.** Battery exhausted (6 measurements) or 4h wall-clock.

**Cost / wall-clock.** 0 metered $. 2–4h including writing the 4 small scripts (a, c, d, e) — b and f use existing infra. **Ceiling:** wall-clock 4h; skip any sub whose script fights the schema and log the skip (no silent truncation).

**What could go wrong unsupervised.** Very little — these are read-only analyses on a copy. The one risk is a sub-script mis-reading the schema (e.g. edge table column names); the session verifies each against a tiny sample before the full run and logs any sub it couldn't complete.

**Residual human review.** Each result is a small claim for `roadmap.md`/`experiments.md` — reviewed before merge. `d_eff` in particular feeds the Geometry-of-Forgetting scale prediction, an interpretation worth a human eye.

**Expected information value.** Medium absolute, very high per hour. Clears six long-standing roadmap open questions in one cheap overnight slot.

---

### 4. Prospective indexing (anticipated queries) — *strong external precedent*

**Objective metric.** NDCG@5k / R@10 on the existing GT query set, before vs after appending anticipated queries to each memory's enriched text and re-embedding.

**Loop design.**
1. Copy store to `$SCRATCH`.
2. For each active memory, generate 2–3 hypothetical future recall queries (one `claude` CLI or Haiku call per memory; the same pattern as REM classification). Append to the enriched text.
3. Re-embed (fastembed, local, free) and rebuild the vec index on the copy.
4. `eval_retrieval.py --dataset production --configs reranker` before and after; report the delta. (First verify `reranker_model.txt`/`reranker_features.json` exist on the copy — present after the arc-2.5 restore lands; see §Feasibility "Reranker load path". If absent, do NOT convert the stale pkl — stop or run labeled formula-only.)

**Stopping criteria.** One pass over the active set (~478 memories). No loop. Optional ablation: 1 vs 3 anticipated queries, capped at those two arms.

**Cost / wall-clock.** <$1 (Haiku for ~478 short generations) or 0 metered (claude CLI). Re-embedding is local. 3–6h. **Ceiling:** $2 hard cap on generation; wall-clock 6h.

**What could go wrong unsupervised.** (a) Generated queries could be generic and add noise rather than signal — the null result ("enriched embedding already captures this") is an accepted, documented outcome (roadmap #10 accept criteria). (b) Re-embedding the whole copy is the heavy step; done locally so no cost, but verify the vec index rebuilds cleanly on a sample first.

**Residual human review.** This is a **write-path** intervention (STEWARDSHIP Priority 2). Running it offline and reporting NDCG is autonomous; deciding whether it graduates into live `remember()`/sleep-REM is a Priority-2 stewardship call. The prose claim is reviewed before merge.

**Expected information value.** High. Two independent external systems (Kumiho 37.5→84.4 on the >6-month cliff; HyperMem `potential` field → 92.73% LoCoMo) converge on this, and it directly targets Somnigraph's measured multi-hop vocabulary gap. Strong prior for a real effect; a null is still a clean finding about our enriched-embedding baseline.

---

### 5. LoCoMo expansion method ablation (roadmap #21) — *closes the P5 remainder*

**Objective metric.** Per-method R@10 / MRR delta on LoCoMo (turn-level, all 10 conversations), via `eval_retrieval.py --dataset locomo`.

**Loop design.** Fixed matrix, all verified flags:
- **Active-only** vs **all-six**: `--expand-session --expand-keyword --expand-entity-bridge` vs `--expand-all`. Confirms the three dead methods (rocchio 0%, multi_query 2%, entity_focus 4%) add no signal.
- **Drop-one-out** over the three active methods (leave-one-out) for marginal contribution.

**Stopping criteria.** 5 configs × 10 conversations, fixed. No loop.

**Cost / wall-clock.** ~0 metered $ (retrieval-only — no reader/judge; 294 MB OpenAI embed cache is warm). 2–4h. **Ceiling:** wall-clock 4h.

**What could go wrong unsupervised.** Low risk — pure retrieval eval on existing DBs and a warm cache. Confirm the per-conversation LoCoMo DBs exist under `~/.somnigraph/benchmark/` before the full sweep (§Feasibility). If a conversation DB is missing it must be rebuilt via `run.py --build-graph` first — a prep step to flag, not silently skip.

**Residual human review.** *Removing* the dead methods is a code change (interactive). The ablation only produces the evidence that removal is safe.

**Expected information value.** Medium. Closes the last open item in STEWARDSHIP Priority 5 and lets `experiments.md` document the retrieval arc end to end. Confirmatory more than exploratory (the fire-rate data already strongly implies the answer).

---

### 6. FSI seed-stability audit (roadmap #9) — *the fan-out shape*

**Objective metric.** Per-feature gain-importance variance across 10–20 LightGBM retrains with different random seeds; flag features with high rank variance, grouped by the four known correlated clusters (fts_rank/fts_bm25; feedback_mean/ucb_bonus/fb_time_weighted; age_hours/hours_since_access/session_recency; fts_bm25_norm/vec_dist_norm).

**Loop design.** Embarrassingly parallel — one training run per seed, fan out across a workflow, then a single reduce that computes per-feature importance mean/variance/rank-stability. First-mover bias (arXiv:2603.22346) predicts correlated features get path-dependent importance concentration; the variance across seeds disambiguates genuine signal from seed-luck, and specifically tests whether the R@10-vs-NDCG feature-set disagreement is a first-mover artifact.

**Stopping criteria.** Fixed seed grid (20 seeds). No loop.

**Cost / wall-clock.** 0 metered $ (LightGBM training only, warm feature cache). 1–3h, less with fan-out. **Ceiling:** 20 seeds or 3h.

**Readiness / the one prep gap.** `train_locomo_reranker.py` already exposes `--random-seeds 1 2 … 20` → **the LoCoMo reranker audit is ready today**. The production 31-feature `train_reranker.py` has **no `--seed` flag** — a one-line passthrough into the LightGBM params is needed to audit the production model that roadmap #9 actually names. Add-the-flag is itself a tiny catalog prep step; the workflow artifact runs the LoCoMo variant now and notes the production hook.

**What could go wrong unsupervised.** Low — deterministic training on cached features. The reduce step must align feature names across runs (stable ordering); the workflow pins the feature list.

**Residual human review.** Interpreting "feature X is seed-unstable, therefore treat its importance skeptically" and deciding whether a future retrain needs seed-averaging or an independent ensemble.

**Expected information value.** Medium. Partly pre-empted by the 2026-05-08 content-residual audit (which already showed fb_count/age_days dominate cold-start demotion), but the *seed-variance* question specifically remains open and this closes it cleanly.

**Artifact:** [`handoff/fsi-seed-audit.workflow.js`](handoff/fsi-seed-audit.workflow.js).

---

### 7. LoCoMo sleep-pass + feedback-loop ablations (roadmap #21 siblings)

**Objective metric.** End-to-end QA accuracy (Opus judge) delta from toggling the sleep pass and the feedback loop on the LoCoMo QA pipeline.

**Loop design.** `run.py` exposes `--feedback-loop` (two-pass: run, feed scores back, re-run) directly → feedback-loop ablation is a single flag flip. The **sleep-pass** ablation needs assembly: run `sleep.py` over the per-conversation LoCoMo DBs, then re-run `run.py`, vs. a no-sleep control.

**Stopping criteria.** 3 arms (baseline / +feedback-loop / +sleep), fixed.

**Cost / wall-clock.** **$5–20 metered** — `run.py` calls `gpt-4.1` reader + judge per question × ~1540 questions × arms. This is the one grindable candidate with a non-trivial dollar cost; the ceiling is real. **Ceiling:** $25 hard cap, `--no-judge` first pass to bound cost, batch-judge second.

**What could go wrong unsupervised.** Cost overrun is the main risk (LLM reader+judge per question) — the `--no-judge` staging and the hard $ cap contain it. Sleep-on-LoCoMo may behave oddly (LoCoMo DBs lack feedback/edges) — log sleep actions.

**Residual human review.** The QA-delta claim for `benchmarks.md`; deciding whether sleep belongs in the LoCoMo pipeline story.

**Expected information value.** Medium. Completes the LoCoMo ablation trilogy (expansion #5 + these two) that STEWARDSHIP P5 asks for.

---

### 8–9. Graph null models / query-difficulty clustering / paraphrase robustness

Grouped because all three are multi-hour judged/deterministic sweeps that need a **variant-builder or clustering harness built first** (Prep), and none is arc-critical.

- **Graph null models (#17):** build similarity-only / edge-shuffled / edge-type-restricted / temporal-only variants of the graph on copies, eval each. Tests whether graph structure adds real signal beyond similarity. Deterministic, 0 metered $. Prep: the four variant builders.
- **Query difficulty clustering (#11):** cluster GT queries by type (length, theme count, temporal, cross-domain), per-cluster NDCG. Deterministic clustering + existing GT. Prep: the clustering harness. Feeds the narrative-summary experiment's cross-domain query subset.
- **Paraphrase robustness (#15):** paraphrase GT queries (one LLM pass, <$2), re-run, measure NDCG delta — tests vocabulary co-adaptation. Prep: the paraphrase generator.

All three are legitimate overnight grinds once the prep piece exists; each prep piece is a small named build that is itself a candidate. Ranked below the top tier because of the build-first gate and non-arc status.

---

### 10. LongMemEval-S retrieval harness (ideas Tier-1 #16) — *build-first, higher unsupervised risk*

Fills Somnigraph's complete lack of external LongMemEval numbers with zero-LLM R@5/R@10/NDCG@10/MRR by question type over the 500-Q distractor set (with the evidence-session-dedup trick so recall can't exceed 1.0). Once built, the metrics are fully objective and deterministic — an ideal grind.

**The catch:** it is a *new adapter*, not a run of existing infra. Building non-trivial eval infrastructure unsupervised is the highest-risk item here — a subtly wrong adapter produces confident, wrong external numbers, which is worse than no numbers (honest accounting). **Recommendation:** build the adapter interactively (or build-then-human-verify against a known-good reference score), *then* the repeated runs are grindable. Listed for completeness and high value, flagged as not-safely-autonomous for the build phase.

---

## Interactive-only (must NOT run unsupervised) — with reasons

- **I1. Ship the instruments — shadow near-dup logging, secret redaction, `memory_meta` fix (arc step 2).** Mutates the live `remember()` code path and, once deployed, the live store. Even shadow-mode (acts on nothing) writes to production code and needs a diff review. Spec-driven mechanism, not a metric loop. → Sonnet, interactive.
- **I2. Write Guard dedup/supersede gate thresholds.** The shadow-mode philosophy is explicit: thresholds must be set from a *measured* production near-dup distribution, not guessed. Choosing the cutoff is a human read of a distribution that doesn't exist yet (I1 must run in production for weeks first). The gate *code* can be drafted; the *thresholds* cannot be autonomously chosen.
- **I3. Stochastic / Thompson floor gating.** Off-policy evaluation (importance-weighted, high variance) over logs collected under a different policy. The proposal itself sequences this *after* the deterministic floor. Subtle enough that an unsupervised run could produce a confidently-biased estimate. → interactive follow-up to #2.
- **I4. Any prose claim merged into `architecture.md` / `experiments.md` / `roadmap.md` / `benchmarks.md` / `STEWARDSHIP.md`.** Honest accounting (STEWARDSHIP invariant #1) requires the *interpretation* of every result be reviewed before it becomes documentation. The autonomous runs produce JSON + a draft findings file; a human ratifies the claim.
- **I5. Deploying any retrained reranker to production (e.g. V5+5).** Deferred in STEWARDSHIP anyway (adversarial supply is structurally tight). Training a model offline is grindable; *cutting production over to it* touches live scoring and is a stewardship decision.
- **I6. STEWARDSHIP priority reordering / arc resequencing.** Definitionally a human judgment.
- **I7. Contradiction-detection research (roadmap #20).** Needs a human-annotated set of ~50 known contradiction pairs before any F1 can be measured. The annotation is the bottleneck and it's human.
- **I8. Event-time schema, DB branchability, autobiographical-narrative sleep mode.** New capabilities gated on design decisions (schema shape, provenance model, merge cadence), not closed metric loops.
- **I9. LongMemEval-S adapter *build* (see #10).** The build phase is not safely autonomous; the runs after verification are.
- **I10. Reranker-loading diagnostic — CONFIRMED and worse than first reported (orchestrator-verified 2026-07-01; fix in flight as arc step 2.5).** Production has run **formula fallback since 2026-04-07** (commit de6613f switched the loader to native `.txt`; no `.txt` ever landed). The on-disk `.pkl` (2026-03-20) is the **26-feature** model — **no V5-era artifact exists anywhere on disk** — so the pkl→txt converter this document originally proposed would deploy a stale, feature-misaligned model under the 31-feature live extractor. The only honest fix is a retrain with the V5+3b recipe (`train_reranker.py` exports the `.txt`+`.json` unconditionally via `_export_for_mcp`). That restore — plus the supersede-path vec/FTS index-hygiene fix and the NaN-encoding it unblocks — is running as stewardship arc step 2.5 (interactive, Opus). Until it lands, any `--configs reranker` run measures the formula and must say so.

---

## Feasibility appendix (verified, not assumed)

Checked by reading argparse blocks and `ls`-ing the paths — not guessed.

**Data roots.** Production store + GT + reranker at `~/.claude/data/` (fastembed 384d). `~/.somnigraph/` is a near-empty default (143 KB stale `memory.db`). `select_real_pathology_targets.py` confirms by hardcoding `SOMNIGRAPH_DATA_DIR=~/.claude/data`, `SOMNIGRAPH_EMBEDDING_BACKEND=fastembed` as its defaults.

**Verified present:**
- `~/.claude/data/memory.db` (live store); `~/.claude/data/ground_truth.json` (319 KB, may be stale vs the reranker's training GT — prefer `tuning_studies/gt_calibrated.json`, the `train_reranker.py --gt` default).
- LoCoMo dataset `~/.claude/repos/locomo/data/locomo10.json` (2.8 MB); embed cache `~/.claude/data/bench_locomo_embeddings.pkl` (294 MB); LoCoMo reranker `~/.somnigraph/benchmark/locomo_reranker_model.pkl`; corrected-GT audit `scripts/locomo_bench/locomo_audit_errors.json` (checked into the repo); v6 extractions `scripts/locomo_bench/extractions/conv{0..9}_v6.json` (checked in).
- **Reranker load path — confirmed outage, fix in flight (see §Interactive #I10 for the full account).** `reranker.py:_load_model()` loads a **native LightGBM `.txt`** via `lgb.Booster(model_file=MODEL_PATH)`, `MODEL_PATH = DATA_DIR/tuning_studies/reranker_model.txt`, features from `reranker_features.json`. The live `~/.claude/data/tuning_studies/` holds **only `reranker_model.pkl` + `reranker_features.pkl` (2026-03-20, the 26-feature era) — no `.txt` anywhere, no V5-era artifact on disk at all.** The loader falls back to the formula and has since 2026-04-07. **Do NOT convert the pkl** — it is the stale 26-feature model, feature-misaligned with the 31-feature live extractor; the fix is the V5+3b-recipe retrain running as arc step 2.5. **Consequence for experiments:** until the restore lands, any `--configs reranker` run on a copy of `~/.claude/data` measures the *formula*. Affected artifacts (#1 sleep fork, #4 prospective indexing) carry a verify-or-stop step (converter removed).

**Verified flags (exact):**
- `eval_retrieval.py`: `--dataset {locomo,production}` (required), `--configs {bare,formula,reranker,+theme,+ucb,+hebbian,locomo_reranker}`, `--conversations`, `--recall-limit` (20), `--recall-limits` (sweep), `--gt-path` (default `~/.claude/data/ground_truth.json`), per-method `--expand-entity-focus/--expand-multi-query/--expand-keyword/--expand-session/--expand-entity-bridge/--expand-rocchio`, `--expand-all`, `--graph-resolve`, `--synthetic-coverage`.
- `run.py`: `--reader-model` (gpt-4.1), `--judge-model` (gpt-4.1), `--prompt-mode {core,somnigraph}`, `--expand-all`, `--build-graph`, `--feedback-loop`, `--no-judge`, `--locomo-reranker`, `--skip-adversarial`.
- `sleep.py`: no args; respects `SOMNIGRAPH_DATA_DIR`; LLM via `claude` CLI subprocess.
- `train_reranker.py`: `--gt` (default `DATA_DIR/tuning_studies/gt_calibrated.json`), `--folds`, `--n-estimators`, `--lambdarank`, … **no `--seed`** (FSI prep gap).
- `train_locomo_reranker.py`: `--random-seeds N …` (FSI-ready), `--select {forward,backward,…}`, `--select-metric {ndcg@10,r@10,r@20,mrr}`, `--save-model`.
- `probe_recall.py`: `--mix`, `--adversarial-source {real,audit}`, `--adversarial-rank-threshold` (real-recall infra present; V5+5 deferred).

**Prep gaps to build (each a small candidate in its own right):**
1. `train_reranker.py --seed` passthrough (one line) — unblocks FSI on the *production* model (#6).
2. Floor-study replay/label-reconstruction script (#2) — reads `recall_meta` + feedback, 60 s join window.
3. One-offs scripts for d_eff / edge-age / bimodal / PPR-perturb (#3).
4. Graph variant-builders, clustering harness, paraphrase generator (#8–9).
5. LongMemEval-S adapter (#10) — build interactively, not overnight.
6. ~~`reranker_model.txt`/`reranker_features.json` export from the existing `.pkl`~~ **Struck 2026-07-01: the converter approach was wrong.** The on-disk pkl is the stale 26-feature March model (no V5-era artifact exists on disk); converting it would deploy a feature-misaligned model. The artifacts are being restored via a V5+3b-recipe retrain (arc step 2.5). See §Interactive #I10.

---

## Handoff

Ready-to-launch artifacts for the top three, safety rails baked in, under [`handoff/`](handoff/):

- [`handoff/sleep-fork-session.md`](handoff/sleep-fork-session.md) — #1, autonomous session prompt (single long grind in a worktree).
- [`handoff/floor-study-session.md`](handoff/floor-study-session.md) — #2, autonomous session prompt (deterministic floor only; stochastic explicitly deferred).
- [`handoff/fsi-seed-audit.workflow.js`](handoff/fsi-seed-audit.workflow.js) — #6, Workflow script (seed fan-out + reduce), LoCoMo-ready today.

The remaining candidates have full loop designs above; their artifacts can be generated on request. **Nothing here has been launched — Alexis chooses what runs overnight.**
