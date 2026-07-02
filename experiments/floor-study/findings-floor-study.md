# Proactive-Injection Offline Floor Study — Findings

**Arc step 3 / autonomous-experiments #2.** Deterministic replay over the existing
event log. No writes to the live store, no paid API calls, deterministic floor only
(stochastic gating out of scope by design).

> **Reviewed and ratified 2026-07-01** (orchestrator review; join gate passed — pinned
> 60s window, 91.8% match, 1 collision). Decision: **the injection hook is not built on
> offline evidence**; outcome recorded in `docs/proposals/proactive-injection.md` and
> `docs/roadmap.md`. Review addition: the β=2 choice does not change the verdict —
> precision-weighted F₀.₅ on the population-A curve also loses to always-inject
> (best floor ≈0.80 vs 0.82), so the operating-point rejection is β-robust, not an
> artifact of the recall-weighted value judgment.

---

## Verdict

**Split, and honest in both directions.**

1. **The RRF-family score carries real, statistically robust discrimination signal** for
   the binary surface/skip decision. AUC = 0.64 (per-candidate, artifact-literal),
   **0.74 among candidates the agent actually rated**, 0.64 per-turn. This is clearly
   above chance and a genuine improvement over the cliff-detector result (list-level
   score features were anti-predictive there, R² < 0). The design's core hypothesis —
   the coarse top-of-ranking binary is easier than the fine-grained in-list cut — is
   **supported on discrimination.**

2. **No floor produces a usable F-beta(2) operating point.** The literal accept test
   ("a floor beats both always-inject and never-inject on F-beta") is met only by a
   **practically-zero margin** in the per-candidate framing (ΔF₂ = +0.00013, bootstrap
   CI95 [+0.00006, +0.00022] — statistically consistent but negligible; the "optimal"
   floor drops 33 of 21,220 candidates), and it **fails outright in the per-turn framing**
   (ΔF₂ median 0.0, CI95 [0, +0.0004], only 26% of bootstrap resamples show any gain).
   The per-turn unit is the design's actual decision ("is there anything worth one line
   this turn"), so the decision-relevant framing rejects.

**Net:** the score is not signal-free (the cliff's null does not fully carry over), but
score-based gating does **not** earn a hook build on this evidence. The discrimination
exists; the recall-weighted operating point that would justify shipping does not. The
reason is structural, not tuning: see "Why the signal doesn't convert" and "The
structural limit."

---

## What was actually measured (read before trusting any number)

**The artifact's premise did not hold, and it forced a documented method change.**

The artifact assumed the copied store was local-fastembed, so pure RRF could be recomputed
for free (Step 2). The **actual copied production store is OpenAI-1536-dim** (verified: the
db.py embedding-dim guard hard-fails fastembed against it; `memory_vec` embeddings are 1536
floats). Consequences:

- Recomputing RRF requires OpenAI query embeddings, i.e. a **paid API call — forbidden by
  the study's cost rail (0 metered $).**
- fastembed cannot substitute: 384-dim vectors are a different, incompatible space; the
  guard blocks the mismatch, and the similarities would be meaningless anyway.

So recompute is impossible inside the cost ceiling. Per the artifact's own Step-2 fallback
("**if recall_meta already stored per-candidate scores, use them**"), this study uses the
**stored** per-candidate scores logged in each `recall_meta` event (`kept` and
`beyond_limit` lists, each `(memory_id[:8], score)`). Zero API calls, zero cost.

**Provenance of the stored score — the credibility hinge:**

`recall_meta`'s stored score is the pipeline's fused score *at decision time*, and its
meaning depends on which scoring path was live:

- **Reranker era** (LightGBM live): stored score = reranker output. **Contaminated** for a
  study of RRF gating. Detectable by magnitude — reranker outputs run large (top_score up
  to 2.216); RRF fusion tops out near 0.27.
- **RRF-fallback era** (`tuning_studies/reranker_model.txt` absent, formula path): stored
  score = `rrf_fuse` (RRF fusion + UCB exploration bonus) + hebbian + PPR-expansion. This
  is the pipeline's **no-reranker, no-LLM** fused score, which is exactly the cheap path
  the hint layer's floor is specified to gate on.

The reranker `.txt` the loader needs is **absent** in the store (only the `.pkl` is present),
so the current path is formula/RRF-fallback, consistent with the known outage. Empirically,
stored top_scores carry a reranker signature (> 0.5) **only through 2026-03-22**; from
2026-04 on they collapse to RRF-fusion magnitudes (monthly median ≈ 0.13, max ≤ 0.27). This
study **classifies each turn's path conservatively** (reranker if *either* the date is before
the 2026-04-08 outage cut *or* the top_score exceeds 0.5) and uses **only the RRF-fallback
era** (1,533 turns) as the study population. Reranker-era turns (1,735) are excluded.

**This is the deviation the artifact told me to log and continue conservatively on. It is
forced (openai store + cost rail) and sanctioned (Step-2 fallback). It is the single
biggest caveat on the study.** See "Provenance threats" for what it does and does not put
at risk.

---

## Join statistics (the other credibility hinge)

Feedback is joined to the `recall_meta` that produced it by **identical query string,
nearest in time, within a 60-second window** — matching
`scripts/select_real_pathology_targets.py` (`DEFAULT_MATCH_WINDOW_SECONDS = 60`).

| quantity | value |
|---|---|
| recall_meta turns loaded | 3,268 |
| feedback events considered | 45,493 |
| feedback matched to a turn | **41,743 (91.8%)** |
| orphaned: no query field | 137 |
| orphaned: no same-query turn | 2,421 |
| orphaned: outside 60s window | 1,192 |
| **total orphaned** | **3,750 (8.2%)** |
| window collisions (>1 same-query turn ≤60s) | **1** |
| 8-char memory_id prefix collisions | **0** |

The join is clean: 91.8% match, a single window collision, no prefix collisions. Orphan
rate 8.2% is dominated by feedback whose query never appears in a `recall_meta` (2,421),
i.e. feedback attached to a recall whose meta was not logged, not a windowing failure.

**Label definition (artifact Step 1):** per (turn, surfaced/kept candidate),
`used = a joined feedback references it with utility > 0`; else `ignored` (this includes
candidates surfaced but never rated — utility 0 by convention).

---

## Results

RRF-fallback era only. Populations A–C are legitimate; D is a cautionary artifact.

| population | n | base rate | AUC | always-inject F₂ | best-floor F₂ | ΔF₂ (bootstrap CI95) | accept? |
|---|---|---|---|---|---|---|---|
| **A** per-candidate (artifact-literal) | 21,220 | 0.783 | 0.640 | 0.9475 | 0.9477 | +0.00013 [+0.00006, +0.00022] | marginal (negligible) |
| **B** per-candidate, explicitly-rated only | 19,548 | 0.850 | **0.739** | 0.9659 | 0.9662 | +0.00013 [+0.00007, +0.00021] | marginal (negligible) |
| **C** per-turn top-score (design-literal) | 1,533 | 0.881 | 0.639 | 0.9738 | 0.9738 | 0.0 [0, +0.0004] | **no** |
| D full-range incl. beyond_limit | 473,479 | 0.035 | 0.994 | 0.1539 | 0.8630 | (not the accept test) | **invalid** |

- **A** is the artifact's literal per-candidate test. The floor "wins" but by ~1e-4 F₂.
- **B** removes the 1,672 unrated kept candidates (noise in the ignored class). AUC rises
  0.64 → **0.74**: among candidates the agent actually judged, the score separates
  "useful" from "explicitly rated zero" respectably. Still no usable F₂ gain.
- **C** is the design's real decision unit (per turn, gate on the top score). It **rejects**:
  the F₂-optimal action is to surface every turn.
- **D** adds the 452k `beyond_limit` candidates as negatives. Its AUC 0.994 is a
  **censoring artifact**, not a result: beyond_limit candidates were never shown to the
  agent, so they are ignored *by construction* and score lower than kept *by construction*.
  The near-perfect separation is mechanical. It is included only to document the trap and
  is not evidence the floor works.

**Discrimination is magnitude, not just rank.** AUC on the continuous score exceeds AUC on
rank position alone (A: 0.640 vs 0.560; B: 0.739 vs 0.660), so a score floor carries ~0.08
AUC beyond a simple rank cutoff. The score's magnitude is doing real work.

### The precision/recall curve (population A) — why the signal doesn't convert

| floor | surfaced | precision | recall | F₂ |
|---|---|---|---|---|
| 0.023 (≈always-inject) | 21,220 | 0.783 | 1.000 | 0.9475 |
| 0.036 | 21,181 | 0.784 | 0.9997 | 0.9476 |
| 0.061 | 13,504 | 0.836 | 0.679 | 0.706 |
| 0.086 | 5,769 | 0.879 | 0.305 | 0.351 |
| 0.124 | 1,790 | 0.910 | 0.098 | 0.119 |
| 0.161 | 80 | 0.963 | 0.005 | 0.006 |

Precision **does** respond to the floor (0.78 → 0.96), confirming the AUC signal is real.
But precision rises **slowly** (roughly +0.13 across the usable range) while recall
**collapses** (1.0 → 0.31 → 0.005). Under β=2 (recall weighted 4×), giving up 69% of recall
to buy +0.10 precision is a losing trade at every floor. The base rate is simply too high
for a recall-weighted gate to help: when 78–88% of what you would surface is already useful,
the always-inject baseline is near-unbeatable.

### Calibration (artifact Step 4) — provably cannot change the verdict

An isotonic (PAVA) calibration of `rrf_score → P(used)` was fit on population A. It is
monotone non-decreasing (verified), therefore the surfaced set `{score ≥ floor}` is
order-identical to `{calibrated ≥ floor'}`, so the entire P/R/F-beta curve **and its
optimum are unchanged** (F₂-optimal 0.9477, identical to raw). Calibration only makes the
threshold interpretable; it is mathematically incapable of rescuing the accept test for any
monotone transform. Reported for completeness, per Step 4's "borrow the method." No
non-monotone calibration is justified by the data.

---

## The honest comparison to the cliff result

The cliff-detector study found list-level score features **anti-predictive** of the in-list
content cutoff (R² < 0: worse than predicting the mean). This study finds the coarser binary
surface/skip decision is **not** in that null regime: AUC 0.64–0.74 is genuine discrimination.
So the design's reasoning ("the top-of-ranking binary is easier than the in-list cut") holds
where it was tested. **But** the improvement is confined to *discrimination* (AUC), and does
not reach the thing that would justify the feature (a recall-weighted operating point that
beats doing nothing special). The cliff failed at "can you predict it at all"; the floor
fails one step later, at "can you turn the prediction into a gate that beats the trivial
baseline." Better than the cliff, still not shippable on this evidence.

## The structural limit (why offline replay may be the wrong instrument here)

The event log only contains turns where `recall()` was **actually called** — that is what
generates a `recall_meta`. The base rate of "useful" among surfaced candidates is 0.78–0.88
precisely because everything in the log was retrieved on purpose by an agent that decided it
wanted memories. But the proactive-injection feature's entire value proposition is the
**opposite** population: turns where the agent did **not** call recall, yet a hint would have
helped. Those turns leave no `recall_meta` and are unobservable here. The offline study can
therefore validate discrimination on already-retrieved sets, but it **structurally cannot**
measure the proactive use case or produce an operating point that represents it. A high
offline base rate is not a property of the floor; it is a selection property of the log. Any
real test of proactive value needs online data (hint shown / not, on no-recall turns), which
is exactly the live A/B the design defers to the hook-build phase.

---

## Provenance threats (what the caveats do and do not put at risk)

- **Hebbian inflation (threatens only the positive finding).** The stored RRF-era score
  includes a hebbian co-retrieval term. A memory used often is retrieved often, gets a
  higher hebbian boost, and correlates with being used again — so some of the AUC 0.64–0.74
  may be *popularity*, not RRF *relevance*. The UCB term pushes the other way (more feedback
  → lower variance bonus → lower score for used items), partially offsetting. Net direction
  is ambiguous and cannot be isolated without recomputing pure RRF, which the cost rail
  precludes. **This threatens the encouraging half (signal exists) only.** The discouraging
  half (no usable F-beta point) is driven by the base rate and is robust to score provenance.
- **Unrated-as-ignored (addressed).** Population A labels 1,672 unrated kept candidates as
  ignored, adding noise. Population B drops them; AUC rises to 0.74 and the F-beta verdict is
  unchanged. The negative operating-point result survives both labelings.
- **Corpus drift (avoided).** Because scores are the ones logged at decision time (not
  recomputed against a drifted corpus), there is no drift confound. The cost of that choice
  is the hebbian provenance caveat above.
- **Reranker contamination (excluded).** Reranker-era turns are removed by a conservative
  date-OR-magnitude filter; the RRF-era population's max top_score is ≤ 0.27, well below any
  reranker signature.

---

## Reproduce

From a worktree off `main` (this study ran in `exp/floor-study`), against a **copy** of the
data store (never the live store):

```sh
# 1. isolate a copy of the store (example path)
S="$HOME/.somnigraph-exp/floor-<run>"
cp -r "$HOME/.claude/data" "$S"
export SOMNIGRAPH_DATA_DIR="$S"
# NOTE: do NOT set SOMNIGRAPH_EMBEDDING_BACKEND — these scripts do no embedding.
# The store is OpenAI-1536; no embedding backend is loaded or needed.

cd experiments/floor-study

# 2. reconstruct labels from the event log (join stats + provenance to stdout)
python build_floor_labels.py --emit-beyond      # writes floor_labels.json (+ beyond jsonl)

# 3. sweep the floor (F-beta curve, AUC, bootstrap CI, calibration)
python sweep_floor.py                             # writes floor_sweep.json
```

`floor_labels_beyond.jsonl` (~667k rows) is regenerable and intentionally **not committed**.
Committed artifacts: the two scripts and this file. `floor_labels.json` (kept rows + meta) and
`floor_sweep.json` (all metrics) are **local-only** (gitignored at review): regenerable from the
event log, and they carry live-usage telemetry (per-turn timestamps, per-memory utility
histories) that doesn't belong in a public repo. All decision-relevant numbers are quoted above.

Environment of record: store copy `~/.somnigraph-exp/floor-5900abd-*`, worktree at commit
`5900abd` (branch `exp/floor-study`, off `main`). Python 3.14, sqlite3 stdlib only (no
numpy/sklearn dependency; PAVA and AUC are implemented in-script).

---

## Both outcomes were pre-registered as publishable

- **Positive** would have been: a floor beats both baselines on F-beta, validating proactive
  recall's core assumption before any hook code.
- **Negative** is: the binary surface signal, though real, does not yield a usable F-beta
  operating point, and the offline log is structurally unable to represent the proactive use
  case. That is the result, and it is worth as much as a positive one: it says **do not build
  the hook on offline evidence** — if the idea is pursued, it needs an online test on
  no-recall turns, and the gate design should not lean on a recall-weighted F-beta at this
  base rate. The one salvageable positive is that RRF score magnitude does carry binary
  signal (AUC 0.74 among rated candidates, beating both the cliff null and a rank-only
  cutoff), which a future online design can build on — with the hebbian-provenance question
  resolved by a pure-RRF recompute once a fastembed store or an embedding budget is available.
