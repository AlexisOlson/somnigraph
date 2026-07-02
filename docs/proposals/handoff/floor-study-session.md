# Handoff: Proactive-Injection Offline Floor Study — autonomous session prompt

**Shape:** single grind → hand this whole file to a fresh autonomous Claude session (Opus, per the arc). Deterministic replay over existing logs; produces a P/R curve + F-beta operating point. **Deterministic floor only — stochastic gating is explicitly out of scope** (off-policy, deferred to an interactive follow-up).

**Catalog entry:** [`../autonomous-experiments.md`](../autonomous-experiments.md) #2. **Design:** [`../proactive-injection.md`](../proactive-injection.md) § Experiment. **Arc step 3.**

---

## SESSION PROMPT (paste from here down)

You are running an unsupervised offline study on Somnigraph. Follow the safety rails exactly. You will NOT be able to ask questions mid-run; when ambiguous, log the choice and continue conservatively.

### Objective

Test the core assumption of the proactive-injection design: does a coarse **binary surface/skip floor** on RRF-only retrieval scores carry usable signal — even though the cliff-detector study found the fine-grained in-list cutoff anti-predictable (R²<0)?

Primary metric: **F-beta (β=2, recall-weighted)** of a candidate score floor at the binary surface/skip decision, scored against use/ignore labels reconstructed from the event log. Accept if a floor exists that beats **both** always-inject and never-inject on F-beta.

### Hard safety rails (non-negotiable)

1. **Never touch the live store** (`~/.claude/data/`). Operate on a copy.
2. **Worktree** off current branch, feature branch `exp/floor-study`. Commit results. **Never merge, never push.**
3. **Cost ceiling: 0 metered $.** Everything is replay over existing logs + RRF-only retrieval with local fastembed. If anything tries to call a paid API, that's a bug — stop.
4. **Wall-clock cap: 4 hours.**
5. **DETERMINISTIC FLOOR ONLY.** Do not implement Thompson/temperature/stochastic gating. Those require importance-weighted off-policy evaluation (the logs were collected under the current pull-only policy) and are a separate interactive experiment. If you find yourself reaching for a stochastic policy, stop — that is out of scope by design.

### Setup

```sh
git -C ~/repos/somnigraph worktree add -b exp/floor-study ~/repos/somnigraph-floor HEAD
cd ~/repos/somnigraph-floor
RUN=$(git rev-parse --short HEAD)-$(printf '%(%s)T' -1)
S="$HOME/.somnigraph-exp/floor-$RUN"
mkdir -p "$(dirname "$S")"
cp -r "$HOME/.claude/data" "$S"
export SOMNIGRAPH_DATA_DIR="$S"
export SOMNIGRAPH_EMBEDDING_BACKEND=fastembed
```

### Step 1 — Reconstruct use/ignore labels

Write `build_floor_labels.py` (commit it). From `$S/memory.db` `memory_events`:
- Pull `recall_meta` events (each carries the surfaced candidate list, `cutoff_rank`, and now `limit`) and `feedback` events (each carries `memory_id` + `utility` in its JSON `context`).
- Join feedback to the recall_meta that produced it using a **60-second window** (this matches `select_real_pathology_targets.py`'s `DEFAULT_MATCH_WINDOW_SECONDS = 60`). Log the join statistics: how many feedback events matched a recall_meta, how many were orphaned, the window-collision rate.
- Per (historical turn, surfaced memory): **used = utility > 0**, else **ignored**. Memories surfaced but never rated are ignored (utility 0).
- For each surfaced candidate, record its **RRF-only score** (recompute if not stored — see Step 2). Emit `floor_labels.json`: list of `{turn_id, memory_id, rrf_score, used}`.

Sanity-check on a 20-row sample before the full extraction; log any schema surprise (column names, JSON shape) and adapt.

### Step 2 — RRF-only scores

The floor gates on RRF *only* (no reranker, no LLM — that's the cheap per-turn path the hint layer would run). For each surfaced candidate in `floor_labels.json`, obtain its RRF fusion score. If `recall_meta` already stored per-candidate RRF scores, use them. If it stored only reranked scores, recompute RRF for the logged query against the store copy using the same FTS+vec fusion the pipeline uses (`memory.scoring` RRF path, fastembed embeddings). Pin exactly which source you used in the findings — reranked-score contamination would invalidate the study.

### Step 3 — Floor sweep

Write `sweep_floor.py` (commit it):
- Sweep the floor over ~50 points spanning the observed RRF-score range.
- At each floor: **surface** = rrf_score ≥ floor. Compute precision, recall, F-beta (β=2) against the `used` labels, treating "surface a used memory" as a true positive.
- Baselines: **always-inject** (floor = −∞: recall 1.0, precision = base rate) and **never-inject** (floor = +∞).
- Emit `floor_sweep.json`: the full P/R/F-beta curve, the F-beta-optimal floor, and the two baselines.

### Step 4 — Calibration note (do not reuse the GT calibration)

If raw RRF scores look miscalibrated for this binary decision, you may **fit a calibration on the surface/skip labels themselves** (e.g. isotonic on `rrf_score → P(used)`), and re-sweep on the calibrated score. **Do NOT reuse the repo's GT isotonic/PAVA calibration** — that operates on LLM-judge relevance scores for tuning, not on live retrieval scores. Borrow the method, not the fitted curve. If you fit a calibration, report both raw-score and calibrated-score sweeps.

### Step 5 — Findings writeup (DRAFT — for human review, do not merge)

Write `findings-floor-study.md`:
- The verdict against the accept test: does any floor beat both baselines on F-beta(2)?
- The honest comparison to the cliff result: is the binary surface signal usably better than the fine-grained cutoff (R²<0), or as weak?
- Join statistics and the RRF-score provenance (Step 2) — these are the credibility of the whole study.
- Exact repro commands.
- **"For review" banner:** the F-beta operating point and the β weight encode a miss-vs-noise value judgment that is Alexis's call; whether a positive result graduates to the `UserPromptSubmit` hook build is a stewardship decision. This draft proposes, it does not decide.

Commit to `exp/floor-study`. Do not merge or push. Print the verdict + path to findings as your final message.

### Both outcomes are publishable

- **Positive:** a floor beats both baselines → validates proactive recall's core assumption before any hook code is written.
- **Negative:** the binary surface signal is as weak as the cliff cutoff (R²≈0) → itself a publishable finding that score-based gating fails at both granularities. Write it up with equal care.
