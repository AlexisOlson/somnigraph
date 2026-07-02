# Handoff: Sleep Counterfactual Fork — autonomous session prompt

**Shape:** single long grind → hand this whole file to a fresh autonomous Claude session (Opus, per the arc). It runs unsupervised in a git worktree and produces JSON metrics + a draft findings file. It does **not** merge, push, or touch the live store.

**Catalog entry:** [`../autonomous-experiments.md`](../autonomous-experiments.md) #1. **Roadmap:** #14 (counterfactual sleep eval) + #3 (sleep impact measurement). **Question:** *Does sleep improve retrieval?* — the central open question in `architecture.md` § Open problems.

---

## SESSION PROMPT (paste from here down)

You are running an unsupervised overnight experiment on Somnigraph. Follow the safety rails exactly. Do not deviate to "improve" the experiment; if something is ambiguous, log it and continue with the conservative choice. You will NOT be able to ask questions mid-run.

### Objective

Measure the causal effect of one full sleep cycle on retrieval quality, by comparing a **slept copy** of the memory store against a **frozen copy** on the identical GT query set.

Primary metric: **NDCG@5k, R@10, MRR** on the production GT, paired per query, with a bootstrap 95% CI on the delta.

### Hard safety rails (non-negotiable)

1. **Never touch the live store.** The live store is `~/.claude/data/`. You operate only on copies.
2. **Work in a git worktree** off the current branch, feature branch `exp/sleep-fork`. Commit results there. **Never merge, never push.**
3. **Cost ceiling:** sleep's LLM calls use the `claude` CLI = subscription. Embeddings follow the store's backend: $0 on a fastembed store (work machine); on the home OpenAI-1536d store, eval query embeds + sleep-generated summary embeds cost cents — **budget ceiling $2**, stop if projected above. If any step tries to call a metered OpenAI *reader/judge*, stop — that's a bug in this design. **Wall-clock cap: 12 hours.** If sleep hasn't finished by then, kill it, checkpoint, and write up the partial.
4. **Machine-readable output + findings markdown**, with exact repro commands.

### Setup

```sh
# 1. Worktree (isolated; new files only)
git -C ~/repos/somnigraph worktree add -b exp/sleep-fork ~/repos/somnigraph-sleepfork HEAD
cd ~/repos/somnigraph-sleepfork

# 2. Two independent copies of the LIVE store
RUN=$(git rev-parse --short HEAD)-$(printf '%(%s)T' -1)
A="$HOME/.somnigraph-exp/sleepfork-$RUN/A-frozen"
B="$HOME/.somnigraph-exp/sleepfork-$RUN/B-slept"
mkdir -p "$(dirname "$A")"
cp -r "$HOME/.claude/data" "$A"      # includes memory.db + tuning_studies/ (GT + reranker .txt)
cp -r "$HOME/.claude/data" "$B"

# 3. Backend rails — match the store you copied; db.py hard-fails on dim mismatch.
#    HOME machine: the store is OpenAI 1536d — leave SOMNIGRAPH_EMBEDDING_BACKEND
#    unset (defaults to openai; key file is inside the copied data dir).
#    WORK machine: the store is fastembed 384d — export SOMNIGRAPH_EMBEDDING_BACKEND=fastembed
# Verify before proceeding: a float[1536] memory_vec schema means OpenAI backend.
```

Before anything else, **verify the learned reranker is loadable on both copies**. The loader (`reranker.py`) needs `tuning_studies/reranker_model.txt` + `reranker_features.json`. A restore session (stewardship arc step 2.5, 2026-07-01) retrains and deploys these to the live store — after it lands, your copies inherit them and this check just passes. If you skip this check and the files are absent, `--configs reranker` silently measures the **formula**, not V5+3b, and the whole experiment is about the wrong scorer.

```sh
ls -la "$A/tuning_studies/"reranker_model.txt "$A/tuning_studies/"reranker_features.json 2>/dev/null
```

If `reranker_model.txt` is absent: **HARD STOP — do NOT convert the on-disk `reranker_model.pkl` to `.txt`.** That pkl is the stale 2026-03-20 **26-feature** model; the live extractor builds **31 features**, so a converted booster is feature-misaligned with every feature matrix this eval would hand it. (An earlier version of this artifact contained a pkl→txt converter snippet — it was wrong and has been removed.) Your two valid options: (a) report that the restore hasn't landed yet and stop, or (b) run the eval anyway and **record loudly in the findings that both A and B used formula scoring** — the delta is then a valid sleep-vs-formula measurement, just not a sleep-vs-reranker one. Never proceed as if the reranker ran when it didn't.

### Step 1 — Baseline on the frozen copy (A)

```sh
SOMNIGRAPH_DATA_DIR="$A" uv run scripts/locomo_bench/eval_retrieval.py \
  --dataset production --configs reranker \
  --gt-path "$A/tuning_studies/gt_calibrated.json" \
  --output "$A/eval_baseline.jsonl"
```
If `gt_calibrated.json` is absent, fall back to `$A/ground_truth.json` and **log which GT file was used** (they may differ; the whole comparison must use the same GT file for A and B). Record per-query NDCG/R@10/MRR.

### Step 2 — Sleep the other copy (B)

```sh
SOMNIGRAPH_DATA_DIR="$B" \
  uv run scripts/sleep.py 2>&1 | tee "$B/sleep.log"
```
Sleep shells out to the `claude` CLI and mutates `$B` only. Capture from `sleep.log` and/or the DB: **how many memories were archived, merged, edited, and how many edges/summaries were created.** These counts are needed to attribute the delta. If sleep stalls (auth/rate-limit) or exceeds the wall-clock cap, kill it and note where it stopped.

### Step 3 — Re-eval the slept copy (B), identical config + GT

```sh
SOMNIGRAPH_DATA_DIR="$B" uv run scripts/locomo_bench/eval_retrieval.py \
  --dataset production --configs reranker \
  --gt-path "$A/tuning_studies/gt_calibrated.json" \
  --output "$B/eval_slept.jsonl"
```
Use the **same GT file** as Step 1 (point `--gt-path` at A's copy so the label set is byte-identical). GT labels are per-(query, memory); sleep-created memories not in GT correctly score 0 — this is the fair, conservative scoring. Do not "fix" this by re-judging; re-judging is a separate (interactive) experiment.

### Step 4 — Paired delta + attribution

Write `analyze_sleep_fork.py` (in the worktree, commit it) that:
- Joins baseline vs slept per query, computes paired deltas for NDCG@5k / R@10 / MRR.
- Bootstrap 95% CI on the mean delta (10k resamples, seed the RNG to a fixed constant for reproducibility).
- Splits the query set by category if the GT carries type tags (factual/thematic/temporal/personal), per the roadmap hypothesis that REM summaries may hurt temporal queries while NREM edges help.
- Emits `sleep_fork_results.json` with: baseline metrics, slept metrics, deltas, CIs, sleep action counts, GT file used, whether the reranker loaded.

### Step 5 — Optional robustness (only if wall-clock allows)

Sleep's LLM calls are nondeterministic. If time remains under the 12h cap, repeat Steps 2–4 with **up to 2 more** slept copies (B2, B3) to bound sleep's own variance. Cap at 3 total. If time is short, skip — one clean fork is the deliverable.

### Step 6 — Findings writeup (DRAFT — for human review, do not merge)

Write `findings-sleep-fork.md`:
- Headline delta with CI, and the honest sign (improves / hurts / null-within-noise).
- The NREM-vs-REM attribution: does the edge-driven adjacency expansion help while summary generation competes for budget?
- Sleep action counts and any divergence weirdness.
- Exact repro commands (the block above).
- **A "for review" banner at the top:** this is an autonomously-produced draft; the interpretation is a claim for `experiments.md`/`architecture.md` and must be ratified by a human before merge (honest accounting).

Commit everything to `exp/sleep-fork`. Do not merge. Do not push. Print the headline delta and the path to `findings-sleep-fork.md` as your final message.

### What a good null result looks like

If the delta CI straddles zero: that is a **publishable finding**, not a failure. "Consolidation runs, produces reasonable-looking edges and summaries, and does not move retrieval metrics on the current corpus" resolves an open question and belongs in the docs with the same care as a positive result. Write it up fully.
