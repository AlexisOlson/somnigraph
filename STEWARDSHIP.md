# Stewardship

This document is the loop. Read it at session start, work from it, update it at session end. It gets better each time.

## What this project is

Somnigraph is a research artifact shared publicly because the ideas are interesting — not because anyone needs another memory server. The value lives in the documentation and the decisions behind them. A reader who understands `docs/architecture.md` and `docs/experiments.md` can build their own system; one who just clones the code gets a tool that works but without understanding why.

The audience is someone building their own memory system who wants to skip the wrong turns.

This means: the docs are the product. Code is the proof that the docs aren't theoretical. Issues about the approach are welcome. Support requests aren't the right use of anyone's time. PRs would need to clear a high bar — not elitism, but the code is tightly coupled to specific design decisions that the docs explain.

## Priorities

Ordered. Each has a reorder condition — the list expects to change.

### 1. Honest accounting

The "What didn't work" sections of `docs/architecture.md` and `docs/experiments.md` are the most valuable parts of the documentation. Extend this ethos everywhere: when something is removed or changed, document why. When an approach was tried and failed, say so. The negative results are what make this a research artifact rather than just another repo.

*Move this down when*: never. This is an invariant, not a work item — it applies to everything above and below it.

### 2. Retrieval quality: reranker iteration

LightGBM pointwise reranker is live in production, running from repo code via directory junction. The hand-tuned scoring formula is preserved as fallback but no longer used.

Current state: **31-feature production model (V5+3b)** trained on 1885 real-data queries with adversarial probing through V5+3 (real-recall pathology mining). Aggregate Reranker NDCG=0.8954 vs RRF +0.0921. Live transfer-learning gain over V5+2b: NDCG +0.0297 (0.7309→0.7606), R@10 +0.0655 (0.8218→0.8873) on unchanged live composition — clean evidence that adversarial training generalizes beyond the probe set. Held-out probe-hard NDCG=0.8785 R@10=0.9321 on 239 queries (flat vs V5+2b, within ±0.01 noise floor). Probe-hard miss rate 0/239 since the V5+2 GT cleanup that filtered inactive-memory feedback. session_recency feature importance is stable but composition-dependent (V5+1b 188 → V5+2b 445 → V5+3b 356 across consecutive retrains on changing GT mixes — partially answers the leakage-vs-signal question without a formal LOFO).

Adversarial probing infrastructure: real-recall pathology miner (`scripts/select_real_pathology_targets.py`) drives `probe_recall.py --mix` to bundle adversarial picks (high-utility memories the model buried in production) with coverage-fill picks. Audit-based selector kept as a fallback but no longer the primary path — synthetic audit pathologies were FTS-handicapped or Goodhart-correlated by construction (V5+2 group analysis). Worst-regression drill-down via `scripts/drill_query_scores.py` exposed that persistent live-NDCG regressions are sometimes GT noise rather than model bugs (V5+3 changelog detail).

Remaining: V5+5 (next adversarial pass) — re-mine real-recall pathologies once post-V5+3b live activity accumulates, or extend supply at a different rank threshold; live-GT re-rate scaffolding to flag suspicious worst-regressions at scale; formal session_recency LOFO audit (no longer load-bearing but would close the question); counterfactual coverage check; sleep impact measurement; documentation pass into `docs/experiments.md` covering the V3→V5+3 retraining arc, the pin-cap rewrite, the GT-cleanup result, and the adversarial-source pivot.

*Move this down when*: V5+3 retraining arc is documented in `docs/experiments.md` and the next-experiment direction (continued adversarial probing vs live-GT scaffolding vs LOFO) is settled.

### 3. Documentation quality *(maintenance)*

The docs are the product. A feature without documentation is half-finished; documentation without code is still valuable. The README's CLAUDE.md snippet is the front door — it determines whether people's first experience is good.

**Quality bar for the snippet**: A fresh Claude session using only the snippet should use the memory tools with reasonable judgment within 2-3 sessions. See `docs/claude-md-guide.md` for the depth behind the snippet.

This was Priority 1 during creation. Reorder condition met (2026-03-14): snippet tested via dogfood (5 gaps, all fixed), simulation (no new gaps), and fresh-session test (correct tool usage across all workflow steps). Repo went public 2026-03-16. Documentation now ships with code changes but is no longer the primary work item.

*Move this back up when*: real users surface snippet gaps, or a system change invalidates the current docs.

### 4. End-to-end LoCoMo QA benchmark

LoCoMo end-to-end QA pipeline is built and producing results. **85.1% overall accuracy** (Opus judge), beating Mem0 (66.88 J), Mem0g (68.44 J), and full-context baseline (72.90 J) on the same benchmark. See `docs/locomo-benchmark.md` for full results.

Current state: Level 5b graph-augmented 15-feature reranker with synthetic coverage scoring (Config K, distinct from 26-feature production model in P2), GPT-4.1-mini reader. **Level 5b expanded: OVERALL R@10=95.4%, MRR=0.882, R@20=96.9%. Multi-hop R@10=88.8%.** All metrics improved substantially over Level 4: MRR +0.169, R@1 +22.4pp, R@10 +6.7pp, multi-hop R@10 +13.5pp. BM25-damped IDF keyword expansion. Corrected GT vendored from locomo-audit (6.4% ceiling). 3 of 6 expansion methods dead (rocchio 0%, multi_query 2%, entity_focus 4%) -- ablation pending.

Key findings: Synthetic vocabulary bridges close the multi-hop vocabulary gap when allowed through Phase 2. L5 filtered synthetics before scoring, wasting their contribution and causing a 6.8pp multi-hop regression. L5b keeps synthetics in results, retrains the reranker with coverage-based labels, and scores using an LLM-judged coverage table (Sonnet, 87.5% agreement with Opus ground truth). Feature selection must match search configuration at eval time (initial Level 5 used features selected at 200-limit, produced 3.5pp regression; re-selecting at 4000-limit recovered it). R@10 and NDCG@10 select structurally different feature sets. See `docs/multihop-failure-analysis.md` for the vocabulary gap analysis that motivated L5b.

Known benchmark limitations: LoCoMo's LLM judge accepts 62.81% of intentionally vague wrong answers. Opus is 3.2pp stricter than GPT-4.1-mini as judge.

Remaining: expansion method ablation, sleep pass ablation, feedback loop ablation, document findings in experiments.md.

*Move this down when*: ablations are complete and findings documented. Move up if external interest in comparative numbers increases.

### 5. PERMA personalization benchmark

PERMA (arXiv:2603.23231, March 2026) is a fresh benchmark evaluating preference-state maintenance across event-driven, multi-session, multi-domain interactions. 10 synthetic users, 20 domains, 2,166 preferences, 1.8M tokens. Decoupled evaluation separates memory fidelity from task performance. See `research/sources/perma.md` for full analysis, `docs/roadmap.md` § PERMA for SOTA targets.

Current SOTA among memory systems: MemOS (MCQ 0.811, Turn=1 0.548 single-domain, 0.306 multi-domain). **Primary goal: multi-domain Turn=1 (currently 0.306)** — all benchmarked systems collapse on cross-domain synthesis, which is exactly the gap Somnigraph's graph-based retrieval is designed to fill. No system in the paper has a learned reranker, feedback loop, or graph-conditioned retrieval. We'd like to beat all metrics, but the multi-domain result is the headline claim.

Remaining: build ingestion pipeline for PERMA's event-driven dialogue format, run MCQ + interactive evaluation, analyze per-dimension results (temporal depth, noise robustness, cross-domain synthesis). Ablate graph contribution specifically on multi-domain tasks.

*Move this up when*: external interest in comparative numbers increases, or LoCoMo ablations complete and PERMA becomes the next benchmark. *Move this down when*: initial results show the benchmark's synthetic data doesn't exercise Somnigraph's strengths (preference tracking via graph is untested — the result could be negative).

## Decision framework

At the start of a session:

1. **Is something broken?** Fix it.
2. **Is this document out of date?** Update it first. Eating your own dogfood.
3. **Does the priority ordering feel wrong?** Propose a reorder with reasoning.
4. **Pick the highest-priority item** that can make meaningful progress in one session.
5. **Propose 1-3 concrete tasks**, explain why, get approval.

This is short by design. The decision framework is the priority list — this just says "look at it and pick something."

## Work patterns

- Feature branches, not direct commits to main
- Review diffs together before merging
- Documentation ships with code in the same branch (not separate PRs)
- The README's CLAUDE.md snippet should be tested by actually using it (the dogfood test)

## The CLAUDE.md snippet

The snippet in the README (the markdown block users paste into their `CLAUDE.md`) deserves special attention because it's the front door to the whole system. Three tiers of guidance exist:

- **Tier 1** (in README, ~30 lines): The essential rhythm. Enough to use the tools without being harmful.
- **Tier 2** (`docs/claude-md-guide.md`, ~100 lines): The judgment layer. Token budgets, when to recall vs. not, category taxonomy, feedback intent, common failure modes.
- **Tier 3** (`docs/architecture.md`): Understanding the system well enough to calibrate your own usage.

The README contains Tier 1 and links to Tier 2. The tiers form a depth gradient — each level adds understanding without requiring the next.

## Retrospective

After finishing work, before closing the session. Three questions — answer all three, even briefly.

### 1. What surprised you about the work?

If nothing, say so — but "nothing surprised me" in a young repo is itself worth examining.

### 2. Is the priority ordering still right?

Look at the list above. Would you reorder anything? Does a priority need rewording? Has a reorder condition been met?

### 3. What does this document need to say that it currently doesn't?

Gaps, missing context, things a future session would stumble on.

### Output

Either:
- **(a)** A concrete diff to this document with reasoning, or
- **(b)** "No changes — [specific reason the document held up]"

Option (b) is fine, but it requires stating *why*. "Everything went well" with no specifics is the equivalent of rating every memory 0.5 — it conveys no signal.

Append a changelog entry below (see format below), and create a per-session file in `docs/sessions/`.

**Changelog entry format (40-80 words):** What shipped, the headline metric or key finding, and a link to the per-session file. Example:

> - 2026-05-09: V5+3 — first real-recall adversarial probe + retrain. Aggregate NDCG +0.0138 (→0.8954) and live transfer-learning gain +0.0297 NDCG / +6.55pp R@10 on unchanged composition. session_recency feature importance dropped 445→356, partially answering Phase 3 priority C. See [`docs/sessions/2026-05-09-v5-3-real-recall-adversarial.md`](docs/sessions/2026-05-09-v5-3-real-recall-adversarial.md).

**Per-session file** (`docs/sessions/YYYY-MM-DD-slug.md`): All the detail — surprises, caveats, files touched, reversibility, what's next. Preserve everything; this is the institutional knowledge. Lightly clean up prose (use `###` subheadings for **Surprises:**/**Caveats:**/**Files touched:** etc.) but do not change facts or numbers.

### Anti-patterns

- "Everything went well" (empty signal)
- Rewriting the whole document (this produces diffs, not rewrites)
- Adding aspirational goals not grounded in session experience
- Skipping the retrospective because "nothing changed" (the assessment *is* the value)
- Writing the full retrospective inline in the changelog (it belongs in `docs/sessions/`)

## Changelog

Recent entries. When this section grows past ~10-15 entries, migrate the oldest to [`docs/stewardship-history.md`](docs/stewardship-history.md). Each entry is a 40-80 word summary + link; full detail (surprises, caveats, files touched, reversibility, what's next) lives in the linked per-session file in `docs/sessions/`.

Earlier entries: see [`docs/stewardship-history.md`](docs/stewardship-history.md).

- 2026-06-28: Phase 18 source sweep — added 7 source analyses (TrueMemory, ByteRover, knowledge-worker, ai-memory-comparison, Recall, agentmemory, MIRIX) from r/AIMemory + the carsteneu leaderboard. Headline: write-path quality, not retrieval, is what the LoCoMo/LME leaders win on (ByteRover BM25-only, MemPalace verbatim, agentmemory write-time grounding) — independent corroboration of the Phase 15 AMemGym finding. Corrected carsteneu errors (agentmemory 96.2% not 95.2%, no RRF; MIRIX 85.38 is gpt-4.1-mini-judged ≈82% Opus-equiv). Recommends write-path discipline over a three-axis confidence build. Also: `sleep.py` defensive fix for malformed REM taxonomy items. Branch also carries the parallel session's practitioner-signal docs. See [`docs/sessions/2026-06-28-phase18-source-sweep.md`](docs/sessions/2026-06-28-phase18-source-sweep.md).
- 2026-05-09: V5+3 — first real-recall adversarial probe + retrain. Verdict: experiment supported. Aggregate NDCG +0.0138 (→0.8954) and live transfer-learning gain +0.0297 NDCG / +6.55pp R@10 on unchanged live composition — clean evidence that adversarial signal generalizes beyond the probe set. session_recency feature importance dropped 445→356, partially answering Phase 3 priority C without a formal LOFO. Worst-regression drill-down revealed persistent live regressions are often GT noise, not model bugs. See [`docs/sessions/2026-05-09-v5-3-real-recall-adversarial.md`](docs/sessions/2026-05-09-v5-3-real-recall-adversarial.md).
- 2026-05-09: V5+2 — GT cleanup + retrain; adversarial redirect (audit → real-recall); PPR cache ~100x speedup. Found 5.9% of GT feedback rows referenced inactive memories; filtering drove held-out miss rate 6.5% → 0.0%, resolving Phase 3 priority B entirely. Built real-recall pathology miner (`select_real_pathology_targets.py`). session_recency importance jumped 2.4x post-cleanup. See [`docs/sessions/2026-05-09-v5-2-gt-cleanup-adversarial-pivot.md`](docs/sessions/2026-05-09-v5-2-gt-cleanup-adversarial-pivot.md).
- 2026-05-09: V5+1 — 200-query mixed probe + retrain; per-memory pin cap replaces probe/real ratio cap; adversarial scaffolding wired; non-miss NDCG metric added. Aggregate NDCG +0.0072 over V5b (→0.8763). Headline held-out hard NDCG dropped but non-miss NDCG improved +0.0133 — composition shift from Phase-1 misses, not regression. See [`docs/sessions/2026-05-09-v5-1-mixed-probe-pin-cap-adversarial-scaffolding.md`](docs/sessions/2026-05-09-v5-1-mixed-probe-pin-cap-adversarial-scaffolding.md).
- 2026-05-09: V3 boost sweep locked DEFAULT_PINNED_BOOST=5.0; V4 8-query smoke validated bundled craft end-to-end; V5 retrain deployed. Plus db.py backend-mismatch guard added (fastembed 384d vs openai 1536d now hard-fails at connection time rather than silently writing wrong-dim vectors). See [`docs/sessions/2026-05-09-v3-boost-sweep-v4-smoke-v5-retrain.md`](docs/sessions/2026-05-09-v3-boost-sweep-v4-smoke-v5-retrain.md).
