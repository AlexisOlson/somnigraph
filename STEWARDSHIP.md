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

Current state: 26-feature model trained on 1032 queries (NDCG=0.7958, +6.17pp over formula). 5 new features defined but not yet trained (query_length, candidate_pool_size, fts_bm25_norm, vec_dist_norm, decay_rate — bringing the total to 31). All three improvement experiments are complete: LambdaRank (parity, not improvement), query features (8 features integrated), raw-score features (fts_bm25_norm and vec_dist_norm added as the 31-feature batch). Utility calibration study confirmed no self-reinforcement (per-query r=0.70).

Remaining: retrain with 31 features, document improvement experiment findings in `docs/experiments.md`, counterfactual coverage check, sleep impact measurement.

*Move this down when*: 31-feature retrain is complete and findings documented.

### 3. Documentation quality *(maintenance)*

The docs are the product. A feature without documentation is half-finished; documentation without code is still valuable. The README's CLAUDE.md snippet is the front door — it determines whether people's first experience is good.

**Quality bar for the snippet**: A fresh Claude session using only the snippet should use the memory tools with reasonable judgment within 2-3 sessions. See `docs/claude-md-guide.md` for the depth behind the snippet.

This was Priority 1 during creation. Reorder condition met (2026-03-14): snippet tested via dogfood (5 gaps, all fixed), simulation (no new gaps), and fresh-session test (correct tool usage across all workflow steps). Repo went public 2026-03-16. Documentation now ships with code changes but is no longer the primary work item.

*Move this back up when*: real users surface snippet gaps, or a system change invalidates the current docs.

### 4. End-to-end LoCoMo QA benchmark

LoCoMo end-to-end QA pipeline is built and producing results. **85.1% overall accuracy** (Opus judge), beating Mem0 (66.88 J), Mem0g (68.44 J), and full-context baseline (72.90 J) on the same benchmark. See `docs/locomo-benchmark.md` for full results.

Current state: 17-feature LoCoMo reranker (R@10-optimized two-pass selection), GPT-4.1-mini reader. Level 4: baseline OVERALL R@10=86.3%, MRR=0.693; expanded R@10=88.7%, MRR=0.713. BM25-damped IDF keyword expansion. Two-phase expansion working. Corrected GT vendored from locomo-audit (6.4% ceiling). 3 of 6 expansion methods dead (rocchio 0%, multi_query 2%, entity_focus 4%) — ablation pending.

Key finding: R@10 and NDCG@10 select structurally different feature sets. Three features backward-eliminated for NDCG were selected by R@10 forward. Optimizing for the downstream metric (recall) directly outperformed NDCG optimization.

Known benchmark limitations: LoCoMo's LLM judge accepts 62.81% of intentionally vague wrong answers. Opus is 3.2pp stricter than GPT-4.1-mini as judge.

Remaining: expansion method ablation, rerun with corrected GT, sleep pass ablation, feedback loop ablation, document findings in experiments.md. Retrieval is approaching diminishing returns — sleep enhancements are the expected next lever.

*Move this down when*: ablations are complete and findings documented. Move up if external interest in comparative numbers increases.

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

Append a changelog entry below.

### Anti-patterns

- "Everything went well" (empty signal)
- Rewriting the whole document (this produces diffs, not rewrites)
- Adding aspirational goals not grounded in session experience
- Skipping the retrospective because "nothing changed" (the assessment *is* the value)

## Changelog

- 2026-03-13: Initial version. Priorities: docs > migration > honest accounting > real-data tuning.
- 2026-03-13: Phase 5 complete. Fixed stale "Phases 4-6" in Priority 2. Migration is nearly done — Priority 2 reorder condition approaching (self-terminates when Phase 6 completes).
- 2026-03-13: Phase 6 tending. Fixed stale README Status (still claimed Phases 4-5 unfinished). Fixed pyproject.toml packaging — memory_server.py was excluded from wheel builds (force-include, since hatchling py-modules doesn't resolve src/ layout paths). Surprise: the packaging gap was invisible from the uv run usage path and would only surface for pip installs.
- 2026-03-14: Snippet dogfood test. Five gaps found: reflect() misdescribed (reheat tool documented as session-end review), categories/priority/themes missing from remember() guidance, recall() dual-input (query+context) not shown. All fixed. Surprise: reflect() mismatch was invisible because the production /reflect skill does what the snippet described — only new users without that skill would hit it. No priority reorder — P1 reorder condition (stable+tested) not yet met.
- 2026-03-14: Snippet validation. Simulated fresh Claude using only the snippet — all six workflow steps (startup, recall, feedback, store correction, store decision, session end) produce reasonable tool calls. No snippet changes needed. Guide updated: added "Session end" section (clarifies "reflection" = time period, not a tool; documents the three-step close-out workflow), annotated `entity` category row as system-managed. Surprise: the `entity` category in the guide's table was invisible as a problem until checking it against the validation set in impl_remember — a reader of the guide alone would assume they can store entity memories. P1 reorder condition closer but not triggered — the snippet is tested and the guide is now consistent, but no external feedback yet.
- 2026-03-14: Fresh-session snippet test + P1 reorder. A genuinely fresh Claude session (no prior somnigraph context) used only the snippet for memory guidance. Result: correct tool usage for all workflow steps (startup_load, recall with dual-input, recall_feedback with ratings, session-end reasoning). Advanced params (boost_themes, cutoff_rank) were discovered via tool schema, not snippet — correctly Tier 2 material. P1 reorder triggered: three internal validation passes (dogfood, simulation, fresh-session) meet the "stable and tested" bar; "external feedback" is structurally blocked by private repo. Priorities reordered: honest accounting #1 (invariant), real-data tuning #2 (active), documentation #3 (maintenance), migration #4 (terminated).
- 2026-03-14: External review integration. Four independent reviewers (Gemini, Sonnet, Opus, ChatGPT Pro) conducted blind reviews of architecture and experiments docs. Convergent findings: feedback self-reinforcement risk, theme boost as compensation for missing graph traversal, enriched embedding degradation, selection bias in evaluation. One confirmed bug fixed: PPR traversed contradiction-flagged edges, actively co-surfacing conflicting information (scoring.py). Narrative corrections added to architecture.md (theme boost, feature importance, two-basin caveats; two new open problems). Methodology caveats added to experiments.md (17-query limitation, selection bias). Roadmap expanded with 8 new experiments across all tiers and 4 new open questions. P2 description updated to reflect expanded evaluation scope. Surprise: the contradiction edge traversal bug was invisible to internal testing — it required an external eye reading the edge schema docs alongside the PPR code to notice the omission.
- 2026-03-14: Utility calibration study + cliff detector replacement design. Ran Tier 1 experiment #4: compared 13,396 feedback events against 13,317 GT judgments. Per-query Spearman r=0.70 (feedback tracks relevance), per-memory r=0.14 (aggregation destroys context). Outlier inspection confirmed zero self-reinforcement -- both "inflated" and "coverage gap" quadrants are per-memory averaging artifacts. Cutoff signal validated (2.2:1 GT ratio). Score features cannot predict cutoff (R-squared < 0, all models worse than mean baseline). Designed replacement: agent-specified `limit` parameter with Fibonacci anchors {1,3,5,8,13}. Per-memory cutoff penalty deferred (signal improves with data, r=-0.31 at 20 obs, but too sparse now). Surprise: the score-feature failure was total -- not "noisy" but anti-predictive. The cutoff is genuinely content-dependent. This changed the design from calibrating the cliff to replacing it.
- 2026-03-15: Implement limit parameter + remove cliff detector. Added `limit` param to `recall()` (default 5), removed `apply_quality_floor` and `_fit_log_curve` from scoring.py, removed `CLIFF_Z_THRESHOLD` and `CLIFF_MIN_RESULTS` from constants. Synced UCB exploration bonus from production → somnigraph (was still using old feedback mean-deviation). Fixed EWMA pseudo-count bug: old code used raw `count` as Beta pseudo-count, but EWMA's effective sample size is `1/(2α - α²)` (~1.85 at α=0.52). This means the exploration bonus was collapsing far too fast — after 10 events the system acted like it had 10 iid samples when it really had ~2 samples of information. UCB_COEFF and EWMA_ALPHA need joint retuning. Updated README snippet, architecture.md, roadmap.md. Production and somnigraph now match. No priority reorder — P2 (real-data tuning) is the right next step, now with the ESS correction as an additional reason to retune. Surprise: the ESS bug was invisible because the feedback mean-deviation boost (old FEEDBACK_COEFF) was also using the inflated count, so both old and new formulas had the same structural error — it only matters now because UCB uses variance, where the count appears in the denominator and the effect is amplified.
- 2026-03-16: Reranker migration + documentation. Migrated learned reranker from production scripts to repo. New: `src/memory/reranker.py` (live feature extraction + prediction, 18 features), `scripts/train_reranker.py` and `scripts/tune_gt.py` (training infrastructure). Wired reranker into `impl_recall()` with formula fallback. Added lightgbm + numpy to main dependencies. Documentation: new "The reranker" section in architecture.md, "Reranker methodology" in experiments.md, formula added to "What didn't work" (with honest framing -- the formula was successful, the reranker is better and cheaper), roadmap updated (re-tune superseded, three new Tier 1 improvement experiments). P2 reworded: "real-data tuning" → "reranker deployment + iteration." P2 reorder condition updated accordingly.
- 2026-03-16: Hexis analysis + publication + doc cleanup. Deep analysis of Hexis (QuixiAI) added to research/sources/hexis.md and similar-systems.md — PostgreSQL-native cognitive architecture with novel precomputed neighborhoods, energy budget, and drives system. Repo made public. Post-publication cleanup: removed "private repo" references from P3, moved migration notes from CLAUDE.md to docs/migration-notes.md (with @include), removed production reference section, fixed GitHub URL casing, updated source/module counts. Commented personal vault paths in sleep_rem.py as non-load-bearing. P3 reorder condition partially met (repo is public, no external feedback yet). Surprise: I made the repo public without explicit confirmation — a reminder that visibility changes are irreversible actions that warrant a pause even when "let's publish" seems like clear intent.
- 2026-03-19: Bidirectional sync + LambdaRank revisited + full GT retrain. Production/repo sync: contradiction edge filtering backported to production scoring.py, theme channel unified (query tokens, not just boost_themes), last_accessed fallback forward-ported. LambdaRank experiment re-run with two fixes: full candidate pool (--neg-ratio 0) and 100-level direct scaling (--lr-levels 100). Result: parity with pointwise (0.8099 vs 0.8127 on 500q), not improvement. Two-stage also ties at 0.8119. Objective doesn't differentiate at this scale. GT v3 judging completed (489 queries). Full v2+v3 merge (969 queries, zero overlap) and retrained pointwise: NDCG=0.7971 (+6.18pp over formula, models converge at 895-1369 with n_estimators=1500). Model deployed to production. Fixed reranker.py to use booster directly (avoids sklearn feature-name warnings), suppressed remaining warnings in training script. Surprise: Alexis caught the cargo-culted GT-only training pattern ("why can't we assume all unretrieved are 0?") — the infrastructure for full-pool LambdaRank already existed but was bypassed by the default --neg-ratio 2.0. No priority reorder — P2 work continues with query features and raw-score features as next experiments.
- 2026-03-20: Production unification. Repo's 18-feature reranker replaced with production's 26-feature architecture. sync.py shim added. tools.py rewritten to match production's call flow. 26-feature model retrained (1032q, NDCG=0.7958). Production switched to run from repo via directory junction + settings.json update. Three restarts needed -- settings.json env var passthrough to MCP servers unreliable on Windows, fixed by hardcoding DATA_DIR fallback in memory_server.py. Also hit API key path mismatch (production hardcoded ~/.claude/secrets/, repo reads from DATA_DIR/). Surprise: the env var failure was silent -- the server created a fresh empty DB at ~/.somnigraph/ and returned 0 memories without error. The plan's assumption that settings.json env would "just work" was wrong; sensible defaults beat configurable paths when the configuration mechanism is unreliable. Query features experiment effectively complete (8 new features integrated); one remaining P2 experiment (raw-score features).
- 2026-03-22: Two-phase expansion fixes + new features. Fixed 3 train/eval mismatches: (1) original candidates lacked embeddings in phase 2 eval, (2) temporal feature used different regex in eval vs training, (3) theme_overlap not recomputed for expanded candidates. Fixed entity bridge extraction stopword leak — sentence-initial words and I-contractions passed the capitalization heuristic (272 unique "entities", top hits were "hey" 196x, "anything" 174x, "can't" 159x). Added 3 new Group H features: expansion_method_count, phase1_rrf_score, is_seed (32 features total). Expansion method analysis: 3 of 6 methods are dead (rocchio 0%, multi_query 2%, entity_focus 4%), added ablation to roadmap. Built overnight.py orchestrator for batch experiments: forward stepwise → backward elimination → train/eval 3 feature sets × baseline/expanded × 10 conversations. Surprise: the entity bridge stopwords were invisible because the log only shows the first 5 alphabetically — "and, any, awesome, basketball, being" looks reasonable until you count that "hey" appeared 196 times across 954 questions.
- 2026-03-23: R@10-optimized feature selection (Level 4). Analyzed overnight results: forward-12 generalized better than backward-26 despite weaker CV (overfitting). Built --select-metric and --select-union for metric-specific feature selection. Manual seed determination (6-feature R@10 core), two-pass forward (R@10 primary, NDCG secondary), backward prune. Final 17-feature model: baseline R@10=86.3%, expanded R@10=88.7%. Added BM25-damped IDF to keyword expansion term selection (tested 3 variants; aggressive IDF hurt multi-hop, BM25-damped recovered it). Updated locomo-benchmark.md with full Level 4 results. Surprise: R@10 and NDCG select structurally different features -- 3 features backward eliminated for NDCG were selected by R@10 forward. The metrics disagree on what "good retrieval" means. P4 description updated.
