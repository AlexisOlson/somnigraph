# 2026-05-09 — V5+3 — first real-recall adversarial probe + retrain

**Verdict: experiment supported.** Aggregate Reranker NDCG climbed 0.8816 → **0.8954 (+0.0138)** on a 1885-query GT; vs RRF +0.0921 (was +0.0885). Live NDCG **+0.0297** (0.7309 → 0.7606), live R@10 **+0.0655** (0.8218 → 0.8873) — and live didn't change in size (540 vs 554, near-flat). **Adversarial probes on memories the model previously buried produced clean transfer-learning gains on live queries** (which were unchanged in composition). This is the V5+3 hypothesis confirmed: real-recall adversarial signal generalizes, doesn't just memorize the probe set.

## V5+3 probe

Command: `--mode mixed --queries 200 --coverage 0.5 --mix 0.5 --adversarial-source real --adversarial-rank-threshold 3`

25 adversarial groups (real-recall pathologies, util>=0.7 + rank>=3 + active, 40 candidates available, 25 selected via Efraimidis-Spirakis weighted by utility*worst_rank softened by 1/(1+pins)) + 25 coverage groups = 50 groups × 4 = 200 queries. Bundled craft 50/50 succeeded, 0 fallbacks, exact distribution {natural: 50, mild: 100, hard: 50}. Target row LLM rating distribution: median=0.95, p10=0.95, p90=0.95, **3/186 below 0.5** (V5+1 was 0/193) — slight uptick from adversarial creativity-drift, fine at 1.6%. **Phase-1 misses 9/200 (4.5%)** vs V5+1's 6/200 (3%) — expected: adversarial probes target memories with weaker candidate-pool surfacing.

Spot-check Group 50 (coverage, JRPO II RDL ProgramDescription ENTERDATA): target ranks 1/1/1/2 across nat/mild#1/mild#2/hard, mild orthogonality biting (storage-mechanism vs edit-workflow-consequence, not paraphrases).

## V5+3a re-emit

1694 → 1885 queries (+200 V5+3 - 27 sparse-skipped + churn). Held-out probe-hard 189 → 239 (exact V5+3 contribution). Pinned targets 1140 → 1345 (+205, ~match). 64 net-new (q, mid) pairs from V5+3's 200 events (most folded into existing labels via dedup).

**One mild surprise: 814 inactive-memory feedback rows skipped** — predicted "near zero" since V5+2a cleaned the corpus. The filter operates row-level not query-level, so rows scatter; only 27 queries actually dropped to sparse. Net effect benign — query count moved as predicted.

## V5+3b retrain

Boost=5.0, 31 features, on cleaned 1885-query GT. Held-out probe-hard NDCG=0.8785 R@10=0.9321 on 239 queries (V5+2b: 0.8814/0.9447 on 189 queries) — **flat within ±0.01 noise floor; composition shifted (V5+3 added 50 new hard probes) but neither signal collapsed nor improved on this metric**. Held-out hard miss rate stayed at **0/239 (0.0%)** despite probe-time having 9 misses — the trainer's candidate pool is wider than live retrieval's top-15, so probe-time misses aren't always train-time misses. Per-mode synthetic 0.9985 (held out from training, perfect ceiling), probe-mild 0.9348, probe-natural 0.9465 — all healthy. Mean RMSE 0.0255 (V5+2b: 0.0265). Per-fold NDCG 0.8817-0.9018, very stable.

## Feature importance shift

**The major surprise: session_recency feature importance dropped 445.4 → 355.8 (-89.6).** V5+1b→V5+2b had *raised* it 2.4x (188 → 445.4 from the cleanup); V5+2b→V5+3b *brought it back down* by 20%. The importance landscape rebalanced: fts_bm25 (265.6), query_coverage (261.0), fts_bm25_norm (254.6), theme_rank (254.2), vec_rank (250.2) all in the same band as #1, instead of the prior single-feature dominance.

**This partially answers Phase 3 priority C (session_recency signal-vs-bias) without running a LOFO audit.** session_recency's V5+2b dominance was *partly* a function of training-data composition: when probes target memories *outside* the high-session-recency band (which adversarial picks do by definition — buried memories aren't recently-touched), the model can't lean on session_recency for those rows and has to learn the channel signals. LOFO is still worth doing eventually but is no longer the most pressing Phase 3 question.

## Audit-vs-real-recall divergence confirmed

Post-V5+3b audit (`audit_reranker_pathology.py --output ""` against the new model, summary-as-query, gap>10): 6 pathologies flagged total, ALL semantic memories, all are consolidation pillars ("Investor Reporting & RDL Tooling", "DAX Patterns & Performance", "RDL → Typst Translation", "MCP & API Integrations", "Scripts & Utilities", "Claude Code Configuration"), gaps 21-299. **Zero overlap with the 25 V5+3 adversarial targets.** Pillars get buried by the reranker because their summaries match-all-keywords on FTS but at real-recall time you want the specific procedural memory, not the pillar that summarizes the whole domain — so the burying is *correct behavior*, not a bug. The audit's view of "pathology" and the real-recall view measure structurally different things; this confirms the V5+2 group analysis at a fresh model state.

## Worst-regression drill-down (V5+3b interpretation lesson)

Built `scripts/drill_query_scores.py` (~110 lines, NEW) — loads the trained booster + features pickle, predicts scores for every (query, candidate) row of a target query, prints top-N predictions alongside GT labels and memory summaries.

Ran against the two persistent live regressions:

**(a) "claude.ai capacity two-tier KB chat attachments"** (R=0.2159, getting worse across V5+1b→V5+2b→V5+3b: -0.25 → -0.25 → -0.37). Score range tight: [-0.0155, 0.1344], top-1=0.13. **Top-2 are BOTH positives** (gt=0.10 and gt=0.40); the regression is just that production ordered the gt=0.40 first while the reranker put the gt=0.10 first.

**(b) "permissions settings.json allow deny Bash composite commands"** (R=0.0000 P=0.1657, NEW worst entry). Top-3 ranked memories are obviously relevant on inspection ("Bash sensitive-file check overrides command permissions", "No compound bash commands — use git -C", "Use git -C <path> to avoid composite commands"), but **the GT labels them all 0.00**. The "true positive" sitting at rank 24 (`PreCompact/SessionEnd hooks`, gt=0.10) isn't really about composite bash commands.

**Root cause for both: it's not the model, it's the GT.** The reranker is identifying the correct memories at top-N. The "regression" is a measurement artifact. **The implication is broader than these two queries: live worst-regression lists are not always model-quality diagnostics.** Live GT on short keyword-bag queries (median 8 tokens) has structural noise that probe GT doesn't. Cross-time live NDCG comparisons remain trustworthy in aggregate; reading individual worst-regression entries as bugs is not.

## Surprises

(1) The live transfer-learning gain is the headline result, not held-out hard. The held-out hard NDCG basically didn't move (-0.0029, within noise), but the *out-of-distribution* live queries jumped substantially. This is direct evidence that adversarial training generalizes.

(2) session_recency feature importance going DOWN after V5+3 is the inverse of what V5+2's cleanup did. Two opposite-direction shifts in two retrains on the same feature suggests its importance is more about training-data composition than fundamental model-architecture properties — that's actually the answer to "is this leakage?": probably not pure leakage, but heavily composition-dependent.

(3) The audit's 6-pathology output being *all consolidation pillars* is structurally informative — it says "the audit's notion of pathology overlaps with consolidation outputs by design."

(4) Worst-regression queries can be GT noise rather than model bugs. The "claude.ai capacity" query has been worsening across three retrains and inspection shows the reranker has the right answer in top-2 the whole time.

## Caveats

Aggregate NDCG +0.0138 includes the new V5+3 adversarial probes themselves in the GT — some of the gain is "model learned the V5+3 labels," not pure transfer. The clean transfer signal is the live improvement (+0.0297 NDCG, +6.5pp R@10) since live didn't change in composition. Held-out hard's flatness is the cleanest cross-time signal at this stage and it's flat. session_recency's drop to 355.8 doesn't mean "session_recency is now safe" — it means it's *less dominant* on this GT composition. A different GT composition would shift it again.

## Reversibility

Trainer-side, the V5+1b model is preserved at `~/.claude/data/tuning_studies/reranker_model.pkl` snapshots if rolled (currently overwritten — no automatic snapshot policy, worth fixing later). Live MCP runs against `reranker_model.txt` (now V5+3b). To revert, regenerate from V5+1b's GT through the patched trainer. The V5+3 probe events are append-only in `memory_events`; can be filtered out at GT-emit time via `--no-probe-targets` if a V5+3-free retrain is needed.

## Files touched

`scripts/drill_query_scores.py` (NEW, ~110 lines, top-N reranker score diagnostic). No other code changes — V5+3 ran on the V5+2-session code (real-recall miner, --adversarial-source, --adversarial-rank-threshold, PPR cache patch were all already in).

## Phase 3 progress

Priority B (candidate-pool widening) → resolved in V5+2 by GT cleanup, confirmed in V5+3 by miss-rate-stays-zero. Priority C (session_recency LOFO audit) → partially answered without running it (V5+3 evidence shows the dominance is composition-dependent, not pure leakage); a formal LOFO is still worth one session for a clean ablation, but no longer load-bearing.

## Next experiment options for V5+4 (in priority order)

(a) **Re-mine real-recall pathologies against V5+3b** to see how the supply has shifted — did the 25 adversarial-targeted memories drop off the pathology list? Direct experimental verification, ~10 minutes.

(b) **Live GT re-rate of suspicious worst-regressions** — write a small script that pulls the bottom-10 live regressions from each retrain, sends them to Opus with the candidate pool for a fresh judgment, and surfaces ones where the rater's labels disagree with current GT. Catches the "permissions settings.json" pattern at scale.

(c) **session_recency LOFO audit** — formally measure per-mode NDCG impact of removing session_recency from the feature set.

(d) Next adversarial probe pass at a different rank threshold.

(e) Begin one of the deferred items: 31-feature retrain documentation in `docs/experiments.md`, FSI stability audit, sleep impact measurement.

My take on EV ordering: (a) > (b) > (d) > (c) > (e).
