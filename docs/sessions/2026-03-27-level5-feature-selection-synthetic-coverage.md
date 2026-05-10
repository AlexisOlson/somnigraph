# 2026-03-27 — Level 5 feature selection + eval + synthetic coverage judging

Re-ran feature selection on 4000-size extractions (initial Level 5 used features selected at 200-limit, producing 3.5pp R@10 regression). Two rounds: 5-seed forward/backward from 21-feature union, then 15 manual variations (configs A-T). Config K won: swap score_percentile for token_count in 15-feature intersection.

Eval results (expanded): R@10=88.4% (L4 parity), MRR=0.790 (+0.077 over L4), R@20=93.7% (+1.6pp). Multi-hop R@10 regressed: 68.5% vs L4 75.3% (-6.8pp).

Built synthetic coverage judging pipeline: LLM determines which GT evidence turns each synthetic covers per question. Calibrated Opus/Haiku/Sonnet: Haiku batch_size=5 (88.2% vs Opus), Sonnet batch_size=10 (87.5% vs Opus, zero nonsensical disagreements). Opus self-consistency check on 11 disputed cases: 5 consistent, 2 agreed with Haiku, 4 genuinely ambiguous. Sonnet batch_size=10 chosen for production run.

### Surprise

Feature selection at wrong search sizes was the entire Level 5 regression — the graph features themselves are valid, the selection context was stale. Also: backward elimination should use the primary forward metric (R@10), not --select-metric default (NDCG) — fixed in code.

P4 updated with Level 5 results and multi-hop regression.
