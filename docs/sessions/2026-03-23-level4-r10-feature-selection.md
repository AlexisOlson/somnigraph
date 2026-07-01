# 2026-03-23 — R@10-optimized feature selection (Level 4)

Analyzed overnight results: forward-12 generalized better than backward-26 despite weaker CV (overfitting). Built --select-metric and --select-union for metric-specific feature selection. Manual seed determination (6-feature R@10 core), two-pass forward (R@10 primary, NDCG secondary), backward prune.

Final 17-feature model: baseline R@10=86.3%, expanded R@10=88.7%. Added BM25-damped IDF to keyword expansion term selection (tested 3 variants; aggressive IDF hurt multi-hop, BM25-damped recovered it).

Updated benchmarks.md with full Level 4 results.

### Surprise

R@10 and NDCG select structurally different features — 3 features backward eliminated for NDCG were selected by R@10 forward. The metrics disagree on what "good retrieval" means.

P4 description updated.
