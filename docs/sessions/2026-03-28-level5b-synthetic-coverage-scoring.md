# 2026-03-28 — Level 5b — synthetic coverage scoring

Retrained reranker with synthetics as labeled candidates (coverage table provides labels). Removed synthetic filtering before Phase 2.

Results: R@10=95.4% (+6.7pp over L4), MRR=0.882 (+0.169), multi-hop R@10=88.8% (+13.5pp over L4, +20.3pp over L5). Multi-hop regression fully recovered and then some — synthetic vocabulary bridges close the gap when allowed through scoring.

graph_synthetic_score dropped from #2 to #13 importance (synthetics compete directly instead of proxying through turns).

P4 updated.
