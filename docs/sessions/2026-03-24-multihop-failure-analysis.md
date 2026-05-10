# 2026-03-24 — Multi-hop failure analysis + 5 source analyses

Analyzed all 22 multi-hop R@10 misses: 4 near-misses (rank 11-20), 18 complete whiffs shared between bare RRF and reranker. Root cause: vocabulary gap — 88% of evidence turns have zero content-word overlap with the query. This characterizes the reranker ceiling for multi-hop and points toward vocabulary-gap solutions (HyDE, query expansion) as the next lever.

Added 5 source analyses (HyDE, Self-RAG, SPLADE, Method Actors, RAG Techniques) — directly motivated by the failure analysis.

P4 updated with failure analysis finding. No priority reorder — P4 remaining items unchanged (ablations still pending), but the failure analysis narrows what retrieval improvements can achieve.
