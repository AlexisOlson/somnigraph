# 2026-03-25 — Graph pipeline + Level 5 reranker

Built `build_graph.py`: inserts synthetic claim/segment nodes + edges (EXTRACTED_FROM, claim_coref) from v6 extractions into all 10 benchmark DBs. 1,270 synthetic nodes, ~8,400 edges. Design: synthetic nodes participate in Phase 1 RRF as vocabulary bridges, get resolved to source turns via edge traversal before Phase 2; reranker only scores turns.

Added Group I features (graph_edge_count, graph_synthetic_score, graph_coref_hits, is_graph_resolved) to train and eval. 5-seed variance check: R@10 86.5-87.3% (0.8pp spread), stable. graph_synthetic_score is #1 feature by importance (856 mean gain, CV 5.6%) — the vocabulary bridging signal dominates the entire model.

Forward selection (R@10 -> NDCG -> backward prune) set up for overnight run with seeds 42+123, JSONL logging.

### Surprise

Initial entity coref approach produced 5,451 edges for conv 0 (O(n^2) per entity); switching to per-claim evidence sets reduced to 330 edges with higher precision.

No priority reorder — P4 work continues with graph-augmented retrieval as the active experiment.
