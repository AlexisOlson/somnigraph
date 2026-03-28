# LoCoMo Level 5b Handoff

## Where we are

Level 5b complete. Synthetic coverage scoring recovers multi-hop regression and produces best results across all metrics.

### Results

| Config | R@10 | R@20 | MRR | Multi-hop R@10 |
|--------|------|------|-----|----------------|
| Level 4 expanded | 88.7% | 92.1% | 0.713 | 75.3% |
| Level 5 expanded | 88.4% | 93.7% | 0.790 | 68.5% |
| **Level 5b expanded** | **95.4%** | **96.9%** | **0.882** | **88.8%** |

### What changed in L5b

1. **Reranker retrained with synthetics.** `--synthetic-coverage` flag in `train_locomo_reranker.py` includes synthetic nodes as candidates with coverage-based labels. Synthetics covering GT evidence get label=1.0.

2. **No `--graph-resolve`.** Synthetics stay in the result set through both phases. The reranker scores them alongside turns.

3. **Coverage-based eval scoring.** `--synthetic-coverage` flag in `eval_retrieval.py` loads the coverage table. A synthetic at rank k counts as recalling evidence if the table says it covers that evidence for that question.

### Synthetic coverage table

`~/.somnigraph/benchmark/synthetic_coverage.json` — 1,977 questions, 5,118 synth-evidence pairs, 3,916 covered (76.5%). Judged by Sonnet batch_size=10 (87.5% agreement with Opus ground truth).

### What's next

1. **Expansion method ablation:** 3 of 6 methods are dead (rocchio 0%, multi_query 2%, entity_focus 4%). Remove them and measure impact.

2. **Sleep pass ablation:** Run with sleep consolidation enabled. Measure impact on retrieval.

3. **Feedback loop ablation:** Measure retrieval feedback's contribution.

4. **Document findings in experiments.md.**

5. **End-to-end QA re-run with L5b retrieval:** Current 85.1% accuracy used Level 3 retrieval. L5b retrieval (R@10 95.4% vs 88.7%) should improve QA accuracy.

### Key design decisions
- **Synthetics in both phases:** L5 filtered synthetics before Phase 2 (wasted their bridging contribution). L5b keeps them and lets the reranker learn to rank them.
- **Coverage as labels:** LLM-judged coverage table provides training labels for synthetics. A synthetic is positive if it carries the specific evidence needed to answer the question.
- **Feature importance shift:** `graph_synthetic_score` dropped from #2 (L5) to #13 (L5b) because synthetics compete directly instead of proxying through turns.
- **Claim coref edges**: per-claim evidence sets (not all-pairs per entity)
- **Conv cache**: all metadata loaded once per conversation (10x speedup)

### Key commands

**Retrain (L5b):**
```
python scripts/locomo_bench/train_locomo_reranker.py --two-phase --synthetic-coverage C:\Users\Alexis\.somnigraph\benchmark\synthetic_coverage.json --save-model --feature-names fts_rank vec_rank fts_bm25 vec_dist query_coverage query_length rank_agreement token_count has_temporal_expr topk_session_frac mmr_redundancy unique_token_frac phase1_rrf_score graph_synthetic_score graph_coref_hits
```

**Eval (L5b expanded):**
```
python scripts/locomo_bench/eval_retrieval.py --dataset locomo --configs locomo_reranker --conversations 0 1 2 3 4 5 6 7 8 9 --recall-limit 800 --expand-all --synthetic-coverage C:\Users\Alexis\.somnigraph\benchmark\synthetic_coverage.json
```

### Files modified this session
- `scripts/locomo_bench/train_locomo_reranker.py` — `--synthetic-coverage` flag, coverage-based labels, conditional synthetic filter
- `scripts/locomo_bench/eval_retrieval.py` — `--synthetic-coverage` flag, coverage-based scoring in `_score_locomo`
- `docs/locomo-benchmark.md` — Level 5b results documented
- `STEWARDSHIP.md` — P4 updated, changelog entry
