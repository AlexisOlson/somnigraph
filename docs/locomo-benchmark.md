# LoCoMo Retrieval Benchmark

Tracking retrieval quality on the [LoCoMo benchmark](https://arxiv.org/abs/2402.17753) (Maharana et al., ACL 2024). 10 conversations, ~300 turns each, 1977 questions across 5 reasoning types.

Metric: R@k (does any evidence turn appear in the top-k retrieved?), MRR (mean reciprocal rank of first evidence hit).

## Run history

### Level 0 — Bare RRF (2026-03-20)

Pure FTS + vector fusion, no boosts. `eval_retrieval.py --configs bare --recall-limit 20`

All 10 conversations, 1977 questions with evidence mappings.

```
Category          N     MRR     R@1     R@3     R@5    R@10    R@20
----------------------------------------------------------------------
single-hop      281   0.412   26.0%   47.0%   62.6%   74.7%   84.7%
temporal        320   0.474   34.7%   53.4%   65.3%   76.6%   85.9%
multi-hop        89   0.289   18.0%   33.7%   46.1%   52.8%   64.0%
open-domain     841   0.494   35.6%   57.7%   68.3%   77.5%   86.1%
adversarial     446   0.312   18.4%   35.7%   46.6%   59.9%   70.9%
----------------------------------------------------------------------
OVERALL        1977   0.429   29.4%   49.4%   61.1%   71.9%   81.4%
```

**Notes:**
- Ingestion: raw dialog turns as `[Speaker] text`, no enrichment. Themes = speaker + session tag. Summary = first 80 chars of content.
- Embeddings: text-embedding-3-small on enriched text (content + category + themes + summary).
- Scoring: bare config zeroes out THEME_BOOST, UCB, Hebbian, PPR. Pure FTS rank + vec rank through RRF.
- No feedback, no edges, no reranker — just the two retrieval channels fused.
- Multi-hop is the clear weak spot (52.8% R@10) — queries reference information spread across turns that share few keywords with any single evidence turn.
- Adversarial R@10 = 59.9% — these have no real answer, so retrieving "evidence" is ambiguous. Lower may be better here depending on evaluation design.

### Level 1 — LoCoMo Reranker (2026-03-22)

LightGBM reranker trained specifically for LoCoMo retrieval. 15 features in 4 groups (core retrieval, query-content text, cross-result, metadata), leave-one-conversation-out CV (10-fold).

The production reranker (26 features) is useless on LoCoMo DBs — most features are dead (no feedback, no edges, no Hebbian, constant category/priority/confidence). This reranker uses only features with signal on fresh benchmark data.

All 10 conversations, 1977 questions. `eval_retrieval.py --configs bare locomo_reranker --recall-limit 20`

```
Config              N     MRR     R@1     R@3     R@5    R@10    R@20
----------------------------------------------------------------------
bare             1977   0.428   29.3%   49.4%   61.0%   71.6%   81.5%
locomo_reranker  1977   0.719   64.2%   76.7%   81.9%   87.1%   90.4%
```

Per-category breakdown (locomo_reranker):
```
Category            N     MRR     R@1     R@3     R@5    R@10    R@20
----------------------------------------------------------------------
single-hop        281   0.726   63.7%   80.1%   84.3%   90.4%   93.6%
temporal          320   0.751   67.8%   80.3%   83.4%   87.8%   91.2%
multi-hop          89   0.561   46.1%   62.9%   70.8%   74.2%   80.9%
open-domain       841   0.760   69.3%   79.9%   84.9%   90.0%   93.2%
adversarial       446   0.646   56.1%   68.8%   75.8%   81.4%   84.3%
```

**Delta vs bare RRF:** MRR +0.291, R@1 +34.9pp, R@10 +15.5pp. Multi-hop improved from 52.8% → 74.2% R@10 (+21.4pp). Adversarial from 59.6% → 81.4% R@10 (+21.8pp).

**Feature groups:**
- **A (0-4)**: Core retrieval — fts_rank, vec_rank, fts_bm25, vec_dist, theme_overlap
- **B (5-8)**: Query-content text — query_coverage, proximity, speaker_match, query_length
- **C (9-11)**: Cross-result — rank_agreement, neighbor_density, score_percentile
- **D (12-14)**: Metadata — token_count, age_days, theme_count

**Scripts:**
```bash
# Full ablation (train + evaluate all feature group combinations)
uv run scripts/locomo_bench/train_locomo_reranker.py --ablation

# Save best model for eval_retrieval integration
uv run scripts/locomo_bench/train_locomo_reranker.py --save-model

# Run eval with LoCoMo reranker (end-to-end via impl_recall pipeline)
uv run scripts/locomo_bench/eval_retrieval.py --dataset locomo --configs bare locomo_reranker --conversations 0 1 2 3 4 5 6 7 8 9
```

**Ablation results (leave-one-conv-out CV, 10-fold):**

Baseline here is lower than Level 0 because the training eval includes evidence memories in the candidate pool even when not retrieved — harder ranking task.

```
Config              N     MRR     R@1     R@3     R@5    R@10    R@20
---------------------------------------------------------------------------
bare_rrf         1977   0.370   27.1%   40.2%   46.1%   55.9%   65.9%
A_only           1977   0.500   36.0%   58.5%   66.8%   77.2%   84.6%
B_only           1977   0.233   15.4%   25.4%   30.7%   37.9%   46.0%
C_only           1977   0.362   25.8%   38.7%   44.9%   57.6%   72.0%
D_only           1977   0.048    1.2%    3.7%    5.7%   10.0%   17.7%
all              1977   0.531   39.6%   61.8%   69.5%   77.7%   85.9%
no_A             1977   0.434   30.2%   48.4%   59.1%   71.2%   82.1%
no_B             1977   0.515   37.5%   60.3%   68.3%   78.0%   86.3%
no_C             1977   0.528   39.3%   61.7%   69.1%   77.7%   85.2%
no_D             1977   0.515   37.3%   61.0%   68.7%   77.5%   85.5%
```

**Findings:**
- Group A (core retrieval signals) carries most of the gain — the model learns better weighting than fixed RRF.
- Groups B, C, D add <1pp individually on top of A — marginal in combination.
- B alone is harmful (below bare RRF) — text features without retrieval signals mislead.
- D alone is nearly useless — age_days and theme_count have zero importance (constant across LoCoMo turns).
- Three dead features: age_days, theme_count, proximity (all zero gain). Proximity was also weak in the production reranker.

**Feature importance (gain, full model):**

```
  token_count         :    987.3
  query_coverage      :    737.0
  vec_dist            :    720.5
  score_percentile    :    657.4
  vec_rank            :    624.8
  fts_bm25            :    513.7
  query_length        :    475.0
  fts_rank            :    417.8
  rank_agreement      :    388.2
  neighbor_density    :    181.7
  theme_overlap       :     67.0
  speaker_match       :     64.6
  age_days            :      0.0
  theme_count         :      0.0
  proximity           :      0.0
```

token_count (#1) makes sense — longer turns contain more information, more likely to be evidence. speaker_match contributes despite being binary (64.6 gain).
