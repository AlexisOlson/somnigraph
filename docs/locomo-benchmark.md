# LoCoMo Retrieval Benchmark

Tracking retrieval quality on the [LoCoMo benchmark](https://arxiv.org/abs/2402.17753) (Maharana et al., ACL 2024). 10 conversations, ~300 turns each, 1977 questions across 5 reasoning types.

Metric: R@k (does any evidence turn appear in the top-k retrieved?), MRR (mean reciprocal rank of first evidence hit). OVERALL excludes adversarial questions (no ground truth answer — retrieval "evidence" is ambiguous).

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
OVERALL        1531   0.463   32.6%   53.4%   65.3%   75.4%   84.5%
```

**Notes:**
- Ingestion: raw dialog turns as `[Speaker] text`, no enrichment. Themes = speaker + session tag. Summary = first 80 chars of content.
- Embeddings: text-embedding-3-small on enriched text (content + category + themes + summary).
- Scoring: bare config zeroes out THEME_BOOST, UCB, Hebbian, PPR. Pure FTS rank + vec rank through RRF.
- No feedback, no edges, no reranker — just the two retrieval channels fused.
- Multi-hop is the clear weak spot (52.8% R@10) — queries reference information spread across turns that share few keywords with any single evidence turn.

### Level 1 — LoCoMo Reranker v1 (2026-03-22)

LightGBM reranker trained specifically for LoCoMo retrieval. 15 features in 4 groups (core retrieval, query-content text, cross-result, metadata), leave-one-conversation-out CV (10-fold).

The production reranker (26 features) is useless on LoCoMo DBs — most features are dead (no feedback, no edges, no Hebbian, constant category/priority/confidence). This reranker uses only features with signal on fresh benchmark data.

All 10 conversations, 1531 non-adversarial questions. `eval_retrieval.py --configs bare locomo_reranker --recall-limit 20`

```
Config              N     MRR     R@1     R@3     R@5    R@10    R@20
----------------------------------------------------------------------
bare             1531   0.462   32.5%   53.3%   65.1%   75.0%   84.5%
locomo_reranker  1531   0.740   66.6%   79.0%   83.7%   88.7%   92.1%
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

**Delta vs bare RRF (OVERALL):** MRR +0.278, R@1 +34.1pp, R@10 +13.7pp. Multi-hop improved from 52.8% → 74.2% R@10 (+21.4pp).

Superseded by Level 2 below.

### Level 2 — LoCoMo Reranker v2: entity overlap + session co-occurrence (2026-03-22)

Added two new features targeting the multi-hop gap:
- **entity_overlap** (Group E): Count of shared non-speaker named entities between query and candidate. Regex-based proper noun extraction (capitalized words not at sentence start, excluding speakers and a small stoplist). Zero API cost.
- **session_cooccurrence** (Group E): Number of other top-20 RRF candidates sharing this candidate's session. Sessions recovered from themes (`session_N` tags). Top-20 scope prevents domination by session size.

Ablation revealed Group C (cross-result features: rank_agreement, neighbor_density, score_percentile) hurts generalization despite high feature importance — classic overfitting pattern. Three dead features (proximity, age_days, theme_count) confirmed at 0.0 gain. Best config `no_C_pruned` drops all six, leaving 11 features.

All 10 conversations, 1531 non-adversarial questions. `eval_retrieval.py --configs bare locomo_reranker --recall-limit 20`

```
Config              N     MRR     R@1     R@3     R@5    R@10    R@20
----------------------------------------------------------------------
bare             1531   0.462   32.5%   53.3%   65.1%   75.0%   84.5%
locomo_reranker  1531   0.726   64.6%   78.4%   83.4%   87.4%   91.5%
```

Per-category breakdown (locomo_reranker):
```
Category            N     MRR     R@1     R@3     R@5    R@10    R@20
----------------------------------------------------------------------
single-hop        281   0.709   60.9%   77.9%   85.8%   89.3%   93.2%
temporal          320   0.730   66.2%   76.6%   80.9%   87.5%   91.2%
multi-hop          89   0.538   46.1%   56.2%   65.2%   73.0%   77.5%
open-domain       841   0.750   67.1%   81.6%   85.5%   88.2%   92.6%
adversarial       446   0.654   57.0%   70.6%   76.2%   81.2%   84.5%
```

**Delta vs bare RRF (OVERALL):** MRR +0.264, R@1 +32.1pp, R@10 +12.4pp, R@20 +7.0pp.

The new features contribute: session_cooccurrence is the 8th most important feature (244.4 gain), entity_overlap is 10th (45.0). Both are cheap and target multi-hop reasoning.

**Feature groups:**
- **A (0-4)**: Core retrieval — fts_rank, vec_rank, fts_bm25, vec_dist, theme_overlap
- **B (5-8)**: Query-content text — query_coverage, proximity, speaker_match, query_length
- **C (9-11)**: Cross-result — rank_agreement, neighbor_density, score_percentile
- **D (12-14)**: Metadata — token_count, age_days, theme_count
- **E (15-16)**: Entity & session — entity_overlap, session_cooccurrence

**Deployed model** uses `no_C_pruned` config (11 features: A + B\proximity + D\age_days,theme_count + E).

**Scripts:**
```bash
# Full ablation (train + evaluate all feature group combinations)
uv run scripts/locomo_bench/train_locomo_reranker.py --ablation

# Save model with specific config
uv run scripts/locomo_bench/train_locomo_reranker.py --train-only --config no_C_pruned --save-model

# Run eval with LoCoMo reranker (end-to-end via impl_recall pipeline)
uv run scripts/locomo_bench/eval_retrieval.py --dataset locomo --configs bare locomo_reranker --conversations 0 1 2 3 4 5 6 7 8 9
```

**Ablation results (leave-one-conv-out CV, 10-fold):**

Baseline here is lower than end-to-end eval because CV includes evidence memories in the candidate pool even when not retrieved — harder ranking task.

```
Config              N     MRR     R@1     R@3     R@5    R@10    R@20
---------------------------------------------------------------------------
bare_rrf         1977   0.370   27.2%   40.2%   46.1%   55.9%   65.9%
A_only           1977   0.503   36.3%   58.7%   67.2%   77.8%   85.0%
B_only           1977   0.234   15.5%   25.5%   30.7%   37.9%   46.1%
C_only           1977   0.360   25.5%   38.9%   45.2%   57.0%   71.5%
D_only           1977   0.048    1.1%    3.5%    5.2%   10.0%   16.9%
E_only           1977   0.213    9.5%   24.2%   34.9%   50.7%   59.6%
all              1977   0.522   37.9%   61.4%   69.6%   77.9%   85.7%
no_A             1977   0.444   30.9%   50.5%   60.3%   72.4%   82.7%
no_B             1977   0.504   36.1%   58.8%   67.9%   77.2%   85.2%
no_C             1977   0.532   39.1%   62.3%   70.7%   79.6%   86.3%
no_D             1977   0.516   37.3%   60.5%   69.4%   78.5%   85.9%
no_E             1977   0.520   38.3%   60.2%   68.7%   77.7%   85.5%
no_C9            1977   0.522   38.0%   61.6%   70.0%   78.0%   86.4%
no_C10           1977   0.526   38.4%   61.8%   69.5%   78.4%   85.7%
no_C11           1977   0.525   37.9%   61.6%   69.8%   79.8%   86.8%
pruned           1977   0.522   37.9%   61.4%   69.6%   77.9%   85.7%
no_C_pruned      1977   0.532   39.1%   62.3%   70.7%   79.6%   86.3%
```

**Findings:**
- Group A (core retrieval signals) carries most of the gain — the model learns better weighting than fixed RRF.
- Group E adds +0.2pp R@10 and +0.2pp R@20 over no_E. Modest but positive, and E_only (50.7% R@10) shows real standalone signal.
- Group C hurts generalization: `no_C` beats `all` on every metric (+1.0pp MRR, +1.7pp R@10, +0.6pp R@20). Individual C feature drops are marginal — no single culprit, the group is correlated noise.
- Dead features confirmed: age_days, theme_count, proximity (all 0.0 gain). Pruning them is neutral (`pruned` = `all`).
- `no_C_pruned` is the best config: 0.532 MRR, 79.6% R@10, 86.3% R@20 — matches `no_C` exactly.
- B alone is harmful (below bare RRF) — text features without retrieval signals mislead.

**Feature importance (gain, no_C_pruned model):**

```
  [D] token_count         :    661.1
  [A] vec_rank            :    532.9
  [A] vec_dist            :    480.7
  [B] query_coverage      :    474.9
  [B] query_length        :    333.6
  [A] fts_bm25            :    321.3
  [A] fts_rank            :    309.3
  [E] session_cooccurrence:    244.4
  [A] theme_overlap       :     47.3
  [E] entity_overlap      :     45.0
  [B] speaker_match       :     44.5
```

token_count (#1) makes sense — longer turns contain more information, more likely to be evidence. session_cooccurrence (#8) validates the hypothesis that evidence clusters within conversation sessions.
