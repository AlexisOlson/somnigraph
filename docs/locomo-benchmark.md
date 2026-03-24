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
- **entity_overlap** (Group E): Count of shared named entities between query and candidate. Uses a curated allowlist of ~200 LoCoMo entities (people, places, brands, media) rather than capitalization heuristics. Zero API cost.
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

**Deployed model** superseded by Level 3 (forward stepwise, 12 features).

**Scripts:**
```bash
# Full ablation (train + evaluate all feature group combinations)
uv run scripts/locomo_bench/train_locomo_reranker.py --ablation

# Forward stepwise feature selection
uv run scripts/locomo_bench/train_locomo_reranker.py --train-only --select forward --n-estimators 200

# Save model with explicit feature set
uv run scripts/locomo_bench/train_locomo_reranker.py --train-only --save-model --n-estimators 800 \
  --feature-names vec_rank query_coverage fts_rank speaker_match token_count \
  has_temporal_expr vec_dist centroid_distance theme_complementarity query_length \
  session_cooccurrence entity_density

# Run retrieval eval with LoCoMo reranker
uv run scripts/locomo_bench/eval_retrieval.py --dataset locomo --configs bare locomo_reranker --conversations 0 1 2 3 4 5 6 7 8 9

# Run end-to-end QA
uv run scripts/locomo_bench/run.py --prompt-mode somnigraph --reader-model gpt-4.1-mini --skip-adversarial --locomo-reranker
uv run scripts/locomo_bench/run.py --prompt-mode somnigraph --reader-model gpt-4.1-mini --skip-adversarial --locomo-reranker --no-judge  # reader only, batch judge later
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

### Level 3 — Forward Stepwise Feature Selection (2026-03-21)

NDCG@10-optimized forward stepwise selection over 25 candidate features (Groups A-G, including inter-passage and embedding diversity features). Leave-one-conversation-out CV, n_estimators=200.

**Selected features (12):** vec_rank, query_coverage, fts_rank, speaker_match, token_count, has_temporal_expr, vec_dist, centroid_distance, theme_complementarity, query_length, session_cooccurrence, entity_density.

```
Config              N     MRR  NDCG@10     R@1     R@3     R@5    R@10    R@20
-----------------------------------------------------------------------------------
bare_rrf         1977   0.370    0.372   27.1%   40.2%   46.1%   55.9%   65.9%
reranker         1977   0.557    0.579   42.6%   63.3%   71.6%   80.0%   86.7%
```

**Feature importance (gain):**
```
  [G] centroid_distance   :   1034.8
  [D] token_count         :   1009.5
  [A] vec_rank            :    979.0
  [A] vec_dist            :    794.6
  [B] query_coverage      :    792.1
  [F] entity_density      :    784.2
  [A] fts_rank            :    703.3
  [B] query_length        :    491.2
  [E] session_cooccurrence:    433.1
  [F] theme_complementarity:    167.7
  [F] has_temporal_expr   :    133.2
  [B] speaker_match       :     66.3
```

centroid_distance (Group G, 2nd-pass embedding diversity) leads — validates the inter-passage hypothesis from v2. New feature groups F and G contribute 4 of 12 selected features.

**New feature groups:**
- **F (17-21)**: Inter-passage — has_temporal_expr, entity_density, topk_session_frac, entity_bridge_count, theme_complementarity
- **G (22-24)**: Set diversity — mmr_redundancy, unique_token_frac, centroid_distance
- **H (25-31)**: Phase-aware — entity_fts_rank, sub_query_hit_count, seed_keyword_overlap, phase, expansion_method_count, phase1_rrf_score, is_seed (sentinels in phase 1, real values in phase 2)

### Two-Phase Expansion (2026-03-22)

Train/eval mismatch fix: original candidates lacked embeddings in phase 2 (vec-dependent features were wrong), temporal feature used a different regex than training. After fix, expansion improves all metrics instead of degrading.

Conversations 0-4, baseline (no expansion):
```
Category            N     MRR     R@1     R@3     R@5    R@10    R@20
----------------------------------------------------------------------
single-hop        141   0.632   51.8%   70.9%   78.7%   87.2%   90.1%
temporal          156   0.722   62.2%   80.8%   84.6%   87.2%   91.7%
multi-hop          44   0.378   27.3%   40.9%   47.7%   61.4%   75.0%
open-domain       418   0.665   57.2%   72.0%   79.2%   85.2%   89.5%
adversarial       237   0.582   48.9%   65.0%   69.6%   75.1%   79.7%
----------------------------------------------------------------------
OVERALL           759   0.654   55.5%   71.8%   78.4%   84.6%   89.2%
(excludes adversarial)
```

With `--expand-all` (conversations 0-4):
```
Category            N     MRR     R@1     R@3     R@5    R@10    R@20
----------------------------------------------------------------------
single-hop        141   0.696   58.9%   76.6%   83.7%   88.7%   92.2%
temporal          156   0.792   73.7%   82.1%   85.9%   88.5%   92.9%
multi-hop          44   0.407   29.5%   45.5%   61.4%   70.5%   77.3%
open-domain       418   0.703   62.2%   76.3%   80.4%   85.9%   90.4%
adversarial       237   0.618   53.6%   66.7%   73.0%   78.1%   81.0%
----------------------------------------------------------------------
OVERALL           759   0.703   62.1%   75.8%   81.0%   86.0%   90.5%
(excludes adversarial)
```

**Delta (expand-all vs baseline):** MRR +0.049, R@1 +6.6pp, R@3 +4.0pp, R@5 +2.6pp, R@10 +1.4pp, R@20 +1.3pp. Multi-hop R@10 +9.1pp (61.4% → 70.5%). Expansion helps across every category and metric.

### What didn't work: capitalization-based entity extraction

The original entity bridge and entity_overlap features used a capitalization heuristic to find proper nouns: capitalized words not at sentence start, not in a stoplist. This failed badly on casual dialog.

**First fix (session 44):** Added sentence-boundary detection (`prev_ends_sentence` flag) and I-contraction filtering. This caught sentence-initial words but missed the dominant failure mode: vocative patterns like "Hey John!" where "Hey" has no trailing punctuation, so "John" is treated as mid-sentence. Also missed all conversational filler that happens to be capitalized mid-sentence ("Thanks", "Wow", "Cool").

**Second fix (session 45):** Replaced the heuristic entirely with a curated allowlist. Extracted all 1,209 distinct capitalized words from LoCoMo, had an Opus agent classify each as entity or noise, producing ~200 real entities (people, places, brands, media, events). The bridge method now does simple allowlist lookup instead of capitalization parsing. This also flipped speaker names from *excluded* to *included* — speakers are the highest-value bridge entity ("seed mentions Caroline → find other Caroline memories") but were previously filtered out.

**Lesson:** Capitalization is not a feature of casual dialog — it's a feature of edited prose. Any heuristic built on it will produce more noise than signal on conversational data. A corpus-specific allowlist is less general but dramatically cleaner.

**Overnight run (session 44):** Retrain with entity stopword fix + 3 new Group H features (expansion_method_count, phase1_rrf_score, is_seed) + theme_overlap fix for expanded candidates. Forward stepwise, backward elimination compared across all 10 conversations, both with and without expansion. Results in `~/.somnigraph/benchmark/overnight/`.

### Overnight Run Results (2026-03-22 → 2026-03-23)

Forward stepwise (NDCG@10) and backward elimination (NDCG@10) across all 10 conversations with the entity stopword fix and 3 new Group H features. n_estimators=300.

**Feature selection (CV on all 10 conversations, ~1.65M pairs):**

| Method | Features | NDCG@10 |
|--------|----------|---------|
| Forward stepwise | 12 | 0.8472 |
| Backward elimination | 26 (removed 3: entity_overlap, entity_density, topk_session_frac) | 0.8590 |
| Full 29 features | 29 | 0.8466 |

**Trained models (leave-one-conv-out CV):**

| Model | MRR | NDCG@10 | R@1 | R@5 | R@10 | R@20 |
|-------|-----|---------|-----|-----|------|------|
| bare RRF | 0.369 | 0.371 | 27.0% | 46.1% | 55.9% | 65.9% |
| Forward (12f) | 0.492 | 0.828 | 40.9% | 61.2% | 69.7% | 79.8% |
| Backward (26f) | 0.505 | 0.859 | 41.5% | 63.6% | 73.7% | 81.4% |

**End-to-end retrieval eval (all 10 conversations, limit=20):**

| Model | Mode | MRR | R@1 | R@5 | R@10 | R@20 |
|-------|------|-----|-----|-----|------|------|
| Forward 12f | baseline | 0.665 | 55.9% | 79.8% | 85.7% | 90.6% |
| Forward 12f | expanded | 0.704 | 61.5% | 81.7% | 87.3% | 91.8% |
| Backward 26f | baseline | 0.652 | 54.6% | 78.4% | 83.5% | 87.3% |

Backward 26f expanded eval did not complete (killed mid-run).

**Key finding:** Backward had higher CV scores but lower end-to-end performance — classic overfitting from 26 features on this data. Forward 12f generalized better despite weaker CV metrics.

**New Group H features:** expansion_method_count was selected at step 6 by forward (NDCG +0.0095). Backward kept all three (expansion_method_count, phase1_rrf_score, is_seed). is_seed had near-zero importance (0.8 gain) but didn't hurt enough to remove.

### Level 4 — R@10-Optimized Feature Selection (2026-03-23)

The overnight results showed NDCG@10 optimization doesn't maximize R@10 (backward-26 had better CV NDCG but worse end-to-end R@10 than forward-12). This prompted a metric-specific selection strategy.

**Approach:** Two-pass forward selection (`--select-union`):
1. Forward greedy on R@10 until exhausted
2. Continue from that set, accepting features that improve NDCG@10
3. Backward elimination on R@10 to prune

**Seed determination:** Manual testing identified a 6-feature R@10 core (vec_rank, fts_rank, token_count, query_coverage, query_length, has_temporal_expr) by incrementally building up from single features and measuring R@10 at each step. Key finding: fts_rank hurts R@10 when paired only with vec_rank (-1.3pp) but becomes useful once content features are added.

**Forward selection (seeded with 6 features, 23 candidates):**

R@10 primary pass:
```
  Step 1: +speaker_match             r@10=0.6869 (+0.0197)
  Step 2: +is_seed                   r@10=0.7142 (+0.0273)
  Step 3: +vec_dist                  r@10=0.7172 (+0.0030)
  Step 4: +entity_density            r@10=0.7198 (+0.0025)
  Step 5: +topk_session_frac         r@10=0.7223 (+0.0025)
  Step 6: +entity_overlap            r@10=0.7289 (+0.0066)
  Step 7: +centroid_distance         r@10=0.7451 (+0.0162)
  Step 8: +entity_fts_rank           r@10=0.7476 (+0.0025)
```

NDCG@10 secondary pass:
```
  Step 10: +theme_complementarity     ndcg@10=0.8583 (+0.0005)
  Step 11: +expansion_method_count    ndcg@10=0.8672 (+0.0089)
  Step 12: +sub_query_hit_count       ndcg@10=0.8684 (+0.0012)
  Step 13: +theme_overlap             ndcg@10=0.8793 (+0.0109)
```

Backward elimination on R@10 removed entity_fts_rank (+0.0051 from removal).

**Final model: 17 features.**

**R@10 vs NDCG divergence:** Three features that backward elimination *removed* for NDCG (entity_density, topk_session_frac, entity_overlap) were *selected* by R@10 forward. These features help get the right answer into the top 10 but hurt ranking precision within it — different metrics, different optimal feature sets.

**Leave-one-conv-out CV (17 features, n_estimators=300):**

```
Config              N     MRR  NDCG@10     R@1     R@3     R@5    R@10    R@20
-----------------------------------------------------------------------------------
bare_rrf         1977   0.370    0.372   27.1%   40.2%   46.1%   55.9%   65.9%
reranker         1977   0.516    0.872   41.9%   57.5%   65.1%   75.3%   83.1%
-----------------------------------------------------------------------------------
Delta vs bare RRF:
  reranker             +0.147   +0.501  +14.9%  +17.2%  +19.0%  +19.3%  +17.2%
```

Beats backward-26 (NDCG-optimized, 26 features) on every metric with 9 fewer features.

**Feature importance (gain):**
```
  [G] centroid_distance   :   1240.4
  [D] token_count         :   1215.4
  [A] vec_rank            :   1070.6
  [A] vec_dist            :    983.1
  [B] query_coverage      :    969.7
  [A] fts_rank            :    828.4
  [F] entity_density      :    682.2
  [B] query_length        :    673.3
  [F] topk_session_frac   :    483.8
  [F] theme_complementarity:    183.5
  [F] has_temporal_expr   :    158.7
  [H] sub_query_hit_count :    118.0
  [E] entity_overlap      :    109.0
  [A] theme_overlap       :    108.6
  [B] speaker_match       :     81.0
  [H] is_seed             :     72.8
  [H] expansion_method_count:      9.5
```

**End-to-end retrieval eval (baseline, all 10 conversations, limit=20):**

```
Category            N     MRR     R@1     R@3     R@5    R@10    R@20
----------------------------------------------------------------------
single-hop        281   0.685   57.7%   76.5%   81.9%   87.2%   92.5%
temporal          320   0.708   61.9%   77.2%   81.6%   86.2%   90.3%
multi-hop          89   0.483   36.0%   57.3%   64.0%   76.4%   82.0%
open-domain       841   0.712   62.3%   77.5%   81.9%   87.2%   90.8%
adversarial       446   0.596   49.6%   66.6%   72.0%   77.8%   83.0%
----------------------------------------------------------------------
OVERALL          1531   0.693   59.8%   76.1%   80.8%   86.3%   90.5%
(excludes adversarial)
```

**Delta vs Level 3 forward-12 baseline (OVERALL):**

| Metric | Forward-12 (NDCG) | R@10-optimized (17f) | Delta |
|--------|-------------------|----------------------|-------|
| MRR | 0.665 | 0.693 | +0.028 |
| R@1 | 55.9% | 59.8% | +3.9pp |
| R@10 | 85.7% | 86.3% | +0.6pp |
| R@20 | 90.6% | 90.5% | -0.1pp |

Per-category R@10 gains: multi-hop +5.6pp (70.8% → 76.4%), adversarial +3.8pp, temporal +0.6pp, open-domain +0.4pp, single-hop flat.

**With `--expand-all` (BM25-damped IDF keyword selection):**

```
Category            N     MRR     R@1     R@3     R@5    R@10    R@20
----------------------------------------------------------------------
single-hop        281   0.693   56.6%   79.0%   87.2%   91.1%   94.0%
temporal          320   0.736   65.9%   79.7%   82.8%   87.8%   91.6%
multi-hop          89   0.478   36.0%   53.9%   61.8%   75.3%   79.8%
open-domain       841   0.735   64.8%   80.7%   84.1%   89.7%   93.0%
adversarial       446   0.638   53.4%   70.4%   76.5%   83.4%   87.0%
----------------------------------------------------------------------
OVERALL          1531   0.713   61.9%   78.6%   83.1%   88.7%   92.1%
(excludes adversarial)
```

**Delta (expanded vs baseline, OVERALL):** MRR +0.020, R@1 +2.1pp, R@10 +2.4pp, R@20 +1.6pp.

**Keyword expansion IDF experiment:** Tested three term selection strategies for keyword expansion: (1) raw frequency (original), (2) aggressive IDF `log(N/df)`, (3) BM25-damped IDF `log((N-df+0.5)/(df+0.5)+1)`. Aggressive IDF improved MRR/R@1 but crushed multi-hop R@10 (-2.3pp) by filtering bridge terms. BM25-damped recovered multi-hop while keeping MRR gains. The damped formula matches BM25's own information-theoretic assumptions — consistent with the scoring already in use.

## End-to-End QA Results

### Run 1 — GPT-4.1-mini reader (2026-03-21)

First end-to-end QA run. LoCoMo reranker (12 features, Level 3), GPT-4.1-mini reader, skip adversarial.

```
Judge             N    single-hop  temporal  multi-hop  open-domain  OVERALL
──────────────────────────────────────────────────────────────────────────────
GPT-4.1-mini   1540      84.0%      87.5%     72.9%      91.8%       88.3%
Opus 4.6       1540      80.9%      84.4%     61.5%      89.5%       85.1%
```

Opus is 3.2pp stricter than GPT-4.1-mini across the board. Both numbers beat every published system on Overall J (Mem0g: 68.44, Mem0: 66.88, Full-context: 72.90).

**Per-conversation accuracy (Opus judge):**
```
Conv  0: 84.9%    Conv  5: 82.1%
Conv  1: 90.1%    Conv  6: 86.0%
Conv  2: 90.8%    Conv  7: 87.4%
Conv  3: 78.4%    Conv  8: 84.6%
Conv  4: 83.7%    Conv  9: 86.7%
```

### Reader model comparison: GPT-4.1-mini vs GPT-5.4-mini

GPT-5.4-mini (reasoning model) tested as reader on conversations 0-4. Same LoCoMo reranker, Opus judge.

```
Conv    4.1-mini    5.4-mini    Delta
────────────────────────────────────────
  0      84.9%       80.9%     -4.0pp
  1      90.1%       85.2%     -4.9pp
  2      90.8%       86.2%     -4.6pp
  3      78.4%       74.4%     -4.0pp
  4      83.7%       71.3%    -12.4pp
────────────────────────────────────────
0-4      85.1%       78.5%     -6.6pp
```

**Finding:** Reasoning models hurt factual QA extraction. GPT-5.4-mini computes dates instead of quoting the context — e.g., answering "8 July 2023" (calculated) vs. "Last Friday before 15 July 2023" (extracted from context). The temporal category shows the most failures. For memory QA, a direct extraction model (4.1-mini) outperforms a reasoning model (5.4-mini) because the task is retrieval + quotation, not inference.

## Reference: Published LoCoMo QA Scores

End-to-end QA accuracy (F1, BLEU-1, LLM-as-a-Judge) from the Mem0 paper (Table 1, arXiv:2504.19413v1). These measure answer quality, not retrieval recall — the metric our Level 0-2 results above don't yet capture.

```
                    Single Hop          Multi-Hop           Open Domain         Temporal
Method           F1    B1     J      F1    B1     J      F1    B1     J      F1    B1     J
─────────────────────────────────────────────────────────────────────────────────────────────
LoCoMo          25.02 19.75   –     12.04 11.16   –     40.36 29.05   –     18.41 14.77   –
ReadAgent        9.15  6.48   –      5.31  5.12   –      9.67  7.66   –     12.60  8.87   –
MemoryBank       5.00  4.77   –      5.56  5.94   –      6.61  5.16   –      9.68  6.99   –
MemGPT          26.65 17.72   –      9.15  7.44   –     41.04 34.34   –     25.52 19.44   –
A-Mem           27.02 20.09   –     12.14 12.00   –     44.65 37.06   –     45.85 36.67   –
A-Mem*          20.76 14.90  39.79   9.22  8.81  18.85  33.34 27.58  54.05  35.40 31.08  49.91
LangMem         35.51 26.86  62.23  26.04 22.32  47.92  40.91 33.63  71.12  30.75 25.84  23.43
Zep             35.74 23.30  61.70  19.37 14.82  41.35  49.56 38.92  76.60  42.00 34.53  49.31
OpenAI          34.30 23.72  63.79  20.09 15.42  42.92  39.31 31.16  62.29  14.04 11.25  21.71
Mem0            38.72 27.13  67.13  28.64 21.58  51.15  47.65 38.72  72.93  48.93 40.51  55.51
Mem0g           38.09 26.03  65.71  24.32 18.82  47.19  49.27 40.30  75.71  51.55 40.28  58.13
─────────────────────────────────────────────────────────────────────────────────────────────
Ori Mnemos      37.69  –      –     29.31  –      –       –     –      –       –     –      –
```

*A-Mem\* = re-run with temperature 0 for judge. Mem0g = Mem0 + graph memory (Neo4j). Ori Mnemos numbers from their README (GPT-4.1-mini generation, BM25 + embedding + PageRank fusion). Adversarial category excluded from all — no ground truth.*

### Additional published Overall J scores

| System | Overall J | Judge | Cats | Notes |
|--------|----------|-------|------|-------|
| MemU | 92.09% | — | 1-4 | No public methodology |
| MemMachine v0.2 | 91.23% | — | 1-4 | $43.5M funding |
| EXIA GHOST | 89.94% | GPT-4o-mini | 1-4 | Proprietary; ChromaDB + MiniLM, LLM extraction of atomic facts. See `research/sources/exia-ghost.md` |
| **Somnigraph** | **88.3%** | **GPT-4.1-mini** | **1-4** | **No sleep enhancements yet** |
| **Somnigraph** | **85.1%** | **Opus 4.6** | **1-4** | **Stricter judge (+3.2pp)** |
| EXIA GHOST | 85.80% | GPT-4o-mini | 1-5 | Only system publishing cat 5; 71.52% adversarial |
| RedPlanet CORE | 88.24% | — | 1-4 | Self-reported, Neo4j + pgvector |
| Memobase | 75.78% | — | 1-4 | |

*All competitor scores are self-reported. No independent cross-verification exists for any system. Judge leniency varies: Opus is 3.2pp stricter than GPT-4.1-mini (measured). GPT-4o-mini leniency relative to GPT-4.1-mini is unknown.*

**Note on Somnigraph headroom:** Current results use baseline retrieval + reader with no sleep enhancements (no consolidation, no edge building, no summary generation, no expansion on full QA pipeline). Sleep pass ablation and feedback loop ablation are pending — these represent the primary expected sources of uplift.

**Key observations:**
- Everyone clusters in the 35-39 F1 range for single-hop. The field is saturated on easy questions.
- Multi-hop separates the systems: Mem0 leads at 28.64 F1, Ori slightly ahead at 29.31.
- Temporal is the biggest gap: Mem0g 58.13 J vs. MemoryBank 6.99 B1. Systems without temporal modeling collapse.
- Open-domain: Zep wins on J (76.60) despite losing overall. Graph traversal helps for broad queries.
- The full-context baseline (LoCoMo paper) scores 72.9% overall — higher than every specialized system. Context management > memory tools for most questions.

### Overall J scores and latency (Table 2)

Aggregate LLM-as-a-Judge scores and latency from the same paper. This is the single-number comparison target for our end-to-end QA benchmark.

```
Method              Tokens   Search p50/p95   Total p50/p95    Overall J
──────────────────────────────────────────────────────────────────────────
RAG K=1, 256          256    0.251 / 0.710    0.745 / 1.628    50.15
RAG K=2, 256          256    0.255 / 0.699    0.802 / 1.907    60.97
Full-context       26,031        –     –      9.870 / 17.117   72.90
──────────────────────────────────────────────────────────────────────────
A-Mem               2,520    0.668 / 1.485    1.410 / 4.374    48.38
LangMem               127   17.99  / 59.82   18.53  / 60.40   58.10
Zep                 3,911    0.513 / 0.778    1.292 / 2.926    65.99
OpenAI              4,437        –     –      0.466 / 0.889    52.90
Mem0                1,764    0.148 / 0.200    0.708 / 1.440    66.88
Mem0g               3,616    0.476 / 0.657    1.091 / 2.590    68.44
```

*RAG rows are the best-performing chunk sizes at K=1 and K=2. Full table in the paper sweeps 128–8192 tokens.*

**Key observations:**
- RAG K=2 at 256 tokens (60.97 J) nearly matches specialized systems with zero memory infrastructure. The bar for "better than naive RAG" is higher than it looks.
- Full-context (72.90 J) remains unbeaten. Every memory system trades accuracy for token efficiency.
- Mem0's speed advantage is real: 148ms search p50 vs. Zep's 513ms. But Zep's richer retrieval buys 65.99 → only 0.89 J behind Mem0.
- LangMem's latency (18s p50) is an outlier — likely LLM calls in the retrieval path.
- Token budget varies 10x across systems (127 to 4,437). Our `recall(limit=N)` parameter controls this directly.
