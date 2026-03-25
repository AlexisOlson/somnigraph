# MemReward: Graph-Based Experience Memory for LLM Reward Prediction with Limited Labels -- Analysis

*Generated 2026-03-24 by Opus 4.6 agent reading arXiv:2603.19310v2*

---

## Paper Overview

**Paper**: Tianyang Luo, Tao Feng, Zhigang Hua, Yan Xie, Shuang Yang, Ge Liu, Jiaxuan You (University of Illinois Urbana-Champaign, Meta). "MemReward: Graph-Based Experience Memory for LLM Reward Prediction with Limited Labels." Preprint (under review), March 2026. 20 pages (9 main + 11 appendix). Code: https://github.com/ulab-uiuc/MemReward.

**Problem addressed**: Reinforcement learning fine-tuning of LLMs (e.g., via GRPO) requires reward signals for every rollout, but obtaining ground-truth labels is expensive -- math proofs need verification, open-ended QA lacks definitive answers. With only 20% of queries labeled, naively discarding the unlabeled 80% (the "R1-p" baseline) wastes most of the training signal. The paper asks: can we propagate rewards from labeled rollouts to unlabeled ones via graph structure?

**Core claim**: By organizing LLM rollouts (query, thinking process, answer) as nodes in a heterogeneous graph -- with query-query similarity edges, query-thinking edges, and thinking-answer edges -- a GNN trained on labeled rollouts can predict rewards for unlabeled rollouts during online RL fine-tuning. With only 20% ground-truth labels, MemReward achieves 97.3% of Oracle (100% labels) performance on Qwen2.5-3B and 96.6% on 1.5B. On out-of-domain tasks, it surpasses Oracle on average.

**Scale of evaluation**: 13 benchmarks spanning 3 domains: math (GSM8K, GSM-Symbolic, MATH), QA (MMLU, CSQA, OBQA, ARC-C, GPQA), code (HumanEval+, MBPP+). 3 out-of-domain benchmarks (NuminaMath, SIQA, PIQA). Two model scales (Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct). Compared against R1-p (20% labels, discard rest) and R1-Oracle (100% labels).

---

## Architecture / Method

### The Core Insight

Standard RL fine-tuning treats each query's rollouts independently -- a query either has labels or it doesn't. MemReward's insight is that semantically similar queries share reward patterns. If query A (labeled) is similar to query B (unlabeled), and their reasoning processes share structural features, A's reward signal can inform B's.

This is implemented as a heterogeneous graph with three node types and three edge types:

**Node types:**
- Query nodes (embedded from the query text)
- Thinking nodes (embedded from the reasoning trace)
- Answer nodes (embedded from the final answer)

**Edge types:**
- Query-query: top-k cosine similarity between query embeddings (k=7)
- Query-thinking: each query connects to its rollouts' thinking nodes
- Thinking-answer: each thinking node connects to its corresponding answer node

All edge weights initialized to 1. Embeddings computed via `all-MiniLM-L6-v2` (384-dim), projected to 512-dim hidden space via type-specific linear transformations.

### GNN Architecture

A 2-layer heterogeneous GNN with GATv2 attention (4 heads, dropout 0.1). Each layer updates node representations via type-specific message passing:

```
h_q^(l) = ReLU(Mean(sum a^qq W_qq^(l) h_q^(l-1), sum a^tq W_tq^(l) h_t^(l-1)))
h_t^(l) = ReLU(Mean(sum a^qt W_qt^(l) h_q^(l-1), sum a^at W_at^(l) h_a^(l-1)))
h_a^(l) = sum a^ta W_ta^(l) h_t^(l-1)
```

Where `a^xy` are learnable edge-type-specific attention weights and `W^(l)` are direction-specific weight matrices. The key design choice: separate weight matrices for each edge direction (6 total: qq, tq, qt, at, ta, plus q-to-thinking not shown above). This captures that "query similar to query" is a different relationship than "query produced this reasoning."

### Reward Prediction

The final-layer embeddings for a query and its rollout (thinking + answer concatenated) produce a reward score via scaled dot product:

```
r_ij = (phi_q(h_qi^(L)) * phi_r([h_tij^(L) || h_aij^(L)])) / sqrt(d) + b
```

Where `phi_q` and `phi_r` are learnable linear projections into a shared d-dimensional space, `||` is concatenation, and `b` is a bias. Trained with binary cross-entropy loss against ground-truth correctness labels (0 or 1).

### Online Integration with GRPO

During GRPO training, for each query in a batch:
1. If the query has ground-truth labels, use them directly
2. If unlabeled, encode the query and rollouts, connect to the warmup graph via top-k similarity, propagate through the frozen GNN, threshold at 0.5

The GNN is trained once on labeled data (warmup phase) and frozen during online RL. No further graph updates occur during training. This is important -- it means the graph is static, not adaptive.

### What the Paper Acknowledges vs. What It Misses

**Acknowledged limitations:**
- Performance depends on the quality of query-query similarity edges (acknowledged implicitly via the homogeneous graph ablation)
- The GNN is trained once and frozen; no mechanism for the graph to improve during online training

**Unacknowledged limitations:**
- The GNN's 0.917 ROC-AUC sounds impressive but the paper never reports what happens when the GNN is wrong. A 0.917 ROC-AUC with 20% labels means roughly 8% of reward predictions are incorrect. These incorrect rewards are used to train the LLM -- the paper never analyzes whether GNN errors systematically bias the model (e.g., rewarding specific error patterns)
- The top-k=7 similarity threshold is never ablated. It's unclear whether the GNN is robust to noisy edges (dissimilar queries connected by accident)
- The out-of-domain result (surpassing Oracle) is presented as pure win, but it could indicate the GNN is regularizing the reward signal -- smoothing over noisy individual labels. If so, the mechanism is label smoothing via graph averaging, not genuine cross-query knowledge transfer
- The binary reward (correct/incorrect) is the simplest possible signal. The paper doesn't address partial correctness, which is where most real reward engineering happens

---

## Key Claims & Evidence

### Main Results (Table 1, in-domain, 10 benchmarks)

| Method | GSM8K | GSM-sym | MATH | MMLU | CSQA | OBQA | ARC-C | GPQA | HumanEval+ | MBPP+ | Avg | Delta |
|--------|-------|---------|------|------|------|------|-------|------|------------|-------|-----|-------|
| *Qwen2.5-1.5B* | | | | | | | | | | | | |
| R1-p (20%) | 77.11 | 62.89 | 44.44 | 53.33 | 70.22 | 68.67 | 71.56 | 20.00 | 38.46 | **55.00** | 62.72 | -7.75 |
| MemReward | **88.67** | **77.78** | **50.89** | **54.67** | **72.44** | 70.00 | **72.67** | **23.33** | **43.59** | **55.00** | 68.10 | -2.37 |
| R1-Oracle | 86.44 | 75.33 | 53.11 | 66.44 | 74.44 | 74.00 | 74.89 | 15.00 | 53.85 | 56.25 | 70.47 | 0 |
| *Qwen2.5-3B* | | | | | | | | | | | | |
| R1-p (20%) | 92.89 | 84.67 | 54.67 | 71.78 | 77.33 | 78.44 | 80.00 | 21.67 | 64.10 | **65.00** | 75.67 | -3.45 |
| MemReward | 92.89 | **96.44** | **61.11** | **72.00** | 74.44 | **81.78** | **81.44** | **30.00** | 61.54 | 63.75 | 77.02 | -2.10 |
| R1-Oracle | 92.89 | 90.22 | 60.33 | 72.22 | **79.11** | 83.11 | **84.00** | 30.00 | 71.79 | 73.75 | 79.12 | 0 |

### Out-of-Domain Results (Table 2)

| Method | NuminaMath | SIQA | PIQA | Avg | Delta |
|--------|-----------|------|------|-----|-------|
| *Qwen2.5-1.5B* | | | | | |
| R1-p | 31.56 | 72.67 | 72.22 | 58.81 | -3.19 |
| MemReward | 34.67 | **74.44** | **79.33** | **62.81** | **+0.81** |
| R1-Oracle | 32.00 | 74.89 | 79.11 | 62.00 | 0 |
| *Qwen2.5-3B* | | | | | |
| R1-p | 36.44 | 74.67 | 82.22 | 64.44 | -1.63 |
| MemReward | **42.22** | **76.89** | 81.78 | **66.96** | **+0.89** |
| R1-Oracle | 39.33 | **76.89** | **82.00** | 66.07 | 0 |

### GNN Prediction Quality (Table 7, Appendix D.4)

| Dataset | Domain | Acc | Prec | Recall | ROC-AUC |
|---------|--------|-----|------|--------|---------|
| GSM8K | Math | 0.873 | 0.843 | 0.789 | 0.946 |
| GSM-Sym | Math | 0.883 | 0.725 | 0.669 | 0.936 |
| MATH | Math | 0.890 | 0.672 | 0.744 | 0.936 |
| MMLU | QA | 0.858 | 0.642 | 0.531 | 0.899 |
| CSQA | QA | 0.838 | 0.667 | 0.724 | 0.892 |
| OBQA | QA | 0.843 | 0.658 | 0.752 | 0.896 |
| ARC-C | QA | 0.868 | 0.756 | 0.663 | 0.913 |
| GPQA | QA | 0.923 | -- | -- | 0.843 |
| HumanEval+ | Code | 0.692 | 0.485 | 0.532 | 0.721 |
| MBPP+ | Code | 0.757 | 0.774 | 0.770 | 0.832 |
| **Overall** | | **0.861** | **0.717** | **0.704** | **0.917** |

### Ablation Results (Figure 3)

Three ablations on 3B:
- **MLP** (no graph, query embeddings only): Math 69%, QA 72%, Code 74%
- **Homogeneous graph** (single edge type): Math 74%, QA 75.6%, Code 77%
- **w/o thinking nodes**: Math 77.6%, QA 76.4%, Code 58%
- **Full model**: Math 80.1%, QA 75.6%, Code 63%

The heterogeneous graph consistently beats homogeneous, confirming that edge-type-specific message passing adds signal. The thinking node ablation is particularly telling -- removing reasoning traces hurts code (-5pp) and math (-2.5pp) most, consistent with these domains requiring multi-step reasoning.

### Label Ratio Scaling (Table 3)

| GT Ratio | Avg Score | Delta from Oracle |
|----------|-----------|-------------------|
| 20% | 77.02 | -2.10 |
| 30% | 77.56 | -1.56 |
| 40% | 77.53 | -1.59 |
| 50% | 77.95 | -1.17 |
| 60% | 78.19 | -0.93 |
| 70% | 78.64 | -0.48 |
| 100% (Oracle) | 79.12 | 0 |

Smooth scaling with diminishing returns. The jump from 20% to 30% gives 0.54pp; 60% to 70% gives 0.45pp; 70% to 100% gives only 0.48pp.

### Methodological Strengths

- Clean experimental design: same RL algorithm (GRPO), same hyperparameters, only the reward source changes
- Comprehensive ablations (graph structure, node types, label ratio)
- Out-of-domain evaluation genuinely tests generalization
- GNN prediction quality reported per-dataset, not just aggregated
- Code released

### Methodological Weaknesses

- **Binary reward only.** The entire framework assumes correct/incorrect. No partial credit, no continuous reward signal. This limits applicability to domains with unambiguous correctness
- **Static graph.** The warmup graph is frozen during online training. As the policy improves, the distribution of rollouts shifts, but the graph's reward predictions don't adapt. The paper never measures how the GNN's accuracy degrades over the 410 training steps as the policy drifts
- **Small models only.** Qwen2.5-1.5B and 3B. No evidence this helps at 7B+ where the base model is already strong enough that label efficiency matters less
- **The "surpasses Oracle" claim deserves scrutiny.** On out-of-domain, MemReward beats Oracle by 0.81-0.89pp. This is within noise for most benchmarks (5 runs with different seeds would clarify). The more parsimonious explanation is label smoothing: aggregating neighbors' labels reduces the variance of the reward signal, which helps RL stability. If so, this is a finding about RL training dynamics, not about memory or graphs per se
- **Top-7 nearest neighbors never ablated.** The paper uses k=7 throughout. At what k does performance degrade? Does the graph become too noisy at k=15? Too sparse at k=3? This is a free parameter that was apparently tuned without reporting the sensitivity
- **No comparison with simpler label propagation.** The obvious baseline is: for unlabeled queries, find k nearest labeled queries, average their labels, threshold. No GNN, no thinking nodes, no heterogeneous edges. If this simple baseline achieves 90% of MemReward's gains, the graph architecture is engineering for marginal improvement

---

## Relevance to Somnigraph

### What MemReward Does That We Don't

**1. Graph-based signal propagation via learned message passing.** Somnigraph's PPR in `scoring.py` propagates influence through the memory graph, but it uses fixed edge weights (modified by co-retrieval reinforcement, +0.02 per event). MemReward uses *learned* attention weights via GATv2, where the message-passing weights are trained to optimize a specific objective (reward prediction accuracy). Somnigraph's graph signal propagation is heuristic; MemReward's is optimized.

**2. Heterogeneous node and edge types in the graph.** Somnigraph's graph in `scoring.py` has one node type (memories) and edges with flags (contradiction, revision, derivation) but treats all nodes identically during PPR. MemReward distinguishes between query nodes, thinking nodes, and answer nodes with separate weight matrices for each relationship direction. This is architecturally richer -- a memory's relationship to a query that recalled it is fundamentally different from its relationship to a sibling memory.

**3. Explicit reward prediction from graph context.** MemReward's GNN produces a scalar reward prediction for each rollout by combining the query embedding with the response embedding via the graph-enriched representations. Somnigraph's feedback loop in `events.py` collects explicit utility scores but doesn't predict what score a new, unrated memory *would* get. The `feedback_mean` EWMA in `reranker.py` is an average of observed ratings, not a prediction for unseen contexts.

### What We Already Do Better

**1. Continuous, context-dependent feedback.** MemReward uses binary reward (correct/incorrect). Somnigraph's feedback loop collects continuous utility scores (0.0-1.0) per query-memory pair, preserving the context-dependent nature of relevance. The utility calibration study confirmed per-query Spearman r=0.70, which means Somnigraph's feedback signal is far richer than MemReward's binary labels.

**2. Adaptive graph that evolves with use.** MemReward's graph is built once from initial rollouts and frozen during training. Somnigraph's graph evolves: edges are created during NREM sleep (`sleep_nrem.py`), weights are reinforced through Hebbian co-retrieval (`events.py`, +0.02 per co-useful retrieval), and the graph grows organically as new memories are added. This is fundamentally more adaptive than a static warmup graph.

**3. The feedback loop itself.** MemReward's entire contribution is solving the label scarcity problem -- propagating labels from 20% of queries to the rest. Somnigraph doesn't have this problem because it collects explicit feedback on every recall. The feedback loop (explicit utility scores aggregated via EWMA with empirical Bayes Beta prior) is the mechanism that makes label scarcity irrelevant. Somnigraph has 100% "labels" on every retrieved memory.

**4. Production reranker with real-world features.** The LightGBM reranker in `reranker.py` uses 26 features including feedback signals, graph signals, metadata, and query-dependent features -- all trained on 1032 real queries. MemReward's GNN has a simpler objective (binary correctness prediction) and operates in a controlled benchmark setting, not a production environment with evolving data.

---

## Worth Stealing (ranked)

### 1. Predicted feedback for unrated memories

**What**: Use graph context to predict what utility score an unrated memory would receive if it were retrieved for a given query, rather than relying solely on the EWMA of past ratings.

**Why it matters**: The `feedback_mean` feature in `reranker.py` is undefined for memories that have never been retrieved (the UCB exploration bonus partially addresses this, but it's a heuristic). MemReward's core idea -- using neighbor signals to predict rewards for unlabeled items -- could fill this gap. 352 of 1120 memories have never been retrieved; 217 have only 1-2 appearances. These memories are essentially in MemReward's "unlabeled" category.

**Implementation**: During reranker feature extraction in `reranker.py`, for memories with fewer than N feedback events, compute a "predicted feedback" feature as the weighted average of feedback_mean from PPR-adjacent memories (already available via `_compute_proximity()` and the PPR cache). Weight by PPR score or edge weight. This is a much simpler version of MemReward's GNN propagation but captures the same intuition: neighbors' feedback is informative for under-observed memories.

**Effort**: Low-Medium. The PPR scores and neighbor feedback data are already computed in `tools.py`. The new feature is a weighted average over existing data. Adding it as feature #32 or replacing the UCB bonus for cold-start memories would require one training iteration.

### 2. Query-response separation in graph modeling

**What**: MemReward's heterogeneous graph distinguishes queries from responses (thinking + answer). Somnigraph's graph treats memories as undifferentiated nodes, but recall events could be modeled as a bipartite relationship: queries on one side, memories on the other.

**Why it matters**: The Hebbian co-retrieval signal in `events.py` currently tracks memory-memory co-occurrence. MemReward's structure suggests that query-memory affinity (which queries consistently retrieve which memories) is a separate, potentially more informative signal. This connects to the open question about vocabulary co-adaptation -- if certain queries always retrieve certain memories, that's either good (strong relevance) or bad (habitual retrieval pattern).

**Implementation**: Log query embeddings alongside recall events in `events.py`. During reranker feature extraction, compute query-memory affinity as a new feature: for a given query, what is its embedding similarity to the "average query" that has historically retrieved each candidate memory? This is conceptually similar to collaborative filtering.

**Effort**: Medium. Requires schema change (storing query embeddings in recall events), a new table or column in the DB, and a new feature in `reranker.py`. The payoff is uncertain -- it may duplicate what `vec_dist` already captures.

### 3. Score separation as a quality metric for feedback predictions

**What**: MemReward reports that GNN-predicted scores have a 0.51 separation between correct (mean 0.63) and incorrect (mean 0.11) rollouts. This separation metric is a useful diagnostic for any prediction system.

**Why it matters**: Somnigraph's feedback_mean EWMA produces scores for memories but there's no diagnostic for whether these scores actually separate relevant from irrelevant memories for a given query. The utility calibration study measured per-query correlation (r=0.70) but not score separation. A score separation metric would be a standing diagnostic for the feedback system's health.

**Implementation**: During GT evaluation in `scripts/build_ground_truth.py` or a new diagnostic script, compute the mean feedback_mean for GT-relevant vs. GT-irrelevant memories per query. Report the average separation. This is a direct analog of MemReward's Figure 6.

**Effort**: Low. Pure analysis on existing data. No code changes needed in production.

---

## Not Useful For Us

### The GNN architecture itself

MemReward's heterogeneous GNN with GATv2 attention is designed for a batch training setting where thousands of similar queries exist in a fixed dataset. Somnigraph operates on ~730 unique memories with ~1032 distinct queries, and the data arrives sequentially over months. Training a GNN on this data would be massively overfitting, and the computational overhead (PyTorch Geometric, GPU inference) is disproportionate to the ~5ms SQLite queries that currently drive retrieval. The LightGBM reranker achieves +6.17pp NDCG with tabular features -- adding a GNN to the scoring path would add latency and complexity for uncertain marginal gain.

### The semi-supervised label propagation framing

The entire paper is motivated by label scarcity in RL training. Somnigraph doesn't have label scarcity -- every recall event produces explicit feedback via `recall_feedback()`. The 100% label coverage rate (for retrieved memories) means the core problem MemReward solves is structurally absent. The *cold-start* problem for unrated memories is related but much smaller in scope and better addressed by the UCB exploration bonus or the simpler predicted-feedback approach described in "Worth Stealing" #1.

### Binary reward signals

MemReward's rewards are binary (correct/incorrect). Somnigraph's utility scores are continuous (0.0-1.0), context-dependent, and aggregated via EWMA with an empirical Bayes prior. Downgrading to binary would lose the nuance that makes feedback the dominant reranker feature. The continuous signal is strictly more informative.

### The GRPO / RL fine-tuning context

MemReward is about training reward models for RL fine-tuning of language models. Somnigraph is a retrieval system that serves a fixed (non-fine-tuned) Claude Code instance. The RL training loop, policy gradient optimization, and advantage estimation that MemReward operates within are entirely outside Somnigraph's scope.

---

## Impact on Implementation Priority

### Minimal impact on current priorities

This paper doesn't change the priority ordering in STEWARDSHIP.md. The main insights are:

- **P2 (Reranker iteration)**: The "predicted feedback for cold-start memories" idea (Worth Stealing #1) is a natural addition to the 31-feature retrain. It's a new feature, not a new experiment. Could be added alongside `query_length`, `candidate_pool_size`, and the other pending features.

- **Roadmap #5 (Counterfactual coverage check)**: MemReward's finding that graph-propagated rewards can predict labels for unseen items is tangentially relevant. The counterfactual check asks "what relevant memories does the retriever never surface?" MemReward's approach of using neighbor signals to infer quality of unseen items is conceptually related but operationally different -- Somnigraph's counterfactual check is about retrieval coverage, not reward prediction.

- **Open question: "Do memories converge to attractor states?"**: MemReward's static graph assumes queries and their similarities are stable. Somnigraph's concern about attractor dynamics (always-retrieved vs. never-retrieved basins) is about the opposite problem -- when the system is *too* adaptive and path-dependent. MemReward's static graph avoids this by design but also can't learn from new data.

- **Roadmap #10 (Prospective indexing)**: MemReward's query-query similarity edges are a retrieval-time analog of prospective indexing. Both try to connect items that should share signals. But prospective indexing operates at write time (enriching embeddings with hypothetical future queries), while MemReward operates at scoring time (connecting queries via embedding similarity). The approaches are complementary, not competing.

---

## Connections

### HippoRAG (PPR-based retrieval)

Both MemReward and HippoRAG use graph structure to propagate signals across related items. HippoRAG uses PPR over an open knowledge graph for multi-hop retrieval; MemReward uses a heterogeneous GNN over an experience graph for reward prediction. The key difference: HippoRAG's graph connects *entities* extracted from documents; MemReward's graph connects *queries* by embedding similarity. Somnigraph's PPR in `scoring.py` is closer to HippoRAG's approach (memory-to-memory edges with linking context) than to MemReward's (query-to-query similarity edges).

### Mem-alpha (RL-trained memory management)

Mem-alpha uses GRPO to train a 4B model to make optimal store/update/delete decisions for memory management. MemReward uses GRPO as the downstream RL algorithm that consumes the GNN's reward predictions. Both papers operate in the GRPO ecosystem but at different levels -- Mem-alpha optimizes the *memory management policy*, MemReward optimizes *reward prediction for a reasoning policy*. Neither directly addresses retrieval quality.

### Memory-R1 (RL-trained retrieval)

Memory-R1 trains a two-agent system with RL rewards to optimize memory retrieval. MemReward doesn't optimize retrieval at all -- it optimizes reward prediction for reasoning. But both share the insight that graph structure between queries/memories carries transferable signal. Memory-R1's approach of training on 152 QA pairs and generalizing to full benchmarks is analogous to MemReward's 20%-label setting.

### The expansion-wip findings (6 expansion methods, vocabulary gap)

MemReward's query-query similarity edges (top-7 cosine neighbors) are essentially an expansion method -- they bring reward signals from similar queries to bear on the current query. The multi-hop failure analysis finding (88% of evidence turns have zero content-word overlap with the query) suggests that embedding similarity between queries may not bridge the vocabulary gap either. MemReward's success in a controlled benchmark setting, where similar queries genuinely share reward patterns, doesn't guarantee that query similarity would help with the vocabulary-gap problem that dominates Somnigraph's multi-hop failures.

---

## Summary Assessment

MemReward is a clean, well-executed paper that applies graph-based semi-supervised learning to a real problem in RL fine-tuning: label scarcity. The heterogeneous GNN design is sound, the ablations demonstrate that each architectural choice contributes, and the out-of-domain generalization result is genuinely interesting (even if the mechanism is likely regularization rather than knowledge transfer). The paper does what it claims -- approaching Oracle performance with 20% labels -- and the evidence supports the claim.

For Somnigraph, the paper's direct applicability is limited. The core problem it solves (label scarcity for RL reward signals) doesn't exist in a system with explicit per-query feedback on every retrieval. The GNN architecture is too heavyweight for a 730-memory SQLite-based system. The binary reward signal is a step backward from continuous utility scores. Where the paper is useful is as an existence proof that graph-propagated signals can predict quality for unseen items -- this directly motivates a simpler "predicted feedback" feature for cold-start memories in the reranker, using PPR-neighbor feedback as a proxy. That single feature idea is the main takeaway.

The most intellectually interesting aspect is the out-of-domain result: MemReward surpasses Oracle by averaging reward signals across similar queries, effectively smoothing out noise in individual labels. This is a cautionary tale for Somnigraph's feedback loop -- per-query feedback (r=0.70) is strong but not perfect, and some form of neighbor-smoothed feedback might capture the remaining 30% of variance. Whether this is worth pursuing depends on whether the `feedback_mean` EWMA already provides sufficient smoothing via the empirical Bayes Beta prior.
