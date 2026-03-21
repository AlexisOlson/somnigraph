# Ori-Mnemos: Source Analysis

**Phase 14 (v0.3.4: 2026-03-06) · Updated v0.5.0: 2026-03-21 | Repo: [aayoawoyemi/Ori-Mnemos](https://github.com/aayoawoyemi/Ori-Mnemos)**

---

## 1. Architecture Overview

**Stack**: TypeScript (93.6%), Node.js, better-sqlite3 for embedding/boost/Q-value storage, `@huggingface/transformers` for local embeddings (all-MiniLM-L6-v2, 384-dim), graphology for graph algorithms, MCP server (stdio, SDK v1.27.1).

**Stars**: ~100+ | **License**: Apache-2.0 | **Version**: 0.5.0 | **Author**: aayoawoyemi (Ayo Awoyemi)

**Core design**: Memory as markdown files on disk, organized in a vault structure. Wiki-links (`[[Note]]`) serve as graph edges. No cloud dependencies — embeddings run locally. SQLite stores embeddings, activation boosts, Q-values, co-occurrence edges, stage learning parameters, and metadata hashes; the actual memory content lives in `.md` files with YAML frontmatter.

**Module organization** (~35 core files, up from 20 in v0.3.4):

Original modules (v0.3.4):
- `vitality.ts` — ACT-R decay with metabolic rates, structural boost, revival, bridge protection
- `graph.ts` — Wiki-link parser, builds directed LinkGraph
- `importance.ts` — graphology integration: PageRank, Louvain communities, Tarjan articulation points, betweenness centrality, personalized PageRank
- `fusion.ts` — Score-weighted RRF combining three signal channels
- `activation.ts` — Spreading activation with BFS, SQLite-persisted boosts
- `engine.ts` — Embedding pipeline, multi-space composite search (6 spaces)
- `bm25.ts` — Custom BM25 implementation with field weighting
- `intent.ts` — Heuristic query intent classification (episodic/procedural/semantic/decision)
- `tracking.ts` — IPS (Inverse Propensity Scoring) access logging and exploration injection
- `promote.ts` — Inbox-to-notes capture pipeline with auto-classification and link detection
- `classify.ts` — Pattern-based note type classification (idea/decision/learning/insight/blocker/opportunity)
- `linkdetect.ts` — Wiki-link detection, suggestion (5 signals)
- `ranking.ts` — Score containers and sort utilities
- `config.ts` — YAML config with extensive defaults

New modules (v0.5.0):
- `explore.ts` — **Recursive Memory Harness**: PPR graph traversal + LLM sub-question decomposition with convergence detection
- `qvalue.ts` — **Layer 1**: Q-value learning via EMA with UCB-Tuned exploration
- `reward.ts` — Behavioral reward inference: forward citation, downstream creation, re-recall, dead-end penalty
- `cooccurrence.ts` — **Layer 2**: Co-occurrence edges with NPMI, Ebbinghaus decay, Turrigiano homeostasis
- `stage-learner.ts` — **Layer 3**: LinUCB contextual bandits for pipeline stage selection
- `stage-tracker.ts` — Quality snapshot before/after each stage
- `rerank.ts` — Phase B Q-value reranking with lambda blend and bias cap
- `dampening.ts` — Three post-fusion corrections: gravity, hub, resolution
- `warmth.ts` — Spreading activation via PPR with surprise detection
- `warmth-audit.ts` — Optional JSONL audit trail for warmth decisions
- `ppr.ts` — Combined wiki-link + co-occurrence PPR (separate from `importance.ts`)
- `llm.ts` — LLM provider abstraction (Anthropic, OpenAI-compat, NullProvider)
- `noteindex.ts` — Shared frontmatter index + vitality computation
- `schema.ts` — Template validation for frontmatter
- `state.ts` — Onboarding state tracking
- `agents/protocol.ts` — Agent protocol helpers

**Vault zones**: Three metabolic spaces with different decay multipliers:
- `self/` — Identity (0.1x decay rate, nearly permanent)
- `notes/` — Knowledge (1.0x baseline)
- `ops/` — Operations (3.0x accelerated decay)

**Tools**: 14 MCP tools, 16 CLI commands, 396 tests. Benchmarks: `bench/hotpotqa-eval.ts`, `bench/locomo-eval.ts`.

**Marketing context**: The author published a website (March 2026) branding the system as "Recursive Memory Harness" (RMH), citing the MIT CSAIL RLM paper (Zhang, Kraska, Khattab, arXiv:2512.24601) as theoretical inspiration. The RLM paper describes an inference paradigm where LLMs recursively self-call via REPL environments to process arbitrarily long prompts — the connection to Ori's sub-question decomposition is philosophical (recursion applied to retrieval), not mechanical (Ori doesn't use the RLM REPL framework). The headline benchmark (90% R@5 vs Mem0's 29% on HotpotQA) compares Ori's recursive multi-pass retrieval with LLM calls against Mem0's single-pass vector search — expected advantage for multi-hop, not apples-to-apples.

---

## 2. Memory Type Implementation

### Schema

Memories are markdown files with YAML frontmatter:

```yaml
---
type: idea | decision | learning | insight | blocker | opportunity
status: inbox | active | archived
project: [project-name]
description: "One-line summary"
tags: [tag1, tag2]
created: 2026-03-01
last_accessed: 2026-03-05
access_count: 7
---

Body content with [[wiki-links]] to other notes.
```

The `type` field drives query-time behavior: the intent classifier maps query patterns to preferred types, which then weight the six scoring spaces differently.

### Memory Lifecycle

**Capture (inbox)**: Notes start in `inbox/` via `ori_add`. Raw capture, minimal metadata.

**Promote**: `ori_promote` moves notes from inbox to `notes/`, auto-enriching:
1. Pattern-based type classification (6 types, priority-ordered regex rules)
2. Project detection via keyword config
3. Wiki-link detection in body text (longest-first matching, offset tracking)
4. Structural link suggestion (5 heuristics: title-match, tag/project overlap, shared-neighborhood, semantic similarity)
5. Auto-apply high-confidence (>=0.8) link suggestions
6. Area routing via project-to-map configuration
7. Footer injection (relevant notes + areas sections, idempotent merge)

**Active**: Notes in `notes/` participate in retrieval, graph metrics, vitality scoring. Access count incremented on retrieval. Frontmatter `last_accessed` updated. Q-values and co-occurrence edges accumulate from session activity.

**Prune/Archive**: `ori_prune` identifies archive candidates using:
- Vitality zone classification (active >= 0.6, stale >= 0.3, fading >= 0.1, archived below)
- Protection: articulation points exempted, inDegree >= 2 exempted
- Dry-run by default, `--apply` to mutate
- Community hotspot reporting

---

## 3. Retrieval Mechanism

### Full Pipeline

The v0.5.0 retrieval pipeline has two modes: **flat** (`ori_query_ranked`, single-pass) and **explore** (`ori_explore`, recursive multi-pass).

#### Phase A: Fusion (both modes)

**Signal 1 — Composite Vector Search** (6 sub-spaces):
- **Text space**: Weighted split similarity across title/description/body embeddings. Split weights vary by intent (episodic favors body at 0.6; semantic favors title at 0.5).
- **Temporal space**: Exponential recency with 30-day half-life, encoded as piecewise-linear vector.
- **Vitality space**: Current ACT-R vitality score, encoded as piecewise-linear vector.
- **Importance space**: Normalized PageRank, encoded as piecewise-linear vector.
- **Type space**: One-hot encoded note type, matched against query-implied type vector.
- **Community space**: Hash-based community projection (sin/cos encoding with small primes).

Each sub-space produces a cosine similarity against a query-side "target" vector. The 6 scores are combined with intent-dependent weights:

```typescript
// Space weight profiles by intent
episodic:   { text: 0.40, temporal: 0.25, vitality: 0.15, importance: 0.05, type: 0.05, community: 0.10 }
procedural: { text: 0.30, temporal: 0.05, vitality: 0.10, importance: 0.30, type: 0.10, community: 0.15 }
semantic:   { text: 0.65, temporal: 0.05, vitality: 0.10, importance: 0.10, type: 0.05, community: 0.05 }
decision:   { text: 0.30, temporal: 0.15, vitality: 0.10, importance: 0.10, type: 0.30, community: 0.05 }
```

**Signal 2 — BM25 Keyword Search**: Custom implementation with field weighting (title 3x, description 2x, body 1x). Standard BM25 formula with k1=1.2, b=0.75.

**Signal 3 — Personalized PageRank**: Seeds from entity extraction (note titles found in query). 20 iterations of power iteration with alpha=0.85. Falls back to uniform personalization if no entities found.

**Fusion**: Score-weighted RRF:
```
score = Sum_s( signal_weight_s * raw_score_s / (k + rank_s + 1) )
```
Default weights: composite=2.0, keyword=1.0, graph=1.5. Default k=60.

#### Dampening Pipeline (v0.5.0)

Three post-fusion corrections, each with ablation-validated impact deltas:

1. **Gravity dampening** (−0.256 P@5 if removed): Halves score for results with high cosine similarity but zero query term overlap in the title. Catches "cosine similarity ghosts" — notes that embed similarly but don't contain the relevant information.

2. **Hub dampening** (−0.104 P@5 if removed): Penalizes notes in the top 10% by edge degree. Formula: `penalty = 1.0 - 0.6 * (degree - P90) / (maxDegree - P90)`, minimum 0.2. Entity-matched notes are exempt.

3. **Resolution boost** (−0.144 P@5 if removed): 1.25× score for notes typed as decision, learning, procedural, fix, or solution. Surfaces actionable knowledge.

#### Phase B: Q-Value Reranking (v0.5.0)

After Phase A fusion, the top ~40 candidates are reranked using a blend of similarity and learned Q-value:

```
score = (1 - λ) * z_norm(simScore) + λ * z_norm(qScore) + UCB_bonus
```

- **Lambda**: Grows from 0.15 to 0.50 as the system matures (first 200 Q-value updates). Query type shifts: semantic −0.10, procedural +0.15, decision +0.05.
- **Z-score normalization** of both similarity and Q-values is marked as "CRITICAL" in the code — without it, lambda is meaningless since the two score distributions have different scales.
- **Cumulative bias cap**: `max(score, baseScore * 3.0) + excess * 0.3` prevents runaway inflation from Q-value compounding.
- **Exposure tracking**: Every surfaced note's exposure counter is incremented. Used for reward correction in Layer 1.
- Returns top 8 results.

#### Explore Mode: Recursive Sub-Question Decomposition (v0.5.0)

The "Recursive Memory Harness" — the headline new feature:

1. **Phase 1** (non-recursive): Run flat retrieval + PPR walk from top-k results at α=0.45 (explore-tuned, different from flat's α=0.85). Seed weights blend retrieval score, warmth activation, and Q-value: `base + (0.2 * warmth) + (0.1 * (q - 0.5))`.

2. **Phase 3** (recursive): Send found notes to LLM with prompt: "Here's what we found. What gaps remain?" LLM generates 1–3 sub-questions. Each sub-question re-seeds from the full vault via the same fusion pipeline. New notes accumulated.

3. **Convergence detection**: Stop when `newNotes / totalNotes < 0.15` (configurable). Also stops at `max_recursion_depth` (default 3) and `max_total_notes` budget.

4. **Graceful degradation**: No LLM configured → Phase 1 only (claimed "95% recall").

Every pass generates learning signals — retrievals feed Q-values and co-occurrence updates at session end.

**Post-fusion**:
- Archive filtering
- Exploration injection: bottom 10% of results replaced with random unseen notes (IPS-inspired diversity)
- Access logging (JSONL)
- Spreading activation: top 3 results propagate boosts to graph neighbors

---

## 4. Standout Feature: Three-Layer Retrieval Learning

The most significant evolution from v0.3.4 to v0.5.0 is the addition of three independent learning layers that close the retrieval-feedback loop — the gap our original analysis identified as the primary weakness.

### Layer 1: Q-Value Learning (`qvalue.ts`, `reward.ts`)

Each note maintains a Q-value (initialized at 0.5) updated via exponential moving average:

```
newQ = oldQ + α * (reward - oldQ)    where α = 0.1
```

**Behavioral reward inference** — the key design difference from explicit feedback systems:

| Signal | Reward | Condition |
|--------|--------|-----------|
| Forward citation | +1.0 | Retrieved note cited in new content via `[[wiki-link]]` |
| Update after retrieval | +0.5 | Note was edited after being retrieved |
| Downstream creation | +0.6 | New content created after retrieval (position-weighted by rank) |
| Within-session re-recall | +0.4 | Same note retrieved again (diminishing per occurrence) |
| Partial follow-up | +0.1 | Some follow-up action, but not involving this note |
| Dead end | −0.15 | Top-3 result, no follow-up action (IPS-debiased) |

All rewards are exposure-corrected: `reward / exposure_count^0.5`. This prevents frequently-surfaced notes from accumulating inflated Q-values.

**Q-informed decay**: High-Q notes (≥0.7) decay 0.7× slower; low-Q notes (≤0.3) decay 1.3× faster. Half-life ~99 days at baseline.

**UCB-Tuned exploration**: `c * sqrt((log(T) / count) * min(0.25, variance))` where c=0.2. Encourages trying notes with uncertain value.

Q-values, rewards, and retrievals are all logged to SQLite tables with full history (`q_history`, `retrieval_log`).

### Layer 2: Co-Occurrence Learning (`cooccurrence.ts`)

Edges grow organically between notes co-retrieved in the same session. Weight formula:

```
weight = NPMI * GloVe_freq * trustWeight * Ebbinghaus_decay
```

**NPMI** (Normalized Pointwise Mutual Information): `log(P(A,B)/(P(A)*P(B))) / (-log(P(A,B)))`. Range [−1, 1]. More principled than raw co-occurrence count.

**GloVe frequency scaling**: `(count/100)^0.75` if count < 100, else 1.0. From the GloVe paper (Pennington et al. 2014) — prevents high-frequency co-occurrences from dominating.

**Ebbinghaus decay**: `exp(-daysSince / (30 * strength))` where `strength = 1 + 0.2 * log(1 + coRetrievalCount)`. Stronger edges (more co-retrievals) decay slower. Floor at 0.05.

**Turrigiano homeostatic scaling**: Per-node mean edge weight is clamped to 0.5. All edges touching a node are scaled up or down to maintain this target. From neuroscience (Turrigiano & Nelson 2004, synaptic scaling) — prevents any single node from dominating through accumulated edge weight. This is a principled anti-rich-get-richer mechanism.

**Bootstrap**: Co-occurrence edges are initialized from wiki-link structure via bibliographic coupling (Jaccard similarity > 0.1 on shared outgoing links), seeding the graph before any retrieval activity.

### Layer 3: Stage Meta-Learning (`stage-learner.ts`)

LinUCB contextual bandits (Li et al. 2010) learn which retrieval pipeline stages to run for each query type.

**Feature vector** (8D):
```
[tokenLength/50, log(uniqueTerms)/10, hasQuestion?, hasTemporal?,
 hasEntity?, embeddingEntropy/10, vaultSize/1000, queryDepth/10]
```

**9 stages**, 2 essential (never skipped), 7 learnable:
- Essential: `semantic_search`, `rrf_fusion`
- Learnable: `bm25`, `pagerank`, `warmth`, `hub_dampening`, `gravity_dampening`, `q_reranking`, `cooccurrence_ppr`

**Decision per stage**: run / skip / abstain. UCB score determines whether a stage's expected quality improvement justifies its compute cost.

**Cost-sensitive reward**: `-0.2 * (computeTimeMs / 100)` penalty applied to reward. Expensive stages need higher UCB to justify running.

**Curriculum**: First 50 samples run all stages (exploration). After 50, optimization phase skips stages with low UCB for the given query features.

**Load balancing**: Coefficient of variation penalty prevents stage dominance (from Shazeer's MoE load balancing).

### Session-End Batch Update

All three layers are updated in a single SQLite transaction at session end:
1. Co-occurrence: Extract pairs from `retrieval_log`, update NPMI weights, run homeostasis
2. Q-values: Compute behavioral rewards, update EMA
3. Stage learning: Update LinUCB A-matrices and b-vectors from stage quality snapshots

This is not offline consolidation (no LLM judgment, no summarization, no relationship classification), but it is a form of online learning that closes the feedback loop.

---

## 5. Other Notable Features

### 5.1 ACT-R Vitality Model with Metabolic Zones and Structural Protection

*(Unchanged from v0.3.4 — see original analysis. The core innovation: ACT-R base-level activation with metabolic rate modulation per vault zone, six-component vitality scoring, and Tarjan articulation-point protection for graph-critical nodes.)*

The ACT-R base-level activation equation:
```
B_i = ln(n / (1 - d)) - d * ln(L)
```
Normalized to [0, 1] via sigmoid. Metabolic rates: self/ = 0.1× (near-permanent), notes/ = 1.0× (baseline), ops/ = 3.0× (fast decay). Six components: ACT-R base, structural boost (inDegree), access saturation, revival spike, spreading activation, bridge protection floor.

### 5.2 Warmth (Spreading Activation via PPR) (v0.5.0)

Replaces the simpler BFS spreading activation from v0.3.4:
- **Surprise detection**: Cosine distance between current query embedding and cached context embedding. If distance exceeds threshold → re-compute warmth signals.
- **PPR integration**: Seeds from semantic search weighted by warmth activation scores.
- **Audit logging**: Optional JSONL audit of warmth decisions (via `ORI_WARMTH_AUDIT` env var).

### 5.3 IPS Exploration Injection

*(Unchanged from v0.3.4.)* Bottom 10% of results replaced with random unseen notes. Propensity computed as `times_surfaced / total_queries` (floored at epsilon=0.01).

### 5.4 Knowledge-Enriched Embeddings

*(Unchanged from v0.3.4.)* Before embedding, notes are enriched with structural metadata:
```
// Line 1: [TYPE] [PROJECTS]
// Title
// Description
// Connected: note1, note2, ... (up to 10 outgoing links)
```

### 5.5 Louvain Community Detection

*(Unchanged from v0.3.4.)* Using graphology-communities-louvain. Communities used for scoring (16-dim hash projection) and pruning analysis (community hotspots).

---

## 6. Benchmarks

### HotpotQA (`bench/hotpotqa-eval.ts`, ~650 LOC)

Multi-hop QA benchmark (each question requires finding and combining information from two documents).

**Setup**: Creates temp vault from 10 context paragraphs (2 gold, 8 distractors). Runs three methods head-to-head on same 50 questions:
1. **Flat** (`ori_query_ranked`): BM25 + semantic + PPR → RRF fusion
2. **Explore Phase 1** (`ori_explore`): Same fusion, then PPR walk from top-k results
3. **Recursive** (`ori_explore` + sub-questions): Phase 1 + LLM decomposition loop

**Metrics**: Recall, Precision, F1, MRR, answer recall proxy (token overlap). Optional LLM judge (GPT-4.1-mini) for answer quality.

**Claimed results**:
| Metric | Ori (recursive) | Mem0 |
|--------|-----------------|------|
| R@5 | 90.0% | 29.0% |
| F1 | 52.3% | 25.7% |
| LLM-F1 | 41.0% | 18.8% |
| Speed | 142s | 1347s |
| Ingestion API calls | 0 (local) | ~500 LLM calls |

**Caveat**: Mem0 baseline comes from external evaluation, not Ori's harness. Ori uses recursive multi-pass with LLM calls; Mem0 uses single-pass vector retrieval. The comparison demonstrates the value of recursive decomposition for multi-hop, not a controlled comparison of memory architectures at equivalent compute.

### LoCoMo (`bench/locomo-eval.ts`, ~800 LOC)

Long conversation memory benchmark (10 conversations, 695 questions).

**Setup**: Each conversation turn becomes a note. Three retrieval methods same as HotpotQA. LLM answer generation via GPT-4.1-mini.

**Results**: 37.69 single-hop, 29.31 multi-hop. Mem0 on same benchmark: 38.72 single-hop, 28.64 multi-hop. **Essentially parity** — the website does not emphasize this comparison.

---

## 7. Gap Ratings (Updated for v0.5.0)

### Layered Memory: 70%
*(Unchanged.)* Three metabolic zones, inbox/active/archived lifecycle. No true episodic vs. semantic distinction.

### Multi-Angle Retrieval: 95% *(was 90%)*
Exceptional, now even stronger. Three-signal fusion via score-weighted RRF, intent-adaptive 6-space scoring, plus dampening pipeline, Phase B Q-reranking, and recursive sub-question decomposition in explore mode.

### Contradiction Detection: 5%
*(Unchanged.)* No contradiction detection. No revision tracking.

### Relationship Edges: 80% *(was 75%)*
Wiki-links as implicit edges, now supplemented by co-occurrence edges that grow organically from retrieval patterns. Co-occurrence edges carry NPMI weights with Ebbinghaus decay and homeostatic scaling. Still no typed edge metadata (no linking_context, no flags, no edge types like contradiction/derivation).

### Sleep Process: 20%
*(Unchanged.)* No offline consolidation. The session-end batch update (Q-values, co-occurrence, stage learning) is online learning, not consolidation. No NREM-style merging, no REM-style gap analysis.

### Reference Index: 60%
*(Unchanged.)* Community-based organization, map notes, area routing.

### Temporal Trajectories: 45%
*(Unchanged.)* Temporal space with 30-day recency half-life. No temporal clustering or evolution tracking.

### Confidence/UQ: 35% *(was 15%)*
Significant improvement. Q-values provide per-note confidence derived from behavioral signals. UCB-Tuned exploration bonus quantifies uncertainty. Exposure tracking enables debiasing. Stage learning provides per-stage confidence via LinUCB posterior. Still no per-memory confidence score visible to the user, and no contradiction-driven confidence degradation.

### Feedback Loop: 75% *(new dimension)*
Three independent learning layers closing the retrieval-feedback loop. Layer 1 (Q-values) learns from behavioral signals; Layer 2 (co-occurrence) learns from co-retrieval patterns; Layer 3 (stage learning) optimizes the pipeline itself. All three update at session end. However: no explicit user feedback mechanism (all signals inferred from behavior), no per-query grading, no durability dimension, and the system has no way to learn from the agent saying "this was useful" vs. inferring it from downstream actions.

---

## 8. Comparison with claude-memory (Updated for v0.5.0)

### Stronger (Ori-Mnemos vs. claude-memory)

**Graph algorithm sophistication**: *(Unchanged.)* Full PageRank, PPR, Louvain communities, betweenness centrality, Tarjan articulation points. More graph-theoretic depth than our novelty-scored adjacency expansion.

**Multi-space composite scoring**: *(Unchanged.)* 6-space composite search with intent-adaptive weights. More granular than our 2-channel RRF.

**Structural decay protection**: *(Unchanged.)* Tarjan articulation-point protection. Metabolic rate modulation.

**Exploration injection**: *(Unchanged.)* IPS-based diversity mechanism counters popularity bias.

**Local embeddings**: *(Unchanged.)* No API key required, no external dependency.

**Behavioral reward inference** *(new)*: Ori's three-layer learning system infers feedback from downstream behavior (citations, edits, re-recalls) rather than requiring explicit grading. This eliminates the cognitive burden on the agent — no `recall_feedback()` call needed, no utility scoring. The tradeoff: behavioral signals are noisier (correlation with ground truth is unmeasured) but capture rewards that explicit feedback misses (the user cited a memory without being asked to rate it).

**Dampening pipeline** *(new)*: The three post-fusion corrections (gravity, hub, resolution) address known failure modes with ablation-validated impact. Gravity dampening specifically targets "cosine similarity ghosts" — a failure mode we haven't measured or addressed.

**Pipeline self-optimization** *(new)*: LinUCB stage selection means the system learns which retrieval stages to run per query type. This is a capability no other system in our corpus has.

### Weaker (Ori-Mnemos vs. claude-memory)

**No sleep/consolidation**: *(Unchanged.)* No offline LLM-mediated processing. No NREM-style merging, no REM-style gap analysis, no cross-memory edge inference with linking_context.

**No contradiction/revision tracking**: *(Unchanged.)* No mechanism for detecting conflicting information. No revision edges, contradiction flags, or temporal validity fields.

**No edge metadata**: Wiki-links are untyped, undirected in practice. Co-occurrence edges add NPMI weight but no semantic linking_context or edge type flags. Our edges carry rich metadata (linking_context with embeddings, flags, creation attribution).

**No durability/shadow modeling**: *(Unchanged.)* No concept of memory shadow load, no durability dimension in feedback, no confidence gradient from contradiction detection.

**No theme/category normalization**: *(Unchanged.)* Notes have types and tags, but no normalization pipeline or variant merging.

**Scale considerations**: The BM25 index is rebuilt from disk on every query. The composite search iterates over all stored vectors. Fine at hundreds of notes; concerning at thousands.

**Explicit feedback precision**: Our `recall_feedback()` achieves r=0.70 per-query correlation with ground truth. Ori's behavioral reward inference has no published correlation with ground truth. Explicit feedback is noisier per-event but aggregates more reliably because the signal is direct. Behavioral inference captures broader signal (every session contributes) but with unknown fidelity.

**Learned reranker**: Our LightGBM reranker learns from 1032 GT queries with 26 features, achieving +6.17pp NDCG over the formula. Ori's Phase B reranking uses a fixed lambda-blend formula (not learned), adapting only through Q-value maturity.

### Shared Strengths

**Cognitive science grounding**: Both use biologically-grounded decay models (ACT-R vs exponential with per-category half-lives).

**Hybrid retrieval**: Both combine vector similarity with keyword matching and graph signals via RRF-family fusion.

**Graph-aware memory**: Both maintain relationship graphs and use topology to influence retrieval and maintenance.

**Metadata-enriched embeddings**: Both embed more than just content.

**Zone/status lifecycle**: Both have multi-stage memory lifecycles.

**Feedback loop** *(new shared strength)*: Both now close the retrieval-feedback loop, though through fundamentally different mechanisms — explicit grading vs. behavioral inference.

---

## 9. Insights Worth Stealing (Updated)

### 9.1 Turrigiano Homeostatic Scaling for Edge Weights
**Effort**: Low | **Impact**: High

Per-node mean edge weight clamped to a target (0.5) by scaling all edges touching that node. Prevents rich-get-richer dynamics in co-retrieval PMI. Our Hebbian PMI system has exactly this problem (noted in the original analysis and in the "false bridge ratchet" open problem). Implementation: after each co-retrieval update batch, compute per-node mean weight, scale all edges touching over-weighted nodes proportionally. One SQL query + one batch update.

This is the single most actionable idea from the v0.5.0 update. The neuroscience grounding is solid (Turrigiano & Nelson 2004, synaptic scaling) and the mechanism directly addresses a known weakness.

### 9.2 Behavioral Reward Signals as Supplementary Feedback
**Effort**: Medium | **Impact**: Medium

The forward-citation signal (+1.0 when a retrieved memory is cited via wiki-link in subsequent `remember()` content) captures real utility without burdening the agent. We could detect when `remember()` content references themes or IDs from a recent `recall()` and log an implicit positive signal. This supplements `recall_feedback()`, not replaces it — our explicit feedback is more precise (r=0.70 with GT) but behavioral signals capture cases where the agent uses a memory without explicitly rating it.

### 9.3 Gravity Dampening (Cosine Similarity Ghost Detection)
**Effort**: Low | **Impact**: Medium

Halve score for results with high vector similarity but zero query term overlap. Easy to implement as a post-scoring adjustment. Worth measuring first: how many of our top-k results have high vector similarity but zero BM25 term overlap? If the rate is non-trivial, this is a cheap fix.

### 9.4 Personalized PageRank as Retrieval Signal
**Effort**: Medium | **Impact**: High

*(Unchanged from v0.3.4 analysis.)* Seeding PPR from entities detected in the query and using scores as a third retrieval signal. In v0.5.0, the explore mode uses α=0.45 with seed weights blending retrieval + warmth + Q-value — more sophisticated than our PPR plan in Tier 3.

### 9.5 Convergence Detection for Recursive Retrieval
**Effort**: Low | **Impact**: Low-Medium

The `newNotes / totalNotes < threshold` stopping criterion is simple and elegant. Applicable to our sleep REM question generation (stop generating questions when new passes yield diminishing new connections). Not for live retrieval (we don't do multi-pass) but for any iterative discovery process.

### 9.6 Tarjan Articulation-Point Protection for Decay
**Effort**: Medium | **Impact**: Medium

*(Unchanged from v0.3.4 analysis.)* Computing articulation points on our edge graph and giving them decay immunity.

### 9.7 ~~Exploration Injection / Anti-Popularity-Bias~~
**Effort**: Low | **Impact**: Low-Medium (downgraded)

With Turrigiano homeostasis (§9.1) as a principled anti-rich-get-richer mechanism, random injection at the tail of results is less necessary. The homeostasis approach is strictly better: it addresses the root cause (weight accumulation) rather than the symptom (result homogeneity).

---

## 10. What's Not Worth It

### Markdown-as-Database
*(Unchanged.)* O(n) filesystem reads for BM25 indexing and vitality computation.

### Local Embeddings (all-MiniLM-L6-v2)
*(Unchanged.)* 384-dim MiniLM is lower quality than text-embedding-3-small (1536-dim).

### Six-Space Composite Scoring
*(Unchanged.)* Theoretically elegant but adds complexity. Our reranker achieves equivalent functionality.

### LinUCB Stage Selection (for us specifically)
The contextual bandit approach to pipeline stage selection is novel and interesting as research, but our reranker already learns feature importance implicitly — a learned model over 26 features subsumes the question of "which signals matter." Stage selection addresses a different problem (compute optimization) that isn't a bottleneck at our scale.

### Recursive Sub-Question Decomposition (for live retrieval)
LLM calls per query are expensive and add latency. Our architecture deliberately defers LLM-intensive processing to sleep. The convergence detection mechanism is worth borrowing for sleep REM, but not for live `recall()`.

### Full Behavioral Reward System
The complete Q-value + reward + co-occurrence + stage learning stack is substantial (~1500 LOC). We already have explicit feedback (r=0.70 with GT) and a learned reranker. Replacing both with a behavioral inference system would be a lateral move, not an upgrade. The specific signals worth borrowing (§9.2: forward citation detection) are much simpler to implement à la carte.

---

## 11. Key Takeaway (Updated)

Ori-Mnemos v0.5.0 is a significant evolution from the "beautiful static architecture" we analyzed at v0.3.4. The addition of three learning layers (Q-values, co-occurrence with homeostasis, LinUCB stage selection) closes the feedback loop — the gap we identified as the primary weakness. The recursive sub-question decomposition is real and working, not vaporware.

The two systems now share more common ground than they differ on: both have hybrid retrieval, graph-aware scoring, decay with cognitive science grounding, and feedback loops. The philosophical difference has shifted from "static vs. adaptive" to "implicit vs. explicit feedback" and "online vs. offline consolidation."

**What Ori does better**: Graph algorithm sophistication (PPR, Tarjan, Louvain), behavioral reward inference (zero agent burden), pipeline self-optimization (LinUCB), and dampening for known failure modes.

**What we do better**: Explicit feedback precision (r=0.70 with GT), learned reranker (LightGBM, 26 features, +6.17pp), offline LLM-mediated consolidation (sleep pipeline), contradiction/revision tracking, and rich typed edge metadata.

**Most transferable contribution**: Turrigiano homeostatic scaling for edge weights — directly addresses our known rich-get-richer problem with a principled, low-effort mechanism.
