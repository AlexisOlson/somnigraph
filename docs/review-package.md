# Somnigraph: Review Package

A research-driven persistent memory system for Claude Code (Anthropic's CLI tool). Live, stable, used daily at personal scale (~730 memories, ~1,500 edges, 249K events). Shared publicly as a research artifact — the documentation and design decisions are the product; the code is proof they're not theoretical.

---

## Architecture

### Stack

SQLite with two extensions: **sqlite-vec** (float32 vector storage, cosine distance KNN) and **FTS5** (full-text search with BM25). Everything — vectors, full-text index, metadata, event log, graph edges — in one `memory.db` file. WAL mode for concurrent reads. No server process, no external dependencies beyond an OpenAI embedding API.

Embeddings: **text-embedding-3-small** (1536 dims). Enriched before embedding — content + category + themes + summary concatenated, so metadata is front-loaded into vector space (borrowed from A-Mem's multi-faceted embedding pattern). Embeddings are created once at write time and not updated afterward.

### Schema highlights

**memories**: Content, summary, category (episodic/semantic/procedural/reflection/meta), themes (JSON array), base_priority (1-10), decay_rate (per-memory override of category default), confidence (0.1-0.95, compounds through feedback), shadow_load (how much newer similar memories suppress this one), valid_from/valid_until (temporal bounds), status (active/pending/deleted), layer (detail/summary/gestalt — CLS hierarchy).

**memory_fts**: FTS5 on summary (5x BM25 weight) + themes (3x weight). Content deliberately excluded — rationale: content is noisy and long; summary and themes are curated signals.

**memory_edges**: Source, target, linking_context (NL explanation of the relationship), linking_embedding (vector of the linking context, intended for query-conditioned expansion), flags (contradiction/revision/derivation), weight (0-1, default 0.5, shaped by feedback). Created by sleep (NREM classification) or explicitly via `link()`.

**memory_events**: Append-only lifecycle log. Every retrieval, feedback signal, edge creation, status change recorded with timestamps, query text, co-retrieved memory IDs, similarity scores. 12,894 feedback events with mean utility 0.244.

### Retrieval pipeline

**Hybrid search**: Vector channel (KNN by cosine distance) + keyword channel (FTS5 BM25 with phrase-aware query sanitization). Known multi-word phrases preserved as quoted groups; remaining tokens joined with AND (≤3 groups) or OR (4+). Fused via Reciprocal Rank Fusion:

```
score(d) = 0.5 / (k + rank_vec(d)) + 0.5 / (k + rank_kw(d))
```

k=6 (production). Vector and keyword weights equal (0.5 each) — tuning found the system insensitive to this ratio.

Four post-RRF scoring stages, applied sequentially:

**1. Feedback boost** (dominant signal): Empirical Bayes Beta prior blends per-memory EWMA utility (alpha=0.3) with population mean. Beta(α,β) fitted from population feedback via method of moments; α+β ≈ 11 replaces old hard gate (which required 14+ events). Memories with few feedback events are pulled toward the population mean; memories with many events express their own signal. `FEEDBACK_COEFF = 5.15`. Boost = FEEDBACK_COEFF × (blended - prior_mean), so memories at the population mean get zero boost; above get positive, below get negative.

**2. Theme boost**: Multiplicative boost per overlapping theme between query boost_themes and memory themes. `THEME_BOOST = 0.19` (was 0.97 before PPR implementation — see constant evolution table). Minor at current value.

**3. Hebbian co-retrieval**: PMI-based boost for memories frequently retrieved together (30-day rolling window). Hub suppression: high-P(A) memories get low PMI with any specific partner. `HEBBIAN_COEFF = 3.0`, `HEBBIAN_CAP = 0.21`, `HEBBIAN_MIN_JOINT = 2`. Minor contribution (<1% importance when feedback works).

**4. PPR graph expansion**: Top 5 scoring memories (seeds) initiate Personalized PageRank walk across graph edges. `PPR_DAMPING = 0.775` (higher = more graph walk, less teleport back to seeds). `PPR_BOOST_COEFF = 2.0` (at search ceiling — true optimum may be higher). `PPR_MIN_SCORE = 0.007` (stricter than original 0.001). Max 20 new neighbors. PPR walks all edge types using feedback-shaped weights.

**Cliff detection**: Log-curve fit to sorted score sequence, cut at 2.0 standard deviations below prediction. Minimum 5 results before activation. Replaces fixed quality floor (removed — optimal ratio was 0.0).

### The feedback loop

`recall_feedback()` accepts utility scores (0-1) + optional durability signals (-1 to 1) for retrieved memories. Six effects:

1. **Confidence update**: Grows asymptotically for useful memories (util >= 0.5, growth rate 0.08), decays multiplicatively for useless ones (util < 0.1, decay rate 0.07). Range 0.1-0.95. Default 0.5. Durability signal nudges ±0.05.

2. **Edge weight adjustment**: +0.02 when both endpoints useful in same retrieval, -0.01 when either is useless. Asymmetric — easier to weaken an edge than strengthen one.

3. **Decay rate adjustment**: Durability signal scales decay rate ±20% max (DECAY_DURABILITY_SCALE = 0.2).

4. **Theme enrichment**: Utility >= 0.6 adds query terms as bridge themes on the memory (max 3 new terms per cycle, max 12 total themes, minimum term length 4 chars).

5. **Co-utility edge creation**: When two memories are both rated >= 0.7 in the same retrieval, a co-utility edge is created between them.

6. **Event logging**: Full context recorded for offline analysis.

### startup_load

Priority-ranked briefing within a token budget (default 3000 tokens). Loads all active memories, scores by effective priority (base_priority × decay factor). Separates questions (category=meta, guaranteed slots) from regular memories. Corrections get interleaved but capped at 40% of slots for diversity. Packs greedily within budget with footer summary of what was excluded.

### The graph

**Shadow load**: When a newer memory has cosine similarity > 0.7 with an older one, the older accumulates shadow load. Informs dormancy decisions during sleep. Was in scoring pipeline, removed (confuses temporal vs. query relevance).

**Temporal evolution**: Newer memory updates older fact → set valid_until on older, create revision-flagged edge. Both remain queryable. Preserves narrative arc. Simplified from Zep/Graphiti's 4-timestamp bi-temporal model to 2 fields.

**Edge lifecycle**: Edges are created during sleep (NREM classification of memory pairs) or explicitly via `link()`. Weights adjust through feedback when both endpoints are co-retrieved. No other degradation mechanism exists for edges — weights only change when endpoints are co-retrieved.

### Sleep pipeline

Three phases, offline, LLM-powered:

**NREM — Relationship detection**: Unprocessed memories → vector neighbor pairs → Sonnet classification (duplicate/supports/hard_contradiction/soft_contradiction/temporal_evolution/contextual/related). Batch processing, 40K token limit per batch. Modes: standard (new memories only) or deep (all memories).

Also: **decay rate correction** — reviews memories with custom decay rates against actual feedback utility. Memories with high decay but consistently useful get slowed; persistently useless memories get accelerated. Finding: custom high-decay memories had *higher* average utility than default episodic memories, violating the assumption that custom high-decay = less important.

**REM — Summary generation**: Taxonomy-driven topic clustering → summary-layer memories above detail-layer (CLS hierarchy). Taxonomy is LLM-managed (Sonnet can add/rename/merge/split/reparent categories). Staleness: regenerate when 20%+ source details are newer than summary.

Dormancy criteria: detail memories covered by an active summary, access count < 3, older than 60 days → dormant status. Recoverable, not deleted. Summary count audit at 25+ to prevent summary bloat.

Health metrics: shadowed ratio, stale summaries, silent topics (details with no summary), combined risk score.

**Consolidation — Cluster judgment**: Edge-weight-filtered BFS detects thematic clusters (min 3 members). Opus judges each cluster holistically. Per-memory actions: keep / archive / merge_into / rewrite / annotate.

`annotate` preserves temporal arcs without merging. Protected memories (pinned, keep-flagged, corrections, zero-decay) cannot be archived.

**Orchestration**: Full cycle = normalize themes → NREM deep → retrieval failure repair → REM → post-sleep theme detection → probe recall.

Retrieval failure repair: reads `recall_miss` events (queries where nothing useful returned), diagnoses failure via LLM, applies theme repairs.

---

## Experiments & Evidence

### Methodology

19 optimization phases (wm1-wm19) over several weeks. 7-parameter search space (RRF_K, FEEDBACK_COEFF, THEME_BOOST, HEBBIAN_COEFF, HEBBIAN_CAP, ADJACENCY_BASE_BOOST, ADJACENCY_NOVELTY_FLOOR — later expanded with PPR params).

Ground truth: historical retrieval events from memory_events where utility >= 0.5 defines "relevant." ~17 testable queries from LoCoMo benchmark. Deterministic re-execution with cached embeddings (no API calls during trials). Framework: Optuna (TPE sampler, NSGA-II for multi-objective). A 200-trial study takes ~10-15 minutes.

Staged optimization: Stage 1 (core trio: RRF_K, FEEDBACK_COEFF, THEME_BOOST — >90% variance) → Stage 2 (graph params) → Stage 3 (joint refinement). Known risk: cross-stage interactions missed (wm2 found RRF_K × HEBBIAN_COEFF interaction at r = -0.77).

### Key findings

**Metric choice drives constants** (wm15 NSGA-II, 500 trials): AUC and MRR barely trade off (shallow Pareto front). MRR trades off harshly with miss_rate@5k (r = -0.517). Different metrics → different "optimal" constants from the same data.

**Feature importance** (wm15): Theme boost 67% AUC / 86% MRR. Feedback 15-20%. RRF_K 10-15%. Adjacency 2-5%. Hebbian <1%. *These importance values predate wm19. PPR implementation reduced THEME_BOOST from 0.97 to 0.19 — PPR absorbed much of theme boost's prior role. Post-PPR importance rankings have not been recomputed.*

**Two basins** (wm9, AUC metric): High-k basin (k≈14, broad candidates, feedback-dependent) vs. low-k basin (k≈3, sharp ranking, feedback-robust). Production chose Basin A (k=6 compromise) to leverage feedback advantage.

**Bayes prior is load-bearing** (wm5): Removing the empirical Bayes prior → FEEDBACK_COEFF collapses to near zero. Without centering, raw feedback scores hurt rather than help.

**Discretization ceiling**: miss_rate@5k with ~17 queries produces only ~3 distinguishable performance levels (integer ratios). AUC resolves this by sweeping all thresholds continuously.

### Ground truth status

Real-memory GT: 1,047 queries extracted from memory_events, ~112 candidates per query. Being judged offline in ~200-query batches using Claude Sonnet as relevance judge. ~200/1,047 complete as of March 2026.

The LoCoMo-tuned constants have not been validated on real data. This is the highest-priority experiment.

### Constant evolution

| Constant | wm1 | wm9 | wm15 | wm19 | Direction |
|----------|-----|-----|------|------|-----------|
| RRF_K | — | 14 | 6 | 6 | Settled |
| FEEDBACK_COEFF | 8.09 | 5.15 | 5.15 | 5.15 | Settled since wm9 |
| THEME_BOOST | — | — | 0.97 | 0.19 | PPR absorbed |
| HEBBIAN_CAP | — | 0.07 | 0.21 | 0.21 | Settled since wm15 |
| PPR_DAMPING | — | — | — | 0.775 | New |
| PPR_BOOST_COEFF | — | — | — | 2.0 | At search ceiling |

---

## What didn't work

**Quality floor** (wm1: optimal ratio = 0.0): Fixed minimum score threshold. Cliff detection handles it adaptively; fixed ratio can't handle varying score distributions.

**Shadow penalty in scoring** (9/10 reviewers recommended removal): Conflates temporal relevance with query relevance. Now informs dormancy during sleep instead.

**Confidence weight in scoring** (<0.1% contribution): Correlates with feedback — double-counts the signal.

**Intent routing** (0% fire rate): Query type classifier with per-type scoring weights. Queries don't decompose cleanly into intents; pipeline is robust without it.

---

## What we know we don't know

- **Sleep impact unmeasured**: Runs, looks reasonable on manual inspection. No retrieval quality metrics before vs. after.
- **LoCoMo → real data generalization unknown**: All tuning on ~17 queries from public benchmark. 1,047 real queries have different distributions.
- **Feedback loop health unknown**: Mean utility 0.244 across 12,894 events. Whether utility correlates with actual relevance is untested.
- **Scale behavior untested**: ~730 memories. Unknown at 2K+.
- **Event-time gap**: No `event_time` column — only `created_at`. Called the highest-impact schema change in the backlog.
- **Write-path quality ungated**: Dedup check (cosine distance < 0.1) and secret stripping, but no quality evaluation.
- **Contradiction detection**: 0.025-0.037 F1 across all surveyed systems. NREM catches adjacent only.

---

## Comparison landscape

Seven systems surveyed in depth (62 total sources):

| System | Approach | Strength | Key limitation |
|--------|----------|----------|----------------|
| **Mem0** | Extract-and-store | LLM-as-classifier, InformationContent() gating | No feedback, no forgetting, binary contradiction |
| **Zep/Graphiti** | Graph (Neo4j) | Bi-temporal edges (4 timestamps), entity resolution | 90% slower, no feedback loop |
| **HippoRAG** | Graph (PPR) | Variable-depth paths, synonym edges, exemplary ablation | Every index = LLM call, 26% extraction failures |
| **GraphRAG** | Hierarchical graph | Hierarchical summarization, grounded claims | One-shot consolidation, no decay/feedback |
| **Generative Agents** | Stream | Question-driven reflection, importance-as-trigger | No decay floor, static importance |
| **A-Mem** | Zettelkasten | Multi-faceted embedding (+121% multi-hop) | 3 LLM calls/write, no forgetting |
| **Cognee** | Document-to-graph | Triplet embeddings, usage frequency tracking | Batch-oriented, additive only |

**Where Somnigraph is unique**: Explicit feedback loop with measured GT correlation (r=0.70) — Ori-Mnemos v0.5.0 also closes a feedback loop via behavioral inference, but without GT validation. Offline consolidation with per-memory LLM judgment. Learned reranker (LightGBM, 26 features, +6.17pp NDCG).

**Where others do better**: Real-time graph construction (Zep, Cognee). Entity resolution (Graphiti). Scale validation (HippoRAG, GraphRAG). Temporal reasoning (Zep's 4-timestamp model).

**What nobody does well**: Contradiction detection. Write-path quality. Consolidation evaluation. Temporal grounding.

---

## Appendix: Full production constants

```python
# Core scoring (wm19 PPR study, 906 trials, 4D)
RRF_K = 6
RRF_VEC_WEIGHT = 0.5
FEEDBACK_COEFF = 5.15
THEME_BOOST = 0.19
HEBBIAN_COEFF = 3.0
HEBBIAN_CAP = 0.21
HEBBIAN_MIN_JOINT = 2

# PPR expansion
PPR_DAMPING = 0.775
PPR_BOOST_COEFF = 2.0
PPR_MIN_SCORE = 0.007
PPR_MAX_ITER = 50
PPR_CONVERGENCE_TOL = 1e-6

# Deprecated by PPR
ADJACENCY_BASE_BOOST = 0.33
ADJACENCY_NOVELTY_FLOOR = 0.67
CONTEXT_RELEVANCE_THRESHOLD = 0.5

# Per-category decay rates (ln(2) / half_life_days)
meta = 0.004        # ~173 day half-life
reflection = 0.006  # ~116 day half-life
semantic = 0.008    # ~87 day half-life
procedural = 0.012  # ~58 day half-life
episodic = 0.023    # ~30 day half-life
DEFAULT_DECAY_RATE = 0.008
DECAY_FLOOR = 0.35

# Cliff detection
CLIFF_Z_THRESHOLD = 2.0
CLIFF_MIN_RESULTS = 5

# Confidence learning
CONF_USEFUL_THRESHOLD = 0.5
CONF_USELESS_THRESHOLD = 0.1
CONF_GROWTH_RATE = 0.08
CONF_DECAY_RATE = 0.07
CONF_DURABILITY_NUDGE = 0.05
CONF_DEFAULT = 0.5
CONF_MIN_DELTA = 0.001

# Edge feedback
EDGE_WEIGHT_POS_STEP = 0.02
EDGE_WEIGHT_NEG_STEP = 0.01

# Theme enrichment
THEME_REFINE_THRESHOLD = 0.6
MAX_THEMES = 12
MAX_NEW_TERMS = 3
BRIDGE_THEME_THRESHOLD = 0.8
BRIDGE_TERM_MIN_LEN = 4

# Co-retrieval
CO_UTILITY_THRESHOLD = 0.7

# Disappointed recall detection
DISAPPOINTED_RECALL_MAX_UTIL = 0.2
DISAPPOINTED_RETRIEVAL_SCORE = 0.25

# Dedup
DEDUP_THRESHOLD = 0.1  # cosine distance; similarity > 0.9

# Sleep limits
MAX_EDGES_PER_CYCLE = 5
MAX_PRIORITY_BOOST_PER_CYCLE = 2
PENDING_STALE_DAYS = 14
```
