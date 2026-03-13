# Architecture

This document tells the story of how Somnigraph's architecture emerged — what problems drove each decision, what was tried and removed, and what remains unsolved. It's written for someone building their own memory system, not as API reference.

## Contents

1. [Why this exists](#why-this-exists)
2. [Foundations](#foundations)
3. [The retrieval pipeline](#the-retrieval-pipeline)
4. [Scoring: where the signal lives](#scoring-where-the-signal-lives)
5. [The graph](#the-graph)
6. [Sleep](#sleep)
7. [The refactor](#the-refactor)
8. [Tuning](#tuning)
9. [What didn't work](#what-didnt-work)
10. [Open problems](#open-problems)

---

## Why this exists

Claude Code sessions start with no memory. Each conversation is a blank slate — the same introductions, the same context-building, the same discoveries made and lost. For a tool that helps with ongoing projects, this is a fundamental limitation.

Several systems address this. Mem0 extracts facts from conversations. Zep/Graphiti builds knowledge graphs. HippoRAG borrows hippocampal indexing for RAG. All of them solve retrieval reasonably well. None of them handle three harder problems:

**Contradiction.** When a user says "we decided to use Redis" in January and "we switched to Postgres" in March, most systems store both as equally valid facts. Detecting that the second supersedes the first — without deleting the history of the decision — is universally catastrophic across published systems (0.025–0.037 F1 on contradiction benchmarks).

**Consolidation.** Memories accumulate. Without periodic maintenance, the store becomes a growing pile where important patterns get buried under recent noise. Biological memory systems solve this with sleep — offline consolidation that strengthens important connections and lets unimportant ones fade. Most AI memory systems never consolidate at all.

**Feedback.** You can't improve retrieval without knowing what was useful. Most systems treat retrieval as a one-way operation: query in, results out. There's no signal flowing back to shape future retrievals. The memories that helped get no reinforcement; the ones that wasted context get no penalty.

Somnigraph is built around these three problems. The retrieval pipeline is solid but not the point — retrieval is a solved-enough problem. The point is the feedback loop (utility scores reshape scoring), the sleep pipeline (offline consolidation with LLM judgment), and the graph (edges that encode relationships between memories, not just content similarity).

---

## Foundations

### Why SQLite

Every memory system needs a persistence layer. The options: Postgres (with pgvector), a dedicated vector database (Pinecone, Qdrant), or SQLite.

SQLite wins on three axes:

1. **Local-first.** No server process, no Docker container, no connection string. The database is a file. Claude Code runs locally; the memory system should too.
2. **Transactional integrity.** WAL mode gives concurrent reads with serialized writes. A single `memory.db` file holds everything — vectors, full-text index, metadata, event log — in one transactional boundary. No distributed consistency problems.
3. **sqlite-vec + FTS5.** Two extensions give SQLite genuine hybrid search capability. `sqlite-vec` provides float32 vector storage with cosine distance KNN. FTS5 provides full-text search with BM25 ranking. Both are mature, battle-tested C extensions.

The cost: no built-in replication, no multi-process writes. For a personal memory system used by one Claude instance at a time, these aren't costs.

### The schema

Nine tables tell the architecture story:

**`memories`** — The core table. Each memory has content, a summary (for compact display), a category (episodic/semantic/procedural/reflection/meta), themes (JSON array), base priority (1–10), and temporal bookkeeping (created_at, last_accessed, access counts split by access type). Notable columns: `decay_rate` (per-memory override of category default), `confidence` (0.1–0.95, compounds through feedback), `shadow_load` (how much newer similar memories suppress this one), `layer` (detail/summary/gestalt — the CLS hierarchy), `valid_from`/`valid_until` (temporal bounds for evolved facts).

**`memory_vec`** — sqlite-vec virtual table. Float32 embeddings at 1536 dimensions (OpenAI text-embedding-3-small), indexed for cosine distance KNN.

**`memory_fts`** — FTS5 virtual table indexing summary and themes (not content). This is deliberate: content is noisy and long; summary and themes are curated signals. BM25 weights: summary 5x, themes 3x.

**`memory_rowid_map`** — Bridges UUID primary keys to integer rowids that sqlite-vec and FTS5 require. Autoincrement ensures stable mapping.

**`memory_edges`** — The relationship graph. Source, target, and rich metadata: `linking_context` (natural language explanation of the relationship), `linking_embedding` (vector of the linking context, used for query-conditioned expansion), `flags` (JSON: contradiction, revision, derivation), `weight` (0–1, shaped by feedback). Schema v2 replaced the original `edge_type` enum with flags + context, which turned out to be far more expressive.

**`memory_events`** — Append-only lifecycle log. Every retrieval, feedback signal, edge creation, and status change is recorded with timestamps, query text, co-retrieved memory IDs, and similarity scores. This table is the training data for the feedback loop.

**`sleep_log`** — Audit trail for consolidation cycles. Records what each sleep phase processed, found, and changed.

### Why text-embedding-3-small

Cost and latency. The system embeds on every write (remember, link, feedback-triggered theme enrichment) and every read (recall). At ~$0.02/1M tokens, embedding cost is negligible even at high volume. The 1536-dimension vectors are large enough for good discrimination, small enough for fast KNN on hundreds of memories.

The embedding is enriched: instead of embedding raw content, we concatenate content + category + themes + summary. This front-loads metadata into the vector space, so a query about "chess engine tuning" naturally gravitates toward memories tagged with chess-related themes even if the content discusses specific technical details.

### CLS theory scaffolding

Complementary Learning Systems theory (McClelland et al.) describes how biological memory works: fast encoding in the hippocampus (episodic, specific), slow consolidation in the neocortex (semantic, general). The interplay between fast and slow systems — with sleep as the consolidation mechanism — is the theoretical backbone of the architecture.

In Somnigraph:
- **Fast encode**: `remember()` stores detail-layer memories immediately, with minimal processing (dedup check, privacy strip, theme normalization)
- **Slow consolidate**: Sleep phases detect relationships (NREM), generate summaries (REM), and perform cluster-level judgment (consolidation)
- **Entity-based pattern completion**: The graph enables retrieval of memories related to the query's neighbors, not just the query itself — analogous to hippocampal pattern completion

This isn't a strict biological model. It's a scaffolding that suggests which capabilities matter and when they should fire.

---

## The retrieval pipeline

Recall is a hybrid search: vector similarity and keyword matching, fused with Reciprocal Rank Fusion (RRF), then reshaped by feedback, graph structure, and quality filtering.

### Vector channel

The query (or explicit context string) is embedded via OpenAI. KNN search against `memory_vec` returns the top candidates ranked by cosine distance. This channel excels at semantic similarity — "how does the scoring pipeline work?" retrieves memories about RRF fusion even if they never use the word "scoring."

### Keyword channel

The query is sanitized for FTS5: known multi-word phrases are preserved as quoted groups ("memory system", "recall feedback", "iceland spar"), remaining tokens are joined with AND (≤3 groups, for precision) or OR (4+ groups, for breadth). BM25 ranks results with field weights: summary 5x, themes 3x.

This channel catches what vectors miss: exact terms, proper nouns, specific identifiers. A query for "wm9" (a tuning study name) relies entirely on keyword matching — there's no semantic content to embed.

### RRF fusion

Reciprocal Rank Fusion combines the two ranked lists without needing score normalization:

```
score(d) = w_vec / (k + rank_vec(d)) + w_kw / (k + rank_kw(d))
```

The parameter `k` controls how much rank matters. High k (e.g., 33) makes RRF a soft union — most candidates get similar scores. Low k (e.g., 3) makes top ranks dominate. Production settled at k=6 after 15 tuning studies showed this balances precision and recall for the ~700-memory scale.

`w_vec` and `w_kw` are currently equal (0.5 each). Tuning studies found the system is insensitive to this ratio — the two channels are complementary enough that equal weighting works.

### Why RRF over learned fusion

RRF is parameter-light (one constant k), requires no training data, and handles the case where a memory appears in only one channel gracefully (the missing channel contributes zero). Learned fusion (cross-encoders, learned sparse retrieval) would require a training set of query-relevance pairs that doesn't exist at system bootstrap. RRF works from day one.

---

## Scoring: where the signal lives

RRF produces candidate scores. Four post-RRF stages reshape them before results are returned.

### Stage 1: Feedback boost

The dominant scoring signal. Each memory accumulates utility ratings (0.0–1.0) from `recall_feedback()`. The question: how do you use sparse, noisy feedback without letting a single early rating dominate?

**The discovery arc:**

The first attempt used raw mean utility with a hard gate: ignore memories with fewer than 14 feedback events. This created a cold-start cliff — new memories couldn't benefit from feedback at all, and the threshold was arbitrary.

The fix: empirical Bayes with a Beta prior. The system fits a Beta distribution to the population of per-memory mean utilities (method of moments). Each memory's score is then a weighted blend of its own EWMA utility and the population prior, where the weight increases with observation count:

```
blended = (n * ewma + prior_strength * prior_mean) / (n + prior_strength)
boost = FEEDBACK_COEFF * (blended - prior_mean)
```

This eliminates the hard gate entirely. A memory with one feedback event gets pulled strongly toward the population mean. A memory with 20 events expresses its own signal clearly. The prior is recomputed whenever the feedback count changes, so it adapts as the system matures.

EWMA (exponential weighted moving average, alpha=0.3) replaced simple mean for aggregating per-memory feedback. This ensures recent ratings dominate — a memory that was useful six months ago but irrelevant now sees its score decay naturally.

**Coefficient:** `FEEDBACK_COEFF = 5.15`, making feedback the strongest post-RRF signal by a wide margin. This was stable across multiple tuning studies; the system wants feedback to dominate.

### Stage 2: Theme boost

Multiplicative boost for memories whose themes overlap with an explicit `boost_themes` list (provided by the caller). At first glance this seems redundant — themes are already embedded in the vector. But the boost serves a different purpose: it lets the caller signal "I know I want memories about X" in a way that bypasses the fuzziness of vector similarity.

`THEME_BOOST = 0.97` per overlapping theme. Tuning studies found this is the dominant parameter in 3D optimization space (67% AUC importance, 86% MRR importance), which was surprising given that it's "just" double-counting information already in the embeddings. The explanation: theme overlap is a discrete, high-confidence signal. Vector similarity is continuous and noisy. The boost amplifies the discrete signal.

### Stage 3: Hebbian co-retrieval

Memories that are frequently retrieved together probably relate to each other in ways that content similarity alone doesn't capture. The Hebbian boost uses pointwise mutual information (PMI):

```
PMI(A, B) = log2(P(A,B) / (P(A) * P(B)))
```

Where P(A) is the probability of memory A appearing in a retrieval set (from the last 30 days of events), and P(A,B) is the probability of both appearing together.

The PMI denominator naturally suppresses "hub" memories that appear in every retrieval — they have high P(A), so their PMI with any specific memory is low. This is important: without it, frequently-retrieved memories would boost everything.

**Cap:** `HEBBIAN_CAP = 0.21`. The old guidance ("degrades above 0.1") was wrong — at k=6, the cap can be higher because RRF scores are more spread out. But the Hebbian contribution is minor compared to feedback (<0.3% feature importance in tuning studies). It matters most when feedback is absent, expanding to fill the vacuum.

### Stage 4: Adjacency expansion

The graph comes alive here. Top-scoring memories (seeds) expand along their edges to pull in neighbors that weren't in the original retrieval set. But naive BFS expansion is catastrophic — HippoRAG's ablation showed BFS drops Recall@2 from 40.9 to 25.4.

The solution: novelty-scored expansion. For each neighbor of a seed:

1. **Novelty score**: Project the neighbor embedding onto the seed embedding. The residual (what the neighbor adds beyond the seed) is compared to the query. High query-residual similarity means the neighbor adds query-relevant information the seed doesn't have.
2. **Context weight**: If the edge has a `linking_embedding` (from the linking context), compare it to the query. High similarity means the *reason* these memories are linked is relevant to the current query.
3. **Seed strength**: The seed's own RRF score (normalized). Strong seeds produce stronger expansions.
4. **Edge weight**: Shaped by feedback over time. Edges between co-useful memories get stronger.

```
boost = ADJACENCY_BASE_BOOST * novelty * context_weight * seed_strength * edge_weight
```

At the current scale (~700 memories), adjacency expansion rarely fires — the graph isn't dense enough for meaningful multi-hop traversal. But the infrastructure is in place for when it matters.

### Cliff detection

After scoring, results are sorted by final score. A log-curve is fit to the score sequence, and the system looks for a "cliff" — a point where actual scores drop sharply below the fitted curve (more than 2.0 standard deviations). Everything below the cliff is cut.

This replaces a fixed quality floor (which was removed — see [What didn't work](#what-didnt-work)). The adaptive approach handles the fact that score distributions vary wildly across queries.

---

## The graph

### Edge schema

Edges connect memories with rich metadata:

- **`linking_context`**: Natural language explaining why these memories relate. "Both discuss the decision to switch from Redis to Postgres." This text is embedded and used for query-conditioned expansion.
- **`linking_embedding`**: The vector of the linking context. Enables the context weight in adjacency expansion — edges fire only when the query is relevant to *why* they're linked.
- **`flags`**: Structural markers. `contradiction` (these memories disagree), `revision` (one updates the other), `derivation` (one was generated from the other, e.g., a summary from details).
- **`weight`**: 0–1, default 0.5. Increases when both endpoints are useful in the same retrieval (+0.02), decreases when either is useless (−0.01). Asymmetric by design — it's easier to weaken an edge than strengthen one.
- **`created_by`**: `sleep` (detected during NREM consolidation) or `session` (created explicitly via `link()`).

### Shadow load

When a newer memory is very similar to an older one (cosine similarity > 0.7), the older memory accumulates "shadow load" — a signal that it's being eclipsed. Shadow load doesn't directly affect scoring (it was removed from the scoring pipeline — see [What didn't work](#what-didnt-work)) but informs dormancy decisions during sleep.

### Temporal evolution

When the system detects that a newer memory updates an older fact (e.g., "we use Redis" → "we switched to Postgres"), it doesn't delete the older memory. Instead:

1. Set `valid_until` on the older memory to the newer memory's `created_at`
2. Create a `revision`-flagged edge between them
3. Both memories remain queryable, but the temporal bounds let the system (and the reader) understand the evolution

This preserves the narrative arc — *when* and *why* a decision changed — rather than just the current state.

---

## Sleep

Sleep is offline consolidation: LLM-powered maintenance that runs between sessions. Three phases, inspired by biological sleep stages but not slavishly modeled on them.

### NREM: Relationship detection

**What it does:** Takes unprocessed memories, finds their nearest neighbors via vector similarity, and sends pairs to an LLM (Sonnet) for classification.

**Classification types:**
- `duplicate` — Same information, different wording. The weaker one is deleted (superseded_by the stronger).
- `supports` — Reinforcing information. Edge created, priority boosted on both.
- `hard_contradiction` — Direct factual conflict. Contradiction-flagged edge, confidence reduced on the older memory.
- `soft_contradiction` — Tension without direct conflict. Edge created, flagged for attention.
- `temporal_evolution` — Same topic, updated over time. Triggers the temporal evolution handler.
- `contextual`, `related` — Weaker relationships. Edges created at lower weight.

**Batching:** Memory pairs are grouped into token-bounded batches (40k token limit) and processed sequentially. Parallel processing was abandoned due to Windows memory constraints with subprocess-based LLM calls.

**Decay rate correction (Step 4b):** After edge detection, the system reviews memories with custom decay rates against their actual feedback utility. A memory set to decay quickly (high decay_rate) but consistently rated useful gets its decay slowed. A memory set to persist (low decay_rate) but never useful gets its decay accelerated. Early assumption "custom high-decay memories are less important" was wrong — they had *higher* average utility than default episodic memories.

### REM: Summary generation

**What it does:** Produces summary-layer memories that sit above detail-layer memories in the CLS hierarchy. Summaries serve two purposes: (1) compact representation for startup_load (where token budget is tight), and (2) vocabulary bridging — a summary that says "chess engine tuning parameters" helps retrieve detail memories that only discuss specific Lc0 configuration flags.

**Taxonomy:** Memories are assigned to a hierarchical taxonomy of topics. The taxonomy itself is LLM-managed — Sonnet can add, rename, merge, split, or reparent categories based on the memory population.

**Staleness:** A summary is stale when 20%+ of its source detail memories were created after the summary. Stale summaries are regenerated by Opus, with a vitality score (exponential decay of cluster age) determining priority — fresh, active clusters get attention first.

**Dormancy:** Detail memories covered by an active summary, with low access count (<3) and older than 60 days, transition to dormant status. They're not deleted — they can be reactivated — but they stop appearing in retrieval. This is how the system manages growth without unbounded accumulation.

**Summary maintenance:** When the summary count exceeds 25, an LLM audit recommends deletions, merges, and re-tiering. This prevents summary bloat from mirroring the detail bloat it was meant to solve.

**Health metrics:** REM computes system health: ratio of shadowed memories, stale summaries, silent topics (those with details but no summary), combined into a single health risk score.

### Consolidation: Cluster judgment

**What it does:** Detects thematic clusters (via edge-weight-filtered BFS, minimum 3 members) and submits each cluster to Opus for holistic judgment.

**Actions per memory in a cluster:**
- `keep` — No change.
- `archive` — Redundant given other cluster members. Soft-deleted, superseded_by the keeper.
- `merge_into` — Unique content folded into a target memory. Source archived.
- `rewrite` — Memory becomes the cluster representative. Re-embedded with new content.
- `annotate` — Temporal arc context appended. Preserves the narrative of how understanding evolved.

**Why `annotate` exists:** Early consolidation destroyed texture. Archiving a sequence like "ordered crystals → shipping delayed → arrived damaged → replacement sent" into a single summary loses the story. Each step has its own weight. `annotate` preserves the arc while marking the connections — the system remembers the *process*, not just the outcome.

**Formation detection:** If a cluster represents a formative pattern (an insight that shaped subsequent thinking), it can be marked as a formation: decay_rate set to 0 (timeless), tagged with a `formation` theme.

**Safety:** Protected memories (pinned, keep-flagged, corrections, zero-decay) cannot be archived. The system errs toward preservation.

### Orchestration

`sleep.py` runs the full cycle: theme normalization → NREM deep → retrieval failure repair → REM → post-sleep theme detection → probe recall.

**Retrieval failure repair:** Between NREM and REM, the orchestrator reads `recall_miss` events (queries where nothing useful was returned), diagnoses the failure via LLM, and applies theme repairs. This closes the loop: bad retrievals → diagnosis → theme adjustment → better future retrievals.

**Post-sleep theme detection:** Heuristic + LLM detection of theme variants (casing, underscore/hyphen, singular/plural). Auto-applies high-confidence merges; queues ambiguous ones for review.

---

## The refactor

The system started as a single `memory_server.py` file that grew past 2,000 lines. The refactor split it into 16 modules organized by responsibility:

| Module | Lines | Responsibility |
|--------|-------|---------------|
| `memory_server.py` | 307 | MCP wiring — `@mcp.tool()` decorators, imports, startup |
| `constants.py` | 96 | All tuning parameters, organized by provenance |
| `db.py` | 383 | Schema, migrations, connection management, ID resolution |
| `embeddings.py` | ~80 | OpenAI client, tokenization, enriched text construction |
| `vectors.py` | ~60 | Serialization, vector math, novelty score operator |
| `write.py` | ~50 | Memory insertion across three tables (memories, vec, FTS) |
| `privacy.py` | ~40 | Regex-based secret redaction (API keys, JWTs, tokens) |
| `themes.py` | ~100 | Theme normalization, variant mappings, content-based expansion |
| `fts.py` | ~80 | FTS5 query sanitization, known phrases, index management |
| `scoring.py` | 454 | RRF fusion, feedback boost, Hebbian, adjacency, cliff detection |
| `graph.py` | 291 | Edge creation, related memory search, shadow load, temporal evolution |
| `decay.py` | 32 | Exponential decay with per-category rates and retention immunity |
| `formatting.py` | ~40 | Display formatting (compact, full, pending) |
| `sync.py` | ~30 | Event logging, safe row accessors |
| `stats.py` | ~80 | Diagnostic statistics |
| `tools.py` | ~600 | Tool implementations (the business logic) |

The refactor changed nothing algorithmic. Its value: sleep scripts can import specific functions instead of importing the entire monolith, and each module can be read and understood in isolation.

The wiring layer (`memory_server.py`) still has re-exports for backward compatibility with sleep scripts that `import memory_server` directly. These should be cleaned up — sleep scripts should import from the package.

---

## Tuning

Fifteen optimization studies (wm1–wm15) over several weeks, searching a 7-dimensional parameter space. This section tells the story of what we learned, not the per-trial results (see `experiments.md` for methodology).

### The arc

**wm1 (staged, 200 trials):** First systematic search. Three revelations: `quality_floor_ratio = 0.0` is optimal (the quality floor adds nothing), `feedback_coeff = 8.09` makes feedback the dominant signal, and the system wants high RRF k (soft union over hard ranking).

**wm2–wm4 (miss_rate metric):** Shifted to miss_rate@5k as the metric (fraction of known-useful memories not retrieved within a 5k token budget). Discovered a discretization ceiling: miss_rate has only ~3 distinguishable performance levels at the current memory count (integer ratios: 1/17, 2/17, ...). The metric can't distinguish between "good" and "slightly better."

**wm5 (validation):** Tested whether the Bayesian prior is load-bearing by running without it. Answer: yes. `feedback_coeff` collapsed to near zero — without centering, raw feedback scores are noisy enough to hurt. The prior *is* the mechanism that makes feedback useful.

**wm9 (AUC metric):** The definitive study. Switched to AUC (area under the threshold curve) which avoids the discretization problem. Revealed two distinct "basins" in the parameter space: a high-k basin (k ≈ 14, broad candidate set, rely on post-RRF signals) and a low-k basin (k ≈ 3, tight candidate set, trust the retrieval). Both are local optima. Production chose the high-k basin — it's more robust to edge cases.

**wm11b–11c (Hebbian focus):** Tested soft Hebbian caps as alternatives to the hard cap. Hard cap is genuinely better — the contribution is so minor that smoothing the cutoff doesn't help.

**wm15 (multi-objective, NSGA-II):** Dual-objective optimization: AUC vs. MRR (mean reciprocal rank). Found the Pareto front and discovered that AUC and MRR barely trade off (you can have both), but miss_rate@5k trades off harshly with MRR (r = −0.517). Selected Pareto point #4 as the production configuration.

### What the studies taught us

1. **Feedback is the system.** With a working feedback loop, the retrieval pipeline is just candidate generation. Without it (wm5), everything else scrambles to compensate.

2. **Theme boost is double-counting, but the right kind.** Themes are in the embeddings, so boosting on theme overlap is redundant information. But it converts continuous similarity into discrete signal, which is more reliable for ranking. The tuning studies want it high (0.97).

3. **Adjacency is non-removable.** Every study that tried to zero out adjacency found worse results. Even at the current scale where expansion rarely fires, the base boost on seed neighbors matters.

4. **Metric choice drives parameter values.** The same system with different metrics produces wildly different "optimal" constants. This isn't a bug — different metrics ask different questions. AUC asks "how well do you rank across all thresholds?" Miss rate asks "do you find the important stuff within a budget?" The answer to each shapes the parameters differently.

5. **The parameter space has structure.** Two basins, a Pareto front, stable feature importances. This isn't a random landscape — it's a system with comprehensible mechanics. Understanding the structure matters more than finding the global optimum.

---

## What didn't work

Honest accounting of ideas that were implemented, tested, and removed.

### Quality floor (removed)

A minimum score threshold: memories scoring below `quality_floor_ratio * top_score` are dropped. The idea — filter noise at the bottom of the ranked list.

wm1 found the optimal ratio is 0.0. The quality floor adds nothing because (a) cliff detection already handles the sharp drop-off case, and (b) a fixed ratio can't adapt to queries with naturally spread-out or compressed score distributions. The parameter was hardcoded to zero and the mechanism left as dead code until removal.

### Shadow penalty in scoring (removed)

Shadow load (how much newer similar memories eclipse older ones) was included as a scoring penalty: high shadow load → lower score. Nine out of ten reviewers recommended removal. The problem: shadow load is a reasonable dormancy signal (should this memory stop appearing?) but a terrible scoring signal (should this memory rank lower *right now*?). A memory might be shadowed in general but be exactly what this specific query needs. Scoring should be query-dependent; shadow load is query-independent.

Shadow load remains as metadata, informing dormancy decisions during sleep. It just doesn't touch the scoring pipeline.

### Confidence weight in scoring (removed)

Confidence (0.1–0.95) tracks how trustworthy a memory is, compounding through feedback and edge signals. Using it as a scoring multiplier was intuitive but marginal — tuning studies showed <0.1% contribution. The reason: confidence correlates with feedback (well-confirmed memories have high confidence), so multiplying by confidence mostly double-counts the feedback signal.

Confidence remains as metadata, used in sleep for classification decisions and visible in diagnostics. It was removed from the scoring multiplication.

### Intent routing (removed)

A classifier that categorized queries into types (factual, exploratory, reflective) and adjusted scoring weights per type. Implementation: 0% fire rate in production. Queries don't cleanly decompose into intent categories, and the scoring pipeline is robust enough across query types that differential treatment adds complexity without benefit.

### Shadow penalty in scoring (details)

The original design assumed that a memory heavily shadowed by newer versions was "stale" and should rank lower. This conflates two meanings of relevance:

- **Temporal relevance**: Is this memory current? (Shadow load is a reasonable proxy.)
- **Query relevance**: Does this memory answer this query? (Shadow load says nothing about this.)

The system needs both signals, but at different times. Temporal relevance matters during consolidation (should we dormant this?). Query relevance matters during retrieval. Mixing them in the scoring pipeline confused both.

---

## Open problems

### Contradiction detection

The hardest unsolved problem in memory systems. When a user says "the API uses REST" and later "we migrated to GraphQL," the system should understand that the second supersedes the first for that specific claim while preserving both as historical records.

Current state: NREM sleep classifies pairs as contradictions, but only between adjacent memories (those similar enough to be neighbors in vector space). Contradictions between distant memories — or between a specific memory and a general principle — go undetected.

Published benchmarks show all systems perform catastrophically on contradiction: 0.025–0.037 F1. This isn't a tuning problem; it's a representation problem. The `valid_from`/`valid_until` schema exists but relies on manual or sleep-detected temporal evolution, which only catches obvious cases.

### Personalized PageRank for graph traversal

The current adjacency expansion is one-hop: seeds expand to neighbors. HippoRAG demonstrated that Personalized PageRank (PPR) — which propagates activation through the graph until convergence — finds multi-hop connections that one-hop expansion misses.

The infrastructure exists (edges with weights, query-conditioned scoring). What's missing: the PPR algorithm itself, and enough graph density for it to matter. At ~700 memories with ~1,500 edges, most memories are reachable in 2 hops anyway. PPR becomes valuable at larger scales with richer graph structure.

### Write-path quality

The current write path is permissive: `remember()` checks for near-duplicates and strips secrets, but doesn't evaluate whether a memory is worth storing. The result: the store accumulates low-value memories that consume budget and dilute retrieval.

Possible fixes: LLM-judged write quality (expensive, slow), heuristic quality signals (token count, specificity, novelty against existing store), or aggressive consolidation (accept everything, clean up during sleep). The current approach is the third option, but sleep can't fix fundamentally low-quality inputs.

### Event-time for temporal queries

Memories have `created_at` (when stored) but not `event_time` (when the described event happened). "What did we decide about the database last month?" requires temporal filtering on the event, not the storage time. A memory stored today about a decision made last week would be missed.

This is the single highest-impact schema change in the backlog. The column is trivial to add; the hard part is getting the timestamp at `remember()` time, which requires either explicit user input or LLM extraction from context.

### Consolidation quality evaluation

No benchmark exists for evaluating whether consolidation improved or degraded the memory store. Current evaluation is manual: run sleep, inspect the results, assess whether the right things were merged/archived/preserved.

Building a consolidation benchmark requires: (a) a memory store with known quality issues (redundancy, contradictions, stale information), (b) a ground-truth consolidation (what should be merged, archived, etc.), and (c) metrics that capture both compression (did it reduce redundancy?) and fidelity (did it preserve important information?). None of these exist in the published literature.

---

*This document describes the system as of March 2026. The architecture continues to evolve — see the git log and `CHANGELOG.md` for ongoing changes.*
