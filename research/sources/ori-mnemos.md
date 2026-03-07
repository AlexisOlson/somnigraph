# Ori-Mnemos: Source Analysis

**Phase 14 | 2026-03-06 | Repo: [aayoawoyemi/Ori-Mnemos](https://github.com/aayoawoyemi/Ori-Mnemos)**

---

## 1. Architecture Overview

**Stack**: TypeScript (93.6%), Node.js, better-sqlite3 for embedding/boost storage, `@huggingface/transformers` for local embeddings (all-MiniLM-L6-v2, 384-dim), graphology for graph algorithms, MCP server (stdio, SDK v1.27.1).

**Stars**: 14 | **License**: Apache-2.0 | **Version**: 0.3.4 | **Author**: aayoawoyemi

**Core design**: Memory as markdown files on disk, organized in a vault structure. Wiki-links (`[[Note]]`) serve as graph edges. No cloud dependencies -- embeddings run locally. SQLite stores only embeddings, activation boosts, and metadata hashes; the actual memory content lives in `.md` files with YAML frontmatter.

**Module organization** (20 core files):
- `vitality.ts` -- ACT-R decay with metabolic rates, structural boost, revival, bridge protection
- `graph.ts` -- Wiki-link parser, builds directed LinkGraph
- `importance.ts` -- graphology integration: PageRank, Louvain communities, Tarjan articulation points, betweenness centrality, personalized PageRank
- `fusion.ts` -- Score-weighted RRF combining three signal channels
- `activation.ts` -- Spreading activation with BFS, SQLite-persisted boosts
- `engine.ts` -- Embedding pipeline, multi-space composite search (6 spaces)
- `bm25.ts` -- Custom BM25 implementation with field weighting
- `intent.ts` -- Heuristic query intent classification (episodic/procedural/semantic/decision)
- `tracking.ts` -- IPS (Inverse Propensity Scoring) access logging and exploration injection
- `promote.ts` -- Inbox-to-notes capture pipeline with auto-classification and link detection
- `classify.ts` -- Pattern-based note type classification (idea/decision/learning/insight/blocker/opportunity)
- `linkdetect.ts` -- Wiki-link detection, suggestion (5 signals: title-match, tag-overlap, project-overlap, shared-neighborhood, semantic-similarity)
- `ranking.ts` -- Score containers and sort utilities
- `config.ts` -- YAML config with extensive defaults

**Vault zones**: Three metabolic spaces with different decay multipliers:
- `self/` -- Identity (0.1x decay rate, nearly permanent)
- `notes/` -- Knowledge (1.0x baseline)
- `ops/` -- Operations (3.0x accelerated decay)

**Tools**: 14 MCP tools, 16 CLI commands, 396 tests.

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

**Active**: Notes in `notes/` participate in retrieval, graph metrics, vitality scoring. Access count incremented on retrieval. Frontmatter `last_accessed` updated.

**Prune/Archive**: `ori_prune` identifies archive candidates using:
- Vitality zone classification (active >= 0.6, stale >= 0.3, fading >= 0.1, archived below)
- Protection: articulation points exempted, inDegree >= 2 exempted
- Dry-run by default, `--apply` to mutate
- Community hotspot reporting

---

## 3. Retrieval Mechanism

### Full Pipeline (`runQueryRanked`)

The retrieval pipeline is a 16-step process:

**Step 1-5**: Setup -- load vault, build link graph, compute graph metrics (PageRank, communities, bridges, betweenness), ensure embedding index, load activation boosts, compute vitality.

**Step 6**: Query intent classification via regex patterns. Four intents map to different space weight profiles:

```typescript
// Space weight profiles by intent
episodic:   { text: 0.40, temporal: 0.25, vitality: 0.15, importance: 0.05, type: 0.05, community: 0.10 }
procedural: { text: 0.30, temporal: 0.05, vitality: 0.10, importance: 0.30, type: 0.10, community: 0.15 }
semantic:   { text: 0.65, temporal: 0.05, vitality: 0.10, importance: 0.10, type: 0.05, community: 0.05 }
decision:   { text: 0.30, temporal: 0.15, vitality: 0.10, importance: 0.10, type: 0.30, community: 0.05 }
```

**Signal 1 -- Composite Vector Search** (6 sub-spaces):
- **Text space**: Weighted split similarity across title/description/body embeddings. Split weights also vary by intent (episodic favors body at 0.6; semantic favors title at 0.5).
- **Temporal space**: Exponential recency with 30-day half-life, encoded as piecewise-linear vector.
- **Vitality space**: Current ACT-R vitality score, encoded as piecewise-linear vector.
- **Importance space**: Normalized PageRank, encoded as piecewise-linear vector.
- **Type space**: One-hot encoded note type, matched against query-implied type vector.
- **Community space**: Hash-based community projection (sin/cos encoding with small primes).

Each sub-space produces a cosine similarity against a query-side "target" vector. The 6 scores are combined with intent-dependent weights.

**Signal 2 -- BM25 Keyword Search**: Custom implementation with field weighting (title 3x, description 2x, body 1x). Standard BM25 formula with k1=1.2, b=0.75.

**Signal 3 -- Personalized PageRank**: Seeds from entity extraction (note titles found in query). 20 iterations of power iteration with alpha=0.85. Falls back to uniform personalization if no entities found.

**Fusion**: Score-weighted RRF:
```
score = Sum_s( signal_weight_s * raw_score_s / (k + rank_s + 1) )
```

Default weights: composite=2.0, keyword=1.0, graph=1.5. Default k=60. This differs from standard RRF by incorporating the raw score magnitude, not just rank.

**Post-fusion**:
- Archive filtering
- Exploration injection: bottom 10% of results replaced with random unseen notes (IPS-inspired diversity)
- Access logging (JSONL)
- Spreading activation: top 3 results propagate boosts to graph neighbors

---

## 4. Standout Feature: ACT-R Vitality Model with Metabolic Zones and Structural Protection

The single most interesting technical contribution is the integration of ACT-R cognitive architecture with graph-theoretic structural protection into a unified "vitality" score that drives both retrieval weighting and pruning decisions. This is not just decay -- it is a multi-factor model with six components, graph-aware floor guarantees, and zone-based lifecycle management.

### ACT-R Base-Level Activation

The core decay function implements ACT-R's base-level activation equation:

```
B_i = ln(n / (1 - d)) - d * ln(L)
```

Where:
- `n` = access count (number of retrievals)
- `L` = lifetime in days since creation
- `d` = decay parameter (default 0.5, from Anderson & Lebiere 1998)

This is then normalized to [0, 1] via sigmoid: `vitality = 1 / (1 + exp(-B_i))`

The equation has a precise cognitive science lineage. In ACT-R (Adaptive Control of Thought -- Rational), base-level activation models the log-odds of a memory chunk being needed, based on its usage history. The `ln(n/(1-d))` term is a simplified form of the full practice equation `ln(sum(t_j^-d))` -- the sum over all past access times `t_j` raised to the negative decay power. The simplification assumes uniform spacing of accesses, which lets you replace the sum with a closed-form expression.

The key property: **access count and recency both matter, but through different mechanisms.** More accesses increase the numerator. More time since creation increases the denominator. The decay parameter `d` controls the relative weighting -- at d=0.5 (the human-calibrated default from ACT-R literature), a memory accessed 10 times over 30 days has similar activation to one accessed 3 times over 3 days.

### Metabolic Rate Modulation

The per-category decay innovation is the `metabolicRate` multiplier applied to the decay parameter before it enters the ACT-R equation:

```typescript
const effectiveDecay = actrDecay * metabolicRate;
```

Three zones with different metabolic rates:
- `self/` (identity): metabolicRate = 0.1 -- decay is 10x slower (effective d = 0.05)
- `notes/` (knowledge): metabolicRate = 1.0 -- baseline (effective d = 0.5)
- `ops/` (operations): metabolicRate = 3.0 -- decay is 3x faster (effective d = 1.5, but clamped to 0.99)

This maps to an intuitive cognitive metaphor: identity memories ("who am I") are nearly permanent, knowledge decays at normal rates, and operational details ("what happened in today's session") decay quickly. The clamping at 0.99 is important -- without it, `(1-d)` goes negative, inverting the activation formula.

### Six-Component Full Vitality

The `computeVitalityFull()` function combines six signals multiplicatively and additively:

```typescript
// 1. ACT-R base with metabolic-adjusted decay
let vitality = computeVitalityACTR(accessCount, lifetimeDays, effectiveDecay);

// 2. Structural stability boost (well-linked notes decay slower)
const structuralBoost = computeStructuralBoost(inDegree);
// Each incoming link adds ~10%, capped at 2x: 1 + 0.1 * min(inDegree, 10)
vitality = vitality * structuralBoost;

// 3. Access saturation modulation (diminishing returns)
// 1 - exp(-accessCount / k), where k=10
// Blend: base * (0.5 + 0.5 * saturation)
const saturation = computeAccessSaturation(accessCount, accessSaturationK);
vitality = vitality * (0.5 + 0.5 * saturation);

// 4. Revival spike (new connection to dormant note)
// exp(-0.2 * daysSinceNewConnection), capped at 14 days
const revival = computeRevivalBoost(daysSinceNewConnection);
vitality = vitality + revival * 0.2;

// 5. Spreading activation boost (from neighbor access)
vitality = vitality + (params.activationBoost ?? 0);

// 6. Bridge protection floor
if (bridges.has(noteTitle)) {
    vitality = Math.max(vitality, bridgeFloor);  // default 0.5
}
```

The structural boost is multiplicative -- a note with 5 incoming links gets 1.5x vitality, meaning it effectively decays 1.5x slower. This creates a "network effect" where well-connected notes persist longer, which is a reasonable cognitive analog (frequently referenced concepts are harder to forget).

The access saturation uses `1 - exp(-n/k)` to model diminishing returns from repeated access. At k=10: 10 accesses = 63%, 20 = 86%, 30 = 95%. This prevents "gaming" vitality through excessive retrieval.

The revival spike models the reactivation phenomenon from memory research -- when a decayed memory receives a new association (wiki-link), it gets a temporary boost that decays over 14 days.

### Tarjan Articulation Point Protection

The bridge protection is the graph-theoretic core. `findBridgeNotes()` identifies four types of structurally critical nodes:

**1. Tarjan's algorithm for articulation points**: Classic O(V+E) DFS-based algorithm. A vertex u is an articulation point if:
- u is root of DFS tree and has 2+ children, OR
- u is not root and `low[v] >= disc[u]` for some child v

```typescript
function dfs(u: string) {
    visited.add(u);
    disc.set(u, timer);
    low.set(u, timer);
    timer++;
    let children = 0;

    // Undirected neighbors (both directions)
    const neighbors = new Set<string>();
    graph.forEachOutNeighbor(u, (n) => neighbors.add(n));
    graph.forEachInNeighbor(u, (n) => neighbors.add(n));

    for (const v of neighbors) {
        if (!visited.has(v)) {
            children++;
            parent.set(v, u);
            dfs(v);
            low.set(u, Math.min(low.get(u)!, low.get(v)!));
            if (parent.get(u) === null && children > 1) bridges.add(u);
            if (parent.get(u) !== null && low.get(v)! >= disc.get(u)!) bridges.add(u);
        } else if (v !== parent.get(u)) {
            low.set(u, Math.min(low.get(u)!, disc.get(v)!));
        }
    }
}
```

**2. High-degree hubs**: Notes with inDegree > 2x median.

**3. Map notes**: Titles ending in " map" or exactly "index".

**4. Cross-project connectors**: Notes tagged with 2+ projects AND inDegree >= 3.

These protected notes get a vitality floor of 0.5 (configurable), meaning they can never decay below the "stale" zone threshold and will never be pruned. This prevents the knowledge graph from becoming disconnected during automated maintenance.

### Zone Classification and Pruning

Vitality scores map to four zones:
- Active (>= 0.6): Participates fully in retrieval
- Stale (>= 0.3): Still retrieved but flagged
- Fading (>= 0.1): Pruning candidate
- Archived (< 0.1): Below threshold

The pruning algorithm (`runPrune`) applies three protection filters before archiving:
1. Not an articulation point (Tarjan)
2. Not a high-degree hub
3. inDegree < 2 (notes with multiple backlinks are protected)

This creates a principled garbage collection system: notes that are structurally important to graph connectivity are preserved regardless of their individual usage patterns.

### Cognitive Science Lineage

The ACT-R model was developed by John R. Anderson at Carnegie Mellon, first published as ACT* in 1983 and refined through ACT-R 4.0 (1998) and ACT-R 6.0 (2004). The base-level activation equation models the "rational analysis of memory" -- the optimal Bayesian strategy for deciding what to remember given the statistical structure of the environment.

The specific form used here (`B_i = ln(n/(1-d)) - d*ln(L)`) is the simplified closed-form of the full practice equation, assuming equal spacing of practice. The full equation is `B_i = ln(sum_{j=1}^{n} t_j^{-d})` where `t_j` is the time since the j-th access. The decay parameter d=0.5 was calibrated against human recall data across multiple experiments.

Ori-Mnemos extends this individual-memory model with structural features (graph topology, metabolic zones) that have no direct ACT-R analog but draw from the broader cognitive science tradition of semantic network models (Collins & Loftus 1975) and spreading activation theory.

---

## 5. Other Notable Features

### 5.1 Spreading Activation with BFS and Persisted Boosts

When a note is retrieved (top 3 results), activation propagates to neighbors via BFS:

```typescript
// boost = utility * damping^hop
// Default: damping=0.6, max_hops=2, min_boost=0.01
```

At 2 hops with damping 0.6: hop-1 neighbors get `score * 0.6`, hop-2 neighbors get `score * 0.36`. Boosts are persisted in SQLite with decay-before-accumulate semantics (existing boosts decay at rate 0.1/day before new boosts are added, capped at 1.0). This means activation traces have a ~7-day half-life and compound over sessions -- frequently co-activated clusters develop persistent readiness.

### 5.2 Intent-Adaptive Six-Space Scoring

The composite vector search operates in six independently-scored spaces, each encoded as separate vectors:
- Text (384-dim MiniLM)
- Temporal (8-bin piecewise linear)
- Vitality (8-bin piecewise linear)
- Importance (8-bin piecewise linear from normalized PageRank)
- Type (6-dim one-hot)
- Community (16-dim hash projection)

The piecewise linear encoding is worth noting: instead of raw scalar values, each metadata dimension is encoded as a vector where bins below the value are 1.0, the value's bin is fractionally filled, and bins above are 0.0. This enables cosine similarity matching against query-side "target" vectors, making all six spaces commensurable.

### 5.3 IPS Exploration Injection

The Inverse Propensity Scoring system counters popularity bias. Access events are logged as JSONL, propensity computed as `times_surfaced / total_queries` (floored at epsilon=0.01), and the bottom 10% of results replaced with random unseen notes. This is a principled approach to the "rich get richer" problem in retrieval systems.

### 5.4 Louvain Community Detection

Using graphology-communities-louvain on an undirected projection of the link graph. Communities are used in two ways:
1. Community identity as a hash-projected 16-dim vector for scoring
2. Community hotspot analysis in pruning (mean vitality per community, top members)

### 5.5 Knowledge-Enriched Embeddings

Before embedding, notes are enriched with structural metadata:

```typescript
// Line 1: [TYPE] [PROJECTS]
// Title
// Description
// Connected: note1, note2, ... (up to 10 outgoing links)
```

This is similar to our strategy of concatenating content+category+themes+summary for enriched embeddings. The inclusion of connected note titles in the embedding text is interesting -- it means semantically-adjacent notes will have some embedding similarity even if their content differs, because they share graph neighbors.

---

## 6. Gap Ratings

### Layered Memory: 70%
Three metabolic zones (self/notes/ops) with different decay rates serve as functional memory layers. The inbox/active/archived lifecycle adds a processing pipeline. However, there is no true episodic vs. semantic distinction -- all notes are markdown files with the same schema. No working memory / long-term memory separation. The type field (idea/decision/learning/insight/blocker/opportunity) provides some categorization but not architectural layering.

### Multi-Angle Retrieval: 90%
Exceptional. Three-signal fusion (composite vector with 6 sub-spaces + BM25 + personalized PageRank) via score-weighted RRF. Intent-adaptive space weights. Entity extraction for PPR seeding. Exploration injection. This is one of the most sophisticated retrieval pipelines in the MCP memory landscape.

### Contradiction Detection: 5%
No contradiction detection at all. No mechanism for identifying conflicting information between notes. No revision tracking. The revision/contradiction/derivation edge flags in our system have no analog here.

### Relationship Edges: 75%
Wiki-links serve as typed edges by convention, but the type is implicit (all edges are just "links to"). The graph is rich -- PageRank, betweenness, communities, articulation points, spreading activation all operate on it. But there is no edge metadata (no linking_context, no flags, no edge types). The link suggestion system (5 heuristics) partially compensates by inferring relationship reasons, but these are computed at suggest-time, not stored.

### Sleep Process: 20%
No offline consolidation process. The closest analog is `ori_prune` which does vitality-based archiving, but it is an on-demand CLI command, not an autonomous background process. No NREM-style merging, no REM-style gap analysis, no cross-memory edge inference. The spreading activation system does provide a form of "priming" that operates across sessions, but it is triggered by retrieval, not by a separate consolidation cycle.

### Reference Index: 60%
Community-based organization via Louvain detection. Map notes and area routing provide navigational structure. The promote pipeline auto-assigns areas and builds footer sections with relevant notes and area links. No hierarchical taxonomy, but the community + area system provides reasonable navigational scaffolding.

### Temporal Trajectories: 45%
Temporal space in composite search with 30-day recency half-life. Access logging (JSONL) with timestamps. Created/last_accessed dates in frontmatter. But no temporal clustering, no "evolution of thought" tracking, no trajectory visualization. The temporal signal is a scalar recency score, not a trajectory.

### Confidence/UQ: 15%
The intent classifier has a confidence field (high/medium/low) based on pattern match count. The promote pipeline has min_confidence thresholds. But there is no per-memory confidence/trust level, no uncertainty quantification on retrieval scores, and no mechanism for confidence to compound through feedback or decay through contradiction. The classification confidence is a property of the classification process, not the memory itself.

---

## 7. Comparison with claude-memory

### Stronger (Ori-Mnemos vs. claude-memory)

**Graph algorithm sophistication**: Ori-Mnemos uses graphology with full PageRank, personalized PageRank, Louvain community detection, betweenness centrality, and Tarjan articulation points. Our system has novelty-scored adjacency expansion and edge flags but nothing at this level of graph-theoretic depth. The PPR seeded from entity extraction as a retrieval signal is particularly powerful -- it provides a principled way to discover notes that are topologically near the query, even if they are not semantically similar.

**Multi-space composite scoring**: The 6-space composite search with intent-adaptive weights is more granular than our 2-channel RRF. Encoding metadata dimensions as piecewise-linear vectors and using cosine similarity makes all spaces commensurable, avoiding the ad-hoc coefficient tuning that our post-RRF scoring requires. The intent classification (episodic/procedural/semantic/decision) dynamically reshapes the scoring profile in a principled way.

**Structural decay protection**: The Tarjan articulation-point protection is elegant. Preventing pruning of graph-critical nodes preserves connectivity. Our system has pinned/keep flags for manual protection, but no automatic topology-aware protection. The metabolic rate system (0.1x / 1.0x / 3.0x) is cleaner than per-category exponential decay with separate half-lives -- it is one parameter that modulates the same underlying ACT-R equation.

**Exploration injection**: The IPS-based diversity mechanism addresses popularity bias systematically. Our system has no equivalent -- frequently-recalled memories compound in feedback score and co-retrieval PMI, creating a rich-get-richer dynamic that we have not addressed.

**Local embeddings**: No API key required, no external dependency, no per-call cost. Our system depends on OpenAI text-embedding-3-small, which requires API access and has a per-token cost.

### Weaker (Ori-Mnemos vs. claude-memory)

**No feedback loop**: Ori-Mnemos has no recall_feedback equivalent. There is no mechanism for the user or agent to grade retrieval quality and have that signal improve future retrieval. Our Bayesian-optimized 9-coefficient post-RRF scoring, Hebbian PMI co-retrieval boost, and feedback-weighted scoring are all absent. The system logs access events but never uses them to tune retrieval.

**No sleep/consolidation**: No NREM-style cluster merging, no REM-style gap analysis, no cross-memory edge inference. Our sleep pipeline (cluster similar memories, merge duplicates, generate questions, create edges with linking_context) has no analog. Maintenance is limited to on-demand pruning.

**No contradiction/revision tracking**: No mechanism for detecting conflicting information, no revision edges, no contradiction flags. Our system tracks contradiction/revision/derivation edges with confidence compounding through support and decaying through contradiction.

**No edge metadata**: Wiki-links are untyped, undirected in practice. No linking_context, no edge flags, no edge-level features. Our edges carry rich metadata (linking_context with embeddings, flags, creation attribution) that supports novelty-scored adjacency expansion.

**No durability/shadow modeling**: No concept of memory shadow load, no durability dimension in feedback, no confidence gradient. The vitality score is the only quality signal.

**No theme/category normalization**: Notes have types and tags, but no normalization pipeline, no variant merging, no automatic taxonomy management.

**Scale considerations**: The BM25 index is rebuilt from disk on every query (`buildBM25IndexFromVault` reads all markdown files). The composite search iterates over all stored vectors. With hundreds of notes this is fine; at thousands it would become slow. Our FTS5 index is persistent and incremental.

### Shared Strengths

**Cognitive science grounding**: Both systems draw on cognitive science models for decay. We use exponential decay with per-category half-lives; they use ACT-R base-level activation. Both model the principle that memories should decay with disuse but persist with reinforcement.

**Hybrid retrieval**: Both combine vector similarity with keyword matching and additional signals. Both use RRF-family fusion. Both recognize that single-channel retrieval is insufficient.

**Graph-aware memory**: Both maintain relationship graphs between memories. Both use graph structure to influence retrieval and maintenance. Both protect structurally important nodes.

**Metadata-enriched embeddings**: Both embed more than just content -- we concatenate category+themes+summary; they concatenate type+projects+connected notes. Both recognized that pure content embedding loses important organizational context.

**Zone/status lifecycle**: Both have multi-stage memory lifecycles (capture -> active -> archived) with different treatment at each stage.

---

## 8. Insights Worth Stealing

### 8.1 Personalized PageRank as Retrieval Signal
**Effort**: Medium | **Impact**: High

Seeding PPR from entities detected in the query and using the resulting scores as a third retrieval signal. This would require building a graphology graph from our memory_edges table and running power iteration. The key insight: PPR finds notes that are topologically related to the query's subject, even when they share no semantic overlap.

Implementation sketch: Extract memory IDs mentioned in the query or matching by title. Build adjacency from memory_edges. Run 20 iterations of PPR with alpha=0.85. Use scores as an additional RRF channel.

### 8.2 Tarjan Articulation-Point Protection for Decay
**Effort**: Medium | **Impact**: Medium

Computing articulation points on our edge graph and giving them decay immunity. Currently we have manual pinned/keep flags; this would be automatic. Any memory whose removal would disconnect the graph gets a floor. This matters more as our edge graph grows.

Implementation note: Our edges are typed and directed, so we would need to decide which edge types form the "structural" graph for articulation analysis. Contradiction edges probably should not count toward structural importance.

### 8.3 Exploration Injection / Anti-Popularity-Bias
**Effort**: Low | **Impact**: Medium

Replacing the bottom 10% of recall results with random unseen or rarely-surfaced memories. Simple to implement: track surfacing counts per memory, inject low-count memories at the tail. Would counteract the Hebbian PMI rich-get-richer dynamic without undermining it for the top results.

### 8.4 Metabolic Rate Instead of Per-Category Half-Life
**Effort**: Low | **Impact**: Low

Conceptual insight more than implementation. Rather than separate half-life parameters per category, a single decay equation with a per-category multiplier is cleaner and more principled. We already have this architecture implicitly (per-category decay rates), but the ACT-R formulation with metabolic modulation is more theoretically grounded.

### 8.5 Piecewise-Linear Metadata Encoding
**Effort**: Medium | **Impact**: Low-Medium

Encoding scalar metadata (vitality, recency, importance) as piecewise-linear vectors and using cosine similarity makes metadata dimensions directly commensurable with text embeddings. This could replace our ad-hoc post-RRF coefficient weighting with a more principled geometric approach, though our Bayesian optimization may already be doing a better job empirically.

---

## 9. What's Not Worth It

### Markdown-as-Database
Ori-Mnemos stores memories as individual markdown files, reading frontmatter from disk on every query. This is charming for human-readable vaults but introduces O(n) filesystem reads for BM25 indexing and vitality computation. Our SQLite-native approach is faster, more atomic, and better suited for the query patterns of an MCP memory system. The "own your files" philosophy has UX appeal but engineering cost.

### Local Embeddings (all-MiniLM-L6-v2)
The 384-dim MiniLM model is smaller and lower-quality than text-embedding-3-small (1536-dim). No API dependency is nice, but the quality tradeoff matters for a system with hundreds of memories where precision is more important than cost. At our scale (~300 memories), the API cost is negligible.

### Six-Space Composite Scoring
The 6-space composite search with piecewise-linear encoding is theoretically elegant but adds complexity we do not need. Our 2-channel RRF with Bayesian-optimized post-RRF coefficients achieves equivalent functionality with less code and the scoring coefficients are empirically tuned to our actual retrieval patterns, not chosen a priori.

### IPS Access Logging
The full JSONL access log with propensity computation is over-engineered for our scale. At ~300 memories with ~10-20 recalls per session, popularity bias is manageable through the Hebbian PMI system (which naturally saturates) and recall_feedback (which down-weights stale results). The exploration injection idea is worth stealing, but the full IPS infrastructure is not.

### Note Type Classification
The 6-type classification (idea/decision/learning/insight/blocker/opportunity) and associated query-type matching adds a layer of indirection that requires maintained regex patterns and per-type weight profiles. Our category system (procedural/episodic/semantic/reflection/meta) with per-category decay rates serves a similar purpose more simply, and our intent-aware query routing achieves similar benefits.

---

## 10. Key Takeaway

Ori-Mnemos is the most graph-theoretically sophisticated MCP memory system in the landscape, combining ACT-R cognitive decay with Tarjan articulation-point protection, PageRank retrieval, Louvain community detection, and spreading activation into a coherent vitality model. Its three-signal retrieval fusion with intent-adaptive 6-space scoring is technically impressive. The key insight to carry forward is that graph topology should actively protect structurally important memories from decay and inform retrieval through personalized PageRank -- two capabilities we lack. However, the system has no feedback loop, no consolidation process, no contradiction detection, and no edge metadata, which are the areas where our architecture is strongest. The design reflects a "beautiful static architecture" philosophy (build the right equations, they will work) versus our "adaptive closed-loop" philosophy (build feedback mechanisms, let them tune themselves). Both approaches have merit; the graph algorithms are the most transferable contribution.
