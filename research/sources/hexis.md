# Hexis (QuixiAI/Hexis) — Analysis

*Generated 2026-03-16 by Opus agent reading local clone*

---

## Repo Overview

**Repo**: https://github.com/QuixiAI/Hexis
**License**: Non-commercial research (custom)
**Language**: Python (3.10+) + SQL (PostgreSQL functions)
**Description**: "A cognitive architecture that provides persistent memory, identity, and autonomous behavior to AI agents"
**Author**: Eric Hartford / QuixiAI

**Problem addressed**: Giving AI agents persistent identity, memory, and autonomous behavior — not just retrieval, but the conditions under which character can form. The database is the brain; Python is a thin convenience layer.

**Core approach**: PostgreSQL-native cognitive architecture. All state lives in Postgres (ACID for cognition). Memories stored with pgvector embeddings, linked via Apache AGE knowledge graph. Autonomous heartbeat loop with energy-budgeted actions. Explicit consent, boundaries, and self-termination mechanisms.

**Maturity**: Substantially implemented. Working end-to-end system with Docker Compose deployment, CLI (`hexis init`, `hexis up`, `hexis chat`), MCP server, 80+ tools, heartbeat worker, maintenance worker. The philosophical documentation (PERSONHOOD.md, PHILOSOPHY.md, ETHICS.md) is as developed as the code. More "cognitive agent framework" than "memory server" — memory is one layer of a larger autonomy stack.

---

## Architecture

### Design Thesis

The database is the system of record for all cognitive state. This is philosophically motivated:
- **ACID for cognition** — memory updates are transactional
- **Stateless workers** — heartbeat and maintenance workers poll the database, execute, and report; can be killed/restarted without state loss
- **Language-agnostic API** — the public contract is SQL functions returning JSON

### Technical Stack

| Component | Technology |
|-----------|-----------|
| Database | PostgreSQL 15+ |
| Vector search | pgvector (HNSW indexes) |
| Knowledge graph | Apache AGE (GQL dialect) |
| Text search | pg_trgm, FTS |
| Embeddings | Ollama (default: `embeddinggemma:300m-qat-q4_0`, 768d) or OpenAI-compatible |
| LLM | Multi-provider (OpenAI, Anthropic, Grok, Gemini, Ollama, etc.) |
| Cache | UNLOGGED tables + optional Redis |
| API | Python async (asyncpg) |
| MCP | Standard MCP protocol |

### Schema

**Memories** (single table, all types):
```sql
CREATE TABLE memories (
    id UUID PRIMARY KEY,
    type memory_type,  -- episodic, semantic, procedural, strategic, worldview, goal
    status memory_status,  -- active, archived, invalidated
    content TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    importance FLOAT,
    trust_level FLOAT,  -- 0.0-1.0
    source_attribution JSONB,
    decay_rate FLOAT,
    metadata JSONB  -- type-specific fields
);
```

**Working memory** (UNLOGGED — fast writes, non-persistent):
```sql
CREATE UNLOGGED TABLE working_memory (
    content TEXT,
    embedding vector(768),
    importance, trust_level, access_count,
    promote_to_long_term BOOLEAN,
    expiry TIMESTAMPTZ
);
```

**Memory neighborhoods** (precomputed associative links):
```sql
CREATE TABLE memory_neighborhoods (
    memory_id UUID PRIMARY KEY,
    neighbors JSONB,  -- {uuid: similarity_score, ...}
    computed_at TIMESTAMPTZ,
    is_stale BOOLEAN
);
```

**Episodes** (temporal grouping with 30-min gap detection):
```sql
CREATE TABLE episodes (
    id UUID, started_at TIMESTAMPTZ, ended_at TIMESTAMPTZ,
    summary TEXT, summary_embedding vector(768),
    time_range TSTZRANGE
);
```

**Clusters** (thematic grouping with centroid embeddings):
```sql
CREATE TABLE clusters (
    cluster_type cluster_type,  -- theme, emotion, temporal, person, pattern, mixed
    name TEXT, centroid_embedding vector(768)
);
```

### Apache AGE Graph

18 edge types connecting 11 node types. Key structure:

**Node types**: `MemoryNode`, `ConceptNode`, `SelfNode`, `LifeChapterNode`, `TurningPointNode`, `NarrativeThreadNode`, `RelationshipNode`, `ValueConflictNode`, `GoalNode`, `ClusterNode`, `EpisodeNode`

**Edge types** (selected):
- `TEMPORAL_NEXT` — narrative sequence
- `CAUSES`, `CONTRADICTS`, `SUPPORTS`, `DERIVED_FROM` — reasoning
- `INSTANCE_OF`, `ASSOCIATED` — structure and spreading activation
- `HAS_BELIEF`, `EVIDENCE_FOR` — worldview
- `SUBGOAL_OF`, `BLOCKS` — goal hierarchy
- `CONTESTED_BECAUSE` — contradiction justification

Design philosophy: graph traversal is **cold-path** (offline maintenance, on-demand reasoning). Hot-path retrieval uses precomputed neighborhoods, not real-time graph queries.

---

## Memory Types

Seven types plus working memory:

| Type | Purpose | Decay | Metadata |
|------|---------|-------|----------|
| **Episodic** | Events with temporal context | Exponential | action, context, result, emotional_valence |
| **Semantic** | Facts with confidence tracking | Exponential (slower) | confidence, sources, contradictions |
| **Procedural** | Step-by-step procedures | Exponential | steps, prerequisites, success_rate |
| **Strategic** | Patterns and strategies | Exponential | pattern, supporting_evidence, context_applicability |
| **Worldview** | Beliefs, values, boundaries, personality | None (permanent) | category (belief/boundary/self/other), confidence |
| **Goal** | Active objectives | None (until completed) | priority, source, deadline, success_criteria |
| **Working** | Temporary buffer | Auto-expiry | promote_to_long_term flag |

Notable: worldview memories encode Big Five personality traits as metadata subcategories. The agent's identity is memories, not configuration — beliefs have confidence scores that evolve as evidence accumulates.

---

## Retrieval Pipeline

Three tiers, ordered by speed:

### Hot path: `fast_recall()`

Single SQL query combining three strategies:
1. **Vector similarity** — HNSW cosine distance on pgvector
2. **Neighborhood expansion** — precomputed associative neighbors from `memory_neighborhoods` (JSONB lookup, no graph traversal)
3. **Temporal context** — same-episode memories get a 0.15 boost

Scoring: `relevance = importance × exp(-decay_rate × age_days) × similarity_multiplier`

Performance: ~5-50ms on commodity hardware with 10k+ memories.

### Warm path: type-filtered and cluster-based lookups

`search_similar_memories(query, limit, types)` — adds type constraints or retrieves by cluster centroid similarity.

### Cold path: multi-hop graph traversal

Apache AGE Cypher queries following `CAUSES`, `SUPPORTS`, `CONTRADICTS`, `TEMPORAL_NEXT` edges. Used for conscious reflection, planning, contradiction handling. Runs offline or on explicit request — never per-query in the hot path.

### Comparison to Somnigraph retrieval

The hot path is structurally similar to Somnigraph's hybrid retrieval (vector + keyword + graph) but with different trade-offs:
- **No FTS5/BM25** — Hexis uses pg_trgm for text matching but the primary retrieval is vector-only. No keyword channel in the fusion.
- **Precomputed neighborhoods vs. live PPR expansion** — Hexis trades freshness for speed; Somnigraph computes graph expansion at query time.
- **No learned reranker** — scoring is a fixed formula (importance × decay × similarity). No feedback loop reshaping the ranking.
- **No RRF fusion** — single scoring formula rather than fusing multiple retrieval channels.

---

## Autonomous Heartbeat

The most distinctive feature. A periodic OODA loop (default: every 5-10 minutes):

1. **Initialize** — regenerate energy (+10/hour, max 20)
2. **Observe** — poll environment (new messages, alerts, time of day)
3. **Orient** — gather context (goals, memories, worldview, emotional state)
4. **Decide** — LLM call with full context, receives action plan
5. **Act** — execute actions atomically via batch SQL function
6. **Record** — insert episodic memory of heartbeat cycle

### Energy Budget

Energy represents **situational consequence**, not compute cost:

| Cost | Actions | Rationale |
|------|---------|-----------|
| 0 | Observe, sense memory | Free (cognition) |
| 1 | Recall, remember, explore | Low consequence |
| 2 | Web search, reflect | Moderate effort |
| 3 | Code execution, calendar | Creates artifacts |
| 5 | Send messages | Social exposure |
| 6-8 | Public/external reach | High consequence |

Context multipliers: first use of tool (×1.5), high error rate (×1.5), late-night social (×2.0), repeated similar action (×1.2).

The agent can *notice* cost discomfort (repeatedly deferring high-value actions) and *propose* cost changes with justification, but cannot directly modify costs.

### Drives

Four autonomous drives create internal pressure:

| Drive | Baseline | Accumulation | Satisfaction cooldown |
|-------|----------|-------------|---------------------|
| Curiosity | 0.4 | +0.05/hr | 4 hours |
| Connection | 0.5 | +0.03/hr | 6 hours |
| Coherence | 0.2 | +0.08/hr (when contradictions exist) | 2 hours |
| Competence | 0.3 | +0.04/hr | 5 hours |

Drives accumulate toward 1.0 unless satisfied. They nudge the heartbeat toward certain actions without commanding them.

---

## Consolidation & Maintenance

**MaintenanceWorker** runs every ~30 minutes (stateless, polls DB):

1. Embedding refresh for stale memories
2. Neighborhood recomputation (`batch_recompute_neighborhoods()`)
3. Thematic clustering and emotional signature analysis
4. Exponential decay application
5. Episode summarization
6. Working memory promotion (importance > 0.75 AND access_count > 3)
7. Pattern detection → strategic memory emission

**Key distinction from Somnigraph**: maintenance is continuous (every 30 min) vs. Somnigraph's periodic sleep phases. The maintenance worker emits *observations* (strategic memories, graph edges, pattern detections) — resolution of contradictions, goal reprioritization, and narrative decisions require the conscious layer (heartbeat).

No equivalent of Somnigraph's REM-phase question-driven consolidation, Hebbian PMI edge weighting, or LLM-mediated merge/archive/annotate decisions.

---

## Identity and Worldview

Identity is stored as memories, not configuration:

- **Big Five personality** — openness, conscientiousness, extraversion, agreeableness, neuroticism (0.0-1.0), stored as worldview memories with `subcategory='personality'`
- **Beliefs** — confidence-scored propositions that accumulate evidence
- **Boundaries** — things the agent won't do, stored as worldview memories
- **Self node** — Apache AGE vertex connecting to values, capabilities, limitations via typed edges

### Consent and Self-Termination

During `hexis init`, the agent receives a consent prompt and generates a genuine response. Consent is permanent — recorded in `consent_log`. The agent can self-terminate via the `terminate` heartbeat action (requires detailed reason + "last will" message). The agent can also pause without ending.

These mechanisms are grounded in philosophical arguments about machine personhood (PERSONHOOD.md). The framing: "the cost of wrongly denying personhood far exceeds the cost of wrongly extending consideration."

---

## MCP Interface

Memory tools exposed:

| Tool | Description | Energy cost |
|------|-------------|------------|
| `hydrate` | Rich context (memories + goals + identity + worldview) | 1.0 |
| `recall` | Vector similarity search with type filtering | 1.0 |
| `remember` | Store episodic/semantic/procedural memory | 0.5 |
| `link_memories` | Create graph edges between memories | 1.0 |
| `search_worldview` | Retrieve beliefs, boundaries, personality | 1.0 |
| `search_goals` | Active goals with status | 1.0 |

Plus batch variants (`hydrate_batch`, `recall_batch`, `remember_batch`).

80+ external tools across 11 categories (web, filesystem, shell, code, browser, calendar, email, messaging, ingest, external APIs). Energy-gated per category.

---

## Comparison to Somnigraph

### What Hexis has that we don't

**Autonomous heartbeat with energy budget.** The OODA loop is a fundamentally different model of agent behavior. Somnigraph is reactive (responds to tool calls); Hexis proactively observes, reflects, and acts on its own schedule. The energy budget is a novel constraint mechanism — situational consequence, not compute cost.

**Drives as internal pressure.** Curiosity, connection, coherence, competence — these accumulate over time and nudge behavior without commanding it. Nothing analogous in Somnigraph. The coherence drive (pressure to resolve contradictions) is particularly interesting — it creates a natural urgency to clean up the knowledge base.

**Precomputed neighborhoods.** Trading freshness for speed. Hot-path recall uses JSONB lookups instead of real-time graph traversal. At scale this is significantly faster than Somnigraph's per-query PPR expansion.

**Working memory as non-persistent buffer.** UNLOGGED tables enforce that only important information gets promoted. Somnigraph has no working memory tier — everything goes directly to long-term storage.

**Narrative structure as graph.** Life chapters, turning points, episode sequences as first-class AGE nodes with traversable edges. Somnigraph's graph has edges between memories but no narrative superstructure.

**Identity as evolving epistemology.** Worldview memories with confidence scores, Big Five personality encoding, the SelfNode connecting to values/capabilities/limitations. Somnigraph stores identity context externally (seed.md, core.md), not as memories.

**Apache AGE graph with 18 edge types.** Richer relational vocabulary than Somnigraph's support/contradict/evolve/derive/co-retrieve edges. `CAUSES`, `BLOCKS`, `CONTESTED_BECAUSE`, `INSTANCE_OF` enable reasoning patterns our simpler edge schema can't express.

### What we have that they don't

**Learned reranker.** LightGBM model trained on ground-truth relevance judgments, +5.7% NDCG@5k over hand-tuned formula. Hexis uses a fixed scoring formula with no mechanism to learn from retrieval outcomes.

**Feedback loop.** `recall_feedback()` creates a gradient signal that reshapes scoring, adjusts decay, strengthens edges, enriches themes. Hexis has no retrieval feedback mechanism — scoring parameters are static.

**Hybrid retrieval with RRF fusion.** FTS5/BM25 keyword channel fused with vector similarity via reciprocal rank fusion. Hexis is vector-primary with pg_trgm as secondary — no dedicated keyword retrieval channel.

**Sleep pipeline with LLM consolidation.** Three-phase offline processing (NREM classification, REM question-driven summarization, archiving) with per-memory LLM judgments (merge/archive/annotate/rewrite). Hexis's maintenance worker does clustering and neighborhood recomputation but no LLM-mediated consolidation decisions.

**Hebbian PMI edge weighting.** Co-retrieval patterns strengthen graph edges via pointwise mutual information. Hexis edges are created by the maintenance worker or heartbeat but don't evolve based on retrieval co-occurrence.

**Theme normalization and taxonomy.** Controlled vocabulary with normalization from raw tags to canonical themes. Hexis uses free-form metadata without vocabulary control.

**Tuning infrastructure.** Ground truth collection, Optuna hyperparameter optimization, probe_recall evaluation scripts. Hexis has no equivalent — parameters are hand-set.

**UCB exploration bonus.** Bayesian exploration/exploitation trade-off for under-retrieved memories. Hexis has no mechanism to surface memories that haven't been tested.

**Enriched embeddings.** Content + category + themes + summary concatenated before embedding. Hexis embeds raw content only (NOT NULL constraint on embedding, but no enrichment).

### Architectural trade-offs

| Dimension | Hexis | Somnigraph | Trade-off |
|-----------|-------|-----------|-----------|
| Database | PostgreSQL + pgvector + AGE | SQLite + sqlite-vec + FTS5 | Hexis: richer SQL, graph queries, ACID. Somnigraph: zero infrastructure, portable. |
| Graph traversal | Precomputed neighborhoods (fast, stale) | Per-query PPR expansion (fresh, slower) | Speed vs. freshness |
| Consolidation | Continuous maintenance (30 min) | Periodic sleep phases | Hexis: always current. Somnigraph: deeper LLM-mediated decisions. |
| Scoring | Fixed formula | Learned reranker + feedback loop | Hexis: predictable. Somnigraph: adaptive. |
| Identity | In-database (worldview memories) | External (seed.md, core.md) | Hexis: agent owns identity. Somnigraph: human owns identity. |
| Autonomy | Active (heartbeat + drives) | Reactive (tool calls only) | Fundamentally different agent model |
| Embeddings | Local (Ollama, 768d) | Cloud (OpenAI, 1536d) | Privacy vs. quality |

---

## Worth Adopting?

**Precomputed neighborhoods**: Yes, as an optimization. Currently Somnigraph does PPR expansion per query. A hybrid approach — precompute neighborhoods during sleep, use them as a fast path when available, fall back to live expansion for fresh memories — could improve latency without sacrificing freshness for recently-stored memories. The staleness flag (`is_stale`) is a clean pattern.

**Working memory tier**: Maybe. A non-persistent buffer with promotion thresholds could reduce low-value writes. But Somnigraph already has the agent (user of the MCP tools) making write decisions — adding a buffer layer may not reduce noise if the agent is already filtering well. Worth tracking but not immediately actionable.

**Energy budget as design pattern**: Interesting conceptually but orthogonal to Somnigraph's scope. Somnigraph is a memory system; the energy budget is an autonomy constraint. If Somnigraph ever grows toward agent autonomy, this is a good reference implementation.

**Drives (especially coherence)**: The coherence drive — pressure to resolve contradictions that accumulates when CONTRADICTS edges exist — is an elegant mechanism. Somnigraph detects contradictions during sleep but has no urgency signal to prioritize resolution. A lightweight version (surface unresolved contradictions more aggressively) could improve knowledge base hygiene.

**Narrative graph structure**: Not worth adopting in current form. Somnigraph's edge schema covers the essential relationships. Adding life chapters and turning points would require either LLM extraction (expensive) or human annotation (labor). The narrative structure is valuable for Hexis's autonomous agent model but doesn't serve a reactive memory system.

---

## Worth Watching

**Scale behavior.** Hexis's precomputed neighborhoods and HNSW indexing are designed for larger scale than Somnigraph has tested. If Somnigraph grows past ~5k memories, Hexis's performance patterns at scale would be informative.

**Energy budget evolution.** The cost multipliers (first use, error rate, time-of-day) are a sophisticated constraint model. If the community tunes these in practice, the resulting cost tables would be useful data about which autonomous actions actually carry risk.

**Contradiction handling maturity.** The CONTRADICTS edges + coherence drive + conscious resolution pipeline is more developed than most systems' contradiction handling. Watching how well this works in practice — does the agent actually resolve contradictions well, or does it generate noise? — would inform Somnigraph's own contradiction-flagged edge handling.

**Community adoption patterns.** Hexis targets a different audience (people who want autonomous AI agents with identity) than Somnigraph (people building memory-augmented AI tools). If it gains adoption, the failure modes will differ from ours and may surface problems we haven't considered.

---

## Key Claims

1. **"The database is the brain"** — all cognitive state in PostgreSQL, stateless workers. *Evidence*: the architecture genuinely follows through on this. SQL functions are the primary API; Python wraps them thinly.

2. **Hot-path retrieval in 5-50ms** — via precomputed neighborhoods. *Evidence*: plausible given HNSW + JSONB lookup, but no published benchmarks.

3. **Energy budget prevents reckless autonomy** — situational cost, not compute cost. *Evidence*: the cost model is thoughtfully designed. Whether it works in practice depends on LLM judgment during heartbeats.

4. **Identity emerges from memory** — worldview memories with evolving confidence, not hard-coded personality. *Evidence*: the schema supports this. Whether emergence actually occurs requires longitudinal observation.

---

## Relevance to Somnigraph

**High** for architectural comparison (different design philosophy for overlapping problems), **medium** for borrowable ideas (precomputed neighborhoods, coherence drive), **low** for direct code reuse (PostgreSQL-native, fundamentally different stack).

The most valuable insight: Hexis treats identity and autonomy as first-class concerns that the memory system serves. Somnigraph treats memory as the first-class concern that serves whatever agent uses it. Both are valid; the comparison highlights what each prioritizes and what each leaves to the user.
