# RedPlanet CORE — Source Analysis

*2026-03-18. Analysis of [RedPlanetHQ/core](https://github.com/RedPlanetHQ/core).*

## 1. Architecture Overview

**Language:** TypeScript/Node.js, monorepo with pnpm workspaces.

**Key dependencies:** Prisma (PostgreSQL ORM), Neo4j driver (knowledge graph), Vercel AI SDK (pluggable LLM), Remix (web app), Trigger.dev (async jobs), WebSocket (real-time).

**Storage model:** Three-layer:
- **PostgreSQL** — Relational data: ingestion queue, documents, embeddings, voice aspects, recall log, labels
- **Neo4j** — Knowledge graph: episodes, statements, entities, compacted sessions
- **Pluggable vector stores** — pgvector, Turbopuffer, or Qdrant (configurable)

**Key components:**
- Memory layer (temporal knowledge graph + vector DB)
- Toolkit layer (MCP proxy + action router, 200+ actions across 50+ app integrations)
- CORE Agent (orchestrator: intent classification, memory recall routing, tool selection, multi-agent spawning)

## 2. Data Model

Four-node graph model in Neo4j:

### EpisodicNode (Episodes)
Source content container: a conversation turn, document chunk, or recorded message. Fields: uuid, content, originalContent, contentEmbedding, metadata, source, createdAt, validAt, sessionId, queueId, chunkIndex, totalChunks. Versioned: version, contentHash, previousVersionSessionId.

### StatementNode (Atomic Facts)
Extracted from episodes via LLM: subject-predicate-object triples + raw facts. Fields: uuid, fact, factEmbedding, createdAt, validAt, invalidAt, invalidatedBy, attributes, aspect, recallCount, provenanceCount. Connected to episodes via `HAS_PROVENANCE`, to entities via `HAS_SUBJECT`, `HAS_PREDICATE`, `HAS_OBJECT`.

### EntityNode (Named Things)
People, organizations, places, projects, tasks, technologies, products, concepts, predicates. 10 entity types. Deduplication via `mergeEntities()` and `deduplicateEntitiesByName()`. Orphan cleanup via `deleteOrphanedEntities()`.

### CompactedSessionNode (Consolidation)
Compressed summary of multiple episodes. Fields: uuid, sessionId, summary, summaryEmbedding, episodeCount, startTime, endTime, confidence, compressionRatio, metadata. Connected via `COMPACTS` → episodes.

## 3. Statement Classification (Aspects)

12 statement aspects split into two groups:

**Graph Aspects** (what the user observes):
- Identity — role, location, affiliation (slow-changing)
- Relationship — connections between people
- Event — specific occurrences with timestamps
- Decision — choices made, conclusions reached
- Knowledge — expertise, skills, understanding
- Problem — blockers, issues, challenges
- Task — one-time commitments, action items

**Voice Aspects** (what the user speaks — stored complete, not decomposed):
- Directive — standing rules ("always X", "notify me if Y")
- Preference — likes, dislikes, style
- Habit — recurring behaviors, routines, patterns
- Belief — values, opinions, reasoning
- Goal — future targets, aspirations

The dual storage (Graph for SPO triples, Voice for complete statements) allows fine-grained fact retrieval alongside undecomposed user-expressed rules.

## 4. Ingestion Pipeline

1. Activity capture: webhooks from integrations → IngestionQueue
2. Extraction: LLM decomposes episodes into statements (SPO triples + facts)
3. Entity linking: resolve entity names to UUIDs, deduplicate synonyms
4. Contradiction detection: flag statements conflicting with existing facts
5. Invalidation: mark contradicted statements with `invalidAt` timestamp
6. Embedding: compute contentEmbedding (episodes) and factEmbedding (statements) via Vercel AI SDK

Dual-model calling: high-cost model (Claude) for extraction/classification, low-cost model for parallel extraction.

## 5. Retrieval

Multi-strategy retrieval (SearchV2):

1. **BM25 full-text search** — keyword relevance on statement facts
2. **Semantic (vector) search** — cosine similarity on statement embeddings, threshold ≥ 0.5
3. **Aspect-based queries** — filter statements by aspect type
4. **Entity lookup** — find episodes mentioning specific entities
5. **Temporal queries** — time range + aspect + label filters on validAt/event_date
6. **Relationship traversal** — find facts linking two entities
7. **Exploratory queries** — label-based filtering for unstructured intent
8. **Graph traversal (BFS)** — multi-hop entity reachability by hop distance
9. **Faceted navigation** — distinct labels/entities/aspects with counts

Ranking combines BM25 score, semantic score, recency, temporal relevance, aspect match, label match, and recall frequency.

## 6. Session Compaction (Consolidation)

Triggered after N episodes (threshold: 1 new episode). Process:
1. Fetch all episodes for a session
2. LLM produces consolidated narrative summarizing all episodes
3. Output stored in Neo4j (CompactedSessionNode with COMPACTS relationships) and PostgreSQL (Document table)
4. Compaction **replaces** episodes in agent recall — agents see summaries, never originals

Compaction is lossy by design. Re-versioned if session continues (old compact → previousVersionSessionId).

## 7. Temporal Handling

Statements have lifecycle:
- `validAt` — when fact becomes true (default now)
- `invalidAt` — when fact becomes false (default null = ongoing)
- `invalidatedBy` — UUID of episode that contradicted this

Contradiction detection: `findContradictoryStatements()` finds valid statements with same subject+predicate. When new statement created, old ones marked with invalidAt. Binary (valid/invalid) — no graded classification.

No decay model. Relies on temporal filtering and recency ranking. `recallCount` tracks access frequency but doesn't drive a formal scoring function.

## 8. Access Tracking

Every recall logged with: access type, search method, target type, parameters (query, minSimilarity, maxResults), results (count, similarity, response time), context (source, session). No feedback loop — access is logged but doesn't reshape future retrieval.

## 9. Integration Surface

- MCP Plugin for Claude Code — auto-loads persona at session start
- REST API — direct SDK calls
- Conversation ingestion — captures sessions into memory
- 200+ toolkit actions across GitHub, Slack, Linear, Gmail, Calendar, Notion, etc.

## 10. Key Claims

- **88.24% accuracy on LoCoMo** — temporal reasoning, multi-hop entity resolution, open-domain recall. This is the highest reported LoCoMo score in our corpus. Not directly comparable to Somnigraph's fastembed R@10 of 67.3% (different metric, different embedding backend, different retrieval strategy).
- **Multi-workspace isolation** — users/workspaces fully partitioned

## Relevance to Somnigraph

### What's stronger than us

**Entity extraction.** CORE builds a proper knowledge graph with typed entities and SPO triples at ingest time. Somnigraph has no entity extraction — memories are atomic blobs. CORE's `mergeEntities()` + name dedup is more practical than the AriGraph exact-match approach we already flagged as a negative result (Phase 14). This directly informs our gap item #13 (lightweight entity resolution).

**Aspect classification.** The 12-type Voice/Graph split is substantially richer than our 5 categories (episodic, semantic, procedural, reflection, meta). The "Voice" aspects — directive, preference, habit, belief, goal — capture distinctions that Somnigraph lumps under "procedural" or "semantic." The Voice/Graph duality (decomposed triples vs. complete statements) is an interesting design choice for preserving user intent without losing structured queryability.

**Multi-hop via entity graph.** Neo4j with typed entities naturally answers "who worked with whom on what" queries. Somnigraph's PPR walks memory-memory edges, which is less natural for entity-centric queries.

**Integration breadth.** 200+ actions across 50+ apps vs. Claude Code MCP only.

### What's weaker than us

**No feedback loop.** The biggest gap. `recallCount` is the only signal — no utility scoring, no Bayes prior, no EWMA, no Hebbian co-retrieval. Retrieval quality doesn't improve with use.

**No decay.** No time-based forgetting, no dormancy, no shadow load. Old facts persist equally with new ones. Relies entirely on temporal filtering and recency ranking.

**Shallow consolidation.** Session compaction is LLM summarization only — no relationship detection, no contradiction classification, no decay correction, no summary maintenance. Compaction is lossy (replaces originals) without the multi-layer preservation (detail → summary → gestalt) that Somnigraph's CLS hierarchy provides.

**Binary contradiction handling.** Valid/invalid with no graded classification (hard/soft/temporal/contextual). No contradiction edges, no tension detection, no temporal evolution tracking.

**BFS graph traversal.** HippoRAG's ablation showed BFS drops Recall@2 from 40.9 to 25.4 vs. PPR. CORE uses BFS.

**No retrieval tuning methodology.** No documented parameter optimization, no ground truth, no evaluation infrastructure.

**Heavy infrastructure.** Neo4j + PostgreSQL + vector DB + Trigger.dev + WebSocket vs. single SQLite file.

### Ideas worth investigating

1. **Statement decomposition at write time.** Even partial decomposition (extracting entities + predicates from `remember()` content) could feed the graph without full SPO overhead. CORE demonstrates the value; the question is whether the LLM cost at write time is justified vs. Somnigraph's zero-LLM write path with offline sleep.

2. **Voice/Graph aspect duality.** Storing some memories as complete undecomposed statements (directives, preferences) while decomposing others into structured triples. This could complement Somnigraph's category system without replacing it.

3. **`superseded_by` explicit chains.** CORE's `invalidatedBy` field creates traversable contradiction chains. Combined with memv's `superseded_by` concept, this is a stronger schema for temporal evolution than Somnigraph's current revision edges.
