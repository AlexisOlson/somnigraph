# OmniMemory (OmniNode-ai/omnimemory) — Analysis

*Generated 2026-03-17 by Opus agent reading local clone*

---

## Repo Overview

**Repo**: https://github.com/OmniNode-ai/omnimemory
**License**: Apache 2.0
**Language**: Python (3.12+)
**Description**: "Memory subsystem for OmniNode agents, built on the ONEX 4-node architecture"
**Author**: OmniNode AI

**Problem addressed**: Enterprise-grade persistent memory for multi-agent systems. Not a standalone memory server but a subsystem within OmniNode's larger platform — memory as one capability among many, coordinated through a shared architectural discipline.

**Core approach**: Multi-backend storage (Qdrant vectors, Memgraph graph, PostgreSQL state, Valkey cache, filesystem archive) orchestrated through the ONEX 4-node pattern (EFFECT/COMPUTE/REDUCER/ORCHESTRATOR). All operations typed through strict Pydantic models with zero `Any` types. Explicit lifecycle state machine with optimistic concurrency. Integrated PII detection and intent event processing.

**Maturity**: Early-to-mid production. Core storage, retrieval, lifecycle, and PII detection are implemented and tested with performance benchmarks. REDUCER nodes (consolidation, statistics) are scaffolded but not complete. Agent coordination is in progress. The codebase is disciplined (~20K+ LOC) but the system hasn't been validated at scale with real agent workloads.

---

## Architecture

### Design Thesis

The ONEX 4-node architecture imposes a strict classification on every operation:

- **EFFECT nodes** — I/O boundary. All external interactions (Kafka ingestion, Qdrant queries, filesystem persistence, embedding API calls) go through effects. No business logic.
- **COMPUTE nodes** — Pure transformation. Semantic analysis, entity extraction, similarity computation. Deterministic given inputs.
- **REDUCER nodes** — Aggregation. Consolidation, statistics, deduplication. Turn many items into fewer items.
- **ORCHESTRATOR nodes** — Workflow coordination. Lifecycle management, cross-agent routing. No direct I/O — delegate to effects.

This is architecturally strict: EFFECT nodes cannot import Kafka directly (mediated through adapters), COMPUTE nodes cannot touch storage, and handlers receive all dependencies via constructor injection from a DI container (`ModelONEXContainer`).

### Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Vector DB | Qdrant | Semantic similarity search |
| Graph DB | Memgraph | Relationship traversal (Cypher + BFS) |
| RDBMS | PostgreSQL | Transactional state, lifecycle tracking |
| Cache | Valkey (Redis-compatible) | Sub-ms ephemeral lookups |
| Cold storage | Filesystem (gzip JSONL) | Archive tier |
| Embeddings | Provider-agnostic HTTP adapter | Rate-limited, with RPM/TPM controls |
| Events | Kafka | Intent event ingestion |
| Document parsing | Kreuzberg | External document processing |

### Data Model

**Core Memory Item** (`ModelMemoryItem`):
```
item_id: UUID
item_type: str
content: str (up to 1MB)
title, summary: str
tags, keywords: list[str]
storage_type, storage_location: str
version: int, previous_version_id: UUID | None
created_at, updated_at, expires_at, archived_at: datetime
state: ACTIVE | STALE | EXPIRED | ARCHIVED | DELETED
lifecycle_revision: int (optimistic concurrency token)
importance_score, relevance_score, quality_score: float (0-1)
access_count: int, last_accessed_at: datetime
parent_item_id: UUID | None, related_item_ids: list[UUID]
processing_complete, indexed: bool
```

All models crossing subsystem boundaries are `frozen=True` (immutable). Every Pydantic field requires an explicit `description`. No `Any` types anywhere in the codebase.

### Lifecycle State Machine

```
ACTIVE ──> STALE ──> EXPIRED ──> ARCHIVED ──> DELETED
```

- **STALE**: Soft TTL exceeded. Still accessible but flagged for review. This intermediate state is the most interesting design choice — it allows graceful degradation rather than a binary alive/dead transition.
- **EXPIRED**: Hard TTL exceeded. Pending archive. Read-only.
- **ARCHIVED**: Moved to cold storage (gzip JSONL). Read-only.
- **DELETED**: Soft delete. Audit trail preserved.

Transitions use optimistic locking: `WHERE lifecycle_revision = :expected` prevents concurrent state corruption.

---

## Retrieval

### Multi-Backend Composition

OmniMemory doesn't have a single retrieval path. Each backend serves a distinct role:

**Vector path** (Qdrant): Cosine similarity on embeddings. Parameterized BFS from semantic neighbors. Configurable depth (2-5 hops), with 3x fetch overhead for score filtering.

**Graph path** (Memgraph): Cypher queries with bidirectional relationship traversal. Connection weight tracking on edges. Used for contextual discovery ("what's related to X?"), not primary retrieval.

**Transactional path** (PostgreSQL): JSONB queries for structured metadata. Lifecycle state filtering. The source of truth for item state.

**Cache path** (Valkey): Sub-ms lookups for hot items. Volatile.

### Query Model

The query interface (`ModelMemoryQuery`) supports:
- Multi-filter composition (item types, tags, keywords, storage targeting)
- Score thresholds (relevance, quality)
- Boosting (recency, popularity)
- Semantic search + fuzzy matching + query expansion
- Result customization (include/exclude metadata, highlight matches)

### Scoring

Results carry dual scoring: `relevance_score` + `confidence_score` + `combined_score`, plus match metadata (type, matched fields, highlighted content) and performance telemetry (processing time, storage source, explanation text).

No learned reranker. No feedback loop. Scoring is composed from embedding similarity, BM25, recency decay, popularity, and graph proximity — but the weights are static.

---

## What It Does Well

### 1. Lifecycle state machine with STALE intermediate

Most memory systems are binary: a memory exists or it doesn't. OmniMemory's five-state lifecycle with an explicit STALE tier allows graceful degradation. A memory past its soft TTL is still retrievable but flagged — the system (or an agent) can decide what to do with it before hard expiry. This is more nuanced than either TTL-based deletion or decay-to-zero approaches.

### 2. Optimistic concurrency for distributed lifecycle

The `lifecycle_revision` counter with CAS-style updates solves a real problem in multi-agent systems: two agents can't simultaneously expire and archive the same memory. Somnigraph's single-writer assumption works for single-user personal memory but wouldn't scale to concurrent agents.

### 3. Frozen models at boundaries

All models crossing subsystem boundaries are immutable (`frozen=True`). This prevents an entire class of mutation bugs that plague distributed systems where objects are passed by reference through multiple layers. Trade-off is verbosity (must construct new instances for updates), but the safety guarantee is real.

### 4. Integrated PII detection

Built-in regex-based PII scanning (email, phone, SSN, credit card, IP, API keys, password hashes) with configurable sensitivity levels. Performance budgeted at <10ms overhead. Not perfect (no NER for names/addresses), but having it in the write path at all is ahead of most memory systems, which store whatever they're given.

### 5. Intent event sourcing

First-class support for intent classification events from Kafka. Intents are stored in the graph with session linking, enabling queries like "what was this agent trying to do?" This bridges memory and decision-making in a way that pure memory systems don't attempt.

### 6. Architectural discipline

The ONEX 4-node pattern enforces clean separation. Handlers are testable (DI injection), adapters are swappable, protocols define explicit contracts. This matters less for correctness than for maintainability at team scale — the architecture constrains how code grows.

---

## Where It Falls Short

### 1. No feedback loop

This is the most significant gap. Scoring is a fixed composition of similarity, BM25, recency, and graph proximity. Nothing learns from retrieval outcomes. There's no mechanism to discover that a memory is consistently useful (or useless) and adjust accordingly. Every memory system we've surveyed shares this weakness except somnigraph, but OmniMemory's enterprise aspirations make it more conspicuous — a multi-agent system generates enough retrieval signal to support learning.

### 2. No consolidation

The REDUCER nodes (consolidator, statistics) are scaffolded but not implemented. No question-driven summarization, no merge/archive decisions, no contradiction detection, no abstraction layers. The lifecycle manages expiry and archival by time, not by content quality or redundancy. Memories accumulate monotonically until they expire.

### 3. No enriched embeddings

Raw content is embedded directly. No concatenation of categories, themes, summaries, or other metadata into the vector space. A-Mem and somnigraph both demonstrate that enriched embeddings significantly improve retrieval quality — embedding `content + category + themes + summary` front-loads relevance signals into the vector space.

### 4. Heavy infrastructure

Requires Qdrant + Memgraph + PostgreSQL + Valkey + (optionally) Kafka + Kreuzberg. Docker Compose orchestration. This is justified for enterprise multi-agent deployments but makes the system unusable for single-user or lightweight scenarios. Compare somnigraph (SQLite + one API key) or Mem0 (single vector store).

### 5. BFS for graph traversal

Graph retrieval uses BFS (breadth-first search) from Qdrant semantic neighbors. HippoRAG demonstrated that BFS is catastrophic for variable-depth multi-hop paths — PPR or novelty-scored expansion handles arbitrary graph topology far better. OmniMemory's configurable depth (2-5) mitigates this somewhat, but the fundamental limitation remains.

### 6. No decay mechanism

Memory scoring doesn't decay over time. The lifecycle state machine handles expiry (hard cutoff), but there's no continuous scoring decay. A memory stored yesterday and a memory stored a year ago, both with the same importance score and both in ACTIVE state, score identically. The `access_count` and `last_accessed_at` fields exist but aren't integrated into scoring as a continuous signal.

---

## Key Claims

| Claim | Evidence | Confidence |
|-------|----------|------------|
| Zero `Any` types across all models | Enforced by convention + visible in codebase (26 model files) | High — verified by inspection |
| <100ms P95 memory operations | Performance benchmarks show 2-5ms P95 (60x margin) | Medium — benchmarks exist but workload is synthetic |
| >10K records/sec bulk throughput | Tests show 30K-60K on reference hardware | Medium — same caveat |
| Optimistic concurrency prevents state corruption | Implemented with `lifecycle_revision` + CAS updates | High — the pattern is well-established |
| PII detection <10ms overhead | Benchmarks show 0.5-2ms for 5KB text | High — tested |

---

## Relevance to Somnigraph

### Ideas worth considering

1. **STALE as intermediate state** — Somnigraph's decay is continuous (EWMA), which is more expressive, but an explicit "flagged for review" state could complement it. A memory that's decayed below a threshold but hasn't been consolidated could be marked STALE and surfaced during sleep for a keep/archive/merge decision.

2. **Optimistic concurrency** — Not needed now (single-writer), but if somnigraph ever supports concurrent recall (e.g., parallel subagent recalls during the same session), optimistic locking on memory state would prevent race conditions.

3. **PII detection in write path** — Somnigraph has no PII filtering. For a personal memory system this is less critical (the user's own data), but for the public repo where others might adopt it, a basic PII scan would be responsible.

4. **Frozen boundary models** — Could catch mutation bugs in somnigraph's longer pipelines (sleep processing chains where a memory dict is passed through multiple stages).

### What's not transferable

- The multi-backend architecture adds complexity without benefit at somnigraph's scale. SQLite handles all storage roles (vector, FTS, relational, graph edges) in a single file.
- The ONEX 4-node pattern is organizational scaffolding for team-scale development. Somnigraph's ~2K LOC doesn't need this level of architectural ceremony.
- Intent event sourcing assumes a Kafka event bus and multi-agent coordination. Orthogonal to somnigraph's scope.

---

## Comparison with Similar Systems

| Dimension | OmniMemory | Somnigraph | Mem0 | Zep/Graphiti |
|-----------|------------|------------|------|-------------|
| Storage | 5 backends | SQLite (one file) | Vector store | Neo4j + vector |
| Retrieval feedback | None | EWMA + UCB | None | None |
| Consolidation | Scaffolded, not impl. | 3-phase sleep pipeline | None | Community detection |
| Decay | TTL states (binary) | Continuous EWMA | None | None |
| Graph | Memgraph (Cypher BFS) | SQLite edges (PPR planned) | Optional (Mem0g) | Neo4j (bi-temporal) |
| Embeddings | Raw content | Enriched (content+meta) | Raw content | Per-entity |
| PII detection | Built-in (7 types) | None | None | None |
| Lifecycle states | 5 (ACTIVE→DELETED) | Continuous decay + archive | Persist/delete | Persist/invalidate |
| Scale target | Multi-agent enterprise | Single-user personal | Multi-user SaaS | Multi-user SaaS |
| Infrastructure | Docker Compose (6 services) | SQLite + API key | Vector store | Neo4j + workers |

---

*OmniMemory represents the enterprise end of the memory system spectrum. Its architectural discipline (strict typing, frozen boundaries, protocol-first design, optimistic concurrency) solves real problems in distributed multi-agent systems. But the core retrieval intelligence — feedback loops, learned scoring, consolidation, enriched embeddings — is absent. The infrastructure is sophisticated; the memory science is conventional. For somnigraph, the most transferable ideas are defensive (PII detection, frozen boundaries, lifecycle states) rather than retrieval-oriented.*
