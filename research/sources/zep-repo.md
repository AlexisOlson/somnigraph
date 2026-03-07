# getzep/zep Repository Analysis

*Generated 2026-02-19 by Opus agent reading local clone at ~/.claude/repos/zep*

## Repository Overview

- **What it is**: Zep's open-source repository -- examples, integrations, benchmarks, an MCP server, and a **deprecated legacy Community Edition** (Go). The actual Temporal Knowledge Graph engine is **not in this repo** -- it lives in [getzep/graphiti](https://github.com/getzep/graphiti), a separate open-source Python library. The Zep Cloud product wraps Graphiti as a managed service.
- **License**: Apache 2.0
- **Languages**: Go (legacy CE + MCP server), Python (examples, benchmarks, integrations)
- **Maturity**: The README states "this repository is currently a work in progress." The legacy CE is deprecated. The valuable code is the benchmarks, integrations, ontology system, and MCP server -- all thin wrappers around the Zep Cloud API.
- **Test coverage**: Minimal. A few unit tests for the MCP server (`config_test.go`, `helpers_test.go`, `types_test.go`), basic integration tests for autogen/crewai/livekit wrappers. No tests for benchmarks. The legacy CE had no tests in the open-source tree.

## File Structure (Key Paths)

```
zep/
  legacy/src/                    # Deprecated Go Community Edition
    api/                         # REST API handlers (fact, memory, message, session, user)
    lib/
      search/rrf.go              # ** RRF implementation ** (simple, clean)
      search/mmr.go              # ** MMR implementation ** (Maximal Marginal Relevance)
      graphiti/service_ce.go     # ** HTTP client to Graphiti service ** (reveals API contract)
      config/                    # Configuration management
      pg/                        # Postgres connection utilities (bun ORM)
    models/                      # Data models: Fact, Message, Session, Memory, Search
    store/                       # Postgres store implementations
      schema_common.go           # ** DB schema ** (SessionSchema, MessageStoreSchema, UserSchema)
      memory_ce.go               # ** Memory retrieval via Graphiti ** (fact assembly)

  mcp/zep-mcp-server/           # Go MCP server wrapping Zep Cloud API
    internal/handlers/
      types.go                   # ** SearchGraphInput ** (reranker, min_fact_rating, mmr_lambda, center_node_uuid, filters)
      search.go                  # Graph search handler
      context.go                 # User context retrieval
    internal/server/tools.go     # 13 MCP tools defined
    pkg/zep/client.go            # Thin wrapper around zep-go SDK

  ontology/
    default_ontology.py          # ** Default Zep ontology ** (9 node types, 2 edge types)

  benchmarks/
    locomo/                      # LOCOMO benchmark harness (graph-based eval)
      benchmark.py               # CLI: --ingest, --eval, --cleanup
      evaluation.py              # ** Eval pipeline ** (search -> context -> response -> grade)
      ingestion.py               # Data ingestion via graph.add API
      prompts.py                 # ** Prompts ** for response generation and grading
      ontology.py                # 8 node types for benchmark
      config.py                  # GraphParams: edge_limit, reranker, node_limit, etc.
      common.py                  # EvaluationResult, BenchmarkMetrics, CompletenessGrade
    longmemeval/
      zep_longmem_eval.py        # ** LongMemEval benchmark ** (ingest + eval)

  zep-eval-harness/              # Standalone eval harness (simpler, user-focused)
    zep_evaluate.py              # ** Context completeness as PRIMARY metric **
    zep_ingest.py                # Conversation + telemetry ingestion
    ontology.py                  # Custom ontology with EntityModel/EdgeModel (5 entity, 6 edge types)

  examples/python/
    advanced.py                  # ** Rich ontology example ** (travel planning with typed entities + edges)
    graph_example/
      entity_types.py            # EntityModel + EdgeModel with typed attributes (EntityText, EntityInt)

  integrations/python/
    zep_autogen/                 # AutoGen memory integration
    zep_crewai/                  # CrewAI memory integration
    zep_livekit/                 # LiveKit voice agent integration
```

## Architecture Overview

### Core Components (from this repo + API contract)

The architecture revealed by reading the code:

1. **Graphiti Service** (external): The actual knowledge graph engine. Zep CE communicated with it via HTTP (`graphiti/service_ce.go`). Endpoints:
   - `POST /get-memory` -- retrieve facts given messages + group_id
   - `POST /messages` -- ingest messages (creates episodes, extracts entities/edges)
   - `POST /search` -- search facts by query + group_ids
   - `POST /entity-node` -- add explicit entity node
   - `GET/DELETE /entity-edge/{uuid}` -- fact (edge) management
   - `DELETE /episode/{uuid}` -- episode management
   - `DELETE /group/{groupID}` -- group deletion

2. **Fact Model** (from `graphiti/service_ce.go`):
   ```go
   type Fact struct {
       UUID      uuid.UUID
       Name      string
       Fact      string      // Natural language fact text
       CreatedAt time.Time
       ExpiredAt *time.Time  // When fact was marked expired
       ValidAt   *time.Time  // When fact became true (bi-temporal)
       InvalidAt *time.Time  // When fact stopped being true (bi-temporal)
   }
   ```

3. **Memory Retrieval** (`memory_ce.go`):
   - Takes last 4 messages (2 chat turns) for fact retrieval
   - Calls Graphiti's `get-memory` with `MaxFacts: 5`
   - Maps `ValidAt` to `CreatedAt` for display (temporal projection)
   - Groups by either UserID or SessionID

4. **Search Pipeline** (from MCP + eval harness):
   - Parallel search across nodes, edges, and optionally episodes
   - 5 reranking strategies: `rrf`, `mmr`, `cross_encoder`, `node_distance`, `episode_mentions`
   - Filtering by `node_labels`, `edge_types`, `min_fact_rating`
   - MMR supports `mmr_lambda` parameter
   - Node distance reranking uses `center_node_uuid`

5. **Ontology System** (from `ontology/` and examples):
   - Typed entity nodes using Pydantic `EntityModel` base class
   - Typed edges using Pydantic `EdgeModel` base class
   - Typed attributes: `EntityText`, `EntityInt`
   - Source-target constraints: `EntityEdgeSourceTarget(source="User", target="Destination")`
   - Per-user or per-graph ontology application
   - Default ontology: 9 node types (User, Assistant, Preference, Location, Event, Object, Topic, Organization, Document) + 2 edge types (LOCATED_AT, OCCURRED_AT)

6. **Evaluation Framework** (from benchmarks):
   - Two metrics: **Context Completeness** (PRIMARY) and **Answer Accuracy** (SECONDARY)
   - Context completeness grades: COMPLETE, PARTIAL, INSUFFICIENT
   - Correlation analysis: accuracy-when-complete vs overall
   - Per-category and per-difficulty breakdowns
   - Latency statistics (median, p95, p99)
   - Token usage tracking

### Storage

- **Legacy CE**: PostgreSQL via `bun` ORM. Tables: sessions, messages, users. Schemas are tenant-isolated (`schemaName.tableName`).
- **Zep Cloud**: Neo4j for the knowledge graph (from paper), managed service.
- Embeddings stored alongside facts (visible in `SessionSearchResult.Embedding []float32`)

### Processing Pipeline

Messages flow through: `PutMemory` -> `_initializeProcessingMemory` -> Graphiti `POST /messages` (async processing) -> entity extraction + edge creation + temporal reasoning. The CE code shows messages are sent to BOTH session-scoped and user-scoped groups if a UserID exists, enabling cross-session knowledge.

## Unique Concepts

### 1. Typed Ontology with Constrained Edge Source/Target

The most distinctive feature is the Pydantic-based ontology system. Entity types and edge types are defined as classes with typed, searchable attributes. Edge types have explicit source-target constraints (e.g., `WORKS_FOR` can only go from `User` or `Person` to `Organization`). This constrains the graph structure and makes extraction more precise.

```python
class Person(EntityModel):
    relationship: EntityText = Field(description="family, friend, colleague...")

class WorksFor(EdgeModel):
    ...

# Edge constraint: Person -> Organization only
"WORKS_FOR": (WorksFor, [EntityEdgeSourceTarget(source="Person", target="Organization")])
```

### 2. Preference Entity Type as First-Class Citizen

The default ontology explicitly prioritizes `Preference` over most other entity types:
> "IMPORTANT: Prioritize this classification over ALL other classifications except User and Assistant."

This suggests Zep learned that preference tracking is critical for agent memory and needs to be extracted aggressively.

### 3. Context Completeness as Primary Evaluation Metric

The evaluation harness uses context completeness (not answer accuracy) as the primary metric. This separates retrieval quality from LLM generation quality -- a valuable design choice for benchmarking memory systems specifically.

### 4. Five Reranking Strategies

The search system supports five distinct rerankers: `rrf`, `mmr`, `cross_encoder`, `node_distance`, `episode_mentions`. The benchmark configs show `cross_encoder` is preferred for edges and `rrf` for nodes.

### 5. Bi-temporal Fact Display

Facts carry both `valid_at`/`invalid_at` (application time) and `created_at`/`expired_at` (system time). The context assembly shows facts as: `"User prefers mild food (Date range: 2024-01-15 - present)"` with explicit "present" for currently valid facts. The prompts carefully instruct: "Facts ending in 'present' are currently valid. Facts with a past end date are NO LONGER VALID."

## Worth Stealing (ranked)

### 1. RRF Implementation -- trivial effort, high value
The `rrf.go` implementation is 48 lines of clean, generic Go code. The core algorithm: sum reciprocal ranks across result sets, sort by combined score. Uses generics (`Rankable` interface). Our claude-memory already has this concept in the design but seeing the clean implementation is useful.

**Key detail**: Their RRF does NOT use a `k` constant (standard formula is `1/(k+rank)`). They use `1/(rank+1)`. This means rank 1 gets score 1.0, rank 2 gets 0.5, etc. The standard `k=60` parameter dampens the influence of high-ranking items. Their version gives more weight to top results.

### 2. Ontology-Constrained Edge Types -- medium effort, high value
Defining edge types with explicit source/target entity constraints is a powerful idea for reducing extraction errors. For claude-memory, this could mean defining relationship types like `CORRECTS(session -> memory)` or `CONTRADICTS(memory -> memory)` with source/target constraints.

### 3. Context Completeness Evaluation -- low effort, high value
Separating retrieval evaluation from generation evaluation. For claude-memory testing, we could evaluate: "Given this recall query, does the returned context contain all necessary information?" This is more diagnostic than end-to-end accuracy.

### 4. Preference-First Entity Extraction Priority -- trivial effort, medium value
Explicitly prioritizing preference extraction over generic entities in prompts. For claude-memory, preferences are already a category, but the prompt engineering pattern of "IMPORTANT: Prioritize this classification over ALL other classifications" is worth noting for our extraction prompts.

### 5. Dual-Group Indexing (Session + User) -- low effort, medium value
`memory_ce.go` shows that when a UserID exists, messages are sent to BOTH session-scoped and user-scoped groups. This enables cross-session retrieval. For claude-memory, we already have user-level memory, but the explicit dual-indexing pattern is worth noting.

### 6. Temporal Context Assembly Pattern -- trivial effort, medium value
The way facts are rendered as `"fact text (Date range: valid_at - invalid_at|present)"` with explicit prompt instructions about validity is a clean pattern for presenting temporal information to LLMs.

### 7. MaxFacts=5 + Last 4 Messages Heuristic -- zero effort, informational
The production retrieval uses only the last 4 messages (2 chat turns) for context-aware fact retrieval, and returns max 5 facts. This is surprisingly parsimonious. For claude-memory's startup_load and recall, this suggests even small fact sets can be effective.

## Not Worth It

### 1. Cross-Encoder Reranking
The benchmarks prefer cross-encoder reranking for best accuracy, but this requires a hosted ML model for inference -- too heavyweight for claude-memory's architecture. RRF is sufficient and runs locally.

### 2. Neo4j/Graph Database
Zep Cloud uses Neo4j for the knowledge graph. For claude-memory, Supabase/pgvector with a `relationships` table is sufficient. We don't need native graph traversal for our scale.

### 3. Typed Entity Attributes (EntityText, EntityInt)
While the ontology system is conceptually interesting, the typed attribute system (`EntityText`, `EntityInt` fields on entity models) adds complexity without clear benefit for our use case. Our memories are already structured with category, priority, and themes.

### 4. MCP Server Architecture
The Go MCP server is a thin wrapper around the Zep Cloud API. Not applicable since claude-memory already has its own MCP server in TypeScript.

### 5. Episode-Level Retrieval
Searching raw episodes (original messages) alongside extracted facts adds recall but also noise. For claude-memory, we already store episodic memories as a category.

## Key Weaknesses

### 1. This Repo is a Shell
The most critical implementation details -- fact extraction prompts, conflict resolution logic, entity resolution, graph traversal, consolidation, and temporal invalidation -- are all in the **closed-source Zep Cloud service** or in the **separate Graphiti repo**. This repo contains zero extraction prompts, zero conflict resolution code, and zero graph traversal implementation. The `graphiti/service_ce.go` is just an HTTP client that forwards to the real engine.

### 2. No Conflict Resolution Visible
The paper describes sophisticated temporal conflict detection, but the code in this repo shows only simple `valid_at`/`invalid_at` timestamps on facts. There is no visible code for:
- Detecting contradictions between facts
- Graded contradiction detection
- Fact invalidation logic
- Entity resolution/merging

### 3. No Consolidation/Background Processing
The legacy CE has a task system (`tasks_common.go`) with Watermill for message routing, but only two task types are defined: `message_embedder` and `purge_deleted`. There is no consolidation, compaction, or sleep-cycle processing visible.

### 4. Minimal Test Coverage
The benchmark harness is well-structured but has very few unit tests. The ontology system has no tests. The legacy CE store implementations have no visible tests.

### 5. RRF Missing k Parameter
Their RRF implementation omits the standard `k=60` smoothing constant. The formula `1/(rank+1)` instead of `1/(k+rank)` makes the algorithm more sensitive to the exact ranking position, which may not be ideal when combining heterogeneous retrieval methods.

## Relevance to claude-memory

### Direct Applicability: MODERATE

This repo is more useful as a **product design reference** than as a code reference. The actual implementation we'd want to study is in the Graphiti repo (which should be analyzed separately).

**Most directly applicable:**

1. **RRF algorithm**: Clean reference implementation confirms our design direction. Note the missing `k` parameter -- we should include it (standard `k=60`).

2. **Evaluation methodology**: Context completeness as a primary metric, separated from answer accuracy. We should adopt this for claude-memory's testing. The `CompletenessGrade` (COMPLETE/PARTIAL/INSUFFICIENT) pattern is clean.

3. **Ontology design patterns**: The idea of constraining edge types with source/target entity types. We could define: `SUPERSEDES(memory -> memory)`, `RELATES_TO(memory -> memory)`, `DERIVED_FROM(memory -> session)`.

4. **Temporal rendering in prompts**: The `"fact (Date range: from - to|present)"` pattern with explicit instructions about validity interpretation is a good prompt engineering pattern.

5. **Preference prioritization**: Making preferences a first-class concern in extraction, not just another entity type.

**Not applicable:**

- The Graphiti service contract (HTTP API) is specific to their architecture
- Neo4j graph storage is overkill for our scale
- The MCP server is a different architecture than ours
- Cross-encoder reranking requires hosted ML models

### Missing Pieces (need Graphiti repo)

To complete the analysis of Zep's implementation, the **Graphiti repository** would need to be examined for:
- Fact extraction prompts (the LLM calls that extract entities and relationships from text)
- Conflict/contradiction detection logic
- Entity resolution and deduplication
- Graph traversal (BFS) implementation
- Temporal invalidation rules
- The actual RRF/MMR integration with graph structure (cosine + BM25 + graph distance)

### Key Takeaway

Zep/Graphiti is the most production-mature temporal knowledge graph for agent memory, but this particular repository is primarily a **consumer** of the engine, not the engine itself. The most valuable artifacts here are the evaluation harness design, the ontology pattern, and the clean RRF implementation. For the core implementation details that the paper describes (and that are most relevant to claude-memory's architecture), the Graphiti repo is essential.
