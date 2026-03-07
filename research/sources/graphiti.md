# Graphiti -- Source Analysis

*Phase 13, 2026-03-05. Analysis of getzep/graphiti (Zep's production implementation).*

## 1. Architecture Overview

**Language:** Python 3.10+, fully async. ~15k LOC in `graphiti_core/`.

**Key dependencies:** Pydantic (models/validation), OpenAI/Anthropic/Gemini/Groq (LLM clients), numpy (cosine similarity/MMR), FalkorDB/Neo4j/Kuzu/Neptune (graph backends), FastMCP (MCP server).

**File structure:**
- `graphiti_core/graphiti.py` -- Main `Graphiti` class, orchestrates `add_episode`, `search`, `add_episode_bulk`
- `graphiti_core/nodes.py` -- Node types: `EpisodicNode`, `EntityNode`, `CommunityNode`, `SagaNode`
- `graphiti_core/edges.py` -- Edge types: `EntityEdge`, `EpisodicEdge`, `CommunityEdge`, `HasEpisodeEdge`, `NextEpisodeEdge`
- `graphiti_core/prompts/` -- All LLM prompts (extraction, dedup, edge resolution)
- `graphiti_core/search/` -- Hybrid search with configurable recipes
- `graphiti_core/utils/maintenance/` -- Dedup helpers, node/edge operations, community operations
- `graphiti_core/driver/` -- Multi-backend driver abstraction (Neo4j, FalkorDB, Kuzu, Neptune)
- `mcp_server/` -- MCP server exposing Graphiti as tools

**Storage model:** Property graph in Neo4j/FalkorDB. Nodes and edges stored as Cypher entities with UUID primary keys. Embeddings stored as node/edge properties. Fulltext indices for BM25. Group partitioning via `group_id` field.

## 2. Entity Resolution

Graphiti has a sophisticated two-pass entity resolution pipeline:

### Pass 1: Deterministic (dedup_helpers.py)

Fast heuristic matching that avoids LLM calls when possible:

1. **Exact match:** Normalize name (lowercase, collapse whitespace). If exactly one existing node matches, resolve immediately.
2. **Fuzzy match via MinHash LSH:** For names with sufficient entropy (Shannon entropy >= 1.5, min 6 chars or 2 tokens):
   - Compute 3-gram shingles from normalized name
   - Generate MinHash signature (32 permutations)
   - LSH banding (band size 4) to find candidate matches
   - Jaccard similarity threshold: **0.9** (very strict)
   - If best candidate exceeds threshold, resolve without LLM
3. **Low entropy guard:** Short/repetitive names (e.g., "AI", "IT") skip fuzzy matching entirely and go to LLM

### Pass 2: LLM (dedupe_nodes.py -> node_operations.py)

Unresolved nodes are batched into a single LLM call. The prompt (`dedupe_nodes.nodes`) provides:
- Previous messages for context
- Current message
- Extracted entities with IDs, names, and type descriptions
- Existing entities with names, types, attributes

The LLM returns `NodeResolutions` -- a list of `{id, name, duplicate_name}` where `duplicate_name` is either empty string (no match) or the name of an existing entity.

**Alias handling:** No explicit alias table. Entity resolution relies on the LLM understanding semantic equivalence ("descriptive label in existing_entities clearly refers to a named entity in context"). The prompt instructs: "Semantic Equivalence: if a descriptive label in existing_entities clearly refers to a named entity in context, treat them as duplicates." Resolved nodes adopt the "best full name."

**Duplicate tracking:** When a node is resolved as a duplicate, the system creates an `IS_DUPLICATE_OF` edge between the extracted node and the existing node (checked via `filter_existing_duplicate_of_edges`).

### Key design choices:
- Candidate nodes are found via **hybrid search per extracted name** (BM25 + cosine similarity on name embeddings), not brute-force comparison
- The two-pass approach means most exact/near-exact matches never hit the LLM
- Robust error handling for malformed LLM responses (invalid IDs, missing resolutions, duplicate IDs)

## 3. Conflict Resolution

Graphiti has explicit contradiction detection and temporal invalidation:

### Edge Dedup + Contradiction Prompt (dedupe_edges.py)

The `resolve_edge` prompt receives:
- **EXISTING FACTS** (edges between same endpoints, found via hybrid search) with continuous idx
- **FACT INVALIDATION CANDIDATES** (broader search results, idx continues from existing)
- **NEW FACT** (the extracted edge)

Returns `EdgeDuplicate`:
- `duplicate_facts: list[int]` -- idx values from EXISTING FACTS only
- `contradicted_facts: list[int]` -- idx values from either list

### Temporal Invalidation Logic (edge_operations.py `resolve_edge_contradictions`)

When contradiction is detected, temporal comparison determines invalidation:

```python
# If old edge became valid before new edge, invalidate old edge
if edge_valid_at < resolved_edge_valid_at:
    edge.invalid_at = resolved_edge.valid_at
    edge.expired_at = now
```

Also handles reverse case: if contradicted facts have *later* `valid_at`, the **new** edge gets expired:

```python
# Expire new edge since we have information about more recent events
if candidate_valid_at > resolved_edge_valid_at:
    resolved_edge.invalid_at = candidate.valid_at
    resolved_edge.expired_at = now
```

### Key properties:
- **Not binary:** An edge can be both a duplicate AND contradicted (e.g., same fact but new info supersedes it)
- **Temporal ordering matters:** Newer facts invalidate older ones, but if the "new" fact is actually about an earlier time period, it gets expired instead
- **Three-field temporal model:** `created_at` (when ingested), `valid_at` (when fact became true), `invalid_at` (when fact stopped being true), `expired_at` (when node was marked stale in the system)
- **Fast path:** Verbatim duplicate detection (same endpoints + normalized fact text) skips LLM entirely

## 4. Temporal Graph Model

### Entity Edges (EntityEdge)
- `created_at: datetime` -- ingestion timestamp
- `valid_at: datetime | None` -- when the relationship became true
- `invalid_at: datetime | None` -- when the relationship stopped being true
- `expired_at: datetime | None` -- when the system marked this edge as superseded

### Temporal Extraction
Edges extract `valid_at` and `invalid_at` directly from text via LLM. The extraction prompt includes `REFERENCE_TIME` (episode timestamp) and instructs:
- "If the fact is ongoing (present tense), set valid_at to REFERENCE_TIME"
- "If a change/termination is expressed, set invalid_at to the relevant timestamp"
- "Leave both fields null if no explicit or resolvable time is stated"
- ISO 8601 format with UTC

### Episode Temporal Model
- `EpisodicNode.valid_at` -- when the original document was created
- `EpisodicNode.created_at` -- when ingested into the system
- Episodes form ordered chains via `NEXT_EPISODE` edges within a Saga

### Sagas
A `SagaNode` groups related episodes. `HAS_EPISODE` edges connect saga to episodes. `NEXT_EPISODE` edges form a temporal chain. This provides conversation-level ordering.

### Limitations:
- No explicit "as-of" temporal queries in the search layer
- Expired edges aren't filtered by default in search (they're still returned)
- No temporal decay scoring in search/retrieval

## 5. FalkorDB Schema

### Node Labels
- `Entity` -- entities extracted from text (name, summary, name_embedding, attributes, group_id)
- `Episodic` -- episode nodes (content, source, valid_at, entity_edges list)
- `Community` -- community summary nodes (name, summary, name_embedding)
- `Saga` -- groups related episodes (name, group_id)

### Edge Types (Cypher relationships)
- `RELATES_TO` -- entity-to-entity facts (name, fact, fact_embedding, episodes, valid_at, invalid_at, expired_at, attributes)
- `MENTIONS` -- episode-to-entity links
- `HAS_MEMBER` -- community-to-entity membership
- `HAS_EPISODE` -- saga-to-episode links
- `NEXT_EPISODE` -- episode ordering within a saga

### Indices
**Range indices** on: Entity(uuid, group_id, name, created_at), Episodic(uuid, group_id, created_at, valid_at), Community(uuid), Saga(uuid, group_id, name), RELATES_TO(uuid, group_id, name, created_at, expired_at, valid_at, invalid_at), MENTIONS(uuid, group_id), etc.

**Fulltext indices** (FalkorDB): `node_name_and_summary` (Entity name+summary), `edge_name_and_fact` (RELATES_TO name+fact), `episode_content` (Episodic content), `community_name` (Community name+summary). Uses RedisSearch syntax with stopword filtering.

**Vector storage:** Embeddings stored directly as node/edge properties (`name_embedding`, `fact_embedding`). Cosine similarity computed in Cypher via `vec.euclideanDistance` (FalkorDB) or `gds.similarity.cosine` (Neo4j).

### Cypher Patterns
- Node save: `MERGE (n:Entity {uuid: $uuid}) SET n += $entity_data SET n:Label1:Label2`
- Edge save: `MATCH (source {uuid: $source_uuid}), (target {uuid: $target_uuid}) MERGE (source)-[e:RELATES_TO {uuid: $uuid}]->(target) SET e += $edge_data`
- BFS traversal: `MATCH (origin {uuid: origin_uuid})-[:RELATES_TO|MENTIONS*1..{max_depth}]->(n:Entity)`

## 6. Extraction Pipeline

### Episode Processing Flow (graphiti.py `add_episode`)

1. **Retrieve context:** Last N episodes from same group/saga for prior context
2. **Extract nodes** (`extract_nodes`): LLM call with episode content + entity type definitions. Three prompts by source type:
   - `extract_message` -- for conversation (extracts speaker first)
   - `extract_json` -- for structured JSON data
   - `extract_text` -- for plain text
3. **Resolve nodes** (`resolve_extracted_nodes`): Two-pass dedup (deterministic + LLM)
4. **Extract edges** (`extract_edges`): LLM call with episode + resolved entities. Returns `(source, target, relation_type, fact, valid_at, invalid_at)` triples
5. **Resolve edges** (`resolve_extracted_edges`): Search for related edges, LLM dedup/contradiction, temporal invalidation
6. **Extract attributes** (`extract_attributes_from_nodes`): Custom Pydantic model attributes + batch summary generation
7. **Save** (`_process_episode_data`): Bulk write to graph DB

### Entity Type System
Custom entity types are Pydantic models passed at ingestion time. Each type has a name, description (docstring), and optional fields (extracted as attributes). The LLM classifies each entity into a type by `entity_type_id`.

### Edge Type System
Custom edge types (also Pydantic models) can constrain which entity type signatures are valid. The system matches `(source_label, target_label)` tuples to determine which edge types to offer.

### Custom Extraction Instructions
A `custom_extraction_instructions` string can be injected into both node and edge extraction prompts -- allows domain-specific guidance.

### LLM Model Selection
Two tiers: default model for extraction, `ModelSize.small` for attributes/dedup/edge resolution. Supports OpenAI, Anthropic, Gemini, Groq.

## 7. Search/Retrieval

### Search Architecture

Highly configurable via `SearchConfig` recipes. Four search domains run in parallel:

1. **Edge search** -- searches entity edges (facts)
2. **Node search** -- searches entity nodes
3. **Episode search** -- searches raw episode content
4. **Community search** -- searches community summaries

### Search Methods (per domain)
- **BM25 (fulltext):** Uses native fulltext indices. FalkorDB uses RedisSearch syntax with stopword filtering.
- **Cosine similarity:** Vector search on embeddings (name_embedding for nodes, fact_embedding for edges)
- **BFS (breadth-first search):** Graph traversal from origin nodes up to configurable depth (default 3). Uses `RELATES_TO|MENTIONS` edges.

### Reranking Strategies
- **RRF (Reciprocal Rank Fusion):** Default. `score[uuid] += 1/(rank + k)` where k=1. Simple, effective.
- **MMR (Maximal Marginal Relevance):** Diversity-promoting reranker with configurable lambda
- **Cross-encoder:** External reranker model scores query-document pairs
- **Node distance:** Uses graph proximity to a center node for relevance
- **Episode mentions:** Ranks by how many episodes reference an entity

### Pre-built Recipes
- `COMBINED_HYBRID_SEARCH_RRF` -- edges + nodes + episodes + communities with RRF
- `EDGE_HYBRID_SEARCH_RRF` -- edges only (default for `search()`)
- `NODE_HYBRID_SEARCH_RRF` -- nodes only (used for node dedup candidate collection)
- Various MMR, cross-encoder, and specialized variants

### Default Search Method
The public `search()` method on Graphiti uses `EDGE_HYBRID_SEARCH_RRF` (BM25 + cosine similarity on edges, RRF reranking). The MCP server's `search_nodes` uses `NODE_HYBRID_SEARCH_RRF`.

## 8. MCP Interface

### Tools Exposed (8 total)

1. **`add_memory`** -- Add episode to graph. Params: name, episode_body, group_id, source (text/json/message), source_description, uuid. Queued for async processing.
2. **`search_nodes`** -- Search entity nodes. Params: query, group_ids, max_nodes (10), entity_types filter. Uses NODE_HYBRID_SEARCH_RRF.
3. **`search_memory_facts`** -- Search edges/facts. Params: query, group_ids, max_facts (10), center_node_uuid. Uses default search.
4. **`delete_entity_edge`** -- Delete by UUID
5. **`delete_episode`** -- Delete by UUID
6. **`get_entity_edge`** -- Retrieve by UUID
7. **`get_episodes`** -- List episodes by group_ids
8. **`clear_graph`** -- Clear all data for group_ids
9. **`get_status`** -- Health check

### Transport
Supports stdio, SSE, and streamable HTTP. Configurable via YAML config file + CLI args.

### Queue Processing
Episodes are processed via `QueueService` -- sequential per group_id to avoid race conditions, with configurable semaphore for LLM rate limits.

## 9. Episode Processing

### Episode Lifecycle

1. **Ingestion:** Raw content arrives as `(name, body, source_type, reference_time, group_id)`
2. **Episodic Node created:** Stored with content, source type, valid_at timestamp
3. **Entity extraction:** LLM extracts entities from current message + previous episode context
4. **Node resolution:** Entities deduplicated against graph (deterministic + LLM)
5. **Edge extraction:** LLM extracts relationship triples between resolved entities
6. **Edge resolution:** Edges deduplicated, contradictions detected, temporal invalidation applied
7. **Attribute extraction:** Custom attributes + summaries generated for nodes
8. **Graph update:** Bulk write of episode, episodic edges (MENTIONS), entity nodes, entity edges (RELATES_TO)
9. **Saga linking:** Optional -- HAS_EPISODE edge from saga, NEXT_EPISODE chain between consecutive episodes

### Context Window
Previous episodes retrieved via `retrieve_episodes` -- last N (default 10) by created_at within the group/saga. This provides conversation continuity.

### Bulk Ingestion
`add_episode_bulk` processes multiple episodes with shared dedup -- extracts all nodes/edges first, deduplicates across the batch, then resolves against graph.

### Episode Types
- `message` -- "speaker: content" format, speaker always extracted first
- `json` -- structured JSON, entities extracted from fields
- `text` -- plain text

## 10. Comparison to claude-memory

### Entity Resolution
**Graphiti:** Two-pass (deterministic MinHash + LLM), `IS_DUPLICATE_OF` edges, per-extraction dedup against search candidates. Very thorough but expensive (multiple LLM calls per episode).
**claude-memory (planned #13):** Not yet implemented. Our current system has no entity model at all -- memories are edges only.
**Assessment:** Graphiti's approach is gold-standard for entity resolution. The MinHash LSH pre-filter is clever and avoids unnecessary LLM calls. Worth studying but our simpler edge-only model may not need this complexity yet.

### Temporal Model
**Graphiti:** `valid_at`/`invalid_at` on edges, `expired_at` for system-level staleness. Temporal bounds extracted by LLM. Bidirectional invalidation (newer facts expire older ones, but older facts about earlier periods also get properly expired).
**claude-memory:** `valid_from`/`valid_until` planned but not yet extracted from text. No LLM-based temporal extraction.
**Assessment:** Graphiti's temporal extraction is the most complete implementation I've seen. The bidirectional invalidation logic in `resolve_edge_contradictions` is particularly well-designed.

### Conflict Resolution
**Graphiti:** LLM-based contradiction detection + temporal invalidation. Can detect both duplicates and contradictions in the same pass. Expired edges kept in graph with timestamps.
**claude-memory (planned #3):** Graded contradiction handling planned. Currently basic.
**Assessment:** Graphiti's approach of keeping contradicted edges with `expired_at`/`invalid_at` timestamps rather than deleting them is a good pattern. The two-list approach (EXISTING FACTS + INVALIDATION CANDIDATES) with continuous indexing is clever prompt engineering.

### Search/Retrieval
**Graphiti:** RRF hybrid search (BM25 + cosine similarity), optional BFS graph traversal, multiple reranker options (MMR, cross-encoder, node distance, episode mentions). Four parallel search domains.
**claude-memory:** RRF hybrid search (BM25 + cosine similarity) on edges. Simpler but same core approach.
**Assessment:** Both use RRF. Graphiti's BFS traversal is interesting for graph-structured data. Our PPR-based graph traversal (#2) would serve a similar purpose. Graphiti's search is more configurable but also more complex.

### Memory Consolidation
**Graphiti:** No explicit consolidation/sleep cycle. Community detection via label propagation generates summaries. Node summaries updated per-episode.
**claude-memory:** NREM+REM sleep consolidation, shadow-load, startup_load, per-category decay with floor.
**Assessment:** We have a more biologically-inspired approach. Graphiti relies on LLM-generated summaries at ingestion time rather than background consolidation.

### Novelty/Decay
**Graphiti:** Episode mentions count as a reranker option. No explicit decay.
**claude-memory:** Novelty operator in edge schema, per-category decay with floor, recall_feedback boosting.
**Assessment:** Our approach is more nuanced. Graphiti lacks explicit memory aging.

### Storage
**Graphiti:** Graph database (Neo4j/FalkorDB). Heavy infrastructure dependency.
**claude-memory:** JSON sync layer. Zero infrastructure.
**Assessment:** Our approach is radically simpler for the single-user Claude Code use case. Graphiti's graph DB is necessary for their multi-tenant, multi-agent design but overkill for us.

## 11. Worth Stealing

### Ranked by value/effort:

1. **Bidirectional temporal invalidation** (effort: low) -- When detecting contradictions, check both directions: does new fact invalidate old fact, AND does old fact invalidate new fact (if new fact is about an earlier period). `resolve_edge_contradictions` logic is clean and correct. Could adapt for our graded contradiction handler.

2. **MinHash LSH pre-filter for dedup** (effort: medium) -- Avoid LLM calls for obvious matches. The Shannon entropy guard for short names is a nice touch. Applicable when we add entity resolution (#13). Self-contained in `dedup_helpers.py`.

3. **Continuous-index contradiction prompt design** (effort: low) -- The pattern of giving the LLM two lists with continuous indexing (EXISTING FACTS idx 0-N, INVALIDATION CANDIDATES idx N+1-M) and asking for both duplicate and contradiction detection in one call is efficient prompt engineering. Adaptable for our contradiction detection.

4. **Custom entity/edge types via Pydantic** (effort: medium) -- Domain-specific entity and relationship types as Pydantic models with docstrings as descriptions. Elegant type-safe approach. Not immediately needed for us but good pattern for structured memory domains.

5. **Saga-based episode ordering** (effort: low) -- Grouping conversations into sagas with NEXT_EPISODE chains. Simple but effective for maintaining conversation order. Analogous to our session logs.

6. **Fast-path verbatim duplicate detection** (effort: trivial) -- Before LLM calls, check if normalized fact text + endpoints match exactly. Cheap and effective.

## 12. Not Worth It

1. **Graph database dependency** -- Neo4j/FalkorDB adds massive operational complexity. Our JSON sync layer with MCP is far simpler for single-user Claude Code. The graph structure could be emulated with in-memory adjacency lists if we ever need graph traversal.

2. **BFS/graph traversal search** -- Graphiti's BFS over RELATES_TO edges is powerful for densely connected knowledge graphs but requires a graph DB. Our planned PPR (#2) can achieve similar results with embeddings.

3. **Community detection (label propagation)** -- Generates summary nodes for clusters of related entities. Useful for large multi-domain graphs, overkill for personal memory. Our category-based organization serves a similar purpose more simply.

4. **Multi-backend driver abstraction** -- 4 graph backends (Neo4j, FalkorDB, Kuzu, Neptune) with shared interface. Massive engineering effort for portability we don't need.

5. **Cross-encoder reranking** -- Requires a separate reranking model deployment. RRF is good enough for our scale.

6. **Custom extraction instructions per-call** -- Nice feature but adds complexity. Our CLAUDE.md-based configuration is simpler.

7. **Episode mentions reranker** -- Counts how many episodes reference an entity. Similar to our access_count but implemented as a search reranker rather than a stored field. Our approach (decay + recall_feedback) is more nuanced.
