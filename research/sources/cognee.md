# cognee (topoteretes/cognee) -- Analysis

*Generated 2026-02-20 by Opus agent analyzing GitHub repo via API*
*Updated 2026-03-27: v0.5.3–v0.5.5 review (skills system, triplet embeddings as default, entity consolidation)*

---

## Repo Overview

**Repo**: https://github.com/topoteretes/cognee
**Stars**: 12,424 | **Forks**: 1,226 | **License**: Apache 2.0
**Language**: Python (3.10-3.13) | **Size**: ~130MB
**Last push**: 2026-02-20 (same day as analysis -- actively maintained)
**Created**: 2023-08-16 | **Contributors**: 5+ core (Vasilije1990 primary)
**Tests**: 49+ test files with integration tests across multiple backends
**Description**: "Knowledge Engine for AI Agent Memory in 6 lines of code"

**Problem addressed**: Converting raw data (documents, conversations, code) into persistent, searchable knowledge graphs for AI agents. Positioned as a replacement for naive RAG by adding graph structure on top of vector search.

**Core approach**: LLM-based knowledge graph extraction (cognify pipeline) -> graph + vector dual storage -> triplet-based retrieval with vector-guided graph traversal. Modular pipeline architecture with pluggable backends.

**Maturity**: **Substantially implemented and functional**. This is a real, working system with multiple storage backends, a genuine pipeline framework, real tests across Neo4j/Kuzu/PGVector/ChromaDB/LanceDB, and an active development pace. Unlike Neuroca (scaffolding) or Mengram (limited implementation), cognee has working end-to-end pipelines. The memify system is more nascent than cognify but has functional components. The codebase shows organic growth patterns (TODO comments, backward compatibility shims, adapter-specific workarounds) rather than AI-generated boilerplate.

---

## Architecture / Method

### cognify Pipeline

The `cognify()` function orchestrates a 5-stage pipeline:

1. **Document Classification**: LLM-based classification into content types (TextContent, AudioContent, etc.) with detailed subclasses (Articles, Research Papers, Source Code, etc.). This is over-engineered -- ~200 lines of content type enums -- but functional.

2. **Text Chunking**: Configurable chunking via `TextChunker` (paragraph-based, default) or `LangchainChunker` (recursive character splitting with overlap). Chunk size auto-calculated from LLM context window: `min(embedding_max_tokens, llm_max_tokens // 2)`.

3. **Knowledge Graph Extraction**: The core step. Uses `extract_content_graph()` which sends each chunk to the LLM with a system prompt and expects structured output matching the `KnowledgeGraph` Pydantic model:
   ```python
   class Node(BaseModel):
       id: str
       name: str
       type: str
       description: str

   class Edge(BaseModel):
       source_node_id: str
       target_node_id: str
       relationship_name: str

   class KnowledgeGraph(BaseModel):
       nodes: List[Node]
       edges: List[Edge]
   ```
   Extraction uses `LLMGateway.acreate_structured_output()` -- LLM-agnostic structured generation. Chunks are processed in parallel via `asyncio.gather()`. Invalid edges (referencing non-existent node IDs) are filtered post-extraction.

4. **Summarization**: `summarize_text` generates summaries of chunks (stored as `TextSummary` data points).

5. **Data Point Storage**: `add_data_points()` extracts the graph model into nodes and edges, deduplicates them, stores in both graph DB and vector index. Optionally creates **triplet embeddings** -- concatenated `"source_text -> relationship_text -> target_text"` strings embedded as vectors. This is a notable feature.

**Custom graph models**: Users can define domain-specific Pydantic models instead of the default `KnowledgeGraph`. The LLM will extract according to whatever schema is provided. This is genuinely flexible.

**Ontology resolution**: Extracted entities are validated against an optional OWL ontology via `RDFLibOntologyResolver` with fuzzy matching. Ontology nodes/edges are merged into the graph with `ontology_valid=True` flags. This allows domain-specific vocabulary enforcement.

**Temporal cognify**: A separate pipeline (`temporal_cognify=True`) that extracts events with timestamps rather than generic entities. Uses `extract_events_and_timestamps()` to pull time-structured data from text, then `extract_knowledge_graph_from_events()` to build temporal graphs.

**Comparison to Zep**: Zep watches conversations in real-time, extracting facts as they flow. Cognee is batch-oriented -- you add data, then run cognify. Zep's temporal knowledge graph tracks when facts were learned and when they changed; cognee's temporal mode extracts events *about* time rather than tracking its own temporal evolution. Zep is conversational-first; cognee is document-first.

**Comparison to A-Mem**: A-Mem creates enriched atomic notes with LLM-judged links (Zettelkasten approach). Cognee creates a traditional knowledge graph with entity nodes and relationship edges. A-Mem's linking is bidirectional and self-assessed; cognee's relationships come from LLM extraction of each chunk independently, then deduplicated. A-Mem is more "bottom-up emergent"; cognee is more "top-down extraction."

### memify Mechanism

`memify()` is an **enrichment pipeline** that operates on an already-built knowledge graph. It does NOT do self-improvement or consolidation in the way our /sleep pipeline or Neuroca's "dreaming" would.

**Default behavior** (when called with no arguments): Extracts document chunks from the existing graph subgraph, then runs `add_rule_associations()` -- which uses an LLM to extract coding rules/best practices from the text and link them to the graph. This is a very specific use case (coding agent rules), not general-purpose memory consolidation.

**Extensible framework**: memify accepts custom `extraction_tasks` and `enrichment_tasks`, making it a general pipeline for graph enrichment. The available built-in tasks are:

- `extract_subgraph_chunks` -- extracts DocumentChunk text from graph nodes
- `add_rule_associations` -- LLM extracts coding rules, stores as `Rule` nodes linked to origin chunks
- `extract_usage_frequency` -- counts how often graph elements are used in search interactions (via `CogneeUserInteraction` nodes created when `save_interaction=True`), writes `frequency_weight` properties back to graph nodes
- `cognify_session` -- takes cached Q&A sessions and re-cognifies them into the knowledge graph
- `extract_user_sessions` -- pulls Q&A pairs from cache for session-based enrichment
- `persist_sessions_in_knowledge_graph` -- stores sessions persistently
- `create_triplet_embeddings` -- embeds triplet text for retrieval

**Session cognification**: The most interesting consolidation feature. When `save_interaction=True` during search, queries and answers are stored as `CogneeUserInteraction` nodes with `used_graph_element_to_answer` edges linking to the graph elements used. Later, `cognify_session` can re-process these interactions through the full cognify pipeline, effectively learning from its own usage patterns. This is a form of self-improvement, though rudimentary.

**Usage frequency tracking**: `extract_usage_frequency()` tallies interaction references within a time window and writes `frequency_weight` properties back to graph nodes. This enables frequency-based ranking -- nodes that are frequently used in answers get boosted. The implementation is long (~350 lines) but handles multiple timestamp formats and graph backends.

**Comparison to our /sleep pipeline**: Our planned consolidation involves synthesizing summaries, detecting evolution of understanding, and building layers of abstraction. Cognee's memify is more of a "graph enrichment toolkit" than a consolidation system. It doesn't synthesize, abstract, or detect patterns across memories. The usage frequency tracking is the closest thing to what we'd call "consolidation" but it's just counting, not synthesizing.

**Comparison to Neuroca's dreaming**: Neuroca proposed simulated annealing, memory abstraction, and cross-tier consolidation (though largely unimplemented). Cognee's memify is less ambitious but more functional -- it actually works and processes real data.

### Post-v0.5.2 Developments (reviewed 2026-03-27)

**Triplet embeddings promoted to default** (v0.5.5): `create_triplet_embeddings` is now a first-class documented pipeline and replaced coding rules as the default memify behavior. Implementation: `get_triplet_datapoints` yields text representations of `source → relationship → target`, `index_data_points` writes them to a `Triplet_text` vector collection. Queried via `SearchType.TRIPLET_COMPLETION`. Notable limitation: standalone collection, not fused into `GRAPH_COMPLETION` — our proposed RRF integration would already be more sophisticated.

**Entity consolidation pipeline** (v0.5.3): Post-cognify enrichment that rewrites `Entity` node descriptions using graph neighborhood context. Three steps: load entities with edges + neighbors, LLM-generate consolidated descriptions, write back in-place. Addresses real fragmentation problem where per-chunk extraction produces partial entity descriptions. LLM-heavy (one call per entity).

**Skills system** (v0.5.4rc1): Self-improving agent workflow framework — ingest SKILL.md files, execute, observe with LLM-based scoring (not binary), then `amendify` to propose improvements. Closed-loop learning for agent behaviors. Interesting conceptually but orthogonal to memory retrieval — closer to agent orchestration.

### Retrieval

Cognee has **14 search types** implemented via a retriever pattern:

1. **GRAPH_COMPLETION** (default, recommended): The flagship retrieval mode. Pipeline:
   - Embed query vector
   - Search multiple vector collections (Entity_name, TextSummary_text, EntityType_name, DocumentChunk_text, EdgeType_relationship_name)
   - Build/project in-memory `CogneeGraph` from graph DB (optionally filtered to relevant node IDs)
   - Map vector distances onto graph nodes and edges
   - Calculate **triplet importance scores** using combined vector distance + graph structure
   - Resolve top-k triplets to text
   - Generate LLM completion with graph context

2. **RAG_COMPLETION**: Traditional RAG -- vector search for chunks, LLM completion without graph structure.

3. **CHUNKS**: Pure vector similarity search, no LLM.

4. **SUMMARIES**: Returns pre-computed summaries.

5. **GRAPH_COMPLETION_COT**: Chain-of-thought variant -- generates initial answer, validates it, generates follow-up questions, retrieves more triplets, iterates up to `max_iter` rounds.

6. **GRAPH_COMPLETION_CONTEXT_EXTENSION**: Iteratively extends context by retrieving additional triplets based on previous completions.

7. **GRAPH_SUMMARY_COMPLETION**: Graph completion + summarization of results.

8. **TRIPLET_COMPLETION**: Direct triplet-based retrieval.

9. **CYPHER**: Raw Cypher queries against the graph database.

10. **NATURAL_LANGUAGE**: Translates natural language to graph queries.

11. **TEMPORAL**: Time-aware retrieval -- extracts time references from query, filters by temporal relevance.

12. **CODING_RULES**: Retrieves coding rules from the memify-enriched graph.

13. **CHUNKS_LEXICAL**: Jaccard similarity-based lexical search (token-based, stopword-aware).

14. **FEELING_LUCKY**: LLM selects the best search type automatically.

**The core algorithm** (brute_force_triplet_search):

The name is honest -- it's brute force. The system:
1. Embeds the query and searches all vector collections (wide search, default top-100)
2. Extracts relevant node IDs from vector results
3. Projects the graph from DB (optionally filtered to those IDs)
4. Maps vector distances onto graph nodes and edges
5. Calculates "triplet importance" -- a scoring function that combines vector similarity of the two nodes and the edge, with a `triplet_distance_penalty` (default 3.5) for missing vector matches

This is NOT RRF (Reciprocal Rank Fusion). It's a custom scoring where vector distances are mapped onto graph structure, and the graph provides the structural relationships while vectors provide the semantic relevance. The graph isn't traversed in the traditional BFS/DFS sense -- it's projected entirely into memory and scored holistically.

**Comparison to our planned RRF fusion**: Our priority #1 is combining BM25 + vector + graph traversal with RRF. Cognee's approach is different -- it uses vector search to seed relevance, then scores graph triplets. It lacks BM25/lexical integration in the main pipeline (CHUNKS_LEXICAL exists but is a separate search type, not fused). The approaches are complementary -- cognee's triplet scoring is an interesting alternative to RRF for graph-aware retrieval.

**Comparison to Hindsight's multi-path retrieval**: Hindsight proposes RRF over multiple retrieval paths. Cognee's approach is more tightly coupled -- vector and graph are scored together rather than independently fused. This gives cognee the advantage of structural awareness but the disadvantage of being less modular.

### Data Model

**Storage backends (graph)**:
- Neo4j (full support, Cypher queries)
- Kuzu (embedded graph DB, good for local use)
- AWS Neptune (cloud graph DB)
- NetworkX (in-memory, for development)

**Storage backends (vector)**:
- PGVector (PostgreSQL extension)
- ChromaDB
- LanceDB
- Custom vector engine abstraction

**Storage backends (relational)**: SQLAlchemy-based for metadata, datasets, users, permissions.

**Storage backends (cache)**: For session/conversation caching.

**Core data points** (Pydantic models extending `DataPoint`):
- `Entity` -- extracted entities with name, description, type
- `EntityType` -- type categorization nodes
- `DocumentChunk` -- text chunks with embeddings
- `TextSummary` -- generated summaries
- `Rule` -- coding rules (from memify)
- `CogneeUserInteraction` -- stored Q&A interactions
- `CogneeUserFeedback` -- user feedback with sentiment
- `NodeSet` -- grouping mechanism for nodes
- `Triplet` -- embedded triplet text (source -> relationship -> target)

**Graph structure**: Nodes store embeddings for indexed fields. Edges have `relationship_name`, `edge_text`, `ontology_valid`, and `feedback_weight` properties. The `feedback_weight` is updated by user feedback (sentiment analysis of feedback text, scored -5 to +5).

**Multi-tenancy**: Full user/dataset/permission model. Backend access control with per-dataset isolation. This is production-oriented.

---

## Key Claims vs. Reality

| Claim | Reality |
|-------|---------|
| "Knowledge Engine for AI Agent Memory in 6 lines of code" | The API is genuinely clean: `add()`, `cognify()`, `memify()`, `search()`. But production use requires backend configuration. |
| "Self-improvement capabilities" | memify's usage frequency tracking and session cognification are real but limited. No true self-improvement (abstraction, evolution detection, pattern synthesis). |
| "Graph and vector-based search" | Implemented and working. The triplet importance scoring is a genuine hybrid approach. |
| "30+ data sources" | Plausible given the document classification system, but the core path is text-based. Audio/video/3D handling would require external preprocessing. |
| "MCP integration" | Real MCP server exists in `cognee-mcp/` with full tool definitions for cognify, search, memify. Functional. |
| "Pythonic pipelines" | The pipeline framework (`Task`, `run_pipeline`) is real and extensible. Custom tasks work. |
| "LLM-agnostic" | `LLMGateway` abstraction supports multiple providers. Gemini requires different data models (no empty dicts). |

**Overall**: Claims are largely backed by code. The gap between marketing and implementation is much smaller than Neuroca or Mengram. The main overstatement is "self-improvement" -- memify is enrichment, not genuine self-improvement.

---

## Relevance to claude-memory

### What cognee Does That We Don't

1. **Knowledge graph extraction from text**: We store memories as atomic units with embeddings. We don't extract entity-relationship graphs from content. This is a fundamentally different approach -- cognee decomposes text into structured knowledge, we preserve text as-is with metadata.

2. **Triplet embeddings**: Embedding the concatenated text of `source -> relationship -> target` and using those embeddings for retrieval. This is clever -- it makes graph relationships directly searchable via vector similarity.

3. **Ontology-constrained extraction**: Validating extracted entities against OWL ontologies with fuzzy matching. We have no ontology system.

4. **Multi-backend graph storage**: Neo4j, Kuzu, Neptune support for persistent graph storage. We use SQLite for everything.

5. **Usage frequency tracking**: Tracking which graph elements are used in answers and writing frequency weights back. We track access counts but don't use them for retrieval scoring.

6. **Chain-of-thought retrieval**: The GRAPH_COMPLETION_COT mode that iteratively validates and refines answers. Our retrieval is single-pass.

7. **Session cognification**: Re-processing Q&A sessions through the extraction pipeline to learn from usage patterns.

8. **User feedback integration**: Sentiment analysis of feedback, writing `feedback_weight` to graph edges. We have no explicit feedback mechanism.

### What We Already Do Better

1. **Memory types and priorities**: Our procedural/episodic/semantic typing with priority levels is more nuanced for personal assistant memory than cognee's generic entity extraction.

2. **Temporal decay and consolidation design**: Our decay model (half-life based, access-count boosted) is purpose-built for personal memory. Cognee has no decay -- everything persists equally.

3. **FTS5 + vector fusion (planned)**: BM25 via FTS5 is better for exact-term matching than cognee's Jaccard-based lexical search.

4. **Lightweight architecture**: SQLite + sqlite-vec is orders of magnitude simpler to deploy than Neo4j + PGVector + PostgreSQL. For a personal memory system (single user, moderate scale), our architecture is appropriate.

5. **Source tracking and deduplication**: Our 0.9-similarity deduplication at write time is more efficient than cognee's approach of storing everything and deduplicating at the graph level.

6. **Consolidation design**: Our planned /sleep pipeline (synthesis, evolution detection, layered abstraction) is more sophisticated in concept than cognee's usage frequency counting, even though neither is fully implemented yet.

---

## Worth Stealing (ranked)

### 1. Triplet Embedding Concept (High value, Medium effort)
**What**: Embed `"entity_A -> relationship -> entity_B"` as a single vector. Search these triplet vectors alongside individual memory vectors.
**Why**: This would allow our system to find relational context that pure memory search misses. Example: searching for "chess opening preferences" could match a triplet embedding of "Alexis -> prefers -> Sicilian Defense" even if neither memory alone mentions preferences.
**How it maps**: We could create a separate `triplets` table with composite text embeddings. During retrieval, search both `memories` and `triplets` tables, fuse results. This fits naturally into our planned RRF fusion as an additional retrieval path.
**Effort**: Moderate -- requires a new table, a background process to generate triplet embeddings from related memories, and integration into retrieval.

### 2. Neighborhood-Aware Entity Consolidation (Medium-High value, Medium effort)
**What**: After extraction, rewrite entity/node descriptions using their full graph neighborhood (edges + neighbors) rather than just the originating chunk.
**Why**: Our sleep pipeline's entity work produces per-memory descriptions. An entity mentioned across 20 memories over 3 months should have a richer description than any single `remember()` call produced. Cognee's `consolidate_entity_descriptions_pipeline` does this in batch; for us it would be incremental during sleep — identify entities whose neighborhood has grown since last consolidation, rewrite only those.
**How it maps**: During sleep consolidation, query entities with new edges since last pass, fetch neighborhood, LLM-consolidate descriptions. The temporal dimension matters — descriptions should reflect how understanding evolved, not just flatten everything. Lower LLM cost than Cognee's batch approach since we'd only touch changed entities.
**Effort**: Moderate — requires tracking "last consolidated" per entity and a sleep step to rewrite descriptions. The LLM call pattern is straightforward.

### 3. Graph-Aware Retrieval Scoring (Medium value, Medium effort)
**What**: Instead of just vector similarity, score results by considering the graph neighborhood. Cognee maps vector distances onto graph structure and scores triplets holistically.
**Why**: Our memories have implicit relationships (shared themes, temporal proximity, corrections of each other). Scoring could consider these connections.
**How it maps**: Even without a full graph DB, we could build a lightweight relationship index. When a memory is retrieved, check if related memories (by theme, by `source`, by correction chains) also score well, and boost accordingly.
**Effort**: Requires defining relationship types and building a scoring function. Could start simple.

### 4. Usage Frequency as Retrieval Signal (Low value for us, Low effort)
**What**: Track which memories are actually used in responses, boost frequently-used ones.
**Why**: We already track `access_count` but don't use it in retrieval scoring. Cognee's `frequency_weight` approach could be adapted.
**How it maps**: We could add `access_count` as a signal in our retrieval scoring, either as a direct boost or through RRF.
**Effort**: Minimal -- we already have the data.

### 5. Session Cognification Pattern (Medium value, High effort)
**What**: Re-process successful Q&A interactions through the extraction pipeline to extract new knowledge.
**Why**: When our system successfully answers a question using context from multiple memories, the synthesized understanding could itself become a memory.
**How it maps**: After a successful recall, the synthesized context could be stored as a new "derived" memory with links to its source memories. This is related to our planned consolidation pipeline.
**Effort**: High -- requires deciding what counts as "worth remembering" from a session, avoiding circular amplification.

### 6. Iterative Retrieval (CoT/Context Extension) (Medium value, Medium effort)
**What**: After initial retrieval and generation, validate the answer, generate follow-up queries, retrieve more context, iterate.
**Why**: Single-pass retrieval misses context that becomes relevant only after initial analysis.
**How it maps**: Could be implemented at the `recall()` level -- if initial results seem insufficient, automatically expand the search.
**Effort**: Moderate -- the retrieval infrastructure exists, need to add the iteration loop and stopping criteria.

---

## Not Useful For Us

1. **Multi-backend graph databases**: We don't need Neo4j/Kuzu/Neptune. SQLite is sufficient for our scale and use case. Adding graph DB complexity would be over-engineering.

2. **LLM-based entity extraction pipeline**: Our memories are already structured (text + metadata + embeddings). We don't need to decompose them into entity-relationship graphs. Our unit of memory is the enriched text chunk, not the knowledge graph triplet.

3. **Document classification system**: The 200+ lines of content type enums (TextSubclass, AudioSubclass, etc.) are irrelevant for conversation-based memory.

4. **Multi-tenancy/RBAC**: We're single-user. The dataset permission system is unnecessary complexity.

5. **MCP server approach**: We already have an MCP server. Cognee's MCP implementation is specific to their API surface.

6. **Ontology resolution**: We don't have a domain ontology and don't need one for personal assistant memory.

---

## Impact on Implementation Priority

No changes to the 14-item priority list. The most relevant ideas (triplet embeddings, usage-frequency scoring, iterative retrieval) all map to existing priorities:

- **Triplet embeddings** -> supports Priority #1 (RRF fusion) by adding another retrieval path
- **Usage frequency** -> minor enhancement to Priority #1 retrieval scoring
- **Iterative retrieval** -> could become a sub-item under Priority #1 or a future enhancement
- **Session cognification** -> relates to Priority #3 (/sleep consolidation) but is a different approach

The main takeaway is that cognee's triplet embedding concept is worth prototyping as part of the RRF work. It's a concrete, implementable idea that could meaningfully improve retrieval quality.

---

## Connections

### To Prior Analyses

**vs. Zep** (conversation-first, real-time fact extraction):
Cognee is document-first and batch-oriented. Zep watches conversation flow and builds a temporal knowledge graph in real-time. Cognee requires explicit `cognify()` calls. For our use case (personal assistant memory), Zep's real-time approach is more relevant, but cognee's retrieval innovations (triplet scoring, CoT retrieval) are more sophisticated.

**vs. GraphRAG** (Microsoft, hierarchical community summaries):
Both use LLM-based knowledge graph extraction. GraphRAG creates hierarchical community structures via Leiden clustering; cognee creates flat entity-relationship graphs with optional ontology constraints. GraphRAG is more focused on summarization quality; cognee is more focused on retrieval flexibility. Cognee's 14 search types vs. GraphRAG's global/local search is a notable breadth advantage.

**vs. Mem0** (memory layer for AI):
Mem0 focuses on user/agent/session memory with fact extraction from conversations. Cognee focuses on document-to-knowledge-graph transformation. Mem0 is closer to our use case; cognee is more general-purpose.

**vs. Neuroca** (bio-inspired scaffolding):
Night and day in maturity. Cognee has ~49 test files, multiple working backends, active development with 5+ core contributors, 12K+ stars. Neuroca had ~110K lines of AI-generated scaffolding with skipped tests. Cognee's memify is less ambitious than Neuroca's "dreaming" but actually works.

**vs. Mengram** (personal memory):
Mengram had good ideas but limited implementation. Cognee has far more working code. However, Mengram's personal memory design (conversation-native, personal context) is more relevant to our use case than cognee's document-processing orientation.

**vs. A-Mem** (Zettelkasten for LLMs):
Different philosophies. A-Mem creates enriched atomic notes with emergent connections (bottom-up). Cognee extracts structured knowledge graphs (top-down). A-Mem's self-assessed linking is more organic; cognee's LLM extraction is more systematic. For personal memory, A-Mem's approach may produce more natural connections.

**vs. Hindsight** (multi-path retrieval with RRF):
Cognee's approach to dual retrieval is different from Hindsight's RRF. Hindsight independently retrieves via multiple paths and fuses with RRF. Cognee tightly couples vector and graph scoring. Both are valid; Hindsight's approach is more modular and easier to extend. Cognee's triplet importance scoring is a unique contribution not found in Hindsight.

**vs. Dynamic Cheatsheet** (iterative context refinement):
Cognee's GRAPH_COMPLETION_COT and context extension retrievers share the iterative refinement philosophy. Dynamic Cheatsheet refines a scratchpad; cognee's CoT retriever refines graph context. Both validate and iterate, but Dynamic Cheatsheet is more focused on task context while cognee is more focused on knowledge retrieval.

---

## Summary Assessment

**Overall value**: **Medium-High for ideas, Low for direct adoption.**

Cognee is the most mature and functional open-source knowledge graph memory system we've analyzed. It's real software with real users, unlike the scaffolding projects (Neuroca, Mengram) or the research papers without code. The codebase shows evidence of actual production use -- backward compatibility shims, telemetry, access control, multiple backend adapters.

**What matters for us**:
1. The **triplet embedding** concept is the standout contribution -- a concrete, novel idea that none of our other 22 sources propose. It deserves prototyping.
2. The **graph-aware retrieval scoring** approach (mapping vector distances onto graph structure) is an interesting alternative to pure RRF.
3. The **iterative retrieval** patterns (CoT, context extension) validate ideas we've considered.
4. The **session cognification** pattern is a real implementation of learning-from-usage.

**What doesn't matter for us**: The heavy infrastructure (Neo4j, multi-tenancy), the document-processing orientation, the LLM-based entity extraction pipeline. Our architecture is deliberately lightweight and conversation-native, and that's the right choice.

**Maturity verdict**: Genuinely implemented, actively maintained, production-capable. Rating: 7/10 for completeness, 6/10 for relevance to our specific project.
