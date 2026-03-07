# Zep: A Temporal Knowledge Graph Architecture for Agent Memory -- Analysis

*Generated 2026-02-18 by Opus agent reading 2501.13956v1*

---

## Paper Overview

- **Authors**: Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, Daniel Chalef (all Zep AI)
- **Venue**: arXiv preprint, January 2025
- **Paper**: arXiv:2501.13956v1
- **Open-source component**: Graphiti (github.com/getzep/graphiti)

**Core problem**: Existing RAG frameworks treat documents as static corpora. Enterprise agents need to integrate continuously evolving data from conversations + structured business data while maintaining historical relationships. Current memory systems lack temporal awareness -- they don't distinguish when a fact was true from when it was stored, and can't represent how facts change over time.

**Key contribution**: A three-tier temporal knowledge graph (episodes -> semantic entities -> communities) with bi-temporal modeling that tracks both real-world validity periods and system ingestion order. Outperforms MemGPT on DMR (94.8% vs 93.4%) and achieves 18.5% accuracy improvement on LongMemEval with 90% latency reduction compared to full-context baselines.

---

## Architecture / Method

### The Knowledge Graph: G = (N, E, phi)

Three hierarchical subgraph tiers:

```
Community Subgraph (Gc)     [highest level]
  |-- community nodes (cluster summaries)
  |-- community edges -> entity members
  |
Semantic Entity Subgraph (Gs)    [middle level]
  |-- entity nodes (resolved named entities)
  |-- semantic edges (facts = relationships between entities)
  |
Episode Subgraph (Ge)       [lowest level]
  |-- episode nodes (raw messages/text/JSON)
  |-- episodic edges -> referenced semantic entities
```

This directly mirrors the episodic/semantic distinction from psychology (Wong & Gonzalez, 2018) and extends it with community-level summaries from GraphRAG (Edge et al., 2024).

**Key design principle**: Dual storage of raw episodic data AND derived semantic entities. Episodes are non-lossy -- you can always trace a semantic fact back to the raw message that produced it. This is a citation/provenance mechanism, not just storage.

### Episode Ingestion

Episodes are raw data units: messages (speaker + text), text blocks, or JSON. Each message carries a **reference timestamp** `t_ref` indicating when it was sent.

Context window for entity extraction: current message + last n=4 messages (two complete conversation turns).

### Entity Extraction and Resolution

1. **Extract** entities from current message using LLM (with last 4 messages as context)
2. **Reflect** on extraction using reflexion technique (Shinn et al., 2023) to reduce hallucinations and improve coverage
3. **Embed** each entity name into 1024-dim vector space
4. **Search** existing entities via both cosine similarity AND full-text search on names/summaries
5. **Resolve** candidates via LLM prompt -- if duplicate, merge with updated name and summary
6. **Write** to graph via predefined Cypher queries (NOT LLM-generated queries, to avoid schema drift)

The speaker is always automatically extracted as an entity.

### Fact Extraction and Resolution

Facts are semantic edges between entity pairs, containing:
- A key predicate (relation type, e.g., `LOVES`, `WORKS_FOR`)
- A detailed fact string with full contextual information
- Embeddings for retrieval

**Fact deduplication** constrains search to edges between the same entity pair. This is both a correctness measure (prevents merging facts about different entity pairs) and a performance optimization (limits search space).

**Hyper-edge implementation**: The same fact can be extracted multiple times between different entity pairs, enabling modeling of complex multi-entity relationships.

### Bi-Temporal Model (THE core innovation)

Each edge (fact) carries **four timestamps**:

| Timestamp | Timeline | Meaning |
|-----------|----------|---------|
| `t_valid` | T (chronological) | When the fact became true in the real world |
| `t_invalid` | T (chronological) | When the fact stopped being true in the real world |
| `t'_created` | T' (transactional) | When the fact was recorded in the system |
| `t'_expired` | T' (transactional) | When the fact was invalidated in the system |

Timeline T = chronological ordering of real-world events
Timeline T' = transactional ordering of system data ingestion

**Temporal extraction**: The system uses `t_ref` (message timestamp) to resolve relative dates in conversation. "I started my new job two weeks ago" with `t_ref = 2025-01-20` becomes `t_valid = 2025-01-06T00:00:00Z`. Both absolute ("June 23, 1912") and relative ("last summer") timestamps are extracted.

**Edge invalidation**: When a new edge contradicts an existing edge:
1. System finds semantically related existing edges (via embedding similarity)
2. LLM compares new edge against existing edges to identify contradictions
3. If temporally overlapping contradiction found: set `t_invalid` of old edge to `t_valid` of new edge
4. **New information always wins** on the T' timeline -- Graphiti consistently prioritizes newer data

This is the conflict resolution mechanism. It's not destructive -- the old edge remains in the graph with its full temporal metadata. The fact is *invalidated*, not *deleted*.

### Temporal Extraction Prompt (from Appendix)

Key rules from the actual prompt:
- ISO 8601 format: `YYYY-MM-DDTHH:MM:SS.SSSSSSZ`
- Reference timestamp used as "now" for relative dates
- Present-tense facts: use reference timestamp for `valid_at`
- Non-spanning relationships: set `valid_at` only (no `invalid_at`)
- Only use dates directly stated to establish or change the relationship -- no inference from related events
- Year-only mentions: January 1st at midnight
- Date-only mentions: midnight of that date
- Always include timezone offset (UTC if unspecified)

### Community Detection

Built on GraphRAG's approach but uses **label propagation** (Zhu & Ghahramani, 2002) instead of Leiden algorithm.

**Why label propagation**: Has a straightforward dynamic extension. When a new entity node is added:
1. Survey communities of neighboring nodes
2. Assign new node to the plurality community
3. Update community summary

This delays the need for full community refreshes, reducing latency and LLM inference costs. Periodic full refreshes are still necessary as communities gradually diverge.

Community nodes contain summaries generated through iterative map-reduce-style summarization (from GraphRAG). Community names contain key terms and subjects, embedded and stored for similarity search.

### Memory Retrieval Pipeline

```
f(alpha) = chi(rho(phi(alpha))) = beta

where:
  alpha = text query
  phi   = search (returns candidate edges, entity nodes, community nodes)
  rho   = reranker (reorders results)
  chi   = constructor (formats nodes/edges into text context)
  beta  = context string for LLM
```

#### Search (phi): Three methods

1. **Cosine similarity search** (`phi_cos`): Vector similarity via Neo4j's Lucene implementation
2. **Okapi BM25 full-text search** (`phi_bm25`): Keyword matching via Lucene
3. **Breadth-first search** (`phi_bfs`): n-hop graph traversal from seed nodes

Search targets vary by object type:
- Semantic edges: search `fact` field
- Entity nodes: search `entity name`
- Community nodes: search `community name` (keywords + phrases)

**BFS is distinctive**: Recent episodes can seed the BFS, incorporating recently mentioned entities and relationships. This addresses contextual similarity -- nodes closer in the graph appear in more similar conversational contexts. This is a third dimension beyond word similarity (BM25) and semantic similarity (vector).

#### Rerankers (rho): Five options

1. **Reciprocal Rank Fusion (RRF)** -- merge multiple search result lists
2. **Maximal Marginal Relevance (MMR)** -- diversity-aware reranking
3. **Episode-mentions reranker** -- prioritize by mention frequency in conversation (frequently referenced = more accessible)
4. **Node distance reranker** -- reorder by graph distance from centroid node (locality)
5. **Cross-encoder** -- LLM-based relevance scoring (highest cost, highest precision)

#### Constructor (chi): Output format

For each semantic edge: `FACT (Date range: t_valid - t_invalid)`
For each entity node: `ENTITY_NAME: entity summary`
For each community node: `community summary`

### Models Used

- **Embeddings + reranking**: BGE-m3 (BAAI), 1024-dim
- **Graph construction**: gpt-4o-mini-2024-07-18
- **Response generation**: gpt-4o-mini or gpt-4o
- **Evaluation judge**: GPT-4o

---

## Key Claims & Evidence

### Claim 1: Zep outperforms MemGPT on DMR

| System | Model | Score |
|--------|-------|-------|
| Recursive Summarization | gpt-4-turbo | 35.3% |
| Conversation Summaries | gpt-4-turbo | 78.6% |
| MemGPT | gpt-4-turbo | 93.4% |
| Full-conversation | gpt-4-turbo | 94.4% |
| **Zep** | **gpt-4-turbo** | **94.8%** |
| Full-conversation | gpt-4o-mini | 98.0% |
| **Zep** | **gpt-4o-mini** | **98.2%** |

**Assessment**: The DMR win is marginal (94.8% vs 93.4% for gpt-4-turbo, 98.2% vs 98.0% for gpt-4o-mini). The authors themselves note this benchmark is limited: only 60 messages per conversation (fits in context window), single-turn fact-retrieval questions, ambiguous phrasing. The full-conversation baseline nearly matches Zep, which undercuts the memory system's value proposition on this benchmark.

### Claim 2: Substantial LongMemEval improvements

| System | Model | Score | Latency | Context Tokens |
|--------|-------|-------|---------|----------------|
| Full-context | gpt-4o-mini | 55.4% | 31.3s | 115k |
| **Zep** | **gpt-4o-mini** | **63.8%** | **3.20s** | **1.6k** |
| Full-context | gpt-4o | 60.2% | 28.9s | 115k |
| **Zep** | **gpt-4o** | **71.2%** | **2.58s** | **1.6k** |

**Assessment**: This is the stronger result. 115k tokens compressed to 1.6k tokens with accuracy *improvements*. The 90% latency reduction is a direct consequence of smaller context. Most impressive on complex question types:

- **single-session-preference**: +184% improvement (gpt-4o)
- **temporal-reasoning**: +48.2% (gpt-4o-mini), +38.4% (gpt-4o)
- **multi-session**: +16.7% (gpt-4o-mini), +30.7% (gpt-4o)

Notable weakness: **single-session-assistant** dropped 17.7% (gpt-4o). The authors acknowledge this needs further work.

### Claim 3: Bi-temporal modeling enables temporal reasoning

The temporal-reasoning category shows the biggest gains after single-session-preference. This directly validates the bi-temporal approach -- by storing `t_valid` and `t_invalid`, the system can answer "when did X happen?" and "what was true at time Y?" questions that flat memory systems struggle with.

### Limitations Acknowledged

1. DMR benchmark is too easy -- full-context baselines nearly match Zep
2. No MemGPT comparison on LongMemEval (couldn't get it to work)
3. single-session-assistant questions degrade with Zep
4. No evaluation of structured business data integration (a core selling point)
5. Community subgraph retrieval not exercised in experiments
6. BFS and advanced rerankers not fully tested
7. No formal ontology -- purely LLM-extracted schema

### Additional Limitations (My Assessment)

- **All authors are Zep AI employees** -- commercial product announcement dressed as a paper
- **No ablation studies** -- which components contribute how much? Is bi-temporal worth the complexity? Are communities needed?
- **Hindsight paper (arxiv 2512.12818) achieves dramatically better LongMemEval scores** (83.6-91.4% vs Zep's 63.8-71.2%) -- Zep's results are good but not SOTA
- **No evaluation of temporal query types** beyond what LongMemEval provides -- custom temporal reasoning benchmarks would strengthen the bi-temporal claim
- **Label propagation for communities is acknowledged as an approximation** that degrades over time

---

## Relevance to claude-memory

### What Directly Applies

**1. Bi-temporal modeling maps directly to our mutation log + memory lifecycle.**

Our current schema tracks only `created_at` and `last_accessed`. Zep adds the crucial distinction: when was this *true* vs when was this *stored*? For our system:

| Zep Timestamp | Our Equivalent | Status |
|---------------|---------------|--------|
| `t'_created` | `created_at` | Already have |
| `t'_expired` | `is_deleted` + `superseded_by` | Partial -- we track deletion but not the temporal invalidation event |
| `t_valid` | **Missing** | When the fact became true in the real world |
| `t_invalid` | **Missing** | When the fact stopped being true |

Adding `valid_from` and `valid_until` (nullable timestamptz) to our `memories` table would give us Zep's chronological timeline. Our existing `created_at` and soft-delete already provide the transactional timeline.

**2. Edge invalidation maps to our graded contradiction detection (priority #3).**

Zep's process: find semantically related edges -> LLM compares for contradiction -> if contradictory, set `t_invalid` on old edge. This is simpler than Engram's 5-type graded model but has one advantage: **temporal invalidation preserves the old fact with its validity period rather than just marking it as contradicted.** A fact can be *true for a period* and then superseded, which is different from being wrong.

Recommendation: merge Engram's grading with Zep's temporal invalidation. When we detect a "temporal evolution" contradiction, instead of just creating an `evolved_from` edge, also set `valid_until` on the old memory. This gives us queryable temporal ranges.

**3. The three-tier search (vector + BM25 + BFS) validates our RRF fusion priority.**

Zep independently arrived at the same hybrid search approach as Hindsight (which we already prioritized at #1). The BFS component is novel for us and adds a dimension we hadn't considered: **graph-neighborhood retrieval** -- once we have `memory_edges`, we can do n-hop traversal from seed memories to find contextually related facts that don't share keywords or semantic similarity.

**4. Fact extraction as atomic subject-predicate-object triples.**

Our memories are free-text blobs with metadata. Zep's facts are structured: `source_entity -> RELATION_TYPE -> target_entity` with a detailed fact string. We don't need to go full knowledge graph, but the extraction prompt pattern (entities first, then facts between entities) is instructive for improving how we structure `remember()` content. Specifically: encouraging callers to include the *entities* involved, not just a narrative description.

**5. Entity resolution at write time.**

When a new memory mentions "Alexis," the system should recognize this refers to the same entity as "the user" or "Alexis Olson." Zep does this via embedding similarity + full-text search on entity names/summaries, then LLM confirmation. We could implement a lightweight version: maintain an entity alias table or resolve entity references in the content before embedding.

### What Validates Our Existing Decisions

1. **Non-lossy storage**: Zep keeps raw episodes alongside extracted semantics. Our approach of storing full `content` + `summary` is aligned. Detail must remain accessible.

2. **LLM-based contradiction detection**: Zep uses LLM to compare new edges against existing ones for contradictions. This validates our planned approach in graded contradiction detection (#3).

3. **Predefined queries over LLM-generated queries**: Zep uses fixed Cypher queries for graph writes, not LLM-generated ones. This is the "don't let the LLM write your SQL" principle. Our direct Supabase RPC calls are already aligned.

4. **Multiple reranking strategies**: RRF, MMR, frequency-based, distance-based, cross-encoder. Validates that reranking is a rich design space worth investing in after basic RRF fusion.

5. **Community/cluster summaries**: Zep's community subgraph is analogous to our planned gestalt layer in `/sleep`. Both are higher-level summaries derived from entity/fact clusters.

### What Challenges Our Existing Decisions

1. **We don't have graph structure for retrieval.** BFS over a knowledge graph is a retrieval channel we can't access. Our planned `memory_edges` table could enable this, but only if we build traversal queries on top of it.

2. **Our contradiction detection is planned as a sleep-time operation.** Zep does it at *ingestion* time (when new edges arrive). This is faster feedback -- contradictions are caught immediately rather than waiting for the next `/sleep` cycle. Consider: lightweight contradiction check on `remember()` (as already planned for #3) with deeper analysis deferred to sleep.

3. **We don't extract temporal validity from content.** When a user stores "Alexis moved to Austin" we record `created_at = now()` but don't extract that the move happened at a specific time. Zep's temporal extraction prompt could be adapted for our `remember()` pipeline.

---

## Worth Stealing (ranked)

### 1. Bi-temporal `valid_from` / `valid_until` fields -- LOW effort, HIGH value

Add two nullable timestamptz columns to `memories`. Populate them when temporal information is available in the content. During contradiction resolution, set `valid_until` on superseded memories.

**Why**: Enables temporal queries ("what was true in January?"), gives contradiction resolution a richer vocabulary than binary supersede/keep, and is the foundation for trajectory detection. Low effort because it's just two columns -- the extraction logic can be added incrementally.

**Effort**: Schema change is trivial. Temporal extraction prompt can be borrowed almost verbatim from Zep's appendix. Integration into `remember()` is medium effort (LLM call to extract dates from content).

### 2. Graph traversal (BFS) as retrieval channel -- MEDIUM effort, HIGH value

Once `memory_edges` exist (priority #2), add a BFS/n-hop query: given seed memories from initial retrieval, walk edges to find connected memories. Merge results into RRF.

**Why**: Catches contextually related memories that share no keywords or semantic similarity. Example: a debugging insight connected via `relates` edge to a chess pattern -- neither keyword nor vector search would surface this, but graph traversal would.

**Effort**: Requires `memory_edges` (#2) to exist first. The SQL query itself is straightforward (recursive CTE for n-hop). Integration into recall pipeline is medium.

### 3. Temporal invalidation (not just supersede) -- LOW effort, MEDIUM value

When contradiction detection identifies a temporal evolution, set `valid_until` on the old memory rather than (or in addition to) creating a `supersedes` edge. The old fact becomes "true from X to Y" rather than just "superseded."

**Why**: Preserves the temporal validity range, making the memory system a faithful timeline of facts rather than a latest-version-only store. Essential for trajectory building.

**Effort**: Low. It's a field update during the existing contradiction detection flow.

### 4. Entity resolution on `remember()` -- MEDIUM effort, MEDIUM value

When storing a new memory, check if referenced entities match existing entities in the memory store. Maintain a lightweight entity index (name + aliases + embedding). Resolve "the user" / "Alexis" / "Alexis Olson" to the same entity.

**Why**: Prevents fragmentation where memories about the same entity are scattered under different names. Improves retrieval for entity-centric queries. Enables future entity-based views ("everything about project X").

**Effort**: Need entity extraction from content (LLM call or regex for common patterns), entity table/index, resolution logic. Medium complexity, but high payoff for retrieval quality.

### 5. Episode-mentions reranker -- LOW effort, LOW-MEDIUM value

Prioritize memories that are mentioned or accessed more frequently. We already track `access_count` -- using it as a reranking signal (not just for decay) would be trivial.

**Why**: Frequently referenced information is likely important in the current context. Simple signal, easy to add to RRF.

**Effort**: Trivial. Include `access_count` as a feature in the reranking formula.

### 6. Temporal extraction prompt (from appendix) -- LOW effort, MEDIUM value

Adapt Zep's temporal extraction prompt for our `remember()` pipeline. When storing a memory, optionally extract `valid_from` / `valid_until` from the content using the prompt pattern.

**Why**: Makes bi-temporal fields (#1) useful without requiring the caller to manually specify dates. "Started learning chess in December" auto-populates `valid_from = 2025-12-01T00:00:00Z`.

**Effort**: Low. The prompt is provided verbatim in the paper's appendix. Need one additional LLM call during `remember()`.

### 7. Community/cluster summaries -- HIGH effort, MEDIUM value (deferred)

Build cluster-level summaries from groups of related memories, analogous to Zep's community subgraph. Our `/sleep` skill's gestalt layer is already planned to do this.

**Why**: Provides high-level domain understanding for broad queries. Enables the "what do I know about X in general?" query type.

**Effort**: High. This is essentially the full `/sleep` pipeline (Phase 2 layer generation). Already planned. Zep's approach (label propagation + map-reduce summarization) provides a concrete algorithm to follow.

---

## Not Useful For Us

### Full knowledge graph (Neo4j + Cypher)

Zep uses Neo4j as its graph database with Cypher queries. This is massive infrastructure overhead (Neo4j server, Cypher query language, graph-specific indexing). Our Postgres + pgvector + `memory_edges` table approach gives us the essential graph capabilities (edges, traversal) without the operational cost. A relational database with an edges table is sufficient for our scale (hundreds to low thousands of memories, not millions).

### Hyper-edges for multi-entity facts

Zep models complex multi-entity relationships by duplicating the same fact across multiple entity pairs. This is a knowledge-graph-specific pattern that addresses graph-representation limitations. Our free-text memory content naturally captures multi-entity facts without this workaround.

### Label propagation for community detection

Label propagation is an algorithm for partitioning a large graph into clusters. At our scale, we can cluster memories by theme/category overlap + vector similarity during `/sleep` without a formal community detection algorithm. Our existing `themes` array + category structure provides a simpler clustering signal.

### Episode subgraph (raw message store)

Zep stores raw messages as episode nodes because its input is conversation streams. Our system's input is already curated -- `remember()` calls contain processed, intentional content. We don't need a raw message archive layer. Our `content` field IS the processed output of the equivalent of Zep's extraction pipeline, done by the agent/user rather than automated.

### Cross-encoder reranking

Highest-quality reranking but also highest cost (LLM inference per query-document pair). At our scale and latency requirements (MCP tool call during a Claude Code session), the latency cost is not justified. RRF + access_count reranking should be sufficient.

---

## Impact on Implementation Priority

### Current priority list (from index.md):

1. RRF fusion
2. Relationship edges
3. Graded contradiction detection
4. Decay floor + power-law
5. Sleep skill
6. Mutation log
7. Confidence scores
8. Reference index
9. Multi-angle indexing

### Proposed changes after this paper:

**No reordering needed.** Zep validates the top 3 priorities rather than disrupting them. Specific adjustments:

**Add to #2 (Relationship edges)**: Plan for BFS/n-hop traversal queries from day one. When designing the `memory_edges` table, ensure it supports efficient graph traversal (indexes on both `source_id` and `target_id`, recursive CTE query template). Don't just store edges -- make them retrievable by traversal.

**Add to #3 (Graded contradiction detection)**: Incorporate temporal invalidation. When contradiction detection classifies a tension as "temporal evolution," set `valid_until` on the old memory in addition to creating the `evolved_from` edge. This requires the bi-temporal fields to exist.

**New sub-item under #3**: Add `valid_from` and `valid_until` (nullable timestamptz) to the `memories` table. This is a prerequisite for temporal invalidation and trajectory detection. Schema change is trivial; extraction logic can be added incrementally.

**Add to #6 (Mutation log)**: Zep's bi-temporal model is essentially a structured mutation log on edges. Our planned append-only event log per memory should record temporal validity changes alongside access/reheat/priority events.

**New item (optional, after #9)-- #10: Temporal extraction on remember()**: Adapt Zep's temporal extraction prompt to auto-populate `valid_from` / `valid_until` when temporal information is present in memory content. Low priority because it can be done manually for now, but the prompt template from the appendix makes it easy to add later.

**New item (optional, after #10) -- #11: Lightweight entity resolution**: Maintain entity alias table, resolve references during `remember()`. Medium effort, medium payoff. Not urgent for single-user system with consistent entity naming, but becomes important if memory count grows into thousands.

---

## Connections

### To other analyses in this project

**[[agent-output-hindsight-paper]]**: Hindsight achieves dramatically better LongMemEval scores (83.6-91.4% vs Zep's 63.8-71.2%) with a different architecture (four epistemically-distinct networks vs three-tier knowledge graph). Both use hybrid retrieval (vector + BM25 + graph traversal) and RRF. Both validate that multi-channel retrieval is essential. Key difference: Hindsight separates by epistemic type (fact vs opinion vs experience vs observation), Zep separates by structural level (episode vs entity vs community). For claude-memory, our category system (episodic, semantic, procedural, reflection, meta) is closer to Hindsight's approach.

**[[agent-output-hindsight-paper]]** (bi-temporal reference): The Hindsight paper specifically flagged Zep's bi-temporal modeling as a feature worth studying. Having now read the Zep paper, the recommendation stands: `valid_from` / `valid_until` fields are the highest-value addition from this paper.

**[[agent-output-evomemory]]**: Evo-Memory flagged Zep as must-read alongside Hindsight. Evo-Memory's key concern (summaries sometimes underperforming raw retrieval) applies to Zep's community summaries too -- the paper doesn't evaluate community retrieval, so we can't know if those summaries help or hurt. This reinforces the caution from [[sleep-skill]]: summaries should be routing aids, not replacements for detail.

**[[agent-output-engram]]**: Engram's 5-type graded contradiction model is more nuanced than Zep's binary contradiction/not-contradiction. Recommendation: use Engram's grading taxonomy but add Zep's temporal invalidation (`valid_until`) as the mechanism for the "temporal evolution" grade.

**[[sleep-skill]]**: Zep's community detection and summarization parallels our `/sleep` Phase 2 (layer generation). Zep uses label propagation + map-reduce; we plan LLM-driven cluster summarization. The approaches are compatible. Zep's dynamic community extension (assign new entities to existing communities incrementally) is a useful optimization -- our sleep could similarly do lightweight incremental updates between full regeneration cycles.

**[[systems-analysis]]**: This paper was item #1 on the must-read follow-up list. Confirmed: the bi-temporal model and temporal extraction are the key contributions. The knowledge graph architecture is interesting but heavier than what we need. The production benchmarking (latency, token costs) is valuable validation that compressed retrieval outperforms full-context even for accuracy, not just cost.

**[[broader-landscape]]**: Zep was surveyed at the repo level. This paper adds: (a) the formal bi-temporal model, (b) the temporal extraction prompt, (c) LongMemEval results, (d) the three-search-method architecture. The repo-level assessment ("fact extraction and conflict resolution logic is worth studying") is confirmed, but the implementation priorities are clearer now.

### To prior phase findings

**Phase 6 new idea "Bi-temporal modeling"** (from [[systems-analysis]]): "Distinguish 'when fact was true' from 'when memory was stored.' We only track storage time." Now fully validated with implementation details. The Zep paper provides: the four-timestamp model, the temporal extraction prompt, and the edge invalidation mechanism. This is no longer just an idea -- it has a concrete implementation path.

**Phase 6 new idea "Temporal filtering as first-class retrieval channel"**: Zep's constructor includes `t_valid` and `t_invalid` in the context string for every fact. This means temporal filtering happens at the *output* stage (the LLM sees the date ranges and reasons about them) rather than the *retrieval* stage. For our system, this suggests: include temporal validity metadata in recall results, let the session model reason about temporal relevance.

---

## Summary Assessment

Zep is a solid production memory system paper with one genuinely novel contribution (bi-temporal modeling) and several well-executed conventional techniques (hybrid search, graph construction, community detection). Its benchmark results are good but not SOTA -- Hindsight significantly outperforms it on the same LongMemEval benchmark. The paper is also a product announcement (all authors are Zep AI employees), which explains the emphasis on production metrics (latency, scalability) over ablation studies.

**For claude-memory**: The bi-temporal model and temporal extraction prompt are the highest-value takeaways. The knowledge graph architecture is interesting but more infrastructure than we need. The validation of hybrid search (vector + BM25 + graph traversal) reinforces our existing priorities. The entity resolution and edge invalidation mechanisms provide useful implementation patterns for our graded contradiction detection and relationship edges work.

**Net impact**: Additive, not transformative. Adds `valid_from` / `valid_until` fields as a new sub-item, reinforces and adds detail to several existing priorities (#2, #3, #6), provides a concrete temporal extraction prompt we can adapt. Does not change the ordering of our implementation priority list.
