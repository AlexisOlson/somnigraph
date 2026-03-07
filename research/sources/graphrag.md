# microsoft/graphrag Repository Analysis

*Generated 2026-02-19 by Opus agent reading local clone at ~/.claude/repos/graphrag*

## Repository Overview

GraphRAG is Microsoft's production-grade knowledge graph construction pipeline. It transforms unstructured text into a structured knowledge graph via LLM-driven entity/relationship extraction, then runs hierarchical community detection (Leiden algorithm) on the resulting graph. Each community gets an LLM-generated summary report. At query time, these community reports power a map-reduce search strategy that can answer questions requiring global understanding of the entire corpus -- something standard vector RAG cannot do.

The repo is a monorepo with 8 packages (`graphrag`, `graphrag-cache`, `graphrag-chunking`, `graphrag-common`, `graphrag-input`, `graphrag-llm`, `graphrag-storage`, `graphrag-vectors`). The core logic lives in `packages/graphrag/graphrag/`.

## File Structure (Key Paths)

```
packages/graphrag/graphrag/
  prompts/
    index/
      extract_graph.py          # Entity/relationship extraction prompt
      summarize_descriptions.py # Merging duplicate entity descriptions
      community_report.py       # Community report generation prompt (graph mode)
      community_report_text_units.py  # Community report (text unit mode)
      extract_claims.py         # Claim/covariate extraction
    query/
      global_search_map_system_prompt.py   # Map step: score key points per community
      global_search_reduce_system_prompt.py # Reduce step: synthesize analyst reports
      drift_search_system_prompt.py        # DRIFT: iterative graph traversal search
      local_search_system_prompt.py        # Local: entity-neighborhood search
  index/
    operations/
      extract_graph/             # LLM-based entity+relationship extraction
      summarize_descriptions/    # Merge multiple descriptions of same entity
      summarize_communities/     # Generate community reports (the core summarization)
      cluster_graph.py           # Hierarchical Leiden community detection
      prune_graph.py             # Graph pruning (degree, frequency, edge weight)
      build_noun_graph/          # NLP-based extraction (fast mode, no LLM)
    workflows/
      factory.py                 # Pipeline definitions (Standard, Fast, Update variants)
      extract_graph.py           # Extract -> summarize descriptions workflow
      create_communities.py      # Leiden clustering on entity graph
      create_community_reports.py  # LLM summarization of each community
    update/                      # Incremental indexing (merge old + delta)
  query/
    structured_search/
      global_search/             # Map-reduce over community reports
      local_search/              # Entity neighborhood + text units
      drift_search/              # Iterative community-guided exploration
      basic_search/              # Simple baseline
  graphs/
    hierarchical_leiden.py       # Wraps graspologic_native for Leiden
  data_model/
    entity.py, relationship.py, community.py, community_report.py, text_unit.py
  prompt_tune/                   # Auto-generate domain-specific prompts
```

## Architecture Overview

The system has two main phases:

### Indexing Pipeline (Standard Mode)
1. **Chunk documents** into text units (~300-600 tokens)
2. **Extract entities & relationships** via LLM (with "gleaning" -- multi-pass extraction where the model is asked if it missed anything)
3. **Summarize descriptions** -- when the same entity appears across multiple chunks, merge all its descriptions into one via LLM
4. **Build graph** from entities (nodes) and relationships (edges), with edge weights from the LLM's relationship_strength score
5. **Cluster** using hierarchical Leiden algorithm (via `graspologic_native`), producing a tree of communities at multiple resolution levels
6. **Generate community reports** -- for each community at each level, LLM produces a structured JSON report (title, summary, importance rating, findings)
7. **Embed** text units and community reports for vector search

### Query Phase
- **Global Search**: Map-reduce over community reports. Map step asks each report batch to produce scored key points. Reduce step synthesizes the top-scoring points into a coherent answer. This is the unique capability -- answering "what are the main themes across all documents?"
- **Local Search**: Starts from relevant entities (found via embedding similarity), expands to their neighborhoods, pulls in community reports and source text.
- **DRIFT Search**: Starts with a "primer" that reads top community summaries, generates follow-up queries, iteratively explores the graph via local search steps, then reduces all intermediate answers.

## Unique Concepts

### 1. Community Reports as an Abstraction Layer
The central innovation. Rather than searching raw chunks, GraphRAG pre-computes summaries of densely-connected entity clusters. These reports become the primary retrieval unit for global queries. Each report has a structured format: title, executive summary, importance rating (0-10), rating explanation, and 5-10 detailed findings with data provenance citations.

### 2. Hierarchical Community Structure with Mixed Context
Communities form a tree. When summarizing a higher-level community, if its local context (entities + edges) is too large for the LLM context window, the system substitutes sub-community reports for the largest sub-communities, keeping raw entity data for the rest. This `build_mixed_context` algorithm greedily replaces the biggest sub-communities first until the context fits.

### 3. Gleaning (Multi-Pass Extraction)
After initial entity extraction, the system sends a CONTINUE_PROMPT: "MANY entities and relationships were missed in the last extraction." Then optionally sends a LOOP_PROMPT asking "Y/N are there more?" This iterative refinement catches entities the model initially overlooked.

### 4. Description Summarization with Token Budgeting
When merging descriptions of the same entity from multiple chunks, the `SummarizeExtractor` tracks token usage. If descriptions exceed the input token budget, it summarizes in batches -- summarizing what fits, then including that summary as a "description" for the next batch. This is a rolling-window summarization pattern.

### 5. Importance Scoring at Multiple Levels
- Entity extraction assigns `relationship_strength` (numeric, set by LLM)
- Community reports get `rating` (0-10 float, LLM-assigned importance)
- Global search map step assigns `score` (0-100 integer, relevance to query)
- These cascade: high-importance communities surface first in the reduce step

### 6. Data Provenance / Grounding Rules
Every claim in a community report must cite specific record IDs: `[Data: Entities (5, 7); Relationships (23)]`. This creates a traceable chain from summary back to source data. The prompt caps citations at 5 IDs per reference with "+more" notation.

### 7. DRIFT Search (Dynamic Reasoning with Iterative Follow-up Trees)
A query exploration strategy that starts from community summaries, generates follow-up queries, performs local searches on each, scores intermediate answers, and iterates. The primer provides an intermediate answer and follow-up queries; each action step refines the search. This is essentially a search tree traversal guided by LLM-generated exploration directions.

## Summarization Prompts -- Detailed Analysis

This is the most relevant section for claude-memory's /sleep design.

### Community Report Prompt (`community_report.py`)
The core summarization prompt. Key design choices:

**Input format**: Entities and relationships are provided as CSV tables with `human_readable_id, title, description` columns. The LLM sees structured tabular data, not raw text.

**Output structure** (JSON):
```json
{
  "title": "short specific name with key entities",
  "summary": "executive summary of structure and relationships",
  "rating": 5.0,
  "rating_explanation": "one sentence",
  "findings": [
    {"summary": "short insight title", "explanation": "multi-paragraph grounded text"}
  ]
}
```

**Key prompt techniques**:
- Requests 5-10 findings (not open-ended, not fixed)
- Each finding has both summary and explanation (two-level detail)
- Grounding rules force citation of specific record IDs
- Caps citations at 5 per reference to prevent list-padding
- Word limit via `{max_report_length}` parameter
- Includes a full worked example with realistic entity names

### Description Summarization Prompt (`summarize_descriptions.py`)
Much simpler. Given an entity name and a list of descriptions from different chunks:
- Concatenate into a single comprehensive description
- **Resolve contradictions** (explicit instruction)
- Write in third person, include entity names for context
- Capped at `{max_length}` words

The contradiction resolution instruction is notable -- it explicitly asks the LLM to handle conflicting information rather than ignoring it or presenting both versions.

### Map-Reduce Query Prompts
The **map step** produces scored key points (JSON array of `{description, score}`). The **reduce step** synthesizes these, with the critical instruction that "analysts' reports are ranked in descending order of importance" -- so the LLM knows the top items matter most.

## Worth Stealing (ranked)

### 1. Hierarchical Summarization with Sub-report Substitution (Priority: HIGH)
**Directly applicable to /sleep.** When consolidating memories at a higher level, if the raw detail is too large, substitute lower-level summaries for the densest clusters. This is exactly the detail -> summary -> gestalt layering we want. The `build_mixed_context` algorithm provides a concrete implementation: greedily replace the largest sub-clusters with their summaries until the context fits the token budget.

**Adaptation**: Instead of communities of entities, we have clusters of related memories. When generating a gestalt-level summary, include detail-level memories for sparse/recent clusters and summary-level digests for dense/older clusters.

### 2. Structured Community Report Format (Priority: HIGH)
The `{title, summary, rating, findings[]}` format is excellent for memory consolidation outputs. Each "finding" is a summary + explanation pair -- this maps directly to a memory layer structure where the summary is the quick-access version and the explanation preserves the grounding detail.

**Adaptation**: For sleep consolidation, each cluster report could have:
- `theme`: short name for the memory cluster
- `summary`: executive summary
- `importance`: 0-10 rating (maps to our priority system)
- `insights`: list of `{summary, detail, source_memory_ids[]}`
- `contradictions`: any conflicts detected (they already ask LLMs to resolve these!)

### 3. Grounding Rules / Data Provenance (Priority: HIGH)
Forcing the LLM to cite specific record IDs prevents hallucination in summaries and enables traceability. For memory consolidation, this means every claim in a gestalt summary should reference the original memory IDs it was derived from, enabling drill-down and verification.

### 4. Contradiction Resolution in Summarization (Priority: HIGH)
The description summarization prompt explicitly says: "If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary." This is a simple but powerful instruction for our graded contradiction detection. We could extend this with our planned approach of detecting and preserving contradictions with confidence scores rather than silently resolving them.

### 5. Multi-Pass Extraction (Gleaning) (Priority: MEDIUM)
The gleaning pattern -- extract, then tell the model it missed things, then ask if there's more -- could be useful for relationship extraction from memories. When building our memory graph edges, a second pass asking "what connections did you miss?" could improve recall.

### 6. Importance Scoring Cascade (Priority: MEDIUM)
Multiple levels of importance scoring (relationship strength -> community importance -> query relevance) creates a natural ranking hierarchy. For memory: memory priority -> cluster importance -> query relevance. This aligns with our existing priority system but adds the cluster-level rating as a middle tier.

### 7. Rolling-Window Summarization (Priority: MEDIUM)
The `SummarizeExtractor`'s approach to handling descriptions that exceed the token budget -- summarize a batch, include that summary in the next batch's input, repeat -- is a practical pattern for consolidating large numbers of memories about the same entity/topic.

### 8. Context Sorting by Degree (Priority: LOW)
When building context for community reports, edges are sorted by combined degree (descending). Higher-degree edges connect more important nodes. For memory context building, sorting by connection count or access frequency could similarly prioritize the most central memories.

## Not Worth It

- **The full knowledge graph construction pipeline**: Way too heavyweight for personal memory. We don't need entity extraction from documents, graph databases, or vector indexing of community reports.
- **Leiden community detection**: Requires a dense entity-relationship graph with numeric edge weights. Our memory graph is sparser and the relationship types are more semantic. Simpler clustering (embedding-based or tag-based) is more appropriate.
- **CSV-as-context format**: GraphRAG dumps entities and relationships as CSV tables into prompts. This works for structured extraction output but is unnecessarily rigid for memory consolidation where narrative context matters more.
- **Map-reduce query strategy**: The parallel map over community chunks + reduce pattern is designed for corpus-scale search. Our memory system operates at a scale where a single LLM call can handle the relevant context.
- **The claim/covariate extraction system**: Designed for compliance and investigative analysis. Overkill for personal memory.
- **Prompt tuning system**: Auto-generates domain-specific prompts (persona, role, entity types) from sample data. Interesting engineering but not relevant to our fixed-domain personal memory use case.

## Key Weakness

**Consolidation is one-shot, not incremental.** Community reports are generated once during indexing and are essentially static until re-indexed. The incremental update system (`index/update/`) simply merges new communities alongside old ones -- it doesn't re-summarize existing communities when new information arrives. Old and new community IDs are concatenated with an offset, not reconciled.

This means GraphRAG cannot handle the core challenge of personal memory: information about the same topic arriving over time and needing to be progressively integrated. Their "update" is append-only for communities. If new information contradicts or refines an existing community, the old community report persists unchanged alongside a new one.

A secondary weakness: the importance ratings are assigned by the LLM with no calibration mechanism. There's no feedback loop, no decay, and no way for ratings to adjust based on subsequent information. The 0-10 scale is absolute, not relative to the corpus.

## Relevance to claude-memory

**Overall relevance: MEDIUM.** GraphRAG solves a different problem (static knowledge base construction from documents) but its consolidation techniques transfer well.

**Most valuable takeaways for /sleep design:**

1. **The structured report format** should inform our consolidation output schema. The `{title, summary, importance, findings[{summary, explanation, sources}]}` pattern is a proven way to produce multi-granularity summaries that maintain grounding.

2. **Hierarchical context substitution** is the key algorithmic insight. When generating a higher-level summary and the raw material is too large, replace the densest sub-clusters with their existing summaries. This is exactly the mechanism needed for detail -> summary -> gestalt layering.

3. **Explicit contradiction resolution in prompts** validates our plan for graded contradiction detection. GraphRAG asks the LLM to resolve contradictions silently; we can extend this to detect, classify, and preserve them with confidence scores.

4. **Data provenance via record ID citation** should be mandatory in our consolidation prompts. Every gestalt-level insight should cite the memory IDs it derives from.

5. **The rolling-window summarization pattern** from `SummarizeExtractor` is directly usable when a topic has accumulated more memories than fit in a single LLM context window.

**What we should NOT adopt:** The graph-theoretic community detection (Leiden), the entity extraction pipeline, the map-reduce query architecture, or the static one-shot indexing model. Our memory system needs to be incremental, lightweight, and continuously updated -- not batch-processed.
