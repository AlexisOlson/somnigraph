# Graphs RAG at Scale: Beyond Retrieval-Augmented Generation With Labeled Property Graphs and Resource Description Framework for Complex and Unknown Search Spaces -- Analysis

*Generated 2026-03-24 by Opus 4.6 agent reading arXiv:2603.22340*

---

## Paper Overview

**Citation**: Manie Tadayon, Mayank Gupta (Capital Group). "Graphs RAG at Scale: Beyond Retrieval-Augmented Generation with Labeled Property Graphs and Resource Description Framework for Complex and Unknown Search Spaces." arXiv preprint arXiv:2603.22340, March 25, 2026. 17 pages. No public code repository.

**Problem addressed**: Traditional embedding-based RAG struggles with semi-structured data (JSON key-value pairs, nested hierarchies) where the search space is unknown and the number of documents to retrieve cannot be pre-specified. The paper argues that graph-based representations -- both RDF triplets and Labeled Property Graphs -- handle these cases better by enabling structured traversal rather than relying on embedding similarity + top-k selection.

**Core claim**: Graph RAG (both RDF and LPG variants) significantly outperforms traditional embedding-based RAG for complex, semi-structured data retrieval, particularly for search/listing and comparison query intents. The LPG approach with text-to-Cypher translation achieves the best results overall.

**Scale of evaluation**: 200 questions across 4 intent types (Search, Compare, Detail, Other), evaluated on a proprietary dataset of 1,104 investment fund records from Capital Group (mutual funds, ETFs, PPS products). The dataset produces ~650,000 RDF triplets, 14,000+ nodes, 9,000+ relationships. Metrics are accuracy and completeness scored as 1 (correct+complete), 0.5 (correct but incomplete), or 0 (incorrect). No standard benchmarks (no MuSiQue, HotpotQA, LoCoMo, etc.).

---

## Architecture / Method

### Core Insight

The paper's central observation is that semi-structured data (JSON records with nested fields) can be represented either as RDF triplets or as an LPG with typed relationships, and that graph query languages (SPARQL, Cypher) provide exact retrieval for structured queries that embedding similarity cannot match. This is strongest for "search/listing" queries like "List all growth funds" where the answer requires filtering on a known attribute across all entities -- a task where embedding-based retrieval with fixed top-k is structurally inadequate.

### Three Approaches Compared

1. **RAG_1** (JSON-to-text, then standard RAG): Convert each JSON record to a narrative text via LLM, chunk, embed, retrieve. Excluded from main evaluation due to scalability problems (4 days for 1,104 records), hallucination risk on numeric data, and incomplete information in generated narratives.

2. **RAG_2** (RDF-to-text agentic RAG): Convert JSON to RDF triplets deterministically, then convert triplets to natural language sentences, chunk, embed with BGE-m3 (1024-dim), retrieve with reranking (BGE-reranker-v2-m3, bge-reranker-v2-gemma, ms-marco-MiniLM-L6-v2). This is the "agentic RAG" baseline -- standard vector retrieval over structured-data-derived text.

3. **RAG_RDF** (Graph RAG via SPARQL): Store RDF triplets in Amazon Neptune. At query time: (a) LLM extracts mentioned fund names/types from query, (b) deterministic mapping to graph nodes via metadata, (c) supervised classification + embedding similarity to select relevant relationships (~50 via embedding, ~100 via LLM), (d) SPARQL traversal to fetch matching triplets. Also replicated with NetworkX + FAISS.

4. **RAG_LPG** (Graph RAG via Cypher): Store data in Neptune as LPG with 79 node labels, 57 relationship types, ~34,500 nodes. Schema designed for multi-hop traversal (shared nodes for common attributes like ProductType, FundType). At query time: LLM generates Cypher query from natural language, augmented with full graph schema + node/relationship metadata. Existing fine-tuned text-to-Cypher models (Neo4j/Gemma-3-27B) evaluated but found inadequate for the dataset's complexity.

### Design Choices Worth Noting

**Predicate clustering** (Section 3.8): The 8,000+ unique RDF relationships are clustered by semantic similarity to reduce the relationship selection search space. Two approaches: unsupervised (K-means/hierarchical on embeddings) and supervised (DeBERTa-v3 fine-tuned on GPT-4.1-labeled categories). The supervised approach is used in production.

**Predicate-to-natural-language** (Section 3.9): RDF predicates with dot-notation hierarchies (e.g., `facts.companiesIssuers`) are converted to natural language ("Facts About Companies Issuers") using a fine-tuned BART Large model. This creates the text that RAG_2 embeds and retrieves over.

**Query intent classification** (Section 3.10): Queries are categorized as Compare, Detail, Search/Listing, or Other. This guides the response generation pipeline (not the retrieval step), providing intent-specific response templates.

**LPG schema as prompt context**: The text-to-Cypher agent receives the full graph schema (all node labels, relationship types, properties, edge directions) plus natural-language descriptions of each element. This is the key enabler -- without schema awareness, Cypher generation fails.

### What the Paper Acknowledges as Limitations

- LPG's text-to-Cypher translation is error-prone: "The primary source of error arises from confusion or mistakes in converting natural language queries to Cypher" (Section 5).
- Existing fine-tuned text-to-Cypher models (Neo4j/Gemma-3-27B) were inadequate for their complex schema.
- The approach requires "careful schema design" -- the schema must be crafted to support the expected multi-hop traversals.

### What the Paper Misses

- **No ablations.** No decomposition of which components contribute what. Is it the graph structure or the schema metadata that helps? Is predicate clustering necessary? What if you just gave the LLM the full JSON and a schema description?
- **No latency or cost analysis.** Each RDF query involves multiple LLM calls (node selection, relationship selection via both embedding and LLM). The LPG path requires LLM-generated Cypher for every query. No timing data.
- **No standard benchmarks.** The evaluation is entirely on proprietary financial data. No comparison with published Graph RAG systems (GraphRAG, LightRAG, HippoRAG) on standard datasets.
- **No error analysis beyond anecdote.** The paper mentions embedding confusion between similar fund names (CGCP vs CGCB) but doesn't quantify error types systematically.
- **The "unknown search space" framing is misleading.** The search space is completely known -- it's a structured database of 1,104 fund records with a fixed schema. What's "unknown" is how many results a query will return, which is a standard database query problem, not a retrieval challenge.

---

## Key Claims & Evidence

### Overall Performance (Table 1)

| Method | Score (out of 200) | Accuracy |
|--------|-------------------|----------|
| RAG_2 (Agentic RAG) | 116.0 | 58.0% |
| RAG_RDF | 172.5 | 86.3% |
| RAG_LPG | 185.5 | 92.8% |

### Performance by Intent (Table 2)

| Method | Intent | Correct | Partial | Incorrect |
|--------|--------|---------|---------|-----------|
| RAG_2 | Search | 23 | 31 | 45 |
| RAG_2 | Compare | 32 | 6 | 7 |
| RAG_2 | Detail | 33 | 3 | 9 |
| RAG_2 | Other | 6 | 4 | 0 |
| RAG_RDF | Search | 70 | 20 | 10 |
| RAG_RDF | Compare | 37 | 8 | 0 |
| RAG_RDF | Detail | 42 | 3 | 0 |
| RAG_RDF | Other | 6 | 4 | 0 |
| RAG_LPG | Search | 91 | 4 | 5 |
| RAG_LPG | Compare | 39 | 4 | 2 |
| RAG_LPG | Detail | 43 | 1 | 1 |
| RAG_LPG | Other | 8 | 0 | 2 |

### Methodological Strengths

- **Honest about RAG_1 failure.** The paper doesn't hide that the naive JSON-to-text approach was impractical and excluded it rather than cherry-picking favorable results.
- **Practical engineering detail.** The predicate clustering, natural language conversion, and schema-as-prompt approaches are well-described enough to reproduce.
- **Real-world data.** Using actual financial product data (even if proprietary) is more convincing than synthetic datasets for demonstrating semi-structured data challenges.

### Methodological Weaknesses

- **200 questions, no statistical significance.** With 200 questions total (100 search, 45 compare, 45 detail, 10 other), per-intent sample sizes are small. No confidence intervals or significance tests reported.
- **No baseline calibration.** The RAG_2 baseline uses BGE-m3 embeddings with generic rerankers. A stronger baseline (e.g., domain-specific fine-tuned embeddings, metadata-augmented retrieval, SQL over a relational schema) would be more informative.
- **Scoring is coarse.** Three-valued scoring (0, 0.5, 1) conflates many failure modes. A query that returns 95% of the correct funds scores the same as one returning 50%.
- **LLM-as-judge without inter-rater reliability.** Completeness assessment requires "either a subject matter expert or a well fine-tuned LLM" (Section 4.4), but no details on judge reliability, agreement rates, or potential biases.
- **The strongest result is the least surprising.** LPG + Cypher achieving 91/100 on search queries is effectively demonstrating that SQL-like querying beats embedding search for database lookups. This is known.

---

## Relevance to Somnigraph

### What This Paper Does That We Don't

**Typed relationship querying.** Somnigraph's edges in `edges` table have `linking_context` and flags (`contradiction`, `revision`, `derivation`), but retrieval doesn't query edges by type or traverse specific relationship patterns. The PPR in `scoring.py` treats all (non-contradiction) edges equally during graph propagation. The paper's RDF approach -- selecting specific relationship types relevant to a query before traversal -- is a capability Somnigraph lacks. However, the relevance is limited: Somnigraph has ~1,500 edges with free-text linking context, not 9,000 typed relationships with a fixed schema.

**Schema-aware query translation.** The LPG pipeline translates natural language to a structured query language (Cypher), using the full graph schema as context. Somnigraph has no equivalent -- all retrieval goes through vector similarity + BM25 keyword matching in `fts.py` and the sqlite-vec channel, fused by RRF. There's no mechanism to express "find all memories linked by derivation edges" as a structured query.

**Relationship classification/clustering.** The paper's predicate clustering (Section 3.8) pre-classifies 8,000+ relationship types into categories to narrow the search space. Somnigraph's themes serve a loosely analogous role (grouping memories by topic), but edge relationships are untyped beyond the three flags.

### What We Already Do Better

**Learned retrieval scoring.** The paper uses off-the-shelf rerankers (BGE-reranker, ms-marco-MiniLM) for RAG_2 and no reranking for the graph approaches. Somnigraph's 26-feature LightGBM reranker in `reranker.py`, trained on 1,032 real-data queries with explicit feedback signal, is a fundamentally different approach -- it learns from actual usage rather than relying on generic cross-encoder similarity.

**Feedback loop.** No feedback mechanism whatsoever in this paper. Retrieval quality is evaluated post-hoc but never feeds back into the system. Somnigraph's EWMA feedback with empirical Bayes Beta prior (r=0.70 GT correlation) is the primary architectural differentiator here.

**Dynamic memory.** The paper operates on a static dataset of 1,104 fund records. No mechanism for adding, updating, or removing knowledge. Somnigraph handles continuous memory creation, decay, consolidation, and archiving.

**Multi-signal fusion.** Somnigraph combines vector similarity, BM25 keyword search, PPR graph signals, feedback signals, temporal features, and metadata features through RRF fusion and learned reranking. The paper's approaches are single-channel: either embedding similarity (RAG_2), graph traversal (RDF), or query translation (LPG).

**Graph evolution.** Somnigraph's edges gain weight through co-retrieval feedback (Hebbian reinforcement) and are created during sleep consolidation (NREM edge detection in `sleep_nrem.py`). The paper's graph is static -- constructed once from the source data and never modified.

---

## Worth Stealing (ranked)

### 1. Relationship-type-aware PPR seeding

**What**: Use edge type information (contradiction, revision, derivation, or linking context clusters) to weight or filter edges during PPR traversal, rather than treating all edges uniformly.

**Why it matters**: Currently `scoring.py` excludes contradiction edges from PPR but treats all other edges equally. A query about "how X changed over time" should preferentially traverse revision edges; a query about "what led to X" should favor derivation edges. This is a lightweight version of the paper's relationship selection without requiring a full query language.

**Implementation**: In `scoring.py` where PPR is computed, add edge-type-conditional weights. For example, if query intent suggests temporal reasoning (detected by keyword heuristic or reranker feature), boost revision-edge weights in the PPR transition matrix. The `edge_type` flags already exist in the edges table.

**Effort**: Low-Medium. The edge type data exists; the change is to the PPR transition matrix construction. The hard part is determining when to apply which weighting, which could be a new reranker feature rather than a hard rule.

### 2. Schema-as-context for structured recall

**What**: When a recall query has clear structural intent (e.g., "all memories about chess" or "what did I decide about X"), expose the memory schema (categories, theme vocabulary, edge types) as additional context to help the calling Claude instance formulate better queries.

**Why it matters**: The paper's strongest result is LPG + Cypher, where giving the LLM full schema awareness enabled precise structured queries. Somnigraph's `recall()` takes a free-text query + optional context, with no schema awareness. The calling Claude instance doesn't know what categories, themes, or relationship types exist, so it can't formulate queries that would exploit the structure.

**Implementation**: Add an optional `schema_hint` to `recall()` response metadata, or expose a lightweight `describe_schema()` tool that returns category distribution, top-N themes, edge type counts. This is informational, not a query mechanism -- it helps the caller write better natural-language queries.

**Effort**: Low. A simple SQL aggregation query over the memories and edges tables, exposed as a new MCP tool or appended to `startup_load` output.

### 3. Predicate clustering as edge context enrichment

**What**: During NREM edge detection (`sleep_nrem.py`), classify discovered edges into semantic categories (temporal, causal, topical, contradictory) rather than just flagging them as links.

**Why it matters**: The paper's relationship classification enables relationship-specific retrieval. Somnigraph edges have free-text `linking_context` but no structured type beyond the three flags. Richer edge typing would enable Worth Stealing #1 and would provide new reranker features (e.g., "does the query's intent match the edge type of the traversal path?").

**Implementation**: Add a `relationship_category` column to the edges table. During NREM, use the existing LLM call that generates `linking_context` to also classify the relationship into a small taxonomy (temporal, causal, elaboration, contradiction, revision). Fine-tuning not needed -- the LLM call already exists, just needs an additional structured output field.

**Effort**: Medium. Schema migration, NREM prompt update, and downstream consumers (PPR, reranker) need to be aware of the new field. The taxonomy itself requires design thought.

---

## Not Useful For Us

### The entire LPG/Cypher pipeline

The paper's strongest result (LPG + text-to-Cypher) depends on a fixed, well-defined schema of a structured database with 79 node labels and 57 relationship types. Somnigraph's memories are semi-structured text with emergent relationships -- there's no fixed schema to generate Cypher against. The paper's own finding that existing fine-tuned text-to-Cypher models failed on their dataset (a structured one!) underscores how brittle this approach is. For a personal memory system with ~730 heterogeneous memories, the overhead of maintaining a formal graph schema and query language would vastly exceed any retrieval benefit.

### RDF triplet representation

Converting Somnigraph memories to subject-predicate-object triplets would destroy the narrative context that makes memories useful. A memory like "Decided to use EWMA for feedback aggregation because raw averages are too noisy" is not well-represented as `(feedback_aggregation, uses, EWMA)` -- the reasoning (the "because") is the valuable part. RDF triplets are suited for structured data with known attributes, not for episodic or reflective memories.

### Amazon Neptune / graph database infrastructure

Somnigraph runs on SQLite -- deliberately, for single-user simplicity and portability. Moving to a graph database would add deployment complexity, external dependencies, and operational overhead for a system serving one Claude instance. The paper's graph traversal benefits come from schema-aware structured queries, not from the graph database engine itself. SQLite's recursive CTEs can handle the traversal patterns Somnigraph needs at its scale (~730 memories, ~1,500 edges).

### Predicate-to-natural-language conversion (BART fine-tuning)

The paper fine-tunes BART to convert dot-notation predicates to English. Somnigraph's memories are already in natural language. This solves a problem we don't have.

### Cross-encoder rerankers for structured data

The paper evaluates BGE-reranker and ms-marco-MiniLM cross-encoders. These are generic models with no domain adaptation. Somnigraph's learned LightGBM reranker with 26 features (including feedback, graph signals, and temporal features) is already more capable and domain-adapted than any off-the-shelf cross-encoder would be for personal memory retrieval.

---

## Impact on Implementation Priority

**Minimal impact on current priorities.** The paper operates in a fundamentally different regime (structured database querying over semi-structured data) from Somnigraph's regime (personal memory retrieval over narrative text). The architectural lessons are limited.

**Slight reinforcement for edge typing during NREM** (related to Roadmap Tier 2 open problems on edge accumulation and pruning). If edges had richer type information, pruning could be type-aware (e.g., prune stale temporal edges more aggressively than causal ones). This connects to the "False bridge ratchet" problem identified by external reviewers -- typed edges would enable type-specific pruning policies.

**No impact on the multi-hop vocabulary gap problem.** The paper's solution to multi-hop (graph traversal over typed relationships) requires a fixed schema with explicit entity-to-entity relationships. Somnigraph's multi-hop failures (88% zero content-word overlap, per the failure analysis) are about vocabulary mismatch between queries and evidence, not about missing graph structure. The 22 R@10 misses identified in `docs/multihop-failure-analysis.md` would not be helped by Cypher queries -- the memories aren't connected by typed relationships that a query language could traverse.

**No impact on LoCoMo benchmark work (P4).** The paper doesn't evaluate on LoCoMo or any conversational memory benchmark. The expansion method ablation and sleep pass ablation remain the relevant next steps.

**No impact on reranker iteration (P2).** The 5 new features awaiting production retrain are already defined and LoCoMo-tested. Nothing in this paper suggests additional features worth extracting.

---

## Connections

### HippoRAG (PPR-based retrieval)

Both papers use graph structure for retrieval, but at opposite ends of the spectrum. HippoRAG builds an open knowledge graph from unstructured text and uses PPR for multi-hop reasoning -- closely analogous to what Somnigraph does. This paper starts from structured data and uses formal query languages (SPARQL, Cypher) for exact retrieval. HippoRAG's approach is far more relevant to Somnigraph because it handles the same type of challenge: retrieving over heterogeneous text where relationships are extracted, not predefined.

### Kumiho (graph-native memory, prospective indexing)

Kumiho uses Neo4j with a rich node/edge schema (Entity, Belief, Event, Relation with 10+ subtypes). This paper's LPG schema design (79 labels, 57 relationship types) faces the same challenge -- schema design determines retrieval quality. Kumiho's prospective indexing (pre-computing anticipated query patterns at write time) is a more sophisticated version of schema-aware retrieval than this paper's approach, and more applicable to Somnigraph's unstructured-memory setting.

### Mem0 (extract-then-update pipeline)

Mem0 extracts facts from conversations and stores them as flat or graph-enhanced memories. This paper's RDF extraction (JSON to triplets) is deterministic rather than LLM-based, avoiding the extraction errors that plague Mem0. But Mem0 operates on conversational text, which is Somnigraph's domain, while this paper operates on structured data -- making Mem0 the more relevant comparison.

### SPLADE (learned sparse representations)

The paper's RAG_2 baseline uses BGE-m3 for dense retrieval. SPLADE's learned sparse representations could be a stronger baseline for the structured-data-derived text, since SPLADE excels at exact term matching (fund names, ticker symbols) where dense embeddings fail. The paper doesn't consider sparse retrieval at all, which is a notable omission given their own discussion of embedding confusion between similar fund names.

---

## Summary Assessment

This is an industry application paper documenting Capital Group's Graph RAG system for querying investment fund data. The engineering is competent -- the RDF pipeline with predicate clustering and the LPG pipeline with schema-aware Cypher generation are practical solutions to a real business problem. The finding that graph-based approaches outperform embedding-based RAG for structured data queries is genuine, though unsurprising: querying a structured database via a query language will outperform approximate nearest-neighbor search for queries that are essentially database lookups.

The paper's value for Somnigraph is limited by a fundamental domain mismatch. Somnigraph manages ~730 heterogeneous narrative memories with emergent relationships, not 1,104 uniform structured records with a fixed schema. The paper's strongest techniques (text-to-Cypher, SPARQL traversal, predicate classification) all depend on having a well-defined, stable schema -- something personal memory systems inherently lack. The evaluation is narrow (200 questions, proprietary data, no standard benchmarks, no ablations) and the "unknown search space" framing overstates the novelty -- the core finding is that SQL-like querying beats vector search for database lookups.

The single most transferable idea is edge-type-aware graph traversal: using relationship categories to weight PPR propagation rather than treating all edges uniformly. This is a lightweight version of the paper's core insight (relationship-specific retrieval) that could work within Somnigraph's existing architecture. However, it requires richer edge typing than currently exists, making it a prerequisite rather than a direct implementation. The practical priority is low relative to Somnigraph's current agenda items (31-feature retrain, expansion ablation, sleep impact measurement), all of which address more pressing retrieval gaps.
