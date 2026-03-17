# Similar Systems

An opinionated comparison with memory systems we studied, built on, or explicitly decided against. Not comprehensive — focused on systems we learned something from. For the full research corpus (63 sources), see `research/sources/index.md`.

## Contents

1. [The landscape](#the-landscape)
2. [System profiles](#system-profiles)
3. [What we borrowed](#what-we-borrowed)
4. [Comparative analysis](#comparative-analysis)
5. [What nobody does well](#what-nobody-does-well)

---

## The landscape

AI memory systems cluster into three approaches:

**Extract-and-store**: Process conversations into discrete facts, store them, retrieve by similarity. Mem0 is the exemplar. Lightweight, fast, but memories are isolated atoms with no structure connecting them.

**Graph-based**: Build knowledge graphs from conversations, retrieve by traversal. Zep/Graphiti, HippoRAG, GraphRAG, Cognee. Richer representations, but heavy infrastructure and the extraction step is fragile (LLM-dependent, error-prone).

**Stream-based**: Append observations to a memory stream, retrieve by a scoring function over recency/importance/relevance. Generative Agents is the exemplar. Simple, but no consolidation and no forgetting.

Somnigraph doesn't fit neatly into any of these. It stores discrete memories (like extract-and-store), builds a graph of relationships between them (like graph-based), and shapes retrieval through a feedback loop (unlike any of them). The sleep pipeline adds offline consolidation that none of the others attempt at this level.

---

## System profiles

### Mem0

**What it is**: An LLM-powered memory layer that extracts facts from conversations via function calling (ADD/UPDATE/DELETE/NOOP operations), deduplicates against existing memories, and retrieves by vector similarity.

**What it does well**:
- The LLM-as-operation-classifier pattern via tool calls is clean and production-proven. Rather than rule-based extraction, the LLM decides whether a piece of conversation is worth storing and how it relates to what's already there.
- InformationContent() gating prevents destructive updates — a vague new fact won't overwrite a specific existing one.
- Token-efficient: ~7K tokens per conversation vs. Zep's 600K+.

**Where it falls short**:
- Binary contradiction handling. When facts conflict, the old one is hard-deleted. No temporal record, no "we used to think X, now we think Y."
- No forgetting mechanism. Everything persists equally forever.
- No consolidation. The store grows monotonically.
- All memories are treated uniformly — no categories, no priority, no decay.
- The graph variant (Mem0g) actually underperforms on simple queries because graph traversal adds noise for non-multi-hop retrievals.

**Key numbers**: p95 latency 1.44s; LoCoMo retrieval F1 reported but single-benchmark.

### Zep / Graphiti

**What it is**: A three-tier temporal knowledge graph. Episodes (raw messages) → semantic entities (facts as edges between entities) → communities (clustered summaries). Graphiti is the production Python implementation (~15K LOC).

**What it does well**:
- Bi-temporal modeling is genuinely novel. Four timestamps per edge track both when facts are true in the world (valid_at/invalid_at) and when the system learned them (created_at/expired_at). This enables temporal reasoning that flat systems fundamentally cannot do.
- Soft invalidation preserves history. Contradicted facts aren't deleted — they're marked with an expiration timestamp.
- Entity resolution is sophisticated: deterministic pre-filter (MinHash LSH, Jaccard 0.9) catches obvious matches cheaply, LLM handles the rest.
- Bidirectional temporal invalidation — if a new fact describes an earlier time period, it can invalidate a "newer" fact that was actually about a later period.

**Where it falls short**:
- Massive infrastructure overhead. Requires Neo4j or FalkorDB, 90% slower than Mem0 on latency benchmarks.
- No decay mechanism. Graph entities persist indefinitely.
- Community detection requires periodic full refreshes.
- No feedback loop — retrieval quality doesn't improve with use.

**What we took**: The `valid_from`/`valid_until` temporal model. Bidirectional invalidation logic. Confirmation that hybrid search (vector + keyword + graph) outperforms any single channel.

### HippoRAG

**What it is**: A neurobiologically-inspired retrieval system that builds a knowledge graph from OpenIE triples, connects entities via synonym edges, and retrieves using Personalized PageRank (PPR) instead of BFS traversal.

**What it does well**:
- PPR handles variable-depth multi-hop paths in a single step. No iterative expansion, no depth limit — activation propagates through the graph until convergence.
- Synonym edges enable cross-reference between surface forms ("the database" vs "Postgres" vs "our persistence layer").
- Node specificity weighting (inverse passage frequency) prevents high-degree hub nodes from dominating results.
- The ablation study is exemplary: each component's contribution is measured. Removing PPR and using BFS drops Recall@2 from 40.9 to 25.4 — a clear demonstration that traversal strategy matters.
- Cost-efficient: 9.2M tokens vs. GraphRAG's 115.5M for the same corpus.

**Where it falls short**:
- Every indexing operation requires an LLM call (OpenIE extraction). This is expensive for a personal memory system that writes frequently.
- No temporal awareness. No decay. No consolidation.
- 26% of v2 failures traced to extraction quality — the LLM extraction step is the binding constraint.
- Fails on questions requiring broad contextual understanding (not multi-hop, just wide).

**What we took**: The theoretical foundation — hippocampal indexing, CLS complementarity, entity-based pattern completion. The empirical proof that BFS is catastrophic and principled graph traversal (PPR or novelty-scored expansion) is necessary. Node specificity as a suppression mechanism for hubs.

### GraphRAG

**What it is**: Microsoft's system for hierarchical community detection over knowledge graphs. Transforms documents into entity-relationship graphs, clusters them with Leiden algorithm, and generates structured reports at multiple resolution levels.

**What it does well**:
- Hierarchical summarization with sub-report substitution. When a cluster is too large for context, its sub-clusters' summaries are substituted — recursively, until it fits. This is exactly the detail→summary→gestalt layering that Somnigraph's CLS hierarchy implements.
- Grounded claims with record ID citations prevent hallucination in summaries.
- Contradiction resolution is explicitly prompted during summarization.
- Three query modes (global map-reduce, local neighborhood, DRIFT iterative) cover different retrieval patterns.

**Where it falls short**:
- One-shot consolidation. The graph is built once during indexing; "incremental update" just appends without re-clustering. No ongoing maintenance.
- Leiden clustering is overkill for personal memory scale (designed for thousands of documents).
- No temporal awareness, no decay, no feedback loop.
- Importance ratings assigned during indexing have no feedback mechanism — they never update.

**What we took**: The hierarchical context substitution pattern. Structured report format (title, summary, importance, grounded findings). Rolling-window summarization for large clusters. The idea that summaries should cite their sources.

### Generative Agents

**What it is**: The Park et al. (Stanford/Google) architecture enabling 25 agents to exhibit believable long-term behavior. Three components: memory stream (append-only log), reflection (LLM-synthesized insights), and planning.

**What it does well**:
- The reflection mechanism is the most reusable contribution. Three phases: identify what needs consolidation (generate 3 questions from recent memories) → gather evidence (retrieve relevant memories for each question) → synthesize insights (5 inferences with citation pointers).
- Question-driven reflection prevents bland summarization. Instead of "summarize recent memories," the system asks "what are the 3 most salient questions?" and then answers them.
- Importance-as-trigger for reflection is elegant: consolidation fires when cumulative importance exceeds a threshold, not on a fixed schedule.
- Decay on last-access time (not creation time) keeps actively-used memories hot regardless of age.

**Where it falls short**:
- No decay floor. Everything eventually decays to zero, including important memories that happen to go unaccessed.
- Static importance scoring — assigned once at creation, never updated.
- No contradiction detection. The memory stream can contain conflicting facts without awareness.
- Reflections compete with observations in the same retrieval pool. A meta-reflection about "what I've been thinking about chess" can be retrieved instead of the actual chess memory.

**What we took**: The question-generation pattern for consolidation. Citation pointers enabling reflection trees. Importance-as-trigger (though we use cumulative unprocessed memories, not importance sum). The decay model, improved with a floor (base_priority × DECAY_FLOOR) and reheat (access resets last_accessed).

### A-Mem

**What it is**: A Zettelkasten-inspired system where every memory is an "atomic note" with 7 metadata fields. New memories trigger 3 LLM calls: note construction (extract keywords/tags/context), link generation (judge which neighbors deserve links), and memory evolution (update neighbor descriptions).

**What it does well**:
- Multi-faceted embedding: embeds the concatenation of content + keywords + tags + context rather than raw content alone. This front-loads metadata into the vector space. Somnigraph adopted this directly (embedding enriched text = content + category + themes + summary).
- Link generation as LLM judgment: rather than linking by embedding threshold, an LLM decides which connections are meaningful. The ablation shows +121% improvement on multi-hop queries.
- Bidirectional evolution: when a new memory is stored, existing neighbors' descriptions are updated to reflect the new connection. Memories aren't static after creation.

**Where it falls short**:
- 3 LLM calls per write is expensive. At scale, the write path becomes the bottleneck.
- No forgetting mechanism. Memories accumulate indefinitely.
- In-place mutation with no audit trail — when a memory's description is updated, the original is lost.
- No consolidation or abstraction layers.

**What we took**: The enriched embedding pattern (the single most directly borrowed idea in the system). Empirical proof that link-based retrieval expansion works. The insight that write-time enrichment pays off at read-time.

### Hexis

**What it is**: A PostgreSQL-native cognitive architecture by Eric Hartford (QuixiAI) that provides persistent memory, identity, and autonomous behavior to AI agents. All state lives in Postgres (memories, worldview, goals, graph); Python workers are stateless. Uses pgvector for embeddings, Apache AGE for knowledge graphs, and an autonomous heartbeat loop with energy-budgeted actions.

**What it does well**:
- Precomputed neighborhoods for spreading activation. Memory neighborhoods are computed during maintenance and stored as JSONB, enabling O(1) associative recall without real-time graph traversal. Hot-path retrieval combines vector similarity + neighborhood expansion + temporal context in a single SQL query (~5-50ms).
- Energy budget as situational constraint. Actions are gated not by compute cost but by irreversibility, social exposure, and identity impact. A web search costs 2; sending a message costs 5. Context multipliers (first use, error rate, time-of-day) add nuance. The agent can propose cost changes but can't modify them directly.
- Drives as internal pressure. Four accumulating drives (curiosity, connection, coherence, competence) nudge the heartbeat toward behaviors without commanding them. The coherence drive — pressure to resolve contradictions — creates natural urgency for knowledge base hygiene.
- Identity as evolving epistemology. Worldview memories with confidence scores, Big Five personality encoding, beliefs that accumulate evidence. The agent's identity is memories, not configuration.
- Rich graph vocabulary. 18 edge types across 11 node types in Apache AGE, including `CAUSES`, `BLOCKS`, `CONTESTED_BECAUSE`, `INSTANCE_OF` — enabling reasoning patterns that simpler edge schemas can't express.

**Where it falls short**:
- No feedback loop. Scoring is a fixed formula (importance × decay × similarity). No mechanism to learn from retrieval outcomes — parameters are hand-set and static.
- No keyword retrieval channel. Vector-primary with pg_trgm as secondary. No BM25 or FTS5 equivalent, no RRF fusion of multiple channels.
- Heavy infrastructure. Requires PostgreSQL + Apache AGE + Ollama (or other embedding service) + Docker. Not portable in the way SQLite-based systems are.
- No LLM-mediated consolidation. Maintenance worker does clustering and neighborhood recomputation, but no question-driven summarization, no merge/archive/annotate decisions, no Hebbian edge weighting from co-retrieval.
- Embeddings are not enriched. Raw content is embedded directly — no concatenation of categories, themes, or summaries into the vector space.

**What's interesting for us**: The precomputed neighborhood pattern could optimize Somnigraph's per-query PPR expansion — compute during sleep, use as fast path, fall back to live expansion for fresh memories. The coherence drive (urgency signal for unresolved contradictions) could improve our contradiction handling. The energy budget is a thoughtful model for autonomous agent constraints, orthogonal to our scope but a good reference.

### Cognee

**What it is**: A document-to-knowledge-graph system with LLM extraction, multi-backend storage, and 14 search types. The flagship retrieval mode (GRAPH_COMPLETION) seeds with vector search, projects through the graph, and scores triplets by importance.

**What it does well**:
- Triplet embeddings: concatenate `source → relationship → target` and embed as a vector. This makes graph relationships directly searchable via vector similarity — you can query for a relationship type, not just an entity.
- Usage frequency tracking writes back to graph edges, creating a lightweight feedback signal.
- Multi-backend support (Neo4j, Kuzu, PGVector, ChromaDB) makes it adaptable.

**Where it falls short**:
- No decay mechanism. Everything persists equally.
- Document-first orientation: designed for batch processing, not conversational use.
- "Memify" enrichment is additive only — no synthesis, no evolution detection, no contradiction handling.

**What we took**: The triplet embedding concept influenced our edge `linking_context` + `linking_embedding` design. Usage frequency as a signal for edge weighting.

---

## What we borrowed

A summary of borrowed ideas, attributed to their source:

| Idea | Source | How we adapted it |
|------|--------|-------------------|
| Enriched embeddings (embed metadata, not just content) | A-Mem | Content + category + themes + summary concatenated before embedding |
| Temporal bounds (valid_from/valid_until) | Zep/Graphiti | Same schema, but populated by sleep classification rather than real-time extraction |
| PPR-style graph traversal | HippoRAG | Novelty-scored expansion (one-hop now, PPR planned) instead of BFS |
| Question-driven consolidation | Generative Agents | Sleep REM generates questions to guide summary writing |
| Hierarchical summarization | GraphRAG | Detail → summary → gestalt layers with context substitution |
| Decay with reheat | Generative Agents | Added floor (memories never fully decay) and per-category rates |
| LLM-as-classifier for memory operations | Mem0 | NREM classification of memory pairs (supports/contradicts/evolves/duplicates) |
| Triplet-style edge embeddings | Cognee | Linking context embedded and used for query-conditioned edge firing |
| Grounded claims in summaries | GraphRAG | Sleep-generated summaries track source memory IDs via derivation edges |

## Comparative analysis

### What Somnigraph does differently

**Feedback loop.** No other system in our corpus closes the retrieval-feedback loop. Mem0 tracks nothing about retrieval quality. Zep builds a graph but doesn't learn which edges are useful. HippoRAG doesn't even record what was retrieved. Somnigraph's `recall_feedback()` creates a gradient signal that reshapes scoring, adjusts decay rates, strengthens/weakens edges, and enriches themes. This is the primary differentiator.

**Offline consolidation with LLM judgment.** Generative Agents has reflection, but it's online (triggered during conversation). GraphRAG has community summarization, but it's one-shot. Somnigraph runs three-phase consolidation offline, with an LLM making per-memory and per-cluster decisions about what to merge, archive, rewrite, or annotate. The `annotate` action (preserving temporal arcs without merging) doesn't exist in any other system we surveyed.

**Adaptive scoring.** Most systems have fixed scoring functions. Somnigraph's scoring adapts through feedback (memories that prove useful score higher), co-retrieval (memories retrieved together develop affinity), and edge weight evolution (graph connections strengthen or weaken based on endpoint utility). The scoring pipeline isn't a static algorithm — it's a learning system.

### What others do better

**Real-time graph construction.** Zep/Graphiti and Cognee extract entities and relationships in real-time during conversations. Somnigraph's graph is built primarily during sleep (offline), which means new memories don't immediately benefit from graph structure. For multi-hop queries about recently-stored information, this is a real disadvantage.

**Entity resolution.** Graphiti's two-pass entity resolution (deterministic pre-filter + LLM) is more sophisticated than anything in Somnigraph, which has no entity resolution at all. "The database," "Postgres," and "our persistence layer" are separate concepts unless a human or sleep cycle links them.

**Scale.** HippoRAG and GraphRAG are designed for thousands of documents and millions of tokens. Somnigraph is tested at ~700 memories. The novelty-scored expansion, the feedback loop, the sleep pipeline — these are all validated at personal scale. Whether they work at 10x or 100x is unknown.

**Temporal reasoning.** Zep's bi-temporal model with four timestamps per edge is more rigorous than Somnigraph's two-field (valid_from/valid_until) approach. Somnigraph doesn't track when the system learned a fact vs. when the fact was true in the world, which limits certain temporal queries.

---

## What nobody does well

These problems are unsolved across all systems we surveyed:

### Contradiction detection

Every system either ignores contradictions or handles them poorly. Mem0 hard-deletes the old fact. Zep invalidates edges but requires extraction to detect the conflict. HippoRAG and GraphRAG don't detect contradictions at all. Generative Agents stores conflicting facts without awareness.

Benchmark performance across all systems: 0.025–0.037 F1 on contradiction detection tasks. This isn't a tuning problem — it's a representation problem. Detecting that "we use REST" and "we migrated to GraphQL" are about the same claim requires understanding what a "claim" is, which current systems don't model.

### Write-path quality

All systems accept whatever they're given. Mem0's InformationContent() gating is the closest to write-path quality control, but it only prevents destructive updates — it doesn't evaluate whether a new memory is worth storing.

The result: every system accumulates low-value memories over time. Consolidation (where it exists) can clean up afterward, but preventing low-quality writes in the first place would be more efficient.

### Consolidation evaluation

No benchmark exists for evaluating whether consolidation improved the memory store. GraphRAG's community reports have no ground truth for comparison. Generative Agents' reflections are evaluated by human judgment only. Somnigraph's sleep pipeline is evaluated manually.

Building a consolidation benchmark requires: a memory store with known quality issues, a ground-truth consolidation, and metrics that capture both compression and fidelity. This is an open research problem.

### Temporal grounding

Most memories lack temporal anchors beyond their creation timestamp. "We decided to use Postgres last month" — when is "last month"? If the memory was stored immediately, creation time works. If it was stored days later during a recap, creation time is wrong.

Zep's bi-temporal model is the most thoughtful approach, but even it requires accurate extraction of temporal references from conversation — which is an unsolved NLP problem for casual temporal expressions.

---

*For the full research corpus of 63 source analyses, see `research/sources/`. Each analysis extracts architecture, key claims with evidence, relevance to this project, and ideas worth borrowing.*
