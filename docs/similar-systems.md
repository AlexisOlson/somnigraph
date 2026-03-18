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

**Agent-managed**: Give the LLM agent direct control over its own memory hierarchy, with tools to read, write, and edit memory at multiple tiers. Letta (MemGPT) is the exemplar. Flexible and agent-driven, but memory quality depends on the agent's meta-cognitive judgment — which is unreliable for long-term decisions.

**Temporal-first**: Extract facts from conversations with bi-temporal validity (event time + transaction time), supersede rather than overwrite on contradiction. memv is the clearest implementation. Clean temporal reasoning, but no offline consolidation and retrieval is basic.

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

### OmniMemory

**What it is**: An enterprise memory subsystem for the OmniNode multi-agent platform. Built on the ONEX 4-node architecture (EFFECT/COMPUTE/REDUCER/ORCHESTRATOR), with five storage backends: Qdrant (vectors), Memgraph (graph), PostgreSQL (state), Valkey (cache), and filesystem (archive). All operations typed through strict Pydantic models with zero `Any` types.

**What it does well**:
- Lifecycle state machine with five states (ACTIVE → STALE → EXPIRED → ARCHIVED → DELETED). The STALE intermediate is the most interesting: memories past their soft TTL are flagged but still accessible, allowing graceful degradation rather than binary alive/dead transitions.
- Optimistic concurrency for lifecycle transitions. A `lifecycle_revision` counter with CAS-style updates prevents concurrent agents from corrupting state. Well-established pattern, cleanly implemented.
- Frozen (immutable) models at subsystem boundaries prevent mutation bugs that plague distributed systems where objects are passed by reference through multiple layers.
- Integrated PII detection in the write path — regex-based scanning for 7 PII types (email, phone, SSN, credit card, IP, API keys, password hashes) with <10ms overhead. Not NER-grade, but having any PII filtering is ahead of most memory systems.
- Intent event sourcing: first-class support for storing and querying intent classifications from Kafka, linking memory to agent decision-making.

**Where it falls short**:
- No feedback loop. Scoring is a fixed composition of similarity, BM25, recency, and graph proximity. Nothing learns from retrieval outcomes.
- No consolidation. REDUCER nodes are scaffolded but not implemented. No question-driven summarization, no merge/archive decisions, no contradiction detection.
- No enriched embeddings. Raw content embedded directly — no metadata concatenation into the vector space.
- BFS for graph traversal (HippoRAG demonstrated PPR is superior for variable-depth multi-hop).
- Heavy infrastructure: Docker Compose with 6 services. Justified for enterprise multi-agent deployments, but a high floor for adoption.

**What's interesting for us**: The STALE intermediate state could complement somnigraph's continuous EWMA decay — a memory decayed below threshold but not yet consolidated could be flagged for sleep review. The PII detection pattern is a responsible addition for a public repo. Frozen boundary models could catch mutation bugs in sleep pipeline chains. The optimistic concurrency pattern would matter if somnigraph ever supported concurrent agent recalls.

### Letta (MemGPT)

**What it is**: A platform for building stateful AI agents, originating from the 2023 paper "MemGPT: Towards LLMs as Operating Systems" (Packer et al., arXiv:2310.08560). The core insight is treating the LLM context window like virtual memory: the agent manages its own memory hierarchy, paging information between fast in-context storage and slower external storage. Apache-2.0, ~21k stars, actively maintained with 100+ database migrations.

**What it does well**:
- The agent-as-memory-manager paradigm is genuinely influential. The LLM decides when and how to move information between three tiers: core memory (editable text blocks pinned in the system prompt), recall memory (searchable conversation history), and archival memory (long-term passage store). No external system decides what to remember.
- Rich memory editing tools. Core blocks support append, find-and-replace, line-insert, full rewrite (`memory_rethink`), and unified diff patches. This reflects real iteration on what agents need.
- Git-backed memory versioning (`BlockManagerGit`) with commit history, author attribution, and point-in-time retrieval. Every edit is attributable and reversible.
- Sleeptime agents: background agents that review conversation history between turns and update shared memory blocks. Same agent loop for background processing and foreground use — architecturally elegant.
- Shared memory blocks across agents via junction table. When one agent updates a shared block, all agents sharing it see the change.

**Where it falls short**:
- Archival retrieval is surprisingly basic — vector-only cosine similarity with no fusion, no learned ranking, no feedback loop. For a system that stores all long-term knowledge in archival passages, this is the weakest point.
- No decay or forgetting. Memories accumulate indefinitely. No system-level lifecycle management.
- No contradiction detection. Old content is simply overwritten when the agent edits core blocks.
- No offline consolidation. Sleeptime agents are online (between turns), not batch. They're an LLM reading recent conversation and updating blocks — shallow compared to multi-phase consolidation with classification, summarization, and maintenance.
- No evaluation framework in the open-source repo. No ground-truth, no retrieval metrics, no A/B infrastructure.
- Agent memory decisions are unreliable. Agents over-store trivial information and under-store important context. The fundamental promise — that the agent will manage its own memory well — depends on meta-cognitive capability that current LLMs don't reliably have.

**What's interesting for us**: The "pinned memory block" concept (editable always-in-context text) addresses layered memory from a different angle than importance scoring — a qualitatively different tier for always-relevant information. Block checkpointing for memory provenance (who changed what, when, with undo) could improve auditability of sleep consolidation edits. The variety of editing operations suggests a lightweight `update_memory()` tool could complement `remember()`.

### memv

**What it is**: A temporal memory library for AI agents (MIT, v0.1.0, Python 3.13+). SQLite + sqlite-vec + FTS5. Synthesizes ideas from Nemori (predict-calibrate extraction), Graphiti (bi-temporal validity), and SimpleMem (write-time temporal normalization). The claim that "no single competitor has all four" core features is credible based on our corpus.

**What it does well**:
- Strongest bi-temporal model in the corpus. Dual timelines: event time (`valid_at`/`invalid_at` — when the fact was true) and transaction time (`created_at`/`expired_at` — when the system learned it). Facts are never deleted on contradiction; they're expired and linked to their successor via `superseded_by`. Point-in-time queries on both axes.
- Predict-calibrate extraction faithfully implements the Nemori/Hindsight approach: predict what the conversation should contain given existing knowledge, then extract only the gaps. Importance emerges from prediction error, not explicit scoring.
- Extraction-source discipline: extraction works from raw messages (ground truth), not from LLM-generated episode narratives. Prevents hallucination propagation into the knowledge base.
- Write-time temporal normalization: relative dates ("last week") resolved to absolute timestamps at extraction time. Accounts for a major class of temporal reasoning failures.
- Episode segmentation groups conversations into topic-coherent units before extraction, providing context for the predict-calibrate loop.

**Where it falls short**:
- No retrieval learning or feedback. Static RRF with fixed k=60, no mechanism to learn from retrieval outcomes.
- No offline processing. Knowledge stays exactly as extracted — no consolidation, no decay, no relationship discovery, no maintenance.
- No graph structure. The planned `extends` relationship (v0.2.0) addresses parent-child facts only.
- LLM-expensive write path: 3-5 LLM calls per conversation turn (segmentation, episode generation, prediction, extraction). Somnigraph's write path is zero LLM calls; LLM work happens during offline sleep.
- v0.1.0 maturity. Acknowledged missing features; the "Not doing (yet)" list is longer than the "Completed" list.
- Benchmark claims unvalidated. LongMemEval harness built but not run.

**What's interesting for us**: The `superseded_by` field for explicit contradiction chains — a schema migration that would make Somnigraph's contradiction handling traversable. An `at_time` parameter on `recall()` for point-in-time queries — Somnigraph has the temporal fields but doesn't expose them through the API. The extraction-source discipline principle — worth auditing whether any sleep consolidation steps extract from LLM-generated content rather than originals. Write-time temporal normalization for `remember()` or during sleep.

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

**Temporal reasoning.** memv has the strongest bi-temporal model in the corpus — dual timelines (event time vs. transaction time) with explicit `superseded_by` chains and point-in-time queries. Zep's four-timestamp model on edges is also more rigorous than Somnigraph's two-field (valid_from/valid_until) approach. Somnigraph doesn't track when the system learned a fact vs. when the fact was true in the world, which limits certain temporal queries.

---

## What nobody does well

These problems are unsolved across all systems we surveyed:

### Contradiction detection

Most systems either ignore contradictions or handle them poorly. Mem0 hard-deletes the old fact. Zep invalidates edges but requires extraction to detect the conflict. HippoRAG, GraphRAG, and Letta don't detect contradictions at all. Generative Agents stores conflicting facts without awareness. memv handles contradictions well at extraction time (predict-calibrate surfaces them, `superseded_by` chains preserve history) but cannot detect transitive contradictions (a child fact surviving when its parent is superseded). Engram has the most sophisticated approach: five-level graded tension classification (hard/temporal/contextual/soft/none) with distinct behaviors per level.

Benchmark performance across all systems: 0.025–0.037 F1 on contradiction detection tasks. This isn't a tuning problem — it's a representation problem. Detecting that "we use REST" and "we migrated to GraphQL" are about the same claim requires understanding what a "claim" is, which current systems don't model.

### Write-path quality

Most systems accept whatever they're given. Mem0's InformationContent() gating is the closest to write-path quality control, but it only prevents destructive updates — it doesn't evaluate whether a new memory is worth storing. memv's predict-calibrate extraction is the notable exception: by predicting what a conversation should contain and extracting only the gaps, it filters at write time rather than relying on post-hoc consolidation. This is the most principled write-path quality approach in the corpus.

The result for most systems: low-value memories accumulate over time. Consolidation (where it exists) can clean up afterward, but preventing low-quality writes in the first place would be more efficient.

### Consolidation evaluation

No benchmark exists for evaluating whether consolidation improved the memory store. GraphRAG's community reports have no ground truth for comparison. Generative Agents' reflections are evaluated by human judgment only. Somnigraph's sleep pipeline is evaluated manually.

Building a consolidation benchmark requires: a memory store with known quality issues, a ground-truth consolidation, and metrics that capture both compression and fidelity. This is an open research problem.

### Temporal grounding

Most memories lack temporal anchors beyond their creation timestamp. "We decided to use Postgres last month" — when is "last month"? If the memory was stored immediately, creation time works. If it was stored days later during a recap, creation time is wrong.

memv makes the strongest attempt: write-time temporal normalization resolves relative dates to absolute timestamps at extraction time, and the bi-temporal model with `superseded_by` chains creates explicit fact-evolution histories. Zep's bi-temporal model on edges is also thoughtful. But even these require accurate extraction of temporal references from conversation — which remains hard for casual temporal expressions.

---

*For the full research corpus of 63 source analyses, see `research/sources/`. Each analysis extracts architecture, key claims with evidence, relevance to this project, and ideas worth borrowing.*
