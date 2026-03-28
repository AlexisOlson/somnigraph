# MemOS: A Memory OS for AI System -- Analysis

*Updated 2026-03-25 by Opus agent reading arXiv:2507.03724v4 (Dec 2025), GitHub repo, and PERMA benchmark results. Replaces prior analysis (2026-02-18) based on short vision paper 2505.22101v1.*

---

## Paper Overview

- **Authors**: Zhiyu Li, Chenyang Xi, Chunyu Li, Ding Chen, et al. (35+ authors)
- **Affiliations**: MemTensor (Shanghai) Technology Co., Institute for Advanced Algorithms Research, China Telecom Research Institute, Tongji/Zhejiang/Peking/Renmin/Beihang universities, Shanghai Jiao Tong University
- **Venue**: arXiv preprint, v4 dated December 3, 2025
- **Paper**: arXiv:2507.03724v4 [cs.CL], 37 pages
- **Code**: https://github.com/MemTensor/MemOS (Apache 2.0)
- **Website**: https://memos.openmem.net/

**Core problem**: LLMs lack a unified framework for managing memory as a first-class system resource. Parametric memory (weights) is opaque and expensive to update. Activation memory (KV-cache) is ephemeral. RAG-based plaintext memory is stateless. The result is four gaps: no long-range dependency modeling, poor knowledge evolution, no persistent personalization, and cross-platform memory silos.

**Key contribution**: A full memory operating system with three-layer architecture (Interface, Operation, Infrastructure), three memory types (plaintext, activation, parametric), and the MemCube abstraction as a universal container. Unlike the earlier short version (2505.22101), this paper includes extensive benchmarks across four evaluation suites and a working open-source implementation.

**Result**: SOTA on LoCoMo (75.80 overall LLM judge), LongMemEval (77.8 overall), PreFEval (77.2 personalized response, 0 turns), and PersonaMem (61.2 precision). Benchmarked against MIRIX, Mem0, Zep, Memobase, Supermemory, and MemU using GPT-4o-mini as the common backbone. Also demonstrates KV-cache memory acceleration (up to 94.2% TTFT reduction on Qwen models) and API robustness (100% success rate at 100 QPS).

---

## Architecture

### Storage & Schema

**Backend**: Neo4j graph database as the primary store. The `TreeTextMemory` class (the main memory implementation) wraps a `Neo4jGraphDB` graph store. Alternative backends exist in code for PostgreSQL (PolarDB) and Nebula, but Neo4j is the tested path. Vector search is done through Neo4j's native vector index, not a separate vector DB. Qdrant and Milvus adapters exist for the vec_db abstraction but are not used by the tree memory path.

**Memory item schema** (from `TextualMemoryItem` / `TreeNodeTextualMemoryMetadata`):
- `id`: UUID
- `memory`: string content
- `status`: "activated" | "resolving" | "archived" | "deleted"
- `memory_type`: "WorkingMemory" | "LongTermMemory" | "UserMemory" | "OuterMemory" | "ToolSchemaMemory" | "ToolTrajectoryMemory" | "RawFileMemory" | "SkillMemory" | "PreferenceMemory"
- `sources`: list of `SourceMessage` (provenance tracking with role, content, chat_time, doc_path, file_info)
- `embedding`: vector
- `tags`: list of strings
- `confidence`: float 0-100
- `version`: int (incremented on update)
- `history`: list of `ArchivedTextualMemory` (preserves prior versions with update_type: conflict/duplicate/extract/unrelated)
- `key`: memory title/key
- `background`: contextual background string
- `usage`: list of usage history entries
- `visibility`: "private" | "public" | "session"
- `created_at`, `updated_at`: ISO timestamps

### Memory Types (as implemented)

The paper describes three conceptual types (plaintext, activation, parametric). In the codebase, the textual/plaintext path is the mature implementation. Activation memory (KV-cache) has basic support (`kv.py`, `vllmkv.py`) for local open-weight models. Parametric memory (LoRA) has a stub implementation (`lora.py`) that is essentially a placeholder.

The real differentiation is within plaintext memory, which is subdivided into:

1. **WorkingMemory**: Short-term, recent context. Capped (default 20 items). Automatically cleaned up when long-term memories are added.
2. **LongTermMemory**: Persistent extracted facts. Default cap 1500.
3. **UserMemory**: User-specific attributes and preferences. Default cap 480.
4. **PreferenceMemory**: Separate extraction pipeline for explicit/implicit preferences with dedicated metadata (`preference_type`, `dialog_id`, `original_text`).
5. **ToolSchemaMemory / ToolTrajectoryMemory / SkillMemory / RawFileMemory**: Specialized types for tool usage patterns, raw document chunks, etc.

### Write Path

1. **MemReader** parses incoming messages (conversation turns, documents, images) into structured `TextualMemoryItem` objects. Uses LLM to extract memories from conversation context, with separate extractors for general memories and preferences.
2. **MemFeedback** (the write-path module, despite the name) handles conflict detection: when new memories are added, it searches for existing memories above a similarity threshold (0.80 default), then uses LLM to judge whether the new memory conflicts with, duplicates, or is unrelated to existing ones. Conflicting/duplicate memories trigger updates with version history.
3. **MemoryManager** adds items to Neo4j. Dual-write: items go to both WorkingMemory (for immediate context) and their target memory type (LongTermMemory, etc.). Batch or parallel insertion. After sync-mode adds, working memory is cleaned up (overflow eviction).
4. **Reorganizer** (`GraphStructureReorganizer`) runs as an optional post-write step, detecting pairwise relations (CAUSE, CONDITION, CONFLICT, RELATE) between new and existing nodes via LLM, and inferring new fact nodes from causal/conditional relations. Currently mostly commented out in `process_node()` -- the relation detection, inference, sequence links, and aggregate concept detection are all wrapped in commented-out blocks.

### Retrieval

The retrieval pipeline (`Searcher` -> `AdvancedSearcher`) follows this flow:

1. **TaskGoalParser**: LLM decomposes query into structured task goal with keywords, topic, concept, time scope, entity focus.
2. **GraphMemoryRetriever**: Parallel hybrid retrieval:
   - **Graph recall**: Traverses Neo4j using parsed task goals (topic/concept/fact nodes)
   - **Vector recall**: Embedding similarity search via Neo4j's vector index
   - **BM25 recall**: Optional in-memory BM25 (custom `EnhancedBM25` class, not a database index)
   - **Fulltext recall**: Optional Neo4j fulltext index search
   - Results are merged by ID (union, no fusion scoring)
3. **Reranker**: Pluggable reranker architecture. Default is `cosine_local` (cosine similarity with level-based weights for topic/concept/fact). Also supports BGE reranker via HTTP and concatenation-based strategies. Strategies split dialogue sources into turn pairs and score them individually.
4. **Deduplication**: Cosine similarity deduplication to remove near-identical results.
5. **Reasoner** (optional, "fine" mode only): LLM-based chain-of-thought reasoning over retrieved memories to filter and synthesize.

**Important gap between paper and code**: The paper describes "task-aligned memory routing" with a `MemoryPathResolver` decomposing queries into topic-concept-fact hierarchies. The code has `TaskGoalParser` which does query decomposition, but the hierarchical routing is implemented as parallel search across memory scopes, not as the structured path resolution the paper suggests.

### Consolidation / Processing

The **MemFeedback** module is the closest thing to consolidation. When new memories arrive:
- It identifies keywords, searches for related existing memories
- Uses LLM to judge if new content should update, replace, or coexist with existing memories
- On conflict/duplicate: updates existing memory content, increments version, archives old content in the `history` field plus a separate archived node
- Supports keyword replacement tracking

The **Reorganizer** is designed to detect structural relations and create graph edges, but the core functionality is largely disabled in the current code (commented out). When enabled, it would detect CAUSE/CONDITION/CONFLICT/RELATE edges and infer new fact nodes.

There is no sleep-like offline consolidation, no scheduled background processing, and no decay mechanism in the codebase. Memory lifecycle states exist in the schema (Generated -> Activated -> Merged -> Archived) but transitions are triggered by write-time conflict detection, not by temporal or usage patterns.

### Lifecycle Management

The lifecycle state machine described in the paper (Generated, Activated, Merged, Archived, with Time Machine rollback and Frozen state) is partially implemented:
- `status` field supports "activated", "resolving", "archived", "deleted"
- Version history is maintained in the `history` field
- No evidence of freeze/frozen state in the code
- No Time Machine / rollback implementation found
- No TTL enforcement or time-based archival

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| SOTA on LoCoMo (75.80 overall) | Table 3: beats MIRIX (64.33), Mem0 (64.57), Zep (59.22), Memobase (72.01), MemU (56.55), Supermemory (55.34) | Strong. All systems use GPT-4o-mini backbone. Comprehensive comparison across 5 task types. |
| SOTA on LongMemEval (77.8 overall) | Table 4: beats all baselines. Notably strong on single-session preference (96.7) and single-session user (95.7) | Strong. Wide margin on preference and user categories. |
| SOTA on PreFEval (77.2 / 71.9) | Table 5: best personalized response in both 0-turn and 10-turn settings, lowest preference unaware error | Strong. Demonstrates robustness to irrelevant conversation noise. |
| SOTA on PersonaMem (61.2 precision) | Table 6: beats Memobase (58.9), Mem0 (57.8), Zep (57.8) | Moderate. Margins are smaller; 1424 context tokens is moderate. |
| Three memory types form a complete hierarchy | Paper describes plaintext/activation/parametric | Overreached. Only plaintext is mature. Activation (KV-cache) is basic. Parametric (LoRA) is a stub. The "unified scheduling across memory types" does not exist in code. |
| MemLifecycle provides formal state management | Paper describes 5 states + Time Machine + Frozen | Partially implemented. Status field exists with 4 states. No freeze, no rollback, no TTL enforcement. |
| Hierarchical graph organization (topic-concept-fact) | Paper and code: TaskGoalParser decomposes queries | Real in retrieval. The parser extracts structured goals. But graph nodes are not themselves organized into topic-concept-fact layers -- retrieval uses the decomposition to search, not to navigate a pre-built hierarchy. |
| KV-cache injection provides up to 94.2% TTFT reduction | Table 8: systematic evaluation across 3 Qwen models, 3 context lengths, 3 query lengths | Strong for open-weight models. Not applicable to API-hosted models. |
| 100% success rate at 100 QPS | Table 7: while all other systems degrade, MemOS maintains 100% success | Strong. Demonstrates engineering quality of the server infrastructure (Redis queues, async handlers). |
| Relation and reasoning detection enriches graph | Code: `RelationAndReasoningDetector` | Designed but disabled. All four detection methods (pairwise relations, inferred nodes, sequence links, aggregate concepts) are commented out in `process_node()`. |

---

## PERMA Benchmark Results

From arXiv:2603.23231 (PERMA paper). All memory systems use GPT-4o-mini as backbone.

| Setting | MCQ Acc. | BERT-F1 | Memory Score | Context Tokens | Completion | Turn=1 | Turn<=2 |
|---------|----------|---------|-------------|----------------|------------|--------|---------|
| Clean Single | 0.811 | 0.830 | **2.27** | 709.1 | 0.842 | **0.548** | 0.801 |
| Clean Multi | 0.732 | 0.819 | **2.14** | 664.7 | 0.643 | 0.306 | 0.592 |
| Noise Single | **0.853** | **0.844** | **2.38** | 1486.7 | **0.837** | **0.567** | **0.837** |
| Noise Multi | **0.752** | 0.816 | **2.17** | 680.6 | 0.650 | 0.268 | 0.580 |
| Style-aligned Single | **0.813** | -- | -- | 647.7 | -- | 0.563 | 0.839 |
| Style-aligned Multi | **0.764** | -- | -- | 644.1 | -- | **0.331** | **0.637** |

### Analysis

**Noise robustness is the headline finding**: MemOS improves with noise (MCQ 0.811 -> 0.853, Memory Score 2.27 -> 2.38). This happens because noise doubles the context token volume (709 -> 1487), providing richer retrieval context. This is counterintuitive -- most systems degrade with noise -- and suggests MemOS's retrieval is robust enough that more context helps even when the additional context contains noise. The implication is that MemOS may be under-retrieving in the clean setting.

**Multi-domain collapse is universal**: Turn=1 accuracy drops from 0.548 (single-domain) to 0.306 (multi-domain) in clean settings. This is the cross-domain synthesis failure that every benchmarked system shares. MemOS's absolute numbers here are the best, but the relative collapse is just as severe (~44% drop).

**Style-aligned multi-domain is MemOS's strongest relative showing**: 0.764 MCQ and 0.331 Turn=1 in the hardest setting, both best among all systems. The preference extraction pipeline (explicit + implicit) gives it a structural advantage for style-adapted responses.

**Context efficiency**: MemOS uses moderate context (644-1487 tokens), roughly comparable to Mem0g's 592-1399. Not the most parsimonious (Supermemory and MemU use less) but delivers more value per token.

---

## Relevance to Somnigraph

### What MemOS does that Somnigraph doesn't

1. **Preference extraction pipeline**: Dedicated `NaiveExtractor` splits explicit and implicit preferences using separate prompts. Preferences get their own metadata type (`PreferenceTextualMemoryMetadata`) with preference_type, dialog_id, original_text. This is a write-path specialization Somnigraph lacks entirely -- Somnigraph stores preferences as regular memories with no structural distinction.

2. **Real-time conflict detection on write**: MemFeedback searches for similar existing memories before inserting, then uses LLM to classify the relationship (conflict/duplicate/update/unrelated). Somnigraph defers all conflict detection to sleep (NREM pairwise classification). This means Somnigraph can accumulate contradicting memories until the next sleep cycle.

3. **Version history per memory**: Each memory carries an inline `history` list of `ArchivedTextualMemory` objects preserving prior content with version numbers and update types. Somnigraph's `revision` edges track evolution across memories but don't preserve inline history within a single memory.

4. **Multi-memory-type retrieval**: The search pipeline can simultaneously query WorkingMemory, LongTermMemory, UserMemory, PreferenceMemory, ToolMemory, and SkillMemory with per-type top_k controls. Somnigraph has category-based filtering but no structural distinction between short-term and long-term memory stores.

5. **Graph-based structured retrieval**: Neo4j graph traversal alongside vector and BM25 search. While Somnigraph has PPR-based graph expansion, MemOS's graph retrieval uses the parsed task structure (topic/concept/fact) to navigate, which is a different (and more query-adaptive) approach.

6. **Internet retrieval integration**: Built-in internet search retriever can augment memory results with web content (Bocha search, Xinyu search). Somnigraph has no external knowledge integration.

### What Somnigraph does better

1. **Learned reranker**: Somnigraph's 26-feature LightGBM reranker trained on 1032 real-data queries (NDCG=0.7958) is a fundamentally different approach from MemOS's cosine similarity or BGE reranker. A learned model that adapts to actual usage patterns vs. a static similarity model.

2. **Explicit feedback loop**: Per-query utility ratings with EWMA aggregation, UCB exploration bonus, and measured GT correlation (r=0.70). MemOS has no user feedback mechanism -- the `MemFeedback` module is about write-path conflict detection, not retrieval quality feedback.

3. **Offline consolidation (sleep)**: Three-phase LLM-mediated consolidation -- NREM (pairwise classification, edge creation, merge/archive), REM (gap analysis, question generation), orchestrator. MemOS has no offline processing at all. Its reorganizer is designed for online use but largely disabled.

4. **Typed edges with semantic meaning**: Somnigraph's edge types (supports, contradicts, evolves, revision, derivation) are detected by LLM during sleep and carry semantic weight. MemOS's graph edges are structural (SUMMARY, MATERIAL, FOLLOWING, PRECEDING) without semantic typing.

5. **Biological decay**: Per-category exponential decay with configurable half-lives, floors, and reheat on access. MemOS has no decay mechanism -- memories persist indefinitely until explicit deletion or write-path conflict resolution.

6. **Retrieval fusion**: Somnigraph's Bayesian-optimized RRF (k=14) for combining BM25 and vector results. MemOS combines retrieval channels by ID union with no score fusion -- just deduplication. This means MemOS cannot leverage the complementary signal from different retrieval channels.

7. **Measured retrieval quality**: Ground truth evaluation, utility calibration study, multi-hop failure analysis, feature importance analysis. MemOS reports end-to-end benchmark scores but no introspective retrieval analysis.

---

## Worth Stealing

### 1. Write-Path Conflict Detection (High impact, Medium effort)

**What**: Before inserting a new memory, search for similar existing memories (threshold ~0.80) and classify the relationship. On conflict: update the existing memory, preserve history. On duplicate: merge or skip.

**Why**: Somnigraph defers all conflict detection to sleep, which means contradicting memories accumulate between sleep cycles. Write-path detection catches the most obvious conflicts immediately.

**Implementation sketch**: In `impl_remember()`, after embedding the new memory, do a quick vector search (top-3, threshold 0.80). If hits, run a lightweight LLM classification (conflict/duplicate/update/unrelated). On conflict: create a `revision` edge and update the existing memory. On duplicate: skip insertion. This adds one vector search + possibly one LLM call to the write path.

**Caution**: MemOS's MemFeedback module is ~1200 lines and handles many edge cases (keyword replacement, multi-language, chunk splitting). Start simple -- just the similarity check + binary conflict/not-conflict classification.

### 2. Preference Memory Type (Medium impact, Low effort)

**What**: Structurally distinguish preference memories from factual memories with dedicated metadata (preference_type: explicit/implicit, original_text linking back to source).

**Why**: PERMA results show that preference-state maintenance is a distinct capability from factual recall. Having a structural distinction would allow preference-specific retrieval weighting and targeted evaluation.

**Implementation sketch**: Add "preference" to the category enum. On write, optionally tag with preference_type (explicit/implicit). In retrieval, allow boosting preference memories for personalization-heavy queries. This is a schema addition, not a pipeline change.

### 3. Working/Long-term Memory Distinction (Medium impact, Medium effort)

**What**: Maintain a separate short-lived working memory buffer (capped, recent items) alongside long-term storage. Working memory always included in context; long-term memory retrieved by query.

**Why**: MemOS's WorkingMemory (capped at 20) ensures recent context is never missed. Somnigraph's `startup_load` serves a similar purpose but mixes high-priority long-term memories with recent ones.

**Implementation sketch**: This is conceptually what `startup_load` already does. The actionable delta is: ensure the most recent N memories from the current session are always surfaced regardless of their priority/temperature, then let the reranker handle the rest.

### 4. Inline Version History (Low impact, Low effort)

**What**: When a memory is updated (by sleep consolidation or write-path conflict), preserve the prior content inline rather than only as an edge to another node.

**Why**: Makes rollback trivial and provides audit trail without graph traversal. MemOS stores `ArchivedTextualMemory` objects in a `history` list field.

**Implementation sketch**: Add a `history` JSON column to the memories table. On update during NREM merge, append `{version, content, updated_at, update_type}` to history before overwriting. Low-cost addition to the existing merge flow.

---

## Not Useful For Us

### 1. Parametric and Activation Memory
The paper's grand vision of unifying LoRA patches, KV-cache tensors, and plaintext under one abstraction. The code shows this is aspirational -- LoRA is a stub, KV-cache is basic and only for local open-weight models. Irrelevant for API-hosted Claude.

### 2. Multi-User Governance / Access Control
MemGovernance, ACLs, watermarking, multi-tenant isolation, compliance auditing. Somnigraph is single-user. The enterprise governance layer is ~15% of the paper but 0% of our use case.

### 3. Memory Marketplace (MemStore)
Publishing, subscribing, licensing memory units across agents. Business model, not technical contribution. Even in the code, this is minimal infrastructure (API endpoints exist but no marketplace logic).

### 4. MemReader (NL Intent Parsing)
Parsing natural language to determine memory operations. We use explicit tool calls (`remember()`, `recall()`), which is more reliable. MemReader is useful for systems where memory operations are implicit in conversation, but Somnigraph's MCP-based approach is intentionally explicit.

### 5. Task-Aligned Memory Routing
The topic-concept-fact decomposition for query routing. Interesting idea, but Somnigraph's learned reranker already handles query-memory alignment in a data-driven way. Adding a fixed hierarchical decomposition would be a step backward from learned features like `query_coverage`, `query_idf_var`, and `proximity`.

### 6. KV-Cache Acceleration
The TTFT reduction results (up to 94.2%) are genuinely impressive for local deployment. Not applicable to API-hosted models where we have no control over KV-cache.

---

## Connections

| System | Connection to MemOS |
|--------|-------------------|
| **Mem0** | MemOS directly benchmarks against Mem0 and beats it on all four evaluation suites. Both do write-path conflict detection (Mem0's extract-then-update vs. MemOS's MemFeedback), but MemOS's version history is richer. MemOS uses graph storage (Neo4j) while Mem0 uses flat vector store (or optionally a knowledge graph via Mem0g). |
| **Zep/Graphiti** | Both use temporal knowledge graphs. Zep's bi-temporal modeling and structured query resolution are more mature for temporal reasoning; MemOS beats Zep on overall LoCoMo (75.80 vs 59.22) and LongMemEval (77.8 vs 63.8) but Zep outperforms on specific subtasks (single-session assistant: 75.0 vs 67.9 on LongMemEval). |
| **EverMemOS** | Similar OS-inspired framing. EverMemOS uses MongoDB/Milvus/Elasticsearch; MemOS uses Neo4j. EverMemOS has more mature cognitive processing (curiosity-driven exploration, sleep cycle). MemOS has better benchmarks and a public repo. |
| **Kumiho** | Both use Neo4j for graph storage. Kumiho's AGM belief revision and prospective indexing are more theoretically grounded. MemOS's reorganizer (when enabled) does similar relation detection but without the formal belief revision framework. Kumiho reports 0.447 F1 on LoCoMo; MemOS reports 75.80 LLM judge score -- different metrics, hard to compare directly. |
| **HippoRAG** | HippoRAG's PPR-based graph traversal for multi-hop retrieval vs. MemOS's parsed-goal graph recall. Different approaches to graph-conditioned retrieval. MemOS does not use PPR. Somnigraph borrows more from HippoRAG's approach. |
| **MIRIX** | MemOS benchmarks against MIRIX. MIRIX uses six memory components (Core, Episodic, Semantic, Procedural, Resource, Knowledge Vault). MemOS's simpler memory type taxonomy (Working/LongTerm/User + specializations) outperforms MIRIX across all benchmarks. |
| **A-Mem** | A-Mem's Zettelkasten linking and autonomous note organization vs. MemOS's LLM-mediated reorganization. A-Mem achieves 93% fewer tokens on LoCoMo. MemOS is heavier but more comprehensive. |
| **Generative Agents** | MemOS does not implement reflection or consolidation in the Generative Agents sense. The absence of offline processing is a gap that Somnigraph fills with its sleep pipeline. |
| **Memory-R1** | RL-trained memory management. Memory-R1 achieves SOTA on LoCoMo with only 152 training QA pairs. MemOS uses engineering rather than learning for memory management. Neither has a learned reranker in Somnigraph's sense. |

---

## Summary Assessment

MemOS v4 is a substantial improvement over the vision paper we analyzed in February 2026. The earlier paper was pure architecture with zero empirical results; this version delivers working code and SOTA benchmarks across four evaluation suites. The LoCoMo result (75.80) beats every system in their comparison set, and the PERMA results confirm genuine preference-tracking capability, particularly the noise robustness finding (improving with noisy input by retrieving more context). The engineering is solid -- 100% success rate at 100 QPS with lowest latency among all compared systems.

However, there is meaningful gap between the paper's claims and the code's reality. The "unified scheduling across memory types" is mostly plaintext memory with stubs for activation and parametric. The reorganizer (graph structure enrichment) is largely disabled. The lifecycle state machine is partial -- no freeze, no rollback, no TTL enforcement. The "Memory OS" framing oversells what is fundamentally a well-engineered graph-based memory system with good conflict detection and multi-type retrieval. Strip away the OS metaphor and MemOS is: Neo4j graph store + LLM-based memory extraction + write-path conflict detection + hybrid retrieval (vector + graph traversal + BM25) + BGE reranker + preference extraction pipeline.

For Somnigraph, the most actionable takeaway is write-path conflict detection. MemOS's MemFeedback module catches contradictions at insertion time rather than deferring everything to sleep. This does not replace sleep (offline consolidation discovers relationships that write-path checks miss) but provides a first line of defense. The preference extraction pipeline is also worth studying for PERMA -- MemOS's structural distinction between preference and factual memory maps directly to the cross-domain synthesis capability that PERMA evaluates. Somnigraph's learned reranker, explicit feedback loop, biological decay, and offline consolidation remain significant differentiators that no system in MemOS's comparison set possesses. The strongest path is to adopt MemOS's write-path ideas while retaining Somnigraph's retrieval and consolidation strengths.
