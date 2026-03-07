# Daem0n-MCP — Source Analysis

*Phase 13, 2026-02-27. Analysis of [DasBluEyedDevil/Daem0n-MCP](https://github.com/DasBluEyedDevil/Daem0n-MCP).*

Daem0n-MCP is an AI memory and decision system built by u/DasBlueEyedDevil. Currently at v6.6.6 with ~350 commits, 55 stars, 8 forks, and 500+ tests. MIT license. It targets Claude Code and OpenCode as primary clients, with a theatrical "Sacred Covenant" ritual theme layered over genuinely substantive infrastructure.

## 1. Architecture Overview

**Language:** Python (async throughout, FastMCP 3.0 framework)

**Key dependencies:**
- FastMCP 3.0 (MCP server framework with middleware)
- SQLite (primary persistence, async via SQLAlchemy)
- Qdrant (vector database backend — local or remote)
- `nomic-ai/modernbert-embed-base` (256-dim embeddings, Matryoshka truncation, ONNX quantized)
- NetworkX + `leidenalg` (knowledge graph + community detection)
- sentence-transformers (embedding model management)
- LLMLingua-2 (context compression)
- tree-sitter (multi-language AST parsing)
- E2B (Firecracker microVM sandboxing)
- tiktoken (token counting)
- Pydantic Settings (config, ~50 env vars)
- Optional: OpenTelemetry, watchdog, plyer, LangGraph

**Storage layout:**
```
<project_root>/.daem0nmcp/
  storage/
    daem0nmcp.db          # SQLite: memories, rules, relationships, entities, communities, versions, etc.
  qdrant/                 # Local Qdrant vector storage (or remote via DAEM0NMCP_QDRANT_URL)
```

Per-project isolated storage — each project gets its own `.daem0nmcp/` directory. Cross-project awareness via `ProjectLink` (federated read access, write isolation).

**File structure (key source files):**

| File | Purpose |
|------|---------|
| `server.py` | MCP server entry — 8 workflow + 3 cognitive tools |
| `memory.py` | Core storage & retrieval pipeline (remember, recall, decay, conflict detection) |
| `models.py` | 10+ SQLAlchemy tables: Memory, Fact, Rule, MemoryVersion, MemoryRelationship, CodeEntity, etc. |
| `vectors.py` | ModernBERT embeddings — asymmetric encoding, ONNX backend, Matryoshka truncation to 256-dim |
| `bm25_index.py` | BM25 Okapi keyword retrieval |
| `fusion.py` | RRF hybrid search combining BM25 + vector |
| `similarity.py` | TF-IDF index (legacy), decay formula, conflict detection |
| `surprise.py` | Titans-inspired novelty scoring (cosine distance to k-nearest neighbors) |
| `recall_planner.py` | TiMem-style complexity-aware retrieval (simple/medium/complex routing) |
| `retrieval_router.py` | Auto-Zoom: query-aware dispatch to vector-only / hybrid / GraphRAG |
| `query_classifier.py` | Exemplar-based complexity classification |
| `graph/knowledge_graph.py` | Dual-storage KG (NetworkX in-memory + SQLite persistence) |
| `graph/temporal.py` | Bi-temporal versioning (valid_time vs transaction_time) |
| `graph/contradiction.py` | Contradiction detection via similarity + negation patterns |
| `graph/traversal.py` | Multi-hop traversal, causal chain tracing |
| `graph/leiden.py` | Leiden community detection wrapper |
| `communities.py` | Community management, hierarchical summarization |
| `dreaming/scheduler.py` | IdleDreamScheduler — idle detection + cooperative scheduling |
| `dreaming/strategies.py` | FailedDecisionReview, ConnectionDiscovery, CommunityRefresh, PendingOutcomeResolver |
| `dreaming/persistence.py` | Dream session/result models |
| `cognitive/simulate.py` | Temporal Scrying — replay past decisions with current knowledge |
| `cognitive/evolve.py` | Rule entropy analysis (staleness detection) |
| `cognitive/debate.py` | Adversarial Council — structured evidence debate |
| `reflexion/nodes.py` | Actor-Evaluator-Reflector metacognitive loop |
| `compression/compressor.py` | LLMLingua-2 integration with code entity preservation |
| `active_context.py` | MemGPT-style always-hot working memory (max 10 items) |
| `workflows/` (8 files) | Consolidated workflow tools (commune, consult, inscribe, reflect, understand, govern, explore, maintain) |

## 2. Memory Type Implementation

**Schema:** The core unit is `Memory` with fields:
- `id`, `project_path` (project isolation)
- `category`: one of `decision`, `pattern`, `warning`, `learning`
- `content`, `rationale`, `context` (structured JSON)
- `tags` (list), `file_path` (file association)
- `is_permanent` (patterns/warnings are permanent; decisions/learnings decay)
- `worked` (bool, nullable), `outcome` (text)
- `importance_score` (EWC-inspired protection), `surprise_score` (novelty)
- `recall_count` (saliency tracking)
- `source_client`, `source_model` (provenance)
- `created_at`, `happened_at` (temporal — when happened vs when recorded)
- Vector embedding stored as packed bytes in SQLite + indexed in Qdrant

**Secondary models:**
- `Fact`: Immutable verified knowledge promoted after N successful outcomes (default 3). Content-hash O(1) lookup, not semantic search.
- `MemoryVersion`: Bi-temporal versioning — `changed_at` (transaction time), `valid_from`/`valid_to` (valid time). Supports point-in-time queries.
- `MemoryRelationship`: Graph edges between memories — `led_to`, `supersedes`, `depends_on`, `conflicts_with`, `related_to`.
- `ActiveContextItem`: MemGPT-style hot memories (max 10, priority-ordered, time-expiring).
- `MemoryCommunity`: Leiden-detected clusters with AI-generated summaries.

**Types/categories:** Four categories with distinct behavior:
- `decision` — Episodic, decays (30-day half-life), tracks outcomes (worked/failed), failed get 1.5x boost
- `pattern` — Semantic, permanent (no decay), represents project-level knowledge
- `warning` — Semantic, permanent, 1.2x relevance boost
- `learning` — Episodic, decays, includes dream-generated insights

**Extraction:** Primarily manual via `inscribe(action="remember")`. Auto-capture via:
- Claude Code stop hook (auto-extracts decisions from conversation)
- Post-edit hook (suggests remembrance for significant changes)
- Dream strategies (generate `learning` memories autonomously)
- Ingest action (import external documentation from URL)

## 3. Retrieval Mechanism

Multi-stage pipeline with adaptive routing:

1. **Query Classification** (Auto-Zoom): Exemplar-based classifier categorizes query as SIMPLE/MEDIUM/COMPLEX with confidence score.

2. **Strategy Routing:**
   - SIMPLE → Vector-only search (fast path via Qdrant)
   - MEDIUM → Hybrid BM25 + vector with RRF fusion
   - COMPLEX → GraphRAG multi-hop traversal + community summaries
   - Below confidence threshold → fallback to hybrid
   - Shadow mode (default) logs classifications without routing

3. **Hybrid Search (default path):**
   - BM25 Okapi (k1=1.5, b=0.75) for keyword matching
   - ModernBERT vector similarity (256-dim, asymmetric query/doc encoding)
   - RRF fusion: `score(d) = sum(1 / (k + rank(d)))` where k=60
   - Configurable weight: `final = (1 - 0.3) * tfidf + 0.3 * vector` (legacy path also maintains TF-IDF)

4. **Post-retrieval scoring:**
   - Exponential decay: `weight = e^(-lambda*t)`, lambda = ln(2)/30 days. Permanent categories exempt.
   - Failed decision boost: 1.5x
   - Warning boost: 1.2x
   - Surprise score applied (0.0-1.0)
   - Importance score (EWC-inspired memory protection)

5. **Result enhancement:**
   - LLMLingua-2 compression at tiered thresholds (4K soft, 8K hard, 16K emergency)
   - Code entity preservation during compression
   - Condensed mode: content truncated to 150 chars, 50-75% token reduction
   - Diversity filtering: max 3 results per file

6. **Hierarchical access** (for COMPLEX queries):
   - Level 3: Community summaries (broad overview)
   - Level 2: Community members
   - Level 1: Raw memories

## 4. Standout Feature

**Idle Dreaming with Multiple Strategies**

The most novel capability is the background dreaming system — a cooperative async process that activates during user idle periods to autonomously maintain and improve the knowledge base.

**Architecture:**
- `IdleDreamScheduler` monitors tool call timestamps via `notify_tool_call()`
- After configurable idle timeout (default 60s), triggers registered dream strategies
- Yields immediately when user returns (`asyncio.Event` signaling)
- Each strategy checks `scheduler.user_active` at yield points for cooperative interruption

**Four dream strategies:**

1. **FailedDecisionReview**: Queries `worked=False` decisions older than threshold, gathers current evidence via semantic search, classifies as `revised` / `confirmed_failure` / `needs_more_data`, persists insights as `learning` memories with `dream` tag and full provenance (decision ID, evidence IDs, timestamps). Cooldown prevents re-reviewing recently processed decisions.

2. **ConnectionDiscovery**: Scans entity references within a lookback window, groups memories by shared entities, creates `related_to` edges for implicitly connected memories. Enforces minimum shared entity count and maximum connections per session.

3. **CommunityRefresh**: Monitors memory creation since last Leiden detection run, triggers rebuild when staleness threshold exceeded. Keeps community summaries current.

4. **PendingOutcomeResolver**: Auto-resolves pending decisions using evidence scoring. Ships with `dry_run=True` by default — auto-resolution only when explicitly enabled. Decision tree: insufficient evidence / flagged for review / auto-resolved success or failure.

**What makes this notable:** This is a genuine *background consolidation* process rather than a batch job triggered externally. It runs within the MCP server's event loop and cooperatively yields, which means the system gets smarter while the user is thinking. The ConnectionDiscovery strategy (auto-discovering implicit edges) and CommunityRefresh (keeping graph clusters current) address the same "sleep" concerns our system handles, but in a different temporal pattern — continuous idle-time vs periodic batch.

## 5. Other Notable Features

**Bi-Temporal Knowledge Graph**: The `MemoryVersion` model implements proper bi-temporal tracking — `changed_at` (when the system learned it) vs `valid_from`/`valid_to` (when it was true in reality). `get_versions_at_time()` does dual filtering: "fact was true at query time AND we knew about it at query time." Contradiction resolution sets `valid_to` without deletion, preserving audit trail via `invalidated_by_version_id`. This is genuinely more complete than most MCP memory systems.

**Titans-Inspired Surprise Scoring**: `calculate_surprise()` computes cosine distance to k-nearest neighbors (default k=5), normalized to [0.0, 1.0]. Novel content gets higher scores, influencing retrieval ranking. Formula: average cosine distance to k-nearest existing embeddings. Not the full Titans surprise metric (which involves gradient-based gating in the neural network), but a reasonable approximation adapted for a retrieval system.

**Metacognitive Reflexion Loop**: Actor-Evaluator-Reflector pattern with max 3 iterations and quality threshold of 0.8. Evaluator extracts claims, verifies against stored knowledge, applies scoring: base verification confidence minus penalties for conflicts (-0.2 each) and unverified claims (-0.05 each), plus bonuses for passing code assertions (+0.1). This is used for the `verify` action.

**Cognitive Tools**: Three standalone meta-reasoning tools — `simulate_decision` (replay past decisions with current knowledge), `evolve_rule` (analyze rule staleness/entropy), `debate_internal` (structured adversarial debate with convergence detection). These represent an unusual level of self-inspection capability.

**Fact Promotion**: Memories automatically promote to immutable `Fact` records after N successful outcomes (default 3). Facts use content-hash O(1) lookup rather than semantic search, creating a fast path for verified knowledge.

## 6. Gap Ratings

| Gap | Rating | Evidence |
|-----|--------|----------|
| Layered Memory | 75% | Four categories (decision/pattern/warning/learning) + Fact model + ActiveContext (hot) + Community summaries (cold). Not as granular as episodic/semantic/procedural/meta but functionally layered with distinct decay/permanence rules per type. |
| Multi-Angle Retrieval | 85% | Auto-Zoom routes SIMPLE/MEDIUM/COMPLEX queries to different strategies. BM25 + vector + GraphRAG multi-hop. File recall, entity recall, hierarchical recall, community drill-down. Strong multi-angle coverage. |
| Contradiction Detection | 70% | Similarity threshold (0.75) + negation pattern matching (~30 pattern pairs). Sets `valid_to` without deletion. Not LLM-based — relies on heuristic negation detection which will miss semantic contradictions that don't use explicit negation words. But preservation via temporal invalidation is clean. |
| Relationship Edges | 80% | Five explicit edge types (`led_to`, `supersedes`, `depends_on`, `conflicts_with`, `related_to`). Auto-discovery via ConnectionDiscovery dream strategy. Multi-hop traversal, causal chain tracing. Edges are in the retrieval path (GraphRAG). Stronger than most systems. |
| Sleep Process | 70% | Four dream strategies running during idle time. FailedDecisionReview + ConnectionDiscovery + CommunityRefresh + PendingOutcomeResolver. Cooperative scheduling with user-return yielding. However: no dedicated offline batch consolidation, no embedding reprocessing, no cluster merging at depth. More "nap" than "deep sleep" — lightweight maintenance rather than architectural restructuring. |
| Reference Index | 40% | Code entities indexed via tree-sitter (10+ languages). Entity extraction from memory content. File associations tracked. But no structured reference index for external sources or citation tracking. `ingest` imports URLs but doesn't maintain provenance chains. |
| Temporal Trajectories | 80% | Full bi-temporal versioning (valid_time + transaction_time). `trace_knowledge_evolution()` reconstructs entity change history. `at_time` point-in-time queries. `versions` action for memory version history. One of the strongest temporal implementations in the survey. |
| Confidence/UQ | 45% | `importance_score` (EWC-inspired) and `surprise_score` (novelty) provide partial confidence signals. Fact promotion after N outcomes is a crude confidence threshold. `worked` boolean tracks outcome but not confidence gradient. No explicit uncertainty quantification, no Bayesian updating, no confidence compounding from multiple feedback signals. |

## 7. Comparison with claude-memory

**Stronger:**
- **Richer retrieval routing**: Auto-Zoom with three strategies (vector-only / hybrid / GraphRAG) vs our single RRF path. Query complexity classification is novel.
- **Background consolidation**: Idle dreaming runs within the server event loop — no external scripts needed. Four strategies vs our three sleep pipelines (NREM standard/deep, REM). Theirs is continuous; ours is batch.
- **Bi-temporal versioning**: Proper valid_time vs transaction_time with point-in-time queries. We have `created_at` but not dual-temporal versioning.
- **Fact promotion**: Auto-promoting stable knowledge to O(1) lookup. We don't distinguish between verified and unverified knowledge at the storage level.
- **Edge auto-discovery**: ConnectionDiscovery dream strategy finds implicit relationships via shared entities. Our edges are manually created during sleep or explicit linking.
- **Code understanding integration**: Tree-sitter AST parsing across 10+ languages, impact analysis, refactor suggestions. We have no code-awareness layer.
- **Tool consolidation**: 8 workflow tools covering 59 actions vs our 10+ individual tools. Better for context overhead.
- **Cognitive meta-tools**: simulate_decision, evolve_rule, debate_internal. Genuine metacognitive capabilities we don't have.
- **ModernBERT with asymmetric encoding**: Query-specific vs document-specific prefixes. We use OpenAI text-embedding-3-small with symmetric encoding.
- **LLMLingua-2 compression**: Integrated context compression with code entity preservation. We don't compress recall output.

**Weaker:**
- **No human-in-the-loop curation**: No equivalent to our `review_pending()` / pending memory workflow. Their auto-capture is fully automated (stop hook extracts decisions). No deliberate human review step for memory quality control.
- **No dimensional feedback**: Our `recall_feedback()` supports utility + durability scoring per memory. They have binary `worked`/`failed` outcomes and `recall_count`, but no fine-grained feedback signals.
- **No confidence gradient**: Our confidence compounds through repeated feedback, decays through contradiction, and influences retrieval scoring. They have flat `importance_score` without compounding dynamics.
- **Simpler decay model**: Their exponential decay with 30-day half-life and binary permanent/decaying categories is less nuanced than our power-law decay with per-memory configurable rates.
- **No gestalt layer**: We have startup_load with curated priority ordering. Their `commune(action="briefing")` provides session start context but without the same level of curation.
- **No shadow load / abuse detection**: We penalize over-recalled memories via quadratic shadow penalty. They boost recall_count but don't penalize it.
- **Edge types less expressive**: Five edge types vs our flags-based system (contradiction/revision/derivation) with linking_context and linking_embedding. Our novelty-scored edge expansion is more sophisticated for retrieval.
- **No sleep depth**: Their dreaming is lightweight idle-time maintenance. Our NREM deep and REM pipelines do structural consolidation (cross-topic enrichment, taxonomy assignment, question-driven reflection) that their system doesn't attempt.
- **No embedding-level edge features**: Our edges have `linking_embedding` and `features` for novelty scoring during expansion. Their edges are plain relationship records.
- **Per-project isolation is rigid**: Each project gets separate storage. Our single-database model with project-agnostic memories is simpler for cross-domain knowledge. Their `ProjectLink` federation adds complexity for what we get free.

## 8. Insights Worth Stealing

1. **Auto-Zoom retrieval routing** (effort: medium, impact: high). Classifying query complexity and routing to appropriate strategies (vector-only for simple, hybrid for medium, graph expansion for complex) is a smart optimization. We could classify queries and skip edge expansion for simple factual lookups while going deep for complex multi-topic queries. The exemplar-based classifier approach is lightweight.

2. **Idle-time ConnectionDiscovery** (effort: low, impact: medium). Auto-discovering implicit edges between memories that share extracted entities is a pattern we could fold into our NREM pipeline. We already have entity extraction via our taxonomy — scanning for co-occurring entities to propose edges would strengthen our graph without LLM calls.

3. **Bi-temporal point-in-time queries** (effort: medium, impact: medium). Adding `valid_from`/`valid_to` to memories (separate from `created_at`) would let us answer "what did I know at time T?" and properly track when facts were true vs when we learned them. Particularly useful for long-running projects where knowledge evolves.

4. **Fact promotion from stable memories** (effort: low, impact: low-medium). Auto-promoting memories with N confirmed outcomes to an immutable fast-lookup tier is a clean pattern. Could be a flag on our memories rather than a separate table — memories with high confidence that have been confirmed multiple times get indexed differently.

5. **LLMLingua-2 compression on recall output** (effort: medium, impact: medium). Compressing recall results before returning them would reduce context consumption, especially for large retrievals. Their tiered thresholds (4K/8K/16K) are pragmatic. The code entity preservation during compression is a nice touch.

6. **PendingOutcomeResolver with evidence scoring** (effort: low, impact: low). Auto-resolving pending decisions based on accumulated evidence is something we could do during sleep. Their decision tree (insufficient_evidence / flagged_for_review / auto_resolved) with dry_run default is appropriately cautious.

## 9. What's Not Worth It

**Sacred Covenant enforcement system**: The ritual-themed preflight token system (5-minute validity, `COMMUNION_REQUIRED`/`COUNSEL_REQUIRED` blocking) adds ceremony without proportional value. It's a creative way to force agents to check context before acting, but the same result is achievable with simpler hook patterns. The theatrical framing may help with agent compliance via prompt engineering but adds real code complexity.

**MemGPT-style Active Context (max 10 items)**: Maintaining a separate always-hot working memory buffer is solving a problem we handle differently — our `startup_load()` with priority ordering already provides session-start context, and our memory system's organic recall handles the rest. A fixed 10-item buffer is both too rigid and too small.

**Cognitive meta-tools (simulate_decision, evolve_rule, debate_internal)**: Clever but these feel like they'd rarely be invoked in practice. Simulating past decisions requires enough context to be useful, and adversarial debate is expensive for what it produces. Rule entropy analysis is interesting but assumes a heavy rule usage pattern we don't follow.

**Per-project storage isolation + ProjectLink federation**: Our single-database model is simpler and more powerful for cross-domain knowledge. Their approach requires explicit federation links to read across projects, adding indirection for a use case (cross-project awareness) that should be natural.

**Qdrant as a required dependency**: Adding an external vector database creates operational complexity vs our sqlite-vec approach. The performance difference only matters at scales most individual developers won't reach. Their migration story (switching from MiniLM to ModernBERT requiring full re-encoding) illustrates the brittleness.

## 10. Key Takeaway

Daem0n-MCP is the most feature-rich MCP memory system in the survey — its scope is genuinely impressive, with 350+ commits, 59 workflow actions, and deep integration across retrieval routing, graph reasoning, temporal versioning, background consolidation, and code understanding. The standout insight for our system is the **idle-time background processing pattern** (continuous consolidation vs our batch sleep), the **query complexity routing** (not all queries need the same retrieval strategy), and the **bi-temporal versioning** (properly separating "when it happened" from "when we knew"). However, it trades off human-in-the-loop curation, feedback sophistication, and memory-level confidence tracking — areas where our system is stronger. The most actionable steal is Auto-Zoom retrieval routing: classifying query complexity and skipping expensive graph expansion for simple lookups would reduce our recall latency without sacrificing depth when it matters.
