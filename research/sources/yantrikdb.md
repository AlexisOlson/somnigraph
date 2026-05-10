# YantrikDB — Analysis

*Generated 2026-04-15 by Opus agent reading local clone*

---

## Repo Overview

**Repo**: https://github.com/yantrikos/yantrikdb-server
**License**: AGPL-3.0 (core engine); MIT (MCP server)
**Language**: Rust (core engine, compiled to `.pyd` via PyO3) + Python (MCP, REST API, agent, eval)
**Description**: "A cognitive memory database engineered for AI agents"
**Author**: Pranab Sarkar (ORCID: 0009-0009-8683-1481)
**Version**: 0.5.11 (hardened alpha)

**Problem addressed**: Memory for AI agents that goes beyond vector retrieval — temporal decay, consolidation, contradiction detection, multi-signal scoring, and proactive triggers. Positioned explicitly against "vector databases are not memory" and "memory frameworks are middleware."

**Core approach**: Rust engine with five coordinated indexes (HNSW vector, graph, temporal, decay heap, KV), exposed via PyO3 bindings. Python layer provides MCP server, FastAPI REST, agent companion framework, and eval harness. SQLite-backed single file per user. Embedding via `all-MiniLM-L6-v2` (384d).

**Maturity**: The Rust engine is distributed as a pre-built `.pyd` — source is not in this repo (closed-source core, open-source shell). The Python layer is well-structured (~2,500 lines across MCP, API, eval, agent) but the actual scoring, consolidation, decay, and graph logic is opaque behind PyO3. The eval harness uses synthetic data only (no external benchmarks). Patent pending (US 19/573,392, March 2026) on "Cognitive Memory Database System with Relevance-Conditioned Scoring."

---

## Architecture

### Design Thesis

Memory is cognition, not storage. The engine performs active maintenance (decay, consolidation, conflict detection, pattern mining) between interactions rather than passively storing and retrieving. Five specialized indexes beat one general-purpose index. The `think()` operation is the core differentiator — an explicit cognitive maintenance loop.

### Technical Stack

| Component | Technology |
|-----------|-----------|
| Core engine | Rust (compiled .pyd, closed source) |
| Vector search | Custom HNSW (M=16, ef=200, 384d) |
| Text search | FTS5 with stemming |
| Graph | In-memory index (entity relationships) |
| Embeddings | `all-MiniLM-L6-v2` (384d, SentenceTransformers) |
| Storage | SQLite (schema v14, 25+ column memories table) |
| MCP | FastMCP (mcp.server.fastmcp) |
| REST API | FastAPI + uvicorn |
| Agent | LLM-driven companion with instinct modules |
| Sync | CRDTs with append-only replication log |
| Encryption | AES-256-GCM |

### Schema (inferred from Python API surface)

Memories carry: rid, text, type (episodic/semantic/procedural), importance, valence, certainty, domain, source, emotional_state, created_at, last_access, consolidation_status, storage_tier, metadata (JSONB), embedding (384d float32).

Entity graph: explicit `relate(source, target, relationship, weight)` calls create edges. Entities are string-named. Graph expansion during recall is togglable (`expand_entities` param).

Conflicts: two-layer detection with `conflict_id`, `conflict_type`, `priority`, `status` (open/resolved/dismissed), `memory_a`, `memory_b`, `entity`, `detection_reason`. Resolution strategies: keep_a, keep_b, keep_both, merge.

### Retrieval Pipeline

Multi-signal scoring with explicit score breakdown returned per result:
- `similarity` — HNSW cosine distance
- `decay` — exponential half-life (importance-gated)
- `recency` — time since last access
- `importance` — stored importance score
- `graph_proximity` — spreading activation from entity matches

The agent summary mentions composite weights of 0.50 sim + 0.20 decay + 0.30 recency with importance gating, but this is in the Rust engine and unverifiable from the Python surface.

Additional retrieval features:
- **Recall-refine**: weighted embedding combination (0.4 original + 0.6 refinement) with RID exclusion for iterative search
- **Retrieval hints**: the engine returns `hints` alongside results (suggested rephrasing, related entities), plus a `confidence` score and `retrieval_summary` (top_similarity, score_spread, sources_used, candidate_count)
- **Recall feedback**: `memory_recall_feedback(rid, feedback, query_text, score_at_retrieval, rank_at_retrieval)` — stores retrieval relevance signals

### Cognitive Maintenance (`think()`)

A single `think()` call runs:
1. Consolidation — agglomerative clustering of similar memories (extractive summary, no LLM)
2. Conflict scan — contradiction detection between entity-related memories
3. Pattern mining — recurring themes across memories
4. Decay checks — flags memories approaching expiry

Returns: triggers (type, reason, urgency, suggested_action), consolidation_count, conflicts_found, patterns_new, patterns_updated, expired_triggers, duration_ms.

---

## Key Features

**Tulving's taxonomy**: Explicit episodic/semantic/procedural memory types with per-type behavior.

**Consolidation**: `find_consolidation_candidates()` + `consolidate()` (extractive, in Rust). Consolidated memories are flagged and excluded from recall by default (`include_consolidated` param).

**Contradiction detection**: Two memories about the same entity with conflicting claims get flagged as conflicts with priority levels. Resolution is manual (keep_a/b/both/merge) with history preserved.

**Proactive triggers**: `check_all_triggers()`, `check_decay_triggers()`, `check_consolidation_triggers()` — the engine surfaces action items without being asked.

**Agent companion layer**: Full conversational agent (`agent/companion.py`) with LLM-driven memory extraction (`agent/learning.py`), instinct modules (check-in, conflict alerting, emotional awareness, follow-up, pattern surfacing, reminders), and voice support. The extraction prompt is well-designed — filters small talk, extracts structured entities, detects open topics.

**Instinct modules** (`agent/instincts/`): check_in, conflict_alerting, emotional_awareness, follow_up, pattern_surfacing, reminder, protocol. These proactively trigger based on memory state — closer to an autonomous agent than a passive server.

**Framework adapters**: LangChain, CrewAI, OpenAI Agents SDK.

**Eval harness**: Synthetic multi-session data with golden queries, measures Recall@K, Precision@K, MRR, per-tag breakdown. Compares multi-signal vs vector-only baseline. CK-5 persona benchmark suite.

---

## What's Novel / Interesting

1. **Importance-gated scoring**: README claims sigmoid suppression of importance contribution at low similarity — prevents high-importance-but-irrelevant memories from surfacing. This is a meaningful design choice (Somnigraph's formula had a similar problem before the reranker).

2. **Retrieval hints**: Returning structured suggestions alongside results ("try rephrasing as X", "related entities: Y, Z") is a good UX pattern for MCP consumers. Somnigraph returns scores but not guidance.

3. **Recall-refine as explicit tool**: Weighted embedding combination for iterative search is cleaner than "just recall again with a different query." The 0.4/0.6 weighting is principled (favor refinement over original).

4. **Confidence + retrieval summary**: Per-query confidence score, top_similarity, score_spread, sources_used, candidate_count — rich metadata for the consumer to decide whether to trust the results.

5. **Agent companion with instincts**: The proactive trigger system (check-in, follow-up, pattern surfacing) goes beyond passive memory into an agentic companion. The instinct modules are small and focused.

6. **Dual deployment**: MCP (MIT) + engine (AGPL) licensing split lets agent frameworks use the MCP tools without AGPL obligations while keeping the core engine copyleft. Pragmatic for adoption.

---

## Limitations / Gaps

1. **Closed-source core**: The Rust engine is distributed as a compiled `.pyd` binary. All scoring, decay, consolidation, graph traversal, and HNSW logic is unverifiable. For a research comparison, we can only observe behavior through the Python API surface, not understand mechanism.

2. **384d embeddings**: `all-MiniLM-L6-v2` is a weak embedding model by 2026 standards. 384 dimensions limits representational capacity. No apparent option to swap in a larger model (dimension is hardcoded in schema).

3. **No external benchmarks**: Eval harness uses only synthetic data (6 sessions, ~30 memories, handful of golden queries). No LoCoMo, no PERMA, no LongMemEval, no comparison with Mem0/MemOS/other systems. Self-reported latency benchmarks (112ms p50 recall) on a small store (1,689 memories).

4. **Patent on retrieval scoring**: US Application 19/573,392 covers "relevance-conditioned scoring" — worth watching for anyone building multi-signal retrieval systems. The AGPL + patent combination is unusually restrictive for an open-source memory system.

5. **Consolidation is extractive-only**: No LLM-mediated merge/summarization. Extractive summaries preserve exact phrases but lose nuance. Contrast with Somnigraph's sleep pipeline which uses LLM consolidation.

6. **Single-threaded lock**: Python layer wraps all engine calls with `threading.Lock()`. The Rust engine may be concurrent internally, but the Python surface is serialized.

7. **Breadth vs depth**: The repo covers memory engine + MCP server + REST API + agent companion + instincts + framework adapters + eval harness + CRDT sync — a lot of surface for a solo dev at v0.5.11. Many features may be thin.

---

## Relevance to Somnigraph

### Architectural Parallels

| Dimension | YantrikDB | Somnigraph |
|-----------|-----------|------------|
| Storage | SQLite (single file) | SQLite + sqlite-vec + FTS5 |
| Vector search | Custom HNSW (Rust, 384d) | sqlite-vec (float32) |
| Text search | FTS5 | FTS5 with BM25 |
| Scoring | Multi-signal composite (fixed weights) | RRF fusion → LightGBM reranker (26 features) |
| Decay | Exponential half-life (importance-gated) | Biological decay (base_importance × power-law) |
| Consolidation | Extractive agglomerative (Rust) | LLM-mediated sleep pipeline (NREM/REM) |
| Contradiction | Entity-level conflict detection + resolution | Contradiction edges in graph |
| Graph | In-memory entity graph with spreading activation | sqlite-vec graph with PPR traversal |
| Feedback | recall_feedback with score/rank context | recall_feedback with EWMA + UCB exploration |
| Reranker | None (fixed formula) | LightGBM pointwise (26 features, 1032q training) |
| Benchmarks | Synthetic eval only | LoCoMo (85.1% QA, 95.4% R@10), real-data GT |
| Embedding | all-MiniLM-L6-v2 (384d) | text-embedding-3-small (1536d) |
| License | AGPL-3.0 + patent | MIT |

### Key Contrasts

1. **Closed vs open core**: Somnigraph's scoring, decay, and retrieval logic is fully readable; YantrikDB's is compiled Rust. This makes YantrikDB unsuitable as a research reference for mechanism design, only for interface design.

2. **Learned vs hand-tuned scoring**: Somnigraph moved from a hand-tuned formula to a learned reranker with real-data ground truth. YantrikDB uses fixed composite weights (0.50/0.20/0.30). The reranker is Somnigraph's key differentiator.

3. **External vs synthetic evaluation**: Somnigraph has LoCoMo end-to-end results beating published systems. YantrikDB has no external benchmark comparison.

4. **Sleep vs think**: Somnigraph's consolidation is a multi-phase offline pipeline (NREM + REM sleep). YantrikDB's `think()` is a single synchronous call. The depth of consolidation differs substantially.

### Borrowable Ideas

- **Retrieval hints**: Structured guidance alongside results is a good MCP UX pattern. Low implementation cost, could improve consumer experience.
- **Recall-refine with weighted embedding**: Cleaner than repeated recall for iterative search. Could complement Somnigraph's existing approach.
- **Confidence scoring + retrieval summary metadata**: Exposing candidate_count, score_spread, top_similarity gives consumers signal about result quality.

---

## Verdict

**Relevance: Medium.** Heavy architectural overlap (SQLite, multi-signal scoring, decay, consolidation, contradiction detection, graph, feedback loop) makes it a natural comparison point for `similar-systems.md`. However, the closed-source Rust core limits research value — we can see *what* it does but not *how*. The lack of external benchmarks means no apples-to-apples comparison. Interface design (hints, recall-refine, confidence metadata) is the most transferable contribution. The AGPL + patent licensing posture is worth noting as a counterpoint to Somnigraph's MIT approach.
