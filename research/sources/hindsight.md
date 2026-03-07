# Hindsight Analysis (Agent Output)

*Generated 2026-02-18 by Opus agent reading local clone*

---

## 1. Architecture Overview

### Core Components

Hindsight is a monorepo: FastAPI server (hindsight-api), Next.js control plane, Rust CLI, generated SDKs. PostgreSQL + pgvector backend.

**Key tables:**
- `banks` -- Isolated memory stores with disposition traits (skepticism, literalism, empathy 1-5), mission statement, background context
- `memory_units` -- All facts/observations. Fields: text, fact_type (world/experience/observation), embedding (dense), search_vector (sparse/BM25), temporal fields (occurred_start/end, mentioned_at), tags, proof_count, source_memory_ids, history (JSONB audit trail), consolidated_at, document_id, chunk_id
- `entities` -- Canonical entity records
- `entity_links` -- Links between memory_units and entities
- `entity_cooccurrences` -- For disambiguation
- `memory_links` -- Typed edges: semantic, temporal, entity, causes, caused_by, enables, prevents
- `mental_models` -- User-curated summaries with tags and refresh triggers
- `documents` -- Source document tracking with chunks

### Three Core Operations

**Retain** (ingest): LLM fact extraction -> embeddings -> dedup -> insert -> entity resolution (spaCy + co-occurrence) -> create temporal/semantic/entity/causal links -> background consolidation -> mental model refresh

**Recall** (retrieval): Query analysis -> four parallel strategies (semantic, BM25, graph/MPFP, temporal) -> RRF fusion -> cross-encoder reranking -> token budget trimming

**Reflect** (reasoning): Agentic tool-use loop (up to 10 iterations) with tools: search_mental_models, search_observations, recall, expand, done. Disposition-shaped reasoning. Citation validation.

### Consolidation Engine

After every retain, for each unconsolidated memory:
1. Find related observations via full recall system
2. LLM decides: create new observation, update existing, or skip (ephemeral)
3. Create: generate embedding, insert with tags from source
4. Update: merge tags (union), update text with temporal narrative, preserve history audit trail, update temporal range
5. Mark consolidated_at = NOW()
6. Trigger mental model refreshes for matching tags

---

## 2. Unique Concepts

### A. Biomimetic Memory Hierarchy
- **World facts** -- objective external knowledge
- **Experience facts** -- first-person experiential memory
- **Observations** -- synthesized knowledge from multiple facts (analogous to beliefs/understanding)

### B. Multi-Path Forward Push (MPFP)
Novel graph traversal combining meta-path patterns from HIN literature with Forward Push from APPR:
- Sublinear in graph size
- Lazy edge loading
- Predefined patterns: `[semantic, semantic]` (topic expansion), `[entity, temporal]` (entity timeline), `[semantic, causes]` (forward reasoning), `[semantic, caused_by]` (backward reasoning), `[temporal, semantic]` (what was happening then)
- All patterns run in parallel, fused via RRF

### C. Typed Edge Graph
Five edge types: semantic, temporal, entity, causes/caused_by, enables/prevents. Causal links get 2x boost during traversal.

### D. Consolidation as Background "Dreaming"
- Distinguishes durable knowledge from ephemeral state
- Tracks temporal evolution via history audit trails
- Handles contradictions by creating temporal narratives ("used to X, now Y")
- Mission-oriented consolidation

### E. Four-Way Parallel Retrieval with Neural Reranking
Semantic + BM25 + graph + temporal, merged via RRF, reranked with cross-encoder.

---

## 3. How Hindsight Addresses Our 7 Known Gaps

| Gap | Assessment |
|-----|-----------|
| 1. Layered Memory | Partially -- Mental Models (manual) + Observations (auto) + Raw facts. No automatic gestalt generation. |
| 2. Multi-Angle Retrieval | **Strongest of all systems reviewed.** Four-way parallel (TEMPR) with MPFP graph traversal + RRF + cross-encoder. |
| 3. Contradiction Detection | At consolidation time only. Creates temporal narratives, preserves history. Not at retrieval time. |
| 4. Relationship Edges | **Strongly addressed.** 5 typed edges, created at retain time, actively traversed during recall. |
| 5. Sleep Process | Partially -- auto-consolidation after every retain. But only processes new facts, no periodic full-corpus sweep. |
| 6. Reference Index | **Not addressed.** No lightweight overview mechanism. |
| 7. Temporal Trajectories | Partially -- history JSONB, temporal markers in observations, date ranges. But no first-class trajectory objects. |

---

## 4. Comparison

### Where Hindsight is Stronger
- Retrieval diversity (4-way parallel + RRF + cross-encoder reranking)
- Relationship edges (5 typed, actively traversed)
- Graph traversal (MPFP with meta-path patterns)
- Fact extraction (LLM decomposes into atomic facts with entities, temporal, causal)
- Consolidation (automatic after every retain with contradiction handling)
- Temporal reasoning (date parsing, range queries, temporal spreading)
- Entity resolution (spaCy NER + co-occurrence disambiguation)
- Causal chains (explicit links extracted and traversed)
- Agentic reasoning (reflect with tool-use loop)
- Cross-encoder reranking

### Where Our System is Stronger
- Priority system (1-10 with p10 pinned, time decay, reheat)
- Memory categories (more nuanced taxonomy)
- Soft deletes + access tracking
- Token budget control (startup_load + recall)
- Simplicity
- Identity/continuity focus
- Cost (no LLM calls for memory ops)
- Startup context (startup_load)

### Neither is Strong At
- Layered gestalt summaries
- Explicit temporal trajectories
- Reference index
- Real-time contradiction detection at retrieval time

---

## 5. Insights Worth Stealing (Ranked)

### A. Multi-Strategy Retrieval with RRF Fusion (HIGH)
Add BM25 as separate path, merge via RRF `score = sum(1/(k+rank))`. k=60 default. Pure win.

### B. Typed Relationship Edges (HIGH)
Minimum viable: semantic links (top-3 similar at remember time), temporal links (same session/day), theme links (shared themes array).

### C. Automatic Observation Consolidation (HIGH)
LLM consolidates related facts into durable observations with:
- Durable knowledge vs ephemeral state distinction
- "Used to X, now Y" temporal narrative for contradictions
- History audit trail with reason, timestamp, source ID
- Proof count for confidence signal

### D. Cross-Encoder Reranking (MEDIUM)
ms-marco-MiniLM-L-6-v2 after initial retrieval. Adds latency + dependency.

### E. Fact Decomposition at Ingest Time (MEDIUM)
Lighter version: extract entities and temporal references at remember() time as structured metadata.

### F. Mission-Oriented Consolidation (MEDIUM)
Consolidation shaped by purpose/mission of the memory system.

### G. Meta-Path Patterns for Graph Traversal (LOW, until we have edges)
Define retrieval intents as sequences of edge types.

---

## 6. What's Not Worth It

- Disposition Traits (skepticism/literalism/empathy scales)
- Memory Banks as isolation units
- Full LLM-based fact extraction on every retain
- Tag-based security boundaries
- Full reflect agentic loop (redundant with Claude itself)
- Local embedding/reranking models
