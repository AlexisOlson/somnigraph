# Engram Analysis (Agent Output)

*Generated 2026-02-18 by Opus agent reading local clone*

---

## 1. Architecture Overview

### Core Components

Engram is a Go HTTP API server (Chi router) backed by PostgreSQL + pgvector. It implements a **multi-system memory architecture** inspired by cognitive science, with four distinct memory stores plus a meta-layer:

**Memory Systems:**
- **Semantic Memory** (`/v1/memories`) -- Facts, preferences, decisions, constraints ("beliefs"). Each has confidence score, provenance tracking, evidence type, and embedding.
- **Episodic Memory** (`/v1/episodes`) -- Rich experiences preserving raw content, emotional valence/intensity, temporal context, entities, causal links, topics, outcome tracking. Episodes go through consolidation: `raw -> processed -> abstracted -> archived`.
- **Procedural Memory** (`/v1/procedures`) -- Learned trigger-action patterns extracted from successful episodes. Track use count, success rate, versioning.
- **Working Memory** (`/v1/cognitive/session`) -- Active mental workspace with limited capacity (7 slots, Miller's Law). Implements spreading activation.
- **Schemas** (`/v1/schemas`) -- Higher-order mental models derived from clusters of semantic memories.

**Key Services:**
- `ConsolidationService` -- 5-stage background pipeline (every 6 hours)
- `DecayService` -- Competition-aware memory decay (hourly), with tier transitions
- `HybridRecallService` -- Vector + graph traversal for retrieval
- `GraphBuilderService` -- Auto-extracts entities and builds relationship graph
- `MetacognitiveService` -- Self-assessment: confidence, uncertainty detection, strategy reflection
- `ConfidenceService` -- Log-odds based confidence updates (Bayesian-ish)
- `TunerService` -- Auto-adjusts per-type memory policies based on feedback
- `ImplicitFeedbackDetector` -- LLM-based detection of implicit feedback from conversation

**Storage:** PostgreSQL with pgvector. `vector(1536)` columns. IVFFlat indexing. Multi-tenant with API key auth.

---

## 2. Unique Concepts

### 2a. Belief Dynamics with Graded Contradiction

When a new memory is stored, engram checks for semantically similar existing memories (cosine > 0.85) and uses an LLM to classify tension:

| Tension Type | Behavior |
|---|---|
| `none` | Reinforce existing belief (+0.05 confidence) |
| `hard` (score > 0.7) | Penalize old belief (-0.2), create new at 0.7, record contradiction edge |
| `temporal` | Archive old belief, create new at 0.7 (belief evolution) |
| `contextual` | Both coexist (context-dependent truths) |
| `soft` (score < 0.3) | Treat as reinforcement |
| `soft` (score >= 0.3) | Create new without demoting old |

### 2b. Confidence as Tiered Behavior

| Tier | Confidence | Auto-inject | Decay Multiplier |
|---|---|---|---|
| Hot | > 0.85 | Yes | 0.5x (slower) |
| Warm | 0.70-0.85 | No | 1.0x |
| Cold | 0.40-0.70 | No | 1.5x (faster) |
| Archive | < 0.40 | No | 2.0x (fastest) |

### 2c. Competition-Aware Decay

`effective_decay = base_rate * (1 + competition_factor)` where competition_factor is computed from memories with cosine similarity > 0.7 that have *higher* confidence.

### 2d. Knowledge Graph

On every memory creation, GraphBuilderService runs:
1. Entity extraction (LLM)
2. Thematic links (cosine > 0.8)
3. Relationship detection (LLM) -- causal, temporal, contradicts, supports, derived_from, supersedes

### 2e. Spreading Activation in Working Memory

Direct activation + goal-directed bias (1.2x) + schema-directed + temporal + spreading (0.5 decay per hop, max 2) + competition (top 7 win).

### 2f. Implicit Feedback Detection

LLM analyzes conversations for: "contradicted" (-1.0 log-odds), "helpful" (+0.3), "unhelpful", "ignored", "outdated".

### 2g. Consolidation Pipeline (5 stages, every 6 hours)

1. Process Episodes -- LLM extracts entities, topics, causal links
2. Extract Beliefs -- LLM extracts semantic memories from processed episodes
3. Learn Procedures -- LLM extracts trigger-action patterns from successes
4. Form Schemas -- Cluster memories, LLM detects patterns
5. Apply Forgetting -- Decay, archive, merge redundant (cosine > 0.92), prune graph

---

## 3. How Engram Addresses Our 7 Known Gaps

| Gap | Rating | Notes |
|-----|--------|-------|
| 1. Layered Memory | 60% | Episode -> belief -> schema progression is meaningful but lacks intra-layer granularity |
| 2. Multi-Angle Retrieval | 80% | Vector + graph traversal + entity + schema-directed + spreading activation + goal bias |
| 3. Contradiction Detection | 90% | 5-level graded tension system with LLM classification. Only checked at write time, not periodically. |
| 4. Relationship Edges | 95% | Full graph with 8 typed edges, traversal stats, decay, pruning |
| 5. Sleep Process | 75% | Strong pipeline but only processes forward (new episodes). Doesn't re-evaluate old beliefs. |
| 6. Reference Index | 50% | /mind and /health endpoints exist but query full corpus each time |
| 7. Temporal Trajectories | 20% | Raw data exists (mutation log, tier transitions) but no synthesis layer |

---

## 4. Comparison

### Where Engram is Stronger
- Separate stores per type with type-specific behavior
- 5-level graded contradiction handling
- Full graph with 8 typed edges
- 5-stage consolidation pipeline
- Spreading activation with limited slots
- Competition-aware decay with tier-based behavior
- Log-odds Bayesian confidence updates
- Implicit feedback detection + policy auto-tuning
- Schemas (mental models from clusters)
- Procedures as first-class objects with success rates

### Where Our System is Stronger
- Simplicity (one table, one MCP, predictable)
- Designed for single persistent identity
- p10 = pinned, never decays
- Human-in-the-loop curation
- Token budget awareness
- Keyword search (hybrid vector + keyword boost)
- Dedup on write (0.9 cosine)
- Operational simplicity (Supabase hosted)

### Fundamental Difference
Our system = **author-curated memory for a single persistent identity**. Engram = **automated cognitive infrastructure for agent swarms**.

---

## 5. Insights Worth Stealing (Ranked)

| Rank | Insight | Effort | Impact |
|---|---|---|---|
| 1 | Graded contradiction detection (supersedes, temporal evolution) | Medium | High |
| 2 | Consolidation pipeline (merge, synthesize, clean up) | Medium | High |
| 3 | Typed relationship edges (supersedes, supports, related_to) | Low | Medium-High |
| 4 | Mutation log (access/change history per memory) | Low | Medium |
| 5 | Evidence type tracking (explicit vs inferred vs derived) | Low | Medium |
| 6 | Explicit tier boundaries for startup_load/recall | Low | Low-Medium |

---

## 6. What's Not Worth It

- Working Memory / Spreading Activation (overkill for session-based tool)
- Multi-Tenant / Multi-Agent Architecture
- Schema Detection and Mental Models (shallow for single user)
- Procedural Memory as Separate System (our calibration patterns work)
- Implicit Feedback Detection via LLM (expensive, we have explicit corrections)
- Auto-Tuning Policies (fights against intentional curation)
- Emotional Valence / Intensity (noise for our use case)
