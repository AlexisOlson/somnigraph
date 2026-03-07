# PLTM-Claude -- Source Analysis

*Phase 13, 2026-02-27. Analysis of Alby2007/PLTM-Claude.*

## 1. Architecture Overview

**Language:** Python 3.11
**Key dependencies:**
- `mcp` (Model Context Protocol SDK)
- `aiosqlite` (async SQLite)
- `sentence-transformers` (all-MiniLM-L6-v2, 384-dim embeddings)
- `numpy` (vector ops, cosine similarity)
- `pydantic` (data models)
- `loguru` (logging)
- `torch` / `transformers` (for embedding model; lite mode available without torch)
- Optional: Groq, DeepSeek, Ollama for LLM-powered ingestion/fact-checking

**Storage layout:**
```
data/pltm_mcp.db          # Single SQLite database (WAL mode)
  typed_memories           # 4-type cognitive memory store
  memory_embeddings        # 384-dim vectors as binary blobs
  knowledge_nodes          # Knowledge graph nodes
  knowledge_edges          # Knowledge graph edges
  prediction_book          # Epistemic claim tracking
  calibration_cache        # Domain accuracy stats
  epistemic_interventions  # Confidence adjustments
  personality_snapshots    # Communication style evolution
  confabulation_log        # Reasoning failure tracking
  meta_judge_events        # Jury performance metrics
  session_history          # Cross-conversation continuity
```

**File structure (key source files):**
```
mcp_server/
  pltm_server.py           # 136 MCP tools, dispatch + handlers
  handlers/
    memory_handlers.py      # Typed memory CRUD
    intelligence_handlers.py # Advanced features
    registry.py             # Tool registration
src/
  core/
    models.py               # MemoryAtom, AtomType, GraphType, JuryDecision (Pydantic)
    decay.py                # DecayEngine -- Ebbinghaus forgetting curve
    ontology.py             # Type-specific rules, predicates, decay rates
    retrieval.py            # MemoryRetriever with reconsolidation
  memory/
    memory_types.py         # TypedMemory, TypedMemoryStore (4-type system)
    embedding_store.py      # EmbeddingStore (brute-force cosine over blobs)
    knowledge_graph.py      # KnowledgeNetworkGraph (nodes + edges in SQLite)
    attention_retrieval.py  # Attention-weighted retrieval (keyword-based)
    memory_intelligence.py  # 11 capabilities: decay, consolidation, clustering, etc.
    memory_jury.cpython-311-darwin.so  # Compiled -- source not readable
    memory_pipeline.cpython-311-darwin.so  # Compiled -- source not readable
  jury/
    base_judge.py           # Abstract base (rule-based, not LLM)
    memory_judge.py         # Ontology compliance + semantic validity
    safety_judge.py         # PII detection, harmful content filtering
    time_judge.py           # Temporal consistency
    consensus_judge.py      # Weighted voting aggregation
    orchestrator.py         # 4-judge pipeline with safety veto
  analysis/
    epistemic_monitor.py    # Claim tracking, calibration curves, confabulation
    fact_checker.py         # External verification
    data_ingestion.py       # URL/arxiv/wikipedia/RSS/file ingestion
```

Note: Several critical files (`memory_jury`, `memory_pipeline`, `session_continuity`, `phi_rms`) are shipped as compiled `.so` binaries, meaning their internals cannot be verified from source.

## 2. Memory Type Implementation

**Schema:** Two parallel memory systems coexist:

1. **TypedMemory** (cognitive layer) -- `typed_memories` table:
   - `id`, `memory_type` (episodic/semantic/belief/procedural), `user_id`
   - `content`, `context`, `source` (user_stated/inferred/observed)
   - `strength` (0-1), `created_at`, `last_accessed`, `access_count`
   - `confidence` (0-1), `evidence_for[]`, `evidence_against[]` (belief-specific)
   - `episode_timestamp`, `participants[]`, `emotional_valence` (episodic-specific)
   - `trigger`, `action`, `success_count`, `failure_count` (procedural-specific)
   - `consolidated_from[]`, `consolidation_count`, `tags[]`

2. **MemoryAtom** (knowledge graph layer) -- Pydantic model:
   - Semantic triple: `subject`, `predicate`, `object`
   - `atom_type`: 15+ types (entity, affiliation, social, preference, belief, skill, personality_trait, communication_style, state, event, hypothesis, invariant, fact, attribute, sensory_observation)
   - `graph`: substantiated / unsubstantiated / historical
   - `provenance`: user_stated / user_confirmed / inferred / corrected
   - `confidence`, `strength`, `last_accessed`, `access_count`
   - Epistemic modeling: `belief_holder`, `epistemic_distance` (0=direct, 1=reported)
   - Evidence: `assertion_count`, `explicit_confirms[]`, `last_contradicted`
   - Conflict resolution: `supersedes`, `superseded_by`, `related_atoms[]`
   - `jury_history[]` -- full audit trail of jury decisions

**Types/categories:**
- 4 cognitive types (TypedMemory): episodic, semantic, belief, procedural
- 15+ ontological types (MemoryAtom): entity, affiliation, social, preference, belief, skill, personality_trait, communication_style, interaction_pattern, state, event, hypothesis, invariant, fact, attribute

**Extraction:** The `process_conversation` tool runs a "3-lane pipeline" that auto-extracts memories. Specific tools (`store_episodic`, `store_semantic`, `store_belief`, `store_procedural`) also allow explicit storage. The extraction pipeline code is in a compiled `.so` file and cannot be inspected.

## 3. Retrieval Mechanism

Multiple retrieval paths:

1. **Embedding search** (`semantic_search`): Brute-force cosine similarity over 384-dim MiniLM embeddings stored as binary blobs. No vector index (FAISS/Annoy) -- acknowledged as sufficient for <100K memories.

2. **Attention-weighted retrieval** (`attention_retrieve`): Keyword extraction from query, then scoring = `semantic_weight * keyword_overlap + recency_weight * exp_decay + confidence_weight * confidence`. Softmax normalization. LRU query cache (100 entries, 5min TTL).

3. **MMR retrieval** (`mmr_retrieve`): Maximal Marginal Relevance for diversity.

4. **Multi-head attention** (`attention_multihead`): Multiple attention passes across knowledge base.

5. **Type-filtered recall** (`recall_memories`): Filter by memory type, then strength-adjusted retrieval.

6. **Knowledge graph queries**: Subject-predicate-object triple queries, direct SQL on atoms table.

No RRF or hybrid fusion across these paths -- they appear to be independent tools the LLM selects between, not a unified pipeline.

## 4. Standout Feature: 4-Judge Belief Governance (Jury System)

The most architecturally novel feature is the multi-judge validation gate that evaluates all incoming memories before storage.

**Architecture (from readable source):**
- **SafetyJudge**: Regex-based PII detection (SSN, credit card, email, phone, passwords) and harmful keyword filtering. Has absolute VETO authority -- if it rejects, consensus is bypassed.
- **MemoryJudge**: Validates ontology compliance (predicate allowed for atom type), checks semantic sense (repeated words, generic objects, contradictory sentiment in predicate+object).
- **TimeJudge**: Validates temporal consistency (source in `time_judge.py`, not fetched but referenced).
- **ConsensusJudge**: Aggregates decisions via weighted voting:
  - Unanimous APPROVE -> APPROVE
  - Any REJECT -> QUARANTINE (not outright reject)
  - Majority QUARANTINE -> QUARANTINE
  - Split -> confidence-weighted vote, REJECTs count 2x
  - Low-confidence approval (<0.7) -> QUARANTINE

**Reality check:** All judges are rule-based, not LLM-powered. The code comments explicitly state "For MVP: rule-based" with "Future: Grammar-constrained LLM with Outlines." The MemoryJudge checks ontology compliance and basic heuristics (repeated words, generic objects, contradictory sentiment), not deep semantic validation. The SafetyJudge is keyword/regex matching. The system is a well-structured validation pipeline but far less sophisticated than "3-judge belief governance" might imply.

**Meta-Judge layer:** The README describes a meta-judge that monitors jury performance with accuracy tracking, calibration curves, and drift detection. This appears to be tracked via the `meta_judge_events` table and dashboard, but the implementation details are in compiled `.so` files.

## 5. Other Notable Features

1. **Epistemic Monitor**: Tracks factual claims with felt confidence, builds domain-specific calibration curves (`prediction_book` table), computes overconfidence ratios, and generates metacognitive prompts. The `check_before_claiming` tool adjusts confidence based on historical accuracy in the claim's domain. This is a genuine and well-implemented feature with ~540 lines of Python.

2. **Dual decay systems**: TypedMemory uses exponential decay with type-specific half-lives (episodic=48h, semantic=720h, belief=168h, procedural=2160h) and rehearsal boosts. MemoryAtom uses Ebbinghaus `R(t) = e^(-t/S)` where S scales with ontology-defined decay rate and confidence. Both include reconsolidation on retrieval.

3. **Knowledge graph**: Persisted graph with nodes and edges in SQLite. Tracks `connections` count and `value_score` per node (Metcalfe's Law analogy: value proportional to connections). Supports bidirectional edges with relationship types and strength.

4. **Episodic-to-semantic consolidation**: ConsolidationEngine clusters similar episodic memories via embedding similarity (threshold 0.55), then either promotes the cluster to a new semantic memory or reinforces an existing one.

5. **React dashboard**: Real-time visualization at localhost with overview, claims tracking, personality evolution, atom search, and meta-judge performance views. Uses Recharts.

## 6. Gap Ratings

| Gap | Rating | Evidence |
|-----|--------|----------|
| Layered Memory | 70% | Dual system: 4-type TypedMemory + 15-type MemoryAtom. Episodic-to-semantic consolidation. No explicit working/STM layer separate from LTM. |
| Multi-Angle Retrieval | 60% | Multiple independent retrieval tools (embedding, attention, MMR, multi-head, type-filtered, graph query) but no unified RRF/hybrid fusion pipeline. LLM must choose which tool to call. |
| Contradiction Detection | 55% | MemoryJudge catches basic contradictory sentiment. `evidence_for`/`evidence_against` fields on beliefs. `surface_conflicts`/`resolve_conflict` tools exist. `supersedes`/`superseded_by` links. But judges are rule-based heuristics, not semantic. |
| Relationship Edges | 65% | Knowledge graph with typed edges (relationship, strength, bidirectional). `related_atoms[]` on MemoryAtom. `knowledge_nodes`/`knowledge_edges` tables. But no novelty-scored expansion or edge-aware retrieval path. |
| Sleep Process | 10% | No offline batch processing. Decay and consolidation are on-demand (tool calls). No background restructuring. |
| Reference Index | 25% | `provenance` field (user_stated/confirmed/inferred/corrected), `assertion_count`, `explicit_confirms[]`. Basic source tracking but not a structured reference index. |
| Temporal Trajectories | 40% | `temporal_validity` dict on atoms, `episode_timestamp`, `first_observed`, `last_contradicted`. TimeJudge validates consistency. `decay_forecast` predicts future strength. But no trajectory analysis across time. |
| Confidence/UQ | 75% | Confidence field on both systems. Epistemic monitor with calibration curves and domain accuracy. `epistemic_distance` for belief attribution. Evidence tracking. Confidence decay on contradicting evidence. Over-confidence detection. |

## 7. Comparison with claude-memory

**Stronger:**
- Epistemic monitoring with calibration curves, claim tracking, and domain-specific accuracy measurement -- we have nothing comparable
- Multi-judge validation gate (even if rule-based) provides structured quality control at write time; our deduplication check is simpler
- Knowledge graph with explicit typed edges stored in SQLite, separate from the memory store
- Belief-specific fields: `evidence_for`/`evidence_against`, `epistemic_distance`, `belief_holder` -- explicit epistemic modeling
- Dual memory system allows different abstraction levels (cognitive types vs ontological atoms)
- Dashboard for observability

**Weaker:**
- No sleep/offline processing -- all operations are synchronous tool calls
- Embedding search is brute-force over binary blobs (no sqlite-vec, no ANN index)
- No FTS5 full-text search; retrieval relies on keyword overlap or embedding similarity
- No RRF hybrid fusion -- multiple independent retrieval tools instead of a unified pipeline
- Judges are all rule-based heuristics (keyword/regex), not semantic analysis
- Several critical modules shipped as compiled `.so` binaries -- untestable and unauditable
- No human-in-the-loop curation model (startup_load / recall_feedback equivalent)
- No shadow load or gestalt-like compressed representations
- 136 tools is an enormous tool surface that will consume context window budget

## 8. Insights Worth Stealing

1. **Epistemic claim tracking with calibration curves** (medium effort, high impact): Log factual claims with felt confidence, track resolution (correct/incorrect), compute per-domain accuracy ratios. When about to make a high-confidence claim in a domain where historical accuracy is low, adjust. Our system has no equivalent. Could be implemented as a new table + recall-time annotation.

2. **Evidence for/against on memories** (low effort, high impact): Explicit bidirectional evidence links. When a new memory supports or contradicts an existing one, record the link with the memory ID. We have edges with flags (contradiction/support) but they're created during sleep, not at write time. Adding explicit evidence_for/evidence_against at `remember()` time would be straightforward.

3. **Epistemic distance / belief holder** (low effort, medium impact): Tracking WHO holds a belief and how many levels removed it is (0=direct, 1=reported). Useful for distinguishing "Alexis said X" from "Alexis said that Y thinks X." Our `source` field partially covers this but doesn't track the attribution chain.

4. **Write-time validation gate** (medium effort, medium impact): Even a lightweight rule-based check at `remember()` time could catch PII, overly generic content, or near-duplicates before they enter the database. We currently only deduplicate by embedding similarity at 0.9 threshold.

5. **Decay forecasting** (low effort, low impact): Predicting when memories will cross threshold strength. Our power-law decay already enables this computation; we just don't surface it.

## 9. What's Not Worth It

- **136-tool MCP surface**: Extreme tool proliferation. Each tool consumes context describing it. Far better to have fewer tools with good parameter design.
- **Compiled .so binaries**: Shipping critical components as compiled binaries makes the system unauditable and platform-locked (these are darwin-only).
- **Dual memory system complexity**: Two parallel systems (TypedMemory + MemoryAtom) with different schemas, different decay models, and different retrieval paths creates maintenance burden without clear benefit over a single well-designed system.
- **Dashboard**: Nice for demos but not load-bearing for memory quality. The engineering effort is better spent on the memory pipeline itself.
- **Knowledge graph with Metcalfe's Law scoring**: The connection-count-equals-value metaphor sounds appealing but doesn't obviously improve retrieval quality compared to embedding similarity + edge traversal.

## 10. Key Takeaway

PLTM-Claude is an ambitious system with genuinely novel ideas -- particularly the epistemic monitoring with calibration curves and the evidence-tracking fields on beliefs. However, there is a significant gap between README claims and implementation reality: the "3-judge belief governance" is rule-based keyword matching, the "136 tools" create an unwieldy interface, and critical pipeline code is locked in compiled binaries. The strongest transferable idea is the epistemic claim tracking pattern: logging felt confidence alongside factual assertions, then building domain-specific accuracy feedback loops. This addresses a real gap in our system where we have no mechanism for calibrating how trustworthy memories are beyond the static confidence gradient.
