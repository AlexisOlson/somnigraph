# Modern-Prometheus-AI/Neuroca Repository Analysis

*Generated 2026-02-19 by Opus agent reading local clone at ~/.claude/repos/Neuroca*

## Repository Overview

**Name**: NeuroCognitive Architecture (NCA) / Neuroca
**Author**: Justin Lietz (jlietz93)
**License**: MIT
**Language**: Python 3.12+
**Status**: Self-described as **ALPHA**. README updated 4/15/2025.
**Key claim**: "Persistent Memory System for LLMs" -- a multi-tiered memory system (STM/MTM/LTM) with biological inspiration, positioned as superior to standard RAG.

**Critical context from the README itself**: The author's "first prototype" was produced by Claude 3.7 Sonnet, which "produced >70% of the codebase in one prompt" using the Apex code generation engine. This is evident in the codebase: extremely verbose, heavily documented scaffolding with consistent patterns suggesting AI-generated boilerplate. The git history is a single commit (shallow clone, but the README corroborates rapid generation).

**Maturity assessment**: This is predominantly **architectural scaffolding with substantial design intent but limited functional implementation**. Many subsystems have well-structured class hierarchies, proper error handling, and detailed docstrings, but the actual algorithmic substance is thin. The integration tests are explicitly skipped with a comment "These tests use the old memory architecture and need to be refactored." The consolidation between tiers is a simplified heuristic. The "lymphatic" system and "annealing" optimizer exist as code but appear untested and unlikely to have been run end-to-end.

## File Structure (Key Paths)

```
src/neuroca/
  memory/
    tiers/
      stm/core.py, components/{cleanup,expiry,lifecycle,operations,strength}.py
      mtm/core.py, components/{consolidation,lifecycle,operations,priority,promotion,strength}.py
      ltm/core.py, components/{category,lifecycle,maintenance,operations,relationship,strength}.py
    lymphatic/
      __init__.py          -- LymphaticMemory class (intermediate tier)
      consolidator.py      -- MemoryConsolidator with strategy pipeline
      scheduler.py         -- LymphaticScheduler (background task runner)
      abstractor.py        -- Abstractor: memory->abstract concept extraction
    annealing/
      optimizer/core.py    -- AnnealingOptimizer (simulated annealing for memory optimization)
      optimizer/components/energy.py  -- Energy function (redundancy, fragmentation, relevance)
      optimizer/components/transformations.py
      phases/{heating,slow_cooling,rapid_cooling,stabilization,maintenance}.py
      scheduler/           -- Multiple annealing schedule implementations
    tubules/
      connections.py       -- TubuleConnection/TubuleConnectionManager (neural pathway simulation)
      pathways.py
      weights.py
    manager/
      core.py              -- MemoryManager: orchestrates STM/MTM/LTM + background tasks
      consolidation.py     -- consolidate_stm_to_mtm(), consolidate_mtm_to_ltm()
      decay.py             -- decay_mtm_memories(), decay_ltm_memories()
      working_memory.py    -- Working memory buffer management
    backends/              -- in_memory, redis, sql, sqlite, vector backends
    models/memory_item.py  -- MemoryItem, MemoryContent, MemoryMetadata (Pydantic)
  core/
    cognitive_control/     -- attention_manager, decision_maker, goal_manager, etc.
    health/                -- Health dynamics system
  integration/             -- LLM adapters (OpenAI, Anthropic, Ollama, VertexAI)
tests/                     -- ~15,600 lines, but integration tests are all skipped
```

**Scale**: ~110,000 lines of Python source. ~15,600 lines of tests. ~200+ documentation files. The sheer volume is characteristic of AI-generated projects -- broad surface area, thin depth.

## Architecture Overview

### Three-Tier Memory (STM / MTM / LTM)

The core design mirrors biological memory stages, but uses **three tiers** rather than the sensory/working/long-term model common in cognitive science:

1. **STM (Short-Term Memory)**: TTL-based (default 1 hour), linear decay (0.05 strength loss per minute). Backed by in-memory storage. Memories automatically expire. Strength resets to 1.0 on access.

2. **MTM (Medium-Term Memory)**: Capacity-limited (default 1000), priority-based management (high/medium/low), weighted scoring combining priority (0.4), recency (0.3), and frequency (0.3). Backed by Redis. Includes a **promotion** component that marks memories for LTM transfer based on access count threshold (default 3) and importance (default 0.6).

3. **LTM (Long-Term Memory)**: Permanent storage with semantic relationships, categories, and maintenance cycles. Backed by SQL. Supports 7 relationship types (semantic, causal, temporal, spatial, associative, hierarchical, **contradictory**). Includes BFS-based path finding between memories.

### Consolidation Pipeline

Managed by `MemoryManager.core.py` with periodic background tasks:
- **STM->MTM**: Every 5 minutes. Scores memories by `importance * (0.5 + 0.5 * min(access_count, 10)/10)`. Threshold: 0.6. Batch size: 5.
- **MTM->LTM**: Every 5 minutes (same cycle). Scores by `importance * 0.5 + access_freq * 0.3 + age * 0.2`. Threshold: 0.7. Batch size: 3.

### Decay

- **MTM decay**: `base_decay * importance_factor * access_factor * time_factor` where factors slow decay for high-importance and high-access memories. Strength floor at 0.1 triggers forgetting. Runs every 10 minutes.
- **LTM decay**: **Not implemented** ("LTM decay not fully implemented yet" -- literal comment in code).
- **STM decay**: Linear: `strength = 1.0 - (elapsed_minutes * 0.05)`. Resets on access.

### The "Lymphatic" System

Named after the brain's glymphatic system that clears waste during sleep. Three components:

1. **LymphaticScheduler**: A general-purpose background task scheduler with priority levels, recurring tasks, retry logic with exponential backoff, and idle detection (uses psutil CPU monitoring). This is actually well-implemented as a generic task runner.

2. **MemoryConsolidator** (lymphatic/consolidator.py): Applies a pipeline of strategies:
   - **ImportanceBasedStrategy**: Filters below threshold (default 0.5)
   - **SemanticClusteringStrategy**: Groups by tag overlap (not embedding similarity -- comment says "In a real implementation, this would use embedding similarity")
   - **TemporalDecayStrategy**: Applies `strength *= 1.0 - (decay_rate * age/max_age)`. Not exponential despite the docstring claiming `e^(-decay_rate * age)`.

3. **Abstractor** (lymphatic/abstractor.py): Extracts "abstract concepts" from memory clusters. Feature extraction (keywords, entities, sentiment), Jaccard/cosine similarity for clustering, concept generation with confidence scores, embedding averaging. **Heavily scaffolded** -- keyword extraction is split-on-whitespace, entity extraction just reads pre-existing attributes, sentiment analysis returns 0.0 by default.

### The Annealing System ("Dreaming")

This is the most novel component and closest to our /sleep design:

**AnnealingOptimizer** (`annealing/optimizer/core.py`): Classic simulated annealing applied to memory organization. It:
- Takes a list of MemoryItems and optimizes their arrangement
- Uses an energy function with three components:
  - **Redundancy energy**: Pairwise similarity squared, normalized
  - **Fragmentation energy**: Connected components analysis (DFS) on a similarity graph
  - **Relevance energy**: Inverse of relevance scores
- Strategy weights (STANDARD: 0.4/0.3/0.3; AGGRESSIVE: 0.5/0.3/0.2; CONSERVATIVE: 0.2/0.3/0.5; ADAPTIVE: adjusts based on redundancy)
- Boltzmann acceptance: `P(accept) = exp(-delta_E / T)`
- Early stopping after N iterations without improvement
- Multiple temperature schedules: exponential, linear, logarithmic, cosine, adaptive

**Annealing Phases**: A multi-phase process:
- **Heating**: Increases temperature, makes memories more "malleable", reinforces important memories
- **Slow Cooling**: Gradual consolidation, strength increases for high-scoring memories, decay for low-scoring
- **Rapid Cooling**: Quick lock-in of memory structures
- **Stabilization**: Final settling
- **Maintenance**: Ongoing upkeep

### Tubules (Neural Pathways)

`TubuleConnectionManager` implements a graph of connections between memory nodes with:
- Connection types: excitatory, inhibitory, modulatory
- Signal propagation with type-dependent modulation
- Time-based decay on connections
- Pruning of weak connections
- State machine: FORMING -> ACTIVE -> WEAKENING -> PRUNED

This is conceptually interesting but appears completely disconnected from the actual memory tier system. No code in the memory manager references tubules.

## Unique Concepts

1. **Simulated annealing for memory optimization**: The energy function combining redundancy, fragmentation, and relevance as a cost function for memory reorganization is genuinely novel. The idea that you can frame memory consolidation as an optimization problem over an energy landscape is compelling.

2. **Lymphatic/glymphatic naming**: The metaphor of brain waste clearance during sleep for background memory maintenance is well-chosen and aligns directly with our /sleep design philosophy.

3. **Multi-phase annealing**: The heating/cooling/stabilization phases map well onto the idea that sleep consolidation has stages (like NREM/REM cycles).

4. **"Contradictory" relationship type**: LTM explicitly defines a "contradictory" relationship type alongside semantic, causal, temporal, etc. This is directly relevant to our graded contradiction detection goal.

5. **Tubule connection types**: Excitatory/inhibitory/modulatory connections between memories, with signal propagation and strength modulation. Though unintegrated, the concept of inhibitory connections (memories that suppress other memories) is interesting.

6. **Idle-aware scheduling**: The scheduler checks CPU utilization via psutil to decide when to run maintenance tasks. While simplistic, the principle of "consolidate during idle time" maps to our /sleep trigger.

## Worth Stealing (ranked)

### 1. Energy Function Design for Memory Optimization (HIGH)
**File**: `src/neuroca/memory/annealing/optimizer/components/energy.py`

The three-component energy function (redundancy + fragmentation + relevance) is a clean framework for evaluating memory system health. For claude-memory, this could define what "good" looks like after a /sleep cycle:
- **Redundancy**: How much semantic overlap exists between stored memories (we want some redundancy at different gestalt levels, but not duplication)
- **Fragmentation**: Are related memories disconnected? (Graph component analysis)
- **Relevance**: Is the overall memory store relevant to the user's needs?

The strategy weights (aggressive/conservative/adaptive) give a knob for tuning consolidation behavior. We could use this as a quality metric: run the energy function before and after /sleep to measure improvement.

### 2. Contradictory Relationship Type (MEDIUM-HIGH)
**File**: `src/neuroca/memory/tiers/ltm/components/relationship.py`

Their explicit "contradictory" type in the relationship taxonomy validates our graded contradiction detection priority. Their implementation is just a string label, but the data model (`Relationship` class with source_id, target_id, type, strength) could inform our schema. We need richer semantics (which memory wins? when was each created? what's the confidence?) but the structural idea is sound.

### 3. Promotion Criteria Pattern (MEDIUM)
**File**: `src/neuroca/memory/tiers/mtm/components/promotion.py`

The MTMPromotion class has a clear pattern for deciding when memories move between tiers:
- Minimum access count threshold (default 3)
- Minimum importance threshold (default 0.6)
- Explicit promotion request flag
- Auto-promote high-priority items meeting importance threshold

This is simpler than our temperature-based approach but the "check on every access" pattern is useful. Our system could benefit from similar trigger points where we re-evaluate tier placement.

### 4. Background Scheduler with Idle Detection (MEDIUM)
**File**: `src/neuroca/memory/lymphatic/scheduler.py`

The LymphaticScheduler is actually well-implemented: priority queue, recurring tasks, exponential backoff retries, concurrent task limits, idle detection. For claude-memory's /sleep, we currently trigger on session events. Their idle detection (checking CPU utilization) is too simplistic for our use case, but the general architecture of a priority-based task scheduler for consolidation work could be useful if we move toward a background service model.

### 5. Multi-Phase Consolidation (LOW-MEDIUM)
**Files**: `src/neuroca/memory/annealing/phases/*.py`

The idea that consolidation has phases (heating = loosen structures, slow cooling = gradually solidify, stabilization = finalize) maps onto our detail->summary->gestalt pipeline. Their implementation is too abstract to use directly, but the phase concept could inform how we structure /sleep cycles.

## Not Worth It

1. **The entire STM/MTM/LTM tier implementation**: Massive volume of code (hundreds of files) for what amounts to simple key-value stores with metadata. The modular decomposition (each tier has lifecycle, operations, strength, etc. components) is over-engineered for what each component actually does. Our Supabase + pgvector approach is more practical.

2. **The Abstractor**: Claims to extract abstract concepts from memories but the actual implementation is keyword-counting and tag-overlap clustering. No LLM calls, no real NLP, no embedding-based similarity. Our LLM-powered summarization for the summary and gestalt layers is vastly more capable.

3. **Tubules system**: Completely disconnected from the rest of the codebase. No integration point exists. This is architectural speculation, not functional code.

4. **Cognitive control modules**: Attention manager, decision maker, goal manager, inhibitor, metacognition, planner -- all exist as files in `core/cognitive_control/` but are not integrated with the memory system.

5. **Backend abstraction layer**: Five backend implementations (in-memory, Redis, SQL, SQLite, vector) with factory pattern. Impressive breadth but our single-backend Supabase approach is more maintainable and sufficient for our needs.

6. **The health dynamics system**: Partially implemented monitoring/health system. Not relevant to our use case.

## Key Weakness

**The fundamental problem is that this is a design document disguised as a codebase.** The README makes ambitious claims ("dynamic, multi-tiered memory system inspired by biological cognition" that allows LLMs to "genuinely learn and evolve") but the actual implementation is:

- **Consolidation**: Threshold-based filtering by importance score. No LLM involvement, no semantic understanding.
- **Decay**: Linear strength reduction. Not power-law, not Ebbinghaus, not biologically modeled.
- **Semantic clustering**: Tag overlap, not embedding similarity (despite having a vector backend).
- **Abstraction**: Word splitting, not NLP.
- **LTM decay**: Literally not implemented.
- **Tests**: Integration tests are all skipped. Unit tests exist but coverage appears low.
- **Integration**: Despite claiming LLM integration, the memory system itself never calls an LLM.

The 110K lines of source code create an impression of completeness that doesn't hold up under examination. Each subsystem has proper class hierarchies, error handling, logging, and documentation, but the algorithmic core of each subsystem is a placeholder or trivial implementation. The author acknowledges this is an "ALPHA" built primarily by AI code generation.

The annealing optimizer is the most algorithmically substantive component, but it too has a gap: the `generate_neighbor()` and `post_process()` functions (in `transformations.py`) that actually mutate memory state during optimization are imported but their implementations would determine whether this approach actually works.

## Relevance to claude-memory

### Direct Analogies

| Neuroca | claude-memory | Assessment |
|---------|--------------|------------|
| STM/MTM/LTM tiers | detail/summary/gestalt layers | Different axis: Neuroca's tiers are about persistence duration; ours are about abstraction level. Both are valid three-tier designs. |
| Lymphatic consolidation | /sleep skill | Same concept, different maturity. Their version runs strategies in a pipeline; ours uses LLM-powered summarization. |
| Annealing optimizer | No equivalent yet | Novel idea. Could inform how we evaluate /sleep cycle quality. |
| Memory decay | Temperature-based decay | Their decay is linear; we use temperature metaphor with power-law intent. Our approach is more biologically grounded. |
| Contradictory relationship | Graded contradiction detection (planned) | Validates the design priority. Their implementation is just a label; we need semantic comparison. |
| Promotion criteria | Temperature thresholds | Both use multi-factor scoring to decide tier transitions. |
| Tubule connections | Relationship edges (planned) | Same concept. Their implementation is more sophisticated (excitatory/inhibitory/modulatory) but unintegrated. |

### What This Confirms for Us

1. **Three-tier memory is the right structure**, but abstraction level (our approach) is more useful than persistence duration (their approach) for LLM memory.

2. **Background consolidation is essential** -- both systems recognize that memory maintenance must happen asynchronously.

3. **Multiple factors should drive tier transitions** -- importance, access frequency, age, explicit flags.

4. **Contradiction detection needs explicit support** -- it shouldn't just be inferred from low similarity; it needs its own data structure.

5. **The energy function concept could be a quality metric for /sleep** -- measuring redundancy, fragmentation, and relevance before and after consolidation tells us if the cycle was productive.

### What This Warns Us About

1. **Don't over-architect the backend layer.** Their five-backend abstraction is impressive but probably unnecessary. Our single Supabase backend is more pragmatic.

2. **The gap between design and implementation is the real risk.** Neuroca has beautiful architecture diagrams and comprehensive documentation but the algorithms are placeholders. We should keep our implementation grounded and test-driven.

3. **AI-generated code can create massive surface area without depth.** The 110K lines look impressive but most are boilerplate. Our leaner approach is healthier.

4. **Without LLM integration in the memory system itself, "biologically-inspired" memory is just priority queues with extra steps.** The actual power of our /sleep design comes from using LLMs to perform the summarization and gestalt extraction. Neuroca's consolidation is just threshold filtering.
