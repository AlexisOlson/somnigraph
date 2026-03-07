# CortexGraph Analysis (Agent Output)

*Generated 2026-02-18 by Opus agent reading local clone*

---

## CortexGraph Analysis: Architecture, Comparison, and Insights for claude-memory

### Repository Overview

**CortexGraph** (formerly "mnemex") is an MCP server providing temporal memory with Ebbinghaus-style forgetting curves. Developed by "Prefrontal Systems," it's a research PoC (AGPL-3.0), well-documented with 791 tests and 98%+ coverage.

### File Structure (Key Paths)

```
src/cortexgraph/
  core/
    decay.py              -- Temporal decay scoring (3 models)
    scoring.py            -- Score-based decision functions
    clustering.py         -- Single-linkage clustering
    consolidation.py      -- Memory merge logic
    similarity.py         -- cosine, jaccard, tfidf, text_similarity
    auto_recall.py        -- Conversational auto-reinforcement
    review.py             -- Spaced repetition / danger zone
  agents/
    decay_analyzer.py     -- Find at-risk memories
    cluster_detector.py   -- Group similar memories
    semantic_merge.py     -- Merge clustered memories
    ltm_promoter.py       -- Promote to Obsidian vault
    relationship_discovery.py -- Cross-domain link discovery
    scheduler.py          -- Pipeline orchestration
  storage/
    jsonl_storage.py      -- JSONL backend (primary)
    sqlite_storage.py     -- SQLite backend
    ltm_index.py          -- Obsidian vault indexing
    models.py             -- All data models (Pydantic)
  tools/
    save.py, search.py, search_unified.py, consolidate.py,
    create_relation.py, promote.py, read_graph.py, etc.
  activation/
    detectors.py, entity_extraction.py, patterns.py
schemas/
  pgvector_schema.sql     -- PostgreSQL migration (planned, not implemented)
  memory_architecture.md  -- Postgres design doc
  temporal_decay_design.md -- Mathematical foundations
```

---

## 1. Architecture Overview

### Core Components

**Storage layer**: Primarily JSONL files loaded fully into memory at startup, with optional SQLite. A PostgreSQL+pgvector migration is designed but not implemented.

**Decay engine**: Three models available:
- **Power-law (default)**: `f(dt) = (1 + dt/t0)^(-alpha)` -- heavier tail, retains older memories
- **Exponential**: `f(dt) = e^(-lambda*dt)` -- classic Ebbinghaus
- **Two-component**: `f(dt) = w*e^(-lambda_fast*dt) + (1-w)*e^(-lambda_slow*dt)` -- fast early forgetting + slow tail

Master scoring formula: `score = (use_count + 1)^beta * f(dt) * strength`

**Two-layer architecture (STM + LTM)**:
- STM: JSONL/SQLite, subject to decay, garbage collection
- LTM: Obsidian markdown files with YAML frontmatter and wikilinks
- Promotion criteria: score >= 0.65 OR use_count >= 5 within 14 days
- Forgetting criteria: score < 0.05

**Knowledge graph**: Memories as nodes, explicit typed relations as edges. Relation types: `related`, `causes`, `supports`, `contradicts`, `has_decision`, `consolidated_from`.

**Multi-agent consolidation pipeline** (5 agents, run in sequence):
1. DecayAnalyzer -- finds memories in "danger zone" (score 0.15-0.35)
2. ClusterDetector -- groups similar memories
3. SemanticMerge -- combines clustered memories
4. LTMPromoter -- writes high-value memories to Obsidian vault
5. RelationshipDiscovery -- creates edges between memories sharing entities

---

## 2. Unique Concepts

**a. Ebbinghaus-inspired temporal decay with multiple models.** Rather than all memories being equal at retrieval time, each memory has a continuously-computed score that decays with time since last access but is boosted by use frequency and importance.

**b. Natural spaced repetition via "danger zone" blending.** Memories with scores in 0.15-0.35 range are blended into search results (30% blend ratio) so they naturally get reinforced through use.

**c. Cross-domain usage detection.** When a memory is used in a context with sufficiently different tags (Jaccard similarity < 0.3), it gets a strength boost.

**d. Silent background reinforcement.** The AutoRecallEngine monitors conversation, extracts topics, and silently touches related memories to prevent decay.

**e. Promotion to human-readable permanent storage.** LTM is an Obsidian vault with markdown files, YAML frontmatter, and wikilinks.

**f. Typed relation edges with "contradicts" support.** Though the system doesn't automatically detect contradictions -- only stores them if created manually or by agents.

---

## 3. How CortexGraph Addresses Our 7 Known Gaps

| Gap | Rating | Notes |
|-----|--------|-------|
| 1. Layered Memory | 2/10 | Two layers (STM/LTM) but about durability, not abstraction. No summary/gestalt generation. |
| 2. Multi-Angle Retrieval | 3/10 | Multiple search paths exist but selected by caller, not automatically combined. |
| 3. Contradiction Detection | 1/10 | Relation type exists but no detection mechanism. |
| 4. Relationship Edges | 6/10 | Good foundation with typed, weighted, auto-discovered relations. But edges don't enhance retrieval. |
| 5. Sleep Process | 5/10 | 5-agent pipeline infrastructure exists, but intelligence is limited to dedup and link creation. |
| 6. Reference Index | 3/10 | Stats exist but no true reference index. |
| 7. Temporal Trajectories | 0/10 | Completely absent. |

---

## 4. Comparison

### Where CortexGraph is Stronger
1. **Temporal decay model** -- Three models (power-law, exponential, two-component), mathematically grounded
2. **Natural spaced repetition** -- Danger-zone detection + review blending
3. **Relationship edges** -- Typed, weighted, auto-discovered
4. **Consolidation pipeline** -- 5-agent pipeline with scheduler, dry-run, rate limiting
5. **LTM as Obsidian markdown** -- Human-readable, user-editable
6. **Silent background reinforcement** -- AutoRecallEngine

### Where Our System is Stronger
1. **Vector search** -- text-embedding-3-large @ 1536 dims vs their optional MiniLM @ 384
2. **Server-side database** -- Supabase/PostgreSQL vs JSONL files
3. **Priority system** -- 1-10 with p10 exempt is transparent and controllable
4. **Categories** -- Semantically meaningful vs freeform tags only
5. **Dedup on write** -- 0.9 cosine at remember() time
6. **Simpler operational model** -- Single MCP server

---

## 5. Insights Worth Stealing

### A. Power-Law Decay (High Priority)
Our `base * 0.5^(days/90)` is exponential. Power-law `(1 + dt/t0)^(-alpha)` has heavier tail that better preserves important old memories. For identity/continuity, this matters.

### B. Danger Zone + Review Blending (Medium Priority)
Memories in 0.15-0.35 score range blended into recall results for natural reinforcement.

### C. Cross-Domain Usage Detection (Medium Priority)
When same memory surfaces in very different theme contexts, boost it.

### D. Silent Background Reinforcement (Low-Medium Priority)
Touch memories related to current conversation topics to prevent decay.

### E. Typed Relationship Edges (High Priority)
Auto-discover based on shared themes/entities, don't rely only on manual creation.

### F. Consolidation Tracking Relations (Low Priority)
`consolidated_from` edges preserve provenance when merging.

---

## 6. What's Not Worth It

- **JSONL Storage** -- We have Supabase
- **Obsidian LTM Integration** -- Adds operational complexity
- **Multi-Agent Pipeline Architecture** -- Overkill for single user. The operations are worth doing; the architecture isn't.
- **NLP-Based Entity Extraction (spaCy)** -- Heavy dependency, better to use LLM
- **Strength 0-2 Multiplier** -- Less expressive than our 1-10
- **Auto-Save/Auto-Recall Intent Detection** -- Regex-based, brittle, unnecessary with Claude
