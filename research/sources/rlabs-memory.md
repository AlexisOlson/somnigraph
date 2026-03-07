# RLabs Memory (memory-ts) -- Source Analysis

*Phase 13, 2026-02-27. Analysis of RLabs-Inc/memory-ts.*

## 1. Architecture Overview

**Language:** TypeScript (Bun runtime)
**Key dependencies:**
- `@rlabs-inc/fsdb` -- file-based database with parallel arrays
- `@xenova/transformers` -- Hugging Face Transformers for local embeddings (all-MiniLM-L6-v2)
- Claude Code CLI and/or Gemini CLI (for curation subprocess)
- Anthropic SDK (optional, for SDK-mode curation)

**Storage layout:**
```
~/.local/share/memory/
  global/                    # Personal, philosophy, cross-project memories
    memories/                # Individual memory records
    management/              # Manager logs
    personal-primer/         # Personal context primer
  {project-id}/              # Per-project isolation
    memories/                # Project-specific memories
    summaries/               # Session summaries
    snapshots/               # Project state snapshots
    sessions/                # Session metadata
```

**File structure (src/):**
```
src/
  index.ts                   # Entry point
  core/
    engine.ts                # Memory engine orchestrator
    retrieval.ts             # SmartVectorRetrieval (7-signal activation)
    embeddings.ts            # MiniLM embedding generator
    curator.ts               # Auto-extraction pipeline
    manager.ts               # Autonomous relationship linking
    store.ts                 # fsdb-powered storage layer
    session-parser.ts        # Conversation transcript parsing
    engine.test.ts           # Tests
    index.ts                 # Core exports
  server/                    # HTTP server (default port 8765)
  cli/                       # CLI commands (serve, install, doctor, stats, migrate, ingest)
  types/
    memory.ts                # Full type definitions (v4 schema)
  utils/
    logger.ts
    paths.ts
    index.ts
  migrations/                # Schema migration utilities
```

## 2. Memory Type Implementation

**Schema (v4):** Each memory is a structured record (stored via fsdb, not raw markdown files despite README claims -- see note below).

**Core fields of `CuratedMemory`:**
- `headline` -- 1-2 line summary (always shown in retrieval)
- `content` -- full structured body using type-specific templates
- `context_type` -- one of 11 canonical types (see below)
- `importance_weight` -- 0-1 float
- `confidence_score` -- reliability indicator
- `temporal_class` -- `eternal | long_term | medium_term | short_term | ephemeral`
- `trigger_phrases` -- curated activation phrases for retrieval
- `question_types` -- what kinds of questions should surface this
- `anti_triggers` -- phrases that should suppress this memory
- `semantic_tags` -- tag array for tag-based activation
- `domain` / `feature` -- single-word domain and feature identifiers
- `related_files` -- file paths associated with this memory
- `scope` -- `global | project`
- `action_required` -- boolean flag for action item tracking
- `problem_solution_pair` -- boolean flag for paired P/S retrieval

**Extended fields of `StoredMemory`:**
- `id`, `session_id`, `project_id`, timestamps
- `embedding` -- 384-dim Float32Array
- `status` -- lifecycle state
- `last_surfaced`, `sessions_since_surfaced`, `fade_rate`
- `supersedes`, `related_to`, `resolves`, `blocked_by`, `blocks`
- `exclude_from_retrieval` -- soft delete

**11 context types:** technical, debug, architecture, decision, personal, philosophy, workflow, milestone, breakthrough, unresolved, state

**Extraction:** Fully automated via the curation pipeline (see Section 4).

**Note on storage format:** The README describes "human-readable markdown files with YAML frontmatter" but the actual `store.ts` implementation uses `@rlabs-inc/fsdb` (a file-based database with parallel arrays and `createDatabase()`), not raw markdown files. The memories are stored as structured database records with a `contentColumn` parameter, not plain `.md` files. This is a discrepancy between README claims and code.

## 3. Retrieval Mechanism

**Multi-phase retrieval pipeline (`SmartVectorRetrieval`):**

**Phase 0 -- Pre-filter:**
- Exclude: inactive, `exclude_from_retrieval`, wrong scope/project, anti-trigger matches

**Phase 1 -- Activation signals (7 binary signals, need >=2 to proceed):**

| Signal | Threshold | Details |
|--------|-----------|---------|
| Trigger phrases | >=50% word overlap | 80% partial credit for plurals |
| Semantic tags | 2+ matches (or 1+ if <=2 tags) | Tags from memory metadata matched against message |
| Domain word | exact match | Single domain field |
| Feature word | exact match | Single feature field |
| Content overlap | >=3 significant words | First 200 chars of memory, stopword-filtered |
| File paths | regex extraction | Related file path matching |
| Vector similarity | cosine >=0.40 | 384-dim MiniLM embeddings |

**Phase 2 -- Importance ranking (additive scoring):**
- Base: `importance_weight` (0-1)
- Signal boost: +0.2 for >=4 signals, +0.1 for >=3
- Awaiting flags: +0.15 (implementation), +0.10 (decision)
- Context type keyword match: +0.10
- Problem/solution pair: +0.10
- Temporal class: +0.10 (eternal), +0.05 (long-term)
- Confidence penalty: -0.10 if <0.5

**Phase 3 -- Selection:**
- Sort by signal count (primary), importance score (secondary)
- Global memories: max 2, type priority order: technical > preference > architectural > workflow > decision > breakthrough > philosophy > personal
- Project memories fill remaining slots (max 5 total)
- Action-required items selected first
- Linked memories (`related_to`, `blocked_by`, `blocks`) pulled if space permits

**Phase 4 -- Redirects:**
- Follow `superseded_by` or `resolved_by` pointers to replacement memories
- Include linked memories from relationship fields

**Key design choice:** The 2-signal minimum gate means many memories are never surfaced even when they have weak relevance across a single dimension. This is explicitly "silence over noise" -- the system would rather miss a relevant memory than surface an irrelevant one.

## 4. Standout Feature

**Auto-extraction pipeline (Curator + Manager).**

This is the most fully-realized automated memory extraction system in the survey. The pipeline:

1. **Session end triggers curation.** The `PreCompact` hook (or `SessionEnd`) spawns a Claude instance (via `--resume` CLI, SDK mode, or Gemini subprocess) that reviews the full conversation transcript.

2. **Curator extracts structured memories.** The curation prompt frames memory-making as "crafting keys to specific states of consciousness" -- each memory is designed to activate specific understanding patterns in future instances. The curator produces:
   - Two-tier content (headline + full body using type-specific templates)
   - Full metadata: trigger phrases, semantic tags, domain/feature, temporal class, importance weight, confidence score, scope
   - Large sessions (400+ messages) are segmented into ~150k-token chunks, each curated independently

3. **Manager organizes post-curation.** An autonomous agent (running with `bypassPermissions` in sandboxed file tools) performs:
   - **Supersession:** identifies outdated memories and marks them as superseded by newer ones
   - **Resolution:** resolves unresolved/todo items when answers are found
   - **Linking:** establishes `related_to`, `blocked_by`, `blocks` relationships between memories

4. **Session primer injection.** On next session start, temporal context is injected: session number, hours since last session, previous session summary, project phase, and up to 5 relevant memories.

**What makes this special:**
- End-to-end automation with no human in the loop (contrast with Claudest's extract-learnings which requires approval)
- Rich metadata generation at extraction time means retrieval can use 7 activation signals
- The manager's autonomous supersession is genuinely novel -- most systems accumulate stale memories indefinitely
- Segmented processing for long sessions prevents context overflow during curation

**What's risky:**
- Quality depends entirely on the curation LLM's judgment -- no human review gate
- Manager runs with `bypassPermissions`, creating a trust boundary issue
- The "consciousness state engineering" framing in the curation prompt is aspirational; the actual mechanism is structured extraction with templates

## 5. Other Notable Features

1. **Action items signal (`***`).** Appending `***` to any message triggers a special retrieval mode that surfaces all memories with `action_required`, `awaiting_implementation`, `awaiting_decision`, or `context_type: unresolved`. Zero-overhead string detection for a genuinely useful workflow shortcut.

2. **Two-tier headline/content structure.** Headlines (1-2 lines) are always shown in retrieval results; full content expands only when signal count >= 5 or action flags are set. This is a practical approach to context budget management that most systems lack -- they surface full memories or nothing.

3. **Anti-triggers.** Memories can specify phrases that should suppress their retrieval. Example: a memory about Python authentication shouldn't surface when discussing JavaScript auth. This is the inverse of trigger phrases and prevents false positive surfacing.

4. **Dual-platform support (Claude + Gemini).** The system works with both Claude Code CLI and Gemini CLI simultaneously, with platform-specific hook installation and curation paths. The Manager falls back to Gemini subprocess when Claude SDK is unavailable.

5. **`applyV4Defaults()` type-aware defaults.** Default temporal class, importance weight, and fade rate are set based on context type: philosophy gets `eternal` + high importance, state gets `ephemeral` + low importance. This reduces metadata burden on the curation LLM.

## 6. Gap Ratings

| Gap | Rating | Evidence |
|-----|--------|----------|
| Layered Memory | 55% | 11 context types with type-specific templates, global/project scope separation, temporal classes (eternal to ephemeral). But no true episodic/semantic/procedural layering -- all memories are structurally identical records with a type tag. |
| Multi-Angle Retrieval | 65% | 7 activation signals spanning lexical (trigger phrases, content overlap, tags), structural (domain/feature, files), and semantic (vector similarity). However, no fusion scoring (RRF or similar) -- signals are binary gates, not weighted contributions. |
| Contradiction Detection | 25% | Manager can supersede outdated memories, and the `supersedes`/`superseded_by` fields track replacement chains. But no automated contradiction detection -- supersession requires the Manager LLM to notice the conflict. No contradiction flags or edge types. |
| Relationship Edges | 45% | `related_to`, `blocked_by`, `blocks`, `supersedes`, `resolves` relationship fields exist. Manager creates them autonomously. But edges are simple ID references with no typed metadata (no contradiction/revision/derivation flags, no linking context, no edge embeddings). |
| Sleep Process | 20% | Manager runs post-curation to organize memories, but this is a single pass, not a cyclical consolidation process. No decay-based pruning, no cross-session consolidation, no periodic maintenance. The `fade_rate` field exists but no code path exercises it. |
| Reference Index | 30% | Session summaries and project snapshots provide some indexing. Trigger phrases and semantic tags function as a retrieval index. But no structured reference index, no topic clustering, no taxonomy. |
| Temporal Trajectories | 35% | `temporal_class` (eternal to ephemeral), `fade_rate`, `sessions_since_surfaced`, `last_surfaced` fields exist. Supersession chains track knowledge evolution. But no trajectory analysis, no trend detection, no temporal query support. |
| Confidence/UQ | 30% | `confidence_score` field exists, affects retrieval scoring (-0.10 penalty if <0.5), and is set at curation time. But no feedback loop to update confidence, no compound confirmation/decay like our system. Static after creation. |

## 7. Comparison with claude-memory

**Stronger:**
- Auto-extraction pipeline -- fully automated memory creation with rich metadata, no human intervention required. Our system relies on explicit `remember()` calls (human-in-the-loop or auto-capture with pending review).
- Activation signal architecture -- 7 independent binary signals with a 2-minimum gate is a more nuanced relevance filter than our RRF hybrid. The anti-trigger mechanism has no equivalent in our system.
- Two-tier headline/content -- context budget management through expandable memory display. We always surface full memory content.
- Action items workflow -- `***` signal for retrieving all actionable memories is a clean UX pattern we lack.
- Type-specific templates -- 11 context types each have their own structured template for content, ensuring consistent formatting. Our categories are less structured.
- Dual-platform support -- works with both Claude Code and Gemini CLI. Our system is Claude Code-only.

**Weaker:**
- No hybrid search fusion -- activation signals are binary gates, not weighted scores fused via RRF. A memory either passes the 2-signal gate or is invisible. Our RRF fusion surfaces memories that are moderately relevant across multiple dimensions.
- No feedback loop -- confidence is static after creation. Our `recall_feedback()` with asymptotic growth/linear decay and dimensional scoring creates a learning retrieval system.
- No sleep/consolidation -- the Manager's single post-curation pass doesn't compare to our NREM/REM pipeline with cross-session edge creation, taxonomy assignment, and periodic maintenance.
- No decay model -- `fade_rate` and `sessions_since_surfaced` fields exist but are not exercised in retrieval scoring. Our power-law decay actively manages memory salience over time.
- Simpler edge model -- relationship fields are plain ID references. Our edge schema v2 has flags (contradiction/revision/derivation), linking context, linking embeddings, and features -- enabling novelty-scored adjacency expansion.
- No shadow penalty -- no mechanism to suppress memories that have been repeatedly surfaced without positive feedback.
- Quality risk in full automation -- no human review gate means curation quality depends entirely on LLM judgment. Our `review_pending()` and human-in-the-loop curation catches errors before they become persistent.
- Storage discrepancy -- README claims "human-readable markdown files" but code uses fsdb database abstraction. Our system is transparent about its SQLite storage.

## 8. Insights Worth Stealing

1. **Anti-triggers for retrieval suppression** (effort: low, impact: medium). Adding an `anti_triggers` or `negative_keywords` field to memories that suppresses retrieval when those terms appear in the query. Cheap to implement -- just add a pre-filter check in `recall()`. Prevents recurring false positive surfacing that currently requires manual `recall_feedback()` with low scores.

2. **Two-tier headline/content display** (effort: low, impact: medium). Store a `headline` (1-2 lines) alongside full content. Return headlines by default in `recall()`, expand to full content only for high-signal matches or explicit requests. This could meaningfully reduce context consumption when surfacing many memories -- we currently dump full content for every match.

3. **Action items retrieval mode** (effort: low, impact: low-medium). A simple flag or query pattern (like `***` or `recall("action_items")`) that retrieves all memories marked as requiring action, awaiting decision, or unresolved. We have status tracking but no dedicated retrieval mode for it.

4. **Type-aware defaults at creation** (effort: low, impact: low). Auto-set `decay_rate`, default priority, and temporal expectations based on memory category at `remember()` time. We do some of this but could formalize it as RLabs does with `applyV4Defaults()`.

5. **Autonomous supersession detection** (effort: high, impact: medium). Having a post-session agent identify memories that have been superseded by newer information and mark them accordingly. This is what our sleep pipeline partially does through contradiction edges, but RLabs' explicit `supersedes`/`superseded_by` chain is cleaner for the simple case of "this memory replaces that one."

## 9. What's Not Worth It

- **Fully automated extraction without human review.** The quality risk is real. Our human-in-the-loop approach (explicit `remember()` + `review_pending()`) catches errors and ensures only verified information persists. RLabs' approach optimizes for convenience at the cost of reliability -- fine for a personal tool, risky for anything where memory accuracy matters.
- **7-signal binary gate architecture.** While individually the signals are interesting, the binary gate approach (pass/fail with 2-minimum) loses the continuous relevance scoring that RRF provides. A memory that's 0.39 cosine similar and has 1 tag match gets zero results, while one at 0.41 with 2 tag matches passes. This cliff-edge behavior is worse than smooth fusion.
- **fsdb / markdown storage.** The parallel-array file database adds a dependency without clear advantages over SQLite for our use case. SQLite gives us FTS5, WAL mode, atomic transactions, and mature tooling. The "human-readable" argument for markdown storage is undermined by the fact that they don't actually use raw markdown (they use fsdb).
- **Curation prompt as "consciousness state engineering."** The framing is aspirational but the mechanism is template-based structured extraction. The elaborate curation prompt likely produces similar results to simpler extraction instructions with less token overhead.

## 10. Key Takeaway

RLabs Memory is the most ambitious auto-extraction system in the survey, and its 7-signal activation architecture represents genuine innovation in retrieval filtering. The anti-trigger mechanism, two-tier headline/content display, and action-items workflow are all practical features worth adopting. However, the system's strengths in extraction automation come with a meaningful tradeoff: no human review gate means quality depends entirely on LLM curation judgment, and the binary signal gates lose the smooth relevance gradients that fusion scoring provides. The most instructive tension with Claudest is philosophical: RLabs bets on rich metadata generated at extraction time to power retrieval, while Claudest bets on the agent's ability to formulate good queries at retrieval time. Our system is better positioned than either -- we have hybrid retrieval (vectors + FTS5 via RRF) like RLabs' multi-signal approach but with continuous scoring, and we have human-in-the-loop curation like Claudest's approval gates but with more structured memory types. The features worth stealing are the ones that complement what we already have: anti-triggers, headline/content tiers, and action-item retrieval modes are all low-effort additions that address real gaps.
