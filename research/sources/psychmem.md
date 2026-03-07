# PsychMem -- Source Analysis

*Phase 13, 2026-02-27. Analysis of muratg98/psychmem.*

## 1. Architecture Overview

**Language:** TypeScript (91.6%), JavaScript (8.4%)
**Key dependencies:**
- `better-sqlite3` / Bun's built-in SQLite (runtime-agnostic adapter)
- `uuid` (memory IDs)
- No embedding model -- retrieval is keyword/Jaccard-based
- No LLM calls in the memory pipeline itself

**Status:** Archived on 2026-02-27. Author concluded "Plugin hooks are the wrong abstraction for memory" and moved development to a successor project called DaySee. Despite being abandoned, the codebase is complete, well-documented, and the architectural ideas are worth studying.

**Storage layout:**
```
~/.psychmem/{agentType}/memory.db    # SQLite (WAL mode)
  sessions              # Session tracking with watermarks
  events                # Raw hook events
  memory_units          # Consolidated memories (STM + LTM)
  memory_evidence       # Evidence linking memories to source events
  retrieval_logs        # Retrieval feedback learning
  feedback              # User feedback (remember/forget/pin/correct)
  schema_version        # Migration tracking
```

**File structure:**
```
src/
  core.ts               # PsychMem class (main entry point)
  types/index.ts        # Full type definitions (MemoryUnit, config, hooks, signals)
  memory/
    context-sweep.ts    # Stage 1: Extract candidates from events (multilingual)
    selective-memory.ts # Stage 2: Score, allocate STM/LTM, detect interference
    patterns.ts         # 15-language importance signal dictionaries
    structural-analyzer.ts # Language-agnostic structural signal detection
  retrieval/index.ts    # Two-level retrieval (index + detail), Jaccard similarity
  storage/
    database.ts         # MemoryDatabase (schema, CRUD, decay, consolidation)
    sqlite-adapter.ts   # Runtime-agnostic SQLite (Node.js + Bun)
  hooks/
    session-start.ts    # Inject memories on session start
    post-tool-use.ts    # Capture tool usage events
    stop.ts             # Extract memories from conversation on stop
    session-end.ts      # Close session
  adapters/
    claude-code/        # Claude Code adapter
    opencode/           # OpenCode adapter
    types.ts            # Adapter interface definitions
  transcript/
    parser.ts           # Transcript parsing (incremental, watermark-based)
    sweep.ts            # Transcript-based extraction
```

## 2. Memory Type Implementation

**Schema:** Single `MemoryUnit` type with rich feature vector:

```typescript
interface MemoryUnit {
  id: string;
  sessionId?: string;
  store: 'stm' | 'ltm';
  classification: MemoryClassification;
  summary: string;
  sourceEventIds: string[];
  projectScope?: string;        // v1.6: project-level scoping

  // Timestamps
  createdAt, updatedAt, lastAccessedAt: Date;

  // Psych-math features (all 0-1)
  recency: number;
  frequency: number;
  importance: number;
  utility: number;
  novelty: number;
  confidence: number;
  interference: number;         // Conflict penalty

  // Computed
  strength: number;             // Overall score
  decayRate: number;            // Lambda for exponential decay

  tags: string[];
  associations: string[];       // Related memory IDs
  status: 'active' | 'decayed' | 'pinned' | 'forgotten';
  version: number;
  evidence: MemoryEvidence[];
}
```

**Types/categories:**
- 8 classifications: `episodic`, `semantic`, `procedural`, `bugfix`, `learning`, `preference`, `decision`, `constraint`
- 2 stores: STM (fast decay, lambda=0.05, ~32h half-life) and LTM (slow decay, lambda=0.01)
- 2 scopes: `user` (always injected, cross-project) and `project` (injected only for matching project)
  - User-level: constraint, preference, learning, procedural
  - Project-level: decision, bugfix, episodic, semantic

**Extraction:** Two-stage pipeline:
1. **Context Sweep** (Stage 1): Detects importance signals via multilingual regex patterns (15 languages) and structural/pragmatic analysis (typography, conversation flow, repetition). Splits conversation into chunks, classifies each, generates candidates.
2. **Selective Memory** (Stage 2): Scores candidates via feature vector, checks for duplicates (Jaccard >= 0.7), detects interference, allocates to STM or LTM, auto-promotes bugfix/learning/decision to LTM.

## 3. Retrieval Mechanism

**Two-level progressive disclosure:**
1. **Index level**: Lightweight list (id, summary, classification, store, strength, estimated tokens, relevance score)
2. **Detail level**: Full MemoryUnit loaded on demand by ID

**Ranking pipeline:**
- Get pool of top memories by strength (up to 200, or 10x requested limit)
- If text query present, rank by text similarity
- Text similarity = `Jaccard(summary, query) * 0.5 + keyword_hit_ratio * 0.5`
- Final score = `textSimilarity * 2.0 * (0.5 + strength * 0.5) + tagScore + recencyBonus`
- Design principle: "text similarity dominates, strength acts as tiebreaker"

**No embedding-based search.** All retrieval is keyword/Jaccard similarity. The author noted BM25+ as a future direction in the README but it was never implemented.

**Scope-aware retrieval (v1.6):**
- User-level memories always included regardless of project
- Project-level memories filtered by `projectScope` match
- Separate `retrieveByScope()`, `searchByScope()` methods

## 4. Standout Feature: Interference Detection via Similarity Scoring

The interference detection mechanism is the most psychologically grounded feature:

**How it works (from `selective-memory.ts`):**
```typescript
detectInterference(candidate, existingMemories): number {
  let interference = 0;
  for (const mem of existingMemories) {
    const topicSimilarity = calculateTextSimilarity(candidate.summary, mem.summary);
    if (topicSimilarity > 0.3 && topicSimilarity < 0.8) {
      // Similar topic but different content = potential conflict
      interference = Math.max(interference, topicSimilarity * 0.5);
    }
  }
  return interference;
}
```

The key insight: memories that are similar (>0.3 Jaccard) but not near-duplicates (<0.8) represent potential interference -- the psychological phenomenon where similar but distinct memories compete and degrade each other. The interference score is stored as a field on the memory and factored into strength calculation with a negative weight (-0.1).

**Limitations:** The implementation is basic -- Jaccard word overlap, not semantic similarity. It catches lexical overlap but misses semantic contradictions (e.g., "prefers Python" vs "switched to Rust" would score low Jaccard overlap despite being a genuine conflict). There's no mechanism to identify WHAT the conflict is or resolve it -- it just applies a penalty.

**Why it matters for us:** Our contradiction detection happens during sleep (edge processing), not at write time. PsychMem's approach of checking at write time is complementary -- catching interference early before it accumulates. The similarity-band heuristic (>0.3 and <0.8) is a useful signal, though we'd want to use embedding distance rather than Jaccard.

## 5. Other Notable Features

1. **Multilingual importance detection (15 languages):** Pattern dictionaries for explicit_remember, emphasis_cue, correction, preference signals across English, Spanish, French, German, Portuguese, Japanese, Chinese, Korean, Russian, Arabic, Hindi, Italian, Dutch, Turkish, Polish. Latin-script patterns use word-boundary regex; CJK/Arabic/Hindi use string.includes(). Ambitious scope, well-structured.

2. **Structural signal analysis (language-agnostic):** Detects importance from typography (ALL CAPS ratio, exclamation density, markdown emphasis, quoted text, code blocks), conversation flow (short reply after long = correction, trigram overlap = repetition, reply >2x median = elaboration), discourse markers (ordered lists, arrows, contrast), and meta signals (proximity to tool errors, file paths, stack traces). Thoughtful and genuinely language-agnostic.

3. **Scope-based memory injection (v1.6):** Clean separation of user-level memories (always injected) from project-level memories (injected only for matching project). The classification-to-scope mapping is hardcoded but sensible: constraints and preferences are universal; decisions and bugfixes are project-specific.

4. **Evidence tracking:** Each memory links to source events via `MemoryEvidence[]` with `contribution` description and `confidenceDelta`. Retrieval logs track whether surfaced memories were actually used, enabling future feedback learning.

5. **Miller's Law constraint:** Max 7 memories extracted per session stop event, referencing Miller's 7 plus/minus 2 working memory limit. A principled cap rather than arbitrary.

## 6. Gap Ratings

| Gap | Rating | Evidence |
|-----|--------|----------|
| Layered Memory | 55% | Explicit STM/LTM with configurable decay rates, promotion thresholds, and auto-promote rules. But no working memory / attention layer. No hierarchical abstraction within LTM. |
| Multi-Angle Retrieval | 25% | Single retrieval path: Jaccard + keyword hits + strength tiebreaker. No embedding search, no FTS, no graph traversal, no RRF fusion. Two-level progressive disclosure (index/detail) is nice UX but not multi-angle. |
| Contradiction Detection | 40% | Interference detection at write time via similarity band (>0.3, <0.8 Jaccard). Stored as interference field with negative scoring weight. But no semantic analysis, no resolution mechanism, no contradiction edges. |
| Relationship Edges | 10% | `associations: string[]` field exists but never populated in the code. No edge types, no graph traversal, no relationship-aware retrieval. |
| Sleep Process | 0% | No offline processing whatsoever. All operations are synchronous within hook lifecycle. |
| Reference Index | 30% | Source event IDs tracked, evidence table with contribution descriptions. Retrieval logs for feedback. But no structured reference resolution or source quality tracking. |
| Temporal Trajectories | 15% | Timestamps on creation/update/access. Recency factor in scoring. Session-based scoping. But no trajectory analysis, no temporal clustering, no evolution tracking. |
| Confidence/UQ | 35% | Confidence field (0-1) on all memories. Evidence with `confidenceDelta`. Interference penalty. But no calibration, no domain-specific accuracy, no confidence decay model beyond the global exponential decay. |

## 7. Comparison with claude-memory

**Stronger:**
- Write-time interference detection -- we detect contradictions only during sleep; PsychMem catches potential conflicts immediately
- Multilingual importance detection -- our extraction is English-only
- Structural signal analysis is genuinely language-agnostic and catches patterns (short correction after long response, repetition, elaboration) that keyword matching misses
- Project-level vs user-level scoping is a clean abstraction we don't have
- Two-level progressive disclosure (index then detail) saves tokens on retrieval
- Evidence tracking with per-event contribution descriptions

**Weaker:**
- No embedding search at all -- Jaccard word overlap is far less capable than our sqlite-vec + cosine similarity
- No FTS5 full-text search
- No RRF hybrid fusion
- No graph/edge system (associations field exists but is unpopulated)
- No sleep/offline processing
- No human-in-the-loop curation (startup_load, recall_feedback)
- No gestalt or compressed representations
- No shadow load or decay sophistication (simple exponential, not power-law)
- Archived/abandoned -- no ongoing development
- The plugin architecture limitation meant memories could only be injected as fake user messages, polluting conversation history

## 8. Insights Worth Stealing

1. **Write-time interference detection with similarity banding** (low effort, high impact): At `remember()` time, check if the new memory falls in the "similar but not duplicate" band (e.g., 0.3-0.8 cosine similarity) against recent memories. If so, flag it for attention or apply a confidence penalty. This catches potential contradictions immediately rather than waiting for sleep. We could implement this with a quick embedding search during `remember()` -- we already compute the embedding for dedup, just extend the threshold check.

2. **Structural signal analysis patterns** (medium effort, medium impact): The specific heuristics are valuable: short reply after long response = likely correction; trigram overlap > 40% = user repeating themselves; reply >2x median length = elaboration deserving attention. These could enhance our auto-capture detection in the stop hook without requiring LLM calls.

3. **Scope-based memory separation** (low effort, medium impact): User-level memories (preferences, constraints) that apply everywhere vs project-level memories that only apply in context. We partially have this via themes/tags, but an explicit scope field with automatic filtering at recall time would be cleaner.

4. **Progressive disclosure retrieval** (low effort, low impact): Return lightweight index first (summary + strength + tokens estimate), let the consumer request full details only for memories they want. Could reduce context consumption in startup_load.

## 9. What's Not Worth It

- **Jaccard-only retrieval**: We already have embedding + FTS5 + RRF, which is strictly superior. The Jaccard approach was a pragmatic choice given no embedding model, not an architectural insight.
- **Multilingual patterns**: 15-language support is impressive engineering but irrelevant for our English-only use case. The architecture (pattern categories with weight-per-signal) is more valuable than the specific dictionaries.
- **Plugin-based hook architecture**: The author themselves concluded this was the wrong abstraction. Memory needs deeper integration than lifecycle hooks provide.
- **Miller's Law cap**: The 7-memory-per-session limit is a nice nod to cognitive science but our system already handles this more flexibly via priority scoring and token budgets.

## 10. Key Takeaway

PsychMem is a thoughtful, well-engineered system that failed not due to its memory model but due to its integration layer (plugin hooks couldn't intercept messages or inject memories cleanly). The psych-math feature vector approach (recency, frequency, importance, utility, novelty, confidence, interference -- all 0-1 with configurable weights) is a clean scoring framework. The most directly transferable idea is write-time interference detection: checking new memories against the similarity band where they're related-but-different, which catches potential contradictions at the moment they're created rather than during periodic sleep processing. The structural signal analysis patterns (correction detection, repetition detection, elaboration detection) are also worth adopting as lightweight heuristics for auto-capture.
