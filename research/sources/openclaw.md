# OpenClaw Memory (Total Recall) — Source Analysis

*Phase 13, 2026-02-27. Analysis of gavdalf/openclaw-memory (total-recall).*

## 1. Architecture Overview

**Language:** Shell/Bash 100%
**Key dependencies:**
- Any OpenAI-compatible LLM API (default: Gemini 2.5 Flash via OpenRouter)
- jq (JSON processing)
- cron (scheduling)
- inotify-tools (Linux reactive watcher, optional)
- curl (API calls)
- md5sum (deduplication)

**Storage layout:**
```
memory/
  observations.md              # Main observation log (plain text)
  observation-backups/         # Reflector safety backups (last 10)
  .observer-last-run          # Timestamp metadata
  .observer-last-hash         # MD5 deduplication hash
  archive/
    observations/             # Archived items by date (YYYY-MM-DD.md)
  dream-logs/                 # Nightly consolidation reports
  .dream-backups/             # Pre-run safety copies
research/
  dream-cycle-metrics/daily/  # JSON performance metrics
```

**File structure:**
```
scripts/
  observer-agent.sh           # 15-min conversation compression
  reflector-agent.sh          # Threshold-triggered consolidation
  session-recovery.sh         # Missed session capture
  observer-watcher.sh         # Linux inotify reactive trigger
  dream-cycle.sh              # Nightly consolidation (8 commands)
  setup.sh                    # One-command installation
  _compat.sh                  # Cross-platform helpers
config/                       # Configuration templates
prompts/                      # LLM system prompts
schemas/                      # Validation schemas
templates/                    # Output templates
```

Total: ~660 lines of bash across all scripts. 18 commits, 58 stars (highest star count of the three repos in this batch).

## 2. Memory Type Implementation

**Schema:** Plain-text markdown entries in `observations.md`. No database. Each observation is a text block under a date header with HTML comment metadata:

```markdown
## 2026-02-27

<!-- type:fact confidence:0.8 source:explicit -->
The API endpoint changed from v2 to v3 last Tuesday.

<!-- type:preference confidence:0.6 source:implicit -->
User prefers TypeScript over JavaScript for new projects.
```

**Types/categories:** Seven memory types with configurable TTLs:
| Type | TTL |
|------|-----|
| event | 14 days |
| fact | 90 days |
| context | 30 days |
| preference | 180 days |
| goal | 365 days |
| habit | 365 days |
| rule | never expires |

**Extraction:** Fully automatic via the Observer agent. Every 15 minutes (cron), the observer reads recent session transcripts (JSONL), filters to user/assistant messages within the lookback window, deduplicates via MD5 hash against last run, sends to LLM for compression, and appends results to observations.md. Morning lookback is extended to 480 minutes to catch overnight gaps. The Reactive Watcher (Linux) can also trigger after 40+ JSONL writes with a 5-minute cooldown.

## 3. Retrieval Mechanism

Retrieval is essentially "the LLM reads the file." Observations.md is loaded into the agent's context at session startup. There is no search pipeline, no ranking, no embedding-based retrieval. The Dream Cycle's multi-hook feature (WP0) generates 4-5 alternative semantic phrasings for archived items, which are stored as search aids, but the retrieval itself depends on the LLM's ability to find relevant content in a flat text file.

The Reflector agent keeps observations.md within token budget by consolidating when it exceeds 8,000 words (40-60% reduction), ensuring the file stays context-window-sized.

This is a radically different approach: rather than building a retrieval system, keep the total memory small enough to load entirely into context. The Dream Cycle archives stale items to keep the working set manageable.

## 4. Standout Feature

**Five-layer redundancy for autonomous observation.** The system's core innovation is not in how it stores or retrieves memories but in how it captures them without any human intervention. Five independent mechanisms ensure no conversation data is lost:

1. **Observer** (cron, every 15 min) — steady-state compression
2. **Reflector** (auto-trigger at 8K words) — consolidation
3. **Session Recovery** (on /new or /reset) — gap filling
4. **Reactive Watcher** (inotify, Linux) — real-time trigger
5. **Pre-compaction Hook** (memoryFlush) — emergency capture

Each layer compensates for the others' failure modes. The cron observer might miss rapid sessions; the reactive watcher catches them. The session recovery catches what both miss during resets. The pre-compaction hook is a last resort before context loss. This defense-in-depth approach to capture is more thoroughly engineered than any other system surveyed.

## 5. Other Notable Features

- **Dream Cycle (hippocampus-inspired nightly consolidation)**: Eight-command pipeline that runs at 3am — preflight, archive, chunk, update-observations, write-log, write-metrics, validate, rollback. Achieves 74.9% token reduction. Classifies observations by type and impact, chunks related items (3+), applies future-date protection for reminders/deadlines, and generates multi-hook semantic search references for archived items.

- **Confidence scoring with source attribution**: Each observation gets a confidence score (0.0-1.0) and source classification (explicit, implicit, inference, weak, uncertain). Higher-confidence items survive longer in the retention pipeline. This is LLM-assigned, not computed, but provides useful signal for the Dream Cycle's archival decisions.

- **Near-zero cost operation**: ~$0.05-0.15/month using Gemini 2.5 Flash via OpenRouter. ~$0.003 per Dream Cycle execution. The entire system runs on cron jobs and cheap LLM calls. No vector database, no embedding API, no infrastructure.

## 6. Gap Ratings

| Gap | Rating | Evidence |
|-----|--------|----------|
| Layered Memory | 35% | Seven memory types with TTLs create implicit layers. Dream Cycle archives create a temporal hierarchy (active observations vs. archived). But no true STM/LTM processing pipeline — it's more like a filing system with expiration dates. |
| Multi-Angle Retrieval | 10% | No retrieval system. Multi-hook phrasings (4-5 per archived item) improve discoverability but rely on LLM in-context search, not a retrieval engine. |
| Contradiction Detection | 5% | The Reflector "removes superseded information" via LLM, which implicitly handles some contradictions during consolidation, but there is no explicit contradiction detection mechanism. |
| Relationship Edges | 0% | No relationship tracking between observations. Each observation is an independent text block. |
| Sleep Process | 55% | The Dream Cycle is explicitly hippocampus-inspired nightly consolidation with classification, chunking, archival, and metric tracking. Achieves 74.9% token reduction. This is a real sleep process, albeit simpler than our NREM/REM pipeline. |
| Reference Index | 15% | Multi-hook semantic phrasings provide alternative access paths to archived content. Memory types provide categorical indexing. No topic extraction or structured index. |
| Temporal Trajectories | 20% | TTL-based decay per memory type (Ebbinghaus-inspired). Date headers on observations. Future-date protection for reminders. But no trajectory analysis or evolution tracking. |
| Confidence/UQ | 30% | Per-observation confidence scores (0.0-1.0) with five source classifications. LLM-assigned, not computed from feedback. Influences retention but not retrieval ranking. |

## 7. Comparison with claude-memory

**Stronger:**
- Fully autonomous capture — no human intervention required for memory creation. Our system requires explicit `remember()` calls or auto-capture during sessions. Total Recall watches and compresses continuously.
- Defense-in-depth capture architecture — five independent mechanisms ensure no data loss. We have no equivalent redundancy for memory capture.
- Near-zero operational cost (~$0.10/month vs. our OpenAI embedding API costs)
- Dream Cycle achieves impressive compression (74.9%) with a well-structured nightly pipeline
- Simpler mental model — "everything goes into one file, LLM reads it" is easier to reason about than database + embeddings + RRF
- Memory type TTLs (never/365/180/90/30/14 days) are a cleaner expression of retention policy than our decay_rate parameter

**Weaker:**
- No retrieval system at all — relies entirely on LLM reading a flat file. Fundamentally does not scale beyond context window size.
- No vector search, no embeddings, no semantic similarity
- No relationship tracking between memories
- No human feedback mechanism (no recall_feedback equivalent)
- No structured query capability
- Cannot handle more than ~8K words of active memory (Reflector threshold)
- Bash scripting limits extensibility and makes complex logic fragile
- Linux-centric (inotify watcher unavailable on macOS/Windows)
- Confidence is LLM-assigned, not refined through usage feedback
- No gestalt or summary layer — just raw observations and archived observations

## 8. Insights Worth Stealing

1. **Defense-in-depth capture architecture** (effort: medium, impact: high). The five-layer redundancy model for memory capture is the most interesting idea here. We currently rely on explicit `remember()` calls and auto-capture patterns. Adding even one autonomous observation layer — a background process that reads session transcripts and proposes memories — would reduce the "forgot to remember" failure mode. The observer's MD5 deduplication and adaptive lookback windows (morning vs. standard) are practical details worth copying.

2. **Memory type TTLs as retention policy** (effort: low, impact: medium). Expressing retention as "facts live 90 days, events live 14 days, rules never expire" is more intuitive than abstract decay rates. We could map our decay_rate parameter to named TTL tiers, making the system easier to reason about without changing the underlying math.

3. **Multi-hook semantic phrasings for archived content** (effort: low, impact: low-medium). Generating 4-5 alternative search terms per archived observation is a cheap way to improve recall for keyword-based search. We could apply this during sleep processing: when a memory enters low-activity dormancy, generate alternative query hooks and store them as searchable metadata.

## 9. What's Not Worth It

- The "no database" philosophy. Plain-text markdown works at the scale of a single observations.md file but creates a hard ceiling. Once you need to search across thousands of memories, you need structured storage. This is a design choice that trades scalability for simplicity.
- Bash as an implementation language for anything beyond simple orchestration. The 660-line constraint is a feature (forced simplicity), but it also means no complex data structures, no type safety, and fragile error handling.
- Relying on LLM in-context retrieval as the sole search mechanism. This works for small memory sets but fails precisely when memory becomes most valuable — when there's too much to fit in context.

## 10. Key Takeaway

Total Recall is the most opinionated system in this batch: it bets entirely on autonomous capture and in-context retrieval, rejecting databases, embeddings, and structured search in favor of "watch everything, compress aggressively, let the LLM read the result." The five-layer capture redundancy is genuinely impressive engineering within 660 lines of bash, and the Dream Cycle's hippocampus-inspired nightly consolidation achieves real compression (74.9%). The convergent design observation is striking: this system independently arrived at confidence scoring, memory type classification, temporal decay (via TTLs), and sleep-like consolidation — the same problems every memory system encounters — using only bash, cron, and cheap LLM calls. The approach has a hard scaling ceiling (context window size), but for agents with modest memory needs, it demonstrates that the core problems of AI memory can be addressed without any of the infrastructure we typically assume is necessary. The 58-star count (highest in this batch) suggests the simplicity resonates with practitioners.
