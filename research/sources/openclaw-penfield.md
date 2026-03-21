# OpenClaw Penfield — Source Analysis

*Phase 14, 2026-03-20. Analysis of penfieldlabs/openclaw-penfield.*

## 1. Architecture Overview

**Language:** TypeScript 100%
**Key dependencies:**
- @sinclair/typebox (JSON Schema via TypeScript decorators, tool parameter validation)
- zod (config validation)
- Node.js native fetch (HTTP client)
- OpenClaw plugin SDK (host environment — peer dependency)

**Storage layout:**
All storage is server-side. The plugin is a thin client — no local database, no embeddings, no local files beyond OAuth credentials. Penfield's backend (api.penfield.app) owns the memory store, knowledge graph, vector index, and BM25 index.

```
Local state:
~/.openclaw/extensions/openclaw-penfield/
  credentials.json              # OAuth tokens (0o600 permissions)
```

**File structure:**
```
index.ts                        # Plugin entry, registration, service lifecycle
src/
  config.ts                     # Zod config schema (authUrl, apiUrl, autoAwaken, autoOrient)
  types.ts                      # OpenClaw plugin API type definitions
  types/typebox.ts              # Shared TypeBox schemas (11 memory types, 24 relationship types)
  hooks.ts                      # Lifecycle hooks (auto-awaken, auto-orient, flush config check)
  auth-service.ts               # Background OAuth token refresh (60-min interval)
  api-client.ts                 # HTTP client wrapper (30s timeout, rate limit handling)
  response-compact.ts           # Response compaction (field-stripping for token savings)
  runtime.ts                    # Runtime factory
  store.ts                      # Credential file I/O (atomic write via rename)
  cli.ts                        # CLI registration (penfield login)
  device-flow.ts                # RFC 8628 Device Code Flow + RFC 9700 token rotation
  validation.ts                 # UUID validation, artifact path traversal prevention
  tools/                        # 17 tool implementations (one file each)
persona-templates/              # Recommended OpenClaw workspace file replacements
skill/SKILL.md                  # OpenClaw skill definition with usage guidance
```

Total: ~3,260 lines of TypeScript across 31 files. 22 commits, first commit 2026-01-27, latest 2026-03-20. Published on npm as `openclaw-penfield` v2.0.0.

**Relationship to original OpenClaw Memory (gavdalf/openclaw-memory):** Despite the shared "OpenClaw" branding, this is not a fork of the original. The original was a pure Bash observation/reflection system (~660 lines) with plain markdown files, cron-based capture, and no database. Penfield is a completely different system: a hosted SaaS backend with a TypeScript client plugin. They share the OpenClaw agent platform as a target but have zero code in common and entirely different architectures. "OpenClaw" is the agent framework; "Penfield" is a separate company (Penfield Labs) building a memory service for it.

## 2. Memory Type Implementation

**Schema:** Memories are JSON objects stored via REST API. Each memory has:
- `content` (string, max 10,000 chars)
- `memory_type` (enum, 11 types)
- `importance` (float 0-1, default 0.5)
- `confidence` (float 0-1, default 0.8)
- `source_type` (string, e.g., "direct_input", "conversation", "document_upload")
- `tags` (string array, max 10)
- `metadata` (arbitrary key-value)

**Types/categories:** Eleven memory types, expanded from the original's seven:
| Type | Purpose |
|------|---------|
| fact | Verified durable information |
| insight | Patterns or realizations |
| conversation | Session summaries |
| correction | Fixing prior understanding |
| reference | Source material, citations |
| task | Work items, action items |
| checkpoint | Context state snapshots |
| identity_core | Immutable identity facts (set via portal) |
| personality_trait | Behavioral patterns (set via portal) |
| relationship | Entity connections |
| strategy | Approaches, methods, plans |

Notable additions vs. original: `identity_core`, `personality_trait`, `relationship`, `strategy`, `checkpoint`. The original's `event`, `context`, `preference`, `goal`, `habit`, `rule` are absent — some mapped to existing types (preference -> fact, rule -> strategy), others dropped.

**Extraction:** Manual via `penfield_store` tool calls. No autonomous observation. The SKILL.md provides detailed guidance on when and how to store (context prefixes, importance calibration, "what not to store" antipatterns), but all storage decisions are made by the agent or user. The pre-compaction memory flush (configured in openclaw.json) provides one automatic capture point: when OpenClaw auto-compacts context, it triggers a `penfield_store` call with a session summary. This is the only autonomous capture mechanism.

## 3. Retrieval Mechanism

Hybrid search via the Penfield API (`POST /api/v2/search/hybrid`):

- **BM25** (keyword, default weight 0.4)
- **Vector** (semantic embeddings, default weight 0.4)
- **Knowledge graph** (relationship traversal, default weight 0.2)

The weights are user-tunable per query. Two tool variants expose this:
- `penfield_recall` — full hybrid with configurable weights, graph expansion toggle
- `penfield_search` — semantic-biased preset (vector 0.6, BM25 0.3, graph 0.1)

Client-side post-processing in `response-compact.ts`:
- Filters results below relevance threshold (0.05)
- Strips debug fields (score_breakdown, search_metadata, knowledge_cloud, analyzed_query) — claimed 60-80% token reduction
- Optional client-side sort by creation date
- Optional content truncation with `penfield_fetch` pointer for full text

Additional filters: `memory_types`, `importance_threshold`, `start_date`/`end_date` (ISO 8601), `enable_graph_expansion`.

All ranking, embedding, and indexing happens server-side. The plugin has no visibility into the retrieval algorithm's internals — it sends parameters and receives ranked results. The SKILL.md includes a weight-tuning guide (exact terms -> high BM25, concepts -> high vector, connected knowledge -> high graph), which is the most practically useful retrieval guidance in any system surveyed.

## 4. Standout Feature

**Knowledge graph as a first-class retrieval channel.** Most memory systems either have no graph (ours, Hexis pre-neighborhoods) or have a graph that's separate from retrieval (Letta). Penfield makes graph traversal a weighted component of every search query. The 24 typed relationship types span seven semantic categories (knowledge evolution, evidence, hierarchy, causation, implementation, conversation, sequence), and `penfield_explore` provides explicit graph traversal up to depth 10. The `contradicts`, `supersedes`, and `evolution_of` relationship types directly address knowledge evolution — something most systems handle implicitly or not at all.

The graph is manually constructed (agent calls `penfield_connect`), which means quality depends entirely on the agent's discipline, but the typed relationships provide far richer semantics than auto-extracted entity links.

## 5. Other Notable Features

- **Auto-inject identity + context on every turn**: The `before_agent_start` hook fires on every agent turn, prepending `<penfield-identity>` (personality briefing, cached 30 min) and `<penfield-recent>` (last 20 memories + active topics, cached 10 min) to the system prompt. This creates continuity without explicit recall — the agent always has recent context. Gracefully degrades (skips silently if API is down or auth fails).

- **Context checkpoints for multi-agent handoff**: `penfield_save_context` creates named checkpoints that bundle a description with linked memory IDs (gathered from three sources: explicit IDs, `memory_id: <uuid>` patterns in description text, and hybrid search). `penfield_restore_context` fetches them by name, UUID, or the special keyword "awakening". This is designed for agent-to-agent handoff, not just session persistence.

- **Personality as a service**: The `awaken` tool and portal-based personality configuration (base personality + custom instructions) move identity and personality out of static files and into the memory system. The `persona-templates/` directory provides empty stubs for files Penfield replaces, with detailed comments explaining why each file is now redundant.

- **Response compaction**: The client strips API debug fields before returning results to the agent, claimed 60-80% token reduction. This is an important practical concern — raw API responses from search endpoints are typically verbose with score breakdowns, analyzed queries, and metadata the agent doesn't need.

- **RFC-compliant OAuth with background refresh**: Full RFC 8628 Device Code Flow with RFC 9700 token rotation. Background refresh service checks every 60 minutes with 240-minute buffer. Exponential backoff on network errors (3 retries). Atomic credential writes via temp file + rename.

## 6. Gap Ratings

| Gap | Rating | Evidence |
|-----|--------|----------|
| Layered Memory | 30% | 11 memory types with importance/confidence scores create categorical structure. Checkpoints provide snapshot layering. But no true STM/LTM distinction or processing pipeline — everything goes to the same store at the same level. The `identity_core` and `personality_trait` types suggest an intent toward layering but there's no evidence of differential treatment beyond type filtering. |
| Multi-Angle Retrieval | 65% | Three-channel hybrid search (BM25 + vector + graph) with tunable weights is the most flexible retrieval interface surveyed. Date filtering, type filtering, importance thresholds. Two tool variants (recall vs. search) with different weight presets. Missing: no feedback loop, no adaptive re-ranking, no learned scoring. |
| Contradiction Detection | 25% | `contradicts` and `disputes` relationship types exist for manual tagging, and `supersedes`/`updates` handle knowledge evolution. But detection is fully manual — the agent must recognize and tag contradictions. No automatic contradiction discovery during storage or retrieval. |
| Relationship Edges | 70% | 24 typed relationship types across 7 semantic categories. Explicit `connect`/`disconnect`/`explore` tools. Graph traversal up to depth 10 with strength and type filtering. Graph weight in retrieval scoring. This is the most complete explicit graph implementation surveyed. Missing: automatic relationship extraction, bidirectional traversal awareness. |
| Sleep Process | 0% | No sleep, consolidation, or background processing. Memory state is static once stored. The pre-compaction flush is an emergency capture mechanism, not consolidation. All "processing" is either at write time (server-side) or at query time (search). |
| Reference Index | 40% | Tags (max 10 per memory), memory types, and the knowledge graph collectively provide multiple access paths. Checkpoints create named reference points. The reflect tool surfaces active topics and patterns. No topic extraction, no FTS index beyond what the server provides. |
| Temporal Trajectories | 15% | Date filters on search. Creation timestamps on memories. Reflect tool supports time windows (recent, 1d, 7d, 30d, 90d). But no decay, no temporal trajectory analysis, no evolution tracking beyond manual `evolution_of` relationships. Memories are permanent once stored. |
| Confidence/UQ | 25% | Per-memory confidence score (0-1, default 0.8) and importance score (0-1, default 0.5). Both are set at write time and can be updated manually. No computed confidence, no feedback-driven refinement, no uncertainty quantification from retrieval patterns. |

## 7. Comparison with claude-memory

**Stronger:**
- Knowledge graph with 24 typed relationships integrated into retrieval scoring — our PPR-based graph traversal is structurally similar but uses only 6 edge types and doesn't weight graph results as a separate retrieval channel
- Three-channel hybrid search with user-tunable weights gives the agent explicit control over retrieval strategy per query — our RRF fusion is fixed
- Auto-inject context on every turn (identity + recent memories) provides continuity without explicit recall calls — we rely on `startup_load()` and explicit `recall()`
- Context checkpoints with three-source memory linking (explicit, pattern-extracted, search-based) are a more complete handoff mechanism than anything we offer
- Response compaction (60-80% token reduction on search results) is a practical win we don't do
- The SKILL.md is the best agent-facing usage guide in any system surveyed — concrete weight-tuning advice, memory writing quality guide, importance calibration table, "what not to store" antipatterns

**Weaker:**
- Fully hosted/SaaS — no local processing, no ability to inspect or modify the retrieval algorithm, no transparency into how search actually works
- No sleep/consolidation process — memories are permanent and static once stored, no background quality improvement
- No retrieval feedback loop — no equivalent of our `recall_feedback` mechanism, no learned reranker, no adaptive scoring
- No temporal decay — memories never fade, no biological decay model, no way to express "this was important 6 months ago but isn't anymore" except manual importance updates
- No autonomous capture beyond pre-compaction flush — relies entirely on agent discipline for memory creation (the original OpenClaw Memory's five-layer capture redundancy is completely absent)
- All intelligence is server-side and opaque — impossible to study, tune, or improve the retrieval pipeline
- Vendor lock-in to Penfield Labs — if the service goes down, all memory is inaccessible
- No deduplication mechanism visible in the client — unclear if the server handles it

## 8. Insights Worth Stealing

1. **Graph weight as a tunable retrieval channel** (effort: low, impact: medium). Exposing the graph contribution as a user-tunable weight alongside BM25 and vector weights is elegant. We could surface a `graph_weight` parameter in `recall()` that scales the PPR contribution in our RRF fusion, letting the agent explicitly request "follow relationships more" for connected-knowledge queries.

2. **Auto-inject recent context on every turn** (effort: medium, impact: medium-high). The `before_agent_start` hook pattern — inject last 20 memories + active topics into the system prompt with caching — provides background continuity without requiring explicit recall. We could add an auto-orient mode to `startup_load()` that fetches recent high-importance memories and formats them as session context, cached for N minutes.

3. **Response compaction for token savings** (effort: low, impact: medium). Stripping score breakdowns, search metadata, and debug fields from retrieval results before returning to the agent is a straightforward win. Our `recall()` returns full memory objects including internal fields the agent doesn't need. A compact response mode that returns only id, content, category, relevance, and created_at would reduce context usage.

4. **Weight-tuning guide in agent instructions** (effort: low, impact: low-medium). The SKILL.md's table mapping query types to weight presets (exact terms -> high BM25, concepts -> high vector, connected knowledge -> high graph) is the most actionable retrieval guidance in any system surveyed. We could add similar guidance to our CLAUDE.md snippet — "for known entities use `boost_themes`, for broad topics use larger `limit`" etc.

5. **Named context checkpoints with multi-source memory linking** (effort: medium, impact: low-medium). The save_context pattern of combining explicit memory IDs, regex-extracted IDs from description text, and search-retrieved IDs to create a checkpoint is a clean handoff mechanism. We don't have an equivalent — our closest is manual `remember()` with a session summary.

## 9. What's Not Worth It

- **Fully hosted architecture for a research system.** The entire point of Somnigraph is studying and improving the retrieval pipeline. A hosted service where retrieval is a black box makes this impossible. Penfield's approach makes sense as a product but is antithetical to a research artifact.

- **24 relationship types.** Seven semantic categories with 24 types is impressive in schema definition but creates a classification burden. In practice, most relationships fall into 3-4 types (supports, supersedes, related_to, part_of). The long enum increases the chance of misclassification and inconsistent usage. Our 6 edge types are probably closer to the useful set.

- **Portal-based personality configuration.** Moving personality out of files and into a web portal adds a dependency and removes version control. For single-user research use, a CLAUDE.md file is simpler and more transparent.

- **Pre-compaction flush as the sole automatic capture mechanism.** The memoryFlush config requires aggressive prompt engineering ("MANDATORY", "SYSTEM OVERRIDE", "Ignore all other instructions") to force a single tool call, and only fires on auto-compaction (not manual /compact, /new, or /reset). This is a fragile bridge over a fundamental gap — the system has no real autonomous observation.

## 10. Key Takeaway

OpenClaw Penfield is not a fork of OpenClaw Memory — it's a completely different system sharing only the platform name. Where the original was 660 lines of bash with five-layer autonomous capture and no database, Penfield is a TypeScript client plugin for a hosted SaaS backend with 17 tools, a knowledge graph, and three-channel hybrid search. The architectural philosophy is inverted: the original kept everything local and autonomous; Penfield keeps everything remote and explicit.

The standout contribution is making the knowledge graph a first-class retrieval channel with tunable weights alongside BM25 and vector search. The 24 typed relationships (especially the knowledge evolution types: supersedes, updates, evolution_of) and the agent-facing SKILL.md with concrete weight-tuning advice represent the most thoughtful graph-retrieval integration in any system surveyed. The auto-inject context pattern (identity + recent memories on every turn, cached) and response compaction (60-80% token reduction) are practical engineering wins.

The critical weakness is the fully hosted architecture. All intelligence — embeddings, BM25 indexing, graph traversal, ranking — lives behind an opaque API. There's no sleep process, no feedback loop, no temporal decay, no autonomous capture, and no way to study or improve the retrieval pipeline. For a product, this is a reasonable trade (Penfield handles the complexity). For a research system, it's a non-starter. The most transferable ideas are the ones that can be implemented locally: graph weight as a tunable parameter, auto-orient on session start, and response compaction for token efficiency.
