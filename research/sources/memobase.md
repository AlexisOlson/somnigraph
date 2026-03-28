# Memobase -- Analysis

*Generated 2026-03-25 by Opus agent reading source code at github.com/memodb-io/memobase*

---

## Overview

- **Builder**: MemoDB Inc. (memodb-io), small commercial team
- **Repository**: https://github.com/memodb-io/memobase
- **License**: Apache 2.0
- **Language**: Python (FastAPI server), with Python/Node/Go client SDKs and MCP adapter
- **Maturity**: Active development, version 0.0.40 (August 2025). Cloud service at memobase.io. Dockerized production deployment. Commercially oriented with billing/auth infrastructure built in.
- **Focus**: User profile-based long-term memory for multi-user LLM applications. Explicitly positions itself as "memory for users, not agents" -- the system builds structured user profiles and event timelines, not general-purpose knowledge bases.

Memobase is the most architecturally opinionated of the open-source memory systems. Where Mem0 stores flat facts and Somnigraph stores categorized memories with decay, Memobase stores **structured profiles** organized into topic/subtopic hierarchies (e.g., `interest::movies`, `work::title`, `psychological::goals`). This makes it fundamentally a user-modeling system that happens to use LLMs, rather than a general memory system.

---

## Architecture

### Storage & Schema

- **PostgreSQL** with pgvector extension for embeddings
- **Redis** for profile caching (20-min TTL), buffer queue management, and distributed locking
- **SQLAlchemy ORM** with Alembic migrations

Core tables:

| Table | Purpose |
|-------|---------|
| `users` | User records, keyed by (id, project_id) composite PK |
| `general_blobs` | Raw input data (chat messages, docs, etc.) |
| `buffer_zones` | Pending items awaiting processing (idle/processing/done/failed) |
| `user_profiles` | Extracted profile entries: content + attributes (topic, sub_topic) |
| `user_events` | Session-level event records with embeddings |
| `user_event_gists` | Fine-grained event fragments with individual embeddings |
| `user_statuses` | Status tracking (e.g., roleplay state) |
| `projects` | Multi-tenant project management |
| `billings` | Usage-based billing |

No graph store. No FTS index. Profiles are flat rows with JSON attributes, not graph nodes.

### Memory Types

**User Profiles**: The primary memory type. Structured as `(topic, sub_topic, content)` tuples. Default topics: `basic_info`, `contact_info`, `education`, `demographics`, `work`, `interest`, `psychological`, `life_event`. Projects can add custom topics with optional descriptions and validation rules. Each topic has a configurable cap (`max_profile_subtopics`, default 15). Profiles are the system's answer to "who is this user?"

**User Events**: Session-level summaries with embeddings. Each event stores an `event_tip` (markdown bullet list of what happened), optional `event_tags` (configurable structured tags), and a `profile_delta` (what changed). Events are the system's answer to "what happened?"

**User Event Gists**: Individual bullet points from event_tips, each independently embedded. Added in v0.0.37 specifically to improve temporal/search retrieval -- the coarser event-level embeddings were too diluted for specific lookups.

### Write Path (Buffer-Flush Pipeline)

1. **Insert**: Client sends a `ChatBlob` (list of OpenAI-format messages) to a user's buffer
2. **Buffer accumulation**: Blobs sit in `buffer_zones` until token threshold (default 1024) is exceeded, idle timeout (1 hour), or manual `flush()` call
3. **Entry summary** (LLM call 1): Summarize the buffered chat blobs into a memo string, guided by the project's profile schema and existing user profiles
4. **Parallel processing**:
   - **Profile extraction** (LLM call 2): Extract `(topic, sub_topic, memo)` tuples from the summary, then merge each against existing profiles via a "YOLO merge" (single batched LLM call deciding APPEND/UPDATE/ABORT for all extracted facts simultaneously)
   - **Event tagging** (LLM call 2, parallel): Tag events with configured event tags
5. **Post-processing**: Organize (consolidate when subtopics exceed cap), re-summarize (truncate profiles exceeding `max_pre_profile_token_size`, default 128 tokens)
6. **Persistence**: Add/update/delete profiles in PostgreSQL, append event with gists, invalidate Redis cache

Key design choice: raw blobs are **deleted after processing** by default (`persistent_chat_blobs = False`). Only the extracted profiles and events persist. This is a strong privacy/cost position -- the system doesn't retain conversation history.

The v0.0.40 "YOLO merge" reduced LLM calls from 3-10 per flush to a fixed 3, cutting token costs ~30-50%.

### Retrieval (Context API)

The `context` API packs a user's memory into a prompt-ready string with configurable token budget:

1. **Profile retrieval**: Load all profiles from PostgreSQL (or Redis cache). Optionally filter by relevance to recent chat messages via an LLM call (`filter_profiles_with_chats`). Truncate by token budget with topic priority ordering.
2. **Event gist retrieval**: If chat context provided and embeddings enabled, vector-search event gists by cosine similarity (threshold 0.2, time range 21 days, top-60). Otherwise, fetch most recent gists. Truncate by remaining token budget.
3. **Pack into prompt**: Combine profile section and event section into a template string.

Retrieval is notably simple: no BM25, no RRF fusion, no reranking, no graph traversal. Profile retrieval is a full table scan (cached) with optional LLM-based filtering. Event retrieval is either recency-based or single-vector cosine similarity. The claimed <100ms latency comes from this simplicity -- it's mostly SQL + cache reads.

### Consolidation / Processing

No offline consolidation pipeline. No sleep cycle. No decay. All memory processing happens synchronously during the flush operation. The `organize_profiles` step handles profile growth by merging subtopics when they exceed the cap, but this is triggered by writes, not scheduled.

Background processing exists (`buffer_background.py`) but only for deferred flush operations -- a Redis-based queue with distributed locking to serialize per-user processing. Not an analog to Somnigraph's sleep pipeline.

### Lifecycle Management

- No decay mechanism. Profiles persist indefinitely once created.
- No confidence/priority scores. Profiles track `update_hits` (how many times merged) but this isn't used for retrieval ranking.
- Events have a time-range filter (default 21 days) but profiles have no temporal scoping.
- No feedback loop. No user-provided relevance signals.
- Deletion is manual (delete user, delete profile, delete event).

---

## Key Claims & Evidence

| Claim (from README/docs) | Code Reality |
|---------------------------|--------------|
| "SOTA" on LoCoMo | README links to benchmark results. The system does perform well on temporal questions via event gists. No peer-reviewed evaluation. |
| "<100ms online latency" | Plausible: context API is SQL reads + Redis cache. No LLM calls in the retrieval hot path (unless `filter_profiles_with_chats` is enabled). |
| "Fixed 3 LLM calls per flush" (v0.0.40) | Confirmed: entry_summary + extract_topics + merge_yolo. Plus optional organize/re-summary for overflow cases. |
| "Memory for User, not Agent" | Accurate: the entire schema is user-centric. No task memory, no agent state, no knowledge base. |
| "Controllable Memory" via profile config | Confirmed: projects define topic schemas with descriptions, validation rules, and update instructions. Strict mode restricts extraction to defined topics only. |
| "Batch-Process" via buffer | Confirmed: buffer accumulates blobs, flushes when threshold or timeout reached. |
| "No agents in the system" (re: cost) | Mostly true: no autonomous loops. Each flush is a fixed pipeline. But `filter_profiles_with_chats` adds an optional LLM call at retrieval time. |

---

## PERMA Benchmark Results

| Setting | MCQ Acc. | BERT-F1 | Memory Score | Context Token | Turn=1 | Turn<=2 |
|---------|----------|---------|-------------|---------------|--------|---------|
| Clean Single | 0.733 | 0.781 | 1.86 | 1033.3 | 0.504 | **0.830** |
| Clean Multi | 0.694 | 0.793 | 1.71 | 1033.2 | **0.331** | **0.656** |
| Noise Single | 0.683 | 0.772 | 1.87 | 1061.0 | **0.551** | 0.787 |
| Noise Multi | 0.643 | 0.796 | 1.72 | 1038.0 | 0.274 | 0.580 |
| Style Single | 0.700 | -- | -- | 1031.0 | **0.551** | 0.816 |
| Style Multi | 0.707 | -- | -- | 1034.7 | 0.274 | 0.573 |

**Pattern analysis**:

1. **Remarkably consistent context tokens (~1033)**: The profile-based architecture produces near-identical context sizes regardless of setting. This is because profiles are a fixed-size summary -- they don't grow with conversation length or noise. This is both a strength (predictable cost) and a weakness (the system can't "look harder" when it needs to).

2. **Strong Turn<=2 in Clean Single (0.830)**: Second-best overall. The profile system excels when preferences are stated clearly in a single domain and the system has had one interaction to build the profile. This is its sweet spot.

3. **Good multi-domain Turn=1 (0.331 Clean)**: Tied with MemOS for best. The structured profile schema gives Memobase a structural advantage on cross-domain synthesis -- if a preference was extracted into the right topic/subtopic, it's retrievable regardless of domain context.

4. **Noise degrades performance unlike MemOS**: MCQ drops from 0.733 to 0.683 in single-domain. The extraction pipeline's LLM calls are susceptible to noisy input -- noise that confuses the extraction prompt reduces profile quality. MemOS's more robust retrieval (presumably) handles this better.

5. **Multi-domain Turn=1 drops significantly with noise (0.331 -> 0.274)**: The extraction quality degradation compounds across domains. When noise causes mis-extraction in one domain, cross-domain synthesis fails.

6. **BERT-F1 is paradoxically stable**: 0.772-0.796 across all settings. The profile structure provides a floor -- even when the wrong profiles are retrieved, they tend to be topically adjacent.

---

## Relevance to Somnigraph

### What Memobase does that Somnigraph doesn't

1. **Structured profile extraction**: Memobase's topic/subtopic schema with LLM-mediated extraction and merge is a real-time entity resolution system for user attributes. Somnigraph has no write-path quality gating or structured extraction.
2. **Multi-user support**: Full multi-tenancy with project isolation, per-user profiles, billing. Somnigraph is single-user by design.
3. **Write-path deduplication**: The merge step (APPEND/UPDATE/ABORT) prevents duplicate profile entries. Somnigraph relies on sleep-time pairwise classification for dedup.
4. **Configurable extraction schema**: Projects define what to extract. Somnigraph's category system is less prescriptive.
5. **Event gist decomposition**: Breaking event summaries into independently-embedded gist fragments improves temporal retrieval granularity.

### What Somnigraph does better

1. **Retrieval sophistication**: Somnigraph's hybrid BM25 + vector + RRF + learned reranker vs. Memobase's single-vector cosine similarity (or bare SQL). Night and day.
2. **Feedback loop**: Somnigraph has explicit per-query utility ratings feeding into EWMA + UCB exploration. Memobase has no feedback mechanism.
3. **Graph-conditioned retrieval**: PPR expansion, betweenness centrality, novelty-scored adjacency. Memobase has no graph at all (it's on their roadmap as "Social Graph").
4. **Offline consolidation**: Sleep pipeline (NREM edge creation, REM gap analysis) vs. no offline processing.
5. **Decay and lifecycle**: Per-category exponential decay with configurable half-lives vs. no decay at all.
6. **BM25/FTS**: Memobase has no full-text search. Vocabulary-gap retrieval failures would be even worse than Somnigraph's 88%.

---

## Worth Stealing (ranked)

1. **Event gist decomposition**: Breaking session summaries into individually-embedded fragments is a simple, high-value idea. Somnigraph's sleep pipeline could produce gist-level embeddings for event memories to improve temporal retrieval granularity. Low implementation cost, clear retrieval benefit.

2. **Buffer-flush pattern for write batching**: Accumulating inputs and processing in batch reduces LLM call overhead. Somnigraph currently processes memories individually. A buffer that batches `remember()` calls before sleep would reduce extraction cost.

3. **Profile-aware context retrieval filtering**: The `filter_profiles_with_chats` step uses an LLM to select which profile topics are relevant to the current conversation. This is a lightweight form of query-dependent retrieval that could inform Somnigraph's reranker -- a "topic relevance" feature extracted from the query.

4. **Schema-guided extraction**: The topic/subtopic hierarchy with descriptions and update instructions produces cleaner extractions than open-ended extraction. Somnigraph's category system could benefit from similar structured guidance during sleep-time processing.

---

## Not Useful For Us

1. **Multi-tenancy/billing infrastructure**: Somnigraph is a single-user research system. The project/billing/auth layers are product concerns, not research concerns.

2. **Profile-only memory model**: Memobase's bet that all memory reduces to user profiles is too narrow for Somnigraph. We need to store procedural knowledge, episodic experiences, and relational context -- not just "who is the user."

3. **No-retrieval-intelligence approach**: Memobase's context API is essentially "dump all profiles + recent events." This works when you have 50-200 profile entries but doesn't scale to Somnigraph's memory volume (thousands of memories with complex relevance signals). The simplicity is their feature, but it's not transferable.

4. **Chinese language support**: Dual-language prompts (en/zh) are product-driven, not research-relevant.

5. **Redis caching layer**: Somnigraph's SQLite is single-process; Redis adds complexity without benefit for our deployment model.

---

## Connections

### Mem0 (closest comparator)

Both extract facts from conversations and store them for later retrieval. Key differences:

| Dimension | Memobase | Mem0 |
|-----------|----------|------|
| Memory structure | Structured profiles (topic/subtopic) | Flat facts (text + embedding) |
| Write path | Buffer-flush batch processing | Incremental per-message-pair |
| Retrieval | Profile dump + optional vector search | Vector similarity search |
| Graph | None (roadmapped) | Neo4j in Mem0g variant |
| Temporal | Event gists with embeddings | Conversation summary refresh |
| LLM cost per write | Fixed 3 calls per flush | Variable (extract + N tool calls) |
| Schema control | Configurable topic hierarchy | None (open extraction) |
| PERMA MCQ (Clean Single) | 0.733 | -- |
| LoCoMo (claimed) | "SOTA" | 66.88% J (Mem0), 68.44% J (Mem0g) |

Memobase's structured extraction is more controlled but less flexible. Mem0's flat facts capture more diverse information but lack organization. The "YOLO merge" (v0.0.40) that batches all merge decisions into one LLM call is a direct response to Mem0's per-fact tool-calling overhead.

### MemOS

Both appear in PERMA benchmarks. MemOS achieves higher MCQ accuracy (0.811 vs 0.733 in Clean Single) and better noise robustness, but Memobase matches on multi-domain Turn=1 (0.331). MemOS's MemCube abstraction is more architecturally ambitious; Memobase is more pragmatic.

### Kumiho

Kumiho's graph-native approach with AGM belief revision is the antithesis of Memobase's profile-only model. Where Kumiho represents memory as a knowledge graph with formal consistency guarantees, Memobase represents memory as flat profile entries with LLM-mediated merge. The approaches trade off: Memobase is simpler and faster at retrieval; Kumiho handles contradictions and multi-hop reasoning.

### A-Mem

A-Mem's Zettelkasten-inspired autonomous organization contrasts with Memobase's schema-prescribed extraction. A-Mem lets the LLM decide structure; Memobase enforces it via topic definitions.

---

## Summary Assessment

Memobase is a well-engineered product that makes a specific architectural bet: user memory is best represented as structured profiles organized into topic/subtopic hierarchies, plus a timeline of events. This bet pays off for its target use case -- personalized chatbots where you need to remember user preferences, demographics, and interaction history. The fixed-schema extraction produces clean, predictable profiles, and the buffer-flush pattern keeps LLM costs bounded. The system is production-ready in a way that most research memory systems are not (multi-tenancy, billing, caching, background processing, distributed locking).

The fundamental limitation is retrieval simplicity. With no BM25, no learned reranker, no graph traversal, and no feedback loop, Memobase's retrieval is either "dump all profiles" or "single vector search on event gists." This works when the profile is small enough to fit in the context window (their ~1033-token budget suggests they typically pack everything), but it means the system cannot selectively retrieve relevant memories from a large memory store. The PERMA results confirm this: strong when the answer is in the profile (Turn<=2 Clean Single: 0.830) but degrading when cross-domain synthesis or noise resistance is needed.

For Somnigraph, the most transferable ideas are event gist decomposition (individual embeddings for session summary fragments) and schema-guided extraction (structured topic definitions that constrain what the LLM extracts). The profile-only memory model and retrieval-free architecture are not transferable -- they solve a different problem than ours. Memobase is the clearest example of a system optimized for the "user profile" use case at the expense of general memory retrieval, which makes it a useful reference point for understanding the trade-off space.
