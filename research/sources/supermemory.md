# Supermemory

- **Source**: https://github.com/supermemoryai/supermemory (17k stars)
- **Type**: Commercial product with open-source SDK/client/visualization (MIT). Core extraction engine is proprietary (hosted API).
- **Authors**: Supermemory Inc.
- **Category**: System (memory layer for AI agents/apps)
- **Analyzed**: 2026-03-25

## Overview

Supermemory is a commercial "memory and context layer for AI" — a hosted API with open-source client SDKs, MCP server, framework integrations, and a graph visualization package. The value proposition is a turnkey memory system: you send content in, it extracts memories, builds user profiles, and returns relevant context. Claims #1 on LongMemEval, LoCoMo, and ConvoMem benchmarks.

The open-source repo (Turbo monorepo, TypeScript throughout) contains the client surface — SDKs, MCP server, Vercel/Mastra/OpenAI integrations, a React memory-graph visualization component, and Zod validation schemas that reveal the data model. The actual extraction, chunking, embedding, and relationship-building logic lives behind the hosted API at `api.supermemory.ai`. This means the code we can inspect shows the data model and retrieval interface, but the intelligence (extraction quality, contradiction resolution, forgetting logic) is a black box.

Maturity is high on the integration side (well-typed schemas, multiple framework wrappers, MCP server with OAuth, multi-provider connectors for Google Drive/Notion/OneDrive) and unknown on the core engine side. The 17k GitHub stars reflect a polished developer experience more than inspectable architecture.

## Architecture

### Storage & Schema

The data model (from `packages/validation/schemas.ts`) has four core entities:

1. **Documents** — raw content (text, PDF, URL, image, video). Schema includes `contentHash` for dedup, `tokenCount`/`wordCount`/`chunkCount` stats, `processingMetadata` with step-level timing, dual embedding fields (`summaryEmbedding` + `summaryEmbeddingNew` suggesting model migration support), and a six-stage status enum: `queued → extracting → chunking → embedding → indexing → done`.

2. **Chunks** — semantic segments of documents. Position-ordered within a document. Also carry dual embeddings plus a `matryokshaEmbedding` field (likely Matryoshka/variable-dimension embeddings for efficiency). Type can be `text` or `image`.

3. **MemoryEntry** — extracted facts. Key fields: `isStatic` (permanent vs dynamic), `isLatest`/`version`/`parentMemoryId`/`rootMemoryId` (version chain), `isForgotten`/`forgetAfter`/`forgetReason` (lifecycle), `isInference` (derived vs directly extracted), `sourceCount`, `memoryRelations` (typed: updates/extends/derives). Also dual embeddings.

4. **Spaces** — container-tag-scoped isolation units with `contentTextIndex` (knowledge base) and role-based membership (`owner/admin/editor/viewer`). Multi-tenant by design.

Relationship between documents and memories is many-to-many via `MemoryDocumentSource` (with `relevanceScore` 0-100).

Backend appears to be PostgreSQL (Drizzle ORM, Hyperdrive connection via Cloudflare) with HNSW vector indexing (claimed, not visible in code). Processing runs as Cloudflare Workflows with cron triggers every 4 hours for connector imports.

### Memory Extraction & Decomposition

**Not visible in the open-source code.** The extraction pipeline is entirely server-side. What we can infer:

- Content goes through the six-stage pipeline (extract → chunk → embed → index)
- Extraction produces `MemoryEntry` objects tagged as `isStatic` or dynamic
- The `isInference` flag suggests some memories are derived rather than directly extracted
- `entityContext` parameter on document creation hints at guided extraction ("Context guidance for memory extraction")
- The architecture doc describes three relationship types discovered during indexing: Updates, Extends, Derives

The MCP `memory` tool description says "Extracts facts, preferences, project context — not noise" but the extraction criteria are not visible. The `shouldLLMFilter` org setting and `filterPrompt`/`includeItems`/`excludeItems` fields in `OrganizationSettings` suggest configurable write-path quality gating via LLM.

### Semantic Relationships

Three typed relationships between memories:

- **Updates**: supersession — new information replaces old ("prefers React 17" → "prefers React 18"). Tracked via version chain (`parentMemoryId`, `rootMemoryId`, `isLatest`).
- **Extends**: enrichment — adds context without replacing ("likes TypeScript" ← "completed advanced TypeScript course").
- **Derives**: inference — synthesized from pattern analysis ("reads ML papers daily" + "asks about neural networks" → "is an ML researcher"). Flagged via `isInference`.

Relationship discovery happens during the "indexing" stage. The graph visualization (`memory-graph` package) renders these as edges between document nodes and memory nodes, with similarity-weighted visual properties. The graph API exposes viewport-based pagination with spatial coordinates (`x`, `y` per document) suggesting force-directed layout computed server-side.

### Retrieval

Two search modes visible in the API:

- **Semantic**: pure vector similarity (cosine on normalized embeddings)
- **Hybrid**: semantic + keyword (described in docs, specific BM25/keyword implementation not visible)

The `/v4/search` endpoint accepts `chunkThreshold` (0-1 sensitivity), `searchMode`, metadata filters with AND/OR logic, and container-tag scoping. The `/v4/profile` endpoint returns a combined view: static profile + dynamic context + optional search results — essentially a user summary + RAG in one call (~50ms claimed).

Client-side deduplication in `tools-shared.ts` is exact-string-match only (priority: static > dynamic > search results). No learned reranking visible — results appear to be ranked by similarity score, recency, and static/dynamic priority.

### Consolidation / Processing

No sleep-like offline consolidation visible. Processing is write-path: content enters → gets chunked → embedded → relationships indexed → done. The `forgetAfter` field on memories and the architecture doc's mention of "automatic forgetting" suggest temporal expiry, but no evidence of periodic consolidation, gap analysis, or quality improvement over time.

The Conversations API (`/v4/conversations`) supports "smart diffing and append detection" — likely incremental processing when the same conversation ID is updated with new messages.

### Lifecycle Management

- **Versioning**: memories form chains via `parentMemoryId`/`rootMemoryId`/`isLatest`. Query default returns latest version only.
- **Forgetting**: soft delete (`isForgotten` flag) with optional `forgetAfter` date and `forgetReason`. Forget via MCP tries exact content match first, falls back to semantic search (0.85 threshold).
- **Temporal expiry**: `forgetAfter` field enables date-based auto-forgetting (e.g., "exam tomorrow" expires after the date).
- **No decay**: no evidence of continuous relevance decay. Memories are either current or forgotten, with no intermediate degradation.

## Key Claims & Evidence

| Claim | Source | Evidence in Code |
|-------|--------|-----------------|
| #1 on LongMemEval (81.6%) | README | No benchmark code in repo; MemoryBench framework referenced but not included |
| #1 on LoCoMo | README | Same — claimed, not verifiable from open source |
| #1 on ConvoMem | README | Same |
| "Living knowledge graph" | Architecture doc | Schema supports it (Updates/Extends/Derives relations, version chains). Actual graph construction is server-side |
| <50ms profile latency | README, architecture doc | Client code shows single API call; plausible for cached profiles |
| "Automatic forgetting" | README | `forgetAfter` field exists; expiry logic is server-side |
| "Resolves contradictions automatically" | README | `Updates` relationship type exists; actual resolution logic not visible |
| "10:1 compression ratio" | Architecture doc | Consistent with PERMA results (~94 context tokens from full conversations) |
| Semantic chunking | Architecture doc | `Chunk` schema with position ordering exists; chunking strategy is server-side |
| HNSW vector indexing | Architecture doc | Not visible in code; plausible given Cloudflare AI / pgvector backend |

## PERMA Benchmark Results

| Setting | MCQ Acc. | BERT-F1 | Memory Score | Context Tokens | Completion | Turn=1 | Turn≤2 |
|---------|----------|---------|-------------|----------------|------------|--------|--------|
| Clean Single | 0.655 | 0.799 | 1.84 | **94.3** | 0.804 | 0.501 | 0.804 |
| Clean Multi | 0.656 | 0.803 | 1.72 | **92.4** | 0.675 | 0.248 | 0.554 |
| Noise Single | 0.674 | 0.796 | 1.96 | **92.6** | 0.806 | 0.501 | 0.811 |
| Noise Multi | 0.637 | 0.803 | 1.75 | **90.7** | 0.675 | 0.248 | 0.612 |
| Style-aligned Single | 0.671 | — | — | **119.6** | — | 0.527 | 0.801 |
| Style-aligned Multi | 0.586 | — | — | **116.1** | — | 0.255 | 0.541 |

### Pattern Analysis

**Extreme compression, competitive accuracy.** Supermemory's ~94 context tokens represent >99% compression of raw conversation content. For comparison, MemOS uses ~6400 tokens, Memobase ~1033, and full-context baselines use the entire conversation. Despite this, MCQ accuracy (0.655-0.674 single-domain) is competitive — not top-tier, but not disqualified either. The compression-to-accuracy tradeoff is the most aggressive in the PERMA field.

**Multi-domain collapse.** Turn=1 accuracy drops from ~0.50 (single) to ~0.25 (multi-domain) — a 50% relative degradation. This is the weakest multi-domain Turn=1 performance in the PERMA benchmark (MemOS achieves 0.306, Memobase 0.504). The extreme compression appears to discard cross-domain linkage information that would connect preferences across domains.

**Noise resilience.** Noise slightly *improves* performance (0.655 → 0.674 single, consistent across settings). This is unusual and suggests the extraction pipeline may use noise-like signals (hedging, uncertainty language) as positive indicators — or more likely, that noise-bearing conversations contain more explicit preference statements that survive the aggressive compression.

**Profile-based retrieval, not search.** The architecture returns a pre-built profile (static facts + dynamic context) plus optional search results in one call. This explains both the extreme compression (the profile is a fixed-size summary) and the multi-domain weakness (a flat profile doesn't maintain cross-domain associative structure).

**Style-aligned degradation.** Style-aligned multi-domain (0.586 MCQ, 0.541 Turn≤2) is Supermemory's weakest setting. When users express preferences indirectly through communication style rather than explicit statements, the extraction pipeline loses signal.

## Relevance to Somnigraph

### What Supermemory does that Somnigraph doesn't

1. **User profile generation.** Static/dynamic profile maintained as a first-class concept, returned in ~50ms. Somnigraph has no equivalent — context is assembled per-query from retrieval results, not precomputed.

2. **Multi-user / multi-tenant.** Container tags, organizations, role-based access, API key authentication. Somnigraph is single-user by design.

3. **Write-path memory extraction.** Content is automatically decomposed into memories with relationship typing at ingest time. Somnigraph stores what the user explicitly tells it to store; relationship detection happens offline during sleep.

4. **Memory versioning with explicit supersession.** The Updates/Extends/Derives relationship types and version chains provide a clean model for preference evolution. Somnigraph detects contradictions during sleep but doesn't maintain version chains.

5. **Temporal expiry.** `forgetAfter` enables date-based auto-forgetting. Somnigraph's decay is continuous (exponential with configurable half-lives) but not event-triggered.

6. **Document ingestion pipeline.** PDFs, images (OCR), videos (transcription), URLs, code (AST-aware chunking) — a full content processing stack. Somnigraph stores text memories only.

7. **Connector ecosystem.** Google Drive, Gmail, Notion, OneDrive, GitHub with real-time webhooks. Somnigraph has no external data integration.

### What Somnigraph does better

1. **Learned reranking.** 26-feature LightGBM reranker trained on 1032 real queries (NDCG=0.7958) vs. apparent similarity-only ranking. The reranker is the single largest retrieval quality lever Somnigraph has.

2. **Explicit feedback loop.** Per-query utility ratings (0-1 float + durability), EWMA aggregation, UCB exploration bonus, Hebbian co-retrieval PMI. Supermemory has PostHog analytics tracking but no visible retrieval feedback mechanism.

3. **Offline consolidation.** Three-phase sleep pipeline (NREM: pairwise classification, edge creation, merge/archive; REM: gap analysis, question generation, taxonomy). Supermemory processes at write time only.

4. **Graph-conditioned retrieval.** PPR-based expansion, novelty-scored adjacency, betweenness centrality as reranker feature. Supermemory's graph relationships exist but their role in retrieval is unclear — the architecture doc mentions "relationship expansion" but no graph traversal algorithm is visible.

5. **Hybrid BM25 + vector search with Bayesian-optimized fusion.** RRF with k=14 specifically tuned. Supermemory offers "hybrid" mode but implementation details are hidden.

6. **Transparent, inspectable architecture.** Every design decision is documented. Supermemory's extraction and relationship-building logic is a closed-source black box.

7. **LoCoMo QA: 85.1%.** Supermemory claims #1 on LoCoMo but doesn't publish the number; Somnigraph's 85.1% with Opus judge is documented with methodology.

## Worth Stealing

Ranked by expected value for Somnigraph:

1. **Static/dynamic memory distinction with precomputed profiles.** A cached user summary (stable facts + recent context) available at ~constant cost per query would reduce latency for common cases. Implementation: a `profile` memory category maintained by sleep, with static facts extracted during NREM and dynamic context refreshed more frequently. Lower priority than retrieval quality, but a clean product feature.

2. **Explicit memory versioning.** The `parentMemoryId → rootMemoryId → isLatest` chain is a more structured model than Somnigraph's contradiction edge detection. During sleep, when a contradiction is detected, creating a version chain (marking the old memory as `isLatest=false`) rather than just adding a contradiction edge would make temporal queries more tractable. Could extend the existing merge/archive logic.

3. **`forgetAfter` for event-triggered expiry.** Complementary to continuous decay. A memory with `forgetAfter: "2026-03-26"` for "meeting tomorrow" would get archived during the next sleep pass after that date, regardless of access patterns. Simple to implement in existing consolidation.

4. **`entityContext` for guided extraction.** When the user provides context about what a piece of content is ("this is a recipe" / "this is meeting notes"), extraction quality improves. Somnigraph could accept optional extraction hints on `remember()`.

5. **Container-tag scoping as a first-class concept.** Somnigraph uses `category` and `themes` for organizing memories but doesn't have project-level isolation. For multi-project use, container tags would enable cleaner context switching.

## Not Useful For Us

1. **Multi-tenant architecture.** Somnigraph is single-user by design. The organizational complexity (roles, API keys, container tag isolation) solves a problem we don't have.

2. **Document ingestion pipeline.** PDF/image/video processing is product infrastructure for a SaaS memory layer. Somnigraph's value is in how it retrieves and consolidates, not in content extraction.

3. **Framework integration wrappers.** Vercel AI SDK, Mastra, OpenAI Agents SDK — these are distribution mechanisms for a commercial product. Somnigraph interfaces via MCP and that's sufficient.

4. **Memory-graph visualization.** The React component for force-directed graph rendering is polished but orthogonal to retrieval quality research. Interesting for product but not for the research agenda.

5. **Connector ecosystem.** Google Drive/Notion/OneDrive sync is product infrastructure. If Somnigraph needed external data, the integration would be via the existing MCP ecosystem, not custom connectors.

## Connections

**Zep/Graphiti influence.** Supermemory's research page explicitly references Zep's temporal knowledge graph paper. The Updates/Extends/Derives relationship types map loosely to Graphiti's edge types, though Supermemory's schema is simpler (three relation types vs. Graphiti's richer semantic triples with temporal validity). The version chain model (`parentMemoryId → isLatest`) is a simplified temporal knowledge graph without Graphiti's bi-temporal `validAt`/`invalidAt` tracking.

**Comparison with Mem0's extraction approach.** Both extract discrete memories from conversations. Mem0's open-source code shows the extraction prompt and update logic; Supermemory's is hidden behind the API. Supermemory's schema is richer (version chains, relationship types, static/dynamic distinction, inference flagging) while Mem0's is flatter but transparent. On PERMA, Supermemory's MCQ (0.655) lags behind Mem0-graph (0.686 reported in PERMA paper) despite the richer schema, suggesting extraction quality matters more than schema richness.

**Comparison with MemOS.** MemOS achieves the highest PERMA MCQ (0.811) using ~6400 context tokens — 68x more than Supermemory's ~94. The compression-accuracy tradeoff is stark: MemOS's approach of retaining more raw content with MemCube containers beats Supermemory's aggressive extraction-to-profile compression, especially on multi-domain Turn=1 (0.306 vs 0.248).

**A-Mem parallel.** A-Mem's Zettelkasten approach (atomic notes with autonomous linking) is conceptually similar to Supermemory's memory entries with relationship types. Both decompose content into atomic units and build explicit relationships. A-Mem reports 85-93% fewer tokens (similar compression ratio to Supermemory's PERMA result) while achieving rank-1 on LoCoMo/DialSim. The difference is A-Mem's linking is agentic (the LLM decides connections during storage) while Supermemory's appears to be pipeline-based.

**PERMA positioning.** Among the 11 systems benchmarked in PERMA, Supermemory occupies an extreme position on the compression-accuracy frontier: lowest context tokens, mid-tier accuracy. Systems that retain more context (MemOS, EverMemOS, full-context) generally perform better on cross-domain synthesis, suggesting that Supermemory's profile-based approach sacrifices the associative structure needed for multi-domain preference tracking.

## Summary Assessment

Supermemory is a well-engineered commercial memory product with a clean developer experience and thoughtful data model. The schema — with its static/dynamic distinction, three-way relationship typing, version chains, and temporal expiry — represents a mature understanding of what memory systems need. The integration surface (MCP server, framework wrappers, connectors, graph visualization) is the most polished in the space. The ~50ms profile endpoint is a genuinely useful product concept: a precomputed user summary that makes personalization cheap at inference time.

However, the core intelligence is proprietary and unverifiable. The extraction engine, relationship discovery, contradiction resolution, and "automatic forgetting" are all server-side black boxes. The PERMA benchmark results reveal the cost of their aggressive compression strategy: competitive on single-domain recall (MCQ 0.655-0.674) but the weakest multi-domain performance in the field (Turn=1: 0.248). The profile-based retrieval model — returning a flat summary rather than performing associative retrieval over a rich graph — explains both the extreme compression and the multi-domain collapse. When cross-domain synthesis requires connecting "prefers dark mode in coding tools" with "likes minimal UI in design apps," a flat profile discards the structural linkage that graph-based or richer retrieval approaches preserve.

For Somnigraph's research agenda, Supermemory validates several design directions (explicit memory relationships, static/dynamic distinction, temporal expiry) while its PERMA weakness on multi-domain tasks reinforces the value of graph-conditioned retrieval and associative structure — exactly the capabilities Somnigraph is building toward. The ideas worth borrowing (precomputed profiles, version chains, event-triggered expiry) are implementable within Somnigraph's existing architecture without requiring Supermemory's infrastructure complexity.
