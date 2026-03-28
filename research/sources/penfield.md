# Penfield Labs — Source Analysis

*Phase 14 (updated), 2026-03-28. Unified analysis of four repositories: openclaw-penfield, claude-penfield, penfield-mcp, docs.*

## 1. Ecosystem Overview

Penfield Labs builds a hosted persistent memory service for AI agents, accessed through thin client plugins and a remote MCP server. The ecosystem spans four public repositories plus a proprietary backend at `api.penfield.app`.

| Repo | Purpose | Stack | Files | License | Maturity |
|------|---------|-------|-------|---------|----------|
| [claude-penfield](https://github.com/penfieldlabs/claude-penfield) | Claude Code plugin (hooks + skill + MCP config) | Bash, Markdown, JSON | 13 | MIT | v1.1.0 (2026-02-09) |
| [penfield-mcp](https://github.com/penfieldlabs/penfield-mcp) | MCP server config + agent-facing docs | JSON, Markdown | 9 | All Rights Reserved | v1.22.0 |
| [docs](https://github.com/penfieldlabs/docs) | Full API docs site (MkDocs) | Markdown, 80+ screenshots | 34 md | Proprietary | Active |
| [openclaw-penfield](https://github.com/penfieldlabs/openclaw-penfield) | OpenClaw native plugin (TypeScript client) | TypeScript (TypeBox, Zod) | ~31 | MIT (per original analysis) | v2.0.0 |

**Key insight:** claude-penfield contains *zero code*. It is config files (hooks.json, .mcp.json, plugin.json), two shell scripts, and markdown. The MCP server is a remote endpoint at `https://mcp.penfield.app` — no local server process. penfield-mcp is similarly just configuration and documentation pointing at the same remote server. All compute, storage, embedding, indexing, and retrieval logic lives in the proprietary backend.

**Client diversity:** The system targets Claude Code (plugin), Claude Desktop/Mobile/Web (MCP connector), Cursor, Windsurf, Cline, Roo Code, Gemini CLI, LM Studio, Manus, Perplexity, and OpenClaw. The onboarding docs include step-by-step screenshots for 10+ platforms — significant go-to-market effort.

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 Penfield Backend (proprietary)           │
│  api.penfield.app / mcp.penfield.app                    │
│                                                         │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  ┌────────┐  │
│  │ BM25    │  │ Vector   │  │ Knowledge │  │ Object │  │
│  │ Index   │  │ Index    │  │ Graph     │  │ Store  │  │
│  └─────────┘  └──────────┘  └───────────┘  └────────┘  │
│         │           │             │              │       │
│         └───────────┼─────────────┘              │       │
│              RRF Fusion (rrf_k configurable)     │       │
│                     │                            │       │
│              ┌──────┴──────┐              ┌──────┴──────┐│
│              │ Search API  │              │ Artifacts   ││
│              │ v2/search/* │              │ v2/artifacts││
│              └─────────────┘              └─────────────┘│
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ Personality  │  │ Analysis     │  │ Documents      │  │
│  │ Personas     │  │ Reflect      │  │ Upload/chunk   │  │
│  │ Traits       │  │ Summarize    │  │ Auto-embed     │  │
│  │ Awakening    │  │ Insights     │  │ Neighbor ctx   │  │
│  └─────────────┘  └──────────────┘  └────────────────┘  │
└──────────────────────────┬──────────────────────────────┘
                           │ MCP (Streamable HTTP)
          ┌────────────────┼────────────────┐
          │                │                │
  ┌───────┴───────┐ ┌─────┴──────┐ ┌───────┴────────┐
  │ claude-penfield│ │ penfield-  │ │ openclaw-      │
  │ (plugin)      │ │ mcp        │ │ penfield       │
  │ Hooks:        │ │ (config)   │ │ (TypeScript)   │
  │  SessionStart │ │            │ │ 17 tools       │
  │  PreCompact   │ │            │ │ Response       │
  │ Skill: SKILL  │ │            │ │  compaction    │
  │  .md          │ │            │ │ Auto-inject    │
  └───────────────┘ └────────────┘ └────────────────┘
```

**What's visible:** The API docs expose the full REST surface: 10 endpoint groups (memories, search, relationships, personality, analysis, documents, artifacts, authentication, tags, awakening) with detailed request/response schemas, error codes, and pagination. This is unusually transparent for a SaaS — enough to understand the data model and query interface completely.

**What's hidden:** Embedding model, vector index type, BM25 implementation, graph storage engine, ranking algorithm internals, any consolidation or background processing logic. The API docs show you can set `rrf_k` (1-100) and individual channel weights, but how those translate to scores is server-side.

**Storage:** All server-side. Local state is limited to OAuth credentials (`~/.openclaw/extensions/` for OpenClaw, none for Claude Code plugin which uses MCP auth).

## 3. Memory Model

### Types (11)

Consistent across all repos. The TOOLS.md, MEMORY-TYPES.md, SKILL.md, and API docs all list the same 11 types with identical descriptions:

| Type | Category | Writable via MCP? |
|------|----------|--------------------|
| fact | Core | Yes |
| insight | Core | Yes |
| correction | Core | Yes |
| conversation | Core | Yes |
| reference | Core | Yes |
| task | Core | Yes |
| strategy | Core | Yes |
| checkpoint | Core | Yes |
| identity_core | Identity | No (portal/API only) |
| personality_trait | Identity | No (portal/API only, Premium+) |
| relationship | Identity | Yes |

**Type auto-detection:** The MCP `store` tool auto-detects type from content — you don't pass it as a parameter. The REST API accepts an optional `memory_type` field. This means the MCP path relies on server-side classification, while the REST API allows explicit typing.

### Memory schema

Each memory carries:
- `content` (string, 1-10,000 chars)
- `memory_type` (enum)
- `importance` (float 0-1, auto-calculated if omitted)
- `confidence` (float 0-1)
- `surprise_score` (float 0-1, set at creation — described as "future versions may update dynamically")
- `source_type` (enum: direct_input, document_upload, web_scrape, api_import, conversation, reflection, checkpoint, checkpoint_recall)
- `tags` (string array, max 10, normalized lowercase)
- `metadata` (arbitrary key-value)
- `access_count` / `last_accessed` (tracked asynchronously, not real-time)
- `embedding_status` (pending/completed/failed)
- `lifecycle_state` (active, superseded, corrected, contradicted, archived)

**Notable:** The `surprise_score` field exists in the schema but appears to be a static default (0.5). The `lifecycle_state` and version history (`/memories/{id}/history`, `/memories/{id}/evolve`) provide explicit knowledge evolution tracking — memories can be evolved, not just updated in place.

### Relationships (24)

Consistent across all repos, organized into 8 categories:

| Category | Types | Count |
|----------|-------|-------|
| Knowledge Evolution | supersedes, updates, evolution_of | 3 |
| Evidence | supports, contradicts, disputes | 3 |
| Hierarchy | parent_of, child_of, sibling_of, composed_of, part_of | 5 |
| Causation | causes, influenced_by, prerequisite_for | 3 |
| Implementation | implements, documents, tests, example_of | 4 |
| Conversation | responds_to, references, inspired_by | 3 |
| Sequence | follows, precedes | 2 |
| Dependencies | depends_on | 1 |

Relationships carry:
- `strength` (float 0-1, default 0.5)
- `confidence` (float 0-1, default 0.8)
- `evidence` (object, free-form reasoning)
- `direction_type` (DIRECTED, BIDIRECTIONAL, INVERSE_PAIRED)
- `traversal_properties` (object, for graph algorithms)

**Bulk creation:** `POST /api/v2/relationships/bulk` supports transactional multi-edge creation — useful for the agent "store then connect" workflow.

### Personality system

10 persona presets across 4 categories (standard, technical, contemplative, fun), tier-gated from free to premium. Custom traits require Premium+. Organization-wide defaults require Premium+. The awakening briefing regenerates from the current preset each time — ensuring users get the latest base instructions even after backend updates.

The persona list includes "Conspiracy Theorist", "Sassy Pirate", and "E-Girl" personas (premium tier) alongside professional ones. This is... a product choice.

## 4. Retrieval Pipeline

### Hybrid search

Three channels with configurable weights (must sum to 1.0):

| Channel | Default Weight | Description |
|---------|---------------|-------------|
| BM25 | 0.4 | Keyword matching |
| Vector | 0.4 | Semantic similarity (embedding model unknown) |
| Graph | 0.2 | Knowledge graph expansion |

Combined via RRF with configurable `rrf_k` (default 60, range 1-100). Higher k produces more balanced fusion; lower k favors top-ranked results.

### Graph expansion parameters
- `enable_graph_expansion` (default: true)
- `graph_max_depth` (default: 2, max: 5)
- `graph_seed_limit` (default: 10, max: 50) — top results used as graph traversal seeds

### Response structure
The `score_breakdown` returns individual channel scores:
```json
{
  "final_score": 0.95,
  "bm25_score": 0.85,
  "vector_score": 0.92,
  "graph_score": 0.7,
  "rank_bm25": 1.0,
  "rank_vector": 2.0
}
```

Plus `knowledge_cloud` (nodes + edges + stats) for graph visualization when graph expansion is enabled. The response also includes timing stats per channel (`bm25_time_ms`, `vector_time_ms`, `graph_time_ms`, `fusion_time_ms`).

### MCP tool mapping

The `recall` and `search` MCP tools both map to `POST /api/v2/search/hybrid` with different default weights:
- `recall`: balanced (0.4/0.4/0.2) — general purpose
- `search`: semantic-biased (higher vector weight) — concept search

**What we can't see:** The actual vector index (HNSW? IVF?), embedding model, BM25 implementation (Okapi? BM25+?), how graph scores are computed during expansion, whether there's any reranking beyond RRF. The `rrf_k` configurability and per-channel timing suggest a reasonably sophisticated backend, but the ranking function itself is opaque.

### Vector search endpoint

Separate pure vector endpoint (`POST /api/v2/search/vector`) for semantic-only queries. Lower latency, simpler response (just `vector_similarity` score). Useful when you know you want concept matching without keyword or graph signals.

### Graph traversal

Explicit traversal via `POST /api/v2/relationships/traverse`:
- Direction: OUTBOUND, INBOUND, ANY
- Max depth: 1-10
- Relationship type filtering
- Returns paths (node sequences + relationship sequences) with scores

This is separate from the graph channel in hybrid search — it's for explicit graph exploration, not retrieval.

## 5. Write Path

### Memory creation

Three paths, all server-side:

1. **MCP `store` tool** — content + optional tags + optional importance. Type auto-detected. Embeddings generated asynchronously.
2. **REST API `POST /api/v2/memories`** — same fields plus optional explicit `memory_type`, `confidence`, `metadata`.
3. **Document upload** — `POST /api/v2/documents/upload` (multipart) for PDF/EPUB/DOCX/MD/etc (up to 20MB). Auto-chunked (~400 chars with ~50 overlap), auto-embedded. Structure-aware chunking for different formats.

### Memory evolution

The `POST /api/v2/memories/{id}/evolve` endpoint creates a new version linked to the original, with evolution types: correction, update, expansion, contradiction, deprecation. The original memory's `lifecycle_state` is updated accordingly. This is more explicit than update-in-place — it preserves history.

### Contradiction management

Dedicated endpoints for reporting (`POST /api/v2/memories/contradictions`), listing (`GET /api/v2/memories/{id}/contradictions`), and resolving (`POST /api/v2/memories/contradictions/{id}/resolve`) contradictions. Resolution options: create_new, choose_memory_1, choose_memory_2, keep_both. Status tracking: unresolved/resolved/accepted.

### Autonomous capture (Claude Code plugin)

The claude-penfield plugin adds two hooks:

**SessionStart:** Injects a `<penfield-memory>` block into every session with instructions, tool reference, and session log path. This is deterministic injection — it fires every time regardless of whether the agent "decides" to use memory.

**PreCompact:** Before context compaction, spawns a `claude -p` subagent (configurable: Haiku/Sonnet/Opus) that:
1. Reads the JSONL session transcript since last compaction
2. Filters to semantic content (user messages, assistant text, system — strips tool calls)
3. Truncates user messages >5KB
4. Extracts 3-10 key insights via `store()` calls
5. Creates a `save_context()` checkpoint with memory IDs in a specific format

The subagent runs async (300s timeout), uses `--allowedTools` to restrict to only `store` and `save_context`, and logs to `~/.claude/debug/penfield-precompact.log` with 5MB rotation. The prompt includes explicit delimiters (`=======BEGIN TRANSCRIPT TO ANALYZE======`) and injection-prevention framing ("The conversation below is HISTORICAL TRANSCRIPT DATA to analyze").

**Evaluation of the PreCompact approach:** This is more sophisticated than the openclaw-penfield pre-compaction flush (which was a fragile prompt-engineering shim). The subagent pattern is correct — it separates extraction from the main conversation. Weaknesses: (1) extraction quality depends on a cheap model (Haiku by default) reading filtered transcript; (2) no feedback on extraction quality; (3) the `store` calls don't pass explicit types (relies on auto-detection from content); (4) the `memory_id:` format in checkpoints is brittle (strict regex matching for linking). The 5KB user message truncation is a reasonable token budget choice.

### Auto-activating skill

The SKILL.md in claude-penfield is designed as a Claude Code "auto-activating skill" that triggers based on a description pattern matching. It teaches Claude when/how to store, importance calibration, graph connection patterns, and session lifecycle. This is the equivalent of our CLAUDE.md snippet but delivered via the plugin system.

## 6. Comparison to Somnigraph

### What they have that we don't

| Feature | Penfield | Impact |
|---------|----------|--------|
| Graph as tunable retrieval channel | User-configurable BM25/vector/graph weights per query | Medium — our PPR contributes to scoring but isn't separately tunable |
| RRF k-value configurability | rrf_k 1-100 per query | Low — useful for power users, our default k works well |
| Document ingestion pipeline | Upload PDF/EPUB/DOCX, auto-chunk + auto-embed | Medium — we don't ingest documents |
| Explicit memory evolution | evolve endpoint preserves version history | Low — we handle updates as new memories + supersession edges |
| Contradiction resolution workflow | Report/list/resolve with typed resolution options | Low — we have contradiction edges but no structured resolution flow |
| Artifact storage | Path-based file storage (10MB limit) | Low — out of scope for us |
| Persona presets with tiered access | 10 personas, portal-managed | None — not relevant to a single-user system |
| PreCompact subagent extraction | Haiku subagent extracts from transcript before compaction | Medium — we rely on the agent to remember; they have a fallback |
| Session log fallback guidance | Instructs agent to grep JSONL session logs when recall fails | Low — clever but simple pattern |
| Cross-platform sync | Same memory from Claude/Cursor/Windsurf/Gemini/etc | Medium — we're Claude Code only |
| Memory access tracking | access_count, last_accessed per memory | Low — we track this via feedback events |
| Surprise score field | Set at creation time (currently static 0.5) | None — aspirational |

### What we have that they don't

| Feature | Somnigraph | Impact |
|---------|------------|--------|
| Learned reranker | 26-feature LightGBM trained on 1032 human-judged queries | High — no evidence of any learned scoring on their side |
| Retrieval feedback loop | EWMA + UCB exploration + Hebbian co-retrieval | High — no feedback mechanism visible |
| Biological decay | Per-category decay rates with configurable per-memory overrides | High — memories are permanent in Penfield |
| Sleep consolidation | 10-step NREM/REM pipeline (prune, merge, enrich, reflect) | High — no background processing visible |
| Hebbian edge strengthening | Co-retrieval-based automatic edge weight updates | Medium — their edges have static strength |
| Graph edges from retrieval patterns | Automatic edges from co-retrieval (support, derive, evolve) | Medium — all their edges are manually created |
| Local + inspectable | SQLite + sqlite-vec + FTS5, all code visible | High for research — their retrieval is a black box |
| Reranker feature engineering | 26 hand-crafted features (proximity, betweenness, burstiness, etc.) | High — RRF is their only fusion mechanism visible |
| Theme-based retrieval | Theme extraction + theme channel in RRF | Low-medium — they have tags but not theme-based scoring |
| Query expansion | BM25-damped IDF keyword expansion, synthetic vocabulary bridges | High — they appear to use raw queries only |
| Configurable per-memory decay | decay_rate per memory for project-specific knowledge | Medium — their importance is static once set |

### Architectural trade-offs

| Dimension | Somnigraph | Penfield |
|-----------|------------|----------|
| Deployment | Local (SQLite) | Hosted SaaS |
| Transparency | Full code visibility | API surface only |
| Scoring | Learned reranker (26 features) | RRF fusion (3 channels, tunable weights) |
| Knowledge evolution | Implicit (decay, consolidation, feedback) | Explicit (evolve, contradict, lifecycle_state) |
| Edge creation | Automatic (Hebbian, sleep extraction) + manual | Manual only (agent calls connect) |
| Memory lifecycle | Decay + consolidation prune + merge | Permanent + manual evolve/deprecate |
| Multi-platform | Claude Code only | 10+ platforms |
| Data ownership | Local files | Vendor-hosted |
| Background processing | Sleep pipeline (10 steps) | None visible |
| Feedback integration | EWMA scoring + UCB exploration | access_count (async, read-only) |

## 7. Worth Adopting?

Revisiting the five ideas from the original openclaw-penfield analysis, updated with context from the other three repos:

### 1. Graph weight as a tunable retrieval channel (REVISIT: still interesting)

The docs confirm the implementation is real — `rrf_k` configurability, per-channel timing stats, `graph_seed_limit` parameter. Exposing graph contribution as a user-tunable weight in `recall()` remains a reasonable idea. Our PPR already computes graph scores; the question is whether making the weight explicit adds value over letting the reranker learn the optimal blend. **Verdict: The reranker makes this less necessary than it seemed.** The reranker already weights graph features appropriately per-query. An explicit `graph_weight` parameter would bypass the reranker's learned judgment. Not worth it unless we drop the reranker.

### 2. Auto-inject recent context (REVISIT: PreCompact is more interesting)

The session-start hook injection is straightforward but the PreCompact subagent pattern is the more novel contribution. Having a cheap model automatically extract and store context before compaction is a genuine safety net. We rely entirely on the main agent's discipline for memory creation. **Verdict: The PreCompact pattern solves a real problem** — context loss during compaction — that we also face. However, our sleep consolidation addresses part of this from the other end (enriching stored memories). The session-start injection is less relevant because our `startup_load()` already serves this purpose.

### 3. Response compaction (CONFIRMED: worth doing)

All three client repos strip debug fields from API responses before returning to the agent. Our `recall()` returns full memory objects. A compact response mode would reduce context usage. **Verdict: Still worth doing.** Low effort, clear token savings.

### 4. Weight-tuning guide (REVISIT: less needed)

The SKILL.md weight-tuning table (exact terms -> high BM25, concepts -> high vector) is practical guidance. But with our learned reranker, the agent doesn't need to tune weights — the reranker handles this automatically. **Verdict: Not needed.** Our guidance should focus on query formulation and limit selection, not weight tuning.

### 5. Named context checkpoints (REVISIT: still low priority)

The docs reveal save_context creates checkpoints with multi-source memory linking (explicit IDs, regex-extracted IDs from description, search-retrieved). Unique names per tenant. This is cleaner than our approach but solves a problem we handle differently (sleep consolidation + startup_load). **Verdict: Still low priority.** Our session-end workflow (remember + reflect) achieves similar handoff without a checkpoint abstraction.

### New idea from the full ecosystem:

**6. Contradiction resolution workflow** (effort: medium, impact: low-medium). Penfield has report/list/resolve endpoints for contradictions with typed resolution (choose_memory_1, choose_memory_2, create_new, keep_both). We have contradiction edges but no structured resolution. Our sleep pipeline could benefit from a resolution step — when contradictions are detected, present resolution options rather than just flagging. **Verdict: Interesting but not pressing.** Our contradiction edge filtering in scoring.py already prevents contradicted information from being co-surfaced. A resolution workflow would be more complete but addresses a problem we've already mitigated.

## 8. Worth Watching

- **Backend sophistication trajectory.** The API surface is unusually well-documented for a startup. If they add learned scoring, feedback loops, or background consolidation, the gap narrows. The `surprise_score` and `access_count` fields suggest they're thinking about these dimensions.
- **PreCompact hook evolution.** The subagent extraction pattern is v1 and clearly iterating (changelog shows PreCompact fixes in every release). If they add extraction quality feedback or multi-pass extraction, the pattern becomes more transferable.
- **Community adoption signals.** 10+ platform integrations suggest they're optimizing for breadth. If the community surfaces retrieval quality issues, their response will reveal backend priorities.
- **Graph-aware features.** The `knowledge_cloud` response and `graph_seed_limit` parameter suggest active investment in the graph channel. If they expose more graph algorithm controls (PageRank-style scoring, community detection), it's worth studying.

## 9. Key Claims vs Evidence

The Reddit post (r/AIMemory, 2026-03-28) titled "What an AI Memory System Should Look Like in 2026" listed 10 requirements. Verification against the actual code and docs:

| Claim | Verified? | Evidence |
|-------|-----------|----------|
| 1. Agent-managed memory (not manual) | Partial | MCP tools exist, but all storage is via explicit tool calls. PreCompact hook adds limited autonomous capture. No observation-based extraction. |
| 2. Typed memories | Yes | 11 types, consistent across all repos, auto-detection from content |
| 3. Knowledge graph with typed relationships | Yes | 24 types, 8 categories, bulk creation, explicit traversal to depth 10, graph as retrieval channel |
| 4. Hybrid search | Yes | BM25 + vector + graph with configurable weights, RRF fusion with tunable k |
| 5. Personality persistence | Yes | 10 persona presets, custom traits (Premium+), portal-managed identity_core/personality_trait |
| 6. GUI portal | Yes | portal.penfield.app exists (screenshots in onboarding docs confirm functional UI) |
| 7. Artifact storage | Yes | Path-based file storage, 10MB limit, CRUD operations |
| 8. Zero-config setup | Partial | Claude Code plugin install is 2 commands, but requires account signup, authentication, and model restart. "Zero-config" is marketing — it's low-config. |
| 9. Full API | Yes | Comprehensive REST API documented with schemas, examples, and error codes. 10 endpoint groups. |
| 10. What it's NOT (not a chatbot) | N/A | Positioning claim, not verifiable |

**The post was marketing** but the technical claims are largely substantiated by the code and docs. The weakest claim is "agent-managed" — the system is primarily agent-invoked (the agent must decide to call tools), with the PreCompact hook as the only autonomous capture mechanism. The "zero-config" claim overstates simplicity — signup + auth + restart is required.

**Missing from the post:** No mention of any feedback mechanism, no learning or adaptation, no background processing, no decay, no consolidation. The system is static once written — memories don't improve over time. This is the most significant gap relative to the claim of being "what an AI memory system should look like in 2026."

## 10. Relevance to Somnigraph

**Rating: 3/5 (reference value, limited adoptable ideas)**

Penfield is a well-executed SaaS memory product with comprehensive documentation and broad platform support. The API docs are the most detailed of any system surveyed and serve as a useful reference for what a production memory API surface looks like. The 11-type/24-relationship taxonomy is one data point for schema design (we use a simpler taxonomy and haven't needed more). The PreCompact subagent pattern for autonomous extraction is the most technically interesting contribution specific to the new repos.

However, the fundamental architecture — fully hosted, no feedback loop, no decay, no consolidation, no learned scoring — means there's little to adopt for a research system focused on retrieval quality improvement. The ideas that seemed most promising from the openclaw-penfield analysis (graph weight tuning, auto-inject context) are less compelling now that we have a learned reranker and sleep consolidation. Response compaction remains the clearest low-effort win.

The ecosystem analysis confirms the original assessment: Penfield makes sense as a product (handle complexity for users across platforms) but is architecturally opposed to a research artifact (where studying and improving the retrieval pipeline is the point). The Reddit post's claim about what memory systems "should" look like in 2026 conspicuously omits everything that makes memory systems actually improve over time — feedback, decay, consolidation, learned ranking — which happen to be Somnigraph's core contributions.
