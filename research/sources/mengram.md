# Mengram Analysis (Agent Output)

*Generated 2026-02-20 by Opus agent reading live repo via WebFetch*

**Repo**: [alibaizhanov/mengram](https://github.com/alibaizhanov/mengram) | **Version**: 2.7.3 | **Author**: Ali Baizhanov
**Stars**: 26 | **Created**: 2026-02-10 | **Last commit**: 2026-02-19 (9 days old)
**License**: Apache 2.0 | **Language**: Python (with JS SDK) | **Co-authored with Claude**

---

## 1. Architecture Overview

### Dual Architecture: Local + Cloud

Mengram has **two completely separate implementations** sharing only the LLM extraction layer:

**Local mode** (`mengram.py` + `engine/`):
- Obsidian-style Markdown vault (`.md` files with wikilinks)
- In-memory knowledge graph built from parsed vault files
- SQLite vector store with numpy cosine similarity
- Local sentence-transformers embeddings
- Essentially a Mem0-compatible wrapper around an Obsidian vault with knowledge graph traversal

**Cloud mode** (`cloud/`):
- PostgreSQL + pgvector
- OpenAI `text-embedding-3-large` at 1536 dimensions (Matryoshka)
- FastAPI REST server on Railway
- API key auth, connection pooling, rate limiting
- **This is where all the interesting features actually live**

The three memory types, procedure evolution, and all production features are cloud-only. Local mode shares the `ConversationExtractor` but diverges completely on storage and retrieval.

### Core Components (Cloud)

```
cloud/api.py          - FastAPI endpoints (~1700 lines)
cloud/store.py        - PostgreSQL storage layer (~4100 lines) -- the real brain
cloud/embedder.py     - OpenAI embedding wrapper
cloud/evolution.py    - Experience-driven procedure evolution
cloud/client.py       - Python SDK (thin HTTP wrapper)
cloud/schema.sql      - PostgreSQL schema definition
api/cloud_mcp_server.py - MCP server (wraps CloudMemory client)
```

### Tech Stack

| Layer | Local | Cloud |
|-------|-------|-------|
| Storage | Markdown files + SQLite | PostgreSQL + pgvector |
| Embeddings | sentence-transformers (local) | OpenAI text-embedding-3-large (API) |
| Vector search | numpy cosine similarity | pgvector HNSW index |
| Text search | ILIKE fallback | PostgreSQL tsvector/GIN (BM25) |
| LLM extraction | Anthropic/OpenAI/Ollama | Anthropic/OpenAI |
| LLM re-ranking | N/A | gpt-4o-mini |
| Framework | None (pure Python) | FastAPI |

---

## 2. Memory Type Implementation

### Schemas

All three types live in **separate PostgreSQL tables** with **separate embedding tables per type**:

**Semantic Memory** (entities + facts + relations + knowledge + embeddings):
```sql
entities(id, user_id, name, type, metadata, created_at, updated_at)
facts(id, entity_id, content, importance, access_count, last_accessed,
      archived, superseded_by, expires_at, created_at)
relations(id, source_id, target_id, type, description)
knowledge(id, entity_id, type, title, content, artifact, scope, confidence)
embeddings(id, entity_id, chunk_text, embedding vector(1536), tsv tsvector)
```

**Episodic Memory**:
```sql
episodes(id, user_id, summary, context, outcome, participants TEXT[],
         emotional_valence, importance, linked_procedure_id,
         failed_at_step, metadata JSONB, created_at, expires_at)
episode_embeddings(id, episode_id, chunk_text, embedding vector(1536), tsv tsvector)
```

**Procedural Memory**:
```sql
procedures(id, user_id, name, trigger_condition, steps JSONB,
           entity_names TEXT[], success_count, fail_count, last_used,
           version, parent_version_id, evolved_from_episode,
           is_current, metadata JSONB, created_at, updated_at, expires_at)
procedure_embeddings(id, procedure_id, chunk_text, embedding vector(1536), tsv tsvector)
procedure_evolution(id, procedure_id, episode_id, change_type, diff JSONB,
                    version_before, version_after, created_at)
```

### Extraction

All three types are extracted in a **single LLM call** via `ConversationExtractor`. One JSON response containing entities, relations, knowledge, episodes, and procedures simultaneously.

Key extraction rules:
- **Entities**: Only named, specific entities with 2+ facts. Types: person, project, technology, company, concept.
- **Episodes**: "something that HAPPENED" — discussions, decisions, milestones. Includes `emotional_valence` (positive/negative/neutral/mixed) and `importance` (0.3-0.9).
- **Procedures**: "a repeatable sequence of steps." Requires 2+ concrete steps. Only from confirmed actions, not hypothetical.

Extraction receives existing context and skips facts that already exist (even if worded differently).

### Data Models

```python
@dataclass
class ExtractedEpisode:
    summary: str                    # up to 15 words
    context: str                    # detailed description
    outcome: str                    # result/outcome
    participants: list[str]         # entity names involved
    emotional_valence: str          # positive/negative/neutral/mixed
    importance: float               # 0.0-1.0

@dataclass
class ExtractedProcedure:
    name: str                       # procedure name
    trigger: str                    # when to apply
    steps: list[dict]               # [{step, action, detail}]
    entities: list[str]             # related entity names
```

---

## 3. Type-Differentiated Retrieval (Critical Finding)

### The Claim vs. The Reality

The README claims different retrieval algorithms per type:
- Semantic → keyword matching
- Episodic → time-range filtering
- Procedural → step-sequence matching

**The reality: the README overclaims. The differentiation is mostly structural, not algorithmic.**

### What Actually Happens

**Semantic search** (`store.search_vector`) — the most sophisticated:
1. **Vector search** via pgvector HNSW (cosine similarity, min_score 0.3)
2. **BM25 text search** via tsvector/plainto_tsquery + entity name ILIKE
3. **RRF fusion** (k=60) to merge vector + BM25 results
4. **Graph expansion** — top 8 seed entities, follow relations to connected entities (added at 50% of best direct match score)
5. **Recency boost** — 15% for last 7 days, 5% for last 30 days
6. **Ebbinghaus decay** on fact importance: `importance * e^(-0.03 * days_since_access)` plus `log(1 + access_count) * 0.05`
7. **LLM re-ranking** via gpt-4o-mini ("which entities are directly relevant?")

This is genuinely rich. RRF + graph expansion + importance decay + LLM re-ranking is a serious pipeline.

**Episodic search** (`store.search_episodes_vector`) — much simpler:
1. **Vector search** via pgvector on `episode_embeddings` table
2. Optional **time-range filtering** (`after`/`before` parameters on `created_at`)
3. That's it. No RRF, no graph expansion, no importance decay, no re-ranking.

**Procedural search** (`store.search_procedures_vector`) — equally simple:
1. **Vector search** via pgvector on `procedure_embeddings` table
2. Filter to `is_current = TRUE` (only latest versions)
3. That's it. **No step-sequence matching exists.** No RRF. No re-ranking.

### The `search_all()` Function

```python
@app.post("/v1/search/all", tags=["Search"])
async def search_all(req: SearchRequest, user_id: str = Depends(auth)):
    emb = embedder.embed(req.query)
    semantic = store.search_vector_with_teams(user_id, emb, top_k=..., query_text=req.query)
    episodic = store.search_episodes_vector(user_id, emb, top_k=...)
    procedural = store.search_procedures_vector(user_id, emb, top_k=...)
    # Re-rank semantic only
    if semantic and len(semantic) > 1:
        semantic = rerank_results(req.query, semantic)
    return {"semantic": semantic, "episodic": episodic, "procedural": procedural}
```

Runs the **same embedding** against three separate tables. Episodic time-range filtering exists but isn't wired into `search_all()` — only available via the dedicated `/v1/episodes/search` endpoint with explicit `after`/`before` parameters.

### What IS Type-Differentiated (Structural Value)

1. **Separate embedding tables per type** — prevents cross-type competition in top-k. A highly relevant procedure won't be displaced by a slightly-more-similar semantic memory. **This is the genuine insight.**
2. **Episodic time-range filtering** — available via dedicated endpoint but not in unified search.
3. **Procedural version filtering** — only returns `is_current = TRUE` (latest versions).
4. **Different return schemas** — semantic returns entities+facts+relations+knowledge; episodic returns summary+context+outcome+participants+emotional_valence; procedural returns name+trigger+steps+success_count+fail_count.

---

## 4. Procedure Evolution (Standout Feature)

The most genuinely novel feature in the repo.

### The Learning Loop

When a procedure fails:
1. User reports failure via `procedure_feedback(id, success=False, context="migration step failed")`
2. System creates a **failure episode** linked to the procedure (`linked_procedure_id`, `failed_at_step`)
3. `EvolutionEngine.evolve_on_failure()` sends old procedure + failure context to LLM
4. LLM produces improved steps + diff (added/removed/modified)
5. New version saved with `version++`, `parent_version_id` pointing to previous
6. Old version gets `is_current=FALSE`
7. Evolution logged in `procedure_evolution` table with full diff

### Auto-Discovery

`detect_and_create_from_episodes()` clusters 3+ similar positive episodes (cosine similarity > 0.75) and auto-creates procedures from detected patterns.

### The Result

```
procedure → use → fail → failure episode → evolve → improved procedure
                ↗ success episodes accumulate → detect patterns → new procedure
```

A genuine learning loop. Procedures are living, versioned documents that improve through experience.

---

## 5. Other Notable Features

### Cognitive Profile

Single API call (`/v1/profile`) generates a system prompt from all memory. Uses gpt-4o-mini to synthesize entities, facts, recent episodes, and known procedures into a 150-250 word personality/context prompt. Cached 1 hour.

### Reflection Engine

Three levels of AI-generated reflections:
- **Entity reflections**: 2-3 sentence summary for entities with 3+ facts
- **Cross-entity patterns**: Career direction, tech preferences, behavioral patterns
- **Temporal reflections**: What changed recently

Triggered when 10+ new facts accumulate or 24h+ since last reflection with 3+ new facts.

### Fact Contradiction Detection

LLM compares new facts against existing ones for the same entity. Contradicted facts get `archived=TRUE` with `superseded_by` reference to the replacement. Not graded (unlike Engram's 5-level system) — it's binary: contradiction or not.

### Ebbinghaus Importance Decay

```python
importance * e^(-0.03 * days_since_access) + log(1 + access_count) * 0.05
```

Applied during semantic search. Naturally surfaces frequently-accessed and recently-accessed facts. Episodes and procedures do NOT have this decay.

### Memory Agents

Autonomous background agents (triggered on demand):
- **Curator**: Health scoring, stale fact detection, contradiction flagging
- **Connector**: Cross-entity pattern detection, missing link suggestions
- **Digest**: Weekly summary generation

### Smart Triggers (v2.6)

Proactive memory: reminders, contradiction detection, pattern alerts. Stored in `memory_triggers` table with fire-at scheduling.

### MCP Server

Full MCP server wrapping the CloudMemory SDK client. Tools: `remember`, `remember_text`, `recall`, `search`, `timeline`, `vault_stats`, `run_agents`, `get_insights`, `list_procedures`, `procedure_feedback`, `procedure_history`.

**Notable gap**: The MCP `recall` tool only searches semantic memory. No MCP tool calls `search_all()`. Episodic and procedural require separate tool calls.

MCP instructions are aggressive: "AUTOMATICALLY call 'remember' when the user shares personal info... Do NOT ask permission — just save it silently."

### Additional Integrations

- LangChain: `ChatMessageHistory` + `BaseRetriever`
- CrewAI: Five tools (remember, recall, recall_all, get_profile, procedure_feedback)
- REST API: Full FastAPI with OpenAPI docs, rate limited 120 req/min

---

## 6. How Mengram Addresses Our 7 Known Gaps

| Gap | Rating | Notes |
|-----|--------|-------|
| 1. Layered Memory | 30% | Three types but no intra-type layering. No summary → detail hierarchy. Reflections are separate, not integrated. |
| 2. Multi-Angle Retrieval | 60% | RRF + graph + LLM reranking for semantic. Plain vector for episodic/procedural. No multi-angle indexing. |
| 3. Contradiction Detection | 50% | Binary LLM-based contradiction on facts. No graded tension (cf. Engram's 5-level). No temporal evolution detection. Archived with `superseded_by` link. |
| 4. Relationship Edges | 70% | Entity-relation graph with typed edges. Graph expansion in search. No edges between memory types (episode→procedure links are FK only). |
| 5. Sleep Process | 20% | No consolidation pipeline. Agents do ad-hoc cleanup on demand. Reflection engine is closest analog but doesn't synthesize across types. |
| 6. Reference Index | 40% | `vault_stats` endpoint exists. Cognitive profile is an interesting overlay. No lightweight memory overview for routing. |
| 7. Temporal Trajectories | 15% | Procedure evolution chain is a trajectory for procedures only. No temporal narrative for beliefs/preferences ("used to X, now Y"). |

---

## 7. Comparison with claude-memory

### Where Mengram is Stronger
- Separate embedding tables per type (prevents cross-type competition)
- Procedure evolution with versioning, success/fail tracking, and LLM-driven improvement
- Entity-relation knowledge graph with graph expansion in search
- Ebbinghaus importance decay on semantic facts
- LLM re-ranking as final stage in semantic search
- Cognitive profile generation
- Multi-framework integration (MCP, LangChain, CrewAI, REST)

### Where Our System is Stronger
- RRF hybrid search (vector + keyword) applied to ALL types equally
- Human-in-the-loop curation (pending/active review)
- Priority as user intent (p1-p10) vs. LLM-inferred importance
- Months of battle-tested usage vs. 9 days old
- Simplicity (single table, single MCP, predictable behavior)
- Token budget awareness in recall
- No external API dependency for embeddings (FastEmbed local)

### Fundamental Difference

Mengram = **cloud-hosted, LLM-automated memory infrastructure for multiple frameworks**. We = **author-curated local memory for a single persistent identity**. Mengram automates everything with LLM calls; we keep the human in the loop. Both are valid; the design spaces are complementary.

---

## 8. Insights Worth Stealing (Ranked)

| Rank | Insight | Effort | Impact |
|------|---------|--------|--------|
| 1 | **Separate embedding spaces per type** — prevents cross-type competition in top-k. Could be added to our single-table model via filtered vector search or separate indices. | Low | Medium |
| 2 | **Procedure evolution loop** — failure feedback → LLM-evolved version → version chain. Concept is powerful; our implementation would be simpler (update content + log change). | Medium | Medium-High |
| 3 | **Success/fail tracking on procedures** — `success_count`/`fail_count` fields. Simple metadata that makes procedural memories evidence-based. | Low | Medium |
| 4 | **Ebbinghaus decay formula** — `importance * e^(-0.03 * days)` with access frequency boost. More principled than flat priority. Could supplement (not replace) our priority system. | Low | Low-Medium |
| 5 | **Auto-procedure detection from episodes** — cluster 3+ similar positive episodes → generate procedure. Interesting for `/sleep` consolidation. | Medium | Medium |

---

## 9. What's Not Worth It

- **Cloud-hosted architecture** — we want local-first, no API dependency
- **Aggressive auto-remember** — fights against curation
- **Cognitive profile generation** — core.md serves this role better with human crafting
- **Memory agents** (curator/connector/digest) — overkill for single user; `/sleep` covers this
- **gpt-4o-mini re-ranking** — adds API cost and latency; our RRF handles ranking adequately
- **Single massive store.py** — 4100-line file with no separation of concerns
- **Minimal testing** — 8 tests total, all for Markdown parser, zero for core functionality

---

## 10. Key Takeaway

**Mengram's README overclaims on type-differentiated retrieval, but the structural separation of embedding spaces per type is a genuine insight.** The procedure evolution loop is the standout novel feature. The system confirms our direction (Tulving taxonomy, multi-channel retrieval, relationship graphs) while providing a few specific design patterns worth adopting — particularly separate embedding indices and success/fail tracking on procedures.

The overclaiming pattern is instructive: the author told Reddit that `search_all()` uses "three different retrieval algorithms, not three folders with the same search." In reality, all three types get the same vector similarity algorithm, and semantic search is richer only because it was built first. The aspirational claims (step-sequence matching for procedures) don't exist in code. Worth noting as a calibration point for evaluating other systems' README claims.
