# MemCP -- Source Analysis

*Phase 13, 2026-03-05. Analysis of maydali28/memcp (v0.3.0).*

## 1. Architecture Overview

**Language**: Python 3.10+, async via ThreadPoolExecutor (not true async yet; aiosqlite is optional Phase 3 future).

**Framework**: FastMCP (`@mcp.tool()` decorator pattern). Core deps: `mcp>=1.0.0`, `pydantic>=2.0.0` -- only 2 required packages. Everything else is optional extras.

**3-layer delegation**:
- `server.py` -- MCP tool endpoints (async, runs in thread pool)
- `tools/*.py` -- orchestration, JSON serialization
- `core/*.py` -- business logic, returns plain dicts

**Storage layout** (`~/.memcp/`):
```
graph.db                    # SQLite WAL -- nodes + edges tables (MAGMA 4-graph)
memory.json                 # Phase 1 legacy JSON (auto-migrated to graph.db)
state.json                  # Turn counter, current project/session
sessions.json               # Session registry
contexts/{name}/            # Named context variables
  content.md + meta.json
chunks/{context_name}/      # Chunked contexts
  index.json + 0000.md...
cache/                      # Embedding cache (insight_embeddings.npz)
archive/                    # Retention archive
  contexts/{name}/content.md.gz + meta.json
  insights.json             # Archived insights array
  purge_log.json            # Audit trail
```

**Key files**:
- `src/memcp/core/graph.py` -- thin facade over NodeStore + EdgeManager + GraphTraversal
- `src/memcp/core/node_store.py` -- SQLite schema, entity extraction, node CRUD
- `src/memcp/core/edge_manager.py` -- 4 edge generators, Hebbian learning, edge decay
- `src/memcp/core/graph_traversal.py` -- intent detection, ranking, BFS traversal
- `src/memcp/core/search.py` -- 5-tier search with RRF fusion
- `src/memcp/core/memory.py` -- remember/recall/forget with graph + JSON backends
- `src/memcp/core/retention.py` -- 3-zone lifecycle (active/archive/purge)
- `src/memcp/core/consolidation.py` -- similarity grouping via Union-Find + merge

**Optional extras** (9 packages, graceful degradation):
| Extra | Package | What |
|-------|---------|------|
| search | bm25s | BM25 ranked search |
| fuzzy | rapidfuzz | Typo-tolerant matching |
| semantic | model2vec + numpy | 256d embeddings |
| semantic-hq | fastembed + numpy | 384d embeddings |
| hnsw | usearch + numpy | O(log N) ANN |
| cache | diskcache | Persistent embedding cache |
| ner | spacy | spaCy NER entity extraction |
| vectors | sqlite-vec | SIMD KNN in SQLite |
| async | aiosqlite | Async SQLite (future) |

## 2. Graph Implementation

### MAGMA 4-Graph (inspired by arXiv:2601.03236)

All four edge types share the same node set (insights) in a single SQLite database with WAL mode. Single `edges` table with `edge_type` column.

**SQLite Schema**:
```sql
CREATE TABLE nodes (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    summary TEXT DEFAULT '',
    category TEXT DEFAULT 'general',        -- decision/fact/preference/finding/todo/general
    importance TEXT DEFAULT 'medium',       -- low/medium/high/critical
    effective_importance REAL DEFAULT 0.5,
    tags TEXT DEFAULT '[]',                 -- JSON array
    entities TEXT DEFAULT '[]',             -- JSON array
    project TEXT DEFAULT 'default',
    session TEXT DEFAULT '',
    token_count INTEGER DEFAULT 0,
    access_count INTEGER DEFAULT 0,
    feedback_score REAL DEFAULT 0.0,       -- [-1.0, 1.0] from reinforce
    last_accessed_at TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE edges (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    edge_type TEXT NOT NULL CHECK(edge_type IN ('semantic','temporal','causal','entity')),
    weight REAL DEFAULT 1.0,
    metadata TEXT DEFAULT '{}',            -- JSON
    created_at TEXT NOT NULL,
    last_activated_at TEXT,                -- Hebbian tracking
    PRIMARY KEY (source_id, target_id, edge_type)
);

CREATE TABLE entity_index (               -- Inverted index for O(1) entity edge lookup
    entity TEXT NOT NULL,
    node_id TEXT NOT NULL,
    PRIMARY KEY (entity, node_id)
);
```

### Edge Type Details

**Semantic edges**: On insert, compares new insight vs all existing in same project. With embeddings: cosine sim via VectorStore, top-3 with score >= 0.3. Without: keyword token overlap, top-3 with score >= 0.1. Weight = similarity score.

**Temporal edges**: Links to insights created within 30 minutes in same project. Queries last 20 nodes by created_at. Weight = `max(0.1, 1.0 - delta_minutes / 30)`.

**Causal edges**: Regex pattern detection for causal language ("because", "therefore", "due to", "decided to", etc.). If found, checks last 10 insights for 3+ token overlap with ratio >= 0.15. Directional (new -> cause). At most one causal link per insert.

**Entity edges**: Extracts entities via pluggable extractors (regex always, spaCy optional, LLM via sub-agent optional). Populates inverted `entity_index` table. Uses index for O(matches) lookup instead of O(N) scan. Weight = 1.0 (binary). Metadata stores the shared entity name.

**Entity extraction chain**:
1. `RegexEntityExtractor` (always) -- file paths, module dotpaths, URLs, @mentions, CamelCase identifiers
2. `SpacyEntityExtractor` (optional, `pip install memcp[ner]`) -- en_core_web_sm NER
3. `CombinedEntityExtractor` -- auto-selected when spaCy available, merges + deduplicates
4. `memcp-entity-extractor` sub-agent -- LLM-based (Haiku), richest extraction

### Edge Dynamics

**Hebbian co-retrieval strengthening**: After every `recall()`, top-10 result nodes get shared edge weights boosted by `MEMCP_HEBBIAN_BOOST` (default 0.05). `weight = min(weight + boost, 1.0)`. Updates `last_activated_at`.

**Activation-based edge decay**: `weight_new = weight * 2^(-days_since_activation / half_life)`. Half-life default 30 days. Edges below `MEMCP_EDGE_MIN_WEIGHT` (0.05) are pruned. Rate-limited to once per hour. Called lazily during `query()`.

**Feedback reinforcement**: `memcp_reinforce(helpful=True)` boosts feedback_score +0.1 and connected edges +0.02. `helpful=False` penalizes -0.2 and weakens edges -0.05. Score clamped [-1.0, 1.0].

### Query/Traversal

**Intent detection** on recall query:
| Pattern | Intent | Primary edge type |
|---------|--------|-------------------|
| "why..." / "reason" / "cause" | why | causal |
| "when..." / "timeline" / "chronolog" | when | temporal |
| "who..." / "which..." / "entity" | who | entity |
| default | what | semantic |

**Ranking formula**: `total_score = keyword_score * 0.7 + edge_boost * 0.3`. Then `total_score *= (1 + feedback_score * 0.3)`. Edge boost = `min(1.0, primary_count / max(1, total_count) + 0.1 * primary_count)`.

**BFS traversal** via `memcp_related(insight_id, edge_type, depth)`: standard BFS from center node, optionally filtered by edge type, returns center + related nodes + edges.

## 3. Tool Surface

**24 MCP tools across 8 categories:**

### Store (2 tools)
| Tool | Description |
|------|-------------|
| `memcp_remember` | Save insight as graph node with auto-edges. Secret detection, hash dedup, optional semantic dedup. |
| `memcp_load_context` | Store content as named variable on disk (text or file path) |

### Retrieve (5 tools)
| Tool | Description |
|------|-------------|
| `memcp_recall` | Intent-aware graph retrieval with token budgeting |
| `memcp_search` | Cross-source search (memory + context chunks), auto-selects best tier |
| `memcp_get_context` | Read stored context content or line range |
| `memcp_peek_chunk` | Read specific chunk from chunked context |
| `memcp_filter_context` | Regex filter within context (RLM "grepping") |

### Navigate/Inspect (4 tools)
| Tool | Description |
|------|-------------|
| `memcp_inspect_context` | Metadata + 5-line preview without loading |
| `memcp_list_contexts` | List all stored context variables |
| `memcp_related` | BFS graph traversal from insight |
| `memcp_graph_stats` | Node/edge counts, top entities |

### Manage (3 tools)
| Tool | Description |
|------|-------------|
| `memcp_forget` | Remove insight + all edges |
| `memcp_clear_context` | Delete context + chunks |
| `memcp_chunk_context` | Split context into chunks (6 strategies) |

### Analyze/Feedback (3 tools)
| Tool | Description |
|------|-------------|
| `memcp_reinforce` | Mark insight as helpful/misleading, affects ranking |
| `memcp_consolidation_preview` | Preview similar insight groups (dry-run) |
| `memcp_consolidate` | Merge similar insights into one |

### Lifecycle (3 tools)
| Tool | Description |
|------|-------------|
| `memcp_retention_preview` | Dry-run archive/purge preview |
| `memcp_retention_run` | Execute archive + optional purge |
| `memcp_restore` | Restore from archive |

### Status/Session (4 tools)
| Tool | Description |
|------|-------------|
| `memcp_ping` | Health check |
| `memcp_status` | Memory statistics |
| `memcp_projects` | List projects with stats |
| `memcp_sessions` | Browse sessions |

## 4. Search/Retrieval Pipeline

### 5-Tier Architecture (graceful degradation)

```
Hybrid RRF (BM25 + semantic + graph)    pip install memcp[search,semantic]
Semantic (model2vec/fastembed + cosine)  pip install memcp[semantic]
Fuzzy (rapidfuzz, Levenshtein)           pip install memcp[fuzzy]
BM25 (bm25s, sparse matrix)             pip install memcp[search]
Keyword (stdlib, always available)       built-in
```

**Auto-selection**: hybrid > bm25 > keyword. Each tier checks import-time availability flags.

**Keyword search**: `re.findall(r"\w+", text.lower())` tokenization, set intersection scoring: `score = len(matches) / len(query_tokens)`. Searches content + summary + tags.

**BM25**: Uses bm25s library with in-memory `_BM25Cache`. Corpus-hash-based cache -- invalidated when insights added/removed. Only rebuilds index when corpus changes.

**Fuzzy**: rapidfuzz `partial_ratio()`, threshold 60%.

**Semantic**: model2vec (256d) or fastembed (384d). Brute-force cosine similarity with optional HNSW (usearch) for O(log N). Embedding cache via diskcache by content hash.

**Hybrid RRF fusion**:
```
score(d) = sum(1 / (k + rank_i(d))) for each ranked list
```
k=60 (Cormack et al. 2009). Three-way fusion: BM25 ranked list + semantic ranked list + graph-boosted list. Score-agnostic -- no normalization needed. Also supports legacy alpha-weighted blend (`method="hybrid-alpha"`).

**Cross-source search** (`search_all`): Searches both memory insights (via recall with empty query to get all candidates, then re-scored through search pipeline) and context chunks (scanned from disk). Results combined and re-ranked by score.

**Token budgeting**: `max_tokens` parameter caps cumulative token_count. First result always included.

### Recall Pipeline (graph path)

1. SQL filter by project/session/category/importance
2. Detect intent from query prefix
3. Rank by `keyword_score * 0.7 + edge_boost * 0.3`, adjusted by feedback_score
4. Hebbian strengthen co-retrieved top-10 nodes
5. Lazy edge decay (rate-limited once/hour)
6. Apply token budget
7. Update access_count and last_accessed_at on returned nodes

## 5. Consolidation & Decay

### Consolidation

**Preview** (`memcp_consolidation_preview`): Finds groups of similar insights via:
1. Compute pairwise similarity matrix (embeddings if available, else keyword Jaccard)
2. Union-Find grouping above threshold (default 0.85, configurable)
3. Returns groups sorted by size

**Merge** (`memcp_consolidate`):
- Keeps the most-accessed insight (or specified keep_id)
- Unions all tags and entities
- Keeps highest importance level
- Sums access counts
- Re-points edges from deleted nodes to keeper
- Deletes others

### Decay

**Edge decay**: Exponential, half-life 30 days, min weight 0.05 before pruning. Lazy (during query), rate-limited to once/hour.

**Importance decay**: `effective_importance = base_weight * (1 + log(1 + access_count)) * time_decay`. Time decay = `0.5 ^ (days_since / half_life)`. Critical insights floor at 0.5.

**Auto-pruning**: At 10K insight limit, prune lowest effective_importance (never critical).

### Retention Lifecycle (3-zone)

**Active -> Archive -> Purge**

Archive candidates: last activity >= 30 days, access_count == 0, not immune. Immunity: importance critical/high, access_count >= 3, tags contain "keep"/"important"/"pinned".

Archive: contexts compressed to .gz, insights moved to archive/insights.json.

Purge: permanently deletes after 180 days in archive, logged to purge_log.json.

Restore: decompresses and re-inserts.

### No Sleep-Like Processing

There is no NREM/REM-style sleep consolidation. Consolidation is agent-triggered (explicit `memcp_consolidate` calls), not autonomous. Edge decay is lazy (during query), not scheduled. No background processing or periodic maintenance cycles.

## 6. Comparison Claims

`docs/COMPARISON.md` compares against 5 systems:

**vs rlm-claude**: Same RLM approach. Key diff: MemCP adds MAGMA 4-graph + SQLite backend + sub-agents (vs skills) + intent-aware traversal. rlm-claude uses flat JSON storage.

**vs CLAUDE.md**: Static text file, manual, no persistence, no search, fills context window.

**vs Letta/MemGPT**: Letta requires separate server + vector DB + agent runtime. MemCP is single-process MCP with SQLite. Letta targets general-purpose agents, MemCP targets Claude Code specifically.

**vs MAGMA (paper)**: MemCP is a working implementation. Uses lightweight regex + optional LLM extraction instead of requiring full LLM backbone. Adapts dual-stream write (fast = regex, slow = sub-agent entity extractor).

**vs mem0**: mem0 is embedding-only, primarily cloud-hosted, no graph, no MCP. Claims BM25 outperforms pure embeddings on many retrieval tasks (cites Letta benchmark: filesystem 74.0% vs mem0 68.5%).

**RLM compliance table**: Maps RLM concepts to MemCP tools -- context-as-variable, peeking, grepping, partition, sub-LLM calls, map-reduce, verification.

## 7. Comparison to claude-memory

| Dimension | claude-memory | MemCP |
|-----------|--------------|-------|
| **Search** | RRF hybrid (BM25 + semantic) | 5-tier with RRF (keyword -> BM25 -> fuzzy -> semantic -> hybrid). Same RRF formula. Also supports legacy alpha blend. |
| **Edge schema** | Novelty operator on edges | 4 typed edges (semantic/temporal/causal/entity) with Hebbian co-retrieval strengthening. No novelty operator. |
| **Decay** | Per-category decay with floor, configurable per category | Exponential edge decay (half-life 30d, min 0.05). Importance decay = base * access_boost * time_decay. Critical floor at 0.5. No per-category configuration. |
| **Sleep** | NREM (cluster + merge duplicates) + REM (gap analysis, question generation) | No sleep. Consolidation is agent-triggered via `memcp_consolidate`. No autonomous background processing. |
| **Recall feedback** | `recall_feedback` tool | `memcp_reinforce` tool (similar: helpful/unhelpful). Affects ranking via feedback_score [-1, 1]. Also propagates to connected edges. |
| **JSON sync** | JSON sync layer for portability | SQLite primary + JSON legacy migration. No JSON sync layer. Contexts on filesystem. |
| **Startup** | `startup_load` with token counting | No explicit startup tool. Template CLAUDE.md instructs `memcp_recall(importance="critical")` at session start. |
| **Shadow-load** | shadow-load of graph state at startup | No shadow-load. Sub-agents have independent context windows. |
| **Storage** | SQLite + JSON sync | SQLite (graph.db) + filesystem (contexts/chunks). JSON is legacy only. |
| **Context management** | Not a focus | Major feature: named variables, 6 chunking strategies, peek/filter/inspect (RLM pattern) |
| **Graph traversal** | Edge-based recall | Intent-aware traversal + explicit BFS via `memcp_related`. Richer graph structure (4 types vs generic edges). |
| **Entity extraction** | None built-in | Regex + optional spaCy + optional LLM sub-agent |
| **Hook system** | None | Auto-save hooks (pre-compact, progressive reminders) |
| **Sub-agents** | None | 4 Claude Code sub-agents for map-reduce analysis |
| **Tool count** | ~10 | 24 |

### Key Architectural Differences

1. **Graph richness**: MemCP's 4-type edge system with intent-aware routing is more structured than claude-memory's generic edges. The "why did we choose X?" -> causal edges pattern is genuinely useful for decision archaeology.

2. **No autonomous maintenance**: claude-memory's sleep cycles (NREM/REM) are a significant differentiator. MemCP has no equivalent -- consolidation requires explicit agent action. Edge decay is lazy, not proactive.

3. **Context management**: MemCP treats large documents as first-class citizens (RLM pattern). claude-memory doesn't focus on this; it relies on Claude Code's built-in Read/Grep.

4. **Startup protocol**: claude-memory's `startup_load` with token counting and shadow-load is more sophisticated than MemCP's "recall critical at session start" instruction.

5. **Causal detection is fragile**: Pattern-matching for "because"/"therefore" has high false-positive rate. The ADR acknowledges this.

## 8. Worth Stealing

### Tier 1: High Value, Moderate Effort

1. **Intent-aware query routing** (effort: 2-3 hours). Map query prefixes ("why", "when", "who") to edge type preferences during recall. claude-memory has edge types but doesn't route by query intent. Would require: intent detector function + edge_type weighting in recall scoring. This is the single most compelling feature -- it makes the graph structure *useful* during recall rather than just stored.

2. **Entity extraction on remember** (effort: 3-4 hours). Regex extraction of file paths, module names, CamelCase identifiers, URLs from content at store time. Auto-generate entity edges. Would require: regex extractor, entity field on memories, entity-based edge creation. Good synergy with existing edge schema.

3. **Named context variables / RLM pattern** (effort: 8-12 hours). Store large content on disk, expose peek/filter/chunk operations. This is orthogonal to memory -- it's about managing large documents across sessions without filling the context window. However, Claude Code's built-in Read/Grep already provides most of this functionality natively. The value-add is mainly the chunking strategies and token-aware inspection.

### Tier 2: Medium Value, Low Effort

4. **Feedback propagation to edges** (effort: 1 hour). When `recall_feedback` marks a memory as helpful/unhelpful, also boost/weaken connected edges. MemCP does +0.02/-0.05 on edges. Simple extension to existing recall_feedback.

5. **BM25 cache with corpus-hash invalidation** (effort: 1-2 hours). Instead of rebuilding BM25 index on every search, cache it and invalidate on remember/forget. Already relevant if search volume is high.

6. **Retention immunity rules** (effort: 1 hour). Protect high-importance, frequently-accessed, and tagged ("keep"/"pinned") memories from decay/archival. claude-memory has per-category floors, but explicit immunity tags add user control.

### Tier 3: Nice to Have

7. **Semantic deduplication on store** (effort: 2 hours). Before storing, check cosine similarity against existing memories. Reject near-duplicates above threshold (0.95). Reduces drift.

8. **Graph statistics tool** (effort: 30 min). Expose node count, edge counts by type, top entities. Low effort, useful for diagnostics.

## 9. Not Worth It

1. **24-tool surface area** -- Too many tools. 24 tools means high system prompt overhead and decision fatigue for the agent. claude-memory's ~10 tools are already near the upper bound of useful. Many of MemCP's tools (peek_chunk, filter_context, chunk_context) replicate what Claude Code does natively with Read/Grep.

2. **Context-as-variable / RLM pattern** (as a separate system) -- Claude Code already has Read, Grep, Glob. Adding a parallel file management layer creates confusion about which system to use. The RLM paper is about models that can't natively read files; Claude Code can.

3. **Sub-agent architecture for memory** -- Launching Haiku sub-agents for entity extraction or map-reduce adds latency and cost. For a memory system, the overhead doesn't justify the benefit vs. doing extraction inline.

4. **Causal edge detection via keyword patterns** -- "because", "therefore" etc. have high false-positive rates. The ADR acknowledges this. Without LLM-based verification, causal edges are noise. Intent-aware routing on top of noisy causal edges is worse than no causal edges.

5. **Hook system (pre-compact, reminders)** -- Clever idea but invasive. Modifying `~/.claude/settings.json` to inject hooks couples the memory server to Claude Code's hook system. If the hook breaks, it blocks /compact. CLAUDE.md instructions achieve the same behavioral outcome without system-level coupling.

6. **Legacy JSON backend + migration** -- Supporting two backends (SQLite + JSON) adds complexity for backward compatibility with a v0.1 format. Not relevant to claude-memory which started with SQLite.

7. **HNSW vector index** -- Premature optimization. With <10K memories, brute-force cosine is fast enough. usearch adds a native dependency for negligible benefit at typical memory scales.

8. **Fuzzy search tier** -- rapidfuzz for typo tolerance sounds good but in practice, typos in memory queries are rare. BM25 + semantic already handle paraphrasing, which is the real recall challenge.
