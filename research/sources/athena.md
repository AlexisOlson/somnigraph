# Athena — Source Analysis

*Phase 12, 2026-02-20. Analysis of winstonkoh87/Athena-Public.*

---

## 1. Architecture Overview

**Language**: Python 3.10+ (`pyproject.toml` requires `>=3.10`, author uses 3.13)

**Primary dependencies**: `supabase>=2.0.0`, `google-generativeai>=0.7.0`, `sentence-transformers>=2.2.0`, `flashrank>=0.2.9`, `dspy-ai>=3.1.3`, `fastmcp`, `torch`

**Storage backend**: Supabase with pgvector (cloud). Optional local ChromaDB for GraphRAG entity vectors. Optional local SQLite FTS5 (`exocortex.db`) built from DBPedia.

**Embeddings**: Google `gemini-embedding-001` (3072-dim), accessed via REST API. Cached locally to a JSON file via MD5 hash keys.

**High-level structure**:

```
src/athena/
    memory/      -- vectors.py (Supabase client, embedding calls), sync.py (delta manifest), delta_manifest.py
    core/        -- config.py (path constants), models.py (SearchResult dataclass), governance.py, permissions.py, cache.py
    tools/       -- search.py (964-line hybrid RAG orchestrator), reranker.py (CrossEncoder), agentic_search.py
    boot/        -- orchestrator.py, loaders/memory.py, etc.
    sessions.py  -- session log lifecycle, checkpoint appending, lineage links
    mcp_server.py -- FastMCP server (9 tools, 2 resources)

scripts/core/
    smart_search.py  -- standalone example with sentence-transformers fallback
    reflection.py    -- ReflectionExtractor (lesson/anti-pattern/checklist storage)
    boot_knowledge.py -- parses User_Profile_Core.md, prints active constraints to stdout
    auditor.py       -- TrilateralAuditor (cross-model disagreement detection)
    retrieval/graphrag.py -- KnowledgeGraphParser for KNOWLEDGE_GRAPH.md

examples/scripts/
    nocturnal_consolidation.py -- DreamState (session compression + conflict check via LLM)
    memory_compressor.py       -- atomic fact extraction from text (Protocol 104)
    distill_sessions.py        -- per-session insight extraction + caching
    pattern_recognition.py     -- ChatAnalyzer, SessionAnalyzer, CostAnalyzer
```

The physical repo is a **public portfolio of a private system**. The private system lives in a separate `Athena/` directory with directories `.framework/`, `.context/`, `.agent/` that are not in this repo. The public repo provides the Python SDK, examples, documentation, and templates, but the actual user data (session logs, protocols, case studies, the Supabase contents) is the user's private instance.

---

## 2. Memory Type Implementation

**There is no single memory schema**. Athena's memory is distributed across eleven Supabase tables, each typed by content domain:

| Table | Content type |
|---|---|
| `sessions` | Session logs (date, session_number, content, embedding, file_path) |
| `case_studies` | Pattern analysis documents (code, name) |
| `protocols` | Reusable decision frameworks (code, name) |
| `capabilities` | Tool/skill definitions |
| `playbooks` | Strategic guides |
| `workflows` | Automation scripts |
| `references` | External citations |
| `frameworks` | Core identity modules |
| `user_profile` | User preferences, constraints, psychology |
| `system_docs` | TAG_INDEX, manifests |
| `entities` | External import data (Telegram exports, etc.) |

All tables follow the same pattern: `id SERIAL`, `content TEXT`, `embedding VECTOR(3072)`, `file_path TEXT UNIQUE`, plus domain-specific fields. Similarity search uses pgvector cosine distance via PostgreSQL RPC functions.

**Memory Bank** (four markdown files: `activeContext.md`, `userContext.md`, `productContext.md`, `systemPatterns.md`) is a flat-file layer intended for O(1) boot cost. These are human-readable, loaded verbatim into context on every session start.

**Session logs** are structured markdown files with YAML frontmatter containing `prev_session`, `next_session`, `focus`, `threads`, and `status`. The `sessions.py` module maintains forward/backward lineage by patching the previous session's `next_session` field when a new session is created.

**Reflections** are stored as JSONL records in `.context/reflections/reflections.jsonl` with types: `lesson`, `anti_pattern`, `checklist_item`. Also mirrored to `LESSONS_LEARNED.md`.

**Episodic compressions** from `nocturnal_consolidation.py` are saved to `.context/memories/episodic/` as markdown summaries.

**Distilled insights** from `distill_sessions.py` are saved to `.context/memories/session_insights/` at one file per session.

**No priority/decay system exists**. There are no `base_priority`, `decay_rate`, or access-count fields anywhere in the schema. Every indexed document is treated with equal weight (though RRF weights differ by table/type at query time).

---

## 3. Retrieval Mechanism

When `smart_search(query)` is called via MCP, it calls `run_search()` in `search.py`, which executes up to seven parallel collectors via `ThreadPoolExecutor`:

1. **`collect_canonical`**: Reads `CANONICAL.md` (user's "constitution" file) and scores it via simple keyword overlap. Always returns it if the file exists.

2. **`collect_tags`**: Runs `grep -i -m 10 <query> TAG_INDEX_A-M.md TAG_INDEX_N-Z.md` in a subprocess. Parses the matching lines as SearchResult objects. Weight 2.2x in RRF.

3. **`collect_vectors`**: Calls `get_embedding(query)` (Gemini API, cached), then fires all eleven Supabase RPC functions in parallel via a nested `ThreadPoolExecutor`. Each table returns up to 10 results at threshold 0.3. Results are typed individually (case_study=3.0x, session=3.0x, protocol=2.8x, etc.).

4. **`collect_graphrag`**: Invokes `query_graphrag.py` as a subprocess with `--json --global-only`. This script (not in the public repo) searches `communities.json` for keyword matches in community member lists and queries ChromaDB for entity vectors. Weight 2.5x.

5. **`collect_filenames`**: Uses `subprocess.run(["find", project_root, "-name", "*.md"])` and fuzzy-matches filename stems against query tokens. Weight 2.0x.

6. **`collect_sqlite`**: Queries `exocortex.db` (DBPedia FTS5 index) if it exists. Weight 1.5x.

7. **`collect_framework_docs`**: Reads markdown files from `.framework/v8.2-stable/modules/`. Weight 2.5x.

All collectors return lists of `SearchResult` objects. `weighted_rrf()` merges them:

```
score = sum(weight[source] * 1/(60 + rank))
```

Optional `rerank=True` passes the top candidates through `cross-encoder/ms-marco-MiniLM-L6-v2` via sentence-transformers.

**Agentic search** (`agentic_search.py`) adds a rule-based query decomposition layer on top: splits on conjunctions/commas/question patterns, runs each sub-query through the full search pipeline in parallel, deduplicates results with ID collision boosting, and optionally validates results via cosine similarity against the original query embedding. The decomposition is purely regex-based — no LLM involved.

**Overclaiming flags**:

- The GRAPHRAG.md doc describes 1,460 community clusters and 2.3MB entity index. These exist in the user's private instance, not in the public repo. The public repo's `query_graphrag.py` is not included — `collect_graphrag()` invokes it as a subprocess and returns empty if the script isn't found.

- The README claims "5-source retrieval + RRF fusion" but the actual code has seven distinct collectors. This is an undercount.

- The README claims "<200ms" semantic search. With Gemini embedding API calls (network round-trip) plus eleven parallel Supabase RPC calls, this is only plausible if the embedding is cached and Supabase is co-located. First-call latency is realistically 500ms-2s.

- "No config files. No API keys." in the quickstart is false. `vectors.py` requires `GOOGLE_API_KEY`, `NEXT_PUBLIC_SUPABASE_URL`, and `SUPABASE_SERVICE_ROLE_KEY` from environment. Local mode is available but lightly documented.

---

## 4. Standout Feature

**The agentic search pipeline's query decomposition with multi-source validation** is the single most complete novel feature in the codebase. The implementation in `agentic_search.py` is genuinely well-thought-out:

1. Rule-based decomposition (regex patterns, no LLM cost) splits compound queries into up to 4 sub-queries.
2. Each sub-query runs the full 7-collector RRF pipeline in parallel.
3. Documents found by multiple sub-queries get a 10% score boost.
4. An optional validation pass embeds each result's content and checks cosine similarity against the original query, filtering noise below threshold 0.25.
5. Provenance is tracked per result: `found_by` lists which sub-queries retrieved it and `multi_source` flags cross-corroboration.

This is a real implementation of multi-angle retrieval that doesn't require LLM calls in the hot path. The provenance metadata is novel: it lets downstream reasoning know which results are corroborated across multiple angles of the query, providing a signal for confidence that most systems don't surface.

---

## 5. Other Notable Features

**Delta manifest sync** (`delta_manifest.py`): A SHA-256 + size/mtime two-stage check determines whether files need to be re-embedded before syncing to Supabase. The O(1) quick-check (size+mtime) avoids hashing unchanged files. Atomic writes via `tempfile.mkstemp + os.replace`. Thread-safe with Lock. This is solid production-grade incremental sync logic.

**Governance Triple-Lock** (`governance.py`, `mcp_server.py`): A state machine enforces that both semantic search and web search are performed before `quicksave` is allowed. The `quicksave` MCP tool checks compliance and flags violations in the response. This is a behavioral constraint baked into the tool interface — unusual and interesting as a governance pattern.

**Secret Mode** (`permissions.py`): A public/internal/secret three-tier sensitivity classification on all tools and data domains. Toggling secret mode blocks internal tools and redacts sensitive content. The tool registry pre-declares permission level, sensitivity tier, and description for every tool. Clean design.

**Nocturnal consolidation** (`nocturnal_consolidation.py`): A `DreamState` class that sweeps session logs from the past 24 hours, sends them to Gemini with a structured JSON extraction prompt comparing against `CANONICAL.md`, and produces a daily report listing conflicts, new facts, and obsolete facts. This is the sleep-like process.

**Session lineage** (`sessions.py`): Every session log has `prev_session` and `next_session` YAML fields, and the code actively patches `next_session` in the previous log when a new session is created. This creates a doubly-linked list of sessions across time.

**Exocortex** (`docs/EXOCORTEX.md` + `examples/scripts/exocortex.py`): Offline DBPedia FTS5 search over Wikipedia abstracts, integrated into the main RRF pipeline as a low-weight source. Clever use of a one-time 6GB database build for zero-cost fact lookup.

**TrilateralAuditor** (`scripts/core/auditor.py`): Framework for routing high-stakes outputs to multiple LLMs for independent review, aggregating disagreements. Currently requires model clients to be configured and is described as a framework, but the structure is complete.

**Token budget gauge** (`boot/loaders/token_budget.py`): Measures boot file sizes, enforces a 15K token hard cap, and auto-compacts `activeContext.md` if the cap is exceeded. The boot sequence displays a visual gauge of context usage.

---

## 6. Gap Ratings

| Gap | Rating | Evidence |
|-----|--------|----------|
| Layered Memory | 72% | Three-tier architecture is explicitly designed: live session (full fidelity) -> session log (~15%) -> activeContext.md entry (~5%) -> absorbed into userContext.md (~0.1%). The `memory_compressor.py` and `distill_sessions.py` tools implement the compression steps. The Memory Bank's 15K boot cap with auto-compaction enforces the O(1) boot guarantee. Missing: no automated triage between layers during session (it's script-driven, not automatic), and the progressive distillation path requires manual invocation or cron. |
| Multi-Angle Retrieval | 85% | `agentic_search.py` decomposes queries into sub-queries and runs each through the full pipeline. Seven distinct source types (canonical, tags, vectors-by-domain, graphrag, filenames, sqlite, framework_docs) feed into weighted RRF. Per-domain vector weights provide differential signal. Cross-sub-query provenance tracking is genuinely novel. Gap: decomposition is regex-only (misses paraphrase/semantic decomposition); no query expansion or synonym enrichment. |
| Contradiction Detection | 40% | `nocturnal_consolidation.py` sends session logs + CANONICAL.md to Gemini with a prompt asking for conflict detection. The router (`orchestration/router.py`) has a `CONTRADICTORY` escalation signal. `auditor.py` extracts `contradictions` fields from model responses. But none of this is automated or continuous — it requires nightly job execution, and the contradiction detection is entirely prompt-based (no structural comparison). No contradiction graph or resolution tracking. |
| Relationship Edges | 45% | `scripts/core/retrieval/graphrag.py` defines typed edges (`source --RELATION--> target`) parsed from a markdown KNOWLEDGE_GRAPH.md. `generate_graph_vis.py` builds a Vis.js graph from markdown wikilinks. Session logs have prev/next session YAML fields creating a temporal chain. But the graph is manually maintained, relationship types are not consistently defined, and the Vis.js graph is structural (file links), not semantic (memory relationships). No SQL/graph DB for edges. |
| Sleep Process | 55% | `nocturnal_consolidation.py` is a real implementation: sweeps recent logs, LLM-based conflict check, LLM-based episodic compression, daily report generation. `distill_sessions.py` extracts or generates per-session insights. `memory_compressor.py` extracts atomic facts. `pattern_recognition.py` runs meta-analysis across sessions. Gap: these are all standalone scripts requiring manual or cron invocation, not an integrated automated pipeline. No synthesis into new long-term memory nodes. The compression outputs go to files but are not automatically synced back to Supabase. |
| Reference Index | 70% | TAG_INDEX (sharded A-M, N-Z) with 1000+ tags, auto-generated by `generate_tag_index.py`. Integrated as a search source with grep-based lookup. `KNOWLEDGE_GRAPH.md` provides a compressed concept index. Session count, protocol count, case study count are tracked in doc metadata. `athena doctor` command checks system health. Missing: no lightweight query-able stats API; no automatic summary of memory contents by domain; token usage is tracked per file but not per domain. |
| Temporal Trajectories | 35% | Session logs are doubly-linked (prev/next pointers). Pattern recognition (`pattern_recognition.py`) analyzes sessions for recurring themes. Temporal analysis in `ChatAnalyzer` tracks message initiation and latency over time. `nocturnal_consolidation.py` identifies new vs. obsolete facts relative to CANONICAL. But there is no structured tracking of how specific beliefs or preferences change across sessions. No "belief at time T" tracking, no contradiction-with-prior-self detection, no preference drift measurement. The chain exists as linked files but is not mined as a trajectory. |

---

## 7. Comparison with claude-memory

**Stronger — what Athena does better:**

- **Breadth of retrieval sources**: Seven heterogeneous collectors (vector, tag index, GraphRAG communities, filename, FTS5/exocortex, framework docs, canonical file) vs. claude-memory's two (FTS5 + sqlite-vec merged via RRF). The multi-collector architecture catches things a single-modality search misses.
- **Multi-angle decomposition**: `agentic_search.py`'s sub-query decomposition with cross-source boosting is more sophisticated than anything in claude-memory. Provenance tracking per result is novel.
- **Session structure and governance**: Structured YAML-frontmatter session logs, the Triple-Lock governance enforcement, the token budget gauge with auto-compaction — Athena has a coherent session lifecycle that claude-memory lacks entirely.
- **Sleep-style consolidation script**: `nocturnal_consolidation.py` is a working LLM-based conflict detection + episodic compression pipeline. Claude-memory's sleep skill does relationship classification but not free-form conflict analysis.
- **Exocortex**: DBPedia offline FTS5 as a fact-checking layer is genuinely creative and not something claude-memory has.
- **Cross-encoder reranking**: FlashRank/MiniLM cross-encoder as an optional reranking stage. Claude-memory has RRF fusion but no cross-encoder re-scoring.
- **Scale**: 11 domain-typed tables, 850+ indexed documents, community detection via Leiden algorithm — built for a larger personal knowledge base than claude-memory's single table.

**Weaker — what claude-memory does better:**

- **Human-in-the-loop curation**: Claude-memory's `review_pending()` workflow — where memories are proposed and confirmed/edited by the human before persisting — is entirely absent from Athena. Athena's memories are either full documents (no curation) or LLM-summarized without human review.
- **Dimensional recall feedback**: Claude-memory's relevance/impact/surprisal scoring on retrieved memories feeds back into priority and decay. Athena has no feedback loop from recall quality to storage priority.
- **Priority and decay**: Claude-memory has `base_priority` (1-10), configurable `decay_rate` per category, and access count — a whole relevance lifecycle. Athena treats all stored documents equally; no document degrades or gains salience over time based on usage.
- **Local-first, offline**: Claude-memory runs entirely locally (SQLite + FastEmbed ONNX). Athena's primary path requires two external API calls (Gemini for embeddings, Supabase for storage). The exocortex is local but the main memory is cloud-dependent.
- **Single coherent model**: Claude-memory has one table with a clear schema. Athena's memory is fragmented across 11 tables, flat files in multiple directories, GraphRAG JSON blobs, and YAML frontmatter — operationally complex.
- **Contradiction and relationship at the memory level**: Claude-memory's sleep skill classifies explicit relationship edges (SUPPORTS, CONTRADICTS, EVOLVES_FROM, etc.) between memory records. Athena's contradiction detection is LLM-prompt-based at the document level, not structured edge classification.
- **No vendor lock-in**: Claude-memory's SQLite is fully portable. Athena's primary storage is Supabase (though files exist locally as markdown source).

---

## 8. Insights Worth Stealing

1. **Multi-source collector architecture with per-source RRF weights** (Effort: Medium, Impact: High). The pattern of having named collectors that each return `List[SearchResult]` with a `source` field, then running `weighted_rrf(lists)` where weights are configurable per source — is cleaner and more extensible than claude-memory's two-path FTS5+vector merge. Adding a new retrieval modality means adding one collector function and one weight entry. The sub-typing of vectors by domain (case_study=3.0x, session=3.0x, protocol=2.8x) within the same RRF framework is directly applicable.

2. **Query decomposition with cross-sub-query boosting** (Effort: Low, Impact: High). The `agentic_search.py` approach — rule-based split on conjunctions, run full search per sub-query, boost results found by multiple sub-queries — is a low-cost improvement with real precision gains on compound queries. The provenance tracking (`found_by`, `multi_source`) provides confidence signals with no added infrastructure.

3. **Token budget gauge with auto-compaction** (Effort: Medium, Impact: Medium). Measuring boot-load token costs, enforcing a hard cap, and auto-compacting the rolling context file when the cap is exceeded is directly applicable to claude-memory's `startup_load()`. The idea of surfacing a visual "budget gauge" to the user at session start is a UX win.

4. **Delta manifest for embedding sync** (Effort: Low, Impact: Medium). The two-stage O(1) size/mtime check before SHA-256 hash before sync is solid. For claude-memory, this would apply if embeddings ever need to be recomputed (e.g., model version change). The atomic write pattern (tempfile + os.replace) is worth copying directly.

5. **Distilled insights as first-class search targets** (Effort: Medium, Impact: Medium). `distill_sessions.py` extracts 3-5 bullet insights per session and stores them as separate, searchable files. This means searches hit compressed signal rather than full verbose logs. Claude-memory could apply this by summarizing older episodic memories and storing the summary as a separate lower-priority memory pointing back to originals.

6. **Session lineage via doubly-linked YAML pointers** (Effort: Low, Impact: Medium). The `prev_session` / `next_session` linking in session YAML frontmatter is cheap to implement and provides a traversable temporal chain. Claude-memory has timestamps but no explicit linking. Adding `preceded_by` / `succeeded_by` edges to the edge table during sleep processing would give the same benefit structurally.

7. **Governance enforcement at the tool interface** (Effort: Low, Impact: Low-Medium). Embedding behavioral constraints (Triple-Lock) into the MCP tool's response — flagging violations rather than blocking — is a useful pattern for surfacing protocol compliance without hard failure. Claude-memory's `quicksave` analogue (`remember()`) has no equivalent compliance check.

8. **Cross-encoder reranking as optional step** (Effort: Low, Impact: Low-Medium). The optional `rerank=True` path through `cross-encoder/ms-marco-MiniLM-L6-v2` is a well-isolated addition. Adding this as an optional step to claude-memory's `recall()` would improve precision on complex queries at the cost of ~100ms local inference.

---

## 9. What's Not Worth It

**GraphRAG community detection**: Building the Leiden-algorithm community graph costs $30-50 in LLM API fees per build (Athena's own docs acknowledge this), requires NetworkX + ChromaDB + a separate indexing pipeline, and returns community-level results that may not map to specific memories. The Athena docs themselves recommend VectorRAG instead for most cases. The `collect_graphrag()` function calls an external subprocess that isn't even in the public repo. Not worth the complexity for a personal memory system at claude-memory's scale.

**Supabase/pgvector as primary storage**: The cloud dependency adds latency (embedding API + Supabase round-trips), requires credential management, has free tier limits, and creates a network reliability dependency. Claude-memory's SQLite + sqlite-vec is strictly better for local-first personal memory. The exocortex's use of local SQLite FTS5 is actually the pattern to extend, not Supabase.

**Eleven-table domain splitting**: Maintaining separate tables per content type (sessions, case_studies, protocols, capabilities, etc.) with per-domain Supabase RPC functions creates substantial maintenance overhead. When adding a new content type, you write a new migration, a new search function, a new wrapper in vectors.py, and a new weight entry. Claude-memory's single-table model with a `category` field and configurable query categories is operationally simpler and handles new memory types with zero schema changes.

**The session log template's `[Lambda+XX]` cognitive load tracking**: The `extract_lambda_stats()` function parses self-reported latency indicators that the AI is supposed to append to each response. This is effectively asking the AI to self-report its own complexity scoring, which is neither reliable nor consistent across model versions. The feature adds template noise without producing usable signal.

**DreamState/nocturnal consolidation as currently designed**: The conflict detection sends all session logs plus CANONICAL.md to a single Gemini call with a structured JSON prompt. At scale (1000+ sessions), this will exceed context limits and produce unreliable JSON. The pattern is right but the implementation isn't ready for production use. More targeted extraction (per-session or per-topic) would be necessary before this is worth adopting.

---

## 10. Key Takeaway

Athena is a genuine, working personal memory system built by a sophisticated practitioner who used it heavily (1000+ sessions), and the engineering reflects that experience — the delta manifest, the thread-safe caching, the governance enforcement, the parallel collectors, the agentic decomposition are all evidence of real-world refinement. The standout insight is the multi-source weighted RRF architecture: treating each retrieval modality (tags, vectors, graph communities, filenames, FTS5) as a named, independently-weighted input to a fusion step is cleaner and more extensible than anything claude-memory currently has. The weak spots are structural: distributed over eleven tables and multiple flat-file systems, cloud-dependent, with no priority/decay lifecycle and no human-in-the-loop curation — the system accumulates data but has no mechanism for forgetting, degrading stale information, or giving the human meaningful control over what persists. Claude-memory's disciplined single-table model with human review, priority decay, and structured relationship edges is architecturally sounder for long-term personal memory; Athena's multi-angle retrieval and query decomposition machinery is worth grafting onto it.
