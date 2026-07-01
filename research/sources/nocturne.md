# nocturne — Human-curated tree/graph memory for MCP agents, FTS-only, with visual diff/rollback review

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

Nocturne Memory (repo `Dataojitori/nocturne_memory`) is an MCP long-term memory *server* built around a **hand-curated concept tree**, a **React/Vite review dashboard**, and **deliberate rejection of automation** (no vector search, no auto-extraction, no auto-dedup, no decay). Its pitch is identity persistence for a "sovereign AI" companion across LLM backends, not benchmark retrieval. FastAPI backend, SQLite or PostgreSQL, `mcp` (FastMCP) server exposing 7 tools.

### Storage & Schema
SQLAlchemy async ORM (`backend/db/models.py`). The data model separates **identity from content**, which is the one genuinely clean idea here:

- **`Node`** — a conceptual entity with a stable `uuid`. Persists across content edits. Tracks `last_accessed_at`.
- **`Memory`** — one *content version* of a node (`content` Text, `deprecated` bool, `migrated_to` → next version's id). Editing content creates a new Memory row and chains the old one; the node uuid never changes, so graph structure is untouched by edits.
- **`Edge`** — directed parent→child between nodes, carrying `name`, `priority` (0 = highest), and `disclosure` (a human-readable "when should this be recalled" trigger string). Unique on `(parent_uuid, child_uuid)`.
- **`Path`** — materialized `(namespace, domain, path)` → edge URI cache (e.g. `core://user/health`). Aliases = multiple Paths to one edge.
- **`GlossaryKeyword`** — keyword → node binding ("豆辞典").
- **`SearchDocument` / `search_documents_fts`** — derived FTS index rows, one per reachable path.
- **`MemoryAccessLog`** — async read-frequency log.
- **`Preset`** — named boot-URI sets (which memories load at session start).

There is a **single-active-memory-per-node** invariant (migration 007) and a sentinel `ROOT_NODE_UUID` to dodge SQLite's `NULL != NULL` uniqueness quirk. Note the `neo4j_client.py` + `migrate_neo4j_to_sqlite.py`: an earlier Neo4j graph backend was migrated down to SQLite — the "graph" is now plain relational adjacency, not a graph DB.

### Memory Types
No category taxonomy (no episodic/semantic/procedural). Organization is purely the **URI namespace tree**: `domain://path` where domains are configurable (`core`, `writer`, etc.), plus a `namespace` column for **multi-agent isolation** (separate AI personas, e.g. "Alice"/"Bob", in one instance — not swarm collaboration). Priority is a 0–N integer per edge, lower = more important.

### Write Path
Fully **agent-authored and human-reviewed; zero automatic extraction or gating.** MCP tools (`mcp_server.py`): `create_memory(parent_uri, content, priority, disclosure, title?)`, `update_memory` (Patch mode `old_string/new_string` — must be unique-match — or Append; **no full-replace mode**, by design, to prevent accidental overwrite), `delete_memory` (removes one *path*, not the content), `add_alias`, `manage_triggers` (bind glossary keywords). `text_patch.py` does fuzzy/normalized matching (curly-quote and dash folding, NFC) so the LLM's re-emitted `old_string` still matches after character drift — a small, practical touch. Every mutation records before/after row state into a **changeset** (`snapshot.py`, JSON at `snapshots/changeset.json`, file-locked) with overwrite semantics (first touch freezes `before`, later touches overwrite `after`, net-zero filtered). The human then reviews diffs in the dashboard and accepts/rejects — this is the "time-travel / rollback" feature.

### Retrieval
**Lexical full-text only — explicitly NOT semantic.** `backend/db/search.py`:
- SQLite path: FTS5 with an explicit `bm25()` weighting over columns `(namespace, domain, path, node_uuid, uri, content, disclosure, search_terms)` = weights `(0, 0, 2.5, 0, 2.0, 1.0, 1.0, 0.75)` — i.e. **path and uri dominate**, content is 1.0, disclosure/search_terms lower. Query built by `_to_sqlite_match_query`: AND of quoted tokens after `expand_query_terms` (jieba CJK segmentation in `search_terms.py`).
- PostgreSQL path: `to_tsvector('simple', …) @@ websearch_to_tsquery` with `ts_rank_cd`.
- Ordering: `score, priority ASC, len(path) ASC`; dedup by node_uuid; candidate_limit = 5×limit.

No vector store, no embeddings, no RRF, no learned reranker, no fusion — I grep-confirmed zero `embed/vector/faiss/cosine/sqlite-vec` in backend code. Retrieval is therefore agent-driven **tree navigation** (`read_memory` returns content + children list + metadata) plus keyword FTS, augmented by two lateral channels:
1. **Glossary auto-linking**: when a bound keyword literally appears in content being read, a hyperlink footnote to the bound node is appended. A manual, curated stand-in for co-retrieval edges.
2. **`disclosure` triggers**: free-text recall conditions stored per edge and indexed into FTS — they do not *fire* automatically; they're just extra searchable/human-facing hints.
- Special system URIs: `system://boot` (preset core memories), `system://index/<domain>`, `system://recent/N`, `system://glossary`, `system://diagnostic/<domain>`, `system://random/<domain>` (weighted random pick — a "spontaneous recall / dreaming" gesture).

### Consolidation / Processing
**None automated.** No sleep, no LLM-mediated merge, no edge inference. The nearest analog is `get_diagnostics()` (`graph.py:438`), a **health-audit report** for humans: flags *stale* nodes (staleness threshold derived from priority — pri 0 → 3 days, then `3.5 * 2^prio`, capped), *crowded* parents (too many children), and *orphan* nodes. It surfaces problems for human curation; it never acts. `docs/skills/memory-audit*/SKILL.md` are prompt-skills instructing the agent to periodically run this audit and propose cleanups — consolidation-as-checklist, not as pipeline.

### Lifecycle Management
Versioning is the strong suit: `migrated_to` chains, `deprecated` soft-delete, `rollback_to_memory()` / `restore_path()` / `restore_orphan_memory()` / `permanently_delete_memory()`, orphan recovery, and full changeset diff review with accept/reject. **No decay model** (the diagnostic staleness is human-facing, not a score input). 14 numbered migrations show real production evolution (namespace, FTS, access logs, presets, cascade paths).

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| "One soul, any engine" — LLM-agnostic memory | MCP stdio/SSE server, no model coupling | Valid but trivially true of any MCP memory server (incl. Somnigraph) |
| Human-controlled versioning / time-travel rollback | `snapshot.py` changeset + dashboard diff review + `rollback_to_memory` | Validated in code; genuinely built out |
| Trigger conditions ("when to recall") bound to memories | `disclosure` field + `manage_triggers` glossary | Real field, but triggers do **not** auto-fire — searchable/curated hints only |
| Full-text, not semantic search, is a deliberate choice | `search.py` FTS5/tsvector only; no embeddings anywhere | Validated; also its biggest retrieval ceiling |
| Multi-agent namespace isolation | `namespace` PK column throughout + tests | Validated (isolation, not collaboration) |
| Retrieval quality / benchmark performance | **No LoCoMo/PERMA/R@k numbers anywhere** | Unbenchmarked — not comparable to Somnigraph's 85.1 LoCoMo QA |

---

## Relevance to Somnigraph

### What nocturne does that Somnigraph doesn't
- **Explicit content-versioning with human diff review + one-click rollback.** Node/Memory separation + changeset before/after + dashboard accept/reject. Somnigraph has `valid_from/valid_until` supersession detected during `sleep_nrem.py`, but no human-in-the-loop diff-review UI and no clean rollback of a bad automated merge. This is a real **write-path quality gate** — the exact capability STEWARDSHIP lists as missing — implemented as human review rather than a learned gate.
- **Write-time recall-condition authoring (`disclosure`).** The agent records *when* a memory should resurface, not just its content/themes. Somnigraph's `embeddings.py` enriches with category+themes+summary but has no "trigger condition" field.
- **URI/tree addressing + progressive disclosure.** Memories are navigable by stable `domain://path` URIs with children listings and `system://index`, letting the agent walk structure rather than only receiving a ranked list. Somnigraph is flat + ranked (`reranker.py`).
- **Self-audit diagnostic view** (stale/crowded/orphan) as an agent-runnable skill.

### What Somnigraph does better
- **Retrieval, decisively.** Somnigraph has hybrid BM25+vector, RRF fusion (k=14), a 26-feature LightGBM reranker (NDCG 0.7958), PPR graph expansion, and a measured feedback loop (Spearman r=0.70). Nocturne is BM25/tsvector over a path-weighted index with `priority` tie-break — no semantic recall, no fusion, no learning. On any vocabulary-gap / multi-hop query nocturne depends entirely on the human having pre-authored the right path or keyword.
- **Automated consolidation.** `sleep_nrem.py`/`sleep_rem.py` infer typed edges, merges, and gaps; nocturne infers nothing (`graph.py` only reports).
- **Decay / salience dynamics** — `scoring.py` per-category decay with reheat vs. nocturne's none.
- **Benchmark grounding** — Somnigraph is measured; nocturne is not.

---

## Worth Stealing (ranked)

### 1. `disclosure` — a write-time "recall condition" field (Low)
**What**: Let the write path capture a short natural-language "when should this resurface" string alongside content, and index it into retrieval.
**Why**: Directly addresses the multi-hop vocabulary gap Somnigraph named as its retrieval ceiling — the author knows future query framing the content text won't contain ("when I'm about to post to Bluesky"). It's a cheap, human/agent-supplied query-side bridge, complementary to synthetic-node bridges from the graph pipeline.
**How**: Add an optional `disclosure`/`recall_cue` column in `db.py`; concatenate into the enriched embedding in `embeddings.py` and add it as an FTS-boosted field in `fts.py`; expose as an optional `remember()` arg in `tools.py`. Testable as an ablation on the LoCoMo multi-hop subset.

### 2. Node-vs-version separation with human diff-review rollback (Medium)
**What**: Stable node identity + append-only content versions + a before/after changeset that a human can review and reject.
**Why**: Somnigraph's sleep merges/supersessions are irreversible-ish and unreviewed; a mis-merge silently corrupts state. A changeset log + rollback would make `sleep_nrem.py` merges safe to run more aggressively.
**How**: Record pre/post rows for each sleep mutation into a JSON changeset (nocturne's `snapshot.py` is a 200-line reference impl); add a `rollback(changeset_id)` to `tools.py`. Even without a UI, a machine-readable changeset makes the honest-accounting story stronger.

### 3. Priority-derived staleness in the diagnostic report (Low)
**What**: `threshold_days = 3.5 * 2^priority` — high-priority nodes are flagged stale sooner, surfacing important-but-untouched memories for review.
**Why**: A cheap, interpretable "what deserves attention" signal orthogonal to Somnigraph's decay score; could seed REM gap-analysis targets.
**How**: One query in a sleep-adjacent report; not a scoring change.

---

## Not Useful For Us

### Deliberate no-semantic-search stance
Nocturne rejects vector search on principle; Somnigraph's whole thesis is learned hybrid retrieval. Not adoptable — it's the opposite bet.

### React/Vite dashboard, desktop-pet, companion-persona framing
Nocturne is a single-user companion-AI product with a heavy human-curation GUI. Somnigraph is a headless MCP research artifact; the UI surface and roleplay use cases are out of scope.

### Neo4j-era graph code / migration scripts
Historical; the live system is relational SQLite. No reusable graph-inference logic (there is none).

---

## Connections

- **Convergent with the Phase 18 write-path finding** (`docs/sessions/2026-06-28-phase18-source-sweep.md`, agentmemory/ByteRover/MemPalace): the LoCoMo leaders win on *write-path quality*, not retrieval. Nocturne is the extreme instance — it invests everything in human-curated write quality and versioning and *nothing* in retrieval sophistication. Independent corroboration that write-time discipline matters, though nocturne can't prove it (unbenchmarked).
- **`disclosure` trigger** rhymes with the proactive-injection design (`docs/proposals/proactive-injection.md`): both surface a memory based on a pre-authored condition rather than a live query. Nocturne's is human-written and passive (FTS-indexed), Somnigraph's would be floor-gated and automatic — same "condition precedes query" shape.
- **Versioning/supersession** overlaps memv's supersession pattern and MIRIX/Recall version chains (see `research/sources/`), but nocturne uniquely pairs it with a human accept/reject gate.
- **Glossary auto-linking** is a manual, curated cousin of Somnigraph's Hebbian co-retrieval PMI edges — same goal (lateral association channels), opposite mechanism (hand-bound vs. learned from co-retrieval).

---

## Summary Assessment

Nocturne's core contribution is an **architecture of deliberate human control**: stable node identity with append-only content versions, a before/after changeset with visual diff review and one-click rollback, and a URI-addressable concept tree the agent navigates. It is explicitly anti-automation — no vector search, no extraction, no dedup, no decay, no consolidation — and reframes "memory system" as "a curation dashboard the AI writes into and the human edits." For its actual target (a persistent-personality companion AI, per the README's roleplay/intimate examples), that trade is coherent. As a *retrieval* system it is weak: path-weighted BM25 with a priority tie-break and no semantic recall, unbenchmarked on any standard suite.

The single most valuable takeaway for Somnigraph is the **`disclosure` field** — capturing a natural-language recall condition at write time, indexed into retrieval. It's a Low-effort, honest-accounting-friendly probe against the exact multi-hop vocabulary gap Somnigraph has already diagnosed as its ceiling, and it comes from the query side rather than the (already-built) synthetic-bridge side. Secondarily, nocturne's changeset+rollback is a clean reference implementation for making Somnigraph's sleep merges reviewable and reversible.

What's overhyped: the carsteneu evidence file frames "Time-Travel" and "Trigger Rules" as headline capabilities via 9 upward corrections. Both are real in code, but "triggers" do **not** fire — `disclosure` is a searchable/human-facing string, and glossary "triggers" only append a hyperlink footnote when a keyword literally appears in text already being read. There is no inference, no automatic surfacing, and no benchmark. It is a well-engineered human-in-the-loop curation tool, not a competitive autonomous retriever — and it should not be read as comparable to Somnigraph's 85.1 LoCoMo QA on any axis, because it has no QA (or R@k) numbers at all.
