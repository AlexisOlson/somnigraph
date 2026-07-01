# Fullerenes — Zero-LLM Tree-sitter code-graph indexer with token-budgeted query output (not an agent memory system)

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

Fullerenes is a local-first CLI/MCP tool that parses a **source tree** into a SQLite graph so a coding agent can find symbols, callers, and blast radius without re-reading files. It is "memory" only in the sense of a persistent code index — it stores no conversational content, user facts, preferences, or episodic history. It is out of scope for Somnigraph's problem domain (persistent conversational memory for Claude Code), but included in the survey for completeness.

TypeScript monorepo, three packages: `core` (parser + SQLite + query), `cli` (commands + MCP server + agent-file generation), `daemon` (chokidar file watcher / auto-reindex).

### Storage & Schema
`packages/core/src/db/schema.ts`. SQLite (`better-sqlite3`, WAL) at `.fullerenes/graph.db`. Four tables:
- `nodes` — id, type (function/class/module/variable/interface/type), name, file_path, line_start/end, signature, docstring, language, hash, metadata, timestamps.
- `edges` — from_id, to_id, type (calls/imports/inherits/implements/contains/references), weight (default 1.0), file_path.
- `files` — path, content hash, language, node_count, last_indexed (drives incremental reindex).
- `meta` — key/value (last_indexed, stats).

No vector column, no FTS5, no embeddings table. Plain B-tree indexes on name/type/file and edge endpoints.

### Memory Types
Code-structure entities only (symbols and their relationships). No memory categories analogous to Somnigraph's episodic/semantic/procedural/reflection/meta. No priority, themes, salience, or validity interval.

### Write Path
`packages/core/src/indexer/index.ts` → `indexProject()`. Walk repo → per-file content hash → Tree-sitter parse (`parsers/{typescript,python,go,java,rust}.ts`) → emit nodes+edges → transactional `INSERT OR REPLACE`. Incremental mode skips files whose hash is unchanged; changed/deleted files have their nodes+edges **deleted and fully regenerated** (`deleteNodesByFile` / `deleteNodeEdgesByFile`). No extraction beyond AST walking, no dedup beyond edge-id uniqueness, no enrichment, no LLM, no quality/salience gating. Deterministic.

### Retrieval
`packages/core/src/graph/queries.ts`. Purely lexical + graph traversal, no ranking model:
- `searchNodes()` — SQL `LOWER(...) LIKE '%term%'` over name/signature/docstring/path, ordered by a `CASE`-based exact/prefix/path priority then in-degree.
- `queryWithBudget()` — the "natural-language retrieval": `extractSearchTerms()` (camelCase split, stopword filter, a **hardcoded synonym map** `TERM_EXPANSIONS`, e.g. `auth→[jwt,session,middleware]`), `inferQueryIntent()` (keyword match → entrypoints/impact/implementation/overview), `scoreNodeMatch()` (hand-tuned additive weights: exact-name +30/+18, prefix +12, in/out-degree bonus), then assembles intent-specific sections (CORE NODES / CALLERS / SIGNATURES / ENTRY POINTS / RELATED FILES) and truncates to a **char budget** (`maxTokens*4`, heuristic 1 token≈4 chars).
- Graph ops: `getSubgraph()` (BFS depth-limited, weight-ordered), `getCallers`/`getCallees`, `getEntryPoints` (modules with 0 incoming imports), `detectCircularDeps` (DFS cycle find), `predictImpact()` (reverse-edge BFS over calls/imports/inherits/implements/references → LOW/MEDIUM/HIGH risk from dependent counts).

No vector search, no BM25/FTS, no RRF fusion, no learned reranker, no feedback signal.

### Consolidation / Processing
None. No offline consolidation, summarization, or clustering. "Generated summaries" in the README refers to the generated `CLAUDE.md`/`AGENTS.md` context files, produced deterministically from graph stats — not LLM summarization. Watch-mode reindex is the only background process.

### Lifecycle Management
None in the memory sense. No decay, no versioning/history (a changed file's prior nodes are dropped, not superseded or dated), no archival, no contradiction/supersession tracking. The graph is a fresh mirror of current source.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Cuts agent token cost (64.2% on SWE-bench) | **One** instance (`sympy__sympy-20590`), same-model Codex A/B | Anecdotal; README itself says "not enough data yet." Not a memory-QA benchmark. |
| 94.4% local context compression | Query result (137 tok) vs concatenating touched files (2452 tok), 1tok≈4char heuristic | Tautological — a symbol digest is smaller than source. Not comparable to anything. |
| "Local-first, no external LLM" | Verified in code — zero API calls anywhere | True. Fully deterministic. |
| "Stronger natural-language retrieval" | `queryWithBudget` = LIKE + static synonym map + hand weights | Real but shallow; keyword matching, not semantic. |

None of these are end-to-end QA accuracy numbers; not comparable to Somnigraph's 85.1% LoCoMo QA.

---

## Relevance to Somnigraph

### What Fullerenes does that Somnigraph doesn't
Nothing in Somnigraph's domain. Its one adjacent capability — **write-time graph construction** (edges emitted synchronously during indexing) — is a shape Somnigraph lacks (Somnigraph builds edges offline during NREM sleep, `scripts/sleep_nrem.py`). But Fullerenes' edges are trivial to derive (AST call/import relations), so this is not evidence that real-time graph building is feasible for Somnigraph's semantic edges (supports/contradicts/evolves), which require LLM judgment.

### What Somnigraph does better
Everything relevant. Somnigraph has hybrid BM25+vector retrieval with RRF and a 26-feature learned reranker (`reranker.py`, `scoring.py`); Fullerenes has SQL `LIKE` + a static synonym dict. Somnigraph has embeddings (`embeddings.py`), a feedback loop with measured GT correlation, decay/lifecycle, and LLM-mediated consolidation — Fullerenes has none of these because they are irrelevant to code indexing.

---

## Worth Stealing (ranked)

**None.** This is a deterministic code-structure indexer; its mechanisms (Tree-sitter AST → SQLite, LIKE search, reverse-edge blast radius, static synonym expansion) don't transfer to conversational memory. The one superficially transferable idea — char-budget-capped, intent-templated query output — Somnigraph already covers better via recall token budgets and structured layered returns (detail/summary/gestalt). The static `TERM_EXPANSIONS` synonym map is exactly the hand-maintained keyword expansion that vector retrieval exists to replace; adopting it would be a regression.

---

## Not Useful For Us

### The entire system
Different problem domain: code-structure index vs conversational persistent memory. No overlap in memory unit, write path, retrieval signal, or evaluation.

### Static synonym expansion (`TERM_EXPANSIONS`)
Hardcoded keyword→keyword map. Somnigraph's enriched embeddings + BM25 fusion already handle vocabulary bridging in a learned, maintenance-free way.

---

## Connections

Closest to other **code-oriented / zero-LLM** tools in the carsteneu set (the ByteRover "BM25-only" and code-index entries). Reinforces the Phase 18 sweep observation that several r/AIMemory-listed "memory" projects are actually retrieval/indexing utilities, not agent-memory systems — the label "memory" spans code indexes, RAG wrappers, and true episodic stores. Fullerenes sits at the code-index end and should be tagged as out-of-scope in `docs/similar-systems.md` if referenced at all.

---

## Summary Assessment

Fullerenes is a competent, fully-deterministic Tree-sitter code-graph indexer with a nice ergonomic touch — token-budgeted, intent-shaped query output and a generated `CLAUDE.md` block — aimed at cutting an agent's repo-exploration token cost. It is not an agent memory system in Somnigraph's sense: it stores code symbols, not conversation, and has zero of the four things that define this corpus's interesting systems (semantic retrieval, write-path quality, lifecycle/decay, consolidation).

The single most important takeaway for Somnigraph is a negative one: it confirms that "AI memory" as a category label is noisy, and a code-level read is required to tell a real episodic-memory system from a code index — the light-touch triage note ("zero-LLM Tree-sitter blast-radius; out of scope") was correct.

Nothing is overhyped in a harmful way — the README is honest that benchmarks are single-instance smoke tests ("not enough data yet"). The evidence file's audit is accurate: it correctly flags the absence of vectors, lifecycle, extraction, and any LLM, and correctly frames the token-reduction numbers as one-instance and heuristic. Verdict: SKIP — nothing to adopt.

**Evidence-file cross-check:** Consistent. The carsteneu audit's checkmarks match the code (SQLite graph ✅, offline/no-LLM ✅, no semantic/vector search ❌, no lifecycle ❌, no extraction ❌). Sharpest correction, which the audit already half-makes: the headline token numbers are (a) a single SWE-bench instance and (b) a tautological "digest < source" char-count ratio using a 1-token≈4-char heuristic — neither is an end-to-end QA accuracy and neither is comparable to Somnigraph's 85.1% LoCoMo QA.
