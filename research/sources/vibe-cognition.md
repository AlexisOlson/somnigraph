# Vibe Cognition Analysis

*Generated 2026-03-28 by Opus agent reading local clone*

---

## 1. Architecture Overview

**Repo**: https://github.com/Haagndaazer/vibe-cognition
**Stars**: 0 (as of 2026-03-28)
**License**: MIT
**Language**: Python 3.11+, ~3,335 lines across 17 source files + 4 test files
**Created**: 2026-03-25, 16 commits to date
**Description**: MCP server for Claude Code that captures project knowledge (decisions, failures, discoveries, patterns) as a typed graph. Fully local: ChromaDB for vector storage, sentence-transformers for embeddings, NetworkX for the in-memory graph, optional Ollama for LLM-driven edge curation.

**Distribution**: Claude Code plugin (via `claude plugin install`). The plugin bundles an MCP server, skills (prompt templates for Claude Code's `/skill` system), and hooks (session-start context injection, post-commit auto-capture). This is the first memory-focused project I've seen using Claude Code's plugin marketplace.

**Stack**:
- **Graph**: NetworkX MultiDiGraph, in-memory only, hydrated from JSONL on startup
- **Vectors**: ChromaDB (local persistent client, HNSW with cosine distance)
- **Embeddings**: nomic-embed-text-v1.5 via sentence-transformers (768d), or Ollama
- **LLM curation**: Optional Ollama (qwen3:8b) for semantic edge creation
- **Persistence**: Single JSONL file (`.cognition/journal.jsonl`) — append-only event log replayed on startup
- **Framework**: FastMCP for MCP server, Pydantic for models and settings

**Module organization**:
- `cognition/` — Graph storage, models, queries, curator, backfill, prime (context injection)
- `embeddings/` — ChromaDB wrapper, pluggable embedding backends
- `tools/` — MCP tool definitions (14 tools)
- `server.py` — FastMCP server with async lifespan and background model loading

**Key design decision**: The JSONL journal is meant to be **committed to Git and shared with the team**. ChromaDB is a regenerable cache (gitignored). This makes the knowledge graph collaborative — teammates pulling the repo get the full graph, and their local ChromaDB rebuilds automatically on next server start.

---

## 2. Memory Model

### Node Types

Eight typed nodes, split into two categories:

**Entities** (concise facts): `decision`, `fail`, `discovery`, `assumption`, `constraint`, `incident`, `pattern` — summaries capped at 250 chars, detail limited to 1-3 sentences.

**Episodes** (full narratives): `episode` — summaries of completed work (debugging sessions, feature implementations, issue resolutions). Detail is unbounded.

Each node has: `id` (12-char SHA-256 hash of type+summary+timestamp), `type`, `summary`, `detail`, `context` (list of file paths and topic terms), `references` (commit hashes, issue/PR numbers), `severity` (optional: critical/high/normal/low), `timestamp`, `author`.

### Edge Types

Seven edge types on a MultiDiGraph (same node pair can have multiple edge types):

| Edge | Meaning | Creation |
|------|---------|----------|
| `part_of` | Entity belongs to episode | Deterministic (reference matching) |
| `led_to` | Causal chain | Semantic (LLM or manual) |
| `resolved_by` | Problem fixed by solution | Semantic |
| `supersedes` | Newer replaces older | Semantic |
| `contradicts` | Logical conflict | Semantic |
| `relates_to` | Same topic, no causal link | Semantic (discouraged) |
| `duplicate_of` | Identical nodes, triggers merge | Curator only |

### No Decay

There is no temporal decay, no access tracking, no importance scoring, no consolidation. Nodes persist indefinitely. The only lifecycle event is `duplicate_of` triggering a merge (redirect edges, union context/references, keep higher severity, delete duplicate).

### No Feedback Loop

No retrieval feedback, no usage tracking, no reinforcement. Nodes are static once created.

---

## 3. Retrieval Pipeline

Retrieval is vector-only via ChromaDB's HNSW index:

1. **Embedding**: Query text embedded with nomic-embed-text-v1.5 using `search_query:` prefix (document embeddings use `search_document:` prefix — a detail many implementations miss)
2. **Vector search**: ChromaDB cosine similarity, optional filters by `entity_type` or `repo`
3. **Results**: Returned with similarity scores, no reranking, no fusion with other signals

The `cognition_search` tool wraps this with a limit cap of 50. There is no BM25, no FTS, no hybrid retrieval, no graph-aware scoring.

**Graph traversal** is separate from search: `cognition_get_chain` follows `led_to` edges (DFS with cycle detection, max depth 5), `cognition_get_neighbors` returns all connected nodes for a given node. These are structural queries, not retrieval — they require knowing a node ID first.

**Session priming**: The `prime.py` module generates a compact markdown summary injected at session start via a SessionStart hook. It surfaces: all constraints (sorted by severity), recent patterns (5), recent decisions (5), recent incidents (30 days). This is a fixed recipe, not adaptive.

---

## 4. Write Path

### Manual Recording

`cognition_record` creates a node, embeds it, indexes it in ChromaDB, runs deterministic edge matching, and optionally enqueues for LLM curation. The embedding and curation happen in the same call (or are skipped if the model isn't loaded yet — startup sync catches orphans later).

### Auto-Capture via Post-Commit Hook

A `PostToolUse` hook intercepts `git commit` Bash commands, runs `git log -1` to get the latest commit, and appends an `episode` node to the JSONL journal directly (bypassing the MCP server). Includes commit message, changed files as context, and `commit:<hash>` as reference. Idempotency check via journal grep.

### Deterministic Edge Creation

When a node is added, `create_deterministic_edges()` matches its references against a normalized reference index (case-insensitive, with short SHA prefix fallback). Creates `part_of` edges between entities and episodes that share references. Only entity-to-episode edges — two entities or two episodes with the same reference do not get linked. This is the only fully automatic edge creation.

### LLM Curation

Optional background worker (single-threaded queue) using Ollama. For each new node:
1. Find candidates via vector search (top 8, similarity > 0.3)
2. Build prompt with new node + candidates
3. Call Ollama (qwen3:8b, temperature 0.1) for JSON edge suggestions
4. Validate and create edges (or merge if `duplicate_of`)

The curator prompt is well-structured: explicit edge type definitions, direction guidance, instruction to prefer specific types over `relates_to`, prohibition on `part_of` suggestions (handled deterministically).

### Backfill Skill

The `/vibe-backfill` skill finds untracked commits and orchestrates subagents (max 5 concurrent) to read diffs and create episode + entity nodes. It's a prompt template, not code — the actual work is done by Claude Code reading git diffs and calling MCP tools.

---

## 5. Comparison to Somnigraph

### What they have that we don't

| Feature | Notes |
|---------|-------|
| **Git-committed knowledge graph** | JSONL journal designed for `git add` — team-shared project memory. Somnigraph's DB is personal, not collaborative. |
| **Claude Code plugin distribution** | Installable via `claude plugin install` with auto-setup. Somnigraph requires manual CLAUDE.md configuration. |
| **Post-commit auto-capture** | Hook creates episode nodes from every commit without user intervention. Somnigraph requires explicit `remember()` calls. |
| **Session-start context injection** | Constraints, patterns, decisions, incidents surfaced automatically at session start. Somnigraph has `startup_load` but it's a recall, not a curated summary. |
| **Deterministic reference matching** | Commit/issue/PR refs automatically create `part_of` edges between entities and episodes. No LLM needed. |
| **Subagent-based curation** | Edge-analyzer and cluster-analyzer subagents for batch semantic curation. |
| **Project-scoped memory** | Each project gets its own `.cognition/` directory. Somnigraph is a single global DB. |
| **Node type taxonomy** | 8 typed nodes (decision, fail, discovery, etc.) vs Somnigraph's 5 categories (episodic, procedural, reflection, entity, meta). The taxonomy is more detailed for development context. |

### What we have that they don't

| Feature | Notes |
|---------|-------|
| **Hybrid retrieval** | BM25 + vector + theme + graph with RRF fusion. They have vector-only (ChromaDB HNSW). |
| **Learned reranker** | 26-feature LightGBM trained on 1032 human-judged queries. They have no reranking at all. |
| **Temporal decay** | Biological decay with per-memory rates, tier-specific half-lives. They have no decay — everything persists indefinitely. |
| **Sleep-based consolidation** | LLM-driven enrichment, contradiction detection, entity extraction during offline passes. They have no consolidation. |
| **Retrieval feedback loop** | EWMA scoring with UCB exploration. They have no feedback mechanism. |
| **Hebbian co-retrieval** | Edge strengthening based on co-retrieval patterns. They have no dynamic edge weights. |
| **Graph-conditioned retrieval** | PPR traversal, graph features in the reranker. Their graph traversal is separate from search. |
| **FTS (BM25)** | Full-text search via SQLite FTS5. They have no keyword search at all. |
| **Enriched embeddings** | Theme-augmented, summary-enriched embedding text. They embed type+summary+detail directly. |
| **Proven benchmark results** | 85.1% on LoCoMo (beating Mem0, Mem0g, full-context). No benchmark evaluation. |

### Architectural Trade-offs

| Dimension | Vibe Cognition | Somnigraph |
|-----------|---------------|------------|
| Storage | JSONL + NetworkX (in-memory) | SQLite + sqlite-vec + FTS5 |
| Scale ceiling | Hundreds to low thousands of nodes (NetworkX in-memory, full graph on every startup) | Tested to ~13K memories with indexed queries |
| Embedding | Local (nomic-embed-text-v1.5, 768d) | Remote (OpenAI text-embedding-3-small, 1536d) |
| Retrieval | Vector-only (ChromaDB) | Hybrid (BM25 + vector + theme + graph + RRF + reranker) |
| Graph persistence | Append-only JSONL, replay on startup | SQLite edge table, indexed |
| Collaboration | Git-committed, team-shared | Personal (single user) |
| Scope | Per-project | Global (cross-project) |
| Decay | None | Biological (configurable per-memory) |
| Edge creation | Deterministic + LLM (Ollama) | LLM (sleep passes) + Hebbian |
| Distribution | Claude Code plugin marketplace | Manual (copy snippet to CLAUDE.md) |

---

## 6. Worth Adopting?

### Deterministic reference matching

**Verdict: Not applicable.** The pattern of linking nodes by shared commit/issue/PR references is clever for a development-context memory system. It's fast (no LLM needed), precise, and creates useful structure. However, Somnigraph's memories don't carry structured reference fields — they're free-form text with optional themes. The pattern solves a problem Somnigraph doesn't have (linking development artifacts) rather than one it does (relevance ranking).

### Session-start context summary

**Verdict: Already have it, different approach.** Somnigraph's `startup_load` recalls recent and important memories at session start. Vibe Cognition's approach is more structured — it generates a curated markdown document with sections for constraints, patterns, decisions, and incidents. The fixed recipe (always show all constraints, always show 5 recent decisions) is simpler but less adaptive than recall-based loading. Neither approach is clearly better; they optimize for different use cases (project context vs personal memory).

### Git-committed knowledge graph (team sharing)

**Verdict: Interesting but orthogonal.** JSONL-in-Git is a genuinely novel distribution mechanism for memory systems. The idea that project knowledge accumulates in the repo and is available to any team member's Claude session is compelling. For Somnigraph, this would require rethinking the entire data model — moving from a personal memory DB to a shared project knowledge base. These are fundamentally different products.

### Plugin distribution model

**Verdict: Worth watching.** Claude Code's plugin marketplace is new and the distribution mechanism (auto-setup of MCP server, hooks, and skills) is clean. If Somnigraph ever needed easier installation, this is the template.

---

## 7. Worth Watching

- **Adoption trajectory**: 0 stars and 16 commits as of 2026-03-28. The project is 3 days old. If it gains traction in the Claude Code plugin marketplace, the team-shared knowledge graph concept could prove out.
- **Scale behavior**: NetworkX in-memory with full JSONL replay on every startup will hit walls. Watch for whether they move to a proper database or if the per-project scope keeps sizes manageable.
- **Curation quality**: The Ollama curator (qwen3:8b) is the most interesting component for edge creation quality. If they publish evaluation of edge precision, that's relevant.
- **Backfill depth**: The claim of "1200-commit backfill capability" from Reddit refers to the `/vibe-backfill` skill, which orchestrates subagents to read git diffs and create nodes. This is a prompt-driven workflow, not a code feature — it's limited by Claude Code's context window and subagent management, not by the memory system itself.

---

## 8. Key Claims

| Claim | Evidence | Assessment |
|-------|----------|------------|
| "Temporal graph RAG memory system" (Reddit) | NetworkX MultiDiGraph with timestamps on nodes and edges. No temporal reasoning, no time-based retrieval, no decay. | **Aspirational**. The graph has timestamps but doesn't use them for retrieval or reasoning. "Temporal" overstates what the system does. |
| "1200-commit backfill capability" (Reddit) | `/vibe-backfill` skill that reads git history and creates nodes via subagents. | **Plausible** but misleading. The capability is Claude Code's, not the memory system's. The skill is a prompt template that orchestrates `git log` + `git diff` + MCP tool calls. Whether it handles 1200 commits depends on Claude Code's context and subagent limits. |
| "Fully local, no API keys" | Uses sentence-transformers (local) and optionally Ollama (local). No cloud API calls for core functionality. | **Verified**. The default configuration is genuinely local-only. |
| "Project knowledge graph" | JSONL-backed NetworkX graph with 8 node types, 7 edge types, deterministic + semantic edge creation. | **Verified**. The graph structure is real and well-implemented for what it is. |
| "Git-committed knowledge base" | JSONL journal designed for git commit, ChromaDB gitignored and regenerable. | **Verified**. The design is clean — regenerable cache separated from source-of-truth journal. |
| "Semantic search" | ChromaDB vector search with nomic-embed-text-v1.5. | **Verified** but minimal. Single-signal vector search with no reranking or fusion. |

---

## 9. Relevance to Somnigraph

**Rating: Low**

Vibe Cognition and Somnigraph solve different problems at different maturity levels. Vibe Cognition is a project-scoped development knowledge graph — it tracks what happened during development (decisions, failures, incidents) so future sessions have context. Somnigraph is a personal persistent memory system with biological decay, learned retrieval, and consolidation.

The retrieval pipeline (vector-only via ChromaDB, no reranking, no fusion, no feedback) is far simpler than Somnigraph's. There's nothing in the retrieval or scoring path to learn from.

The graph structure is reasonable but unambitious — typed nodes and edges with deterministic reference matching, but no graph-conditioned retrieval, no edge weights, no traversal-integrated scoring.

The two genuinely interesting ideas are:
1. **Git-committed team-shared memory** — a distribution/collaboration model Somnigraph hasn't explored (and may not need to, given its personal-memory scope)
2. **Deterministic edge creation via reference matching** — fast, precise, no-LLM linking of related artifacts (applicable to systems where memories carry structured references)

Neither is directly actionable for Somnigraph's current research priorities. The project is worth a quick check-in in 3-6 months if it gains traction or adds more sophisticated retrieval.
