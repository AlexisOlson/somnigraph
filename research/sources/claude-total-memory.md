# Claude Total Memory -- Source Analysis

*Phase 13, 2026-03-05. Analysis of vbcherepanov/claude-total-memory v4.0.*

## 1. Architecture Overview

**Language:** Python 3.10+, single-file server (`src/server.py`, ~2115 lines).

**Key dependencies:** `mcp[cli]>=1.0.0`, `chromadb>=0.4.0`, `sentence-transformers>=2.2.0`. ChromaDB and sentence-transformers are optional -- the server gracefully degrades if either is missing (sets `HAS_CHROMA`/`HAS_ST` flags at import time).

**Storage layout** (`~/.claude-memory/`):
- `memory.db` -- SQLite with WAL mode, FTS5 virtual tables
- `chroma/` -- ChromaDB persistent client (cosine similarity, collection `knowledge`)
- `raw/` -- JSONL append logs per session
- `transcripts/` -- session transcript archives
- `extract-queue/` -- pending/done transcript extraction files
- `backups/` -- JSON exports
- `extract-markers/` -- timestamp markers for live extraction dedup

**Embedding model:** `all-MiniLM-L6-v2` by default (env `EMBEDDING_MODEL`), loaded lazily on first embed call.

**Supporting scripts:**
- `src/extract_transcript.py` -- post-session transcript compression + auto-knowledge extraction (3 records: task, worklog, recovery context)
- `src/auto_extract_active.py` -- periodic (3-min launchd) live extraction from active sessions
- `src/dashboard.py` -- standalone stdlib HTTP server, serves SPA + REST API + SSE

**Hooks** (Claude Code settings integration):
- `SessionStart` -- prints hint to use `memory_recall` + `self_rules_context`
- `Stop` -- reminds to `memory_save` + `self_reflect`
- `PostToolUse:Bash` -- suggests `memory_save` after git commit, docker, migrations, package installs
- `PostToolUse:Write/Edit` -- suggests `memory_observe` after file changes

All hooks emit plain-text `MEMORY_HINT:` / `MEMORY_WARNING:` strings -- they are suggestions, not enforced.

## 2. Search Pipeline

Four-tier search in `Recall.search()` (lines 1109-1283), all results merged into a single `results` dict keyed by knowledge ID.

### Tier 1: FTS5 Keyword (BM25)

```python
fts_q = " OR ".join(Store._fts_escape(w) for w in re.split(r'\s+', query) if len(w) > 2)
```

- Queries `knowledge_fts` virtual table (indexed on `content`, `context`, `tags`)
- Joins back to `knowledge` table for status/project/type/branch filtering
- Fetches `limit * 3` candidates, ordered by `bm25(knowledge_fts)`
- Normalizes BM25: `(abs(raw_bm25) / max_bm25) * 2.0`, floored at 0.5
- Branch filter: `(k.branch=? OR k.branch='')` -- branch-specific + branch-agnostic

### Tier 2: Semantic (ChromaDB)

- Embeds query via `SentenceTransformer.encode()`
- Queries ChromaDB collection with `n_results=limit*3`, cosine distance
- Score: `max(0, 1.0 - distance)`
- If ID already in results from Tier 1, **adds** score (hybrid RRF-like fusion by addition)
- Otherwise fetches full record from SQLite and adds as new result

### Tier 3: Fuzzy (SequenceMatcher)

- Only runs if `len(results) < limit` (backfill strategy)
- Fetches `limit * 5` candidates ordered by `last_confirmed DESC`
- Compares `query.lower()` against first 200 chars of each candidate via `SequenceMatcher.ratio()`
- Threshold: ratio > 0.35
- Score: `ratio * 0.6` (deliberately low to avoid dominating)

### Tier 4: Graph Expansion (1-hop)

- Takes top 5 results by score
- For each, queries `relations` table for connected knowledge IDs (both directions)
- Score: `parent_score * 0.4`
- Only adds records not already in results

### Post-processing

All results get **decay scoring** applied:

```python
decay = Store._decay_factor(last_confirmed, DECAY_HALF_LIFE)  # e^(-days * ln2 / 90)
recall_boost = min(0.3, recall_count * 0.05)
item["score"] *= (decay + recall_boost)
```

Final ranking, top `limit` returned. All returned IDs get `bump_recall()` called (increments `recall_count`, updates `last_recalled` and `last_confirmed`).

**Key difference from claude-memory's RRF:** Scores are added across tiers rather than using reciprocal rank fusion. This means a record appearing in both FTS5 and semantic gets a raw sum, not a normalized RRF score.

## 3. Progressive Disclosure

Three detail levels controlled by `detail` parameter on `memory_recall`:

| Level | Token budget | What's included |
|-------|-------------|-----------------|
| `compact` | ~50 tokens/result | `id`, `type`, `title` (first 80 chars), `project`, `score`, `created_at` |
| `summary` | ~150 tokens/result | Above + `content` truncated to 150 chars, `tags`, `confidence`, `via`, `recall_count`, `decay` |
| `full` | Unlimited | Everything including full `content`, `context`, `branch` |

**How summaries are generated:** Pure truncation -- `content[:80]` for compact, `content[:150]` for summary. No LLM-generated summaries, no abstractive compression. Context is set to empty string for summary level.

Each result includes `_tokens` (estimated) and the response includes `total_tokens`. Token estimation is `len(text) // 4` (the ~4 chars/token heuristic).

## 4. Branch-Aware Context

**Detection:** `_detect_git_branch()` runs `git rev-parse --abbrev-ref HEAD` at server startup (2-second timeout, returns `""` on failure). Stored as global `BRANCH`.

**Storage:** `branch` column on `knowledge` and `sessions` tables (added via migration for older DBs). Every `memory_save` and `memory_observe` call stores the current branch. Hooks also detect branch via `git rev-parse` and include it in hints.

**Filtering:** On `memory_recall`, optional `branch` parameter adds `(k.branch=? OR k.branch='')` -- so branch-specific queries still include branch-agnostic knowledge.

**Limitation:** Branch is detected once at server startup. If the user switches branches mid-session, the server won't notice (MCP servers are long-running processes). The transcript extractor reads `gitBranch` from JSONL metadata per-message, which is more accurate.

## 5. Tool Surface

### Memory CRUD (7 tools)
| Tool | Purpose |
|------|---------|
| `memory_save` | Store knowledge (5 types: decision, fact, solution, lesson, convention) |
| `memory_recall` | 4-tier search with progressive disclosure |
| `memory_update` | Find-and-supersede existing knowledge |
| `memory_delete` | Soft-delete a record |
| `memory_history` | Walk superseded_by chain for version history |
| `memory_relate` | Create typed relation (causal, solution, context, related, contradicts) |
| `memory_search_by_tag` | Browse by tag (partial match) |

### Session & Lifecycle (5 tools)
| Tool | Purpose |
|------|---------|
| `memory_timeline` | Browse session history (by number, date range, search query) |
| `memory_stats` | Health metrics, storage sizes, config, self-improvement stats |
| `memory_consolidate` | Find and merge similar records (Jaccard > threshold) |
| `memory_export` | JSON backup to `~/.claude-memory/backups/` |
| `memory_forget` | Apply retention policy (archive stale, purge old) |

### Self-Improvement (6 tools)
| Tool | Purpose |
|------|---------|
| `self_error_log` | Log structured errors with category/severity |
| `self_insight` | ExpeL-style insight management (add/upvote/downvote/edit/list/promote) |
| `self_rules` | Manage behavioral rules (list/fire/rate/suspend/activate/retire/add_manual) |
| `self_patterns` | Analyze error patterns and improvement trends |
| `self_reflect` | Save verbal self-reflections (Reflexion pattern) |
| `self_rules_context` | Load active rules for current session context |

### Observations (2 tools)
| Tool | Purpose |
|------|---------|
| `memory_observe` | Lightweight file-change tracking (no dedup, no embeddings) |
| `memory_extract_session` | Manage pending transcript extractions (list/get/complete) |

**Total: 20 tools.**

## 6. Spaced Repetition & Decay

### Decay

Exponential decay function:

```python
decay = max(0.01, exp(-days * ln(2) / half_life))
```

- `DECAY_HALF_LIFE` defaults to 90 days (env configurable)
- Applied as a multiplier to search scores
- `last_confirmed` timestamp is the reference (updated on dedup hit, recall, or explicit update)
- If `last_confirmed` is null, returns 0.5 (neutral)

### Spaced Repetition

Not true spaced repetition (no scheduled review intervals). Instead:

- **Recall boost:** `min(0.3, recall_count * 0.05)` added to decay factor
- Every time a record is returned in search results, `bump_recall()` increments `recall_count`, updates `last_recalled` and `last_confirmed`
- This means frequently-accessed knowledge stays fresh and ranks higher -- a passive reinforcement loop

### Retention Zones (Forgetting)

Three-stage lifecycle with configurable thresholds:

| Stage | Criteria | Default |
|-------|----------|---------|
| `active` -> `archived` | `last_confirmed` > 180d AND `recall_count = 0` AND `confidence < 0.8` | `ARCHIVE_AFTER_DAYS=180` |
| `archived` -> `purged` | `last_confirmed` > 365d | `PURGE_AFTER_DAYS=365` |

Triggered manually via `memory_forget` tool (supports `dry_run=true`). Also removes from ChromaDB.

### Observation Cleanup

Observations auto-delete after 30 days (`OBSERVATION_RETENTION_DAYS`), run on server startup.

### Self-Improvement Lifecycle

- Insights with importance <= 0 are auto-archived
- Rules with success_rate < 0.2 after 10+ fires are auto-suspended
- Stale rules (not fired in 60+ days) are flagged in `self_patterns` reports

## 7. Token Management

**Estimation:** `len(text) // 4` -- the simplest possible heuristic. Each search result includes `_tokens` field, and the response includes `total_tokens` sum.

**Budget awareness:** None. There is no token budget cap or automatic detail-level downgrade. The caller is responsible for choosing `compact`/`summary`/`full` and interpreting `total_tokens`.

**Comparison to claude-memory:** No token-counting library (like tiktoken), no automatic truncation to fit a budget, no shadow-load concept. The token count is informational only.

## 8. Web Dashboard

Standalone HTTP server (`src/dashboard.py`, ~2100 lines) using only Python stdlib. Port 37737 by default.

**Architecture:** Single-file SPA -- all HTML/CSS/JS is embedded as a string in the Python file. Threaded HTTP server with read-only SQLite connection.

**API endpoints:**
- `/api/stats` -- memory health, counts by type/project, storage sizes
- `/api/knowledge` -- paginated list with search/type/project/branch filters
- `/api/knowledge/<id>` -- detail view with relations and version history
- `/api/sessions` -- session list
- `/api/graph` -- nodes (top 200 by recall_count) and edges for visualization
- `/api/errors`, `/api/insights`, `/api/rules` -- self-improvement data
- `/api/self-improvement` -- aggregated pipeline stats
- `/api/observations` -- observation list
- `/api/branches` -- distinct branch names
- `/api/events` -- SSE (Server-Sent Events) live feed

**Frontend tabs:**
1. **Dashboard** -- stats cards, knowledge by type/project charts
2. **Knowledge** -- searchable/filterable table with detail modal
3. **Sessions** -- session timeline browser
4. **Errors** -- error log with category/project filters
5. **Insights** -- insight list with promotion eligibility
6. **Rules (SOUL)** -- behavioral rules with success rates
7. **Graph** -- force-directed knowledge graph (Canvas-based, custom implementation)
8. **Self-Improvement** -- pipeline overview (errors -> insights -> rules flow)
9. **Live Feed** -- SSE-powered real-time activity stream

## 9. Comparison to claude-memory

| Dimension | claude-total-memory | claude-memory |
|-----------|-------------------|---------------|
| **Search fusion** | Score addition across tiers | Reciprocal Rank Fusion (RRF) with configurable weights |
| **Embedding** | sentence-transformers (all-MiniLM-L6-v2, local) | sentence-transformers (same default, local) |
| **Vector store** | ChromaDB | ChromaDB |
| **FTS** | SQLite FTS5 with BM25 | SQLite FTS5 with BM25 |
| **Graph** | `relations` table with typed edges (5 types), 1-hop expansion in search | Edge schema with novelty operator, richer edge types |
| **Progressive disclosure** | 3 levels via truncation (80/150/full chars) | Shadow-load with LLM-generated summaries |
| **Decay** | Single exponential decay (half-life 90d) + recall boost | Per-category decay with configurable floors |
| **Forgetting** | 3-stage retention zones (active/archived/purged) with manual trigger | Per-category decay with floor, continuous |
| **Sleep consolidation** | None -- consolidation is manual (`memory_consolidate`) | NREM+REM cycle (dedup then synthesize) |
| **Recall feedback** | None -- all returned results implicitly "bump recall" | Explicit `recall_feedback` tool for relevance signal |
| **Token estimation** | `len(text) // 4` | tiktoken-based counting |
| **Token budgeting** | Informational only (no cap) | Budget-aware with automatic truncation |
| **Privacy** | Regex-based auto-redaction + `<private>` tags | Not present |
| **Self-improvement** | Full pipeline: errors -> insights -> rules (ExpeL + Reflexion patterns) | Not present |
| **Web dashboard** | Full SPA with graph viz, SSE live feed | Not present |
| **Observations** | Lightweight auto-capture tool with 30-day TTL | Not present |
| **Branch awareness** | branch column + filter | Not present |
| **Session extraction** | Auto-extract from Claude Code transcripts (live + post-session) | Not present |
| **Hooks** | 5 Claude Code hooks for lifecycle hints | Not present |
| **Sync layer** | None (SQLite only) | JSON sync layer for external integration |
| **Startup** | Branch detection only | `startup_load` with token counting and context assembly |
| **Knowledge types** | 5 fixed types (decision, fact, solution, lesson, convention) | Flexible categories |
| **Tool count** | 20 | ~12 |

### Key architectural differences

1. **claude-total-memory is monolithic** -- one 2100-line file for the server, one for the dashboard. Everything in one process. claude-memory separates concerns more.

2. **Self-improvement pipeline is unique** -- the errors -> insights -> rules -> SOUL system is a genuine differentiator. It's an ExpeL-style learning loop with voting, promotion thresholds, and auto-suspension of ineffective rules.

3. **Session extraction is a real workflow** -- hooks + transcript parsing + auto-save of 3 knowledge records per session is a thoughtful lifecycle story. claude-memory's sleep consolidation serves a similar purpose but is LLM-driven.

4. **Search fusion is simpler** -- raw score addition vs RRF. No weighting parameters. Less tunable but simpler to reason about.

5. **No LLM in the loop** -- everything is algorithmic. Summaries are truncations, consolidation keeps the longest record, no LLM calls. This means zero additional API cost but also less intelligence in the pipeline.

## 10. Worth Stealing

**Ranked by implementation effort vs value:**

1. **Privacy stripping (high value, low effort)** -- Regex-based auto-redaction of API keys, JWTs, emails, credit cards before storage. Plus `<private>` tag support. Simple to add, real safety benefit. Our system stores whatever the agent sends -- this is a gap.

2. **Observations with TTL (medium value, low effort)** -- Lightweight ephemeral captures (no dedup, no embeddings) with automatic 30-day cleanup. Good for "what happened during this session" without polluting long-term memory. Could supplement our existing system cheaply.

3. **Self-improvement pipeline concept (high value, high effort)** -- The errors -> insights -> rules -> SOUL flow is compelling. Worth studying for a lighter-weight version. Key mechanism: pattern detection (3+ errors of same category triggers suggestion), voting (importance/confidence), promotion thresholds, auto-suspension of bad rules. We could potentially model this as a special memory category with automated consolidation rules.

4. **Hooks as soft suggestions (medium value, low effort)** -- Their hooks just emit `MEMORY_HINT:` text, not enforce behavior. The session-start hook reminding to load rules and the stop hook reminding to reflect are simple but effective nudges. We already have slash commands that serve a similar purpose, but hooks fire automatically.

5. **Branch-aware filtering (low-medium value, low effort)** -- Simple addition: store branch on save, filter with `(branch=? OR branch='')` on recall. Useful for feature-branch-heavy workflows. We don't currently track this.

6. **Version history via supersession chains (medium value, medium effort)** -- The `superseded_by` pointer + `memory_history` tool creates a git-like version history for knowledge. We handle updates differently (edge-based), but explicit version chains are nice for "what did we used to think about X?"

## 11. Not Worth It

1. **Score addition instead of RRF** -- Their hybrid search fusion is simpler but less principled. RRF is better calibrated across different score distributions. No reason to switch.

2. **Truncation-based progressive disclosure** -- Cutting content at 80/150 chars loses semantic meaning. Our shadow-load with LLM-generated summaries produces much better compact representations. Their approach saves compute but sacrifices quality.

3. **`len(text) // 4` token estimation** -- Too crude for budget management. Tiktoken or similar is worth the dependency.

4. **Manual consolidation trigger** -- Requiring the agent to call `memory_consolidate` means it usually won't happen. Our automatic sleep consolidation is better (though we should verify it actually fires).

5. **No recall feedback** -- Every returned result implicitly gets a "useful" bump. There's no way to say "that result was irrelevant." Our explicit `recall_feedback` provides cleaner signal.

6. **Monolithic single-file architecture** -- 2100+ lines in one file is manageable today but won't scale. Not a pattern to emulate.

7. **20 tools** -- Tool surface area is large. Many tools could be consolidated (e.g., `self_insight` with 6 sub-actions could be 6 tools, or the 6 sub-actions could be parameters on fewer tools). The agent has to reason about when to use each. Our smaller surface with richer individual tools is better for agent reliability.

8. **Dashboard** -- Nice for debugging/exploration but not a core memory capability. Building a web dashboard is significant effort for marginal memory-system benefit. If we want visualization, Obsidian dataview queries or a simple script that dumps stats is sufficient.
