# Continuity v2 - Raw-transcript indexer that turns the Claude Code JSONL session record into a searchable episodic backstop

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

Continuity v2 is not a curated memory store. It is a read-only **index and retrieval layer over raw conversation transcripts**: Claude Code's on-disk JSONL session files (`~/.claude/projects/<project>/<session>.jsonl`) plus an optional claude.ai data export (`conversations.json`). The core thesis (README): the episodic record already exists on disk, append-only and free; nobody had wired it into recall. So v2 walks it, mirrors it into SQLite+FTS5+embeddings, and exposes MCP search tools. There is no extraction, no summarization, no salience gating - the memory unit is a verbatim conversation turn.

### Storage & Schema
Single SQLite DB (`data/continuity.db`, ~50MB, gitignored). Three real tables (`index.py`):
- `sessions(id, project, ai_title, cwd, started_at, ended_at, turn_count, file_path, file_mtime, indexed_at, source)` - one row per session; `source` is `code` or `chat`.
- `turns(id, session_id, turn_idx, ts, role, text)` - one row per user/assistant turn. `text` is flattened from content blocks; tool_use blocks become `[tool:name] desc`, tool_results truncated to 500 chars.
- `turns_fts` - FTS5 external-content mirror of `turns.text`, kept in sync by insert/delete triggers.
- `turn_vecs` - sqlite-vec `vec0` virtual table, `embedding float[384]` (`embed.py`).
- `edges(src_turn_id, dst_turn_id, edge_type, weight)` - TEMPORAL and SIMILAR_TO edges.

No priority, no category taxonomy, no themes, no valid_from/valid_until, no decay_rate field. Incremental indexing is by file mtime (`index_file` skips when `sessions.file_mtime` matches disk).

### Memory Types
None in the semantic sense. The only distinction is `source` (`code` vs `chat`) and `role` (user/assistant, plus a synthetic `summary` pseudo-turn at `turn_idx=-1` for claude.ai conversations that carry an export summary, `chat_index.py`). No episodic/semantic/procedural split, no importance tiers.

### Write Path
Effectively a parser, not a write path. `index.py` / `reindex()` (in `mcp_server.py`, duplicated inline to dodge cross-process SQLite lock contention) walk JSONL, flatten content blocks, and bulk-insert turns. **No LLM extraction, no deduplication, no enrichment, no quality/salience gating.** Every turn >= a trivial length is stored. Embedding (`embed.py`) skips turns shorter than 30 chars and anything starting with `[tool:`/`[result]`, batches through `all-MiniLM-L6-v2` (384d, L2-normalized), idempotent by `turn_id`.

### Retrieval
Five modes, all fixed-rule, no learned component:
1. **FTS5 keyword** (`search_sessions`) - BM25 `ORDER BY rank`, optional project/source filters, snippet highlighting. Hyphen/number tokens must be double-quoted (raw FTS5 exposure).
2. **Semantic + fixed hybrid rerank** (`find_similar`) - ANN over `turn_vecs` (over-fetches `limit*3`), then re-ranks by a hard-coded formula: `0.7*cos_sim + 0.2*recency + 0.1*complexity`, where recency is linear decay to 0 at 365 days and complexity is `min(1, turn_count/50)` (rewards long sessions regardless of relevance). Weights are module constants (`_W_SEMANTIC` etc.), never tuned or learned.
3. **BFS thread recall** (`thread_recall`) - seeds from top-N FTS5 matches, then `_bfs_expand` walks the edge graph up to `max_hops=8` / `max_turns=60`, returning the surrounding **narrative thread** grouped by session, seeds marked `[MATCH]`. This is the repo's signature move: return contiguous context (what led up to and followed the hit), not isolated rows.
4. **Session replay** (`recall_session`) - full or index-sliced dump of one session.
5. **Chronological browse** (`recent_sessions`).

There is **no fusion across channels** (FTS5 and vector are separate tools; the user/agent picks one). No cross-encoder, no learned reranker.

### Consolidation / Processing
None. No sleep cycle, no merge/archive, no contradiction detection, no gap analysis, no question generation. `wire_edges.py` (TEMPORAL = turn i -> i+1, purely derived) and `wire_similar.py` (SIMILAR_TO = cosine >= 0.85 among K=5 NN, skipping temporal-adjacent pairs) are one-shot batch graph builders, not consolidation. `drift_check.py` is an operational staleness detector (mtime diff vs disk) that says whether a reindex is warranted - housekeeping, not memory processing.

### Lifecycle Management
None at storage level. No archival, no deletion, no forget/delete MCP tool, no versioning, no supersession. The only "decay" is the **query-time** recency term inside `find_similar` - it reweights results, it never expires or removes a turn. Turns live forever; the index only grows.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| The JSONL session record is a zero-cost episodic store | True - `index.py` reads existing on-disk files, no capture cost | Validated (genuine, and the repo's real insight) |
| Hybrid score `0.7 sem + 0.2 rec + 0.1 cplx` | Present verbatim in `find_similar` | Validated as implemented; weights are asserted, never tuned/ablated |
| BFS "narrative thread, not just rows" | `thread_recall` + `_bfs_expand` real | Validated - but defaults to TEMPORAL edges only (see below) |
| Semantic SIMILAR_TO edges bridge concept-not-word gaps | `wire_similar.py` builds them at cos>=0.85 | Partially unrealized - built but `thread_recall` default `edge_types=("TEMPORAL",)` never traverses them |
| Dual-source (Claude Code + claude.ai) | `chat_index.py` real | Validated |
| No published benchmarks | Confirmed - no eval harness in repo | Honest; evidence file agrees |

---

## Relevance to Somnigraph

### What Continuity v2 does that Somnigraph doesn't
- **Indexes the raw session transcript as a fallback episodic store.** Somnigraph can only recall what it *chose* to `remember()`; a conversation it never wrote a memory about is simply gone. Continuity v2 recalls anything ever said, because it indexes the JSONL Claude Code already writes. This is a different philosophy - recall-what-happened vs. store-what-matters - and it is the one place Somnigraph has a genuine structural gap (no module owns "what did we actually say three weeks ago" if no memory was captured). Nearest Somnigraph touchpoint: `tools.py`/`events.py` capture only curated units; there is no transcript backstop.
- **Narrative-thread output** (`thread_recall`): returns contiguous surrounding turns. Somnigraph's `scoring.py` PPR expansion returns *related discrete memories*, not a temporally contiguous replay.
- **Dual-source ingestion** (claude.ai export alongside Claude Code) - Somnigraph is single-source.

### What Somnigraph does better
Almost everything on the memory-science axis. Somnigraph has a 26-feature learned LightGBM reranker (`reranker.py`) vs. Continuity's hand-set 0.7/0.2/0.1 weights; RRF fusion of BM25+vector (`scoring.py`) vs. no fusion at all; typed semantic edges (supports/contradicts/evolves) detected during LLM-mediated sleep (`sleep_nrem.py`) vs. only TEMPORAL adjacency + cosine SIMILAR_TO; PPR graph-conditioned retrieval vs. plain BFS; an explicit feedback loop with r=0.70 GT correlation vs. no feedback; per-category decay with reheat (`db.py`) vs. query-time recency only; write-path curation vs. none; 85.1% LoCoMo QA vs. no benchmark. Continuity v2 is a search tool; Somnigraph is a memory system.

---

## Worth Stealing (ranked)

### 1. Raw-transcript episodic backstop (Medium)
**What**: A cheap secondary index over the Claude Code JSONL record as a fallback for recalls that curated memory misses - the "I told you this two weeks ago, and you never made a memory of it" case.
**Why**: It targets Somnigraph's one honest structural blind spot: recall is bounded by what `remember()` captured. A verbatim-transcript FTS index is a floor under that ceiling, and it's near-free because the JSONL already exists.
**How**: A standalone FTS5 table over `~/.claude/projects/*.jsonl` (Continuity's `index.py` is ~150 lines), exposed as a distinct low-priority `search_transcript` path that the agent consults only when curated recall returns nothing relevant. Keep it *out* of the reranker/graph - it's a different store with different trust, and mixing verbatim turns into the curated graph would pollute PPR and feedback signal. This is the buried idea worth not losing; adoption is optional and arguably out of Somnigraph's curated-memory scope.

### 2. Narrative-thread expansion for episodic hits (Low)
**What**: When a recall hit is an episodic memory, optionally return a short window of temporally adjacent context rather than the isolated unit.
**Why**: Episodic "what happened around then" queries benefit from contiguity; PPR returns semantically-related, not temporally-contiguous, neighbors.
**How**: Marginal - Somnigraph units aren't sequential turns, so this only applies if a transcript backstop (#1) exists. Note-only unless #1 lands.

---

## Not Useful For Us

### Fixed 0.7/0.2/0.1 hybrid weights and turn-count "complexity" bonus
Somnigraph's learned reranker already subsumes hand-tuned linear weighting and does it better; the complexity term (reward long sessions) is a crude length prior that would be an anti-feature. Nothing to take.

### v1 compaction hooks (SSE proxy, precompact/stop checkpoints)
`hooks/` are session-continuity plumbing (token-pressure bells, in-session checkpoint save/inject) inherited from continuity v1. Orthogonal to persistent memory; not relevant.

### SIMILAR_TO edge graph
Built but effectively dead - `thread_recall` traverses TEMPORAL only by default. Somnigraph's Hebbian PMI co-retrieval edges are a strictly stronger version of "link semantically related memories."

---

## Connections

Same write-path-vs-retrieval theme as the Phase 18 sweep (see `byterover.md`, `agentmemory.md`): those leaders win on write-time quality. Continuity v2 is the inverse extreme - **zero** write-path curation, pure retrieval over verbatim turns - which is exactly why it can't compete on QA but *can* offer total recall of raw history. Its "index what already exists on disk" stance echoes the general observation that the Claude Code JSONL is an underused substrate. Contrast with Somnigraph's curated units and with any system doing LLM extraction at write time (e.g. the MemPalace/agentmemory write-time grounding line).

---

## Summary Assessment

Continuity v2's core contribution is a philosophy, not a mechanism: the Claude Code session transcript is an episodic memory store that already exists, append-only and free, and the only missing piece was an index. The code is clean, honest, and does exactly that - FTS5 + a 384d embedding index + a TEMPORAL/SIMILAR_TO edge graph + a BFS "return the narrative thread" retrieval, over both Claude Code sessions and claude.ai exports. It carries no pretense of being a curated memory system: no extraction, no dedup, no salience gating, no consolidation, no learned ranking, no feedback, no benchmarks.

The single thing worth taking is the **raw-transcript backstop** - a floor under Somnigraph's recall that catches what curation never captured. It is buried, additive, and genuinely different from anything Somnigraph does, but it is also arguably out of scope for a system whose entire thesis is *selective* curated memory; the risk is polluting the graph/feedback signal if merged carelessly, so it should stay a separate low-trust store consulted only on curated-recall misses.

Everything else is either strictly weaker than what Somnigraph already has (fixed-weight hybrid scoring vs. learned reranker; BFS vs. PPR; adjacency+cosine edges vs. typed sleep-detected edges) or off-axis (compaction hooks). The evidence file is accurate and admirably honest (it states the absence of benchmarks and entity extraction plainly); the sharpest correction to add is that its "semantic SIMILAR_TO graph traversal" is realized only as a built-but-unused edge set - `thread_recall` walks TEMPORAL edges by default, so narrative recall is temporal-adjacency reconstruction, not semantic graph walk. Verdict: MAYBE - one revisit-if angle (episodic transcript fallback), no core mechanism to adopt.
