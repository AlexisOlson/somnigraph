# opencode-mem - OpenCode plugin: SQLite + local-embedding vector store with LLM auto-capture and a reinforced user-profile/persona layer

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

A TypeScript/Bun plugin for the OpenCode coding agent (not Claude Code). Single-user, local-first, no server engine wrapped — it is its own store. Roughly ~55 source files under `src/services/`. The interesting parts are the write-path auto-capture and the user-profile learner; retrieval is deliberately plain.

### Storage & Schema
- **SQLite, sharded per scope.** `shard-manager.ts` splits data into per-project and per-user shard DB files. Container tags are `opencode_project_<sha16>` (from git remote URL or cwd) and `opencode_user_<sha16>` (from `git config user.email`). See `src/services/tags.ts`.
- **`memories` table (17 columns)**: `id, content, vector, tags_vector, container_tag, tags, type, created_at, updated_at, metadata (JSON), display_name, user_name, user_email, project_path, project_name, git_repo_url, is_pinned` (schema in `vector-search.ts` insert + `is_pinned`/`pinMemory`). `metadata` JSON carries `source`, `sessionID`, `promptId`, `captureTimestamp`, `reasoning`.
- **Vectors**: two per memory — a content vector and a separate `tags_vector` — stored as raw Float32 blobs. An in-memory **USearch** index is built on top (`vector-backends/usearch-backend.ts`), with an **ExactScan** brute-force fallback (`exact-scan-backend.ts`) selected by `backend-factory.ts`. SQLite is the source of truth; indexes are rebuilt from shard rows on demand (`rebuildFromShard`).
- **Embeddings**: local **Xenova/transformers.js**, 384-dim (default `Xenova/nomic-embed-text-v1`), `embedding.ts`. No remote embedding calls.

### Memory Types
- Memory `type` is a **free-form LLM-assigned string**, not a fixed taxonomy: the capture prompt suggests `feature, bug-fix, refactor, analysis, configuration, discussion, other`, plus a sentinel `skip` for non-technical turns (`auto-capture.ts` system prompt). No episodic/semantic/procedural distinction; no priority; no valid-from/until.
- A separate **user-profile** structure (`user-profile/types.ts`): `preferences[]` (category, description, confidence 0-1, evidence[]), `patterns[]`, `workflows[]`. This is the "persona" layer.

### Write Path
- **Auto-capture on session idle** (`auto-capture.ts`). It pulls the last uncaptured user prompt, fetches the assistant messages that followed, and builds a markdown context of `User Request / AI Response / Tools Used` (tool inputs truncated to 100 chars), prepended with the single latest project memory (first 500 chars) as "Previous Memory Context."
- An LLM (default `claude-haiku-4-5` via OpenCode's session API, or a manual OpenAI-compatible endpoint) returns structured `{summary, type, tags}`. `type="skip"` drops the prompt with no write. This is a **quality/salience gate** — non-technical turns are discarded. Retries with exponential backoff (3 attempts).
- Language detection: summary is written in the user's detected language (`language-detector.ts`).
- **Deduplication is NOT at write time.** `deduplication-service.ts` is a manually triggered (web endpoint) pass: exact `container_tag:content` string-match delete (keep newest), plus an O(n²) cosine near-duplicate scan that only *reports* groups (`nearDuplicateGroups`) above `deduplicationSimilarityThreshold` — it does not auto-merge them. Exact dedup also runs opportunistically during cleanup.
- **No enrichment beyond tags** — no graph edges, no entity extraction, no cross-memory linking at write time.

### Retrieval
- **Pure embedding retrieval, dual-channel.** `vector-search.ts::searchInShard`: query vector is searched against both the content index and the tags index (each `limit*4`), then blended: `similarity = contentSim*0.6 + tagsSim*0.4`.
- **Lexical tag boost**: query is tokenized and matched against each memory's tag string; the fraction of matched query words becomes `exactMatchBoost`, and `finalTagsSim = max(tagsSim, exactMatchBoost)`. This is the only lexical signal — a fuzzy `includes()` substring match on tags, not BM25/FTS.
- `searchAcrossShards` flattens per-shard results, filters by a similarity threshold, and slices top-k. `scope: "all-projects"` fans out over every project shard.
- **No FTS5, no BM25, no RRF fusion, no learned reranker, no graph expansion.** No feedback loop. `list` mode is chronological SQL (`created_at DESC`); there is also a `sessionID` metadata `LIKE` lookup. The evidence file's "at least two search approaches" = vector + chronological list.

### Consolidation / Processing
- **No sleep/consolidation of the memory store.** The only offline-ish process is the **user-profile learner** (`user-memory-learning.ts`): every `userProfileAnalysisInterval` (default 10) user prompts, an LLM re-analyzes recent prompts and produces/updates the persona. Merge logic (`user-profile-manager.ts::mergeProfileData`) reinforces matching preferences by **+0.1 confidence (capped at 1.0)**, unions evidence (cap 5), then **sorts by confidence and truncates** to a max count. This is a lightweight, periodic persona-consolidation — but it operates on raw prompts, not on stored memories, and never merges/edges the memory rows.

### Lifecycle Management
- **Retention-window cleanup, not decay** (`cleanup-service.ts`): once per 24h, delete memories whose `updated_at` is older than `autoCleanupRetentionDays`, **skipping `is_pinned` and prompt-linked memories**. No exponential decay curve, no reheat-on-access, no score-based eviction — a hard TTL with pin/link protection.
- `is_pinned` flag protects memories from cleanup and dedup. No versioning, no supersession, no `valid_until`.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| "Intelligent prompt-based memory extraction" | LLM summarize + `skip` gate in `auto-capture.ts` | Validated as a real salience gate; genuinely the strongest part |
| "Smart deduplication" | `deduplication-service.ts` | **Oversold** — exact dedup only; near-dup is a manual O(n²) *report*, no auto-merge, not at write time |
| "Automatic user profile learning" | `user-memory-learning.ts` reinforced merge | Validated; naive (no staleness decay, re-derives from prompts each interval) |
| Local vector DB, USearch + ExactScan fallback | `vector-backends/*` | Validated |
| 12+ local embedding models, 384-dim | `embedding.ts`, config | Validated (Xenova/transformers.js) |
| Multi-provider (OpenAI/Anthropic/copilot) | `ai/providers/*`, opencode session API | Validated |
| Any retrieval-quality / benchmark claim | none | **Absent** — no LoCoMo/PERMA, no R@k, nothing comparable to our 85.1 LoCoMo QA |

---

## Relevance to Somnigraph

### What opencode-mem does that Somnigraph doesn't
- **A persona / preference-state layer.** The `preferences/patterns/workflows` profile with confidence reinforcement is a first-class user-model that Somnigraph has no equivalent of — Somnigraph stores memories but never synthesizes a standing preference vector. This is exactly the capability **PERMA (STEWARDSHIP Priority 5)** measures, and the gap lives in the fact that no Somnigraph module owns preference-state maintenance.
- **A hard salience gate at write time** (`type="skip"`). Somnigraph's write path (`tools.py::impl_remember`) accepts whatever the caller decides to store; opencode-mem forces an LLM yes/no on every captured turn. This is write-path quality gating, a named Somnigraph gap.
- **Per-project sharding by git remote hash** — clean scoping primitive; Somnigraph is single-DB.

### What Somnigraph does better
- **Retrieval, decisively.** Somnigraph has hybrid BM25 (`fts.py`) + vector with RRF fusion, a 26-feature LightGBM reranker (`reranker.py`, NDCG=0.7958), and PPR graph expansion (`scoring.py`). opencode-mem has a fixed `0.6/0.4` content-vs-tags blend with a substring tag boost and zero learned ranking.
- **Graph + consolidation.** Somnigraph's sleep pipeline (`sleep_nrem.py`/`sleep_rem.py`) builds typed edges, merges, and archives. opencode-mem never links or merges memory rows.
- **Feedback loop.** Somnigraph has explicit utility ratings with r=0.70 GT correlation; opencode-mem has none.
- **Principled lifecycle.** Per-category exponential decay with reheat vs. a flat TTL delete.

---

## Worth Stealing (ranked)

### 1. Confidence-reinforced preference/persona layer (Medium)
**What**: A standing user-model of `preferences[]` (category, description, confidence, evidence) that is re-derived from recent prompts every N turns, where a re-observed preference gets `confidence = min(1, prev + 0.1)`, evidence is unioned, and the list is sorted-by-confidence and capped (`mergeProfileData`).
**Why**: This is the one mechanism here that maps directly to a Somnigraph gap and a live priority — **PERMA multi-domain preference-state (Priority 5, the headline metric)**. Somnigraph has no component that maintains a decaying/reinforcing preference state distinct from raw memories. Even as a baseline to beat, the shape is instructive: reinforcement-on-reobservation + confidence ranking + cap.
**How**: A new `reflection`/`meta`-category synthesis produced during REM (`sleep_rem.py`), or a dedicated `preferences` table, updated when the same preference recurs. Somnigraph could do better than opencode-mem by adding *staleness decay* to preference confidence (opencode-mem only ever increments) and by grounding evidence in memory IDs rather than raw prompt text.

### 2. Write-time LLM salience gate returning a `skip` sentinel (Low)
**What**: Every candidate memory passes an LLM classifier that can return `skip`, dropping non-technical/low-value turns before any write.
**Why**: Independent corroboration of the Phase 18 "write-path quality, not retrieval, is what leaders win on" finding. Cheap, and Somnigraph's write path currently trusts the caller.
**How**: An optional pre-write gate in `tools.py::impl_remember` (or the auto-capture equivalent) that classifies salience; already partially covered by the caller's own judgment, so value is marginal for the current single-agent usage.

---

## Not Useful For Us

### Dual content/tags vector blend + substring tag boost
Somnigraph's `embeddings.py` already enriches a single embedding with category/themes/summary, and `fts.py` gives real BM25 over themes and summary. A fixed 0.6/0.4 blend plus `includes()` tag matching is strictly weaker; redundant.

### USearch/ExactScan backend + per-project sharding
Engineering for a multi-project plugin distribution. Somnigraph is single-user single-DB with sqlite-vec; sharding solves a problem we don't have.

### Flat TTL cleanup
Somnigraph's per-category exponential decay with reheat is strictly more expressive than delete-if-older-than-N-days.

---

## Connections
- **Write-path-over-retrieval thesis**: converges with `agentmemory.md`, `byterover.md`, and the Phase 18 sweep — the systems that win do write-time curation, and opencode-mem's `skip` gate is another instance (though it never benchmarks to prove it).
- **Persona/preference layer**: closest kin among analyzed systems to anything PERMA-shaped; see `perma.md` for the benchmark this would target. Contrast with systems that store preferences as ordinary memories rather than a reinforced standing model.
- **Local-embedding, no-reranker vector plugin**: same class as `opencode-supermemory` (its stated inspiration) — thin retrieval, differentiator is the capture UX, not ranking quality.

---

## Summary Assessment

opencode-mem is a well-engineered *product* (web UI, sharding, multi-provider auth via OpenCode's session API, local embeddings) wrapped around a deliberately plain *memory science* core. Retrieval is a fixed content/tags cosine blend with a substring tag boost — no FTS, no fusion, no learned reranking, no graph, no feedback. There is no consolidation of the memory store and no decay, only a TTL cleanup with pin/link protection. Nothing here moves Somnigraph's retrieval-quality agenda (Priorities 2/4), and there are no benchmark numbers at all, so none of it is comparable to Somnigraph's 85.1 LoCoMo QA.

The single genuinely valuable idea is the **user-profile / persona layer**: a confidence-reinforced `preferences/patterns/workflows` model re-synthesized every N prompts. It is naive (increment-only confidence, re-derived from raw prompts, never merged into the graph), but it is the exact capability Somnigraph lacks and PERMA measures, so it is worth holding as a concrete baseline-shape for the eventual PERMA work — with the explicit improvements of staleness decay and memory-ID-grounded evidence.

The sharpest correction to the marketing and to the carsteneu audit: the README's "smart deduplication" is not smart and not automatic — it is an exact string-match delete plus a manual O(n²) near-duplicate *report* with no merge; the write-time "intelligence" is entirely the LLM summarizer, not any dedup or ranking logic. Verdict: **MAYBE** — revisit only when PERMA preference-tracking becomes active work.
