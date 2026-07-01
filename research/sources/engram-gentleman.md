# engram (Gentleman-Programming) - Go/SQLite FTS5-only agent memory with save-time conflict surfacing and cloud sync

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

Single Go binary (147 `.go` files), SQLite + FTS5, TUI dashboard, MCP server + HTTP API, plus an optional cloud-sync tier. No Node/Python/embeddings/Docker required for the core. This is a DIFFERENT project from Harshitk-cp/engram (`engram.md`) — do not conflate.

### Storage & Schema
Core tables (`internal/store/store.go` ~L695-935):
- **`observations`**: `id, sync_id, session_id, type, title, content, tool_name, project, scope, topic_key, normalized_hash, revision_count, duplicate_count, last_seen_at, pinned, created_at, updated_at, deleted_at`. Later additive columns: `review_after, expires_at, embedding BLOB, embedding_model, embedding_created_at`.
- **`observations_fts`** and **`prompts_fts`**: two FTS5 external-content virtual tables (title/content/tool_name/type/project/topic_key; and prompt content).
- **`user_prompts`** / `prompt_tombstones`: raw user-prompt capture stream.
- **`memory_relations`** (~L903): `source_id, target_id, relation, reason, evidence, confidence, judgment_status, marked_by_{actor,kind,model}, superseded_by_relation_id`. Relation verbs: `related, compatible, scoped, conflicts_with, supersedes, not_conflict`. No UNIQUE on (source,target) — deliberately allows multi-actor disagreement.
- **Sync machinery**: `sync_chunks, sync_state, sync_mutations` (a per-target append-only mutation log), `cloud_upgrade_state`. Substantial part of the codebase (`internal/cloud/*`, `internal/sync/*`, chunkcodec, autosync) is devoted to multi-device replication.

Note: `embedding BLOB` and `review_after`/`expires_at` columns are **reserved but inert** — no code populates embedding for search, and `expires_at` is "intentionally NULL for all types in Phase 1" (store.go ~L2368). Confirms evidence: semantic=false, decay=false.

### Memory Types
`type` is a free-text field (architecture/bug/decision/pattern/config/discovery/learning/session_summary/manual...), not an enforced enum. `scope` is project/global. No episodic/semantic/procedural taxonomy, no priority, no themes array, no layered memory (evidence: layeredMemory=false — confirmed).

### Write Path
`mem_save` → `SaveObservation` (store.go ~L2280-2367). Three-tier identity handling, all **rule-based, no LLM at write time**:
1. **topic_key upsert**: if a `topic_key` matches an existing row (same project/scope), UPDATE in place and `revision_count += 1` — stable-key semantic versioning.
2. **Exact dedup window**: else if `normalized_hash` matches within `DedupeWindow = 15 * time.Minute` (same project/scope/type/title), `duplicate_count += 1`, bump `last_seen_at`.
3. Else INSERT new.

`topic_key` derivation (`SuggestTopicKey`/`inferTopicFamily`, store.go ~L6391-6467): keyword heuristics map type+title+content to a family (`architecture/*`, `bug/*`, `decision/*`, `pattern/*`, `config/*`, `discovery/*`, `learning/*`) then a normalized slug → e.g. `architecture/hexagonal-boundary`. Extraction is **manual** — `mem_capture_passive` still requires explicit invocation (evidence autoExtract=false — confirmed). No LLM refinement, no salience/quality gating, no narrative generation.

### Retrieval
Pure FTS5. `Store.Search` (store.go ~L3102):
- If query contains `/`, a **topic_key direct lookup** shortcut (`ORDER BY updated_at DESC`, rank sentinel -1000 so it sorts first).
- Otherwise FTS5 `MATCH` with `match_mode` all(AND)/any(OR), filtered by type/project/scope, `ORDER BY fts.rank` (SQLite BM25) `LIMIT`.
- No vector channel, no RRF fusion, no learned reranker, no graph expansion. "searchModes: 4" in the evidence file = {FTS full-text, recent-session context, chronological timeline, direct ID fetch} — these are distinct query *endpoints*, not fused retrieval channels.

### Consolidation / Processing
No offline/sleep/batch consolidation cycle. The nearest analogue is a **save-time conflict-surfacing loop** (`FindCandidates`, relations.go ~L324): on save, run FTS5 on the new title, apply a **BM25 floor** (default -2.0, closer-to-0 = better), take top-N as candidate conflicts, write `pending` relation rows. Then `mem_judge`/`JudgeBySemantic` (relations.go ~L747) calls an **LLM (Claude or OpenCode backend)** to classify each candidate pair into a relation verb and write `confidence`/`evidence`. `mem_compare` adjudicates two specific memories. This is on-demand/at-save, LLM-adjudicated, not a scheduled batch pass.

### Lifecycle Management
Soft-delete via `deleted_at` (`--hard` for physical). `supersedes` relations + `superseded_by_relation_id` chain for versioning. `pinned` flag. **No decay** (`review_after` written for some types but never acted on; `expires_at` always NULL). `mem_timeline` gives chronological "time travel" over an observation's revision history.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Single Go binary, SQLite+FTS5, no external deps | `go.mod`, store.go schema | Validated |
| FTS5 full-text search | `Search`, `observations_fts` | Validated |
| Conflict detection + supersede | `FindCandidates` + `memory_relations` + LLM `JudgeBySemantic` | Validated (LLM-adjudicated, not automatic) |
| Dedup (15-min hash window) | `DedupeWindow`, `normalized_hash`, `duplicate_count` | Validated (evidence's own correction) |
| topic_key upsert / revision versioning | `SaveObservation` path 1, `revision_count` | Validated |
| Cloud multi-device sync | `internal/cloud/*`, `sync_mutations` log | Validated (large, real) |
| Semantic / hybrid / vector search | absent; `embedding` column inert | Correctly absent |
| Decay | `review_after`/`expires_at` reserved but inert | Correctly absent |
| Any QA / retrieval benchmark (LoCoMo etc.) | none in repo or docs | **No benchmark exists** |
| 3.8k GitHub stars | README badge | Plausible (popularity ≠ retrieval quality) |

---

## Relevance to Somnigraph

### What engram does that Somnigraph doesn't
- **Save-time relationship detection.** Somnigraph detects typed edges only during NREM sleep (`scripts/sleep_nrem.py`). engram surfaces candidate conflicts *at write time* via cheap FTS5 + BM25 floor, then defers the expensive LLM adjudication to an explicit `mem_judge` call. The FTS-floor prefilter is a concrete pattern Somnigraph could borrow to give real-time conflict hints without waiting for sleep.
- **Stable-key upsert / semantic versioning.** A rule-derived `topic_key` acts as an identity so re-saving the "same" fact updates in place (`revision_count++`) instead of creating a near-duplicate. Somnigraph's `remember()` (`tools.py`) dedupes at 0.9 embedding similarity but has no stable human-readable revision key.
- **Cloud multi-device sync** via an append-only mutation log + chunk codec. Somnigraph is deliberately single-user; not a gap it wants to fill.
- **Multi-actor disagreement modeling**: `memory_relations` intentionally omits a UNIQUE(source,target) so different actors can assert conflicting relations. Somnigraph's edge table assumes a single truth.

### What Somnigraph does better
- **Retrieval**: hybrid BM25+vector RRF fusion + 26-feature LightGBM reranker + PPR graph expansion vs engram's single-channel FTS5 `ORDER BY rank`. On any lexical-gap/multi-hop query engram has no recall mechanism at all (no embeddings).
- **Feedback loop**: Somnigraph has explicit per-query utility ratings with measured Spearman r=0.70 to GT; engram has none.
- **Consolidation**: Somnigraph's LLM-mediated three-phase sleep (edges, merge/archive, gap analysis, taxonomy) is far richer than engram's save-time candidate pass.
- **Decay / lifecycle**: real per-category exponential decay vs engram's inert `review_after`.
- **Evidence**: Somnigraph has 85.1% LoCoMo QA + multi-hop failure analysis; engram publishes no retrieval or QA numbers.

---

## Worth Stealing (ranked)

### 1. FTS5 + BM25-floor candidate prefilter for real-time conflict hints (Low)
**What**: On `remember()`, run the cheap lexical channel over the new memory's title/summary, keep only candidates above a BM25 floor, and surface them as "possible conflict/duplicate" before the memory is committed — without invoking the LLM.
**Why**: Somnigraph only learns that a new memory contradicts an old one at the next sleep pass. A save-time lexical prefilter gives an immediate cheap signal (and a shortlist the sleep LLM can reuse), closing the write→consolidation latency gap that the proactive-injection work is also circling.
**How**: In `tools.py::remember`, after insert, reuse `fts.py` to fetch top-k on the summary, apply a score floor, and either warn or stash a `pending` edge for `sleep_nrem.py` to adjudicate — mirroring engram's `FindCandidates` (relations.go) → `JudgeBySemantic` split (cheap detect now, LLM judge later).

### 2. Stable rule-derived topic_key as an upsert/revision identity (Low)
**What**: Derive a stable slug (family/segment) from a memory and treat it as an upsert key: re-saving updates in place and increments a revision counter, rather than creating a 0.9-similar near-dup.
**Why**: Somnigraph's similarity-threshold dedup can still accumulate slow-drift near-duplicates of the same evolving fact; a human-readable revision key makes "this is the same thing, newer" explicit and cheap, and pairs naturally with `valid_from/valid_until`.
**How**: Optional `topic_key`-style field on the memory row; `remember()` prefers exact-key upsert (bump a `revision_count`) before falling back to embedding-similarity dedup. Rule-based inference (keyword families) is a zero-LLM starting point.

---

## Not Useful For Us

### Cloud sync / mutation-log replication (internal/cloud, internal/sync)
Large, well-built, but Somnigraph is single-user by design. No adoption value.

### TUI dashboard, plugin/marketplace, multi-platform installers
Product/packaging surface, orthogonal to retrieval-quality research.

### Free-text `type` + keyword topic families
Somnigraph already has a richer enforced category taxonomy + themes + reranker features; engram's keyword heuristics are a downgrade for us.

---

## Connections

- **FTS5-only, write-path-focused, no learned retrieval** — same class as ByteRover (BM25-only) and the agentmemory/MemPalace cluster flagged in the Phase 18 source sweep (`ai-memory-comparison.md`): these systems win adoption on write-path discipline and UX, not retrieval mechanism. engram is another data point that popular agent-memory tools skip embeddings entirely and lean on lexical + good capture ergonomics.
- **Save-time conflict surfacing** is a real-time analogue of Somnigraph's sleep-time NREM contradiction classification — convergent goal (typed supersede/conflict edges), opposite timing (write-time cheap-detect + on-demand LLM judge vs scheduled LLM batch). Contrast worth noting when documenting the proactive/real-time-graph direction in `docs/roadmap.md`.
- **topic_key upsert** echoes the supersession/versioning patterns in memv and memos analyses (stable identity → in-place revision).

---

## Summary Assessment

engram (Gentleman-Programming) is a polished, popular (3.8k-star) single-binary Go agent-memory tool whose engineering weight is in **capture ergonomics and cross-device cloud sync**, not retrieval science. Retrieval is single-channel SQLite FTS5 (`ORDER BY rank`) with a topic-key exact-lookup shortcut — no embeddings, no fusion, no reranker, no graph, no decay. The `embedding` and `expires_at` columns exist in the schema but are inert. There are **no published retrieval or QA benchmarks**, so nothing here is numerically comparable to Somnigraph's 85.1% LoCoMo QA; the popularity is a UX/DX signal, not a quality one.

The single most transferable idea is the **cheap-detect / expensive-judge split for conflicts**: FTS5 + a BM25 floor to surface candidate contradictions *at save time*, deferring LLM adjudication to an explicit tool call. Somnigraph currently binds all relationship detection to the sleep cycle; a save-time lexical prefilter would give immediate conflict/dup hints and hand the sleep LLM a pre-scored shortlist. The secondary idea — a stable rule-derived `topic_key` as an upsert/revision identity — is a clean complement to similarity-threshold dedup. Both are Low-effort and additive; neither changes Somnigraph's core retrieval stack, which remains strictly ahead of engram's.

Evidence-file cross-check: the carsteneu audit is unusually accurate for this repo (it self-corrects dedup=true, dataSources=2, llmFlex=2, all confirmed in code). The one framing caution for our corpus: its "searchModes: 4" reads like retrieval sophistication but is four separate query endpoints over one FTS5 index, not multi-channel fused retrieval — and the report reports zero end-to-end accuracy numbers because none exist.

---

*Cross-check note*: Verified against code — schema (store.go L695-935), FTS-only `Search` (L3102), 15-min dedup window (L2316), inert embedding/expires_at columns, `FindCandidates` BM25 floor (relations.go L324) + LLM `JudgeBySemantic` (L747). Evidence file consistent with code; no material discrepancies, no benchmark inflation (no benchmark at all).
