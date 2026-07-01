# Nanobot — File-based, git-versioned agent memory (whole-file injection + LLM "Dream" consolidation, no retrieval engine)

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

Nanobot (HKUDS/nanobot, ~43k stars, MIT) is a standalone **agent framework**, not a memory system. Memory is one subsystem (`nanobot/agent/memory.py`, 1057 lines). There is **no retrieval engine** — no embeddings, no vector store, no BM25/FTS, no reranker, no graph. Confirmed by grep: `embedding|faiss|sqlite-vec|rerank|bm25|cosine` returns zero hits in `nanobot/`.

### Storage & Schema
Plain files in a workspace (`MemoryStore`, memory.py:40):
- `SOUL.md` — agent voice/behavior rules (durable, unstructured markdown)
- `USER.md` — user profile/preferences (durable, unstructured markdown)
- `memory/MEMORY.md` — project facts/decisions (durable, unstructured markdown)
- `memory/history.jsonl` — append-only conversation archive; each line `{cursor:int, timestamp:str, content:str, session_key?:str}` (memory.py:283). **3 schema fields**, not 7 as the comparison table claims.
- `.cursor` (consolidator write position), `.dream_cursor` (dream read position), `.git/` (version history).

The only "structured" store is history.jsonl (3 fields). The three long-term files have **zero schema** — freetext markdown edited by an LLM.

### Memory Types
Type distinction is **by file, not by tag**: voice (SOUL), user (USER), project (MEMORY), raw history (jsonl). The Dream prompt (`templates/agent/dream.md`) enforces MECE routing between files and a companion `SKILL.md` for reusable procedures. No episodic/semantic/procedural typing on individual entries; no priority, themes, or valid_from/valid_until fields.

### Write Path
Two stages, both LLM-driven:
1. **Consolidator** (`Consolidator`, memory.py:640) — token-budget triggered. When a session's estimated prompt exceeds a budget (`context_window - completion - safety`), it picks a user-turn boundary (`pick_consolidation_boundary`), LLM-summarizes the evicted slice (`archive()`, system prompt `consolidator_archive.md`), and appends the summary to history.jsonl. On LLM failure it falls back to `raw_archive()` (dump raw messages, tagged `[RAW]`). Purely a context-window pressure valve — no extraction, dedup, salience scoring, or enrichment.
2. **Dream** (cron, default `intervalH=2`) — reads unprocessed history.jsonl since `.dream_cursor` (`build_dream_prompt`, memory.py:484, batch ≤20 entries), plus current SOUL/USER/MEMORY, and runs an ephemeral agent with a restricted tool set (`build_dream_tools`: Read/Edit/Write/ApplyPatch scoped to the memory files + skills dir). The Dream system prompt (`dream.md`) is the actual "quality" logic: MECE file routing, atomic-fact extraction, in-place correction of conflicts, aggressive pruning, and a "**do not store facts a quick web search would surface**" gate. Output is committed via GitStore.

No deduplication in code — the Dream *prompt* instructs the LLM to keep the most-specific copy and delete duplicates. Whether that happens depends entirely on the LLM.

### Retrieval
**There is none in the search sense.** "Retrieval" = injecting the *entire* MEMORY.md into the system prompt every turn (`get_memory_context`, memory.py:232; `context.py:86`) plus recent unprocessed history entries (`read_recent_history_for_prompt`). Searching older history is a **manual grep/jq workflow** documented in `skills/memory/SKILL.md` — the agent is told to call a `grep` tool over history.jsonl. No ranking, no fusion, no relevance scoring.

### Consolidation / Processing
The Dream cycle *is* the consolidation: LLM re-reads new history + long-term files and makes "the smallest honest change." This is genuinely offline LLM-mediated consolidation — conceptually the same family as Somnigraph's sleep — but it operates on **whole markdown files via text edits**, not on a graph of typed memory nodes. No relationship detection, no contradiction typing, no merge/archive of discrete units.

### Lifecycle Management
- **Versioning**: strong. `GitStore` (utils/gitstore.py, 390 lines) auto-commits after each Dream edit (`auto_commit`), supports `log`, `diff_commits`, `show_commit_diff`, `revert`, and `line_ages` (git-blame line age). User commands `/dream-log`, `/dream-log <sha>`, `/dream-restore`, `/dream-restore <sha>` inspect and roll back memory to a prior state.
- **Decay**: none programmatic. Git line-age (`_compute_line_ages`) can be surfaced to the Dream LLM as an informational "← Nd" staleness hint, but nothing decays or is removed automatically. Pruning is whatever the LLM chooses to delete under the prompt's age/decay rules.
- **Forget**: no user-facing `/forget`. History compaction (`compact_history`) just drops oldest entries past a 1000-entry cap.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| "Memory is alive but not chaotic" / interpretive not archival | Dream LLM edits durable files surgically | Plausible — but quality rides entirely on the LLM + prompt; no measured eval |
| Git-versioned, restorable memory | `GitStore.auto_commit/revert`, `/dream-restore <sha>` | Validated in code — real, working audit/restore layer |
| Auto-extraction of durable knowledge | Dream 2-stage pipeline, `dream.md` MECE routing prompt | Validated as a mechanism; no benchmark of extraction quality |
| "decay" (comparison table) | Only informational git line-age hint to the LLM | **Overstated** — no programmatic decay/forgetting |
| "semantic" / "searchModes=2" / "entities" / "keywords" | grep finds no embeddings, no entity/tag extraction | **False** — text-only; at most 1 manual grep "mode" |
| "schemaFields=7" | history.jsonl has `{cursor,timestamp,content}` | **False** — 3 fields; durable files are schemaless |
| "p_claude" (Claude Code plugin) | Own CLI/agent loop; `CLAUDE.md` is dev guidance | **False** — standalone framework, not a CC memory backend |
| No end-to-end QA benchmark (LoCoMo/PERMA) | none in repo | Not comparable to Somnigraph's 85.1% LoCoMo QA |

---

## Relevance to Somnigraph

### What Nanobot does that Somnigraph doesn't
- **Git-versioned, restorable consolidation output.** Every Dream edit is an auditable commit with diff and one-command rollback. Somnigraph's sleep (`sleep_nrem.py` merge/archive) mutates the SQLite store without a version-diff/restore surface — a bad consolidation pass is not trivially inspectable or reversible. This is the one place Nanobot is cleanly ahead.
- **In-prompt write-path quality gating.** The `dream.md` prompt is an explicit salience/MECE/dedup gate ("keep the most specific copy," "don't store what a web search surfaces," atomic facts, in-place correction). Somnigraph has **no write-path quality gate** (a known gap — `tools.py` `remember` stores whatever it's given). Nanobot's gating is prompt-only and unmeasured, but the *checklist* is concrete.

### What Somnigraph does better
- **Everything retrieval.** Somnigraph has hybrid BM25+vector RRF fusion, a 26-feature LightGBM reranker (NDCG=0.7958), and PPR graph expansion. Nanobot has whole-file injection + manual grep. Not comparable.
- **Discrete, typed, scored memory units** vs. Nanobot's schemaless markdown blobs. Somnigraph's `db.py` schema (category, priority, themes, valid_from/until, decay_rate) enables ranking, decay, and graph edges that Nanobot structurally cannot do.
- **Measured feedback loop** (Spearman r=0.70 with GT) and **typed graph edges** detected during sleep. Nanobot has neither.
- **Programmatic decay** (`per-category exponential`) vs. Nanobot's LLM-discretionary pruning.
- **Scale**: whole-file injection breaks when MEMORY.md grows; Somnigraph retrieves top-k from an unbounded store.

---

## Worth Stealing (ranked)

### 1. Git-versioned consolidation with diff/restore (Medium)
**What**: Wrap sleep's mutations (NREM merge/archive, REM taxonomy edits) in a commit-per-run audit layer with a diff view and one-command rollback of a bad pass. Nanobot's `GitStore.auto_commit/diff_commits/revert` + `/dream-log`/`/dream-restore` are the model.
**Why**: Somnigraph's sleep does destructive-ish merge/archive with no easy "what did this pass change / undo it" surface. This directly serves the repo's honest-accounting ethos: consolidation becomes auditable rather than a silent mutation. The seed's "witness reaches the artifact" theme maps onto it — a diff you can inspect before trusting a sleep run.
**How**: Sleep already archives rather than hard-deletes, so the substrate is partly there. Add a per-sleep-run manifest (memory IDs touched, before/after summaries, edges added) written to a versioned log, plus a `sleep-restore <run_id>` that reverses archive/merge/edge ops from the manifest. SQLite, not git, but the diff/restore UX is the borrowed idea. Touches `scripts/sleep_nrem.py`, `scripts/sleep_rem.py`, `db.py` (archive/edge provenance).

### 2. Write-path salience gate as a prompt checklist (Low)
**What**: Nanobot's Dream prompt encodes a concrete keep/delete rubric: atomic facts, correct-in-place on conflict, "don't store what a quick web search surfaces," migrate procedures out of fact-memory. Adopt the *checklist* (not the file-routing) as a salience/quality gate.
**Why**: Somnigraph lacks write-path quality gating — a self-identified gap. Independent corroboration (Phase 18 sweep found write-path quality, not retrieval, is what LoCoMo leaders win on). This is a cheap, prompt-level lever.
**How**: Fold the rubric into the NREM merge/archive prompts and, optionally, a lightweight pre-store salience hint surfaced to the agent at `remember()` time. No schema change. Touches sleep prompts + optionally `tools.py`.

---

## Not Useful For Us

### Whole-file context injection + manual grep retrieval
Injecting all of MEMORY.md every turn and grepping history.jsonl is exactly the architecture Somnigraph exists to replace. Doesn't scale, no ranking. Nothing to take from the retrieval side.

### File-based type routing (SOUL/USER/MEMORY/SKILL)
Somnigraph's per-entry `category` field is strictly more expressive than routing facts to one of four markdown files. The MECE file boundaries are a workaround for having no schema.

### Consolidator token-budget mechanics
Nanobot's context-window pressure valve is agent-runtime plumbing (session eviction), not persistent-memory consolidation. Irrelevant to an MCP memory server.

---

## Connections

Nanobot's Dream is a **markdown-file** cousin of the "LLM-mediated write-time curation" pattern seen across the Phase 18 sweep (ByteRover, MemPalace, agentmemory) — the finding that write-path quality beats retrieval sophistication. It corroborates that thesis while lacking any retrieval to compare. Its git-versioning is a differentiator none of those had. The whole-file-injection approach is the same anti-pattern documented in the full-context baseline that Somnigraph's LoCoMo results beat. Nanobot is closest in *spirit* to the "agent framework with a memory feature" class (vs. dedicated memory systems like Mem0/MemOS); the evidence file explicitly flags it should be classified separately in the comparison table.

---

## Summary Assessment

Nanobot is a popular, well-engineered **agent framework** whose memory subsystem is deliberately minimal: three LLM-curated markdown files plus an append-only JSONL history, versioned in git, with retrieval reduced to whole-file injection and manual grep. Its core contribution is not retrieval (there is none) but **auditable, restorable, LLM-curated durable memory** — the git-versioned Dream loop that lets a user diff and roll back what the memory system decided to remember. That transparency-and-restore layer is genuinely nice and is the one idea worth carrying into Somnigraph's sleep pipeline.

The comparison table significantly overstates Nanobot's memory capabilities — the evidence file itself corrects this well (9 of 20 claimed features false/unverified: no semantic search, no entities, no keywords, no dedup, no explicit forget, no Claude Code integration, 3 schema fields not 7). My code read confirms every one of those corrections. There is no benchmark comparable to Somnigraph's 85.1% LoCoMo QA; Nanobot doesn't do task-level QA eval at all.

Net for Somnigraph: **MAYBE**, not DIVE. Architecturally distant (no vector/BM25/graph/reranker), so nothing to adopt on the retrieval core. But two revisit-if angles survive scrutiny: (1) a git-diff/restore audit layer over sleep consolidation, which aligns tightly with the repo's honest-accounting invariant, and (2) Nanobot's concrete write-path salience checklist, which maps onto Somnigraph's self-identified write-path-gating gap and the Phase 18 write-path thesis.
