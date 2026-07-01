# AIPass - Multi-agent CLI workspace with file-based per-agent memory + a bolt-on symbolic-fragment vector store

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

AIPass (`AIOSAI/AIPass`, v2.4.0 beta, MIT, ~150 stars, `pip install aipass`) is **not a memory system** — it is a CLI-native multi-agent scaffold ("Persistent Agent Workspace") where 13 core agents share a project via plain files and local mailboxes (`.ai_mail.local/`). Memory is *one* of those agents (`src/aipass/memory/`). This analysis covers only the memory agent.

### Storage & Schema

Two layers, both local, no server/daemon:

- **Hot layer — plain JSON in each agent's `.trinity/`**: `passport.json` (identity/persona), `local.json` (session history: `sessions[]`, `key_learnings[]`, `todos[]`), `observations.json` (collaboration patterns). Schema is a `document_metadata` block (10 sub-fields: type, name, version, schema_version, created, last_updated, managed_by, tags, limits, status) plus the content arrays. Entries are date-stamped `[YYYY-MM-DD]`, newest-first. See `templates/LOCAL.template.json`.
- **Cold layer — ChromaDB** (`.chroma/`), embedded with fastembed ONNX `all-MiniLM-L6-v2` (384-dim). IDs are content-hash `sha256[:16]` (dedup by construction). This is an *archive* layer, populated only when the hot JSON overflows.

There is no relational schema, no priority field, no themes, no valid_from/valid_until, no decay_rate. The "memory unit" is a free-form session/learning entry.

### Memory Types

Distinguished only by which JSON file holds them: session history, key learnings, todos, observations (collaboration). A separate **symbolic** subsystem (added 2026-03-17, `handlers/symbolic/`) adds fragment "dimensions": technical_flow, emotional_journey, collaboration_patterns, key_learnings, context_triggers — extracted from chat history.

### Write Path

Two independent paths:

1. **Main path (rollover, FIFO)** — `handlers/rollover/extractor.py`. There is *no* extraction, salience scoring, or quality gating at write time; agents just append to their `.trinity/` JSON. When an array exceeds a config-defined count limit (e.g. max sessions/key_learnings/observations), `_extract_items_v2()` pops the **oldest** entries (`array[-excess:]`), writes them to ChromaDB via `extract_with_metadata()` → `embedder.encode_batch()`. Pure age-based eviction, one backup kept (`rollover_backup_*.json`). The template explicitly forbids manual trimming/deletion — "rollover handles overflow automatically."
2. **Symbolic path** — `handlers/symbolic/extractor.py` runs LLM fragment extraction (`extract_fragments_llm`, model `meta-llama/llama-3.3-70b-instruct:free` via OpenRouter) with a regex fallback (`extract_technical_flow` etc. = keyword-count heuristics). New fragments go through **AUDN dedup** (`deduplicator.py`): an LLM compares each new fragment to existing similar ones and returns **Add / Update(merge) / Delete(supersede) / Noop(duplicate)**. If no API key, defaults to ADD.

### Retrieval

Main path (`handlers/search/query_executor.py`): **vector-only**. Encode query with fastembed (in a separate `.venv` subprocess), query ChromaDB, convert cosine distance to similarity `max(0, 1 - dist/2)`, drop anything below `MIN_SIMILARITY_THRESHOLD = 0.40`, return top-n. No BM25, no RRF, no rerank, no graph — grep for `bm25|rerank|rrf|lightgbm|pagerank|ppr` over `src/aipass/memory` returns **nothing**.

Symbolic path (`handlers/symbolic/retriever.py`) is the closest thing to hybrid retrieval: `retrieve_fragments()` runs up to three channels — vector similarity, ChromaDB metadata **dimension filters**, and **trigger-keyword** matching (Python substring scan over comma-joined `triggers` metadata) — then `_merge_results()` dedups by id and `_rank_results()` does **additive fusion**: `score = similarity + 0.1*(n_methods_found - 1)`. Results get a `relevance_tier`: strong ≥0.65 / moderate ≥0.45 / **serendipity** ≥0.30 / weak.

There is also a proactive **"this reminds me of…" hook** (`handlers/symbolic/hook.py`): surfaces fragments during conversation without an explicit query, gated by `threshold=0.3`, `max_fragments_per_session=5`, `min_messages_between=10`, `cooldown_seconds=300`, with a `surfaced_ids` set for anti-repetition.

### Consolidation / Processing

No sleep/consolidation cycle. "Consolidation" = the FIFO rollover from JSON to ChromaDB. The symbolic AUDN merge is the only content-reconciliation logic, and it runs synchronously at write time, not offline.

### Lifecycle Management

Age-based FIFO only. `learnings/manager.py:get_entry_age()` computes day-age; untimestamped entries → `999999` (evicted first). Limits: ~20 sessions, ~25 key_learnings, ~600 lines/file (config-driven). No exponential decay, no reheat-on-access, **no explicit forget/delete API** (prohibited by template design). Supersession exists only implicitly: same key overwrites in `add_learning()`, content-hash upsert in ChromaDB, or an AUDN DELETE.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Persistent per-agent memory across sessions | `.trinity/` JSON files, read at agent startup (AGENTS.md protocol) | Validated — genuinely works, plain-file simplicity |
| Semantic search over archived memory | ChromaDB + fastembed MiniLM, `execute_search()` | Validated but shallow — single-channel vector, 0.40 floor, top-5 |
| "layeredMemory" (evidence file ✅) | hot JSON + cold ChromaDB rollover | Validated but it's a 2-tier *archive*, not layered abstraction like Somnigraph detail/summary/gestalt |
| "fulltext"/keyword search (evidence file claimed) | none — grep finds no BM25 | **Refuted** — semantic-only; trigger-keyword is substring scan on a metadata field, not full-text |
| "decay" (evidence file ✅) | `get_entry_age`, FIFO rollover | Validated as *age-based eviction*, not decay-weighted ranking |
| "qualityRefine" (evidence file claimed) | seedgo's 36 checks are **code** quality | **Refuted for memory** — no content scoring/salience/relevance ranking on the write path |
| dedup | content-hash IDs + AUDN LLM merge | Validated (two mechanisms) |
| Benchmark numbers (LoCoMo/LME) | **none in repo** | No end-to-end QA or retrieval benchmark exists; not comparable to Somnigraph's 85.1 LoCoMo QA |

---

## Relevance to Somnigraph

### What AIPass does that Somnigraph doesn't
- **Multi-agent workspace + mailboxes** — per-agent identity (`passport.json`) and local file mailboxes for agent-to-agent handoff. Somnigraph is single-user by design; this is out of scope, not a gap.
- **Synchronous write-time AUDN reconciliation** — Somnigraph's `remember()` dedups at 0.9 cosine similarity but defers *merge/supersede* to offline NREM sleep (`sleep_nrem.py`). AIPass applies Add/Update/Delete/Noop synchronously at write. Different timing, same intent.
- **Proactive associative surfacing already shipped** — the "reminds me of…" hook is exactly the shape of Somnigraph's *design-stage* `docs/proactive-injection.md` (UserPromptSubmit hint, session cooldown, anti-repetition). AIPass has a working reference implementation with concrete default parameters.

### What Somnigraph does better
Essentially everything on the retrieval/quality axis: hybrid BM25+vector with **RRF fusion** (`fts.py`+`scoring.py`) vs AIPass's single-channel vector or additive `+0.1` bonus; a **26-feature learned reranker** (`reranker.py`) vs a 0.40 threshold; **PPR graph expansion** with typed edges vs no graph; **explicit feedback loop** with measured r=0.70 GT correlation vs none; **LLM-mediated sleep consolidation** (NREM/REM) vs FIFO eviction; **per-category exponential decay with reheat** vs age-FIFO; a real **LoCoMo QA benchmark (85.1%)** vs no benchmarks at all.

---

## Worth Stealing (ranked)

### 1. AUDN synchronous write-time reconciliation taxonomy (Low/Medium)
**What**: `deduplicator.py`'s clean 4-action framing — **Add / Update(merge) / Delete(supersede) / Noop(duplicate)** — decided by an LLM comparing a new item against its nearest neighbors, returning a merged summary on UPDATE.
**Why**: Somnigraph already does the *classification* offline in NREM sleep (supports/contradicts/evolves + merge/archive), but `remember()` only dedups-or-inserts synchronously; it never merges or supersedes at write time, so near-duplicate churn accumulates until the next sleep. A lightweight AUDN pass in `tools.py:impl_remember()` (over the top-k neighbors already computed for the 0.9 dedup check) could collapse obvious UPDATE/NOOP cases immediately.
**How**: In `impl_remember`, when the nearest neighbor is between the dedup floor and 1.0, run a small AUDN prompt; on UPDATE, edit the existing memory and skip insert; on NOOP, skip. Keep supports/contradicts/evolves edge creation in sleep. Guard behind a flag; the risk is a synchronous LLM call on the write path.

### 2. Proactive "reminds me of…" hook as a reference implementation (Low)
**What**: `handlers/symbolic/hook.py` — a working proactive-surfacing gate with `threshold`, `max_fragments_per_session`, `min_messages_between`, `cooldown_seconds`, and a `surfaced_ids` anti-repetition set.
**Why**: This is **convergent validation** of `docs/proactive-injection.md`. An independent project arrived at the same control surface (session cap + cooldown + anti-repetition) Somnigraph designed on paper — evidence the design shape is right. Its concrete defaults (0.3 threshold, 5/session, 10 messages between, 300s cooldown) are a useful sanity anchor when picking Somnigraph's operating point.
**How**: No code to port — cite it in the proactive-injection design doc as prior art and use its parameters as a starting reference for the floor sweep.

---

## Not Useful For Us

### Multi-agent workspace, mailboxes, passports, 13-agent scaffold
Somnigraph is deliberately single-user MCP memory. The coordination layer is the whole point of AIPass and entirely orthogonal.

### File-first (`.trinity/` JSON as primary store) + FIFO rollover
A plain-JSON hot store with age-based eviction is strictly weaker than Somnigraph's SQLite + decay-weighted retention. Nothing to adopt; it's a simplicity trade-off, not a capability.

### Additive multi-method fusion (`+0.1` per method) and 0.40 similarity floor
Strictly inferior to RRF + learned reranker. Do not adopt.

---

## Connections

- **Write-path discipline theme**: The Phase 18 sweep (see `docs/sessions/2026-06-28-phase18-source-sweep.md`, `agentmemory.md`, `byterover`) found that LoCoMo/LME leaders win on *write-path quality*, not retrieval. AIPass is a counter-example that quietly corroborates it: its retrieval is trivial (vector + 0.40 floor), yet its *only* content-quality mechanism is the write-time AUDN dedup — the same "reconcile at write" instinct, just without any benchmark to show it helps.
- **Proactive injection**: direct convergence with `docs/proactive-injection.md` — same cooldown/anti-repetition/session-cap control surface, independently invented.
- **AUDN vs supersession**: the ADD/UPDATE/DELETE/NOOP taxonomy echoes the supersession/versioning patterns catalogued in other sources (cf. `memv`, MemOS-style updates); AIPass's contribution is naming it as a 4-way LLM decision applied synchronously.

---

## Summary Assessment

AIPass's real product is a multi-agent CLI workspace; memory is a supporting subsystem, and as a *memory system* it is deliberately minimal: plain-JSON hot store, ChromaDB cold archive, vector-only retrieval with a hard similarity floor, and FIFO age eviction. There is no reranker, no fusion, no graph, no feedback loop, no consolidation cycle, and no benchmark. The carsteneu evidence file is broadly accurate and appropriately skeptical — it correctly refutes the "fulltext," "keywords," "explicitForget," and IDE-support claims, and correctly reframes "decay" as age-based eviction and "qualityRefine" as code-quality (not memory-quality) checks. The sharpest correction to carry forward: **AIPass has zero end-to-end QA or retrieval benchmark numbers, so none of its cells are comparable to Somnigraph's 85.1 LoCoMo QA**, and its layered/decay/quality claims describe file-management plumbing, not retrieval quality.

The one genuinely interesting corner is the `symbolic/` subsystem (a late v0.1.0 addition): LLM fragment extraction, the **AUDN** write-time dedup taxonomy, and a working **proactive "reminds me of…" hook**. Neither beats an existing Somnigraph module, but two are worth logging — AUDN as a candidate synchronous complement to sleep-deferred merging, and the proactive hook as independent validation (with concrete default parameters) of the proactive-injection design. Verdict: **MAYBE** — nothing to adopt wholesale, but the AUDN framing and the proactive-hook convergence are worth a revisit when those two threads (write-path reconciliation, proactive injection) become active work.
