# TencentDB-AM — Layered L0-L3 personalization memory + Mermaid symbolic context-offload, as an OpenClaw/Hermes plugin

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

TencentDB Agent Memory (`@tencentdb-agent-memory/memory-tencentdb`, MIT, Tencent) is a **TypeScript/Node 22+** plugin, not a paper. It bundles two largely independent subsystems:

1. **Long-term personalization** — a semantic pyramid L0 Conversation → L1 Atom → L2 Scenario → L3 Persona, backed by SQLite.
2. **Short-term symbolic offload** — verbose tool logs are dumped to files and condensed into a compact **Mermaid canvas** with `node_id` back-references, kept in context instead of the raw logs.

It ships as an OpenClaw plugin and a Hermes gateway sidecar (HTTP on `:8420`). Everything is LLM-mediated (default DeepSeek-V3.2 via Tencent LKE).

### Storage & Schema
SQLite via Node's built-in `node:sqlite` (`DatabaseSync`) + `sqlite-vec` (vec0 virtual tables) + FTS5 (`src/core/store/sqlite.ts`, 2332 lines). Two layers persisted:
- `l1_records` (structured memories) + `l1_vec` (cosine vec0) + `l1_fts` (BM25, jieba-segmented).
- `l0_conversations` (raw messages) + `l0_vec` + `l0_fts`.
- JSONL daily shards (`records/YYYY-MM-DD.jsonl`) are the append-only source of truth; SQLite is the retrieval engine (`l1-writer.ts` dual-write).
- Upper layers (L2 scenes, L3 persona) are **plain Markdown files** on disk (`scene_blocks/*.md`, `persona.md`) — deliberately white-box / human-inspectable.

`MemoryRecord` (`l1-writer.ts:49`): `id, content, type, priority, scene_name, source_message_ids[], metadata, timestamps[], createdAt, updatedAt, sessionKey, sessionId`. `type` ∈ {`persona`, `episodic`, `instruction`} (reduced from 4). `priority` is 0-100 (`-1` = strict global instruction). `metadata` carries episodic activity time ranges. No graph edges, no decay_rate, no valid_from/valid_until.

### Memory Types
Three L1 types only (persona / episodic / instruction). Higher structure is expressed by the L2/L3 layering (scenes group atoms; persona summarizes scenes), not by a rich category taxonomy. Compare Somnigraph's 5 categories + priority + themes + typed edges — TencentDB's typing is thinner but its *layering* is richer.

### Write Path
Triggered every N conversations (`pipeline.everyNConversations`, default 5, with warmup doubling 1→2→4…) or on idle timeout. Pipeline:
1. **L1 extraction** — LLM reads L0 messages, emits atomic memories (`l1-extractor.ts`, prompt-driven).
2. **Batch dedup / conflict detection** (`record/l1-dedup.ts`, on by default) — for each new memory, recall top-K candidates via **vector search (primary) or FTS5 BM25 (degraded)**, then a **single batched LLM call** returns one of `store / update / merge / skip` per memory (with `merged_content/type/priority/timestamps`). If neither vector nor FTS is available, dedup is skipped and everything is stored.
3. **Write** (`l1-writer.ts`) — update/merge deletes target rows from SQLite in real time and appends the new record to JSONL + vec + FTS.
4. **L2 scene aggregation** (`scene/scene-extractor.ts`) — a *sandboxed* LLM agent with file tools reads/writes `scene_blocks/*.md`, forced to MERGE when scene count hits a cap (default 15).
5. **L3 persona** (`persona/persona-generator.ts`) — regenerated every `persona.triggerEveryN` (default 50) memories, or on an explicit `[PERSONA_UPDATE_REQUEST]` signal parsed from L2 output.

This is genuine **write-time quality gating** — the thing Somnigraph explicitly lacks (it defers merge/archive to offline sleep).

### Retrieval
`core/tools/memory-search.ts` + `core/hooks/auto-recall.ts`. Hybrid by default:
- FTS5 keyword (jieba `cutForSearch` segmentation, BM25 rank → 0-1 score) **and** vector cosine, run in parallel, over-retrieve `candidateK = limit*3`.
- Fuse via **Reciprocal Rank Fusion, k=60** (`store/search-utils.ts:18`, textbook constant, not tuned). Degrades to single-channel when one side is empty.
- Secondary `type` / `scene` filters, then trim to `limit` (default 5).
- **No learned reranker. No graph expansion. No feedback loop.** The Mermaid "symbolic graph" is short-term task-state, not a retrieval graph — it plays no role in L1/L3 recall.

### Consolidation / Processing
No sleep cycle. Consolidation is inline/threshold-driven: L2 scene merges (cap-forced) and L3 persona regeneration (every 50 memories). The **L2 Mermaid offload pipeline** (`offload/pipelines/l2-mermaid.ts`) is a separate consolidation of *tool logs*: batches null-`node_id` offload entries, sends them to a backend to generate a Mermaid canvas, backfills `node_id`s. Triggered by null-entry threshold or timeout — not a biological/offline analog.

### Lifecycle Management
TTL only: `deleteL1Expired(cutoffIso)` prunes by `updated_time` with an 80%-of-table safety cap (`sqlite.ts:1280`). `capture.l0l1RetentionDays` default 0 = never clean up. No exponential decay, no reheat-on-access, no versioning beyond the append-only JSONL trail + `.backup/` snapshots of scene dirs.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Hybrid BM25+vector+RRF retrieval | Real; default `recall.strategy: "hybrid"`, `rrfMerge` k=60 | **Validated** (code-confirmed) |
| Write-time dedup/conflict detection | Real; `batchDedup` LLM store/update/merge/skip, `enableDedup: true` default | **Validated** |
| L0→L3 layering with drill-down traceability | Real; SQLite lower layers + Markdown scenes/persona, `result_ref`/`node_id` links | **Validated** (design), unmeasured |
| "−61.38% tokens, +51.52% pass rate" (WideSearch), PersonaMem 48%→76% | README table only; **no repro scripts, datasets, or eval harness in repo** | **Unvalidated** (marketing) |
| Symbolic Mermaid memory cuts token cost while preserving traceability | Plausible mechanism (offload + node_id grep), no ablation | Plausible, unmeasured |
| Entity extraction / NER | Absent — `MemoryRecord` has no entity fields, no NER pipeline | Correctly **false** |

---

## Relevance to Somnigraph

### What TencentDB-AM does that Somnigraph doesn't
- **Write-time conflict resolution** (`l1-dedup.ts`): a single batched LLM call classifies each new memory as store/update/merge/skip against vector-recalled candidates. Somnigraph does this only offline in `scripts/sleep_nrem.py`; `tools.py::remember` has no write-path gate. This is the "write-path quality gating" gap named in STEWARDSHIP.
- **Symbolic context offloading** (Mermaid canvas + `node_id` drill-down): compress in-context tool logs to a few hundred tokens, grep `node_id` to recover raw text. Somnigraph has no in-context/short-term compression layer at all — it is a pure retrieval store.
- **White-box editable upper layers**: L2 scenes and L3 persona are Markdown a human/agent can open and edit. Somnigraph's layers (detail/summary/gestalt) live in SQLite, not files.
- **Multi-session/multi-channel keying** (`sessionKey` vs `sessionId`) and Chinese-first FTS (jieba). Somnigraph is single-user English.

### What Somnigraph does better
- **Retrieval quality**: Somnigraph adds a 26-feature LightGBM reranker (`reranker.py`, NDCG 0.7958) and PPR graph expansion (`scoring.py`) on top of RRF. TencentDB stops at raw RRF with an untuned k=60 — no reranking, no learned scoring, no graph-conditioned recall.
- **Feedback loop**: Somnigraph has explicit per-query utility ratings (Spearman r=0.70 with GT) and Hebbian co-retrieval; TencentDB has none — retrieval never learns from use.
- **Graph reasoning**: Somnigraph's typed edges (supports/contradicts/evolves) + betweenness feature + PPR give real multi-hop retrieval. TencentDB's only "graph" is the Mermaid *task-state* canvas, which never touches long-term recall.
- **Principled lifecycle**: per-category exponential decay with reheat vs TencentDB's blunt TTL-or-never.
- **Reproducible evaluation**: Somnigraph reports LoCoMo QA 85.1% with a vendored harness; TencentDB's headline numbers ship without a repro path.

---

## Worth Stealing (ranked)

### 1. Write-time batch conflict detection (Medium)
**What**: For each newly extracted memory, recall top-K similar existing memories (vector, FTS fallback) and make **one batched LLM call** that returns store/update/merge/skip + merged fields per memory (`l1-dedup.ts::batchDedup`, `runLlmJudgment`).
**Why**: STEWARDSHIP's Phase 18 sweep already concluded write-path discipline is what the LoCoMo/LME leaders win on. Somnigraph only merges/archives during sleep, so between sleeps the store accumulates near-duplicates and stale contradictions that dilute retrieval. A cheap write-time gate is the missing complement — and TencentDB shows it can be a single batched call, not per-memory.
**How**: A new lightweight path in `tools.py::impl_remember` (or a pre-sleep micro-pass): reuse existing hybrid recall to fetch candidates for the incoming memory, prompt an LLM for a store/update/merge/skip verdict, and apply update/merge inline. Keep it optional and conservative (default to store on any parse failure, exactly as `batchDedup` does) so it never blocks a write. Design tension to note: Somnigraph's thesis is *offline* consolidation; this is a deliberate, bounded exception, not a rearchitecture.

### 2. Symbolic index with deterministic drill-down (note-only convergence)
**What**: A compact top-layer symbol (Mermaid canvas / persona) that indexes into progressively rawer layers via stable IDs (`node_id`, `result_ref`), guaranteeing a lossless path back to evidence.
**Why**: This is the *same* philosophy as Somnigraph's detail/summary/gestalt layers — independent arrival at "upper layers preserve structure, lower layers preserve evidence." Worth citing as convergent validation of the layered-memory design, not as something to adopt (Somnigraph already has the mechanism, just in SQLite rather than Markdown).

---

## Not Useful For Us

### Mermaid short-term context offload
Solves in-context token overload during long agent tasks — a context-window/host concern. Somnigraph is an MCP retrieval server that never manages the live context window, so the offload subsystem has no home here.

### jieba Chinese segmentation, sessionKey/sessionId multi-channel keying, Hermes/OpenClaw adapters
Single-user English deployment; these are deployment-surface features with no analog in Somnigraph's design.

### RRF k=60 default
Somnigraph already Bayesian-optimized its fusion (k=14); adopting a textbook constant would be a regression.

---

## Connections

- **Write-path discipline theme**: directly corroborates the Phase 18 finding (see `docs/sessions/2026-06-28-phase18-source-sweep.md`, and `byterover.md` / `agentmemory.md`) that *write-time* quality — not retrieval cleverness — is where the leaders win. TencentDB is another independent vote: its retrieval is deliberately plain RRF, and all its sophistication is in extraction/dedup/layering.
- **Layered/progressive-disclosure memory**: convergent with the summary/gestalt layering and with hierarchical systems like MemPalace/MemOS (verbatim-plus-structure). "Lower layers preserve evidence, upper preserve structure" is the same insight as Somnigraph's layers.
- **White-box Markdown memory**: shares the human-editable-file stance with vault-style systems; contrast with Somnigraph's SQLite-first store.
- **No feedback loop / no reranker**: same profile as most surveyed OSS memory systems — reinforces that Somnigraph's learned reranker + feedback loop remain rare differentiators.

---

## Summary Assessment

TencentDB Agent Memory's real contribution is **layering as a unified paradigm** — the same discipline applied to long-term personalization (Conversation→Atom→Scenario→Persona), short-term task state (logs→Mermaid canvas), and skills — each with a guaranteed drill-down path from an abstract symbol back to raw evidence. It is well-engineered production code (fault-tolerant SQLite wrappers, degraded-mode no-ops, sandboxed LLM file agents, WAL/backup discipline) rather than a research artifact, and the two most interesting mechanisms — write-time batch conflict detection and Mermaid context offload — are genuinely useful, not vaporware.

The single most valuable thing for Somnigraph is the **write-time batch dedup pattern**: it is a concrete, conservative template for the write-path quality gate that STEWARDSHIP and the Phase 18 sweep both flag as Somnigraph's clearest gap, and it shows the gate can be one batched LLM call over hybrid-recalled candidates. Everything else is either orthogonal (context offload belongs to the host, not an MCP store) or something Somnigraph already does better (reranking, feedback, graph, decay).

What's overhyped: the headline benchmark numbers (−61% tokens, PersonaMem 48%→76%, WideSearch 33%→50%) ship with **no reproduction path** — no datasets, scripts, or harness in the repo — and they measure agent-task pass rate / token cost, not memory QA recall, so they are **not comparable to Somnigraph's 85.1% LoCoMo QA**. The prior-triage label "Mermaid symbolic-graph memory" also oversells: the Mermaid graph is a short-term task-state canvas, not a retrieval knowledge graph; long-term recall is plain BM25+vector+RRF with no graph and no reranker. The carsteneu evidence file's own corrections (hybrid→true, dedup→true, entities→false, benchmark-methodology→false) are all accurate against the code.
