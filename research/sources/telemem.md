# TeleMem — Mem0 fork adding write-time semantic-cluster dedup + per-character extraction, for Chinese roleplay dialogue memory

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Paper Overview
<!-- Tech report, not a peer-reviewed paper -->

**Paper**: TeleMem tech report, arXiv:2601.06037 (v4, 2026-01-22); PDF vendored at `docs/TeleMem_Tech_Report.pdf`.
**Authors**: TeleAI-UAGI (China Telecom AI). Repo: `https://github.com/TeleAI-UAGI/telemem` (461 stars).
**Code**: The user-provided URL `Tele-AI/TeleMem` 404s; the org was renamed to `TeleAI-UAGI` and repo to `telemem`.

**Problem addressed**: Mem0's flat fact-extraction loses character identity and conflates speakers in ultra-long multi-character roleplay dialogues (600+ turns). TeleMem targets Chinese roleplay/companion-chat memory: keep per-character memory isolated, dedup aggressively, and survive very long conversations.

**Core approach**: A drop-in Mem0 subclass (`TeleMemory(mem0.Memory)`) that overrides the write and search paths. On write it (1) LLM-summarizes each dialogue round into a short (20-100 char) Chinese summary, (2) additionally runs a per-character focused extraction that pulls four info types (relationships, plot events, traits, items/locations) into separate `person_{id}` scopes plus a global `events` scope, (3) buffers summaries and, on flush, clusters new+similar-existing memories by cosine ≥0.95 and issues one LLM merge/dedup call per cluster. Retrieval is plain vector search across the character scope + `events`, with an optional (default-off) reranker. A separate optional pipeline does video memory (frame extraction → VLM captions → nano-vectordb → ReAct QA).

**Key claims**: 86.33% QA accuracy on the ZH-4O Chinese multi-character benchmark vs Mem0 70.20 (README table). "Greatly reduced token cost" (no number). Video memory is a genuine differentiator among memory libs.

---

## Architecture

### Storage & Schema
Mem0's vector-store abstraction; repo configs (`config/config.yaml` and variants) all set `vector_store.provider: faiss` at `db/faiss_db`. Note: bare `Memory()` inherits mem0's *default* store (not FAISS) unless a config is loaded explicitly — README line 260 flags this. Memory unit is a flat record: `summary` text + embedding + metadata (`user_id`/`agent_id`/`run_id`, timestamp). No graph store, no entity table, no keywords/tags, no decay/valid_from fields. Scoping is done purely by a `user_id` metadata value: `"events"` = global summaries, `"person_{id}"` = per-character summaries (`_get_buffer_key`, `mem0.py:66`).

### Memory Types
No typed taxonomy in the Somnigraph sense (no episodic/semantic/procedural/reflection distinction with different handling). The only axis is **scope**: global `events` vs per-character `person_{uid}`. `memory_type="procedural_memory"` is nominally accepted via the Mem0 signature but not specially implemented here.

### Write Path
The interesting part. Two code paths:
- Non-batch `add()` → `_extract_summary_from_messages()` → `_sync_memory_to_vector_store()` (`mem0.py:403,472`): summarize the latest round with prior rounds as context; for each extracted summary, vector-search up to 5 similar existing memories (threshold 0.95), then one LLM "update memory" call (`get_update_memory_prompt`) that keeps new info and drops duplicates/valueless, writing survivors.
- Batch `add_batch()` → buffer per scope, flush at `buffer_size` (default 64) via `_flush_buffer()` (`mem0.py:117`): pool new + all similar-existing memories, **cluster by embedding cosine ≥ `similarity_threshold` (0.95)** (`_cluster_memories_by_embedding`, greedy single-linkage), then **one LLM merge call per multi-item cluster** (`get_update_memory_prompt`) that fuses new+old summaries into a deduped list. Singleton new memories are written directly; clusters with only old memories are skipped.

Extraction prompts are Chinese (`utils.py`): generic round summary (`get_recent_messages_prompt`) and character-focused four-slot extraction (`get_person_prompt`). `extract_events_from_text` is a large, brittle regex parser to pull summaries out of free-form LLM text (marker `这段内容的摘要是：[...]`, JSON fallback, bullet fallback, Chinese trigger-word sentence splitting). No salience/quality scoring, no confidence, no contradiction handling — dedup is the only gate.

### Retrieval
Plain vector search. `search()` (`mem0.py:510`) fans out over `[user_id, "events"]` scopes, calls mem0's `_search_vector_store` per scope, concatenates, and **optionally** reranks: `if rerank and self.reranker` — but `self.reranker` is never set by TeleMem code or any config, so it is `None` by default (inherited mem0 attribute). **Default retrieval is single-channel dense vector only.** No BM25, no hybrid/RRF fusion, no graph expansion, no learned reranker, no scoring formula beyond cosine.

### Consolidation / Processing
No offline/sleep pass. The write-time cluster-and-merge dedup is the only consolidation. No decay-driven summarization, no gap analysis, no re-clustering of the corpus over time. (The video pipeline's "smart caching" just skips recomputation of frames/captions — not memory consolidation.)

### Lifecycle Management
None beyond what Mem0 provides. No decay, no reheat, no archival, no versioning/supersession, no custom forget. The evidence file confirms all of `decay/contradiction/quarantine/autoResolve/trustModel/explicitForget` are absent.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| 86.33% QA on ZH-4O, +19% over Mem0 | README table, own benchmark; Qwen3-8B reader/judge | Plausible **but on a Chinese roleplay benchmark, not LoCoMo** — not comparable to Somnigraph's 85.1 LoCoMo QA. Own-benchmark, own-eval-harness. |
| "Semantic clustering & deduplication using LLMs" | `_cluster_memories_by_embedding` + `_flush_buffer` per-cluster LLM merge — real, in code | Validated (code does exactly this) |
| "Character-profiled memory, precise isolation" | `get_person_prompt` per-character extraction into `person_{id}` scopes | Validated as a mechanism; "isolation" = a metadata filter, not hard partitioning |
| "Reduced token cost" | No number, no methodology | Unvalidated |
| Drop-in Mem0 replacement | `from .mem0 import TeleMemory as Memory`; matching `add`/`search` signatures | Validated |
| Multimodal video memory/QA | `add_mm`/`search_mm` + `mm_utils/` ReAct agent | Validated; genuinely uncommon feature |

---

## Relevance to Somnigraph

### What TeleMem does that Somnigraph doesn't
- **Write-time batched dedup via clustering.** Somnigraph's dedup/merge is a *sleep-time pairwise* NREM classification (`scripts/sleep_nrem.py`) — O(n²)-ish over the newly-active set. TeleMem clusters candidates by cosine first, then does **one LLM call per cluster**, which is a cheaper batching strategy for the merge decision. This is a write-path move; Somnigraph has essentially no write-path dedup gate (memories land, sleep reconciles later).
- **Per-entity focused extraction at write time.** `get_person_prompt` extracts a clean, single-subject summary per target character. Somnigraph has no entity resolution and no per-entity extraction (a known gap — no module owns this; extraction is whatever the caller wrote).
- **Multimodal video memory.** Entirely outside Somnigraph's scope.

### What Somnigraph does better
- **Retrieval.** Somnigraph is hybrid BM25+vector with RRF fusion and a 26-feature LightGBM reranker (`reranker.py`, `fts.py`, `scoring.py`); TeleMem is single-channel dense with the reranker hook wired but off. On the multi-hop vocabulary gap Somnigraph diagnosed as its ceiling, TeleMem's dense-only path would fare worse.
- **Graph + PPR expansion, typed edges, Hebbian PMI** — TeleMem has no graph at all.
- **Feedback loop** with measured GT correlation (r=0.70) — TeleMem has none.
- **Lifecycle**: decay, reheat, versioning — TeleMem has none.
- **Offline LLM-mediated sleep** (NREM/REM) — TeleMem has none; its only consolidation is inline dedup.

---

## Worth Stealing (ranked)

### 1. Cluster-then-merge dedup (one LLM call per cluster) (Medium)
**What**: Before merging candidate memories, greedily cluster them by embedding cosine (≥ threshold) and issue a single LLM merge call per cluster rather than a pairwise call per candidate pair (`_flush_buffer`, `_cluster_memories_by_embedding`).
**Why**: Somnigraph's `sleep_nrem.py` does pairwise classification, whose LLM-call count grows quadratically with the active set. For dense duplicate neighborhoods, clustering collapses many pairwise calls into one merge, cutting sleep cost. Additive, not redundant: it's a *batching optimization* of the same merge decision, and could complement (pre-filter) the typed-edge classifier rather than replace it.
**How**: In `sleep_nrem.py`, before the pairwise loop, run single-linkage clustering on the candidate embeddings at a high threshold (e.g. 0.9+); route each cluster to one merge/archive LLM call, and only run the full typed-edge pairwise classifier *across* cluster representatives. Keep the pairwise path for supports/contradicts/evolves edges (which clustering can't infer). Guard: greedy single-linkage can chain — cap cluster size.

### 2. Per-entity focused extraction as a write-path quality gate (Medium)
**What**: A focused-extraction prompt that rewrites a raw memory into a clean, single-subject summary (TeleMem does this per character with four fixed slots).
**Why**: Phase 18's cross-repo finding was that *write-path quality*, not retrieval, is where LoCoMo/LME leaders win. Somnigraph has no write-time normalization step. A "focus + slot" rewrite could raise the signal of stored summaries and improve embedding quality (`embeddings.py` enriches but doesn't rewrite).
**How**: Optional pre-store LLM pass in `tools.py:remember` that normalizes the summary to subject + fact, feeding the same enriched-embedding pipeline. Cost/benefit unproven — gate behind a flag and A/B on LoCoMo recall.

---

## Not Useful For Us

### Multimodal video pipeline (`add_mm`/`search_mm`, `mm_utils/`)
Frame extraction + VLM captioning + ReAct video QA. Off-scope for a single-user text memory server.

### Mem0 API compatibility / drop-in packaging
Somnigraph is MCP-native and single-user; Mem0 signature parity is irrelevant.

### `extract_events_from_text` regex parser
A ~180-line brittle Chinese-specific regex cascade to recover summaries from unstructured LLM output — a symptom of not using structured decoding. Somnigraph should not import this fragility; use JSON-mode/structured output instead.

---

## Connections
- **Mem0 fork lineage**: same base as several audited systems; the table-note "Mem0 drop-in" is accurate. Convergent with the write-path-quality thesis from **agentmemory** and **ByteRover** (Phase 18 sweep) — TeleMem also wins its benchmark on write-side processing (per-character extraction + dedup), not retrieval (dense-only). Independent corroboration that leaders bank on the write path.
- **MOOM** (arXiv:2509.11860) supplies the ZH-4O benchmark and is a listed baseline (72.60); TeleMem is essentially a leaner competitor to MOOM's dual-branch narrative memory.
- Dedup-by-clustering echoes the **semantic-cluster merge** patterns noted in earlier fork analyses; nothing novel over the general "cluster + LLM merge" idiom, but the per-cluster single-call framing is a clean efficiency point.

---

## Summary Assessment

TeleMem is a focused Mem0 fork for Chinese multi-character roleplay dialogue. Its two real contributions are (1) write-time **semantic-cluster dedup** — cluster candidates by cosine, one LLM merge per cluster — and (2) **per-character focused extraction** into isolated scopes plus a global `events` scope. Retrieval is deliberately plain: single-channel dense vector, with a reranker hook that is wired but off by default. There is no graph, no feedback loop, no decay, no offline consolidation, and no lifecycle management. It is a narrower, shallower system than Somnigraph on every retrieval and memory-dynamics axis, and its headline 86.33% is an own-benchmark, own-harness number on Chinese roleplay QA — not comparable to our LoCoMo 85.1.

The single most useful takeaway is the **cluster-then-merge batching** as an efficiency lever for `sleep_nrem.py`: it doesn't add capability, but it could cut the quadratic LLM-call cost of dense duplicate neighborhoods. The per-entity focused-extraction idea is worth a note against Somnigraph's absent write-path quality gate, consistent with the Phase 18 conclusion — but it's an untested bet, not an adoption.

**Evidence-file cross-check (sharpest correction)**: The carsteneu evidence file is materially **stale and cites wrong file/function names**. It dates the audit 2026-05-28 and asserts (a) "No MCP server, no hooks" and (b) "no LongMemEval results/harness" — but the current repo ships `telemem-mcp` (an actual MCP stdio server, `telemem/mcp/server.py`, `uvx telemem`, v1.7.1 2026-06-12) and a LongMemEval harness under `baselines/longmemeval/`. It also attributes the write path to `telemem/main.py` with functions `_find_similar_memories`/`_cluster_memories`/`_process_cluster`/`_save_faiss_index`/`_process_single_round`; the real code lives in `telemem/mem0.py` with `_cluster_memories_by_embedding`/`_flush_buffer`/`_extract_summary_from_messages`/`_sync_memory_to_vector_store` — those exact names do not exist. The *mechanisms* (cluster dedup, per-character extraction, FAISS+JSON) are right; the file/function citations and the MCP/LongMemEval "absent" checkmarks are wrong. Separately, the 86.33% cell is end-to-end QA but on **ZH-4O (Chinese roleplay)**, and must not be read against LoCoMo QA numbers.
