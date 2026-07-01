# memanto - Typed-memory management wrapper over the proprietary Moorcheh "information-theoretic" search engine

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Paper Overview

**Paper**: "Memanto: Typed Semantic Memory with Information-Theoretic Retrieval for Long-Horizon Agents" (arXiv 2604.22085, referenced from README; not read here — the repo carries the code, not the PDF).
**Authors**: Moorcheh.ai team.
**Code**: https://github.com/moorcheh-ai/memanto (MIT, created 2026-03-23).

**Problem addressed**: Give agents (Claude Code, Cursor, Codex, CrewAI, LangGraph, etc.) cross-session persistent memory with three primitives — remember / recall / answer — without the operator running a vector DB or an indexing pipeline. "Write a memory and it's searchable immediately" (zero ingestion latency).

**Core approach**: Memanto is a FastAPI service + CLI + integration adapters. It stores typed memory "cards" as documents in **Moorcheh**, a closed-source "information-theoretic semantic engine" that runs either as a free local Docker container or as a hosted cloud service. All retrieval, ranking, and RAG answer-generation happen **inside Moorcheh**. The open-source repo contributes the memory schema (13 typed categories), namespace-per-agent isolation, temporal query modes, TTL enforcement, LLM-based extraction, and an LLM-based daily conflict/summary report.

**Key claims**: 87.1% LoCoMo, 89.8% LongMemEval (self-reported, vendor paper). "Single-query retrieval — no multi-stage pipelines, no graph schema, no rerankers." Sub-90ms retrieval, zero indexing delay.

---

## Architecture

### Storage & Schema
No local database. Memories are Moorcheh **documents** in a namespace `memanto_agent_{agent_id}` (`core.py::agent_namespace`). The unit is a `MemoryRecord` (pydantic, `core.py`): `id` (UUID/ULID), `type`, `title` (≤100), `content` (≤10000), `agent_id`, `actor_id`, `source`, `source_ref`, `confidence` (0-1, default 0.8), `status` (active/superseded/deleted/provisional), `tags[]`, `provenance`, `created_at`, `updated_at`, `expires_at`, `ttl_seconds`. `to_moorcheh_document()` renders the card as text `"[TYPE] title\n\ncontent\n\nTags: ..."` plus **flat** metadata fields so Moorcheh can filter with a `#key:value` query syntax.

### Memory Types
13 flat types (`constants.py::VALID_MEMORY_TYPES`): fact, preference, goal, decision, artifact, learning, event, instruction, relationship, context, observation, commitment, error. Plus a **provenance ladder** (`explicit_statement, inferred, corrected, validated, observed, imported`). No priority/decay_rate/theme-graph fields — this is a flat typed store, not a layered/graph schema like Somnigraph's.

### Write Path
`memory_write_service.py::store_memory` / `batch_store_memories` (≤100 docs/req). Extraction from chat is `conversation_memory_extraction_service.py`: it calls Moorcheh's own `answer.generate` with a header/footer prompt to emit a JSON array of `{type,title,content,confidence}`, then normalizes and **exact-dedupes** on `(type, whitespace-normalized-lowercased-content)`. That is the only dedup — no semantic/near-duplicate gating at write time. **Quality/validation gating is present in code but disabled**: both `store_memory` and `batch_store_memories` contain commented-out `validation_service.validate_memory(...)` blocks replaced by `validation_result = {"action": "store", "reason": "MVP direct store"}` with the comment `# skip validation for speed`. The `legacy/memory_validation_service.py` and `legacy/safe_deletion.py` machinery is not wired into the live path.

### Retrieval
A **single call** to the proprietary engine: `memory_read_service.py::search_memories` → `self.client.similarity_search.query(query, namespaces, top_k, threshold, kiosk_mode)`. `top_k` capped at 100 ("Moorcheh max"). Everything the repo adds is *around* that call:
- `_build_filtered_query` appends Moorcheh `#memory_type:x #status:active #confidence:high #tag` tokens to the query string (server-side metadata filtering).
- Temporal filters (`created_after/before`), TTL expiry (`_filter_expired_memories`, fail-open), and offset pagination are **client-side post-processing** in Python.
- **No BM25, no RRF fusion, no graph expansion, no reranker anywhere in the repo** (grep for `rerank|rrf|bm25|reciprocal|hebbian|ppr` returns only UI/visualization "graph" and a ULID comment). The "information-theoretic" ranking is entirely inside the closed Moorcheh `similarity_search`. The `answer` (RAG) path is likewise `client.answer.generate` — a Moorcheh-internal retrieve-then-read.

Temporal query modes worth naming (`memory_read_service.py`): `search_as_of` (point-in-time: what was true at T, honoring expiry), `search_changed_since` (differential: created/updated after T, tagged with `change_type`), `search_recent`. These fetch-all-then-filter in Python via `documents.fetch_text_data` cursor pagination.

### Consolidation / Processing
`daily_analysis_service.py` — an **offline, per-day** pass over session markdown files (`{agent}_{date}_*_summary.md`), not over the memory store directly:
- `generate_summary`: LLM (Moorcheh `answer.generate`) writes a natural-language daily summary markdown + ASCII timeline visualizations.
- `generate_conflict_report`: LLM classifies new-vs-historical interactions into `contradiction / update / duplicate / conflict`, each with a `recommendation ∈ {keep_new, keep_old, merge, remove_both}`, written to `~/.memanto/conflicts/{agent}_{date}_conflicts.json` for **interactive human resolution** (`memanto conflicts`). Resolution marks the loser `superseded`, never auto-merges. This is LLM-prompted and single-pass — no pairwise graph edge creation, no Hebbian/PMI, no taxonomy generation like Somnigraph's NREM/REM.

### Lifecycle Management
TTL via `expires_at`/`ttl_seconds`, enforced app-side (Moorcheh does not auto-delete; filter fails *open*). Updates are **delete-and-recreate** (`update_memory`) because Moorcheh has no in-place update. Conflict resolution sets `status=superseded` (retention over erasure). **No decay/half-life/reheat** — the flat `confidence` is set once at extraction and never re-scored.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| 87.1% LoCoMo, 89.8% LongMemEval, "SOTA", beats Mem0/Zep/Letta | Vendor paper (arXiv 2604.22085) + README; no eval harness in repo | **Unvalidated from this repo.** The scored retriever+reader is the *closed* Moorcheh engine; the open code cannot reproduce the number without Moorcheh. Judge model/methodology undisclosed. Category split (Open-Domain 92.4 / Temporal 85.4 / Multi-Hop 70.8) matches LoCoMo QA structure → likely end-to-end QA accuracy, but judge strictness unknown (cf. our Opus is 3.2pp stricter than GPT-4.1-mini). |
| "No rerankers, no graph, single-query retrieval" | Confirmed in code — retrieval is one `similarity_search.query` call | **True of the repo**, but it means the intelligence lives in a proprietary black box, not in anything auditable here. |
| Zero ingestion latency / immediately searchable | Property asserted of Moorcheh; `documents.upload` returns queued/success | **Plausible but unverifiable** — it is a Moorcheh backend property, not repo code. |
| Automated conflict detection & superseding | `daily_analysis_service.generate_conflict_report` (LLM) | **Real but shallow**: one LLM prompt over session MD files, human-in-the-loop resolution, not an automatic store-wide consolidation. |
| Typed memory (13 categories) + provenance | `constants.py`, `core.py` | **Validated** — cleanly implemented. |

---

## Relevance to Somnigraph

### What memanto does that Somnigraph doesn't
- **Point-in-time (`as_of`) and differential (`changed_since`) query modes** as first-class retrieval verbs (`memory_read_service.py`). Somnigraph has `valid_from`/`valid_until` in `db.py` but exposes no "what was true at T" or "what changed since T" tool in `tools.py`.
- **A typed provenance ladder** (`explicit_statement/inferred/corrected/validated/observed/imported`) distinct from `source`. Somnigraph tracks `source` strings but not a provenance *type*.
- **Human-in-the-loop conflict queue** with per-item `recommendation` (keep_new/keep_old/merge/remove_both). Somnigraph's `sleep_nrem.py` auto-classifies and applies; there is no review queue.
- **Multi-agent namespace isolation** (`memanto_agent_{id}`). Somnigraph is single-user by design.
- Broad **integration surface** (Claude Code lifecycle hooks, CrewAI, LangGraph store/saver, MCP). Not an algorithmic idea, but a distribution model.

### What Somnigraph does better
- **Everything about retrieval quality is auditable and owned.** `reranker.py` (26-feature learned LightGBM, NDCG=0.7958), `scoring.py` (RRF k=14 + PPR graph expansion), `fts.py` (BM25 hybrid) are all in-repo and measured. Memanto delegates 100% of ranking to a closed engine — it has no reranker, no fusion, no feedback loop, and cannot tune retrieval.
- **Feedback loop with measured GT correlation** (Spearman r=0.70) and Hebbian co-retrieval edges. Memanto has no feedback signal at all; `confidence` is write-once.
- **Real consolidation**: `sleep_nrem.py`/`sleep_rem.py` build typed graph edges, merge/archive, gap-analysis and taxonomy across the whole store. Memanto's "consolidation" is a per-day LLM summary/conflict report over markdown files.
- **Decay/lifecycle**: per-category half-lives + reheat vs memanto's write-once confidence + TTL.
- **Write-path quality gating** — even though memanto *has* a validation service, it is commented out for speed; Somnigraph's write path is the live product.

---

## Worth Stealing (ranked)

### 1. Temporal query verbs: `as_of` and `changed_since` (Low)
**What**: Two explicit retrieval modes — point-in-time ("what did I believe about X at date T", honoring `valid_until`/expiry) and differential ("what memories were created/updated since T", each tagged `change_type: created|updated`).
**Why**: Somnigraph already stores `valid_from`/`valid_until` and `updated_at` but never exposes temporal navigation. A "what changed since last session" recall is a natural fit for a cross-session agent memory and cheap given the columns already exist.
**How**: Add two thin `impl_recall_as_of` / `impl_recall_changed_since` variants in `tools.py` that filter the candidate set by `valid_from/valid_until` (as_of) or `created_at/updated_at > T` (changed_since) before the normal rerank — pure SQL predicates on existing columns, no new schema.

### 2. Typed provenance ladder (Low)
**What**: A small enum on each memory recording *how it entered the store* — `explicit_statement / inferred / corrected / validated / observed / imported` — separate from `source`.
**Why**: Somnigraph's write path already distinguishes user-stated vs auto-captured-pending vs correction, but only via `source` strings and `status`. A first-class provenance type is a clean feature for the reranker (an `inferred` memory could be trusted less than a `corrected` one) and for honest-accounting audits.
**How**: Add a `provenance` column in `db.py`; populate it in `remember()` from the existing `source` argument; optionally expose it as a reranker feature in `reranker.py`. Additive, reversible.

---

## Not Useful For Us

### The Moorcheh engine itself / "information-theoretic indexless retrieval"
The entire retrieval and RAG-answer intelligence is a closed backend (cloud API or a `moorcheh` Docker/pip package the repo imports but does not contain). There is no algorithm to study or port — it is packaging over a proprietary engine. Somnigraph's whole thesis is the opposite: own and measure the retrieval stack.

### `#key:value` in-query metadata filter syntax
A Moorcheh-specific query-string convention (`"auth #memory_type:fact #confidence:high"`). Somnigraph filters in SQL/FTS directly; embedding filters in the query string is a workaround for a backend that only takes a text query.

### Multi-agent namespaces, integration adapters, delete-and-recreate updates
Multi-tenant isolation and the CrewAI/LangGraph adapters are irrelevant to a single-user MCP system. Delete-and-recreate updates are a workaround for Moorcheh's lack of in-place update; Somnigraph's SQLite updates in place.

---

## Connections

- **Corroborates the Phase 18 write-path finding** (`ai-memory-comparison.md`, ByteRover/agentmemory sweep): the LoCoMo/LME leaders win on *write-time structure and a strong closed retriever*, not on an elaborate open retrieval pipeline. Memanto is the extreme case — a flat typed store whose entire retrieval quality is outsourced to one vendor engine, still claiming SOTA. The claim is unauditable from the repo, which is exactly the "packaging over an upstream engine" pattern to flag.
- **Conflict-resolution taxonomy** (contradiction/update/duplicate/conflict + keep_new/keep_old/merge) rhymes with the contradiction/evolves/revision edge types in `sleep_nrem.py` and with the supersession patterns noted in `memv`/`memos` analyses — independent convergence on "supersede, don't delete." Memanto's differentiator is the *human review queue*; Somnigraph auto-applies.
- **Temporal `as_of` retrieval** echoes bitemporal designs seen in Zep/Graphiti-style analyses — another independent vote that "valid-time" navigation is worth exposing, which Somnigraph stores but hasn't surfaced.

---

## Summary Assessment

Memanto's real contribution is a **clean, typed memory-management wrapper and a broad integration surface** (Claude Code hooks, CrewAI, LangGraph, MCP) around Moorcheh, a proprietary "information-theoretic" search engine. The open-source repo is genuinely tidy — 13 typed categories, a provenance ladder, namespace isolation, TTL, temporal query modes, and an LLM-driven daily summary/conflict report. But the part that would matter for a retrieval-quality research artifact — the ranking — is **not in the repo at all**. Retrieval is a single `similarity_search.query` call and answering is a single `answer.generate` call, both into a closed backend. There is no reranker, no fusion, no feedback loop, no graph, no decay in the code.

For Somnigraph the single sharpest takeaway is a **cross-check discipline point, not an algorithm**: the headline 87.1 LoCoMo / 89.8 LME numbers are vendor-paper, undisclosed-judge, and irreproducible from this repository because the scored retriever+reader is closed. Treat them as non-comparable to our 85.1 Opus-judged LoCoMo QA until the judge and the engine are pinned. The two genuinely portable ideas are small and additive: expose the **temporal query verbs** (`as_of`, `changed_since`) that our schema already supports but never surfaces, and add a **typed provenance field** that could feed the reranker. Neither changes the architecture; both are low-effort features that Somnigraph's data model is already 90% of the way to.

What is overhyped: "SOTA with no reranker, no graph, single-query retrieval" reads as an architectural insight but is really "we outsourced ranking to a closed engine." What is missing versus us: any owned, measured, tunable retrieval path, and any write-time quality gating (the validation service exists in `legacy/` but is commented out with `# skip validation for speed`).
