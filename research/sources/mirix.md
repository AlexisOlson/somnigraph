# MIRIX — Multi-agent 6-type memory system with per-type LLM extraction and parallel retrieval

*Generated 2026-06-28 by Sonnet agent reading local clone + paper*

---

## Paper Overview

**Paper**: arXiv:2507.07957, "MIRIX: Multi-Agent Memory System for LLM-Based Agents", July 10, 2025, 10 pages
**Authors**: Yu Wang, Xi Chen (MIRIX AI)
**Code**: https://github.com/Mirix-AI/MIRIX — 3,544 stars, Apache 2.0, ~14,650 Python LOC, last commit "add auto-dream" (51f3342)

**Problem addressed**: Existing memory systems use flat stores (Mem0, Letta/MemGPT) that conflate qualitatively different memory types, or knowledge graphs (Zep/Graphiti) that cannot represent sequential events, emotional states, or multi-modal inputs. Text-centric systems fail on visual/multi-modal inputs. The paper claims routing and retrieval are the critical capabilities, and that separate specialized stores enable both.

**Core approach**: Six typed memory stores (Core, Episodic, Semantic, Procedural, Resource, Knowledge Vault), each managed by a dedicated LLM sub-agent. A MetaMemoryAgent routes incoming content to relevant sub-agents in parallel. Retrieval is per-type and automatic (topic-generation → top-10 from each store), injected into the system prompt with typed XML tags. Includes an AutoDream offline reflection agent and a Reflexion agent.

**Key claims**:
- 85.38% LoCoMo QA (three-run average on gpt-4.1-mini reader; beats LangMem 77.05%, Zep 75.14%)
- 59.50% ScreenshotVQA accuracy vs Gemini 11.66% / SigLIP@50 44.10%, at 99.9% storage reduction
- Multi-hop outperforms full-context (83.70% vs 77.70%) due to evidence consolidation

---

## Architecture

### Storage & Schema

Primary store: **PostgreSQL + pgvector** (SQLite fallback). Six separate ORM tables (`mirix/orm/`), each with its own schema, plus Redis caching. Each memory type stores dual embeddings on the main text fields (summary and details separately). All tables include `filter_tags` (JSONB), `agent_id`, and `client_id` for multi-user/multi-client isolation.

No unified `memories` table — type-specific fields are first-class, not nullable extras. This is the architectural bet: better schemas per type at the cost of no cross-type fusion.

### Memory Types

| Type | ORM table | Key fields | Somnigraph equivalent |
|------|-----------|------------|----------------------|
| **Core** | `block` (Letta-inherited) | `label` (persona/human), `value`, character limit, rewrite at >90% | Partial: `meta` category covers agent profile; no human-block |
| **Episodic** | `episodic_memory` | `occurred_at`, `actor`, `event_type`, `summary`, `details`, `last_modify` | `episodic` category |
| **Semantic** | `semantic_memory` | `name`, `summary`, `details`, `source`, `last_modify` | `semantic` category |
| **Procedural** | `procedural_memory` | `entry_type` (workflow/guide/script), `description`, `steps` (JSON) | `procedural` category |
| **Resource** | `resource_memory` | `title`, `summary`, `resource_type` (doc/markdown/pdf_text/image/voice_transcript), full content | **No equivalent** |
| **Knowledge Vault** | `knowledge_vault` | `entry_type` (credential/bookmark/contact_info/api_key), `sensitivity_level`, `secret_value` | **No equivalent** |

The split between Resource and Semantic is meaningful: Semantic stores extracted facts about concepts; Resource stores the source documents/transcripts themselves — an input-vs-derived distinction Somnigraph collapses into a single `memories` table.

Knowledge Vault is a privacy-first compartment for credentials and sensitive values. It is the only type where `secret_value` is a first-class field, and the paper envisions per-type encryption with user-controlled sharing.

The Core memory type inherits Letta/MemGPT's block architecture: two fixed blocks (`persona`, `human`) with character limits. When the human block exceeds 90% capacity, a rewrite is triggered (`core_memory_rewrite` in `mirix/functions/function_sets/memory_tools.py`). This is the only memory type with an explicit size management policy.

Somnigraph's `reflection` and `meta` categories have no MIRIX counterparts as storage types. MIRIX handles reflection via agents (AutoDream, Reflexion) rather than as a category of stored memory.

### Write Path

```
User input → MetaAgent.step() → MetaMemoryAgent
           → parallel dispatch to up to 6 sub-agents
           → each sub-agent calls memory tools (episodic_create, semantic_upsert, etc.)
           → Managers (mirix/services/) persist via SQLAlchemy ORM → PostgreSQL
```

Kafka queue (`mirix/queue/`) supports async write (`async_add=True`). The eval harness uses `chaining=False` (`evals/mirix_memory_system.py:83`) to skip the full multi-agent chain per session chunk. Two separate LLMs are configured: a `topic_extraction_llm_config` (gpt-4.1-nano in eval config `evals/configs/0201c.yaml`) and the main `llm_config` (gpt-4.1-mini) — a cost-tiered split the paper describes as "streaming upload" that reduces write latency from ~50s to <5s.

At write time, sub-agents perform LLM-mediated extraction and deduplication: the episodic agent decides what constitutes an event worth recording; the semantic agent decides whether to create a new entry or update an existing one ("conceptually overwritten" is the paper's term). No content-hash deduplication — the LLM is the quality gate.

### Retrieval

Two-phase automatic retrieval:
1. **Topic generation**: A lightweight LLM (gpt-4.1-nano at eval time) generates `current_topic` from the query.
2. **Per-type search**: Each memory manager runs its own search in parallel, returning top-10 results. Methods (from `mirix/services/episodic_memory_manager.py`):
   - `embedding`: vector cosine via pgvector's `<->` operator (L2 distance)
   - `bm25`: PostgreSQL native `to_tsvector`/`ts_rank_cd` with GIN indexes; falls back to in-memory `rank_bm25` (PyPI `rank-bm25`) for SQLite
   - `string_match`: SQL `LIKE`/`ILIKE` containment
   - `fuzzy_match`: `rapidfuzz` (legacy, kept for backward compat)

No RRF fusion across types. No cross-type reranker. Results are injected into the system prompt as typed XML blocks (`<episodic_memory>…</episodic_memory>`). The consuming LLM sees all retrieved memories and decides relevance. Redis caches individual memory objects but not retrieval results.

The eval wrapper (`evals/mirix_memory_system.py:wrap_user_prompt`) retrieves from all types and formats them into a single `<episodic_memory>` block for simplicity — type tags are flattened at eval time.

### Consolidation / Processing

**AutoDream** (`mirix/agent/auto_dream_agent.py`, `mirix/services/auto_dream_manager.py`): An offline reflection agent that operates on a time window since the last checkpoint. It fetches memories of selected types (configurable: core/episodic/semantic/resource/procedural/knowledge/experience), runs a single LLM pass over them, and writes a checkpoint event back to episodic memory. This is MIRIX's closest analog to Somnigraph's sleep — but it is a single-pass LLM call over raw memories, not a structured pairwise classification, edge-building, or gap-analysis pipeline.

**Reflexion agent**: A self-reflection agent. Exists as a named agent type in `meta_agent.py` but the implementation is a thin wrapper around the base Agent class (`reflexion_agent.py`). System prompt drives behavior.

Core memory rewrite: triggered at >90% capacity, reduces the block to ~50% by LLM synthesis.

No batch clustering. No edge detection. No taxonomy evolution. No question generation.

### Lifecycle Management

No decay. No archival. No versioning. Memories are created, updated in place (semantic type), or deleted. The `last_modify` JSON field on each table records the most recent operation and timestamp but is not used for scoring or ranking — it is audit metadata.

A cleanup job (`mirix/jobs/cleanup_raw_memories.py`) handles raw input buffers (screenshots, transcripts before extraction), not the typed memory stores themselves.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| LoCoMo 85.38% (overall) | Three-run average (83.98, 87.34, 84.82) with gpt-4.1-mini reader/judge | **Plausible but not directly comparable to Somnigraph's 85.1% Opus-judged result**. Somnigraph's own calibration shows Opus is ~3.2pp stricter than gpt-4.1-mini. Applying that correction, MIRIX's Opus-equivalent score may be ~82%. Also: different LoCoMo GT versions, different reader models, different retrieval setups. |
| Multi-hop 83.70% vs full-context 77.70% | Paper Table 2 breakdown | **Plausible**. The structural argument is sound: explicit consolidation of related events into single episodic entries collapses multi-hop into single-hop retrieval. Somnigraph's Level 5b achieves 88.8% multi-hop R@10 via vocabulary bridges and graph expansion. |
| ScreenshotVQA 59.50% vs SigLIP@50 44.10%, 99.9% storage reduction | Novel benchmark, 3 PhD students, 87 questions | **Unvalidated at scale**. Novel benchmark, no community replication. The storage claim (images abstracted to text) is definitionally true. The accuracy gain is plausible given abstraction quality. |
| Beats LangMem (77.05%), Mem0, Zep (75.14%) | Paper Table 2 | **Plausible**, but these comparisons may use different judge configurations. No evidence all baselines used identical judge prompts. |
| "Routing and Retrieving are the key capabilities" | Core architectural claim, no ablation of the routing mechanism itself | **Unvalidated**. No ablation comparing multi-agent routing vs single-agent with same 6 stores. |

---

## Relevance to Somnigraph

### What MIRIX does that Somnigraph doesn't

**Separate-table taxonomy with type-specific schemas** (`mirix/orm/`): Each of the 6 memory types has its own table with fields appropriate to that type — `occurred_at` for episodic, `entry_type`+`steps` for procedural, `sensitivity_level`+`secret_value` for knowledge vault. Somnigraph uses a single `memories` table with a `category` flag. The MIRIX approach avoids nullable-column sprawl and enables per-type indexing strategies. The gap exists in `db.py`.

**Resource memory** (document/file store): No equivalent in Somnigraph. A user who pastes a PDF, transcript, or code file into their assistant has no structured place to store it in Somnigraph beyond a generic memory. MIRIX treats documents as a first-class memory type with `resource_type` classification and full content storage. The gap is in `db.py` and `tools.py`.

**Knowledge Vault** (sensitivity-gated compartment): Credentials, API keys, contact info, and bookmarks stored with `sensitivity_level` and isolated from semantic/episodic search by default. Somnigraph has no privacy-tier memory. The gap is in `db.py`.

**LLM-mediated extraction at write time**: MIRIX sub-agents decide what to extract and where to store it. Somnigraph relies on the user/agent explicitly calling `remember()` with an explicit `category`. MIRIX's approach scales better to passive ingestion (screenshots, long conversations); Somnigraph's approach produces higher-signal memories because the caller decides what matters. Neither is strictly better.

**Multi-user organizational isolation**: `organization_id` on every table, per-client scoping via `filter_tags`. Somnigraph is single-user by design.

### What Somnigraph does better

**Retrieval quality infrastructure**: Somnigraph's RRF fusion (BM25 + vector + theme channel) with a 31-feature LightGBM reranker is absent in MIRIX. MIRIX retrieves top-10 per type independently with no cross-type merging or learned ranking. The retrieval ceiling for MIRIX is fundamentally lower — a memory that scores well semantically but not via BM25 may be missed, and there is no mechanism to promote memories that have historically been useful.

**Explicit feedback loop**: Somnigraph's `recall_feedback()` provides per-query utility ratings that compound through EWMA, UCB exploration, and Hebbian PMI edge strengthening. MIRIX has no feedback mechanism. Retrieval quality cannot improve from usage.

**Sleep pipeline** (`scripts/sleep_nrem.py`, `scripts/sleep_rem.py`): Somnigraph's NREM detects pairwise relationships (supports/contradicts/evolves/revision/derivation), creates typed edges, and merges redundant memories via LLM classification. REM runs gap analysis and generates questions. MIRIX's AutoDream is a single-pass reflection per time window — no pairwise analysis, no edge creation, no gap-identification loop.

**Graph-conditioned retrieval** (`scoring.py`, `reranker.py`): PPR expansion via sleep-built typed edges gives Somnigraph multi-hop reach within the retrieval step itself. MIRIX has no graph; it achieves multi-hop coverage by consolidating events at write time (collapsing multi-hop into single-hop). This is a write-time vs read-time tradeoff: MIRIX's approach is irreversible (consolidation loses detail), Somnigraph's is lossless.

**Per-category decay** (`db.py`): Somnigraph's exponential decay with per-category half-lives and reheat-on-access lets old memories fade while recently-accessed ones persist. MIRIX retains everything permanently. At scale, MIRIX retrieval quality degrades as the episodic store fills with stale events.

**Benchmark methodology transparency**: Somnigraph documents corrected GT (6.4% ceiling correction from locomo-audit), judge model identity (Opus), three-run stability checks, and multi-hop failure analysis. MIRIX's paper does not document the judge model clearly; the eval config uses gpt-4.1-mini as reader and the paper doesn't specify whether the judge is the same model.

---

## Worth Stealing (ranked)

### 1. Resource memory type (Medium effort)

**What**: A dedicated storage tier for source documents — PDFs, transcripts, code files, markdown — with `resource_type` classification, `title`, `summary`, and full content. Separate from semantic memory (extracted facts) and episodic memory (events).

**Why**: Somnigraph has no structured place for documents. When a user pastes a long text, it either becomes a single large memory (noisy for retrieval) or gets summarized (loses retrievability of the original). A resource store with its own retrieval path (BM25 on full content, embedding on summary) fills this gap cleanly.

**How**: New `resource` category in `db.py` with `resource_type` metadata field. New `resource_memory_search()` in `tools.py`. Sleep can reference resources when building semantic abstractions. Modest schema change; retrieval integration is the main work.

### 2. Write-path LLM routing for passive ingestion (High effort)

**What**: A meta-agent that classifies incoming content and dispatches to type-specific sub-agents in parallel, without requiring the user to specify category.

**Why**: Somnigraph's `remember()` requires the caller to choose category, priority, and themes. For passive ingestion scenarios (long conversation dumps, document uploads) this is impractical. MIRIX's routing is fully automatic.

**How**: Would require an orchestration layer above the current MCP tools — a `remember_auto(content)` tool that calls a lightweight LLM to determine category and attributes before passing to the existing `remember()`. The multi-agent parallelism is MIRIX-specific complexity; the routing classification alone is simpler and more useful.

### 3. Separate-table per type (High effort, low priority)

**What**: Each memory category gets its own table with type-appropriate fields rather than a single `memories` table with a `category` column and many NULLs.

**Why**: Somnigraph's single-table design works well at current scale but accumulates nullable-column debt as types gain unique fields (e.g., `steps` for procedural, `resource_type` for documents). Type-specific FTS indexes are also cleaner per-table.

**How**: Requires db.py rewrite, migration tooling, and updates to every query in scoring.py, fts.py, tools.py. High disruption. Worth considering if the taxonomy stabilizes and type-specific retrieval strategies diverge further.

### 4. Knowledge Vault sensitivity tier (Low effort)

**What**: A `sensitivity_level` field (low/medium/high) plus a `is_sensitive` boolean that gates inclusion in default retrieval. Sensitive memories require explicit opt-in to surface.

**Why**: Somnigraph currently has no privacy tier. A memory containing an API key or personal credential is retrievable by any `recall()` call. A `sensitive=true` flag in `db.py` that suppresses these from default recall (but not `recall(include_sensitive=True)`) would be a low-cost safety improvement.

**How**: Add `sensitive` boolean to `db.py` schema. Filter in `fts.py` and `scoring.py` by default. Expose `include_sensitive` parameter in `tools.py`'s `recall()`. One migration, two filter additions.

---

## Not Useful For Us

### PostgreSQL + Kafka + Redis deployment

MIRIX's production stack (PostgreSQL with pgvector + pgBM25/tsvector, Kafka queue, Redis cache, FastAPI REST server) is engineered for multi-user cloud deployment. Somnigraph is single-user, MCP-stdio, SQLite. The infrastructure complexity does not transfer.

### Multi-user organizational isolation

`organization_id` + `client_id` + `filter_tags` scoping on every table. Somnigraph has one user, one context, one database. This is organizational overhead with no benefit for our use case.

### Multi-agent parallel extraction with full LLM calls per sub-agent

MIRIX routes every input through 6 sub-agents, each making an LLM call. At eval, this is run with `chaining=False` (skipping full multi-agent chaining) to be tractable. For real-time use with Somnigraph's write-at-decision-time model, this would be prohibitively expensive. Our `remember()` is a single explicit tool call — appropriate cost for explicit capture.

### AutoDream as a sleep replacement

AutoDream's single-pass reflection over recent memories is too shallow to replace Somnigraph's NREM pipeline. It produces no typed edges, no pairwise contradiction detection, no taxonomy evolution, no gap questions. It is useful as a "did anything important happen?" summary, not as structural consolidation.

---

## Connections

**msam.md** (4-stream cognitive memory): MSAM uses working/episodic/semantic/procedural — 4 streams with ACT-R activation, no resource or vault. MIRIX's 6-type split adds resource and knowledge_vault, driven by multimodal ingestion requirements rather than cognitive science theory. Both use per-type managers, but MSAM fuses retrieval via activation scoring while MIRIX does not fuse.

**cogmem.md** (3-layer Oberauer): CogMem's DA/FoA/LTM is a retrieval-and-context-construction framework, not a taxonomy. MIRIX's 6 types live at the LTM layer in CogMem's vocabulary. Both systems treat retrieval as automatic (MIRIX's topic-generation → retrieve, CogMem's FoA reconstruction). Both lack feedback loops.

**pltm-claude.md** (4-type cognitive memory with jury): PLTM's 4-type taxonomy (episodic/semantic/belief/procedural) is closer to the Tulving tradition that MIRIX departs from. MIRIX replaces belief with knowledge_vault (operational rather than epistemic) and adds resource (practical rather than cognitive). The belief/procedural split in PLTM has no counterpart in MIRIX; MIRIX's procedural is narrower (how-to guides, not all rule-based knowledge). Both PLTM and MIRIX use LLM-mediated write-path extraction.

**mengram.md** (Tulving types): Mengram's episodic/semantic split maps directly to MIRIX's, but Mengram adds a knowledge-graph layer (wikilinks, triples) that MIRIX does not. MIRIX's consolidation via AutoDream is lighter than Mengram's graph-building. Both face the absence-of-graph problem for multi-hop retrieval (MIRIX compensates via write-time consolidation; Mengram via explicit KG traversal).

---

## Summary Assessment

MIRIX's core architectural bet is that typed stores with LLM-mediated extraction produce better memories than a flat store with post-hoc classification. The 6-type split is partly principled (episodic vs semantic is Tulving; core blocks inherit Letta's persona/human distinction) and partly pragmatic (resource and knowledge_vault address real limitations in text-centric systems, not cognitive theory). The type-specific schemas are genuine value — `occurred_at` on episodic, `steps` on procedural, `sensitivity_level` on knowledge_vault are not achievable without per-type tables. The multi-agent parallel extraction is more interesting as a design claim than as a performance claim: whether 6 LLM sub-agents actually route better than one well-prompted agent extracting to a categorized store is not ablated.

The retrieval architecture is the weakest part. No RRF fusion, no cross-type reranker, no feedback loop, no decay means MIRIX's retrieval quality is static and cannot improve with use. Top-10 per type injected into system prompt hands the ranking problem to the reading LLM, which is a valid approach but not a principled one. The AutoDream consolidation partially compensates by collapsing dispersed episodic events into single entries, which is why multi-hop improves — but this trades recall precision for retrieval simplicity, and the consolidation is irreversible.

The LoCoMo 85.38% comparison to Somnigraph's 85.1% is the most interesting number in the paper, but it is not a direct comparison. MIRIX uses gpt-4.1-mini as reader and (implicitly) as judge; Somnigraph uses Opus judge, which is ~3.2pp stricter by its own calibration. Adjusting for judge severity, MIRIX's Opus-equivalent score is approximately 82%, placing it meaningfully below Somnigraph. The multi-hop result (83.70%) is the genuine strength and the most relevant result for Somnigraph's roadmap: MIRIX achieves it via write-time consolidation, Somnigraph via graph expansion — two different architectures reaching the same sub-problem from opposite directions.

---

*Clone path*: `C:\Users\Alexis\AppData\Local\Temp\claude\C--Users-Alexis\5a0237ef-6345-4355-8cd2-d6ecf6e4044b\scratchpad\repos\mirix`
