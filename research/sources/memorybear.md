# MemoryBear — Graph-first (Neo4j) production memory platform with a real but read-decoupled ACT-R decay engine

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

MemoryBear (RedBear AI, Apache-2.0) is a large FastAPI monolith — ~980 Python files under `api/app/` — that packages a full multi-tenant memory *product*: knowledge graph, multi-agent orchestration, emotion analytics, ontology/OWL export, React frontend, MCP + REST. Only the memory core (`api/app/core/memory/`) is architecturally relevant; most of the repo is application plumbing.

### Storage & Schema
Four backing stores (Docker Compose): **PostgreSQL** (config/users/tenancy), **Neo4j 4.4+** (the knowledge graph — primary memory store), **Elasticsearch 8.x** (BM25 + dense vector), **Redis** (cache/queue). Memory units are Neo4j nodes: `Statement`, `ExtractedEntity`, `MemorySummary`, `Community`, `Chunk`, `Dialogue`. Node schema (`core/memory/models/graph_models.py`) carries `activation_value: Optional[float]`, `access_history: List[str]` (ISO timestamps), `access_count`, `last_access_time`, `importance_score`, `created_at`, plus statement triple fields (`entity1_name`, `predicate`, `entity2_name`, `statement`). This is far richer than a single vector row — it is a genuine graph.

### Memory Types
Explicit (`memory_explicit_schema.py`: episodic 情景 / semantic 语义), implicit (`implicit_memory_schema.py`: preferences, 4-dimension personality portrait, interest areas, behavior habits with frequency patterns), perceptual, short-term, and emotional memory. The implicit/persona layer is the most differentiated part — it is a structured user-profile builder, directly PERMA-relevant territory Somnigraph does not model.

### Write Path
`pipelines/write_pipeline.py`: preprocess → **prune** (LLM memory-value gate: each candidate gets `assistant_memory_hint` or `"NULL"` = no memory value, dropped) → chunk (8 chunker strategies) → **LLM triple extraction** (`extraction_engine/`) → graph build (idempotent Neo4j MERGE) → **LLM blockwise dedup + disambiguation** (`enable_llm_dedup_blockwise`, `enable_llm_disambiguation`, default on) → store to Neo4j + ES → **clustering** (community detection / label propagation, `clustering_engine/label_propagation.py`). Real write-time entity resolution and a real quality gate — both things Somnigraph's `tools.py` write path lacks.

### Retrieval
`core/memory/src/search.py::rerank_with_activation` (the actual scorer, 294-line `search_service.py` wraps it; a LangGraph agent in `core/memory/agent/` adds optional query-breakdown/expansion/verification on top). **Not RRF.** Two-stage:
- **Stage 1 (content relevance):** normalize BM25 and embedding scores (min-max, or z-score+sigmoid for outliers), then `content_score = alpha*bm25 + (1-alpha)*embedding` (α default 0.6 in the fn; `search_service` passes `rerank_alpha=0.4` — an inconsistency). Keep top `limit*3` candidates.
- **Stage 2 (activation reorder):** nodes with an `activation_value` are re-sorted by that ACT-R activation *alone* (content score becomes a pure gate, discarded from the final ranking); nodes without activation keep content order and only backfill if stage 2 is under `limit`. Optional `forgetting_weight` multiplier from the ACT-R engine.
Community hits are expanded to member statements (`expand_communities_to_statements`) — graph expansion at read time. Linear weighted fusion + a heuristic activation reorder; **no learned reranker, no feedback signal.**

### Consolidation / Processing
Three offline/scheduled engines, analogous to Somnigraph's sleep but split by concern: **reflection** (`reflection_engine/self_reflexion.py` — LLM conflict/contradiction detection, resolution into versioned memories, quality assessment 0-100, periodic per-user via `reflection_time`), **clustering** (community detection), **forgetting cycle** (`memory_forget_service.trigger_forgetting_cycle` → `forgetting_scheduler.run_forgetting_cycle`: decay, node merge, dormancy → clearance). Triggered by API/scheduler, not a single unified pass.

### Lifecycle Management
Supersession via `ResolvedSchema` (`original_memory_id` + `resolved_memory` + `ChangeRecordSchema` field-level old/new diffs). Contradiction flagging (`ConflictResultSchema`, `ReflexionSchema`). Three-stage forgetting: dormancy → decay → clearance. Configurable forgetting params (`decay_constant`, `lambda_time`, `lambda_mem`, `offset`, `forgetting_threshold`, `min_days_since_access`).

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| ACT-R activation drives forgetting (recency + frequency + importance) | `actr_calculator.py`: full `R(i)=offset+(1-offset)*exp(-λt/Σ(I·t_k^(-d)))` implemented and wired into forgetting scheduler + Stage-2 rank read | **Realized as offline decay** — the math is real and runs in the batch cycle |
| Activation is reinforced by retrieval (frequency term) | `graph_search.py::_update_search_results_activation` is **stubbed** (`return results`; body commented out) at all 9 search entry points | **Not realized on read path** — retrieval never records access, so the frequency term is fed only at write/scheduler time, not by actual recall usage |
| "Retrieval accuracy 92%, +35% over single-mode" | README assertion, no reproducible harness in repo | Unvalidated marketing |
| LoCoMo: vector 72.90%, graph 75.00% overall (LLM-as-Judge) | README benchmark images; `redbear-mem-benchmark` submodule (not vendored in clone) | Plausible; **end-to-end QA, comparable scale to our 85.1 — and below it** |
| Hybrid = "semantic expands, keyword filters" | `rerank_with_activation` does linear α-weighted fusion, not sequential filter | README over-describes; code is weighted sum |

---

## Relevance to Somnigraph

### What MemoryBear does that Somnigraph doesn't
- **Write-time graph construction + entity resolution**: LLM triple extraction, idempotent MERGE, blockwise dedup/disambiguation at write. Somnigraph builds edges only during sleep (`sleep_nrem.py`) and has no entity resolution.
- **Write-path quality gate**: prune-to-`NULL` drops no-value candidates before storage. Somnigraph's `tools.py::remember` stores whatever it is handed — the exact gap Phase 18 flagged as the real lever.
- **Structured persona / implicit-memory layer** (preferences, personality dimensions, behavior-habit frequency): directly the PERMA preference-tracking surface Somnigraph has no module for.
- **Field-level supersession diffs** (`ChangeRecordSchema`) — finer than Somnigraph's `valid_until` invalidation.
- **Community-detection clustering** as a first-class store layer; multi-tenant; emotion analytics.

### What Somnigraph does better
- **Learned reranker** (`reranker.py`, 26-feature LightGBM, NDCG 0.7958). MemoryBear's ranking is a hand-weighted linear fusion plus a heuristic activation reorder — no learned model, no training data.
- **Explicit feedback loop with measured GT correlation** (Spearman r=0.70). MemoryBear has *no* per-query utility feedback; worse, its one usage-reinforcement path (read-time access recording) is stubbed out.
- **Principled fusion**: RRF (k=14, Bayesian-tuned) vs an α whose two call sites disagree (0.6 vs 0.4).
- **Higher end-to-end QA**: 85.1 LoCoMo J vs MemoryBear's self-reported 75.00 (graph) / 72.90 (vector) on the same LLM-judge scale.
- **Working reheat-on-access**: Somnigraph's decay actually reheats on retrieval; MemoryBear's equivalent loop is disabled.

---

## Worth Stealing (ranked)

### 1. ACT-R base-level activation as a spacing-aware decay variant (Low/Medium — consider)
**What**: `R(i) = offset + (1-offset)·exp(-λt / Σ_k I·t_k^(-d))` — a closed-form that folds *recency* (t since last access), *frequency + spacing* (power-law sum over the full access-history), and *importance* (I) into one score, with a floor (`offset`). See `actr_calculator.py:66-128`.
**Why**: Somnigraph's decay (`db.py`/`scoring.py`) is per-category exponential with a single reheat-on-access. That captures recency but not spacing — two accesses a week apart should strengthen a memory more than two in one minute, and the ACT-R power-law sum encodes exactly that. It is a principled, single-parameter (`d≈0.5`) upgrade to the reheat rule.
**How**: Somnigraph already logs retrieval events; store a bounded `access_history` per memory and replace the scalar reheat in the decay update with the Σ term. Prototype offline against existing feedback logs before touching live decay. Additive to, not replacing, the learned reranker (which stays the ranker; this only shapes the decay/priority input feature).

*Everything else (graph-first storage, multi-agent, emotion, ontology export) is product surface, not a mechanism Somnigraph should absorb.*

---

## Not Useful For Us

### The whole application stack (multi-tenant, React/AntV frontend, emotion analytics, OWL export, multi-agent orchestration)
Somnigraph is deliberately single-user, MCP, one SQLite file. A four-service Docker deployment is the opposite design point.

### Neo4j graph-first storage
Somnigraph's PPR-over-sqlite-vec covers graph retrieval without a graph DB dependency; adopting Neo4j would be a strict complexity regression for a single-user tool.

### Their retrieval scorer
Hand-weighted linear fusion + heuristic activation reorder is strictly behind Somnigraph's RRF + LightGBM reranker; nothing to port.

---

## Connections

- **Write-path gate corroboration**: the prune-to-`NULL` memory-value classifier is independent evidence for the Phase 18 source-sweep conclusion (ByteRover/MemPalace/agentmemory) that *write-path discipline*, not retrieval cleverness, is what the LoCoMo/LME leaders win on. Same lesson from a fourth, unrelated system.
- **Supersession/versioning**: `ResolvedSchema` field-diffs converge with the supersession pattern noted in `memv` / MIRIX analyses.
- **ACT-R decay**: shares the biological-forgetting framing with Somnigraph's own decay and with any Ebbinghaus-curve systems in the corpus, but is the most fully-specified activation formula seen so far.
- **Emotion/affective memory** (A-MBER benchmark, `emotion_schema.py`) is a dimension no other profiled system in the corpus models.

---

## Summary Assessment

MemoryBear is a large, seriously-engineered *product* — graph-first (Neo4j), multi-tenant, with real write-time triple extraction, entity dedup, community clustering, versioned supersession, and a genuine ACT-R forgetting engine. Its core contribution relative to Somnigraph is on the **write and lifecycle** side: it does at ingestion (extract → gate → resolve → cluster) what Somnigraph defers to sleep or skips entirely, and its prune-to-NULL quality gate is a clean, small, adoptable pattern that corroborates our standing hypothesis that write-path discipline is the real lever.

The single sharpest code-level correction — and the one worth carrying forward — is on decay. The prior nugget-mining note that "ACT-R spacing decay is unrealized" is **too strong**: the activation formula (`actr_calculator.py`) is fully implemented and wired into both the offline forgetting scheduler and the Stage-2 read ranking. What is actually unrealized is narrower and more interesting: `_update_search_results_activation` is stubbed to `return results` at every one of the nine search entry points, so **retrieval never records an access**. The ACT-R "frequency" term therefore only ever sees write-time and scheduler-time events, not real recall usage — the reinforcement loop that would make popular memories resist forgetting is silently disabled. It is the exact assert-the-mechanism-exists-but-the-wire-is-cut failure worth flagging: the paper-grade formula is there, the plumbing that would give it live signal is commented out.

On retrieval and ranking Somnigraph is ahead: MemoryBear has no learned reranker, no explicit feedback loop, an α that disagrees between its two call sites, and self-reported LoCoMo J (75.0 graph / 72.9 vector) below Somnigraph's 85.1 on the same judge scale. Net verdict: **MAYBE** — one low-confidence, additive idea (ACT-R spacing as a decay refinement) and a useful confirmation of the write-path-gate direction; nothing that touches the reranker/feedback core.
