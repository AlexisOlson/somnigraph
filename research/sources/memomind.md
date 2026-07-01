# MemoMind — Windows/dashboard packaging over the Hindsight memory engine

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

MemoMind (`github.com/24kchengYe/MemoMind`, ~716 stars, MIT) is **not an independent memory engine**. The actual memory system lives in `engine/hindsight_api/` and `engine/hindsight/`, which are the vendored upstream **Hindsight** engine (see `hindsight.md`, `hindsight-paper.md`). The banner still reads "Hindsight API startup"; the `hindsight` package `__init__.py` documents `HindsightEmbedded`/`retain`/`recall`/`bank_id` verbatim from upstream. MemoMind wraps this with Windows/WSL2 launch scripts, a local dashboard, and importers.

### Storage & Schema
PostgreSQL + pgvector + `pg_trgm`, all inherited from Hindsight (see the 50+ Alembic migrations under `engine/hindsight_api/alembic/versions/`). Memory units, `entities`, `memory_links`, `observations`, `mental_models`, `chunks`, `documents`. None of this schema is MemoMind's — it is upstream Hindsight's "retain → observations → mental models" architecture.

### Memory Types
Four upstream categories surfaced in the README as World / Experience / Observation / Mental Model. The evidence file's correction is right: these are Hindsight fact-type categories, **not** hierarchical L0→L3 layers.

### Write Path
Upstream Hindsight two-pass retain pipeline: `engine/hindsight_api/engine/retain/fact_extraction.py` (LLM fact extraction) → `fact_storage.py` → `orchestrator.py`, plus `entity_resolver.py` for entity linking and `consolidation/consolidator.py` for dedup/merge. MemoMind's own contribution here is a set of **patches** (`patches/`, applied by `patch_hindsight.py` over an installed `hindsight-api` venv): an `occurred_start` fallback, a Windows `strftime('%-d')` compatibility fix, a consolidation-prompt "language rule," and a "skip trivial observation" guard. These are bug-fix/localization patches, not new mechanisms.

### Retrieval
Upstream Hindsight 4-way hybrid: pgvector semantic + BM25 (Postgres FTS) + knowledge-graph link expansion + temporal, with a cross-encoder reranker (`engine/hindsight_api/engine/cross_encoder.py`, `jina_mlx_reranker.py`). The **one MemoMind-authored retrieval change** is in `engine/link_expansion_retrieval.py`: a `super_entities` CTE that excludes hub entities with `mention_count > 200` (e.g. "user", "用户" that appear in nearly every memory) from graph-link expansion to prevent noise flooding. Small, sensible, single-purpose.

### Consolidation / Processing
Upstream Hindsight consolidation engine (`consolidation/consolidator.py`, `prompts.py`) — observation synthesis and mental-model updates. MemoMind patches the prompt (language rule) and adds a trivial-observation skip. No independent consolidation design.

### Lifecycle Management
Evidence file is correct: only observation-level auto-pruning exists (upstream); full time-weighted decay is a roadmap item. No versioning beyond upstream mental-model history tables.

### MemoMind's actual product surface
The novel work is packaging, not memory: a Flask/HTML **dashboard** (`dashboard.py`, `dashboard.html`, 150KB) with graph/timeline views; importers for AI chats (`import_ai_chats.py`) and a "DayLife" activity planner (`import_daylife.py`, `sync_daylife_smart.py`); backup/restore (`backup-memomind.py`, `restore_backup.py`); Windows service/task scaffolding (`.vbs`, `.bat`, `.ps1`, `memomind.service`); and marketing PPT builders (`build-ppt.js`). A companion repo "Recall" handles human-side conversation browsing.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| "4-way hybrid retrieval" | Real, in `link_expansion_retrieval.py` + cross_encoder | Validated — but it is upstream **Hindsight's**, not MemoMind's |
| "Knowledge graph with entity linking" | `entity_resolver.py`, `memory_links` table | Validated, upstream |
| "100% local, GPU-accelerated" | Local embeddings + optional LLM API for retain | Plausible; CUDA path is packaging value |
| "Living knowledge graph that grows" | Consolidation engine | Validated, upstream Hindsight |
| Super-hub entity noise filtering | `mention_count > 200` CTE | Validated — MemoMind's own, small tweak |
| No benchmark numbers | README is marketing-heavy | No LoCoMo/PERMA/QA numbers — **not comparable to our 85.1 LoCoMo QA** |

No end-to-end QA or retrieval-recall benchmarks are reported. Nothing here is comparable to Somnigraph's measured numbers.

---

## Relevance to Somnigraph

### What MemoMind does that Somnigraph doesn't
Nothing at the engine level that Hindsight doesn't already give us (already analyzed). At the **product** level: a human-facing dashboard with graph + timeline views, and multi-source importers (chat exports, activity planner). Somnigraph is single-user MCP with no UI — but that is a deliberate scope choice, not a gap in the memory design.

### What Somnigraph does better
Everything MemoMind claims as intelligence belongs to upstream Hindsight; against **that** engine Somnigraph's differentiators stand (learned 26-feature reranker in `reranker.py`, explicit feedback loop with measured Spearman r=0.70, LLM-mediated sleep in `scripts/sleep_nrem.py`/`sleep_rem.py`, PPR graph expansion in `scoring.py`). MemoMind itself adds no retrieval-quality machinery beyond one hub-filter CTE.

---

## Worth Stealing (ranked)

### 1. Super-hub entity filter (Low) — note-only
**What**: Exclude entities with very high `mention_count` (>200) from graph-link expansion so ubiquitous nodes ("user") don't flood traversal with noise.
**Why**: Any entity/co-retrieval graph accumulates hub nodes that dominate expansion.
**How**: In `scoring.py` PPR expansion. **But**: Somnigraph's novelty-scored adjacency already damps hub dominance in a more principled (continuous, IDF-like) way than a hard `mention_count > 200` cutoff, and our edges are Hebbian-PMI weighted (PMI already penalizes high-frequency co-occurrence). This is convergent evidence that hub-damping matters, not a new mechanism to adopt.

There is no adopt-worthy core idea here.

---

## Not Useful For Us

### The dashboard, importers, Windows service scaffolding
Product-packaging for a multi-source personal digital-twin, not memory-algorithm work. Out of scope for Somnigraph's research-artifact focus.

### The vendored Hindsight engine
Already covered by `hindsight.md` and `hindsight-paper.md`. Analyze the upstream, credit it correctly; nothing new is contributed downstream.

---

## Connections

- **Directly downstream of Hindsight** — see `hindsight.md`, `hindsight-paper.md`. MemoMind vendors `hindsight_api`/`hindsight` and patches it; treat any "MemoMind" retrieval/consolidation claim as a Hindsight claim.
- The hub-node damping insight is convergent with our **Hebbian-PMI edge weighting** (`scoring.py`) and novelty-scored adjacency — independent arrival at "downweight ubiquitous connections."
- Same pattern as other repos in this sweep: **write-path/packaging over an upstream engine**, corroborating the Phase 18 finding that leaderboard-style repos often re-brand an existing engine.

---

## Summary Assessment

MemoMind is a Windows/WSL2-friendly **distribution** of the Hindsight memory engine with a personal-digital-twin dashboard and importers layered on top. Its genuine authored contribution to the memory system is a handful of localization/bug-fix patches (`patches/`) and one retrieval tweak (super-hub entity filtering in link expansion). Everything the README markets as intelligence — hybrid retrieval, knowledge graph, entity linking, consolidation, mental models — is upstream Hindsight, already in our corpus.

The single most important thing for Somnigraph to take from it is: nothing new. The hub-filter idea is real but already subsumed by our novelty-scored adjacency and PMI edge weighting. There are no benchmark numbers to compare, and the evidence file's checkmarks describe Hindsight's capabilities, not MemoMind's own engineering.

Verdict: **SKIP**. Credit the upstream (`hindsight.md`), do not double-count MemoMind as an independent system. Worth remembering only as another data point that many "memory" repos in the wild are packaging over a small set of upstream engines.
