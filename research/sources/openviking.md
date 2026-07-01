# OpenViking — Filesystem-paradigm "context database" for agents (L0/L1/L2 tiering, directory-recursive vector retrieval)

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

OpenViking (volcengine/OpenViking, AGPL-3.0) is a large polyglot system: a Rust `ragfs` engine (`crates/`), a C++ vector index (`src/`), and the Python memory/agent layer (`openviking/`). It bills itself as a "Context Database," not a memory system — memories, resources, and skills are all files under `viking://` URIs organized in a directory tree. This analysis focuses on the memory-relevant Python layer.

### Storage & Schema
Everything is a node in a virtual filesystem (`VikingFS` / `VikingDBManager`) addressed by `viking://` URIs (e.g. `viking://user/{space}/memories/entities/...`). Each node carries three content tiers materialized as sibling files: `.abstract.md` (L0, ~100 tokens), `.overview.md` (L1, ~2k tokens), and full L2 content. A separate vector index stores `{uri, vector, metadata}` with no file content. The retrieval schema (`MatchedContext`, `openviking_cli/retrieve/types.py`) has 6 fields: `uri`, `context_type`, `level`, `abstract`, `score`, `relations`.

### Memory Types
`context_type` ∈ {Resource, Memory, Skill} is first-class on every node. Within Memory, extraction targets 8 categories (profile, preferences, entities, events, cases/experiences, patterns, tools, skills — see `session/memory/graph_view.py` TYPE_COLORS and memory_type_registry.py). These are storage directories, not user keywords and not a NER pipeline.

### Write Path
Session commit triggers background extraction (`session/compressor_v2.py` → `session/memory/extract_loop.py`, an LLM ReAct orchestrator). Extraction produces typed memory files; **LLM dedup decisions** run during compression. Updates use typed field-merge operations (`session/memory/merge_op/`: `immutable`, `patch`, `replace`, `sum`, `link_merge`) rather than blind append. L0/L1 tiers are generated async bottom-up by the parse module (`openviking/parse/`, SemanticQueue). There is no salience/quality scoring gate on stored content — dedup is the only write-path filter.

### Retrieval
Core memory retrieval is `retrieve/hierarchical_retriever.py`. Two modes:
- **QUICK** (`find()`, no rerank): single global vector search, threshold filter, sort by score. Dense + optional sparse vector (hybrid embedder), but **no BM25/RRF fusion** in memory retrieval — `grep` (Rust/FS BM25) is a separate service method for fulltext, not fused into the ranked memory search.
- **THINKING** (`search()`, LLM `IntentAnalyzer` + rerank): global vector search → optional external rerank (`models/rerank.RerankClient` — a hosted rerank API, not a learned local model) → **directory-recursive descent**: high-scoring L0/L1 directory nodes are pushed onto a priority queue and their children searched, with score propagation `final = α·child + (1-α)·parent`. Convergence after ≤3 stagnant rounds.

Two "novel" ranking knobs both ship **DEFAULT-DISABLED** (`openviking_cli/utils/config/retrieval_config.py`):
- `hotness_alpha = 0.0` → the recency/frequency "hotness" blend is off (see Lifecycle).
- `score_propagation_alpha = 1.0` → parent-directory score contributes nothing; final = pure child score. So the shipped default is plain vector search + optional hosted reranker + directory descent that ignores parent scores.

`relations` in every returned `MatchedContext` is hard-coded to `[]` (`_convert_to_matched_contexts`, line ~533). Extracted links are **not** used in retrieval scoring.

### Consolidation / Processing
No offline sleep/consolidation cycle. Extraction, dedup, and L0/L1 generation happen at commit time as background async tasks. There is no pairwise relationship classification, no gap analysis, no merge/archive pass over the corpus.

### Lifecycle Management
`retrieve/memory_lifecycle.py` (issue #296) defines `hotness_score = sigmoid(log1p(active_count)) · exp(-ln2·age/7d)` — a retrieval-time recency+frequency boost with a 7-day half-life. It is dormant: `hotness_alpha=0` disables it by default. No archival, no forgetting, no version-supersede, no valid_from/valid_until. `merge_op.replace`/`patch` overwrite fields in place (no history kept).

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Filesystem paradigm unifies memory/resource/skill | `viking://` URI tree, VikingFS; genuinely the core abstraction | Validated (design), real |
| L0/L1/L2 tiered loading cuts tokens up to 91% | README table (OpenClaw -91%, Claude Code -63.2%) | Plausible; this is context-shaping/indexing, not retrieval quality |
| LoCoMo QA lift: Claude Code 57.21%→80.32%, OpenClaw 24.2%→82.08% | `benchmark/locomo/*/judge.py` LLM-judged accuracy | Real end-to-end QA, but confound: it measures the lift of adding a memory layer to weak/native-memory agents, not head-to-head vs other memory systems; judge model undisclosed in README |
| "Directory recursive retrieval improves retrieval effect" | `hierarchical_retriever._recursive_search` | Mechanism exists, but its distinguishing knob (score propagation) is default-off |
| Hybrid retrieval | Dense+sparse embedder + directory structure | Not BM25+vector RRF; "hybrid" = vector + FS-structure + optional grep |
| HotpotQA top-20 91%, RAG-bench 66.87% | `benchmark/RAG/`, README | These are retrieval-recall/accuracy numbers, NOT comparable to Somnigraph's 85.1 LoCoMo QA |
| Decay / hotness lifecycle | `memory_lifecycle.py` exists | Dormant — `hotness_alpha=0` default; effectively no decay |

---

## Relevance to Somnigraph

### What OpenViking does that Somnigraph doesn't
- **Write-time link extraction + memory graph.** LLM-extracted `related_to`/typed links are stored as memory fields at write time and rendered as an Obsidian-style D3 graph (`graph_view.py`, `relation_service.py`). Somnigraph builds its graph only during sleep (`scripts/sleep_nrem.py`) — OpenViking's is real-time. Caveat: OpenViking's graph is visualization/observability only; it is not fed into retrieval scoring (`relations=[]`), so it is weaker than Somnigraph's PPR-conditioned retrieval.
- **Typed field-merge operations at write time** (`merge_op/`: immutable/patch/replace/sum/link_merge) over structured memory templates. Somnigraph's `tools.py remember()` is free-form; it has no per-field merge vocabulary.
- **L0/L1/L2 progressive-disclosure loading with async bottom-up generation** and per-level URIs, driving large token reductions. Somnigraph has detail/summary/gestalt layers but does not expose graded on-demand loading tuned for context-window economy.
- **Multi-tenant / account isolation** (`memory_isolation_handler.py`, `RequestContext` tenant filtering). Somnigraph is single-user.

### What Somnigraph does better
- **Retrieval quality mechanisms are live, not dormant.** Somnigraph's 26-feature LightGBM reranker (`reranker.py`), RRF fusion (`fts.py`+vector), and PPR graph expansion (`scoring.py`) are the default path. OpenViking's two differentiating knobs are default-disabled, leaving shipped retrieval = vector + hosted-API rerank + directory descent.
- **Learned local reranker with measured NDCG gain** vs OpenViking's dependence on an external rerank API with no training/eval loop.
- **Explicit feedback loop** (per-query utility ratings, EWMA/UCB, Hebbian PMI, Spearman r=0.70 with GT). OpenViking has no feedback loop; `active_count` feeds only the dormant hotness score.
- **Offline LLM-mediated consolidation** (NREM/REM sleep) — typed edge detection, contradiction classification, gap analysis. OpenViking has no consolidation pass.
- **Graph-conditioned retrieval.** Somnigraph's edges actually change ranking via PPR; OpenViking's links never touch the score.
- **Live decay with reheat.** Somnigraph's per-category decay is on by default; OpenViking's is off.

---

## Worth Stealing (ranked)

**None.** Confirms the prior nugget-mining result (0 build nuggets). The two candidate mechanisms are either default-disabled (hotness decay, directory score-propagation) or already better-realized in Somnigraph. The write-time LLM-dedup + typed field-merge touches Somnigraph's acknowledged write-path-quality gap, but it is tightly coupled to OpenViking's structured filesystem-template model and is redundant with the write-path-discipline direction already captured in the Phase 18 source sweep — not worth porting.

---

## Not Useful For Us

### Filesystem paradigm + L0/L1/L2 tiering
This is an indexing/context-window-economy strategy (fit huge agent context into a token budget), not a retrieval-quality strategy. Somnigraph is a single-user MCP store where token budget is set by the recall caller, not the memory system; the headline 91% token reduction is orthogonal to Somnigraph's NDCG/recall goals.

### Directory-recursive retrieval + score propagation
Presupposes a deep, meaningful directory hierarchy (their whole paradigm). Somnigraph's store is flat by design and uses a learned graph (PPR) for structure. Adopting directory descent would mean rebuilding the storage model to gain a feature that is default-off upstream.

### Hotness score
`sigmoid(log-freq)·exp(-recency)` is subsumed by Somnigraph's per-category decay + reheat-on-access + the `session_recency` reranker feature — all live.

---

## Connections

- **Context-shaping over retrieval-quality**, like the Phase 18 finding (see `ai-memory-comparison.md`, `byterover.md`): the LoCoMo leaders win on write-path/context discipline, not fusion or reranking. OpenViking's headline is token reduction, and its retrieval sophistication (score propagation, hotness) is unshipped — same pattern.
- **Verbatim/structured storage over lossy summarization**, convergent with MemPalace/agentmemory (see `agentmemory.md`): the L0/L1/L2 tiers keep full L2 detail and layer summaries on top rather than discarding.
- **Write-time link/graph construction** contrasts with Somnigraph's sleep-time graph and echoes other real-time-graph systems, but here the graph is visualization-only — a cautionary example that a "knowledge graph" can be present in the schema yet absent from ranking.
- **Benchmark-comparability caveat** shared with the carsteneu leaderboard corrections: OpenViking's RAG/HotpotQA cells are retrieval recall/accuracy, and the LoCoMo cells are agent-lift (native vs +OpenViking), neither directly comparable to Somnigraph's 85.1 head-to-head LoCoMo QA.

---

## Summary Assessment

OpenViking's genuine contribution is an engineering paradigm: treat all agent context (memory, resources, skills) as a tiered virtual filesystem addressed by URIs, with L0/L1/L2 progressive disclosure that lets an agent pull ~100-token abstracts before paying for full detail. That drives real, large token reductions and is a clean answer to context-window economy for long-running agents. It is a well-built, production-scale system (Rust engine, C++ index, multi-tenant, encryption, export/pack).

For Somnigraph specifically there is nothing to take. The features that would matter to a retrieval-quality research artifact — a decay/lifecycle model and a directory score-propagation ranker — both ship default-disabled, so the shipped retrieval is plainer than Somnigraph's (vector + hosted-API rerank + directory descent, no local learned reranker, no RRF, no feedback loop, no graph-conditioned scoring, no consolidation). The one place OpenViking is architecturally ahead — write-time link extraction into a memory graph — is undercut by the fact that those links never influence retrieval (`relations=[]`); it is an observability/visualization feature, not graph retrieval.

The sharpest correction to the evidence file: it marks `decay=false`, but decay code exists (`memory_lifecycle.py`, 7-day half-life) and is simply disabled by default — the honest statement is "present but dormant," which is the same story as the directory score-propagation knob. And the benchmark framing needs care: the LoCoMo numbers are LLM-judged agent-lift (native-memory agent vs +OpenViking), and the RAG/HotpotQA numbers are retrieval recall — neither is a head-to-head end-to-end QA comparison against a memory system like Somnigraph's 85.1. Verdict: SKIP.
