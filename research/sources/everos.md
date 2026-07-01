# EverOS — Local-first, markdown-source-of-truth memory runtime wrapping the `everalgo` engine

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

**One-line orientation**: EverOS (open-source `EverMind-AI/EverOS`) is a *runtime and persistence orchestration layer*. The actual retrieval/extraction/consolidation algorithms live in closed, PyPI-published `everalgo-*` packages (`everalgo-user-memory==0.3.1`, `everalgo-agent-memory==0.3.1`, `everalgo-rank==0.4.1`, `everalgo-knowledge==0.1.1`). The repo you clone is the local-first shell around that engine — it owns storage, the cascade daemon, the offline job engine (OME), and the FastAPI/CLI surface, but *imports* the fusion/MaxSim/cluster/agentic ranking and the LLM extractors from `everalgo`. The headline benchmarks are the engine's, not this repo's (see cross-check).

### Storage & Schema
Three-part local stack, **markdown is the source of truth**:
- **Markdown files** — canonical `.md` per entry under `users/<owner_id>/episodes/episode-<YYYY-MM-DD>.md`, `agents/<owner_id>/...`. Human-readable, editable, diffable, Git-versionable. Direct file edits are picked up by a watcher.
- **SQLite** (`infra/persistence/sqlite`) — operational tables: `memcell`, `cluster`, `reflection_report`, `unprocessed_buffer`, `knowledge`, `conversation_status`, `md_change_state`.
- **LanceDB** (`infra/persistence/lancedb`) — Arrow-based vector + BM25 + scalar-filter index. One table per memory type. Vector dim 1024.

The `cascade` daemon (watcher → scanner → worker, `memory/cascade/`) reconciles markdown ↔ indexes. Each episode row carries `content_sha256` over content-bearing fields only, so re-reconcile skips re-embedding when nothing embedding-relevant changed (audit fields excluded from the hash) — a cheap re-index gate.

Episode schema (`lancedb/tables/episode.py`): `id=<owner_id>_<entry_id>`, `entry_id`, `owner_id/owner_type`, `app_id/project_id/session_id` (orthogonal scoping), `timestamp`, `parent_type/parent_id` (source memcell), `sender_ids[]`, `subject`, `summary`, `episode` (narrative), `episode_tokens` (pre-tokenized BM25 field, jieba for CJK), `md_path`, `content_sha256`, `deprecated_by` (soft-delete supersession pointer), `vector(1024)`.

### Memory Types
Seven, split into two hard-partitioned **tracks**:
- **User track**: `episode` (narrative), `atomic_fact` (~28 per episode, each independently embedded), `foresight` (forward-looking prediction with `evidence` + `start_time/end_time/duration_days`), `user_profile`.
- **Agent track**: `agent_case` (task_intent/approach/quality_score/key_insight), `agent_skill`.
- Cross-cutting: `knowledge_topic` (editable Markdown wiki with taxonomy + topic search).

### Write Path
`service.memorize` → `ingest` (normalize multimodal ContentItems; only `type=text` parsed today) → `_boundary` (single-pass MemCell segmentation, one sqlite `memcell` row per cell) → `UserMemoryPipeline` extracts per-sender Episodes via `everalgo.user_memory.EpisodeExtractor`, writes markdown, and fires `UserPipelineStarted`. That event kicks off **OME async strategies in parallel**: `extract_atomic_facts`, `extract_foresight`, `trigger_profile_clustering`, `trigger_skill_clustering`, `extract_agent_case/skill`. All extraction is LLM-driven inside `everalgo`; EverOS supplies engineering context (owner/session/parent ids) and persistence. No quality/salience gate is visible in EverOS code — gating (if any) is inside the engine.

### Retrieval
`SearchManager` (`memory/search/manager.py`) resolves one of four methods, delegating fusion to `everalgo.rank`:
- **KEYWORD** — LanceDB BM25 over `episode_tokens` (single route, no fusion).
- **VECTOR** — dense ANN; or `maxsim_atomic` strategy (ANN over the ~28× denser atomic_fact table, max-pool by parent episode, fetch parents).
- **HYBRID (default, no LLM rerank)** — a **four-layer hierarchy** (`hierarchy.py`): (L1) RRF of sparse+dense episodes; (L2) MaxSim re-score via atomic-fact child ANN grouped to parent; (L3) RRF merge of L1+L2, slice top_k; (L4) **fact eviction** — parent episode and its top facts are each calibrated to an LR probability via `cosine_to_lr_score(cosine, bm25)`, and the single best atomic fact *replaces* its episode in the result when its blended LR score beats the parent's. Puts a fact cosine and an episode's recall relevance on one comparable `[0,1]` scale.
- **AGENTIC** — the benchmark path (`agentic.py`, "1:1 with everalgo benchmark"): fact-level MaxSim (dense+sparse) → `ahybrid_retrieve` → `acluster_retrieve` (cluster-scoped expansion) → `aagentic_retrieve` with LLM **sufficiency check + 3-way multi-query generation + cross-encoder (Qwen3-Reranker) rerank + a second round**. Cross-encoder rerank providers: DashScope, DeepInfra, vLLM.

Note: the learned/cross-encoder rerank is a *cross-encoder API* (or LLM listwise), not a trained gradient-boosted model over engineered features.

### Consolidation / Processing
- **Reflection** (`memory/reflection/orchestrator.py`, cron `0 2 * * 1`, **`enabled=False` by default**): selects episode clusters with ≥2 members, merges them via `everalgo.user_memory.EpisodeReflector` (init/update modes), writes a merged episode, re-extracts facts, and **deprecates** originals (`deprecated_by = cluster entry_id`). Produces a `ReflectionReport`.
- **Evolution jobs** (`extract/evolution`): event/counter/cron-triggered Foresight/AtomicFact/Profile/Skill merge.
- **Cascade**: continuous markdown↔index reconciliation (not consolidation per se, but the offline sync loop).
- **OME** (`infra/ome`): the offline job engine — event/counter/cron triggers, idle scanner, crash recovery, gates, dispatch registry. This is real infrastructure, more built-out than a cron script.

### Lifecycle Management
- **Supersession, not decay**: `deprecated_by` soft-delete chain set by Reflection; reads default to `exclude_deprecated=True` for episodes. Explicit delete/forget API exists.
- **No time-based exponential decay** anywhere (grep confirms). No reheat-on-access. No priority/half-life.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| LoCoMo 93.05% / LongMemEval 83.00% | README, arXiv 2601.02163 (EverCore) / 2604.08256 (HyperMem); harness `benchmarks/EverMemBench` (NOT in this repo) | Plausible but **engine-attributed and not reproducible from this repo**. It's the `everalgo`/EverCore algorithm's number via the AGENTIC pipeline (multi-query + cross-encoder + cluster + LLM sufficiency + round-2), LLM-judged. Roughly comparable to our 85.1 end-to-end but different (looser) judge and a much heavier retrieval pipeline. |
| Local-first, no MongoDB/ES/Redis | `pyproject.toml` (only lancedb + everalgo + sqlite), README stack table | Validated for this repo. Contradicts the evidence file's MongoDB/ES/Milvus/Redis description (that was a different product state). |
| Markdown source of truth, editable + Git-versioned | `persistence/markdown/`, cascade watcher, `content_sha256` gate | Validated — genuine and unusual. |
| Hierarchical (episode→fact) retrieval with MaxSim | `hierarchy.py`, `maxsim_atomic_recall`, `amaxsim_retrieve` | Validated in code. The most interesting mechanism. |
| Contradiction resolution during consolidation | Reflection merges clusters via engine reflector | Unverified in EverOS code — merge logic is inside closed `everalgo`; no explicit contradiction classifier visible. |
| Foresight (predictive) memory type | `models.py` Foresight, `extract_foresight.py` | Validated as a schema + extraction strategy; quality unmeasured here. |

---

## Relevance to Somnigraph

### What EverOS does that Somnigraph doesn't
- **Atomic-fact child decomposition + MaxSim fact eviction** — directly attacks the exact ceiling Somnigraph named in `docs/multihop-failure-analysis.md` (the ~88% vocabulary gap). Somnigraph has detail/summary/gestalt *layers* but scores at the memory level in `scoring.py`; it has no per-child ANN + max-pool-to-parent + fact-replaces-parent eviction. This is a real gap.
- **Markdown as source of truth with a reconciling cascade daemon** — Somnigraph's source of truth is SQLite (`db.py`); no human-editable/Git-diffable representation, no `content_sha256`-gated re-embed skip.
- **Foresight / forward-looking memory type** — Somnigraph has no anticipatory category; connects to the unrealized `docs/proactive-injection.md` work.
- **Iterative agentic retrieval (LLM sufficiency + multi-query + round-2)** — Somnigraph does single-shot retrieve→rerank.
- **Orthogonal multi-tenant scoping** (user/agent/app/project/session) and a separate agent-memory track (cases/skills) — Somnigraph is single-user.

### What Somnigraph does better
- **Learned pointwise reranker over 26 engineered features** (`reranker.py`, NDCG 0.7958) trained on real-data queries — EverOS has no learned feature-based ranker; its "rerank" is a cross-encoder API call or LLM listwise. Somnigraph's is cheaper per query and offline-trainable.
- **Explicit measured feedback loop** (per-query utility + durability, EWMA, UCB, Spearman r=0.70 with GT) — EverOS has *no* retrieval feedback signal at all. This is Somnigraph's single biggest edge.
- **Graph-conditioned retrieval** — typed edges (supports/contradicts/evolves), PPR expansion, betweenness as a feature (`scoring.py`, `sleep_nrem.py`). EverOS clusters episodes but has no typed-edge graph or PPR.
- **Biological decay with reheat** — EverOS has only supersession soft-delete, no time decay.
- **Self-contained** — Somnigraph's algorithms are in-repo and inspectable; EverOS's core IP is in closed `everalgo` wheels.

---

## Worth Stealing (ranked)

### 1. Atomic-fact child decomposition + MaxSim fact eviction (High)
**What**: Decompose each episode/memory into ~N atomic facts, embed each fact independently, and at retrieval time (a) re-score the parent by its best-matching child fact (max-pool), and (b) let the child fact *replace* the parent in results when the child is more query-relevant than the parent. Calibrate parent-cosine, child-cosine, and BM25 onto one comparable scale before the comparison.
**Why**: Somnigraph's own multi-hop failure analysis pins the retrieval ceiling on a vocabulary gap — a long memory's single mean-pooled embedding dilutes the one specific fact a multi-hop query needs. Fact-level embeddings + max-pool is the mechanism that recovers it, and it's orthogonal to the existing detail/summary/gestalt layers (those are LLM-summarized abstractions; this is atomic decomposition with independent vectors).
**How**: New child table (or reuse the layers table) storing atomic facts with `parent_id` + own embedding; in `scoring.py` add a MaxSim pass (ANN over facts → group by parent → parent score = max child score) merged with the existing RRF episode pool; optionally emit the winning fact as the surfaced unit. Could be evaluated on the LoCoMo multi-hop slice where the ceiling was measured.

### 2. Foresight — a forward-looking memory type (Medium)
**What**: A distinct memory category capturing predictions/anticipated needs, with an `evidence` field and a time window (`start/end/duration_days`).
**Why**: Feeds directly into the stalled `docs/proactive-injection.md` design — a stored "the user will likely need X around date Y" is exactly the signal a proactive-recall hint could surface, and it's evaluable against use/ignore labels.
**How**: Add a `foresight` category (or a `valid_from`-anchored reflection subtype) written during REM gap-analysis in `sleep_rem.py`; the proactive-injection floor study could include foresight items as candidates.

### 3. Cosine→probability calibration for fusion (`cosine_to_lr_score`) (Medium, consider)
**What**: Map raw cosine and BM25 to a shared logistic-regression probability so heterogeneous scores are directly comparable on `[0,1]`, instead of rank-only RRF.
**Why**: Somnigraph's RRF (k=14) discards score magnitude. A calibrated probability could be a better-conditioned *input feature* to the learned reranker than raw ranks.
**How**: Fit a 1-2 param logistic on (cosine, bm25) → relevance using existing GT queries; expose as a reranker feature in `reranker.py`. Likely partly redundant with what the LightGBM reranker already learns — worth an ablation, not a rewrite.

---

## Not Useful For Us

### Markdown-as-source-of-truth + cascade daemon
Solves a multi-client/multi-tenant, human-edits-the-store, Git-versioning problem Somnigraph (single-user, MCP, SQLite-canonical) does not have. The one transferable crumb is the `content_sha256`-gated re-embed skip for any bulk re-index path.

### Agent track (cases/skills), orthogonal app/project scoping, OME multi-tenant engine
Built for a multi-agent SaaS runtime. Somnigraph's single-user MCP model makes the partitioning overhead pure cost.

### Iterative agentic retrieval (sufficiency + multi-query + round-2)
Well-known self-RAG territory; expensive (multiple LLM calls per query) and at odds with Somnigraph's offline/deterministic-scoring bias. Note-only.

---

## Connections

- **Atomic-fact + MaxSim decomposition** is the same write-path-quality lever the Phase 18 sweep flagged (`ai-memory-comparison.md`, ByteRover/agentmemory/MemPalace): the LoCoMo leaders win on *what gets stored and at what granularity*, not on fusion tricks. EverOS is another data point — its 93.05 comes from atomic-fact granularity + agentic retrieval, not a clever scorer.
- **Same org as `evermemos.md`** (EverMemOS). This EverOS repo is the *local-first, markdown* reincarnation; the carsteneu evidence file audited an older enterprise stack (MongoDB/ES/Milvus/Redis, `methods/EverCore/`) — treat the two as different artifacts.
- **Supersession via `deprecated_by`** convergent with memv/memos supersession chains; contrast Somnigraph's decay+reheat, which is time-based rather than event-based.
- **Foresight** is a novel type not seen in the other profiled systems; nearest neighbor is Somnigraph's own unrealized proactive-injection idea.

---

## Summary Assessment

EverOS's real contribution is **granularity plus packaging**: it decomposes conversations into episodes *and* independently-embedded atomic facts, retrieves with a MaxSim parent/child hierarchy that lets a precise fact evict a diluted parent, and wraps the whole thing in a genuinely nice local-first runtime (markdown source of truth, LanceDB, an event-driven offline engine, content-hash-gated re-indexing). The single most important thing for Somnigraph to take is the **atomic-fact + MaxSim eviction mechanism** — it targets, by construction, the exact vocabulary-gap ceiling Somnigraph already measured and could not cross with summary/gestalt layering.

What's overhyped or missing: the headline 93.05 LoCoMo / 83.00 LME numbers are **not this repo's** — the algorithms live in closed `everalgo` wheels and the benchmark harness (`EverMemBench`) isn't vendored here, so the open repo is effectively packaging over an upstream engine whose IP you can't inspect. The numbers are also LLM-judged (looser than our Opus judge) and produced by the heavyweight AGENTIC pipeline, not the default HYBRID path. On the axes Somnigraph has invested in, EverOS is *weaker*: no learned feature reranker, **no retrieval feedback loop at all**, no typed-edge graph or PPR, no time decay. The default HYBRID path even ships with LLM rerank off. So: mine the fact-decomposition idea and the foresight type; ignore the runtime architecture; and read the benchmark cell as engine-attributed and not directly comparable to our 85.1.

**Evidence-file correction**: the carsteneu audit describes a MongoDB/Elasticsearch/Milvus/Redis stack under `methods/EverCore/` with `benchmarks/EverMemBench/` — none of which exist in the cloned `EverMind-AI/EverOS` repo, which is a LanceDB+SQLite+markdown local runtime importing `everalgo-*` wheels. The audit also lists `supersede ❌` (wrong — `deprecated_by` is an explicit supersession chain set by Reflection) and `layeredMemory ⚠️ type-based` (understated — the episode→atomic_fact parent/child hierarchy is load-bearing in retrieval, and Reflection adds a cluster-merged layer above episodes). Benchmarks are end-to-end QA (Add→Search→Answer→Evaluate) per the evidence, so comparable in *kind* to our 85.1, but LLM-judged and engine-attributed.
