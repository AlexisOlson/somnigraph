# memoir - Git-for-AI-memory: versioned semantic-path store over a prollytree/git backend

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

<!-- No paper. Alpha OSS project; project page memoir-ai.dev, PyPI memoir-ai. -->

## Architecture

memoir's pitch is "Git for AI Memory": replace opaque vector DBs with a **transparent, versioned, cryptographically-integral** memory store keyed by **hierarchical semantic paths** (`preferences.coding.style`) instead of UUIDs or raw embeddings. The heavy lifting sits in two upstream deps: **`prollytree>=0.4`** (a Rust probabilistic Merkle/B-tree over git that provides content-addressed storage, branch/commit/merge, and an optional MiniLM text index) and **`langmem`/`langgraph`** (BaseStore interface, MemoryStoreManager). memoir is the taxonomy + LLM-classification + LLM-search + CLI/TUI/UI/plugin orchestration on top.

### Storage & Schema
- Backend: `ProllyTreeStore` (`store/prolly_adapter.py`) implementing LangGraph `BaseStore` over prollytree's `VersionedKvStore`. Git-backed, content-addressed, Merkle-hashed. This is where "cryptographic integrity" and "git-like versioning" come from — they are prollytree features, not memoir's own code.
- Memory unit: keyed at a dotted **semantic path**; value is a blob. Schema evolved to a **facet model (`schema_version: 2`, `merge_policy.py`)**: a key holds a list of timestamped `entries` (`content`, `confidence`, `timestamp`, `status`, optional `source`), with a projected top-level `content/confidence/timestamp` for legacy readers. v1 bare-content blobs are lazily upgraded on write. Per-key facet cap default **50** (`MEMOIR_FACET_MAX_ENTRIES`, oldest pruned).
- Namespaces = user/agent scoping (`user:agent` tuples).

### Memory Types
Two orthogonal taxonomies. (1) The **semantic path taxonomy** (`taxonomy/`) — a 3-level `category.subcategory.type` tree, LLM-classified, iteratively expandable. (2) A **classical memory-type map** (`merge_policy.py`): prefix rules map keys to WORKING / EPISODIC / SEMANTIC / PROCEDURAL, which in turn pick a default conflict strategy. Also dedicated "mementos": `ProfileMemento`, `TimelineMemento`, `LocationMemento` (`memento/`) for profile facts, time-ordered events, and places.

### Write Path
This is memoir's most substantive layer. `classifier/intelligent.py` runs an LLM to decide **memory-worthiness** (`ClassificationAction`: SKIP / CLASSIFY / EXPAND / USE_PARENT) and assign a taxonomy path with a **confidence** score and reasoning (multi-label paths supported; can trigger taxonomy expansion when confidence is low). On collision at a key, `merge_policy.apply_strategy` resolves via a **per-type conflict strategy**:
- `APPEND` (episodic; capped), `REPLACE` (last-write-wins; working/flat), `CONFIDENCE_GATED` (write only if incoming confidence ≥ existing — semantic default), `LLM_MERGE` (caller runs a haiku consolidation, passes merged text — procedural default), `MERGE_ON_READ` (store like append, consolidate at read), `REJECT` (surface a machine-readable `ConflictInfo` for interactive/RMW callers).
- The module is deliberately **pure** (no I/O, no LLM) so the strategy table is unit-testable; the caller wires the LLM for `LLM_MERGE`.

### Retrieval
No RRF, no learned reranker, no score fusion. Three channels:
1. **Path get** (`memoir get preferences.coding.style`): direct O(log n) tree lookup, offline, no LLM. This is the primary "read back a known fact" path.
2. **LLM path-selection search** (`search/intelligent.py`, `memoir recall`): loads *all* paths in the namespace, hands the LLM a list of paths + 100-char content samples, LLM returns the relevant path names. `mode="single"` = one LLM call over the full inventory; `mode="tiered"` = staged drill-down (pure-compute L1 histogram → LLM picks 2-4 L1 prefixes → optional L2 pick when an L1 exceeds 40 keys → LLM picks exact keys). `relevance_score` is just the stored **classification confidence**, not a query-similarity score.
3. **Vector search** (`services/vector_service.py`): thin wrapper over prollytree's `proximity_text` index (MiniLM, 384d). Feature-gated on the wheel build; used by the `watch` indexer, not fused into `recall`.

### Consolidation / Processing
No sleep/offline cycle. Consolidation is **write-time or read-time**: `LLM_MERGE` collapses colliding entries at write; `MERGE_ON_READ` + `read_project(consolidator=...)` defers consolidation to read when >1 active entry exists. "Memory Aggregation" = multiple entries accumulating at one semantic path. All deterministic except the optional LLM merge the caller injects.

### Lifecycle Management
The headline feature, all inherited from prollytree/git: **branch, commit, merge, checkout (rollback), blame (provenance audit)**. `branch_service.py` maps merge strategies to prollytree `ConflictResolution` (OURS/THEIRS/SKIP). Motivation: branch-aware memory that respects your `git checkout` state, and the ability to `blame`/revert a single "poisoning" memory rather than wiping the store. **No decay, no half-lives, no reheat** — lifecycle is version-control, not biological. Growth bounded only by the facet cap.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Git-like versioning (branch/commit/merge/rollback/blame) with cryptographic integrity | `branch_service.py`, `prolly_adapter.py` over prollytree `VersionedKvStore` | **Validated but upstream** — real, but it is prollytree's git backend, not memoir code |
| "O(log n) lookups instead of expensive vector ops" | Path `get` is a tree lookup | Plausible for *known-path* reads; irrelevant for semantic recall, which is an LLM-over-all-paths scan (loads up to 10000 memories) |
| "Multiple search engines: keyword + LLM-powered" | `search/intelligent.py` single/tiered; path `get` | Validated; add a 3rd (vector) that's real but not surfaced in `recall` |
| Auto-extraction / memory-worthiness gating | `classifier/intelligent.py` SKIP/CLASSIFY/EXPAND | Validated — genuine write-path gating |
| Confidence-gated / typed conflict merge | `merge_policy.py` pure strategy table | Validated, clean design |
| Replaces "opaque vector databases" | Storage is git+paths; vector is optional/secondary | Positioning claim; true that vectors aren't the primary index |
| End-to-end QA accuracy (LoCoMo etc.) | **None** — `benchmarks/` holds only classifier test data | **Absent** — no QA numbers exist to compare against Somnigraph's 85.1 LoCoMo |

---

## Relevance to Somnigraph

### What memoir does that Somnigraph doesn't
- **Write-path conflict/merge policy.** Somnigraph's `tools.py` `remember` has no dedup or merge-on-write — every call is a new row; consolidation waits for `sleep_nrem.py`. memoir's `merge_policy.py` resolves collisions synchronously with a typed strategy table (confidence-gated, LLM-merge, reject). This is exactly the write-path-quality gap flagged in the Phase 18 source sweep.
- **True version history + rollback + blame.** Somnigraph has `valid_from/valid_until` supersession and typed edges, but no branchable history and no "revert this one memory / who taught the agent this" audit. memoir gets it free from git.
- **Branch-aware memory** tied to the repo's git state (context isolation per branch) — Somnigraph has no notion of the host repo's VCS state.
- **Confidence as a first-class write field** driving gating decisions; Somnigraph uses priority + decay, not a classifier confidence.

### What Somnigraph does better
- **Actual relevance ranking.** memoir's `recall` relevance is stored classification confidence, and path selection is an LLM reading a flat list of every path — no BM25, no vector fusion, no learned reranker. Somnigraph's `reranker.py` (26-feature LightGBM, NDCG 0.7958) + RRF fusion (`scoring.py`, `fts.py`) is a categorically stronger retrieval stack, and there is no benchmark showing memoir retrieves well.
- **Graph-conditioned retrieval.** memoir has no PPR/edge expansion; its "graph" is the taxonomy tree only.
- **Feedback loop.** No per-query utility feedback, no Hebbian co-retrieval — Somnigraph's measured r=0.70 GT correlation has no analog.
- **Offline consolidation.** Somnigraph's three-phase sleep (gap analysis, edge typing, taxonomy) is far richer than memoir's write/read-time merge.
- **Decay/lifecycle.** memoir never forgets by time; Somnigraph's per-category exponential decay manages salience.
- **Scale of `recall`.** memoir's single-mode search loads *all* namespace memories (limit 10000) into one prompt per query; Somnigraph's indexed hybrid search doesn't degrade this way.

---

## Worth Stealing (ranked)

### 1. Typed write-time conflict/merge policy (effort: Medium)
**What**: A pure, unit-testable strategy table that resolves a write landing on an existing memory: `CONFIDENCE_GATED` (skip if incoming confidence < existing), `LLM_MERGE` (consolidate), `REPLACE`, `APPEND`, `REJECT`-with-`ConflictInfo`. Keyed by memory type.
**Why**: Directly addresses Somnigraph's missing write-path quality gating — the Phase 18 finding was that write-path discipline, not retrieval, is what LoCoMo/LME leaders win on. Somnigraph currently defers all merge/dedup to sleep; a synchronous confidence-gated or supersede-on-write path would cut low-value duplicates before they dilute retrieval.
**How**: New `src/memory/merge_policy.py` mirroring memoir's pure design; `tools.py::impl_remember` calls it before insert, using an embedding-similarity collision check (not just key-equality — Somnigraph has no semantic-path key) plus the existing category to pick a strategy. `REJECT`/`ConflictInfo` maps naturally to surfacing a "this contradicts memory X" prompt.

### 2. Git blame / rollback semantics for memory provenance (effort: High)
**What**: Ability to audit *who/what session* wrote a memory and revert a single poisoning entry without wiping the store.
**Why**: Somnigraph's honest-accounting ethos values provenance; a "one bad session poisons retrieval" failure mode is real. This is a concept worth holding, not necessarily the git implementation.
**How**: Somnigraph already stamps `valid_from`/session context; a lightweight "supersede + reason + revertible" flag on `db.py` rows (soft-delete with a restore path) captures most of the value without adopting prollytree. Note-only unless a poisoning incident actually surfaces.

---

## Not Useful For Us

### prollytree/git storage substrate
Somnigraph is single-user SQLite + sqlite-vec + FTS5 with a learned reranker tightly coupled to that schema. Swapping in a git-backed prolly tree buys versioning at the cost of the entire retrieval stack. The branch-per-git-state feature assumes a coding-agent-in-a-repo context Somnigraph doesn't target.

### LLM-over-all-paths retrieval
Loading every memory into a path-selection prompt is the pattern Somnigraph's hybrid+reranker was built to replace. The tiered drill-down is a mild token optimization but still strictly weaker than indexed retrieval.

### Semantic-path taxonomy as primary key
memoir addresses memories by human-readable path; Somnigraph addresses by content embedding + FTS. Adopting paths would mean rebuilding retrieval around an LLM classifier's taxonomy choices — a step backward from measured NDCG.

---

## Connections

- **Write-path discipline thesis**: memoir is independent corroboration of the Phase 18 sweep (ByteRover, agentmemory, MemPalace) — its richest, most-tested layer is `classifier` + `merge_policy` (the write path), while retrieval is thin. Cross-ref `ai-memory-comparison.md`, `agentmemory.md`, and the Phase 15 AMemGym finding.
- **Supersession**: memoir's `CONFIDENCE_GATED`/`REPLACE` and prior-values-survive-in-git-history is convergent with memv's supersession pattern and Somnigraph's `valid_from/valid_until`, but resolved at write rather than during sleep.
- **Taxonomy classification**: shares the LLM-taxonomy-classification approach with systems that route memories into fixed category trees; contrast with Somnigraph's emergent themes[] + sleep-built taxonomy.
- **Packaging-over-engine pattern**: like several carsteneu entries, the headline capability (versioning, vector) is an upstream library (prollytree); the repo's own contribution is orchestration + taxonomy + agent plugins.

---

## Summary Assessment

memoir's genuine contribution is a **crisp write-path**: an LLM memory-worthiness gate feeding a pure, typed conflict-resolution table (confidence-gated / LLM-merge / reject), over a **git-backed versioned store** that gives branch/commit/merge/blame/rollback for free. The "Git for AI memory" framing is coherent and the versioning is real — but it is prollytree's, and memoir is best understood as taxonomy + classification + CLI/plugin glue on top of prollytree + langmem. There are **no end-to-end QA benchmarks** anywhere in the repo, so nothing here is comparable to Somnigraph's 85.1 LoCoMo; the "O(log n) beats vector" claim only covers known-path reads, and its semantic `recall` is an LLM scanning every path — a retrieval design Somnigraph already outperforms with RRF + a learned reranker.

The single most valuable takeaway is the **write-time merge policy** (§Worth Stealing 1): a synchronous, confidence-gated / typed conflict resolver is exactly the write-path gate Somnigraph lacks and defers entirely to sleep, and it lands on the same lever the Phase 18 sweep identified as what benchmark leaders actually win on. That's a "consider," not an "adopt" — Somnigraph keys by embedding, not path, so the collision-detection half needs a similarity check memoir doesn't need.

Overhyped/missing: the versioning is upstream, not memoir's engineering; retrieval quality is unmeasured and structurally weak; there is no decay, no feedback loop, no graph beyond the taxonomy tree. **Verdict: MAYBE** — one revisit-if angle (write-path merge policy), no core idea to adopt wholesale.
