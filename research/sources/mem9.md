# mem9 — PingCAP/TiDB-backed shared memory server with Mem0-style LLM reconciliation write path

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

mem9 (Go server `mnemo-server`, internal package name `mnemos`) is a production, multi-tenant, multi-agent memory service built by PingCAP on TiDB Cloud. Unlike most systems in this corpus it is not a single-user research artifact — it is a hosted SaaS with a REST API, a dashboard, plugins for six agent frameworks (OpenClaw, Hermes, Claude Code, OpenCode, Codex, Dify), metering, encryption, and rate limiting. The memory logic that matters for us lives entirely in the Go server (`server/internal/service/` and `server/internal/repository/`), not the plugins.

### Storage & Schema
- **Backends**: TiDB (primary, MySQL-compatible with native `VECTOR` type + `VEC_COSINE_DISTANCE` + FTS), Postgres, and an internal `db9` HTTP data-API backend. Backend chosen via `MNEMO_DB_BACKEND` (`server/internal/repository/factory.go`).
- **Memory unit** (`server/internal/domain/types.go`): `id` (UUID), `space_id`, `content`, `source`, `tags` (JSON), `metadata` (JSON), `embedding`, `version`, `updated_by`, `created_at`, `updated_at`, plus `memory_type` and `state`.
- **memory_type**: `pinned | insight | session`. `insight` = LLM-reconciled facts; `pinned` = verbatim user-forced writes (bypass reconciliation); `session` = raw captured turns.
- **state**: `active | paused | archived | deleted`.
- Embeddings: 1536d OpenAI `text-embedding-3-small` default; Ollama / any OpenAI-compatible endpoint; TiDB server-side `EMBED_TEXT()` "auto model" path also supported.

### Memory Types
Three-way type tag, not a layered architecture (no detail/summary/gestalt equivalent). Type is used at retrieval as a scoring weight (`applyTypeWeights`) and to gate reconciliation (pinned skips UPDATE — `memory.go:1295`, "skipping UPDATE for pinned memory — treating as ADD").

### Write Path
The interesting part. `MemoryService.Create` (`service/memory.go:63`) routes to `IngestService.ReconcileContent` (`service/ingest.go:438`) when an LLM is configured — a **Mem0-style extract-then-reconcile pipeline**:
1. **Extract** facts from raw content/turns via LLM, capped at 50 (`maxFacts`, ingest.go:581).
2. **Shadow-mode near-dup probe** (ingest.go:1043-1052): for each fact, `NearDupSearch` finds the nearest existing memory and records its cosine similarity to a Prometheus histogram (`metrics.NearDupCosineScore`). Facts **always pass through** — suppression is deliberately deferred until the score distribution is analyzed from prod. Comment: "Once a threshold is validated, add: if score >= threshold { drop or annotate }".
3. **Gather** existing memories relevant to each fact (`gatherExistingMemories`, per-fact vector search, `minSimilarityScore = 0.3`).
4. **Integer-ID indirection** (ingest.go:1083-1100): real UUIDs are mapped to small integer IDs (0,1,2…) via `idMap` before being shown to the LLM, "to prevent LLM hallucination." Each ref also carries a `relativeAge` string for temporal conflict resolution.
5. **Single batch LLM call** decides `ADD / UPDATE / DELETE / NOOP` per fact against the numbered existing set; the system maps integer IDs back to UUIDs and applies changes. UPDATE keeps the same ID; out-of-range/invalid IDs are skipped with warnings (ingest.go:1247-1326).
- **Space Chain routing** (ingest.go:~340): the extraction prompt can also classify each fact for routing into *other* shared spaces based on natural-language policies ("route facts related to PingCAP"). LLM-as-router for multi-space fan-out.

### Retrieval
Contrary to the evidence file's DESIGN.md summary (naive "vector wins, keyword=0.5" merge), the **code uses RRF** (`rrfMerge`, `rrfK = 60.0`, memory.go:244) — same family as Somnigraph but with a fixed textbook k, not Bayesian-tuned. Pipeline (`autoHybridSearch` / `autoHybridCandidates`):
1. Vector ANN (TiDB `VEC_COSINE_DISTANCE`) + keyword (FTS if available, else `LIKE`), each at `limit×3`.
2. `defaultMinScore = 0.3` floor on vector results; loose-token keyword fallback if both channels empty.
3. RRF fusion.
4. **Second-hop query-time expansion** (memory.go:771 `secondHopAutoSearch`): take top-3 first-hop results as seeds, run concurrent vector searches from their embeddings, RRF-blend at `secondHopWeight = 0.3`, excluding seeds. **Gated** by `secondHopGateScore = 0.5` — if the best first-hop cosine is below 0.5 the query "likely has no strong match (e.g. adversarial)" and expansion is skipped to avoid injecting noise.
5. `applyTypeWeights` (pinned/insight/session), sort, paginate.
- **Adjacent-turn expansion** for session recall (`AdjacentTurnRadius`), pulling neighboring conversation turns.
- No learned reranker. No feedback loop. No persistent graph.

### Consolidation / Processing
**None.** No sleep, no offline batch job, no clustering, no summarization pass. All intelligence is inline at write time (reconciliation) and query time (second-hop). Grep for decay/consolidate/sleep/reheat in `server/internal` returns nothing.

### Lifecycle Management
- **Supersession**: upsert-by-`key`, monotonic `version++`, `If-Match` optimistic concurrency; conflict resolution is **LWW** (LLM merge is a documented Phase-2 plan, not implemented).
- **Contradiction**: handled only at write time by the reconcile LLM's DELETE action; no standing contradiction edges.
- **No decay / no forgetting curve.** Explicit `DELETE` endpoint only.
- States (`paused`, `archived`) exist but there is no automated transition engine.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Hybrid vector+keyword recall | RRF code present and exercised (`rrfMerge`) | Validated — and better than the DESIGN.md doc claims |
| LoCoMo 58.84% F1 / 71.95% LLM-score | mem9.ai, qwen3.5-plus judge; multi-hop only 22.60% F1, temporal 13.79% | Weak, and **not comparable** to our 85.1% Opus-judge QA — different judge model, F1-primary, no multi-hop strength |
| Mem0-style ADD/UPDATE/DELETE write discipline | `ingest.go` reconcile prompt + apply loop | Validated (real code, real prompt) |
| Second-hop expansion improves recall | Code present, gated; no ablation numbers published | Plausible, unvalidated |
| "Automatically upgrades to hybrid, no re-indexing" | Graceful degradation across FTS/keyword/vector availability | Validated (engineering, not research) |
| Multi-agent shared memory (spaces) | Space + space-chain code, token-per-agent | Validated — genuine multi-tenant design |

---

## Relevance to Somnigraph

### What mem9 does that Somnigraph doesn't
- **LLM reconciliation at write time** (`ingest.go`) — Somnigraph has no write-path quality/dedup gating; it defers all merge/archive to `sleep_nrem.py`. mem9 collapses that into a synchronous per-write ADD/UPDATE/DELETE decision. This is the recurring "write-path discipline beats retrieval" theme (see agentmemory.md, byterover, Phase 18 sweep).
- **Query-time second-hop expansion with an adversarial gate** — Somnigraph's graph expansion (`scoring.py` PPR) depends on edges built during sleep; mem9 does neighbor expansion purely at query time by re-embedding top hits, with a similarity floor that suppresses expansion on weak/adversarial queries.
- **Multi-tenant, multi-agent spaces + LLM fact routing** — entirely outside Somnigraph's single-user MCP scope.

### What Somnigraph does better
- **Learned 26-feature LightGBM reranker** (`reranker.py`) — mem9 has no reranker at all, only RRF + fixed type weights.
- **Explicit feedback loop** with measured Spearman r=0.70 (`tools.py`, `recall_feedback`) — mem9 has zero feedback signal.
- **Offline LLM-mediated sleep consolidation + typed graph** (`sleep_nrem.py`, `sleep_rem.py`) — mem9 has no consolidation whatsoever.
- **Biological decay with reheat** — mem9 has no decay.
- **Bayesian-tuned RRF k=14** vs mem9's textbook k=60.
- **End-to-end QA rigor**: 85.1% Opus-judged LoCoMo vs mem9's 58.84% F1 (weaker judge, weaker multi-hop).

---

## Worth Stealing (ranked)

### 1. Shadow-mode near-dup metric before setting a dedup threshold (Low)
**What**: Before enforcing any dedup/suppression rule, log the cosine distance of each incoming item's nearest existing neighbor to a histogram and never act on it — decide the threshold from the observed prod distribution (`ingest.go:1043-1052`, `NearDupCosineScore.Observe`).
**Why**: Somnigraph currently has no write-path dedup and would eventually want one; picking a threshold blind is how you get either silent memory loss or no effect. This is a disciplined, honest-accounting way to earn the number.
**How**: Add a `near_dup_cosine` logging line in `tools.py:impl_remember` (compute NN cosine over the sqlite-vec table at write), accumulate offline, inspect the distribution against known dup/non-dup pairs before ever gating. Pairs cleanly with the existing sleep-time merge.

### 2. Integer-ID indirection when an LLM is in the consolidation loop (Low)
**What**: When handing existing memories to an LLM for merge/classify decisions, present them as small integers (0,1,2…) with a server-side map back to real IDs, and reject out-of-range/invented IDs (`ingest.go:1083-1100`, `idMap`).
**Why**: Cleaner than Somnigraph's post-hoc "drop malformed LLM consolidation suggestions before apply" (commit 164fcb5) — it prevents the malformed/hallucinated-ID class of output at the source rather than filtering it after. Also shrinks prompt tokens vs UUIDs.
**How**: In `sleep_nrem.py`'s pairwise/merge prompts, number the candidate set and validate returned indices against the map; keep the 164fcb5 drop-filter as a belt-and-suspenders backstop.

---

## Not Useful For Us

- **TiDB/Postgres/db9 multi-backend, spaces, tenancy, metering, encryption, plugins** — SaaS infrastructure irrelevant to a single-user SQLite MCP server.
- **Space Chain LLM fact routing** — solves a multi-pool sharing problem Somnigraph doesn't have.
- **LWW conflict resolution** — Somnigraph's typed edges + sleep classification already model supersession/contradiction more richly than last-writer-wins.
- **The published LoCoMo number** — qwen3.5-plus-judged F1; not a target we should benchmark against.

---

## Connections

- **Mem0 lineage**: the ADD/UPDATE/DELETE/NOOP reconcile prompt is nearly identical in shape to Mem0's (see mem0-related analyses) — mem9 is essentially Mem0's write path re-implemented in Go on TiDB. The `old_memory` field, entity/attribute-slot rules, and age-tiebreaker all match.
- **Write-path-over-retrieval thesis**: independent corroboration of the Phase 18 sweep finding (agentmemory.md, byterover) — mem9's only real "intelligence" is at the write path; its retrieval is plain RRF with no reranker/feedback and it still ships. The LoCoMo adversarial category (96.19%) vs multi-hop (22.60%) split mirrors our own multi-hop vocabulary-gap ceiling.
- **Adversarial gating convergence**: `secondHopGateScore = 0.5` (don't expand on weak matches) is the same instinct as Somnigraph's proactive-injection **floor gate** (`docs/proposals/proactive-injection.md`) and adversarial-probe defense — a coarse similarity floor used to suppress low-confidence expansion.

---

## Summary Assessment

mem9 is the most *productized* system in this corpus — a real multi-tenant Go service by the TiDB team — but architecturally it is a Mem0 write path bolted onto TiDB's native vector+FTS, with plain RRF retrieval and no reranker, no feedback loop, no consolidation, and no decay. Its single genuine idea is **synchronous LLM reconciliation at write time** (extract → near-dup probe → integer-ID batch ADD/UPDATE/DELETE), which is the same write-path-discipline lesson we've now seen from three independent sources. Somnigraph deliberately made the opposite bet (rich offline sleep + learned reranker + feedback), and the head-to-head benchmark (85.1% Opus QA vs 58.84% F1) favors ours, though the judges differ.

The single most useful takeaway is small and process-level, not architectural: the **shadow-mode near-dup metric** — instrument the near-duplicate cosine distribution in prod and refuse to set a suppression threshold until you can see the distribution. It is exactly the kind of verify-before-you-gate discipline the honest-accounting priority values, and it is cheap to add. The **integer-ID indirection** is a smaller, cleaner refinement of a fix we already shipped (164fcb5). Everything else — spaces, tenancy, second-hop, LWW, TiDB — is either infrastructure we don't need or a weaker version of something Somnigraph already does. The sharpest correction to the evidence file: its retrieval is RRF (k=60), not the naive merge the vendored DESIGN.md describes, and its headline LoCoMo number is retrieval-F1 under a qwen judge, not comparable to our end-to-end Opus-judged QA.
