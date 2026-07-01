# Memory Palace — Human-in-the-loop MCP memory with a write-time dedup guard, snapshot/rollback, and review-gated maintenance engines

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

Memory Palace (`AGI-is-going-to-arrive/Memory-Palace`, v3.9.0, MIT, ~300 stars) is a FastAPI backend + React dashboard + MCP server. Python 77.5% / JS 17.9%, ~33 MB, Docker-first. The design philosophy is **safety-first, human-approved**: every derived or destructive operation is draft-by-default and gated behind a review token. This is the opposite of Somnigraph's autonomous LLM-mediated sleep.

### Storage & Schema
SQLite (async SQLAlchemy core) with 8 numbered migrations. Core tables: `memories` (content, `vitality_score`, `access_count`, `last_accessed_at`, `deprecated` tombstone flag), `paths` (URI namespace: `core://`, `writer://` — domain + path + priority + `disclosure`), `memory_tags` (entity tag values w/ confidence), `memory_gists` and `memory_summaries` (derived L2 rows), `archived_memories` (soft-delete target), `access_log`, plus snapshot tables. The public memory unit has ~8 schema fields (uri, content, priority, title, `disclosure`, domain, gist, trace). No `category`, `themes`, `valid_from/until`, or typed edges.

### Memory Types
Layered by derivation, not by category: **L0** (raw memories) → **L1** (linked clusters) → **L2** (topic summaries / gists). Plus procedural memories (trigger + ordered steps) extracted into a separate table. No episodic/semantic/procedural/reflection taxonomy on the base row.

### Write Path
The headline mechanism is the **Write Guard** (`sqlite_client.py:6374 write_guard()`). Every create/update runs a three-level dedup/reconciliation cascade against existing memories and returns an **ADD / UPDATE / NOOP / DELETE** verdict with a human-readable `guard_reason`:
1. **Semantic**: dual `search_advanced(mode=semantic|keyword, max_results=6)`. Top vector score ≥ **0.92** → `NOOP` (duplicate); ≥ **0.78** → `UPDATE` (near-dup, update in place).
2. **Keyword** (BM25/FTS5): text overlap ≥ **0.82** → `NOOP`; ≥ **0.55** → `UPDATE`.
3. **LLM tiebreak** (optional, `WRITE_GUARD_LLM_ENABLED`, default off): only reached when both lexical/semantic signals are weak.
4. Else → `ADD` ("no strong duplicate signal").
Notably **fail-closed**: if both retrieval channels error, or the embedding provider is degraded on a non-hash backend, it returns `NOOP` rather than risk a spurious ADD (`sqlite_client.py:6474-6481`, `6458-6463`). All other extraction (gist generation via `compact_context`, procedural extraction, L2 layering) is **explicit tool call + human review**, not automatic — so there is no autonomous extraction pipeline.

### Retrieval
Three modes (`keyword`, `semantic`, `hybrid`). Channels live in `backend/db/search/`: `fts5_channel`, `vector_channel`, `entity_channel`, fused by `rrf_fusion`.
- **RRF** (`rrf_fusion.py`): standard `Σ 1/(k+rank)`. **OFF by default** (`RRFConfig.enabled=False`, `DEFAULT_RRF_K=60`), channels = `("fts5","vector")` only, with adaptive-k suggestion `clamp(2·depth, 10, 60)` from an offline calibration harness. Single-channel fallback returns that channel untouched.
- **Entity boost** (`entity_channel.py`): deliberately **not** a co-equal RRF channel (constraint "C6"). Entities are regex-extracted from the query (URIs, dotted paths, `ERR_`/`E\d+` error codes, snake_case, versions, words), intersected with `memory_tags.tag_value`, and applied as a **multiplicative post-fusion boost** (default weight 0.0). Rationale in the docstring: sparse, high-precision signals make better boosters than fresh-ranking contributors and would penalize untagged semantic matches.
- **Intent-aware routing** (`sqlite_client.py:2462 classify_intent()`): rule-based keyword scorer classifies queries as factual / exploratory / temporal / causal / unknown and routes to a `strategy_template` (`factual_high_precision`, `temporal_time_filtered`, `causal_wide_pool`, `exploratory_high_recall`). Optional LLM upgrade when rule confidence is low. No learned reranker in the shipped default — the "reranker" is an external `/rerank` endpoint only in deployment Profiles C/D.
- MMR diversity rerank (`_apply_mmr_rerank`, Jaccard-based) exists on the retrieval path.

**Default embedding is a 64-dim local hash** (Profile B, "zero external dependencies"). Real semantic quality requires Profiles C/D (external API embeddings + reranker).

### Consolidation / Processing
No autonomous sleep cycle. Four **maintenance engines**, all read-only by default, mutations gated behind review tokens:
- **Forgetting** (`forgetting_engine.py`): `simulate_decay()` projects vitality forward N days as a **pure read** (`score·exp(-λ·effective_days)`, λ = ln2/60d, hot-row floor for frequently+recently accessed), produces a candidate queue with `archive/review/keep` recommendations; `approve_archive()` is the only mutator — requires a validated `review_token`, moves the row to `archived_memories` and sets `deprecated=1` (never DELETE).
- **Layering** (`layering_engine.py`): generates L2 summary **drafts** (LLM or deterministic bullet fallback); `persist_draft()` is explicit — no auto-save.
- **Compression** (`compression_engine.py`): read-only "what-if" cascade preview (mild/aggressive/emergency tiers by context-budget utilization) ordered by a `replaceability_score`; pinned/critical/`core` rows exempt.
- **Procedural** (`procedural_engine.py`): extracts trigger+steps drafts; `recommend_for_trigger` surfaces only `human_reviewed` rows.

### Lifecycle Management
- **Vitality decay** + reinforce-on-retrieval (`VITALITY_REINFORCE_DELTA=0.08`, `HALF_LIFE_DAYS=30`, `CLEANUP_THRESHOLD=0.35`).
- **Snapshot + rollback** ("time travel"): every write snapshots; rollback returns HTTP `409` if a newer snapshot exists (content revalidation). Review page renders version diffs.
- **Provenance contract** (constraint "C3") on every derived row: `source_memory_ids`, `source_hashes`, `derivation_method`, `confidence`, `review_state`, `storage_budget_bytes`. `drill_down()` walks provenance and surfaces `live` / `from_archive` / `purged` tombstones and `is_stale` (current source hash ≠ hash-at-derivation) — never silently drops a source.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Write Guard prevents duplicate/redundant writes | Concrete three-tier cascade w/ explicit thresholds + fail-closed (`write_guard()`); gold-set precision/recall = 1.000 | **Plausible mechanism**, but 1.000 is on a tiny hand-built `write_guard_gold_set.jsonl` — not a stress test |
| Intent classification accuracy 1.000 | `intent_gold_set.jsonl` + `test_intent_accuracy_metrics.py` | **Questionable as a headline** — rule-based keyword scorer on a small curated gold set; trivially gameable |
| Retrieval quality HR@10/MRR/NDCG@10/Recall@10 by profile | `docs/EVALUATION_EN.md`: A=0.125, B=0.188, C=0.812, D=0.875 HR@10 | **Not comparable to Somnigraph 85.1 LoCoMo QA** — these are retrieval recall on **16 queries** (SQuAD v2 + BEIR NFCorpus, 8 each) with 200 distractors, not end-to-end QA |
| Offline / zero-dependency default | Profile B: 64-dim local hash embedding, local SQLite | **Validated but weak** — B's NDCG@10=0.164 shows the hash embedding is near-useless; real quality needs external APIs (C/D) |
| Auditable writes (snapshot + rollback) | Snapshot on every write, 409 on stale rollback | **Validated**, clean implementation |
| No auto-delete (safety invariant "C2") | Every engine read-only; only review-token methods mutate; `deprecated=1` tombstone, never DELETE | **Validated** — strongest, most consistent design property |

---

## Relevance to Somnigraph

### What Memory Palace does that Somnigraph doesn't
- **Write-time dedup/reconciliation gate** (ADD/UPDATE/NOOP/DELETE). Somnigraph's `tools.py remember()` dedupes at ~0.9 cosine but has no tiered semantic→keyword→LLM cascade, no UPDATE-in-place verdict, and no fail-closed behavior. This is exactly the "write-path discipline" gap flagged in the Phase 18 source sweep.
- **Snapshot + rollback versioning.** Somnigraph has `valid_from/valid_until` and sleep-detected `revision`/`evolves` edges but no per-write snapshot or one-click rollback with stale-snapshot detection.
- **Provenance contract + staleness detection on derived rows.** Somnigraph's `sleep_nrem.py` merge/archive creates summaries but does not record `source_hashes` to later detect when a gestalt went stale relative to its sources (`drill_down.is_stale`).
- **Simulate-before-mutate forgetting.** Somnigraph applies decay live in `scoring.py`; Memory Palace projects decay forward as a pure read and produces a reviewable candidate queue before any archival.
- **Human-in-the-loop review tokens** on every destructive/derived op. Somnigraph is autonomous by design (this is a philosophy difference, not strictly a gap).

### What Somnigraph does better
- **Learned reranker** (`reranker.py`, 26-feature LightGBM, NDCG=0.7958). Memory Palace's default ranking is RRF-off/hash-embedding; a reranker exists only as an external endpoint in Profiles C/D.
- **Explicit feedback loop** with measured Spearman r=0.70 to GT + UCB exploration. Memory Palace has vitality reinforce-on-access but no per-query utility rating or GT correlation.
- **Graph-conditioned retrieval** (typed edges, PPR expansion, betweenness feature in `scoring.py`). Memory Palace has no memory-to-memory graph; its "entity" signal is a query→tag boost, not a knowledge graph.
- **Autonomous LLM-mediated sleep** (NREM/REM in `sleep_nrem.py`/`sleep_rem.py`). Memory Palace's consolidation is all human-gated preview.
- **Real embeddings by default** (1536d OpenAI / 384d fastembed) vs 64-dim hash.
- **End-to-end QA validation** (85.1 LoCoMo, Opus judge) vs 16-query retrieval-recall.

---

## Worth Stealing (ranked)

### 1. Write Guard: a write-time ADD/UPDATE/NOOP/DELETE reconciliation cascade (Medium)
**What**: Before persisting, run the incoming content as both a semantic and a keyword query against existing memories; decide ADD (no dup), UPDATE (near-dup, ≥0.78 vector / ≥0.55 keyword), NOOP (dup, ≥0.92 vector / ≥0.82 keyword), or LLM-tiebreak — with fail-closed NOOP when signals are unavailable.
**Why**: Somnigraph flagged write-path discipline as the thing LoCoMo/LME leaders win on (Phase 18 sweep: ByteRover, agentmemory, MemPalace). This is a concrete, threshold-explicit implementation template. It also gives `remember()` an "update the existing memory instead of adding a near-duplicate" path it currently lacks.
**How**: In `tools.py remember()`, reuse the existing hybrid retrieval to fetch top-k, apply the two-tier threshold gate, and return a verdict; UPDATE would edit the matched memory's content + bump priority rather than insert. Keep the LLM tier optional. The thresholds are a starting point for a Bayesian sweep.

### 2. Simulate-before-mutate forgetting (Low)
**What**: A pure-read `simulate_decay(days_forward=N)` that projects vitality forward and yields a candidate queue with `archive/review/keep` recommendations and per-row reasons, decoupled from the mutation that acts on it.
**Why**: Somnigraph applies decay silently; a dry-run projection is an honest-accounting surface — you can inspect *what would be forgotten in 30 days* before sleep archives anything, and it makes the decay math auditable.
**How**: Add a read-only projection helper alongside `scoring.py` decay; expose it as a diagnostic (not necessarily an MCP tool). Hot-row floor (frequent+recent access ⇒ protected) is a nice guard against age-only false positives.

### 3. Provenance hashing + staleness detection on derived memories (Low)
**What**: Store `source_memory_ids` + `source_hashes` on every summary/gist; on read, recompute source hashes and flag `is_stale` when a source changed, `purged` when a source is gone.
**Why**: Somnigraph's sleep creates summaries/gestalts that can silently drift out of sync with their edited source memories. Recording source hashes lets sleep re-derive only stale summaries and lets a reader trust-but-verify a gestalt.
**How**: Extend the summary/merge rows written in `sleep_nrem.py` with a JSON `source_hashes` column in `db.py`; add a staleness check to the sleep pass that re-summarizes only drifted clusters.

### 4. "Boost, don't fuse" for sparse precise signals (Note-only)
**What**: The design decision to apply the entity/tag signal as a multiplicative post-fusion boost rather than a co-equal RRF channel, because sparse exact-match signals would penalize untagged high-quality matches if fused equally.
**Why**: Calibration insight if Somnigraph ever adds a structured tag/entity channel. Somnigraph already expresses the same instinct (graph enters via PPR expansion + a reranker *feature*, not as an equal RRF list) — independent convergence worth noting.

---

## Not Useful For Us

- **Multi-client skill packaging** (Claude/Codex/Gemini/Cursor/Windsurf/Antigravity install scripts), React dashboard, Docker profiles, `MCP_API_KEY` auth — product/distribution concerns irrelevant to a single-user research artifact.
- **64-dim hash embedding default** — a deliberate zero-dependency tradeoff; Somnigraph runs real embeddings.
- **Rule-based keyword intent classifier** — Somnigraph's learned reranker already subsumes query-type routing implicitly; a hand-keyword classifier would be a regression.
- **A/B/C/D deployment profiles** — deployment ergonomics, not a memory-quality idea.

---

## Connections

- **Corroborates the Phase 18 write-path thesis** (`docs/sessions/2026-06-28-phase18-source-sweep.md`; `byterover.md`, `agentmemory.md`): Memory Palace independently lands on write-time reconciliation as the quality lever, with an unusually explicit threshold cascade. Fourth independent vote for write-path discipline over retrieval tricks.
- **Distinct from `mempalace.md`** despite the near-identical name (the triage note is correct): MemPalace is verbatim-storage/write-quality; Memory Palace here is the forgetting-engine + snapshot-rollback + 4-maintenance-engine system.
- **Provenance/derivation contract** rhymes with supersession-tracking systems (`memv`, `memos`) — but Memory Palace's `is_stale` hash check is a sharper mechanism than most for detecting summary drift.
- **Review-token human-gating** is the philosophical opposite of Somnigraph's autonomous sleep — a useful contrast when arguing why autonomy needs the feedback loop + GT correlation to be trustworthy.

---

## Summary Assessment

Memory Palace's core contribution is **a disciplined, fail-closed, human-in-the-loop write and lifecycle path**. The Write Guard (semantic→keyword→LLM cascade returning ADD/UPDATE/NOOP/DELETE), the simulate-before-mutate forgetting engine, snapshot/rollback, and the provenance-hash contract with staleness detection are all clean, concrete, and reusable — and they cluster around exactly the axis (write quality, not retrieval) that the Phase 18 sweep identified as where the benchmark leaders actually win. The single most valuable thing to take is the Write Guard's threshold cascade as an implementation template for giving Somnigraph's `remember()` a real dedup/update gate.

What's overhyped: the numbers. The published benchmarks are **retrieval recall on 16 queries** from SQuAD/BEIR with 200 distractors — not comparable to Somnigraph's 85.1% end-to-end LoCoMo QA, and the "precision 1.000 / recall 1.000 / intent accuracy 1.000" figures are on tiny curated gold sets. The shipped default (Profile B) uses a 64-dim hash embedding whose NDCG@10 of 0.164 is near-useless; genuine quality requires external API profiles. And the evidence file's "hybrid = RRF + reranker" glosses over the fact that RRF is OFF by default and the reranker exists only as an external endpoint in the higher profiles.

What's missing relative to Somnigraph: no learned reranker, no explicit feedback loop with measured GT correlation, no memory-to-memory graph or PPR, and no autonomous consolidation. Memory Palace is a well-engineered, safety-first *product* with several genuinely good write-path ideas buried under a thin benchmark story; Somnigraph is a stronger *retrieval* research artifact that would benefit from grafting Memory Palace's write-time reconciliation gate onto its autonomous machinery.
