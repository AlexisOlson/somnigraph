# agentmemory (rohitg00) - Broad TS agent-memory engine on iii-engine; 3-way RRF, LLM-extracted graph, generic cross-encoder (default off)

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

> **Name-collision note.** This is **NOT** the system in `agentmemory.md` (that one is JordanMcCann/agentmemory — a single-file Python system, six-signal linear scoring, 96.2% LongMemEval **QA**). This is **rohitg00/agentmemory**: a ~39k-LOC TypeScript engine built on the `iii-engine` / `iii-sdk` runtime, 53 MCP tools, 12 lifecycle hooks, multi-agent, viewer UI. Different codebase, different author, different design. The two share only a name. `agentmemory.md` is left untouched.

## Architecture

### Storage & Schema

No direct SQLite. Storage is a **key-value abstraction (`StateKV`) provided by iii-engine's StateModule** (file-backed, described as SQLite under the hood). Everything is namespaced KV scopes (`src/state/schema.ts` `KV.*`): `memories`, `observations(sessionId)`, `summaries`, `semantic`, `procedural`, `graphNodes/graphEdges`, plus ~40 more scopes (leases, signals, mesh, sketches, facets, sentinels, crystals, lessons, audit, retention, accessLog...). The breadth of scopes is the tell: this is a product surface, not a retrieval kernel.

Vector search is a **custom in-memory index** (`src/state/vector-index.ts`, 167 lines) — brute-force cosine over Float32Arrays, not sqlite-vec/HNSW. BM25 is a hand-rolled inverted index with Porter stemming + synonym expansion (`src/state/search-index.ts`, `src/state/stemmer.ts`, `src/state/synonyms.ts`). Both indexes are rebuilt/persisted separately from the KV store.

`Memory` schema (`src/functions/remember.ts`): id, createdAt, updatedAt, type (pattern/preference/architecture/bug/workflow/fact), title (first 80 chars), content, concepts[], files[], sessionIds[], `strength` (default 7), `version`, `parentId`, `supersedes[]`, `isLatest`, `sourceObservationIds[]`, optional `agentId`, `project`, `forgetAfter` (TTL). Bi-temporal is absent — only createdAt/updatedAt.

### Memory Types

Four-tier consolidation lifecycle: **working → episodic → semantic → procedural**. Observations are the raw episodic unit; `SemanticMemory` (fact + confidence) and `ProceduralMemory` (name/steps/trigger) are produced by the consolidation pipeline. `Memory` (the `mem::remember` unit) is a separate user/agent-authored crystallized entry with a 6-value `type` enum.

### Write Path

`mem::remember` (`remember.ts`): validates, picks a type, then does **supersession dedup via token Jaccard > 0.7** against all latest memories (`jaccardSimilarity` in `schema.ts` — word-set overlap, tokens length > 2). On match: bump `version`, set `parentId`/`supersedes`, mark old `isLatest=false`, fire a `mem::cascade-update` trigger. Then index into BM25 and the vector index. **No LLM at write time, no salience/quality gate, no entity extraction on this path.** `strength` is a hardcoded 7. Project-scoped supersession guard prevents cross-project clobber.

A separate `DedupMap` (`dedup.ts`) is only a 5-minute TTL hash cache to drop duplicate tool-call captures (SHA-256 of sessionId+tool+input) — hook-noise suppression, not semantic dedup.

Graph construction is **observe-time and LLM-mediated**: `mem::graph-extract` (`graph.ts`) calls `provider.compress(GRAPH_EXTRACTION_SYSTEM, ...)`, parses `<entity>`/`<relationship>` XML into nodes/edges, dedups by `type|name` name-index, and maintains a top-degree **snapshot** for viewer/stats scale (much of `graph.ts`'s 1000 lines is scale-hardening for 25k–75k-node corpora hitting the iii payload ceiling). Requires an LLM provider key; absent one, no graph is built.

### Retrieval

**3-way weighted RRF** (`src/state/hybrid-search.ts`, `RRF_K = 60`). Channels: BM25 (`limit*2`), custom vector ANN (`limit*2`), and **graph spreading** (`GraphRetrieval.searchByEntities`, depth 2, + `expandFromChunks` off the top-5 vector hits). Entities for the graph arm come from regex `extractEntitiesFromQuery`. Fusion:

```
combined = wB * 1/(60+bm25Rank) + wV * 1/(60+vectorRank) + wG * 1/(60+graphRank)
```

with hand-set weights `bm25=0.4, vector=0.6, graph=0.3`, **renormalized to drop channels that returned nothing**. RRF_K=60 and all weights are untuned constants. After fusion: `diversifyBySession` (max 3 per session — an MMR-lite dedup), enrich, then an **optional cross-encoder rerank**.

Reranker (`src/state/reranker.ts`): `Xenova/ms-marco-MiniLM-L-6-v2` (quantized, local) over `query [SEP] title narrative`, replaces combinedScore with the CE score. **Gated behind `RERANK_ENABLED === "true"` — OFF by default.** Generic, untrained, no domain features. `searchWithExpansion` runs the triple-stream over query + reformulations + temporalConcretizations + entity extractions and max-merges (`smart-search.ts` builds the expansion via the LLM provider).

### Consolidation / Processing

`mem::consolidate-pipeline` (`consolidation-pipeline.ts`), **LLM-mediated, gated behind `CONSOLIDATION_ENABLED` or any provider key**. Tiers: (1) **semantic** — merge ≥5 recent session summaries via `provider.summarize`, parse `<fact confidence=..>` XML, dedupe by lowercased fact string; (2) **reflect** — trigger `mem::reflect` clustering; (3) **procedural** — extract `<procedure>` from patterns seen ≥2×; (4) **decay** — multiplicative strength decay. XML-tag parsing of LLM output (same pattern as the graph extractor). This is convergent-in-spirit with Somnigraph's sleep (offline, LLM-mediated) but coarser: no typed edge classification, no pairwise contradiction judgment, no gap/question generation.

### Lifecycle Management

Versioned supersession chains (`parentId`/`supersedes`/`isLatest`/`version`). TTL via `forgetAfter`. Decay in two places: consolidation's multiplicative `strength * 0.9^periods` (floor 0.1), and a richer **Ebbinghaus retention model** in `retention.ts`: `retention = min(1, salience * exp(-lambda*ageDays) + reinforcementBoost)`, where `reinforcementBoost = sigma * Σ (1/daysSinceAccess)` over **all** access timestamps (a spaced-repetition curve), `salience` from type-weights + access bonus, tier thresholds hot/warm/cold. `lambda=0.01, sigma=0.3` default. Auto-forget/evict functions exist (`auto-forget.ts`, `evict.ts`).

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| "LongMemEval 95.2%" | `benchmark/LONGMEMEVAL.md`: R@5=95.2% (BM25+Vector) | **Retrieval recall, not QA.** R@5 = "does any gold session appear in top-K." No answer generation, no judge. The repo states this plainly ("we do NOT claim these as LongMemEval scores"). **Not comparable to Somnigraph's 85.1 LoCoMo QA.** |
| Headline number uses the "3-way RRF + reranker" architecture | Same file: run is **BM25+Vector only** | **No.** The 95.2% excludes the graph channel and the cross-encoder (default-off). The marquee retrieval features do not participate in the marquee number. |
| "BM25+Vector (95.2%) nearly matches pure vector (96.6%)" | Own table vs MemPalace self-report | **Plausible but self-limited.** Same 384-d all-MiniLM embeddings; BM25 adds +9pp over BM25-only. Honest internal ablation. |
| 3-way RRF fusion | `hybrid-search.ts` `RRF_K=60`, three arms | **True (code exists).** But weights/K are untuned constants and the graph arm only fires when regex finds query entities. |
| Cross-encoder reranker | `reranker.ts` ms-marco-MiniLM | **True but default-off**, generic, untrained. Not a learned/feature reranker. |
| 4-tier consolidation, Ebbinghaus decay, KG | code present in `consolidation-pipeline.ts`, `retention.ts`, `graph.ts` | **True**, but consolidation + graph both require an LLM provider key to do anything. |
| "92% fewer tokens" | README badge | **Unvalidated** projection, not a controlled measurement (evidence file concurs). |

**Sharpest correction:** the "95.2 LME" cell is **R@5 retrieval recall on BM25+Vector only** — not end-to-end QA, and it does not exercise the graph channel or the reranker that headline the architecture. The carsteneu evidence file already flags it as retrieval-only; the extra correction is that even the *retrieval config* benchmarked is the two-channel subset, not the advertised 3-way+rerank stack.

---

## Relevance to Somnigraph

### What agentmemory (rohitg00) does that Somnigraph doesn't

- **Observe-time LLM graph extraction** (`graph.ts`): entities/relations built as memories arrive (given a provider key), vs Somnigraph's sleep-time graph. Gap lives in `scripts/sleep_nrem.py` (edges only appear post-sleep). Caveat: their extraction is single-pass XML-tag parsing with no typed-edge taxonomy or contradiction judgment — shallower than NREM's pairwise classification.
- **Versioned supersession chains at write time** (`parentId`/`supersedes`/`version`/`isLatest`): immediate, between-sleep supersession. Somnigraph supersedes during sleep; `db.py`/`tools.py` don't maintain an explicit version chain. (Same steal already flagged from the *other* agentmemory and simplemem.)
- **Spaced-repetition reinforcement in decay** (`retention.ts`): `Σ 1/daysSinceAccess` accumulates every access into a strengthening curve, richer than a single reheat-on-access bump. Maps to Somnigraph's per-category decay (`scoring.py`/`db.py`).
- **Session-diversity cap in fusion** (`diversifyBySession`, max 3/session): cheap MMR-lite before rerank. Somnigraph has no source/session diversity constraint.
- **Breadth**: 53 MCP tools, 12 hooks, multi-agent leases/signals/mesh, viewer UI, Obsidian export, audit trail, privacy redaction. Product surface Somnigraph deliberately doesn't have.

### What Somnigraph does better

- **Learned reranker with a feedback loop.** Somnigraph's 31-feature LightGBM (NDCG≈0.895) trained on real utility ratings, per-query Spearman r=0.70. agentmemory's reranker is a generic ms-marco cross-encoder, **off by default**, untrained, no feedback signal. `reranker.py` has no counterpart here.
- **Tuned fusion.** RRF k=14 Bayesian-optimized + tuned BM25 field weights (`bm25(13.3, 5.7)`) vs untuned `RRF_K=60` and hand-set channel weights.
- **PPR graph-conditioned retrieval** (`scoring.py`) with betweenness as a reranker feature, vs BFS spreading activation + degree snapshots.
- **Richer sleep**: typed edges (supports/contradicts/evolves/revision/derivation), pairwise contradiction classification, REM gap analysis + question generation — vs XML-tag fact/procedure extraction.
- **Real vector infra** (sqlite-vec) vs a brute-force in-memory cosine index that won't scale.

---

## Worth Stealing (ranked)

### 1. Spaced-repetition reinforcement term in decay (Low) — *marginal, consider*
**What**: `retention = salience·exp(-λ·ageDays) + σ·Σ(1/daysSinceAccess)` over the full access history (`retention.ts` `computeRetention`/`computeReinforcementBoost`).
**Why**: Somnigraph reheats on access, but (per the brief) as a bump/reset rather than an accumulating spaced-repetition curve. Summing recency-weighted accesses gives a smoother "frequently-revisited memories resist decay" signal — a memory accessed 5× over a month stays hotter than one accessed once, independent of the most-recent touch.
**How**: In the decay pass (`scoring.py`/`db.py`), replace/augment reheat with `σ·Σ 1/daysSinceAccess` from the access log. Small, self-contained. **Honest caveat**: unlikely to move any measured metric; Somnigraph's existing reheat already captures the intent. Note-only unless a decay-tuning experiment is already open.

*(No other mechanism here clears the bar. Query expansion is already covered by the L5b BM25-damped IDF keyword expansion; session-diversity capping is low-value for a single-user store; the graph/consolidation are shallower than Somnigraph's; the reranker and fusion are strictly weaker.)*

---

## Not Useful For Us

- **iii-engine coupling.** The whole system is a plugin surface over an external runtime (`iii-sdk`, StateModule KV, trigger bus). Nothing ports without that substrate.
- **Custom in-memory BM25 + vector index.** Somnigraph already has sqlite-vec + FTS5; the brute-force cosine index is a scaling liability, not an upgrade.
- **Generic default-off cross-encoder.** Strictly subsumed by the learned LightGBM reranker.
- **Multi-agent leases/signals/mesh, viewer, Obsidian export, GDPR/audit, 53 MCP tools.** Product breadth irrelevant to a single-user research artifact.
- **Untuned RRF (k=60) + hand-set channel weights.** Somnigraph's tuned fusion is ahead.

---

## Connections

- **agentmemory.md (JordanMcCann)** — same name, unrelated code. Both do write/observe-time graphs and versioned supersession, but JordanMcCann's is a tighter research kernel (six-signal scoring, temporal grounding, 96.2% **QA**); rohitg00's is a broad product on iii-engine. Do not conflate the numbers: 96.2 (JordanMcCann) is QA; 95.2 (rohitg00) is R@5 retrieval recall.
- **simplemem.md / memv.md** — versioned supersession and write-time freshness recur again here; convergent signal that between-sleep supersession is worth having in Somnigraph.
- **a-mem.md** — Ebbinghaus/activation decay with recency+frequency reinforcement is the same family as `retention.ts`; another independent arrival at spaced-repetition-style strengthening.
- **Write-path-quality thesis (Phase 18 sweep: ByteRover/MemPalace/agentmemory-JMc)** — rohitg00 is a **counter-example on the write path**: its `mem::remember` has *no* quality/salience gate (hardcoded strength=7, Jaccard-only dedup), and its best number comes from a plain BM25+Vector retrieval subset. Consistent with "retrieval leaders win on write-path quality" only in the negative: this system's breadth doesn't buy retrieval-science depth.

---

## Summary Assessment

rohitg00/agentmemory is an ambitious, broad **agent-memory product** — 53 MCP tools, 12 hooks, multi-agent coordination, a viewer, Obsidian export, LLM-mediated consolidation and graph — built as a plugin layer over the `iii-engine` runtime. As engineering breadth it is impressive; as a source of retrieval-science ideas for Somnigraph it is thin. Its retrieval kernel is standard-and-untuned (RRF k=60, hand-set weights, a custom brute-force vector index, a generic default-off cross-encoder), and every "advanced" subsystem (graph, consolidation) is gated behind an LLM provider key and shallower than Somnigraph's sleep pipeline.

The single most important thing to take is a **calibration on its headline number**: "LongMemEval 95.2%" is **R@5 retrieval recall on a BM25+Vector subset** — not end-to-end QA, and not even the 3-way+rerank stack it advertises. The repo is commendably explicit about this ("we do NOT claim these as LongMemEval scores"), and the carsteneu evidence file already tags the cell retrieval-only; the added nuance is that the benchmarked config omits the graph channel and reranker entirely. It is **not comparable** to Somnigraph's 85.1 LoCoMo QA.

The only mechanism worth even a second look is the spaced-repetition reinforcement term in `retention.ts` (accumulate recency-weighted accesses into decay resistance), and that is marginal against Somnigraph's existing reheat. Verdict: **MAYBE**, leaning SKIP — one low-value decay tweak to revisit only if a decay-tuning experiment is already on the bench; nothing that changes the roadmap.
