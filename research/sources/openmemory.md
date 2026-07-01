# OpenMemory — Regex-classified 5-sector memory with a multi-vector waypoint graph, all heuristic (no LLM extraction, no benchmarks)

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

OpenMemory (CaviraOSS, Apache-2.0, ~4.2k stars, TypeScript-first with a thinner Python port) is a self-hosted local memory server for LLM/IDE clients over MCP + REST. The core engine is `packages/openmemory-js/src`. The headline abstraction is **HSG = Hierarchical Sector Graph** (`memory/hsg.ts`, 1334 lines) — five cognitive "sectors" over a per-memory multi-vector store plus a "waypoint" similarity graph. There is a separate **temporal knowledge graph** subsystem (`temporal_graph/`) for subject-predicate-object facts with `valid_from/valid_to`. README banner says the project is being rewritten; `main` reflects the pre-rewrite state.

### Storage & Schema
- SQLite by default (`core/db.ts`, 942 lines); pluggable Postgres/Valkey vector backends (`core/vector/`).
- `memories` table: `id, user_id, project_id, segment, content, simhash, primary_sector, tags, meta(JSON), created_at, updated_at, last_seen_at, salience, decay_lambda, version, mean_dim, mean_vec(BLOB), compressed_vec, feedback_score`. (Evidence lists 13 "core" fields; the actual insert in `hsg.ts:1186` binds ~19 columns including `simhash`, `feedback_score`, `segment`, `project_id` — the evidence undercounts.)
- Separate `vectors` table (one row **per sector per memory** — multi-vector) and `waypoints` table (`src_id, dst_id, user_id, project_id, weight, created_at, updated_at`).
- Memories are bucketed into "segments" that rotate at `env.seg_size` — a coarse sharding for cache locality, not semantic.

### Memory Types
Five sectors defined in `sector_configs` (`hsg.ts:50-118`): **episodic / semantic / procedural / emotional / reflective**, each with its own `decay_lambda` (0.001–0.02), a scoring `weight` (0.8–1.3), and a hand-written **regex pattern list**. Classification (`classify_content`, `hsg.ts:249`) is pure regex pattern-counting: sum `matches × weight` per sector, pick the argmax as `primary`, add secondaries above a 30%-of-max threshold. No LLM, no learned classifier (roadmap item admits "current is regex-based").

### Write Path (`add_hsg_memory`, hsg.ts:1126)
1. **Near-dup dedup**: compute a 64-bit **SimHash** over canonical tokens (`compute_simhash`); if an existing memory is within Hamming distance ≤ 3, skip insert and instead boost the existing memory's salience by 0.15 (`hsg.ts:1139-1151`). *(Contradicts the evidence file's "dedup ❌".)*
2. Chunk long text (512-token windows, 50 overlap; `utils/chunking.ts`).
3. Regex-classify into primary + additional sectors.
4. **Extractive "essence" summary** (`extract_essence`, hsg.ts:368) — heuristic sentence scoring (position, headers, dates, money, proper nouns, action verbs) to compress content when `use_summary_only` is set.
5. Embed **once per sector** (`embedMultiSector`), store each vector; compute a softmax-over-sector-weight `mean_vec` (`calc_mean_vec`).
6. Create a single "waypoint" edge to the highest-cosine existing memory (`create_single_waypoint`, threshold 0.75, self-loop fallback).
- No LLM extraction, no salience/quality gate on the content itself (initial salience is just `0.4 + 0.1×#secondary_sectors`).

### Retrieval (`hsg_query`, hsg.ts:812)
Per-query pipeline:
1. Regex-classify the query into sectors; derive per-sector fusion weights (query's primary sector gets 1.1–1.5×, others 0.5–0.8×).
2. Embed query per sector; run **per-sector vector cosine search** (`vector_store.searchSimilar`, k×3 candidates each).
3. **Adaptive expansion**: `adapt_exp = ceil(0.3·k·(1−avg_top_sim))` — pull more candidates when top similarities are weak; `high_conf = avg_top ≥ 0.55`.
4. **Waypoint graph expansion** — BFS over the similarity graph (`expand_via_waypoints`, multi-hop, weight-decayed ×0.8/hop, cutoff 0.1) — **skipped entirely when `high_conf`** (confidence-gated graph traversal).
5. **Keyword channel** (only when `tier === "hybrid"`): `keyword_filter_memories` (`utils/keyword.ts`) combines exact-phrase match + char-trigram/bigram overlap + a **BM25 term** (`compute_bm25_score`). *(Contradicts evidence "fulltext/BM25 ❌" — but the BM25 uses hardcoded `corpus_size=10000, avg_doc_length=100`, so IDF is a constant, not a real corpus statistic. It's degenerate BM25-flavored TF weighting, not an FTS index.)*
6. Composite score per candidate (`compute_hybrid_score`, hsg.ts:451): `sigmoid(0.35·boosted_sim + 0.2·token_overlap + 0.15·waypoint + 0.1·recency + 0.2·tag_match + keyword_boost)`, where `boosted_sim = 1−e^(−3·sim)`. A **cross-sector resonance penalty** multiplies similarity by a hand-set sector-affinity matrix (`sector_relationships` / `dynamics.ts` `SECTORAL_INTERDEPENDENCE_MATRIX`) when the memory's sector differs from the query's. *(Note: the evidence file's formula `0.6·sim + 0.2·salience + 0.1·recency + 0.1·waypoint` is from stale docs and is wrong — salience is NOT a ranking term.)*
7. **Z-score normalization** across the top-`eff_k` candidates, then truncate to k.
8. **On-retrieval side effects**: EWMA `feedback_score = 0.9·old + 0.1·score` (feeds the *model's own score* back — no external signal); Hebbian **co-activation buffer** strengthens waypoints between co-retrieved pairs with a temporal-proximity factor (async, `setInterval`); retrieval-trace salience reinforcement + spreading-activation salience propagation to linked nodes.

### Consolidation / Processing (`memory/reflect.ts`)
Rule-based, interval-timer "reflection" (`run_reflection`): fetch up to 100 memories, **cluster** by Jaccard token similarity > 0.8 within the same sector (`cluster`), emit a templated summary memory (`"N {sector} pattern: ..."`) into the reflective sector, mark sources `consolidated`, boost their salience ×1.1. No LLM, no edge typing, no contradiction handling. *(Contradicts evidence "clustering ❌".)*

### Lifecycle Management
Three overlapping decay implementations (a redundancy smell):
- `hsg.calc_decay`: per-sector `salience·e^(−λ·days) + reinforcement_term`, optional segment-index damping.
- `memory/decay.ts`: **tiered hot/warm/cold** decay (λ 0.005/0.02/0.05) with progressive **vector + summary compression** — cold memories get dimensionally down-pooled (1536→…→64) and summaries shortened to save space.
- `ops/dynamics.ts`: "dual-phase" decay (fast term + slow consolidation term). Also holds an elaborately-named spreading-activation / energy-threshold retrieval path that appears largely unused by the main `hsg_query`.
- Temporal KG auto-evolution: a new fact with the same subject+predicate closes the prior fact's `valid_to` (supersession, no contradiction flagging). Explicit delete via REST/CLI (not exposed as an MCP tool).

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| 5-sector cognitive memory with adaptive per-sector decay | `sector_configs`, `calc_decay` — real, but sectors assigned by **regex**, not semantics | Plausible mechanism, brittle assignment |
| "Multi-signal hybrid" retrieval | `compute_hybrid_score` fuses sim + overlap + waypoint + recency + tag + keyword | Real, but weights hand-set, not tuned; no learned reranker |
| Temporal knowledge graph / point-in-time truth | `temporal_graph/` with valid_from/to, auto-evolution | Real, separate subsystem; no contradiction detection |
| Dedup / clustering / reflection | SimHash dedup + Jaccard clustering both present in code | Real — **evidence file marks both ❌ (wrong)** |
| "Works with any LLM" | 6 **embedding** providers; classifier is regex | Misleading — no LLM in extraction/classification path |
| Benchmarks (LoCoMo/LongMemEval/etc.) | None published anywhere | Confirmed absent — **no comparable QA numbers** |
| Landing-page stats (94.7% retention, 36ms latency) | Marketing page only, no methodology | Unvalidated marketing |

---

## Relevance to Somnigraph

### What OpenMemory does that Somnigraph doesn't
- **Write-time near-dedup (SimHash + Hamming ≤ 3)** with boost-instead-of-insert. Somnigraph has no write-path dedup/quality gate (`tools.py` `impl_remember`; STEWARDSHIP explicitly names "write-path quality gating" as a gap). This is the one concretely transferable mechanism.
- **Multi-vector per memory** (one embedding per sector) with sector-conditioned query fusion weights. Somnigraph uses a single enriched embedding (`embeddings.py`).
- **Confidence-gated retrieval depth**: skip graph expansion when top-sim is high; expand more candidates when it's low. Somnigraph always runs PPR expansion (`scoring.py`).
- **Real-time waypoint graph + real-time Hebbian co-activation** — the graph is built at write/read time, not during an offline sleep pass. Somnigraph builds edges only during NREM sleep (`sleep_nrem.py`).
- **Progressive lossy compression of cold memories** (dimensionality reduction) — a storage-scaling idea Somnigraph doesn't attempt.

### What Somnigraph does better
- **Extraction/classification**: OpenMemory's sector routing is regex pattern-counting; Somnigraph doesn't hard-classify at write but the reranker learns category interactions from data. Regex classification is fragile (a memory about "feeling productive" is forced into `emotional`).
- **Learned reranking**: Somnigraph's 26-feature LightGBM reranker (NDCG 0.7958) vs OpenMemory's hand-set sigmoid weights with no tuning or ablation.
- **Real feedback loop**: Somnigraph uses explicit per-query user utility (Spearman r=0.70 w/ GT). OpenMemory's `feedback_score` EWMA feeds the model's *own* output back — a self-reinforcing loop with no external signal (the exact anti-pattern Somnigraph's proactive-injection design guards against).
- **Consolidation quality**: Somnigraph's sleep is LLM-mediated (typed edges, contradiction classification, gap analysis). OpenMemory's reflection is Jaccard-cluster + string-template, no edge types, no contradiction handling.
- **Evidence discipline**: Somnigraph reports LoCoMo QA 85.1%; OpenMemory publishes no benchmarks at all.

---

## Worth Stealing (ranked)

### 1. SimHash write-time near-dedup with boost-on-collision (Low/Medium)
**What**: On write, compute a 64-bit SimHash over canonical tokens; if an existing memory is within Hamming ≤ 3, skip the insert and bump the existing memory's salience/access instead of creating a duplicate (`hsg.ts:322-364, 1139-1151`).
**Why**: Directly addresses Somnigraph's named "write-path quality gating" gap and the Phase 18 finding (write-path quality, not retrieval, is what LoCoMo/LME leaders win on). Cheap, deterministic, no LLM call. Reduces the near-duplicate churn that degrades reranker GT and Hebbian PMI.
**How**: Add a `simhash` column in `db.py`; in `tools.py` `impl_remember`, compute SimHash and query for candidates sharing a high-order prefix (or scan a bounded recent window), reject/merge on Hamming ≤ threshold, reheat the survivor instead. Keep it advisory (log merges) before making it silent.

### 2. Confidence-gated retrieval depth (Low)
**What**: Gate graph expansion on retrieval confidence — skip PPR/adjacency expansion when the top vector similarities are already strong, and widen the candidate pool only when they're weak (`hsg.ts:874-888`).
**Why**: PPR expansion in `scoring.py` runs unconditionally; on high-confidence queries it adds latency and can inject noise. A cheap `avg_top_sim` gate could cut cost without hurting recall on easy queries.
**How**: Compute mean top-N fused score in `scoring.py` before expansion; skip/limit PPR when above a tuned floor. Validate on the existing 1032-query GT that R@10/NDCG are unchanged on easy queries and only expansion cost drops.

---

## Not Useful For Us

### Regex sector classification
Hand-written pattern lists are brittle and locale/vocabulary-bound; Somnigraph deliberately avoids hard write-time categorization in favor of learned ranking. Not worth importing.

### Three overlapping decay engines + spreading-activation retrieval path
`hsg.calc_decay`, `decay.ts` tiered decay, and `dynamics.ts` dual-phase decay coexist with much unused, verbosely-named machinery (`SECTORAL_INTERDEPENDENCE_MATRIX_FOR_COGNITIVE_RESONANCE`, `performSpreadingActivationRetrieval`). This reads as AI-generated accretion, not a design. Somnigraph's single per-category exponential decay is cleaner.

### Self-referential `feedback_score` EWMA
Feeding the model's own ranking score back as "feedback" is a self-reinforcement trap, not a real learning signal. Somnigraph's explicit-utility loop is strictly better; do not adopt.

---

## Connections
- **Convergent Hebbian co-retrieval**: OpenMemory strengthens waypoint edges between co-retrieved memories with a temporal-proximity factor at read time — the same mechanism as Somnigraph's Hebbian co-retrieval PMI (built during sleep). Independent arrival at co-activation → edge weight is mild corroboration that the idea is sound; the difference is online (OpenMemory) vs offline batch (Somnigraph).
- **Write-path-quality thesis**: The SimHash dedup + cluster-reflection reinforce the Phase 18 source-sweep conclusion (see the ByteRover / agentmemory / MemPalace analyses) that write-time discipline, not retrieval cleverness, is where leaders separate — even though OpenMemory itself publishes no benchmarks to prove it.
- **Multi-sector vs single enriched embedding**: contrast with Somnigraph's `embeddings.py` (content+category+themes+summary in one vector); OpenMemory splits into per-sector subspaces — an untested alternative, not obviously better.

---

## Summary Assessment

OpenMemory's core contribution is a **coherent-sounding cognitive framing** (five decaying "sectors" over a multi-vector store with a self-strengthening similarity graph) implemented **entirely with heuristics**: regex classification, hand-set fusion weights, SimHash dedup, Jaccard-cluster reflection, and degenerate BM25 (constant IDF). There is no LLM in the extraction/classification path, no learned reranker, no feedback signal that isn't the model grading itself, and — critically — **no published benchmark of any kind**. The landing-page numbers (94.7% retention, 36ms latency) are marketing with no methodology. Popularity (4.2k stars) tracks packaging and IDE integrations, not retrieval quality.

The single most useful thing for Somnigraph is the **SimHash write-time near-dedup** (boost-on-collision instead of insert) — a cheap, deterministic mechanism that maps onto a named Somnigraph gap (write-path quality gating) and the Phase 18 write-path-quality thesis. A secondary, minor idea is **confidence-gated expansion depth**. Everything else Somnigraph already does better, does differently on purpose, or should actively avoid (the self-referential feedback loop, the triple-decay accretion).

Sharpest evidence correction: the carsteneu evidence file is **wrong on three feature cells** — it marks dedup ❌, clustering ❌, and fulltext/BM25 ❌, but the code contains SimHash dedup (`hsg.ts:1139`), Jaccard clustering (`reflect.ts:29`), and a BM25 keyword channel (`utils/keyword.ts:69`, gated on `tier==="hybrid"`, though its IDF is a hardcoded constant). It also reproduces a **stale scoring formula** (`0.6·sim + 0.2·salience + …`) that does not match the actual `scoring_weights` in code, where salience is not a ranking term at all. None of these corrections make OpenMemory competitive — but they matter for an honest feature comparison. Verdict: **MAYBE** (one revisit-if idea: SimHash dedup).
