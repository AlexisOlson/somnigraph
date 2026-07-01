# shodh-memory - Rust, no-LLM cognitive memory: Hebbian graph + spreading activation (PPR) + power-law decay, in one offline binary

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

A ~132k-line Rust project (`varun29ankuS/shodh-memory`, Apache-2.0, v0.2.0, ~215 stars). It ships as a single ~17MB offline binary and positions itself as a **neuroscience-grounded, no-LLM-in-the-loop** memory substrate for AI agents *and robots*. The design bet is the mirror image of the LoCoMo/LME leaders (ByteRover, agentmemory) we profiled: instead of putting the LLM on the write path, shodh puts **no LLM anywhere** in store/recall — all "intelligence" is algorithmic (local embeddings, mathematical decay, learned linear weights, graph spreading activation).

### Storage & Schema
- **Storage**: RocksDB (KV + O(1) content-hash index) for memories; Tantivy 0.25 (BM25) for full-text; a custom **Vamana/DiskANN** HNSW graph with product quantization (`src/vector_db/{vamana,spann,pq}.rs`) for vectors. jemalloc allocator.
- **Embeddings**: MiniLM-L6-v2 (384-dim) via ONNX, downloaded once (`src/embeddings/minilm.rs`). Circuit-breaker + downloader wrap it.
- **Memory unit**: the `Experience` struct — **45+ fields**. Core (content, experience_type, entities, ner_entities, embeddings, tags, temporal_refs) plus a `RichContext` with 12 sub-contexts: Conversation, User, Project, Temporal, Semantic, Code, Document, Environment, **Emotional** (valence/arousal/emotion), **Source** (source_type/credibility/verified/source_chain), Episode. Plus 26 robotics fields (robot_id, geo_location, heading, sensor_data, reward, is_failure, root_cause…) and multimodal image/audio/video embedding slots.

### Memory Types
Three-tier hierarchy (Cowan's working-memory model): **Working (≤100 items) → Session (≤100MB) → Long-Term (RocksDB)**, with tier promotion/demotion. Edges carry their own tiers (L1 Working / L2 Episodic / L3 Semantic) with per-tier trust weights and decay rates. `ExperienceType` enum (Task, etc.) discriminates content.

### Write Path
Genuinely rich and **entirely local** (`src/handlers/remember.rs`):
- Parallel **NER (TinyBERT) + YAKE keyword** extraction via `spawn_blocking` (`src/embeddings/{ner,keywords}.rs`).
- **SHA-256 content-hash dedup** — O(1) RocksDB index, identical content never stored twice; batch path also dedups within-batch (`seen_content`).
- **Temporal fact extraction** + entity grounding into the knowledge graph **at ingest time** (`SemanticFactStore`, `temporal_facts.rs`). Entity nodes and relationship edges are built on write, not deferred.
- Segmentation engine (`memory/segmentation.rs`) with a `DeduplicationEngine`.

### Retrieval
Three modes exposed (`semantic`, `associative`, `hybrid`):
- **BM25 + vector → RRF fusion** (`src/memory/hybrid_search.rs`, `RRFusion::fuse`, formula `weight/(k+rank)`, configurable bm25/vector/graph weights, `RRF_K_HYBRID_FUSION`).
- **Associative / graph** = **spreading activation** (`memory/graph_retrieval.rs::spreading_activation_retrieve_with_stats`, 2446 lines). Seeds activation from query focal entities (POS-heuristic + optional neural NER, ACT-R salience-weighted "attention budget"), then spreads. **Default path is Personalized PageRank** (`SHODH_PPR=1` default) which the code explicitly frames as "the convergent, mass-conserving form of spreading activation," replacing the hand-rolled BFS spread. Includes **HippoRAG node-specificity restart weighting** (`SHODH_PPR_SPECIFICITY` default on) and optional HippoRAG-2 **passage PPR** (`SHODH_PPR_PASSAGE`, off). Density-adaptive semantic/graph/linguistic weights; degree normalization to tame hubs; edge-tier trust + LTP boosts modulate spread.
- **"Layer 4.9" ontological rerank**: a type-aware additive boost (`ONTOLOGICAL_RERANK_BOOST=0.08`, cap 0.25), *not* a learned or cross-encoder reranker.
- **Feedback fusion** (`src/relevance.rs`): a 7-dim `LearnedWeights` (semantic, entity, tag, importance, momentum, access_count, graph_strength) updated by **online gradient descent** on helpful/not-helpful signals, normalized to sum 1; plus a per-memory **feedback momentum EMA** folded into the fused score.

### Consolidation / Processing
`src/handlers/consolidation.rs` — **pattern-triggered** (not fixed-interval; "PATTERN-TRIGGERED REPLAY", PIPE-2) maintenance: memory **replay**, tier consolidation, decay application, and **entity-entity Hebbian reinforcement** for replayed memories. No LLM; it is a mechanical replay + edge-strengthening loop. Also: retroactive/proactive **interference detection** (SHO-106) for contradictory memories.

### Lifecycle Management
- **Hybrid decay** (`src/decay.rs`, SHO-103): exponential for t < 3 days (consolidation phase), then **power-law tail** for long-term retention (`A(t)=A_cross·(t/t_cross)^(-β)`), citing Wixted & Ebbesen. Explicitly designed to avoid the exponential "cliff."
- **Multi-scale LTP** (`RelationshipEdge::strengthen`, `graph_memory.rs`): Hebbian `w_new = w_old + η(1-w_old)·boost`, with burst (5+/24h → 2× protection), weekly (3+/wk×2wk → 3×), and full (10+ total → 10× slower decay) potentiation tiers, plus tier promotion L1→L2→L3.
- Explicit `forget` (7 criteria), `MemoryRevision` history, `external_id` upsert semantics.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| No LLM in store/recall loop | Code confirms: NER+YAKE+embeddings+PPR+decay are all algorithmic; LLM parser is an *optional* mode (`query_parsing/llm_parser.rs`, default rule-based) | **Validated** for memory ops. (The LoCoMo *MCQ eval* `benchmarks/locomo_mc10_eval.py` still calls an external LLM as the answer *reader*.) |
| Hybrid BM25+Vector+RRF+**reranking** | RRF present; `hybrid_search.rs` module doc advertises a "Cross-Encoder" stage that **does not exist** — `src/bin/recall_eval.rs:87` states plainly "there is no cross-encoder." Only the ontological additive boost | **Overstated**: no learned/cross-encoder reranker. RRF + type boost only |
| PPR / spreading activation lifts recall | Inline CI notes: recall@10 ALL 0.6853→0.6976 with PPR default-on; specificity +0.0066 multi_hop | **Plausible but self-reported** retrieval recall@k from CI runs, not an audited benchmark |
| Power-law long-term decay is biologically accurate | `decay.rs` implements exp→power-law crossover with Wixted citations | **Validated as implemented**; biological-accuracy claim is a modeling choice, not measured |
| Three-tier cognitive architecture (central claim) | Working→Session→Long-Term with promotion/demotion; edge tiers | **Validated** — this is real, not marketing |
| Robotics / ROS2 first-class | Zenoh transport, 26 robotics fields, haversine spatial recall, fleet discovery | **Validated but orthogonal** to Claude-memory use |
| Benchmark performance | `BENCHMARKS.md` is **all latency** (763ns entity lookup, 2-5ms vector, 150-250ms store) | **No end-to-end QA accuracy published.** Not comparable to Somnigraph's 85.1% LoCoMo QA |

---

## Relevance to Somnigraph

### What shodh-memory does that Somnigraph doesn't
- **Write-time entity/graph construction.** Somnigraph builds its graph only during NREM sleep (`scripts/sleep_nrem.py`); shodh grounds entities (local NER) and builds relationship edges **at ingest** in `remember.rs`. This is exactly the "real-time graph construction" gap named in Somnigraph's own limitations list.
- **Power-law long-term decay tail.** Somnigraph's `decay.py` is pure per-category exponential; shodh's exp→power-law crossover directly targets the exponential "cliff."
- **Graduated, cadence-based decay protection (multi-scale LTP).** Somnigraph's reheat-on-access is effectively binary; shodh distinguishes burst vs weekly vs sustained access and grants different protection factors.
- **HippoRAG node-specificity restart weighting** on PPR. Somnigraph runs PPR over sleep-detected edges but doesn't weight the restart vector by node specificity/IDF.
- **Fully offline, no-API-key operation** and a robotics/ROS2 substrate (irrelevant to us but a real capability).

### What Somnigraph does better
- **Learned reranker.** Somnigraph's 26-feature LightGBM reranker (`reranker.py`, NDCG=0.7958, +6.17pp over formula) is a categorically stronger scoring stage than shodh's 7-weight gradient-descent linear fusion + type boost (`relevance.rs`) — and shodh's advertised cross-encoder is vaporware.
- **Measured feedback→GT correlation.** Somnigraph's explicit 0-1 utility ratings with EWMA + UCB and Spearman r=0.70 vs ground truth beat shodh's helpful/not-helpful gradient nudges, which have no reported validation.
- **LLM-mediated sleep consolidation.** Somnigraph's NREM/REM pipeline (typed edge classification, contradiction/evolves detection, gap analysis, question generation) extracts semantic structure shodh's mechanical replay cannot.
- **Audited end-to-end QA.** Somnigraph has 85.1% LoCoMo QA with an Opus judge and a multi-hop vocabulary-gap failure analysis; shodh publishes only latency and self-reported retrieval recall@k.

---

## Worth Stealing (ranked)

### 1. Hybrid exponential→power-law decay tail (Medium)
**What**: For long-term retention switch from pure exponential to a power-law tail after a crossover point (~3 days), preserving important memories against the exponential "cliff" (`decay.rs`, Wixted-cited).
**Why**: Somnigraph's per-category exponential decay is the single place where the biological-fidelity story is weakest, and the "cliff" concern is already live in project notes. A power-law long tail is a small, well-grounded change.
**How**: In `src/memory/decay` (or wherever `decay_rate` is applied), add a crossover: exponential below `t_cross`, then `value_at_crossover · (t/t_cross)^(-β)` above it, with `β` tuned per category. Ablate on the feedback logs before adopting.

### 2. Graduated, cadence-based decay protection (multi-scale LTP) (Low)
**What**: Distinguish burst (N accesses in 24h), weekly (sustained over weeks), and total-count access patterns, each granting a different decay-slowdown factor — rather than a single reheat-on-access bump.
**Why**: Somnigraph's reheat is binary; a memory accessed in a tight burst vs steadily over a month means different things about durability. Cheap to add to the existing access bookkeeping.
**How**: Track activation timestamps per memory (shodh keeps a small ring); compute a protection multiplier feeding `decay_rate`. Maps onto the existing durability signal.

### 3. HippoRAG node-specificity restart weighting for PPR (Medium)
**What**: Weight the PPR restart/seed vector by node specificity (IDF-like), so rare, discriminative seed entities carry more restart mass than ubiquitous hubs (`SHODH_PPR_SPECIFICITY`).
**Why**: Somnigraph already runs PPR expansion (`scoring.py`); this is a drop-in refinement that shodh's CI notes claim helps multi-hop (+0.0066) with zero regression — directly relevant to Somnigraph's known multi-hop vocabulary-gap ceiling.
**How**: When building the PPR restart vector in `scoring.py`, scale each seed's mass by a specificity term (e.g. inverse graph degree or corpus IDF of the entity/theme). A/B on the LoCoMo multi-hop split.

### 4. Write-time lightweight entity tagging (note-only / High)
**What**: Local NER at ingest to pre-seed the graph, rather than deferring all graph construction to sleep.
**Why**: Fills the named "no real-time graph construction" gap. But Somnigraph *deliberately* defers to LLM-mediated sleep for higher-quality typed edges, so this is a design-tension note, not a clear win — a lightweight write-time entity tag could bridge the gap between ingest and the next sleep pass without displacing sleep's richer typing.

---

## Not Useful For Us
- **Robotics stack** (Zenoh/ROS2 transport, 26 robotics fields, haversine spatial recall, fleet discovery) — a different product entirely.
- **Multimodal embedding slots** (image/audio/video) and the 45-field `Experience` struct — over-engineered scaffolding for a single-user Claude text-memory system.
- **Bundled GTD** (todos/projects/reminders, ~20 of the 37 MCP tools) — app features, not memory research.
- **The three physical tiers** (Working/Session/Long-Term with MB caps) — Somnigraph's single SQLite store + decay achieves the same recency gradient without tier-migration machinery.

---

## Connections
- **Convergent with our PPR direction** and with HippoRAG: shodh independently lands on Personalized PageRank as "the convergent form of spreading activation," corroborating Somnigraph's PPR-over-graph bet — but applies it at *retrieval* time over a *write-built* graph, where Somnigraph applies it over a *sleep-built* graph.
- **Opposite pole from the write-path-quality leaders** (see `ai-memory-comparison.md`, ByteRover/agentmemory): those win LoCoMo by putting the LLM on the *write* path; shodh removes the LLM entirely and competes on algorithmic retrieval. Both agree the write path is where structure is won — shodh just insists it can be won without an LLM (local NER + fact extraction).
- **Decay lineage**: same Wixted/Anderson-Schooler grounding Somnigraph's decay design cites, but shodh actually ships the power-law tail Somnigraph left on the table.
- **Feedback loop**: a weaker cousin of Somnigraph's learned reranker + UCB — linear gradient-descent weights instead of LightGBM, no GT-correlation validation.

---

## Summary Assessment

shodh-memory's core contribution is an **existence proof that a serious, cognitively-grounded memory system can run fully offline with no LLM in the store/recall loop** — local MiniLM embeddings, Tantivy BM25, a custom Vamana vector index, entity NER at ingest, Personalized-PageRank spreading activation, multi-scale Hebbian LTP, and a hybrid power-law decay, all in a 17MB Rust binary. The engineering is real and dense (132k lines, extensive CI ablation history), and several biological primitives are implemented more faithfully than Somnigraph's.

The single most valuable thing to take is the **hybrid exponential→power-law decay tail** (with graduated cadence-based protection as a rider), plus the **HippoRAG node-specificity restart weighting** for our existing PPR — all small, low-risk refinements to `decay.py` and `scoring.py`, none of which require adopting shodh's architecture.

What's overhyped or missing: the **"cross-encoder reranking" advertised in the hybrid-search module doc does not exist** (the code says so itself), so shodh's scoring stage is materially weaker than Somnigraph's learned reranker. And critically, **shodh publishes no end-to-end QA accuracy** — `BENCHMARKS.md` is entirely latency, and the LoCoMo numbers surfacing in CI comments are self-reported *retrieval* recall@k (~0.70 R@10), not comparable to Somnigraph's audited 85.1% LoCoMo QA (and below Somnigraph's own R@10). The evidence file's feature audit is accurate and fair (it correctly flags the absent cross-encoder by listing only RRF, and correctly frames `schemaFields=6` as API-surface vs a 45-field internal struct); the sharpest correction it *doesn't* make is that **none of shodh's headline benchmark cells are end-to-end QA** — the whole comparison lives at the retrieval-recall and latency layer.
