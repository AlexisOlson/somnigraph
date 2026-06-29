# agentmemory — World-record LongMemEval via six-signal hybrid scoring and write-time graph construction

*Generated 2026-06-28 by Sonnet agent reading local clone of https://github.com/JordanMcCann/agentmemory*

---

## Architecture

### Storage & Schema

SQLite (WAL mode, `PRAGMA foreign_keys=ON`). Embeddings stored as raw binary blobs via `struct.pack` — no sqlite-vec extension. FTS5 virtual table with Porter + unicode61 tokenizer for lexical search. Separate `edges` table with `(source_id, relation, target_id)` primary key. No pgvector, no external store.

The in-memory SQLite mode (`:memory:`) is central to the benchmark run: each of the 500 LongMemEval cases gets a fresh store, ensuring zero cross-case contamination. This is a benchmark-only design choice; for production use the path parameter points to a file.

`MemoryNode` schema (from `models.py`): content, kind (12 kinds), tier (working/episodic/semantic), created_at, event_time (bi-temporal), valid_from, valid_until, importance, confidence, decay_rate, embedding (blob), provenance (8 fields), namespace (org/team/agent/session hierarchy), superseded_by, access_count, accessed_at, feedback_correct, feedback_incorrect, tags, version.

`Edge` schema: source_id, target_id, relation (string), weight, metadata, created_at. Relations include: mentions, is_X, works_at, manages, uses, located_in, consolidated_into, superseded_by, contradicts, same_entity.

### Memory Types

Three tiers with 12 kinds:

- **Working**: instruction, observation, scratch — transient, lowest importance prior
- **Episodic**: event, dialogue, action, outcome — raw experiences
- **Semantic**: fact, entity, preference, procedure, belief — consolidated knowledge

Tier is inferred from kind via `_KIND_TIER_MAP` in `models.py`. Preferences get `importance=0.9` prior (highest), entities `0.7`, instructions `0.7`, observations `0.3`, scratch `0.1`.

### Write Path

Extraction via `ExtractionPipeline` in `extraction.py`. Default is rule-based with no LLM at write time — an optional LLM extractor function can be passed in. Three-pass hierarchical extraction:

1. **Pass 1** (single-message): sentence splitting, preference pattern matching (14 regex patterns), verb-presence heuristics, first-person restriction on extended past-tense verbs (ITER-37 finding: restricting to first-person avoids extracting assistant noise)
2. **Pass 2** (adjacency pairs): assistant confirmation phrases ("Got it, you prefer...") extracted as preferences
3. **Pass 3** (QA pairs): question + short answer fused into declarative facts

Temporal grounding at write time (`temporal.py`): `TemporalGrounder` resolves relative date expressions ("last Tuesday", "three months ago", "as of today") to absolute Unix timestamps using zero-dependency stdlib `re` + `datetime`. Every memory gets an `event_time` anchored to the conversation's reference date.

Importance scoring via `ImportanceClassifier` (logistic regression with hand-tuned weights): kind prior, tier prior, urgency keywords, entity density, preference signal, quantitative markers, specificity, negation. No ground truth; these are domain priors.

On each write, `on_write()` in `consolidation.py` runs streaming dedup (cosine radius 0.08, 5 candidates) and contradiction detection (radius 0.2, expanding to 0.4 for update signals). Immediate supersession if update signal detected (`"is now"`, `"no longer"`, `"has changed to"`, etc.).

Knowledge graph auto-construction: `auto_link_node()` in `graph.py` extracts entity triples via regex patterns (`_IS_RE`, `_WORKS_RE`, `_MANAGES_RE`, `_USES_RE`, `_LOCATED_RE`, etc.) or optionally spaCy NER. Creates real `MemoryNode` records for each new entity, persists `mentions`/relation edges to storage. Cross-session entity matching via token Jaccard ≥ 0.6.

### Retrieval

**No RRF. The system uses a weighted linear combination of six signals, not Reciprocal Rank Fusion.** The carsteneu index description "3-way RRF" is a mischaracterization; no RRF appears anywhere in the codebase.

Candidate generation (`_candidates()` in `retrieval.py`):
- HNSW ANN: custom deterministic HNSW (SHA-256 content-based level assignment, `PYTHONHASHSEED=42` subprocess re-exec), `ef_search=100`, up to 200 candidates
- FTS5: `fulltext_search()`, up to 50 additional candidates
- Graph spreading activation: BFS-with-exponential-decay from `context_ids`, `max_depth=2`
- Query expansion: synonym-expanded FTS for each expanded term (up to 30 candidates/term)
- Entity-centric: proper noun regex, entity index lookup, spreading activation (capped at 50)

All candidate ID sets are unioned before scoring.

Six-signal linear scoring (`retrieve()` in `retrieval.py`):
```
score = 0.30 * semantic          # cosine(query_emb, node_emb)
      + 0.12 * lexical           # normalized FTS5 BM25 score
      + 0.18 * activation        # importance + 0.3*recency + 0.1*log(access_count) − decay*age
      + 0.18 * graph             # spreading_activation score from context_ids
      + 0.10 * importance * calibrated_confidence
      + 0.12 * temporal          # Gaussian centered on query time, σ = temporal_width_hours
```

After scoring, optional cross-encoder reranking: `CrossEncoderReranker` (`ms-marco-MiniLM-L-6-v2`) applied to top `limit * 4` candidates. Blend: `0.70 * hybrid_score + 0.30 * CE_sigmoid`. The world record run used this reranker (1,236 calls across 500 cases). Weights are per-query-overridable and can be adapted via `RetrieverWeightAdapter` (EMA-based; requires an external quality function, not wired in the benchmark).

Adaptive weight learning: `RetrieverWeightAdapter` computes EMA of `signal_value * quality_score` per signal. Produces normalized adapted weights when ≥10 queries have been recorded. The quality function must be provided by the caller — not active in the benchmark run.

### Consolidation / Processing

`ConsolidationEngine.run_full_cycle()` in `consolidation.py`, triggered by `ConsolidationScheduler` (background asyncio task, default threshold: 100 episodic nodes, check every 60s):

1. **prune_decayed**: remove nodes with `activation < 0.01`
2. **deduplicate**: cosine radius 0.08 (similarity ≥ 0.92); higher-importance node absorbs the other, `superseded_by` set
3. **promote_working_to_episodic**: after 5 min age + ≥2 accesses or importance ≥ 0.7
4. **consolidate_preferences**: cluster by entity + token Jaccard ≥ 0.4, pick best member, create consolidated preference node
5. **consolidate_episodic_to_semantic**: HNSW clustering (cosine ≥ 0.75, cluster ≥ 3); optional LLM summarizer, default is pick highest-importance member. Quality gate: cosine(summary_emb, centroid) ≥ 0.6; failed summaries are logged but don't supersede source memories
6. **detect_and_resolve_contradictions**: pairwise check via `_structural_check()` (prefix match ≥ 3 words with divergent remainder, or subject-predicate pattern with shared entity, or Jaccard ≥ 0.5 with different content); heuristic resolution by recency × confidence
7. **propagate_confidence_decay**: penalize both sides of unresolved contradicts edges by `base_pen * edge.weight * other.importance`
8. **decay importance**: `ImportanceEvolver` decays nodes stale > 168h toward `baseline=0.3`

This is entirely rule-based and heuristic — no LLM calls in the default consolidation path. The LLM summarizer is optional.

### Lifecycle Management

Bi-temporal schema: `created_at` (ingestion time) vs `event_time` (when the event occurred). `valid_from`/`valid_until` for fact expiration. Losers in contradiction resolution get `valid_until = event_time` of the winning update. Expired facts excluded from retrieval by default (`include_expired=False`).

Decay via `ImportanceEvolver`: exponential decay toward `baseline=0.3` for nodes stale > 168h. Activation (recency + frequency + importance − decay_rate*age) falls below 0.01 → pruned by consolidation. No dormancy detection, no per-category configurable half-lives.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| 96.20% on LongMemEval_S (481/500) | `longmemeval_results_opus6.json` (per-case results included in repo), full run log `fullrun_opus6.log`, legitimacy audit in `LEGITIMACY.md` | **Strong**. Single deterministic run enforced by PYTHONHASHSEED=42 subprocess re-exec. `USE_DIRECT_CONTEXT=False` enforced via hard assert. Standard LME J-score formula. GPT-4o judge with `seed=42`. |
| Beats PwC Chronos (95.60%) by +0.60pp | Published Chronos arXiv score cited in README table | **Plausible**. Comparison is valid (same LME_S benchmark, same judge standard) but caveated: different generator (Claude Opus 4.6 vs Chronos's "enhanced config"). Score comparisons are generator-confounded. |
| ~92% token reduction vs full-context | README: ~170K tokens/year vs 19.5M+ for full-context | **Unvalidated**. Rough annual estimate, not a controlled measurement. Benchmark run used 4.3M tokens for 500 cases (~8.6K avg/case). No methodology for the annual projection. The carsteneu "92%" figure is consistent with this estimate but doesn't constitute evidence. |
| 3-way RRF fusion | Claimed in carsteneu index; also described as "RRF_K=60" in a third-party article | **Wrong**. No RRF in the codebase. Six-signal weighted linear sum with optional cross-encoder blend. Candidate sources (ANN + FTS + graph) are unioned, not reciprocal-rank fused. |
| World record as of March 2026 | Self-reported; Issue #29 in xiaowu0162/LongMemEval repo referenced | **Plausible at time of submission** (March 26 2026). MemMachine (published April 2026) subsequently reports 93.0% on LME_S under a different configuration. agentmemory's 96.2% appears to remain the highest single-pass real-retrieval score on LME_S as of the analysis date. |
| 100% abstention accuracy (30/30) | Per-category results in README | **Strong**. Correctly identifying all 30 unanswerable questions is the clearest signal of the system's confidence calibration quality. |

Note on the LME 95.2 figure in the carsteneu index: this is incorrect. The actual score is 96.2% (96.20%). The 95.2% may have been confused with the Chronos score (95.60%) or an intermediate iteration result.

---

## Relevance to Somnigraph

### What agentmemory does that Somnigraph doesn't

**1. Write-time entity graph construction.** `auto_link_node()` extracts entity nodes and typed relation triples at ingest time via `EntityRelationExtractor` (regex or spaCy). Real `MemoryNode` records are created for entities; edges are immediately persisted to the `edges` table. Somnigraph's graph builds during NREM sleep — entities and relationships aren't available for graph-conditioned retrieval until after the next sleep pass. agentmemory's approach means every new memory is immediately connected to the graph.

**2. Bi-temporal schema with validity windows.** Every `MemoryNode` has separate `event_time` (when the event occurred) and `created_at` (ingestion time), plus `valid_from`/`valid_until` for explicit fact expiration. Contradiction losers get `valid_until` anchored to the winner's `event_time`, creating a clean temporal record. Somnigraph has `valid_from`/`valid_until` in concept but doesn't use them systematically — the contradiction detection in `sleep_nrem.py` marks superseded memories but doesn't anchor validity to event time. This is the `db.py` / `tools.py` gap.

**3. Temporal grounding at write time.** `TemporalGrounder` in `temporal.py` resolves relative date language ("last Tuesday", "three months ago") to absolute Unix timestamps at ingestion, not retrieval. Somnigraph's memories record `created_at` but don't normalize temporal references within content. This was already recommended in simplemem.md as the highest-leverage low-effort steal for Somnigraph.

**4. Cross-encoder reranking as a retrieval component.** `CrossEncoderReranker` applies `ms-marco-MiniLM-L-6-v2` to the top `limit * 4` candidates as a blended signal (70% hybrid + 30% CE). This is a generic cross-encoder, not domain-adapted — but it contributed to closing the final 0.6pp gap between agentmemory and PwC Chronos. Somnigraph's LightGBM reranker serves a similar purpose but uses 31 domain-specific features and was trained on real-data queries. The CE approach requires no training data; the LightGBM approach is more precise but requires ground truth.

**5. Streaming write-time dedup + contradiction detection.** `on_write()` runs cosine radius checks and update-signal detection on every write, not in a batch sleep pass. This means near-duplicate memories never accumulate: if you say "I use Python" and later "Actually, I now use Go", the older fact is immediately superseded without waiting for NREM. Somnigraph's dedup is sleep-only; between sleep passes, near-duplicates can accumulate and dilute retrieval.

**6. Activation as a distinct scoring signal.** `activation = importance + 0.3*recency + 0.1*log(access_count) − decay_rate*age_hours`. This is computed per-node at query time and weighted at 0.18 in retrieval. Somnigraph's reranker includes `fb_time_weighted` and `session_recency` features but these are distinct signals over feedback data, not a unified activation formula combining all three dimensions.

**7. Namespace hierarchy for multi-tenancy.** Four-level namespace (org/team/agent/session) with containment semantics for filtering. Somnigraph is single-user and has no namespace concept.

### What Somnigraph does better

**1. Learned reranker with measured feedback loop.** Somnigraph's 31-feature LightGBM reranker (NDCG=0.8954, +0.0921 over RRF formula baseline) was trained on 1,885 real-data queries with per-query float utility ratings. The GT correlation is measured: per-query Spearman r=0.70. agentmemory's `RetrieverWeightAdapter` is EMA-based, requires an external quality function the caller must supply, and wasn't active in the benchmark run. The CE blend (70/30) is a fixed heuristic. Somnigraph's reranker learns which signals matter for which queries from data; agentmemory's weights are hand-set constants. See `src/memory/reranker.py` for feature implementation.

**2. PPR graph expansion vs spreading activation.** Somnigraph uses Personalized PageRank for graph expansion (power-iteration on 2-hop subgraph, `scoring.py`). agentmemory uses BFS-with-exponential-decay (`spreading_activation()` in `graph.py`). PPR is globally normalized and properly accounts for graph structure; spreading activation is a local heuristic that can over-activate high-degree hub nodes. PPR betweenness centrality is also a reranker feature in Somnigraph (`src/memory/reranker.py`).

**3. RRF fusion of three channels vs weighted linear sum.** Somnigraph fuses BM25, vector, and theme results via RRF (k=14, Bayesian-optimized), which is robust to score distribution differences across channels. agentmemory's linear weighted sum is sensitive to score scale differences between its six signals — if one signal's range expands (e.g., graph activation on a dense graph), it can dominate. RRF's rank-based fusion is more robust to this. See `src/memory/fts.py` and `src/memory/scoring.py`.

**4. LLM-mediated sleep pipeline.** Somnigraph's NREM sleep detects relationships between memories through pairwise LLM classification (`sleep_nrem.py`), creating typed edges (supports, contradicts, evolves, revision, derivation) with confidence and linking context. REM sleep performs gap analysis and generates questions about missing knowledge. agentmemory's consolidation is rule-based: cosine thresholds for dedup and clustering, regex and Jaccard for contradiction detection. No gap analysis, no question generation.

**5. BM25 field weighting.** Somnigraph's FTS5 uses `bm25(memory_fts, 13.3, 5.7)` weighting — summary field 13x, themes 5.7x (tuned to wm38). agentmemory's FTS5 uses plain `porter unicode61` with no field weighting. The field weighting allows Somnigraph to emphasize structured theme labels over raw content for thematic retrieval. See `src/memory/fts.py`.

**6. Theme channel as third RRF component.** Somnigraph has a separate theme retrieval channel using explicit theme arrays on each memory (procedural/calibration/gotcha etc.), contributing as a third RRF arm. agentmemory has tags but they're not a distinct retrieval channel.

**7. Typed semantic edges with linking context.** Somnigraph's edges from sleep carry `linking_context`, `linking_embedding`, and typed relations (supports/contradicts/evolves/revision/derivation). These are embedded and searchable. agentmemory's graph uses structural relation strings (works_at, manages, contradicts) without linking context. Somnigraph's typed edges carry richer semantic signal for PPR traversal.

**8. Explicit per-query feedback loop.** Somnigraph's `recall_feedback()` takes per-query float utility (0-1) plus durability, EWMA-aggregated with UCB exploration bonus. Measured GT correlation r=0.70. Hebbian PMI boost strengthens edges between co-retrieved memories. agentmemory's `feedback_correct`/`feedback_incorrect` counters support calibrated confidence but aren't wired to retrieval weight adaptation in production.

---

## Worth Stealing (ranked)

### 1. Temporal grounding at write time (Low)

**What**: `TemporalGrounder` in `temporal.py` — zero-dependency stdlib resolver for date expressions. Converts "last Tuesday", "three months ago", "as of today" to absolute Unix timestamps anchored to the conversation reference date. Every memory gets an `event_time` field.

**Why**: Temporal reasoning accounts for 133/500 questions in LongMemEval (agentmemory got 96.2% on that category). Somnigraph currently stores `created_at` but doesn't normalize relative time expressions within content. A query for "what happened last March" can't match if the stored content says "last March" without grounding. SimpleMem.md already recommended this; agentmemory has the most complete zero-dependency implementation of any system analyzed.

**How**: Add `TemporalGrounder` call in `tools.py` `impl_remember()` before storing. Resolve all time expressions in the memory `content` + `summary` relative to `datetime.now()` at write time. Store the earliest resolved timestamp as `valid_from` and the latest as `valid_until` if a range is detected. The temporal.py implementation can be ported directly with minor adaptation.

### 2. Write-time streaming dedup + update-signal contradiction detection (Low)

**What**: `on_write()` in `consolidation.py` — cosine radius check (0.08 for duplication, 0.2 for contradiction) on every write. Update signals ("is now", "no longer", "has changed to", "as of today") trigger immediate supersession without waiting for sleep.

**Why**: Between sleep passes, Somnigraph accumulates near-duplicates (especially for frequently-referenced topics) and stale contradicted facts. agentmemory's approach prevents accumulation at source, which reduces retrieval noise. The update-signal detection heuristic (`_detect_update_signal()` in `consolidation.py`) uses layered regex patterns with confidence scoring (1.0 for explicit corrections, 0.8 for temporal supersession, 0.5 for weak signals) — a clean design worth adopting.

**How**: Add an `on_write_check()` step in `tools.py` `impl_remember()`. Call `embed_text(content)` and run a sqlite-vec radius query against existing memories (threshold ~0.12 cosine distance). If an update signal is detected (port `_detect_update_signal()`), find the nearest semantic match and set its `valid_until = NOW`. This is 20-30 lines and doesn't require sleep. Update `db.py` to index `valid_until` for efficient filtering — already done in agentmemory's schema.

### 3. Consolidation quality gate for episodic→semantic promotion (Low)

**What**: After generating a cluster summary in `consolidate_episodic_to_semantic()`, compute cosine(summary_emb, cluster_centroid). If below threshold (0.6), log audit but do NOT supersede source memories. The sources remain retrievable even if the summary is poor.

**Why**: Somnigraph's NREM sleep merges memories based on pairwise LLM classification; if the LLM misclassifies a merge, the source memories are lost. A centroid-similarity quality gate (computable without an LLM call) ensures the merge doesn't destroy evidence if consolidation quality is low. Cheap to add, defends against sleep-pipeline noise.

**How**: In `sleep_nrem.py`, after generating a merged memory: embed the merged text, compute cosine against centroid of source embeddings, check against a threshold (0.6 is a reasonable starting point). If below threshold, store the merged memory but don't archive the sources. Add a `consolidation_quality` field to the merged memory metadata for later inspection.

### 4. Activation signal combining recency + access frequency + decay (Medium)

**What**: `activation = importance + 0.3 * (1/(1 + recency_hours)) + 0.1 * log1p(access_count) - decay_rate * age_hours`. Computed at query time per node.

**Why**: Somnigraph's reranker has `session_recency` and `fb_time_weighted` features, but these are feedback-derived signals. A unified activation signal combining importance, recency, frequency, and decay into a single number could be a useful reranker feature that works without any feedback data (especially for new memories with no feedback history). The `session_recency` importance instability (V5+1b→V5+2b→V5+3b: 188→445→356) suggests the reranker is trying to capture this kind of signal indirectly.

**How**: Compute `activation` at retrieval time for each candidate (negligible cost), add as `activation_score` feature to `reranker.py`'s `_load_memory_meta()`. Include in next GT collection cycle to measure its weight. This is a small addition to an existing pipeline.

---

## Not Useful For Us

### Custom HNSW implementation

agentmemory builds a custom HNSW (`ann_index.py`, 376 lines) with SHA-256 content-based level assignment for determinism. Somnigraph uses sqlite-vec for vector search (already integrated, tested, and maintained). Switching would require replacing the entire vector infrastructure for marginal benefit. The determinism fix (SHA-256 levels, PYTHONHASHSEED=42) is relevant only for benchmark reproducibility, not production use.

### Namespace hierarchy

Four-level org/team/agent/session namespacing is designed for multi-tenant SaaS. Somnigraph is single-user, single-agent. The namespace system adds schema complexity and query overhead that don't apply.

### EMA adaptive weight adapter

`RetrieverWeightAdapter` requires a `quality_evaluator_fn` to be passed in by the caller. It computes signal-quality EMA correlations and normalizes them into adjusted weights. This is a much weaker learning system than the LightGBM reranker already in production — it has no cross-validation, no held-out evaluation, and requires a quality oracle the benchmark doesn't even provide. The LightGBM reranker subsumes this.

### Rule-based contradiction detection

`_structural_check()` in `consolidation.py` uses prefix matching (≥3 shared words with divergent remainder) and entity Jaccard (≥0.5) to detect contradictions. This generates false positives on semantically similar but not contradictory content. Somnigraph's LLM-mediated pairwise contradiction classification in NREM sleep is more precise. The rule-based approach's value is speed (no LLM call), but Somnigraph's sleep already batches LLM calls for this purpose.

### GDPR deletion and audit trail

agentmemory has a full GDPR deletion pipeline (`gdpr.py`), audit log table, and `DeletionReceipt` dataclass. Somnigraph is a personal single-user system; GDPR machinery doesn't apply.

---

## Connections

**simplemem.md**: Both recommend temporal grounding at write time as the highest-leverage low-effort steal. agentmemory's `TemporalGrounder` is the most complete zero-dependency implementation in the corpus — more thorough than SimpleMem's ISO-8601 anchoring. Both systems independently arrived at this as a critical mechanism for temporal reasoning accuracy.

**memmachine.md**: Both use cross-encoder reranking as a final precision pass. MemMachine uses AWS Cohere rerank-v3-5-0 (commercial, higher quality) with multi-query concatenation; agentmemory uses ms-marco-MiniLM-L-6-v2 (open-source, lighter) with a 70/30 blend. MemMachine's LME score (93.0% on a slightly different config) is notably lower than agentmemory's 96.2%, suggesting agentmemory's write-time extraction quality and temporal reasoning are doing more work than the cross-encoder.

**memv.md**: memv explicitly credits SimpleMem for write-time temporal normalization. agentmemory independently converged on the same pattern but went further — bi-temporal schema, validity windows, and event_time propagation through contradiction resolution. Three systems in the corpus have now independently identified temporal grounding as critical.

**a-mem.md**: A-Mem uses Ebbinghaus-inspired activation scoring (strength + recency + frequency). agentmemory's `activation` formula is structurally similar: importance (strength) + recency term + frequency term − decay. Both treat activation as a live scalar property of each memory node, computed from access history. Convergent design.

**hindsight.md** and **cognee.md**: Both do write-time knowledge graph construction. agentmemory's entity/relation extraction is regex-only (or optional spaCy), which is faster but less accurate than LLM-based triple extraction. The write-time graph is a design choice that trades accuracy for latency: you get some graph immediately, vs Somnigraph's more accurate but delayed sleep-built graph.

---

## Summary Assessment

agentmemory's 96.20% on LongMemEval_S is a credible benchmark result with strong legitimacy controls — the run log, per-case results JSON, and hard assert against oracle access are all included in the repo. The score reflects a system specifically engineered for LongMemEval: per-case SQLite in-memory stores prevent contamination, token budgets are tuned per question type, the HNSW determinism fix addressed ±3-case noise, and 46 iteration cycles of targeted optimization drove the 68% → 96.2% improvement arc.

The core retrieval is not RRF (contrary to the carsteneu description) but a six-signal weighted linear combination with optional cross-encoder blending. The weights are hand-tuned constants, not learned. The EMA adaptive adapter exists but requires an external quality oracle and wasn't active in the benchmark run. The graph expansion uses spreading activation (BFS-with-decay) rather than PPR, and the consolidation pipeline is rule-based rather than LLM-mediated. Compared to Somnigraph, agentmemory has a shallower learning infrastructure but a more aggressive write-time pipeline: temporal grounding, streaming dedup, immediate contradiction resolution, and entity graph construction all happen at ingest time rather than in a sleep pass.

The most important steal for Somnigraph is temporal grounding at write time — resolving relative date expressions to absolute timestamps at the point of memory creation. SimpleMem.md already recommended this; agentmemory has the most complete implementation. The update-signal streaming contradiction detection is the second steal: between sleep passes, stale contradicted facts currently persist in Somnigraph's active recall pool. The consolidation quality gate is a low-cost defensive addition for sleep. None of these require architectural changes — they're additions to the write path and sleep pipeline.

The 96.2% score is the highest published real-retrieval score on LongMemEval_S as of the analysis date. Whether it reflects production-quality memory for long-running personal use (Somnigraph's domain) rather than benchmark-optimized episodic recall (LongMemEval's domain) is an open question. The per-case fresh store design means there's no measurement of how the system behaves with hundreds of accumulated memories, decay, or evolving facts across months — exactly what Somnigraph's sleep pipeline, feedback loop, and decay model are designed for.
