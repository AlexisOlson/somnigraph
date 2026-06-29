# TrueMemory — Six-layer local-first agent memory with write-path quality gating

*Generated 2026-06-28 by Sonnet agent reading local clone*

---

## Architecture

### Storage & Schema

Single SQLite file (`~/.truememory/memories.db`) with WAL mode. Core tables (from `truememory/storage.py`):

- **`messages`** (line 32): content, sender, recipient, timestamp, category, modality, episode_id, emotional_valence, `embedding_separation` BLOB, `directive` flag, metadata JSON.
- **`messages_fts`** (line 48): FTS5 virtual table with Porter stemmer + unicode61. Kept in sync via INSERT/DELETE/UPDATE triggers (lines 55–68).
- **`entity_profiles`** (line 71): L0 Personality Engram — per-entity traits dict, communication_style dict, topics list, relationships dict, message_count.
- **`entity_style_vectors`** (line 81): char n-gram fingerprints for tone/style matching.
- **`fact_timeline`** (line 90): subject, fact, source_message_id, timestamp, `superseded_by`, valid_from, valid_to, status. Tracks contradiction history.
- **`summaries`** (line 105): period (monthly/entity_monthly/structured_fact), date range, entity, extractive text, key_facts JSON.
- **`episodes`** (line 118): start/end time, message_count, summary. 6-hour gap heuristic.
- **`landmark_events`** (line 127): job changes, moves, launches — regex-detected.
- **`causal_edges`** (line 138): cause_msg_id, effect_msg_id, relationship string, confidence float.
- **`entity_relationships`** (line 149): Dunbar hierarchy (dunbar_layer, strength, last_interaction).
- **`surprise_scores`** (line 170): per-message surprise_score float + fact_count/new_fact_count.
- **`message_clusters`** (line 178): HDBSCAN clusters.

No per-memory decay field. No priority field. No themes array.

### Memory Types

Six named layers (`truememory/engine.py` lines 11–16), but only four are fully implemented:

| Layer | Status | What it does |
|-------|--------|--------------|
| L0 Personality Engram | Active | Entity profiles: communication style, traits, topics, relationships. `personality.py`. |
| L1 Working Memory | Deferred / not implemented | Placeholder only. |
| L2 Episodic | Active | FTS5 keyword search + temporal intent detection + SQL date filtering. `temporal.py`. |
| L3 Semantic | Active | Model2Vec completion vectors + separation vectors; 3-way RRF (k=60). `hybrid.py`. |
| L4 Salience Guard | Active | 13-feature logistic salience filter (l3_weights.json); entity boosting. `salience.py`. |
| L5 Consolidation | Active | Monthly extractive summaries, contradiction detection, surprise index, HDBSCAN clustering, landmark events, causal edges. `consolidation.py`, `predictive.py`. |

Three deployment tiers (`truememory/tier_config.py` lines 25–47):
- **Edge**: Model2Vec (256-dim, ~8MB), 22M-param MiniLM cross-encoder. CPU-only.
- **Base**: Qwen3-Embedding-0.6B (256-dim, ~600MB), 149M-param gte-reranker-modernbert-base.
- **Pro**: Same models as Base + HyDE LLM query expansion (`truememory/hyde.py`). Claims 93% LoCoMo, 92% LongMemEval.

### Write Path

Auto-capture pipeline (`truememory/ingest/pipeline.py`) processes transcripts and passes each extracted fact through an **encoding gate** (`truememory/ingest/encoding_gate.py`):

```
gate_score = 0.25 * novelty + 0.20 * salience + 0.30 * prediction_error
encode if gate_score >= 0.30 AND salience >= 0.10
```
(line 43, encoding_gate.py)

Three signals:
1. **Novelty** (line 337): Compression-based — `(gzip(memory+msg) - gzip(memory)) / gzip(msg)`. Validated AUC 0.788 vs 0.484 for cosine baseline in 120-variant sweep. Replaces cosine inversion (PR #105) because embedding distance is anti-correlated with novelty in conversational data.
2. **Salience** (line 429): Delegates to `truememory.salience` with category boost. Per-category threshold overrides: corrections −0.06, decisions −0.04.
3. **Prediction error** (line 473): Embedding pair-difference — embeds `(msg, nearest_memory)` pair vs `(memory, memory)` self-pair; divergence is PE. AUC 0.730 standalone, 0.816 in combination (200-variant sweep).

Gate degrades open if the PE model fails (line 271). Contradiction/correction facts bypass the gate entirely (line 277). Gate applies only to auto-capture; manual `add()` bypasses it.

On ingest, L0 personality profiles are updated incrementally and an auto-consolidation fires after every 25 adds (configurable via `TRUEMEMORY_AUTO_CONSOLIDATE_EVERY`). Directive memories (`directive=1` flag, line 43 of storage.py) auto-inject at session start and are excluded from encoding gate novelty computation to prevent standing instructions from making real facts look redundant.

### Retrieval

Six-stage pipeline (`truememory/engine.py`):

1. **Query classification** (`query_classifier.py` line 11): regex-routes to temporal, personality, entity, factual, or synthesis mode with per-type RRF weight profiles (e.g., temporal weights: `{"fts": 0.8, "vec": 0.6, "temporal": 2.0, "personality": 0.2}`).

2. **L2 Episodic** (`fts_search.py`): FTS5 with Porter stemmer + temporal SQL filtering when a time reference is detected. `temporal.py` provides `detect_temporal_intent()` and `parse_date_reference()` — handles "early 2025" → `2025-01-01`, "mid 2025" → `2025-05-01`, relative refs ("last month"), ISO strings, ordinals.

3. **L3 Semantic RRF** (`hybrid.py` line 51): Three-way RRF k=60:
   - FTS5 results
   - Completion vector results (Model2Vec cosine via sqlite-vec)
   - Separation vector results (embedding of `[sender] [recipient] [timestamp] content`, weighted 0.8x; disabled if <5 unique senders to avoid uniform ranking in small corpora)

4. **L4 Salience Guard** (`salience.py`): Entity detection from query → proportional boost (+30% max_score if from/to target entity, +20% if mentioned in content, −15% if no connection). Salience floor 0.10; contradiction-source rows are exempt to prevent current short facts from being dropped.

5. **L5 Supplement**: `search_consolidated()` queries summaries + fact_timeline if consolidation data exists. `search_contradictions()` keyword-matches query words against fact subjects and returns current fact + full supersession history.

6. **Cross-encoder reranking** (`reranker.py`): Score fusion `0.7 * rerank_score + 0.3 * rrf_score`. Modality-aware adjustment: detail queries demote episode/fact summaries; synthesis queries boost them. Edge tier: MiniLM 22M. Base/Pro: gte-reranker-modernbert-base 149M.

**HyDE** (Pro only, `hyde.py`): LLM generates hypothetical answer, embeds it as query vector.

**`search_deep()`** (`agentic_search.py`): Multi-round agentic search — runs primary search, assesses sufficiency, generates refined sub-queries, merges results.

### Consolidation / Processing

Background thread fires after every 25 adds (no manual cron). Nine steps at consolidation:
1. HDBSCAN clustering (`clustering.py`)
2. Preference/fact extraction
3. Monthly extractive summaries and per-entity summaries (`consolidation.py`)
4. Contradiction detection: 10+ regex patterns (tech changes, pricing, location, status, schedule, informal corrections, negations, retractions) → fact_timeline updates with `superseded_by` links
5. Structured facts
6. Surprise index: scores each message by how many new facts it introduces that did not appear in preceding messages (`predictive.py`)
7. Episode detection: 6-hour gap heuristic → episodes table
8. Landmark event extraction: job changes, moves, launches via regex → landmark_events
9. Dunbar hierarchy update: entity relationship strength → dunbar_layer (intimate/close/friends/acquaintances)

Transaction hygiene: consolidation writes use SAVEPOINT (`_consolidation_write()` context manager, `consolidation.py` lines 47–80) instead of committing the caller's transaction — prevents the leaked-txn bug documented in issue #649/M-32.

No LLM-mediated synthesis. Summaries are extractive (picks most information-dense messages). No gap analysis for missing knowledge (no REM equivalent).

### Lifecycle Management

No decay. No dormancy. Episodes persist indefinitely. No importance-weighted retention. `fact_timeline` tracks supersession but episodic messages are append-only. Right-to-be-forgotten: `delete_message()` cascades to entity_profiles, style_vectors, entity_relationships, fact_timeline, landmark_events, causal_edges.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| 93.0% LoCoMo (Pro) | README badge + comparison table (line 54); benchmark scripts in `benchmarks/locomo/` | **Unvalidated — cross-vendor non-comparable.** TrueMemory uses its own evaluation harness. Somnigraph benchmarks at 85.1% (Opus judge, GPT-4.1-mini reader); MemMachine at 91.69% (gpt-4.1-mini judge). These use different judges, different result parsers, and different adversarial-question handling. Cannot compare directly. |
| 92.0% LongMemEval (Pro) | README badge; `benchmarks/longmemeval/` | **Unvalidated — same cross-vendor caveat.** MemMachine reports 93.0% LongMemEvalS on a 12-config ablation with GPT-5-mini; TrueMemory's eval setup is undocumented in the README. |
| 76.6% BEAM-1M (SOTA claim) | README badge only; `benchmarks/beam/` | **Unvalidated.** BEAM-1M evaluates very long-context retrieval. No published comparison baseline. No ablation evidence in the repo. Weakest claim. |
| Encoding gate novelty AUC 0.788 | `encoding_gate.py` lines 16–19 (docstring) | **Internal claim, plausible.** 120-variant sweep documented inline. Not reproducible from repo without the sweep data. |
| Competitors scored ≤2/10 on temporal queries | `temporal.py` lines 8–9 | **Assertion without methodology.** Plausible — pure vector search handles temporal queries poorly. No published test set or methodology. |
| PE gate AUC 0.816 in three-signal combination | `encoding_gate.py` lines 32–33 | **Internal claim, plausible.** 200-variant sweep documented inline. |

Cross-vendor benchmark numbers (LoCoMo, LongMemEval) are not comparable across systems without a shared eval harness, shared judge model, and shared adversarial-question handling decisions. Flag all comparisons accordingly.

---

## Relevance to Somnigraph

### What TrueMemory does that Somnigraph doesn't

1. **Write-path quality gating.** `encoding_gate.py` — novelty (gzip compression), salience, and prediction error combine to decide whether auto-captured facts get stored at all. Somnigraph has no ingest-time filter; every `remember()` call writes. The gate is specifically designed for auto-capture pipelines where noise is high. Somnigraph's explicit per-call `remember()` design partially substitutes (the model self-selects what to store), but the encoding gate would help auto-capture paths (source="auto").

2. **Temporal reasoning module.** `temporal.py` — regex-based temporal intent detection, NL date parsing ("early 2025" → ISO, "last month" → SQL date), SQL WHERE clause filtering. Somnigraph has `valid_from`/`valid_until` fields in the schema but no temporal intent detection in the retrieval pipeline (`fts.py`, `scoring.py`). There is no query-time date range filtering.

3. **Separation vectors as a second embedding channel.** Each message gets a second embedding of `[sender] [recipient] [timestamp] content` stored in a separate vec table (`vec_messages_sep`). This encodes speaker-context cues independent of semantic content and participates in 3-way RRF. Somnigraph's embeddings (`embeddings.py`) enrich content with `[category] [themes] summary` but store a single vec. The separation channel is a different trade-off: speaker/time context vs category/theme context.

4. **Query-type routing.** `query_classifier.py` routes to temporal/personality/entity/factual/synthesis modes with per-type RRF weight profiles. Somnigraph's pipeline (`scoring.py`, `fts.py`) does not classify queries — all queries go through the same BM25 + vector + theme + RRF + reranker pipeline.

5. **Surprise index as retrieval signal.** `predictive.py` — each message scored by how many new facts it introduces vs preceding messages. Stored in `surprise_scores` table, used as retrieval boost (`alpha_surprise=0.2`). Somnigraph has priority and Hebbian PMI but no write-time information-theoretic novelty signal.

6. **Personality Engram (L0).** Per-entity communication style, traits, topics, relationship map. Separate `search_personality()` path triggered by strong intent signals. Somnigraph's entity hub nodes (`category="entity"`) capture people but don't infer communication style or trait patterns.

7. **Modality-aware reranking.** Query classified as detail/synthesis/general; summaries boosted for synthesis queries, demoted for detail queries. Somnigraph's reranker (`reranker.py`) has query-proximity features but no query-modality classification.

8. **Transaction SAVEPOINT hygiene.** `_consolidation_write()` in `consolidation.py` prevents consolidation from silently committing the caller's in-flight writes. `sleep_nrem.py` uses separate connections but the pattern is worth adopting if consolidation ever moves inline.

### What Somnigraph does better

1. **Learned reranker with explicit feedback loop.** Somnigraph's 31-feature LightGBM reranker (V5+3b, NDCG=0.8954) is trained on 1885 real-data queries with explicit per-query utility ratings, EWMA aggregation, UCB exploration bonus, and adversarial probing (`select_real_pathology_targets.py`). TrueMemory uses a generic cross-encoder with no domain adaptation and no retrieval quality feedback. TrueMemory's cross-encoder is not trained on the user's own queries.

2. **PPR graph expansion and typed edges.** `scoring.py` — Somnigraph's memory_edges with typed relationships (supports/contradicts/evolves/revision/derivation) built during NREM sleep enable PPR expansion and betweenness centrality as a reranker feature. TrueMemory has `causal_edges` but performs no graph traversal at retrieval time. The graph is structural metadata (cause → effect provenance), not a traversable knowledge graph.

3. **LLM-mediated offline consolidation.** `sleep_nrem.py` — pairwise relationship classification, edge creation, semantic merge/archive decisions via LLM. `sleep_rem.py` — gap analysis, question generation, taxonomy. TrueMemory's consolidation is rule-based (regex contradiction detection, extractive summaries) and does not use LLM-mediated synthesis. The trade-off: TrueMemory is faster and deterministic; Somnigraph can detect paraphrased contradictions and generative summaries.

4. **Per-category exponential decay with feedback reheat.** Somnigraph memories decay on a configurable half-life and reactivate on retrieval. TrueMemory has no decay; episodic memory is permanent and undifferentiated. Long-running users accumulate stale memories with no dormancy detection.

5. **Contradiction detection across paraphrases.** Somnigraph's NREM sleep detects semantic contradictions ("I quit" vs "I resigned"). TrueMemory's contradiction detection is regex-only — cannot catch paraphrased corrections.

6. **Measured retrieval quality signal.** Somnigraph has per-query Spearman r=0.70 with ground truth and live NDCG metrics. TrueMemory has no retrieval quality measurement at runtime; benchmark claims are one-time evaluations.

---

## Worth Stealing (ranked)

### 1. Encoding gate for auto-capture stream (Medium)
**What**: Pre-storage filter using gzip compression novelty + salience + prediction error. Threshold `0.25n + 0.20s + 0.30pe >= 0.30` with salience floor 0.10.
**Why**: Somnigraph's explicit `remember()` design handles deliberate capture but lacks any quality gate for auto-capture (source="auto"). The gate would reduce noise in the auto-capture stream without touching deliberate memories. The gzip compression novelty signal is validated (AUC 0.788 vs cosine baseline) and avoids the cost of an embedding call.
**How**: New module `src/memory/encoding_gate.py`. Add `should_encode(content, category)` check in `impl_remember()` in `tools.py` for source="auto" paths only. The compression novelty check is pure Python stdlib (no new dependencies). Add `novelty_score` as a reranker feature in `reranker.py` — surprise at write time is a useful retrieval signal.

### 2. Temporal intent detection + SQL date filtering (Low)
**What**: Regex-based temporal query classifier in `temporal.py`; detect "early 2025", "last month", ISO dates, relative refs → SQL WHERE on timestamp.
**Why**: Somnigraph has `valid_from`/`valid_until` in the schema but no retrieval-time temporal filtering. A "what did I decide in April?" query currently goes through BM25 + vector search with no date scoping, wasting recall budget on chronologically irrelevant memories.
**How**: Add `detect_temporal_intent(query)` and `parse_date_reference(query)` (< 150 lines, no new dependencies) to `fts.py` or a new `src/memory/temporal.py`. In `impl_recall()` in `tools.py`, if temporal intent is detected, add a `valid_from >= date_start AND valid_until <= date_end` clause to the initial SQL filter before BM25/vector search. This is a pure-SQL filter — zero latency cost.

### 3. Surprise index as reranker feature (Medium)
**What**: At NREM time, score each memory by how many distinct facts it introduces that did not appear in the N memories preceding it chronologically. Store `surprise_score` float in the schema.
**Why**: High-surprise memories encode definitional signal (first mention of a person, a decision, a changed fact). Low-surprise memories (acknowledgments, confirmations) should already rank low but explicit scores give the reranker a training signal for cases where BM25/vector/theme channels get it wrong. This is a write-time feature, not a query-time cost.
**How**: Add `surprise_score REAL DEFAULT NULL` to the `memories` table (`db.py`). Compute in `sleep_nrem.py` during the pairwise pass — extract named entities/numbers/verbs from each memory, compare against rolling fact set from preceding N memories. Add `surprise_score` as a 32nd feature in `reranker.py`'s `_load_memory_meta()`. Requires retrain but expected contribution is meaningful for episodic memories.

### 4. Modality-aware reranking (Low)
**What**: Post-reranker score multiplier based on query type: detail queries (when/who/how many) penalize consolidated summaries by ~0.7x; synthesis queries (explain/describe/why) boost them by ~1.2x.
**Why**: Somnigraph's reranker (`reranker.py`) has 31 features but no query-type signal. A "what was the exact date" query should not return a monthly summary; a "how did the project evolve" query should. The query_classifier result can be used as a post-reranker filter coefficient with no model change.
**How**: Add `classify_query_modality(query) -> Literal["detail", "synthesis", "general"]` (< 30 lines, regex patterns in `query_classifier.py`). In `impl_recall()` after reranking, apply a score coefficient to results with `category in ("semantic", "gestalt")` for detail queries. No retrain needed.

### 5. Transaction SAVEPOINT pattern (Low)
**What**: `_consolidation_write()` context manager using `SAVEPOINT sp; yield; RELEASE sp` with `ROLLBACK TO sp` on error, instead of committing the caller's transaction.
**Why**: `sleep_nrem.py` uses separate connections so the bug doesn't currently apply, but if consolidation ever moves inline (e.g., a write-time mini-consolidation for auto-capture), this prevents a class of silent data corruption.
**How**: Add `_consolidation_write(conn, name)` context manager to `db.py`. Use it if any future `remember()` flow triggers inline consolidation.

---

## Not Useful For Us

**Multi-tier local embedding system (Edge/Base/Pro).** Somnigraph uses OpenAI text-embedding-3-small (1536-dim) for higher quality than TrueMemory's Edge (Model2Vec 256-dim) or Base/Pro (Qwen3 256-dim). The tier-switch machinery, NaN migration for macOS SDPA kernel bugs, WAL coordination for async rebuilds — none of this translates. Not worth the complexity for a single-user system with existing API access.

**Dunbar hierarchy for entity relationships.** TrueMemory's `entity_relationships` with dunbar_layer (intimate/close/friends/acquaintances) models person-to-person closeness in multi-party chat. Somnigraph's use case is single-user personal memory; the relevant relationship structure is memory-to-memory (edges), not person-to-person closeness.

**HDBSCAN clustering.** Somnigraph's NREM sleep does LLM-mediated semantic clustering with generative merge decisions. HDBSCAN would add an embedding-only clustering step producing noisier clusters with no generative summaries. The value is marginal given we already have NREM.

**Cross-encoder reranker as the primary scoring path.** TrueMemory's cross-encoder jointly encodes (query, document) pairs, which is theoretically sound. Somnigraph's LightGBM reranker with 31 retrieval-aware features (session_recency, burstiness, betweenness, PPR, Hebbian PMI) trained on real-data feedback is faster, cheaper, and measurably better on Somnigraph's own retrieval distribution. The cross-encoder has no mechanism for domain adaptation or feedback incorporation.

**AGPL-3.0 license.** No direct code copying. Read for patterns only.

---

## Connections

- **memmachine.md**: MemMachine is the strongest comparable on LoCoMo (91.69% with gpt-4.1-mini, agent mode). TrueMemory claims 93.0% but the eval harness differs. Both use cross-encoder reranking; neither has a learned feedback-adaptive reranker. MemMachine's retrieval agent (ChainOfQuery/SplitQuery routing) is a more principled version of TrueMemory's `search_deep()`.

- **mem0-paper.md**: TrueMemory's comparison table (README line 54) shows Mem0 at 61.4% LoCoMo vs TrueMemory Pro at 93.0%. This is the same Mem0 vs alternative contrast in memmachine.md (Mem0 at 67.13% in MemMachine's eval). Cross-vendor gap is consistent — Mem0's per-message LLM extraction approach lags both ground-truth-preserving and consolidation-based systems on LoCoMo.

- **evermemos.md** and **simplemem.md**: Both share TrueMemory's local-first SQLite architecture. TrueMemory is more engineered than either — proper tier system, encoding gate, benchmark harness — but all three converge on SQLite + FTS5 + vector search as the right stack for single-user memory.

- **a-mem.md**: A-Mem's Ebbinghaus-inspired activation scoring contrasts with TrueMemory's no-decay design. Both have auto-consolidation triggers; A-Mem's forgetting curve would address TrueMemory's stale memory accumulation problem.

- **hindsight.md** and **hindsight-paper.md**: Hindsight also has a structured personality/preference layer. TrueMemory's Personality Engram is more complete (communication style, char n-gram style vectors, Dunbar hierarchy) but Hindsight's disposition-shaped reasoning is more conceptually rigorous.

- **hyde.md**: TrueMemory's Pro tier uses HyDE as analyzed in the source file. Somnigraph analyzed HyDE separately; the same conclusion applies — useful for long-tail queries but adds LLM latency per search.

- **perma.md**: TrueMemory's personality layer (L0) and fact_timeline should handle PERMA's preference-state maintenance tasks reasonably. However, PERMA's cross-domain synthesis tasks (multi-domain Turn=1, current SOTA 0.306) would hit TrueMemory's lack of graph traversal — the same gap Somnigraph aims to close.

---

## Summary Assessment

TrueMemory is the most fully-engineered open-source agent memory library in this survey. It has a coherent six-layer architecture, a real benchmark harness, multi-CLI hook integration (Claude Code, Cursor, Codex CLI, Gemini CLI), and a production-grade write-path quality gate that no other surveyed system has implemented. The temporal reasoning module is exceptional — the most complete implementation seen — and the encoding gate (gzip compression novelty + salience + prediction error) is a genuine contribution grounded in empirical validation (120- and 200-variant sweeps documented inline).

The benchmark claims (93% LoCoMo, 92% LongMemEval, 76.6% BEAM-1M) should be read with the standard caveat: these numbers use TrueMemory's own evaluation harness with unspecified judge configurations. Cross-vendor comparisons without a shared harness are marketing, not science. The BEAM-1M SOTA claim is the weakest — no published comparison baseline, no ablation. The LoCoMo and LongMemEval numbers are plausible given the system's sophistication, but Somnigraph's 85.1% and MemMachine's 91.69% were obtained with different judges and parsers. The gap between TrueMemory's 93% and Somnigraph's 85.1% is not interpretable as-is.

The highest-value items for Somnigraph are the encoding gate and the temporal intent detection. The encoding gate addresses Somnigraph's most documented gap (write-path quality gating listed in the template's "what Somnigraph lacks"). The temporal filter is low-effort and directly addresses a retrieval failure mode (chronologically irrelevant memories crowding recall results). The surprise index would add a 32nd reranker feature with meaningful expected contribution but requires a retrain. The cross-encoder paradigm and multi-tier local embedding system are not transferable — Somnigraph's learned reranker and remote embedding backend are categorically stronger for a system with real feedback data.
