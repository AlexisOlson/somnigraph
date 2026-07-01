# omega-memory — Local-first MCP memory for coding agents; deep multi-phase hybrid pipeline with write-time contradiction/supersession and an off-the-shelf cross-encoder reranker

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

OMEGA (`omega-memory`, Apache-2.0, ~148 stars, created 2026-02-13) is a persistent-memory MCP server for AI coding agents. SQLite + sqlite-vec + FTS5 at `~/.omega/omega.db`, fully local (ONNX embeddings on CPU), pip-installable, with a Free/Pro split (Pro adds multi-agent coordination, entity registry, LLM router, knowledge base — mostly absent or gated in the open tree). This analysis reads the **core open-source memory engine**, which is where the retrieval/write/consolidation logic lives.

The engine is unusually deep for a solo/community project: `src/omega/sqlite_store/` splits into `_base.py` (constants), `_store.py` (write path), `_search.py` (channels), `_query.py` (the ~1700-line multi-phase query orchestrator), `_maintenance.py` (consolidation/decay/feedback/graph). The whole thing is defensively engineered (locks around sqlite-vec C state, circuit breakers, FTS5 auto-repair, fail-open hooks).

### Storage & Schema

Single SQLite file. Tables: `memories`, `memories_vec` (sqlite-vec vec0, 384-dim bge-small-en-v1.5 ONNX cosine), `memories_fts` (FTS5 BM25), `edges` (typed graph), `forgetting_log` (deletion audit trail), `cloud_delete_queue`, `thompson_arms` (see dead-code note), entity index tables. The `memories` row is wider than the evidence file's "15 columns": beyond id/node_id/content/metadata/created_at/access_count/ttl_seconds/session_id/event_type/project/content_hash/priority/referenced_date/entity_id it also carries `canonical_hash`, `extracted_keywords`, `memory_type`, `valid_from`, `valid_until`, `agent_type`, `derived_from`, `source_uri`, `status`. Bi-temporal (`valid_from`/`valid_until`) is real and wired into supersession.

### Memory Types

Two orthogonal axes. (1) `event_type` — a ~35-value operational taxonomy (`decision`, `lesson_learned`, `error_pattern`, `user_preference`, `constraint`, `checkpoint`, `session_summary`, coordination types, `skill_template`, `public_statement`/`outcome_resolution` for say/do tracking…) that drives type-weighted scoring, default priority, and decay lambda (`_base.py:_TYPE_WEIGHTS`, `_DEFAULT_PRIORITY`, `_DECAY_LAMBDAS`). (2) `memory_type` — a cognitive category (`episodic`/`procedural`/`semantic`) mapped from event_type (`_MEMORY_TYPE_MAP`). This is close to Somnigraph's category axis but far more granular on the operational side and coupled tightly to a coding-agent workflow.

### Write Path (`_store.py:store`)

Synchronous and **rich at write time** — this is OMEGA's most notable divergence from Somnigraph:
- **Three-layer dedup**: exact SHA256 content hash; `canonical_hash` (whitespace/format-normalized hash to catch reformatted duplicates); embedding cosine ≥ 0.88 (`DEFAULT_EMBEDDING_DEDUP_THRESHOLD`). Dedup hits bump `access_count` and return the existing node.
- **Keyword extraction** (`_extract_keywords`) into `extracted_keywords` for BM25.
- **Write-time contradiction detection** (`_check_contradictions` → `contradictions.detect_contradictions`): pulls top-10 vec-similar neighbors and runs a stateless heuristic engine (negation asymmetry, ~20 antonym pairs, preference key-value change, temporal-override markers, gated by cross-encoder/Jaccard similarity). On a hit it annotates both memories' metadata (`contradicts` / `contradicted_by`) and inserts a `contradicts` edge weighted by confidence.
- **Write-time temporal supersession**: if a neighbor shares event_type, similarity ≥ 0.75, and is older (restricted to `decision`/`user_preference`/`lesson_learned`/`error_pattern`), it is marked superseded immediately (`valid_until` set, `status='superseded'`) — no signal words needed.
- Edges for explicit `dependencies` (causal) and `derived_from` (lineage) are written inline.

So OMEGA builds the contradiction/supersession/lineage graph **at ingest**, not offline.

### Retrieval (`_query.py:query`, ~1700 lines)

A genuine multi-phase pipeline, not a README fiction:
1. **Tiered caches / fast paths**: query-result cache (confidence-tiered TTL), trigram fingerprint fast-path, hot-tier (top by access_count), keyword-sufficiency check that can skip the vector channel.
2. **Query intent classification** (`_classify_query_intent`: navigational/factual/conceptual) → adaptive per-intent channel weights, plus **query decomposition** (compound queries split into sub-queries, re-merged with max-score dedup and a 1.15× multi-hit boost).
3. **Channels**: vector (sqlite-vec), FTS5 BM25 (internally blended 0.7·BM25norm + 0.3·word-ratio), and a **temporal proximity channel** (`_temporal_search`, date-distance-decayed).
4. **True RRF fusion** (`_rrf_fuse`, k=60, per-channel normalized before weighted accumulation) across vector+text+temporal. This is a real Cormack-style RRF, closer to Somnigraph's design than most systems surveyed.
5. **Metadata scoring** on the RRF base: `type_weight × feedback_factor × priority_factor × decay_factor × thompson_boost`, plus consolidation-quality boost and word/tag-overlap boost with negative-feedback damping.
6. **Per-event-type retrieval profiles** (`_RETRIEVAL_PROFILES`, "ALMA-inspired") reweight the five phases (vec, text, word, ctx, graph) by `query_hint`.
7. **Strong-signal short-circuit** (QMD-inspired): if FTS5 has a slam-dunk top result with a big gap, skip vector+rerank.
8. **LLM query expansion + HyDE** (`query_expansion.expand_query`): generates lexical variants, natural-language rephrasings, and a hypothetical answer passage (HyDE) for vague/conceptual queries, embedded and fed back as extra vec candidates.
9. **Graph multi-hop spreading activation** (`_query_phase_rerank`): 2 hops from top-5 seeds, 0.4 per-hop decay × edge weight.
10. **Cross-encoder reranking**: `bge-reranker-v2-m3` ONNX (ms-marco-MiniLM fallback), position-aware — top-3 get only 0.15 weight (preserve exact matches), rank 11+ get 0.50 (trust reranker more). Temporal metadata is prepended to passages.
11. **Adaptive retry**: on low confidence, re-query with relaxed abstention thresholds and dropped temporal/hint filters; keep only if confidence improves.
12. **Abstention**: min vec similarity 0.60, min text 0.35, min composite 0.10 — returns nothing rather than junk.

### Consolidation / Processing (`_maintenance.py:consolidate`, `apply_strength_decay`)

**Rule-based, not LLM-mediated.** `consolidate()` prunes zero-access stale memories (protected types exempt), fast-prunes low-priority decisions, caps session summaries, prunes orphaned edges/vec rows, and runs strength decay (`strength = type_weight × feedback_factor × decay_factor`; below floor → mark superseded). `omega_compact` does Jaccard clustering into summary nodes. Auto-compaction runs ~every 14 days at session start. There is no offline LLM pass doing relationship discovery / gap analysis / question generation — that work is done cheaply at write time (contradictions) or skipped.

### Lifecycle Management

Per-event-type exponential decay (`_DECAY_LAMBDAS`: permanent for constraint/preference/error/reminder; 50% at ~14–139 days for the rest), floors (0.35 accessed / 0.15 never-accessed to suppress zombie noise), bi-temporal validity, supersession, feedback-flag-driven review, and a forgetting audit log. Free-tier hard cap of 5000 nodes (grandfathered), Pro/env raises it. Very close in spirit to Somnigraph's decay module.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Hybrid vector + BM25 + temporal with RRF fusion | `_rrf_fuse` (k=60, per-channel normalized), 3 channels wired in `_query_phase_fusion` | **Validated** — real RRF, not a marketing "blend"; evidence file undersells it as "70% vector + 30% text" (that ratio is only the within-FTS blend). |
| Write-time contradiction detection + temporal supersession | `_check_contradictions`, `contradictions.detect_contradictions`, `mark_superseded` all in the store() path | **Validated** — genuinely synchronous; heuristic (not LLM), but real edges + metadata written. |
| Cross-encoder reranking improves conversational retrieval | `reranker.cross_encoder_score` (bge-reranker-v2-m3 ONNX), position-aware weighting | **Validated as present**; quality delta is asserted (P2 comments) not ablated in-repo. |
| HyDE + query decomposition for vague/multi-hop queries | `query_expansion.py`, `_decompose_query` | **Validated as present**; contribution not independently ablated. |
| 76.8% on LongMemEval (beats Zep 71.2) | `docs/benchmark-report.md`; harness `scripts/longmemeval_official.py` | **Plausible but category-tuned** — see cross-check. End-to-end QA (GPT-4.1 gen, GPT-4o/4.1 judge) but the harness feeds the **ground-truth question_type as `query_hint`**, selecting per-category retrieval profiles hand-tuned with the achieved accuracy baked into `_base.py` comments. Not a zero-knowledge generalization number, and it is LongMemEval not LoCoMo (not comparable to Somnigraph's 85.1 LoCoMo QA). |
| "Thompson sampling boost (outcome-correlated learning)" in scoring | `thompson_arms` table exists; `_get_thompson_boost`, `record_feedback` import `omega.thompson` | **Unrealized in OSS** — the `omega.thompson` module is **absent from the entire source tree**. Both call sites hit `ImportError` (caught, logged debug) → boost is always 1.0, `record_outcome` is a no-op. Dead reference or Pro-gated. |
| Implicit feedback loop (auto-"helpful" on surfaced memories) | `hooks/session_stop.py:_auto_feedback_on_surfaced` (caps 10, rating "helpful") | **Validated as present**, but self-reinforcing (see Not Useful). |

---

## Relevance to Somnigraph

### What omega-memory does that Somnigraph doesn't

- **Write-time graph construction.** Somnigraph builds contradiction/evolves/revision edges during **sleep** (`sleep_nrem.py`); OMEGA writes `contradicts`, `derived_from`, `causal`, and supersession edges **synchronously in `store()`**. This is exactly the "real-time graph construction" gap named in Somnigraph's context. OMEGA's version is a cheap heuristic (antonyms/negation/preference-change + a similarity gate), not an LLM classifier — lower precision, but it means stale facts get `valid_until` set the moment a newer one lands, before any offline pass runs.
- **HyDE + query decomposition** aimed squarely at the vocabulary gap. Somnigraph's multi-hop failure analysis pins an ~88% vocabulary-gap ceiling; OMEGA generates a hypothetical answer passage and embeds it (HyDE), and splits compound questions into independently-retrieved sub-queries. Somnigraph has no HyDE and no query decomposition in `tools.py`/`scoring.py`.
- **Intent-/category-conditioned channel reweighting** (`_RETRIEVAL_PROFILES`, `_INTENT_WEIGHTS`) before scoring — Somnigraph applies one learned reranker regardless of query shape.
- **Canonical-hash dedup layer** (format-normalized exact match) — cheaper than Somnigraph's embedding-similarity dedup for catching reformatted duplicates.
- **Adaptive retry with relaxed abstention** — a second pass when confidence is low; Somnigraph has no confidence-triggered re-query.

### What Somnigraph does better

- **Learned reranker.** Somnigraph's `reranker.py` is a 26-feature LightGBM trained on 1032 real queries (NDCG 0.7958, +6.17pp over formula) consuming feedback, graph (betweenness/PPR), and decay features. OMEGA's reranker is an **off-the-shelf cross-encoder** with zero training on its own data and no access to feedback/graph/decay signals — it only sees (query, passage) text. Different philosophy: OMEGA reranks by pretrained semantic relevance; Somnigraph learns-to-rank on its own distribution.
- **Feedback loop quality.** Somnigraph uses 0–1 float utility + durability, EWMA aggregation, UCB exploration, with a measured Spearman r=0.70 to GT. OMEGA uses an **integer counter** (+1/−1/−2, flag at ≤−3) with no measured GT correlation, and its advertised Thompson-bandit outcome learning is dead code in the OSS tree.
- **LLM-mediated consolidation.** Somnigraph's sleep does pairwise relationship classification, gap analysis, and question generation; OMEGA's `consolidate` is deterministic pruning + Jaccard clustering. Somnigraph discovers non-obvious edges OMEGA can't.
- **Graph retrieval depth.** Somnigraph has PPR expansion, novelty-scored adjacency, betweenness as a feature, and Hebbian co-retrieval PMI. OMEGA has fixed 2-hop spreading activation with static per-hop decay — no PPR, no centrality, no co-retrieval learning.

---

## Worth Stealing (ranked)

### 1. Cross-encoder as a feedback-free reranker fallback (Medium)
**What**: A pretrained cross-encoder (`bge-reranker-v2-m3` ONNX, CPU, int8 ~571MB) rescoring the top-10 RRF candidates, with position-aware weighting (trust it more on lower ranks, less on the top-3 exact matches).
**Why**: Somnigraph's seed already flags that the live LightGBM reranker artifact is missing and the system is running on the **RRF fallback**. A cross-encoder rerank is a strictly stronger fallback than RRF-only, and it needs no training data or feedback — the exact "reranker as a feedback-free complement" idea in the cognee thread. It also directly attacks the multi-hop vocabulary gap by scoring semantic (query, passage) relevance rather than lexical overlap.
**How**: New optional module mirroring OMEGA's `reranker.py` (ONNX Runtime + tokenizers, lazy load + circuit breaker). In `scoring.py`, when `reranker.py`'s LightGBM model is unavailable, apply cross-encoder scores as a multiplicative boost on the RRF-ranked top-k instead of returning raw RRF. Keep it behind an env flag; measure on the LoCoMo QA harness.

### 2. HyDE + query decomposition for the multi-hop vocabulary gap (Medium)
**What**: For vague/conceptual queries, generate a hypothetical answer passage (HyDE) and embed it as an extra vector candidate; for compound queries, split into sub-queries, retrieve each, merge with a multi-hit boost.
**Why**: Somnigraph's own multi-hop failure analysis identifies vocabulary mismatch as the ~88% ceiling, and L5b already leans on synthetic vocabulary bridges — HyDE is the query-side dual of that write-side idea, and decomposition is a clean attack on multi-hop questions the reranker can't fix.
**How**: A subagent-driven (not Python-SDK) LLM call producing `{lex, vec, hyde}` variants, cached; feed variants as additional FTS/vec candidates in `tools.py` recall before RRF. Gate on an intent/vagueness check so it only fires when cheap paths underperform. Ablate against L5b synthetics to see if they're redundant or additive.

### 3. Cheap write-time supersession heuristic as a pre-sleep filter (Low–Medium)
**What**: At `remember()` time, if a new memory shares category with a high-similarity older memory in the supersession-eligible set (preference/decision/lesson/error), set the older one's `valid_until` immediately rather than waiting for sleep.
**Why**: Corroborates the Phase 18 "write-path discipline beats retrieval tuning" finding with a concrete mechanism, and reduces stale/zombie retrieval in the window between write and the next sleep run. Somnigraph deliberately defers edges to sleep; this would be a narrow, reversible exception for the highest-churn categories only.
**How**: In `tools.py:impl_remember`, after embedding, run one vec-neighbor lookup; on category-match + similarity over a conservative threshold + older, set `valid_until` and let sleep confirm/upgrade to a typed `revision` edge. Keep the LLM classification in sleep as the source of truth; this is just an early floor.

### 4. Canonical-hash dedup layer (Low)
**What**: A whitespace/format-normalized SHA256 alongside the exact content hash, to catch reformatted re-captures of the same fact.
**Why**: Somnigraph dedups at 0.9 embedding similarity (an API call); a canonical exact-hash is free and catches the common "same text, different formatting" case before spending an embedding.
**How**: Add `canonical_hash` column + a `_canonicalize()` normalizer; check it in `impl_remember` before the similarity path.

---

## Not Useful For Us

### Auto-"helpful" feedback on surfaced memories
`session_stop.py` auto-rates every surfaced memory "helpful" — a self-reinforcing exposure bias that inflates whatever the ranker already favored. This is precisely the feedback-loop pathology Somnigraph's `docs/proactive-injection.md` was designed against (cooldown + Thompson gating to flatten exposure). Adopt the *warning*, not the mechanism.

### Multi-agent coordination, entity registry, LLM router, knowledge base
Pro-tier modules for team/corporate coding-agent fleets (file claims, branch guards, entity hierarchies, multi-provider routing). Out of scope for single-user Somnigraph.

### Coding-agent-specific event taxonomy
The ~35-value `event_type` set (file_claimed, branch_released, code_chunk, coordination_snapshot…) is tuned to an IDE-agent workflow, not Somnigraph's cognitive-category model.

---

## Connections

- **Write-path discipline corroboration**: OMEGA's synchronous dedup + contradiction + supersession is more evidence for the Phase 18 sweep conclusion (TrueMemory/ByteRover/agentmemory) that write-time quality, not retrieval cleverness, is what separates the leaders. OMEGA is unusual in doing this with cheap heuristics rather than an LLM.
- **Cross-encoder-as-complement** echoes the cognee-thread stance in the seed ("reranker as a feedback-free complement") and contrasts with Somnigraph's learned-reranker bet — OMEGA is the concrete off-the-shelf datapoint.
- **HyDE / synthetic vocabulary bridges**: convergent with Somnigraph L5b's synthetic-node bridges and with any source that closes the multi-hop vocabulary gap by generation rather than lexical expansion.
- **Category-tuned benchmark caution**: same genre of caveat as the carsteneu MIRIX/agentmemory corrections in the Phase 18 sweep — a headline number that needs a methodology asterisk (here: oracle question-type as a retrieval hint).
- **RRF fusion**: one of the few surveyed systems using true Cormack-style RRF like Somnigraph's `scoring.py`, rather than a fixed linear blend.

---

## Summary Assessment

OMEGA's core contribution is a **deep, defensively-engineered, fully-local retrieval pipeline** that does at write time what most systems (Somnigraph included) defer to offline processing: dedup across three layers, contradiction detection, and temporal supersession with bi-temporal validity. It stacks a lot of individually-sensible retrieval tricks — RRF over vec+BM25+temporal, intent-conditioned channel weights, HyDE, query decomposition, strong-signal short-circuit, position-aware cross-encoder rerank, adaptive retry, abstention floors — into one pipeline. As a catalog of "retrieval moves worth knowing," it's one of the richer repos in the corpus.

The single most valuable thing for Somnigraph is the **cross-encoder reranker as a feedback-free fallback**: it's a drop-in improvement over the current RRF-only fallback (the LightGBM artifact is missing per the seed), needs no training data, and attacks the known multi-hop vocabulary ceiling. HyDE and query decomposition are the next tier — genuinely aimed at the same ceiling Somnigraph has measured, worth an ablation against L5b synthetics.

What's overhyped or missing: the **76.8 LongMemEval is category-tuned** — the harness passes the ground-truth `question_type` as a retrieval hint that selects per-category profiles hand-tuned (with achieved accuracies literally in the code comments), so it's not a clean generalization number and it's a different benchmark than Somnigraph's 85.1 LoCoMo. The advertised **Thompson-sampling outcome learning is dead code** in the open-source tree (the `omega.thompson` module doesn't exist; the boost is always 1.0). And the reranker, despite the depth of everything around it, is a stock pretrained cross-encoder with no feedback loop — the opposite of Somnigraph's learned-to-rank-on-own-data bet. OMEGA is broad and pragmatic; Somnigraph is narrower and more principled about the parts that most affect quality (learned reranker, measured feedback, LLM consolidation).
