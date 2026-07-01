# mcp-memory-service — Broad-surface local MCP memory server with a reproducible LongMemEval retrieval harness and dream-inspired consolidation

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

mcp-memory-service (Heinrich Krupp / "doobidoo", Apache 2.0, ~1.9k stars, v10.68.0) is a self-hosted memory backend that leads with **integration surface**, not retrieval sophistication. It exposes MCP (stdio/SSE/HTTP), a 76-endpoint FastAPI REST API, OAuth 2.0 + DCR, a CLI, and an 8-tab web dashboard with a D3.js knowledge-graph view. Four storage backends: sqlite-vec (default), Cloudflare (D1+Vectorize+R2), a hybrid sync of the two, and Milvus. Codeberg mirror is canonical for cloning (GitHub URL 404s).

### Storage & Schema

- **sqlite-vec** default: `memories` table (`content_hash UNIQUE`, `content`, `tags` comma-sep, `memory_type`, `metadata` JSON, `created_at/updated_at` + iso, `deleted_at` soft-delete), a `vec0` virtual table `FLOAT[384] distance_metric=cosine`, and an FTS5 **trigram** table synced by triggers.
- **memory_graph** table (migration 008): `source_id, target_id, relationship_type, weight, valid_from, valid_until` — supports point-in-time ("time-travel") edge queries.
- Rich metadata lives in the `metadata` JSON blob under the "SHODH" interop spec: `quality_score`, `access_count`, `last_accessed_at`, `source_type`, `credibility`, `emotion`, `emotional_valence`, `emotional_arousal`, `episode_id`, `sequence_number`, `preceding_memory_id`, `conversation_id`. The "~28 schema fields" in the evidence file is 11 columns + 6 graph columns + these JSON sub-fields — an inflated count of what is really one text unit + a JSON bag.

### Memory Types

A 12-base-type ontology (observation, decision, learning, error, pattern, planning, ceremony, milestone, stakeholder, meeting, research, communication) with 63+ subtypes, extensible via `MCP_CUSTOM_MEMORY_TYPES`. This is a flat taxonomy on a free-text field, not Somnigraph's structured category + priority + themes + valid_from/valid_until + decay_rate schema. Evidence correctly marks `layeredMemory: false` — no detail/summary/gestalt equivalent.

### Write Path

- **Extraction/enrichment**: `reasoning/entities.py` (`EntityExtractor`) auto-links `@mentions`, `#tags`, URLs, file paths; `entity_linker.py` creates `shares_entity` edges. Content auto-splits when it exceeds max length (boundary-preserving, with overlap).
- **Dedup**: exact `content_hash` always; semantic dedup at `semantic_dedup_threshold` default **0.85**; `conversation_id` bypasses dedup for incremental capture.
- **Quality gating** (`quality/`): a composite `QualityScorer` combines an AI evaluator (`ai_evaluator.py`, ONNX ms-marco / nvidia-quality-classifier-deberta, or any OpenAI-compatible endpoint) with `implicit_signals.py` (length, structure, metadata heuristics) into a 0–1 `quality_score` — `ai_weight*(1-boost_weight) + implicit*boost_weight`. This is a real **write-path quality signal**, which Somnigraph has no equivalent of.

### Retrieval

- Default path is **pure semantic vector** (`storage.retrieve()`), optionally boosted by `retrieve_with_quality_boost(quality_weight=…)`.
- `reasoning/multi_strategy.py` implements textbook **RRF fusion** (`rrf_fuse`, k=60) but only wires **two strategies: `semantic` + `tag`** — despite the evidence file's "BM25 + vector combined; RRF fusion" claim, the fusion in code does not fuse the FTS5/BM25 channel with the vector channel. BM25 exists (FTS5 trigram) but the RRF fuser fuses semantic-vec with tag-chronological.
- Graph reasoning (`reasoning/inference.py`, `nli.py`, `mutability.py`): transitive closure + abductive inference (v10.66), NLI contradiction detection (v10.67), fact-mutability classification stable/volatile/ephemeral with `contradiction_action()` (v10.68).
- **No learned reranker, no retrieval feedback loop.** Ranking is cosine + optional quality boost + RRF. This is the sharpest architectural gap vs Somnigraph.

### Consolidation / Processing

A genuinely interesting **"Dream-Inspired" 6-phase consolidator** (`consolidation/consolidator.py`), scheduled by time horizon (daily/weekly/monthly/quarterly/yearly via `scheduler.py`):

1. Exponential **decay** scoring (`decay.py`) — base importance × time-decay, boosted by connection count and recent access, per-type retention periods.
2. Semantic **clustering** (`clustering.py`, min_cluster_size 5).
3. **Creative associations** (`associations.py`) — the standout: randomly samples memory pairs and keeps only those whose cosine similarity falls in a **mid-similarity "sweet spot" (0.45–0.7)**, deliberately skipping near-duplicates and unrelated pairs to surface non-obvious links. Regex concept extraction (camelCase, acronyms, URLs, dates) supplies connection reasons.
4. Semantic **compression** (`compression.py`) — cluster → summary.
5. Controlled **forgetting** (`forgetting.py`) — archival of low-relevance memories; mistake notes with `failure_count >= 3` are protected.
6. **Insights** (`insights.py`) — pattern/trend/gap "insight cards".

Crucially this is **embedding- and rule-based, not LLM-mediated** (regex + cosine + exponential math). Somnigraph's NREM/REM sleep is LLM-classified. Same biological metaphor, very different engine.

### Lifecycle Management

Decay + soft-delete (`deleted_at`) + `evolve_memory` lineage (`superseded_by`) + temporal edge validity windows + controlled forgetting/archival. Evidence marks `quarantine: false` (only soft-delete) — correct.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| LongMemEval-S 86.0% R@5 (session), 80.4% (turn), 0 LLM calls | `docs/BENCHMARKS.md`; `benchmark_longmemeval.py` calls `storage.retrieve()` and computes recall/ndcg/mrr | **Valid but retrieval-recall only** — NOT end-to-end QA, not comparable to Somnigraph's 85.1 LoCoMo QA. Repo is admirably honest: it prints "0 LLM calls" and shows mempalace beating it (96.6% R@5) |
| Reproducible open harness on a standard dataset | `benchmark_longmemeval.py` + `longmemeval_dataset.py` load LongMemEval-S from HuggingFace, isolated temp DB, session/turn/hybrid ingestion modes, quality-boost ablation | **Validated** — clean, portable, zero-LLM; genuinely useful |
| Hybrid BM25+vector RRF fusion | `multi_strategy.py` `rrf_fuse(k=60)` fuses `semantic` + `tag` only | **Overstated** — BM25/FTS5 exists but is not the fused channel; default retrieval is pure vector |
| Knowledge graph with 7 typed edges + temporal validity | `memory_graph` table, `entity_linker.py`, `mutability.py` | **Validated**; write-time entity linking is real (Somnigraph builds edges only during sleep) |
| Dream-inspired auto-consolidation w/ decay & forgetting | `consolidation/` full 6-phase pipeline | **Validated** but **rule/embedding-based, not LLM-mediated** |
| Write-path quality scoring | `quality/scorer.py` composite AI + implicit | **Validated** as a signal; no evidence it improves retrieval (ablation config exists, results not published) |
| "7 search modes / 6 data sources / ~28 schema fields" | evidence file counts | **Inflated taxonomy** — tag search and temporal filter counted as distinct "modes"; one text unit + JSON bag counted as 28 fields |

---

## Relevance to Somnigraph

### What mcp-memory-service does that Somnigraph doesn't

- **A reproducible, zero-LLM retrieval harness on a standard public dataset (LongMemEval-S, 500 Q).** Somnigraph has only LoCoMo QA (85.1) and internal GT — **zero external LME numbers**. This is a real evidence gap: our `scripts/` has no LongMemEval adapter.
- **Write-time entity linking + graph construction** (`entity_linker.py`) — Somnigraph builds edges only during NREM sleep (`sleep_nrem.py`), never at write time.
- **Write-path quality gating** (`quality/scorer.py`) — Somnigraph's `tools.py` remember path has no salience/quality score; every memory enters equal.
- **Fact-mutability classification** (stable/volatile/ephemeral) driving contradiction handling — Somnigraph has typed contradicts edges but no per-fact volatility class.
- **Integration breadth** (REST, OAuth, Cloudflare/Milvus backends, multi-agent tag namespaces, 27 platforms) — irrelevant to Somnigraph's single-user research mandate, but real engineering.

### What Somnigraph does better

- **Learned ranking.** Somnigraph's `reranker.py` (26-feature LightGBM, NDCG 0.7958, +6.17pp over formula) vs. this system's cosine + RRF + optional quality boost. No learned reranker here at all.
- **Explicit feedback loop** with measured Spearman r=0.70 to GT, EWMA/UCB — absent here (only passive `access_count`/`last_accessed`).
- **LLM-mediated consolidation.** Somnigraph's sleep classifies pairs and creates typed edges with an LLM; this system's consolidation is regex + cosine + exponential math.
- **Graph-conditioned retrieval via PPR** (`scoring.py`) — here the graph is visualized and reasoned over but does not feed a PPR-style retrieval expansion into ranking.
- **Honest-accounting depth** — Somnigraph documents failure modes and vocabulary-gap ceilings; this repo publishes recall numbers cleanly but with less negative-result narrative.

---

## Worth Stealing (ranked)

### 1. LongMemEval-S retrieval harness adapter (Medium)
**What**: A reproducible, 0-LLM benchmark that loads LongMemEval-S (500 Q, 45–62 distractor sessions each) from HuggingFace into an isolated temp DB, ingests in session/turn/hybrid modes, and reports R@5/R@10/NDCG@10/MRR by question type. See `scripts/benchmarks/benchmark_longmemeval.py`, `longmemeval_dataset.py`, and the reusable `recall_at_k`/`ndcg_at_k`/`mrr` in `locomo_evaluator.py`.
**Why**: Somnigraph has zero external LongMemEval numbers. A retrieval-recall benchmark on a standard dataset would let us report a comparable metric and stress-test the reranker/RRF stack against a distractor-heavy haystack, complementing (not replacing) LoCoMo QA. The `_match_evidence` trick — crediting each evidence session only once so recall can't exceed 1.0 when multiple turns from one session are retrieved — is a subtle correctness detail worth copying.
**How**: Add `scripts/benchmarks/longmemeval.py` that ingests LongMemEval-S turns as memories (tagged with session_id for evidence matching), runs our real recall path (RRF + reranker), and scores R@k/NDCG/MRR. Reuse our existing GT infra; label by `answer_session_ids`. **Caveat to document**: this is retrieval recall, not end-to-end QA — the 86.0 cell is R@5, not comparable to our 85.1 LoCoMo QA.

### 2. Mid-similarity "sweet spot" (0.45–0.7) neighbor sampling for consolidation (Low)
**What**: `associations.py` deliberately keeps only memory pairs whose cosine similarity is **moderate** (config `min_similarity=0.45`, `max_similarity=0.7`), skipping both near-duplicates (>0.7, no new information) and unrelated pairs (<0.45, spurious). Random pair sampling with a coupon-collector-safe sampler caps work per run.
**Why**: Somnigraph's NREM pair selection (`sleep_nrem.py`) tends to feed top-K nearest neighbors to the LLM classifier, which biases toward near-duplicate pairs the classifier will merge — the interesting `supports`/`evolves`/`contradicts` edges live at *moderate* similarity. A band filter could raise the yield of non-trivial edges per LLM call.
**How**: In `sleep_nrem.py` candidate generation, replace/augment top-K with a similarity **band** filter (tune the 0.45–0.7 window against our embedding distribution) before sending pairs to the LLM. Cheap, offline-testable against existing sleep logs.

### 3. Write-path quality signal as a reranker feature (Medium)
**What**: A composite quality score (AI evaluator + implicit structural signals) computed at write time, stored on the memory (`quality/scorer.py`).
**Why**: Somnigraph's reranker has no write-time salience feature; a per-memory quality prior could help demote low-value memories that currently rank on embedding proximity alone. This aligns with the Phase-18 "write-path quality is what LME leaders win on" finding already in our corpus.
**How**: Compute a cheap implicit-signals score at remember time (length, structure, has-code, theme count), store as a column, add as a `reranker.py` feature. Keep the LLM/ONNX AI-scorer optional — start with the free implicit signals only.

---

## Not Useful For Us

### Multi-transport / OAuth / Cloudflare / Milvus / 76 REST endpoints
Single-user local research artifact; enterprise deployment surface is out of scope.

### Multi-agent tag namespaces (agent:/crew:/proj:) as a messaging bus
Solves inter-agent coordination Somnigraph doesn't have.

### D3.js 3D knowledge-graph visualization
Demo-facing; no retrieval-quality contribution.

### 12-type × 63-subtype ontology + SHODH emotional metadata
A larger flat taxonomy than our category set with no evidence it improves retrieval; Somnigraph's structured schema + learned reranker subsumes the utility.

---

## Connections

- **Convergent consolidation metaphor** with Somnigraph's sleep and with other dream/consolidation systems in the corpus — but this is the rule-based branch (cosine + decay math), where Somnigraph and the LLM-mediated systems are the learned branch. Good contrast datapoint for `docs/similar-systems.md`.
- **Corroborates the Phase-18 source-sweep thesis** (`docs/sessions/2026-06-28-phase18-source-sweep.md`): the LME/LoCoMo leaders win on *write-path* (mempalace verbatim beats this system 96.6 vs 86.0 R@5 with the *same* zero-LLM retrieval). This system's own benchmark table concedes the point.
- **RRF fusion** overlaps our `scoring.py`/RRF (k=14) but here k=60 textbook and only 2 channels — a weaker instance of a mechanism we've already Bayesian-tuned.
- **Fact mutability (stable/volatile/ephemeral)** rhymes with typed-edge and supersession patterns seen in memv/memos-style analyses; here it's a per-fact class rather than an edge type.

---

## Summary Assessment

mcp-memory-service's core contribution is **breadth and packaging**: it is the most integration-complete open MCP memory server in this survey — four storage backends, REST+MCP+OAuth+CLI+dashboard, 27 platform integrations, and a polished knowledge-graph UI. On the axes Somnigraph optimizes (learned ranking, feedback loop, LLM-mediated consolidation, graph-conditioned retrieval), it is **behind**: retrieval is cosine + textbook RRF over two channels with no learned reranker and no feedback loop, and its "dream-inspired" consolidation is regex + exponential-decay math rather than LLM classification.

The single most valuable thing to take is the **reproducible LongMemEval-S retrieval harness** — it fills a concrete evidence gap (we have no external LME numbers) and is honestly built (0 LLM calls, per-question-type breakdown, evidence-session dedup). The second is the **mid-similarity 0.45–0.7 sweet-spot** heuristic for consolidation pair selection, a cheap tweak to our NREM candidate generation. A distant third is treating a **write-time quality signal** as a reranker feature.

The one number to not misread: **86.0% is LongMemEval-S R@5 retrieval recall with zero LLM calls — it is not end-to-end QA and is not comparable to Somnigraph's 85.1 LoCoMo QA.** The repo itself states this plainly; the risk is a survey table flattening "86.0" next to "85.1". The evidence file's "hybrid BM25+vector RRF" checkmark is also an overstatement — the code's RRF fuser combines the semantic-vector and tag channels, not BM25 and vector.
