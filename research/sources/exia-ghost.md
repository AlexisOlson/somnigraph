# EXIA GHOST (NexaSeed AI) — Analysis

*Generated 2026-03-23 by Opus agent reading GitHub repo + published benchmark results*

---

## Repo Overview

**Repo**: https://github.com/francisdu53/exia-ghost-benchmarks
**License**: Proprietary (benchmark adapters only are public)
**Language**: Python
**Author**: Francis Babin / NexaSeed AI ($0 funding)
**Description**: "Bio-inspired cognitive memory architecture" with five memory stores and biological forgetting

**Problem addressed**: Long-term conversational memory for AI agents with typed memory stores and biological forgetting — competing on LoCoMo and HaluMem against funded systems.

**Core approach**: Five specialized memory stores (contextual/short-term, semantic/facts, episodic/events, procedural/skills, prospective/plans), ChromaDB with MiniLM 384d embeddings for retrieval, LLM extraction of atomic facts in third person, NLI cross-encoder ensemble for contradiction detection, and a "Cognitive Contract" prompt engineering framework for controlling LLM output.

**Maturity**: Closed-source system with public benchmark adapters and results. The repo contains only benchmark harness code, results JSON, and documentation — not the core system. Claims are unverifiable beyond the published result files. Imports reference `exia.core.types`, `exia.memories.episodic`, `exia.memories.semantic`, `exia.providers.embedder` — none of which are in the public repo.

---

## Architecture (from documentation)

### Technical Stack

| Component | Technology |
|-----------|-----------|
| Vector DB | ChromaDB (temporary per-conversation instances for benchmarks) |
| Embeddings | all-MiniLM-L6-v2 (384d, local) |
| LLM (extraction) | Claude Haiku 4.5 |
| LLM (QA) | Claude Haiku 4.5 |
| Contradiction detection | DeBERTa-v3 NLI cross-encoder ensemble (base + small, OR logic) |
| Judge | GPT-4o-mini (LoCoMo), GPT-4o (HaluMem) |

### Memory Stores

Five typed stores, each with distinct decay and consolidation behavior:

| Store | Purpose | Notes |
|-------|---------|-------|
| Contextual | Short-term conversation buffer | Promotes to other stores |
| Semantic | Facts extracted as third-person atomic statements | Primary retrieval target |
| Episodic | Events with temporal context | |
| Procedural | Skills and patterns | |
| Prospective | Plans and intentions | |

### Retrieval

Pure vector similarity search via ChromaDB, top-k=15 per speaker. No BM25, no FTS, no RRF fusion, no learned reranker. The simplicity is striking given the competitive results — retrieval sophistication is not what drives this system's performance.

### Extraction Pipeline

LLM-based extraction of atomic facts in third person via Claude Haiku 4.5. Sentence splitting + inline dedup before LLM extraction. This is the likely source of the system's strength — what gets stored matters more than how it's retrieved when the retrieval mechanism is pure vector search.

### Contradiction Detection

NLI cross-encoder ensemble using two DeBERTa-v3 models (base + small) with OR logic — if either model flags a contradiction, it counts. Claimed 92.31% F1, 96% precision on contradiction detection. Three documented failure cases are world-knowledge inference (e.g., "no children" vs. "pregnant").

### Cognitive Contract

Prompt engineering framework with explicit constraints on LLM output. Key technique: when no memories match a query, the prompt includes "NO relevant memories found — NEVER fabricate" rather than softer phrasing. Claimed to eliminate hallucinations across all 6 LLMs tested. The finding: "A weak model with a strong contract outperforms a strong model with a weak prompt."

### Forgetting Formula

`(1 - usage) * time * (1 - emotion)` — conceptually similar to biological decay but cruder than Ebbinghaus-inspired models. No floor, no reheat mechanism, no per-category rates.

---

## Benchmark Results

### LoCoMo

**89.94% accuracy (cats 1-4), 85.80% including adversarial (cat 5).** GPT-4o-mini judge.

10 conversations, 1,986 QA pairs. Each conversation gets a fresh ChromaDB instance. Memory extraction via Claude Haiku 4.5 with QA generation also via Haiku.

Per-category:
```
Category          Accuracy
─────────────────────────
Multi-hop (1)      86.17%
Temporal (2)       85.36%
World knowledge(3) 93.75%
Single-hop (4)     92.51%
Adversarial (5)    71.52%
```

**Category 5 is notable** — they're the first system to publish adversarial abstention results. 71.52% means the system correctly abstains ~72% of the time when asked about information not in the conversation.

### HaluMem

**F1 71.99%** on 1 of 20 users (Martin Mark, 65 sessions). GPT-4o judge (official standard).

- Extraction: precision 92.90%, recall 58.76%
- Memory update: correct 77.46%, hallucination 1.41%, omission 21.13%
- QA: correct 58.54%, hallucination 18.90%, omission 22.56%
- False memory resistance: 54.40%

High precision, low recall — when it extracts something, it's correct 93% of the time. The 1.41% hallucination rate on updates is very low.

### Internal Benchmarks

- 435 automated pipeline tests
- 92.9% long-term recall accuracy over 96 exchanges
- 0% hallucination rate (Cognitive Contract)
- p99 latency < 3ms
- Targeted recall 93% vs. free recall 40-60% (significant gap)

### Judge Robustness

Re-ran LoCoMo with GPT-4o-mini as both QA generator and judge. Max delta 2.80%, confirming the system is relatively LLM-agnostic. Somnigraph already uses dual-judging (Opus vs GPT-4.1-mini), so this validation approach aligns.

---

## Comparison to Somnigraph

### What EXIA GHOST has that we don't

**Explicit absence declaration.** The Cognitive Contract's "NO relevant memories found — NEVER fabricate" prompt technique is a simple, high-value intervention. Somnigraph's reader prompt could adopt this directly.

**NLI cross-encoder contradiction detection.** Two DeBERTa models with OR logic (92.31% F1) is more principled than heuristic-based detection. This could strengthen our sleep consolidation pipeline, particularly NREM classification which currently relies on LLM judgment for contradiction detection.

**Category 5 evaluation.** EXIA GHOST is the only system publishing adversarial/abstention results. Treating "Not mentioned" as the gold answer is a simple approach that makes cat 5 evaluable.

**Third-person atomic fact extraction.** Extracting memories as third-person atomic facts ("Caroline mentioned she enjoys hiking") rather than raw dialog turns creates cleaner retrieval targets. The extraction quality is likely the primary driver of their results — pure vector search on well-extracted facts outperforms sophisticated retrieval on raw dialog.

### What we have that they don't

**Learned reranker.** LightGBM model trained on GT relevance judgments. EXIA GHOST uses pure vector similarity with no learned scoring.

**Hybrid retrieval with RRF fusion.** FTS5/BM25 + vector similarity through reciprocal rank fusion. EXIA GHOST is vector-only (ChromaDB).

**Feedback loop.** `recall_feedback()` reshapes scoring, adjusts decay, strengthens edges. EXIA GHOST has no retrieval feedback mechanism.

**Sleep pipeline with LLM consolidation.** Three-phase offline processing with per-memory LLM decisions. EXIA GHOST has maintenance but documentation suggests no LLM-mediated consolidation decisions.

**Hebbian co-retrieval edges.** PMI-weighted graph structure from co-retrieval patterns. No equivalent.

**Graph traversal.** PPR expansion for multi-hop reasoning. EXIA GHOST has no graph structure.

**Enriched embeddings.** Content + category + themes + summary concatenated before embedding. EXIA GHOST embeds extracted facts directly (though the extraction itself is a form of enrichment).

**Open and reproducible.** Full source code, documented methodology, reproducible benchmarks. EXIA GHOST is proprietary with only benchmark adapters public.

### Architectural trade-offs

| Dimension | EXIA GHOST | Somnigraph | Trade-off |
|-----------|-----------|-----------|-----------|
| Retrieval | Pure vector (ChromaDB, 384d) | Hybrid FTS + vector + graph + learned reranker | Simplicity vs. precision |
| Extraction | LLM extraction of atomic facts (write-time) | Raw dialog turns + sleep enrichment (offline) | Write-time quality vs. write-time cost |
| Contradiction | NLI cross-encoder ensemble (inline) | LLM classification (offline, during sleep) | Real-time vs. batch |
| Embeddings | MiniLM 384d (local, free) | text-embedding-3-small 1536d (cloud, cost) | Privacy/cost vs. quality |
| Forgetting | `(1-usage)*time*(1-emotion)` | Ebbinghaus decay with floor + reheat + per-category rates | Simple vs. principled |
| Evaluation | LoCoMo + HaluMem + internal | LoCoMo + GT-based tuning + utility calibration | Broader benchmarks vs. deeper methodology |
| Reproducibility | Proprietary (unverifiable) | Open source | Trust vs. verification |

---

## Comparative Numbers

Direct comparison is complicated by judge differences:

| System | LoCoMo Overall | Judge | Notes |
|--------|---------------|-------|-------|
| EXIA GHOST | 89.94% (cats 1-4) | GPT-4o-mini | Proprietary, unverifiable |
| Somnigraph | 85.1% (cats 1-4) | Opus 4.6 | Open source, reproducible |
| Somnigraph | 88.3% (cats 1-4) | GPT-4.1-mini | Same run, more lenient judge |

Opus is 3.2pp stricter than GPT-4.1-mini (measured). GPT-4o-mini leniency relative to GPT-4.1-mini is unknown but likely comparable. The true gap is probably 1-2pp, not 4.8pp. And Somnigraph hasn't yet run sleep enhancements or expansion on the full QA pipeline — the current numbers are baseline retrieval + reader.

With cat 5 included, EXIA GHOST reports 85.80%. Somnigraph doesn't currently evaluate cat 5.

---

## Worth Adopting?

**Explicit absence declaration in reader prompt**: Yes, immediately. A one-line prompt change with potentially significant impact on hallucination in the QA pipeline. Zero cost.

**NLI cross-encoder for contradiction detection**: Worth investigating as a complement to LLM-based NREM classification. Two DeBERTa models are cheap to run locally. Could serve as a fast pre-filter before the LLM judges contradiction severity. The 92.31% F1 claim needs verification, but the approach is principled.

**Category 5 evaluation**: Yes, low effort. Treating "Not mentioned" as gold answer for adversarial questions makes them evaluable. The current exclusion loses signal about abstention quality.

**Third-person atomic fact extraction at write time**: This is the big question. EXIA GHOST's results suggest that extraction quality may have more retrieval headroom than scoring sophistication. However, Somnigraph's architecture defers LLM work to sleep time — moving extraction to write time would fundamentally change the cost model. More relevant: could the sleep pipeline's enrichment step be improved to produce cleaner atomic facts? This is a sleep enhancement, not a write-path change.

**Targeted vs. free recall measurement**: Worth adding to evaluation. Their 93% targeted / 40-60% free recall gap suggests vector search has a structural limitation on open-ended queries that graph traversal could address — measuring this on Somnigraph would quantify the value of PPR expansion for broad queries.

---

## Worth Watching

**Extraction quality as the binding constraint.** If EXIA GHOST's results hold (and the proprietary core makes this a big "if"), the implication is that what you store matters more than how you retrieve it — at least for LoCoMo-scale evaluation. This aligns with HippoRAG's finding that 26% of failures trace to extraction quality.

**Cognitive Contract evolution.** The prompt engineering framework is the most practically transferable contribution. If the community adopts and iterates on explicit absence declaration, the resulting prompt patterns would be useful for any memory-augmented system.

**HaluMem as second benchmark.** The 1.41% hallucination rate on memory updates and 92.90% extraction precision are strong numbers. If HaluMem gains adoption as a standard benchmark (alongside LoCoMo), having a comparison point is valuable.

---

## Key Claims and Evidence Assessment

1. **89.94% LoCoMo accuracy (cats 1-4).** *Evidence*: Result JSON files in repo. Unverifiable — proprietary core, no reproducible pipeline. The benchmark adapters show clean methodology (fresh ChromaDB per conversation, standard judge). Plausible but not confirmed.

2. **Cognitive Contract eliminates hallucination across 6 LLMs.** *Evidence*: Internal testing only. The specific technique (explicit absence declaration) is well-motivated and aligns with known prompt engineering best practices. Plausible and testable.

3. **92.31% F1 on contradiction detection via NLI ensemble.** *Evidence*: Internal benchmark. The OR-logic ensemble approach is principled. The three documented failure cases (world-knowledge inference) are reasonable edge cases. Plausible.

4. **"A weak model with a strong contract outperforms a strong model with a weak prompt."** *Evidence*: Anecdotal from internal testing. The general principle is well-established in prompt engineering literature. The specific claim about their Cognitive Contract is untestable without the proprietary system.

5. **$0 funding achieving competitive results.** *Evidence*: Self-reported funding status. The competitive results (if real) suggest that extraction quality + prompt engineering can compensate for infrastructure simplicity — MiniLM + ChromaDB + good prompts vs. sophisticated retrieval pipelines.

---

## Relevance to Somnigraph

**Medium-high** for actionable techniques (absence declaration, NLI contradiction detection, cat 5 evaluation), **medium** for strategic insight (extraction quality as headroom), **low** for architectural borrowing (simpler stack, proprietary core, no learned scoring).

The most valuable insight is directional: EXIA GHOST's competitive results with a dramatically simpler retrieval stack (pure vector search, no BM25, no reranker, no graph) point toward extraction quality and prompt engineering as underexplored levers. Somnigraph's sleep pipeline — which already does offline enrichment — is well-positioned to improve extraction quality without changing the write path. The sleep enhancements not yet applied to LoCoMo (consolidation, edge building, summary generation) may close more of the gap than further reranker iteration.
