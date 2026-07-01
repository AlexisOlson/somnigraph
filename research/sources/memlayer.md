# MemLayer — Salience-gated LLM-memory library (ChromaDB vectors + NetworkX entity graph), write-time extraction, no reranker/fusion

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

MemLayer (`github.com/divagr18/memlayer`, MIT, ~275 stars, created 2025-11-16) is a Python library that wraps five LLM providers (OpenAI/Claude/Gemini/Ollama/LMStudio) behind a common `chat()` API and transparently adds memory. It is a library/SDK, not an MCP server or agent-platform integration. No benchmarks published.

### Storage & Schema
Two independent stores, no shared ID space or fusion layer:
- **Vector**: ChromaDB embedded, cosine space, one collection per embedding dimension (`memlayer/storage/chroma.py`). A memory ("fact") carries `content`, `importance_score` (float, from the extractor LLM), `expiration_timestamp` (parsed from text), `status` (active/archived), `access_count`, `last_accessed_timestamp`, `user_id`. Note ChromaDB stores only `content[:100]` as the document; full text lives in metadata.
- **Graph**: NetworkX `DiGraph` persisted as a `.pkl` (`memlayer/storage/networkx.py`); an optional Memgraph backend exists (`storage/memgraph.py`). Nodes are entities (name + `type` such as Person/Organization/Concept) with the same lifecycle attributes; edges are `(subject, predicate, object)` relationships. Tasks are also graph nodes with `due_timestamp`.

Effective schema is ~3 real fields per fact — flat, no themes, no category taxonomy, no source/context/trust fields, no layering.

### Memory Types
Three storage forms, not a semantic taxonomy: facts (vector), entity+relationship triples (graph), and time-triggered task reminders (graph). No episodic/semantic/procedural distinction, no priority, no valid_from/valid_until versioning.

### Write Path
The interesting part. On each chat turn (`wrappers/openai.py` ~L278) the user text is lightly pronoun-normalized ("My" → "The user's", "I'm" → "The user is") then handed to `ConsolidationService.consolidate()` (`services.py:262`), which runs entirely in a **daemon background thread**:
1. **Salience gate** (`ml_gate.py`, `SalienceGate.is_worth_saving`) decides whether to save at all. Two stages: (a) fast regex heuristics — non-salient patterns (greetings/thanks/ack), then salient patterns (names/dates/prefs/IDs), plus "<10 chars → skip", "has ProperName / number → save"; (b) for the uncertain middle, an **embedding-prototype margin classifier**: max cosine sim of the text against ~32 hardcoded *salient* prototype sentences vs ~40 *non-salient* ones; save iff `max_salient > max_non_salient + threshold` (threshold 0.0 default, -0.1..0.2 range). Three modes: LOCAL (sentence-transformers), ONLINE (OpenAI embeddings, cached to disk), LIGHTWEIGHT (keyword counts, graph-only).
2. If salient, one LLM call (`analyze_and_extract_knowledge`) returns `{facts[], entities[], relationships[]}`. Facts embedded + written to Chroma; entities/relationships written to the graph **at write time** (not offline).
3. **Entity dedup** at insert (`networkx._find_canonical_entity`, threshold 0.85): case-insensitive exact match → word-subset containment (prefer longer name, "Dr. Watson" ⊂ "Dr. Emma Watson") → Jaccard word-overlap. Purely lexical; no embedding-based coreference.

### Retrieval
`SearchService.search()` (`services.py:53`) with three tiers set only by vector `top_k`: fast=2, balanced=5 (default), deep=10.
- **Vector**: ChromaDB cosine query filtered to `status=active` and `user_id`; score = `1 - distance`. LRU-cached query embeddings. That is the entire ranking — no BM25, no RRF, no learned reranker, no scoring formula beyond raw cosine.
- **Graph** (deep tier only, needs an LLM client): extract query entities → fuzzy `find_matching_nodes` → 2-hop `get_subgraph_context`; plus a heuristic pass that regex-scrapes capitalized words from the top-3 vector results and traverses from those; plus a hardcoded "User" node workaround.
- **"Hybrid" is string concatenation, not fusion**: `final_result = vector_context + graph_context`. The two channels are formatted into separate text blocks and glued together; there is no rank merge, no score combination, no dedup across channels. `synthesize_answer()` then optionally feeds this blob to an LLM for a grounded answer with a confidence field.

### Consolidation / Processing
None in the offline-reprocessing sense. `ConsolidationService` is misleadingly named — it is the synchronous-per-turn extraction/write path, not a sleep/merge/summarize cycle. There is no pairwise re-classification, no edge inference over existing memories, no summarization, no gap analysis, no question generation.

### Lifecycle Management
`CurationService` (`services.py:470`), a background thread every `interval_seconds` (~3600s): hard-deletes facts past `expiration_timestamp`; for active memories computes `relevance = (importance_score + log1p(access_count) + recency_boost) / age_factor`, where `recency_boost` fades over 7 days and only applies after first access, `age_factor = max(1, age_days)`; archives when `relevance < 0.3`. Access tracking bumps `access_count`/`last_accessed` on retrieval. No supersession chain, no contradiction handling, no versioning, no explicit user-facing forget.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| "Hybrid Search: vector similarity + knowledge graph traversal" | `services.py` concatenates two text blocks | **Misleading** — no fusion/ranking; string concat of two channels |
| Salience-based write gating | `ml_gate.py` two-stage regex + prototype-margin classifier | **Real** and the system's genuine differentiator |
| Auto entity extraction + dedup ("John = John Smith") | `_find_canonical_entity` word-subset + Jaccard | **Real but lexical-only**; no embedding coreference; brittle on aliases/nicknames |
| Automatic decay/curation | `CurationService._calculate_relevance`, archive<0.3 | **Real**, simple interpretable formula |
| 100% offline LOCAL mode | sentence-transformers + Ollama path | **Real** |
| Benchmarks (LoCoMo/LongMemEval/etc.) | none | **Absent** — no published scores at all |

---

## Relevance to Somnigraph

### What MemLayer does that Somnigraph doesn't
- **Write-path salience gating** (`ml_gate.py`). Somnigraph has no equivalent — `tools.py::impl_remember` writes whatever the agent decides to store; there is no automated "is this turn worth saving" pre-filter. This is the one place MemLayer touches a named Somnigraph gap (write-path quality gating).
- **Write-time graph construction.** MemLayer builds the entity graph synchronously on each turn; Somnigraph builds typed edges only during NREM sleep (`scripts/sleep_nrem.py`). Different design choice, not strictly better — MemLayer pays LLM latency per turn (mitigated by threading) and gets a cruder, untyped graph.
- **Entity resolution at all.** Somnigraph has none; MemLayer has a (lexical) canonicalizer.

### What Somnigraph does better
Essentially everything on the retrieval and research axes. Somnigraph has hybrid BM25+vector with RRF fusion and a 26-feature LightGBM reranker (`reranker.py`) — MemLayer has raw cosine top-k and calls text concatenation "hybrid." Somnigraph has a measured feedback loop (Spearman 0.70), PPR graph-conditioned retrieval, typed edges (supports/contradicts/evolves), LLM-mediated sleep consolidation, per-category exponential decay, and an 85.1% LoCoMo QA result. MemLayer has no reranker, no fusion, no feedback, no contradiction handling, no consolidation, and no benchmarks.

---

## Worth Stealing (ranked)

### 1. Prototype-margin salience classifier as a cheap write-gate (effort: Low) — *consider, likely note-only*
**What**: Decide "worth persisting?" by max-cosine of the candidate text against two small hand-curated prototype sets (salient vs non-salient) and requiring `salient - nonsalient > threshold`, with a regex fast-path for the obvious cases. No training, no labels beyond the prototype lists.
**Why**: Somnigraph's write path is entirely agent-judgment-driven; the Phase 18 sweep concluded write-path quality is what the LoCoMo/LME leaders actually win on. A near-zero-cost margin gate is a candidate pre-filter for the *auto-capture pending* path (`status="pending"` memories) or as a floor in the proactive-injection design — surface/store only when a coarse prototype-margin signal clears a bar.
**How**: A small module mirroring `ml_gate.py`: reuse `embeddings.py` to embed candidate + cached prototype sets, compute the margin, gate `pending` writes. Honest caveat: Somnigraph writes are already LLM-judged, so the marginal value is real only if we want an *automated* capture path that doesn't consult the agent — otherwise this is redundant with the agent's own judgment. Mark as an idea to revisit if/when auto-capture graduates from manual `remember()`.

That is the only transferable idea. The graph, retrieval, dedup, and decay are all weaker than Somnigraph's existing versions.

---

## Not Useful For Us

- **"Hybrid" retrieval / tiered top-k**: it's raw cosine with a channel-concatenation dressed up as hybrid; Somnigraph's RRF + reranker is strictly ahead.
- **Lexical entity dedup (word-subset + Jaccard)**: brittle (no nickname/alias/coreference); if Somnigraph ever adds entity resolution it would want embedding- or LLM-based linking, not this.
- **Curation relevance formula**: `(importance + log(access) + recency)/age` is a reasonable interpretable decay, but Somnigraph's per-category exponential decay with reheat is more principled and already covers this.
- **Provider-wrapper flexibility, offline/local modes, task reminders**: product-packaging concerns irrelevant to a single-user MCP research artifact.

---

## Connections

MemLayer is a **write-gate** system in the same family as the Phase 18 finding (`2026-06-28-phase18-source-sweep.md`) that write-path quality — not retrieval — is where practical memory systems win. It corroborates that thesis by construction: its only real engineering investment is the salience gate, and its retrieval is deliberately thin. It sits below ByteRover/agentmemory/MemPalace (see `ai-memory-comparison.md`) on write-path sophistication (those ground or verbatim-preserve; MemLayer just filters). Its ChromaDB+NetworkX, write-time entity graph, LLM-triple-extraction stack is the same generic shape as many mid-tier r/AIMemory libraries — convergent with the "vector store + entity graph + per-turn extractor" pattern, without the fusion or feedback that distinguish Somnigraph.

---

## Summary Assessment

MemLayer's core contribution is a **cheap, transparent write-time salience gate**: a two-stage regex-then-prototype-margin classifier that decides whether a conversation turn is worth persisting before spending an extraction LLM call. That mechanism is genuinely the point of the library and is the one idea with a thread back to a real Somnigraph gap (no write-path quality gating). Everything downstream — ChromaDB cosine top-k, a write-time NetworkX entity graph with lexical dedup, and a simple curation-decay formula — is competent but well behind Somnigraph's equivalents.

The single most important correction to the carsteneu evidence file: its `hybrid: true` checkmark is technically defensible under a broad definition but materially overstates the system. There is **no fusion** — `SearchService.search()` literally concatenates the vector-result text block and the graph-fact text block (`final_result = f"{vector_context}{graph_context}"`). No RRF, no rank merge, no score combination, no cross-channel dedup. Any reader comparing this to Somnigraph's RRF+reranker should treat MemLayer's "hybrid search" as parallel retrieval with string concatenation. Also worth flagging: `ConsolidationService` is not consolidation in the sleep/reprocessing sense — it is the per-turn write path; there is no offline reprocessing anywhere in the code. And there are no benchmarks, so none of MemLayer's numbers are comparable to our 85.1 LoCoMo QA.

Verdict for the corpus: **MAYBE** — one low-effort idea (prototype-margin salience gate) worth parking against the future auto-capture/proactive-injection work, nothing to adopt now. The system is a well-packaged library, not a research advance.
