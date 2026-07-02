# ArcRift — Local-first browser/MCP memory for chat platforms, with sentence-level "surgical trim" retrieval

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

ArcRift is a consumer-facing product, not a research artifact: a Chrome extension + Tauri desktop app + Node backend (MCP server) that scrapes ChatGPT/Claude/Gemini web conversations, extracts memory, and re-injects context into the prompt box on later turns. TypeScript throughout. Local-first via Ollama (embeddings + graph extraction), no cloud required. v1.6.x, MIT, ~150 stars.

### Storage & Schema
Two backends selected by `ARCRIFT_STORAGE_MODE`:
- **SQLite mode (default, "Zero-Docker")**: `better-sqlite3` + `sqlite-vec` vec0 virtual tables (768-dim float32, `nomic-embed-text`) + FTS5. Tables: `vec_chunks`/`chunk_metadata`/`fts_chunks` (chunk layer), `vec_sentences`/`sentence_metadata` (sentence layer), `facts` (graph triples), sessions.
- **Docker mode**: ChromaDB (vectors) + Neo4j (graph) + MongoDB (full chat). Same logical schema.

Memory unit = a **session** (one project/conversation). Within it: sliding-window chunks (300 words, 80 overlap, `chunker.ts`), per-sentence embeddings, and LLM-extracted triples. No priority, no decay_rate, no valid_from/until, no category taxonomy on the chunk side. The graph has 22 entity types and 20+ relation types (`extractor.ts`).

### Memory Types
No episodic/semantic/procedural typing. Two parallel representations of the same text: (1) **vector chunks/sentences** (unstructured) and (2) **knowledge-graph triples** (structured subject-relation-object). Triples carry the type system; chunks do not.

### Write Path (`chat.ts` → `storage.ts`, `extractor.ts`)
DOM scrape → **FNV-1a fingerprint dedup** (skip re-saving identical turns) → **PII scrub in-browser** (`privacy.ts`: API keys, JWTs, connection strings, emails → `[REDACTED]`) → POST `/api/chat/save`. Then two tracks:
- **Vector track**: `slidingWindowChunks()` (pure function, zero API calls, zero data loss) → `generateEmbeddings()` parallel via Ollama → DELETE+INSERT (clean re-save). Sentence indexing is offloaded to a **background job** (`jobs.ts`) so save stays fast.
- **Graph track**: `summarizeChunk()` (LLM "extract ALL facts") → `extractTriplesFromSummary()` (LLM → JSON triples with a large hand-written type/relation prompt). Dedup by exact `subject|relation|object` key.

No salience/quality gating — the prompt explicitly says "Extract ALL relationships. If in doubt, include it." No entity resolution beyond exact string match.

### Retrieval (`sqlite-vector.ts::retrieveRelevantChunks`, `rag.ts`)
On every prompt (debounced 300ms):
1. **HyDE** (`hyde.ts`): generate a hypothetical answer, concat `query + hydeAnswer`, embed the combination.
2. **Three channels in parallel**: sentence-vector search (`vec_sentences` MATCH, k=100), chunk-vector search (`vec_chunks` MATCH, k=20), FTS5 keyword (prefix-OR: `word*`).
3. **Fusion** = a candidate `Map` keyed by `chunk_id`; each channel unions in its matched sentences and takes `max` score. **Not RRF, not a learned reranker** — max-score union with engine-provenance tracking. Score = `exp(-L2_distance / 20)`, thresholded (0.30 in-store, then 0.50 at the route to cut hallucination).
4. **"Surgical trim"**: the returned `content` is the *union of sentences that matched the vector search*, not the whole chunk. If no sentence matched but the chunk scored high, fall back to first 3 sentences.
5. **Graph enrichment**: LLM extracts entities from the query → `findRelatedTriples()` = single SQL `WHERE subject IN (...) OR object IN (...) LIMIT 15`. **1-hop lookup, no traversal, no PPR.** Prepended as "RELATED KNOWLEDGE".
6. Char-budget fill (6000 chars), sanitize (10 injection patterns → `[Content redacted]`), wrap, inject.

Note: `getRelevantSentences()` (a *keyword*-based sentence trimmer) is defined but **dead code** — the live trimming is purely the vector `vec_sentences` path.

### Consolidation / Processing
**None.** No sleep cycle, no offline reprocessing, no merge/archive of memories, no edge detection. The only background job is sentence-embedding indexing. Project summaries are LLM-regenerated and cached, invalidated on triple-count change.

### Lifecycle Management
**No decay, no versioning, no reheat.** Deletion only: `prune_memory` (LLM entity-extract the prompt → delete matching triples + `content LIKE '%prompt%'` chunk delete) and clean re-save (DELETE+INSERT per session). Facts accrete monotonically until manually pruned.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| "Surgical sentence trimming reduces prompt noise up to 95%" | Self-run `mcp-benchmark.ts` "context compression" on synthetic input | Plausible as compression ratio; it's chars-in/chars-out, not a QA-quality measure |
| "90% web recall@1, 0.806 MRR" | Self-run `rag-audit.ts` against a 1,000-chunk synthetic noise haystack | **Retrieval-recall on self-generated data — NOT end-to-end QA, not comparable to LoCoMo QA** |
| "90% MCP recall, 81.3% noise redaction" | Self-run `mcp-benchmark.ts`/`mcp-stress-test.ts` | Self-defined harness; no external/standard benchmark |
| Hybrid search (vector + graph + FTS5) | Real in `sqlite-vector.ts` + `rag.ts` | Confirmed, but fusion is naive max-score union, not RRF/rerank |
| Local/private via Ollama | `extractor.ts` backend auto-detect | Confirmed; Groq fallback leaks PII-scrubbed text if Ollama absent |
| 100% project isolation | `mcp-stress-test.ts`, `sessionId` scoping in every query | Confirmed at code level (all queries filter `sessionId`) |

---

## Relevance to Somnigraph

### What ArcRift does that Somnigraph doesn't
- **Sentence-granular retrieval ("small-to-big")**: indexes and vector-searches at sentence level, returns only matching sentences with the chunk as anchor. Somnigraph returns whole memory rows (`tools.py` recall). For Somnigraph's already-atomic memories this is low-value, but the pattern matters for any future long-form/document memory.
- **HyDE query expansion** (`hyde.ts`): hallucinate an answer, embed query+answer. Somnigraph's recall path (`embeddings.py`/`tools.py`) embeds the bare query. This directly targets the vocabulary gap Somnigraph documented as its multi-hop ceiling (`docs/benchmarks.md`, ~88%).
- **In-browser PII scrubbing before persistence** (`privacy.ts`) — Somnigraph has no write-path redaction (single-user, less critical, but still a gap).
- **Write-time FNV-1a fingerprint dedup** at scrape time — cheap exact-dup suppression before any embedding.

### What Somnigraph does better
Nearly everything on the retrieval-science axis. Somnigraph has a **26-feature LightGBM reranker** (`reranker.py`) vs ArcRift's max-score union; **RRF fusion** (`scoring.py`) vs naive merge; a **measured feedback loop** (Spearman r=0.70) vs no feedback at all; **LLM-mediated sleep consolidation** (`sleep_nrem.py`/`sleep_rem.py`) vs no consolidation; **typed graph edges + PPR expansion** detected during sleep vs a 1-hop `IN (...)` SQL lookup with no traversal; **per-category decay/reheat** vs monotonic accretion; and **85.1% end-to-end LoCoMo QA (Opus judge)** vs self-run recall@k on synthetic haystacks. ArcRift's graph is entity-string-match only — no entity resolution, no edge typing beyond the relation label, no centrality.

---

## Worth Stealing (ranked)

### 1. HyDE query expansion for the multi-hop vocabulary gap (Medium)
**What**: Before embedding a recall query, generate a short hypothetical answer with a cheap LLM and embed `query + hypothetical` (ArcRift `hyde.ts`, used in `retrieveRelevantChunks`).
**Why**: Somnigraph's own `docs/benchmarks.md` names an ~88% vocabulary-gap ceiling as the retrieval limit. HyDE is the query-time analog of Somnigraph's *write-time* synthetic vocabulary bridges (L5b synthetic nodes) — it manufactures answer-vocabulary at read time to bridge query↔memory lexical mismatch. Convergent evidence: two independent systems attacking the same gap from opposite ends of the pipeline.
**How**: Add an optional HyDE step in the recall path (`tools.py`/`embeddings.py`): one small local/cheap LLM call → embed the concatenation → feed into existing RRF+reranker. Gate behind a flag; the cost is one extra LLM call + latency on a currently non-LLM recall path, so measure the recall@k lift on the multi-hop slice before making it default. Its write-time cousin (synthetic bridges) already works, which makes the query-time version worth an ablation rather than a leap of faith.

### 2. Sentence-level "small-to-big" trim as a future long-document primitive (Low, note-only)
**What**: Index sentences separately from chunks; vector-search at sentence granularity but return the sentence(s) plus enough chunk context, instead of the whole passage.
**Why**: Not useful for Somnigraph's current atomic memories, but if Somnigraph ever ingests long documents/transcripts, this is a clean noise-reduction pattern (return the matched line, not the paragraph). File it against a hypothetical document-memory layer, not today's schema.
**How**: Would require a sentence sub-index parallel to the memory row — deliberately out of scope now.

---

## Not Useful For Us

### Browser-extension / DOM-scraping / selector-resolver layer
ArcRift's bulk (extension, platform selectors, Tauri UI, dashboard) is product plumbing for capturing web-chat conversations. Somnigraph is MCP-native and single-user; none of it transfers.

### The self-run benchmark suite
`rag-audit.ts`/`mcp-benchmark.ts` measure retrieval recall and char-compression against synthetically generated haystacks the same repo produces. Not an external benchmark, not end-to-end QA, and not comparable to Somnigraph's LoCoMo QA numbers.

### Naive max-score fusion + 1-hop SQL graph lookup
Strictly weaker than Somnigraph's RRF+reranker and PPR expansion. Nothing to port.

---

## Connections

- **HyDE + write-time synthetic bridges**: the strongest connection — ArcRift's query-time HyDE is the mirror image of Somnigraph's L5b synthetic vocabulary nodes (both bridge the multi-hop vocabulary gap). Two systems, opposite ends of the pipeline, same target.
- **Write-path-quality convergence (inverted)**: the Phase 18 sweep (`ai-memory-comparison.md`, ByteRover, agentmemory, MemPalace) found that write-path quality, not retrieval cleverness, is what LoCoMo leaders win on. ArcRift is a *counterexample-by-omission*: it has zero write-path quality gating ("extract ALL, if in doubt include it") and reports only self-run synthetic recall — reinforcing that undisciplined write-time extraction plus self-defined benchmarks is exactly the profile the sweep warned about.
- **Local-first Ollama + sqlite-vec + FTS5 stack** echoes several profiled local systems; the storage substrate is now a commodity. The differentiation lives entirely above it (reranker/feedback/consolidation), which ArcRift lacks.

---

## Summary Assessment

ArcRift is a competently built **consumer memory product** — local-first, privacy-conscious, easy to install, wired into real chat UIs via extension and MCP. Its engineering (background sentence indexing, FNV-1a dedup, in-browser PII scrub, multi-backend storage, injection sanitizing) is solid product work. But as a *memory-retrieval research artifact* it is thin: fusion is a max-score union (no RRF, no learned ranking), the "knowledge graph" is a 1-hop `WHERE ... IN (...)` SQL lookup with no traversal or entity resolution, there is no consolidation, no decay, no feedback loop, and the headline "90% recall / 95% compression" numbers are self-run retrieval-recall/compression against haystacks the repo itself generates — **not end-to-end QA and not comparable to Somnigraph's 85.1% LoCoMo**. The evidence file's mechanism descriptions are accurate; the sharpest needed correction is that its benchmark cells are self-defined recall/compression, not QA.

The single thing worth Somnigraph's attention is **HyDE**: query-time answer-hallucination to bridge the exact multi-hop vocabulary gap Somnigraph has already diagnosed and attacked from the write side. That's a genuine, testable idea with a clear home in `tools.py`/`embeddings.py` and a natural ablation (compare against, or stack with, the existing L5b synthetic bridges). Everything else is either strictly weaker than what Somnigraph already runs or is product plumbing irrelevant to a single-user MCP server. Verdict: **MAYBE** — one revisit-if angle (HyDE), nothing to adopt wholesale.
