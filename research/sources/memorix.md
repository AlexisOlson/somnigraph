# memorix - early-stage vector-store SDK skeleton (store/retrieve/update/delete) with unimplemented embedders

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

<!-- No paper. Skipping Paper Overview. -->

## Architecture

Repo recovered: the sweep URL `memorix-ai/memorix` is 404; the actual code is at **`memorix-ai/memorix-sdk`** (named in the evidence file). Cloned and read in full — the SDK is ~640 lines of Python across 5 modules.

Memorix is **not an agent memory system**. It is a thin, generic key-value + vector wrapper library (`store`/`retrieve`/`update`/`delete`/`list_memories`) with pluggable backends. Critically, it is also **not functionally complete**: reading the code shows the core pieces are stubs.

### Storage & Schema
- **Vector store** (`vector_store.py`): factory over `FAISSVectorStore` and `QdrantVectorStore`. The "FAISS" store is *not* FAISS — it is three Python dicts (`embeddings`, `contents`, `ids`) with `search()` doing a manual Python-loop cosine similarity over every stored vector (`_cosine_similarity`, lines 82-100). No FAISS import, no ANN index. The Qdrant store's `search()` is a placeholder that `return []` (line 175).
- **Metadata store** (`metadata_store.py`): SQLite / in-memory / JSON-file backends. SQLite schema is a single `metadata(memory_id PK, metadata_json TEXT, created_at, updated_at)` table — metadata is an opaque JSON blob, no columns, no FTS5.
- **Memory unit**: `id` (uuid4), `content`, `embedding`, free-form `metadata` dict. `store()` auto-injects only `timestamp` and `content_length` (`memory_api.py:49-50`). No category, priority, themes, valid_from/until, decay_rate.

### Memory Types
None. There is no type/category taxonomy. Users may put arbitrary keys in the metadata dict, but nothing is system-recognized.

### Write Path
`MemoryAPI.store()` (`memory_api.py:26-53`): generate uuid → `embedder.embed(content)` → vector_store.store → metadata_store.store. **No extraction, no dedup, no enrichment, no salience/quality gating.** It is a passive put — you call it yourself with the exact string to store.

### Retrieval
`MemoryAPI.retrieve()` (`memory_api.py:55-87`): embed the query, cosine top-k against the in-memory dict, enrich each hit with its metadata blob. **Single channel (vector only). No BM25/FTS, no graph, no fusion, no reranking, no scoring formula beyond raw cosine.** `top_k` default 5.
- **The embeddings are fake.** All three embedders (`OpenAIEmbedder`, `GeminiEmbedder`, `SentenceTransformersEmbedder`) return `_dummy_embedding()` — an md5/sha256/sha1 hash of the text mapped to floats (`embedder.py:55-75, 104-124, 149-169`). The real API clients are `None` placeholders ("Would be initialized with openai.OpenAI…"). So retrieval ranks by hash-derived pseudo-vectors, which carry **no semantic signal** — similar sentences do not get similar vectors. Semantic search does not actually work in the shipped code.

### Consolidation / Processing
None. No offline processing, no summarization, no clustering, no sleep cycle.

### Lifecycle Management
None automatic. `update()` re-embeds and overwrites (supersede-by-id); `delete()` removes by id. No decay, no versioning, no archival. Memory expiration is a README roadmap item, not code.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Pluggable vector stores (FAISS, Qdrant) | `vector_store.py` factory | **Questionable** — "FAISS" is a dict+Python-loop; Qdrant search returns `[]`. Neither backend is real. |
| Multiple embedders (OpenAI/Gemini/ST) | `embedder.py` factory | **Unvalidated/stub** — all three return hash-based dummy vectors; no API client wired. |
| Semantic retrieval | `retrieve()` cosine top-k | **Questionable** — mechanism exists but operates on fake embeddings; no real semantic signal. |
| Supersede / explicit forget | `update()`, `delete()` | **Validated** — trivially correct by-id ops. |
| Offline-capable | SQLite + local option | **Partial** — nothing calls the network anyway (embedders are stubbed); with real ST backend it could be offline. |
| Auto-extraction / decay / keywords / full-text / platform integrations | — | **Absent** — confirmed by code; matches evidence-file corrections. |

---

<!-- No PERMA benchmark. Skipping. -->

## Relevance to Somnigraph

### What memorix does that Somnigraph doesn't
Nothing. Every capability memorix exposes (vector put/get, metadata blob, supersede/delete) is a strict subset of Somnigraph's, and Somnigraph implements them for real (working embeddings via `embeddings.py`, real ANN via sqlite-vec in `db.py`).

### What Somnigraph does better
Everything relevant: real embeddings, hybrid BM25+vector RRF fusion (`fts.py`, `scoring.py`), a 26-feature learned reranker (`reranker.py`), graph edges + PPR, sleep consolidation (`scripts/sleep_*.py`), decay, an explicit feedback loop, and a structured schema (`db.py`). memorix has none of these — and its retrieval is non-functional because embeddings are hashed placeholders.

---

## Worth Stealing (ranked)

**None.** The repo is a pre-alpha scaffold: stubbed embedders, a non-FAISS "FAISS" store, and a placeholder Qdrant. There is no mechanism, formula, or design decision here that Somnigraph doesn't already implement more completely. No idea buried in the code survives contact with the actual implementation.

---

## Not Useful For Us

### The whole SDK
It is a generic store/retrieve library at the abstraction level of `chromadb`/`faiss`, minus a working implementation. Somnigraph operates several layers above this and already has real backends.

---

## Connections

- Same "category error" the evidence file flags applies to other thin wrappers in the corpus: this belongs with generic vector-DB SDKs, not agent-memory systems. Contrast with the write-path-quality finding from the Phase 18 sweep (`ai-memory-comparison.md`, ByteRover/agentmemory) — memorix is the anti-example: zero write-path processing, purely a passive put.
- Its supersede-by-id / delete-by-id lifecycle is the minimal version of the richer supersession patterns seen in memv and MIRIX (`mirix.md`).

---

## Summary Assessment

memorix (`memorix-ai/memorix-sdk`) is an early-stage, effectively non-functional vector-store SDK. Reading the code — not just the README — shows all three embedders return hash-based dummy vectors (md5/sha256/sha1) with the real API clients left as `None`, the "FAISS" backend is three in-memory Python dicts with a hand-rolled cosine loop, and the Qdrant backend's search returns an empty list. So the one thing it advertises, semantic retrieval, does not produce semantically meaningful results in the shipped code.

The single most important takeaway for Somnigraph is **nothing to adopt** — this is a corpus-completeness / correct-the-leaderboard entry, not an idea source. The sharpest correction to the comparison table: the evidence file already downgrades most claims to ❌, but even the surviving ✅ ("semantic") is over-generous — semantic search is stubbed, not merely minimal. The org has 0 stars across all repos and most features are roadmap items.

Verdict: SKIP. Zero nuggets. The value of this analysis is documenting that the "generic FAISS/Qdrant wrapper" is actually a placeholder skeleton, so no future sweep re-litigates it.
