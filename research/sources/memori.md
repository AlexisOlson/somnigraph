# Memori - Cloud-extraction memory SDK; open client is a thin dense+BM25 fact store, the intelligence is a closed server-side augmentation API

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

Memori (MemoriLabs, Apache-2.0, Python + TypeScript + a Rust `core/` engine) markets itself as "memory from what agents do, not just what they say." Reading the code, the open-source repo is a **thin client SDK around a proprietary cloud API** (`https://api.memorilabs.ai`). The parts that make memories good — fact/triple extraction, "Advanced Augmentation" — run server-side and are not in the repo. The SDK captures conversation, ships it to the cloud, receives facts + semantic triples back, embeds them locally, and stores/retrieves them. There is a local (BYODB) storage layer and a local hybrid search, but no local extraction.

### Storage & Schema
Pluggable SQL/NoSQL backends (SQLite, Postgres, MySQL, Oracle, MongoDB, TiDB, CockroachDB, OceanBase) via `memori/storage/drivers/*`. SQLite schema (`memori/storage/migrations/_sqlite.py`) is the reference:
- **Attribution/scoping tables**: `memori_entity` (the user/thing), `memori_process` (the agent), `memori_session`, `memori_conversation`, `memori_conversation_message` (columns: `role`, `type`, `content` — plain text, no tool-call structure).
- **`memori_entity_fact`** — the retrievable memory unit: `content` (text), `content_embedding` (BLOB), `num_times` (mention count), `date_last_time`, `uniq` (dedup key, `UNIQUE(entity_id, uniq)`).
- **`memori_process_attribute`** — per-agent facts (same shape).
- **Knowledge graph**: `memori_subject` / `memori_predicate` / `memori_object` / `memori_knowledge_graph` (subject/predicate/object triple with `num_times` dedup, `UNIQUE(entity_id, subject_id, predicate_id, object_id)`).
- **`memori_entity_fact_mention`** (migration v2) — links a fact to the conversation(s) that produced it (light provenance).

No decay column, no `valid_until`/supersession, no priority/salience field beyond `num_times`, no category taxonomy.

### Memory Types
Two first-class stores: **entity facts** (embedded text strings) and **semantic triples** (S-P-O knowledge graph). The cloud augmentation additionally advertises facts / preferences / skills / attributes / people / rules / events (README "Advanced Augmentation"), but locally these all land as either `entity_fact` content or `process_attribute` content — there is no per-type metadata field, so the taxonomy is not queryable client-side. Scoping is 3-level: entity / process / session (attribution, not memory tiers).

### Write Path
`memori/memory/augmentation/augmentations/memori/_augmentation.py::AdvancedAugmentation.process()`:
1. Grab conversation summary + the last user/assistant message pair (`_select_messages_for_summary`).
2. `POST` the conversation to the cloud augmentation API (`api.augmentation_async(payload)`). **Extraction happens in the closed cloud.**
3. Response returns `entity.facts` and/or `entity.triples`; if only triples are returned, facts are synthesized as `"{subject} {predicate} {object}"` (`build_fact_text_from_triple_entry`).
4. Facts embedded locally (`_embed_facts` → OpenAI/TEI/ONNX), then `entity_fact.create` + `knowledge_graph.create` writes scheduled.
Dedup: fact-level via the `uniq` unique key incrementing `num_times` (contrary to the evidence file, which claims dedup is triple-level only). No quality gating, no salience threshold, no contradiction check — add-only accumulation. If no cloud API key/attribution, **no memories are made at all** (README: "If you do not provide any attribution, Memori cannot make memories for you").

### Retrieval
This is the most interesting local code and where the evidence file is wrong. `memori/memory/recall.py` → `memori/search/_api.py::search_facts` → `_core.py::search_entity_facts_core`:
1. Fetch up to `embeddings_limit` (default 1000) fact embeddings for the entity.
2. Dense candidate pool via cosine (`_faiss.find_similar_embeddings`), pool size `max(limit, min(rows, max(limit*10, 50)))`.
3. **Hybrid re-rank** (`_rank_candidates`): `rank_score = w_cos*cosine + w_lex*BM25`, where BM25 (`_lexical.py`, k1=1.2, b=0.75) is computed **per-query over the candidate pool only** (not a persistent FTS index) and max-normalized to [0,1].
4. **Query-length-adaptive weighting** (`dense_lexical_weights`): `w_lex=0.15` default, but `0.30` for queries ≤2 tokens (short queries → trust exact terms more), clamped [0.05, 0.40], env-overridable.
The Rust `core/src/retrieval/pipeline.rs` mirrors this exactly (dense pool → `search_facts` linear fusion). **No learned reranker, no RRF, and — critically — no knowledge-graph traversal at recall.** Triples are flattened into fact text and embedded as facts; the S-P-O tables are never traversed during retrieval in the open code. A cloud recall path (`_cloud_recall`) delegates ranking entirely to the server.

### Consolidation / Processing
None in the open repo. No sleep, no clustering, no narrative synthesis, no merge. There is a `memori_compaction` MCP tool named in cloud docs, but no implementation here. "Augmentation occurs in the background" refers to the cloud API, not a local offline pass.

### Lifecycle Management
None. No decay, no supersession, no contradiction handling, no TTL, no forget/delete tool in the MCP surface. The only time signals are `num_times` (mention frequency) and `date_last_time`. Deletion exists only at the coarse `delete_entity_memories` (wipe all facts + graph for an entity) level.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| LoCoMo 81.95% overall (single-hop 87.87, multi-hop 72.70, open 63.54, temporal 80.37) | `docs/.../benchmark/results.mdx`, arXiv 2603.19935 | Plausible but **reflects the closed cloud extractor**, not this repo. End-to-end QA (LLM-judge), so comparable *in kind* to our 85.1 — and lower. |
| 1,294 tokens/query, ~5% of full context (67% fewer than Zep) | benchmark docs | Plausible; token-efficiency is a real, separate axis from accuracy. Not something the open code demonstrates. |
| "Memory from what agents do, not just what they say" (agent-native tool-call capture) | README, OpenClaw plugin | **Overstated for the open repo.** `memori_conversation_message` stores only `role`/`content`; no tool-call/decision/outcome structure. Tool-call capture lives in the separate OpenClaw/Hermes integrations and is still fed to the cloud as conversation. |
| Hybrid BM25+vector search | `_lexical.py` + `_core.py` | **Present and real** in the code (weighted linear fusion, query-adaptive weight) — the evidence file marks this ❌ (see cross-check). |
| Knowledge-graph-augmented recall | schema + docs | Graph is **written but not read** at recall in open code. Retrieval is dense+BM25 over flattened facts. |
| Multi-backend BYODB | `storage/drivers/*` | Validated — 8 real driver implementations + migrations. Genuinely broad. |

---

## Relevance to Somnigraph

### What Memori does that Somnigraph doesn't
- **Multi-backend storage abstraction** (`storage/drivers/*`, `storage/adapters/*`): SQLAlchemy/Django/DBAPI/Mongo adapters + 8 SQL/NoSQL drivers. Somnigraph is SQLite-only (`db.py`). Not a gap we care about (single-user by design) but a real engineering surface.
- **Multi-tenant attribution model** (entity/process/session) baked into the schema. Somnigraph is single-user; no equivalent and no need.
- **Rust core engine** (`core/`) with Python + Node bindings for the retrieval/augmentation hot path. Somnigraph is pure Python.
- **Query-length-adaptive lexical weighting** — a small ranking idea we don't have as an explicit rule (below).

### What Somnigraph does better
- **Retrieval**: Somnigraph's RRF fusion + 26-feature LightGBM reranker (`reranker.py`, `scoring.py`) is a generation beyond Memori's fixed `w_cos*cos + w_lex*bm25` linear blend. Memori has no learned ranking, no PPR graph expansion, no feedback signal.
- **Consolidation**: `sleep_nrem.py`/`sleep_rem.py` (typed-edge detection, merge/archive, gap analysis) have no counterpart — Memori's open code is add-only.
- **Lifecycle**: per-category decay + reheat (`scoring.py`) vs. Memori's zero decay.
- **Feedback loop**: explicit utility ratings with measured r=0.70 GT correlation vs. Memori's none.
- **Graph actually used**: our typed edges feed PPR expansion + a betweenness reranker feature; Memori's triple store is inert at query time.
- **Self-contained**: Somnigraph's quality is in the repo. Memori's headline quality (extraction that wins LoCoMo) is behind a cloud API.

---

## Worth Stealing (ranked)

### 1. Query-length-adaptive lexical weight (effort: Low)
**What**: `dense_lexical_weights` raises the BM25 weight from 0.15 to 0.30 when the query is ≤2 content tokens, on the theory that short queries are dominated by high-signal exact terms where lexical match beats fuzzy semantic match.
**Why**: Our multi-hop failure analysis pins ~88% of the ceiling on a vocabulary gap; the *inverse* case (very short, keyword-y queries) is where dense over-generalizes and BM25 should win. This is a cheap, interpretable prior.
**How**: We fuse with RRF (rank-based), so we can't weight-tilt the same way — but query token count is a natural **reranker feature** in `reranker.py`, or an RRF-k modulation (smaller k for the BM25 channel on short queries). Given the learned reranker already sees query features, this is likely subsumed; worth a LOFO check rather than a new rule.

### 2. Write-side mention-count salience (`num_times`) (effort: Low)
**What**: Each fact carries `num_times` = how many distinct conversations produced it (incremented on dedup), plus `date_last_time`. A frequency-of-*extraction* signal, distinct from frequency-of-*retrieval*.
**Why**: Somnigraph's salience signals are retrieval-side (session_retrievals, feedback EWMA). "How often did the world re-assert this fact" is an orthogonal, write-time salience prior that could help cold memories that are important but never yet retrieved.
**How**: Add an extraction/dedup counter on remember() when a near-duplicate is stored, expose as a reranker feature. Additive to the existing feature set; low risk, uncertain payoff.

---

## Not Useful For Us

- **Cloud augmentation API dependency**: the extraction engine is closed and network-bound. Somnigraph's whole thesis is local, inspectable, single-user.
- **Multi-backend driver matrix**: 8 database drivers is product surface for a hosted company, pure liability for us.
- **Entity/process/session tenancy**: multi-user scoping we explicitly don't want.
- **Rust core**: performance engineering for scale we don't have.

---

## Connections

- **Convergent with the Phase 18 write-path finding** (see `ai-memory-comparison.md`, `byterover.md`, `agentmemory.md`): Memori is another data point that **write-path extraction quality, not retrieval machinery, is what wins LoCoMo** — its open retrieval is a plain linear BM25+dense blend, yet the cloud extractor scores 81.95. Independent corroboration that our reranker sophistication is over-indexed relative to write-time quality.
- **Contrast with graph systems** (see `memos.md`, `mirix.md`): Memori stores S-P-O triples like a graph system but, unlike them, never traverses the graph at recall — a cautionary example of graph-as-decoration.
- **Same "flatten triples into embedded text" shortcut** seen in several profiled systems; convergent evidence that write-time triples mostly serve as a vocabulary bridge for dense retrieval, echoing our L5b synthetic-node finding in `docs/benchmarks.md`.

---

## Summary Assessment

Memori's real product is a **hosted memory-as-a-service**: a proprietary "Advanced Augmentation" cloud pipeline that distills conversations into facts + triples, wrapped in a genuinely broad, multi-backend, multi-language client SDK. The open-source repo is the wrapper, not the engine. Its local retrieval is a competent-but-basic dense-candidate + max-normalized-BM25 linear fusion with one nice touch (query-length-adaptive lexical weighting) — no RRF, no learned reranker, no graph traversal, no consolidation, no decay, no feedback. Architecturally it sits a full generation behind Somnigraph on every retrieval and lifecycle axis we care about.

The single most important takeaway is negative and corroborating: a system whose *open* retrieval is this plain still posts a competitive LoCoMo QA number (81.95, below our 85.1), which reinforces the Phase 18 thesis that **extraction/write quality dominates retrieval cleverness**. The two small ideas worth a look — adaptive lexical weighting and write-side mention-count salience — are both plausibly already subsumed by our learned reranker and rate as "consider," not "adopt."

What's overhyped: the "memory from what agents do, not just what they say" tagline is not realized in the open code (conversation messages only, no tool-call structure), and the knowledge graph is written but never read at query time. What's missing relative to us: everything in the lifecycle and feedback layers. Verdict: MAYBE, leaning SKIP — read for the write-path corroboration and the one ranking heuristic, not for anything to adopt wholesale.

---

## Evidence File Cross-Check

The carsteneu evidence file (`evidence/memori.md`, audited 2026-05-28) was built from **docs, not the source tree**, and gets three things wrong against the actual code:

1. **Hybrid (BM25+Vec) marked ❌** — but the code demonstrably implements it: `memori/search/_lexical.py` computes BM25 (k1=1.2, b=0.75, max-normalized) and `memori/search/_core.py::_rank_candidates` fuses it as `w_cos*cosine + w_lex*BM25` with query-adaptive weights. The Rust `core/` mirrors it. This is a weighted-linear hybrid (not RRF), but it is unambiguously present. **Sharpest correction.**
2. **"Recall uses semantic search + knowledge graph traversal"** — the open retrieval path never traverses the S-P-O tables; triples are flattened to fact text and embedded, and only fact embeddings are searched. The graph is write-only at the SDK level.
3. **"Dedup is triple-level only"** — fact-level dedup also exists via the `uniq` unique constraint on `memori_entity_fact` incrementing `num_times`.

The evidence file's lifecycle findings (no decay/supersede/contradiction/forget, add-only) match the code exactly. The LoCoMo 81.95 cell is genuine end-to-end QA (LLM-judge) and thus comparable in kind to our 85.1 — but it is produced by the **closed cloud extractor**, not reproducible from this repo, a caveat the evidence file's benchmark section does not flag.
