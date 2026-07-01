# Octopoda-OS — Multi-tenant agent-memory SaaS (Synrix engine) with hash-chained audit + deterministic loop detection

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

Octopoda-OS is the public repackaging of the **Synrix Memory Engine** (the code, SQLite path `~/.synrix/`, and package internals are all `synrix*`; "Octopoda" is a marketing skin — the top-level `octopoda/` package is an empty stub). It is a **multi-agent memory platform + SaaS**, not a retrieval-quality research system. The interesting engineering is in the *platform* layer (tenancy, audit, loop detection), not the memory ranking.

Two layers:
- `synrix/` — the SDK / client. `Memory`, `SynrixMemory` (agent memory), `FactExtractor`, `VectorIndex`, sqlite/postgres clients, framework integrations (LangChain, CrewAI, AutoGen, OpenAI Agents).
- `synrix_runtime/` — the runtime/cloud platform. FastAPI cloud server with Stripe billing, tenancy, `audit_v2/` (hash-chained trail), `loop_intel_v2/` (loop detector), dashboard, MCP server.

### Storage & Schema

Two backends selected by mode:
- **Local**: SQLite (`synrix/sqlite_client.py`) + an in-memory `VectorIndex` (`synrix/vector_index.py`, numpy matrix or FAISS `IndexFlatIP`).
- **Cloud**: PostgreSQL + pgvector (`init.sql`). Vectors are **384-dim** (local embedding model, bge/MiniLM class), HNSW index `(m=16, ef_construction=200)`, cosine.

Schema (`init.sql`) is a **key-value node store**, not a memory-unit store:
- `nodes(id, tenant_id, name, data JSONB, metadata JSONB, embedding vector(384), valid_from, valid_until, created_at)` — `name` is the memory *key* (e.g. `agents:{id}:user_name`); `data` is arbitrary JSON. Versioning is via `valid_from`/`valid_until` with a unique index per `(tenant, name, valid_from)`; the active row is `valid_until = 0`.
- `fact_embeddings(node_id, node_name, fact_text, category, embedding, collection)` — LLM-decomposed facts, embedded separately.
- `entities(name, entity_type, mention_count, first_seen, last_seen)` and `relationships(source_entity_id, target_entity_id, relation, confidence)` — a **write-time knowledge graph**.
- `tenants`, `api_keys`, `platform_usage`, `tenant_settings` — SaaS plumbing.

There is **no** typed memory-unit schema in Somnigraph's sense (no category/priority/themes/decay_rate/layers). "Schema fields" the evidence file counts (key, value, agent, version, timestamp) are just the KV envelope.

### Memory Types

Effectively none as first-class types. `SynrixMemory` (`synrix/agent_memory.py`) imposes a *convention* — keys like `task:{type}:{attempt}` with values containing "fail"/"error"/"success", filtered by substring match to reconstruct "episodic attempts", "failures", "successes". This is string-matching over a KV store, not a typed episodic/semantic/procedural taxonomy.

### Write Path

The one genuinely interesting memory mechanism. `synrix/fact_extractor.py`:
- On write, raw text is decomposed by an LLM into **self-contained facts** (`EXTRACTION_PROMPT` → JSON array of `"User is vegetarian (food preference, diet)"`-style statements) *before embedding*. Multi-provider: `platform` (Octopoda shared OpenAI key, free tier 100 extractions), `openai` (gpt-4o-mini), `anthropic` (claude-haiku-4-5), `ollama` (llama3.2 local), or `none`. Graceful fallback chain openai/anthropic → ollama → raw text; concurrency-limited by a semaphore.
- Facts land in `fact_embeddings`; entities/relationships extracted into the graph tables.
- Quality gating is minimal: skip text `< 4 words`, skip empty. No salience scoring, no dedup at write time, no contradiction check on the write path.
- Docstring claims fact decomposition lifts semantic search "50% → 88%+" — **no eval in the repo backs this number**.

### Retrieval

Weakest area relative to Somnigraph. Three *separate* modes, no fusion:
1. **Prefix / key lookup** (primary): `query_prefix` on `nodes.name` — this is what `Memory.search`, `SynrixMemory.read/get_last_attempts`, and `Runtime.search` (`runtime.py:613`) all use. It is a `LIKE 'prefix%'` scan, not relevance ranking.
2. **Semantic search**: `recall_similar` → `backend.semantic_search` → single-channel cosine over pgvector HNSW (cloud) or `VectorIndex.search` (local, `vector_index.py:108`, top-k inner product). Requires `octopoda[ai]` extra; silently returns empty if embeddings unavailable.
3. **Filtered / fulltext**: JSONB `gin` index (`idx_nodes_data_gin`) + `search_filtered` (tags/importance/age filters).

There is **no RRF fusion, no learned reranker, no score blending, no graph-conditioned re-ranking on the read path**. The `entities`/`relationships` tables are populated at write time but I found no retrieval path that expands or reranks results through the graph. "searchModes: 3" in the evidence file = three independent modes, not a fused hybrid.

### Consolidation / Processing

`consolidate(similarity_threshold=0.90, dry_run=True)` (`runtime.py:2232`) — **on-demand, user-triggered dedup**, not a scheduled sleep cycle:
- Embeds all agent memories, greedy O(n²) clustering by cosine ≥ 0.90, keeps the newest per cluster, deletes the rest.
- A second **version-churn** pass (added after an audit) walks each key's `get_history` and flags ≥3 versions where ≥half are byte-identical to current — explicitly to make `consolidate()` non-inert against the `recall_write`/`key_overwrite` loop signal it is recommended for.
- No LLM-mediated pairwise relationship classification, no edge creation, no gap analysis, no taxonomy. Purely embedding-cosine + exact-text.

### Lifecycle Management

`forget(key)`, `forget_stale(max_age_seconds=604800)` (age TTL), `forget_by_tag(tag)`, `snapshot(label)`/`restore(label)` (time-travel via full-state copy), and version supersession via `valid_until`. **No biological/exponential decay, no reheat-on-access, no per-category half-lives.** Staleness is a hard age cutoff, not a decay curve.

### Platform features (the actual differentiators)

- **Tamper-evident audit trail** (`synrix_runtime/audit_v2/`): every audit event computes `sha256(prev_hash + canonical_event)`, forming a per-agent hash chain; `verify_chain` walks it and reports breaks. Correlation IDs via `contextvars` (`trace.py`) thread a `trace_id` across agent-to-agent handoffs and LLM calls.
- **Loop detection** (`synrix_runtime/loop_intel_v2/`): **10** deterministic classifiers (README markets "5-signal") — retry, polling, decision_oscillation, cost_inflation, self_correction, ping_pong, tool_nondeterminism, **recall_write**, clarification, reflection. Similarity is stdlib `difflib.SequenceMatcher` (deterministic, no embeddings) so classifiers stay unit-testable. `recall_write` flags read(X)→write(X, ≥0.85-similar)→… churn cycles.
- **Multi-tenancy**: Postgres Row-Level Security ("the trust wall") — `SET LOCAL app.tenant_id`; DB refuses cross-tenant rows even on application bugs.
- Framework integrations, MCP server, cloud dashboard, Stripe billing.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Fact decomposition lifts semantic search "50% → 88%+" | Docstring assertion in `fact_extractor.py` | **Unvalidated** — no eval, dataset, or harness in repo |
| Knowledge-graph auto-extraction | `entities`/`relationships` tables + write-path extraction | **Real at write time** but **unused on the read path** (no graph-ranked retrieval found) |
| "5-signal loop engine" | `loop_intel_v2` has 10 classifiers | **Understated** — code exceeds the claim; classifiers are stdlib-deterministic and testable |
| Tamper-evident audit | sha256 prev-hash chain + `verify_chain` (`audit_v2/storage.py`) | **Validated in code** — genuine per-agent hash chain |
| Multi-tenant isolation | Postgres RLS policies in `init.sql` | **Validated** — RLS is enforced at the DB, not the app |
| `consolidate()` merges near-duplicates | `runtime.py:2232`, cosine ≥0.90 + version-churn | **Validated but shallow** — on-demand, no LLM, no edges |
| Time-travel / supersession | `snapshot`/`restore`, `valid_until` versioning | **Validated** |
| Retrieval quality / QA benchmark | **None** — no LoCoMo/PERMA/R@k numbers anywhere | **Absent** — this system publishes no retrieval or QA benchmark |

---

## Relevance to Somnigraph

### What Octopoda-OS does that Somnigraph doesn't

- **Write-time fact decomposition + entity/relationship extraction** (`fact_extractor.py`, `entities`/`relationships`). Somnigraph builds its graph *during sleep* (`scripts/sleep_nrem.py`), not at write, and does no fact decomposition in `embeddings.py`. This is the write-path-quality axis the Phase 18 sweep flagged as what LoCoMo leaders actually win on.
- **Tamper-evident audit + correlation-ID tracing** — Somnigraph has no audit chain (single-user, not needed, but the mechanism is clean).
- **Deterministic agent-loop detection** — no analog in Somnigraph; `recall_write`/churn detection is conceptually adjacent to feedback-loop self-reinforcement concerns.
- **Multi-tenancy (RLS)** — Somnigraph is single-user by design.

### What Somnigraph does better

- **Retrieval quality, decisively.** Somnigraph fuses BM25 + vector via RRF (`fts.py` + RRF), then a 26-feature LightGBM reranker (`reranker.py`), then PPR graph expansion (`scoring.py`). Octopoda offers prefix lookup **or** single-channel cosine **or** JSONB filter — no fusion, no learned ranking, no graph-conditioned retrieval. There is no reranker and no feedback loop of any kind.
- **Consolidation depth.** Somnigraph's sleep is LLM-mediated (typed edge creation, contradiction/evolution classification, merge/archive, REM gap analysis). Octopoda's `consolidate()` is cosine-threshold dedup + byte-identical churn detection.
- **Decay/lifecycle.** Somnigraph has per-category exponential decay with reheat; Octopoda has a hard age TTL.
- **Measured quality.** Somnigraph reports 85.1% LoCoMo QA and per-query Spearman r=0.70 with GT; Octopoda publishes no retrieval/QA number at all.

---

## Worth Stealing (ranked)

### 1. Write-time fact decomposition before embedding (Medium) — *consider, likely redundant*
**What**: Decompose a raw memory into atomic, self-contained facts (LLM, JSON-array prompt) and embed the facts, not the blob.
**Why**: Directly targets Somnigraph's multi-hop vocabulary gap (the ~88% retrieval ceiling in `docs/multihop-failure-analysis.md`) — the same intuition as Somnigraph's L5b synthetic vocabulary bridges, but applied at *write* time instead of during sleep.
**How**: A pre-embedding step in the write path / `embeddings.py`, or a sleep-phase fact-splitter. **Caveat**: this is the mem0/agentmemory pattern already covered in `research/sources/` — additive corroboration, not a new idea, and Somnigraph deliberately defers graph/enrichment to sleep. File under "another data point for the write-path thesis," not a new adoption.

### 2. `recall_write` / version-churn detector as a redundancy signal (Low) — *note-only*
**What**: Deterministic detector for read(X)→write(X, near-identical) churn and "N byte-identical versions of one key."
**Why**: A cheap flag for redundant writes that a feedback/consolidation pass could act on.
**How**: `difflib`-ratio comparison over recent write history. **But** Somnigraph's sleep merge/archive already covers redundancy at higher fidelity (LLM + embedding), so this is marginal.

---

## Not Useful For Us

- **Multi-tenancy / RLS, Stripe billing, cloud server, dashboard** — SaaS concerns; Somnigraph is single-user research.
- **Hash-chained audit trail** — elegant, but Somnigraph has no adversarial/compliance threat model to justify it.
- **Framework integrations (LangChain/CrewAI/AutoGen)** — Somnigraph is MCP-native and Claude-Code-specific.
- **Prefix-lookup "memory" API** — strictly weaker than Somnigraph's hybrid retrieval.

---

## Connections

- **Write-path-quality thesis**: reinforces the Phase 18 finding (`docs/sessions/2026-06-28-phase18-source-sweep.md`) and the `agentmemory`/`byterover`/mem0 cluster — LoCoMo/LME leaders win on *write-time grounding*, not retrieval cleverness. Octopoda's fact-decomposition + entity extraction is the same move; its retrieval is deliberately thin.
- **Convergent with the "packaging over an engine" pattern** seen in other repos: Octopoda markets a name over the `synrix` engine, and the headline "engine" binary (`synrix-server-evaluation`) is a closed download, not in this repo — the open code is the SDK + platform shell.
- **Loop detection** is a genuinely orthogonal capability not present in any other system in the corpus; adjacent to Somnigraph's concern about feedback-loop self-reinforcement (`docs/proactive-injection.md`), though solved for a different problem (agent runtime cost, not retrieval feedback).

---

## Summary Assessment

Octopoda-OS is a **multi-agent memory SaaS platform**, and its real engineering investment is in operations — Postgres RLS tenancy, a sha256 hash-chained audit trail, correlation-ID tracing, and a 10-classifier deterministic agent-loop detector. As a *retrieval-quality* memory system it is thin: a key-value node store whose primary access path is prefix lookup, with optional single-channel cosine and JSONB filtering, **no fusion, no reranker, no feedback loop, and no graph-conditioned retrieval** despite maintaining entity/relationship tables. Its `consolidate()` is on-demand cosine dedup, not an LLM sleep cycle, and lifecycle is a hard age TTL, not decay. It publishes no LoCoMo/PERMA/R@k number, so there is nothing benchmark-comparable to Somnigraph's 85.1% LoCoMo QA.

The single most transferable idea is **write-time fact decomposition + entity extraction** — but that is the mem0/agentmemory pattern already in the corpus, and it corroborates rather than extends the Phase 18 write-path-quality thesis. Somnigraph deliberately defers enrichment to sleep, so this is a data point, not an adoption.

Overhype to flag: the "50%→88%+" search-quality claim is an unbacked docstring assertion with no eval in the repo; the "5-signal" loop engine is actually 10 classifiers (under-, not over-claimed); and the "Octopoda-OS" product is a skin over the closed-source `synrix-server-evaluation` engine binary, with the open repo being the SDK + platform shell. Net: **MAYBE** — worth one revisit only for the write-path fact-extraction corroboration; nothing in the retrieval or consolidation design improves on what Somnigraph already ships.

### Evidence-file cross-check

The carsteneu evidence file (`evidence/octopoda.md`) is a "corrections to data.js" audit, not a benchmark. Its feature flips are accurate against the code (MIT license, local web dashboard, export/import, `forget()`, KG auto-extract, `consolidate()` dedup, snapshot/restore time-travel, openclaw skill dir, and the Synrix→Octopoda rebrand). Sharpest corrections: (1) `searchModes: 3` counts three **independent** modes (prefix/fulltext, semantic, filtered) — it is **not** a fused hybrid and there is no reranker, so it must not be read as retrieval parity with a hybrid system; (2) the evidence file records no benchmark because there is **none** — no LoCoMo QA, no R@k, nothing comparable to Somnigraph's 85.1%; (3) "autoExtract: true" is real at write time but the extracted graph is **not used on the read path**, so KG-extraction ≠ graph retrieval; (4) the loop detector is 10 classifiers, not the "5" the README markets.
