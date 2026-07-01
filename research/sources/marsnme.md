# MarsNMe — Agent-agnostic MCP memory gateway over Supabase/pgvector with a short-term-TTL → promoted-chunk two-tier store

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

MarsNMe is not a retrieval-research system; it is a **cross-platform memory continuity gateway**. Its pitch is: one user-owned Supabase, many AI front-ends (Claude Desktop, Cursor, Perplexity, Warp, OpenClaw, Hermes), sharing memory over MCP. The repo ships three deployables:

- `soul-memory/server.mjs` (5040 LOC) — the real product: a hand-rolled Node MCP gateway that proxies to Supabase Postgres + pgvector via PostgREST RPC. This is what I read closely.
- `marsnme-local/` — a Cloudflare Workers + D1 + Vectorize variant (secondary, ~170 LOC of tool code).
- `supabase/migrations/` — the canonical schema (11 migrations).

### Storage & Schema

Postgres, two schemas `coco` and `toto` — these are **hard-isolated agent "bodies"/personas**, each with its own MCP port (18790/18791) and its own copy of every table. No cross-body write path (enforced by design, repeated in code comments).

Two tiers per body:
- `memories` — short-term. Columns: `body`, `source` (CHECK-constrained whitelist: perplexity/cursor/warp/openclaw/intent/hermes), `session_id`, `tags[]`, `promoted`, `created_at`, `expires_at DEFAULT now()+30 days`, `embedding vector(1024)`. HNSW cosine index.
- `marsvault_chunks` — long-term persistent. Columns: `content`, `embedding vector(1024)`, `source_file`, `section`, `body`, `visibility` (global/shared/private), `tags[]`, `type` (default 'digest'), `date`, `content_hash`, `origin`, `deprecated_at`, `deprecated_reason`, `superseded_by`, `source_memory_id`. IVFFlat cosine index; unique upsert key `(source_file, section, content_hash, body)`.

Later migrations add: provenance audit trail, a `source_registry` table with regex-validated source constraints and runtime reload, per-row `scope`/`agent_body`/`environment` filters, and usage/cost telemetry. The evidence file's "28 schema fields" is a fair tally across both tables + migrations.

### Memory Types

No cognitive taxonomy (nothing like Somnigraph's episodic/semantic/procedural/reflection/meta). "Types" here are operational: short-term vs long-term (promotion), `type` string on chunks (default 'digest'), and `visibility` tiers. Categorization is by **provenance** (which tool wrote it) and **scope** (which body/environment), not by content role.

### Write Path

Fully explicit, agent-driven. `insert_memory` takes agent-supplied `body` text, embeds it with Jina (`createJinaEmbedding(text, 'retrieval.passage')`), writes the row. `dream_ingest` takes a markdown digest, splits it with `splitDigestContent(raw, maxChunkChars≈1200)`, dedups by `content_hash` unique-upsert, embeds each chunk. **No LLM extraction, no salience/quality gating, no auto-summarization, no entity resolution** — the evidence file's "no auto-extraction" is correct. Dedup is exact `content_hash` only (not semantic).

Embeddings are **asymmetric/task-typed**: `retrieval.passage` at write time, `retrieval.query` at read time (Jina's instruction-tuned dual encoding). Locked to Jina AI (1024-dim); no local/alternative backend in `soul-memory` (the Cloudflare variant uses Workers AI instead).

### Retrieval

**Pure vector cosine, single channel.** `search_memories` and `recall` both: embed the query (`retrieval.query`) → call a Postgres RPC (`search_memories_semantic` / `search_marsvault_chunks_semantic`) that orders by `embedding <=> q` and returns `1 - distance` as similarity. Filters are SQL predicates (source, unexpired-only, visibility, scope, type, `min_similarity` cutoff applied in JS post-hoc). `p_match_count` clamped ≤100/≤50.

There is **no BM25, no hybrid fusion, no RRF, no learned reranker, no graph expansion.** The one lexical-looking piece, `explainRecall()`, tokenizes query + hit content and computes `keywords_overlap` / `matched_spans` — but this is **presentation-only explainability layered on top of already-ranked results; it does not re-order anything.** Ranking is 100% vector similarity.

### Consolidation / Processing

`dream_runner.py` ("Dream Digest") is a **scheduled context-packing/journaling job, not memory reorganization.** It calls MCP tools to collect: recent memories (`list_memories`), semantic memories (a few canned queries), optional GitHub issue signals, an optional local-repo markdown keyword scan, and a "SOUL.md" baseline — then concatenates them into a dated markdown digest and `dream_ingest`s it back as chunks. The "Dream synthesized notes" appended at the end are **hardcoded static strings**, not model output. No pairwise relationship detection, no edge creation, no merge/archive, no gap analysis. This is categorically weaker than Somnigraph's `sleep_nrem`/`sleep_rem`.

`detect_marsvault_conflicts` (SQL, via `health_check`) is the closest thing to consolidation: a LATERAL self-join computing pairwise cosine similarity, returning pairs above a threshold (default 0.85). It is **read-only reporting** — it surfaces near-duplicate candidates for an agent/human to then `demote_memory`; it does not create edges, merge, or resolve anything.

### Lifecycle Management

The most developed part. (1) **Hard 30-day TTL** on short-term `memories` (`expires_at`), filtered out of recall when `unexpired_only`. Not a decay curve — a cliff. (2) **Promotion**: short-term → persistent chunk (`promoted`/`promoted_at`, triggers sync the flags, `batch_promote` tool, `source_memory_id` FK links chunk back to origin). (3) **Supersession/versioning**: `demote_memory` sets `deprecated_at`/`deprecated_reason`/`superseded_by`; the `recall_exclude_deprecated_chunks` migration removes deprecated chunks from recall while preserving them for audit. `soft_forget` is a reversible demote.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Cross-platform memory continuity across 4+ AI tools | Source whitelist + MCP gateway; README anecdote "3 months daily use" | Plausible (design supports it); unmeasured |
| Two-tier short-term→long-term promotion | Confirmed in schema + `batch_promote` + triggers | Validated (code-level) |
| Semantic search via Jina 1024-dim | `createJinaEmbedding`, pgvector HNSW/IVFFlat | Validated |
| Conflict detection via vector similarity | `detect_marsvault_conflicts` SQL, threshold 0.85 | Validated but **read-only reporting, not resolution** |
| Data sovereignty / zero vendor lock-in | All storage in user's Supabase | Validated (but embedding is Jina-locked, a counter-lock-in) |
| Any retrieval-quality / QA benchmark | **None** | Absent — no LoCoMo, no R@k, nothing comparable |

---

## Relevance to Somnigraph

### What MarsNMe does that Somnigraph doesn't

- **Hard TTL staging tier with explicit promotion** — a short-term buffer that self-expires in 30 days unless an agent promotes it. Somnigraph has soft per-category decay + reheat (`scoring.py`/decay), but nothing auto-*removes* low-value memories; everything persists. MarsNMe's cliff is cruder but is a genuine write-path attrition mechanism, touching the gap Somnigraph names as "write-path quality gating."
- **Explicit write-time supersession pointer** (`superseded_by` + recall exclusion of deprecated) — Somnigraph achieves supersession only *offline* via NREM-detected `evolves`/`revision` edges + `valid_until`. MarsNMe makes it a synchronous agent action.
- **Asymmetric task-typed query/passage embeddings** — Somnigraph's `embeddings.py` uses one enriched embedding for both store and query.
- **Multi-body/multi-agent isolation** (coco/toto schemas, scope/environment filters) — Somnigraph is explicitly single-user; this is out of scope for it.

### What Somnigraph does better

Essentially everything on the retrieval-research axis. Somnigraph has hybrid BM25+vector with RRF fusion (`fts.py` + `scoring.py`), a 26-feature LightGBM learned reranker (`reranker.py`, NDCG 0.7958), PPR graph-conditioned expansion over typed edges, an explicit feedback loop with measured GT correlation (r=0.70), and LLM-mediated three-phase sleep consolidation (`sleep_nrem.py`/`sleep_rem.py`). MarsNMe is single-channel vector-only with zero learned ranking, zero graph, zero feedback, and no benchmark. It is a well-engineered *plumbing* layer, not a retrieval-quality artifact.

---

## Worth Stealing (ranked)

### 1. Asymmetric task-typed embeddings for the store vs. query side (Low)
**What**: Embed stored content with a "passage/document" instruction and queries with a "query" instruction, rather than one symmetric embedding for both. MarsNMe does this via Jina's `retrieval.passage`/`retrieval.query` task tags.
**Why**: Somnigraph's fallback/production embedding backend is `fastembed` BAAI/bge-small-en-v1.5, which is trained with exactly this asymmetry (`"query: "` / `"passage: "` prefixes) and typically loses a few points of retrieval quality when queried symmetrically. It's a free, well-established lift on the *first-stage recall* that feeds the reranker — and the multi-hop analysis pegs recall as the current ceiling.
**How**: In `embeddings.py`, add a passage prefix to the enriched stored embedding and a query prefix on the recall path (branch on backend; OpenAI text-embedding-3 ignores prefixes so it's a no-op there, bge benefits). Guard behind the backend check already in `db.py`. Measure on the existing GT set before/after — this is a clean A/B, not a redesign.

---

## Not Useful For Us

- **Two-schema coco/toto body isolation, scope/environment filters, source whitelist + registry** — multi-agent/multi-tool partitioning; Somnigraph is single-user MCP, no application.
- **`dream_runner` digest packing** — a cron journaling job with hardcoded "synthesized" notes; Somnigraph's sleep already does real LLM-mediated consolidation, which is strictly more than this.
- **Hard 30-day TTL cliff** — Somnigraph's soft decay + reheat + priority is a more nuanced version of the same intent; adopting a hard cliff would be a regression.
- **`content_hash` exact-match dedup** — weaker than Somnigraph's ~0.9 semantic-similarity dedup at `remember()`.
- **Cloudflare/Supabase deployment plumbing, OAuth client persistence, Kong service-key extraction** — infra, not memory research.

---

## Connections

- **Convergent supersession** with the `evolves`/`revision`/`valid_until` pattern Somnigraph builds during sleep, and with the explicit-supersession approach seen in memv (see `memv.md`) — MarsNMe corroborates that write-time `superseded_by` + recall-exclusion is a common, simpler alternative to offline edge detection.
- **Write-path-is-the-lever** thesis: MarsNMe is another vector-only, BM25-free system whose actual value is in the write/ingest/lifecycle plumbing, not retrieval — consistent with the Phase 18 source-sweep finding (ByteRover BM25-only, agentmemory write-time grounding) that the leaders win on write-path quality, not fusion cleverness. Here though the write path has *no* quality gating, so it's a weak instance of that pattern.
- **TTL staging tier** echoes the short-term→long-term promotion buffers in MemoryOS / working-vs-long-term splits profiled elsewhere in the corpus.

---

## Summary Assessment

MarsNMe's real contribution is **operational, not algorithmic**: a genuinely useful "one user-owned Supabase behind many AI front-ends over MCP" continuity layer, with careful lifecycle plumbing (TTL staging, promotion, supersession-with-audit, provenance, source whitelisting). If you want your Cursor and Perplexity and Warp sessions to share one memory you control, this is a reasonable thing to run. It's honestly scoped — the README claims platform-agnosticism, not SOTA retrieval.

For Somnigraph the take is thin. On every retrieval-research axis Somnigraph is far ahead (hybrid fusion, learned reranker, feedback loop, graph, real sleep, and — critically — actual benchmarks; MarsNMe has none). The single transferable idea is the **asymmetric passage/query embedding**, which is a low-effort, testable lift for the bge backend and worth an A/B on the existing GT set. Everything else is either already covered by a more sophisticated Somnigraph mechanism (decay > TTL cliff, semantic dedup > content_hash, LLM sleep > digest packing, typed edges > read-only conflict scan) or out of scope (multi-agent isolation).

The evidence file is **accurate and fair** — its checkmarks match the code (vector-only, no graph, no auto-extraction, Jina-locked, no formal benchmark all confirmed). The two sharpenings I'd add: (1) the lexical `explainRecall` overlap is cosmetic and does **not** re-rank, so "vector-only search" is if anything understated; and (2) "conflict detection" is **read-only reporting**, not resolution, and the "dream" consolidation is **static context-packing with hardcoded notes**, not model-mediated reorganization — so this system has *no* autonomous consolidation despite the evocative "dream" branding.

**Verdict: MAYBE** — one revisit-if nugget (asymmetric embeddings), nothing to adopt wholesale.
