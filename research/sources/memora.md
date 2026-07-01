# Memora — MCP memory server with write-time LLM reconciliation, a kNN cross-ref graph, and a live graph UI

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

Memora (github.com/agentic-box/memora, MIT, ~407 stars, v0.2.29 May 2026) is a local-first MCP
memory server in Python with an optional cloud tier (S3/R2 SQLite-file sync, or Cloudflare D1) and
a separate TypeScript web app (`memora-graph/`) that renders a live knowledge-graph UI. It is a
single-user tool like Somnigraph, but architecturally simpler on retrieval and richer on the
write path and UI.

### Storage & Schema
SQLite. Flat `memories` table: `id, content, metadata (JSON), tags (JSON), created_at, updated_at,
importance REAL, last_accessed, access_count` (`schema.py:112`). Satellite tables: `memories_fts`
(FTS5 over content/metadata/tags), `memories_embeddings` (**vectors stored as TEXT/JSON, not
sqlite-vec**), `memories_crossrefs` (the graph — one row per memory holding a JSON list of related
ids), `memories_events` (inter-agent poll queue), `memories_actions` (audit log of
create/update/delete/merge/boost/link), `memories_meta`. No entity table; metadata is a free-form
JSON dict. "Hierarchy" is derived from dotted tags / `section`+`subsection` metadata
(`hierarchy.py`), not a stored tree.

Consequence worth flagging: vectors are JSON text and cosine is computed in Python by scanning every
row (`_search_by_vector` / `storage.py:~1900`). There is no ANN index. Fine at personal scale,
won't scale, and is strictly weaker than Somnigraph's sqlite-vec path.

### Memory Types
No first-class category enum. `_infer_type()` (`server.py:133`) and `_detect_memory_type()`
(`storage.py:173`) classify content into todo/issue/note/idea/question/warning by keyword prefixes
and regex (e.g. `_ISSUE_KEYWORDS`, `_RESOLVED_PATTERNS`). Structured `memory_create_todo` /
`memory_create_issue` tools attach status/priority/severity. This is task-tracking flavor, not
Somnigraph's episodic/semantic/procedural/reflection/meta taxonomy with per-category decay.

### Write Path
This is Memora's strongest area. `absorb_memory()` (`storage.py:3220`) is a reconciling ingest:
1. **Secret redaction** — `_redact_secrets()` (`storage.py:116`) runs ~13 regex patterns
   (OpenAI/Anthropic/AWS keys, bearer tokens, GitHub PATs, Slack tokens, private keys, passwords,
   credit cards) and replaces with `[REDACTED]` **before** anything is stored.
2. **Content validation** — `_validate_content()` trims, collapses `\n{3,}`, enforces min 3 /
   max 50000 chars.
3. **Dedup short-circuit** — embed the fact, kNN search; if top cosine ≥ `_ABSORB_DUPLICATE_THRESHOLD`,
   skip **without** calling the LLM (a deliberate cost gate). Only ambiguous matches go to the LLM.
4. **LLM relationship classification** — `_classify_fact_against_matches()` returns
   DUPLICATE / UPDATE(→supersedes) / CONTRADICT / RELATED; the fact is created with the corresponding
   typed edge, or skipped. Empty LLM output falls through to *create-as-related* rather than dropping
   knowledge.
5. **Consolidation** — remaining pure-new facts are grouped by embedding similarity
   (`_group_facts_by_similarity`) and LLM-synthesized into single richer memories.
6. **Auto cross-ref at write time** — `_update_crossrefs_for_memory()` (`storage.py:2046`) stores the
   top-5 cosine neighbors as `related_to` edges. This is a real-time similarity graph built on every
   write.

### Retrieval
Three modes: `memory_list` (FTS5 + tag/date/metadata filters), `memory_semantic_search` (brute-force
cosine), `memory_hybrid_search` (`storage.py:4208`). Hybrid is a **weighted RRF**: `rrf_k = 60`
(hardcoded), `semantic_weight` default 0.6, keyword_weight = 1−that. Note the non-standard twist — it
adds a raw-cosine `score_boost = semantic_weight * semantic_score * 0.1` term on top of the RRF
reciprocal (`storage.py:4295`), so it isn't pure rank fusion. No learned reranker; no feedback loop.
`importance` (recency decay + log(access) boost, `calculate_importance` `storage.py:4339`, half-life
30d) is available only as an **opt-in sort key** (`sort_by_importance`, default False) and is *not*
folded into hybrid ranking. `memory_digest` assembles a topic bundle (related memories, TODOs, issues,
lineage) for agent consumption.

### Consolidation / Processing
No background/offline cycle — nothing sleep-like. Everything is on-demand tool calls with cooldowns:
`memory_detect_supersessions` (`storage.py:1592`, a 6-way LLM relation enum: a_supersedes_b /
b_supersedes_a / duplicate / related / contradicts / neither), `detect_clusters` (`storage.py:2596`,
**connected-components or Louvain community detection** over the crossref graph),
`memory_find_duplicates`, `memory_merge`. The reconciliation Somnigraph does in NREM sleep, Memora
does synchronously at write (absorb) or as a manually-triggered scan.

### Lifecycle Management
Supersession **lineage** with query-time `follow` modes — `"latest"` (resolve to current version),
`"active"` (exclude superseded), `"full_history"` (walk the whole chain) (`semantic_search`
`follow` param, `apply_follow`). Explicit `memory_delete` / `memory_delete_batch`. **No automatic
decay or forgetting of content** — importance decay is ranking-only and optional. `memory_insights`
flags stale items for human review but does not act on them.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Hybrid RRF search | `hybrid_search` fuses FTS5 + vector, k=60 | Real, but weighted + a non-standard cosine boost term; k untuned |
| Semantic search | `memories_embeddings` + Python cosine | Real but brute-force over TEXT-encoded vectors; no ANN/sqlite-vec |
| Auto knowledge graph | write-time top-5 kNN crossrefs + LLM typed edges on absorb | Real — genuine write-time graph construction |
| LLM dedup / reconciliation | `absorb_memory` + `_classify_fact_against_matches` + 6-way enum | Real and well-structured; the standout feature |
| Secret redaction | `_redact_secrets` 13 patterns pre-storage | Real, cheap, effective write-path safety gate |
| Live graph UI | `memora/graph/server.py` (port 8765) + `memora-graph/` Cloudflare app | Real; the headline demo feature |
| Community/cluster detection | `detect_clusters` connected-components + Louvain | Real, on-demand |
| Decay / forgetting | importance recency factor | Ranking-only, opt-in sort, no content forgetting — evidence file correctly marks decay absent |
| Benchmarks | none published | Correctly reported absent; **no LoCoMo/QA number to compare to our 85.1** |

---

## Relevance to Somnigraph

### What Memora does that Somnigraph doesn't
- **Write-time reconciliation.** Dedup skip, supersede/contradict/related classification, and fact
  consolidation happen synchronously in `absorb_memory`. Somnigraph defers all of this to `sleep_nrem.py`
  (offline), so between sleeps duplicates and contradictions coexist. Different-by-design, but Memora
  proves the synchronous path is viable, and its **embedding-threshold short-circuit before the LLM
  call** is a cost pattern Somnigraph's sleep pass could borrow.
- **Secret/PII redaction gate** (`_redact_secrets`). Somnigraph's `tools.py` remember path has no
  equivalent — anything pasted in (including tool output) is stored verbatim. Genuinely additive.
- **Query-time lineage `follow` modes** (latest/active/full_history). Somnigraph has valid_from/valid_until
  and evolves/revision edges but no recall-time toggle to resolve to the latest version vs. expand full
  history. A small addition to `tools.py` recall.
- **Live graph UI + inter-agent event bus.** `memora/graph/server.py` SSE visualization and a
  `memories_events` poll queue. Somnigraph has neither (single-agent, no UI). Not a research gap.
- **Louvain community detection** over the memory graph — Somnigraph has PPR + betweenness but no
  community labels.

### What Somnigraph does better
- **Learned reranker** (`reranker.py`, 26-feature LightGBM, NDCG 0.7958). Memora has no reranker at all —
  weighted RRF is the final ranking.
- **Feedback loop** with measured Spearman r=0.70 (`tools.py` recall_feedback, EWMA/UCB). Memora has no
  retrieval feedback; `importance` is heuristic and not query-conditioned.
- **Offline LLM-mediated sleep** (`sleep_nrem.py`/`sleep_rem.py`): gap analysis, question generation,
  taxonomy, typed-edge detection with novelty scoring. Memora's graph is kNN-similarity + on-demand LLM.
- **Graph-conditioned retrieval via PPR** (`scoring.py`). Memora's crossref graph feeds the UI and
  clustering but is **not used to re-rank retrieval** — it's decorative at query time.
- **Real vector index** (sqlite-vec) vs Memora's brute-force Python cosine over JSON-text vectors.
- **Per-category biological decay** vs Memora's opt-in, ranking-only importance.

---

## Worth Stealing (ranked)

### 1. Write-time secret/PII redaction gate (Low)
**What**: `_redact_secrets()` — a fixed regex battery (API keys, tokens, private keys, passwords, cards)
that rewrites content to `[REDACTED]` before storage.
**Why**: Somnigraph ingests session context and tool output through `remember`; a leaked key persisted
to the DB (and later surfaced in recall/injection) is a real hazard with zero current guard.
**How**: add a `_redact()` pass in `tools.py` `impl_remember` before the insert; log redaction types in
the memory metadata. Pure add, no schema change.

### 2. Embedding-threshold LLM short-circuit in reconciliation (Low)
**What**: skip the expensive LLM classification when top cosine ≥ a duplicate threshold (~0.85) and
create-as-related on empty LLM output instead of dropping the fact.
**Why**: `sleep_nrem.py` pairwise classification spends LLM calls on obvious dup/near-dup pairs; a
cheap cosine gate cuts cost and the "never silently drop" fallback is a good honest-accounting default.
**How**: pre-filter candidate pairs in the NREM pairwise loop by cosine before the LLM prompt.

### 3. Query-time lineage `follow` modes (Medium)
**What**: recall parameter resolving supersession chains — `latest` / `active` / `full_history`.
**Why**: Somnigraph stores evolves/revision edges but recall returns whatever matched; a caller can't
say "give me only the current version" vs "show the whole evolution."
**How**: post-process recall results in `tools.py` using existing edges to collapse or expand chains.

---

## Not Useful For Us

### Live graph UI + Cloudflare web app (`memora-graph/`)
Somnigraph is a headless single-agent research artifact; a D3/SSE visualization and R2/D1 sync layer
are product surface, not research value.

### Inter-agent event poll queue (`memories_events`)
Multi-agent coordination; Somnigraph is single-user.

### TEXT-encoded embeddings + brute-force cosine
Strictly worse than the existing sqlite-vec path; nothing to take.

---

## Connections

Memora is the clearest independent corroboration yet of the **Phase 18 write-path finding**
(`docs/sessions/2026-06-28-phase18-source-sweep.md`): the leaders invest in write-time quality, not
exotic retrieval. Memora has *no* reranker and *no* feedback loop yet ships a genuinely disciplined
absorb path (redact → validate → dedup-skip → LLM-classify → consolidate). Its synchronous
supersede/contradict/consolidate mirrors, in timing-inverted form, Somnigraph's NREM sleep — same
operations, write-time vs offline. The 6-way relation enum echoes memv's supersession pattern and our
own typed-edge set. Its brute-force vector store and untuned k=60 RRF put it below the RRF+reranker
tier (cf. our RRF k=14 + LightGBM), consistent with agentmemory/ByteRover: strong write path can carry
a plain retriever a long way.

---

## Summary Assessment

Memora's core contribution is a **well-engineered synchronous write path** — secret redaction,
LLM-mediated dedup/supersede/contradict reconciliation with a cheap cosine short-circuit, fact
consolidation, and write-time kNN graph construction — wrapped in a polished live-graph UI and an
optional cloud tier. It is a competent, product-minded MCP memory server, not a research system: no
learned reranker, no retrieval feedback loop, no offline consolidation cycle, brute-force Python cosine
over JSON-text vectors, and a hardcoded weighted RRF with a non-standard cosine-boost term. The
evidence file is honest — it correctly reports no published benchmarks and correctly lists decay,
autoExtract, entities, and learned reranking as absent — so there is no benchmark-comparability trap
here.

The single most valuable takeaway is the **write-time redaction gate**: cheap, safety-relevant, and a
real hole in Somnigraph's `tools.py` remember path. Secondary: the embedding-threshold LLM
short-circuit (a cost pattern for `sleep_nrem.py`) and query-time lineage `follow` modes. Nothing here
is an adopt-worthy *core* idea Somnigraph lacks conceptually — Somnigraph already has the harder
machinery (learned reranker, feedback loop, PPR, sleep). Verdict: MAYBE, on the strength of the
redaction gate and two small additive mechanisms, not a paradigm.

**Evidence cross-check**: The carsteneu evidence is accurate and unusually honest; no inflated
benchmark cells (none exist). Sharpest corrections it omits: (1) vector search is brute-force Python
cosine over TEXT-encoded embeddings with no ANN index — a scaling wall it doesn't mention; (2) the
"hybrid RRF ✅" is a *weighted* RRF plus a non-standard raw-cosine boost term with an untuned k=60, not
textbook RRF; (3) its `importance` has a recency-decay factor (evidence marks decay absent, which is
right for *forgetting*, but the ranking-time decay exists and is opt-in via `sort_by_importance`).
