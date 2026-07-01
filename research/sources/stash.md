# stash - Go/Postgres agent memory with a rich 8-stage LLM consolidation pipeline (facts, causal links, hypotheses, goals, failures) but pure vector retrieval

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

Stash (github.com/alash3al/stash, ~700 stars, Apache-2.0, Go) is a self-hosted MCP SSE memory server. Its identity is **write-path heavy, retrieval-path thin**: the interesting engineering is a background consolidation pipeline that turns raw episodes into a structured belief graph, while retrieval is plain single-channel vector search.

### Storage & Schema
- **PostgreSQL + pgvector** (not SQLite). `docker-compose.yml` bundles Postgres; the "single binary" is the server only. Migrations `internal/db/migrations/0000X_*.sql`.
- Core tables: `episodes` (raw append-only), `facts` (LLM-synthesized beliefs), `relationships`, `patterns`, `contradictions`, `causal_links`, `hypotheses`, `goals`, `failures`, `contexts`, `namespaces`, `embedding_cache`, `consolidation_progress`.
- **Fact schema** (`internal/models/models.go`, migration 00013): `content`, `confidence` (float), optional `entity`/`property`/`value` (EAV triple, free-text `*string`), `valid_from`/`valid_until` (temporal scoping), `embedding`. ~7 meaningful fields.
- **Namespaces are hierarchical paths** (e.g. `/self`, `/projects/x`); recall over a namespace matches it and all descendants. This is the multi-tenant / multi-scope axis Somnigraph lacks.

### Memory Types
Three-layer abstraction hierarchy (README: "Episodes become facts. Facts become relationships. Relationships become patterns."):
1. **Episodes** — raw observations, append-only.
2. **Facts** — LLM-synthesized beliefs with confidence + optional EAV structure.
3. **Patterns** — higher-order abstractions over facts + relationships.
Plus orthogonal typed records: **contradictions**, **causal_links**, **hypotheses** (proposed→testing→confirmed/rejected state machine), **goals** (active/completed/abandoned hierarchy), **failures** (recurring-mistake records with lessons). This taxonomy is far richer than Somnigraph's 5 flat categories.

### Write Path
The `remember` tool just inserts a raw episode (embed + store). All structure is deferred to consolidation. No write-time quality gating or extraction on the hot path.

### Retrieval
`internal/brain/recall.go` + `internal/queries/recall.sql.tmpl`: **pure pgvector cosine search, single channel.** Facts are queried first (`ORDER BY embedding <=> $vec LIMIT n`), episodes fill remaining slots, results sorted by score. **No BM25/FTS, no hybrid fusion, no RRF, no reranker, no feedback signal, no graph expansion at query time.** A repo-wide grep for `rerank|feedback|rrf|bm25|fts|tsvector|fulltext` returns **zero hits**. The graph (relationships/causal_links) is queried only via explicit `query_relationships` / `trace_causal_chain` tools, never fused into `recall`. This is the sharpest contrast with Somnigraph.

### Consolidation / Processing
The centerpiece. `internal/brain/consolidate.go` `ConsolidateByID` runs 8 stages on a background ticker (`runConsolidationTicker`, configurable interval) or via the `consolidate` tool, per namespace, with per-stage resumable checkpoints (`consolidation_progress`; checkpoint only advances if `len(errs)==0` — "bullet-proof" against losing episodes):
1. **Episodes→Facts**: `clusterEpisodes` (greedy cosine grouping, `SimilarityThreshold=0.85`) → `reasoner.ReasonStructured` LLM extracts an EAV summary per cluster → dedup via `factExistsByVector` (`DedupThreshold=0.85`) → insert with `calculateConfidence(n, hasEAV) = n/(n+2)`, boosted 30% toward 1 if structured. `fact_sources` links provenance.
2. **Facts→Relationships**: LLM extracts `(from_entity, relation_type, to_entity)` edges per fact.
3. **Facts→Causal Links**: LLM extracts cause→effect pairs across a batch of facts (`DetectCausalLinks`).
4. **Goal Progress Inference**: annotate/suggest-complete goals from new facts.
5. **Failure Pattern Detection** (`consolidate_failure.go`): detect recurring mistakes across episodes.
6. **Facts+Relationships→Patterns**: LLM abstraction; pattern confidence = `min(source confidences) * coherence_score`.
7. **Hypothesis Evidence Scanning**: auto-confirm/reject pending hypotheses against accumulated facts.
8. **Confidence Decay** (`decay.go`, pure SQL).
(Contradiction detection is inline in stage 1, not a separate stage — the "8" counts it under Episodes→Facts.)

### Lifecycle Management
- **Decay** (`decay.go`): pure-SQL. Facts not `updated_at`-touched within `Window` (default 168h) get `confidence *= DecayFactor` (0.95); facts below `ExpiryThreshold` (0.1) get `valid_until = now()` (soft-expired). No reheat-on-access, no per-category half-lives.
- **Contradictions**: `DetectContradictions` on each new fact; tracks `old_value`/`new_value`, supports auto-resolution.
- **Soft-delete** (`deleted_at`) on all tables; `forget` tool soft-deletes episodes by content match.
- No versioning/supersession chain (contradictions old/new tracking is the closest analog).

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| 8-stage consolidation pipeline | `consolidate.go` — all 8 stages present and wired | **Validated** (code-confirmed) |
| Causal-link + hypothesis engine | `causal.go`, `hypothesis.go`, migrations 00016/00017; confirm materializes a fact in a tx | **Validated** as mechanism; no evidence it improves any benchmark |
| Layered memory (episodes→facts→patterns) | 3 tables + 3 consolidation stages | **Validated** |
| Dedup / clustering / quality refine | `factExistsByVector`, `clusterEpisodes`, `ReasonStructured` | **Validated** |
| Entity extraction (EAV) | `entity`/`property`/`value` on facts (free-text, not a resolved entity table) | **Partial** — extraction yes, entity *resolution* no |
| "fulltext: true" (some external tables) | No FTS/tsvector/trigram anywhere; recall is `<=>` only | **False** — vector-only (evidence file correctly flags this) |
| Any retrieval-quality / QA benchmark | None in repo | **Unvalidated** — zero benchmarks; no LoCoMo/PERMA/R@k numbers exist |

No benchmark numbers exist at all, so nothing here is comparable to Somnigraph's 85.1% LoCoMo QA.

---

## Relevance to Somnigraph

### What stash does that Somnigraph doesn't
- **Hypothesis lifecycle as a first-class memory type** (`hypothesis.go`): a provisional belief with a `verification_plan`, a `proposed→testing→confirmed/rejected` state machine, and promotion — `ConfirmHypothesis` materializes the hypothesis into a durable `fact` inside a transaction. Somnigraph's REM (`sleep_rem.py`) generates *questions* and the CLAUDE.md workflow tracks knowledge-gap questions, but they are inert notes with no verification/promotion machinery.
- **Failure-pattern recurrence detection** (`consolidate_failure.go`): auto-detects repeated mistakes across episodes and records a lesson. Somnigraph relies on *manual* procedural "gotcha" capture; nothing auto-detects recurrence during sleep.
- **Causal links as a distinct directional edge type with chain tracing** (`trace_causal_chain`). Somnigraph's edges are typed (supports/contradicts/evolves/revision/derivation) but has no explicit cause→effect edge nor forward/backward chain traversal.
- **Goal hierarchy** with status lifecycle — no analog in Somnigraph.
- **Hierarchical namespaces** — multi-scope isolation Somnigraph (single-user, flat) lacks.

### What Somnigraph does better
- **Retrieval, decisively.** Somnigraph has hybrid BM25+vector RRF fusion (`fts.py`, k=14), a 26-feature LightGBM reranker (`reranker.py`, NDCG 0.7958), and PPR graph-conditioned expansion (`scoring.py`). Stash has none of these — single-channel vector `ORDER BY <=>`. Stash's rich graph (relationships, causal_links, patterns) never touches the recall path.
- **Feedback loop**: Somnigraph's explicit per-query utility ratings, EWMA/UCB, and Hebbian PMI edge strengthening (measured Spearman r=0.70) have no counterpart; stash has zero retrieval feedback.
- **Decay sophistication**: Somnigraph's per-category half-lives, floors, and reheat-on-access vs stash's single global `DecayFactor` with no reheat.
- **Evidence discipline**: Somnigraph is benchmark-grounded (LoCoMo, ablations); stash ships no numbers.

---

## Worth Stealing (ranked)

### 1. Hypothesis lifecycle for REM-generated questions (Medium)
**What**: Give Somnigraph's REM-generated knowledge-gap questions a real lifecycle — `proposed → testing → confirmed/rejected` — with an optional verification plan, and auto-promote a confirmed hypothesis into a durable memory. Stash's `ConfirmHypothesis` does exactly this: confirming writes a `fact` transactionally.
**Why**: Today `sleep_rem.py` questions and CLAUDE.md's `category="meta"` questions are inert — they accumulate but nothing closes the loop. A confirm/reject state machine turns "open question" into a tracked, resolvable object, and confirmation becomes the natural moment to `remember()` the answer (which the CLAUDE.md workflow already describes manually as `forget(question) → remember(answer)`).
**How**: Add a `status` + `verification_plan` field to meta/question memories; in `sleep_rem.py`, scan whether accumulated NREM facts/edges provide evidence for a pending question and auto-transition. Promotion = create a semantic memory linked (`derivation` edge) to the question. Low risk, reversible, no retrieval-path change.

### 2. Failure-pattern recurrence detection during sleep (Medium)
**What**: A sleep sub-pass that clusters failure/gotcha episodes and, when the same failure recurs ≥N times, synthesizes a single procedural "lesson" memory with elevated priority.
**Why**: Somnigraph captures gotchas only when a human/agent notices and manually stores one. Auto-detecting recurrence would surface systemic mistakes the operator missed, and recurrence count is a natural priority/durability signal.
**How**: In `sleep_nrem.py`, add a pass over episodic memories tagged as failures (or classified as such by the pairwise LLM step); cosine-cluster, count recurrences, emit a consolidated procedural memory. Reuses the existing NREM clustering + merge machinery.

### 3. Causal edge type with chain tracing (Low)
**What**: Add a `causes` typed edge (directional) alongside the existing supports/contradicts/evolves set, plus a forward/backward chain-walk helper.
**Why**: Multi-hop "why did X happen" queries are exactly where Somnigraph's multi-hop vocabulary-gap ceiling bites; an explicit causal chain gives a structured traversal path independent of lexical overlap.
**How**: `db.py` edge type enum + `scoring.py` traversal. Marginal — only worth it if NREM can reliably classify causal pairs; otherwise it dilutes the edge taxonomy.

---

## Not Useful For Us

### Postgres/pgvector backend & hierarchical namespaces
Stash targets multi-scope, server-deployed use; Somnigraph is deliberately single-user SQLite. The namespace hierarchy solves a problem Somnigraph doesn't have.

### EAV (entity/property/value) fact structure
Free-text triples with no entity resolution add schema surface without the graph payoff; Somnigraph's themes[] + typed edges already cover the "structured belief" need, and true value would require entity resolution stash doesn't do.

### Goal-tracking subsystem
Out of scope — Somnigraph is a memory store, not a task/goal planner.

---

## Connections

- **Write-path-strong / retrieval-weak** is the recurring shape from the Phase 18 sweep: like ByteRover (BM25-only) and agentmemory, stash invests in write-time/consolidation-time structuring and leaves retrieval simple. Corroborates the AMemGym finding that write-path quality, not retrieval cleverness, is where several LME/LoCoMo leaders win — except stash never benchmarks, so it's the *hypothesis* without the evidence.
- **LLM-mediated offline consolidation** converges with Somnigraph's sleep and with memos/MIRIX-style consolidation — but stash's is the most *taxonomically diverse* (hypotheses, causal, goals, failures) of the systems surveyed, even if each stage is a single LLM call rather than the pairwise-classification rigor of `sleep_nrem.py`.
- **Hypothesis promotion** rhymes with the supersession/versioning patterns seen elsewhere: a provisional record graduating to a durable one on evidence.

---

## Summary Assessment

Stash's real contribution is a **broad, checkpointed, LLM-mediated consolidation taxonomy**: episodes cluster into confidence-scored facts, which spawn relationships, causal links, and patterns, while a parallel machinery tracks hypotheses, goals, and recurring failures — all resumable and namespace-scoped. As a catalog of "what structured objects an agent memory could maintain," it is the richest in the survey.

But the retrieval path is a single pgvector `ORDER BY <=>` query. All that structure is built and then **never used at recall time** — the graph is reachable only through explicit inspection tools, not fused into search. There is no reranker, no hybrid, no feedback, no benchmark. So relative to Somnigraph it is strictly weaker where Somnigraph is strong (retrieval, feedback, evidence) and broader where Somnigraph is thin (memory-type taxonomy, provisional/hypothesis state).

The single most valuable takeaway is the **hypothesis lifecycle**: Somnigraph already *generates* questions during REM but treats them as inert; stash shows the closing move — a verification state machine that promotes a confirmed hypothesis into a durable memory. That, plus auto-detecting recurring failures, are the two buried ideas worth a revisit. Everything else (Postgres, namespaces, EAV, goals) is scope Somnigraph deliberately doesn't want. Verdict: MAYBE — no adopt-now core idea, but the hypothesis-promotion loop is a genuine revisit-if angle for `sleep_rem.py`. The evidence file's audit is accurate and its `fulltext: true → false` correction is confirmed by the code.
