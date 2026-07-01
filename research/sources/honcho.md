# Honcho - Peer-centric "reasoning-first" memory infra: LLM-derived conclusions per observer/observed pair, write-time dedup, surprisal-targeted dream consolidation

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

Honcho (plastic-labs/honcho, `Server 3.0.9`) is a FastAPI + Postgres/pgvector service, offered managed at `api.honcho.dev` or self-hosted. It is **memory infrastructure for multi-agent products**, not a single-user CLI memory like Somnigraph. The organizing abstraction is the **peer**: users, agents, groups, projects, and ideas are all "peers," and memory is stored per **(observer, observed)** collection — i.e. what peer A has concluded about peer B. This theory-of-mind framing is the whole point of the system and most of it is out of Somnigraph's scope.

### Storage & Schema
- **Postgres + pgvector** (HNSW index, `m=16, ef_construction=64`, cosine ops). No SQLite/FTS5 stack; FTS is Postgres `to_tsvector`/`plainto_tsquery` with ILIKE fallback.
- Two main memory units:
  - **Message** (`models.Message`): raw conversation turns, embedded per-chunk.
  - **Document** (`models.Document`, `src/models.py:379`): a derived *observation/conclusion*. Fields: `content`, `level` (`explicit` | `deductive`, extensible), `times_derived` (reinforcement counter, default 1), `embedding`, `source_ids`, `observer`, `observed`, `session_name`, `deleted_at` (soft delete), `sync_state`. No priority, no themes, no decay_rate, no valid_from/valid_until.
- Collections keyed by `(observer, observed, workspace)`.

### Memory Types
Level-based, not category-based: `explicit` (facts stated in dialogue) vs `deductive`/`inductive` (conclusions reasoned in the background). Contrast Somnigraph's episodic/semantic/procedural/reflection/meta taxonomy. Honcho's "types" are reasoning provenance, not content class.

### Write Path (this is where Honcho invests)
1. **Deriver** (`src/deriver/deriver.py`): batches of messages per (observed peer) go through a single LLM call (`minimal_deriver_prompt`, JSON mode, structured `PromptRepresentation`) that extracts explicit + deductive observations. Reasoning-first: it stores *conclusions*, not chunks.
2. **Write-time dedup + reinforcement** (`src/crud/document.py:970 is_rejected_duplicate`, on by default `DERIVER.DEDUPLICATE=true`): for each new observation, cosine kNN (top-1, `max_distance=0.05`, i.e. sim ≥ 0.95). If a near-duplicate exists, a **token-set superiority score** decides who wins: `score = len(tokens) + 10 * len(unique_tokens_vs_other)`. Superior new doc soft-deletes the old and **carries `times_derived` forward** (`max(new, existing+1)`); inferior new doc is dropped and the existing doc's `times_derived` is atomically incremented. So repeated derivation of the same fact acts as a salience/reinforcement count.
3. Saved to every observer collection that observes the peer (multi-perspective fan-out).

### Retrieval
- `src/utils/search.py`: hybrid **pgvector semantic + Postgres FTS**, fused with **Reciprocal Rank Fusion, k=60 (library default), no tuning, no learned reranker**. Oversamples 2-4x then dedups. This is a plain, un-tuned version of Somnigraph's retrieval front half.
- Higher-level query modes layered on top: `search()` (raw hybrid), `chat()`/**dialectic** (`src/dialectic/core.py`, LLM reasons over stored representations to answer NL questions — the headline product feature), `context()` (token-budgeted summarization), `representation()` (static snapshot of what observer knows about observed). `get_documents_by_reinforcement` orders by `times_derived DESC` — the only place reinforcement feeds retrieval.

### Consolidation / Processing — the "dreamer"
`src/dreamer/` runs offline "dreams" (default `dream.ENABLED=true`, `ENABLED_TYPES=["omni"]`). A dream runs two self-directed LLM **specialists** — **deduction** then **induction** — that explore the observation space and write higher-level observations (`src/dreamer/orchestrator.py`, `specialists.py`).
- **Surprisal sampling** (`src/dreamer/surprisal.py`, **default `dream.surprisal.ENABLED=false`**): before the specialists run, build a spatial tree over observation embeddings and score each observation's **geometric surprisal**, then hand the **top 10%** most novel/outlier observations to the specialists as exploration *hints* (not hard filters). Tree options: `kdtree` (default), `balltree`, `rptree`, `covertree`, `lsh`, `graph`, `prototype` (`src/dreamer/trees/`). The RP-tree surprisal is `S(x) = Σ -log(n_child / n_parent)` along the root-to-leaf path (`rptree.py:125`) — a cheap, embedding-only, LLM-free novelty proxy.

### Lifecycle Management
- Soft delete (`deleted_at`) + async reconciliation cleanup of vectors (`cleanup_soft_deleted_documents`, `src/reconciler/`).
- `times_derived` reinforcement counter.
- **No decay/half-life, no versioning, no user-facing supersede or time-travel, no explicit "forget" for conclusions** (evidence file confirms; only messages/sessions have CRUD delete).

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| "Defined the Pareto Frontier of Agent Memory" (README) | Links out to a blog/evals page/video; **no numbers in the repo** | Unvalidated from code. Marketing line; methodology not in-repo. |
| Reasoning-first memory beats chunk-matching | Deriver extracts conclusions; dialectic answers over them | Plausible and architecturally real, but unmeasured here. Convergent with the Phase-18 "write-path quality is what wins" finding. |
| Surprisal targeting improves consolidation | Full tree machinery + orchestrator wiring present | Mechanism is real and clever, but **default-disabled** and no ablation shipped — unvalidated. |
| Write-time dedup preserves the more-informative fact | `is_rejected_duplicate` token-superiority logic, on by default | Validated as *implemented*; heuristic (token count) is crude but cheap. |
| LoCoMo/LongMemEval strong results | External only | Prior Somnigraph sessions found the headline LoCoMo (~89.9) uses a **lenient gpt-4o-mini judge — not comparable to our Opus-judged 85.1**. Treat any cross-quote as apples-to-oranges. |

---

## Relevance to Somnigraph

### What Honcho does that Somnigraph doesn't
- **Write-time quality gating + reinforcement counting** (`is_rejected_duplicate`): Somnigraph's `tools.py:remember()` has *no* dedup/salience gate — every write lands. Honcho's cosine-95 + token-superiority + `times_derived` carry-forward is exactly the write-path discipline the Phase-15/18 findings kept flagging as the actual differentiator of LoCoMo leaders.
- **Surprisal-targeted consolidation**: Somnigraph's `sleep_nrem.py` selects consolidation targets by recency/priority; Honcho picks **novel geometric outliers** to reason about. This is a principled "what deserves attention tonight" selector that Somnigraph lacks.
- **Peer/theory-of-mind model**: observer/observed representations, multi-agent perspective — genuinely absent in single-user Somnigraph (and mostly not wanted).
- **Reasoning-first extraction** at write time (deductive/inductive conclusions), vs Somnigraph storing user-authored memories verbatim.

### What Somnigraph does better
- **Learned reranker** (`reranker.py`, 26-feature LightGBM, NDCG=0.7958): Honcho stops at un-tuned RRF k=60. No reranking, no learned scoring.
- **Explicit feedback loop with measured GT correlation** (Spearman r=0.70): Honcho has no per-query utility feedback at all; `times_derived` is the closest analog and it's write-side, not retrieval-outcome-side.
- **Typed graph + PPR expansion** (`scoring.py`): Honcho has no memory-to-memory edge graph (a `graph` tree exists only as a surprisal index, not a semantic edge store).
- **Decay/lifecycle** (per-category half-lives, reheat): Honcho has none.
- **Benchmark rigor**: Somnigraph reports Opus-judged LoCoMo; Honcho punts numbers to a blog and (per prior sessions) uses a lenient judge.

---

## Worth Stealing (ranked)

### 1. Surprisal-based consolidation targeting (Medium)
**What**: Score each memory's *geometric surprisal* against a spatial tree over embeddings (RP-tree: `S = Σ -log(n_child/n_parent)` down the path; or simply kNN-distance to nearest neighbors), and prioritize the top-N% novel outliers as NREM/audit targets. LLM-free, cheap.
**Why**: `sleep_nrem.py` currently spends its LLM budget on recency/priority-ranked pairs; novelty-targeting would spend it where the memory store is genuinely surprised — likely higher-yield edges and contradictions. Complements (doesn't replace) the existing pairwise classifier.
**How**: Add a pre-pass in `scripts/sleep_nrem.py` (or `scoring.py`): pull active-memory embeddings from sqlite-vec, build kNN-distance or an RP-tree over them (numpy, ~50 LOC per `rptree.py`), rank by mean-kNN-distance or path-surprisal, feed the top decile as candidate anchors to pairwise classification. Offline-testable against existing edges: do surprisal-picked anchors yield more accepted edges per LLM call than recency-picked?

### 2. Write-time dedup with token-superiority + reinforcement carry-forward (Medium)
**What**: On `remember()`, cosine kNN top-1; if sim ≥ ~0.95, decide keep-new-vs-existing by an information score (`len(tokens) + 10*unique_tokens`), delete the loser, and **carry a `times_derived` counter forward** so re-derived facts accrue a reinforcement count.
**Why**: Somnigraph has no write-path quality gate — the single most-cited gap in our recent source sweeps. `times_derived` is also a free salience signal the reranker could consume (a memory independently written 5x is probably important), analogous to but cheaper than the Hebbian co-retrieval signal.
**How**: New guard in `tools.py:remember()` using existing `embeddings.py` + a sqlite-vec cosine query; add a `times_derived`/`reinforcement_count` column in `db.py`; expose it as a `reranker.py` feature. Note the token-count superiority heuristic is crude — Somnigraph could instead keep the higher-priority or more-recent one, or merge via the sleep merge path.

---

## Not Useful For Us

### Peer / observer-observed / multi-agent theory-of-mind model
The entire (observer, observed) collection structure, `peer_perspective` temporal filtering, and dialectic "what does A know about B" framing assume a multi-agent product. Somnigraph is single-user; this adds schema and reasoning cost for a capability we don't want.

### Managed-service infra (queue_manager, reconciler, telemetry, webhooks, dialectic chat)
Production plumbing for a hosted SaaS — irrelevant to a local MCP server.

### Postgres/pgvector retrieval stack
Somnigraph's SQLite + sqlite-vec + FTS5 is a deliberate single-user choice; Honcho's un-tuned RRF k=60 is strictly less capable than our tuned-fusion + learned reranker.

---

## Connections
- **Strongest tie: the Phase-18 write-path thesis.** Honcho independently invests in write-path quality (reasoning-first extraction + dedup + reinforcement) while leaving retrieval un-tuned (RRF k=60, no reranker) — the same shape as ByteRover (BM25-only), agentmemory (write-time grounding), MemPalace (verbatim). Another data point that LoCoMo/LME leaders win on *what gets written*, not on retrieval cleverness. See `ai-memory-comparison.md`, `agentmemory.md`, the Phase-18 retrospective.
- **Surprisal targeting** rhymes with Somnigraph's own `scoring.py` novelty-scored adjacency, but Honcho's is embedding-geometric and pre-graph (no edges needed), so it's usable *earlier* in the pipeline. Convergent motive, different substrate.
- **`times_derived` reinforcement** is a write-side cousin of Somnigraph's Hebbian co-retrieval PMI — both let repetition strengthen a signal, one at write time, one at retrieval time.

---

## Summary Assessment

Honcho's real contribution is a **coherent reasoning-first, peer-centric write path**: it treats memory as LLM-derived conclusions about entities, dedups them at write time with a superiority test, counts reinforcement, and (optionally) uses geometric surprisal to decide what to reason about during offline "dreams." The retrieval side is deliberately plain (hybrid + RRF, no reranker, no feedback) — the opposite of where Somnigraph invests. That makes the two systems complementary rather than competing: Honcho is a case study in write-path quality, which is precisely Somnigraph's biggest gap.

The single most valuable takeaway is **surprisal-based consolidation targeting** — a cheap, LLM-free way to point the sleep pipeline at genuinely novel memories rather than merely recent ones. Second is the **write-time dedup + `times_derived` reinforcement** pattern, which would give `remember()` a quality gate and the reranker a free salience feature. Both are offline-testable against existing Somnigraph data before any commitment.

What's overhyped: the "Pareto Frontier of Agent Memory" README banner is unbacked in-repo (numbers punted to a blog), and prior sessions found the headline LoCoMo score rests on a lenient gpt-4o-mini judge — **not comparable to our Opus-judged 85.1**. Notably, the most novel mechanism (surprisal sampling) ships **default-disabled**, so it's an unvalidated research feature, not a proven production win. Verdict: DIVE, specifically on the surprisal-targeting idea for `sleep_nrem.py`.
