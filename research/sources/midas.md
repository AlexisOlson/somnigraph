# Midas — Local-first, eval-first, no-LLM-at-ingest agentic memory with a provenance trust guard

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

Midas (`vornicx/Midas`, PyPI `midas-memory`, MIT, created 2026-06-04, ~6.4k Python LOC in the core package) is a single-user-ish local memory SDK + MCP server. Its thesis is deliberately contrarian: **no LLM runs at ingest or at query**. Everything — importance, supersession, abstention, context assembly — is rule-based or local-model (ONNX embedder / cross-encoder / NLI). An optional LLM "distiller" tier exists but is off by default. The pitch is "$0 API spend, zero data egress, reproducible eval."

### Storage & Schema

- Backends behind a store protocol: `InMemoryStore` (numpy cosine matrix, `store.py`) and `SqliteStore` (`sqlite_store.py`), plus experimental ANN/TurboVec vector indexes (`ann.py`, `turbovec_*.py`). The default `Memory()` is in-memory; MCP server uses SQLite.
- `MemoryRecord` (`types.py:32-45`): `content`, `kind`, `importance` (1-5 int), `source`, `provenance`, `actor`, `metadata` (dict), `superseded_by`, plus `id`/`created_at`/`updated_at`/`embedding`. **8 authored fields.** Much leaner than Somnigraph's schema (no themes[], no priority 1-10, no valid_from/valid_until columns — validity is derived from the supersession link + `metadata["superseded_at"]`).
- `created_at` is **event time** (when the fact was true/stated), not ingest order — a bitemporal signal used by recency + chronological assembly + `as_of` historical queries.
- SQLite store detects other-process writes via `PRAGMA data_version` and refreshes (`_refresh_if_stale`), enabling live multi-client sharing of one DB file. A TypeScript port (`packages/midas-ts`) shares the same schema and a bit-comparable hashing embedder.

### Memory Types

`MEMORY_KINDS` (`types.py:13-21`): note / chat / fact / preference / constraint / mission. "Durable kinds" = fact/preference/constraint (+ mission for state views) — these are the semantic/identity tier that forgetting never evicts. `provenance` is a separate 4-level axis: planning < observation < action < user_confirmation (a trust rank, `_PROVENANCE_RANK`).

### Write Path

Three tiers, only the third touches an LLM:
- `remember()` — unconditional store. Derives importance if not given (`ContentImportance`, optionally blended with novelty-vs-store), stamps event time, optionally reinforces a near-duplicate, optionally runs supersession.
- `capture()` (`memory.py:395-455`) — **policy-gated auto-remember, no LLM.** Scores content salience, then enforces `policy.accept_kinds` + `policy.min_importance` floor + `policy.dedup_threshold`. Returns a `CaptureResult(stored, reason, ...)` so the agent *learns the relevance bar* instead of guessing. Near-duplicates upgrade the existing record's provenance (never downgrade) and/or reinforce it rather than duplicating.
- `distill()` (`memory.py:457-509`) — optional LLM distiller; `keep_raw=True` is the measured-safe default (distill-to-replace tanked judged answer 0.30→0.08; augmenting recovered to ~neutral). Distilled records stamped `metadata["distilled"]=True`.

Importance scoring (`importance.py`): `ContentImportance` = content-word density + digit/proper-noun specifics + an anti-backchannel floor, bucketed to 1-5. `StructuralImportance` adds first-person / durable-attribute / copula boosts and question/meta penalties — an assertion of a personal fact outranks a question mentioning the same words. All regex, all auditable.

### Retrieval

- **Semantic**: vector prefilter to a pool (`pool = max(limit*5, limit)`), cosine scoring.
- **Hybrid** (`_hybrid_candidates`, `memory.py:766-831`): union semantic top-pool with BM25 top-pool (pure-Python Okapi BM25, `bm25.py`), fuse with **RRF, k=60** (default; legacy `max` fusion kept). BM25 index cached on the store's version counter (rebuild-per-query was 35× slower). Lexical backend is pluggable (`lexical_index_factory`) — SPLADE/BM42 drop in without touching fusion.
- **Cross-encoder rerank** (optional, `LocalReranker`, ONNX): reorders the pool; relevance = `max(sigmoid(ce), cosine)`.
- Final score = `w_relevance·relevance + w_importance·importance_norm + w_recency·recency` (defaults 1.0 / 0.3 / 0.2), recency = exp decay with 30-day half-life.
- **Parsimony floors**: a scale-free `min_relevance_ratio` (default 0.3 — drop hits below 0.3× the top hit) plus an optional absolute `min_relevance`. Philosophy: drop weak hits rather than pad the budget (padding *lowers* answer quality via distraction).
- Diversification re-selection (no scoring change): `mmr_lambda` (MMR), `thread_cap` (cap hits per session/thread then backfill), `anchor_boost` (pseudo-relevance feedback — treat top-3 hits as anchors, let their neighbors earn relevance; helps evolving-fact queries where phrasing matches only one instance).
- `build_context()` packs highest-value first into a token budget, with a **pinned channel** for standing directives / mission records (Letta-style always-in-context core memory, detected by regex `is_standing_instruction`, no LLM), and an **abstention** layer.

### Consolidation / Processing

No sleep/offline graph build. `consolidate()` (`memory.py:904-950`) is an on-demand extractive dedup: collapse cosine ≥ threshold near-duplicates, keep the highest-value copy. `_reinforce_*` implements repetition⇒salience at write time. There is no graph, no edge typing, no PPR, no scheduled/autonomous processing.

### Lifecycle Management

- **Supersession** (`_maybe_supersede`, `memory.py:906-996`): LLM-free belief revision. High cosine + same kind, with a change-cue regex (`_UPDATE_RE`) that *lowers* the similarity bar for paraphrased updates. Guards: proper-entity negative guard (Apollo vs Artemis → dup, not update), content-word overlap anchor for the medium band, an ambiguity margin (skip if top-two heads within `supersede_margin`), provenance integrity (a user-confirmed belief is only revised by another user confirmation), and an optional **local NLI contradiction gate** (`nli.py`, int8 ONNX MNLI) — only revise a belief the new record actually *contradicts*. Stores `metadata["superseded_at"]` = revising record's event time → bitemporal validity window. **Supersession is `supersede=False` by default.**
- **`as_of` historical queries**: `recall(as_of=T)` walks the chain to the version valid at T (`_resolve_at`) and excludes later-born records — a Zep-style time-travel query with no graph DB.
- **Decay / forgetting** (`forget_decayed`): evict lowest `memory_value` = importance × recency, protecting durable kinds; auto-runs over `MIDAS_MCP_MAX_RECORDS`. Exposed as the `maintain` MCP tool.
- **Explicit forget**: single (chain-safe relinking), `forget_matching` (relevance-matched topic erasure with dry-run preview + returned audit trail), `forget_all`.

### Control-plane views (`state.py`)

The idea most relevant to us. Two **deterministic, no-LLM, no-similarity** views:
- `memory_state(mem, scope)` — the live non-superseded durable records of a scope, newest-first. Answers "what do we currently believe about Apollo?" / project onboarding.
- `memory_diff(mem, since)` — beliefs `added` and `revised` (old→new pairs) since a timestamp. The "what's new since our last session" primitive.

Both read the store + supersession links directly. Explicit rationale: broad "current state / what changed" queries **don't resemble any single stored turn**, so top-k similarity under-retrieves them.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| No LLM at ingest or query | Code confirms: importance/supersession/abstention all rule-based or local-ONNX; distiller off by default (`memory.py:481`) | **Validated** |
| LoCoMo recall@k 0.73 (full public set, n=1,540) | `BENCHMARKS.md §1` with reproduce cmd; explicit correction that earlier 0.85 (n=50) did **not** reproduce | **Validated but retrieval-only** — NOT end-to-end QA; not comparable to our 85.1 LoCoMo QA |
| LongMemEval-s recall@k 0.92 (full 500 Q, 246k turns) | `BENCHMARKS.md §1`, deterministic, reproduce cmd | Plausible; retrieval recall, not answer accuracy |
| LongMemEval answer 0.84 with gpt-4o (n=40) | `BENCHMARKS.md §6`; "ties LLM-ingest SOTA at $0 ingest" | Plausible but n=40 sample, single reader |
| 30-40% fewer context tokens via parsimony floor | Deterministic A/B tables, recall@k unchanged | **Validated** (deterministic, reproducible) |
| Dumb-reader ablation proves numbers aren't reader-inflated | `--dumb-reader` extractive reader; `answer_dumb` tracks recall@k | **Validated** — genuine eval-honesty mechanism |
| Time-aware retrieval lifts temporal recall 0.86→0.95 | A/B via `--midas-no-time`, n=40 | Plausible; small n |
| Provenance trust guard gates external actions | `guard.py` fully implemented, tested; MCP `check_memory_use` | **Validated** (real code, conservative rules) |

---

## Relevance to Somnigraph

### What Midas does that Somnigraph doesn't

- **Deterministic state/diff control-plane views** (`state.py`). Somnigraph has no equivalent — `recall()` is always top-k similarity. Broad "current state of project X" / "what changed since last session" queries have no stored turn to match and under-retrieve. We have the exact substrate to build these: the event log (`events.py`), supersession/`evolves` edges, and `valid_until` in `db.py`.
- **Provenance/trust guard** (`guard.py`): a security boundary that only lets *user-confirmed* memories authorize external/destructive actions, with a forbidden-action veto and a currency check that re-verifies supersession itself rather than trusting recall. Somnigraph has no provenance axis and no action-authorization concept.
- **Eval-honesty floor** (dumb-reader ablation): a deterministic extractive reader that can't reason, isolating retrieval quality from reader "heroic recovery." This directly addresses STEWARDSHIP Priority 4's known LoCoMo judge-leniency problem (their LLM judge accepts 62.81% of vague wrong answers).
- **No-LLM write-path quality gating** (`capture` + `ContentImportance`/`StructuralImportance` + policy floor): a salience heuristic that keeps fact-bearing turns and drops backchannel, returning *why*. Somnigraph's write path has no quality/salience gate at all (a named gap in STEWARDSHIP).
- **Provenance-upgrade-on-duplicate** and **reinforcement (repetition⇒salience)**: a re-stated memory gains importance + recency instead of duplicating. Somnigraph dedups but has no reinforcement signal.
- **Cross-runtime shared store** (Python↔TypeScript over one SQLite file) and live multi-client refresh. Not a Somnigraph goal (single-user MCP), but architecturally notable.

### What Somnigraph does better

- **Learned reranker** (`reranker.py`, 26-feature LightGBM, NDCG=0.7958). Midas's ranking is a hand-weighted 3-term formula + optional generic cross-encoder — exactly the hand-tuned formula stage Somnigraph already beat by +6.17pp.
- **Explicit feedback loop with measured GT correlation** (Spearman r=0.70), EWMA + UCB. Midas has *no* retrieval feedback loop; reinforcement is write-time textual restatement only.
- **Graph-conditioned retrieval**: typed edges + PPR expansion + betweenness feature (`scoring.py`, `sleep_nrem.py`). Midas has no graph at all (`anchor_boost` pseudo-relevance feedback is the closest analogue, and it's much weaker).
- **Offline LLM-mediated consolidation** (sleep): gap analysis, question generation, taxonomy. Midas's consolidation is on-demand extractive dedup only.
- **End-to-end QA benchmarking** (85.1% LoCoMo QA, Opus judge). Midas deliberately reports retrieval recall@k and treats answer correctness as secondary — its headline 0.73/0.92 numbers are **not comparable** to our QA accuracy.

---

## Worth Stealing (ranked)

### 1. Deterministic `memory_state` / `memory_diff` control-plane views (Medium)
**What**: Two non-similarity views over the store — current non-superseded durable memories for a scope (`memory_state`), and added/revised-since-timestamp (`memory_diff`). Both read the supersession links directly, no LLM, no embedding query.
**Why**: Broad "what do we currently believe / what changed since last session" queries have no single stored turn to match, so top-k similarity structurally under-retrieves them. Somnigraph's `recall()` is the only retrieval surface and shares this blind spot. This is the strongest idea in the repo for us.
**How**: New tool impls in `tools.py`. `memory_state`: filter active (non-superseded, valid_until null/future) memories by category ∈ {semantic, procedural, reflection} + theme/scope, order by recency — reuses existing `db.py` state. `memory_diff`: query the event log (`events.py`) for creates/supersessions since a timestamp; pair `evolves`/`revision` edges into old→new. A natural fit for a session-start "what's new" injection, adjacent to the proactive-injection design already on main.

### 2. Dumb-reader ablation as an eval-honesty floor (Low-Medium)
**What**: A deterministic extractive "reader" (pick the single retrieved turn with max content-word overlap with the question; score verbatim-hit=1.0 else token-F1) run alongside the LLM judge. `answer_dumb` can only move with retrieval quality.
**Why**: Directly attacks STEWARDSHIP Priority 4's documented judge-leniency (62.81% vague-wrong acceptance). A dumb-reader column separates "our retrieval is good" from "Opus recovered a weak retrieval," making the LoCoMo numbers defensible without a stricter (costlier) judge.
**How**: Add a no-LLM reader mode to the LoCoMo QA harness (`scripts/`): for each question, take the top retrieved memory by content-word overlap, compute token-F1 vs gold. Report `answer_dumb` beside the judged score. Cheap, deterministic, reproducible.

### 3. Currency surfaced *at recall time*, not silently filtered (Low)
**What**: Midas returns supersession/provenance state on hits (guard `blocked_ids`, `superseded_at`, `as_of` resolution) rather than only dropping stale records. The reader can see "this belief was current until T."
**Why**: Somnigraph's sleep marks `contradicts`/`evolves` edges and `valid_until`, but recall silently filters — losing the "X was true until Y, now it's Z" signal that temporal-reasoning questions need. Annotating instead of hiding is nearly free given our existing edges.
**How**: In `tools.py` recall assembly, when a returned memory has an outgoing `evolves`/`revision` edge or a set `valid_until`, append a compact `[superseded <date>]` / `[was current until <date>]` marker instead of dropping it. Pairs with the `as_of` idea below.

### 4. Small supersession-precision guards (Low)
**What**: Three cheap gates on belief revision: a proper-noun negative guard (different named entities → duplicate, not update), an ambiguity margin (don't guess which head to revise when top-two are ~equal), and an NLI contradiction gate (only revise what the new record actually contradicts).
**Why**: Somnigraph's contradiction/evolves classification happens in `sleep_nrem.py` via an LLM. These are cheap deterministic pre-filters that could reduce false-positive edge creation before the LLM pass (or validate it), and the entity/ambiguity guards are useful regardless of LLM.
**How**: In `sleep_nrem.py` pairwise classification, add an entity-overlap negative check and a margin check on the candidate pair before/after the LLM call; they're a few lines each and mirror `_proper_entities` / `supersede_margin`.

---

## Not Useful For Us

### No-LLM-at-ingest as a hard constraint
Midas's core identity ($0 API, zero egress) forces every mechanism to be rule-based. Somnigraph already commits to sleep-time LLM consolidation and an LLM reader; the no-LLM constraint would forbid our strongest components. The *techniques* (rule-based importance, regex change-cues) are worth borrowing selectively, but the constraint itself is not our axis.

### Provenance trust guard / action authorization
`guard.py` is a genuine, well-built security boundary — but it targets multi-agent tool-using systems where memory can authorize external/destructive actions. Somnigraph is a single-user recall layer for Claude Code; there is no action-authorization surface to gate.

### Cross-runtime / multi-client shared store
Python↔TS shared SQLite with live refresh is real engineering, but Somnigraph is deliberately single-user MCP. No use.

### Retrieval recall@k as the headline metric
Reasonable for Midas's "measure the memory layer, not the reader" thesis, but Somnigraph's value proposition is end-to-end QA; adopting recall-only headline numbers would hide the reranker/feedback contribution that shows up downstream.

---

## Connections

- **Supersession / belief revision**: convergent with the supersession pattern seen in memv and Zep-style temporal-validity systems — Midas independently arrives at bitemporal validity windows (`superseded_at` + `as_of`) *without* a graph DB, using only the supersession link. Strong corroboration that "annotate validity, resolve at query time" is the right shape.
- **Write-path quality gating**: reinforces the Phase 18 source-sweep headline (ByteRover / MemPalace / agentmemory) that **write-path quality, not retrieval cleverness, is what the LoCoMo/LME leaders win on**. Midas is another independent vote: its `capture` policy floor + `ContentImportance` are write-time salience gates, and its own ablations say parsimony (dropping weak context) beats padding.
- **Standing-directive pinning**: same idea as Letta core-memory (cited in-code), reached with a regex detector instead of an LLM — cf. any Letta analysis in the corpus.
- **Eval honesty**: aligns with STEWARDSHIP Priority 1 (honest accounting) and the LoCoMo judge-leniency caveat in `docs/locomo-benchmark.md` — the dumb-reader ablation is the mechanism that caveat is asking for.

---

## Summary Assessment

Midas's core contribution is a **disciplined, eval-first, no-LLM-at-ingest memory layer** whose real value to us is not its architecture (a leaner subset of ours — no graph, no learned reranker, no feedback loop) but three well-executed *mechanisms*: (1) deterministic `memory_state`/`memory_diff` control-plane views that answer the broad "current state / what changed" questions top-k similarity structurally misses; (2) a dumb-reader ablation that isolates retrieval quality from reader recovery, which is exactly the eval-honesty floor STEWARDSHIP Priority 4 needs; and (3) currency surfaced at recall time (bitemporal `as_of`, `superseded_at`) rather than silently filtered. The supersession-precision guards (entity negative-guard, ambiguity margin, NLI contradiction gate) are cheap, transferable pre-filters.

The single most important thing to take is the **state/diff control-plane view** — it's a genuine retrieval blind spot in Somnigraph, we already have the event-log + supersession substrate to build it, and it dovetails with the proactive-injection work on main.

What's overhyped, and the sharpest cross-check: the benchmark numbers are **retrieval recall@k, not end-to-end QA**, and the repo's *own* correction notice (2026-06-10) walks the headline LoCoMo figure from 0.85 (n=50) down to **0.73 (full public set)** after finding a benchmark-tuned `min_relevance=0.75` floor was pruning gold — an unusually honest disclosure, but it means the carsteneu-adjacent "0.85 LoCoMo" figure is stale and, either way, **not comparable to Somnigraph's 85.1% LoCoMo QA accuracy** (different metric entirely: fraction of gold *turns retrieved* vs fraction of *questions answered correctly*). To Midas's credit, its methodology is the most transparent in the sweep — every number ships with a reproduce command and a dated correction trail. That eval discipline, more than any single feature, is what's worth importing.
