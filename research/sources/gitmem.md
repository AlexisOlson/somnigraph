# gitmem — Coding-agent "institutional memory" guardrail (scars/wins/patterns) with an enforcement protocol, not a ranking engine

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

gitmem (`gitmem-dev/gitmem`, TypeScript, MIT, v1.6.1, MCP server over stdio) is a memory layer for *coding agents* (Claude Code, Cursor, Windsurf, Copilot). Its unit of memory is a **Learning** — a `scar` (mistake), `win`, `pattern`, or `anti_pattern` — plus `Decision` and `Thread` records. The product framing is "check institutional memory before you act, so you don't repeat a past mistake." It is not a general personal-memory system; it is a mistake-avoidance guardrail with heavy prompt-engineering to make the agent actually obey surfaced lessons.

Two tiers with **mutually exclusive** retrieval backends:
- **Free**: everything in local `.gitmem/*.json` files; search is a hand-rolled BM25 (`src/services/bm25.ts`, ~100 lines).
- **Pro**: self-hosted Supabase (Postgres + pgvector), OpenRouter embeddings, plus an in-memory cosine cache (`src/services/local-vector-search.ts`).

### Storage & Schema
`schema/setup.sql` — `gitmem_learnings` (~27 columns), `gitmem_decisions`, `gitmem_threads`, `knowledge_triples`, `gitmem_scar_usage`, `gitmem_query_metrics`, `scar_enforcement_variants`. Notable learning fields: `learning_type`, `severity`, `counter_arguments TEXT[]`, `applies_when TEXT[]`, `keywords TEXT[]`, `domain TEXT[]`, `embedding vector(1536)` (Pro only), `decay_multiplier FLOAT`, `is_active BOOLEAN`, plus enforcement fields `required_verification`, `why_this_matters`, `action_protocol[]`, `self_check_criteria[]`. Free tier writes the same records to JSON without the embedding.

### Memory Types
Flat type tag (scar/win/pattern/anti_pattern), no working/short/long-term layering, no episodic/semantic split. Decisions and Threads are separate tables. `persona_name`/`rapport_summary` fields track which human/agent a lesson came from.

### Write Path
**No auto-extraction.** The agent must explicitly call `create_learning` (`src/tools/create-learning.ts`); `validateScar()` does light field validation only — no quality/salience gate, no dedup on learnings (thread creation has a cosine>0.85 semantic dedup, Pro only, `thread-dedup.ts`). Pro tier embeds `buildEmbeddingText()` client-side. A **rule-based** knowledge-triple writer (`src/services/triple-writer.ts`) fires-and-forgets triples with a controlled predicate vocabulary (`created_in`, `influenced_by`, `supersedes`, `demonstrates`, `affects_doc`, thread relations) and per-predicate `half_life_days`. The **Closing Ceremony** (`session_close`, 1511 lines) prompts the agent to reflect and mint learnings at session end — agent-mediated, not system-mined.

### Retrieval
`src/tools/recall.ts`. Free tier: `getStorage().search()` → BM25 over local JSON with field boosting + Porter-ish stemming, top-k, threshold 0.4. Pro tier: in-memory cosine similarity **multiplied by `decay_multiplier`**, threshold 0.45; falls back to a Supabase `gitmem_scar_search()` RPC (`weighted_similarity = raw * temporal_decay * behavioral_decay`) when the local cache isn't warm. **No hybrid fusion** (BM25 and vector never combine — they are separate tiers), **no learned reranker**, no RRF. Below `LOW_CONFIDENCE_THRESHOLD = 0.55`, matches are rendered as header-only "stubs" and tagged `[low confidence]` (a token-saving UX calibration, claimed ~66% N/A rate). recall output injects a **citation rule** ("cite the record ID or say 'not in institutional memory'") and surfaces `why_this_matters` / `action_protocol` / `self_check_criteria` / blocking `required_verification` SQL gates.

### Consolidation / Processing
No sleep, no LLM-mediated consolidation, no clustering/merge. The only offline job is `refresh_scar_behavioral_scores()` (Supabase RPC, fire-and-forget from `session_start`): aggregates the last 90 days of `scar_usage`, computes a dismiss rate, and writes `decay_multiplier`. `reflect_scars` is a per-session OBEYED/REFUTED protocol (min evidence lengths) that feeds `session_close`.

### Lifecycle Management
`archive_learning` sets `is_active=false` (soft-forget; no hard delete). Behavioral decay (dismiss-rate → `decay_multiplier` down-weights ranking) + temporal decay in the Pro RPC + per-predicate triple half-life. No reheat-on-access, no versioning/time-travel, no supersede for learnings (only a `supersedes` *triple* predicate on the graph, not a version chain on the record).

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| "Semantic search — recall returns the right scars" | pgvector + OpenRouter embeddings, Pro tier only | Plausible but **Pro/cloud-only**; the free/offline tier has no semantic search |
| Local-first / offline / no telemetry | Free tier writes `.gitmem/` JSON, BM25 only | Validated for free tier; Pro requires Supabase cloud |
| Behavioral decay (dismissed scars fade) | `refresh_scar_behavioral_scores` RPC, `decay_multiplier` multiplies similarity | Validated as coded; coarse (dismiss-rate, not utility) |
| Counter-arguments prevent rigid rules | `counter_arguments TEXT[]`, rendered in recall | Validated as a mechanism; efficacy unmeasured |
| Enforcement / "check before act" works | Advisory warnings (`enforcement.ts`), citation rule, blocking `required_verification` | Prompt-engineering; no benchmark of behavior change |
| A/B variant testing of enforcement text | `scar_enforcement_variants`, `variant_performance_metrics` | Real infra; needs multi-agent volume to yield signal |
| Retrieval/QA accuracy | **No benchmarks at all** (evidence file: all LoCoMo/LongMemEval/PersonaMem "—") | **Unvalidated — not comparable to Somnigraph's 85.1 LoCoMo QA on any measured axis** |

---

## Relevance to Somnigraph

### What gitmem does that Somnigraph doesn't
- **Intra-memory counter-arguments** — each scar carries `counter_arguments[]` ("reasons someone might reasonably ignore this"). Somnigraph models *inter*-memory contradiction as a `contradicts` edge (detected in `sleep_nrem.py`) but has no field on a single memory that carries its own caveat/anti-rigidity note.
- **Active enforcement protocol** — recall → confirm → act → reflect, with blocking `required_verification` gates, a citation rule injected into output, and `self_check_criteria`. Somnigraph's recall (`tools.py`) is passive: it returns memories and grades feedback, but never gates or nags the agent.
- **Measuring memory *efficacy*, not just retrieval** — `scar_enforcement_variants` A/B-tests which phrasing of a lesson gets obeyed (`variant_performance_metrics`). Somnigraph measures retrieval quality (NDCG, R@10, Spearman-to-GT in `reranker.py`) but never whether a surfaced memory changed behavior.
- **Implicit dismiss-rate feedback** — "surfaced but not confirmed/refuted" auto-decays ranking with no explicit user action.

### What Somnigraph does better
- **Ranking**: 26-feature LightGBM reranker (`reranker.py`) + RRF hybrid fusion (`scoring.py`) vs gitmem's tier-exclusive BM25 *or* raw-cosine×decay with a threshold filter — no fusion, no learned model.
- **Feedback loop**: explicit per-query utility+durability ratings, EWMA, UCB, measured r=0.70 to GT — strictly richer than gitmem's binary dismiss-rate.
- **Consolidation**: LLM-mediated three-phase sleep (`sleep_nrem.py`/`sleep_rem.py`) with typed-edge detection, merge/archive, gap analysis. gitmem has none — only a dismiss-rate SQL aggregate.
- **Graph retrieval**: PPR expansion + betweenness reranker feature vs gitmem's `graph_traverse` (a display/exploration tool over rule-based triples, not folded into scoring).
- **Benchmarks**: 85.1% LoCoMo QA vs gitmem's zero end-to-end evaluation.

---

## Worth Stealing (ranked)

### 1. Memory-carried counter-arguments / caveats (Low effort, marginal value)
**What**: A field on a memory that states when *not* to trust or apply it ("you might think X, but…"). gitmem attaches `counter_arguments[]` to every scar and renders them at recall time to stop memory from "becoming a pile of rigid rules."
**Why**: Somnigraph surfaces high-confidence memories that a fresh instance may over-apply. The `contradicts` edge helps only when a *second* memory disagrees; a self-carried caveat helps a lone memory. This is the one genuinely transferable, non-domain-specific idea here.
**How**: Optional `caveats`/`counter` text on the memory schema (`db.py`), populated at write or minted during REM (`sleep_rem.py`) when a `contradicts`/`evolves` edge exists, and appended to recall output in `tools.py`. Additive, low risk. Marginal because Somnigraph's typed-edge graph already covers most of the anti-rigidity need.

*(Considered and set aside: A/B variant testing of surfaced-memory phrasing and behavior-change measurement is a genuinely different metric axis, but it needs multi-agent volume that single-user Somnigraph can't generate — note-only. Blocking `required_verification` SQL gates and the recall→confirm→act→reflect enforcement protocol are specific to a coding-agent guardrail product and don't fit a general personal-memory system — reject.)*

---

## Not Useful For Us

### Enforcement protocol + blocking verification gates
Domain-specific to dev-agent guardrails (make the agent run a SQL check before deploying). Somnigraph is passive personal memory; gating/nagging is out of scope.

### Behavioral dismiss-rate decay
Somnigraph's explicit utility+durability feedback with EWMA/UCB (r=0.70 to GT) strictly dominates a binary surfaced/dismissed counter. The implicit "unused = down-weight" is already covered by scoring unused recalls 0.0.

### Tier-split BM25-vs-vector, free-tier custom BM25
Somnigraph already has FTS5 BM25 + sqlite-vec fused via RRF; a from-scratch 100-line BM25 with Porter-ish stemming is a downgrade.

---

## Connections

- **Write-path discipline over retrieval cleverness**: gitmem corroborates the Phase 18 source-sweep finding (ByteRover BM25-only, MemPalace verbatim, agentmemory write-time grounding) — the LoCoMo/LME *leaders* win on write quality, but gitmem inverts this: it has heavy write-path *ceremony* (scars, counter-args, enforcement fields) yet **no benchmark to show it pays off**. It's the write-path-emphasis thesis without the evidence.
- **Scars/wins framing** parallels other "lesson" memory systems (Recall, TrueMemory in the recent sweep); the distinctive addition is `counter_arguments` and enforcement A/B variants.
- **Rule-based write-time triples** contrast with Somnigraph's LLM-at-sleep edge detection — same "typed relations" goal, opposite build-time (cheap/rigid vs expensive/flexible). Convergent with systems that build graphs eagerly at write.

---

## Summary Assessment

gitmem is a well-engineered **product** in a different category from Somnigraph: a coding-agent guardrail whose core loop is "surface the relevant past mistake and pressure the agent to obey it." Its real investment is in the *enforcement UX* — citation rules, blocking verification gates, `why_this_matters`/`action_protocol`/`self_check_criteria`, low-confidence stubbing, and A/B testing which phrasing gets obeyed. The retrieval itself is deliberately simple: tier-exclusive BM25 or cosine×decay with a threshold, no fusion, no learned ranker, no LLM consolidation.

The single most transferable idea is the smallest one: **memories that carry their own counter-arguments/caveats** to resist over-application — a low-effort, additive schema+recall tweak, though largely redundant with Somnigraph's `contradicts` edges. Everything else (enforcement gates, dismiss-rate decay, rule-based triples, custom BM25) is either domain-specific to dev guardrails or strictly weaker than what Somnigraph already ships.

**Sharpest correction to the evidence file**: the carsteneu audit is accurate on features (hybrid ❌, autoExtract ❌, no benchmarks), but the framing to keep is that gitmem has **no end-to-end evaluation whatsoever** — nothing comparable to Somnigraph's 85.1% LoCoMo QA — and its two headline capabilities are mutually exclusive: the only truly local/offline tier (free) has **no semantic search at all** (custom BM25 over JSON), while "semantic search" is entirely a Pro/Supabase/cloud feature. The prior triage note ("scars/wins **markdown**") is also wrong: the free tier stores JSON in `.gitmem/`, not markdown files. Verdict: MAYBE — one small, marginal idea worth a note; nothing that moves the ranker, feedback, sleep, or graph modules.
