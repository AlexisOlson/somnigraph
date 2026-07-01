# ai-memory (akitaonrails) - Rust git-versioned markdown-wiki memory engine with hybrid RRF retrieval and an eval-gated LLM auto-improvement loop

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

A Rust workspace (v1.5.0, 9 crates + evals harness) by Fabio Akita. Not a library you embed — a standalone daemon (`ai-memory serve`, HTTP on :49374) plus a CLI, exposed to agents via MCP and lifecycle hooks. Explicitly multi-agent (Claude Code, Codex, OpenCode, Gemini CLI, Cursor, OpenClaw, Antigravity) and, as of v0.8+, multi-user (auth "rungs": anonymous / root-token / DB-user). This is a much larger, more production-shaped system than the "git-versioned markdown wiki, zero-LLM" table note implies — zero-LLM is one *degraded mode*, not the default posture (see cross-check).

### Storage & Schema
Two coupled stores: a **markdown wiki** (`<data_dir>/wiki/`, one `.md` per page, git-versioned via `ai-memory-wiki/git.rs`) as the source of truth, and a **SQLite** mirror (`ai-memory-store`) with FTS5 for search plus an `embeddings` table keyed by `(provider, model, dim)`. `pages` schema (evidence-confirmed ~10+ cols): title, body, body_sha256, tier, frontmatter_json, `is_latest`, `supersedes`, `pinned`, `access_count`, `last_accessed_at`, `superseded_at`, plus embedding provenance and (V15) `author_id`. A separate `links` table holds the resolved wiki-link graph.

### Memory Types
Four **tiers** (`core/src/page.rs`), credited explicitly to agentmemory: `working` (session-bound), `episodic` (per-session summaries), `semantic` (distilled durable facts — "wiki pages proper"), `procedural` (repeated patterns). Tier drives decay policy, not retrieval weighting.

### Write Path
Observations stream in through lifecycle hooks (`ai-memory-hooks`: prompt / tool-call / session boundaries, spooled). At `SessionEnd`, `hooks/synth.rs` does **rule-based, deterministic** synthesis of a `sessions/<id>.md` page (first user prompt → title, files touched, tool-call counts — no LLM). Every write passes through an **admission webhook chain** (`wiki/admission.rs`): a synchronous chain of external HTTP services that may mutate frontmatter/body (200), pass (204), or reject the write (4xx/5xx under a `FailurePolicy`). This is the engine's only extension point — new enrichment behaviours become new HTTP services, never engine changes (deliberate OCP design). A privacy-strip sanitizer (`core/sanitize.rs`) redacts tokens/keys/creds on every write, always-on.

### Retrieval
`reader.rs::hybrid_search` — genuine **three-stream RRF fusion** (k=60, the canonical constant, hardcoded):
1. FTS5 BM25 over pages (`search_pages_for_project`, fetches `limit*2`).
2. Vector cosine over stored embeddings of the matching `(provider, model, dim)` — **skipped entirely if no embedder / `query_vec=None`**.
3. **Graph-link neighbours** (`graph_neighbors_for_project`): 1-hop bidirectional expansion over the `links` table from the FTS+vector seed set, ordered by seed rank then recency.

Fusion is plain reciprocal-rank sum `Σ 1/(k + rank_i)`; no learned reranker, no weights, no tuning. The graph stream is the notable piece: because wiki links are extracted at **write** time (from `[[wikilinks]]` and cross-project `[[project:path]]` links the authoring LLM writes), graph expansion is available at query time with zero offline pass.

### Consolidation / Processing
Several LLM-mediated passes (all require a configured provider):
- **Consolidator** (`consolidate/consolidator.rs`) — the "Karpathy LLM Wiki" pattern: an LLM rewrites raw session observations into a durable page; the store's sha256 short-circuit + supersession chain make the rewrite a *new version*, not a destructive overwrite. Single-page (M7a) and atomic multi-page fan-out (M7b).
- **Lint** (`consolidate/lint.rs`) — two layers: rule-based (stale episodic, empty bodies, duplicate titles; no LLM) + optional LLM contradiction/stale-claim detection over clustered semantic pages. Findings written to git-tracked `wiki/_lint/<date>.md`.
- **Auto-improve** (`consolidate/auto_improve.rs` + `auto_improve_telemetry.rs`) — a background scheduler reviews each completed session, asks the LLM for structured wiki-edit *proposals*, and gates them: a **confidence floor** (default 0.75), size caps, an optional **executable eval gate** (runs an operator-supplied command with JSON stdin, expects `{score_before, score_after, passed}`), and a **rejection buffer** — rejected proposals are fingerprinted and fed back into future prompts so the LLM stops re-proposing them. Telemetry tracks terminal rates (approved / rejected / conflict / failed) and flags repeated rejection fingerprints.
- **Bootstrap** and **curator** — ingest existing repo docs; report on wiki health.

### Lifecycle Management
**M8 decay** (`store/decay.rs`, `consolidate/sweep.rs`), formula adapted from agentmemory:
`retention = salience·exp(−λ·age) + σ·ln(1+access_count)·exp(−μ·days_since_access)` (λ=0.02 ≈ 35-day half-life, σ=0.6, μ=0.04, cold_threshold=0.20). Only **episodic** pages decay ("semantic compounds, only episodic decays"); semantic/procedural/pinned are exempt. Below threshold → soft-delete; soft-deleted with zero subsequent access for `hard_delete_after_days` (180) → hard-delete. Versioning is supersession chains + git history.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Hybrid FTS + vector + graph RRF retrieval | `hybrid_search` implements all three streams | Validated (code-confirmed) |
| Graph retrieval without an offline pass | Links extracted at write; `graph_neighbors_for_project` 1-hop at query | Validated — real-time graph, unlike Somnigraph's sleep-built PPR |
| Zero-LLM operation | `synth.rs` + rule-based lint layer are LLM-free | Partly true: a *degraded mode*. Consolidation, contradiction lint, auto-improve, and the vector stream all require an LLM/embedder |
| Biological decay w/ access reinforcement | `decay.rs` formula + property tests | Validated (formula is agentmemory-derived, simplified) |
| Self-improving memory | auto_improve loop with confidence floor + eval gate + rejection buffer | Validated as *mechanism*; no published quality metric that it improves retrieval/QA |
| Multi-agent / multi-user | MCP configs for 8 agents; auth rungs + `users` table + `author_id` | Validated |
| End-to-end QA accuracy (e.g. LoCoMo) | none — `evals/` is an internal A/B harness | Unbenchmarked on any standard QA suite |

---

## Relevance to Somnigraph

### What ai-memory does that Somnigraph doesn't
- **Write-time graph, query-time expansion.** Somnigraph's graph is built during NREM sleep and used via PPR (`scoring.py`, `sleep_nrem.py`). ai-memory extracts an explicit link graph at write time and injects a 1-hop neighbour stream directly into RRF. This is the "real-time graph construction" the Somnigraph context lists as a known gap — though the mechanisms differ sharply (ai-memory's edges are LLM-authored `[[wikilinks]]`, not detected semantic relations).
- **Eval-gated, rejection-aware LLM edit loop.** The auto-improve loop is closer to write-path quality gating than anything in Somnigraph's sleep pipeline. The **rejection buffer** — persisting fingerprints of declined proposals and feeding them back to the LLM to suppress re-proposal — has no analogue in `sleep_nrem.py`, which can re-surface the same merge/archive suggestion a user already declined.
- **Admission webhook chain** — a clean OCP write-path extension point (enrich/validate/reject via external services). Somnigraph has no write-path admission hook.
- **Multi-agent + multi-user** with auth rungs. Somnigraph is deliberately single-user.

### What Somnigraph does better
- **Retrieval quality.** Somnigraph has a 26-feature LightGBM reranker (NDCG=0.7958, +6.17pp over formula) and Bayesian-optimized RRF k=14; ai-memory uses fixed k=60 reciprocal-rank sum with no reranking or tuning. `reranker.py` has no counterpart here.
- **Explicit feedback loop.** Somnigraph's per-query utility ratings + EWMA + UCB + Hebbian PMI (measured Spearman r=0.70) are a graded signal ai-memory lacks — its only reinforcement is decay's `access_count`.
- **Measured consolidation.** Somnigraph's sleep is benchmarked (85.1% LoCoMo QA); ai-memory's consolidation/auto-improve loop ships without a quality metric.
- **Typed semantic edges** (supports/contradicts/evolves/revision/derivation) detected during sleep vs. ai-memory's untyped, LLM-authored wiki links.

---

## Worth Stealing (ranked)

### 1. Rejection buffer / declined-suggestion memory (Medium)
**What**: Persist a fingerprint of every consolidation suggestion a user declines (a merge, an archive, an edit), and feed the recent-and-relevant declined set back into the next consolidation prompt so the LLM stops re-proposing what was already rejected. ai-memory does exactly this in `auto_improve.rs` (rejection buffer, `MAX_REJECTION_CONTEXT`, `REJECTION_CONTEXT_DAYS=180`) and tracks "repeated_rejection_fingerprint" as a telemetry alarm.
**Why**: Directly convergent with the active `research/comparison-declined-list` branch and the new `docs/declined.md`. Somnigraph's `sleep_nrem.py` merge/archive proposals have no memory of prior rejections — a user who declines a merge can be re-asked next sleep. A declined-list with fingerprints closes that loop.
**How**: New table (or a `_declined` wiki-analogue) keyed by a stable fingerprint of the (pair, action) suggestion; `sleep_nrem.py` filters candidate suggestions against it before presenting, and injects a compact "previously declined" block into the classification prompt. Age out at ~180d like ai-memory.

### 2. Executable eval gate on LLM-proposed edits (Medium)
**What**: Before applying an LLM-proposed memory edit/merge, run an operator-supplied command that scores before/after (`{score_before, score_after, passed}`) and reject if it doesn't improve. ai-memory's `[auto_improve.eval]` gate is disabled by default but the plumbing is real.
**Why**: Somnigraph already has a rich eval harness (probe_recall, ground truth, reranker NDCG). Wiring a *self-check* — "does this sleep-proposed merge improve held-out recall on affected queries?" — would make consolidation edits verifiable rather than trusted. Matches the repo's honest-accounting ethos.
**How**: In the NREM merge/archive path, for high-impact edits, run a scoped probe over queries that touch the affected memories; gate on ΔNDCG ≥ min_delta. High effort to do well; the concept is what's worth borrowing.

### 3. Graph expansion as an explicit RRF stream (Low, note-only)
**What**: Rather than blending PPR into a single score, treat graph neighbours as their own ranked list and RRF-fuse it alongside FTS and vector (`hybrid_search`).
**Why**: Somnigraph already does graph expansion via PPR in `scoring.py`, so this is largely redundant — but the RRF-stream framing is simpler to reason about and ablate than a score blend, and cleanly separable in eval.
**Redundant**: Somnigraph's PPR is more expressive; adopt only if a simpler, more ablatable fusion is wanted.

---

## Not Useful For Us

### Admission webhook chain / multi-agent / multi-user / git-versioned wiki
Architecturally elegant but orthogonal to Somnigraph's single-user, single-agent, SQLite-native MCP design. The git-wiki-as-source-of-truth is a substrate choice Somnigraph deliberately doesn't make.

### Fixed-k reciprocal-rank fusion, no reranker
Strictly weaker than Somnigraph's tuned RRF + learned reranker; nothing to import.

### Rule-based deterministic session summarization
Somnigraph's LLM-mediated sleep already covers this ground with higher fidelity; the zero-LLM synth is a fallback, not an advance.

---

## Connections

- **agentmemory** (see `agentmemory.md`) — ai-memory *explicitly* credits agentmemory for both its four-tier model (`page.rs`) and its decay formula (`decay.rs`). This is a direct lineage, not convergence. Somnigraph's Phase 18 finding (write-path quality, not retrieval, is what leaders win on) is echoed here: ai-memory's differentiated work is all write-side (admission chain, auto-improve, consolidation), while retrieval is deliberately plain.
- **Karpathy "LLM Wiki"** — the consolidator names this pattern (LLM rewrites a page as a new supersession version). Convergent with any memv-style supersession lineage in the corpus.
- **Rejection buffer** convergent with the active `docs/declined.md` work — external corroboration that a declined-suggestion memory is a real need, not a one-off.

---

## Summary Assessment

ai-memory is a serious, production-shaped Rust memory *daemon* whose center of gravity is the **write and maintenance path**, not retrieval. Retrieval is deliberately plain — three-stream RRF at fixed k=60, no reranker, no feedback — while the interesting engineering is in admission webhooks, git-versioned supersession, LLM consolidation, contradiction lint, and an eval-gated auto-improvement loop. That is the mirror image of Somnigraph, whose sophistication is concentrated in retrieval (learned reranker, feedback loop, PPR) and whose write path is comparatively thin. The two systems are complementary more than competitive, and both independently land on the Phase 18 thesis that write-path discipline is where quality lives.

The single most valuable thing to take is the **rejection buffer**: a persisted, fingerprinted memory of declined consolidation suggestions that is fed back to suppress re-proposal. It maps cleanly onto `sleep_nrem.py` and is directly convergent with the branch this analysis was written on. The eval gate on proposed edits is a strong second — it would let Somnigraph *verify* consolidation edits with its existing harness rather than trusting them.

What's overhyped: the "zero-LLM" framing. The rule-based mode exists, but every capability that makes ai-memory more than an FTS index over markdown — consolidation, contradiction detection, auto-improvement, vector search — requires an LLM or embedder. And nothing is benchmarked on end-to-end QA, so there is no comparability to Somnigraph's 85.1% LoCoMo number; the `evals/` harness is internal A/B only.
