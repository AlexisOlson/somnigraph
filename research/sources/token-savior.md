# token-savior — token-efficiency layer for Claude Code with a code graph + a bandit-ranked memory subsystem

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

token-savior (`pip install token-savior-recall`, MIT, ~913★, 97.7% Python) is primarily a **token-compaction / code-navigation** MCP server for coding agents. Its headline is "-77% to -80% active tokens on tsbench" — a *tool-output compaction* benchmark (it wraps `git`/`grep`/`pytest`/`docker`/`gh` output through per-tool "compactors" and serves a Tree-sitter code graph for 15+ languages). The persistent-memory subsystem in `src/token_savior/memory/` is a secondary component. This analysis focuses on that subsystem, which is the only part comparable to Somnigraph.

### Storage & Schema
SQLite (WAL) via `memory_db.py` / `memory_schema.sql`. Central unit is the **observation** with ~18 structured fields: `type`, `title`, `content`, `why`, `how_to_apply`, `symbol`, `file_path`, `context`, `tags` (JSON), `importance` (1-10), `is_global`, `agent_id`, `narrative`, `facts`, `concepts`, `expires_at_epoch`, `decay_immune`, `content_hash`. Satellite tables: `sessions`, `session_summaries`, `reasoning_chains`, `user_prompts`, `observation_links`, `consistency_scores`, `decay_config`, `corpora`, `adaptive_lattice`, `obs_vectors` (sqlite-vec vec0). Five separate FTS5 tables (observations, session_summaries, user_prompts, reasoning_chains, tool_captures).

### Memory Types
No episodic/semantic axis. Instead **18 observation `type`s** double as a taxonomy *and* a trust/decay tier: `guardrail`, `convention`, `decision`, `user`, `feedback`, `error_pattern`, `bugfix`, `warning`, `command`, `config`, `infra`, `reference`, `research`, `note`, `idea`, `ruled_out`, etc. The type drives everything downstream: decay rate, ROI multiplier, LinUCB `type_score`, contradiction eligibility. Rule-ish types (`guardrail`/`convention`/`decision`/`user`/`feedback`) are `_DECAY_IMMUNE_TYPES`.

### Write Path
Two paths. (1) **Manual** `memory_save` — computes SHA-256 `content_hash` (exact-dup skip), runs `global_dedup_check` (cross-project Jaccard on titles, threshold 0.85) and `semantic_dedup_check` (same-type Jaccard); ≥0.95 skips insert, 0.85-0.95 tags `near-duplicate`. `_is_corrupted_content()` rejects tool-artifact noise; `strip_private()` removes `<private>` spans. (2) **Auto-extract** (`memory/auto_extract.py`) is **opt-in and default-off** — needs `TS_AUTO_EXTRACT=1` + `TS_API_KEY`. When on, a PostToolUse hook spawns a daemon thread that POSTs the tool I/O to the Anthropic API (`claude-sonnet-4-6`) asking for 0-3 observations (types limited to bugfix/convention/warning/guardrail/infra/command), tags them `auto-extract`, dedup collapses downstream. Note: dedup is **title/content-hash Jaccard only**, no embedding-based semantic dedup on the write path despite the "semantic_dedup" name.

### Retrieval
`memory/search.py` `hybrid_search()`: FTS5 rows (computed by caller for DRY quarantine/type filtering) fused with a sqlite-vec k-NN pass via **RRF, k=60** (Cormack 2009 reference constant). Vector side uses FastEmbed + Nomic `nomic-embed-text-v1.5-Q` (768-d, task-prefixed `search_query:`/`search_document:`). **Vector search is opt-in** (`token-savior-recall[memory-vector]` extra) — with `VECTOR_SEARCH_AVAILABLE=False` (the default install) it silently degrades to FTS-only. Quarantined obs (validity<0.40) excluded unless `include_quarantine=True`.

Injection ranking is separate and more interesting: **`linucb_injector.py` — a LinUCB contextual bandit** (Li et al. 2010) picks which observations to inject at tool boundaries. 10 hand-defined features φ = [type_score, age_score, access_score, semantic_sim (Jaccard prompt↔obs), mode_match, tokens_used_pct, task_is_edit, task_is_debug, symbol_match, has_context]. UCB score = θᵀφ + α·√(φᵀA⁻¹φ), online updates A←A+φφᵀ, b←b+rφ, pure-Python 10×10 Gauss-Jordan inverse, persisted to `linucb_model.json`. Wired live in `server_handlers/memory.py:866` (`rank_observations`) with **delayed implicit reward**: injected obs go into `_linucb_pending`; if referenced within 30 min, `_linucb_credit_reward(ids, reward=1.0)` applies the online update (line 771/112).

### Consolidation / Processing
No sleep/LLM-mediated consolidation cycle. Instead a set of **synchronous sweeps**: `run_decay`, `run_roi_gc`, `run_consistency_check`, `dedup_sweep`, `auto_link_observation`/`run_promotions` (links.py), `leiden_communities.py` (Leiden clustering on the *code* graph, not memories). Contradiction detection (`consistency.py detect_contradictions`) is **rule-based regex**: 6 hardcoded opposite-phrase pairs (never/always, disable/enable, avoid/use, off/on, EN+FR) trigger an FTS lookup among rule-type obs — no LLM, no semantic classification.

### Lifecycle Management
Three overlapping eviction mechanisms, all archive (soft-delete `archived=1`), never hard-delete:
- **`decay.py`**: per-type `decay_config` (rate/min_score/boost_on_access); `relevance_score = decay_rate^days + boost·access_count`, floored. Archival candidates = age>90d AND unread>30d AND access<3, plus per-type TTLs and zero-access rules (note>30d, research>45d, idea>60d, bugfix>90d). Reheat on read via `_bump_access`.
- **`roi.py`**: token-economy GC. `ROI = tokens_saved_per_hit(200) × P(hit) × horizon(30d) × type_multiplier − tokens_stored`, `P(hit)=exp(−0.05·days_since_access)·(1+0.1·access)`. ROI<0 → archival candidate. decay_immune forced multiplier≥5.
- **`consistency.py`**: per-obs Beta(α,β) **validity posterior** (prior 2/1, biased valid). Symbol-linked obs are swept with `git log -S symbol` (3s timeout) — if the symbol moved after the obs was written, β++ (staleness). validity<0.60 → `stale_suspected`, <0.40 → `quarantine` (dropped from search).
- `observation_links` with `supersedes`/`contradicts` relations; `memory_delete` soft-delete + `observation_restore`.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| "-77–80% active tokens, 100% on tsbench" | README/CLAUDE.md; `benchmarks/` | Plausible **but it's a tool-output-compaction benchmark, not memory QA**. Not comparable to Somnigraph's 85.1 LoCoMo QA. |
| Hybrid FTS5+vector RRF retrieval | `memory/search.py`, real k=60 RRF | Real, clean — **but vector side is opt-in** (`[memory-vector]` extra); default install is FTS-only. |
| LinUCB bandit ranks injected memories, learns online | `linucb_injector.py` + `server_handlers/memory.py` wiring | Validated as code: fully wired, delayed 30-min implicit reward, persisted weights. No reported effectiveness metric (no ablation showing it beats a static ranker). |
| "Thompson-sampled persona/personality lattice, profile adaptation" (evidence file + table note) | `memory/lattice.py` | **Mischaracterized.** It is a Beta-Binomial bandit over *source-code fetch granularity* (full body / signature / name-only), a compression policy keyed on code-tool context (navigation/edit/review). Nothing to do with persona/personality. |
| Bayesian trust + contradiction + quarantine | `consistency.py` | Validity posterior + git-staleness real; contradiction detection is 6 hardcoded regex pairs, brittle vs an LLM classifier. |
| LLM auto-extraction write path | `auto_extract.py` | Real but **default-disabled**; simple 0-3-obs-per-tool-use prompt, no quality/salience gating beyond type whitelist + dedup. |

---

## Relevance to Somnigraph

### What token-savior does that Somnigraph doesn't
- **Feature-based contextual bandit (LinUCB) for injection ranking** with online delayed implicit reward. Somnigraph's `docs/proposals/proactive-injection.md` design uses **per-memory Thompson gating** (a Beta posterior *per arm/memory*). LinUCB is the complementary axis: a *shared* linear model over memory+context features, so unseen/cold-start memories are ranked from generalizable weights, not a per-memory prior. This is the single most relevant artifact in the repo — a working implementation of the exact problem Somnigraph is currently designing. Gap module: `docs/proposals/proactive-injection.md` / (future) injection hook.
- **ROI-as-eviction-criterion**: keep a memory iff expected token-savings exceeds storage cost. Somnigraph's `scoring.py`/decay uses exponential half-lives with no explicit value-vs-cost ledger. Different framing of the same lifecycle question.
- **Artifact-grounded staleness**: memories linked to a code `symbol` are auto-invalidated by `git log -S`. Somnigraph has no entity/symbol grounding (`db.py` has no entity table), so it structurally cannot verify a memory against an external artifact.
- **Injection ROI accounting** (`get_injection_stats`: tokens_injected vs tokens_saved_est per session) — a self-measured proxy for whether recall is net-positive.

### What Somnigraph does better
- **Consolidation**: Somnigraph's `sleep_nrem.py`/`sleep_rem.py` do LLM-mediated pairwise edge typing (supports/contradicts/evolves), merge/archive, gap analysis, taxonomy. token-savior's "contradiction detection" is 6 regex pairs — no semantic reasoning.
- **Learned reranker**: Somnigraph's `reranker.py` is a 26-feature LightGBM model trained on 1032 labeled queries (NDCG=0.7958, +6.17pp). token-savior's ranking is either RRF (retrieval) or a 10-feature *linear* bandit — no supervised learning-to-rank, no held-out eval.
- **Graph-conditioned retrieval**: Somnigraph's PPR expansion over typed memory edges vs token-savior's memory graph being just `supersedes`/`contradicts` links with no propagation (its real graph is the *code* graph).
- **Measured feedback→GT correlation** (Spearman r=0.70). token-savior reports no such validation for LinUCB or the lattice.
- **End-to-end QA evidence** (85.1 LoCoMo). token-savior has no memory-QA number at all.

---

## Worth Stealing (ranked)

### 1. LinUCB (feature-based) as an alternative/complement to per-memory Thompson gating (Medium)
**What**: A shared linear contextual bandit over [memory features × context features] scoring whether to surface a memory, with UCB exploration and online delayed reward, instead of (or alongside) a per-memory Beta posterior.
**Why**: Somnigraph's `docs/proposals/proactive-injection.md` currently proposes per-memory Thompson gating, which cold-starts every new memory from scratch and can't share signal across arms. A feature-based bandit generalizes: a never-injected memory inherits weights learned from similar memories (type, recency, query-similarity, context-budget). It directly addresses the anti-starvation concern in the design doc from a different angle. The 10-feature set here (esp. `tokens_used_pct` — inject heavier hints when context budget is low, and `semantic_sim` between prompt and memory) is a concrete starting feature vector. Pure-Python, no numpy — trivially embeddable.
**How**: New module paralleling the design doc; features reuse `reranker.py` extraction + a live context (last query, session token budget). Reward = the same implicit signal Somnigraph already has (a hinted memory later `recall`'d or rated >0). The delayed-credit window pattern (`_linucb_pending` + 30-min horizon in `server_handlers/memory.py`) is the mechanism to copy for implicit reward without an explicit rating.

### 2. ROI ledger as a decay/eviction lens (Low)
**What**: Score each memory by `expected_tokens_saved(P(hit), horizon, type) − tokens_stored` and treat negative-ROI memories as archival candidates; expose an aggregate `net_roi` stat.
**Why**: Somnigraph decays on time alone. An explicit value-vs-cost ledger is a cheap, interpretable second signal — and a *reporting* tool (`get_roi_stats` by type) would make "is this corpus paying for itself" answerable, echoing the STEWARDSHIP honest-accounting ethos. Not necessarily an eviction driver, but a diagnostic.
**How**: A read-only analysis pass over the memory table using access_count + category multipliers; surface in a stats tool, don't wire to deletion initially.

### 3. Delayed implicit-reward credit window (Low)
**What**: When a proactive hint is shown, stash its features + timestamp; if the corresponding memory is used within N minutes, credit the ranker positively; otherwise it decays to no-reward.
**Why**: Somnigraph's feedback loop is *explicit* per-query ratings. A proactive-injection hook has no natural rating moment — this pattern manufactures an implicit label (shown-then-used = good) that the proactive design needs. It's the missing reward-signal design for injection.
**How**: Small pending-dict in the injection hook keyed by memory id; on next `recall`/reference, resolve pending → reward.

---

## Not Useful For Us

### Tree-sitter code graph, compactors, MDL distiller, Markov prefetcher
The bulk of the repo (15-language annotators, `graph_ranker.py`, `markov_prefetcher.py`, per-tool output compactors, `leiden_communities.py` on code symbols) is a coding-agent token-efficiency product. Somnigraph is domain-agnostic conversational memory — none of the code-structure machinery transfers.

### The "adaptive lattice" (source-fetch granularity bandit)
`lattice.py` chooses how much of a function body to return. Interesting Beta-Binomial bandit, but the decision it governs (code compression level) has no Somnigraph analog. (Flagging because the table note/evidence file miscall it a "persona lattice" — it is not.)

### Rule-based contradiction regex
6 hardcoded opposite-phrase pairs are strictly weaker than Somnigraph's `sleep_nrem.py` LLM contradiction classification.

---

## Connections

- **Convergent with the Phase-18 write-path-discipline finding** (`ai-memory-comparison.md`, ByteRover/agentmemory): token-savior's core also ships **FTS-first** (vector opt-in) and leans on write-time structure (typed observations, symbol grounding, dedup) rather than fancy retrieval — another data point that the leaders win on write-path quality, not retrieval cleverness.
- **LinUCB vs Thompson**: the proactive-injection design (`docs/proposals/proactive-injection.md`) chose Thompson; this is the closest external implementation of the *same gating problem* with the alternative bandit family. Worth a direct compare in that doc.
- **Bayesian per-item validity + artifact staleness** rhymes with supersession/temporal-validity patterns seen in memv/memos analyses, but token-savior is unusual in grounding the check to an *external* artifact (git history) rather than another memory.

---

## Summary Assessment

token-savior is a code-agent **token-efficiency** tool that happens to carry a surprisingly complete memory subsystem. The evidence file is accurate on breadth (it corrects a naive "everything absent" prior and rightly credits FTS5×5, RRF hybrid, decay/quarantine/contradiction, auto-extract) but overstates realized capability in two places that matter: the "personality/persona lattice" is actually a *code-source compression* bandit, and the headline "-80% tokens / 100% on tsbench" is **tool-output compaction, not memory QA** — there is no end-to-end recall/QA number to compare against Somnigraph's 85.1 LoCoMo. Several advertised mechanisms (vector search, LLM auto-extract) are **default-disabled**; the shipping core is FTS-only with hash/Jaccard dedup.

The single most valuable thing for Somnigraph is the **LinUCB injection bandit**: a live, wired, pure-Python contextual bandit that ranks which memories to surface at tool boundaries and learns online from a **delayed implicit reward** (injected-then-referenced-within-30-min = +1). That is precisely the problem `docs/proposals/proactive-injection.md` is designing right now, solved with the *other* bandit family (feature-generalizing LinUCB vs per-memory Thompson). It won't be adopted wholesale — Somnigraph's memories aren't code observations and it already has a supervised reranker — but the feature vector, the delayed-credit window, and the shared-weight generalization argument are direct, concrete inputs to an active design, and the ROI-ledger framing is a cheap honest-accounting diagnostic worth a stats tool.

**Verdict: DIVE** — the injection bandit maps onto an open Somnigraph work item and offers a genuinely different design point, not a redundant one.
