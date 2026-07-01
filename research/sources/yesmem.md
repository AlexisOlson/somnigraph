# YesMem — Go-based local-first agent memory with provenance-weighted turn-decay scoring and anticipated-query vocabulary bridges

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

YesMem (`github.com/carsteneu/yesmem`) is a large Go monorepo (~218 MB, 50+ `cmd_*.go` entry points) that is much more than a memory store: it bundles a memory system, an LLM proxy with cache optimization, a multi-agent orchestrator, a sandboxed tool-execution system ("Caps"), a code-graph indexer, and a persona engine. This analysis focuses on the memory subsystem, which is the part comparable to Somnigraph.

### Storage & Schema
- **SQLite** with FTS5 virtual tables. The core table is `learnings` (`internal/storage/schema.go`), with ~22+ columns and several junction tables: `learning_entities`, `learning_actions`, `learning_keywords`, `learning_anticipated_queries`.
- The schema is unusually rich in **scoring/telemetry columns**: `use_count`, `save_count`, `fail_count`, `noise_count`, `match_count`, `inject_count`, `stability`, `ebbinghaus_stability`, `turns_at_creation`, `current_turn_count`, `emotional_intensity`, `importance`, `source`, `origin_tool`, `domain`, `trigger_rule`, `task_type`.
- Vectors are **512-dimensional** (README "Foundations"), stored via an internal embedding package (`internal/embedding/`, `internal/ivf/` for an IVF index). Somnigraph uses 1536d (OpenAI) or 384d (fastembed).

### Memory Types
- Category taxonomy is workflow/coding-oriented, not the episodic/semantic/procedural split Somnigraph uses. `categoryWeight()` in `internal/models/scoring.go` enumerates: `pivot_moment` (1.6), `gotcha` (1.5), `explicit_teaching` (1.4), `decision`/`strategic` (1.3), `unfinished` (1.2), `fact` (1.1), `pattern` (1.0), `preference` (0.8), `relationship` (0.7). Category is a scoring weight, not a storage partition.
- Procedural memory is realized separately as **"Caps"** — user-defined, versioned, sandboxed executable tools (bash/JS) run via an "ai-jail" sandbox (`internal/cap/executor.go`). This is closer to a skill/tool registry than a memory type.

### Write Path
This is YesMem's strongest area and where it most differs from Somnigraph. A multi-phase extraction pipeline (`internal/daemon/extract.go`, `internal/extraction/`) runs offline over Claude Code sessions:
- **Content-aware preprocessing**: per-content-type truncation limits (paste 1000 chars, tool results 200, thinking 0) claimed to cut Pass-1 input ~70%.
- **3-method dedup** (`internal/extraction/dedup.go`, `consolidate.go`): rule-based `IsSubstanzlos()` (drops substance-less fragments) + `BigramJaccard() > 0.85`; then LLM/embedding dedup (token Jaccard ≥0.5, cosine ≥0.92); plus **pre-admission dedup** at `remember()` time via `TokenSimilarity` (`internal/daemon/handler_learnings.go:141`).
- **Provenance labeling**: every learning carries `source` (5-tier: user_stated / agreed_upon / claude_suggested / llm_extracted / hook_auto_learned) and `origin_tool`, both feeding trust multipliers at score time.
- **Anticipated queries**: extraction generates 3-5 concrete search phrases per learning, stored in a junction table + porter-stemmed FTS (`anticipated_queries_fts`) and, in the benchmark path, **each phrase embedded as a separate vector** for query-to-query matching (`internal/benchmark/locomo/runner.go:262-273`). This is a deliberate vocabulary-gap bridge.
- Later phases: clustering (agglomerative on embeddings, cosine 0.85), recurrence detection, narrative/handover generation, persona trait extraction.

### Retrieval
- **Hybrid BM25 + vector.** BM25 is *tiered* (`SearchLearningsBM25Ctx`, `learnings_search.go:162`): Tier 1 = 100% terms AND-matched (score ≤100), Tier 2 = 66% (≤60), Tier 3 = 33% (≤40), with terms sorted by IDF (rarest first) so dropped terms are the common ones. Scores normalized within tier.
- **Fusion is nominally RRF but the RRF score is discarded.** `RRFMerge` (`internal/embedding/hybrid.go`) accumulates `1/(k+rank)` per doc but the *final sort key* is: `semantic` = cosine×100, `hybrid` = cosine×100 + 5, `keyword` = pre-normalized BM25. The RRF sum is computed and never used in `finalScore`. So the effective ranker is "cosine-priority with a small bonus for appearing in both channels," not Reciprocal Rank Fusion. (See cross-check.)
- **Relevance re-scoring** happens separately via `ComputeScore()` / `ComputeContextualScore()` in `scoring.go` — a hand-tuned multiplicative formula (see below), applied to candidates. There is **no learned reranker**; this is the equivalent of Somnigraph's *fallback* formula, not its LightGBM model.
- **9 search "modes"** (search, deep_search with ±3 message context and thinking, hybrid_search, keyword_search, docs_search, code-graph traversal, query_facts, etc.) exposed as MCP tools. Much of the LoCoMo gain is attributed to agentic iteration across these tools, not to fusion quality.

### The scoring formula (`ComputeScore`)
A product of hand-tuned factors, notable for baking write-path quality and a feedback signal directly into ranking:
```
score = categoryWeight × turnBasedDecay × useBoost × noisePenalty
        × precisionFactor × explorationBonus × emotionalBoost
        × importanceBoost × fixationPenalty
```
- `TurnBasedDecay`: `exp(-turns_since / effective_stability)` where "turns" are **project conversation turns, not wall-clock**. A project paused 3 months with 0 turns has *zero* decay. `effective_stability = stability × (1 + log2(1 + use + 2·save))` — spaced-reinforcement growth. Floors: 0.1 universal, 0.5 for user-stated.
- `precisionFactor`: use/inject ratio, **gradually activated between 3–12 injections to avoid a cliff** (0.5–1.5 range). This is an *implicit, no-label* analogue of Somnigraph's explicit feedback loop.
- `explorationBonus`: +30% until injected 3× (cold-start exploration).
- `OriginMultiplier` (`scoring.go:268`): trust weight by provenance — user 1.0, file_read 0.9, bash_input 0.7, llm_extracted 0.6, web_external 0.4, cap_* 0.5. Untrusted-source memories are down-weighted at rank time.
- `ComputeContextualScore` adds project-match (turn-graduated 1.5/1.3/1.1×), entity-in-current-filepath (1.4×), and domain (1.2×) boosts.

### Consolidation / Processing
Offline daemon-driven extraction + clustering + recurrence detection + persona updates (above). Contradiction handling ("Pearce & Hall" contradiction-boost — corrections get boosted), supersede-chain resolution via recursive CTE (max 10 hops, cycle detection). This is rule/LLM-hybrid pipeline processing, not a biologically-framed NREM/REM sleep cycle, but functionally overlaps Somnigraph's sleep (merge/dedup/contradiction/supersession).

### Lifecycle Management
- Turn-based Ebbinghaus decay (above); no hard deletion by default.
- Supersede/replace with recursive-CTE chain resolution; `unfinished` tasks auto-archive after TTL (default 30 days).
- `quarantine_session()` / `skip_indexing()` for soft exclusion. Sessions archived permanently to `~/.claude/yesmem/archive/`.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| LoCoMo overall 0.87 | `docs/BENCHMARK.md`: agentic mode, Claude Opus 0.8733 (150-q 10% sample, Sonnet judge); gpt-5.4 0.8649 (full 1540q, gpt-5.4-mini judge) | **Plausible and genuinely end-to-end QA** (not R@k) — directly comparable to Somnigraph's 85.1%. But self-authored harness, self-corrected dataset, and self-selected LLM judge; the 0.87 headline is a 150-q sample. |
| Hybrid BM25 + vector with RRF | `hybrid.go RRFMerge` | **Overstated.** RRF is computed but discarded; final rank is cosine-priority. Fusion is real (union of channels), the *reciprocal-rank-fusion mechanism* is not the ranking function. |
| Turn-based (not wall-clock) decay | `scoring.go:44 TurnBasedDecay` | **Validated in code.** Genuinely different and defensible for coding workflows. |
| Provenance/trust-weighted scoring | `scoring.go:268 OriginMultiplier`, 5-tier `source` | **Validated.** Real write-path quality gating baked into rank — something Somnigraph lacks. |
| Anticipated-query vocabulary bridge | `learnings_search.go:406`, `benchmark/.../runner.go:269` | **Validated.** Each memory carries predicted query phrases, FTS-indexed and (in bench) separately embedded. |
| Agentic iteration +0.20 over static | `docs/BENCHMARK.md` static 0.6675 vs agentic 0.86 | **Plausible**, consistent across two models; but confounds retrieval quality with LLM tool-use skill (their own "retrieval is not the bottleneck" finding). |
| "Reproducible / open methodology" | `docs/BENCHMARK.md` with full CLI | **True and commendable** — parameters, corrected dataset (locomo-audit vendored), and judge prompts all published. |

---

## Relevance to Somnigraph

### What YesMem does that Somnigraph doesn't
- **Write-path quality gating at scale** (`extract.go`, `dedup.go`): content-aware truncation, 3-method dedup including pre-admission dedup, provenance labeling. Somnigraph has no write-time quality/dedup gate — this is the exact gap the Phase-18 source sweep flagged (write-path quality, not retrieval, is what LoCoMo leaders win on). YesMem is independent corroboration.
- **Provenance-weighted ranking** (`OriginMultiplier`): trust as a first-class rank feature. Somnigraph's `reranker.py`/`scoring.py` have no source-trust feature.
- **Anticipated queries**: a stored, embedded query-phrase bridge that targets Somnigraph's #1 documented ceiling — the ~88% multi-hop *vocabulary gap* (`docs/multihop-failure-analysis.md`). Convergent with Somnigraph L5b's synthetic-node bridges but realized at the memory level (each memory predicts its own queries).
- **Turn-based decay**: decay clocked on project activity, not wall time — arguably better-aligned to bursty coding work than Somnigraph's per-category wall-clock exponential decay.
- **Entity/filepath and domain context boosts** at query time (`ComputeContextualScore`) — Somnigraph has no current-working-file signal.

### What Somnigraph does better
- **Learned reranker.** YesMem's ranker is entirely hand-tuned multiplicative factors — exactly what Somnigraph's LightGBM model (`reranker.py`, NDCG 0.7958, +6.17pp over formula) *replaced*. YesMem is at Somnigraph's fallback-formula tier for ranking.
- **Measured feedback loop.** Somnigraph has explicit 0-1 utility ratings with Spearman r=0.70 vs GT and UCB exploration. YesMem's `precisionFactor`/`explorationBonus` are the *same idea* but implicit (use/inject counts) with no validation of correlation to relevance.
- **Graph-conditioned retrieval.** Somnigraph's PPR expansion over typed sleep-detected edges (`scoring.py`) and betweenness reranker feature have no equivalent in YesMem's memory ranker (it has a *code* graph, not a memory graph).
- **Actually uses its fusion.** Somnigraph's RRF (k=14, Bayesian-optimized) is the real ranking primitive; YesMem's is nominal.

---

## Worth Stealing (ranked)

### 1. Anticipated-query vocabulary bridge (Medium)
**What**: At write time, generate 3-5 concrete search phrases the memory should answer, store them, FTS-index them, and embed each as its own vector keyed to the parent memory.
**Why**: Somnigraph's documented retrieval ceiling is an ~88% multi-hop vocabulary gap. This attacks it at the memory level — the memory advertises the questions it answers, so a query phrased differently from the memory's content can still hit via query-to-query match. Complements L5b's synthetic bridges rather than duplicating them.
**How**: Add an `anticipated_queries` field populated during `remember()` (small LLM call) and during sleep for legacy memories; add an FTS channel + separate embedding rows in `embeddings.py`/`fts.py`; add "matched via anticipated query" as a reranker feature in `reranker.py`. Offline-testable against LoCoMo multi-hop before wiring live.

### 2. Provenance/trust as a reranker feature (Low)
**What**: Down-weight memories by source trust (user-stated > agreed > llm-extracted > web-external) at rank time.
**Why**: Somnigraph stores category/priority but no provenance-trust signal; low-trust auto-captured memories rank identically to user corrections. A trust feature is cheap write-path quality gating.
**How**: Add an `origin`/`source` column (Somnigraph already distinguishes `source="correction"|"auto"|"session"` in the memory schema) and expose an `origin_trust` numeric feature to `reranker.py`. The label already exists in captures — this is mostly a feature-extraction change.

### 3. Cliff-free gradual activation of a feedback signal (Low)
**What**: `precisionFactor` blends from neutral (1.0) toward the use/inject ratio only between 3-12 impressions, avoiding a hard cutoff when data is sparse.
**Why**: Somnigraph's feedback EWMA can over-react to a single early rating. The gradual-activation envelope is a clean pattern for trusting a per-memory statistic only once enough observations exist.
**How**: When forming the feedback/UCB term in `scoring.py`/`reranker.py`, gate its weight by an activation ramp on observation count rather than applying it from the first rating.

### 4. Turn-based (activity-clocked) decay as an alternative half-life axis (Low, note-only)
**What**: Decay measured in conversation turns since creation, with effective stability growing via `log2(1 + use + 2·save)`, so paused projects don't decay.
**Why**: Worth *considering* as a second decay axis for procedural/project memories where wall-clock is the wrong clock. Somnigraph deliberately chose wall-clock biological decay, so this is a contrast to note, not an obvious adopt.
**How**: Would require a per-session turn counter; likely not worth the schema change unless project-scoped decay becomes a goal.

---

## Not Useful For Us

### Caps, ai-jail sandbox, LLM proxy, multi-agent orchestrator, code-graph indexer
Large parts of the repo (sandboxed executable tools, the sawtooth cache-collapse proxy, spawn/heartbeat multi-agent system, Tree-sitter code graph) are agent-runtime infrastructure orthogonal to Somnigraph's "memory library for one Claude Code user" scope.

### Persona engine (50+ traits, 6 dimensions)
Preference/persona modeling is interesting but YesMem's is trait-extraction bookkeeping, not the cross-domain preference-state synthesis PERMA measures. Not a source of a portable mechanism.

### Their RRF implementation
Since the RRF score is discarded in favor of cosine-priority, there is nothing to copy — Somnigraph's actual RRF is more principled.

---

## Connections
- **Write-path-quality thesis** (`docs/sessions/2026-06-28-phase18-source-sweep.md`, `ai-memory-comparison.md`, `agentmemory.md`, `byterover`): YesMem is another data point that the LoCoMo leaders win on the write path (dedup, provenance, anticipated queries), not on exotic fusion — its ranker is hand-tuned and its "RRF" is nominal, yet it hits 0.87. Strong convergent evidence.
- **Vocabulary-bridge convergence**: anticipated queries are the memory-level cousin of Somnigraph's L5b synthetic-node bridges (`docs/multihop-failure-analysis.md`) and MemPalace-style query anticipation — multiple systems independently bridging the multi-hop vocabulary gap.
- **Corrected-LoCoMo lineage**: like Somnigraph L5b, YesMem vendors the `dial481/locomo-audit` corrections (6.4% ceiling) — same dataset hygiene, making the 0.87 vs 85.1% comparison unusually apples-to-apples.
- **Implicit-vs-explicit feedback**: YesMem's use/inject `precisionFactor` is the un-validated implicit form of Somnigraph's explicit feedback loop with measured r=0.70.

---

## Summary Assessment

YesMem's memory core is a **write-path-heavy, hand-tuned-scoring** system whose real contribution is disciplined ingestion: content-aware truncation, three layers of dedup, five-tier provenance labeling, and anticipated-query generation. On the retrieval side it is behind Somnigraph — no learned reranker, no memory graph, no validated feedback loop, and a "RRF" fusion whose reciprocal-rank scores are computed and then thrown away in favor of a cosine-priority sort. That it still reaches 0.87 on corrected LoCoMo QA reinforces the corpus-wide finding that the write path, not the fusion function, is where these benchmarks are won.

The single most valuable idea for Somnigraph is **anticipated queries**: memories that carry the questions they answer, FTS-indexed and separately embedded, directly targeting the multi-hop vocabulary gap that Somnigraph's own failure analysis names as its retrieval ceiling. Provenance-weighted ranking and cliff-free feedback activation are cheap, additive borrowings.

What's overhyped: the "Hybrid BM25 + vector, Reciprocal Rank Fusion" framing (RRF is nominal), and the benchmark's self-contained nature. The 0.87 is genuine end-to-end agentic QA — a fair comparison to Somnigraph's 85.1% — but it is self-authored (the same owner maintains the `ai-memory-comparison` table this analysis cross-checks), self-corrected, and LLM-judged by the authors' own harness (the headline Opus number is a 150-question sample). Every evidence-file checkmark maps to real code, but the comparison table is graded by the system's own author, so treat cross-system cells as self-reported. The engineering is real and unusually well-documented; the retrieval sophistication is below Somnigraph's, and the honest-accounting bar (published judge prompts, corrected dataset, "retrieval is not the bottleneck" caveat) is high.

**Cross-check (sharpest correction)**: The evidence file lists "Hybrid (BM25+Vec) — Reciprocal Rank Fusion ✅". The code (`internal/embedding/hybrid.go`) computes RRF scores but never uses them for the final ranking, which is cosine-priority (`cosine×100`, `+5` if in both channels); BM25-only hits use a tier-normalized score. The union-of-channels is real; the *RRF ranking mechanism* is not realized. Separately, the "0.87 LoCoMo" cell is legitimate end-to-end QA (comparable to our 85.1%, unlike R@k cells elsewhere in the table) but is self-authored and self-judged, and the headline figure is a 150-question 10% sample.
