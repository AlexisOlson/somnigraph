# cortex-engine Analysis

*Generated 2026-03-28 by Opus agent reading local clone*

---

## 1. Architecture Overview

**Repo**: https://github.com/Fozikio/cortex-engine
**Stars**: ~500+ (growing; has been on r/mcp, r/clawdbot)
**License**: MIT
**Language**: TypeScript (Node 20+), ~18,750 lines across 111 source files.
**Created**: 2026-03-10, 121 commits to date, primarily solo developer (idapixl).
**Version**: v1.1.0 (published on npm as `@fozikio/cortex-engine`)
**Description**: "Portable cognitive engine for AI agents" — persistent memory with MCP server, 57 cognitive tools, FSRS-6 scheduling, dream consolidation, graph metrics.

**Stack**:
```
MCP Server (57 tools) / REST API (optional)
    |
Cognitive Engines (memory, cognition, FSRS, graph-metrics, keywords)
    |
Storage Layer (SQLite via better-sqlite3 / Firestore)
    |
Providers (embeddings: built-in/OpenAI/Vertex/Ollama; LLMs: pluggable)
```

**Module organization** (under `src/`):
- `core/` — Types, config, store abstraction, session, embed/LLM provider interfaces
- `engines/` — Pure cognitive functions: memory (PE gating, HyDE, spreading activation, GNN aggregation, multi-anchor retrieval), cognition (8-phase dream), FSRS-6, graph-metrics (Fiedler value, PE saturation), keywords
- `stores/` — SQLite (better-sqlite3, brute-force cosine search) and Firestore backends
- `tools/` — 57 tool implementations, one file each (query, observe, believe, dream, wander, goal_set, etc.)
- `mcp/` — MCP server wiring, tool registry, plugin loader
- `namespace/` — Multi-namespace isolation with per-namespace config
- `federation/` — Cross-instance coordination
- `triggers/`, `bridges/`, `plugins/` — Extensibility plumbing

**Key design choice**: Pure cognitive functions in `engines/` — no storage imports, no side effects. All state access through injected `CortexStore` interface. This is clean and testable, though the test coverage is minimal (one test file for memory engine, one for namespace manager, one for federation client).

## 2. Memory Model

### Memory types and schema

Memories have: `name`, `definition`, `category` (one of: belief, pattern, entity, topic, value, project, insight, observation, goal), `salience` (0-1), `confidence` (0-1), `access_count`, `embedding`, `tags`, FSRS state, optional `memory_origin` ('organic', 'dream', 'abstract').

Observations are separate from memories. An observation is raw input (declarative, interrogative, speculative, or reflective) that queues for consolidation. Memories are promoted observations or LLM-synthesized abstractions.

### FSRS-6 implementation

Cortex-engine has a complete FSRS-6 implementation in `src/engines/fsrs.ts` (149 lines). Compared to Vestige's FSRS usage (see `vestige-fsrs.md`), cortex-engine's implementation is:

- **Simpler**: Uses fixed `DECAY = -0.5` instead of the FSRS-6 personalized `w[20]` exponent. This makes it technically FSRS-4/5 behavior despite using the v6 weight vector. The personalized decay exponent — FSRS-6's headline feature — is declared but unused.
- **Passive review only**: FSRS is used during dream consolidation (Phase 5) to update stability/difficulty, not for active review scheduling. The "when to review" signal that FSRS was designed for is repurposed as a "how to decay" signal during offline consolidation.
- **Composite rating**: Instead of human-provided ratings (1-4), the system derives ratings automatically from retrieval metadata: recently accessed = Good(3), not recent = Hard(2), high direct retrieval score = Easy(4), weak/indirect retrieval = penalty, contradiction edge = penalty. This is documented in `experiments/002-composite-fsrs-rating.md` as an active experiment.

The FSRS-6 weight vector (`w[0]..w[20]`) uses the standard defaults from the open-source paper. No per-user optimization. Same limitation as Vestige: the parameter optimization that makes FSRS valuable (learning from your forgetting patterns) is absent.

**Key difference from Vestige**: Vestige positions FSRS as a scheduling mechanism (when to review). Cortex-engine uses it as a **confidence signal** — retrievability modulates the composite query score (`similarity * retrievability * salienceFactor`). This is closer to Somnigraph's decay-as-scoring-signal approach, though cortex-engine uses the FSRS power-law curve while Somnigraph uses exponential decay.

### Prediction-error gating with adaptive density

The `predictionErrorGate()` function in `memory.ts` is the write-path deduplication mechanism. Three outcomes: merge (duplicate), link (related), novel (new concept).

The adaptive density mechanism is the most interesting part: it computes local density from the mean similarity of the 5 nearest neighbors, then adjusts both merge and link thresholds. Dense regions (high mean similarity) lower the merge threshold to cluster more aggressively; sparse regions raise it to be conservative. The adjustment range is roughly +/-0.075 around the base thresholds (0.85 merge, 0.50 link), clamped to [0.70, 0.95] and [0.35, 0.65] respectively.

This is a genuine information-geometric insight — the same cosine similarity means different things in dense vs. sparse embedding neighborhoods. At 0.85 similarity, a memory in a dense cluster of related concepts is likely a duplicate, while the same score in a sparse region might be the closest concept to a genuinely novel observation.

**Somnigraph comparison**: Somnigraph has no write-path deduplication. The `remember()` call in production checks 0.9 cosine similarity for exact-duplicate rejection, but there's no density adaptation and no merge/link/novel triage. The entire deduplication burden falls on the sleep pipeline's NREM phase.

### Hindsight phase (Phase 7)

The hindsight phase audits memories that have "silently hardened" — high stability, zero lapses, multiple reps, no contradiction/tension edges. These are beliefs reinforced through unchallenged repetition, which may have accumulated confidence through narrow confirmation rather than diverse evidence.

For each candidate (up to 5 per cycle, stability >= 21 days, reps >= 4, confidence >= 0.7), an LLM critically examines whether the confidence is earned. It can:
- Reduce confidence by up to 0.25
- Revise the definition (logged via belief history)
- Create a TENSION signal for follow-up

This is thoughtful design. The failure mode it targets — beliefs that harden through repetition without challenge — is real in any system with positive feedback loops. Somnigraph's feedback loop has a similar risk (highly-retrieved memories get more feedback, which increases their score, which makes them more retrieved). The utility calibration study (per-query r=0.70) showed no self-reinforcement in practice, but the failure mode remains theoretical.

**Important caveat**: The hindsight phase was added 3 days before the repo's most recent commit (commit `6693f06`). It was immediately buggy (fixed in `56dd6d8`), and Experiment 002's results table is empty. There is no evidence it has been tested beyond basic functionality.

## 3. Retrieval Pipeline

### Query flow

1. **HyDE expansion** (default on): LLM generates a hypothetical 2-3 sentence passage answering the query, then embeds that instead of the raw query. Clean implementation, standard approach from Gao et al. 2022.

2. **Vector search**: Brute-force cosine similarity over all stored embeddings (via SQLite, no ANN index). Fetches `limit * 3` candidates (default 15).

3. **Query-conditioned spreading activation**: BFS over graph edges from initial results, max depth 2. Activation decays by 0.5 per hop, weighted by edge weight. When a query embedding is provided, propagation is query-conditioned: branches whose targets are more similar to the query receive higher activation (clamped to [0.1, 1.0]). Cites Synapse (arXiv 2601.02744).

4. **GNN-style aggregation** (`aggregatedRetrieval`): For each candidate, fetch up to 5 graph neighbors, compute weighted mean embedding (0.6 self + 0.4 mean-neighbors), re-score the aggregated embedding against the query. Available as an alternative retrieval method but **not wired into the default query tool**.

5. **Multi-anchor / Borda-count retrieval** (`multiAnchorRetrieval`): LLM rephrases the query 3 ways, embeds all 4 variants, runs parallel ANN searches, aggregates via Borda count. Available but **also not wired into the default query tool**.

6. **Composite scoring**: `similarity * retrievability * salienceFactor`, where retrievability is FSRS-derived and salience is mapped from [0,1] to [0.5, 1.0].

7. **Filter and touch**: Min-score threshold (default 0.3), optional category filter. Accessed memories are touched (access_count, last_retrieval_score, last_hop_count stored for Phase 5 composite rating).

### What's actually used vs. what exists

The default `query` tool uses HyDE + vector search + spreading activation + composite scoring. The GNN aggregation and multi-anchor retrieval are exported from `memory.ts` but not used by any tool — they're available for custom implementations or future integration. This is a pattern throughout: the engine exports more capability than it wires up.

### No BM25/FTS

Despite SQLite as the backend, there is **no FTS5 or BM25 implementation**. Retrieval is vector-only (plus graph spreading activation). This is a notable gap for keyword-heavy queries.

## 4. Consolidation / Dream Cycle

The dream cycle is an 8-phase pipeline split into NREM (compression) and REM (integration). Compare to Somnigraph's 3-phase sleep pipeline (NREM classification + REM per-memory LLM decisions + archiving).

### Phase A — NREM analog

**Phase 1 (Cluster)**: Route unprocessed observations to nearest memory. Schema congruence check: sparse neighborhoods (< 2 edges) with borderline similarity keep observations episodic rather than clustering prematurely. Temporal replay order (sorted by creation time). Evidence preserved for Phase 2.

**Phase 2 (Refine)**: For memories that received clustered observations, LLM rewrites the definition incorporating new evidence. Logs belief changes. Re-embeds the refined definition. Re-validates existing typed edges against the new definition (downweights invalid edges to 30% weight). This edge re-validation is a nice touch — Somnigraph doesn't re-evaluate edges when memory content changes.

**Phase 3 (Create)**: Promote unclustered declarative/reflective observations to first-class memories. LLM infers category. Interrogative and speculative observations are marked processed but never promoted — they stay as observations.

### Phase B — REM analog

**Phase 4 (Connect)**: Pairwise edge discovery for recent memories. Two strategies:
- *Sequential* (default): O(n^2) LLM calls for pairs of recent memories (capped at 15). Each pair gets a single LLM call asking for relationship type.
- *Long-context*: Single LLM call with the full memory graph visible (up to 200 nodes + all existing edges). Can find transitive patterns, cross-domain contradictions, and causal chains. Requires a large-context model (Kimi, Gemini).

**Phase 5 (Score)**: Passive FSRS review. Only processes memories due for review (elapsed >= 80% of stability). Uses the composite rating (base + retrieval quality + contradiction penalty). Relearning memories use stricter 1-day access window.

**Phase 6 (Abstract)**: Cross-domain synthesis. Samples from 4 random categories, asks LLM for higher-level patterns. Novelty-gated (< 0.88 similarity to existing memories). Creates 'insight' memories with provenance edges back to source memories.

**Phase 7 (Hindsight)**: Audit silently hardened memories (described above).

**Phase 8 (Report)**: Narrative summary of the cycle.

### Comparison to Somnigraph's sleep pipeline

| Aspect | Cortex-engine (8 phases) | Somnigraph (3 phases) |
|--------|--------------------------|----------------------|
| Trigger | Manual (`dream` tool) or scheduled | Cron-triggered sleep script |
| Observation routing | Phase 1 clusters to nearest memory | NREM classifies each memory individually |
| Definition refinement | Phase 2 rewrites definitions | REM decides per-memory actions (rewrite, archive, split, etc.) |
| Edge discovery | Phases 4 pairwise/long-context LLM | REM creates edges; Hebbian co-retrieval strengthens |
| FSRS scheduling | Phase 5 passive review | N/A (exponential decay, feedback-modulated) |
| Abstraction | Phase 6 cross-domain synthesis | No explicit abstraction phase |
| Self-audit | Phase 7 hindsight | No equivalent |
| Edge revalidation | Phase 2 checks edges after refinement | No post-update edge validation |
| Feedback integration | None — no feedback loop | Feedback drives decay adjustment, Hebbian edges, UCB |
| LLM cost per cycle | High (O(n^2) for Phase 4, multiple per-memory calls) | Moderate (one call per memory for REM decisions) |

Cortex-engine's pipeline is more structured (more phases, clearer separation of concerns) but more expensive per cycle. Somnigraph's pipeline is simpler but has the feedback loop advantage — it learns from usage patterns, not just consolidation results.

## 5. Graph Structure

### Edge types

9 relation types: `extends`, `refines`, `contradicts`, `tensions-with`, `questions`, `supports`, `exemplifies`, `caused`, `related`. Compare to Somnigraph's 4: `support`, `contradict`, `evolve`, `derive`.

Cortex-engine has richer edge semantics — `tensions-with` (softer than contradiction), `questions` (epistemic uncertainty about a relationship), `exemplifies` (abstraction to instance). Somnigraph's edges are simpler but carry more metadata (flags like `is_contradiction`, Hebbian weight from co-retrieval PMI).

### Fiedler value as graph health metric

`graph-metrics.ts` computes the Fiedler value (algebraic connectivity) of the memory graph — the second-smallest eigenvalue of the graph Laplacian. Implemented via two-pass power iteration, O(n * iterations), with 120 iterations.

Interpretation: high Fiedler value = well-integrated knowledge graph; low = disconnected clusters (compartmentalized knowledge). Zero = disconnected graph.

This is computed during dream consolidation and reported in the cycle results. It's a **diagnostic signal**, not an operational one — nothing in the system acts on the Fiedler value. But it's a genuinely useful graph health metric that Somnigraph doesn't have.

### PE saturation detection

`detectPESaturation()` tracks the trend in prediction error over recent identity-related observations. When mean PE drops below 0.10 over 14 days with sufficient data (>= 5 observations), the system reports saturation and recommends adversarial or counterfactual prompting. This is connected to the hindsight phase conceptually but not mechanically — PE saturation is reported, hindsight acts on FSRS stability.

## 6. Comparison to Somnigraph

### What they have that we don't

1. **Prediction-error gating with adaptive density**: Density-aware deduplication at write time. Somnigraph has no write-path deduplication beyond 0.9 similarity rejection.

2. **Hindsight phase**: Proactive audit of unchallenged high-confidence memories. Somnigraph relies on the feedback loop and sleep pipeline but doesn't specifically target silently hardened beliefs.

3. **Fiedler value as graph diagnostic**: Algebraic connectivity metric for graph health. Somnigraph has no graph-level diagnostic.

4. **HyDE query expansion**: LLM-generated hypothetical document for better conceptual matching. Somnigraph uses BM25-damped IDF keyword expansion in the LoCoMo benchmark but not HyDE. (HyDE is analyzed in `research/sources/hyde.md`.)

5. **GNN-style aggregation retrieval**: Neighbor-weighted embedding re-scoring. Novel approach, though not wired into the default pipeline.

6. **Multi-anchor / Borda-count retrieval**: Multiple query formulations with consensus voting. Similar to Somnigraph's multi_query expansion method (which was found dead at 2% contribution in LoCoMo ablation).

7. **Goal-directed cognition**: Goals as memory nodes with forward prediction error biasing consolidation and exploration. Somnigraph has no explicit goal representation.

8. **Schema congruence in clustering**: Sparse neighborhoods prevent premature generalization from single observations. Somnigraph doesn't consider graph density when processing observations.

9. **Edge revalidation after refinement**: When a memory's definition changes, existing typed edges are re-checked and downweighted if no longer valid.

10. **Belief revision tracking**: Full audit trail of definition changes with reasons. Somnigraph's sleep pipeline logs decisions but doesn't maintain a structured belief revision history.

### What we have that they don't

1. **Learned reranker with ground truth**: 26-feature LightGBM trained on 1032 human-judged queries (NDCG=0.7958). Cortex-engine has no learned scoring — composite score is `similarity * retrievability * salience`, a hand-tuned formula.

2. **BM25/FTS5**: Hybrid BM25 + vector retrieval with RRF fusion. Cortex-engine is vector-only despite using SQLite.

3. **Retrieval feedback loop**: `recall_feedback()` → EWMA aggregation → UCB exploration → decay adjustment → Hebbian edge strengthening. Cortex-engine has no mechanism for the user/agent to say which results were useful.

4. **Hebbian co-retrieval strengthening**: PMI-based edge weight increases from co-retrieval patterns. Cortex-engine creates edges via LLM during dream but doesn't strengthen them from usage.

5. **UCB exploration bonus**: Bayesian exploration for under-retrieved memories. Cortex-engine's `wander` tool has epistemic scoring for exploration but it's a separate tool, not integrated into the main query path.

6. **Evaluation infrastructure**: Ground truth pipeline, LLM-as-judge, LoCoMo benchmark (85.1% overall accuracy, 95.4% R@10 with graph augmentation). Cortex-engine has two early-stage experiments with incomplete results tables.

7. **Theme normalization**: Canonical vocabulary for topics and categories. Cortex-engine uses free-form tags with keyword extraction.

8. **Multi-channel retrieval**: BM25 + vector + theme overlap + PPR graph expansion. Cortex-engine is vector + spreading activation only.

9. **PPR graph expansion**: Personalized PageRank for global graph context. Cortex-engine has BFS spreading activation (local, depth 2).

### Architectural trade-offs

| Dimension | Cortex-engine | Somnigraph |
|-----------|--------------|------------|
| Language | TypeScript (npm package) | Python (MCP server) |
| Storage | SQLite (better-sqlite3) / Firestore | SQLite + sqlite-vec + FTS5 |
| Vector search | Brute-force cosine in JS | sqlite-vec (optimized) |
| Text search | None | FTS5 BM25 |
| Retrieval | Vector + BFS spreading activation | BM25 + vector + theme + PPR + learned reranker |
| Write-path dedup | PE gating with adaptive density | 0.9 similarity rejection |
| Decay model | FSRS-6 (passive review during dream) | Exponential (per-memory rate, feedback-modulated) |
| Consolidation | 8-phase dream (manual/scheduled) | 3-phase sleep (cron) |
| Feedback | None | recall_feedback → EWMA + UCB + Hebbian |
| Graph edges | 9 types, LLM-discovered during dream | 4 types, LLM + Hebbian co-retrieval + manual |
| Graph metrics | Fiedler value, PE saturation | None (graph features in reranker) |
| Multi-user | Namespace isolation, federation | Single-user |
| Evaluation | 2 early experiments | LoCoMo benchmark, ground truth, reranker NDCG |
| Deployment | npm package, MCP + REST | MCP server, pip-installable |

## 7. Worth Adopting?

### Prediction-error gating with adaptive density

**Maybe, but low priority.** The adaptive density mechanism is a genuine insight — cosine similarity thresholds should vary by local embedding density. However, Somnigraph's write path is human-curated (`remember()` is called by the agent with user review via `review_pending()`), so false-positive deduplication is caught by human oversight. The density adaptation would matter more if Somnigraph had automated ingestion. Worth noting as a technique if write automation is ever added.

### FSRS-6 confidence hardening

**No.** Cortex-engine's FSRS implementation is less complete than Vestige's (no personalized decay exponent, no parameter optimization). The vestige-fsrs analysis already concluded that FSRS scheduling is irrelevant for on-demand retrieval systems: "Our Phase 14 experiment confirmed this: power-function decay and exponential decay produce identical Spearman correlations (0.158) with actual utility at our timescale." Cortex-engine's repurposing of FSRS as a confidence signal is interesting conceptually but doesn't add anything beyond what Somnigraph already gets from feedback-modulated exponential decay.

### Fiedler value as graph diagnostic

**Yes, as a lightweight diagnostic.** The implementation is ~150 lines, has no dependencies, and provides a genuinely useful metric: "is the knowledge graph well-connected or fragmenting into clusters?" For Somnigraph, this could inform the sleep pipeline — if Fiedler value drops significantly between cycles, it might indicate that recent memories aren't being integrated with the existing graph. The implementation scales to ~5,000 nodes (Somnigraph currently has ~600 memories), and the two-pass power iteration is O(n * 120 iterations). **Effort: low. Impact: low-medium (diagnostic, not operational).**

### Hindsight review

**Watch, don't adopt.** The concept is sound — proactively audit unchallenged beliefs to prevent ossification. But:
- Somnigraph's utility calibration study (per-query r=0.70) found no self-reinforcement in practice
- The hindsight phase has zero evidence of effectiveness (empty experiment results)
- Somnigraph's feedback loop provides external signal that cortex-engine lacks
- The sleep pipeline already re-evaluates memories via LLM during REM

If Somnigraph's feedback loop were ever found to produce self-reinforcement (the utility calibration study said no, but the coverage was limited), hindsight-style auditing would become relevant.

### Edge revalidation after refinement

**Worth considering.** When a memory's definition changes during sleep, existing edges may become invalid. Cortex-engine re-checks edges and downweights invalid ones. Somnigraph currently doesn't do this — the REM phase can modify individual memories without checking downstream edge consistency. Effort: medium (requires LLM calls per edge per modified memory). Impact: depends on how often memory definitions change enough to invalidate edges.

## 8. Worth Watching

- **Composite FSRS rating results** (Experiment 002): If the retrieval-quality-informed FSRS rating shows measurable improvement in memory health metrics, it would validate that retrieval signals can improve consolidation quality. Currently no results.
- **Long-context dream strategy**: Using a single large LLM call for edge discovery (seeing the full graph) is a better architecture than pairwise calls. If results show improved edge quality, it's relevant to Somnigraph's REM phase design.
- **Multi-namespace federation**: Cross-instance coordination could inform multi-agent memory sharing if Somnigraph ever needs it.
- **Community adoption**: At v1.1.0 with 57 tools, this is the most feature-rich MCP memory system in the survey. Whether the feature breadth produces better outcomes than Somnigraph's focused depth is an open question.

## 9. Key Claims

| Claim | Evidence |
|-------|----------|
| "57 cognitive tools" | **Verified**: 57 tool files in `src/tools/`, all wired into MCP server. |
| "Two-phase dream consolidation modeled on biological sleep" | **Partially verified**: 8 phases implemented in `cognition.ts` (~700 lines). The biological analogy (NREM compression + REM integration) is reasonable. Phase effectiveness is untested. |
| "Neuroscience-grounded retrieval — GNN neighborhood aggregation, query-conditioned spreading activation, multi-anchor Thousand Brains voting" | **Code exists but overstated**: GNN aggregation and multi-anchor retrieval are implemented but not wired into the default query tool. Only spreading activation is active by default. "Thousand Brains voting" is a reference to Numenta's framework that doesn't map cleanly to Borda count over LLM-rephrased queries. |
| "Information geometry — locally-adaptive clustering thresholds that respect embedding space curvature" | **Partially verified**: The adaptive density threshold in PE gating adjusts based on local k-NN density. "Information geometry" is a stretch — there's no Riemannian metric or Fisher information. It's density-adaptive thresholding, which is useful but not information geometry. |
| "FSRS-6 spaced repetition" | **Partially verified**: The weight vector and core formulas are FSRS-6, but the personalized decay exponent (`w[20]`) is declared without being used. Effectively FSRS-4/5 behavior. No parameter optimization from review data. |
| "Fiedler value measures knowledge integration" | **Verified**: Clean implementation of algebraic connectivity via power iteration. Computed during dream, reported in results. Not acted upon. |
| "PE saturation detection prevents identity model ossification" | **Code exists**: `detectPESaturation()` computes the metric and recommends intervention. No evidence it triggers anything automatically. |
| "Works with Claude Code, Cursor, Windsurf, or any MCP-compatible client" | **Plausible**: Standard MCP server implementation with `@modelcontextprotocol/sdk`. |

## 10. Relevance to Somnigraph

**Medium.** Cortex-engine shares several design goals with Somnigraph (persistent memory, graph-based knowledge, consolidation cycles, biological inspiration) but takes a significantly different approach: broader tool surface (57 vs. Somnigraph's ~10 MCP tools), no learned reranker, no feedback loop, no BM25, FSRS instead of exponential decay. The breadth-vs-depth trade-off is clear: cortex-engine offers many cognitive tools with thin implementations behind each; Somnigraph has fewer tools with deep, evaluated implementations.

The most interesting ideas for Somnigraph are the **Fiedler value diagnostic** (low effort, genuinely useful graph health metric), the **adaptive density deduplication** (useful if write automation is ever added), and the **edge revalidation after refinement** (addresses a real gap in the sleep pipeline). The FSRS integration adds nothing beyond what the vestige-fsrs analysis already covered. The hindsight phase is conceptually sound but has zero evidence of effectiveness.

The system is young (18 days old at analysis time, 121 commits) and rapidly evolving. The feature count is impressive but the evaluation depth is minimal — two experiments with incomplete results, no benchmark comparisons, no ground truth. Whether the broad cognitive toolset produces better memory outcomes than Somnigraph's evaluated, feedback-driven approach is unknown and likely unknowable without comparative benchmarking.
