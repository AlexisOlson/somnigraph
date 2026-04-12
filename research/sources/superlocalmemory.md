# SuperLocalMemory V3.3 — Biologically-Inspired Zero-LLM Agent Memory

*Generated 2026-04-11 by Opus agent reading arXiv:2604.04514*

---

## Paper Overview

**Paper**: Bhardwaj, V.P. "SuperLocalMemory V3.3: The Living Brain — Biologically-Inspired Forgetting, Cognitive Quantization, and Multi-Channel Retrieval for Zero-LLM Agent Memory Systems." arXiv:2604.04514, April 2026.
**Author**: Varun Pratap Bhardwaj, Senior Manager and Solution Architect at Accenture, India. Independent researcher. ORCID: 0009-0002-8726-4289.
**Code**: https://github.com/qualixar/superlocalmemory (npm: `superlocalmemory`, PyPI: `superlocalmemory`). Elastic License 2.0.
**Prior work**: Third paper in a trilogy. Paper 1 (arXiv:2602.22302): Bayesian trust defense. Paper 2 (arXiv:2603.14588): information-geometric foundations, Fisher-Rao geodesic, 4-channel retrieval, 74.8% LoCoMo Mode A.

**Problem addressed**: AI coding agents (Claude Code, Cursor, etc.) have session amnesia — every session starts from scratch. Existing memory systems (Mem0, Letta, Zep) are cloud-dependent, single-channel, and lack cognitive processes (forgetting, consolidation, parameterization).

**Core approach**: Local-first memory system implementing the full cognitive memory taxonomy (sensory through long-term implicit). SQLite + sqlite-vec storage. Seven parallel retrieval channels fused via weighted RRF + ONNX cross-encoder reranking. Ebbinghaus forgetting with lifecycle-aware embedding quantization. Memory parameterization into soft prompts. Zero-friction auto-cognitive hooks for Claude Code.

**Key claims**:
1. FRQAD achieves 100% precision at distinguishing full-fidelity from quantized embeddings (vs. 85.6% cosine, 70.7% Fisher-Rao)
2. Ebbinghaus forgetting provides 6.7x discriminative power between access groups over 30 simulated days
3. 70.4% on LoCoMo in zero-LLM Mode A (7-channel retrieval)
4. First system implementing Long-Term Implicit memory via soft prompt generation
5. Zero-configuration auto-cognitive pipeline via Claude Code hooks

---

## Architecture

### Storage & Schema

Three local SQLite databases:
- **memory.db**: Core fact store, knowledge graph, embeddings, temporal data, quantized embeddings, soft prompts, forgetting schedules. Uses sqlite-vec for vector operations.
- **learning.db**: Behavioral patterns, feedback signals. GDPR-erasable via `slm learning reset`.
- **code_graph.db**: Code knowledge graph (nodes, edges, communities, flows) via rustworkx. Tree-sitter for multi-language AST parsing.

17 packages, 215 source modules, 60 MCP tools (including 22 code graph tools), 3,000+ tests.

### Memory Types

Maps to Li et al.'s cognitive memory taxonomy:
1. **Sensory** — handled natively by LLMs (prompt tokens)
2. **Short-Term (STM)** — context window / KV cache
3. **Long-Term Explicit** — facts stored in SQLite with embeddings, knowledge graph edges, temporal metadata, emotional salience
4. **Long-Term Implicit** — consolidated patterns converted to natural language soft prompts injected at session start (Section 7). This is the novel tier — no other system implements it.

### Write Path

**Encoding pipeline**: Fact extraction -> entity resolution -> entropy gate -> emotional tagging -> graph construction -> consolidation decision (ADD/UPDATE/SUPERSEDE/NOOP).

Immediate SQLite write (~0.1s) to `pending.db`, then asynchronous processing (embedding generation, graph construction). Pending memories retried on engine initialization.

Embeddings: sentence-transformers in an isolated subprocess (nomic-embed-text-v1.5, 768d). Auto-killed after 2 minutes idle. Main process stays torch-free at 63.3 MB RSS.

### Retrieval

Seven parallel channels fused via **weighted Reciprocal Rank Fusion** (k=15, optimized for 50-200 fact candidate pools):

| # | Channel | Source | Weight | What it retrieves |
|---|---------|--------|--------|-------------------|
| 1 | Semantic | sqlite-vec KNN (float32/768d) | 1.2 | Meaning-similar facts |
| 2 | BM25 | FTS5 keyword index | 1.0 | Exact term matches |
| 3 | Entity Graph | Knowledge graph traversal | 1.0 | Entity-connected facts |
| 4 | Temporal | Bi-temporal timestamps | 1.0 | Recently relevant facts |
| 5 | Spreading Activation | SYNAPSE-based energy propagation | 1.0 | Causally-connected facts |
| 6 | Consolidation | CCQ gist blocks | 1.0 | Compressed knowledge |
| 7 | Hopfield | Modern Hopfield network (softmax update) | 0.8 | Pattern-completed associations |

After RRF fusion: **ONNX cross-encoder reranking** (ms-marco-MiniLM-L-6-v2, ~90MB) produces final relevance scores.

**Cross-channel intersection** (new in V3.3): For multi-hop queries, entity-channel and temporal-channel results are intersected before RRF fusion to prevent noise dilution of precise entity-temporal matches. This is the main driver of the +23.8pp multi-hop improvement.

**Session diversity enforcement**: Prevents returning the same facts repeatedly within a session. Contributes to adversarial reasoning improvement (+12.7pp).

### Consolidation / Processing

**Memory parameterization pipeline** (Section 7):
1. Cluster related episodic memories, extract semantic patterns
2. Filter by confidence (>0.7, minimum 5 observations): `confidence(p) = min(evidence/10, 1.0) * |rate(p) - 0.5| * 2`
3. Generate natural language soft prompts from structured pattern fields (template-based, zero LLM cost)
4. Inject at session start via SessionStart hook, capped at 1,500 tokens

This implements Long-Term Implicit memory — the agent's behavior is shaped by past experience without explicit retrieval. Unlike LoRA, works with any API-based LLM.

### Lifecycle Management

**Ebbinghaus Adaptive Forgetting**:

Memory strength is a four-factor function:
```
S(m) = max(S_min, alpha * log(1 + access_count) + beta * importance + gamma_c * confirmations + delta * emotional_salience)
```

Retention: `R(m, t) = exp(-t / S(m))`

Five lifecycle states based on retention R:
- Active (R > 0.8) -> 32-bit embeddings
- Warm (0.5 < R <= 0.8) -> 8-bit embeddings
- Cold (0.2 < R <= 0.5) -> 4-bit embeddings
- Archive (0.05 < R <= 0.2) -> 2-bit embeddings
- Forgotten (R <= 0.05) -> deleted

**Forgetting-quantization coupling**: As memories fade, their embeddings are progressively quantized. This is self-consistent with FRQAD — quantized embeddings automatically receive higher effective variance, producing lower similarity scores. Faded memories naturally rank lower without explicit re-weighting.

**Trust-weighted forgetting** (from Paper 1): Low-trust sources decay 3x faster (kappa=2.0 sensitivity). `lambda_eff = lambda * (1 + kappa * (1 - trust_score))`.

**Fokker-Planck integration**: Forgetting drift term added to Riemannian Langevin dynamics (from Paper 2). Provable convergence to unique stationary distribution (Theorem 5.3).

### Operating Modes

- **Mode A (Local Guardian)**: Zero-LLM. sentence-transformers embeddings, ONNX cross-encoder. 70.4% LoCoMo.
- **Mode B (Smart Local)**: Adds Ollama for LLM synthesis, data stays local.
- **Mode C (Full Power)**: Cloud LLM for maximum quality. 87.7% LoCoMo (Paper 2 result).

### Other Notable Components

**Code Knowledge Graph**: Tree-sitter AST parsing + rustworkx in-memory graph. Links code entities (functions, classes, imports) to related memories. 22 dedicated MCP tools. Bidirectional event bus between code graph and memory.

**Daemon serve mode**: Warm MemoryEngine on `127.0.0.1:8767`, 30-minute idle auto-shutdown. 32x cold-start speedup (19s -> 0.6s for CLI recall).

**Zero-friction auto-cognitive pipeline**: `npm install -g superlocalmemory` auto-configures Claude Code hooks (SessionStart, PostToolUse, Stop). All hooks fail silently. No PreToolUse gates. Opt-out via `slm hooks remove`.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| FRQAD 100% precision on mixed-precision preference | 18,840 query-fact pairs (943 facts, 768d nomic embeddings), each with float32 and 4-bit versions. FRQAD correctly prefers f32 in all 18,840 cases vs. cosine 85.6%. | **Solid but narrow.** The benchmark only tests f32 vs. 4-bit discrimination. In practice, all V3.3 embeddings are currently stored at float32 — FRQAD's advantage only emerges when the EAP scheduler actually promotes mixed-precision storage. The paper acknowledges this: "In the current release, all embeddings are stored at float32." |
| 6.7x discriminative power in forgetting | 170 facts over 30 simulated days, 3 access patterns. Hot S=11.28 vs. Cold S=1.69. | **Reasonable simulation** but synthetic. The strength formula parameters (alpha, beta, gamma_c, delta) are hand-set, not learned. No comparison against simpler alternatives (e.g., plain recency). The 6.7x figure measures strength, not retrieval quality. |
| 70.4% on LoCoMo Mode A (zero-LLM) | 2 of 10 conversations, 304 QA pairs, 1,585 facts. GPT-5.4-mini judge (Likert 1-5, >=4 threshold). 5-turn chunks for ingestion. Best of 5 rounds (R3). | **Partially convincing.** Only 2/10 conversations evaluated — small sample. GPT-5.4-mini judge is lenient (the paper doesn't characterize this but we know from our own LoCoMo work that LLM judges accept ~63% of vague wrong answers). "Best of 5 rounds" selects the best run, not average. The baseline-to-R3 gap (62.8% -> 70.4%) is substantial but cherry-picked. Paper 2 reported 74.8% Mode A — the 4.4pp regression is acknowledged and attributed to 7-channel fusion complexity on single-hop (-14.9pp). |
| Surpasses Paper 2 on adversarial (+6.1pp) | Table 8: 76.1% vs. ~70% Paper 2. | **Credible.** Session diversity enforcement is a reasonable mechanism for adversarial improvement. |
| +23.8pp multi-hop improvement | Table 8: 49.2% vs. 25.4% baseline. | **Large and plausible.** Cross-channel intersection is a targeted mechanism. But 49.2% is still modest — our L5b achieves 88.8% R@10 multi-hop (different metric, not directly comparable). |
| First system implementing Long-Term Implicit | Soft prompt generation from consolidated patterns. | **Technically true** for local agent memory systems, but the claim is stronger than the mechanism. Template-based soft prompts are preference configuration, not learned implicit skills. The paper's own limitations section acknowledges "less powerful than LoRA." |
| Zero prior art for FRQAD | "Systematic literature search found zero prior work combining information geometry with vector quantization for similarity retrieval." | **Plausible** but unverifiable. The combination is niche enough that absence of prior work is believable. |

---

## Relevance to Somnigraph

### What SuperLocalMemory does that Somnigraph doesn't

1. **Embedding quantization with lifecycle coupling**: Progressive precision reduction (32->8->4->2 bit) as memories fade, with a quantization-aware distance metric. Somnigraph uses uniform float32 for all embeddings regardless of age/importance. This is a genuine storage efficiency mechanism — 2-bit is 192x compression at 0.801 cosine fidelity.

2. **Spreading activation channel**: SYNAPSE-based energy propagation for causal connectivity. Somnigraph's PPR graph expansion is conceptually similar but operates differently — PPR is a random-walk model, spreading activation is an energy-propagation model with fan-out and lateral inhibition.

3. **Hopfield associative memory**: Modern continuous Hopfield network for pattern completion from partial cues. Novel retrieval modality that Somnigraph doesn't have. Unclear how much it contributes (weight 0.8, lowest of all channels).

4. **Memory parameterization / soft prompts**: Converting consolidated patterns into behavioral configuration injected at session start. Somnigraph's `startup_load` serves a related purpose but retrieves explicit memories, not distilled behavioral patterns.

5. **Code knowledge graph**: Tree-sitter AST parsing linked to memory. Domain-specific but potentially powerful for coding agents.

6. **Trust-weighted forgetting**: Bayesian trust scores modulate decay rate. Somnigraph has no source-trust mechanism.

7. **Auto-cognitive hooks**: Zero-config lifecycle automation. Somnigraph relies on CLAUDE.md instructions for the agent to use tools correctly — a fundamentally different approach (guidance vs. automation).

### What Somnigraph does better

1. **Learned reranker**: LightGBM with 26 production features trained on 1,032 real queries (NDCG=0.7958). SLM uses ONNX cross-encoder (ms-marco-MiniLM-L-6-v2) — a generic model, not trained on memory-specific signals. The cross-encoder doesn't see access patterns, feedback history, graph features, or any of the memory-specific metadata that makes retrieval context-aware.

2. **Explicit feedback loop**: User utility ratings drive reranker training and UCB exploration. SLM's learning pipeline mines behavioral patterns but has no direct feedback mechanism for retrieval quality.

3. **Retrieval benchmarks**: Somnigraph L5b R@10=95.4%, MRR=0.882 on LoCoMo (10 conversations). SLM evaluated on 2/10 conversations with a different metric (end-to-end QA accuracy). The retrieval quality comparison favors Somnigraph significantly.

4. **End-to-end QA**: 85.1% LoCoMo (all 10 conversations, corrected GT). SLM's 70.4% Mode A is zero-LLM, which is impressive for the constraint, but the absolute quality gap is 14.7pp. SLM Mode C (87.7%) is closer but uses cloud LLM.

5. **Graph-conditioned retrieval via PPR**: Personalized PageRank over the full memory graph with synthetic vocabulary bridges. SLM has an entity graph channel but it's one of seven independent channels without the synthetic node mechanism that closes vocabulary gaps.

6. **Evaluation rigor**: Somnigraph evaluates on all 10 LoCoMo conversations, uses Opus judge calibration, has corrected GT, runs multiple retrieval metrics (R@10, MRR, R@20, NDCG). SLM evaluates on 2/10 conversations, uses GPT-5.4-mini judge, reports best-of-5-rounds.

7. **Offline LLM consolidation**: Somnigraph's sleep pipeline (relationship detection, contradiction classification, temporal evolution, summary generation) uses LLM judgment for consolidation quality. SLM's consolidation is pattern-based without LLM involvement (which is consistent with its zero-LLM philosophy but limits quality).

---

## Worth Stealing (ranked)

### 1. Cross-channel intersection for multi-hop (low effort)

SLM's cross-channel intersection — intersecting entity and temporal channel results before fusion for multi-hop queries — is a simple, targeted mechanism that produced +23.8pp multi-hop improvement. Somnigraph could implement a similar pre-fusion filtering: when a query is classified as multi-hop, require that RRF candidates appear in both semantic and graph results before scoring. This is cheaper than our current approach of letting the reranker sort it out post-fusion.

**Caveat**: Our L5b already achieves 88.8% multi-hop R@10 via synthetic vocabulary bridges. The marginal value may be small. Worth testing as an ablation.

### 2. Forgetting-quantization coupling concept (medium effort)

The idea of coupling memory decay with embedding precision is elegant and storage-efficient. For Somnigraph, the mechanism would be different (we don't need FRQAD since we don't mix precisions in a single search), but the principle of progressive compression as memories age is sound. Could be implemented as: archived memories get lower-dimensional embeddings via PCA projection, reducing sqlite-vec index size.

**Caveat**: Storage is not currently a bottleneck. This is a "someday" optimization, not a current need.

### 3. Spreading activation as retrieval signal (medium-high effort)

SYNAPSE-based energy propagation is conceptually distinct from PPR. Where PPR models steady-state importance, spreading activation models temporal energy dynamics with fan-out limits and lateral inhibition. Adding activation energy as a reranker feature (rather than a full retrieval channel) could capture causal connectivity signals that PPR misses.

**Caveat**: Our PPR already captures graph-based relevance. The marginal signal from a second graph-walk model is unclear.

### 4. Session diversity enforcement (low effort)

Tracking which memories have been retrieved within a session and penalizing re-retrieval. Simple to implement as a reranker feature (`session_recency` already exists in our 26-feature model). SLM reports +12.7pp on adversarial from this mechanism.

**Note**: We already have `session_recency` as a feature. The question is whether it's weighted correctly. Worth checking feature importance.

---

## Not Useful For Us

**FRQAD / TurboQuant**: The quantization-aware distance metric solves a problem we don't have. Somnigraph stores all embeddings at full precision and uses OpenAI's API for embedding generation, not local sentence-transformers. The mixed-precision search scenario requires local embedding control. Even if we moved to local embeddings, the storage savings aren't worth the retrieval quality risk at our scale.

**Hopfield associative memory**: The modern Hopfield network channel (weight 0.8, lowest) is theoretically interesting but adds complexity for unclear benefit. SLM doesn't ablate individual channels, so the marginal contribution is unknown. Pattern completion from partial cues is better handled by query expansion (which we already do).

**Auto-cognitive hooks**: The zero-friction pipeline is well-engineered for adoption but philosophically opposed to Somnigraph's approach. Somnigraph gives the agent judgment about when to remember/recall via CLAUDE.md guidance. Automatic observation and storage removes that judgment. For a personal memory system, user/agent control over what gets stored matters more than friction reduction.

**Soft prompt parameterization**: Template-based soft prompts are a form of rule extraction, not genuine implicit learning. Somnigraph's `startup_load` with category-aware retrieval achieves the behavioral-configuration goal more flexibly. The "Long-Term Implicit" framing oversells what is essentially template-based preference injection.

**Code knowledge graph**: Domain-specific to coding agents. Somnigraph is a general memory system. The tree-sitter + rustworkx architecture is well-built but out of scope.

**Trust framework**: Multi-agent trust defense (from Paper 1) addresses a threat model Somnigraph doesn't face. Somnigraph is single-user, single-agent. There's no untrusted source to distrust.

---

## Connections

- **SYNAPSE** (arXiv:2601.02744): SLM implements SYNAPSE's spreading activation as Channel 5. Somnigraph's PPR expansion is a related but distinct graph-walk approach.
- **Mem0** (research/sources/mem0-paper.md): SLM positions against Mem0 (64.2% LoCoMo, cloud-only, single-channel). The comparison is fair.
- **Zep/Graphiti** (research/sources/zep-paper.md, zep-repo.md): SLM compares against Zep v3 (85.2% LoCoMo, cloud). Zep's bi-temporal KG + triple-modality search is the closest architectural parallel to SLM's multi-channel approach.
- **Letta/MemGPT** (research/sources/letta.md): SLM's Mode A is the anti-Letta — no LLM for any memory operation. Mode C converges toward Letta's architecture.
- **LoCoMo** (research/sources/locomo.md): Shared benchmark. SLM evaluates on 2/10 conversations; Somnigraph evaluates on all 10. Direct score comparison requires normalization.
- **Vestige/FSRS** (research/sources/vestige-fsrs.md): FSRS implements spaced repetition with a learned forgetting curve. SLM's Ebbinghaus formula is hand-tuned. FSRS's learned parameters would likely outperform SLM's fixed coefficients.
- **CLS theory** (McClelland et al. 1995): SLM's consolidation-then-quantization pipeline directly implements the hippocampal-to-neocortical transfer hypothesis. Somnigraph's sleep pipeline is also CLS-inspired.
- **Context-as-Memory** (arXiv:2506.03141): Validates SLM's sliding-window recency approach. Non-contiguous retrieval outperforms contiguous.

---

## Summary Assessment

**Relevance**: Medium. SLM is an ambitious system that tackles many problems simultaneously (forgetting, quantization, multi-channel retrieval, parameterization, code graphs, trust, compliance). The breadth is impressive but means no single contribution goes as deep as Somnigraph's focused work on retrieval quality and reranker learning.

**Novelty**: The forgetting-quantization coupling is the most original contribution — the insight that progressive embedding compression is self-consistent with information-geometric distance metrics is elegant and (per the paper's claim) genuinely novel. FRQAD itself is well-motivated mathematically. The 7-channel retrieval architecture is thorough but not fundamentally different from multi-signal fusion approaches.

**Rigor concerns**:
- LoCoMo evaluation on 2/10 conversations with best-of-5 selection undermines the headline 70.4% claim. Average across all rounds would be more informative.
- No ablation of individual channels — the marginal contribution of Hopfield, spreading activation, and consolidation channels is unknown.
- Forgetting parameters are hand-tuned without comparison to simpler baselines or learned alternatives.
- The paper presents many mathematical theorems (Fokker-Planck convergence, monotonic degradation) but the practical impact of these guarantees on retrieval quality is not measured.
- "Zero prior art" claims are inherently hard to verify.

**Engineering quality**: High. 3,000+ tests, 5,000+ monthly downloads, daemon serve mode with 32x speedup, subprocess architecture keeping main process at 63 MB. The production engineering is serious.

**Bottom line**: SLM V3.3 is the most ambitious local-first agent memory system in the research corpus. It covers more of the cognitive memory taxonomy than any other system. But ambition and coverage aren't the same as depth — the retrieval quality (70.4% zero-LLM) lags systems with learned rerankers and feedback loops (Somnigraph 85.1%, Zep 85.2%). The forgetting-quantization coupling is the idea most worth understanding; the multi-channel retrieval architecture validates the general approach of fusing heterogeneous signals but doesn't advance it beyond what weighted RRF + cross-encoder already provides. The paper's value is strongest as a design reference for someone building a complete local memory system — it shows what the full stack looks like, even if individual components aren't best-in-class.
