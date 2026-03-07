# Vestige + FSRS (Free Spaced Repetition Scheduler) -- Source Analysis

*Phase 14, 2026-03-01. Analysis of samvallad33/vestige and the open-spaced-repetition/FSRS ecosystem.*

## 1. Architecture Overview

**Vestige** is an open-source AI agent memory system built in Rust (77,840+ lines) by Sam Valladares. It applies FSRS-6 spaced repetition to AI memory management — "a hippocampus, not just a hard drive." Submitted to the MCP AI Agents Hackathon. 398 GitHub stars, AGPL-3.0 license.

**FSRS** (Free Spaced Repetition Scheduler) is the underlying algorithm, created by Jarrett Ye (L-M-Sherlock) at MaiMemo Inc. It replaces Anki's 38-year-old SM-2 algorithm with machine-learning-optimized scheduling based on the DSR (Difficulty, Stability, Retrievability) model. Built into Anki since November 2023. Published at KDD 2022 and IEEE TKDE 2023.

**Vestige stack:**
```
SvelteKit Dashboard (Three.js 3D visualization)
    |
Axum HTTP + WebSocket Server (port 3927)
    |
MCP Server (stdio JSON-RPC 2.0)
    |
Cognitive Engine (29 modules)
    |
Storage Layer (SQLite + FTS5 + USearch HNSW)
```

**FSRS ecosystem:**
- GitHub org: open-spaced-repetition (15+ repos)
- Core algorithm spec: free-spaced-repetition-scheduler
- Implementations in 16+ languages (Rust, Python, TypeScript, Go, Swift, etc.)
- Benchmark dataset: 349 million reviews across 10,000 users
- Adopted by: Anki, RemNote, Mochi Cards, Obsidian (plugin), Logseq, others

## 2. Memory Type Implementation

**Vestige:** 21 MCP tools including `smart_ingest` (CREATE/UPDATE/SUPERSEDE), `search` (7-stage pipeline), `dream` (consolidation), `memory_health`, `explore_connections`. Each memory has:
- Storage strength (encoding quality) vs. retrieval strength (accessibility) — per Bjork & Bjork (1992) dual-strength model
- FSRS stability, difficulty, and retrievability parameters
- Prediction error gating: compares incoming info against existing memories; redundant data merged, contradictory data supersedes, novel info stored with high priority

**FSRS memory model (DSR):**
- **Retrievability (R):** Probability of recall at time t. Range [0, 1].
- **Stability (S):** Time (in days) for R to drop from 100% to 90%. Higher = slower forgetting.
- **Difficulty (D):** How hard a memory is to maintain. Affects stability growth rate. Range [1, 10].

## 3. Retrieval Mechanism

**Vestige's 7-stage search pipeline:**
1. HyDE (Hypothetical Document Embedding) query expansion → 3-5 semantic variants
2. Keyword search (FTS5)
3. Semantic search (USearch HNSW, Nomic Embed Text v1.5 768d → 256d)
4. Reranking (Jina Reranker v1 Turbo, 38M params)
5. Temporal weighting
6. Competition
7. Spreading activation (Collins & Loftus 1975, graph-based)

**FSRS scheduling (not retrieval per se):** Determines when to review, not what to retrieve. The interval calculation given desired retention r:
```
I(r, S) = (S / FACTOR) * (r^(1/DECAY) - 1)
```

## 4. Standout Feature

**The power-function forgetting curve and its 130-year lineage.**

FSRS's central contribution is replacing exponential decay with a power function, grounded in extensive empirical evidence:

**FSRS-6 (current):**
```
R(t, S) = (1 + FACTOR * t/S)^(-w20)
```
where `FACTOR = 0.9^(1/w20) - 1`, `w20` is optimizable (0.1-0.8), and the constraint `R(S, S) = 0.9` holds by construction.

**Earlier versions:**
- FSRS v4/v5: `R(t, S) = (1 + 19/81 * t/S)^(-0.5)` (fixed decay)
- FSRS v3: `R(t, S) = 0.9^(t/S)` (exponential — abandoned)

**Why power functions beat exponentials:** Wixted & Ebbesen (1997) established in "Genuine power curves in forgetting" (Memory & Cognition) that forgetting follows power functions even at the individual level, not just as averaging artifacts. The theoretical explanation: when memories of different complexity superpose, the aggregate forgetting curve is better approximated by a negative power function. Power functions have **heavier tails** — they drop rapidly early but retain higher residual probability at long delays.

**The "almost forgotten" principle:** Counterintuitively, reviewing material at low retrievability produces maximum stability gain. FSRS parametrizes this via `e^(w10*(1-R)) - 1` in the stability update formula. This is consistent with the desirable difficulty literature (Bjork 1994).

**Historical arc (Ebbinghaus → SM-2 → FSRS):**
- 1885: Ebbinghaus publishes forgetting curve research using nonsense syllables
- 1972: Leitner creates first practical spaced repetition system (card boxes)
- 1987: Wozniak's SM-2 algorithm — first computer-based scheduler. Default in Anki for 35+ years.
- 2022: FSRS v3 (first widely-adopted post-SM-2 algorithm). Provoked by a Reddit comment: "nobody actually implements" academic scheduling algorithms.
- 2023: FSRS v4 switches from exponential to power-function decay
- 2023: Anki integrates FSRS (opt-in). SM-2's 38-year reign ends.
- 2025: FSRS-6 adds personalized decay exponent (w20)

**Scale:** FSRS parameters are optimized against 220 million review logs. Default weights derived from ~10,000 users. In benchmarks (349M reviews, 10K collections), FSRS-6 beats SM-2 on 99.6% of collections and requires 20-30% fewer reviews for equivalent retention.

## 5. Other Notable Features

1. **Vestige's prediction error gating.** A "hippocampal bouncer" that compares incoming information against existing memories. Redundant data is merged, contradictory data supersedes, novel information is stored with high priority. Based on Frey & Morris (1997) synaptic tagging. Our Phase 14 Experiment 5 tested this concept (Spearman -0.1952 — negative correlation, suggesting closer memories are more useful, not less).

2. **FSRS parameter optimization.** Uses gradient descent with binary cross-entropy loss. Each review is treated as a binary classification event (recall vs. lapse). 21 parameters optimized jointly. This is substantially more principled than heuristic parameter tuning — the algorithm literally learns from your forgetting patterns.

3. **Vestige's dreaming module.** Sleep-consolidation analog: replays recent memories, discovers hidden connections, synthesizes insights. Triggered after 6 hours of staleness or 2 hours of active use. Similar in concept to our NREM/REM sleep pipeline.

4. **FSRS's stability after recall formula:**
```
S'_r = S * (e^w8 * (11-D) * S^(-w9) * (e^(w10*(1-R))-1) * mods + 1)
```
Key terms: `(11-D)` penalizes difficult memories, `S^(-w9)` creates stability saturation (diminishing returns from repeated review), `(e^(w10*(1-R))-1)` rewards reviewing near-forgotten material. This is the most principled stability update function we've seen.

5. **SM-2's persistence and limitations.** SM-2 (1987) uses a simple Ease Factor: `I(n) = I(n-1) * EF` with `EF' = EF + 0.1 - (5-q)(0.08 + (5-q)*0.02)`. The minimum EF of 1.3 creates "ease hell" where cards get stuck in tight review loops. FSRS eliminates this via difficulty mean reversion toward a global default. SM-2 remained dominant not because it was good, but because nothing better was widely available for 35 years.

## 6. Gap Ratings

| Gap | Rating | Evidence |
|-----|--------|----------|
| Layered Memory | 20% | Vestige has dual-strength (storage/retrieval) per memory but no semantic/episodic/procedural distinction. FSRS doesn't address this. |
| Multi-Angle Retrieval | 70% | Vestige's 7-stage pipeline (HyDE + FTS5 + vector + reranking + spreading activation) is one of the most diverse in the survey. FSRS doesn't address retrieval. |
| Contradiction Detection | 40% | Vestige's prediction error gating handles contradictions at write time (supersede on conflict). No graded classification. |
| Relationship Edges | 30% | Vestige has spreading activation over a memory graph. No typed relationships (supports/contradicts/etc.). |
| Sleep Process | 50% | Vestige's dreaming module consolidates memories. FSRS scheduler runs every 6 hours. Less sophisticated than our NREM/REM pipeline. |
| Reference Index | 20% | `memory_health` tool provides store-wide diagnostics. No topic-level indexing. |
| Temporal Trajectories | 40% | FSRS tracks full review history per item. Vestige's synaptic tagging allows retroactive importance assignment. No explicit trajectory detection. |
| Confidence/UQ | 50% | FSRS retrievability is a principled confidence-like score. Difficulty tracks per-memory hardness. No feedback loop in Vestige beyond review grades. |

## 7. Comparison with claude-memory

**Stronger (Vestige/FSRS):**
- **Principled decay model.** FSRS's power-function decay is optimized against 220M reviews. Our exponential decay uses hand-tuned rates per category. FSRS is empirically superior — but our Phase 14 experiment showed no difference at our timescale.
- **Parameter optimization.** FSRS learns from data via gradient descent. Our decay rates are fixed per source type with manual adjustment via recall_feedback. No optimization loop.
- **Vestige's retrieval diversity.** 7-stage pipeline with reranking. We have 2-channel RRF (FTS5 + vector), though Phase 14 shows FTS alone is sufficient at our scale.
- **Dual-strength model.** Storage strength vs. retrieval strength is a meaningful distinction we don't make. A memory might be well-encoded but currently inaccessible, or easily accessible but shallowly stored.

**Weaker (Vestige/FSRS):**
- **No human curation.** Vestige auto-ingests everything. No `review_pending()`, no manual priority setting, no editorial control. Our human-in-the-loop approach produces higher-quality memories at the cost of coverage.
- **No dimensional feedback.** FSRS gets binary grades (Again/Hard/Good/Easy). Our recall_feedback provides continuous utility scores (0.0-1.0) with durability (-1.0 to 1.0). Richer signal.
- **No edge types.** Vestige has graph traversal but no typed relationships (contradiction, revision, derivation). Our memory_edges table with flags enables structured reasoning about memory relationships.
- **No gestalt/summary layer.** Both systems store individual memories without higher-order synthesis. Our sleep pipeline produces cross-memory summaries; Vestige's dreaming is lighter.
- **Scale mismatch.** FSRS is designed for 1,000-100,000+ flashcards. Our system manages ~300 curated memories. The scheduling problem (when to review) doesn't arise for us — our memories are surfaced on demand via recall, not on a review schedule.

**Shared strengths:** Both use SQLite + FTS5. Both implement forgetting/decay. Both have consolidation processes. Both value local-first architecture.

## 8. Insights Worth Stealing

1. **Stability saturation in decay updates** (effort: low, impact: medium). FSRS's `S^(-w9)` term means repeated review of the same material yields diminishing stability gains. Our confidence gradient compounds linearly — we could add a saturation term so that a memory reviewed 50 times doesn't gain proportionally more confidence than one reviewed 10 times.

2. **The "almost forgotten" review bonus** (effort: medium, impact: medium). FSRS's `e^(w10*(1-R)) - 1` rewards retrieving memories at low retrievability. We could weight recall_feedback higher for memories that were surfaced with low priority (they were hard to find but still useful), strengthening the feedback signal for precisely the memories that need it most.

3. **Binary cross-entropy optimization of decay parameters** (effort: high, impact: high). Instead of hand-tuned decay rates per category, treat each recall_feedback event as a binary outcome (useful/not useful) and optimize our decay parameters to predict utility. Requires sufficient data — probably viable once we have 2,000+ feedback events.

4. **Difficulty mean reversion** (effort: low, impact: low-medium). FSRS prevents "ease hell" by reverting difficulty toward a default. Our confidence gradient could similarly revert toward a baseline over time if a memory isn't accessed — preventing very old, unretrieved memories from retaining artificially high confidence.

## 9. What's Not Worth It

- **Adopting FSRS scheduling for our use case.** FSRS solves "when should I review this flashcard?" Our system solves "which memories are relevant to this query?" These are different problems. FSRS's review scheduling is irrelevant — we don't schedule reviews; we surface memories on demand. **Our Phase 14 experiment confirmed this:** power-function decay and exponential decay produce identical Spearman correlations (0.158) with actual utility at our timescale. Access intervals are near-zero (all feedback occurs within sessions), so the curves never diverge.
- **Vestige's full 7-stage pipeline.** HyDE query expansion, neural reranking, USearch HNSW, and spreading activation are impressive engineering but massive dependencies (Nomic embeddings, Jina reranker, HNSW index). At our scale, FTS5 with curated metadata achieves equivalent MRR.
- **Prediction error gating.** Our Experiment 5 showed surprising memories are *less* useful (Spearman -0.20), not more. Novelty-based boosting would hurt ranking.
- **Dual-strength model implementation.** The storage/retrieval distinction is theoretically sound but adds schema complexity. Our simpler priority + confidence + decay model covers the practical need.

## 10. Key Takeaway

FSRS represents the most principled, empirically-validated approach to memory decay in any system we've surveyed. Its power-function forgetting curve, optimized against 220 million reviews, is grounded in 130 years of cognitive science from Ebbinghaus through Wixted & Ebbesen. The algorithm's key innovations — stability saturation, difficulty mean reversion, the "almost forgotten" review bonus — solve real problems in spaced repetition scheduling.

But FSRS solves a fundamentally different problem than ours. Spaced repetition asks "when should I re-present this item to maximize retention?" Our system asks "which stored memories are relevant to this query?" The forgetting curve is one signal among many in our retrieval ranking, not the primary scheduling mechanism. And our Phase 14 experiment showed that at our timescale (access intervals measured in seconds, not days), exponential and power-function decay produce identical predictions. The theoretical superiority of power decay only manifests at multi-week intervals we don't yet have in our data.

Vestige's contribution is demonstrating that FSRS can be adapted for AI agent memory — but the adaptation is mostly surface-level. The 7-stage retrieval pipeline and 29 cognitive modules are Vestige's own engineering; the FSRS algorithm itself is used for scheduling memory decay, not for retrieval. The insight for us: FSRS's parameter optimization approach (binary cross-entropy over recall events) could inform future optimization of our own decay parameters, once we accumulate sufficient feedback data to make optimization meaningful.

## References

**FSRS papers:**
- Ye, Su, Cao. "A Stochastic Shortest Path Algorithm for Optimizing Spaced Repetition Scheduling." KDD 2022. DOI: 10.1145/3534678.3539081
- Ye, Su, Cao. "Optimizing Spaced Repetition Schedule by Capturing the Dynamics of Memory." IEEE TKDE 2023. DOI: 10.1109/TKDE.2023.3251721
- Wixted & Ebbesen. "Genuine power curves in forgetting." Memory & Cognition, 1997.

**Repositories:**
- github.com/open-spaced-repetition/free-spaced-repetition-scheduler
- github.com/samvallad33/vestige

## See Also

- [[retrospective-experiments]] — Phase 14 Experiment 3 (power function vs. exponential, null result)
- [[agent-output-cortexgraph]] — Power-law decay recommendation (Phase 6)
- [[agent-output-locomo]] — Simple retrieval beating complex retrieval (related finding)
