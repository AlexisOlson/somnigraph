# The Geometry of Forgetting

*Generated 2026-04-11 by Opus agent reading arXiv:2604.06222*

---

## Paper Overview

**Paper**: Barman, S.R., Starenky, A., Bodnar, S., Narasimhan, N., & Gopinath, A. (2026). The Geometry of Forgetting. arXiv:2604.06222 [q-bio.NC], 27 March 2026. 20 pages (15 main + 5 extended data). Sentra / MIT.
**Code**: https://github.com/Dynamis-Labs/hide-project

**Problem addressed**: Why do we forget, and why do we "remember" things that never happened? The conventional answer blames biological hardware decay. This paper argues that these phenomena are geometric properties of any similarity-based retrieval system operating in a low-effective-dimensionality embedding space -- they arise from interference, not time.

**Core approach**: The authors build HIDE (High-dimensional Interference and Decay in Embeddings), a minimal embedding-based memory system using frozen pre-trained encoders (MiniLM, BGE-base, BGE-large) with learned linear projections that add positional, temporal, and episodic context. They store memories as contextually enriched embeddings and retrieve via cosine similarity with temporal decay modulation: `score = cos(q, m) * (1 + beta*t)^(-psi)`. They then subject this system to five classical memory experiments (Ebbinghaus forgetting, interference, DRM false memory, spacing effect, cross-modal binding) and show it reproduces human-like phenomena quantitatively with minimal or zero parameter tuning.

**Key claims**:
- Interference from competing memories, not temporal decay, is the dominant driver of power-law forgetting in embedding-based retrieval. **Evidence: strong.** Decay alone produces b ~= 0.009; with 10K competitors, b = 0.460 +/- 0.183 vs human b ~= 0.5 (Fig. 1, R^2 = 0.757).
- Production embedding models (384-1024 nominal dims) concentrate variance in ~16 effective dimensions, placing them in the interference-vulnerable regime. **Evidence: solid.** Participation ratio d_eff = 15.7-16.6 across three models (p. 3).
- Raw cosine similarity on unmodified pre-trained embeddings reproduces DRM false memory rates (0.583 vs human ~0.55) with zero parameter tuning. **Evidence: strong.** All 24 published DRM lists tested, consistent across lists (Fig. 2d).
- Geometric merging of nearby embeddings (centroid averaging) produces 62.5% compression but increases backward interference, making it a destructive operation. **Evidence: informative negative result** (Extended Data Fig. 5).
- The spacing effect survives age-dependent degradation through recency of the most recent trace. **Evidence: moderate.** Qualitative ordering matches human data; quantitative dynamic range is wider than human (long retention 0.994 vs human ~0.65).

---

## Key Findings

### Power-Law vs Exponential Decay

The central experiment (Section "Interference, not decay"): encode 1,000 facts spanning 30 simulated days with 10,000 distractor sentences from TempLAMA, query at day 30, bin retrieval accuracy by age. The decay function used is `S(t) = (1 + beta*t)^(-psi)` with psi = 0.5.

Critical result: **decay alone (no competitors) produces b ~= 0.009** -- effectively flat. Adding 10,000 distractors raises b to 0.460 +/- 0.183 (95% CI: [0.354, 0.644]), within the range of human data (b ~= 0.5). The forgetting curves steepen progressively as competitors accumulate (Fig. 1a), converging toward the classical Ebbinghaus curve. The dose-response relationship is monotonic (Fig. 1b).

The parameter sweep (Extended Data Fig. 3) shows b = 0.5 is achievable across a range of sigma x beta configurations, not a single tuned point -- but it does sit in a tuned region, not universally.

Semantic proximity matters: same-article ("near") competitors produce b = 0.161 at 40K competitors, while cross-article ("far") competitors only reach b = 0.132 at 50K (p. 3). Interference strength depends on angular distance.

### False Memory from Geometry

The DRM (Deese-Roediger-McDermott) result is the paper's strongest claim. All 24 published DRM word lists were encoded with BGE-large (1024-dim). Critical lures (e.g., "sleep" for the list bed/rest/awake/tired/dream) occupy positions geometrically indistinguishable from studied words in the embedding space, while unrelated words remain clearly separated (Fig. 2a UMAP).

At threshold theta = 0.82 (chosen by zero-unrelated-false-alarms criterion), the critical-lure false alarm rate is **0.583 vs human ~0.55** -- within 3.3 percentage points (Fig. 2b). This holds consistently across all 24 lists (Fig. 2d, mean lure cosine similarity 0.828). The full threshold operating curve (Fig. 2c) shows lure false alarms remain elevated above unrelated items across a wide range of thresholds.

The key insight: **no parameter was tuned to match human false memory rates**. The phenomenon is "unbaked" -- raw cosine similarity on raw pre-trained embeddings produces it. The semantic geometry that makes embeddings useful for retrieval is the same geometry that produces false memories. The paper frames this as a tradeoff frontier: eliminating DRM-type false memories would require sacrificing the semantic clustering that gives these models their power.

### Effective Dimensionality

The "dimensionality illusion" finding (p. 3): three production embedding models concentrate their variance in far fewer dimensions than their nominal count.

| Model | d_nom | d_eff | d_95 |
|-------|-------|-------|------|
| MiniLM-L6-v2 | 384 | 15.7 +/- 0.0 | 17-18 |
| BGE-base-en-v1.5 | 768 | 16.6 +/- 0.1 | 17-18 |
| BGE-large-en-v1.5 | 1024 | 16.3 +/- 0.1 | 17-18 |

d_eff computed as participation ratio: (sum lambda_i)^2 / sum(lambda_i^2) where lambda_i are PCA eigenvalues. All three converge to ~16 effective dimensions regardless of nominal dimensionality.

The functional consequence: at d >= 256, interference is near zero (b remains below 0.004 regardless of competitor count). At d = 128, the maximum b drops to 0.020. The interference-vulnerable regime is d <= 64 (Fig. 1c). Since production models have d_eff ~= 16, they are deep in this regime.

The paper notes cortical neural codes operate at estimated d_eff = 100-500 (refs 22, 23) -- "non-negligible but not catastrophic" interference, consistent with biology having meaningful but manageable forgetting.

### Interference vs Time-Based Decay

The core argument decomposes into three parts:

1. **Retrieval strength vs storage strength**: The paper operationalizes Bjork & Bjork's "new theory of disuse" (ref 3) geometrically. Stored embeddings retain their fidelity (storage strength is permanent), but retrieval strength -- the probability of landing at rank 1 -- decreases as competitors populate the angular neighborhood. Forgetting is a search problem, not a storage problem.

2. **Competition drives the curve shape**: Without competitors, decay alone produces near-flat retrieval (b ~= 0.009). The power law emerges from the accumulation of interfering neighbors over time. Older memories have had more time for competitors to accumulate nearby.

3. **Dimensionality gates vulnerability**: In high dimensions, the concentration of measure makes it exponentially unlikely for random points to fall within a fixed angular neighborhood. At d >= 128, interference effectively vanishes. The ~16 effective dimensions of production embeddings place them where interference is substantial.

Evidence quality: The interference experiment is well-controlled (varying competitors independently of decay). The DRM result is particularly compelling because it requires zero tuning. The spacing effect is weaker (boundary-condition-dependent, wider dynamic range than human data). The topological analysis (Rips complex, persistent homology, Fig. 4) is descriptive rather than causal -- the authors acknowledge this.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Power-law forgetting from interference, not decay | b = 0.460 +/- 0.183 with competitors vs 0.009 without; R^2 = 0.757 | Strong. Clean isolation of variables. But the temporal decay function is still present -- interference amplifies it rather than replacing it entirely. |
| DRM false memory at human rates with zero tuning | 0.583 vs ~0.55 across 24 lists | Strong. The zero-parameter claim is legitimate -- theta = 0.82 is set by an independent criterion (zero unrelated false alarms), not to match lure rates. |
| Production embeddings have ~16 effective dims | Participation ratio 15.7-16.6 across three models | Solid. But the models tested (MiniLM, BGE) are 2023-era. Newer models may differ. OpenAI text-embedding-3-small (used by Somnigraph) is not tested. |
| Consolidation via centroid merging is destructive | 62.5% compression, backward interference -0.100 to -0.394 | Informative. Directly warns against a common engineering practice. |
| Spacing effect from recency of most recent trace | Correct ordering (long > medium > short > massed), Cohen's d = 13.1 | Moderate. Ordering matches but dynamic range overshoots human data (0.994 vs ~0.65 for long). |
| Tip-of-tongue from retrieval competition | 3.66% vs human ~1.5% | Weak-moderate. Qualitative emergence but 2.4x overshoot; operational definition (rank 2-20) is looser than phenomenological criterion. |
| Topological structure (loops at phase transition) | H_1 peaks at 534 +/- 24 at epsilon = 1.0 | Descriptive only. Authors explicitly frame as non-causal. |

---

## Implications for Somnigraph

### Challenge to exponential decay

Somnigraph uses per-category exponential decay with configurable half-lives (constants.py: episodic 0.099/~7d, procedural 0.012/~58d, semantic 0.008/~87d, etc.) and a floor at 0.35. This paper argues power-law forgetting from interference is the correct model, and that exponential decay alone (without competition) produces essentially no forgetting.

However, the practical impact for Somnigraph is **limited for several reasons**:

1. **Somnigraph's decay affects scoring, not retrieval filtering.** The decay multiplier modulates the reranker score, but the reranker has 26 features -- decay is one signal among many, and the reranker learned its own weighting. The paper's strongest claim (interference dominates decay) is about pure cosine-similarity retrieval; Somnigraph's hybrid BM25 + vector + reranker pipeline already sidesteps the pure-geometry regime.

2. **The shadow penalty was already removed.** Somnigraph already learned (experiments.md) that time-based signals confuse temporal relevance with query relevance. The system has moved toward content-based scoring. This paper provides theoretical backing for that empirical finding.

3. **Somnigraph has ~2,000 memories, not 10,000+.** The interference effect scales with competitor count. At small store sizes, the geometric effects are less pronounced (Extended Data Fig. 2b shows precision declining with store size, but modestly at <500).

4. **Switching to power-law would change little in practice.** The decay formula `(1 + beta*t)^(-psi)` vs `exp(-lambda*t)` differs mainly in tail behavior -- power-law retains old memories longer. But Somnigraph already has a floor (0.35) and reheat-on-access, which produce a similar practical effect (old but accessed memories don't vanish).

Where the challenge is real: if Somnigraph scales to tens of thousands of memories, interference-based retrieval degradation will matter. The paper predicts retrieval accuracy will degrade as a power law of database size, which is a more pessimistic scaling story than "just add more storage." The reranker may mitigate this -- it was trained to distinguish relevant from interfering memories -- but the geometric floor on vector-similarity recall is real.

### Implications for retrieval

The effective dimensionality finding has a concrete implication: **Somnigraph's text-embedding-3-small at 384 dimensions likely has d_eff in the same ~16 range** (the paper tested MiniLM at 384 dims, same ballpark). This means vector-similarity recall in Phase 1 (the RRF fusion stage) is operating in the interference-vulnerable regime.

Somnigraph already mitigates this through:
- Hybrid retrieval (BM25 provides an orthogonal signal not subject to geometric interference)
- Graph-based expansion (PPR, synthetic vocabulary bridges) that bypasses the angular neighborhood problem
- The reranker (discriminates between true matches and interference-driven false matches)

The paper's "vector averaging fallacy" finding (Extended Data Fig. 5) is a **direct warning against embedding consolidation**: centroid merging of nearby memories increases backward interference. Somnigraph's sleep pipeline does not merge embeddings -- it archives low-value memories and creates summary nodes -- but any future consolidation that averages embeddings should be avoided.

### Implications for consolidation

The consolidation failure result is the most directly actionable finding for Somnigraph's sleep pipeline. The paper tested geometric merging on 100 visual categories (CIFAR-100) and found:
- 62.5% compression achieved
- Backward interference increased from -0.100 to -0.394
- The tradeoff is not favorable: compression gains don't justify accuracy loss

This vindicates Somnigraph's current approach where NREM consolidation operates at the **metadata level** (pairwise relationships, edge creation, merge/archive decisions) rather than at the embedding level. The paper says explicitly: "centroid merging erases the fine angular structure that separates semantically adjacent memories, collapsing distinct traces into a blurred centroid that confuses retrieval" (p. 10).

For Somnigraph's sleep pipeline, this means:
- Summary nodes should have their own independently computed embeddings, not averaged from children
- Archiving (removing low-value memories from the active store) is actually beneficial from the interference perspective -- it reduces the competitor pool
- The current design of keeping source memories separate and linking them via graph edges is geometrically sound

---

## Worth Stealing (ranked)

### 1. Effective dimensionality audit (low effort)
**What**: Compute participation ratio for Somnigraph's actual embedding matrix (text-embedding-3-small, 384-dim) using the formula d_eff = (sum lambda_i)^2 / sum(lambda_i^2) on PCA eigenvalues.
**Why**: If d_eff is ~16 like the paper's models, confirms we're in the interference-vulnerable regime and quantifies how much vector recall degrades with scale. If d_eff is higher (OpenAI may differ from open-weight models), it changes the scaling story.
**How**: One-off script: load all memory embeddings from SQLite, compute PCA, report d_eff, d_95, d_99. Takes minutes.

### 2. Interference-aware archival scoring (medium effort)
**What**: When the sleep pipeline evaluates memories for archival, factor in local embedding density (number of neighbors within angular threshold) as a signal. Dense neighborhoods have higher interference; archiving low-value members reduces interference for the remaining high-value ones.
**Why**: The paper shows forgetting scales with competitor count in the angular neighborhood. Targeted archival in dense regions would disproportionately improve retrieval quality for the remaining memories.
**How**: During NREM, compute pairwise cosine similarity for each memory's k-nearest neighbors. Memories that are low-value AND in dense neighborhoods become higher-priority archival candidates. Would require adding a feature to the archival scoring in sleep_nrem.py.

### 3. Power-law decay as an option (low effort)
**What**: Add `(1 + beta*t)^(-psi)` as an alternative decay function alongside the current exponential.
**Why**: Heavier tail retains old memories longer, which may be more appropriate for semantic and procedural categories where slow-but-never-zero forgetting is desired. The floor mechanism (DECAY_FLOOR = 0.35) already approximates this behavior, so the practical difference may be small.
**How**: Add a config option in constants.py, implement the alternative formula in whatever computes decay scores. Compare against floor-bounded exponential on the tuning dataset.

---

## Not Useful For Us

- **Cross-modal retrieval** (Section "Shared embedding geometry enables cross-modal retrieval"): Somnigraph is text-only. The lightweight projection alignment is interesting but irrelevant.
- **The HIDE system itself**: Not a production memory system. It's a minimal experimental harness (frozen encoders + linear projection + cosine retrieval). The insights matter; the code doesn't.
- **Topological analysis** (Rips complex, Betti numbers): Descriptive, non-causal, and on Wikipedia data. The phase-transition finding (clusters merge at epsilon ~= 0.9-1.0) is intellectually interesting but has no engineering application.
- **The bAbI proof-of-concept** (mean accuracy 0.475): Below full-context ceiling, included as a sanity check, not competitive. The fan-effect observation (precision declines with memory load) is just restating the interference thesis in task form.

---

## Connections

- **Decay models**: Somnigraph's exponential decay with floor (constants.py, CATEGORY_DECAY_RATES, DECAY_FLOOR) is the direct target of this paper's claims. The existing shadow penalty removal (experiments.md, architecture.md) already moved away from time-based signals, consistent with this paper's argument.
- **vestige-fsrs.md**: Vestige implements Bjork & Bjork's dual-strength model (storage strength vs retrieval strength) -- the same theoretical framework this paper provides geometric foundations for. FSRS's stability parameter maps to what this paper calls storage strength; FSRS's retrievability maps to retrieval strength modulated by interference.
- **cortexgraph.md**: CortexGraph offers three decay models including power-law `(1 + dt/t0)^(-alpha)` -- exactly the functional form this paper uses. CortexGraph chose power-law as default based on "heavier tail, retains older memories"; this paper provides the theoretical justification.
- **aurora.md**: Aurora's design philosophy of biological plausibility in memory systems connects to this paper's argument that biological memory phenomena are geometric, not biological.
- **architecture.md**: The "vector averaging fallacy" finding directly supports the current design decision to not merge embeddings during consolidation. The enriched-embedding degradation finding from external reviews (architecture.md, "What didn't work") is a related concern -- enriching embeddings with metadata can change the effective geometry.
- **Embedding dimensionality**: Somnigraph uses text-embedding-3-small at 384 dimensions (configurable via CLAUDE.md). The paper's finding that 384-dim MiniLM has d_eff = 15.7 suggests our vector space may have similar effective dimensionality, making interference a real concern at scale.

---

## Summary Assessment

**Relevance**: Medium-high. This is a theoretical paper, not a system, but its core findings speak directly to design decisions in any embedding-based memory system. The interference-vs-decay argument, the effective dimensionality finding, and the vector averaging fallacy are all actionable.

**Strength of evidence**: The DRM and interference experiments are well-controlled and compelling. The spacing effect and TOT results are weaker (boundary-condition-dependent, wider dynamic range than human data). The paper is honest about scope: "what remains to be tested is whether alternative decay functions, noise models, or retrieval rules produce qualitatively different conclusions" (p. 11).

**Practical impact for Somnigraph**: Low-to-moderate in the short term. Somnigraph already moved away from time-based signals for scoring (shadow penalty removal), uses hybrid retrieval that mitigates pure-geometry interference, and doesn't merge embeddings. The main actionable item is the effective dimensionality audit (Worth Stealing #1) -- understanding whether our embedding space has the same ~16 effective dimensions would inform scaling decisions. At current scale (~2K memories), interference is unlikely to be a dominant problem. At 10K+, it could become one.

**Limitations to note**: All experiments used English-language data. The temporal decay function is present in all experiments (it's not purely interference -- it's interference amplifying decay). The "zero parameter tuning" claim for DRM is legitimate but the theta = 0.82 threshold is itself a parameter, even if chosen by an independent criterion. The paper tests open-weight models (MiniLM, BGE); commercial models like OpenAI's may have different effective dimensionality profiles.

**Category**: Theoretical/empirical (neuroscience-AI crossover). Not a memory system. Relevance level: medium.
