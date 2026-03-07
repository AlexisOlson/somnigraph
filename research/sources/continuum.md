# Continuum Memory Architectures — Analysis

*Generated 2026-02-18 by Opus agent reading 2601.09913v1*

---

## Paper Overview

- **Author**: Joe Logan, Mode7 GK (Tokyo, Japan)
- **Venue**: arXiv preprint, January 14, 2026
- **Paper**: arXiv:2601.09913v1 [cs.AI]

**Core problem**: RAG treats memory as a stateless lookup table — information persists indefinitely, retrieval is read-only, temporal continuity is absent. This leaves LLM agents without continuity of identity or purpose across extended time horizons.

**Key contribution**: Formalizes **Continuum Memory Architecture (CMA)** as an architectural class defined by six behavioral requirements that any memory system must satisfy to support long-horizon agents. Rather than proposing a specific implementation, the paper names the abstraction and specifies the necessary conditions, then validates with behavioral probes against a RAG baseline.

This is a definitional paper, not a systems paper. Its value is in vocabulary, taxonomy, and the behavioral checklist — not in novel algorithms or benchmark results.

---

## Architecture / Method

### The Six CMA Requirements (Behavioral Checklist)

CMA is defined by behavioral properties, not mechanisms. A system missing any single element "remains a form of RAG, even if heavily engineered." The requirements:

| # | Requirement | Operational Test | Cognitive Source |
|---|-------------|-----------------|-----------------|
| 1 | **Persistence** | Retrieve fragments from interactions days old without replaying transcripts | Basic memory function |
| 2 | **Selective retention** | Observable divergence between updated and outdated facts | Ebbinghaus (1885), Anderson et al. (1994) |
| 3 | **Retrieval-driven mutation** | Measurable changes in ranking after repeated queries | Retrieval-induced forgetting/strengthening |
| 4 | **Associative routing** | Multi-hop recall where answers lack lexical overlap with query | Collins & Loftus (1975) spreading activation |
| 5 | **Temporal continuity** | Queries about context return events within bounded time window around anchor | Tulving (1972) episodic memory |
| 6 | **Consolidation & abstraction** | Derived fragments (gists, insights) that summarize clusters and influence future retrieval | Squire & Alvarez (1995), Walker & Stickgold (2006) |

These are framed as "necessary and collectively sufficient conditions for CMA compliance." This is the paper's strongest contribution — a pass/fail checklist for evaluating any agent memory system.

### Reference CMA Lifecycle

The paper describes a four-stage lifecycle without prescribing specific data structures:

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────┐     ┌────────────────┐
│   INGEST    │ ──→ │   ACTIVATION    │ ──→ │  RETRIEVAL   │ ──→ │ CONSOLIDATION  │
│             │     │     FIELD       │     │  + MUTATION   │     │                │
│ - metadata  │     │ - spreading     │     │ - multi-factor│     │ - replay       │
│ - salience  │     │   activation    │     │   ranking     │     │ - abstraction  │
│ - temporal  │     │ - edge decay    │     │ - reinforce   │     │ - gist extract │
│   classify  │     │ - context cues  │     │ - suppress    │     │ - dormancy     │
│ - novelty   │     │                 │     │   near-misses │     │                │
│   detect    │     │                 │     │ - co-link     │     │                │
└─────────────┘     └─────────────────┘     └──────────────┘     └────────────────┘
```

#### Memory Substrate

A structured store where fragments are nodes connected by three types of edges:
- **Semantic edges** (content similarity)
- **Temporal edges** (FOLLOWED_BY, session/episode ordering)
- **Structural edges** (associative links formed through co-activation)

Each node retains: reinforcement history, salience, timestamps, provenance.

#### Activation Field

The most distinctive conceptual contribution. Queries, context, and system events inject activation that propagates along edges with decay — echoing Collins & Loftus (1975) spreading activation theory. The activation field converts intent into *graded availability*, enabling disambiguation without lexical cues.

Example given: "everything mentioning Python" vs. "the subset of memories activated by a zoo visit" — activation field distinguishes these even though both might share the word "Python."

Implementation-agnostic: "Whether implemented via message passing, diffusion on a graph, or attention over learned embeddings."

#### Ingest Pipeline (Evaluated Instantiation)

The paper describes what was actually implemented:

1. **Sentiment/salience analysis** — scalar approximating affective intensity, governs downstream retention
2. **Temporal classification** — labels fragments as `episodic`, `habitual`, or `timeless`
3. **Conversation buffer** — captures session context for associative retrieval
4. **Novelty detection** — merges fragments exceeding a similarity threshold (strengthens reinforcement rather than duplicating)
5. **Capacity management** — evicts low-salience, low-reinforcement memories when storage budgets are tight

#### Consolidation (Background Process)

Three sub-processes:
1. **Replay** — traverses recent sequences to strengthen temporal chains (NREM-inspired)
2. **Abstraction** — synthesizes latent themes/"insights" from clusters (REM-inspired)
3. **Gist extraction** — converts repeated episodes into semantic knowledge that survives even if original details decay (systems consolidation)

**Dormant memories remain recoverable under strong cues** — preventing catastrophic forgetting. This is a design choice we should note: not deletion, but dormancy.

### Appendix: Component Architecture

The evaluated instantiation had four services:

| Component | Function |
|-----------|----------|
| **Ingest Service** | Text analysis (sentiment, salience), temporal classification, novelty detection, capacity management |
| **Activation Engine** | Seeds with top-k semantic matches, propagates along edges with damped spreading activation |
| **Retrieval Scorer** | Combines: vector similarity + activation + recency decay + structural reinforcement + contextual relevance → scalar score. Implements retrieval-induced mutation (reinforce returned nodes, suppress near-misses) |
| **Consolidation Jobs** | Background workers for replay walks, LLM-based cluster abstraction, gist extraction. Abstracted nodes link back to sources. Low-utility fragments go dormant. |

---

## Key Claims & Evidence

### Experimental Setup

- **Baseline**: Supabase pgvector RAG with `text-embedding-3-small`, enhanced with recency-weighted scoring: `score_adjusted = score × e^(-λΔt)` where `Δt` is document age
- **Judge**: GPT-4o (anonymized System A vs System B)
- **Statistical testing**: Two-sided permutation tests (10,000 shuffles), McNemar's test for win/loss counts. All p < 0.01.
- **Embeddings**: Identical between CMA and RAG (`text-embedding-3-small`) to isolate retrieval behavior

### Results Summary

| Study | RAG Wins | CMA Wins | Ties | Effect Size | N |
|-------|----------|----------|------|-------------|---|
| Knowledge Updates | 1 | 38 | 1 | d = 1.84 | 40 |
| Temporal Association | 1 | 13 | 2* | h = 2.06 | 30 |
| Associative Recall | 5 | 14 | 10 | h = 0.99 | 30 |
| Disambiguation | 3 | 17 | 26 | h = 1.55 | 48 |
| **Total** | **10** | **82** | **39** | — | 148 |

*Temporal Association: 47% "both wrong" rate — nearly half of temporal queries stumped both systems. This is an honest acknowledgment of how hard temporal chaining remains.

**Overall**: CMA won 82 of 92 decisive trials across all probes.

**Latency cost**: 2.4x increase (mean 1.48s vs 0.65s RAG). This is the expected price of graph traversal and post-retrieval mutation.

### Study Details

**Study 1: Knowledge Updates** (Selective Retention)
- 40 scenarios across 8 domains, introduce fact then issue correction
- CMA consistently surfaced updated facts (38/40 wins)
- RAG returned outdated information due to higher similarity between query and original statement
- Judge quote: "System A keeps resurfacing the REST answer even though the question references the migration"
- **Mechanism**: recency + interference + activation collectively prioritize current knowledge without manual versioning

**Study 2: Temporal Association** (Temporal Continuity)
- 10 naturalistic episodes (runs, travel delays, remote work, medical visits) — semantically diverse events linked only by time
- Query: "what happened around X?"
- CMA: 13/14 decisive wins
- Judge quote: "System B reminded me about tripping on the sidewalk and the knee pain afterward; System A repeated the deer sighting only"
- **Critical finding**: 47% "both wrong" rate. Temporal chaining remains challenging even for CMA. Sensitive to episode-boundary heuristics.

**Study 3: Associative Recall** (Associative Routing)
- 5 project knowledge graphs interlinked people, technologies, milestones, subsystems
- Queries required multi-hop retrieval (e.g., technologies used by a given team)
- CMA: 14/19 decisive wins
- Judge quote: "System B connects Sarah Chen to Atlas and surfaces TensorFlow/Kubernetes, while System A stays at the personnel level"
- **Mechanism**: graph traversal through association edges, not just embedding similarity

**Study 4: Disambiguation** (Contextual Partitioning)
- 8 ambiguous terms (Python, Apple, Java, Bug, Mercury, Jaguar, Shell, Trunk) embedded in distinct contexts
- CMA: 17/20 decisive wins
- Judge quote: "System B stayed entirely within the zoo memories when prompted about my visit, whereas System A mixed in programming references"
- **Mechanism**: cluster-sensitive activation field suppresses contamination from other contexts

### Limitations Acknowledged

1. **Latency and scaling**: Activation propagation grows with edge count. Mitigation: multi-resolution graphs, capped activation fan-out per hop
2. **Memory drift**: Retrieval-induced updates can slowly distort facts via feedback loops. Mitigation: provenance logging, reinforcement history, anomaly scores
3. **Temporal sensitivity**: 47% both-wrong rate on temporal queries. Needs: session signals, learned boundary detectors, user controls
4. **Interpretability**: Evolving graph is hard to audit. Needs: interactive viewers, queryable audit trails
5. **Data governance**: Persistent memories raise privacy/compliance concerns. Needs: retention policies, deletion workflows, encrypted shards

### Limitations Not Acknowledged (My Assessment)

1. **Single author, single implementation** — no independent replication. The behavioral probes are custom-designed, making cross-comparison impossible.
2. **No established benchmarks** — explicitly declines LongBench and LoCoMo. The custom probes isolate CMA properties but prevent comparison with Zep (71.2% LongMemEval), Hindsight (83.6-91.4% LongMemEval), or any other system.
3. **No ablation studies** — which CMA properties contribute most? Would selective retention alone get 80% of the benefit? We don't know.
4. **GPT-4o as judge** — informal spot-checks of only 30 random trials. No human evaluation. The paper acknowledges this.
5. **RAG baseline is too weak** — Supabase pgvector with time-decay is a reasonable starting point, but not a strong baseline by 2026 standards. No comparison against RRF-enhanced RAG, Zep, Hindsight, or any production memory system.
6. **Implementation details withheld** — "Rather than disclosing implementation specifics" is frustrating. Appendix A provides component-level descriptions but no algorithms, formulas, or thresholds. This makes the paper reproducibility-limited.
7. **No consolidation evaluation** — consolidation is a CMA requirement but none of the four probes test it directly. The paper evaluates the *results* of consolidation but not the consolidation process itself.

---

## Relevance to claude-memory

### Where Our System Fits in Their Taxonomy

Using the CMA compliance checklist to audit claude-memory:

| CMA Requirement | claude-memory Status | Notes |
|----------------|---------------------|-------|
| **Persistence** | **PASS** | Supabase stores memories across sessions indefinitely |
| **Selective retention** | **PARTIAL** | Temperature-based decay + priority exist but no interference-based suppression. `superseded_by` flag exists but is manual. No automatic recency-boosting on contradiction. |
| **Retrieval-driven mutation** | **PARTIAL** | `last_accessed` and `access_count` updated on recall. But no reinforcement of co-retrieved items, no suppression of near-misses, no competitive dynamics. |
| **Associative routing** | **FAIL** | No graph traversal. No edges yet (`memory_edges` is planned but not implemented). Retrieval is vector-only. BM25 planned but not implemented. |
| **Temporal continuity** | **FAIL** | No temporal edges. No FOLLOWED_BY links. No episode boundary detection. Time is metadata, not structure. |
| **Consolidation & abstraction** | **PARTIAL** | `consolidate()` exists (dedup by similarity) but is mechanical merge, not semantic synthesis. `/sleep` designed but not implemented. No layer generation, no gist extraction, no replay. |

**Assessment**: claude-memory is currently a "heavily engineered RAG" by CMA's definition — it meets persistence and has partial selective retention/mutation, but fails on associative routing, temporal continuity, and meaningful consolidation. This is not surprising — these are exactly the features at the top of our implementation priority list.

**Achieving CMA compliance requires implementing priorities #1-5 on our list**: RRF fusion (#1), relationship edges (#2), graded contradiction detection (#3), decay floor (#4), and sleep skill (#5). After those five, claude-memory would satisfy all six CMA requirements.

### What Directly Applies

**1. The behavioral checklist is a useful evaluation framework for our own system.**

Rather than only measuring recall accuracy, we can design our own probes modeled on their four studies:
- **Knowledge Update probe**: Store fact, store correction, query for current state. Does the system return the correction?
- **Temporal Association probe**: Store several memories from the same session/time period. Query "what else was happening around X?" Does the system return temporally adjacent memories?
- **Associative Recall probe**: Store memories with implicit entity connections. Query via a multi-hop path. Does the system traverse edges?
- **Disambiguation probe**: Store memories about the same term in different contexts. Query with context cues. Does the system stay in the right cluster?

These are cheap to run (small corpus, GPT-4o-mini judge, a few dozen queries each) and would give us concrete pass/fail signals as we implement features.

**2. The "activation field" concept maps to multi-signal retrieval scoring.**

The paper's activation field is a generalization of what Hindsight does with multi-path retrieval + RRF. CMA names it more precisely: activation propagates along edges from query seeds, producing graded availability that considers structural position (not just similarity).

For claude-memory, this means our RRF fusion (#1) + graph traversal (from `memory_edges`, #2) together constitute an activation field implementation. We don't need to build an explicit propagation engine — RRF over vector + BM25 + edge-traversal results achieves the same behavioral outcome.

**3. Retrieval-driven mutation is more than just `access_count++`.**

CMA requires that "every lookup alters future accessibility." Our current mutation is minimal:
- `last_accessed` updated
- `access_count` incremented
- `temperature` affected by decay formula

What CMA adds:
- **Reinforcement**: retrieved memories gain strength (beyond simple access count)
- **Suppression**: near-misses (retrieved but not used/returned) are weakened
- **Co-activation linking**: memories retrieved together form new edges

The suppression mechanism is the most interesting. In our system, when `recall()` returns 10 memories but the agent only uses 3, the other 7 are near-misses. CMA suggests these should be slightly suppressed in future retrieval — implementing retrieval-induced forgetting. This is a genuinely novel behavioral signal we haven't considered.

**4. Temporal classification of fragments (episodic / habitual / timeless) maps to our category system.**

CMA classifies fragments at ingest time as episodic, habitual, or timeless. Our categories (episodic, semantic, procedural, reflection, meta) already capture this distinction with finer granularity:
- `episodic` → episodic
- `habitual` → procedural
- `timeless` → semantic

No change needed, but the CMA framing validates our decision to classify at storage time.

**5. Dormancy over deletion.**

CMA explicitly states that dormant memories "remain recoverable under strong cues, preventing catastrophic forgetting." Our `is_deleted` flag is binary deletion. A dormancy state would be a softer alternative: memories that have decayed below a floor but aren't deleted, and can be revived by a sufficiently specific query.

This relates to our planned decay floor (#4). The combination of `decay_floor` + dormancy state would satisfy this CMA requirement: memories never fully disappear, they just become very hard to activate unless strongly cued.

### What Validates Our Existing Decisions

1. **The six requirements are exactly our top-5 priorities.** RRF fusion enables multi-factor retrieval (requirement 4). Relationship edges enable temporal continuity (5) and associative routing (4). Contradiction detection implements selective retention (2). Sleep skill implements consolidation (6). The paper independently arrived at the same priority ordering we did.

2. **Consolidation as background process.** CMA's consolidation is explicitly a background operation, not inline. Our `/sleep` design is validated.

3. **Multi-factor retrieval over pure vector similarity.** "Semantic similarity is only one vote among many." Validates RRF fusion (#1) as highest priority.

4. **Memory as mutable, not append-only.** CMA requires that retrieval changes state. Our `temperature` / `access_count` system is a rudimentary version; CMA pushes us toward richer mutation semantics.

5. **Novelty detection / deduplication at ingest.** Our `remember()` already checks similarity threshold (0.9) before storing. CMA does the same — merge near-duplicates, strengthen reinforcement rather than duplicating.

### What Challenges Our Existing Decisions

1. **We don't implement near-miss suppression.** CMA requires that retrieval *weakens* competitors. Our system only strengthens accessed memories (via `access_count`). We have no mechanism for suppressing memories that were retrieved but not used. This is a genuinely new behavioral requirement we haven't considered.

2. **We don't co-link retrieved memories.** When multiple memories are recalled together, CMA suggests they should form new associative edges. We don't do this. Once `memory_edges` exists, we could auto-create `co_retrieved` edges when memories appear together in the same recall result.

3. **Our temporal classification is category-level, not fragment-level.** CMA classifies each fragment as episodic/habitual/timeless independently of its category. A procedural memory could contain temporal information ("Alexis started using Rust in December") that should be temporally indexed. Our category system doesn't capture this per-fragment temporal scope.

---

## Worth Stealing (ranked)

### 1. CMA Behavioral Checklist as Evaluation Framework — LOW effort, HIGH value

Implement the four behavioral probes as automated tests for our memory system. Run after each major feature addition to verify CMA compliance. Takes a small corpus (~50 memories), a dozen queries per probe, and GPT-4o-mini as judge.

**Why**: Gives us concrete pass/fail metrics for features we're building. Currently we have no systematic evaluation framework. "Did adding RRF improve associative recall?" becomes measurable.

**Effort**: 1-2 days to build the test harness. Reusable across all future development.

### 2. Near-Miss Suppression on Retrieval — LOW effort, MEDIUM value

When `recall()` returns results, track which memories were retrieved but not included in the final response (or not used by the agent). Apply a small temperature/priority penalty. Implements retrieval-induced forgetting.

**Why**: Introduces competitive dynamics into the memory ecosystem. Memories that repeatedly fail to be useful in recall slowly fade. Currently, unused memories persist at full strength indefinitely unless manually adjusted.

**Effort**: Low. Requires tracking which recalled memories were actually consumed (may need a `recall_feedback()` or heuristic based on whether the memory appeared in the response). The penalty itself is a simple temperature adjustment.

### 3. Co-Activation Edge Creation — MEDIUM effort, MEDIUM value

When memories are recalled together in the same query, create lightweight `co_retrieved` edges between them. Over time, this builds an associative graph from actual usage patterns.

**Why**: Builds organic associative structure without requiring LLM classification or manual curation. The edges reflect real retrieval patterns. Prerequisite: `memory_edges` table (#2 on priority list).

**Effort**: Medium. Requires `memory_edges` table to exist. The creation logic is simple (for each pair in a recall result, create or strengthen edge). Risk: could create too many edges. Needs a threshold (e.g., only create edge after co-retrieval count exceeds 3).

### 4. Dormancy State (Alternative to Deletion) — LOW effort, MEDIUM value

Add a `dormant` state alongside `active` and `deleted`. Dormant memories are excluded from normal retrieval but can be revived by high-similarity queries or explicit recall with expanded scope.

**Why**: Prevents catastrophic forgetting. Memories that have decayed below usefulness aren't lost — they're just harder to activate. Combined with decay floor (#4), this creates a two-tier fallback: memories decay to the floor, then eventually go dormant, but never fully disappear.

**Effort**: Low. Add a `state` enum (`active`, `dormant`, `deleted`) to the `memories` table. Modify `recall()` to filter by `state = 'active'` normally, with an option to include dormant. Add a reactivation mechanism (recall with high similarity to dormant memory → set state back to active).

### 5. Fragment-Level Temporal Scope — LOW effort, LOW-MEDIUM value

At `remember()` time, classify each memory's temporal scope as `episodic` (specific moment), `habitual` (recurring pattern), or `timeless` (permanent fact). Store as a metadata field.

**Why**: Enables different decay behaviors. Episodic memories should decay faster unless reinforced. Timeless memories (facts about the world) shouldn't decay at all. Habitual memories (procedures) should decay at a moderate rate. Currently all memories decay identically regardless of temporal scope.

**Effort**: Low. One additional LLM classification in `remember()`, one enum field. Could even be rule-based from the existing `category` field as a starting point.

### 6. Activation Propagation Scoring — HIGH effort, HIGH value (deferred)

Build a multi-signal retrieval scorer that combines vector similarity, BM25, graph traversal, recency decay, access frequency, and structural reinforcement into a single scalar. This is the full "activation field" described in CMA.

**Why**: This is the ultimate expression of CMA-compliant retrieval. Each signal is "one vote among many." Currently we only have one signal (vector similarity).

**Effort**: High. Requires RRF fusion (#1), memory_edges (#2), and BM25 to all exist first. The scoring function itself is the integration layer on top. Defer until prerequisites are complete.

---

## Not Useful For Us

### Salience/Sentiment Analysis on Ingest

CMA's ingest pipeline runs sentiment analysis to derive "affective intensity" that governs retention. For our system, the human caller already makes salience judgments by choosing what to `remember()` and at what priority. Automated sentiment scoring would be redundant and potentially misleading (the *content* of a memory might be about something negative, but its *salience* is determined by the user's judgment, not the text's emotional valence).

### Capacity Management / Eviction

CMA describes a capacity manager that evicts low-salience memories when storage budgets are tight. Our system is single-user with hundreds to low-thousands of memories. We'll never hit a storage budget. Dormancy + decay floor handles the "too many memories" problem without actual eviction.

### The Paper's Specific Implementation (Withheld)

The paper deliberately withholds implementation details. What is disclosed in Appendix A is high-level component descriptions without algorithms, thresholds, or data structures. There's nothing concrete to steal beyond the architectural concepts already captured above.

### Custom Behavioral Probes as a Publication Strategy

The paper's custom-designed probes are useful for our internal evaluation but wouldn't serve as a publication-grade benchmark. If we ever need to evaluate claude-memory formally, we should use LongMemEval (which Zep and Hindsight both use) rather than custom probes.

---

## Impact on Implementation Priority

### Current priority list (from index.md):

1. RRF fusion
2. Relationship edges
3. Graded contradiction detection
4. Decay floor + power-law
5. Sleep skill
6. Mutation log
7. Confidence scores
8. Reference index
9. Multi-angle indexing

### Assessment: No reordering needed.

The CMA paper validates our priority ordering rather than disrupting it. It provides a theoretical framework (the six requirements) that maps directly to our top 5 priorities. Specific adjustments:

**Add to #2 (Relationship edges)**: Plan for co-activation edge creation. When memories are co-retrieved, create/strengthen `co_retrieved` edges automatically. This builds organic associative structure from usage patterns.

**Add to #3 (Graded contradiction detection)**: CMA's "selective retention" requirement strengthens the case for doing contradiction detection at `remember()` time (not just during sleep). Already planned, now theoretically grounded.

**Add to #4 (Decay floor)**: Consider adding a dormancy state. Memories that hit the decay floor stay there for a period, then transition to dormant (excluded from normal retrieval but recoverable). This satisfies CMA's "dormant memories remain recoverable under strong cues."

**Expand #6 (Mutation log)**: Add near-miss tracking. When `recall()` returns candidates that don't make the final cut, log them as near-misses. This is the raw material for implementing retrieval-induced suppression later.

**New item #10: CMA behavioral probes**: Build a lightweight test harness implementing the four probe types (knowledge update, temporal association, associative recall, disambiguation). Run after each major feature to verify CMA compliance. Low effort, high diagnostic value.

---

## Connections

### To Zep Paper (Source 1)

The CMA and Zep papers are complementary in an almost perfect way: CMA provides the *what* (behavioral requirements) while Zep provides the *how* (specific implementation with benchmarks).

| CMA Requirement | Zep Implementation |
|----------------|-------------------|
| Persistence | Postgres + Neo4j, indefinite storage |
| Selective retention | Temporal invalidation (`t_invalid`), edge versioning |
| Retrieval-driven mutation | **Not implemented** — Zep retrieval is read-only |
| Associative routing | BFS graph traversal from seed nodes |
| Temporal continuity | Bi-temporal model (`t_valid` / `t_invalid`), episode subgraph |
| Consolidation | Community detection + map-reduce summarization |

**Key insight**: Zep fails the retrieval-driven mutation requirement. Despite being a sophisticated production system, Zep's retrieval is read-only — it doesn't implement reinforcement, suppression, or co-activation linking. By CMA's definition, Zep is "heavily engineered RAG" for this specific requirement. This is a useful diagnostic: even strong systems can miss fundamental behavioral properties.

**Complementary strengths**: Zep gives us bi-temporal fields, temporal extraction prompts, and entity resolution — concrete implementation patterns. CMA gives us the evaluation framework to know whether those implementations actually produce the right *behaviors*. Using both together: implement Zep's features, validate with CMA's probes.

### To Prior Phase Findings

**CLS Theory** ([[broader-landscape#Complementary Learning Systems (CLS)]]): CMA's consolidation requirement is a direct application of CLS. The fast system (`remember()`) collects; the slow system (`/sleep`) consolidates. CMA adds the specific behavioral test: consolidation must produce "derived fragments (gists, insights) that summarize clusters and influence future retrieval." CLS predicts this is necessary; CMA operationalizes it.

**Generative Agents** (Park et al.): CMA's consolidation subsumes Park's reflection mechanism. The paper explicitly cites Walker & Stickgold's "dreaming-inspired routine" for abstraction — the same biological basis as Park's reflection. Our `/sleep` skill design is validated from both directions: as periodic explicit consolidation (Park) and as a CMA-compliant abstraction process.

**Hindsight** ([[agent-output-hindsight]]): Hindsight's four-way retrieval with RRF is the closest existing implementation to CMA's activation field. CMA generalizes what Hindsight does concretely. Hindsight also implements retrieval-driven mutation (reinforcement on `retain`), making it more CMA-compliant than Zep on that axis.

**Engram** ([[agent-output-engram]]): Engram's 5-stage consolidation pipeline + graded contradiction detection maps to CMA requirements 2 (selective retention) and 6 (consolidation). Engram's working memory spreading activation is the closest implementation to CMA's activation field concept among the repos we've reviewed.

**Evo-Memory** (DeepMind): Evo-Memory's warning about summaries underperforming raw retrieval maps to CMA's careful language: consolidation produces "derived fragments" that "influence future retrieval" — not *replace* detail retrieval. CMA and Evo-Memory agree: abstraction is a routing aid, not a replacement.

**CortexGraph**: CortexGraph's danger-zone blending (surfacing memories about to decay for natural reinforcement) is an instance of CMA's retrieval-driven mutation — retrieval changes the memory's future accessibility. CortexGraph's power-law decay connects to CMA's requirement that dormant memories remain recoverable.

### To Our Architecture Overall

CMA provides the vocabulary we've been building toward. Our system is:
- A **CMA-aspiring** architecture currently at the **enhanced RAG** stage
- On a clear path to CMA compliance through our implementation priority list
- Already satisfying persistence, partial selective retention, and partial retrieval-driven mutation
- Missing associative routing, temporal continuity, and meaningful consolidation

The implementation priorities (#1-#5) are exactly the features needed to cross from "enhanced RAG" to "CMA-compliant." This paper gives us the language to describe what we're building and why, and the behavioral probes to verify we've actually achieved it.

---

## Summary Assessment

This is a definitional paper, not a systems paper. Its contribution is naming and formalizing an architectural class, not proposing novel algorithms. The behavioral checklist and lifecycle framework provide useful vocabulary and evaluation criteria. The experimental evidence is preliminary and non-comparable (custom probes, single implementation, no established benchmarks, GPT-4o judge with limited human spot-checks).

**For claude-memory**: The highest-value takeaway is the six-requirement behavioral checklist as an evaluation framework. The four behavioral probes provide a template for testing our own system as we implement features. The conceptual contributions (activation field, retrieval-driven mutation, dormancy) are useful for design thinking but don't provide concrete implementation details.

**Compared to other sources**: Less directly actionable than Zep (which gave us bi-temporal fields and temporal extraction prompts), Hindsight (which gave us RRF and MPFP), or Engram (which gave us graded contradiction detection). More useful as a *framing* paper — it names what we're building and provides the rubric for evaluating it.

**Net impact**: Framing and evaluation, not implementation. Adds behavioral probes as a new evaluation item (#10). Adds near-miss suppression and co-activation edges as sub-items under existing priorities. Validates our priority ordering. Does not change the implementation sequence.
