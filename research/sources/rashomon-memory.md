# Rashomon Memory — Argumentation-Driven Retrieval for Multi-Perspective Agent Memory

*Generated 2026-04-11 by Opus agent reading arXiv:2604.03588*

---

## Paper Overview

**Citation:** Albert Sadowski, Jaroslaw A. Chudziak. "Rashomon Memory: Towards Argumentation-Driven Retrieval for Multi-Perspective Agent Memory." arXiv:2604.03588, April 2026. Warsaw University of Technology, Poland. 12 pages (10 main + 2 references).

**Code:** https://github.com/albsadowski/rashomon-memory-demo

**Problem addressed:** Current memory architectures for LLM agents assume a single correct encoding of experiences, or at best provide multiple views over unified storage. When the same event serves multiple concurrent goals (e.g., a price concession is simultaneously a trust signal and a margin erosion), these systems commit to one interpretive frame at encoding time and lose the alternatives. This is simultaneously a memory problem (maintain competing framings) and an explainability problem (surface which framing was selected and why).

**Core approach:** Parallel goal-conditioned agents each maintain their own ontology and knowledge graph. At query time, perspectives propose interpretations, critique each other via distributed peer critique using asymmetric domain knowledge, and Dung's argumentation semantics (grounded extension) resolves which proposals survive. The resulting attack graph is itself the explanation — it records what was selected, what alternatives existed, and why they were rejected.

**Key claims:**
1. Retrieval modes (selection, composition, conflict surfacing) emerge from attack graph topology without being designed directly (Sec. 2.4, Table 2)
2. Attack graph topology is query-dependent, not perspective-dependent — the same perspectives produce different topologies for different queries (Sec. 3, Table 2)
3. The conflict surfacing mode (empty grounded extension) is the distinctive contribution: the system reports genuine disagreement rather than forcing resolution (Sec. 2.4, 5)
4. Contrastive explanations emerge from the argumentation structure without a separate explanation generation step (Sec. 2.3)

---

## Architecture

### Storage & Schema

Each goal-perspective maintains a separate OWL/RDF knowledge graph in Turtle format. No shared storage — perspectives have separate memories with separate vocabularies, shaped by separate goals (Sec. 3). The schema uses the LLMs4OL methodology (Hamed et al., 2023): term typing (Task A) classifies entities against the perspective's TBox, taxonomy discovery (Task B) places new classes, and relation extraction (Task C) identifies properties as RDF triples added to the ABox along with any TBox extensions.

Ontologies are living structures. Three construction paths: (1) seeded from domain taxonomies (e.g., Risk Management from compliance frameworks), (2) emergent from encoding patterns when agents struggle to categorize, (3) evolved through use as domains shift (Sec. 2.2). The paper acknowledges but does not solve the retroactive reinterpretation problem: when ontologies evolve, old encodings may need re-encoding under the new lens (Sec. 4).

### Memory Types

The paper does not use the standard episodic/semantic/procedural taxonomy. Instead, all memories are encoded as RDF triples within perspective-specific knowledge graphs. The differentiation is not by memory type but by perspective: the same raw observation produces different triples in different graphs (Figure 1). For example, the 15% discount concession produces "trust signal, guarded, low reciprocity" in Relationship Strategy vs. "pricing precedent, moderate severity" in Risk Management vs. "margin impact -$340K, significant deviation" in Financial Planning.

### Write Path

Two-stage pipeline per perspective (Sec. 3):

1. **Relevance filter:** Each perspective independently decides whether an observation matters for its goals. Of 24 possible encodings (8 observations x 3 perspectives), only 13 passed (Table 1). Each perspective encoded 5/8 observations but different subsets.

2. **Goal-conditioned encoding:** Observations that pass are encoded via separate LLM calls with goal-specific system prompts and structured output schemas. Each call returns both ontology updates (TBox extensions) and instance triples (ABox additions). Temperature set to 0.

The write path is entirely parallel — no coordination between perspectives during encoding. The encoding pipeline for the demonstration required 24 relevance checks and 13 encoding calls (37 LLM invocations total for 8 observations).

### Retrieval

Retrieval is negotiation, not search (Sec. 2.3). Four-phase protocol:

1. **Broadcast:** The arbiter broadcasts the query with context (querier identity, decision type, current priorities) to all perspectives.

2. **Proposal:** Each perspective with relevant encodings searches its knowledge graph and submits an interpretation with a relevance claim.

3. **Peer critique (attacks):** Proposals are shared among perspectives. Each may submit attacks on others' proposals. An attack claims that, given the query context, the attacking perspective's framing should dominate. Crucially, each perspective evaluates others from its own domain expertise — Risk Management attacks Relationship Strategy using compliance and precedent knowledge that Relationship Strategy never stored. Attacks must be selective (not everything) and contextual (same perspectives might or might not attack each other depending on the query).

4. **Resolution:** The arbiter constructs a directed attack graph and computes Dung's grounded extension — the unique maximal set of arguments that are (a) unattacked or (b) only attacked by arguments that are themselves defeated. This is deterministic given the attack graph.

Each query required up to 7 LLM invocations: 3 proposals, 3 attack evaluations, 1 response assembly (Sec. 3).

### Retrieval Modes

Three modes emerge from the grounded extension topology (Sec. 2.4, Table 2, Figure 3):

| Mode | Grounded Extension | Meaning |
|------|-------------------|---------|
| **Selection** | Single perspective survives | One framing dominates; others defeated with reasons |
| **Composition** | Multiple perspectives survive | Complementary contributions; structured aggregation of distinct viewpoints |
| **Surfacing** | Empty extension (complete mutual attack) | Genuine irreconcilable conflict; system reports the disagreement rather than resolving it |

The surfacing mode is the paper's distinctive contribution. When all perspectives attack each other and none can be defended, the system presents all framings as legitimate alternatives with the attack graph available for inspection. The authors cite Miller (2019): human-interpretable explanations are contrastive ("why P rather than Q?" not "why P?"), and the attack graph answers exactly this.

### Consolidation / Processing

No offline consolidation step. The paper proposes retrieval-driven curation as future work: encodings that are frequently retrieved gain importance weight, those never retrieved decay (Sec. 2.2). This is described as a stigmergic dynamic analogous to ant pheromone trails, citing Kirkpatrick et al.'s elastic weight consolidation as a possible mechanism. The observation buffer (raw experience staging area) faces a retention policy question the paper explicitly defers (Sec. 2.1).

### Lifecycle Management

Not addressed. The paper acknowledges ontology evolution as an open problem (Sec. 4) — when goals shift, the vocabulary of past encodings becomes stale. It raises the retroactive reinterpretation question (can a perspective re-encode past experiences under an updated ontology?) and the fabrication risk (what prevents perspectives from fabricating convenient histories?), but proposes no solutions.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Retrieval modes emerge from attack graph topology | Table 2: 4 queries produce selection, composition (x2), and surfacing modes from the same 3 perspectives | **Demonstrated but not validated.** The 4-query demonstration is hand-chosen and not compared to any baseline. The emergence is real but trivially achievable at this scale — 3 perspectives and 4 queries could be resolved by simpler heuristics (the authors acknowledge this, p. 7). |
| Attack graph topology is query-dependent, not perspective-dependent | Table 2, Figure 3: same 3 perspectives produce 4 distinct topologies (0, 2, 3, and 6 attacks) across 4 queries | **Convincing as a demonstration.** The variation is genuine — perspectives don't have fixed dominance relationships. Whether this holds at scale with more perspectives and diverse queries is untested. |
| Conflict surfacing is preferable to forced resolution | Argued from Miller [14] (contrastive explanations) and Kurosawa's Rashomon (irreconcilable accounts) | **Theoretical argument only.** No user study. The paper explicitly notes this (p. 9): "We have not conducted user studies to evaluate whether this form of explanation improves decision quality." |
| Distributed peer critique produces grounded attacks | Demonstration shows Risk Management attacking Relationship Strategy using compliance knowledge RS never stored (p. 4) | **Plausible mechanism.** The asymmetric knowledge property is the key design insight — attacks are grounded in domain expertise, not generic disagreement. But the demonstration uses only 3 hand-crafted perspectives with clear domain boundaries. Whether LLM-generated attacks remain grounded at scale is an open question; multi-agent debate systems suffer from semantic drift [36]. |
| No quantitative evaluation against baselines | Explicitly stated: "We provide no quantitative comparison with baseline retrieval methods such as RAG over raw observations" (p. 9) | **Major limitation.** The paper is a proof-of-concept demonstrating architectural feasibility, not retrieval quality. |

---

## Relevance to Somnigraph

### What Rashomon Memory does that Somnigraph doesn't

1. **Multi-perspective encoding.** Somnigraph stores one encoding per memory. Rashomon stores N parallel encodings of the same experience, each shaped by a different goal. This is a fundamentally different data model — the raw experience is not the memory; the goal-filtered interpretation is.

2. **Retrieval-time argumentation.** Somnigraph's contradiction edges are detected during offline sleep consolidation and passively stored. They do not participate in retrieval-time negotiation. Rashomon's entire retrieval mechanism is negotiation — contradictions are discovered and resolved (or surfaced) dynamically per-query. This means the same memories can produce different conflict structures depending on what's being asked, which static contradiction edges cannot.

3. **Contrastive explanation.** Somnigraph returns ranked results. Rashomon returns ranked results plus a structured explanation of why alternatives were rejected, grounded in the attack graph. The explanation is a byproduct of the retrieval mechanism, not a separate generation step.

4. **Conflict surfacing mode.** When memories genuinely contradict, Somnigraph has no mechanism to present the conflict as a structured object. It will return both sides ranked by the reranker, but the user sees a list, not a disagreement. Rashomon's surfacing mode explicitly presents irreconcilable perspectives as legitimate alternatives.

### What Somnigraph does better

1. **Retrieval quality at scale.** Somnigraph has a 26-feature learned reranker trained on 1032 real queries, benchmarked on LoCoMo (85.1% QA accuracy, R@10=95.4% with graph augmentation). Rashomon has no quantitative evaluation at all — 4 hand-selected queries over 8 observations with 3 perspectives.

2. **Computational efficiency.** Somnigraph retrieval is a single forward pass through BM25 + vector search + reranker (milliseconds). Rashomon requires O(N) LLM calls per query for proposals plus O(N) for attack evaluation plus assembly — 7 LLM calls for 3 perspectives. At 10 perspectives this becomes 101 calls (p. 9). This is infeasible for interactive use.

3. **Offline consolidation.** Somnigraph's sleep pipeline (NREM relationship detection, contradiction classification, temporal evolution, summary generation) performs the expensive LLM work offline. Rashomon defers all inferential work to query time, making encoding cheap but retrieval expensive.

4. **Feedback loop.** Somnigraph's explicit retrieval feedback (utility ratings) and UCB exploration bonus create a learning signal. Rashomon treats retrieval as read-only — no mechanism for the system to learn which perspectives or retrieval modes were useful.

5. **Single-user coherence.** Somnigraph's single-user, single-perspective design avoids the ontology evolution, perspective boundary, and fabrication risks that Rashomon acknowledges but doesn't solve (Sec. 4). These are real problems, not theoretical — the paper's own discussion of retroactive reinterpretation (p. 9) suggests that multi-perspective systems may need to fabricate histories when ontologies shift.

6. **Contradiction detection.** Somnigraph detects contradictions during sleep consolidation and stores them as typed edges. Performance is poor across all systems (0.025-0.037 F1 on LoCoMo), but the mechanism exists. Rashomon detects contradictions at query time through peer critique, which is more dynamic but also more expensive and untested at scale.

---

## Worth Stealing (ranked)

### 1. Conflict surfacing as a retrieval mode (medium effort)

**What:** When memories with contradiction edges are retrieved together, surface the conflict as a structured object rather than just returning both ranked by score. Present the contradiction with the competing claims and the grounds for disagreement.

**Why:** Somnigraph already detects contradictions during sleep and stores them as typed edges. But these edges are invisible at retrieval time — they exist in the graph but don't change how results are presented. The surfacing insight is that some queries have no single right answer, and the system should say so rather than silently ranking one side higher.

**How:** In `impl_recall()`, after reranking, check whether any top-K results are connected by contradiction edges. If so, annotate the response with the conflicting claims and the contradiction relationship. This doesn't require multi-agent negotiation — it's a post-processing step on existing graph data. The hard part is the presentation format: how does an MCP tool communicate "these two memories disagree" to the consuming LLM?

### 2. Goal-conditioned relevance filtering (low effort, high conceptual value)

**What:** The insight that encoding should be parameterized by goals — not just "is this worth remembering?" but "is this worth remembering for this purpose?" Different goals filter differently (Table 1: each perspective encoded 5/8 observations but different subsets).

**Why:** Somnigraph's `remember()` stores everything the user asks to store, with category/priority as metadata. The system never asks "is this memory relevant to your current goals?" This matters less in a single-user personal memory system than in a multi-agent negotiation tool, but the principle applies to retrieval: boost_themes is a weak version of goal-conditioned retrieval, and the reranker's query features (query_coverage, query_idf_var) capture some of this signal.

**How:** This is more of a conceptual note than an implementation task. The existing boost_themes mechanism already provides soft goal-conditioning at retrieval time. Making it stronger (e.g., allowing the user to specify "recall for the purpose of X") would be a natural extension but adds complexity to the MCP interface.

### 3. Contrastive explanation from graph structure (high effort)

**What:** When the system selects one memory over a related alternative (e.g., an evolved version over its predecessor), explain why — not as a separate generation step but as a byproduct of the graph structure (evolution edges, contradiction edges, revision edges already encode the relationship).

**Why:** Somnigraph has the graph edges to support this. When a memory has an `evolves` or `revision` edge, the system could note "this supersedes [earlier memory] because [evolution reason]." The explanation is already in the edge metadata — it just isn't surfaced.

**How:** Requires changes to the recall response format and the consuming LLM's ability to use graph context. The reranker already deprioritizes superseded memories (via graph features), but it does so silently. Surfacing the reason adds transparency at the cost of response verbosity. Probably best implemented as an optional verbose mode rather than default behavior.

---

## Not Useful For Us

- **Multi-agent encoding architecture.** Somnigraph is single-user, single-perspective. Maintaining N parallel knowledge graphs with separate ontologies is architecturally orthogonal to our design. The O(N^2) LLM cost at query time is prohibitive for interactive use.

- **OWL/RDF ontology management.** Somnigraph uses SQLite + sqlite-vec + FTS5. The ontology evolution problem (Sec. 4) — how to re-encode past experiences when the schema changes — is a real and unsolved problem, but it's specific to the OWL/RDF representation. Our schema (memories with typed edges) is simple enough that evolution happens through new edges and memory updates, not ontology migration.

- **Dung's argumentation semantics.** The formal framework is elegant but overkill for single-perspective retrieval. The grounded extension computation is polynomial in N (perspectives), but N=1 in our case. The value of the formalism is in guaranteeing uniqueness of the resolution — but with a single perspective, there's nothing to resolve.

- **Distributed peer critique.** The asymmetric-knowledge attack mechanism is the paper's most interesting technical contribution, but it requires multiple domain-specialized agents. In a single-user system, there's no natural analog — you'd need to synthesize multiple perspectives, which reintroduces the computational cost without the architectural benefit of genuinely separate knowledge bases.

---

## Connections

- **contradiction-reconciliation.md:** Chan et al. (2026) address the same problem space (contradictions in NLI) but from the opposite direction — they try to generate reconciling explanations that make both sides compatible, while Rashomon surfaces the irreconcilable disagreement. The two approaches are complementary: reconciliation for resolvable contradictions, surfacing for genuine conflicts.

- **Somnigraph contradiction edges:** Somnigraph's sleep pipeline detects contradictions and stores them as typed edges, but does nothing with them at retrieval time. Rashomon's surfacing mode is the missing retrieval-time use of this information. The "Worth Stealing #1" proposal directly addresses this gap.

- **PPR graph expansion:** Somnigraph's PPR-based graph expansion already traverses typed edges during retrieval. The existing infrastructure could support contradiction-aware retrieval without the full multi-agent framework — PPR could be configured to flag rather than traverse contradiction edges.

- **Minsky's Society of Mind [13]:** The paper frames multi-goal agents as societies of agents with competing priorities, citing Minsky's argument that minds are better understood this way. This resonates with the broader question of whether single-agent memory systems are fundamentally limited — but the practical answer for Somnigraph is that single-perspective with graph augmentation achieves strong retrieval quality (R@10=95.4%) without the multi-agent overhead.

- **graphiti.md, hexis.md:** Other systems with knowledge graph foundations. Graphiti uses temporal knowledge graphs with unified encoding; Hexis uses precomputed neighborhoods. Neither addresses multi-perspective encoding or retrieval-time negotiation.

---

## Summary Assessment

**Relevance:** Medium. Rashomon Memory addresses a genuine gap in memory system design — the inability to maintain and surface competing interpretations of the same experience. The conceptual contribution (conflict surfacing as a retrieval mode, explanation as a byproduct of argumentation) is more valuable than the implementation, which is a small-scale proof-of-concept with no quantitative evaluation.

**Maturity:** Very early. 12-page preprint with a 4-query demonstration over 8 observations and 3 hand-crafted perspectives. No baseline comparison, no user study, no scalability analysis beyond cost estimates. The code is public but the demo is a controlled scenario, not a general system.

**For Somnigraph:** The actionable insight is not the multi-agent architecture (which is orthogonal to our design) but the retrieval-time use of contradiction information. Somnigraph already detects contradictions during sleep and stores them as graph edges — but these edges are currently invisible during retrieval. The surfacing concept could be implemented as a lightweight post-processing step over existing contradiction edges, without any of the multi-agent machinery. This is a concrete, bounded improvement to an existing capability gap.

**Key limitation the paper acknowledges:** The architecture has clear boundaries — factual recall, ground-truth domains, and latency-sensitive applications are explicitly out of scope (Sec. 4). The approach is best suited for "deliberative settings, such as negotiation preparation, strategic planning, and post-mortem analysis, where the cost of a richer explanation is justified by the stakes of the decision." This is a much narrower use case than general-purpose agent memory.
