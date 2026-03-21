# Kumiho — Graph-Native Cognitive Memory with Formal Belief Revision (arXiv:2603.17244)

*Generated 2026-03-21 by Opus agent reading PDF*

---

## Paper Overview

**Paper**: arXiv:2603.17244v1 (March 2026), 56 pages
**License**: CC BY-NC-ND 4.0
**Author**: Young Bin Park (Kumiho Inc.)
**Code**: Cloud service at kumiho.io; Python SDK, MCP plugin, benchmark suite open-source at github.com/KumihoIO

**Problem addressed**: Agent memory systems have individual components (versioning, retrieval, consolidation) but lack architectural synthesis and formal grounding. Additionally, agents produce work products (code, designs, documents) that accumulate without systematic versioning or provenance — and the structural primitives needed for cognitive memory are identical to those needed for asset management.

**Core approach**: A graph-native cognitive memory architecture (Neo4j + Redis) grounded in formal belief revision semantics (AGM postulates). The central contribution is a correspondence between the AGM belief revision framework and the operational semantics of a property graph memory system. Immutable revisions, mutable tag pointers, typed dependency edges, and URI-based addressing serve both as cognitive memory primitives and as operational infrastructure for multi-agent work product management.

**Key claims**:
- 0.447 four-category token-level F1 on LoCoMo (n=1,540) — highest reported on retrieval categories
- 97.5% adversarial refusal accuracy (n=446)
- 93.3% judge accuracy on LoCoMo-Plus (n=401), vs best baseline Gemini 2.5 Pro at 45.7%
- 98.5% recall accuracy on LoCoMo-Plus; remaining 6.7% gap entirely from answer model fabrication
- 100% pass rate on 49-scenario AGM compliance verification
- Independent partial reproduction by LoCoMo-Plus authors (mid-80% range)

---

## Architecture

### Design Thesis

The paper's central insight: cognitive memory and asset management share identical structural primitives (immutable revisions, typed edges, mutable tag pointers, URI addressing). Rather than building two separate systems, a single graph architecture serves both purposes. The author's VFX pipeline background is explicitly credited as the source of this correspondence — production pipeline asset tracking maps directly to belief revision.

The formal contribution grounds this in AGM belief revision theory, proving satisfaction of postulates K*2–K*6, Relevance, and Core-Retainment, with a principled rejection of Recovery (immutable versioning makes it both unnecessary and semantically wrong) and honest identification of K*7/K*8 as open questions.

### Technical Stack

| Component | Technology |
|-----------|-----------|
| Long-term memory | Neo4j (property graph, cloud-hosted) |
| Working memory | Redis (library-level SDK, <5ms, TTL-based) |
| Embeddings | OpenAI text-embedding-3-small |
| LLM operations | GPT-4o-mini (bulk), GPT-4o (answer model), pluggable adapter |
| Retrieval | Hybrid fulltext + vector, max-based fusion |
| Protocol | MCP (Model Context Protocol) |
| Addressing | URI scheme: `kref://project/space/item.kind?r=N&a=artifact` |

### Graph Schema

**Core entities** (following the structural correspondence):

| Graph Entity | Memory Role | Asset Role |
|-------------|-------------|------------|
| Project | Scope container | Project container |
| Space | Topic namespace | Folder/category |
| Item | Memory unit identity | Asset identity |
| Revision | Belief at time T (immutable) | Version snapshot |
| Tag | Mutable status pointer | Current/approved pointer |
| Edge | Typed relationship | Dependency link |
| Bundle | Memory cluster | Asset collection |
| Artifact | Evidence pointer | File reference |

**Edge types**: `Supersedes` (belief revision), `Depends_On` (dependency), `Derived_From` (provenance), `Related_To` (association). All typed, all traversable for impact analysis.

**Key design choices**:
- Revisions are immutable — content never changes after creation. Belief evolution is modeled through new revisions + `Supersedes` edges.
- Tags are mutable pointers to revisions — the `latest` tag always points to the current belief. Moving a tag is the mechanism for belief revision.
- Deprecated items are excluded from retrieval by default but preserved in the graph. Contraction is behaviorally effective without being informationally destructive.

### Formal Belief Revision (AGM Correspondence)

The paper frames memory operations as belief base operations following Hansson (1999):

| AGM Operation | Graph Implementation |
|---------------|---------------------|
| Revision (B * A) | Create new revision on existing item, re-point tag, create `Supersedes` edge |
| Contraction (B ÷ A) | Remove tag from revision, or soft-deprecate item |
| Expansion (B + A) | Create new revision, add tag, no retraction |

**Proven**: K*2 (Success), K*3 (Inclusion), K*4 (Vacuity), K*5 (Consistency), K*6 (Extensionality), Relevance, Core-Retainment.

**Deliberately violated**: Recovery — contracting then re-expanding doesn't restore co-located beliefs. This is correct: immutable versioning provides a stronger mechanism (explicit rollback via tag reassignment) that doesn't require the system to guess what was lost.

**Open questions**: K*7 (Superexpansion) and K*8 (Subexpansion) — the paper shows they hold for the common case (atomic ground triples) but defers formal proof via the representation theorem. The obstacle is type-dependent entrenchment: preference beliefs should be entrenched by recency, factual beliefs by evidential support, inferred beliefs by confidence. This suggests a type-dependent entrenchment function, which is non-standard in AGM theory.

### Retrieval Pipeline

Hybrid retrieval with deliberate CombMAX fusion:

1. **Fulltext search** (Neo4j fulltext index on summary + tags + metadata)
2. **Vector search** (cosine similarity on text-embedding-3-small embeddings stored as revision properties)
3. **Max-based fusion** — the higher score wins, rather than averaging. Design rationale: a strong exact match on one branch shouldn't be diluted by a weak score on the other. The paper acknowledges this is susceptible to poorly-calibrated retrievers producing inflated scores.

**Client-side LLM reranking**: When recall returns multiple sibling revisions (same item, different versions), the consuming agent's own LLM selects the most relevant one from structured metadata. This costs nothing — the selection is subsumed into the agent's existing inference call. As agent models improve, reranking quality improves automatically.

**Notable limitation**: In the practical evaluation (Section 15.6), an older revision ranked higher than the current revision for "favorite color" due to fulltext keyword matching. The retrieval pipeline doesn't yet incorporate temporal recency as a signal — the agent's prompt instructions handle conflict resolution at the application layer.

### Dream State (Consolidation)

Nine-stage pipeline, event-driven with cursor-based resumption:

1. Ensure cursor (create internal `_dream_state` space/item if needed)
2. Load cursor (persisted position; `None` on first run)
3. Collect events (stream from cursor, dedup by item, latest revision wins)
4. Fetch revisions (batch-load metadata, filter to episodic, exclude deprecated)
5. Inspect bundles (fetch membership for topical grouping context)
6. LLM assessment (batches of 20: relevance scoring 0–1, deprecation recommendations, tag suggestions, metadata enrichment, relationship identification)
7. Apply actions (under safety guards)
8. Save cursor
9. Generate audit report (markdown artifact on `_dream_state` item)

**Safety guards** — the paper's claimed novel contribution over prior consolidation systems:

| Guard | Mechanism |
|-------|-----------|
| Dry run | Assessment-only mode, preview before commit |
| Published protection | Never deprecate items tagged "published" |
| Circuit breaker | Max 50% deprecation per run (configurable 0.1–0.9) |
| Error isolation | Per-action try/except |
| Audit report | Markdown report artifact per run |
| Cursor persistence | Resume from checkpoint after interruption |

The paper is honest that this is an engineering contribution, not a formal one: composing a batch of belief revision operations across multiple memories doesn't provably preserve all AGM postulates simultaneously. Proving compositional preservation is identified as an open problem in belief revision theory.

### Three Architectural Innovations (Driving LoCoMo-Plus Results)

**1. Prospective indexing**: At write time, the LLM generates hypothetical future scenarios ("implications") where this memory might be relevant, and indexes them alongside the summary. This bridges the cue-trigger semantic disconnect that LoCoMo-Plus is designed to stress. On LoCoMo-Plus, this eliminated the >6-month accuracy cliff (37.5% → 84.4%).

**2. Event extraction**: Structured events with consequences are appended to summaries during ingestion. This preserves causal detail that narrative compression would otherwise drop. The paper provides concrete evidence: for a 730-day-gap query, events provided the factual anchor (cottage purchase), implications provided the semantic bridge (financial planning goals). Neither alone would have retrieved across that gap.

**3. Client-side LLM reranking**: Described above — sibling revision selection delegated to the consuming agent's own LLM at zero additional cost.

---

## Key Claims with Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| 0.447 four-category F1 on LoCoMo | n=1,540, token-level F1 with Porter stemming | Credible methodology. Uses the official metric. But: no standardized leaderboard exists — all systems use varying eval configs (different judge models, question subsets, scoring parameters). Direct cross-system comparison is unreliable. |
| 97.5% adversarial refusal | n=446 | Natural consequence of the architecture — memory graph contains no fabricated information, so nothing to hallucinate from. Clean result but tests the architecture's constraint, not a tuned capability. |
| 93.3% on LoCoMo-Plus | n=401, GPT-4o answer model, LLM judge | Partially independently reproduced by LoCoMo-Plus authors (mid-80% range). The paper honestly notes that GPT-family models may score unusually well on this benchmark due to alignment between model family and LLM-assisted benchmark construction. |
| 98.5% recall accuracy | 395/401 on LoCoMo-Plus | The remaining 6.7% gap is attributed to answer model fabrication on correctly retrieved context, not retrieval failure. This decomposition is valuable. |
| 100% AGM compliance (49 scenarios) | Automated test suite, 5 categories, 7 postulates | Good engineering practice. Tests operational adherence to the formal specification. The proofs are theoretical; this verifies the implementation matches. |
| >200K nodes without degradation | LoCoMo-Plus benchmark evaluation | Credible — Neo4j handles this scale routinely. Retrieval precision under adversarial scale (10M+ items, low relevance ratio) is identified as untested. |

**Caveats the paper itself raises** (commendably):
- No standardized LoCoMo leaderboard — all reported numbers use varying configurations
- GPT-family alignment with LoCoMo-Plus benchmark construction
- Retrieval ranking doesn't yet incorporate temporal recency
- K*7/K*8 formally unproven
- Compositional preservation of AGM postulates under batch consolidation is an open problem
- LLM-based PII redaction has non-zero false negative rates

---

## Relevance to Somnigraph

### Ideas worth evaluating

**Prospective indexing.** This is the most actionable idea. At write time, generate hypothetical future scenarios where this memory might be relevant, and index them alongside the content. This directly addresses the cue-trigger semantic disconnect problem — the reason queries fail when the question uses different language than the stored memory. Somnigraph's enriched embeddings (content + category + themes + summary) are a step in this direction, but prospective indexing goes further by imagining the *question* that would need this memory, not just describing the memory's content. The LoCoMo-Plus results (37.5% → 84.4% on >6-month gaps) are compelling evidence.

**Experiment**: During sleep consolidation (or at remember() time), generate 2-3 hypothetical recall queries for each memory and append them to the enriched text before embedding. Measure impact on NDCG@5k. The cost is one LLM call per memory during sleep — already the pattern for Somnigraph's REM classification.

**Event extraction at write time.** Extracting structured events with consequences from memory content preserves causal detail that narrative compression drops. Somnigraph's sleep pipeline does this during consolidation (NREM classification extracts relationships), but doing it at write time means the causal structure is available for retrieval immediately, without waiting for a sleep cycle. This addresses the "new memories don't benefit from graph structure until sleep" gap noted in similar-systems.md.

**Formal belief revision as a design lens.** Somnigraph handles contradictions through NREM classification (supports/contradicts/evolves/duplicates) and contradiction edges. Kumiho's AGM framing provides a more principled vocabulary: revision creates a new version with `Supersedes` edge, contraction removes beliefs from the active state while preserving them for audit, expansion adds without retraction. The practical implications:
- Somnigraph's `forget()` is contraction — the memory should be excluded from retrieval but preserved in the graph. Currently it's a hard delete.
- Somnigraph's contradiction handling could use `Supersedes` semantics: when a new memory contradicts an old one, create a supersession link and deprioritize the old one, rather than just flagging the contradiction edge.
- The Recovery postulate rejection is relevant: when a forgotten memory is re-added, it should be a fresh incorporation, not a rollback. Somnigraph already does this correctly (forget + remember = new memory), but the formal justification is useful documentation.

**Safety guards for consolidation.** Somnigraph's sleep pipeline has no circuit breaker, no dry-run mode, no published-item protection. These are practical engineering safeguards that become important as the system matures:
- Circuit breaker: cap the fraction of memories that can be archived/deprecated per sleep cycle
- Published/pinned protection: memories with high confidence or high access count should be immune to automated consolidation
- Dry-run mode: preview consolidation actions before committing

**CombMAX fusion.** Somnigraph uses RRF for score fusion. Kumiho uses CombMAX (take the higher score from fulltext or vector). The argument: a strong exact match shouldn't be diluted by a weak semantic score. This is worth testing against RRF — particularly for queries where BM25 produces a strong exact match but the vector channel adds noise.

**URI-based addressing.** Every memory has a dereferenceable address (`kref://project/space/item.kind?r=N`). Somnigraph uses UUIDs. URI-based addressing enables deterministic cross-system references, provenance traversal, and makes the memory system more inspectable. Not a retrieval quality improvement, but a valuable architectural pattern for a public research artifact.

### Ideas not worth importing

**Neo4j as storage backend.** Somnigraph's SQLite + sqlite-vec + FTS5 stack is lighter, faster for single-user deployment, and sufficient for the current scale. Neo4j's graph traversal advantages are real but come with infrastructure overhead that doesn't match Somnigraph's design philosophy.

**Redis working memory.** Somnigraph doesn't need a separate working memory layer — the MCP server is stateless and the agent's context window serves as working memory.

**Asset management / multi-agent pipeline infrastructure.** Orthogonal to memory quality. Interesting for enterprise multi-agent workflows but not relevant to Somnigraph's scope.

**The AGM formal proofs themselves.** The proofs are correct and intellectually satisfying, but Somnigraph's practical needs are served by the design patterns the proofs justify (supersession, soft deprecation, principled Recovery rejection), not by carrying the formal machinery into the codebase.

**Client-side LLM reranking of sibling revisions.** Somnigraph doesn't have a multi-revision model — each memory is atomic, not versioned. The LightGBM reranker handles scoring; delegating to the consuming LLM would lose the feedback-trained ranking signal.

### Convergent patterns

| Pattern | Kumiho | Somnigraph |
|---------|--------|------------|
| Hybrid retrieval | Fulltext + vector, CombMAX | BM25 + vector, RRF |
| Consolidation | 9-stage Dream State, LLM-assessed | 10-step REM, LLM-classified |
| Soft deletion | Deprecated items excluded from retrieval, preserved in graph | Archived memories excluded from retrieval |
| Contradiction handling | `Supersedes` edge + tag re-pointing | Contradiction edges flagged during NREM |
| Enriched storage | Summary + tags + metadata + implications + events | Content + category + themes + summary |
| LLM-decoupled design | Pluggable LLM adapter, model-independent graph | OpenAI embeddings configurable, LLM used only in sleep |

---

## Code Quality Assessment

This is a paper, not an open-source implementation (the core graph server is a cloud service). The SDK, MCP plugin, and benchmark suite are open-source but the memory engine itself is proprietary. This limits reproducibility and independent verification of architectural claims. The paper is well-written, unusually honest about limitations, and provides the most rigorous formal treatment of belief revision in the agent memory literature.

The LoCoMo-Plus results include independent partial reproduction by the benchmark authors — unusual and commendable. The paper explicitly questions whether GPT-family alignment with the benchmark construction biases absolute scores — a level of self-critique rarely seen.

---

## Summary

Kumiho is the most formally rigorous agent memory system in Somnigraph's research corpus. The AGM belief revision correspondence is a genuine intellectual contribution that provides principled vocabulary for operations all memory systems perform (revision, contraction, expansion). The LoCoMo/LoCoMo-Plus results are strong, with appropriate caveats about cross-system comparability.

For Somnigraph, the most actionable ideas are: (1) prospective indexing (generating hypothetical future queries at write/sleep time to bridge the cue-trigger gap), (2) event extraction at write time for immediate causal structure, (3) consolidation safety guards (circuit breaker, dry-run, pinned protection), and (4) the formal vocabulary for justifying existing design decisions (forget-as-contraction, supersession-over-deletion, Recovery rejection). The supersession pattern converges with memv and AWM — three independent systems arriving at the same mechanism increases confidence.
