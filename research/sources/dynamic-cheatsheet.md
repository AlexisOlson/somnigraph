# Dynamic Cheatsheet -- Analysis

*Generated 2026-02-18 by Opus agent reading 2504.07952v1*

---

## Paper Overview

**Paper**: Suzgun, Yuksekgonul, Bianchi, Jurafsky, Zou (2025). Stanford + Together AI. arXiv:2504.07952v1, Apr 10, 2025.

**Problem addressed**: LLMs operate in a vacuum -- each query is processed independently, with no retention of insights from prior attempts. The same strategies are re-derived and the same mistakes re-committed. Dynamic Cheatsheet (DC) gives a black-box LLM a persistent, evolving external memory that accumulates strategies, code snippets, and heuristics at inference time, without any parameter updates.

**Core claim**: By maintaining a curated "cheatsheet" of reusable insights, LLMs can improve substantially on sequential tasks. Claude 3.5 Sonnet doubled its AIME accuracy (23% to 50%), GPT-4o went from 10% to 99% on Game of 24, and GPQA-Diamond saw a 9% lift. These gains come without ground-truth labels or human feedback -- the model self-curates.

**Scale**: 22-page paper. 8 benchmarks tested across 4 models. Two DC variants (DC-Cu and DC-RS). Compared against 4 baselines. Includes full prompt templates in appendix.

---

## Architecture / Method

### Two Variants

**DC-Cu (Cumulative)**:
1. Generator receives query `x_i` + current memory `M_i`
2. Generator produces answer `y_i`
3. Curator updates memory: `M_{i+1} = Cur(M_i, x_i, y_i)`
4. No retrieval step -- memory grows cumulatively

**DC-RS (Retrieval & Synthesis)**:
1. Retriever finds top-k (k=3) similar past queries using cosine similarity on OpenAI `text-embedding-3-small`
2. Curator refines memory *before* generation: `M_i = Cur(M_{i-1}, x_i, R_i)`
3. Generator produces answer using updated memory: `y_i = Gen(x_i, M_i)`
4. Key difference: memory update happens *before* the response, incorporating new context proactively

### The Curator's Job

The curator (same LLM, different prompt) evaluates:
- **Usefulness and generalizability** of the new answer
- **Refinement or removal** of existing entries (superseded by better strategies)
- **Clarity and compactness** of the entire memory

No ground-truth labels. The model assesses its own output quality.

### Memory Structure

The cheatsheet is a structured document (~2000-2500 words) organized into sections:
- Reusable Code Snippets and Solution Strategies
- General Problem-Solving Heuristics
- Optimization Techniques & Edge Cases
- Specialized Knowledge & Theorems

Each entry is a `<memory_item>` with:
- `<description>`: problem context, purpose, key aspects, reference tags (Q1, Q14, etc.)
- `<example>`: worked solution, code snippet, or strategy
- `Count`: usage counter (incremented when a strategy is successfully reused)

### Baselines

| Method | Memory | Retrieval | Curation |
|--------|--------|-----------|----------|
| BL (Baseline) | None | None | None |
| DC-empty | None (structured prompt only) | None | None |
| FH (Full History) | Append all | None | None |
| DR (Dynamic Retrieval) | Verbatim past pairs | Yes (top-k) | None |
| DC-Cu | Curated cheatsheet | None | Yes (post-generation) |
| DC-RS | Curated cheatsheet | Yes (top-k) | Yes (pre-generation) |

---

## Key Claims & Evidence

### 1. Curated memory massively outperforms full history

On AIME 2024 with GPT-4o: FH scored 13.3% (below baseline 20%), while DC-RS hit 40%. Full history *hurts* -- it overwhelms context, dilutes signal, and increases cost. This is the paper's strongest result for our purposes: naive accumulation degrades performance.

### 2. The "code discovery" effect

The most dramatic gains come when the model discovers that code solves a class of problems. GPT-4o on Game of 24: once it found a Python brute-force solver and stored it, accuracy went from 10% to 99%. Math Equation Balancer: 50% to 100% via code. This is not general reasoning improvement -- it is the model learning to delegate to tools.

### 3. DC helps most with structurally similar tasks

Game of 24 (100% structural overlap): 10% to 99%. AIME (partial overlap): 20% to 40%. GPQA-Diamond (diverse domains): 57% to 58% for GPT-4o. The more homogeneous the task distribution, the more memory helps. This aligns exactly with Evo-Memory's task similarity correlation (r=0.717).

### 4. Smaller models cannot benefit

GPT-4o-mini and Claude 3.5 Haiku showed minimal or negative gains. Two failure modes:
- **Generative competence**: base model doesn't produce enough correct solutions to seed quality memory
- **Curation ability**: smaller models can't effectively curate, retrieve, or synthesize from stored memory

DC amplifies existing capability -- it cannot compensate for foundational gaps.

### 5. DC outperforms majority voting

On AIME 2024, MV(BL) = 23.3% (same as baseline), while DC-Cu = 50.0%. Passive statistical aggregation of multiple attempts is far less effective than active knowledge accumulation. Memory-based adaptation provides a qualitatively different improvement pathway.

### 6. Retrieval noise is a real problem

GPT-4o's GPQA-Diamond performance occasionally *dipped* under DC-RS due to "suboptimal retrieval choices." With diverse topics, poor retrieval can introduce confusion. The paper explicitly calls for "robust retrieval methods (e.g., dense vector search, advanced ranking algorithms)" and proposes "hierarchical and modular memory" to isolate errors within domains.

### 7. Long-context generation is a bottleneck

Models sometimes abbreviate or reference prior memory rather than fully rewriting it ("Previous content [...] preserved"). This causes memory quality to degrade over time as entries become increasingly abbreviated. The paper suggests maintaining a "structured, external database that the LM can reference without regenerating large swaths of text each time."

---

## Relevance to claude-memory

### The Summary Layer Question (Critical)

This is the paper that Evo-Memory flagged as a WARNING. Now that we have the full picture, the warning is more nuanced than "summaries bad":

**When DC summaries HELP:**
- Tasks with high structural similarity (Game of 24, Math Equation Balancer, AIME)
- When the cheatsheet captures *transferable strategies* (not just facts)
- When the curated summary is genuinely more compact and actionable than raw history
- When code snippets can be stored and reused verbatim

**When DC summaries HURT or stall:**
- Diverse knowledge domains (GPQA-Diamond for GPT-4o: virtually no gain)
- When retrieval noise introduces irrelevant strategies (DC-RS < DR on some tasks)
- When the model's curation quality is low (smaller models, verbose reasoning models like R1)
- When memory entries become abbreviated through repeated rewriting

**The precise lesson for our summary/gestalt layers:** DC's failure mode isn't that summaries are bad. It's that summaries *replace* detail. The DC cheatsheet is the *only* memory -- there's no detail tier underneath. When it works (homogeneous tasks), the summary IS the useful knowledge. When it fails (diverse domains), you needed the specific detail that got compressed away.

Our layered architecture (detail -> summary -> gestalt) avoids this by design, *as long as*:
1. Summaries point to details but don't replace them
2. Retrieval can bypass summaries and hit details directly
3. The gestalt layer is a routing aid ("here's what I know about chess") not a knowledge store
4. Summary generation preserves or links to the specific examples/evidence that generated it

### Curator Design Informs Sleep Skill

DC's curator prompt (Figures 14-15) is the most detailed specification of a memory curation process in any paper we've reviewed. Key design choices:

- **Usage counter per entry** -- entries track how often they've been successfully applied. This is our mutation log in miniature.
- **Reference tagging** -- entries cite which queries (Q1, Q14, Q22) contributed to them. This is provenance tracking.
- **Explicit instruction to preserve prior content** -- "any previous content not directly included will be lost and cannot be retrieved." This is the fatal flaw: the entire memory must be rewritten each time, causing information loss.
- **Size cap** -- "circa 2000-2500 words." Forces compression. Our system has no such cap, which is an advantage.

### DC's Architecture vs. Ours

| Dimension | DC | claude-memory |
|-----------|----|----|
| Memory structure | Single flat document, rewritten each step | Database with typed entries, layers, edges |
| Curation | LLM rewrites entire memory | Human-curated, LLM-assisted at sleep |
| Retrieval | Cosine similarity on raw queries (DC-RS) | Hybrid vector + BM25 + RRF (planned) |
| Persistence | Session-scoped (no cross-session) | Permanent with decay |
| Detail preservation | None -- compressed away | Detail tier always accessible |
| Contradiction handling | Implicit (curator may overwrite) | Graded detection (from Engram) |
| Temporal awareness | Usage counter only | Created_at, accessed_at, decay model |
| Scale | ~2500 words max | 438+ memories, unlimited growth with decay |

---

## Worth Stealing (Ranked)

### 1. Usage Counter on Memory Entries (HIGH value, LOW effort)

Each `memory_item` tracks how many times it has been successfully applied. This is cheap to implement (increment on retrieval + confirmed use) and provides direct signal for:
- Priority adjustment: frequently-used memories deserve higher priority
- Pruning: zero-use memories after N sleep cycles are candidates for archival
- Diagnostic: which categories of memory actually get used?

**Implementation**: Add `use_count integer DEFAULT 0` to memories table. Increment on `recall()` when returned memories are subsequently referenced in the session. Use as input to decay/priority calculations during `/sleep`.

### 2. Curator's "Generalizability Test" (MEDIUM value, MEDIUM effort)

DC's curator explicitly evaluates whether a new insight is *generalizable* vs. *problem-specific*. Problem-specific details are discarded. For us, this is relevant to `remember()`: when storing a new procedural memory, assess whether it captures a transferable pattern or a one-off solution. One-off solutions get lower priority.

**Implementation**: During sleep's relationship detection phase, flag procedural memories that have no structural similarity to other memories in the same category. These may be too specific to justify their priority.

### 3. Pre-Generation Memory Update (MEDIUM value, requires architecture change)

DC-RS updates memory *before* generating a response, incorporating the new query context. DC-Cu updates *after*. DC-RS generally outperforms DC-Cu on diverse-domain tasks (GPQA: 68.7% vs 61.1%). The insight: seeing the upcoming problem helps the memory system surface the right content.

**Relevance**: Our `recall()` already works this way -- you provide a query and get relevant memories before responding. But `startup_load()` doesn't know what the session will be about. Consider: if the first user message contains topic signals, a quick targeted recall could supplement startup_load.

### 4. Structured Cheatsheet Sections as Summary Organization (LOW value, design insight)

DC organizes its cheatsheet into named sections (Code Snippets, Heuristics, Optimization, Specialized Knowledge). This maps loosely to our category system but suggests that summary-layer memories in `/sleep` should be organized by function, not just by topic. A "procedural summaries" section and a "semantic summaries" section might be more useful than topic-based grouping.

### 5. Reference Tagging with Source Queries (LOW value, LOW effort)

DC tags entries with which queries contributed to them (Q1, Q6, Q14). This is lightweight provenance tracking. We could extend `generated_from` on summary/gestalt memories to include not just source memory IDs but brief descriptions of what sessions or queries produced them.

---

## Not Useful For Us

### 1. The Single-Document Memory Model

DC's entire memory is one document rewritten in full each step. This is fundamentally limited: it forces lossy compression, creates the abbreviation problem, and caps storage at ~2500 words. Our database-backed approach with typed entries, decay, and layers is categorically superior. Nothing to adopt here.

### 2. Self-Assessment Without Ground Truth

DC's curator must judge its own outputs' correctness without labels. This works for math (checkable) and code (executable), but is unreliable for open-ended domains. Our system doesn't face this problem -- human curation provides the ground truth signal. We should not move toward self-assessed memory quality.

### 3. Task-Homogeneous Benchmark Design

The benchmarks (Game of 24, AIME, equation balancing) have high structural similarity by design. DC's impressive numbers (10% to 99%) come from this similarity. Our memory serves an open-ended, maximally diverse use case. DC's headline results don't transfer to our setting.

### 4. Sequential Task Stream Assumption

DC processes tasks in a linear sequence, with memory evolving monotonically. Real sessions are not sequential streams of similar problems. Our memory must handle interleaved topics, long gaps between related interactions, and revisiting old topics after months. DC's online-learning formalization doesn't capture this.

### 5. Reasoning Model (R1/o1) Results

DC showed "minimal or inconsistent improvements" with reasoning-heavy models. Their solutions were "far too verbose and long" for effective curation. This is relevant trivia but doesn't affect our design -- we're not using DC's curation mechanism.

---

## Impact on Implementation Priority

No changes to the ranked priority list from [[systems-analysis]]. DC reinforces existing decisions rather than introducing new requirements:

| Priority | Feature | DC's Impact |
|----------|---------|-------------|
| 1 | RRF fusion | DC explicitly calls for "advanced ranking algorithms." Validates. |
| 2 | Relationship edges | DC has no edges -- and suffers for it. Validates. |
| 3 | Graded contradiction | DC's curator implicitly handles contradictions by overwriting, losing history. Our graded approach is better. |
| 4 | Decay + power-law | DC has no decay -- cheatsheet just grows to the word limit then compresses. Our decay model is an advantage. |
| 5 | Sleep skill | **Strongest validation.** DC's curator IS a primitive sleep process. It runs after every query, evaluates what to keep/discard/generalize, and rewrites the memory. Our /sleep is a more sophisticated version of this. |
| 6 | Mutation log | DC's usage counter is a minimal mutation log. Add use_count as described above. |
| 7 | Confidence scores | DC has no confidence model. No impact. |
| 8 | Reference index | DC has no reference index. No impact. |
| 9 | Multi-angle indexing | DC uses single embedding for retrieval. Their retrieval noise problem is partly caused by this. Validates multi-angle approach. |

**The one new insight that should influence sleep-skill design**: DC's usage counter. Cheap, informative, directly useful for priority adjustment. Add to Phase 1 of sleep implementation.

---

## Connections

### To Evo-Memory (arXiv:2511.20857)

Evo-Memory tested DC directly and found it sometimes underperforms ExpRAG (simple structured retrieval). The full DC paper reveals why: DC replaces detail with summary. When the summary captures the right abstraction (homogeneous tasks), it wins. When it doesn't (diverse tasks), detail retrieval wins. This is NOT "summaries are bad" -- it's "summaries without a detail fallback are fragile." Our layered architecture with detail tier preserved is the correct response.

Evo-Memory's task-similarity correlation (r=0.717) is exactly replicated in DC's results: Game of 24 (homogeneous) sees massive gains; GPQA-Diamond (diverse) sees minimal gains for GPT-4o.

### To Generative Agents (Park et al.)

DC's curator is a per-query version of Generative Agents' reflection process. GA uses an importance trigger (sum > 150) to batch reflections; DC reflects on every query. GA produces higher-order "reflection" memories that cite source memories; DC produces a revised cheatsheet that loses its sources. Our `/sleep` should follow GA's model (periodic, explicit, preserving provenance) rather than DC's (continuous, lossy).

### To CogMem (Oberauer model)

CogMem's "summary-first, detail-on-demand" retrieval pattern directly addresses DC's failure mode. When CogMem retrieves, it starts at the summary level and drills down to detail when needed. DC has no drill-down -- the cheatsheet IS the memory. Our implementation should follow CogMem's retrieval pattern for the summary/gestalt layers.

### To MT-DNC (Zhao et al.)

MT-DNC's write protection for high-tier memories contrasts with DC's approach of rewriting the entire memory each step. DC's abbreviation problem (models truncating prior content during rewrites) is exactly the failure that write protection prevents. Our `/sleep` should not regenerate summary/gestalt layers unless the underlying details have actually changed.

### To Continuum (CMA requirements)

DC satisfies CMA requirements partially:
- Persistent storage: Yes (within session)
- Selective retention: Yes (curator filters)
- Associative routing: Partial (cosine retrieval only)
- Temporal chaining: No
- Consolidation: Yes (curator is consolidation)
- Cross-session: No

Our system aims to satisfy all six.

### To Zep (bi-temporal)

DC has no temporal model at all. Usage counters are the closest thing to temporal awareness. Zep's bi-temporal modeling (when fact was true vs. when memory was stored) would have helped DC handle the GPQA-Diamond retrieval noise problem -- temporal context would filter out irrelevant strategies from different knowledge domains.

### To Hindsight (RRF)

DC uses single-path cosine similarity retrieval with k=3. The paper explicitly acknowledges retrieval noise as a problem and calls for "dense vector search, advanced ranking algorithms." Hindsight's multi-path retrieval with RRF fusion is the direct answer to DC's retrieval limitation.

---

## Summary of Key Takeaways

1. **Curated summaries > raw history** -- but only when the summary captures the right abstraction. Full history (FH) actively hurts performance.
2. **Summaries without detail fallback are fragile** -- this is the core lesson. Our layered architecture must preserve detail-tier accessibility.
3. **Usage counting is cheap and valuable** -- add to our schema.
4. **Pre-generation memory update helps** for diverse domains -- our `recall()` already does this.
5. **DC's curator prompt is the best-documented curation specification we've seen** -- use as reference when writing `/sleep` prompts.
6. **Model capability is a prerequisite** -- memory amplifies existing strengths, cannot compensate for weakness. Relevant to our choice of model for sleep processing.
7. **DC confirms that memory curation is the correct direction** but shows the limitations of a flat, single-layer approach. Our multi-tier architecture is the right response.
