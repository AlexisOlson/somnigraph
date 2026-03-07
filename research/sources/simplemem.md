# SimpleMem: Efficient Long-Term Memory via Semantic Lossless Compression -- Analysis

*Phase 14 source analysis, 2026-03-06. Repo: [aiming-lab/SimpleMem](https://github.com/aiming-lab/SimpleMem). Paper: [arXiv 2601.02553v3](https://arxiv.org/abs/2601.02553) (Liu et al., 2025).*

---

## 1. Architecture Overview

**Stack**: Python 3.10, LanceDB (v0.4.5, IVF-PQ indexing) for vector+keyword storage, Tantivy FTS for BM25, Qwen3-embedding-0.6b (1024d) for dense embeddings, any OpenAI-compatible LLM backbone (default GPT-4.1-mini). MCP server (SSE transport) and REST API via FastAPI. Docker deployment available.

**Stars**: ~3.1k (as of March 2026). 304 forks, 57 commits. PyPI package (`pip install simplemem`).

**License**: MIT (AIMING Lab, 2025).

**Authors**: Jiaqi Liu, Yaofeng Su, Peng Xia, Siwei Han, Zeyu Zheng, Cihang Xie, Mingyu Ding, Huaxiu Yao. AIMING Lab (affiliations not specified in paper; GitHub org is `aiming-lab`).

**Core thesis**: Memory for LLM agents should be treated as *active organization* rather than passive storage. SimpleMem implements this through a three-stage pipeline: (1) semantic structured compression at write time, (2) online semantic synthesis (intra-session consolidation during writes, not batch), and (3) intent-aware retrieval planning at read time. The system achieves 43.24% F1 on LoCoMo with ~531 tokens per query -- a 30x reduction versus full-context approaches.

**Cross-session extension**: `SimpleMem-Cross` adds persistent memory across conversations with SQLite for session metadata, automatic context injection at session start, 3-tier secret redaction, observation extraction (decisions/discoveries/learnings), and its own consolidation pipeline (decay, merging, pruning). This is a separate module (`cross/` directory, ~5k lines across 11 modules + tests).

---

## 2. Memory Type Implementation

### Schema

Each memory unit (`MemoryEntry`) is a Pydantic model:

| Field | Type | Description |
|-------|------|-------------|
| `entry_id` | UUID string | Auto-generated |
| `lossless_restatement` | string | Self-contained fact, coreference-resolved, temporally anchored |
| `keywords` | List[str] | Core entities/topic words for BM25 matching |
| `timestamp` | Optional[str] | ISO-8601 absolute time (converted from relative expressions) |
| `location` | Optional[str] | Natural language location |
| `persons` | List[str] | Extracted person names |
| `entities` | List[str] | Companies, products, organizations |
| `topic` | Optional[str] | LLM-summarized topic phrase |
| `vector` | embedding | 1024d Qwen3-embedding-0.6b |

This is a single-layer flat schema -- no memory categories, no priority levels, no decay parameters, no confidence scores. Every memory is an atomic, context-independent factual statement.

### Tools

The system exposes memory operations through MCP (SSE transport at `/mcp/sse`) and REST API:

- **Write**: `add_dialogue()` / `add_dialogues()` -- ingests raw conversation turns, processes through sliding window
- **Read**: `retrieve(query)` -- intent-aware retrieval with optional reflection
- **Cross-session**: Session lifecycle management, context injection, consolidation

No explicit `forget`, `update`, or `feedback` tools in the core system. Cross-session module adds decay/pruning.

### Memory Lifecycle

1. **Ingestion**: Raw dialogue turns enter a buffer
2. **Windowing**: Sliding window (W=40 turns, overlap=2) segments the conversation
3. **Extraction**: LLM processes each window, performing joint coreference resolution, temporal anchoring, and factual extraction. Output is a JSON array of `MemoryEntry` objects
4. **Implicit gating**: Empty extraction results = low-density window (phatic chitchat, greetings). No explicit density threshold -- the LLM's decision to produce zero entries IS the gate
5. **Synthesis**: New entries are checked against previous window's entries (top 3) to avoid duplication. The LLM prompt includes previous entries as context
6. **Storage**: Entries are embedded and stored in LanceDB with FTS index on `lossless_restatement`
7. **Cross-session decay**: SimpleMem-Cross adds automatic decay, merging, and pruning over time

---

## 3. Retrieval Mechanism

### Search Pipeline

SimpleMem implements a three-layer retrieval with LLM-driven planning:

**Stage 1: Intent Analysis** (`_analyze_information_requirements`)
The LLM analyzes the query to produce:
- Question type (factual, temporal, relational, explanatory)
- Key entities
- Required information pieces with priority (high/medium/low)
- Relationships to establish
- Estimated minimal queries needed

**Stage 2: Query Generation** (`_generate_targeted_queries`)
Based on the plan, the LLM generates 1-3 targeted search queries, each focused on a specific information requirement. Guidelines enforce minimality: "fewer, more targeted queries are better."

**Stage 3: Multi-View Execution** (parallel across three layers)
- **Semantic**: Dense vector search via LanceDB (top-k=25 default)
- **Lexical**: BM25 keyword search via Tantivy FTS on `lossless_restatement` (top-k=5)
- **Symbolic**: SQL WHERE clauses on structured metadata -- `array_has_any()` for persons/entities, LIKE for locations, timestamp range comparisons (top-k=5)

**Stage 4: Deduplication**
Set-union by `entry_id` across all three layers. No scoring fusion (RRF, weighted combination, etc.) -- results are simply deduplicated.

**Stage 5: Intelligent Reflection** (optional, up to 2 rounds)
The LLM evaluates result completeness against the original information plan. If gaps are identified, it generates targeted follow-up queries and executes additional searches. This is an iterative self-correction loop.

### Scoring

Notably, there is **no explicit scoring or ranking fusion**. The retrieval pipeline collects candidates from all three layers, deduplicates, and passes the full set to the answer generator. The LLM backbone does the implicit ranking by attending to the most relevant context during answer generation.

This is a significant architectural difference from systems that use RRF, weighted scoring, or learned re-ranking. SimpleMem delegates the "which results matter most" question entirely to the backbone LLM's attention mechanism.

### Retrieval Depth

The planning module infers retrieval depth `d` (range 3-20), proportionally scaling the candidate limit. Table 6 in the paper shows rapid saturation: F1 of 42.85 at k=3, reaching 43.45 at k=10. This validates the high information density of atomic entries -- you don't need many because each one is self-contained and non-redundant.

---

## 4. Standout Feature: Write-Time Online Synthesis

### The Design Choice

SimpleMem's most technically interesting contribution is performing memory consolidation *during the write phase* rather than in a separate batch process. The paper frames this as online semantic synthesis (Section 3.2): "a transformation function F_syn that maps a set of new observations O_session to a consolidated memory entry."

In practice, this means that when a new window of dialogue is processed, the extraction prompt includes the previous window's top-3 memory entries as context. The LLM is instructed to "avoid duplication" against these entries. The result is that related facts from adjacent windows get merged into unified statements at write time rather than stored separately and consolidated later.

### How It Actually Works (Code-Level)

Reading `memory_builder.py`, the mechanism is more modest than the paper's formalism suggests:

```python
context = ""
if self.previous_entries:
    context = "\n[Previous Window Memory Entries (for reference to avoid duplication)]\n"
    for entry in self.previous_entries[:3]:
        context += f"- {entry.lossless_restatement}\n"
```

The previous entries are injected into the extraction prompt as a deduplication signal. The LLM is expected to:
1. Not re-extract facts already captured in previous windows
2. Merge related facts when the current window adds detail to a previously captured fact

This is **implicit synthesis via prompt context**, not an explicit merge algorithm. There is no embedding similarity check, no threshold-based dedup, no graph-based clustering. The LLM does all the work.

### The Cognitive Science Lineage

The paper cites Complementary Learning Systems (CLS) theory, but the actual mapping is loose. In CLS, the hippocampus does fast episodic encoding and the neocortex does slow semantic consolidation during sleep (offline). SimpleMem inverts this: consolidation happens at encoding time (online), not during a separate offline phase. This is closer to the **levels-of-processing** framework (Craik & Lockhart, 1972) -- deeper processing at encoding produces more durable, more organized memory traces.

The biological analogy that actually fits: **schema-consistent encoding**. When new information is consistent with existing schemas (here, previous window entries), the brain integrates it immediately rather than storing it for later consolidation. Van Kesteren et al. (2012) showed that schema-consistent information bypasses hippocampal replay and goes directly to neocortical integration. SimpleMem does exactly this -- new facts that relate to existing entries get integrated at write time.

### Does This Challenge Batch Consolidation?

This is the central question for our system. Our sleep pipeline (NREM + REM) runs offline: NREM clusters similar memories, merges them, and creates edges; REM performs gap analysis and question generation. SimpleMem's online synthesis eliminates the need for the merge step of NREM entirely.

**Arguments that online synthesis is superior:**

1. **Information freshness**: Synthesis happens while the full conversational context is available. Our NREM pipeline operates on stored memory text alone, which has already lost the conversational context that motivated the memory. SimpleMem's write-time synthesis has access to the raw dialogue, the previous window's entries, and the LLM's in-context understanding of the conversation flow.

2. **Reduced redundancy at source**: Our system stores memories individually and relies on sleep to find and merge duplicates. SimpleMem never creates the duplicates in the first place. At ~300 memories, this doesn't matter much for us. At SimpleMem's benchmark scale (LoCoMo: hundreds of sessions, thousands of dialogue turns), it prevents combinatorial growth of near-duplicate entries.

3. **Computational efficiency**: SimpleMem's construction phase is 92.6s per sample vs Mem0's 1,350.9s -- 14x faster. No separate consolidation pass needed.

4. **Ablation evidence**: Removing online synthesis drops multi-hop F1 by 31.3% and average F1 by 11.6%. This is a real, measured contribution.

**Arguments that batch consolidation still has advantages:**

1. **Cross-session patterns**: Online synthesis only merges within adjacent windows of a single session. Our NREM pipeline can discover that a memory from Session 47 and a memory from Session 112 are about the same topic and should be merged. SimpleMem's cross-session module adds its own consolidation for this, but it's separate from the core online synthesis.

2. **Structural discovery**: Our NREM creates *edges* between related memories, building a graph structure that supports multi-hop traversal. SimpleMem's online synthesis produces flat, independent entries with no relational structure. The graph is what enables our novelty-scored adjacency expansion.

3. **REM-style gap analysis**: Our REM pipeline identifies *what's missing* -- questions that should be answerable but aren't, gaps in coverage, contradictions. Online synthesis can only merge what's present; it cannot identify what's absent.

4. **Evolving understanding**: When new information changes the interpretation of old information, batch consolidation can retroactively update. Online synthesis is forward-only -- it can avoid duplicating past entries, but it cannot revise them in light of new context.

**Synthesis**: The two approaches are complementary, not competing. Online synthesis handles the *redundancy reduction* aspect of consolidation (the merge step). Batch consolidation handles *structural enrichment* (edges, contradictions, gap analysis) and *cross-temporal integration* (connecting memories across sessions). A hybrid approach -- online dedup at write time, offline graph construction and gap analysis during sleep -- would capture both benefits. This is essentially what SimpleMem-Cross does by adding its own consolidation pipeline on top of the online synthesis.

The deeper insight: **the unit of consolidation matters**. SimpleMem consolidates *facts* (merging "likes coffee" + "prefers oat milk" into one statement). Our system consolidates *relationships* (creating edges between memories, detecting contradictions, tracking confidence). These are different levels of organization. SimpleMem's approach is optimal for the LoCoMo benchmark because LoCoMo tests factual recall. Our approach is better suited for long-running personal memory where the *structure* of knowledge (what contradicts what, what evolved from what, what's uncertain) matters more than raw factual coverage.

---

## 5. Other Notable Features

### 5.1 Implicit Semantic Density Gating

The LLM itself serves as the density gate. No explicit classifier, no entropy threshold, no heuristic. If a window of dialogue contains only phatic content ("Hi!" "How are you?" "Good, thanks!"), the extraction prompt returns an empty JSON array. The empty set IS the gate signal.

This is elegant in its simplicity. The paper formally describes it as "Phi_gate(W) -> {m_k}" where empty output = low density, but the implementation is just: the LLM decides what's worth extracting. No separate classification step, no wasted API calls on a density classifier. The cost is that gating quality is entirely dependent on the backbone LLM's judgment.

### 5.2 De-Linearization via Joint Coreference + Temporal Resolution

The extraction prompt performs three transformations simultaneously:
- **g_coref**: All pronouns resolved to explicit names ("he" -> "Bob")
- **g_time**: All relative time expressions converted to ISO-8601 ("tomorrow" -> "2025-11-16T14:00:00")
- **g_ext**: Complex dialogue atomized into independent factual statements

This produces memories that are genuinely context-independent -- each entry can be understood without any surrounding context. This is critical for retrieval: a query about "Bob's meeting" will match the memory because Bob's name is explicit, not hidden behind a pronoun.

The temporal anchoring is particularly well-designed. The extraction prompt includes the dialogue timestamps, and the LLM is instructed to compute absolute times from relative expressions. The paper shows this contributes the largest single ablation effect: removing semantic compression drops temporal F1 by 56.7%.

### 5.3 Reflection-Based Retrieval Self-Correction

The retrieval pipeline includes an optional reflection loop (up to 2 rounds by default) where the LLM:
1. Evaluates whether the retrieved results cover all information requirements from the planning phase
2. Identifies specific gaps ("we found Bob's role but not the meeting time")
3. Generates targeted follow-up queries for the gaps
4. Executes additional searches and merges results

This is a form of **retrieval-augmented self-correction** that's distinct from re-ranking. Rather than re-scoring existing results, it identifies what's *missing* and actively searches for it. The ablation shows removing intent-aware retrieval drops open-domain F1 by 26.6%.

### 5.4 Parallel Window Processing

The `MemoryBuilder` supports parallel processing of dialogue windows via `ThreadPoolExecutor` (configurable up to 16 workers). This enables processing large dialogue histories significantly faster. The implementation includes proper fallback to sequential processing on error, and the previous-window context is shared across workers (though this creates a race condition where parallel workers all see the same "previous entries" rather than a cascading chain -- acceptable because the dedup is approximate anyway).

### 5.5 Cross-Session Module (SimpleMem-Cross)

A substantial extension (~5k lines) adding:
- Session lifecycle management with SQLite metadata
- Token-budgeted context injection at session start
- 3-tier automatic secret redaction
- Heuristic extraction of decisions, discoveries, and learnings
- Consolidation pipeline with decay, merging, and pruning
- 127 passing tests

The cross-session module reports 64% improvement over "Claude-Mem" on LoCoMo (48 vs 29.3), though "Claude-Mem" is not well-defined in the paper (likely refers to Anthropic's built-in memory feature, not our claude-memory MCP stack).

---

## 6. Gap Ratings

Rating SimpleMem against the gap dimensions:

| Dimension | Rating | Evidence |
|-----------|--------|----------|
| **Layered Memory** | 15% | Single flat layer. No episodic/semantic/procedural distinction. No priority, no decay parameters on individual memories. Cross-session module adds decay but still no category differentiation. |
| **Multi-Angle Retrieval** | 75% | Three-layer retrieval (semantic + lexical + symbolic) with LLM-driven planning and reflection. No scoring fusion, but the multi-view approach is well-implemented. Missing: feedback-weighted scoring, co-retrieval boost. |
| **Contradiction Detection** | 5% | No contradiction detection mechanism at all. New memories that contradict old ones are simply stored alongside them. The online synthesis might incidentally avoid storing contradictory facts within the same session, but there's no explicit detection or resolution. |
| **Relationship Edges** | 5% | No graph structure. Memories are independent atoms with no edges, links, or relational metadata. The retrieval planner infers relationships at query time, but nothing is stored structurally. |
| **Sleep Process** | 30% | No offline consolidation in core system. Online synthesis handles intra-session dedup. Cross-session module adds decay/merging/pruning, but no gap analysis, no edge creation, no structural enrichment. |
| **Reference Index** | 60% | Strong metadata indexing (persons, entities, locations, timestamps, topics, keywords). The symbolic search layer enables deterministic filtering. Missing: source provenance, session linking, dialogue-to-memory traceability. |
| **Temporal Trajectories** | 55% | ISO-8601 temporal anchoring is well-implemented. Timestamp range queries in structured search. Temporal F1 is strong (58.62). But no temporal trajectory modeling -- no tracking of how facts evolve over time, no version history. |
| **Confidence/UQ** | 5% | No confidence scores, no uncertainty quantification, no trust levels. Every memory is treated as equally reliable. The `salience` field mentioned in the paper's formalism doesn't appear in the actual MemoryEntry schema. |

---

## 7. Comparison with claude-memory

### 7.1 Stronger (SimpleMem vs us)

**Write-time deduplication**: SimpleMem's online synthesis prevents redundant memories at creation time. Our system stores memories individually and relies on sleep NREM to find and merge duplicates. At our current scale (~300 memories), this isn't a problem. But it's architecturally cleaner -- prevention over remediation.

**Multi-view indexing with symbolic layer**: SimpleMem's structured metadata search (persons, entities, locations, timestamp ranges) provides deterministic filtering that our system lacks. Our retrieval is purely vector + FTS. A query like "what happened with Bob in January" can be answered by SimpleMem with a symbolic filter (`persons CONTAINS 'Bob' AND timestamp BETWEEN '2025-01-01' AND '2025-01-31'`), whereas we rely on vector/FTS similarity to surface the right memories.

**Intent-aware retrieval planning**: The LLM analyzing the query before executing searches is a form of query decomposition that our system doesn't do. Our recall takes a query string and runs it through vector + FTS directly. SimpleMem's planning step can break "what was the outcome of Bob and Alice's discussion about the product launch timeline?" into targeted sub-queries for each information need.

**Temporal anchoring at write time**: Converting relative time expressions to absolute ISO-8601 at extraction time is simple but high-impact (56.7% temporal F1 drop without it). Our system stores memories with creation timestamps but doesn't normalize temporal references within memory content.

**Benchmark validation**: SimpleMem has rigorous benchmark results on LoCoMo (43.24% F1) and LongMemEval-S (76.87%) with ablation studies. Our system has retrospective experiments and MRR measurements but no standardized benchmark evaluation.

### 7.2 Weaker (SimpleMem vs us)

**No memory categories or layered organization**: Our system distinguishes episodic, procedural, semantic, reflection, and meta memories with per-category decay rates. SimpleMem treats all memories identically. A correction pattern and a one-off fact have the same persistence and retrieval weight.

**No feedback loop**: Our recall_feedback tool (continuous 0-1 utility + durability scoring) and Hebbian PMI co-retrieval boost create a learning retrieval system that gets better with use. SimpleMem has no feedback mechanism -- retrieval quality is static.

**No graph structure**: Our ~600 edges with contradiction/revision/derivation flags, novelty-scored adjacency expansion, and linking_context enable multi-hop reasoning and structural understanding. SimpleMem's memories are isolated atoms.

**No scoring fusion**: Our Bayesian-optimized 9-coefficient post-RRF scoring (k=14) produces ranked results. SimpleMem does set-union deduplication with no ranking -- the backbone LLM implicitly ranks via attention, but this is uncontrolled and not optimizable.

**No confidence or trust modeling**: Our confidence gradient (0.1-0.95 trust level compounding through feedback and edges) provides uncertainty quantification. SimpleMem has no equivalent.

**No contradiction detection or resolution**: Our edges can flag contradictions; NREM sleep processes detect conflicting memories. SimpleMem can store contradictory facts without any awareness of the conflict.

**No gap analysis**: Our REM sleep identifies missing knowledge and generates questions. SimpleMem's retrieval reflection identifies retrieval gaps at query time, but doesn't proactively identify knowledge gaps.

**No shadow load or retention control**: Our shadow penalty (quadratic cost for high-load memories) and pinned flag (retention immunity) provide fine-grained lifecycle control. SimpleMem's cross-session module has basic decay/pruning but nothing this nuanced.

### 7.3 Shared Strengths

**Hybrid retrieval**: Both systems use vector + BM25/FTS dual-channel retrieval. The insight that neither channel alone is sufficient is well-validated in both systems (our Experiment 6 showed vector rescues 58 vocabulary-gap cases; SimpleMem's ablation shows each layer contributes independently).

**LLM-driven extraction**: Both systems use the backbone LLM to transform raw content into structured memory entries rather than storing raw text. Our `remember()` tool uses LLM-generated summaries; SimpleMem's extraction prompt produces structured JSON with resolved references.

**Atomic memory units**: Both systems favor self-contained memory entries that can be understood independently. Our memories have summaries; SimpleMem's have lossless restatements with resolved coreferences.

**CLS theory influence**: Both systems cite CLS theory as architectural motivation. Our NREM/REM sleep pipeline maps more directly to the hippocampal replay model; SimpleMem's online synthesis maps to schema-consistent encoding.

---

## 8. Insights Worth Stealing

### 8.1 Temporal Anchoring at Write Time
**Effort**: Low | **Impact**: High

Add a normalization step in `remember()` that converts relative time expressions ("yesterday", "last week", "this morning") to absolute ISO-8601 timestamps using the current date. This is a simple prompt modification or regex-based transformation that would significantly improve temporal retrieval.

### 8.2 Write-Time Dedup Check Against Recent Memories
**Effort**: Low | **Impact**: Medium

Before storing a new memory via `remember()`, retrieve the top-3 most similar existing memories and include them in a prompt asking: "Is this new information, or a refinement of existing memory? If refinement, produce a merged version." This captures SimpleMem's online synthesis benefit without changing our architecture. We already have the vector search infrastructure; this adds one LLM call per write.

### 8.3 Symbolic Metadata Extraction and Filtering
**Effort**: Medium | **Impact**: Medium

Extract structured metadata (persons, entities, locations) at write time and store as indexed columns. Add a pre-filter step to `recall()` that uses deterministic metadata matching before vector/FTS scoring. This would improve precision for entity-specific queries without replacing our existing retrieval pipeline.

### 8.4 Query Decomposition Before Retrieval
**Effort**: Medium | **Impact**: Medium-High

Add a lightweight planning step before `recall()` that analyzes complex queries and decomposes them into targeted sub-queries. Not the full SimpleMem treatment (which requires 2 LLM calls before any search), but a single-call analysis that identifies whether the query needs multiple search angles. Our staged curated recall already does something like this via the Sonnet subagent, but the planning could happen in the main recall path for complex queries.

### 8.5 Implicit Density Gating for Auto-Capture
**Effort**: Low | **Impact**: Low-Medium

Apply the "empty set = low density" principle to our auto-capture patterns. When detecting potential memories to auto-capture, the LLM's decision not to extract anything IS the quality gate. This validates our current approach of selective auto-capture rather than capturing everything.

---

## 9. What's Not Worth It

### No Scoring Fusion
SimpleMem's decision to skip RRF or any scoring fusion and instead rely on the backbone LLM's attention is a reasonable choice for benchmarks (where the LLM sees all retrieved context at once) but would be a regression for our system. Our Bayesian-optimized scoring directly controls what surfaces in the recall response; removing it would lose the feedback loop, the Hebbian boost, the confidence weighting, and the shadow penalty. The LLM's attention is not a substitute for learned retrieval preferences.

### LanceDB Migration
SimpleMem uses LanceDB for its combined vector + metadata storage. Our SQLite + sqlite-vec + FTS5 stack is well-integrated, supports our custom scoring pipeline, and has the edge/graph schema that LanceDB doesn't natively support. Migrating would be a high-effort lateral move with no clear benefit for our use case.

### Sliding Window Ingestion
SimpleMem's 40-turn sliding window approach is designed for processing long dialogue transcripts after the fact. Our system captures memories in real-time during conversation via the `remember()` tool. The window-based approach makes sense for bulk ingestion of historical data but doesn't match our interactive, selective capture pattern.

### Reflection-Based Self-Correction in Retrieval
SimpleMem's retrieval reflection loop (evaluate completeness, generate follow-up queries, re-search) adds 1-2 extra LLM calls per retrieval. Our staged curated recall already provides a similar capability via background Sonnet subagent. Adding another reflection loop in the main recall path would increase latency without clear benefit beyond what we already have.

---

## 10. Key Takeaway

SimpleMem's core contribution is showing that **write-time organization can substitute for post-hoc consolidation** when the goal is factual recall from conversation histories. The three-stage pipeline -- implicit density gating, online synthesis with coreference/temporal resolution, and intent-aware retrieval planning -- achieves strong benchmark results (43.24% F1 on LoCoMo, 76.87% accuracy on LongMemEval-S) with high token efficiency (~531 tokens/query, 30x reduction vs full-context). The most transferable insight for our system is not the online synthesis itself (our sleep pipeline serves different purposes -- structural enrichment, not just dedup) but the *temporal anchoring* and *write-time dedup check*, both of which are low-effort additions that would improve our retrieval quality. SimpleMem's fundamental limitation is its flat, unstructured memory model: no categories, no edges, no confidence, no feedback loop. It excels at the benchmark task (answering questions about past conversations) but lacks the machinery for the kind of evolving, self-correcting, structurally-rich memory that a long-running personal system requires. The two systems optimize for different things: SimpleMem for information retrieval accuracy on standardized benchmarks; claude-memory for cumulative knowledge organization with learning feedback loops.
