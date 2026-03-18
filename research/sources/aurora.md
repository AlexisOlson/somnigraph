# Aurora Analysis (Agent Output)

*Generated 2026-03-17 by Opus agent reading local clone*

---

## Aurora Analysis: Architecture, Comparison, and Insights for Somnigraph

### Repository Overview

**Repo**: https://github.com/hamr0/aurora
**License**: MIT
**Package**: `aurora-actr` on PyPI (v0.17.6)
**Language**: Python 3.12+
**Commits**: 743 (active development)
**Source files**: ~493 Python files, ~150 test files (2,608+ tests per CLAUDE.md)

Aurora ("Adaptive Unified Reasoning and Orchestration Architecture") is a code intelligence and multi-agent orchestration tool for AI coding assistants. It indexes codebases using tree-sitter AST parsing and provides hybrid retrieval (BM25 + semantic embeddings + ACT-R activation decay) via CLI and MCP tools. Unlike Somnigraph's focus on personal episodic/procedural memory, Aurora's memory is primarily a *codebase index* — it stores parsed code chunks, markdown knowledge, and reasoning traces from its SOAR pipeline.

The system is designed to work with 20+ AI coding tools (Claude Code, Cursor, Aider, etc.) and provides slash commands for planning workflows (`/aur:plan`, `/aur:implement`). It includes a multi-agent spawner with circuit breakers, a friction analysis pipeline that extracts "antigens" (learned rules) from stuck coding sessions, and a SOAR orchestrator for decomposing complex queries into agent-routed subgoals.

### File Structure (Key Paths)

```
packages/
  core/           # Storage, activation engine, chunk types, budget tracking
    store/sqlite.py          # SQLiteStore with connection pooling, FTS5
    activation/engine.py     # ACT-R activation: BLA + spreading + context + decay
    activation/decay.py      # Type-specific decay rates, churn factor
    activation/spreading.py  # BFS spreading activation over relationship graph
    activation/retrieval.py  # Activation-based retrieval with threshold filtering
    chunks/                  # Chunk types: code, kb, doc, reasoning
  context-code/   # Tree-sitter parsing, BM25, embeddings, hybrid retriever
    semantic/hybrid_retriever.py  # Tri-hybrid: BM25 + ACT-R + semantic (1253 lines)
    semantic/bm25_scorer.py       # Code-aware BM25 tokenizer
    semantic/embedding_provider.py # all-MiniLM-L6-v2 embeddings
    languages/               # Python, JS, TS, Go, Java parsers
  context-doc/    # PDF/DOCX parsing
  reasoning/      # LLM clients, decomposition, verification prompts
  soar/           # 7-phase orchestrator (assess→retrieve→decompose→verify→collect→synthesize→record)
  spawner/        # Subprocess agent spawner with circuit breaker, heartbeat
  planning/       # Plan generation, PRD creation
  lsp/            # LSP code intelligence (usage, callers, deadcode)
  cli/            # Click CLI, memory manager, agent discovery
  implement/      # Sequential task execution
src/aurora_mcp/   # MCP server (lsp tool, mem_search tool)
scripts/          # friction.py, antigen_extract.py
```

---

## 1. Architecture Overview

### Core Components

**Storage**: SQLite per-project (`.aurora/memory.db`), schema v6. Tables: `chunks` (id, type, content JSON, metadata JSON, embeddings BLOB, timestamps), `activations` (base_level, access_count, access_history JSON), `relationships` (from/to/type/weight), `file_index` (incremental indexing with content hash), `doc_hierarchy` (PDF/DOCX section tree), `chunks_fts` (FTS5 with Porter stemmer). No vector index extension — embeddings stored as raw BLOBs and compared via numpy cosine similarity.

**Write Path**: `aur mem index .` triggers file discovery (respecting `.auroraignore`), incremental change detection (git status → mtime → content hash), tree-sitter AST parsing per language, embedding generation via `all-MiniLM-L6-v2` (384d, local), and chunk storage. Each code element (function, class, method) becomes a separate chunk. The SOAR pipeline also writes "reasoning chunks" when caching successful query patterns (`record_pattern_lightweight` in `phases/record.py`).

**Retrieval**: Two-stage hybrid retrieval (`hybrid_retriever.py`):
1. **Stage 1 (Candidate Selection)**: FTS5 keyword search (primary) or activation-based retrieval (fallback for old DBs), top 100 candidates.
2. **Stage 2 (Tri-Hybrid Re-ranking)**: Each candidate scored on three axes with chunk-type-aware weights:
   - Code chunks: BM25 50% / ACT-R 30% / Semantic 20%
   - KB chunks: BM25 30% / ACT-R 30% / Semantic 40%
   Scores are min-max normalized independently then combined as a weighted sum.

**Activation Engine**: Full ACT-R implementation with four components:
- **Base-Level Activation (BLA)**: `ln(sum(t_j^(-d)))` over access history, d=0.5
- **Spreading Activation**: BFS over relationship graph, `weight * 0.7^hop`, max 3 hops
- **Context Boost**: Keyword overlap fraction * boost_factor (0.5)
- **Decay Penalty**: `-decay_factor * log10(days_since_access)`, type-specific rates (kb=0.05, function=0.40, doc=0.02), churn factor adds `0.1 * log10(commit_count + 1)`

**No consolidation or sleep process**. Memory is a codebase index that refreshes on re-index, not an accumulating personal memory.

### Scoring Details

The scoring formula is a hand-tuned weighted sum (no learned reranker). Weights are fixed per chunk type. There is no RRF fusion, no feedback loop, no ground-truth evaluation infrastructure. The system does support MMR (Maximal Marginal Relevance) reranking for diversity when `diverse=True`.

---

## 2. Unique Concepts

**a. ACT-R Cognitive Architecture as Retrieval Framework**
Aurora explicitly implements the ACT-R cognitive architecture (Anderson, 2007) as its retrieval model, including BLA, spreading activation, context boost, and decay — all as separate composable modules with Pydantic configs. This is more theoretically grounded than most memory systems that use ad-hoc decay functions.

**b. Type-Specific Decay Rates with Churn Factor**
Different chunk types decay at different rates (kb=0.05, function=0.40, class=0.20, doc=0.02, toc_entry=0.01). High-churn code (many git commits) decays faster via a logarithmic penalty. This is a principled approach to the observation that volatile code is less reliable to remember.

**c. Chunk-Type-Aware Retrieval Weights**
Code chunks use BM25-heavy scoring (50%) because identifiers are exact tokens; KB chunks use semantic-heavy scoring (40%) because prose benefits from embeddings. This is a clean insight that most hybrid systems miss by using uniform weights.

**d. Friction Analysis → Antigen Extraction**
The `aur friction` command analyzes past coding sessions for "stuck" patterns (58% BAD rate, 14 signals weighted), then an LLM extracts "antigens" — learned rules to add to CLAUDE.md to prevent the same mistakes. This is a novel approach to closing the loop between failure patterns and behavioral guidance.

**e. SOAR Orchestrator with Agent Spawning**
A 7-phase pipeline (assess → retrieve → decompose → verify → collect → synthesize → record) that decomposes complex queries, routes subgoals to appropriate agents, spawns subprocess agents with circuit breakers and heartbeat monitoring, and caches successful reasoning patterns.

**f. Code Intelligence via LSP + Ripgrep Hybrid**
`mem_search` enriches results with LSP-derived metadata (usage count, callers, callees, complexity, risk level), falling back to ripgrep text search when LSP returns zero refs. This provides contextual code understanding beyond what pure search offers.

**g. Document Hierarchy with Section Paths**
The `doc_hierarchy` table stores parent-child relationships and pre-computed breadcrumb paths (`["Ch 2", "2.1 Install"]`) for PDF/DOCX chunks, enabling section-aware document navigation.

**h. Reasoning Pattern Caching**
The SOAR Record phase stores successful reasoning traces as `ReasoningChunk` objects with confidence-based caching policy (>=0.8: pattern +0.2 activation, >=0.5: learning +0.05, <0.5: skip). This feeds future retrieval with past solutions.

---

## 3. How Aurora Addresses Our 7 Known Gaps

| Gap | Assessment |
|-----|-----------|
| 1. Layered Memory | **Partially addressed.** Four chunk types (code, kb, doc, reas) with type-specific decay rates create implicit memory layers. However, this is content-type layering, not the kind of episodic→semantic consolidation that Somnigraph's gap refers to. There's no promotion between layers or abstraction over time. |
| 2. Multi-Angle Retrieval | **Addressed differently.** The chunk-type-aware weight system (code: BM25-heavy, KB: semantic-heavy) is a form of multi-angle retrieval. MMR reranking provides diversity. But there's no query decomposition into multiple retrieval angles or multi-probe approach — it's a single query dispatched once. |
| 3. Contradiction Detection | **Not addressed.** No mechanism for detecting contradictory information. Relationships are structural (calls, imports, depends_on) not semantic (contradicts, supersedes). The system indexes a codebase snapshot, which by nature has fewer contradictions than personal episodic memory. |
| 4. Relationship Edges | **Addressed for code, not for knowledge.** The `relationships` table stores `depends_on`, `calls`, and `imports` edges with weights. Spreading activation traverses these via BFS. However, these are code-structural relationships extracted from AST/LSP — not semantic relationships between ideas or memories. No link/unlink user operations. |
| 5. Sleep Process | **Not addressed.** No offline consolidation, no background maintenance beyond re-indexing. Memory is a codebase index, not a personal accumulating memory, so consolidation is less relevant. The SOAR Record phase is the closest analog — caching successful patterns — but it's online and per-query. |
| 6. Reference Index | **Not addressed in the Somnigraph sense.** The `file_index` table tracks indexed files for incremental updates, and doc_hierarchy provides document structure, but there's no reference index of external sources cited within memories. |
| 7. Temporal Trajectories | **Not addressed.** Access history is tracked (access_count, access_history JSON, last_access) and BLA uses recency weighting, but there's no trajectory analysis — no tracking of how a concept evolves over time, no temporal clustering, no narrative arc detection. |

---

## 4. Comparison

### Where Aurora Contributes

1. **Chunk-type-aware scoring weights** — Somnigraph uses uniform RRF fusion across all memory categories. Aurora's insight that code benefits from BM25 while prose benefits from semantics is applicable if Somnigraph ever indexes heterogeneous content types.

2. **Type-specific decay rates** — Somnigraph has per-category decay rate overrides but the taxonomy is less granular. Aurora's empirical rates (kb=0.05, function=0.40, class=0.20) with churn factor are well-calibrated for code but suggest a similar approach for memory categories (procedural vs. episodic vs. reflection).

3. **Friction analysis as a feedback mechanism** — While Somnigraph has `recall_feedback()` for per-query relevance grading, Aurora's friction analysis operates at a higher level: analyzing entire sessions for behavioral patterns and extracting preventive rules. This is a different and complementary kind of feedback loop.

4. **ACT-R spreading activation** — Somnigraph has PPR (Personalized PageRank) over its relationship graph. Aurora's BFS spreading activation with hop decay is simpler and could be compared.

5. **FTS5 as primary candidate gate** — Aurora moved from activation-based candidate selection to FTS5 keyword search as the primary gate, with activation used only in the re-ranking stage. Somnigraph uses FTS5 as one of several parallel retrieval channels fused via RRF.

### Where Somnigraph Is Stronger

1. **Feedback loop** — Somnigraph's `recall_feedback()` grades retrieval results, shaping future scoring via EWMA utility and UCB exploration bonus. Aurora has no equivalent — its scoring weights are static.

2. **Learned reranker** — Somnigraph's LightGBM reranker trained on ground-truth judgments (+5.7% NDCG@5k) vs. Aurora's hand-tuned weighted sum with no evaluation infrastructure.

3. **Sleep/consolidation** — Somnigraph's three-phase offline pipeline (NREM classification → REM summarization → maintenance) has no counterpart in Aurora.

4. **Enriched embeddings** — Somnigraph embeds `content + category + themes + summary` as a single enriched string. Aurora embeds raw code/content without enrichment.

5. **Evaluation methodology** — Somnigraph has ground-truth judging, NDCG metrics, utility calibration studies, counterfactual experiments. Aurora has no retrieval quality measurement.

6. **Semantic relationships** — Somnigraph's edges include `related_to`, `contradicts`, `derived_from` with decay and confidence. Aurora's relationships are purely structural (calls, imports, depends_on).

7. **Personal memory** — Somnigraph stores episodic, procedural, and reflective memories with nuanced categorization. Aurora stores code chunks and reasoning traces — it's an index, not a memory.

### Fundamental Differences

Aurora and Somnigraph solve different problems despite sharing technical components (SQLite, FTS5, hybrid retrieval, decay). Aurora is a **codebase index with cognitive-inspired retrieval** — it indexes external artifacts (source code, documents) and provides search/planning tools. Somnigraph is a **personal memory system with biological-inspired consolidation** — it stores agent-generated memories and improves retrieval quality over time through feedback and offline processing.

The key architectural divergence: Aurora's memory is *refreshable* (re-index overwrites stale chunks), while Somnigraph's memory is *accumulating* (memories decay but are never re-indexed from source). This makes consolidation, contradiction detection, and feedback loops essential for Somnigraph but irrelevant for Aurora.

---

## 5. Insights Worth Stealing

1. **Chunk-type-aware retrieval weights** — When Somnigraph's reranker already learns per-feature weights, this is partially captured. But the *principle* — that different content types benefit from different retrieval signal ratios — could inform feature engineering for the reranker (e.g., category-specific BM25 weight as a feature). **Effort: LOW. Impact: MEDIUM.**

2. **Type-specific decay rates with empirical calibration** — Aurora's decay taxonomy (0.01 for structural anchors → 0.40 for volatile code) suggests that Somnigraph's per-category decay_rate defaults could be more intentionally calibrated. Currently Somnigraph defaults are set per-memory; Aurora shows value in per-type defaults. **Effort: LOW. Impact: LOW** (Somnigraph already supports this, just hasn't tuned the defaults systematically).

3. **Churn factor for decay** — The idea that high-churn content (frequently updated) should decay faster is novel and applicable. For Somnigraph, memories that get frequently corrected or superseded might benefit from accelerated decay. This could be implemented as: if a memory has been `forget()`-and-re-`remember()`-ed multiple times, increase its decay rate. **Effort: MEDIUM. Impact: LOW** (edge case for Somnigraph's scale).

4. **Friction analysis concept** — Analyzing sessions for failure patterns and extracting preventive rules is a higher-level feedback mechanism than per-query grading. Somnigraph could potentially analyze recall patterns across sessions to identify systematic retrieval failures (e.g., "queries about X consistently return irrelevant results about Y"). **Effort: HIGH. Impact: MEDIUM.**

5. **MMR diversity reranking** — Aurora's MMR implementation (`lambda * relevance - (1-lambda) * max_similarity_to_selected`) prevents echo-chamber results. Somnigraph's roadmap includes diversity as an open question. Aurora's implementation is clean and could be adapted. **Effort: LOW. Impact: MEDIUM** (depends on whether diversity is a real problem at Somnigraph's scale).

---

## 6. What's Not Worth It

1. **ACT-R engine architecture** — Aurora's modular ACT-R engine (BLA, spreading, context boost, decay as separate composable components) is well-engineered but solves a problem Somnigraph has already moved past. Somnigraph's learned reranker subsumes what these hand-tuned components do. Adopting Aurora's architecture would be a regression.

2. **FTS5 as primary gate** — Aurora uses FTS5 as the sole candidate selection mechanism (top 100 → re-rank). Somnigraph already uses FTS5 as one of several parallel channels (vector, FTS, graph, theme) fused via RRF. The parallel approach is more robust; switching to sequential gating would lose coverage.

3. **SOAR orchestrator** — The 7-phase reasoning pipeline is application-level orchestration, not memory infrastructure. It's well-designed for Aurora's use case (multi-agent task decomposition) but irrelevant to Somnigraph's scope.

4. **Code indexing / tree-sitter parsing** — Somnigraph stores agent-generated memories, not parsed source code. The entire `context-code` package is out of scope.

5. **Agent spawner / circuit breaker** — Infrastructure for multi-agent orchestration, not memory retrieval. Out of scope.

---

## 7. Critical Assessment

**Strengths:**
- Principled ACT-R foundation with proper citations and faithful implementation
- Clean separation between storage, activation, and retrieval layers
- Practical code intelligence features (LSP enrichment, risk levels, complexity metrics)
- Type-specific decay rates are empirically sensible and well-documented
- Chunk-type-aware retrieval weights are a genuine insight
- Broad tool support (20+ AI coding assistants)
- Active development (743 commits, 2600+ tests)

**Limitations:**
- No retrieval quality evaluation — no ground truth, no NDCG, no feedback loop. The hand-tuned weights (50/30/20 for code, 30/30/40 for KB) are plausible but unvalidated
- No vector index — embeddings stored as BLOBs, compared via brute-force numpy cosine similarity. This works at codebase scale (thousands of chunks) but won't scale to tens of thousands
- Heavy dependency footprint — sentence-transformers + torch for local embeddings adds significant installation overhead (~2GB+)
- The "memory" framing is somewhat misleading — it's primarily a codebase search index with activation-based ranking, not persistent memory in the agent cognition sense
- No feedback mechanism — retrieval quality can't improve over time; weights are static
- Reasoning pattern caching (SOAR Record phase) is rudimentary — simple keyword extraction, truncated summaries, confidence-gated storage. No deduplication, no contradiction handling with existing patterns
- The `Long-Term Memory` design doc (`LONG_TERM_MEMORY.md`) is status "DESIGN" — describing a planned system of stash/friction/remember that isn't yet integrated with the existing indexing memory. This suggests awareness that the current system isn't actually long-term memory
- Documentation is scattered across many small files in a deep directory hierarchy, making it harder to form a coherent picture than Somnigraph's narrative docs

**Honest assessment for Somnigraph research corpus**: Aurora is a well-built codebase indexing tool with cognitive-science-inspired retrieval scoring. Its strongest contributions to the Somnigraph corpus are the chunk-type-aware weight insight and the type-specific decay rate taxonomy. It does not advance the state of the art on any of Somnigraph's 7 tracked gaps. The systems operate in adjacent but distinct problem spaces — code search vs. personal memory — and the architectural decisions that are right for one are often wrong for the other.
