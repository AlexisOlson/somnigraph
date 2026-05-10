# context-mem v3.2 — Persistent Memory for AI Coding Assistants

*Phase 29, 2026-04-15. Analysis of https://github.com/JubaKitiashvili/context-mem (Juba Kitiashvili).*

---

## 1. Overview

**What it is**: An MCP memory server for AI coding assistants (Claude Code, Cursor, Windsurf, VS Code, Cline, Roo Code) that automatically captures, compresses, and retrieves session context. TypeScript, MIT license.

**Repo**: https://github.com/JubaKitiashvili/context-mem. 285 commits over ~1 month (2026-03-16 to 2026-04-15). v3.2.0. npm package.

**The core claim**: "98%+ retrieval accuracy on 4 academic benchmarks" (LongMemEval, LoCoMo, MemBench, ConvoMem), "100% with optional LLM reranking", "99% token compression", "fully local and free."

**What it actually does**: Hooks into AI coding assistants to observe tool outputs, extracts entities and importance scores deterministically (no LLM), stores in SQLite with FTS5, and retrieves via multi-strategy BM25 + optional vector search. Background "Dreamer" agent handles progressive compression and staleness management.

**Dependencies**: `better-sqlite3`, `@modelcontextprotocol/sdk`, `ws`. Optional: `@huggingface/transformers` for vector embeddings.

---

## 2. Architecture

### Storage

SQLite with WAL mode, FTS5 full-text index, trigram index for substring matching. Core tables:

- **observations** — content, summary, metadata, importance_score, compression_tier, pinned, embeddings, content_hash
- **knowledge** — structured facts with temporal validity (valid_from/to), contradiction_count, relevance_score, authority scoring
- **entities** / **relationships** — knowledge graph (10 entity types, 11 relationship types)
- **session_chains** — cross-session continuity with handoff summaries
- **token_stats** — per-session token economics tracking

17 migrations. 64MB mmap, 8MB cache.

### Write Path

Every tool output → privacy screening (9 secret detectors) → parallel extraction (entities via regex + 100+ tech aliases, importance scoring, topic detection) → 14 content-type summarizers (code, shell, JSON, error, log, TS-error, test output, build output, git log, CSV, HTML, markdown, network, binary) → triple storage (verbatim content, compressed summary, knowledge graph entries).

Importance scoring is fully deterministic: base score by observation type (decision=0.9, error=0.8, commit=0.7, test=0.6, code=0.5, log=0.3), +0.2 for critical/breaking/vulnerability keywords, +0.15 for resolution patterns, +0.1 for entity mentions, +0.1 for >2KB content. Clamped to [0.0, 1.0]. Six significance flags (DECISION, ORIGIN, PIVOT, CORE, MILESTONE, PROBLEM) auto-pin DECISION and MILESTONE entries.

### Read Path (Retrieval)

**SearchFusion** orchestrates parallel execution:

1. **Intent classification** — deterministic regex into causal/temporal/lookup/recommendation/general
2. **Temporal resolution** — "3 days ago", "last Saturday" → absolute date ranges (12 hardcoded patterns)
3. **Parallel search** — BM25 (8 strategies), trigram, Levenshtein, optional vector (nomic-embed 768-dim)
4. **Score fusion** — weighted merge (BM25 0.45, trigram 0.15, Levenshtein 0.05, vector 0.35), multi-strategy confidence boost (+15% per extra strategy match)
5. **Intent-adaptive reranking** — recency/relevance/access weights shift by intent type
6. **IDF-weighted content reranking** — bigram density, synonym matching
7. **Optional LLM judge** — Claude Haiku scores top-N, 50/50 blend with retrieval score
8. **Rate limiting** — 60-second sliding window, full results for calls 1-3, limited for 4-8, blocked 9+

The 8 BM25 strategies run in sequence with cumulative scores: AND-mode (weight 2.0) → phrase matching (1.9) → entity-focused (1.8) → sanitized FTS5 (1.5) → relaxed AND (1.2) → OR with synonym expansion (1.0) → individual keywords (0.5) → individual synonym search (0.2). Temporal keyword resolution at weight 1.6 if reference date is available.

### Consolidation ("Dreamer")

Background agent running every 5 minutes:
- **Progressive compression**: verbatim (0-7d) → light (7-30d, first sentence per paragraph + keywords) → medium (30-90d, summarizer-level ~150-200 chars) → distilled (90+d, bullet facts only). High-importance entries skip a tier. Pinned entries never compress.
- **Staleness**: mark entries stale after 30 days without access, auto-archive after 90 days.
- **Contradiction detection**: word-overlap heuristic (≥4 shared words, ≥40% overlap). User must manually resolve.
- **Promotion**: entries accessed in 3+ distinct sessions get promoted to global cross-project store.
- **Causal chain extraction**: reconstructs DECISION → PROBLEM → MILESTONE trails from temporally-adjacent observations sharing entities.
- **Corroboration boost**: entries referenced multiple times get relevance bumps.

### Feedback Engine

Tracks search result IDs → checks if result's mentioned files were subsequently modified → marks as "useful". At session end, batch-updates `last_useful_at` and boosts `relevance_score`. Entries never marked useful decay faster. This is a **file-modification-based signal**, not a retrieval relevance signal — it only fires when search results mention specific files and those files are later edited.

---

## 3. Benchmark Methodology — Critical Analysis

### Claimed Results

| Benchmark | Queries | Claimed Score |
|-----------|---------|---------------|
| LongMemEval | 500 | 97.8% (pure), 100% (Haiku) |
| LoCoMo | 1,977 | 98.1% |
| MemBench | 500 | 98.0% |
| ConvoMem | 250 | 97.7% |

### Methodological Concerns

**1. Session-level granularity inflates LoCoMo scores.** The LoCoMo benchmark operates at `GRANULARITY = 'session'` by default. Each "document" is an entire session (20-50+ dialog turns joined into one text block). A typical LoCoMo conversation has ~25 sessions, so retrieval is effectively "find the right session from ~25 options at R@10." Somnigraph evaluates LoCoMo at the **turn level** (300-600 individual turns per conversation, R@10). These are not comparable tasks. A session-level R@10 of 98% could correspond to a turn-level R@10 well below 80%.

**2. Enriched ingestion uses LoCoMo metadata.** The benchmark adapter (`benchmarks/locomo.js:126-161`) appends LoCoMo's `session_summary`, `observation`, and `event_summary` metadata to each session document. This metadata contains structured facts and keywords extracted by LoCoMo's creators — it's essentially a condensed version of what the questions are asking about. This dramatically improves BM25 keyword matching. A fair comparison would ingest only the raw dialog turns.

**3. Benchmark-specific synonym expansions.** The file `benchmarks/lib/expansions.js` contains ~50 manually-curated synonym mappings targeting specific failure patterns found in LoCoMo, LongMemEval, and MemBench (e.g., `violin → practice, instrument, music, play`; `supervillain → villain, comic, hero, fan`). These are merged into the search engine via `mergeExpansions()` during benchmark runs only. The comment says "NOT part of the core product" — which means the benchmark measures a different system than what users install. This is test-set leakage by another name.

**4. Comparison framing.** The README comparison table pits context-mem against MemPalace (60.3% LoCoMo), Mem0 (~85% LongMemEval), and Zep (~85% LongMemEval). These are not the strongest systems available. No comparison is made against Somnigraph (85.1% end-to-end QA, 95.4% R@10 at turn level), MemOS (75.80 LoCoMo F1), or EXIA GHOST (89.94% LoCoMo cats 1-4). The comparison also does not note the session-vs-turn granularity difference.

**5. No end-to-end QA evaluation.** All scores are retrieval metrics (Recall@K). No reader model generates answers, no judge evaluates answer quality. This measures whether the right document was retrieved, not whether the system can actually answer questions — an important distinction when the "document" is an entire session of 50 turns.

**Bottom line on benchmarks**: The 98%+ headline numbers are not fraudulent, but they measure a substantially easier task than what other systems report on the same benchmarks. Session-level retrieval with enriched metadata and benchmark-specific synonyms is not comparable to turn-level retrieval with raw dialog only.

---

## 4. Interesting Ideas

**Adaptive compression tiers**: The progressive compression model (verbatim → light → medium → distilled) with importance-gated tier skipping is well-designed. Most systems either keep everything or delete old content. The insight that decisions and milestones deserve verbatim preservation while log output can be aggressively distilled is sound and practical for the coding-assistant use case.

**Content-type-aware summarization**: 14 deterministic summarizers (code, shell, JSON, error, TypeScript error, test output, build output, git log, CSV, HTML, markdown, network, binary) that understand the structure of each content type. No LLM needed. This is the kind of domain-specific engineering that most systems skip — they just truncate.

**4-layer wake-up primer**: Token-budgeted session initialization with layered priority: project profile (15%), critical knowledge by importance×recency×access (40%), last session decisions + open TODOs (30%), top entities by relationship count (15%). Default 700 tokens. This is a practical solution to the "what to inject at session start" problem.

**Multi-strategy confidence boost**: Results found by 2+ search strategies (BM25, trigram, Levenshtein, vector) get a per-match bonus (+15% per extra strategy). Simple but sound — agreement across heterogeneous signals is a reliable quality indicator.

**Intent-adaptive weight shifting**: The search fusion layer adjusts BM25 vs. vector weights based on classified intent (lookup → boost BM25 1.4x, reduce vector 0.5x; causal/temporal → boost vector 1.5x, reduce BM25 0.7x). Plus an AttnRes mechanism for general queries: if relevance scores have low variance (too close to distinguish), shift weight toward recency; if timestamps are clustered, shift toward relevance.

**Per-prompt memory injection**: UserPromptSubmit hook auto-injects relevant memories on every user message, rate-limited (2/min, 5-min topic cooldown, 0.6 relevance threshold). This is aggressive but addresses context drift proactively.

---

## 5. What's Not Novel

- BM25 multi-strategy search is a well-known pattern (Anserini, Pyserini, Terrier all document multi-strategy query generation)
- Entity extraction via regex + alias tables is standard NER-lite
- Knowledge graph with typed entities and relationships is table stakes
- SQLite + FTS5 is the same storage choice as Somnigraph, SLM, Claudest, etc.
- File-modification-based feedback is a weak proxy for retrieval relevance (only fires on code-related memories with explicit file paths)
- Contradiction detection via word overlap is primitive compared to NLI-based approaches (EXIA GHOST) or embedding-based approaches

---

## 6. Useful for Us

**Wake-up primer design**: The 4-layer budgeted injection approach is worth studying for `startup_load` improvements. Our current startup_load doesn't explicitly budget by layer priority — it uses category-aware retrieval but doesn't allocate token budgets to different signal types.

**Content-type summarization patterns**: If Somnigraph ever handles structured coding outputs (tool results, shell output, error traces), the 14-summarizer approach provides a good template for domain-specific compression. Currently not applicable (we store atomic facts, not raw tool output).

**Rate-limiting search calls**: The sliding-window throttle (full → limited → blocked) is a practical defense against retrieval loops. Our retrieval doesn't have this protection.

---

## 7. Not Useful for Us

**Session-level retrieval approach**: Somnigraph stores atomic facts/turns, not session blobs. Session-level retrieval avoids the hard problem (turn-level precision) in exchange for easier recall at the cost of context size.

**Deterministic importance scoring**: Somnigraph's learned reranker already outperforms any keyword-based importance heuristic. The type + keyword + entity counting approach works for ingest-time triage but can't replace a trained model for retrieval ranking.

**Auto-observe everything**: context-mem's philosophy is to capture every tool output automatically. Somnigraph gives the agent judgment about what to remember. These are fundamentally different design philosophies — context-mem optimizes for friction reduction, Somnigraph for signal quality.

**Dreamer consolidation**: The Dreamer's operations (staleness marking, archiving, compression, contradiction detection) are all simpler versions of what Somnigraph's sleep pipeline does. The contradiction detection (word overlap ≥40%) would miss most real contradictions. No graph-based operations, no feedback-driven reranker training, no Hebbian strengthening.

**Feedback engine**: File-modification correlation is too narrow a signal. It only works for code-related memories with explicit file paths. Somnigraph's explicit `recall_feedback` with Likert ratings from the agent provides a much richer signal that feeds directly into reranker training.

---

## 8. Connections

- **MemPalace** (research/sources/mempalace.md): Direct competitor. context-mem positions against MemPalace in all benchmarks. Both achieve 100% LongMemEval with LLM reranking. MemPalace uses ChromaDB (vector-first), context-mem uses SQLite/FTS5 (text-first).
- **Somnigraph**: Shared SQLite + FTS5 foundation. context-mem's 8-strategy BM25 is structurally similar to Somnigraph's FTS5 + vec hybrid, but without a learned reranker. context-mem's LoCoMo benchmark operates at session level, making direct score comparison invalid without normalization.
- **Honcho**: The "Dreamer" concept is credited to Honcho's dreamer agent, adapted for deterministic local use.
- **LoCoMo** (research/sources/locomo.md): Shared benchmark. context-mem evaluates at session level with enriched metadata; Somnigraph evaluates at turn level with raw dialog. Not comparable.
- **LongMemEval** (research/sources/longmemeval.md): Both systems achieve near-perfect scores. context-mem's 97.8% pure-local is strong but uses session-level granularity (~53 sessions per question).
- **AWM** (research/sources/awm.md): Similar architectural ambition (multi-strategy BM25, knowledge graph, consolidation). AWM has a learned reranker (cross-encoder), context-mem does not.
- **TheBrain** (research/sources/thebrain.md): Similar philosophy of deterministic extraction + importance scoring without LLM dependency. TheBrain uses brain-region metaphor; context-mem uses more conventional module names.

---

## 9. Summary Assessment

**Relevance**: Medium-low. context-mem is a well-engineered product for the AI coding assistant market, but its research contributions are limited. The retrieval pipeline is thorough BM25 engineering without learned components. The benchmark claims require significant methodological caveats.

**Novelty**: The adaptive compression tiers and 14-summarizer pipeline are genuinely useful engineering. The wake-up primer design is worth studying. The retrieval architecture itself (multi-strategy BM25 + intent classification + synonym expansion) is competent but not novel — it's standard IR techniques assembled well.

**Engineering quality**: High. 1,135 tests, minimal dependencies (3 runtime), 44 MCP tools, 14 content summarizers, multi-platform support (6 editors), dashboard with WebSocket real-time updates, CLI with init/serve/status/doctor commands. 285 commits in ~1 month is rapid development (likely AI-assisted). Recent v3.2 audit fixed 17 critical bugs, suggesting fast iteration with incomplete testing.

**Rigor concerns**:
- LoCoMo benchmark uses session-level granularity with enriched metadata — not comparable to turn-level evaluations.
- Benchmark-specific synonym expansions constitute test-set leakage. The system measured in benchmarks is not the system users install.
- Comparison table cherry-picks weaker competitors and omits granularity differences.
- No end-to-end QA evaluation (retrieval only, no reader + judge).
- "98%+" headline number is technically true but misleading in context.

**Bottom line**: context-mem is a solid product with good engineering and a thoughtful compression model. As a research artifact, the benchmark claims need to be read with the methodological caveats in mind. The session-level enriched LoCoMo evaluation and benchmark-specific synonyms make the headline numbers incomparable to turn-level evaluations. The ideas most worth understanding are the adaptive compression tiers and the wake-up primer design; the retrieval pipeline itself is competent BM25 engineering without the learned components (reranker, feedback loop, graph traversal) that differentiate systems at the research frontier.
