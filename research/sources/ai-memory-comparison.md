# AI-Memory-Comparison (carsteneu) — Community-curated landscape index and benchmark leaderboard for 74 AI memory systems

*Generated 2026-06-28 by Sonnet agent reading local clone*

---

## What This Is

Not a memory system. A community-curated, source-backed feature-level comparison of 74 AI memory systems designed for coding agents. The repository's value is as a reading-list generator and feature-vocabulary reference, not as a source of architectural insight.

**Repo**: https://github.com/carsteneu/ai-memory-comparison  
**License**: CC0 (PR-welcome, submission open)  
**Live site**: carsteneu.github.io/ai-memory-comparison  
**Last commit**: 2026-06-27  
**Maintainer disclosure**: Authored by the creator of YesMem, which is listed alongside all others under the same evidence rules.

---

## Architecture (Index Structure)

**Scope**: 74 systems × 79 features across 7 axes: Architecture, Data Model, Search & Retrieval, Knowledge Lifecycle, Extraction Pipeline, Platform Support, and Benchmarks.

**Seven axes** (from CRITERIA.md, verified in local clone):

| Axis | Feature count | Highlights for Somnigraph |
|------|--------------|--------------------------|
| Architecture | 16 | offline, scheduled/autonomous, procedural memory, cache optimization, privacy/encrypt |
| Data Model | 16 | keywords/tags, source attribution, layered memory, conflict surfacing, **anticipated queries** (gap) |
| Search & Retrieval | 10 | full-text, semantic/vector, hybrid BM25+Vec, fact metadata query, **timeline view** (gap) |
| Knowledge Lifecycle | 7 | decay/forgetting, supersede/replace, contradiction detect, explicit forget |
| Extraction Pipeline | 8 | auto-extraction, dedup, quality refinement, narrative generation, recurrence detection |
| Platform Support | 11 | Claude Code through Antigravity — mostly irrelevant for single-agent design |
| Benchmarks | 5 | LoCoMo, LongMemEval, PersonaMem, token reduction, methodology open |

**Key files** (local clone: `evidence/`, `CRITERIA.md`, `comparison.md`, `data.js`):

- `CRITERIA.md` — precise, contestable definitions for all 79 features. This is the document to read. Definitions like "anticipated queries" (generates predicted search queries for each memory entry, not just keywords) and "layered memory" (L0 raw → L1 summary → L2 persona/semantic; compression artifacts don't count unless they constitute deliberate architectural layers) are tighter than anything in individual system READMEs.
- `evidence/` — 74 per-system files, each citing specific source lines, READMEs, or paper sections for every ✅. `evidence/byterover.md` is a model of rigor: it identifies features incorrectly listed as absent, documents borderline cases with reasoning, and verifies absent features with explicit negative evidence.
- `data.js` — structured source-of-truth (140 KB), generated into `comparison.md` and `index.html` via `build.js`.
- `comparison.md` — auto-generated Markdown table (63 KB). The Benchmarks section is the most load-bearing part for Somnigraph.

**Methodology rule**: "Code beats docs — if docs claim a feature but the implementation doesn't exist, it's ❌." Every ✅ must cite a public source. Systems inactive for extended periods may be removed ("not abandoned" freshness signal).

**Sorting**: Live table sorts by ascending star count ("underdogs first"), which surfaces newer systems rather than burying them behind established names.

**Curation criteria**: Specifically for AI agent memory (not general vector DBs or RAG frameworks); open source only; designed for coding agents (Claude Code, Codex, OpenCode, etc.).

---

## Key Claims & Evidence (Benchmark Leaderboard)

The benchmark table in `comparison.md` is where the index earns its keep. All scores drawn from `comparison.md` § "Benchmarks (published)".

| System | LoCoMo | LongMemEval | Token reduction | Methodology open | Assessment |
|--------|--------|-------------|-----------------|------------------|------------|
| ByteRover | 96.1 | 92.8 | — | ✅ | **Needs scrutiny.** arXiv:2604.01599. BM25-only (MiniSearch; no vector search), context tree with git-like VC, importance × recency × maturity decay scoring. Highest LoCoMo number but uses a fundamentally different retrieval stack than hybrid systems; comparison is apples-to-oranges. Analyzed in `byterover.md`. |
| EverOS | 93.05 | 83.00 | — | ✅ | Self-evolving agent memory. LoCoMo 93.05 is second-highest with open methodology. Not yet analyzed. |
| memU | 92.09 | — | — | ✅ | Always-on 3-tier memory, 5 modality preprocessing. Third-highest LoCoMo, no LME score. Not yet analyzed. |
| Mem0 | 91.6 | 94.8 | — | ✅ | Well-documented methodology; arXiv:2504.19413 analyzed in `mem0-paper.md`. |
| MemPalace | 88.9 | 96.6 | — | ✅ | **Retracted headline.** Verbatim storage, ChromaDB. The 96.6 LME score is the current published number; an earlier headline claim (100%) was test-set-tuned and retracted. Analyzed in `mempalace.md`. |
| agentmemory | — | 95.2 | 92% fewer | ✅ | 53 MCP tools, 12 hooks, 4-tier lifecycle, 3-way RRF, SQLite, pi native. LME 95.2 is second-highest published after MemPalace. Not yet analyzed; next priority. |
| MIRIX | 85.38 | — | — | ✅ | 6-type memory architecture, 99.9% storage reduction claim. LoCoMo 85.38 ≈ Somnigraph (85.1 Opus judge). Not yet analyzed; second priority. |
| Somnigraph | 85.1 | — | — | not listed | Comparable to MIRIX on LoCoMo; not submitted to the index. |

**Cross-vendor comparability warning**: Every score in this table uses a different reader LLM, judge LLM, and evaluation script. ByteRover's 96.1 on LoCoMo uses its own evaluation harness (BM25-only retrieval makes it structurally different from hybrid systems). MemPalace's 96.6 on LME is raw retrieval on verbatim storage, not comparable to systems that store extracted summaries. Mem0 and agentmemory use different LME versions (original vs -S). These are benchmark marketing numbers, not a calibrated cross-system ranking. Use the leaderboard as a "who's worth reading" signal, not a performance ranking.

---

## Relevance to Somnigraph

### What this index does that Somnigraph doesn't

1. **Feature vocabulary (CRITERIA.md's 79 dimensions)**. The precise definitions give language to describe Somnigraph features accurately. Notable gaps this surfaces:
   - **Anticipated queries**: at ingest time, generate predicted search queries per memory entry to improve recall. CRITERIA.md definition is tight. Somnigraph has no equivalent — theme injection partially covers this but not the query-prediction sense. Worth considering; would complement BM25 term injection.
   - **Timeline view**: chronological browsing or temporal search (`since`/`before` parameters). Somnigraph has no temporal query interface (`valid_from`/`valid_until` fields exist in the schema but no MCP tool exposes them as filter parameters).
   - **Quarantine**: exclude a session's memories from retrieval without deleting. Not in Somnigraph's tool surface.

2. **External audit trail (evidence files)**. Submitting Somnigraph would force precise documentation of each claimed feature and produce an audit in a CC0 repo. The contribution process (CONTRIBUTING.md + evidence file template) is a structured feature-description exercise.

3. **Benchmark-ranked reading list**. The leaderboard directly identifies the highest-leverage systems to analyze next (see Worth Stealing below).

### Somnigraph self-assessment against CRITERIA.md axes

Estimated coverage: ~45-50% of 79 boolean features. Key gaps by axis:

| Axis | Estimated Somnigraph ✅ | Key gaps |
|------|------------------------|----------|
| Architecture | offline ✅, scheduled/autonomous ✅, privacy ✅, procedural memory ✅ | single binary ❌, web/TUI ❌, multi-agent ❌ |
| Data Model | keywords/tags ✅ (themes), source attribution ✅, layered memory ✅, conflict surfacing ✅ | anticipated queries ❌, timeline view ❌, trigger rules ❌ |
| Search & Retrieval | full-text ✅, semantic/vector ✅, hybrid ✅, fact metadata query ✅ | code graph ❌, docs search ❌, timeline view ❌ |
| Knowledge Lifecycle | decay/forgetting ✅, contradiction detect ✅, explicit forget ✅, trust model partial | quarantine ❌ |
| Extraction Pipeline | auto-extraction ✅, deduplication ✅, quality refinement ✅, narrative generation ✅, recurrence detection ✅ | clustering ❌, persona extraction ❌ |
| Benchmarks | — (not submitted, 85.1% LoCoMo informal) | methodology not published |

### What Somnigraph does better

The index's binary feature encoding is the key structural limitation: "Hybrid (BM25+Vec)" marks every system doing any BM25+vector combination the same ✅ regardless of whether it's naive concatenation, learned RRF, or a 26-feature LightGBM reranker. Somnigraph's learned reranker, explicit feedback loop (per-query utility ratings with EWMA aggregation), PPR graph expansion, and Hebbian PMI co-retrieval boost are all invisible to this framework. The coverage % metric treats all 79 features equally, so platform support for Antigravity CLI counts the same as contradiction detection. Don't optimize for coverage %.

---

## Worth Stealing (ranked)

### 1. Benchmark-ranked reading list (Low effort, High impact)

**What**: Use the leaderboard to prioritize next analyses. The two highest-leverage unanalyzed systems with open methodology: agentmemory (LME 95.2, 92% token reduction, 3-way RRF, SQLite — close to Somnigraph's stack) and MIRIX (LoCoMo 85.38, 6-type memory architecture, directly comparable score to Somnigraph).  
**Why**: These are the systems most likely to yield actionable architectural differences at similar benchmark positions.  
**How**: Read `evidence/agentmemory.md` and `evidence/mirix.md` in the clone first (fast scan of what features they claim), then deep-read the repos.

### 2. CRITERIA.md as feature vocabulary (Low effort, Medium impact)

**What**: Map Somnigraph's current features against CRITERIA.md's 79 definitions. For each ✅ Somnigraph can claim, verify the implementation actually meets the criterion (not just the self-description). For each ❌, assess whether it's a genuine gap worth closing.  
**Why**: "Anticipated queries" is the most interesting gap — not just a feature Somnigraph lacks but a retrieval-vocabulary technique that would complement existing theme injection. CRITERIA.md's definition is tighter than anything in the literature.  
**How**: Work through CRITERIA.md sections in order. The Data Model and Search & Retrieval axes are the highest-signal for Somnigraph.

### 3. Evidence file submission (Low effort, Low-medium impact)

**What**: Submit Somnigraph to the index. CC0, PR-welcome, `evidence/_TEMPLATE.md` provides the structured form.  
**Why**: Forces precise documentation of each feature claim; produces an external audit; adds visibility in a community directory. The evidence file process is as valuable as the listing itself.  
**How**: Fill out `evidence/_TEMPLATE.md` section by section against CRITERIA.md definitions. Estimated coverage ~45-50% of 79 features given gaps in platform support, timeline view, quarantine, and benchmarks columns.

---

## Not Useful For Us

**Binary feature encoding for retrieval sophistication**: The framework cannot distinguish Somnigraph's 26-feature LightGBM reranker from a naive score average. "Hybrid (BM25+Vec)" gets a single ✅. The comparison gives no signal on retrieval quality, fusion sophistication, or feedback loop presence. Don't use the feature table to assess retrieval systems — read the code.

**Coverage % as a quality signal**: Treats all 79 features equally. Platform support for Antigravity CLI counts the same as contradiction detection. A system at 70% coverage may be weaker on every dimension that matters for Somnigraph than a system at 30%. Ignore this number.

**Platform support section**: 11 columns including Antigravity, pi/omp, OpenClaw — niche or ephemeral agents. No architectural signal; skip.

**Sleep/consolidation evaluation**: The framework has no axis for offline consolidation quality. NREM/REM distinction, contradiction resolution during sleep, and pattern synthesis are invisible. "Scheduled/autonomous" (background daemon) and "narrative generation" (session summaries) partially overlap but don't capture what Somnigraph's sleep pipeline does.

---

## Connections

- **byterover.md**: ByteRover is the LoCoMo leader (96.1) in this index. Evidence file `evidence/byterover.md` in the clone is unusually rigorous — identifies errors in the comparison table itself and documents borderline cases. Already analyzed.
- **mempalace.md**: MemPalace is the LME leader (96.6) in this index. Already analyzed. Verbatim storage; the 96.6 score reflects raw retrieval, not memory quality.
- **memmachine.md**: MemMachine (analyzed) has the strongest LoCoMo claim in the literature (0.9169, gpt-4.1-mini) but does not appear in the benchmark table — its methodology apparently didn't qualify under the index's reproducibility criterion.
- **similar-systems.md**: The index covers 74 systems; `docs/similar-systems.md` is Somnigraph's own comparative analysis covering a subset. The index is a useful gap-check against similar-systems.md's 14 profiled systems.
- **index.md**: agentmemory and MIRIX are the two systems in the leaderboard not yet in the Somnigraph source corpus. They are the natural next analyses.
- **locomo.md**: The LoCoMo benchmark (analyzed separately) is the most-used evaluation axis in the index's benchmark table. Cross-referencing the index's leaderboard against locomo.md's methodology notes helps contextualize which scores are comparable.
- **mem0-paper.md**: Mem0 at LoCoMo 91.6 / LME 94.8 with open methodology is the most legible leaderboard entry — arXiv:2504.19413 analyzed, evaluation scripts public. Other top-scorers (ByteRover, MemPalace) have less comparable setups.
- **hypermem.md**: HyperMem (paper) claims 92.73% LoCoMo SOTA but does not appear in the index — because it's a paper without a deployable repo that meets the "coding agent memory" curation criteria. The index covers deployable systems only; paper-only claims are absent.
- **memos.md**: MemOS analyzed; appears in index at LoCoMo 75.80 / LME +40.43% (relative) / PersonaMem +40.75% (relative). The relative-change format for LME and PersonaMem is not commensurable with absolute scores from other systems; the index records what was published without standardizing format.

---

## Summary Assessment

This is the best publicly available directory for the AI memory space — 74 systems, source-backed evidence files, a benchmark leaderboard, and a rigorously defined 79-feature vocabulary in CRITERIA.md. No comparable resource exists for scanning the landscape before deciding what to analyze next. The "code beats docs" methodology and per-system evidence files are genuinely unusual for a community comparison table.

Its value is as a target-finder, not a source of architectural insight. The binary feature encoding cannot distinguish retrieval sophistication, and the benchmark leaderboard conflates scores across different LLM judges, evaluation harnesses, and dataset splits. ByteRover's 96.1 on LoCoMo (BM25-only) and MemPalace's 96.6 on LME (verbatim storage) are not commensurable with each other or with hybrid systems like Somnigraph. The leaderboard tells you who to read; it doesn't tell you who built a better system.

The two immediate uses: (1) treat the benchmark table as a ranked reading list — agentmemory (LME 95.2, SQLite, 3-way RRF) and MIRIX (LoCoMo 85.38, comparable to Somnigraph's own score, 6-type architecture) are the next deep-read targets; (2) work through CRITERIA.md against Somnigraph's feature set to identify vocabulary gaps, especially "anticipated queries" (predicted search queries per memory entry at ingest time) and "timeline view" (temporal search interface on `valid_from`/`valid_until`). Submitting Somnigraph as a PR is low-cost and forces the feature-documentation exercise.
