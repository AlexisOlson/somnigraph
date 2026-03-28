# Source Analysis Agent Prompt

This file is the reusable prompt template for launching subagents to analyze memory systems (papers, repos, or both). Copy the relevant sections, fill in the `{placeholders}`, and pass as the agent prompt.

---

## Prompt

You are analyzing **{system_name}** for the Somnigraph research corpus. Your output will be saved to `research/sources/{filename}.md` in the Somnigraph repo.

### What to read

{reading_instructions}

### What to produce

Write a single markdown file following the structure below. Target 200-300 lines. Be specific and evidence-based — cite file paths, function names, line numbers, or paper sections. Do not pad with filler.

---

## Output Structure

```markdown
# {System Name} — {One-line description}

*Generated {date} by {model} agent reading {source_type}*

---

## Paper Overview
<!-- Skip this section if no paper exists -->

**Paper**: {citation with arXiv ID, venue, date, page count}
**Authors**: {names and affiliations}
**Code**: {repo URL if any}

**Problem addressed**: {What gap does the paper claim to fill? 2-3 sentences.}

**Core approach**: {What is the system/method? 3-5 sentences covering the key architectural idea.}

**Key claims**: {Bulleted list of quantitative claims with evidence quality notes.}

---

## Architecture

<!-- For repos: describe what the code actually does, not what the README claims.
     For papers: describe the proposed system.
     For both: reconcile any gaps between paper claims and code reality. -->

### Storage & Schema
{What databases/stores? What is the memory unit schema? What metadata fields?}

### Memory Types
{What categories/types of memory does the system distinguish? How are they represented?}

### Write Path
{What happens when a memory is stored? Extraction? Enrichment? Deduplication?}

### Retrieval
{Search channels (vector, BM25, graph, etc.), fusion method, reranking, scoring formula.}

### Consolidation / Processing
{Offline processing, sleep-like cycles, summarization, garbage collection. "None" is a valid answer.}

### Lifecycle Management
{Decay, archival, deletion, versioning. "None" is a valid answer.}

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| {claim} | {what supports it — benchmark numbers, ablation, or just assertion} | {your assessment: validated/plausible/unvalidated/questionable} |

---

## PERMA Benchmark Results
<!-- Include if the system was benchmarked in PERMA (arXiv:2603.23231). Skip otherwise. -->

| Setting | MCQ Acc. | BERT-F1 | Memory Score | Context Token | Completion | Turn=1 | Turn≤2 |
|---------|----------|---------|-------------|---------------|------------|--------|--------|
| Clean Single | | | | | | | |
| Clean Multi | | | | | | | |
| Noise Single | | | | | | | |
| Noise Multi | | | | | | | |
| Style-aligned Single | | | | | | | |
| Style-aligned Multi | | | | | | | |

**Notable patterns**: {What stands out? Where does it excel or collapse?}

---

## Relevance to Somnigraph

### What {system_name} does that Somnigraph doesn't
{Capabilities, features, or architectural choices that Somnigraph lacks. Be specific — name the Somnigraph module where the gap exists.}

### What Somnigraph does better
{Where Somnigraph's architecture is stronger. Reference specific modules (reranker.py, scoring.py, sleep_nrem.py, etc.).}

---

## Worth Stealing (ranked)

### 1. {Idea name} ({effort: Low/Medium/High})
**What**: {The idea, concretely.}
**Why**: {Why it matters for Somnigraph specifically.}
**How**: {Implementation sketch — which files would change, what the mechanism would be.}

### 2. ...

---

## Not Useful For Us

### {Feature/idea}
{Why it doesn't apply. 1-2 sentences.}

---

## Connections

{How does this system relate to other systems we've analyzed? Cross-reference specific source analyses by filename (e.g., "see memos.md", "convergent with memv's supersession pattern"). Note where multiple systems independently arrive at the same solution — that's strong evidence.}

---

## Summary Assessment

{2-3 paragraphs. What is this system's core contribution? What's the single most important thing for Somnigraph to take from it? What's overhyped or missing? Be honest.}
```

---

## Somnigraph Context for Agents

Include a condensed version of this section in each agent prompt so the agent knows what Somnigraph is and can write informed comparisons.

### Somnigraph in brief

Somnigraph is a persistent memory system for Claude Code. SQLite + sqlite-vec + FTS5, single-user, MCP-based.

**Retrieval**: Hybrid BM25 + vector search with RRF fusion (k=14, Bayesian-optimized). 26-feature LightGBM reranker trained on 1032 real-data queries (NDCG=0.7958, +6.17pp over hand-tuned formula). PPR-based graph expansion via edges detected during sleep.

**Feedback loop**: Explicit per-query utility ratings (0-1 float + durability). EWMA aggregation, UCB exploration bonus. Per-query Spearman r=0.70 with ground truth. Hebbian co-retrieval PMI strengthens edges between memories frequently retrieved together.

**Graph**: Typed edges (supports, contradicts, evolves, revision, derivation) detected during NREM sleep. PPR expansion, novelty-scored adjacency, betweenness centrality as reranker feature.

**Sleep pipeline**: Three-phase offline consolidation — NREM (pairwise classification, edge creation, merge/archive), REM (gap analysis, question generation, taxonomy), orchestrator. LLM-mediated, not rule-based.

**Decay**: Per-category exponential decay with configurable half-lives, floors, and reheat on access.

**Schema**: Memories have category (episodic/semantic/procedural/reflection/meta), priority 1-10, themes array, summary, valid_from/valid_until, decay_rate. Layers: detail/summary/gestalt.

**Benchmarks**: 85.1% on LoCoMo QA (Opus judge). 26-feature reranker. Multi-hop failure analysis shows 88% vocabulary gap as the retrieval ceiling.

**Key architectural differentiators**: (1) Learned reranker, (2) Explicit feedback loop with measured GT correlation, (3) Offline LLM-mediated consolidation (sleep), (4) Graph-conditioned retrieval via PPR.

**What Somnigraph lacks**: Real-time graph construction (graph builds during sleep, not at write time), entity resolution, multi-user support, write-path quality gating, preference-state tracking, cross-domain synthesis evaluation.

### Key files

| Module | Purpose |
|--------|---------|
| `src/memory/reranker.py` | 26-feature LightGBM reranker, cached meta-computation |
| `src/memory/scoring.py` | PPR expansion, novelty-scored adjacency, formula fallback |
| `src/memory/tools.py` | MCP tool implementations (remember, recall, forget, etc.) |
| `src/memory/fts.py` | FTS5 BM25 search with theme/summary boosting |
| `src/memory/embeddings.py` | Enriched embedding (content + category + themes + summary) |
| `src/memory/db.py` | SQLite schema, edge table, memory CRUD |
| `scripts/sleep_nrem.py` | Pairwise classification, edge creation, merge/archive |
| `scripts/sleep_rem.py` | Gap analysis, question generation, taxonomy |
| `docs/architecture.md` | Master narrative of design decisions |
| `docs/similar-systems.md` | Comparative analysis with 14 profiled systems |
| `research/sources/` | 90+ source analyses (this is where output goes) |

---

## Checklist Before Launching

- [ ] `{reading_instructions}` filled in (paper PDF path, repo path, specific files to read)
- [ ] PERMA results included if the system was benchmarked there
- [ ] Existing analysis referenced if this is an update (pass the old analysis file path)
- [ ] Output filename specified
- [ ] Date filled in
