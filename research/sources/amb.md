# AMB: Agent Memory Benchmark -- Analysis

*Phase 28, 2026-04-11. Analysis of https://github.com/vectorize-io/agent-memory-benchmark (Vectorize.io / Hindsight team).*

---

## 1. Overview

**What it is**: AMB is not a novel benchmark dataset. It is a **meta-benchmark harness** that aggregates 7 existing datasets under a unified evaluation framework, with standardized memory-provider adapters, consistent LLM judging, and a public leaderboard at agentmemorybenchmark.ai. It was built by the Hindsight team (Vectorize.io) and announced alongside Hindsight v0.4.20 in early 2026.

**Repo**: https://github.com/vectorize-io/agent-memory-benchmark. MIT license. ~3,300 lines of Python (excluding UI). Initial commit plus ~30 follow-ups, mostly UI/deployment fixes.

**The core claim**: "We built AMB because we wanted to be honest about how Hindsight performs -- and because no existing benchmark gave us the full picture." The README positions AMB as a response to benchmarks designed for the 32K era that can't differentiate systems at million-token scale.

**What it actually does**: For each dataset/provider combination, AMB (1) ingests documents into a memory provider, (2) retrieves context per query, (3) generates answers using a Gemini model, and (4) judges correctness using a second Gemini call. The framework tracks ingestion time, retrieval latency, context token count, and accuracy.

---

## 2. Datasets Aggregated

AMB wraps 7 datasets. None are original to AMB -- all are published benchmarks repackaged into AMB's Document/Query format.

| Dataset | Source | Task Type | Queries | Total Tokens | Scale | Isolation |
|---------|--------|-----------|---------|-------------|-------|-----------|
| **BEAM** | arXiv:2510.27246 | LLM-judged (rubric) | 2,000 (4 splits) | 2.6M--110M | 100K--10M | conversation |
| **LifeBench** | arXiv:2603.03781 | LLM-judged (pass/fail) | 2,003 | 11.8M | ~1.2M/user | user |
| **LoCoMo** | snap-research | LLM-judged (pass/fail) | 1,540 | 380K | ~38K/conv | conversation |
| **LongMemEval** | xiaowu0162 | LLM-judged (pass/fail) | 500 | 56.6M | ~113K/question | question |
| **PersonaMem** | arXiv:2504.14225 | MCQ (exact match) | 5,990 (3 splits) | 1M--31M | 32K--1M | -- |
| **MemBench** | import-myself | MCQ (exact match) | 4 splits | external | varies | -- |
| **MemSim** | nuster1128 | MCQ (Chinese) | 6 splits | external | varies | -- |

### Notable design choices per dataset adapter

- **LoCoMo**: AMB excludes adversarial questions (category 5), matching most external evaluations. Uses all 10 conversations. The judge prompt is a LoCoMo-faithful "generous grading" prompt that accepts paraphrases.
- **BEAM**: AMB implements the full BEAM paper scoring methodology -- rubric-level 0/0.5/1 scores with LLM judge, Kendall tau-b for event ordering, LLM equivalence matching for item alignment. This is significantly more sophisticated than the binary pass/fail used for other datasets. Category-specific RAG prompts are heavily engineered (e.g., abstention gets "you MUST respond with..." guidance, summarization gets rubric hints injected into the prompt).
- **LongMemEval**: Each question gets its own document bank (isolation_unit="question"), meaning every question has its own haystack of sessions. Per-category judge prompts match the original paper's evaluation criteria (temporal gets off-by-one leniency, knowledge-update accepts old+new answers).
- **PersonaMem**: MCQ with exact letter match. The RAG prompt instructs focus on "pain points, past frustrations, and explicit preferences" -- a specific prompt engineering choice.
- **LifeBench**: LoCoMo-compatible format (sessions per user, dia_id evidence mapping). 10 synthetic users, 364 sessions/user (one per calendar day), 5 question categories.

---

## 3. Memory Provider Architecture

### The adapter interface

```python
class MemoryProvider(ABC):
    def ingest(self, documents: list[Document]) -> None: ...
    def retrieve(self, query: str, k: int = 10, user_id=None, query_timestamp=None) -> tuple[list[Document], dict | None]: ...
    def direct_answer(self, query: str, ...) -> tuple[str, str, dict | None]: ...  # optional
```

The `Document` model carries: `id`, `content`, `user_id`, `messages` (structured turns), `timestamp`, `context` (provenance hint). This is rich enough for most memory systems to work with.

### Providers implemented

| Provider | Type | Extraction Model | Embedding | Notes |
|----------|------|-----------------|-----------|-------|
| **Hindsight** (embedded) | local | gemini-2.5-flash-lite | internal | 3 variants: embedded, cloud, HTTP |
| **Hybrid Search** | local | -- | Qwen3-Embedding-0.6B + BM42 | Qdrant RRF fusion, k=50, chunk=512 |
| **Mem0** | local | gemini-2.0-flash | multi-qa-MiniLM-L6-cos-v1 | Also cloud variant |
| **Cognee** | local | gpt-4o-mini | BAAI/bge-small-en-v1.5 | Graph-based extraction |
| **Mastra** | local | gpt-4o-mini | FastEmbed | LibSQL store, topK=10 |
| **Supermemory** | cloud | internal | internal | Hybrid search mode |
| **BM25** | local | -- | -- | Chunk + BM25Okapi baseline |

### Three evaluation modes

1. **RAG** (default): Provider retrieves, Gemini generates answer from retrieved context.
2. **Agentic RAG**: Gemini acts as agent with a `recall` tool, can issue multiple retrieval queries.
3. **Agent**: Bypasses RAG entirely, calls provider's `direct_answer()` (Hindsight's `reflect`).

### Hindsight-specific features

The Hindsight adapter has significantly more engineering than others:

- **Dataset-specific `retain_mission`** for BEAM ("Extract ALL factual claims... including NEGATIVE statements")
- **Dataset-specific `max_tokens` and `max_chunk_tokens`** per dataset (4096--32768 for max_tokens, 8192--16384 for max_chunk_tokens)
- **Per-unit bank creation** with tag-based user scoping
- **Async operation polling** with 8-hour timeout for large documents
- **Failed operation detection** (aborts run rather than scoring incomplete ingestion)
- **`include_chunks=True`** to get source text alongside extracted facts
- **`budget="high"`** on all recall calls
- **Event loop session management** (5 separate try/except blocks managing aiohttp sessions across sync/async boundaries)

This level of Hindsight-specific tuning is not available to other providers. The hybrid-search baseline, by contrast, has a fixed k=50 and no dataset-specific configuration.

---

## 4. Evaluation Methodology

### Generation pipeline

- **Answer LLM**: Defaults to Groq (via `OMB_ANSWER_LLM`), configurable. The published Hindsight results use `gemini-2.5-flash-lite` (visible in results-manifest.json as "single-query" mode).
- **Judge LLM**: Defaults to Gemini (`gemini-2.5-flash-lite`), configurable via `OMB_JUDGE_LLM`.
- **Structured output**: Both answer and judge use JSON schema enforcement via Gemini's `response_mime_type="application/json"`.

### Scoring

- **LLM-judged datasets** (BEAM, LifeBench, LoCoMo, LongMemEval): Binary correct/incorrect from judge, except BEAM which uses continuous 0--1 rubric scores.
- **MCQ datasets** (PersonaMem, MemBench, MemSim): Exact letter match, no LLM judge needed.
- **BEAM scoring**: Per-rubric-item 0/0.5/1 from LLM judge, averaged per question. Event ordering uses Kendall tau-b with LLM-based item alignment. This is faithful to the BEAM paper methodology.
- **Empty context handling**: If retrieval returns nothing, the query is marked incorrect without calling the judge.

### Auxiliary metrics

- Ingestion time (wall clock)
- Per-query retrieval time
- Average context tokens
- Oracle mode (ingest only gold documents, isolating retrieval quality from generation quality)

---

## 5. Published Results

From `results-manifest.json` and `external_results.json`:

### AMB-evaluated results

| Dataset | Split | Hindsight | Hybrid Search | Cognee | Mem0 |
|---------|-------|-----------|---------------|--------|------|
| BEAM | 100k | **73.4%** | -- | -- | -- |
| BEAM | 500k | **71.1%** | -- | -- | -- |
| BEAM | 1m | **73.9%** | -- | -- | -- |
| BEAM | 10m | **64.1%** | -- | -- | -- |
| LifeBench | en | **71.5%** | 61.0% | -- | -- |
| LoCoMo | locomo10 | **92.0%** | 79.1% | 80.3% | -- |
| LongMemEval | s | **94.6%** | 74.0% | -- | -- |
| PersonaMem | 32k | **86.6%** | 84.4% | 81.8% | -- |

### External results (from papers/blogs, curated by AMB team)

LoCoMo locomo10 external: Honcho 89.9%, MemMachine v0.2 91.7% (gpt-4.1-mini), Letta 74.0%, Memobase 75.8%, Zep 75.1%.

PersonaMem 128k external: Full-context LLMs range 26--52% (Claude-3.7-Sonnet at 26%, GPT-4.5 at 52%).

The external results include candid annotations: "Self-reported by MemMachine. LLM-as-a-judge with GPT-4o-mini. Adversarial questions excluded. Uses gpt-4.1-mini as backbone -- stronger LLM directly inflates scores vs. gpt-4o-mini entries." This level of annotation honesty is genuinely good practice.

---

## 6. Task Taxonomy Comparison

### AMB's coverage (via aggregated datasets)

Since AMB aggregates existing benchmarks, its taxonomy is the union of its constituent datasets. Mapping to standard taxonomies:

| Competency | BEAM | LoCoMo | LongMemEval | LifeBench | PersonaMem |
|------------|------|--------|-------------|-----------|------------|
| Single-fact recall | information_extraction | single-hop | single-session-user/assistant | information-extraction | recall_user_shared_facts |
| Multi-hop reasoning | multi_session_reasoning | multi-hop | multi-session | multi-hop | -- |
| Temporal reasoning | temporal_reasoning | temporal | temporal-reasoning | temporal-updating | -- |
| Knowledge update | knowledge_update | -- | knowledge-update | -- | track_full_preference_evolution |
| Preference tracking | preference_following | -- | single-session-preference | nondeclarative | acknowledge_latest_user_preferences |
| Contradiction detection | contradiction_resolution | -- | -- | -- | -- |
| Abstention (knowing unknowns) | abstention | -- | -- | unanswerable | -- |
| Event ordering | event_ordering | -- | -- | -- | -- |
| Instruction following | instruction_following | -- | -- | -- | -- |
| Summarization | summarization | -- | -- | -- | -- |
| Recommendation/generalization | -- | -- | -- | -- | suggest_new_ideas, generalize_to_new_scenarios |

**vs. BEAM (10 abilities)**: AMB includes BEAM directly, so all 10 abilities are present via that dataset.

**vs. LongMemEval (5 abilities)**: All 5 LongMemEval types are present (single-session-user, single-session-assistant, multi-session, temporal-reasoning, knowledge-update, single-session-preference -- actually 6).

**vs. LoCoMo (5 types)**: 4 of 5 present (adversarial excluded). Single-hop, multi-hop, temporal, open-domain.

**vs. MemoryAgentBench (4 competencies)**: MemoryAgentBench tests information retention, relation association, temporal understanding, and knowledge updating. AMB covers all four via BEAM + LoCoMo.

**Unique to AMB via aggregation**: The combination of PersonaMem's MCQ preference tracking at 1M-token scale + BEAM's 10M-token conversations + LifeBench's multi-source daily life data creates a scale diversity that no single benchmark offers.

---

## 7. Critical Assessment

### Conflict of interest

This is the central concern. Hindsight's team built AMB, and Hindsight tops the leaderboard on every dataset where it's evaluated. The specific risks:

1. **Prompt engineering asymmetry**: The Hindsight adapter has dataset-specific `retain_mission` strings, dataset-specific `max_tokens`/`max_chunk_tokens`, `budget="high"`, tag-based scoping, and chunk inclusion. Other providers get generic configuration. The hybrid-search baseline has no dataset-specific tuning.

2. **No full-context baseline**: AMB explicitly argues that "dump everything into context" is competitive -- but doesn't include it as a baseline. For PersonaMem 128k, external results show full-context LLMs scoring 26--52%, vs. Hindsight at 86.6% on 32k. But these are different splits and models -- not a controlled comparison.

3. **Selective provider coverage**: Hindsight has results on all datasets. Cognee has results on 2 datasets (LoCoMo, PersonaMem 32k). Hybrid search has results on 4 datasets. No provider other than Hindsight has been evaluated on BEAM or LongMemEval. The leaderboard compares incomplete evaluation sets.

4. **Judge model alignment**: Both answer generation and judging default to Gemini models. Hindsight's extraction model is also Gemini (gemini-2.5-flash-lite). If Hindsight's extracted facts are phrased in Gemini's style, the Gemini judge may be more lenient toward them than toward raw chunks from hybrid search. This is speculative but structurally possible.

### What AMB does well

1. **Reproducibility**: Everything is published -- prompts, judge prompts, model IDs, raw outputs. The emphasis on reproducibility is genuine and rare among benchmark publications.

2. **Scale coverage**: BEAM's 10M-token split (110M tokens total across 10 conversations) is genuinely beyond what most benchmarks offer. This is where memory systems should differentiate.

3. **Honest external annotations**: The `external_results.json` annotates every entry with its source, methodology, and potential biases ("evaluated by a competitor", "stronger LLM directly inflates scores").

4. **Engineering quality**: The codebase is clean, well-structured, and functional. The dataset adapters handle real edge cases (LoCoMo dia_id mapping, BEAM 10M nested plan structures, PersonaMem session splitting). The runner supports incremental saving, crash recovery, and oracle mode.

5. **BEAM scoring fidelity**: The BEAM adapter implements the full paper scoring methodology faithfully -- rubric-level judging, Kendall tau-b, LLM equivalence matching. This is non-trivial work.

### What AMB does not do well

1. **No novel evaluation contribution**: AMB is a harness, not a benchmark. Every dataset and every evaluation methodology is borrowed. The "benchmark" contribution is packaging + leaderboard + prompt engineering.

2. **Unfair baseline comparison**: Hybrid search (Qdrant RRF, Qwen3-Embedding-0.6B, 512-token chunks) is a reasonable but untuned baseline. Hindsight has dataset-specific extraction missions, token budgets, and budget="high". A fair comparison would give equal tuning effort to all providers.

3. **No retrieval-only evaluation**: AMB measures end-to-end accuracy (retrieval + generation + judging). It cannot isolate retrieval quality. Oracle mode tests generation quality given perfect retrieval, but there's no mode that evaluates retrieval recall/precision independently.

4. **Missing ablation of the judge**: We know from our own work that LLM judges accept ~63% of intentionally vague wrong answers (LoCoMo) and that judge choice swings scores by 3+ percentage points. AMB uses Gemini for judging across all datasets but doesn't report judge calibration or inter-annotator agreement.

5. **Misleading "single-query" mode name**: The results-manifest.json shows mode as "single-query" for what's actually RAG mode. This is confusing naming.

---

## 8. Relevance to Somnigraph

### Could we run against AMB?

**Technically yes**, with moderate effort. The adapter interface (`ingest` + `retrieve`) maps cleanly to Somnigraph's `remember` + `recall`. We'd need:

1. A `SomnigraphMemoryProvider` class that wraps our MCP tools (or imports from `memory.tools` directly).
2. Document-to-memory ingestion: each Document would need `remember()` calls, but AMB documents are full conversation sessions, not individual memories. We'd need an extraction step -- either our existing sleep pipeline or a simpler pre-processing pass.
3. Retrieval: `recall()` maps directly to AMB's `retrieve()`.

**Estimated effort**: 2--4 sessions for the adapter + one full evaluation run per dataset.

### What AMB would reveal that LoCoMo doesn't

1. **Scale behavior**: BEAM 100K--10M tests whether our retrieval degrades at scale. LoCoMo is ~38K/conversation -- well within our comfort zone.
2. **PersonaMem preference tracking**: MCQ format gives unambiguous scoring. PersonaMem's 1M split would test our system at a scale where full-context LLMs score 28--45%.
3. **LongMemEval's per-question isolation**: Each question gets its own document bank, which is a different pattern from LoCoMo's shared conversations.

### What AMB would NOT reveal

1. **Retrieval quality in isolation**: AMB only measures end-to-end accuracy. Our LoCoMo pipeline gives us R@10, MRR, NDCG -- much more diagnostic.
2. **Sleep/consolidation value**: AMB doesn't test memory evolution over time. Documents are ingested in batch and queried immediately.
3. **Feedback loop value**: No mechanism for iterative retrieval feedback.
4. **Graph contribution**: AMB can't measure whether our graph-augmented retrieval (synthetic nodes, coref edges) helps compared to flat retrieval.

### Recommendation

**Low priority for running**. AMB's value to us is primarily as a marketing/comparison data point ("we beat X on AMB"), not as a diagnostic tool. Our LoCoMo pipeline with retrieval-level metrics + ablations is more informative for research purposes. If we want a second benchmark, PERMA (already Priority 5) is more aligned with our graph strengths.

However, **the LoCoMo adapter code is worth studying** -- their judge prompts, dataset loading, and category handling are clean implementations that we could compare against our own.

---

## 9. Insights Worth Stealing

### 1. Oracle mode

AMB's `--oracle` flag ingests only gold documents, isolating generation quality from retrieval quality. This is trivially useful for diagnosing whether failures are retrieval or reader errors. We could add this to our LoCoMo pipeline with minimal effort.

### 2. Multi-mode evaluation

The RAG / Agentic RAG / Agent mode split is a clean abstraction. Agentic RAG (LLM issues multiple recall queries) is an evaluation mode we haven't tried. If our system supports it, it could reveal whether multi-hop failures are addressable by query decomposition rather than retrieval improvements.

### 3. BEAM dataset adapter

AMB's BEAM implementation is thorough and could save us significant effort if we decide to evaluate on BEAM. The rubric scoring, event ordering with Kendall tau-b, and LLM equivalence matching are non-trivial.

### 4. External results with annotations

The `external_results.json` pattern -- curating competitor results with methodology notes and bias annotations -- is excellent practice for honest benchmarking. Worth adopting if we publish comparative results.

### 5. Ingestion/retrieval time tracking

AMB records wall-clock ingestion time and per-query retrieval time alongside accuracy. This is useful context that our LoCoMo pipeline doesn't track.

---

## 10. What's Not Worth It

1. **Running AMB as primary evaluation**: The harness measures end-to-end accuracy with a fixed Gemini reader/judge. We can't swap in our reranker training pipeline, can't get retrieval-level metrics, and can't run ablations.

2. **MemBench and MemSim datasets**: MemBench requires a separate Google Drive download and tests agent trajectories (not conversation memory). MemSim is Chinese-language. Neither aligns with our use case.

3. **The Vercel UI/leaderboard**: The web UI is a presentation layer, not a research tool. It serves Hindsight's marketing needs.

4. **Matching AMB's Gemini pipeline**: AMB's answer generation defaults to Groq/Gemini, and judging to Gemini. Matching this exactly for comparison purposes would add complexity without diagnostic value. If we run AMB, using our existing GPT-4.1-mini reader is fine -- the comparison should be retrieval quality, not reader quality.

---

## 11. Summary Verdict

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Novel contribution | **Low** | Harness, not benchmark. All datasets are existing. |
| Engineering quality | **High** | Clean codebase, thorough dataset adapters, good error handling. |
| Evaluation fairness | **Medium-Low** | Hindsight adapter is heavily tuned; baselines are generic. |
| Reproducibility | **High** | Everything published: prompts, models, raw outputs. |
| Relevance to Somnigraph | **Medium** | Scale testing is valuable; diagnostic depth is poor. |
| Conflict of interest risk | **High** | Benchmark builder tops own leaderboard with tuned adapter. |
| Research value | **Low-Medium** | Useful as a comparison point, not as a research instrument. |

AMB is a well-engineered evaluation harness with a genuine conflict-of-interest problem. The Hindsight adapter receives substantially more tuning than any other provider, making the leaderboard results non-comparable. The honest external result annotations and BEAM scoring implementation are genuinely good contributions. For Somnigraph, AMB is a "nice to have" comparison point but not a research priority -- our LoCoMo pipeline with retrieval-level metrics remains more diagnostic.
