# memory-lancedb-pro — OpenClaw memory plugin with write-time typed-relationship dedup, a rule-based "dreaming" sidecar, and Weibull decay

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

memory-lancedb-pro is a ~25k-LOC TypeScript OpenClaw plugin (also installable as a Claude Code skill), LanceDB-backed, ~4.4k GitHub stars. It is a *product*, not a research artifact: 12-language READMEs, multi-scope agent isolation, CLI, OAuth LLM config. But under the packaging there is a genuinely rich write path. This is a code-level read of `src/*.ts` (the `dist/` mirrors it).

### Storage & Schema
- **LanceDB** single table (`src/store.ts`, ~2680 lines). 8 columns: `id`, `text`, `vector`, `category`, `scope`, `importance`, `timestamp`, `metadata` (JSON blob holding everything else). No SQLite/FTS-native schema — BM25 is via LanceDB's FTS index.
- **Layered memory unit** (`memory-categories.ts`): every candidate carries `abstract` (L0 one-sentence index), `overview` (L1 structured markdown), `content` (L2 full narrative) — a semantic pyramid that maps directly onto Somnigraph's detail/summary/gestalt layers.
- **Multi-scope**: `global`, `agent:<id>`, `project:<id>`, `user:<id>`, `custom:<name>` with per-agent access control. (Somnigraph is single-user; not relevant.)

### Memory Types
Six write categories (`memory-categories.ts`): `profile`, `preferences`, `entities`, `events`, `cases`, `patterns` (UserMemory vs AgentMemory split). Crucially, **category drives dedup policy**:
- `ALWAYS_MERGE`: profile
- `MERGE_SUPPORTED`: preferences, entities, patterns
- `TEMPORAL_VERSIONED` (supersedable): preferences, entities
- `APPEND_ONLY` (create/skip only): events, cases

Three lifecycle **tiers** (`core`/`working`/`peripheral`) orthogonal to category, driving decay and promotion.

### Write Path (the strongest part)
Pipeline in `smart-extractor.ts` (1801 lines) + `admission-control.ts`:
1. **Noise pre-filter** (`noise-filter.ts` regex + `noise-prototypes.ts` embedding bank): drops agent denials ("I don't recall…"), memory meta-questions, greetings, envelope/metadata headers, and diagnostic artifacts. The **NoisePrototypeBank self-grows**: when LLM extraction returns *zero* candidates for a text, that text's embedding is added to the learned noise bank (threshold cosine ≥ 0.82), so similar future inputs are pre-filtered before embedding cost.
2. **LLM 6-category extraction** (≤3 memories/turn), regex fallback.
3. **Batch-internal dedup** (`batch-dedup.ts`): O(n²) cosine dedup across candidates in the same turn (threshold 0.85) before spending per-candidate LLM dedup calls.
4. **Admission control** (`admission-control.ts`, default `enabled:false`): weighted gate over five features — `utility` (LLM-scored), `confidence` (ROUGE-like grounding of candidate against the actual conversation spans — a hallucination guard), `novelty` (1 − max cosine vs existing), `recency` gap, and `typePrior` (profile 0.95 … events 0.45). Reject/admit thresholds; rejected candidates are audited to disk (`amac-v1` records).
5. **Two-stage dedup → typed decision** (`smart-extractor.ts:875-955`): vector pre-filter (cosine ≥ 0.7) then an LLM decision among **create / merge / skip / support / contextualize / contradict / supersede**. `supersede` invalidates the prior memory (sets `invalidated_at`, `superseded_by`, writes a `supersedes` relationship) for temporal-versioned categories; `contradict` either supersedes (if temporal + "general" context) or records a contradiction relationship; `support`/`contextualize` attach context labels and increment support stats. Temporal type and `valid_until` are inferred (`temporal-classifier.ts`, rule-based keyword→expiry).

This is **real-time, write-time typed-relationship construction** — the exact thing the Somnigraph context names as a gap (Somnigraph builds typed edges only during NREM sleep).

### Retrieval
`retriever.ts` (2018 lines). `adaptive-retrieval.ts` first decides whether to search at all (skips greetings/commands/affirmations via regex; force-retrieves on "remember/last time/my name" patterns, incl. CJK) to save embedding calls and noise injection.
- **Channels**: LanceDB vector (cosine) + BM25 FTS, plus a lexical fallback when FTS is unavailable. A tag-prefix path (`proj:AIF`) uses BM25-only + mustContain.
- **Fusion** (`fuseResults`, line 1301): despite the file header comment saying "RRF fusion," the implementation is **weighted linear score blending** — `vectorScore*vectorWeight + bm25Score*bm25Weight`, with a BM25 high-score floor (≥0.75 ⇒ keep at 0.92× even if vector-cold) to preserve exact keyword hits (API keys, ticket numbers). No reciprocal-rank term. The evidence file correctly flags "not standard RRF."
- **Rerank**: external cross-encoder API (Jina v3 default; Cohere/Voyage/Pinecone/DashScope supported) with cosine fallback on timeout/no-key. No learned reranker of their own.
- **Post-processing**: MMR-inspired diversity filter, length-density normalization, decay search-boost, optional BM25 neighbor enrichment. **No graph traversal at retrieval time** — the "graph" is the write-time relationship metadata, not a retrieval expansion (no PPR/adjacency walk).
- Lightweight Chinese BM25 query expansion via a hand-built synonym map (`query-expander.ts`).

### Consolidation / Processing — "Dreaming" sidecar (rule-based, not LLM)
`dreaming-engine.ts` (935 lines), cron-scheduled (default `0 3 * * *`), three phases:
- **light**: cosine dedup-archive within category (threshold configurable) + tier transitions from the decay engine.
- **deep**: promote frequently-recalled (`access_count ≥ minRecallCount`, unique-query count ≥ threshold, decay score ≥ minScore) memories to `core`/`durable`.
- **rem**: **term-frequency pattern counting** (`buildPatterns`) — tokenizes L0 abstracts, counts category/memory_category/term occurrences, writes the top-6 as a "reflection" memory. Despite a `model` config field, none of the three phases calls an LLM. This is statistical, not the LLM-mediated gap-analysis / question-generation Somnigraph's REM does.
- Separately, `memory-compactor.ts` does progressive summarization of semantically similar old clusters (cosine 0.88, minClusterSize 2, cooldown 24h).

### Lifecycle Management
`decay-engine.ts`: **Weibull stretched-exponential** recency — `recency = exp(-λ · daysSince^β)` with tier-specific β (core 0.8 sub-exponential / working 1.0 / peripheral 1.3 super-exponential), importance-modulated half-life (`HL·exp(1.5·importance)`), and dynamic-type memories decaying 3× faster. Composite = 0.4·recency + 0.3·frequency(log-saturation) + 0.3·intrinsic(importance×confidence). Per-tier decay floors (0.9/0.7/0.5). Reinforcement half-life extension on access (`access-tracker.ts`), `bad_recall_count` suppression with 24h decay, explicit forget/delete-bulk.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Hybrid vector+BM25 "RRF" retrieval | `retriever.ts:1301` | **Questionable labeling** — code is weighted-linear fusion, not RRF; header comment is misleading. Works, but don't cite it as RRF. |
| Cross-encoder reranking | `retriever.ts:1384` Jina/Cohere/Voyage APIs | Real, but external API dependency; no learned/feedback reranker. |
| Weibull tier-specific decay | `decay-engine.ts` | Real and clean; the most novel single mechanism. |
| Write-time typed dedup (support/contradict/supersede) | `smart-extractor.ts:875-955` | Real and substantial. **Evidence file wrongly marks contradiction ❌** (see cross-check). |
| Admission-control write gate | `admission-control.ts` | Real but **default-disabled** in all three presets; typePrior weight (0.6) dominates. |
| Self-growing NoisePrototypeBank | `noise-prototypes.ts` | Real, cheap, genuinely clever feedback loop. |
| "Dreaming" reflection engine | `dreaming-engine.ts` | Real but **rule-based/statistical**, not LLM reasoning; REM = term counting. |
| Benchmark performance | `package.json` `bench:locomo`/`bench:longmemeval` | **No published scores** — benchmark dir 404s. Zero numbers comparable to Somnigraph's 85.1 LoCoMo QA. |

---

## Relevance to Somnigraph

### What memory-lancedb-pro does that Somnigraph doesn't
- **Write-time typed relationships / conflict handling** (`smart-extractor.ts`). Somnigraph builds supports/contradicts/evolves/supersede edges only during NREM sleep (`scripts/sleep_nrem.py`); this system classifies and acts on the new-vs-existing relationship *at write time*, invalidating superseded facts immediately. This is the "real-time graph construction" gap named in the Somnigraph brief.
- **Write-path quality gating**: the grounding-based `confidence` score (`admission-control.ts:scoreConfidenceSupport`) checks that an extracted memory is actually supported by the conversation before persisting — a hallucination guard Somnigraph's `tools.py` remember path has nothing analogous to.
- **Self-improving noise filter**: NoisePrototypeBank learns from extraction failures. Somnigraph has no write-time noise gate.
- **Per-category dedup policy** (append-only vs merge vs temporal-versioned) enforced at write time.

### What Somnigraph does better
- **Learned reranker with a measured feedback loop**: Somnigraph's 26-feature LightGBM reranker (`reranker.py`, NDCG 0.7958, Spearman r=0.70 with GT) vs an external cross-encoder API call. memory-lancedb-pro has *no* explicit per-query utility rating loop — only `bad_recall_count` suppression.
- **Graph-conditioned retrieval**: Somnigraph's PPR expansion + betweenness reranker feature (`scoring.py`) genuinely uses the graph at retrieval time; here the "graph" is write-time metadata never traversed during search.
- **LLM-mediated consolidation**: Somnigraph's NREM/REM sleep reasons (pairwise classification, gap analysis, question generation); the dreaming engine's REM is term-frequency counting.
- **Validated benchmarks**: Somnigraph has 85.1 LoCoMo QA; this repo publishes none.

---

## Worth Stealing (ranked)

### 1. Write-time relationship classification (support / contextualize / contradict / supersede) (Medium–High)
**What**: On `remember`, run a cheap vector pre-filter for the nearest existing active memory, then classify the relationship and act — supersede invalidates the prior fact, contradict records a conflict, support increments corroboration. `smart-extractor.ts:875-955`.
**Why**: Directly attacks Somnigraph's named "no real-time graph construction" gap and the Phase 18 finding that write-path quality (not retrieval) is what LME/LoCoMo leaders win on. Immediate invalidation of superseded prefs/facts prevents stale memories from surfacing before the next sleep pass.
**How**: Add a pre-write classify step in `tools.py` remember: vector-search top-1 active neighbor, and for a bounded set of categories call the sleep classifier logic (already exists in `sleep_nrem.py`) inline. Reuse the existing typed-edge machinery; the only new piece is the write-time trigger + immediate `valid_until` set on the superseded row.

### 2. Grounding-based confidence gate on the write path (Low–Medium)
**What**: A non-LLM `confidence` signal (`scoreConfidenceSupport`): ROUGE-like F1 of candidate text against conversation spans + token-coverage − unsupported-ratio penalty. Reject or down-weight memories not grounded in what was actually said.
**Why**: Somnigraph has zero write-path quality gating; a hallucinated or over-generalized `remember` persists unchecked. This is cheap (token overlap, no extra LLM call) and catches the common failure of storing invented specifics.
**How**: Optional gate in `tools.py` remember when the caller passes source context; compute overlap against the provided conversation window, attach as a `confidence` field feeding decay's intrinsic term and as a low-priority reject signal. Keep default-off (like their preset) to preserve honest-accounting reversibility.

### 3. Self-growing embedding noise bank from extraction failures (Low)
**What**: `noise-prototypes.ts` — maintain a small bank of "noise" embeddings; anything within cosine 0.82 is skipped pre-embedding. The bank grows when extraction yields zero candidates: that input's embedding is added.
**Why**: A learned, language-agnostic noise filter that improves itself without labels. Somnigraph's auto-capture has only static heuristics.
**How**: Small module beside `embeddings.py`; a JSON-persisted vector bank in `DATA_DIR`. Low risk, isolated.

### 4. Weibull tier-specific decay shape (Low — consider)
**What**: `exp(-λ·t^β)` with β<1 for durable memories (heavy tail: slow early decay then flat) and β>1 for peripheral (fast cliff), vs Somnigraph's pure exponential (β=1).
**Why**: More expressive than a single half-life per category; a stretched-exponential tail better matches "important-but-idle" memories.
**How**: One-line change in the decay formula to add a per-category β exponent on the day-delta. But Somnigraph already achieves much of this with floors + reheat, so the marginal gain is modest — note-only unless decay tuning resurfaces.

---

## Not Useful For Us

### LanceDB backend, multi-scope/agent isolation, OpenClaw plugin wiring, 12-language READMEs
Packaging for a multi-agent OpenClaw product. Somnigraph is single-user SQLite/MCP; none of this transfers.

### External cross-encoder rerank API + Chinese query expansion
Somnigraph's learned feedback-driven reranker is strictly stronger than an API cross-encoder; the hand-built CJK synonym map is domain-specific and irrelevant.

### The "dreaming" engine as designed
Rule-based dedup + tier-promotion + term counting is a weaker subset of what Somnigraph's LLM-mediated sleep already does. Nothing to import.

---

## Connections

- **Convergent L0/L1/L2 pyramid** with Somnigraph's detail/summary/gestalt and TencentDB's layered memory — independent arrival at multi-resolution memory units.
- **Write-path-quality thesis**: strongly corroborates the Phase 18 sweep conclusion (ByteRover, MemPalace, agentmemory) that leaders win on write-time grounding/discipline, not retrieval cleverness. This system's admission control + typed dedup is the most *built-out* write path in the corpus so far, even if default-off.
- **Contrast with Somnigraph's sleep-time edges**: where `sleep_nrem.py` defers relationship detection to an offline pass, this repo does it synchronously — the same taxonomy (supports/contradicts/supersede), different timing. Worth a design note in `docs/architecture.md` on write-time vs sleep-time relationship construction trade-offs.

---

## Summary Assessment

The core contribution is a **disciplined write path**: a self-growing noise gate, batch dedup, an (optional) grounding-based admission gate, and — most valuably — **synchronous, category-aware typed-relationship dedup** that supersedes/contradicts/supports existing memories at store time. That last mechanism is the single most relevant thing here for Somnigraph, because it targets a gap the project explicitly acknowledges (real-time graph construction) and reinforces the strongest recent finding in the corpus (write-path quality beats retrieval tricks).

What's overhyped: the retrieval side. The "RRF fusion" is weighted-linear blending mislabeled in a code comment; reranking is an outsourced API call; the "graph" is never traversed at query time; and the "dreaming" engine is statistics with sleep-stage names, not reasoning. There are **no published benchmarks** — the `bench:locomo`/`bench:longmemeval` scripts exist but the benchmark directory 404s, so nothing here is comparable to Somnigraph's 85.1 LoCoMo QA. The ~4.4k stars reflect OpenClaw ecosystem adoption, not evaluated quality.

Net: **DIVE** — not for its retrieval, but because three write-path mechanisms (write-time typed dedup, grounding confidence gate, self-growing noise bank) are concrete, isolated, and map onto named Somnigraph gaps. Take the write path; ignore the rest.

**Evidence cross-check (sharpest correction)**: The carsteneu evidence file (audited v1.0.26 / v1.1.0-beta.11, 2026-05-28) marks **contradiction ❌ "No contradiction detection"** and describes supersede as merely `memory_update` re-embedding. The current `main` code contradicts this: `smart-extractor.ts:875-955` implements a full seven-way context-aware dedup including `contradict` (records conflict relationship or supersedes) and `supersede` (invalidates prior fact with `superseded_by`/`valid_until`). The evidence audited an older release; the write-time conflict handling is real and is in fact this system's most valuable feature. The evidence's other flags hold up: "not standard RRF" (correct), "no published benchmarks" (correct), admission control present (correct — but it omits that it is default-disabled).
