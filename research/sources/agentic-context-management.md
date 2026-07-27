# Agentic Context Management (Maximem Synap) — Vendor paper proposing a five-primitive context lifecycle, with SOTA-looking benchmark numbers that do not survive audit

*Generated 2026-07-27 by Opus 5 agent reading the paper and three linked repos*

---

## Paper Overview

**Paper**: Dadhich, G. (2026). *Agentic Context Management: Solving Agent Memory and Cost by Treating Them as Lifecycle and Architecture Problems.* arXiv:2607.21503v1, submitted 23 July 2026, cs.AI / cs.IR. 23 pages, 6 figures, 4 tables, 4 appendices.
**Authors**: Gaurav Dadhich, sole author. Affiliation given only in a title-page footnote: "Maximem. Correspondence to: gaurav@maximem.ai" (p.1). Acknowledgments (p.19) credit Shreyansh Singh Gautam and Anish Yadav for the implementation — single-author paper, team-built product.
**Code**:
- `github.com/maximem-ai/memory_and_context_eval_harness` — the eval harness. Genuinely open (116 files), but the Synap adapter requires `SYNAP_API_KEY` and calls a hosted API.
- `github.com/maximem-ai/eval_benchmark_runs_output` — "reproducible benchmark results." **Contains five files: README.md, RESULTS.md, METHODOLOGY.md, CITATION.cff, LICENSE. Zero run artifacts.**
- `github.com/maximem-ai/maximem_synap_sdk` — 496 files, ~49k LOC Python, but the core SDK (`packages/sdks/maximem-synap/`) is a **thin HTTP/gRPC client**. Every memory operation is a POST to `/api/v1/memories/*` or `/v1/conversations/*` (`memories/interface.py:130,201,268,358,435,541`; `orchestration/response_handler.py:140-251`). The bulk of the repo is 15+ framework integrations. Extraction, entity resolution, retrieval, architecting and compaction all run server-side and are **unverifiable**. The paper says so plainly (p.9: "the mechanism internals are proprietary. This single statement covers the whole section").

**Problem addressed**: Production agents fail less from weak reasoning than from undisciplined context: they re-send growing histories at quadratic token cost, lose facts across sessions, and leak one user's context into another's. The paper argues that "memory" — framed as a store with a write and a read — names only two moments of a five-moment lifecycle, and that a store makes none of the other decisions.

**Core approach**: Reframe the problem as *Agentic Context Management*: deciding what an agent holds in context, when, for how long, and at what cost, across a scope hierarchy (user → customer → client). Decompose into five primitives — architecting, ingesting, scoping, anticipating, compacting & consolidation — that are *coupled*, since the architecture chosen by the first governs what the other four do. Maximem Synap is offered as a reference implementation: a multi-tenant hosted service with polyglot storage (vector/graph/relational/object/time-series/cache, Figure 5 p.10), asynchronous non-blocking ingestion, scope-aware retrieval, predictive prefetch, and compaction that emits an explicit validation score.

**Key claims**:
- 92.0% on LongMemEval (460/500), 93.2% on LoCoMo cat 1–4 — **unvalidated; see audit below. The evidence is a markdown table.**
- Naive full-append context costs O(n²) tokens — *analytic, correct, and trivial* (Appendix A p.21).
- Crude summarization buys linear cost at an "accuracy cliff" — *borrowed, not measured*: the single data point (18,282→122 tokens, 66.7%→57.1%) is cited from Zhang et al. 2025 (ACE, arXiv:2510.04618), not run by the authors.
- Keyword and vector retrieval win in different regimes (MRR@10 across 5 corpora) — *measured, honestly caveated* (Appendix B, seven stated limitations).
- Anticipatory prefetch achieves "60%+ hit-rate consistently across clients" (p.11) — *bare assertion, no data*.
- Compaction is loss-validated with auto-retry (p.11) — *mechanism explicitly withheld*.

---

## The Five Primitives

**Architecting** (§2 p.2-3, §4 p.9). At agent-connection time an LLM generates a bespoke memory architecture from the customer's description of the agent's purpose — which categories to capture, how to extract, store, retrieve, retain and compact them — "not selection from a fixed menu." This is the paper's most genuinely novel claim and the one it uniquely claims in Table 4 (p.17). **Not in the open code.** The SDK exposes fixed enums (`models/enums.py`: `ContextType` = FACTS/PREFERENCES/EPISODES/EMOTIONS/TEMPORAL/ALL), which is a fixed schema from the client's side; any per-agent generation happens server-side and is unobservable.

**Ingesting** (§4 p.9-11). Async, queue-backed, multi-stage: returns an ingestion ID immediately, then qualifies content, extracts categories + entities + relationships + temporal validity, resolves entities against canonical identities, and persists across relational/graph/vector stores. Entity resolution runs a "confidence-ordered cascade" from exact identifiers through softer lexical/semantic/contextual signals, is best-effort, and **never blocks ingestion**. The motivating failure (Table 1 p.5) is a real audit: 10,134 entries stored over 32 days by a popular memory library, 38 usable — a 99.6% junk rate. Footnote g (p.18) corrects their own blog's arithmetic ("the source post states 97.8%") — a small, creditable act of honesty. **Not in the open code**; the client just POSTs a document.

**Scoping** (§2 p.3, §4 p.11). Retrieval resolves a hierarchy narrowest-first (user → customer → client) under strict isolation, with a separate global knowledge layer for entity canonicalization only (Figure 1 p.4). Isolation is enforced at both the storage layer (per-tenant namespaces) and the query layer (scope predicates), with identity derived from the stored credential rather than asserted by the client (p.9). **Partially visible**: `models/enums.py` defines `ContextScope` = CLIENT/CUSTOMER/USER/CONVERSATION, and `synap_integrations_common/scope.py` exists. Enforcement is server-side and unverifiable.

**Anticipating** (§4 p.11). Speculative prefetch: predict and prepare context before the request arrives, moving retrieval off the agent's critical path. Explicitly *not* a query-result cache — it predicts from evolving behavior rather than replaying repeated queries. The prediction mechanism is "proprietary." **This is the one primitive with a real open implementation of its client half.** `cache/anticipation_cache.py` is an in-memory TTL cache for bundles pushed over gRPC, gated by BM25 score plus a novel-term fraction (`_DEFAULT_BM25_THRESHOLD = 1.5`, `_DEFAULT_NOVEL_TERM_THRESHOLD = 0.45`). Its docstrings are unusually candid about failure modes: an "echo-poisoning plan" (WS3) where a bundle's own `ttl_hint` must only ever *shorten* its lease, and a recall-bypass (WS4) noting that "**stem coverage can't gate this (measured false hits at coverage=1.00)**." Both default OFF.

**Compacting & Consolidation** (§2 p.3, §4 p.11, §5 p.13). Compaction as a *verified* operation: test whether key information from the original remains recoverable from the compacted result, emit a validation score and compression ratio, and auto-retry with less aggressive compression when validation fails. Cost is analyzed as `N·W·(1+c/p)` — the linear cost raised by a fixed factor. **Validation mechanism explicitly not described** (p.11). `CompactionLevel` has seven values (LOW/MEDIUM/HIGH/ADAPTIVE/AGGRESSIVE/BALANCED/CONSERVATIVE) — three of which are documented as backward-compat aliases of the first three, i.e. the enum is redundant. **Not exercised by either benchmark**: both are single-turn QA over retrieved memories.

**Verdict on the taxonomy**: three of five primitives (architecting, ingesting, compacting-validation) are defined but absent from all open code. Scoping is a set of enums. Only anticipating has a readable implementation, and only its client half. On novelty, Table 4's self-assessment is too generous: this corpus already holds write-time query anticipation (Kumiho, EverMemOS), loss-validated consolidation (ai-memory), ingest-time entity resolution (Mem0g), multi-tenant scoping (six systems), and emergent per-domain schema (Acontext). What is new is the *assembly* and the framing, not the parts.

---

## Architecture

### Storage & Schema
Polyglot, per-tenant-namespaced (Figure 5 p.10): vector (ChromaDB), graph (Neo4j), relational (Postgres, source of truth), object (S3-compatible), time-series (telemetry), Redis (queues/cache). Embeddings are "locally hosted sentence-transformer models by default" (p.12). Client-visible schema: `memory`, `event_date`, `valid_until`, `temporal_category` (incl. `perpetual`), `context_type`, `source_evidence`, `similarity_score` (from the harness's context formatter, `scripts/iterate_prompt.py:46-107`).

### Memory Types
Five: facts, preferences, episodes, emotions, temporal — plus ALL. Comparable to Somnigraph's episodic/semantic/procedural/reflection/meta but cut differently: Synap has no procedural or reflection type and adds *emotions*. Which types a given agent captures is nominally decided by the architecting step.

### Write Path
Non-blocking by design: `memories.create()` returns an ingestion ID, work happens in background, `memories.status()` polls. The paper defends the read-your-writes gap by noting recent turns are carried verbatim in working context (p.13). Entity resolution and temporal-validity assignment sit on this path — the two capabilities Somnigraph most conspicuously lacks. **All server-side; unverifiable.**

### Retrieval
Scope-aware pipeline, narrowest-first, returning provenance-tagged, token-budgeted, ranked items, in a `fast` or `accurate` mode (the latter adding query decomposition and LLM reranking). Beyond vector similarity, "vector-guided multi-hop traversal of a knowledge graph" — semantic similarity picks the entry point, graph relations are then followed (p.11). Prior art is credited honestly (GraphRAG, HippoRAG, ReMindRAG); the specific traversal and scoring are not described. **Note: both committed benchmark configs run `retrieval_mode: "fast"` with `retrieval_max_results: 100` and adapter `top_k: 50`** — i.e. the headline numbers were produced *without* the accurate mode's LLM reranking, handing up to 100 memories to the reader.

### Consolidation / Processing
No offline consolidation cycle. Consolidation is folded into compaction and runs periodically over the already-compacted context plus recent turns (p.13), not over the durable store. There is nothing analogous to Somnigraph's NREM edge detection or REM summarization. Table 4 (p.17) correctly identifies Letta's "sleep-time agents" as the closest published kin.

### Lifecycle Management
Temporal validity (`valid_from`/`valid_until`, `perpetual`) is first-class, and retention is governed by the generated architecture's guardrails (Appendix D p.23). **No decay model, no dormancy, no reheat-on-access.** Supersession is handled by temporal validity rather than by scoring — the opposite of Somnigraph's exponential-decay-plus-reheat approach.

---

## Benchmark Audit

**What is committed.** `eval_benchmark_runs_output` has exactly one commit (`6d97542`, "Set both models to gpt-5-mini; enrich Synap positioning from website/SDK") and **no git tags** — although paper Table 2 (p.14) states "Run date / repo tag: Tagged in the results repository." There are no per-question answers, no retrieved contexts, no judge verdicts, no seeds, no config snapshots, no harness commit SHA. In the harness, `data/`, `results/` and `checkpoints/` are all gitignored. **Nothing from any run exists publicly for either benchmark.** §6.3 (p.16) admits artifacts are "available on request rather than published" — but **Appendix C (p.23) contradicts this**, claiming the paper states for every result "the harness commit SHA" and "what we release publicly (harness code, configs, and raw per-question outputs)." Neither the SHA nor the outputs exist.

**Judge and reader.** `gpt-5-mini` for both. A model judging its own family's output, single run, no variance reported — while their own METHODOLOGY.md scores Zep's "10 runs, mean ± std" as better practice.

**The judge prompt is extraordinarily lenient**, well beyond the "be generous" prompt that locomo-audit measured accepting **62.81% of intentionally vague wrong answers**. `prompts/judge.md` has eleven leniency rules, including: 1-of-N partial credit for any list gold (rule 1); dates within **14 days** and durations within **50%** count as correct (rule 4 — which explicitly blesses "5 months matches six months" and "19 days matches two weeks", *precisely* the temporal GT corruptions locomo-audit catalogued in locomo_1_qa31 and locomo_6_qa31); relative-date windows plus ±7 days on each edge (rule 6); and rule 7, "**Do not penalize the model for picking a different valid answer than the gold**." Rule 10 grants credit for merely naming the same entity. The stated failure condition is that the answer contains "ZERO correct items from the gold AND no plausible interpretation matches."

**The answer prompt is written to farm that judge, and says so.** `prompts/locomo_agent.md:63`: "Do not omit candidate items because they appear in lower-ranked memories or because another item seems 'more central' — **the grader credits any matching item**." Step 6a instructs enumeration of all plausible answers; Step 7 forbids abstention outright. locomo-audit measured answer length correlating with accuracy at Spearman ρ = 0.64 and the prompt alone accounting for **10.67 points**.

**Both prompts were tuned against the score.** `scripts/iterate_prompt.py` replays cached Synap retrieval and sweeps candidate answer prompts (the usage example names `prompts/locomo_agent_v7.md`) *and* candidate judge prompts (`--judge-prompt`), printing Improved/Regressed/Unchanged counts against a baseline run. The headline numbers therefore measure a co-adapted prompt-and-judge pair. **The paper's Table 2 lists dataset, answer model, judge, retrieval config, harness, artifacts and run date — but not the answer prompt or the judge prompt**, despite their own METHODOLOGY.md naming prompt as one of four variables that "swing the same system 10 to 30 points."

**Ingestion feeds the model the annotator metadata behind the corrupted gold answers.** `datasets/locomo/loader.py:89-101` concatenates both `blip_caption` and the image `query` field into ingested turn text. locomo-audit identified that `query` field — stock-photo search strings like "Ferrari 488 GTB" — as never part of any dialog, yet the source of **33 hallucinated golden answers**. Injecting it makes those questions answerable. Somnigraph's loader does not do this (benchmarks.md: "raw dialog turns as `[Speaker] text`, no enrichment"; grep confirms no `blip`/`query` handling in `scripts/locomo_bench/`).

**The LongMemEval variant is very likely oracle, not `_S`.** `scripts/download_datasets.py:93-95` sets `--variant` **`default="oracle"`**, so the documented reproduction path ("Download the official datasets") fetches `longmemeval_oracle.json`. `scripts/prepare_longmemeval_interleaved.py:23` builds `longmemeval_interleaved.json` **from that oracle file**, and `datasets/longmemeval/loader.py:18-22` lists `longmemeval_interleaved.json` **first** in its auto-detect preference order, ahead of `longmemeval_s_cleaned.json`. The oracle variant retains only evidence sessions — the distractor haystack is removed, so retrieval is near-trivial. This predicts exactly the reported pattern: **100.0% on four categories** (single-session-user 70/70, single-session-preference 30/30, knowledge-update 78/78, temporal-reasoning 133/133) with residual error concentrated in multi-session (75.2%), the only category still hard once every session is an evidence session. I cannot *prove* the reported run used oracle — no artifacts are published — but every default in the harness points there while all four public documents claim `LongMemEval_S`.

**Competitor numbers were copied, not re-run.** The paper is explicit and honest about this (Table 3, p.16: "each vendor's own best self-reported figure... the rows are not comparable to one another"). The **README is not**: it renders a mermaid bar chart plotting Synap 92.0 against Supermemory 85.2 and Zep 71.2, and claims Synap is "ahead of every published competitor result." The README does precisely what the paper disclaims.

**The comparison set is also incomplete in a self-serving direction.** Table 3 and the README list only Supermemory, Zep, Mem0 and Letta — the last two as "not published." Systems in this corpus with published numbers at or above Synap's are simply absent: **MemMachine reports LongMemEval-S 93.0%** (config C15, k=100, GPT-5-mini — *above* Synap's 92.0%) **and LoCoMo 0.9169** with gpt-4.1-mini (`memmachine.md`); **EverMemOS reports LoCoMo 92.32%** (`evermemos.md`); MemOS reports LongMemEval 77.8 / LoCoMo 75.80 (`memos.md`). Table 4's related-work survey *does* cover MemOS-adjacent systems, so the omission from Table 3 is not ignorance of the field. "Ahead of every published competitor result" is unsupported as written.

| Claim | Evidence | Assessment |
|-------|----------|------------|
| 92% LongMemEval | A markdown table. No artifacts. Judge = answerer = gpt-5-mini, single run. Harness defaults resolve to the **oracle** variant while docs claim `_S`. Four categories at exactly 100.0%. | **Questionable.** Not merely unvalidated — the most likely reading is that it measures reader accuracy on oracle contexts, which is not a memory-system result. |
| 93.2% LoCoMo | A markdown table. Judge prompt grants ±14-day dates, ±50% durations, 1-of-N credit, and credit for "a different valid answer than the gold"; answer prompt explicitly written to exploit it; both prompt families swept for score. Loader injects the annotator `query` metadata behind 33 hallucinated golds. | **Questionable.** Internally consistent arithmetic (per-category × n reproduces 93.21%), but 0.37pp below locomo-audit's **93.57% stock-GT ceiling** — i.e. within ~6 questions of a mathematically perfect system, on a benchmark whose judge accepts 62.81% of vague wrong answers. The score measures judge tolerance. |
| Quadratic cost of naive accumulation | Appendix A p.21: closed-form `t·n(n+1)/2`, plus a sensitivity table. Stated as "illustrative, not measured." | **Validated (trivially).** Correct arithmetic, honestly labelled. Not a contribution — it is a sum of an arithmetic series. |
| Summarization accuracy cliff | One data point (18,282→122 tokens; 66.7%→57.1%), **cited from Zhang et al. 2025**, not measured here. | **Unvalidated as their claim.** The underlying result is real but borrowed; no evidence that *validated* compaction avoids the cliff is offered anywhere. |
| Keyword vs vector regime-dependence | Appendix B: 5 corpora × 10k docs × 1k queries, MRR@10, Tables B1/B2, seven stated limitations. | **Plausible / honest.** The best-evidenced part of the paper. Caveat 2 (no chunking) is self-acknowledged as biasing against vector on long docs. |
| Anticipatory prefetch "60%+ hit-rate" | Bare sentence, p.11. | **Unvalidated.** No data, no definition of a hit. |

### Comparability to Somnigraph's 85.1% LoCoMo

**These two numbers must not appear in the same ranking.** Somnigraph's 85.1% is Opus 4.6-judged on original GT (87.2% with corrected GT vendored from locomo-audit, at Level 5b), GPT-4.1-mini reader, N=1540, raw dialog ingestion. ACM's 93.2% is gpt-5-mini judging gpt-5-mini under a prompt tuned for the score, with image-metadata injection. Somnigraph's own data shows Opus is **3.2pp stricter than GPT-4.1-mini** on identical answers; the gap to a prompt-optimized gpt-5-mini self-judge is unmeasured and larger.

There is a second, independent blocker: **the category labels are swapped.** Somnigraph (`scripts/locomo_bench/config.py:58-61`) maps 1=single-hop, 2=temporal, 3=multi-hop, 4=open-domain. The harness (`datasets/locomo/loader.py:17-23`) maps 1=single-hop, 2=**multi-hop**, 3=**temporal**, 4=open-ended. locomo-audit's prose uses a third mapping again. The bucket *sizes* match almost exactly, which makes the conflict decisive:

| LoCoMo cat id | n (Somnigraph / ACM) | ACM label + score | Somnigraph label + score (Opus, orig GT) | Alignable? |
|---|---|---|---|---|
| 1 | 281 / 282 | single-hop 88.8% | single-hop 80.9% | yes (+7.9pp) |
| 2 | 320 / 321 | **multi-hop 97.3%** | **temporal 84.4%** | no — label conflict |
| 3 | 89 / 96 | **temporal 90.8%** | **multi-hop 61.5%** | no — label conflict |
| 4 | 841 / 841 | open-ended 93.4% | open-domain 89.5% | yes (+3.9pp) |
| all | 1540 | 93.2% | 85.1% | judge-incomparable |

So **ACM's headline "multi-hop 97.3%" is not Somnigraph's multi-hop.** It is the 321-question bucket that Somnigraph's pipeline calls temporal; Somnigraph's multi-hop is the 89-question bucket. The apparent 97.3 vs 61.5 chasm is substantially a labeling artifact. On the two buckets that *do* align, the gaps are +7.9pp and +3.9pp — still favouring ACM, still confounded by judge and prompt, but an order of magnitude less dramatic than the headline implies.

**Could a third party reproduce this?** No. `SYNAP_API_KEY` is mandatory (`adapters/synap_adapter.py`), so every run bills a commercial vendor and no independent party can run the system at all. Even with a key, the exact answer/judge prompt versions used are not identified, no harness SHA is given, and no baseline artifacts exist to diff against. **To settle it would require Maximem to publish the per-question outputs** — which §6.3 says exist.

### Recommended settling runs (not executed — all require a paid vendor key and/or LLM judging)

Two of the four findings above are *decidable without a Synap key*, because they concern the harness rather than the service:

```bash
# 1. Which LongMemEval variant does the documented repro path actually produce?
#    Expect longmemeval_oracle.json, NOT longmemeval_s_cleaned.json.
python scripts/download_datasets.py --dataset longmemeval   # --variant defaults to "oracle"
ls -la datasets/longmemeval/            # then compare SHA256 against the HF _s_cleaned file

# 2. How much does the annotator image-`query` injection move the score?
#    Patch datasets/locomo/loader.py:89-101 to drop the `query` term (keep blip_caption),
#    then diff the two runs on the 33 hallucination-class question ids in locomo-audit's errors.json.
```

The third and fourth need a key: re-run `configs/orchestrator_synap_gpt5mini.yaml` with `--variant s` forced, and re-score the resulting answers under Somnigraph's Opus judge against locomo-audit's corrected GT. Only that pairing produces a number placeable beside 85.1%. **None of this is worth Somnigraph's budget** — the taxonomy is takeable for free and the benchmark claim is not load-bearing for any decision here.

---

## Relevance to Somnigraph

### What ACM/Synap does that Somnigraph doesn't

- **Ingest-time entity resolution** (paper §4 p.11) — a confidence-ordered cascade mapping "Sarah" / "Sarah Chen" / "SC" to one canonical identity. Somnigraph has none; `src/memory/tools.py` `impl_remember` does a single cosine-similarity dedup pass at ≥0.9 and nothing entity-aware. This is a named gap in CLAUDE.md.
- **A write-path quality gate.** Their Table 1 junk-accumulation row (38 usable of 10,134) is exactly Somnigraph's Priority 2 in concrete form: `remember()` embeds anything below the 0.9 dedup threshold. Their answer — qualify content *before* extraction — is the gate the shadow-mode histogram is being collected to calibrate.
- **Predictive prefetch with a shipped client-side gate.** `cache/anticipation_cache.py` uses a two-condition gate (BM25 ≥ 1.5 **and** novel-term fraction ≤ 0.45) to decide whether a prefetched bundle may serve a query without a fetch. Somnigraph has no prospective indexing (a Tier 1 adopt candidate) and no proactive injection.
- **Compaction as a verified operation** with an emitted validation score, compression ratio, and auto-retry. Somnigraph's REM summarization in `scripts/sleep_rem.py` has no post-hoc information-loss check — and the whole-corpus run measured summaries at **+0.078 NDCG judged** while dormancy cost **−0.136 NDCG**, i.e. exactly the unvalidated-compaction failure the paper names.
- **Multi-tenant scope hierarchy** (user → customer → client) with a separate global knowledge layer. Somnigraph is single-user by construction.
- **Per-agent generated schema** rather than a fixed category taxonomy.

### What Somnigraph does better

- **A learned reranker with measured lift.** `src/memory/reranker.py` runs a 31-feature LightGBM model (V6, +8.42pp NDCG@5k over the hand-tuned formula on its own GT). ACM has no learned ranker anywhere in the open code; its "accurate" mode does LLM reranking, and its benchmark numbers were produced in `fast` mode without it.
- **A closed feedback loop.** `recall_feedback()` utility/durability ratings, EWMA aggregation, UCB exploration, per-query Spearman r=0.70 with GT, Hebbian co-retrieval PMI. Nothing comparable exists in ACM; retrieval quality is asserted, never learned from use.
- **Offline consolidation measured as its own signal.** The sleep-bench program (H3 link-not-merge, H7 summary floor, H10 reranker 2×2, H12 keep-and-downweight) produced CI-bounded, sometimes *negative* results. ACM's consolidation is undescribed and unmeasured.
- **Honest accounting.** Somnigraph publishes refuted pre-registrations and reversed verdicts. ACM's paper is careful in places (Table 3's non-comparability disclaimer, footnote g's self-correction) and its README then plots the chart the paper disclaimed.
- **Decay and lifecycle** — per-category half-lives with floors and reheat-on-access, versus temporal validity alone.

---

## Worth Stealing (ranked)

### 1. The five-primitive taxonomy as a coverage checklist (Low effort)
**What**: Adopt architecting / ingesting / scoping / anticipating / compacting as an axis in `docs/similar-systems.md`, replacing or supplementing the current ad-hoc comparison dimensions. Table 4 (p.17) is already a filled-in systems × primitives matrix for eight systems.
**Why**: It is the paper's real contribution and it survives the benchmark collapse entirely. It also *diagnoses Somnigraph*: we are strong on retrieval and consolidation, absent on architecting and scoping, and have exactly one unbuilt item (anticipating) already sitting in the Tier 1 adopt list. The decomposition is not a rebranding of Mem0/MemOS — "architecting" (generated schema) and "anticipating" (predictive prefetch, distinct from caching) are dimensions the existing corpus comparison does not have a column for.
**How**: A new table in `docs/similar-systems.md`; add the two missing columns to the profiles already there.

### 2. Validated compaction — an information-loss check on REM summaries (Medium effort)
**What**: After `sleep_rem.py` writes a summary, test whether the key facts of its member memories remain recoverable *from the summary alone*; emit a validation score and compression ratio; retry with a gentler compression when it fails.
**Why**: This is the highest-value idea in the paper and it lands on a measured Somnigraph wound. The whole-corpus run found summaries help (+0.078 NDCG judged) while dormancy erodes (−0.136 NDCG); the fix direction already recorded is "downweight-not-status-death." A per-summary validation score is the natural gate: only downweight members whose content is provably recoverable from the summary. It also makes the summary layer auditable rather than hopeful.
**How**: A new step in `scripts/sleep_rem.py` after summary generation — for each member, ask the LLM whether the member's core fact is recoverable from the summary text; store the pass fraction as a `validation_score` column on the summary memory; gate the member-downweight step on it. The H7 coverage-judge rig (`opus` coverage judging, already built for the sleep-bench credit bracket) is directly reusable as the scorer.

### 3. Ingest-time qualification before extraction (Medium effort)
**What**: A cheap qualify step in `impl_remember` that classifies a candidate write as worth-storing before embedding, distinct from the near-duplicate check.
**Why**: Priority 2, with ACM's 99.6%-junk audit as independent corroboration of the salience half. Note the ordering lesson: they qualify *before* extraction, whereas Somnigraph's shadow-mode logging measures *after* the dedup KNN. The shadow histogram gives the dup threshold; it does not give a salience threshold, and this is the gap.
**How**: `src/memory/tools.py` `impl_remember`, after `privacy.py` redaction and before `embed_text()`; log the verdict into the existing `write_shadow` event rather than acting on it, so the gate is measured before it is enforced.

### 4. Two-condition prefetch gate (Low effort, high information)
**What**: Their cache gate is BM25 score **and** novel-term fraction — lexical coverage of the query by the candidate bundle — not a retrieval score.
**Why**: Somnigraph's floor study concluded the RRF-family score has real discrimination (AUC 0.64–0.74) but no usable F-beta operating point. ACM independently reached the same negative result from the other side: their own docstring says "**measured false hits at coverage=1.00**," i.e. coverage alone does not gate either, and both of their gates ship **default OFF**. That is a second instrument failing differently and agreeing — strong evidence the surface/skip decision is not solvable on a single scalar, and a reason not to spend another session on it.
**How**: No build. Record the corroboration in `experiments/floor-study/findings-floor-study.md`.

### Not worth stealing: generated per-agent architecture. Somnigraph has exactly one agent and one user; a generated schema would be complexity with no consumer, and the claim is unverifiable anyway.

---

## Not Useful For Us

**Scope hierarchy and multi-tenant isolation.** Genuinely central to their product and genuinely irrelevant here: user → customer → client isolation solves a B2B data-leakage problem a single-user MCP server does not have. Worth checking rather than dismissing, though — the *conversation* scope in `ContextScope` is a per-thread partition, which is a real question for Somnigraph (should a memory written in one Claude Code session be scoped to that project?). The answer today is "no, everything is global," and that is a deliberate choice, not an oversight.

**The eval harness itself.** Despite being open and reasonably built, it is not adoptable: its LoCoMo loader injects annotator metadata, its LongMemEval default resolves to oracle, and its judge prompt is calibrated for leniency. Somnigraph's `scripts/locomo_bench/` is stricter on all three counts.

**Async non-blocking ingestion.** A latency optimization for a network service. `remember()` is a local SQLite write.

---

## Connections

- **`locomo-audit.md`** — the decisive instrument here. Its 93.57% stock-GT ceiling, 62.81% vague-answer judge acceptance, and 10.67-point prompt effect are what convert ACM's 93.2% from "impressive" to "measuring the judge." Its catalogue of the image-`query` hallucination source is what made the loader finding legible. Note the recursion: the audit found **EverMemOS's scores exceeding the mathematical ceiling** in two categories, and ACM now shows the same signature. Ceiling violation is becoming this benchmark's characteristic artifact, and it always resolves to judge leniency.
- **`locomo.md`, `longmemeval.md`, `locomo-plus.md`** — the methodology baseline. `locomo.md` already warns that "cross-paper comparisons are unreliable" and documents the Zep↔Mem0 dispute ACM itself recounts. `longmemeval.md` is the most damaging: the official judge is **GPT-4o with per-question-type prompts at >97% human agreement**, and prompt choice alone moves oracle-retrieval scores from 0.862 to 0.924 — ACM used a self-family gpt-5-mini judge under a swept prompt. `locomo-plus.md` supplies the field's best judge-calibration data (human–LLM agreement 0.801–0.820; output-side length bias peaking at the 5.18-token GT length), which is exactly the bias ACM's enumerate-everything answer prompt exploits.
- **`kumiho.md` and `evermemos.md` — ACM's "anticipating" is not novel, and the corpus has better-evidenced versions.** Kumiho does *prospective indexing*: at write time the LLM generates hypothetical future scenarios where a memory would be relevant and indexes them alongside the summary, eliminating the >6-month accuracy cliff (**37.5% → 84.4%** on LoCoMo-Plus). EverMemOS does *Foresight*: 4–10 time-bounded, evidence-grounded predictions per MemCell with validity intervals, auto-expired at retrieval. Both are write-time query anticipation; ACM's is request-time prefetch — genuinely a different mechanism, but the paper's Table 4 gives itself the only filled "Anticipating" circle among commercial systems, and the corpus does not support that. Sobering caveat from `evermemos.md`: foresight did **not** help EverMemOS on PERMA preference tracking.
- **Validated compaction is also not unclaimed.** `docs/ideas-considered.md` Tier 2 cites **ai-memory (akitaonrails)**, which runs `score_before` / `score_after` / `passed` on a consolidation merge and rejects it if it fails to improve, gated on a confidence floor and `min_delta` — that is loss-validated compaction with auto-reject, shipped. `mem0-paper.md`'s `InformationContent()` gate is a pre-write cousin (an UPDATE cannot replace a more informative fact); `lightthinker-pp.md` chooses reversibility over validation (`expand(k)` restores archived reasoning; removing it costs 6.5pp); `kumiho.md` has process guards (dry run, 50%-per-run deprecation circuit breaker, audit report) but no loss score. ACM's contribution is the *contract* — emitting a validation score and compression ratio to the caller — not the idea.
- **`docs/ideas-considered.md` § Recurring themes** and **`docs/similar-systems.md` § "Write-path is the lever"** — ACM's "ingesting" primitive and its 99.6%-junk audit are **independent corroboration** of the corpus's top recurring theme, now at least the fifth arrival (Phase 15 AMemGym, Phase 18, the 59-system scan, Acontext, ACM). Evidence *for Somnigraph's existing Priority 2*, not for anything ACM built. **The dissent worth carrying**: `memmachine.md` — currently among the LoCoMo leaders — ablates the other way (retrieval-stage +4.2/+2.0/+1.8/+1.4% vs ingestion-stage +0.8% chunking) and bets explicitly *against* write-time LLM extraction; `longmemeval.md` finds 15–19% of instances have correct retrieval with wrong generation. Neither path alone bounds the score.
- **`memos.md`, `mem0-paper.md`, `supermemory.md`, `memmachine.md`, `virtual-context.md`** — entity resolution and scoping precedents. Mem0g does true ingest-time ER (embed source/dest entities, threshold-match, create only on miss) and marks contradicting relations *invalid rather than deleted*; `virtual-context.md` canonicalizes at the tag level with transitive cross-batch merge. On scoping, **six of fifteen surveyed systems already ship multi-tenancy** (MemMachine `org_id/project_id`, Supermemory Spaces + roles, EverMemOS tenant isolation + HMAC, Kumiho Project/Space, MemOS MemGovernance/ACLs, Acontext LearningSpaces) — making it the least novel of the five primitives, which both `memos.md` and `supermemory.md` classify as product infrastructure rather than research.
- **`acontext.md`** is the closest thing to "architecting" in the corpus: `SKILL.md` mandates only `name` + `description` and leaves all other structure LLM-defined, plus a write-time `applies_when` scoping gate ("do NOT over-generalize"). Emergent rather than generated-up-front, but it means ACM's uniquely-claimed primitive has a nearer neighbour than Table 4 admits.
- **`context-mem.md`** — the methodological mirror. It was faulted in this corpus for ingesting LoCoMo's own `session_summary`/`observation` metadata and for hand-curated benchmark-only synonyms ("test-set leakage by another name"). ACM's loader injecting the annotator image-`query` field is the same class of move, found independently.

---

## Summary Assessment

The taxonomy is a real contribution and the benchmark numbers are not — and those two judgments are almost independent. Factoring context management into architecting, ingesting, scoping, anticipating, and compacting is a useful decomposition that names two dimensions the existing corpus lacks columns for (generated per-agent schema; predictive prefetch as distinct from caching), and it is grounded in a table of observed production failures rather than derived a priori. The cost argument is arithmetic and correct; the retrieval motivating study is small but honestly caveated, with seven stated limitations including the one that undercuts its own headline. Where the paper is careful it is genuinely careful: Table 3 states outright that its competitor rows are not comparable, and a footnote corrects the author's own blog's arithmetic against himself.

The evaluation does not survive contact with the artifacts. `eval_benchmark_runs_output`, presented as "reproducible benchmark results," contains five markdown files, one commit, no tags, and zero per-question data; Appendix C's reproducibility statement claims released per-question outputs and a harness SHA that do not exist, contradicting §6.3 four pages earlier. In the harness, the LoCoMo loader injects the annotator stock-photo `query` strings that locomo-audit identified as the source of 33 hallucinated gold answers; the judge prompt grants ±14-day dates, ±50% durations, 1-of-N list credit and explicit credit for "a different valid answer than the gold"; the answer prompt instructs the model to enumerate candidates because "the grader credits any matching item"; and `iterate_prompt.py` exists to sweep both prompt families against the score. The LongMemEval story is worse: the dataset downloader defaults to `--variant oracle`, the interleave script builds from the oracle file, and the loader prefers that file over `_S` — which would make four categories at exactly 100.0% unsurprising and would mean the number measures a reader on evidence-only contexts. The paper's own methodology document lists prompt as a variable worth 10–30 points and then omits it from the configuration table. The 93.2% sits 0.37pp under the mathematically possible ceiling on stock GT.

For Somnigraph, take three things and leave the numbers. First, the taxonomy, as a coverage checklist in `similar-systems.md` — it diagnoses our gaps cleanly. Second, validated compaction: an information-loss check on REM summaries, gating the member-downweight step, is the single most actionable idea here and it lands directly on the measured dormancy erosion. Third, the corroborations — ACM's ingesting primitive is another independent arrival at write-path quality as the field's lever, and their prefetch gate's shipped-default-OFF plus "false hits at coverage=1.00" docstring is a second instrument agreeing with our floor study's null. What to record loudest, though, is the comparability finding: ACM's "multi-hop 97.3%" is scored on the 321-question bucket that our pipeline labels *temporal*, while our multi-hop bucket holds 89 questions. Anyone placing 93.2% next to our 85.1% would be comparing a lenient self-judge on possibly-oracle data against an Opus judge on corrected GT, with the category labels crossed. That sentence should not be written.
