# VIR — Obsidian-native "LLM Wiki": retroactive Claude Code transcript distillation with confidence-gated write path

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

VIR (Serbian for "whirlpool") is a TypeScript/Node CLI (`@djolex999/vir-cli`, v0.8.0, MIT) that implements Karpathy's "LLM Wiki" pattern. It reads Claude Code session transcripts (`~/.claude/projects/**/*.jsonl`) retroactively, distills durable knowledge into typed Markdown notes in an Obsidian vault, and feeds the top notes back into `CLAUDE.md`. It is a **write-path / knowledge-curation tool**, not a retrieval engine. Runs as a scheduled daemon (launchd/systemd/cron, every 3h) and exposes an MCP server (5 read tools).

### Storage & Schema
- **Markdown vault is the primary store** — notes are plain `.md` files with YAML frontmatter (Dataview-compatible), organized into `patterns/`, `gotchas/`, `decisions/`, `tools/`, `articles/`, `projects/`, `archived/`.
- **SQLite** (`~/.vir/vir.db`) is a *sidecar*, not the source of truth: content hashes (SHA-256 for idempotent reruns), embeddings, and distilled-row metadata (`DistilledRow`: topic, project, category, confidence, startedAt, content). The markdown files are canonical; the DB is a cache/index.
- Note schema fields: `category` (4 fixed values), `confidence: 0.0–1.0`, `verified: true|false`, `themes[]`, project, topic, date, source URL (for articles).

### Memory Types
Fixed 4-category taxonomy for sessions: **pattern / gotcha / decision / tool** (`distiller.ts:12`). Web articles get a parallel 4-category taxonomy: concept / technique / reference / opinion. No entities, no keywords, no emotional valence, no layered detail/summary/gestalt — the taxonomy is deliberately small and hand-fixed.

### Write Path *(the substantive part of the system)*
A multi-stage cost-and-quality funnel (`pipeline/run.ts`, `filter.ts`, `distiller.ts`):
1. **Heuristic pre-filter** (`filter.ts:11 scoreSession`): pure-code, no LLM. Additive score from structural signals — `lineCount>50 (+0.3)`, `toolCalls>5 (+0.3)`, `filesTouched>2 (+0.2)`, and a regex for signal words `(error|fixed|bug|learned|gotcha|workaround|root cause) (+0.2)`. Sessions scoring below `filterThreshold` (default 0.4) never reach an LLM.
2. **Tool-call filtering** (`toolCallFilter.ts`): preserves intent (file paths, commands, errors, short results) while truncating large tool outputs to bound token cost. Configurable `aggressive|moderate|off`.
3. **Classify** (Haiku): assigns category + topic + themes + `confidence`. Anything with `confidence <= 0.6` is **dropped before the expensive distill call** (`distiller.ts:445`). This is a genuine write-path quality gate.
4. **Hybrid model routing** (`selectDistillModel`, `distiller.ts:91`): route `decision`-category or large (>100k input token) sessions to Sonnet; everything else to Haiku. Cost optimization based on calibration that Haiku only loses on decision-heavy/large sessions.
5. **Distill** (Sonnet/Haiku): produces the markdown note (Summary / What Was Learned / Context / Related).
6. **Active learning** (`vir review`): human approve/edit/reject; approved → `verified: true`, rejected → `.rejected/` (recoverable, not deleted).

### Retrieval (`search/retriever.ts`)
Deliberately simple, and the evidence file describes it accurately:
- **Semantic**: optional Ollama `nomic-embed-text` embeddings, cosine similarity, `MIN_EMBEDDING_SCORE = 0.3` floor. Sessions, articles, topics, PDFs share one vector space with no per-layer boost.
- **Lexical**: hand-rolled **TF-IDF** (`searchTfIdf:244`) — length-normalized TF × `log(N/df)`. Not BM25 (no saturation/length-norm params).
- **"Hybrid" is a sequential fallback, not fusion** (`search:68`): if Ollama is up and returns ≥1 hit above floor, use embeddings; else fall through to TF-IDF. There is **no score fusion, no RRF, no learned reranker**.
- **MMR diversity rerank** (`mmrRerank:161`) on the embedding path only: greedy `lambda*relevance - (1-lambda)*maxSimToSelected`, `lambda = 1 - retrievalDiversity` (default diversity 0.3). TF-IDF path is score-only.
- **Verified boost** (`VERIFIED_BOOST = 0.2`): human-approved notes get a flat +0.2 added to their score before the topK slice, in both paths. Crude (additive on a 0.3–1.0 cosine range) but functional — a lightweight human-feedback signal.

### Consolidation / Processing
No sleep-like consolidation cycle. Two on-demand LLM operations approximate it:
- **`vir dedupe`** (`dedupe/detector.ts`): cheap candidate generation (Jaccard on topic-slug tokens + significant-word overlap in first 100 chars, `scoreCandidate:67`), top-30 pairs sent to Haiku for duplicate judgment with `keepWhich: A|B|merge`, threshold `MIN_DUP_CONFIDENCE = 0.7`. Interactive/human-in-loop; merged notes go to `archived/`, never deleted.
- **`vir lint`** (`lint/linter.ts`): orphan/staleness checks (free) + Haiku-based contradiction detection (cheap).
- **`vir compose`/`vir summarize`**: LLM synthesis of a topic page or cross-session project summary from embedding-retrieved sources (wikilinked for Obsidian backlinks).

### Lifecycle Management
- **No automatic decay.** Staleness is surfaced by `vir lint --stale` for human action, never applied automatically.
- Idempotency via SHA-256 content hashing (unchanged sessions skip reprocessing).
- Dedup/reject move files to `archived/` or `.rejected/` — recoverable, non-destructive.
- **CLAUDE.md sync** (`claude/updater.ts`): top-5-per-category by confidence (`TOP_N_PER_CATEGORY = 5`) rendered into a `<!-- VIR:START -->…<!-- VIR:END -->` block, with a diff (`added/removed/upgraded/unchanged`, confidence-delta > 0.05 = "upgraded") and human confirmation before write. This is the procedural-memory feedback loop.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Confidence-gated distillation keeps signal high | Code confirms: heuristic pre-filter (`filter.ts`) + `confidence <= 0.6` drop before distill (`distiller.ts:445`) | **Validated** as implemented; efficacy unmeasured (no benchmark, no held-out labels) |
| "Avg confidence 0.91, 121/126 high-signal" across 226 sessions | Author's single self-run in README | Anecdotal; confidence is the model's self-report, not external validation |
| Verified notes rank above unverified | `VERIFIED_BOOST = 0.2` additive (`retriever.ts:57,124,277`) | **Validated** but crude (flat additive, no learning) |
| MMR gives diverse results | `mmrRerank` implemented and wired (embedding path only) | **Validated** as code; standard MMR, no eval |
| Hybrid routing cuts cost ~60-70% | `selectDistillModel` + README calibration table | **Plausible**; based on author calibration, not published ablation |
| Hybrid search | Sequential fallback, **not** fusion (`search:68`) | **Overstated word** — README itself is honest ("Falls back to TF-IDF"); evidence file correctly flags "*not* true score fusion" |
| No published benchmarks (LoCoMo/LME) | Confirmed — none exist | Honest; not a retrieval-competition entrant |

---

## Relevance to Somnigraph

### What VIR does that Somnigraph doesn't
- **Write-path quality gating.** This is the one place VIR is architecturally ahead. Somnigraph's `remember()` (tools.py) stores whatever the agent decides to store — there is no pre-storage confidence gate, no cheap heuristic filter, no classify-then-threshold funnel. VIR's `filter.ts` → classify → `confidence <= 0.6` drop is a concrete, cheap implementation of exactly the "write-path discipline > retrieval tuning" thesis from the Phase 18 source sweep (ByteRover/agentmemory/MemPalace). Somnigraph has no analog because it is agent-curated rather than auto-ingesting — but if it ever adds bulk transcript ingest, this is the reference recipe.
- **Automated retroactive transcript ingest.** VIR turns months of existing `.jsonl` history into notes in one run. Somnigraph has no ingest pipeline at all; memories arrive only via live `remember()` calls.
- **Human active-learning loop with a verified flag** (`vir review`). Somnigraph's feedback is agent-supplied per-query utility (richer, but no human approve/edit/reject gate on the memory itself).
- **Markdown-as-source-of-truth transparency.** Every memory is an inspectable, editable file. Somnigraph memories live in SQLite (opaque without tooling).

### What Somnigraph does better
- **Retrieval is a different league.** Somnigraph: BM25+vector with **RRF fusion** (`fts.py`, k=14 Bayesian-tuned), a **26-feature LightGBM reranker** (`reranker.py`, NDCG 0.7958), **PPR graph expansion** (`scoring.py`). VIR: TF-IDF-or-embedding sequential fallback + a flat +0.2 verified boost + MMR. No fusion, no learning, no graph.
- **Feedback loop.** Somnigraph has explicit utility ratings, EWMA, UCB exploration, Spearman r=0.70 with GT, Hebbian PMI edge strengthening. VIR's "feedback" is a static +0.2 flag.
- **Sleep consolidation & graph.** Somnigraph's NREM/REM LLM-mediated edge detection, typed edges, decay. VIR has no graph, no decay, no consolidation cycle — only on-demand dedupe/lint.
- **Benchmarked.** 85.1% LoCoMo QA. VIR has zero benchmarks.

---

## Worth Stealing (ranked)

### 1. Cheap heuristic pre-filter before any LLM write (Low)
**What**: `filter.ts`'s no-LLM structural score (line count, tool-call count, files touched, a signal-word regex) as a gate before spending tokens on classify/distill.
**Why**: Corroborates the Phase 18 write-path-discipline finding with a working, dirt-cheap implementation. If Somnigraph ever adds a transcript-ingest or sleep-time bulk-capture path, this is the salience gate that keeps the DB from filling with low-signal episodics.
**How**: A `salience_score()` helper analogous to `filter.ts:scoreSession`, called before an auto-`remember()` in any future ingest module. Not applicable to the current agent-initiated `remember()` (the agent *is* the gate), so revisit-if, not adopt-now.

### 2. Classify-then-confidence-threshold write funnel (Medium)
**What**: Two-stage LLM gate — cheap classify emits a `confidence`, drop below a threshold *before* the expensive extraction step (`distiller.ts:445`).
**Why**: Somnigraph has no write-path confidence signal at all. A stored memory's trustworthiness is implicit. A `confidence` field + threshold would let sleep-time consolidation deprioritize or quarantine low-confidence memories.
**How**: Add an optional `confidence` to the memory schema (`db.py`), populated at sleep time or on auto-capture; use it as a reranker feature and a consolidation/archival input. Redundant with priority to a degree, but confidence ≠ priority (one is trust, the other is importance).

### 3. MMR diversity term as a reranker complement (Low-Medium)
**What**: `mmrRerank` — greedily trade relevance against max-similarity-to-already-selected so the top-k covers distinct facets, not near-duplicates.
**Why**: Somnigraph's pointwise LightGBM reranker (`reranker.py`) scores each memory independently — it has no notion of result-set diversity, so a cluster of near-duplicate memories can dominate the top-k. A post-rerank MMR pass could improve multi-hop coverage (relevant to the 88% vocabulary-gap ceiling).
**How**: A diversity pass in `scoring.py` after reranker scoring, using existing embeddings for the pairwise-similarity term. Low risk, additive; worth a small experiment against the LoCoMo multi-hop set.

---

## Not Useful For Us

### Obsidian-native / markdown-vault storage
Somnigraph is deliberately SQLite-backed and single-user via MCP; the vault-as-frontend model is a product choice orthogonal to Somnigraph's research goals.

### Hybrid Haiku/Sonnet cost routing
Real cost engineering, but Somnigraph's sleep pipeline cost is not a documented pain point and its LLM calls are already few and offline. Note-only.

### Kie.ai gateway, daemon/launchd/systemd plumbing, npm packaging
Deployment concerns irrelevant to a research artifact.

---

## Connections

- **Phase 18 source sweep** (ByteRover, agentmemory, MemPalace; see `docs/sessions/2026-06-28-phase18-source-sweep.md`): VIR is another independent data point for "write-path quality, not retrieval, is what the leaders win on." VIR's whole value proposition *is* the write path (distillation + confidence gating); its retrieval is intentionally trivial. Strong convergent evidence.
- **`docs/proactive-injection.md`**: VIR's `sync-claude` (top-5-per-category, confidence-gated, diff+confirm injection into `CLAUDE.md`) is a *static, non-query-conditioned* cousin of Somnigraph's planned floor-gated per-prompt hint injection. Somnigraph's design is strictly richer (query-conditioned, cooldown, Thompson gating) — VIR confirms the injection-into-CLAUDE.md channel is a real pattern others ship, but offers nothing to add.
- **Other LLM-Wiki implementations** (lucasastorian/llmwiki, Pratiyush/llm-wiki, nashsu/llm_wiki): VIR positions itself as the Obsidian-native + retroactive variant. Same family; none benchmarked.

---

## Summary Assessment

VIR is a well-engineered, honest, small-scope **knowledge-distillation tool**, not a memory-retrieval system. Its core contribution is a disciplined write path: a cheap no-LLM heuristic pre-filter, a classify-then-confidence-threshold gate that drops low-signal material before the expensive extraction, tool-output truncation to bound cost, hybrid model routing, and a human active-learning loop (`vir review` → `verified` flag). The retrieval side is deliberately minimal — TF-IDF-or-embedding sequential fallback, MMR diversity, a flat +0.2 verified boost — with no fusion, no learned reranker, no graph, no decay, and no benchmarks.

The single most valuable thing for Somnigraph is corroboration, not code: VIR is another independent vote for the Phase 18 thesis that write-path quality beats retrieval tuning. Somnigraph is the mirror image — a research-grade retrieval stack (RRF + LightGBM + PPR + feedback loop) bolted onto an *ungated* write path (`remember()` stores whatever the agent hands it). VIR's `filter.ts` + `distiller.ts:445` funnel is the reference implementation of the gate Somnigraph lacks. It is not directly adoptable today because Somnigraph is agent-curated rather than auto-ingesting, but it becomes the blueprint the moment any bulk/auto-capture path is added.

The evidence file is unusually accurate and honest — it correctly flags the "not true score fusion" caveat, the absence of benchmarks, and the missing decay/entities/layers, and its feature-absence list matches the code. No sharp corrections needed; the only thing to underline is that VIR's "hybrid search" is a fallback chain, not fusion, and its "confidence" is model self-report, not externally validated — so the 0.91-avg-confidence headline is a process metric, not a quality measurement. Verdict: **MAYBE** — nothing to adopt wholesale, but the write-path gating recipe is a genuine revisit-if for a future ingest path, and the MMR diversity term is a cheap experiment worth a look.
