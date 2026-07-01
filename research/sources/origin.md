# Origin (now "Wenlan") — Local-daemon Rust memory system with source-cited wiki pages, write-path quality gating, and faithfulness-gated distillation

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

<!-- No academic paper; this is a product repo. Skipping Paper Overview. -->

## Architecture

Origin is a ~122k-LOC Rust monorepo (crates: `wenlan-core`, `wenlan-server`, `wenlan-mcp`, `wenlan-cli`, `wenlan-types`). It runs as a single **local daemon** (`127.0.0.1:7878`) that owns one store; every MCP client (Claude Code, Cursor, Codex, Gemini, etc.) reads the same daemon. Works fully offline with no API key; a local Qwen (llama-cpp-2) or Anthropic key is optional and only enables LLM classification/extraction/distillation.

**Naming note:** the clone URL `7xuanlu/origin` now redirects to `7xuanlu/wenlan` (文瀾, an imperial library). The carsteneu evidence file audited it under the old name "origin"; the current code, README, and npm packages all say "wenlan." Same system.

### Storage & Schema
- **libSQL** (Turso's SQLite fork) + **FTS5** (`memories_fts` virtual table) + **DiskANN vector index**. Embeddings are **BGE-Base-EN-v1.5-Q, 768-dim**, local via fastembed. `SCHEMA_VERSION = 54`.
- `memories` table is wide (~40 columns): `content, title, summary, memory_type, domain, source_agent, confidence, confirmed, supersedes, supersede_mode, pinned, pending_revision, structured_fields, retrieval_cue, source_text, stability, access_count, last_accessed, effective_confidence, embedding, version, changelog, …`. Plus `entities`, `relations`, `observations`, `pages`, `page_sources`, `rejected_memories`, `agent_connections`, `spaces`, `access_log`.
- **Two-tier unit model**: atomic **memories** + distilled **source-cited wiki pages** (`pages` + `page_sources` map every page sentence back to memory IDs). Pages are themselves retrievable and feed retrieval alongside the atomic notes.
- **Git versioning**: every memory/page/session write auto-commits into `~/.wenlan/.git/`, so artifacts can be diffed/reverted/branched.

### Memory Types
Six semantic types chosen at capture: `decision, lesson, gotcha, preference, fact, correction`. Plus a knowledge graph of `entities` (people/projects/tools) and `relations`, working memory (`working_memory.rs`), and a profile narrative (`narrative.rs`). "Spaces" act as tags/keywords with a 6-layer resolution order.

### Write Path
This is the system's center of gravity and its richest area. `post_write.rs` (2.3k LOC) is the canonical create flow shared by both agent-triggered and daemon-triggered paths:
1. **Rule-based quality gate** (`quality_gate.rs`, 1.6k LOC): rejects noise pre-store — system-prompt echoes, heartbeats, credential leaks (regex), trivially short content.
2. **Novelty/dedup gate** (`db.check_novelty_batch`): batch-embeds survivors and rejects a store whose cosine similarity to an existing memory ≥ `novelty_threshold` (default **0.75**) → `RejectionReason::NotNovel`. Rejected items land in `rejected_memories`.
3. **Contradiction check** (`contradiction.rs`): cheap structured-field pre-filter (bigram Jaccard on claims, context-keyed preference comparison) gates a fuller LLM contradiction check; outcomes are `Consistent / Contradicts / Supersedes{merged}`. Conflicts surface for human review rather than silently entering context.
4. **Post-ingest enrichment** (`post_ingest.rs`): entity linking, title enrichment, recap detection, page growth, effective-confidence update.

### Retrieval
- **Hybrid RRF** (`db.rs::search_memory*`): vector + FTS5 fused with `1/(rrf_k + rank)`, **rrf_k = 60**, per-channel weights (`cw.vector`, `cw.fts`). FTS channel has a normalized-score variant (`fts_weight * norm / rrf_k`) scale-matched to the vector channel's `1/rrf_k` max. An opt-in **5th RRF stream** injects entity→memory graph hits or cross-episode context.
- **Reranking is off by default.** Two optional rerankers exist: (a) `reranker.rs` — off-the-shelf **cross-encoder** (fastembed BGE-Reranker-V2-M3 / Jina), `reranker_mode` default OFF; (b) `rerank.rs` — an **on-device LLM listwise** scorer that prompts the local model to emit 0–1 scores. There is **no learned/trained reranker** and no feature model.
- **CE⊕RRF blend** (`retrieval/blend.rs`, opt-in `WENLAN_ENABLE_RERANK_BLEND`, default OFF): `α·σ(CE) + (1−α)·norm(RRF)` with query-type-weighted α — 0.5 for temporal/relational queries, 0.75 for single-hop — explicitly credited to "SuperLocalMemory." Replaces an earlier REPLACE behavior where the CE logit erased recency/salience boosters.
- Three exposed search modes: `search_memory`, `search_memory_reranked`, `search_memory_expanded` (query expansion).

### Consolidation / Processing
- **Refinery** (`refinery/`, 2.5k LOC) + **synthesis/distill.rs** (1.6k LOC): background daemon "distill cycles" cluster related memories (`cluster_by_similarity`, cosine threshold), LLM-merge/split clusters, and recompile them into source-cited pages. `/distill` triggers a deliberate pass.
- **Faithfulness gate on generated prose** (the standout): distilled page bodies and the hierarchical corpus-summary prelude (`refinery/summary.rs`) are gated behind a **lexical content-token overlap floor** (`BODY_OVERLAP_FLOOR = 0.5`): each sentence must have ≥50% of its content tokens (len≥4, non-stopword) whole-word-match the union of source memories, or the LLM body is discarded and a **deterministic template** ships instead. Never ships unverified prose; never a silent empty node. Same 50% rule is the `eval/page_faithfulness.rs` benchmark.
- **GraphRAG-style prelude** (`refinery/summary.rs`, ship-dark/opt-in): two-level rollup (per-community bucket summaries + root) prepended as `## Corpus Overview` at read time.

### Lifecycle Management
- **Decay** (`decay.rs`): `recency_boost = 1/(1 + rate·days_since_access)` × `access_boost = 1 + 0.1·ln(1+access_count)`, multiplied into an `effective_confidence`. Rates by **stability tier** — Ephemeral 0.05, Standard 0.01, Protected 0.001; `confirmed` or `pinned` → rate 0.0 (decay-immune). Reheat via access.
- **Supersession** chains (`supersedes`, `supersede_mode=hide`) keep old versions visible; explicit `forget` deletes by ID; PII redaction (`privacy.rs`).

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| "Refuses unsourced pages / no hallucinated summaries" | Real code: `BODY_OVERLAP_FLOOR=0.5` faithfulness gate in `refinery/summary.rs` + `synthesis/distill.rs`, page_sources provenance | **Validated** (lexical, not semantic — paraphrase can false-reject, acknowledged in code) |
| "Dedupes facts, supersedes old versions" | `check_novelty` (≥0.75 reject), `contradiction.rs` supersession, `supersedes` field | **Validated** |
| LoCoMo Recall@5 70.0%, MRR 0.647, NDCG@10 0.684 | README eval table, self-labeled "Retrieval-only, not end-to-end answer quality" | **Retrieval-recall only — NOT comparable to Somnigraph's 85.1% LoCoMo QA** |
| LongMemEval-Oracle R@5 93.6% | README; oracle = gold session provided | Validated as retrieval metric; oracle setting inflates vs real retrieval |
| End-to-end QA exists | `docs/eval/`: LME-S deep full-stack + CE reranker = **76.7%** (46/60); prior 500-Q full-stack = 282/500 = **56.4%** | Plausible; small-N (60), LME not LoCoMo, so still not a like-for-like vs our 85.1 |
| "Evolves on its own between sessions" | Daemon distill cycles + scheduler, real code | Validated |

---

## Relevance to Somnigraph

### What Origin does that Somnigraph doesn't
- **Write-path admission control.** `quality_gate.rs` (noise/credential/too-short rejection) + `check_novelty` store-time dedup at 0.75 is exactly the **write-path quality gating** Somnigraph lacks (STEWARDSHIP notes this gap; `tools.py::impl_remember` only dedupes at 0.9 similarity and admits everything else). Origin keeps a `rejected_memories` audit table.
- **Faithfulness gate on generated prose.** Somnigraph's `scripts/sleep_rem.py` generates summaries/gestalt/questions with **no groundedness check**; Origin gates every distilled sentence against source tokens and falls back to a deterministic template.
- **Real-time entity/KG construction + contradiction/supersession at write time** (`kg/`, `contradiction.rs`) vs Somnigraph building edges only during NREM sleep (`sleep_nrem.py`).
- **Source-cited retrievable wiki pages** as a second memory tier feeding retrieval alongside atomic notes — richer than Somnigraph's detail/summary/gestalt *layers* of a single memory.
- **One daemon, many MCP clients** + git-versioned Markdown artifacts. Somnigraph is single-client, SQLite-only.

### What Somnigraph does better
- **Learned reranker.** Somnigraph's 26-feature LightGBM reranker (NDCG 0.7958, +6.17pp over formula, `reranker.py`) is a real trained model with measured lift. Origin has **no learned reranker** — only an off-the-shelf CE and an on-device LLM scorer, both **off by default**, so live Origin retrieval is essentially bare RRF.
- **Explicit feedback loop with measured GT correlation** (per-query Spearman r=0.70, EWMA+UCB). Origin has decay/access boosts but **no retrieval-feedback signal** — nothing learns from whether a recall was useful.
- **Graph-conditioned retrieval via PPR** (`scoring.py`). Origin's graph is an entity KG surfaced as a boost-only 5th RRF stream, not PPR expansion with betweenness as a ranking feature.
- **Benchmarking rigor.** Somnigraph reports end-to-end LoCoMo QA (85.1% Opus judge) with multi-hop failure analysis; Origin's headline LoCoMo/LME numbers are retrieval-recall, and its one end-to-end number is small-N LME-S.

---

## Worth Stealing (ranked)

### 1. Faithfulness gate on sleep-generated prose (Low)
**What**: Before REM writes a summary/gestalt/question, require ≥N% of its content tokens (len≥4, non-stopword) to whole-word-match the union of source memories; below the floor, ship a deterministic template instead of the LLM prose. Origin uses a 0.5 lexical floor (`refinery/summary.rs::BODY_OVERLAP_FLOOR`, `eval/page_faithfulness.rs::score_sentence_faithful`).
**Why**: `sleep_rem.py` currently trusts LLM summaries/gestalts blindly; commit 164fcb5 already had to defensively drop *malformed* consolidation items — a groundedness floor is the principled version. This is the write-path-discipline theme Phase 18 flagged as what the LoCoMo/LME leaders win on.
**How**: Add a pure `score_faithful(sentence, source_union)` helper (token overlap, no embedding needed — even cheaper than the prior nugget's cosine framing) to the REM pipeline; on failure, fall back to a template-assembled summary or skip. Could optionally regenerate once before falling back.

### 2. Store-time novelty + noise admission gate (Medium)
**What**: A pre-store gate that (a) rejects structural noise (system-prompt echoes, heartbeats, credential-shaped strings, sub-threshold length) and (b) rejects near-duplicates at a tunable cosine threshold (Origin: 0.75), logging rejections to an audit table rather than dropping silently.
**Why**: Somnigraph admits nearly everything (dedup only at 0.9); a lower, tunable novelty floor plus a noise filter would raise write-path quality — the corroborated lever from AMemGym/ByteRover/agentmemory analyses.
**How**: Extend `tools.py::impl_remember` with a `quality_gate` step and a `rejected_memories` table; expose `novelty_threshold` in config. Watch calibration: too-aggressive dedup can bury legitimate reinforcement.

### 3. Query-type-weighted CE⊕RRF blend, not REPLACE (note-only)
**What**: When a cross-encoder is added, blend `α·σ(CE)+(1−α)·norm(RRF)` with α by query type (temporal/relational 0.5, single-hop 0.75) so CE never erases recency/salience boosters (`retrieval/blend.rs`).
**Why**: Relevant only if Somnigraph adds a CE stage; our learned reranker already integrates signals as features, so this is mostly a cautionary pattern (don't let a reranker overwrite fused signal).

---

## Not Useful For Us

- **On-device LLM listwise reranker** (`rerank.rs`): a local model scoring candidates 0–1 is slower and less calibrated than our trained LightGBM; no reason to adopt.
- **Git-versioned Markdown artifacts / Obsidian export**: nice product feature, orthogonal to Somnigraph's research goals.
- **Multi-client daemon architecture**: Somnigraph is deliberately single-user single-client.
- **Off-the-shelf CE reranker models**: we have a domain-trained reranker that outperforms generic CE.

---

## Connections

- **Write-path discipline** convergence: reinforces the Phase 18 finding (`ai-memory-comparison.md`, `byterover.md`, `agentmemory.md`, AMemGym) that leaders win on write-time quality, not exotic retrieval — Origin independently lands on rule-gate + novelty-dedup + contradiction-at-write.
- **Faithfulness/groundedness gate**: same instinct as verbatim-source systems (MemPalace) and any provenance-first design; Origin's cheap *lexical* implementation is the notable twist.
- **CE⊕RRF blend** explicitly cites **SuperLocalMemory**'s query-type-weighted α — cross-reference if that system is profiled.
- **Two-tier atomic + distilled-page** model echoes GraphRAG community summaries (its `## Corpus Overview` prelude is literally a GraphRAG rollup) and any "notes → synthesized wiki" system.
- **RRF k=60 + weighted channels**: same family as Somnigraph's RRF (ours k=14, Bayesian-optimized; Origin uses the textbook 60).

---

## Summary Assessment

Origin/Wenlan is a serious, well-engineered local-first memory *product* whose real contribution is **write-path and provenance discipline**, not retrieval sophistication. The retrieval core is familiar and, tellingly, its two rerankers and its CE⊕RRF blend all ship **off by default** — live Origin runs on bare weighted-RRF, which is *weaker* than Somnigraph's learned-reranker + feedback-loop retrieval stack. On the axes Somnigraph optimizes (learned ranking, measured feedback correlation, PPR graph retrieval, end-to-end QA benchmarking), Somnigraph is ahead.

Where Origin is genuinely ahead is everything *around* the store: a 1.6k-LOC rule-based quality gate, store-time novelty/dedup, real-time entity extraction and contradiction/supersession, and — the single most transferable idea — a **faithfulness gate that refuses to ship LLM-generated prose unless it is lexically grounded in its sources**, degrading to a deterministic template otherwise. That mechanism maps directly onto Somnigraph's `sleep_rem.py`, is embarrassingly cheap (token overlap, no model call), and is the principled generalization of the malformed-item guard we already bolted on. Adopt that; consider the store-time admission gate; skip the rest.

One honest-accounting correction to carry forward: Origin's headline **LoCoMo Recall@5 (70.0%) is retrieval-recall, not end-to-end QA** — the evidence file itself labels it so — and must **not** be compared to Somnigraph's 85.1% LoCoMo QA. Its only end-to-end numbers are small-N LongMemEval-S (76.7%, 46/60) and a 56.4% 500-question full-stack snapshot, on a different benchmark.
