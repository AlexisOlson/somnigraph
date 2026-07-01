# gbrain - Multi-user "company brain" daemon: zero-LLM write-time typed KG + gap-aware LLM synthesis + 20-phase dream cycle

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

gbrain (by Garry Tan) is a large, mature TypeScript/Bun product (v0.42.53, ~2600 files, ~2MB CHANGELOG). It is not a research artifact — it is a shipping personal/company knowledge daemon. The memory unit is a **markdown page in a git repo** ("brain repo"), chunked and embedded into Postgres. Default storage is PGLite (Postgres 17 WASM, zero-config); production uses real Postgres + pgvector + pg_trgm.

### Storage & Schema
- **Postgres + pgvector (HNSW) + pg_trgm**, `src/schema.sql`. Content lives as markdown files on disk; DB holds `pages`, `chunks`, embeddings, `links` (the graph edge table), `takes`, `facts`, `slug_aliases`, `sources`.
- **Multi-tenancy via `sources`**: every page/chunk carries `source_id` (a "brain-within-the-DB"). `config` JSONB holds federation + ACL. Team deployments scope reads by login (OAuth 2.1). Soft-delete with 72h recovery window (`archived`/`archive_expires_at`).
- Pages are typed by top-level directory: `people/`, `companies/`, `deals/`, `meetings/` — a CRM/founder domain model, not generic memory.

### Memory Types
Pages (typed by dir) → chunks (embedded units) → `facts` (structured `## Facts` fences parsed off entity pages) → `takes` (consolidated opinion/fact claims, kind='fact', with holder + row_num audit trail) → `concepts`/`atoms` (LLM-extracted during dream). Plus `emotional_weight` + `take_count` as an orthogonal salience axis.

### Write Path — the differentiator
**Zero-LLM typed-edge extraction on every `put_page`** (`src/core/link-extraction.ts`, 1229 lines, called from `operations.ts put_page`). Pure functions turn page content into edge candidates:
- Parses markdown links, `[[wikilinks]]`, `[[source:slug]]` qualified links, bare `[[basename]]` (opt-in global-basename resolution), and frontmatter fields (`key_people`, `investors`, `attendees`, `founded`).
- `inferLinkType()` assigns a **typed verb via regex precedence** `founded > invested_in > advises > works_at > role-prior > mentions` (`FOUNDED_RE`, `INVESTED_RE`, `ADVISES_RE`, `WORKS_AT_RE`, plus global-context role priors). Meeting pages → `attended`. No LLM call.
- **Extraction freshness watermark**: `LINK_EXTRACTOR_VERSION_TS` (an ISO timestamp bumped when extractor logic changes); pages with `links_extracted_at < VERSION_TS OR updated_at > links_extracted_at` are re-extracted lazily by `gbrain extract --stale`. Same pattern as `CHUNKER_VERSION`.

### Retrieval
Hybrid, `src/core/search/hybrid.ts` (1973 lines): keyword (BM25/pg_trgm) + vector (HNSW cosine) → **RRF fusion (k=60)** → normalize → `COMPILED_TRUTH_BOOST=2.0` → **cosine re-score blend `0.7*rrf + 0.3*cosine`** → post-fusion stages → dedup → token budget.
- **Post-fusion graph signals** (`search/graph-signals.ts`): adjacency-within-top-K hub boost (1.05×, ≥2 in-set inbound links), cross-source hub boost (1.10×), and **session-diversification MMR-lite** (0.95× DEMOTE of lower-scoring members of the same session/slug-prefix cluster — explicitly *demote*, not boost, to protect token budget). All floor-gated and fail-open.
- **Reranker** (`search/rerank.ts`): external cross-encoder via gateway — default `zeroentropyai:zerank-2` (or self-hosted via llama.cpp). **Fail-open, default OFF** except `tokenmax` mode. Not a learned in-house model.
- Recency decay (`recency-decay.ts`): per-slug-prefix **hyperbolic** decay `coef * halflife/(halflife+days)`, longest-prefix-match. Source-tier boost (curated outranks bulk chat, 0.3× demote for extracts). Autocut, alias-normalize, title-match, intent-weighted RRF-k.
- `gbrain search` = raw retrieval; `gbrain think` = retrieval + **LLM synthesis with citations + gap note**. The synthesis/gap-analysis ("nothing added since April 22, six weeks ago") is a **skill prompt** (`skills/query/SKILL.md` step 5 "Flag gaps"), not a code mechanism.

### Consolidation / Processing — the "dream cycle"
`src/core/cycle.ts` runs ~20 ordered phases (`gbrain dream`, cron-friendly): lint → backlinks → sync → synthesize (transcripts→pages via Haiku triage + Sonnet subagents) → extract → extract_facts → extract_atoms → resolve_symbol_edges → patterns → synthesize_concepts → recompute_emotional_weight → **consolidate** → propose_takes → grade_takes → calibration_profile → enrich_thin → embed → orphans → purge. Mix of deterministic (lint, extract, backlinks, citation-fix) and LLM-backed (synthesize, atoms, concepts, takes) phases with per-phase USD budget caps.
- **consolidate phase** (`cycle/phases/consolidate.ts`): per `(source_id, entity_slug)` bucket of unconsolidated facts, greedy embedding-cosine cluster (threshold 0.85), promote each cluster≥2 into a `take`, mark contributing facts `consolidated_into` — **NEVER DELETE** (facts stay as audit trail). v0.31 deterministic; Sonnet synthesis is the v0.32 rewrite.

### Lifecycle Management
Source-tier ranking (CASE-expression demote, not true decay), hyperbolic recency boost, git-delete → soft-delete cascade, `slug_aliases` canonical resolution, 72h archive recovery, contradiction flagging during dream (LLM judge samples retrieval pairs).

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Zero-LLM typed-edge KG built at write time | `link-extraction.ts` regex verb inference + frontmatter mapping; called from `put_page` | **Validated** — genuinely deterministic, real code, not aspirational |
| Graph adds +31.4 P@5 over vector-only | Internal "BrainBench", 240-page Opus-generated corpus (sibling gbrain-evals repo) | **Plausible but not comparable** — retrieval precision on a *synthetic self-generated* corpus, not QA |
| P@5 49.1%, R@5 97.9% | Same internal BrainBench | **Retrieval-recall metric, NOT end-to-end QA** — not comparable to Somnigraph 85.1 LoCoMo QA |
| Gap-aware synthesis ("what the brain doesn't know") | `skills/query/SKILL.md` prompt step 5 | **Prompt-level, not architectural** — an LLM instruction, no dedicated code mechanism |
| Reranking via zerank-2 cross-encoder | `gateway.ts` `DEFAULT_RERANKER_MODEL`; `rerank.ts` | Validated but **external, fail-open, default-OFF** except tokenmax |
| Multi-user, zero-leak team brain | `sources` tenancy + OAuth + ACL; "fuzz-tested, zero leaks" | Plausible; access-policy enforcement is partly forward-compat slots in schema |
| LongMemEval harness | `gbrain eval longmemeval` exists | **No committed score in repo** — harness present, number not reported |

---

## Relevance to Somnigraph

### What gbrain does that Somnigraph doesn't
- **Write-time graph construction.** Somnigraph builds edges only during sleep (`sleep_nrem.py`); gbrain extracts typed edges deterministically on every write (`link-extraction.ts`). This is exactly the "no real-time graph construction" gap in Somnigraph's own lacks list.
- **Multi-user tenancy** (`sources` + OAuth + ACL). Somnigraph is single-user by design — not a gap worth closing.
- **Structured entity pages + `## Facts` fences → consolidated `takes`** with an append-only audit trail (facts never deleted). Somnigraph's sleep merge/archive is coarser (memory-level).
- **Extraction freshness watermark** — version-stamped lazy re-extraction; Somnigraph has no analogue for re-running an improved edge classifier over old memories.
- **Contradiction flagging surfaced to the user** during dream (Somnigraph detects `contradicts` edges but does not surface a "heads-up" at query time).

### What Somnigraph does better
- **Learned reranker.** Somnigraph's 26-feature LightGBM (`reranker.py`, NDCG 0.7958) is a self-owned, feedback-trained model measured on real queries. gbrain leans on an external cross-encoder (zerank-2), default-off, with only a hand-tuned RRF+cosine+boost formula as the always-on path.
- **Explicit feedback loop with measured GT correlation** (Spearman 0.70) and Hebbian co-retrieval PMI. gbrain has no per-query utility feedback loop; salience is `emotional_weight`, not retrieval-outcome-driven.
- **PPR graph-conditioned retrieval** (`scoring.py`). gbrain's graph use in retrieval is lightweight multiplicative hub boosts (1.05×/1.10×) applied post-fusion, not spreading-activation.
- **Honest end-to-end QA benchmarking** (85.1 LoCoMo, Opus judge, multi-hop failure analysis). gbrain reports only internal retrieval P@5/R@5 on a self-generated corpus.

---

## Worth Stealing (ranked)

### 1. Write-time zero-LLM typed-edge extraction (Medium)
**What**: Extract graph edges deterministically at write time — explicit references, co-mentions, and pattern-inferred typed relations — instead of waiting for sleep.
**Why**: Closes the named "no real-time graph construction" gap. Sleep-detected edges lag write by a full cycle; a cheap write-time pass gives immediate adjacency for PPR expansion, and sleep can still refine/type them later. gbrain proves regex verb inference is enough to hit useful precision without an LLM.
**How**: A pure `extract_edges(summary, themes, content)` in a new module, called from `tools.py impl_remember`, writing provisional edges into `db.py`'s edge table with a `source='write'` marker; `sleep_nrem.py` upgrades/retypes them. Somnigraph's memory units lack wikilinks, so the signal would be explicit cross-references + shared-theme co-mention, not CRM verbs.

### 2. Extraction/edge freshness watermark (Low)
**What**: A `LINK_EXTRACTOR_VERSION_TS`-style version stamp so that improving the edge classifier marks all older-stamped memories stale and re-runs only the delta lazily.
**Why**: Somnigraph retrains its reranker but has no mechanism to re-derive sleep edges when the edge-detection logic improves — old memories keep stale-logic edges forever.
**How**: Add `edges_derived_at` + an `EDGE_LOGIC_VERSION` constant; `sleep_nrem.py` selects `WHERE edges_derived_at IS NULL OR edges_derived_at < EDGE_LOGIC_VERSION` per batch.

### 3. Session-diversification MMR-lite demote (Low)
**What**: In post-fusion, when multiple top-K results share a session/source cluster, keep the top one at full score and *demote* the rest (0.95×) rather than boosting the cluster — token-budget-aware anti-redundancy.
**Why**: Somnigraph has novelty-scored adjacency but no explicit "same-session near-duplicate demote" for the returned set. gbrain's own changelog notes the original "boost the cluster" framing was structurally backwards.
**How**: In `scoring.py` post-fusion, group candidates by a session/day key, demote non-max members before the final slice.

---

## Not Useful For Us

### Multi-user tenancy, OAuth, ACL, federation
Somnigraph is single-user Claude-Code memory by design. The `sources` machinery is pure overhead for that use case.

### CRM/founder entity model (people/companies/deals/meetings, invested_in/founded/advises)
Domain-specific to Garry Tan's YC use case. The *pattern* (write-time typed edges) generalizes; the *verbs* do not.

### Gap-aware synthesis prompt
It's a skill instruction ("say the brain doesn't have info rather than hallucinate"), not a mechanism. Somnigraph's REM gap-analysis (`sleep_rem.py`) already does offline gap detection more rigorously.

---

## Connections

- **Write-path-quality thesis**: gbrain is another datapoint for the Phase 18 finding (see `2026-06-28-phase18-source-sweep`) that leaders win on write-path quality, not retrieval cleverness — its whole differentiator is a *deterministic write-time* graph, and its retrieval is a fairly standard RRF+cosine+boost stack.
- **Typed edges convergent with Somnigraph's NREM classifier** and with memv/memos supersession patterns — but gbrain does it at write time with regex, where Somnigraph does it at sleep with an LLM. Same target, opposite cost/latency tradeoff.
- **External cross-encoder reranker** contrasts with Somnigraph's learned LightGBM and with the reranker-as-feedback-free-complement stance in the cognee thread.
- **"Dream cycle"** is a broader, product-flavored cousin of Somnigraph's sleep — it also does filesystem lint, citation repair, and skill self-optimization, which are out of scope for a memory server.

---

## Summary Assessment

gbrain's real contribution is **deterministic, zero-LLM, typed knowledge-graph construction at write time**, with a version-stamped re-extraction watermark and a lightweight graph-signal layer bolted onto an otherwise standard hybrid-RRF retriever. That write-time graph is the single idea worth carrying back: it directly targets Somnigraph's acknowledged "graph builds only during sleep" gap, and gbrain demonstrates that regex verb inference plus wikilink/frontmatter parsing is enough to build a useful typed graph without paying per-write LLM cost. Everything downstream of it — the "answer with a gap note," the 24/7 dream cycle, the company-brain framing — is either LLM-prompt-level or product/tenancy scaffolding that doesn't transfer to a single-user memory server.

The most important correction to the evidence file: **gbrain's headline numbers (P@5 49.1%, R@5 97.9%, +31.4 P@5) are internal retrieval precision/recall on a self-generated 240-page Opus corpus — not end-to-end QA and not comparable to Somnigraph's 85.1% LoCoMo QA.** The LongMemEval harness exists in-repo but reports no committed score. The "gap-aware synthesis" that anchors the marketing is a skill-prompt instruction, not an architectural mechanism, and the reranker is an external, default-off, fail-open cross-encoder — not a learned model like Somnigraph's. Where Somnigraph is genuinely behind is write-time graph latency; where gbrain is behind is learned ranking, a measured feedback loop, and honest QA benchmarking.

Verdict: **MAYBE** — one revisit-worthy idea (write-time zero-LLM edge extraction closing a named gap), tightly coupled to gbrain's markdown-page/CRM domain, so it's a "consider and adapt," not a drop-in adopt.
