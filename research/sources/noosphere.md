# Noosphere — Self-hosted multi-agent knowledge wiki with a draft→curated memory pipeline (Postgres FTS, no vector, no LLM)

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

Noosphere is a Next.js 16 / TypeScript app backed by PostgreSQL 16 (Prisma 7 ORM) with an optional Redis 7 recall cache. It is a **wiki-first** system: the storage unit is an **Article** (Markdown page with title/slug/content/topic/tags/excerpt/confidence/status), not an extracted fact or entity. Agents read and write the same corpus humans browse in a web dashboard. The repo also ships four client plugins (OpenClaw, Hermes, Opencode, Kilo Code) and a provider-orchestration layer meant to fan out to multiple memory backends. Deliberately **no embedded LLM** — agents bring their own model; Noosphere is pure storage + retrieval.

### Storage & Schema
- Postgres via Prisma. Core models: `Article`, `Topic` (unlimited-depth parent/child tree), `Tag` (many-to-many via `ArticleTag`), `ApiKey` (SHA-256 hashed, permission + scope arrays), `Revision`, `ActivityLog`.
- Article fields include `confidence` (low/medium/high), `status` (draft/reviewed/published), `restrictedTags` (scope-based ACL), `deletedAt` (soft delete), `lastReviewed`, `sourceType`, `sourceUrl`, `authorName`, `revisions[]`.
- Restricted-tag scopes act as row-level access control: unrestricted articles are always visible; restricted ones require a matching key scope (`src/lib/memory/article-search.ts:buildArticleSearchFilters`).

### Memory Types
No cognitive taxonomy (no episodic/semantic/procedural). Instead a **curation ladder** mapped from article `status`: `draft → ephemeral`, `reviewed → managed`, `published → curated` (`noosphere.ts:mapCurationLevel`). Confidence (low/medium/high) is a separate orthogonal axis mapped to 0.33/0.66/1.0 (`mapConfidenceScore`).

### Write Path
This is the most interesting part of the codebase. `src/lib/memory/api/save.ts` runs real **write-time quality gating** before an article is created:
- **Injected-memory-block stripping** (`noosphere-injected-memory/src/index.ts`): strips `<recall>`, `<hindsight_memories>`, `<noosphere_auto_recall>` blocks (with nested-depth balancing) from content before save, so an agent cannot re-ingest its own recall-injection output back into the store. Directly attacks the recall→save feedback-pollution loop.
- **Secret detection**: regex panel for OpenAI/GitHub/AWS/JWT/bearer keys, `-----BEGIN PRIVATE KEY-----`, and `key=…`-style assignments; rejects the save.
- **Transient-content rejection**: `TRANSIENT_ONLY_PATTERNS` reject "thanks", "ok", "remind me…"; `validateDurableContent` requires ≥40 chars and a 12+-letter prose run.
- Agents **always** save as `status: "draft"` (`SanitizedMemorySaveInput.status: "draft"`). Humans (or a scheduled review) promote through the ladder. Slug auto-dedup, `connectOrCreate` tags, revision row created atomically.

### Retrieval
Single-channel **PostgreSQL full-text search only** — no vector, no BM25 fusion, no learned reranker. `article-search.ts:buildSearchableCTE` builds a weighted `tsvector` (title=A, excerpt=B, content=C, tag-names=B), ranked by `ts_rank(document, websearch_to_tsquery('simple', q))` ordered `rank DESC, updatedAt DESC` (`noosphere.ts:queryRankedArticleRows`). If the strict query returns zero rows, a **looser fallback tsquery** OR-joins stopword-filtered terms plus a tiny hardcoded synonym table (photo/image/screenshot/attachment). Relevance is normalized as `rank / maxRank` within the result set.
- On top sits a **RecallOrchestrator** (`orchestrator.ts`) that fans out concurrently to N providers (Noosphere FTS + Hindsight HTTP), then applies a hand-weighted **composite score** = `0.4·relevance + 0.25·confidence + 0.2·recency + 0.15·curation` (`types.ts:COMPOSITE_WEIGHTS/computeBaseCompositeScore`). Recency is `exp(-ageDays·ln2/90)` (90-day half-life). Constants are fixed, not learned or tuned.
- **Dedup** (`dedup.ts`): collapses results sharing an exact `canonicalRef` (`noosphere:article:<id>`), keeping best-score / provider-priority / most-recent winner and preserving provenance.
- **Conflict resolution** (`conflict.ts`): pairwise divergence score; `content !== content` string inequality contributes 0.4 (the dominant term), plus curation/confidence/recency deltas; strategies surface/accept-highest/accept-recent/accept-curated/suppress-low.
- **Token-budget manager** (`budget.ts`) trims the injected recall block to a token cap; output is formatted as `<recall query="…"><memory …>…</memory></recall>`.

### Consolidation / Processing
Rule-based, **not LLM, not sleep**. A local scheduler (`scheduler.ts`, `npm run memory:scheduler`) runs maintenance jobs. **Promotion** (`promotion.ts`): a memory recalled ≥`minRecallCount` (default 3) times with avg relevance ≥0.5 becomes a `pending` promotion candidate → human approve/reject → **backfill/synthesis** (`backfill.ts`) creates or updates an article. Content-merge strategies are `append` / `replace` / `merge` — but **`merge` is an explicit placeholder ("future — placeholder for now")**. No semantic synthesis; no clustering beyond the topic tree.

### Lifecycle Management
Soft delete (`deletedAt` + trash UI, restore), per-article revision history, `PATCH` supersede. **No decay/forgetting** of stored articles — recency is only a retrieval-score multiplier, never eviction. No versioning of valid-from/valid-until temporal ranges.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| "Cross-provider deduplication collapses exact, canonical, or **semantic** overlap" (evidence file) | `dedup.ts` groups strictly by `canonicalRef` string equality (or `providerId:id` fallback) | **Overstated** — no semantic/embedding dedup; exact-key only |
| "Conflict detection … resolve **semantic** conflicts" | `conflict.ts:computeConflictScore` is `content !== content` string inequality + metadata deltas | **Overstated** — flags any two differently-worded results as "conflicting"; no contradiction semantics |
| Hybrid BM25+vector search | Evidence marks vector ❌ "planned"; code confirms only Postgres `ts_rank` FTS exists | Accurate (absent) |
| "Synthesize older material into wiki articles" | `backfill.ts` does template append/replace; `merge` is a stubbed placeholder; no LLM in repo | **Overstated** — mechanical concat, not synthesis |
| Composite ranking (relevance+confidence+recency+curation) | `types.ts` fixed weights 0.4/0.25/0.2/0.15 | Real but hand-tuned; no learning/eval |
| Write-time secret + transient + injected-block gating | `api/save.ts`, `noosphere-injected-memory/src/index.ts` — fully implemented + tested | **Validated** (strongest real capability) |
| Promotion by recall frequency → human review → article | `promotion.ts` + `backfill.ts` fully implemented | Validated (rule-based) |
| No published benchmarks | No LoCoMo/LongMemEval/PERMA anywhere in repo | Accurate — **no comparable QA numbers** |

---

## Relevance to Somnigraph

### What Noosphere does that Somnigraph doesn't
- **Write-path quality gating** — Somnigraph's `tools.py:impl_remember` accepts whatever it's given; Noosphere's `api/save.ts` rejects secrets, transient chatter, and too-short content before write. This is exactly the gap named in `STEWARDSHIP.md` ("write-path quality gating") and corroborated by the Phase 18 finding that write-path discipline, not retrieval, is what LoCoMo leaders win on.
- **Recall-injection strip guard** — prevents an agent's recall output from being re-saved as new memory. Somnigraph has no analog; its `docs/proposals/proactive-injection.md` design injects recall hints into prompts, so the same self-ingestion risk exists if that content ever loops back into `remember()`/sleep ingestion.
- **Human-in-the-loop curation ladder** (draft→reviewed→published with a web UI, revision history, trash/restore, scoped ACL). Somnigraph is single-user autonomous with no review surface.
- **Multi-provider recall orchestration** — concurrent fan-out + provenance across heterogeneous backends. Somnigraph is single-store.

### What Somnigraph does better
- **Retrieval is a different league.** Somnigraph = hybrid BM25+vector, RRF (k=14 Bayes-tuned), 26-feature LightGBM reranker (NDCG 0.7958), PPR graph expansion. Noosphere = one Postgres `ts_rank` channel + a fixed-weight linear composite. No embeddings, no learned ranking, no graph (`scoring.py`/`reranker.py` have no counterpart).
- **Consolidation**: Somnigraph's LLM-mediated NREM/REM sleep (edge typing, merge/archive, gap analysis) vs Noosphere's threshold-count promotion + template concat (`merge` unimplemented).
- **Feedback loop**: Somnigraph has explicit 0–1 utility ratings, EWMA, UCB, Hebbian PMI edges, measured Spearman r=0.70. Noosphere's "feedback" is a recall-count tally feeding promotion eligibility — no utility signal.
- **Decay/lifecycle**: Somnigraph has per-category exponential decay with reheat; Noosphere never forgets stored articles.

---

## Worth Stealing (ranked)

### 1. Recall-injection strip guard on the write path (Low)
**What**: Before persisting any user/agent-supplied content, strip previously-injected recall blocks (Noosphere strips `<recall>…</recall>` etc. with nested-depth balancing) so the store never re-ingests its own retrieval output.
**Why**: Somnigraph's `docs/proposals/proactive-injection.md` plan surfaces recall hints in prompts. If any downstream path (a save, or sleep ingesting a transcript/journal) captures that text, the feedback loop self-reinforces — the exact self-ingestion pathology the strip guard blocks. Cheap insurance for a capability Somnigraph is about to add.
**How**: A small sanitizer in `tools.py:impl_remember` (and any sleep content-ingest in `scripts/sleep_*.py`) that removes Somnigraph's own recall-block delimiters before storing. ~30 lines, matching `noosphere-injected-memory/src/index.ts`.

### 2. Write-time transient/secret rejection (Low)
**What**: Reject saves that are pure acknowledgements ("thanks", "ok", "remind me…"), too short, non-prose, or contain credential patterns (API keys, JWTs, private-key blocks) — before they become durable memories.
**Why**: Directly addresses Somnigraph's own stated `STEWARDSHIP.md` gap and the Phase 18 write-path-discipline conclusion. Somnigraph's `remember()` currently has no such filter; junk/secret memories dilute retrieval and leak.
**How**: Port `save.ts:validateDurableContent` + `SECRET_PATTERNS` + `TRANSIENT_ONLY_PATTERNS` as a pre-write gate in `tools.py`. Config-driven, returns a soft rejection rather than silently storing.

---

## Not Useful For Us

- **Wiki UI / topic tree / revision history / trash / scoped API keys** — human-collaboration surface for a multi-agent shared corpus; Somnigraph is single-user autonomous.
- **Composite linear ranking + FTS-only retrieval** — strictly weaker than Somnigraph's learned reranker + hybrid fusion; nothing to import.
- **Conflict "detection" (string inequality)** — too coarse to be useful; Somnigraph's typed supports/contradicts edges (detected in sleep) are far more principled.
- **Multi-provider orchestrator** — solves a problem (heterogeneous backends) Somnigraph doesn't have.

---

## Connections

- **Convergent with the Phase 18 sweep** (`docs/sessions/2026-06-28-phase18-source-sweep.md`, ByteRover/MemPalace/agentmemory): another leader-adjacent system whose real strength is **write-path quality, not retrieval** — Noosphere's retrieval is bare Postgres FTS, yet its write gating is genuinely engineered. Independent corroboration of "discipline the write path" over "add a third confidence axis."
- **Curation ladder** (ephemeral/managed/curated + draft/reviewed/published) echoes the draft-to-curated staging seen in wiki-style systems; contrast with Somnigraph's priority + decay, which achieves salience ranking without a human review gate.
- **No-LLM stance** aligns with the "bring-your-own-model storage layer" pattern; unlike systems that embed extraction LLMs, Noosphere pushes all intelligence to the agent — which is why its consolidation is mechanical.

---

## Summary Assessment

Noosphere's core contribution is a **well-engineered shared knowledge wiki for agents with a genuine write-time discipline layer** — secret scrubbing, transient rejection, and (notably) stripping the system's own recall-injection blocks out of content before it is stored. That last mechanism is the one idea here worth carrying into Somnigraph, precisely because Somnigraph's proactive-injection design is about to create the self-ingestion risk it defends against. The write-path gating as a whole maps cleanly onto Somnigraph's acknowledged weakest seam.

Everything downstream of the write path, though, is well behind Somnigraph. Retrieval is a single Postgres `ts_rank` channel dressed up with a fixed-weight linear composite; there is no vector search (the README says "planned"), no learned ranking, no graph, and no LLM anywhere. The evidence file's "semantic deduplication," "semantic conflict resolution," and "synthesis" claims all overstate what the code does: dedup is exact-key, conflict is `content !== content`, and synthesis is a template concat whose `merge` mode is an unimplemented placeholder. There are **no published benchmarks**, so nothing here is comparable to Somnigraph's 85.1% LoCoMo QA.

Verdict **MAYBE**: no core idea reshapes Somnigraph, but the write-path gating — especially the recall-block strip guard — is a concrete, low-effort revisit-if item that lands on a real, self-identified gap. Take the write-path hygiene; ignore the retrieval and consolidation stack.
