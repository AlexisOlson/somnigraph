# ClawMem — TypeScript/Bun local memory server synthesizing QMD retrieval + SAME scoring + Thoth-style consolidation for Claude Code

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

ClawMem (`yoloshii/ClawMem`, 177★, TypeScript on Bun, MIT) is a fully-local memory server for Claude Code / OpenClaw / Hermes. It is explicitly a **synthesis of several upstream systems**, visible in source comments: **QMD** (the hybrid retrieval engine: BM25 + vector + RRF + query expansion + reranking), **SAME** (composite scoring / half-life decay layer), **Thoth** (`dream_cycle.py`, `memory_tool.py` — deductive synthesis + contradiction check, adapted verbatim per `deductive-guardrails.ts:31` and `merge-guards.ts:19`), **A-MEM** (self-evolving keyword/tag/link enrichment), and **MAGMA** (intent-classified multi-graph routing). This is an integration project, not a from-scratch engine — an honest-accounting point the evidence file omits.

### Storage & Schema
SQLite + `sqlite-vec` + FTS5. Vault at `~/.cache/clawmem/index.sqlite`. Rich document row: `id`/`docid` (6-char hash), `path`, `title`, `content`, `collection`, `content_type` (decision/preference/milestone/problem/deductive/handoff/conversation/note/research/project/antipattern/hub), `confidence` (0–1), `quality_score` (0–1), `keywords`, `tags`, `status` (active/inactive/invalidated), `pinned`, `snoozed_until`, `revision_count`, `duplicate_count`, `invalidated_at`/`superseded_by`/`invalidated_by` (soft-delete chain), `source_doc_ids` (provenance). Separate tables: `entity_mentions`, `entity_cooccurrences`, `consolidated_observations`, `memory_relations` (typed edges), co-activation counts.

### Memory Types
Content-first taxonomy (12 `content_type` values) rather than Somnigraph's cognitive categories (episodic/semantic/procedural/…). Observations captured by a local GGUF **observer model** are typed decision/preference/milestone/problem/discovery/feature. SPO knowledge-graph triples use a tight predicate vocabulary (adopted, migrated_to, deployed_to, replaced, depends_on, prefers, avoids, caused_by, resolved_by…).

### Write Path
Multiple automated paths, mostly hook-driven: a `decision-extractor` Stop-hook runs the local observer LLM over the transcript → writes observations, infers causal links, persists SPO triples; a `handoff-generator` summarizes the session. Enrichment: **A-MEM** adds keywords/tags/links; **entity extraction** → quality filters → type-agnostic canonical resolution within compatibility buckets (person/org/location/tech). **Quality scoring** is compositional (structure + keywords + metadata richness). **Dedup** at hook-output level via normalized content hashing within a 30-min window. Notably ClawMem *does* have write-path quality/entity gating — a gap Somnigraph has.

### Retrieval
Full hybrid. `reciprocalRankFusion` (`search-utils.ts:85`, k=**60**, plus a small top-rank bonus +0.05/+0.02) fuses BM25 + vector lists; **original-query lists get 2× RRF weight, expanded-query lists 1×**. Cross-encoder reranking via `blendRerank` (`search-utils.ts:159`): reranker is dominant (weight 0.9), RRF normalized to [0,1] as a thin tiebreaker. The reranker is an **external pretrained `zerank-2-seq` cross-encoder** run as a Docker sidecar (`extras/rerankers/`) — *not* a learned/feedback-trained model. `blendRerank` has a **degenerate-floor guard** (`RERANK_DEGENERATE_FLOOR=1e-4`): if no rerank score clears the floor, it falls back to pure RRF and fires `onFallback` — added after a broken zerank GGUF silently emitted ~1e-11 scores that collapsed to RRF invisibly. Intent classifier (`intent.ts`, WHY/WHEN/ENTITY/WHAT via regex + optional LLM) routes to graph structures; graph traversal is beam-search + a personalized-PageRank variant (`graph-traversal.ts`). 5+ distinct retrieval modes (search / vsearch / query / intent_search / query_plan / memory_retrieve auto-router) plus `timeline`, `find_similar`, `find_causal_links`.

**Composite scoring** (`memory.ts:216-394`, SAME-derived): `0.5·search + 0.25·recency + 0.25·confidence` (recency weight rises to 0.70 on detected recency intent), then multiplied by: quality multiplier (0.7–1.3×), length normalization (log penalty on verbose docs), revision/duplicate frequency boost (capped +10%, revisions weighted 2× duplicates), a canonical-memory multiplier, `+0.3` additive pin boost, and a **co-activation boost** (top-quartile results boost their frequent co-retrieval partners, capped +15%) — a Hebbian-flavored analog to Somnigraph's PMI edges but applied at score time, not as graph edges.

### Consolidation / Processing
A background **consolidation worker** (`consolidation.ts`) with two lanes (light / heavy, lease-based). Phase 2: dedup-merge of observations gated by a **name-aware dual-threshold merge-safety gate** (see Worth Stealing #1). Phase 3: **deductive synthesis** — combines related observations into new "deductive" conclusions, wrapped in three **anti-contamination guardrails** (`deductive-guardrails.ts`): evidence sentence-filtering by lexical overlap, relation-context injection from `memory_relations`, and a **contamination scan** that hard-rejects a synthesized conclusion mentioning any entity/anchor present in the broader candidate pool but *absent* from the cited sources. Contradiction handling (`merge-guards.ts`): LLM-first classifier with a **deterministic heuristic fallback** (negation asymmetry + number/date-set mismatch), confidence-thresholded (`CLAWMEM_CONTRADICTION_MIN_CONFIDENCE=0.5`), policy `link` (default, both rows stay active + backlink) or `supersede` (old row → inactive).

### Lifecycle Management
Content-type half-lives (handoff 30d, conversation 45d, problem/milestone/note 60d, research 90d, project 120d, decision/preference/hub/antipattern ∞). Attention decay: non-durable types lose 5% confidence/week without access; access reinforcement extends half-life up to 3×. Contradiction supersession, `snoozed_until` temporary suppression, `memory_forget` (search → deactivate closest match with audit trail), automatic stale-embedding cleanup. Feedback: a Stop-hook (`hooks/feedback-loop.ts` + `recall-attribution.ts`) detects which *injected* notes the assistant actually referenced in its reply and boosts their access counts — **implicit** feedback, no explicit rating and no learned model.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Full BM25+vector+RRF+rerank hybrid | `search-utils.ts` reciprocalRankFusion + blendRerank; zerank Docker sidecar | Validated (code present) |
| Contradiction detection at merge time | `merge-guards.ts` LLM + heuristic; `applyContradictoryConsolidation` | Validated |
| Name-aware merge safety prevents "Dan"/"Dad" collisions | `text-similarity.ts:290 passesMergeSafety`, hard-reject on differing anchors | Validated, genuinely novel guard |
| SAME composite scoring w/ half-lives + co-activation | `memory.ts:216-394` | Validated |
| Local/offline, no API keys | GGUF observer + embed via llama.cpp; cloud optional | Validated |
| Retrieval **quality** (recall/NDCG/QA accuracy) | **None published** | Unvalidated — no benchmark at all |
| Learned/feedback-trained reranker | zerank-2-seq is a **pretrained** cross-encoder; feedback loop is implicit access-count boost only | Not present as claimed elsewhere |

---

## Relevance to Somnigraph

### What ClawMem does that Somnigraph doesn't
- **Write-path quality + entity gating** (observer model, quality_score, canonical entity resolution). Somnigraph's `tools.py remember()` writes whatever the caller passes — no extraction, no quality scoring, no entity resolution. This is Somnigraph's biggest structural gap (corroborated by the Phase 18 write-path-discipline finding).
- **Subject-anchor merge guard** on the destructive dedup/supersede path (`passesMergeSafety`). Somnigraph's `remember()` dedupes at 0.9 cosine and `sleep_nrem.py` merges/archives without a proper-noun subject check.
- **Deterministic contradiction heuristics** as an offline fallback to the LLM classifier. Somnigraph's NREM contradiction classification is LLM-only.
- **Anti-contamination scan** on synthesized conclusions — a hallucination guard Somnigraph's REM gestalt/synthesis lacks.
- **Real-time graph** (entities + SPO triples at write time via hooks) vs Somnigraph's sleep-time-only edge detection.
- Intent-routed multi-graph retrieval and a `timeline` temporal-neighborhood tool.

### What Somnigraph does better
- **Retrieval is measured.** Somnigraph has 85.1% LoCoMo QA, NDCG=0.7958, r=0.70 GT correlation. ClawMem publishes *no* retrieval or QA numbers — every capability is presence-audited, not quality-audited.
- **Learned reranker** (`reranker.py`, 26-feature LightGBM trained on 1032 real queries) vs ClawMem's off-the-shelf pretrained cross-encoder with no training on user data.
- **Explicit feedback loop** with utility+durability ratings and UCB exploration (`tools.py`) vs ClawMem's implicit access-count boost.
- **Bayesian-optimized fusion** (RRF k=14 tuned) vs ClawMem's default k=60.
- **Graph-conditioned retrieval via PPR + betweenness as a reranker feature** — deeper graph integration than ClawMem's score-time co-activation nudge.

---

## Worth Stealing (ranked)

### 1. Subject-anchor guard on the destructive merge/supersede path (Low–Medium)
**What**: Before merging/superseding a near-duplicate, extract proper-noun anchors from both sides (entity IDs when available, else capitalized-token lexical fallback, `text-similarity.ts:121/236/290`). If the anchor sets *materially differ* (empty intersection, or ≤50% of the smaller set shared), **hard-reject the merge regardless of text/vector similarity**. Both-empty → strictest similarity threshold.
**Why**: Somnigraph's `remember()` dedupes at 0.9 cosine and `sleep_nrem.py` merges without any subject check. Two memories that share vocabulary but refer to different subjects ("Dan visited Paris" / "Dad visited Paris", "alice+auth-service" / "bob+auth-service") can collide and destroy the minority. Gate adoption on a cheap offline measurement: replay the memory table, count how often the 0.9-cosine dedup fires between differing-anchor pairs — if it's ~never, skip; if it collides, this is a cheap destructive-merge safety net.
**How**: Port `extractSubjectAnchorsLexical` + `anchorSetsMateriallyDiffer` (~40 LOC, no deps) into a helper; call it in the `tools.py` remember dedup branch and in `sleep_nrem.py`'s merge/archive decision before deletion. char-3-gram cosine (`normalizedCosine3Gram`) is their tighter paraphrase check vs Jaccard.

### 2. Anti-contamination scan for synthesized/gestalt memories (Medium)
**What**: When an LLM generates a synthesized memory (deductive conclusion / gestalt), extract the entity/anchor set of the *output* and reject it if it names any entity that appears in the broader candidate pool but NOT in the cited source memories — the LLM imported content it shouldn't have (`deductive-guardrails.ts scanConclusionContamination`).
**Why**: Somnigraph's REM synthesis and gestalt-layer generation trust the LLM's output. This is a deterministic hallucination guard that catches cross-memory bleed without a second LLM call.
**How**: Add a post-generation check in `sleep_rem.py` (and any gestalt writer): anchor(output) ⊆ anchor(cited sources) ∪ query, else drop/flag. Reuses the anchor extractor from #1.

### 3. Deterministic contradiction heuristics as an LLM fallback (Low)
**What**: Cheap deterministic contradiction signal — **negation asymmetry** (one side has an explicit negation token, the other doesn't → conf 0.6) and **number/date-set mismatch** (both cite numbers, zero shared → conf 0.5) — used whenever the LLM classifier returns null (cooldown/timeout/malformed), plus a tunable confidence floor (`merge-guards.ts:92`).
**Why**: Somnigraph's NREM contradiction classification is LLM-only; a deterministic floor makes the pass robust to LLM outages and offline-testable, and gives a confidence knob for the merge/supersede decision.
**How**: ~30 LOC into `sleep_nrem.py`'s contradiction step as a fallback path; feed the existing negation/number regexes.

### 4. Reranker degenerate-floor fallback (Low)
**What**: Treat the reranker as unusable and fall back to RRF *visibly* (log `onFallback`) if no rerank score clears a small floor (`1e-4`), not just if the model is missing (`search-utils.ts:159`).
**Why**: Somnigraph currently runs on the RRF fallback because the live reranker artifact is missing (per project notes). A "present but degenerate" model (all-near-zero scores) is a distinct failure mode from "absent" — the floor catches a silently-broken reranker instead of trusting garbage scores at weight 0.9.
**How**: In `reranker.py`/`scoring.py`, add a floor check on predicted scores before blending; emit a warning and drop to RRF order when it trips.

---

## Not Useful For Us

### Multi-framework packaging / hooks-vs-MCP dual integration
ClawMem's breadth (OpenClaw plugin, Hermes MemoryProvider, 31 MCP tools, lifecycle hooks) is integration surface for its target ecosystems; Somnigraph is single-user MCP and doesn't need it.

### Content-type half-life table
Somnigraph already has per-category exponential decay with configurable half-lives, floors, and reheat-on-access — ClawMem's fixed content-type table is a coarser version of what `db.py`/decay already do.

### Intent-classified multi-graph routing
Somnigraph's PPR expansion over typed edges already conditions retrieval on graph structure; a WHY/WHEN/ENTITY regex router is a lighter, less-principled variant.

---

## Connections

- **Write-path discipline**: reinforces the Phase 18 sweep finding (ByteRover, MemPalace, agentmemory) that the LoCoMo/LME leaders win on write-time quality, not retrieval. ClawMem is the same shape — observer-model extraction + entity resolution + quality scoring on ingest — though unlike those it publishes no benchmark to prove it helps.
- **Merge/supersede safety**: convergent with the supersession patterns noted across memv/memos-style systems; ClawMem's contribution is the *anchor-difference hard-reject*, a sharper destructive-merge guard than similarity-threshold-only approaches.
- **Upstream lineage**: ClawMem adapts Thoth's `dream_cycle`/`memory_tool` (contradiction + deductive synthesis) and A-MEM's self-evolving notes — cross-reference any existing Thoth / A-MEM analyses in `research/sources/`.
- **Cross-encoder reranker**: same off-the-shelf-reranker choice as several profiled systems, contrasting Somnigraph's learned-on-user-data LightGBM.

---

## Summary Assessment

ClawMem is a competently engineered, fully-local TypeScript/Bun memory server whose real contribution is **integration breadth plus a set of small, sharp safety guards around the write/consolidation path** — not a novel retrieval engine (that's ported QMD) or novel scoring (ported SAME) or novel consolidation (ported Thoth). The evidence file is accurate and thorough on *capability presence* and correctly upgrades several mis-scored cells (hybrid, conflict, timeline, dedup, schema-field counts) and correctly downgrades `singleBinary` (it needs the Bun runtime). Its one blind spot: it never notes that **ClawMem publishes zero retrieval-quality or end-to-end QA numbers**, and that its "reranker" is a pretrained cross-encoder with no feedback training — so despite feature-parity checkmarks against Somnigraph, there is nothing comparable to our 85.1 LoCoMo QA / NDCG 0.7958 / r=0.70 GT correlation. Feature presence ≠ measured quality.

The single most valuable thing to take is the **subject-anchor merge guard** (Worth Stealing #1): a cheap, dependency-free hard-reject that prevents a destructive 0.9-cosine dedup from collapsing two memories with the same vocabulary but different subjects — a real risk in `tools.py` and `sleep_nrem.py` that our similarity-only dedup doesn't defend against. Gate it on a one-off offline collision measurement first, per honest-accounting. Secondary picks (anti-contamination scan for gestalts, deterministic contradiction heuristics, reranker degenerate-floor) are all small, additive robustness improvements to the sleep pipeline and reranker fallback, none load-bearing.

What's overhyped: the feature-checkmark framing invites reading ClawMem as at-parity-or-better than benchmarked systems. It is a solid engineering artifact with good instincts on write-path hygiene, but its quality is unmeasured, and much of its machinery is adapted from named upstream projects rather than original.
