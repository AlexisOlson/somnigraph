# second-brain - Serverless single-user memory on Cloudflare (D1 + Vectorize), now with RRF hybrid retrieval, a contradiction win/loss survival signal, and nightly LLM tag-digest compression

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

Second Brain (`rahilp/second-brain-cloudflare`, MIT, ~91 stars) is a personal memory layer that runs entirely inside a user's own Cloudflare account: Workers (edge functions) + D1 (serverless SQLite) + Vectorize (vector DB) + Workers AI (embeddings + LLM). One-click deploy, 6 MCP tools, a built-in web dashboard. The entire backend is a single 2,293-line `src/index.ts`. No external services, no API keys — bge-small embeddings and a Llama-4-Scout LLM both run on Workers AI.

The important finding of this read: **the code has evolved substantially past the carsteneu audit** (dated 2026-05-29, v1.5.1). The current HEAD adds always-on hybrid retrieval, a contradiction win/loss tally, and a nightly compression cron — none of which the evidence file reflects.

### Storage & Schema
Single D1 table `entries` (`db/schema.sql`): `id, content, tags` (JSON array text), `source, created_at` (unix ms), `vector_ids` (JSON array), `recall_count, importance_score, contradiction_wins, contradiction_losses`. Two indexes (created_at DESC, source). Vectorize holds 384-dim bge-small-en-v1.5 chunk vectors (cosine), each pointing back to a parent entry via metadata. Flat text model — no entities, no graph, no edge table. Lifecycle "status" (canonical/draft/deprecated) and "kind" (episodic/semantic) live as reserved `status:`/`kind:` tags, not columns.

### Memory Types
Loosely typed via tags: user topic tags (personal/work/task/idea/context) plus a classifier-assigned `kind:episodic|semantic` and `status:canonical|draft|deprecated`. System-generated kinds: `synthesized` (digests), `auto-pattern` (derived themes), `rolled-up` (compressed sources). No procedural/reflection/meta distinction.

### Write Path (`captureEntry`, line 1392)
1. Extract inline `#hashtags` → tags; lowercase and dedupe.
2. `checkDuplicateAndContradiction` (line 303): embed a sample, query top-5 Vectorize. Three similarity bands: **≥0.95 → blocked** (nothing stored); **0.85-0.95 → "flagged"**, triggers a single combined LLM call ("smart merge") that returns one of `contradiction / replace / merge / keep_both`; **0.45-0.85 → contradiction-only** LLM check.
3. Smart-merge `replace`/`merge` mutate the target entry in place (re-embed, delete stale vectors) instead of inserting — but **high-importance (≥4) or canonical targets are protected** from silent overwrite (line 1427).
4. Contradiction resolution (line 1471): if the loser is `canonical`, the incumbent survives and the new entry is demoted to `draft`; otherwise the incumbent is deprecated (vectors deleted, row kept for audit). Either way `contradiction_wins`/`contradiction_losses` counters are incremented on the survivor/loser.
5. `scheduleClassifyAndTag` (async, `ctx.waitUntil`): a Llama call assigns importance 1-5, canonical bool, and kind. Non-blocking.
6. Chunking: >1,600 chars split at sentence/newline boundaries with 200-char overlap; each chunk a separate vector.

### Retrieval (`recallEntries`, line 1186)
**Always-on hybrid RRF** — this is the biggest divergence from the audit. In parallel: (a) dense Vectorize query (topK = 3×requested, capped 50), and (b) `keywordSearch` (line 1124) = a SQL `content LIKE %token%` scan over tokenized, stop-word-filtered query terms, bounded to 100 rows. `fuseDenseAndKeyword` (line 1152) merges them via `rrfFuse` (RRF_K=60): dense contributes `1/(k+rank)`; each keyword hit contributes `weight/(k+rank)` where **weight = number of distinct query tokens the entry matched** (a crude IDF-free relevance proxy). On the default path keyword can surface an entry the dense top-K missed entirely; on the tag-filtered path it is a re-ranking signal only.

Fused candidates then go through `rerankWithTimeDecay` (line 487), a hand-tuned **multiplicative formula**:
```
score = rrf_score
      × min(1.0, exp(-age/halfLife) × (1 + log1p(recall_count)))   # recency×frequency, capped
      × appendPenalty(0.2 if short update chunk)
      × rolledUpPenalty(0.4 if rolled-up)
      × importanceMultiplier(0.8..1.2, adjusted by net contradiction wins-losses)
      × tagBoost(1 + 0.15×overlap, max 1.5)                        # query-tag ↔ entry-tag overlap
```
Half-life is tag-dependent (task 7d, work 90d, context 180d, default 30d). Final scores normalized so top match = 1.0. Dedup by parent ID, slice to topK. Then multi-match results get a Llama-synthesized 2-4 sentence "insight" (`synthesizeInsight`, grounded-only prompt), and ≥5 results trigger async `derivePattern`.

### Consolidation / Processing
Two real offline mechanisms (the audit missed both):
- **Nightly cron** (`wrangler.jsonc` `crons: ["0 1 * * *"]` → `runNightlyCompression`, line 1059): groups compression-eligible entries by topic tag (count >10), and for each tag `compressTag` (line 994) fetches up to 50 eligible entries, calls `synthesizeDigest` (a Llama "state of this area" paragraph), stores it as a `synthesized` entry, and marks each source `rolled-up` (which carries a 0.4 recall penalty). **Compression-eligibility gate** (`compressionEligibilitySql`, line 56) is a clean multi-condition protective filter: `importance < 4 AND (recall_count = 0 OR (recall_count < 2 AND older than 60 days)) AND contradiction_wins = 0`. Battle-tested and useful memories are exempt.
- **`derivePattern`** (line 907, inline during recall, ≤1 per 48h): samples 20 recalled entries and asks the LLM for a genuine cross-memory pattern ("You tend to…", "There's a recurring…"), stored as an `auto-pattern` memory. A lightweight persona/theme extractor.

### Lifecycle Management
Time-decay is **retrieval-time only** — entries are never auto-deleted. Explicit `forget` (deletes row + all vectors). `deprecateEntry` (keeps row for audit, deletes vectors, tags `status:deprecated`). `update` replaces content in place. Contradiction losers are deprecated or demoted to draft. No versioning/time-travel.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Hybrid keyword+vector retrieval | `rrfFuse`/`fuseDenseAndKeyword`/`keywordSearch` in code, `rrf-fuse.test.ts` | Validated in code — but keyword side is `LIKE` scan + token-count weight, **not BM25/FTS** |
| Time-decay + smart-merge (table note) | `rerankWithTimeDecay`, `checkDuplicateAndContradiction` | Validated; formula is richer than the note (6 multiplicative factors) |
| Contradiction detection + survival tracking | win/loss columns feed `importanceMultiplier` | Validated; distinct implicit-feedback idea |
| Nightly digest compression | cron + `runNightlyCompression`/`compressTag` | Validated; LLM-mediated, tag-clustered |
| Benchmark performance | none published | No LoCoMo/LME/PERMA numbers — **not comparable to our 85.1 LoCoMo QA** |
| "Auto-extraction" | instruction-file driven only (CLAUDE_INSTRUCTIONS.md) | Unvalidated as code — relies on the client model calling `remember` |

---

## Relevance to Somnigraph

### What second-brain does that Somnigraph doesn't
- **Contradiction win/loss survival counter feeding the reranker.** Somnigraph's `sleep_nrem.py` detects `contradicts` edges but does not keep a running "this memory keeps winning/losing contradictions" tally that adjusts its rank. second-brain's `contradiction_wins/losses` → `importanceMultiplier` is a cheap durability signal earned by surviving conflicts, orthogonal to explicit utility feedback.
- **Write-time contradiction/merge gating with canonical protection.** Somnigraph resolves contradictions offline during sleep; second-brain does it synchronously at capture (block ≥0.95, smart-merge 0.85-0.95, contradiction-check 0.45-0.85) and refuses to silently overwrite importance≥4/canonical entries. This is exactly the write-path quality gating Somnigraph lacks (`tools.py` remember has none).
- **Tag-clustered digest compression with an explicit protective eligibility gate.** Somnigraph merges pairwise; second-brain compresses a whole topic cluster into one digest and demotes (not deletes) the sources. The eligibility SQL (low-importance AND low-recall AND old AND never-won-a-contradiction) is a tidy archival gate.

### What Somnigraph does better
Nearly everything on quality: real FTS5 BM25 (vs `LIKE` + token count), Bayesian-tuned RRF (k=14 vs hardcoded 60), a **26-feature learned LightGBM reranker** (vs a 6-factor hand-tuned formula — the exact thing Somnigraph's `reranker.py` replaced and measured at +6.17pp NDCG), an **explicit measured feedback loop** (Spearman r=0.70) vs an implicit contradiction tally, a typed graph with PPR expansion and betweenness (`scoring.py`) vs no graph at all, and 85.1% LoCoMo QA vs zero benchmarks. Somnigraph's sleep is a three-phase LLM pipeline (`sleep_nrem.py`/`sleep_rem.py`); second-brain's consolidation is a single nightly digest loop.

---

## Worth Stealing (ranked)

### 1. Contradiction-survival counter as a reranker feature (Low)
**What**: Track per-memory `contradiction_wins`/`losses` and fold a `log1p(net)`-shaped adjustment into rank, so a fact that repeatedly survives conflicts rises and a repeatedly-contradicted one sinks — without touching its stored priority.
**Why**: Somnigraph already *classifies* contradictions during sleep but throws away the outcome as a persistent signal. This is a durability estimate earned by conflict survival, distinct from click-based utility feedback, and it's essentially free once contradiction edges exist.
**How**: Add `contradiction_wins/losses` counters to `db.py`, increment them in `sleep_nrem.py` when a `contradicts` edge is resolved (winner/loser already determined there), and expose net-survival as one more feature in `reranker.py`. It complements, not replaces, the existing feedback loop.

### 2. Multi-condition "compression-eligibility" gate for archival (Low)
**What**: Before archiving/compressing a memory, require ALL of: low importance AND (never recalled, or rarely recalled and old) AND never won a contradiction. Useful, battle-tested, or high-priority memories are structurally exempt.
**Why**: Somnigraph's decay + sleep-archive decisions are driven mostly by decay score; this adds a cheap, legible AND-gate that protects proven-useful memories from consolidation loss. It encodes "recall_count and contradiction survival are proof of value" directly into the archive filter.
**How**: Mirror `compressionEligibilitySql` (line 56) as a guard in `sleep_nrem.py`'s merge/archive candidate selection — recall_count and priority already exist; the contradiction-survival term pairs with idea #1.

---

## Not Useful For Us

- **Cloudflare/serverless stack, web dashboard, iOS/Obsidian/bookmarklet integrations, OAuth.** Delivery-layer choices for a self-hosted multi-client product; irrelevant to Somnigraph's single-user MCP-on-SQLite design.
- **`LIKE`-based keyword search + token-count RRF weight.** Strictly weaker than Somnigraph's FTS5 BM25; adopting it would be a regression.
- **Instruction-file "auto-extraction."** Not a code mechanism — just prompt text telling the client model to call `remember`.

---

## Connections

Convergent with the Phase 18 sweep finding (`ai-memory-comparison.md`, `agentmemory.md`, `byterover`) that **write-path quality, not retrieval sophistication, is where lightweight systems invest** — second-brain has no learned reranker and no benchmarks, yet spends its complexity budget on synchronous dedup/merge/contradiction gating at capture. Its contradiction win/loss survival signal rhymes with the supersession/versioning patterns in `memv.md` and the trust-decay ideas elsewhere, but reframed as a *conflict-survival tally* rather than a version chain. The tag-digest compression is a cruder cousin of Somnigraph's gestalt layer and of the cluster-summarization seen in `memos.md`/MemPalace-style verbatim-plus-summary designs.

---

## Summary Assessment

Second Brain's core contribution is a clean, honest demonstration that a genuinely useful personal memory system fits in one file on free serverless infrastructure. Its real engineering investment is on the **write path** — a three-band synchronous dedup/merge/contradiction gate with canonical protection — and in two small but thoughtful offline touches: a nightly tag-digest compressor and a contradiction win/loss survival signal. None of this is benchmark-driven; there are no published numbers, and the retrieval stack (LIKE-based keyword + hand-tuned multiplicative rerank) is exactly the pre-learned-reranker design Somnigraph already measured itself past.

The single most valuable takeaway for Somnigraph is the **contradiction-survival counter**: Somnigraph already does the expensive part (classifying contradictions during sleep) but discards the outcome, whereas second-brain persists it as a durability signal that shapes future rank. That, plus the legible multi-condition archival-eligibility gate, are two low-effort additive ideas worth a revisit — everything else is either a delivery-layer concern or something Somnigraph does more rigorously.

The sharpest correction to the carsteneu evidence file: it was audited at v1.5.1 (2026-05-29) and marks **hybrid = NO, fulltext = NO**, but the current HEAD ships always-on RRF hybrid retrieval (`rrfFuse`, `fuseDenseAndKeyword`), a contradiction win/loss tally, and a nightly compression cron — none captured in the audit. The scoring formula it lists (`similarity × time_decay × (1+log1p(recall_count))`) is also stale; the live formula has six multiplicative factors. The repo moved; the evidence didn't.
