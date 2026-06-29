# ByteRover — BM25-only SOTA via write-path quality, not retrieval sophistication

*Generated 2026-06-28 by Sonnet agent reading local clone + paper*

---

## Paper Overview

**Paper**: Andy Nguyen et al. (ByteRover team). "ByteRover: Agent-Native Memory Through LLM-Curated Hierarchical Context." arXiv:2604.01599. 2026. (Formerly titled "Cipher".)
**Authors**: ByteRover / Campfire team.
**Code**: https://github.com/campfirein/byterover-cli (Elastic 2.0, v3.16.1)

**Problem addressed**: Agent memory systems typically append raw facts and rely on retrieval sophistication (vector search, reranking, graph expansion) to compensate for low write quality. ByteRover challenges this assumption: if the write process is structured and LLM-mediated, can a simple retrieval stack reach SOTA?

**Core approach**: A filesystem-based context tree of `<bv-topic>` HTML files organized as Domain > Topic > Subtopic. The calling agent authors each entry at capture time (not the daemon), using typed elements (`<bv-decision>`, `<bv-rule>`, `<bv-snippet>`), explicit cross-references via `related=` attributes, and structured narrative sections. Retrieval is MiniSearch BM25 — no vectors, no embeddings, no external services. Importance/recency/maturity signals provide post-BM25 reranking. The bet: high write quality makes retrieval a solved problem.

**Key claims**:
- 96.1% on LoCoMo (arXiv Table 3, production codebase, LLM-as-Judge).
- 92.2% on LoCoMo (v2.0 blog run: curate Gemini Flash, judge Gemini Pro/Flash).
- 92.8% on LongMemEval-S (500 questions, 23,867 docs).
- BM25-only retrieval, zero vector infrastructure.

---

## Architecture

### Storage & Schema

No SQLite, no vector database. All knowledge lives in `.brv/context-tree/` as human-readable HTML files. Each file is a `<bv-topic>` with four structured sections: Relations (cross-refs via `related=@domain/topic`), Raw Concept (provenance: task, changes, files, timestamp), Narrative (structure, rules, examples), Snippets (code, formulas). Per-file lifecycle signals (importance [0,100], maturity draft/validated/core, recency, accessCount, updateCount) live in a sidecar `RuntimeSignalStore`, not in the file itself. Directory `_index.md` files act as topic-group summaries.

### Memory Types

No episodic/semantic/procedural taxonomy. All entries are "knowledge entries" differentiated by maturity tier only: `draft` (importance < 65), `validated` (65-84), `core` (≥85). Maturity uses hysteresis thresholds to prevent oscillation (promote at 65/85, demote at 35/60). Core entries are never pruned. The hierarchy Domain > Topic > Subtopic is structural, not a memory-type distinction.

### Write Path

The deepest design choice: ByteRover delegates authorship to the calling agent's LLM. `brv curate "<intent>"` returns a `generate-html` prompt; the agent authors the `<bv-topic>` HTML and sends it back. ByteRover validates, detects structural loss via `conflict-detector.ts`, auto-merges arrays (union-dedup) to prevent data loss, and writes. Up to 3 correction rounds (MAX_ATTEMPTS=4). The daemon never calls an LLM in tool mode. The result is a knowledge base where each entry has been reflected on: where it fits in the hierarchy, how to structure it with typed elements, and which existing entries it relates to via `related=` annotations.

This contrasts with Somnigraph's `remember()`: flat content string, category/priority metadata, with sleep-deferred edge discovery. ByteRover forces structure and cross-referencing at write time; Somnigraph defers it to sleep.

### Retrieval

**BM25-only. Confirmed from code. Zero vector search in the codebase.**

`search-knowledge-service.ts` (the sole retrieval implementation) uses MiniSearch configured at lines 168-177:
```
fields: ['title', 'content', 'path']
boost: { title: 3, path: 1.5 }
fuzzy: 0.2, prefix: true
```
No embedding calls, no vector index, no `sqlite-vec`, no external embedding service appear anywhere in the retrieval path.

Post-BM25 compound score (`memory-scoring.ts` lines 73-79):
```
score = 0.6 × BM25_normalized + 0.2 × (importance/100) + 0.2 × recency
score *= TIER_BOOST[maturity]   // core=1.15, validated=1.0, draft=0.85
```
where `BM25_normalized = rawScore / (1 + rawScore)` (lines 85-87).

OOD gate (`search-knowledge-service.ts` lines 65-71): if top normalized score < 0.45 OR any significant query term (≥4 chars) is completely unmatched and score < 0.85, the query is flagged out-of-domain. Returns explicit OOD signal rather than noise.

Additional signals: BM25 score propagation up the symbol tree (factor 0.55 per level, lines 111-162); bidirectional `related=` reference index (backlink count returned per result); symbolic path bypass for direct path lookups; AND-first multi-word with OR fallback.

A 5-tier `QueryExecutor` adds LLM synthesis on top (Tier 0: exact cache, Tier 1: fuzzy cache, Tier 2: direct BM25, Tier 3: single LLM call, Tier 4: full agentic loop). The benchmarks use Tier 3/4. `brv search` is pure Tier 2 (BM25, no LLM).

### Consolidation

`brv dream`: three-phase BM25-similarity-based consolidation — scan (enumerate near-duplicate pairs, low-importance entries, cross-link opportunities), act (agent LLM merges/prunes/synthesizes), finalize. Human-triggered (not autonomous). Bit-exact undo including sidecar signal restoration. Dream-lock prevents concurrent runs.

Importance decay runs automatically: `importance *= 0.995^days` (~78% after 50 days), `recency = exp(-days/30)` (~21-day half-life).

### Lifecycle Management

Prune candidates surfaced when importance < 35 or stale > 60/120 days. Core entries never pruned. Maturity tiers with hysteresis prevent oscillation. Git-like version control (`brv vc`, isomorphic-git) for whole-tree snapshots and rollback. No per-entry version history. No temporal validity (valid_from/valid_until). No explicit decay schedule — runs during dream consolidation, not autonomously.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| 96.1% LoCoMo (SOTA) | arXiv Table 3, production codebase, LLM-as-Judge | **Plausible, self-reported.** Judging model unspecified in paper. Peer-reviewed venue. Same production codebase, no held-out prototype. |
| 92.2% LoCoMo (v2.0 blog) | Public methodology: curate Gemini Flash, judge Gemini Pro/Flash | **Verified independent run, lower bar.** 3.9pp gap vs paper is partially evaluation artifact (weaker judging model in blog run). |
| 92.8% LongMemEval-S | arXiv Table 4, 500 questions | **Moderate.** Self-reported. No per-category breakdown in prior analysis. |
| BM25-only achieves these scores | Code: `search-knowledge-service.ts` imports only MiniSearch; no embedding imports anywhere in retrieval | **Confirmed from code.** This is the headline finding and it holds. |
| Write quality compensates for retrieval simplicity | Architecture argument + benchmark numbers | **Plausible but not directly ablated.** No ablation comparing BM25-only vs BM25+vector on same write-quality entries. |

**96.1% vs 92.2% reconciliation**: The gap reflects at minimum (a) a stronger judging model in the paper run and (b) codebase improvements between v2.0 and v3.x. Cross-vendor comparison is not apples-to-apples: MemMachine's 91.69% uses gpt-4.1-mini as judge; ByteRover's 96.1% uses an unspecified judge. Numbers are directionally informative, not directly comparable.

---

## Relevance to Somnigraph

### What ByteRover does that Somnigraph doesn't

1. **Write-path quality gating.** Somnigraph's `remember()` (tools.py) accepts flat content strings. ByteRover forces the writing agent to reflect on hierarchy placement, typed element structure, and cross-references before the entry is written. Somnigraph's write side is the known gap (per STEWARDSHIP.md). ByteRover is the sharpest evidence in the corpus that this matters.

2. **Structural loss detection on update.** `conflict-detector.ts` compares proposed vs. existing content field-by-field; `conflict-resolver.ts` union-merges arrays to prevent data loss. Somnigraph's `remember()` deduplicates by embedding similarity but has no mechanism to detect that an update would shrink an existing memory's content.

3. **Out-of-domain detection.** ByteRover's dual OOD gate (absolute score floor 0.45 + significant-term unmatched check) returns an explicit "not in knowledge base" signal. Somnigraph's `recall()` (tools.py) returns whatever it finds; no low-confidence flag. This risks injecting noise into LLM context on weak matches.

4. **Human-readable, navigable knowledge base.** Domain > Topic > Subtopic hierarchy, overview mode, per-topic HTML files. Somnigraph's SQLite schema is opaque without tooling.

5. **Git-like version control.** Whole-tree snapshots with branch/merge/rollback. No equivalent in Somnigraph.

### What Somnigraph does better

1. **Multi-channel retrieval.** Three-channel RRF (vector via sqlite-vec + FTS5 BM25 + theme channel, k=14, Bayesian-optimized). ByteRover is BM25-only; its high scores argue write quality can substitute, but Somnigraph's retrieval catches semantic variants BM25 misses — confirmed by the 88% vocabulary gap finding in `docs/multihop-failure-analysis.md`.

2. **Learned reranker.** 31-feature LightGBM reranker (V5+3b, NDCG=0.8954, trained on 1885 real-data queries with adversarial probing). ByteRover's compound score is a hand-tuned 3-weight formula. Somnigraph's reranker (reranker.py) captures interaction effects invisible to any formula.

3. **PPR graph expansion.** Sleep-built typed edges (supports/contradicts/evolves/revision/derivation) enable personalized PageRank expansion and betweenness centrality as a reranker feature (scoring.py). ByteRover's `related=` reference index is bidirectional and informs backlink counts but does not drive multi-hop graph traversal during retrieval.

4. **Explicit feedback loop.** Per-query utility ratings (0-1 float + durability), EWMA aggregation, UCB exploration bonus, Hebbian PMI co-retrieval boost. ByteRover tracks accessCount and importance bumps but has no explicit per-query feedback signal and no measured correlation with ground truth. Somnigraph's feedback gradient is the selection mechanism, not bookkeeping.

5. **Offline LLM-mediated consolidation.** Three-phase NREM/REM/orchestrator: pairwise relationship classification, typed edge creation, contradiction detection, temporal evolution, gap analysis. ByteRover's dream is human-triggered and BM25-similarity-based (structural candidates only); edge detection in Somnigraph is LLM-mediated and semantic. Temporal contradiction detection (valid_from/valid_until) has no equivalent in ByteRover.

6. **Typed memory schema.** Category (episodic/semantic/procedural/reflection/meta), priority 1-10, themes array, valid_from/valid_until, per-category decay rates. ByteRover's entries are undifferentiated knowledge entries with a single maturity axis.

---

## Worth Stealing (ranked)

### 1. Out-of-domain detection (Low effort)
**What**: Add a dual-signal OOD gate to `recall()`: flag results where the top normalized BM25 score < threshold OR significant query terms (≥4 chars) are completely unmatched in top-N results.
**Why**: Somnigraph currently returns whatever it finds. Weak matches injected into LLM context are a real failure mode — the model hallucinates rather than admitting the gap. An explicit `ood: true` flag would let the caller (or the agent) signal "this query is outside stored knowledge."
**How**: Add to `fts.py` after BM25 scoring: normalize top score, check query term coverage. Return `ood` flag alongside results in `impl_recall()` in tools.py. No new dependencies.

### 2. Structural loss detection on update (Low effort)
**What**: Before writing an update to an existing memory, compare proposed summary/content against existing to detect field shrinkage. Auto-merge rather than overwrite if loss detected.
**Why**: Somnigraph's `remember()` can silently shrink a memory if a near-duplicate update passes the 0.9 similarity threshold but drops content. ByteRover's union-merge pattern prevents this class of data loss. Critical for memories that accumulate corrections over multiple sessions.
**How**: In `impl_remember()` (tools.py), when update path is taken: parse existing and proposed into component fields, count missing items, union-merge arrays/lists if loss detected. Pure structural comparison, no LLM required.

### 3. LLM-as-author at write time, lightweight version (Medium effort)
**What**: At `remember()` time, surface top-3 existing recall candidates and prompt the model to create `link()` calls explicitly — which existing memories does this new one relate to, and of what type (supports/contradicts/evolves)?
**Why**: ByteRover achieves SOTA BM25-only by building dense cross-references at write time, not sleeping them in later. Somnigraph's edge detection runs in sleep (sleep_nrem.py), which means new memories sit unlinked until the next sleep cycle. A lightweight write-time edge prompt would increase link density immediately. ByteRover demonstrates the pattern works; Somnigraph's graph infrastructure (typed edges, PPR) would make better use of those links than ByteRover's reference index does.
**How**: In `impl_remember()` (tools.py), after embedding: recall top-3 candidates (internal=True), include in a brief prompt asking for explicit edge types. Call `impl_link()` for each confirmed relation. Add latency guard: skip if recall returns no high-confidence candidates. This is a prompt engineering change + tool orchestration, no schema changes.

### 4. Maturity tier hysteresis (Low effort)
**What**: Promote/demote Somnigraph's priority tiers using importance thresholds with hysteresis gaps rather than sharp thresholds, preventing oscillation.
**Why**: ByteRover's promote-at-65/85, demote-at-35/60 pattern (memory-scoring.ts lines 119-142) prevents rapid tier oscillation that could flip retrieval behavior on slight importance changes. Somnigraph's priority is set at write time and rarely updated automatically.
**How**: Add hysteresis logic to the `consolidate()` step in tools.py or sleep pipeline: if a memory's effective importance (feedback-adjusted) crosses a threshold with hysteresis gap, update priority. Minor addition to NREM processing.

---

## Not Useful For Us

### Context Tree HTML format
`<bv-topic>` with typed child elements is designed for large, human-navigable codebases with thousands of entries and team collaboration. At our ~300-memory single-user scale, file-per-entry HTML with daemon/agent-pool/Socket.IO transport is architectural overhead without payoff. SQLite + free-text content is more appropriate.

### Git-like version control
Solves team collaboration (branch, merge, push/pull) we don't have. Somnigraph's SQLite journal provides append-only history at lower cost.

### 5-tier progressive retrieval (QueryExecutor)
Tiers 0-2 are latency caching optimizations (Somnigraph's SQLite queries are already fast). Tiers 3-4 are LLM synthesis layers handled separately in Somnigraph's MCP tool pattern.

### Swarm federation
We have one memory store. Multi-provider federation across obsidian, memory-wiki, etc. is irrelevant for a single embedded MCP server.

### Full dream consolidation transplant
ByteRover's dream is BM25-similarity-based structural candidate detection. Somnigraph's sleep is LLM-mediated semantic classification. The dream approach is weaker for Somnigraph's use case; the existing three-phase sleep pipeline is the right architecture.

---

## Connections

**memmachine.md**: Convergent on the write-path-is-the-lever thesis from opposite directions. MemMachine bets on raw episode preservation (no LLM extraction error); ByteRover bets on LLM-mediated structuring (no raw noise). Both reach SOTA without Somnigraph-class retrieval sophistication. The convergence is strong evidence that write quality is the underinvested axis.

**simplemem.md**: SimpleMem also argues for write-time synthesis over retrieval-time complexity. ByteRover is the large-scale empirical proof of that thesis; SimpleMem is the theoretical argument.

**evermemos.md**: EverMemos uses structured note templates at write time to improve retrieval. ByteRover's `<bv-topic>` HTML with typed elements is a more rigorous version of the same insight: forcing structure at capture time reduces retrieval ambiguity.

**mem0-paper.md**: ByteRover's flat-content write path would be analogous to Mem0's extraction approach — the difference is ByteRover makes the *structure and organization* LLM-authored, not the extraction. Both contrast with MemMachine's raw-episode approach.

**perma.md**: ByteRover's maturity tier system (draft/validated/core) with importance-based hysteresis is the closest analog in the corpus to PERMA's preference-state maintenance requirements. ByteRover doesn't evaluate on PERMA but its lifecycle management is better suited than most systems surveyed there.

The write-path-is-the-lever convergence (ByteRover + MemMachine + SimpleMem + EverMemos) is the strongest multi-system finding in the corpus: retrieval sophistication is not the bottleneck when write quality is low.

---

## Summary Assessment

ByteRover's core contribution is empirical: BM25-only retrieval achieves competitive-to-SOTA LoCoMo scores when the write process is high-quality. The prior analysis (agent-output-byterover.md) confirmed this from the code — `search-knowledge-service.ts` contains no embedding calls, no vector index, no external services. The 96.1% figure is the paper's number with an unspecified judge; the blog's independently-run 92.2% is the more conservative bound. Cross-vendor comparisons are not apples-to-apples, but the directional argument holds: ByteRover with BM25 beats most systems that use vector search.

The most important thing for Somnigraph to take from ByteRover is the write-path framing. Somnigraph's retrieval is strong (31-feature learned reranker, three-channel RRF, PPR graph expansion, measured feedback loop). Somnigraph's write side is a flat content string fed to `remember()` with sleep-deferred edge discovery. ByteRover demonstrates that structured LLM-authored entries with explicit cross-references at capture time outperform sophisticated retrieval over flat content. The gap is not closing on the retrieval side; it needs to close on the write side. The three worth-stealing items above are all write-path or retrieval-quality-guard improvements, not retrieval mechanism additions.

What ByteRover does not have: semantic retrieval (vector search catches vocabulary-gap queries that BM25 misses, and Somnigraph's 88% vocabulary-gap ceiling in multi-hop confirms this), a feedback loop with measured GT correlation, graph-conditioned retrieval, or autonomous consolidation. For Somnigraph's use case (persistent memory across long sessions with complex multi-hop dependencies), retrieval sophistication remains load-bearing. The lesson from ByteRover is additive: improve the write path alongside the retrieval stack, not instead of it.
