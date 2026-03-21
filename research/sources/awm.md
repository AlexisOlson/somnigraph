# AgentWorkingMemory (CompleteIdeas/agent-working-memory) — Analysis

*Generated 2026-03-21 by Opus agent reading local clone*

---

## Repo Overview

**Repo**: https://github.com/CompleteIdeas/agent-working-memory
**License**: Apache 2.0
**Language**: TypeScript (ES2022, strict)
**Description**: "Persistent working memory for AI agents"
**Author**: Robert Winter / Complete Ideas

**Problem addressed**: Giving coding agents cross-session project knowledge without manual notes. The system filters for salience at write time, builds associative links between memories, and consolidates offline — positioning itself as a cognitive memory system rather than a vector database with retrieval.

**Core approach**: Single-process Node.js with SQLite + FTS5 + three local ONNX models (MiniLM-L6-v2 for embeddings, ms-marco-MiniLM for cross-encoder reranking, flan-t5-small for query expansion). 10-phase retrieval pipeline, 7-phase sleep consolidation, write-time salience filtering, staging buffer, Hebbian edges, retraction with confidence contamination. All local — no API keys required.

**Maturity**: v0.5.x, actively developed. 13 MCP tools, HTTP API, Claude Code hooks for auto-checkpointing. Tested with internal eval suites (self-test, edge cases, stress test, workday sim, A/B, real-world). No external benchmark validation. Tested at ~300 memories; large-scale behavior is untested.

---

## Architecture

### Design Thesis

Write-time filtering + offline consolidation > store-everything-and-retrieve. AWM's claim is that rejecting 77% of incoming information at write time produces cleaner retrieval paths than post-hoc ranking over a noisy pool. The cognitive science framing (ACT-R, Hebbian learning, CLS, synaptic homeostasis) is genuine — the implementation maps to the theory more consistently than most systems that name-drop these concepts.

### Technical Stack

| Component | Technology |
|-----------|-----------|
| Database | SQLite via better-sqlite3 + FTS5 |
| Embeddings | Xenova/all-MiniLM-L6-v2 (384d, ~23MB, ONNX) |
| Reranker | Xenova/ms-marco-MiniLM-L-6-v2 (cross-encoder, ~23MB) |
| Query expansion | Xenova/flan-t5-small (~78MB) |
| HTTP | Fastify 5 |
| MCP | @modelcontextprotocol/sdk |
| Tests | Vitest 4 |
| Validation | Zod 4 |

Total model footprint: ~124MB, downloaded on first use, cached in `~/.cache/huggingface/`.

### Schema

**engrams** — Individual memories:
- `id` (UUID), `agent_id`, `concept` (title/topic), `content` (body), `event_type`
- `salience` (write-time score), `confidence` (evolves via feedback), `access_count`, `last_access`
- `embedding` (384d float blob)
- `disposition` (active/staging/archived/retracted)
- `supersededBy` (explicit supersession tracking)
- `tags` (string array for entity tagging)
- Task fields: `task_status`, `task_priority`, `blocked_by`

**edges** — Associations:
- `source_id`, `target_id`, `weight`, `edge_type` (hebbian/temporal/bridge/connection/invalidation)
- `confidence`, `lastActivated`

**engram_fts** — FTS5 virtual table on concept + content

**episodes** — Grouping of related memories

**conscious_state** — Checkpoint storage (JSON blob per agent)

### Retrieval Pipeline (10 phases)

| Phase | What it does |
|-------|-------------|
| 0. Query expansion | flan-t5-small synonym generation on query text |
| 1. Embed query | MiniLM-L6-v2 384d embedding (original query, not expanded) |
| 2. Parallel retrieval | FTS5/BM25 (expanded query, limit×3) + all active engrams |
| 3a. Score candidates | BM25 continuous + Jaccard (concept 0.6, content 0.4) + concept exact-match bonus (up to 0.3) |
| 3b. Vector scoring | Z-score normalization with stddev floor 0.10; gate at z > 1.0, map z=1..3 → 0..1 |
| 3c. Composite | `0.6 * textMatch + 0.4 * temporal * relevanceGate + centrality + feedback` × confidence |
| 3.5. Rocchio PRF | Top-3 results feed back terms into BM25 re-search |
| 3.7. Entity-bridge | Boost candidates sharing entity tags with top text-match results; IDF-filtered (>30% frequency tags removed) |
| 4–5. Graph walk | Beam search (width 15, depth 2, hop penalty 0.3) from text-relevant seeds |
| 6. Pool filter | Score ≥ minScore (default 0.01) |
| 7. Cross-encoder rerank | ms-marco-MiniLM scores (query, passage) pairs; adaptive blend weight: `0.3 + 0.4 * (1 - bm25Max)` |
| 8a. Semantic drift penalty | If no candidate has vectorMatch > 0.05, apply 0.5× multiplier (query off-topic) |
| 8b. Entropy gating | If top-5 reranker scores have low variance, abstain (return empty) |
| 8c. Supersession penalty | Superseded memories get 0.15× score |
| 9. Final sort + limit | Top-k with per-phase explanation strings |

**Side effects on retrieval**: Touch access count, push to co-activation buffer, update Hebbian edge weights, log activation event for eval.

### Consolidation Pipeline (7 phases)

| Phase | What it does |
|-------|-------------|
| 1. Replay | Greedy agglomerative clustering by cosine similarity (threshold 0.65), seeded from highest-access memories |
| 2. Strengthen | Reinforce intra-cluster edges; access-weighted signal (more retrieved = stronger consolidation) |
| 3. Bridge | Cross-cluster edges between centroids with moderate similarity (0.25–0.65); bridge weight proportional to inter-cluster cosine |
| 4. Decay | Confidence-modulated half-life: base 7 days, up to 21 days (3× cap) for high-confidence edges |
| 5. Homeostasis | Normalize outgoing edge weight per node to target 10.0; prune below 0.01 |
| 6. Forget | Grace period (7 days), then archive based on access×confidence; 0-access memories archived after 5 consolidation cycles; hard cap 12× base (360 days); delete truly orphaned archived memories after 90 days |
| 6.5. Redundancy prune | Archive semantically similar (>0.85 cosine) low-confidence, low-access duplicates; max 10 per cycle |
| 7. Sweep staging | Promote staged memories with cosine > 0.6 to any active memory; discard if >24h with no resonance |

**Design choice**: No summary nodes created during consolidation. The graph gets denser where knowledge overlaps and sparser where it doesn't. Beam search naturally propagates through strengthened pathways. This is a deliberate contrast to systems like GraphRAG that create hierarchical summaries.

### Write-Time Salience Filter

The most distinctive design choice. Every incoming memory is scored:

| Signal | Weight |
|--------|--------|
| Novelty (BM25 duplicate check) | 0.45 |
| Surprise (caller-provided) | 0.15 |
| Decision made (boolean) | 0.15 |
| Causal depth (caller-provided) | 0.15 |
| Resolution effort | 0.10 |

Plus event-type bonuses: decision (+0.15), friction (+0.20), surprise (+0.25), causal (+0.20).

Disposition thresholds: active (≥ 0.4), staging (≥ 0.2), discard (< 0.2).

Novelty is computed via BM25 similarity to existing memories — the cheapest possible duplicate detection (~1ms, synchronous SQLite FTS5). An exact concept-string match adds a 0.4 penalty.

Memory classes: `canonical` gets a floor of 0.7 and always goes active; `ephemeral` is tagged but scored normally.

**Claimed result**: Only 23 of 100 events stored, 100% recall accuracy on their A/B eval. Pool cleanliness > pool completeness.

### Retraction

When a memory is discovered wrong:
1. Original marked `retracted` (soft delete — audit trail preserved)
2. Optional counter-engram created with `correction:{concept}` title, confidence 0.7, invalidation edge linking to retracted original
3. **Confidence contamination**: direct neighbors get penalty of 0.1 × edge weight. Depth 1 only; invalidation edges skipped (corrections aren't penalized)

Retracted memories are excluded from normal activation but visible via explicit search or ID lookup.

### Supersession

The `supersededBy` field tracks when a memory is updated rather than retracted. Superseded memories get a 0.15× score multiplier during retrieval — deprioritized but not hidden. This preserves "we used to do X" context while ensuring "we now do Y" dominates.

### Staging Buffer

Borderline-salience memories (0.2–0.4) enter staging with a 24-hour TTL. The buffer sweeps every 60 seconds:
- **Resonance check**: Run an activation query against active memories. If results above score 0.3 exist, the staged memory resonates with existing knowledge → promote.
- **No resonance after 24h** → hard delete.
- **Capacity cap**: 1,000 staged memories; excess deleted oldest-first.

During consolidation (Phase 7), the same resonance logic runs: cosine > 0.6 to any active memory → promote; >24h old with no match → discard.

### Hooks and Lifecycle

AWM installs Claude Code hooks for deterministic memory operations:
- **Stop hook**: Reminds Claude to write/recall after each response
- **PreCompact hook**: Auto-checkpoints before context compression
- **SessionEnd hook**: Auto-checkpoints + full consolidation on close
- **15-minute timer**: Silent auto-checkpoint while session is active

The hook sidecar runs as an HTTP server inside the MCP process on a separate port.

### Checkpoint/Restore

`memory_checkpoint` serializes execution state (JSON blob) to the `conscious_state` table. `memory_restore` recovers it and runs a recall for relevant context. This is designed to survive context compaction — the most common way Claude Code sessions lose state.

---

## Key Claims with Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| 100% edge case score (34/34) | Internal eval testing 9 specific failure modes | Self-testing own pipeline; designed to pass |
| 92.3% stress test (48/52) | 500 memories, 100 sleep cycles, adversarial spam | Better signal; stress conditions are non-trivial |
| AWM 100% vs baseline 83% on A/B | 100 project events, 24 recall questions | Baseline is keyword-only; the gap reflects basic retrieval, not a competitive comparison |
| 97.4% self-test (31 checks) | Pipeline component verification | Unit test suite, not a benchmark |
| 86.7% workday (43 memories, 4 sessions) | Simulated work sessions with recall challenges | Closest to realistic use; the failure modes would be informative |
| 93.1% real-world (300 chunks, 71K-line codebase) | Production monorepo with 16 recall challenges | Most credible eval; still internal design |
| 64.5% token savings | Memory-guided context vs full conversation history | Measures a different axis (efficiency, not quality) |
| 77% write-time filtering | A/B eval: 23 of 100 events stored | Credible; the question is whether the 77% truly contained no useful information |
| 15.4% LoCoMo multi-hop | Acknowledged in known-limitations.md | Honest disclosure; multi-hop is genuinely hard for all systems |

**Assessment**: All benchmarks are internal evals designed by the same author who built the system. No external benchmark comparison (LoCoMo QA, LongMemEval, etc.). The known-limitations document is the most informative part — candidly acknowledges multi-hop weakness, salience filter bias toward coding, scale limitations, and several unimplemented metrics.

---

## Relevance to Somnigraph

### Ideas worth evaluating

**Write-time salience filtering.** Somnigraph stores everything the agent sends and relies on retrieval quality (reranker, feedback loop, decay) to surface the right memories. AWM rejects 77% at write time. The question: does Somnigraph's pool contain noise that measurably hurts retrieval? If retrieval quality degrades as a function of pool size (more noise = worse BM25 candidate sets = more work for the reranker), write-time filtering becomes valuable. If the reranker handles noise effectively, it's unnecessary complexity.

**Experiment**: Measure NDCG@5k as a function of pool size (subsample GT queries at different memory counts). If flat, the reranker handles noise fine. If declining, investigate write-time novelty detection.

**Retraction with confidence contamination.** Somnigraph has contradiction edges (flagged during NREM classification) but doesn't propagate confidence penalties to neighbors when a memory is retracted. AWM's mechanism is simple: retract → neighbors lose 0.1 × edge weight confidence, depth 1 only. This could improve Somnigraph's handling of cascading misinformation without requiring a complex propagation model.

**Supersession tracking.** Somnigraph doesn't distinguish between "wrong" (retraction) and "outdated" (supersession). AWM's `supersededBy` field with a 0.15× score multiplier is a clean schema addition that preserves historical context while deprioritizing it. memv also has this (noted in similar-systems.md). Both systems arriving at the same solution independently increases confidence.

**Abstention gate.** When the reranker can't discriminate (low variance in top-5 scores), AWM returns empty rather than returning low-confidence garbage. Somnigraph's `limit` parameter puts this decision on the caller. An automatic abstention signal (as a flag or as a separate return field, not necessarily empty results) would let callers distinguish "nothing relevant found" from "here are the best I have."

**Adaptive reranker blend.** AWM's cross-encoder blend weight scales with BM25 signal strength: `0.3 + 0.4 * (1 - bm25Max)`. When keywords match strongly, trust keywords; when weak, lean on the cross-encoder. Somnigraph's LightGBM reranker already incorporates BM25 as a feature, so this is already implicitly handled — but the principle is worth verifying: does the reranker appropriately weight BM25 vs semantic features when BM25 signals are strong vs weak?

### Ideas not worth importing

**Query expansion via flan-t5-small.** 78MB model for synonym generation. Somnigraph's hybrid BM25 + vector search already provides semantic expansion through the vector channel. The added latency and model weight aren't justified.

**Local-only ONNX models.** AWM uses MiniLM-L6-v2 (384d) for embeddings — significantly less powerful than Somnigraph's OpenAI text-embedding-3-small (1536d). The "no API keys" constraint is a deployment philosophy, not a quality improvement.

**Greedy agglomerative clustering for consolidation.** Somnigraph's sleep pipeline does more sophisticated clustering with LLM-mediated classification, merge/archive/annotate decisions, and question-driven summarization.

**Task management tools.** Orthogonal to memory quality. AWM bundles task management (add/update/list/begin/end) into the memory system; Somnigraph correctly keeps these concerns separate.

**Staging buffer's resonance mechanism.** Conceptually interesting (hippocampal consolidation metaphor) but mechanistically simple (cosine > 0.6 to any active memory). Somnigraph's pending memory system with human review is more principled for high-stakes decisions.

### Convergent design patterns

| Pattern | AWM | Somnigraph |
|---------|-----|------------|
| ACT-R decay | `ln(n+1) - d * ln(age/(n+1))`, d=0.5 | Power-law per-category with floor and reheat |
| Hebbian edges | Co-activation buffer → weight strengthening | Co-retrieval tracking → EWMA + UCB |
| BM25 + vector hybrid | FTS5 + cosine, max fusion | FTS5 + sqlite-vec, RRF fusion |
| Confidence from feedback | Binary useful/not-useful adjusting scalar | Per-query ratings, EWMA, UCB exploration |
| Graph walk | Beam search (width 15, depth 2) | PPR-style novelty-scored expansion |
| Consolidation | 7-phase sleep cycle | 10-step REM pipeline with LLM classification |
| Reranker | Cross-encoder at inference (ms-marco-MiniLM) | LightGBM trained on GT with feedback features |

The convergence on similar cognitive primitives (ACT-R, Hebbian, CLS, homeostasis) across independently developed systems is worth noting. It suggests these are natural building blocks for memory systems, not arbitrary design choices.

---

## Code Quality Assessment

Clean TypeScript with strict mode. Reasonable separation of concerns (core primitives are stateless, engine pipelines are stateful). The Rocchio PRF code in activation.ts is duplicated (scoring logic copy-pasted for newly discovered candidates rather than extracted into a function) — a minor code quality issue. The consolidation engine's O(n²) pairwise cosine comparisons in clustering and redundancy pruning will not scale beyond a few hundred memories.

Documentation is thorough: architecture, cognitive model, feature docs, known limitations, open questions. The known-limitations.md is notably honest — most repos don't document their own weaknesses this clearly.

---

## Summary

AWM is a well-crafted system that converges on many of the same cognitive science foundations as Somnigraph. The primary novelty is write-time salience filtering, which is a genuinely different philosophy from store-everything-and-rank. The retraction/contamination mechanism and supersession tracking are clean implementations of ideas that Somnigraph should evaluate. The lack of a learned reranker, feedback loop beyond binary confidence, and external benchmark validation are the main gaps relative to Somnigraph's current state.

The most useful thing to take from AWM is the question it raises: is Somnigraph's pool noisy enough that write-time filtering would improve retrieval quality? The answer is an empirical measurement, not a design argument.
