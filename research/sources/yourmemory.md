# YourMemory — Ebbinghaus-decay MCP memory server with cheap real-time NER entity-graph linking

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

YourMemory (`sachitrafa/YourMemory`, v1.4.29, CC BY-NC 4.0, ~231 stars) is a self-hosted MCP server for AI coding agents (Claude Code, Cursor, Cline, Windsurf, OpenCode). FastAPI + uvicorn HTTP/SSE + stdio MCP. Local-first, all-local models. Positioned as "memory built on the Ebbinghaus forgetting curve."

### Storage & Schema
Three interchangeable backends selected at runtime: **DuckDB** (default, `array_cosine_similarity`), **SQLite** (numpy cosine in Python + FTS5), **Postgres+pgvector** (`content_tsv` generated tsvector). Graph stored separately: **NetworkX** pickle (`~/.yourmemory/graph.pkl`) default, or Neo4j.

`memories` table (~12-14 fields): `id, user_id, content, category, importance (0-1), embedding (768-dim), recall_count, created_at, last_accessed_at, agent_id, visibility (shared/private), context_paths (JSON path array)`. Auxiliary tables: `user_activity` (active-day tracking for decay), `memory_history` / `memory_archive` (audit + compaction), `agent_registrations`. UNIQUE `(user_id, content)` prevents exact dupes. Embeddings: sentence-transformers `multi-qa-mpnet-base-dot-v1` (768d, local).

### Memory Types
Four categories, and the category **is** the decay rate (`src/services/decay.py`): `strategy` (λ=0.10, ~38d), `fact` (λ=0.16, ~24d), `assumption` (λ=0.20, ~19d), `failure` (λ=0.35, ~11d). `fact`/`assumption` are auto-assigned by spaCy dependency parse (has subject → fact, else assumption); `strategy`/`failure` set manually by the agent at store time. Flat scheme — no episodic/semantic/reflection distinction, no layers, no priority scale.

### Write Path (`memory_mcp.store_memory` → `resolve.py`, `extract.py`)
Genuinely the richest part of the system, and mostly rule-based (no LLM needed):
1. **Question rejection** (`is_question`): ends with `?` or starts with what/who/where/... → 422.
2. **Salience gate** (`should_store_llm`): asks local Ollama `qwen2.5:7b` STORE/SKIP, with an explicit "skip meta-observations about the conversation" instruction. **Fails open** (stores) if Ollama unreachable — so on a default box with no Ollama this gate is a silent no-op.
3. **4-tier dedup** (`resolve.py`) on cosine vs nearest existing memory: ≥0.92 → **reinforce** (bump recall_count); 0.85-0.92 + contradiction → **replace**; 0.85-0.92 no contradiction → **merge** (entity-append); <0.85 → **new**.
4. **Subject-aware gate**: embeds the leading 2 words of each memory; if subject cosine <0.60 the two are declared different entities and never merged even at 0.95 similarity ("Sachit uses DuckDB" vs "YourMemory uses DuckDB").
5. **Rule-based contradiction detection** (`detect_contradiction`), no model: (a) polarity flip via hand-listed positive/negative verb sets + negation, (b) shared-ROOT-verb negation asymmetry, (c) 3-4 digit number conflict with ≥4 shared context words.
6. **Graph index** (`index_memory`): SVO triple extraction for verb-weighted similarity edges (top-5 neighbours ≥0.4 cosine × verb_weight) + **entity edges** (spaCy NER LIKE-search, weight 0.55) connecting memories sharing a named entity even when embeddings are dissimilar.

### Retrieval (`src/services/retrieve.py`)
Two rounds:
- **Round 1 hybrid**: `score = W_BM25·bm25_norm + W_VECTOR·cosine`. **Code sets both weights to 0.5** (`W_BM25 = 0.5, W_VECTOR = 0.5`, lines 21-22) — the evidence file's "0.4/0.6" is stale. Vector threshold 0.50 with a 0.20 fallback when Round 1 is empty. BM25 leg is backend-native (FTS5/DuckDB FTS/ts_rank_cd), each max/sigmoid-normalized to 0-1. **Decay strength is deliberately excluded from ranking** (computed only for display + pruning) — a documented, defensible choice.
- **Round 2 graph BFS** (`expand_with_graph`, depth 2): pulls neighbours of Round-1 seeds, scores them by `W_VECTOR·min(edge_weight, 0.74)` so a bridge node can compete for a top-k slot but never trigger recall reinforcement. `expand_k>0` makes this additive (keep top_k direct hits + up to expand_k neighbours).
- Boosts: spatial (+0.08 when `context_paths` overlaps `current_path`), temporal (time-window phrase detection). Optional off-by-default Ollama **relevance-judge filter** (`YOURMEMORY_RELEVANCE_JUDGE=1`) that drops off-topic candidates so recall can stay silent — one batched call, fails open.
- No learned reranker anywhere. No RRF — a fixed linear blend.

### Consolidation / Processing (`src/services/compaction.py`)
Real, and closer to Somnigraph's sleep than most systems in this corpus: greedy cosine clustering of live memories (sim ≥0.62, ≥5 members), LLM-summarize each cluster into one memory, copy originals to `memory_archive`, delete from `memories`, re-index the graph onto the summary. Conservative (prompt told to preserve all facts, originals recoverable). But it is a manual/endpoint-triggered compaction, not a scheduled multi-phase pipeline, and has **no edge typing, no gap analysis, no contradiction re-classification** — it is deduplicating summarization, not graph-building consolidation.

### Lifecycle Management
Ebbinghaus decay is the headline. `effective_λ = base_λ·(1 − importance·0.8)`; `strength = importance·e^(−effective_λ·days)·(1 + recall_count·0.2)`. Two nice touches: **activity-aware days** (`days` counts only user-active days from `user_activity`, falling back to wall-clock — vacations don't decay memories) and **chain-aware pruning** (`chain_safe_to_prune`: a sub-threshold memory is kept alive if ANY graph neighbour is still strong). Prune threshold 0.05, APScheduler 24h job. Supersession audit via `update_memory` → `memory_history`. DELETE is REST-only, **not** an MCP tool (agents can't forget).

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| "2x better recall than Zep on LoCoMo (59% vs 28%)" | `benchmarks/locomo_4way.py` self-run | **Retrieval Recall@5, not QA** — not comparable to Somnigraph's 85.1% LoCoMo QA. Lenient hit rule (substring OR ≥50% token overlap). Competitors (Zep/Supermemory/Mem0) run on free-tier cloud, README notes quotas "exhausted mid-benchmark" (asterisks). Confounded. |
| 89.4% LongMemEval-S | `benchmarks/longmemeval_fullstack.py` | Recall-any@5 (retrieval), self-run. Again not end-to-end QA. Plausible as a recall number; not head-to-head with QA systems. |
| HotpotQA: entity graph adds +12pp over similarity-only graph | `benchmarks/hotpotqa_reasoning.py` (BOTH_FOUND@5) | The most useful internal result — direct evidence that cheap NER entity-bridge edges help multi-hop retrieval. Self-run but the ablation is meaningful. |
| Ebbinghaus / activity-aware decay | `decay.py` — matches description | Validated in code. |
| 4-tier dedup + rule-based contradiction | `resolve.py` — matches | Validated. Contradiction detection is brittle (hand-listed verb sets, digit regex) but genuinely runs at write time with zero LLM cost. |
| Hybrid weight 0.4/0.6 (evidence file) | Code says 0.5/0.5 | **Evidence file wrong.** Minor, but flagged. |

---

## Relevance to Somnigraph

### What YourMemory does that Somnigraph doesn't
- **Real-time entity-graph construction at write time** (`graph_store.index_memory` + `_entity_linked_nodes`). Somnigraph builds all edges offline during NREM sleep (`sleep_nrem.py`); it explicitly lacks write-time graph construction and entity resolution. YourMemory links memories sharing a named entity via cheap spaCy NER + SQL LIKE the instant they're stored — the exact "bridge entity" pattern (`"Obama born in Hawaii" ←→ "Hawaii became a state in 1959"`) that Somnigraph's `docs/multihop-failure-analysis.md` identifies as its ~88% vocabulary-gap retrieval ceiling.
- **Write-path salience + dedup gating.** Somnigraph has no write-path quality gate (STEWARDSHIP lists this as a known gap). YourMemory has a question filter, an Ollama STORE/SKIP gate, and a 4-tier dedup/merge/replace resolver with a subject-aware guard — all at store time.
- **Activity-aware decay.** Somnigraph decay (`scoring.py`) is wall-clock exponential. YourMemory counts only active user-days, so a quiet stretch doesn't age memories. (Contrast with Somnigraph's deliberate opposite: project-gotcha `decay_rate=0.1` so memories *do* fade when a project goes quiet.)
- **Chain-aware pruning** — decayed memory survives if a graph neighbour is strong. Somnigraph archives during sleep independent of neighbour strength.
- Multi-backend portability (DuckDB/SQLite/Postgres) and multi-agent (API keys, shared/private visibility) — out of scope for Somnigraph (single-user).

### What Somnigraph does better
- **Learned 26-feature LightGBM reranker** (`reranker.py`, NDCG 0.7958) vs YourMemory's fixed 0.5/0.5 linear blend with no reranking.
- **Explicit feedback loop** with measured Spearman r=0.70 vs GT; YourMemory has no per-query utility feedback at all — `recall_count` is the only usage signal.
- **PPR graph-conditioned retrieval** + typed edges (supports/contradicts/evolves) + betweenness reranker feature vs YourMemory's untyped BFS depth-2 with a hard weight cap.
- **LLM-mediated multi-phase sleep** (NREM edge creation/merge/archive + REM gap analysis/question generation) vs a single clustering-summarization compaction.
- **Real end-to-end QA benchmark** (85.1% LoCoMo QA, Opus judge) vs retrieval-Recall@k only.

---

## Worth Stealing (ranked)

### 1. Write-time entity-bridge edges to attack the multi-hop vocabulary gap (Medium)
**What**: When a memory is stored, run cheap NER, and create graph edges to existing memories that mention the same named entity — even when embeddings are dissimilar (`_entity_linked_nodes`, `ENTITY_EDGE_WEIGHT=0.55`). YourMemory's own HotpotQA ablation shows +12pp BOTH_FOUND@5 from this over similarity-only edges.
**Why**: Somnigraph's documented retrieval ceiling is an ~88% vocabulary gap on multi-hop (`docs/multihop-failure-analysis.md`), and its graph is built only during sleep via LLM pairwise classification — expensive and latent. A cheap, immediate entity co-mention edge is exactly the "bridge that embeddings miss" mechanism, and it's complementary to (not a replacement for) the sleep-built typed graph.
**How**: A NER pass in the write path could seed provisional `entity:X` edges in `db.py`'s edge table, which `scoring.py`'s PPR expansion already consumes; NREM sleep would later confirm/type/prune them. Test whether provisional entity edges close any of the vocabulary gap before sleep runs. Revisit-if angle, not an adopt — it partly cuts against Somnigraph's "edges are LLM-judged, not co-mention" philosophy, so the experiment is: does cheap co-mention recall beat the false-edge noise it introduces.

### 2. Chain-aware archival gate (Low)
**What**: Don't archive/prune a decayed memory if a strong graph neighbour keeps its chain intact (`chain_safe_to_prune`).
**Why**: Somnigraph's sleep archival doesn't consult neighbour strength; a low-decay memory that's a load-bearing hop in a multi-hop chain can be archived, silently breaking retrieval paths.
**How**: In `sleep_nrem.py` archival, gate the archive decision on `max(neighbour PPR-weight or strength) < threshold`. Somnigraph already has the graph and betweenness — betweenness could be a sharper gate than raw neighbour strength.

### 3. Activity-aware decay clock (Low)
**What**: Decay by active-user-days, not wall-clock (`user_activity` table + `get_active_days_since`).
**Why**: Somnigraph is a single-user tool that goes quiet for stretches; wall-clock decay penalizes memories for the user's downtime. Worth a conscious decision either way (Somnigraph currently wants project gotchas to fade during quiet — so this is a knob to consider per-category, not a blanket adopt).
**How**: An optional active-day counter feeding `scoring.py`'s decay term; likely category-scoped (durable facts on active-day clock, ephemeral gotchas on wall-clock).

---

## Not Useful For Us

### Rule-based contradiction detection (hand-listed verb sets, digit regex)
Brittle by construction; Somnigraph already classifies contradictions during NREM sleep with an LLM (higher quality). A cheap write-time pre-filter isn't worth the false-positive rate for a single-user store.

### Multi-backend + multi-agent + Web UI + platform auto-injection
DuckDB/Postgres portability, API-key agent auth, shared/private visibility, the force-directed graph visualizer, and the 6-client `yourmemory-setup` installer are product/distribution surface irrelevant to a single-user research artifact.

### The benchmark suite as comparative evidence
Self-run Recall@k against rate-limited competitor free tiers with a lenient token-overlap hit rule. Not usable as a comparison point against Somnigraph's LLM-judged QA numbers.

---

## Connections

- **Convergent with the Phase 18 sweep finding** (`docs/sessions/2026-06-28-phase18-source-sweep.md`): write-path quality, not retrieval sophistication, is where the LoCoMo/LME leaders win. YourMemory is another data point — its richest, most differentiated code is the write path (dedup/merge/replace/subject-gate/salience), while retrieval is a plain 0.5/0.5 blend with no reranker. Independent corroboration of the AMemGym finding.
- **Entity-bridge edges** echo the graph-augmented reranker work in Somnigraph's LoCoMo Level 5b (`project_extraction_v6.md`: synthetic nodes as Phase-1 bridges) — both target the multi-hop vocabulary gap, but YourMemory does it at write time with NER rather than via synthetic-node generation during retrieval.
- **Rule-based contradiction detection** parallels the sleep-based contradiction classification in `sleep_nrem.py` — same goal (flag conflicting memories), opposite cost/quality trade (regex vs LLM).
- Same "decay computed but excluded from ranking" stance that several corpus systems land on independently — decay is a lifecycle/GC signal, not a relevance signal.

---

## Summary Assessment

YourMemory's core contribution is a **disciplined, mostly-LLM-free write path** (question filter → salience gate → 4-tier dedup with subject-aware and rule-based contradiction guards → immediate NER entity-graph linking) bolted onto a conventional hybrid retriever and an Ebbinghaus decay model with two thoughtful refinements (activity-aware days, chain-aware pruning). It is a solid, well-engineered *product* — three backends, multi-agent, a web UI, one-command setup for six clients — but architecturally it is a generation behind Somnigraph on the things Somnigraph optimizes: no learned reranker, no feedback loop, no typed graph, no multi-phase consolidation.

The single most valuable idea for Somnigraph is the **write-time entity-bridge edge**: cheap NER co-mention linking that directly attacks the multi-hop vocabulary gap Somnigraph has documented as its retrieval ceiling, with a self-run +12pp HotpotQA ablation as at least suggestive evidence. It's a revisit-if experiment rather than a clear adopt, because it cuts against Somnigraph's "edges are LLM-judged, not co-mention" design — the open question is whether cheap provisional entity edges recall more than the false-edge noise they add, pending sleep confirmation.

What's overhyped: the benchmark framing. "2x better than Zep" is Recall@5 with a lenient hit rule against rate-limited competitor free tiers — a retrieval-recall number wearing QA-benchmark clothes, and explicitly not comparable to Somnigraph's 85.1% LLM-judged LoCoMo QA. What's genuinely missing relative to its own pitch: the "Ebbinghaus forgetting curve" branding oversells a decay model that is deliberately excluded from ranking and only drives a 24h prune job.
