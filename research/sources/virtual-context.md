# Virtual-Context Analysis

*Generated 2026-03-22 by Opus agent reading local clone*

---

## 1. Architecture Overview

**Repo**: https://github.com/virtual-context/virtual-context
**License**: Apache 2.0
**Language**: Python, ~42K lines (excluding tests)
**Created**: 2025, 369 commits (single author, actively developed)
**Description**: OS-style context window manager for LLMs — a proxy that compresses, indexes, and pages conversation history to fit within model context limits.

**Core design**: Virtual-context sits between client and LLM, intercepting API calls. The client declares a large context window (e.g., 20M tokens) while the model sees a bounded working set (~60K). The system manages what fits via three compression layers, topic-based retrieval, and tool-augmented paging. This is fundamentally different from a memory server — it manages the conversation window itself, not a separate persistent store.

**Three compression layers**:
- **Layer 0** (Active Memory): Raw conversation turns in the context window
- **Layer 1** (Compressed Pages): Per-topic segment summaries + extracted structured facts
- **Layer 2** (Working Set Descriptors): Tag summaries via greedy set cover — bird's-eye topic map

**Delivery modes**: HTTP proxy (zero code changes — intercepts API calls), MCP server, or direct Python SDK. Includes a TUI and dashboard.

**Module organization** (key packages under `virtual_context/`):
- `core/retriever.py` — 3-signal retrieval orchestration
- `core/retrieval_scoring.py` — RRF fusion with gravity/hub dampening
- `core/assembler.py` — Context assembly within token budgets
- `core/paging_manager.py` — Depth-level paging with LRU eviction
- `core/tool_loop.py` — Tool-augmented retrieval (6 tools)
- `core/fact_query.py` — Structured fact querying with verb expansion
- `core/compactor.py` — LLM-based summarization
- `core/tag_consolidator.py` — Semantic tag merging
- `core/temporal_resolver.py` — Time-bounded recall
- `ingest/supersession.py` — Fact contradiction detection and merging
- `ingest/curator.py` — LLM-based fact relevance filtering

**Storage backends**: SQLite (default), PostgreSQL, Neo4j, FalkorDB via abstract `CompositeStore`.

---

## 2. Retrieval Pipeline

Three independent signals fused via RRF, with pre- and post-fusion dampening.

### Signal 1: IDF Tag Overlap (`compute_idf_candidates()`)
Tag-to-segment matching weighted by IDF: `log(1 + total_segments / tag_usage_count)`. Primary tag matches get full IDF weight; related tag matches get 0.5×. Overfetches to 30 candidates.

### Signal 2: BM25 Full-Text Search (`compute_bm25_candidates()`)
FTS5 on tag summaries (primary) and segment full_text (secondary). Default limit 20.

### Signal 3: Embedding Cosine Similarity (`compute_embedding_candidates()`)
`all-MiniLM-L6-v2` via sentence-transformers. Threshold 0.25, limit 20. Precomputed tag summary embeddings loaded from store.

### RRF Fusion (`rrf_fuse()`)
```
score = Σ(weight_signal × 1.0 / (k + rank + 1))
```
- k = 60 (default)
- Missing signal → penalty rank = k × 2 = 120
- Weights: IDF 0.50, BM25 0.30, embedding 0.20 (normalized to sum to 1.0)

### Post-Fusion Dampening

**Gravity dampening** (pre-RRF on embedding scores): Halves the embedding score of any tag with BM25 score = 0 and embedding score > 0.5. Penalizes "hallucinated" semantic matches that have no keyword support. Factor: 0.5.

**Hub dampening** (post-RRF): Penalizes tags with segment count above the 90th percentile. Formula: `penalty = 1.0 - 0.6 × (count - p90) / (max_count - p90)`, floored at 0.2× original score. Query tags are exempt (user intent preserved).

**Resolution boost** (post-RRF): Tags with actionable structured facts get 1.15× score boost.

---

## 3. Fact Management

### Fact Schema
Rich structured facts extracted during compaction:
- `subject`, `verb`, `object` — SVO triple
- `status` — `active | completed | planned | abandoned | recurring`
- `what` — Full-sentence durable knowledge with all specifics
- `who`, `when_date`, `where`, `why` — 5W provenance
- `fact_type` — `personal | experience | world`
- `tags`, `segment_ref`, `conversation_id`, `turn_numbers`, `mentioned_at`, `session_date`
- `superseded_by` — Fact ID that replaces this fact

### Supersession Detection (`SupersessionChecker`)
Three-tier candidate pool construction:
1. **Tag-scoped**: Facts sharing tags with the new fact
2. **Object-keyword**: Regex extraction of proper nouns from object field for cross-session matching
3. **Embedding-based**: Semantic similarity catch-all (if embed function provided)

LLM prompt presents old vs. new fact side-by-side and asks for merged `{verb, object, status, what}`. Merged fact replaces both; old fact marked `superseded_by`. Optional `FactLink` with `relation_type=supersedes` for graph mode.

Deterministic pre-pass: `promote_planned_facts()` auto-promotes planned facts past their `when_date` to `status=completed` (optional LLM rewrite to past tense).

### Verb Expansion (`FactQueryEngine`)
10 manual synonym clusters (e.g., `{visited, returned from, completed, went on, toured}`) plus embedding-based expansion at cosine threshold 0.53. Combined manual + embedding matches for broader recall.

### Temporal Resolution (`TemporalResolver`)
Relative presets (`last_7_days`, `this_week`, `last_month`, etc.) and absolute `between_dates` with ISO parsing. Returns both conversation snippets (via `find_quote`) and structured facts (via `query_experience_facts_by_date`), filtered by `session_date` bounds.

---

## 4. Paging System

### Depth Levels
- `NONE` — Listed in topic hint only (0 tokens)
- `SUMMARY` — Tag summary (~200 tokens) — **default**
- `SEGMENTS` — Individual segment summaries (~2K tokens)
- `FULL` — Original full_text (~8K+ tokens)

### Working Set & Eviction
Each topic tracked as `WorkingSetEntry` with depth, token cost, and `last_accessed_turn` (for LRU). When expanding a topic exceeds `tag_context_max_tokens` (default 30K), LRU eviction collapses coldest topics to SUMMARY or removes them entirely.

### Tool-Augmented Retrieval
Six tools exposed to the LLM via the tool loop:

| Tool | Purpose |
|------|---------|
| `vc_expand_topic` | Load full or segment-level text for a topic |
| `vc_find_quote` | Full-text search across conversation history |
| `vc_find_session` | Retrieve full text from a specific older session |
| `vc_query_facts` | Structured fact query with subject/verb/object/status filters |
| `vc_recall_all` | Load all tag summaries at once |
| `vc_remember_when` | Time-bounded fact + quote search |

Anti-repetition: tracks `presented_refs` and `presented_facts` sets across loop iterations to suppress duplicate results. Intent-context re-ranking: if the user's question is available, fact results are re-sorted by embedding similarity to the question. Max 10 iterations; timeout forces final response.

---

## 5. Compaction & Tag Management

### Summarization
Two-pass compaction pipeline: Pass 1 (sequential, no LLM) detects stubs and merge candidates via tag overlap + embedding similarity + keyword match + recency. Pass 2 (batch LLM) compresses segments.

Prompts enforce number preservation ("CRITICAL — Any text involving numbers is mandatory"), personal disclosure retention, and user voice preservation. Multi-segment rollup produces a single tag summary with `{summary, description, entities, key_decisions, action_items}`.

### Tag Consolidation
Post-compaction semantic clustering merges equivalent tags (e.g., "db" → "database"). LLM groups tags in batches of 500. Transitive merge for cross-batch aliases. Backfills `segment_tags` with canonical tags.

---

## 6. Benchmark Results

### LongMemEval (100 questions)
- **Virtual-context: 95%** accuracy (MiMo-V2-Flash tagger + Claude Sonnet 4.5 reader + Gemini 3 Pro judge)
- **Baseline (full history): 33%** (Claude Sonnet 4.5 with ~118K tokens)
- VC uses 52K tokens/question (2.2× fewer than baseline) at $0.16/question
- Per-category: knowledge-update 100%, single-session 100%, temporal 92.9%, multi-session 88.5%
- Full-context fails on knowledge-update (29.4%) and temporal (32.1%) — classic "lost in the middle"

### LoCoMo
Benchmark harness exists (`benchmarks/locomo/`) with F1 scoring (Porter stemming), LLM judge, per-category breakdown, and crash-recovery saves. Category-specific handling: adversarial = binary, multi-hop = partial F1 over sub-answers, temporal = off-by-one-day tolerance. No published LoCoMo numbers found in the README.

---

## 7. Comparison to Somnigraph

### What Virtual-Context has that we don't

**Gravity dampening.** Halving embedding scores with zero BM25 support is a simple cross-signal agreement check. Somnigraph's reranker has both `fts_bm25_norm` and `vec_dist_norm` as features, so the model could learn this pattern — but it's not an explicit mechanism, and we haven't checked whether the model does learn it.

**Hub dampening.** Penalizing high-frequency tags above p90 is a form of IDF at the topic level. Somnigraph doesn't have an equivalent. The concept of penalizing "hub" memories that match too many queries is interesting — though our feedback loop partially addresses this (frequently-surfaced-but-unhelpful memories get downrated).

**Structured fact supersession.** LLM-based detection and merging of contradicted facts with `superseded_by` chains. Somnigraph's contradiction handling is edge-based: sleep flags contradictions but doesn't merge facts or create explicit supersession chains.

**Tool-augmented retrieval.** The LLM can drill down on demand (`vc_expand_topic`, `vc_find_quote`) rather than getting one-shot results. Somnigraph's `recall()` is a single call — there's no mechanism for the agent to request more detail on a specific result.

**Temporal query tools.** `vc_remember_when` with relative/absolute date ranges and structured fact filtering by `session_date`. Somnigraph has `created_at` but no time-bounded recall API.

**Planned fact promotion.** Auto-promoting `planned` facts past their date to `completed`. Somnigraph has no temporal status lifecycle for memories.

### What we have that they don't

**Retrieval feedback loop.** `recall_feedback()` creates a gradient signal. EWMA aggregation, UCB exploration, Hebbian co-retrieval. Virtual-context has no mechanism for the agent to tell it which results were useful.

**Learned reranker.** LightGBM model trained on 1032 human-judged queries (NDCG=0.7958). Virtual-context's RRF fusion uses fixed weights and hand-tuned dampening — no offline evaluation or ground truth.

**Biological decay.** Per-memory exponential decay with configurable rates, dormancy detection, shadow load tracking. Virtual-context manages staleness through paging eviction and tag consolidation, not decay.

**Sleep consolidation.** Three-phase offline processing (NREM classification, REM per-memory LLM decisions, archiving). Virtual-context's compaction is online and triggered by token pressure, not scheduled offline processing.

**Graph structure.** PPR graph expansion over typed edges (support, contradict, evolve). Virtual-context has no memory graph — topics are flat tags, not connected nodes.

**Hebbian co-retrieval.** PMI-based edge strengthening from retrieval co-occurrence. No equivalent.

### Architectural trade-offs

| Dimension | Virtual-Context | Somnigraph | Trade-off |
|-----------|----------------|-----------|-----------|
| Scope | Context window manager (proxy) | Persistent memory system (MCP) | VC manages what's in the window; Somnigraph manages what's in the store |
| Retrieval paradigm | Multi-turn tool loop (LLM drills down) | One-shot scored recall | VC: higher precision possible. Somnigraph: lower latency, deterministic |
| RRF fusion | IDF + BM25 + embedding (0.50/0.30/0.20) | FTS5 + sqlite-vec (equal weight) | Similar hybrid approach; VC weights IDF highest |
| Post-fusion | Gravity + hub dampening + resolution boost | Learned LightGBM reranker | VC: hand-tuned heuristics. Somnigraph: data-driven |
| Fact handling | Structured SVO + supersession chains | Freeform text + contradiction edges | VC: richer schema, explicit lifecycle. Somnigraph: simpler, relies on sleep |
| Consolidation | Online compaction (token pressure trigger) | Offline sleep pipeline (scheduled) | VC: always current. Somnigraph: deeper decisions, higher latency |
| Decay/forgetting | Paging eviction (LRU) | Per-memory exponential decay | VC: spatial pressure. Somnigraph: temporal biological model |
| Feedback | None | Explicit per-query utility scoring | Somnigraph adapts to usage; VC does not |
| Embeddings | Local (MiniLM-L6-v2, 384d) | Cloud (OpenAI text-embedding-3-small, 1536d) | Privacy vs. quality/convenience |

---

## 8. Worth Adopting?

**Gravity dampening as reranker feature**: The cross-signal agreement concept (penalize embedding-only hits with no BM25 support) could be expressed as a binary or ratio feature for the reranker: `has_bm25_support = 1 if fts_rank < K else 0`, or `bm25_embed_agreement = fts_bm25_norm × vec_dist_norm`. Worth testing as a 32nd feature — the reranker may already learn this implicitly from having both raw scores, but an explicit interaction term could help. **Low effort, worth trying.**

**Hub dampening concept**: Penalizing high-frequency topics is IDF at the memory level. Somnigraph doesn't have a "how many queries does this memory match" feature. A `retrieval_frequency` feature (how often has this memory appeared in top-K across all queries) could serve the same purpose. However, the feedback loop already downrates frequently-surfaced-but-unhelpful memories, so the signal may be partially captured. **Medium effort, unclear value.**

**Structured fact supersession**: Somnigraph's contradiction handling via sleep edges is weaker — it flags contradictions but doesn't merge or create traversable supersession chains. memv's `superseded_by` field was already noted as interesting (see similar-systems.md § memv). Adopting this would require schema changes and write-path LLM calls. **High effort, worth considering for a future tier.**

**Tool-augmented retrieval**: Fundamentally different paradigm. Adding drill-down tools (`expand_memory`, `get_related`) would be a major architectural change. Worth documenting as a design alternative but not adopting incrementally. **Not adopting.**

**Temporal query API**: An `after`/`before` parameter on `recall()` would be simple to implement and useful. The schema already has `created_at`. **Low effort, worth considering** — but event_time (roadmap Tier 3 #18) is the real solution.

---

## 9. Worth Watching

**LoCoMo benchmark results.** The harness exists but no published numbers. When they publish, direct comparison with our 85.1% overall would be informative — especially since VC's approach (tool-augmented retrieval with paging) is structurally different from our approach (one-shot reranked recall).

**LongMemEval as a second benchmark.** Their 95% on LongMemEval is impressive but uses a different methodology than standard. If LongMemEval gains adoption as a benchmark, we should consider implementing it.

**Dampening tuning.** The gravity/hub dampening parameters are hand-tuned (gravity factor 0.5, hub penalty 0.6, p90 threshold). If they publish ablations or tuning studies, the results would inform whether these heuristics are robust or brittle.

---

## 10. Relevance to Somnigraph

**Medium.** Virtual-context operates in a different design space (context window management vs. persistent memory), so architectural lessons are limited. The most transferable ideas are at the scoring level: gravity dampening as a cross-signal agreement mechanism, and the structured fact supersession model. The tool-augmented retrieval paradigm is interesting as a documented alternative but not something to adopt.

The LongMemEval results (95% vs. 33% full-context) validate the compression-and-retrieve approach over full-context, which is the same conclusion our LoCoMo results support from a different angle (85.1% vs. 72.9% full-context).

**Primary value**: Two concrete reranker feature ideas (cross-signal agreement, retrieval frequency), one design pattern worth documenting (tool-augmented paging), and a potential second benchmark (LongMemEval).
