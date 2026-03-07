# Ember MCP — Source Analysis

*Phase 12, 2026-02-20. Analysis of Arkya-AI/ember-mcp.*

## 1. Architecture Overview

**Language:** Python 3.10+, async throughout (`asyncio`, `aiofiles`, `aiosqlite`).

**Key dependencies:**
- `sentence-transformers` — `all-MiniLM-L6-v2` for 384-dim local ONNX embeddings (same model dimension as claude-memory's FastEmbed)
- `faiss-cpu` — Meta's FAISS for vector similarity search, wrapped in a custom `IndexIDMap`
- `pydantic v2` — All data models
- `aiosqlite` — Async SQLite for edges, region stats, and metrics log
- `mcp` (FastMCP) — MCP server interface

**Storage layout:**
```
~/.ember/
├── config.json
├── embers/*.json          # One JSON file per memory (Pydantic serialized)
├── index/
│   ├── vectors.faiss      # FAISS IndexIDMap
│   └── id_map.json        # UUID <-> integer ID mapping
└── cells/
    ├── centroids.npy      # 16 frozen Voronoi centroids (seed=42)
    └── stats.db           # SQLite: edges, region_stats, metrics_log
```

**File structure:**
- `models.py` — Pydantic schemas (`Ember`, `EmberConfig`, `RegionStats`, etc.)
- `core.py` — `VectorEngine`: embeddings, FAISS index, Voronoi cell assignment
- `storage.py` — `StorageManager`: JSON file I/O for embers, SQLite for edges/stats, ID map
- `utils.py` — Pure math: HESTIA scoring, shadow-decay functions, temporal scoring
- `server.py` — FastMCP tool definitions (1101 lines, the main event)
- `config.py` — Config load/save with defaults
- `bootstrap.py` / `scanners/` — Cold-start scanner pipeline
- `migration.py` — `~/.anchor/` to `~/.ember/` migration (prior product name was "Anchor")
- `cli.py` — `ember-mcp init/run/status/bootstrap/migrate` subcommands

**Scalability note:** Embers stored as individual JSON files (one file per memory). This does not scale beyond several thousand embers but is zero-dependency simple. FAISS index is a flat in-memory index persisted to disk with write batching and cross-process file locking.

---

## 2. Memory Type Implementation

**Schema (`Ember` model):**

The core memory unit has these notable fields:
- `name` + `content` — required, no summary vs. detail split
- `importance` — five types: `fact`, `decision`, `preference`, `context`, `learning`. Each has a configurable half-life in days (`DECAY_HALF_LIVES`: fact=365, learning=90, preference=60, decision=30, context=7)
- `cell_id` — Voronoi partition assignment (0-15)
- `is_stale` + `stale_reason` — explicit staleness flag, set by `ember_contradict`
- `shadow_load` (float 0-1) — how much newer overlapping memory suppresses this one (the Shadow-Decay signal)
- `shadowed_by` — ID of the dominant newer memory
- `related_ids` — up to 5 KG neighbor IDs (populated at insert time)
- `supersedes_id` / `superseded_by_id` — bidirectional contradiction chain
- `source_path` — path to originating file on disk (enables `ember_deep_recall`)
- `access_count`, `last_accessed_at` — recency tracking
- `session_id`, `source` — provenance (`manual`, `auto`, `session`, `bootstrap`)

**No explicit category taxonomy** analogous to claude-memory's `episodic/semantic/procedural/reflection/meta`. The five `importance` levels serve a different purpose (decay rate) than category (type of knowledge).

**Extraction:** `ember_learn` relies on the LLM to prefix content with `TYPE: ...` before calling the tool. There is no AI-powered extraction within Ember itself — the LLM does the classification. Near-duplicate detection uses a hard L2-distance threshold (< 0.1 ≈ cosine similarity > 0.95).

---

## 3. Retrieval Mechanism

**The pipeline: `_fetch_and_rerank()` → HESTIA scoring**

1. Embed the query string via `all-MiniLM-L6-v2` (async, semaphore-limited to 2 concurrent)
2. Broad FAISS search: retrieve `top_k × 10` candidates (fetch multiplier = 10)
3. Radius search around the query vector for "topic vitality" neighbors (all embers within cosine distance ≥ 0.5 of the query)
4. Batch-load all candidate + radius ember JSON files concurrently (`asyncio.gather` with semaphore=20)
5. Compute **Topic Vitality** `V(x,t)` = sum of `exp(-λ × age_days)` for all neighbors within radius — a measure of how actively this topic region has been discussed recently
6. Compute **HESTIA score** for each candidate:
   `S = cos_sim × (1 - Φ)^γ × (α + (1-α) × V/V_max)`
   where Φ = shadow_load (cached on the ember, not recomputed at query time), γ = 2.0 (shadow hardness), α = 0.1 (nostalgia floor)
7. Sort by HESTIA score, return top_k
8. Concurrently write back access count updates

**Shadow-on-Insert pipeline:** When a new ember is stored, it searches its k=10 nearest neighbors and computes a shadow potential `φ` for each neighbor. If `φ > neighbor.shadow_load`, it updates the neighbor's shadow_load, marks `shadowed_by`, writes a `shadow` edge in SQLite, and updates the cell's conflict density via EMA. It also detects "related but not shadowing" neighbors (cos_sim > 0.4 AND φ < 0.1) and stores them as `related` edges.

**Critical overclaiming in README vs. code reality:**

The README claims "Voronoi partitioning" and "Laplacian smoothing to suppress isolated noise" and "adaptive thresholding." The actual `ember_drift_check` implementation is much simpler: it queries per-cell `region_stats` from SQLite and applies two static threshold checks (`shadow_accum > 0.3` → drifting, `vitality_score < 0.01` → silent). There is no Laplacian smoothing or k-NN adjacency graph in the actual code. The Welford statistics described in the README are also absent — shadow accumulation uses a simple EMA. The README's "drift detection pipeline" description does not match what `ember_drift_check` actually does.

The README also says "Source Linking... it points you to the specific meeting note." This is implemented but only works if you proactively set `source_path` when storing memories — it is not automatic.

No FTS (full-text search) — unlike claude-memory's FTS5 + vector RRF hybrid, Ember is purely vector-based.

---

## 4. Standout Feature

**The Shadow-Decay framework (HESTIA scoring with shadow-on-insert propagation).**

This is the most coherent and genuinely novel element. The idea: when you store a new memory that is semantically similar to an older one, the older memory's retrieval score is *automatically suppressed* without requiring explicit contradiction. The suppression is proportional to the cosine similarity between old and new memories and to the temporal ordering — only *newer* memories can shadow *older* ones.

The math is clean and physically motivated:

```
Shadow potential: φ(newer | older) = max(0, (cos_sim - (1-δ)) / (δ - ε))
                  only when t_newer > t_older

Shadow load: Φ = max shadow potential across all k neighbors

HESTIA score: S = cos_sim × (1 - Φ)^γ × vitality_factor
```

The `shadow_load` value is cached on each ember and updated eagerly at insert time, making query-time scoring O(1) per candidate — no recomputation needed at retrieval. This is architecturally smart: insert is slightly expensive (searches k neighbors and updates them), but recall is fast.

This is a concrete, implemented answer to the problem: "how do I prevent the model from using my old architecture decisions after I've pivoted?" Rather than relying on explicit human-in-the-loop staleness marking, shadowing happens automatically when related newer memories exist.

---

## 5. Other Notable Features

**Bootstrap scanner pipeline.** On first install, Ember scans `~/.claude/projects/` (Claude Code JSONL sessions), `~/.codex/sessions/` (OpenAI Codex CLI), VS Code/Cursor `state.vscdb` SQLite files (GitHub Copilot), recent git repos (READMEs, tech stacks, last 20 commits), and `~/Desktop/*.md` files. This seeds the memory store automatically without any user action. The cross-IDE coverage is practical and thoughtfully implemented.

**Knowledge graph with BFS traversal.** Edges (`shadow`, `related`, `supersedes`) are stored in SQLite and exposed via `ember_graph_search`, which does: HESTIA vector search → pick entry node → BFS traversal up to N hops → return connected embers. The traversal uses batch SQL `IN (...)` queries per BFS frontier level rather than N individual queries.

**`ember_explain` and `ember_health` tools.** `ember_explain` shows the full HESTIA breakdown for a specific memory (shadow load, vitality, edge list, scoring factors). `ember_health` computes a hallucination risk score across the whole store: `0.4 × shadowed_ratio + 0.3 × stale_ratio + 0.3 × silent_topics_ratio`, logged to the metrics table with trend history. These observability tools are missing from most memory systems.

**Deferred write batching.** FAISS index writes are batched with a 2-second defer and a background flush task, avoiding I/O on every insert. Cross-process file locking (`filelock`) is used for both the FAISS index and the ID map JSON, with mtime-based staleness detection for multi-client consistency.

**Import taxonomy with differentiated decay rates.** The `DECAY_HALF_LIVES` dict ties memory type to decay behavior. Facts decay over a year; context decays in a week. This is baked into the HESTIA scoring via the `compute_temporal_score` function (used as a fallback) and implicitly via shadow-on-insert (older, more decayed embers get higher shadow loads from newer ones).

---

## 6. Gap Ratings

| Gap | Rating | Evidence |
|-----|--------|----------|
| Layered Memory | 10% | Single layer: all embers are equal depth. No gestalt/summary/detail tiers. `ember_deep_recall` reads source files but that is file I/O, not a memory layer. |
| Multi-Angle Retrieval | 15% | Pure single-vector FAISS search with no FTS, no multi-perspective indexing. The fetch multiplier (10x) broadens the candidate pool but does not add angles. |
| Contradiction Detection | 65% | Shadow-Decay automatically detects suppression relationships between semantically similar memories of different ages. `ember_contradict` provides explicit hard contradiction. Edge case: shadow detection requires cosine sim > 0.7 (delta=0.3) so paraphrased contradictions may not be caught. No semantic contradiction detection between dissimilar-but-conflicting content. |
| Relationship Edges | 70% | SQLite `edges` table with typed edges (`shadow`, `related`, `supersedes`), BFS traversal, `ember_graph_search`. Missing: no LLM-generated semantic relationship classification — edges are purely geometric (similarity-based) or structural (contradiction chain). |
| Sleep Process | 5% | No background consolidation, no synthesis, no layer-building. `ember_drift_check` and `ember_recompute_shadows` exist but must be called explicitly by the LLM. |
| Reference Index | 40% | `ember_inspect` shows per-cell ember counts and conflict density in a bar chart. `ember_list` with pagination. `ember_health` gives a quantitative store-wide risk score. No lightweight keyword/topic index for fast overview. |
| Temporal Trajectories | 35% | `supersedes_id` / `superseded_by_id` chains track explicit contradiction history. Shadow-Decay implicitly tracks "newer overwrites older" suppression over time. No belief trajectory queries, no visualization of how a topic evolved. |

---

## 7. Comparison with claude-memory

**Stronger:**

- **Automatic staleness detection.** Shadow-Decay is a genuine improvement: new memories automatically suppress semantically similar older ones without human curation. Claude-memory's contradiction detection requires manual `remember()` calls or a sleep cycle.
- **Richer knowledge graph.** Ember has typed edges in SQLite, BFS traversal, and graph-based search via `ember_graph_search`. Claude-memory has `sleep_write_results` edges but no graph traversal tool exposed to the LLM.
- **Bootstrap seeding.** First-run population from Claude Code, Codex, Copilot, git repos, and documents is practical and production-ready. Claude-memory has no equivalent.
- **Observability.** `ember_explain` (per-memory HESTIA breakdown), `ember_health` (composite risk score with trend), `ember_inspect` (cell distribution map), and a persistent metrics log are absent from claude-memory.
- **Source linking.** `source_path` on every ember + `ember_deep_recall` that reads the originating file at query time is a useful pattern for linking memories back to artifacts.
- **Differentiated decay by type.** Five importance levels with per-type half-lives is cleaner than claude-memory's single configurable `decay_rate` per memory.

**Weaker:**

- **No FTS.** Claude-memory's FTS5 + RRF fusion catches keyword-based queries that vector search misses. Ember is purely semantic.
- **No human-in-the-loop curation.** Claude-memory's `review_pending()` flow lets the user confirm, edit, or discard auto-captured memories before they enter the store. Ember's `ember_learn` writes immediately, with near-duplicate detection only at a high similarity threshold (0.95). Calibration errors accumulate silently.
- **No sleep consolidation.** Claude-memory's sleep skill synthesizes relationships, detects evolution, and builds graph structure via LLM reasoning. Ember's relationship edges are purely geometric.
- **No dimensional recall feedback.** Claude-memory's `recall_feedback` allows the LLM to score retrieved memories (relevance/impact/surprisal), feeding back into priority. Ember has access-count boosting but no per-retrieval quality signal.
- **Storage architecture does not scale.** One JSON file per memory is simple but creates filesystem pressure at thousands of memories. Claude-memory uses a single SQLite table.
- **No category taxonomy.** Claude-memory's `episodic/semantic/procedural/reflection/meta` categories shape retrieval filtering and startup loading. Ember's `importance` field serves decay, not type-based routing.
- **No startup briefing.** Claude-memory's `startup_load()` provides a priority-ranked briefing at session start. Ember has `ember_auto` but it is a simple HESTIA search, not a curated briefing optimized for context budget.
- **README overclaims.** The "Voronoi partitioning with Laplacian smoothing and adaptive thresholding" description is not implemented. The actual drift check is a two-threshold rule on EMA values. This is a credibility concern for evaluating what else the README may overstate.

---

## 8. Insights Worth Stealing

1. **Shadow-on-Insert for automatic suppression** (effort: high, impact: high). The core insight — cache a `shadow_load` scalar on each memory, updated eagerly when new similar memories arrive — is clean and query-efficient. A future claude-memory sleep cycle could compute shadow loads during consolidation rather than at query time. The HESTIA formula `S = cos_sim × (1 - Φ)^γ × vitality_factor` is a composable scoring function worth studying. The specific parameters (delta=0.3, gamma=2.0, alpha=0.1) would need calibration for claude-memory's use case.

2. **Topic Vitality as a query-time context signal** (effort: medium, impact: medium). Computing "how actively has this topic region been accessed/created recently" via a radius search and exponential decay sum is a good signal that complements retrieval score. A "hot topic" should rank higher than an equally-similar "cold topic." This could augment claude-memory's RRF with a vitality term.

3. **`ember_explain` — per-memory score breakdown** (effort: low, impact: medium). An `explain(memory_id)` tool that shows exactly why a memory scores how it does (cosine sim contribution, decay factor, shadow load, vitality, edges) is a debugging/calibration tool that claude-memory lacks. Would directly help diagnose why the wrong memories surface.

4. **`ember_health` — composite hallucination risk score with trend** (effort: low, impact: medium). A quantitative store-wide health metric logged to a metrics table with trend history is a practical observability primitive. Claude-memory could expose a similar `store_stats` or `health_check` call.

5. **Bootstrap scanner pipeline for cold start** (effort: high, impact: medium). Scanning `~/.claude/projects/` for existing Claude Code session history is immediately applicable and requires no changes to how sessions work. Could seed initial episodic memories without any manual work.

6. **Typed edges with BFS traversal** (effort: medium, impact: medium). Claude-memory's `sleep_write_results` already creates edges, but there is no graph traversal tool exposed to the LLM. Adding a `traverse_graph(start_id, depth=2)` tool and a graph-search mode (vector → entry node → BFS → related context) would let the LLM follow association chains.

7. **`source_path` on memories with source-file reading at recall time** (effort: low, impact: low-medium). Storing the originating file path and optionally reading it during deep recall creates a two-tier system: the memory is a bookmark, the file is the full content. Cheap to add.

---

## 9. What's Not Worth It

**The Voronoi cell partitioning.** Using 16 frozen random centroids with seed=42 for spatial partitioning does not add meaningful retrieval quality over a flat FAISS search. The cells are used for per-cell statistics (conflict density, vitality), which is the actually useful part. The partitioning itself is aesthetic — FAISS performs full brute-force L2 search across all vectors anyway via `IndexFlatL2`. At the scale Ember targets (~1K-10K memories), Voronoi partitioning provides no speed benefit and adds the complexity of centroid management and cell assignment.

**Per-file JSON storage.** One JSON file per memory is simple to implement but creates a maintenance burden at scale. It creates filesystem pressure on hot directories (stat/glob operations), makes atomic updates harder, and requires a separate ID map. A single SQLite table (as claude-memory uses) handles all of this transparently.

**The bootstrap scanner depth and breadth.** Scanning VS Code SQLite state.vscdb files, JetBrains configs, and Codex CLI sessions is thorough but adds substantial code for scanners that most users will never trigger. The Claude Code scanner is immediately useful for this use case; the others represent long-tail coverage that may not warrant the maintenance burden.

**`ember_recompute_shadows` as an LLM-callable tool.** Full shadow recalculation across every ember is an O(n^2) operation that should be a background CLI job, not an LLM-invokable MCP tool. Including it in the tool list pollutes the tool space and risks accidental expensive invocations.

---

## 10. Key Takeaway

Ember MCP's standout contribution is the Shadow-Decay framework: a mathematically principled approach to automatic staleness detection where newer semantically-similar memories suppress older ones via a cached scalar (`shadow_load`), without requiring explicit human contradiction. This is a genuinely useful idea that addresses a real failure mode in memory systems (stale data being retrieved with high confidence). The HESTIA scoring function that integrates cosine similarity, shadow suppression, and topic vitality into a single composable retrieval score is clean and worth adapting. The observability tools (`ember_explain`, `ember_health`, `ember_inspect`) represent a level of memory introspection that most systems including claude-memory lack. Against these strengths, Ember is weaker on curation quality (no human-in-the-loop review), retrieval breadth (no FTS), and consolidation intelligence (no LLM-driven sleep cycle). The README significantly overclaims on the sophistication of the drift detection pipeline — what ships is a two-threshold EMA check, not the Laplacian-smoothed multi-stage pipeline described. For claude-memory's purposes, the most portable insights are: the Shadow-Decay scoring formula, the `ember_explain` observability pattern, and the bootstrap scanner approach for cold-start seeding from existing Claude Code session history.
