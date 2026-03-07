# Empirica -- Source Analysis

*Phase 13, 2026-02-27. Analysis of Nubaeon/empirica.*

## 1. Architecture Overview

**Language:** Python 3.10+, synchronous core with subprocess CLI delegation. MCP server layer uses `mcp` (FastMCP). No async I/O in the epistemic middleware path itself.

**Key dependencies:**
- `pydantic >= 2.4.0` -- all data models and vector schemas
- `sqlalchemy >= 2.0` -- ORM for session/reflex/calibration tables
- `gitpython >= 3.1.41` -- git notes storage layer
- `qdrant-client` (optional) -- vector search for semantic retrieval across findings/unknowns/lessons
- `tiktoken >= 0.5.0` -- token counting for compression estimates
- `rich >= 13.0` -- terminal UI
- `typer >= 0.9` -- CLI framework
- `anthropic >= 0.39.0` -- Anthropic SDK integration
- `httpx >= 0.24` -- HTTP client
- `mcp >= 1.0.0` -- Model Context Protocol server (in `empirica-mcp` subpackage)
- Embedding providers: Jina, Voyage AI, Ollama, OpenAI (configurable)

**Storage layout:**
```
.empirica/
  sessions/
    sessions.db             # SQLite: reflexes, grounded_beliefs, calibration_trajectory, etc.
  sentinel_enabled          # Feature flag file
.empirica-project/
  project.yaml              # Project metadata + ID
.git/
  refs/notes/empirica/
    session/{sid}/{phase}/{round}   # Git notes: compressed checkpoints (~450 tokens)
    goals/{goal-id}/                # Goal metadata + epistemic state at creation
    handoff/{session-id}/           # Compressed handoff reports (~238 tokens)
.empirica_reflex_logs/
  {YYYY-MM-DD}/{agent_id}/{session_id}/   # Full JSON audit logs (~6,500 tokens)
localhost:6333 (Qdrant, optional)
  project_{id}_docs          # 1536-dim doc embeddings
  project_{id}_memory        # Findings, unknowns, mistakes, dead-ends
  project_{id}_epistemics    # Learning trajectories with epistemic deltas
  global_learnings           # Cross-project high-impact findings (>= 0.7 impact)
  empirica_lessons           # Procedural knowledge, 384-dim embeddings
  personas                   # Agent profiles, 13-dim or 1536-dim embeddings
```

**File structure (key source files):**
- `empirica/cli/cli_core.py` -- Main CLI entry point, 55+ commands
- `empirica-mcp/empirica_mcp/server.py` -- MCP server, 55 registered tools, routes to CLI subprocess or direct handlers
- `empirica-mcp/empirica_mcp/epistemic_middleware.py` -- Wraps MCP tool calls with assess -> route -> execute pipeline
- `empirica-mcp/empirica_mcp/epistemic/state_machine.py` -- 13-vector state tracking with conditional adjustments
- `empirica-mcp/empirica_mcp/epistemic/router.py` -- VectorRouter: priority-based mode selection from 4 vector thresholds
- `empirica-mcp/empirica_mcp/epistemic/personality.py` -- 4 personality profiles that bias routing thresholds
- `empirica-mcp/empirica_mcp/epistemic/modes.py` -- 5 execution modes (load_context, investigate, confident/cautious_impl, clarify)
- `empirica-mcp/empirica_mcp/epistemic/shared_vector_semantics.py` -- Shared vector definitions, readiness assessment, tier grouping
- `docs/architecture/` -- 20+ architecture docs (many aspirational)

**Scale:** v1.5.9 (2026-02-26). Very active development -- changelog shows near-daily releases through February 2026. MIT license. Author: David S. L. Van Assche.

---

## 2. Memory Type Implementation

**Schema:** The core knowledge unit is **not a memory** in the traditional sense. Empirica tracks two things:

1. **Epistemic checkpoints** (the `reflexes` table): 13 floating-point vectors (0.0-1.0) captured at PREFLIGHT, CHECK, and POSTFLIGHT phases. Each row stores `session_id, cascade_id, phase, round, timestamp, engagement, know, do, context, clarity, coherence, signal, density, state, change, completion, impact, uncertainty, reflex_data (JSON), reasoning, evidence`. This is the primary data type.

2. **Learning artifacts** (the ledger): Four categories logged during sessions:
   - `findings` -- confirmed knowledge
   - `unknowns` -- identified gaps
   - `dead_ends` -- paths that failed
   - `mistakes` -- errors made

These are logged via `finding_log`, `unknown_log`, `deadend_log`, `mistake_log` CLI/MCP commands, stored in both SQLite and (optionally) Qdrant vector collections.

**Types/categories:** Learning artifacts are classified by the four ledger categories above. Additionally, there are `assumption_log` and `decision_log` for structured reasoning capture. The system does not use episodic/semantic/procedural taxonomy.

**Extraction:** The LLM is responsible for self-assessment. The framework provides schemas as "guidance, not enforcement" (`validate_input=False`). Epistemic vectors are self-reported by the AI agent, not computed from observable data. The `grounded_calibration` tables (v1.5.0) attempt to anchor self-reported vectors against objective evidence (test results, git metrics, file coverage), but this is a verification layer, not an alternative extraction path.

---

## 3. Retrieval Mechanism

Retrieval in Empirica is fundamentally different from traditional memory systems. There is **no query-time semantic search over memories** in the core path. Instead:

1. **Session bootstrap** (`project-bootstrap`): Loads prior session findings, unknowns, and handoff reports at session start. This is a structured load, not a similarity search. Compressed to ~450 tokens per checkpoint.

2. **Qdrant semantic search** (optional): When enabled, supports similarity queries over findings, unknowns, dead-ends, and lessons across projects. Uses cosine similarity (768-1536 dim depending on embedding provider). Collections are auto-populated from logging events.

3. **Handoff reports**: Compressed session summaries (~238 tokens) capturing epistemic deltas, key findings, remaining unknowns. Stored in git notes. This replaces full context replay.

4. **Git notes traversal**: Session history queryable by session ID, phase, or round through git note refs.

There is **no RRF, no hybrid search, no multi-signal fusion** in the retrieval pipeline. Qdrant provides single-mode vector similarity when available. The system's strength is not retrieval -- it's the *assessment framework that tells you whether you need to retrieve*.

---

## 4. Standout Feature

**The 13-vector epistemic self-assessment framework with phase-gated execution.**

This is genuinely novel in the AI memory/agent tooling space. The core idea:

Rather than storing and retrieving memories, Empirica quantifies what the agent *knows it knows* across 13 dimensions, then uses those scores to gate execution:

**Epistemic Readiness** = `(know + context + (1 - uncertainty)) / 3`

**Action Momentum** = `(do + change + completion) / 3`

The CASCADE workflow enforces:
- **PREFLIGHT**: Baseline vector snapshot before work begins
- **CHECK gate**: Must pass `know >= 0.70 AND uncertainty <= 0.35 AND coherence >= 0.65` to proceed from investigation (Noetic) to action (Praxic)
- **POSTFLIGHT**: Delta measurement -- `learning_delta = postflight_vectors - preflight_vectors`

The phase separation is the key insight: the system loops in Noetic (investigation) mode while `uncertainty > 0.5 OR know < 0.70`, then crosses to Praxic (execution) only when the gate passes. This prevents the common failure mode of agents charging into implementation with insufficient understanding.

**Personality-modulated thresholds** encode different philosophical stances toward uncertainty:
- `CAUTIOUS_RESEARCHER`: uncertainty tolerance 0.4, know threshold 0.8
- `PRAGMATIC_IMPLEMENTER`: uncertainty tolerance 0.7, know threshold 0.6
- `ADAPTIVE_LEARNER`: starts moderate, adjusts thresholds based on outcomes

**Phase-aware calibration** creates earned autonomy: `threshold = base_threshold - (calibration_accuracy * autonomy_factor)`. An agent well-calibrated in security investigation earns a looser gate for security tasks, independent of its calibration in other domains.

**Honest caveat:** The vectors are self-reported by the LLM, not computed from objective data. The `grounded_calibration` system (comparing self-reports to evidence) is a mitigation, but the fundamental signal is still the agent's own assessment of its knowledge. This is both the system's strength (it's easy to adopt) and weakness (it can only be as honest as the agent).

**Merge scoring for multi-agent results:**
```
merge_score = (learning_delta * quality * confidence) / cost_penalty
```

This quantifies the value of an investigation branch, enabling rational triage of parallel agent work.

---

## 5. Other Notable Features

**5.1 Four-layer storage with extreme compression.** The SQLite-hot / Git-warm / JSON-audit / Qdrant-search architecture achieves 97% token reduction (15,000 -> 450 tokens) for checkpoints. The handoff system compresses to ~238 tokens while preserving epistemic deltas. This is a real engineering achievement for context-window management.

**5.2 Drift detection and confabulation prevention.** The system classifies epistemic drift into three patterns: TRUE_DRIFT (actual knowledge loss), LEARNING (knowledge decrease + clarity increase), and SCOPE_DRIFT (task expansion). The MemoryGapDetector compares claimed knowledge against logged breadcrumbs to catch confabulation. Five gap categories: unreferenced findings, unincorporated unknowns, file unawareness, memory compaction, confabulation. Graduated enforcement: inform -> warn -> strict -> block.

**5.3 EpistemicBus event system.** Pub/sub architecture for internal coordination. Components publish domain events about epistemic state changes (phase transitions, drift alerts, calibration changes, memory pressure). Observers react independently. Synchronous and simple -- errors in observers don't block. Optional: the system works without it.

**5.4 Findings deprecation engine.** Learning artifacts have lifecycle management -- automatic or manual deprecation based on evidence age, supersession, or contradiction. This is a lightweight form of temporal validity tracking.

**5.5 Multi-agent orchestration.** `investigate-multi` spawns parallel persona-specific investigations with independent epistemic tracking. Results merge using epistemic-score weighting. Qdrant stores agent profiles with pre/post vector states and reputation scores. This is architecturally ambitious; unclear how much real-world multi-agent usage it sees.

---

## 6. Gap Ratings

| Gap | Rating | Evidence |
|-----|--------|----------|
| Layered Memory | 35% | Four storage *layers* (SQLite/Git/JSON/Qdrant) but these are redundant persistence tiers, not semantic abstraction tiers (gestalt/summary/detail). No automatic summarization or compression of memory content itself. Compressed checkpoints (~450 tokens) are a fixed format, not a summary of richer detail. |
| Multi-Angle Retrieval | 25% | Qdrant provides single-mode vector similarity when available. SQLite enables structured queries by session/phase/round. Git notes enable ref-based lookup. But there is no hybrid fusion (no RRF, no FTS5, no graph traversal in the retrieval path). Each storage layer is queried independently. |
| Contradiction Detection | 55% | Drift detection classifies TRUE_DRIFT vs LEARNING vs SCOPE_DRIFT. MemoryGapDetector catches confabulation against logged evidence. Findings deprecation tracks supersession. Grounded calibration compares self-reports to objective evidence. But no explicit contradiction edges between memories, no graded contradiction scoring, no temporal contradiction tracking. |
| Relationship Edges | 15% | No explicit relationship/edge data structure between memories or findings. Qdrant enables semantic proximity discovery, but there are no typed, traversable edges. Agent-to-agent relationships tracked via reputation scores, but this is not memory-to-memory. |
| Sleep Process | 10% | No consolidation, no offline processing, no memory synthesis. The handoff system compresses context for session transfer, but this is event-driven (session end), not a background maintenance process. No decay, no pruning, no cluster-based synthesis. |
| Reference Index | 30% | `get_calibration_report` provides per-session epistemic analytics. `memory_stats` equivalent exists in calibration trajectory tables. But no global reference index, no portfolio-level overview across all stored knowledge. Goal progress tracking provides partial coverage. |
| Temporal Trajectories | 60% | The `calibration_trajectory` table tracks historical calibration evolution. Learning deltas (PREFLIGHT vs POSTFLIGHT) are stored per session, enabling "Session 1: know 0.4->0.65; Session 2: know 0.7->0.85" trajectories. Cross-session knowledge compounding is explicitly designed. But this tracks *self-assessed vector evolution*, not belief-level temporal tracking. |
| Confidence/UQ | 80% | This is Empirica's core competency. 13-dimensional uncertainty quantification with phase gates, personality-modulated thresholds, grounded calibration against evidence, earned autonomy, domain-specific calibration accuracy, drift detection. The only weakness: vectors are self-reported, not computed from objective evidence (grounded calibration partially mitigates). |

---

## 7. Comparison with claude-memory

**Stronger:**
- Uncertainty quantification is the central design primitive, not a bolted-on field. The 13-vector framework with phase gates is architecturally deeper than our single `confidence` column (0.1-0.95).
- Phase separation (Noetic/Threshold/Praxic) enforces "investigate before acting" -- our system has no execution gating.
- Calibration trajectories track self-assessment accuracy over time, with earned autonomy. Our `confidence` compounds through feedback but doesn't track calibration accuracy itself.
- Extreme compression (97% reduction for checkpoints, 238-token handoffs) -- our startup_load is parsimonious but doesn't achieve this level of compression.
- Multi-agent coordination with epistemic profiles, reputation scoring, and merge scoring for parallel investigations. We are single-agent.
- Domain-specific calibration -- accuracy tracked per domain, gates adjust independently. Our confidence is global.
- Confabulation detection via MemoryGapDetector comparing claims against breadcrumb evidence. We have no equivalent.

**Weaker:**
- No persistent memory in the traditional sense. Empirica tracks *what you know about what you know*, not *what you know*. Our system stores 282 memories with content, embeddings, edges, and multi-dimensional metadata. Empirica stores epistemic vectors and learning artifacts.
- No hybrid retrieval. Our RRF fusion (vector + FTS5) with novelty-scored edge expansion significantly outperforms Empirica's single-mode Qdrant or structured SQL queries.
- No relationship edges or graph structure. Our typed edges with contradiction/revision/derivation flags and adjacency expansion have no analog.
- No sleep/consolidation process. Our three-pipeline sleep system (NREM standard, NREM deep, REM) for offline maintenance, synthesis, and clustering is absent entirely.
- No decay model. Our power-law decay with per-memory rates, shadow load penalty, and temporal scoring has no equivalent. Empirica's findings deprecation is manual/rule-based, not mathematical.
- No human-in-the-loop curation. Our `review_pending()` with auto-capture and explicit confirmation has no analog. Empirica's quality gate is the CHECK threshold, not human review.
- No gestalt layer. Our gestalt summaries that compress memory clusters into overview narratives have no equivalent.
- Qdrant dependency for semantic search adds operational complexity vs our embedded sqlite-vec.
- Self-reported vectors are fundamentally limited by LLM honesty. Our confidence gradient compounds through observed feedback, not self-assessment.

---

## 8. Insights Worth Stealing

1. **Phase-gated execution** (effort: medium, impact: high). The Noetic/Praxic separation with CHECK gates (`know >= 0.70 AND uncertainty <= 0.35`) is a powerful pattern. We could implement this not for memory storage but for memory *confidence*: before a memory graduates from `pending` to `confirmed`, it must pass an evidence gate. This maps to our existing `review_pending()` flow but adds quantitative thresholds.

2. **Calibration accuracy tracking** (effort: medium, impact: high). Track how well our confidence scores predict actual memory utility (via recall_feedback). Compute `calibration_accuracy = 1.0 - mean_divergence` over the last N feedback cycles. This meta-metric tells us whether our confidence model is learning. If confidence is well-calibrated, we can trust it more in retrieval scoring; if poorly calibrated, weight it less.

3. **Epistemic readiness formula** (effort: low, impact: medium). `readiness = (know + context + (1 - uncertainty)) / 3` is a simple composite that could inform our `startup_load` token budget. High readiness = smaller budget needed. Low readiness = load more context. We already have the inputs: confidence (know), recency of recall (context proxy), and shadow_load (uncertainty proxy).

4. **Confabulation detection via breadcrumb comparison** (effort: high, impact: medium). Comparing claimed knowledge against logged evidence to catch unsupported assertions. We could implement a lightweight version: when recalling memories, check if the recall query's domain has any contradicted or high-shadow memories, and surface a warning.

5. **Handoff compression** (effort: low, impact: medium). The 238-token handoff format -- epistemic deltas, key findings, remaining unknowns -- is a useful template for our gestalt summaries. Our gestalt layer already does compression, but the "delta-first" framing (what changed, not what is) could make gestalt summaries more useful for session-start briefings.

6. **Domain-specific calibration** (effort: high, impact: medium). Tracking calibration accuracy per theme/topic rather than globally. If our confidence scores are well-calibrated for `work` memories but poorly calibrated for `skyrim` memories, retrieval scoring should weight confidence differently per domain. Requires enough recall_feedback volume per domain to be statistically meaningful.

---

## 9. What's Not Worth It

**The 13-vector framework itself.** The dimensionality is impressive on paper, but in practice the vectors are self-reported by the LLM and the state machine uses simple conditional adjustments (+0.2 for precision language, -0.2 for vague phrasing). The theoretical foundation (engagement, clarity, coherence, signal, density, state, change, completion, impact) is philosophically interesting but empirically unvalidated -- we have no evidence that 13 dimensions perform better than our single confidence float with feedback-driven compounding. Adopting this would add complexity without proven benefit.

**The CASCADE workflow.** PREFLIGHT/CHECK/POSTFLIGHT is a good idea for agentic task execution but is orthogonal to memory storage and retrieval. It's a workflow framework, not a memory architecture. Integrating it would require restructuring how sessions work for marginal memory-system benefit.

**Multi-agent orchestration.** Our system is single-agent by design. The agent spawning, reputation tracking, and merge scoring are interesting but irrelevant to our architecture. If we ever need multi-agent coordination, this is the wrong layer to add it.

**The git notes storage layer.** Clever for portability and distributed access, but we already have SQLite + embeddings which is simpler, faster to query, and doesn't require a git repo. Git notes add operational complexity without clear benefit for a personal memory system.

**The EpistemicBus event system.** Pub/sub is useful for loosely-coupled multi-component systems. Our architecture is a single Python server with direct function calls. Adding an event bus would be over-engineering.

---

## 10. Key Takeaway

Empirica and claude-memory solve fundamentally different problems. Empirica is a *metacognitive framework* -- it helps an agent know what it knows, gate its actions on that knowledge, and measure its learning. claude-memory is a *persistent knowledge store* -- it helps an agent remember, retrieve, connect, and maintain what it has learned. Empirica's standout contribution is the formalization of epistemic uncertainty as a first-class dimension with phase-gated execution, and the insight that self-assessment accuracy (calibration) should itself be tracked and used to modulate behavior. The concrete takeaway for our system is not the 13-vector framework (too much self-report, too little empirical validation) but the *meta-metric*: tracking how well our confidence scores predict actual utility, and using that calibration accuracy to adjust how much weight confidence gets in retrieval. Our confidence gradient already compounds through feedback; what it lacks is a measure of whether that compounding is *accurate*.
