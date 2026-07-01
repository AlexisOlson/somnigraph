# Jumbo — Goal-driven, event-sourced project-context orchestrator for coding agents (no semantic retrieval)

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

Jumbo (`jumbocontext/jumbo.cli`, TypeScript, AGPL-3.0, v3.11.0) is **not a retrieval-quality memory system**. It is a CLI that maintains a curated, event-sourced knowledge graph of *structured project artifacts* and assembles a deterministic context packet for whichever **goal** the coding agent is working on. The unit of "memory" is an architecture-decision-record-style entity, not a captured episodic snippet. All the "intelligence" (what to capture, dedup judgment, contradiction detection) is delegated to the coding agent via shipped skill prompts; the engine itself is deterministic CRUD over an event store plus a graph join.

### Storage & Schema
- **Event store as source of truth, SQLite views as read model (CQRS).** Every state change is an append-only domain event. `BaseEvent` (`src/domain/BaseEvent.ts`) carries `type`, `aggregateId` (stream id), `version` (optimistic concurrency), `timestamp`, `loggedBy: "human"|"machine"`, `payload`. Events are persisted per-aggregate on the **filesystem** (`src/infrastructure/context/**/Fs*EventStore.ts`, e.g. `FsDecisionAddedEventStore.ts`) and projected into SQLite read models via projectors (`Sqlite*Projector.ts`).
- **11 structured entity types** (`src/domain/relations/Constants.ts` `EntityType`): project, audience, pain, value, architecture, component, dependency, decision, guideline, invariant, goal (+ session, relation). Each has a rich typed schema — e.g. Decision stores `context`, `rationale`, `alternatives`, `consequences`, `status`, `supersededBy` (`DecisionView.ts`); Component stores name/type/description/responsibility/path/status.
- **Relations are first-class graph edges**: `fromEntityType/Id → toEntityType/Id`, plus `relationType` (free-text semantic label, ≤50 chars, e.g. "uses", "depends-on", "involves"), `strength` (strong/medium/weak), `status` (active/deactivated/removed), `description`. Not a fixed typed ontology like Somnigraph's supports/contradicts/evolves — the relation *type* is an arbitrary string set by the agent.
- **Global search index** is a separate projection (`search_index_entries`: sourceType, sourceId, category, title, summary, content, facets, metadata, version) built from entity events (`SqliteSearchIndexStore.ts`).

### Memory Types
Fixed taxonomy of project-knowledge categories (component / dependency / decision / guideline / invariant are the searchable ones; goal / architecture / project / audience / pain / value are contextual). Guidelines further carry a category enum (testing, coding style, process, communication, documentation, security, performance, other). This is a **project-ontology**, not an episodic/semantic/procedural cognitive taxonomy.

### Write Path
- **Agent-driven registration, not auto-extraction.** There is no NLP extraction pipeline. The coding agent explicitly runs `jumbo component add`, `jumbo decision add`, `jumbo relation add`, etc. The evidence file's "auto-extraction ❌" is correct.
- **Quality gating is prompt-level, in shipped skills.** `assets/skills/codify-jumbo-goal/SKILL.md` instructs the agent to capture only learnings that are **"Universal, Dense (one sentence), Actionable"** and to "avoid restating what's already captured." `refine-jumbo-goals/SKILL.md` curates relations before implementation. None of this is enforced in code — it is a rubric the LLM is asked to apply.
- **Dedup** is a name-equality check in command handlers (`AddComponentCommandHandler.ts` — same-name component is updated, not duplicated) plus idempotent relation-add, plus skill prompts telling the agent to "search before adding, update/supersede rather than duplicate." No embedding or fuzzy dedup.

### Retrieval
Two distinct paths, **neither semantic**:
1. **Primary path — deterministic goal-context assembly** (`SqliteGoalContextAssembler.assembleContextForGoal`). Given a goal id: fetch active relations from the goal, group `toEntityId`s by type, batch-fetch each entity type, merge with relation metadata. ~7 indexed queries, **no ranking, no scoring, no query** — it is a graph join returning everything relation-bound to the goal. This is the "context packet."
2. **Secondary path — `jumbo search`** (`SqliteSearchIndexStore.search`): case-normalized SQL `LIKE '%q%'` across title/summary/content, with a hardcoded `CASE` score (title match=30, summary=20, content=10, else 1), ordered by score then `updatedAt`. No vector, no BM25/FTS5, no RRF, no reranker, no embeddings anywhere in `src` (verified by grep — the only "embedding" hit refers to embedding data in an event payload).

### Consolidation / Processing
No offline sleep/merge/clustering engine. What the evidence file labels "clustering" is just relation-add + grouping-by-type in the assembler. **Consolidation is a workflow checkpoint, not a background job**: at goal close, the `codify` skill walks the agent through reviewing each entity category for staleness and capturing new learnings. Daemons exist (`work refine`, `work review`, `CodifierProcessManager`) but they *poll for goals and spawn an agent subprocess* — orchestration, not memory consolidation.

### Lifecycle Management
- **No decay/forgetting** (evidence "Decay ❌" correct). Nothing ages out.
- **Explicit lifecycle via event-sourced status transitions**: decisions `supersede`/`reverse`/`restore`; components `deprecate`/`undeprecate`/`remove`; relations `deactivate`/`reactivate`/`remove`; goals a full state machine (defined→refined→doing→in-review→approved→codifying→done, plus blocked/paused/rejected). Removal deletes from the active read model but **preserves event history** (full time-travel / audit via replay).
- **Contradiction detection** (evidence ✅) is prompt-driven: the QA `review` skill asks the agent to check the implementation against bound decisions and invariants and record issues — not an algorithmic contradiction classifier.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Goal-conditioned context: agent gets exactly the memory bound to the current goal | `SqliteGoalContextAssembler` graph join over active relations | **Validated** — deterministic and real, but precision depends entirely on the agent having wired correct relations |
| Event-sourced, full history / time-travel | `Fs*EventStore` append-only + `Sqlite*Projector` read models; removal preserves events | **Validated** — genuine CQRS/event-sourcing |
| Harness/model-agnostic (6 agent CLIs) | `AgentCliGateway` maps claude/gemini/copilot/codex/cursor/vibe; per-harness configurers write hooks | **Validated** |
| Write-path quality ("universal, dense, actionable") | Rubric text in `codify`/`refine` SKILL.md | **Plausible but unenforced** — it is an LLM instruction, not a code gate; quality = agent compliance |
| Search across memory | `LIKE '%q%'` with CASE score in `SqliteSearchIndexStore` | **Validated but primitive** — substring match, no relevance model |
| Semantic / hybrid / vector retrieval | none | **Absent** (evidence correctly marks ❌) |
| Benchmarks (LoCoMo/LME/PersonaMem) | none; `benchmarks/` is an internal replay-perf test; `evals/` are project-simulation scenarios | **None** — not comparable to Somnigraph's 85.1% LoCoMo QA |

---

## Relevance to Somnigraph

### What Jumbo does that Somnigraph doesn't
- **Goal/task-scoped context binding.** Memory is attached to an active goal via typed relation edges, and the delivered context is a *deterministic graph join on that goal*, not a ranked query over the whole store. Somnigraph has no notion of a "current task" that memories bind to — its graph (`scoring.py` PPR expansion) is query-time retrieval expansion, not a task-scoped assembly. This is a genuinely different retrieval philosophy: **curate-and-join** vs **rank-and-retrieve**.
- **Full event-sourcing / time-travel.** Every mutation is an append-only event with replay. Somnigraph uses `valid_from`/`valid_until` + archive (`db.py`), which is lighter but cannot reconstruct arbitrary past states.
- **Explicit goal state machine + orchestration daemons** driving the agent through plan→implement→review→codify. Somnigraph is a passive memory store; it does not orchestrate work.
- **Multi-harness portability** (claude/codex/gemini/copilot/cursor). Somnigraph is Claude-Code/MCP-only.

### What Somnigraph does better
- **Everything on the retrieval axis.** Somnigraph has hybrid BM25+vector RRF fusion, a 26-feature LightGBM reranker (NDCG 0.7958), PPR graph expansion, and a measured feedback loop (Spearman r=0.70). Jumbo's retrieval is SQL `LIKE` + a hardcoded 30/20/10 CASE score. No embeddings, no learned ranking, no fusion.
- **Offline LLM-mediated consolidation** (`sleep_nrem.py`/`sleep_rem.py`): real pairwise contradiction classification, edge creation, merge/archive, gap analysis. Jumbo's "consolidation" is a checkpoint prompt the agent runs at goal close.
- **Decay / biological lifecycle.** Somnigraph ages memories; Jumbo never forgets.
- **Measured retrieval quality.** Somnigraph reports LoCoMo QA numbers; Jumbo has no retrieval benchmark at all.

---

## Worth Stealing (ranked)

### 1. Goal/task-scoped deterministic context binding (Medium — revisit-if only)
**What**: Bind memories to an explicit active-task node via typed edges, and for that task assemble a *deterministic* context packet (graph join, no ranking) rather than always running a ranked query.
**Why**: Somnigraph's recall is always rank-and-retrieve; for high-precision, low-recall contexts (e.g. "what invariants govern the file I'm editing") a curated join can beat a reranker that must guess relevance. This is only relevant **if** Somnigraph ever grows a "current task/goal" concept — today it has none, so this is a note, not a task.
**How**: Would require a new `task`/`goal` aggregate in `db.py` and a task→memory edge, plus a `tools.py` path that returns task-bound memories deterministically alongside (not instead of) ranked recall. Large, product-shape change; low fit with the single-user always-on memory model.

### 2. Write-time "universal / dense / actionable" quality rubric (Low)
**What**: Jumbo's codify skill gates new memories on three prompt-level criteria — universal (not one-off), dense (one sentence), actionable (changes behavior) — and explicitly "don't restate what's already captured."
**Why**: This is the *same write-path-discipline thesis* the Phase 18 sweep landed on (ByteRover/agentmemory/MemPalace win on write quality, not retrieval). Jumbo operationalizes it as a checkpoint rubric. Somnigraph's CLAUDE.md snippet already has capture guidance, so this is **largely redundant** — but the terse three-word gate ("universal, dense, actionable") is a cleaner phrasing than the current snippet and could tighten it.
**How**: Copy-edit the capture section of the README snippet / `docs/claude-md-guide.md`. No code.

---

## Not Useful For Us

### Event-sourcing / CQRS for every entity
Full append-only event streams + per-aggregate filesystem stores + SQLite projectors is heavyweight infrastructure justified by multi-agent concurrency and audit needs Somnigraph doesn't have. Somnigraph's single-user valid_from/valid_until + archive is a deliberate, lighter choice. Architectural mismatch.

### `LIKE`-based search and the 30/20/10 CASE score
Strictly inferior to Somnigraph's FTS5+vector+RRF+reranker. Nothing to take.

### Goal state machine + orchestration daemons
Jumbo is a coding-workflow orchestrator (plan/implement/review/codify, multi-harness worker daemons). Somnigraph is a memory store, not an agent orchestrator. Out of scope.

---

## Connections

- **Corroborates the Phase 18 write-path thesis** (see `byterover.md`, `agentmemory.md`, `ai-memory-comparison.md`): Jumbo wins whatever value it has via *disciplined, agent-authored, structured writes* — not retrieval sophistication (which is near-zero). Independent arrival at "curate the write path" from a very different (project-ADR) angle.
- **Supersession pattern**: Jumbo's `decision supersede`/`supersededBy` mirrors the supersession lineage seen in `memv`/`memos` — explicit versioned replacement rather than deletion. Convergent with Somnigraph's `evolves`/`revision` edge types, but Jumbo does it as an event-sourced status transition rather than a detected edge.
- **Structured-entity / ontology memory** (vs free-text episodic): closest to systems that store typed project knowledge rather than conversational memory; contrasts sharply with Somnigraph's episodic/semantic snippet model.

---

## Summary Assessment

Jumbo's core contribution is a **stance, not a mechanism**: memory for a coding agent should be a small, curated, agent-authored knowledge graph of *structured project artifacts* (decisions, components, invariants, guidelines) bound to *goals*, delivered deterministically as a per-goal context packet — with event-sourcing for auditability and multi-harness portability. It deliberately has no semantic retrieval, no ranking, no decay, and no benchmarks, because it is betting that a well-curated small graph beats a large ranked store for the specific job of orienting a coding agent. On its own terms that is a coherent product; on Somnigraph's terms (retrieval quality) it is not a competitor and has essentially nothing to steal on the retrieval axis.

The single most useful thing for Somnigraph is the **confirmation, from a very different design lineage, that write-path discipline is the lever** — Jumbo encodes it as a terse codify-time rubric ("universal, dense, actionable; don't restate"). That is worth a copy-edit to the capture snippet, but it is redundant with what Phase 18 already established. The one idea that is genuinely *additive* rather than redundant is **goal/task-scoped deterministic binding** (curate-and-join as a complement to rank-and-retrieve), but it presupposes a "current task" concept Somnigraph doesn't have and would be a large product-shape change with poor fit to the always-on single-user model — so it is a revisit-if note, not an adoption.

The evidence file is unusually honest: it correctly marks vector/hybrid/semantic/decay and all benchmarks as ❌. The sharpest correction is a category one rather than a factual one: many of its ✅ cells under Data Model, Lifecycle, and Extraction (contradiction detection, clustering, quality refinement, narrative generation, content-aware preprocessing) are **realized as agent-prompt instructions in shipped skill markdown, not as algorithmic code** — the engine is deterministic CRUD-over-event-store plus a graph join, and every "smart" behavior is delegated to the coding agent. A reader scanning the checkmarks could mistake Jumbo for an intelligent memory pipeline; it is a disciplined scaffold that makes the *agent* do that work.
