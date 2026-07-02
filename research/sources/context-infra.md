# context-infra — Prompt-driven markdown observation log with cron-fired LLM observer/reflector agents

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

context-infra (`grapeot/context-infrastructure`) is explicitly a **reference implementation / blueprint**, not a tool ("这不是开箱即用的工具，而是一个可以参考的蓝图"). The "memory system" is a three-tier markdown hierarchy maintained by two cron-fired LLM agents. Almost all intelligence is delegated to an LLM (OpenCode/Claude) via prompt SOPs; the Python code is thin trigger scripts plus one brute-force semantic-search utility.

### Storage & Schema
Plain Markdown files on disk. No database.
- **L1/L2** live in `contexts/memory/OBSERVATIONS.md` — an append-only log. Each entry is a `Date: YYYY-MM-DD` header followed by single-line records tagged with a traffic-light emoji: `🔴 High: [Category] desc`, `🟡 Medium`, `🟢 Low`. The four "schema fields" the evidence file counts (Date, Priority, Category, Content) are just this markdown convention, not typed columns.
- **L3** is `rules/` — `SOUL.md` (agent identity), `USER.md` (user profile), `COMMUNICATION.md`, `WORKSPACE.md`, plus `axioms/` (43 distilled decision principles) and `skills/` (25+ workflow docs). Loaded every session (passive context), not retrieved.

### Memory Types
Tiers by durability, encoded in the traffic-light tags (`KNOWLEDGE_BASE.md` §3.1):
- 🔴 = cross-project methodology / hard constraints / architecture decisions — permanent, candidate for promotion to axiom/skill.
- 🟡 = active-project state, technical tradeoffs relevant "for the next few weeks."
- 🟢 = daily task flow, transient debug notes — periodically garbage-collected.

### Write Path
`periodic_jobs/ai_heartbeat/src/v0/observer.py` is a cron trigger. It (1) does an idempotency check (`if "Date: {target_date}" in OBSERVATIONS.md: skip`), then (2) creates an OpenCode session and sends `PROMPT_TEMPLATE`, instructing the agent to scan root-dir file changes (`find . -mtime -1`), filter noise, and **append** observations directly to disk (`echo ... >> OBSERVATIONS.md`). The agent does the extraction/classification; the script only fires it and deletes the ephemeral session.
- Noise filtering is prompt-level and content-type-aware: blog files are only counted as new if their Markdown `Date` header is current, so 300+ reformatted blog posts don't create false "new content" observations (`PRD.md` §1.2, `KNOWLEDGE_BASE.md` §2.2). Path whitelist/blacklist (`daily_records/` ignored, `life_record/*.csv` included).
- **No deduplication** beyond the per-date idempotency guard. No embedding, no salience scoring, no structured enrichment.

### Retrieval
Two documented modes, neither integrated:
1. **grep** over OBSERVATIONS.md (`grep -n "关键词"`, `grep -A 20 "Date: ..."`) — the primary/default path.
2. **Semantic search** (`tools/semantic_search/`): a standalone CLI. `MarkdownChunker` chunks files, `EmbeddingClient` hits a local embedding server (`text-embedding-qwen3-embedding-8b` at `localhost:1234`), `ForwardIndex` caches chunk vectors to `.knowledge_cache` (pickle + `.npy` mmap, mtime-invalidated). Search in `search/cli.py` is **brute-force NumPy cosine** (`norm_embeddings @ norm_query`, `np.argsort`, top-k). No BM25, no fusion, no reranking, no graph. dim=4096.

The L3 rules are not retrieved at all — they are statically loaded into every session's context.

### Consolidation / Processing
`reflector.py` — a weekly cron trigger that fires an LLM agent (SOP in `KNOWLEDGE_BASE.md` §4.2) to: (1) read 🔴/high-🟡 entries, (2) **promote** generalizable ones to the right `rules/` file by responsibility boundary (identity→SOUL, user traits→USER, comms→COMMUNICATION, methodology→skills/), and (3) **GC**: rewrite OBSERVATIONS.md, deleting promoted content and expired 🟢 entries. Promotion gate (prompt-stated): "cross-project general + verified multiple times + clear applicable scenario." Role isolation is deliberate — observer never touches `rules/`, so no unreviewed rule changes leak in during observation (§5).

### Lifecycle Management
Binary, schedule-based GC only — the weekly reflector deletes expired 🟢 entries and promoted content. No gradual decay, no supersession marking, no versioning, no contradiction handling, no time-travel. The evidence file correctly rejects `decay`/`supersede` on the grounds that this is scheduled deletion, not relevance decay.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Three-tier layered memory (L1/L2/L3) | PRD.md, AGENTS.md, dir structure | Validated — but tiers are markdown files + prompt conventions, not a data model |
| Auto-extraction of memories | observer.py fires LLM agent via cron | Validated as *behavior*; the code doesn't extract — it prompts an agent to |
| Quality refinement / promotion to rules | reflector.py + KNOWLEDGE_BASE §4.2 | Validated as prompt-driven; no deterministic pipeline, entirely LLM judgment |
| Persona extraction (updates USER.md) | reflector prompt updates rules/USER.md | Plausible — depends on LLM following the SOP; no verification loop |
| Content-type-aware noise filtering | PRD §1.2 blog metadata Date check | Validated as a documented prompt rule; not enforced in code |
| Semantic + full-text search | semantic_search CLI + grep | Validated but primitive — brute-force cosine + shell grep, unfused |
| Benchmarks (LoCoMo/LME/etc.) | none | Absent — no scores, explicitly a blueprint not a benchmarked system |

---

## Relevance to Somnigraph

### What context-infra does that Somnigraph doesn't
- **Write-time, content-type-aware noise filtering.** The observer distinguishes genuinely new content from formatting churn (blog `Date` header check) before recording anything. Somnigraph has no write-path quality/salience gating (`tools.py` `impl_remember` stores what it's given) — a gap the corpus keeps flagging (Phase 18 ByteRover/agentmemory finding). context-infra's version is domain-specific and prompt-level, but the *placement* (filter before write) is the point.
- **Promotion into an always-loaded tier.** Durable, cross-context patterns graduate (L1→L2→L3) out of the retrieval pool into `rules/` that load every session. Somnigraph has a gestalt layer but auto-loads nothing; everything competes in retrieval. This is a different answer to "how does the most important knowledge reach the agent" than Somnigraph's proactive-injection.md hint design.

### What Somnigraph does better
Nearly everything on the retrieval/consolidation axis. Somnigraph has hybrid BM25+vector with RRF fusion and a 26-feature learned reranker (`reranker.py`) vs context-infra's brute-force cosine + grep. Somnigraph has a typed graph with PPR expansion; context-infra has none. Somnigraph's sleep pipeline (`sleep_nrem.py`/`sleep_rem.py`) does pairwise edge classification, merge/archive, gap analysis, and taxonomy — a superset of reflector's promote-and-GC. Somnigraph has an explicit feedback loop with measured GT correlation (r=0.70); context-infra has no feedback signal at all. Somnigraph has graded exponential decay with reheat vs binary scheduled deletion.

---

## Worth Stealing (ranked)

### 1. Promotion into an always-loaded tier, gated on durability + cross-context verification (Medium)
**What**: A distinct "graduated" memory tier that is loaded into every session's context (not retrieved), populated only by memories that clear a "general across contexts + verified multiple times" bar during offline consolidation.
**Why**: Somnigraph currently makes even its most durable, universally-relevant memories win a retrieval competition every query. context-infra's insight is that a small curated set earns unconditional presence. This is adjacent to — and a possible complement to — `docs/proposals/proactive-injection.md`: instead of (or alongside) floor-gated hints, a tiny always-on tier for the highest-durability items.
**How**: Sleep (`sleep_rem.py`) already computes taxonomy/gestalt; add a promotion pass that flags memories with high durability + high cross-session retrieval breadth into a `gestalt`/`always_load` set, surfaced by the proactive-injection hook without a floor gate. Bounded size to control token cost. Note the risk Somnigraph already respects: an always-loaded tier can re-bloat (the core.md re-bloat dynamic) — cap it hard.

---

## Not Useful For Us

### Brute-force cosine + grep retrieval
A strict downgrade from Somnigraph's fused, reranked, graph-expanded retrieval. Nothing to take.

### Cron-fired ephemeral LLM agents as the write mechanism
context-infra delegates all extraction/classification to an OpenCode agent per-run via prompt. Somnigraph's MCP-tool write path is more controllable and testable; moving intelligence into an opaque prompt SOP is the opposite direction from the corpus's write-path-discipline conclusion.

### Traffic-light (🔴🟡🟢) tiering
Redundant with Somnigraph's priority 1-10 + per-category decay, which is finer-grained and drives real math rather than a GC boolean.

---

## Connections

- **Convergent with the Phase 18 write-path finding** (ByteRover, agentmemory, MemPalace): the corpus keeps concluding that *write-time discipline* beats retrieval sophistication. context-infra is another data point — its only genuinely differentiated mechanic is write-time noise filtering, and its retrieval is deliberately trivial.
- **Contrasts with `docs/proposals/proactive-injection.md`**: both address "get the important thing in front of the agent without it asking," but context-infra answers with a static always-loaded rules tier (promotion) rather than a gated per-turn hint.
- **Same family as other markdown-log/prompt-SOP systems** in the sweep (append-only observation logs promoted to rules) — the L2→L3 promote-and-GC cadence is a poor-man's sleep consolidation.

---

## Summary Assessment

context-infra's core contribution is **organizational, not algorithmic**: a legible three-tier markdown hierarchy (daily observations → weekly-distilled rules → always-loaded L3) maintained by two cron-fired LLM agents whose behavior lives entirely in prompt SOPs (`KNOWLEDGE_BASE.md`). It is honestly framed as a blueprint you clone to *see the shape* of a year-old context system, explicitly not a reusable engine — "no shortcuts, you must collect your own behavioral data." As a memory *system* it is thin: append-only log, brute-force cosine search, grep, and schedule-based deletion.

The single most useful idea for Somnigraph is the **promotion-into-an-always-loaded-tier** pattern — graduating the most durable, cross-context memories out of the retrieval competition and into unconditional session context, gated on repeated verification. That is a real design lever Somnigraph hasn't pulled (its gestalt layer exists but auto-loads nothing), and it dovetails with the proactive-injection work. Everything else is either strictly weaker than existing Somnigraph modules or a domain-specific prompt trick.

The evidence file's 16 corrections are individually defensible but collectively **overstate the system**: `autoExtract`, `qualityRefine`, `narrative`, `recurrence`, and `persona` are all marked ✅ on the strength of *prompt instructions to an LLM agent*, not implemented pipeline code. The sharpest correction to carry forward: these are capabilities the SOP *asks an LLM to perform each run*, with no deterministic code, dedup, verification, or feedback behind them — so they are not comparable to the same-named mechanisms in systems that implement them, and none of it is benchmarked.
