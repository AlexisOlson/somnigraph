# Acontext — Agent memory as editable Markdown "skill" files, written by an LLM agent, retrieved by grep + progressive disclosure

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

Acontext (memodb-io) is a cloud/self-hosted **skill-memory service**, not a retrieval-ranking memory. The thesis (README): "Skill is All You Need" — every piece of agent memory is a plain Markdown file in Anthropic's Agent-Skill format (`SKILL.md` + data files), inspectable/editable/git-grep-able, and reused across agents by giving the agent `get_skill`/`get_skill_file`/`search_skills` tools. Retrieval is "progressive disclosure, not search": the *consuming* agent decides what to read; there is no semantic top-k. Stack (from `AGENTS.md`): Go/Gin/GORM API + Python/FastAPI core + PostgreSQL + pgvector + Redis + RabbitMQ + S3, plus a sandbox and a virtual "Disk".

### Storage & Schema
- Skills stored as S3-backed **artifacts** (files) grouped under an `AgentSkill` ORM row (`schema/orm/agent_skill.py`): `name`, `description`, `disk_id`, `user_id`, `meta` (JSONB). The actual knowledge lives in the Markdown files on the disk, not in DB columns.
- `SKILL.md` requires YAML front matter with `name` + `description` (parsed in `service/data/agent_skill.py::_parse_skill_md`). Everything else (structure, per-file layout) is LLM/user-defined.
- Other domains: `Session` (raw messages), `Task` (`task_description`, `status` pending/running/success/failed, `progresses[]`, `user_preferences[]`, `order`), `LearningSpace` (binds sessions↔skills). Optional per-object encryption with a user KEK (base64 → AES); the code hard-fails rather than store plaintext.

### Memory Types
Three distilled outcome types (`llm/tool/skill_learner_lib/distill.py`): **SOP** (success → reusable procedure), **Warning/anti-pattern** (failure → symptom/root-cause/correct-approach/prevention), **Fact** (people/preferences/entities, third-person). Plus a `skip_learning` outcome for trivia. User-preference facts get their own routing branch. No episodic/semantic/graph typing beyond this.

### Write Path (the substance of the system)
Two-stage async pipeline over RabbitMQ (`service/skill_learner.py`, `service/controller/skill_learner.py`, `llm/agent/skill_learner.py`):
1. **Task extraction** (`llm/agent/task.py`): an LLM agent watches the message stream and CRUDs Tasks (insert/update/append/progress), detecting task boundaries and marking success/failure against optional `task_success_criteria`. Task completion is the *trigger* for learning — not per-message.
2. **Distillation** (single LLM call, `process_context_distillation`): given a finished task + its messages, the model calls one of `skip_learning` / `report_success_analysis` / `report_factual_content` / `report_failure_analysis`. Success/failure tools **require an `applies_when` field** ("the website, tool, API, service, environment that scopes this learning — do NOT over-generalize"). This is an explicit anti-over-generalization salience gate at write time.
3. **Skill agent** (`skill_learner_agent`, up to `max_iterations`, Redis-locked per learning space): a tool-using loop that reads existing skills (`get_skill`/`get_skill_file`), `report_thinking`s, then decides **update vs create** via a hard decision tree — "existing skill covers/partially covers the domain → update it; zero coverage → create a category-level skill; never create narrow single-purpose skills." Edits via `str_replace_skill_file`/`create_skill`/`create_skill_file`/`mv_skill_file`. Concurrency: a per-space Redis lock serializes writers; contexts that arrive mid-run are drained from a pending queue and injected as follow-up messages (batched consolidation).

### Retrieval
No ranking model. Surfaces: (a) `get_skill` = name→file-list lookup from an in-memory dict; (b) `get_skill_file` = read one file by path; (c) `acontext_search_skills` (MCP, `packages/claude-code/src/mcp-server.ts`) = literal **grep/regex over skill Markdown** via `bridge.grepSkills`, first-N by iteration order, **no scoring/fusion/rerank**; (d) `acontext_session_history` = recent task summaries. The consuming agent's reasoning is the "ranker."

### Consolidation / Processing
LLM-mediated but **at write time**, not an offline sleep cycle. The update-vs-create decision tree + mid-run batch draining is the only consolidation. No pairwise edge detection, no graph, no periodic re-processing of the whole store.

### Lifecycle Management
`explicit delete` only (`skills.delete`, `learning_spaces.exclude_skill`). **No decay, no supersession chain, no contradiction detection, no TTL/archival, no versioning.** Skills are overwritten in place.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| "No embeddings, no semantic search — progressive disclosure" | Confirmed: zero `get_embedding` call sites in the Python core outside the `embeddings/` module; retrieval is grep + name lookup | validated |
| Automatic learning from agent runs, zero instrumentation | Task-extraction agent + distillation pipeline over MQ, triggered on task success/failure | validated (code is real and non-trivial) |
| Memory is human-inspectable/portable (plain Markdown, export as ZIP) | Skills are S3 Markdown artifacts; genuine transparency win vs opaque vector DBs | validated |
| "Deduplication ✅" (evidence file) | Refers to S3 **object** content-addressing dedup (disabled under encryption), not memory-entry dedup | misleading — see cross-check |
| Prevents over-generalized memories | `applies_when` required field + "do NOT over-generalize" + "never create narrow single-purpose skills" decision tree | plausible, prompt-enforced only (no eval) |
| Quality/retrieval benchmark performance | **None reported** — no LoCoMo/LME/QA numbers anywhere in repo | not benchmarked |

---

## Relevance to Somnigraph

### What Acontext does that Somnigraph doesn't
- **Write-path outcome triage + salience gate.** Distillation forces the model to pick success/failure/fact/**skip**, and to fill `applies_when` scoping. Somnigraph's `tools.py::remember()` has no write-time quality/salience gate or scope-condition field — this is exactly the "write-path quality gating" gap named in the brief.
- **First-class failure/anti-pattern memory.** Structured `symptom / root_cause / correct_approach / prevention` schema for lessons-from-failure. Somnigraph's categories (episodic/semantic/procedural/reflection/meta) have no dedicated counterfactual failure type.
- **Write-time anti-fragmentation routing** (update-existing-broad-skill vs create-narrow) — proactive consolidation before storage, complementary to Somnigraph's *post-hoc* merge during `sleep_nrem.py`.
- **Transparency/portability** — Markdown files a human can read/edit/git; Somnigraph's SQLite blob is opaque to the user.

### What Somnigraph does better
- **Retrieval quality is the entire point** and Acontext has none: no BM25/vector fusion (`fts.py`+RRF), no learned reranker (`reranker.py`, NDCG 0.7958), no feedback loop, no graph/PPR (`scoring.py`). At scale, "grep the Markdown, agent decides" has no relevance ranking — the exact ceiling Somnigraph's 26-feature reranker was built to raise.
- **Offline LLM-mediated sleep** (NREM edge detection, REM gap analysis) vs Acontext's write-time-only consolidation.
- **Decay/lifecycle** — Somnigraph has per-category exponential decay + reheat; Acontext has none.
- **Measured benchmarks** — 85.1% LoCoMo QA vs Acontext's zero reported numbers.

---

## Worth Stealing (ranked)

### 1. `applies_when` scope-condition field on procedural memories (Medium)
**What**: Require every procedural/how-to memory to carry an explicit scope string ("applies when: on `flower-sunshine.com`, using the X API") and instruct extraction to refuse over-generalization.
**Why**: Somnigraph's multi-hop ceiling is a vocabulary/precision problem; over-broad procedural memories retrieve for the wrong query. A scope field is both a write-time precision gate and a candidate reranker feature (does query context match `applies_when`?).
**How**: Add an optional `scope`/`applies_when` field to the memory schema (`db.py`), surface it in the `remember()` guidance (`tools.py`), and expose a query↔scope match signal to `reranker.py`.

### 2. Outcome-typed distillation with a `skip_learning` gate (Medium)
**What**: When capturing procedural knowledge, classify success-SOP vs failure-anti-pattern vs fact vs skip-trivial, each with a fixed field schema.
**Why**: Independent corroboration of the Phase 18 write-path thesis (agentmemory write-time grounding, ByteRover). A cheap typed template raises stored-memory quality more than any retrieval tweak. The explicit failure/anti-pattern type ("what should have been done" + "prevention") is a genuinely useful sub-type Somnigraph lacks.
**How**: A prompt/template in the remember guidance + an optional `outcome_type` enum; no new infra. Aligns with STEWARDSHIP "honest accounting" — store the negative results, not just the wins.

### 3. Write-time update-vs-create anti-fragmentation routing (Low, note)
**What**: Before writing, route into an existing broad memory rather than minting a narrow near-duplicate.
**Why/How**: Somnigraph already does this at *sleep* time (merge/archive in `sleep_nrem.py`); the only additive idea is doing a lightweight version at `remember()` time. Likely redundant with sleep — track, don't build.

---

## Not Useful For Us

### Skill-file / progressive-disclosure retrieval paradigm
Replacing ranked retrieval with "agent greps Markdown and reads what it wants" is orthogonal to (and weaker than) Somnigraph's learned-reranker core. Adopting it would discard the project's central differentiator.

### The whole cloud service surface (sandbox, Disk, S3, RabbitMQ, dashboard, Go API, multi-tenant KEK encryption)
Product/infra for a hosted multi-user SaaS; irrelevant to a single-user MCP memory server.

### pgvector / embedding config
Present in the stack but vestigial for memory — embeddings are unused for skill retrieval (semantic session search is a still-open roadmap TODO).

---

## Connections

- **Strong convergence with the Phase 18 source-sweep thesis** ("write-path quality, not retrieval, is what the leaders win on"): Acontext is a third independent data point after **agentmemory** (write-time grounding) and **ByteRover** (BM25-only + curated writes). Its entire value is in the distillation/skill-agent write path; retrieval is deliberately dumb. See `agentmemory.md`, `byterover.md`, and the 2026-06-28 Phase 18 session.
- **Anti-pattern/failure memory** echoes systems that store counterfactuals; the `applies_when` scoping is a sharper, more concrete version of "supersession/scoping" motifs seen elsewhere (cf. `memv`-style scoping).
- **Skill-as-memory format** shares DNA with Anthropic Agent Skills / Claude Code plugins — Acontext is essentially "auto-authoring of skill files from run traces."

---

## Summary Assessment

Acontext's core contribution is a **transparent, write-path-centric skill-memory**: an LLM task-extractor + distiller + skill-writing agent that turns agent run traces into human-editable Markdown SOPs, anti-patterns, and facts, then hands them back via grep and progressive disclosure. The engineering (MQ pipeline, per-space Redis lock with pending-queue draining and batched consolidation, optional KEK encryption) is real and competent. Its deliberate bet is that *good writing plus an agent that can read files* beats *ranked retrieval over opaque vectors* — the mirror image of Somnigraph's bet.

The single most valuable takeaway is not the paradigm but two write-path mechanisms: the **`applies_when` scope field** (anti-over-generalization gate that doubles as a reranker feature) and the **outcome-typed distillation with an explicit `skip_learning` gate and a first-class failure/anti-pattern schema**. Both target Somnigraph's acknowledged write-path-quality gap and cost only prompt/schema work, no infra. Everything retrieval-side is a step down from Somnigraph and should be left alone.

What's overhyped/missing: there are **no benchmarks** (no comparability to our 85.1 LoCoMo QA), **no decay/supersession/contradiction handling**, and the "search" is literal grep with no ranking — so the system has no principled answer to relevance at scale, which is precisely where Somnigraph invests. The evidence file's "deduplication ✅" also overstates: it is S3 object content-addressing (disabled under encryption), not memory-entry dedup; memory-level "dedup" is the LLM's update-vs-create judgment, not an algorithm. Verdict for us: **MAYBE** — mine two write-path ideas, adopt none of the architecture.
