# memU - LLM-orchestrated memory stored as a human-readable Markdown file tree, with an auto-synthesized "skill" line from tool traces

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

memU (NevaMind-AI/memU, Apache-2.0, ~13.7k stars, Python 3.13+) is a multi-user memory service whose thesis is *"personal memory, stored as files."* Conversations, documents, images, audio, video and tool traces are compiled by LLMs into a human-readable Markdown tree (`INDEX.md`, `MEMORY.md`, `SKILL.md`, plus `memory/<topic>.md` and `skill/<name>.md`). Agents "traverse the tree and load only what the moment needs." The current repo is a **file-based rewrite** (v1.5.x); the retrieval and write paths are almost entirely LLM-orchestrated rather than algorithmic.

### Storage & Schema
Pluggable backends via `database/factory.py`: `inmemory`, `sqlite`, `postgres` (pgvector). The unit schema (`database/models.py`):
- **`RecallEntry`** — the memory item: `memory_type`, `summary` (the extracted text), `embedding`, `happened_at`, and an untyped `extra` dict (holds `content_hash`, `reinforcement_count`, `last_reinforced_at`, `ref_id`, and tool-memory fields `when_to_use`/`metadata`/`tool_calls`).
- **`RecallFile`** — a category (a Markdown file): `name`, `track` (`"memory"` or `"skill"`), `description`, `embedding`, `content` (the rendered Markdown body).
- **`RecallFileEntry`** — a plain `item_id → category_id` join. This is the only "edge" in the system; there are **no typed relations, no edge weights, no graph table**.
- **`Resource`** — the raw source (url, modality, local_path, caption, embedding).

Scope fields (`user_id`, `agent_id`, …) are mixed into every model via `merge_scope_model`, giving multi-tenant `where`-filtering (`agent_id__in`, etc.).

### Memory Types
`EntryType = profile | event | knowledge | behavior | skill | tool`. Categories (the Markdown files) are LLM-classified per source, and new categories can be created on the fly when `allow_new_categories` is set.

### Write Path (`app/memorize.py`, 7-step workflow)
`ingest → preprocess_multimodal → extract_entries → dedupe_merge → categorize_entries → persist_and_index → generate_skills`.
- **Preprocess** (`preprocess/`): real multimodal handling — audio/video/image/document/conversation each get an LLM/VLM pass to text/caption.
- **Extract**: an LLM produces structured `(memory_type, summary, category_names)` tuples per source.
- **`dedupe_merge` is a literal no-op placeholder** (`memorize.py:467-470`: `# Placeholder for future dedup/merge logic`).
- **Reinforcement dedup** (`_persist_recall_entries` → `create_item_reinforce`): if `enable_item_reinforcement` is on, an **exact SHA-256 content-hash** of `(normalized_summary, memory_type)` is looked up in the same user scope; a hit increments `reinforcement_count` and bumps `last_reinforced_at` instead of inserting. This is exact-string dedup only — no semantic/near-duplicate merge.
- **File summaries** are re-synthesized by an LLM on every memorize (incremental merge into the category Markdown).
- **`generate_skills`** (opt-in, ADR 0006): an LLM reads the source text plus all existing skill files and emits skill Markdown (workflows, patterns, "mistakes to avoid"), persisted directly as `RecallFile(track="skill")`, bypassing the entry plane.
- **No salience/quality/importance gating on write.** Everything extracted is stored.

### Retrieval (`app/retrieve.py`)
Two methods, both LLM-heavy; default `method="rag"`:
- **RAG pipeline** — a tiered, agentic loop: `route_intention` (LLM decides *whether* retrieval is needed and rewrites the query) → rank category files by summary (vector) → **`sufficiency_check` (LLM decides after each tier whether to dig deeper)** → recall entries (vector) → sufficiency → recall resources (vector) → `build_context`. The query is re-embedded per tier.
- **LLM method** — delegates ranking wholesale to the model via `llm_category_ranker` / `llm_item_ranker` / `llm_resource_ranker` prompts.
- **Scoring is cosine-only** (`vector.py:cosine_topk`). **No BM25, no FTS, no RRF, no learned reranker.** The only "rerank" is an LLM reading candidates.
- **Optional salience ranking** (`vector.py:salience_score`): `similarity · log(reinforcement_count+1) · exp(-ln2·days/half_life)`, half-life default 30d. **Off by default** (`ranking="similarity"`, `settings.py:360`).

### Consolidation / Processing
No background/offline/sleep cycle, no relationship detection, no contradiction handling, no garbage collection. The only "consolidation" is the per-write LLM re-summarization of category files and the opt-in `synthesize` of `MEMORY.md`/`SKILL.md` overviews (`memory_fs/synthesizer.py`).

### Lifecycle Management
Opt-in reinforcement counters + recency decay in salience (both default off). Cascade delete by source URL (`_cascade_delete_by_urls`). **No versioning, no `valid_from/valid_until`, no supersession, no archival/dormancy.**

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| LoCoMo 92.09% average accuracy | README banner image only; **no benchmark harness, no eval script, no methodology anywhere in the repo** | **Unverifiable.** Judge model, prompt, and split undisclosed. Almost certainly predates this file-based rewrite. Not comparable to Somnigraph's 85.1% Opus-judged QA. |
| "Fast retrieval, higher accuracy, lower cost" | Marketing tagline; no ablation or numbers in repo | Unvalidated assertion |
| Multimodal ingestion (5 modalities) | `preprocess/{audio,video,image,document,conversation}.py` real LLM/VLM passes | Validated (code present) |
| Deduplication | `dedupe_merge` step is a no-op; exact content-hash reinforcement exists but is opt-in | Partially real; **no semantic dedup** |
| Salience-aware decay/reinforcement | `vector.py:salience_score`, `recency_decay_days` | Real mechanism, **default-disabled** |
| Multi-agent / scope filtering | `merge_scope_model`, `where` with `__in` operators | Validated |
| Hybrid BM25+vector, "wiki-graph" | Only **proposed** in ADR 0007 (Status: Proposed, dated 2026-07-01) | **Not implemented** |

---

## Relevance to Somnigraph

### What memU does that Somnigraph doesn't
- **Auto-synthesized skill line from tool/execution traces** (`generate_skills`): distills reusable workflows and anti-patterns ("mistakes to avoid") into their own retrievable Markdown files, refined incrementally as new sources arrive. Somnigraph has a `procedural` category but relies on the agent/user to *author* those memories; it does not mine tool-call traces into skill docs. Gap sits at the write path (`tools.py` remember) / a potential `sleep_rem.py` step.
- **Tool memory with a `when_to_use` retrieval cue + `avg_success_rate`/`ToolCallResult` metadata** — memories carry an explicit "surface me when the task looks like X" hint and outcome stats. Somnigraph has no equivalent retrieval-cue field.
- **Native multimodal write path** (image/audio/video → text). Somnigraph is text-only.
- **Multi-user/agent scoping** as a first-class data-model concern. Somnigraph is single-user by design.
- **Human-readable, directly-editable Markdown store** — auditable/portable by construction.

### What Somnigraph does better
- **Retrieval quality**: Somnigraph's hybrid BM25+vector RRF (`fts.py`, `scoring.py`) + 26-feature LightGBM reranker (`reranker.py`, NDCG=0.7958) is a measured, learned pipeline. memU is cosine-only with an LLM reading candidates — no fusion, no learned ranking, and a per-tier LLM cost that Somnigraph avoids.
- **Graph-conditioned retrieval**: Somnigraph has typed edges + PPR expansion + betweenness features (`sleep_nrem.py`, `scoring.py`). memU's "graph" is a flat category-membership join; ADR 0007's real graph is unbuilt.
- **Offline consolidation**: Somnigraph's three-phase LLM sleep (edge detection, merge/archive, gap analysis) has no memU counterpart — memU does only per-write re-summarization.
- **Feedback loop**: Somnigraph's explicit per-query utility ratings with r=0.70 GT correlation and UCB exploration have no memU analog (its "reinforcement" is just a write-time exact-dup counter, default off).
- **Lifecycle**: per-category decay with reheat is *live* in Somnigraph; in memU it's opt-in and default-off.

---

## Worth Stealing (ranked)

### 1. Skill line: distill reusable procedures/anti-patterns from tool traces (Medium)
**What**: A dedicated write/consolidation step that reads execution and tool-call traces and emits compact, retrievable "skill" notes — including *what not to do* — refined incrementally rather than authored by hand.
**Why**: Somnigraph's `procedural` memories are user/agent-authored; the recurring "how did we do X, and what went wrong last time" knowledge is exactly what a sleep pass could mine automatically. This is an *additive* capability, not a reranking tweak.
**How**: Add a REM-phase step in `sleep_rem.py` that clusters recent procedural/episodic memories about the same task and emits a consolidated procedural memory tagged as a "skill," carrying a short "when to use" summary. Reuse existing decay/priority machinery; no schema change strictly required (could ride the `procedural` category + a theme).

### 2. `when_to_use` retrieval cue + outcome metadata on procedural memories (Low)
**What**: Store an explicit "surface me when the task looks like X" hint and success/failure stats alongside a procedural memory.
**Why**: A retrieval cue written at consolidation time is a cheap signal the reranker or a query-time match could exploit, and outcome stats let bad procedures decay faster.
**How**: Optional fields in the memory `extra`/metadata; feed `when_to_use` text into the enriched embedding (`embeddings.py`) and expose success-rate as a reranker feature. Low effort, easily reversible.

---

## Not Useful For Us

### Markdown-file-tree as the store
memU's headline ("memory stored as files") trades retrieval precision for human-auditability and multi-tenant portability. Somnigraph's SQLite + sqlite-vec + FTS5 is deliberately the opposite trade-off; adopting a file tree would regress retrieval quality.

### Tiered LLM sufficiency-check retrieval
The "LLM decides after each tier whether to recall more" loop spends an LLM call per tier at query time. Somnigraph's single-shot reranker is cheaper and measured. The only transferable sliver is "stop early when context is sufficient" as a token-cost lever — note-only.

### Salience formula (similarity · log(reinforce+1) · recency)
Convergent with Somnigraph's decay + reheat-on-access, but coarser (exact-hash reinforcement, no per-category half-lives, no feedback). Redundant.

---

## Connections

- **Convergent with the Phase 18 write-path thesis** (`2026-06-28-phase18-source-sweep.md`): like ByteRover (BM25-only), MemPalace (verbatim), and agentmemory (write-time grounding), memU's differentiation lives on the *write* side (multimodal extraction, skill synthesis), and its retrieval is thin (cosine + LLM). More independent corroboration that leaders win on write quality, not fusion.
- **Skill/procedure distillation** echoes systems that separate "facts" from "learned procedures" — cross-reference agentmemory's write-time grounding and any MIRIX/MemOS procedural-memory notes.
- **Opt-in-but-default-off decay** mirrors a pattern seen elsewhere where a mechanism is coded but unrealized in defaults — a recurring evidence-file trap (claiming a feature the defaults disable).

---

## Summary Assessment

memU's genuine contribution is an **LLM-native, multimodal, multi-tenant write path that compiles heterogeneous sources into an auditable Markdown tree**, plus a **self-refining "skill" line** that distills reusable procedures (and mistakes) from tool traces. That skill line is the one idea here worth a second look for Somnigraph — it targets the write path, which our own corpus keeps flagging as where the leaders actually win, and Somnigraph currently leaves procedural-skill capture to manual authoring.

The retrieval side is thin: cosine-only vector search wrapped in an expensive per-tier LLM orchestration, no BM25/RRF/learned reranker, and a "graph" that is just a category-membership join. The advertised decay/reinforcement and dedup exist in code but are default-disabled or literal placeholders. The headline **LoCoMo 92.09% has no reproduction harness in the repo** and predates this architecture — treat it as marketing, not evidence, and do not place it on the same axis as Somnigraph's 85.1% Opus-judged QA.

Single most important takeaway: **auto-distill procedural "skills" from execution traces during sleep** (Worth-Stealing #1). Everything else is either already stronger in Somnigraph or unrealized in memU.

**Evidence-file correction**: the carsteneu file reports LoCoMo 92.09% as a verified metric and lists decay/dedup ambiguously. Sharper reading of the code: (1) the 92.09% figure has **no harness in the repo** and is architecturally stale; (2) decay + reinforcement **exist but default off** (not simply "absent"); (3) the pipeline `dedupe_merge` step is a **no-op placeholder** while a separate **exact-content-hash** reinforcement path is the only real dedup (opt-in) — there is no semantic dedup; (4) BM25/hybrid and the "wiki-graph" are **ADR-0007 proposals, not implemented.**
