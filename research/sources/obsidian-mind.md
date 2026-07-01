# obsidian-mind - Obsidian-vault + multi-agent scaffold with QMD (upstream) hybrid search as the retrieval engine

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

obsidian-mind is **not a memory engine**. It is a Claude Code / Codex / Gemini *vault template* (v6.2.1) for engineers' personal knowledge management — 1:1 notes, incident logs, brag docs, perf-review evidence, decision records. "Memory" = plain Markdown notes in an Obsidian vault, git-tracked. All actual retrieval (BM25, vector, fusion) is delegated to an **external npm package `@tobilu/qmd`** ("QMD"); this repo contains **zero retrieval, ranking, or storage code** — only cross-platform launcher glue (`.claude/scripts/lib/qmd.ts`, `qmd-mcp.mjs`) and lifecycle hooks.

### Storage & Schema
- **Store**: Markdown files in a vault (`brain/`, `work/`, `org/`, `perf/`, `reference/`) + a QMD-owned SQLite index living *outside* the repo at `~/.cache/qmd/<index>.sqlite` (`vault-manifest.json` → `qmd_index`).
- **Memory unit schema**: YAML frontmatter on a note. Vault convention requires `date`, `description` (~150 chars), `tags` (and `context`); enforced only as *hygiene warnings* by a hook, not a hard gate. This is the entire "schema" — no priority, no category taxonomy, no valid_from/until, no decay fields.
- vs Somnigraph's structured schema (category/priority/themes/valid_from/decay_rate) this is essentially untyped free text + tags.

### Memory Types
None as a data type. Organization is by *folder convention* (decisions in `brain/Key Decisions.md`, gotchas in `brain/Gotchas.md`, incidents in `work/incidents/`). No episodic/semantic/procedural distinction in the store.

### Write Path
- Notes authored by the agent via slash commands (`/om-capture-1on1`, `/om-incident-capture`, `/om-meeting`, etc.) and templates in `templates/`.
- **PostToolUse validation hook** (`.claude/scripts/validate-write.ts` → `lib/frontmatter.ts`): after any Write/Edit to a vault `.md`, checks for frontmatter (`date`/`tags`/`description`) and at least one `[[wikilink]]`; emits "vault hygiene warnings" as additional context. This is a lightweight *deterministic write-path quality nudge* — but it warns, never blocks or scores.
- **UserPromptSubmit classifier** (`classify-message.ts` → `lib/signals.ts`, `lib/matcher.ts`): regex signal patterns (DECISION / INCIDENT / etc., multilingual EN/JA/KO/ZH) match the user's prompt and inject a hint nudging the agent to create the right structured note. A *proactive-capture* prompt.
- No extraction, no dedup, no enrichment, no salience scoring. "Confirmed Absent: dedup automation, entity extraction, conflict detection" (evidence file, verified — no such code exists).

### Retrieval
- **All upstream in QMD.** Three surfaces (MCP `mcp__qmd__query/get/multi_get`, CLI, or Grep fallback), all reading the same SQLite store. Three modes: `search` (BM25), `vsearch` (vector), `query` (hybrid).
- QMD's `query` mode = "FTS + Vector + Query Expansion + Re-ranking" fused with **RRF** (per QMD docs), and its rerank step is an **LLM reranker** (`brain/Skills.md`: "hybrid BM25 + vector + LLM reranking"; README notes it downloads a ~1.28GB model). Not a learned/pointwise reranker, no feedback signal.
- This repo's only retrieval contribution is a `PostToolUse` **debounced incremental re-index** (`lib/qmd-refresh.ts`, `qmd update`, ~1-2s) so the index tracks vault edits, and a `SessionStart` context primer.

### Consolidation / Processing
None. No sleep, no summarization, no merge/archive. (Evidence: "Confirmed Absent: layered memory, conflict detection, dedup automation.")

### Lifecycle Management
None. No decay, no versioning, no archival logic (moving a file to `work/archive/` is manual). Evidence file: "Confirmed Absent: decay functions, time-travel."

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Hybrid BM25 + vector + RRF search | Real — but 100% provided by upstream `@tobilu/qmd`; this repo has no fusion code | Valid feature, **not this project's contribution** |
| "Vault-first memory keeps context across sessions" | Markdown notes + QMD index + SessionStart hook | Plausible as UX; it's file-based note-taking, not a memory model |
| Cross-platform (Win/mac/Linux) QMD invocation | Genuine engineering in `qmd.ts`/`qmd-mcp.mjs` (bypasses Windows .cmd shim, works around qmd 2.1.0 `--index` mcp bug) | Validated — the real code craft here |
| Multi-agent (Claude/Codex/Gemini) | Parallel hook/command dirs `.claude`/`.codex`/`.gemini` | Valid but cosmetic (same vault, three front-ends) |
| No benchmark numbers | No LoCoMo/QA/recall figures anywhere | N/A — nothing comparable to Somnigraph's 85.1 QA |

---

## Relevance to Somnigraph

### What obsidian-mind does that Somnigraph doesn't
- **Deterministic write-path hooks**: a `UserPromptSubmit` signal-classifier that nudges *capture* ("this looks like a decision — write a record") and a `PostToolUse` frontmatter/wikilink hygiene check. Somnigraph has no write-path quality gating or capture-prompting at all (gap is at the `tools.py` `remember` boundary). These are cheap, regex-level, deterministic — not the LLM-mediated approach Somnigraph would default to.
- **Human-browsable store**: memories are Obsidian notes a human reads/edits directly. Somnigraph's SQLite rows are opaque to the user.

### What Somnigraph does better
- Essentially everything on the retrieval-science axis. Somnigraph owns its retrieval stack (`fts.py`, `scoring.py`, RRF k=14 Bayesian-tuned, 26-feature LightGBM learned reranker with r=0.70 GT correlation); obsidian-mind rents an LLM reranker from QMD with no feedback loop.
- Consolidation (`sleep_nrem.py`/`sleep_rem.py`), typed graph edges + PPR, per-category decay, explicit feedback ratings — all absent here.

---

## Worth Stealing (ranked)

### 1. Signal-classifier proactive-capture hook (Low effort, note-only)
**What**: `classify-message.ts`/`signals.ts` — a `UserPromptSubmit` hook with a small data-driven table of regex signal patterns (DECISION, INCIDENT, ...) that, on match, injects a one-line hint telling the agent to persist the right kind of note.
**Why**: It is the *capture-side mirror* of Somnigraph's in-progress `docs/proactive-injection.md` (which is proactive-*recall*). A deterministic, near-zero-cost nudge to *store* — "the cost of a false-positive hint is ~0, a false negative is a missed memory" (their own comment) is exactly the asymmetry Somnigraph's write path ignores today.
**How**: A `UserPromptSubmit` hook that, on strong capture signals, hints "consider remember() with category=decision". But: Somnigraph already stores generously and has a sleep pass to reconcile; adding a capture nudge risks noise. **Note-only** — worth remembering as convergent design, not adopting now.

---

## Not Useful For Us

### QMD engine, RRF, hybrid search
Provided by an upstream package; Somnigraph already has a stronger, self-owned, learned-reranked equivalent. Nothing to port.

### Vault/Obsidian frontmatter + wikilink conventions
Tied to a human-facing Markdown PKM workflow; irrelevant to Somnigraph's single-user MCP SQLite model.

### Multi-agent (Codex/Gemini) front-ends, perf-review/brag-doc commands
Product-scaffold features for an engineer's work vault, orthogonal to memory research.

---

## Connections

- **Packaging-over-upstream-engine** pattern: same shape flagged in the Phase 18 sweep (ByteRover, TrueMemory) — the repo's headline "features" are an upstream engine's; the value-add is glue and UX. QMD here is the analogue of what those wrap.
- **Write-path over retrieval**: reinforces the Phase 18 / AMemGym finding that leaders win on the write path. obsidian-mind's only *original* memory logic is on the write side (capture hints, hygiene validation), not retrieval — consistent, though its write path is shallow (no dedup/salience).
- **Proactive-capture vs proactive-recall**: complements `docs/proactive-injection.md`. Their `classify-message` hook is the capture-direction instance of the same "deterministic hook injects a hint the agent's reflex skipped" seam.

---

## Summary Assessment

obsidian-mind's core contribution is **not a memory system** — it is a polished, multi-agent Obsidian *vault template* for engineers, with genuinely careful cross-platform launcher engineering (`qmd.ts`, `qmd-mcp.mjs`) around an external retrieval engine, `@tobilu/qmd`. Every retrieval capability the carsteneu evidence file credits (BM25, vector, hybrid, RRF fusion, reranking) lives entirely in that upstream package; this repo contains no storage, ranking, consolidation, or lifecycle code. There are no benchmark numbers, so nothing here is comparable to Somnigraph's 85.1 LoCoMo QA.

The single thing worth Somnigraph's attention is the **deterministic hook layer on the write path**: a `UserPromptSubmit` signal classifier that nudges memory *capture*, and a `PostToolUse` frontmatter-hygiene check. Both are cheap, regex-level, and honest about their false-positive/false-negative asymmetry — and the capture-nudge is a clean convergent instance of the proactive-injection idea already on Somnigraph's roadmap, aimed at the store direction instead of recall. It is note-only, not adopt-worthy: Somnigraph already stores generously and reconciles in sleep.

Evidence-file cross-check: the carsteneu report is unusually accurate and honest (it correctly lists decay, dedup, layered memory, conflict detection, entity extraction as **absent**, and correctly upgrades hybrid=true / searchModes=3). The one correction worth stating sharply: the report presents hybrid/RRF/semantic search as attributes of *obsidian-mind*, but they are wholly inherited from upstream `@tobilu/qmd` — obsidian-mind ships no retrieval code — and QMD's reranker is an **LLM reranker**, not a learned/feedback-driven one, so it is not comparable to Somnigraph's LightGBM reranker despite both being called "reranking."
