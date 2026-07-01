# Handoff — Complete the ai-memory-comparison survey (Opus workflow)

**Created 2026-06-30. For a future session. Do NOT run in the session that wrote this.**

## Goal

Produce a durable per-system record for every system in the [carsteneu/ai-memory-comparison](https://github.com/carsteneu/ai-memory-comparison) directory that Somnigraph has not yet analyzed, answering two questions beyond basic documentation:

1. **Compare/contrast with Somnigraph** — where each system is stronger/weaker/different.
2. **Good ideas worth considering** — concrete mechanisms/concepts/tools worth knowing about, even if not adopted. Zero ideas is a valid, honest answer.

Directive from Alexis: **err on the side of wasting tokens rather than losing valuable ideas.** Light-touch README triage demonstrably under-counts (see "Why deep reads" below), so this pass does real code-level reads of all remaining systems.

## Status at handoff (what this session already did)

- **Branch**: `research/comparison-declined-list` (work is **uncommitted** — review + commit before/after this pass).
- **`docs/declined.md`** drafted: Part 1 (comparison-table triage, repo-grounded light-touch), Part 2 (our own historical declines), Part 3 (collisions/dead-links). This survey pass should **upgrade Part 1** from light-touch verdicts to real compare/contrast + ideas.
- **`research/sources/ai-memory-comparison.md`** and **`research/sources/index.md`** got small pointer edits (dated 2026-06-30 update + negative-space pointer).
- **9 systems already deep-mined this session** (Opus-quality, code-level) — BUT the output is nugget-mining format, **NOT template-conforming**, so they must be **redone in the template** (see `research/source-analysis-prompt.md`). The prior findings (see "Nuggets already found" below) are a strong head-start input — the redo agent references them so the code-reading isn't from scratch — but each still needs a full template file. The 9: OpenViking, Honcho, Midas, Origin, ClawMem, mcp-memory-service, MemoryBear, MemoMind, mem9.
- **15 systems already have full analyses** in `research/sources/` (skip): Mem0, Graphiti, Cognee, ByteRover, MemPalace, vestige, hindsight, Letta, Supermemory, MemOS, MIRIX, MemMachine, LightMem, memsearch, Mengram.
- **Evidence-page cross-check pending**: `declined.md` was built from `data.js` + light READMEs, NOT the rigorously-sourced `evidence/*.md` per-feature files. Cross-check declinations against those (memU notably has **no** evidence file, supporting its SKIP).
- **Subagent artifacts** (both created without authorization by read-only subagents this session): `src/memory/MEMORY_SYSTEM.md` was a fabricated "relocated from MEMORY.md" file — **removed**. `research/carsteneu-submission/somnigraph.md` is a ~20KB draft evidence-file submission for Somnigraph (template-conforming, citations pinned to `d56b769`) — **left untracked, unverified, deferred to a separate future session** (not this survey). Submitting Somnigraph to the directory is an outward-facing decision for Alexis; the draft needs an accuracy pass first.

## Scope — all 59 untouched systems (50 cold + 9 template-redo)

**59 total** = the 50 cold reads in the table below **+ the 9 already nugget-mined** (listed in "Redo in template" directly under the table — they get a full template file too, seeded by the nuggets already found). Tier = priority hint only; all get done. "Bench" = benchmark-cell leader that light-touch called thin but must get a real read to confirm. URLs current as of 2026-06-30 `main`.

### The 50 cold reads

| # | Tier | System | Stars | LoCoMo | LME | Repo | Note |
|---|------|--------|-------|--------|-----|------|------|
| 1 | High | Acontext | 3562 | — | — | https://github.com/memodb-io/Acontext | "Agent Skills as a memory layer" — procedural/skill memory paradigm we don't cover |
| 2 | High | gbrain | 24572 | — | — | https://github.com/garrytan/gbrain | zero-LLM KG auto-linking + gap-aware synthesis + dream cycle |
| 3 | High | Kage | 6 | — | 96.17% R@5 | https://github.com/kage-core/Kage | write-time code-citation verification (WRIT-adjacent) |
| 4 | High | memanto | 1266 | 87.1 | 89.8 | https://github.com/moorcheh-ai/memanto | "information-theoretic, indexless" retrieval (proprietary backend) |
| 5 | High | Memori | 15504 | 81.95 | — | https://github.com/MemoriLabs/Memori | agent-native execution capture (tool calls, not just conversation) |
| 6 | High | memory-lancedb-pro | 4445 | — | — | https://github.com/CortexReach/memory-lancedb-pro | Weibull stretched-exponential decay + dreaming sidecar |
| 7 | High | Nanobot | 44872 | — | — | https://github.com/HKUDS/nanobot | git-versioned memory diffs (dream-log/restore); mostly agent framework |
| 8 | High | OpenMemory | 4281 | — | — | https://github.com/CaviraOSS/OpenMemory | HMD v2 5-sector decay + waypoint graph |
| 9 | High | shodh-memory | 224 | — | — | https://github.com/varun29ankuS/shodh-memory | Hebbian + spreading activation as no-LLM reranker alt |
| 10 | High | stash | 720 | — | — | https://github.com/alash3al/stash | 8-stage consolidation w/ causal-link + hypothesis engine |
| 11 | High | TencentDB-AM | 4506 | — | — | https://github.com/Tencent/TencentDB-Agent-Memory | Mermaid symbolic-graph memory + L0–L3 pyramid |
| 12 | High | token-savior | 1024 | — | — | https://github.com/Mibayy/token-savior | Thompson-sampled persona lattice (parallels our proactive-injection gating) |
| 13 | High | YesMem | 17 | 0.87 | — | https://github.com/carsteneu/yesmem | sawtooth proxy context-collapse; table maintainer's own system |
| 14 | Bench | agentmemory | 24321 | — | 95.2 | https://github.com/rohitg00/agentmemory | **DIFFERENT repo** than our JordanMcCann/agentmemory analysis — reconcile |
| 15 | Bench | EverOS | 9801 | 93.05 | 83.00 | https://github.com/EverMind-AI/EverOS | distinct product, same org as analyzed EverMemOS |
| 16 | Bench | m_flow | 4412 | 81.8 | 89.0 | https://github.com/FlowElement-ai/m_flow | **404 at sweep — recover URL first** (gh search / web) |
| 17 | Bench | memU | 13700 | 92.09 | — | https://github.com/NevaMind-AI/memU | 92.09 unverified (no evidence file); light-touch called it thin |
| 18 | Med | ArcRift | 243 | — | — | https://github.com/Eshaan-Nair/ArcRift | sentence-level "surgical trimming" of retrieved context |
| 19 | Med | Continuity v2 | 32 | — | — | https://github.com/Haustorium12/continuity-v2 | raw-transcript indexer + BFS thread recall |
| 20 | Med | engram | 4760 | — | — | https://github.com/Gentleman-Programming/engram | **DIFFERENT repo** than our Harshitk-cp/engram analysis |
| 21 | Med | Fullerenes | 19 | — | — | https://github.com/codebreaker77/Fullerenes | zero-LLM Tree-sitter blast-radius (out of scope, note concept) |
| 22 | Med | MemLayer | 277 | — | — | https://github.com/divagr18/memlayer | salience-based write gating + tiered vector+graph |
| 23 | Med | memoir | 587 | — | — | https://github.com/zhangfengcdt/memoir | git-like branch/commit/merge memory versioning |
| 24 | Med | Memora | 417 | — | — | https://github.com/agentic-box/memora | RRF + auto-hierarchy + live graph UI |
| 25 | Med | Memory Palace | 306 | — | — | https://github.com/AGI-is-going-to-arrive/Memory-Palace | forgetting engine + snapshot rollback + 4 maintenance engines |
| 26 | Med | MoltBrain | 250 | — | — | https://github.com/nhevers/MoltBrain | ChromaDB + hooks; crypto-paid storage tier |
| 27 | Med | obsidian-mind | 3166 | — | — | https://github.com/breferrari/obsidian-mind | Obsidian vault + "QMD hybrid RRF" |
| 28 | Med | Octopoda-OS | 348 | — | — | https://github.com/RyjoxTechnologies/Octopoda-OS | 5-signal loop detector + hash-chained audit trail |
| 29 | Med | omega-memory | 170 | — | 76.8 | https://github.com/omega-memory/omega-memory | 28-tool lifecycle; LME figure self-disclosed category-tuned |
| 30 | Med | second-brain | 91 | — | — | https://github.com/rahilp/second-brain-cloudflare | serverless Cloudflare; time-decay + smart-merge LLM |
| 31 | Med | TeleMem | 461 | — | — | https://github.com/Tele-AI/TeleMem | Mem0 drop-in; semantic-cluster dedup + multimodal video |
| 32 | Med | YourMemory | 231 | 59.0 | 89.4 | https://github.com/sachitrafa/YourMemory | Ebbinghaus decay + NER BFS graph; benchmark confounds |
| 33 | Low | ai-memory | 864 | — | — | https://github.com/akitaonrails/ai-memory | git-versioned markdown wiki, zero-LLM |
| 34 | Low | AIPass | 217 | — | — | https://github.com/AIOSAI/AIPass | multi-agent workspace + mailboxes |
| 35 | Low | claude-mem | 85116 | — | — | https://github.com/thedotmack/claude-mem | progressive-disclosure UX; high stars, thin mechanism |
| 36 | Low | CommonGround | 138 | — | — | https://github.com/Intelligent-Internet/CommonGround | multi-agent coordination ledger |
| 37 | Low | context-infra | 635 | — | — | https://github.com/grapeot/context-infrastructure | markdown observation log + cron |
| 38 | Low | fidelis | 1 | — | 83.2% R@1 | https://github.com/hermes-labs-ai/fidelis | non-LLM BM25+rerank verbatim; depends on mem0 |
| 39 | Low | gitmem | 8 | — | — | https://github.com/gitmem-dev/gitmem | scars/wins markdown; semantic search paywalled |
| 40 | Low | icarus | 292 | — | — | https://github.com/esaradev/icarus-memory-infra | 3-layer git wiki; provenance/supersession |
| 41 | Low | Jumbo | 102 | — | — | https://github.com/jumbocontext/jumbo.cli | event-sourced (CQRS) project-context assembly |
| 42 | Low | LangMem | 1531 | — | — | https://github.com/langchain-ai/langmem | LangChain toolkit; prompted extraction |
| 43 | Low | MarsNMe | 5 | — | — | https://github.com/marsmanleo/MarsNMe | MCP gateway + Supabase; TTL decay |
| 44 | Low | memorix | 433 | — | — | https://github.com/memorix-ai/memorix | **404 at sweep — recover URL first**; generic FAISS/Qdrant wrapper |
| 45 | Low | Memvid | 15700 | +35% SOTA | — | https://github.com/memvid/memvid | single-file .mv2 "video" packaging; +35% claim no methodology |
| 46 | Low | nocturne | 1240 | — | — | https://github.com/Dataojitori/nocturne_memory | graph/tree non-vector + visual rollback |
| 47 | Low | Noosphere | 53 | — | — | https://github.com/SweetSophia/noosphere | multi-agent wiki; draft-to-curated pipeline |
| 48 | Low | opencode-mem | 995 | — | — | https://github.com/tickernelz/opencode-mem | OpenCode plugin; vector + persona extraction |
| 49 | Low | VIR | 15 | — | — | https://github.com/djolex999/vir | Obsidian transcript-distillation write-path tool |
| 50 | Low | Wax | 767 | — | — | https://github.com/christopherkarani/Wax | Swift/Metal single-file perf engine |

### The 9 to redo in template (nugget findings already in hand — see "Nuggets already found" for the head-start)

| System | Stars | Repo | Prior result |
|--------|-------|------|--------------|
| Midas | 6 | https://github.com/vornicx/Midas | 3 nuggets (strongest overall) — write full file |
| mcp-memory-service | 1901 | https://github.com/doobidoo/mcp-memory-service (clone via codeberg.org mirror) | 2 nuggets incl. LME harness — write full file |
| Origin | 31 | https://github.com/7xuanlu/origin | 1 nugget (faithfulness gate) |
| ClawMem | 189 | https://github.com/yoloshii/ClawMem | 1 conditional nugget (subject-anchor guard) |
| Honcho | 5652 | https://github.com/plastic-labs/honcho | 1 minor nugget (surprisal targeting) + judge-leniency note |
| OpenViking | 26179 | https://github.com/volcengine/OpenViking | 0 nuggets (default-disabled knobs) — short file |
| MemoMind | 699 | https://github.com/24kchengYe/MemoMind | 0 (packaging over hindsight-api) — short file |
| mem9 | 1152 | https://github.com/mem9-ai/mem9 | 0 + 2 crumbs (int-ID, shadow-metric) — short file |
| MemoryBear | 4167 | https://github.com/SuanmoSuanyangTechnology/MemoryBear (renamed) | 0 (confirm-ahead) — short file |

## Method

**Orchestration**: `Workflow` tool, one Opus agent per system, chunked to **~4–6 concurrent** (Opus is expensive; the earlier light-touch run used chunks of 4 — reuse that pattern). Total 59 → ~10–15 waves (the 9 redo agents get their prior nuggets pasted into the prompt as a head-start). Reuse/adapt the prior light-touch triage workflow script from the session's scratchpad workflow-scripts directory
(the `typeof args === 'string' ? JSON.parse(args) : args` guard is required — `args` arrives JSON-encoded).

**Per-agent depth**: real code read, not README skim. Each agent should `git clone --depth 1 <url>` into a scratchpad subdir (fall back to `gh`/WebFetch on raw files), then read the retrieval, write-path, consolidation, and decay implementations. First step for any 404 row: recover the URL (`gh repo view`, `gh search repos`, web search) before giving up — this session found MemoryBear had been renamed to `SuanmoSuanyangTechnology/MemoryBear` and mcp-memory-service was live at a Codeberg mirror.

**Per-agent output**: a **template-conforming draft** following `research/source-analysis-prompt.md` (Paper Overview / Architecture / Key Claims & Evidence / PERMA / Relevance to Somnigraph / Worth Stealing / Not Useful / Connections / Summary Assessment), **length proportional to substance** — thin systems get short files with sections marked "None"; don't pad. Plus a structured summary object for the aggregate index:
`{ name, url, verdict: DIVE|MAYBE|SKIP, one_line_compare, nuggets: [{idea, redundant_or_additive, maps_to_module, effort, verdict}], promote_to_source_file: bool }`.
Give each agent the condensed Somnigraph context block from `research/source-analysis-prompt.md` §"Somnigraph Context for Agents" so comparisons are accurate. **Cross-check each system's `evidence/<name>.md`** (raw: `https://raw.githubusercontent.com/carsteneu/ai-memory-comparison/main/evidence/<name>.md`) against the code — the evidence files carry sourced per-feature ✅/❌ and honest corrections (e.g. Midas LoCoMo 0.85→0.73) that sharpen the record.

**Whether agents write files directly**: recommended they return content and the orchestrator/you writes `research/sources/<name>.md` after a sanity pass, OR agents write directly (distinct filenames, no conflict) and the human reviews the branch diff before merge (STEWARDSHIP review gate). Either way this stays on the feature branch until reviewed.

## Deliverables the next session produces

1. Up to 59 `research/sources/<name>.md` files (template-conforming, length ∝ substance) — 50 cold + 9 redone from nugget format.
2. `research/sources/index.md` — add the new rows to the Repositories table.
3. `docs/declined.md` — upgrade Part 1 from light-touch verdicts to compare/contrast + link each row to its new source file; move genuinely-thin systems to a compact "surveyed, nothing to adopt" table (keep the reason).
4. A consolidated **ideas ledger** — every "worth considering" nugget across all systems (incl. the 9 already mined + this pass), each mapped to a Somnigraph module and given an effort/verdict. Put in `docs/roadmap.md` (new subsection) or a new `docs/ideas-considered.md`; decide at run time.
5. Redo the 9 already-mined systems as template files (seeded by their prior nuggets); **Midas** and **mcp-memory-service** get full-length files, the four zero-nugget ones get short files.

## Durable inputs / re-fetch (scratchpad is ephemeral)

The 50-row list above is self-contained. Re-fetch the reference material fresh:
```
curl -s https://raw.githubusercontent.com/carsteneu/ai-memory-comparison/main/CRITERIA.md   -o CRITERIA.md
curl -s https://raw.githubusercontent.com/carsteneu/ai-memory-comparison/main/data.js        -o data.js
curl -s https://raw.githubusercontent.com/carsteneu/ai-memory-comparison/main/comparison.md   -o comparison.md
# per-system evidence: https://raw.githubusercontent.com/carsteneu/ai-memory-comparison/main/evidence/<name>.md
```
Somnigraph context: `docs/architecture.md`, `research/source-analysis-prompt.md`, `STEWARDSHIP.md`.

## Nuggets already found this session (head-start for the 9 template redos — findings, not a substitute for a template file)

- **mcp-memory-service** — (1) **reproducible open LongMemEval *retrieval* harness** (R@k/NDCG/MRR, 0 LLM, standard dataset) — fills a real gap (we have zero external LME numbers); port an adapter to `scripts/benchmarks/`; caveat: recall-only, complements not replaces LoCoMo QA. (2) **mid-similarity sweet-spot (0.45–0.7) sampling** for NREM neighbor selection (our top-K biases to near-dups).
- **Midas** — (1) **`memory_state`/`memory_diff` deterministic views** (broad "current state / what changed" queries under-retrieve under top-k; we have the event-log + supersession substrate). (2) **dumb-reader ablation** (eval-honesty floor; addresses STEWARDSHIP §4 LoCoMo judge-leniency). (3) **currency surfaced at recall time** (annotate superseded/expired instead of silently filtering). Plus small: entity negative-guard + ambiguity margin on supersession.
- **Origin** — **faithfulness gate on REM summaries** (reject/regenerate if `cosine(summary, sources) < τ`; cheap on existing embed path).
- **ClawMem** — **subject-anchor guard on the destructive supersede path** (lexical proper-noun check before `remember()` deletes a near-dup), gated on a cheap offline measurement that the collision occurs at our 0.9 cosine cutoff.
- **Honcho** — **surprisal/novelty-based consolidation targeting** (kNN-distance/RP-tree over sqlite-vec embeddings to pick novel outliers as NREM/audit targets). Plus docs note: their 89.9 LoCoMo uses a lenient GPT-4o-mini judge, not comparable to our Opus 85.1.
- **mem9** — **integer-ID indirection** when an LLM is in the consolidation loop (cleaner than post-hoc commit `164fcb5`); **shadow-mode near-dup metric** (log NN-cosine distribution before setting a dedup threshold).
- **OpenViking / MemoryBear / MemoMind** — zero build nuggets (confirm-ahead); OpenViking's two "novel" retrieval knobs ship default-disabled; MemoMind is packaging over `hindsight-api`; MemoryBear's ACT-R spacing decay is unrealized in its own code. One `similar-systems.md` line each at most.

Recurring cross-system theme worth capturing: **"measure before you threshold"** (mem9 shadow-mode, ClawMem dry-run, Midas dumb-reader) independently matches our honest-accounting ethos.

## Why deep reads (lessons that justify the token spend)

- Light-touch README triage **under-counts**: mcp-memory-service was nearly a SKIP and held the most valuable find; token-savior (SKIP'd) has Thompson-sampling that parallels our own proactive-injection design.
- Watch for: **repo renames/404s** (recover URL first), **packaging over an upstream engine** (MemoMind was hindsight-api — analyze the upstream, credit correctly), **benchmark cells that are retrieval-recall not end-to-end QA** (mcp-memory-service 86.0 is R@5, not comparable to our 85.1 QA), **mechanisms shipped default-disabled** (OpenViking), and **name collisions** (engram, agentmemory rows point to different repos than our analyses).
- Expected yield (calibrated on this session's 9): roughly half the High/Bench tier yields a keeper idea; Low tier is mostly confirm-thin but the point is not to lose the occasional buried concept.

## Open decisions for the next session

- Ideas ledger location: `docs/roadmap.md` subsection vs new `docs/ideas-considered.md`.
- Agents write source files directly vs return-and-orchestrator-writes.
- Whether to also cross-check the 15 already-analyzed systems against their evidence files (probably not — out of scope).
- The `research/carsteneu-submission/somnigraph.md` draft is deferred to its own future session (accuracy pass, then Alexis decides on submission) — not part of this survey.
