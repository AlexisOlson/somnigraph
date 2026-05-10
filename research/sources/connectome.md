# Connectome (anima-research) — Analysis

*Generated 2026-05-03. First draft was based only on the public landing page (anima.ai/connectome) and was substantially incorrect; this version reflects code reading from local clones of all five repos with parallel investigation by general-purpose agents. File and line references throughout are to the cloned trees at `C:/Users/Alexis/tmp/connectome-analysis/`.*

---

## Project Overview

**Project**: Connectome — research infrastructure for stateful agents
**Maintainer**: Anima Labs (501(c)(3) studying emergent LLM phenomena, machine cognition, alignment)
**License**: MIT (all five repos)
**Lead author / package scope**: `@animalabs/*` on npm; `membrane` is the antra-tess fork (Antra Tessera is the canonical author — there is no anima-research upstream)
**Spec sibling**: `anima-research/mcpl` (MCP Live extension)

**Five repos investigated**:

| Repo | Version | Last commit | LOC est. | Notes |
|------|---------|-------------|----------|-------|
| `chronicle` | 0.1.1 | 2026-03-31 | ~15k Rust + N-API | No top-level README; quiet ~5 weeks |
| `membrane` | 0.5.46 | recent | TS | Antra-tess origin; canonical despite scope name |
| `context-manager` | 0.2.0 | 2026-04-21 | 6,483 (≈3,610 src + 2,875 tests) | Active; no README |
| `agent-framework` | 0.4.0 | 2026-04-23 | TS | Active; minimal TODOs |
| `connectome-host` | 0.3.0 | 2026-04-24 | 103 commits | Active; reference TUI; "dogfood the AF" |

**Maturity assessment**: Active research code, not production. All repos under 6 months old, mostly 0.x versions, several have no README. `chronicle` has the largest gap between marketing language and implementation; `connectome-host` is the most coherent end-to-end consumer.

**Welfare-research framing**: The landing page foregrounds Anima's stance that "records of a model's life should survive the model" and pitches the architecture as model-welfare-shaped (causation graphs, branching as research method, observation without forced reflection). The code partially supports this framing — chronicle's append-only log + branching is real, autobiographical compression is implemented — but several of the most distinctive-sounding architectural claims are aspirational documents rather than running code.

---

## Per-Repo Findings

### Chronicle

A branchable append-only record store with N-API bindings to Node. **Bespoke filesystem layout** (`records.log`, `records.idx`, `state.bin`, `branches.bin`, `blobs/<shard>/<hash>`) — not SQLite, not RocksDB.

**Verified ✓**:
- Content-addressed blobs with SHA-256, sharded by first hex byte, CRC32 checksums, LRU cache (`types.rs:60-89`, `blobs/storage.rs:51-95`)
- N-API bindings; Rust core; Node 8+ ABI; prebuilt for darwin/linux x64+arm64 (no Windows triple — `package.json:11-15`)
- Records carry `id, sequence, recordType, payload, timestamp, causedBy[]` ✓ but…

**Refined ~**:
- Records have **two more fields** the marketing page omits: `branch: BranchId`, `encoding: PayloadEncoding {Json,MessagePack,Raw}`, and **`linked_to: Vec<RecordId>`** — soft references, distinct from `causedBy[]` hard refs (`types.rs:138-166`, `docs/loom-of-looms.md:350-352`)
- **Five state strategies, not three**: `Snapshot`, `Delta`, `AppendLog` plus **`Tree`** (filesystem-tree, BTreeMap<path, TreeEntry> with blob refs, added Mar 23) and **`Struct`** (recursive, per-field strategies) — `types.rs:235-271`
- `causedBy[]` is **optional and unenforced** — defaults to empty `Vec` (`types.rs:185, 201-205`); JS API exposes it as `Option<Vec<String>>` (`napi.rs:124`)
- Branches are cheap to create (only allocates `Branch{}` + map insert, `branches/manager.rs:141-179`) but **branches are virtual filters over one global log, not isolated stores**. There is no per-branch index. `query_range` walks the global `BTreeMap<(BranchId, Sequence), offset>` (`records/index.rs:21`), which is **rebuilt from full log scan on startup** (`records/index.rs:51-75`, "1-3 seconds per 1M records on SSD") — not persisted.

**Contradicted ✗**:
- **"Reconstructing state at any past point is `O(log #checkpoints)`, not full replay"** — not implemented as described. The phrase appears only in `docs/loom-of-looms.md` §4 and as a comment in `wal.rs:220`. No `checkpoint_index: BTreeMap<...>` exists in source. What's actually implemented (`state/manager.rs:290-344`): each `StateUpdateRecord` carries `prev_update_offset` (`types.rs:341`) forming a backward chain; reconstruction walks the chain to the most recent `last_full_snapshot_offset` then forward-applies. **Real complexity is O(ops since last full snapshot)**, controlled by `delta_snapshot_every`/`full_snapshot_every`. No tests for the `O(log #checkpoints)` claim — `tests/scaling.rs` tests append throughput only.
- **"Loom of Looms" nested chronicles** — algebraic spec only (`docs/loom-of-looms.md`). No `Loom`, `embed`, `observe`, `CTRL`, or envelope-record code in `src/`. The `linked_to` field plus branch namespacing are the building blocks; the embedding layer is unbuilt.

**Other findings**:
- **Storage backend**: bespoke filesystem (`store.rs:115-128`), single-process, file-locked via `fs2::FileExt`. No replication/sync (Loom doc §13 marks it future work).
- **Tests cover**: branch isolation, snapshots, scaling (append throughput). **Not covered**: multi-process correctness, crash recovery beyond WAL, the checkpoint-reconstruction performance claim.
- **Python consumability**: **No.** N-API → Node only. No FFI/C ABI export, no PyO3, no HTTP server in `src/`. There's a separate `server/` with a TS HTTP layer; somnigraph would need to either spawn that or re-implement via PyO3.
- **Only 2 TODOs in src** — small surface; real gaps are the unbuilt features above.

### Context-manager

An on-top-of-chronicle compression layer separating the immutable MessageStore from the editable ContextLog. **Has no retrieval system at all** — sequential compression pyramid only.

**Verified ✓**:
- MessageStore (immutable, Chronicle-backed, `src/message-store.ts:40`) vs ContextLog (editable working set, `src/context-log.ts:15`)
- `sourceRelation: 'copy' | 'derived' | 'referenced'` (`src/types/context.ts:8-14`); semantic is edit propagation. Tests at `integration.test.ts:368-513`.
- Three exported strategies: passthrough / autobiographical / knowledge (`src/strategies/`, `src/index.ts:11-14`)
- Hierarchical L1→L2→L3 is real (`autobiographical.ts:499-717`); hierarchical mode is the **default** (line 92: `this.config.hierarchical ??= true`); threshold = **6 unmerged L1s → 1 L2** (`autobiographical.ts:585`, `types/strategy.ts:175`)

**Refined ~**:
- **Anti-redundancy filtering** is *level shadowing*, not deduplication. `getAntiRedundantSummaries` (`autobiographical.ts:464-493`) excludes an L2/L3 summary from the prompt only if **all** of its constituent children are already shown at a lower level (`every` not `some`). Combined with budget carryover (`selectHierarchical:763-799`) where unused L3 budget flows to L2, then L2 to L1.
- **Cache markers**: one mark only (end of head window, `autobiographical.ts:259, 751`), surfaced via `cacheBreakpoint: true` on the NormalizedMessage. **Not** multi-layer alignment to all Anthropic cache boundaries.
- **The "voice" effect is structural, not voice-trained.** The autobiographical compression prompt (`autobiographical.ts:861`): *"Starting from my last message, please describe everything that has happened. Aim for about ${targetTokens} tokens. Describe it as you would to yourself, as if you are remembering what has happened."* System: *"You are forming autobiographical memories of a conversation."* (line 537). The "voice" rides on (a) **role spoofing** — prior summaries replayed as assistant turns from `'Claude'` (line 519), and (b) `temperature: 0`. No fine-tuning, no persona prompt, no fidelity check that the summary is grounded in the chunk.
- **Knowledge strategy** is implemented but doesn't extract structured artifacts. `PhaseType = 'research' | 'synthesis' | 'lesson' | 'subagent'` (`types/strategy.ts:223`). It **classifies messages by tool name** (`mcpl:` → research, `subagent:` → subagent, exact `lessons:create`/`lessons:update` → lesson, else synthesis; `knowledge.ts:292-302`) and uses phase-tagged prompts plus an asymmetric L1 budget: research capped at 30%, synthesis 40-70%, lessons uncapped (`knowledge.ts:204-260`). It also tracks unresolved leads via the literal string `[LEAD]`. **Lessons are text in summary content, not separate records.**

**Other findings**:
- **No fidelity tracking, no compression-quality regression suite, no contradiction handling** — all absent.
- Compression model: `claude-sonnet-4-20250514` default (`autobiographical.ts:386, 539, 673`)
- Strategy interface is small and clean (`types/strategy.ts:69-111`): `name`, optional `initialize/tick/onNewMessage`, required `checkReadiness/select`. Extension is straightforward.
- Test coverage exists for chunking, classification, and budgets, but **production compression behavior is exercised only by one live API test** (`autobiographical.test.ts:20`) that asserts `messages.length > 0` — no fidelity check.
- **Open test scenarios** marked unaddressed: edit-during-compression, token-estimation drift, branching with pending compression, concurrent addMessage (`TEST-SCENARIOS.md`).

### Agent-framework

The agent runtime: ProcessQueue event loop, modules with hooks, yielding inference, turn checkpoints, ephemeral subagents, MCPL host integration.

**Verified ✓**:
- Single-event-at-a-time ProcessQueue (`queue.ts:36-51`), with one twist: external-message events jump to the front (`queue.ts:24-29`)
- Module hooks: `onProcess`, `getTools`, `handleToolCall`, `start/stop`, optional `onAgentSpeech`/`gatherContext` (`README.md:84-98`, `types/module.ts`). Note: README spec says "onEvent" but actual interface uses "onProcess".
- Yielding inference with inline tool dispatch (`framework.ts:1766-1806`): stream yields `tool-calls` event; framework iterates calls in a synchronous `for` loop calling `dispatchToolCall` each, which fires `.then()` chains in parallel (`framework.ts:2275-2294`)
- Abort handling for `inferring` / `streaming` / `waiting_for_tools` states (`agent.ts:424-439`)

**Refined ~**:
- **Turn checkpoints are sequence markers, not snapshots.** `TurnCheckpoint = {agentName, turnIndex, sequenceBefore, branchName, timestamp}` (`framework.ts:83-89`). Undo (`framework.ts:1055-1111`) is `store.createBranchAt(name, currentBranch.name, sequenceBefore)` then `switchBranch` — **O(1) branch op, no replay, cheap.** Max 20 checkpoints per agent (`framework.ts:81`).
- **Ephemeral subagents are not separate processes or stores.** `createEphemeralAgent` (`framework.ts:635-659`): same `JsStore` instance, `namespace = 'subagent/${name}'`, `isolate: true`. Same Chronicle, namespaced state slots. Data persists in store under namespace after cleanup (`framework.ts:632, 653-656`).
- Zombie detection is real: 30s `STARTUP_TIMEOUT_MS` watchdog rejects with "zombie detected" if `inference:started`/`inference:tokens` never fires (`runEphemeralToCompletion`, `framework.ts:671-773`)
- **The `SubagentModule` is NOT in this repo** — `createEphemeralAgent` is a hook the framework exposes for an external module.

**Critical finding for somnigraph**:
- **undo() does NOT reverse external state.** Chronicle's branch-at-sequence undo only reverses state stored in chronicle. **MCP server state (e.g., somnigraph's SQLite) is outside undo()'s reach.** MCPL has a Section 8 stateful-checkpoint protocol (`mcpl/checkpoint-manager.ts:262-327`) for rollback support, but it requires the server to declare `hostState`/`rollback` in feature-set declaration and emit checkpoint metadata per write (host-managed JSON-Patch deltas or server-managed checkpoint IDs). **Plain MCP servers — and MCPL servers without the rollback feature-set — leave their writes committed when the framework "undoes" a turn.** JSON Patch `move`/`copy` ops are not supported (`checkpoint-manager.ts:166-169`).

**Other findings**:
- **MCPL is backward-compatible.** Plain MCP servers work without modification (`server-connection.ts:225-242`): standard MCP `initialize`, MCPL features tucked into `experimental.mcpl`. Servers without `experimental.mcpl` get `mcplCaps = null` and operate as plain MCP via `tools/list` + `tools/call` (`server-connection.ts:277, 467-481`).
- **Test coverage**: 7 test files, 2,372 LoC. Covers framework lifecycle, event-gate, mcpl-gate, channel typing, tool-policy, usage tracker, workspace-watcher. **No tests for undo/redo, ephemeral agents, or abort interrupts** — three of the most distinctive marketing claims are untested.
- Minimal TODOs (4 trivial)

### Membrane

Provider abstraction with participant-first messaging. **Antra-tess origin is canonical**; no anima-research upstream.

**Verified ✓**:
- `NormalizedMessage.participant: string` is the core type (`src/types/message.ts:39-55`). Comment: *"This is the core abstraction - no artificial 'user' vs 'assistant' roles."*
- Anthropic, Bedrock, OpenRouter, OpenAI, Gemini all supported

**Refined ~**:
- **Nine providers, not five**: Anthropic, OpenRouter, OpenAI (Chat), OpenAICompatible, OpenAICompletions, Mock, Bedrock, Gemini, OpenAIResponses (`src/providers/index.ts`)
- **Three formatters, not four**: `AnthropicXmlFormatter`, `NativeFormatter`, `CompletionsFormatter` (`src/formatters/index.ts:17-19`). **"Pseudo-Prefill" was fabricated** in the first draft — it's mentioned hypothetically in `prefill-formatter-architecture.md:200` as a possible `ChatMLFormatter` but is not implemented.

**Contradicted ✗**:
- **"If input tokens exceed maxStreamTokens mid-stream, Membrane signals overflow and the framework restarts with recompressed context"** — fabricated. Grep for `maxStreamTokens|overflow|recompress|StreamOverflow` returns zero hits in `src/`. Streaming has retry/abort plumbing but no input-budget overflow signal.

**How participant flattening actually works**:
- **XML formatter (Claude)**: participants preserved in-band as `${name}: ${text}` lines inside an assistant-role block; sent as a single concatenated assistant turn with bot's name as a prefill prefix (`anthropic-xml.ts:270, 294-297`). Auto-generates stop sequences from participant names (`buildStopSequences`, `anthropic-xml.ts:351`, `maxParticipantsForStop: 10`). Injects a fake `<cmd>cat untitled.txt</cmd>` user turn because Claude's API requires a user-role first message (`anthropic-xml.ts:328-340`) — Claude-CLI-simulation flavor.
- **Native formatter, simple mode**: enforces strict 2-party (errors on a third participant, line 213).
- **Native formatter, multiuser mode** (`native.ts:165-246`): prepends `${name}: ` only to non-assistant text blocks (`native.ts:220-225, 337-340`); merges consecutive same-role turns. **Information loss**: assistant-side participant names are dropped. Claude-A vs Claude-B collapse into role=`assistant`; only human-side names survive as content prefixes. The XML formatter preserves all names; native does not.
- Recent commit "prepend participant names in native tool streaming path" suggests participant identity in native tool flows was an active bug.

### Connectome-host

Reference TUI. **Plain MCP works as-is** — somnigraph plugs in directly.

**Verified ✓**:
- Recipe-driven, JSON schema in `src/recipe.ts:313-321`, validator at `:494-728`. Loader supports local file, HTTP URL, or saved `data/.recipe.json`. `${VAR}` and `${VAR:-default}` env substitution.
- Knowledge-mining workflow shipped: `recipes/knowledge-miner.json` (Zulip + Notion + GitLab) and `recipes/knowledge-reviewer.json` (~6kB system prompts each)
- WorkspaceModule + EventGate ✓ (imported from `@animalabs/agent-framework`)
- Fleet tree, Ctrl+B background sync subagent without killing it (`tui.ts:163, 976, 1043`)

**Stronger than claimed**:
- **Triumvirate recipe** (`triumvirate.json`, `TRIUMVIRATE-SETUP.md`): three agents — clerk (frontdesk), miner, reviewer — communicating via watched workspace mounts (tickets/ → products/ → review-output/), with `request_id`/`origin` provenance chain through YAML frontmatter so replies route back to the originating Zulip thread. Not just an example pattern — a real shipped multi-agent workflow.

**Refined ~**:
- Recipes are now **opt-in** for MCP servers (`src/index.ts:243-276`): file provides `command`/`args`/`env`/credentials; recipe overrides only policy fields (channelSubscription, toolPrefix, enabled/disabledTools, reconnect). Older docs still say "file wins" flatly (`README.md:83`, `ARCHITECTURE.md:97`) — out of date.

**Somnigraph integration**:
- connectome-host speaks **standard MCP over stdio or WebSocket** — same shape as Claude Desktop's `.mcp.json` (`mcpl-config.ts:4`). No MCPL-specific extension needed.
- Concrete plug-in: add to `mcpl-servers.json`, then reference in recipe `mcpServers`:

```json
{ "mcplServers": { "somnigraph": {
  "command": "uv",
  "args": ["run", "C:/Users/Alexis/.claude/servers/memory_server.py"],
  "env": { "OPENAI_API_KEY": "${OPENAI_API_KEY}" }
} } }
```

- **Acknowledged roadmap gaps** (`ARCHITECTURE.md:312-343`): TUI doesn't refresh after Chronicle branch ops, single-level (not pyramid) compression in AutobiographicalStrategy, MCPL server status not surfaced in TUI, MCPL-pushed events not visually distinguished from user input. Self-described purpose: "Dogfood the AF" — reference consumer, not polished product.

---

## Architecture Reading

The marketing page presents Connectome as a coherent four-layer stack with welfare-research framing baked into design. Reading the code, the picture is more textured:

- **Chronicle is real but smaller than billed.** Append-only records with branches, blob storage, state-update chains, N-API bindings — all present. But the most distinctive-sounding architectural claims (`O(log #checkpoints)` reconstruction, "Loom of Looms" nested chronicles, the `Checkpoints` primitive) live in `docs/loom-of-looms.md` as algebra, not in `src/` as code. Branches are virtual filters over one global log, not isolated stores; the global record-type index is `HashMap<String, Vec<RecordId>>` rebuilt from the log on startup. This is fine for a research artifact but very different from "Git for data" implies.
- **Context-manager is the most production-shaped piece.** Strategy interface, hierarchical compression with anti-redundancy filtering, role-spoofing for the autobiographical voice, knowledge-strategy with phase-tagged prompts. Active development. The honest framing is that it's a clean **sequential compression pyramid for one conversation**, not a memory system — there's no retrieval, no semantic search, no decay, no contradiction handling.
- **Agent-framework is the structural piece that makes the stack work.** ProcessQueue + module hooks + yielding inference + chronicle-branch-based undo. The undo() implementation is genuinely cheap (O(1) branch op). MCPL backward-compatibility with plain MCP is real and well-engineered. But the most consequential claim — that undo() rolls back the agent's interactions with the world — is incomplete: external state stays committed.
- **Membrane is participant-preserving where format permits.** XML formatter keeps full names; native multiuser drops assistant-side names. The fork situation is just authorship: antra-tess is the original.
- **Connectome-host is the most coherent end-to-end piece.** Real shipped recipes including a three-agent workflow. Plain MCP works. The dogfooding posture is honest.

The deepest tension I now see between marketing and code: the architectural language of "agents that exist as branches of an append-only record" is compelling, but the implementation has **one global log with virtual branch filters**, no per-branch persistence, and undo that doesn't reverse external state. This isn't a contradiction so much as a smaller scope than the framing implies. Which is fine — research code earns the right to be smaller than the framing — but it changes what's portable to somnigraph.

---

## What Somnigraph Could Take

Three concrete ideas, with feasibility now qualified by code reading:

### 1. Causation links at the record level

**The Connectome analog**: `causedBy[]` is optional, unenforced, and `linked_to` is a separate "soft" reference vector. The framing is *graph of causation* but the implementation is *optional metadata vec on each record*.

**For somnigraph**: a `caused_by` field on memories (event ID + type), populated by write-path code, would be a small schema change with the same shape as Connectome's. Diagnostic: trace high-utility memories back to feedback-driven sleep runs vs. direct writes; detect feedback-distribution drift in sleep-modified memories. Tracked as roadmap open question.

**Feasibility**: high. The Connectome version is small enough to copy exactly — optional vec of foreign keys to event records — without requiring chronicle's other infrastructure.

### 2. Branchable DB as research primitive

**The Connectome analog**: branches are virtual filters over one global log. Cheap branch creation (struct + map insert, no copy); querying a branch reads from the global log filtered by `BranchId`. This is *not* the copy-on-write isolated-store branching the marketing language implies.

**For somnigraph**: the analog at SQLite level is straightforward — add a `branch_id` column to `memories`/`edges`/`feedback_events`, query with appropriate filters, branch creation is one row in a `branches` table. The expensive part is keeping the FTS5/vec indexes consistent across branches — the marketing language hides that Chronicle has none of these. SQLite's WAL mode + a `branch_id` filter on every retrieval query is the simpler path; isolated DB copies via `cp` is the simplest.

**Feasibility**: high but the ROI question shifts. If branches are just filters, the value proposition vs. just copying the SQLite file is whether you want simultaneous branches in one process. For our counterfactual studies (sleep-on vs sleep-off, different feedback histories), DB copies are probably fine. Tracked as roadmap open question.

### 3. Autobiographical narrative summaries as a sleep mode

**The Connectome analog**: prompt + role spoofing + temperature 0. *"Describe it as you would to yourself, as if you are remembering what has happened."* Prior summaries replayed as `'Claude'` assistant turns. Hierarchical L1→L2→L3 with threshold 6, level-shadowing anti-redundancy. **No fidelity check.**

**For somnigraph**: the bar is lower than the marketing implied. We don't need fine-tuning — just role spoofing in the prompt and low temperature. The hierarchical merge pattern is a known recipe (level-shadowing, not deduplication). The fidelity-failure-mode concern is now empirically supported: Connectome ships this without fidelity tracking.

**Feasibility**: high for a prototype as a sleep mode. Risk: same drift Connectome has. Mitigation: comparison against source memories at merge time as a fidelity gate. Tracked as roadmap open question.

### What I'd skip

- **Chronicle as a substrate**. No Python bindings; would require PyO3 or HTTP shim. Bespoke filesystem layout means we'd lose SQLite's tooling. The "Git for data" framing oversells what's there.
- **MCPL Section 8 stateful-checkpoint protocol** for rollback support — heavy lift (per-write JSON-Patch deltas, full state snapshots, version chains) and undo() doesn't reverse external state in plain MCP anyway. If a future framework wanted to roll somnigraph back, it could, but that's downstream demand we don't have.
- **The Loom-of-Looms nested-chronicle algebra** — it's not built and wouldn't help us if it were.

---

## Comparison with Somnigraph

| Dimension | Connectome | Somnigraph |
|---|---|---|
| Storage primitive | Append-only records (bespoke FS, branched), state chains | SQLite tables: memories, edges, feedback events |
| Compression | Autobiographical (model-as-self-narrator, hierarchical L1-L3, level-shadowing anti-redundancy, no fidelity check) | Sleep pipelines (theme normalization, contradiction classification, summary refresh, decay correction) |
| Branching | Virtual filters over one global log (cheap creation, no per-branch index) | None — single timeline, edges between memories |
| Retrieval | **None in context-manager itself.** Sequential pyramid only. | RRF (FTS+vec+theme) + PPR + Hebbian PMI + LightGBM reranker (NDCG 0.7958 on 1032q GT) |
| Causation | Optional `causedBy[]` on records; soft `linked_to` vec | Edges with retrieval-shaped weights; no source-event link |
| Scope | One agent's continuous existence (with undo via cheap branch ops) | Knowledge substrate across sessions/instances |
| Time model | No decay, no dormancy, no contradiction handling | Per-memory decay rates; dormancy via sleep; contradiction edges with classification |
| Tuning regime | None visible | 20+ wm studies, learned reranker, 1032-query GT, calibrated against production feedback |
| External state | Plain MCP works; rollback requires server to implement MCPL Section 8 | N/A — somnigraph IS the external state |

**The fundamental difference is unchanged after code reading**: Connectome stores conversation; somnigraph stores knowledge. But the fence between them is more permeable than the marketing implied — connectome-host speaks plain MCP, so somnigraph could be a recipe entry tomorrow. The harder question is the inverse: would lifting from chronicle into somnigraph at session end be worth the integration cost given chronicle's Node-only bindings? Probably not, unless someone wants to consume somnigraph from connectome-host specifically.

---

## What the First Draft Got Wrong

For honest accounting, errors in the prior version of this file — generated only from the public landing page — that the code investigation corrected:

1. ✗ **"O(log #checkpoints) reconstruction"** — claimed as a chronicle property; not implemented. Real cost is O(ops since last full snapshot).
2. ✗ **"Three state strategies"** — actually five (added Tree, Struct).
3. ✗ **"Pseudo-Prefill formatter"** — invented; only three formatters exist.
4. ✗ **"maxStreamTokens overflow → recompress restart"** — fabricated; no such mechanism.
5. ✗ **"Loom of Looms"** presented as a feature — algebraic spec only, no code.
6. ~ **"Branches are copy-on-write — cheap to create, cheap to visit"** — cheap to create yes; "cheap to visit" misleading because branches are virtual filters over one global log, not isolated stores.
7. ~ **"Records of a model's life should survive the model"** as architectural commitment — true at the chronicle layer, but undo() doesn't reverse external state, which means the framework's notion of "what happened" silently diverges from external services' notion of the same.
8. ~ The autobiographical "voice" as a deep architectural commitment — actually role spoofing + temperature 0 in a normal API call. Fine framing for the agent's experience; less impressive as engineering than the marketing implies.

The general failure mode in the first draft: dressing an architectural-overview reading at the format/depth of a code-reading analysis. The frontmatter cited an irrelevant field note as a source. Most of the *descriptive* claims about what Connectome is were sourceable to the landing page, but the page is detailed enough to look like documentation while skipping the gaps between framing and implementation. Reading code is what closed the gap.

---

## Assessment

Connectome is interesting research code with a genuine welfare-research stance, run by a small group doing active iteration. The code is honest about its scope — minimal TODOs, acknowledged roadmap gaps, no Windows binaries, no top-level READMEs in some repos, dogfooding posture in the host. The marketing page significantly oversells the chronicle layer and invents some properties wholesale; the strongest pieces are context-manager (real, working compression strategies) and connectome-host (real, working multi-agent recipes consuming plain MCP).

For somnigraph specifically:
- The three roadmap open questions (causation links, branchable DB, autobiographical sleep mode) are still worth tracking, but the bar from Connectome is lower than the first reading suggested.
- The most concrete useful thing is that **somnigraph could be a connectome-host recipe entry tomorrow** — plain MCP works as-is. If anyone wants to actually consume somnigraph from a Connectome-style stateful-agent runtime, the integration is one JSON entry away.
- The most interesting empirical question: does context-manager's autobiographical compression hold up over long runs without fidelity checks? The risk is real and Connectome ships without addressing it. Watching that play out is useful evidence for the autobiographical-sleep-mode question.

The deeper observation stands: Connectome and Somnigraph operate on different theories of what to preserve. Connectome's bet is process-shaped continuity within a single agent's life; somnigraph's bet is knowledge-shaped continuity across sessions of the same agent. Code reading sharpens but doesn't change that frame.
