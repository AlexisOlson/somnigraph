# Claude Subconscious Analysis

*Generated 2026-03-28 by Opus agent reading GitHub repo*

---

## 1. Architecture Overview

**Repo**: https://github.com/letta-ai/claude-subconscious
**Stars**: ~2,115 (as of 2026-03-28)
**Forks**: 151
**License**: MIT
**Language**: TypeScript (Node.js 18+)
**Created**: 2026-01-14, effectively a single squashed commit
**Description**: "A subconscious for Claude Code. A Letta agent watches your sessions, accumulates context, and whispers guidance back."
**Version**: 2.0.2

This is not a memory system. It is a Claude Code plugin that connects Claude Code to the Letta platform via hooks, allowing a background Letta agent ("the Subconscious") to observe session transcripts, update its memory blocks, and inject guidance back into Claude's context. The actual memory storage and retrieval happens server-side on Letta's infrastructure (or a self-hosted Letta server). The repo itself is ~1,300 lines of TypeScript glue code.

**Key distinction from Letta proper**: Our [existing Letta analysis](letta.md) covers the full Letta platform — its three-tier memory hierarchy, agent loop, sleeptime agents, etc. Claude Subconscious is a thin integration layer that uses Letta as its backend. The novel contribution is the *integration pattern* (hooks, transcript forwarding, whisper injection), not the memory model.

**Module organization**:
```
scripts/
  session_start.ts       — SessionStart hook: creates conversation, sends start notification
  sync_letta_memory.ts   — UserPromptSubmit hook: fetches memory blocks + messages, injects via stdout
  pretool_sync.ts        — PreToolUse hook: checks for mid-workflow updates
  send_messages_to_letta.ts — Stop hook: parses transcript, spawns background worker
  send_worker_sdk.ts     — Background worker: sends transcript via Letta Code SDK
  agent_config.ts        — Agent resolution, model auto-detection, .af import
  conversation_utils.ts  — State management, API helpers, XML formatting
  transcript_utils.ts    — JSONL transcript parsing, message formatting
hooks/
  hooks.json             — Claude Code hook registration
Subconscious.af          — Bundled default agent definition (56KB JSON)
```

**Dependencies**: `@letta-ai/letta-code-sdk` (client-side tool execution), `tsx` (TypeScript runner). No database, no embeddings, no retrieval logic — all of that lives on the Letta server.

---

## 2. Memory Model

The plugin itself has no memory model. It delegates entirely to the Letta agent's memory architecture. The bundled default agent uses Letta's standard block-based memory with 8 named blocks:

| Block | Limit | Purpose |
|-------|-------|---------|
| `core_directives` | 5,000 chars | Agent role and behavioral guidelines |
| `guidance` | 3,000 chars | Active whisper message for next session |
| `user_preferences` | 3,000 chars | Learned coding style, tool preferences |
| `project_context` | 3,000 chars | Codebase knowledge, architecture decisions |
| `session_patterns` | 3,000 chars | Recurring behaviors, time-based patterns |
| `pending_items` | 3,000 chars | Unfinished work, TODOs |
| `self_improvement` | 5,000 chars | Memory architecture evolution guidelines |
| `tool_guidelines` | 5,000 chars | Tool usage patterns |

Hard limits: 12 memory files max, 30,000 total characters. These are tiny by any standard — the entire agent memory is smaller than a single Somnigraph recall result set.

There is no decay, no consolidation, no automatic cleanup. The agent's `self_improvement` block instructs it to manually prune stale information ("hasn't been relevant in 10+ sessions, cut it") and consolidate blocks when they overlap. Memory hygiene is entirely LLM-driven.

The agent's `conversation_search` tool can search all past messages across all sessions, providing a form of recall memory. But there is no learned ranking, no feedback loop, no graph structure over the stored messages.

---

## 3. Retrieval Pipeline

There is no retrieval pipeline in this repo. The Letta server handles retrieval when the agent calls `conversation_search` (hybrid vector + FTS with RRF, as documented in our Letta analysis) or `archival_memory_search` (vector-only cosine similarity). The plugin's contribution is limited to fetching the agent's current memory blocks and new messages via the Letta API, then formatting them as XML for stdout injection.

The sync flow:
1. **UserPromptSubmit**: Fetch agent blocks and conversation messages from Letta API. On first prompt, inject all blocks. On subsequent prompts, inject only changed blocks (line-based diff) and new assistant messages.
2. **PreToolUse**: Lightweight check for new messages or block changes since last sync. Injects via `additionalContext` if anything changed. Silent no-op otherwise.
3. **Stop (async)**: Parse Claude Code's JSONL transcript, extract user messages, assistant responses, thinking blocks, and tool usage. Format as XML, send to Letta agent via SDK. Agent processes transcript, updates its blocks, generates a response that will be picked up by the next sync.

The diff algorithm is rudimentary: set-based line comparison (lines present in old but not new = removed, vice versa). No edit distance, no context lines. Good enough for small blocks.

---

## 4. Write Path

Writes happen in two directions:

**Claude Code -> Letta** (transcript forwarding): The Stop hook reads the session transcript (JSONL), extracts messages with role attribution, truncates tool results to 1,500 chars and thinking blocks to 500 chars, formats as XML, and sends as a single user message to the Letta conversation. The background worker uses the Letta Code SDK, which gives the agent client-side tool access (Read, Grep, Glob by default) so it can explore the codebase while processing the transcript.

**Letta -> Claude Code** (whisper injection): The agent's responses and block updates are fetched by the sync hooks and injected into Claude's context via stdout. Messages appear as `<letta_message>` XML elements. Block changes appear as diffs wrapped in `<letta_memory_update>`. The plugin explicitly instructs Claude to acknowledge messages: "Briefly acknowledge what Sub said — just a short note like 'Sub notes: [key point]'."

There is no write path for Claude Code to directly update Letta blocks. Communication is indirect: Claude writes something in its response, the transcript is forwarded to the Letta agent, the agent decides whether to update its blocks.

---

## 5. Comparison to Somnigraph

### What they have that we don't

| Feature | Claude Subconscious | Somnigraph |
|---------|-------------------|------------|
| **Background agent with tool access** | Letta agent can Read/Grep/Glob the codebase while processing transcripts | No background agent; all processing is synchronous or in sleep pipeline |
| **Multi-session shared memory** | One agent serves multiple Claude Code sessions simultaneously with shared blocks | Single-agent, single-session design |
| **Transcript-driven observation** | Full session transcript (including thinking blocks and tool usage) forwarded to background agent | Only explicit `remember()` calls create memories; no passive observation |
| **Mid-workflow injection** | PreToolUse hook injects updates between tool calls | No mid-conversation context injection |
| **Personality/rapport framing** | Agent designed to "develop perspective," "share partial thoughts," "have opinions" | Functional tool interface; no personality layer |
| **Zero-config setup** | Install plugin, set API key, agent auto-imports | Requires manual CLAUDE.md configuration, understanding of tool semantics |

### What we have that they don't

| Feature | Somnigraph | Claude Subconscious |
|---------|-----------|-------------------|
| **Retrieval quality** | 4-signal hybrid (BM25 + vector + graph + theme), RRF fusion, 26-feature LightGBM reranker, R@10=95.4% on LoCoMo | Letta's vector + FTS with RRF; no learned ranking, no graph, no feedback |
| **Biological decay** | Per-category exponential decay, reheat on access | No decay; manual cleanup instructions |
| **Sleep consolidation** | Three-phase offline pipeline (NREM classification, REM summarization, maintenance) | No offline processing; agent updates blocks during transcript processing |
| **Feedback loop** | EWMA + UCB exploration bonus on retrieval results, utility calibration (r=0.70) | No feedback mechanism |
| **Graph structure** | Typed edges (support, contradict, evolve, derive), Hebbian co-retrieval strengthening, PPR expansion | Flat block storage, no relationships |
| **Evaluation methodology** | Ground-truth judging, NDCG metrics, reranker CV, LoCoMo benchmark (85.1% overall) | No evaluation framework |
| **Memory capacity** | ~700+ memories with rich metadata, unbounded growth with decay management | 30,000 chars total (~5 pages of text) |
| **Self-contained** | SQLite + sqlite-vec, no external dependencies | Requires Letta server (cloud or self-hosted) |

### Architectural trade-offs

| Dimension | Claude Subconscious | Somnigraph |
|-----------|-------------------|------------|
| **Integration point** | Claude Code hooks (plugin) | MCP server (tool calls) |
| **Memory location** | Remote (Letta server) | Local (SQLite) |
| **Processing model** | Online, asynchronous (background agent) | Synchronous + offline batch (sleep) |
| **Intelligence location** | LLM decides everything (what to store, retrieve, update) | System decides scoring/ranking/decay, LLM decides what to store |
| **Observation model** | Passive (sees full transcript) | Active (user/agent must call remember()) |
| **Scalability** | Multi-user, multi-agent, cloud-native | Single-user, single-agent, local |
| **Failure mode** | Network latency, API availability, LLM judgment errors | SQLite corruption (rare), model file versioning |
| **Cost** | LLM calls for every transcript (background agent runs inference) | LLM calls only for embedding + sleep consolidation |

---

## 6. Worth Adopting?

**The passive observation pattern** — MEDIUM interest, NOT worth adopting directly.

The idea that a background agent watches full session transcripts is appealing for capturing context that explicit `remember()` calls miss. However, the implementation has a fundamental cost problem: every Claude Code response triggers a full LLM inference cycle on the background agent (transcript parsing, tool calls, block updates). For a heavy session, this could easily double the token cost. Somnigraph's explicit `remember()` approach is cheaper and more precise — the human decides what matters, not a background LLM.

That said, there is a middle ground worth considering: the sleep pipeline already processes session data offline. If the sleep pipeline could access Claude Code session transcripts (not just stored memories), it could extract memories that the user forgot to `remember()`. This is already partially what the REM phase does with existing memories, but it doesn't have access to raw session logs.

**Mid-workflow context injection** — LOW interest.

The PreToolUse hook that injects updates between tool calls is clever for the multi-agent case (another agent discovered something useful while you were working). For Somnigraph, this is irrelevant — there is no background agent generating mid-session updates. The MCP `recall()` already serves the "get context when you need it" use case.

**Transcript formatting** — LOW interest, useful reference.

The transcript parsing code (`transcript_utils.ts`) shows a practical approach to extracting structured data from Claude Code's JSONL format. The truncation heuristics (1,500 chars for tool results, 500 for thinking blocks) are reasonable defaults. Could be useful if Somnigraph ever processes session transcripts directly.

**The personality/rapport framing** — Interesting philosophically but orthogonal to Somnigraph's concerns. The instructions to "develop perspective" and "share partial thoughts" are about shaping the agent's communication style, not about memory quality. Somnigraph's value proposition is retrieval quality, not personality.

---

## 7. Worth Watching

**Adoption-driven iteration**: With 2,115 stars and active use, the Letta team will get real feedback on what persistent memory across sessions actually needs. Their user base is much larger than Somnigraph's. Watch for patterns in their issues and feature requests — they may surface problems we haven't hit yet.

**Letta Code SDK evolution**: The SDK that gives the background agent tool access (`@letta-ai/letta-code-sdk`) is a new abstraction for agent-to-agent communication within Claude Code. If this stabilizes, it could enable patterns beyond what MCP servers can do.

**Multi-project memory**: The "one brain, many projects" model (shared agent across repos) is an interesting architectural bet. If it works well in practice, it validates cross-project memory as a user need. Somnigraph already handles this (single DB serves all contexts) but hasn't specifically optimized for it.

---

## 8. Key Claims

| # | Claim | Evidence | Assessment |
|---|-------|----------|------------|
| 1 | "Remembers across sessions, projects, and time" | Memory blocks persist on Letta server across conversations. Conversation isolation per Claude Code session with shared blocks. | **Verified** — standard Letta block persistence. No novel mechanism. |
| 2 | "Gets smarter the more you use it" | Agent has `self_improvement` block with learning procedures (scan for corrections, note patterns, capture preferences). | **Plausible** — depends entirely on the background LLM's ability to follow these instructions consistently. No evaluation data. No mechanism prevents regression (overwriting good blocks with bad updates). |
| 3 | "Never blocks — runs in the background" | Stop hook spawns detached background worker, exits immediately. Async hook flag prevents blocking Claude Code. | **Verified** — the code clearly implements non-blocking async processing. The `child.unref()` pattern and detached spawn are correct. |
| 4 | "Not just a memory layer — a background agent with real tool access" | SDK worker gives agent Read/Grep/Glob access. Can spawn sub-agents in `full` mode. | **Verified** — the SDK integration is real. However, tool access is limited by the SDK session's permissions and the 120s Stop hook timeout. |
| 5 | "Pattern detection — 'You've been debugging auth for 2 hours, maybe step back?'" | No code implements temporal pattern detection. Depends on the LLM agent noticing patterns in transcripts. | **Aspirational** — there is no pattern detection mechanism. The agent receives transcripts and might notice patterns, but this is entirely dependent on LLM inference quality with no supporting infrastructure. |
| 6 | "Watches every Claude Code session transcript" | Stop hook reads transcript JSONL, extracts all message types, sends to Letta. | **Verified** — transcript forwarding is implemented and thorough (user messages, assistant responses, thinking blocks, tool usage). |

---

## 9. Relevance to Somnigraph

**Rating: Low**

Claude Subconscious is a product integration, not a memory system. The actual memory model is Letta's standard block architecture (already analyzed in `letta.md`), and the novel contribution — the Claude Code plugin glue — solves a different problem than Somnigraph addresses.

The 2,115 stars reflect distribution advantages (Letta's brand, Claude Code marketplace, zero-config setup) more than memory innovation. The entire memory capacity (30,000 chars) is less than what Somnigraph returns in a single generous `recall()`. The retrieval is Letta's standard approach with no learned ranking, no graph, no feedback.

That said, the popularity validates a real user need: people want persistent memory for Claude Code sessions without manual configuration. Somnigraph serves a different audience (someone building their own memory system who wants to understand the design space), but the adoption signal is worth noting.

The one genuinely interesting pattern is passive transcript observation — the idea that memories should emerge from full session context, not just explicit storage calls. This aligns with how biological memory works (you don't consciously decide to remember most things). But the execution here (full LLM inference on every transcript) is expensive, and Somnigraph's offline sleep pipeline is a more efficient place to add this capability if desired.
