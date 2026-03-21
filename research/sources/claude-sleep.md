# claude-sleep (jesung/claude-sleep) — Analysis

*Generated 2026-03-21 by Opus agent reading local clone*

---

## Repo Overview

**Repo**: https://github.com/jesung/claude-sleep
**License**: Unlicensed (no LICENSE file)
**Language**: Shell (hooks) + Markdown (prompt)
**Description**: "Structured memory consolidation for Claude Code"
**Author**: jesung

**Problem addressed**: Claude Code's native auto-memory is selective and unstructured. Sessions that don't trigger Claude's built-in memory heuristics lose context. This skill adds an explicit two-layer memory pattern — daily notes as short-term memory, a curated `MEMORY.md` as long-term — with automated background consolidation.

**Core approach**: A Claude Code slash command (`/sleep`) containing a 4-step consolidation prompt, plus two hooks that automate hourly execution. No database, no embeddings, no retrieval pipeline. The entire system is one markdown prompt file and one hooks config.

**Maturity**: Minimal. Three files total (`sleep.md`, `hooks.json`, `README.md`). No tests, no benchmarks, no configuration beyond the throttle interval. Linux-only (POSIX shell dependencies). Explicitly credits OpenClaw for the two-layer pattern idea, noting that OpenClaw described but never automated consolidation.

---

## Architecture

### Two-Layer Memory

| Layer | File | Role |
|-------|------|------|
| Short-term | `memory/YYYY-MM-DD.md` | Raw session notes, written by the agent during conversations |
| Long-term | `MEMORY.md` (project root) | Curated distillation, updated by the sleep process |

The agent is instructed (via CLAUDE.md guidance) to write daily notes when it completes tasks, learns preferences, makes mistakes, or receives standing instructions. One sentence per entry is sufficient.

### Consolidation Process

The `/sleep` prompt is a 4-step procedure:

1. **Read**: Find all `memory/YYYY-MM-DD.md` files from the last 7 days
2. **Filter**: Extract significant decisions, recurring patterns, useful context, mistakes to avoid. Skip completed tasks, ephemeral state, duplicates, noise.
3. **Update**: Read current `MEMORY.md`, then add new entries (with date, one paragraph max), update changed entries, remove stale entries.
4. **Report**: Brief confirmation of what changed.

The consolidation is performed by Claude itself — the prompt instructs the LLM to make all judgment calls about what's worth keeping. There is no programmatic filtering, scoring, or embedding.

### Hook Automation

Two Claude Code hooks:

**UserPromptSubmit**: Seeds a timestamp file (`/tmp/.claude_sleep_last`) on first prompt if it doesn't exist, preventing immediate sleep trigger on session start.

**Stop**: After each Claude response, checks if 3600 seconds (1 hour) have elapsed since last sleep. If so, spawns a background Claude session:
```bash
CLAUDE_SLEEP_SESSION=1 claude --dangerously-skip-permissions -p "$SLEEP_CMD" &
```

The `CLAUDE_SLEEP_SESSION=1` environment variable prevents the background session from recursively triggering its own sleep. The `--dangerously-skip-permissions` flag allows unattended writes to `MEMORY.md`.

### What It Doesn't Have

- No embeddings or vector search
- No retrieval pipeline — `MEMORY.md` is loaded wholesale at session start
- No feedback loop
- No decay or forgetting beyond the consolidation prompt's "remove stale entries" instruction
- No graph structure or associations
- No scoring or ranking
- No deduplication beyond LLM judgment
- No persistence beyond markdown files

---

## Relevance to Somnigraph

### Minimal direct relevance

claude-sleep operates at a completely different layer than Somnigraph. It's a consolidation prompt pattern, not a memory system. The "retrieval" is loading a markdown file into context; the "consolidation" is an LLM reading notes and rewriting a file.

### What's mildly interesting

**The hook pattern for automated background consolidation.** Spawning a background Claude session on a timer to perform maintenance work is a clean integration pattern. Somnigraph's sleep pipeline is invoked manually; this automates the trigger. The approach is fragile (POSIX-only, relies on `/tmp/` state files, `--dangerously-skip-permissions`), but the concept of hooks-triggered background consolidation is sound.

**The two-layer framing as a teaching tool.** "Daily notes are your journal; MEMORY.md is the distilled wisdom you'd want to wake up knowing" is an effective metaphor for explaining CLS (complementary learning systems) to non-technical users. The README's comparison with Claude Code's native auto-memory is clearly written and honest about limitations.

**The 7-day sliding window.** Consolidation only reads the last 7 days of notes. This is a simple but effective attention boundary — old daily notes age out of the consolidation window naturally. Somnigraph's sleep pipeline processes all unprocessed memories regardless of age, which is more thorough but potentially less efficient for maintenance runs.

### Not worth importing

Everything else. The system has no programmatic intelligence — it's entirely dependent on the LLM's judgment about what to keep, what to discard, and what's stale. This is the approach Somnigraph explicitly moved beyond with structured classification, embedding-based deduplication, and learned scoring.

---

## Summary

claude-sleep is a minimal consolidation pattern, not a memory system. It's useful as a reference for how to wire background consolidation into Claude Code's hook system, and as a clear articulation of the two-layer (short-term notes / long-term curated memory) pattern. There are no technical ideas to borrow for Somnigraph's retrieval or consolidation pipelines.
