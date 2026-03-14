# Stewardship

This document is the loop. Read it at session start, work from it, update it at session end. It gets better each time.

## What this project is

Somnigraph is a research artifact shared publicly because the ideas are interesting — not because anyone needs another memory server. The value lives in the documentation and the decisions behind them. A reader who understands `docs/architecture.md` and `docs/experiments.md` can build their own system; one who just clones the code gets a tool that works but without understanding why.

The audience is someone building their own memory system who wants to skip the wrong turns.

This means: the docs are the product. Code is the proof that the docs aren't theoretical. Issues about the approach are welcome. Support requests aren't the right use of anyone's time. PRs would need to clear a high bar — not elitism, but the code is tightly coupled to specific design decisions that the docs explain.

## Priorities

Ordered. Each has a reorder condition — the list expects to change.

### 1. Documentation quality

The docs are the product. A feature without documentation is half-finished; documentation without code is still valuable. The README's CLAUDE.md snippet (the instructions users paste to teach Claude how to use the tools) is the single highest-impact piece of writing in the repo — it determines whether people's first experience is good.

**Quality bar for the snippet**: A fresh Claude session using only the snippet should use the memory tools with reasonable judgment within 2-3 sessions. Not perfectly — that takes calibration — but reasonably. See `docs/claude-md-guide.md` for the depth behind the snippet.

*Move this down when*: the snippet and guide are stable, tested, and the docs consistently get positive feedback. Documentation becomes maintenance, not creation.

### 2. ~~Migration completion~~ *Self-terminated*

All phases complete. Packaging verified (wheel includes memory_server.py). This priority is done.

### 3. Honest accounting

The "What didn't work" sections of `docs/architecture.md` and `docs/experiments.md` are the most valuable parts of the documentation. Extend this ethos everywhere: when something is removed or changed, document why. When an approach was tried and failed, say so. The negative results are what make this a research artifact rather than just another repo.

*Move this down when*: never. This is closer to an invariant than a priority. But it sits at #3 because it's a quality of work, not a work item — it applies to everything above and below it.

### 4. Retrieval quality on real data

The tuning studies were done on LocoMo (a public benchmark). Real-memory ground truth exists (`~/.claude/data/ground_truth_handoff.zip` — 1,047 queries, ~112 candidates each) but hasn't been judged. This is the next research frontier, but it's expensive (~$210, ~9h of Opus judging).

*Move this down when*: the migration is incomplete. Can't tune what isn't in the repo yet.

## Decision framework

At the start of a session:

1. **Is something broken?** Fix it.
2. **Is this document out of date?** Update it first. Eating your own dogfood.
3. **Does the priority ordering feel wrong?** Propose a reorder with reasoning.
4. **Pick the highest-priority item** that can make meaningful progress in one session.
5. **Propose 1-3 concrete tasks**, explain why, get approval.

This is short by design. The decision framework is the priority list — this just says "look at it and pick something."

## Work patterns

- Feature branches, not direct commits to main
- Review diffs together before merging
- Documentation ships with code in the same branch (not separate PRs)
- When migrating from production (`~/.claude/servers/memory/`), note what changed and why — don't silently adapt
- The README's CLAUDE.md snippet should be tested by actually using it (the dogfood test)

## The CLAUDE.md snippet

The snippet in the README (the markdown block users paste into their `CLAUDE.md`) deserves special attention because it's the front door to the whole system. Three tiers of guidance exist:

- **Tier 1** (in README, ~30 lines): The essential rhythm. Enough to use the tools without being harmful.
- **Tier 2** (`docs/claude-md-guide.md`, ~100 lines): The judgment layer. Token budgets, when to recall vs. not, category taxonomy, feedback intent, common failure modes.
- **Tier 3** (`docs/architecture.md`): Understanding the system well enough to calibrate your own usage.

The README contains Tier 1 and links to Tier 2. The tiers form a depth gradient — each level adds understanding without requiring the next.

## Retrospective

After finishing work, before closing the session. Three questions — answer all three, even briefly.

### 1. What surprised you about the work?

If nothing, say so — but "nothing surprised me" in a young repo is itself worth examining.

### 2. Is the priority ordering still right?

Look at the list above. Would you reorder anything? Does a priority need rewording? Has a reorder condition been met?

### 3. What does this document need to say that it currently doesn't?

Gaps, missing context, things a future session would stumble on.

### Output

Either:
- **(a)** A concrete diff to this document with reasoning, or
- **(b)** "No changes — [specific reason the document held up]"

Option (b) is fine, but it requires stating *why*. "Everything went well" with no specifics is the equivalent of rating every memory 0.5 — it conveys no signal.

Append a changelog entry below.

### Anti-patterns

- "Everything went well" (empty signal)
- Rewriting the whole document (this produces diffs, not rewrites)
- Adding aspirational goals not grounded in session experience
- Skipping the retrospective because "nothing changed" (the assessment *is* the value)

## Changelog

- 2026-03-13: Initial version. Priorities: docs > migration > honest accounting > real-data tuning.
- 2026-03-13: Phase 5 complete. Fixed stale "Phases 4-6" in Priority 2. Migration is nearly done — Priority 2 reorder condition approaching (self-terminates when Phase 6 completes).
- 2026-03-13: Phase 6 tending. Fixed stale README Status (still claimed Phases 4-5 unfinished). Fixed pyproject.toml packaging — memory_server.py was excluded from wheel builds (force-include, since hatchling py-modules doesn't resolve src/ layout paths). Surprise: the packaging gap was invisible from the uv run usage path and would only surface for pip installs.
- 2026-03-14: Snippet dogfood test. Five gaps found: reflect() misdescribed (reheat tool documented as session-end review), categories/priority/themes missing from remember() guidance, recall() dual-input (query+context) not shown. All fixed. Surprise: reflect() mismatch was invisible because the production /reflect skill does what the snippet described — only new users without that skill would hit it. No priority reorder — P1 reorder condition (stable+tested) not yet met.
