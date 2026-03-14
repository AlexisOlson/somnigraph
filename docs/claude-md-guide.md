# Memory usage guide

Deeper guidance for using Somnigraph's memory tools effectively. The README has the essential rhythm; this document explains the judgment behind it.

## Token budgets

`startup_load()` and `recall()` accept a `budget` parameter (in tokens) that controls how much context is returned. This matters because context is finite and most valuable at session start.

- **`startup_load(3000)`** — Session start. Be parsimonious. You're loading the highest-priority memories into the most valuable context real estate.
- **`recall("topic", 2000)`** — Mid-session default. Scale with need: a quick factual check might only need 500; exploring a complex topic with lots of prior context might warrant 4000+.
- **`recall()` during reflection** — Be generous. The session is ending and context is about to be discarded anyway.

## When to recall

Recall when prior context would change your response. Specific triggers:

- The user references past work ("didn't we...", "what was the decision on...")
- You're starting work where prior gotchas or decisions likely exist
- You're about to recommend an approach and stored experience might disagree

Don't recall:
- What `startup_load` already covered (it loads the highest-priority memories)
- To narrate that you're checking memory — just do it and use the results naturally

## Feedback: the selection mechanism

`recall_feedback({id: score, ...})` is how the system learns. Every score shapes what surfaces in future sessions. This is not bookkeeping — it's the selection pressure that produces curvature in retrieval over time.

- **Rate all surfaced memories.** Unused memories get 0.0. This tells the system they weren't relevant in that context.
- **Recent ratings dominate.** The system uses EWMA (exponential weighted moving average, alpha=0.3), so a memory's utility score reflects recent experience, not a lifetime average.
- **Cold-start is handled.** New memories without feedback use an empirical Bayes prior fitted from the population — you don't need to worry about bootstrapping.

Scoring guidance:
- `1.0` — directly answered a question or prevented a mistake
- `0.7-0.9` — useful context that shaped the response
- `0.3-0.5` — tangentially relevant, didn't hurt
- `0.0` — surfaced but not used

## What to store

Use `remember()` for things a future session would need that aren't derivable from the codebase or git history.

**High value** (store immediately):
- Corrections from the user — prevents repeated mistakes across sessions
- Verified fixes and gotchas — but only after the fix is confirmed working
- Decisions with reasoning that would be lost without context

**Lower value** (consider during reflection):
- Observations about workflow or patterns
- Questions worth tracking across sessions

**Don't store**:
- One-off facts (look them up again)
- Things derivable from code or git log
- Unverified guesses (wait until confirmed)
- Debugging details (the fix is in the code; the commit message has the context)

## Memory categories

Each memory has a `category` that determines its default decay rate and retrieval behavior.

| Category | Decay | Use for |
|----------|-------|---------|
| `episodic` | Medium | Events, decisions, milestones — things that happened |
| `procedural` | Slow | How-to knowledge, corrections, gotchas — things you do |
| `semantic` | Slow | Facts, reference material, domain knowledge — things you know |
| `reflection` | Slow | Insights, patterns, lessons learned — things you've understood |
| `meta` | None | System-level notes, questions, configuration — things about the system |
| `entity` | None | Named entity summaries (people, projects) — retrieval hubs for PPR |

When in doubt, use `episodic` for things that happened and `procedural` for things you learned.

## Priority

Priority (1-10) affects `startup_load` ordering and scoring. Use the full range:

- **8-10**: Corrections, critical gotchas, active project context
- **5-7**: Decisions, verified patterns, useful background
- **3-4**: Reference material, less-active context
- **1-2**: Low-confidence observations, speculative notes

## Linking memories

`link()` creates edges between related memories. These edges are what Personalized PageRank traverses during recall — linking two memories means recalling one helps surface the other.

Link when:
- A new memory is clearly related to an existing one (a decision and its rationale, a gotcha and the project it applies to)
- You notice during recall that two memories should be connected but aren't

Don't over-link. The graph should reflect genuine relationships, not "these were in the same session."

## Consolidation

`consolidate()` triggers sleep-like offline processing. This is a heavy operation — it merges similar memories, detects gaps, and runs maintenance. Use it sparingly:

- After a batch of related `remember()` calls
- When `memory_stats()` shows high memory count or low health metrics
- Not every session — weekly or when needed

## Common failure modes

**Storing too much.** Every memory competes for retrieval slots. Storing low-value memories dilutes the signal. Be selective.

**Never rating feedback.** Without `recall_feedback`, the system can't learn which memories are useful in which contexts. The feedback loop is load-bearing.

**Recalling what startup already covered.** `startup_load` returns the highest-priority active memories. Immediately calling `recall` with the same topics wastes context.

**Narrating memory operations.** "Let me check my memories about..." adds no value. Just recall and use the results.

**Storing unverified fixes.** A fix that hasn't been confirmed working can mislead future sessions. Wait until it works, then store.
