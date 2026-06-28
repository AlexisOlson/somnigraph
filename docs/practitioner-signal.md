# Practitioner signal

What people running similar systems in the wild have corroborated or challenged about Somnigraph's design choices. Sourced from public discussion of Karpathy's "LLM Wiki" pattern (mid-2026): the [gist thread](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f), r/ObsidianMD, and r/ClaudeAI.

Honesty caveat: most of that discourse is either philosophy ("does AI organizing my notes ruin *my* thinking?") or self-promotion. This file is the filtered residue — the parts that bear on our actual design, with the single-anecdote claims flagged as such. The Karpathy pattern is also a *document-KB* pattern, not a conversational-memory one; only the transferable subset is here. The document-KB landscape itself is a sibling project and lives in the vault, not this repo (see *Off-mission* below).

## On-mission corroborations

Each maps to an existing decision or priority; none is a new idea we hadn't reached. The value is independent convergence.

1. **Deterministic checks beat prose instructions for anything load-bearing.** The sharpest formulation: *"for anything load-bearing, a deterministic check you run every loop beats a sentence you hope is still in the context window"* (rapter200, r/ClaudeAI). Corroborated by hmbseaotter's contradiction architecture (a commit gate implemented as `os.walk` + grep for an `Unresolved` marker — zero context cost — rather than an LLM pass) and by the Superpowers hook ecosystem. This is exactly why `recall()` returns the feedback IDs in its output footer rather than trusting the model to remember to rate — the brake is coded into the tool surface, not left to a prose reminder. Reinforces the roadmap's auto-capture/forced-recall hook direction over passive tool exposure (cf. the MEMTRACK finding in `research/sources/`).

2. **Instruction decay over long context is real.** Rules buried tens of thousands of tokens back get quietly dropped; *"2-3 hard rules repeated beat 5 nuanced ones"* (Altruistic_Pound3237, and the auto-generated thread summary). This is the *why* behind keeping the CLAUDE.md snippet to ~30 front-loaded lines rather than inlining the whole judgment layer. Added as a failure mode in `claude-md-guide.md`.

3. **Confident-but-stale is the binding failure mode, not capacity.** From someone running a ~1.8k-file memory store: *"past ~day 60 the real failure mode is confident-but-stale memory making the agent worse, not better"* (LARIkoz / Eidetic, gist thread). Independent support for the decay floor + dormancy design (`docs/architecture.md`): the danger isn't running out of room, it's old memories retrieved with undiminished confidence. Argues against pure-accumulation memory and for staleness penalties as first-class.

4. **Contradiction handling wants tiered severity + scoped checking.** hmbseaotter's pattern (gist thread): classify each new claim as soft / scope-mismatch / hard at ingest; block only on hard; run the expensive reasoning lint only over the changed node plus its 1st/2nd-degree graph neighbors, not the whole store. A complementary design (demartinogiuseppe / Antinomia) treats a contradiction as its own *node* carrying the proof that resolved it, rather than an edge. Both bear on our graded contradiction detection and sleep consolidation — the neighborhood-scoping is a concrete cost-control pattern worth considering for consolidation (see *Roadmap implications*).

5. **Linting/consolidation is the token sink as the graph grows.** Multiple reports that whole-store contradiction/lint passes become slow and expensive at scale (tracagnotto and others). Corroborates keeping consolidation offline (sleep) and scoped, never per-turn.

## Off-mission (document-KB branch)

The bulk of the Karpathy discourse is about a *document* wiki — a separate, AI-maintained vault distinct from a conversational memory system. Recurring patterns there: keep an agent vault separate from the human vault; a `hot.md` boot cache (a ~150-word running summary read at session start instead of scanning the vault); strict epistemic discipline (every claim cites an immutable source or is flagged as a gap); and [`qmd`](https://github.com/tobi/qmd) (local BM25 + vector + LLM-rerank over markdown) as the search layer. These belong to the document-KB landscape, which is tracked in the vault (`Memory Research/kb-landscape.md`), not reproduced here — Somnigraph is a conversational-memory system and that is a sibling problem.

## Roadmap implications (candidate — for review)

Proposals, not decisions:

- **Neighborhood-scoped contradiction lint in sleep.** If consolidation ever does a graph-wide contradiction pass, bound it to changed nodes + graph neighbors (per #4) before it becomes a cost problem.
- **Keep coding load-bearing memory hygiene deterministic.** Where a behavior matters (feedback rating, dedup-on-write), prefer a coded gate or tool-surface nudge over a snippet sentence (per #1). The `recall()` feedback-ID footer is the model to extend.

## Sources & flags

- Karpathy LLM-Wiki gist + comment thread: <https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f>
- r/ObsidianMD "What's the deal with the hype around Karpathy's LLM wiki?" and "Karpathy's LLM Wiki setup" (Apr/Jun 2026).
- r/ClaudeAI "I added a clause to Andrej Karpathy's 4 CLAUDE.MD clauses" (Jun 2026).
- **Flagged single-anecdote:** the day-60 staleness figure (#3) is one practitioner's experience, not a measurement.
- **Flagged needs-verification:** a Microsoft paper, cited in-thread as "LLMs Corrupt Your Documents When You Delegate," is claimed to show repeated-edit degradation of documents. If real, it is the empirical backing for source-immutability; confirm the citation before relying on it.
- **Flagged not-Karpathy:** the widely-circulated "Karpathy's 4 CLAUDE.md clauses" are community-derived; Karpathy framed them as mistakes that bite him, not authored rules.
</content>
