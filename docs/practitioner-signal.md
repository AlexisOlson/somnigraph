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

## Source-scan addendum (2026-06-28)

Extended by two parallel sessions: a practitioner/benchmark source-scan and a community repo sweep (9 repos; full detail in the vault's Memory Research Phase 18, not this repo). On-mission findings:

**Write-path quality is the lever, not retrieval sophistication — the headline.** The two LoCoMo/LongMemEval leaders win on the write side: ByteRover hits LoCoMo 96.1% BM25-only, zero embeddings (verified in code), via LLM curation at capture time; MemPalace hits LongMemEval 96.6% storing conversation verbatim. Somnigraph's retrieval (3-channel RRF, PPR, LightGBM reranker, feedback) is stronger than both; its write side (flat content strings, connections deferred to sleep) is the liability. Converges with Phase 15's AMemGym finding (write-path is the binding constraint) and wm1 (confidence was marginal, removed from scoring). The genuine fork: verbatim retention wins these benchmarks, but extraction is what enables decay/graph/feedback — don't mistake recall benchmarks for the whole game; they don't test contradiction or long-horizon staleness, which is where extraction earns its cost.

**Confidence-scoring math: deferred. Contradiction/currency handling: kept.** Both sessions independently reached the three-axis confidence decomposition (evidence / currency / salience) from the r/AIMemory push-vs-pull threads. The repo-sweep's cold-context challenge: the benchmark leaders model no confidence at all and win, so elaborate confidence math (tanh saturation, calibration discount) over-invests in a layer that isn't where the wins come from — accepted for the scoring math. But the saturating LoCoMo/LME benchmarks don't test contradiction (Somnigraph's #1 differentiator, Phase 15: universally catastrophic), so the cheap non-scoring half is kept: the evidence-vs-currency *action* distinction (weak evidence → find evidence; stale → re-verify) and **annotate-not-suppress** (surface a stale-but-true fact with a flag rather than decay it below the retrieval budget).

**Push vs pull + a free contradiction probe.** Move contradiction reconciliation from read-time (the reader's burden, every read) to write-time (the substrate, once); Somnigraph's `remember()`-time contradiction classification already leans push. Ready-made CMA probe (roadmap #10): store two conflicting facts — does the system surface the conflict unprompted and say which is current?

**From the practitioner scan (memory-relevant):** Zep's completeness ≠ accuracy — a reranker tuned on answer-correctness is rewarded for lucky guesses (LoCoMo: 23.8% of correct answers had insufficient retrieved context); add a context-sufficiency metric alongside NDCG/recall. Bi-temporal validity (`valid_at`/`invalid_at`) as an axis distinct from age/usage decay. Letta's sleep-time compute as a writer/reader privilege split. Shankar's RAG-Without-the-Lag (pre-materialize an index/feature grid for instant A/B; visualize near-misses — the rank-k vs rank-(k+1) margin is the reranker-separation signal) and MOAR (bandit-allocate the eval budget over whole pipelines, not uniform Optuna trials).

**Concrete adopts from the repo sweep** (detail in vault Phase 18): separation vectors (keep the primary semantic embedding pure, add a context-only channel), structural-loss-safe `update()` (union-merge when an update drops fields), OOD recall gate (explicit `ood:bool` below a score floor), MemPalace dual-layer index, staleness×centrality probe weighting.

## Off-mission (document-KB branch)

The bulk of the Karpathy discourse is about a *document* wiki — a separate, AI-maintained vault distinct from a conversational memory system. Recurring patterns there: keep an agent vault separate from the human vault; a `hot.md` boot cache (a ~150-word running summary read at session start instead of scanning the vault); strict epistemic discipline (every claim cites an immutable source or is flagged as a gap); and [`qmd`](https://github.com/tobi/qmd) (local BM25 + vector + LLM-rerank over markdown) as the search layer. These belong to the document-KB landscape, which is tracked in the vault (`Memory Research/kb-landscape.md`), not reproduced here — Somnigraph is a conversational-memory system and that is a sibling problem.

## Roadmap implications (candidate — for review)

Proposals, not decisions:

- **Neighborhood-scoped contradiction lint in sleep.** If consolidation ever does a graph-wide contradiction pass, bound it to changed nodes + graph neighbors (per #4) before it becomes a cost problem.
- **Keep coding load-bearing memory hygiene deterministic.** Where a behavior matters (feedback rating, dedup-on-write), prefer a coded gate or tool-surface nudge over a snippet sentence (per #1). The `recall()` feedback-ID footer is the model to extend.
- **Write-path before scoring.** Prioritize capture-time curation quality (the binding constraint per Phase 15 + the benchmark leaders) over reranker/confidence tuning. Evaluate the repo-sweep adopts: separation vectors, structural-loss-safe `update()`, OOD recall gate.
- **Defer the confidence-scoring math; keep the contradiction/currency handling.** Hold the three-axis tanh/calibration math until the write path is fixed and it earns its complexity. Adopt only the cheap diagnostic half now (evidence-vs-currency action split, annotate-not-suppress) — it serves the contradiction differentiator the saturating benchmarks don't measure.

## Sources & flags

- Karpathy LLM-Wiki gist + comment thread: <https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f>
- r/ObsidianMD "What's the deal with the hype around Karpathy's LLM wiki?" and "Karpathy's LLM Wiki setup" (Apr/Jun 2026).
- r/ClaudeAI "I added a clause to Andrej Karpathy's 4 CLAUDE.MD clauses" (Jun 2026).
- **Flagged single-anecdote:** the day-60 staleness figure (#3) is one practitioner's experience, not a measurement.
- **Flagged needs-verification:** a Microsoft paper, cited in-thread as "LLMs Corrupt Your Documents When You Delegate," is claimed to show repeated-edit degradation of documents. If real, it is the empirical backing for source-immutability; confirm the citation before relying on it.
- **Flagged not-Karpathy:** the widely-circulated "Karpathy's 4 CLAUDE.md clauses" are community-derived; Karpathy framed them as mistakes that bite him, not authored rules.
- **r/AIMemory push-vs-pull threads** (Jun 2026): the sources are vendors building their own substrates (Recall, github.com/H-XX-D; hakuya.ai) — substance mined, the `tanh` constants are illustrative, and SENTINEL is the author's own benchmark.
- **Benchmark receipts** (community-repo claims, not independently reproduced here): MemPalace's 100% LongMemEval was teaching-to-the-test (fixed 3 question IDs, retested the same set; retracted after issue #875) — trustworthy figures are 96.6% raw / 98.4% held-out. ByteRover 96.1 (paper) vs 92.2 (v2 blog) is judge-model strength + version. Reinforces: treat any single cross-vendor number as marketing.
</content>
