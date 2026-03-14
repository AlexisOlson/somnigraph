# Blindspot Review

You're reviewing **Somnigraph**, a research-grade persistent memory system for LLM agents (specifically Claude Code). It uses SQLite + sqlite-vec + FTS5, hybrid retrieval with Reciprocal Rank Fusion, biological decay, a sleep-based offline consolidation pipeline, and an explicit retrieval feedback loop.

The attached package contains architecture, experiments, negative results, acknowledged unknowns, and the competitive landscape. The system is well-documented and self-aware of many gaps. **We're looking for what we can't see from inside** — the assumptions we're making without realizing it, the structural problems we've normalized, the failure modes that are invisible when you're the one running the system daily.

We built this. We also wrote the critique. The same perspective ceiling applies to both. Your job is to find what's outside that ceiling.

---

## What earns its place in your response

- Structural problems in the approach, not missing features
- Alternative explanations for empirical findings that fit the same evidence
- Specific failure scenarios with concrete triggers, not abstract risks
- Research directions we haven't considered
- Things that seem obvious from outside but invisible from inside
- Interactions between components that create emergent problems

## What doesn't

- Feature suggestions (we have a roadmap)
- Generic engineering advice (testing, CI, etc.)
- Technology swaps ("use Neo4j") — infrastructure choices are deliberate and documented
- Restating gaps we've already identified — extend or challenge them instead

---

## Probe questions

Starting points. If you see something we didn't ask about, prioritize that.

1. **Evaluation methodology.** Our tuning studies (19 phases, up to 906 trials) optimized against ~17 testable queries from a public benchmark (LoCoMo). A real-memory ground truth set (1,047 queries) exists but is only ~20% judged. What should we be worried about in the relationship between the evaluation set, the optimization, and the conclusions we've drawn?

2. **Feedback signal quality.** 12,894 feedback events, mean utility 0.244, dominant scoring coefficient (FEEDBACK_COEFF = 5.15). The empirical Bayes Beta prior (fitted from population feedback via method of moments) blends per-memory EWMA utility with a population mean. What are the implications of this distribution shape and these design choices?

3. **Scoring pipeline interactions.** Four post-RRF stages (feedback boost, theme boost, Hebbian co-retrieval, PPR graph expansion) are applied sequentially. Each was tuned with the others present. What emergent behaviors might arise from their interaction that wouldn't be visible when analyzing each in isolation?

4. **The enrichment strategy.** Memories are embedded once at write time using enriched text (content + category + themes + summary). Themes evolve over time via feedback-driven bridge theme addition, but embeddings don't update. FTS5 indexes summary and themes but not content. BM25 weights: summary 5x, themes 3x. What are the implications of these choices as the system matures?

5. **Sleep as an architectural choice.** Consolidation happens offline via LLM judgment (Sonnet for classification, Opus for cluster decisions). Graph edges, summaries, dormancy transitions, and archive decisions all happen during sleep, not during normal operation. What does this offline-only consolidation pattern imply for the system's behavior between sleep cycles and over its lifetime?

6. **Decay and memory lifecycle.** Exponential decay with per-category rates (episodic 30-day half-life through meta 173-day), a floor at 0.35, reheat on access, and dormancy for low-access memories covered by summaries. Confidence compounds through feedback (grows asymptotically above 0.5 utility, decays multiplicatively below 0.1). What dynamics emerge from the interaction of these lifecycle mechanisms?

7. **The graph's role.** Edge weights change asymmetrically (+0.02 useful, -0.01 useless). PPR traverses all edge types (contradiction, revision, derivation, contextual) with feedback-shaped weights. Edges are created primarily during sleep. What does this imply about the graph's actual contribution to retrieval quality vs. what we think it contributes?

8. **What aren't we asking?** The questions above reflect our current understanding. What questions should we be asking that we're not? What aspects of the system or its documentation suggest problems we haven't framed yet?

---

## Requested output

Structure your response as:

### Blindspots
Things we can't see from inside, ranked by potential impact. For each: what the blindspot is, why we might be missing it, and what it would change if we're wrong.

### Alternative explanations
For findings we're confident about: what other explanations fit the same evidence? Where might our causal narratives be post-hoc rationalizations of empirical results?

### Missing failure modes
Specific scenarios (not abstract risks) where the system would fail in ways we haven't considered. Concrete triggers, not hypothetical concerns.

### Research directions
What would you investigate that isn't in our roadmap? What questions are we not asking?
