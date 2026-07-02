# Proactive Recall: Surfacing Hints Before the Agent Asks

**Status: proposal, not built — offline study complete, hook build declined on this evidence.** The offline floor study ran 2026-07-01 ([findings](../../experiments/floor-study/findings-floor-study.md)): split verdict. The core assumption held on discrimination — the RRF-family score carries real binary surface/skip signal (AUC 0.64–0.74; the cliff's R² < 0 null does not carry over, and score magnitude beats a rank-only cutoff by ~0.08 AUC). But no floor beats always-inject on any reasonable F-beta, because the event log's 0.78–0.88 usefulness base rate is a selection artifact: the log only contains turns where the agent *did* call `recall()`, and the proactive population — no-recall turns — is structurally unobservable offline. **Decision: no injection hook on offline evidence.** If pursued, the honest next instrument is online: a log-only shadow hook that computes the floor decision each turn, injects nothing, and records what the agent did anyway. The design below is preserved as-written for that eventuality.

This document describes the design and the offline experiment that was used to test its core assumption. Read alongside `roadmap.md` (where this is listed as an open question) and `architecture.md` (the pull-based system it extends).

Recall today is entirely pull-based. The agent calls `recall()` when it judges prior context would help, and every retrieval-quality investment (RRF tuning, the reranker, PPR, the limit parameter) improves what comes back *given that the agent asks*. The limit-parameter work (`roadmap.md` § "Can cutoff history calibrate the cliff detector?") deliberately moved even the count decision to the agent, on the finding that the cutoff is a content-level judgment scores can't capture.

The blind spot is the inverse: the agent can't decide to recall what it doesn't know exists. How much relevant memory never gets retrieved because the agent never knew to ask? Counterfactual coverage (`roadmap.md` proposed experiment #5) measures unseen-relevant memories for queries the agent *did* make; it says nothing about the queries never made. That is the larger gap, it is currently unmeasured and unaddressed, and it is the system's main missing capability rather than a tuning refinement.

## The proposal: a doorbell, not a delivery

A lightweight layer runs cheap retrieval (RRF candidates only, no reranker, no LLM) against each user prompt. When the signal clears a floor, it injects an ultra-compact hint into context: a count, the top score, and a few topic handles, nothing more. The hint's only job is to let the session agent decide whether to pull. The agent then calls `recall()` normally if a handle looks relevant to what it's doing.

The score gates whether the doorbell rings; the agent still makes both the relevance decision and the count decision. This keeps the hard-won "let the agent decide" principle intact. The floor does strictly *less* than the cliff detector did: it never decides what is relevant, only whether there is enough signal to be worth one line.

## Hint shape: the compactness floor

One line. Count, top score, and 2-4 topic handles (theme tags or entity names, not snippets). Example:

```
[memory] 3 hits ≥0.62 — fastembed config, cognify adapter, RRF gating
```

Snippets are the thing to resist: they turn the doorbell back into a delivery and reintroduce the per-turn token cost the pull model exists to avoid. The minimum viable payload is exactly enough topic discrimination for the agent to judge task-relevance, and no more. Too compact ("signal exists") and the agent can't discriminate, so it either always pulls or never does, which just relocates the gating problem. The topic handles are the floor below which the hint stops being useful.

## Delivery layer: a new surface for the repo

Proactive injection is not an MCP capability. An MCP server answers tool calls; it cannot push into context each turn. Delivery requires a Claude Code `UserPromptSubmit` hook: a thin companion to the MCP server that shares the same DB and retrieval code. (cognee's Claude Code plugin uses exactly this hook for its context injection.) Today everything in the repo is MCP tools; proactive hints need a hook script alongside the server. The hook must be fast enough to run synchronously every turn, which is why the hint path is RRF-only with no reranker or LLM call.

## Write-back

When a hint leads the agent to pull and synthesize something expensive (multi-hop traversal, cross-domain bridging), write the result back as a memory so the next retrieval gets it cheaply. The hint layer then surfaces *that* on future turns. The deep understanding amortizes into the store instead of being recomputed, and it composes with the autobiographical-narrative idea (`roadmap.md` § "Should sleep produce autobiographical narrative summaries?"): both are write-back of synthesized context.

## Session cooldown: the novelty gate, made concrete

A hint must not repeat itself. Suppression is session-scoped and keyed on what the agent did with the hint, not uniform:

- **Pulled** (the agent recalled it): the content is now in context, so re-hinting is pure noise. Suppress hard for the rest of the session, precisely until a `PreCompact` drops it back out of context.
- **Ignored** (surfaced, not pulled): the agent judged it irrelevant *this turn*, but the conversation may turn toward it later. Apply a decaying penalty, not a ban: `penalty = P0 · decay^(turns_since_hinted)`, fully expiring after a few turns so nothing is permanently locked out within a session. A permanent within-session lockout would recreate the exact blind spot the feature exists to remove.

Two granularities, because re-surfacing a *different* memory on a theme the agent just waved off is nearly as annoying as repeating the same one: suppress the specific memory, and apply a softer decaying penalty to its themes/handles so the channel doesn't rotate through near-duplicates of one declined topic.

This state is ephemeral and lives in the hook's per-session store. It must not become a scoring signal in the reranker. That would recreate the shadow-penalty mistake (`roadmap.md` § "Intuitive scoring signals can be counterproductive"): letting a presentation/recency concern leak into the durable, cross-session relevance model. "What was already shown this session" is exactly the kind of context the model should not learn from. The cooldown is the within-session half of the novelty gate the self-reinforcement risk calls for; the cross-session half (suppressing feedback-saturated memories from the hint channel) is separate and durable. Cooldown params are tunable on the offline replay below: a memory hinted-and-ignored, then hinted-and-ignored again a turn later, is a repeated-ignore the cooldown should have prevented, and repeated ignores are countable in the event log.

## Stochastic gating: exploration against the ratchet

A hard floor is a deterministic policy, and self-reinforcement is fundamentally what a deterministic policy does to its own training data: it only ever surfaces the same high-scoring memories, only those get feedback, their scores stay high, and the unobserved tail never climbs out. This is the "never-retrieved basin" of the attractor-states question (`roadmap.md` § "Do memories converge to attractor states?") restated at the hint layer. The fix is the standard bandit one: give the surface decision exploration so every memory keeps nonzero exposure probability. This debiases rather than merely adds noise: feedback collected under a policy with support over more memories is closer to unbiased.

Three forms, in increasing principle:

- **Epsilon-floor:** with probability ε, surface a candidate sampled from the near-threshold band instead of strictly above the floor. Crude: it explores the band but treats a never-observed memory and a confidently-weak one identically, wasting exploration on known-weak content.
- **Temperature sigmoid:** replace the step at the floor with `P(surface) = σ((score − floor)/τ)`. Surfacing probability rises smoothly through the threshold; τ→0 recovers the hard gate. Calibrated and cheap, but still keyed on the point estimate rather than its uncertainty.
- **Thompson sampling over the per-memory Beta model (recommended).** The system already runs an empirical-Bayes Beta feedback model per memory (the live aggregation is EWMA under a Beta prior); a posterior is reconstructable from the per-memory feedback counts already logged. Sample a utility from it and surface if the sample clears the floor. Exploration becomes proportional to *uncertainty*: a memory with few feedback events has a wide posterior and is explored often; a well-established one is near-deterministic. This is the right discriminator, because it separates "low score because genuinely irrelevant" (narrow posterior, stays down) from "low score because never observed" (wide posterior, worth a look) — the exact distinction a flat ε cannot make, and exactly where the ratchet bias lives.

Stochastic gating and the cooldown are orthogonal and complementary: the cooldown pushes down the over-exposed (anti-repetition, top of the distribution, within session); the stochastic gate lifts up the under-exposed (anti-starvation, the corpus tail). Together they flatten the exposure distribution, which is what actually breaks self-reinforcement. The stochastic gate is also the online complement to `probe_recall`, which already does exploration offline during sleep by generating synthetic feedback for underserved memories.

Costs to be honest about: stochastic surfacing is non-reproducible (seed the RNG per session+turn+query so it is deterministic given context but varied across the corpus); it adds an exploration-rate parameter that interacts with both the floor and the cooldown; and its value depends on the prior being right — if utility is bimodal (`roadmap.md` § "External review findings", bimodal utility prior), Thompson over a single Beta may over-explore noise. It also complicates the offline study, as noted under Experiment.

## The floor, and the honest tension with the cliff result

The floor is a score threshold, and the nearest prior evidence is discouraging: the cliff-detector study found list-level score features anti-predictive of the content-level cutoff (R² < 0; rank position dominated, score features added nothing). That result is a yellow flag this design has to clear, not route around. Two reasons it might still hold where the cliff did not:

1. **The decision is coarser.** The cliff predicted *where in a list to cut*; the floor predicts only *is there anything worth a one-line mention*, a binary at the very top of the ranking where scores are most trustworthy.
2. **The cost of error is smaller.** A wrong cutoff returns wrong content; a wrong floor shows or hides one line, and the agent can always recall manually.

The experiment is precisely whether a top-score floor carries a usable binary signal even though the fine-grained in-list cutoff did not.

## The risk that matters: self-reinforcement amplification

External review already flagged feedback self-reinforcement: only surfaced memories get rated, and several downstream signals amplify the same exposure event. Proactive hints *increase* exposure of already-high-scoring memories by surfacing them unprompted, which could tighten the ratchet (high score → hint → pull → feedback → higher score → hint). This is the design's central hazard and it has to be measured, not assumed away.

The two gating designs above are the mitigation, working from opposite ends: the cooldown suppresses the over-exposed within a session, and stochastic (Thompson) gating keeps the under-exposed tail in play across the corpus. Additional levers to test: suppress memories that are already feedback-saturated from the hint channel entirely, and cap hint frequency per session. The `startup_load`-as-cache concern (review finding) is the same shape: preload shapes query formation, and proactive hints would too.

## Experiment

The gate's label is not retrieval quality; it is a binary surface/skip decision, and the feedback event log already holds the supervision. `recall_meta` events (cutoff_rank, now limit) plus the 13,396 feedback events give, per historical turn, whether surfaced memory was used (utility > 0) or ignored. Replay logged prompts offline, compute what each candidate floor would have surfaced, score against the use/ignore labels. Sweep the floor, read the precision/recall curve, pick the operating point with an F-beta that weights misses (real signal not surfaced) against noise (a wasted line plus a possible wasted pull). Weight slightly toward recall, since a miss is recoverable (the agent can still call `recall`) but is the failure the pull model already has. If raw RRF scores prove miscalibrated for this binary decision, calibrate the gate input first — but note this is a calibration the floor would have to fit on the surface/skip labels, not one that already exists. The repo's isotonic/PAVA GT calibration (which found LLM-judged relevance overscores the 0.4–0.8 band) operates on judge scores for tuning, not on live retrieval scores; it is the method to borrow, not a live calibrated score to reuse.

If the gate is stochastic rather than a hard floor, this clean replay becomes off-policy evaluation: the logged feedback was collected under the current pull-only policy, so candidate stochastic policies must be scored by importance-weighted expectation, not direct replay, at higher variance. Sequence accordingly — tune the deterministic floor first, then layer stochasticity in a second pass.

**Effort.** 1 session for the offline floor study (data exists, no new collection). 1-2 sessions for the hook delivery layer and the write-back path, if the study is positive.

**Hypothesis.** A coarse binary floor carries usable signal where the fine-grained cutoff did not, because the top-of-ranking surface decision is easier than the in-list cut. Proactive hints surface a real coverage gap the pull model misses. They also worsen feedback self-reinforcement unless gated on novelty rather than raw score. Net positive on coverage, conditional on the novelty gate to contain the ratchet.

**Accept if:** a floor exists that beats both always-inject and never-inject on the F-beta.
**Negative result if:** the binary surface signal is as weak as the cliff cutoff (R² near zero), which would itself be a publishable finding that score-based gating fails at both granularities.
