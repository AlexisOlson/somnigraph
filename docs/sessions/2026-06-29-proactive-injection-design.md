# 2026-06-29 — Proactive recall: design doc

Design-only session. No code. Produced `docs/proposals/proactive-injection.md` (new standalone design) and a pointer entry in `docs/roadmap.md`, capturing what is arguably the system's main missing capability: surfacing recall hints before the agent asks.

## What shipped

- **`docs/proposals/proactive-injection.md`** (new) — full design for proactive recall:
  - **Doorbell, not delivery.** A `UserPromptSubmit` hook runs cheap RRF-only retrieval each turn; when a floor is cleared it injects a one-line hint (count + top score + 2-4 topic handles, no snippets) that lets the agent decide whether to pull via normal `recall()`. The score only rings the doorbell; the agent keeps the relevance and count decisions. The floor does strictly *less* than the removed cliff detector.
  - **Delivery is a new repo surface.** Proactive injection is not an MCP capability (MCP only answers tool calls); it needs a hook companion to the server, RRF-only so it can run synchronously every turn.
  - **Write-back.** Expensive synthesis triggered by a hint is written back as a memory, so the deep understanding amortizes and composes with the autobiographical-narrative idea.
  - **Session cooldown** (anti-repetition): pulled → suppress hard until `PreCompact`; ignored → decaying penalty that fully expires; two granularities (memory + theme). Ephemeral hook state, explicitly *not* a reranker signal (would repeat the shadow-penalty mistake).
  - **Stochastic gating** (anti-starvation): a hard floor is a deterministic policy that starves the unobserved tail (the never-retrieved basin). Recommended form is Thompson sampling over the per-memory Beta feedback model, so exploration is proportional to uncertainty. Online complement to `probe_recall`.
  - Cooldown and stochastic gating are complementary opposites that flatten the exposure distribution — the actual mechanism that contains feedback self-reinforcement.
- **`docs/roadmap.md`** — the open question shrank to a 3-paragraph pointer to the standalone doc (gap statement + brief + effort), kept discoverable in the agenda.

## Surprises

- The design grew substantially through the conversation. It started as a floor-gated hint and accreted the cooldown, then stochastic gating, each in response to the same underlying hazard (self-reinforcement). The two gates turned out to be complementary opposites (suppress over-exposed / lift under-exposed), which is a cleaner story than either alone.
- The repo's *own* hardest-won findings frame this design well: the limit-parameter "let the agent decide" conclusion (the hint preserves it), the cliff-detector negative result (the nearest prior evidence, and a yellow flag the design has to clear), and the attractor-states / self-reinforcement review findings (the hazard the gates exist to contain). The design slots into existing tensions rather than inventing new ones.

## Caveats

- **Proposal, not built.** Marked as such at the top of the doc. The core assumption — that a coarse top-of-ranking binary surface floor carries usable signal even though the cliff detector found the fine-grained in-list cutoff anti-predictable (R² < 0) — is unproven. The doc makes it offline-testable against existing feedback logs *before* any code.
- **Review caught two honest-accounting overclaims (both fixed):** (1) the doc told the reader to "gate on the calibrated score" as if a live calibrated RRF score exists — it does not; the repo's isotonic/PAVA calibration operates on GT judge scores for tuning, and the 0.4–0.8 overscoring is a judge-score finding, not an RRF-score one. Reworded as a method to borrow, not an artifact to reuse. (2) "already maintains a Beta posterior" overstated the live EWMA-under-Beta-prior aggregation; reworded to a posterior reconstructable from logged counts.
- Stochastic gating breaks the clean offline replay (it becomes off-policy evaluation, importance-weighted, higher variance). Sequencing is explicit in the doc: deterministic floor first, stochasticity second.

## Files touched

- `docs/proposals/proactive-injection.md` (new)
- `docs/roadmap.md` (open-question entry replaced with pointer)
- `STEWARDSHIP.md` (changelog entry)
- `docs/sessions/2026-06-29-proactive-injection-design.md` (this file)

## Reversibility

Fully reversible — docs only, no code, no schema, no scoring change.

## What's next

- **The offline floor study (1 session, data exists, no new collection).** Replay logged prompts, sweep the deterministic floor against use/ignore labels from `recall_meta` + feedback events, read the precision/recall curve, pick an F-beta operating point. Accept if a floor beats both always-inject and never-inject; negative result (floor as weak as the cliff cutoff) is itself publishable.
- If positive: the `UserPromptSubmit` hook delivery layer + write-back path (1-2 sessions), then layer cooldown and stochastic gating.

## Retrospective

**1. What surprised you about the work?** Covered above — the design's growth into two complementary gates, and how cleanly it sits against the repo's existing negative results rather than ignoring them.

**2. Is the priority ordering still right?** No change proposed. This adds a new candidate Tier-1 experiment (the offline floor study) — unblocked, data exists, ~1 session — but it doesn't displace the reranker-iteration priority. Flag it as a strong next-session option, not a reorder.

**3. What does this document need to say that it currently doesn't?** Nothing structural surfaced. The standalone-design-doc pattern is a mild precedent (designs previously lived inline in roadmap, e.g. the limit parameter); if more designs get promoted this way, a short note in `docs/README.md` on where design docs live may eventually help. Not yet warranted for one doc.
