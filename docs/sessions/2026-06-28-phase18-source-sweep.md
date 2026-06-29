# 2026-06-28 — Phase 18 source sweep (r/AIMemory + carsteneu leaderboard)

Sourced and analyzed seven memory systems surfaced from r/AIMemory threads and the carsteneu/ai-memory-comparison index, all written to `research/sources/` in the canonical repo template. This branch also carries a parallel session's `practitioner-signal.md` work and a `sleep.py` fix.

## What shipped

**Seven source analyses** (commits `99fef76`, `a4f6a36`):
- `truememory.md` — production 6-layer memory library; steals: encoding gate on auto-capture, separation vectors.
- `byterover.md` — LoCoMo 96.1% **BM25-only, zero vectors** (verified in `search-knowledge-service.ts`) via LLM curation at write time; steals: structural-loss-safe update, OOD recall gate.
- `knowledge-worker.md` — local KG builder; steals: de-spined betweenness for NREM link proposals, staleness×centrality for `probe_recall`.
- `ai-memory-comparison.md` (filed under Surveys) — carsteneu's source-backed comparison of 74 systems; the leaderboard that surfaced the targets below.
- `recall-substrate.md` — the push-memory / compute-confidence substrate behind the r/AIMemory threads.
- `agentmemory.md` — closest stack to Somnigraph; LME 96.2% on fresh stores.
- `mirix.md` — 6-type multi-agent memory; LoCoMo 85.38 (gpt-4.1-mini judge).

Catalog rows added to `research/sources/index.md`.

**`sleep.py` fix** (commit `164fcb5`): REM taxonomy step could emit merge/split/drop/keep_distinct items missing required keys; filter malformed items (with a warning count) and make the print loop key-access defensive. Behavior-safe.

**Also on this branch (parallel session):** `docs/practitioner-signal.md` synthesis + a `claude-md-guide.md` instruction-decay failure-mode entry (`2b16de4`, `40c88a9`). That workstream is the sibling session's; its deeper retrospective detail belongs to that session.

## Surprises

- **The repo is the canonical corpus; the vault is stale.** `research/sources/` (119 files) is current and governed; the Obsidian vault's `Memory Research/` (62 files, Phases 1–17) is a divergent older copy. The initial sweep mistakenly ran against the vault and "found" repos as unevaluated that were already done here — MemPalace (analyzed 2026-04-07), virtual-context, Penfield. Lesson: check the governed corpus, not whichever one you're standing in. Vault Phase 18 duplicates were deleted; the vault is back to its pre-session state.
- **Write-path quality is the lever, not retrieval sophistication.** Every LoCoMo/LME leader wins on the write side: ByteRover (BM25-only + LLM curation), MemPalace (verbatim), agentmemory (write-time temporal grounding + immediate entity/edge extraction), MIRIX (write-time evidence consolidation). Somnigraph's retrieval (3-channel RRF, PPR, 31-feature reranker, feedback) is stronger than all of them; its write side (flat content strings, edges deferred to sleep) is the liability. Independent corroboration of the Phase 15 AMemGym finding that write-path is the binding constraint.
- **carsteneu data had errors.** agentmemory is **96.2%** not 95.2%, and uses a 6-signal weighted sum, **not** 3-way RRF. MIRIX's 85.38 is gpt-4.1-mini-judged (~82% Opus-equivalent, ~3pp below Somnigraph's 85.1 — the apparent tie is a judge artifact).

## Caveats

- Cross-vendor LoCoMo/LongMemEval numbers are non-comparable (different judges, configs, eval granularity). Treat as marketing, not ranking.
- Recall's computed-confidence formula: the *architecture* (inspectable, deterministic, usable before feedback accumulates) earns its keep, but the ceiling constants (SUPPORT 0.15, CHALLENGE 0.60, overconfidence calibration) have zero ablation/GT validation. SENTINEL proves the mechanism fires on synthetic streams; it does not show the scores are calibrated.
- Three off-mission analyses (oculusai, agents-remember, ar-agents-remember) were judged out of scope for this curated corpus and deliberately not added.

## Files touched

- Added: `research/sources/{truememory,byterover,knowledge-worker,ai-memory-comparison,recall-substrate,agentmemory,mirix}.md`
- Modified: `research/sources/index.md` (catalog rows), `scripts/sleep.py` (defensive fix)
- Parallel session: `docs/practitioner-signal.md`, `docs/claude-md-guide.md`

## What's next

- **Write-path discipline is the highest-leverage direction** the sweep points to: encoding gate on auto-capture (TrueMemory), write-time admission firewall that attenuates unevidenced high-confidence claims (Recall), write-time temporal grounding (agentmemory/SimpleMem). All cheap, all targeting the documented gap.
- **Hold off on a full three-axis confidence build.** The benchmark leaders model no confidence at all and still top the boards. If pursued, validate any confidence constants against ground truth first.
- Phase 18 next-target queue is empty (Recall, agentmemory, MIRIX done).
- A `resource` memory type for source documents (MIRIX) is a medium-effort idea worth a roadmap note.

## Reversibility

Source analyses and catalog rows are additive (no behavior change). The `sleep.py` change only drops malformed LLM output that would otherwise crash downstream — strictly defensive.
