# 2026-07-01 — Documentation IA overhaul

A full information-architecture overhaul of the docs, executed off a handoff plan built the prior session. Goal: one canonical home per fact, placement by provenance × disposition, the negative space discoverable in one place, a depth gradient instead of duplication. Built on a fresh `docs/overhaul` branch off clean public main (`5031347`); **not pushed** — left for Alexis to review and push.

## What shipped, by phase

- **Phase 0 — rebuild.** Rebuilt the survey deliverables (`declined.md`, `ideas-considered.md`) LF-clean off main; re-added the negative-space pointer in `research/sources/index.md`.
- **Phase 1 — external cluster.** `similar-systems.md`: added three carsteneu-survey profiles (m_flow, EverOS, Honcho) under a "Systems from the carsteneu survey" bridge, corrected the system count to 184, and **fully absorbed** `practitioner-signal.md` (then deleted it) as a "Practitioner signal" section. Reciprocal `ideas-considered ↔ similar-systems` bridge.
- **Phase 2 — forward cluster.** `roadmap.md`: thinned "What we learned" (7 findings) and "External review findings" to pointers after verifying every finding was already canonized in `architecture.md`/`experiments.md`; preserved three uncanonized reviewer observations (sleep canonizing bias, category-as-wrong-decay-axis, vocabulary co-adaptation). Moved `proactive-injection.md` to `docs/proposals/`. Added a "Candidates from the external survey" pointer.
- **Phase 3 — benchmark cluster.** New `docs/benchmarks.md` consolidates the LoCoMo retrieval run history (Level 0 → 5b), end-to-end QA, and cross-system score tables (from `locomo-benchmark.md`, the spine); the multi-hop vocabulary-gap analysis (from `multihop-failure-analysis.md`); and the PERMA/AMB/measurable-but-unmeasured/proposed-benchmarking sections moved out of `roadmap.md` (now a forward-only pointer).
- **Phase 4 — sweep + close.** Rewrote `docs/README.md` from a 3-item stub into a one-page documentation map by cluster. De-lined `declined.md` Part 2's fragile `file.md:NN` refs to the living narrative docs into section-name anchors. Cut the spent one-shot review artifacts. Deleted the gitignored local `research/clones/`. Updated `CLAUDE.md` repo-structure/key-files accuracy. Repo-wide broken-link sweep.

## The stub reversal (mid-session course correction)

Phases 2–3 left three redirect **stubs** at old paths (`proactive-injection.md`, `locomo-benchmark.md`, `multihop-failure-analysis.md`) so inbound references would keep resolving. Alexis, on waking, asked "Why do we keep stubs?" and then: **"I don't want stubs."**

Reassessed and agreed: the benchmark stubs were weakly justified (their ~25 inbound refs are mostly inline-code *citations* in historical `research/sources/*.md`, which don't navigate) versus the proactive stub's genuine links. But "no stubs" is the cleaner IA regardless. Removed all three and repointed every reference directly to its canonical home via a mechanical path-string swap across 33 files — a pure path swap, no prose rewritten. This reversed part of Phases 2–3 but produced a strictly better result: clean `docs/`, zero dangling refs.

### Surprises

- **The bare-substring swap collided with a session filename.** Repointing `multihop-failure-analysis.md → benchmarks.md` as a bare substring also matched the session file `docs/sessions/2026-03-24-multihop-failure-analysis.md`, mangling a link in `stewardship-history.md` into a nonexistent `2026-03-24-benchmarks.md`. The **broken-link sweep caught it** — exactly the verify-at-the-output-end discipline paying off. Fixed to the correct filename; re-swept clean. (The proactive swap used the `docs/`-prefixed form and so dodged the analogous `2026-06-29-proactive-injection-design.md` collision.)
- **Most of roadmap's "What we learned" was already canonized.** Verify-first (reading `architecture.md`/`experiments.md` before thinning) confirmed the seven findings lived there at full fidelity, so thinning to pointers duplicated nothing — but the check is what made that safe to assert.

### Caveats

- **Session-file references were swapped too.** Historical session/history files that cited the moved docs now point at `benchmarks.md`/`proposals/`. This is mildly revisionist (the doc *was* at the old path then) but keeps every link live; judged the right trade for referential integrity.
- **Not all `declined.md` line refs were de-lined.** Only refs to the four living narrative docs (which move) were converted to section anchors. The `research/sources/*.md` and session-file line refs (`hipporag.md:258`, `a-mber.md:212`, etc.) were left as-is — those files did not move, and converting all would balloon scope. A future pass could finish them.
- **EOL policy honored throughout** (match-per-file): CRLF files stayed CRLF, LF stayed LF, new files (`benchmarks.md`, `README.md`, session file) are LF. Every edit verified content-only via `git diff --numstat` (no whole-file reflow) and byte-level BOM/CR checks.

### Files touched (this session)

- New: `docs/benchmarks.md`, `docs/proposals/proactive-injection.md` (moved), `docs/sessions/2026-07-01-docs-overhaul.md`.
- Deleted: `docs/locomo-benchmark.md`, `docs/multihop-failure-analysis.md`, `docs/proactive-injection.md` (stubs), `docs/practitioner-signal.md` (absorbed), `docs/review-prompt.md`, `docs/review-package.md` (spent). Local-only: `research/clones/`.
- Rewritten/edited: `docs/README.md`, `docs/roadmap.md`, `docs/similar-systems.md`, `docs/ideas-considered.md`, `docs/declined.md`, `docs/stewardship-history.md`, `STEWARDSHIP.md`, `CLAUDE.md`, plus ~25 `research/sources/*.md` and `scripts/locomo_bench/*.md` (path-ref swaps only).

### Reversibility

All work is local commits on `docs/overhaul` (off `5031347`), tree clean, **not pushed**. Nothing is destructive to history: deleted files remain recoverable from git; `research/clones/` was gitignored and disposable. The prior-session content refs `docoverhaul-good` (7bb4a40) and `survey-prerebase-backup` (80af34e) still exist as backstops.

### What's next

- Alexis reviews `docs/overhaul` and pushes it himself.
- His call on deleting the stale refs `research/comparison-survey`, `docoverhaul-good`, `survey-prerebase-backup` once the overhaul is confirmed pushed.
- Optional follow-up: finish de-lining the remaining `research/sources`/session line refs in `declined.md` Part 2.
- Still open from earlier threads: the proactive-injection offline floor study; the LoCoMo expansion-method / sleep / feedback ablations that would let `experiments.md` document the retrieval arc end to end.
