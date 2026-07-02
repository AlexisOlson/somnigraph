# 2026-07-01 — Reranker arc documentation + STEWARDSHIP restructure

Docs-only session (no code). Paid the documentation debt on the May 2026 reranker retraining arc, fixed current-state version skew, and restructured the STEWARDSHIP priority list to make write-path quality the lead active priority.

## What shipped

**Task 1 — the V3→V5+3b retraining arc in `docs/experiments.md`.** New subsection under § Reranker methodology (~90 lines) telling the arc as methodology, not a per-trial log:

- The measurement problem the arc solves (probe GT vs live GT; aggregate NDCG / held-out probe-hard / live per-mode as the three carrying metrics).
- V3 pinned-boost sweep (locked 5.0; boost=3.0 landing below 2.0 as sub-step noise, not a monotonicity failure), V4 smoke, V5 deploy.
- V5+1 bundled-orthogonal crafting + the per-memory pin cap replacing the backwards `MAX_PROBE_RATIO`; the composition-not-regression lesson that motivated the non-miss NDCG metric.
- V5+2 GT cleanup (5.9% of feedback rows referenced inactive memories; held-out miss 6.5% → 0.0%) and the adversarial pivot from FTS-handicapped audit pathologies to real-recall mining (supply structurally tight: 0 at rank ≥ 10).
- V5+3 first real-recall adversarial retrain: held-out hard flat but live NDCG +0.0297 / R@10 +6.55pp on unchanged composition — the clean transfer-learning evidence.
- session_recency importance trajectory (188 → 445 → 356) partially answering the leakage-vs-signal question without a LOFO.
- The GT-noise finding: live worst-regressions are often measurement artifacts, not model bugs.

Linked to `architecture.md § What didn't work` for the sentinel-bug material (NaN encoding, always-zero `fts_bm25_norm`, audit ceiling) rather than duplicating it.

**Task 2 — version skew.**
- `README.md`: reranker.py `26 features` → `31 features` (only stale count in the file).
- `architecture.md`: closing line `as of March 2026` → `July 2026`; dropped the `CHANGELOG.md` reference (file does not exist). Added a "Current production model" note reframing the 18-feature narrative as the reranker *as first built* and pointing to the new experiments.md arc + migration-notes for the 18→26→31 genealogy.
- `roadmap.md`: closing line `as of March 2026` → `July 2026`.
- `STEWARDSHIP.md`: the 2026-07-01 changelog entry's `All local on docs/overhaul; not pushed` → `Merged to main (ad91e37) and pushed`.
- Also fixed the LoCoMo priority's stale `26-feature production model in P2` → `31-feature production model in P3` (follows from both the version-skew fix and the renumber).

**Task 3 — STEWARDSHIP restructure (a diff, not a rewrite).**
- New **Priority 2: Write-path quality** — the program the evidence converged on (Phase 18 lever + survey recurring themes 1–3). Inputs: the Tier 1 adopt candidates in `ideas-considered.md`. First planned steps recorded: (1) shadow-mode near-dup logging + secret redaction, (2) prospective indexing + a Write Guard gate with thresholds set from the measured distribution. Given a move-down condition.
- **Priority 2 → 3: reranker iteration**, converted to *(maintenance)*. Move-down condition met by Task 1. Settled its two open directions: V5+5 deferred until months of post-V5+3b live activity accumulate (adversarial supply is structurally tight and only regenerates through real usage); the formal session_recency LOFO deprioritized (no longer load-bearing — the 188→445→356 trajectory shows composition-dependence, not pure leakage). Both recorded with reasoning.
- Renumbered docs (3→4), LoCoMo (4→5), PERMA (5→6).

## Surprises

The architecture.md reranker section was written entirely in the present tense around 18 features — the whole subsection reads as current-state when it is really the origin story. The clean fix wasn't to renumber every "18" (that would have broken the +5.1%-vs-formula historical result, which genuinely was the 18-feature number) but to add one note reframing the section as "the reranker as first built" and pointing forward. That reframing resolves every stale current-count claim in the subsection at once without touching the historical narrative.

## Caveats

- The experiments.md section takes all numbers from the twelve May 8–9 session files; no numbers invented. Spot re-verified the aggregate progression, boost sweep, session_recency trajectory, and the V5+3 live deltas against source before writing.
- Anchor links (`#the-v3v53b-retraining-arc-may-2026`, `#what-didnt-work`, `#tier-1--adopt-candidates-17`, `#write-path-is-the-lever-2026-06-28-source-scan`, etc.) were hand-derived from GitHub's slug rules; not click-tested in a renderer.
- Write-path priority's "first planned steps" are transcribed from the orchestrator's arc, not independently designed this session.

## Files touched

`docs/experiments.md` (+92), `README.md` (1), `docs/architecture.md` (+3/−1), `docs/roadmap.md` (1), `STEWARDSHIP.md` (+26/−8, restructure + changelog fix), `docs/sessions/2026-07-01-reranker-arc-docs.md` (new). All edits content-only, verified via `git diff --numstat`; CRLF/LF discipline preserved per file (no mixed endings).

## Reversibility

Pure documentation. Every change is a text edit revertable with `git revert` or by restoring the prior wording. No code, no data, no schema.

## What's next

- Alexis reviews the diff before merge (branch `docs/reranker-arc`; not pushed).
- With reranker at maintenance and write-path quality now Priority 2, the next working session's natural pick is step 1 of the write-path program: shadow-mode near-dup logging + the regex secret-redaction pass.
