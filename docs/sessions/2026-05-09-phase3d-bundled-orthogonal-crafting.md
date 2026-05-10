# 2026-05-09 — Phase 3d — bundled-orthogonal probe crafting for `--mode mixed`

Replaced the 3-separate-LLM-calls-per-target craft path with a single bundled call that produces 4 mode-tagged queries per pinned target (1 natural + 2 mild + 1 hard) attacking the same target from explicitly different angles. Single-mode invocations (`--mode natural`/`mild`/`hard`) keep the legacy per-mode path so ablations and debugging stay straightforward.

## Budget reframing

In mixed mode, each group now consumes exactly 4 from the query budget instead of the previous 1-3 cycle slot — `--queries N --mode mixed` plans `N // 4` groups × 4 queries. Single-mode budget unchanged at 1-3 per group. Budget allocation moved above the dry-run check so `--dry-run` reports the same plan that would actually execute.

## Prompt design

New `STYLE_REFERENCE_BUNDLED` shows one memory rendered as 4 orthogonal queries with explicit `angle:` labels; the two mild queries demonstrate mechanism-vs-symptom orthogonality (paraphrases of the same sentence are explicitly called out as a failure case). New `CRAFT_TEMPLATE_BUNDLED` lists 4 examples of valid orthogonal mild pairs (mechanism-vs-symptom, cause-vs-workaround, who-vs-what, before-vs-after) and ends with a strict 4-element JSON schema in n/m/m/h order. Each query carries an `angle` field that is logged (not persisted to `memory_events`) so the probe operator can spot near-identical mild angles in the log.

## Asymmetric orthogonality framing (the design crux)

Mild-vs-mild is the only orthogonality the prompt leans hard on. Mild is the highest-EV training signal — in-distribution, sample-weight 1.2, distinctive IDs preserved — and also the most likely place for the LLM to lazily produce paraphrases of one another (failure mode: "two queries, same training signal"). Cross-mode orthogonality (mild vs natural, mild vs hard) is mentioned but not belabored: hard is naturally orthogonal by construction (vocabulary stripped), and natural orthogonality matters least because keyword-heavy is exactly the form we want there.

## Validation infrastructure

Parse failures or wrong mode distribution (≠ 1/2/1 nat/mild/hard) trigger one retry, then fall back to the legacy per-mode path for that target — keeps the budget intact even when bundled fails. New stat at end of probe (`Bundled craft: N/M targets succeeded ... produced {natural: A, mild: B, hard: C}`) makes fallback rate visible per run.

Latent bug fixed in passing: empty-queries early return was `return 0, 0` while the caller destructured 4 values — now returns the full 7-tuple so an empty group no longer crashes the run.

## Dry-run validation results

`--mode mixed --queries 4 --coverage 1.0 --dry-run` → "1 groups x 4 queries = 4 queries planned". `--mode mixed --queries 8 --coverage 1.0 --dry-run` → "2 groups x 4 queries = 8 queries planned". `--mode hard --queries 5 --coverage 1.0 --dry-run` → "2 groups, 5 queries planned (per-mode path, mode=hard)". Single-mode regression unaffected.

## Deferred to Alexis's terminal

**(a) Live smoke:** `uv run scripts/probe_recall.py --mode mixed --queries 8 --coverage 1.0`. Expect 2 groups × 4 queries = 8 probe_target events keyed to 2 distinct memories, exact mode counts {natural: 2, mild: 4, hard: 2}, and visibly different `angle:` strings on the two mild queries per target.

**(b) V5+1:** `uv run scripts/probe_recall.py --mode mixed --queries 200 --coverage 0.5`. ~50 targets × 4 = 200 queries. Doubles current held-out probe-hard count (87 → ~137) and contributes ~100 new mild rows, the highest-EV training data Somnigraph generates. After the V5+1 probe lands, re-emit GT, then retrain at the V3-locked pinned-boost.

**(c) Operator inspection of bundled `angle:` quality** after the V5+1 run.

## Reversibility

Change the `if mode == "mixed":` gate in `process_group` to `if False:` to revert to the per-mode path with no other changes. No event-schema, sidecar, or DB changes.

## Files touched

`scripts/probe_recall.py` (~+200 lines net — bundled template, `craft_queries_bundled`, gated process_group, restructured budget + dry-run, new stat block; the per-mode `craft_queries`, `_modes_for_group`, and `CRAFT_TEMPLATES` left intact as the fallback).

## Surprises

Expected the dry-run reordering (moving budget allocation above the dry-run check) to be a one-line cosmetic move, but it forced grappling with the implicit invariant that `groups` was the same length pre- and post-budget. Trimming `groups` early before dry-run actually clarified the operator-facing semantics — now the dry-run plan exactly matches what executes, no mental subtraction required.
