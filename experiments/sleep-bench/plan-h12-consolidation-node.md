# Plan: consolidation-node rewrite of the sleep consolidate stage (H12)

Decision (Alexis, 2026-07-09): update the sleep pipeline so consolidate **synthesizes the
core details to preserve for factual lookup**, with `consolidated_from` edges preserving the
originals for full flavor. Nothing is removed from the index; members are downweighted, not
archived.

## Why (evidence chain, all 2026-07-09 unless noted)

- H3 (07-04): destructive merge's entire cost was deletion; link-and-keep was free.
- H11: consolidate's per-pass archival eroded the active set 419→360 and archived 16.5% of
  GT-evidence ids; raw retrieval damage tracked archival exactly.
- Archive-loss audit (`findings-h11-archive-loss-audit.md`): **zero facts destroyed** — the
  current merge/rewrite pattern appends member turns verbatim (a de-facto union node), so all
  raw damage decomposed into scorer blindness (16/23 pairs) + ranking failures (6/23): the
  focused original leaves the index and the long mixed-content superseder ranks worse on
  specific queries. Keep+downweight targets 100% of the realized harm.
- Full-union vs de-dup (Alexis): with members kept, node-level losslessness is no longer a
  requirement — the *system* is lossless. De-duped synthesis wins: focused embedding, less
  recall-budget waste, actual consolidation; and synthesis can temporally normalize relative
  dates, which verbatim append structurally cannot (one session theme per node).

## Behavior change (scripts/sleep_consolidate.py)

Current: LLM proposes keep/archive/merge_into/rewrite/annotate per cluster member;
archive/merge set `superseded_by`, hard-remove the loser from vec/FTS, and write a
"consolidated: archived as redundant" derivation edge.

New default (legacy behavior behind an opt-out flag, pattern = NREM's `--legacy-merge`):

1. **Synthesis action.** Where the LLM today rules archive/merge_into for redundancy
   clusters, it instead (same call — it already reads the full cluster) returns a
   **consolidation node**: de-duplicated core facts of the cluster, written for factual
   lookup. Generation constraints (from the audit's drop taxonomy):
   - preserve every number, exact count, name, and speaker attribution;
   - preserve incidental objects/specifics, not just the narrative spine;
   - **temporally normalize**: every relative time expression ("yesterday", "next month")
     is rewritten with its anchor resolved from the member's session theme / created_at;
     this removes the cross-session anchor-orphaning risk entirely.
2. **Members stay active and indexed.** No `superseded_by` (that field's consumers —
   vec/FTS cleanup, self-healing backfill, inactive GT filters — treat it as
   dead-for-retrieval; members are NOT dead). `superseded_by` is reserved for true
   write-path supersession.
3. **Typed provenance edges:** consolidation node → each member, `consolidated_from`
   linking context (mirror the summary `generated_from` schema exactly so provenance/credit
   tooling — impute_rescore/impute_iterated — picks it up without changes; verify the
   exact field/format against what those scripts read).
4. **Member downweight:** v1 = fixed conservative priority reduction on members
   (constant, env-overridable, e.g. `SOMNIGRAPH_CONSOLIDATED_MEMBER_WEIGHT`), entering
   scoring where priority/boost already enters (formula boost path + reranker
   `base_priority` feature). Coverage-proportional downweight (judge-verified) is a
   planned v2, after the H12 bench sweep finds the knee — do not build the judge into the
   pipeline yet. NOTE: raw RRF is rank-based and will not see the downweight; that is a
   known instrument property (measure via formula/reranker), not a bug.
5. **Node properties:** propose in implementation review — leaning layer='detail' with a
   distinguishing source/marker (avoid new-layer plumbing through floors and gather());
   category/themes inherited from members; session themes NOT inherited (dates are
   normalized inline instead).
6. **Scope control:** `keep`/`annotate`/`rewrite` semantics unchanged in v1 unless the
   implementer finds rewrite is redundant with the new action — if so, propose, don't
   silently retire. PROTECTED handling unchanged. MAX_ARCHIVES_PER_CLUSTER becomes
   MAX (synthesis) budget per cluster analog.

## Verification plan

- Execute-path verification on a scratch DB with mocked LLM judgments (pattern:
  2026-07-01 write-path instruments session) — node created, edges typed correctly,
  members still active + still in vec/FTS, downweight applied, legacy flag restores old
  behavior.
- Full end-to-end (real LLM) = the H12 bench run on a conv0 fork: one cycle with new
  consolidate, scored vs frozen and vs c1 with the existing rig. Pre-registered
  predictions: raw ≥ frozen everywhere (no carrier ever leaves the index — the H3
  guarantee); the 6 H11 ranking-failure pairs recover; judged/provenance ≥ c1.
  Sleep + scoring runs → Alexis's terminal.

## Gating

- Feature branch (`sleep/consolidation-node`); do NOT merge to main before the ~07-15
  live-fork decision: old pipeline = clean pre-registration, new pipeline = measure what
  will actually deploy. Alexis's call at fork time.
- Downweight v2 (coverage-proportional) and result-set family collapse (union node +
  member co-occurring in top-k) are follow-ups, not v1.
