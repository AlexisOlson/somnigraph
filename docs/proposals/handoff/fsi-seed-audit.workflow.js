// Handoff: FSI seed-stability audit — Workflow script (fan-out + reduce)
// ---------------------------------------------------------------------------
// Catalog entry: ../autonomous-experiments.md #6.  Roadmap #9 (FSI stability audit).
// Shape: embarrassingly parallel — one training run per random seed, then one
// reduce that computes per-feature gain-importance mean/variance/rank-stability.
//
// HOW TO RUN (Alexis, when chosen): Workflow({ scriptPath:
//   "C:\\Users\\Alexis\\repos\\somnigraph\\docs\\proposals\\handoff\\fsi-seed-audit.workflow.js" })
// It runs in the background; a task-notification fires on completion.
//
// READINESS: this targets the *LoCoMo* reranker (train_locomo_reranker.py has
// --random-seeds and prints a "Feature importance (gain)" table) — ready today.
// The *production* 31-feature train_reranker.py has NO --seed flag; auditing that
// model (which roadmap #9 actually names) needs a one-line seed passthrough first.
// See the PRODUCTION-VARIANT note at the bottom.
//
// SAFETY RAILS (baked into every agent prompt below):
//  - Never mutate the live store (~/.claude/data) or the production reranker model.
//  - Each seed run writes only to a per-seed scratch dir; --save-model is NOT used
//    (we parse the printed importance table instead, so nothing is clobbered).
//  - Work in a throwaway worktree; commit nothing from the agents.
//  - 0 metered API $ (LightGBM training on the warm feature cache only).
//  - Fixed seed grid (stopping criterion) — no open-ended loop.
// ---------------------------------------------------------------------------

export const meta = {
  name: 'fsi-seed-audit',
  description: 'FSI seed-stability audit: train the LoCoMo reranker across N seeds, measure per-feature gain-importance variance (roadmap #9)',
  whenToUse: 'When scoping whether the reranker feature importances are seed-stable or path-dependent (first-mover bias).',
  phases: [
    { title: 'Train', detail: 'one LoCoMo reranker retrain per random seed, parse gain importances' },
    { title: 'Reduce', detail: 'per-feature mean/variance/rank-stability across seeds + correlated-group analysis' },
  ],
}

// Fixed seed grid — the stopping criterion. 20 seeds.
const SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

// The four known correlated feature clusters that first-mover bias predicts will
// show path-dependent importance concentration (from roadmap #9 / STEWARDSHIP).
const CORRELATED_GROUPS = [
  ['fts_rank', 'fts_bm25'],
  ['feedback_mean', 'ucb_bonus', 'fb_time_weighted'],
  ['age_hours', 'hours_since_access', 'session_recency'],
  ['fts_bm25_norm', 'vec_dist_norm'],
]

const IMPORTANCE_SCHEMA = {
  type: 'object',
  required: ['seed', 'importances', 'ok'],
  properties: {
    seed: { type: 'integer' },
    ok: { type: 'boolean', description: 'true if the training run completed and importances were parsed' },
    note: { type: 'string', description: 'any deviation, missing file, or parse fallback' },
    importances: {
      type: 'object',
      description: 'feature name -> gain importance (number), parsed from the trainer output table',
      additionalProperties: { type: 'number' },
    },
  },
}

const REDUCE_SCHEMA = {
  type: 'object',
  required: ['per_feature', 'unstable_features', 'group_findings', 'summary'],
  properties: {
    per_feature: {
      type: 'array',
      items: {
        type: 'object',
        required: ['feature', 'mean_gain', 'std_gain', 'cv', 'mean_rank', 'rank_std'],
        properties: {
          feature: { type: 'string' },
          mean_gain: { type: 'number' },
          std_gain: { type: 'number' },
          cv: { type: 'number', description: 'coefficient of variation std/mean' },
          mean_rank: { type: 'number' },
          rank_std: { type: 'number' },
        },
      },
    },
    unstable_features: {
      type: 'array',
      description: 'features whose importance rank varies a lot across seeds',
      items: { type: 'string' },
    },
    group_findings: {
      type: 'array',
      description: 'per correlated-group: does importance concentrate on one member (first-mover), and does the winner change across seeds?',
      items: {
        type: 'object',
        required: ['group', 'winner_changes_across_seeds', 'note'],
        properties: {
          group: { type: 'array', items: { type: 'string' } },
          winner_changes_across_seeds: { type: 'boolean' },
          note: { type: 'string' },
        },
      },
    },
    summary: { type: 'string' },
  },
}

// ---------------------------------------------------------------------------

log(`FSI seed-stability audit: ${SEEDS.length} seeds over the LoCoMo reranker.`)

const railsPreamble = `
SAFETY RAILS — obey exactly:
- Do NOT touch the live store (~/.claude/data) or the production reranker model file.
- Do NOT pass --save-model (it would clobber a shared model path). We parse the printed importance table instead.
- Run inside the somnigraph repo (~/repos/somnigraph). Set SOMNIGRAPH_EMBEDDING_BACKEND appropriately for the LoCoMo bench (the LoCoMo scripts use the OpenAI backend with the warm 294MB embed cache; do NOT re-embed if the cache is warm).
- 0 metered API $: training only, warm feature cache. If a step tries to spend money, stop and report ok:false.
`.trim()

phase('Train')
const runs = await parallel(
  SEEDS.map((seed) => () =>
    agent(
      `${railsPreamble}

TASK: Train the LoCoMo reranker with a single random seed and return its per-feature gain importances.

Command (from ~/repos/somnigraph):
    uv run scripts/locomo_bench/train_locomo_reranker.py --random-seeds ${seed} --select-metric r@10

Notes:
- --random-seeds takes a list; pass just ${seed} so this run uses exactly that seed.
- The script prints a "Feature importance (gain)" table (feature name -> gain number). Parse it.
- If the script's --random-seeds path does not emit a single-seed importance table, fall back to: run without --random-seeds but set the seed another supported way, or run --train-only and read importances from the trained booster in-process. Whatever you do, record it in "note".
- If the run fails or you cannot parse importances, return ok:false with the error in "note" and importances:{}.

Return ONLY the structured object: {seed:${seed}, ok, note, importances:{feature:gain,...}}.`,
      { label: `train:seed-${seed}`, phase: 'Train', schema: IMPORTANCE_SCHEMA, agentType: 'general-purpose' }
    )
  )
)

const ok = runs.filter(Boolean).filter((r) => r.ok && r.importances && Object.keys(r.importances).length)
log(`Trained ${ok.length}/${SEEDS.length} seeds successfully.`)

if (ok.length < 3) {
  return {
    error: 'Too few successful seed runs to compute variance (need >=3).',
    successful: ok.length,
    raw: runs,
    likely_cause:
      'train_locomo_reranker.py --random-seeds may not emit a single-seed importance table, or the warm embed cache / feature cache was cold. Inspect one run manually before re-launching.',
  }
}

phase('Reduce')
const reduce = await agent(
  `You are the reduce step of an FSI seed-stability audit. You are given ${ok.length} training runs, each with per-feature gain importances from a different random seed. Compute stability statistics.

RUNS (JSON):
${JSON.stringify(ok)}

CORRELATED FEATURE GROUPS to analyze for first-mover bias (importance concentrating on whichever member wins early splits):
${JSON.stringify(CORRELATED_GROUPS)}

For each feature that appears across runs:
- mean_gain, std_gain, cv (=std/mean), mean_rank (rank by gain within each run, 1=most important), rank_std.
Flag features whose rank_std is high (>~2) as unstable.

For each correlated group: is importance concentrated on one member? Does the winning member CHANGE across seeds (the first-mover signature)? Note it.

Also comment on whether the R@10-vs-NDCG feature-set disagreement documented in STEWARDSHIP could be a first-mover artifact vs genuine metric disagreement, given what the variance shows.

Return the structured object per the schema.`,
  { label: 'reduce:fsi', phase: 'Reduce', schema: REDUCE_SCHEMA, effort: 'high' }
)

log('FSI audit complete. Review the reduce.summary and write findings-fsi.md for human review before any doc merge.')

return {
  seeds_requested: SEEDS.length,
  seeds_ok: ok.length,
  correlated_groups: CORRELATED_GROUPS,
  result: reduce,
  raw_runs: runs,
  next_step_for_human:
    'This is an autonomously-produced measurement. Interpreting "feature X is seed-unstable -> discount its importance / seed-average the next retrain" and the R@10-vs-NDCG artifact question are claims for experiments.md and need human review before merge (honest accounting).',
}

// ---------------------------------------------------------------------------
// PRODUCTION-VARIANT note (roadmap #9 actually names the production 31-feature model):
//   train_reranker.py has NO --seed flag. To audit the production model, first add a
//   one-line passthrough of a --seed arg into the LightGBM params (random_state), then
//   swap the Train-phase command to:
//     uv run scripts/train_reranker.py --train-only --seed <S>
//   against a SCRATCH copy of ~/.claude/data (fastembed backend), never the live store.
//   That flag addition is itself a small prep candidate in the catalog.
// ---------------------------------------------------------------------------
