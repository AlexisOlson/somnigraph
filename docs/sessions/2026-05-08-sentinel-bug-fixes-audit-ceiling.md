# 2026-05-08 — Three sentinel-bug fixes + audit ceiling reached

Branch B-light implemented: synthetic anchors restructured to mirror production (query, context) calls — `query=themes-joined` (FTS-style, ~10 tokens, matching real-GT distribution), `context=summary` (vec-style longer text). The (query, context) asymmetry recorded in a sidecar JSON (`gt_feedback_vec_inputs.json`) that merges into `vector_input_map` at training time, with real `recall_meta` taking precedence on collision.

Touched `build_gt_from_feedback.py`, `tune_gt.py:load_tuning_data`, `train_reranker.py` CLI, and the retrain wrapper.

Result: content-residual 239 → 167 (-30%).

## Sentinel fix cascade

Then per-case SHAP on the residual top-30 implicated `session_recency` as next bias channel. Same bug shape as May 8 fb-NaN fix: `session_recency_map.get(mid, -1)` — real values are queries_ago >= 1, so `-1` reads as "more recent than any real co-retrieval." NaN-fixed in both files. 167 → 108 (-35%).

Then audited every feature's missing encoding: found five more bug-shape sentinels (`fts_rank`, `vec_rank`, `theme_rank` defaulted `-1` against real ranks 0-199 smaller=better; `fts_bm25` defaulted `0` against negative real values; `vec_dist` defaulted `0` reading as "perfect match" — wrong-direction; `fts_bm25_norm` and `vec_dist_norm` had per-candidate `0` defaults). All seven NaN-encoded in one pass. 108 → 37 (-66%).

Cumulative: 239 → 37 (-85%). Codified a missing-encoding policy comment block in both files.

## Audit ceiling

Per-case SHAP on the residual 8 worst showed dominant culprits are `fts_rank` (-0.28) and `fts_bm25_norm` (-0.18) — *not* the self-reinforcement features. Realized: content-residual queries strip summary tokens by construction, but FTS indexes summary+themes, so target's FTS signal on its own residual is structurally weak by design while other memories' FTS is unaffected. The audit isn't a clean OOD test — it's an FTS-handicap-target test.

Two of the worst 8 pathologies (`f29ed752`, `81324e46`) have top-1 with *better* `vec_rank` than target — they're not pathologies at all, top-1 is the better answer.

## Documentation + memories

Documented in architecture.md "What didn't work" (5 new entries: synthetic GT shape match, NaN sentinel for session_recency, NaN sentinel for ranks/scores, the audit ceiling).

New memory: `feedback_gt_must_mirror_usage.md` (principle: synthetic GT must mirror real production query/context distribution). New memory: `feedback_dont_run_long_commands.md` (operational: long-running scripts belong in Alexis's terminal).

## Surprises

Confidently predicted `age_days`/`fb_count` would dominate the post-fix SHAP analysis. They were -0.006 each, basically silent. The remaining audit pathologies aren't bias at all; they're the audit construction systematically suppressing target's FTS while imposters' FTS fires normally.

The audit is now retired as a primary quality signal (still useful for sentinel-class regression) — probe Phase-1 miss-rate (`scripts/probe_recall.py`) is the trustworthy generalization metric.

P2 progress: five more sentinel bugs closed (seven total today + earlier May 8); reranker is structurally cleaner than this morning. FSI stability audit and probe miss-rate validation still pending.
