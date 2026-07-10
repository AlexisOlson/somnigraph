# Plan: The pre/post-sleep reranker 2×2 (H10) — v2, whole-corpus revision

Pre-registered as §4.6 of `docs/proposals/sleep-pipeline-optimization.md` (H10 in the §6
ledger). **v2 (2026-07-10)** rewrites v1 for the whole-corpus pivot: the two stores already
exist (`sleepbench-full/conv0-frozen`, 5,882 memories; and its slept sibling), the conv0
eval judge table already exists (`results-full/summary_coverage.json`, opus, 125/229), and
the whole-corpus findings (`findings-whole-corpus.md`) sharpened every prediction. v1's
per-conversation fork pairs (9 ingests + 9 sleeps + 9 judge runs) are **deleted** —
replaced by query-level hold-out on the two existing stores.

This plan is written to be handed to build agents as self-contained work packages
(WP1–WP5, each with interfaces pinned by recon and acceptance checks).

## Question

H3/H7/H11 and the whole-corpus arc measured sleep under fixed-weight scoring (rrf,
formula). H10 asks: **does sleep's value require a reranker that learned sleep's
structure?** Sleep-created features (edges, summaries, consolidation nodes, member
downweights) are identically zero/absent on a never-slept store, so a reranker trained
pre-sleep *structurally cannot* have learned them. The 2×2 separates substrate value from
scoring-policy value.

Sharpened by 2026-07-10: on the slept store, ~97% of GT evidence is dormant — **no
reranker can retrieve what isn't in the candidate pool**, so raw cell values on the slept
column are recall-capped for every model. The live question narrows to: *can R_post learn
to rank covering summaries (and arbitrate node-vs-member via the priority downweight)
better than a pre-sleep policy can?* Judged credit is therefore not a refinement here; it
is the primary cell metric on the slept column. This is also the **first measurement H12's
member downweight can register anywhere** — `base_priority` is invisible to rrf/formula by
construction, and `priority` is a model feature.

## Design

|                                   | Store: frozen    | Store: slept (c2)   |
|-----------------------------------|------------------|---------------------|
| **R_pre** (fit on frozen store)   | ① baseline       | ② naive deploy      |
| **R_post** (fit on slept store)   | ③ counterfactual | ④ aligned           |

- **Hold-out:** both models train on **convs 1–9 queries (~1,800)**; all cells evaluate on
  **conv0's 196 queries**. Same store, disjoint queries — the leakage surface v1 worried
  about (fitting the instrument to the eval arm) is closed at the query level.
- **Headline:** interaction (④−②)−(③−①) > 0 ⇒ sleep pays off only with a sleep-aware
  reranker.
- **Operational money shot:** cell ② vs ①. If ② < ① (judged), sleeping a store regresses
  retrieval under a stale scoring policy — the direct warning for any deployment that
  sleeps without retraining.
- **Bonus column (near-free, no new training):** score both models on
  `conv0-c1_rem_undormant` — the additive-regime store. Answers "with details kept active,
  does R_post still beat R_pre?" and bounds how much of ④'s value is dormancy-coping vs
  structure-exploiting.
- **Metrics:** NDCG@5k, R@10, MRR + recall family on conv0, paired bootstrap
  (`analyze_h3.paired_bootstrap_ci`, n_boot=10k, seed=1234). Slept/undormant columns get
  raw + judged (existing opus table); frozen column raw==judged trivially.

## Pre-registered predictions (so results can surprise us)

1. ② ≤ ① on judged recall (near-certain given dormancy; magnitude is the finding).
2. ④ > ② (R_post learns summary-reaching: `is_summary`/`priority`/`member_count` splits).
3. Interaction > 0.
4. In R_post's importances, the summary/priority cluster ranks high; in R_pre it cannot
   (all-constant at train time). Caveat: `priority` ~aliases `is_summary` on this corpus
   (REM summaries sit at p7–9, nodes p5–8, downweighted members p4) — read the two as one
   credit pool, not as separate signals.
5. Undormant column: R_post ≥ R_pre, both ≥ their slept-column cells (recall ceiling
   lifted).

## Key decisions (D1–D6; leans marked, Alexis ratifies)

### D1. Stores: frozen = `conv0-frozen`, slept = `conv0-c2` — SETTLED (lean)

c2 is the terminal/plateau state of the overnight run: all 8 consolidation nodes, 396
summaries, the full edge set, and the H12 member downweights (18 members at ×0.8). c1 is
defensible ("one cycle = what the live pipeline does per night") but c2 differs from c1
only by pass-2's 5 nodes + 400 edges and is where the node features have the most support.
H11/whole-corpus showed c1≈c2 under fixed scorers, so this choice is low-stakes; take the
structure-richest store.

**Pre-req before any feature extraction on c2:** the node priority retier (parked
`94aad00` band; nodes in c2 were generated under the old max-member rule):
`UPDATE memories SET base_priority = 6 WHERE source = 'consolidate' AND base_priority < 6;`
(bumps exactly 1 node, verified). `priority` is a model feature — retier first, then
capture, or R_post trains on a priority landscape the merged code will never produce again.

### D2. Feature schema: one 24-feature list, shared by R_pre and R_post — SETTLED

Extraction computes **35 features** (the 31 production ones + 4 new), then column-selects
a **24-feature schema** by name before fitting; the model artifact stores its own feature
list. Both models get the identical schema; on the frozen store the sleep-created features
are all-zero and LightGBM never splits on them — that asymmetry IS the experiment.

**Dropped (11):** `fb_last`, `fb_mean`, `fb_count`, `fb_time_weighted`, `session_recency`,
`burstiness`, `hebbian_pmi` (dead on any benchmark store: no feedback events, no session
log, no co-retrieval history); `category`, `confidence`, `decay_rate` (constant on this
corpus: single-category ingest, default confidence, category-default decay); and
**`age_days` (Alexis 2026-07-10):** sleep-created rows are stamped `created_at`=now while
details are backdated to 2022–2024, so on the slept store `age_days` is a ~1,000-day cliff
that duplicates `is_summary` on an artifact scale — it would absorb splits belonging to
the explicit indicators and contaminate WP6's substitution map, while offering nothing on
the frozen store (LoCoMo temporal questions are about dates in content, not recency vs a
2026 wall clock).

**Kept (20 of 31):** `fts_rank`, `vec_rank`, `theme_rank`, `ppr_score`, `fts_bm25`,
`vec_dist`, `theme_overlap`, **`priority`** (v1 wanted this dropped as near-constant —
REVERSED: it is H12's only scoring channel and now encodes the node-type ladder
4 < 5 < 6 < 7–9), `token_count`, `edge_count`, `theme_count`, `query_coverage`,
`proximity`, `query_idf_var`, `betweenness`, `diversity_score`, `query_length`,
`candidate_pool_size`, `fts_bm25_norm`, `vec_dist_norm`.
(Notes: `theme_rank`/`theme_overlap`/`theme_count` are all-zero on frozen — ingest runs
`enrich=False`, details carry only speaker + session-N themes — but REM summaries have
generated themes: sleep-created features on this corpus, same epistemic role as the edge
features. `token_count` and `theme_count` partially alias `is_summary` (summaries are
longer, more-themed) but keep legitimate within-class variance — they join the
one-credit-pool reading {priority, is_summary, token_count, theme_count, member_count}
that WP6's substitution map quantifies.)

**New (4):** `is_summary` (layer=='summary'), `is_consolidation_node`
(source=='consolidate'), `consolidated_member` (flag present), `member_count`
(len(generated_from), 0 for details). All-zero on frozen by construction.

**WP2 empirical audit nuance (2026-07-10):** the "constant on this corpus" rationale for
`category`/`confidence`/`decay_rate` is frozen-store reasoning — on c2 they acquire small
variance (category var 0.055) because sleep-created rows differ from ingested details.
Drops still hold: that variance aliases the explicit `is_summary`/`is_consolidation_node`
indicators (one credit pool). Also confirmed: `age_days` is wall-clock nondeterministic
(computed from `datetime.now()` per extraction) — the SCHEMA_24 matrix is byte-
deterministic only because it's excluded. The seven feedback/session features are dead on
both stores exactly as predicted (all-NaN or zero-variance).

### D3. Training data: convs 1–9 queries on the TWO existing stores — SETTLED

Replaces v1's fork-pair design entirely (no new ingests, no new sleeps, no per-conv judge
runs). ~1,800 train queries vs V6's 4,316: expect a modest model; internal validity is
what matters (R_pre/R_post share queries, labels, schema, hyperparams — only the store
varies).

### D4. Labels: one coverage-credited table, judged by SONNET for train — Alexis call

- Base: evidence-turn ids from `gt.py build_query_set(conv)` per train conv → label 1.0.
- Coverage credit on the slept store: summaries/nodes the judge rules cover a question's
  evidence → label 1.0. **Train-label judge = sonnet** (recommended: ~2,000 pairs ≈ 180
  batches; L5b precedent — sonnet coverage judging agreed 87.5% with opus GT; label noise
  at that level is acceptable for training, and the *eval* table stays opus). Opus for
  both would be the gold option at ~5–10× the terminal time.
- R_pre uses the same table; summary ids simply never appear among its candidates (they
  don't exist in the frozen store) — harmless.
- No label changes for provenance-covered *members* (no double-crediting).

### D5. Trainer: production `scripts/train_reranker.py` via a thin bench driver — SETTLED

Recon (2026-07-10, pinned for the build agents):
- GT input: `--gt <path>` JSON, shape **`{query_text: {memory_id: grade}}`**
  (`gt_calibrated.json` shape). `load_tuning_data(gt_path)` re-runs vec/FTS/theme searches
  itself against the store selected by `SOMNIGRAPH_DATA_DIR` (embeds query texts —
  OPENAI_API_KEY needed; bench stores are OpenAI 1536d) and builds
  `full_data = {search_data, feedback_raw, hebb_data, gt_queries, token_map}`.
- Feature extraction: `extract_features_for_query(qtext, params, search_data,
  feedback_raw, hebb_data, memory_meta, gt_for_query)` → `(features, labels, memory_ids)`.
  `memory_meta` from `memory.reranker._load_memory_meta(db)`.
- Artifacts: booster `.txt` (`booster_.save_model`) + feature-names `.json` — the V6
  bundle format.
- **Hazard, confirmed by recon: production `rerank()` (src/memory/reranker.py:369) builds
  a HARDCODED 31-feature matrix and does not select columns by the model's feature JSON.**
  A 24-feature model can never be served by the live path as-is. Consequence: the bench
  eval scorer must NOT route through `rerank()`; train and eval share the *trainer's*
  extraction instead (WP2/WP3). Production code is not modified except optionally
  `_load_memory_meta` gaining the 4 new meta fields (additive, ignored by live `rerank()`).
- Hyperparams: V6 parity (pointwise LightGBM, same n_estimators/folds defaults), 5-fold
  CV grouped by query.
- **tune_gt module-cache trap (WP2 recon):** `tune_gt` binds `DATA_DIR`/`MEMORY_DB` at
  import. Switching stores within one process requires
  `ingest_baseline._point_at_fresh_store(store)` THEN `importlib.reload(tune_gt)`.
  `bench_features.py`'s selftest demonstrates the working sequence. Feature extraction
  needs `uv run --extra tuning`; `load_tuning_data` writes benign sidecar caches into the
  store dir (memory.db itself is not mutated by this path — read-only SELECTs).
- Query-text keying hazard: GT is keyed by query TEXT; identical question text in two
  train convs would merge labels. The bridge must detect collisions and drop them
  (reported), keeping the table unambiguous.

### D6. Cell scoring: bench model scorer emitting Stage-A JSONL — SETTLED

A new `score_model_arm.py` (not a retrieve.py scorer): loads booster + feature JSON, runs
the same tune_gt-style search/extraction as training (train/serve parity by
construction), ranks conv0's queries on a given store, and writes rows in the exact
Stage-A format (`query_id`, `query`, `gt_answer`, `category`, `evidence_ids`, `retrieved`,
`metrics`) to `results-full/{label}__model-{name}.jsonl`. Then `analyze_iterated.py` /
`impute_iterated.py` apply **unchanged** for contrasts and judged credit. Candidate
generation is identical across all four (six) cells — store determines candidates, model
only re-scores.

## Label/eval circularity — why this is not training on the answer key

(Raised by Alexis 2026-07-10; the Goodhart shape that bit V5+2's synthetic pathologies.)
Two distinct concerns, two answers:

**Literal leakage — eval key reaching training: NO, by three separations.** (1) Disjoint
questions: train = convs 1–9 (~1,800 q), eval = conv0 (196 q). (2) Disjoint label pairs:
summaries are ~98% conv-pure (whole-corpus finding), so the eval key's (question, summary)
rows are conv0-summary pairs — none appear among the ~2,000 train pairs. (3) The eval
table is frozen (opus, produced 2026-07-10 before any model existed) and the TRAIN judge
is a different model (sonnet) — judge-quirk learning isn't rewarded by the eval judge.
Residual exposure, named in caveats: train/eval share the store, so eval-conv summaries
can appear in training pools as unlabeled negatives — features seen, answer-key membership
not (standard shared-corpus/disjoint-queries IR setup; also how production lives).

**Shared target definition — training toward "covering summary = relevant" and scoring
with it: INTENTIONAL.** That is the task definition, not contamination (V6 was likewise
trained and evaluated on the same relevance notion). The honest residual failure mode is
learning judge artifacts rather than the task; covered by different train/eval judges,
raw metrics always reported beside judged, the judge-free frozen column, and the headline
being contrasts under one fixed key rather than absolute numbers.

**Hardening (added to run sequence): 50-pair cross-judge audit.** Sonnet AND opus judge
the same random 50 of the train pairs; report agreement on THIS corpus (don't import
L5b's 87.5%). Poor agreement = stop-and-rethink before training, not a footnote after.

## Work packages (hand-off units; each self-contained, testable, LLM-free)

**WP1 — label bridge** (`experiments/sleep-bench/build_2x2_labels.py`)
Inputs: locomo10.json, the two stores' dia_maps, `gt.py build_query_set` per conv 1–9,
optional coverage JSON(s). Output: `results-full/gt_2x2.json` in `{qtext: {mid: 1.0}}`
shape + a stats block (queries, labels/query, collision count, coverage-credit count).
Steps: resolve evidence ids per query per conv (conv-scoped dia_map keys work — verified);
fold in judged coverage when given (`--coverage <json>`, additive labels for summary ids);
detect and drop cross-conv query-text collisions (print them). Acceptance: runs against
`conv0-frozen` dia_map for conv 1–9 without error; stats printed; a re-run is byte-identical.

**WP2 — bench feature layer** (`experiments/sleep-bench/bench_features.py` + minimal
trainer touch)
Add the 4 new features. Locus: a bench module that (a) wraps
`memory.reranker._load_memory_meta` and augments each memory's meta with
`is_summary`/`is_consolidation_node`/`consolidated_member`/`member_count` read from the
store (one SQL pass over `memories`: layer, source, flags LIKE, generated_from length);
(b) wraps `train_reranker.extract_features_for_query` to append the 4 columns → 35, and
(c) provides `select_schema(features, all_names, wanted_names)` for the 24-feature
column-select. Production `src/memory/` untouched (live `rerank()` hardcodes 31 — do not
disturb). Acceptance: extract features for 5 queries on frozen and on c2; assert the 4 new
columns are all-zero on frozen and non-trivial on c2 (≥389 summary rows exist); assert
column order matches the declared 35-name list; dead-feature audit prints per-column
variance on both stores (confirms the D2 drop list empirically).

**WP3 — model eval scorer** (`experiments/sleep-bench/score_model_arm.py`)
Args: `--data-dir <store> --model <booster.txt> --features <names.json> --label <cell>
--conv 0 --out-dir results-full`. Uses WP1/WP2 machinery + `metrics.py` to emit Stage-A
rows (see D6). Must set `SOMNIGRAPH_DATA_DIR` + reset the module caches the way
`ingest_baseline._point_at_fresh_store` does (module-level DB/embedding caches are the
known trap — reuse that helper or import it). Acceptance: deterministic across reruns
(same rows); 196 rows; `impute_iterated.py` accepts the file (run it with
`--judged-coverage` on a c2-scored output); a smoke model trained on 20 queries scores
without error on both stores.

**WP4 — train driver** (`experiments/sleep-bench/train_2x2.py`)
For each store in {frozen, c2}: point env at the store, `load_tuning_data(gt_2x2.json)`
(train-conv queries only — the bridge already excludes conv0), extract 35 features via
WP2, column-select the 24 schema, fit pointwise LightGBM with V6-parity hyperparams,
5-fold GroupKFold CV by query, save `results-full/R_pre.txt|R_post.txt` +
`R_*__features.json` + importances dump + CV metrics JSON. Acceptance: both artifacts
train end-to-end; CV NDCG prints; importances show zero gain on sleep features for R_pre
(structural check) and nonzero for R_post; artifacts reload and predict.

**WP5 — analyzer** (`experiments/sleep-bench/analyze_2x2.py`)
Reads the 4 (+2 undormant) cell JSONLs + the frozen/slept raw references (rrf/formula
columns already in `results-full/`), computes: per-cell means (raw + judged via the
impute machinery on summary-bearing stores), the five contrasts (②−①, ④−③, ③−①, ④−②,
interaction) with paired bootstrap CIs, and a feature-importance side-by-side for
R_pre/R_post. Acceptance: runs on stand-in cells (relabeled rrf JSONLs) and produces the
full table with CIs; interaction arithmetic cross-checks against the cell means.

**WP6 — feature-structure study: random subsets → importance map → hypothesis tests
(post-headline mechanism study; design Alexis 2026-07-10)**
No stepwise forward/backward selection anywhere: greedy paths are ordering artifacts on
a set with known aliasing (`priority` ≈ `is_summary`) and correlated clusters (graph
features share one edge set); L5b-era precedent: R@10 vs NDCG selected structurally
different sets. Three stages instead:

- **Stage A — random-subset sweep.** Channel core (12) always-on; train R_post on
  ~100–130 random masks over the 12 toggleable features. Identical GroupKFold splits +
  seeds across masks (all comparisons paired). Record per-mask CV metrics + per-feature
  gain importances. Minutes per retrain, parallelizable — ~1 h of compute.
- **Stage B — structure readout.** (1) Marginal value per feature = paired include-vs-
  exclude metric delta over random coalitions (Monte-Carlo Shapley; no ordering effects
  by construction); fit metric~mask with pairwise terms for interaction candidates.
  (2) Substitution/complement map from **importance instability across masks** (within-
  one-model gain is unreliable under correlation; gain that jumps when a competitor is
  masked = substitutes/one credit pool — the priority/is_summary test; gain that only
  appears when a partner is present = complements). The map IS the hypothesized grouping.
- **Stage C — hypothesis tests.** Confirm Stage B's proposed clusters/interactions with
  targeted runs (small factorial over the DISCOVERED groups, or knockout pairs). Each
  hypothesis stated before its test runs — Stage C is confirmation, not more exploration.

**Hygiene: all three stages on train-side CV only; conv0 is touched only by the four
pre-registered fixed-schema headline cells** (no forking paths on the eval conv).
Extends `train_2x2.py` with `--feature-mask` + a sweep driver; analyzer gets
mask-regression + importance-instability table modes. Acceptance: sweep resumes after
interruption; the all-features mask reproduces WP4's R_post CV numbers exactly.

## Run sequence (after builds land)

1. **Alexis, 10 s:** retier SQL on c2 (D1 pre-req).
2. **Me, in-session (~25 min, LLM-free):** `retrieve.py` convs 1–9 on c2, rrf only — this
   is the judge's query/evidence source (its loader reads Stage-A JSONL) and a free rrf
   reference for the train convs.
3. **Alexis terminal (~2–4 h, sonnet):** train-label coverage judge over the convs-1–9
   pairs on c2 (adapted `judge_summary_coverage.py` invocation; checkpoint-resumable).
3b. **Alexis terminal (~10 min, opus):** cross-judge audit — re-judge a fixed random 50 of
   those pairs with opus; agreement rate gates training (see § circularity).
4. **Me:** WP1 bridge → `gt_2x2.json` (evidence + sonnet coverage).
5. **Alexis terminal (minutes–tens of minutes each):** WP4 training, R_pre then R_post.
6. **Me, in-session:** WP3 scoring × 4 cells (+2 undormant cells), then WP5 analyzer,
   then `findings-h10-reranker-2x2.md`.

## Risks / honest caveats

- **Small training set** (~1,800 q) and modest absolute numbers; the interaction is the
  result, not leaderboard NDCG. No comparison to L5b/Config K (different machinery).
- **Query-level hold-out shares the store between train and eval.** Store-level statistics
  (IDF, graph topology, meta distributions) are seen at train time; query identity is not.
  This is exactly how the production reranker lives (one store, new queries), so it's the
  ecologically valid choice — but name it in the findings.
- **Sonnet train labels** carry ~12% disagreement vs opus (L5b estimate) — label noise in
  training only; eval credit stays opus.
- **The slept column's raw numbers are recall-capped by dormancy for every model** —
  report judged as primary there, raw as the artifact it is.
- **Nondeterminism:** LightGBM seeds fixed; capture/search deterministic given store; the
  stores themselves are single sleep runs (edge sets nondeterministic across runs — H11).
- n=1 eval conversation (conv0), n=1 corpus (LoCoMo), single slept store.
- Scoring runs mutate `recall_count` in arm stores (2026-07-10 rig finding). Harmless for
  H10 (no further sleeps planned on these arms; no kept feature reads recall counts), but
  do not sleep these stores again afterward without `--scrub-access-counts` (queued).

## Out of scope (explicitly)

Dormancy→downweight redesign (next experiment; this 2×2 measures the *current* pipeline),
REM prompt fidelity work, the pass-2 re-consolidation guard, live-store anything
(abstinence until ~07-15).
