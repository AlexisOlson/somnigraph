# Plan: The pre/post-sleep reranker 2×2 (H10)

Pre-registered as §4.6 of `docs/proposals/sleep-pipeline-optimization.md` (H10 in the §6
ledger). This plan turns the pre-registration into a build/run sequence now that both gates
have cleared: the coverage judge exists (`judge_summary_coverage.py`, H7) and REM produces
summaries on LoCoMo (member-count floor `SOMNIGRAPH_SUMMARY_MIN_MEMBERS=5`, branch
`sleep-bench/h7-rem-floor`, inherited by `sleep-bench/iterated-cycle`).

## Question

H3/H7/H11 measured sleep under fixed-weight scoring (`rrf`, `formula`) and found sleep's
edges are a recall↔ranking **tradeoff** and summaries a judged win. H10 asks: **does
sleep's value require a reranker that learned sleep's structure?** Sleep-created features
(`edge_count`, `betweenness`, `diversity_score`, `ppr_score` — all fed by `memory_edges`)
are identically zero on a never-slept store, so a reranker trained pre-sleep *structurally
cannot* have learned them. The 2×2 separates substrate value from scoring-policy value.

## Design

|                                | Store: frozen   | Store: slept                 |
|--------------------------------|-----------------|------------------------------|
| **R_pre** (fit on frozen forks) | ① baseline      | ② naive deploy               |
| **R_post** (fit on slept forks) | ③ counterfactual | ④ aligned                    |

- **Headline:** interaction (④−②)−(③−①) > 0 ⇒ sleep pays off only with a sleep-aware
  reranker (substrate and policy complementary, not additive).
- **Operational money shot:** cell ②. If ② < ①, sleep *hurts* retrieval under the shipped
  reranker — a direct warning for the ~07-15 live fork, which does not retrain between arms.
- **Metrics** (per ledger): R@10, NDCG@5k on conv0's 196 GT queries; paired bootstrap CIs
  per contrast (reuse `analyze_iterated.py` machinery). Judged bracket (raw ≤ judged ≤
  provenance) on the slept-store column — H7 proved raw sign-flips on summary-bearing arms.

## Hygiene (from §4.6 — "else it measures leakage, not sleep")

1. Both rerankers fit only on **held-out conversations** (train convs ≠ conv0). Never fit
   the instrument to the eval arm.
2. R_pre and R_post: **same feature schema, same hyperparameters, same label table** —
   only the training store-state varies. Freeze both, apply unchanged to both store columns.
3. Feature set **must read `memory_edges`** (production `_load_memory_meta` extractor),
   NOT the L5b synthetic-extraction graph.

## Key decisions

### D1. Trainer base: production `scripts/train_reranker.py` (settled by recon)

- Already extracts `memory_edges`-fed features (`train_reranker.py` → `_load_memory_meta`,
  `src/memory/reranker.py:169`); already store-selectable via `SOMNIGRAPH_DATA_DIR`.
- The L5b trainer (`scripts/locomo_bench/train_locomo_reranker.py`) is disqualified: its
  Group-I graph features read `extracted_from:`/`claim_coref:` synthetic edges (null by
  construction for H10), and its store paths are hardcoded to
  `~/.somnigraph/benchmark/retrieval_{N}`.

### D2. Feature schema: benchmark-valid subset of the 31 (proposed — prune, don't keep-all)

Drop features recon confirmed dead on ANY benchmark store, slept or not: `fb_last`,
`fb_mean`, `fb_count`, `fb_time_weighted`, `session_recency`, `burstiness`, `hebbian_pmi`,
plus near-constant-on-LoCoMo metadata (`category`, `priority`, `confidence`). Keep the
content/lexical core + the four sleep-populated graph features (`edge_count`,
`betweenness`, `diversity_score`, `ppr_score`). Rationale: V6's invalidity on benchmark
stores was a Phase-0 correction; leaving dead features in adds noise-splits and muddies
the R_pre/R_post comparison. Both models get the identical pruned schema; on frozen
stores the graph features are all-zero and LightGBM simply never splits on them — that
asymmetry IS the experiment.

Open sub-question: add an explicit summary-layer indicator feature (is-summary / member
count)? Leaning yes — it's the summary analog of the edge features and equally
sleep-created — but it must exist in both schemas (all-zero on frozen).

### D3. Training data: how many conversations to sleep (Alexis call — cost trade-off)

LoCoMo has 10 convs; conv0 is the eval conv and is excluded from training. Each training
conv needs a **pair**: frozen fork (for R_pre) + one-cycle slept fork (for R_post). Sleep
runs are LLM jobs → Alexis's terminal, one cycle per conv.

- **Minimal:** convs 1–3 (~600 train queries). 3 sleep runs + 3 coverage-judge runs.
- **Full pre-registration:** convs 1–9 (~1800 queries). 9 of each. Strongest model,
  matches §4.6's "train 1–9 / eval 0".
- **Recommended middle:** convs 1–5 (~1000 queries, comparable to V3-era production
  training sets), expandable to 1–9 later without redesign (labels/forks accumulate).

Side benefit either way: the slept forks double as conv1+ replication material for
H3/H7/H11's n=1 caveat.

### D4. Which slept state defines "slept" (H11 decomposition landed — resolved)

- **Iteration count: 1 cycle, settled.** H11 refuted compounding (H11b): cycles 2–4 add
  nothing on any metric under any scorer or credit regime; cycle 2 was the *worst* single
  pass (−0.037 NDCG ✱ / −0.045 R@10 ✱, raw) as consolidate archived 24 more memories.
  The graph effect Δ(formula−rrf) is flat as edges triple 351→984 — saturated at pass 1.
- Consolidate is a confirmed drag on raw metrics: by c4 it had archived 22/133 distinct
  GT-evidence ids (16.5%), and raw damage tracks that archival exactly. REM additionally
  dormants details (31 at c1_rem, incl. 2 GT-evidence turns) — so even `c1_rem` is not a
  pure "structure-added" arm.
- Default stands: **full cycle `c1`** for training forks and the primary eval column
  (external validity — it's what the live pipeline does), with **`c1_rem` as a firmly
  justified secondary eval column** (cheap; bounds the consolidate-erosion confound).
  One R_post, trained on full-cycle forks.
- Caveat for interpreting cells ②/④: the H11 raw-vs-provenance bracket on the slept store
  straddles zero (c4: −0.075 ✱ ≤ truth ≤ +0.127 ✱; c1 bracket narrower but same shape),
  pending the summary-coverage judge. H10's slept-column cells inherit the same bracket
  requirement.

### D5. Labels: one table, coverage-credited, shared by both trainers

- Base: evidence-turn membership from `experiments/sleep-bench/gt.py` `build_query_set`
  (per-conv, dia_map-resolved, status-agnostic) → label 1.0.
- Coverage credit: summaries the judge rules cover a question's evidence → label 1.0
  (the L5b synthetic-coverage pattern, lifted to sleep summaries via
  `judge_summary_coverage.py`). Judge runs per slept training fork → Alexis's terminal.
- Format bridge (build item): production trainer wants `{query: {memory_id: label}}`
  (`gt_calibrated.json` shape); sleep-bench GT is evidence-id lists. One converter script
  emits the shared table; R_pre never sees summary rows simply because they don't exist
  in its stores — the table itself is identical.
- Provenance-mapped members of a summary do NOT get label changes (no double-crediting);
  the judged bracket handles eval-side credit separately.

### D6. Cell scoring: extend the Stage A rig with a model scorer

`retrieve.py` currently has `rrf` / `formula` / `v6` scorers. Add `--scorers model
--model-path <artifact>` that loads an arbitrary trained booster + feature list and scores
the same candidate sets. Same 196 queries, same capture format → `analyze_iterated.py`
paired-bootstrap machinery applies unchanged. Candidate generation identical across all
four cells (store determines candidates; model only re-scores) — per §4.6, only (store,
model) varies per cell.

## Build items (me) vs terminal runs (Alexis)

Build (orchestrated, small, testable):
1. GT→trainer label bridge (+ coverage fold-in) script.
2. Feature-schema audit/prune config for `train_reranker.py` on benchmark stores
   (may be a flag or a thin wrapper — decide after reading the trainer's feature-list
   plumbing).
3. `retrieve.py` model scorer.
4. Cross-store train driver: for each store-state, set `SOMNIGRAPH_DATA_DIR` per conv
   fork, pool extracted features across train convs, fit one model per store-state,
   freeze artifacts (`R_pre.txt`, `R_post.txt` + feature JSONs).
5. `analyze_2x2.py`: four-cell table, five contrasts, interaction CI.

Terminal (Alexis, in order):
1. Ingest baselines convs 1–N (`ingest_baseline.py` per conv; embedding cost, minutes each).
2. Fork + sleep one cycle per training conv (`run_iterated.py --max-cycles 1` or the
   underlying stage scripts; pins SUMMARY_MIN_MEMBERS=5).
3. Coverage judge per slept training fork.
4. Train R_pre, R_post (multi-minute → terminal per standing rule).
5. Score 4 cells + judged bracket on slept column (scoring ~2 min/cell·scorer; judge
   already done for conv0 if summary IDs match H7/H11's).

## Risks / honest caveats up front

- **Training-set size**: even 9 convs ≈ 1800 queries is small vs V6's 4316; expect a
  modest model. The comparison is internally valid (R_pre vs R_post share everything),
  but absolute numbers won't rival L5b/Config K, which had feature selection tuned for
  this benchmark. We are NOT trying to beat L5b — the interaction is the result.
- **Confound to watch**: slept training stores differ from frozen not only by structure
  but by consolidate archival (fewer active memories). D4's `c1_rem` secondary column
  bounds this at eval; on the training side it's part of "what sleep does."
- **Reader/QA relevance**: benchmarks.md notes the remaining LoCoMo QA gap is reader
  extraction, not retrieval — any H10 retrieval delta may not move end-to-end QA. H10 is
  a retrieval-layer claim only.
- n=1 eval conversation (conv0) unless we rotate the held-out conv later.

## Sequencing

1. H11 closeout lands (informs D4).
2. Alexis approves D2/D3/D4 defaults (or adjusts).
3. Build items 1–5; smoke on conv0-frozen + existing c1 fork before any new sleeps.
4. Terminal runs 1–5.
5. Findings: `findings-h10-reranker-2x2.md`.
