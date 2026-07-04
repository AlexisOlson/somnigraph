# Sleep Pipeline Optimization via Benchmark Corpora

**Status: DRAFT, in progress (2026-07-04). Phase 0 done (gate PASS), Stage A rig built.** This
is the wire-it-up plan for measuring and then optimizing each sleep stage on a controllable
accumulation regime (a benchmark corpus), as the complement to the one-shot ~2026-07-15
abstinence fork on the live store. Predictions are pre-registered; what can't be settled
on a benchmark is flagged as such.

**Correction (2026-07-04, post-Phase-0 — supersedes the original V6-primary instrument bet).**
This draft was written with the **deployed V6 reranker as the primary scorer** (the §0 and §3.4
argument below, preserved for the record). Phase 0 plus a closer look overturned that: V6's
learned weights are calibrated to the *live* store's joint feature distribution, and on a
benchmark store its dominant features (`session_recency`, the `fb_*` feedback family) stay
**dead or out-of-distribution even post-sleep** — no real recall history, and probe feedback is
synthetic and sparse. So V6 is a degraded formula plus learned noise, **not a valid sleep
instrument**. The principled instrument is **rerank-free**, a two-level decomposition (§3.4):
`rrf` (raw `rrf_fuse`, graph-blind) vs `formula` (RRF + PPR, graph-aware via `memory_edges`);
**Δ(formula − rrf)** isolates sleep's edge contribution with no learned-weight transfer. V6 is
demoted to an OOD reference. This correction is already reflected in the shipped rig (the `rrf`
scorer; V6 dropped from the default). See also [`experiments/sleep-bench/findings-phase0.md`](../../experiments/sleep-bench/findings-phase0.md)
for the other Phase-0 corrections (REM inert on the token floor; consolidate schema-recovery).

---

## 0. Why this exists (the one-paragraph frame)

The steady-state counterfactual fork already answered *"does sleep help retrieval on a
saturated, regularly-slept store?"* — **no, measurably not** ([`experiments/sleep-fork/findings-sleep-fork.md`](../../experiments/sleep-fork/findings-sleep-fork.md)).
A full `--deep` pass did **0 merges / 0 dormancy / 0 archival** and touched ~108 edges,
because prior nights already did the work. That leaves one live question: **does sleep help
when consolidation has real work — an accumulation regime with backlog, duplicates, and an
unbuilt graph?** The live store can't answer this until the abstinence window closes (~07-15,
one shot, nondeterministic). A **benchmark corpus is the reproducible accumulation regime**:
fresh ingest starts with no edges, no feedback, no summaries, and a backlog of near-duplicate
turns — exactly the state the live store is *not* in. It re-runs cheaply, never touches the
live DB, and its GT is not the probe's target set (so it measures the probe stage without the
self-fulfillment circularity the live store has). This plan builds that harness and uses it to
(a) measure which stages move retrieval, and (b) find the knobs that make them move more —
de-risking the single-shot abstinence fork before it fires.

**The pivotal instrument choice.** A fresh LoCoMo ingest has dead reranker features by
construction — no feedback, no edges, constant category/priority/confidence
([`benchmarks.md`](../benchmarks.md) Level 1: *"the production reranker is useless on LoCoMo DBs — most features are dead"*).
But **sleep is precisely what populates those columns**: NREM writes `memory_edges`
(→ `edge_count`, `hebbian_pmi`, `betweenness`, `diversity_score`), the probe writes `feedback`
events (→ `fb_mean`, `fb_count`, `fb_last`, `fb_time_weighted`), REM writes summaries. So
scoring the same corpus **pre- vs post-sleep with the deployed V6 reranker** is not a
degenerate measurement — it is the *only* scorer that reads the columns sleep creates, and the
pre/post delta isolates whether sleep's structural work carries retrieval signal. This is the
V6 measurement the sleep-fork couldn't make (it ran formula-fallback both sides). See §3.4.

---

## 1. Big-picture roadmap (the arc, sequenced)

The arc mirrors the reranker arc's discipline: **measure honestly first, then optimize the
lever the measurement exposes.** Six phases; each gates the next.

```
Phase 0  Harness + born-saturation precheck        (1 session)   [BLOCKS everything]
   |     Does a fresh ingest give sleep real work? If no, fix ingest density.
   v
Phase 1  Frozen-vs-slept baseline, rrf + formula    (1 session)   [the headline delta]
   |     Full sleep cycle vs no-sleep on the corpus. Is there ANY effect to decompose?
   v
Phase 2  Per-stage decomposition (add-one-in arms)  (1-2 sessions)[which stage matters]
   |     NREM / REM / probe / consolidate increments, paired bootstrap CIs.
   v
Phase 3  Stage-specific optimization A/Bs           (2-3 sessions)[turn the knobs]
   |     For each stage with signal (or a principled null), A/B its dominant knob.
   v
Phase 4  Feed the abstinence fork                   (folds into ~07-15)
   |     Port the winning arm design + scoring rig to the live abstinence fork.
   v
Phase 5  End-to-end QA confirmation (optional)      (1-2 sessions)[does retrieval -> answers]
         Does any retrieval delta survive into LoCoMo QA accuracy? (reader bottleneck known.)
```

**Dependency facts that force the ordering** (from the code inventory, §2):

- **Consolidation requires NREM first.** `sleep_consolidate.py:gather_clusters()` does BFS over
  `memory_edges` with `weight >= EDGE_WEIGHT_THRESHOLD (0.4)`. A fresh ingest has **zero edges**,
  so consolidate-from-frozen does nothing. Its increment must be measured as
  `Δ_consol = (NREM+REM+consol) − (NREM+REM)`, not `consol − frozen`.
- **REM dormancy is gated on coverage, and its age gate is already satisfied.** REM step-6
  dormancy needs `age_days >= 60` **AND** the memory to be covered by an active summary
  (`derived_from` edge) **AND** `(recall+reflect) < 3`. The age gate is **not** a blocker here:
  `ingest.py:227-231` already force-overwrites each memory's `created_at`/`last_accessed` to the
  parsed 2023 LoCoMo session date, so benchmark memories are "old" at eval time by construction.
  The real gate is **coverage** — dormancy can only fire on details a REM summary already covers,
  so it is downstream of REM, not a standalone arm. (Access gate is trivially satisfied: benchmark
  recalls run `internal=True` and log no access.)
- **`sleep_consolidate.py` is NOT in `sleep.py`'s cycle** — it is a standalone script (its own
  docstring gives the intended manual order: `nrem deep → rem → consolidate → probe`). So the
  "full cycle" has two definitions and we must test both: `sleep.py` (NREM+REM+themes+probe+repair,
  **no consolidate**) and the extended manual pipeline that adds consolidate.

**How this feeds the ~07-15 abstinence fork.** Phase 1-3 hand the fork three things it needs and
currently lacks: (1) a **validated rerank-free scoring rig** for slept stores — the `rrf`/`formula`
decomposition (§3.4), plus the MODEL_PATH `.pkl`→`.txt` fix that would otherwise silently fall back
to formula on the live fork too, and the **H10 warning that the fork must not assume its deployed
reranker transfers across slept/frozen arms**; (2) a **per-stage arm design** already debugged
(add-one-in vs strip, §3.5), so the fork's
single shot is spent on the arms that showed signal, not on re-discovering the decomposition; and
(3) a **coverage-scoring method for sleep-created summaries** (§3.6) — without which the fork
would score every REM summary as GT-irrelevant and systematically under-credit consolidation, the
same bug the LoCoMo L5b work already hit and fixed. The benchmark is the rehearsal; the fork is
the performance.

---

## 2. The sleep pipeline as it actually is (stage map)

Condensed from a code-level read of `scripts/sleep.py`, `sleep_nrem.py`, `sleep_rem.py`,
`sleep_consolidate.py`, `probe_recall.py`. Only what the ablation design needs.

| Stage | Script / entry | Standalone? | Mutates | LLM |
|---|---|---|---|---|
| **NREM edge-building** | `sleep_nrem.py [standard\|deep] [--limit N]` | yes | `memory_edges` (typed edges), dup→`status=deleted`+`superseded_by`, `base_priority` (supports), `confidence`, `shadow_load`, `decay_rate`, entity edges, `last_sleep_processed` | Sonnet (pair classify) |
| **REM summary refresh** | `sleep_rem.py [--limit N]` (10 steps) | yes | new `layer=summary` memories, `derived_from` edges, `taxonomy.json`, summary dormancy, `topic_id` | Opus (summary/parent), Sonnet (taxonomy assign, audit) |
| **REM dormancy/health/gaps** | inside `sleep_rem.py` steps 5-8 | no (part of REM) | `status=dormant` (details covered by summary, `age>=60`, low access) | none (pure compute) |
| **Probe feedback** | `probe_recall.run_probe()` / `probe_recall.py`; `sleep.py --probes N` | yes | `feedback` events (utility), `probe_target` events, `recall_miss` events, edge weights, themes, co-utility edges, confidence, decay | configurable (default Opus): craft + rate |
| **Retrieval-failure repair** | `sleep.py:repair_retrieval_failures()` | no (Phase 4 of `sleep.py`) | memory `themes` (add/remove), FTS | Sonnet |
| **Consolidation (merge/archive/rewrite)** | `sleep_consolidate.py [--limit N] [--dry-run]` | yes, **not in `sleep.py`** | `status=deleted`+`superseded_by` (archive), content append (merge/rewrite/annotate), `derivation` edges, `decay_rate=0` (formation), `last_consolidated` | Opus (judge + merge-content) |

Key thresholds the optimization phase will turn (constants in `src/memory/constants.py` unless noted):

- NREM candidate gate: pair skipped if `vec_distance > 0.5`; fast-path needs summary neighbor at
  `vec_distance < 0.25`; confidence floor `weighted_confidence >= 0.3`; `MAX_EDGES_PER_CYCLE = 5`,
  `MAX_PRIORITY_BOOST_PER_CYCLE = 2`. Dup supersede at cosine `> 0.9` (`DEDUP_THRESHOLD = 0.1`).
- REM: `SUMMARY_MIN_DETAIL_TOKENS = 2000` (cluster must exceed this to get a summary), summary
  content `< 1400` chars, `SUMMARY_AUDIT_THRESHOLD = 100` (Tier-2 LLM audit trigger),
  `SUMMARY_DORMANCY_DAYS = 90`, dormancy `age_days >= 60` + `(recall+reflect) < 3`.
  (These live in `sleep_rem.py`.)
- Probe: `--probes` memories × 3 modes (natural/mild/hard) or bundled "mixed" (1 nat + 2 mild +
  1 hard); `MAX_PINS_PER_MEMORY = 4`; utility scale 0.0/0.3/0.6/0.9; feedback thresholds
  `CO_UTILITY_THRESHOLD = 0.7`, `THEME_REFINE_THRESHOLD = 0.6`, `BRIDGE_THEME_THRESHOLD = 0.8`.
- Consolidate: `EDGE_WEIGHT_THRESHOLD = 0.4`, `MIN_CLUSTER_SIZE = 3`, `MAX_CLUSTER_SIZE = 20`,
  `CROSS_TOPIC_EDGE_MIN = 2`, `MAX_ARCHIVES_PER_CLUSTER = 10`; protected if `decay_rate==0` /
  `source==correction` / `flags∋pinned|keep`. (These live in `sleep_consolidate.py`.)

---

## 3. Harness wiring

Extends `scripts/locomo_bench/` and reuses the sleep-fork's analysis rig. Folds in the ranked
"best parts to steal" from the external-harness survey (cited inline as **[steal N]**).

### 3.1 Corpus choice

**Primary: LoCoMo, per-conversation stores** (what `eval_retrieval.py --dataset locomo` and
`run.py` already build via `setup_isolated_db` + `ingest_conversation`, into
`~/.somnigraph/benchmark/retrieval_{conv}/`). Rationale: the harness, the corrected GT
(locomo-audit, 6.4% ceiling), the L5b coverage-judging method, and the metrics already exist here;
per-conversation stores (~500-800 memories) are close to the live store's scale (~1,165 active),
so V6's feature distribution is not wildly off-domain. Run the ablation on **3 conversations**
first (0, 3, 5 — spanning the multi-hop-hard set from the vocabulary-gap analysis), expand to all
10 only if a signal appears (cheap-ablation-iteration discipline).

**Why not PERMA / a synthetic dup corpus first:** PERMA ingest isn't built yet (a separate
2-3 session lift, [`benchmarks.md`](../benchmarks.md) § PERMA) and its preference-tracking strength
is orthogonal to consolidation. Keep it as a Phase-3 stretch if LoCoMo shows sleep needs a
denser-dup regime than LoCoMo naturally provides.

### 3.2 Ingest + the born-saturation precheck **[BLOCKS all downstream — do first]**

The sleep-fork's single strongest fact was that the live store had *nothing to consolidate*. The
symmetric risk here is a corpus that is *born saturated the other way* — too clean/sparse for NREM
to find near-dup pairs or for consolidate to find mergeable clusters. **Before trusting any pre/post
delta, confirm sleep does non-trivial work on the fresh ingest.**

Precheck protocol (one script, `experiments/sleep-bench/precheck.py`, new):
1. Fresh `ingest_conversation(conv, enrich=False)` into an isolated store. (Note: ingest already
   backdates `created_at` to the 2023 session dates, `ingest.py:227-231` — memories are "old" at
   eval time, so age-gated stages are not artificially blocked.)
2. Run `sleep_nrem.py deep`; count: edges created, duplicates superseded, entity edges.
3. Run `sleep_rem.py`; count: summaries created, clusters, dormancy (should be non-zero *once
   summaries cover details*, since the age gate is already met — a zero here means REM made too
   few summaries, not that the corpus is too young).
4. Run `sleep_consolidate.py --dry-run -v`; count: clusters found, proposed archives/merges.
5. **Acceptance gate:** sleep is doing real work if **(NREM edges > ~100) AND (consolidate proposes
   > 0 merges/archives on the dry run)**.

If the gate fails (corpus too clean), two ingest knobs, in order of preference:
- **Finer ingest granularity** (per-utterance already; consider per-sentence splitting of long
  turns) to raise near-dup density for NREM/consolidate.
- **Controlled duplicate injection** (paraphrase N% of turns via one Haiku pass) — last resort,
  because it makes the corpus less representative; document the injection rate if used.

(Backdating is *not* one of the knobs — it already happens in ingest. This is a pleasant surprise
vs the live store: dormancy/decay behave realistically on LoCoMo for free.)

Record the precheck counts in the findings doc regardless — they are the "is there work to do"
number that the sleep-fork taught us to report up front.

### 3.3 Snapshot-and-fork primitive **[steal 3: AMemGym save_state/load_state]**

Replace the sleep-fork's hand-rolled `cp memory.db` + hardcoded-snapshot-path dance (which nearly
broke on `snapshot_production_db`'s hardcoded live path, `eval_retrieval.py:1480`) with a small
store-agnostic primitive in `experiments/sleep-bench/fork.py`:

```python
# fork_store(src_dir) -> new_dir : shutil.copytree the whole SOMNIGRAPH_DATA_DIR
#   (memory.db + tuning_studies/ + taxonomy.json + *.json personal data)
# reset_store(dir)               : rmtree + re-ingest (fresh baseline)
```

The whole benchmark store dir is small; `copytree` is the AMemGym pattern and makes every arm a
byte-identical fork of the frozen post-ingest state — reproducible by construction. Each stage-arm
gets its own fork; no arm shares mutable state. This is strictly better here than the live-store
`cp`-of-one-file recipe because the benchmark store is self-contained under one `DATA_DIR`.

### 3.4 Scoring instrument — the rerank-free decomposition **[corrected 2026-07-04; see §0 correction]**

*Original framing (preserved): "V6 scoring — the load-bearing fix." The two blockers below were
real and are fixed; the conclusion they served (V6 as primary scorer) is superseded by the corrected
scorer set at the end of this section.*

Two concrete blockers to scoring these runs with V6 (both found in the code, both would otherwise
silently degrade to formula — the exact failure mode that hid the 2026-04-07 outage). **Blocker 1 is
fixed and shipped (`d798810`); both are retained here for the record even though V6 is no longer the
primary instrument:**

1. **Both benchmark eval paths resolve the wrong model file.** `eval_retrieval.py:1194-1202`
   (and the identical bug in `run.py:106-113`) override `MODEL_PATH` to search **only** for
   `reranker_model.pkl`. On this machine `~/.claude/data/tuning_studies/` holds a stale
   `reranker_model.pkl` (Mar 20, old sklearn pickle) *and* the current `reranker_model.txt`
   (V6, Jul 2). The `.pkl` is found first, `lgb.Booster(model_file=".pkl")` raises
   `LightGBMError: Unknown model format`, `_load_model()` catches it broadly, sets
   `_cache["failed"]=True`, and **silently falls back to the formula** — so `--configs reranker`
   currently never loads V6, in either script. This is the same silent-fallback failure mode as the
   2026-04-07 outage. **Fix:** change both overrides to prefer the deployed `.txt`
   (`constants.py:21`, `MODEL_PATH = DATA_DIR/tuning_studies/reranker_model.txt`), or drop the
   override entirely so it inherits `SOMNIGRAPH_DATA_DIR`. Add a hard assert that the reranker
   loaded (fail loud — the anti-recurrence lesson from the restore). Verify via `memory_stats()`
   scorer identity before trusting any V6 arm.
2. **The reranker features must exist post-sleep for the delta to mean anything.** On the *frozen*
   arm most V6 features are null/constant (no edges, no feedback). That is fine and intended — it is
   the pre-sleep state. Just confirm the reranker doesn't NaN-crash on all-null feature rows on the
   frozen arm (the NaN-encoding block from the restore should cover this; verify on the precheck DB).

**Scorer set — CORRECTED (2026-07-04, supersedes the V6-primary ranking below).** The MODEL_PATH
`.pkl`→`.txt` fix shipped (`d798810`), so V6 *can* now load on these stores — but the instrument
correction (§0) demoted it. The rig's actual scorer set:

- **Primary — the rerank-free decomposition (`rrf` + `formula`), both wired.** Neither carries
  learned weights transferred from the live store, so neither has V6's OOD problem, and both are
  fixed across arms.
  - **`rrf`** = `impl_recall(expand=False)`: raw `rrf_fuse` (vec+FTS+theme fusion) only, **no
    Hebbian, no PPR**. Blind to `memory_edges`. Isolates sleep's **content/index** effect — merges,
    rewrites, dormancy, re-embedding — i.e. what changed about the searchable text itself.
  - **`formula`** = the legacy `rrf_fuse → apply_hebbian → expand_via_ppr` path (`expand=True`). On
    benchmark data Hebbian is inert (no feedback history), so this is effectively **RRF + PPR**, and
    PPR traverses `memory_edges` — so it **sees NREM's edges**. Adds sleep's **graph/edge** effect.
  - **The decomposition: Δ(formula − rrf)** on a given arm isolates the graph/edge contribution;
    Δ(rrf_slept − rrf_frozen) isolates the content contribution. Validated 2026-07-04: on the frozen
    arm (0 edges) `rrf` and `formula` are **bit-identical across all metrics** — the clean-baseline
    prediction (nothing for the graph layer to do ⇒ no divergence), confirming the two scorers only
    diverge where sleep actually built structure.
- **Reference only — V6 production reranker (31 features).** Kept behind `--scorers v6` but **dropped
  from the default and explicitly not a valid sleep instrument**: its `session_recency` / `fb_*`
  features are dead-or-OOD on a benchmark store even post-sleep (§0). Use only to answer "what would
  the *deployed* reranker see," never as the measurement of sleep's effect. (This is also exactly the
  `R_pre` naive-deploy cell of the 2×2 in §4.6.)
- **Context only — L5b LoCoMo 15-feature reranker (Config K).** Its graph features
  (`graph_synthetic_score`, `graph_coref_hits`) read the **v6-extraction synthetic graph
  (`build_graph.py`), NOT sleep's `memory_edges`.** So L5b does *not* isolate sleep — use it only as
  the QA-retrieval reference in Phase 5, never as a sleep-ablation scorer. (Registered-but-stubbed in
  the rig; wiring it would couple the rig to the eval-harness machinery it exists to avoid.)

### 3.5 Staged predict / retrieve / judge separation **[steal 1: highest-leverage]**

`run.py` resumes via `done_keys = {(conv_id, question)}` but does **not** separate retrieval from
generation from judge — a re-score or reader swap re-pays retrieval. For the ablation (many arms ×
same questions) this is the biggest cost sink. Restructure the sleep-bench harness into three
checkpointed stages writing JSONL to disk between each **[steal 7: run_id namespacing + per-question
crash-safe checkpoints]**:

```
stage A  retrieve   : per arm, per query -> {query, retrieved:[ids], scores, budget, limit}
                      (this is the only stage that touches the store; V6 + formula both scored here)
stage B  read       : optional (Phase 5) -> {query, answer}  (reader model swappable w/o re-retrieval)
stage C  judge      : optional (Phase 5) -> {query, correct, judge_model}  (re-judgeable, N runs)
```

Stage A is all Phase 1-3 needs (it's retrieval-metric-only, like the sleep-fork). Stages B/C only
run in Phase 5. The eval already emits the ranked `retrieved` list + budget/limit (the sleep-fork's
one committed change to `eval_production`); reuse it verbatim.

### 3.6 GT + coverage scoring for sleep-created memories **[the methodological crux]**

LoCoMo GT is per-(question, evidence-turn). **Sleep creates new memories** — REM summaries,
consolidate rewrites — that are *not* evidence turns, so a naive scorer counts them as irrelevant
and **systematically penalizes REM/consolidate** (any query the summary now answers looks like a
miss because the summary's id isn't in GT). This is exactly the L5b synthetic-node problem
([`benchmarks.md`](../benchmarks.md) Level 5→5b), and it has the same fix:

- Extend the L5b **synthetic-coverage judging** (`judge_synthetic_coverage.py`, Sonnet,
  batch_size=10, 87.5% Opus agreement) to **sleep-created summaries/rewrites**: for each question,
  LLM-judge whether each sleep memory *covers* the question's evidence. A sleep memory at rank k
  counts as recalling its covered evidence turns.
- Report **both** scorings: raw (sleep memories score 0 — the pessimistic bound) and coverage-credited
  (the fair comparison). The gap between them is itself the measurement of "how much evidence sleep
  relocated into summaries." Pre-register that coverage-credited is the headline and raw is the floor.

Also carry the sleep-fork's honest note: NDCG uses each store's own `token_map`, so a detail memory
that dormancy/consolidate archived correctly becomes unretrievable (contributes 0) — that is correct
behavior, not a bug.

### 3.7 Metric set **[steal 2: LongMemEval alignment; steal 4: judge noise]**

- **Our set (paired, sleep-fork parity):** NDCG@5k (5000-token budget — pass `--recall-budget 5000`
  explicitly; the `eval_retrieval.py` CLI default is 10000, so parity with the sleep-fork requires
  overriding it), R@10, MRR (cutoff 20),
  computed as **paired per-query deltas vs the frozen arm, bootstrap 95% CI (10,000 resamples,
  fixed seed)** — reuse `experiments/sleep-fork/analyze_sleep_fork.py` almost verbatim (it already
  takes `--baseline`, `--slept`, `--gt`, `--db-a`, `--db-b`).
- **LongMemEval-aligned IR metrics [steal 2]:** add `recall_any` (loose: any evidence turn in top-k)
  and `recall_all` (strict: all evidence turns) alongside NDCG. Adopt the **`effective_k`
  turn→session-collapse correction** (`evaluate_retrieval_turn2session`: walk k up until *k unique
  sessions* recovered) — this fixes the top-k instrument distortion the expansion ablation hit, and
  matters more post-sleep because summaries collapse many turns into one node.
- **Judge noise [steal 4]:** if Phase 5 runs (QA), report judge accuracy as **mean ± std over N=3
  runs** (MemOS `--num_runs`), the defense that caught the ~25pp Zep inflation. Single-run judge
  labels are not trustworthy at the pp-scale deltas sleep produces.
- **Oracle-retrieval + write/read decomposition [steal 8, 5]:** in Phase 5, an `_oracle` arm (feed
  gold evidence to the reader) isolates reader failures from retrieval failures; AMemGym's
  write-vs-read diagnosis (a wrong read counts as read-failure only if the write-time answer was
  correct) tells us whether a sleep-induced QA change is retrieval or extraction — directly relevant
  since the known bottleneck is the reader.

### 3.8 Files to create vs extend

| Action | Path | What |
|---|---|---|
| **create** | `experiments/sleep-bench/` | new experiment dir (mirror `sleep-fork/`) |
| create | `experiments/sleep-bench/precheck.py` | §3.2 born-saturation gate |
| create | `experiments/sleep-bench/fork.py` | §3.3 copytree fork/reset primitive |
| create | `experiments/sleep-bench/run_arms.py` | drive arms: fork → run stage(s) → eval stage A |
| create | `experiments/sleep-bench/coverage_judge.py` | §3.6 sleep-memory coverage (wrap `judge_synthetic_coverage.py`) |
| **reuse** | `experiments/sleep-fork/analyze_sleep_fork.py` | paired bootstrap deltas (parameterize db/gt paths) |
| **reuse** | `experiments/sleep-fork/strip_probe.py` / `verify_strip.py` | probe strip fallback (§3.5 add-one-in is primary) |
| **extend** | `scripts/locomo_bench/eval_retrieval.py` | fix MODEL_PATH `.pkl`→`.txt` (`:1194-1202`, §3.4); add `recall_any`/`recall_all`/`effective_k` |
| **extend** | `scripts/locomo_bench/run.py` | fix duplicate MODEL_PATH bug (`:106-113`); Phase 5 only: split retrieve/read/judge checkpoints (§3.5) |
| — | `scripts/locomo_bench/ingest.py` | no change needed — `created_at` backdating already present (`:227-231`) |

---

## 4. Per-stage plan (measure, then the lever)

Ranked by expected value. Each stage: **what it does → falsifiable hypothesis → instrument (metric +
arm) → isolation method → improvement lever if the null repeats**. Arms use **add-one-in** on fresh
forks (cheaper and cleaner than strip on a corpus we control; strip kept as fallback for probe and
for interaction terms). Increment notation from §1.

### Stage A — REM summary refresh **[EV: HIGHEST]**

- **What it does:** clusters detail memories by topic, generates `< 1400`-char summaries (Opus),
  wires `derived_from` edges, refreshes stale summaries. On a fresh corpus this is where *new
  searchable text* enters the store.
- **Hypothesis (directional, pre-registered):** REM summaries **raise multi-hop R@10** on the
  benchmark, because the vocabulary-gap analysis ([`benchmarks.md`](../benchmarks.md) § Multi-hop
  vocabulary gap) shows 13/18 hard multi-hop misses are fixable by exactly the "entity/topic-scoped
  summary that rephrases scattered evidence" that REM produces — and this is what EXIA GHOST does to
  reach 89.94% with no reranker. Prediction: coverage-credited multi-hop R@10 up **≥ +3pp**;
  single-hop ~flat.
- **Instrument:** `Δ_REM = A_{nrem+rem} − A_{nrem}`, V6 + formula, coverage-credited (§3.6) and raw.
  Multi-hop R@10 is the primary sub-metric; NDCG@5k secondary. Paired bootstrap CI.
- **Isolation:** run `sleep_nrem.py deep` then `sleep_rem.py` on one fork; compare to an NREM-only
  fork. (REM needs `topic_id`, which its own step-1 `assign_memories` fills — no extra dependency.)
- **Lever if null / to push further:** `SUMMARY_MIN_DETAIL_TOKENS = 2000` likely suppresses summaries
  on small benchmark clusters — **lower it and re-measure summary count → R@10** (A/B: 2000 vs 800 vs
  400). Second lever: **summary granularity** — topic-scoped (current) vs **entity-scoped** summaries
  (the benchmarks.md § "For Type B — Entity-scoped summaries" recommendation), A/B on multi-hop R@10.
  Third: **implication extraction** as a REM sub-mode (store what a turn *implies*, not just says) —
  addresses Type-A misses; this is the single highest-value *new* sleep capability the vocabulary-gap
  analysis identified.

### Stage B — NREM edge-building **[EV: HIGH]**

- **What it does:** classifies memory pairs (Sonnet) into typed edges, supersedes duplicates, boosts
  `base_priority`/`confidence`, computes shadow load, links entities. Builds the graph the rest of the
  pipeline (and V6's graph features) depends on.
- **Hypothesis (reframed 2026-07-04 to the corrected instrument):** NREM edges **raise retrieval
  under `formula` (RRF+PPR, graph-aware) but are ~null under `rrf` (raw fusion, graph-blind)**,
  because PPR traverses the new `memory_edges` while raw fusion cannot see them. This is the cleanest
  test of *"does the graph sleep builds carry retrieval signal"* — and **Δ(formula − rrf) on the
  slept arm is the edge contribution, isolated with no learned weights** (the divergence the
  sleep-fork couldn't see, since it ran formula-only both sides — and which V6 would only muddy with
  OOD noise). Prediction: `formula` Δ positive and CI-clear; `rrf` Δ near-null (edge-blind, moves only
  via NREM's dup-merge changing the index); deep > standard (more pairs classified).
- **Instrument:** `Δ_NREM = A_{nrem} − A_0` for both standard and deep; **`rrf` vs `formula`
  side-by-side is the key contrast**, and Δ(formula−rrf) the headline decomposition. NDCG@5k primary.
  (V6 optional as an OOD-reference line only.)
- **Isolation:** `sleep_nrem.py standard` / `sleep_nrem.py deep` on separate forks from frozen.
- **Lever if null:** the candidate gate is conservative (`vec_distance > 0.5` skip; only top-3
  related; `MAX_EDGES_PER_CYCLE = 5`). **A/B a denser graph:** loosen the distance gate (0.5→0.65),
  raise the edge cap, and measure edge count → V6 NDCG. If a denser graph helps V6, that is a direct
  production lever (and interacts with the Turrigiano homeostatic-scaling proposal #25 — a denser
  graph is exactly when per-node clamping starts to matter).

### Stage C — Probe feedback **[EV: MEDIUM — but the cleanest measurement here]**

- **What it does:** selects underserved/coverage-gap memories, crafts natural/mild/hard queries
  (LLM), runs real recall, LLM-rates results, injects `feedback` (utility) + edge-weight + theme +
  co-utility-edge mutations. On the live store this is entangled with GT via self-fulfillment.
- **Hypothesis:** on the benchmark, **the probe's effect is smaller and possibly null**, because the
  circularity that inflated it on the live store is broken — probe targets (coverage-gap memories)
  are **not** the benchmark GT (evidence turns). If the live fork's borderline +0.0009 NDCG probe
  increment was mostly self-fulfillment, it should **not** reproduce here. This is the pre-registered
  test of the self-reinforcement loop's *real* (non-circular) contribution.
- **Instrument:** `Δ_probe = A_full − A_{full−probe}`. Two ways, run both: **add-one-in**
  (`sleep.py --probes 0` fork vs `--probes N` fork) and **strip** (reuse `strip_probe.py`'s verified
  3-channel removal — feedback events, hebbian window, edge weights). Agreement between the two
  validates the strip; divergence is a finding.
- **Isolation:** `--probes 0` vs `--probes N`; or `strip_probe.py` post-hoc.
- **Lever:** if the probe *does* help non-circularly, its knobs are **probe volume** (`--probes`),
  **mode mix** (natural/mild/hard ratio), and **query realism** (roadmap #24: seed the hard-mode
  crafter with real-recall pathology queries). A/B `--probes {10,30,60}` → NDCG plateau (mirrors the
  "where does the feedback loop plateau" open question, measurable here in one shot).

### Stage D — Consolidation (merge / archive / rewrite) **[EV: MEDIUM — the accumulation-regime core]**

- **What it does:** BFS-clusters over the NREM graph, LLM-judges (Opus) keep/archive/merge/rewrite,
  shrinks the store by removing/merging near-dups. **This is the stage the saturated live store had
  no work for** — the whole reason for an accumulation regime.
- **Hypothesis:** consolidation is **retrieval-neutral-to-slightly-positive**: removing near-dup
  detail memories reduces candidate-pool crowding (helps) but risks archiving a memory that was some
  query's best evidence (hurts). Net near zero on NDCG, with a possible small R@10 gain from reduced
  redundancy. Falsifiable: if archiving is *net harmful*, the merge thresholds are too aggressive.
- **Instrument:** `Δ_consol = A_{nrem+rem+consol} − A_{nrem+rem}`. Track store deltas explicitly
  (archives, merges, rewrites — the counts the sleep-fork reported) so a null is attributable to
  "0 work" vs "work but no retrieval effect." Coverage-credited scoring is essential (rewrites are
  new text).
- **Isolation:** `sleep_consolidate.py` standalone on the (NREM+REM) fork; `--dry-run` first to log
  proposed actions without mutating (the precheck already does this).
- **Lever:** thresholds `EDGE_WEIGHT_THRESHOLD (0.4)`, `MIN_CLUSTER_SIZE (3)`, `MAX_ARCHIVES_PER_CLUSTER
  (10)`. **A/B archival aggressiveness** (conservative vs aggressive merge) → R@10, to find whether
  de-duplication helps or hurts retrieval. The `rewrite`/`annotate` action is also an implication-
  extraction surface (overlaps Stage A's lever).

### Stage E — Dormancy / health / gaps **[EV: LOW-MEDIUM — measurable but coupled to REM]**

- **What it does:** marks covered, old, low-access detail memories dormant (REM step 6, `age>=60` +
  covered-by-summary + low access); summary dormancy (step 5, `age>=90`); health/gaps are read-only
  diagnostics.
- **Hypothesis:** dormancy **does fire** on LoCoMo (unlike the live store) because ingest backdates
  `created_at` to 2023, so the age gate is met and access is zero — but it fires **only on details a
  REM summary already covers**. Dormant-ing those covered details is **retrieval-neutral** (they were
  redundant with the summary) or **slightly positive** (less candidate crowding), never strongly
  positive. Falsifiable: if dormant-ing covered details *hurts* R@10, the summary didn't actually
  carry the evidence (a REM-quality signal, not a dormancy signal).
- **Instrument:** dormancy is REM steps 5-6 inside `sleep_rem.py:main()` (no isolating flag), so
  measure it as a **within-REM contrast**: `A_{rem}` (dormancy on) vs a fork where the covered
  details are un-dormant-ed post-hoc (flip `status` back to active for memories dormant-ed this
  cycle — a small strip, analogous to `strip_probe.py`). The store-change count (details dormant-ed)
  is the "is there work" number; R@10 is the effect.
- **Lever:** age gates (60/90) and access thresholds. A/B only if dormancy moves memory counts
  materially. Honest call: **low priority — fold into the full-REM increment (H2) and report the
  dormancy store-change as a diagnostic; give it its own strip arm only if H2 shows REM helping and
  we need to attribute how much is summaries-add vs details-remove.**

### 4.6 The pre/post-sleep reranker 2×2 **[EV: HIGH — but gated; a scoring-policy experiment, not a sleep stage]**

The §3.4 decomposition asks "does sleep make content more *retrievable* (rerank-free)?" This asks the
complementary question: **does sleep's value require a reranker that learned sleep's structure?** It
is a factorial over two axes — **store** (frozen vs slept) × **reranker** (fit on unslept vs slept
data) — and it is the principled way to handle "V6 isn't the right instrument for a separate corpus":
instead of transferring live weights, *fit the instrument on the corpus, under hold-out hygiene*.

| | Store: frozen | Store: slept |
|---|---|---|
| **Reranker R_pre** (fit on unslept forks) | ① baseline | ② **naive deploy** (old reranker meets slept store) |
| **Reranker R_post** (fit on slept forks) | ③ counterfactual | ④ aligned |

**What the contrasts buy:**
- **Substrate value (rerank-free-adjacent):** ②−① and ④−③ — does sleep make content more retrievable
  holding the reranker fixed? (The rerank-free `rrf`/`formula` decomposition is the *learning-free*
  version of this row; this is its reranked echo.)
- **Sleep-aware-scoring value:** ③−① and ④−② — does a reranker that learned sleep's features score
  better, holding the store fixed?
- **The interaction (the interesting term):** (④−②) − (③−①). **Positive ⇒ sleep's edges/summaries
  are only exploitable by a reranker that learned them** — substrate and policy are complementary,
  not additive.
- **Cell ② is the operational money shot:** you sleep the store but keep the deployed reranker. If
  **② < ①**, sleep *hurts* retrieval under the reranker you actually ship (the OOD-on-summaries
  problem, the same shape as V6 here). The finding then becomes **"sleep requires reranker retraining
  to pay off"** — a genuinely useful negative-if-true result, and a direct warning for the ~07-15
  live fork (which will *not* retrain its reranker between arms).

**Hygiene that makes it valid** (else it measures leakage, not sleep):
- Both rerankers fit on **held-out conversations** (train on convs ≠ the eval conv; LoCoMo has 10 —
  k-fold or train 1–9 / eval 0). Never fit the instrument to the eval arm.
- R_pre trained on **frozen** held-out forks, R_post on **slept** held-out forks — **same features,
  same hyperparameters, same coverage labels (§3.6); only the training store-state varies** (the one
  controlled variable). Freeze both; apply unchanged to both store columns.
- **Feature set must read `memory_edges`** (production-style betweenness/diversity from
  `_load_memory_meta`), **not** L5b's synthetic-extraction graph — otherwise R_post is structurally
  blind to sleep's edges and the whole experiment is null by construction. This is a real fork from
  the existing L5b model (`train_reranker.py`'s extractor, re-pointed at sleep's graph).

**Gating (why it is not the next session):**
1. Needs the **coverage judge** (§3.6, unbuilt) — without it R_post can't be credited for valuing
   sleep-created summaries/rewrites, so evidence-turn-only labels blind it to exactly what it's
   supposed to learn.
2. The **summary half** of the interaction needs REM actually producing summaries — currently gated
   by `SUMMARY_MIN_DETAIL_TOKENS = 2000` (Phase-0 Finding 1; the floor sweep, H7). Until that lands,
   only a **"2×2-lite"** is possible: R_post learns NREM edges + consolidation changes but not
   summaries, evidence-turn labels only. Real, but measures the NREM/consolidate half only.

**Sequencing:** Step 1 (H3, NREM/graph, rerank-free) already delivers the substrate row's
learning-free version today. The full 2×2 comes after H7 (REM floor) + the coverage judge — it is the
capstone that says whether sleep + a sleep-aware reranker beat either alone. Tracked as **H10** (§6).

---

## 5. Where the improvement levers are (the optimization surface)

Not just measurement — the "look for improvements" ask. Consolidated across stages, ranked by
expected retrieval payoff, each with its A/B.

1. **REM summary generation is the headline lever.** The vocabulary-gap analysis already did the
   forensics: **implication extraction + entity-scoped summaries address 13/18 hard multi-hop misses**,
   and that is a *sleep-time* operation. Three A/Bs, in order: (a) lower `SUMMARY_MIN_DETAIL_TOKENS`
   so summaries actually generate on benchmark clusters; (b) entity-scoped vs topic-scoped summary
   granularity → multi-hop R@10; (c) add an implication-extraction REM sub-mode → Type-A miss recovery.
   Success metric: multi-hop R@10 toward the 88.8% L5b ceiling *without* the synthetic-extraction graph
   (i.e., sleep summaries doing what v6-extraction synthetics do — a cheaper, production-path win).
2. **NREM graph density.** Loosen the candidate gate / raise edge cap → does a denser graph raise V6
   NDCG? If yes, it's a free production lever (NREM already runs). Interacts with proposal #25
   (Turrigiano clamp) — measure density effect first, then whether the clamp preserves it while
   killing hub runaway.
3. **Probe volume/plateau.** `--probes {10,30,60}` → NDCG plateau — answers "where does the feedback
   loop plateau" (open roadmap question) on a corpus where the answer isn't self-fulfilling.
4. **Consolidation aggressiveness.** Merge/archive thresholds → R@10; find the de-duplication sweet
   spot (crowding reduction vs evidence loss).
5. **Ingest as a lever, not a constant.** Backdating + granularity aren't just precheck fixes — they
   define the accumulation regime's shape. Report which ingest settings make sleep matter; that itself
   is the transferable finding for "when does consolidation earn its cost."

---

## 6. Hypotheses ledger (execution checklist)

| # | Stage | Hypothesis (pre-registered) | Prediction | Arm | Scorer | Primary metric |
|---|---|---|---|---|---|---|
| H0 | precheck | Fresh ingest gives sleep real work | NREM edges >100 AND consolidate proposes >0 merges | run all stages once, count | — | store deltas |
| H1 | full cycle | Full sleep beats no-sleep on the corpus | NDCG@5k Δ > 0, CI-clear (unlike saturated live store) | `A_full − A_0` | V6, formula | NDCG@5k |
| H2 | REM | Summaries raise multi-hop R@10 (vocab-gap logic) | multi-hop R@10 ≥ +3pp, coverage-credited | `A_{nrem+rem} − A_{nrem}` | V6, formula | multi-hop R@10 |
| H3 | NREM | Edges help `formula` (PPR-graph) but not `rrf` (graph-blind) | formula Δ>0 CI-clear; rrf Δ~null; deep>standard | `A_{nrem} − A_0` (std, deep) | **rrf vs formula**; Δ(formula−rrf) | NDCG@5k |
| H4 | probe | Off the live store, probe effect shrinks/nulls (no circularity) | Δ_probe ≤ live's +0.0009, plausibly null | `A_full − A_{full−probe}` (add-in + strip) | V6, formula | NDCG@5k, R@10 |
| H5 | consolidate | Merge/archive is retrieval-neutral-to-slightly-positive | |Δ| small; not net-harmful | `A_{nrem+rem+consol} − A_{nrem+rem}` | V6 + coverage | R@10, NDCG@5k |
| H6 | dormancy | Fires (age gate met via 2023 backdating), retrieval-neutral | >0 details dormant-ed; R@10 neutral | within-REM strip (un-dormant fork) | V6 | store change, R@10 |
| H7 | lever/REM | Lower token floor + entity-scoping raises multi-hop | monotone R@10 gain vs floor; entity ≥ topic | A/B floor {2000,800,400}; scope {topic,entity} | V6 + coverage | multi-hop R@10 |
| H8 | lever/NREM | Denser graph raises V6 NDCG | monotone up to a point | A/B dist-gate {0.5,0.65}, cap {5,15} | V6 | NDCG@5k |
| H9 | lever/probe | Feedback plateaus by ~30-60 probes | plateau visible | A/B `--probes {10,30,60}` | V6 | NDCG@5k |
| H10 | reranker 2×2 | Sleep pays off only with a sleep-aware reranker (interaction >0); naive-deploy cell ② may regress | (④−②)−(③−①) > 0; watch ②<① | store{frozen,slept} × reranker{R_pre,R_post}, held-out fit (§4.6) | R_pre, R_post (coverage labels) | R@10, NDCG@5k |

---

## 7. Risks / honest-accounting (pre-register these caveats)

- **Benchmark-vs-real distribution gap.** LoCoMo is casual multi-speaker dialogue; the live store is
  a single agent's technical/personal memory. Sleep behaviors that help LoCoMo (entity-scoped
  summaries over many speakers) may not transfer, and vice versa. **What transfers is the *mechanism
  ranking* (which stage moves retrieval and why), not the absolute pp numbers.** State this as the
  headline caveat, as the sleep-fork stated "saturated regime only."
- **V6 features are partly dead on any single-corpus store.** `category`, `base_priority`,
  `confidence` are near-constant on LoCoMo even post-sleep, so V6 can only exercise its
  retrieval/text/edge/feedback features. The measurement is real but **understates** what V6 would do
  on a heterogeneous store — a slept-benchmark V6 null does not prove a live-store null. This is the
  main reason the benchmark **de-risks but does not replace** the abstinence fork.
- **Coverage-scoring is itself an LLM judgment.** Crediting sleep summaries via LLM coverage (§3.6)
  imports the L5b caveat: the improvement is *partly definitional*. Mitigate by always reporting the
  raw (uncredited) floor alongside, and by spot-checking coverage labels against Opus (L5b got 87.5%
  agreement — reuse the calibration).
- **Single-fork nondeterminism.** Sleep's LLM steps (NREM classify, REM summarize, probe, consolidate
  judge) are nondeterministic; one fork per arm can't separate a small true effect from run-to-run
  noise. Unlike the live one-shot fork, **the benchmark can afford N=3 forks per arm** — budget for it
  on any arm whose CI touches zero, and report arm-level variance (the sleep-fork couldn't; we can).
- **Probe circularity residue.** Breaking the GT-target correlation reduces but doesn't eliminate
  circularity: the probe still injects feedback that V6's `fb_*` features read, and the rater is an
  LLM judging the same content the scorer ranks. The benchmark makes probe *cleaner* than the live
  store, not *clean*. Report Δ_probe with this explicitly stated.
- **Born-saturation the other way.** If H0 fails and we reach for duplicate injection, the corpus
  drifts from LoCoMo's real distribution — document the injection rate and treat those arms as
  "synthetic accumulation regime," reported separately from the natural-ingest arms.
- **Dormancy fires but is coupled to REM** — it can't be cleanly isolated (steps 5-6 live inside
  `sleep_rem.py:main()`); measure it as a within-REM strip contrast, not a standalone arm (§4 Stage E).
  Low priority; don't spend a session forcing an independent attribution.
- **What we cannot settle on this benchmark:** whether sleep helps the *live* store's heterogeneous,
  feedback-rich composition. That is the abstinence fork's job. The benchmark answers the mechanistic
  question (which stage, which knob, does the graph/summary work carry signal at all) that tells the
  fork where to spend its single shot.

---

*Draft prepared 2026-07-04 for review. Sequenced to complete Phases 0-3 before the ~2026-07-15
abstinence fork so the fork inherits a validated V6 rig, a debugged arm design, and sleep-memory
coverage scoring. Update as phases land; land findings in `experiments/sleep-bench/` and link from
`docs/roadmap.md` § "Does sleep improve retrieval?".*
