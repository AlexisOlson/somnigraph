# LoCoMo Expansion-Method Ablation — Findings (DRAFT)

**Status:** Draft for orchestrator review. Measurement only — no expansion method was
removed from the code. The claim that lands in `experiments.md` / `benchmarks.md` gets
written after review, not by this session.

**Date:** 2026-07-01
**Branch:** `exp/locomo-expansion-ablation` (worktree off `main`)
**Roadmap item:** experiment #21 (expansion-method ablation)

## Question

Six candidate-expansion methods run in the L5b retrieval pipeline. Three were believed
dead from fire-rate observation (rocchio 0%, multi_query ~2%, entity_focus ~4%), while
session, keyword, and entity_bridge were believed to do the work. This ablation produces
the per-arm recall evidence that confirms or refutes removability of the three suspected
dead methods, and measures each active method's marginal contribution.

## Method

**Canonical L5b eval config** (from `scripts/locomo_bench/HANDOFF.md`, reproduced and
validated below). Only the expansion flags vary across arms; everything else is held fixed.

Base command (per arm, `<flags>` varies):

```
python scripts/locomo_bench/eval_retrieval.py \
  --dataset locomo --configs locomo_reranker \
  --conversations 0 1 2 3 4 5 6 7 8 9 \
  --recall-limit 800 --synthetic-coverage <synthetic_coverage.json> \
  <flags> --output arm_<name>.jsonl
```

- Reranker: 15-feature `locomo_reranker` model (L5b Config K), `locomo_reranker_model.pkl`.
- Synthetic coverage scoring on (`synthetic_coverage.json`, 1,977 questions).
- All 10 conversations. OVERALL excludes adversarial (category 5), matching
  `report_locomo` semantics.

**Six arms:**

| Arm | Flags | Meaning |
|-----|-------|---------|
| a | *(none)* | no expansion (baseline anchor) |
| b | `--expand-all` | all six methods (documented L5b reference arm) |
| c | `--expand-session --expand-keyword --expand-entity-bridge` | active-trio only |
| d | `--expand-keyword --expand-entity-bridge` | trio minus session |
| e | `--expand-session --expand-entity-bridge` | trio minus keyword |
| f | `--expand-session --expand-keyword` | trio minus entity_bridge |

Arms d/e/f are the three leave-one-out arms over the active trio (each is also a *pair*
of active methods).

**Read-only isolation.** The eval's schema-init opens each conversation DB in write mode
(`CREATE ... IF NOT EXISTS`, idempotent but a write lock). To keep the canonical benchmark
DBs untouched, the 10 per-conversation DBs were copied to a scratch dir and `base_dir` was
redirected there via a `SOMNIGRAPH_BENCH_DIR` env override (added to `config.py` on this
branch). Verified: the source DB mtimes are unchanged after the full sweep. The large
embedding pkls were not needed — ingestion is skipped via the reuse path (DBs already
populated), and query embeddings hit the warm cache. Total API spend ≈ negligible
(query-embedding cache hits only).

**Reference-arm validation (HARD STOP gate).** Arm b reproduced the documented L5b numbers
exactly before any conclusions were drawn:

| Metric | Documented L5b | Arm b (this run) |
|--------|----------------|------------------|
| OVERALL R@10 | 95.4% | 95.4% |
| OVERALL MRR | 0.882 | 0.882 |
| OVERALL R@1 | 84.3% | 84.2% |
| OVERALL R@20 | 96.9% | 96.9% |
| multi-hop R@10 | 88.8% | 88.8% |
| multi-hop MRR | 0.758 | 0.758 |

N=1531 OVERALL and N=89 multi-hop match. The config is correct.

## Results

Per-arm, per-category (OVERALL excludes adversarial). N: single-hop 281, temporal 320,
multi-hop 89, open-domain 841, OVERALL 1531.

### OVERALL

| Arm | MRR | R@1 | R@5 | R@10 | R@20 |
|-----|-----|-----|-----|------|------|
| a (no expansion) | 0.710 | 60.3% | 84.5% | 90.5% | 93.7% |
| b (all-six) | 0.882 | 84.2% | 93.1% | 95.4% | 96.9% |
| c (active-trio) | 0.882 | 84.3% | 93.1% | 95.4% | 96.9% |
| d (−session) | 0.882 | 84.2% | 93.1% | 95.4% | 96.9% |
| e (−keyword) | 0.882 | 84.3% | 93.1% | 95.4% | 96.9% |
| f (−entity_bridge) | 0.882 | 84.2% | 93.1% | 95.4% | 96.9% |

### multi-hop (N=89)

| Arm | MRR | R@1 | R@5 | R@10 | R@20 |
|-----|-----|-----|-----|------|------|
| a (no expansion) | 0.532 | 41.6% | 68.5% | 76.4% | 83.1% |
| b (all-six) | 0.758 | 69.7% | 80.9% | 88.8% | 91.0% |
| c (active-trio) | 0.758 | 69.7% | 80.9% | 88.8% | 91.0% |
| d (−session) | 0.758 | 69.7% | 80.9% | 88.8% | 91.0% |
| e (−keyword) | 0.758 | 69.7% | 80.9% | 88.8% | 91.0% |
| f (−entity_bridge) | 0.758 | 69.7% | 80.9% | 88.8% | 91.0% |

(Full per-category numbers for single-hop / temporal / open-domain in
`ablation_results.json`.)

## Headline results

**1. Every leave-one-out and subset arm reproduces all-six exactly.** Active-trio (c) and
each leave-one-out (d/e/f) match all-six (b) to 0.00pp on every OVERALL and multi-hop recall
metric. The only movement anywhere is a single question's R@1 (84.193% → 84.259%) and 2–3
questions shuffling in first-hit rank.

| Arm vs b | OVERALL R@10 Δ | multi-hop R@10 Δ | # questions R@10 differs (of 1,519) |
|----------|----------------|------------------|-------------------------------------|
| c (trio, drop 3 dead) | +0.0pp | +0.0pp | 0 |
| d (−session) | +0.0pp | +0.0pp | 0 |
| e (−keyword) | +0.0pp | +0.0pp | 0 |
| f (−entity_bridge) | +0.0pp | +0.0pp | 0 |

**2. Expansion *collectively* is highly valuable — but the value is not candidate
expansion.** No-expansion (a) → any expansion config: OVERALL R@10 +4.9pp (90.5→95.4), R@1
+24.0pp (60.3→84.3), MRR +0.172; multi-hop R@10 +12.4pp (76.4→88.8). 90 of 1,519 questions
change at R@10. So "expansion" clearly matters — but the mechanism below shows it is the
Phase-2 rerank, not the methods adding candidates.

**3. The mechanism: on benchmark-sized DBs, no expansion method adds a single candidate, and
the reranker cannot see which methods ran.** Two verified facts explain the exact identity of
arms b–f:

- **The whole DB is already in the Phase-1 candidate pool.** Per-conversation DBs hold
  464–861 memories (max conv 6 = 861), all well under the 4000 internal FTS/vec search limit.
  Instrumentation confirms the candidate pool equals the entire DB (e.g. conv 0: `cand=535`),
  so every expansion method returns **0 net-new candidates** — `all_new_ids` is empty on every
  query. Net-new fire rate is **0% for all six methods** (see fire-rate section).
- **The 15 selected reranker features contain no method-identity signal.** The model's
  feature indices are `[0,1,2,3,5,8,9,12,17,19,22,23,30,33,34]`. `exp_method_counts` (idx 29)
  is *not* selected; nor are `entity_fts_rank` (25), `sub_query_hits` (26), or
  `seed_keyword_overlap` (27). The only expansion-touched selected feature is
  `graph_coref_hits` (34), computed as `len(coref_nbrs & expanded_ids)` — and `expanded_ids`
  is the full DB regardless of which methods run, so it too is invariant. **The reranker
  literally cannot distinguish which expansion methods fired**, which is why arms b/c/d/e/f
  are mathematically identical, not merely close.

The +4.9pp a→b gain is therefore the **Phase-2 second-rerank pass** itself: no-expansion
returns Phase-1 predictions directly (`eval_retrieval.py:497`), while any non-empty expansion
flag set triggers Phase 2, which recomputes the selected features over the expanded RRF and
re-predicts (`preds2`, `eval_retrieval.py:855`). The gate is `any(_expansion_flags.values())`
(line 492) — *which* methods are on is irrelevant to the outcome.

## Per-method fire rates

Measured as the fraction of queries where a method contributed ≥1 **net-new** candidate
(one not already in the Phase-1 pool), via `SOMNIGRAPH_FIRE_STATS` instrumentation on the
`--expand-all` run, all 10 conversations:

| Method | net-new fire rate | net-new candidates added |
|--------|-------------------|--------------------------|
| entity_focus | 0% | 0 |
| multi_query | 0% | 0 |
| keyword | 0% | 0 |
| session | 0% | 0 |
| entity_bridge | 0% | 0 |
| rocchio | 0% | 0 |

All six are candidate-addition no-ops on this benchmark, for the structural reason above
(pool ⊇ DB). Full machine-readable counts in `fire_stats_full.json`.

**This does not match the fire rates in `HANDOFF.md`** (session 100%, keyword ~95%,
entity_bridge ~96%, rocchio 0%, multi_query ~2%, entity_focus ~4%). Those numbers cannot have
come from this benchmark config with net-new semantics — they must reflect a different
measurement point: raw method yield *before* the `existing_ids` dedup filter, and/or a
production-scale DB where the candidate pool does *not* already contain the whole database.
Flagging this as an honest-accounting discrepancy for the orchestrator to resolve; I did not
reconcile it (the original measurement code/conditions are not in this tree).

## Interpretation

- **Confirmed removable, with a caveat:** rocchio, multi_query, entity_focus cost 0 recall
  here. But so does dropping any *one* of session/keyword/entity_bridge, and so would dropping
  *all six* — because on this benchmark none of them add candidates. The ablation confirms
  "removing the 3 dead methods is safe on LoCoMo retrieval metrics," but it does **not**
  isolate them from the three "active" methods: all six are equally inert in this config.
- **The benchmark cannot test candidate-expansion value.** Because the DB fits inside the
  candidate pool, this benchmark structurally cannot exercise what expansion is *for* (pulling
  in bridge candidates that Phase-1 missed). Any removability decision that hinges on
  production behavior needs a setting where the pool is smaller than the corpus.
- **Fire rate ≠ contribution — and here, contribution is 0 for all.** The redundancy is not
  "methods overlap in useful candidates"; it is "no method adds candidates at all, and the
  reranker is blind to method identity."

## Honest caveats

- **This is a benchmark-scale artifact, not a production measurement.** The whole result is
  conditioned on DBs (≤861 memories) far smaller than the 4000-candidate Phase-1 pool. On a
  production-scale corpus the conclusions could differ entirely.
- **Exact-match on one fixed config.** L5b reranker (fixed 15-feature model), corrected GT,
  synthetic coverage, recall-limit 800. A retrained reranker that *selected* a method-identity
  feature (e.g. `exp_method_counts`, `entity_fts_rank`) would break the b=c=d=e=f identity.
- **Singletons not run.** Arms cover no-expansion, all-six, trio, and the three pairs
  (leave-one-out). Singleton arms (session-only, etc.) were not run — though given the
  mechanism (0 net-new + no method-identity feature), they would also be expected to match b.
- **Fire-rate discrepancy unreconciled** (see above) — a real open item, not a settled fact.

## Recommended next steps (not done here — measurement-only scope)

1. **Reconcile the fire-rate discrepancy** with `HANDOFF.md` before acting on removability:
   determine whether the original fire rates were raw pre-dedup yield or production-scale.
   The removal decision for the 3 dead methods should rest on *that* evidence, since this
   benchmark shows all six are inert for a structural reason.
2. If a benchmark test of expansion is wanted, **shrink the candidate pool below the DB size**
   (or grow the DB) so methods can actually add net-new candidates — otherwise the benchmark
   only measures the Phase-2 rerank.
3. Any code change (removing methods) remains a separate session.

## Artifacts (in scratch unless noted)

- Per-arm per-question JSONL: `arm_{a,b,c,d_drop_session,e_drop_keyword,f_drop_entitybridge}.jsonl`
- Machine-readable aggregate (committed alongside this file): `ablation_results.json`
- Fire-rate counts: `fire_stats_full.json`
- This file: `findings-expansion-ablation.md`
