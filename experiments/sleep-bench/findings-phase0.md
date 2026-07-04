# Findings — Sleep-Bench Phase 0 (born-saturation precheck)

**Status: DRAFT, 2026-07-04.** Produced during an interactive session (Opus + three
delegated build/survey agents). All work on an **isolated** benchmark store
(`~/.somnigraph/benchmark/sleepbench-precheck/conv0`); the live store (`~/.claude/data`,
under the hard no-sleep abstinence hold until ~07-15) was never touched. No retrieval was
scored in Phase 0 — this phase measures only whether **sleep has work to do** on a fresh
LoCoMo ingest. Retrieval measurement is Phase 1 (the Stage A rig is built but not yet run
on a slept fork).

## Question

Phase 0 of [`docs/proposals/sleep-pipeline-optimization.md`](../../docs/proposals/sleep-pipeline-optimization.md) §3.2:
**does a fresh benchmark ingest give sleep real work**, or is it "born saturated the other
way" (too clean/sparse for NREM to find pairs or consolidate to find clusters)? The
steady-state fork proved the *live* store has nothing to consolidate; the benchmark
accumulation regime only means something if sleep actually acts on it.

## TL;DR

**Gate: PASS — LoCoMo is not born-saturated.** NREM built 361 edges (gate: >100) and
consolidate proposed 5 applied actions (4 archive + 1 rewrite; gate: >0). But the run
surfaced four findings more valuable than the gate itself:

1. **REM is completely inert on LoCoMo** — the `SUMMARY_MIN_DETAIL_TOKENS = 2000` floor
   blocks every summary, so the highest-EV stage in the plan can't be studied here without
   lowering it. **Promotes the floor sweep from a Phase-3 optimization to a Phase-1
   prerequisite.**
2. **The token floor is the wrong *shape*** (absolute, not relative) — it silently encodes
   the live store's token distribution and doesn't transfer to a small-turn corpus.
3. **Consolidate rejects 22 of ~27 proposed actions on schema errors** — effective
   throughput ~18%, and (critically) this **blocks convergence**: re-proposed-and-rejected
   clusters can never reach a fixed point.
4. **Direct evidence for the multi-pass convergence hypothesis** — NREM deferred 50 pairs
   to `cap_hit`; one pass provably undershoots steady state.

Plus a fifth, found by the rig: an **active-status GT bug** in the *old* harness that would
have attributed phantom retrieval losses to sleep.

## Setup

- **Store:** fresh ingest of LoCoMo conversation 0 → 419 turns → 419 memories, isolated at
  `~/.somnigraph/benchmark/sleepbench-precheck/conv0`. Pre-sleep state: 419 active details,
  0 edges, 0 summaries, 0 topics.
- **Harness:** `experiments/sleep-bench/precheck.py` (isolation guard → ingest →
  `sleep_nrem.py deep` → `sleep_rem.py` → `sleep_consolidate.py --dry-run -v` → counts →
  gate). OpenAI 1536d backend (home machine), `PYTHONUTF8=1`.
- **Scorer:** none — Phase 0 measures store mutations, not retrieval.

## The gate

| Half | Threshold | Observed | Verdict |
|---|---|---|---|
| NREM edges created | > 100 | **361** | PASS |
| Consolidate proposes archives/merges | > 0 | **4 archived + 1 rewritten** | PASS |

Sleep has real work on this corpus. Phase 1 has a foundation.

## Finding 1 — REM generates zero summaries (token floor)

REM assigned 387 memories to 33 topic clusters (several with 19–25 members), then:
**Summaries generated: 0, refreshed: 0; dormancy: 0.** 29 clusters flagged "Needs summary."

Cause confirmed in code: `sleep_rem.py:671` — `if cluster_tokens < SUMMARY_MIN_DETAIL_TOKENS
(2000): continue`. LoCoMo's chat turns are short, so even a 25-member cluster is ~66–315
tokens (visible in the consolidate cluster listing) — none clears 2000. The
highest-EV stage (REM summaries — the vocabulary-gap analysis says they'd fix 13/18 hard
multi-hop misses) is therefore **gated completely off**, out of the box.

**Consequence:** REM can't be studied on LoCoMo until the floor comes down. The first REM
experiment must be a floor sweep (see Finding 2), not the entity-scoping A/B originally
slated for Phase 3.

## Finding 2 — the token floor is the wrong shape (absolute, not relative)

`git blame` → commit `50f4df2` *"Fix theme churn and summary thrashing in sleep pipeline"*.
The floor was introduced deliberately (to stop summary thrashing), but the **value 2000 is
un-derived** — no doc, no experiment, tuned against the live store's token distribution.

Its problem is *shape*: an absolute constant encodes an assumption about corpus scale. On
the live store, 2000 tokens ≈ a few substantial memories → real clusters clear it. On
LoCoMo, nothing does → REM inert. The scalar also **conflates three separable decisions**:

1. **Eligibility (size)** — should be *relative* (percentile / ratio of the corpus
   cluster-size distribution), not absolute.
2. **Value (dispersion)** — does the cluster have a vocabulary-bridge gap a summary would
   fix? Largely *size-independent* — the multi-hop payoff, and the one the token floor is
   blindest to.
3. **Timing (stability)** — settled enough not to thrash (partly the existing staleness
   gate; the thrashing problem that birthed 2000).

**Recommendation:** sweep the floor as a **percentile** where possible (so what we learn on
LoCoMo transfers across scales); at minimum sweep absolute {2000, 800, 400, 200} for the
size→retrieval curve. The sweep disambiguates the redesign: helps regardless of size →
dispersion is the real signal; helps only for relatively-large clusters → relative-size is
right; thrashes → stability needs its own gate. Candidate architecture change: replace the
scalar with (relative-size × stability × dispersion).

## Finding 3 — consolidate rejects most proposed actions (schema errors)

`sleep_consolidate.py --dry-run` on the post-NREM/REM store: 16 clusters, **4 archived, 1
rewritten, 0 merged, 33 kept, 22 REJECTED (schema errors)**. The rejections:
`no merge_content`, `no new_content`, `no arc_context`, `invalid superseded_by` — the Opus
judge *decided* to merge/rewrite/archive/annotate but didn't populate the payload fields,
so ~18% of proposed non-keep actions executed (only the running-fitness cluster fully
consolidated).

**Root cause (2026-07-04 follow-up, confirmed against the raw judge log**
`~/.somnigraph/benchmark/sleepbench-precheck/conv0/sleep_logs/consolidate_20260704_112323.log`**):
neither of the two readings originally hypothesized. It is **not** prompt-vagueness in the
diffuse sense, and **not** the model failing to substantiate — the judge's decisions were
sound and its content was written. The pipeline discards well-formed, substantiated actions
over **field-name and required-field-redundancy mismatches on the execute side.** Breakdown
of the 22:

| Rejection | Count | What the judge emitted | Cause |
|---|---|---|---|
| `merge_into` → no merge_content | **11** | `target_id` only, no content | Judge reads "merge A into B" as *fold A's existing content into B* (correct semantics); validator demanded a hand-typed `merge_content` duplicating the source memory that's about to be archived. |
| `archive` → invalid superseded_by | **5** | `{action:"archive"}`, no superseder | Judge archives true-null filler ("near-duplicate pleasantries, no distinct content"); the `superseded_by = which memory captures its info` framing has no answer when nothing captures nothing. |
| `rewrite` → no new_content | **4** | canonical text under `content`/`summary` | Pure key-name mismatch — validator reads `new_content`/`new_summary`. The judge did the work (e.g. wrote *"Started playing acoustic guitar about five years ago… her main way to express herself"*) and it was thrown away over a key prefix. |
| `annotate` → no arc_context | **2** | action verb only | Same shape as merge; `arc_context` is genuinely new text, so no safe auto-default. |

Consequences unchanged:
- Consolidation throughput is far below the opportunity count.
- **This blocks convergence.** A cluster proposed-then-rejected this pass is unchanged, so
  it is re-proposed and re-rejected next pass — the store cannot reach a fixed point until
  the rejection cause is resolved. This must be fixed before the multi-pass convergence
  experiment can converge.

**Fix (branch `fix/consolidate-schema-recovery`, execute-side only, no prompt change):**
two conservative validator recoveries that only ever *execute more of what the judge already
decided and substantiated* — (1) merge: default absent `merge_content` to the source
memory's own content; (2) rewrite: accept `content`/`summary` as aliases for
`new_content`/`new_summary`. Together these recover **15/22 (68%)** with zero new burden on
the model. The `archive` superseded_by=null case (5) is genuinely judgment-laden and *dangerous*
on a GT-evidence corpus (archiving unconditionally), so it is **deliberately left for Alexis's
review** before 07-15; `annotate` (2) needs prompt-tightening or a retry, also deferred.

**Caveat:** the *rate* may inflate on benchmark data (short, near-duplicate turns) vs the
live store, but the *root cause* (field-name/redundant-field mismatch) is corpus-independent
and the recovery fixes apply identically to the live 07-15 fork.

## Finding 4 — direct evidence for multi-pass convergence

NREM's skip breakdown: `none=751, deleted=59, edge_exists=2, cap_hit=50`. **50 pairs were
deferred because a memory hit `MAX_EDGES_PER_CYCLE` this cycle** — before counting the new
summary→detail edges a future pass would create. One pass provably undershoots steady
state, confirming the convergence hypothesis: watching the store settle over iterations
(with the cap *kept* as the rate-limiting engine, not removed) is a real, measurable
phenomenon on this corpus. Note: LoCoMo's density (everyone discussing a few shared topics)
may make convergence take more passes here than on the heterogeneous live store.

## Finding 5 — active-status GT bug in the old harness (found by the Stage A rig)

`scripts/locomo_bench/run.py:409` (`_rebuild_dia_map_from_turns`) matches
`status = 'active' AND source != 'extraction'`. Any GT evidence turn that sleep merges away
(`status='deleted'`) or makes dormant silently drops from the dia_map → the query counts as
a **miss even when the content is fully recoverable in the survivor memory** — a phantom
retrieval *loss* attributed to sleep, i.e. exactly the bias that would invalidate a
pre/post-sleep measurement. Already live on this corpus: 31 NREM merges + 4 consolidate
archives = **35 relocated memories** that would misresolve.

The new rig's `gt.py` (`resolve_dia_map`) matches over **all statuses**, avoiding this.
`run.py:409` is **fixed on this branch** to do the same (the `status = 'active'` filter
dropped), so the old harness no longer carries the bias either.

## Bugs fixed en route (production scripts)

1. **`eval_retrieval.py` / `run.py` MODEL_PATH `.pkl`→`.txt`** — the benchmark overrides
   searched only for `reranker_model.pkl`; a stale March `.pkl` shadowed the deployed V6
   `.txt`, so `--configs reranker` silently scored on formula fallback. **Merged to main**
   (`d798810`); stale `.pkl` deleted.
2. **`sleep_consolidate.py` PEP 723 header** — missing `numpy`/`lightgbm`/`fastembed`, so
   `uv run` `ModuleNotFoundError`'d before reaching consolidate. Fixed (matches the sibling
   sleep scripts).
3. **`sleep_consolidate.py` `KeyError: 'annotated'`** — `execute_actions` initialized
   `stats` without `annotated` (added lazily only when an annotate action fired), so the
   totals rollup crashed on any cluster that didn't annotate. Latent because the script is
   standalone (not in `sleep.py`'s cycle) and hadn't run recently under the abstinence hold.
   Root-cause fix: `annotated` is now part of the stats contract.

(2) and (3) are uncommitted on `exp/sleep-bench-phase0` pending review.

## What Phase 0 changed about the plan

- **REM floor sweep is now a Phase-1 prerequisite**, not a Phase-3 optimization — REM does
  nothing measurable on LoCoMo until the floor drops.
- **The consolidate schema-rejection must be resolved before the convergence experiment** —
  otherwise the store can't converge.
- **The NREM/graph measurement (H3) is ready now** via the Stage A rig — NREM built a real
  graph and doesn't create new candidate rows, so it can be scored (V6 vs formula) without
  coverage-scoring. Report raw-as-floor (35 relocated memories mean the raw-vs-coverage gap
  is small but nonzero even here).
- **Coverage-scoring stays deferred** but becomes load-bearing the moment the floor drops
  and summaries appear.

## Reproduction

```powershell
$env:SOMNIGRAPH_DATA_DIR = "C:\Users\Alexis\.somnigraph\benchmark\sleepbench-precheck\conv0"
$env:PYTHONUTF8 = "1"
uv run experiments/sleep-bench/precheck.py --conv 0 --fresh `
  --results-path experiments/sleep-bench/precheck_results_conv0.json
# consolidate step alone (after NREM+REM already ran on the store):
uv run scripts/sleep_consolidate.py --dry-run -v
```

**Provenance note (honest accounting):** the numbers here were *not* produced by one clean
end-to-end run. The initial `precheck.py` run crashed at the consolidate stage on the
`KeyError: 'annotated'` fixed above, so no `precheck_results*.json` was ever written — the
NREM/REM counts are from that run's stdout, and the consolidate counts from the standalone
`--dry-run` re-run after the fix. A clean end-to-end re-run (post-fix) now completes and
writes the results JSON; it is pending, and should be run before these findings are ratified.

## Raw counts

- **NREM deep:** 364 relationships, 361 edges, 31 dups merged, 275 priority boosts, 1
  contradiction, 21 temporal evolutions, 204 shadow-load updates; skips none=751 deleted=59
  edge_exists=2 cap_hit=50; 871s. Post: 388 active / 31 deleted / 361 edges.
- **REM:** 387 assigned, 33 topic clusters (35 taxonomy topics), 0 summaries, 0 dormancy, 29
  needs-summary gaps, 119 disconnected memories, health risk 0.3753.
- **Consolidate (dry-run):** 16 clusters / 60 memories; 4 archived, 1 rewritten, 0 merged,
  33 kept, 22 rejected (schema), 0 blocked, 0 failed; 282s.
</content>
</invoke>
