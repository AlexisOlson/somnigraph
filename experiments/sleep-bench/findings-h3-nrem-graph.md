# Findings — Sleep-Bench Step 1 (H3: NREM / graph decomposition)

**Status: DRAFT, 2026-07-04.** First retrieval measurement on a slept fork. Isolated
benchmark arms only (`~/.somnigraph/benchmark/sleepbench-arms/conv0-{frozen,nrem}`); the
live store (`~/.claude/data`, hard no-sleep abstinence hold until ~07-15) was never touched.
Companion to [`findings-phase0.md`](findings-phase0.md) (born-saturation gate) and the
[sleep-pipeline proposal](../../docs/proposals/sleep-pipeline-optimization.md) §3.4 (the
rerank-free instrument), §4 Stage B (NREM).

## Question

Does **one NREM pass** improve retrieval on LoCoMo, and if so, is the benefit carried by the
**graph edges** it builds (visible only to a PPR-aware scorer) or by its **content/index
changes** (the dup-merges, visible to raw RRF)? H3 as pre-registered: *sleep helps under
formula but ~null under rrf ⇒ the graph carries the signal.*

## The instrument (rerank-free decomposition)

Two scorers, no learned weights (V6 dropped — its feedback features are dead/OOD on a
benchmark store, see proposal §3.4):

- **`rrf`** = `impl_recall(expand=False)`: raw `rrf_fuse`, no Hebbian, no PPR. Blind to
  `memory_edges`. Isolates the **content/index** effect.
- **`formula`** = RRF + PPR over `memory_edges` (Hebbian inert here). Adds the **graph/edge**
  effect. **Δ(formula − rrf)** isolates the edge contribution.

**Validation: on the frozen arm (0 edges) Δ(formula − rrf) = exactly +0.0000 on all six
metrics (n=196).** PPR has nothing to traverse, so formula ≡ rrf bit-identical. The
decomposition is clean; every delta below is real signal.

Arms: `conv0-frozen` (pristine 419-memory ingest) and `conv0-nrem` (a fork of frozen after
one `sleep_nrem.py deep`: 377 relationships, 376 embedded edges, 24 dup-merges → 395 active).
196 scoreable GT queries. (An earlier NREM run silently produced **zero** usable edges because
the edge-embedding pass hit missing OpenAI credentials and PPR skips contextless-embedding
edges — a false-null trap now blocked by an edge-embedding health guard in `retrieve.py`; see
the "Instrument hazard" note below.)

## Result 1 — the graph is a recall↔ranking tradeoff (artifact-free)

GRAPH Δ(formula − rrf) on the **nrem** arm (same store, same GT, only the `expand` flag
differs — no merge artifact can touch this contrast):

| metric | Δ(formula − rrf) | 95% CI | verdict |
|---|---|---|---|
| recall_any@10 | **+0.0408** | [+0.005, +0.082] | ✱ graph **adds recall** |
| r@10 | **+0.0357** | [+0.002, +0.072] | ✱ graph **adds recall** |
| recall_all@10 | +0.0255 | [−0.010, +0.061] | leans + |
| recall_any@effk | +0.0153 | [−0.010, +0.041] | leans + |
| ndcg@5k | −0.0104 | [−0.024, +0.003] | leans − |
| mrr | **−0.0175** | [−0.034, −0.002] | ✱ graph **costs ranking** |

**NREM's edges carry real but mixed signal:** PPR pulls more relevant memories into the
top-10 via the graph (recall up, CI-clear), but displaces the single best hit downward (MRR
down, CI-clear). Not a uniform lift — a recall-vs-precision tradeoff. (`pos` fractions are
small, 0.03–0.12: PPR changes nothing for most queries; the effect lives in the ~10–14
queries where expansion fires, and there it nets positive on recall, negative on top-rank.)

## Result 2 — one NREM pass is recall-neutral, ranking-negative

SLEEP Δ(nrem − frozen), per scorer:

| metric | under RRF (content/index) | under FORMULA (content+graph) |
|---|---|---|
| r@10 | −0.0332 ✱ | +0.0026 (null) |
| recall_any@10 | −0.0357 ✱ | +0.0051 (null) |
| recall_any@effk | −0.0153 (null) | +0.0000 (null) |
| ndcg@5k | −0.0188 ✱ | −0.0292 ✱ |
| mrr | −0.0260 ✱ | −0.0435 ✱ |

Under raw RRF the merges look uniformly harmful. Under formula the graph's recall gain
**exactly cancels** the recall penalty (r@10 −0.033 → +0.003; recall_any@effk restored to the
frozen value 0.7704) — but the ranking metrics stay negative, because the graph neither fixes
nor offsets the MRR/NDCG hit. **Net: one NREM pass on conv0 is recall-neutral (graph
compensates) and ranking-negative.** Not a retrieval win here.

## Result 3 — the raw recall penalty is ~5/7 metric artifact

The −0.0357 raw recall_any@10 penalty is exactly 7 regressed queries. Decomposed (per-query,
against the frozen baseline):

- **6/7 — direct eviction** of a GT-evidence turn (one of the 7 evidence memories NREM merged
  away; the merge marks the loser `status='deleted'` and **hard-deletes its vector/FTS**, so
  its GT id can never be retrieved again):
  - **5 of these 6 — pure artifact:** the merge *winner* IS in nrem's top-10, but
    `recall_any@10` scores a **miss because the winner's id is not the GT evidence id.** A
    deterministic `superseded_by`-redirect (credit the winner as the loser) recovers all five.
  - **1 (query 0:77) — genuine:** the winner is not retrieved in top-10 → a real
    merge-caused top-10 loss.
- **1/7 — indirect index perturbation:** no evicted evidence; deleting 24 memories reshuffled
  RRF/BM25 enough that a *different* evidence id fell out of top-10 for one query.

**So only ~2/7 of the raw recall penalty is real** (1 genuine merge loss + 1 index
perturbation); ~5/7 is the evidence-id metric being unable to follow a merge to its survivor.
Recovery detail: across the 12 (query, evicted-evidence) pairs, the winner is retrieved in
8/12 under formula, 7/12 under rrf.

## The NREM "merge" is not a merge

`sleep_nrem.py:537-556`: on an `edge_type=="duplicate"` pair NREM picks a winner (higher
`base_priority`, tie → newer) and:

1. Sets the loser `superseded_by=<winner>`, `status='deleted'`.
2. **Hard-deletes** the loser's `memory_vec` / `memory_fts` / `memory_rowid_map` rows.
3. Leaves the **winner completely untouched** — it absorbs none of the loser's content,
   summary, or embedding. The loser's `content` column survives in the row (auditable,
   re-embeddable), but nothing at retrieval time reads it.

Two consequences:

- **The `superseded_by` tombstone is retrieval-dead.** It cannot fire at query time, because
  the thing that would trigger a redirect — the loser matching a query — is exactly what got
  deleted (its vector/FTS are gone). The pointer only helps *offline* scoring where the GT id
  is known. So on the live store there is **no retrieval-time safety net** for a merge.
- **A mis-classified "duplicate" permanently loses the loser's unique phrasing.** On LoCoMo's
  redundant chat turns this is low-stakes (the winner is a genuine near-duplicate), which is
  why only 1/196 queries genuinely regressed. On a denser personal store, over-merging risk
  is higher and the loss is unrecoverable.

**This motivates a genuine content-absorbing merge** (fold the loser's distinctive content
into the winner and re-embed) instead of keep-winner-verbatim + delete. But a cheaper option
turned out to win outright — see the next section.

## Result 4 — don't merge at all: link duplicates instead (third arm)

If index size is a non-concern, why collapse duplicates at all rather than **link and cluster**
them? Tested directly with a third arm: `sleep_nrem.py deep --no-merge` (commit `759f484`, an
opt-in, default-off flag) adds a **contextless duplicate link edge** on a `duplicate` pair —
PPR-traversable at full weight, no embedding needed — and **keeps both memories active**.
Everything else (supports/temporal edges) is identical. This isolates the delete-vs-link
contrast. Same conv0, 21 duplicate pairs linked, all 419 memories retained.

Three-way, all Δ vs frozen (n=196, paired bootstrap):

| | **link (no-merge)** | **merge (delete)** |
|---|---|---|
| under RRF, all 6 metrics | **+0.0000 exactly** (byte-identical to frozen, CI-confirmed) | −0.03 to −0.036; CI-clear negative on 4/6 |
| NDCG@5k under formula | −0.0002 (null, flat) | **−0.0292 ✱** |
| MRR under formula | −0.0027 (null, flat) | **−0.0435 ✱** |
| recall under formula | leans + (recall_any@effk **+0.0255**, CI `[0.0000, 0.0510]`) | recall-neutral (nulls) |

**Link strictly dominates merge.** What's proven CI-solid: linking has **zero** content/index
penalty (`nomerge.rrf` is byte-identical to frozen — nothing evicted, and RRF ignores the
added edges) and carries **none** of merge's significant ranking damage (merge's NDCG −0.029✱
/ MRR −0.044✱ collapse to −0.0002 / −0.0027, dead flat). What's suggested-not-proven: the
recall upside from the links — recall_any@effk +0.0255 sits with its CI lower bound *exactly*
on zero, so it's borderline, not CI-clear at n=196; the other recall metrics lean + but null.

**The airtight claim is elimination of harm at zero cost, with a hint of upside.** Deletion was
100% of the measured cost of NREM merge; linking captures the graph benefit without evicting
anything, so the top ranks stay intact (no compensation-churn → no MRR crash). The
genuinely-collapse-worthy cases (stale / contradicted info) are already separate edge types
(`temporal_evolution`, `contradiction`), not `duplicate` — so dropping destructive
duplicate-merge doesn't touch supersession-of-stale-facts.

**Recommendation:** make link-not-delete the NREM `duplicate` behavior (the `--no-merge` path).
The one case the benchmark doesn't stress — high-multiplicity *exact* re-ingested duplicates
crowding a single result — is a **retrieval-time** concern (MMR / result-level near-dup dedup),
still non-destructive. Flipping the NREM default runs on the live 07-15 fork → **Alexis's
decision**; the flag ships now, the default flip is his call.

## Verdict

- **H3 as pre-registered — not supported.** The graph does not give sleep a uniform lift; it
  gives a recall↔ranking tradeoff.
- **Refined, defensible finding:** one NREM pass on LoCoMo conv0 is **recall-neutral,
  ranking-negative**; the graph edges add recall@10 (+0.036 ✱) at the cost of MRR (−0.018 ✱);
  and the alarming raw "sleep hurts recall" signal is ~5/7 a GT-id artifact, not information
  loss. The instrument is validated (frozen Δ = 0) and the merge vs graph channels are cleanly
  separated.
- **The actionable result (Result 4): link, don't merge.** A `--no-merge` third arm that links
  duplicates instead of deleting them **strictly dominates** the merge arm — byte-identical to
  frozen under RRF (zero eviction penalty, CI-confirmed) and free of merge's CI-clear ranking
  damage (NDCG/MRR flat vs frozen), with a borderline recall gain. Deletion was 100% of the
  measured cost; the graph benefit survives linking. Recommend flipping NREM `duplicate`
  handling to link-and-cluster (default flip = Alexis's call, runs on the live 07-15 fork).
- **n = 1 conversation.** conv0 only; treat magnitudes as indicative, not final. The patterns
  (graph tradeoff; merge harm dominated by uncreditable-survivor artifact; link ⪰ merge) are
  the transferable claims.

## Instrument hazard fixed en route

A credential-less `sleep_nrem.py` run **commits its edges but soft-fails the separate
`linking_embedding` pass** (nullable column). PPR's context gate
(`scoring.py`: `if has_context and not edge["linking_embedding"]: continue`) then skips every
such edge, so a formula score of that store reads as **zero-edge — a false null on the exact
graph hypothesis.** `retrieve.py` now runs `store.edge_embedding_health()` after
`point_at_store` and **hard-fails before writing any row** when an expansion scorer is run
against a store with >2% blind (contextful, unembedded) edges (commit `9545fa6`). The clean
run above shows `edge-embedding health OK: 376 contextful edges all embedded`.

## Reproduction

```powershell
$FROZEN = "C:\Users\Alexis\.somnigraph\benchmark\sleepbench-arms\conv0-frozen"
$NREM   = "C:\Users\Alexis\.somnigraph\benchmark\sleepbench-arms\conv0-nrem"
$env:PYTHONUTF8 = "1"
$env:OPENAI_API_KEY = (Get-Content "C:\Users\Alexis\.claude\data\openai_api_key" -Raw).Trim()

Remove-Item $NREM -Recurse -Force
uv run experiments/sleep-bench/fork.py fork $FROZEN $NREM
$env:SOMNIGRAPH_DATA_DIR = $NREM
uv run scripts/sleep_nrem.py deep          # GATE: report must show "Embedded <N> edge linking contexts"

$env:SOMNIGRAPH_DATA_DIR = $FROZEN
uv run experiments/sleep-bench/retrieve.py --data-dir $FROZEN --label frozen --conv 0 --scorers rrf formula --fresh
$env:SOMNIGRAPH_DATA_DIR = $NREM
uv run experiments/sleep-bench/retrieve.py --data-dir $NREM   --label nrem   --conv 0 --scorers rrf formula --fresh
Remove-Item Env:SOMNIGRAPH_DATA_DIR
uv run experiments/sleep-bench/analyze_h3.py
```

Attribution scripts (read-only, GT-evidence eviction / winner-recovery / channel split) are
in the session scratchpad; fold into a committed `analyze_merge_recovery.py` if this becomes a
standing measurement.
