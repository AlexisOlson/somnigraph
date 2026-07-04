# 2026-07-04 — Sleep-Bench Step 1 (H3) + the merge→link default flip

Continuation of the 2026-07-04 sleep-bench Phase 0 session. This one ran Step 1 (the
NREM/graph H3 measurement), found a false-null instrument hazard, and — driven by Alexis's
questions — turned NREM's destructive duplicate-merge into non-destructive linking as the new
default. All work on isolated benchmark arms; live store never touched (abstinence hold to ~07-15).

## What shipped

- **H3 result** (`experiments/sleep-bench/findings-h3-nrem-graph.md`): three-arm decomposition on
  conv0 (frozen / nrem-merge / nomerge-link), 196 GT queries, paired bootstrap, rerank-free
  instrument (rrf vs formula; V6 dropped as OOD).
- **Edge-embedding health guard** (`retrieve.py` + `store.edge_embedding_health()`, `9545fa6`).
- **`--no-merge` → link default** flip in `sleep_nrem.py`: `write_results(link_duplicates=True)`,
  `--legacy-merge` opt-out (`759f484` → `61f4511` → `30942fd`).

## Findings

1. **Instrument validated.** GRAPH Δ(formula−rrf) on frozen = exactly +0.0000 on all 6 metrics
   (0 edges → PPR no-op → formula ≡ rrf bit-identical). Everything downstream is real signal.
2. **H3 as pre-registered — not supported.** NREM's edges are a **recall↔ranking tradeoff**, not
   a uniform lift: Δ(formula−rrf) on nrem = recall@10 +0.036✱, recall_any@10 +0.041✱, MRR
   −0.018✱, NDCG ~−0.010. PPR finds more via the graph but displaces the top hit.
3. **Destructive merge is a net loss.** One NREM pass (merge arm) vs frozen = recall-neutral,
   ranking CI-clear negative (NDCG −0.029✱, MRR −0.044✱). The scary raw recall −0.036 decomposes
   (per-query, vs frozen): **5/7 pure GT-id artifact** (merge winner is retrieved but scores a
   miss because its id ≠ the GT evidence id), **1/7 genuine merge regression** (query 0:77), **1/7
   indirect index perturbation**. Only ~2/7 real.
4. **The NREM "merge" was never a merge.** `sleep_nrem.py`: pick a winner, set loser
   `superseded_by`+`status='deleted'`, **hard-delete** its vec/FTS, leave the winner untouched
   (absorbs nothing). The `superseded_by` tombstone is **retrieval-dead** — it can't redirect at
   query time because the loser's index rows are deleted; it only helps offline scoring.
5. **★ Link strictly dominates merge (the actionable result).** `--no-merge` third arm links
   duplicates (contextless PPR-traversable edge, weight 1.0, no embedding) and keeps both memories.
   `nomerge.rrf` is **byte-identical to frozen** (zero eviction penalty, CI-confirmed);
   ranking flat under formula (merge's −0.029✱/−0.044✱ → −0.0002/−0.0027); recall leans + but
   borderline (recall_any@effk +0.0255, CI lower bound on 0). **Deletion was 100% of merge's cost.**

## Decisions (Alexis)

- **Link is the new NREM `duplicate` default**, flipped in `write_results` so the orchestrated
  `sleep.py` cycle inherits it. `--legacy-merge` restores the destructive path for ablation. He
  called the result "obvious/clean" and flipped on n=1 (conv0) without waiting for conv1–2.
- **All layers, option (b).** NREM's `gather()` has no layer filter, so summary/gestalt duplicates
  flow through this branch on the live store. NREM **links them non-destructively** rather than
  collapsing — leaving the pair intact for **consolidate** to synthesize into one canonical summary
  (the `duplicate` edge is the signal). Verbatim-collapse would destroy the loser before consolidate
  could rewrite it.

## Surprises

- The raw "sleep hurts recall" signal was almost entirely a **measurement artifact** (the metric
  can't follow a merge to its survivor), not information loss — only found by decomposing per-query
  against the frozen baseline and checking winner-retrieval + the direct/indirect channel split.
- A **credential-less NREM run silently produces a false null**: it commits edges but soft-fails
  the separate `linking_embedding` pass (nullable column), and PPR's context gate skips every
  unembedded-but-contextful edge → the store scores as zero-edge. Caught by reading the scoring
  code path, not the run summary ("377 relationships found" looked fine). Now blocked by the guard.
- Frozen (all duplicates present) **out-recalled** the merged arm — direct evidence that collapsing
  duplicates bought nothing on this benchmark; deletion was pure cost.

## Caveats / open

- **n = 1 conversation** (conv0). Pattern (graph tradeoff; merge harm ~artifact; link ⪰ merge) is
  the transferable claim; magnitudes indicative. conv1–2 generalization deprioritized by Alexis.
- **TODO: wire consolidate to rewrite duplicate-edged summary/gestalt pairs** into one canonical
  node. Until then duplicate summaries stay *linked* (harmless) but not deduplicated. Only testable
  after the REM floor sweep (H7) — LoCoMo produces 0 summaries today (REM inert, phase0 Finding 1).
- **High-multiplicity exact detail dupes crowding a result** is unstressed by the benchmark; the
  fix is retrieval-time (MMR / result-level near-dup dedup), still non-destructive.

## Files touched

- `scripts/sleep_nrem.py` — `--legacy-merge` flag; `write_results(link_duplicates=True)`; duplicate
  branch links (all layers) instead of collapsing; `duplicates_linked` stat + report line.
- `experiments/sleep-bench/{retrieve.py,store.py}` — edge-embedding health guard.
- `experiments/sleep-bench/{scorers.py,capture.py,analyze_h3.py}` — (prior in-session) rrf scorer,
  expand flag, H3 analyzer.
- `experiments/sleep-bench/findings-h3-nrem-graph.md` — the Step 1 finding (new).
- `src/memory/tools.py` — (prior in-session) `impl_recall(expand=…)`.

## State for the next session

- `main` @ `30942fd`, ~25 commits ahead of origin, **unpushed** (spans Phase 0 + sleep-fork + H3).
  Push is Alexis's call.
- Working tree carries ONLY the unrelated 2026-07-03 thread (`docs/roadmap.md`,
  `research/sources/index.md`, + 2 untracked) — **never stage these**.
- Arms live at `~/.somnigraph/benchmark/sleepbench-arms/conv0-{frozen,nrem,nomerge}`.
- Constraints unchanged: no sleep on live `~/.claude/data` until ~07-15; sleep/probe runs →
  Alexis's terminal; OpenAI 1536d (leave `SOMNIGRAPH_EMBEDDING_BACKEND` unset, `PYTHONUTF8=1`, set
  `OPENAI_API_KEY` from `~/.claude/data/openai_api_key` for direct sleep runs on keyless forks).

## Next

1. **Consolidate rewrite path** for duplicate-edged summary pairs (closes the TODO; reuses the
   `merge_into` fix from the Phase 0 session).
2. **Step 2 — REM floor sweep (H7)**: `SUMMARY_MIN_DETAIL_TOKENS` is an absolute floor that blocks
   every LoCoMo summary; sweep it (percentile-relative) to unblock REM and make the summary-dup /
   2×2 work testable.
3. Then coverage judge, then the pre/post-sleep reranker 2×2 (H10).
