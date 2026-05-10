# 2026-05-08 — Phase 1 of pinned-target labels — probe_target events + legacy backfill

Added a dedicated `probe_target` event type, emitted once per probe query (hits and misses, all modes including `extra`), keyed to the full pinned memory id. Captures `mode`, 1-indexed `target_rank` (None on Phase-1 miss), `candidate_pool_size`, and `vector_input` so the GT builder can reconstruct the (query, context) pair without timestamp-window joins to `recall_meta`.

`build_gt_from_feedback.py` gained `add_probe_target_gt()` (live reader) and `_backfill_probe_targets_from_legacy()` (one-time recovery from `[probe-{mode}] target_rank=...` reason strings + recall_miss probe rows), with shared dedup helper `_apply_probe_pin` enforcing keep-existing-on->=0.9 / overwrite-on-<0.5 / max-merge-on-0.5..0.9.

CLI flags `--no-probe-targets` and `--no-backfill` enable ablation. Output is byte-stable across consecutive runs (sort_keys=True, verified by file hash).

## Backfill results

Backfill against production DB: 182 distinct (query, target) pairs surfaced (154 hits + 28 misses) from 7,304 feedback rows + 51 recall_miss probe rows; **28 net-new (q, mid) pairs** vs the no-probe baseline (most legacy targets already had real-feedback labels >=0.9 and were correctly dedup'd as collisions_kept), 183 new vec_input_overrides recovered from recall_meta.

Forward path will start writing probe_target events on the next live probe run; legacy reasons stay as fallback until `--no-backfill` becomes default. One-off `scripts/_probe_backfill_check.py` deleted; the backfill itself logs the same numbers.

## Caveat

Phase-1-miss rows are GT-only and won't influence the gradient until train_reranker.py's candidate pool is widened or weighted differently — that's where Phase 2 (mode-stratified weighting + neighbor down-weighting) picks up.

## Surprises

Net-new pairs (28) came in well below the audit's distinct-pair count (182) because the legacy probe queries that *did* land hits had already been LLM-rated 0.9-1.0 in real feedback events; the pinned label was redundant for the easy case. The structural value of the change is for future probes (especially Phase-1 misses) where the pin records ground truth even when the LLM judge never sees the target.
