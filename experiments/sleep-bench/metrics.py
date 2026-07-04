"""Stage A retrieval metrics.

- `ndcg_at_budget`: a clean, from-scratch reimplementation of
  eval_retrieval.py's `_ndcg_at_budget` (token-budget-packed DCG/IDCG).
  Read for reference, not imported -- this rig is independent of
  eval_retrieval.py by design (see retrieve.py's module docstring).
- `recall_at_k` / `mrr`: imported VERBATIM from
  experiments/sleep-fork/analyze_sleep_fork.py. That module's math is
  correct and is what gives sleep-fork parity (spec requirement) -- lifting
  it (rather than re-deriving it) is the point.
- `recall_any` / `recall_all` / `effective_k`: LongMemEval-aligned
  additions with no upstream implementation in this repo to steal from.
"""

import math
import sys
from pathlib import Path

_SLEEP_FORK_DIR = Path(__file__).resolve().parent.parent / "sleep-fork"
if str(_SLEEP_FORK_DIR) not in sys.path:
    sys.path.insert(0, str(_SLEEP_FORK_DIR))

from analyze_sleep_fork import mrr, recall_at_k  # noqa: E402  (sleep-fork parity; see docstring)

__all__ = [
    "recall_at_k", "mrr", "ndcg_at_budget",
    "recall_any", "recall_all", "effective_k",
]


def ndcg_at_budget(
    ranked: list[str],
    gt_scores: dict[str, float],
    token_map: dict[str, int],
    budget: int,
    default_tokens: int = 100,
) -> float:
    """Token-budget NDCG. `ranked` is a full-id ranked list, `gt_scores` is
    {id: relevance} (binary 1.0/0.0 for LoCoMo evidence turns is fine --
    the DCG/IDCG math doesn't care whether relevance is graded or binary),
    `token_map` is {id: token_count} for the CURRENTLY ACTIVE pool (an
    archived/dormant-ed memory is correctly absent -- see store.
    load_token_map's docstring).

    DCG packs `ranked` into `budget` tokens greedily in rank order.
    IDCG packs the highest-relevance GT ids into `budget` tokens greedily
    by relevance (the best achievable ranking under the same budget).
    """
    shown = []
    used = 0
    for mid in ranked:
        t = token_map.get(mid, default_tokens)
        if used + t > budget:
            break
        used += t
        shown.append(mid)
    if not shown:
        return 0.0

    dcg = sum(
        gt_scores.get(mid, 0.0) / math.log2(i + 2)
        for i, mid in enumerate(shown)
    )

    ideal_candidates = sorted(gt_scores.items(), key=lambda kv: kv[1], reverse=True)
    ideal_rels = []
    used = 0
    for mid, rel in ideal_candidates:
        t = token_map.get(mid, default_tokens)
        if used + t > budget:
            continue
        used += t
        ideal_rels.append(rel)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


def recall_any(ranked_window: list[str], evidence_ids) -> bool:
    """LongMemEval-aligned loose recall: True iff ANY evidence turn is in
    the given window (caller passes `ranked[:k]` or an effective_k window)."""
    return bool(set(ranked_window) & set(evidence_ids))


def recall_all(ranked_window: list[str], evidence_ids) -> bool | None:
    """LongMemEval-aligned strict recall: True iff ALL evidence turns are
    in the given window. None if there's no evidence to check (shouldn't
    happen -- gt.build_query_set already filters those out)."""
    evidence_ids = set(evidence_ids)
    if not evidence_ids:
        return None
    return evidence_ids.issubset(set(ranked_window))


def effective_k(ranked: list[str], session_map: dict[str, str], k: int) -> int:
    """The LongMemEval turn->session-collapse correction: walk down
    `ranked`, counting unique sessions, until `k` unique sessions have been
    seen (or the list runs out). Returns the number of items consumed.

    Why this matters here specifically: a REM summary or a consolidate
    merge/rewrite can represent many raw evidence turns as ONE ranked slot.
    A literal `ranked[:k]` then unfairly shrinks the effective session
    coverage a post-sleep store is judged on, relative to a pre-sleep store
    where the same evidence is spread across k separate turn-level
    candidates. Walking k up to a session count instead gives both sides an
    equal *session* budget rather than an equal *item-count* budget.

    An id with no session mapping (e.g. a cross-session summary) is
    conservatively treated as its own singleton session -- it can advance
    the count by at most 1, same as any single turn, never inflating
    coverage.
    """
    seen = set()
    for i, mid in enumerate(ranked):
        seen.add(session_map.get(mid, mid))
        if len(seen) >= k:
            return i + 1
    return len(ranked)
