"""Score-capturing wrapper around the real `impl_recall` retrieval path.

`impl_recall()` returns a formatted markdown string -- with `internal=True`
(the convention `scripts/locomo_bench/eval_retrieval.py` uses) it does not
even log a `recall_meta` event, so there is no numeric score anywhere in the
call's observable output; both `eval_locomo` and `eval_production` already
live without per-candidate scores for this reason (their JSONL records have
no `scores` field). This rig's schema asks for one.

To get real per-candidate scores WITHOUT (a) reimplementing retrieval or
(b) polluting the store's event log with query-order-dependent `retrieved`
events -- which would contaminate later queries' Hebbian / session-recency
features within the same eval run, a subtler measurement-validity problem
than a store-size side effect -- this module calls `impl_recall(internal=
False)`, which *does* compute and log a `recall_meta` event carrying
per-candidate scores, but temporarily replaces `memory.tools._log_event`
with a capturing stub so that event (and the `retrieved` events) are
captured in memory and never actually written to the DB.

This is instrumentation of a logging side-channel, not a reimplementation
of scoring: every score comes from the one real `impl_recall` call.

Known, unavoidable side effect (inherited, not introduced by this module):
`impl_recall` unconditionally bumps `last_accessed` / `recall_count` /
`access_count` for whatever it returns, regardless of `internal` (see
`memory/tools.py` around the `UPDATE memories SET last_accessed = ...`
call, which sits outside the `if not internal:` block). Existing harnesses
already accept this; this rig does too.
"""

import memory.tools as tools_mod


def recall_with_scores(
    query: str,
    context: str,
    budget: int,
    limit: int,
) -> tuple[list[str], dict[str, float], float]:
    """Returns (ranked_short_ids, {short_id: score}, top_score).

    ranked_short_ids is exactly the order impl_recall would have returned
    (the `kept` list in its recall_meta -- already filtered to `limit`
    results and budget-truncated within the formatting loop).
    """
    captured: dict = {}

    def _capturing_log_event(
        db, memory_id, event_type,
        query=None, session_id=None, co_memory_ids=None,
        similarity_score=None, context=None,
    ):
        if event_type == "recall_meta":
            captured["recall_meta"] = context
        # 'retrieved' events intentionally dropped: no DB write, no
        # cross-query contamination of hebb_data/session features within
        # this eval run's own query stream.

    original_log_event = tools_mod._log_event
    tools_mod._log_event = _capturing_log_event
    try:
        tools_mod.impl_recall(
            query=query, context=context, budget=budget, limit=limit,
            internal=False,
        )
    finally:
        tools_mod._log_event = original_log_event

    meta = captured.get("recall_meta")
    if not meta:
        return [], {}, 0.0

    kept = meta.get("kept", [])  # list of [short_id, score], rank order
    ranked_short_ids = [pair[0] for pair in kept]
    scores = {pair[0]: pair[1] for pair in kept}
    return ranked_short_ids, scores, meta.get("top_score", 0.0)
