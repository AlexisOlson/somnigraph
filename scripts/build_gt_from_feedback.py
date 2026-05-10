# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "sqlite-vec>=0.1.6",
#     "openai>=2.0.0",
#     "tiktoken>=0.7.0",
#     "mcp[cli]>=1.2.0",
#     "numpy>=1.26",
#     "lightgbm>=4.0",
#     "fastembed>=0.4.0",
# ]
# ///
"""
Build ground-truth labels from retrieval feedback events.

Extracts (query, memory_id, utility) triples from the memory_events table
and writes a GT JSON compatible with tune_gt.py / train_reranker.py.

For each query, the mean utility across feedback events is used as the
relevance label. Queries with fewer than --min-memories rated memories
are dropped (too sparse to learn from).

Usage:
  uv run build_gt_from_feedback.py                    # Default output
  uv run build_gt_from_feedback.py --min-memories 3   # Require 3+ rated mems
  uv run build_gt_from_feedback.py --output path.json # Custom output path
"""

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from memory.constants import DATA_DIR
from memory.db import DB_PATH


def build_gt(
    db_path: Path, min_memories: int = 3
) -> tuple[dict[str, dict[str, float]], set[str]]:
    """Build GT dict from feedback events.

    Returns (gt, live_queries) where:
      gt: {query_text: {memory_id: mean_utility}}
      live_queries: queries with at least one non-probe feedback row (the
        "real usage" set used by the sample-weights sidecar to distinguish
        live recall from probe-driven feedback).
    """
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row

    # Filter feedback rows whose memory_id is no longer active. Inactive
    # memories (deleted/superseded/dormant or hard-deleted) can't be
    # retrieved by the candidate-pool channels (FTS/vec only return active),
    # so any GT label on them is unscoreable — they bloat query counts and
    # inflate the Phase-1 miss rate without contributing training signal.
    # Probe_target parser and legacy backfill already do this filter; the
    # main feedback parser previously did not (caught 2026-05-09 during V5+2
    # retrain analysis: 856/14,618 = 5.9% of feedback rows referenced inactive
    # memories, inflating the held-out hard miss rate).
    active_ids = {
        r["id"] for r in db.execute("SELECT id FROM memories WHERE status = 'active'")
    }

    # Collect all feedback events with query context
    rows = db.execute("""
        SELECT query, memory_id, context FROM memory_events
        WHERE event_type = 'feedback'
          AND context IS NOT NULL
          AND query IS NOT NULL
          AND query != ''
        ORDER BY created_at
    """).fetchall()

    # Aggregate: (query, memory_id) -> [utility scores]
    # live_queries = queries with at least one non-probe feedback event
    utilities: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    live_queries: set[str] = set()
    skipped_inactive_rows = 0
    for r in rows:
        if r["memory_id"] not in active_ids:
            skipped_inactive_rows += 1
            continue
        try:
            ctx = json.loads(r["context"])
            if "utility" not in ctx:
                continue
            utilities[r["query"]][r["memory_id"]].append(ctx["utility"])
            reason = ctx.get("reason") or ""
            if not reason.startswith("[probe-"):
                live_queries.add(r["query"])
        except (json.JSONDecodeError, TypeError):
            continue

    # Build GT: mean utility per (query, memory)
    gt = {}
    skipped = 0
    for query, mems in utilities.items():
        if len(mems) < min_memories:
            skipped += 1
            continue
        gt[query] = {
            mid: sum(scores) / len(scores)
            for mid, scores in mems.items()
        }

    db.close()

    # Drop live_queries that didn't survive the min_memories filter so the
    # set matches what the sidecar will actually classify.
    live_queries &= set(gt.keys())

    print(f"Feedback events parsed: {sum(len(m) for m in utilities.values())}")
    print(f"Feedback rows skipped (memory no longer active): {skipped_inactive_rows}")
    print(f"Queries with feedback: {len(utilities)}")
    print(f"Queries after min_memories={min_memories} filter: {len(gt)}")
    print(f"Queries skipped (too sparse): {skipped}")

    # Distribution stats
    all_labels = [v for mems in gt.values() for v in mems.values()]
    if all_labels:
        import statistics
        print(f"\nLabel distribution:")
        print(f"  Count: {len(all_labels)}")
        print(f"  Mean:  {statistics.mean(all_labels):.3f}")
        print(f"  Stdev: {statistics.stdev(all_labels):.3f}")
        print(f"  Min:   {min(all_labels):.3f}")
        print(f"  Max:   {max(all_labels):.3f}")

        # Bin distribution
        bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
        labels_arr = sorted(all_labels)
        for i in range(len(bins) - 1):
            count = sum(1 for v in labels_arr if bins[i] <= v < bins[i+1])
            print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {count}")

    return gt, live_queries


# Legacy reason format produced by probe_recall.py before the dedicated
# probe_target event existed:
#   [probe-{mode}] target_rank=<id8>=<rank-or-miss>[, <id8>=...] | <free text>
_PROBE_REASON_RE = re.compile(
    r"\[probe-(?P<mode>[a-z]+)\]\s+target_rank=(?P<blob>[^|]+)\s*\|"
)

# Mode priority (lower = wins). When the same query appears in multiple probe
# modes (across runs or sources), pick the strictest. Hard wins so a query
# that ever showed up in adversarial-hard stays held out.
_PROBE_MODE_PRIORITY = {
    "probe-hard": 0,
    "probe-mild": 1,
    "probe-extra": 2,
    "probe-natural": 3,
}


def _merge_probe_mode(modes: dict, query: str, full_mode: str) -> None:
    """Update modes[query] with full_mode, keeping the highest-priority mode."""
    prev = modes.get(query)
    if prev is None:
        modes[query] = full_mode
        return
    prev_pri = _PROBE_MODE_PRIORITY.get(prev, 99)
    new_pri = _PROBE_MODE_PRIORITY.get(full_mode, 99)
    if new_pri < prev_pri:
        modes[query] = full_mode


def _resolve_short_ids(db: sqlite3.Connection, prefixes: set) -> dict:
    """Map 8-char id prefixes to full ids.

    Returns {prefix: full_id} only for prefixes that resolve to exactly one
    active memory. Ambiguous prefixes (collisions) are dropped and counted
    by the caller via comparing input/output sizes.
    """
    if not prefixes:
        return {}
    resolved = {}
    placeholders = ",".join("?" for _ in prefixes)
    rows = db.execute(
        f"SELECT id FROM memories "
        f"WHERE substr(id, 1, 8) IN ({placeholders})",
        list(prefixes),
    ).fetchall()
    counts = defaultdict(list)
    for r in rows:
        full = r["id"]
        counts[full[:8]].append(full)
    for short, fulls in counts.items():
        if len(fulls) == 1:
            resolved[short] = fulls[0]
    return resolved


def _apply_probe_pin(
    gt: dict,
    vec_input_overrides: dict,
    query: str,
    target_id_full: str,
    label: float,
    vector_input,
    counters: dict,
) -> None:
    """Apply a single pinned-target row with the dedup rules from the plan.

    Dedup vs existing gt[query][target_id]:
      - existing >= 0.9 : keep existing (real-feedback already strong-positive)
      - existing < 0.5  : overwrite with pinned label (likely a rating error)
      - 0.5 <= existing < 0.9 : keep max(existing, pinned)

    vec_input_overrides[query] only set when not already present.
    """
    if query in gt and target_id_full in gt[query]:
        existing = gt[query][target_id_full]
        if existing >= 0.9:
            counters["collisions_kept"] += 1
        elif existing < 0.5:
            gt[query][target_id_full] = label
            counters["collisions_overwritten"] += 1
            print(f"  collision overwrite: query={query!r} mid={target_id_full[:8]} "
                  f"existing={existing:.2f} -> pinned={label:.2f}", file=sys.stderr)
        else:
            new_val = max(existing, label)
            if new_val != existing:
                gt[query][target_id_full] = new_val
            counters["collisions_max"] += 1
    else:
        if query not in gt:
            gt[query] = {}
        gt[query][target_id_full] = label
        counters["added"] += 1

    if vector_input and query not in vec_input_overrides and vector_input != query:
        vec_input_overrides[query] = vector_input
        counters["vec_inputs_written"] += 1


def add_probe_target_gt(
    db_path: Path,
    gt: dict,
    vec_input_overrides: dict,
    label_hit: float = 1.0,
    label_miss: float = 1.0,
) -> tuple[dict, dict[str, str]]:
    """Inject pinned-target rows from live probe_target events.

    Every probe_target event becomes one (query, target_id, label) row.
    Hits and misses both get label 1.0 by default — the pinned target IS
    the right answer regardless of whether the live retrieval surfaced it.
    Phase 2 (mode-stratified weighting) is where hit/miss starts to differ.

    Returns (counters, probe_query_modes, probe_pinned) where:
      probe_query_modes: query → "probe-{mode}" for queries actually
        contributed to gt (after dedup). Used by the sample-weights sidecar.
      probe_pinned: query → set of pinned target memory_ids observed across
        probe_target events. The sidecar resolver collapses this to a single
        id (or null when ambiguous, with a warning).
    """
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT memory_id, query, context, created_at FROM memory_events "
        "WHERE event_type = 'probe_target' "
        "  AND query IS NOT NULL AND query != '' "
        "  AND memory_id IS NOT NULL "
        "ORDER BY created_at"
    ).fetchall()

    counters = {
        "rows_scanned": len(rows),
        "added": 0,
        "collisions_kept": 0,
        "collisions_overwritten": 0,
        "collisions_max": 0,
        "vec_inputs_written": 0,
        "hits": 0,
        "misses": 0,
        "skipped_missing_memory": 0,
    }

    # Filter to active memories only — pins targeting forgotten memories
    # would be label-only with no candidate-pool presence.
    active = {
        r["id"] for r in db.execute(
            "SELECT id FROM memories WHERE status = 'active'"
        )
    }

    probe_query_modes: dict[str, str] = {}
    probe_pinned: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        mid = r["memory_id"]
        if mid not in active:
            counters["skipped_missing_memory"] += 1
            continue
        try:
            ctx = json.loads(r["context"]) if r["context"] else {}
        except (json.JSONDecodeError, TypeError):
            ctx = {}
        target_rank = ctx.get("target_rank")
        if target_rank is None:
            counters["misses"] += 1
            label = label_miss
        else:
            counters["hits"] += 1
            label = label_hit
        vec_input = ctx.get("vector_input")
        mode = ctx.get("mode") or "natural"
        full_mode = f"probe-{mode}"
        _apply_probe_pin(
            gt, vec_input_overrides, r["query"], mid, label, vec_input, counters,
        )
        _merge_probe_mode(probe_query_modes, r["query"], full_mode)
        probe_pinned[r["query"]].add(mid)

    db.close()
    return counters, probe_query_modes, dict(probe_pinned)


def _backfill_probe_targets_from_legacy(
    db_path: Path,
    gt: dict,
    vec_input_overrides: dict,
    label_hit: float = 1.0,
    label_miss: float = 1.0,
) -> tuple[dict, dict[str, str]]:
    """One-time backfill: parse pinned targets from legacy probe artifacts.

    Two legacy sources, in priority order for dedup:
      1. recall_miss events with source='probe' — memory_id is the full
         target id, target_rank=miss by definition (Phase-1 miss).
      2. feedback events whose context.reason is "[probe-{mode}] target_rank=
         <id8>=<rank-or-miss> | <free text>" — id is an 8-char prefix that
         must be resolved against memories.id.

    Skips backfilled rows whose (query, target_id) already exists in gt
    (live probe_target events ran first; they win).

    Vector input is recovered from a recall_meta event for the same query
    when available — confirmed ~50% recoverable by the audit script.
    """
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row

    counters = {
        "feedback_rows_scanned": 0,
        "reason_matched": 0,
        "miss_rows_scanned": 0,
        "distinct_pairs_short": 0,
        "prefixes_unresolved": 0,
        "prefixes_collision": 0,
        "added": 0,
        "collisions_kept": 0,
        "collisions_overwritten": 0,
        "collisions_max": 0,
        "vec_inputs_written": 0,
        "vec_inputs_recovered": 0,
        "hits": 0,
        "misses": 0,
    }

    # query → "probe-{mode}" for queries this backfill actually wrote into
    # gt. Populated only after _apply_probe_pin runs to avoid classifying
    # queries we ended up filtering out.
    candidate_modes: dict[str, str] = {}

    # Pass 1: collect (query, short_id, mode, label) tuples from feedback reasons.
    pending_short = []  # list of (query, short_id, label)
    short_prefixes = set()

    for r in db.execute(
        "SELECT query, context FROM memory_events "
        "WHERE event_type = 'feedback' "
        "  AND context IS NOT NULL "
        "  AND query IS NOT NULL AND query != ''"
    ):
        counters["feedback_rows_scanned"] += 1
        try:
            ctx = json.loads(r["context"])
        except (json.JSONDecodeError, TypeError):
            continue
        reason = ctx.get("reason") or ""
        if not reason.startswith("[probe"):
            continue
        m = _PROBE_REASON_RE.match(reason)
        if not m:
            continue
        counters["reason_matched"] += 1
        mode = m.group("mode")
        full_mode = f"probe-{mode}"
        _merge_probe_mode(candidate_modes, r["query"], full_mode)
        blob = m.group("blob").strip()
        for part in blob.split(","):
            part = part.strip()
            if "=" not in part:
                continue
            short, rank = part.split("=", 1)
            short, rank = short.strip(), rank.strip()
            if len(short) != 8:
                continue
            label = label_miss if rank == "miss" else label_hit
            pending_short.append((r["query"], short, label, rank))
            short_prefixes.add(short)

    # Pass 2: collect from recall_miss events (full id already available).
    pending_full = []  # list of (query, full_id, label, vec_input_or_none)
    for r in db.execute(
        "SELECT memory_id, query, context FROM memory_events "
        "WHERE event_type = 'recall_miss' "
        "  AND query IS NOT NULL AND query != '' "
        "  AND memory_id IS NOT NULL"
    ):
        counters["miss_rows_scanned"] += 1
        try:
            ctx = json.loads(r["context"]) if r["context"] else {}
        except (json.JSONDecodeError, TypeError):
            ctx = {}
        if ctx.get("source") != "probe":
            continue
        if len(r["memory_id"]) <= 8:
            continue
        miss_mode = ctx.get("mode")
        if miss_mode:
            _merge_probe_mode(candidate_modes, r["query"], f"probe-{miss_mode}")
        pending_full.append((r["query"], r["memory_id"], label_miss, None))

    # Resolve short prefixes to full ids (active memories only).
    active_ids = {
        x["id"] for x in db.execute("SELECT id FROM memories WHERE status = 'active'")
    }
    resolved = _resolve_short_ids(db, short_prefixes)
    counters["distinct_pairs_short"] = len({(q, s) for q, s, _, _ in pending_short})
    counters["prefixes_unresolved"] = len(short_prefixes - resolved.keys())
    counters["prefixes_collision"] = sum(
        1 for s in short_prefixes
        if s not in resolved and any(
            (x["id"][:8] == s) for x in db.execute(
                "SELECT id FROM memories WHERE substr(id,1,8) = ?", (s,)
            )
        )
    )

    # Dedup to distinct (query, full_id) pairs. Same (q, mid) often appears
    # in many feedback rows because the LLM judge rates it across multiple
    # sessions; we want one pinned label per distinct pair, with hit
    # winning over miss if they conflict (a single hit is enough evidence).
    pair_info = {}  # (query, full_id) -> (label, was_hit)
    for q, short, label, rank in pending_short:
        full = resolved.get(short)
        if not full or full not in active_ids:
            continue
        key = (q, full)
        is_hit = rank != "miss"
        prev = pair_info.get(key)
        if prev is None or (is_hit and not prev[1]):
            pair_info[key] = (label, is_hit)
    for q, full, label, _vec_input in pending_full:
        if full not in active_ids:
            continue
        key = (q, full)
        if key not in pair_info:
            pair_info[key] = (label, False)

    legacy_rows = [(q, full, label, None) for (q, full), (label, _) in pair_info.items()]
    for _key, (_, is_hit) in pair_info.items():
        if is_hit:
            counters["hits"] += 1
        else:
            counters["misses"] += 1

    # Recover vector_input from recall_meta for queries we'll actually inject.
    queries_needing_vec = {
        q for q, _, _, _ in legacy_rows if q not in vec_input_overrides
    }
    vec_recovered = {}
    if queries_needing_vec:
        # Batch fetch — one row per query is enough; LIMIT 1 inside SQL.
        for q in queries_needing_vec:
            row = db.execute(
                "SELECT context FROM memory_events "
                "WHERE event_type = 'recall_meta' AND query = ? "
                "ORDER BY created_at LIMIT 1",
                (q,),
            ).fetchone()
            if not row:
                continue
            try:
                ctx = json.loads(row["context"]) if row["context"] else {}
            except (json.JSONDecodeError, TypeError):
                continue
            vi = ctx.get("vector_input")
            if vi:
                vec_recovered[q] = vi

    counters["vec_inputs_recovered"] = len(vec_recovered)

    # Apply rows via shared dedup helper. Sort for deterministic / byte-stable
    # output across runs.
    legacy_rows.sort(key=lambda x: (x[0], x[1]))
    queries_actually_applied: set[str] = set()
    backfill_pinned: dict[str, set[str]] = defaultdict(set)
    for q, full, label, vec_input in legacy_rows:
        vi = vec_input or vec_recovered.get(q)
        _apply_probe_pin(
            gt, vec_input_overrides, q, full, label, vi, counters,
        )
        queries_actually_applied.add(q)
        backfill_pinned[q].add(full)

    # Only return modes for queries that survived dedup into gt — queries
    # whose only contribution was an unresolved/inactive prefix never made
    # it into the relevance signal and shouldn't get a probe-mode tag.
    probe_query_modes = {
        q: m for q, m in candidate_modes.items()
        if q in queries_actually_applied and q in gt
    }
    backfill_pinned = {
        q: mids for q, mids in backfill_pinned.items()
        if q in queries_actually_applied and q in gt
    }

    db.close()
    return counters, probe_query_modes, backfill_pinned


def add_synthetic_self_anchor_gt(
    db_path: Path,
    gt: dict,
    vec_input_overrides: dict,
    label: float = 1.0,
) -> tuple[int, set[str]]:
    """Augment GT with one anchor per active memory shaped like a real recall call.

    Production recall takes (query, context) where the FTS channel runs on
    `query` and the vec channel runs on `context` (or `query` if context is
    empty). Real feedback events log only the query string — short keyword
    bags, median 8 tokens. The first synthetic-GT pass used (summary, summary)
    and produced Goodhart: 0 audit pathologies on summary-as-query but 239 on
    content-residual, because the model learned "summary-shape -> trust
    channels" without learning the (query, context) asymmetry.

    This anchor mirrors real production usage:
      query   = themes-joined (kebab-split, lowercased) — FTS-style topical
                keywords, ~10 tokens, close to real-GT distribution
      context = summary — the longer informative text vec embeds, matching
                what an agent passes as recall(context=...)

    The (query != context) split is recorded in vec_input_overrides so
    training runs FTS on the keyword query and vec on the summary context,
    just like impl_recall. Real-feedback queries that share text with a
    synthetic anchor keep their existing context (real recall_meta wins).

    Memories without themes (or whose themes-joined string equals their
    summary) fall back to (summary, summary) — same as the legacy single-shape
    behavior — so coverage stays at 1 anchor per active memory.

    Returns (added, synthetic_queries, synthetic_pinned) where
    synthetic_pinned maps query → memory_id (or None if two memories
    collide on the same themes-joined query and disagree on target).
    """
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT id, summary, themes FROM memories "
        "WHERE status = 'active' AND summary IS NOT NULL AND summary != ''"
    ).fetchall()
    db.close()

    added = 0
    skipped_collision = 0
    n_themes_shape = 0
    n_summary_fallback = 0
    synthetic_queries: set[str] = set()
    synthetic_pinned: dict[str, str | None] = {}

    for r in rows:
        summary = (r["summary"] or "").strip()
        if not summary:
            continue
        try:
            themes = json.loads(r["themes"]) if r["themes"] else []
        except (json.JSONDecodeError, TypeError):
            themes = []

        themes_q = " ".join(
            str(t).replace("-", " ").strip().lower()
            for t in themes
            if str(t).strip()
        )
        themes_q = " ".join(themes_q.split())  # normalize whitespace

        if themes_q and themes_q != summary:
            query, context = themes_q, summary
            n_themes_shape += 1
        else:
            query, context = summary, summary
            n_summary_fallback += 1

        if query in gt:
            if r["id"] not in gt[query]:
                gt[query][r["id"]] = label
                added += 1
            else:
                skipped_collision += 1
        else:
            gt[query] = {r["id"]: label}
            added += 1
        synthetic_queries.add(query)
        # Track pinned target. Two memories whose themes_q strings collide
        # would each claim the same query; mark None so the sidecar pins
        # nothing (would be a coin flip otherwise).
        prev = synthetic_pinned.get(query, "__unset__")
        if prev == "__unset__":
            synthetic_pinned[query] = r["id"]
        elif prev != r["id"]:
            synthetic_pinned[query] = None
        # Only record a context override when it differs from the query.
        # Real recall_meta entries (loaded later from the DB) take precedence —
        # we don't overwrite hard-won real-usage context.
        if context != query and query not in vec_input_overrides:
            vec_input_overrides[query] = context

    n_synth_ambiguous = sum(1 for v in synthetic_pinned.values() if v is None)
    print(f"\nSynthetic self-anchor GT:")
    print(f"  Memories scanned:                     {len(rows)}")
    print(f"  Anchors with themes-query+summary-ctx: {n_themes_shape}")
    print(f"  Anchors falling back to (summary,summary): {n_summary_fallback}")
    print(f"  Total anchors added:                  {added}")
    print(f"  (query, mid) collisions (kept existing label): {skipped_collision}")
    print(f"  Pinned targets ambiguous (themes_q collision): {n_synth_ambiguous}")
    print(f"  vec_input_overrides recorded:         {len(vec_input_overrides)}")
    return added, synthetic_queries, synthetic_pinned


# ---------------------------------------------------------------------------
# Sample-weights sidecar (Phase 2)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHT_SCHEDULE: dict[str, float] = {
    "live": 1.0,
    "probe-natural": 1.0,
    "probe-mild": 1.2,
    "probe-hard": 0.0,
    "probe-extra": 1.0,
    "synthetic-self-anchor": 0.5,
}

DEFAULT_HOLDOUT_MODES: set[str] = {"probe-hard"}

# Phase 3a (2026-05-09): per-(q,m) up-weighting of pinned target rows.
# Sidecar metadata records the trainer's default boost so reruns and audits
# can see what value was assumed at GT-build time. Train-time --pinned-boost
# overrides this; the sidecar is informational, not consumed for training.
# Default locked at 5.0 by the V3 sweep (2026-05-09): held-out probe-hard NDCG
# 0.9138 (boost=1.0) → 0.9302 (boost=5.0); R@10 0.9755 → 0.9821; RMSE flat;
# live NDCG -0.0075 (designed trade-off, live ceiling is labels not model).
# 3.0 sat below 2.0 on hard NDCG (0.9187 vs 0.9226) — fold variance ~±0.005.
DEFAULT_PINNED_BOOST: float = 5.0


def build_sample_weights_sidecar(
    gt: dict,
    *,
    live_queries: set[str],
    probe_query_modes: dict[str, str],
    synthetic_queries: set[str],
    weight_schedule: dict[str, float],
    holdout_modes: set[str],
    probe_pinned: dict[str, set[str]] | None = None,
    backfill_pinned: dict[str, set[str]] | None = None,
    synthetic_pinned: dict[str, str | None] | None = None,
    pinned_boost_default: float = DEFAULT_PINNED_BOOST,
) -> dict:
    """Build the per-query sample-weights sidecar.

    Resolution priority when a query appears in multiple sources:
      live > probe-{mode} > synthetic-self-anchor.
    Real-usage signal trumps probe origin which trumps synthetic coverage.

    Schema v2 (Phase 3a, 2026-05-09): each weight entry also carries a
    pinned_target_id (null when no canonical answer or sources disagree).
    The trainer applies a per-row boost (default DEFAULT_PINNED_BOOST,
    overridable via --pinned-boost) to rows where memory_id == pinned_target_id.
    """
    probe_pinned = probe_pinned or {}
    backfill_pinned = backfill_pinned or {}
    synthetic_pinned = synthetic_pinned or {}

    weights_block: dict[str, dict] = {}
    by_source: dict[str, int] = defaultdict(int)
    default_weight = float(weight_schedule.get("live", 1.0))
    n_pinned = 0
    n_probe_ambiguous = 0
    n_synth_ambiguous = 0

    for q in gt.keys():
        if q in live_queries:
            mode = "live"
            source = "live-feedback"
        elif q in probe_query_modes:
            mode = probe_query_modes[q]
            source = "probe-target-event"
        elif q in synthetic_queries:
            mode = "synthetic-self-anchor"
            source = "synthetic-self-anchor"
        else:
            mode = "live"
            source = "unclassified"
        weight = float(weight_schedule.get(mode, default_weight))
        holdout = mode in holdout_modes

        # Resolve pinned_target_id from the source that classified this query.
        pinned_target_id: str | None = None
        if mode.startswith("probe-"):
            cands = set(probe_pinned.get(q, set()))
            cands |= set(backfill_pinned.get(q, set()))
            if len(cands) == 1:
                pinned_target_id = next(iter(cands))
            elif len(cands) > 1:
                n_probe_ambiguous += 1
                print(
                    f"WARNING: probe pinned ambiguous for query={q!r} "
                    f"(targets={sorted(cands)}); leaving pinned_target_id null",
                    file=sys.stderr,
                )
        elif mode == "synthetic-self-anchor":
            sp = synthetic_pinned.get(q)
            if sp is None and q in synthetic_pinned:
                n_synth_ambiguous += 1
            pinned_target_id = sp
        # live & unclassified deliberately leave pinned null.

        if pinned_target_id is not None:
            n_pinned += 1

        weights_block[q] = {
            "weight": weight,
            "mode": mode,
            "source": source,
            "holdout": bool(holdout),
            "pinned_target_id": pinned_target_id,
        }
        by_source[mode] += 1

    return {
        "metadata": {
            "schema_version": 2,
            "default_weight": default_weight,
            "weight_schedule": dict(weight_schedule),
            "holdout_modes": sorted(holdout_modes),
            "queries_in_sidecar": len(weights_block),
            "queries_with_pinned_target": n_pinned,
            "pinned_boost_default": float(pinned_boost_default),
            "by_source": dict(sorted(by_source.items())),
            "pinned_ambiguous_probe": n_probe_ambiguous,
            "pinned_ambiguous_synthetic": n_synth_ambiguous,
        },
        "weights": weights_block,
    }


def main():
    parser = argparse.ArgumentParser(description="Build GT from feedback events")
    parser.add_argument("--min-memories", type=int, default=3,
                        help="Minimum rated memories per query (default: 3)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: DATA_DIR/tuning_studies/gt_feedback.json)")
    parser.add_argument("--vec-input-overrides-output", type=str, default=None,
                        help="Sidecar path for synthetic (query → context) overrides "
                             "(default: alongside --output as gt_feedback_vec_inputs.json)")
    parser.add_argument("--db", type=str, default=str(DB_PATH),
                        help="Path to memory.db")
    parser.add_argument("--no-synthetic", action="store_true",
                        help="Skip synthetic GT augmentation entirely")
    parser.add_argument("--synthetic-label", type=float, default=1.0,
                        help="Label for synthetic anchors (default: 1.0)")
    parser.add_argument("--no-probe-targets", action="store_true",
                        help="Skip probe_target GT injection (for ablation)")
    parser.add_argument("--no-backfill", action="store_true",
                        help="Skip legacy reason-string backfill (use only "
                             "probe_target events)")
    parser.add_argument("--no-sample-weights", action="store_true",
                        help="Skip sample-weights sidecar emission "
                             "(legacy uniform-weight=1.0 behavior)")
    parser.add_argument("--sample-weights-output", type=str, default=None,
                        help="Sidecar path (default: <output>_sample_weights.json)")
    parser.add_argument("--weight-schedule", type=str, default=None,
                        help="JSON file overriding the default weight schedule")
    parser.add_argument("--weight-live", type=float, default=None,
                        help="Override weight for live (real-feedback) queries")
    parser.add_argument("--weight-natural", type=float, default=None,
                        help="Override weight for probe-natural")
    parser.add_argument("--weight-mild", type=float, default=None,
                        help="Override weight for probe-mild")
    parser.add_argument("--weight-hard", type=float, default=None,
                        help="Override weight for probe-hard (held-out)")
    parser.add_argument("--weight-extra", type=float, default=None,
                        help="Override weight for probe-extra")
    parser.add_argument("--weight-synthetic", type=float, default=None,
                        help="Override weight for synthetic-self-anchor")
    args = parser.parse_args()

    output = Path(args.output) if args.output else DATA_DIR / "tuning_studies" / "gt_feedback.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.vec_input_overrides_output:
        overrides_output = Path(args.vec_input_overrides_output)
    else:
        overrides_output = output.with_name(output.stem + "_vec_inputs.json")

    gt, live_queries = build_gt(Path(args.db), min_memories=args.min_memories)

    vec_input_overrides: dict[str, str] = {}

    # Track query origin across sources for the sample-weights sidecar.
    probe_query_modes: dict[str, str] = {}
    synthetic_queries: set[str] = set()
    # Pinned-target tracking (Phase 3a). Kept per-source so the sidecar
    # resolver can distinguish probe-event pins from backfill-resolved pins
    # — useful for the ambiguity warning (different sources disagreeing on
    # what the pinned target was is a meaningful signal).
    probe_pinned: dict[str, set[str]] = defaultdict(set)
    backfill_pinned: dict[str, set[str]] = defaultdict(set)
    synthetic_pinned: dict[str, str | None] = {}

    # Order matters:
    #   1. Real-feedback GT (build_gt) — highest priority by default.
    #   2. Live probe_target events — pinned-truth, dedups against real feedback.
    #   3. Legacy reason-string + recall_miss backfill — same dedup, plus skips
    #      any (query, full_id) the live pass already set.
    #   4. Synthetic self-anchors — coverage filler, lowest priority.
    if not args.no_probe_targets:
        live_counters, live_modes, live_pinned = add_probe_target_gt(
            Path(args.db), gt, vec_input_overrides,
        )
        for q, m in live_modes.items():
            _merge_probe_mode(probe_query_modes, q, m)
        for q, mids in live_pinned.items():
            probe_pinned[q].update(mids)
        print(f"\nProbe target events (live):")
        print(f"  Live probe_target rows: {live_counters['rows_scanned']}")
        print(f"  Hits added:    {live_counters['hits']}")
        print(f"  Misses added:  {live_counters['misses']}")
        print(f"  Skipped (memory inactive/missing): {live_counters['skipped_missing_memory']}")
        print(f"  Total new (query, mid) pairs: {live_counters['added']}")
        print(f"  Collisions kept (existing>=0.9):  {live_counters['collisions_kept']}")
        print(f"  Collisions overwritten (<0.5):    {live_counters['collisions_overwritten']}")
        print(f"  Collisions max-merged:            {live_counters['collisions_max']}")
        print(f"  Vec inputs written:    {live_counters['vec_inputs_written']}")

    if not args.no_backfill:
        bf_counters, bf_modes, bf_pinned = _backfill_probe_targets_from_legacy(
            Path(args.db), gt, vec_input_overrides,
        )
        for q, m in bf_modes.items():
            _merge_probe_mode(probe_query_modes, q, m)
        for q, mids in bf_pinned.items():
            backfill_pinned[q].update(mids)
        print(f"\nProbe target backfill (legacy):")
        print(f"  Legacy feedback rows scanned: {bf_counters['feedback_rows_scanned']}")
        print(f"  Reason strings matched:       {bf_counters['reason_matched']}")
        print(f"  Distinct (query, target8) pairs: {bf_counters['distinct_pairs_short']}")
        print(f"  Prefixes unresolved (no match):     {bf_counters['prefixes_unresolved']}")
        print(f"  Prefixes ambiguous (collision):     {bf_counters['prefixes_collision']}")
        print(f"  Recall_miss probe rows scanned:     {bf_counters['miss_rows_scanned']}")
        print(f"  Distinct hits:   {bf_counters['hits']}")
        print(f"  Distinct misses: {bf_counters['misses']}")
        print(f"  Total new (query, mid) pairs: {bf_counters['added']}")
        print(f"  Collisions kept (existing>=0.9):  {bf_counters['collisions_kept']}")
        print(f"  Collisions overwritten (<0.5):    {bf_counters['collisions_overwritten']}")
        print(f"  Collisions max-merged:            {bf_counters['collisions_max']}")
        print(f"  Vec inputs recovered from recall_meta: {bf_counters['vec_inputs_recovered']}")
        print(f"  Vec inputs written: {bf_counters['vec_inputs_written']}")

    if not args.no_synthetic:
        _, synthetic_queries_added, synthetic_pinned_added = add_synthetic_self_anchor_gt(
            Path(args.db),
            gt,
            vec_input_overrides,
            label=args.synthetic_label,
        )
        synthetic_queries |= synthetic_queries_added
        for q, mid in synthetic_pinned_added.items():
            # Earlier sources (probe) take precedence over synthetic for
            # pinning, but here we just record what synthetic claims; the
            # sidecar resolver gates on resolved mode.
            if q in synthetic_pinned and synthetic_pinned[q] != mid:
                synthetic_pinned[q] = None
            else:
                synthetic_pinned[q] = mid

    with open(output, "w") as f:
        json.dump(gt, f, indent=2, sort_keys=True)
    print(f"\nGT written to {output} ({len(gt)} total queries)")

    if not args.no_sample_weights:
        # Resolve schedule: defaults < --weight-schedule JSON < per-flag overrides.
        schedule = dict(DEFAULT_WEIGHT_SCHEDULE)
        if args.weight_schedule:
            with open(args.weight_schedule) as f:
                file_schedule = json.load(f)
            schedule.update({k: float(v) for k, v in file_schedule.items()})
        flag_overrides = {
            "live": args.weight_live,
            "probe-natural": args.weight_natural,
            "probe-mild": args.weight_mild,
            "probe-hard": args.weight_hard,
            "probe-extra": args.weight_extra,
            "synthetic-self-anchor": args.weight_synthetic,
        }
        for mode_key, val in flag_overrides.items():
            if val is not None:
                schedule[mode_key] = float(val)

        sidecar = build_sample_weights_sidecar(
            gt,
            live_queries=live_queries,
            probe_query_modes=probe_query_modes,
            synthetic_queries=synthetic_queries,
            weight_schedule=schedule,
            holdout_modes=DEFAULT_HOLDOUT_MODES,
            probe_pinned=dict(probe_pinned),
            backfill_pinned=dict(backfill_pinned),
            synthetic_pinned=synthetic_pinned,
            pinned_boost_default=DEFAULT_PINNED_BOOST,
        )
        sidecar_path = (
            Path(args.sample_weights_output) if args.sample_weights_output
            else output.with_name(output.stem + "_sample_weights.json")
        )
        with open(sidecar_path, "w") as f:
            json.dump(sidecar, f, indent=2, sort_keys=True)

        meta = sidecar["metadata"]
        print(f"\nSample weights sidecar:")
        print(f"  Schema version: {meta['schema_version']}")
        sched_str = ", ".join(f"{k}={v}" for k, v in sorted(meta["weight_schedule"].items()))
        print(f"  Schedule: {{{sched_str}}}")
        print(f"  Holdout modes: {meta['holdout_modes']}")
        print(f"  By source:")
        for mode_key, n in meta["by_source"].items():
            tag = "  [HOLDOUT]" if mode_key in meta["holdout_modes"] else ""
            print(f"    {mode_key:25s} {n:5d}{tag}")
        held_out = sum(1 for v in sidecar["weights"].values() if v["holdout"])
        print(f"  Total queries with explicit weight: {len(sidecar['weights'])}")
        print(f"  Holdout queries: {held_out}")
        print(f"\nPinned-target injection:")
        print(f"  Queries with pinned_target_id: {meta['queries_with_pinned_target']}")
        live_unpinned = sum(
            1 for v in sidecar["weights"].values()
            if v["mode"] == "live" and v["pinned_target_id"] is None
        )
        print(f"  Live queries (always unpinned): {live_unpinned}")
        print(f"  Pinned ambiguous (probe sources disagreed):     {meta['pinned_ambiguous_probe']}")
        print(f"  Pinned ambiguous (synthetic themes_q collision): {meta['pinned_ambiguous_synthetic']}")
        print(f"  Pinned boost (default in metadata): {meta['pinned_boost_default']}")
        print(f"  Sidecar written to {sidecar_path}")

    if vec_input_overrides:
        with open(overrides_output, "w") as f:
            json.dump(vec_input_overrides, f, indent=2, sort_keys=True)
        print(f"Vec input overrides written to {overrides_output} "
              f"({len(vec_input_overrides)} entries)")
        usage = (f"uv run scripts/train_reranker.py --gt {output} "
                 f"--vec-input-overrides {overrides_output}")
    else:
        usage = f"uv run scripts/train_reranker.py --gt {output}"

    if not args.no_sample_weights:
        sidecar_path = (
            Path(args.sample_weights_output) if args.sample_weights_output
            else output.with_name(output.stem + "_sample_weights.json")
        )
        usage += f" --sample-weights {sidecar_path}"

    print(f"\nUse with: {usage}")


if __name__ == "__main__":
    main()
