#!/usr/bin/env python3
"""Bench-side feature layer for the H10 reranker 2x2 (WP2) -- experiments/sleep-bench/bench_features.py

The production reranker computes 31 features (`train_reranker.FEATURE_NAMES`)
and `rerank()` (src/memory/reranker.py) hardcodes that 31-wide matrix. H10 needs
FOUR MORE features that only sleep can create -- `is_summary`,
`is_consolidation_node`, `consolidated_member`, `member_count` -- and then a
24-feature schema column-selected from the 35-wide superset (plan sec D2).

Rather than touch `src/memory/` (live `rerank()` would break on a 35-wide or
24-wide matrix -- see plan sec D5 hazard), this module wraps the trainer's
extraction:

  1. `augment_memory_meta(db, meta)` -- ONE SQL pass over `memories`, adds the
     4 new fields to each real meta entry (sentinel `__*` keys untouched) and
     stashes a full-store lookup so non-active candidates resolve too.
  2. `extract_features_35(...)` -- wraps `train_reranker.extract_features_for_query`
     and appends the 4 columns in canonical order -> (n_candidates, 35).
  3. `select_schema(features, all_names, wanted_names)` -- column-select by name.
  4. `SCHEMA_24` / `ALL_FEATURE_NAMES_35` -- the canonical name lists.

All store access is read-only. Module-level DB/embedding/reranker caches are the
known cross-store trap; callers that point env at a store must reset them the way
`ingest_baseline._point_at_fresh_store` does (reused verbatim below).

Acceptance / self-test:
    uv run experiments/sleep-bench/bench_features.py --selftest
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_reranker import (  # noqa: E402
    FEATURE_NAMES,
    PRODUCTION_PARAMS,
    extract_features_for_query,
)

# ---------------------------------------------------------------------------
# Canonical name lists
# ---------------------------------------------------------------------------

# The 4 sleep-created features, in the ORDER they are appended to the 31
# production features. Do not reorder without updating every consumer.
NEW_FEATURE_NAMES = [
    "is_summary",              # layer == 'summary'
    "is_consolidation_node",   # source == 'consolidate'
    "consolidated_member",     # 'consolidated_member' present in flags JSON
    "member_count",            # len(generated_from JSON list); 0 if null/empty
]

# The full 35-name superset: 31 production features (in FEATURE_NAMES order)
# followed by the 4 new columns.
ALL_FEATURE_NAMES_35 = list(FEATURE_NAMES) + list(NEW_FEATURE_NAMES)

# The 24-feature schema shared by R_pre and R_post (plan sec D2).
# 20 kept production features + the 4 new. `age_days` is EXCLUDED (Alexis
# 2026-07-10: sleep rows stamp created_at=now while details are backdated, so
# age_days becomes a ~1,000-day cliff that duplicates is_summary on an artifact
# scale). Also dropped: fb_last, fb_mean, fb_count, fb_time_weighted,
# session_recency, burstiness, hebbian_pmi (dead on any benchmark store);
# category, confidence, decay_rate (constant on this corpus).
SCHEMA_24 = [
    # 20 kept production features
    "fts_rank",
    "vec_rank",
    "theme_rank",
    "ppr_score",
    "fts_bm25",
    "vec_dist",
    "theme_overlap",
    "priority",
    "token_count",
    "edge_count",
    "theme_count",
    "query_coverage",
    "proximity",
    "query_idf_var",
    "betweenness",
    "diversity_score",
    "query_length",
    "candidate_pool_size",
    "fts_bm25_norm",
    "vec_dist_norm",
    # 4 new sleep-created features
    "is_summary",
    "is_consolidation_node",
    "consolidated_member",
    "member_count",
]

# Sentinel key under which the full-store new-feature lookup is stashed on the
# meta dict (starts with '__' so it never masquerades as a memory id and is
# skipped by every meta consumer).
_LUT_KEY = "__bench_new_features_lut__"


# ---------------------------------------------------------------------------
# (1) meta augmentation
# ---------------------------------------------------------------------------


def _new_feature_lut(db) -> dict:
    """One SQL pass over `memories` -> {memory_id: (is_summary,
    is_consolidation_node, consolidated_member, member_count)} for EVERY row
    (all statuses), so a non-active candidate still resolves.
    """
    lut: dict[str, tuple[float, float, float, float]] = {}
    rows = db.execute(
        "SELECT id, layer, source, flags, generated_from FROM memories"
    ).fetchall()
    for r in rows:
        is_summary = 1.0 if r["layer"] == "summary" else 0.0
        is_node = 1.0 if r["source"] == "consolidate" else 0.0
        flags = r["flags"] or ""
        member = 1.0 if "consolidated_member" in flags else 0.0
        member_count = 0.0
        gf = r["generated_from"]
        if gf:
            try:
                parsed = json.loads(gf)
                if isinstance(parsed, list):
                    member_count = float(len(parsed))
            except (json.JSONDecodeError, TypeError):
                member_count = 0.0
        lut[r["id"]] = (is_summary, is_node, member, member_count)
    return lut


def augment_memory_meta(db_or_meta, db=None):
    """Augment a reranker meta dict with the 4 sleep-created fields.

    Two call shapes:
      augment_memory_meta(db)            -> loads meta via _load_memory_meta,
                                            augments, returns it.
      augment_memory_meta(meta, db=db)   -> augments an existing meta dict in
                                            place, returns it.

    In both cases the 4 fields are written onto every real meta entry (keys
    starting with '__' are left untouched) and a full-store lookup is stashed
    under `_LUT_KEY` so `extract_features_35` can resolve candidates that are
    absent from the active-only meta.
    """
    from memory.reranker import _load_memory_meta

    if db is None:
        # single-arg form: first positional IS the db connection
        db = db_or_meta
        meta = _load_memory_meta(db)
    else:
        meta = db_or_meta

    lut = _new_feature_lut(db)
    zero = (0.0, 0.0, 0.0, 0.0)
    for mid, m in meta.items():
        if isinstance(mid, str) and mid.startswith("__"):
            continue  # sentinel key -- do not disturb
        if not isinstance(m, dict):
            continue
        vals = lut.get(mid, zero)
        m["is_summary"] = vals[0]
        m["is_consolidation_node"] = vals[1]
        m["consolidated_member"] = vals[2]
        m["member_count"] = vals[3]
    meta[_LUT_KEY] = lut
    return meta


# ---------------------------------------------------------------------------
# (2) 35-feature extraction
# ---------------------------------------------------------------------------


def extract_features_35(
    qtext,
    params,
    search_data,
    feedback_raw,
    hebb_data,
    memory_meta,
    ground_truth_for_query,
):
    """Wrap `extract_features_for_query` -> (n_candidates, 35) matrix.

    Columns 0..30 are the production features in FEATURE_NAMES order; columns
    31..34 are the 4 new features in NEW_FEATURE_NAMES order. `memory_meta` must
    have been passed through `augment_memory_meta` first (raises otherwise).

    Returns (features_35, labels, memory_ids) -- same tuple shape as the wrapped
    function.
    """
    lut = memory_meta.get(_LUT_KEY)
    if lut is None:
        raise ValueError(
            "memory_meta was not augmented -- call augment_memory_meta(db, meta) "
            "before extract_features_35()"
        )

    feats31, labels, mids = extract_features_for_query(
        qtext, params, search_data, feedback_raw, hebb_data,
        memory_meta, ground_truth_for_query,
    )

    new_cols = np.zeros((len(mids), 4), dtype=np.float32)
    zero = (0.0, 0.0, 0.0, 0.0)
    for i, mid in enumerate(mids):
        vals = lut.get(mid, zero)
        new_cols[i, 0] = vals[0]
        new_cols[i, 1] = vals[1]
        new_cols[i, 2] = vals[2]
        new_cols[i, 3] = vals[3]

    feats35 = np.hstack([feats31.astype(np.float32), new_cols])
    return feats35, labels, mids


# ---------------------------------------------------------------------------
# (3) schema column-select
# ---------------------------------------------------------------------------


def select_schema(features, all_names, wanted_names):
    """Column-select `features` down to `wanted_names`.

    Returns (subset_matrix, wanted_names) with column order == wanted_names.
    Raises ValueError on any name not present in `all_names`.
    """
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    idx = []
    for w in wanted_names:
        if w not in name_to_idx:
            raise ValueError(f"unknown feature name: {w!r}")
        idx.append(name_to_idx[w])
    features = np.asarray(features)
    return features[:, idx], list(wanted_names)


# ---------------------------------------------------------------------------
# Self-test / acceptance
# ---------------------------------------------------------------------------


def _extract_all_on_store(store, gt_path, key_file):
    """Point env at `store`, (re)bind tune_gt's store-scoped module globals,
    build search_data via load_tuning_data, and return the vertically stacked
    35-feature matrix over all GT queries.

    tune_gt binds DATA_DIR / MEMORY_DB at import time, so switching stores in
    one process requires reloading it AFTER _point_at_fresh_store mutates the
    memory.constants / memory.db attributes (the module-level cache trap).
    """
    import importlib
    import os

    from ingest_baseline import _point_at_fresh_store
    from memory.constants import PPR_DAMPING

    _point_at_fresh_store(store)
    os.environ["OPENAI_API_KEY"] = key_file.read_text().strip()

    import tune_gt
    importlib.reload(tune_gt)  # re-bind DATA_DIR / MEMORY_DB from the mutated modules

    from memory.db import get_db

    db = get_db()
    meta = augment_memory_meta(db)

    full_data, ground_truth, _ = tune_gt.load_tuning_data(
        str(gt_path), ppr_dampings=[PPR_DAMPING]
    )
    params = dict(PRODUCTION_PARAMS)

    per_query = []
    for qtext in full_data["gt_queries"]:
        feats, _labels, mids = extract_features_35(
            qtext, params, full_data["search_data"],
            full_data["feedback_raw"], full_data["hebb_data"],
            meta, ground_truth.get(qtext, {}),
        )
        assert feats.shape[1] == 35, f"expected 35 cols, got {feats.shape[1]}"
        per_query.append((feats, mids))
    stacked = np.vstack([f for f, _ in per_query])
    return stacked, per_query


def _selftest():
    """Acceptance checks from WP2. Extracts features for 5 GT queries on both
    the frozen and c2 stores and audits the new columns + per-column variance.
    """
    import os

    LOCOMO = Path(r"C:\Users\Alexis\.claude\repos\locomo\data\locomo10.json")
    KEY_FILE = Path(r"C:\Users\Alexis\.claude\data\openai_api_key")
    STORES = {
        "frozen": Path(r"C:\Users\Alexis\.somnigraph\benchmark\sleepbench-full\conv0-frozen"),
        "c2": Path(r"C:\Users\Alexis\.somnigraph\benchmark\sleepbench-full\conv0-c2"),
    }
    SCRATCH = Path(
        r"C:\Users\Alexis\AppData\Local\Temp\claude"
        r"\C--Users-Alexis-repos-somnigraph"
        r"\d2cafcdd-0ff6-4bb5-a74a-acd677d71759\scratchpad"
    )
    SCRATCH.mkdir(parents=True, exist_ok=True)

    os.environ["OPENAI_API_KEY"] = KEY_FILE.read_text().strip()

    import gt as gtmod

    # Build a tiny GT (5 conv-1 queries) once; shape {qtext: {mid: 1.0}}.
    # dia_map resolution is store-agnostic (same conv-1 ingest in both stores),
    # so we build against frozen and reuse the text for both.
    qset = gtmod.build_query_set(LOCOMO, STORES["frozen"], 1)[:5]
    tiny_gt = {q["question"]: {mid: 1.0 for mid in q["evidence_ids"]} for q in qset}
    gt_path = SCRATCH / "bench_features_selftest_gt.json"
    gt_path.write_text(json.dumps(tiny_gt))
    print(f"[selftest] built tiny GT: {len(tiny_gt)} queries -> {gt_path}")

    results = {}  # store -> features_35 stacked
    for label, store in STORES.items():
        print(f"\n[selftest] === store: {label} ({store}) ===")
        stacked, per_query = _extract_all_on_store(store, gt_path, KEY_FILE)
        results[label] = stacked
        print(f"[selftest] {label}: {stacked.shape[0]} candidate rows across "
              f"{len(per_query)} queries, {stacked.shape[1]} cols")

        # New-column audit
        new_idx = [ALL_FEATURE_NAMES_35.index(n) for n in NEW_FEATURE_NAMES]
        new_block = stacked[:, new_idx]
        col_sums = new_block.sum(axis=0)
        print(f"[selftest] {label}: new-col sums {dict(zip(NEW_FEATURE_NAMES, col_sums.tolist()))}")
        if label == "frozen":
            assert np.all(new_block == 0.0), "frozen store must have all-zero new columns"
            print("[selftest] PASS: frozen new columns all zero")
        else:
            assert new_block.sum() > 0.0, "c2 store must have non-trivial new columns"
            n_summary = int((new_block[:, 0] > 0).sum())
            n_node = int((new_block[:, 1] > 0).sum())
            n_member = int((new_block[:, 2] > 0).sum())
            print(f"[selftest] c2 candidate rows flagged: summary={n_summary} "
                  f"node={n_node} member={n_member}")
            assert new_block[:, 0].sum() > 0, "c2 must surface >=1 summary candidate"
            print("[selftest] PASS: c2 new columns non-trivial")

    # Determinism: re-run extraction on c2 and compare.
    # NOTE: raw feature `age_days` is computed from datetime.now() at each
    # _load_memory_meta call, so it drifts by wall-clock microseconds between
    # runs (an artifact -- exactly why the plan DROPS it from SCHEMA_24). The
    # meaningful determinism check is on the model-input schema (SCHEMA_24),
    # which excludes age_days. We report both.
    print("\n[selftest] === determinism re-run (c2) ===")
    stacked2, _ = _extract_all_on_store(STORES["c2"], gt_path, KEY_FILE)

    raw_identical = np.array_equal(
        np.nan_to_num(results["c2"], nan=-999.0),
        np.nan_to_num(stacked2, nan=-999.0),
    )
    if not raw_identical:
        d = np.abs(np.nan_to_num(results["c2"], nan=-999.0) - np.nan_to_num(stacked2, nan=-999.0))
        drift_cols = [
            (ALL_FEATURE_NAMES_35[c], float(d[:, c].max()))
            for c in np.where(d.sum(axis=0) > 0)[0]
        ]
        print(f"[selftest] raw-35 differs only in (wall-clock) cols: {drift_cols}")

    sub1, _ = select_schema(results["c2"], ALL_FEATURE_NAMES_35, SCHEMA_24)
    sub2, _ = select_schema(stacked2, ALL_FEATURE_NAMES_35, SCHEMA_24)
    identical = np.array_equal(
        np.nan_to_num(sub1, nan=-999.0), np.nan_to_num(sub2, nan=-999.0)
    )
    print(f"[selftest] determinism (SCHEMA_24 identical across reruns): {identical}")
    assert identical, "non-deterministic SCHEMA_24 feature matrix on rerun"
    print("[selftest] PASS: deterministic (model-input schema; age_days drift is "
          "the dropped wall-clock artifact)")

    # Column-order check + schema select
    print("\n[selftest] === schema select ===")
    sub, names = select_schema(results["c2"], ALL_FEATURE_NAMES_35, SCHEMA_24)
    assert sub.shape[1] == 24, f"expected 24 cols, got {sub.shape[1]}"
    assert names == SCHEMA_24
    print(f"[selftest] select_schema -> {sub.shape[1]} cols, order matches SCHEMA_24")
    try:
        select_schema(results["c2"], ALL_FEATURE_NAMES_35, ["not_a_feature"])
        raise AssertionError("select_schema should have raised on unknown name")
    except ValueError as e:
        print(f"[selftest] PASS: select_schema raises on unknown name ({e})")

    # Per-column variance audit on both stores (empirically confirms drop list)
    print("\n[selftest] === per-column variance audit (35 features) ===")
    predicted_dead = {
        "fb_last", "fb_mean", "fb_count", "fb_time_weighted", "session_recency",
        "burstiness", "hebbian_pmi", "category", "confidence", "decay_rate",
        "age_days",
    }
    header = f"{'feature':24s} {'frozen_var':>14s} {'c2_var':>14s}  note"
    print(header)
    print("-" * len(header))
    surprises = []
    for i, name in enumerate(ALL_FEATURE_NAMES_35):
        fv = float(np.nanvar(results["frozen"][:, i]))
        cv = float(np.nanvar(results["c2"][:, i]))
        note = ""
        if name in predicted_dead:
            note = "PREDICTED-DEAD"
            # "dead" = no variance on either store (or all-NaN)
            fv_all_nan = np.all(np.isnan(results["frozen"][:, i]))
            cv_all_nan = np.all(np.isnan(results["c2"][:, i]))
            alive = (fv > 1e-9 and not fv_all_nan) or (cv > 1e-9 and not cv_all_nan)
            if alive:
                note += " <-- ALIVE! (unexpected variance)"
                surprises.append((name, fv, cv))
        print(f"{name:24s} {fv:14.6g} {cv:14.6g}  {note}")

    if surprises:
        print("\n" + "!" * 72)
        print("SURPRISE: predicted-dead features show real variance:")
        for name, fv, cv in surprises:
            print(f"   {name}: frozen_var={fv:.6g} c2_var={cv:.6g}")
        print("!" * 72)
    else:
        print("\n[selftest] all predicted-dead features confirmed dead "
              "(zero variance / all-NaN on both stores)")

    print("\n[selftest] ALL ACCEPTANCE CHECKS PASSED")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
        print("Run with --selftest for acceptance checks.")
