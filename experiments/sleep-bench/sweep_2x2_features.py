#!/usr/bin/env python3
"""WP6 Stage-A random-subset feature sweep -- experiments/sleep-bench/sweep_2x2_features.py

Trains many R_post CV models over random feature masks to build the raw material
for the WP6 importance-instability structure map (`analyze_wp6.py`, Stage B). The
design is pre-registered in `wp6-prereg.md`; read it before changing anything here.

WHAT THIS DOES (and does NOT do)
--------------------------------
- Loads the training data ONCE (the expensive step: `load_tuning_data` embeds the
  ~1,757 train queries and re-runs vec/FTS/theme search on the c2 store, then
  `extract_features_35` builds the per-candidate feature rows). The full 24-column
  SCHEMA_24 matrix + query groups + ranking scaffolding are built a single time and
  reused for every mask -- NO per-mask shell-out to train_2x2.py.
- Computes the GroupKFold(5) fold assignments ONCE (GroupKFold is deterministic
  given the group vector and n_splits, so this is identical to what train_2x2's
  per-run CV would produce -- that identity is what makes the all-on anchor
  reproduce WP4's R_post numbers exactly).
- Per mask: column-selects SCHEMA_24 down to (12 channel-core always-on + the
  mask's toggleables, in SCHEMA_24 order), runs the SAME CV recipe as WP4
  (byte-identical LGB_PARAMS, early_stopping(50), seed 42, same folds), records
  per-fold + mean CV metrics and per-feature gain importances. NO final-model fit
  (Stage A is CV-only; importances are averaged across the 5 fold models).

The data-loading + fold-CV logic is IMPORTED from `train_2x2.py` (its
`_point_and_reload`, `_extract_matrix`, `_maybe_truncate_gt`, `_rank_and_score`,
`LGB_PARAMS`, folds recipe). train_2x2's CLI/outputs are untouched -- this module
only reuses its already-module-level functions, so its frozen smoke contract holds.

MASKS
-----
Channel core (12, always-on): fts_rank, vec_rank, theme_rank, fts_bm25, vec_dist,
  fts_bm25_norm, vec_dist_norm, query_idf_var, query_length, candidate_pool_size,
  query_coverage, proximity.
Toggleable (12, included independently at p=0.5): priority, token_count,
  theme_count, member_count, is_summary, is_consolidation_node, consolidated_member,
  edge_count, betweenness, diversity_score, ppr_score, theme_overlap.

Sequence (reproducible from --seed): anchor all-on, anchor core-only, then
--n-masks random masks (dedup against anchors + each other, redraw on collision),
then any `--extra-mask name=feat,feat` Stage-C masks. mask_id = sha1 of the sorted
toggleable-name tuple (stable across runs/machines, unlike Python hash()).

RESUME
------
Output JSONL is append-mode. On start the existing file is read and every mask_id
already present is skipped. Because the mask sequence is a pure function of --seed,
--n-masks and the --extra-mask flags, an interrupted overnight run resumes over the
identical sequence and each mask lands exactly once.

Usage:
    # acceptance check 1 -- just the two anchors, reproduces WP4 R_post CV:
    uv run --extra tuning experiments/sleep-bench/sweep_2x2_features.py --n-masks 0

    # full Stage-A sweep (Alexis terminal job):
    uv run --extra tuning experiments/sleep-bench/sweep_2x2_features.py --n-masks 128

    # smoke:
    uv run --extra tuning experiments/sleep-bench/sweep_2x2_features.py \
        --max-queries 150 --n-masks 6
"""

import argparse
import gc
import hashlib
import json
import pickle
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import lightgbm as lgb  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from bench_features import SCHEMA_24  # noqa: E402

# Reuse train_2x2's proven data path + fold recipe verbatim (do NOT reimplement).
import train_2x2 as t2  # noqa: E402

DEFAULT_STORE = Path(
    r"C:\Users\Alexis\.somnigraph\benchmark\sleepbench-full\conv0-c2"
)
DEFAULT_GT = t2.DEFAULT_OUT_DIR / "gt_2x2.json"
DEFAULT_OUT = t2.DEFAULT_OUT_DIR / "wp6_sweep.jsonl"
DEFAULT_KEY_FILE = t2.DEFAULT_KEY_FILE
SCRATCH = Path(
    r"C:\Users\Alexis\AppData\Local\Temp\claude"
    r"\C--Users-Alexis-repos-somnigraph"
    r"\d2cafcdd-0ff6-4bb5-a74a-acd677d71759\scratchpad"
)

# ---------------------------------------------------------------------------
# Feature split (must match wp6-prereg.md exactly)
# ---------------------------------------------------------------------------

CORE = [
    "fts_rank", "vec_rank", "theme_rank", "fts_bm25", "vec_dist",
    "fts_bm25_norm", "vec_dist_norm", "query_idf_var", "query_length",
    "candidate_pool_size", "query_coverage", "proximity",
]
TOGGLEABLE = [
    "priority", "token_count", "theme_count", "member_count", "is_summary",
    "is_consolidation_node", "consolidated_member", "edge_count", "betweenness",
    "diversity_score", "ppr_score", "theme_overlap",
]

# Guardrails: the split must partition SCHEMA_24 (no overlap, full cover).
assert set(CORE) | set(TOGGLEABLE) == set(SCHEMA_24), "CORE+TOGGLEABLE != SCHEMA_24"
assert not (set(CORE) & set(TOGGLEABLE)), "CORE and TOGGLEABLE overlap"
assert len(CORE) == 12 and len(TOGGLEABLE) == 12


# ---------------------------------------------------------------------------
# Mask identity + column selection
# ---------------------------------------------------------------------------


def mask_id(toggleables_on) -> str:
    """Stable 12-hex id from the sorted toggleable-name tuple. hashlib (not the
    salted builtin hash()) so ids are identical across runs and machines."""
    key = "|".join(sorted(toggleables_on))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def feature_list_for(toggleables_on) -> list[str]:
    """Core (all 12) + the mask's toggleables, in SCHEMA_24 canonical order.
    Canonical order is what lets the all-on mask reproduce WP4 exactly (LightGBM's
    colsample_bytree feature sampling is column-order dependent)."""
    on = set(toggleables_on)
    return [f for f in SCHEMA_24 if f in CORE or f in on]


# ---------------------------------------------------------------------------
# Mask sequence generation (reproducible from seed; resume-stable)
# ---------------------------------------------------------------------------


def build_mask_sequence(n_masks: int, seed: int, extra_masks, full_masks=None) -> list[dict]:
    """Deterministic ordered list of masks. Each item:
        {label, toggleables_on (sorted list), mask_id}
    Order: all_on anchor, core_only anchor, n_masks random masks (p=0.5 per
    toggleable, dedup w/ redraw), then Stage-C extra masks (dedup against all
    prior). The RNG stream depends ONLY on `seed` and `n_masks`, so resume sees
    the identical sequence."""
    seq: list[dict] = []
    seen: set[str] = set()

    def add(label, toggleables_on):
        mid = mask_id(toggleables_on)
        if mid in seen:
            return False
        seen.add(mid)
        seq.append({"label": label, "toggleables_on": sorted(toggleables_on), "mask_id": mid})
        return True

    # Anchors first (anchor all-on is acceptance check 1; must come first).
    add("all_on", list(TOGGLEABLE))
    add("core_only", [])

    # Random masks. Fixed toggleable iteration order + one rng.random() draw per
    # toggleable per candidate mask -> fully reproducible. Redraw on collision.
    rng = random.Random(seed)
    drawn = 0
    guard = 0
    max_guard = max(1000, n_masks * 1000)
    while drawn < n_masks and guard < max_guard:
        guard += 1
        cand = [t for t in TOGGLEABLE if rng.random() < 0.5]
        if add(f"random_{drawn:04d}", cand):
            drawn += 1
    if drawn < n_masks:
        # Only possible if n_masks approaches 2^12 unique masks; anchors excluded.
        print(f"[sweep] WARNING: only {drawn}/{n_masks} unique random masks found "
              f"(exhausted the space after {guard} draws)")

    # Stage-C extra masks (appended after the random block; dedup against all).
    for name, toggs in extra_masks:
        add(name, toggs)

    # Full masks (any SCHEMA_24 subset + noiseK columns; core NOT implied).
    # Distinct id-space ("full|" prefix) so a full mask can never collide with
    # a core+toggle mask; records carry `full_features` so analyze_wp6 can
    # exclude them from the toggleable marginal statistics.
    for name, feats in (full_masks or []):
        ordered = canonical_full_order(feats)
        mid = hashlib.sha1(("full|" + "|".join(ordered)).encode("utf-8")).hexdigest()[:12]
        if mid in seen:
            continue
        seen.add(mid)
        seq.append({
            "label": name,
            "toggleables_on": sorted(set(ordered) & set(TOGGLEABLE)),
            "full_features": ordered,
            "mask_id": mid,
        })

    return seq


# ---------------------------------------------------------------------------
# In-sweep full-scale anchor gate (runs on mask #1 of a real terminal run)
# ---------------------------------------------------------------------------

# WP4 R_post CV reference (full data, SCHEMA_24 all-on). Fallback if the
# R_post__cv.json artifact is absent; the artifact is read in preference.
_ANCHOR_EXPECT = {
    "ndcg_5k": [0.5432089479421287, 0.5762550014918678, 0.5526494143132069,
                0.5398499359207325, 0.563230055432243],
    "best_iter": [134, 131, 55, 60, 147],
    "val_q": [351, 352, 351, 351, 352],
}


def _load_anchor_expect() -> dict:
    ref = t2.DEFAULT_OUT_DIR / "R_post__cv.json"
    try:
        cv = json.loads(ref.read_text(encoding="utf-8"))
        fm = cv["fold_metrics"]
        return {
            "ndcg_5k": [f["ndcg_5k"] for f in fm],
            "best_iter": [f["best_iter"] for f in fm],
            "val_q": [f["n_val_queries"] for f in fm],
            "_source": str(ref),
        }
    except (OSError, KeyError, json.JSONDecodeError):
        return {**_ANCHOR_EXPECT, "_source": "hardcoded fallback"}


def anchor_gate(per_fold, ignore: bool) -> None:
    """Compare the all-on anchor's per-fold CV against WP4 R_post; loud PASS/FAIL.
    Aborts on FAIL unless `ignore`. Proves identical folds + recipe at full scale."""
    exp = _load_anchor_expect()
    got_ndcg = [f["ndcg_5k"] for f in per_fold]
    got_bi = [f["best_iter"] for f in per_fold]
    got_vq = [f["val_q"] for f in per_fold]
    ndcg_ok = len(got_ndcg) == len(exp["ndcg_5k"]) and all(
        abs(a - b) < 5e-5 for a, b in zip(got_ndcg, exp["ndcg_5k"]))
    bi_ok = got_bi == exp["best_iter"]
    vq_ok = got_vq == exp["val_q"]
    ok = ndcg_ok and bi_ok and vq_ok
    print("=" * 70)
    print(f"[sweep] ANCHOR GATE (all-on vs WP4 R_post; ref={exp['_source']})")
    for i in range(len(got_ndcg)):
        e_nd = exp["ndcg_5k"][i] if i < len(exp["ndcg_5k"]) else None
        print(f"  fold{i}: ndcg={got_ndcg[i]:.6f} (exp {e_nd})  "
              f"best_iter={got_bi[i]} (exp {exp['best_iter'][i]})  "
              f"val_q={got_vq[i]} (exp {exp['val_q'][i]})")
    print(f"[sweep] ANCHOR GATE: {'PASS' if ok else 'FAIL'} "
          f"(ndcg={ndcg_ok} best_iter={bi_ok} val_q={vq_ok})")
    print("=" * 70)
    if not ok and not ignore:
        raise SystemExit(
            "[sweep] anchor mismatch -- folds/recipe drifted from WP4 R_post. "
            "Fix before trusting the sweep, or pass --ignore-anchor-mismatch."
        )


def parse_extra_mask(spec: str) -> tuple[str, list[str]]:
    """`name=feat1,feat2,...` -> (name, [feat,...]). Feats must be toggleables."""
    if "=" not in spec:
        raise ValueError(f"--extra-mask must be name=feat,feat -- got {spec!r}")
    name, csv = spec.split("=", 1)
    name = name.strip()
    feats = [f.strip() for f in csv.split(",") if f.strip()]
    unknown = [f for f in feats if f not in TOGGLEABLE]
    if unknown:
        raise ValueError(f"--extra-mask {name}: names not toggleable: {unknown}")
    return name, feats


NOISE_RE = re.compile(r"^noise(\d+)$")


def parse_full_mask(spec: str) -> tuple[str, list[str]]:
    """`name=feat1,feat2,...` -> (name, [feat,...]). Feats may be ANY SCHEMA_24
    column (core included) and/or synthetic `noiseK` columns (seeded N(0,1),
    the capacity-artifact control). Bypasses the core-always-on composition."""
    if "=" not in spec:
        raise ValueError(f"--full-mask must be name=feat,feat -- got {spec!r}")
    name, csv = spec.split("=", 1)
    name = name.strip()
    feats = [f.strip() for f in csv.split(",") if f.strip()]
    unknown = [f for f in feats if f not in SCHEMA_24 and not NOISE_RE.match(f)]
    if unknown:
        raise ValueError(f"--full-mask {name}: not in SCHEMA_24 nor noiseK: {unknown}")
    if not feats:
        raise ValueError(f"--full-mask {name}: empty feature list")
    return name, feats


def canonical_full_order(feats) -> list[str]:
    """SCHEMA_24 canonical order first, then noiseK ascending. Column order is
    load-bearing for reproducibility (colsample_bytree is order-dependent)."""
    fs = set(feats)
    reals = [f for f in SCHEMA_24 if f in fs]
    noises = sorted((f for f in fs if NOISE_RE.match(f)),
                    key=lambda s: int(NOISE_RE.match(s).group(1)))
    return reals + noises


def make_noise_columns(n_rows: int, names: list[str]) -> dict[str, "np.ndarray"]:
    """Deterministic per-name standard-normal columns. Seed = 9999 + K so
    noise3 is the same column in every run regardless of which masks request it."""
    cols = {}
    for name in names:
        k = int(NOISE_RE.match(name).group(1))
        # float32: column_stack must not upcast the real columns to float64,
        # which would silently break anchor byte-parity on noise-bearing runs.
        cols[name] = (np.random.default_rng(9999 + k)
                      .standard_normal(n_rows).astype(np.float32))
    return cols


# ---------------------------------------------------------------------------
# CV for one mask (CV-only; mirrors train_2x2.train_and_cv's fold loop exactly,
# minus the final all-rows fit -- Stage A does no final-model fits)
# ---------------------------------------------------------------------------


def cv_one_mask(tune_gt, full, folds, feats, col_idx):
    """Run GroupKFold CV for the masked feature set. `full` is train_2x2's
    _extract_matrix dict; `folds` is the precomputed list of (train_idx, val_idx);
    `col_idx` selects the mask columns out of the full SCHEMA_24 matrix.

    Returns (per_fold list, mean dict, importances_gain dict)."""
    X = full["X"][:, col_idx]
    y = full["y"]
    groups = full["groups"]

    per_fold = []
    gain_accum = np.zeros(len(feats), dtype=np.float64)
    for fold_i, (train_idx, val_idx) in enumerate(folds):
        model = lgb.LGBMRegressor(**t2.LGB_PARAMS)
        model.fit(
            X[train_idx], y[train_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        preds = np.clip(model.predict(X[val_idx]), 0, 1)
        rmse = float(np.sqrt(np.mean((preds - y[val_idx]) ** 2)))
        rank_m = t2._rank_and_score(
            tune_gt, preds, val_idx, groups, full["memory_ids"],
            full["queries"], full["ground_truth"], full["token_map"],
        )
        gain_accum += model.booster_.feature_importance(importance_type="gain")
        per_fold.append({
            "ndcg_5k": rank_m["ndcg_5k"],
            "recall_10": rank_m["recall_10"],
            "mrr": rank_m["mrr"],
            "graded_recall_5k": rank_m["graded_recall_5k"],
            "rmse": rmse,
            "val_q": len(set(groups[val_idx].tolist())),
            "best_iter": int(model.best_iteration_ or t2.LGB_PARAMS["n_estimators"]),
        })
        # Free the booster's C++ memory before the next fold. 130 masks x 5 folds
        # = 650 booster/dataset constructions in one process; without this the
        # native heap creeps until LGBM_DatasetCreateFromMat throws bad_alloc
        # (observed at mask 119/130 on the 8 GB-pagefile box, 2026-07-10).
        del model
        gc.collect()

    # Drop this mask's ~100 MB column-slice copy of the full matrix eagerly
    # rather than waiting for the next mask's reassignment.
    del X
    gc.collect()

    keys = ["ndcg_5k", "recall_10", "mrr", "graded_recall_5k", "rmse", "best_iter"]
    mean = {k: float(np.mean([f[k] for f in per_fold])) for k in keys}
    # CV-only importances: mean gain across the fold models (no final-model fit).
    importances = {feats[i]: float(gain_accum[i] / len(folds)) for i in range(len(feats))}
    return per_fold, mean, importances


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gt", default=str(DEFAULT_GT))
    ap.add_argument("--store", default=str(DEFAULT_STORE))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--n-masks", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1234,
                    help="RNG seed for the random-mask stream (default 1234)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-queries", type=int, default=None,
                    help="smoke only: cap training queries (forwarded to the loader)")
    ap.add_argument("--key-file", default=str(DEFAULT_KEY_FILE))
    ap.add_argument("--extra-mask", action="append", default=[], metavar="name=feat,feat",
                    help="Stage-C targeted mask; repeatable. Feats are toggleables.")
    ap.add_argument("--full-mask", action="append", default=[], metavar="name=feat,feat",
                    help="exact feature set: any SCHEMA_24 column (core included) "
                         "and/or noiseK synthetic columns; repeatable.")
    ap.add_argument("--ignore-anchor-mismatch", action="store_true",
                    help="do not abort if the full-scale all-on anchor gate FAILs")
    ap.add_argument("--no-matrix-cache", action="store_true",
                    help="skip the extracted-matrix disk cache (force a fresh "
                         "load_tuning_data + extract)")
    args = ap.parse_args()

    store = Path(args.store).expanduser().resolve()
    gt_path = Path(args.gt).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    key_file = Path(args.key_file).expanduser()

    # Never touch a store outside the sleepbench-full tree.
    sentinel = Path(r"C:\Users\Alexis\.somnigraph\benchmark\sleepbench-full").resolve()
    if sentinel not in store.parents and store != sentinel:
        raise SystemExit(f"[sweep] refusing store outside {sentinel}: {store}")

    extra = [parse_extra_mask(s) for s in args.extra_mask]
    fulls = [parse_full_mask(s) for s in args.full_mask]
    seq = build_mask_sequence(args.n_masks, args.seed, extra, fulls)
    print(f"[sweep] mask sequence: {len(seq)} masks "
          f"(2 anchors + {args.n_masks} random + {len(extra)} extra + "
          f"{len(fulls)} full, post-dedup {len(seq)})")

    # Resume: read any already-completed mask_ids from the output file.
    done: set[str] = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["mask_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"[sweep] resume: {len(done)} masks already present in {out_path.name}")

    todo = [m for m in seq if m["mask_id"] not in done]
    print(f"[sweep] {len(todo)} masks to run, {len(seq) - len(todo)} skipped")
    if not todo:
        print("[sweep] nothing to do.")
        return

    # ---- Load data ONCE (disk-cached across invocations) ----
    gt_used = t2._maybe_truncate_gt(gt_path, args.max_queries, SCRATCH)
    print(f"[sweep] store={store}")
    tune_gt = t2._point_and_reload(store, key_file)

    # The extracted matrix depends only on (store, gt bytes, max_queries, schema)
    # -- all stores here are frozen snapshots, so a disk cache turns the ~60-90 s
    # search+extract startup into a ~2 s pickle load for every batch after the
    # first. Keyed so a different store/gt/truncation can never collide.
    cache_key = hashlib.sha1((
        str(store) + "|" + hashlib.sha1(gt_used.read_bytes()).hexdigest()
        + f"|mq={args.max_queries}|" + "|".join(SCHEMA_24)
    ).encode("utf-8")).hexdigest()[:16]
    cache_file = out_path.parent / f"matrix_cache_{cache_key}.pkl"
    if cache_file.exists() and not args.no_matrix_cache:
        print(f"[sweep] matrix cache HIT: {cache_file.name}")
        with cache_file.open("rb") as cf:
            full = pickle.load(cf)
    else:
        print(f"[sweep] loading data once (this is the expensive step)...")
        full = t2._extract_matrix(tune_gt, gt_used, list(SCHEMA_24))
        if not args.no_matrix_cache:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with cache_file.open("wb") as cf:
                pickle.dump(full, cf, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[sweep] matrix cache saved: {cache_file.name} "
                  f"({cache_file.stat().st_size / 1e6:.0f} MB)")
    n_q = len(full["queries"])
    print(f"[sweep] loaded: {len(full['y'])} rows, {full['X'].shape[1]} cols, {n_q} queries")

    # ---- Fold assignments ONCE (deterministic; == train_2x2's per-run CV) ----
    n_folds_eff = min(args.folds, n_q)
    if n_folds_eff < 2:
        raise SystemExit(f"[sweep] need >=2 query-groups, have {n_q}")
    gkf = GroupKFold(n_splits=n_folds_eff)
    folds = list(gkf.split(full["X"], full["y"], full["groups"]))
    print(f"[sweep] {n_folds_eff} folds fixed once for all masks")

    schema_idx = {n: i for i, n in enumerate(SCHEMA_24)}

    # Synthetic noise columns (capacity-artifact control): appended to the
    # matrix once, deterministic per name regardless of run or mask set.
    noise_names = sorted(
        {f for m in seq for f in m.get("full_features", []) if NOISE_RE.match(f)},
        key=lambda s: int(NOISE_RE.match(s).group(1)),
    )
    if noise_names:
        noise_cols = make_noise_columns(len(full["y"]), noise_names)
        full["X"] = np.column_stack([full["X"]] + [noise_cols[n] for n in noise_names])
        for j, n in enumerate(noise_names):
            schema_idx[n] = len(SCHEMA_24) + j
        print(f"[sweep] appended {len(noise_names)} seeded noise columns: "
              f"{', '.join(noise_names)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(seq)
    first_elapsed = None
    with out_path.open("a", encoding="utf-8") as fh:
        for mi, m in enumerate(seq):
            if m["mask_id"] in done:
                continue
            feats = m.get("full_features") or feature_list_for(m["toggleables_on"])
            col_idx = [schema_idx[f] for f in feats]

            t0 = time.time()
            per_fold, mean, importances = cv_one_mask(
                tune_gt, full, folds, feats, col_idx
            )
            elapsed = time.time() - t0

            rec = {
                "mask_id": m["mask_id"],
                "label": m["label"],
                "toggleables_on": m["toggleables_on"],
                "n_features": len(feats),
                "per_fold": per_fold,
                "mean": mean,
                "importances_gain": importances,
                "elapsed_s": round(elapsed, 2),
            }
            if "full_features" in m:
                rec["full_features"] = m["full_features"]
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done.add(m["mask_id"])

            print(f"[sweep] {mi + 1}/{total} [{m['label']}] "
                  f"{len(m['toggleables_on'])} toggles on  "
                  f"NDCG={mean['ndcg_5k']:.4f} R@10={mean['recall_10']:.4f}  "
                  f"{elapsed:.1f}s")

            # Full-scale anchor gate: verify the all-on mask reproduces WP4's
            # R_post CV before spending compute on the rest. Only at full scale
            # (smoke truncation shifts the numbers by construction).
            if m["label"] == "all_on" and args.max_queries is None:
                anchor_gate(per_fold, args.ignore_anchor_mismatch)

            if first_elapsed is None:
                first_elapsed = elapsed
                remaining = len(todo) - 1
                est = remaining * elapsed
                print(f"[sweep] est. remaining: {remaining} masks x {elapsed:.1f}s "
                      f"~= {est / 60:.1f} min ({est / 3600:.2f} h)")

    print(f"[sweep] done. {out_path}")


if __name__ == "__main__":
    main()
