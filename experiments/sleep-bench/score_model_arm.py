#!/usr/bin/env python3
"""H10 reranker-2x2 model eval scorer (WP3) -- experiments/sleep-bench/score_model_arm.py

Scores ONE cell of the pre/post-sleep reranker 2x2: given a trained LightGBM
booster + its feature-name list and a store, rank conv0's GT queries and emit
rows in the EXACT Stage-A JSONL format that `retrieve.py` produces, so the
existing `analyze_iterated.py` / `impute_iterated.py` contrast + judged-credit
machinery applies UNCHANGED (plan sec D6).

WHY A SEPARATE SCORER (not a retrieve.py scorer): production `rerank()`
(src/memory/reranker.py) builds a HARDCODED 31-feature matrix and does NOT
column-select by the model's feature JSON, so a 24-feature H10 model can never
be served by the live path (plan sec D5 hazard). This scorer instead shares the
TRAINER's extraction (bench_features.extract_features_35 + select_schema), giving
train/serve parity by construction.

CANDIDATE POOL IS MODEL-INDEPENDENT. The pool is whatever the shared
search/extraction path yields -- `fts_results` u `vec_results` u `theme_ranked`
u `ppr_scores` (all active-only, since the FTS/vec indices only hold active
rows). It is determined by the STORE, identical for R_pre and R_post; the model
only RE-SCORES that fixed pool. This is the same pool the four (six) cells share
so cell contrasts isolate scoring policy, not candidate generation.

STORE ACCESS IS READ-ONLY apart from tune_gt's benign sidecar caches
(tune_gt_embeddings.pkl / tune_gt_data.pkl / tuning_studies/tune_gt_ppr_cache.pkl
in DATA_DIR) -- memory.db itself is only SELECTed. Scoring does NOT mutate
recall_count (this path never calls impl_recall; it drives tune_gt's precomputed
searches directly), so unlike the live retrieve.py path it leaves the arm store's
access counters untouched.

Module-level DB/embedding/reranker caches are the known cross-store trap: this
script points env at the store via `ingest_baseline._point_at_fresh_store` THEN
`importlib.reload(tune_gt)` (the WP2-documented sequence) so tune_gt's
import-bound DATA_DIR / MEMORY_DB rebind to the target store.

Output (mirrors retrieve.py):
  results-full/{label}__model-{model-stem}.jsonl   (one row per scoreable query)
  results-full/{label}__model-{model-stem}.summary.json

The `scorer` field / filename component is `model-{model-stem}`, so
`impute_iterated.load_rows(results_dir, label, "model-{stem}")` addresses it.

Usage:
    uv run --extra tuning experiments/sleep-bench/score_model_arm.py \
        --data-dir ~/.somnigraph/benchmark/sleepbench-full/conv0-c2 \
        --model results-full/R_post.txt \
        --features results-full/R_post__features.json \
        --label c2 --conv 0
"""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(HERE))

import gt as gt_mod  # noqa: E402
import metrics  # noqa: E402
import store  # noqa: E402
from bench_features import (  # noqa: E402
    ALL_FEATURE_NAMES_35,
    augment_memory_meta,
    extract_features_35,
    select_schema,
)

DEFAULT_LOCOMO_DATA = Path.home() / ".claude" / "repos" / "locomo" / "data" / "locomo10.json"
DEFAULT_OUT_DIR = HERE / "results-full"
DEFAULT_KEY_FILE = Path(r"C:\Users\Alexis\.claude\data\openai_api_key")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True,
                   help="store to score (a 2x2 store: frozen or c2)")
    p.add_argument("--model", required=True,
                   help="LightGBM booster .txt (booster_.save_model output)")
    p.add_argument("--features", required=True,
                   help="feature-names .json -- the model's column list (subset of the 35)")
    p.add_argument("--label", required=True,
                   help="cell label, e.g. frozen / c2 (becomes the JSONL name prefix)")
    p.add_argument("--conv", type=int, default=0,
                   help="LoCoMo conversation to evaluate (H10 headline eval conv = 0)")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--locomo-data", default=str(DEFAULT_LOCOMO_DATA))
    p.add_argument("--recall-budget", type=int, default=5000,
                   help="NDCG token budget -- sleep-fork/Stage-A parity default")
    p.add_argument("--session-k", type=int, default=10,
                   help="target k for effective_k / recall_any / recall_all")
    # Not in the WP3 arg headline but required for Stage-A parity: retrieve.py
    # scores the top `recall_limit` (impl_recall's limit=20) candidates, and the
    # existing rrf/formula rows in results-full carry 20 retrieved. We truncate
    # the model-ranked pool to the same cutoff so cell contrasts are apples-to-
    # apples and impute_iterated re-scores the same window.
    p.add_argument("--recall-limit", type=int, default=20,
                   help="rank cutoff for the emitted `retrieved` list (Stage-A default 20)")
    p.add_argument("--openai-key-file", default=str(DEFAULT_KEY_FILE))
    return p


def now_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def load_feature_names(features_path: Path) -> list[str]:
    """Read the model's feature-name list. Accepts a bare JSON list or a dict
    with a `feature_names` / `features` key (the V6-bundle shape)."""
    obj = json.loads(features_path.read_text())
    if isinstance(obj, list):
        names = obj
    elif isinstance(obj, dict):
        for key in ("feature_names", "features", "names"):
            if key in obj and isinstance(obj[key], list):
                names = obj[key]
                break
        else:
            raise ValueError(
                f"{features_path}: dict has no feature-name list under "
                "feature_names/features/names"
            )
    else:
        raise ValueError(f"{features_path}: unexpected JSON type {type(obj).__name__}")
    unknown = [n for n in names if n not in ALL_FEATURE_NAMES_35]
    if unknown:
        raise ValueError(
            f"{features_path}: feature name(s) not in the 35-name superset: {unknown}"
        )
    return list(names)


def point_and_reload(store_dir: Path, key_file: Path):
    """WP2 store-pointing sequence: mutate memory.* module globals to target the
    store, set the key, THEN reload tune_gt so its import-bound DATA_DIR/MEMORY_DB
    rebind. Returns the (reloaded) tune_gt module + the PPR damping constant."""
    from ingest_baseline import _point_at_fresh_store

    _point_at_fresh_store(store_dir)
    key = key_file.read_text().strip() if key_file.exists() else store.resolve_openai_key()
    if key:
        os.environ["OPENAI_API_KEY"] = key

    import tune_gt
    importlib.reload(tune_gt)  # rebind DATA_DIR / MEMORY_DB from the mutated modules
    from memory.constants import PPR_DAMPING
    return tune_gt, PPR_DAMPING


def main() -> None:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    key_file = Path(args.openai_key_file).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = now_run_id()

    model_path = Path(args.model).expanduser().resolve()
    features_path = Path(args.features).expanduser().resolve()
    model_stem = model_path.stem
    scorer_name = f"model-{model_stem}"

    feature_names = load_feature_names(features_path)
    print(f"[score-model] model={model_path.name} ({len(feature_names)} features) "
          f"label={args.label} conv={args.conv} store={data_dir}")

    # --- point at the store (WP2 trap sequence) + load booster ---
    tune_gt, ppr_damping = point_and_reload(data_dir, key_file)
    import lightgbm as lgb  # noqa: E402
    from train_reranker import PRODUCTION_PARAMS  # noqa: E402

    booster = lgb.Booster(model_file=str(model_path))
    n_model_feats = booster.num_feature()
    if n_model_feats != len(feature_names):
        raise ValueError(
            f"model expects {n_model_feats} features but --features lists "
            f"{len(feature_names)}"
        )

    # --- GT / eval query set (conv0) ---
    queries = gt_mod.build_query_set(Path(args.locomo_data), data_dir, args.conv)
    qtext_to_q: dict[str, dict] = {}
    collisions = 0
    for q in queries:
        if q["question"] in qtext_to_q:
            collisions += 1
        qtext_to_q[q["question"]] = q
    if collisions:
        print(f"[score-model] WARNING: {collisions} duplicate question texts in conv "
              f"{args.conv}; last occurrence wins (search_data is keyed by text).")

    # Temp GT {question: {mid: 1.0}} -- labels are irrelevant for SCORING; the GT
    # file only tells load_tuning_data which query texts to precompute searches
    # for (plan WP3).
    tiny_gt = {q["question"]: {mid: 1.0 for mid in q["evidence_ids"]} for q in queries}
    gt_tmp = out_dir / f".score_model_gt_{args.label}_{scorer_name}.json"
    gt_tmp.write_text(json.dumps(tiny_gt))

    # --- shared search/extraction context (train/serve parity) ---
    from memory.db import get_db

    db = get_db()
    meta = augment_memory_meta(db)
    full_data, ground_truth, _ = tune_gt.load_tuning_data(
        str(gt_tmp), ppr_dampings=[ppr_damping]
    )
    params = dict(PRODUCTION_PARAMS)

    # metric-side maps -- IDENTICAL to retrieve.py (active-only token_map,
    # theme-derived session_map). NOT tune_gt's internal token_map.
    token_map = store.load_token_map(data_dir)
    session_map = gt_mod.build_session_map(data_dir)

    jsonl_path = out_dir / f"{args.label}__{scorer_name}.jsonl"
    summary_path = out_dir / f"{args.label}__{scorer_name}.summary.json"

    rows_out = []
    n = 0
    t0 = time.time()
    for q in queries:
        qtext = q["question"]
        feats35, _labels, mids = extract_features_35(
            qtext, params, full_data["search_data"],
            full_data["feedback_raw"], full_data["hebb_data"],
            meta, ground_truth.get(qtext, {}),
        )
        if len(mids) == 0:
            ranked = []
            score_map = {}
        else:
            X, _ = select_schema(feats35, ALL_FEATURE_NAMES_35, feature_names)
            preds = booster.predict(np.asarray(X, dtype=np.float64))
            # Deterministic ranking: predicted score desc, memory id as stable
            # tiebreak (byte-identical across reruns regardless of sort stability).
            order = sorted(range(len(mids)), key=lambda i: (-float(preds[i]), mids[i]))
            ranked = [mids[i] for i in order]
            score_map = {mids[i]: round(float(preds[i]), 6) for i in order}

        retrieved_full = ranked[: args.recall_limit]
        scores_full = {mid: score_map[mid] for mid in retrieved_full}

        evidence_ids = q["evidence_ids"]
        gt_scores = {mid: 1.0 for mid in evidence_ids}
        rel_ids = set(evidence_ids)
        eff_k = metrics.effective_k(retrieved_full, session_map, args.session_k)

        row = {
            "query_id": q["query_id"],
            "query": qtext,
            "category": q["category"],
            "gt_answer": q["gt_answer"],
            "evidence_ids": evidence_ids,
            "retrieved": retrieved_full,
            "scores": scores_full,
            "budget": args.recall_budget,
            "limit": args.recall_limit,
            "metrics": {
                "ndcg_at_budget": metrics.ndcg_at_budget(
                    retrieved_full, gt_scores, token_map, args.recall_budget),
                "r10": metrics.recall_at_k(retrieved_full, rel_ids, 10),
                "mrr": metrics.mrr(retrieved_full, rel_ids),
                "recall_any_10": metrics.recall_any(retrieved_full[:10], evidence_ids),
                "recall_all_10": metrics.recall_all(retrieved_full[:10], evidence_ids),
                "effective_k": eff_k,
                "recall_any_effk": metrics.recall_any(retrieved_full[:eff_k], evidence_ids),
                "recall_all_effk": metrics.recall_all(retrieved_full[:eff_k], evidence_ids),
            },
        }
        row = {"run_id": run_id, "condition": args.label, "scorer": scorer_name, **row}
        rows_out.append(row)
        n += 1
        if n % 40 == 0:
            el = time.time() - t0
            print(f"  {n}/{len(queries)} scored ({el:.1f}s, {n / el:.1f} q/s)")

    # Fresh write (overwrite) -> byte-identical across reruns for the same store.
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row) + "\n")
    gt_tmp.unlink(missing_ok=True)

    # --- summary (mirrors retrieve.py write_summary) ---
    metric_keys = [
        "ndcg_at_budget", "r10", "mrr", "recall_any_10", "recall_all_10",
        "effective_k", "recall_any_effk", "recall_all_effk",
    ]

    def mean(key):
        vals = [r["metrics"][key] for r in rows_out if r["metrics"].get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    summary = {
        "label": args.label,
        "scorer": scorer_name,
        "scorer_status": f"model:{model_path.name} ({len(feature_names)} feats)",
        "conv": args.conv,
        "data_dir": str(data_dir),
        "model": str(model_path),
        "features": str(features_path),
        "recall_limit": args.recall_limit,
        "recall_budget": args.recall_budget,
        "session_k": args.session_k,
        "n_queries": len(rows_out),
        "generated_at": now_run_id(),
        "metrics_mean": {key: mean(key) for key in metric_keys},
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    mm = summary["metrics_mean"]
    print(f"[score-model] {len(rows_out)} rows -> {jsonl_path}")
    print(f"[score-model] summary -> {summary_path}")
    print(
        f"           NDCG@{args.recall_budget}={mm['ndcg_at_budget']:.4f}  "
        f"R@10={mm['r10']:.4f}  MRR={mm['mrr']:.4f}  "
        f"recall_any@10={mm['recall_any_10']:.4f}  "
        f"recall_all@10={mm['recall_all_10']:.4f}  "
        f"recall_any@effk={mm['recall_any_effk']:.4f}"
    )


if __name__ == "__main__":
    main()
