"""Inspect top-N reranker scores for specific queries.

Diagnostic for "is the regression because the model is wrongly confident on
irrelevant memories (high scores everywhere) or uncertain across the board
(low scores everywhere)?" The per-query NDCG output in train_reranker only
shows the *ranking* outcome, not the underlying score values.

Loads the trained model + features pickle, predicts scores for every (query,
candidate) row in the cached features, prints top-N predicted scores per
query alongside the GT label and memory summary.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import lightgbm as lgb
import numpy as np

from memory.constants import DATA_DIR
from memory.db import get_db


FEATURES_PATH = DATA_DIR / "tuning_studies" / "reranker_features.pkl"
MODEL_PATH = DATA_DIR / "tuning_studies" / "reranker_model.txt"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", action="append", required=True,
                        help="Query text to inspect (repeatable)")
    parser.add_argument("--top-n", type=int, default=5,
                        help="How many top-scored candidates to show per query")
    parser.add_argument("--show-target-context", action="store_true",
                        help="If GT memory isn't in top-N, show its score and rank")
    args = parser.parse_args()

    print(f"Loading model from {MODEL_PATH}...")
    booster = lgb.Booster(model_file=str(MODEL_PATH))

    print(f"Loading features from {FEATURES_PATH}...")
    with open(FEATURES_PATH, "rb") as f:
        feature_data = pickle.load(f)

    feature_matrix = feature_data["features"]
    memory_ids = feature_data["memory_ids"]  # list of (qtext, mid)
    gt_labels = feature_data["labels"]
    feature_names = feature_data.get("feature_names", [])

    print(f"  {len(memory_ids)} (query, candidate) rows")
    print(f"  {feature_matrix.shape[0]} samples x {feature_matrix.shape[1]} features")

    # Index rows by query
    rows_by_query: dict[str, list[int]] = {}
    for i, (qtext, mid) in enumerate(memory_ids):
        rows_by_query.setdefault(qtext, []).append(i)

    # Load memory summaries for display
    db = get_db()
    summaries = {
        r["id"]: (r["summary"] or "")[:80]
        for r in db.execute(
            "SELECT id, summary FROM memories WHERE status = 'active'"
        )
    }
    db.close()

    for query in args.query:
        rows = rows_by_query.get(query)
        if rows is None:
            print(f"\n=== Query NOT FOUND in features: '{query}' ===")
            print("  Try a substring match...")
            matches = [q for q in rows_by_query if query[:30].lower() in q.lower()]
            for m in matches[:5]:
                print(f"    candidate: '{m}'")
            continue

        # Predict scores for all candidates of this query
        X = feature_matrix[rows]
        scores = booster.predict(X)
        # Sort by score desc
        order = np.argsort(-scores)

        print(f"\n=== Query: '{query}' ===")
        print(f"  {len(rows)} candidates, predicted score range "
              f"[{scores.min():.4f}, {scores.max():.4f}], "
              f"mean={scores.mean():.4f}")

        positives = sum(1 for i in rows if gt_labels[i] > 0)
        print(f"  Positive GT labels: {positives}")

        print(f"\n  Top-{args.top_n} predicted:")
        print(f"    rank  score    gt     mid       summary")
        for k in range(min(args.top_n, len(order))):
            idx_in_rows = order[k]
            row_idx = rows[idx_in_rows]
            mid = memory_ids[row_idx][1]
            summary = summaries.get(mid, "(inactive or missing)")
            gt = gt_labels[row_idx]
            print(f"    {k+1:>3}.  {scores[idx_in_rows]:>6.4f}  "
                  f"{gt:>5.2f}  {mid[:8]}  {summary}")

        if args.show_target_context:
            # Find any positives outside top-N
            top_n_row_idxs = {rows[order[k]] for k in range(min(args.top_n, len(order)))}
            outside_positives = [
                (rows[i], scores[i]) for i in range(len(rows))
                if gt_labels[rows[i]] > 0 and rows[i] not in top_n_row_idxs
            ]
            if outside_positives:
                # Compute their actual rank in the predicted order
                row_to_rank = {rows[order[k]]: k + 1 for k in range(len(order))}
                outside_positives.sort(key=lambda x: -x[1])
                print(f"\n  Positives outside top-{args.top_n}:")
                print(f"    rank  score    gt     mid       summary")
                for row_idx, score in outside_positives[:5]:
                    mid = memory_ids[row_idx][1]
                    summary = summaries.get(mid, "(inactive)")
                    gt = gt_labels[row_idx]
                    print(f"    {row_to_rank[row_idx]:>4}.  {score:>6.4f}  "
                          f"{gt:>5.2f}  {mid[:8]}  {summary}")


if __name__ == "__main__":
    main()
