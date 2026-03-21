"""Results aggregation and reporting."""

import json
from collections import defaultdict
from pathlib import Path

from .config import CATEGORY_NAMES


def load_results(path: Path) -> list[dict]:
    """Load per-question results from JSONL."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def aggregate(records: list[dict]) -> dict:
    """Compute per-category and overall accuracy metrics."""
    cats = defaultdict(lambda: {
        "count": 0, "correct": 0, "f1_sum": 0.0,
        "ret_hit": 0, "ret_count": 0,
    })

    for r in records:
        cat = r["category"]
        cats[cat]["count"] += 1
        if r.get("judge_label") == "CORRECT":
            cats[cat]["correct"] += 1
        cats[cat]["f1_sum"] += r.get("f1", 0.0)
        if r.get("retrieval_hit"):
            cats[cat]["ret_hit"] += 1
        cats[cat]["ret_count"] += r.get("retrieved_count", 0)

    # Overall metrics
    total_n = sum(m["count"] for m in cats.values())
    total_c = sum(m["correct"] for m in cats.values())
    main_n = sum(m["count"] for k, m in cats.items() if k != 5)
    main_c = sum(m["correct"] for k, m in cats.items() if k != 5)

    return {
        "per_category": dict(cats),
        "total": {"count": total_n, "correct": total_c,
                  "accuracy": total_c / total_n if total_n else 0},
        "main_1_4": {"count": main_n, "correct": main_c,
                     "accuracy": main_c / main_n if main_n else 0},
    }


def print_report(records: list[dict]):
    """Print formatted report to stdout."""
    agg = aggregate(records)

    print(f"\n{'=' * 65}")
    print("LoCoMo End-to-End QA Results")
    print(f"{'=' * 65}")
    print(f"{'Category':<20} {'N':>5} {'Acc':>7} {'F1':>7} {'RetHit':>7}")
    print("-" * 50)

    cats = agg["per_category"]
    for cat in sorted(cats):
        m = cats[cat]
        n = m["count"]
        name = CATEGORY_NAMES.get(cat, f"cat_{cat}")
        acc = m["correct"] / n if n else 0
        f1 = m["f1_sum"] / n if n else 0
        ret = m["ret_hit"] / n if n else 0
        print(f"{name:<20} {n:>5} {acc:>7.1%} {f1:>7.3f} {ret:>7.1%}")

    print("-" * 50)
    t = agg["total"]
    m14 = agg["main_1_4"]
    if t["count"]:
        print(f"{'ALL (1-5)':<20} {t['count']:>5} {t['accuracy']:>7.1%}")
    if m14["count"]:
        print(f"{'OVERALL (1-4)':<20} {m14['count']:>5} {m14['accuracy']:>7.1%}")

    if 5 in cats:
        m5 = cats[5]
        print(f"\nAdversarial: {m5['correct']}/{m5['count']} correctly identified "
              f"({m5['correct'] / m5['count']:.1%})" if m5["count"] else "")

    print()


def write_summary(records: list[dict], path: Path):
    """Write full results summary to JSON."""
    agg = aggregate(records)

    # Per-conversation breakdown
    per_conv = defaultdict(lambda: {"count": 0, "correct": 0})
    for r in records:
        conv = r["conv_id"]
        per_conv[conv]["count"] += 1
        if r.get("judge_label") == "CORRECT":
            per_conv[conv]["correct"] += 1

    for conv in per_conv:
        m = per_conv[conv]
        m["accuracy"] = m["correct"] / m["count"] if m["count"] else 0

    summary = {
        **agg,
        "per_conversation": dict(per_conv),
        "question_count": len(records),
    }

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
