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


def aggregate(records: list[dict], label_key: str = "judge_label",
              f1_key: str = "f1") -> dict:
    """Compute per-category and overall accuracy metrics.

    label_key and f1_key allow aggregating over corrected results.
    """
    cats = defaultdict(lambda: {
        "count": 0, "correct": 0, "f1_sum": 0.0,
        "ret_hit": 0, "ret_count": 0,
    })

    for r in records:
        cat = r["category"]
        cats[cat]["count"] += 1
        if r.get(label_key) == "CORRECT":
            cats[cat]["correct"] += 1
        cats[cat]["f1_sum"] += r.get(f1_key, 0.0)
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


def _has_corrected(records: list[dict]) -> bool:
    """Check if any records have corrected GT judgments."""
    return any(r.get("corrected_label") for r in records)


def _aggregate_corrected(records: list[dict]) -> dict:
    """Aggregate using corrected labels where available, falling back to original.

    For questions with corrected GT, uses corrected_label and corrected_f1.
    For all other questions, uses the original judge_label and f1.
    """
    cats = defaultdict(lambda: {
        "count": 0, "correct": 0, "f1_sum": 0.0,
        "ret_hit": 0, "ret_count": 0,
        "corrected_count": 0,  # how many used corrected GT
    })

    for r in records:
        cat = r["category"]
        cats[cat]["count"] += 1

        if r.get("corrected_label"):
            # Use corrected judgment
            if r["corrected_label"] == "CORRECT":
                cats[cat]["correct"] += 1
            cats[cat]["f1_sum"] += r.get("corrected_f1", r.get("f1", 0.0))
            cats[cat]["corrected_count"] += 1
        else:
            # Use original judgment
            if r.get("judge_label") == "CORRECT":
                cats[cat]["correct"] += 1
            cats[cat]["f1_sum"] += r.get("f1", 0.0)

        if r.get("retrieval_hit"):
            cats[cat]["ret_hit"] += 1
        cats[cat]["ret_count"] += r.get("retrieved_count", 0)

    total_n = sum(m["count"] for m in cats.values())
    total_c = sum(m["correct"] for m in cats.values())
    main_n = sum(m["count"] for k, m in cats.items() if k != 5)
    main_c = sum(m["correct"] for k, m in cats.items() if k != 5)
    total_corrected = sum(m["corrected_count"] for m in cats.values())

    return {
        "per_category": dict(cats),
        "total": {"count": total_n, "correct": total_c,
                  "accuracy": total_c / total_n if total_n else 0},
        "main_1_4": {"count": main_n, "correct": main_c,
                     "accuracy": main_c / main_n if main_n else 0},
        "corrected_questions": total_corrected,
    }


def _print_table(agg: dict, header: str):
    """Print one results table."""
    print(f"\n{'=' * 65}")
    print(header)
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
        if m5["count"]:
            print(f"\nAdversarial: {m5['correct']}/{m5['count']} correctly identified "
                  f"({m5['correct'] / m5['count']:.1%})")


def print_report(records: list[dict]):
    """Print formatted report to stdout.

    If corrected GT data is present, prints both original and corrected
    tables, plus a delta summary showing which questions flipped.
    """
    agg_orig = aggregate(records)
    _print_table(agg_orig, "LoCoMo End-to-End QA Results (original GT)")

    if not _has_corrected(records):
        print()
        return

    agg_corr = _aggregate_corrected(records)
    _print_table(agg_corr, "LoCoMo End-to-End QA Results (corrected GT)")

    # Delta summary
    n_corrected = agg_corr.get("corrected_questions", 0)
    orig_acc = agg_orig["main_1_4"]["accuracy"]
    corr_acc = agg_corr["main_1_4"]["accuracy"]
    delta = corr_acc - orig_acc

    print(f"\n--- GT Correction Impact ---")
    print(f"Questions with corrected GT: {n_corrected}")
    print(f"OVERALL (1-4): {orig_acc:.1%} original -> {corr_acc:.1%} corrected "
          f"({delta:+.1%})")

    # Show flips
    flips_to_correct = []
    flips_to_wrong = []
    for r in records:
        if not r.get("corrected_label"):
            continue
        orig = r.get("judge_label")
        corr = r["corrected_label"]
        if orig == "WRONG" and corr == "CORRECT":
            flips_to_correct.append(r)
        elif orig == "CORRECT" and corr == "WRONG":
            flips_to_wrong.append(r)

    if flips_to_correct:
        print(f"\nFlipped WRONG->CORRECT ({len(flips_to_correct)}):")
        for r in flips_to_correct:
            cat = CATEGORY_NAMES.get(r["category"], "?")
            print(f"  [{cat}] {r['question'][:60]}")
            print(f"    orig GT: {r['ground_truth'][:50]}")
            print(f"    corr GT: {r.get('corrected_gt', '?')[:50]}")

    if flips_to_wrong:
        print(f"\nFlipped CORRECT->WRONG ({len(flips_to_wrong)}):")
        for r in flips_to_wrong:
            cat = CATEGORY_NAMES.get(r["category"], "?")
            print(f"  [{cat}] {r['question'][:60]}")
            print(f"    orig GT: {r['ground_truth'][:50]}")
            print(f"    corr GT: {r.get('corrected_gt', '?')[:50]}")

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

    # Add corrected GT summary if available
    if _has_corrected(records):
        agg_corr = _aggregate_corrected(records)
        summary["corrected"] = agg_corr

        # Per-conversation corrected
        per_conv_corr = defaultdict(lambda: {"count": 0, "correct": 0})
        for r in records:
            conv = r["conv_id"]
            per_conv_corr[conv]["count"] += 1
            if r.get("corrected_label"):
                if r["corrected_label"] == "CORRECT":
                    per_conv_corr[conv]["correct"] += 1
            elif r.get("judge_label") == "CORRECT":
                per_conv_corr[conv]["correct"] += 1
        for conv in per_conv_corr:
            m = per_conv_corr[conv]
            m["accuracy"] = m["correct"] / m["count"] if m["count"] else 0
        summary["corrected"]["per_conversation"] = dict(per_conv_corr)

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
