#!/usr/bin/env python3
"""GT -> trainer label bridge for the H10 pre/post-sleep reranker 2x2 (WP1).

Turns LoCoMo evidence-turn ground truth (plus an optional judged summary/node
coverage table) into the production trainer's `gt_calibrated` input shape:

    {query_TEXT: {memory_id: 1.0}}

Two label sources merge into that table:
  1. Evidence turns from `gt.build_query_set(conv)` per training conv -> 1.0.
  2. (optional, --coverage) each summary/node id the coverage judge ruled
     covers >=1 of a query's evidence turns -> 1.0. Empty covered-lists earn
     NO label (a judged-non-covering summary is not relevant).

CRITICAL -- query-TEXT keying hazard (plan-reranker-2x2.md D5): the production
trainer keys GT by the query TEXT, not by query_id. Two convs that share an
identical question string would silently MERGE their label sets into one
ambiguous row. This bridge detects every question text that occurs more than
once across the included convs and DROPS all colliding queries entirely,
printing each dropped text and the convs it collided across. The surviving
table is unambiguous under text keying.

Output is byte-identical on re-run (sort_keys, no timestamps).

Usage:
    uv run experiments/sleep-bench/build_2x2_labels.py \
        --store ~/.somnigraph/benchmark/sleepbench-full/conv0-frozen \
        --convs 1-9 \
        [--coverage experiments/sleep-bench/results-full/summary_coverage.json] \
        [--out experiments/sleep-bench/results-full/gt_2x2.json]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # gt.py -> locomo_bench.ingest

import gt as gt_mod
from ingest_baseline import _parse_convs

DEFAULT_LOCOMO = Path.home() / ".claude" / "repos" / "locomo" / "data" / "locomo10.json"
DEFAULT_OUT = HERE / "results-full" / "gt_2x2.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True,
                    help="store dir supplying the dia_map (H10: conv0-frozen)")
    ap.add_argument("--convs", default="1-9", help="training conversations, e.g. '1-9'")
    ap.add_argument("--locomo-data", default=str(DEFAULT_LOCOMO))
    ap.add_argument("--coverage", default=None,
                    help="judged coverage table {query_id: {summary_id: [evidence_ids]}} "
                         "(optional; adds summary/node labels)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    store = Path(args.store).expanduser().resolve()
    convs = _parse_convs(args.convs)

    coverage = {}
    if args.coverage:
        coverage = json.loads(Path(args.coverage).expanduser().read_text(encoding="utf-8"))

    # Collect one record per query, tracking the question text for collision
    # detection. build_query_set already drops questions with no resolvable
    # evidence, so every record here carries >=1 evidence id.
    records = []  # {query_id, conv, question, evidence_ids}
    text_owners = defaultdict(list)  # question_text -> [(query_id, conv), ...]
    for conv in convs:
        for row in gt_mod.build_query_set(Path(args.locomo_data), store, conv):
            rec = {
                "query_id": row["query_id"],
                "conv": conv,
                "question": row["question"],
                "evidence_ids": row["evidence_ids"],
            }
            records.append(rec)
            text_owners[row["question"]].append((row["query_id"], conv))

    # Collision detection on question TEXT. Any text owned by >1 query is
    # ambiguous under text keying -> drop every colliding query.
    colliding_texts = {t for t, owners in text_owners.items() if len(owners) > 1}
    print(f"[bridge] {len(records)} queries across convs {args.convs} "
          f"({len(text_owners)} distinct question texts)")
    if colliding_texts:
        print(f"[bridge] COLLISIONS: {len(colliding_texts)} question texts occur "
              f"in >1 query -- dropping all colliding queries:")
        for t in sorted(colliding_texts):
            owners = text_owners[t]
            convs_hit = sorted({c for _, c in owners})
            qids = sorted(q for q, _ in owners)
            print(f"  - convs {convs_hit} | query_ids {qids}")
            print(f"      text: {t[:160]}")
    else:
        print("[bridge] COLLISIONS: none")

    # Build the label table over surviving (unique-text) queries.
    out = {}
    kept = dropped = 0
    evidence_label_count = 0
    coverage_label_count = 0
    labels_per_query = []
    for rec in records:
        if rec["question"] in colliding_texts:
            dropped += 1
            continue
        kept += 1
        labels = {}
        for eid in rec["evidence_ids"]:
            labels[eid] = 1.0
        evidence_label_count += len(rec["evidence_ids"])

        if coverage:
            cov = coverage.get(rec["query_id"], {})
            for sid, covered in cov.items():
                if not covered:  # judged non-covering -> no label
                    continue
                if sid not in labels:  # do not double-credit provenance members
                    coverage_label_count += 1
                labels[sid] = 1.0

        out[rec["question"]] = labels
        labels_per_query.append(len(labels))

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    mean_labels = sum(labels_per_query) / len(labels_per_query) if labels_per_query else 0.0
    print("\n[bridge] STATS")
    print(f"  queries kept:     {kept}")
    print(f"  queries dropped:  {dropped} (collision)")
    print(f"  labels/query mean: {mean_labels:.3f}")
    print(f"  evidence labels:  {evidence_label_count}")
    print(f"  coverage labels:  {coverage_label_count}"
          + ("" if coverage else "  (no --coverage table given)"))
    print(f"  total labels:     {evidence_label_count + coverage_label_count}")
    print(f"[bridge] -> {out_path}")


if __name__ == "__main__":
    main()
