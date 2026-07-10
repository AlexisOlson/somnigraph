#!/usr/bin/env python3
"""Cross-judge agreement audit -- experiments/sleep-bench/cross_judge_audit.py

Gates H10's D4 (sonnet train labels): sonnet AND opus judge the SAME sampled
(question, summary) coverage pairs from the H10 training conversations, and we
report agreement measured on THIS corpus instead of importing L5b's 87.5%.
Poor agreement = stop-and-rethink before training (plan-reranker-2x2.md
§ circularity).

Reuses judge_summary_coverage.py's prompt, block builder, batching, parser,
and claude -p caller VERBATIM (one source of truth -- the audit must measure
the judges, not a prompt difference). Query/evidence sourcing goes straight
through gt.build_query_set (no retrieval runs needed).

Sampling: seeded shuffle of convs 1-9 queries, accumulating whole queries
(a query contributes all its provenance-eligible summary pairs, matching the
judge's batch format) until >= --min-pairs pairs.

RUN IN TERMINAL (two LLM judge passes; ~5-6 batches per model).

Usage:
    uv run experiments/sleep-bench/cross_judge_audit.py \
        --rem-dir ~/.somnigraph/benchmark/sleepbench-full/conv0-c2 \
        [--convs 1-9] [--min-pairs 50] [--seed 1234] [--models sonnet opus]
"""

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # gt.py -> locomo_bench.ingest

import gt as gt_mod
from ingest_baseline import _parse_convs
from judge_summary_coverage import (
    BATCH_SIZE,
    BATCH_FOOTER,
    SYSTEM_PROMPT,
    build_question_block,
    call_claude,
    load_store,
    parse_batch_response,
)

DEFAULT_LOCOMO = Path.home() / ".claude" / "repos" / "locomo" / "data" / "locomo10.json"


def judge_pass(model: str, pending: list, content: dict, genfrom: dict) -> dict:
    """Run one judge model over the sampled queries. Returns {qkey: {sid: [eids]}}."""
    results = {}
    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start:start + BATCH_SIZE]
        question_map, blocks = {}, []
        for i, (qkey, qinfo) in enumerate(batch, 1):
            q_label = f"Q{i}"
            bd = build_question_block(qinfo, content, genfrom, q_label)
            if bd is None:
                continue
            block, ev_labels, sum_labels = bd
            question_map[q_label] = {"evidence_labels": ev_labels,
                                     "summary_labels": sum_labels, "qkey": qkey}
            blocks.append(block)
        if not blocks:
            continue
        response = call_claude(SYSTEM_PROMPT, "\n".join(blocks) + BATCH_FOOTER, model)
        parsed = parse_batch_response(response, question_map)
        for q_label, qinfo in question_map.items():
            qkey = qinfo["qkey"]
            results[qkey] = parsed.get(qkey, {})
        print(f"  [{model}] {start + len(batch)}/{len(pending)} queries")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Sonnet-vs-opus coverage-judge agreement audit")
    ap.add_argument("--rem-dir", required=True, help="slept store with the summaries (c2)")
    ap.add_argument("--locomo-data", default=str(DEFAULT_LOCOMO))
    ap.add_argument("--convs", default="1-9", help="training conversations to sample from")
    ap.add_argument("--min-pairs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--models", nargs=2, default=["sonnet", "opus"])
    ap.add_argument("--out", default=str(HERE / "results-full" / "cross_judge_audit.json"))
    ap.add_argument("--dry-run", action="store_true", help="enumerate + sample, no LLM")
    args = ap.parse_args()

    rem = Path(args.rem_dir).expanduser().resolve()
    content, genfrom = load_store(rem)

    # Eligible queries from the training convs, judge-format qinfo dicts.
    eligible = []  # (qkey, qinfo, n_pairs)
    for conv in _parse_convs(args.convs):
        for row in gt_mod.build_query_set(Path(args.locomo_data), rem, conv):
            ev = set(row["evidence_ids"])
            n_pairs = sum(1 for gf in genfrom.values() if gf & ev)
            if n_pairs == 0:
                continue
            qinfo = {"question": row["question"], "answer": row["gt_answer"],
                     "category": row["category"], "evidence_ids": row["evidence_ids"]}
            eligible.append((row["query_id"], qinfo, n_pairs))

    total_pairs = sum(n for _, _, n in eligible)
    print(f"[audit] {len(eligible)} eligible queries / {total_pairs} pairs across convs {args.convs}")

    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    pending, n_pairs = [], 0
    for qkey, qinfo, n in eligible:
        pending.append((qkey, qinfo))
        n_pairs += n
        if n_pairs >= args.min_pairs:
            break
    print(f"[audit] sampled {len(pending)} queries = {n_pairs} pairs (seed {args.seed})")
    if args.dry_run:
        return

    verdicts = {}
    for model in args.models:
        print(f"[audit] judging with {model}...")
        verdicts[model] = judge_pass(model, pending, content, genfrom)

    m1, m2 = args.models
    # Pair-level comparison. Binary = "covers >=1 evidence id" vs "covers none";
    # exact = identical covered-evidence sets. A pair missing from a model's
    # parse output counts as a disagreement (parse failure is a real failure).
    rows, bin_agree, exact_agree, parse_miss = [], 0, 0, 0
    for qkey, _ in pending:
        v1, v2 = verdicts[m1].get(qkey, {}), verdicts[m2].get(qkey, {})
        for sid in sorted(set(v1) | set(v2)):
            if sid not in v1 or sid not in v2:
                parse_miss += 1
                rows.append({"query": qkey, "summary": sid, m1: v1.get(sid),
                             m2: v2.get(sid), "binary_agree": False, "exact_agree": False})
                continue
            b1, b2 = bool(v1[sid]), bool(v2[sid])
            ba, ea = b1 == b2, set(v1[sid]) == set(v2[sid])
            bin_agree += ba
            exact_agree += ea
            rows.append({"query": qkey, "summary": sid, m1: v1[sid], m2: v2[sid],
                         "binary_agree": ba, "exact_agree": ea})

    n = len(rows)
    print(f"\n[audit] {n} judged pairs ({parse_miss} parse-miss counted as disagreement)")
    print(f"[audit] BINARY agreement (covers-anything): {bin_agree}/{n} = {bin_agree / max(1, n):.1%}")
    print(f"[audit] EXACT agreement (same evidence set): {exact_agree}/{n} = {exact_agree / max(1, n):.1%}")
    print(f"\n[audit] disagreements ({m1} vs {m2}):")
    for r in rows:
        if not r["binary_agree"]:
            q = next(qi for qk, qi in pending if qk == r["query"])
            print(f"  {r['query']} / {r['summary'][:8]}: {m1}={r[m1]} {m2}={r[m2]}")
            print(f"    Q: {q['question'][:110]}")
            print(f"    S: {content.get(r['summary'], '')[:110]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "models": args.models, "seed": args.seed, "convs": args.convs,
        "n_queries": len(pending), "n_pairs": n, "parse_miss": parse_miss,
        "binary_agreement": bin_agree / max(1, n),
        "exact_agreement": exact_agree / max(1, n),
        "pairs": rows,
    }, indent=2))
    print(f"\n[audit] -> {out}")


if __name__ == "__main__":
    main()
