#!/usr/bin/env python3
"""Stage A (retrieval-only) measurement rig -- experiments/sleep-bench/retrieve.py

Spec: docs/proposals/sleep-pipeline-optimization.md §3.4 (scorer set), §3.5
(staged retrieve/read/judge separation), §3.7 (metric set).

WHAT THIS IS: for each LoCoMo GT query, run the REAL retrieval path
(`memory.tools.impl_recall`, via capture.recall_with_scores) against a given
store, record the ranked candidates + scores, and score retrieval-only
metrics against evidence-turn ground truth. No reader, no LLM judge --
that's Stage B/C, deliberately not built here (see "STAGE B/C SEAM" below).

INDEPENDENT of scripts/locomo_bench/eval_retrieval.py: this is a new module
that does not import its `eval_production` / `eval_locomo` / `apply_
scoring_config` / `--snapshot` machinery (eval_retrieval.py's hardcoded
live-DB snapshot path is exactly the cruft this rig exists to escape). It
DOES call the real system-under-test (`memory.tools.impl_recall`) and DOES
reuse pure data-loading helpers (`locomo_bench.ingest.{load_locomo,
extract_turns,extract_questions}`) and the sleep-fork's verified R@10/MRR
math (`experiments/sleep-fork/analyze_sleep_fork.py`) -- see gt.py and
metrics.py's docstrings for exactly what's reused vs. freshly written and
why, per file.

CONDITION = STORE. This rig takes one `--data-dir` (a fork in some sleep
state) and a `--label`; it does not know or care what stage produced that
fork. Sweeping sleep stages means running this rig against different forks
of the same conversation (produced by fork.py / a future run_arms.py), not
a flag on this script.

STAGE B/C SEAM: every JSONL row is keyed by `query_id` (`"{conv}:{qa_idx}"`,
stable across reruns and forks). A future Stage B ({query_id, answer}) and
Stage C ({query_id, correct, judge_model}) join on that key without
reshaping anything written here -- this rig never renames or repurposes
`query_id`.

COVERAGE-SCORING EXTENSION POINT (proposal §3.6, NOT built here): sleep
creates new memories (REM summaries, consolidate rewrites) that aren't
evidence turns, so today they can only ever contribute to the RAW
(uncredited) floor -- a summary that answers a question is invisible to
`evidence_ids`/`gt_scores` as constructed in this file. The clean seam for
later coverage-crediting is right where `gt_scores` is built in `main()`
below (`gt_scores = {mid: 1.0 for mid in evidence_ids}`): a coverage judge
would extend that dict with `{summary_id: judged_relevance}` per query
before it reaches `metrics.ndcg_at_budget` / `recall_any` / `recall_all`,
and nothing else in this file would need to change. Left as a TODO, not
wired -- building the LLM coverage judge is out of scope for this session.
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from precheck import assert_isolated_data_dir  # reuse the isolation guard, don't duplicate it

import capture
import gt as gt_mod
import metrics
import scorers
import store

DEFAULT_LOCOMO_DATA = Path.home() / ".claude" / "repos" / "locomo" / "data" / "locomo10.json"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "results"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage A (retrieval-only) measurement rig for a sleep-bench store"
    )
    p.add_argument("--data-dir", required=True,
                    help="Isolated store to score (a fork in some sleep-state)")
    p.add_argument("--label", required=True,
                    help="Condition label for this store, e.g. frozen / nrem / nrem_rem / full "
                         "(freeform -- this rig is state-agnostic; the label is just a name)")
    p.add_argument("--conv", type=int, required=True,
                    help="LoCoMo conversation index this store was ingested from")
    p.add_argument("--scorers", nargs="+", default=["rrf", "formula"],
                    choices=list(scorers.SCORER_NAMES),
                    help="Default is the sleep decomposition pair: rrf (raw rrf_fuse, "
                         "content/index effect) + formula (RRF+PPR, adds graph/edge effect). "
                         "v6 is the OOD production reranker (feedback features dead on "
                         "benchmark data -- not a valid sleep instrument, kept only as a "
                         "reference); l5b is a registered-but-stubbed axis (see scorers.py).")
    p.add_argument("--recall-limit", type=int, default=20,
                    help="rank cutoff passed to impl_recall (also the MRR cutoff)")
    p.add_argument("--recall-budget", type=int, default=5000,
                    help="NDCG token budget -- sleep-fork parity default. NOTE: "
                         "eval_retrieval.py's own CLI default is 10000; don't inherit that "
                         "here (proposal §3.7 flags this exact parity trap).")
    p.add_argument("--session-k", type=int, default=10,
                    help="target k for effective_k / recall_any / recall_all")
    p.add_argument("--max-queries", type=int, default=0,
                    help="0 = all resolvable GT queries; N = first N (deterministic slice) "
                         "-- for cheap validation runs, not full sweeps")
    p.add_argument("--run-id", default=None, help="default: UTC timestamp")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--locomo-data", default=str(DEFAULT_LOCOMO_DATA))
    p.add_argument("--fresh", action="store_true",
                    help="ignore existing JSONL for this (label, scorer) and recompute all queries")
    return p


def now_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def load_existing(jsonl_path: Path) -> dict:
    """query_id -> row, for resumability. A crash-safe checkpoint format:
    each row is appended + flushed individually (see main()'s write loop),
    so a killed run leaves a valid, reloadable prefix."""
    done = {}
    if jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                done[row["query_id"]] = row
    return done


def score_one_query(
    q: dict,
    data_dir: Path,
    token_map: dict,
    session_map: dict,
    recall_budget: int,
    recall_limit: int,
    session_k: int,
    expand: bool = True,
) -> dict:
    """Run the real retrieval path for one GT query and score it. Returns
    the full JSONL row (minus run_id/condition/scorer, added by the caller)."""
    ranked_short, scores_short, _top_score = capture.recall_with_scores(
        query=q["question"], context=q["question"],
        budget=recall_budget, limit=recall_limit,
        expand=expand,
    )
    full_map = store.resolve_full_ids(data_dir, ranked_short)
    retrieved_full = [full_map[s] for s in ranked_short if s in full_map]
    scores_full = {full_map[s]: sc for s, sc in scores_short.items() if s in full_map}

    evidence_ids = q["evidence_ids"]
    # Extension point for coverage scoring (see module docstring): a later
    # pass would merge {summary_id: judged_relevance} into gt_scores here.
    gt_scores = {mid: 1.0 for mid in evidence_ids}
    rel_ids = set(evidence_ids)

    eff_k = metrics.effective_k(retrieved_full, session_map, session_k)

    return {
        "query_id": q["query_id"],
        "query": q["question"],
        "category": q["category"],
        "gt_answer": q["gt_answer"],
        "evidence_ids": evidence_ids,
        "retrieved": retrieved_full,
        "scores": scores_full,
        "budget": recall_budget,
        "limit": recall_limit,
        "metrics": {
            "ndcg_at_budget": metrics.ndcg_at_budget(
                retrieved_full, gt_scores, token_map, recall_budget),
            "r10": metrics.recall_at_k(retrieved_full, rel_ids, 10),
            "mrr": metrics.mrr(retrieved_full, rel_ids),
            "recall_any_10": metrics.recall_any(retrieved_full[:10], evidence_ids),
            "recall_all_10": metrics.recall_all(retrieved_full[:10], evidence_ids),
            "effective_k": eff_k,
            "recall_any_effk": metrics.recall_any(retrieved_full[:eff_k], evidence_ids),
            "recall_all_effk": metrics.recall_all(retrieved_full[:eff_k], evidence_ids),
        },
    }


def write_summary(
    done: dict, queries: list[dict], scorer_name: str, status: str,
    summary_path: Path, args: argparse.Namespace,
) -> dict:
    rows = [done[q["query_id"]] for q in queries if q["query_id"] in done]
    n = len(rows)

    def mean(key):
        vals = [r["metrics"][key] for r in rows if r["metrics"].get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    metric_keys = [
        "ndcg_at_budget", "r10", "mrr", "recall_any_10", "recall_all_10",
        "effective_k", "recall_any_effk", "recall_all_effk",
    ]
    summary = {
        "label": args.label,
        "scorer": scorer_name,
        "scorer_status": status,
        "conv": args.conv,
        "data_dir": str(Path(args.data_dir).resolve()),
        "recall_limit": args.recall_limit,
        "recall_budget": args.recall_budget,
        "session_k": args.session_k,
        "n_queries": n,
        "generated_at": now_run_id(),
        "metrics_mean": {key: mean(key) for key in metric_keys},
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    assert_isolated_data_dir(data_dir)

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or now_run_id()

    store.point_at_store(data_dir)

    # Edge-embedding health: INFORMATIONAL ONLY (corrected 2026-07-04). The
    # earlier version hard-refused expansion scoring when contextful edges
    # lacked a linking_embedding, on the belief that PPR's context gate skips
    # them (a false null). That gate exists ONLY in
    # scoring._expand_adjacency_legacy, which nothing in src/ calls. The real
    # expansion path -- impl_recall(expand=True) -> scoring.expand_via_ppr --
    # SELECTs only (source_id, target_id, weight, flags) and traverses every
    # non-contradiction edge by WEIGHT; it never reads linking_embedding
    # (verified: scoring.py:316-320 + full fn to 395; tools.py:748-751). So
    # unembedded (REM-created) edges are NOT invisible and do NOT null the graph
    # signal. We report the blind count as a REM/NREM edge-hygiene note -- REM
    # creates contextful edges without embedding them, NREM embeds -- and
    # proceed. This is a cosmetic asymmetry, not a scoring hazard.
    edge_health = store.edge_embedding_health(data_dir)
    if edge_health["blind_edges"] > 0:
        print(
            f"[stage-a] edge-embedding note: {edge_health['blind_edges']} of "
            f"{edge_health['contextful_edges']} contextful edges "
            f"({edge_health['blind_frac']:.1%}) have no linking_embedding "
            f"(REM does not embed its edges). Harmless: expand_via_ppr "
            f"traverses by weight, not embedding -- proceeding."
        )
    elif edge_health["total_edges"]:
        print(f"[stage-a] edge-embedding health OK: "
              f"{edge_health['contextful_edges']} contextful edges all embedded "
              f"({edge_health['total_edges']} total).")

    print(f"[stage-a] run_id={run_id} label={args.label} conv={args.conv} data_dir={data_dir}")
    print(f"[stage-a] building GT from {args.locomo_data}")
    queries = gt_mod.build_query_set(Path(args.locomo_data), data_dir, args.conv)
    if args.max_queries:
        queries = queries[: args.max_queries]
    session_map = gt_mod.build_session_map(data_dir)
    token_map = store.load_token_map(data_dir)
    print(f"[stage-a] {len(queries)} scoreable GT queries, "
          f"{len(session_map)} memories session-mapped, "
          f"{len(token_map)} active memories in token_map")

    for scorer_name in args.scorers:
        print(f"\n[stage-a] === scorer={scorer_name} ===")
        status = scorers.apply_scorer(scorer_name)  # raises loud on failure -- no silent fallback
        expand = scorers.wants_expansion(scorer_name)
        print(f"[stage-a] scorer_status: {status}  (expand={expand})")

        jsonl_path = out_dir / f"{args.label}__{scorer_name}.jsonl"
        summary_path = out_dir / f"{args.label}__{scorer_name}.summary.json"

        if args.fresh and jsonl_path.exists():
            jsonl_path.unlink()

        done = load_existing(jsonl_path)
        remaining = sum(1 for q in queries if q["query_id"] not in done)
        print(f"[stage-a] {len(done)} rows already in {jsonl_path.name}, {remaining} remaining")

        n_new = 0
        t0 = time.time()
        with open(jsonl_path, "a", encoding="utf-8") as f:
            for q in queries:
                if q["query_id"] in done:
                    continue
                row = score_one_query(
                    q, data_dir, token_map, session_map,
                    args.recall_budget, args.recall_limit, args.session_k,
                    expand=expand,
                )
                row = {"run_id": run_id, "condition": args.label, "scorer": scorer_name, **row}
                f.write(json.dumps(row) + "\n")
                f.flush()
                done[q["query_id"]] = row
                n_new += 1
                if n_new % 20 == 0:
                    elapsed = time.time() - t0
                    print(f"  {n_new}/{remaining} new queries "
                          f"({elapsed:.1f}s, {n_new / elapsed:.1f} q/s)")

        print(f"[stage-a] {scorer_name}: {n_new} newly scored, {len(done)} total -> {jsonl_path}")

        summary = write_summary(done, queries, scorer_name, status, summary_path, args)
        mm = summary["metrics_mean"]
        print(f"[stage-a] summary -> {summary_path}")
        print(
            f"           NDCG@{args.recall_budget}={mm['ndcg_at_budget']:.4f}  "
            f"R@10={mm['r10']:.4f}  MRR={mm['mrr']:.4f}  "
            f"recall_any@10={mm['recall_any_10']:.4f}  "
            f"recall_all@10={mm['recall_all_10']:.4f}  "
            f"recall_any@effk={mm['recall_any_effk']:.4f}"
        )


if __name__ == "__main__":
    main()
