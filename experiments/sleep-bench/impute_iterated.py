#!/usr/bin/env python3
"""H11 coverage-credit re-scoring for the iterated-cycle chain -- experiments/sleep-bench/impute_iterated.py

analyze_iterated.py reports the RAW substrate, which UNDER-credits every
summary-bearing arm (c1_rem onward): a REM summary that out-ranks the evidence
turn it covers is scored as a miss (H7's sign flip). This adapter applies
impute_rescore.py's rank-rewrite (credit each retrieved leaf summary, at its
rank, to the evidence ids it covers) to the H11 arm JSONLs, which
impute_rescore.py itself cannot do -- it is hardcoded to the H7 floor-sweep
rungs (`rem_*__rrf.jsonl`, control `rem_none`).

Reuses impute_rescore.py's leaf_genfrom / impute_ranking / score_row VERBATIM
(one source of truth for the rewrite) and analyze_h3's paired_bootstrap_ci
(one source of truth for the stats). Two modes, same as impute_rescore:

  - PROVENANCE (no --judged-coverage): a retrieved summary credits ALL its
    generated_from evidence. UPPER bound (a dense summary may have compressed
    away the specific fact).
  - JUDGED (--judged-coverage <json>): credit only judge-confirmed coverage.
    The answer inside the RAW <= JUDGED <= PROVENANCE bracket. NOTE: the H7
    `results/summary_coverage.json` does NOT cover the H11 summaries (fresh
    REM pass, fresh ids -- verified 0/28 overlap); the judge must be re-run
    on an H11 arm before this mode is usable here.

Per-arm store state is respected: token_map / session_map / generated_from are
read from EACH arm's own db (consolidate archives memories across c2-c4, so
the active token pool differs per arm).

Deltas reported per (label, scorer), paired bootstrap over common queries:
  - IMPUTE(label) - RAW(label):   the size of the evidence-id artifact.
  - IMPUTE(label) - RAW(frozen):  the coverage-credited total sleep effect.

Pure re-scoring of existing JSONL + read-only sqlite reads of the arm stores.
Touches no live store; safe under the abstinence hold.

Usage:
    uv run experiments/sleep-bench/impute_iterated.py
    uv run experiments/sleep-bench/impute_iterated.py \
        --labels c1_rem c1 c2 c3 c4 --scorers rrf formula \
        [--judged-coverage experiments/sleep-bench/results/summary_coverage_h11.json]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sleep-fork"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # gt.py -> locomo_bench.ingest

import gt as gt_mod
import store
from analyze_h3 import fmt_ci, paired_bootstrap_ci
from impute_rescore import METRICS, impute_ranking, leaf_genfrom, mean, score_row

DEFAULT_ARMS = Path.home() / ".somnigraph" / "benchmark" / "sleepbench-arms"
DEFAULT_LABELS = ["c1_rem", "c1", "c2", "c3", "c4"]


def load_rows(results_dir: Path, label: str, scorer: str) -> dict:
    path = results_dir / f"{label}__{scorer}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing {path} -- run retrieve.py for that arm first")
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["query_id"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="H11 coverage-credit re-scoring (chain adapter)")
    ap.add_argument("--arms-dir", default=str(DEFAULT_ARMS))
    ap.add_argument("--conv", type=int, default=0)
    ap.add_argument("--labels", nargs="+", default=DEFAULT_LABELS,
                    help="summary-bearing arms to re-credit (c1_rem onward)")
    ap.add_argument("--baseline", default="frozen",
                    help="RAW baseline label for the total-effect delta")
    ap.add_argument("--scorers", nargs="+", default=["rrf", "formula"])
    ap.add_argument("--judged-coverage", default=None,
                    help="summary_coverage-style json for the H11 summaries. "
                         "Omitted: provenance upper bound.")
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    ap.add_argument("--recall-budget", type=int, default=5000)
    ap.add_argument("--session-k", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)  # sleep-fork parity
    args = ap.parse_args()

    arms = Path(args.arms_dir).expanduser().resolve()
    rdir = Path(args.results_dir).expanduser()

    judged = None
    if args.judged_coverage:
        judged = json.loads(Path(args.judged_coverage).expanduser().read_text())
    mode = "JUDGED (answer, inside bracket)" if judged else "PROVENANCE (upper bound)"

    # Per-arm store context (token/session maps + provenance differ per arm).
    arm_ctx = {}
    for lab in args.labels:
        d = arms / f"conv{args.conv}-{lab}"
        genfrom = leaf_genfrom(d / "memory.db")
        arm_ctx[lab] = {
            "genfrom": genfrom,
            "is_summary": set(genfrom),
            "token_map": store.load_token_map(d),
            "session_map": gt_mod.build_session_map(d),
        }
        print(f"[impute-h11] {lab}: {len(genfrom)} leaf summaries, "
              f"{len(arm_ctx[lab]['token_map'])} active tokens")

    if judged is not None:
        all_sids = set().union(*(c["is_summary"] for c in arm_ctx.values()))
        judged_sids = {s for cov in judged.values() for s in cov}
        missing = all_sids - judged_sids
        if missing:
            sys.exit(f"[impute-h11] judged coverage missing {len(missing)}/{len(all_sids)} "
                     f"H11 summary ids -- re-run judge_summary_coverage.py on an H11 arm.")

    print(f"[impute-h11] mode={mode}; baseline={args.baseline}; "
          f"n_boot={args.n_boot}, seed={args.seed}\n")

    for sc in args.scorers:
        base_rows = load_rows(rdir, args.baseline, sc)
        per_label = {lab: load_rows(rdir, lab, sc) for lab in args.labels}
        common = set(base_rows)
        for rows in per_label.values():
            common &= set(rows)
        common = sorted(common)

        imp = {}
        for lab in args.labels:
            ctx = arm_ctx[lab]
            imp[lab] = {}
            for q in common:
                row = per_label[lab][q]
                evidence_set = set(row["evidence_ids"])
                if judged is not None:
                    jc = judged.get(q, {})
                    cover_map = {sid: (set(elist) & evidence_set)
                                 for sid, elist in jc.items()}
                else:
                    cover_map = {sid: (gf & evidence_set)
                                 for sid, gf in ctx["genfrom"].items() if gf & evidence_set}
                r_prime = impute_ranking(row["retrieved"], ctx["is_summary"], cover_map)
                imp[lab][q] = score_row(
                    r_prime, row["evidence_ids"], ctx["token_map"], ctx["session_map"],
                    args.recall_budget, args.session_k)

        print(f"=== [scorer={sc}] means: RAW vs IMPUTE [{mode}] "
              f"(n={len(common)} common queries) ===")
        hdr = f"{'cell':<22}  " + "  ".join(f"{m[:8]:>8}" for m in METRICS)
        print(hdr); print("-" * len(hdr))
        base_means = "  ".join(
            f"{mean(base_rows[q]['metrics'][m] for q in common):>8.4f}" for m in METRICS)
        print(f"{args.baseline + ' (raw)':<22}  {base_means}")
        for lab in args.labels:
            raw_means = "  ".join(
                f"{mean(per_label[lab][q]['metrics'][m] for q in common):>8.4f}" for m in METRICS)
            imp_means = "  ".join(
                f"{mean(imp[lab][q][m] for q in common):>8.4f}" for m in METRICS)
            print(f"{lab + ' (raw)':<22}  {raw_means}")
            print(f"{lab + ' (impute)':<22}  {imp_means}")

        def paired(title, get_b, get_a):
            print(f"\n  --- {title} ---")
            for m in METRICS:
                deltas = []
                for q in common:
                    vb, va = get_b(q).get(m), get_a(q).get(m)
                    if vb is None or va is None:
                        continue
                    deltas.append(float(vb) - float(va))
                print(f"    {m:<16} {fmt_ci(paired_bootstrap_ci(deltas, args.n_boot, args.seed))}")

        print(f"\n=== [scorer={sc}] IMPUTE(label) - RAW(label): the evidence-id artifact ===")
        for lab in args.labels:
            paired(f"{lab}: impute - raw", lambda q, l=lab: imp[l][q],
                   lambda q, l=lab: per_label[l][q]["metrics"])

        print(f"\n=== [scorer={sc}] IMPUTE(label) - RAW({args.baseline}): "
              f"coverage-credited total effect ===")
        for lab in args.labels:
            paired(f"{lab} (impute) - {args.baseline} (raw)", lambda q, l=lab: imp[l][q],
                   lambda q: base_rows[q]["metrics"])
        print()


if __name__ == "__main__":
    main()
