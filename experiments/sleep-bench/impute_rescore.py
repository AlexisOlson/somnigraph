#!/usr/bin/env python3
"""H7 provenance-impute re-scoring -- experiments/sleep-bench/impute_rescore.py

The coverage decomposition for the REM floor sweep, done STRUCTURALLY (no LLM).
The raw sweep (analyze_rem_floor.py) credits only evidence-turn ids, so a summary
that out-ranks its own evidence turn scores as a loss even if it carries that
turn's content. This re-scorer tests how much of that raw penalty is that
artifact by imputing each retrieved summary's rank-credit back to the evidence
turns it was GENERATED FROM.

THE REWRITE (per query, ranked full-id list R):
  walk R in rank order ->
    - a detail memory: keep it (first occurrence).
    - a LEAF summary S: replace it, at its rank, with the evidence ids in
      (S.generated_from INTERSECT this query's evidence) not already seen. If S
      covers none of the query's evidence it KEEPS its slot (a non-covering
      summary still displaces -- this is the honest, tighter upper bound, not the
      looser "delete every summary" one).
  Then score the rewritten R' against the UNCHANGED evidence GT with the same
  metrics retrieve.py used.

WHY THIS IS AN UPPER BOUND: generated_from is exact provenance (S was built with
turn e in context), but a dense ~600-char summary may have compressed away e's
specific fact. So impute can OVER-credit. Bracket:
    RAW (analyze_rem_floor.py) <= true coverage <= IMPUTE (here)
The LLM coverage judge is the tighter measure inside the bracket; this bounds it
for free.

READING IT: control is rem_none (no summaries -> impute == raw). For each rung,
the decisive number is IMPUTE(rung) - rem_none, paired:
    - >= 0 (CI)  -> the raw penalty is fully coverage-displacement; summaries are
                    neutral-to-helpful once their origins are credited. Verdict
                    then needs the judge to place the truth in the bracket.
    - <  0 (CI)  -> REAL regression: even crediting full provenance coverage,
                    summaries displace evidence they don't cover (or the rerank
                    churn costs ranking). Summaries hurt, judge or no judge.

Pure re-scoring of existing JSONL + a read-only generated_from / token / session
read of the REM arm. Touches no live store; safe under the abstinence hold.

Usage:
    uv run experiments/sleep-bench/impute_rescore.py \
        --rem-dir ~/.somnigraph/benchmark/sleepbench-arms/conv0-rem
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sleep-fork"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # gt.py -> locomo_bench.ingest

import gt as gt_mod
import metrics
import store
from analyze_h3 import fmt_ci
from analyze_sleep_fork import paired_bootstrap_ci

METRICS = ["ndcg_at_budget", "r10", "mrr", "recall_any_10", "recall_all_10", "recall_any_effk"]


def leaf_genfrom(db_path: Path) -> dict[str, set]:
    """{leaf_summary_id: set(source detail ids)} -- the provenance map.

    Leaf = no source is itself a summary. Matches sweep_rem_floor's classifier.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        layer = {r[0]: r[1] for r in conn.execute("SELECT id, layer FROM memories").fetchall()}
        out: dict[str, set] = {}
        for sid, gen_from in conn.execute(
            "SELECT id, generated_from FROM memories WHERE layer='summary'"
        ).fetchall():
            if not gen_from:
                continue
            members = json.loads(gen_from)
            if members and not any(layer.get(m) == "summary" for m in members):
                out[sid] = set(members)
        return out
    finally:
        conn.close()


def impute_ranking(retrieved: list, is_summary: set, cover_map: dict) -> list:
    """Rewrite a ranked list, crediting each retrieved summary to the evidence
    ids it covers FOR THIS QUERY (cover_map: {summary_id: set(evidence_ids)}).
    A summary covering nothing keeps its slot (still displaces). The cover_map is
    provenance (generated_from) or judged, set by the caller -- same rewrite."""
    out, seen = [], set()
    for x in retrieved:
        if x in is_summary:
            covered = [e for e in sorted(cover_map.get(x, ())) if e not in seen]
            if covered:
                for e in covered:
                    out.append(e)
                    seen.add(e)
            else:
                out.append(x)  # non-covering summary keeps its displacing slot
        elif x not in seen:
            out.append(x)
            seen.add(x)
    return out


def score_row(retrieved: list, evidence: list, token_map: dict, session_map: dict,
              budget: int, session_k: int) -> dict:
    """Replicate retrieve.py's metric block for a (possibly rewritten) list."""
    rel = set(evidence)
    gt_scores = {mid: 1.0 for mid in evidence}
    effk = metrics.effective_k(retrieved, session_map, session_k)
    return {
        "ndcg_at_budget": metrics.ndcg_at_budget(retrieved, gt_scores, token_map, budget),
        "r10": metrics.recall_at_k(retrieved, rel, 10),
        "mrr": metrics.mrr(retrieved, rel),
        "recall_any_10": metrics.recall_any(retrieved[:10], evidence),
        "recall_all_10": metrics.recall_all(retrieved[:10], evidence),
        "recall_any_effk": metrics.recall_any(retrieved[:effk], evidence),
    }


def load_rows(results_dir: Path, label: str) -> dict:
    path = results_dir / f"{label}__rrf.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing {path} -- run sweep_rem_floor.py first")
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["query_id"]] = r
    return out


def rungs_in(results_dir: Path) -> list:
    labels = [p.name[:-len("__rrf.jsonl")] for p in results_dir.glob("rem_*__rrf.jsonl")]
    def key(l):
        if l == "rem_none":
            return (0, 0)
        try:
            return (1, -int(l.split("_m")[1]))
        except (IndexError, ValueError):
            return (2, 0)
    return sorted(set(labels), key=key)


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="H7 provenance-impute re-scoring")
    ap.add_argument("--rem-dir", required=True)
    ap.add_argument("--judged-coverage", default=None,
                    help="summary_coverage.json from judge_summary_coverage.py. "
                         "Given: credit summaries only where the judge confirmed "
                         "coverage (the answer inside the bracket). Omitted: "
                         "provenance upper bound (generated_from).")
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    ap.add_argument("--recall-budget", type=int, default=5000)
    ap.add_argument("--session-k", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rem = Path(args.rem_dir).expanduser().resolve()
    rdir = Path(args.results_dir).expanduser()
    db_path = rem / "memory.db"

    genfrom = leaf_genfrom(db_path)
    is_summary = set(genfrom)
    token_map = store.load_token_map(rem)
    session_map = gt_mod.build_session_map(rem)
    judged = None
    if args.judged_coverage:
        judged = json.loads(Path(args.judged_coverage).expanduser().read_text())
    mode = "JUDGED (answer, inside bracket)" if judged else "PROVENANCE (upper bound)"
    print(f"[impute] {len(genfrom)} leaf summaries; {len(token_map)} active tokens; "
          f"{len(session_map)} session-mapped; mode={mode}")

    labels = rungs_in(rdir)
    control = load_rows(rdir, "rem_none")

    # Common query set across all rungs.
    per_rung = {lbl: load_rows(rdir, lbl) for lbl in labels}
    common = set(control)
    for rows in per_rung.values():
        common &= set(rows)
    common = sorted(common)
    print(f"[impute] {len(common)} common queries across {labels}\n")

    # RAW control metrics (== impute control, no summaries to expand).
    raw_ctrl = {q: control[q]["metrics"] for q in common}

    # For each rung compute IMPUTE metrics per query, then table + deltas vs ctrl.
    imp = {}  # label -> {q: metrics}
    for lbl in labels:
        rows = per_rung[lbl]
        imp[lbl] = {}
        for q in common:
            row = rows[q]
            evidence_set = set(row["evidence_ids"])
            if judged is not None:
                jc = judged.get(q, {})
                cover_map = {sid: (set(elist) & evidence_set) for sid, elist in jc.items()}
            else:
                cover_map = {sid: (gf & evidence_set)
                             for sid, gf in genfrom.items() if gf & evidence_set}
            r_prime = impute_ranking(row["retrieved"], is_summary, cover_map)
            imp[lbl][q] = score_row(
                r_prime, row["evidence_ids"], token_map, session_map,
                args.recall_budget, args.session_k)

    print("=== Means: RAW vs IMPUTE (rrf) ===")
    hdr = f"{'rung':<9} {'n':>4}  " + "  ".join(f"{m[:8]:>8}" for m in METRICS)
    print("RAW"); print(hdr); print("-" * len(hdr))
    for lbl in labels:
        rowm = per_rung[lbl]
        vals = "  ".join(f"{mean(rowm[q]['metrics'][m] for q in common):>8.4f}" for m in METRICS)
        print(f"{lbl:<9} {'':>4}  {vals}")
    print(f"\nIMPUTE [{mode}] (summary credit -> covered evidence)"); print(hdr); print("-" * len(hdr))
    for lbl in labels:
        vals = "  ".join(f"{mean(imp[lbl][q][m] for q in common):>8.4f}" for m in METRICS)
        print(f"{lbl:<9} {'':>4}  {vals}")

    print("\n=== DECISIVE: IMPUTE(rung) - rem_none control, paired bootstrap ===")
    print("  (>=0 => raw penalty was coverage-displacement; <0 => real regression)")
    for lbl in labels:
        if lbl == "rem_none":
            continue
        print(f"\n  --- {lbl} ---")
        for m in METRICS:
            deltas = []
            for q in common:
                vb = imp[lbl][q].get(m)
                va = raw_ctrl[q].get(m)
                if vb is None or va is None:
                    continue
                deltas.append(float(vb) - float(va))
            print(f"    {m:<16} {fmt_ci(paired_bootstrap_ci(deltas, args.n_boot, args.seed))}")

    # For contrast: the RAW penalty at full summaries (rem_m3 - rem_none).
    print("\n=== For contrast: RAW rem_m3 - rem_none (the penalty we're explaining) ===")
    full = per_rung.get("rem_m3")
    if full:
        for m in METRICS:
            deltas = [float(full[q]["metrics"][m]) - float(raw_ctrl[q][m])
                      for q in common
                      if full[q]["metrics"].get(m) is not None and raw_ctrl[q].get(m) is not None]
            print(f"    {m:<16} {fmt_ci(paired_bootstrap_ci(deltas, args.n_boot, args.seed))}")


if __name__ == "__main__":
    main()
