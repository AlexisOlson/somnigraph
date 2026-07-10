#!/usr/bin/env python3
"""H10 pre/post-sleep reranker 2x2 analyzer -- experiments/sleep-bench/analyze_2x2.py

WP5 of plan-reranker-2x2.md. Reads the four (or six) cell Stage-A JSONL files
produced by score_model_arm.py (WP3), plus the frozen/slept rrf/formula
references already in results-full/, and reports the decomposition H10 tests:

  |                                | Store: frozen  | Store: slept (c2) |
  | R_pre  (fit on frozen store)   | (1) baseline   | (2) naive deploy  |
  | R_post (fit on slept store)    | (3) counterfac | (4) aligned       |

The five contrasts (paired bootstrap over the query set common to all cells):
  (2)-(1)  slept_Rpre  - frozen_Rpre   naive deploy vs baseline (the money shot)
  (4)-(3)  slept_Rpost - frozen_Rpost
  (3)-(1)  frozen_Rpost - frozen_Rpre  scoring-policy value on the frozen store
  (4)-(2)  slept_Rpost - slept_Rpre    scoring-policy value on the slept store
  interaction = ((4)-(2)) - ((3)-(1))  per-query double-difference, then bootstrap.
     interaction > 0  <=>  sleep pays off only with a sleep-aware reranker.

Bonus (6 cells): undormant_Rpost - undormant_Rpre and undormant_Rpost - slept_Rpost.

RAW vs JUDGED: the slept column is recall-capped by dormancy (~97% of GT evidence
is dormant), so raw NDCG there is an artifact -- judged credit (a retrieved summary
credits the evidence ids the judge ruled it covers, at the summary's rank) is the
PRIMARY cell metric on any summary-bearing store. On the frozen store there are no
summaries, so judged == raw trivially (imputation is skipped). Both are reported and
both dumped; contrasts are computed on each.

Every piece of scoring machinery is REUSED, not reimplemented:
  - impute_rescore.{METRICS, leaf_genfrom, impute_ranking, score_row, mean}
    (the judged/provenance rank-rewrite; one source of truth, shared with
     impute_iterated.py / impute_rescore.py)
  - analyze_h3.{paired_bootstrap_ci, fmt_ci}  (the stats; sleep-fork parity)
  - store.load_token_map / gt.build_session_map  (per-store active pool + sessions)

Each cell maps to its own store (arms-dir/conv{conv}-{label}, label = stem before
'__') so token_map / session_map / generated_from are read from the store that cell
was scored on -- essential because consolidate archives members, so the active pool
differs per store.

Pure re-scoring of existing JSONL + read-only sqlite reads of the arm stores.
Touches no live store; safe under the abstinence hold; deterministic across reruns.

Usage:
    # stand-in acceptance test (relabelled rrf/formula columns for the model cells)
    uv run experiments/sleep-bench/analyze_2x2.py \
        --cell frozen_Rpre=frozen__rrf   --cell slept_Rpre=c2__rrf \
        --cell frozen_Rpost=frozen__formula --cell slept_Rpost=c2__formula \
        --judged-coverage experiments/sleep-bench/results-full/summary_coverage.json

    # real run once WP3/WP4 have produced the model cells
    uv run experiments/sleep-bench/analyze_2x2.py \
        --cell frozen_Rpre=frozen__model-R_pre  --cell slept_Rpre=c2__model-R_pre \
        --cell frozen_Rpost=frozen__model-R_post --cell slept_Rpost=c2__model-R_post \
        --cell undormant_Rpre=c1_rem_undormant__model-R_pre \
        --cell undormant_Rpost=c1_rem_undormant__model-R_post \
        --judged-coverage experiments/sleep-bench/results-full/summary_coverage.json \
        --importances results-full/R_pre.json results-full/R_post.json
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

DEFAULT_RESULTS = HERE / "results-full"
DEFAULT_ARMS = Path.home() / ".somnigraph" / "benchmark" / "sleepbench-full"

# The 2x2 cell names -> the numbered contrast positions in the design table.
CANON_ORDER = ["frozen_Rpre", "slept_Rpre", "frozen_Rpost", "slept_Rpost"]
UNDORMANT = ["undormant_Rpre", "undormant_Rpost"]
POS = {"frozen_Rpre": 1, "slept_Rpre": 2, "frozen_Rpost": 3, "slept_Rpost": 4}


def parse_cells(cell_args: list[str]) -> dict[str, str]:
    """--cell NAME=STEM  ->  {name: stem}. STEM is a results-dir file stem
    (e.g. frozen__rrf or c2__model-R_post), i.e. the filename minus '.jsonl'."""
    cells = {}
    for spec in cell_args or []:
        if "=" not in spec:
            sys.exit(f"--cell expects NAME=STEM, got {spec!r}")
        name, stem = spec.split("=", 1)
        name, stem = name.strip(), stem.strip()
        if name not in POS and name not in UNDORMANT:
            sys.exit(f"unknown cell name {name!r}; expected one of "
                     f"{CANON_ORDER + UNDORMANT}")
        cells[name] = stem
    return cells


def store_label(stem: str) -> str:
    """The store a cell was scored on = the stem prefix before '__'."""
    return stem.split("__", 1)[0]


def load_rows(results_dir: Path, stem: str) -> dict:
    path = results_dir / f"{stem}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"missing Stage-A output: {path}\n"
            f"  score_model_arm.py must have written {stem}.jsonl first."
        )
    _assert_stem_matches_store(results_dir, stem)
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["query_id"]] = r
    return out


def _assert_stem_matches_store(results_dir: Path, stem: str) -> None:
    """Guard against a mislabeled cell: score_model_arm stamps the actual
    --data-dir into {stem}.summary.json, while this analyzer derives each
    cell's store purely from the filename stem. If an operator paired
    --label c2 with the frozen store, raw metrics and judged imputation
    would silently disagree — one corrupted cell, wrong interaction."""
    summary_path = results_dir / f"{stem}.summary.json"
    if not summary_path.exists():
        return
    data_dir = json.loads(summary_path.read_text()).get("data_dir")
    if not data_dir:
        return  # retrieve.py summaries don't stamp data_dir; nothing to check
    actual = Path(data_dir).name          # e.g. conv0-c2
    expected_label = store_label(stem)    # e.g. c2
    if actual.split("-", 1)[-1] != expected_label:
        sys.exit(
            f"ERROR: cell stem {stem!r} implies store label {expected_label!r} "
            f"but {summary_path.name} says it was scored on {actual!r} "
            f"(data_dir={data_dir}).\n  The --label/--data-dir pairing of the "
            f"score_model_arm run was wrong; re-score that cell."
        )


def build_store_ctx(arms: Path, conv: int, label: str) -> dict:
    """Per-store impute context. genfrom empty (no summaries) -> frozen-type:
    judged == raw, imputation skipped."""
    d = arms / f"conv{conv}-{label}"
    if not (d / "memory.db").exists():
        raise FileNotFoundError(f"no store at {d} for cell label {label!r}")
    genfrom = leaf_genfrom(d / "memory.db")
    return {
        "dir": d,
        "genfrom": genfrom,
        "is_summary": set(genfrom),
        "token_map": store.load_token_map(d),
        "session_map": gt_mod.build_session_map(d),
        "summary_bearing": bool(genfrom),
    }


def judged_metrics_for_cell(rows: dict, common: list, ctx: dict, judged: dict | None,
                            budget: int, session_k: int) -> dict:
    """{q: metrics_dict} of JUDGED-credited metrics for one cell.

    Frozen-type store (no summaries): judged == raw, return the row metrics as-is.
    Summary-bearing store: apply the impute rank-rewrite (judged coverage if given,
    else provenance upper bound) exactly as impute_iterated/impute_rescore do."""
    if not ctx["summary_bearing"]:
        return {q: rows[q]["metrics"] for q in common}
    out = {}
    for q in common:
        row = rows[q]
        evidence_set = set(row["evidence_ids"])
        if judged is not None:
            jc = judged.get(q, {})
            cover_map = {sid: (set(elist) & evidence_set) for sid, elist in jc.items()}
        else:
            cover_map = {sid: (gf & evidence_set)
                         for sid, gf in ctx["genfrom"].items() if gf & evidence_set}
        r_prime = impute_ranking(row["retrieved"], ctx["is_summary"], cover_map)
        out[q] = score_row(r_prime, row["evidence_ids"], ctx["token_map"],
                            ctx["session_map"], budget, session_k)
    return out


def contrast_block(title, name_b, name_a, metric_by_cell, common, n_boot, seed):
    """One (b - a) contrast across all METRICS; returns {metric: ci_dict} and prints."""
    print(f"\n  --- {title} ---")
    out = {}
    mb, ma = metric_by_cell[name_b], metric_by_cell[name_a]
    for m in METRICS:
        deltas = []
        for q in common:
            vb, va = mb[q].get(m), ma[q].get(m)
            if vb is None or va is None:
                continue
            deltas.append(float(vb) - float(va))
        ci = paired_bootstrap_ci(deltas, n_boot, seed)
        out[m] = ci
        print(f"    {m:<16} {fmt_ci(ci)}")
    return out


def interaction_block(title, metric_by_cell, common, n_boot, seed):
    """Per-query double-difference ((4)-(2)) - ((3)-(1)), then bootstrap those."""
    print(f"\n  --- {title} ---")
    c1, c2 = metric_by_cell["frozen_Rpre"], metric_by_cell["slept_Rpre"]
    c3, c4 = metric_by_cell["frozen_Rpost"], metric_by_cell["slept_Rpost"]
    out = {}
    for m in METRICS:
        dd = []
        for q in common:
            v1, v2 = c1[q].get(m), c2[q].get(m)
            v3, v4 = c3[q].get(m), c4[q].get(m)
            if None in (v1, v2, v3, v4):
                continue
            dd.append((float(v4) - float(v2)) - (float(v3) - float(v1)))
        ci = paired_bootstrap_ci(dd, n_boot, seed)
        out[m] = ci
        print(f"    {m:<16} {fmt_ci(ci)}")
    return out


def load_importances(path: Path) -> dict[str, float]:
    """Lenient loader -> {feature: gain}. Tolerates several dump shapes."""
    obj = json.loads(Path(path).read_text())
    # {"feat": gain, ...}
    if isinstance(obj, dict) and all(isinstance(v, (int, float)) for v in obj.values()):
        return {k: float(v) for k, v in obj.items()}
    # {"importances": {...}} / {"feature_importance": {...}} / {"gain": {...}}
    for key in ("importances", "feature_importance", "importance", "gain", "gains"):
        if isinstance(obj, dict) and key in obj:
            sub = obj[key]
            if isinstance(sub, dict):
                return {k: float(v) for k, v in sub.items()}
            # parallel arrays
            names = obj.get("feature_names") or obj.get("features") or obj.get("names")
            if isinstance(sub, list) and names and len(names) == len(sub):
                return {n: float(g) for n, g in zip(names, sub)}
    # parallel arrays at top level
    if isinstance(obj, dict):
        names = obj.get("feature_names") or obj.get("features") or obj.get("names")
        gains = obj.get("gain") or obj.get("gains") or obj.get("values")
        if names and gains and len(names) == len(gains):
            return {n: float(g) for n, g in zip(names, gains)}
    # list of [name, gain]
    if isinstance(obj, list) and obj and isinstance(obj[0], (list, tuple)):
        return {row[0]: float(row[1]) for row in obj}
    raise ValueError(f"unrecognized importance dump shape in {path}")


def means_table(metric_by_cell_raw, metric_by_cell_judged, present, common,
                baseline_stem, baseline_raw_rows):
    print("=== Per-cell means (n=%d common queries) ===" % len(common))
    hdr = f"{'cell':<24} {'kind':<7}  " + "  ".join(f"{m[:9]:>9}" for m in METRICS)
    print(hdr)
    print("-" * len(hdr))
    order = [n for n in CANON_ORDER + UNDORMANT if n in present]
    table = {}
    for name in order:
        raw = "  ".join(
            f"{mean(metric_by_cell_raw[name][q].get(m) for q in common):>9.4f}"
            for m in METRICS)
        jud = "  ".join(
            f"{mean(metric_by_cell_judged[name][q].get(m) for q in common):>9.4f}"
            for m in METRICS)
        pos = POS.get(name)
        tag = f"({pos}) {name}" if pos else name
        print(f"{tag:<24} {'raw':<7}  {raw}")
        print(f"{'':<24} {'judged':<7}  {jud}")
        table[name] = {
            "raw": {m: mean(metric_by_cell_raw[name][q].get(m) for q in common)
                    for m in METRICS},
            "judged": {m: mean(metric_by_cell_judged[name][q].get(m) for q in common)
                       for m in METRICS},
        }
    if baseline_raw_rows is not None:
        raw = "  ".join(
            f"{mean(baseline_raw_rows[q]['metrics'].get(m) for q in common):>9.4f}"
            for m in METRICS)
        print(f"{('ref:' + baseline_stem):<24} {'raw':<7}  {raw}")
        table[f"ref:{baseline_stem}"] = {
            "raw": {m: mean(baseline_raw_rows[q]["metrics"].get(m) for q in common)
                    for m in METRICS}}
    return table


def main() -> None:
    ap = argparse.ArgumentParser(
        description="H10 pre/post-sleep reranker 2x2 analyzer (WP5)")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--arms-dir", default=str(DEFAULT_ARMS))
    ap.add_argument("--conv", type=int, default=0)
    ap.add_argument("--cell", action="append", default=[], metavar="NAME=STEM",
                    help="cell name = Stage-A file stem (minus .jsonl). Names: "
                         + ", ".join(CANON_ORDER) + " [+ " + ", ".join(UNDORMANT)
                         + "]. Give the 4 core cells (and optionally the 2 "
                         "undormant cells).")
    ap.add_argument("--judged-coverage", default=None,
                    help="summary_coverage.json (judge-confirmed coverage). Applied "
                         "to summary-bearing stores; omit for the provenance upper "
                         "bound. Frozen-store cells ignore it (judged==raw).")
    ap.add_argument("--baseline-raw", default=None,
                    help="optional reference stem (e.g. frozen__rrf) shown as a raw "
                         "reference row; not part of the five contrasts.")
    ap.add_argument("--importances", nargs=2, default=None, metavar=("R_pre", "R_post"),
                    help="two importance-dump JSONs for a side-by-side table "
                         "(tolerated absent).")
    ap.add_argument("--recall-budget", type=int, default=5000)
    ap.add_argument("--session-k", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)  # sleep-fork parity
    ap.add_argument("--out", default=None,
                    help="JSON dump path (default: <results-dir>/analyze_2x2.json)")
    args = ap.parse_args()

    rdir = Path(args.results_dir).expanduser().resolve()
    arms = Path(args.arms_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser() if args.out else (rdir / "analyze_2x2.json")

    cells = parse_cells(args.cell)
    missing_core = [n for n in CANON_ORDER if n not in cells]
    if missing_core:
        sys.exit(f"need all four core cells; missing {missing_core}. "
                 f"Got {sorted(cells)}.")
    have_undormant = all(n in cells for n in UNDORMANT)
    present = [n for n in CANON_ORDER + UNDORMANT if n in cells]

    judged = None
    if args.judged_coverage:
        judged = json.loads(Path(args.judged_coverage).expanduser().read_text())
    mode = "JUDGED (coverage table)" if judged else "PROVENANCE (upper bound)"

    # Load rows + per-store context (one ctx per unique store label).
    rows_by_cell, store_ctx = {}, {}
    for name in present:
        stem = cells[name]
        rows_by_cell[name] = load_rows(rdir, stem)
        label = store_label(stem)
        if label not in store_ctx:
            store_ctx[label] = build_store_ctx(arms, args.conv, label)

    # Common query set across all present cells (paired everything).
    common = set(next(iter(rows_by_cell.values())))
    for rws in rows_by_cell.values():
        common &= set(rws)
    common = sorted(common)
    if not common:
        sys.exit("ERROR: no query_id common to all cells.")

    print(f"[2x2] {len(common)} query_ids common to {len(present)} cells "
          f"({', '.join(present)})")
    print(f"[2x2] mode={mode}  n_boot={args.n_boot}  seed={args.seed}  "
          f"budget={args.recall_budget}  session_k={args.session_k}")
    for name in present:
        stem = cells[name]
        ctx = store_ctx[store_label(stem)]
        kind = f"{len(ctx['genfrom'])} summaries" if ctx["summary_bearing"] else "no summaries (frozen-type)"
        print(f"       {name:<16} <- {stem:<28} store=conv{args.conv}-{store_label(stem)} ({kind})")
    print()

    # Per-cell raw + judged per-query metrics.
    raw_by_cell = {name: {q: rows_by_cell[name][q]["metrics"] for q in common}
                   for name in present}
    judged_by_cell = {
        name: judged_metrics_for_cell(
            rows_by_cell[name], common, store_ctx[store_label(cells[name])],
            judged, args.recall_budget, args.session_k)
        for name in present}

    baseline_raw_rows = None
    if args.baseline_raw:
        baseline_raw_rows = load_rows(rdir, args.baseline_raw)

    table = means_table(raw_by_cell, judged_by_cell, present, common,
                        args.baseline_raw, baseline_raw_rows)

    # --- The five contrasts, on JUDGED (primary) then RAW ---
    contrasts = {"judged": {}, "raw": {}}
    contrast_specs = [
        ("(2)-(1)  slept_Rpre - frozen_Rpre  [naive deploy vs baseline]",
         "slept_Rpre", "frozen_Rpre"),
        ("(4)-(3)  slept_Rpost - frozen_Rpost",
         "slept_Rpost", "frozen_Rpost"),
        ("(3)-(1)  frozen_Rpost - frozen_Rpre  [policy value on frozen store]",
         "frozen_Rpost", "frozen_Rpre"),
        ("(4)-(2)  slept_Rpost - slept_Rpre    [policy value on slept store]",
         "slept_Rpost", "slept_Rpre"),
    ]
    for metric_source, mbc in (("judged", judged_by_cell), ("raw", raw_by_cell)):
        print(f"\n=== Contrasts on {metric_source.upper()} metrics "
              f"(* = 95% bootstrap CI excludes 0) ===")
        for title, nb, na in contrast_specs:
            key = title.split()[0]
            contrasts[metric_source][key] = contrast_block(
                title, nb, na, mbc, common, args.n_boot, args.seed)
        contrasts[metric_source]["interaction"] = interaction_block(
            "interaction = ((4)-(2)) - ((3)-(1))  [headline: >0 => sleep needs a sleep-aware reranker]",
            mbc, common, args.n_boot, args.seed)

        # Arithmetic cross-check: interaction point est == (4-2 mean) - (3-1 mean).
        for m in METRICS:
            i_pt = contrasts[metric_source]["interaction"][m]["mean"]
            c42 = contrasts[metric_source]["(4)-(2)"][m]["mean"]
            c31 = contrasts[metric_source]["(3)-(1)"][m]["mean"]
            if None not in (i_pt, c42, c31):
                delta = abs(i_pt - (c42 - c31))
                flag = "" if delta < 1e-9 else f"  !! MISMATCH {delta:.2e}"
                if m == "ndcg_at_budget":
                    print(f"\n  [cross-check {metric_source}] interaction NDCG "
                          f"{i_pt:+.6f} == (4)-(2) {c42:+.6f} - (3)-(1) {c31:+.6f} "
                          f"= {c42 - c31:+.6f}{flag}")

    # --- Bonus undormant contrasts (6 cells) ---
    undormant_contrasts = None
    if have_undormant:
        undormant_contrasts = {"judged": {}, "raw": {}}
        for metric_source, mbc in (("judged", judged_by_cell), ("raw", raw_by_cell)):
            print(f"\n=== Undormant bonus contrasts on {metric_source.upper()} ===")
            undormant_contrasts[metric_source]["undormant_Rpost-undormant_Rpre"] = \
                contrast_block("undormant_Rpost - undormant_Rpre",
                               "undormant_Rpost", "undormant_Rpre", mbc, common,
                               args.n_boot, args.seed)
            undormant_contrasts[metric_source]["undormant_Rpost-slept_Rpost"] = \
                contrast_block("undormant_Rpost - slept_Rpost  [recall ceiling lifted]",
                               "undormant_Rpost", "slept_Rpost", mbc, common,
                               args.n_boot, args.seed)

    # --- Feature-importance side-by-side (optional) ---
    importance_table = None
    if args.importances:
        p_pre, p_post = args.importances
        try:
            imp_pre = load_importances(Path(p_pre))
            imp_post = load_importances(Path(p_post))
            feats = sorted(set(imp_pre) | set(imp_post),
                           key=lambda f: imp_post.get(f, 0.0), reverse=True)
            print("\n=== Feature-importance side-by-side (gain) ===")
            print(f"{'feature':<24} {'R_pre':>12} {'R_post':>12}")
            print("-" * 50)
            for f in feats:
                print(f"{f:<24} {imp_pre.get(f, 0.0):>12.2f} {imp_post.get(f, 0.0):>12.2f}")
            importance_table = {
                "R_pre": imp_pre, "R_post": imp_post, "order": feats}
        except (FileNotFoundError, ValueError) as e:
            print(f"\n[2x2] importances skipped: {e}")

    # --- JSON dump ---
    dump = {
        "config": {
            "results_dir": str(rdir),
            "arms_dir": str(arms),
            "conv": args.conv,
            "cells": cells,
            "mode": mode,
            "n_common": len(common),
            "n_boot": args.n_boot,
            "seed": args.seed,
            "recall_budget": args.recall_budget,
            "session_k": args.session_k,
            "metrics": METRICS,
            "have_undormant": have_undormant,
        },
        "cell_means": table,
        "contrasts": contrasts,
        "undormant_contrasts": undormant_contrasts,
        "importances": importance_table,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dump, indent=2))
    print(f"\n[2x2] wrote {out_path}")


if __name__ == "__main__":
    main()
