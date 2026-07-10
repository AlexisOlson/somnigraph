#!/usr/bin/env python3
"""WP6 Stage-B structure readout -- experiments/sleep-bench/analyze_wp6.py

Reads the Stage-A sweep JSONL (`sweep_2x2_features.py` output) and extracts the
feature *structure* the random-subset design was built to expose: which toggleable
features carry marginal CV value, which pairs interact, and -- the headline of the
prereg -- which features substitute for (share a credit pool with) or complement
each other, read from importance INSTABILITY across masks rather than any single
model's unreliable within-model gain.

Four sections (all on train-side CV; these rank feature SETS, they never compete
with the conv0 headline cells -- see wp6-prereg.md guardrails):

  1. Marginal value table -- per toggleable, include-vs-exclude delta on mean CV
     NDCG@5k / R@10 / MRR, with a bootstrap CI over masks (resample masks,
     n_boot=10000, seed=1234). n_in / n_out reported per feature.
  2. Mask regression -- OLS of mean NDCG on the 12 inclusion indicators, then a
     second fit adding the top-k pairwise interaction terms selected by
     |corr(product term, main-effects residual)|. (Choice justified in-code below.)
  3. Importance-instability map -- per ordered pair (f, g): mean gain of f across
     masks where g is ON vs OFF (f ON in both). Substitution = gain of f JUMPS when
     g is masked (they share a credit pool); complement = gain of f COLLAPSES when g
     is masked. Plus the always-on core features' instability vs the graph toggles
     (H-B's theme_rank read).
  4. Hypotheses -- the three pre-registered decision rules (H-A/H-B/H-C) computed
     verbatim. Where a needed pool-out / knockout / graph-out mask is ABSENT from a
     random sweep, the rule is not approximated: the exact Stage-C mask spec is
     emitted instead (and `--stage-c` prints the ready-to-run sweep command).

Output: human-readable report to stdout + `results-full/wp6_structure.json`.

Usage:
    uv run experiments/sleep-bench/analyze_wp6.py                 # default sweep JSONL
    uv run experiments/sleep-bench/analyze_wp6.py --sweep <path> --stage-c
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sweep_2x2_features import (  # noqa: E402
    CORE,
    DEFAULT_OUT as DEFAULT_SWEEP,
    TOGGLEABLE,
    mask_id,
)

# Feature clusters used by the pre-registered hypotheses (wp6-prereg.md).
POOL = ["member_count", "priority", "token_count", "theme_count", "is_summary"]
GRAPH = ["edge_count", "betweenness", "diversity_score", "ppr_score"]
RESULTS_DIR = HERE / "results-full"
N_BOOT = 10000
BOOT_SEED = 1234


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_sweep(path: Path) -> list[dict]:
    recs = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    # De-dup by mask_id (defensive; resume should already guarantee uniqueness).
    seen, out = set(), []
    for r in recs:
        if r["mask_id"] not in seen:
            seen.add(r["mask_id"])
            out.append(r)
    return out


def _on(rec, feat) -> bool:
    """Is `feat` present in this mask's model? Core is always on; toggleables vary."""
    return feat in CORE or feat in rec["toggleables_on"]


# ---------------------------------------------------------------------------
# (1) Marginal value table
# ---------------------------------------------------------------------------


def marginal_value(recs, metrics=("ndcg_5k", "recall_10", "mrr")):
    rng = np.random.default_rng(BOOT_SEED)
    out = {}
    for f in TOGGLEABLE:
        in_mask = np.array([f in r["toggleables_on"] for r in recs])
        row = {"n_in": int(in_mask.sum()), "n_out": int((~in_mask).sum())}
        for metric in metrics:
            vals = np.array([r["mean"][metric] for r in recs], dtype=float)
            in_v, out_v = vals[in_mask], vals[~in_mask]
            if len(in_v) == 0 or len(out_v) == 0:
                row[metric] = {"delta": None, "ci": [None, None]}
                continue
            delta = float(in_v.mean() - out_v.mean())
            # Bootstrap over masks: resample the full population, re-split, re-delta.
            n = len(vals)
            boots = np.empty(N_BOOT)
            for b in range(N_BOOT):
                idx = rng.integers(0, n, n)
                bm = in_mask[idx]
                if bm.all() or (~bm).all():
                    boots[b] = np.nan
                    continue
                boots[b] = vals[idx][bm].mean() - vals[idx][~bm].mean()
            boots = boots[~np.isnan(boots)]
            lo, hi = (float(np.percentile(boots, 2.5)),
                      float(np.percentile(boots, 97.5))) if len(boots) else (None, None)
            row[metric] = {"delta": delta, "ci": [lo, hi]}
        out[f] = row
    return out


# ---------------------------------------------------------------------------
# (2) Mask regression (main effects + interactions)
# ---------------------------------------------------------------------------


def _ols(X, y):
    """OLS via lstsq. X already includes an intercept column. Returns coefs."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def mask_regression(recs, top_k=10):
    """OLS of mean NDCG@5k on 12 inclusion indicators, then a second fit adding
    the top-k pairwise interactions.

    Interaction selection = top-k by |corr(product term, main-effects residual)|
    rather than lasso-over-all-66. Justification: with ~130 masks and 12 main
    effects, a full 66-interaction design (78 params) is near-saturated and
    lasso would need a nested CV for lambda, importing tuning noise into a
    descriptive readout; ranking product terms by their correlation with the
    main-effects residual directly targets the pairs the additive model misfits
    and keeps the second OLS well-conditioned."""
    n = len(recs)
    ind = np.array([[1.0 if (f in r["toggleables_on"]) else 0.0 for f in TOGGLEABLE]
                    for r in recs])
    y = np.array([r["mean"]["ndcg_5k"] for r in recs], dtype=float)

    X1 = np.hstack([np.ones((n, 1)), ind])
    coef1 = _ols(X1, y)
    main = {"intercept": float(coef1[0])}
    for i, f in enumerate(TOGGLEABLE):
        main[f] = float(coef1[i + 1])
    resid = y - X1 @ coef1

    # Candidate interaction pairs ranked by |corr with residual|.
    pairs = []
    for i in range(len(TOGGLEABLE)):
        for j in range(i + 1, len(TOGGLEABLE)):
            prod = ind[:, i] * ind[:, j]
            if prod.std() < 1e-12 or resid.std() < 1e-12:
                corr = 0.0
            else:
                corr = float(np.corrcoef(prod, resid)[0, 1])
            pairs.append((TOGGLEABLE[i], TOGGLEABLE[j], corr))
    pairs.sort(key=lambda t: -abs(t[2]))
    chosen = pairs[:top_k]

    inter_cols = [ind[:, TOGGLEABLE.index(a)] * ind[:, TOGGLEABLE.index(b)]
                  for a, b, _ in chosen]
    X2 = np.hstack([X1] + [c.reshape(-1, 1) for c in inter_cols]) if inter_cols else X1
    coef2 = _ols(X2, y)
    inter = {}
    base = X1.shape[1]
    for k, (a, b, corr) in enumerate(chosen):
        inter[f"{a}*{b}"] = {"coef": float(coef2[base + k]), "resid_corr": corr}

    main_ranked = sorted(
        ((f, main[f]) for f in TOGGLEABLE), key=lambda t: -abs(t[1])
    )
    inter_ranked = sorted(inter.items(), key=lambda t: -abs(t[1]["coef"]))
    return {
        "main_effects": main,
        "main_ranked": main_ranked,
        "interactions": inter,
        "interactions_ranked": inter_ranked,
        "note": "top_k interactions by |corr(product, main-effects residual)|",
    }


# ---------------------------------------------------------------------------
# (3) Importance-instability map
# ---------------------------------------------------------------------------


def instability_map(recs, min_support=5, jump=1.5):
    """For ordered (f, g): mean gain of f in masks where g ON vs OFF, requiring f ON
    in both. ratio = gain_f(g_off) / gain_f(g_on).
       ratio high  -> f's gain JUMPS when g is masked  -> substitution / shared pool
       ratio low   -> f's gain COLLAPSES when g masked -> complement (needs partner)
    `f` may be any feature (core always-on features included); `g` is a toggleable."""
    all_feats = list(CORE) + list(TOGGLEABLE)
    substitutions, complements, raw = [], [], {}
    for f in all_feats:
        for g in TOGGLEABLE:
            if f == g:
                continue
            g_on_gains, g_off_gains = [], []
            for r in recs:
                if not _on(r, f):
                    continue
                gain = r["importances_gain"].get(f)
                if gain is None:
                    continue
                if g in r["toggleables_on"]:
                    g_on_gains.append(gain)
                else:
                    g_off_gains.append(gain)
            n_on, n_off = len(g_on_gains), len(g_off_gains)
            if n_on < min_support or n_off < min_support:
                continue
            m_on = float(np.mean(g_on_gains))
            m_off = float(np.mean(g_off_gains))
            denom = m_on if m_on > 1e-9 else 1e-9
            ratio = m_off / denom
            entry = {
                "f": f, "g": g, "gain_g_on": m_on, "gain_g_off": m_off,
                "ratio_off_over_on": ratio, "n_g_on": n_on, "n_g_off": n_off,
            }
            raw[f"{f}|{g}"] = entry
            if ratio >= jump:
                substitutions.append(entry)
            elif ratio <= 1.0 / jump:
                complements.append(entry)
    substitutions.sort(key=lambda e: -e["ratio_off_over_on"])
    complements.sort(key=lambda e: e["ratio_off_over_on"])

    # Core-feature instability vs the graph toggles (H-B's theme_rank read).
    core_vs_graph = {}
    for f in CORE:
        per_g = {}
        for g in GRAPH:
            e = raw.get(f"{f}|{g}")
            if e:
                per_g[g] = {"gain_g_on": e["gain_g_on"], "gain_g_off": e["gain_g_off"],
                            "ratio_off_over_on": e["ratio_off_over_on"],
                            "n_on": e["n_g_on"], "n_off": e["n_g_off"]}
        core_vs_graph[f] = per_g
    return {
        "substitutions": substitutions,
        "complements": complements,
        "core_vs_graph": core_vs_graph,
        "params": {"min_support": min_support, "jump": jump},
    }


# ---------------------------------------------------------------------------
# (4) Hypotheses (pre-registered decision rules, verbatim)
# ---------------------------------------------------------------------------


def required_masks() -> dict:
    """Named -> toggleables_on for every mask the hypotheses need."""
    req = {
        "all_on": list(TOGGLEABLE),
        "core_only": [],
        "pool_out": [t for t in TOGGLEABLE if t not in POOL],
        "graph_out": [t for t in TOGGLEABLE if t not in GRAPH],
        "graph_only": list(GRAPH),
    }
    for m in POOL:
        req[f"knockout_{m}"] = [t for t in TOGGLEABLE if t != m]
    return req


def _by_id(recs):
    return {r["mask_id"]: r for r in recs}


def _paired_fold_delta(recA, recB, metric="ndcg_5k"):
    """Paired per-fold delta A-B + bootstrap CI over the (few) folds."""
    a = np.array([f[metric] for f in recA["per_fold"]], dtype=float)
    b = np.array([f[metric] for f in recB["per_fold"]], dtype=float)
    d = a - b
    rng = np.random.default_rng(BOOT_SEED)
    nb = min(N_BOOT, 5000)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(nb)])
    return {
        "mean_delta": float(d.mean()),
        "per_fold_delta": d.tolist(),
        "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "n_folds": len(d),
    }


def hypotheses(recs):
    idx = _by_id(recs)
    req = required_masks()
    present, missing = {}, {}
    for name, toggs in req.items():
        mid = mask_id(toggs)
        if mid in idx:
            present[name] = idx[mid]
        else:
            missing[name] = {"mask_id": mid, "toggleables_on": sorted(toggs),
                             "extra_mask_arg": f"{name}=" + ",".join(sorted(toggs))}

    out = {"present": sorted(present), "missing": missing, "results": {}}

    # H-A: pool-out vs all-on paired delta CI excludes 0 while single knockouts don't.
    ha = {"rule": "pool_out vs all_on paired CV NDCG delta CI excludes 0 while single "
                  "pool-member knockouts do not"}
    need_ha = ["all_on", "pool_out"] + [f"knockout_{m}" for m in POOL]
    miss_ha = [n for n in need_ha if n in missing]
    if miss_ha:
        ha["status"] = "MASKS MISSING -- run Stage C"
        ha["missing"] = {n: missing[n]["extra_mask_arg"] for n in miss_ha}
    else:
        ha["status"] = "computable"
        ha["pool_out_vs_all_on"] = _paired_fold_delta(present["pool_out"], present["all_on"])
        ha["knockouts"] = {
            m: _paired_fold_delta(present[f"knockout_{m}"], present["all_on"])
            for m in POOL
        }
        pool_ci = ha["pool_out_vs_all_on"]["ci"]
        pool_excludes = not (pool_ci[0] <= 0 <= pool_ci[1])
        knock_excludes = {m: not (ha["knockouts"][m]["ci"][0] <= 0 <= ha["knockouts"][m]["ci"][1])
                          for m in POOL}
        ha["verdict"] = {
            "pool_out_CI_excludes_0": pool_excludes,
            "knockout_CIs_exclude_0": knock_excludes,
            "H_A_supported": pool_excludes and not any(knock_excludes.values()),
        }
    out["results"]["H_A"] = ha

    # H-B: theme_rank mean gain across masks graph-off >= 3x its gain graph-on.
    hb = {"rule": "theme_rank mean gain, graph-off masks (all 4 graph toggles OFF) "
                  ">= 3x graph-on masks (all 4 ON); qualitative"}
    goff, gon = [], []
    for r in recs:
        gr = [g in r["toggleables_on"] for g in GRAPH]
        gain = r["importances_gain"].get("theme_rank")
        if gain is None:
            continue
        if not any(gr):
            goff.append(gain)
        elif all(gr):
            gon.append(gain)
    hb["n_graph_off"] = len(goff)
    hb["n_graph_on"] = len(gon)
    if not goff or not gon:
        hb["status"] = "INSUFFICIENT SUPPORT in random sweep"
        hb["stage_c_masks"] = {
            "graph_off_extreme": "graph_off_extreme=" + ",".join(
                sorted(t for t in TOGGLEABLE if t not in GRAPH)),
            "graph_on_extreme": "graph_on_extreme=" + ",".join(sorted(GRAPH)),
        }
        hb["note"] = ("emit these two masks (and their intermediates) for a clean "
                      "graph-off vs graph-on theme_rank read")
    else:
        m_off, m_on = float(np.mean(goff)), float(np.mean(gon))
        ratio = m_off / (m_on if m_on > 1e-9 else 1e-9)
        hb["status"] = "computable"
        hb["theme_rank_gain_graph_off"] = m_off
        hb["theme_rank_gain_graph_on"] = m_on
        hb["ratio"] = ratio
        hb["H_B_supported"] = ratio >= 3.0
    out["results"]["H_B"] = hb

    # H-C: delta(graph-out) > delta(pool-out); graph-only retains >= 60% of
    #      (all-on - core-only).
    hc = {"rule": "paired delta(graph_out) > delta(pool_out) AND graph_only retains "
                  ">= 60% of (all_on - core_only) on mean NDCG@5k"}
    need_hc = ["all_on", "core_only", "pool_out", "graph_out", "graph_only"]
    miss_hc = [n for n in need_hc if n in missing]
    if miss_hc:
        hc["status"] = "MASKS MISSING -- run Stage C"
        hc["missing"] = {n: missing[n]["extra_mask_arg"] for n in miss_hc}
    else:
        hc["status"] = "computable"
        d_graph = _paired_fold_delta(present["all_on"], present["graph_out"])
        d_pool = _paired_fold_delta(present["all_on"], present["pool_out"])
        allon = present["all_on"]["mean"]["ndcg_5k"]
        coreonly = present["core_only"]["mean"]["ndcg_5k"]
        graphonly = present["graph_only"]["mean"]["ndcg_5k"]
        span = allon - coreonly
        retained = (graphonly - coreonly) / span if abs(span) > 1e-12 else None
        hc["delta_graph_out"] = d_graph["mean_delta"]
        hc["delta_pool_out"] = d_pool["mean_delta"]
        hc["retained_fraction"] = retained
        hc["H_C_supported"] = (
            d_graph["mean_delta"] > d_pool["mean_delta"]
            and retained is not None and retained >= 0.60
        )
    out["results"]["H_C"] = hc
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_ci(ci):
    if ci is None or ci[0] is None:
        return "[  n/a  ]"
    return f"[{ci[0]:+.4f},{ci[1]:+.4f}]"


def print_report(recs, marg, reg, inst, hyp, stage_c):
    print("=" * 78)
    print(f"WP6 STRUCTURE READOUT  ({len(recs)} masks)")
    print("=" * 78)

    print("\n--- (1) MARGINAL VALUE (include - exclude, mean CV; bootstrap CI over masks) ---")
    print(f"{'feature':22s} {'nIn/nOut':>9s}  {'dNDCG':>8s} {'CI(NDCG)':>19s}  "
          f"{'dR@10':>8s}  {'dMRR':>8s}")
    order = sorted(TOGGLEABLE, key=lambda f: -(marg[f]['ndcg_5k']['delta'] or 0))
    for f in order:
        m = marg[f]
        nd, r10, mrr = m["ndcg_5k"], m["recall_10"], m["mrr"]
        print(f"{f:22s} {m['n_in']:4d}/{m['n_out']:<4d}  "
              f"{(nd['delta'] or 0):+8.4f} {_fmt_ci(nd['ci']):>19s}  "
              f"{(r10['delta'] or 0):+8.4f}  {(mrr['delta'] or 0):+8.4f}")

    print("\n--- (2) MASK REGRESSION: main effects on mean NDCG@5k (ranked |coef|) ---")
    for f, c in reg["main_ranked"]:
        print(f"   {f:22s} {c:+.5f}")
    print(f"   [intercept {reg['main_effects']['intercept']:+.5f}]")
    print(f"   interactions ({reg['note']}):")
    for name, d in reg["interactions_ranked"]:
        print(f"      {name:34s} coef={d['coef']:+.5f}  (resid_corr={d['resid_corr']:+.3f})")

    print("\n--- (3) IMPORTANCE INSTABILITY (f gain: g_off/g_on ratio) ---")
    print(f"   params: {inst['params']}")
    print("   SUBSTITUTION candidates (f gain jumps when g masked -> shared pool):")
    for e in inst["substitutions"][:12]:
        print(f"      f={e['f']:20s} g={e['g']:20s} ratio={e['ratio_off_over_on']:6.2f} "
              f"(on={e['gain_g_on']:.1f} off={e['gain_g_off']:.1f}; n_on={e['n_g_on']} n_off={e['n_g_off']})")
    if not inst["substitutions"]:
        print("      (none above threshold with sufficient support)")
    print("   COMPLEMENT candidates (f gain collapses when g masked -> needs partner):")
    for e in inst["complements"][:12]:
        print(f"      f={e['f']:20s} g={e['g']:20s} ratio={e['ratio_off_over_on']:6.2f} "
              f"(on={e['gain_g_on']:.1f} off={e['gain_g_off']:.1f}; n_on={e['n_g_on']} n_off={e['n_g_off']})")
    if not inst["complements"]:
        print("      (none below threshold with sufficient support)")
    print("   theme_rank (core) vs graph toggles [H-B read]:")
    tr = inst["core_vs_graph"].get("theme_rank", {})
    if tr:
        for g, d in tr.items():
            print(f"      g={g:18s} gain g_on={d['gain_g_on']:.2f} g_off={d['gain_g_off']:.2f} "
                  f"ratio={d['ratio_off_over_on']:.2f} (n_on={d['n_on']} n_off={d['n_off']})")
    else:
        print("      (insufficient support -- theme_rank/graph pairs below min_support)")

    print("\n--- (4) HYPOTHESES (pre-registered decision rules) ---")
    for h in ("H_A", "H_B", "H_C"):
        r = hyp["results"][h]
        print(f"\n   [{h}] {r['rule']}")
        print(f"       status: {r['status']}")
        if r["status"].startswith("MASKS MISSING") or r["status"].startswith("INSUFF"):
            spec = r.get("missing") or r.get("stage_c_masks", {})
            for name, arg in spec.items():
                print(f"         MISSING {name}: --extra-mask {arg}")
        else:
            for k, v in r.items():
                if k in ("rule", "status"):
                    continue
                print(f"       {k}: {v}")

    if hyp["missing"]:
        print("\n--- STAGE-C: masks absent from this sweep ---")
        for name, d in hyp["missing"].items():
            print(f"   {name:22s} --extra-mask {d['extra_mask_arg']}")
        if stage_c:
            args = " ".join(f'--extra-mask {d["extra_mask_arg"]}'
                            for d in hyp["missing"].values())
            print("\n   READY-TO-RUN Stage-C sweep (appends to the same JSONL):")
            print(f"   uv run --extra tuning experiments/sleep-bench/sweep_2x2_features.py \\")
            print(f"       --n-masks 0 {args}")
    else:
        print("\n--- STAGE-C: all pre-registered hypothesis masks already present ---")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sweep", default=str(DEFAULT_SWEEP))
    ap.add_argument("--out", default=str(RESULTS_DIR / "wp6_structure.json"))
    ap.add_argument("--top-k-interactions", type=int, default=10)
    ap.add_argument("--min-support", type=int, default=5,
                    help="min masks per subset for an instability pair (default 5)")
    ap.add_argument("--stage-c", action="store_true",
                    help="emit the ready-to-run sweep command for missing hypothesis masks")
    args = ap.parse_args()

    sweep_path = Path(args.sweep).expanduser().resolve()
    if not sweep_path.exists():
        raise SystemExit(f"[analyze_wp6] sweep JSONL not found: {sweep_path}")
    recs = load_sweep(sweep_path)
    if not recs:
        raise SystemExit("[analyze_wp6] no records in sweep JSONL")
    print(f"[analyze_wp6] loaded {len(recs)} masks from {sweep_path}")

    marg = marginal_value(recs)
    reg = mask_regression(recs, top_k=args.top_k_interactions)
    inst = instability_map(recs, min_support=args.min_support)
    hyp = hypotheses(recs)

    print_report(recs, marg, reg, inst, hyp, args.stage_c)

    dump = {
        "n_masks": len(recs),
        "sweep_path": str(sweep_path),
        "marginal_value": marg,
        "mask_regression": {
            "main_effects": reg["main_effects"],
            "main_ranked": reg["main_ranked"],
            "interactions": reg["interactions"],
            "note": reg["note"],
        },
        "instability_map": inst,
        "hypotheses": hyp,
    }
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dump, indent=2), encoding="utf-8")
    print(f"\n[analyze_wp6] structure dump -> {out_path}")


if __name__ == "__main__":
    main()
