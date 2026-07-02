#!/usr/bin/env python3
"""Floor sweep for the proactive-injection floor study (arc step 3, artifact #2).

Reads floor_labels.json (from build_floor_labels.py). Sweeps a score floor,
computes precision / recall / F-beta(beta=2) against use/ignore labels, and the
two baselines (always-inject, never-inject). Also reports threshold-free
discrimination (AUC, point-biserial r) — the honest comparison to the cliff
study's R²<0 — and an isotonic calibration whose monotonicity makes the
F-beta-optimal floor invariant (documented in findings).

Populations (RRF-fallback era only; reranker-era stored scores are contaminated):
  A per_candidate_primary  kept candidates; used=utility>0, unrated=ignored (artifact literal)
  B per_candidate_explicit kept candidates that were explicitly rated; used=util>0 vs util==0
  C per_turn_topscore      one row/turn; feature=top kept score; label=turn had any used
  D full_range_censored    kept ∪ beyond_limit (beyond all ignored) — CENSORING-CONFOUNDED, illustrative only

Usage:
  python sweep_floor.py [--labels floor_labels.json] [--beyond floor_labels_beyond.jsonl]
"""
import argparse
import bisect
import json
import math
import os
import random

BETA = 2.0
N_BOOT = 1000


def fbeta(precision, recall, beta=BETA):
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def auc_mannwhitney(scores, labels):
    """AUC = P(score_pos > score_neg), tie-corrected via average ranks."""
    pairs = sorted(zip(scores, labels))
    n = len(pairs)
    npos = sum(labels)
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return None
    # average ranks (1-based) with tie handling
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0  # average of ranks i+1..j
        for k in range(i, j):
            ranks[k] = avg
        i = j
    sum_pos = sum(r for r, (_, lab) in zip(ranks, pairs) if lab)
    return (sum_pos - npos * (npos + 1) / 2.0) / (npos * nneg)


def point_biserial(scores, labels):
    n = len(scores)
    npos = sum(labels)
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return None
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / n
    sd = math.sqrt(var)
    if sd == 0:
        return None
    mpos = sum(s for s, l in zip(scores, labels) if l) / npos
    mneg = sum(s for s, l in zip(scores, labels) if not l) / nneg
    return (mpos - mneg) / sd * math.sqrt((npos * nneg) / (n * n))


def sweep(scores, labels, n_points=50):
    """Return curve + F-beta-optimal + baselines. surface = score >= floor."""
    n = len(scores)
    npos = sum(labels)
    base_rate = npos / n if n else None
    lo, hi = min(scores), max(scores)
    # candidate thresholds: 50-pt linspace over range, plus every distinct score
    grid = set()
    if hi > lo:
        for i in range(n_points + 1):
            grid.add(lo + (hi - lo) * i / n_points)
    grid.update(scores)
    grid.add(lo - 1e-9)          # always-inject
    grid.add(hi + 1e-9)          # never-inject
    order = sorted(zip(scores, labels))
    sc = [x[0] for x in order]
    lb = [x[1] for x in order]
    # suffix sums for score >= floor
    import bisect
    total_pos = npos
    curve = []
    for floor in sorted(grid):
        idx = bisect.bisect_left(sc, floor)      # first index with score >= floor
        surfaced = n - idx
        tp = sum(lb[idx:])                        # positives with score >= floor
        precision = (tp / surfaced) if surfaced else 1.0
        recall = (tp / total_pos) if total_pos else 0.0
        curve.append({
            "floor": floor, "surfaced": surfaced,
            "precision": precision, "recall": recall,
            "fbeta": fbeta(precision, recall),
        })
    # F-beta-optimal (exclude the two synthetic baseline floors from "floor found")
    interior = [c for c in curve if lo - 1e-9 < c["floor"] <= hi + 1e-9]
    best = max(curve, key=lambda c: c["fbeta"])
    always = min(curve, key=lambda c: c["floor"])   # floor = -inf
    never = max(curve, key=lambda c: c["floor"])    # floor = +inf
    return {
        "n": n, "n_pos": npos, "base_rate": base_rate,
        "score_min": lo, "score_max": hi,
        "auc": auc_mannwhitney(scores, labels),
        "point_biserial_r": point_biserial(scores, labels),
        "always_inject": {"precision": always["precision"], "recall": always["recall"],
                          "fbeta": always["fbeta"]},
        "never_inject": {"fbeta": never["fbeta"]},
        "fbeta_optimal": best,
        "beats_always": best["fbeta"] > always["fbeta"],
        "beats_never": best["fbeta"] > never["fbeta"],
        "curve": curve,
    }


def pava(x, y):
    """Isotonic regression (pool-adjacent-violators) of y on sorted-by-x.
    Returns list of (x_threshold, fitted_prob) blocks — monotone non-decreasing."""
    order = sorted(range(len(x)), key=lambda i: x[i])
    xs = [x[i] for i in order]
    ys = [float(y[i]) for i in order]
    # blocks: [sum, count, xmax]
    blocks = []
    for xi, yi in zip(xs, ys):
        blocks.append([yi, 1.0, xi])
        while len(blocks) >= 2 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s2, c2, _ = blocks.pop()
            blocks[-1][0] += s2
            blocks[-1][1] += c2
            blocks[-1][2] = xi
    return [(b[2], b[0] / b[1]) for b in blocks]


def bootstrap_delta_f2(scores, labels, n_boot=N_BOOT, n_grid=50, seed=0):
    """Bootstrap the F-beta improvement of the best refit floor over always-inject.
    Refits the optimal floor on each resample (accounts for selection), so the CI
    answers: is the 'a floor beats always-inject' win distinguishable from noise?"""
    rng = random.Random(seed)
    n = len(scores)
    lo, hi = min(scores), max(scores)
    grid = [lo - 1e-9] + [lo + (hi - lo) * i / n_grid for i in range(n_grid + 1)]
    idxs = list(range(n))
    deltas = []
    best_floors = []
    for _ in range(n_boot):
        samp = [idxs[rng.randrange(n)] for _ in range(n)]
        pairs = sorted((scores[i], labels[i]) for i in samp)
        sc = [p[0] for p in pairs]
        lb = [p[1] for p in pairs]
        tot = sum(lb)
        m = len(lb)
        if tot == 0:
            continue
        # always-inject F2
        f2_always = fbeta(tot / m, 1.0)
        # best over grid
        best = 0.0
        best_fl = None
        for floor in grid:
            j = bisect.bisect_left(sc, floor)
            surf = m - j
            tp = sum(lb[j:])
            p = (tp / surf) if surf else 1.0
            r = (tp / tot) if tot else 0.0
            f2 = fbeta(p, r)
            if f2 > best:
                best = f2
                best_fl = floor
        deltas.append(best - f2_always)
        best_floors.append(best_fl)
    deltas.sort()
    lo_ci = deltas[int(0.025 * len(deltas))]
    hi_ci = deltas[int(0.975 * len(deltas))]
    frac_pos = sum(1 for d in deltas if d > 1e-6) / len(deltas)
    bf = sorted(f for f in best_floors if f is not None)
    return {
        "delta_f2_median": deltas[len(deltas) // 2],
        "delta_f2_ci95": [lo_ci, hi_ci],
        "frac_resamples_delta_gt_0": frac_pos,
        "best_floor_median": bf[len(bf) // 2] if bf else None,
        "n_boot": len(deltas),
    }


def load_kept(path):
    d = json.load(open(path, encoding="utf-8"))
    rows = [r for r in d["labels_kept"] if r["path"] == "rrf"]
    return d["meta"], rows


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(__file__)
    ap.add_argument("--labels", default=os.path.join(here, "floor_labels.json"))
    ap.add_argument("--beyond", default=os.path.join(here, "floor_labels_beyond.jsonl"))
    ap.add_argument("--out", default=os.path.join(here, "floor_sweep.json"))
    args = ap.parse_args()

    meta, kept = load_kept(args.labels)
    results = {}

    # A. per-candidate primary: used=utility>0, unrated=ignored
    scores = [r["rrf_score"] for r in kept]
    labels = [1 if r["used"] else 0 for r in kept]
    results["A_per_candidate_primary"] = sweep(scores, labels)
    results["A_per_candidate_primary"]["bootstrap"] = bootstrap_delta_f2(scores, labels)

    # B. per-candidate explicit: only rated candidates
    rated = [r for r in kept if r["rated"]]
    scoresB = [r["rrf_score"] for r in rated]
    labelsB = [1 if r["used"] else 0 for r in rated]
    results["B_per_candidate_explicit"] = sweep(scoresB, labelsB)
    results["B_per_candidate_explicit"]["bootstrap"] = bootstrap_delta_f2(scoresB, labelsB)

    # C. per-turn: feature = top (rank-0) kept score; label = any kept used
    turns = {}
    for r in kept:
        t = turns.setdefault(r["turn_id"], {"top": None, "any_used": False})
        if r["rank"] == 0:
            t["top"] = r["rrf_score"]
        if r["used"]:
            t["any_used"] = True
    scoresC = [t["top"] for t in turns.values() if t["top"] is not None]
    labelsC = [1 if t["any_used"] else 0 for t in turns.values() if t["top"] is not None]
    results["C_per_turn_topscore"] = sweep(scoresC, labelsC)
    results["C_per_turn_topscore"]["bootstrap"] = bootstrap_delta_f2(scoresC, labelsC)

    # D. full-range (kept ∪ beyond), CENSORING-CONFOUNDED illustration
    if os.path.exists(args.beyond):
        scoresD = list(scores)
        labelsD = list(labels)
        nb = 0
        with open(args.beyond, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["path"] != "rrf":
                    continue
                scoresD.append(r["rrf_score"])
                labelsD.append(1 if r["used"] else 0)
                nb += 1
        res = sweep(scoresD, labelsD)
        res["note"] = ("CENSORING-CONFOUNDED: beyond_limit candidates were never shown, so "
                       "they are ignored by construction and score lower than kept. High AUC "
                       "here is largely mechanical, NOT evidence the floor works. Not the accept test.")
        res["n_beyond_added"] = nb
        results["D_full_range_censored"] = res

    # Calibration (Step 4): isotonic on population A; monotone => F-beta invariant.
    iso = pava(scores, labels)
    # map each score to its isotonic P(used) and re-sweep to demonstrate invariance
    import bisect
    xs_iso = [b[0] for b in iso]
    ps_iso = [b[1] for b in iso]
    def calibrate(s):
        i = bisect.bisect_left(xs_iso, s)
        if i >= len(ps_iso):
            i = len(ps_iso) - 1
        return ps_iso[i]
    cal_scores = [calibrate(s) for s in scores]
    cal_sweep = sweep(cal_scores, labels)
    results["A_calibrated_isotonic"] = {
        "n_iso_blocks": len(iso),
        "monotone": all(ps_iso[i] <= ps_iso[i + 1] for i in range(len(ps_iso) - 1)),
        "fbeta_optimal_fbeta": cal_sweep["fbeta_optimal"]["fbeta"],
        "auc": cal_sweep["auc"],
        "note": ("Isotonic is monotone, so surface sets (score>=floor) are order-identical to "
                 "(calibrated>=floor'); the P/R/F-beta curve and its optimum are UNCHANGED. "
                 "Calibration only makes the threshold interpretable, it cannot change the verdict."),
        "iso_blocks_sample": iso[:: max(1, len(iso) // 12)][:12],
    }

    out = {
        "beta": BETA,
        "source_meta": {k: meta[k] for k in ("match_window_seconds", "reranker_outage_cut",
                        "last_reranker_signature_date", "join_stats", "score_provenance")},
        "results": results,
    }
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2)

    # console
    def show(name, r):
        ai = r["always_inject"]; opt = r["fbeta_optimal"]
        print(f"\n[{name}] n={r['n']} pos={r['n_pos']} base={r['base_rate']:.4f} "
              f"AUC={r['auc']:.4f} r_pb={r['point_biserial_r']:.4f}")
        print(f"   always-inject: P={ai['precision']:.4f} R={ai['recall']:.4f} F2={ai['fbeta']:.4f}")
        print(f"   never-inject:  F2={r['never_inject']['fbeta']:.4f}")
        print(f"   F2-optimal floor={opt['floor']:.5f} surfaced={opt['surfaced']} "
              f"P={opt['precision']:.4f} R={opt['recall']:.4f} F2={opt['fbeta']:.4f}")
        print(f"   beats_always={r['beats_always']} beats_never={r['beats_never']}  "
              f"=> ACCEPT={r['beats_always'] and r['beats_never']}")
        if "bootstrap" in r:
            b = r["bootstrap"]
            print(f"   bootstrap dF2 median={b['delta_f2_median']:.5f} "
                  f"CI95=[{b['delta_f2_ci95'][0]:.5f},{b['delta_f2_ci95'][1]:.5f}] "
                  f"frac(dF2>0)={b['frac_resamples_delta_gt_0']:.3f}")
    for name in ("A_per_candidate_primary", "B_per_candidate_explicit",
                 "C_per_turn_topscore", "D_full_range_censored"):
        if name in results:
            show(name, results[name])
    c = results["A_calibrated_isotonic"]
    print(f"\n[A_calibrated_isotonic] monotone={c['monotone']} "
          f"F2_opt={c['fbeta_optimal_fbeta']:.4f} AUC={c['auc']:.4f} (invariant vs raw)")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
