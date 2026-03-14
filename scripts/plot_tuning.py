# /// script
# requires-python = ">=3.11"
# dependencies = ["optuna>=4.0", "numpy>=1.26", "matplotlib>=3.8", "scikit-learn>=1.0"]
# ///
"""
Chess-tuning-tools style corner plots for Optuna tuning studies.

Handles discrete (integer with few levels) vs continuous params:
- Diagonal: bar chart for discrete, smooth curve for continuous
- Off-diagonal: heatmap bands for discrete axes, contours for continuous pairs

Usage:
  uv run scripts/plot_tuning.py                          # plot all studies
  uv run scripts/plot_tuning.py --study group1_core_t0.5 # plot one study
  uv run scripts/plot_tuning.py --output plots/          # save to directory
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from memory.constants import DATA_DIR

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.interpolate import RBFInterpolator

STUDY_DIR = DATA_DIR / "tuning_studies"

DARK_BG = "#3a3a3a"
PANEL_BG = "#2a2a2a"
CURVE_COLOR = "#e8d44d"
BEST_LINE_COLOR = "#ff6b35"
BEST_DOT_COLOR = "#ff4500"

# A param is "discrete" if it's integer-valued with <= this many unique levels
DISCRETE_MAX_LEVELS = 15


def load_study_rdb(db_path: Path, study_name: str, metric="mrr"):
    """Load an Optuna RDB storage study and return trial data.

    Same return format as load_study() for compatibility with plot_corner().
    """
    import optuna
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{db_path}",
        engine_kwargs={"connect_args": {"timeout": 30}},
    )
    study = optuna.load_study(study_name=study_name, storage=storage)
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    return _extract_trial_data(trials, study.study_name, metric)


def _extract_trial_data(trials, study_name, metric="mrr"):
    """Common extraction logic for both journal and RDB storage."""
    if not trials:
        return None

    if metric != "mrr":
        trials = [t for t in trials if metric in t.user_attrs]
        if not trials:
            return None

    param_names = list(trials[0].params.keys())
    params = {n: np.array([t.params[n] for t in trials]) for n in param_names}
    if metric == "mrr":
        values = np.array([t.value for t in trials])
    else:
        values = np.array([t.user_attrs[metric] for t in trials])

    # Extract search space bounds from distributions
    bounds = {}
    discrete = set()
    for n in param_names:
        dist = trials[0].distributions[n]
        bounds[n] = (dist.low, dist.high)
        if hasattr(dist, 'step') and isinstance(dist.low, int):
            step = getattr(dist, 'step', 1) or 1
            n_levels = (dist.high - dist.low) // step + 1
            if n_levels <= DISCRETE_MAX_LEVELS:
                discrete.add(n)

    log_scale = set()
    for n in param_names:
        lo, hi = bounds[n]
        if lo >= 1 and hi / lo >= 50:
            log_scale.add(n)

    minimize = metric != "mrr"
    best_idx = int(np.argmin(values) if minimize else np.argmax(values))
    best_params = {n: params[n][best_idx] for n in param_names}

    return dict(
        param_names=param_names, params=params, values=values,
        study_name=study_name, best_params=best_params,
        discrete=discrete, bounds=bounds, log_scale=log_scale,
        minimize=minimize, metric=metric,
    )


def load_study(log_path: Path, metric="mrr"):
    """Load an Optuna journal log and return trial data."""
    import optuna
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(
            str(log_path),
            lock_obj=optuna.storages.journal.JournalFileOpenLock(str(log_path)),
        )
    )
    study = optuna.load_study(study_name=None, storage=storage)
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    return _extract_trial_data(trials, study.study_name, metric)


PD_MAX_POINTS = 150  # Subsample for partial dependence (RBF scales ~O(n**2))


def _build_rbf(params, values, param_names, bounds, log_scale=None):
    """Build a normalized RBF interpolator over all params.

    Normalizes using search space bounds (not observed min/max) so the
    full search space maps to [0, 1] and extrapolation is well-behaved.
    Log-scale params are transformed to log-space before normalization.
    When n_trials > PD_MAX_POINTS, subsamples for faster PD computation.
    """
    log_scale = log_scale or set()
    cols = []
    for n in param_names:
        col = params[n].astype(float)
        if n in log_scale:
            col = np.log(col)
        cols.append(col)
    X = np.column_stack(cols)

    X_min = np.array([np.log(bounds[n][0]) if n in log_scale else bounds[n][0]
                       for n in param_names], dtype=float)
    X_max = np.array([np.log(bounds[n][1]) if n in log_scale else bounds[n][1]
                       for n in param_names], dtype=float)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1
    X_norm = (X - X_min) / X_range

    # Subsample for PD averaging if too many points
    if len(X_norm) > PD_MAX_POINTS:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_norm), PD_MAX_POINTS, replace=False)
        X_pd = X_norm[idx]
    else:
        X_pd = X_norm

    rbf = RBFInterpolator(X_norm, values, kernel="thin_plate_spline", smoothing=1.0)
    return rbf, X_pd, X_min, X_range


def _to_norm(val, idx, X_min, X_range, param_names, log_scale):
    """Convert a raw param value to normalized RBF space."""
    if param_names[idx] in log_scale:
        return (np.log(val) - X_min[idx]) / X_range[idx]
    return (val - X_min[idx]) / X_range[idx]


def _pd_at_values(rbf, X_norm, X_min, X_range, idx, grid_vals,
                   param_names=None, log_scale=None):
    """Compute partial dependence at specific values for one param."""
    log_scale = log_scale or set()
    param_names = param_names or []
    pd = np.zeros(len(grid_vals))
    for i, g in enumerate(grid_vals):
        Xc = X_norm.copy()
        Xc[:, idx] = _to_norm(g, idx, X_min, X_range, param_names, log_scale)
        pd[i] = rbf(Xc).mean()
    return pd


def _pd_2d(rbf, X_norm, X_min, X_range, idx_x, idx_y, gx, gy,
            param_names=None, log_scale=None):
    """Compute 2D partial dependence on a grid."""
    log_scale = log_scale or set()
    param_names = param_names or []
    Z = np.zeros((len(gy), len(gx)))
    for i, yv in enumerate(gy):
        for j, xv in enumerate(gx):
            Xc = X_norm.copy()
            Xc[:, idx_x] = _to_norm(xv, idx_x, X_min, X_range, param_names, log_scale)
            Xc[:, idx_y] = _to_norm(yv, idx_y, X_min, X_range, param_names, log_scale)
            Z[i, j] = rbf(Xc).mean()
    return Z


def _style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_linewidth(0.5)


def _edges(levels):
    """Compute cell edges for pcolormesh from cell centers."""
    levels = np.asarray(levels, dtype=float)
    if len(levels) == 1:
        return np.array([levels[0] - 0.5, levels[0] + 0.5])
    gaps = np.diff(levels)
    edges = np.empty(len(levels) + 1)
    edges[0] = levels[0] - gaps[0] / 2
    edges[-1] = levels[-1] + gaps[-1] / 2
    edges[1:-1] = (levels[:-1] + levels[1:]) / 2
    return edges


def _format_val(v):
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if abs(v) >= 10:
        return f"{v:.1f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.4f}"


def _make_grid(name, bounds, log_scale, n_points=50):
    """Generate a grid: geomspace for log-scale, linspace otherwise."""
    lo, hi = bounds[name]
    if name in log_scale:
        return np.geomspace(lo, hi, n_points)
    return np.linspace(lo, hi, n_points)


def plot_corner(data, output_dir=None):
    """Create chess-tuning-tools style corner plot."""
    param_names = data["param_names"]
    params = data["params"]
    values = data["values"]
    best = data["best_params"]
    discrete = data["discrete"]
    bounds = data["bounds"]
    log_scale = data.get("log_scale", set())
    minimize = data.get("minimize", False)
    metric = data.get("metric", "mrr")
    study_name = data["study_name"]
    n = len(param_names)
    # For minimize metrics (miss rate), reverse colormap so dark = good
    cmap = "viridis_r" if minimize else "viridis"

    if n < 2:
        print(f"  Skipping {study_name}: only {n} parameter(s)")
        return

    rbf, X_norm, X_min, X_range = _build_rbf(
        params, values, param_names, bounds, log_scale)
    # Common kwargs for PD functions
    pd_kw = dict(param_names=param_names, log_scale=log_scale)

    fig, axes = plt.subplots(n, n, figsize=(3.0 * n, 3.0 * n), facecolor=DARK_BG)
    axes = np.atleast_2d(axes)

    # Hide upper triangle
    for i in range(n):
        for j in range(n):
            if j > i:
                axes[i, j].set_visible(False)

    # --- Diagonal: 1D partial dependence ---
    for i, name in enumerate(param_names):
        ax = axes[i, i]
        _style_ax(ax)
        idx = param_names.index(name)
        best_val = best[name]
        is_log = name in log_scale

        if name in discrete:
            # Bar chart at each discrete level (full search space range)
            lo_d, hi_d = int(bounds[name][0]), int(bounds[name][1])
            levels = np.arange(lo_d, hi_d + 1)
            pd_vals = _pd_at_values(rbf, X_norm, X_min, X_range, idx, levels,
                                     **pd_kw)
            bar_width = (levels[-1] - levels[0]) / (len(levels) * 1.5) if len(levels) > 1 else 0.4
            bars = ax.bar(levels, pd_vals, width=bar_width, color=CURVE_COLOR,
                          edgecolor=CURVE_COLOR, alpha=0.85, zorder=2)
            # Highlight best bar
            for lev, bar in zip(levels, bars):
                if lev == best_val:
                    bar.set_edgecolor(BEST_LINE_COLOR)
                    bar.set_linewidth(2)
            ax.axvline(best_val, color=BEST_LINE_COLOR, linestyle="--", alpha=0.6, linewidth=1)
            ax.set_xticks(levels)
            # Zoom y-axis to PD range so differences are visible
            pd_range = pd_vals.max() - pd_vals.min()
            if pd_range > 0:
                margin = max(pd_range * 0.3, 1e-4)
                ax.set_ylim(pd_vals.min() - margin, pd_vals.max() + margin)
        else:
            # Smooth curve
            grid = _make_grid(name, bounds, log_scale, 50)
            pd_vals = _pd_at_values(rbf, X_norm, X_min, X_range, idx, grid,
                                     **pd_kw)
            ax.plot(grid, pd_vals, color=CURVE_COLOR, linewidth=2)
            ax.axvline(best_val, color=BEST_LINE_COLOR, linestyle="--", alpha=0.8, linewidth=1)
            if is_log:
                ax.set_xscale("log")

        # Best value label
        ax.text(best_val, 0.85, _format_val(best_val), color=BEST_LINE_COLOR,
                fontsize=9, fontweight="bold", ha="left",
                transform=ax.get_xaxis_transform())

        # Axis setup
        if i == 0:
            ax.set_title(name, fontsize=10, color="white", pad=8)
            ax.xaxis.set_label_position("top")
            ax.xaxis.tick_top()
        else:
            ax.tick_params(labelbottom=(i == n - 1))
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.set_ylabel("Partial dependence", fontsize=8, color="white")

    # --- Off-diagonal: 2D partial dependence ---
    for i in range(n):
        for j in range(i):
            ax = axes[i, j]
            _style_ax(ax)

            name_x, name_y = param_names[j], param_names[i]
            idx_x, idx_y = j, i
            x_disc = name_x in discrete
            y_disc = name_y in discrete
            x_log = name_x in log_scale
            y_log = name_y in log_scale

            if x_disc and y_disc:
                # Both discrete: heatmap grid
                lx = np.arange(int(bounds[name_x][0]), int(bounds[name_x][1]) + 1)
                ly = np.arange(int(bounds[name_y][0]), int(bounds[name_y][1]) + 1)
                Z = _pd_2d(rbf, X_norm, X_min, X_range, idx_x, idx_y, lx, ly,
                            **pd_kw)
                ex, ey = _edges(lx), _edges(ly)
                ax.pcolormesh(ex, ey, Z, cmap=cmap, shading="flat")
                ax.set_xticks(lx)
                ax.set_yticks(ly)

            elif x_disc and not y_disc:
                # X discrete, Y continuous: vertical bands
                lx = np.arange(int(bounds[name_x][0]), int(bounds[name_x][1]) + 1)
                gy = _make_grid(name_y, bounds, log_scale, 40)
                Z = _pd_2d(rbf, X_norm, X_min, X_range, idx_x, idx_y, lx, gy,
                            **pd_kw)
                ex = _edges(lx)
                ey = _edges(gy)
                ax.pcolormesh(ex, ey, Z, cmap=cmap, shading="flat")
                ax.set_xticks(lx)
                if y_log:
                    ax.set_yscale("log")

            elif not x_disc and y_disc:
                # X continuous, Y discrete: horizontal bands
                ly = np.arange(int(bounds[name_y][0]), int(bounds[name_y][1]) + 1)
                gx = _make_grid(name_x, bounds, log_scale, 40)
                Z = _pd_2d(rbf, X_norm, X_min, X_range, idx_x, idx_y, gx, ly,
                            **pd_kw)
                ex = _edges(gx)
                ey = _edges(ly)
                ax.pcolormesh(ex, ey, Z, cmap=cmap, shading="flat")
                ax.set_yticks(ly)
                if x_log:
                    ax.set_xscale("log")

            else:
                # Both continuous: smooth contour
                gx = _make_grid(name_x, bounds, log_scale, 40)
                gy = _make_grid(name_y, bounds, log_scale, 40)
                Z = _pd_2d(rbf, X_norm, X_min, X_range, idx_x, idx_y, gx, gy,
                            **pd_kw)
                ax.contourf(gx, gy, Z, levels=15, cmap=cmap)
                if x_log:
                    ax.set_xscale("log")
                if y_log:
                    ax.set_yscale("log")

            # Scatter trials (only those within zoomed bounds)
            bx = bounds[name_x]
            by = bounds[name_y]
            mask = ((params[name_x] >= bx[0]) & (params[name_x] <= bx[1]) &
                    (params[name_y] >= by[0]) & (params[name_y] <= by[1]))
            ax.scatter(params[name_x][mask], params[name_y][mask],
                       c="black", s=6, alpha=0.4, zorder=2)
            # Best point (only if in range)
            if bx[0] <= best[name_x] <= bx[1] and by[0] <= best[name_y] <= by[1]:
                ax.scatter(best[name_x], best[name_y],
                           c=BEST_DOT_COLOR, s=40, zorder=3,
                           edgecolors="white", linewidth=0.5)
            # Clamp axis limits to bounds
            ax.set_xlim(bx)
            ax.set_ylim(by)

            # Labels
            if i == n - 1:
                ax.set_xlabel(name_x, fontsize=9, color="white")
            else:
                ax.tick_params(labelbottom=False)
            if j == 0:
                ax.set_ylabel(name_y, fontsize=9, color="white")
            else:
                ax.tick_params(labelleft=False)

    # Top-left title
    axes[0, 0].xaxis.set_label_position("top")
    axes[0, 0].xaxis.tick_top()
    axes[0, 0].set_title(param_names[0], fontsize=10, color="white", pad=8)

    metric_label = metric.replace("_", " ") if metric != "mrr" else ""
    title = f"{study_name} — {metric_label}" if metric_label else study_name
    fig.suptitle(title, color="white", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        suffix = f"-{metric}" if metric != "mrr" else ""
        path = out / f"{study_name}{suffix}.png"
        fig.savefig(path, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
        print(f"  Saved: {path}")
    else:
        plt.show()
    plt.close(fig)


def _parse_zoom(zoom_args):
    """Parse --zoom args like 'rrf_k=1:30' into {name: (lo, hi)}."""
    zooms = {}
    if not zoom_args:
        return zooms
    for z in zoom_args:
        name, rng = z.split("=", 1)
        lo, hi = rng.split(":")
        zooms[name] = (float(lo), float(hi))
    return zooms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", help="Specific study name (for .log files or --rdb)")
    parser.add_argument("--rdb", help="Path to RDB SQLite file (instead of .log files)")
    parser.add_argument("--output", "-o", help="Output directory for PNGs")
    parser.add_argument("--metric", default="mrr",
                        help="Metric to plot: 'mrr' (default) or a user_attr "
                             "name like 'miss_rate_5k'")
    parser.add_argument("--zoom", nargs="+", metavar="PARAM=LO:HI",
                        help="Override bounds for specific params, e.g. rrf_k=1:30")
    args = parser.parse_args()

    zooms = _parse_zoom(args.zoom)

    datasets = []

    if args.rdb:
        # Load from RDB storage
        import optuna
        rdb_path = Path(args.rdb)
        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{rdb_path}",
            engine_kwargs={"connect_args": {"timeout": 30}},
        )
        summaries = storage.get_all_studies()
        for s in summaries:
            if args.study and s.study_name != args.study:
                continue
            data = load_study_rdb(rdb_path, s.study_name, metric=args.metric)
            if data:
                datasets.append(data)
        if not datasets:
            print(f"No matching studies in {rdb_path}")
            sys.exit(1)
    else:
        # Load from journal log files
        logs = sorted(STUDY_DIR.glob("*.log"))
        if not logs:
            print(f"No .log files in {STUDY_DIR}")
            sys.exit(1)
        for log_path in logs:
            data = load_study(log_path, metric=args.metric)
            if data is None:
                print(f"Skipping {log_path.name}: no completed trials"
                      f" (or no '{args.metric}' attr)")
                continue
            if args.study and data["study_name"] != args.study:
                continue
            datasets.append(data)

    for data in datasets:
        # Apply zoom overrides to bounds
        for name, (lo, hi) in zooms.items():
            if name in data["bounds"]:
                data["bounds"][name] = (lo, hi)

        n_trials = len(data["values"])
        names = ", ".join(data["param_names"])
        disc = [n for n in data["param_names"] if n in data["discrete"]]
        disc_str = f" (discrete: {', '.join(disc)})" if disc else ""
        zoom_str = f" [zoom: {', '.join(f'{n}={lo}:{hi}' for n, (lo, hi) in zooms.items() if n in data['bounds'])}]" if zooms else ""
        print(f"Plotting {data['study_name']} ({n_trials} trials, {names}){disc_str}{zoom_str}")
        plot_corner(data, args.output)


if __name__ == "__main__":
    main()
