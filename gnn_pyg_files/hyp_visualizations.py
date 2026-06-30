#!/usr/bin/env python3
"""Per-hyperparameter comparison reports from a sweep run folder.

Reads ``<sweep_run>/summary.csv`` + ``sweep_meta.json`` (written by sweep.py).
For EACH swept axis it builds one PNG that overlays the per-episode panels from
report.png -- one colored line per hyperparameter value, the BASELINE value in
red -- plus summary-vs-param scalar panels and grouped-bar panels (resolution
rate vs multiplicity and gap distribution, one bar per level side by side).

The axes, baseline config, and x-scaling are discovered from sweep_meta.json, so
this works for ANY swept train.py argument.  Numeric axes get a value x-axis
(log when the values span >=25x); non-numeric axes are treated categorically.

When a config has multiple SEEDS every curve/bar/scalar is the seed mean, curves
get a +/-1 std band, and scalars get error bars.

  python3 hyp_visualizations.py <sweep_run_dir>   # folder holding summary.csv
  python3 hyp_visualizations.py                    # newest run under sweep/
"""
from __future__ import annotations
import csv, glob, json, os, sys, warnings
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from read_diagnostics import (load_episode_csv, load_calibration_csv,
                              load_loss_csv, rolling_mean, _resolved_mask)

CMAP = plt.get_cmap("viridis")
LEGACY_AXES = ["lr", "c_puct", "value_weight"]   # pre-sweep_meta flat folders


# ============================================================================ discovery
def find_latest_run(base: str) -> str:
    cands = [d for d in glob.glob(os.path.join(base, "*"))
             if os.path.isdir(d) and os.path.exists(os.path.join(d, "summary.csv"))]
    if os.path.exists(os.path.join(base, "summary.csv")):
        cands.append(base)
    if not cands:
        raise SystemExit(f"no summary.csv found under {base}")
    return max(cands, key=os.path.getmtime)


def load_summary(run_dir: str) -> list[dict]:
    p = os.path.join(run_dir, "summary.csv")
    if not os.path.exists(p):
        raise SystemExit(f"no summary.csv in {run_dir}")
    return [r for r in csv.DictReader(open(p)) if r.get("status") == "ok"]


def discover(run_dir: str, rows: list[dict]):
    """Return (axes, baseline_dict).  Prefer sweep_meta.json; fall back to legacy."""
    meta_p = os.path.join(run_dir, "sweep_meta.json")
    if os.path.exists(meta_p):
        meta = json.load(open(meta_p))
        return list(meta.get("axes", [])), dict(meta.get("baseline", {}))
    axes = [a for a in LEGACY_AXES if any(r["axis"] == a for r in rows)]
    base_row = next((r for r in rows if r["axis"] == "baseline"), None)
    baseline = {a: base_row[a] for a in axes} if base_row else {}
    return axes, baseline


def axis_groups(rows: list[dict], axis: str):
    """Ordered [(value_str, [seed rows]), ...] for one axis (incl. the baseline anchor)."""
    groups = defaultdict(list)
    for r in rows:
        if r["axis"] == axis or r["axis"] == "baseline":
            groups[r[axis]].append(r)
    keys = list(groups)
    nums = _as_floats(keys)
    keys.sort(key=float) if nums is not None else keys.sort()
    return [(k, groups[k]) for k in keys]


def _as_floats(keys):
    try:
        return [float(k) for k in keys]
    except (TypeError, ValueError):
        return None


# ============================================================================ data IO
def _locate_episode_dir(tdir: str) -> str | None:
    if os.path.exists(os.path.join(tdir, "episode_quality.csv")):
        return tdir
    subs = [d for d in glob.glob(os.path.join(tdir, "*"))
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "episode_quality.csv"))]
    return max(subs, key=os.path.getmtime) if subs else None


def resolve_run_dirs(rows: list[dict], run_dir: str) -> None:
    """Repair stale absolute run_dir paths (e.g. after a sweep folder was moved)
    by relocating each trial under the summary's own folder."""
    for r in rows:
        d = r.get("run_dir", "")
        if d and os.path.exists(os.path.join(d, "episode_quality.csv")):
            continue
        try:
            tdir = os.path.join(run_dir, f"t{int(r['trial']):03d}_s{int(r['seed'])}")
        except (KeyError, ValueError):
            continue
        inner = _locate_episode_dir(tdir)
        if inner:
            r["run_dir"] = inner


def load_run(row: dict):
    d = row["run_dir"]
    ep = load_episode_csv(os.path.join(d, "episode_quality.csv"))
    cal = load_calibration_csv(os.path.join(d, "value_calibration.csv"))
    tl = load_loss_csv(os.path.join(d, "train_loss.csv"))
    return ep, cal, tl


# ----- per-episode series (x, y) -----
def gap_series(ep):
    res = _resolved_mask(ep)
    x = [e for e, r in zip(ep["episode"], res) if r]
    y = [float(g) for g, r in zip(ep["gap"], res) if r]
    return x, y

def rays_saved_series(ep):
    """Rays saved vs min_sum (baseline - agent) from logged ray counts, all
    episodes.  >0 beats baseline; timeout episodes are censored lower bounds.
    This is a fixed quality metric, NOT the training reward -- the actual reward
    depends on resolved_reward_type/timeout_reward_type (and is per-state for the
    shaped variants), so it cannot be faithfully reconstructed here."""
    x = [e for e in ep["episode"]]
    y = [float(b - a) for a, b in zip(ep["agent_rays"], ep["baseline_rays"])]
    return x, y

def resolution_series(ep):
    return ep["episode"], [float(r) for r in ep["resolved"]]

def tie_series(ep):
    res = _resolved_mask(ep)
    x = [e for e, r in zip(ep["episode"], res) if r]
    y = [1.0 if g == 0 else 0.0 for g, r in zip(ep["gap"], res) if r]
    return x, y

def calib_err_series(cal):
    if not cal["episode"]:
        return [], []
    by = defaultdict(list)
    for e, p, t in zip(cal["episode"], cal["predicted_value"], cal["target_value"]):
        by[e].append(abs(p - t))
    xs = sorted(by)
    return xs, [float(np.mean(by[e])) for e in xs]

def loss_series(tl, key):
    return tl["episode"], [float(v) for v in tl[key]]


# ============================================================================ plotting
def _seed_mean(arr):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(arr, axis=0)


def plot_agg(ax, series_per_seed, w, color, grid_max):
    """Mean (+/-1 std band over seeds) of a per-episode series on a 0..grid_max grid."""
    xg = np.arange(grid_max)
    rows = []
    for x, y in series_per_seed:
        if len(x) == 0:
            continue
        ys = rolling_mean(list(y), w)
        rows.append(np.interp(xg, np.asarray(x, float), np.asarray(ys, float),
                              left=np.nan, right=np.nan))
    if not rows:
        return
    M = np.vstack(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(M, axis=0)
        sd = np.nanstd(M, axis=0) if M.shape[0] > 1 else None
    ax.plot(xg, mean, color=color, lw=1.6, alpha=0.9)
    if sd is not None:
        ax.fill_between(xg, mean - sd, mean + sd, color=color, alpha=0.12, linewidth=0)


def res_rate_bins(ep, edges):
    out = []
    for i in range(len(edges) - 1):
        sel = [r for m, r in zip(ep["mult"], ep["resolved"]) if edges[i] <= m < edges[i + 1]]
        out.append(sum(sel) / len(sel) if sel else np.nan)
    return out

def gap_frac_bins(ep, edges):
    rg = [g for g, r in zip(ep["gap"], _resolved_mask(ep)) if r]
    tot = len(rg) or 1
    return [sum(1 for g in rg if edges[i] <= g < edges[i + 1]) / tot for i in range(len(edges) - 1)]

def bin_label(a_edge, b_edge):
    a, b = int(round(a_edge)), int(round(b_edge)) - 1
    return f"{a}" if a == b else f"{a}–{b}"


def make_palette(keys, basev):
    """Return (color_fn(key), xval(key), logscale, categorical, is_base(key))."""
    nums = _as_floats(keys)
    bnum = None
    try:
        bnum = float(basev) if basev is not None else None
    except (TypeError, ValueError):
        bnum = None

    if nums is not None:
        vmin, vmax = min(nums), max(nums)
        logscale = vmin > 0 and (vmax / vmin) >= 25.0
        def is_base(k):
            return bnum is not None and abs(float(k) - bnum) < 1e-12
        def color(k):
            if is_base(k):
                return "red"
            v = float(k)
            if logscale:
                t = (np.log10(v) - np.log10(vmin)) / max(np.log10(vmax) - np.log10(vmin), 1e-9)
            else:
                t = (v - vmin) / max(vmax - vmin, 1e-9)
            return CMAP(0.05 + 0.9 * t)
        return color, (lambda k: float(k)), logscale, False, is_base

    idx = {k: i for i, k in enumerate(keys)}
    def is_base(k):
        return str(k) == str(basev)
    def color(k):
        if is_base(k):
            return "red"
        t = idx[k] / max(len(keys) - 1, 1)
        return CMAP(0.05 + 0.9 * t)
    return color, (lambda k: idx[k]), False, True, is_base


def build_axis_report(rows, axis, baseline, out_dir):
    groups = axis_groups(rows, axis)
    if not groups:
        print(f"[{axis}] no runs"); return None
    keys = [k for k, _ in groups]
    basev = baseline.get(axis)
    color_fn, xval, logscale, categorical, is_base = make_palette(keys, basev)

    loaded = {k: [load_run(r) for r in rs] for k, rs in groups}
    grid_max = max(len(ep["episode"]) for runs in loaded.values() for ep, _, _ in runs)
    n_seed = max(len(rs) for _, rs in groups)
    w = max(1, grid_max // 12)

    fig = plt.figure(figsize=(20, 21))
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1.1], hspace=0.5, wspace=0.28)
    ax = np.empty((3, 4), dtype=object)
    for r in range(3):
        for c in range(4):
            ax[r, c] = fig.add_subplot(gs[r, c])
    ax_resmult = fig.add_subplot(gs[3, 0:2])
    ax_gapdist = fig.add_subplot(gs[3, 2:4])

    # ---- overlaid (seed-averaged) episode curves --------------------------------------
    for k, rs in groups:
        col = color_fn(k)
        eps = [t[0] for t in loaded[k]]; cals = [t[1] for t in loaded[k]]; tls = [t[2] for t in loaded[k]]
        plot_agg(ax[0, 0], [gap_series(e) for e in eps], w, col, grid_max)
        plot_agg(ax[0, 1], [resolution_series(e) for e in eps], w, col, grid_max)
        plot_agg(ax[0, 2], [rays_saved_series(e) for e in eps], w, col, grid_max)
        plot_agg(ax[0, 3], [tie_series(e) for e in eps], w, col, grid_max)
        plot_agg(ax[1, 0], [calib_err_series(c) for c in cals], w, col, grid_max)
        plot_agg(ax[1, 1], [loss_series(t, "loss") for t in tls], w, col, grid_max)
        plot_agg(ax[1, 2], [loss_series(t, "policy_loss") for t in tls], w, col, grid_max)
        plot_agg(ax[1, 3], [loss_series(t, "value_loss") for t in tls], w, col, grid_max)

    ax[0, 0].axhline(0, color="gray", lw=1, ls="--")
    ax[0, 0].set_title("Optimality gap vs min_sum (resolved)\nrolling mean, lower better")
    ax[0, 0].set_xlabel("episode"); ax[0, 0].set_ylabel("gap (agent - min_sum)")
    ax[0, 1].set_title("Resolution rate\nrolling mean, higher better")
    ax[0, 1].set_xlabel("episode"); ax[0, 1].set_ylabel("resolved fraction"); ax[0, 1].set_ylim(-0.02, 1.02)
    ax[0, 2].axhline(0, color="gray", lw=1, ls="--")
    ax[0, 2].set_title("Rays saved vs min_sum (baseline - agent)\nall episodes; >0 beats baseline, timeouts censored")
    ax[0, 2].set_xlabel("episode"); ax[0, 2].set_ylabel("baseline - agent rays")
    ax[0, 3].set_title("Tie rate with min_sum\nrolling mean"); ax[0, 3].set_xlabel("episode")
    ax[0, 3].set_ylabel("tie fraction (resolved)"); ax[0, 3].set_ylim(-0.02, 1.02)
    ax[1, 0].set_title("Value calibration error\nmean |pred - target|, rolling")
    ax[1, 0].set_xlabel("episode"); ax[1, 0].set_ylabel("|pred - target|")
    ax[1, 1].set_title("Total train loss\nrolling mean"); ax[1, 1].set_xlabel("episode"); ax[1, 1].set_ylabel("loss")
    ax[1, 2].set_title("Policy loss\nrolling mean"); ax[1, 2].set_xlabel("episode"); ax[1, 2].set_ylabel("policy loss")
    ax[1, 3].set_title("Value loss\nrolling mean"); ax[1, 3].set_xlabel("episode"); ax[1, 3].set_ylabel("value loss")

    # ---- summary-vs-param scalars (seed mean +/- std) ---------------------------------
    def scalar_panel(a, field, title, ylabel):
        xs, ms = [], []
        for k, rs in groups:
            vals = [float(r[field]) for r in rs]
            m = float(np.mean(vals)); sd = float(np.std(vals)) if len(vals) > 1 else 0.0
            col = color_fn(k); x = xval(k)
            if sd > 0:
                a.errorbar([x], [m], yerr=[sd], fmt="none", ecolor=col, capsize=3, zorder=2)
            a.scatter([x], [m], color=col, s=70, zorder=3, edgecolor="k", linewidth=0.5)
            xs.append(x); ms.append(m)
        a.plot(xs, ms, color="0.6", lw=1, zorder=1)
        bk = next((k for k, _ in groups if is_base(k)), None)
        if bk is not None:
            a.axvline(xval(bk), color="C3", lw=1, ls=":", label="baseline", zorder=0)
            a.legend(fontsize=7)
        if logscale:
            a.set_xscale("log")
        if categorical:
            a.set_xticks(range(len(keys))); a.set_xticklabels([str(k) for k in keys])
        a.set_title(title); a.set_xlabel(axis); a.set_ylabel(ylabel)

    scalar_panel(ax[2, 0], "mean_gap", f"Final mean gap vs {axis}\n(tail third, lower better)", "mean gap")
    scalar_panel(ax[2, 1], "resolution_rate", f"Final resolution rate vs {axis}\n(tail third, higher better)", "resolved frac")
    scalar_panel(ax[2, 2], "tie_rate", f"Final tie rate vs {axis}\n(tail third)", "tie frac")
    scalar_panel(ax[2, 3], "score", f"Sweep score vs {axis}\n(lower better)", "score")

    # ---- grouped bars: one (seed-averaged) bar per level ------------------------------
    n = len(groups); gw = 0.8; bw = gw / n

    all_m = [m for runs in loaded.values() for ep, _, _ in runs for m in ep["mult"]]
    if all_m:
        lo, hi = min(all_m), max(all_m)
        nb = max(1, min(6, hi - lo + 1))
        edges = np.linspace(lo, hi + 1, nb + 1); xpos = np.arange(nb)
        for j, (k, rs) in enumerate(groups):
            heights = _seed_mean(np.array([res_rate_bins(ep, edges) for ep, _, _ in loaded[k]]))
            ax_resmult.bar(xpos - gw / 2 + bw * (j + 0.5), heights, width=bw * 0.95, color=color_fn(k))
        ax_resmult.set_xticks(xpos); ax_resmult.set_xticklabels([bin_label(edges[i], edges[i + 1]) for i in range(nb)])
        ax_resmult.set_ylim(0, 1.08)
        ax_resmult.set_xlabel("mult(σ) of root cone (binned)"); ax_resmult.set_ylabel("fraction resolved")
        ax_resmult.set_title(f"Resolution rate vs multiplicity\n(bars = {axis} levels, side by side; higher better)")

    all_g = [g for runs in loaded.values() for ep, _, _ in runs
             for g, r in zip(ep["gap"], _resolved_mask(ep)) if r]
    if all_g:
        lo, hi = min(all_g), max(all_g)
        nb = max(1, min(8, hi - lo + 1))
        edges = np.linspace(lo, hi + 1, nb + 1); xpos = np.arange(nb)
        for j, (k, rs) in enumerate(groups):
            heights = _seed_mean(np.array([gap_frac_bins(ep, edges) for ep, _, _ in loaded[k]]))
            ax_gapdist.bar(xpos - gw / 2 + bw * (j + 0.5), heights, width=bw * 0.95, color=color_fn(k))
        ax_gapdist.set_xticks(xpos); ax_gapdist.set_xticklabels([bin_label(edges[i], edges[i + 1]) for i in range(nb)])
        ax_gapdist.set_xlabel("gap (agent − min_sum), binned  (≤0 beats baseline)")
        ax_gapdist.set_ylabel("fraction of resolved episodes")
        ax_gapdist.set_title(f"Gap distribution\n(bars = {axis} levels, side by side; left/lower gap better)")

    # ---- shared legend + title --------------------------------------------------------
    def fmt(k):
        f = _as_floats([k])
        return f"{f[0]:g}" if f is not None else str(k)
    handles = [Line2D([0], [0], color=color_fn(k), lw=3,
                      label=("* " if is_base(k) else "") + fmt(k)) for k in keys]
    fig.legend(handles=handles, title=f"{axis}  (* = baseline)", loc="upper center",
               ncol=min(len(keys), 12), fontsize=9, title_fontsize=11,
               bbox_to_anchor=(0.5, 0.99), frameon=True)
    seed_note = f"mean of {n_seed} seeds (±1 std band)" if n_seed > 1 else "single seed"
    fig.suptitle(f"Sweep comparison — varying {axis}  "
                 f"({len(keys)} values, others pinned at baseline; {seed_note})",
                 y=1.015, fontsize=15, fontweight="bold")

    out = os.path.join(out_dir, f"compare_{axis}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def make_reports(run_dir: str, out_dir: str | None = None) -> list[str]:
    out_dir = out_dir or run_dir
    rows = load_summary(run_dir)
    resolve_run_dirs(rows, run_dir)
    axes, baseline = discover(run_dir, rows)
    if not axes:
        print(f"no swept axes found in {run_dir}"); return []
    print(f"{len(rows)} ok runs in {run_dir}  |  axes: {axes}")
    return [p for axis in axes if (p := build_axis_report(rows, axis, baseline, out_dir))]


def main():
    base = os.path.join(HERE, "sweep")
    run_dir = sys.argv[1] if len(sys.argv) > 1 else find_latest_run(base)
    make_reports(run_dir)


if __name__ == "__main__":
    main()
