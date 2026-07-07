#!/usr/bin/env python3
"""General-purpose hyperparameter search for train.py, with reports built in.

Combines a linear (one-axis-at-a-time) sweep and a grid (cross-product) search
behind one tool.  EVERY train.py argument is a tunable hyperparameter -- the full
set is discovered straight from train.py's own parser, so the two never drift --
except the harness-managed identity fields (seed, save, diag_dir, resume).

  * pin a constant for the whole run with   --set NAME=VALUE      (repeatable)
  * declare a search axis with              --sweep NAME=v1,v2    (1 or 2 axes)
  * repeat every config over               --seeds N
  * run configs in parallel with           --workers N   (each capped to
                                            --threads-per-run math threads)

MODES
  --mode linear (default)  the all-defaults BASELINE anchor plus one axis varied
                           at a time.  With two --sweep axes you get two
                           independent coordinate sweeps sharing the anchor.
  --mode grid              the full cross-product of the --sweep axes is run (no
                           separate baseline anchor; include a default value
                           explicitly to keep it in the grid).  Mind the blowup:
                           4x4 values x 3 seeds = 48 runs.

REPORTS (auto-generated when the run finishes; rebuild with --report DIR)
  linear:  compare_<axis>.png  -- per-axis overlay of the training curves, the
           summary-vs-value scalar panels, and the multiplicity/gap grouped bars.
  grid:    grid_report.png      -- one combined figure: the training curves (the
           same per-episode metrics as the linear report -- gap, resolution, rays,
           tie, calibration, total/policy/value loss), the score / resolution /
           mean-gap / tie+win scalar bars, the resolution-rate and gap-distribution
           vs determinant bars, and (for 2-axis grids) the score / resolution /
           mean-gap / tie-rate heatmaps over the two axes.

Every run gets its own folder sweep/sweep_<N>/ holding summary.csv,
sweep_meta.json, the per-trial subdirs, and the report PNGs.  A run can be paused
(Ctrl-C / --budget-hours) and continued with --resume <folder>.

  python3 hyperparam.py --sweep lr=1e-4,3e-4,1e-3 --seeds 3          # linear
  python3 hyperparam.py --mode grid --sweep lr=1e-4,3e-4 --sweep c_puct=1,1.5
  python3 hyperparam.py --resume sweep/sweep_3                       # continue
  python3 hyperparam.py --leaderboard                               # standings
  python3 hyperparam.py --report sweep/sweep_3                      # rebuild PNGs
  python3 hyperparam.py --list                                     # tunable args
"""
from __future__ import annotations
import argparse
import csv
import glob
import itertools
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import warnings
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train import build_parser
from read_diagnostics import (load_episode_csv, load_calibration_csv,
                              load_loss_csv, rolling_mean, _resolved_mask)

CMAP = plt.get_cmap("viridis")


def heatmap_cmap(lower: bool) -> str:
    """Red/blue diverging colormap oriented so that GOOD cells read blue and BAD
    cells read red, whichever direction is better for the metric."""
    return "RdBu_r" if lower else "RdBu"   # RdBu_r: low->blue, high->red


# Fixed value ranges so a heatmap's colors reflect ABSOLUTE quality rather than
# each panel's own spread: if every configuration is good, every cell reads blue
# (and a uniformly bad grid reads red), instead of stretching tiny differences
# across the full red<->blue range.  Rates are naturally [0,1]; gap/score use a
# fixed domain anchored at 0 = perfect with an upper bound that reads "clearly bad".
HEATMAP_RANGE = {
    "score": (0.0, 15.0), "resolution_rate": (0.0, 1.0), "res_rate": (0.0, 1.0),
    "mean_gap": (0.0, 12.0), "tie_rate": (0.0, 1.0), "tie": (0.0, 1.0),
}


def heatmap_range(field):
    """Fixed (vmin, vmax) for a heatmap metric, or (None, None) to auto-scale."""
    return HEATMAP_RANGE.get(field, (None, None))


def cell_text_color(im, value) -> str:
    """Black or white label, whichever reads on this cell's fill.  Uses the cell's
    actual luminance so it is correct for any colormap -- in particular the
    diverging red/blue map, whose extremes are BOTH dark (white text) while the
    midpoint is near-white (black text)."""
    r, g, b, _ = im.cmap(im.norm(value))
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.6 else "black"


# ===========================================================================================
#  TUNABLE SURFACE -- discovered from train.py's own argument parser.
# ===========================================================================================
def _discover():
    defaults, types, bools = {}, {}, set()
    for a in build_parser()._actions:
        if a.dest == "help":
            continue
        defaults[a.dest] = a.default
        types[a.dest] = a.type or str
        if isinstance(a, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            bools.add(a.dest)                          # store_true/false take no value
    return defaults, types, bools

DEFAULTS, TYPES, BOOL_FLAGS = _discover()
RESERVED = {"seed", "save", "diag_dir", "resume"}     # managed by the harness; not swept
TUNABLE = [k for k in DEFAULTS if k not in RESERVED]
SUMMARY_META = ["trial", "seed", "axis", "status", "score", "resolution_rate",
                "mean_gap", "tie_rate", "win_rate", "n_timeout", "elapsed_s", "run_dir"]
SUMMARY_FIELDS = SUMMARY_META + sorted(TUNABLE)       # every config's full param vector


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")

def _cast(name: str, s: str):
    if name in BOOL_FLAGS:
        return str(s).strip().lower() in ("1", "true", "yes", "on")
    try:
        return TYPES[name](s)
    except (ValueError, TypeError):
        tn = getattr(TYPES[name], "__name__", "value")
        raise SystemExit(f"bad value {s!r} for {_flag(name)} (expected {tn})")


def _parse_specs(args):
    """Build the baseline param vector and the {axis: [values]} sweep map from CLI."""
    base = {k: DEFAULTS[k] for k in TUNABLE}
    for spec in args.set:
        name, sep, val = spec.partition("=")
        name = name.strip().replace("-", "_")
        if not sep or name not in TUNABLE:
            raise SystemExit(f"--set: unknown/off-limits hyperparameter {name!r} (try --list)")
        base[name] = _cast(name, val)
    sweeps = {}
    for spec in args.sweep:
        name, sep, rest = spec.partition("=")
        name = name.strip().replace("-", "_")
        if not sep or name not in TUNABLE:
            raise SystemExit(f"--sweep: unknown/off-limits hyperparameter {name!r} (try --list)")
        vals = [_cast(name, x.strip()) for x in rest.split(",") if x.strip()]
        if not vals:
            raise SystemExit(f"--sweep {name}: no values given")
        sweeps[name] = vals
    return base, sweeps


# ===========================================================================================
#  CONFIG CONSTRUCTION
# ===========================================================================================
def linear_configs(base: dict, sweeps: dict) -> list[dict]:
    """Baseline anchor + one axis perturbed at a time (values equal to baseline
    collapse onto the anchor)."""
    configs = [{"id": 0, "axis": "baseline", "params": dict(base)}]
    cid = 1
    for axis, values in sweeps.items():
        for v in values:
            if v == base[axis]:
                continue
            configs.append({"id": cid, "axis": axis, "params": {**base, axis: v}})
            cid += 1
    return configs


def grid_configs(base: dict, sweeps: dict) -> list[dict]:
    """Full cross-product of the swept axes; ids follow itertools.product order
    over the axes sorted by name, so a repeated CLI resumes cleanly."""
    axes = sorted(sweeps)
    return [{"id": i, "axis": "grid", "params": {**base, **dict(zip(axes, combo))}}
            for i, combo in enumerate(itertools.product(*(sweeps[a] for a in axes)))]


def build_configs(base: dict, sweeps: dict, mode: str) -> list[dict]:
    return grid_configs(base, sweeps) if mode == "grid" else linear_configs(base, sweeps)


# ===========================================================================================
#  RUN A SINGLE TRIAL  (train.py subprocess) + SCORE IT
# ===========================================================================================
def score_run(run_dir: str) -> dict:
    """Summarize a finished run from episode_quality.csv.  LOWER score = better:
       mean inserted-ray gap over the last third + a penalty for unresolved episodes."""
    rows = list(csv.DictReader(open(os.path.join(run_dir, "episode_quality.csv"))))
    n = len(rows)
    if n == 0:
        return {"score": float("inf"), "episodes": 0, "resolution_rate": 0.0,
                "mean_gap": float("inf"), "tie_rate": 0.0, "win_rate": 0.0, "n_timeout": 0}
    res = lambda r: r["resolved"] == "1"
    tail = rows[2 * n // 3:]
    res_tail = [r for r in tail if res(r)]
    gaps = [int(r["gap"]) for r in tail]
    res_rate = sum(res(r) for r in tail) / len(tail)
    mean_gap = statistics.mean(gaps) if gaps else float("inf")
    tie_rate = (sum(1 for r in res_tail if int(r["gap"]) == 0) / len(res_tail)) if res_tail else 0.0
    win_rate = (sum(1 for r in res_tail if int(r["gap"]) < 0) / len(res_tail)) if res_tail else 0.0
    return {"score": round(mean_gap + 5.0 * (1.0 - res_rate), 4), "episodes": n,
            "resolution_rate": round(res_rate, 4), "mean_gap": round(mean_gap, 4),
            "tie_rate": round(tie_rate, 4), "win_rate": round(win_rate, 4),
            "n_timeout": sum(not res(r) for r in tail)}


def find_run_dir(diag_dir: str) -> str | None:
    subs = [d for d in glob.glob(os.path.join(diag_dir, "*"))
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "episode_quality.csv"))]
    return max(subs, key=os.path.getmtime) if subs else None


def load_done(p: str) -> set:
    done = set()
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            try:
                done.add((int(r["trial"]), int(r["seed"])))
            except (KeyError, ValueError):
                pass
    return done


def append_summary(p: str, row: dict) -> None:
    new = not os.path.exists(p)
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})


def run_one(cfg: dict, seed: int, run_dir: str, threads: int = 1) -> dict:
    diag_dir = os.path.join(run_dir, f"t{cfg['id']:03d}_s{seed}")
    os.makedirs(diag_dir, exist_ok=True)
    params = dict(cfg["params"])

    # ===== MANUAL OVERRIDE (temporary, this run only) =====
    # reward_type 0 -> force timeout_penalty to 60, overriding any --set/--sweep value.
    if params.get("reward_type") == 0:
        params["timeout_penalty"] = 60.0
    # ===== END MANUAL OVERRIDE =====

    cmd = [sys.executable, os.path.join(HERE, "train.py"),
           "--diag-dir", diag_dir, "--save", os.path.join(diag_dir, "model.pt"),
           "--seed", str(seed)]
    for k, v in params.items():
        if k in BOOL_FLAGS:
            if v:                                     # store_true: emit the bare flag only when enabled
                cmd += [_flag(k)]
        else:
            cmd += [_flag(k), str(v)]

    # cap each training job's math-library thread pools so N parallel workers do
    # not oversubscribe the machine (each job is otherwise ~1 core / Python-bound)
    t = str(max(1, threads))
    env = {**os.environ, "OMP_NUM_THREADS": t, "MKL_NUM_THREADS": t,
           "OPENBLAS_NUM_THREADS": t, "NUMEXPR_NUM_THREADS": t,
           "VECLIB_MAXIMUM_THREADS": t}
    t0 = time.perf_counter()
    with open(os.path.join(diag_dir, "train.log"), "w") as logf:
        proc = subprocess.run(cmd, cwd=HERE, stdout=logf, stderr=subprocess.STDOUT, env=env)
    row = {"trial": cfg["id"], "seed": seed, "axis": cfg["axis"],
           "elapsed_s": round(time.perf_counter() - t0, 1),
           **{k: params[k] for k in TUNABLE}}
    if proc.returncode != 0:
        return {**row, "status": "FAILED", "run_dir": diag_dir}
    rd = find_run_dir(diag_dir)
    if rd is None:
        return {**row, "status": "no_csv", "run_dir": diag_dir}
    return {**row, "status": "ok", "run_dir": rd, **score_run(rd)}


# ===========================================================================================
#  SUMMARY / META LOADING
# ===========================================================================================
def load_summary(run_dir: str) -> list[dict]:
    p = os.path.join(run_dir, "summary.csv")
    if not os.path.exists(p):
        raise SystemExit(f"no summary.csv in {run_dir}")
    return [r for r in csv.DictReader(open(p)) if r.get("status") == "ok"]


def load_meta(run_dir: str) -> dict:
    try:
        return json.load(open(os.path.join(run_dir, "sweep_meta.json")))
    except (OSError, ValueError):
        return {}


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None

def _as_floats(keys):
    try:
        return [float(k) for k in keys]
    except (TypeError, ValueError):
        return None


# ===========================================================================================
#  PRINTED ANALYSIS
# ===========================================================================================
def _grouped_by_value(rows, axis):
    groups = defaultdict(list)
    for r in rows:
        groups[r[axis]].append(r)
    keys = sorted(groups, key=lambda k: (_num(k) is None, _num(k) if _num(k) is not None else str(k)))
    return [(k, groups[k]) for k in keys]


def linear_analyze(run_dir: str, rows: list[dict], meta: dict) -> None:
    axes = meta.get("axes") or []
    base_rows = [r for r in rows if r["axis"] == "baseline"]
    print(f"\n=== linear sweep ({len(rows)} runs) -- lower score is better, mean over seeds ===")
    for axis in axes:
        by_val = defaultdict(list)
        for r in base_rows + [r for r in rows if r["axis"] == axis]:
            by_val[r[axis]].append(r)
        pts = []
        for val, rs in by_val.items():
            sc = [float(x["score"]) for x in rs]
            pts.append((val, statistics.mean(sc),
                        statistics.pstdev(sc) if len(sc) > 1 else 0.0, len(rs),
                        statistics.mean(float(x["resolution_rate"]) for x in rs),
                        statistics.mean(float(x["tie_rate"]) for x in rs)))
        pts.sort(key=lambda p: (_num(p[0]) is None, _num(p[0]) if _num(p[0]) is not None else str(p[0])))
        if not pts:
            continue
        best = min(pts, key=lambda p: p[1])
        print(f"\n  {axis}")
        print(f"    {'value':>12} {'score(mean±std)':>20} {'seeds':>5} {'res_rate':>9} {'tie_rate':>9}")
        for val, m, s, k, rr, tr in pts:
            star = "  <- best" if (val, m) == (best[0], best[1]) else ""
            print(f"    {str(val):>12} {f'{m:.3f}±{s:.3f}':>20} {k:>5} {rr:>9.3f} {tr:>9.3f}{star}")


def grid_analyze(run_dir: str, rows: list[dict], meta: dict) -> None:
    axes = meta.get("axes") or []
    print(f"\n=== grid search ({len(rows)} runs) -- lower score is better ===")
    for axis in axes:
        print(f"\n  {axis}  (marginal: mean over the other axes and seeds)")
        print(f"    {'value':>12} {'score(mean±std)':>20} {'runs':>5} {'res_rate':>9} {'tie_rate':>9}")
        pts = []
        for val, rs in _grouped_by_value(rows, axis):
            sc = [float(x["score"]) for x in rs]
            pts.append((val, statistics.mean(sc),
                        statistics.pstdev(sc) if len(sc) > 1 else 0.0, len(rs),
                        statistics.mean(float(x["resolution_rate"]) for x in rs),
                        statistics.mean(float(x["tie_rate"]) for x in rs)))
        best = min(pts, key=lambda p: p[1])
        for val, m, s, k, rr, tr in pts:
            star = "  <- best" if (val, m) == (best[0], best[1]) else ""
            print(f"    {str(val):>12} {f'{m:.3f}±{s:.3f}':>20} {k:>5} {rr:>9.3f} {tr:>9.3f}{star}")

    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[tuple(r[a] for a in axes)].append(r)
    board = sorted(((statistics.mean(float(x["score"]) for x in rs), key, len(rs))
                    for key, rs in by_cfg.items()))
    print(f"\n  combination leaderboard (top {min(10, len(board))} of {len(board)}):")
    for rank, (m, key, k) in enumerate(board[:10], 1):
        combo = ", ".join(f"{a}={v}" for a, v in zip(axes, key))
        print(f"    {rank:>2}. score {m:7.3f}  ({k} seeds)  {combo}")


def analyze(run_dir: str) -> None:
    rows, meta = load_summary(run_dir), load_meta(run_dir)
    if not rows:
        print(f"no successful runs yet in {run_dir}"); return
    (grid_analyze if meta.get("mode") == "grid" else linear_analyze)(run_dir, rows, meta)


# ===========================================================================================
#  SHARED PER-RUN DATA IO  (episode_quality.csv etc. for the training-curve reports)
# ===========================================================================================
def _locate_episode_dir(tdir: str) -> str | None:
    if os.path.exists(os.path.join(tdir, "episode_quality.csv")):
        return tdir
    subs = [d for d in glob.glob(os.path.join(tdir, "*"))
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "episode_quality.csv"))]
    return max(subs, key=os.path.getmtime) if subs else None


def resolve_run_dirs(rows: list[dict], run_dir: str) -> None:
    """Repair stale absolute run_dir paths (e.g. after the sweep folder was moved)
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


def _seed_mean(arr):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(arr, axis=0)


def _seed_std(arr):
    """Per-bin population std over seeds (0 for a single seed); nan bins -> 0 so
    the error bars never blow up on empty bins."""
    if arr.shape[0] < 2:
        return np.zeros(arr.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = np.nanstd(arr, axis=0)
    return np.nan_to_num(sd, nan=0.0)


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


# ---- shared 8-panel training-curve block (used by BOTH the linear per-axis report
#      and the grid per-configuration report so the two draw the same metrics) -------
def _plot_training_curves(ax, eps, cals, tls, w, col, grid_max):
    """Draw one group's 8 seed-averaged per-episode curves onto ax[0..1, 0..3]."""
    plot_agg(ax[0, 0], [gap_series(e) for e in eps], w, col, grid_max)
    plot_agg(ax[0, 1], [resolution_series(e) for e in eps], w, col, grid_max)
    plot_agg(ax[0, 2], [rays_saved_series(e) for e in eps], w, col, grid_max)
    plot_agg(ax[0, 3], [tie_series(e) for e in eps], w, col, grid_max)
    plot_agg(ax[1, 0], [calib_err_series(c) for c in cals], w, col, grid_max)
    plot_agg(ax[1, 1], [loss_series(t, "loss") for t in tls], w, col, grid_max)
    plot_agg(ax[1, 2], [loss_series(t, "policy_loss") for t in tls], w, col, grid_max)
    plot_agg(ax[1, 3], [loss_series(t, "value_loss") for t in tls], w, col, grid_max)


def _label_training_curves(ax):
    """Titles/axis labels for the 8 training-curve panels (linear + grid reports)."""
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


# ===========================================================================================
#  LINEAR REPORT  (per-axis comparison -- identical layout to the old hyp_visualizations)
# ===========================================================================================
def axis_groups(rows: list[dict], axis: str):
    """Ordered [(value, [seed rows]), ...] for one axis (incl. the baseline anchor)."""
    groups = defaultdict(list)
    for r in rows:
        if r["axis"] == axis or r["axis"] == "baseline":
            groups[r[axis]].append(r)
    keys = list(groups)
    nums = _as_floats(keys)
    keys.sort(key=float) if nums is not None else keys.sort()
    return [(k, groups[k]) for k in keys]


def make_palette(keys, basev):
    """Return (color_fn(key), xval(key), logscale, categorical, is_base(key))."""
    nums = _as_floats(keys)
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
        _plot_training_curves(ax, eps, cals, tls, w, col, grid_max)
    _label_training_curves(ax)

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
            arr = np.array([res_rate_bins(ep, edges) for ep, _, _ in loaded[k]])
            ax_resmult.bar(xpos - gw / 2 + bw * (j + 0.5), _seed_mean(arr), width=bw * 0.95,
                           color=color_fn(k), yerr=_seed_std(arr), capsize=2,
                           error_kw=dict(lw=0.7, alpha=0.6))
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
            arr = np.array([gap_frac_bins(ep, edges) for ep, _, _ in loaded[k]])
            ax_gapdist.bar(xpos - gw / 2 + bw * (j + 0.5), _seed_mean(arr), width=bw * 0.95,
                           color=color_fn(k), yerr=_seed_std(arr), capsize=2,
                           error_kw=dict(lw=0.7, alpha=0.6))
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


# ===========================================================================================
#  GRID REPORT  (per-configuration panels + score heatmaps)
# ===========================================================================================
def config_groups(rows, axes):
    """Ordered [(config_key_tuple, [seed rows]), ...] over the grid axes."""
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[a] for a in axes)].append(r)
    def sortkey(key):
        return tuple((_num(v) is None, _num(v) if _num(v) is not None else str(v)) for v in key)
    return [(k, groups[k]) for k in sorted(groups, key=sortkey)]


def _config_label(key, axes):
    return "\n".join(f"{a}={v}" for a, v in zip(axes, key))


def _draw_grid_heatmaps_row(fig, gs, rows, axes):
    """Draw the four score / resolution / mean-gap / tie-rate heatmaps over the two
    swept axes onto the LAST gridspec row (2-axis grids only)."""
    ax_a, ax_b = axes
    va = [v for v, _ in _grouped_by_value(rows, ax_a)]
    vb = [v for v, _ in _grouped_by_value(rows, ax_b)]

    def matrix(field):
        cell = defaultdict(list)
        for r in rows:
            if _num(r.get(field)) is not None:
                cell[(r[ax_a], r[ax_b])].append(float(r[field]))
        M = np.full((len(va), len(vb)), np.nan)
        for i, a_ in enumerate(va):
            for j, b_ in enumerate(vb):
                if cell[(a_, b_)]:
                    M[i, j] = statistics.mean(cell[(a_, b_)])
        return M

    panels = [("score", "score", True), ("resolution_rate", "resolution rate", False),
              ("mean_gap", "mean gap", True), ("tie_rate", "tie rate", False)]
    last = gs.nrows - 1
    for col, (field, title, lower) in enumerate(panels):
        a = fig.add_subplot(gs[last, col])
        M = matrix(field)
        vmin, vmax = heatmap_range(field)
        im = a.imshow(np.ma.masked_invalid(M), cmap=heatmap_cmap(lower), aspect="auto",
                      vmin=vmin, vmax=vmax)
        im.cmap.set_bad("0.85")
        a.set_xticks(range(len(vb))); a.set_xticklabels([str(v) for v in vb], fontsize=8)
        a.set_yticks(range(len(va))); a.set_yticklabels([str(v) for v in va], fontsize=8)
        a.set_xlabel(ax_b); a.set_ylabel(ax_a)
        a.set_title(f"{title}  ({'lower' if lower else 'higher'} better; blue = good)", fontsize=10)
        if np.isfinite(M).any():
            for i in range(len(va)):
                for j in range(len(vb)):
                    if M[i, j] == M[i, j]:
                        a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                               color=cell_text_color(im, M[i, j]))
            bi, bj = np.unravel_index((np.nanargmin if lower else np.nanargmax)(M), M.shape)
            a.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, edgecolor="#00b140", lw=2.5))
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)


def grid_report(run_dir, rows, meta, out_dir):
    """One combined grid_report.png for a grid search, stacking every grid view:
      * rows 0-1  training curves -- the SAME per-episode metrics as the linear
                  report (gap, resolution, rays saved, tie rate, calibration error,
                  total/policy/value loss), one seed-averaged curve (+/-1 std band)
                  per configuration;
      * row 2     per-configuration scalar bars (score / resolution / mean gap /
                  tie+win), with ±1 std over seeds;
      * row 3     resolution-rate and gap-distribution vs determinant grouped bars;
      * row 4     score / resolution / mean-gap / tie-rate heatmaps (2-axis grids)."""
    axes = meta.get("axes") or []
    if not axes:
        print("no grid axes in meta; skipping grid report"); return []
    resolve_run_dirs(rows, run_dir)
    cfgs = config_groups(rows, axes)
    if not cfgs:
        print("[grid report] no configurations; skipping"); return []
    labels = [_config_label(k, axes) for k, _ in cfgs]
    xs = np.arange(len(cfgs))
    colors = CMAP(np.linspace(0.05, 0.95, len(cfgs)))
    n_seed = max((len(rs) for _, rs in cfgs), default=1)
    two_axis = len(axes) == 2

    def agg(field):
        ms, sds = [], []
        for _, rs in cfgs:
            vals = [float(r[field]) for r in rs if _num(r[field]) is not None]
            ms.append(statistics.mean(vals) if vals else np.nan)
            sds.append(statistics.pstdev(vals) if len(vals) > 1 else 0.0)
        return np.array(ms), np.array(sds)

    # full per-run diagnostics (episode + calibration + loss) for every seed
    full = {k: [load_run(r) for r in rs
                if os.path.exists(os.path.join(r.get("run_dir", ""), "episode_quality.csv"))]
            for k, rs in cfgs}
    eps_by_cfg = {k: [t[0] for t in runs] for k, runs in full.items()}   # episode dicts only
    grid_max = max((len(ep["episode"]) for runs in eps_by_cfg.values() for ep in runs), default=0)
    w = max(1, grid_max // 12) if grid_max else 1

    nrows = 5 if two_axis else 4
    height_ratios = [1, 1, 1.25, 1.25] + ([1.15] if two_axis else [])
    fig = plt.figure(figsize=(23, 5.0 * nrows + 1.5))
    gs = fig.add_gridspec(nrows, 4, hspace=0.75, wspace=0.30, height_ratios=height_ratios)

    # ---- rows 0-1: training curves (identical metrics to the linear report) ----
    if grid_max:
        cax = np.array([[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(2)], dtype=object)
        for j, (k, _) in enumerate(cfgs):
            eps = [t[0] for t in full[k]]; cals = [t[1] for t in full[k]]; tls = [t[2] for t in full[k]]
            _plot_training_curves(cax, eps, cals, tls, w, colors[j], grid_max)
        _label_training_curves(cax)

    # ---- row 2: per-configuration scalar bars ----
    def bar_panel(a, field, title, ylabel, lower=True):
        m, sd = agg(field)
        a.bar(xs, m, yerr=sd, color=colors, capsize=3, error_kw=dict(lw=0.8, alpha=0.7))
        finite = [(x, v) for x, v in zip(xs, m) if v == v]
        if finite:
            bx, bv = (min if lower else max)(finite, key=lambda p: p[1])
            a.scatter([bx], [bv], s=170, facecolors="none", edgecolors="#00b140", lw=2, zorder=5)
        a.set_xticks(xs); a.set_xticklabels(labels, fontsize=6.5, rotation=90)
        a.set_ylabel(ylabel); a.grid(alpha=0.3, axis="y")
        a.set_title(title + ("  (lower better)" if lower else "  (higher better)"), fontsize=10)

    bar_panel(fig.add_subplot(gs[2, 0]), "score", "Score vs configuration", "score", lower=True)
    bar_panel(fig.add_subplot(gs[2, 1]), "resolution_rate", "Resolution rate vs configuration", "resolved frac", lower=False)
    bar_panel(fig.add_subplot(gs[2, 2]), "mean_gap", "Mean gap vs min_sum vs configuration", "mean gap", lower=True)

    a = fig.add_subplot(gs[2, 3])
    tm, tsd = agg("tie_rate"); wm, wsd = agg("win_rate")
    a.bar(xs - 0.2, tm, width=0.4, yerr=tsd, color="C7", capsize=2,
          error_kw=dict(lw=0.8, alpha=0.7), label="tie rate (gap=0)")
    a.bar(xs + 0.2, wm, width=0.4, yerr=wsd, color="C2", capsize=2,
          error_kw=dict(lw=0.8, alpha=0.7), label="win rate (gap<0)")
    a.set_xticks(xs); a.set_xticklabels(labels, fontsize=6.5, rotation=90)
    a.set_ylabel("fraction of resolved"); a.set_title("Tie / win rate vs configuration", fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.3, axis="y")

    # ---- row 3: resolution-rate & gap-distribution vs determinant grouped bars ----
    a = fig.add_subplot(gs[3, 0:2])
    all_m = [m for runs in eps_by_cfg.values() for ep in runs for m in ep["mult"]]
    if all_m:
        lo, hi = min(all_m), max(all_m)
        nb = max(1, min(6, hi - lo + 1))
        edges = np.linspace(lo, hi + 1, nb + 1); bxs = np.arange(nb)
        gw = 0.8; bw = gw / max(1, len(cfgs))
        for j, (k, _) in enumerate(cfgs):
            if not eps_by_cfg[k]:
                continue
            arr = np.array([res_rate_bins(ep, edges) for ep in eps_by_cfg[k]])
            a.bar(bxs - gw / 2 + bw * (j + 0.5), _seed_mean(arr), width=bw * 0.95,
                  color=colors[j], yerr=_seed_std(arr), capsize=2, error_kw=dict(lw=0.7, alpha=0.6))
        a.set_xticks(bxs); a.set_xticklabels([bin_label(edges[i], edges[i + 1]) for i in range(nb)])
        a.set_ylim(0, 1.08)
    a.set_xlabel("determinant (root mult, binned)"); a.set_ylabel("fraction resolved")
    a.set_title("Resolution rate vs determinant  (bars = configs, side by side; higher better)", fontsize=10)

    a = fig.add_subplot(gs[3, 2:4])
    all_g = [g for runs in eps_by_cfg.values() for ep in runs
             for g, r in zip(ep["gap"], _resolved_mask(ep)) if r]
    if all_g:
        lo, hi = min(all_g), max(all_g)
        nb = max(1, min(8, hi - lo + 1))
        edges = np.linspace(lo, hi + 1, nb + 1); bxs = np.arange(nb)
        gw = 0.8; bw = gw / max(1, len(cfgs))
        for j, (k, _) in enumerate(cfgs):
            if not eps_by_cfg[k]:
                continue
            arr = np.array([gap_frac_bins(ep, edges) for ep in eps_by_cfg[k]])
            a.bar(bxs - gw / 2 + bw * (j + 0.5), _seed_mean(arr), width=bw * 0.95,
                  color=colors[j], yerr=_seed_std(arr), capsize=2, error_kw=dict(lw=0.7, alpha=0.6))
        a.set_xticks(bxs); a.set_xticklabels([bin_label(edges[i], edges[i + 1]) for i in range(nb)])
    a.set_xlabel("gap (agent − min_sum), binned  (≤0 beats baseline)")
    a.set_ylabel("fraction of resolved episodes")
    a.set_title("Gap distribution vs determinant  (bars = configs, side by side; left/lower better)", fontsize=10)

    # ---- row 4: score / resolution / mean-gap / tie-rate heatmaps (2-axis grids) ----
    if two_axis:
        _draw_grid_heatmaps_row(fig, gs, rows, axes)

    handles = [Line2D([], [], color=colors[j], lw=6, label=labels[j].replace("\n", ", "))
               for j in range(len(cfgs))]
    fig.legend(handles=handles, title="configuration", loc="upper center",
               ncol=min(len(cfgs), 8), fontsize=7.5, title_fontsize=9,
               bbox_to_anchor=(0.5, 0.955), framealpha=0.9)
    seed_note = f"mean of {n_seed} seeds (±1 std band)" if n_seed > 1 else "single seed"
    fig.suptitle(f"Grid search — {len(cfgs)} configurations over {', '.join(axes)}  ({seed_note})",
                 y=0.995, fontsize=16, fontweight="bold")

    fig.subplots_adjust(left=0.05, right=0.985, top=0.92, bottom=0.03)
    out = os.path.join(out_dir, "grid_report.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return [out]


def make_reports(run_dir: str) -> list[str]:
    rows, meta = load_summary(run_dir), load_meta(run_dir)
    if not rows:
        print(f"[report] no successful runs in {run_dir}; skipping"); return []
    mode = meta.get("mode", "linear")
    if mode == "grid":
        return grid_report(run_dir, rows, meta, run_dir)
    resolve_run_dirs(rows, run_dir)
    axes = meta.get("axes") or []
    baseline = meta.get("baseline", {})
    if not axes:
        print(f"[report] no swept axes in {run_dir}"); return []
    return [p for axis in axes if (p := build_axis_report(rows, axis, baseline, run_dir))]


# ===========================================================================================
#  RUN-FOLDER HELPERS
# ===========================================================================================
def _next_run_dir(outdir: str) -> str:
    n = 0
    while os.path.exists(os.path.join(outdir, f"sweep_{n}")):
        n += 1
    return os.path.join(outdir, f"sweep_{n}")


def _latest_run(outdir: str) -> str | None:
    subs = [d for d in glob.glob(os.path.join(outdir, "*"))
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "summary.csv"))]
    return max(subs, key=os.path.getmtime) if subs else None


# ===========================================================================================
#  MAIN
# ===========================================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["linear", "grid"], default="linear",
                    help="linear = baseline + one axis at a time; grid = cross-product of the axes")
    ap.add_argument("--sweep", action="append", default=[], metavar="NAME=v1,v2,...",
                    help="hyperparameter axis to vary + values (repeatable; 1 or 2 axes)")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="pin a non-default constant for every run (repeatable)")
    ap.add_argument("--seeds", type=int, default=1, help="seeds per config")
    ap.add_argument("--workers", type=int, default=1,
                    help="training runs to execute in parallel (each ~1 core; "
                         f"this machine has {os.cpu_count()} cores)")
    ap.add_argument("--threads-per-run", type=int, default=1,
                    help="math-library threads each training subprocess may use")
    ap.add_argument("--outdir", default=os.path.join(HERE, "sweep"),
                    help="container; each invocation gets its own sweep_<N> subfolder")
    ap.add_argument("--resume", default=None, help="continue a previous run folder")
    ap.add_argument("--budget-hours", type=float, default=10.0)
    ap.add_argument("--leaderboard", action="store_true", help="print standings and exit")
    ap.add_argument("--list", action="store_true", help="list tunable args + defaults and exit")
    ap.add_argument("--report", default=None, metavar="RUN_DIR",
                    help="rebuild the report PNGs for a finished run folder and exit")
    ap.add_argument("--no-report", action="store_true", help="skip the report PNGs")
    ap.add_argument("--smoke", action="store_true", help="tiny self-contained plumbing test")
    args = ap.parse_args()

    if args.list:
        print("tunable train.py hyperparameters (name: default):")
        for k in sorted(TUNABLE):
            print(f"  {k}: {DEFAULTS[k]}")
        print(f"\nmanaged by the harness, not tunable: {sorted(RESERVED)}")
        return

    if args.report:
        make_reports(args.report)
        return

    os.makedirs(args.outdir, exist_ok=True)
    if args.leaderboard:
        run_dir = args.resume or _latest_run(args.outdir)
        if run_dir is None:
            print(f"no runs under {args.outdir}"); return
        print(f"leaderboard for {run_dir}")
        analyze(run_dir)
        return

    if args.resume:
        # reconstruct the whole search from the run's own sweep_meta.json so the
        # user only needs --resume (mode/axes/values/baseline/seeds all restored,
        # keeping trial ids -- and therefore the done-set -- consistent).
        run_dir = args.resume
        if not os.path.isdir(run_dir):
            raise SystemExit(f"--resume folder not found: {run_dir}")
        meta = load_meta(run_dir)
        if not meta.get("values"):
            raise SystemExit(f"--resume: no sweep_meta.json with values in {run_dir}")
        args.mode = meta.get("mode", args.mode)
        args.seeds = meta.get("seeds", args.seeds)
        sweeps = {a: list(v) for a, v in meta["values"].items()}
        base = {k: DEFAULTS[k] for k in TUNABLE}
        base.update({k: v for k, v in meta.get("baseline", {}).items() if k in base})
        print(f"resuming {run_dir} (mode={args.mode}, {args.seeds} seeds)")
    else:
        base, sweeps = _parse_specs(args)
        if args.smoke:
            for k, v in (("episodes", 5), ("max_steps", 40)):
                base[k] = v
            sweeps = sweeps or {"lr": [_cast("lr", "3e-05"), _cast("lr", "1e-4")]}
            args.seeds = 1
            print("[smoke] tiny regime")
        if not sweeps:
            raise SystemExit("nothing to sweep: pass at least one --sweep NAME=v1,v2,... (or --list)")
        if len(sweeps) > 2:
            raise SystemExit(f"at most 2 --sweep axes supported, got {len(sweeps)}: {sorted(sweeps)}")
        run_dir = _next_run_dir(args.outdir)
        os.makedirs(run_dir, exist_ok=True)
        print(f"new {args.mode} run -> {run_dir}")
    summary_path = os.path.join(run_dir, "summary.csv")

    configs = build_configs(base, sweeps, args.mode)
    if args.smoke:
        configs = configs[:2]
    seeds = list(range(args.seeds))

    with open(os.path.join(run_dir, "sweep_meta.json"), "w") as f:
        json.dump({"mode": args.mode,
                   "axes": sorted(sweeps),
                   "values": {a: list(sweeps[a]) for a in sweeps},
                   "baseline": {k: base[k] for k in TUNABLE},
                   "seeds": args.seeds}, f, indent=2, default=str)

    done = load_done(summary_path)
    total = len(configs) * len(seeds)
    print(f"mode {args.mode} | axes {sorted(sweeps)} | {len(configs)} configs x {len(seeds)} seeds "
          f"= {total} runs | workers {args.workers} x {args.threads_per_run} threads "
          f"| budget {args.budget_hours}h | done {len(done)}")
    print(f"writing -> {summary_path}\n")

    t_start = time.perf_counter()
    budget_s = args.budget_hours * 3600
    workers = max(1, args.workers)
    write_lock = threading.Lock()

    def _where(cfg: dict) -> str:
        if cfg["axis"] == "baseline":
            return "baseline"
        if cfg["axis"] == "grid":
            return ",".join(f"{a}={cfg['params'][a]}" for a in sorted(sweeps))
        v = cfg["params"][cfg["axis"]]
        return f"{cfg['axis']}={v:g}" if isinstance(v, float) else f"{cfg['axis']}={v}"

    # seeds outermost -> breadth-first coverage if the budget cuts us short
    pending = [(cfg, seed) for seed in seeds for cfg in configs
               if (cfg["id"], seed) not in done]
    print(f"running {len(pending)} trials with {workers} parallel worker(s)\n")

    task_iter = iter(pending)
    inflight: dict = {}
    launched = completed = 0
    budget_hit = False

    with ThreadPoolExecutor(max_workers=workers) as ex:
        while True:
            while len(inflight) < workers and not budget_hit:    # top up the worker pool
                if time.perf_counter() - t_start > budget_s:
                    budget_hit = True
                    break
                try:
                    cfg, seed = next(task_iter)
                except StopIteration:
                    break
                inflight[ex.submit(run_one, cfg, seed, run_dir, args.threads_per_run)] = (cfg, seed)
                launched += 1
                print(f"[{launched}/{len(pending)}] -> trial {cfg['id']} "
                      f"({_where(cfg)}) seed {seed}", flush=True)
            if not inflight:                                     # nothing left running -> done
                break
            for fut in wait(inflight, return_when=FIRST_COMPLETED).done:
                cfg, seed = inflight.pop(fut)
                row = fut.result()
                with write_lock:
                    append_summary(summary_path, row)
                completed += 1
                tag = (f"score={row['score']} res={row.get('resolution_rate')} tie={row.get('tie_rate')}"
                       if row["status"] == "ok" else f"!! {row['status']} (see {row['run_dir']}/train.log)")
                print(f"     <- [{completed}/{len(pending)}] trial {cfg['id']} "
                      f"({_where(cfg)}) seed {seed}: {tag}  [{row['elapsed_s']}s]", flush=True)

    if budget_hit:
        print(f"\n[budget] {args.budget_hours}h reached -- stopping. "
              f"Re-run with --resume {run_dir} to continue.")
    else:
        print("\nall runs complete.")
    analyze(run_dir)
    if not args.no_report:
        try:
            make_reports(run_dir)
        except Exception as e:                       # plotting must never sink a finished run
            print(f"[report] skipped: {e}")


if __name__ == "__main__":
    main()
