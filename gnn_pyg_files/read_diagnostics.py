#!/usr/bin/env python3
"""Read a training run's diagnostics CSVs and render an interpretation report.

WHAT IT READS  (all relative to the run directory, --directory)
  episode_quality.csv    REQUIRED  cols: episode,n,mult,agent_rays,baseline_rays,gap,resolved
  value_calibration.csv  optional  cols: episode,state_idx,predicted_value,target_value
  train_loss.csv         optional  cols: episode,loss,policy_loss,value_loss
  timing.json            optional  {elapsed_seconds, episodes, sec_per_episode}
  config.json            optional  hyperparameter dump (see also --config / --checkpoint)
Missing optional files degrade gracefully -- their panels just print a placeholder.

WHAT IT WRITES  (into the run directory, unless --out is given)
  report.png         the multi-panel figure described below
  initial_cones.png  a second figure: histograms of the dim/mult of cones fed to self-play
  panel_*.png        one PNG per panel, only when --split is passed

REPORT PANELS
   1. Comparison to min_sum        gap learning curve (resolved-only mean; timeouts marked)
   2. Resolution rate              rolling fraction of episodes reaching a unimodular fan
   3. Win / tie / loss             resolved-only: gap <0 / ==0 / >0 vs min_sum
   4. Agent vs min_sum (parity)    resolved-only scatter, below y=x means agent used fewer rays
   5. Gap vs multiplicity          resolved-only; where it struggles by root mult(sigma)
   6. Resolution rate vs mult      bucketed; where it fails to resolve at all
   7. Resolution rate vs dimension per-dimension resolve rate
   8. Gap distribution             resolved-only histogram; mass <=0 ties or beats min_sum
   9. Value-head calibration       predicted vs target value, colored by episode
  10. Value-head error             per-episode mean |pred - target|, rolling
  11. Raw episode reward           terminal_reward - rays (= -gap when resolved); >0 beats min_sum
  12. Training loss                total / policy / value, fit-to-replay
  13. Win / tie / loss (dim>=3)    panel 3 restricted to 3D+ (beating min_sum is impossible in 2D)
  + a hyperparameter strip at the bottom (from config.json / --config / the checkpoint args).

NOTE: agent_rays is censored at max_steps on timed-out episodes, so a timeout's gap is a
lower bound, not a real optimality gap. Every gap/parity panel uses RESOLVED episodes only;
timeouts are summarized separately (panels 2, 6, 7).

USAGE
  # default: pick the latest run under gnn_pyg_files/results/ and write report.png there
  python read_diagnostics.py

  # a specific run, with a wider smoothing window and per-panel PNGs
  python read_diagnostics.py --directory results/20260627_222614 --rolling 25 --split

  # supply hyperparameters explicitly, or pull them from a checkpoint, and skip the note prompt
  python read_diagnostics.py --config run_config.json --note "baseline, det_max=12"
  python read_diagnostics.py --checkpoint cone_action_gnn.pt --note ""

KEY FLAGS
  --directory DIR     run dir holding the CSVs (default: latest under results/)
  --rolling N         rolling-window width for smoothed curves (default 20)
  --split             also write each panel as its own panel_*.png
  --note TEXT         caption stamped in report.png's corner; if omitted you are prompted
                      (pass --note "" to suppress the prompt in non-interactive runs)
  --config JSON       hyperparameter file to display; else config.json, else --checkpoint args
  --checkpoint PT     model .pt to read hyperparameters from (default cone_action_gnn.pt)
  --out PNG / --episode-csv / --calibration-csv   override the default output / input paths
"""
from __future__ import annotations
import argparse
import csv
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ===========================================================================================
#  IO
# ===========================================================================================

def load_episode_csv(path: str) -> dict[str, list]:
    cols = ["episode", "n", "mult", "agent_rays", "baseline_rays", "gap", "resolved"]
    out: dict[str, list] = {c: [] for c in cols}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for c in cols:
                out[c].append(int(round(float(row[c]))))
    return out


def load_calibration_csv(path: str) -> dict[str, list]:
    out = {"episode": [], "predicted_value": [], "target_value": []}
    if not os.path.exists(path):
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out["episode"].append(int(round(float(row["episode"]))))
            out["predicted_value"].append(float(row["predicted_value"]))
            out["target_value"].append(float(row["target_value"]))
    return out


def load_loss_csv(path: str) -> dict[str, list]:
    out = {"episode": [], "loss": [], "policy_loss": [], "value_loss": []}
    if not os.path.exists(path):
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out["episode"].append(int(round(float(row["episode"]))))
            out["loss"].append(float(row["loss"]))
            out["policy_loss"].append(float(row["policy_loss"]))
            out["value_loss"].append(float(row["value_loss"]))
    return out


def rolling_mean(xs: list[float], w: int) -> list[float]:
    out = []
    for i in range(len(xs)):
        lo = max(0, i - w + 1)
        window = xs[lo:i + 1]
        out.append(sum(window) / len(window))
    return out


def _fmt_hms(seconds: float) -> str:
    """Format a duration in seconds as M:SS (or H:MM:SS when >= 1 hour)."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def load_timing(directory: str, n_ep: int) -> Optional[dict]:
    """Return {elapsed_seconds, sec_per_episode, approximate} for the run, or None.

    Prefers timing.json (written by train.py).  If that is absent (e.g. an older
    run), estimates the elapsed time from file mtimes -- config.json is written at
    the start of the run and episode_quality.csv is last flushed at the end of the
    training loop -- and flags the result as approximate.
    """
    import json
    p = os.path.join(directory, "timing.json")
    if os.path.exists(p):
        try:
            with open(p) as f:
                d = dict(json.load(f))
            el = float(d["elapsed_seconds"])
            eps = int(d.get("episodes", n_ep)) or n_ep
            spe = float(d.get("sec_per_episode", el / max(1, eps)))
            return {"elapsed_seconds": el, "sec_per_episode": spe, "approximate": False}
        except Exception as e:
            print(f"[timing] could not read {p}: {e}")

    cfg_p = os.path.join(directory, "config.json")
    ep_p = os.path.join(directory, "episode_quality.csv")
    try:
        if os.path.exists(cfg_p) and os.path.exists(ep_p):
            el = os.path.getmtime(ep_p) - os.path.getmtime(cfg_p)
            if el > 0:
                return {"elapsed_seconds": el,
                        "sec_per_episode": el / max(1, n_ep),
                        "approximate": True}
    except OSError:
        pass
    return None


def _resolved_mask(ep: dict) -> list[bool]:
    return [bool(r) for r in ep["resolved"]]


def _jitter(vals, scale=0.12):
    return np.asarray(vals, float) + np.random.uniform(-scale, scale, size=len(vals))


# ===========================================================================================
#  REPORT PANELS
# ===========================================================================================

def panel_gap_curve(ax, ep: dict, w: int) -> None:
    # Resolved-only: a timed-out episode's gap is a censored lower bound, not a gap.
    res = _resolved_mask(ep)
    xr = [e for e, r in zip(ep["episode"], res) if r]
    gr = [g for g, r in zip(ep["gap"], res) if r]
    xt = [e for e, r in zip(ep["episode"], res) if not r]

    ax.axhline(0.0, color="gray", lw=1, ls="--", label="min_sum baseline")
    if xr:
        ax.scatter(xr, gr, s=12, alpha=0.3, color="C0", label="resolved")
        ax.plot(xr, rolling_mean([float(g) for g in gr], w), color="C1", lw=2,
                label=f"rolling mean (w={w})")
    if xt:  # show where timeouts happened along the bottom, without polluting the curve
        ymin = min(gr) if gr else 0
        ax.scatter(xt, [ymin - 0.5] * len(xt), s=10, marker="x", color="C3",
                   alpha=0.5, label="timed out")
    ax.set_xlabel("episode"); ax.set_ylabel("gap (agent \u2212 min_sum)")
    ax.set_title("1. Comparison to min_sum\n(lower better, <0 beats baseline)")
    ax.legend(fontsize=8)


def panel_resolution(ax, ep: dict, w: int) -> None:
    x, res = ep["episode"], [float(r) for r in ep["resolved"]]
    ax.plot(x, rolling_mean(res, w), color="C2", lw=2)
    ax.scatter(x, res, s=10, alpha=0.2, color="C2")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("episode"); ax.set_ylabel("fraction resolved")
    ax.set_title("2. Resolution rate\n(reaching a unimodular fan)")


def panel_win_tie_loss(ax, ep: dict, w: int) -> None:
    # Among resolved episodes: win = gap<0, tie = gap==0, loss = gap>0.
    res = _resolved_mask(ep)
    xr = [e for e, r in zip(ep["episode"], res) if r]
    gr = [g for g, r in zip(ep["gap"], res) if r]
    if not xr:
        ax.text(0.5, 0.5, "no resolved episodes", ha="center", va="center")
        ax.set_title("3. Win / tie / loss"); return
    win = rolling_mean([1.0 if g < 0 else 0.0 for g in gr], w)
    tie = rolling_mean([1.0 if g == 0 else 0.0 for g in gr], w)
    loss = rolling_mean([1.0 if g > 0 else 0.0 for g in gr], w)
    ax.stackplot(xr, win, tie, loss,
                 labels=["beats min_sum", "ties", "worse"],
                 colors=["C2", "C7", "C3"], alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_xlabel("episode"); ax.set_ylabel("fraction (rolling)")
    ax.set_title("3. Win / tie / loss vs min_sum\n(resolved only)")
    ax.legend(fontsize=8, loc="lower left")


def panel_win_tie_loss_highdim(ax, ep: dict, w: int, min_dim: int = 3) -> None:
    # Same as panel 3 but restricted to dim >= min_dim (2D ties are unavoidable).
    res = _resolved_mask(ep)
    mask = [r and n >= min_dim for r, n in zip(res, ep["n"])]
    xr = [e for e, m in zip(ep["episode"], mask) if m]
    gr = [g for g, m in zip(ep["gap"], mask) if m]
    if not xr:
        ax.text(0.5, 0.5, f"no resolved dim≥{min_dim} episodes",
                ha="center", va="center")
        ax.set_title(f"13. Win / tie / loss (dim≥{min_dim})"); return
    win  = rolling_mean([1.0 if g < 0 else 0.0 for g in gr], w)
    tie  = rolling_mean([1.0 if g == 0 else 0.0 for g in gr], w)
    loss = rolling_mean([1.0 if g > 0 else 0.0 for g in gr], w)
    ax.stackplot(xr, win, tie, loss,
                 labels=["beats min_sum", "ties", "worse"],
                 colors=["C2", "C7", "C3"], alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_xlabel("episode"); ax.set_ylabel("fraction (rolling)")
    ax.set_title(f"13. Win / tie / loss (dim≥{min_dim} only)\n"
                 f"(2D excluded — beating min_sum is impossible there)")
    ax.legend(fontsize=8, loc="lower left")


def panel_parity(ax, ep: dict) -> None:
    # Each resolved episode: (min_sum rays, agent rays). Below y=x line = agent beat it.
    res = _resolved_mask(ep)
    b = [v for v, r in zip(ep["baseline_rays"], res) if r]
    a = [v for v, r in zip(ep["agent_rays"], res) if r]
    if not b:
        ax.text(0.5, 0.5, "no resolved episodes", ha="center", va="center")
        ax.set_title("4. Agent vs min_sum (parity)"); return
    lo = min(min(b), min(a)); hi = max(max(b), max(a))
    ax.plot([lo, hi], [lo, hi], color="gray", ls="--", lw=1, label="y = x (tie)")
    ax.scatter(_jitter(b), _jitter(a), s=14, alpha=0.4, color="C0")
    ax.set_xlabel("min_sum inserted rays"); ax.set_ylabel("agent inserted rays")
    ax.set_title("4. Agent vs min_sum (parity)\n(below line = agent used fewer)")
    ax.legend(fontsize=8)


def panel_gap_vs_mult(ax, ep: dict) -> tuple[int, int]:
    res = _resolved_mask(ep)
    rm = [m for m, r in zip(ep["mult"], res) if r]
    rg = [g for g, r in zip(ep["gap"], res) if r]
    dropped = len(ep["mult"]) - len(rm)
    ax.axhline(0.0, color="gray", lw=1, ls="--")
    ax.scatter(_jitter(rm), rg, s=14, alpha=0.4, color="C4")
    if rm:
        by: dict[int, list[int]] = {}
        for m, g in zip(rm, rg):
            by.setdefault(m, []).append(g)
        xs = sorted(by)
        ax.plot(xs, [sum(by[m]) / len(by[m]) for m in xs],
                color="C1", lw=2, marker="o", ms=4, label="mean per mult")
        ax.legend(fontsize=8)
    ax.set_xlabel("mult(\u03c3) of root cone"); ax.set_ylabel("gap (resolved only)")
    ax.set_title("5. Gap vs multiplicity")
    return len(rm), dropped


def panel_resolution_vs_mult(ax, ep: dict, nbins: int = 8) -> None:
    mult, res = ep["mult"], ep["resolved"]
    if not mult:
        ax.set_title("6. Resolution rate vs multiplicity"); return
    lo, hi = min(mult), max(mult)
    if lo == hi:
        edges = [lo, lo + 1]
    else:
        edges = np.linspace(lo, hi + 1, min(nbins, hi - lo + 1) + 1)
    centers, rates, counts = [], [], []
    for i in range(len(edges) - 1):
        sel = [r for m, r in zip(mult, res) if edges[i] <= m < edges[i + 1]]
        if sel:
            centers.append((edges[i] + edges[i + 1]) / 2)
            rates.append(sum(sel) / len(sel))
            counts.append(len(sel))
    if not centers:
        ax.set_title("6. Resolution rate vs multiplicity"); return
    width = (edges[1] - edges[0]) * 0.85
    bars = ax.bar(centers, rates, width=width, color="C2", alpha=0.8)
    for b, c in zip(bars, counts):  # annotate episode count per bucket
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, str(c),
                ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("mult(\u03c3) of root cone"); ax.set_ylabel("fraction resolved")
    ax.set_title("6. Resolution rate vs multiplicity\n(harder cones to the right)")

def panel_resolution_vs_dim(ax, ep: dict) -> None:
    dim, res = ep["n"], ep["resolved"]
    ax.set_title("7. Resolution rate vs dimension")
    if not dim:
        return

    # group episodes by exact integer dimension
    by_dim: dict[int, list] = {}
    for d, r in zip(dim, res):
        by_dim.setdefault(int(d), []).append(r)

    dims   = sorted(by_dim)                                    # every dim that occurs
    rates  = [sum(by_dim[d]) / len(by_dim[d]) for d in dims]
    counts = [len(by_dim[d]) for d in dims]

    bars = ax.bar(dims, rates, width=0.85, color="C2", alpha=0.8)
    for b, c in zip(bars, counts):                            # annotate episode count
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, str(c),
                ha="center", va="bottom", fontsize=7)

    ax.set_xticks(dims)                                       # integer ticks only
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("dim(\u03c3) of root cone")
    ax.set_ylabel("fraction resolved")


def panel_gap_hist(ax, ep: dict) -> None:
    rg = [g for g, r in zip(ep["gap"], _resolved_mask(ep)) if r]
    if not rg:
        ax.text(0.5, 0.5, "no resolved episodes", ha="center", va="center")
        ax.set_title("8. Gap distribution"); return
    lo, hi = min(rg), max(rg)
    bins = range(lo, hi + 2)
    ax.hist(rg, bins=bins, align="left", rwidth=0.85, color="C4")
    ax.axvline(0.0, color="gray", lw=1, ls="--")
    n_opt = sum(1 for g in rg if g <= 0)
    ax.set_xlabel("gap (agent \u2212 min_sum)"); ax.set_ylabel("episodes")
    ax.set_title(f"8. Gap distribution\n(\u2264baseline: {n_opt}/{len(rg)})")


def panel_calibration(ax, cal: dict) -> None:
    p, t, e = cal["predicted_value"], cal["target_value"], cal["episode"]
    if not p:
        ax.text(0.5, 0.5, "no calibration data\n(check extract_value)",
                ha="center", va="center")
        ax.set_title("9. Value-head calibration"); return
    lo, hi = min(min(p), min(t)), max(max(p), max(t))
    ax.plot([lo, hi], [lo, hi], color="gray", ls="--", lw=1, label="perfect (y=x)")
    sc = ax.scatter(t, p, s=10, alpha=0.4, c=e, cmap="viridis")
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("episode", fontsize=8)
    ax.set_xlabel("target value"); ax.set_ylabel("predicted value")
    ax.set_title("9. Value-head calibration\n(below line = optimistic; color = training time)")
    ax.legend(fontsize=8)


def panel_value_error(ax, cal: dict, w: int) -> None:
    if not cal["predicted_value"]:
        ax.text(0.5, 0.5, "no calibration data", ha="center", va="center")
        ax.set_title("10. Value-head error over training"); return
    # per-episode mean absolute error
    by: dict[int, list[float]] = {}
    for e, p, t in zip(cal["episode"], cal["predicted_value"], cal["target_value"]):
        by.setdefault(e, []).append(abs(p - t))
    eps = sorted(by)
    mae = [sum(by[e]) / len(by[e]) for e in eps]
    ax.scatter(eps, mae, s=10, alpha=0.3, color="C5")
    ax.plot(eps, rolling_mean(mae, w), color="C1", lw=2, label=f"rolling mean (w={w})")
    ax.set_xlabel("episode"); ax.set_ylabel("mean |pred \u2212 target|")
    ax.set_title("10. Value-head error over training\n(lower = better calibrated)")
    ax.legend(fontsize=8)

def panel_reward(ax, ep: dict, w: int, timeout_penalty: float = 0.0) -> None:
    # Raw episode return from the root: terminal_reward - (rays + tail)
    #   = baseline_rays - agent_rays            (resolved)
    #   = baseline_rays - agent_rays - penalty  (timed out)
    # This is the scalar the value head is trained toward at t=0; equals -gap when resolved.
    res = _resolved_mask(ep)
    x = ep["episode"]
    reward = [b - a - (0.0 if r else float(timeout_penalty))
              for a, b, r in zip(ep["agent_rays"], ep["baseline_rays"], res)]

    xr = [e for e, r in zip(x, res) if r]
    yr = [v for v, r in zip(reward, res) if r]
    xt = [e for e, r in zip(x, res) if not r]
    yt = [v for v, r in zip(reward, res) if not r]

    ax.axhline(0.0, color="gray", lw=1, ls="--", label="reward = 0 (ties min_sum)")
    if xr:
        ax.scatter(xr, yr, s=10, alpha=0.3, color="C0", label="resolved")
    if xt:
        ax.scatter(xt, yt, s=10, alpha=0.3, color="C3", marker="x", label="timed out")
    ax.plot(x, rolling_mean([float(v) for v in reward], w), color="C1", lw=2,
            label=f"rolling mean (w={w})")
    pen = "" if not timeout_penalty else f"; timeout penalty {timeout_penalty:g}"
    ax.set_xlabel("episode")
    ax.set_ylabel("episode return  (terminal_reward \u2212 rays)")
    ax.set_title(f"11. Raw episode reward over training\n"
                 f"(higher better; >0 beats min_sum{pen})")
    ax.legend(fontsize=8, ncol=2)

def panel_train_loss(ax, tl: dict, w: int) -> None:
    x = tl["episode"]
    if not x:
        ax.text(0.5, 0.5, "no train_loss.csv\n(no training steps logged yet)",
                ha="center", va="center")
        ax.set_title("12. Training loss"); return
    series = [("loss", "total", "C0"),
              ("policy_loss", "policy", "C1"),
              ("value_loss", "value", "C3")]
    for key, label, color in series:
        y = tl[key]
        # ax.scatter(x, y, s=8, alpha=0.2, color=color)
        ax.plot(x, rolling_mean(y, w), color=color, lw=2, label=label)
    # log-y only if every value is strictly positive (a logged 0.0 means a dead loss)
    if all(v > 0 for k in ("loss", "policy_loss", "value_loss") for v in tl[k]):
        ax.set_yscale("log")
    ax.set_xlabel("episode"); ax.set_ylabel("loss (mean over epochs_per_iter)")
    ax.set_title("12. Training loss over training\n(fit-to-replay; total = policy + w\u00b7value)")
    ax.legend(fontsize=8)

# ===========================================================================================
#  INITIAL CONE PANELS
# ===========================================================================================

def panel_mult_hist(ax, ep: dict) -> None:
    mult = ep["mult"]
    if not mult:
        ax.set_title("Initial-cone multiplicity"); return
    lo, hi = min(mult), max(mult)
    bins = range(lo, hi + 2)
    ax.hist(mult, bins=bins, align="left", rwidth=0.85, color="C0")
    ax.set_xlabel("mult(\u03c3) of initial cone"); ax.set_ylabel("episodes")
    ax.set_title(f"Initial-cone multiplicity\n({len(mult)} cones, "
                 f"min {lo} / max {hi})")


def panel_dim_hist(ax, ep: dict) -> None:
    n = ep["n"]
    if not n:
        ax.set_title("Initial-cone dimension"); return
    lo, hi = min(n), max(n)
    bins = range(lo, hi + 2)
    counts, _, _ = ax.hist(n, bins=bins, align="left", rwidth=0.7, color="C2")
    ax.set_xticks(range(lo, hi + 1))
    ax.set_xlabel("dimension n of initial cone"); ax.set_ylabel("episodes")
    ax.set_title(f"Initial-cone dimension\n({len(n)} cones, n \u2208 [{lo}, {hi}])")

def load_config(directory: str, config_path: Optional[str],
                checkpoint_path: Optional[str]) -> Optional[dict]:
    """Find the run's hyperparameters. Priority:
      1. explicit --config JSON
      2. <diag-dir>/config.json  (if you ever dump one)
      3. the saved checkpoint's "args" (train.py stores vars(args) there)
    Returns None if nothing is found."""
    import json
    if config_path:
        try:
            with open(config_path) as f:
                return dict(json.load(f))
        except Exception as e:
            print(f"[config] could not read {config_path}: {e}")

    j = os.path.join(directory, "config.json")
    if os.path.exists(j):
        try:
            with open(j) as f:
                return dict(json.load(f))
        except Exception as e:
            print(f"[config] could not read {j}: {e}")

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            import torch  # lazy: only needed if reading args from a .pt
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "args" in ckpt:
                return dict(ckpt["args"])
        except Exception as e:
            print(f"[config] could not read args from {checkpoint_path}: {e}")
    return None


# hyperparameters worth surfacing first; the rest follow alphabetically
_CONFIG_PRIORITY = [
    "episodes", "mcts_sims", "max_steps", "max_dimension", "det_max",
    "c_puct", "temperature", "dirichlet_alpha", "dirichlet_eps", "timeout_penalty",
    "lr", "batch_size", "value_weight", "hidden_dim", "embedding_size",
    "num_blocks", "dropout", "epochs_per_iter", "replay_capacity", "seed", "device",
]


def render_config(ax, cfg: Optional[dict], ncols: int = 4) -> None:
    ax.axis("off")
    if not cfg:
        ax.text(0.5, 0.5, "no hyperparameters found\n"
                "(pass --config <json> or --checkpoint <.pt>)",
                ha="center", va="center", fontsize=10, color="gray")
        ax.set_title("Hyperparameters", fontsize=11, loc="left")
        return

    # order: priority keys first (those present), then any remaining keys sorted
    keys = [k for k in _CONFIG_PRIORITY if k in cfg]
    keys += [k for k in sorted(cfg) if k not in _CONFIG_PRIORITY]
    items = [f"{k} = {cfg[k]}" for k in keys]

    width = max(len(s) for s in items)
    items = [s.ljust(width) for s in items]
    nrows = -(-len(items) // ncols)  # ceil
    lines = []
    for r in range(nrows):
        cells = [items[r + c * nrows] for c in range(ncols) if r + c * nrows < len(items)]
        lines.append("    ".join(cells))
    text = "\n".join(lines)

    ax.set_title("Hyperparameters", fontsize=11, loc="left")
    ax.text(0.0, 1.0, text, ha="left", va="top", family="monospace", fontsize=8.5,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#cccccc"))


def render_note(ax, note: Optional[str]) -> None:
    """Draw the run note in its own box, sized to sit right of the hyperparameters."""
    ax.axis("off")
    ax.set_title("Note", fontsize=11, loc="left")
    if not note:
        return
    ax.text(0.0, 1.0, note, ha="left", va="top", fontsize=9.5, color="#333333",
            wrap=True, transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="#fff7cc", edgecolor="#cccccc"))

def write_initial_cones(ep: dict, out_png: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    panel_mult_hist(axes[0], ep)
    panel_dim_hist(axes[1], ep)
    fig.suptitle(f"Initial cones  |  {len(ep['episode'])} episodes", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- main
def _latest_run_dir(base: str = "results") -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_abs = base if os.path.isabs(base) else os.path.join(script_dir, base)
    if os.path.isdir(base_abs):
        runs = sorted(d for d in os.listdir(base_abs) if os.path.isdir(os.path.join(base_abs, d)))
        if runs:
            return os.path.join(base_abs, runs[-1])
    return base_abs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--directory", default=None,
                    help="run directory holding the CSVs (default: latest in diagnostics/results/)")
    ap.add_argument("--episode-csv", default=None, help="override path")
    ap.add_argument("--calibration-csv", default=None, help="override path")
    ap.add_argument("--out", default=None, help="output PNG (default <diag-dir>/report.png)")
    ap.add_argument("--rolling", type=int, default=20, help="rolling-window width")
    ap.add_argument("--config", default=None,
                    help="JSON file of hyperparameters to display (optional)")
    ap.add_argument("--checkpoint", default="cone_action_gnn.pt",
                    help="model .pt to read hyperparameters from if --config absent")
    ap.add_argument("--split", action="store_true",
                    help="also write each panel as its own PNG")
    ap.add_argument("--note", default=None,
                    help="note to display in the bottom-right of report.png "
                         "(if omitted, you will be prompted to type one)")
    args = ap.parse_args()

    # ask the user for a note unless one was passed on the command line
    if args.note is None:
        try:
            args.note = input("Note for report.png (blank for none): ").strip()
        except EOFError:  # non-interactive (piped) invocation
            args.note = ""
    if args.directory is None:
        args.directory = _latest_run_dir()

    ep_path = args.episode_csv or os.path.join(args.directory, "episode_quality.csv")
    cal_path = args.calibration_csv or os.path.join(args.directory, "value_calibration.csv")
    out_png = args.out or os.path.join(args.directory, "report.png")

    ep = load_episode_csv(ep_path)
    cal = load_calibration_csv(cal_path)
    tl = load_loss_csv(os.path.join(args.directory, "train_loss.csv"))
    n_ep = len(ep["episode"])
    if n_ep == 0:
        raise SystemExit(f"no episodes found in {ep_path}")
    w = max(1, min(args.rolling, n_ep))
    cfg = load_config(args.directory, args.config, args.checkpoint)

    # 4x3 panel grid + 1 wide extra row + config strip
    fig = plt.figure(figsize=(18, 22))
    gs = fig.add_gridspec(6, 3, height_ratios=[1, 1, 1, 1, 0.7, 0.42], hspace=0.5, wspace=0.25)
    axes = np.empty((4, 3), dtype=object)
    for r in range(4):
        for c in range(3):
            axes[r, c] = fig.add_subplot(gs[r, c])

    panel_gap_curve(axes[0, 0], ep, w)
    panel_resolution(axes[0, 1], ep, w)
    panel_win_tie_loss(axes[0, 2], ep, w)
    panel_parity(axes[1, 0], ep)
    n_res, dropped = panel_gap_vs_mult(axes[1, 1], ep)
    panel_resolution_vs_mult(axes[1, 2], ep)
    panel_resolution_vs_dim(axes[2, 0], ep)
    panel_gap_hist(axes[2, 1], ep)
    panel_calibration(axes[2, 2], cal)
    panel_value_error(axes[3, 0], cal, w)

    timeout_penalty = float(cfg.get("timeout_penalty", 0.0)) if cfg else 0.0
    panel_reward(axes[3, 1], ep, w, timeout_penalty=timeout_penalty)  # panel 10

    panel_train_loss(axes[3, 2], tl, w)  # panel 11

    ax_highdim = fig.add_subplot(gs[4, :])
    panel_win_tie_loss_highdim(ax_highdim, ep, w)  # panel 13

    cfg_ax = fig.add_subplot(gs[5, :2])
    render_config(cfg_ax, cfg)

    # note lives in its own box directly to the right of the hyperparameters
    note_ax = fig.add_subplot(gs[5, 2])
    render_note(note_ax, args.note)

    res_rate = sum(ep["resolved"]) / n_ep
    win  = sum(1 for g, r in zip(ep["gap"], ep["resolved"]) if r and g < 0)
    ties = sum(1 for g, r in zip(ep["gap"], ep["resolved"]) if r and g == 0)

    timing = load_timing(args.directory, n_ep)
    timing_str = ""
    if timing:
        approx = "~" if timing["approximate"] else ""
        timing_str = (f"  |  elapsed {approx}{_fmt_hms(timing['elapsed_seconds'])} "
                      f"({approx}{timing['sec_per_episode']:.2f} s/ep)")

    fig.suptitle(
        f"Diagnostics report  |  {n_ep} episodes  |  resolved {res_rate:.0%}  "
        f"|  beats min_sum {win}/{n_res} resolved ({win/n_res:.0%})"
        f"|  ties min_sum {ties}/{n_res} resolved ({ties/n_res:.0%})  "
        f"|  gap panels drop {dropped} timed-out"
        f"{timing_str}",
        fontsize=10)

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_png}")

    # separate figure: distribution of the INITIAL cones fed to self-play
    init_png = os.path.join(args.directory, "initial_cones.png")
    write_initial_cones(ep, init_png)
    print(f"wrote {init_png}")

    if args.split:
        panels = [
            ("gap_curve", lambda a: panel_gap_curve(a, ep, w)),
            ("resolution_rate", lambda a: panel_resolution(a, ep, w)),
            ("win_tie_loss", lambda a: panel_win_tie_loss(a, ep, w)),
            ("parity", lambda a: panel_parity(a, ep)),
            ("gap_vs_mult", lambda a: panel_gap_vs_mult(a, ep)),
            ("resolution_vs_mult", lambda a: panel_resolution_vs_mult(a, ep)),
            ("resolution_vs_dim", lambda a: panel_resolution_vs_dim(a, ep)),
            ("gap_hist", lambda a: panel_gap_hist(a, ep)),
            ("calibration", lambda a: panel_calibration(a, cal)),
            ("value_error", lambda a: panel_value_error(a, cal, w)),
            ("reward", lambda a: panel_reward(
                a, ep, w, float(cfg.get("timeout_penalty", 0.0)) if cfg else 0.0)),
            ("train_loss", lambda a: panel_train_loss(a, tl, w)),
            ("initial_mult", lambda a: panel_mult_hist(a, ep)),
            ("initial_dim", lambda a: panel_dim_hist(a, ep)),
        ]
        for name, fn in panels:
            f1, a1 = plt.subplots(figsize=(6, 5))
            fn(a1)
            f1.tight_layout()
            p = os.path.join(args.directory, f"panel_{name}.png")
            f1.savefig(p, dpi=150); plt.close(f1)
            print(f"wrote {p}")

    plt.close(fig)


if __name__ == "__main__":
    main()