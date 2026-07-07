#!/usr/bin/env python3
"""Evaluate trained models on the validation depot.

Two modes, both driving a deterministic GREEDY (argmax-policy, no MCTS) rollout
on every validation cone that passes the dimension/determinant filter
(``min_dimension <= n <= max_dimension`` AND ``det_min <= d <= det_max``).  Per cone we record the
agent's subdivision count, whether it resolved, and the gap vs the precomputed
min_sum baseline stored in the depot.

  SWEEP MODE (--sweep-dir):  evaluate every t*/model.pt checkpoint under a
    hyperparam.py sweep folder (grid OR linear -- both use the same t###_s#
    layout) and render a comparison grid across a chosen hyperparameter, plus a
    per-model CSV and a records.json.  Outputs land in the sweep folder.

  SINGLE-MODEL MODE (--model):  render a read_diagnostics-style per-cone report
    for ONE model -- a sweep trial dir, a plain train.py / read_diagnostics run
    folder (results/<timestamp>/), or a model.pt path.  Outputs land in that
    model's own folder, so a run evaluated straight out of results/ keeps its
    report beside its training diagnostics.

Parameters:
  --sweep-dir DIR    sweep folder holding the t*/model.pt checkpoints to evaluate
  --model PATH       single-model mode: trial dir, run folder, or model.pt
  --dataset NAME     which depot to evaluate on: validation | test   (default validation)
  --min-dimension N  keep only depot cones with dimension n >= N        (default 2)
  --max-dimension N  keep only depot cones with dimension n <= N        (default 4)
  --det-min D        keep only depot cones with determinant d >= D      (default 2)
  --det-max D        keep only depot cones with determinant d <= D      (default 20)
  --max-steps       max greedy subdivisions per cone before unresolved (default 40)
  --group-by HPARAM  any train.py hyperparameter to compare models across (lr,
                     c_puct, layer_type, ...). Default: auto-detect the sweep's
                     varied axis from sweep_meta.json, else the arg that varies
                     across the evaluated models.
  --workers W        parallel evaluation processes for SWEEP mode; single-model
                     mode (--model) always rolls out serially on one thread
                                                                     (default 1)
  --out PATH         output PNG path (records.json/CSV derived from it)
  --replot           rebuild the PNG from a saved records.json, no rollouts
                     (with --group-by, re-groups the comparison on a new axis)
"""
import argparse, glob, json, os, statistics, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# fixed categorical series colors, in CVD-safe order (blue, aqua, yellow, green,
# violet, red, magenta, orange); assign by slot, never cycle
CATEGORICAL = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
               "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]

# selectable evaluation datasets (same cone schema; --dataset picks one)
DEPOTS = {
    "validation": os.path.join(HERE, "..", "datasets", "validation_set-3.json"),
    "test":       os.path.join(HERE, "..", "datasets", "test_set.json"),
}


def load_depot(dataset, min_dimension, max_dimension, det_min, det_max):
    depot = json.load(open(DEPOTS[dataset]))
    return [c for c in depot
            if min_dimension <= c["n"] <= max_dimension and det_min <= c["d"] <= det_max]


# per-run identity / environment fields -- never used as a grouping axis
_NONGROUP = {"save", "diag_dir", "resume", "device", "seed"}


def heatmap_cmap(lower):
    """Red/blue diverging colormap oriented so GOOD cells read blue, BAD red."""
    return "RdBu_r" if lower else "RdBu"   # RdBu_r: low->blue, high->red


# Fixed value ranges so a heatmap's colors reflect ABSOLUTE quality rather than each
# panel's own spread: an all-good grid reads all-blue, an all-bad grid all-red.
# Rates are naturally [0,1]; gap/score use a fixed domain anchored at 0 = perfect.
HEATMAP_RANGE = {
    "score": (0.0, 15.0), "res_rate": (0.0, 1.0), "resolution_rate": (0.0, 1.0),
    "mean_gap": (0.0, 12.0), "tie": (0.0, 1.0), "tie_rate": (0.0, 1.0),
}


def heatmap_range(field):
    """Fixed (vmin, vmax) for a heatmap metric, or (None, None) to auto-scale."""
    return HEATMAP_RANGE.get(field, (None, None))


def cell_text_color(im, value):
    """Black or white label, whichever reads on this cell's fill.  Luminance-based
    so it stays correct for the diverging red/blue map (dark at both extremes,
    near-white at the midpoint)."""
    r, g, b, _ = im.cmap(im.norm(value))
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.6 else "black"


def group_value(m, key):
    """The value of hyperparameter `key` for one evaluated model, read from its
    stored train.py args (falling back to the top-level record for the legacy
    embedding_size / layer_type / seed fields)."""
    args = m.get("args") or {}
    return args[key] if key in args else m.get(key)


def infer_group_key(models):
    """Pick a grouping axis from the models themselves: the train.py arg that
    actually varies across the evaluated models, preferring the one with the most
    distinct values (ties broken alphabetically).  None if nothing varies."""
    args = [m.get("args", {}) for m in models if m.get("args")]
    if not args:
        return None
    keys = set.intersection(*(set(a) for a in args)) - _NONGROUP
    varying = [(len({a[k] for a in args}), k) for k in keys
               if len({a[k] for a in args}) > 1]
    if not varying:
        return None
    varying.sort(key=lambda t: (-t[0], t[1]))
    return varying[0][1]


def detect_group_key(sweep_dir):
    """Grouping axis for the comparison plots, from sweep_meta.json: the sweep's
    varied axis (or the first, if a grid varied several).  None when there is no
    meta file, so the caller infers it from the evaluated models instead."""
    try:
        axes = json.load(open(os.path.join(sweep_dir, "sweep_meta.json"))).get("axes", [])
        if axes:
            return axes[0]
    except (OSError, ValueError):
        pass
    return None


def detect_grid(sweep_dir):
    """Return the two grid axes if this sweep was a 2-axis grid search (from
    sweep_meta.json), else None -- signalling the grid-style evaluation report."""
    try:
        meta = json.load(open(os.path.join(sweep_dir, "sweep_meta.json")))
        if meta.get("mode") == "grid":
            axes = meta.get("axes") or []
            if len(axes) == 2:
                return list(axes)
    except (OSError, ValueError):
        pass
    return None


def eval_one_model(model_path, cones, max_steps, progress=False):
    """Runs in a worker process: load model, greedy-rollout every cone.
    progress=True shows a live per-cone bar (single-model mode only; the sweep
    grid runs many workers in parallel and would interleave bars)."""
    import torch
    from Cone import Cone
    from CGLGraph import CGLGraph
    from network import network
    from utils import validActionMask
    torch.set_num_threads(1)

    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return {"path": model_path, "error": f"load: {e}"}
    a = ckpt["args"]
    ## feature width must match training: padded runs store it in "padding",
    ## older checkpoints fall back to max_dimension
    dim = a.get("padding") or a["max_dimension"]

    ## a cone only fits in the model's input layer if its dimension <= feature width
    n_skipped = sum(1 for c in cones if c["n"] > dim)
    if n_skipped:
        print(f"[{os.path.basename(model_path)}] skipping {n_skipped}/{len(cones)} cones "
              f"with n > feature width {dim} (model cannot represent them)", flush=True)
        cones = [c for c in cones if c["n"] <= dim]
    if not cones:
        return {"path": model_path, "error": f"no depot cones fit feature width {dim}"}

    def build(cone):
        g = CGLGraph(dimension=dim)
        g.addConeNode(Cone(tuple(tuple(r) for r in cone["rays"])))
        return g

    try:
        model = network(build(cones[0]).metadata(), hidden=a["hidden_dim"],
                        embedding_size=a["embedding_size"], num_layers=a["num_blocks"],
                        dropout=a["dropout"], layer_type=a.get("layer_type", "GAT"))
        with torch.no_grad():
            model(build(cones[0]).toHeteroData())      # materialize lazy params
        model.load_state_dict(ckpt["model"])
        model.eval()
    except Exception as e:
        return {"path": model_path, "error": f"build: {e}"}

    @torch.no_grad()
    def rollout(cone):
        g = build(cone); steps = 0
        for _ in range(max_steps):
            if g.isDecomposed():
                return steps, True
            out = model(g.toHeteroData())
            vidx = validActionMask(g).nonzero(as_tuple=False).flatten().tolist()
            if not vidx:
                return steps, False
            acts = [g._lattice_idx_to_id[i] for i in vidx]
            g.subdivide(acts[int(torch.argmax(out["log_p"]))])
            steps += 1
        return steps, g.isDecomposed()

    recs = []
    n_res = n_tie = fwd = 0
    bar = tqdm(cones, desc=os.path.basename(os.path.dirname(model_path)),
               unit="cone", disable=not progress)
    for c in bar:
        s, r = rollout(c)
        recs.append((c["n"], c["d"], c["min_sum_steps"], s, int(r)))
        if progress:
            n_res += r; n_tie += int(r and s == c["min_sum_steps"]); fwd += s
            bar.set_postfix(n=c["n"], d=c["d"], forwards=fwd,
                            res=f"{n_res/len(recs):.0%}",
                            tie=f"{n_tie/max(n_res,1):.0%}", refresh=False)
    return {"path": model_path, "embedding_size": a["embedding_size"],
            "layer_type": a.get("layer_type", "GAT"),
            "seed": a.get("seed", -1), "args": dict(a), "records": recs}


def shard_cones(cones, n_shards):
    """Split the depot into n_shards lists of cone INDICES, balanced by
    difficulty: cones are dealt round-robin in descending min_sum_steps order,
    so every shard gets an equal cut of the hard and the easy cones and no
    single shard becomes the straggler that bounds wall time."""
    order = sorted(range(len(cones)),
                   key=lambda i: -(cones[i].get("min_sum_steps") or 0))
    return [order[k::n_shards] for k in range(n_shards)]


def n_shards_for(n_cones, workers, n_models=1):
    """Shards per model: ~4 tasks per worker over the whole job, so slow shards
    load-balance.  Per-task overhead (one checkpoint load) is bounded at ~4
    loads per worker regardless of depot size, so shards may be tiny."""
    return max(1, min(-(-4 * workers // max(1, n_models)), n_cones))


def merge_shard_results(shard_results, shards, cones):
    """Reassemble one model's per-shard eval results into a single result blob
    with records back in depot order.  shard_results holds (shard_idx, result)
    for ALL of the model's shards; shards holds each shard's depot indices.
    Cones a shard skipped (n > the model's feature width -- mirroring
    eval_one_model's filter) are dropped, exactly as a serial rollout would."""
    ok = [(k, r) for k, r in shard_results if "error" not in r]
    if not ok:
        # every shard failed alike (they all load the same checkpoint)
        return shard_results[0][1]
    merged = [None] * len(cones)
    for k, r in ok:
        a = r["args"]
        dim = a.get("padding") or a["max_dimension"]
        kept = [i for i in shards[k] if cones[i]["n"] <= dim]
        for i, rec in zip(kept, r["records"]):
            merged[i] = rec
    return {**ok[0][1], "records": [rec for rec in merged if rec is not None]}


def summarize(res):
    """Per-model scalar metrics from its per-cone records."""
    recs = res["records"]
    n = len(recs)
    resolved = [(ms, s) for (_, _, ms, s, r) in recs if r]
    gaps = [s - ms for (ms, s) in resolved]
    res_rate = len(resolved) / n
    mean_gap = statistics.mean(gaps) if gaps else float("nan")
    tie = (sum(g == 0 for g in gaps) / len(gaps)) if gaps else 0.0
    beat = (sum(g < 0 for g in gaps) / len(gaps)) if gaps else 0.0
    score = (mean_gap if gaps else 0.0) + 5.0 * (1.0 - res_rate)
    return dict(res_rate=res_rate, mean_gap=mean_gap, tie=tie, beat=beat, score=score)


def constant_hparams(models, group_key):
    """Hyperparameters held fixed across every evaluated model (same value everywhere),
    excluding per-run identity fields and the swept axis itself."""
    args = [m.get("args", {}) for m in models if m.get("args")]
    if not args:
        return {}
    skip = _NONGROUP | {group_key}
    keys = set.intersection(*(set(a) for a in args))
    consts = {}
    for k in sorted(keys):
        if k in skip:
            continue
        vals = {a[k] for a in args}
        if len(vals) == 1:
            consts[k] = next(iter(vals))
    return consts


def _nrange(meta):
    """'min_dimension<=n<=max_dimension' string for titles, tolerant of records without min_dimension."""
    return f"{meta.get('min_dimension', '?')}≤n≤{meta['max_dimension']}"


def make_png(models, out_png, meta):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    gkey = meta.get("group_key", "embedding_size")     # which swept axis to compare across
    by_grp = defaultdict(list)
    for m in models:
        v = group_value(m, gkey)
        by_grp["n/a" if v is None else v].append(m)
    # numeric axes (lr, embedding_size, ...) sort/scale numerically; categorical
    # (layer_type) and booleans (early_termination) are treated as discrete labels.
    # int-coded enums (reward_type) and any axis containing a value <= 0 are also
    # categorical — a log axis silently drops those points
    def _numeric(g):
        return isinstance(g, (int, float)) and not isinstance(g, bool)
    groups = sorted(by_grp, key=lambda g: (not _numeric(g), g if _numeric(g) else str(g)))
    numeric = (all(_numeric(g) and g > 0 for g in groups)
               and gkey not in {"reward_type"})
    xpos = {g: (g if numeric else i) for i, g in enumerate(groups)}
    # fixed categorical order (CVD-validated); falls back to viridis past 8 groups
    palette = (CATEGORICAL[:len(groups)] if len(groups) <= len(CATEGORICAL)
               else plt.cm.viridis(np.linspace(0, 1, len(groups))))
    color = {g: palette[i] for i, g in enumerate(groups)}

    def style_x(a, ylabel):
        if numeric:
            a.set_xscale("log")
        else:
            a.set_xticks([xpos[g] for g in groups]); a.set_xticklabels([str(g) for g in groups])
            a.set_xlim(-0.5, len(groups) - 0.5)
        a.set_xlabel(gkey); a.set_ylabel(ylabel)

    def agg(metric):
        out = {}
        for g in groups:
            vals = [summarize(m)[metric] for m in by_grp[g]]
            vals = [v for v in vals if v == v]  # drop nan
            if vals:
                out[g] = (statistics.mean(vals),
                          statistics.pstdev(vals) if len(vals) > 1 else 0.0, len(vals))
        return out

    rng = np.random.default_rng(0)

    def real_points(a, g, metric, marker="o"):
        ## one small transparent dot per model (seed) behind the group mean
        vals = [v for v in (summarize(m)[metric] for m in by_grp[g]) if v == v]
        if numeric:  # log x-axis: jitter multiplicatively
            xs = xpos[g] * (1 + rng.uniform(-0.04, 0.04, len(vals)))
        else:
            xs = xpos[g] + rng.uniform(-0.09, 0.09, len(vals))
        a.scatter(xs, vals, s=16, alpha=0.45, color=color[g], marker=marker,
                  edgecolors="0.4", linewidths=0.4, zorder=2)

    fig, ax = plt.subplots(2, 3, figsize=(19, 11))
    fig.suptitle(f"{meta.get('dataset', 'validation')}-set eval — greedy policy (no MCTS)  |  "
                 f"{_nrange(meta)}, {meta['det_min']}≤d≤{meta['det_max']}, "
                 f"max_steps={meta.get('max_steps', '?')}  "
                 f"({meta['n_cones']} cones)  |  mean over seeds  |  {meta['n_models']} models",
                 fontsize=15, fontweight="bold")

    def scalar_panel(a, metric, title, ylabel, lower=True):
        d = agg(metric)
        gs = [g for g in groups if g in d]
        a.plot([xpos[g] for g in gs], [d[g][0] for g in gs], color="0.75", lw=1, zorder=1)  # trend
        for g in gs:
            m, s, k = d[g]
            a.errorbar([xpos[g]], [m], yerr=[s], marker="o", ms=10, capsize=4,
                       color=color[g], ecolor=color[g], zorder=3)
            real_points(a, g, metric)
            a.annotate(f"n={k}", (xpos[g], m), textcoords="offset points", xytext=(7, 7), fontsize=7)
        if gs:
            best = (min if lower else max)(gs, key=lambda g: d[g][0])
            a.scatter([xpos[best]], [d[best][0]], s=230, facecolors="none",
                      edgecolors="red", lw=2, zorder=5)
        style_x(a, ylabel)
        a.set_title(title + ("  (lower better)" if lower else "  (higher better)"))
        a.grid(alpha=0.3)

    scalar_panel(ax[0, 0], "score",    f"Sweep-style score vs {gkey}", "score", lower=True)
    scalar_panel(ax[0, 1], "res_rate", f"Resolution rate vs {gkey}",   "resolved fraction", lower=False)
    scalar_panel(ax[0, 2], "mean_gap", "Mean gap vs min_sum (resolved)",      "mean gap", lower=True)

    # tie (circle) + beat (square), both colored per group
    a = ax[1, 0]
    dt, db = agg("tie"), agg("beat")
    gs = [g for g in groups if g in dt]
    a.plot([xpos[g] for g in gs], [dt[g][0] for g in gs], color="0.75", lw=1, zorder=1)
    a.plot([xpos[g] for g in gs], [db[g][0] for g in gs], color="0.85", lw=1, ls="--", zorder=1)
    for g in gs:
        a.errorbar([xpos[g]], [dt[g][0]], yerr=[dt[g][1]], marker="o", ms=10, capsize=4, color=color[g], zorder=3)
        a.errorbar([xpos[g]], [db[g][0]], yerr=[db[g][1]], marker="s", ms=9,  capsize=4, color=color[g], zorder=3)
        real_points(a, g, "tie", marker="o")
        real_points(a, g, "beat", marker="s")
    style_x(a, "fraction of resolved")
    a.set_title("Tie (circle) / win (square) vs min_sum")
    a.legend(handles=[Line2D([], [], marker="o", color="0.4", ls="", label="tie (gap=0)"),
                      Line2D([], [], marker="s", color="0.4", ls="", label="beat min_sum (gap<0)")],
             fontsize=8)
    a.grid(alpha=0.3)

    W = 0.8 / max(1, len(groups))

    def grouped_bars(a, nbin, per_model_vec):
        """Grouped bars whose height is the mean over a group's models and whose
        error bar is the std (pstdev) across those models, per bin.
        per_model_vec(m) -> length-nbin array (or None to skip the model)."""
        for j, g in enumerate(groups):
            vecs = [v for v in (per_model_vec(m) for m in by_grp[g]) if v is not None]
            if not vecs:
                continue
            arr = np.vstack(vecs)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0) if arr.shape[0] > 1 else np.zeros(nbin)  # population std over models
            a.bar(np.arange(nbin) + j * W, mean, width=W, yerr=np.nan_to_num(std), color=color[g],
                  label=str(g), capsize=2, error_kw=dict(lw=0.8, alpha=0.7))

    # resolution vs determinant (binned) grouped; bar = mean over models, err = std over models
    # integer-width buckets sized to the observed d range (all models share the same cone depot)
    a = ax[1, 1]
    all_d = [d for m in models for (_, d, _, _, _) in m["records"]]
    d_lo, d_hi = (min(all_d), max(all_d)) if all_d else (meta["det_min"], meta["det_max"])
    bucket = max(1, -(-(d_hi - d_lo + 1) // 6))
    bins = [(s, min(s + bucket - 1, d_hi)) for s in range(d_lo, d_hi + 1, bucket)]
    labels = [f"{lo}" if lo == hi else f"{lo}-{hi}" for lo, hi in bins]

    def res_by_det(m):
        out = []
        for lo, hi in bins:
            vals = [r for (_, d, _, _, r) in m["records"] if lo <= d <= hi]
            out.append(statistics.mean(vals) if vals else np.nan)  # nan -> bar skipped
        return np.array(out, float)

    grouped_bars(a, len(bins), res_by_det)
    a.set_xticks(np.arange(len(bins)) + 0.4 - W / 2); a.set_xticklabels(labels, fontsize=8)
    a.set_xlabel("determinant d (binned)"); a.set_ylabel("fraction resolved")
    a.set_title("Resolution rate vs determinant  (±std over models)")

    # gap distribution grouped; bar = mean over models, err = std over models
    a = ax[1, 2]
    gbins = [-2, 0, 2, 5, 10, 20, 40, 80]
    glabels = ["<0", "0-1", "2-4", "5-9", "10-19", "20-39", "40+"]

    def gap_dist(m):
        gaps = [s - ms for (_, _, ms, s, r) in m["records"] if r]
        if not gaps:
            return None
        counts = np.array([sum(gbins[k] <= gg < gbins[k + 1] for gg in gaps)
                           for k in range(len(gbins) - 1)], float)
        return counts / len(gaps)

    grouped_bars(a, len(glabels), gap_dist)
    a.set_xticks(np.arange(len(glabels)) + 0.4 - W / 2); a.set_xticklabels(glabels, fontsize=8)
    a.set_xlabel("gap (agent - min_sum)"); a.set_ylabel("fraction of resolved episodes")
    a.set_title("Gap distribution  (left/lower better; ±std over models)")

    # one shared legend for the whole figure
    handles = [Line2D([], [], marker="o", color=color[g], ls="", ms=9, label=str(g)) for g in groups]
    fig.legend(handles=handles, title=gkey, loc="upper right",
               ncol=len(groups), fontsize=9, framealpha=0.9)

    # footer: every hyperparameter held fixed across the sweep (training range,
    # model/optim settings, etc.) so the plot is self-documenting
    consts = meta.get("constants", {})
    bottom = 0.0
    if consts:
        items = [f"{k}={v}" for k, v in consts.items()]
        per_line = 6
        lines = ["   ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]
        footer = "constant hyperparameters (held fixed across sweep):\n" + "\n".join(lines)
        bottom = 0.015 + 0.021 * (len(lines) + 1)
        fig.text(0.5, 0.01, footer, ha="center", va="bottom",
                 fontsize=7.5, family="monospace", color="0.2")

    fig.tight_layout(rect=[0, bottom, 1, 0.95])
    fig.savefig(out_png, dpi=130)
    print(f"wrote {out_png}")


def _grid_sort_key(key):
    def numeric(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    return tuple((not numeric(v), v if numeric(v) else str(v)) for v in key)


def make_grid_png(models, out_png, meta):
    """Grid-search evaluation report: one figure covering every (axis0, axis1)
    configuration.  Per-configuration bars (score / resolution / mean gap /
    tie+win), per-configuration resolution-rate and gap-distribution vs
    determinant, and one heatmap per statistic over the two swept axes.  All
    stats are computed on the depot rollouts, meaned over seeds."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    axes = meta["axes"]                                   # exactly two grid axes
    ax0, ax1 = axes

    def cfg_key(m):
        return tuple(group_value(m, a) for a in axes)
    by_cfg = defaultdict(list)
    for m in models:
        by_cfg[cfg_key(m)].append(m)
    configs = sorted(by_cfg, key=_grid_sort_key)
    labels = ["\n".join(f"{a}={v}" for a, v in zip(axes, k)) for k in configs]
    xs = np.arange(len(configs))
    colors = (CATEGORICAL[:len(configs)] if len(configs) <= len(CATEGORICAL)
              else plt.cm.plasma(np.linspace(0.05, 0.95, len(configs))))

    def cfg_agg(metric):
        ms, sds = [], []
        for k in configs:
            vals = [summarize(m)[metric] for m in by_cfg[k]]
            vals = [v for v in vals if v == v]
            ms.append(statistics.mean(vals) if vals else np.nan)
            sds.append(statistics.pstdev(vals) if len(vals) > 1 else 0.0)
        return np.array(ms), np.array(sds)

    n_seed = max((len(v) for v in by_cfg.values()), default=1)
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(3, 4, hspace=0.75, wspace=0.32, height_ratios=[1, 1, 1.1])
    fig.suptitle(
        f"{meta.get('dataset', 'validation')}-set grid eval — greedy policy (no MCTS)  |  "
        f"{ax0} × {ax1}  |  {len(configs)} configs  |  {_nrange(meta)}, "
        f"{meta['det_min']}≤d≤{meta['det_max']}  ({meta['n_cones']} cones)  |  "
        f"{'mean of ' + str(n_seed) + ' seeds' if n_seed > 1 else 'single seed'}",
        fontsize=15, fontweight="bold")

    # ---- row 0: per-configuration scalar bars ----
    def bar_panel(a, metric, title, ylabel, lower=True):
        m, sd = cfg_agg(metric)
        a.bar(xs, m, yerr=sd, color=colors, capsize=3, error_kw=dict(lw=0.8, alpha=0.7))
        finite = [(x, v) for x, v in zip(xs, m) if v == v]
        if finite:
            bx, bv = (min if lower else max)(finite, key=lambda p: p[1])
            a.scatter([bx], [bv], s=170, facecolors="none", edgecolors="#00b140", lw=2, zorder=5)
        a.set_xticks(xs); a.set_xticklabels(labels, fontsize=6.5, rotation=90)
        a.set_ylabel(ylabel); a.grid(alpha=0.3, axis="y")
        a.set_title(title + ("  (lower better)" if lower else "  (higher better)"), fontsize=10)

    bar_panel(fig.add_subplot(gs[0, 0]), "score", "Score vs configuration", "score", lower=True)
    bar_panel(fig.add_subplot(gs[0, 1]), "res_rate", "Resolution rate vs configuration", "resolved frac", lower=False)
    bar_panel(fig.add_subplot(gs[0, 2]), "mean_gap", "Mean gap vs min_sum vs configuration", "mean gap", lower=True)

    a = fig.add_subplot(gs[0, 3])
    tm, tsd = cfg_agg("tie"); wm, wsd = cfg_agg("beat")
    a.bar(xs - 0.2, tm, width=0.4, yerr=tsd, color="C7", capsize=2,
          error_kw=dict(lw=0.8, alpha=0.7), label="tie rate (gap=0)")
    a.bar(xs + 0.2, wm, width=0.4, yerr=wsd, color="C2", capsize=2,
          error_kw=dict(lw=0.8, alpha=0.7), label="win rate (gap<0)")
    a.set_xticks(xs); a.set_xticklabels(labels, fontsize=6.5, rotation=90)
    a.set_ylabel("fraction of resolved"); a.set_title("Tie / win rate vs configuration", fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.3, axis="y")

    # ---- row 1: resolution rate & gap distribution vs determinant (one bar per config) ----
    W = 0.8 / max(1, len(configs))

    def grouped(a, nbin, per_model_vec):
        for j, k in enumerate(configs):
            vecs = [v for v in (per_model_vec(m) for m in by_cfg[k]) if v is not None]
            if not vecs:
                continue
            arr = np.vstack(vecs)
            std = arr.std(axis=0) if arr.shape[0] > 1 else np.zeros(nbin)  # std over seeds
            a.bar(np.arange(nbin) - 0.4 + W * (j + 0.5), arr.mean(axis=0), width=W * 0.95,
                  color=colors[j], yerr=np.nan_to_num(std), capsize=2,
                  error_kw=dict(lw=0.7, alpha=0.6))

    a = fig.add_subplot(gs[1, 0:2])
    all_d = [d for m in models for (_, d, _, _, _) in m["records"]]
    if all_d:
        lo, hi = min(all_d), max(all_d)
        nb = max(1, min(6, hi - lo + 1))
        edges = np.linspace(lo, hi + 1, nb + 1)

        def res_by_det(m):
            out = []
            for i in range(nb):
                sel = [r for (_, d, _, _, r) in m["records"] if edges[i] <= d < edges[i + 1]]
                out.append(statistics.mean(sel) if sel else np.nan)
            return np.array(out, float)

        grouped(a, nb, res_by_det)
        a.set_xticks(range(nb))
        a.set_xticklabels([f"{int(edges[i])}-{int(edges[i+1]) - 1}" for i in range(nb)], fontsize=8)
        a.set_ylim(0, 1.08)
    a.set_xlabel("determinant d (binned)"); a.set_ylabel("fraction resolved")
    a.set_title("Resolution rate vs determinant  (bars = configs, side by side; higher better)", fontsize=10)

    a = fig.add_subplot(gs[1, 2:4])
    gbins = [-2, 0, 2, 5, 10, 20, 40, 80]
    glabels = ["<0", "0-1", "2-4", "5-9", "10-19", "20-39", "40+"]

    def gap_dist(m):
        gaps = [s - ms for (_, _, ms, s, r) in m["records"] if r]
        if not gaps:
            return None
        return np.array([sum(gbins[k] <= g < gbins[k + 1] for g in gaps)
                         for k in range(len(gbins) - 1)], float) / len(gaps)

    grouped(a, len(glabels), gap_dist)
    a.set_xticks(range(len(glabels))); a.set_xticklabels(glabels, fontsize=8)
    a.set_xlabel("gap (agent − min_sum), binned  (≤0 beats baseline)")
    a.set_ylabel("fraction of resolved episodes")
    a.set_title("Gap distribution vs determinant  (bars = configs, side by side; left/lower better)", fontsize=10)

    # ---- row 2: heatmaps over the two axes ----
    va = sorted({group_value(m, ax0) for m in models}, key=lambda v: _grid_sort_key((v,)))
    vb = sorted({group_value(m, ax1) for m in models}, key=lambda v: _grid_sort_key((v,)))

    def stat_matrix(field):
        cell = defaultdict(list)
        for m in models:
            cell[(group_value(m, ax0), group_value(m, ax1))].append(summarize(m)[field])
        M = np.full((len(va), len(vb)), np.nan)
        for i, a_ in enumerate(va):
            for j, b_ in enumerate(vb):
                vals = [v for v in cell[(a_, b_)] if v == v]
                if vals:
                    M[i, j] = statistics.mean(vals)
        return M

    hpanels = [("score", "score", True), ("res_rate", "resolution rate", False),
               ("mean_gap", "mean gap", True), ("tie", "tie rate", False)]
    for col, (field, title, lower) in enumerate(hpanels):
        a = fig.add_subplot(gs[2, col])
        M = stat_matrix(field)
        vmin, vmax = heatmap_range(field)
        im = a.imshow(np.ma.masked_invalid(M), cmap=heatmap_cmap(lower), aspect="auto",
                      vmin=vmin, vmax=vmax)
        im.cmap.set_bad("0.85")
        a.set_xticks(range(len(vb))); a.set_xticklabels([str(v) for v in vb], fontsize=8)
        a.set_yticks(range(len(va))); a.set_yticklabels([str(v) for v in va], fontsize=8)
        a.set_xlabel(ax1); a.set_ylabel(ax0)
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

    handles = [Line2D([], [], color=colors[j], lw=6, label=labels[j].replace("\n", ", "))
               for j in range(len(configs))]
    fig.legend(handles=handles, title="configuration", loc="lower center",
               ncol=min(len(configs), 6), fontsize=7, framealpha=0.9, bbox_to_anchor=(0.5, 0.005))

    # manual spacing (tight_layout can't handle the colorbar axes); bbox_inches
    # keeps the bottom configuration legend from being clipped on save
    fig.subplots_adjust(left=0.05, right=0.97, top=0.93, bottom=0.10, hspace=0.8, wspace=0.35)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


# ===========================================================================================
#  SINGLE-MODEL DETAIL MODE  (read_diagnostics-style report for ONE model on the depot)
# ===========================================================================================

def _resolve_model_path(p):
    """Accept a trial dir, a run folder, or a model.pt path; return the model.pt."""
    return os.path.join(p, "model.pt") if os.path.isdir(p) else p


def find_model_record(records_path, model_path):
    """Return the stored per-cone record blob for model_path from a records.json
    (either a sweep-level one or a per-model detail one), or None."""
    if not os.path.exists(records_path):
        return None
    try:
        blob = json.load(open(records_path))
    except (OSError, ValueError):
        return None
    tgt = os.path.realpath(model_path)
    for m in blob.get("models", []):
        if os.path.realpath(m["path"]) == tgt:
            return blob["meta"], m
    return None


def make_detail_png(model, out_png, meta):
    """read_diagnostics-style multi-panel report for a single model's depot rollout.
    Records are per-cone tuples (n, d, min_sum_steps, agent_steps, resolved)."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def jit(vals, scale=0.15):
        return np.asarray(vals, float) + np.random.uniform(-scale, scale, size=len(vals))

    recs = [tuple(r) for r in model["records"]]
    n = len(recs)
    resolved = [(nn, dd, ms, ss) for (nn, dd, ms, ss, r) in recs if r]
    n_res = len(resolved)
    gaps = [ss - ms for (_, _, ms, ss) in resolved]
    res_rate = n_res / n if n else 0.0
    win = sum(g < 0 for g in gaps); tie = sum(g == 0 for g in gaps); loss = sum(g > 0 for g in gaps)
    mean_gap = statistics.mean(gaps) if gaps else float("nan")

    # dynamic grid: 12 fixed panels + one gap-histogram panel per dimension (8a, 8b, ...),
    # dealt row-major into however many rows of 3 that needs; leftover slots are blanked
    dim_list = sorted({nn for (nn, _, _, _, _) in recs})
    n_panels = 12 + len(dim_list)
    n_rows = -(-n_panels // 3)
    fig, ax = plt.subplots(n_rows, 3, figsize=(19, 5.4 * n_rows))
    panels = iter(ax.ravel())
    for spare in ax.ravel()[n_panels:]:
        spare.axis("off")
    name = os.path.basename(os.path.dirname(model["path"]))
    beat_str = f"{win}/{n_res} ({win / n_res:.0%})" if n_res else "n/a"
    tie_str = f"{tie}/{n_res} ({tie / n_res:.0%})" if n_res else "n/a"
    fig.suptitle(
        f"Single-model {meta.get('dataset', 'validation')} detail — {name}  "
        f"(emb={model.get('embedding_size', '?')}, layer={model.get('layer_type', '?')}, "
        f"seed={model.get('seed', '?')})\n"
        f"greedy policy (no MCTS)  |  {_nrange(meta)}, {meta['det_min']}≤d≤{meta['det_max']}  |  "
        f"{n} cones  |  resolved {res_rate:.0%}  |  beats min_sum {beat_str}  |  "
        f"ties {tie_str}  |  mean gap {mean_gap:.2f}",
        fontsize=13, fontweight="bold")

    def annotate_counts(a, bars, counts):
        for b, c in zip(bars, counts):
            a.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, str(c),
                   ha="center", va="bottom", fontsize=7)

    # 1. resolution rate vs dimension n
    a = next(panels)
    by_dim = defaultdict(list)
    for (nn, _, _, _, r) in recs:
        by_dim[nn].append(r)
    dims = sorted(by_dim)
    rates = [statistics.mean(by_dim[k]) for k in dims]
    stds = [statistics.pstdev(by_dim[k]) for k in dims]
    bars = a.bar(dims, rates, width=0.7, color="C2", alpha=0.85,
                 yerr=stds, capsize=4, ecolor="k")
    annotate_counts(a, bars, [len(by_dim[k]) for k in dims])
    a.set_xticks(dims); a.set_ylim(-0.06, 1.08)
    a.set_xlabel("dimension n"); a.set_ylabel("fraction resolved")
    a.set_title("1. Resolution rate vs dimension")

    # 2. resolution rate vs determinant d (integer-width buckets sized to the observed d range)
    a = next(panels)
    dvals = [dd for (_, dd, _, _, _) in recs]
    d_lo, d_hi = (min(dvals), max(dvals)) if dvals else (meta["det_min"], meta["det_max"])
    bucket = max(1, -(-(d_hi - d_lo + 1) // 7))
    edges = np.arange(d_lo, d_hi + bucket + 1, bucket)
    centers, rrates, rstds, rcounts = [], [], [], []
    for i in range(len(edges) - 1):
        sel = [r for (_, dd, _, _, r) in recs if edges[i] <= dd < edges[i + 1]]
        if sel:
            centers.append((edges[i] + edges[i + 1] - 1) / 2)
            rrates.append(statistics.mean(sel)); rstds.append(statistics.pstdev(sel))
            rcounts.append(len(sel))
    bars = a.bar(centers, rrates, width=bucket * 0.85, color="C0", alpha=0.85,
                 yerr=rstds, capsize=4, ecolor="k")
    annotate_counts(a, bars, rcounts)
    a.set_ylim(-0.06, 1.08)
    a.set_xlabel("determinant d"); a.set_ylabel("fraction resolved")
    a.set_title("2. Resolution rate vs determinant")

    # 3. gap distribution (resolved only)
    a = next(panels)
    if gaps:
        lo, hi = min(gaps), max(gaps)
        a.hist(gaps, bins=range(lo, hi + 2), align="left", rwidth=0.85, color="C4")
        a.axvline(0.0, color="gray", lw=1, ls="--")
        ## rug strip of the individual gaps along the bottom
        a.scatter(jit(gaps), np.random.uniform(0.015, 0.06, len(gaps)) * a.get_ylim()[1],
                  s=8, alpha=0.25, color="k", zorder=3)
        n_opt = sum(1 for g in gaps if g <= 0)
        a.set_title(f"3. Gap distribution  (≤min_sum: {n_opt}/{len(gaps)})")
    else:
        a.text(0.5, 0.5, "no resolved cones", ha="center", va="center")
        a.set_title("3. Gap distribution")
    a.set_xlabel("gap (agent − min_sum)"); a.set_ylabel("cones")

    # 4. parity: min_sum steps vs agent steps (resolved only)
    a = next(panels)
    b = [ms for (_, _, ms, _) in resolved]; ag = [ss for (_, _, _, ss) in resolved]
    if b:
        lo, hi = min(min(b), min(ag)), max(max(b), max(ag))
        a.plot([lo, hi], [lo, hi], color="gray", ls="--", lw=1, label="y = x (tie)")
        a.scatter(jit(b), jit(ag), s=14, alpha=0.4, color="C0")
        a.legend(fontsize=8)
    else:
        a.text(0.5, 0.5, "no resolved cones", ha="center", va="center")
    a.set_xlabel("min_sum steps"); a.set_ylabel("agent steps")
    a.set_title("4. Agent vs min_sum (parity)\n(below line = agent used fewer)")

    # 5. gap vs determinant (resolved only)
    a = next(panels)
    a.axhline(0.0, color="gray", lw=1, ls="--")
    dd = [d for (_, d, _, _) in resolved]
    if dd:
        a.scatter(jit(dd), gaps, s=14, alpha=0.4, color="C4")
        by_d = defaultdict(list)
        for d, g in zip(dd, gaps):
            by_d[d].append(g)
        xs = sorted(by_d)
        a.plot(xs, [statistics.mean(by_d[x]) for x in xs], color="C1", lw=1.5,
               marker="o", ms=3, label="mean per d")
        a.legend(fontsize=8)
    a.set_xlabel("determinant d"); a.set_ylabel("gap (resolved only)")
    a.set_title("5. Gap vs determinant")

    # 6. win / tie / loss vs min_sum (resolved only)
    a = next(panels)
    if gaps:
        fr = [win / len(gaps), tie / len(gaps), loss / len(gaps)]
        bars = a.bar(["beats", "ties", "worse"], fr, color=["C2", "C7", "C3"], alpha=0.85)
        for bar_, c in zip(bars, [win, tie, loss]):
            a.text(bar_.get_x() + bar_.get_width() / 2, bar_.get_height() + 0.01, str(c),
                   ha="center", va="bottom", fontsize=8)
        a.set_ylim(0, 1.08)
    else:
        a.text(0.5, 0.5, "no resolved cones", ha="center", va="center")
    a.set_ylabel("fraction of resolved"); a.set_title("6. Win / tie / loss vs min_sum")

    # ---- per-dimension breakdowns (7-12): same metrics, split by n to compare across dimension ----
    # fixed color per dimension (n=2 -> slot 0, n=3 -> slot 1, ...) so colors are stable across reports
    dcolor = {k: CATEGORICAL[(k - 2) % len(CATEGORICAL)] for k in dim_list}
    res_by_n = {k: [(d2, ms, ss) for (nn, d2, ms, ss, r) in recs if r and nn == k] for k in dim_list}
    gaps_by_n = {k: [ss - ms for (_, ms, ss) in res_by_n[k]] for k in dim_list}

    # 7. resolution rate vs determinant, one line per dimension (same d buckets as panel 2)
    a = next(panels)
    for k in dim_list:
        cs, rs = [], []
        for i in range(len(edges) - 1):
            sel = [r for (nn, d2, _, _, r) in recs if nn == k and edges[i] <= d2 < edges[i + 1]]
            if sel:
                cs.append((edges[i] + edges[i + 1] - 1) / 2)
                rs.append(statistics.mean(sel))
        a.plot(cs, rs, color=dcolor[k], lw=2, marker="o", ms=4, label=f"n={k}")
    a.set_ylim(-0.06, 1.08)
    a.legend(fontsize=8)
    a.set_xlabel("determinant d"); a.set_ylabel("fraction resolved")
    a.set_title("7. Resolution rate vs determinant, by dimension")

    # 8a-8x. gap distribution, one panel per dimension (fraction of that dim's
    # resolved cones; bins shared across the panels so the x-axes compare directly)
    all_g = [g for k in dim_list for g in gaps_by_n[k]]
    bins = range(min(all_g), max(all_g) + 2) if all_g else None
    for j, k in enumerate(dim_list):
        a = next(panels)
        g = gaps_by_n[k]
        if g:
            w = np.full(len(g), 1.0 / len(g))
            a.hist(g, bins=bins, align="left", rwidth=0.85, weights=w, color=dcolor[k])
            a.axvline(0.0, color="gray", lw=1, ls="--")
        else:
            a.text(0.5, 0.5, "no resolved cones", ha="center", va="center")
        a.set_xlabel("gap (agent − min_sum)"); a.set_ylabel("fraction of dim's resolved")
        a.set_title(f"8{chr(ord('a') + j)}. Gap distribution, n={k}  ({len(g)} resolved)")

    # 9. parity by dimension (resolved only)
    a = next(panels)
    if resolved:
        allb = [ms for (_, _, ms, _) in resolved] + [ss for (_, _, _, ss) in resolved]
        lo, hi = min(allb), max(allb)
        a.plot([lo, hi], [lo, hi], color="gray", ls="--", lw=1, label="y = x (tie)")
        for k in dim_list:
            if res_by_n[k]:
                a.scatter(jit([ms for (_, ms, _) in res_by_n[k]]),
                          jit([ss for (_, _, ss) in res_by_n[k]]),
                          s=12, alpha=0.35, color=dcolor[k], label=f"n={k}")
        a.legend(fontsize=8)
    else:
        a.text(0.5, 0.5, "no resolved cones", ha="center", va="center")
    a.set_xlabel("min_sum steps"); a.set_ylabel("agent steps")
    a.set_title("9. Agent vs min_sum parity, by dimension\n(below line = agent used fewer)")

    # 10. mean gap vs determinant, one line per dimension (same d buckets; resolved only)
    a = next(panels)
    a.axhline(0.0, color="gray", lw=1, ls="--")
    for k in dim_list:
        cs, mg = [], []
        for i in range(len(edges) - 1):
            sel = [ss - ms for (d2, ms, ss) in res_by_n[k] if edges[i] <= d2 < edges[i + 1]]
            if sel:
                cs.append((edges[i] + edges[i + 1] - 1) / 2)
                mg.append(statistics.mean(sel))
        a.plot(cs, mg, color=dcolor[k], lw=2, marker="o", ms=4, label=f"n={k}")
    a.legend(fontsize=8)
    a.set_xlabel("determinant d"); a.set_ylabel("mean gap (resolved only)")
    a.set_title("10. Gap vs determinant, by dimension")

    # 11. win / tie / loss vs min_sum, grouped by dimension (fractions of each dim's resolved)
    a = next(panels)
    cats = ["beats", "ties", "worse"]
    xs0 = np.arange(len(cats))
    bw = 0.8 / max(len(dim_list), 1)
    for j, k in enumerate(dim_list):
        g = gaps_by_n[k]
        if not g:
            continue
        fr = [sum(x < 0 for x in g) / len(g), sum(x == 0 for x in g) / len(g),
              sum(x > 0 for x in g) / len(g)]
        bars = a.bar(xs0 + j * bw, fr, width=bw * 0.9, color=dcolor[k], label=f"n={k}")
        annotate_counts(a, bars, [sum(x < 0 for x in g), sum(x == 0 for x in g), sum(x > 0 for x in g)])
    a.set_xticks(xs0 + bw * (len(dim_list) - 1) / 2)
    a.set_xticklabels(cats)
    a.set_ylim(0, 1.14)
    a.legend(fontsize=8)
    a.set_ylabel("fraction of dim's resolved")
    a.set_title("11. Win / tie / loss vs min_sum, by dimension")

    # 12. per-dimension summary table
    a = next(panels)
    a.axis("off")
    if dim_list:
        cols = ["cones", "resolved", "mean gap", "tie", "beat"]
        cell = []
        for k in dim_list:
            rows_k = [t for t in recs if t[0] == k]
            g = gaps_by_n[k]
            nres = len(res_by_n[k])
            cell.append([f"{len(rows_k)}",
                         f"{nres / len(rows_k):.1%}" if rows_k else "n/a",
                         f"{statistics.mean(g):.2f}" if g else "n/a",
                         f"{sum(x == 0 for x in g) / len(g):.1%}" if g else "n/a",
                         f"{sum(x < 0 for x in g) / len(g):.1%}" if g else "n/a"])
        tbl = a.table(cellText=cell, rowLabels=[f"n={k}" for k in dim_list],
                      colLabels=cols, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.6)
        for j, k in enumerate(dim_list):
            tbl[j + 1, -1].set_facecolor(dcolor[k]); tbl[j + 1, -1].set_alpha(0.25)
    a.set_title("12. Summary by dimension")

    # 13. cumulative resolution vs step budget: fraction of cones resolved in <= k steps
    a = next(panels)
    max_s = int(meta.get("max_steps") or max((ss for (_, _, _, ss, _) in recs), default=1))
    ks = np.arange(1, max_s + 1)
    if recs:
        agg = np.sort([ss for (_, _, _, ss, r) in recs if r])
        a.plot(ks, np.searchsorted(agg, ks, side="right") / len(recs),
               color="0.15", lw=2.5, label="all")
        for k in dim_list:
            rows_k = [t for t in recs if t[0] == k]
            st = np.sort([ss for (_, _, _, ss, r) in rows_k if r])
            a.plot(ks, np.searchsorted(st, ks, side="right") / len(rows_k),
                   color=dcolor[k], lw=1.8, label=f"n={k}")
        a.legend(fontsize=8)
    else:
        a.text(0.5, 0.5, "no records", ha="center", va="center")
    a.set_xticks(np.arange(0, max_s + 1, max(1, max_s // 10)))
    a.grid(axis="x", alpha=0.3)
    a.set_ylim(0, 1.05)
    a.set_xlabel("step budget k"); a.set_ylabel("fraction resolved in ≤ k steps")
    a.set_title("13. Cumulative resolution vs step budget")

    # footer: this model's full hyperparameters
    a = model.get("args", {})
    bottom = 0.0
    if a:
        items = [f"{k}={a[k]}" for k in sorted(a) if k not in ("save", "diag_dir", "resume")]
        per_line = 6
        lines = ["   ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]
        footer = "hyperparameters:\n" + "\n".join(lines)
        bottom = 0.015 + 0.021 * (len(lines) + 1)
        fig.text(0.5, 0.01, footer, ha="center", va="bottom",
                 fontsize=7.5, family="monospace", color="0.2")

    fig.tight_layout(rect=[0, bottom, 1, 0.95])
    fig.savefig(out_png, dpi=130)
    print(f"wrote {out_png}")


def run_detail(args):
    """Single-model mode: render a detailed depot report for args.model, reusing
    stored per-cone records when available, else doing a fresh rollout.  Output
    lands in the model's own folder (a sweep trial or a results/<timestamp> run)."""
    model_path = _resolve_model_path(args.model)
    if not os.path.exists(model_path):
        print(f"model not found: {model_path}"); return
    out_png = args.out or os.path.join(os.path.dirname(model_path), f"{args.dataset}_detail.png")
    detail_records = os.path.splitext(out_png)[0] + "_records.json"
    sweep_records = os.path.join(args.sweep_dir, f"{args.dataset}_eval_records.json")

    # prefer a per-model detail records file, then the sweep-level one.
    # stored records are only reusable if they were rolled out under the same
    # dataset + cone filter; --replot skips the check and takes whatever is stored.
    want = dict(dataset=args.dataset, min_dimension=args.min_dimension, max_dimension=args.max_dimension,
                det_min=args.det_min, det_max=args.det_max, max_steps=args.max_steps)
    found = None
    for rp in (detail_records, sweep_records):
        hit = find_model_record(rp, model_path)
        if hit:
            meta_stored = hit[0]
            if args.replot or all(meta_stored.get(k) == v for k, v in want.items()):
                found = hit
                print(f"reusing stored records from {rp}")
                break
            print(f"stored records in {rp} used a different filter "
                  f"({ {k: meta_stored.get(k) for k in want} } vs {want}); re-rolling")

    if found:
        meta, model = found
    else:
        if args.replot:
            print("no stored records for this model; run without --replot to roll out first.")
            return
        cones = load_depot(args.dataset, args.min_dimension, args.max_dimension, args.det_min, args.det_max)
        if not cones:
            print(f"no {args.dataset} cones match {args.min_dimension}≤n≤{args.max_dimension}, "
                  f"{args.det_min}≤d≤{args.det_max}; widen the filter."); return
        print(f"rolling out {os.path.basename(os.path.dirname(model_path))} on {len(cones)} "
              f"{args.dataset} cones ({args.min_dimension}≤n≤{args.max_dimension}, {args.det_min}≤d≤{args.det_max}) ...")
        model = eval_one_model(model_path, cones, args.max_steps, progress=True)
        if "error" in model:
            print(f"eval failed: {model['error']}"); return
        meta = dict(dataset=args.dataset, min_dimension=args.min_dimension, max_dimension=args.max_dimension,
                    det_min=args.det_min, det_max=args.det_max,
                    n_cones=len(cones), max_steps=args.max_steps)
        with open(detail_records, "w") as f:
            json.dump({"meta": meta, "models": [model]}, f)
        print(f"wrote {detail_records}")

    make_detail_png(model, out_png, meta)


def main():
    ap = argparse.ArgumentParser(description="Evaluate trained models on the validation depot.")
    ap.add_argument("--sweep-dir", default=os.path.join(HERE, "sweep", "sweep_0"),
                    help="sweep folder holding the t*/model.pt checkpoints to evaluate")
    ap.add_argument("--dataset", choices=sorted(DEPOTS), default="validation",
                    help="which depot to evaluate on (default: validation)")
    ap.add_argument("--min-dimension", type=int, default=2,
                    help="keep only depot cones with dimension n >= this")
    ap.add_argument("--max-dimension", type=int, default=4,
                    help="keep only depot cones with dimension n <= this")
    ap.add_argument("--det-min", type=int, default=2,
                    help="keep only depot cones with determinant d >= this")
    ap.add_argument("--det-max", type=int, default=20,
                    help="keep only depot cones with determinant d <= this")
    ap.add_argument("--max-steps", type=int, default=40,
                    help="max greedy subdivisions per cone before it counts as unresolved")
    ap.add_argument("--group-by", default=None, metavar="HPARAM",
                    help="any train.py hyperparameter to compare models across, e.g. lr, "
                         "c_puct, layer_type (default: auto-detect the sweep's varied axis "
                         "from sweep_meta.json, else the arg that varies across the models)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel evaluation processes for sweep mode; single-model "
                         "mode (--model) always rolls out serially on one thread")
    ap.add_argument("--out", default=None,
                    help="output PNG path (default: <sweep-dir>/validation_eval.png)")
    ap.add_argument("--replot", action="store_true",
                    help="rebuild the PNG from saved records (no rollouts)")
    ap.add_argument("--model", default=None,
                    help="single-model detail mode: render a read_diagnostics-style depot "
                         "report for ONE model (sweep trial, results/ run folder, or model.pt). "
                         "Reuses stored records if present, else does a fresh rollout")
    args = ap.parse_args()

    if args.model:
        run_detail(args)
        return

    out_png = args.out or os.path.join(args.sweep_dir, f"{args.dataset}_eval.png")
    records_path = os.path.splitext(out_png)[0] + "_records.json"

    if args.replot:
        blob = json.load(open(records_path))
        # keep every stored field (incl. args) so grouping can use any hyperparameter
        models = [{**m, "records": [tuple(r) for r in m["records"]]} for m in blob["models"]]
        meta = blob["meta"]
        if meta.get("mode") == "grid" and len(meta.get("axes", [])) == 2:
            make_grid_png(models, out_png, meta)       # grid layout ignores --group-by
        else:
            if args.group_by:                          # let --replot re-group on a new axis
                meta = {**meta, "group_key": args.group_by,
                        "constants": constant_hparams(models, args.group_by)}
            make_png(models, out_png, meta)
        print(f"replotted from {records_path}")
        return

    group_key = args.group_by or detect_group_key(args.sweep_dir)
    cones = load_depot(args.dataset, args.min_dimension, args.max_dimension, args.det_min, args.det_max)
    if not cones:
        print(f"no {args.dataset} cones match {args.min_dimension}<=n<={args.max_dimension}, "
              f"{args.det_min}<=d<={args.det_max}; widen the filter."); return
    model_paths = sorted(glob.glob(os.path.join(args.sweep_dir, "t*", "model.pt")))
    n_shards = n_shards_for(len(cones), args.workers, len(model_paths))
    shards = shard_cones(cones, n_shards)
    print(f"{len(model_paths)} models  x  {len(cones)} {args.dataset} cones  "
          f"({args.min_dimension}<=n<={args.max_dimension}, {args.det_min}<=d<={args.det_max})  "
          f"max_steps={args.max_steps}  workers={args.workers}  shards/model={n_shards}")

    # each model's depot is split into n_shards cone shards and every
    # (model, shard) pair is its own pool task, so one slow model -- or a sweep
    # of a single model -- still spreads across all workers
    models, errors = [], []
    done_shards = defaultdict(list)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(eval_one_model, p,
                          [cones[i] for i in sh], args.max_steps): (p, k)
                for p in model_paths for k, sh in enumerate(shards)}
        bar = tqdm(as_completed(futs), total=len(futs), desc="eval", unit="shard")
        for fut in bar:
            p, k = futs[fut]
            done_shards[p].append((k, fut.result()))
            if len(done_shards[p]) < n_shards:
                continue
            r = merge_shard_results(done_shards.pop(p), shards, cones)
            name = os.path.basename(os.path.dirname(p))
            if "error" in r:
                errors.append(r); bar.write(f"  SKIP {name}: {r['error']}")
            else:
                s = summarize(r)
                models.append(r)
                bar.write(f"  {name} emb={r['embedding_size']} res={s['res_rate']:.3f} "
                          f"gap={s['mean_gap']:.2f} score={s['score']:.2f}")

    if not models:
        print("no models evaluated; aborting."); return

    common = dict(dataset=args.dataset, min_dimension=args.min_dimension, max_dimension=args.max_dimension,
                  det_min=args.det_min, det_max=args.det_max, n_cones=len(cones),
                  n_models=len(models), max_steps=args.max_steps)
    grid_axes = detect_grid(args.sweep_dir)
    if grid_axes:
        # 2-axis grid: render the grid report and label CSV rows by both axes
        print(f"grid evaluation over axes {grid_axes}")
        meta = {**common, "mode": "grid", "axes": grid_axes}
        extra = grid_axes
        render = lambda: make_grid_png(models, out_png, meta)
    else:
        # resolve the grouping axis now that the models (and their args) are loaded
        if group_key is None:
            group_key = infer_group_key(models) or "embedding_size"
        print(f"grouping comparison by: {group_key}")
        meta = {**common, "group_key": group_key,
                "constants": constant_hparams(models, group_key)}
        extra = [] if group_key in ("embedding_size", "layer_type", "seed") else [group_key]
        render = lambda: make_png(models, out_png, meta)

    # persist raw per-cone records so future restyles need no rollouts (--replot)
    with open(records_path, "w") as f:
        json.dump({"meta": meta, "models": models}, f)
    print(f"wrote {records_path}")

    render()

    # also dump raw per-model metrics next to the png, labelled by the swept axis
    # (or both grid axes) so the CSV pins down every configuration.
    with open(os.path.splitext(out_png)[0] + ".csv", "w") as f:
        f.write(",".join(["trial_dir", *extra, "embedding_size", "layer_type",
                          "seed", "res_rate", "mean_gap", "tie", "beat", "score"]) + "\n")
        def rowkey(m):
            return tuple(str(group_value(m, k)) for k in extra) + \
                   (m["embedding_size"], m["layer_type"], m["seed"])
        for m in sorted(models, key=rowkey):
            s = summarize(m)
            cells = [os.path.basename(os.path.dirname(m["path"]))]
            cells += [str(group_value(m, k)) for k in extra]
            cells += [str(m["embedding_size"]), str(m["layer_type"]), str(m["seed"]),
                      f"{s['res_rate']:.4f}", f"{s['mean_gap']:.4f}",
                      f"{s['tie']:.4f}", f"{s['beat']:.4f}", f"{s['score']:.4f}"]
            f.write(",".join(cells) + "\n")
    print(f"wrote {os.path.splitext(out_png)[0] + '.csv'}")


if __name__ == "__main__":
    main()
