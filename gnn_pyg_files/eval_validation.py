#!/usr/bin/env python3
"""Evaluate every trained model in a sweep folder on the validation depot.

Each model plays a deterministic GREEDY (argmax-policy, no MCTS) rollout on every
validation cone from the depot that passes the training-range filter:
    det_min <= d <= det_max   AND   min_dimension <= n <= max_dimension
(here fixed by the CLI defaults to n<=4, d<=30 to match this sweep).

Per cone we record: agent subdivisions, whether it resolved, and gap vs the
precomputed min_sum baseline stored in the depot.  Results are aggregated over the
swept axis (--group-by; mean over seeds) and written as a single PNG in the sweep
folder alongside a per-model CSV and a records.json (for fast --replot).

Parameters:
  --sweep-dir DIR    sweep folder holding the t*/model.pt checkpoints to evaluate
                     (default: sweep/sweep_0)
  --n-max N          keep only depot cones with dimension n <= N        (default 4)
  --d-min D          keep only depot cones with determinant d >= D      (default 2)
  --d-max D          keep only depot cones with determinant d <= D      (default 30)
  --max-steps K      max greedy subdivisions per cone before it counts
                     as unresolved                                      (default 80)
  --group-by AXIS    axis to compare models across: embedding_size | layer_type
                     (default: auto-detect the sweep's varied axis from
                     sweep_meta.json, else embedding_size)
  --workers W        parallel model-evaluation processes                (default 12)
  --out PATH         output PNG path (records.json/CSV derived from it)
                     (default: <sweep-dir>/validation_eval.png)
  --replot           rebuild the PNG from a saved records.json, no rollouts
  --model PATH       SINGLE-MODEL DETAIL MODE (trial dir or model.pt): instead of
                     the sweep-comparison grid, render a read_diagnostics-style
                     per-cone report for that one model on the depot — resolution
                     rate vs n and vs d, gap histogram, agent-vs-min_sum parity,
                     gap vs d, win/tie/loss, plus a full hyperparameter strip.
                     Reuses the model's stored per-cone records if a records.json
                     is present (its own or the sweep's), else does a fresh
                     rollout and saves one; --replot forces the stored path.
                     Output: <model-dir>/validation_detail.png
"""
import argparse, glob, json, os, statistics, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEPOT = os.path.join(HERE, "..", "datasets", "validation_set-2.json")


def load_depot(n_max, d_min, d_max):
    depot = json.load(open(DEPOT))
    return [c for c in depot if c["n"] <= n_max and d_min <= c["d"] <= d_max]


# per-model fields we can group the comparison plots by
GROUPABLE = ("embedding_size", "layer_type")

def detect_group_key(sweep_dir):
    """Pick the axis to compare across: the sweep's own varied axis if we can plot
    it (from sweep_meta.json), else embedding_size."""
    try:
        axes = json.load(open(os.path.join(sweep_dir, "sweep_meta.json"))).get("axes", [])
        if len(axes) == 1 and axes[0] in GROUPABLE:
            return axes[0]
    except (OSError, ValueError):
        pass
    return "embedding_size"


def eval_one_model(model_path, cones, max_steps):
    """Runs in a worker process: load model, greedy-rollout every cone."""
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
    dim = a["max_dimension"]

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
    for c in cones:
        s, r = rollout(c)
        recs.append((c["n"], c["d"], c["min_sum_steps"], s, int(r)))
    return {"path": model_path, "embedding_size": a["embedding_size"],
            "layer_type": a.get("layer_type", "GAT"),
            "seed": a.get("seed", -1), "args": dict(a), "records": recs}


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


# args fields that vary by design or are per-run identity, not "constant hyperparameters"
_HPARAM_SKIP = {"save", "diag_dir", "resume", "device", "seed"}

def constant_hparams(models, group_key):
    """Hyperparameters held fixed across every evaluated model (same value everywhere),
    excluding per-run identity fields and the swept axis itself."""
    args = [m.get("args", {}) for m in models if m.get("args")]
    if not args:
        return {}
    skip = _HPARAM_SKIP | {group_key}
    keys = set.intersection(*(set(a) for a in args))
    consts = {}
    for k in sorted(keys):
        if k in skip:
            continue
        vals = {a[k] for a in args}
        if len(vals) == 1:
            consts[k] = next(iter(vals))
    return consts


def make_png(models, out_png, meta):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    gkey = meta.get("group_key", "embedding_size")     # which swept axis to compare across
    by_grp = defaultdict(list)
    for m in models:
        by_grp[m.get(gkey, m.get("embedding_size"))].append(m)
    # numeric axes (embedding_size) sort/scale numerically; categorical (layer_type) don't
    groups = sorted(by_grp, key=lambda g: (not isinstance(g, (int, float)), g))
    numeric = all(isinstance(g, (int, float)) for g in groups)
    xpos = {g: (g if numeric else i) for i, g in enumerate(groups)}
    palette = plt.cm.viridis(np.linspace(0, 1, len(groups)))
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

    fig, ax = plt.subplots(2, 3, figsize=(19, 11))
    fig.suptitle(f"Validation-depot eval — greedy policy (no MCTS)  |  "
                 f"n≤{meta['n_max']}, {meta['d_min']}≤d≤{meta['d_max']}, "
                 f"max_steps={meta.get('max_steps', '?')}  "
                 f"({meta['n_cones']} cones)  |  mean over seeds  |  {meta['n_models']} models  "
                 f"|  color = {gkey}",
                 fontsize=15, fontweight="bold")

    def scalar_panel(a, metric, title, ylabel, lower=True):
        d = agg(metric)
        gs = [g for g in groups if g in d]
        a.plot([xpos[g] for g in gs], [d[g][0] for g in gs], color="0.75", lw=1, zorder=1)  # trend
        for g in gs:
            m, s, k = d[g]
            a.errorbar([xpos[g]], [m], yerr=[s], marker="o", ms=10, capsize=4,
                       color=color[g], ecolor=color[g], zorder=3)
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
            a.bar(np.arange(nbin) + j * W, mean, width=W, yerr=std, color=color[g],
                  label=str(g), capsize=2, error_kw=dict(lw=0.8, alpha=0.7))

    # resolution vs determinant (binned) grouped; bar = mean over models, err = std over models
    a = ax[1, 1]
    bins = [(2, 6), (7, 11), (12, 15), (16, 20), (21, 25), (26, 30)]
    labels = [f"{lo}-{hi}" for lo, hi in bins]

    def res_by_det(m):
        out = []
        for lo, hi in bins:
            vals = [r for (_, d, _, _, r) in m["records"] if lo <= d <= hi]
            out.append(statistics.mean(vals) if vals else 0.0)
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


# ===========================================================================================
#  SINGLE-MODEL DETAIL MODE  (read_diagnostics-style report for ONE model on the depot)
# ===========================================================================================

def _resolve_model_path(p):
    """Accept either a trial dir or a model.pt path; return the model.pt path."""
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

    fig, ax = plt.subplots(2, 3, figsize=(19, 11))
    name = os.path.basename(os.path.dirname(model["path"]))
    beat_str = f"{win}/{n_res} ({win / n_res:.0%})" if n_res else "n/a"
    tie_str = f"{tie}/{n_res} ({tie / n_res:.0%})" if n_res else "n/a"
    fig.suptitle(
        f"Single-model validation detail — {name}  "
        f"(emb={model.get('embedding_size', '?')}, layer={model.get('layer_type', '?')}, "
        f"seed={model.get('seed', '?')})\n"
        f"greedy policy (no MCTS)  |  n≤{meta['n_max']}, {meta['d_min']}≤d≤{meta['d_max']}  |  "
        f"{n} cones  |  resolved {res_rate:.0%}  |  beats min_sum {beat_str}  |  "
        f"ties {tie_str}  |  mean gap {mean_gap:.2f}",
        fontsize=13, fontweight="bold")

    def annotate_counts(a, bars, counts):
        for b, c in zip(bars, counts):
            a.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, str(c),
                   ha="center", va="bottom", fontsize=7)

    # 1. resolution rate vs dimension n
    a = ax[0, 0]
    by_dim = defaultdict(list)
    for (nn, _, _, _, r) in recs:
        by_dim[nn].append(r)
    dims = sorted(by_dim)
    rates = [statistics.mean(by_dim[k]) for k in dims]
    bars = a.bar(dims, rates, width=0.7, color="C2", alpha=0.85)
    annotate_counts(a, bars, [len(by_dim[k]) for k in dims])
    a.set_xticks(dims); a.set_ylim(0, 1.08)
    a.set_xlabel("dimension n"); a.set_ylabel("fraction resolved")
    a.set_title("1. Resolution rate vs dimension")

    # 2. resolution rate vs determinant d (binned)
    a = ax[0, 1]
    edges = np.linspace(meta["d_min"], meta["d_max"] + 1, 7)
    centers, rrates, rcounts = [], [], []
    for i in range(len(edges) - 1):
        sel = [r for (_, dd, _, _, r) in recs if edges[i] <= dd < edges[i + 1]]
        if sel:
            centers.append((edges[i] + edges[i + 1]) / 2)
            rrates.append(statistics.mean(sel)); rcounts.append(len(sel))
    width = (edges[1] - edges[0]) * 0.85
    bars = a.bar(centers, rrates, width=width, color="C0", alpha=0.85)
    annotate_counts(a, bars, rcounts)
    a.set_ylim(0, 1.08)
    a.set_xlabel("determinant d"); a.set_ylabel("fraction resolved")
    a.set_title("2. Resolution rate vs determinant")

    # 3. gap distribution (resolved only)
    a = ax[0, 2]
    if gaps:
        lo, hi = min(gaps), max(gaps)
        a.hist(gaps, bins=range(lo, hi + 2), align="left", rwidth=0.85, color="C4")
        a.axvline(0.0, color="gray", lw=1, ls="--")
        n_opt = sum(1 for g in gaps if g <= 0)
        a.set_title(f"3. Gap distribution  (≤min_sum: {n_opt}/{len(gaps)})")
    else:
        a.text(0.5, 0.5, "no resolved cones", ha="center", va="center")
        a.set_title("3. Gap distribution")
    a.set_xlabel("gap (agent − min_sum)"); a.set_ylabel("cones")

    # 4. parity: min_sum steps vs agent steps (resolved only)
    a = ax[1, 0]
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
    a = ax[1, 1]
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
    a = ax[1, 2]
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
    stored per-cone records when available, else doing a fresh rollout."""
    model_path = _resolve_model_path(args.model)
    if not os.path.exists(model_path):
        print(f"model not found: {model_path}"); return
    out_png = args.out or os.path.join(os.path.dirname(model_path), "validation_detail.png")
    detail_records = os.path.splitext(out_png)[0] + "_records.json"
    sweep_records = os.path.join(args.sweep_dir, "validation_eval_records.json")

    # prefer a per-model detail records file, then the sweep-level one
    found = None
    for rp in (detail_records, sweep_records):
        hit = find_model_record(rp, model_path)
        if hit:
            found = hit
            print(f"reusing stored records from {rp}")
            break

    if found:
        meta, model = found
    else:
        if args.replot:
            print("no stored records for this model; run without --replot to roll out first.")
            return
        cones = load_depot(args.n_max, args.d_min, args.d_max)
        print(f"rolling out {os.path.basename(os.path.dirname(model_path))} on "
              f"{len(cones)} cones (n≤{args.n_max}, {args.d_min}≤d≤{args.d_max}) ...")
        model = eval_one_model(model_path, cones, args.max_steps)
        if "error" in model:
            print(f"eval failed: {model['error']}"); return
        meta = dict(n_max=args.n_max, d_min=args.d_min, d_max=args.d_max,
                    n_cones=len(cones), max_steps=args.max_steps)
        with open(detail_records, "w") as f:
            json.dump({"meta": meta, "models": [model]}, f)
        print(f"wrote {detail_records}")

    make_detail_png(model, out_png, meta)


def main():
    ap = argparse.ArgumentParser(description="Evaluate every sweep model on the validation depot.")
    ap.add_argument("--sweep-dir", default=os.path.join(HERE, "sweep", "sweep_0"),
                    help="sweep folder holding the t*/model.pt checkpoints to evaluate")
    ap.add_argument("--n-max", type=int, default=4,
                    help="keep only depot cones with dimension n <= this")
    ap.add_argument("--d-min", type=int, default=2,
                    help="keep only depot cones with determinant d >= this")
    ap.add_argument("--d-max", type=int, default=30,
                    help="keep only depot cones with determinant d <= this")
    ap.add_argument("--max-steps", type=int, default=80,
                    help="max greedy subdivisions per cone before it counts as unresolved")
    ap.add_argument("--group-by", choices=GROUPABLE, default=None,
                    help="axis to compare models across (default: auto-detect the sweep's varied axis)")
    ap.add_argument("--workers", type=int, default=12,
                    help="parallel model-evaluation processes")
    ap.add_argument("--out", default=None,
                    help="output PNG path (default: <sweep-dir>/validation_eval.png)")
    ap.add_argument("--replot", action="store_true",
                    help="rebuild the PNG from saved records (no rollouts)")
    ap.add_argument("--model", default=None,
                    help="single-model detail mode: render a read_diagnostics-style depot "
                         "report for ONE model (trial dir or model.pt). Reuses stored "
                         "records if present, else does a fresh rollout")
    args = ap.parse_args()

    if args.model:
        run_detail(args)
        return

    out_png = args.out or os.path.join(args.sweep_dir, "validation_eval.png")
    records_path = os.path.splitext(out_png)[0] + "_records.json"

    if args.replot:
        blob = json.load(open(records_path))
        models = [{"path": m["path"], "embedding_size": m["embedding_size"],
                   "layer_type": m.get("layer_type", "GAT"),
                   "seed": m["seed"], "records": [tuple(r) for r in m["records"]]}
                  for m in blob["models"]]
        make_png(models, out_png, blob["meta"])
        print(f"replotted from {records_path}")
        return

    group_key = args.group_by or detect_group_key(args.sweep_dir)
    cones = load_depot(args.n_max, args.d_min, args.d_max)
    model_paths = sorted(glob.glob(os.path.join(args.sweep_dir, "t*", "model.pt")))
    print(f"grouping comparison by: {group_key}")
    print(f"{len(model_paths)} models  x  {len(cones)} cones  "
          f"(n<={args.n_max}, {args.d_min}<=d<={args.d_max})  "
          f"max_steps={args.max_steps}  workers={args.workers}")

    models, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(eval_one_model, p, cones, args.max_steps): p for p in model_paths}
        bar = tqdm(as_completed(futs), total=len(futs), desc="eval", unit="model")
        for fut in bar:
            r = fut.result()
            name = os.path.basename(os.path.dirname(r["path"]))
            if "error" in r:
                errors.append(r); bar.write(f"  SKIP {name}: {r['error']}")
            else:
                s = summarize(r)
                models.append(r)
                bar.write(f"  {name} emb={r['embedding_size']} res={s['res_rate']:.3f} "
                          f"gap={s['mean_gap']:.2f} score={s['score']:.2f}")

    if not models:
        print("no models evaluated; aborting."); return
    meta = dict(n_max=args.n_max, d_min=args.d_min, d_max=args.d_max,
                n_cones=len(cones), n_models=len(models), group_key=group_key,
                max_steps=args.max_steps,
                constants=constant_hparams(models, group_key))

    # persist raw per-cone records so future restyles need no rollouts (--replot)
    with open(records_path, "w") as f:
        json.dump({"meta": meta, "models": models}, f)
    print(f"wrote {records_path}")

    make_png(models, out_png, meta)

    # also dump raw per-model metrics next to the png
    with open(os.path.splitext(out_png)[0] + ".csv", "w") as f:
        f.write("trial_dir,embedding_size,layer_type,seed,res_rate,mean_gap,tie,beat,score\n")
        for m in sorted(models, key=lambda x: (x["embedding_size"], x["layer_type"], x["seed"])):
            s = summarize(m)
            f.write(f"{os.path.basename(os.path.dirname(m['path']))},{m['embedding_size']},"
                    f"{m['layer_type']},{m['seed']},"
                    f"{s['res_rate']:.4f},{s['mean_gap']:.4f},{s['tie']:.4f},{s['beat']:.4f},{s['score']:.4f}\n")
    print(f"wrote {os.path.splitext(out_png)[0] + '.csv'}")


if __name__ == "__main__":
    main()
