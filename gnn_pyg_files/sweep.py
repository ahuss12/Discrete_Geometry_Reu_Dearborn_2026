#!/usr/bin/env python3
"""General coordinate (one-at-a-time) hyperparameter sweep for train.py.

ANY train.py argument is a tunable hyperparameter (the full set is discovered
straight from train.py, so the two never drift apart).  Everything stays at its
train.py default unless you:
  * pin a different constant for the whole run with  --set NAME=VALUE, or
  * sweep it with  --sweep NAME=v1,v2,...  (repeatable; one axis varied at a time).
The all-defaults point is the shared BASELINE anchor and is run once.  Every
config is repeated over --seeds seeds (the reports average over them).

Each invocation writes a fresh timestamped folder results/sweep/<ts>/ holding
summary.csv, sweep_meta.json, the per-trial subdirs, and the comparison PNGs --
so repeated sweeps stay separate.  Pass --resume <folder> to continue one.

  python3 sweep.py --list                         # every tunable arg + its default
  python3 sweep.py --sweep lr=3e-5,1e-4,3e-4,1e-3 --sweep c_puct=0.5,1,1.5,2 \
                   --set episodes=300 --set device=cpu --seeds 3 --budget-hours 10
  python3 sweep.py --resume results/sweep/20260629_123456   # continue that run
  python3 sweep.py --leaderboard                  # standings of the newest run
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train import build_parser

# ===========================================================================================
#  TUNABLE SURFACE -- discovered from train.py's own argument parser.
# ===========================================================================================
def _discover():
    defaults, types = {}, {}
    for a in build_parser()._actions:
        if a.dest == "help":
            continue
        defaults[a.dest] = a.default
        types[a.dest] = a.type or str
    return defaults, types

DEFAULTS, TYPES = _discover()
RESERVED = {"seed", "save", "diag_dir", "resume"}     # managed by the harness; not swept
TUNABLE = [k for k in DEFAULTS if k not in RESERVED]
SUMMARY_META = ["trial", "seed", "axis", "status", "score", "resolution_rate",
                "mean_gap", "tie_rate", "n_timeout", "elapsed_s", "run_dir"]
SUMMARY_FIELDS = SUMMARY_META + sorted(TUNABLE)       # every config's full param vector


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")

def _cast(name: str, s: str):
    try:
        return TYPES[name](s)
    except (ValueError, TypeError):
        tn = getattr(TYPES[name], "__name__", "value")
        raise SystemExit(f"bad value {s!r} for {_flag(name)} (expected {tn})")


# ===========================================================================================
#  CONFIG CONSTRUCTION  -- baseline anchor + one axis perturbed at a time.
# ===========================================================================================
def build_configs(base: dict, sweeps: dict) -> list[dict]:
    configs = [{"id": 0, "axis": "baseline", "params": dict(base)}]
    cid = 1
    for axis, values in sweeps.items():
        for v in values:
            if v == base[axis]:
                continue                              # equals baseline -> the anchor covers it
            configs.append({"id": cid, "axis": axis, "params": {**base, axis: v}})
            cid += 1
    return configs


def score_run(run_dir: str) -> dict:
    """Summarize a finished run from episode_quality.csv.  LOWER score = better:
       mean inserted-ray gap over the last third + a penalty for unresolved episodes."""
    rows = list(csv.DictReader(open(os.path.join(run_dir, "episode_quality.csv"))))
    n = len(rows)
    if n == 0:
        return {"score": float("inf"), "episodes": 0, "resolution_rate": 0.0,
                "mean_gap": float("inf"), "tie_rate": 0.0, "n_timeout": 0}
    res = lambda r: r["resolved"] == "1"
    tail = rows[2 * n // 3:]
    res_tail = [r for r in tail if res(r)]
    gaps = [int(r["gap"]) for r in tail]
    res_rate = sum(res(r) for r in tail) / len(tail)
    mean_gap = statistics.mean(gaps) if gaps else float("inf")
    tie_rate = (sum(1 for r in res_tail if int(r["gap"]) == 0) / len(res_tail)) if res_tail else 0.0
    return {"score": round(mean_gap + 5.0 * (1.0 - res_rate), 4), "episodes": n,
            "resolution_rate": round(res_rate, 4), "mean_gap": round(mean_gap, 4),
            "tie_rate": round(tie_rate, 4), "n_timeout": sum(not res(r) for r in tail)}


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


def run_one(cfg: dict, seed: int, run_dir: str) -> dict:
    diag_dir = os.path.join(run_dir, f"t{cfg['id']:03d}_s{seed}")
    os.makedirs(diag_dir, exist_ok=True)
    params = cfg["params"]
    cmd = [sys.executable, os.path.join(HERE, "train.py"),
           "--diag-dir", diag_dir, "--save", os.path.join(diag_dir, "model.pt"),
           "--seed", str(seed)]
    for k, v in params.items():
        cmd += [_flag(k), str(v)]

    t0 = time.perf_counter()
    with open(os.path.join(diag_dir, "train.log"), "w") as logf:
        proc = subprocess.run(cmd, cwd=HERE, stdout=logf, stderr=subprocess.STDOUT)
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
#  ANALYSIS  -- generic over whatever axes were swept (read from summary.csv).
# ===========================================================================================
def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None

def analyze(run_dir: str) -> None:
    sp = os.path.join(run_dir, "summary.csv")
    if not os.path.exists(sp):
        print(f"no results yet at {sp}"); return
    rows = [r for r in csv.DictReader(open(sp)) if r.get("status") == "ok"]
    if not rows:
        print("no successful runs yet."); return
    axes = sorted(set(r["axis"] for r in rows if r["axis"] not in ("baseline", "")))
    base_rows = [r for r in rows if r["axis"] == "baseline"]

    print(f"\n=== coordinate sweep ({len(rows)} runs) -- lower score is better, mean over seeds ===")
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
        best = min(pts, key=lambda p: p[1])
        print(f"\n  {axis}")
        print(f"    {'value':>12} {'score(mean±std)':>20} {'seeds':>5} {'res_rate':>9} {'tie_rate':>9}")
        for val, m, s, k, rr, tr in pts:
            star = "  <- best" if (val, m) == (best[0], best[1]) else ""
            print(f"    {str(val):>12} {f'{m:.3f}±{s:.3f}':>20} {k:>5} {rr:>9.3f} {tr:>9.3f}{star}")

    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[tuple(r[a] for a in axes)].append(r)
    bkey = min(by_cfg, key=lambda k: statistics.mean(float(x["score"]) for x in by_cfg[k]))
    bm = statistics.mean(float(x["score"]) for x in by_cfg[bkey])
    print(f"\n  overall best: {', '.join(f'{a}={v}' for a, v in zip(axes, bkey))}"
          f"  (mean score {bm:.3f}, {len(by_cfg[bkey])} seeds)")


# ===========================================================================================
#  RUN-FOLDER HELPERS
# ===========================================================================================
def _latest_run(outdir: str) -> str | None:
    subs = [d for d in glob.glob(os.path.join(outdir, "*"))
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "summary.csv"))]
    if os.path.exists(os.path.join(outdir, "summary.csv")):
        subs.append(outdir)                           # tolerate the old flat layout
    return max(subs, key=os.path.getmtime) if subs else None


def _write_reports(run_dir: str) -> None:
    try:
        import visualizations
        visualizations.make_reports(run_dir)
    except Exception as e:                            # plotting must never sink a finished sweep
        print(f"[report] skipped: {e}")


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", action="append", default=[], metavar="NAME=v1,v2,...",
                    help="hyperparameter axis to vary + values to test (repeatable)")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="pin a non-default constant for every run (repeatable)")
    ap.add_argument("--seeds", type=int, default=3, help="seeds per config")
    ap.add_argument("--outdir", default=os.path.join(HERE, "results", "sweep"),
                    help="container; each invocation gets its own timestamped subfolder")
    ap.add_argument("--resume", default=None, help="continue a previous run folder")
    ap.add_argument("--budget-hours", type=float, default=10.0)
    ap.add_argument("--leaderboard", action="store_true", help="print standings and exit")
    ap.add_argument("--list", action="store_true", help="list tunable args + defaults and exit")
    ap.add_argument("--no-report", action="store_true", help="skip the PNG comparison reports")
    ap.add_argument("--smoke", action="store_true", help="tiny self-contained plumbing test")
    args = ap.parse_args()

    if args.list:
        print("tunable train.py hyperparameters (name: default):")
        for k in sorted(TUNABLE):
            print(f"  {k}: {DEFAULTS[k]}")
        print(f"\nmanaged by the harness, not tunable: {sorted(RESERVED)}")
        return

    os.makedirs(args.outdir, exist_ok=True)
    if args.leaderboard:
        run_dir = args.resume or _latest_run(args.outdir)
        if run_dir is None:
            print(f"no sweep runs under {args.outdir}"); return
        print(f"leaderboard for {run_dir}")
        analyze(run_dir)
        return

    base, sweeps = _parse_specs(args)
    if args.smoke:
        for k, v in (("episodes", 5), ("max_steps", 40)):
            base[k] = v
        sweeps = sweeps or {"lr": [_cast("lr", "3e-05"), _cast("lr", "1e-4")]}
        args.seeds = 1
        print("[smoke] tiny regime")
    if not sweeps:
        raise SystemExit("nothing to sweep: pass at least one --sweep NAME=v1,v2,... (or --list)")

    if args.resume:
        run_dir = args.resume
        if not os.path.isdir(run_dir):
            raise SystemExit(f"--resume folder not found: {run_dir}")
        print(f"resuming {run_dir}")
    else:
        run_dir = os.path.join(args.outdir, time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(run_dir, exist_ok=True)
        print(f"new sweep run -> {run_dir}")
    summary_path = os.path.join(run_dir, "summary.csv")

    configs = build_configs(base, sweeps)
    if args.smoke:
        configs = configs[:2]
    seeds = list(range(args.seeds))

    # record the sweep layout so visualizations.py / leaderboard need no guessing
    with open(os.path.join(run_dir, "sweep_meta.json"), "w") as f:
        json.dump({"axes": sorted(sweeps),
                   "values": {a: list(sweeps[a]) for a in sweeps},
                   "baseline": {k: base[k] for k in TUNABLE},
                   "seeds": args.seeds}, f, indent=2, default=str)

    done = load_done(summary_path)
    total = len(configs) * len(seeds)
    print(f"axes {sorted(sweeps)} | {len(configs)} configs x {len(seeds)} seeds = {total} runs "
          f"| budget {args.budget_hours}h | done {len(done)}")
    print(f"writing -> {summary_path}\n")

    t_start = time.perf_counter()
    budget_s = args.budget_hours * 3600
    launched = 0
    for seed in seeds:                                # seeds outermost -> full breadth if cut short
        for cfg in configs:
            if (cfg["id"], seed) in done:
                continue
            if time.perf_counter() - t_start > budget_s:
                print(f"\n[budget] {args.budget_hours}h reached -- stopping. "
                      f"Re-run with --resume {run_dir} to continue.")
                analyze(run_dir)
                if not args.no_report:
                    _write_reports(run_dir)
                return
            launched += 1
            lbl = cfg["axis"]
            where = f"{lbl}={cfg['params'][lbl]:g}" if lbl != "baseline" and isinstance(cfg["params"][lbl], float) \
                    else (f"{lbl}={cfg['params'][lbl]}" if lbl != "baseline" else "baseline")
            print(f"[{launched}] trial {cfg['id']} ({where}) seed {seed}", flush=True)
            row = run_one(cfg, seed, run_dir)
            append_summary(summary_path, row)
            tag = (f"score={row['score']} res={row.get('resolution_rate')} tie={row.get('tie_rate')}"
                   if row["status"] == "ok" else f"!! {row['status']} (see {row['run_dir']}/train.log)")
            print(f"     -> {tag}  [{row['elapsed_s']}s]", flush=True)

    print("\nall runs complete.")
    analyze(run_dir)
    if not args.no_report:
        _write_reports(run_dir)


if __name__ == "__main__":
    main()
