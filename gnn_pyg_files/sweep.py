#!/usr/bin/env python3
"""Overnight coordinate (one-at-a-time) hyperparameter sweep for train.py.

Searches THREE knobs -- lr, c_puct, value_weight -- across 10 values each, varying
ONE at a time while the other two stay at BASELINE.  Everything else is pinned at
your validated settings, so each run is a clean controlled perturbation.  Every
config is repeated over multiple seeds (the only honest way to compare, given how
noisy single runs are).

Runs are ISOLATED train.py subprocesses (one at a time), scored from
episode_quality.csv, and logged one line each to <outdir>/summary.csv.  RESUMABLE
(re-run to continue after a crash/interrupt) and stops on a wall-clock budget.
Seeds are run outermost, so an early stop still leaves full breadth across configs.

  python3 sweep.py --smoke                       # ~30s plumbing test
  python3 sweep.py --seeds 3 --budget-hours 10   # the real run
  python3 sweep.py --leaderboard                 # read results in the morning
"""
from __future__ import annotations
import argparse
import csv
import glob
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

# ===========================================================================================
#  WHAT WE SEARCH  -- 3 knobs, 10 values each, varied one at a time around BASELINE.
# ===========================================================================================
BASELINE = {"lr": 3e-4, "c_puct": 1.5, "value_weight": 0.25}   # your current values

SWEEP_AXES = {
    "lr":           [3e-5, 1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3, 1.5e-3, 2e-3, 3e-3],
    "c_puct":       [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    "value_weight": [0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
}
# (each axis list includes its BASELINE value, so the baseline config runs once and
#  anchors all three curves.)

# ===========================================================================================
#  EVERYTHING ELSE  -- pinned at your validated config (NOT searched).  Override via CLI.
#  Regime first, then the held hyperparameters.
# ===========================================================================================
FIXED_DEFAULTS = {
    # --- problem regime ---
    "episodes":      300,
    "max_steps":     100,
    "min_dimension": 2,
    "max_dimension": 3,
    "det_min":       2,
    "det_max":       35,
    "device":        "cpu",
    # --- held hyperparameters (your current settings) ---
    "mcts_sims":       48,
    "hidden_dim":      128,
    "num_blocks":      4,
    "dropout":         0.05,
    "embedding_size":  7,
    "batch_size":      8,
    "epochs_per_iter": 1,
    "temperature":     1.0,
    "timeout_penalty": 1.0,
    "dirichlet_alpha": 0.3,
    "dirichlet_eps":   0.25,
}


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def build_configs() -> list[dict]:
    """One config per (axis, value); the baseline point is shared across axes."""
    out, seen = [], {}
    cid = 0
    for axis, values in SWEEP_AXES.items():
        for v in values:
            cfg = dict(BASELINE); cfg[axis] = v
            key = (cfg["lr"], cfg["c_puct"], cfg["value_weight"])
            if key in seen:
                continue
            is_base = all(abs(cfg[k] - BASELINE[k]) < 1e-15 for k in BASELINE)
            seen[key] = True
            out.append({"id": cid, "axis": "baseline" if is_base else axis, **cfg})
            cid += 1
    return out


def score_run(run_dir: str) -> dict:
    """Summarize a finished run from episode_quality.csv.  LOWER score = better:
       mean inserted-ray gap over the last third (timeouts counted at their censored
       gap) + a penalty for any episodes that failed to resolve."""
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


SUMMARY_FIELDS = ["trial", "seed", "axis", "status", "score", "resolution_rate",
                  "mean_gap", "tie_rate", "n_timeout", "elapsed_s",
                  "lr", "c_puct", "value_weight", "run_dir"]


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


def run_one(cfg: dict, seed: int, fixed: dict, outdir: str) -> dict:
    diag_dir = os.path.join(outdir, f"t{cfg['id']:03d}_s{seed}")
    os.makedirs(diag_dir, exist_ok=True)
    hp = {k: cfg[k] for k in BASELINE}
    cmd = [sys.executable, os.path.join(HERE, "train.py"),
           "--diag-dir", diag_dir, "--save", os.path.join(diag_dir, "model.pt"),
           "--seed", str(seed)]
    for k, v in {**fixed, **hp}.items():
        cmd += [_flag(k), str(v)]

    t0 = time.perf_counter()
    with open(os.path.join(diag_dir, "train.log"), "w") as logf:
        proc = subprocess.run(cmd, cwd=HERE, stdout=logf, stderr=subprocess.STDOUT)
    row = {"trial": cfg["id"], "seed": seed, "axis": cfg["axis"],
           "elapsed_s": round(time.perf_counter() - t0, 1), **hp}
    if proc.returncode != 0:
        return {**row, "status": "FAILED", "run_dir": diag_dir}
    rd = find_run_dir(diag_dir)
    if rd is None:
        return {**row, "status": "no_csv", "run_dir": diag_dir}
    return {**row, "status": "ok", "run_dir": rd, **score_run(rd)}


# ----------------------------------------------------------------------------- analysis
def analyze(summary_path: str) -> None:
    if not os.path.exists(summary_path):
        print(f"no results yet at {summary_path}")
        return
    rows = [r for r in csv.DictReader(open(summary_path)) if r.get("status") == "ok"]
    if not rows:
        print("no successful runs yet.")
        return
    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[(float(r["lr"]), float(r["c_puct"]), float(r["value_weight"]))].append(r)

    def cell(rs, f): return statistics.mean(float(x[f]) for x in rs)

    print(f"\n=== coordinate sweep ({len(rows)} runs, {len(by_cfg)} configs) "
          f"-- lower score is better, mean over seeds ===")
    for axis, others in [("lr", ("c_puct", "value_weight")),
                         ("c_puct", ("lr", "value_weight")),
                         ("value_weight", ("lr", "c_puct"))]:
        pts = []
        for key, rs in by_cfg.items():
            d = {"lr": key[0], "c_puct": key[1], "value_weight": key[2]}
            if all(abs(d[o] - BASELINE[o]) < 1e-12 for o in others):
                sc = [float(x["score"]) for x in rs]
                pts.append((d[axis], statistics.mean(sc),
                            statistics.pstdev(sc) if len(sc) > 1 else 0.0, len(sc),
                            cell(rs, "resolution_rate"), cell(rs, "tie_rate")))
        if not pts:
            continue
        pts.sort()
        best = min(pts, key=lambda p: p[1])
        print(f"\n  {axis}  (others at baseline: "
              f"{', '.join(f'{o}={BASELINE[o]}' for o in others)})")
        print(f"    {'value':>10} {'score(mean±std)':>20} {'seeds':>5} {'res_rate':>9} {'tie_rate':>9}")
        for v, m, s, k, rr, tr in pts:
            star = "  <- best" if (v, m) == (best[0], best[1]) else ""
            print(f"    {v:>10g} {f'{m:.3f}±{s:.3f}':>20} {k:>5} {rr:>9.3f} {tr:>9.3f}{star}")

    best_key = min(by_cfg, key=lambda k: statistics.mean(float(x["score"]) for x in by_cfg[k]))
    bm = statistics.mean(float(x["score"]) for x in by_cfg[best_key])
    print(f"\n  overall best config: lr={best_key[0]:g} c_puct={best_key[1]:g} "
          f"value_weight={best_key[2]:g}  (mean score {bm:.3f}, "
          f"{len(by_cfg[best_key])} seeds)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(HERE, "results", "sweep"))
    ap.add_argument("--seeds", type=int, default=3, help="seeds per config")
    ap.add_argument("--budget-hours", type=float, default=10.0)
    ap.add_argument("--leaderboard", action="store_true", help="print standings and exit")
    ap.add_argument("--smoke", action="store_true", help="2 configs x 1 seed x 5 episodes")
    for k, v in FIXED_DEFAULTS.items():
        ap.add_argument(_flag(k), default=v, type=type(v))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    summary_path = os.path.join(args.outdir, "summary.csv")
    if args.leaderboard:
        analyze(summary_path)
        return

    fixed = {k: getattr(args, k) for k in FIXED_DEFAULTS}
    configs = build_configs()
    if args.smoke:
        fixed.update({"episodes": 5, "max_steps": 40})
        configs = configs[:2]
        args.seeds = 1
        print("[smoke] 2 configs x 1 seed x 5 episodes")
    seeds = list(range(args.seeds))

    done = load_done(summary_path)
    total = len(configs) * len(seeds)
    print(f"sweep: {len(configs)} configs x {len(seeds)} seeds = {total} runs | "
          f"budget {args.budget_hours}h | done {len(done)}")
    print(f"regime/held: {fixed}\nwriting -> {summary_path}\n")

    t_start = time.perf_counter()
    budget_s = args.budget_hours * 3600
    launched = 0
    for seed in seeds:                       # seeds outermost -> full breadth if cut short
        for cfg in configs:
            if (cfg["id"], seed) in done:
                continue
            if time.perf_counter() - t_start > budget_s:
                print(f"\n[budget] {args.budget_hours}h reached -- stopping. "
                      f"Re-run to resume.")
                analyze(summary_path)
                return
            launched += 1
            print(f"[{launched}] trial {cfg['id']} ({cfg['axis']}) seed {seed}: "
                  f"lr={cfg['lr']:g} c_puct={cfg['c_puct']:g} value_weight={cfg['value_weight']:g}",
                  flush=True)
            row = run_one(cfg, seed, fixed, args.outdir)
            append_summary(summary_path, row)
            tag = (f"score={row['score']} res={row.get('resolution_rate')} "
                   f"tie={row.get('tie_rate')}" if row["status"] == "ok"
                   else f"!! {row['status']} (see {row['run_dir']}/train.log)")
            print(f"     -> {tag}  [{row['elapsed_s']}s]", flush=True)

    print("\nall runs complete.")
    analyze(summary_path)


if __name__ == "__main__":
    main()
