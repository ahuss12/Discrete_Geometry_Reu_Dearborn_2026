## WARNING: I just got this from Claude. This will need to be reviewed and edited in the future.

from __future__ import annotations
import json
import os
from typing import Optional, Sequence
from itertools import zip_longest


class TrajectoryLogger:
    """Append-only per-episode trajectory logger (JSONL, one line per episode).

    Records the root cone, the ordered subdivision points (inserted rays), and
    the final fan, for BOTH the agent and the min_sum baseline. Cost per episode
    is a handful of dict lookups plus one json.dumps -- negligible next to MCTS.
    The root cone + ordered points are enough to deterministically replay the
    whole trajectory via fanSubdivide, so no intermediate graphs are stored.
    """

    def __init__(self, path: str = "diagnostics/trajectories.jsonl"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._f = open(path, "w")

    def log(self, *, episode, root_rays, points, final_fan,
            resolved, agent_rays, baseline_rays,
            baseline_points=(), baseline_fan=()) -> None:
        rec = {
            "episode": int(episode),
            "resolved": bool(resolved),
            "agent_rays": int(agent_rays),
            "baseline_rays": int(baseline_rays),
            "root_cone": [list(r) for r in root_rays],
            "subdivisions": [list(p) for p in points],
            "final_fan": [[list(r) for r in cone] for cone in final_fan],
            "baseline_subdivisions": [list(p) for p in baseline_points],
            "baseline_fan": [[list(r) for r in cone] for cone in baseline_fan],
        }
        self._f.write(json.dumps(rec) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


# ---- tiny reader / pretty-printer (run: python trajectory_log.py <path>) ----
def read_trajectories(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def compare_to_min_sum(rec: dict) -> dict:
    """Pure dict reads, zero recomputation. Diffs the agent's inserted rays
    against the logged min_sum baseline rays."""
    agent = [tuple(p) for p in rec["subdivisions"]]
    base = [tuple(p) for p in rec.get("baseline_subdivisions", [])]
    base_set, agent_set = set(base), set(agent)
    return {
        "agent_rays":      len(agent),
        "min_sum_rays":    rec["baseline_rays"],
        "gap":             len(agent) - rec["baseline_rays"],  # <0 = agent beat baseline
        "agent_actions":   agent,
        "min_sum_actions": base,
        "shared":          agent_set & base_set,
        "agent_only":      [p for p in agent if p not in base_set],
        "min_sum_only":    [p for p in base if p not in agent_set],
    }


def show(rec: dict) -> None:
    """Print, for one episode: root cone, agent actions + final fan, then
    min_sum actions + final fan."""
    print(f"episode {rec['episode']}  resolved={rec['resolved']}  "
          f"agent={rec['agent_rays']} rays  min_sum={rec['baseline_rays']} rays  "
          f"gap={rec['agent_rays'] - rec['baseline_rays']:+d}")
    print(f"  root cone: {rec['root_cone']}")

    # ----- agent -----
    print(f"  agent inserted rays ({len(rec['subdivisions'])}):")
    for i, p in enumerate(rec["subdivisions"], 1):
        print(f"    step {i}: {p}")
    print(f"  agent final fan ({len(rec['final_fan'])} cones):")
    for cone in rec["final_fan"]:
        print(f"    {cone}")

    # ----- min_sum baseline -----
    base_pts = rec.get("baseline_subdivisions", [])
    base_fan = rec.get("baseline_fan", [])
    print(f"  min_sum inserted rays ({len(base_pts)}):")
    for i, p in enumerate(base_pts, 1):
        print(f"    step {i}: {p}")
    print(f"  min_sum final fan ({len(base_fan)} cones):")
    for cone in base_fan:
        print(f"    {cone}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="diagnostics/trajectories.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="only show first N episodes")
    args = ap.parse_args()
    recs = read_trajectories(args.path)
    if args.limit:
        recs = recs[:args.limit]
    for rec in recs:
        show(rec)
        print()