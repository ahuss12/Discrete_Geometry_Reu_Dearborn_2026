from __future__ import annotations

"""Baseline heuristics used by training and MCTS.

The important function is ``min_sum_steps`` / ``min_sum``.  We keep it outside
``train.py`` so MCTS can use the same baseline-to-go potential without importing
``train.py`` and creating a circular dependency.
"""

import random
import numpy as np

from CGLGraph import CGLGraph
from Cone import Cone, fanSubdivide
from utils import isPrimitiveNonzero

Vector = tuple[int, ...]


def min_sum(graph: CGLGraph) -> tuple[int, list[Cone], list[Vector]]:
    """Resolve a fan using the min-sum heuristic.

    Returns:
        steps: number of inserted rays used by the heuristic
        cones: final fan after heuristic resolution
        actions: inserted rays, as lattice coordinate tuples
    """
    cones = list(graph._cone_objects.values())
    step_count = 0
    actions: list[Vector] = []

    # Cache FPP/extraneous sets for Cone objects observed during this run.
    extraneous_set_cache: dict[Cone, dict[Vector, tuple]] = {}

    def extraneous(cone: Cone) -> dict[Vector, tuple]:
        d = extraneous_set_cache.get(cone)
        if d is None:
            pts, lams = cone.extraneousSet
            d = {p: lam for p, lam in zip(pts, lams) if isPrimitiveNonzero(p)}
            extraneous_set_cache[cone] = d
        return d

    while any(cone.isSingular for cone in cones):
        data = [(c.multiplicity, extraneous(c)) for c in cones]

        # No legal candidate: this should not happen for valid singular cones, but
        # protects callers from an empty min() crash.
        if not any(d for _, d in data):
            break

        total = sum(mult for mult, _ in data)
        det_sum = {p: total for _, d in data for p in d}

        for mult, d in data:
            for point, lambdas in d.items():
                det_sum[point] -= int(mult * (1 - sum(lambdas)))

        subdivision_point = min(det_sum, key=det_sum.get)
        cones = fanSubdivide(cones, subdivision_point)
        step_count += 1
        actions.append(subdivision_point)

    return step_count, cones, actions


def min_sum_steps(graph: CGLGraph) -> int:
    """Return only the min_sum steps-to-go for a graph state."""
    steps, _, _ = min_sum(graph)
    return int(steps)


def random_baseline(graph: CGLGraph, rng: random.Random, max_steps: int = 200) -> tuple[int, bool]:
    """Random rollout baseline used only for diagnostics."""
    cones = list(graph._cone_objects.values())
    step_count = 0

    while any(cone.isSingular for cone in cones):
        if step_count >= max_steps:
            return step_count, False

        candidates = set()
        for cone in cones:
            pts, _ = cone.extraneousSet
            for point in pts:
                if isPrimitiveNonzero(point):
                    candidates.add(point)

        if not candidates:
            return step_count, False

        point = rng.choice(list(candidates))
        cones = fanSubdivide(cones, point)
        step_count += 1

    return step_count, True


def graph_signature(graph: CGLGraph) -> tuple:
    """Hashable signature for caching baseline-to-go inside MCTS.

    Cone ids are irrelevant; only the current fan matters.
    """
    return tuple(sorted(tuple(cone.rays) for cone in graph._cone_objects.values()))
