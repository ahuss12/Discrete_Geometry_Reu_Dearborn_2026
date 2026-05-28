# @author Garrett Rhoads
# @file resolve_singularity.py
# @date 05/21/26
# @brief Discrete geometry - brute force backtracking algo to resolve singularity

import PyNormaliz
from PyNormaliz import *
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from sympy import Matrix
from itertools import product
from functools import reduce, lru_cache
from math import gcd

def lattice_in_parallelepiped(vecs):
    n = len(vecs)
    M = Matrix(vecs).T
    determinate = int(M.det())
    sign = 1 if determinate > 0 else -1

    adj = M.adjugate()

    corner = [0] * n
    for v in vecs:
        for j in range(n):
            corner[j] += v[j]

    ranges = []
    for j in range(n):
        ranges.append(range(1, corner[j]))

    candidates = list(product(*ranges))

    result = []
    for pt in candidates:
        v = sign * adj * Matrix(pt)
        inside = True
        for i in range(n):
            if not (0 < v[i] < abs(determinate)):
                inside = False
                break
        if inside:
            result.append(pt)

    return result

def split(vectors, p):
    sub_cones = []
    for i in range(len(vectors)):
        sub_cone = []
        for j in range(len(vectors)):
            if j == i:
                sub_cone.append(p)
            else:
                sub_cone.append(vectors[j])
        sub_cones.append(tuple(tuple(v) for v in sub_cone))
    return sub_cones

def is_primitive(pt):
    g = reduce(gcd, pt)
    return g == 1

def is_unimodular(vectors):
    return abs(int(Matrix(vectors).det())) == 1

# i pressed the dynamic programming button and im scared
@lru_cache(maxsize=None)
def decompose(vectors, depth=0):
    indent = "  " * depth
    det = abs(int(Matrix(vectors).det()))
    print(f"{indent}decompose | det={det} | vectors={list(vectors)}")

    if is_unimodular(vectors):
        print(f"{indent}  unimodular, done")
        return ()

    candidates = []
    for pt in lattice_in_parallelepiped(vectors):
        if is_primitive(pt):
            candidates.append(pt)

    print(f"{indent}  {len(candidates)} primitive candidates: {candidates}")

    if not candidates:
        print(f"{indent}  no candidates, returning")
        return ()

    best_splits = None
    for idx, p in enumerate(candidates):
        print(f"{indent}  trying candidate {idx + 1}/{len(candidates)}: {p}")
        sub_cones = split(vectors, p)
        splits = (p,)
        for sub_cone in sub_cones:
            splits += decompose(sub_cone, depth + 1)
        print(f"{indent}  candidate {p} -> {len(splits)} total splits")
        if best_splits is None or len(splits) < len(best_splits):
            best_splits = splits
            print(f"{indent}  new best: {len(best_splits)} splits")

    print(f"{indent}  returning best: {best_splits}")
    return best_splits

def main():
    vectors = tuple(tuple(v) for v in [[1, 3, 0], [3, 0, 1], [0, 1, 1]])
    print(f"vectors = ", end='')
    for v in vectors:
        print(f"{v}, ", end='')
    print()
    print(f"lattice points in parallelepiped = {lattice_in_parallelepiped(vectors)}")
    print(f"n = {len(lattice_in_parallelepiped(vectors))}")
    print(f"best splits = {decompose(vectors)}")

main()
