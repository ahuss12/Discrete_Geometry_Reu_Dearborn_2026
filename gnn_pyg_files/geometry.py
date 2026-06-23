#math/environment file, tells what a cone state is, what actions are legal etc. So this the 'dseigning environment' part from the iclr blog 
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from fractions import Fraction
from itertools import product
import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


IntTuple = Tuple[int, ...]

#bareiss algo for det, dont wont weird rounding via numpy
def det_int(mat: np.ndarray | Sequence[Sequence[int]]) -> int:
    """Exact integer determinant for small integer matrices using Bareiss.

    This avoids SymPy determinant calls inside candidate enumeration/MCTS.
    """
    M = [list(map(int, row)) for row in np.asarray(mat, dtype=object).tolist()]
    n = len(M)
    if n == 0:
        return 1
    if any(len(row) != n for row in M):
        raise ValueError("det_int expects a square matrix")

    sign = 1
    prev = 1
    for k in range(n - 1):
        if M[k][k] == 0:
            swap = None
            for r in range(k + 1, n):
                if M[r][k] != 0:
                    swap = r
                    break
            if swap is None:
                return 0
            M[k], M[swap] = M[swap], M[k]
            sign *= -1
        pivot = M[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                M[i][j] = (M[i][j] * pivot - M[i][k] * M[k][j]) // prev
            M[i][k] = 0
        prev = pivot
    return int(sign * M[-1][-1])

#gcd of list of vectors basically a helper
def gcd_many(values: Iterable[int]) -> int:
    g = 0
    for v in values:
        g = math.gcd(g, abs(int(v)))
    return g

#primitivize it eg (6,9,3) to (2,3,1)
def primitive_vector(v: Sequence[int]) -> IntTuple:
    vals = tuple(int(x) for x in v)
    g = gcd_many(vals)
    if g == 0:
        raise ValueError("zero vector has no primitive direction")
    return tuple(x // g for x in vals)


def is_zero_vector(v: Sequence[int]) -> bool:
    return all(int(x) == 0 for x in v)

#feature scaling helper, prevents huge matrices entires from dominating neural features
def sign_log1p_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def extended_euclid(a: int, b: int) -> tuple[int, int, int]:
    """Return (g,x,y) with g = x*a + y*b = gcd(a,b)."""
    old_r, r = int(a), int(b)
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    if old_r < 0:
        old_r, old_s, old_t = -old_r, -old_s, -old_t
    return old_r, old_s, old_t


def canonical_form(M: list[list[int]]) -> list[list[int]]:
#to efficiently enumerate lattice points in FPP
    n = len(M)
    if any(len(row) != n for row in M):
        raise ValueError("canonical_form expects a square matrix")

    # Triangularization: reduce column entries [x,y] -> [g,0].
    for j in range(n):
        for i in range(j + 1, n):
            if M[i][j] == 0:
                continue
            g, a, b = extended_euclid(M[j][j], M[i][j])
            row_j = [a * M[j][col] + b * M[i][col] for col in range(n)]
            row_i = [
                -(M[i][j] // g) * M[j][col] + (M[j][j] // g) * M[i][col]
                for col in range(n)
            ]
            M[j], M[i] = row_j, row_i
        if M[j][j] == 0:
            raise ValueError("matrix appears singular during canonical_form")
        if M[j][j] < 0:
            M[j] = [-x for x in M[j]]

    # Reduce entries above pivots.
    for j in range(n):
        pivot = M[j][j]
        for i in range(j):
            quotient = M[i][j] // pivot
            if quotient:
                M[i] = [M[i][col] - quotient * M[j][col] for col in range(n)]
    return M


def fundamental_parallelepiped_points(H: np.ndarray) -> list[IntTuple]:
    #Enumerate lattice points in the half-open fundamental parallelepiped.For columns v_i of H, this enumerates lattice points in{ sum_i lambda_i v_i : 0 <= lambda_i < 1 }.

#    The number of returned points should be |det(H)| for a full-rank integer H.

    A = np.asarray(H, dtype=int)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("H must be square")
    n = int(A.shape[0])
    T = canonical_form(A.tolist())

    lambdas: list[list[Fraction]] = [[Fraction(i, T[-1][-1])] for i in range(T[-1][-1])]
    for i in reversed(range(n - 1)):
        new_lambdas: list[list[Fraction]] = []
        for curr in lambdas:
            s = sum(curr[col - (i + 1)] * T[i][col] for col in range(i + 1, n))
            for k in range(T[i][i]):
                new_lambdas.append([Fraction(math.ceil(s) - s + k, T[i][i])] + curr)
        lambdas = new_lambdas

    points: list[IntTuple] = []
    rows = A.tolist()
    for vec in lambdas:
        p = tuple(int(sum(rows[row][col] * vec[col] for col in range(n))) for row in range(n))
        points.append(p)
    return points


@dataclass(frozen=True)
class Cone:
    # represents Full-dimensional rational cone represented by an integer generator matrix. columns of H are primitive generators v_1, ..., v_n.

    H: np.ndarray
    depth: int = 0
#self explanatory with the pretty prints below
    def __post_init__(self) -> None:
        H = np.asarray(self.H, dtype=int)
        object.__setattr__(self, "H", H)
        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise ValueError("Cone.H must be a square integer matrix.")
        if H.shape[0] == 0:
            raise ValueError("Cone dimension must be positive")
        if det_int(H) == 0:
            raise ValueError("Cone.H must be full rank")
        for j in range(H.shape[1]):
            col = tuple(int(x) for x in H[:, j])
            if is_zero_vector(col):
                raise ValueError("cone generators cannot be zero")
            if gcd_many(col) != 1:
                raise ValueError(
                    "cone generators must be primitive; use primitive-ray training cones"
                )
   #ambient dimension
    @property
    def n(self) -> int:
        return int(self.H.shape[0])
    #absoulte det ie multiplicity and det
    @cached_property
    def det_signed(self) -> int:
        return det_int(self.H)

    @cached_property
    def det(self) -> int:
        return abs(self.det_signed)

    @property
    def is_unimodular(self) -> bool:
        return self.det == 1

    def key(self) -> Tuple[int, ...]:
        # Include shape, depth and flattened matrix. For MCTS/enumerator caching this is enough.
        return (self.n, self.depth, *tuple(int(x) for x in self.H.reshape(-1)))


@dataclass(frozen=True)
class ConeState:
  #A state is the current list of active singular cones.

    #We discard unimodular cones after subdivision, so terminal states often have an empty tuple of cones.

    cones: Tuple[Cone, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cones", tuple(c for c in self.cones if not c.is_unimodular))

    @property
    def is_terminal(self) -> bool:
        return len(self.cones) == 0

    def key(self) -> Tuple[Tuple[int, ...], ...]:
        # Active cones are mathematically unordered for the objective. Sorting helps cache hits.
        return tuple(sorted(c.key() for c in self.cones))


@dataclass(frozen=True)
class LocalRayCandidate:
    """A legal local ray insertion candidate inside one cone.

    If H has determinant D and r = sum_i b_i v_i, we store b_i as

        b_i = bary_num[i] / bary_den.

    For a primitive ray inside the cone, child determinants are

        child_dets[i] = |det(v_1,...,r,...,v_n)| = D * b_i.

    In the common case here, bary_den = D and bary_num = child_dets.
    """

    ray: IntTuple
    bary_num: IntTuple
    bary_den: int
    child_dets: IntTuple

    def score_tuple(self) -> Tuple[int, int, int, Tuple[int, ...]]:
        smooth = sum(1 for d in self.child_dets if d == 1)
        return (max(self.child_dets), sum(self.child_dets), -smooth, self.child_dets)


@dataclass(frozen=True)
class GlobalAction:
    """An action in a whole state: choose a candidate ray inside one active cone."""

    cone_index: int
    candidate: LocalRayCandidate


class CandidateEnumerator:
    """Interface for legal ray/candidate enumeration."""

    def enumerate_for_cone(self, cone: Cone) -> List[LocalRayCandidate]:
        raise NotImplementedError


class FundamentalParallelepipedEnumerator(CandidateEnumerator):
    #Candidate generator from fundamental-parallelepiped lattice points.


    def __init__(
        self,
        max_dim: int = 7,
        max_candidates: int = 128,
        max_points: int = 20_000,
        strict_interior: bool = True,
        cache: bool = True,
    ) -> None:
        self.max_dim = int(max_dim)
        self.max_candidates = int(max_candidates)
        self.max_points = int(max_points)
        self.strict_interior = bool(strict_interior)
        self.cache = bool(cache)
        self._cache: Dict[Tuple[int, ...], Tuple[LocalRayCandidate, ...]] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    def enumerate_for_cone(self, cone: Cone) -> List[LocalRayCandidate]:
        if cone.is_unimodular:
            return []
        if cone.n > self.max_dim:
            raise ValueError(f"cone dimension {cone.n} exceeds max_dim={self.max_dim}")

        key = cone.key()
        if self.cache and key in self._cache:
            return list(self._cache[key])

        # This enumerator is exact for det <= max_points.  For larger determinants,
        # use a fallback/hybrid enumerator or increase max_points.
        if cone.det > self.max_points:
            result: list[LocalRayCandidate] = []
            if self.cache:
                self._cache[key] = tuple(result)
            return result

        seen_rays: set[IntTuple] = set()
        out: list[LocalRayCandidate] = []
        for point in fundamental_parallelepiped_points(cone.H):
            if is_zero_vector(point):
                continue
            cand = candidate_from_ray(
                cone,
                point,
                require_strict_positive=self.strict_interior,
                require_progress=True,
            )
            if cand is None:
                continue
            if cand.ray in seen_rays:
                continue
            seen_rays.add(cand.ray)
            out.append(cand)

        out.sort(key=lambda c: c.score_tuple())
        out = out[: self.max_candidates]
        if self.cache:
            self._cache[key] = tuple(out)
        return list(out)


class GridCandidateEnumerator(CandidateEnumerator):
   

    #This is not meant to replace fundamental-parallelepiped/Hilbert-basis
    #enumeration. It tries barycentric numerator vectors d with entries in
    #1,...,D}, checks whether H d / D is an integer lattice ray, then converts
    #the result into a primitive ray and computes exact child determinants.


    def __init__(
        self,
        max_dim: int = 7,
        max_candidates: int = 128,
        max_grid_points: int = 200_000,
        random_trials: int = 2_000,
        seed: int = 0,
        cache: bool = True,
    ) -> None:
        self.max_dim = int(max_dim)
        self.max_candidates = int(max_candidates)
        self.max_grid_points = int(max_grid_points)
        self.random_trials = int(random_trials)
        self.rng = random.Random(seed)
        self.cache = bool(cache)
        self._cache: Dict[Tuple[int, ...], Tuple[LocalRayCandidate, ...]] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    def enumerate_for_cone(self, cone: Cone) -> List[LocalRayCandidate]:
        if cone.is_unimodular:
            return []
        if cone.n > self.max_dim:
            raise ValueError(f"cone dimension {cone.n} exceeds max_dim={self.max_dim}")

        key = cone.key()
        if self.cache and key in self._cache:
            return list(self._cache[key])

        D = cone.det
        n = cone.n
        seen = set()
        out: List[LocalRayCandidate] = []

        def try_d(d: Sequence[int]) -> None:
            if len(out) >= self.max_candidates:
                return
            cand = candidate_from_bary_nums(cone, tuple(int(x) for x in d))
            if cand is None:
                return
            key2 = (cand.ray, cand.child_dets)
            if key2 in seen:
                return
            seen.add(key2)
            out.append(cand)

        # Cheap candidates for cyclic/HNF examples.
        for i in range(n):
            d = [D] * n
            d[i] = 1
            try_d(d)

        for i in range(n):
            for j in range(i + 1, n):
                d = [D] * n
                d[i] = 1
                d[j] = 1
                try_d(d)

        grid_size = D**n if D < 10_000 else self.max_grid_points + 1
        if self.max_grid_points > 0 and grid_size <= self.max_grid_points:
            for d in product(range(1, D + 1), repeat=n):
                if all(x == D for x in d):
                    continue
                try_d(d)
                if len(out) >= self.max_candidates:
                    break
        elif self.random_trials > 0:
            for _ in range(self.random_trials):
                d = tuple(self.rng.randint(1, D) for _ in range(n))
                if all(x == D for x in d):
                    continue
                try_d(d)
                if len(out) >= self.max_candidates:
                    break

        out.sort(key=lambda c: c.score_tuple())
        out = out[: self.max_candidates]
        if self.cache:
            self._cache[key] = tuple(out)
        return list(out)


class HybridCandidateEnumerator(CandidateEnumerator):
    #use fpp first

    def __init__(
        self,
        fpp: Optional[FundamentalParallelepipedEnumerator] = None,
        grid: Optional[GridCandidateEnumerator] = None,
        min_candidates: int = 1,
        max_candidates: int = 128,
    ) -> None:
        self.fpp = fpp or FundamentalParallelepipedEnumerator(max_candidates=max_candidates)
        self.grid = grid or GridCandidateEnumerator(max_candidates=max_candidates)
        self.min_candidates = int(min_candidates)
        self.max_candidates = int(max_candidates)

    def enumerate_for_cone(self, cone: Cone) -> List[LocalRayCandidate]:
        out = self.fpp.enumerate_for_cone(cone)
        if len(out) >= self.min_candidates:
            return out[: self.max_candidates]
        seen = {(c.ray, c.child_dets) for c in out}
        for cand in self.grid.enumerate_for_cone(cone):
            key = (cand.ray, cand.child_dets)
            if key in seen:
                continue
            out.append(cand)
            seen.add(key)
            if len(out) >= self.max_candidates:
                break
        out.sort(key=lambda c: c.score_tuple())
        return out[: self.max_candidates]


def candidate_from_ray(
    cone: Cone,
    ray: Sequence[int],
    *,
    require_strict_positive: bool = True,
    require_progress: bool = True,
) -> Optional[LocalRayCandidate]:
    """Turn an integer ray into a LocalRayCandidate if it is legal/useful."""
    if is_zero_vector(ray):
        return None
    try:
        r = primitive_vector(ray)
    except ValueError:
        return None

    H = cone.H
    D = cone.det
    detH = cone.det_signed
    n = cone.n
    if D <= 1:
        return None

    child: List[int] = []
    for i in range(n):
        Hc = H.copy()
        Hc[:, i] = np.asarray(r, dtype=int)
        num = det_int(Hc)
        # Normalize orientation so positive determinant means positive barycentric coefficient.
        if detH < 0:
            num = -num
        if require_strict_positive:
            if num <= 0:
                return None
        else:
            if num < 0:
                return None
        child.append(int(num))

    if sum(1 for x in child if x > 0) <= 1:
        return None
    if require_progress:
        if max(child) > D:
            return None
        if all(x == D for x in child):
            return None

    return LocalRayCandidate(
        ray=tuple(int(x) for x in r),
        bary_num=tuple(child),
        bary_den=D,
        child_dets=tuple(child),
    )


def candidate_from_bary_nums(cone: Cone, d: IntTuple) -> Optional[LocalRayCandidate]:
    """Convert tentative D*barycentric numerators into a primitive ray candidate."""
    H = cone.H
    D = cone.det
    n = cone.n
    if len(d) != n:
        raise ValueError("barycentric numerator length must match cone dimension")
    if D <= 1:
        return None
    if any(x <= 0 for x in d):
        return None
    if all(x == D for x in d):
        return None

    d_vec = np.asarray(d, dtype=int)
    r_num = H @ d_vec
    if not np.all(r_num % D == 0):
        return None
    raw_r = tuple(int(x // D) for x in r_num)
    return candidate_from_ray(
        cone,
        raw_r,
        require_strict_positive=True,
        require_progress=True,
    )


def enumerate_actions(state: ConeState, enumerator: CandidateEnumerator) -> List[GlobalAction]:
    actions: List[GlobalAction] = []
    for ci, cone in enumerate(state.cones):
        if cone.is_unimodular:
            continue
        for cand in enumerator.enumerate_for_cone(cone):
            actions.append(GlobalAction(cone_index=ci, candidate=cand))
    return actions


def subdivide_cone(cone: Cone, candidate: LocalRayCandidate) -> List[Cone]:
    """Stellar subdivision of one cone by a candidate ray.

    Returns only singular child cones; unimodular children are considered solved.
    This version assumes strict-interior candidates, so every child determinant is positive.
    """
    H = cone.H
    r = np.asarray(candidate.ray, dtype=int)
    children: List[Cone] = []
    for i in range(cone.n):
        Hc = H.copy()
        Hc[:, i] = r
        # If someone passes a boundary candidate, skip degenerate replacements.
        if det_int(Hc) == 0:
            continue
        child = Cone(Hc, depth=cone.depth + 1)
        if not child.is_unimodular:
            children.append(child)
    return children


def apply_action(state: ConeState, action: GlobalAction) -> ConeState:
    cones = list(state.cones)
    parent = cones.pop(action.cone_index)
    children = subdivide_cone(parent, action.candidate)
    cones.extend(children)
    return ConeState(tuple(cones))
