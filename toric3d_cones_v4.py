#!/usr/bin/env python3
"""
toric3d_cones.py

Generate 3D full-dimensional simplicial toric cones in HNF form up to a
chosen determinant, enumerate legal primitive subdivision rays, and test
several ray-choice strategies.

Cone convention
---------------
A cone state is stored in the HNF-like form

    H = [[1, u, x],
         [0, v, y],
         [0, 0, z]]

whose columns are the primitive ray generators

    e1 = (1,0,0),  (u,v,0),  (x,y,z).

We require

    v > 0, z > 0,
    gcd(u,v) = 1,
    gcd(x,y,z) = 1.

The determinant is D = v*z.  The cone is smooth iff D = 1.

Subdivision rays
----------------
For a cone with matrix H and determinant D, a candidate ray is represented
by barycentric residue data

    b = (b1,b2,b3),  0 <= bi < D,

such that

    r = H*b / D

is an integer primitive vector.  The determinants of the child cones are
exactly b1, b2, b3, omitting zero entries for face rays.

For HNF states, legal candidates can be enumerated in O(D) time.

hot to run, change det as needed-
    python toric3d_cones.py --max-det 100 --count
    python toric3d_cones.py --max-det 30 --unique --count
    python toric3d_cones.py --cone 1 2 1 1 3 --show-actions
    python toric3d_cones.py --cone 1 2 1 1 3 --strategy lex --simulate
    python toric3d_cones.py --cone 1 2 1 1 3 --exact-total
    python toric3d_cones.py --max-det 100 --export cones_det_le_100.jsonl
    python toric3d_cones.py --max-det 30 --unique --batch-greedy
    python toric3d_cones.py --max-det 30 --unique --compare-exact both
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from math import gcd
import argparse
import json
import random
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

Vec3 = Tuple[int, int, int]
Mat3 = Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]



#integer utilities



def gcd3(a: int, b: int, c: int) -> int:
    return gcd(gcd(abs(a), abs(b)), abs(c))


def egcd(a: int, b: int) -> Tuple[int, int, int]:
    """Return (g,x,y) with g >= 0 and x*a + y*b = g."""
    if a == 0 and b == 0:
        return 0, 0, 0

    aa, bb = abs(a), abs(b)
    old_r, r = aa, bb
    old_s, s = 1, 0
    old_t, t = 0, 1

    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t

    x = old_s if a >= 0 else -old_s
    y = old_t if b >= 0 else -old_t
    return old_r, x, y


def det3(M: Mat3) -> int:
    """Determinant of a 3x3 integer matrix stored by rows."""
    a, b, c = M[0]
    d, e, f = M[1]
    g, h, i = M[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def mat_vec_mul(M: Mat3, v: Vec3) -> Vec3:
    return tuple(sum(M[i][j] * v[j] for j in range(3)) for i in range(3))  # type: ignore[return-value]


def mat_mul(A: Mat3, B: Mat3) -> Mat3:
    return tuple(
        tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )  # type: ignore[return-value]


def col(M: Mat3, j: int) -> Vec3:
    return (M[0][j], M[1][j], M[2][j])



def matrix_from_columns(columns: Sequence[Vec3]) -> Mat3:
    if len(columns) != 3:
        raise ValueError("Need exactly three columns.")
    return tuple(tuple(columns[j][i] for j in range(3)) for i in range(3))  # type: ignore[return-value]


def permute_cols(M: Mat3, p: Tuple[int, int, int]) -> Mat3:
    return tuple(tuple(M[i][j] for j in p) for i in range(3))  # type: ignore[return-value]


def replace_col(M: Mat3, j: int, new_col: Vec3) -> Mat3:
    return tuple(
        tuple(new_col[i] if k == j else M[i][k] for k in range(3))
        for i in range(3)
    )  # type: ignore[return-value]


def row_linear_comb(a: Sequence[int], ca: int, b: Sequence[int], cb: int) -> List[int]:
    return [ca * a[i] + cb * b[i] for i in range(3)]





@dataclass(frozen=True, order=True)
class Cone:
    """A 3D simplicial cone in HNF-like coordinates."""

    u: int
    v: int
    x: int
    y: int
    z: int

    def matrix(self) -> Mat3:
        return ((1, self.u, self.x), (0, self.v, self.y), (0, 0, self.z))

    def det(self) -> int:
        return self.v * self.z

    def is_smooth(self) -> bool:
        return self.det() == 1

    def as_tuple(self) -> Tuple[int, int, int, int, int]:
        return (self.u, self.v, self.x, self.y, self.z)

    def canonical_key(self) -> Tuple[int, int, int, int, int, int]:
        """Sort key used when choosing among the 6 ray orderings."""
        return (self.det(), self.v, self.z, self.u, self.y, self.x)

    def check(self) -> None:
        if self.v <= 0 or self.z <= 0:
            raise ValueError(f"Need v,z > 0, got {self}")
        if not (0 <= self.u < self.v):
            raise ValueError(f"Need 0 <= u < v, got {self}")
        if not (0 <= self.x < self.z):
            raise ValueError(f"Need 0 <= x < z, got {self}")
        if not (0 <= self.y < self.z):
            raise ValueError(f"Need 0 <= y < z, got {self}")
        if gcd(self.u, self.v) != 1:
            raise ValueError(f"Second ray not primitive: {self}")
        if gcd3(self.x, self.y, self.z) != 1:
            raise ValueError(f"Third ray not primitive: {self}")


SMOOTH_CONE = Cone(0, 1, 0, 0, 1)


def reduce_primitive_vector_to_e1(a: Vec3) -> Mat3:
    """
    Return a unimodular row-operation matrix U with U*a = e1.

    This assumes a is primitive.  It is a pure-Python Euclidean algorithm.
    """
    U: List[List[int]] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    w = [a[0], a[1], a[2]]

    for idx in (1, 2):
        A, B = w[0], w[idx]
        if B == 0:
            continue
        g, s, t = egcd(A, B)
        if g == 0:
            continue
        old0 = U[0][:]
        oldi = U[idx][:]
        U[0] = row_linear_comb(old0, s, oldi, t)
        U[idx] = row_linear_comb(old0, -B // g, oldi, A // g)
        w[0], w[idx] = g, 0

    if w[0] == -1:
        U[0] = [-q for q in U[0]]
        w[0] = 1

    if w != [1, 0, 0]:
        raise ValueError(f"Vector {a} is not primitive; gcd is {abs(w[0])}.")

    return tuple(tuple(row) for row in U)  # type: ignore[return-value]


@lru_cache(maxsize=None)
def normalize_ordered_matrix(M: Mat3) -> Cone:
    """
    Normalize an ordered primitive ray matrix to

        [[1,u,x],[0,v,y],[0,0,z]]

    with v,z > 0 and 0 <= u < v, 0 <= x,y < z.

    The order of the three columns is respected.  To forget ray order, use
    canonical_cone(M), which tries all 6 permutations.
    """
    if det3(M) == 0:
        raise ValueError("Cone matrix is rank deficient.")

    for j in range(3):
        if gcd3(*col(M, j)) != 1:
            raise ValueError(f"Column {j} is not primitive: {col(M, j)}")

    U1 = reduce_primitive_vector_to_e1(col(M, 0))
    H = [list(row) for row in mat_mul(U1, M)]

    # Use a 2x2 unimodular transformation on rows 1,2 to send the lower
    # part of column 1 from (A,B) to (g,0).
    A, B = H[1][1], H[2][1]
    g, s, t = egcd(A, B)
    if g == 0:
        raise ValueError("First two rays are dependent.")

    old1 = H[1][:]
    old2 = H[2][:]
    H[1] = row_linear_comb(old1, s, old2, t)
    H[2] = row_linear_comb(old1, -B // g, old2, A // g)

    # Make z positive.  v should already be positive g.
    if H[1][1] < 0:
        H[1] = [-q for q in H[1]]
    if H[2][2] < 0:
        H[2] = [-q for q in H[2]]

    v = H[1][1]
    z = H[2][2]
    if v <= 0 or z <= 0:
        raise ValueError(f"Bad normalization, got v={v}, z={z}, H={H}")

    # Reduce u modulo v by row0 <- row0 - q row1.
    r = H[0][1] % v
    q = (H[0][1] - r) // v
    H[0] = [H[0][k] - q * H[1][k] for k in range(3)]

    # Reduce y modulo z by row1 <- row1 - q row2.
    r = H[1][2] % z
    q = (H[1][2] - r) // z
    H[1] = [H[1][k] - q * H[2][k] for k in range(3)]

    # Reduce x modulo z by row0 <- row0 - q row2.
    r = H[0][2] % z
    q = (H[0][2] - r) // z
    H[0] = [H[0][k] - q * H[2][k] for k in range(3)]

    if not (H[0][0] == 1 and H[1][0] == 0 and H[2][0] == 0 and H[2][1] == 0):
        raise AssertionError(f"Normalization failed: {H}")

    C = Cone(H[0][1], H[1][1], H[0][2], H[1][2], H[2][2])
    C.check()
    return C


@lru_cache(maxsize=None)
def canonical_cone(M: Mat3) -> Cone:
    """Canonicalize an unordered 3-ray cone by trying all 6 ray orders."""
    best: Optional[Cone] = None
    for p in permutations((0, 1, 2)):
        C = normalize_ordered_matrix(permute_cols(M, p))
        if best is None or C.canonical_key() < best.canonical_key():
            best = C
    assert best is not None
    return best



# HNF cone generation up to determinant max_det



def generate_ordered_hnf_cones(max_det: int) -> Iterable[Cone]:
    """
    Generate all ordered HNF-like primitive cones with determinant <= max_det.

    This includes many GL_3(Z)-equivalent duplicates because it fixes an
    ordered normal form rather than quotienting by the 6 choices of ray order.
    """
    if max_det < 1:
        return

    for v in range(1, max_det + 1):
        for z in range(1, max_det // v + 1):
            for u in range(v):
                if gcd(u, v) != 1:
                    continue
                for y in range(z):
                    for x in range(z):
                        if gcd3(x, y, z) != 1:
                            continue
                        yield Cone(u, v, x, y, z)


def generate_cones(max_det: int, unique: bool = False) -> List[Cone]:
    """
    Generate cones up to determinant max_det.

    If unique=True, canonicalizes under permutations of rays and unimodular
    lattice changes, returning one representative per canonical key.

    For max_det=100, unique=False is much faster and is the recommended
    dataset-generation mode.  Use unique=True for smaller ranges or when you
    really need one representative per equivalence class.
    """
    if not unique:
        return sorted(generate_ordered_hnf_cones(max_det), key=lambda c: c.canonical_key())

    seen: Dict[Tuple[int, int, int, int, int, int], Cone] = {}
    for C in generate_ordered_hnf_cones(max_det):
        K = canonical_cone(C.matrix())
        seen[K.canonical_key()] = K
    return sorted(seen.values(), key=lambda c: c.canonical_key())



# Legal ray enumeration and star subdivision



@dataclass(frozen=True)
class Action:
    """A candidate subdivision ray for a Cone."""

    b: Vec3
    ray: Vec3

    def child_dets(self) -> Tuple[int, ...]:
        return tuple(q for q in self.b if q > 0)

    def singular_child_count(self) -> int:
        return sum(1 for q in self.b if q > 1)

    def smooth_child_count(self) -> int:
        return sum(1 for q in self.b if q == 1)


@lru_cache(maxsize=None)
def legal_actions(C: Cone) -> Tuple[Action, ...]:
    """
    Enumerate all primitive legal rays r = H*b/D in O(D) time.

    For H = [[1,u,x],[0,v,y],[0,0,z]], D=vz, integrality gives

        b3 = v*t,                t = 0,...,z-1,
        b2 = -y*t mod z + z*s,   s = 0,...,v-1,
        b1 = -u*b2 - x*b3 mod D.

    The child cone determinants are the positive entries of b.
    """
    C.check()
    D = C.det()
    if D == 1:
        return []

    H = C.matrix()
    out: List[Action] = []

    for t in range(C.z):
        b3 = C.v * t
        base_b2 = (-C.y * t) % C.z
        for s in range(C.v):
            b2 = base_b2 + C.z * s
            b1 = (-C.u * b2 - C.x * b3) % D
            b = (b1, b2, b3)

            if b == (0, 0, 0):
                continue
            if sum(1 for q in b if q > 0) < 2:
                continue

            numerator = mat_vec_mul(H, b)
            if any(q % D != 0 for q in numerator):
                raise AssertionError(f"Internal integrality error: C={C}, b={b}, H*b={numerator}")

            ray = tuple(q // D for q in numerator)  # type: ignore[assignment]
            if gcd3(*ray) != 1:
                continue

            out.append(Action(b=b, ray=ray))

    out.sort(key=lambda a: (max(a.child_dets()), sum(a.child_dets()), a.b))
    return tuple(out)


@lru_cache(maxsize=None)
def star_subdivide(C: Cone, action: Action) -> Tuple[Cone, ...]:
    """
    Insert action.ray into C and return the canonical child cones.

    If action.b has a zero entry, the ray is on a face and only the
    nondegenerate children are returned.
    """
    H = C.matrix()
    children: List[Cone] = []

    for j, bj in enumerate(action.b):
        if bj <= 0:
            continue
        child_matrix = replace_col(H, j, action.ray)
        d = abs(det3(child_matrix))
        if d != bj:
            raise AssertionError(f"Child determinant mismatch: got {d}, expected {bj}")
        children.append(canonical_cone(child_matrix))

    return tuple(children)



# Ray-choice strategies



def _dets(a: Action) -> Tuple[int, ...]:
    return a.child_dets()


def score_min_max(C: Cone, a: Action) -> Tuple[int, int, int, Vec3]:
    ds = _dets(a)
    return (max(ds), sum(ds), a.singular_child_count(), a.b)


def score_min_sum(C: Cone, a: Action) -> Tuple[int, int, int, Vec3]:
    ds = _dets(a)
    return (sum(ds), max(ds), a.singular_child_count(), a.b)


def score_few_singular(C: Cone, a: Action) -> Tuple[int, int, int, Vec3]:
    ds = _dets(a)
    return (a.singular_child_count(), max(ds), sum(ds), a.b)


def score_smooth_first(C: Cone, a: Action) -> Tuple[int, int, int, Vec3]:
    ds = _dets(a)
    return (-a.smooth_child_count(), max(ds), sum(ds), a.b)


def score_balanced(C: Cone, a: Action) -> Tuple[int, int, int, Vec3]:
    ds = _dets(a)
    spread = max(ds) - min(ds)
    return (spread, max(ds), sum(ds), a.b)


def score_lex(C: Cone, a: Action) -> Tuple[int, int, int, int, Vec3]:
    """
    A strong default greedy rule:
      1. minimize largest child determinant,
      2. minimize sum of child determinants,
      3. minimize number of singular children,
      4. maximize number of smooth children.
    """
    ds = _dets(a)
    return (max(ds), sum(ds), a.singular_child_count(), -a.smooth_child_count(), a.b)


def score_random(C: Cone, a: Action) -> Tuple[float]:
    return (random.random(),)


STRATEGY_SCORES: Dict[str, Callable[[Cone, Action], Tuple]] = {
    "min_max": score_min_max,
    "min_sum": score_min_sum,
    "few_singular": score_few_singular,
    "smooth_first": score_smooth_first,
    "balanced": score_balanced,
    "lex": score_lex,
    "random": score_random,
}


def choose_action(C: Cone, strategy: str = "lex") -> Action:
    actions = legal_actions(C)
    if not actions:
        raise ValueError(f"No legal actions for {C}; det={C.det()}")
    if strategy not in STRATEGY_SCORES:
        raise KeyError(f"Unknown strategy {strategy!r}; choices: {sorted(STRATEGY_SCORES)}")
    score = STRATEGY_SCORES[strategy]
    return min(actions, key=lambda a: score(C, a))


def simulate_greedy(C: Cone, strategy: str = "lex", max_steps: int = 100000) -> Dict:
    """
    Resolve C by repeatedly applying a greedy strategy to each singular leaf.

    Returns a dictionary with total ray insertions and the maximum tree depth.
    """
    C = canonical_cone(C.matrix())
    stack: List[Tuple[Cone, int]] = [(C, 0)]
    steps = 0
    max_depth = 0
    leaves_seen = 0

    while stack:
        cone, depth = stack.pop()
        leaves_seen += 1
        if cone.is_smooth():
            max_depth = max(max_depth, depth)
            continue

        if steps >= max_steps:
            raise RuntimeError(f"Exceeded max_steps={max_steps}; possible strategy bug.")

        action = choose_action(cone, strategy)
        children = star_subdivide(cone, action)
        steps += 1
        for child in children:
            stack.append((child, depth + 1))

    return {
        "strategy": strategy,
        "start": C.as_tuple(),
        "start_det": C.det(),
        "ray_insertions": steps,
        "max_depth": max_depth,
        "processed_leaves": leaves_seen,
    }



# Exact dynamic programming strategies - got these from chatgpt pro



class ExactSolver:
    """
    Exact optimal solver for one cone.

    objective='total': minimize total number of inserted rays in the full
    subdivision tree:

        S(C) = 0 if det(C)=1,
        S(C) = 1 + min_a sum_i S(child_i).

    objective='depth': minimize parallel depth / worst branch:

        P(C) = 0 if det(C)=1,
        P(C) = 1 + min_a max_i P(child_i).

    This is feasible for many small examples because every child determinant
    is strictly smaller than the parent determinant.  It can still become
    expensive for large batches near det 100.
    """

    def __init__(self, objective: str = "total"):
        if objective not in {"total", "depth"}:
            raise ValueError("objective must be 'total' or 'depth'")
        self.objective = objective
        self.memo: Dict[Cone, int] = {}
        self.policy: Dict[Cone, Action] = {}

    def value(self, C: Cone) -> int:
        C = canonical_cone(C.matrix())
        if C.is_smooth():
            return 0
        if C in self.memo:
            return self.memo[C]

        best_value: Optional[int] = None
        best_action: Optional[Action] = None

        for a in legal_actions(C):
            children = star_subdivide(C, a)
            child_values = [self.value(child) for child in children]
            if self.objective == "total":
                val = 1 + sum(child_values)
            else:
                val = 1 + max(child_values)

            if best_value is None or val < best_value or (val == best_value and score_lex(C, a) < score_lex(C, best_action)):  # type: ignore[arg-type]
                best_value = val
                best_action = a

        if best_value is None or best_action is None:
            raise RuntimeError(f"No legal action found for singular cone {C}")

        self.memo[C] = best_value
        self.policy[C] = best_action
        return best_value

    def best_action(self, C: Cone) -> Action:
        C = canonical_cone(C.matrix())
        if C.is_smooth():
            raise ValueError("Smooth cone has no action.")
        self.value(C)
        return self.policy[C]

    def explain_first_move(self, C: Cone, top_k: int = 10) -> List[Dict]:
        """Return scored actions for the first move, sorted best first."""
        C = canonical_cone(C.matrix())
        rows: List[Dict] = []
        for a in legal_actions(C):
            children = star_subdivide(C, a)
            child_values = [self.value(child) for child in children]
            if self.objective == "total":
                val = 1 + sum(child_values)
            else:
                val = 1 + max(child_values)
            rows.append(
                {
                    "value": val,
                    "b": a.b,
                    "ray": a.ray,
                    "child_dets": a.child_dets(),
                    "child_values": tuple(child_values),
                    "children": [ch.as_tuple() for ch in children],
                }
            )
        rows.sort(key=lambda r: (r["value"], max(r["child_dets"]), sum(r["child_dets"]), r["b"]))
        return rows[:top_k]








class GreedyMemoSolver:
    """
    Memoized greedy resolver for batch experiments.

    For a fixed greedy strategy, this computes the same quantities as
    simulate_greedy(C, strategy), but shares work across cones.  This matters
    when running thousands of cones because many child cones repeat.
    """

    def __init__(self, strategy: str):
        if strategy not in STRATEGY_SCORES:
            raise KeyError(f"Unknown strategy {strategy!r}; choices: {sorted(STRATEGY_SCORES)}")
        self.strategy = strategy
        self.memo: Dict[Cone, Tuple[int, int]] = {}
        self.policy: Dict[Cone, Action] = {}

    def value(self, C: Cone) -> Tuple[int, int]:
        """Return (total_inserted_rays, parallel_depth) under this strategy."""
        C = canonical_cone(C.matrix())
        if C.is_smooth():
            return (0, 0)
        if C in self.memo:
            return self.memo[C]

        a = choose_action(C, self.strategy)
        children = star_subdivide(C, a)
        child_vals = [self.value(ch) for ch in children]
        total = 1 + sum(v[0] for v in child_vals)
        depth = 1 + max(v[1] for v in child_vals)

        self.memo[C] = (total, depth)
        self.policy[C] = a
        return total, depth


def summarize_numbers(values: Sequence[int]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def batch_greedy_analysis(
    cones: Sequence[Cone],
    strategies: Sequence[str],
    csv_path: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Run each greedy strategy on every cone in `cones`.

    Returns a nested summary dictionary.  If csv_path is given, writes one row
    per (cone, strategy) containing total inserted rays and depth.
    """
    summaries: Dict[str, Dict] = {}
    csv_rows: List[Dict[str, object]] = []

    for strategy in strategies:
        solver = GreedyMemoSolver(strategy)
        totals: List[int] = []
        depths: List[int] = []

        for C in cones:
            total, depth = solver.value(C)
            totals.append(total)
            depths.append(depth)
            if csv_path is not None:
                csv_rows.append(
                    {
                        "strategy": strategy,
                        "u": C.u,
                        "v": C.v,
                        "x": C.x,
                        "y": C.y,
                        "z": C.z,
                        "det": C.det(),
                        "ray_insertions": total,
                        "max_depth": depth,
                    }
                )

        summaries[strategy] = {
            "total_insertions": summarize_numbers(totals),
            "max_depth": summarize_numbers(depths),
            "memoized_states": len(solver.memo),
        }

    if csv_path is not None:
        import csv

        fieldnames = ["strategy", "u", "v", "x", "y", "z", "det", "ray_insertions", "max_depth"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    return summaries


def batch_exact_analysis(
    cones: Sequence[Cone],
    objective: str,
    csv_path: Optional[str] = None,
) -> Dict[str, object]:
    """
    Run exact DP on every cone in `cones` for one objective.

    objective='total' computes S(C); objective='depth' computes P(C).
    This can be expensive for large max_det values.
    """
    solver = ExactSolver(objective)
    values: List[int] = []
    csv_rows: List[Dict[str, object]] = []

    for C in cones:
        val = solver.value(C)
        values.append(val)
        if csv_path is not None:
            if C.is_smooth():
                b = ray = child_dets = None
            else:
                a = solver.best_action(C)
                b = a.b
                ray = a.ray
                child_dets = a.child_dets()
            csv_rows.append(
                {
                    "objective": objective,
                    "u": C.u,
                    "v": C.v,
                    "x": C.x,
                    "y": C.y,
                    "z": C.z,
                    "det": C.det(),
                    "value": val,
                    "best_b": b,
                    "best_ray": ray,
                    "best_child_dets": child_dets,
                }
            )

    if csv_path is not None:
        import csv

        fieldnames = [
            "objective",
            "u",
            "v",
            "x",
            "y",
            "z",
            "det",
            "value",
            "best_b",
            "best_ray",
            "best_child_dets",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    return {
        "objective": objective,
        "values": summarize_numbers(values),
        "memoized_states": len(solver.memo),
    }


def batch_compare_against_exact(
    cones: Sequence[Cone],
    objectives: Sequence[str],
    strategies: Sequence[str],
    csv_path: Optional[str] = None,
    worst_k: int = 5,
) -> Dict[str, object]:
    """
    Compare greedy strategies against exact optimal DP values.

    objective='total': exact value is minimum total inserted rays.
    objective='depth': exact value is minimum parallel/worst-branch depth.

    Returns machine-readable data and optionally writes one row per
    (objective, strategy, cone) to CSV.
    """
    report: Dict[str, object] = {
        "cone_count": len(cones),
        "objectives": {},
    }
    csv_rows: List[Dict[str, object]] = []

    for objective in objectives:
        if objective not in {"total", "depth"}:
            raise ValueError("objective must be 'total' or 'depth'")

        exact = ExactSolver(objective)
        exact_values: Dict[Cone, int] = {}
        exact_first: Dict[Cone, Optional[Action]] = {}
        vals: List[int] = []

        for C0 in cones:
            C = canonical_cone(C0.matrix())
            val = exact.value(C)
            exact_values[C] = val
            vals.append(val)
            exact_first[C] = None if C.is_smooth() else exact.best_action(C)

        objective_report: Dict[str, object] = {
            "exact_values": summarize_numbers(vals),
            "exact_memoized_states": len(exact.memo),
            "strategies": {},
        }

        for strategy in strategies:
            greedy = GreedyMemoSolver(strategy)
            greedy_vals: List[int] = []
            overheads: List[int] = []
            ratios: List[float] = []
            optimal_count = 0
            first_move_optimal_count = 0
            non_smooth_count = 0
            worst_records: List[Dict[str, object]] = []

            for C0 in cones:
                C = canonical_cone(C0.matrix())
                total, depth = greedy.value(C)
                gval = total if objective == "total" else depth
                oval = exact_values[C]
                overhead = gval - oval
                if overhead < 0:
                    raise AssertionError(
                        f"Greedy value below exact optimum: objective={objective}, "
                        f"strategy={strategy}, cone={C}, greedy={gval}, exact={oval}"
                    )
                if overhead == 0:
                    optimal_count += 1
                if oval > 0:
                    ratios.append(gval / oval)

                greedy_vals.append(gval)
                overheads.append(overhead)

                greedy_action: Optional[Action] = None if C.is_smooth() else choose_action(C, strategy)
                exact_action = exact_first[C]
                greedy_first_is_optimal = None
                if not C.is_smooth() and greedy_action is not None:
                    non_smooth_count += 1
                    children = star_subdivide(C, greedy_action)
                    child_vals = [exact.value(ch) for ch in children]
                    if objective == "total":
                        first_move_value = 1 + sum(child_vals)
                    else:
                        first_move_value = 1 + max(child_vals)
                    greedy_first_is_optimal = first_move_value == oval
                    if greedy_first_is_optimal:
                        first_move_optimal_count += 1

                record = {
                    "cone": C.as_tuple(),
                    "det": C.det(),
                    "exact": oval,
                    "greedy": gval,
                    "overhead": overhead,
                    "greedy_first_b": None if greedy_action is None else greedy_action.b,
                    "greedy_first_ray": None if greedy_action is None else greedy_action.ray,
                    "greedy_first_child_dets": None if greedy_action is None else greedy_action.child_dets(),
                    "exact_first_b": None if exact_action is None else exact_action.b,
                    "exact_first_ray": None if exact_action is None else exact_action.ray,
                    "exact_first_child_dets": None if exact_action is None else exact_action.child_dets(),
                    "greedy_first_is_optimal": greedy_first_is_optimal,
                }
                worst_records.append(record)

                if csv_path is not None:
                    csv_rows.append(
                        {
                            "objective": objective,
                            "strategy": strategy,
                            "u": C.u,
                            "v": C.v,
                            "x": C.x,
                            "y": C.y,
                            "z": C.z,
                            "det": C.det(),
                            "exact_value": oval,
                            "greedy_value": gval,
                            "overhead": overhead,
                            "greedy_first_b": None if greedy_action is None else greedy_action.b,
                            "greedy_first_ray": None if greedy_action is None else greedy_action.ray,
                            "greedy_first_child_dets": None if greedy_action is None else greedy_action.child_dets(),
                            "exact_first_b": None if exact_action is None else exact_action.b,
                            "exact_first_ray": None if exact_action is None else exact_action.ray,
                            "exact_first_child_dets": None if exact_action is None else exact_action.child_dets(),
                            "greedy_first_is_optimal": greedy_first_is_optimal,
                        }
                    )

            worst_records.sort(key=lambda r: (r["overhead"], r["greedy"], r["det"]), reverse=True)
            strat_report = {
                "greedy_values": summarize_numbers(greedy_vals),
                "overheads": summarize_numbers(overheads),
                "mean_ratio_non_smooth": sum(ratios) / len(ratios) if ratios else 1.0,
                "optimal_count": optimal_count,
                "optimal_rate": optimal_count / len(cones) if cones else 0.0,
                "first_move_optimal_count": first_move_optimal_count,
                "first_move_optimal_rate": first_move_optimal_count / non_smooth_count if non_smooth_count else 1.0,
                "greedy_memoized_states": len(greedy.memo),
                "worst_examples": worst_records[:worst_k],
            }
            objective_report["strategies"][strategy] = strat_report  # type: ignore[index]

        report["objectives"][objective] = objective_report  # type: ignore[index]

    if csv_path is not None:
        import csv

        fieldnames = [
            "objective",
            "strategy",
            "u",
            "v",
            "x",
            "y",
            "z",
            "det",
            "exact_value",
            "greedy_value",
            "overhead",
            "greedy_first_b",
            "greedy_first_ray",
            "greedy_first_child_dets",
            "exact_first_b",
            "exact_first_ray",
            "exact_first_child_dets",
            "greedy_first_is_optimal",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    return report


def print_greedy_report(max_det: int, unique: bool, cone_count: int, summary: Dict[str, Dict]) -> None:
    print(f"Greedy batch report: {cone_count} cones, max_det={max_det}, unique={unique}")
    print()
    rows_total = []
    rows_depth = []
    for strategy, data in sorted(summary.items()):
        t = data["total_insertions"]
        d = data["max_depth"]
        rows_total.append([strategy, t["mean"], t["max"], t["min"], data["memoized_states"]])
        rows_depth.append([strategy, d["mean"], d["max"], d["min"], data["memoized_states"]])

    print("Total inserted rays under each greedy strategy")
    print_table(["strategy", "mean", "max", "min", "memo states"], rows_total)
    print()
    print("Maximum branch depth under each greedy strategy")
    print_table(["strategy", "mean", "max", "min", "memo states"], rows_depth)


def print_exact_report(max_det: int, unique: bool, cone_count: int, summary: Dict[str, object]) -> None:
    values = summary["values"]  # type: ignore[index]
    print(f"Exact {summary['objective']} report: {cone_count} cones, max_det={max_det}, unique={unique}")
    print_table(
        ["count", "mean exact", "max exact", "min exact", "memo states"],
        [[values["count"], values["mean"], values["max"], values["min"], summary["memoized_states"]]],
    )


def print_compare_report(max_det: int, unique: bool, report: Dict[str, object], worst_k: int) -> None:
    cone_count = report["cone_count"]
    print(f"Exact-vs-greedy comparison: {cone_count} cones, max_det={max_det}, unique={unique}")
    print("Exact DP is a full search over all legal box rays; greedy rows show how far each strategy is from that optimum.")

    objectives: Dict[str, object] = report["objectives"]  # type: ignore[assignment]
    for objective in ("total", "depth"):
        if objective not in objectives:
            continue
        obj = objectives[objective]  # type: ignore[index]
        exact_vals = obj["exact_values"]  # type: ignore[index]
        title = "TOTAL INSERTED RAYS" if objective == "total" else "MAXIMUM BRANCH DEPTH"
        print()
        print(title)
        print(f"Exact optimum: mean={exact_vals['mean']:.3f}, max={exact_vals['max']}, memo states={obj['exact_memoized_states']}")

        rows = []
        strategies = obj["strategies"]  # type: ignore[index]
        for strategy, data in sorted(strategies.items(), key=lambda kv: (kv[1]["overheads"]["mean"], kv[0])):  # type: ignore[index]
            ov = data["overheads"]
            gv = data["greedy_values"]
            rows.append(
                [
                    strategy,
                    100.0 * data["optimal_rate"],
                    100.0 * data.get("first_move_optimal_rate", 0.0),
                    ov["mean"],
                    ov["max"],
                    gv["mean"],
                    gv["max"],
                    data["mean_ratio_non_smooth"],
                ]
            )
        print_table(
            ["strategy", "% opt value", "% opt first", "mean overhead", "max overhead", "mean greedy", "max greedy", "mean ratio"],
            rows,
        )

        if worst_k > 0:
            print()
            print(f"Worst {worst_k} examples by overhead")
            for strategy, data in sorted(strategies.items(), key=lambda kv: (kv[1]["overheads"]["max"], kv[1]["overheads"]["mean"]), reverse=True):  # type: ignore[index]
                examples = data["worst_examples"][:worst_k]
                if not examples or examples[0]["overhead"] == 0:
                    continue
                print(f"  {strategy}:")
                for ex in examples:
                    print(
                        "    "
                        f"cone=[{', '.join(map(str, ex['cone']))}] det={ex['det']} | "
                        f"exact={ex['exact']} greedy={ex['greedy']} overhead={ex['overhead']}"
                    )
                    print(
                        "      "
                        f"exact first: b={fmt_vec(ex['exact_first_b'] or [])} "
                        f"ray={fmt_vec(ex['exact_first_ray'] or [])} "
                        f"dets={fmt_vec(ex['exact_first_child_dets'] or [])}"
                    )
                    print(
                        "      "
                        f"greedy first: b={fmt_vec(ex['greedy_first_b'] or [])} "
                        f"ray={fmt_vec(ex['greedy_first_ray'] or [])} "
                        f"dets={fmt_vec(ex['greedy_first_child_dets'] or [])}"
                    )



#CLI


def cone_from_args(vals: Sequence[int]) -> Cone:
    if len(vals) != 5:
        raise ValueError("Cone must be given as five integers: u v x y z")
    C = Cone(*map(int, vals))
    C.check()
    return canonical_cone(C.matrix())


def fmt_vec(v: Sequence[int]) -> str:
    return "[" + ", ".join(str(x) for x in v) + "]"


def fmt_cone(C: Cone) -> str:
    return f"u={C.u}, v={C.v}, x={C.x}, y={C.y}, z={C.z}, det={C.det()}"


def fmt_matrix(M: Mat3) -> str:
    rows = ["[" + ", ".join(f"{x:>3d}" for x in row) + "]" for row in M]
    return "\n".join(rows)


def format_action(a: Action) -> str:
    return f"b={fmt_vec(a.b)} | ray={fmt_vec(a.ray)} | child dets={fmt_vec(a.child_dets())}"


def fmt_number(x: object) -> str:
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def print_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    str_rows = [[fmt_number(x) for x in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "  ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    sep = "  ".join("-" * widths[i] for i in range(len(headers)))
    print(line)
    print(sep)
    for row in str_rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def count_by_det(cones: Iterable[Cone]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for C in cones:
        counts[C.det()] = counts.get(C.det(), 0) + 1
    return dict(sorted(counts.items()))


def export_jsonl(path: str, cones: Iterable[Cone]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for C in cones:
            obj = {"u": C.u, "v": C.v, "x": C.x, "y": C.y, "z": C.z, "det": C.det()}
            f.write(json.dumps(obj, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="3D toric cone generator and subdivision strategy tester")
    parser.add_argument("--max-det", type=int, default=100, help="maximum determinant for generation")
    parser.add_argument("--unique", action="store_true", help="canonicalize/quotient by ray permutations; slower")
    parser.add_argument("--count", action="store_true", help="print counts by determinant")
    parser.add_argument("--export", type=str, default=None, help="export generated cones to JSONL")
    parser.add_argument("--cone", nargs=5, type=int, metavar=("u", "v", "x", "y", "z"), help="specific cone in HNF coordinates")
    parser.add_argument("--show-actions", action="store_true", help="show legal actions for --cone")
    parser.add_argument("--strategy", choices=sorted(STRATEGY_SCORES), default="lex", help="greedy strategy")
    parser.add_argument("--simulate", action="store_true", help="simulate greedy resolution for --cone")
    parser.add_argument("--exact-total", action="store_true", help="compute exact total-step optimum for --cone")
    parser.add_argument("--exact-depth", action="store_true", help="compute exact parallel-depth optimum for --cone")
    parser.add_argument("--top-k", type=int, default=10, help="number of actions to show")
    parser.add_argument("--seed", type=int, default=0, help="random seed for random strategy")
    parser.add_argument("--batch-greedy", action="store_true", help="run all requested greedy strategies on every generated cone")
    parser.add_argument("--strategies", nargs="+", choices=sorted(STRATEGY_SCORES), default=None, help="strategies for --batch-greedy; default is all non-random strategies")
    parser.add_argument("--batch-exact", choices=("total", "depth"), default=None, help="run exact DP on every generated cone for this objective; can be expensive")
    parser.add_argument("--compare-exact", "--batch-compare-exact", choices=("total", "depth", "both"), default=None, help="compare greedy strategies against exact DP optimum over every generated cone")
    parser.add_argument("--worst-k", type=int, default=3, help="number of worst examples to show in exact-vs-greedy reports")
    parser.add_argument("--json", action="store_true", help="print batch reports as JSON instead of readable tables")
    parser.add_argument("--batch-csv", type=str, default=None, help="optional CSV output path for batch runs")
    args = parser.parse_args()

    random.seed(args.seed)

    if args.count or args.export or args.batch_greedy or args.batch_exact or args.compare_exact:
        cones = generate_cones(args.max_det, unique=args.unique)
        if args.count:
            counts = count_by_det(cones)
            print(f"generated_cones={len(cones)} unique={args.unique} max_det={args.max_det}")
            for D, n in counts.items():
                print(f"det {D:3d}: {n}")
        if args.export:
            export_jsonl(args.export, cones)
            print(f"wrote {len(cones)} cones to {args.export}")
        if args.batch_greedy:
            strategies = args.strategies or [s for s in sorted(STRATEGY_SCORES) if s != "random"]
            summary = batch_greedy_analysis(cones, strategies, csv_path=args.batch_csv)
            obj = {"max_det": args.max_det, "unique": args.unique, "cone_count": len(cones), "greedy": summary}
            if args.json:
                print(json.dumps(obj, indent=2, sort_keys=True))
            else:
                print_greedy_report(args.max_det, args.unique, len(cones), summary)
            if args.batch_csv:
                print(f"wrote batch CSV to {args.batch_csv}")
        if args.batch_exact:
            summary = batch_exact_analysis(cones, args.batch_exact, csv_path=args.batch_csv)
            obj = {"max_det": args.max_det, "unique": args.unique, "cone_count": len(cones), "exact": summary}
            if args.json:
                print(json.dumps(obj, indent=2, sort_keys=True))
            else:
                print_exact_report(args.max_det, args.unique, len(cones), summary)
            if args.batch_csv:
                print(f"wrote batch CSV to {args.batch_csv}")
        if args.compare_exact:
            strategies = args.strategies or [s for s in sorted(STRATEGY_SCORES) if s != "random"]
            objectives = ["total", "depth"] if args.compare_exact == "both" else [args.compare_exact]
            report = batch_compare_against_exact(cones, objectives, strategies, csv_path=args.batch_csv, worst_k=args.worst_k)
            obj = {"max_det": args.max_det, "unique": args.unique, **report}
            if args.json:
                print(json.dumps(obj, indent=2, sort_keys=True))
            else:
                print_compare_report(args.max_det, args.unique, report, args.worst_k)
            if args.batch_csv:
                print(f"wrote comparison CSV to {args.batch_csv}")

    if args.cone is not None:
        C = cone_from_args(args.cone)
        print("Cone")
        print(f"  {fmt_cone(C)}")
        print("Matrix")
        print(fmt_matrix(C.matrix()))

        if args.show_actions:
            actions = legal_actions(C)
            print()
            print(f"Legal primitive box rays: {len(actions)}")
            rows = []
            for idx, a in enumerate(actions[: args.top_k], start=1):
                rows.append([idx, fmt_vec(a.b), fmt_vec(a.ray), fmt_vec(a.child_dets()), a.singular_child_count(), a.smooth_child_count()])
            print_table(["#", "b", "ray", "child dets", "singular kids", "smooth kids"], rows)

            print()
            print("Strategy first-move choices")
            rows = []
            for name in sorted(STRATEGY_SCORES):
                a = choose_action(C, name)
                rows.append([name, fmt_vec(a.b), fmt_vec(a.ray), fmt_vec(a.child_dets())])
            print_table(["strategy", "b", "ray", "child dets"], rows)

        if args.simulate:
            result = simulate_greedy(C, args.strategy)
            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                print()
                print(f"Greedy simulation: strategy={args.strategy}")
                print_table(["start det", "ray insertions", "max depth", "processed leaves"], [[result["start_det"], result["ray_insertions"], result["max_depth"], result["processed_leaves"]]])

        if args.exact_total:
            solver = ExactSolver("total")
            value = solver.value(C)
            best = solver.best_action(C)
            print()
            print(f"Exact total optimum: {value} inserted rays")
            print(f"Best first move: {format_action(best)}")
            rows = []
            for i, row in enumerate(solver.explain_first_move(C, top_k=args.top_k), start=1):
                rows.append([i, row["value"], fmt_vec(row["b"]), fmt_vec(row["ray"]), fmt_vec(row["child_dets"]), fmt_vec(row["child_values"])])
            print_table(["#", "future cost", "b", "ray", "child dets", "child exact costs"], rows)

        if args.exact_depth:
            solver = ExactSolver("depth")
            value = solver.value(C)
            best = solver.best_action(C)
            print()
            print(f"Exact depth optimum: {value} rounds")
            print(f"Best first move: {format_action(best)}")
            rows = []
            for i, row in enumerate(solver.explain_first_move(C, top_k=args.top_k), start=1):
                rows.append([i, row["value"], fmt_vec(row["b"]), fmt_vec(row["ray"]), fmt_vec(row["child_dets"]), fmt_vec(row["child_values"])])
            print_table(["#", "future depth", "b", "ray", "child dets", "child exact depths"], rows)


if __name__ == "__main__":
    main()
