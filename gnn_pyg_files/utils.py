import math
import random
import torch

Vector = tuple[int, ...]

# ===========================================================================================
#  CONE HELPER FUNCTIONS
# ===========================================================================================

## Makes an input vector primitive
def primitive(v: Vector) -> Vector:
    gcd = math.gcd(*v)

    if gcd == 0:
        raise ValueError("zero vector not allowed")

    return tuple(x // gcd for x in v)

def isPrimitiveNonzero(point: Vector) -> bool:
    return math.gcd(*point) == 1

## computes g = x*a + y*b using extended euclidean algo and returns the triple (g,x,y)
def extendedEuclid(a: int, b: int) -> tuple[int, int, int]:
    old_r, r = a, b
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

## gives exact integer determinant using the Bareiss algorithm
def intDet(M: tuple[Vector, ...]) -> int:
    M = [list(row) for row in M]
    n = len(M)
    sign, prev = 1, 1
    for k in range(n - 1):
        if M[k][k] == 0:  # swap with a non-zero row if needed
            for r in range(k + 1, n):
                if M[r][k] != 0:
                    M[r], M[k] = M[k], M[r]
                    sign *= -1
                    break
            else:
                return 0
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                M[i][j] = (M[i][j] * M[k][k] - M[i][k] * M[k][j]) // prev
            M[i][k] = 0
        prev = M[k][k]
    return sign * M[-1][-1]

## puts a matrix into our canonical form
def canonicalForm(M: list[list[int]]) -> list[list[int]]:
    n = len(M)
    ## triangularization step: for each column, reduce [x,y] -> [g,0] where g = gcd(x,y)
    for j in range(n):
        for i in range(j + 1, n):
            if M[i][j] == 0:
                continue
            g, a, b = extendedEuclid(M[j][j], M[i][j])
            rowJ = [a * M[j][col] + b * M[i][col] for col in range(n)]
            rowI = [-(M[i][j] // g) * M[j][col] + (M[j][j] // g) * M[i][col] for col in range(n)]
            M[j], M[i] = rowJ, rowI
        if M[j][j] < 0:
            M[j] = [-x for x in M[j]]

    ## reduce each column by the pivot so all entries are non-negative and < pivot
    for j in range(n):
        for i in range(j):
            quotient = M[i][j] // M[j][j]
            if quotient:
                M[i] = [M[i][col] - quotient * M[j][col] for col in range(n)]

    return M

# ===========================================================================================
#  GRAPH NEURAL NETWORK HELPER FUNCTIONS
# ===========================================================================================

## provides a boolean (Tensor) mask of valid actions for the current state. Returns in idx order. 
def validActionMask(data) -> torch.Tensor:
    ei = data['lattice', 'in', 'cone'].edge_index
    n = data['lattice'].num_nodes
    mask = torch.zeros(n, dtype=torch.bool, device=data['lattice'].x.device)
    mask[ei[0]] = True
    return mask

## returns a dictionary with key = node type, value = vector of length (# nodes of that type) where the entries are the batch that node belongs to. 
## also returns B, the number of batches.
def _batch_vectors(data) -> tuple[dict, int]:
    batch_dict, B = {}, 1
    for node_type in data.x_dict:
        store = data[node_type]
        b = getattr(store, "batch", None) ## vector with batch number of each node of node_type

        ## default to batch = 0 if no batch attribute
        if b is None:
            b = torch.zeros(store.num_nodes, dtype = torch.long, device = store.x.device)
        
        batch_dict[node_type] = b

        if b.numel():
            B = max(B, int(b.max().item()) + 1) ## cautious maximum, sets B to the maximum observed batch number so far

    return batch_dict, B

# ===========================================================================================
#  TRAINING LOOP HELPER FUNCTIONS
# ===========================================================================================

def generateRandomCone(n: int, d: int, numOps: int = None) -> list[tuple[int, ...]]:
    if numOps is None:
        numOps = n * n * 10
    
    retryTimes = 0

    while True:
        ## seed: identity with last row = (1, 0, ..., 0, d)
        ## det = d, all rows primitive since gcd(1, 0, ..., 0, d) = 1
        M = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(1)
                else:
                    row.append(0)
            M.append(row)
        M[n-1][0] = 1
        M[n-1][n-1] = d

        for _ in range(numOps):
            i, j = random.sample(range(n), 2)
            sign = random.choice([-1, 1])
            if random.random() < 0.5:
                ## row op: row[i] += ±1 * row[j]
                for k in range(n):
                    M[i][k] += sign * M[j][k]
            else:
                ## col op: col[i] += ±1 * col[j]
                for r in range(n):
                    M[r][i] += sign * M[r][j]

        ## retry if any row is non-primitive -- buildCone would primitivize it
        ## and change the effective determinant away from d
        allPrimitive = True
        for row in M:
            if math.gcd(*row) != 1:
                allPrimitive = False
                retryTimes += 1
                break

        if allPrimitive:
            result = []
            for row in M:
                result.append(tuple(row))
            print(retryTimes)
            return result

# Append to the bottom of the module (same file as the functions under test),
# or `from <module> import *`. Run: python <module>.py

# ===========================================================================================
#  TEST FUNCTIONS
# ===========================================================================================

def _test_primitive():
    assert primitive((-2, -4, 6)) == (-1, -2, 3)      # gcd 2, sign preserved
    assert primitive((3, 0, 0)) == (1, 0, 0)
    assert primitive((5, 7)) == (5, 7)                # already primitive
    try:
        primitive((0, 0)); assert False
    except ValueError:
        pass


def _test_isPrimitiveNonzero():
    assert isPrimitiveNonzero((1, 0, 0))
    assert isPrimitiveNonzero((2, 3))
    assert not isPrimitiveNonzero((2, 4))


def _test_extendedEuclid():
    for a, b in [(240, 46), (-12, 18), (7, 0), (0, 5), (13, 1)]:
        g, x, y = extendedEuclid(a, b)
        assert g == math.gcd(a, b)                    # canonical (nonneg) gcd
        assert a * x + b * y == g                     # Bezout identity


def _test_intDet():
    assert intDet(((1, 0, 0), (0, 1, 0), (0, 0, 1))) == 1
    assert intDet(((2, 1), (1, 3))) == 5
    assert intDet(((1, 2), (2, 4))) == 0              # singular
    # row swap path: sign tracked correctly
    assert intDet(((0, 1), (1, 0))) == -1


def _test_canonicalForm():
    # canonicalForm acts on ROWS: preserves the row lattice, hence |det|,
    # and returns an upper-triangular Hermite-style normal form.
    for M in ([[2, 1], [1, 3]], [[1, 0], [2, 5]], [[4, 7], [2, 6]]):
        before = abs(intDet([tuple(r) for r in M]))
        C = canonicalForm([row[:] for row in M])
        n = len(C)
        assert abs(intDet([tuple(r) for r in C])) == before   # |det| invariant
        for j in range(n):                                    # upper triangular
            for i in range(j + 1, n):
                assert C[i][j] == 0
            assert C[j][j] > 0                                # positive pivots
            for i in range(j):                                # reduced above pivot
                assert 0 <= C[i][j] < C[j][j]


def _test_generateRandomCone():
    # Seed is a cone of multiplicity d; only unimodular row/col ops are applied,
    # so mult(sigma) = |det A| = d is invariant and every ray stays primitive.
    random.seed(0)
    for n, d in [(2, 1), (3, 5), (4, 12)]:
        cone = generateRandomCone(n, d)
        assert len(cone) == n
        assert abs(intDet(cone)) == d                         # mult(sigma) = d
        assert all(isPrimitiveNonzero(r) for r in cone)       # primitive generators


def _test_gnn_helpers():
    try:
        import torch
        from torch_geometric.data import HeteroData
    except Exception as e:
        print(f"[skip] GNN helpers (torch/torch_geometric unavailable: {e})")
        return
    data = HeteroData()
    data['lattice'].x = torch.zeros((4, 1))                   # 4 lattice nodes
    data['cone'].x = torch.zeros((2, 1))
    # rays 0 and 2 lie in some cone -> valid actions
    data['lattice', 'in', 'cone'].edge_index = torch.tensor([[0, 2], [0, 1]])
    mask = validActionMask(data)
    assert mask.tolist() == [True, False, True, False]
    bd, B = _batch_vectors(data)
    assert B == 1 and set(bd) == {'lattice', 'cone'}
    assert bd['lattice'].tolist() == [0, 0, 0, 0]


def main():
    tests = [
        _test_primitive, _test_isPrimitiveNonzero, _test_extendedEuclid,
        _test_intDet, _test_canonicalForm, _test_generateRandomCone,
        _test_gnn_helpers,
    ]
    for t in tests:
        t()
        print(f"[ok] {t.__name__}")
    print("all passed")


if __name__ == "__main__":
    main()