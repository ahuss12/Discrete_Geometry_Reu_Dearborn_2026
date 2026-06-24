import math
from sympy import Matrix, Rational, linsolve
from dataclasses import dataclass
from functools import cached_property
from fractions import Fraction
from torch_geometric.data import HeteroData
import torch
import random

DIMENSION: int = 7

Vector = tuple[int, ...]

## Makes an input vector primitive
def primitive(v: Vector) -> Vector:
    gcd = math.gcd(*v)

    if gcd == 0:
        raise ValueError("zero vector not allowed")

    return tuple(x // gcd for x in v)

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


## Custom (rational, simplicial, full-dimensional) cone class
## n-dimensional cone with exactly n generators
@dataclass(frozen=True)
class Cone:
    rays: tuple[Vector, ...]

    @classmethod
    def buildCone(cls, rays) -> "Cone":
        n = len(rays)

        if not rays:
            raise ValueError("need at least one generator")
        
        rays = tuple(sorted(primitive(tuple(r)) for r in rays))

        if any(all(x == 0 for x in r) for r in rays):
            raise ValueError("generators cannot be zero vector")

        if any(len(r) != n for r in rays):
            raise ValueError("cone must be full dimensional")

        if Matrix(rays).rank() != n:
            raise ValueError("cone must be simplicial")

        return cls(rays)

    @property
    def dimension(self) -> int:
        return len(self.rays[0])

    @property
    def numGenerators(self) -> int:
        return len(self.rays)

    @property
    def isSingular(self) -> bool:
        return self.multiplicity != 1

    @cached_property
    def multiplicity(self) -> int:
        return abs(intDet(self.rays))

    def barycentricCoords(self, p: Vector) -> tuple[Fraction, ...]:
        A = Matrix(self.rays).T
        b = Matrix(p)
        (coords,) = linsolve((A, b))
        coords = tuple(Fraction(c.p, c.q) for c in coords)

        if any(x < 0 for x in coords):
            raise ValueError("point p must be in the cone")

        return coords

    def contains(self, p: Vector) -> bool:
        A = Matrix(self.rays).T
        b = Matrix(p)
        (coords,) = linsolve((A, b))
        coords = tuple(coords)

        if any(x < 0 for x in coords):
            return False

        return True

    def extraneousSet(self) -> list[Vector]:
        n = len(self.rays)
        A = [[r[j] for r in self.rays] for j in range(n)]
        H = canonicalForm([row[:] for row in A])

        lambdas = [[Fraction(i, H[-1][-1])] for i in range(H[-1][-1])]

        for i in reversed(range(n - 1)):
            newLambdas = []
            for curr in lambdas:
                s = sum(curr[col - (i + 1)] * H[i][col] for col in range(i + 1, n))
                for k in range(H[i][i]):
                    newLambdas.append([Fraction(math.ceil(s) - s + k, H[i][i])] + curr)
            lambdas = newLambdas

        lambdas = [
            tuple(int(math.sumprod(A[i], vec)) for i in range(n))
            for vec in lambdas
        ]

        return lambdas

    def HNF(self) -> "Cone":
        n = len(self.rays)
        H = [[r[j] for r in self.rays] for j in range(n)]
        H = canonicalForm(H)
        self.rays = tuple(tuple(H[i][j] for i in range(n)) for j in range(n))

    def subdivide(self, p: Vector) -> list["Cone"]:
        p = primitive(p)
        coords = self.barycentricCoords(p)

        if sum(1 for c in coords if c > 0) == 1:
            raise ValueError("subdividing point cannot lie on a generating ray")

        fan = []
        for i in range(len(coords)):
            if coords[i] > 0:
                generators = list(self.rays)
                generators[i] = p
                fan.append(Cone.buildCone(generators))
        return fan

def conesAdjacent(coneA: "Cone", coneB: "Cone") -> bool:
    shared = set(coneA.rays) & set(coneB.rays)
    return len(shared) >= 2


def computeConeFeatures(cone: "Cone") -> torch.Tensor:
    sorted_rays = sorted(cone.rays)                     
    flat: list[float] = [float(x) for ray in sorted_rays for x in ray]
    flat.append(float(cone.multiplicity))
    return torch.tensor(flat, dtype=torch.float)


def computeLatticeFeatures(coord: tuple) -> torch.Tensor:
    return torch.tensor([float(x) for x in coord], dtype=torch.float)

def isPrimitiveNonzero(point: Vector) -> bool:
    return math.gcd(*point) == 1

## Heterogeneous graph over a fan
class ConeLatticeGraph(HeteroData):

    def __init__(self, dimension: int = DIMENSION):
        super().__init__()

        self._dimension: int = dimension

        # Feature shapes:
        #   cone    -> (n² + 1,)  [flattened sorted rays + multiplicity]
        #   lattice -> (n,)       [coordinate vector]
        cone_feat_dim: int    = dimension * dimension + 1
        lattice_feat_dim: int = dimension

        # cone nodes
        self._cone_id_to_idx: dict[int, int]  = {}
        self._cone_idx_to_id: dict[int, int]  = {}
        self._next_cone_id: int               = 0
        self._cone_objects: dict[int, "Cone"] = {}

        # lattice nodes, keyed globally by coordinate to avoid duplicates
        self._coord_to_lattice_id: dict[tuple, int] = {}
        self._lattice_id_to_idx: dict[int, int]     = {}
        self._lattice_idx_to_id: dict[int, int]     = {}
        self._lattice_id_to_coord: dict[int, tuple] = {}
        self._next_lattice_id: int                  = 0

        # Initialise feature matrices with the correct column count (zero rows)
        self['cone'].x    = torch.empty((0, cone_feat_dim),    dtype=torch.float)
        self['lattice'].x = torch.empty((0, lattice_feat_dim), dtype=torch.float)

        self['cone',    'adjacent', 'cone'   ].edge_index = torch.empty((2, 0), dtype=torch.long)
        self['cone',    'contains', 'lattice'].edge_index = torch.empty((2, 0), dtype=torch.long)
        self['lattice', 'contains', 'cone'   ].edge_index = torch.empty((2, 0), dtype=torch.long)

    def listCones(self, verbose: bool = False) -> list[tuple]:
        cones = []
        for cid in sorted(self._cone_id_to_idx):
            cone = self._cone_objects[cid]
            if verbose:
                print(f"Cone {cid}: rays={cone.rays}  mult={cone.multiplicity}")
            cones.append(cone.rays)
        return cones

    def listConeLatticePoints(self, coneId: int, verbose: bool = False) -> list[tuple]:
        coneIdx = self._cone_id_to_idx[coneId]
        ei = self['cone', 'contains', 'lattice'].edge_index
        latticeIds = [self._lattice_idx_to_id[i] for i in ei[1][ei[0] == coneIdx].tolist()]
        points = []
        for lid in sorted(latticeIds):
            coord = self._lattice_id_to_coord[lid]
            if verbose:
                print(f"  Lattice {lid}: coord={coord}")
            points.append(coord)
        return points
    
    def listLatticePoints(self, verbose: bool = False) -> list[tuple[int, tuple]]:
        points = []

        for latticeId in sorted(self._lattice_id_to_idx):
            coord = self._lattice_id_to_coord[latticeId]

            if verbose:
                print(f"Lattice {latticeId}: coord={coord}")

            points.append((latticeId, coord))

        return points

    def addAdjacentEdge(self, coneIdA: int, coneIdB: int) -> None:
        idxA = self._cone_id_to_idx[coneIdA]
        idxB = self._cone_id_to_idx[coneIdB]
        existing = self['cone', 'adjacent', 'cone'].edge_index
        newEdges = torch.tensor([[idxA, idxB], [idxB, idxA]], dtype=torch.long)
        self['cone', 'adjacent', 'cone'].edge_index = torch.cat([existing, newEdges], dim=1)

    def removeAllAdjacentEdges(self, coneId: int) -> None:
        idx = self._cone_id_to_idx[coneId]
        edgeIndex = self['cone', 'adjacent', 'cone'].edge_index
        keepMask = (edgeIndex[0] != idx) & (edgeIndex[1] != idx)
        self['cone', 'adjacent', 'cone'].edge_index = edgeIndex[:, keepMask]

    def getConeNeighbors(self, coneId: int) -> list[int]:
        idx = self._cone_id_to_idx[coneId]
        edgeIndex = self['cone', 'adjacent', 'cone'].edge_index
        mask = edgeIndex[0] == idx
        neighborIdxs = edgeIndex[1][mask].tolist()
        return [self._cone_idx_to_id[i] for i in neighborIdxs]

    def addContainsEdge(self, coneId: int, latticeId: int, barycentric: tuple[Fraction, ...]) -> None:
        coneIdx = self._cone_id_to_idx[coneId]
        latticeIdx = self._lattice_id_to_idx[latticeId]

        attr = torch.tensor([[float(x) for x in barycentric]], dtype=torch.float)

        forward_store = self['cone', 'contains', 'lattice']
        forward_store.edge_index = torch.cat([
                forward_store.edge_index,
                torch.tensor([[coneIdx], [latticeIdx]], dtype=torch.long)
            ],
            dim=1
        )
        forward_store.edge_attr = torch.cat(
            [forward_store.edge_attr, attr],
            dim=0
        )

        backward_store = self['lattice', 'contains', 'cone']
        backward_store.edge_index = torch.cat([
                backward_store.edge_index,
                torch.tensor([[latticeIdx], [coneIdx]], dtype=torch.long)
            ],
            dim=1
        )
        backward_store.edge_attr = torch.cat(
            [backward_store.edge_attr, attr],
            dim=0
        )

    def getOrCreateLatticeNode(self, coord: tuple) -> int:
        if coord in self._coord_to_lattice_id:
            return self._coord_to_lattice_id[coord]

        latticeId = self._next_lattice_id
        self._next_lattice_id += 1

        idx = self['lattice'].x.size(0)
        self._coord_to_lattice_id[coord]    = latticeId
        self._lattice_id_to_idx[latticeId]  = idx
        self._lattice_idx_to_id[idx]        = latticeId
        self._lattice_id_to_coord[latticeId] = coord

        #  Real lattice features (coordinate vector) 
        featureRow = computeLatticeFeatures(coord).unsqueeze(0)
        self['lattice'].x = torch.cat([self['lattice'].x, featureRow], dim=0)

        return latticeId
    
    def addConeCandidateEdges(self, coneId: int, cone: Cone) -> None:
        for point in cone.extraneousSet():
            if isPrimitiveNonzero(point):
                latticeId = self.getOrCreateLatticeNode(point)
                barycentric = cone.barycentricCoords(point)
                self.addContainsEdge(coneId, latticeId, barycentric)

    def addConeNode(self, cone: "Cone") -> int:
        coneId = self._next_cone_id
        self._next_cone_id += 1

        idx = self['cone'].x.size(0)
        self._cone_id_to_idx[coneId]  = idx
        self._cone_idx_to_id[idx]     = coneId
        self._cone_objects[coneId]    = cone

        #  Real cone features (sorted-flattened rays + multiplicity) 
        featureRow = computeConeFeatures(cone).unsqueeze(0)
        self['cone'].x = torch.cat([self['cone'].x, featureRow], dim=0)

        self.addConeCandidateEdges(coneId, cone)
        return coneId

    def removeAllContainsEdges(self, coneId: int) -> None:
        coneIdx = self._cone_id_to_idx[coneId]
        fwd = self['cone', 'contains', 'lattice'].edge_index
        self['cone', 'contains', 'lattice'].edge_index = fwd[:, fwd[0] != coneIdx]
        bwd = self['lattice', 'contains', 'cone'].edge_index
        self['lattice', 'contains', 'cone'].edge_index = bwd[:, bwd[1] != coneIdx]

    def overwriteConeNode(self, coneId: int, cone: "Cone") -> None:
        self.removeAllContainsEdges(coneId)
        idx = self._cone_id_to_idx[coneId]
        self._cone_objects[coneId] = cone

        self['cone'].x[idx] = computeConeFeatures(cone)

        self.addConeCandidateEdges(coneId, cone)

    def subdivide(self, coneId: int, latticeId: int) -> list[int]:
        oldCone = self._cone_objects[coneId]
        coord   = self._lattice_id_to_coord[latticeId]

        results = oldCone.subdivide(coord)
        if not results:
            raise ValueError("subdivide produced no resulting cones")

        formerNeighbors = self.getConeNeighbors(coneId)
        self.removeAllAdjacentEdges(coneId)

        # results[0] overwrites the existing row/id; results[1:] get new ids
        self.overwriteConeNode(coneId, results[0])
        newConeIds = [coneId]
        for cone in results[1:]:
            newConeIds.append(self.addConeNode(cone))

        # every pair of new cones is mutually adjacent (they all share p)
        for i in range(len(newConeIds)):
            for j in range(i + 1, len(newConeIds)):
                self.addAdjacentEdge(newConeIds[i], newConeIds[j])

        # re-check each former neighbour against each new cone
        for neighborId in formerNeighbors:
            neighborCone = self._cone_objects[neighborId]
            for newId in newConeIds:
                newCone = self._cone_objects[newId]
                if conesAdjacent(neighborCone, newCone):
                    self.addAdjacentEdge(neighborId, newId)

        return newConeIds
    
    def getLatticeConeNeighbors(self, latticeId: int) -> list[int]:
        if latticeId not in self._lattice_id_to_idx:
            raise KeyError(f"Unknown lattice ID: {latticeId}")

        latticeIdx = self._lattice_id_to_idx[latticeId]

        # lattice --contains--> cone
        edgeIndex = self['lattice', 'contains', 'cone'].edge_index

        mask = edgeIndex[0] == latticeIdx
        coneIdxs = edgeIndex[1][mask].tolist()

        # `dict.fromkeys` removes duplicates while preserving order.
        return list(dict.fromkeys(
            self._cone_idx_to_id[coneIdx]
            for coneIdx in coneIdxs
        ))

    def printAdjacencyList(self) -> None:
        print("=== Cone adjacency ===")
        edge_index = self['cone', 'adjacent', 'cone'].edge_index
        cone_adj = {cid: [] for cid in self._cone_id_to_idx.keys()}
        for src, dst in edge_index.t().tolist():
            src_id = self._cone_idx_to_id[src]
            dst_id = self._cone_idx_to_id[dst]
            cone_adj[src_id].append(dst_id)
        for cid in sorted(cone_adj):
            print(f"Cone {cid}: {sorted(cone_adj[cid])}")

        print("\n=== Cone -> Lattice containment ===")
        edge_index = self['cone', 'contains', 'lattice'].edge_index
        cone_contains = {cid: [] for cid in self._cone_id_to_idx.keys()}
        for src, dst in edge_index.t().tolist():
            cone_id    = self._cone_idx_to_id[src]
            lattice_id = self._lattice_idx_to_id[dst]
            cone_contains[cone_id].append(lattice_id)
        for cid in sorted(cone_contains):
            print(f"Cone {cid}: {sorted(cone_contains[cid])}")

        print("\n=== Lattice -> Cone containment ===")
        edge_index = self['lattice', 'contains', 'cone'].edge_index
        lattice_contains = {lid: [] for lid in self._lattice_id_to_idx.keys()}
        for src, dst in edge_index.t().tolist():
            lattice_id = self._lattice_idx_to_id[src]
            cone_id    = self._cone_idx_to_id[dst]
            lattice_contains[lattice_id].append(cone_id)
        for lid in sorted(lattice_contains):
            print(f"Lattice {lid}: {sorted(lattice_contains[lid])}")

    def toHeteroData(self) -> HeteroData:
        data = HeteroData()
        data['cone'].x    = self['cone'].x.clone()
        data['lattice'].x = self['lattice'].x.clone()
        data['cone',    'adjacent', 'cone'   ].edge_index = self['cone',    'adjacent', 'cone'   ].edge_index.clone()
        data['cone',    'contains', 'lattice'].edge_index = self['cone',    'contains', 'lattice'].edge_index.clone()
        data['lattice', 'contains', 'cone'   ].edge_index = self['lattice', 'contains', 'cone'   ].edge_index.clone()
        return data
    
    def isDecomposed(self) -> bool:
        for cone in self._cone_objects.values():
            if cone.isSingular:
                return False
        return True

def generateRandomCone(n: int, d: int, numOps: int = None) -> list[tuple[int, ...]]:
    if numOps is None:
        numOps = n * n * 10

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
                break

        if allPrimitive:
            result = []
            for row in M:
                result.append(tuple(row))
            return result

def main():
    # # This demo uses 4-D cones, so we pass dimension=4 explicitly.
    # # For your real work, omit the argument and the DIMENSION=7 default applies.
    # c = Cone.buildCone([[1, 0, 0, 0], [1, 2, 0, 0], [1, 2, 1, 0], [0, 1, 2, 3]])
    # fpp = c.extraneousSet()
    # print(fpp)
    # for i in range(len(fpp)):
    #     print(c.barycentricCoords(fpp[i]))

    # CLG = ConeLatticeGraph(dimension=4)
    # CLG.addConeNode(c)

    # CLG.printAdjacencyList()

    # CLG.subdivide(0, 2)

    # print()
    # CLG.printAdjacencyList()

    # CLG.listCones(True)
    # CLG.listConeLatticePoints(1, True)

    # # Demonstrate that toHeteroData() now returns real features
    # data = CLG.toHeteroData()
    # print(f"\ncone feature shape:    {data['cone'].x.shape}")     # (num_cones, 4²+1 = 17)
    # print(f"lattice feature shape: {data['lattice'].x.shape}")    # (num_lattice, 4)
    # print(f"\nFirst cone features:\n{data['cone'].x[0]}")

    M = Matrix(generateRandomCone(4, 12, 20))
    print(M)
    print(M.det())


main()