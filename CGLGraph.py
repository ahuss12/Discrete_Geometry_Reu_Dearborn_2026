import math
from sympy import Matrix, linsolve
from dataclasses import dataclass
from functools import cached_property
from fractions import Fraction
from torch_geometric.data import HeteroData
import torch

DIMENSION: int = 4  ## set to 4 for testing, 7 in final

Vector = tuple[int, ...]

def primitive(v: Vector) -> Vector:
    gcd = math.gcd(*v)
    if gcd == 0:
        raise ValueError("zero vector not allowed")
    return tuple(x // gcd for x in v)

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

def intDet(M: tuple[Vector, ...]) -> int:
    M = [list(row) for row in M]
    n = len(M)
    sign, prev = 1, 1
    for k in range(n - 1):
        if M[k][k] == 0:
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

def canonicalForm(M: list[list[int]]) -> list[list[int]]:
    n = len(M)
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
    for j in range(n):
        for i in range(j):
            quotient = M[i][j] // M[j][j]
            if quotient:
                M[i] = [M[i][col] - quotient * M[j][col] for col in range(n)]
    return M


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
        if any(x < 0 for x in tuple(coords)):
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

    ## minimal change: wrap buildCone in try/except to silently skip degenerate
    ## (zero-determinant) cones that arise during stellar subdivision
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
                try:
                    fan.append(Cone.buildCone(generators))
                except ValueError:
                    pass  ## degenerate cone, skip
        return fan


def isPrimitiveNonzero(point: Vector) -> bool:
    return math.gcd(*point) == 1


## Heterogeneous graph over a fan with three node types:
##   cone      - feature: multiplicity repeated n times, zero-padded to DIMENSION
##   generator - feature: coordinate vector, zero-padded to DIMENSION
##   lattice   - feature: coordinate vector, zero-padded to DIMENSION
##
## Edge types:
##   (cone, has, generator)         - cone uses this ray as a generator
##   (generator, of, cone)          - reverse
##   (cone, contains, lattice)      - lattice point is in cone's FPP
##   (lattice, in, cone)            - reverse
##   (generator, reaches, lattice)  - with edge_attr: scalar barycentric coord λ_i
##   (lattice, reached_by, generator) - reverse, same edge_attr
##   (cone, adjacent, cone)         - share >= 2 generator nodes (both directions)
##
## Generator and lattice nodes are globally unique by coordinate.
## Generator-lattice edges are permanent once created.
class CGLGraph(HeteroData):

    def __init__(self):
        super().__init__()

        ## cone bookkeeping
        self._cone_id_to_idx: dict[int, int]  = {}
        self._cone_idx_to_id: dict[int, int]  = {}
        self._next_cone_id: int               = 0
        self._cone_objects: dict[int, Cone]   = {}

        ## generator bookkeeping
        self._coord_to_gen_id: dict[tuple, int] = {}
        self._gen_id_to_idx: dict[int, int]     = {}
        self._gen_idx_to_id: dict[int, int]     = {}
        self._gen_id_to_coord: dict[int, tuple] = {}
        self._next_gen_id: int                  = 0

        ## lattice bookkeeping
        self._coord_to_lattice_id: dict[tuple, int] = {}
        self._lattice_id_to_idx: dict[int, int]     = {}
        self._lattice_idx_to_id: dict[int, int]     = {}
        self._lattice_id_to_coord: dict[int, tuple] = {}
        self._next_lattice_id: int                  = 0

        ## tracks existing (genId, latticeId) pairs to prevent duplicate gen-lattice edges
        self._gen_lattice_edge_set: set[tuple[int, int]] = set()

        ## all node feature vectors are shape (DIMENSION,)
        self['cone'].x      = torch.empty((0, DIMENSION), dtype=torch.float)
        self['generator'].x = torch.empty((0, DIMENSION), dtype=torch.float)
        self['lattice'].x   = torch.empty((0, DIMENSION), dtype=torch.float)

        self['cone',      'has',        'generator'].edge_index = torch.empty((2, 0), dtype=torch.long)
        self['generator', 'of',         'cone'     ].edge_index = torch.empty((2, 0), dtype=torch.long)
        self['cone',      'contains',   'lattice'  ].edge_index = torch.empty((2, 0), dtype=torch.long)
        self['lattice',   'in',         'cone'     ].edge_index = torch.empty((2, 0), dtype=torch.long)
        self['generator', 'reaches',    'lattice'  ].edge_index = torch.empty((2, 0), dtype=torch.long)
        self['generator', 'reaches',    'lattice'  ].edge_attr  = torch.empty((0, 1), dtype=torch.float)
        self['lattice',   'reached_by', 'generator'].edge_index = torch.empty((2, 0), dtype=torch.long)
        self['lattice',   'reached_by', 'generator'].edge_attr  = torch.empty((0, 1), dtype=torch.float)
        self['cone',      'adjacent',   'cone'     ].edge_index = torch.empty((2, 0), dtype=torch.long)

    ## --- feature computation ---

    def computeConeFeature(self, cone: Cone) -> torch.Tensor:
        n    = cone.dimension
        mult = float(cone.multiplicity)
        feat = []
        for _ in range(n):
            feat.append(mult)
        while len(feat) < DIMENSION:
            feat.append(0.0)
        return torch.tensor(feat, dtype=torch.float)

    def computeVectorFeature(self, coord: tuple) -> torch.Tensor:
        feat = []
        for x in coord:
            feat.append(float(x))
        while len(feat) < DIMENSION:
            feat.append(0.0)
        return torch.tensor(feat, dtype=torch.float)

    ## --- node creation ---

    def getOrCreateGeneratorNode(self, coord: tuple) -> int:
        if coord in self._coord_to_gen_id:
            return self._coord_to_gen_id[coord]

        genId = self._next_gen_id
        self._next_gen_id += 1

        idx = self['generator'].x.size(0)
        self._coord_to_gen_id[coord] = genId
        self._gen_id_to_idx[genId]   = idx
        self._gen_idx_to_id[idx]     = genId
        self._gen_id_to_coord[genId] = coord

        featureRow = self.computeVectorFeature(coord).unsqueeze(0)
        self['generator'].x = torch.cat([self['generator'].x, featureRow], dim=0)

        return genId

    def getOrCreateLatticeNode(self, coord: tuple) -> int:
        if coord in self._coord_to_lattice_id:
            return self._coord_to_lattice_id[coord]

        latticeId = self._next_lattice_id
        self._next_lattice_id += 1

        idx = self['lattice'].x.size(0)
        self._coord_to_lattice_id[coord]     = latticeId
        self._lattice_id_to_idx[latticeId]   = idx
        self._lattice_idx_to_id[idx]         = latticeId
        self._lattice_id_to_coord[latticeId] = coord

        featureRow = self.computeVectorFeature(coord).unsqueeze(0)
        self['lattice'].x = torch.cat([self['lattice'].x, featureRow], dim=0)

        return latticeId

    ## --- edge addition ---

    def addConeGeneratorEdge(self, coneId: int, genId: int) -> None:
        coneIdx = self._cone_id_to_idx[coneId]
        genIdx  = self._gen_id_to_idx[genId]

        ei  = self['cone', 'has', 'generator'].edge_index
        if ((ei[0] == coneIdx) & (ei[1] == genIdx)).any():
            return

        newFwd = torch.tensor([[coneIdx], [genIdx]], dtype=torch.long)
        self['cone', 'has', 'generator'].edge_index = torch.cat([ei, newFwd], dim=1)

        ei     = self['generator', 'of', 'cone'].edge_index
        newBwd = torch.tensor([[genIdx], [coneIdx]], dtype=torch.long)
        self['generator', 'of', 'cone'].edge_index = torch.cat([ei, newBwd], dim=1)

    def addConeLatticeEdge(self, coneId: int, latticeId: int) -> None:
        coneIdx    = self._cone_id_to_idx[coneId]
        latticeIdx = self._lattice_id_to_idx[latticeId]

        ei  = self['cone', 'contains', 'lattice'].edge_index
        if ((ei[0] == coneIdx) & (ei[1] == latticeIdx)).any():
            return

        newFwd = torch.tensor([[coneIdx], [latticeIdx]], dtype=torch.long)
        self['cone', 'contains', 'lattice'].edge_index = torch.cat([ei, newFwd], dim=1)

        ei     = self['lattice', 'in', 'cone'].edge_index
        newBwd = torch.tensor([[latticeIdx], [coneIdx]], dtype=torch.long)
        self['lattice', 'in', 'cone'].edge_index = torch.cat([ei, newBwd], dim=1)

    def addGeneratorLatticeEdge(self, genId: int, latticeId: int, baryCoord: float) -> None:
        if (genId, latticeId) in self._gen_lattice_edge_set:
            return

        genIdx     = self._gen_id_to_idx[genId]
        latticeIdx = self._lattice_id_to_idx[latticeId]
        attr       = torch.tensor([[baryCoord]], dtype=torch.float)

        ei = self['generator', 'reaches', 'lattice'].edge_index
        self['generator', 'reaches', 'lattice'].edge_index = torch.cat(
            [ei, torch.tensor([[genIdx], [latticeIdx]], dtype=torch.long)], dim=1)
        self['generator', 'reaches', 'lattice'].edge_attr = torch.cat(
            [self['generator', 'reaches', 'lattice'].edge_attr, attr], dim=0)

        ei = self['lattice', 'reached_by', 'generator'].edge_index
        self['lattice', 'reached_by', 'generator'].edge_index = torch.cat(
            [ei, torch.tensor([[latticeIdx], [genIdx]], dtype=torch.long)], dim=1)
        self['lattice', 'reached_by', 'generator'].edge_attr = torch.cat(
            [self['lattice', 'reached_by', 'generator'].edge_attr, attr], dim=0)

        self._gen_lattice_edge_set.add((genId, latticeId))

    def addAdjacentEdge(self, coneIdA: int, coneIdB: int) -> None:
        idxA = self._cone_id_to_idx[coneIdA]
        idxB = self._cone_id_to_idx[coneIdB]
        ei   = self['cone', 'adjacent', 'cone'].edge_index
        if ((ei[0] == idxA) & (ei[1] == idxB)).any():
            return
        newEdges = torch.tensor([[idxA, idxB], [idxB, idxA]], dtype=torch.long)
        self['cone', 'adjacent', 'cone'].edge_index = torch.cat([ei, newEdges], dim=1)

    ## --- edge removal ---

    def removeAllConeGeneratorEdges(self, coneId: int) -> None:
        coneIdx = self._cone_id_to_idx[coneId]

        ei   = self['cone', 'has', 'generator'].edge_index
        self['cone', 'has', 'generator'].edge_index = ei[:, ei[0] != coneIdx]

        ei   = self['generator', 'of', 'cone'].edge_index
        self['generator', 'of', 'cone'].edge_index = ei[:, ei[1] != coneIdx]

    def removeAllConeLatticeEdges(self, coneId: int) -> None:
        coneIdx = self._cone_id_to_idx[coneId]

        ei   = self['cone', 'contains', 'lattice'].edge_index
        self['cone', 'contains', 'lattice'].edge_index = ei[:, ei[0] != coneIdx]

        ei   = self['lattice', 'in', 'cone'].edge_index
        self['lattice', 'in', 'cone'].edge_index = ei[:, ei[1] != coneIdx]

    def removeAllAdjacentEdges(self, coneId: int) -> None:
        idx  = self._cone_id_to_idx[coneId]
        ei   = self['cone', 'adjacent', 'cone'].edge_index
        keep = (ei[0] != idx) & (ei[1] != idx)
        self['cone', 'adjacent', 'cone'].edge_index = ei[:, keep]

    ## --- cone wiring (wireConeGenerators MUST be called before wireConeLattice) ---

    def wireConeGenerators(self, coneId: int, cone: Cone) -> None:
        for ray in cone.rays:
            genId = self.getOrCreateGeneratorNode(ray)
            self.addConeGeneratorEdge(coneId, genId)

    def wireConeLattice(self, coneId: int, cone: Cone) -> None:
        for point in cone.extraneousSet():
            if not isPrimitiveNonzero(point):
                continue
            latticeId   = self.getOrCreateLatticeNode(point)
            self.addConeLatticeEdge(coneId, latticeId)
            barycentric = cone.barycentricCoords(point)
            for i in range(len(cone.rays)):
                if barycentric[i] > 0:
                    genId = self._coord_to_gen_id[cone.rays[i]]
                    self.addGeneratorLatticeEdge(genId, latticeId, float(barycentric[i]))

    ## --- cone node operations ---

    def addConeNode(self, cone: Cone) -> int:
        coneId = self._next_cone_id
        self._next_cone_id += 1

        idx = self['cone'].x.size(0)
        self._cone_id_to_idx[coneId] = idx
        self._cone_idx_to_id[idx]    = coneId
        self._cone_objects[coneId]   = cone

        featureRow = self.computeConeFeature(cone).unsqueeze(0)
        self['cone'].x = torch.cat([self['cone'].x, featureRow], dim=0)

        self.wireConeGenerators(coneId, cone)  ## must precede wireConeLattice
        self.wireConeLattice(coneId, cone)

        return coneId

    def overwriteConeNode(self, coneId: int, cone: Cone) -> None:
        self.removeAllConeGeneratorEdges(coneId)
        self.removeAllConeLatticeEdges(coneId)

        idx = self._cone_id_to_idx[coneId]
        self._cone_objects[coneId] = cone
        self['cone'].x[idx] = self.computeConeFeature(cone)

        self.wireConeGenerators(coneId, cone)  ## must precede wireConeLattice
        self.wireConeLattice(coneId, cone)

    ## --- adjacency queries ---

    def getConeNeighbors(self, coneId: int) -> list[int]:
        idx  = self._cone_id_to_idx[coneId]
        ei   = self['cone', 'adjacent', 'cone'].edge_index
        mask = ei[0] == idx
        return [self._cone_idx_to_id[i] for i in ei[1][mask].tolist()]

    def getConeGeneratorIds(self, coneId: int) -> set[int]:
        coneIdx = self._cone_id_to_idx[coneId]
        ei      = self['cone', 'has', 'generator'].edge_index
        mask    = ei[0] == coneIdx
        return set(self._gen_idx_to_id[i] for i in ei[1][mask].tolist())

    def conesAdjacentByGraph(self, coneIdA: int, coneIdB: int) -> bool:
        return len(self.getConeGeneratorIds(coneIdA) & self.getConeGeneratorIds(coneIdB)) >= 2

    ## --- subdivide ---

    def subdivide(self, latticeId: int) -> list[int]:
        coord = self._lattice_id_to_coord[latticeId]

        latticeIdx = self._lattice_id_to_idx[latticeId]
        ei         = self['lattice', 'in', 'cone'].edge_index
        mask       = ei[0] == latticeIdx
        coneIds    = [self._cone_idx_to_id[i] for i in ei[1][mask].tolist()]

        allNewConeIds = []
        for coneId in coneIds:
            oldCone = self._cone_objects[coneId]
            results = oldCone.subdivide(coord)

            if not results:
                continue

            formerNeighbors = self.getConeNeighbors(coneId)
            self.removeAllAdjacentEdges(coneId)

            self.overwriteConeNode(coneId, results[0])
            newConeIds = [coneId]

            for cone in results[1:]:
                newConeIds.append(self.addConeNode(cone))

            ## all new cones mutually adjacent (they all share the subdivision point as a generator)
            for i in range(len(newConeIds)):
                for j in range(i + 1, len(newConeIds)):
                    if self.conesAdjacentByGraph(newConeIds[i], newConeIds[j]):
                        self.addAdjacentEdge(newConeIds[i], newConeIds[j])

            ## re-check former neighbors against each new cone
            for neighborId in formerNeighbors:
                for newId in newConeIds:
                    if self.conesAdjacentByGraph(neighborId, newId):
                        self.addAdjacentEdge(neighborId, newId)

            allNewConeIds.extend(newConeIds)

        return allNewConeIds

    ## --- RL interface ---

    def isDecomposed(self) -> bool:
        for cone in self._cone_objects.values():
            if cone.isSingular:
                return False
        return True

    def getValidActions(self) -> list[int]:
        ei = self['lattice', 'in', 'cone'].edge_index
        return list(set(self._lattice_idx_to_id[i] for i in ei[0].tolist()))

    def toHeteroData(self) -> HeteroData:
        data = HeteroData()
        data['cone'].x      = self['cone'].x.clone()
        data['generator'].x = self['generator'].x.clone()
        data['lattice'].x   = self['lattice'].x.clone()

        edgeTypes = [
            ('cone',      'has',        'generator'),
            ('generator', 'of',         'cone'     ),
            ('cone',      'contains',   'lattice'  ),
            ('lattice',   'in',         'cone'     ),
            ('generator', 'reaches',    'lattice'  ),
            ('lattice',   'reached_by', 'generator'),
            ('cone',      'adjacent',   'cone'     ),
        ]
        for key in edgeTypes:
            data[key].edge_index = self[key].edge_index.clone()
            store = self[key]
            if hasattr(store, 'edge_attr') and store.edge_attr is not None:
                data[key].edge_attr = store.edge_attr.clone()

        return data

    ## --- debug printing ---

    def printGraphInfo(self) -> None:
        print("=== Cone nodes ===")
        for cid in sorted(self._cone_id_to_idx):
            cone = self._cone_objects[cid]
            gens = sorted(self.getConeGeneratorIds(cid))
            print(f"  Cone {cid}: mult={cone.multiplicity}  generators={gens}  rays={cone.rays}")

        print("\n=== Generator nodes ===")
        for gid in sorted(self._gen_id_to_idx):
            coord = self._gen_id_to_coord[gid]
            print(f"  Generator {gid}: coord={coord}")

        print("\n=== Lattice nodes ===")
        for lid in sorted(self._lattice_id_to_idx):
            coord = self._lattice_id_to_coord[lid]
            print(f"  Lattice {lid}: coord={coord}")

        print("\n=== Cone adjacency ===")
        ei = self['cone', 'adjacent', 'cone'].edge_index
        adj = {cid: [] for cid in self._cone_id_to_idx}
        for src, dst in ei.t().tolist():
            adj[self._cone_idx_to_id[src]].append(self._cone_idx_to_id[dst])
        for cid in sorted(adj):
            print(f"  Cone {cid}: {sorted(adj[cid])}")

        print("\n=== Cone -> Lattice containment ===")
        ei = self['cone', 'contains', 'lattice'].edge_index
        cl = {cid: [] for cid in self._cone_id_to_idx}
        for src, dst in ei.t().tolist():
            cl[self._cone_idx_to_id[src]].append(self._lattice_idx_to_id[dst])
        for cid in sorted(cl):
            print(f"  Cone {cid}: lattice {sorted(cl[cid])}")

        print("\n=== Generator -> Lattice (barycentric) ===")
        ei   = self['generator', 'reaches', 'lattice'].edge_index
        attr = self['generator', 'reaches', 'lattice'].edge_attr
        for k in range(ei.size(1)):
            gid = self._gen_idx_to_id[ei[0, k].item()]
            lid = self._lattice_idx_to_id[ei[1, k].item()]
            lam = attr[k, 0].item()
            gcoord = self._gen_id_to_coord[gid]
            lcoord = self._lattice_id_to_coord[lid]
            print(f"  Gen {gid} {gcoord}  ->  Lattice {lid} {lcoord}  lambda={lam:.4f}")


def main():
    c = Cone.buildCone([[1, 0, 0, 0], [1, 2, 0, 0], [1, 2, 1, 0], [0, 1, 2, 3]])
    print(f"Initial cone: rays={c.rays}  mult={c.multiplicity}")
    print(f"FPP: {c.extraneousSet()}\n")

    g = CGLGraph()
    g.addConeNode(c)

    print("--- Before subdivision ---")
    g.printGraphInfo()

    ## pick the first non-trivial lattice point (non-zero, non-degenerate action)
    validActions = g.getValidActions()
    print(f"\nValid actions (lattice ids): {validActions}")

    latticeId = validActions[0]
    subdivPoint = g._lattice_id_to_coord[latticeId]
    print(f"Subdividing on lattice {latticeId} at {subdivPoint}\n")

    newIds = g.subdivide(latticeId)
    print(f"Resulting cone ids: {newIds}")
    print(f"isDecomposed: {g.isDecomposed()}")

    print("\n--- After subdivision ---")
    g.printGraphInfo()

    print("\n--- Verification ---")

    ## all new cones should contain the subdivision point as a generator
    allHaveSubdivPoint = True
    for cid in newIds:
        cone = g._cone_objects[cid]
        if subdivPoint not in cone.rays:
            allHaveSubdivPoint = False
    print(f"All new cones have subdivision point as a ray: {allHaveSubdivPoint}")

    ## all new cones should be mutually adjacent
    allMutuallyAdjacent = True
    for i in range(len(newIds)):
        for j in range(len(newIds)):
            if i == j:
                continue
            if newIds[j] not in g.getConeNeighbors(newIds[i]):
                allMutuallyAdjacent = False
    print(f"All new cones mutually adjacent: {allMutuallyAdjacent}")

    ## subdivision point should no longer be a valid action
    print(f"Subdivision point no longer a valid action: {latticeId not in g.getValidActions()}")

    ## toHeteroData should work
    data = g.toHeteroData()
    print(f"\ntoHeteroData shapes:")
    print(f"  cone.x:      {data['cone'].x.shape}")
    print(f"  generator.x: {data['generator'].x.shape}")
    print(f"  lattice.x:   {data['lattice'].x.shape}")
    print(f"  cone-adj edge_index: {data['cone','adjacent','cone'].edge_index.shape}")


main()

