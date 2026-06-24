import math
from sympy import Matrix
from dataclasses import dataclass
from fractions import Fraction
from torch_geometric.data import HeteroData
from coneEnvironment import *
import torch
import random

Vector = tuple[int, ...]

DIMENSION: int = 7

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
        
        self['cone',    'contains', 'lattice'].edge_attr = torch.empty((0, dimension), dtype=torch.float)
        self['lattice', 'contains', 'cone'   ].edge_attr = torch.empty((0, dimension), dtype=torch.float)

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
        ei = self['cone', 'adjacent', 'cone'].edge_index
        duplicate = ((ei[0] == idxA) & (ei[1] == idxB)).any()
        if duplicate:
            return
        newEdges = torch.tensor([[idxA, idxB], [idxB, idxA]], dtype=torch.long)
        self['cone', 'adjacent', 'cone'].edge_index = torch.cat([ei, newEdges], dim=1)

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
        
        fwd = self['cone', 'contains', 'lattice']
        keep = fwd.edge_index[0] != coneIdx
        fwd.edge_index = fwd.edge_index[:, keep]
        fwd.edge_attr  = fwd.edge_attr[keep]

        bwd = self['lattice', 'contains', 'cone']
        keep = bwd.edge_index[1] != coneIdx
        bwd.edge_index = bwd.edge_index[:, keep]
        bwd.edge_attr  = bwd.edge_attr[keep]

    def overwriteConeNode(self, coneId: int, cone: "Cone") -> None:
        self.removeAllContainsEdges(coneId)
        idx = self._cone_id_to_idx[coneId]
        self._cone_objects[coneId] = cone

        self['cone'].x[idx] = computeConeFeatures(cone)

        self.addConeCandidateEdges(coneId, cone)

    def subdivide(self, latticeId: int) -> list[int]:
        coord = self._lattice_id_to_coord[latticeId]
        
        ## find all cones this lattice point belongs to
        latticeIdx = self._lattice_id_to_idx[latticeId]
        ei = self['lattice', 'contains', 'cone'].edge_index
        mask = ei[0] == latticeIdx
        coneIds = [self._cone_idx_to_id[i] for i in ei[1][mask].tolist()]

        allNewConeIds = []
        for coneId in coneIds:
            oldCone = self._cone_objects[coneId]
            results = oldCone.subdivide(coord)

            if not results:
                raise ValueError(f"subdivide produced no cones for cone {coneId}")

            formerNeighbors = self.getConeNeighbors(coneId)
            self.removeAllAdjacentEdges(coneId)

            self.overwriteConeNode(coneId, results[0])
            newConeIds = [coneId]
            for cone in results[1:]:
                newConeIds.append(self.addConeNode(cone))

            ## all new cones from this subdivision are mutually adjacent
            for i in range(len(newConeIds)):
                for j in range(i + 1, len(newConeIds)):
                    self.addAdjacentEdge(newConeIds[i], newConeIds[j])

            ## re-check former neighbors against each new cone
            for neighborId in formerNeighbors:
                neighborCone = self._cone_objects[neighborId]
                for newId in newConeIds:
                    if conesAdjacent(neighborCone, self._cone_objects[newId]):
                        self.addAdjacentEdge(neighborId, newId)

            allNewConeIds.extend(newConeIds)

        return allNewConeIds
    
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
    
    def getValidActions(self) -> list[int]:
        ei = self['lattice', 'contains', 'cone'].edge_index
        return list(set(self._lattice_idx_to_id[i] for i in ei[0].tolist()))

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

def main():
    # This demo uses 4-D cones, so we pass dimension=4 explicitly.
    # For your real work, omit the argument and the DIMENSION=7 default applies.
    c = Cone.buildCone([[1, 0, 0, 0], [1, 2, 0, 0], [1, 2, 1, 0], [0, 1, 2, 3]])
    fpp = c.extraneousSet()
    print(fpp)
    for i in range(len(fpp)):
        print(c.barycentricCoords(fpp[i]))

    CLG = ConeLatticeGraph(dimension=4)
    CLG.addConeNode(c)

    CLG.printAdjacencyList()

    CLG.subdivide(2)

    print()
    CLG.printAdjacencyList()

    CLG.listCones(True)
    CLG.listConeLatticePoints(1, True)

    # Demonstrate that toHeteroData() now returns real features
    data = CLG.toHeteroData()
    print(f"\ncone feature shape:    {data['cone'].x.shape}")     # (num_cones, 4²+1 = 17)
    print(f"lattice feature shape: {data['lattice'].x.shape}")    # (num_lattice, 4)
    print(f"\nFirst cone features:\n{data['cone'].x[0]}")

    # M = Matrix(generateRandomCone(4, 12, 100))
    # print(M)
    # print(M.det())
