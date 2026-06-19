import math
from sympy import Matrix, Rational, linsolve
from dataclasses import dataclass
from functools import cached_property
from fractions import Fraction
from torch_geometric.data import HeteroData
import torch

Vector = tuple[int, ...]

## Makes an input vector primitive
def primitive (v: Vector) -> Vector:
    gcd = math.gcd(*v) 
    
    if (gcd == 0):
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
def intDet (M: tuple[Vector, ...]) -> int:
    M = [list(row) for row in M]
    n = len(M)
    sign, prev = 1, 1
    for k in range(n-1): 
        if M[k][k] == 0: # if the leading principal minor is 0, we need to swap the row with a non-zero element
            for r in range(k+1, n):
                if M[r][k] != 0:
                    M[r], M[k] = M[k], M[r]
                    sign *= -1
                    break
            else:
                return 0
        for i in range(k+1, n):
            for j in range (k+1, n):
                M[i][j] = (M[i][j]*M[k][k] - M[i][k]*M[k][j])//prev
            M[i][k] = 0
        prev = M[k][k]
    return sign * M[-1][-1]

## puts a matrix into our canonical form
def canonicalForm(M: list[list[int]]) -> list[list[int]]:
    n = len(M)
    ## triangularization step: for each column, reduce the column [x,y] -> [g, 0] where g = gcd(x,y). 
    for j in range(n):
        for i in range(j + 1, n): ## reduce M[i][j] by M[j][j]
            if M[i][j] == 0:
                continue
            g, a, b = extendedEuclid(M[j][j],M[i][j])
            ## triangularize via left mult by unimodular mx:
            rowJ = [a * M[j][col] + b * M[i][col] for col in range(n)]
            rowI = [-(M[i][j]//g) * M[j][col] + (M[j][j]//g) * M[i][col] for col in range(n)] ## turns M[i][j] -> 0
            M[j], M[i] = rowJ, rowI
        if M[j][j] < 0: ## pivot should be positive
            M[j] = [-x for x in M[j]]
    
    ## reduce each column by the pivot. Makes all entries nonnegative and less than the pivot. 
    for j in range(n):
        for i in range(j):
            quotient = M[i][j] // M[j][j]
            if quotient:
                M[i] = [M[i][col] - quotient * M[j][col] for col in range(n)]
    
    return M


## Custom (rational, simplicial, full dimensional) cone class for our environment (n-dim cone with n generators)
@dataclass(frozen = True)
class Cone:
    rays: tuple[Vector, ...]

    ## Builds cone from list of rays and stores primitive data
    ## rays must be linearly independent and full-dimensional
    @classmethod
    def buildCone(cls, rays) -> "Cone":
        n = len(rays)

        if not rays:
            raise ValueError("need at least one generator")
        
        if any(all(x == 0 for x in r) for r in rays):
            raise ValueError("generators cannot be zero vector")
            
        tuple(sorted(rays)) ## sort to have a canonical form
        rays = tuple(primitive(tuple(r)) for r in rays)

        if any(len(r) != n for r in rays):
            raise ValueError("cone must be full dimensional")

        if Matrix(rays).rank() != n:
            raise ValueError("cone must be simplicial")

        return cls(rays)

    ## dimension of cone
    @property
    def dimension(self) -> int:
        return len(self.rays[0])

    ## number of generators
    @property
    def numGenerators(self) -> int:
        return len(self.rays)

    ## returns True if cone is singular
    @property
    def isSingular(self) -> bool:
        return self.multiplicity != 1

    ## multiplicity of cone
    @cached_property
    def multiplicity(self) -> int:
        return abs(intDet(self.rays))
    
    ## returns the barycentric coordinates of the point p. 
    def barycentricCoords(self, p: Vector) -> tuple[Fraction,...]:
        A = Matrix(self.rays).T
        b = Matrix(p)
        (coords,) = linsolve((A,b))
        coords = tuple(Fraction(c.p, c.q) for c in coords)

        if any(x<0 for x in coords):
            raise ValueError("point p must be in the cone")

        return coords
    
    ## returns True if the cone contains p
    def contains(self, p: Vector) -> bool:
        A = Matrix(self.rays).T
        b = Matrix(p)
        (coords,) = linsolve((A,b))
        coords = tuple(coords)

        if any(x<0 for x in coords):
            return False
        
        return True
    
    ## gives the extraneous set (all points in the fundamental parallelepiped) of the cone
    def extraneousSet(self) -> list[Vector]:
        n = len(self.rays)
        A = [[r[j] for r in self.rays] for j in range(n)]
        H = canonicalForm([row[:] for row in A]) ## copy over

        lambdas = [[Fraction(i, H[-1][-1])] for i in range(H[-1][-1])]

        for i in reversed(range(n-1)):
            newLambdas = []
            for curr in lambdas:
                s = sum(curr[col - (i+1)] * H[i][col] for col in range(i+1, n))
                for k in range(H[i][i]): 
                    newLambdas.append([Fraction(math.ceil(s) - s + k,H[i][i])] + curr)
            lambdas = newLambdas 

        lambdas = [tuple(int(math.sumprod(A[i], vec)) for i in range(n)) for vec in lambdas]

        return lambdas

    ## put cone into HNF
    def HNF(self) -> "Cone":
        n = len(self.rays)
        H = [[r[j] for r in self.rays] for j in range(n)]
        H = canonicalForm(H)
        self.rays = tuple(tuple(H[i][j] for i in range(n)) for j in range(n))

    
    ## stellar subdivides through the given point, and returns the resulting cones. 
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
    
## Heterogeneous graph over a fan
class ConeLatticeGraph(HeteroData):
    def __init__(self):
        super().__init__()
 
        # cone nodes
        self._cone_id_to_idx: dict[int, int] = {}
        self._cone_idx_to_id: dict[int, int] = {}
        self._next_cone_id: int = 0
        self._cone_objects: dict[int, "Cone"] = {}
 
        # lattice nodes, keyed globally by coordinate to avoid duplicates
        self._coord_to_lattice_id: dict[tuple, int] = {}
        self._lattice_id_to_idx: dict[int, int] = {}
        self._lattice_idx_to_id: dict[int, int] = {}
        self._lattice_id_to_coord: dict[int, tuple] = {}
        self._next_lattice_id: int = 0
 
        self['cone'].x = torch.empty((0, 0))
        self['lattice'].x = torch.empty((0, 0))
 
        self['cone', 'adjacent', 'cone'].edge_index = torch.empty((2, 0), dtype=torch.long)
        self['cone', 'contains', 'lattice'].edge_index = torch.empty((2, 0), dtype=torch.long)
        self['lattice', 'contains', 'cone'].edge_index = torch.empty((2, 0), dtype=torch.long)

    def listCones(self, verbose: bool=False) -> list[tuple]:
        cones = []
        for cid in sorted(self._cone_id_to_idx):
            cone = self._cone_objects[cid]
            if verbose:
                print(f"Cone {cid}: rays={cone.rays}  mult={cone.multiplicity}")
            cones.append(cone.rays)
        return cones
     
    def listConeLatticePoints(self, coneId: int, verbose: bool=False) -> list[tuple]:
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
 
    ## adds a single ('cone','adjacent','cone') edge pair (both directions, since
    ## adjacency is undirected but edge_index is inherently directional)
    def addAdjacentEdge(self, coneIdA: int, coneIdB: int) -> None:
        idxA = self._cone_id_to_idx[coneIdA]
        idxB = self._cone_id_to_idx[coneIdB]
 
        existing = self['cone', 'adjacent', 'cone'].edge_index
        newEdges = torch.tensor([[idxA, idxB], [idxB, idxA]], dtype=torch.long)
        self['cone', 'adjacent', 'cone'].edge_index = torch.cat([existing, newEdges], dim=1)
 
    ## removes every ('cone','adjacent','cone') edge touching coneId (both directions).
    ## used before re-deriving adjacency for a cone whose generators just changed
    ## since its old edges no longer reflect the new Cone object occupying that row
    def removeAllAdjacentEdges(self, coneId: int) -> None:
        idx = self._cone_id_to_idx[coneId]
        edgeIndex = self['cone', 'adjacent', 'cone'].edge_index
        keepMask = (edgeIndex[0] != idx) & (edgeIndex[1] != idx)
        self['cone', 'adjacent', 'cone'].edge_index = edgeIndex[:, keepMask]
 
    ## returns the cone_ids currently adjacent to coneId
    def getConeNeighbors(self, coneId: int) -> list[int]:
        idx = self._cone_id_to_idx[coneId]
        edgeIndex = self['cone', 'adjacent', 'cone'].edge_index
        mask = edgeIndex[0] == idx
        neighborIdxs = edgeIndex[1][mask].tolist()
        return [self._cone_idx_to_id[i] for i in neighborIdxs]
 
    ## adds a single ('cone','contains','lattice') / ('lattice','contains','cone') edge pair
    def addContainsEdge(self, coneId: int, latticeId: int) -> None:
        coneIdx = self._cone_id_to_idx[coneId]
        latticeIdx = self._lattice_id_to_idx[latticeId]
 
        forward = self['cone', 'contains', 'lattice'].edge_index
        newForward = torch.tensor([[coneIdx], [latticeIdx]], dtype=torch.long)
        self['cone', 'contains', 'lattice'].edge_index = torch.cat([forward, newForward], dim=1)
 
        backward = self['lattice', 'contains', 'cone'].edge_index
        newBackward = torch.tensor([[latticeIdx], [coneIdx]], dtype=torch.long)
        self['lattice', 'contains', 'cone'].edge_index = torch.cat([backward, newBackward], dim=1)
 
    ## gets the lattice_id for this coordinate, creating the node only if a lattice
    ## node with this coordinate doesn't already exist
    def getOrCreateLatticeNode(self, coord: tuple) -> int:
        if coord in self._coord_to_lattice_id:
            return self._coord_to_lattice_id[coord]
 
        latticeId = self._next_lattice_id
        self._next_lattice_id += 1
 
        idx = self['lattice'].x.size(0)
        self._coord_to_lattice_id[coord] = latticeId
        self._lattice_id_to_idx[latticeId] = idx
        self._lattice_idx_to_id[idx] = latticeId
        self._lattice_id_to_coord[latticeId] = coord
 
        placeholderRow = torch.zeros((1, self['lattice'].x.size(1)))
        self['lattice'].x = torch.cat([self['lattice'].x, placeholderRow], dim=0)
 
        return latticeId
 
    ## adds a new cone node, then automatically populates its fundamental
    ## parallelepiped's lattice points via Cone.extraneousSet(), reusing any
    ## lattice node that already exists at a given coordinate, and wires a 
    ## contains-edge from the cone to each such lattice point
    def addConeNode(self, cone: "Cone") -> int:
        coneId = self._next_cone_id
        self._next_cone_id += 1
 
        idx = self['cone'].x.size(0)
        self._cone_id_to_idx[coneId] = idx
        self._cone_idx_to_id[idx] = coneId
        self._cone_objects[coneId] = cone
 
        placeholderRow = torch.zeros((1, self['cone'].x.size(1)))
        self['cone'].x = torch.cat([self['cone'].x, placeholderRow], dim=0)
 
        for point in cone.extraneousSet():
            if math.gcd(*point) == 1:
                latticeId = self.getOrCreateLatticeNode(point)
                self.addContainsEdge(coneId, latticeId)
 
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
        self['cone'].x[idx] = torch.zeros(self['cone'].x.size(1))
 
        for point in cone.extraneousSet():
            latticeId = self.getOrCreateLatticeNode(point)
            self.addContainsEdge(coneId, latticeId)
 
    ## stellar-subdivides the cone identified by coneId through the lattice
    ## point identified by latticeId, using Cone.subdivide(p) to compute the
    ## resulting fan, then wires adjacency
    def subdivide(self, coneId: int, latticeId: int) -> list[int]:
        oldCone = self._cone_objects[coneId]
        coord = self._lattice_id_to_coord[latticeId]
 
        results = oldCone.subdivide(coord)
        if not results:
            raise ValueError("subdivide produced no resulting cones")
 
        formerNeighbors = self.getConeNeighbors(coneId)
        self.removeAllAdjacentEdges(coneId)
 
        # results[0] overwrites the existing row/id, results[1:] get appended
        self.overwriteConeNode(coneId, results[0])
        newConeIds = [coneId]
        for cone in results[1:]:
            newConeIds.append(self.addConeNode(cone))
 
        # every pair of new cones is mutually adjacent (they all share p)
        for i in range(len(newConeIds)):
            for j in range(i + 1, len(newConeIds)):
                self.addAdjacentEdge(newConeIds[i], newConeIds[j])
 
        # re-check each former neighbor against each new cone
        for neighborId in formerNeighbors:
            neighborCone = self._cone_objects[neighborId]
            for newId in newConeIds:
                newCone = self._cone_objects[newId]
                if conesAdjacent(neighborCone, newCone):
                    self.addAdjacentEdge(neighborId, newId)
 
        return newConeIds

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
            cone_id = self._cone_idx_to_id[src]
            lattice_id = self._lattice_idx_to_id[dst]
            cone_contains[cone_id].append(lattice_id)

        for cid in sorted(cone_contains):
            print(f"Cone {cid}: {sorted(cone_contains[cid])}")

        print("\n=== Lattice -> Cone containment ===")
        edge_index = self['lattice', 'contains', 'cone'].edge_index

        lattice_contains = {lid: [] for lid in self._lattice_id_to_idx.keys()}

        for src, dst in edge_index.t().tolist():
            lattice_id = self._lattice_idx_to_id[src]
            cone_id = self._cone_idx_to_id[dst]
            lattice_contains[lattice_id].append(cone_id)

        for lid in sorted(lattice_contains):
            print(f"Lattice {lid}: {sorted(lattice_contains[lid])}")

def main():
    c = Cone.buildCone([[1, 0, 0, 0], [1, 2, 0, 0], [1, 2, 1, 0], [0, 1, 2, 3]])
    fpp = c.extraneousSet()
    print(fpp)
    for i in range(0, len(fpp)):
        print(c.barycentricCoords(fpp[i]))

    CLG = ConeLatticeGraph()
    CLG.addConeNode(c)

    CLG.printAdjacencyList()

    CLG.subdivide(0, 2)

    print()
    CLG.printAdjacencyList()

    CLG.listCones(True)
    CLG.listConeLatticePoints(1, True)

main()
