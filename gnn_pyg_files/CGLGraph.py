import math
from torch_geometric.data import HeteroData
from Cone import Cone
from utils import isPrimitiveNonzero
import torch

DIMENSION: int = 4  ## set to 4 for testing, 7 in final

Vector = tuple[int, ...]

## Heterogeneous graph over a fan with four node types:
##   cone      - feature: multiplicity repeated n times, zero-padded to DIMENSION
##   generator - feature: coordinate vector, zero-padded to DIMENSION
##   lattice   - feature: coordinate vector, zero-padded to DIMENSION
##   virtual   - no features (single virtual node connected to every cone)
##
## Edge types:
##   (cone, has, generator)         - cone uses this ray as a generator
##   (generator, of, cone)          - reverse
##   (cone, contains, lattice)      - lattice point is in cone's FPP
##   (lattice, in, cone)            - reverse
##   (generator, reaches, lattice)  - with edge_attr: scalar barycentric coord λ_i
##   (lattice, reached_by, generator) - reverse, same edge_attr
##   (cone, adjacent, cone)         - share >= 2 generator nodes (both directions)
##   (virtual, overview, cone)       - virtual node -> every cone
##   (cone, summary, virtual)        - reverse
##
## Generator and lattice nodes are globally unique by coordinate.
## Generator-lattice edges are permanent once created.
## The virtual node is always node index 0 and is created in __init__.
class CGLGraph(HeteroData):

    def __init__(self, dimension: int = DIMENSION):
        super().__init__()

        self._dimension: int = dimension
        self._singular_count: int = 0

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

        ## all node feature vectors are shape (_dimension,)
        self['cone'].x      = torch.empty((0, self._dimension), dtype=torch.float)
        self['generator'].x = torch.empty((0, self._dimension), dtype=torch.float)
        self['lattice'].x   = torch.empty((0, self._dimension), dtype=torch.float)

        ## virtual node: single featureless node, always index 0
        self['virtual'].x = torch.zeros((1, 0), dtype=torch.float)

        self['cone',      'has',        'generator'].edge_index = torch.empty((2, 0), dtype=torch.long)
        self['generator', 'of',         'cone'     ].edge_index = torch.empty((2, 0), dtype=torch.long)
        self['cone',      'contains',   'lattice'  ].edge_index = torch.empty((2, 0), dtype=torch.long)
        self['lattice',   'in',         'cone'     ].edge_index = torch.empty((2, 0), dtype=torch.long)
        self['generator', 'reaches',    'lattice'  ].edge_index = torch.empty((2, 0), dtype=torch.long)
        self['generator', 'reaches',    'lattice'  ].edge_attr  = torch.empty((0, 1), dtype=torch.float)
        self['lattice',   'reached_by', 'generator'].edge_index = torch.empty((2, 0), dtype=torch.long)
        self['lattice',   'reached_by', 'generator'].edge_attr  = torch.empty((0, 1), dtype=torch.float)
        self['cone',      'adjacent',   'cone'     ].edge_index = torch.empty((2, 0), dtype=torch.long)
        self['virtual',    'overview',   'cone'     ].edge_index = torch.empty((2, 0), dtype=torch.long)
        self['cone',      'summary',    'virtual'   ].edge_index = torch.empty((2, 0), dtype=torch.long)

    ## --- feature computation ---

    def computeConeFeature(self, cone: Cone) -> torch.Tensor:
        n    = len(cone.rays[0])
        mult = float(cone.multiplicity)
        feat = []
        for _ in range(n):
            feat.append(mult)
        while len(feat) < self._dimension:
            feat.append(0.0)
        return torch.tensor(feat, dtype=torch.float)

    def computeVectorFeature(self, coord: tuple) -> torch.Tensor:
        feat = []
        for x in coord:
            feat.append(float(x))
        while len(feat) < self._dimension:
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

    def addVirtualConeEdge(self, coneId: int) -> None:
        ## virtual node is always index 0; connects to every cone
        coneIdx = self._cone_id_to_idx[coneId]

        ei = self['virtual', 'overview', 'cone'].edge_index
        if ((ei[0] == 0) & (ei[1] == coneIdx)).any():
            return

        newFwd = torch.tensor([[0], [coneIdx]], dtype=torch.long)
        self['virtual', 'overview', 'cone'].edge_index = torch.cat([ei, newFwd], dim=1)

        ei     = self['cone', 'summary', 'virtual'].edge_index
        newBwd = torch.tensor([[coneIdx], [0]], dtype=torch.long)
        self['cone', 'summary', 'virtual'].edge_index = torch.cat([ei, newBwd], dim=1)

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
        for point, lam in zip(*cone.extraneousSet):
            if not isPrimitiveNonzero(point):
                continue
            latticeId   = self.getOrCreateLatticeNode(point)
            self.addConeLatticeEdge(coneId, latticeId)
            for i in range(len(lam)):
                if lam[i] > 0:
                    genId = self._coord_to_gen_id[cone.rays[i]]
                    self.addGeneratorLatticeEdge(genId, latticeId, float(lam[i]))

    ## --- cone node operations ---

    def addConeNode(self, cone: Cone) -> int:
        if cone.isSingular:
            self._singular_count += 1

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
        self.addVirtualConeEdge(coneId)         ## connect new cone to virtual node

        return coneId

    def overwriteConeNode(self, coneId: int, cone: Cone) -> None:
        self.removeAllConeGeneratorEdges(coneId)
        self.removeAllConeLatticeEdges(coneId)

        oldCone = self._cone_objects[coneId]
        if oldCone.isSingular:
            self._singular_count -= 1
        if cone.isSingular:
            self._singular_count += 1

        idx = self._cone_id_to_idx[coneId]
        self._cone_objects[coneId] = cone
        self['cone'].x[idx] = self.computeConeFeature(cone)

        self.wireConeGenerators(coneId, cone)  ## must precede wireConeLattice
        self.wireConeLattice(coneId, cone)
        ## virtual edge is retained: overwrite keeps the same coneId/coneIdx,
        ## so the existing virtual <-> cone edges remain valid and are not re-added

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
                newConeIds.append(self.addConeNode(cone))  ## addConeNode calls addVirtualConeEdge

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
        return self._singular_count == 0

    def getValidActions(self) -> list[int]:
        ei = self['lattice', 'in', 'cone'].edge_index
        return list(set(self._lattice_idx_to_id[i] for i in ei[0].tolist()))

    def toHeteroData(self) -> HeteroData:
        data = HeteroData()
        data['cone'].x      = self['cone'].x.clone()
        data['generator'].x = self['generator'].x.clone()
        data['lattice'].x   = self['lattice'].x.clone()
        data['virtual'].x    = self['virtual'].x.clone()

        edgeTypes = [
            ('cone',      'has',        'generator'),
            ('generator', 'of',         'cone'     ),
            ('cone',      'contains',   'lattice'  ),
            ('lattice',   'in',         'cone'     ),
            ('generator', 'reaches',    'lattice'  ),
            ('lattice',   'reached_by', 'generator'),
            ('cone',      'adjacent',   'cone'     ),
            ('virtual',    'overview',   'cone'     ),
            ('cone',      'summary',    'virtual'   ),
        ]
        for key in edgeTypes:
            data[key].edge_index = self[key].edge_index.clone()
            store = self[key]
            if hasattr(store, 'edge_attr') and store.edge_attr is not None:
                data[key].edge_attr = store.edge_attr.clone()

        return data
    
    def copy(self) -> "CGLGraph":
        new = CGLGraph(dimension = self._dimension)
        new._singular_count = self._singular_count

        ## node features
        new['cone'].x      = self['cone'].x.clone()
        new['generator'].x = self['generator'].x.clone()
        new['lattice'].x   = self['lattice'].x.clone()
        new['virtual'].x   = self['virtual'].x.clone()

        ## edges (indices + optional attrs)
        edgeTypes = [
            ('cone',      'has',        'generator'),
            ('generator', 'of',         'cone'     ),
            ('cone',      'contains',   'lattice'  ),
            ('lattice',   'in',         'cone'     ),
            ('generator', 'reaches',    'lattice'  ),
            ('lattice',   'reached_by', 'generator'),
            ('cone',      'adjacent',   'cone'     ),
            ('virtual',   'overview',   'cone'     ),
            ('cone',      'summary',    'virtual'  ),
        ]
        for key in edgeTypes:
            new[key].edge_index = self[key].edge_index.clone()
            store = self[key]
            if hasattr(store, 'edge_attr') and store.edge_attr is not None:
                new[key].edge_attr = store.edge_attr.clone()

        ## cone bookkeeping (Cone is frozen/immutable, so shared by reference)
        new._cone_id_to_idx = dict(self._cone_id_to_idx)
        new._cone_idx_to_id = dict(self._cone_idx_to_id)
        new._next_cone_id   = self._next_cone_id
        new._cone_objects   = dict(self._cone_objects)

        ## generator bookkeeping
        new._coord_to_gen_id = dict(self._coord_to_gen_id)
        new._gen_id_to_idx   = dict(self._gen_id_to_idx)
        new._gen_idx_to_id   = dict(self._gen_idx_to_id)
        new._gen_id_to_coord = dict(self._gen_id_to_coord)
        new._next_gen_id     = self._next_gen_id

        ## lattice bookkeeping
        new._coord_to_lattice_id = dict(self._coord_to_lattice_id)
        new._lattice_id_to_idx   = dict(self._lattice_id_to_idx)
        new._lattice_idx_to_id   = dict(self._lattice_idx_to_id)
        new._lattice_id_to_coord = dict(self._lattice_id_to_coord)
        new._next_lattice_id     = self._next_lattice_id

        ## gen-lattice edge set
        new._gen_lattice_edge_set = set(self._gen_lattice_edge_set)

        return new

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

        print("\n=== Virtual node -> Cone edges ===")
        ei = self['virtual', 'overview', 'cone'].edge_index
        coneIdxs = ei[1].tolist()
        print(f"  Virtual -> cones: {sorted(self._cone_idx_to_id[i] for i in coneIdxs)}")

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
    print(f"FPP: {c.extraneousSet[0]}\n")

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

    ## virtual node should be connected to all cones
    ei = g['virtual', 'overview', 'cone'].edge_index
    connectedConeIdxs = set(ei[1].tolist())
    allConeIdxs = set(g._cone_id_to_idx[cid] for cid in g._cone_id_to_idx)
    print(f"Virtual node connected to all cones: {connectedConeIdxs == allConeIdxs}")

    ## toHeteroData should work
    data = g.toHeteroData()
    print(f"\ntoHeteroData shapes:")
    print(f"  cone.x:      {data['cone'].x.shape}")
    print(f"  generator.x: {data['generator'].x.shape}")
    print(f"  lattice.x:   {data['lattice'].x.shape}")
    print(f"  virtual.x:   {data['virtual'].x.shape}")
    print(f"  cone-adj edge_index:      {data['cone','adjacent','cone'].edge_index.shape}")
    print(f"  virtual-overview edge_index: {data['virtual','overview','cone'].edge_index.shape}")


if __name__ == "__main__":
    main()