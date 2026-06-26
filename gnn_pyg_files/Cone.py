import math
from dataclasses import dataclass
from functools import cached_property
from fractions import Fraction
from utils import primitive, intDet, canonicalForm, linsolve

Vector = tuple[int, ...]

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

        if intDet(rays) == 0:
            raise ValueError("cone must be simplicial")

        return cls(rays)
    
    ## makes sure that if the cone is constructed by bypassing the constructor that rays are stored in the correct format. 
    def __post_init__(self):
        object.__setattr__(self, "rays", tuple(tuple(r) for r in self.rays))

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
        A = [[gen[i] for gen in self.rays] for i in range(len(self.rays))]
        b = [x for x in p]
        coords = linsolve(A,b)
        if any(x < 0 for x in coords):
            raise ValueError("point p must be in the cone")
        return coords

    def contains(self, p: Vector) -> bool:
        A = [[gen[i] for gen in self.rays] for i in range(len(self.rays))]
        b = [x for x in p]
        coords = linsolve(A,b)
        if any(x < 0 for x in coords):
            return False

        return True

    ## returns a tuple, first is a list of extraneous set points, and second is the barycentric coordinates of those points. 
    def extraneousSet(self) -> tuple[list[Vector], list[Vector]]:
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

        vectors = [
            tuple(int(math.sumprod(A[i], vec)) for i in range(n))
            for vec in lambdas
        ]

        ## don't return the zero vector
        z = vectors.index((0,) * n)  
        del vectors[z], lambdas[z]

        return vectors, lambdas

    def HNF(self) -> "Cone":
        n = len(self.rays)
        H = [[r[j] for r in self.rays] for j in range(n)]
        H = canonicalForm(H)
        return Cone(rays = tuple(tuple(H[i][j] for i in range(n)) for j in range(n)))

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

# ===========================================================================================
#  FAN OPERATIONS
# ===========================================================================================

## divides a list of cones through a lattice point. All cones that contain that lattice point are subdivided . 
def fanSubdivide(fan: list["Cone"], p: Vector) -> list["Cone"]:
    p = primitive(p)
    new_cones = []
    for cone in fan:
        if cone.contains(p):
            new_cones += cone.subdivide(p)
        else:
            new_cones.append(cone)
    return new_cones


# ===========================================================================================
#  TESTING
# ===========================================================================================

def main():
    # ---- nonsingular 2D cone: generators form a Z-basis, mult(σ)=1 ----
    smooth = Cone.buildCone([(1, 0), (0, 1)])
    assert smooth.dimension == 2 and smooth.numGenerators == 2
    assert smooth.multiplicity == 1
    assert smooth.isSingular is False
    # extraneous set of a nonsingular cone is {0} only
    assert smooth.extraneousSet() == ([], [])

    # ---- singular 2D cone: σ = <(1,0),(1,2)>, mult(σ)=2 ----
    sing = Cone.buildCone([(1, 0), (1, 2)])
    assert sing.multiplicity == 2
    assert sing.isSingular is True
    # |extraneous set| = mult(σ) = det(σ)
    assert len(sing.extraneousSet()[0]) == sing.multiplicity - 1 == 1

    # barycentricCoords: express a generator -> standard basis coord
    assert sing.barycentricCoords((1, 0)) == (Fraction(1), Fraction(0))
    assert sing.contains((1, 1)) is True
    assert sing.contains((-1, 0)) is False

    # HNF: canonical form must preserve multiplicity (unimodular column ops)
    assert sing.HNF().multiplicity == sing.multiplicity

    # ---- stellar subdivision through an extraneous (irreducible*) point ----
    # K \ {0} is nonempty since σ is singular; subdivide through such a w.
    w = next(v for v in sing.extraneousSet()[0] if any(c != 0 for c in v))
    star = sing.subdivide(w)
    # Subdivision Multiplicity Lemma: mult(δ_h) = λ_h · mult(σ),
    # so the refinement's multiplicities sum to mult(σ).
    assert sum(c.multiplicity for c in star) == sing.multiplicity
    # subdividing on a generating ray is rejected
    try:
        sing.subdivide((1, 0)); assert False
    except ValueError:
        pass

    # ---- singular 3D cone: <e3, e3+3e1, e3+3e2>, mult(σ)=9 ----
    cone3 = Cone.buildCone([(0, 0, 1), (3, 0, 1), (0, 3, 1)])
    assert cone3.dimension == 3 and cone3.multiplicity == 9

    # ---- fanSubdivide: refine every maximal cone whose FPP contains p ----
    #fan = [Cone.buildCone([(1, 0), (1, 2)])]
    #refined = fanSubdivide(fan, w)
    #assert sum(c.multiplicity for c in refined) == 2  # multiplicity conserved

    # ---- construction guards ----
    for bad in ([(0, 0)], [(1, 0, 0), (0, 1, 0)], [(1, 0), (2, 0)]):
        try:
            Cone.buildCone(bad); assert False
        except ValueError:
            pass

    print("all tests passed")


if __name__ == "__main__":
    main()
