import math
import sympy as sp

Vector = tuple[int, ...]

## Makes an input vector primitive
def primitive (v: Vector) -> Vector:
    gcd = math.gcd(*v) 
    
    if (gcd == 0):
        raise ValueError("zero vector not allowed")

    return tuple(x // gcd for x in v)

## gives exact integer determinant using the Bareiss algorithm
def intDet (M: tuple[Vector, ...]) -> int:
    M = [row[:] for row in M]
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
                M[i][j] = (M[i][j]*M[k][k] - M[i][k]*M[k][j])/prev
            M[i][k] = 0
        prev = M[k][k]
    return sign * M[-1][-1]



## Custom (rational, simplicial, full dimensional) cone class for our environment (n-dim cone with n generators)
@dataclass(frozen = True)
class Cone:
    rays: tuple[Vector, ...]

    ## Builds cones from list of rays, and stores primitive data
    @classmethod
    def buildCone(cls, rays) -> "Cone":
        if not rays:
            raise ValueError("need at least one generator")
            
        sortedRays = tuple(sorted(primitive(tuple(r)) for r in rays))

        if any(len(r) != len(rays) for r in rays):
            raise ValueError("cone must be full dimensional and simplicial")

        return cls(sortedRays)

    ## dimension of cone
    @property
    def dimension(self) -> int:
        return len(rays[0])

    ## number of generators
    @property
    def numGenerators(self) -> int:
        return len(rays)

    ## returns True if cone is singular
    @property
    def isSingular(self) -> bool:
        return self.multiplicity != 1
    
    ## multiplicity of cone
    @property
    def multiplicity(self) -> int:
        return abs(intDet(rays))
    
    ## returns the barycentric coordinates of the point p
    def barycentricCoords(self, p: Vector) -> Vector:
        system = (Matrix([self.rays],Matrix(p)))
        return sympy.solvers.solveset.linsolve(system)
    
    ## returns True if the cone contains p
    def contains(self, p: Vector) -> bool:

        return
    
    ## gives the extraneous set (all points in the fundamental parallelepiped) of the cone
    def extraneousSet(self) -> list[Vector]:
        return
    
    ## stellar subdivides through the given point, and returns the resulting cones. 
    def subdivide(self, p: Vector) -> list["Cone"]:
        return 
    
    ## turns the cone into canonical form 
    def canonical(self) -> "Cone":
        return 







