import math
from sympy import Matrix, Rational, linsolve
from dataclasses import dataclass
from functools import cached_property
from fractions import Fraction

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
       





