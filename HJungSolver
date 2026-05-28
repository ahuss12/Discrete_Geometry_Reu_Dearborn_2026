import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import math
import sys

## Unpackage the inputs - makes sure also that inputs are rational
def parseVector (vector):
    raw = input(vector)
    components = [p.strip() for p in raw.split(",")]

    try: 
        generator = np.array([int(components[0]),int(components[1])])
    except ValueError:
       raise ValueError("Vector components must be integers") from None

    if not np.any(generator):
        raise ValueError("Generator cannot be 0 vector")

    generator //= np.gcd(generator[0],generator[1])

    return generator

## exact 2D determinant (v1,v2 cols)
def det(v1,v2):
    return v1[0]*v2[1]-v2[0]*v1[1]

## Check if cone is singular 
def isSingular (v1,v2):
    return abs(det(v1,v2)) != 1
    
## Check if 2 vects lin indep
def isLinIndep (v1,v2):
    return det(v1,v2) != 0

## Get cone into canonical form
## Inputs: Two PRIMITIVE integer vectors, with the first returned in the form (0,1)
def canonical(v1,v2):
    if isCanonical(v1,v2): 
        return (v1, v2) if (v1[0] == 0 and v1[1] == 1) else (v2, v1)

    else: 
        s, t, _ = sp.gcdex(v1[0], v1[1])
        A = np.array([[-v1[1], v1[0]], [s, t]]) ## Matrix to convert (v1 --> (0,1))
        u1 = np.array([0,1])
        u2 = A @ v2 ## Temporary second generator

        ## get u2 into correct form
        if u2[0] < 0: ## Flip if negative (want m >= 0)
            u2[0] = int(-u2[0])
        
        u2[1] = -(-u2[1] % u2[0])

        return u1, u2
        
## Check if cone is in canonical form 
## Inputs: Two integer vectors
def isCanonical(v1,v2):
    return (v1[0] == 0 and v1[1] == 1 and v2[0] > -v2[1] and v2[1] < 0) or (v2[0] == 0 and v2[1] == 1 and v1[0] > -v1[1] and v1[1] < 0)

## Main Loop
print("Cone must be 2D and have exactly 2 generators:\n")

## Get generators
gen1 = parseVector("Enter first generator (e.g. 1,0): ")
gen2 = parseVector("Enter second generator (e.g. 0,1): ")
print("\n")

print("Initial Multiplicity: ", abs(det(gen1, gen2)))

## Check that cone is non-degenerate (not 0, not a line, strongly convex)
if not isLinIndep(gen1, gen2):
    raise ValueError("Generators Must be Linearly Independent")

## Check if already nonsingular
if not isSingular(gen1, gen2):
    print("Cone is already nonsingular!")
    sys.exit(0)

u1, u2 = canonical(gen1, gen2)

m = u2[0] 
k = -u2[1]
coefficients = []

while m != 1 and k!=0: 
    m_old = m
    k_old = k
    c = math.ceil(m_old/k_old)
    k = c * k_old - m_old
    m = k_old 
    coefficients.append(c)

n = len(coefficients) + 2
refinement = np.zeros((n, 2), dtype=int)
refinement[0] = [0, 1]
refinement[1] = [1, 0]

for i in range(2, n):
    refinement[i] = coefficients[i - 2] * refinement[i - 1] - refinement[i - 2]

for i in range (n-1):
    print("Subdivision ",i+1)
    print("\t Generator 1: ", refinement[i])
    print("\t Generator 2: ", refinement[i+1])
    print("\t Cone is nonsingular: ", not isSingular(refinement[i], refinement[i+1]),"\n")


## Claude-generated visualizer
def _draw(ax, rays, title, accent, subdivide=False):
    rays = [np.array(r, dtype=float) for r in rays]
    pts = np.array(rays)
    pad = 1.5
    xmin = min(pts[:,0].min(), 0) - pad
    xmax = max(pts[:,0].max(), 0) + pad
    ymin = min(pts[:,1].min(), 0) - pad
    ymax = max(pts[:,1].max(), 0) + pad
    span = max(xmax - xmin, ymax - ymin)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    xmin, xmax = cx - span / 2, cx + span / 2
    ymin, ymax = cy - span / 2, cy + span / 2
    scale = 5 * span

    # Cone shading -- either single wedge or alternating sub-cones
    if subdivide and len(rays) > 2:
        for i in range(len(rays) - 1):
            e1 = rays[i]     * scale / np.linalg.norm(rays[i])
            e2 = rays[i + 1] * scale / np.linalg.norm(rays[i + 1])
            color = '#a8d5e8' if i % 2 == 0 else '#f5c9b8'
            ax.add_patch(plt.Polygon([(0,0), e1, e2], color=color, alpha=0.4, zorder=1))
    else:
        p1 = rays[0]  * scale / np.linalg.norm(rays[0])
        p2 = rays[-1] * scale / np.linalg.norm(rays[-1])
        ax.add_patch(plt.Polygon([(0,0), p1, p2], color=accent, alpha=0.18, zorder=1))

    # Lattice points
    for x in range(int(np.floor(xmin)), int(np.ceil(xmax)) + 1):
        for y in range(int(np.floor(ymin)), int(np.ceil(ymax)) + 1):
            ax.plot(x, y, 'o', color='dimgray', markersize=2.5, alpha=0.5, zorder=2)

    # Rays + labels
    for i, r in enumerate(rays):
        ext = r * scale / np.linalg.norm(r)
        boundary = (i == 0) or (i == len(rays) - 1)
        ax.plot([0, ext[0]], [0, ext[1]],
                '-' if boundary else '--',
                color=accent if boundary else '#444',
                linewidth=2.2 if boundary else 1.3,
                zorder=4 if boundary else 3)
        ax.plot(r[0], r[1], 'o',
                color=accent if boundary else '#222',
                markersize=9 if boundary else 6, zorder=5)
        ax.annotate(f'$v_{{{i}}}=({int(r[0])},{int(r[1])})$',
                    xy=tuple(r), xytext=(10, 8), textcoords='offset points',
                    fontsize=10.5,
                    color=accent if boundary else '#222',
                    fontweight='bold' if boundary else 'normal', zorder=6)

    ax.axhline(0, color='black', lw=0.6, alpha=0.4)
    ax.axvline(0, color='black', lw=0.6, alpha=0.4)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.set_title(title, fontsize=11)


fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6.5))

mult = abs(det(gen1, gen2))
_draw(axL, [gen1, gen2],
      f'Original cone  $\\sigma$\n'
      f'gens $({gen1[0]},{gen1[1]})$, $({gen2[0]},{gen2[1]})$,  mult $={mult}$',
      accent='#c44e52')

hj = '[' + ', '.join(map(str, coefficients)) + ']'
_draw(axR, refinement.tolist(),
      f'Nonsingular refinement (canonical form)\n'
      f'$m={u2[0]},\\ k={-u2[1]}$,  HJ: ${hj}$',
      accent='#3b6cb7', subdivide=True)

fig.suptitle('Hirzebruch–Jung resolution of a 2-dim toric singularity', fontsize=13)
plt.tight_layout()
plt.show()

