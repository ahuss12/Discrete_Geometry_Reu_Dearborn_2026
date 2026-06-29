from fractions import Fraction
from math import gcd
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from sympy import Matrix
from sympy.matrices.normalforms import hermite_normal_form
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



# helpers


def det_of_cone(cone):
    """
    Determinant of the matrix with cone generators as columns.
    """
    A = Matrix.hstack(*[Matrix(v) for v in cone])
    return int(A.det())


def abs_det(cone):
    return abs(det_of_cone(cone))


def primitive(v):
    """
    Make a lattice vector primitive.
    Example: (2,4,6) -> (1,2,3)
    """
    g = 0
    for x in v:
        g = gcd(g, abs(int(x)))

    if g == 0:
        return tuple(v)

    return tuple(int(x) // g for x in v)


def generator_matrix(cone):
    """
    Matrix with cone generators as columns.
    """
    return Matrix.hstack(*[Matrix(v) for v in cone])


def hnf_of_cone(cone):
    A = generator_matrix(cone)
    return hermite_normal_form(A)


def barycentric_coordinates(x, cone):
    """
    Solve x = a1 v1 + a2 v2 + a3 v3.
    Returns exact rational barycentric coordinates.
    """
    A = generator_matrix(cone)
    coeffs = A.inv() * Matrix(x)

    return tuple(Fraction(int(c.p), int(c.q)) for c in coeffs)


def in_half_open_parallelepiped(x, cone):
    """
    Checks whether x is in:
        0 <= ai < 1
    """
    coeffs = barycentric_coordinates(x, cone)
    return all(Fraction(0) <= a < Fraction(1) for a in coeffs)


def is_strictly_inside_cone(x, cone):
    """
    Checks whether x is in the relative interior:
        ai > 0 for every i.
    """
    coeffs = barycentric_coordinates(x, cone)
    return all(a > 0 for a in coeffs)



# Fundamental parallelepiped


def bounding_box_for_parallelepiped(cone):
    """
    Crude bounding box containing the fundamental parallelepiped.
    """
    corners = []

    for eps in product([0, 1], repeat=3):
        p = tuple(
            eps[0] * cone[0][i]
            + eps[1] * cone[1][i]
            + eps[2] * cone[2][i]
            for i in range(3)
        )
        corners.append(p)

    mins = tuple(min(p[i] for p in corners) for i in range(3))
    maxs = tuple(max(p[i] for p in corners) for i in range(3))

    return mins, maxs


def lattice_points_in_parallelepiped(cone):
    """
    Lists all integer lattice points in the half-open fundamental parallelepiped.
    """
    mins, maxs = bounding_box_for_parallelepiped(cone)

    points = []

    for x in range(mins[0], maxs[0] + 1):
        for y in range(mins[1], maxs[1] + 1):
            for z in range(mins[2], maxs[2] + 1):
                p = (x, y, z)

                if in_half_open_parallelepiped(p, cone):
                    points.append(p)

    return points


# Ray choice and subdivision


def choose_subdivision_ray(cone):
    """
    Choose an interior lattice point in the fundamental parallelepiped.

    We choose the primitive candidate with smallest l1 norm.
    """
    points = lattice_points_in_parallelepiped(cone)

    candidates = []

    for p in points:
        if p == (0, 0, 0):
            continue

        if is_strictly_inside_cone(p, cone):
            candidates.append(primitive(p))

    candidates = list(dict.fromkeys(candidates))

    if not candidates:
        return None

    candidates.sort(key=lambda p: sum(abs(x) for x in p))
    return candidates[0]


def stellar_subdivide(cone, w):
    """
    Stellar subdivision of a 3D cone.

    Cone(v1,v2,v3) becomes:
        Cone(w,v2,v3)
        Cone(v1,w,v3)
        Cone(v1,v2,w)
    """
    v1, v2, v3 = cone

    return [
        (w, v2, v3),
        (v1, w, v3),
        (v1, v2, w),
    ]


# Visuals

def plot_vector(ax, v, label=None, linewidth=2):
    """
    Plot a vector from the origin.
    """
    v = np.array(v, dtype=float)

    ax.quiver(
        0, 0, 0,
        v[0], v[1], v[2],
        arrow_length_ratio=0.08,
        linewidth=linewidth
    )

    if label is not None:
        ax.text(v[0], v[1], v[2], label, fontsize=12)


def plot_cone_face(ax, cone, alpha=0.18):
    """
    Plot the triangular slice connecting the three generators.
    Uses Poly3DCollection instead of plot_trisurf to avoid Qhull errors.
    """
    v1, v2, v3 = [np.array(v, dtype=float) for v in cone]

    triangle = [[v1, v2, v3]]

    poly = Poly3DCollection(
        triangle,
        alpha=alpha,
        edgecolor="black",
        linewidths=1
    )

    ax.add_collection3d(poly)


def set_equal_axes(ax, vectors):
    """
    Makes the 3D plot scale equally in all directions.
    """
    vectors = np.array(vectors, dtype=float)

    mins = vectors.min(axis=0)
    maxs = vectors.max(axis=0)

    center = (mins + maxs) / 2
    radius = max(maxs - mins) / 2

    if radius == 0:
        radius = 1

    ax.set_xlim(center[0] - 1.2 * radius, center[0] + 1.2 * radius)
    ax.set_ylim(center[1] - 1.2 * radius, center[1] + 1.2 * radius)
    ax.set_zlim(center[2] - 1.2 * radius, center[2] + 1.2 * radius)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def visualize_one_subdivision(cone):
    """
    Visualize original cone and one stellar subdivision.
    """
    w = choose_subdivision_ray(cone)

    if w is None:
        print("No interior parallelepiped lattice point found.")
        return

    new_cones = stellar_subdivide(cone, w)

    fig = plt.figure(figsize=(14, 6))

    # Original cone
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.set_title("Original Cone")

    for i, v in enumerate(cone, start=1):
        plot_vector(ax1, v, label=f"v{i}", linewidth=3)

    plot_cone_face(ax1, cone, alpha=0.15)

    all_vecs = list(cone) + [w]
    set_equal_axes(ax1, all_vecs)

    # Subdivided cone
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.set_title("Stellar Subdivision")

    for i, v in enumerate(cone, start=1):
        plot_vector(ax2, v, label=f"v{i}", linewidth=3)

    plot_vector(ax2, w, label="w", linewidth=5)

    for C in new_cones:
        plot_cone_face(ax2, C, alpha=0.22)

    set_equal_axes(ax2, all_vecs)

    plt.tight_layout()
    plt.show()

    print("Chosen subdivision ray w =", w)
    print("Barycentric coordinates of w =", barycentric_coordinates(w, cone))
    print()
    print("Original determinant:", abs_det(cone))
    print()

    for i, C in enumerate(new_cones, start=1):
        print(f"New cone {i}:")
        print("  generators:", C)
        print("  determinant:", abs_det(C))
        print()

# subdisvions happens here

def resolve_cone(cone, max_steps=100):
    """
    Recursively subdivide until every cone has determinant 1,
    or until no ray is found.
    """
    active = [cone]
    smooth = []
    stuck = []

    steps = 0

    while active and steps < max_steps:
        steps += 1

        current = active.pop()
        d = abs_det(current)

        if d == 1:
            smooth.append(current)
            continue

        w = choose_subdivision_ray(current)

        if w is None:
            stuck.append(current)
            continue

        new_cones = stellar_subdivide(current, w)

        for C in new_cones:
            if abs_det(C) != 0:
                active.append(C)

    return smooth, stuck


def visualize_final_cones(cones):
    """
    Visualize a list of final cones.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_title("Final Cone Subdivision")

    all_vecs = []

    for C in cones:
        for v in C:
            all_vecs.append(v)
        plot_cone_face(ax, C, alpha=0.20)

    # Plot unique rays
    unique_rays = list(dict.fromkeys(all_vecs))

    for i, v in enumerate(unique_rays, start=1):
        plot_vector(ax, v, label=f"r{i}", linewidth=2)

    set_equal_axes(ax, unique_rays)

    plt.tight_layout()
    plt.show()



#Toy Example, change it to test your own, works better with primitive as it will go down to det=1

if __name__ == "__main__":


    cone = (
    (1, 0, 0),
    (0, 1, 0),
    (1, 1, 2),

    )

    print("Original cone:")
    print("  generators:", cone)
    print("  determinant:", abs_det(cone))
    print("  HNF:")
    print(hnf_of_cone(cone))
    print()

    print("Lattice points in the fundamental parallelepiped:")
    points = lattice_points_in_parallelepiped(cone)

    for p in points:
        print(" ", p, "barycentric =", barycentric_coordinates(p, cone))

    print()

    visualize_one_subdivision(cone)

    smooth, stuck = resolve_cone(cone, max_steps=100)

    print("Number of smooth cones:", len(smooth))
    print("Number of stuck cones:", len(stuck))

    if smooth:
        visualize_final_cones(smooth)

    if stuck:
        print()
        print("Stuck cones:")
        for C in stuck:
            print(C, "det =", abs_det(C))