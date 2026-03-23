from datetime import datetime

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform


def convex_hull_faces(coords):
    """
    Compute the faces of the convex hull of a set of points.

    Parameters
    ----------
    coords : array-like, shape (n_points, 3)
        The coordinates of the points.

    Returns
    -------
    faces : array-like, shape (n_faces, 3)
        The indices of the vertices that form each face of the convex hull.
    """
    hull = ConvexHull(coords)
    return hull.simplices


def orient_faces(coords, faces):
    """
    Orient the faces of a surface so that the normals point outward.

    Parameters
    ----------
    coords : array-like, shape (n_vertices, 3)
        The coordinates of the vertices of the surface.
    faces : array-like, shape (n_faces, 3)
        The indices of the vertices that form each face of the surface.

    Returns
    -------
    faces : array-like, shape (n_faces, 3)
        The oriented faces of the surface.
    """
    for i, (a, b, c) in enumerate(faces):
        p = np.cross(coords[b] - coords[a], coords[c] - coords[b]) @ coords[
            [a, b, c]
        ].mean(axis=0)
        if p < 0:
            faces[i] = [c, b, a]
    return faces


def optimize_faces(faces, mat, verbose=True, seed=0, return_stats=False):
    """
    Optimize the faces of a surface to minimize the total edge length.
    For two faces ABC and ABD, if the edge CD is shorter than AB, we can flip
    the edge to get faces ACD and BCD.

    Parameters
    ----------
    faces : array-like, shape (n_faces, 3)
        The indices of the vertices that form each face of the surface.
    mat : array-like, shape (n_vertices, n_vertices)
        The distance matrix between the vertices.
    verbose : bool, default=True
        Whether to print progress information.
    return_stats : bool, default=False
        If True, also return the total edge length and number of invalid
        topology cases.

    Returns
    -------
    faces : array-like, shape (n_faces, 3)
        The optimized faces of the surface.
    total_distance : float
        Sum of all edge lengths in the final mesh. Only returned if
        ``return_stats=True``.
    n_invalid : int
        Number of invalid topology cases. Only returned if
        ``return_stats=True``.
    """
    rng = np.random.default_rng(seed)

    edge2faces = {}  # edge -> [face1, face2]
    for i, f in enumerate(faces):
        f = np.sort(f)
        for e in [(f[0], f[1]), (f[0], f[2]), (f[1], f[2])]:
            if e not in edge2faces:
                edge2faces[e] = []
            edge2faces[e].append(i)

    n_iter = 0
    while True:
        count = 0
        invalid = []
        edges = list(edge2faces)
        rng.shuffle(edges)
        for ii, edge in enumerate(edges):
            i, j = edge2faces[edge]
            f1, f2 = faces[i], faces[j]
            a, b = edge
            c = f1[np.isin(f1, edge, invert=True)]
            d = f2[np.isin(f2, edge, invert=True)]
            assert len(c) == 1
            assert len(d) == 1
            c, d = c[0], d[0]

            alt_edge = tuple(sorted([c, d]))
            if alt_edge in edge2faces:
                # if verbose:
                #     print(a, b, c, d, (i, j), edge2faces[alt_edge])
                #     print(faces[[i, j]], faces[edge2faces[alt_edge]])
                # invalid.append([edge, alt_edge])
                # print(count_tetrahedra(faces))
                continue

            assert alt_edge not in edge2faces

            if mat[c, d] < mat[a, b]:
                # print(f"{i}, {faces[i]}, {j}, {faces[j]}")
                faces[i] = sorted([c, d, a])
                faces[j] = sorted([c, d, b])

                edge2faces[alt_edge] = [i, j]

                # Only 2 of the 4 remaining edges need update
                key = tuple(sorted([b, c]))
                edge2faces[key][edge2faces[key].index(i)] = j
                key = tuple(sorted([a, d]))
                edge2faces[key][edge2faces[key].index(j)] = i

                del edge2faces[edge]

                count += 1

        if not count:
            if invalid:
                print(f"{n_iter:5d}, {invalid} invalid")
                for edge, alt_edge in invalid:
                    print(edge2faces[edge], edge2faces[alt_edge])
                    print(faces[edge2faces[edge]], faces[edge2faces[alt_edge]])
            break
        n_iter += 1

        if verbose:
            print(datetime.now(), f"{n_iter:5d}, {count}")

    if return_stats:
        total_distance = sum(mat[a, b] for a, b in edge2faces)
        n_invalid = len(invalid)
        return faces, total_distance, n_invalid
    return faces


def count_tetrahedra(faces):
    """
    Count the number of tetrahedra in a triangulated surface mesh.

    A tetrahedron is defined by 4 vertices such that all 4 triangular faces
    connecting them (ABC, ABD, ACD, BCD) are present in the mesh.

    Parameters
    ----------
    faces : array-like, shape (n_faces, 3)
        The indices of the vertices that form each face of the surface.

    Returns
    -------
    n_tetrahedra : int
        The number of tetrahedra found in the mesh.
    """
    from collections import defaultdict

    faces = np.asarray(faces)
    face_set = {frozenset(f) for f in faces}

    neighbors = defaultdict(set)
    for a, b, c in faces:
        neighbors[a].update([b, c])
        neighbors[b].update([a, c])
        neighbors[c].update([a, b])

    tetrahedra = set()
    for a, b, c in faces:
        common = (neighbors[a] & neighbors[b] & neighbors[c]) - {a, b, c}
        for d in common:
            if (
                frozenset([a, b, d]) in face_set
                and frozenset([a, c, d]) in face_set
                and frozenset([b, c, d]) in face_set
            ):
                tetrahedra.add(frozenset([a, b, c, d]))

    return len(tetrahedra)


def optimize_faces_workflow(
    coords, faces=None, mat=None, verbose=True, seed=0, return_stats=False
):
    """
    Optimize the faces of a surface to minimize the total edge length.
    If faces are not provided, compute the convex hull of the coordinates to
    get the initial faces.

    Parameters
    ----------
    coords : array-like, shape (n_vertices, 3)
        The coordinates of the vertices of the surface.
    faces : array-like, shape (n_faces, 3), optional
        The indices of the vertices that form each face of the surface. If not
        provided, the convex hull of the coordinates will be used.
    mat : array-like, shape (n_vertices, n_vertices)
        The distance matrix between the vertices. If not provided, Euclidean
        distances will be computed from the coordinates.
    verbose : bool, default=True
        Whether to print progress information.
    seed : int, default=0
        The random seed for reproducibility.
    return_stats : bool, default=False
        If True, also return the total edge length and number of invalid
        topology cases.

    Returns
    -------
    faces : array-like, shape (n_faces, 3)
        The optimized faces of the surface.
    """
    if faces is None:
        faces = convex_hull_faces(coords)
        print(faces.shape)
        print(count_tetrahedra(faces))
    if mat is None:
        mat = squareform(pdist(coords))
    if return_stats:
        faces, total_distance, n_invalid = optimize_faces(
            faces, mat, verbose=verbose, seed=seed, return_stats=return_stats
        )
    print(count_tetrahedra(faces))
    faces = orient_faces(coords, faces)
    if return_stats:
        return faces, total_distance, n_invalid
    return faces
