import os
import numpy as np


def triangle_subdivision(n_div):
    faces = []
    for i in range(1, n_div + 1)[::-1]:
        for j in range(n_div + 1 - i):
            k = n_div - i - j
            f = [(i, j, k), (i - 1, j + 1, k), (i - 1, j, k + 1)]
            faces.append(f)
    for i in range(1, n_div + 1)[::-1]:
        for j in range(1, n_div + 1 - i):
            k = n_div - i - j
            f = [(i, j, k), (i - 1, j, k + 1), (i, j - 1, k + 1)]
            faces.append(f)
    return faces


def subdivide_edges(coords, faces, n_div):
    n_edges = faces.shape[0] * 3 // 2
    nv_new = n_edges * (n_div - 1)
    new_coords = np.zeros((nv_new, 3), dtype=coords.dtype)

    nv = coords.shape[0]
    count = 0
    edges = set()
    e_mapping = {}
    for f in faces:
        for a, b in [[f[0], f[1]], [f[0], f[2]], [f[1], f[2]]]:
            e = (a, b) if a < b else (b, a)
            if e in edges:
                continue
            edges.add(e)
            for i in range(1, n_div):
                c = (coords[a] * i + coords[b] * (n_div - i)) / n_div
                # c /= np.linalg.norm(c)
                # new_coords.append(c)
                new_coords[count] = c
                e_mapping[(e[0], e[1], i)] = count + nv
                count += 1

    return new_coords, e_mapping


def subdivide_inside(coords, faces, e_mapping, n_div, count_base):
    # new_coords, new_faces = [], []
    nv_new = (n_div - 1) * (n_div - 2) // 2 * len(faces)
    new_coords = np.zeros((nv_new, 3), dtype=coords.dtype)
    count = 0
    new_faces = []
    nf_base = triangle_subdivision(n_div)

    for f in faces:
        mapping = {
            (n_div, 0, 0): f[0],
            (0, n_div, 0): f[1],
            (0, 0, n_div): f[2],
        }

        for ii, jj in [[0, 1], [0, 2], [1, 2]]:
            a, b = f[[ii, jj]]
            aa, bb = (a, b) if a < b else (b, a)
            for step in range(1, n_div):
                val = e_mapping[aa, bb, step]
                key = [0, 0, 0]
                key[ii] = step
                key[jj] = n_div - step
                mapping[tuple(key)] = val

        for i in range(n_div)[::-1]:
            for j in range(n_div + 1 - i):
                k = n_div - i - j
                if (i, j, k) not in mapping:
                    mapping[(i, j, k)] = count + count_base
                    c = np.sum(coords[f] * (np.array([i, j, k])[:, np.newaxis] / n_div), axis=0)
                    # c /= np.linalg.norm(c)
                    # new_coords.append(c)
                    new_coords[count] = c
                    count += 1

        nf = [[mapping[v] for v in f] for f in nf_base]
        new_faces += nf

    # new_coords = np.array(new_coords)
    new_faces = np.array(new_faces)
    return new_coords, new_faces


def surface_subdivision(coords, faces, n_div):
    edge_coords, e_mapping = subdivide_edges(coords, faces, n_div)
    count = coords.shape[0] + edge_coords.shape[0]
    inside_coords, new_faces = subdivide_inside(coords, faces, e_mapping, n_div, count)
    print(coords.shape, edge_coords.shape, inside_coords.shape)
    new_coords = np.concatenate([coords, edge_coords, inside_coords], axis=0)
    return new_coords, new_faces
#     new_coords, new_faces = [], []

#     # count = coords.shape[0]

#     new_coords_edges, e_mapping = subdivide_edges(coords, faces, n_div)


#     new_coords = np.array(new_coords)
#     new_faces = np.array(new_faces)
#     coords = np.concatenate([coords, new_coords], axis=0)

#     return coords, new_faces
