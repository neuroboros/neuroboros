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


def surface_subdivision(coords, faces, n_div):
    new_coords, new_faces = [], []

    count = coords.shape[0]
    edges = []
    e_mapping = {}
    for f in faces:
        for a, b in [[f[0], f[1]], [f[0], f[2]], [f[1], f[2]]]:
            if (a, b) in edges or (b, a) in edges:
                continue
            edges.append((a, b))
            for i in range(1, n_div):
                c = (coords[a] * i + coords[b] * (n_div - i)) / n_div
                c /= np.linalg.norm(c)
                new_coords.append(c)

                e_mapping[(a, b, i)] = count
                count += 1

    for f in faces:
        mapping = {
            (n_div, 0, 0): f[0],
            (0, n_div, 0): f[1],
            (0, 0, n_div): f[2],
        }

        for ii, jj in [[0, 1], [0, 2], [1, 2]]:
            a, b = f[[ii, jj]]
            kk = 3 - ii - jj
            for step in range(1, n_div):
                if (a, b, step) in e_mapping:
                    val = e_mapping[(a, b, step)]
                elif (b, a, n_div - step) in e_mapping:
                    val = e_mapping[(b, a, n_div - step)]
                else:
                    raise ValueError

                key = [0, 0, 0]
                key[ii] = step
                key[jj] = n_div - step
                key = tuple(key)
                mapping[key] = val

        for i in range(n_div)[::-1]:
            for j in range(n_div + 1 - i):
                k = n_div - i - j
                if (i, j, k) not in mapping:
                    mapping[(i, j, k)] = count
                    count += 1
                    c = np.sum(coords[f] * (np.array([i, j, k])[:, np.newaxis] / n_div), axis=0)
                    c /= np.linalg.norm(c)
                    new_coords.append(c)

        nf = triangle_subdivision(n_div)
        nf = [[mapping[v] for v in f] for f in nf]
        new_faces += nf
    new_coords = np.array(new_coords)
    new_faces = np.array(new_faces)
    coords = np.concatenate([coords, new_coords], axis=0)

    return coords, new_faces
