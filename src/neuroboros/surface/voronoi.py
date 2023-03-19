import heapq
from datetime import datetime
import numpy as np
import scipy.sparse as sparse
from scipy.spatial.distance import cdist


def subdivide_edges(coords, faces, n_div):
    n_edges = faces.shape[0] * 3 // 2
    nv_new = n_edges * (n_div - 1)
    new_coords = np.zeros((nv_new, 3), dtype=coords.dtype)
    print(new_coords.shape)
    neighbors = {}

    nv = coords.shape[0]
    count = 0
    e_mapping = {}
    arng = np.arange(1, n_div)
    for f in faces:
        for i in f:
            if i not in neighbors:
                neighbors[i] = {}

        steiner = []
        for i, j, k in [[0, 1, 2], [1, 2, 0], [2, 0, 1]]:
            a, b, c = f[[i, j, k]]
            e = (a, b) if a < b else (b, a)
            if e not in e_mapping:
                cc = (coords[[e[1]]] * arng[:, np.newaxis] + coords[e[0]] * (n_div - arng[:, np.newaxis])) / n_div
                new_coords[count:count+n_div-1] = cc
                indices = (count + nv - 1) + arng
                e_mapping[e] = indices
                count += n_div - 1
            else:
                indices = e_mapping[e]
                cc = new_coords[indices - nv]
            steiner.append([cc, indices])

            # connecting points on an edge and the opposite point
            idx2, c2 = c, coords[c]
            for idx1, c1 in zip(indices, cc):
                if idx1 not in neighbors:
                    neighbors[idx1] = {}
                if idx1 not in neighbors[idx2]:
                    d = np.linalg.norm(c1 - c2)
                    neighbors[idx2][idx1] = d
                    neighbors[idx1][idx2] = d

        # connecting points on an edge and points on another edge
        for i, j in [[0, 1], [1, 2], [2, 0]]:
            cc1, indices1 = steiner[i]
            cc2, indices2 = steiner[j]
            for idx1, c1 in zip(indices1, cc1):
                for idx2, c2 in zip(indices2, cc2):
                    if idx1 not in neighbors[idx2]:
                        d = np.linalg.norm(c1 - c2)
                        neighbors[idx2][idx1] = d
                        neighbors[idx1][idx2] = d

        # connecting points of the original triangle
        for i in f:
            for j in f:
                if i != j and i not in neighbors[j]:
                    d = np.linalg.norm(coords[i] - coords[j])
                    neighbors[i][j] = d
                    neighbors[j][i] = d

    return new_coords, e_mapping, neighbors


def dijkstra_distances(nv, candidates, neighbors, max_dist=None):
    dists = np.full((nv, ), np.inf)
    finished = np.zeros((nv, ), dtype=bool)
    for d, idx in candidates:
        dists[idx] = d

    while candidates:
        d, idx = heapq.heappop(candidates)
        if finished[idx]:
            continue

        for nbr, nbr_d in neighbors[idx].items():
            new_d = d + nbr_d
            if max_dist is not None and new_d > max_dist:
                continue
            if new_d < dists[nbr]:
                dists[nbr] = new_d
                heapq.heappush(candidates, (new_d, nbr))
        finished[idx] = True

    return dists


def subdivision_voronoi(coords, faces, e_mapping, neighbors, f_indices, weights, max_dist=None):
    nv = coords.shape[0]
    assert len(np.unique(f_indices)) == len(f_indices)
    if max_dist is None:
        max_dist = 4.0 * np.sqrt(10242 / f_indices.shape[0]) + 2.0
    log_step = (f_indices.shape[0] // 100 + 1)
    nn = np.full((nv, ), -1, dtype=int)
    nnd = np.full((nv, ), np.inf)
    while np.any(np.isinf(nnd)):
        for i, (f_idx, w) in enumerate(zip(f_indices, weights)):
            cc = w @ coords[faces[f_idx]]
            a, b, c = sorted(faces[f_idx])
            indices = np.concatenate([e_mapping[(a, b)], e_mapping[(a, c)], e_mapping[(b, c)], [a, b, c]])
            dd = cdist(cc[np.newaxis], coords[indices], 'euclidean').ravel()
            candidates = []
            for d, idx in zip(dd, indices):
                heapq.heappush(candidates, (d, idx))
            d = dijkstra_distances(nv, candidates, neighbors, max_dist=max_dist)
            mask = d < nnd
            nn[mask] = i
            nnd[mask] = d[mask]
            if i % log_step == 0:
                print(datetime.now(), i, mask.sum(), np.isfinite(d).sum(), d.shape, d.max(), d.min(), len(candidates), np.isinf(nnd).sum())
        max_dist *= 1.5
    print(nnd.max())
    return nn, nnd


def native_voronoi(coords, faces, e_mapping, neighbors):
    nv = coords.shape[0]
    nn = np.full((nv, ), -1, dtype=int)
    nnd = np.full((nv, ), np.inf)
    max_dist = np.max([
        np.linalg.norm(coords[faces[:, 0]] - coords[faces[:, 1]], axis=1).max(),
        np.linalg.norm(coords[faces[:, 1]] - coords[faces[:, 2]], axis=1).max(),
        np.linalg.norm(coords[faces[:, 2]] - coords[faces[:, 0]], axis=1).max(),])
    max_dist = max_dist * 0.5 + 1e-3
    print(max_dist)

    seeds = []

    for f in faces:
        f = np.sort(f)
        cc1 = coords[f]
        a, b, c = f
        for i, e in enumerate([(b, c), (a, c), (a, b)]):
            indices = e_mapping[e]
            cc2 = coords[indices]
            d = cdist(cc1, cc2)

            min_idx = np.argmin(d, axis=0)
            # wide short triangle
            if np.any(min_idx == i):
                seeds.append(f[i])
            d = d.min(axis=0)
            mask = (d < nnd[indices])

            nnd[indices[mask]] = d[mask]
            nn[indices[mask]] = f[min_idx[mask]]
        nn[f] = f
        nnd[f] = 0.
    print(np.isfinite(nnd).mean(), nnd.max())

    seeds = np.unique(seeds)
    print(len(seeds), seeds[:10])
    for i, seed in enumerate(seeds):
        candidates = [(0.0, seed)]
        d = dijkstra_distances(nv, candidates, neighbors, max_dist=max_dist)
        mask = d < nnd
        nn[mask] = seed
        nnd[mask] = d[mask]
        if i % 10000 == 0:
            print(datetime.now(), i, seed, mask.sum(), nnd.max(), np.isfinite(d).sum(), d.shape, d.max(), d.min(), np.isinf(nnd).sum())

    return nn, nnd


def inverse_face_mapping(f_indices, weights, coords, faces):
    f_inv = {}
    for i, f_idx in enumerate(f_indices):
        if f_idx not in f_inv:
            f_inv[f_idx] = []
        f_inv[f_idx].append([i, weights[i] @ coords[faces[f_idx]]])
    return f_inv


def split_triangle(t_div):
    ww1 = []
    for i in range(t_div):
        for j in range(t_div - i):
            k = t_div - i - j - 1
            ww1.append([i, j, k])
    ww1 = (np.array(ww1) + 1./3) / t_div
    ww2 = []
    for i in range(t_div - 1):
        for j in range(t_div - i - 1):
            k = t_div - i - j - 1
            ww2.append([i, j, k])
    ww2 = (np.array(ww2) + np.array([[2/3, 2/3, -1/3]])) / t_div
    ww = np.concatenate([ww1, ww2])
    return ww


def compute_occupation(f_idx, f, coords, indices, nn, nnd, f_inv, ww):
    nn1 = [nn[indices]]
    u = np.unique(nn1)
    if len(u) == 1 and len(f_inv) == 0:
        return {u[0]: np.ones((ww.shape[0], ), dtype=bool)}
    cc1 = [coords[indices]]
    dd1 = [nnd[indices]]
    if f_idx in f_inv:
        for i, c in f_inv[f_idx]:
            cc1.append([c])
            dd1.append([0.])
            nn1.append([i])
    cc1 = np.concatenate(cc1)
    dd1 = np.concatenate(dd1)
    nn1 = np.concatenate(nn1)
    cc2 = ww @ coords[f]
    d = cdist(cc1, cc2, 'euclidean') + dd1[:, np.newaxis]
    idx = np.argmin(d, axis=0)
    occupation = nn1[idx]
    return {u: occupation == u for u in np.unique(occupation)}


def compute_overlap(faces, face_areas, e_mapping, coords, nn, nnd, f_inv, nn2, nnd2, f_inv2, nv1, nv2, t_div=32):
    mat = sparse.lil_matrix((nv1, nv2))
    ww = split_triangle(t_div)
    for f_idx, f in enumerate(faces):
        a, b, c = sorted(faces[f_idx])
        indices = np.concatenate([e_mapping[(a, b)], e_mapping[(a, c)], e_mapping[(b, c)], [a, b, c]])
        uu1 = compute_occupation(f_idx, f, coords, indices, nn, nnd, f_inv, ww)
        uu2 = compute_occupation(f_idx, f, coords, indices, nn2, nnd2, f_inv2, ww)
        for u1, m1 in uu1.items():
            for u2, m2 in uu2.items():
                overlap = np.logical_and(m1, m2).mean()
                if overlap:
                    mat[u1, u2] += overlap * face_areas[f_idx]
    return mat.tocsr()


"""
Similar to https://github.com/mojocorp/geodesic subdivision
"""
