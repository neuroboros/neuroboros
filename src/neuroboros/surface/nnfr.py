import numpy as np
import scipy.sparse as sparse
from scipy.spatial import cKDTree


def nnfr(source_coords, target_coords, reverse=True):
    ns = source_coords.shape[0]
    nt = target_coords.shape[0]
    source_tree = cKDTree(source_coords)
    target_tree = cKDTree(target_coords)

    forward_indices = source_tree.query(target_coords)[1]
    if reverse:
        u, c = np.unique(forward_indices, return_counts=True)
        counts = np.zeros((ns, ), dtype=int)
        counts[u] += c
        remaining = np.setdiff1d(np.arange(ns), u)
        reverse_indices = target_tree.query(source_coords[remaining])[1]
        counts[remaining] += 1

    T = sparse.lil_matrix((ns, nt))
    for t_idx, s_idx in zip(np.arange(nt), forward_indices):
        T[s_idx, t_idx] += 1
    if reverse:
        for t_idx, s_idx in zip(reverse_indices, remaining):
            T[s_idx, t_idx] += 1

    T = T.tocsr()
    t_counts = T.sum(axis=0).A.ravel()
    T = T @ sparse.diags(np.reciprocal(t_counts))

    return T
