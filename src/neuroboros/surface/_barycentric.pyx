import warnings
import numpy as np
from scipy.spatial import cKDTree

cimport numpy as np


def barycentric_weights(vecs, coords):
    """Compute the weights for barycentric interpolation.

    Parameters
    ----------
    vecs : ndarray of shape (6, 3)
        The 6 vectors used to compute barycentric weights.
        a, e1, e2,
        np.cross(e1, e2),
        np.cross(e2, a),
        np.cross(a, e1)
    coords : ndarray of shape (3, )

    Returns
    -------
    (w, u, v, t) : tuple of float
        ``w``, ``u``, and ``v`` are the weights of the three vertices of the
        triangle, respectively. ``t`` is the scale that needs to be multiplied
        to ``coords`` to make it in the same plane as the three vertices.
    """
    det = coords[0] * vecs[3, 0] + coords[1] * vecs[3, 1] + coords[2] * vecs[3, 2]
    if det == 0:
        if vecs[3, 0] == 0 and vecs[3, 1] == 0 and vecs[3, 2] == 0:
            warnings.warn("Zero cross product of two edges: "
                          "The three vertices are in the same line.")
        else:
            print(vecs[3])
        y = coords - vecs[0]
        u, v = np.linalg.lstsq(vecs[1:3].T, y, rcond=None)[0]
        t = 1.
    else:
        uu  = coords[0] * vecs[4, 0] + coords[1] * vecs[4, 1] + coords[2] * vecs[4, 2]
        vv  = coords[0] * vecs[5, 0] + coords[1] * vecs[5, 1] + coords[2] * vecs[5, 2]
        u = uu / det
        v = vv / det
        tt = vecs[0, 0] * vecs[3, 0] + vecs[0, 1] * vecs[3, 1] + vecs[0, 2] * vecs[3, 2]
        t = tt / det
    w = 1. - (u + v)
    return w, u, v, t


def barycentric_weights_multi_faces_multi_points(
        np.ndarray vecs, np.ndarray coords,
        v2f, tree, double eps=5e-9):
    # eps up to 3.09e-10 for ds002330_02
    # changed from 1e-10 to 5e-10 on 2022-09-03
    # eps up to 3.46e-9 for ds002330_03
    # changed from 5e-10 to 5e-9 on 2022-09-06
    cdef size_t i, j, k, c, nv, kk, max_kk
    cdef double w, u, v, t, m, mm
    nv = coords.shape[0]
    f_indices = np.zeros((nv, ), dtype=np.int64)
    weights = np.zeros((nv, 3), dtype=np.float64)
    max_kk = tree.data.shape[0]
    for i in range(nv):
        mm = -1
        kk = 1
        while mm < -eps:
            c = tree.query(coords[i], [kk])[1]
            ff = v2f[c]
            for f in ff:
                w, u, v, t = barycentric_weights(vecs[f], coords[i])
                m = min(min(w, u), v)
                if m > mm and t > 0:
                    f_indices[i] = f
                    weights[i, 0] = w
                    weights[i, 1] = u
                    weights[i, 2] = v
                    mm = m
            kk += 1
            if kk > max_kk:
                print(mm, weights[i])
                break

    return f_indices, weights
