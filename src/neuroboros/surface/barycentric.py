import numpy as np
import scipy.sparse as sparse

from ._barycentric import barycentric_weights, barycentric_weights_multi_faces_multi_points


def barycentric_vectors(f_coords):
    """Compute the 6 vectors used in barycentric interpolation.

    Parameters
    ----------
    f_coords : ndarray of shape (3, 3) or (nf, 3, 3)

    Returns
    -------
    vecs : ndarray of shape (6, 3) or (nf, 6, 3)

    """
    a = f_coords[..., 0, :]
    e1 = f_coords[..., 1, :] - f_coords[..., 0, :]
    e2 = f_coords[..., 2, :] - f_coords[..., 0, :]
    vecs = np.stack([
        a, e1, e2,
        np.cross(e1, e2),
        np.cross(e2, a),
        np.cross(a, e1)],
        axis=-2)
    return vecs


def barycentric(vecs, coords, v2f, tree, faces=None, nv=None, eps=5e-9, return_sparse=True):
    f_indices, weights = barycentric_weights_multi_faces_multi_points(vecs, coords, v2f, tree, eps)
    if not return_sparse:
        return f_indices, weights

    assert faces is not None
    if nv is None:
        nv = faces.max() + 1
    mat = sparse.lil_matrix((nv, coords.shape[0]))
    for i, (f_idx, w) in enumerate(zip(f_indices, weights)):
        idx = faces[f_idx]
        mat[idx, i] = w
    mat = mat.tocsr()
    return mat
