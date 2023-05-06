"""
===================================================
Linear algebra utilities (:mod:`neuroboros.linalg`)
===================================================

.. currentmodule:: neuroboros.linalg

.. autosummary::
    :toctree:

    safe_svd - Singular value decomposition without occasional LinAlgError crashes.
    safe_polar - Polar decomposition without occasional LinAlgError crashes.
    gram_pca - Principal component analysis based on the Gram matrix.

"""
import numpy as np
from scipy.linalg import svd, eigh, polar, LinAlgError


def safe_svd(X, remove_mean=True):
    """
    Singular value decomposition without occasional LinAlgError crashes.

    The default ``lapack_driver`` of ``scipy.linalg.svd`` is ``'gesdd'``,
    which occassionaly crashes even if the input matrix is not singular.
    This function automatically handles the ``LinAlgError`` when it's
    raised and switches to the ``'gesvd'`` driver in this case.

    The input matrix ``X`` is factorized as ``U @ np.diag(s) @ Vt``.

    Parameters
    ----------
    X : ndarray of shape (M, N)
        The matrix to be decomposed in NumPy array format.
    remove_mean : bool, default=True
        Whether to subtract the mean of each column before the actual SVD
        (True) or not (False). Setting ``remove_mean=True`` is helpful when
        the SVD is used to perform PCA.

    Returns
    -------
    U : ndarray of shape (M, K)
        Unitary matrix.
    s : ndarray of shape (K,)
        The singular values.
    Vt : ndarray of shape (K, N)
        Unitary matrix.
    """
    if remove_mean:
        mean = X.mean(axis=0, keepdims=True)
        if not np.allclose(mean, 0, atol=1e-10):
            X = X - mean

    try:
        U, s, Vt = svd(X, full_matrices=False)
    except LinAlgError:
        U, s, Vt = svd(X, full_matrices=False, lapack_driver='gesvd')

    return U, s, Vt


def safe_polar(a, side='left'):
    """
    Polar decomposition without occasional LinAlgError crashes.

    The default ``lapack_driver`` of ``scipy.linalg.polar`` is ``'gesdd'``,
    which occassionaly crashes even if the input matrix is not singular.
    This function automatically handles the ``LinAlgError`` when it's
    raised and switches to the ``'gesvd'`` driver in this case.

    The input matrix ``a`` is factorized as ``u @ p`` (or ``p @ u`` when
    ``side='right'``).

    Parameters
    ----------
    a : ndarray of shape (M, N)
        The matrix to be decomposed in NumPy array format.
    side : {'left', 'right'}, default='left'
        Whether to return ``u @ p`` (``side='left'``) or ``p @ u``
        (``side='right'``).

    Returns
    -------
    u : ndarray of shape (M, M) or (N, N)
        Unitary matrix (rotation and reflection).
    p : ndarray of shape (M, N) or (N, M)
        Hermitian positive semi-definite matrix (scaling and shearing).

    """

    try:
        u, p = polar(a, side=side)
    except Exception as e:
        w, s, vh = safe_svd(a)
        u = w.dot(vh)
        if side == 'right':
            # a = up
            p = (vh.T.conj() * s).dot(vh)
        else:
            # a = pu
            p = (w * s).dot(w.T.conj())
    return u, p


def gram_pca(gram, tol=1e-7):
    """
    Principal component analysis (PCA) based on the Gram matrix.

    Parameters
    ----------
    gram : ndarray of shape (N, N)
        The Gram matrix to be decomposed in NumPy array format.
    tol : float, default=1e-7
        Tolerance for the eigenvalues to be considered positive.

    Returns
    -------
    PCs : ndarray of shape (N, N - 1)
        The principal components (PCs) derived from the Gram matrix.
    """

    w, v = eigh(gram, lower=False)
    assert np.all(w > -tol)
    w[w < 0] = 0
    U = v[:, ::-1][:, :-1]
    s = np.sqrt(w[::-1][:-1])
    PCs = U * s[np.newaxis]
    return PCs
