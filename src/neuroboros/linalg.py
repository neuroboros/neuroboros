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
from joblib import Parallel, cpu_count, delayed
from scipy.linalg import LinAlgError, eigh, polar, svd
from scipy.stats import zscore

from .ensemble import kfold_bagging


def safe_svd(X, remove_mean=True):
    """
    Singular value decomposition without occasional LinAlgError crashes.

    The default ``lapack_driver`` of ``scipy.linalg.svd`` is ``'gesdd'``,
    which occasionally crashes even if the input matrix is not singular.
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
        U, s, Vt = svd(X, full_matrices=False, lapack_driver="gesvd")

    return U, s, Vt


def safe_polar(a, side="left"):
    """
    Polar decomposition without occasional LinAlgError crashes.

    The default ``lapack_driver`` of ``scipy.linalg.polar`` is ``'gesdd'``,
    which occasionally crashes even if the input matrix is not singular.
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
        if side == "right":
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


def _ensemble_lstsq_chunk(X, Y, indices_li):
    n_samples, n_features = X.shape
    n_targets = Y.shape[1]

    beta = np.zeros((n_features, n_targets))
    Yhat = np.zeros((n_samples, n_targets))
    counts = np.zeros((n_samples,))

    for train_idx, test_idx in indices_li:
        b = np.linalg.lstsq(X[train_idx], Y[train_idx], rcond=None)[0]
        beta += b
        Yhat[test_idx] += X[test_idx] @ b
        counts[test_idx] += 1

    beta /= len(indices_li)
    Yhat /= counts[:, np.newaxis]

    Yhat0 = (np.sum(Y, axis=0, keepdims=True) - Y) / (n_samples - 1)
    ss0 = np.sum((Y - Yhat0) ** 2, axis=0)
    ss = np.sum((Y - Yhat) ** 2, axis=0)
    R2 = 1 - ss / ss0

    r = np.mean(zscore(Yhat, axis=0) * zscore(Y, axis=0), axis=0)

    ss0 = np.sum((Y - Yhat0) ** 2, axis=0)
    ss = np.sum((Y - Yhat) ** 2, axis=0)
    R2 = 1 - ss / ss0
    r = np.mean(zscore(Yhat, axis=0) * zscore(Y, axis=0), axis=0)

    return beta, Yhat, R2, r


def ensemble_lstsq(X, Y, n_folds=5, n_perms=20, seed=0, n_jobs=1):
    """
    Linear regression with k-fold bagging.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The design matrix.
    Y : ndarray of shape (n_samples, n_targets)
        The target matrix.
    n_folds : int, default=5
        Number of folds.
    n_perms : int, default=20
        Number of permutations.
    seed : int, default=0
        Random seed for the random number generator.

    Returns
    -------
    beta : ndarray of shape (n_features, n_targets)
        The regression coefficients.
    Yhat : ndarray of shape (n_samples, n_targets)
        The predicted values (out-of-bag cross-validation).
    R2 : ndarray of shape (n_targets,)
        The R-squared values (cross-validated).
    r : ndarray of shape (n_targets,)
        The correlation coefficients between the predicted and actual values.
    """
    assert X.shape[0] == Y.shape[0]
    n_samples, n_features = X.shape
    n_targets = Y.shape[1]

    indices_li = kfold_bagging(n_samples, n_folds=n_folds, n_perms=n_perms, seed=seed)
    if n_jobs == 1:
        beta, Yhat, R2, r = _ensemble_lstsq_chunk(X, Y, indices_li)
    else:
        if n_jobs < 0:
            n_jobs = int(cpu_count() + n_jobs + 1)
        n_chunks = min(n_jobs, n_targets)
        Ys = np.array_split(Y, n_chunks, axis=1)
        with Parallel(n_jobs=n_jobs) as parallel:
            results = parallel(
                delayed(_ensemble_lstsq_chunk)(X, Y_, indices_li) for Y_ in Ys
            )
        beta = np.concatenate([result[0] for result in results], axis=1)
        Yhat = np.concatenate([result[1] for result in results], axis=1)
        R2 = np.concatenate([result[2] for result in results], axis=0)
        r = np.concatenate([result[3] for result in results], axis=0)

    return beta, Yhat, R2, r
