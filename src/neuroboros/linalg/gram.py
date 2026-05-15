import numpy as np
from scipy.linalg import eigh


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
