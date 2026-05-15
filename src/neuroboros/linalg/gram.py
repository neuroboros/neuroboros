import numpy as np
from scipy.linalg import eigh


def gram(X, split=None, reduce="sum", remove_mean=True):
    """
    Compute the Gram matrix (``X @ X.T``).

    Parameters
    ----------
    X : ndarray of shape (N, M)
        Input matrix.
    split : int or 1-D array or None, default=None
        If given, passed directly to ``numpy.array_split`` as
        ``indices_or_sections`` along axis=1: an int splits into that many
        roughly equal chunks; a 1-D array specifies the column indices at
        which to split.  Accumulating chunk by chunk reduces peak memory usage.
    reduce : {'sum', 'stack', 'list'}, default='sum'
        How to combine the per-chunk Gram matrices when ``split`` is given.
        ``'sum'`` accumulates them into a single ``(N, N)`` matrix;
        ``'stack'`` returns an ``(n_chunks, N, N)`` array;
        ``'list'`` returns a list of ``(N, N)`` arrays.
        Ignored when ``split`` is None.
    remove_mean : bool, default=True
        Whether to subtract the column mean before computing the Gram matrix.

    Returns
    -------
    gram : ndarray of shape (N, N) or (n_chunks, N, N), or list of ndarray
        The Gram matrix, or per-chunk Gram matrices when ``split`` is given
        and ``reduce`` is not ``'sum'``.
    """
    if remove_mean:
        mean = X.mean(axis=0, keepdims=True)
        if not np.allclose(mean, 0, atol=1e-10):
            X = X - mean
    if split is None:
        return X @ X.T

    grams = [chunk @ chunk.T for chunk in np.array_split(X, split, axis=1)]

    if reduce == "sum":
        return sum(grams)
    elif reduce == "stack":
        return np.stack(grams)
    elif reduce == "list":
        return grams
    else:
        raise ValueError(f"reduce must be 'sum', 'stack', or 'list', got {reduce!r}")


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
