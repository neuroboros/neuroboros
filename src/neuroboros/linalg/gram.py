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


def gram_pca(gram, tol=1e-7, return_us=False):
    """
    Principal component analysis (PCA) based on the Gram matrix.

    Parameters
    ----------
    gram : ndarray of shape (N, N)
        The Gram matrix to be decomposed in NumPy array format.
    tol : float, default=1e-7
        Tolerance for the eigenvalues to be considered positive.
    return_us : bool, default=False
        If True, also return the eigenvectors ``U`` and sqrt-eigenvalues ``s``
        in addition to the PCs.

    Returns
    -------
    PCs : ndarray of shape (N, N - 1)
        The principal components (PCs) derived from the Gram matrix.
        Only returned when ``return_us=False``.
    U : ndarray of shape (N, N - 1)
        Eigenvectors in descending eigenvalue order. Only returned when
        ``return_us=True``.
    s : ndarray of shape (N - 1,)
        Square roots of eigenvalues in descending order. Only returned when
        ``return_us=True``.
    """

    w, v = eigh(gram, lower=False)
    assert np.all(w > -tol)
    w[w < 0] = 0
    U = v[:, ::-1][:, :-1]
    s = np.sqrt(w[::-1][:-1])
    if return_us:
        return U, s
    return U * s[np.newaxis]


def beta2w(beta, gram, tol=1e-7):
    """
    Convert PC-space coefficients to a sample-space weight vector.

    Computes ``w = U @ (beta / s)``, where ``U`` and ``s`` are the
    eigenvectors and sqrt-eigenvalues of ``gram``.  The result can be used
    to recover feature-space weights via ``X.T @ w``.

    Parameters
    ----------
    beta : ndarray of shape (k,)
        Coefficients in PC space, where ``k <= N - 1``.
    gram : ndarray of shape (N, N)
        Gram matrix.
    tol : float, default=1e-7
        Tolerance for the eigenvalues to be considered non-negative.

    Returns
    -------
    w : ndarray of shape (N,)
        Sample-space weight vector.
    """
    U, s = gram_pca(gram, tol=tol, return_us=True)
    k = len(beta)
    return U[:, :k] @ (beta / s[:k])


def beta2beta(beta, gram, X, return_shift=False, tol=1e-7):
    """
    Convert PC-space coefficients to feature-space coefficients.

    The gram matrix is built from the column-centered ``X_c = X - mu``, so
    the feature-space beta is ``X_c.T @ w = X.T @ w - mu * w.sum()``.  The
    intercept shift ``mu @ beta_orig`` can be optionally returned so the
    caller can adjust their PC-space intercept as ``b_orig = b - shift``.

    Parameters
    ----------
    beta : ndarray of shape (k,)
        Coefficients in PC space, where ``k <= N - 1``.
    gram : ndarray of shape (N, N)
        Gram matrix (``X_c @ X_c.T`` where ``X_c = X - mu``).
    X : ndarray of shape (N, M)
        Original data matrix used to compute the Gram matrix. Column means
        are always removed when converting coefficients.
    return_shift : bool, default=False
        If True, also return ``mu @ beta_orig``.
    tol : float, default=1e-7
        Tolerance for the eigenvalues to be considered non-negative.

    Returns
    -------
    beta_orig : ndarray of shape (M,)
        Equivalent coefficients in the original feature space.
    shift : float
        ``mu @ beta_orig``, only returned when ``return_shift=True``.
    """
    w = beta2w(beta, gram, tol=tol)
    beta_orig = X.T @ w
    mu = X.mean(axis=0)
    if not np.allclose(mu, 0, atol=1e-10):
        beta_orig -= mu * w.sum()
    if return_shift:
        return beta_orig, mu @ beta_orig
    return beta_orig
