import numpy as np
from scipy.spatial.distance import cdist


def classification(
    U, Uhat, size=None, npc=None, metric='correlation', reduce_func=np.mean
):
    """Movie time point/segment classification accuracy.

    Parameters
    ----------
    U : ndarray, shape (nt, nv)
        The measured response patterns, often the columns are normalized PCs of
        the measured data rather than the original voxels/vertices.
    Uhat : ndarray, shape (nt, nv)
        The predicted response patterns.
    size : int or None, default=None
        The number of time points in each segment. If None, each time point is
        treated as a segment.
    npc : int or None, default=None
        The number of PCs used for classification. If None, all PCs are used.
        If not None, assuming that ``U`` and ``Uhat`` are PCs, the first
        ``npc`` PCs are used for classification.
    metric : str, default='correlation'
        The metric used for calculating the distance between two segments.
        See ``scipy.spatial.distance.cdist`` for the available metrics.
    reduce_func : callable or None, default=np.mean
        The function used for reducing the classification accuracy across
        segments. If None, the raw classification accuracy is returned, where
        the length of the returned array is equal to the number of segments.

    Returns
    -------
    acc : float or ndarray
        The classification accuracy. If ``reduce_func`` is None, the returned
        value is an ndarray of shape (nc,), where nc is the number of segments.
        Otherwise, the returned value is a float.
    """
    U, Uhat = U[:, :npc], Uhat[:, :npc]
    nt, nv = U.shape
    if size is None:
        nc = nt
        Y, Yhat = U, Uhat
    else:
        nc = nt // size
        Y = U[: nc * size].reshape(nc, size, nv).reshape(nc, size * nv)
        Yhat = Uhat[: nc * size].reshape(nc, size, nv).reshape(nc, size * nv)
    d = cdist(Y, Yhat, metric=metric)
    arng = np.arange(nc)
    acc = np.argmin(d, axis=1) == arng
    if reduce_func is not None:
        acc = reduce_func(acc)
    return acc
