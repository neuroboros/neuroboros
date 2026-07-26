import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist,cdist
from scipy.stats import zscore

def compute_isc_two_matrix(dms1,dms2,metric="correlation",n_jobs = -1):
    nS,_,nV = dms1.shape
    nSS,_,_ = dms2.shape
    dms1 = dms1.transpose(2,0,1) # reshape into nV,nS,nT
    dms2 = dms2.transpose(2,0,1)
    
    jobs = [delayed(cdist)(dm1,dm2,metric) for (dm1,dm2) in zip(dms1,dms2)]
    with Parallel(n_jobs=n_jobs) as parallel:
        isc = parallel(jobs)
    isc = 1 - np.stack(isc, axis=2).reshape(nS * nSS, nV)
    return isc

def compute_isc_pairwise(dms, metric="correlation", n_jobs=-1):
    jobs = [delayed(pdist)(_, metric) for _ in dms.transpose(2, 0, 1)]
    with Parallel(n_jobs=n_jobs) as parallel:
        isc = parallel(jobs)
    isc = 1 - np.stack(isc, axis=1)
    return isc


def compute_isc_ovr_single(dm, dm_sum, metric="correlation"):
    a, b = dm, dm_sum - dm
    if metric == "correlation":
        a = np.nan_to_num(zscore(a, axis=0))
        b = np.nan_to_num(zscore(b, axis=0))
        return np.mean(a * b, axis=0)
    elif metric == "cosine":
        a = np.nan_to_num(a / np.linalg.norm(a, axis=0, keepdims=True))
        a = np.nan_to_num(b / np.linalg.norm(b, axis=0, keepdims=True))
        return np.mean(a * b, axis=0)
    else:
        raise ValueError(f"Got metric={metric}.")


def compute_isc_ovr(dms, metric="correlation", n_jobs=-1):
    dm_sum = np.sum(dms, axis=0)
    jobs = [delayed(compute_isc_ovr_single)(_, dm_sum, metric) for _ in dms]
    with Parallel(n_jobs=n_jobs) as parallel:
        isc = parallel(jobs)
    isc = np.stack(isc, axis=0)
    return isc


def compute_isc(dms, pairwise=False, metric="correlation", n_jobs=-1):
    """Compute inter-subject correlation (ISC) from data matrices.

    For each vertex, compute either the correlation between all pairs of
    subjects (pairwise ISC) or the correlation between each subject and the
    average of the other subjects (one-vs-rest ISC).

    Parameters
    ----------
    dms : array_like
        Data matrices of shape (n_subjects, n_timepoints, n_vertices).
    pairwise : bool, default=False
        If True, compute pairwise ISC, which is the ISC between all pairs of
        subjects. Otherwise, compute one-vs-rest ISC, which is the ISC between
        each subject and the average of the other subjects.
    metric : str, default='correlation'
        Distance metric to use for pairwise ISC. Can be 'correlation' or
        'cosine'.
    n_jobs : int, default=-1
        Number of jobs to run in parallel. If -1, use all available CPUs.

    Returns
    -------
    isc : ndarray
        ISC of shape (n_subjects, n_vertices) if pairwise is False, or
        (n_pairs_of_subjects, n_vertices) if pairwise is True.
    """
    if pairwise:
        isc = compute_isc_pairwise(dms, metric=metric, n_jobs=n_jobs)
    else:
        isc = compute_isc_ovr(dms, metric=metric, n_jobs=n_jobs)

    return isc
