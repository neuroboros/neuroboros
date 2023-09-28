import numpy as np
from scipy.stats import f, zscore


def cronbach_alpha_ci(alpha, n, m, ci):
    """Confidence interval for Cronbach's alpha.

    Parameters
    ----------
    alpha : float
        Cronbach's alpha.
    n : int
        Number of repetitions, e.g., items, parallel tests, repeated measures
        of brain response patterns.
    m : int
        Number of individuals, e.g., test-takers, voxels, vertices.
    ci : float or tuple of float
        Confidence level. If float, it should be between 0 and 1, e.g., 0.95
        for 95% confidence interval. If tuple, it should be a pair of floats
        between 0 and 1, e.g., (0.025, 0.975) for 95% confidence interval.

    Returns
    -------
    ci : tuple of float
        Confidence interval.

    Notes
    -----
    Feldt, L. S., Woodruff, D. J., & Salih, F. A. (1987). Statistical Inference for Coefficient Alpha. Applied Psychological Measurement, 11(1), 93–103. https://doi.org/10.1177/014662168701100107

    """
    if isinstance(ci, (tuple, list)):
        u, l = ci
    else:
        assert ci >= 0 and ci <= 1
        u = (1 - ci) * 0.5
        l = 1 - u
    ratios = [f.ppf(_, m - 1, (n - 1) * (m - 1)) for _ in [l, u]]
    ci = tuple(1 - (1 - alpha) * _ for _ in ratios)
    return ci


def cronbach_alpha(X, rep_axis=0, var_axis=1, ci=None, squeeze=True):
    """Cronbach's alpha.

    Parameters
    ----------
    X : ndarray
        The data.
    rep_axis : int, default=0
        The axis along which the repetitions are stacked, e.g., items,
        parallel tests, repeated measures of brain response patterns.
    var_axis : int, default=1
        The axis along which individuals differ, e.g., test-takers, voxels,
        vertices.
    ci : float or tuple of float or None, default=None
        Confidence level. If float, it should be between 0 and 1, e.g., 0.95
        for 95% confidence interval. If tuple, it should be a pair of floats
        between 0 and 1, e.g., (0.025, 0.975) for 95% confidence interval.
        If None, no confidence interval is computed or returned.
    squeeze : bool, default=True
        Whether to squeeze the returned ndarray for ``rep_axis`` and
        ``var_axis``. If False, the returned ndarray has the same number of
        dimensions as ``X``.

    Returns
    -------
    alpha : float or ndarray
        Cronbach's alpha. If ``X`` is 2D, a float is returned. Otherwise, an
        ndarray is returned.
    ci : ndarray
        Confidence interval. Returned only if ``ci`` is not None.
    """
    v1 = X.var(axis=var_axis, ddof=1, keepdims=True).sum(axis=rep_axis, keepdims=True)
    v2 = X.sum(axis=rep_axis, keepdims=True).var(axis=var_axis, ddof=1, keepdims=True)
    if squeeze:
        v1 = np.squeeze(v1, axis=(rep_axis, var_axis))
        v2 = np.squeeze(v2, axis=(rep_axis, var_axis))
    n = X.shape[rep_axis]
    alpha = (n / (n - 1.0)) * (1.0 - v1 / v2)
    if ci is not None:
        m = X.shape[var_axis]
        ci = cronbach_alpha_ci(alpha, n, m, ci)
        ci = np.stack(ci, axis=0)
        return alpha, ci
    return alpha


def noise_ceilings(X, rep_axis=0, var_axis=1, return_alpha=False):
    """Noise ceilings.

    Parameters
    ----------
    X : ndarray
        The data.
    rep_axis : int
        The axis along which the repetitions are stacked, e.g., items,
        parallel tests, repeated measures of brain response patterns.
    var_axis : int
        The axis along which individuals differ, e.g., test-takers, voxels,
        vertices.
    return_alpha : bool, default=False
        Whether to return Cronbach's alpha coefficients based on the n - 1
        individuals during the lower bound estimates.

    Returns
    -------
    ceilings : ndarray
        The noise ceilings. The first axis has two elements, the first element
        is the lower bound, and the second element is the upper bound. If
        ``return_alpha`` is True, the first axis has three elements, the third
        element is the Cronbach's alpha coefficient.
    """
    n = X.shape[rep_axis]
    if return_alpha:
        alpha = [
            cronbach_alpha(
                np.delete(X, i, axis=rep_axis), rep_axis, var_axis, squeeze=False
            )
            for i in range(n)
        ]
        alpha = np.concatenate(alpha, axis=rep_axis)
        alpha = np.squeeze(alpha, axis=var_axis)

    s = X.sum(axis=rep_axis, keepdims=True)
    d = zscore(s - X, axis=var_axis)
    X = zscore(X, axis=var_axis)
    lower = np.mean(X * d, axis=var_axis)
    s = zscore(s, axis=var_axis)
    upper = np.mean(X * s, axis=var_axis)
    if return_alpha:
        ceilings = np.stack([lower, upper, alpha], axis=0)
    else:
        ceilings = np.stack([lower, upper], axis=0)
    return ceilings


def spearman_brown(rs, n, inverse=False):
    """Spearman-Brown prediction formula.

    Parameters
    ----------
    rs : float
        The reliability coefficient. By default, it's the reliability of a
        single measurement. If ``inverse`` is True, it's the reliability of
        repetitions.
    n : int
        The number of repetitions.
    inverse : bool, default=False
        If True, instead of predicting the reliability of repetitions based on
        the reliability of a single measurement, predict the reliability of a
        single measurement based on the reliability of repetitions.

    Returns
    -------
    rs : float
        The predicted reliability coefficient.
    """
    if inverse:
        rs1 = 1.0 / (1.0 + n * (1.0 - rs) / rs)
        return rs1
    else:
        rsn = (n * rs) / (1.0 + (n - 1) * rs)
        return rsn