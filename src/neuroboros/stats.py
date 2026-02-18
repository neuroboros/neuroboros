import numpy as np
from scipy.stats import f, norm, rankdata, zscore


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


def spearman_brown(r, n):
    """Spearman-Brown prediction formula.

    Parameters
    ----------
    r : float
        The reliability coefficient.
    n : int or float
        The length of the new "test" relative to the current one. For example,
        if the current test has 10 items and the new test has 20 items, then
        ``n`` is 2. If the current test has 20 items and the new test has 10
        items, then ``n`` is 0.5.

    Returns
    -------
    r_new : float
        The predicted reliability coefficient.
    """
    r_new = (n * r) / (1.0 + (n - 1.0) * r)
    return r_new


def spearman_brown_inv(r, r_new):
    """Predicting required test length for a given reliability.

    Parameters
    ----------
    r : float
        The current reliability coefficient.
    r_new : float
        The desired reliability coefficient.

    Returns
    -------
    n : float
        The required test length relative to the current one.
    """
    prod = r * r_new
    n = (r_new - prod) / (r - prod)
    return n


def normalize(d, clip=(0.0005, 0.9995), keep_stats=True):
    """Normalize the data to normal distribution.

    Parameters
    ----------
    d : ndarray
        The data to be normalized. The function assumes that ``d`` is 1D.
    clip : float or tuple of float, default=(0.0005, 0.9995)
        The lower and upper bounds for clipping the percentiles. By default,
        the function clips percentiles above 99.95% and below 0.05%. If a float
        is provided, it is used as the percentage of clipping, e.g., 0.01 for
        clipping percentiles above 99.5% and below 0.5%.
    keep_stats : bool, default=True
        Whether to keep the mean and standard deviation of the input data. If
        False, the returned data is normalized to standard normal distribution
        (zero mean and unit variance).

    Returns
    -------
    d_new : ndarray
        The normalized data.

    """
    rank = rankdata(d)
    pct = (rank - 0.5) / len(rank)

    if isinstance(clip, float):
        clip = (clip * 0.5, 1 - clip * 0.5)
    pct = np.clip(pct, *clip)

    if keep_stats:
        d_new = norm.ppf(pct, loc=d.mean(), scale=d.std())
    else:
        d_new = norm.ppf(pct, loc=0, scale=1)
    return d_new


def pearsonr(X, Y, axis=0, keepdims=False, nan=None):
    """
    Pearson correlation coefficient.

    Parameters
    ----------
    X : ndarray
        First input array.
    Y : ndarray
        Second input array.
    axis : int, default=0
        Axis along which to compute the correlation.
    keepdims : bool, default=False
        If True, the output will have the same number of dimensions as the
        input arrays.
    nan : float or None, default=None
        If not None, NaNs in the input arrays will be replaced with this
        value.

    Returns
    -------
    r : float or ndarray
        Pearson correlation coefficient.
    """
    X = zscore(X, axis=axis)
    Y = zscore(Y, axis=axis)
    r = np.mean(X * Y, axis=axis, keepdims=keepdims)
    if nan is not None:
        r = np.nan_to_num(r, copy=False, nan=nan)
    return r
