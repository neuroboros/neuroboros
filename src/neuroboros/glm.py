"""
================================================
General Linear Model (:mod:`neuroboros.glm`)
================================================

.. currentmodule:: neuroboros.glm

.. autosummary::
    :toctree:

"""

import numpy as np


def glm(dm, design, nuisance=None, contrasts=None):
    """General linear model.

    Parameters
    ----------
    dm : array, shape (n_samples, n_features)
        Data matrix.
    design : array, shape (n_samples, n_design_regressors)
        Experimental design regressors.
    nuisance : array, shape (n_samples, n_nuisance_regressors), optional
        Nuisance regressors.
    contrasts : array, shape (n_contrasts, n_design_regressors), optional
        Contrasts. If None, identity matrix is used, which models the response
        to each design regressor separately.

    Returns
    -------
    betas : array, shape (n_contrasts, n_features)
        Estimated beta coefficients.
    ts : array, shape (n_contrasts, n_features)
        Estimated t-statistics.
    """
    if nuisance is not None:
        regressors = np.concatenate([design, nuisance], axis=1)
    else:
        regressors = design
    nr = regressors.shape[1]

    if contrasts is None:
        contrasts = np.eye(design.shape[1])

    beta = np.linalg.lstsq(regressors, dm, rcond=None)[0]
    diff = dm - regressors @ beta
    sigma = np.sqrt(np.sum(diff**2, axis=0) / (dm.shape[0] - nr))
    cov = regressors.T @ regressors
    inv = np.linalg.inv(cov)

    betas, ts = [], []
    for contrast in contrasts:
        R = np.concatenate([contrast, [0] * (nr - len(contrast))]).reshape(1, -1)
        mid = np.linalg.inv(R @ inv @ R.T)
        ratio = np.sqrt(float(mid)) / sigma
        R_beta = (R @ beta).ravel()
        t = R_beta * ratio
        betas.append(R_beta)
        ts.append(t)

    betas, ts = np.array(betas), np.array(ts)
    return betas, ts
