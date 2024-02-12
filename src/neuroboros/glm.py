"""
================================================
General Linear Model (:mod:`neuroboros.glm`)
================================================

.. currentmodule:: neuroboros.glm

.. autosummary::
    :toctree:

"""

import numpy as np


def glm(dm, design, nuisance=None, contrasts=None, return_r2=False):
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
    R2s : array, shape (n_features,)
        Adjusted R-squared values. Only returned if `return_r2` is True.
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
    ss_res = np.sum(diff**2, axis=0)
    dof_res = dm.shape[0] - nr
    sigma = np.sqrt(ss_res / dof_res)
    cov = regressors.T @ regressors
    inv = np.linalg.inv(cov)

    # TODO all contrasts at once based on
    # https://torwager.github.io/ComputationalFoundations/glm2_param_inference.html
    betas, ts = [], []
    for contrast in contrasts:
        R = np.concatenate([contrast, [0] * (nr - len(contrast))]).reshape(1, -1)
        mid = R @ inv @ R.T
        assert mid.shape == (1, 1)
        mid = 1.0 / mid[0, 0]
        ratio = np.sqrt(mid) / sigma
        R_beta = (R @ beta).ravel()
        t = R_beta * ratio
        betas.append(R_beta)
        ts.append(t)

    betas, ts = np.array(betas), np.array(ts)

    if return_r2:
        if nuisance is None:
            ss_all = np.sum(dm**2, axis=0)
            R2s = 1 - ss_res * dm.shape[0] / (ss_all * dof_res)
        else:
            beta2 = np.linalg.lstsq(nuisance, dm, rcond=None)[0]
            ss_all = np.sum((dm - nuisance @ beta2) ** 2, axis=0)
            dof_all = dm.shape[0] - nuisance.shape[1]
            R2s = 1 - ss_res * dof_all / (ss_all * dof_res)
        return betas, ts, R2s

    return betas, ts
