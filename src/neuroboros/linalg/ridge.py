from multiprocessing import Pool

import numpy as np
from joblib import cpu_count

from ..ensemble import kfold_bagging_groups
from .base import safe_svd


# Module-level state for worker processes (set once per worker via Pool initializer).
_worker_X = None
_worker_y = None
_worker_alphas = None
_worker_npcs = None
_worker_fit_intercept = None


def _worker_init(X, y, alphas, npcs, fit_intercept):
    global _worker_X, _worker_y, _worker_alphas, _worker_npcs, _worker_fit_intercept
    _worker_X = X
    _worker_y = y
    _worker_alphas = alphas
    _worker_npcs = npcs
    _worker_fit_intercept = fit_intercept


def _worker_fn(task):
    train_idx, tgi = task
    beta = ridge_grid(
        _worker_X[train_idx],
        _worker_y[train_idx],
        _worker_alphas,
        _worker_npcs,
        fit_intercept=_worker_fit_intercept,
    )
    return beta, tgi


def ridge(X_train, y_train, alpha, npc=None, X_test=None, fit_intercept=True):
    """
    Parameters
    ----------
    X_train : ndarray of shape (n_training, n_features)
    y_train : ndarray of shape (n_training,)
    alpha : float
        The regularization parameter for ridge regression.
    npc : {int, None}, default=None
        The number of PCs used in the prediction model. `None` means all PCs are used.
    X_test : ndarray of shape (n_test, n_features), optional
        If provided, return predicted values. If None, return beta coefficients.
    fit_intercept : bool, default=True

    Returns
    -------
    yhat : ndarray of shape (n_test,)
        Predicted values, returned when X_test is provided.
    beta : ndarray of shape (n_features,) or (n_features + 1,)
        Coefficients, returned when X_test is None. If fit_intercept, the intercept
        is appended as the last element.
    """
    # TODO multiple measures
    if fit_intercept:
        X_offset = np.mean(X_train, axis=0, keepdims=True)
        y_offset = np.mean(y_train, axis=0, keepdims=True)
        X_train = X_train - X_offset
        y_train = y_train - y_offset

    U, s, Vt = safe_svd(X_train, remove_mean=False)

    UT_y = U.T[:npc, :] @ y_train
    d = s[:npc] / (s[:npc] ** 2 + alpha)
    d_UT_y = d * UT_y

    if X_test is None:
        beta = Vt.T[:, :npc] @ d_UT_y
        if fit_intercept:
            intercept = y_offset.squeeze() - X_offset.squeeze() @ beta
            beta = np.append(beta, intercept)
        return beta

    if fit_intercept:
        X_test_V = (X_test - X_offset) @ Vt.T[:, :npc]
    else:
        X_test_V = X_test @ Vt.T[:, :npc]

    yhat = X_test_V @ d_UT_y
    if fit_intercept:
        yhat += y_offset

    return yhat


def ridge_grid(X_train, y_train, alphas, npcs, X_test=None, fit_intercept=True):
    """
    Parameters
    ----------
    X_train : ndarray of shape (n_training, n_features)
    y_train : ndarray of shape (n_training,)
    alphas : list or ndarray of shape (n_alphas,)
        Choices of the regularization parameter for ridge regression.
    npcs : list of {int, None}
        Choices of the number of PCs used in the prediction model, in increasing order.
        `None` means all PCs are used and must be the last element if included.
    X_test : ndarray of shape (n_test, n_features), optional
        If provided, return predicted values. If None, return beta coefficients.
    fit_intercept : bool, default=True

    Returns
    -------
    yhat : ndarray of shape (n_test, n_alphas, n_npcs)
        Predicted values, returned when X_test is provided.
    beta : ndarray of shape (n_features, n_alphas, n_npcs) or (n_features + 1, n_alphas, n_npcs)
        Coefficients, returned when X_test is None. If fit_intercept, shape is
        (n_features + 1, ...) with the intercept appended along axis 0.
    """
    if fit_intercept:
        X_offset = np.mean(X_train, axis=0, keepdims=True)
        y_offset = np.mean(y_train, axis=0, keepdims=True)
        X_train = X_train - X_offset
        y_train = y_train - y_offset

    U, s, Vt = safe_svd(X_train, remove_mean=False)
    UT_y = U.T @ y_train  # (k, )
    d = s[:, np.newaxis] / (
        (s**2)[:, np.newaxis] + np.array(alphas)[np.newaxis, :]
    )  # (k, n_alphas)
    d_UT_y = d * UT_y[..., np.newaxis]  # (k, n_alphas)

    npcs_ = [0] + list(npcs)

    if X_test is None:
        beta = np.zeros((Vt.shape[1], len(alphas), len(npcs)))
        for i in range(len(npcs)):
            beta[..., i] = np.tensordot(
                Vt.T[:, npcs_[i] : npcs[i]], d_UT_y[npcs_[i] : npcs[i]], axes=(1, 0)
            )
        beta = np.cumsum(beta, axis=-1)
        if fit_intercept:
            intercept = y_offset.squeeze() - np.tensordot(
                X_offset.squeeze(), beta, axes=(0, 0)
            )
            beta = np.concatenate([beta, intercept[np.newaxis]], axis=0)
        return beta

    if fit_intercept:
        X_test_V = (X_test - X_offset) @ Vt.T  # (n_test, k)
    else:
        X_test_V = X_test @ Vt.T

    yhat = np.zeros((X_test.shape[0], len(alphas), len(npcs)))
    for i in range(len(npcs)):
        yhat[..., i] = np.tensordot(
            X_test_V[:, npcs_[i] : npcs[i]], d_UT_y[npcs_[i] : npcs[i]], axes=(1, 0)
        )
    yhat = np.cumsum(yhat, axis=-1)
    if fit_intercept:
        yhat += y_offset

    return yhat


def ridge_cv(
    X, y, groups, alphas, npcs, n_folds=5, n_reps=20, fit_intercept=True, seed=0
):
    """
    Ridge regression cross-validation without hyperparameter selection.

    Trains a grid of ridge models using group-aware k-fold bagging.  For each
    group ``g``, predictions are made using the average beta from folds where
    ``g`` was withheld (leave-one-group-out).  All ``(alpha, npc)`` combinations
    are returned.

    Parameters
    ----------
    X : ndarray of shape (n_observations, n_features)
    y : ndarray of shape (n_observations,)
    groups : list of arrays or None
        Each array contains the observation indices belonging to one group.
        If None, each observation is treated as its own group.
    alphas : list or ndarray of shape (n_alphas,)
    npcs : list of {int, None}
        In increasing order; ``None`` must be last if included.
    n_folds : int, default=5
    n_reps : int, default=20
    fit_intercept : bool, default=True
    seed : int, default=0

    Returns
    -------
    yhat : ndarray of shape (n_observations, n_alphas, n_npcs)
        Leave-one-group-out predictions for every ``(alpha, npc)`` combination.
    beta : ndarray of shape (n_coef, n_alphas, n_npcs)
        Average beta across all folds, where ``n_coef = n_features + 1`` if
        ``fit_intercept`` else ``n_features``.
    """
    n_obs, n_features = X.shape
    if groups is None:
        groups = [np.array([i]) for i in range(n_obs)]
    n_groups = len(groups)
    n_coef = n_features + (1 if fit_intercept else 0)
    n_alphas, n_npcs = len(alphas), len(npcs)

    diag_betas = np.zeros((n_groups, n_coef, n_alphas, n_npcs))
    diag_counts = np.zeros(n_groups, dtype=int)
    beta = np.zeros((n_coef, n_alphas, n_npcs))
    avg_count = 0

    for train_idx, tgi in kfold_bagging_groups(
        groups, n_folds=n_folds, n_reps=n_reps, seed=seed
    ):
        fold_beta = ridge_grid(
            X[train_idx],
            y[train_idx],
            alphas,
            npcs,
            X_test=None,
            fit_intercept=fit_intercept,
        )  # (n_coef, n_alphas, n_npcs)

        beta += fold_beta
        avg_count += 1

        diag_betas[tgi] += fold_beta
        diag_counts[tgi] += 1

    beta /= avg_count
    cnt = np.where(diag_counts > 0, diag_counts, 1)
    diag_betas /= cnt[:, np.newaxis, np.newaxis, np.newaxis]

    yhat = np.zeros((n_obs, n_alphas, n_npcs))
    for g, g_obs in enumerate(groups):
        b = diag_betas[g]  # (n_coef, n_alphas, n_npcs)
        yhat[g_obs] = np.tensordot(X[g_obs], b[:n_features], axes=(1, 0))
        if fit_intercept:
            yhat[g_obs] += b[-1]

    return yhat, beta


def ridge_nested_cv(
    X, y, groups, alphas, npcs, n_folds=5, n_reps=20, fit_intercept=True, seed=0
):
    """
    Ridge regression with nested cross-validation for hyperparameter selection.

    Trains a grid of ridge models using group-aware k-fold bagging.  For each
    group ``g``, the best ``(alpha, npc)`` pair is chosen by minimizing the sum
    of squared errors on every other group ``h`` using betas from folds where
    both ``g`` and ``h`` were withheld.  Final predictions for ``g`` use the
    diagonal beta ``(g, g)`` at the selected hyperparameters.  A global best
    pair is also selected by minimizing leave-one-group-out SSE across all groups.

    Parameters
    ----------
    X : ndarray of shape (n_observations, n_features)
    y : ndarray of shape (n_observations,)
    groups : list of arrays or None
        Each array contains the observation indices belonging to one group.
        Groups are never split across training and test sets. If None, each
        observation is treated as its own group.
    alphas : list or ndarray of shape (n_alphas,)
        Regularization parameter choices.
    npcs : list of {int, None}
        Number of PCs choices, in increasing order. `None` means all PCs are used
        and must be the last element if included.
    n_folds : int, default=5
    n_reps : int, default=20
    fit_intercept : bool, default=True
    seed : int, default=0

    Returns
    -------
    yhat : ndarray of shape (n_observations,)
        Predicted values for each observation using the per-group optimal beta.
    beta : ndarray of shape (n_coef,)
        Average beta (across all folds) at the globally best ``(alpha, npc)``
        pair, where ``n_coef = n_features + 1`` if ``fit_intercept`` else
        ``n_features``.
    choices : ndarray of shape (n_groups + 1, 2)
        Selected ``(alpha_index, npc_index)`` per group, with the global best
        appended as the last row.
    """
    n_obs, n_features = X.shape
    if groups is None:
        groups = [np.array([i]) for i in range(n_obs)]
    n_groups = len(groups)
    n_coef = n_features + (1 if fit_intercept else 0)
    n_tri = n_groups * (n_groups + 1) // 2

    triu_rows, triu_cols = np.triu_indices(n_groups)
    idx_map = np.full((n_groups, n_groups), -1, dtype=int)
    idx_map[triu_rows, triu_cols] = np.arange(n_tri)

    betas = np.zeros((n_tri, n_coef, len(alphas), len(npcs)))
    counts = np.zeros(n_tri, dtype=int)
    avg_beta = np.zeros((n_coef, len(alphas), len(npcs)))
    avg_count = 0

    for train_idx, tgi in kfold_bagging_groups(
        groups, n_folds=n_folds, n_reps=n_reps, seed=seed
    ):
        beta = ridge_grid(
            X[train_idx],
            y[train_idx],
            alphas,
            npcs,
            X_test=None,
            fit_intercept=fit_intercept,
        )  # (n_coef, n_alphas, n_npcs)

        avg_beta += beta
        avg_count += 1

        gi, gj = np.meshgrid(tgi, tgi, indexing="ij")
        mask = gi <= gj
        flat = idx_map[gi[mask], gj[mask]]
        betas[flat] += beta
        counts[flat] += 1

    cnt = np.where(counts > 0, counts, 1)
    betas /= cnt[:, np.newaxis, np.newaxis, np.newaxis]
    avg_beta /= avg_count

    n_alphas, n_npcs = len(alphas), len(npcs)
    yhat = np.zeros(n_obs)
    choices = np.zeros((n_groups, 2), dtype=int)
    for g, g_obs in enumerate(groups):
        sse = np.zeros((n_alphas, n_npcs))

        for h, h_obs in enumerate(groups):
            if h == g:
                continue
            k = idx_map[min(g, h), max(g, h)]
            if counts[k] == 0:
                continue

            y_hat_h = np.tensordot(X[h_obs], betas[k, :n_features], axes=(1, 0))
            if fit_intercept:
                y_hat_h += betas[k, -1]

            residuals = y[h_obs, np.newaxis, np.newaxis] - y_hat_h
            sse += (residuals**2).sum(axis=0)

        best = np.unravel_index(np.argmin(sse), sse.shape)
        choices[g] = best

        b = betas[idx_map[g, g], :, best[0], best[1]]
        yhat[g_obs] = X[g_obs] @ b[:n_features]
        if fit_intercept:
            yhat[g_obs] += b[-1]

    global_sse = np.zeros((n_alphas, n_npcs))
    for g, g_obs in enumerate(groups):
        k = idx_map[g, g]
        if counts[k] == 0:
            continue
        y_hat_g = np.tensordot(X[g_obs], betas[k, :n_features], axes=(1, 0))
        if fit_intercept:
            y_hat_g += betas[k, -1]
        global_sse += ((y[g_obs, np.newaxis, np.newaxis] - y_hat_g) ** 2).sum(axis=0)

    global_best = np.unravel_index(np.argmin(global_sse), global_sse.shape)
    beta = avg_beta[:, global_best[0], global_best[1]]
    choices = np.vstack([choices, global_best])

    return yhat, beta, choices


def ridge_cv_parallel(
    X,
    y,
    groups,
    alphas,
    npcs,
    n_folds=5,
    n_reps=20,
    fit_intercept=True,
    seed=0,
    n_jobs=-1,
    deterministic=True,
):
    """
    Parallel version of :func:`ridge_cv`.

    Worker processes compute :func:`ridge_grid` on different training sets
    concurrently while the main process aggregates results as they arrive.
    ``X`` and ``y`` are copied once per worker at startup via the pool
    initializer, not once per task.

    Parameters
    ----------
    X : ndarray of shape (n_observations, n_features)
    y : ndarray of shape (n_observations,)
    groups : list of arrays or None
    alphas : list or ndarray of shape (n_alphas,)
    npcs : list of {int, None}
        In increasing order; ``None`` must be last if included.
    n_folds : int, default=5
    n_reps : int, default=20
    fit_intercept : bool, default=True
    seed : int, default=0
    n_jobs : int, default=-1
        Total number of processes. ``-1`` uses all available CPUs. One
        process is reserved for aggregation; the rest compute betas.
    deterministic : bool, default=True
        If True, results are aggregated in submission order (``imap``),
        giving output bit-identical to :func:`ridge_cv`. If False,
        uses ``imap_unordered`` for better throughput at the cost of
        floating-point non-determinism.

    Returns
    -------
    yhat : ndarray of shape (n_observations, n_alphas, n_npcs)
    beta : ndarray of shape (n_coef, n_alphas, n_npcs)
    """
    n_obs, n_features = X.shape
    if groups is None:
        groups = [np.array([i]) for i in range(n_obs)]
    n_groups = len(groups)
    n_coef = n_features + (1 if fit_intercept else 0)
    n_alphas, n_npcs = len(alphas), len(npcs)

    diag_betas = np.zeros((n_groups, n_coef, n_alphas, n_npcs))
    diag_counts = np.zeros(n_groups, dtype=int)
    beta = np.zeros((n_coef, n_alphas, n_npcs))
    avg_count = 0

    if n_jobs < 0:
        n_jobs = int(cpu_count() + n_jobs + 1)
    n_workers = max(1, n_jobs - 1)

    fold_list = kfold_bagging_groups(groups, n_folds=n_folds, n_reps=n_reps, seed=seed)

    with Pool(
        n_workers,
        initializer=_worker_init,
        initargs=(X, y, alphas, npcs, fit_intercept),
    ) as pool:
        imap = pool.imap if deterministic else pool.imap_unordered
        for fold_beta, tgi in imap(_worker_fn, fold_list):
            beta += fold_beta
            avg_count += 1
            diag_betas[tgi] += fold_beta
            diag_counts[tgi] += 1

    beta /= avg_count
    cnt = np.where(diag_counts > 0, diag_counts, 1)
    diag_betas /= cnt[:, np.newaxis, np.newaxis, np.newaxis]

    yhat = np.zeros((n_obs, n_alphas, n_npcs))
    for g, g_obs in enumerate(groups):
        b = diag_betas[g]
        yhat[g_obs] = np.tensordot(X[g_obs], b[:n_features], axes=(1, 0))
        if fit_intercept:
            yhat[g_obs] += b[-1]

    return yhat, beta


def ridge_nested_cv_parallel(
    X,
    y,
    groups,
    alphas,
    npcs,
    n_folds=5,
    n_reps=20,
    fit_intercept=True,
    seed=0,
    n_jobs=-1,
    deterministic=True,
):
    """
    Parallel version of :func:`ridge_nested_cv`.

    Worker processes compute :func:`ridge_grid` on different training sets
    concurrently while the main process aggregates results as they arrive.
    ``X`` and ``y`` are copied once per worker at startup via the pool
    initializer, not once per task.

    Parameters
    ----------
    X : ndarray of shape (n_observations, n_features)
    y : ndarray of shape (n_observations,)
    groups : list of arrays or None
    alphas : list or ndarray of shape (n_alphas,)
    npcs : list of {int, None}
        In increasing order; ``None`` must be last if included.
    n_folds : int, default=5
    n_reps : int, default=20
    fit_intercept : bool, default=True
    seed : int, default=0
    n_jobs : int, default=-1
        Total number of processes. ``-1`` uses all available CPUs. One
        process is reserved for aggregation; the rest compute betas.
    deterministic : bool, default=True
        If True, results are aggregated in submission order (``imap``),
        giving output bit-identical to :func:`ridge_nested_cv`. If False,
        uses ``imap_unordered`` for better throughput at the cost of
        floating-point non-determinism.

    Returns
    -------
    yhat : ndarray of shape (n_observations,)
    beta : ndarray of shape (n_coef,)
    choices : ndarray of shape (n_groups + 1, 2)
    """
    n_obs, n_features = X.shape
    if groups is None:
        groups = [np.array([i]) for i in range(n_obs)]
    n_groups = len(groups)
    n_coef = n_features + (1 if fit_intercept else 0)
    n_tri = n_groups * (n_groups + 1) // 2

    triu_rows, triu_cols = np.triu_indices(n_groups)
    idx_map = np.full((n_groups, n_groups), -1, dtype=int)
    idx_map[triu_rows, triu_cols] = np.arange(n_tri)

    betas = np.zeros((n_tri, n_coef, len(alphas), len(npcs)))
    counts = np.zeros(n_tri, dtype=int)
    avg_beta = np.zeros((n_coef, len(alphas), len(npcs)))
    avg_count = 0

    if n_jobs < 0:
        n_jobs = int(cpu_count() + n_jobs + 1)
    n_workers = max(1, n_jobs - 1)

    fold_list = kfold_bagging_groups(groups, n_folds=n_folds, n_reps=n_reps, seed=seed)

    with Pool(
        n_workers,
        initializer=_worker_init,
        initargs=(X, y, alphas, npcs, fit_intercept),
    ) as pool:
        imap = pool.imap if deterministic else pool.imap_unordered
        for beta, tgi in imap(_worker_fn, fold_list):
            avg_beta += beta
            avg_count += 1
            gi, gj = np.meshgrid(tgi, tgi, indexing="ij")
            mask = gi <= gj
            flat = idx_map[gi[mask], gj[mask]]
            betas[flat] += beta
            counts[flat] += 1

    cnt = np.where(counts > 0, counts, 1)
    betas /= cnt[:, np.newaxis, np.newaxis, np.newaxis]
    avg_beta /= avg_count

    n_alphas, n_npcs = len(alphas), len(npcs)
    yhat = np.zeros(n_obs)
    choices = np.zeros((n_groups, 2), dtype=int)
    for g, g_obs in enumerate(groups):
        sse = np.zeros((n_alphas, n_npcs))

        for h, h_obs in enumerate(groups):
            if h == g:
                continue
            k = idx_map[min(g, h), max(g, h)]
            if counts[k] == 0:
                continue

            y_hat_h = np.tensordot(X[h_obs], betas[k, :n_features], axes=(1, 0))
            if fit_intercept:
                y_hat_h += betas[k, -1]

            residuals = y[h_obs, np.newaxis, np.newaxis] - y_hat_h
            sse += (residuals**2).sum(axis=0)

        best = np.unravel_index(np.argmin(sse), sse.shape)
        choices[g] = best

        b = betas[idx_map[g, g], :, best[0], best[1]]
        yhat[g_obs] = X[g_obs] @ b[:n_features]
        if fit_intercept:
            yhat[g_obs] += b[-1]

    global_sse = np.zeros((n_alphas, n_npcs))
    for g, g_obs in enumerate(groups):
        k = idx_map[g, g]
        if counts[k] == 0:
            continue
        y_hat_g = np.tensordot(X[g_obs], betas[k, :n_features], axes=(1, 0))
        if fit_intercept:
            y_hat_g += betas[k, -1]
        global_sse += ((y[g_obs, np.newaxis, np.newaxis] - y_hat_g) ** 2).sum(axis=0)

    global_best = np.unravel_index(np.argmin(global_sse), global_sse.shape)
    beta = avg_beta[:, global_best[0], global_best[1]]
    choices = np.vstack([choices, global_best])

    return yhat, beta, choices
