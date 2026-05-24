import numpy as np

from .base import safe_svd


def ridge(X_train, y_train, alpha, npc=None, X_test=None, fit_intercept=True):
    """
    Parameters
    ----------
    X_train : ndarray of shape (n_training, n_features)
    y_train : ndarray of shape (n_training, )
    alpha : float
        The regularization parameter for ridge regression.
    npc : {int, None}, default=None
        The number of PCs used in the prediction model. `None` means all PCs are used.
    X_test : ndarray of shape (n_test, n_features), optional
        If provided, return predicted values. If None, return beta coefficients.

    Returns
    -------
    yhat : ndarray of shape (n_test, )
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
    y_train : ndarray of shape (n_training, )
    alphas : {list, ndarray of shape (n_alphas, )}
        Choices of the regularization parameter for ridge regression.
    npcs : {list, ndarray of shape (n_npcs, )}
        Choices of the number of PCs used in the prediction model in increasing order.
        Each element should be an integer or `None`. `None` means all PCs are used.
    X_test : ndarray of shape (n_test, n_features), optional
        If provided, return predicted values. If None, return beta coefficients.

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
