import numpy as np
import pytest
from sklearn.linear_model import Ridge as SklearnRidge

from neuroboros.ensemble import kfold_bagging_groups
from neuroboros.linalg.ridge import (
    ridge,
    ridge_cv,
    ridge_cv_parallel,
    ridge_grid,
    ridge_nested_cv,
    ridge_nested_cv_parallel,
)


class TestRidge:
    def test_yhat_shape(self):
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((50, 20))
        X_test = rng.standard_normal((10, 20))
        y_train = rng.standard_normal((50,))
        yhat = ridge(X_train, y_train, alpha=1.0, X_test=X_test)
        assert yhat.shape == (10,)

    def test_beta_shape_with_intercept(self):
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((50, 20))
        y_train = rng.standard_normal((50,))
        beta = ridge(X_train, y_train, alpha=1.0)
        assert beta.shape == (21,)

    def test_beta_shape_no_intercept(self):
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((50, 20))
        y_train = rng.standard_normal((50,))
        beta = ridge(X_train, y_train, alpha=1.0, fit_intercept=False)
        assert beta.shape == (20,)

    def test_matches_sklearn(self):
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((50, 20))
        X_test = rng.standard_normal((10, 20))
        y_train = rng.standard_normal((50,))
        alpha = 1.0
        yhat = ridge(X_train, y_train, alpha, X_test=X_test)
        clf = SklearnRidge(alpha=alpha, fit_intercept=True, solver="svd")
        clf.fit(X_train, y_train)
        np.testing.assert_allclose(yhat, clf.predict(X_test), atol=1e-10)

    def test_matches_sklearn_no_intercept(self):
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((50, 20))
        X_test = rng.standard_normal((10, 20))
        y_train = rng.standard_normal((50,))
        alpha = 2.5
        yhat = ridge(X_train, y_train, alpha, X_test=X_test, fit_intercept=False)
        clf = SklearnRidge(alpha=alpha, fit_intercept=False, solver="svd")
        clf.fit(X_train, y_train)
        np.testing.assert_allclose(yhat, clf.predict(X_test), atol=1e-10)

    def test_beta_predicts_consistently(self):
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((50, 20))
        X_test = rng.standard_normal((10, 20))
        y_train = rng.standard_normal((50,))
        alpha = 1.0
        beta = ridge(X_train, y_train, alpha)
        yhat = ridge(X_train, y_train, alpha, X_test=X_test)
        np.testing.assert_allclose(X_test @ beta[:-1] + beta[-1], yhat, atol=1e-10)

    def test_beta_no_intercept_predicts_consistently(self):
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((50, 20))
        X_test = rng.standard_normal((10, 20))
        y_train = rng.standard_normal((50,))
        alpha = 1.0
        beta = ridge(X_train, y_train, alpha, fit_intercept=False)
        yhat = ridge(X_train, y_train, alpha, X_test=X_test, fit_intercept=False)
        np.testing.assert_allclose(X_test @ beta, yhat, atol=1e-10)

    def test_npc_reduces_rank(self):
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((50, 20))
        X_test = rng.standard_normal((10, 20))
        y_train = rng.standard_normal((50,))
        yhat_all = ridge(X_train, y_train, alpha=1.0, X_test=X_test)
        yhat_npc = ridge(X_train, y_train, alpha=1.0, npc=5, X_test=X_test)
        assert yhat_npc.shape == (10,)
        assert not np.allclose(yhat_all, yhat_npc)

    def test_npc_beta_predicts_consistently(self):
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((50, 20))
        X_test = rng.standard_normal((10, 20))
        y_train = rng.standard_normal((50,))
        alpha, npc = 1.0, 8
        beta = ridge(X_train, y_train, alpha, npc=npc)
        yhat = ridge(X_train, y_train, alpha, npc=npc, X_test=X_test)
        np.testing.assert_allclose(X_test @ beta[:-1] + beta[-1], yhat, atol=1e-10)


class TestRidgeGrid:
    def test_yhat_shape(self):
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((50, 20))
        X_test = rng.standard_normal((10, 20))
        y_train = rng.standard_normal((50,))
        yhat = ridge_grid(
            X_train, y_train, [0.1, 1.0, 10.0], [5, 10, 15], X_test=X_test
        )
        assert yhat.shape == (10, 3, 3)

    def test_beta_shape_with_intercept(self):
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((50, 20))
        y_train = rng.standard_normal((50,))
        beta = ridge_grid(X_train, y_train, [0.1, 1.0], [5, 10])
        assert beta.shape == (21, 2, 2)

    def test_beta_shape_no_intercept(self):
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((50, 20))
        y_train = rng.standard_normal((50,))
        beta = ridge_grid(X_train, y_train, [0.1, 1.0], [5, 10], fit_intercept=False)
        assert beta.shape == (20, 2, 2)

    def test_yhat_consistent_with_ridge(self):
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((50, 20))
        X_test = rng.standard_normal((10, 20))
        y_train = rng.standard_normal((50,))
        alphas = [0.1, 1.0, 10.0]
        npcs = [5, 10, 15]
        yhat = ridge_grid(X_train, y_train, alphas, npcs, X_test=X_test)
        for i, alpha in enumerate(alphas):
            for j, npc in enumerate(npcs):
                expected = ridge(X_train, y_train, alpha, npc=npc, X_test=X_test)
                np.testing.assert_allclose(
                    yhat[:, i, j],
                    expected,
                    atol=1e-10,
                    err_msg=f"alpha={alpha}, npc={npc}",
                )

    def test_beta_consistent_with_ridge(self):
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((50, 20))
        y_train = rng.standard_normal((50,))
        alphas = [0.1, 1.0, 10.0]
        npcs = [5, 10, 15]
        beta = ridge_grid(X_train, y_train, alphas, npcs)
        for i, alpha in enumerate(alphas):
            for j, npc in enumerate(npcs):
                expected = ridge(X_train, y_train, alpha, npc=npc)
                np.testing.assert_allclose(
                    beta[:, i, j],
                    expected,
                    atol=1e-10,
                    err_msg=f"alpha={alpha}, npc={npc}",
                )

    def test_beta_predicts_consistently(self):
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((50, 20))
        X_test = rng.standard_normal((10, 20))
        y_train = rng.standard_normal((50,))
        alphas = [0.1, 1.0]
        npcs = [5, 10]
        beta = ridge_grid(X_train, y_train, alphas, npcs)
        yhat = ridge_grid(X_train, y_train, alphas, npcs, X_test=X_test)
        # beta[:-1]: (n_features, n_alphas, n_npcs), beta[-1]: (n_alphas, n_npcs)
        predicted = np.tensordot(X_test, beta[:-1], axes=(1, 0)) + beta[-1]
        np.testing.assert_allclose(predicted, yhat, atol=1e-10)

    def test_beta_no_intercept_predicts_consistently(self):
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((50, 20))
        X_test = rng.standard_normal((10, 20))
        y_train = rng.standard_normal((50,))
        alphas = [0.1, 1.0]
        npcs = [5, 10]
        beta = ridge_grid(X_train, y_train, alphas, npcs, fit_intercept=False)
        yhat = ridge_grid(
            X_train, y_train, alphas, npcs, X_test=X_test, fit_intercept=False
        )
        predicted = np.tensordot(X_test, beta, axes=(1, 0))
        np.testing.assert_allclose(predicted, yhat, atol=1e-10)

    def test_none_npc_yhat_consistent_with_ridge(self):
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((50, 20))
        X_test = rng.standard_normal((10, 20))
        y_train = rng.standard_normal((50,))
        alphas = [0.1, 1.0]
        npcs = [5, 10, None]
        yhat = ridge_grid(X_train, y_train, alphas, npcs, X_test=X_test)
        assert yhat.shape == (10, 2, 3)
        for i, alpha in enumerate(alphas):
            for j, npc in enumerate(npcs):
                expected = ridge(X_train, y_train, alpha, npc=npc, X_test=X_test)
                np.testing.assert_allclose(
                    yhat[:, i, j],
                    expected,
                    atol=1e-10,
                    err_msg=f"alpha={alpha}, npc={npc}",
                )

    def test_none_npc_beta_consistent_with_ridge(self):
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((50, 20))
        y_train = rng.standard_normal((50,))
        alphas = [0.1, 1.0]
        npcs = [5, 10, None]
        beta = ridge_grid(X_train, y_train, alphas, npcs)
        assert beta.shape == (21, 2, 3)
        for i, alpha in enumerate(alphas):
            for j, npc in enumerate(npcs):
                expected = ridge(X_train, y_train, alpha, npc=npc)
                np.testing.assert_allclose(
                    beta[:, i, j],
                    expected,
                    atol=1e-10,
                    err_msg=f"alpha={alpha}, npc={npc}",
                )


class TestRidgeCV:
    def _make_data(self):
        rng = np.random.default_rng(0)
        n_obs, n_features, n_groups = 30, 10, 6
        X = rng.standard_normal((n_obs, n_features))
        y = X @ rng.standard_normal(n_features) + rng.standard_normal(n_obs) * 0.5
        groups = [np.arange(i * 5, (i + 1) * 5) for i in range(n_groups)]
        alphas = [0.1, 1.0, 10.0]
        npcs = [3, 5]
        return X, y, groups, alphas, npcs

    def test_output_shapes(self):
        X, y, groups, alphas, npcs = self._make_data()
        n_obs, n_features = X.shape
        yhat, beta = ridge_cv(X, y, groups, alphas, npcs, n_folds=3, n_reps=3, seed=0)
        assert yhat.shape == (n_obs, len(alphas), len(npcs))
        assert beta.shape == (n_features + 1, len(alphas), len(npcs))

    def test_output_shapes_no_intercept(self):
        X, y, groups, alphas, npcs = self._make_data()
        n_obs, n_features = X.shape
        yhat, beta = ridge_cv(
            X, y, groups, alphas, npcs, n_folds=3, n_reps=3, fit_intercept=False, seed=0
        )
        assert yhat.shape == (n_obs, len(alphas), len(npcs))
        assert beta.shape == (n_features, len(alphas), len(npcs))

    def test_yhat_consistent_with_ridge(self):
        # yhat[:, i, j] must match manual LOGO-CV predictions using ridge()
        X, y, groups, alphas, npcs = self._make_data()
        n_obs, n_features = X.shape
        n_groups = len(groups)
        yhat, _ = ridge_cv(X, y, groups, alphas, npcs, n_folds=3, n_reps=3, seed=0)

        for i, alpha in enumerate(alphas):
            for j, npc in enumerate(npcs):
                diag_betas = np.zeros((n_groups, n_features + 1))
                diag_counts = np.zeros(n_groups, dtype=int)
                for train_idx, tgi in kfold_bagging_groups(
                    groups, n_folds=3, n_reps=3, seed=0
                ):
                    b = ridge(X[train_idx], y[train_idx], alpha, npc=npc)
                    diag_betas[tgi] += b
                    diag_counts[tgi] += 1
                cnt = np.where(diag_counts > 0, diag_counts, 1)
                diag_betas /= cnt[:, np.newaxis]

                yhat_manual = np.zeros(n_obs)
                for g, g_obs in enumerate(groups):
                    b = diag_betas[g]
                    yhat_manual[g_obs] = X[g_obs] @ b[:n_features] + b[-1]

                np.testing.assert_allclose(
                    yhat[:, i, j],
                    yhat_manual,
                    atol=1e-10,
                    err_msg=f"alpha={alpha}, npc={npc}",
                )

    def test_alpha_order_consistent(self):
        X, y, groups, alphas, npcs = self._make_data()
        yhat, beta = ridge_cv(X, y, groups, alphas, npcs, n_folds=3, n_reps=3, seed=0)
        alphas_rev = alphas[::-1]
        yhat_rev, beta_rev = ridge_cv(
            X, y, groups, alphas_rev, npcs, n_folds=3, n_reps=3, seed=0
        )
        np.testing.assert_allclose(yhat, yhat_rev[:, ::-1, :], atol=1e-10)
        np.testing.assert_allclose(beta, beta_rev[:, ::-1, :], atol=1e-10)

    def test_beta_consistent_with_ridge_grid(self):
        # beta[:, i, j] must be the average of ridge_grid over all folds
        X, y, groups, alphas, npcs = self._make_data()
        n_features = X.shape[1]
        _, beta = ridge_cv(X, y, groups, alphas, npcs, n_folds=3, n_reps=3, seed=0)

        avg = np.zeros((n_features + 1, len(alphas), len(npcs)))
        count = 0
        for train_idx, _ in kfold_bagging_groups(groups, n_folds=3, n_reps=3, seed=0):
            avg += ridge_grid(X[train_idx], y[train_idx], alphas, npcs)
            count += 1
        avg /= count

        np.testing.assert_allclose(beta, avg, atol=1e-10)

    def test_parallel_deterministic_matches_sequential(self):
        X, y, groups, alphas, npcs = self._make_data()
        yhat_seq, beta_seq = ridge_cv(
            X, y, groups, alphas, npcs, n_folds=3, n_reps=3, seed=0
        )
        yhat_par, beta_par = ridge_cv_parallel(
            X,
            y,
            groups,
            alphas,
            npcs,
            n_folds=3,
            n_reps=3,
            seed=0,
            n_jobs=2,
            deterministic=True,
        )
        np.testing.assert_array_equal(yhat_seq, yhat_par)
        np.testing.assert_array_equal(beta_seq, beta_par)

    def test_parallel_nondeterministic_shapes(self):
        X, y, groups, alphas, npcs = self._make_data()
        n_obs, n_features = X.shape
        yhat, beta = ridge_cv_parallel(
            X,
            y,
            groups,
            alphas,
            npcs,
            n_folds=3,
            n_reps=3,
            seed=0,
            n_jobs=2,
            deterministic=False,
        )
        assert yhat.shape == (n_obs, len(alphas), len(npcs))
        assert beta.shape == (n_features + 1, len(alphas), len(npcs))
        assert np.all(np.isfinite(yhat))
        assert np.all(np.isfinite(beta))


class TestRidgeNestedCV:
    def _make_data(self):
        rng = np.random.default_rng(0)
        n_obs, n_features, n_groups = 30, 10, 6
        X = rng.standard_normal((n_obs, n_features))
        y = X @ rng.standard_normal(n_features) + rng.standard_normal(n_obs) * 0.5
        groups = [np.arange(i * 5, (i + 1) * 5) for i in range(n_groups)]
        alphas = [0.1, 1.0, 10.0]
        npcs = [3, 5]
        return X, y, groups, alphas, npcs

    def test_output_shapes(self):
        X, y, groups, alphas, npcs = self._make_data()
        n_obs, n_features = X.shape
        n_groups = len(groups)
        yhat, beta, choices = ridge_nested_cv(
            X, y, groups, alphas, npcs, n_folds=3, n_reps=3, seed=0
        )
        assert yhat.shape == (n_obs,)
        assert beta.shape == (n_features + 1,)
        assert choices.shape == (n_groups + 1, 2)

    def test_output_shapes_no_intercept(self):
        X, y, groups, alphas, npcs = self._make_data()
        n_obs, n_features = X.shape
        n_groups = len(groups)
        yhat, beta, choices = ridge_nested_cv(
            X, y, groups, alphas, npcs, n_folds=3, n_reps=3, fit_intercept=False, seed=0
        )
        assert yhat.shape == (n_obs,)
        assert beta.shape == (n_features,)
        assert choices.shape == (n_groups + 1, 2)

    def test_choices_in_range(self):
        X, y, groups, alphas, npcs = self._make_data()
        _, _, choices = ridge_nested_cv(
            X, y, groups, alphas, npcs, n_folds=3, n_reps=3, seed=0
        )
        assert np.all((choices[:, 0] >= 0) & (choices[:, 0] < len(alphas)))
        assert np.all((choices[:, 1] >= 0) & (choices[:, 1] < len(npcs)))

    def test_parallel_deterministic_matches_sequential(self):
        X, y, groups, alphas, npcs = self._make_data()
        yhat_seq, beta_seq, choices_seq = ridge_nested_cv(
            X, y, groups, alphas, npcs, n_folds=3, n_reps=3, seed=0
        )
        yhat_par, beta_par, choices_par = ridge_nested_cv_parallel(
            X,
            y,
            groups,
            alphas,
            npcs,
            n_folds=3,
            n_reps=3,
            seed=0,
            n_jobs=2,
            deterministic=True,
        )
        np.testing.assert_array_equal(yhat_seq, yhat_par)
        np.testing.assert_array_equal(beta_seq, beta_par)
        np.testing.assert_array_equal(choices_seq, choices_par)

    def test_parallel_nondeterministic_shapes(self):
        X, y, groups, alphas, npcs = self._make_data()
        n_obs, n_features = X.shape
        n_groups = len(groups)
        yhat, beta, choices = ridge_nested_cv_parallel(
            X,
            y,
            groups,
            alphas,
            npcs,
            n_folds=3,
            n_reps=3,
            seed=0,
            n_jobs=2,
            deterministic=False,
        )
        assert yhat.shape == (n_obs,)
        assert beta.shape == (n_features + 1,)
        assert choices.shape == (n_groups + 1, 2)
        assert np.all(np.isfinite(yhat))
        assert np.all(np.isfinite(beta))
