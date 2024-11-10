import numpy as np
import pytest
from joblib import cpu_count

import neuroboros as nb


class TestEnsemble:
    def test_kfold_bagging(self):
        n = 500
        indices_li = nb.ensemble.kfold_bagging(n, n_folds=20, n_perms=20, seed=0)

        test_idx = np.concatenate([_[1] for _ in indices_li])
        uu, cc = np.unique(test_idx, return_counts=True)
        counts_test = np.zeros((n,))
        counts_test[uu] = cc
        assert np.all(counts_test >= 20)

        for train_idx, test_idx in indices_li:
            assert np.all(np.intersect1d(train_idx, test_idx) == [])

    def test_ensemble_lstsq(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 50))
        beta0 = rng.standard_normal((50, 20))
        Y = rng.standard_normal((100, 20)) * 0.1 + X @ beta0

        beta, Yhat, R2, r = nb.linalg.ensemble_lstsq(
            X, Y, n_folds=5, n_perms=20, seed=0
        )

        assert beta.shape == (50, 20)
        assert Yhat.shape == (100, 20)
        assert R2.shape == (20,)
        assert r.shape == (20,)

        assert np.all(r > 0)
        assert np.all(R2 > 0)

    def test_ensemble_lstsq_parallel(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 50))
        beta0 = rng.standard_normal((50, 20))
        Y = rng.standard_normal((100, 20)) * 0.1 + X @ beta0

        beta_1, Yhat_1, R2_1, r_1 = nb.linalg.ensemble_lstsq(
            X, Y, n_folds=5, n_perms=20, seed=0, n_jobs=1
        )
        beta_2, Yhat_2, R2_2, r_2 = nb.linalg.ensemble_lstsq(
            X, Y, n_folds=5, n_perms=20, seed=0, n_jobs=max(cpu_count(), 2)
        )
        np.testing.assert_allclose(beta_1, beta_2)
        np.testing.assert_allclose(Yhat_1, Yhat_2)
        np.testing.assert_allclose(R2_1, R2_2)
        np.testing.assert_allclose(r_1, r_2)
