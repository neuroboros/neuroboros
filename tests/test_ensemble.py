import numpy as np
import pytest

import neuroboros as nb


class TestEnsemble:
    def test_kfold_bagging(self):
        n = 500
        train_idx_li, test_idx_li = nb.ensemble.kfold_bagging(
            n, n_folds=20, n_perms=20, seed=0
        )

        test_idx = np.concatenate(test_idx_li)
        uu, cc = np.unique(test_idx, return_counts=True)
        counts_test = np.zeros((n,))
        counts_test[uu] = cc
        assert np.all(counts_test >= 20)

        for train_idx, test_idx in zip(train_idx_li, test_idx_li):
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
