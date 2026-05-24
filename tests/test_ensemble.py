import numpy as np
import pytest
from joblib import cpu_count

import neuroboros as nb
from neuroboros.ensemble import kfold_bagging_groups


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

    def test_kfold_bagging_groups_no_group_overlap(self):
        rng = np.random.default_rng(42)
        sizes = rng.choice([1, 2, 3, 4], size=30)
        groups, idx = [], 0
        for s in sizes:
            groups.append(np.arange(idx, idx + s))
            idx += s
        n_perms, n_folds = 10, 5
        splits = kfold_bagging_groups(groups, n_folds=n_folds, n_perms=n_perms, seed=0)
        assert len(splits) == n_perms * n_folds
        for train_idx, test_idx in splits:
            assert len(np.intersect1d(train_idx, test_idx)) == 0
            for g in groups:
                in_train = np.isin(g, train_idx).any()
                in_test = np.isin(g, test_idx).any()
                assert not (in_train and in_test)

    def test_kfold_bagging_groups_each_sample_tested_n_perms_times(self):
        rng = np.random.default_rng(42)
        sizes = rng.choice([1, 2, 3, 4], size=30)
        groups, idx = [], 0
        for s in sizes:
            groups.append(np.arange(idx, idx + s))
            idx += s
        n = idx
        n_perms, n_folds = 10, 5
        splits = kfold_bagging_groups(groups, n_folds=n_folds, n_perms=n_perms, seed=0)
        all_test = np.concatenate([test_idx for _, test_idx in splits])
        counts = np.bincount(all_test, minlength=n)
        assert np.all(counts >= n_perms)

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
