import numpy as np
from scipy.stats import zscore

import neuroboros as nb


class TestSpearmanBrown:
    def test_spearman_brown(self):
        rng = np.random.default_rng(0)
        for i in range(10):
            r1 = rng.random()
            n = rng.integers(2, 100)
            rn = nb.stats.spearman_brown(r1, n)
            assert rn > r1
            assert rn <= 1
            inv = nb.stats.spearman_brown(rn, n, inverse=True)
            np.testing.assert_allclose(inv, r1)

    def test_spearman_brown_ndarray(self):
        rng = np.random.default_rng(0)
        for i in range(10):
            r1 = rng.random((4, 5))
            n = rng.integers(2, 100, size=(4, 5))
            rn = nb.stats.spearman_brown(r1, n)
            np.testing.assert_array_less(r1, rn)
            np.testing.assert_array_less(rn, 1.0)
            inv = nb.stats.spearman_brown(rn, n, inverse=True)
            np.testing.assert_allclose(inv, r1)


class TestAlpha:
    def test_alpha(self):
        rng = np.random.default_rng(0)
        n = 100
        for i in range(10):
            X = rng.standard_normal((2, n))
            X = zscore(X, axis=1)
            alpha = nb.stats.cronbach_alpha(X)
            r1 = np.mean(X[0] * X[1])
            r2 = nb.stats.spearman_brown(r1, 2)
            np.testing.assert_allclose(alpha, r2)

    def test_alpha_ndarray(self):
        rng = np.random.default_rng(0)
        n = 100
        m = 5
        for i in range(10):
            X = rng.standard_normal((2, m, n))
            alphas1 = nb.stats.cronbach_alpha(X, var_axis=2)
            alphas2 = np.array([nb.stats.cronbach_alpha(X[:, i]) for i in range(m)])
            np.testing.assert_array_equal(alphas1, alphas2)

    def test_alpha_ci(self):
        rng = np.random.default_rng(0)
        n = 100
        m = 5
        for i in range(10):
            X = rng.standard_normal((2, m, n))
            alphas1, cis1 = nb.stats.cronbach_alpha(X, var_axis=2, ci=0.95)
            returns = [
                nb.stats.cronbach_alpha(X[:, i], ci=[0.025, 0.975]) for i in range(m)
            ]
            alphas2 = np.stack([_[0] for _ in returns])
            cis2 = np.stack([_[1] for _ in returns], axis=1)
            np.testing.assert_array_equal(alphas1, alphas2)
            np.testing.assert_array_equal(cis1, cis2)
        np.testing.assert_array_less(cis1[:, 0], cis1[:, 1])
