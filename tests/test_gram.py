import numpy as np
import pytest
from scipy.linalg import eigh

from neuroboros.linalg.base import svd_pca
from neuroboros.linalg.gram import beta2beta, beta2w, gram, gram_pca


class TestKram:
    def test_basic(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 100))
        K = gram(X)
        assert K.shape == (20, 20)
        X_c = X - X.mean(axis=0, keepdims=True)
        np.testing.assert_allclose(K, X_c @ X_c.T)

    def test_remove_mean_false(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 100))
        K = gram(X, remove_mean=False)
        np.testing.assert_allclose(K, X @ X.T)

    def test_already_centered(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 100))
        X -= X.mean(axis=0, keepdims=True)
        np.testing.assert_allclose(gram(X), gram(X, remove_mean=False))

    def test_split_int(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 100))
        np.testing.assert_allclose(gram(X, split=4), gram(X))

    def test_split_array(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 100))
        np.testing.assert_allclose(gram(X, split=np.array([25, 50, 75])), gram(X))

    def test_reduce_stack(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 100))
        K_sum = gram(X, split=4, reduce="sum")
        K_stack = gram(X, split=4, reduce="stack")
        assert K_stack.shape == (4, 20, 20)
        np.testing.assert_allclose(K_stack.sum(axis=0), K_sum)

    def test_reduce_list(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 100))
        K_stack = gram(X, split=4, reduce="stack")
        K_list = gram(X, split=4, reduce="list")
        assert isinstance(K_list, list)
        assert len(K_list) == 4
        np.testing.assert_allclose(np.stack(K_list), K_stack)

    def test_reduce_invalid(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 100))
        with pytest.raises(ValueError):
            gram(X, split=4, reduce="invalid")


class TestKramPCA:
    def test_shape(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 100))
        PCs = gram_pca(gram(X))
        assert PCs.shape == (20, 19)

    def test_orthogonality(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 100))
        PCs = gram_pca(gram(X))
        # PCs = U * s, so PCs.T @ PCs must be diagonal
        inner = PCs.T @ PCs
        np.testing.assert_allclose(np.triu(inner, k=1), 0, atol=1e-10)

    def test_eigenvalues(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 100))
        K = gram(X)
        PCs = gram_pca(K)
        # Diagonal of PCs.T @ PCs must match the top-(N-1) eigenvalues of K
        w = eigh(K, lower=False, eigvals_only=True)[::-1][:-1]
        np.testing.assert_allclose(np.diag(PCs.T @ PCs), w, atol=1e-10)

    def test_consistency_with_svd(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 100))
        PCs = gram_pca(gram(X))
        # svd_pca returns all K components; drop the last to match gram_pca's N-1
        scores = svd_pca(X)[:, :-1]
        signs = np.sign(np.sum(PCs * scores, axis=0))
        np.testing.assert_allclose(PCs, scores * signs, atol=1e-10)

    def test_negative_eigenvalue_raises(self):
        # eigenvalues are 3 and -1 — not positive semidefinite
        K = np.array([[1.0, 2.0], [2.0, 1.0]])
        with pytest.raises(AssertionError):
            gram_pca(K)

    def test_return_us_shapes(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 100))
        K = gram(X)
        U, s = gram_pca(K, return_us=True)
        assert U.shape == (20, 19)
        assert s.shape == (19,)

    def test_return_us_consistent(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 100))
        K = gram(X)
        PCs = gram_pca(K)
        U, s = gram_pca(K, return_us=True)
        np.testing.assert_allclose(U * s[np.newaxis], PCs)

    def test_return_us_orthonormal(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 100))
        U, _ = gram_pca(gram(X), return_us=True)
        np.testing.assert_allclose(U.T @ U, np.eye(19), atol=1e-10)


class TestBeta2W:
    def test_shape(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 100))
        K = gram(X)
        beta = rng.standard_normal((19,))
        assert beta2w(beta, K).shape == (20,)

    def test_feature_weights(self):
        # X_c.T @ beta2w(beta, K) should equal beta2beta(beta, K, X)
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 100))
        K = gram(X)
        beta = rng.standard_normal((19,))
        X_c = X - X.mean(axis=0)
        np.testing.assert_allclose(
            X_c.T @ beta2w(beta, K), beta2beta(beta, K, X), atol=1e-10
        )


class TestBeta2Beta:
    def test_shape(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 100))
        K = gram(X)
        beta = rng.standard_normal((19,))
        assert beta2beta(beta, K, X).shape == (100,)

    def test_predictions_match(self):
        # Predictions from PC-space model must equal those from feature-space model
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 100))
        K = gram(X)
        PCs = gram_pca(K)
        beta = rng.standard_normal((19,))
        b = float(rng.standard_normal())
        y_pc = PCs @ beta + b
        beta_orig, shift = beta2beta(beta, K, X, return_shift=True)
        np.testing.assert_allclose(X @ beta_orig + (b - shift), y_pc, atol=1e-10)

    def test_return_shift_false(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 100))
        K = gram(X)
        beta = rng.standard_normal((19,))
        result = beta2beta(beta, K, X, return_shift=False)
        assert isinstance(result, np.ndarray)

    def test_centered_x_zero_shift(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 100))
        X -= X.mean(axis=0)
        K = gram(X)
        beta = rng.standard_normal((19,))
        _, shift = beta2beta(beta, K, X, return_shift=True)
        np.testing.assert_allclose(shift, 0, atol=1e-10)
