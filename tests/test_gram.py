import numpy as np
import pytest
from scipy.linalg import eigh

import neuroboros as nb
from neuroboros.linalg.base import svd_pca
from neuroboros.linalg.gram import gram, gram_pca


class TestGram:
    def test_basic(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 100))
        G = gram(X)
        assert G.shape == (20, 20)
        X_c = X - X.mean(axis=0, keepdims=True)
        np.testing.assert_allclose(G, X_c @ X_c.T)

    def test_remove_mean_false(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 100))
        G = gram(X, remove_mean=False)
        np.testing.assert_allclose(G, X @ X.T)

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
        G_sum = gram(X, split=4, reduce="sum")
        G_stack = gram(X, split=4, reduce="stack")
        assert G_stack.shape == (4, 20, 20)
        np.testing.assert_allclose(G_stack.sum(axis=0), G_sum)

    def test_reduce_list(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 100))
        G_stack = gram(X, split=4, reduce="stack")
        G_list = gram(X, split=4, reduce="list")
        assert isinstance(G_list, list)
        assert len(G_list) == 4
        np.testing.assert_allclose(np.stack(G_list), G_stack)

    def test_reduce_invalid(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 100))
        with pytest.raises(ValueError):
            gram(X, split=4, reduce="invalid")


class TestGramPCA:
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
        G = gram(X)
        PCs = gram_pca(G)
        # Diagonal of PCs.T @ PCs must match the top-(N-1) eigenvalues of G
        w = eigh(G, lower=False, eigvals_only=True)[::-1][:-1]
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
        G = np.array([[1.0, 2.0], [2.0, 1.0]])
        with pytest.raises(AssertionError):
            gram_pca(G)
