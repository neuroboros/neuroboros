"""
===================================================
Linear algebra utilities (:mod:`neuroboros.linalg`)
===================================================

.. currentmodule:: neuroboros.linalg

.. autosummary::
    :toctree:

    safe_svd - Singular value decomposition without occasional LinAlgError crashes.
    safe_polar - Polar decomposition without occasional LinAlgError crashes.
    gram_pca - Principal component analysis based on the Gram matrix.
    ensemble_lstsq - Linear regression with k-fold bagging.

"""

from .base import ensemble_lstsq, safe_polar, safe_svd
from .gram import gram_pca

__all__ = [
    "safe_svd",
    "safe_polar",
    "gram_pca",
    "ensemble_lstsq",
]
