import numpy as np


def kfold_bagging(n, n_folds=5, n_perms=20, seed=0):
    """Generate train and test indices for k-fold bagging.

    Similar to bagging, but ensures that each sample is used as a test sample
    at least once in each permutation by incorporating k-fold cross-validation.

    In each permutation, the samples are randomly shuffled and divided into
    k folds. Each time, samples in one of the k folds is withheld as test
    candidates, and the remaining samples are used as training candidates.
    The training candidates are then resampled with replacement to generate
    the final training set. The remaining samples are used as the test set.
    In other words, the test set is the union of the withheld fold and the
    training candidates that were not selected for.

    The total number of train-test pairs is n_perms * n_folds.

    Parameters
    ----------
    n : int
        Number of samples.
    n_folds : int, default=5
        Number of folds.
    n_perms : int, default=20
        Number of permutations.
    seed : int, default=0
        Random seed for the random number generator.

    Returns
    -------
    indices_li : list of tuples
        List of tuples of training and test indices.
    """
    rng = np.random.default_rng(seed=seed)
    arng = np.arange(n)

    indices_li = []
    for i in range(n_perms):
        folds = np.array_split(rng.permutation(n), n_folds)
        for test_idx in folds:
            train_idx = np.setdiff1d(arng, test_idx)
            train_idx = rng.choice(train_idx, size=n, replace=True)
            test_idx = np.setdiff1d(arng, train_idx)
            indices_li.append((train_idx, test_idx))

    return indices_li
