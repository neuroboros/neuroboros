import numpy as np


def kfold_bagging(n, n_folds=5, n_reps=20, seed=0):
    """Generate train and test indices for k-fold bagging.

    Similar to bagging, but ensures that each sample is used as a test sample
    at least once in each repetition by incorporating k-fold cross-validation.

    In each repetition, the samples are randomly shuffled and divided into
    k folds. Each time, samples in one of the k folds is withheld as test
    candidates, and the remaining samples are used as training candidates.
    The training candidates are then resampled with replacement to generate
    the final training set. The remaining samples are used as the test set.
    In other words, the test set is the union of the withheld fold and the
    training candidates that were not selected for.

    The total number of train-test pairs is n_reps * n_folds.

    Parameters
    ----------
    n : int
        Number of samples.
    n_folds : int, default=5
        Number of folds.
    n_reps : int, default=20
        Number of repetitions of the k-fold procedure.
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
    for i in range(n_reps):
        folds = np.array_split(rng.permutation(n), n_folds)
        for test_idx in folds:
            train_idx = np.setdiff1d(arng, test_idx)
            train_idx = rng.choice(train_idx, size=n, replace=True)
            test_idx = np.setdiff1d(arng, train_idx)
            indices_li.append((train_idx, test_idx))

    return indices_li


def kfold_bagging_groups(groups, n_folds=5, n_reps=20, seed=0):
    """Generate train and test indices for k-fold bagging with group constraints.

    Like kfold_bagging, but operates at the group level so that members of the
    same group are never split across training and test sets. Groups are assigned
    to folds while balancing by sample count: larger groups are placed first,
    and each group is assigned to a randomly chosen fold that still has room.

    The total number of train-test pairs is n_reps * n_folds.

    Notes
    -----
    The fold assignment assumes that there are enough small groups (e.g.,
    single-member groups) to fill folds to their expected size. If all
    groups are large relative to the expected fold size, a group may not
    fit into any fold, causing an infinite loop. For example, with two
    groups of sizes 1 and 5 and n_folds=2, the expected fold size is 3,
    but the group of size 5 cannot be placed.

    Parameters
    ----------
    groups : list of arrays
        Each array contains the sample indices of members of one group.
    n_folds : int, default=5
        Number of folds.
    n_reps : int, default=20
        Number of repetitions of the k-fold procedure.
    seed : int, default=0
        Random seed for the random number generator.

    Returns
    -------
    indices_li : list of tuples
        List of tuples of training and test indices.
    """
    rng = np.random.default_rng(seed=seed)
    groups = [np.asarray(g) for g in groups]
    n_groups = len(groups)
    all_indices = np.concatenate(groups)
    expected_sizes = [len(_) for _ in np.array_split(all_indices, n_folds)]

    size_to_idxs = {}
    for i, g in enumerate(groups):
        size_to_idxs.setdefault(len(g), []).append(i)
    lengths = sorted(size_to_idxs.keys())[::-1]

    indices_li = []
    for _ in range(n_reps):
        d = {
            l: list(idxs) for l, idxs in size_to_idxs.items()
        }  # copy because pop() mutates it
        folds = [[] for _ in range(n_folds)]
        current_sizes = np.zeros(n_folds, dtype=int)
        for l in lengths:
            while d[l]:
                idx = d[l].pop(rng.choice(len(d[l])))
                choices = [
                    k
                    for k in range(n_folds)
                    if current_sizes[k] + l <= expected_sizes[k]
                ]
                if not choices:
                    raise ValueError(
                        f"Group of size {l} cannot fit in any fold. "
                        f"Ensure there are enough small groups to balance the folds."
                    )
                k = rng.choice(choices)
                folds[k].append(idx)
                current_sizes[k] += l

        for withheld_idxs in folds:
            candidate_group_idxs = np.setdiff1d(np.arange(n_groups), withheld_idxs)
            candidates = np.concatenate([groups[i] for i in candidate_group_idxs])
            train_idx = rng.choice(candidates, size=len(all_indices), replace=True)
            train_idx = np.unique(train_idx)
            test_idx = np.concatenate(
                [g for g in groups if not np.isin(g, train_idx).any()]
            )
            indices_li.append((train_idx, test_idx))

    return indices_li


def permute_groups(groups, n_perms, seed=0):
    """Generate permutations of group members.

    In each permutation, groups of the same size are randomly reordered among
    themselves, and members within each placed group are shuffled. Members
    never cross group boundaries.

    Parameters
    ----------
    groups : list of arrays
        Each array contains the sample indices of members of one group.
        ``np.sort(np.concatenate(groups))`` must equal ``np.arange(n)``.
    n_perms : int
        Number of permutations to generate.
    seed : int, default=0
        Random seed for the random number generator.

    Returns
    -------
    perms : ndarray of int, shape (n_perms, n)
        Each row is a permutation array. ``data[perms[p]]`` gives the
        permuted data for permutation ``p``.
    """
    rng = np.random.default_rng(seed)
    groups = [np.asarray(g) for g in groups]
    n = sum(len(g) for g in groups)

    size_to_group_idxs = {}
    for i, g in enumerate(groups):
        size_to_group_idxs.setdefault(len(g), []).append(i)

    perms = np.empty((n_perms, n), dtype=int)
    for p in range(n_perms):
        perm = perms[p]
        for size, group_idxs in size_to_group_idxs.items():
            shuffled_idxs = rng.permutation(group_idxs)
            for new_pos, old_idx in zip(group_idxs, shuffled_idxs):
                perm[groups[new_pos]] = rng.permutation(groups[old_idx])

    return perms
