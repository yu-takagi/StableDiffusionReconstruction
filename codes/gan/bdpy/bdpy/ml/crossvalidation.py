'''Cross-validation functions

This module provides utility functions for cross-validation.
'''


import numpy as np


def cvindex_groupwise(group, nfolds=None, return_bool=False, exclusive=None):
    '''Return k-fold iterator for group-wise cross-validation (e.g, run-wise, block-wise, ...)

    n_folds` specification are not supported yet.

    Parameters
    ----------
    group : array-like (shape =  (n_samples, )
        Group labels (e.g., run labels, block labels, ...)
    n_folds : int, optional
        Number of folds (default: the number of unique elements in `group`)
    return_bool : bool, optional
        Return boolean arrays if True (default: False)
    exclusive : array_like, optional
        If specified, training samples that have the same labels as the test
        data are removed in each fold.

    Returns
    -------
    K-fold iterator
    '''

    # TODO: add size checking of group

    group_set = np.unique(group)  # Unique labels in `group`

    if nfolds is None:
        nfolds = len(group_set)

    if exclusive is not None:
        exclusive = exclusive.flatten()

    for gl in group_set:
        index_train_bool = (group != gl).flatten()
        index_test_bool = (group == gl).flatten()

        if return_bool:
            index_train = index_train_bool
            index_test = index_test_bool
        else:
            index_train = np.where(index_train_bool)[0]
            index_test = np.where(index_test_bool)[0]

        if exclusive is not None:
            test_labels  = exclusive[index_test]
            train_labels = exclusive[index_train]

            index_train = np.array([
                ind for ind, lab in zip(index_train, train_labels) if not lab in test_labels
            ])

        yield index_train, index_test


def make_cvindex(group):
    """
    Make indexes of training and test samples for cross-validation

    Parameters
    ----------
    group : array_like
        Index that partitions samples to groups (N * 1; N is the total number of
        samples). If 'group' is runs_groups, 'make_cvindex' returns indexes for
        cross-validation between run (leave-run-out). If all elements in 'group'
        are 0 or 1, 'make_cvindex' returns index for training only or test only,
        respectively.

    Returns
    -------
    train_index : array_like
        Boolean matrix specifying training samples (N * K; N is the total number
        of samples, K is the number of folds)
    test_index : array_like
        Boolean matrix specifying test samples (N * K)

    Example
    -------

        >>> bdpy.util.make_crossvalidationindex(np.array([1, 1, 2, 2, 3, 3]))
        (array([[False,  True,  True],
                [False,  True,  True],
                [ True, False,  True],
                [ True, False,  True],
                [ True,  True, False],
                [ True,  True, False]], dtype=bool),
         array([[ True, False, False],
                [ True, False, False],
                [False,  True, False],
                [False,  True, False],
                [False, False,  True],
                [False, False,  True]], dtype=bool))
    """

    # Get and sort unique group index
    group_set = sorted(list(set(group.flatten())))

    # The number of sample
    n_sample = len(group);
    # The number of run
    n_group = len(group_set);

    if n_group == 0:
        # Training only
        train_index = np.ones((n_sample,))
        test_index = np.zeros((n_sample,))
    elif n_group == 1:
        # Test only
        train_index = np.zeros((n_sample,))
        test_index = np.ones((n_sample,))
    else:
        # K-fold crossvalidation
        group_t = np.reshape(group, (group.shape[0], 1))
        mat_group = np.tile(group_t, (1, n_group))
        mat_group_set = np.tile(np.array(group_set), (n_sample, 1))

        train_index = mat_group != mat_group_set
        test_index = mat_group == mat_group_set

    return train_index, test_index


def make_crossvalidationindex(group):
    """
    Make indexes of training and test samples

    See 'make_cvindex' for the details.
    """

    return make_cvindex(group)


def make_cvindex_generator(group, folds=None, exclusive=None, return_bool=False):
    '''Return cross-validation iterator.

    n_folds` specification are not supported yet.

    Parameters
    ----------
    group : array-like (shape =  (n_samples, )
        Group labels (e.g., run labels, block labels, ...)
    folds : list, optional
        List of CV folds(training and test labels in each fold)
    return_bool : bool, optional
        Return boolean arrays if True (default: False)
    exclusive : array_like, optional
        If specified, training samples that have the same labels as the test
        data are removed in each fold.

    Returns
    -------
    K-fold iterator
    '''


    if folds is None:
        group_set = np.unique(group)
        folds = [{'train': np.delete(group_set, i), 'test': gl} for i, gl in enumerate(group_set)]

    if exclusive is not None:
        exclusive = exclusive.flatten()

    for f in folds:
        ind_train = f['train']
        ind_test  = f['test']

        #index_train_bool = (group != gl).flatten()
        #index_test_bool = (group == gl).flatten()

        index_train_bool = np.isin(group, ind_train)
        index_test_bool  = np.isin(group, ind_test)

        if return_bool:
            index_train = index_train_bool
            index_test = index_test_bool
        else:
            index_train = np.where(index_train_bool)[0]
            index_test = np.where(index_test_bool)[0]

        if exclusive is not None:
            test_labels  = exclusive[index_test]
            train_labels = exclusive[index_train]

            index_train = np.array([
                ind for ind, lab in zip(index_train, train_labels) if not lab in test_labels
            ])

        yield index_train, index_test
