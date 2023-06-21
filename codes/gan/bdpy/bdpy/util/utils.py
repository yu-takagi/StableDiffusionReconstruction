"""
Utility functions.

This file is a part of BdPy.
"""


from __future__ import division


__all__ = ['create_groupvector',
           'divide_chunks',
           'get_refdata',
           'makedir_ifnot']


import os
import warnings

import numpy as np


def create_groupvector(group_label, group_size):
    """Create a group vector.

    Parameters
    ----------
    group_label : array_like
        List or array of group labels.
    group_size : array_like
        Sample size of each group.

    Returns
    -------
    group_vector : array_like, shape = (N,)
        A vector specifying groups.

    Example
    -------

        >>> bdpy.util.create_groupvector([ 1, 2, 3 ], 2)
        array([1, 1, 2, 2, 3, 3])

        >>> bdpy.util.create_groupvector([ 1, 2, 3 ], [ 2, 4, 2 ])
        array([1, 1, 2, 2, 2, 2, 3, 3])
    """

    group_vector = []

    if isinstance(group_size, int):
        # When 'group_size' is integer, create array in which each group label
        # is repeated for 'group_size'
        group_size_list = [group_size for _ in range(len(group_label))]
    elif isinstance(group_size, list) | isinstance(group_size, np.ndarray):
        if len(group_label) != len(group_size):
            raise ValueError("Length of 'group_label' and 'group_size' "
                             "is mismatched")
        group_size_list = group_size
    else:
        raise TypeError("Invalid type of 'group_size'")

    group_list = [np.array([label for _ in range(group_size_list[i])])
                  for i, label in enumerate(group_label)]
    group_vector = np.hstack(group_list)

    return group_vector


def divide_chunks(input_list, chunk_size=100):
    '''Divide elements in the input list into groups.

    Parameters
    ----------
    input_list : array_like
        List or array to be divided.
    chunk_size : int
        The number of elements in each chunk.

    Returns
    -------
    list
        List of chunks.

    Example
    -------

        >>> a = [0, 1, 2, 3, 4, 5, 6]
        >>> divide_chunks(a, chunk_size=2)
        [[0, 1], [2, 3], [4, 5], [6]]
        >>> divide_chunks(a, chunk_size=3)
        [[0, 1, 2], [3, 4, 5], [6]]
    '''
    n_chunk = np.int(np.ceil(len(input_list) / chunk_size))
    chunks = [input_list[i * chunk_size:(i + 1) * chunk_size]
              for i in range(n_chunk)]
    return chunks


def get_refdata(data, ref_key, foreign_key):
    """Get data referred by `foreign_key`.

    Parameters
    ----------
    data : array_like
        Data array.
    ref_key
        Reference keys for `data`.
    foreign_key
        Foreign keys referring `data` via `ref_key`.

    Returns
    -------
    array_like
        Referred data.
    """

    ind = [np.where(ref_key == i)[0][0] for i in foreign_key]

    if data.ndim == 1:
        return data[ind]
    else:
        return data[ind, :]


def makedir_ifnot(dir_path):
    '''Make a directory if it does not exist.

    Parameters
    ----------
    dir_path : str
        Path to the directory to be created.

    Returns
    -------
    bool
        True if the directory was created.
    '''
    if not os.path.isdir(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError:
            warnings.warn('Failed to create directory %s.' % dir_path)
            return False
        return True
    else:
        return False
