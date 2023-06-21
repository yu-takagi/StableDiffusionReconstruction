"""
Utilities for ensemble learning
"""

from collections import Counter

import numpy as np


__all__ = ['get_majority']


def get_majority(data, axis=0):
    """
    Returns a list of majority elements in each row or column.

    If more than two elements occupies the same numbers in each row or column,
    'get_majority' returns the first-sorted element.

    Parameters
    ----------
    data : array_like
    axis : 0 or 1, optional
        Axis in which elements are counted (default: 0)


    Returns
    -------
    majority_list : list
        A list of majority elements
    """

    majority_list = []

    if axis == 0:
        data = np.transpose(data) 

    for i in range(data.shape[0]):
        target = data[i].tolist()
        # Change KS for returning first element if the same numbers
        #c = Counter(target)
        c = Counter(np.sort(target))
        majority = c.most_common(1)
        majority_list.append(majority[0][0])

    return majority_list
