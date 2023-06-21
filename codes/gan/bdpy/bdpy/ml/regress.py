"""
This file is a part of BdPy
"""


__all__ = ['add_bias']


import numpy as np


def add_bias(x, axis=0):
    """
    Add bias terms to x

    Parameters
    ----------
    x : array_like
        Data matrix
    axis : 0 or 1, optional
        Axis in which bias terms are added (default: 0)

    Returns
    -------
    y : array_like
        Data matrix with bias terms
    """

    if axis == 0:
        vlen = x.shape[1]
        y = np.concatenate((x, np.array([np.ones(vlen)])), axis=0)
    elif axis == 1:
        vlen = x.shape[0]
        y = np.concatenate((x, np.array([np.ones(vlen)]).T), axis=1)
    else:
        raise ValueError('axis should be either 0 or 1')
    
    return y
