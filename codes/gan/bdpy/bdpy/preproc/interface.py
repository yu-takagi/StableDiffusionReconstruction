"""
Interface functions for preprocessing

This file is a part of BdPy
"""


from .preprocessor import Average,Detrender,Normalize,Regressout,ReduceOutlier,ShiftSample
from .util import print_start_msg, print_finish_msg


def average_sample(x, group=[], verbose=True):
    """
    Average samples within groups

    Parameters
    ----------
    x : array
        Input data array (sample num * feature num)
    group : array_like
        Group vector (length = sample num)

    Returns
    -------
    y : array
        Averaged data array (group num * feature num)
    index_map : array_like
        Vector mapping row indexes from y to x (length = group num)
    """

    if verbose:
        print_start_msg()

    p = Average()
    y, ind_map = p.run(x, group)

    if verbose:
        print_finish_msg()

    return y, ind_map


def detrend_sample(x, group=[], keep_mean=True, verbose=True):
    """
    Apply linear detrend

    Parameters
    ----------
    x : array
        Input data array (sample num * feature num)
    group : array_like
        Group vector (length = sample num)

    Returns
    -------
    y : array
        Detrended data array (group num * feature num)
    """

    if verbose:
        print_start_msg()

    p = Detrender()
    y, _ = p.run(x, group, keep_mean=keep_mean)

    if verbose:
        print_finish_msg()

    return y


def normalize_sample(x, group=[], mode='PercentSignalChange', baseline='All',
                     zero_threshold=1, verbose=True):
    """
    Apply normalization

    Parameters
    ----------
    x : array
        Input data array (sample num * feature num)
    group : array_like
        Group vector (length = sample num)
    Mode : str
        Normalization mode ('PercentSignalChange', 'Zscore', 'DivideMean', or 'SubtractMean'; default = 'PercentSignalChange')
    Baseline : array_like or str
        Baseline index vector (default: 'All')
    ZeroThreshold : float
        Zero threshold (default: 1)

    Returns
    -------
    y : array
        Normalized data array (sample num * feature num)
    """

    if verbose:
        print_start_msg()

    p = Normalize()
    y, _ = p.run(x, group, mode = mode, baseline = baseline, zero_threshold = zero_threshold)

    if verbose:
        print_finish_msg()

    return y


def reduce_outlier(x, group=[], std=True, maxmin=True, remove=False, dimension=1, n_iter=10, std_threshold=3, max_value=None, min_value=None, verbose=True):
    '''Outlier reduction.'''

    if verbose:
        print_start_msg()

    if remove:
        raise NotImplementedError('"remove" option is not implemented yet.')
        
    p = ReduceOutlier()
    y, _ = p.run(x, group, std=std, maxmin=maxmin, dimension=dimension, n_iter=n_iter, std_threshold=std_threshold, max_value=max_value, min_value=min_value)

    if verbose:
        print_finish_msg()

    return y
    

def regressout(x, group=[], regressor=[], remove_dc=True, linear_detrend=True, verbose=True):
    '''Remove nuisance regressors.

    Parameters
    ----------
    x : array, shape = (n_sample, n_feature)
        Input data array
    group : array_like, lenght = n_sample
        Group vector.
    regressor : array_like, shape = (n_sample, n_regressor)
        Nuisance regressors.
    remove_dc : bool
        Remove DC component (signal mean) or not (default: True).
    linear_detrend : bool
        Remove linear trend or not (default: True).

    Returns
    -------
    y : array, shape = (n_sample, n_feature)
        Signal without nuisance regressors.
    '''

    if verbose:
        print_start_msg()

    p = Regressout()
    y, _ = p.run(x, group, regressor=regressor, remove_dc=remove_dc, linear_detrend=linear_detrend)

    if verbose:
        print_finish_msg()

    return y


def shift_sample(x, group=[], shift_size = 1, verbose = True):
    """
    Shift sample within groups

    Parameters
    ----------
    x : array
        Input data (sample num * feature num)
    group : array_like
        Group vector (length: sample num)
    shift_size : int
        Shift size (default: 1)

    Returns
    -------
    y : array
        Averaged data array (group num * feature num)
    index_map : array_like
        Vector mapping row indexes from y to x (length: group num)

    Example
    -------

    import numpy as np
    from bdpy.preprocessor import shift_sample

    x = np.array([[  1,  2,  3 ],
                  [ 11, 12, 13 ],
                  [ 21, 22, 23 ],
                  [ 31, 32, 33 ],
                  [ 41, 42, 43 ],
                  [ 51, 52, 53 ]])
    grp = np.array([ 1, 1, 1, 2, 2, 2 ])

    shift_size = 1

    y, index = shift_sample(x, grp, shift_size)

    # >>> y
    # array([[11, 12, 13],
    #        [21, 22, 23],
    #        [41, 42, 43],
    #        [51, 52, 53]])

    # >>> index
    # array([0, 1, 3, 4])
    """

    if verbose:
        print_start_msg()

    p = ShiftSample()
    y, index_map = p.run(x, group, shift_size = shift_size)

    if verbose:
        print_finish_msg()

    return y, index_map
