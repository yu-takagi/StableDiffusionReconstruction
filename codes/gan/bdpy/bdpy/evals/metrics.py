'''bdpy.evals.metrics'''

import warnings

import numpy as np
from scipy.spatial.distance import cdist


def profile_correlation(x, y):
    '''Profile correlation.'''

    sample_axis = 0

    orig_shape = x.shape
    n_sample = orig_shape[sample_axis]

    _x = x.reshape(n_sample, -1)
    _y = y.reshape(n_sample, -1)

    n_feat = _y.shape[1]

    r = np.array(
        [
            np.corrcoef(
                _x[:, j].flatten(),
                _y[:, j].flatten()
            )[0, 1]
            for j in range(n_feat)
        ]
    )

    r = r.reshape((1,) + orig_shape[1:])

    return r


def pattern_correlation(x, y, mean=None, std=None, remove_nan=True):
    '''Pattern correlation.'''

    sample_axis = 0

    orig_shape = x.shape
    n_sample = orig_shape[sample_axis]

    _x = x.reshape(n_sample, -1)
    _y = y.reshape(n_sample, -1)

    if mean is not None and std is not None:
        m = mean.reshape(-1)
        s = std.reshape(-1)

        _x = (_x - m) / s
        _y = (_y - m) / s

    if remove_nan:
        # Remove nan columns based on the decoded features
        nan_cols = np.isnan(_x).any(axis=0) | np.isnan(_y).any(axis=0)
        if nan_cols.any():
            warnings.warn('NaN column removed ({})'.format(np.sum(nan_cols)))
        _x = _x[:, ~nan_cols]
        _y = _y[:, ~nan_cols]

    r = np.array(
        [
            np.corrcoef(
                _x[i, :].flatten(),
                _y[i, :].flatten()
            )[0, 1]
            for i in range(n_sample)
        ]
    )

    return r


def pairwise_identification(pred, true, metric='correlation', remove_nan=True, remove_nan_dist=True, single_trial=False, pred_labels=None, true_labels=None):
    '''Pair-wise identification.'''

    p = pred.reshape(pred.shape[0], -1)
    t = true.reshape(true.shape[0], -1)

    if remove_nan:
        # Remove nan columns based on the decoded features
        nan_cols = np.isnan(p).any(axis=0) | np.isnan(t).any(axis=0)
        if nan_cols.any():
            warnings.warn('NaN column removed ({})'.format(np.sum(nan_cols)))
        p = p[:, ~nan_cols]
        t = t[:, ~nan_cols]

    if single_trial:
        cr = []
        for i in range(p.shape[0]):
            d = 1 - cdist(p[i][np.newaxis], t, metric=metric)
            # label の情報
            ind = np.where(np.array(true_labels) == pred_labels[i])[0][0]

            s = (d - d[0, ind]).flatten()
            if remove_nan_dist and np.isnan(s).any():
                warnings.warn('NaN value detected in the distance matrix ({}).'.format(np.sum(np.isnan(s))))
                s = s[~np.isnan(s)]
            ac = np.sum(s < 0) / (len(s) - 1)
            cr.append(ac)
        cr = np.asarray(cr)
    else:
        d = 1 - cdist(p, t, metric=metric)

        if remove_nan_dist:
            cr = []
            for d_ind in range(d.shape[0]):
                pef = d[d_ind, :] - d[d_ind, d_ind]
                if np.isnan(pef).any():
                    warnings.warn('NaN value detected in the distance matrix ({}).'.format(np.sum(np.isnan(pef))))
                    pef = pef[~np.isnan(pef)] # Remove nan value from the comparison for identification
                pef = np.sum(pef < 0) / (len(pef) - 1)
                cr.append(pef)
            cr = np.asarray(cr)
        else:
            cr = np.sum(d - np.diag(d)[:, np.newaxis] < 0, axis=1) / (d.shape[1] - 1)

    return cr


def remove_nan_value(array, nan_flag=None, return_nan_flag=False):
    '''Helper function:
    Remove columns (units) which contain nan values

    array (numpy.array) ... shape should be [sample x units]
    nan_flag (numpy.array or list) ... if exist, remove columns according to the nan_flag
    return_nan_flag (bool) ... if True, return nan_flag to remove columns of the array.

    '''

    if nan_flag is None:
        nan_flag = np.isnan(array).any(axis=0)
    nan_removed_array = array[:, ~nan_flag]

    if return_nan_flag:
        return nan_removed_array, nan_flag
    else:
        return nan_removed_array
