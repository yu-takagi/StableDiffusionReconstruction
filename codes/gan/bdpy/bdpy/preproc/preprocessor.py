"""
Classes for preprocessing

This file is a part of BdPy.
"""


import copy
import sys
from abc import ABCMeta, abstractmethod

import numpy as np
from numpy.matlib import repmat


## Abstract preprocessor #######################################################

class Preprocessor(object):
    """
    Abstract class for preprocessing
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def proc(self, x, ind, opt):
        """
        Abstract method of preprocessing

        `proc` should return `y` and `ind_map`
        """
        pass

    def run(self, x, group=[], **kargs):
        """
        Template method of preprocessing
        """

        # If `group` is empty, apply preprocessing to the whole data
        if len(group) == 0:
            group = np.ones(x.shape[0])

        group = np.array(group) # Input `group` can be either np.array or list

        group_set = sorted(list(set(group)))

        prec_data = []
        ind_maps = []

        for g in group_set:
            group_index = np.where(group == g)[0]
            group_data = x[group_index]

            pdata, pind = self.proc(group_data, group_index, kargs)

            prec_data.append(pdata)
            ind_maps.append(pind)

        y = np.vstack(prec_data)
        index_map = np.hstack(ind_maps) # `index_map` should be a vector

        return y, index_map


## Preprocessor classes ########################################################

class Average(Preprocessor):

    def proc(self, x, ind, opt):

        x_ave = np.average(x, axis = 0)

        ind_map = ind[0]

        return x_ave, ind_map


class Detrender(Preprocessor):

    def proc(self, x, ind, opt):

        from scipy.signal import detrend

        keep_mean = opt['keep_mean']

        x_mean = np.mean(x, axis = 0)
        x_detl = detrend(x, axis = 0, type = 'linear')

        if keep_mean:
            x_detl += x_mean

        ind_map = ind

        return x_detl, ind_map


class Normalize(Preprocessor):

    def proc(self, x, ind, opt):

        mode = opt['mode']
        baseline = opt['baseline']

        if baseline == 'All':
            x_mean = np.mean(x, axis=0)
            x_std = np.std(x, axis=0)
        else:
            # Baseline contains 'True' and 'False' with a length of runs.size
            ind_baseline = baseline[ind]
            x_mean = np.mean(x[ind_baseline, :], axis=0)
            x_std = np.std(x[ind_baseline, :], axis=0)

        if mode == "PercentSignalChange":
            # zero division outputs nan
            # np.nan_to_num convert nan to 0.
            # Is this correct? Should cells divided by zero return inf?
            x = np.nan_to_num(np.divide(100.0 * (x - x_mean), x_mean))
        elif mode == "Zscore":
            x = np.nan_to_num(np.divide((x - x_mean), x_std))
        elif mode == "DivideMean":
            x = np.nan_to_num(np.divide(100.0 * x, x_mean))
        elif mode == "SubtractMean":
            x = x - x_mean
        else:
            NameError("Unknown normalization mode: %s', norm_mode" % (mode))

        ind_map = ind

        return x, ind_map


class ReduceOutlier(Preprocessor):

    def proc(self, x, ind, opt):
        std = opt['std']                      # Bool
        maxmin = opt['maxmin']                # Bool
        dim = opt['dimension']                # int
        n_iter = opt['n_iter']                # int
        std_threshold = opt['std_threshold']  # float
        max_value = opt['max_value']          # float
        min_value = opt['min_value']          # float

        # TODO: add remove operation

        dim = dim - 1

        y = copy.deepcopy(x)

        if std:
            # Reduce outliers by SD
            for i in range(n_iter):
                mu = np.mean(y, axis=dim)
                sd = np.std(y, axis=dim)

                thres_up = mu + sd * std_threshold
                thres_lw = mu - sd * std_threshold

                if dim == 0:
                    thres_up = repmat(thres_up, y.shape[0], 1)
                    thres_lw = repmat(thres_lw, y.shape[0], 1)
                elif dim == 1:
                    thres_up = repmat(np.c_[thres_up], 1, y.shape[1])
                    thres_lw = repmat(np.c_[thres_lw], 1, y.shape[1])

                out_ind_up = thres_up < y
                out_ind_lw = thres_lw > y

                # Clip outliers
                y[out_ind_up] = thres_up[out_ind_up]
                y[out_ind_lw] = thres_lw[out_ind_lw]

            num_out = np.sum(out_ind_up) + np.sum(out_ind_lw)
            print('Num outliers (SD): %d (%f %%)' % (num_out, 100.0 * num_out / y.size))

        if maxmin:
            # Reduce outliers by max-min values
            if max_value is not None:
                out_ind_max = y > max_value
                y[out_ind_max] = max_value
            if min_value is not None:
                out_ind_min = y < min_value
                y[out_ind_min] = min_value

        ind_map = ind

        return y, ind_map


class Regressout(Preprocessor):

    def proc(self, x, ind, opt):

        regressor = opt['regressor']            # Numpy array
        remove_dc = opt['remove_dc']            # Bool
        linear_detrend = opt['linear_detrend']  # Bool

        regressor = regressor[ind, :]

        n_smp = x.shape[0]

        dc_cmp = np.ones((n_smp, 1))  # DC component (mean)

        if linear_detrend:
            ln_cmp = np.c_[(np.arange(n_smp) + 1) / np.float(n_smp)]
            regmat = np.hstack([dc_cmp, ln_cmp, regressor])
        else:
            regmat = np.hstack([dc_cmp, regressor])

        try:
            w = np.linalg.solve(np.dot(regmat.T, regmat),  np.dot(regmat.T, x))
        except:
            print('Error with np.linalg.solve. Trying np.linalg.lstsq')
            w = np.linalg.lstsq(np.dot(regmat.T, regmat),  np.dot(regmat.T, x))[0]

        y = x - np.dot(regmat, w)

        if not remove_dc:
            dc = np.mean(x, axis=0)
            y = y + dc

        # No index mapping
        ind_map = ind

        return y, ind_map


class ShiftSample(Preprocessor):

    def proc(self, x, ind, opt):

        s = opt['shift_size']

        y = x[s:]
        ind_map = ind[:-s]

        return y, ind_map
