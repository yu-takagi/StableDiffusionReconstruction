'''Loading EPIs module.

This file is a part of BdPy.
'''


import itertools as itr
import os
import re
import string

import nipy
import numpy as np
import scipy.io as sio


def load_epi(datafiles):
    '''Load EPI files.

    The returned data and xyz are flattened by C-like order.

    Parameters
    ----------
    datafiles: list
        EPI image files.

    Returns
    -------
    data: array_like, shape = (M, N)
        Voxel signal values (M: the number of samples, N: the nubmer of
        voxels).
    xyz_array: array_like, shape = (3, N)
        XYZ Coordiantes of voxels.
    '''

    data_list = []
    xyz = np.array([])

    for df in datafiles:
        print("Loading %s" % df)

        # Load an EPI image
        img = nipy.load_image(df)

        xyz = _check_xyz(xyz, img)
        data_list.append(np.array(img.get_data().flatten(), dtype=np.float64))

    data = np.vstack(data_list)

    return data, xyz


def _check_xyz(xyz, img):
    '''Check voxel xyz consistency.'''

    xyz_current = _get_xyz(img.coordmap.affine, img.get_data().shape)

    if xyz.size == 0:
        xyz = xyz_current
    elif (xyz != xyz_current).any():
        raise ValueError("Voxel XYZ coordinates are inconsistent across volumes")

    return xyz


def _get_xyz(affine, volume_shape):
    '''Return voxel XYZ coordinates based on an affine matrix.

    Parameters
    ----------
    affine : array
        Affine matrix.
    volume_shape : list
        Shape of the volume (i, j, k lnegth).

    Returns
    -------
    array, shape = (3, N)
        x-, y-, and z-coordinates (N: the number of voxels).
    '''

    i_len, j_len, k_len = volume_shape
    ijk = np.array(list(itr.product(range(i_len),
                                    range(j_len),
                                    range(k_len),
                                    [1]))).T

    return np.dot(affine, ijk)[:-1]
