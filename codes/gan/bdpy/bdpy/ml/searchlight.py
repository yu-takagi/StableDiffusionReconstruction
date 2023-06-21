'''Utilities for searchlight analysis.'''


__all__ = ['get_neighbors']


import numpy as np


def get_neighbors(xyz, space_xyz, shape='sphere', size=9):
    '''
    Returns neighboring voxels (cluster).

    Parameters
    ----------
    xyz : array_like, shape=(3,) or len=3
        Voxel XYZ coordinate in the center of the cluster.
    space_xyz : array_like, shape=(3, N) or (N, 3)
        XYZ coordinate of all voxels.
    shape : {'sphere'}, optional
        Shape of the cluster.
    size : float, optional
        Size of the cluster.

    Returns
    -------
    cluster_index : array_like, dtype=bool, shape=(N,)
        Boolean index of voxels in the cluster.
    '''

    # Input check
    if isinstance(xyz, list):
        xyz = np.array(xyz)

    if xyz.ndim != 1:
        raise TypeError('xyz should be 1-D array')

    if space_xyz.ndim != 2:
        raise TypeError('space_xyz should be 2-D array')

    # Fix input shape
    if space_xyz.shape[0] == 3:
        space_xyz = space_xyz.T

    if shape == 'sphere':
        dist = np.sum((space_xyz - xyz) ** 2, axis=1)
        cluster_index = dist <= size ** 2
    else:
        raise ValueError('Unknown shape: %s' % shape)

    return cluster_index
