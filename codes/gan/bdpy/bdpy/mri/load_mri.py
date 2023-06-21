'''load_mri'''


import numpy as np
import nipy


def load_mri(fpath):
    '''Load a MRI image.

    - Returns data as 2D array (sample x voxel)
    - Returns voxle xyz coordinates (3 x voxel)
    - Returns voxel ijk indexes (3 x voxel)
    - Data, xyz, and ijk are flattened by Fortran-like index order
    '''
    img = nipy.load_image(fpath)

    data = img.get_data()
    if data.ndim == 4:
        data = data.reshape(-1, data.shape[-1], order='F').T
        i_len, j_len, k_len, t = img.shape
        affine = np.delete(np.delete(img.coordmap.affine, 3, axis=0), 3, axis=1)
    elif data.ndim == 3:
        data = data.flatten(order='F')
        i_len, j_len, k_len = img.shape
        affine = img.coordmap.affine
    else:
        raise ValueError('Invalid shape.')

    ijk = np.array(np.unravel_index(np.arange(i_len * j_len * k_len),
                                    (i_len, j_len, k_len), order='F'))
    ijk_b = np.vstack([ijk, np.ones((1, i_len * j_len * k_len))])
    xyz_b = np.dot(affine, ijk_b)
    xyz = xyz_b[:-1]

    return data, xyz, ijk
