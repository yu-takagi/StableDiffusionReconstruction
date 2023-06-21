'''bdpy.mri.image'''


from itertools import product

import numpy as np
import nibabel

from bdpy.mri import load_mri


def export_brain_image(brain_data, template, xyz=None, out_file=None):
    '''Export a brain data array as a brain image.

    Parameters
    ----------
    brain_data : array
        Brain data array, shape = (n_sample, n_voxels)
    template : str
        Path to a template brain image file
    xyz : array, optional
        Voxel xyz coordinates of the brain data array

    Returns
    -------
    nibabel.Nifti1Image
    '''

    if brain_data.ndim == 1:
        brain_data = brain_data[np.newaxis, :]

    if brain_data.shape[0] > 1:
        raise RuntimeError('4-D image is not supported yet.')
        
    template_image = nibabel.load(template)
    _, brain_xyz, _ = load_mri(template)

    out_table = {}
    if xyz is None:
        xyz = brain_xyz
    
    for i in range(brain_data.shape[1]):
        x, y, z = xyz[0, i], xyz[1, i], xyz[2, i]
        out_table.update({(x, y, z): brain_data[0, i]})

    out_image_array = np.zeros(template_image.shape[:3])
    for i, j, k in product(range(template_image.shape[0]), range(template_image.shape[1]), range(template_image.shape[2])):
        x, y, z = template_image.affine[:3, :3].dot([i, j, k]) + template_image.affine[:3, 3]
        if (x, y, z) in out_table:
            out_image_array[i, j, k] = out_table[(x, y, z)]

    out_image = nibabel.Nifti1Image(out_image_array, template_image.affine)
            
    return out_image
