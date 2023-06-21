'''Utilities for ROIs'''


import os
import glob
import re
import hashlib

import numpy as np
import nibabel.freesurfer

from bdpy.mri import load_mri


def add_roimask(
        bdata, roi_mask, roi_prefix='',
        brain_data='VoxelData', xyz=['voxel_x', 'voxel_y', 'voxel_z'],
        return_roi_flag=False,
        verbose=True,
        round=None
):
    '''Add an ROI mask to `bdata`.

    Parameters
    ----------
    bdata : BData
    roi_mask : str or list
        ROI mask file(s).
    round : int
        Number of decimal places to round the voxel coordinate.

    Returns
    -------
    bdata : BData
    '''

    if isinstance(roi_mask, str):
        roi_mask = [roi_mask]

    # Get voxel xyz coordinates in `bdata`
    voxel_xyz = np.vstack([bdata.get_metadata(xyz[0], where=brain_data),
                           bdata.get_metadata(xyz[1], where=brain_data),
                           bdata.get_metadata(xyz[2], where=brain_data)])

    if round is not None:
        voxel_xyz = np.round(voxel_xyz, round)

    # Load the ROI mask files
    mask_xyz_all = []
    mask_v_all = []

    voxel_consistency = True

    for m in roi_mask:
        mask_v, mask_xyz, mask_ijk = load_mri(m)
        if round is not None:
            mask_xyz = np.round(mask_xyz, round)
        mask_v_all.append(mask_v)
        mask_xyz_all.append(mask_xyz[:, (mask_v == 1).flatten()])

        if voxel_xyz.shape != mask_xyz.shape or not (voxel_xyz == mask_xyz).all():
            voxel_consistency = False

    # Get ROI flags
    if voxel_consistency:
        roi_flag = np.vstack(mask_v_all)
    else:
        roi_flag = get_roiflag(mask_xyz_all, voxel_xyz, verbose=verbose)

    # Add the ROI flag as metadata in `bdata`
    md_keys = []
    md_descs = []

    for i, roi in enumerate(roi_mask):
        roi_name = roi_prefix + '_' + os.path.basename(roi).replace('.nii.gz', '').replace('.nii', '')

        with open(roi, 'rb') as f:
            roi_md5 = hashlib.md5(f.read()).hexdigest()

        roi_desc = '1 = ROI %s (source file: %s; md5: %s)' % (roi_name, roi, roi_md5)
        if verbose:
            print('Adding %s' % roi_name)
            print('  %s' % roi_desc)
        md_keys.append(roi_name)
        md_descs.append(roi_desc)

    bdata.metadata.key.extend(md_keys)
    bdata.metadata.description.extend(md_descs)

    brain_data_index = bdata.get_metadata(brain_data)
    new_md_v = np.zeros([roi_flag.shape[0], bdata.metadata.value.shape[1]])
    new_md_v[:, :] = np.nan
    new_md_v[:, brain_data_index == 1] = roi_flag

    bdata.metadata.value = np.vstack([bdata.metadata.value, new_md_v])

    if return_roi_flag:
        return bdata, roi_flag
    else:
        return bdata


def get_roiflag(roi_xyz_list, epi_xyz_array, verbose=True):
    '''Get ROI flags.

    Parameters
    ----------
    roi_xyz_list : list, len = n_rois
        List of arrays that contain XYZ coordinates of ROIs. Each element is an
        array of shape = (3, n_voxels_in_roi).
    epi_xyz_array : array, shape = (3, n_voxels)
        Voxel XYZ coordinates
    verbose : boolean
        If True, 'get_roiflag' outputs verbose message

    Returns
    -------
    roi_flag : array, shape = (n_rois, n_voxels)
        ROI flag array
    '''

    epi_voxel_size = epi_xyz_array.shape[1]

    if verbose:
        print("EPI num voxels: %d" % epi_voxel_size)

    roi_flag_array = np.zeros((len(roi_xyz_list), epi_voxel_size))

    epi_xyz_dist = np.sum(epi_xyz_array ** 2, axis=0)

    for i, roi_xyz in enumerate(roi_xyz_list):
        if verbose:
            print("ROI %d num voxels: %d" % (i + 1, len(roi_xyz[0])))

        roi_xyz_dist = np.sum(roi_xyz ** 2, axis=0)

        roi_flag_temp = np.zeros(epi_xyz_dist.shape)

        for j, mv_dist in enumerate(roi_xyz_dist):
            candidate_index = epi_xyz_dist == mv_dist
            roi_flag_in_candidate = [np.array_equal(v_xyz, roi_xyz[:, j].flatten())
                                     for v_xyz in epi_xyz_array[:, candidate_index].T]
            roi_flag_temp[candidate_index] = roi_flag_temp[candidate_index] + roi_flag_in_candidate

        roi_flag_temp[roi_flag_temp > 1] = 1
        roi_flag_array[i, :] = roi_flag_temp

    return roi_flag_array


def add_roilabel(bdata, label, vertex_data=['VertexData'], prefix='', verbose=False):
    '''Add ROI label(s) to `bdata`.

    Parameters
    ----------
    bdata : BData
    roi_mask : str or list
        ROI label file(s).

    Returns
    -------
    bdata : BData
    '''

    def add_roilabel_file(bdata, label, vertex_data=['VertexData'], prefix='', verbose=False):
        # Read the label file
        roi_vertex = nibabel.freesurfer.read_label(label)

        # Make meta-data vector for ROI flag
        vindex = bdata.get_metadata('vertex_index', where=vertex_data)

        roi_flag = np.zeros(vindex.shape)

        for v in roi_vertex:
            roi_flag[vindex == v] = 1

        roi_name = prefix + '_' + os.path.basename(label).replace('.label', '')

        with open(label, 'rb') as f:
            roi_md5 = hashlib.md5(f.read()).hexdigest()

        roi_desc = '1 = ROI %s (source file: %s; md5: %s)' % (roi_name, label, roi_md5)
        if verbose:
            print('Adding %s (%d vertices)' % (roi_name, np.sum(roi_flag)))
            print('  %s' % roi_desc)
        bdata.add_metadata(roi_name, roi_flag, description=roi_desc, where=vertex_data)

        return bdata

    if isinstance(label, str):
        label = [label]

    for lb in label:
        if os.path.splitext(lb)[1] == '.label':
            # FreeSurfer label file
            bdata = add_roilabel_file(bdata, lb, vertex_data=vertex_data, prefix=prefix, verbose=verbose)
        elif os.path.splitext(lb)[1] == '.annot':
            # FreeSurfer annotation file
            annot = nibabel.freesurfer.read_annot(lb)
            labels = annot[0]  # Annotation ID at each vertex (shape = (n_vertices,))
            ctab = annot[1]    # Label color table (RGBT + label ID)
            names = annot[2]   # Label name list

            with open(lb, 'rb') as f:
                roi_md5 = hashlib.md5(f.read()).hexdigest()

            for i, name in enumerate(names):
                label_id = i  # Label ID is zero-based
                roi_flag = (labels == label_id).astype(int)

                if sum(roi_flag) == 0:
                    if verbose:
                        print('Label %s not found in the surface.' % name)
                    continue

                # FIXME: better way to decide left/right?
                if 'Left' in vertex_data:
                    hemi = 'lh'
                elif 'Right' in vertex_data:
                    hemi = 'rh'
                else:
                    raise ValueError('Invalid vertex_data: %s' % vertex_data)

                roi_name = prefix + '_' + hemi + '.' + name
                roi_desc = '1 = ROI %s (source file: %s; md5: %s)' % (roi_name, lb, roi_md5)

                if verbose:
                    print('Adding %s (%d vertices)' % (roi_name, np.sum(roi_flag)))
                    print('  %s' % roi_desc)

                bdata.add_metadata(roi_name, roi_flag, description=roi_desc, where=vertex_data)
        else:
            raise TypeError('Unknown file type: %s' % os.path.splitext(lb)[0])

    return bdata


def add_rois(bdata, roi_files, data_type='volume', prefix_map={}, remove_voxel=True):
    '''Add ROIs in bdata from files.'''

    roi_prefix_from_annot = {'lh.aparc.a2009s.annot': 'freesurfer_destrieux',
                             'rh.aparc.a2009s.annot': 'freesurfer_destrieux',
                             'lh.aparc.annot': 'freesurfer_dk',
                             'rh.aparc.annot': 'freesurfer_dk'}

    if data_type == 'volume':
        # List all ROI mask files
        roi_files_all = []
        for roi_files_pt in roi_files:
            roi_files_all.extend(glob.glob(roi_files_pt))

        # Group ROI mask files by directories (prefix)
        roi_group = {}
        for roi_file in roi_files_all:
            roi_prefix = os.path.basename(os.path.dirname(roi_file))
            if roi_prefix in roi_group.keys():
                roi_group[roi_prefix].append(roi_file)
            else:
                roi_group.update({roi_prefix: [roi_file]})

        # Add ROIs
        roi_flag_all = []
        for roi_prefix, roi_files_group in roi_group.items():
            print('Adding ROIs %s' % roi_prefix)
            bdata, roi_flag= add_roimask(bdata, roi_files_group, roi_prefix=roi_prefix, return_roi_flag=True, verbose=True)
            roi_flag_all.append(roi_flag)
        print('')

        # Remove voxels out of ROIs
        if remove_voxel:
            roi_flag_all = np.vstack(roi_flag_all)
            remove_voxel_ind = np.sum(roi_flag_all, axis=0) == 0
            _, voxel_ind = bdata.select('VoxelData = 1', return_index=True)
            remove_column_ind = np.where(voxel_ind)[0][remove_voxel_ind]

            bdata.dataset = np.delete(bdata.dataset, remove_column_ind, 1)
            bdata.metadata.value = np.delete(bdata.metadata.value, remove_column_ind, 1)
            # FIXME: needs cleaning

    elif data_type == 'surface':
        # List all ROI labels files
        roi_files_lh_all = []
        roi_files_rh_all = []
        for roi_files_pt in roi_files:
            roi_files_lh_all.extend(glob.glob(roi_files_pt[0]))
            roi_files_rh_all.extend(glob.glob(roi_files_pt[1]))

        for roi_file_lh, roi_file_rh in zip(roi_files_lh_all, roi_files_rh_all):
            _, ext = os.path.splitext(roi_file_lh)
            if ext == '.annot':
                roi_prefix_lh = re.sub('^lh\.', '', os.path.splitext(os.path.basename(roi_file_lh))[0])
            else:
                roi_prefix_lh = os.path.basename(os.path.dirname(roi_file_lh))
            _, ext = os.path.splitext(roi_file_rh)
            if ext == '.annot':
                roi_prefix_rh = re.sub('^rh\.', '', os.path.splitext(os.path.basename(roi_file_rh))[0])
            else:
                roi_prefix_rh = os.path.basename(os.path.dirname(roi_file_rh))

            if roi_prefix_lh != roi_prefix_rh:
                raise ValueError('The left and right hemi labels should share the same prefix.')

            if roi_prefix_lh in prefix_map.keys():
                roi_prefix_lh = prefix_map[roi_prefix_lh]
                roi_prefix_rh = prefix_map[roi_prefix_rh]

            print('Adding ROIs %s' % roi_prefix_lh)
            bdata = add_roilabel(bdata, roi_file_lh, vertex_data='VertexLeft', prefix=roi_prefix_lh, verbose=True)
            bdata = add_roilabel(bdata, roi_file_rh, vertex_data='VertexRight', prefix=roi_prefix_rh, verbose=True)

    else:
        raise ValueError('Invalid data type: %s' % data_type)

    return bdata


def merge_rois(bdata, roi_name, merge_expr):
    '''Merage ROIs.'''

    print('Adding merged ROI %s' % roi_name)

    # Get tokens
    tokens_raw = merge_expr.split(' ')
    tokens_raw = [t for t in tokens_raw if t != '']

    tokens = []
    for tkn in tokens_raw:
        if tkn == '+' or tkn == '-':
            tokens.append(tkn)
        else:
            # FIXME: dirty solution
            tkn = tkn.replace('"', '')
            tkn = tkn.replace("'", '')
            tkn_e = re.escape(tkn)
            tkn_e = tkn_e.replace('\*', '.*')
            tkn_e = tkn_e + '$'  # To mimic `fullmatch` that is available in Python >= 3.4

            mks = [k for k in bdata.metadata.key if re.match(tkn_e, k)]
            if len(mks) == 0:
                raise RuntimeError('ROI %s not found' % merge_expr)
            s = ' + '.join(mks)
            tokens.extend(s.split(' '))

    print('Merging ROIs: ' + ' '.join(tokens))

    # Parse tokens
    op_stack = []
    rpn_que = []
    for tkn in tokens:
        if tkn == '+' or tkn == '-':
            while op_stack:
                rpn_que.append(op_stack.pop())
            op_stack.append(tkn)
        else:
            rpn_que.append(tkn)
    while op_stack:
        rpn_que.append(op_stack.pop())

    # Get merged ROI meta-data
    out_stack = []
    for tkn in rpn_que:
        if tkn == '+':
            b = out_stack.pop()
            a = out_stack.pop()
            out = a + b
        elif tkn == '-':
            b = out_stack.pop()
            a = out_stack.pop()
            out = a - b
        else:
            out = bdata.get_metadata(tkn)
        out[np.isnan(out)] = 0
        out[out > 1] = 1
        out[out < 0] = 0
        out_stack.append(out)

    if len(out_stack) != 1:
        raise RuntimeError('Something goes wrong in merge_rois.')

    merged_roi_mv = out_stack[0]
    description = 'Merged ROI: %s' % ' '.join(tokens)
    bdata.add_metadata(roi_name, merged_roi_mv, description)

    num_voxels = np.nansum(merged_roi_mv).astype(int)
    print('Num voxels or vertexes: %d' % num_voxels)

    return bdata


def add_hcp_rois(bdata, overwrite=False):
    '''Add HCP ROIs in `bdata`.

    Note
    ----
    This function assumes that the HCP ROIs (splitted by left and right) are
    named as "hcp180_r_lh.L_{}__*_ROI" and "hcp180_r_rh.R_{}__*_ROI".
    '''

    hcp180_rois = [
        '1', '10d', '10pp', '10r', '10v', '11l', '13l', '2', '23c', '23d',
        '24dd', '24dv', '25', '31a', '31pd', '31pv', '33pr', '3a', '3b', '4',
        '43', '44', '45', '46', '47l', '47m', '47s', '52', '55b', '5L', '5m',
        '5mv', '6a', '6d', '6ma', '6mp', '6r', '6v', '7AL', '7Am', '7PC',
        '7PL', '7Pm', '7m', '8Ad', '8Av', '8BL', '8BM', '8C', '9-46d', '9a',
        '9m', '9p', 'A1', 'A4', 'A5', 'AAIC', 'AIP', 'AVI', 'DVT', 'EC',
        'FEF', 'FFC', 'FOP1', 'FOP2', 'FOP3', 'FOP4', 'FOP5', 'FST', 'H',
        'IFJa', 'IFJp', 'IFSa', 'IFSp', 'IP0', 'IP1', 'IP2', 'IPS1', 'Ig',
        'LBelt', 'LIPd', 'LIPv', 'LO1', 'LO2', 'LO3', 'MBelt', 'MI', 'MIP',
        'MST', 'MT', 'OFC', 'OP1', 'OP2-3', 'OP4', 'PBelt', 'PCV', 'PEF',
        'PF', 'PFcm', 'PFm', 'PFop', 'PFt', 'PGi', 'PGp', 'PGs', 'PH', 'PHA1',
        'PHA2', 'PHA3', 'PHT', 'PI', 'PIT', 'POS1', 'POS2', 'PSL', 'PeEc',
        'Pir', 'PoI1', 'PoI2', 'PreS', 'ProS', 'RI', 'RSC', 'SCEF', 'SFL',
        'STGa', 'STSda', 'STSdp', 'STSva', 'STSvp', 'STV', 'TA2', 'TE1a',
        'TE1m', 'TE1p', 'TE2a', 'TE2p', 'TF', 'TGd', 'TGv', 'TPOJ1', 'TPOJ2',
        'TPOJ3', 'V1', 'V2', 'V3', 'V3A', 'V3B', 'V3CD', 'V4', 'V4t', 'V6',
        'V6A', 'V7', 'V8', 'VIP', 'VMV1', 'VMV2', 'VMV3', 'VVC', 'a10p',
        'a24', 'a24pr', 'a32pr', 'a47r', 'a9-46v', 'd23ab', 'd32', 'i6-8',
        'p10p', 'p24', 'p24pr', 'p32', 'p32pr', 'p47r', 'p9-46v', 'pOFC',
        's32', 's6-8', 'v23ab'
    ]

    hcp_22_regions = {
        'PVC': ['V1'],
        'EVC': ['V2', 'V3', 'V4'],
        'DSVC': ['V3A', 'V7', 'V3B', 'V6', 'V6A', 'IPS1'],
        'VSVC': ['V8', 'VVC', 'VMV1', 'VMV2', 'VMV3', 'PIT', 'FFC'],
        'MTcVA': ['V3CD', 'LO1', 'LO2', 'LO3', 'MT', 'MST', 'V4t', 'FST', 'PH'],
        'SMC': ['4', '3a', '3b', '1', '2'],
        'PCL_MCC': ['5L', '5m', '5mv', '24dd', '24dv', '6mp', '6ma', 'SCEF'],
        'PMC': ['6a', '6d', 'FEF', 'PEF', '55b', '6v', '6r'],
        'POC': ['43', 'FOP1', 'OP4', 'OP2-3', 'OP1', 'PFcm'],
        'EAC': ['A1', 'MBelt', 'LBelt', 'PBelt', 'RI'],
        'AAC': ['A4', 'A5', 'STSdp', 'STSda', 'STSvp', 'STSva', 'TA2', 'STGa'],
        'IFOC': ['52', 'PI', 'Ig', 'PoI1', 'PoI2', 'FOP2', 'Pir', 'AAIC', 'MI', 'FOP3', 'FOP4', 'FOP5', 'AVI'],
        'MTC': ['H', 'PreS', 'EC', 'PeEc', 'PHA1', 'PHA2', 'PHA3'],
        'LTC': ['TGd', 'TGv', 'TF', 'TE2a', 'TE2p', 'TE1a', 'TE1m', 'TE1p', 'PHT'],
        'TPOJ': ['TPOJ2', 'TPOJ3', 'TPOJ1', 'STV', 'PSL'],
        'SPC': ['MIP', 'LIPv', 'VIP', 'LIPd', 'AIP', '7PC', '7Am', '7AL', '7Pm', '7PL'],
        'IPC': ['PGp', 'IP0', 'IP1', 'IP2', 'PF', 'PFt', 'PFop', 'PFm', 'PGi', 'PGs'],
        'PCC': ['DVT', 'ProS', 'POS2', 'POS1', 'RSC', '7m', 'PCV', 'v23ab', 'd23ab', '31pv', '31pd', '31a', '23c', '23d'],
        'ACC_mPFC': ['33pr', 'a24pr', 'p24pr', 'p24', 'a24', 'p32pr', 'a32pr', 'd32', 'p32', 's32', '8BM', '9m', '10r', '10v', '25'],
        'OPFC': ['OFC', 'pOFC', '13l', '11l', '47s', '47m', 'a47r', '10pp', 'a10p', 'p10p', '10d'],
        'IFC': ['44', '45', '47l', 'IFJp', 'IFJa', 'IFSp', 'IFSa', 'p47r'],
        'DLPFC': ['SFL', 's6-8', 'i6-8', '8BL', '8Ad', '8Av', '8C', '9p', '9a', '9-46d', 'a9-46v', 'p9-46v', '46'],
    }

    # Merge left and right ROIs
    for roi in hcp180_rois:
        name = 'hcp180_{}'.format(roi)
        if not overwrite and name in bdata.metadata.key:
            continue
        select = '"hcp180_r_lh.L_{}_ROI" + "hcp180_r_rh.R_{}_ROI"'.format(roi, roi)
        print('{}: {}'.format(name, select))
        bdata = merge_rois(bdata, name, select)

    # Merge HCP ROI groups
    # See "Supplementary Neuroanatomical Results" of Glasser et al. (2016)
    # https://www.nature.com/articles/nature18933
    for name, rois in hcp_22_regions.items():
        name = 'hcp180_reg_{}'.format(name)
        if not overwrite and name in bdata.metadata.key:
            continue
        select = ' + '.join(['"hcp180_{}"'.format(a) for a in rois])
        bdata = merge_rois(bdata, name, select)

    return bdata


def add_hcp_visual_cortex(bdata, overwrite=False):
    '''Add HCP-based visual cortex in `bdata`.'''

    # Whole VC
    vc_rois = [
        'V1', 'V2', 'V3', 'V4',
        'V3A', 'V3B', 'V6', 'V6A', 'V7', 'IPS1',
        'V8', 'VVC', 'VMV1', 'VMV2', 'VMV3', 'PIT', 'FFC',
        'V3CD', 'LO1', 'LO2', 'LO3', 'MT', 'MST', 'V4t', 'FST', 'PH',
        'MIP', 'AIP', 'VIP', 'LIPv', 'LIPd', '7PC', '7Am', '7AL', '7Pm', '7PL',
        'PHA1', 'PHA2', 'PHA3',
        'IP0', 'IP1', 'IP2', 'PGp',
        'TPOJ2', 'TPOJ3',
        'DVT', 'ProS', 'POS1', 'POS2'
    ]
    select = ' + '.join(['"hcp180_{}"'.format(a) for a in vc_rois])

    if overwrite or 'hcp180_hcpVC' not in bdata.metadata.key:
        bdata = merge_rois(bdata, 'hcp180_hcpVC', select)

    # Early, Ventral, Dorsal, and MT VC
    if overwrite or 'hcp180_EarlyVC' not in bdata.metadata.key:
        bdata = merge_rois(bdata, 'hcp180_EarlyVC',   'hcp180_V1 + hcp180_V2 + hcp180_V3')
    if overwrite or 'hcp180_MTVC' not in bdata.metadata.key:
        bdata = merge_rois(bdata, 'hcp180_MTVC',      'hcp180_reg_MTcVA + hcp180_TPOJ2 + hcp180_TPOJ3 - hcp180_EarlyVC')
    if overwrite or 'hcp180_VentralVC' not in bdata.metadata.key:
        bdata = merge_rois(bdata, 'hcp180_VentralVC', 'hcp180_V4 + hcp180_reg_VSVC + hcp180_PHA1 + hcp180_PHA2 + hcp180_PHA3 - hcp180_EarlyVC - hcp180_MTVC')
    if overwrite or 'hcp180_DorsalVC' not in bdata.metadata.key:
        bdata = merge_rois(bdata, 'hcp180_DorsalVC',  'hcp180_reg_DSVC + hcp180_reg_SPC + hcp180_PGp + hcp180_IP0 + hcp180_IP1 + hcp180_IP2 - hcp180_EarlyVC - hcp180_MTVC - hcp180_VentralVC')

    return bdata
