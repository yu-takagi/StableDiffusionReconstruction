'''Utilities for fmriprep.'''


import csv
import glob
import itertools
import os
import re
import json
import warnings
from collections import OrderedDict

import numpy as np
import nipy
import nibabel
import pandas as pd

import bdpy


class FmriprepData(object):
    '''FMRIPREP data class.'''

    def __init__(self, datapath=None, fmriprep_version='1.2', fmriprep_dir='derivatives/fmriprep'):
        self.__datapath = datapath
        self.__data = OrderedDict()
        self.__fmriprep_version = fmriprep_version
        self.__fmriprep_dir = fmriprep_dir

        if self.__datapath is not None:
            self.__parse_data()
            self.__get_task_event_files()

        return None

    @property
    def data(self):
        return self.__data

    # Private methods --------------------------------------------------------

    def __parse_data(self):
        # FMRIPREP results directory
        prepdir = os.path.join(self.__datapath, self.__fmriprep_dir, 'fmriprep')

        # Get subjects
        subjects = self.__get_subjects(prepdir)

        # Get sessions
        for sbj in subjects:
            self.__data.update({sbj: OrderedDict()})
            sessions = self.__get_sessions(prepdir, sbj)

            # Get runs in the sesssion
            for ses in sessions:
                if self.__fmriprep_version == '1.2':
                    if ses == 'ses-anat':
                        continue
                runs = self.__parse_session(prepdir, sbj, ses)
                self.__data[sbj].update({ses: runs})

        return None

    def __get_subjects(self, dpath):
        subjects = []
        for d in os.listdir(dpath):
            if not os.path.isdir(os.path.join(dpath, d)):
                continue

            m = re.match('^sub-.*', d)
            if not m:
                continue

            subjects.append(d)
        return subjects

    def __get_sessions(self, dpath, subject):
        subpath = os.path.join(dpath, subject)
        sessions = []
        for d in os.listdir(subpath):
            if not os.path.isdir(os.path.join(subpath, d)):
                continue

            m = re.match('^ses-.*', d)
            if not m:
                continue

            sessions.append(d)
        return sorted(sessions)

    def __parse_session(self, dpath, subject, session):
        sespath = os.path.join(dpath, subject, session)
        funcpath = os.path.join(sespath, 'func')

        # File name patterns
        # FIXME
        if self.__fmriprep_version == '1.2':
            file_pattern = {'volume_native'   : '.*_space-T1w_desc-preproc_bold\.nii\.gz$',
                            'volume_standard' : '.*_space-MNI152NLin2009cAsym_desc-preproc_bold\.nii\.gz$',
                            'surf_native_left'  : '.*_space-fsnative_hemi-L\.func\.gii$',
                            'surf_native_right' : '.*_space-fsnative_hemi-R\.func\.gii$',
                            'surf_standard_left'  : '.*_space-fsaverage_hemi-L\.func\.gii$',
                            'surf_standard_right' : '.*_space-fsaverage_hemi-R\.func\.gii$',
                            'surf_standard_41k_left'  : '.*_space-fsaverage6_hemi-L\.func\.gii$',
                            'surf_standard_41k_right' : '.*_space-fsaverage6_hemi-R\.func\.gii$',
                            'surf_standard_10k_left'  : '.*_space-fsaverage5_hemi-L\.func\.gii$',
                            'surf_standard_10k_right' : '.*_space-fsaverage5_hemi-R\.func\.gii$',
                            'confounds'       : '.*_desc-confounds_regressors\.tsv$'}
        elif self.__fmriprep_version in ['1.0', '1.1']:
            file_pattern = {'volume_native'   : '.*_bold_space-T1w_preproc\.nii\.gz$',
                            'volume_standard' : '.*_bold_space-MNI152NLin2009cAsym_preproc\.nii\.gz$',
                            'surf_native_left'  : '.*_space-fsnative\.L\.func\.gii$',
                            'surf_native_right' : '.*_space-fsnative\.R\.func\.gii$',
                            'surf_standard_left'  : '.*_space-fsaverage\.L\.func\.gii$',
                            'surf_standard_right' : '.*_space-fsaverage\.R\.func\.gii$',
                            'confounds'       : '.*_bold_confounds\.tsv'}
        else:
            raise ValueError('Unsuppored fmriprep version %s' % self.__fmriprep_version)

        run_dict = {}
        for f in os.listdir(funcpath):
            if os.path.isdir(os.path.join(funcpath, f)):
                continue

            m = re.match('.*_run-([0-9]+)_.*', f)
            if not m:
                continue

            run_num = m.group(1)

            for key in file_pattern:
                m = re.search(file_pattern[key], f)
                if m:
                    if run_num in run_dict:
                        run_dict[run_num].update({key: f})
                    else:
                        run_dict.update({run_num: {key: f}})
                    break

        run_index = sorted([int(k) for k in run_dict.keys()])

        runs = []
        for r in run_index:
            rf = run_dict['%02d' % r]
            basedir = os.path.join(self.__fmriprep_dir, 'fmriprep', subject, session, 'func')
            run_files = {'volume_native' : os.path.join(basedir, rf['volume_native']) if 'volume_native' in rf else None,
                         'volume_standard' : os.path.join(basedir, rf['volume_standard']) if 'volume_standard' in rf else None,
                         'surface_native' : (os.path.join(basedir, rf['surf_native_left']) if 'surf_native_left' in rf else None,
                                             os.path.join(basedir, rf['surf_native_right']) if 'surf_native_right' in rf else None),
                         'surface_standard' : (os.path.join(basedir, rf['surf_standard_left']) if 'surf_standard_left' in rf else None,
                                               os.path.join(basedir, rf['surf_standard_right']) if 'surf_standard_right' in rf else None),
                         'surface_standard_41k' : (os.path.join(basedir, rf['surf_standard_41k_left']) if 'surf_standard_41k_left' in rf else None,
                                                   os.path.join(basedir, rf['surf_standard_41k_right']) if 'surf_standard_41k_right' in rf else None),
                         'surface_standard_10k' : (os.path.join(basedir, rf['surf_standard_10k_left']) if 'surf_standard_10k_left' in rf else None,
                                                   os.path.join(basedir, rf['surf_standard_10k_right']) if 'surf_standard_10k_right' in rf else None),
                         'confounds' : os.path.join(basedir, rf['confounds']) if 'confounds' in rf else None}

            runs.append(run_files)

        return runs

    def __get_task_event_files(self):
        for sbj, sbjdata in self.__data.items():
            for ses, sesdata in sbjdata.items():
                raw_func_dir = os.path.join(self.__datapath, sbj, ses, 'func')
                for run in sesdata:
                    # Get run label
                    if self.__fmriprep_version == '1.2':
                        m = re.search('.*_(run-.*)_desc-confounds_.*', run['confounds'])
                    elif self.__fmriprep_version in ['1.0', '1.1']:
                        m = re.search('.*_(run-.*)_bold_.*', run['confounds'])
                    else:
                        raise ValueError('Unsuppored fmriprep version %s' % self.__fmriprep_version)
                    if m:
                        run_label = m.group(1)
                    else:
                        raise RuntimeError('Run not found!')
                    # Get task event file
                    event_file_name_glob = '%s_%s_task-*_%s_events.tsv' % (sbj, ses, run_label)
                    event_file_list = glob.glob(os.path.join(raw_func_dir, event_file_name_glob))
                    if len(event_file_list) != 1:
                        raise RuntimeError('Something is wrong on task event files.')
                    event_file = event_file_list[0].replace(os.path.normpath(self.__datapath) + '/', '')
                    # Add the task event file in data
                    run.update({'task_event_file': event_file})

                    # Get bold json file
                    bold_json_file_name_glob = '%s_%s_task-*_%s_bold.json' % (sbj, ses, run_label)
                    bold_json_file_list = glob.glob(os.path.join(raw_func_dir, bold_json_file_name_glob))
                    if len(bold_json_file_list) != 1:
                        raise RuntimeError('Something is wrong on bold parameter json files.')
                    bold_json_file = bold_json_file_list[0].replace(os.path.normpath(self.__datapath) + '/', '')
                    run.update({'bold_json': bold_json_file})
        return None


def create_bdata_fmriprep(dpath, data_mode='volume_native',
                          fmriprep_version='1.2',
                          fmriprep_dir='derivatives/fmriprep',
                          label_mapper=None, exclude={},
                          split_task_label=False,
                          with_confounds=False,
                          return_data_labels=False,
                          return_list=False):
    '''Create BData from FMRIPREP outputs.

    Parameters
    ----------
    dpath : str
        Path to a BIDS data directory.
    data_mode : {'volume_standard', 'volume_native', 'surface_standard',
                 'surface_standard_41k', 'surface_standard_10k',
                 'surface_native'}
        Data to be loaded.
    fmriprep_version : {'1.2, '1.1', '1.0'}
        The version of fmriprep (default: '1.2')
    label_mapper : dict
        A dictionary of tables that define mapping between non-numerical value
        (e.g., string) in task event files and float value in BData.dataset.
    exclude : dict
        A dictionary that defines subjects, sessions, or runs excluded from
        the resulting BData.

    Returns
    -------
    BData or list of BData
        One subject, one BData.
    '''

    print('BIDS data path: %s' % dpath)

    # Label mapper
    if label_mapper is None:
        label_mapper_dict = {}
    else:
        if not isinstance(label_mapper, dict):
            raise TypeError('Unsupported label mapper (type: %s)' % type(label_mapper))

        label_mapper_dict = {}
        for lbmp in label_mapper:
            if not isinstance(label_mapper[lbmp], str):
                raise TypeError('Unsupported label mapper (type: %s)' % type(label_mapper[lbmp]))

            lbmp_file = label_mapper[lbmp]

            ext = os.path.splitext(lbmp_file)[1]
            lbmp_dict = {}

            if ext == '.csv':
                with open(lbmp_file, 'r') as f:
                    reader = csv.reader(f, delimiter=',')
                    for row in reader:
                        lbmp_dict.update({row[0]: int(row[1])})
            elif ext == '.tsv':
                with open(lbmp_file, 'r') as f:
                    reader = csv.reader(f, delimiter='\t')
                    for row in reader:
                        lbmp_dict.update({row[0]: int(row[1])})
            else:
                raise ValueError('Unsuppored label mapper file: %s' % lbmp_file)

            #lbmp_dict.update({'n/a': np.nan})
            label_mapper_dict.update({lbmp: lbmp_dict})

    # Load fmriprep outputs
    fmriprep = FmriprepData(dpath, fmriprep_version=fmriprep_version, fmriprep_dir=fmriprep_dir)

    # Exclude subject, session, or run
    # TODO: add subject/session, subject/session/run specification
    for sub in fmriprep.data:
        # Exclude subject
        if 'subject' in exclude and sub in exclude['subject']:
            del(fmriprep.data[sub])
            continue

        for i, ses in enumerate(fmriprep.data[sub]):
            # Exclude session
            if 'session' in exclude and i + 1 in exclude['session']:
                del(fmriprep.data[sub][ses])
                continue

            # Exclude run
            run_survive = [run for j, run in enumerate(fmriprep.data[sub][ses])
                           if not ('run' in exclude and j + 1 in exclude['run'])]
            fmriprep.data[sub][ses] = run_survive

            # Exclude session/run
            if 'session/run' in exclude:
                ex_runs = exclude['session/run'][i]
                if ex_runs is not None and len(ex_runs) != 0:
                    run_survive = [run for j, run in enumerate(fmriprep.data[sub][ses])
                                   if not j + 1 in ex_runs]
                    fmriprep.data[sub][ses] = run_survive

    # Split data
    # TODO: needs refactoring, obviously
    if split_task_label:
        fmriprep_data = OrderedDict()
        for sub in fmriprep.data:

            if sub in fmriprep_data:
                # Do nothing
                pass
            else:
                fmriprep_data.update({sub: OrderedDict()})

            for i, ses in enumerate(fmriprep.data[sub]):
                for j, run in enumerate(fmriprep.data[sub][ses]):
                    m = re.search('.*_(task-[^_]*)_.*', os.path.basename(run['task_event_file']))
                    if m:
                        task_label = m.group(1)
                    else:
                        raise RuntimeError('Failed to detect task label.')
                    if task_label in fmriprep_data[sub]:
                        if ses in fmriprep_data[sub][task_label]:
                            fmriprep_data[sub][task_label][ses].append(run)
                        else:
                            fmriprep_data[sub][task_label].update({ses: [run]})
                    else:
                        ses_data = OrderedDict()
                        ses_data.update({ses: [run]})
                        fmriprep_data[sub].update({task_label: ses_data})
    else:
        # Do not split the data
        fmriprep_data = fmriprep.data

    # Create BData from fmriprep outputs
    bdata_list = []
    data_labels = []

    if split_task_label:
        # One subject/task, one BData
        for sbj, sbjdata in fmriprep_data.items():
            for tsk, tskdata in sbjdata.items():
                print('----------------------------------------')
                print('Subject: %s\n' % sbj)
                print('Task:    %s'   % tsk)
                bdata = __create_bdata_fmriprep_subject(tskdata, data_mode, data_path=dpath, label_mapper=label_mapper_dict, with_confounds=with_confounds)
                bdata_list.append(bdata)
                data_labels.append('%s_%s' % (sbj, tsk))
    else:
        # One subject, one BData (default)
        for sbj, sbjdata in fmriprep.data.items():
            print('----------------------------------------')
            print('Subject: %s\n' % sbj)

            bdata = __create_bdata_fmriprep_subject(sbjdata, data_mode, data_path=dpath, label_mapper=label_mapper_dict, with_confounds=with_confounds)
            bdata_list.append(bdata)
            data_labels.append(sbj)

    if return_data_labels:
        if not return_list and len(bdata_list) == 1:
            return bdata_list[0], data_labels[0]
        else:
            return bdata_list, data_labels
    else:
        if not return_list and len(bdata_list) == 1:
            return bdata_list[0]
        else:
            return bdata_list


class BrainData(object):
    '''fMRI data class (volume or surface).'''

    def __init__(self, dpath, dtype='volume'):
        self.__dpath = dpath
        self.__dtype = dtype
        self.__data = np.array([])
        self.__xyz = np.array([])
        self.__index = np.array([])

        self.__n_vertex = (-1, -1)

        if self.__dtype == 'volume':
            self.__load_volume()
        elif self.__dtype == 'surface':
            self.__load_surface()
        else:
            raise ValueError('Unknown dtype: %s' % self.__dtype)

    @property
    def data(self):
        return self.__data

    @property
    def xyz(self):
        if self.__dtype is 'surface':
            raise NotImplementedError('Vertex xyz coordinates are not implemented yet.')
        return self.__xyz

    @property
    def index(self):
        return self.__index

    @property
    def n_vertex(self):
        if self.__dtype is not 'surface':
            raise TypeError('Not surface data.')
        return self.__n_vertex

    def __load_volume(self):
        '''Load a MRI image.

        - Returns data as 2D array (sample x voxel)
        - Returns voxle xyz coordinates (3 x voxel)
        - Returns voxel ijk indexes (3 x voxel)
        - Data, xyz, and ijk are flattened by Fortran-like index order
        '''
        img = nipy.load_image(self.__dpath)

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

        self.__data = data
        self.__xyz = xyz
        self.__index = ijk

        return None

    def __load_surface(self):
        print('Loading %s ...' % self.__dpath[0])
        vertex_left = self.__load_surf_func_file(self.__dpath[0])
        print('Loading %s ...' % self.__dpath[1])
        vertex_right = self.__load_surf_func_file(self.__dpath[1])

        n_vertex_left = vertex_left.shape[1]
        n_vertex_right = vertex_right.shape[1]

        # TOOD: check size

        self.__data = np.hstack([vertex_left, vertex_right])
        self.__index = np.array([np.hstack([np.arange(vertex_left.shape[1]),
                                            np.arange(vertex_right.shape[1])])])
        self.__n_vertex = (n_vertex_left, n_vertex_right)

        # TODO: add vertex xyz

        return None

    def __load_surf_func_file(self, fpath):
        surf = nibabel.load(fpath)
        data_arrays = surf.darrays
        data_matrix_list = []
        for d in data_arrays:
            # TODO: checks vertex num
            data_matrix_list.append(d.data)
        data_matrix = np.vstack(data_matrix_list)
        return data_matrix


class LabelMapper(object):
    '''Label mapper class.'''

    def __init__(self, l2v_map):
        self.__l2v_map = l2v_map
        self.__v2l_map = {}

    def get_value(self, mkey, label):
        if not mkey in self.__l2v_map:
            raise RuntimeError('%s not found in label mapper' % mkey)

        if label == 'n/a':
            return np.nan

        val = self.__l2v_map[mkey][label]
        if not mkey in self.__v2l_map:
            self.__v2l_map.update({mkey: {val: label}})
        else:
            if label in self.__v2l_map[mkey].values():
                label_exist = self.__v2l_map[mkey][val]
                if label != label_exist:
                    raise RuntimeError('Invalid label-value mapping (possibly non-unique values in label mapper)')
            self.__v2l_map[mkey].update({val: label})

        return val

    def dump(self):
        return self.__v2l_map


def create_bdata_singlesubject(subject_data, data_mode, data_path='./', label_mapper={}, with_confounds=False, cut_run=False):
    return __create_bdata_fmriprep_subject(subject_data, data_mode, data_path=data_path, label_mapper=label_mapper, with_confounds=with_confounds, cut_run=cut_run)


def __create_bdata_fmriprep_subject(subject_data, data_mode, data_path='./', label_mapper={}, with_confounds=False, cut_run=False):
    if data_mode in ['surface_standard', 'surface_standard_41k', 'surface_standard_10k', 'surface_native']:
        is_surf = True
    else:
        is_surf = False

    braindata_list = []
    xyz = np.array([])
    ijk = np.array([])

    motionparam_list = []
    # confounds = {'global_signal':          [],
    #              'white_matter':           [],
    #              'csf':                    [],
    #              'dvars':                  [],
    #              'std_dvars':              [],
    #              'framewise_displacement': [],
    #              'a_comp_cor_00':          [],
    #              'a_comp_cor_01':          [],
    #              'a_comp_cor_02':          [],
    #              'a_comp_cor_03':          [],
    #              'a_comp_cor_04':          [],
    #              'a_comp_cor_05':          [],
    #              't_comp_cor_00':          [],
    #              't_comp_cor_01':          [],
    #              't_comp_cor_02':          [],
    #              't_comp_cor_03':          [],
    #              't_comp_cor_04':          [],
    #              't_comp_cor_05':          [],
    #              'cosine00':               [],
    #              'cosine01':               [],
    #              'cosine02':               [],
    #              'cosine03':               [],
    #              'cosine04':               [],
    #              'cosine05':               [],
    # }
    confounds = {}

    ses_label_list = []
    run_label_list = []
    block_label_list = []
    labels_list = []

    last_run = 0
    last_block = 0

    act_label_map = LabelMapper(label_mapper)

    run_idx = 0
    
    for i, (ses, sesdata) in enumerate(subject_data.items()):
        print('Session: %d (%s)' % (i + 1, ses))
        print('Data: %s\n' % data_mode)

        for j, run in enumerate(sesdata):
            run_idx += 1
            print('Run %d' % (j + 1))
            epi = run[data_mode]
            event_file = run['task_event_file']
            confounds_file = run['confounds']
            if is_surf:
                print('EPI:             %s, %s' % epi)
            else:
                print('EPI:             %s' % epi)
            print('Task event file: %s' % event_file)
            print('Confounds file:  %s' % confounds_file)

            mp_label_col = ['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ',
                            'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

            # Load brain data (volume or surface)
            if is_surf:
                brain = BrainData((os.path.join(data_path, epi[0]), os.path.join(data_path, epi[1])), dtype='surface')
            else:
                brain = BrainData(os.path.join(data_path, epi), dtype='volume')
                xyz = brain.xyz

            braindata_list.append(brain.data)
            ijk = brain.index

            num_vol = brain.data.shape[0]

            # Load motion parameters (and the other confounds)
            conf_pd = pd.read_csv(os.path.join(data_path, confounds_file), delimiter='\t')

            mp_label = [c for c in conf_pd.columns if c in mp_label_col]
            if len(mp_label) != 6:
                raise RuntimeError('Invalid confounds file: %s' % os.path.join(data_path, confounds_file))

            mp = np.hstack([np.c_[conf_pd[a]] for a in mp_label])
            motionparam_list.append(mp)

            if with_confounds:
                confounds_keys = [k for k in list(conf_pd.columns) if not k in mp_label_col]
                for c in confounds_keys:
                    x = np.c_[conf_pd[c]]
                    if c in confounds:
                        confounds[c].update({run_idx: x})
                    else:
                        confounds.update({c: {run_idx: x}})

            # Load task event file
            event_file = os.path.join(data_path, run['task_event_file'])
            events = pd.read_csv(event_file, delimiter='\t', keep_default_na=False)

            # Check time length
            tlen_event = events.tail(1)['onset'].values[0] + events.tail(1)['duration'].values[0]
            n_sample = brain.data.shape[0]

            with open(os.path.join(data_path, run['bold_json']), 'r') as f:
                bold_metainfo = json.load(f)
            tr = bold_metainfo['RepetitionTime']
            tr_ms = tr * 1000 # To avoid numerical error

            if int(tlen_event * 1000) != int(n_sample * tr_ms):
                if cut_run:
                    cut_duration = (tlen_event * 1000 - (n_sample * tr_ms)) / 1000
                    warnings.warn('The number of volumes in the EPI file (%s) '
                                  'and time duration in the corresponding task (%s) '
                                  'event file mismatch! The first %f sec in the task '
                                  'event file will be cropped.'
                                  % (epi, run['task_event_file'], cut_duration))
                else:
                    raise ValueError('The number of volumes in the EPI file (%s) '
                                     'and time duration in the corresponding task (%s) '
                                     'event file mismatch!'
                                     % (epi, run['task_event_file']))

            # Make block and labels
            blocks = []
            labels = []

            cols = events.columns.values
            cols = cols[~(cols == 'onset')]
            cols = cols[~(cols == 'duration')]

            for k, row in events.iterrows():
                onset = row['onset']
                duration = row['duration']

                if cut_run:
                    if cut_duration > 0:
                        onset = onset - cut_duration
                        if onset < 0:
                            onset = 0
                            duration = duration - cut_duration
                    elif cut_duration < 0:
                        raise NotImplementedError

                nsmp = int(np.round(duration / tr))

                # Block
                blocks.append(np.ones((nsmp, 1)) * (k + 1))

                # Label
                label_vals = []
                for p in cols:
                    if p in label_mapper:
                        v = act_label_map.get_value(p, row[p])
                        label_vals.append(v)
                    else:
                        label_vals.append(row[p])
                label_vals = np.array([np.nan if x == 'n/a' else np.float(x)
                                       for x in label_vals])
                label_mat = np.tile(label_vals, (nsmp, 1))
                labels.append(label_mat)

            ses_label_list.append(np.ones((num_vol, 1)) * (i + 1))
            run_label_list.append(np.ones((num_vol, 1)) * (j + 1) + last_run)
            block_label_list.append(np.vstack(blocks) + last_block)
            labels_list.append(np.vstack(labels))

            last_block = block_label_list[-1][-1]

        last_run = run_label_list[-1][-1]

        print('')

    braindata = np.vstack(braindata_list)
    motionparam = np.vstack(motionparam_list)
    ses_label = np.vstack(ses_label_list)
    run_label = np.vstack(run_label_list)
    block_label = np.vstack(block_label_list)
    labels_label = np.vstack(labels_list)

    if with_confounds:
        for c in confounds:
            conf_val_list = []
            for k in range(run_idx):
                if (k + 1) in confounds[c]:
                    conf_val_list.append(confounds[c][k + 1])
                else:
                    run_length = braindata_list[k].shape[0]
                    nan_array = np.zeros([run_length, 1])
                    nan_array[:, :] = np.nan
                    conf_val_list.append(nan_array)
            confounds.update({c: np.vstack(conf_val_list)})

        if len(set([c.shape for c in confounds.values()])) != 1:
            raise RuntimeError('Invalid confounds.')

    # Create BData (one subject, one file)
    bdata = bdpy.BData()

    if is_surf:
        bdata.add(braindata, 'VertexData')
        n_vertex = brain.n_vertex
        bdata.add_metadata('VertexLeft', np.array([1] * n_vertex[0] + [0] * n_vertex[1]),
                           where='VertexData')
        bdata.add_metadata('VertexRight', np.array([0] * n_vertex[0] + [1] * n_vertex[1]),
                           where='VertexData')
    else:
        bdata.add(braindata, 'VoxelData')

    bdata.add(ses_label, 'Session')
    bdata.add(run_label, 'Run')
    bdata.add(block_label, 'Block')
    bdata.add(labels_label, 'Label')
    bdata.add(motionparam, 'MotionParameter')
    bdata.add_metadata('MotionParameter_trans_x', [1, 0, 0, 0, 0, 0], 'Motion parameter: x translation', where='MotionParameter')
    bdata.add_metadata('MotionParameter_trans_y', [0, 1, 0, 0, 0, 0], 'Motion parameter: y translation', where='MotionParameter')
    bdata.add_metadata('MotionParameter_trans_z', [0, 0, 1, 0, 0, 0], 'Motion parameter: z translation', where='MotionParameter')
    bdata.add_metadata('MotionParameter_rot_x', [0, 0, 0, 1, 0, 0], 'Motion parameter: x rotation', where='MotionParameter')
    bdata.add_metadata('MotionParameter_rot_y', [0, 0, 0, 0, 1, 0], 'Motion parameter: y rotation', where='MotionParameter')
    bdata.add_metadata('MotionParameter_rot_z', [0, 0, 0, 0, 0, 1], 'Motion parameter: z rotation', where='MotionParameter')

    if with_confounds:
        default_confounds_keys = [
            'global_signal',
            'white_matter',
            'csf',
            'dvars',
            'std_dvars',
            'framewise_displacement',
            'a_comp_cor',
            'a_comp_cor_00',
            'a_comp_cor_01',
            'a_comp_cor_02',
            'a_comp_cor_03',
            'a_comp_cor_04',
            'a_comp_cor_05',
            't_comp_cor',
            't_comp_cor_00',
            't_comp_cor_01',
            't_comp_cor_02',
            't_comp_cor_03',
            't_comp_cor_04',
            't_comp_cor_05',
            'cosine',
            'cosine00',
            'cosine01',
            'cosine02',
            'cosine03',
            'cosine04',
            'cosine05',
        ]
        confounds_key_desc = {
            'global_signal':          {'key': 'GlobalSignal',          'desc': 'Confounds: Average signal in brain mask'},
            'white_matter':           {'key': 'WhiteMatterSignal',     'desc': 'Confounds: Average signal in white matter'},
            'csf':                    {'key': 'CSFSignal',             'desc': 'Confounds: Average signal in CSF'},
            'dvars':                  {'key': 'DVARS',                 'desc': 'Confounds: Original DVARS'},
            'std_dvars':              {'key': 'STD_DVARS',             'desc': 'Confounds: Standardized DVARS'},
            'framewise_displacement': {'key': 'FramewiseDisplacement', 'desc': 'Confounds: Framewise displacement (bulk-head motion)'},
            'a_comp_cor':             {'key': 'aCompCor',              'desc': 'Confounds: Anatomical CompCor'},
            'a_comp_cor_00':          {'key': 'aCompCor_0',            'desc': 'Confounds: Anatomical CompCor'},
            'a_comp_cor_01':          {'key': 'aCompCor_1',            'desc': 'Confounds: Anatomical CompCor'},
            'a_comp_cor_02':          {'key': 'aCompCor_2',            'desc': 'Confounds: Anatomical CompCor'},
            'a_comp_cor_03':          {'key': 'aCompCor_3',            'desc': 'Confounds: Anatomical CompCor'},
            'a_comp_cor_04':          {'key': 'aCompCor_4',            'desc': 'Confounds: Anatomical CompCor'},
            'a_comp_cor_05':          {'key': 'aCompCor_5',            'desc': 'Confounds: Anatomical CompCor'},
            't_comp_cor':             {'key': 'tCompCor',              'desc': 'Confounds: Temporal CompCor'},
            't_comp_cor_00':          {'key': 'tCompCor_0',            'desc': 'Confounds: Temporal CompCor'},
            't_comp_cor_01':          {'key': 'tCompCor_1',            'desc': 'Confounds: Temporal CompCor'},
            't_comp_cor_02':          {'key': 'tCompCor_2',            'desc': 'Confounds: Temporal CompCor'},
            't_comp_cor_03':          {'key': 'tCompCor_3',            'desc': 'Confounds: Temporal CompCor'},
            't_comp_cor_04':          {'key': 'tCompCor_4',            'desc': 'Confounds: Temporal CompCor'},
            't_comp_cor_05':          {'key': 'tCompCor_5',            'desc': 'Confounds: Temporal CompCor'},
            'cosine':                 {'key': 'Cosine',                'desc': 'Confounds: Discrete cosine-basis regressors'},
            'cosine00':               {'key': 'Cosine_0',              'desc': 'Confounds: Discrete cosine-basis regressors'},
            'cosine01':               {'key': 'Cosine_1',              'desc': 'Confounds: Discrete cosine-basis regressors'},
            'cosine02':               {'key': 'Cosine_2',              'desc': 'Confounds: Discrete cosine-basis regressors'},
            'cosine03':               {'key': 'Cosine_3',              'desc': 'Confounds: Discrete cosine-basis regressors'},
            'cosine04':               {'key': 'Cosine_4',              'desc': 'Confounds: Discrete cosine-basis regressors'},
            'cosine05':               {'key': 'Cosine_5',              'desc': 'Confounds: Discrete cosine-basis regressors'},
        }
        confounds_array = np.hstack([
            confounds[dck]
            for dck in default_confounds_keys if dck in confounds
            ])

        bdata.add(confounds_array, 'Confounds')

        cnf_p = 0
        for cnf in default_confounds_keys:
            if (not cnf in ['a_comp_cor', 't_comp_cor', 'cosine']) and (not cnf in confounds):
                continue

            cnf_colidx = np.zeros(confounds_array.shape[1])

            if cnf in ['a_comp_cor', 't_comp_cor', 'cosine']:
                ncol = sum([1 for k in confounds.keys() if cnf in k])
                for k in range(ncol):
                    cnf_colidx[cnf_p + k] = 1
            else:
                cnf_colidx[cnf_p] = 1
                cnf_p += 1

            bdata.add_metadata(confounds_key_desc[cnf]['key'],
                               cnf_colidx,
                               confounds_key_desc[cnf]['desc'],
                               where='Confounds')

    for i, col in enumerate(cols):
        metadata_vec = np.empty((len(cols),))
        metadata_vec[:] = np.nan
        metadata_vec[i] = 1
        bdata.add_metadata(col, metadata_vec, 'Label %s' % col, where='Label')

    if is_surf:
        # bdata.add_metadata('vertex_x', xyz[0, :], 'Vertex x coordinate', where='VertexData')
        # bdata.add_metadata('vertex_y', xyz[1, :], 'Vertex y coordinate', where='VertexData')
        # bdata.add_metadata('vertex_z', xyz[2, :], 'Vertex z coordinate', where='VertexData')
        bdata.add_metadata('vertex_index', ijk[0, :], 'Vertex index', where='VertexData')
    else:
        bdata.add_metadata('voxel_x', xyz[0, :], 'Voxel x coordinate', where='VoxelData')
        bdata.add_metadata('voxel_y', xyz[1, :], 'Voxel y coordinate', where='VoxelData')
        bdata.add_metadata('voxel_z', xyz[2, :], 'Voxel z coordinate', where='VoxelData')
        bdata.add_metadata('voxel_i', ijk[0, :], 'Voxel i index', where='VoxelData')
        bdata.add_metadata('voxel_j', ijk[1, :], 'Voxel j index', where='VoxelData')
        bdata.add_metadata('voxel_k', ijk[2, :], 'Voxel k index', where='VoxelData')

    # Value-label mapper
    vmap = act_label_map.dump()
    for k in vmap:
        bdata.add_vmap(k, vmap[k])

    return bdata


def __get_xyz(img):
    if len(img.shape) == 4:
        # 4D-image
        i_len, j_len, k_len, t = img.shape
        affine = np.delete(np.delete(img.coordmap.affine, 3, axis=0), 3, axis=1)
    else:
        # 3D-image
        i_len, j_len, k_len = img.shape
        affine = img.coordmap.affine
    ijk = np.array(list(itertools.product(xrange(i_len),
                                          xrange(j_len),
                                          xrange(k_len),
                                          [1]))).T
    return np.dot(affine, ijk)[:-1]


def __load_mri(fpath):
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


if __name__ == '__main__':
    pass
