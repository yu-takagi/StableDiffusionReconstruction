import csv
import glob
import itertools
import os
import re
import json
from collections import OrderedDict

import numpy as np
import nipy
import nibabel
import pandas as pd

#import bdpy
from .fmriprep import create_bdata_singlesubject


def create_bdata_spm_domestic(dpath, data_mode='volume_native',
                              label_mapper=None, exclude={},
                              split_task_label=False,
                              with_confounds=False,
                              return_data_labels=False,
                              return_list=False):
    '''Create BData from SPM outputs (Kamitani lab domestic data structure).'''

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

    # Load SPM outputs
    spm_out = SpmDomestic(dpath)

    # Exclude subject, session, or run
    for sub in spm_out.data:
        # Exclude subject
        if 'subject' in exclude and sub in exclude['subject']:
            del(spm_out.data[sub])
            continue

        for i, ses in enumerate(spm_out.data[sub]):
            # Exclude session
            if 'session' in exclude and i + 1 in exclude['session']:
                del(spm_out.data[sub][ses])
                continue

            # Exclude run
            run_survive = [run for j, run in enumerate(spm_out.data[sub][ses])
                           if not ('run' in exclude and j + 1 in exclude['run'])]
            spm_out.data[sub][ses] = run_survive

            # Exclude session/run
            if 'session/run' in exclude:
                ex_runs = exclude['session/run'][i]
                if ex_runs is not None and len(ex_runs) != 0:
                    run_survive = [run for j, run in enumerate(spm_out.data[sub][ses])
                                   if not j + 1 in ex_runs]
                    spm_out.data[sub][ses] = run_survive

    # Split data
    # TODO: needs refactoring, obviously
    if split_task_label:
        spm_out_data = OrderedDict()
        for sub in spm_out.data:

            if sub in spm_out_data:
                # Do nothing
                pass
            else:
                spm_out_data.update({sub: OrderedDict()})

            for i, ses in enumerate(spm_out.data[sub]):
                for j, run in enumerate(spm_out.data[sub][ses]):
                    m = re.search('.*_(task-[^_]*)_.*', os.path.basename(run['task_event_file']))
                    if m:
                        task_label = m.group(1)
                    else:
                        raise RuntimeError('Failed to detect task label.')
                    if task_label in spm_out_data[sub]:
                        if ses in spm_out_data[sub][task_label]:
                            spm_out_data[sub][task_label][ses].append(run)
                        else:
                            spm_out_data[sub][task_label].update({ses: [run]})
                    else:
                        ses_data = OrderedDict()
                        ses_data.update({ses: [run]})
                        spm_out_data[sub].update({task_label: ses_data})
    else:
        # Do not split the data
        spm_out_data = spm_out.data

    # Create BData from fmriprep outputs
    bdata_list = []
    data_labels = []

    if split_task_label:
        # One subject/task, one BData
        for sbj, sbjdata in spm_out_data.items():
            for tsk, tskdata in sbjdata.items():
                print('----------------------------------------')
                print('Subject: %s\n' % sbj)
                print('Task:    %s'   % tsk)
                bdata = create_bdata_singlesubject(tskdata, data_mode, data_path=dpath, label_mapper=label_mapper_dict, with_confounds=with_confounds, cut_run=True)
                bdata_list.append(bdata)
                data_labels.append('%s_%s' % (sbj, tsk))
    else:
        # One subject, one BData (default)
        for sbj, sbjdata in spm_out.data.items():
            print('----------------------------------------')
            print('Subject: %s\n' % sbj)

            bdata = create_bdata_singlesubject(sbjdata, data_mode, data_path=dpath, label_mapper=label_mapper_dict, with_confounds=with_confounds, cut_run=True)
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


class SpmDomestic(object):
    '''SPM outputs (domestic data structure) class.'''

    def __init__(self, datapath=None):
        self.__datapath = datapath
        self.__data = OrderedDict()

        if self.__datapath is not None:
            self.__parse_data()

    @property
    def data(self):
        return self.__data

    def __parse_data(self):

        sub_dirs = glob.glob(os.path.join(self.__datapath, 'sub-*/'))
        print(sub_dirs)

        for sub_dir in sub_dirs:
            sub_id = sub_dir.split('/')[-2]
            self.__data.update({sub_id: self.__parse_subject(sub_dir)})

    def __parse_subject(self, dpath):

        sub_id = dpath.split('/')[-2]
        print('Subject ID: %s' % sub_id)

        bids_dir = dpath
        spm_dir = os.path.join(self.__datapath,
                               'derivatives/preproc_spm/output',
                               sub_id)

        print('  BIDS directory:       %s' % bids_dir)
        print('  SPM output directory: %s' % spm_dir)

        # Get functional session(s)
        ses_dirs = [d for d in os.listdir(bids_dir)
                    if os.path.isdir(os.path.join(bids_dir, d))
                    and d != 'ses-anat']

        # Get EPI files from SPM outputs
        epi_prep = {
            'volume-native':
            {(re.match('.*_(ses-\d+)_.*', f).group(1),
              re.match('.*_(run-\d+)_.*', f).group(1)):
             f
             for f in os.listdir(os.path.join(spm_dir, 'epi'))
             if os.path.splitext(f)[1] == '.nii'
             and f[0:2] == 'ra'},
            'volume-standard':
            {(re.match('.*_(ses-\d+)_.*', f).group(1),
              re.match('.*_(run-\d+)_.*', f).group(1)):
             f
             for f in os.listdir(os.path.join(spm_dir, 'epi'))
             if os.path.splitext(f)[1] == '.nii'
             and f[0:2] == 'wa'}
        }

        # Get motion param files
        mp_files = {
            re.match('.*_(ses-\d+)_.*', f).group(1):
            f
            for f in os.listdir(os.path.join(spm_dir, 'epi'))
            if os.path.splitext(f)[1] == '.txt'
            and f[0:3] == 'rp_'
        }

        ses_dict = OrderedDict()

        for ses_i, ses in enumerate(ses_dirs):
            print('Session %s' % ses)

            ses_dir_bids = os.path.join(bids_dir, ses, 'func')

            # Get runs
            runs = sorted([re.match('.*_(run-\d+)_.*', f).group(1)
                           for f in os.listdir(ses_dir_bids)
                           if os.path.splitext(f)[1] == '.nii'])

            # Get bold param files
            bold_param_files = {
                (re.match('.*_(ses-\d+)_.*', f).group(1),
                 re.match('.*_(run-\d+)_.*', f).group(1)):
                f
                for f in os.listdir(ses_dir_bids)
                if os.path.splitext(f)[1] == '.json'
            }

            # Get task event files
            event_files = {
                (re.match('.*_(ses-\d+)_.*', f).group(1),
                 re.match('.*_(run-\d+)_.*', f).group(1)):
                f
                for f in os.listdir(ses_dir_bids)
                if os.path.splitext(f)[1] == '.tsv'
            }

            # Split motion param files into runs and convert to BIDS format
            mp_file = os.path.join(spm_dir, 'epi', mp_files[ses])
            with open(mp_file, 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                mps = np.vstack([[a for a in row if a] for row in reader])

            if mps.shape[0] % len(runs) != 0:
                raise RuntimeError('Invalid EPI or realign parameter length')
            run_length = mps.shape[0] / len(runs)

            mp_header = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
            for i, run in enumerate(runs):
                mp_in_run = mps[0:run_length, :]
                mps = np.delete(mps, slice(0, run_length), 0)
                mp_run_file = os.path.join(spm_dir, 'epi', 'rp_a%s_%s_%s_bold.tsv' %(sub_id, ses, run))
                with open(mp_run_file, 'w') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerow(mp_header)
                    for k in range(mp_in_run.shape[0]):
                        writer.writerow(mp_in_run[k, :])

            if mps.shape[0] != 0:
                raise RuntimeError('Something was wrong on realign parameter processing')
                        
            runs_list = []

            for i, run in enumerate(runs):
                print('Run %s' % run)

                bids_path = os.path.join(bids_dir.replace(self.__datapath, ''), ses, 'func')
                if bids_path[0] == '/': bids_path = bids_path[1:]
                derv_path = spm_dir.replace(self.__datapath, '')
                if derv_path[0] == '/': derv_path = derv_path[1:]

                runs_list.append({
                    'volume_native': os.path.join(derv_path, 'epi', epi_prep['volume-native'][(ses, run)]),
                    'volume_standard': os.path.join(derv_path, 'epi', epi_prep['volume-standard'][(ses, run)]),
                    'bold_json': os.path.join(bids_path, bold_param_files[(ses, run)]),
                    'task_event_file': os.path.join(bids_path, event_files[(ses, run)]),
                    'confounds': os.path.join(derv_path, 'epi', 'rp_a%s_%s_%s_bold.tsv' %(sub_id, ses, run)),
                })

            ses_dict.update({ses: runs_list})

        return ses_dict
