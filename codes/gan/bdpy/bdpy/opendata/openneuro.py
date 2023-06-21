import os
import re
import shutil
import json
from glob import glob

from bdpy import makedir_ifnot


def makedata(src, source_type='bids_daily', output_dir='./output', root_dir='./', bids_dir='bids', mri_filetype='nii', dry_run=False):
    '''Create BIDS dataset for OpenNeuro.'''

    if not source_type == 'bids_daily':
        raise NotImplementedError('Source type %s not supported.' % source_type)

    __create_dir(output_dir)
    __create_dir(os.path.join(output_dir, 'sourcedata'))
    __create_dir(os.path.join(output_dir, 'derivatives'))

    readme = os.path.join(output_dir, 'README')
    description = os.path.join(output_dir, 'dataset_description.json')
    if not os.path.exists(readme):
        print('Creating %s' % readme)
        with open(readme, 'w'):
            pass
    if not os.path.exists(description):
        print('Creating %s' % description)
        desc_init = {
            'Name': '',
            'License': '',
            'Authors': [],
            'Acknowledgements': '',
            'HowToAcknowledge': '',
            'Funding': [],
            'ReferencesAndLinks': [],
            'DatasetDOI': '',
            'BIDSVersion': '1.0.2',
        }
        with open(description, 'w') as f:
            json.dump(desc_init, f, indent=4)

    for subject, sub_data in src.items():
        print('----------------------------------------')
        print(subject)

        subject_dir = os.path.join(output_dir, subject)
        __create_dir(subject_dir)

        # Reference anatomy
        src_anat = os.path.join(root_dir, sub_data['anat'])
        trg_anat_dir = os.path.join(subject_dir, 'ses-anatomy', 'anat')
        trg_anat_raw = os.path.join(trg_anat_dir, '%s_ses-anatomy_T1w_raw.nii.gz' % subject)
        trg_anat_defaced = os.path.join(trg_anat_dir, '%s_ses-anatomy_T1w.nii.gz' % subject)
        __create_dir(trg_anat_dir)
        if not dry_run:
            if mri_filetype == 'nii.gz':
                shutil.copy2(src_anat, trg_anat_raw)  # FIXME: convert to nii.gz
            else:
                convert_command = 'mri_convert %s %s' % (src_anat, trg_anat_raw)
                os.system(convert_command)
            deface_command = 'pydeface.py %s %s' % (trg_anat_raw, trg_anat_defaced)
            os.system(deface_command)
            os.remove(trg_anat_raw)

        # Functionals
        tasks = sub_data['func']
        if isinstance(tasks, list):
            tasks = {'': tasks}

        print('%d task(s)' % len(tasks))
        print('')

        for task, srcdata in tasks.items():
            print('--------------------')
            print(task)
            print('%d data' % len(srcdata))
            print('')

            task_sessions = []

            for src in srcdata:
                if isinstance(src, dict):
                    src_name = src.keys()[0]
                    src_info = src[src_name]
                else:
                    src_name = src
                    src_info = None

                src_bids_path = os.path.join(root_dir, src_name, bids_dir)

                print(src_name)

                if not os.path.isdir(src_bids_path):
                    raise RuntimeError('Invalid BIDS directory: %s' % src_bids_path)

                sessions = __parse_bids_dir(src_bids_path, data_info=src_info, mri_filetype=mri_filetype)
                task_sessions.extend(sessions)

            print('Total sessions: %d' % len(task_sessions))
            print('Total runs: %d' % sum([len(ses['functionals']) for ses in task_sessions]))
            print('')

            # Copy files
            for i, ses in enumerate(task_sessions):
                session_label = 'ses-%s%02d' % (task, i +1)
                print('Session: %s' % session_label)

                session_dir = os.path.join(subject_dir, session_label)

                if not dry_run:
                    __create_dir(session_dir)
                    __create_dir(os.path.join(session_dir, 'anat'))
                    __create_dir(os.path.join(session_dir, 'func'))

                # T2 inplane image
                src_inplane = ses['inplane']
                if not src_inplane is None:
                    rename_table = {
                        os.path.basename(src_inplane).split('_')[0]: subject,       # SUbject ID
                        os.path.basename(src_inplane).split('_')[1]: session_label, # Session label
                    }
                    trg_inplane = os.path.join(session_dir, 'anat',
                                               __rename_file(os.path.basename(src_inplane).split('.')[0] + '.nii.gz',
                                                             rename=rename_table))
                    print('Copying\n  from: %s\n  to: %s' % (src_inplane, trg_inplane))
                    if not dry_run:
                        if mri_filetype == 'nii.gz':
                            shutil.copy2(src_inplane, trg_inplane)  # FIXME: convert to nii.gz
                        else:
                            convert_command = 'mri_convert %s %s' % (src_inplane, trg_inplane)
                            os.system(convert_command)

                # Functionals
                for j, run in enumerate(ses['functionals']):
                    src_bold = run['bold']
                    src_bold_json = run['bold_json']
                    src_event = run['event']

                    # File renaming
                    rename_table = {
                        os.path.basename(src_bold).split('_')[0]: subject,       # SUbject ID
                        os.path.basename(src_bold).split('_')[1]: session_label, # Session label
                                   }

                    # Fix run label
                    run_label = re.match('.*_run-(\d+)_.*', os.path.basename(src_bold)).group(1)
                    if (j + 1) != int(run_label):
                        print('Fix run label: run-%s --> run-%02d' % (run_label, j + 1))
                        rename_table.update({'run-%s' % run_label: 'run-%02d' % (j + 1)})

                    trg_bold = os.path.join(session_dir, 'func',
                                            __rename_file(os.path.basename(src_bold).split('.')[0] + '.nii.gz',
                                                          rename=rename_table))
                    trg_bold_json = os.path.join(session_dir, 'func',
                                                 __rename_file(os.path.basename(src_bold_json), rename=rename_table))
                    trg_event = os.path.join(session_dir, 'func',
                                             __rename_file(os.path.basename(src_event), rename=rename_table))

                    print('Copying\n  from: %s\n  to: %s' % (src_bold, trg_bold))
                    print('Copying\n  from: %s\n  to: %s' % (src_bold_json, trg_bold_json))
                    print('Copying\n  from: %s\n  to: %s' % (src_event, trg_event))
                    if not dry_run:
                        if mri_filetype == 'nii.gz':
                            shutil.copy2(src_bold, trg_bold)
                        else:
                            convert_command = 'mri_convert %s %s' % (src_bold, trg_bold)
                            os.system(convert_command)
                        shutil.copy2(src_bold_json, trg_bold_json)
                        shutil.copy2(src_event, trg_event)

    return None


def __parse_bids_dir(dpath, data_info=None, mri_filetype='nii'):
    print('BIDS directory: %s' % dpath)

    sub_dirs = glob(os.path.join(dpath, 'sub-*'))
    if len(sub_dirs) != 1:
        raise RuntimeError('Unsupported BIDS data (invalid number of subjects: %d)' % len(sub_dirs))
    sub_dir = os.path.join(dpath, sub_dirs[0])
    print('Subject directory: %s' % sub_dir)

    ses_dirs = sorted(glob(os.path.join(sub_dirs[0], 'ses-*', 'func')))
    print('%d func session(s) found' % len(ses_dirs))

    sessions = []
    for i, ses_dir in enumerate(ses_dirs):

        # Session selection
        skip = False
        if (not data_info is None) and 'ses' in data_info:
            if isinstance(data_info['ses'], int) and (i + 1) != data_info['ses']:
                skip = True
            elif isinstance(data_info['ses'], list) and (i + 1) not in data_info['ses']:
                skip = True
            elif isinstance(data_info['ses'], str) and (i + 1) not in [int(x) for x in data_info['ses'].split(',')]:
                skip = True

        if skip:
            print('Skipping session %02d' % (i + 1))
            continue

        # T2 inplane image
        inplane_file = __aggregate_mri_files(os.path.join(ses_dir, '../anat'), mri_filetype=mri_filetype)
        if len(inplane_file) != 1:
            raise RuntimeError('Invalid inplane anatomy')
        inplane_file = inplane_file[0]

        # Functionals
        run_files = __aggregate_runs(ses_dir, mri_filetype=mri_filetype)
        print('Ses %02d: %d run(s) found' % (i + 1, len(run_files)))
        #print(run_files)

        # Run selection
        run_files_keep = []
        if (not data_info is None) and 'discard_run' in data_info:
            for j, rf in enumerate(run_files):
                skip_run = False
                if isinstance(data_info['discard_run'], int) and (j + 1) == data_info['discard_run']:
                    skip_run = True
                elif isinstance(data_info['discard_run'], list) and (j + 1) in data_info['discard_run']:
                    skip_run = True
                elif isinstance(data_info['discard_run'], str) and (j + 1) in [int(x) for x in data_info['discard_run'].split(',')]:
                    skip_run = True
                if skip_run:
                    print('Skipping run %02d' % (j + 1))
                else:
                    run_files_keep.append(rf)
        else:
            run_files_keep = run_files


        sessions.append({'inplane': inplane_file,
                         'functionals': run_files_keep})

    print('')

    return sessions


def __aggregate_runs(dpath, mri_filetype='nii'):
    mri_files = __aggregate_mri_files(dpath, mri_filetype=mri_filetype)

    run_files = []
    for mri_file in mri_files:
        basename = os.path.basename(mri_file)

        # Json file
        json_file = os.path.join(dpath, basename.replace('_bold.' + mri_filetype, '_bold.json'))
        if not os.path.isfile(json_file):
            json_file = None

        # Task event file
        task_event_file = os.path.join(dpath, basename.replace('_bold.' + mri_filetype, '_events.tsv'))
        if not os.path.isfile(task_event_file):
            task_event_file = None

        scan_files = sorted(glob(os.path.join(dpath, basename + '*')))

        run_files.append({'bold': mri_file,
                          'bold_json': json_file,
                          'event': task_event_file})

    return run_files


def __aggregate_mri_files(dpath, mri_filetype='nii'):
    mri_files = glob(os.path.join(dpath, '*.' + mri_filetype))
    return mri_files


def __rename_file(fname, rename={}):
    for before, after in rename.items():
        fname = fname.replace(before, after)
    return fname


def __create_dir(dirpath):
    print('Creating %s' % dirpath)
    makedir_ifnot(dirpath)
    return None
