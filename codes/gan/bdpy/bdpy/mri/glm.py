'''bdpy.mri.glm'''


import csv

import numpy as np
from nipy.modalities.fmri.experimental_paradigm import BlockParadigm, EventRelatedParadigm


def make_paradigm(event_files, num_vols, tr=2., cond_col=2, label_col=None, regressors=None, ignore_col=None, ignore_value=[], trial_wise=False, design='block'):
    '''
    Make paradigm for GLM with Nipy from BIDS task event files.

    Parameters
    ----------
    event_files : list
      List of task event files.
    num_vols : list
      List of the number of volumes in each run.
    tr : int or float
      TR in sec.
    cond_col : int
      Index of the condition column in the task event files.
    label_col : int
      Index of the label column in the task event files.
    regressors : list
      Names of regressors (conditions) included in the design matrix.
    ignore_col : int
      Index of the column to be ingored.
    ignore_value : list
      List of values to be ignored.
    design : 'block' or 'event_related
      Specifying experimental design.
    trial_wise : bool
      Returns trial-wise design matrix if True.

    Returns
    -------
    dict
      paradigm : nipy.Paradigm
      condition_labels : labels for task regressors
      run_regressors : nuisance regressors for runs
      run_regressors_label : labels for the run regressors
    '''

    onset = []
    duration = []
    conds = []
    labels = []

    # Run regressors
    run_regs = []
    run_regs_labels = []

    n_total_vols = np.sum(num_vols)

    trial_count = 0

    # Combining all runs/sessions into a single design matrix
    for i, (ef, nv) in enumerate(zip(event_files, num_vols)):
        n_run = i + 1

        with open(ef, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            header = reader.next()
            for row in reader:
                if not regressors is None and not row[cond_col] in regressors:
                    continue
                if not ignore_col is None:
                    if row[ignore_col] in ignore_value:
                        continue
                trial_count += 1
                onset.append(float(row[0]) + i * nv * tr)
                duration.append(float(row[1]))
                if trial_wise:
                    conds.append('trial-%06d' % trial_count)
                else:
                    conds.append(row[cond_col])
                if not label_col is None:
                    labels.append(row[label_col])

        # Run regressors
        run_reg = np.zeros((n_total_vols, 1))
        run_reg[i * nv:(i + 1) * nv] = 1
        run_regs.append(run_reg)
        run_regs_labels.append('run-%02d' % n_run)

    run_regs = np.hstack(run_regs)

    if design == 'event_related':
        paradigm = EventRelatedParadigm(con_id=conds, onset=onset)
    else:
        paradigm = BlockParadigm(con_id=conds, onset=onset, duration=duration)

    return {'paradigm': paradigm,
            'run_regressors': run_regs,
            'run_regressor_labels': run_regs_labels,
            'condition_labels': labels}
