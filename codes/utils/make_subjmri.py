import argparse
import os
import numpy as np
import pandas as pd
from nsd_access import NSDAccess
import scipy.io

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )

    opt = parser.parse_args()
    subject = opt.subject
    atlasname = 'streams'
    
    nsda = NSDAccess('../../nsd/')
    nsd_expdesign = scipy.io.loadmat('../../nsd/nsddata/experiments/nsd/nsd_expdesign.mat')

    # Note that most of nsd_expdesign indices are 1-base index!
    # This is why subtracting 1
    sharedix = nsd_expdesign['sharedix'] -1 

    behs = pd.DataFrame()
    for i in range(1,38):
        beh = nsda.read_behavior(subject=subject, 
                                session_index=i)
        behs = pd.concat((behs,beh))

    # Caution: 73KID is 1-based! https://cvnlab.slite.page/p/fRv4lz5V2F/Behavioral-data
    stims_unique = behs['73KID'].unique() - 1
    stims_all = behs['73KID'] - 1

    savedir = f'../../mrifeat/{subject}/'
    os.makedirs(savedir, exist_ok=True)

    if not os.path.exists(f'{savedir}/{subject}_stims.npy'):
        np.save(f'{savedir}/{subject}_stims.npy',stims_all)
        np.save(f'{savedir}/{subject}_stims_ave.npy',stims_unique)

    for i in range(1,38):
        print(i)
        beta_trial = nsda.read_betas(subject=subject, 
                                session_index=i, 
                                trial_index=[], # empty list as index means get all for this session
                                data_type='betas_fithrf_GLMdenoise_RR',
                                data_format='func1pt8mm')
        if i==1:
            betas_all = beta_trial
        else:
            betas_all = np.concatenate((betas_all,beta_trial),0)    
    
    atlas = nsda.read_atlas_results(subject=subject, atlas=atlasname, data_format='func1pt8mm')
    for roi,val in atlas[1].items():
        print(roi,val)
        if val == 0:
            print('SKIP')
            continue
        else:
            betas_roi = betas_all[:,atlas[0].transpose([2,1,0])==val]

        print(betas_roi.shape)
        
        # Averaging for each stimulus
        betas_roi_ave = []
        for stim in stims_unique:
            stim_mean = np.mean(betas_roi[stims_all == stim,:],axis=0)
            betas_roi_ave.append(stim_mean)
        betas_roi_ave = np.stack(betas_roi_ave)
        print(betas_roi_ave.shape)
        
        # Train/Test Split
        # ALLDATA
        betas_tr = []
        betas_te = []

        for idx,stim in enumerate(stims_all):
            if stim in sharedix:
                betas_te.append(betas_roi[idx,:])
            else:
                betas_tr.append(betas_roi[idx,:])

        betas_tr = np.stack(betas_tr)
        betas_te = np.stack(betas_te)    
        
        # AVERAGED DATA        
        betas_ave_tr = []
        betas_ave_te = []
        for idx,stim in enumerate(stims_unique):
            if stim in sharedix:
                betas_ave_te.append(betas_roi_ave[idx,:])
            else:
                betas_ave_tr.append(betas_roi_ave[idx,:])
        betas_ave_tr = np.stack(betas_ave_tr)
        betas_ave_te = np.stack(betas_ave_te)    
        
        # Save
        np.save(f'{savedir}/{subject}_{roi}_betas_tr.npy',betas_tr)
        np.save(f'{savedir}/{subject}_{roi}_betas_te.npy',betas_te)
        np.save(f'{savedir}/{subject}_{roi}_betas_ave_tr.npy',betas_ave_tr)
        np.save(f'{savedir}/{subject}_{roi}_betas_ave_te.npy',betas_ave_te)


if __name__ == "__main__":
    main()
