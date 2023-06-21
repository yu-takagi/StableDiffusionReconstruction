import numpy as np
import scipy.io
from tqdm import tqdm
import argparse
import os

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="layer of VGG19",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )

    opt = parser.parse_args()
    subject=opt.subject
    layer = opt.layer
    datdir = '../../nsdfeat/vgg19_features/'
    savedir = f'../../nsdfeat//subjfeat/'
    nsd_expdesign = scipy.io.loadmat('../../nsd/nsddata/experiments/nsd/nsd_expdesign.mat')

    # Note that most of them are 1-base index!
    # This is why I subtract 1
    sharedix = nsd_expdesign['sharedix'] -1 
    stims = np.load(f'../../mrifeat/{subject}/{subject}_stims_ave.npy')
    feats = []
    tr_idx = np.zeros(len(stims))

    for idx, s in tqdm(enumerate(stims)): 
        if s in sharedix:
            tr_idx[idx] = 0
        else:
            tr_idx[idx] = 1    
        feat = scipy.io.loadmat(f'{datdir}/{layer}/nsd/org/VGG19-{layer}-nsd-org-{s:06}.mat')
        feats.append(feat['feat'].flatten())

    feats = np.stack(feats)    

    os.makedirs(savedir, exist_ok=True)

    feats_tr = feats[tr_idx==1,:]
    feats_te = feats[tr_idx==0,:]

    np.save(f'{savedir}/{subject}_{layer}_tr.npy',feats_tr)
    np.save(f'{savedir}/{subject}_{layer}_te.npy',feats_te)


if __name__ == "__main__":
    main()
