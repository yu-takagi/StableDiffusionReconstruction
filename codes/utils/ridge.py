import argparse, os
import numpy as np
from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV
from himalaya.scoring import correlation_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        type=str,
        default='',
        help="Target variable",
    )
    parser.add_argument(
        "--roi",
        required=True,
        type=str,
        nargs="*",
        help="use roi name",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )

    opt = parser.parse_args()
    target = opt.target
    roi = opt.roi

    backend = set_backend("numpy", on_error="warn")
    subject=opt.subject

    if target == 'c' or target == 'init_latent': # CVPR
        alpha = [0.000001,0.00001,0.0001,0.001,0.01, 0.1, 1]
    else: # text / GAN / depth decoding (with much larger number of voxels)
        alpha = [10000, 20000, 40000]

    ridge = RidgeCV(alphas=alpha)

    preprocess_pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
    )
    pipeline = make_pipeline(
        preprocess_pipeline,
        ridge,
    )    
    mridir = f'../../mrifeat/{subject}/'
    featdir = '../../nsdfeat/subjfeat/'
    savedir = f'../..//decoded/{subject}/'
    os.makedirs(savedir, exist_ok=True)

    X = []
    X_te = []
    for croi in roi:
        if 'conv' in target: # We use averaged features for GAN due to large number of dimension of features
            cX = np.load(f'{mridir}/{subject}_{croi}_betas_ave_tr.npy').astype("float32")
        else:
            cX = np.load(f'{mridir}/{subject}_{croi}_betas_tr.npy').astype("float32")
        cX_te = np.load(f'{mridir}/{subject}_{croi}_betas_ave_te.npy').astype("float32")
        X.append(cX)
        X_te.append(cX_te)
    X = np.hstack(X)
    X_te = np.hstack(X_te)
    
    Y = np.load(f'{featdir}/{subject}_each_{target}_tr.npy').astype("float32").reshape([X.shape[0],-1])
    Y_te = np.load(f'{featdir}/{subject}_ave_{target}_te.npy').astype("float32").reshape([X_te.shape[0],-1])
    
    print(f'Now making decoding model for... {subject}:  {roi}, {target}')
    print(f'X {X.shape}, Y {Y.shape}, X_te {X_te.shape}, Y_te {Y_te.shape}')
    pipeline.fit(X, Y)
    scores = pipeline.predict(X_te)
    rs = correlation_score(Y_te.T,scores.T)
    print(f'Prediction accuracy is: {np.mean(rs):3.3}')

    np.save(f'{savedir}/{subject}_{"_".join(roi)}_scores_{target}.npy',scores)

if __name__ == "__main__":
    main()
