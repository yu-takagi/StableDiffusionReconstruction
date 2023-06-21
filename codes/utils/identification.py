import argparse
import numpy as np
from tqdm import tqdm

def load_feat_org(imgid, subject, method, usefeat):
    featdir = f'../../identification/{method}/{subject}/'
    feat = np.load(f'{featdir}/{imgid:05}_org_{usefeat}.npy').flatten().squeeze()
    return feat

def load_feat_gen(imgid, subject, method, usefeat):
    featdir = f'../../identification/{method}/{subject}'
    nrep = 5

    feats_gen = []
    for rep in range(nrep):
        feat_gen = np.load(f'{featdir}/{imgid:05}_{rep:03}_{usefeat}.npy').flatten().squeeze()
        feats_gen.append(feat_gen)

    return feats_gen

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--usefeat",
        required=True,
        type=str,
        help="feature for calculating PSM"
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )
    parser.add_argument(
        "--method",
        required=True,
        type=str,
        help="cvpr or text or gan or depth",
    )

    opt = parser.parse_args()
    usefeat = opt.usefeat
    subject = opt.subject
    method = opt.method
    nimage = 982

    # Load all images
    print("Now Loading all images......")
    feat_orgs = []
    feat_gens = []
    for imgid in tqdm(range(nimage)):
        feat_org = load_feat_org(imgid, subject, method, usefeat)
        feat_orgs.append(feat_org)

        feat_gen = load_feat_gen(imgid, subject, method, usefeat)
        feat_gens.append(feat_gen)

    # Calculate similarity
    print("Now Calculating similarity......")
    rs_all = []
    for row in tqdm(range(nimage)):
        rs_row = []
        for col in range(nimage):
            feat_org = feat_orgs[row]  
            feat_gen = feat_gens[col]  
            r = np.corrcoef(feat_org,feat_gen)[0,1:]
            rs_row.append(r)
        rs_all.append(rs_row)

    # Calculate accuracy
    print("Now Calculating identification accuracy......")
    acc_all = []
    for imgid_org in tqdm(range(nimage)):
        # Calculate true R
        r_true = rs_all[imgid_org][imgid_org]

        # Calculate Fake R
        acc = []
        fakeimgs = list(range(nimage))
        fakeimgs.remove(imgid_org)
        for imgid_fake in fakeimgs:
            r_fake = rs_all[imgid_org][imgid_fake]
            acc.append(r_true.mean() > r_fake.mean())
        
        acc_all.append(sum(acc)/len(acc))
    acc_all = np.array(acc_all)

    print(f'{subject}_{usefeat}:\t ACC = {np.mean(acc_all):.03} ')

if __name__ == "__main__":
    main()
