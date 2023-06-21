import argparse, os
import numpy as np
from tqdm import tqdm
import torch
from scipy.io import savemat
import torch
import numpy as np

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
    roinames = ['early','ventral','midventral','midlateral','lateral','parietal']

    maps = {
            'conv1_1': [64,224,224],
            'conv1_2': [64,224,224],
            'conv2_1': [128,112,112],
            'conv2_2': [128,112,112],
            'conv3_1': [256,56,56],
            'conv3_2': [256,56,56],
            'conv3_3': [256,56,56],
            'conv3_4': [256,56,56],
            'conv4_1': [512,28,28],
            'conv4_2': [512,28,28],
            'conv4_3': [512,28,28],
            'conv4_4': [512,28,28],
            'conv5_1': [512,14,14],
            'conv5_2': [512,14,14],
            'conv5_3': [512,14,14],
            'conv5_4': [512,14,14],
            'fc6':     [1,4096],
            'fc7':     [1,4096],
            'fc8':     [1,1000],
        }
    datdir = f'../../decoded/{subject}/'
    savedir = f'../../decoded/gan_mod/'
    os.makedirs(savedir, exist_ok=True)

    for layer in tqdm(maps.keys()):
        print(f'Now Layer: {layer}')
        os.makedirs(f'{savedir}/{layer}/{subject}/streams/', exist_ok=True)
        feat = np.load(f'{datdir}/{subject}_{"_".join(roinames)}_scores_{layer}.npy')
        for i in range(feat.shape[0]):
            cfeat = feat[i,:].reshape(maps[layer])[np.newaxis]
            mdic = {"feat":cfeat}
            savemat(f'{savedir}/{layer}/{subject}/streams/VGG19-{layer}-{subject}-streams-{i:06}.mat', mdic)

if __name__ == "__main__":
    main()
