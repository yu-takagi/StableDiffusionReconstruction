import argparse, os
import torch
import numpy as np
from tqdm import tqdm
import torch
from scipy.io import savemat
from bdpy.dl.torch.models import VGG19
from bdpy.dl.torch.torch import FeatureExtractor, ImageDatasetNSD
import torch
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imgidx",
        required=True,
        nargs="*",
        type=int,
        help="start and end imgs"
    )
    opt = parser.parse_args()
    imgidx = opt.imgidx
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_param_file = './models/pytorch/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.pt'
    image_mean_file = './models/pytorch/VGG_ILSVRC_19_layers/ilsvrc_2012_mean.npy'
    image_mean = np.load(image_mean_file)
    image_mean = np.float32([image_mean[0].mean(), image_mean[1].mean(), image_mean[2].mean()])
    encoder = VGG19()
    encoder.to(device)
    encoder.load_state_dict(torch.load(encoder_param_file))
    encoder.eval()
    maps = {
            'conv1_1': 'features[0]',
            'conv1_2': 'features[2]',
            'conv2_1': 'features[5]',
            'conv2_2': 'features[7]',
            'conv3_1': 'features[10]',
            'conv3_2': 'features[12]',
            'conv3_3': 'features[14]',
            'conv3_4': 'features[16]',
            'conv4_1': 'features[19]',
            'conv4_2': 'features[21]',
            'conv4_3': 'features[23]',
            'conv4_4': 'features[25]',
            'conv5_1': 'features[28]',
            'conv5_2': 'features[30]',
            'conv5_3': 'features[32]',
            'conv5_4': 'features[34]',
            'fc6':     'classifier[0]',
            'fc7':     'classifier[3]',
            'fc8':     'classifier[6]',
        }

    fext = FeatureExtractor(encoder,layers=list(maps.values()))

    outdir = f'../../nsdfeat/vgg19_features/'
    os.makedirs(outdir, exist_ok=True)
    imgdata = ImageDatasetNSD(resize=(224,224),shape='chw', transform=None, scale=255, rgb_mean=image_mean, nsd_dir='../../nsd/')

    for s in tqdm(range(imgidx[0],imgidx[1])):
        print(f"Now processing image {s:06}")
        with torch.no_grad():
            feat = fext(imgdata[s].unsqueeze(0))
            
        for key, layer in maps.items():
            cfeat = feat[layer].copy().squeeze()
            mdic = {"feat":cfeat}
            os.makedirs(f'{outdir}/{key}/nsd/org/', exist_ok=True)
            savemat(f'{outdir}/{key}/nsd/org/VGG19-{key}-nsd-org-{s:06}.mat', mdic)

if __name__ == "__main__":
    main()
