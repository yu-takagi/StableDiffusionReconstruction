import argparse, os
from PIL import Image
import torch
from torchvision import transforms
from models.blip import blip_decoder
import sys
sys.path.append("../../util/")
from nsd_access.nsda import NSDAccess
from tqdm import tqdm
import numpy as np

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpu",
        required=True,
        type=int,
        help="gpu"
    )

    # Set parameters
    opt = parser.parse_args()
    gpu = opt.gpu
    torch.cuda.set_device(gpu)
    nimage = 73000
    image_size = 240
    model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth"
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = blip_decoder(pretrained=model_url, image_size=image_size, vit="base")
    model.eval()
    model = model.to(device)
    savedir = f'../../../nsdfeat/blip/'
    os.makedirs(savedir, exist_ok=True)

    # Make feature
    nsda = NSDAccess('../../../nsd/')
    for s in tqdm(range(nimage)):
        img_arr = nsda.read_images(s)
        image = Image.fromarray(img_arr).convert("RGB").resize((image_size,image_size), resample=Image.LANCZOS)
        img_arr = transforms.ToTensor()(image).to('cuda').unsqueeze(0)
        with torch.no_grad():
            vit_feat = model.visual_encoder(img_arr).cpu().detach().numpy().squeeze()        
        np.save(f'{savedir}/{s:06}.npy',vit_feat)

if __name__ == "__main__":
    main()
