import argparse, os
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import repeat
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
from nsd_access import NSDAccess
from PIL import Image
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def load_model_from_config(config, ckpt, gpu, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.cuda(f"cuda:{gpu}")
    model.eval()
    return model

def load_img_from_arr(img_arr,resolution):
    image = Image.fromarray(img_arr).convert("RGB")
    w, h = resolution, resolution
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--imgidx",
        required=True,
        nargs="*",
        type=int,
        help="start and end imgs"
    )
    parser.add_argument(
        "--gpu",
        required=True,
        type=int,
        help="gpu"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )

    # Set Parameters
    opt = parser.parse_args()
    seed_everything(opt.seed)
    imgidx = opt.imgidx
    gpu = opt.gpu
    resolution = 320
    batch_size = 1
    ddim_steps = 50
    ddim_eta = 0.0
    strength = 0.8
    scale = 5.0
    nsda = NSDAccess('../../nsd/')
    config = '../diffusion_sd1/stable-diffusion/configs/stable-diffusion/v1-inference.yaml'
    ckpt = '../diffusion_sd1/stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt'
    config = OmegaConf.load(f"{config}")
    torch.cuda.set_device(gpu)
    os.makedirs(f'../../nsdfeat/init_latent/', exist_ok=True)
    os.makedirs(f'../../nsdfeat/c/', exist_ok=True)

    # Load moodels
    precision = 'autocast'
    precision_scope = autocast if precision == "autocast" else nullcontext
    model = load_model_from_config(config, f"{ckpt}", gpu)
    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    # Sample
    for s in tqdm(range(imgidx[0],imgidx[1])):
        print(f"Now processing image {s:06}")
        prompt = []
        prompts = nsda.read_image_coco_info([s],info_type='captions')
        for p in prompts:
            prompt.append(p['caption'])    
        
        img = nsda.read_images(s)
        init_image = load_img_from_arr(img,resolution).to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    uc = model.get_learned_conditioning(batch_size * [""])
                    c = model.get_learned_conditioning(prompt).mean(axis=0).unsqueeze(0)

                    # encode (scaled latent)
                    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                    # decode it
                    samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc,)
                    
        init_latent = init_latent.cpu().detach().numpy().flatten()
        c = c.cpu().detach().numpy().flatten()
        np.save(f'../../nsdfeat/init_latent/{s:06}.npy',init_latent)
        np.save(f'../../nsdfeat/c/{s:06}.npy',c)


if __name__ == "__main__":
    main()
