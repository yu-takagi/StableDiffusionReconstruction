import argparse, os
from tqdm import tqdm, trange
from torch import autocast
from contextlib import nullcontext
import torch
import numpy as np
import pandas as pd
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat, rearrange
from pytorch_lightning import seed_everything
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.data.util import AddMiDaS

def initialize_model(config, ckpt,device):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)
    model = model.to(device)
    sampler = DDIMSampler(model)
    return sampler

def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded


def make_batch_sd(
        image,
        txt,
        device,
        num_samples=1,
        model_type="dpt_hybrid"
):
    image = np.array(image.convert("RGB"))
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    # sample['jpg'] is tensor hwc in [-1, 1] at this point
    midas_trafo = AddMiDaS(model_type=model_type)
    batch = {
        "jpg": image,
        "txt": txt,
    }
    batch = midas_trafo(batch)
    batch["jpg"] = rearrange(batch["jpg"], 'h w c -> 1 c h w')
    batch["jpg"] = repeat(batch["jpg"].to(device=device),
                          "1 ... -> n ...", n=num_samples)
    batch["midas_in"] = repeat(torch.from_numpy(batch["midas_in"][None, ...]).to(
        device=device), "1 ... -> n ...", n=num_samples)
    return batch

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--imgidxs",
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
    parser.add_argument(
        "--subject",
        required=True,
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )

    # Set parameters
    opt = parser.parse_args()
    seed_everything(opt.seed)
    imgidxs = opt.imgidxs
    gpu = opt.gpu
    torch.cuda.set_device(gpu)
    subject=opt.subject
    config = './stablediffusion/configs/stable-diffusion/v2-midas-inference.yaml'
    ckpt = './stablediffusion/models/512-depth-ema.ckpt'
    steps = 50
    scale = 5.0
    eta = 0.0
    strength = 0.8
    num_samples= 1
    callback=None
    n_iter = 5

    # Save Directories
    outdir = f'../../decoded/image-depth-new/{subject}/'
    os.makedirs(outdir, exist_ok=True)
    sample_path = os.path.join(outdir, f"samples")
    os.makedirs(sample_path, exist_ok=True)

    precision = 'autocast'
    precision_scope = autocast if precision == "autocast" else nullcontext

    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    sampler = initialize_model(config, ckpt,device)
    model = sampler.model

    sampler.make_schedule(ddim_num_steps=steps, ddim_eta=eta, verbose=True)

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    
    t_enc = min(int(strength * steps), steps-1)
    print(f"target t_enc is {t_enc} steps")

    # Load Prediction (C, InitLatent, Depth(cc))
    captdir = f'../../decoded/{subject}/captions/'
    dptdir = f'../../decoded/{subject}/dpt_fromemb/'
    gandir = f'../../decoded/gan_recon_img/all_layers/{subject}/streams/'

    # C
    captions = pd.read_csv(f'{captdir}/captions_brain.csv', sep='\t',header=None)

    for imgidx in tqdm(range(imgidxs[0],imgidxs[1])):    
        prompt = [captions.iloc[imgidx][0]]
        cc = torch.Tensor(np.load(f'{dptdir}/{imgidx:06}.npy')).to('cuda')

        # Generate image from Text + GAN + Depth
        shenpath = f'{gandir}/recon_image_normalized-VGG19-fc8-{subject}-streams-{imgidx:06}.tiff'
        init_image = Image.open(shenpath).resize((512,512))
        image = pad_image(init_image)
        base_count = 0
        with torch.no_grad():
            for n in trange(n_iter, desc="Sampling"):
                torch.autocast("cuda")
                batch = make_batch_sd(image, txt=prompt, device=device, num_samples=num_samples)
                z = model.get_first_stage_encoding(model.encode_first_stage(
                    batch[model.first_stage_key]))  # move to latent space
                c = model.cond_stage_model.encode(batch["txt"]).mean(axis=0).unsqueeze(0)
                c_cat = list()
                c_cat.append(cc)
                c_cat = torch.cat(c_cat, dim=1)

                # cond
                cond = {"c_concat": [c_cat], "c_crossattn": [c]}

                # uncond cond
                uc_cross = model.get_unconditional_conditioning(num_samples, "")
                uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

                # encode (scaled latent)
                z_enc = sampler.stochastic_encode(
                    z, torch.tensor([t_enc] * num_samples).to(model.device))

                # decode it
                samples = sampler.decode(z_enc, cond, t_enc, unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc_full, callback=callback)
                x_samples_ddim = model.decode_first_stage(samples)
                result = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
                Image.fromarray(result[0,:,:,:].astype(np.uint8)).save(
                    os.path.join(sample_path, f"{imgidx:05}_{base_count:03}.png"))   
                base_count += 1


if __name__ == "__main__":
    main()
