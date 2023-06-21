import argparse, os
from tqdm import tqdm
import torch
import numpy as np
from transformers import DPTForDepthEstimation
from PIL import Image

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpu",
        required=True,
        type=int,
        help="gpu"
    )
    parser.add_argument(
        "--subject",
        required=True,
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )

    opt = parser.parse_args()
    gpu = opt.gpu
    subject = opt.subject
    roi = ['early', 'ventral', 'midventral', 'midlateral', 'lateral', 'parietal']

    datdir = f'../../decoded/{subject}/'
    savedir = f'../../decoded/{subject}/dpt_fromemb/'
    os.makedirs(f'{savedir}', exist_ok=True)
    savedir_img = f'../../decoded/{subject}/dpt_fromemb_image/'
    os.makedirs(f'{savedir_img}', exist_ok=True)

    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    model.to(device)
    imsize = (512,512)
    latentsize = (64,64)

    dpt_embs = [] 
    for idx in range(4):
        fname = f'{datdir}/{subject}_{"_".join(roi)}_scores_dpt_emb{idx}.npy'
        dpt_embs.append(np.load(fname))
    dpt_embs = np.stack(dpt_embs)
    dpt_embs = torch.Tensor(dpt_embs).to(device)

    for s in tqdm(range(dpt_embs.shape[1])):
        hidden_states = [dpt_embs[idx,s,:].reshape(1,577,1024) for idx in range(4)]
        with torch.no_grad():
            hidden_states = model.neck(hidden_states)
            predicted_depth = model.head(hidden_states)

        # Make depth Image for visual inspection
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=imsize,
            mode="bicubic",
            align_corners=False,
        )

        # visualize the prediction
        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        depth = Image.fromarray(formatted)

        # Make latent reps for SD2
        cc = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=latentsize,
            mode="bicubic",
            align_corners=False,
        )
        depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3],
                                                                                        keepdim=True)
        cc = 2. * (cc - depth_min) / (depth_max - depth_min) - 1.

        np.save(f'{savedir}/{s:06}.npy',cc.to('cpu').detach().numpy())
        depth.save(f'{savedir_img}/{s:06}.png')

if __name__ == "__main__":
    main()
