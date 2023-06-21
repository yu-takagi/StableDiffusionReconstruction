import argparse, os, sys, glob
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import torchvision
from torchvision import transforms
from tqdm import tqdm

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
    parser.add_argument(
        "--method",
        required=True,
        type=str,
        help="cvpr or text or gan or depth",
    )

    # Parameters
    opt = parser.parse_args()
    subject=opt.subject
    gpu = opt.gpu
    method = opt.method
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    imglist = sorted(glob.glob(f'../../decoded/image-{method}/{subject}/samples/*'))
    outdir = f'../../identification/{method}/{subject}/'
    os.makedirs(outdir, exist_ok=True)

    # Load Models 
    # Inception V3
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model_inception = torchvision.models.inception_v3(pretrained=True)
    model_inception.eval()
    model_inception.to(device)
    model_inception = torchvision.models.feature_extraction.create_feature_extractor(model_inception, {'flatten':'flatten'})

    # AlexNet
    model_alexnet = torchvision.models.alexnet(pretrained=True)
    model_alexnet.eval()
    model_alexnet.to(device)
    model_alexnet = torchvision.models.feature_extraction.create_feature_extractor(model_alexnet,{'features.5':'features.5',
                                                                                                  'features.12':'features.12',
                                                                                                  'classifier.5':'classifier.5'})

    # CLIP
    model_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    model_clip.to(device)
    processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    print(f"Now processing start for : {method}")
    for img in tqdm(imglist):
        imgname = img.split('/')[-1].split('.')[0]
        print(img)
        image = Image.open(img)

        # Inception
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        input_batch = input_batch.to(device)
        with torch.no_grad():
            feat = model_inception(input_batch)
        feat_inception = feat['flatten'].cpu().detach().numpy().copy()    

        # AlexNet
        with torch.no_grad():
            feat = model_alexnet(input_batch)
        feat_alexnet5 = feat['features.5'].flatten().cpu().detach().numpy().copy()    
        feat_alexnet12 = feat['features.12'].flatten().cpu().detach().numpy().copy()    
        feat_alexnet18 = feat['classifier.5'].flatten().cpu().detach().numpy().copy()    

        # CLIP
        inputs = processor_clip(text="",images=image, return_tensors="pt").to(device)
        outputs = model_clip(**inputs,output_hidden_states=True)
        feat_clip = outputs.image_embeds.cpu().detach().numpy().copy()
        feat_clip_h6 = outputs.vision_model_output.hidden_states[6].flatten().cpu().detach().numpy().copy()
        feat_clip_h12 = outputs.vision_model_output.hidden_states[12].flatten().cpu().detach().numpy().copy()

        # SAVE
        fname = f'{outdir}/{imgname}'
        np.save(f'{fname}_inception.npy',feat_inception)
        np.save(f'{fname}_alexnet5.npy',feat_alexnet5)
        np.save(f'{fname}_alexnet12.npy',feat_alexnet12)
        np.save(f'{fname}_alexnet18.npy',feat_alexnet18)
        np.save(f'{fname}_clip.npy',feat_clip)
        np.save(f'{fname}_clip_h6.npy',feat_clip_h6)
        np.save(f'{fname}_clip_h12.npy',feat_clip_h12)

if __name__ == "__main__":
    main()
