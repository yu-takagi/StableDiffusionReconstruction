import torch
from models.blip import blip_decoder
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse, os

def generate_from_imageembeds(model, device, image_embeds, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
    if not sample:
        image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)

    image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(device)
    model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}

    prompt = [model.prompt]
    input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(device) 
    input_ids[:,0] = model.tokenizer.bos_token_id
    input_ids = input_ids[:, :-1] 

    if sample:
        #nucleus sampling
        outputs = model.text_decoder.generate(input_ids=input_ids,
                                              max_length=max_length,
                                              min_length=min_length,
                                              do_sample=True,
                                              top_p=top_p,
                                              num_return_sequences=1,
                                              eos_token_id=model.tokenizer.sep_token_id,
                                              pad_token_id=model.tokenizer.pad_token_id, 
                                              repetition_penalty=1.1,                                            
                                              **model_kwargs)
    else:
        #beam search
        outputs = model.text_decoder.generate(input_ids=input_ids,
                                              max_length=max_length,
                                              min_length=min_length,
                                              num_beams=num_beams,
                                              eos_token_id=model.tokenizer.sep_token_id,
                                              pad_token_id=model.tokenizer.pad_token_id,     
                                              repetition_penalty=repetition_penalty,
                                              **model_kwargs)            

    captions = []    
    for output in outputs:
        caption = model.tokenizer.decode(output, skip_special_tokens=True)    
        captions.append(caption[len(model.prompt):])
    return captions

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )

    # Set parameters
    gpu = 0
    opt = parser.parse_args()
    subject = opt.subject
    norepeat = 1.5
    torch.cuda.set_device(gpu)
    image_size = 240
    savedir = f'../../../decoded/{subject}/captions/'
    os.makedirs(savedir, exist_ok=True)

    # Model Load
    model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth"
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model_decoder = blip_decoder(pretrained=model_url, image_size=image_size, vit="base")
    model_decoder.eval()
    model_decoder = model_decoder.to(device)

    # Load decoding results
    roi = ['early', 'ventral', 'midventral', 'midlateral', 'lateral', 'parietal']
    scores = np.load(f'../../../decoded/{subject}/{subject}_{"_".join(roi)}_scores_blip.npy')

    # Caption generation
    captions_brain = []
    for imidx in tqdm(range(982)):
        scores_test = torch.Tensor(scores[imidx,:].reshape(-1,768)).to(device).unsqueeze(0)
        print(scores_test.shape)
        caption = generate_from_imageembeds(model_decoder, device, scores_test,num_beams=3, max_length=20, min_length=5, repetition_penalty=norepeat)
        captions_brain.append(caption)
        print(f'{imidx:04}:  {caption}')

    df = pd.DataFrame(captions_brain)
    df.to_csv(f'{savedir}/captions_brain.csv', sep='\t', header=False, index=False)

if __name__ == "__main__":
    main()
