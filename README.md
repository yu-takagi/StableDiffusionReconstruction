# High-resolution image reconstruction with latent diffusion models from human brain activity
Takagi and Nishimoto, CVPR 2023

[[Paper](https://www.biorxiv.org/content/10.1101/2022.11.18.517004v3)]
[[Technical Paper](https://arxiv.org/abs/2306.11536)]
[[Project Page](https://sites.google.com/view/stablediffusion-with-brain/)]
[[FAQ](https://sites.google.com/view/stablediffusion-with-brain/faq_en)]
[[FAQ(Japanese)](https://sites.google.com/view/stablediffusion-with-brain/faq_jp)]

# General Information
This is a repository for reproducing the method we presented (Takagi and Nishimoto, CVPR 2023) for visual experience reconstruction from brain activity using Stable Diffusion.

<p align="center">
<img src=/visual_summary.jpg />
</p>

Based on our earlier work (Takagi and Nishimoto, CVPR 2023), we further examined the extent to which various additional decoding techniques affect the performance of reconstructing visual experience in a following [Technical Paper](https://arxiv.org/abs/2306.11536), including a method of decoding text prompt from the brain (see figure below). These methods are also available in this repository.
<p align="center">
<img src=/visual_summary_techpaper.jpg />
</p>

We confirmed that adding several techniques contribute to improving the accuracy from Takagi and Nishimoto CVPR 2023. In the figure below, for each method, three generated images from different stochastic noise were randomly chosen.

<p align="center">
<img src=/results_tech_paper.jpg />
</p>

# Environment setup
1. Download `nsddata`, `nsddata_betas`, and `nsddata_stimuli` from NSD and place them in the `nsd` directory. These are the required files:
  - nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session01.hdf5 (Note: You will need the `betas_sessionXX.hdf5` files for sessions 01 to 37)
  - nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5
  - nsddata/experiments/nsd/nsd_expdesign.mat
  - nsddata/freesurfer/fsaverage/label/streams.mgz.ctab
  - nsddata/ppdata/subj01/behav/responses.tsv
  - nsddata/ppdata/subj01/func1pt8mm/roi/streams.nii.gz
2. Navigate to ``codes/diffusion_sd1/stable-diffusion/`` and run the command ``conda env create -f environment.yaml``. This command will create a conda environment named ``ldm``. Activate this environment with the command ``conda activate ldm``. This environment is used for all methods except the last one (Reconstruction with Decoded Text Prompt + GAN + Decoded Depth).
3. For the last method (Reconstruction with Decoded Text Prompt + GAN + Decoded Depth), navigate to ``codes/diffusion_sd2/stablediffusion/`` and run the command `conda env create -f environment.yaml` to create a separate conda environment named ``ldm2``, then activate it.
4. Download checkpoint (``sd-v1-4.ckpt``), and place it under the ``codes/diffusion_sd1/stable-diffusion/models/ldm/stable-diffusion-v1/`` directory.
5. For incorporating GAN, download ``VGG_ILSVRC_19_layers`` and ``bvlc_reference_caffenet_generator_ILSVRC2012_Training`` from https://figshare.com/articles/dataset/brain-decoding-cookbook/21564384, and place them under the ``codes/gan/models/pytorch/`` directory.
6. For incorporating decoded depth, download checkpoint (``512-depth-ema.ckpt``), and place it under the ``codes/diffusion_sd2/stablediffusion/models/`` directory. 

# MRI Preprocessing
```
cd codes/utils/
python make_subjmri.py --subject subj01
```

# Reconstruction based on CVPR method
```
cd codes/utils/
python img2feat_sd.py --imgidx 0 73000 --gpu 0
python make_subjstim.py --featname init_latent --use_stim each --subject subj01
python make_subjstim.py --featname init_latent --use_stim ave --subject subj01
python make_subjstim.py --featname c --use_stim each --subject subj01
python make_subjstim.py --featname c --use_stim ave --subject subj01
python ridge.py --target c --roi ventral --subject subj01
python ridge.py --target init_latent --roi early --subject subj01

cd codes/diffusion_sd1/
python diffusion_decoding.py --imgidx 0 --gpu 1 --subject subj01 --method cvpr
```

# Reconstruction with Decoded Text Prompt
```
cd codes/caption/BLIP/
python img2feat_blip.py --gpu 0

cd codes/utils/
python make_subjstim.py --featname blip --use_stim ave --subject subj01
python make_subjstim.py --featname blip --use_stim each --subject subj01
python ridge.py --target blip --roi early ventral midventral midlateral lateral parietal  --subject subj01

cd codes/caption/BLIP/
python decode_captions.py --subject subj01

cd codes/diffusion_sd1/
python diffusion_decoding.py --imgidx 0 --gpu 1 --subject subj01 --method text
```

# Reconstruction with Decoded Text Prompt + GAN
```
cd codes/gan/bdpy/
python setup.py install

cd codes/gan/
python make_vgg19bdpy.py --imgidx 0 73000

(run the following code from conv1_1 to fc8.)
python make_subjstim_vgg19.py --layer conv1_1 --subject subj01
python ridge.py --target conv1_1 --roi early ventral midventral midlateral lateral parietal --subject subj01

python make_vgg19fromdecode.py --subject subj01
python recon_icnn_image_vgg19_dgn_relu7gen_gd.py

cd codes/diffusion_sd1/
python diffusion_decoding.py --imgidx 0 --gpu 0 --subject subj01 --method gan
```

# Reconstruction with Decoded Text Prompt + GAN + Decoded Depth
Need to ``pip install -U transformers``. Note that this update may cause BLIP to stop working. It is recommended to do this in a different environment.

```
cd codes/depth/
python img2feat_dpt.py --imgidx 0 73000 --gpu 0

cd codes/utils/
(run the following code from dpt_emb0 to dpt_emb3.)
python make_subjstim.py --featname dpt_emb0 --use_stim ave --subject subj01
python make_subjstim.py --featname dpt_emb0 --use_stim each --subject subj01
python ridge.py --target dpt_emb0 --roi early ventral midventral midlateral lateral parietal --subject subj01
python ridge.py --target dpt_emb0 --roi early ventral midventral midlateral lateral parietal --subject subj01

cd codes/depth/
python dptemb2dpt.py --gpu 0 --subject subj01

cd codes/diffusion_sd2/
python diffusion_decoding.py --imgidxs 0 1 --gpu 0 --subject subj01
```

# Evaluation
```
cd codes/utils/
python img2feat_decoded.py --gpu 0 --subject subj01 --method cvpr
python identification.py --usefeat inception --subject subj01 --method cvpr
```

# Citation
Original paper.
```
@inproceedings{takagi2023high,
  title={High-resolution image reconstruction with latent diffusion models from human brain activity},
  author={Takagi, Yu and Nishimoto, Shinji},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14453--14463},
  year={2023}
}
```

Following technical report.
```
@misc{takagi2023improving,
      title={Improving visual image reconstruction from human brain activity using latent diffusion models via multiple decoded inputs}, 
      author={Takagi, Yu and Nishimoto, Shinji},
      year={2023},
      eprint={2306.11536},
      archivePrefix={arXiv},
      primaryClass={q-bio.NC}
}
```

# Acknowledgement
Our codebase builds on these repositories. We would like to thank the authors. 
 
> https://github.com/CompVis/stable-diffusion

> https://github.com/Stability-AI/stablediffusion

> https://github.com/gallantlab/himalaya

> https://github.com/tknapen/nsd_access

> https://github.com/salesforce/BLIP

> https://github.com/KamitaniLab/bdpy

> https://github.com/KamitaniLab/brain-decoding-cookbook-public

> https://github.com/isl-org/DPT

# Contact
Plase [email](takagi.yuu.fbs@osaka-u.ac.jp) if you have any questions.
