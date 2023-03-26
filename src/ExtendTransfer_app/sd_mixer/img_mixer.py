import logging
import progressbar as pb
import numpy as np
import imageio

from celery import current_task
from tqdm import tqdm
from PIL import Image
from ExtendTransfer_app.cstm_clases import Fdout
# from ExtendTransfer_app.file_handler import handle_uploaded_file

from .ldm.models.diffusion.ddim import DDIMSampler
from .ldm.models.diffusion.plms import PLMSSampler
from .ldm.extras import load_model_from_config, load_training_dir
from .params import MixerParams

import clip
from einops import rearrange
from torch import autocast
import torch

from huggingface_hub import hf_hub_download


class ImageMixer:
    def __init__(self):
        self.device = "cuda"
        self.ckpt = hf_hub_download(
            repo_id="lambdalabs/image-mixer", filename="image-mixer-pruned.ckpt")
        self.config = hf_hub_download(
            repo_id="lambdalabs/image-mixer", filename="image-mixer-config.yaml")
        self.model: torch.nn.Module = None
        self.clip_model: torch.nn.Module
        self.preprocess = None

    def _prepare(self):
        torch.cuda.empty_cache()
        self.model = load_model_from_config(
            self.config, self.ckpt, device=self.device, verbose=False)
        self.clip_model, self.preprocess = clip.load(
            "ViT-L/14", device=self.device)

    def clean(self):
        self.model = None
        self.clip_model, self.preprocess = None, None
        torch.cuda.empty_cache()

    def _create_pbar(self, max_iter):
        self.grad_iter = 0
        logging.info(current_task)
        self.pbar = pb.ProgressBar(term_width=0, fd=Fdout(tsk=current_task))
        self.pbar.widgets = [pb.Percentage()]
        self.pbar.maxval = max_iter

    @torch.no_grad()
    def get_im_c(self, im_path, clip_model):
        im = Image.open(im_path).convert("RGB")
        # prompts = self.preprocess(im_path).to(self.device).unsqueeze(0)
        prompts = self.preprocess(im).to(self.device).unsqueeze(0)
        return clip_model.encode_image(prompts).float()
    
    @staticmethod
    def to_im_list(x_samples_ddim):
        x_samples_ddim = torch.clamp(
            (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        ims = []
        for x_sample in x_samples_ddim:
            x_sample = 255. * \
                rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            ims.append(Image.fromarray(x_sample.astype(np.uint8)))
        return ims

    @torch.no_grad()
    def sample(self, sampler, model, c, uc, scale, start_code, h=512, w=512, precision="autocast", ddim_steps=50):
        ddim_eta = 0.0
        precision_scope = autocast if precision == "autocast" else nullcontext
        with precision_scope("cuda"):
            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=c,
                                             batch_size=c.shape[0],
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=start_code)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
        return self.to_im_list(x_samples_ddim)

    def process(self, params: MixerParams):        
        if params.device == "CPU":
            self.device = 'cpu'
        elif params.device == "GPU":
            self.device = 'cuda'

        if not self.model:
            self._prepare()

        sampler = DDIMSampler(
            self.model) if params.sampler == "DDIM" else PLMSSampler(self.model)

        torch.manual_seed(params.seed)
        start_code = torch.randn(
            params.n_samples, 4, params.h//8, params.w//8, device=self.device)
        conds = []

        for strength, img in params.imgs:
            cond = strength * self.get_im_c(img, self.clip_model)
            conds.append(cond)

        conds = torch.cat(conds, dim=0).unsqueeze(0)
        conds = conds.tile(params.n_samples, 1, 1)
        ims = self.sample(sampler, self.model, conds, 0*conds,
                          params.scale, start_code, ddim_steps=params.steps)
        
        from django.conf import settings
        imageio.imwrite(settings.MEDIA_ROOT + "/images/mixer/results/" + params.id  + ".jpg", ims[0])
        print('SAVED!')