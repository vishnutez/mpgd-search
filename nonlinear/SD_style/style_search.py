import argparse, os, sys, glob
import cv2
import json
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from torch import autocast
from contextlib import contextmanager, nullcontext
import csv  # Add CSV import
from datetime import datetime  # Add datetime import

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim_search import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from ldm.models.diffusion.clip.base_clip import CLIPEncoder
from tqdm import tqdm

from transformers import logging
logging.set_verbosity_error()


# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
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

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def seed_everything(seed):
    import random
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy random module
    torch.manual_seed(seed)  # PyTorch (CPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch (CUDA)
        torch.cuda.manual_seed_all(seed)  # All GPUs (if using multi-GPU)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    # torch.backends.cudnn.benchmark = False  # Disable benchmark mode for reproducibility
    # Set seed for PyTorch Lightning
    try:
        from pytorch_lightning import seed_everything as pl_seed_everything
        pl_seed_everything(seed)
        print(f"Seed set to {seed} for PyTorch Lightning.")
    except ImportError:
        pass

import yaml
def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a cat wearing glasses",
        help="the prompt to render"
    )
    parser.add_argument(
        "--style_ref_path",
        type=str,
        nargs="?",
        default="./style_images/",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs_search/"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=1.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/sd-v1-4.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--tt",
        type=int,
        default=1,
        help="time travel cycle number",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=1,
        help="time travel cycle number",
    )

    # arugments for search_algo and reward_eval
    parser.add_argument(
        "--search_algo_config",
        type=str,
        default='configs/search_resample.yaml',
    )
    parser.add_argument(
        "--reward_eval_config",
        type=str,
        default='configs/reward_style.yaml',
    )
    parser.add_argument(
        "--eval_fn_list",
        type=str,
        nargs="+",
        default=["style_loss", "clip_score"],
        help="list of eval fn names",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="batch size for each eval fn",
    )
    parser.add_argument(
        "--num_particles",
        type=int,
        default=4,
    )
   
    

    opt = parser.parse_args()

    search_algo_config = load_yaml(opt.search_algo_config)
    reward_eval_config = load_yaml(opt.reward_eval_config)

    from reward_eval import get_reward_eval
    from search_algo import get_search_algo

    reward_eval = get_reward_eval(**reward_eval_config)
    search_algo = get_search_algo(**search_algo_config)

    search_algo_config['num_particles'] = opt.num_particles
    search_algo_name = search_algo_config['name']

    from st_eval import get_eval_fn, Evaluator

    # get evaluator
    eval_fn_list = []
    for eval_fn_name in opt.eval_fn_list:
        eval_fn_list.append(get_eval_fn(eval_fn_name))
    evaluator = Evaluator(eval_fn_list)

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print('Device:', device)

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    
    model = model.to(device)

    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)
    elif opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    # opt.n_samples = 2 # current version only supprt batchsize 1
    batch_size = opt.batch_size
    # Get current timestamp
    timestamp = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    # sample_path = os.path.join(outpath, f"{timestamp}_ddim{opt.ddim_steps}_tt{opt.tt}_rho{opt.rho}")

    sample_path = os.path.join(outpath, f"{timestamp}_{search_algo_name}")
    metrics_path = os.path.join(sample_path, 'metrics')
    
    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([batch_size, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
        
    image_encoder = CLIPEncoder().cuda()

    file_id = 0

    markdown_table = ''

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with precision_scope("cuda") and model.ema_scope():
        for j in range(opt.n_iter):
            tic = time.time()
            for filename in tqdm(sorted(os.listdir(opt.style_ref_path))):
                print('*' * 20)
                seed_everything(opt.seed)  # TODO: remove this line for different seeds in each generation
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    style_ref_img_path = os.path.join(opt.style_ref_path, filename)
                    image_encoder.calc_ref_feat(style_ref_img_path)
                    prompts = batch_size * [opt.prompt]
                    ref_style_img = Image.open(style_ref_img_path).convert("RGB")

                    from torchvision import transforms
                    transforms = transforms.Compose([
                        transforms.Resize((opt.H, opt.W), interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5])
                    ])
                    ref = transforms(ref_style_img).to(device)  # convert ref style img to tensor
                    ref = ref.unsqueeze(0)  # add batch dimension

                    print('ref style img:', ref.shape)

                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, intermediates = sampler.sample(S=opt.ddim_steps,
                                                        batch_size=batch_size,
                                                        conditioning=c,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=start_code,
                                                        image_encoder=image_encoder,
                                                        tt = opt.tt,
                                                        rho = opt.rho,
                                                        reward_eval=reward_eval,
                                                        search_algo=search_algo,
                                                        ref=ref,                                                    
                                                        )

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).detach().numpy()

                    # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                    x_checked_image = x_samples_ddim  # todo: switch back
                    x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                    for n, x_sample in enumerate(x_checked_image_torch):
                        results = evaluator.report(gt=ref, x=x_sample.unsqueeze(0), text=opt.prompt)
                        markdown_text, metrics = evaluator.display(results)
                        markdown_table += '\n' + markdown_text

                        # jump metrics to json
                        json.dump(metrics, open(os.path.join(metrics_path, f"{'.'.join(filename.split('.')[:-1])}_metrics_{file_id}_{n}.json"), 'w'), indent=4)


                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        # img = put_watermark(img, wm_encoder)
                        
                        # Save image with timestamp in filename
                        img.save(os.path.join(sample_path, f"{'.'.join(filename.split('.')[:-1])}_{file_id}_{n}.png"))

                        base_count += 1
                   

                file_id += 1

            toc = time.time()

            print('Table:', markdown_table)



if __name__ == "__main__":
    main()
