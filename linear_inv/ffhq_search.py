from functools import partial
import os
import argparse
import yaml

from search_algo import get_search_algo
from reward_eval import get_reward_eval

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model

from guided_diffusion.search_guided_gaussian_diffusion import create_sampler  # changed to search guided gaussian diffusion


from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger
import torchvision

import numpy as np
import random
from PIL import Image
import glob
from pathlib import Path

def seed_everything(seed: int):
    """Seed all random number generators for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    random.seed(seed)
    np.random.seed(seed)

    # os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # os.environ['CUDNN_DETERMINISTIC'] = '1'


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--timestep', type=int, default=100)
    parser.add_argument('--eta', type=float, default=0.5)
    parser.add_argument('--scale', type=float, default=10)
    parser.add_argument('--method', type=str, default='mpgd_wo_proj')
    parser.add_argument('--save_dir', type=str, default='./outputs/ffhq/')
    parser.add_argument('--eval_fn_list', type=str, nargs='+', default=['psnr', 'ssim', 'lpips', 'face_sim_l2'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--search_algo_config', type=str, default='./configs/search_resample.yaml')
    parser.add_argument('--reward_eval_config', type=str, default='./configs/reward_eval_facenet.yaml')
    parser.add_argument('--ref_faces_path', type=str, default='./data/ref-face-images/')
    args = parser.parse_args()
   
    # logger
    logger = get_logger()

    # Set random seed
    seed_everything(args.seed)
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
    search_algo_config = load_yaml(args.search_algo_config)
    reward_eval_config = load_yaml(args.reward_eval_config)
    
    if args.timestep < 1000:
        diffusion_config["timestep_respacing"] = f"ddim{args.timestep}"
        diffusion_config["rescale_timesteps"] = True
    else:
        diffusion_config["timestep_respacing"] = f"1000"
        diffusion_config["rescale_timesteps"] = False
    
    diffusion_config["eta"] = args.eta
    task_config["conditioning"]["method"] = args.method
    task_config["conditioning"]["params"]["scale"] = args.scale
    
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    # cond_method = get_conditioning_method(cond_config['method'], operator, noiser, resume = "../nonlinear/SD_style/models/ldm/celeba256/model.ckpt", **cond_config['params']) # in the paper we used this checkpoint
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, resume = "../nonlinear/SD_style/models/ldm/ffhq256/model.ckpt", **cond_config['params']) # you can probably also use this checkpoint, but you probably want to tune the hyper-parameter a bit
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)

    print(f"Search algorithm: {search_algo_config['name']}")
    print(f"Reward evaluation: {reward_eval_config['name']}")

    search_algo = get_search_algo(**search_algo_config)
    reward_eval = get_reward_eval(**reward_eval_config)

    num_particles = search_algo_config['num_particles']
   
    # Working directory
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_path = f"{timestamp}_{diffusion_config['timestep_respacing']}_eta{args.eta}_scale{args.scale}"

    task_name = measure_config['operator']['name']

    if task_name == 'super_resolution':
        task_name = f"{task_name}_x{measure_config['operator']['scale_factor']}"
    out_path = os.path.join(args.save_dir, task_name, task_config['conditioning']['method'], dir_path)
    
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )

    from eval import get_eval_fn, Evaluator

    # get evaluator
    eval_fn_list = []
    for eval_fn_name in args.eval_fn_list:
        eval_fn_list.append(get_eval_fn(eval_fn_name))
    evaluator = Evaluator(eval_fn_list)

    images = []
    samples = []

    img_size = 256
    transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    ref_faces = [file for ext in extensions for file in Path().rglob(ext)]

    n_images = 10
        
    # Do Inference
    for i, ref_img in enumerate(loader):

        ref_face_img = Image.open(ref_faces[i]).convert('RGB')

        ref_face_img = transform(ref_face_img)
        ref_face_img = ref_face_img.to(device)
        ref_face_img = ref_face_img.unsqueeze(0)

        print(f"ref_face_img: {ref_face_img}")
        
        if i >= n_images:
            break
        logger.info(f"Inference for image {i}")
        fname = f'{i:03}.png'
        ref_img = ref_img.to(device)

        print(f'ref_img shape: {ref_img.shape}')
        print(f'ref_face_img shape: {ref_face_img.shape}')

        # Forward measurement model (Ax + n)
        y = operator.forward(ref_img)
        y_n = noiser(y)

        # Sampling
        x_start = torch.randn((num_particles, 3, img_size, img_size), device=device).requires_grad_()
        print(f"x_start shape: {x_start.shape}")

        sample = sample_fn(x_start=x_start, 
                           measurement=y_n, 
                           record=False, 
                           save_root=out_path, 
                           reward_eval=reward_eval, 
                           search_algo=search_algo,
                           ref=ref_face_img)

    
        images.append(ref_img)
        samples.append(sample)

        plt.imsave(os.path.join(out_path, 'input', f'{i:03}_input.png'), clear_color(y_n))
        plt.imsave(os.path.join(out_path, 'label', f'{i:03}_label.png'), clear_color(ref_img))
        
        for n, sample_n in enumerate(sample):
            plt.imsave(os.path.join(out_path, 'recon', f'{i:03}_recon_{n}.png'), clear_color(sample_n))

    
    images = torch.cat(images, dim=0)
    samples = torch.cat(samples, dim=0)

    # log metrics
    n_uniq_samples = len(samples) // num_particles

    markdown_table = ''

    for n in range(num_particles):
        idxs = np.arange(n_uniq_samples) * num_particles + n
        print('idxs:', idxs)
        results = evaluator.report(images, y, samples[idxs])
        markdown_text = evaluator.display(results)
        markdown_table += '\n' + markdown_text

    print(markdown_table)

    # log the evaluation metrics
    eval_file_path = os.path.join(out_path, 'eval.md')
    with open(eval_file_path, 'w') as file:
        file.write(markdown_table)

if __name__ == '__main__':
    main()
