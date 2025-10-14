#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=dm-mpgd     #Set the job name to "JobExample1"
#SBATCH --time=24:00:00            #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                 #Request 1 task
#SBATCH --ntasks-per-node=1        #Request 1 task/core per node
#SBATCH --mem=32G               #Request 64GB per node
#SBATCH --gres=gpu:a100:1     #Request 1 GPU
#SBATCH --output=logs/search-for-inv.%j  #Output file name stdout to [JobID]


cd $SCRATCH/semiblind-dps/mpgd-search/linear_inv
ml Miniconda3
module load WebProxy

source activate mpgd


python inference_time_mpgd.py \
        --model_config=configs/model_config.yaml \
        --diffusion_config=configs/mpgd_diffusion_search_config.yaml \
        --task_config=configs/box_inpainting_det_full_images.yaml \
        --reward_eval_config=configs/reward_adaface.yaml \
        --timestep=100 \
        --scale=4 \
        --method="mpgd_wo_proj" \
        --save_dir='./outputs_paper/box_inpainting_s64/mpgd/' \
        --n_images=70 \
        --temp=0.05  \
        --num_particles=1 \
        --batch_size=8 \
        --resample_rate=8 \
        --ref_faces_path='./data/additional_images/' \
        --best_of_n \

python inference_time_mpgd.py \
        --model_config=configs/model_config.yaml \
        --diffusion_config=configs/mpgd_diffusion_search_config.yaml \
        --task_config=configs/box_inpainting_det_full_images.yaml \
        --reward_eval_config=configs/reward_adaface_gradient.yaml \
        --timestep=100 \
        --scale=4 \
        --method="mpgd_wo_proj" \
        --save_dir='./outputs_paper/box_inpainting_s64/grad/' \
        --n_images=70 \
        --temp=0.05  \
        --num_particles=1 \
        --batch_size=8 \
        --resample_rate=8 \
        --ref_faces_path='./data/additional_images/' \
        --gradient_scale=0.5 \
        --best_of_n \


python inference_time_mpgd.py \
        --model_config=configs/model_config.yaml \
        --diffusion_config=configs/mpgd_diffusion_search_config.yaml \
        --task_config=configs/box_inpainting_det_full_images.yaml \
        --reward_eval_config=configs/reward_adaface.yaml \
        --timestep=100 \
        --scale=4 \
        --method="mpgd_wo_proj" \
        --save_dir='./outputs_paper/box_inpainting_s64/best_of_n/' \
        --n_images=70 \
        --temp=0.05  \
        --num_particles=8 \
        --batch_size=8 \
        --resample_rate=8 \
        --ref_faces_path='./data/additional_images/' \
        --best_of_n \

python inference_time_mpgd.py \
        --model_config=configs/model_config.yaml \
        --diffusion_config=configs/mpgd_diffusion_search_config.yaml \
        --task_config=configs/box_inpainting_det_full_images.yaml \
        --reward_eval_config=configs/reward_adaface.yaml \
        --search_algo_config=configs/search_greedy.yaml \
        --timestep=100 \
        --scale=4 \
        --method="mpgd_wo_proj" \
        --save_dir='./outputs_paper/box_inpainting_s64/global_search/' \
        --n_images=70 \
        --temp=0.05  \
        --num_particles=8 \
        --batch_size=8 \
        --resample_rate=8 \
        --ref_faces_path='./data/additional_images/' 

python inference_time_mpgd.py \
        --model_config=configs/model_config.yaml \
        --diffusion_config=configs/mpgd_diffusion_search_config.yaml \
        --task_config=configs/box_inpainting_det_full_images.yaml \
        --reward_eval_config=configs/reward_adaface.yaml \
        --search_algo_config=configs/search_group_recursive_greedy.yaml \
        --timestep=100 \
        --scale=4 \
        --method="mpgd_wo_proj" \
        --save_dir='./outputs_paper/box_inpainting_s64/group_search/' \
        --n_images=70 \
        --temp=0.05  \
        --num_particles=8 \
        --batch_size=8 \
        --resample_rate=8 \
        --ref_faces_path='./data/additional_images/' 
