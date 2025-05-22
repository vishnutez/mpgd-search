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

grad_scales=(0.1 0.25 0.5)

# # Define the style reference path

# # Loop through each seed

for grad_scale in "${grad_scales[@]}"; do
        python batched_ffhq_coarse_lookahead.py \
        --model_config=configs/model_config.yaml \
        --diffusion_config=configs/mpgd_diffusion_search_config.yaml \
        --task_config=configs/gaussian_deblur_config.yaml \
        --reward_eval_config=configs/reward_adaface_gradient.yaml \
        --timestep=100 \
        --scale=4 \
        --method="mpgd_wo_proj" \
        --save_dir='./outputs_final_paper_mpgd_vs_grad/' \
        --n_images=70 \
        --temp=0.05  \
        --num_particles=1 \
        --batch_size=8 \
        --resample_rate=2 \
        --ref_faces_path='./data/additional_images/' \
        --gradient_scale=$grad_scale \
        --best_of_n 
done


