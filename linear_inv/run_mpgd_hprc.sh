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
            --task_config=configs/gaussian_deblur_config.yaml \
            --reward_eval_config=configs/reward_adaface.yaml \
            --timestep=100 \
            --scale=4 \
            --method="mpgd_wo_proj" \
            --num_lookahead_steps=1 \
            --save_dir='./outputs_final_paper_mpgd/' \
            --n_images=70 \
            --temp=0.05 \
            --num_particles=1 \
            --batch_size=8 \
            --ref_faces_path='./data/additional_images/' \
            --best_of_n 