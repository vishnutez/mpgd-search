#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=dm-mpgd     #Set the job name to "JobExample1"
#SBATCH --time=01:30:00            #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                 #Request 1 task
#SBATCH --ntasks-per-node=1        #Request 1 task/core per node
#SBATCH --mem=32G               #Request 64GB per node
#SBATCH --gres=gpu:a100:1     #Request 1 GPU
#SBATCH --output=logs/search-for-inv.%j  #Output file name stdout to [JobID]


cd $SCRATCH/semiblind-dps/mpgd-search/linear_inv
ml Miniconda3
module load WebProxy

source activate mpgd

python3 ffhq_jump_lookahead.py \
            --model_config=configs/model_config.yaml \
            --diffusion_config=configs/mpgd_diffusion_search_config.yaml \
            --task_config=configs/super_resolution_config.yaml \
            --reward_eval_config=configs/reward_adaface.yaml \
            --timestep=100 \
            --scale=4 \
            --method="mpgd_wo_proj" \
            --num_lookahead_steps=1 \
            --perform_lookahead \
            --save_dir='./outputs_jump_la/' \
            --n_images=10 \
            --temp=0.5 \
            --num_particles=2 \
            --best_of_n \
            # --ref_faces_path='./data/samples/' \
            # --best_of_n \
