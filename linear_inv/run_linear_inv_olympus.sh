#!/bin/bash
#SBATCH --job-name=dm      # Job name
#SBATCH --mail-type=BEGIN,END,FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=vishnukunde@tamu.edu  #Where to send mail    
#SBATCH --ntasks=8                      # Run on a 8 cpus (max)
#SBATCH --gres=gpu:a100:1              # Run on a single GPU (max)
#SBATCH --partition=gpu-research                 # Select GPU Partition
#SBATCH --qos=olympus-research-gpu          # Specify GPU queue
#SBATCH --time=0:30:00                 # Time limit hrs:min:sec current 5 min - 36 hour max
#SBATCH --output=logs/%x_%j.out        # Standard output and error log

# select your singularity shell (currently cuda10.2-cudnn7-py36)
singularity shell /mnt/lab_files/ECEN403-404/containers/cuda_10.2-cudnn7-py36.sif

# # Define seeds
# temps=(2 1 0.5 0.2 0.1 0.05)
# num_particles=(2 4)
# num_lookahead_steps=(1 2 4)
# # Define the style reference path

# # Loop through each seed

# for num_particle in "${num_particles[@]}"; do
#     for num_lookahead_step in "${num_lookahead_steps[@]}"; do
#         for temp in "${temps[@]}"; do

#             python ffhq_jump_lookahead.py \
#                 --model_config=configs/model_config.yaml \
#                 --diffusion_config=configs/mpgd_diffusion_search_config.yaml \
#                 --task_config=configs/super_resolution_config.yaml \
#                 --reward_eval_config=configs/reward_facenet.yaml \
#                 --timestep=200 \
#                 --scale=4 \
#                 --method="mpgd_wo_proj" \
#                 --num_lookahead_steps=$num_lookahead_step \
#                 --perform_lookahead \
#                 --save_dir='./outputs_effect_of_temp_facenet/' \
#                 --n_images=10 \
#                 --temp=$temp  \
#                 --num_particles=$num_particle \
#                 --jump_size=1 \
#             # --ref_faces_path='./data/samples/' \
#             # --best_of_n \

#         done
#     done
# done


python ffhq_jump_lookahead.py \
        --model_config=configs/model_config.yaml \
        --diffusion_config=configs/mpgd_diffusion_search_config.yaml \
        --task_config=configs/super_resolution_config.yaml \
        --reward_eval_config=configs/reward_adaface.yaml \
        --timestep=200 \
        --scale=4 \
        --method="mpgd_wo_proj" \
        --num_lookahead_steps=1 \
        --perform_lookahead \
        --save_dir='./outputs_jump_la/' \
        --n_images=10 \
        --temp=0.5  \
        --num_particles=2 \
        --jump_size=1 \

