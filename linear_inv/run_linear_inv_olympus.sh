#!/bin/bash
#SBATCH --job-name=dm-lin-inv      # Job name
#SBATCH --mail-type=BEGIN,END,FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=vishnukunde@tamu.edu  #Where to send mail    
#SBATCH --ntasks=1                      # Run on a 8 cpus (max)
#SBATCH --gres=gpu:a100:1              # Run on a single GPU (max)
#SBATCH --partition=gpu-research                 # Select GPU Partition
#SBATCH --qos=olympus-research-gpu          # Specify GPU queue
#SBATCH --time=00:30:00                 # Time limit hrs:min:sec current 5 min - 36 hour max
#SBATCH --output=logs/%x_%j.out        # Standard output and error log

# select your singularity shell (currently cuda10.2-cudnn7-py36)
singularity shell /mnt/lab_files/ECEN403-404/containers/cuda_10.2-cudnn7-py36.sif

# # Define seeds
temps=(0.5)
num_particles=(2)
num_lookahead_steps=(1)
resample_rates=(1)

# # Define the style reference path

# # Loop through each seed

# gradient
python batched_ffhq_coarse_lookahead.py \
                        --model_config=configs/model_config.yaml \
                        --diffusion_config=configs/mpgd_diffusion_search_config.yaml \
                        --task_config=configs/motion_blur_config.yaml \
                        --reward_eval_config=configs/reward_facenet_gradient.yaml \
                        --timestep=200 \
                        --scale=4 \
                        --method="mpgd_wo_proj" \
                        --num_lookahead_steps=1 \
                        --save_dir='./outputs_test_gradient_vs_search/' \
                        --n_images=4 \
                        --temp=0.5 \
                        --num_particles=1 \
                        --batch_size=4 \
                        --resample_rate=1 \
                        --best_of_n \
                        --seed=8 \

# # search
# python batched_ffhq_coarse_lookahead.py \
#                         --model_config=configs/model_config.yaml \
#                         --diffusion_config=configs/mpgd_diffusion_search_config.yaml \
#                         --task_config=configs/motion_blur_config.yaml \
#                         --reward_eval_config=configs/reward_facenet.yaml \
#                         --timestep=200 \
#                         --scale=4 \
#                         --method="mpgd_wo_proj" \
#                         --num_lookahead_steps=1 \
#                         --save_dir='./outputs_test_gradient_vs_search/' \
#                         --n_images=4 \
#                         --temp=0.5 \
#                         --num_particles=4 \
#                         --batch_size=4 \
#                         --resample_rate=1 \
#                         --seed=8 \
#                         # --best_of_n \


# python batched_ffhq_coarse_lookahead.py \
#                         --model_config=configs/model_config.yaml \
#                         --diffusion_config=configs/mpgd_diffusion_search_config.yaml \
#                         --task_config=configs/super_resolution_config.yaml \
#                         --reward_eval_config=configs/reward_facenet.yaml \
#                         --timestep=200 \
#                         --scale=4 \
#                         --method="mpgd_wo_proj" \
#                         --num_lookahead_steps=1 \
#                         --save_dir='./outputs_test_no_gradient/' \
#                         --n_images=1 \
#                         --temp=0.5 \
#                         --num_particles=1 \
#                         --batch_size=1 \
#                         --resample_rate=1 \
#                         --best_of_n \


# for resample_rate in "${resample_rates[@]}"; do
#     for temp in "${temps[@]}"; do
#         for num_lookahead_step in "${num_lookahead_steps[@]}"; do
#             for num_particle in "${num_particles[@]}"; do
#                         python batched_ffhq_coarse_lookahead.py \
#                         --model_config=configs/model_config.yaml \
#                         --diffusion_config=configs/mpgd_diffusion_search_config.yaml \
#                         --task_config=configs/box_inpainting_det.yaml \
#                         --reward_eval_config=configs/reward_adaface.yaml \
#                         --timestep=200 \
#                         --scale=4 \
#                         --method="mpgd_wo_proj" \
#                         --num_lookahead_steps=$num_lookahead_step \
#                         --save_dir='./outputs_test_box_inpainting/' \
#                         --n_images=10 \
#                         --temp=$temp  \
#                         --num_particles=$num_particle \
#                         --batch_size=2 \
#                         --resample_rate=$resample_rate \
#                         # --jump_la \
#                         # --perform_lookahead \
#                         # --ref_faces_path='./data/samples/' \
#                         # --best_of_n \
                        
#             done
#         done
#     done
# done

# for num_particle in "${num_particles[@]}"; do
#         python batched_ffhq_coarse_lookahead.py \
#         --model_config=configs/model_config.yaml \
#         --diffusion_config=configs/mpgd_diffusion_search_config.yaml \
#         --task_config=configs/super_resolution_config.yaml \
#         --reward_eval_config=configs/reward_adaface.yaml \
#         --timestep=200 \
#         --scale=4 \
#         --method="mpgd_wo_proj" \
#         --num_lookahead_steps=1 \
#         --perform_lookahead \
#         --save_dir='./outputs_final_batched/' \
#         --n_images=1 \
#         --temp=0.5  \
#         --num_particles=$num_particle \
#         --batch_size=16 \
#         --best_of_n \

# done

#     done
# done


# python batched_ffhq_coarse_lookahead.py \
#         --model_config=configs/model_config.yaml \
#         --diffusion_config=configs/mpgd_diffusion_search_config.yaml \
#         --task_config=configs/super_resolution_config.yaml \
#         --reward_eval_config=configs/reward_adaface.yaml \
#         --timestep=20 \
#         --scale=17.5 \
#         --method="mpgd_wo_proj" \
#         --num_lookahead_steps=1 \
#         --perform_lookahead \
#         --save_dir='./outputs_batched/' \
#         --n_images=1 \
#         --temp=0.5  \
#         --num_particles=2 \
#         --batch_size=16 \
#         --best_of_n \

