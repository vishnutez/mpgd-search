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

num_particles=(2)
resample_rates=(2)

# # Loop through each seed
for resample_rate in "${resample_rates[@]}"; do
    for num_particle in "${num_particles[@]}"; do
        python batched_ffhq_coarse_lookahead.py \
        --model_config=configs/model_config.yaml \
        --diffusion_config=configs/mpgd_diffusion_search_config.yaml \
        --task_config=configs/gaussian_deblur_config.yaml \
        --search_algo_config=configs/search_resample_greedy.yaml \
        --reward_eval_config=configs/reward_adaface.yaml \
        --timestep=100 \
        --scale=4 \
        --method="mpgd_wo_proj" \
        --save_dir='./outputs_ultimate_search_greedy/' \
        --n_images=70 \
        --temp=0.05  \
        --num_particles=$num_particle \
        --batch_size=8 \
        --resample_rate=$resample_rate \
        --ref_faces_path='./data/additional_images/' 
    done
done


# # # Loop through each seed
# for resample_rate in "${resample_rates[@]}"; do
#     for num_particle in "${num_particles[@]}"; do
#         python batched_ffhq_coarse_lookahead.py \
#         --model_config=configs/model_config.yaml \
#         --diffusion_config=configs/mpgd_diffusion_search_config.yaml \
#         --task_config=configs/super_resolution_6x_config_full_images.yaml \
#         --search_algo_config=configs/search_resample_greedy.yaml \
#         --reward_eval_config=configs/reward_adaface.yaml \
#         --timestep=100 \
#         --scale=4 \
#         --method="mpgd_wo_proj" \
#         --save_dir='./outputs_ultimate_search_greedy/' \
#         --n_images=70 \
#         --temp=0.05  \
#         --num_particles=$num_particle \
#         --batch_size=8 \
#         --resample_rate=$resample_rate \
#         --ref_faces_path='./data/additional_images/' 
#     done
# done


# # # Loop through each seed
# for resample_rate in "${resample_rates[@]}"; do
#     for num_particle in "${num_particles[@]}"; do
#         python batched_ffhq_coarse_lookahead.py \
#         --model_config=configs/model_config.yaml \
#         --diffusion_config=configs/mpgd_diffusion_search_config.yaml \
#         --task_config=configs/box_inpainting_det_full_images.yaml \
#         --search_algo_config=configs/search_resample_greedy.yaml \
#         --reward_eval_config=configs/reward_adaface.yaml \
#         --timestep=100 \
#         --scale=4 \
#         --method="mpgd_wo_proj" \
#         --save_dir='./outputs_ultimate_search_greedy/' \
#         --n_images=70 \
#         --temp=0.05  \
#         --num_particles=$num_particle \
#         --batch_size=8 \
#         --resample_rate=$resample_rate \
#         --ref_faces_path='./data/additional_images/' 
#     done
# done



num_particles=(4)
resample_rates=(4)

# # Loop through each seed
for resample_rate in "${resample_rates[@]}"; do
    for num_particle in "${num_particles[@]}"; do
        python batched_ffhq_coarse_lookahead.py \
        --model_config=configs/model_config.yaml \
        --diffusion_config=configs/mpgd_diffusion_search_config.yaml \
        --task_config=configs/gaussian_deblur_config.yaml \
        --search_algo_config=configs/search_resample_greedy.yaml \
        --reward_eval_config=configs/reward_adaface.yaml \
        --timestep=100 \
        --scale=4 \
        --method="mpgd_wo_proj" \
        --save_dir='./outputs_ultimate_search_greedy/' \
        --n_images=70 \
        --temp=0.05  \
        --num_particles=$num_particle \
        --batch_size=8 \
        --resample_rate=$resample_rate \
        --ref_faces_path='./data/additional_images/' 
    done
done


# # Loop through each seed
for resample_rate in "${resample_rates[@]}"; do
    for num_particle in "${num_particles[@]}"; do
        python batched_ffhq_coarse_lookahead.py \
        --model_config=configs/model_config.yaml \
        --diffusion_config=configs/mpgd_diffusion_search_config.yaml \
        --task_config=configs/super_resolution_6x_config_full_images.yaml \
        --search_algo_config=configs/search_resample_greedy.yaml \
        --reward_eval_config=configs/reward_adaface.yaml \
        --timestep=100 \
        --scale=4 \
        --method="mpgd_wo_proj" \
        --save_dir='./outputs_ultimate_search_greedy/' \
        --n_images=70 \
        --temp=0.05  \
        --num_particles=$num_particle \
        --batch_size=8 \
        --resample_rate=$resample_rate \
        --ref_faces_path='./data/additional_images/' 
    done
done


# # # Loop through each seed
# for resample_rate in "${resample_rates[@]}"; do
#     for num_particle in "${num_particles[@]}"; do
#         python batched_ffhq_coarse_lookahead.py \
#         --model_config=configs/model_config.yaml \
#         --diffusion_config=configs/mpgd_diffusion_search_config.yaml \
#         --task_config=configs/box_inpainting_det_full_images.yaml \
#         --search_algo_config=configs/search_resample_greedy.yaml \
#         --reward_eval_config=configs/reward_adaface.yaml \
#         --timestep=100 \
#         --scale=4 \
#         --method="mpgd_wo_proj" \
#         --save_dir='./outputs_ultimate_search_greedy/' \
#         --n_images=70 \
#         --temp=0.05  \
#         --num_particles=$num_particle \
#         --batch_size=8 \
#         --resample_rate=$resample_rate \
#         --ref_faces_path='./data/additional_images/' 
#     done
# done