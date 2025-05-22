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

# # Define seeds
num_particles=(2 4 8 16)
etas=(0.5 1.0)

# # Loop through each seed
for eta in "${etas[@]}"; do
    for num_particle in "${num_particles[@]}"; do
        python inference_time_mpgd.py \
            --model_config=configs/model_config.yaml \
            --diffusion_config=configs/mpgd_diffusion_search_config.yaml \
            --task_config=configs/box_inpainting_det_full_images.yaml \
            --reward_eval_config=configs/reward_adaface.yaml \
            --timestep=100 \
            --scale=4 \
            --eta=$eta \
            --method="mpgd_wo_proj" \
            --num_lookahead_steps=1 \
            --save_dir='./outputs_final_paper_mpgd_best_of_n/' \
            --n_images=70 \
            --temp=0.05 \
            --num_particles=$num_particle \
            --batch_size=32 \
            --ref_faces_path='./data/additional_images/' \
            --best_of_n 
    done
done


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

# #!/bin/bash
# #SBATCH --job-name=dm      # Job name
# #SBATCH --mail-type=BEGIN,END,FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
# #SBATCH --mail-user=vishnukunde@tamu.edu  #Where to send mail    
# #SBATCH --ntasks=8                      # Run on a 8 cpus (max)
# #SBATCH --gres=gpu:a100:1              # Run on a single GPU (max)
# #SBATCH --partition=gpu-research                 # Select GPU Partition
# #SBATCH --qos=olympus-research-gpu          # Specify GPU queue
# #SBATCH --time=0:30:00                 # Time limit hrs:min:sec current 5 min - 36 hour max
# #SBATCH --output=logs/%x_%j.out        # Standard output and error log

# # select your singularity shell (currently cuda10.2-cudnn7-py36)
# singularity shell /mnt/lab_files/ECEN403-404/containers/cuda_10.2-cudnn7-py36.sif

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

