#!/bin/bash
#SBATCH --job-name=dm      # Job name
#SBATCH --mail-type=BEGIN,END,FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=vishnukunde@tamu.edu  #Where to send mail    
#SBATCH --ntasks=8                      # Run on a 8 cpus (max)
#SBATCH --gres=gpu:a100:1              # Run on a single GPU (max)
#SBATCH --partition=gpu-research                 # Select GPU Partition
#SBATCH --qos=olympus-research-gpu          # Specify GPU queue
#SBATCH --time=2:00:00                 # Time limit hrs:min:sec current 5 min - 36 hour max
#SBATCH --output=logs/%x_%j.out        # Standard output and error log

# select your singularity shell (currently cuda10.2-cudnn7-py36)
singularity shell /mnt/lab_files/ECEN403-404/containers/cuda_10.2-cudnn7-py36.sif

# Define seeds
seeds=(42 43 44 45 46 47 48 49 50 51)
# Define the style reference path

# Loop through each seed
for seed in "${seeds[@]}"; do
    # Run with --resample

    python style.py --style_ref_path './style_images/' \
                    --ddim_steps 100 --n_iter 1 --H 512 --W 512 \
                    --scale 5.0 --rho 15 --tt 1 --seed $seed \
                    --prompt "a knight holding his sword" \
                    --fixed_code --n_samples 4 \
                    --resample_every_t 20 \
                    --tilt_lambda_style 2 \
                    --tilt_lambda_text 1 \
                    --text_style_slider 1 \
                    --outdir './output_4/' \
                    --resample

    # Run with --resample and --updated_style_loss
    python style.py --style_ref_path './style_images/' \
                    --ddim_steps 100 --n_iter 1 --H 512 --W 512 \
                    --scale 5.0 --rho 15 --tt 1 --seed $seed \
                    --prompt "a knight holding his sword" \
                    --fixed_code --n_samples 4 \
                    --resample_every_t 20 \
                    --tilt_lambda_style 2 \
                    --tilt_lambda_text 1 \
                    --text_style_slider 1 \
                    --outdir './output_4/' \
                    --resample --updated_style_loss

    # Run without --resample and --updated_style_loss
    python style.py --style_ref_path './style_images/' \
                    --ddim_steps 100 --n_iter 1 --H 512 --W 512 \
                    --scale 5.0 --rho 15 --tt 1 --seed $seed \
                    --prompt "a knight holding his sword" \
                    --fixed_code --n_samples 4 \
                    --resample_every_t 20 \
                    --tilt_lambda_style 2 \
                    --tilt_lambda_text 1 \
                    --outdir './output_4/' \
                    --text_style_slider 1
                    
done  # Add this missing 'done'

