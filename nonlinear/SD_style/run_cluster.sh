#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=dm     #Set the job name to "JobExample1"
#SBATCH --time=03:00:00            #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                 #Request 1 task
#SBATCH --ntasks-per-node=1        #Request 1 task/core per node
#SBATCH --mem=32G               #Request 32GB per node
#SBATCH --gres=gpu:a100:1     #Request 1 GPU
#SBATCH --output=logs/style-guid-latest.%j  #Output file name stdout to [JobID]


cd $SCRATCH/semiblind-dps/mpgd-search/nonlinear/SD_style
ml Miniconda3
ml WebProxy

source activate mpgd

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
                    --fixed_code --n_samples 2 \
                    --resample_every_t 20 \
                    --tilt_lambda_style 2 \
                    --tilt_lambda_text 1 \
                    --text_style_slider 1 \
                    --resample

    # Run with --resample and --updated_style_loss
    python style.py --style_ref_path './style_images/' \
                    --ddim_steps 100 --n_iter 1 --H 512 --W 512 \
                    --scale 5.0 --rho 15 --tt 1 --seed $seed \
                    --prompt "a knight holding his sword" \
                    --fixed_code --n_samples 2 \
                    --resample_every_t 20 \
                    --tilt_lambda_style 2 \
                    --tilt_lambda_text 1 \
                    --text_style_slider 1 \
                    --resample --updated_style_loss

    # Run without --resample and --updated_style_loss
    python style.py --style_ref_path './style_images/' \
                    --ddim_steps 100 --n_iter 1 --H 512 --W 512 \
                    --scale 5.0 --rho 15 --tt 1 --seed $seed \
                    --prompt "a knight holding his sword" \
                    --fixed_code --n_samples 2 \
                    --resample_every_t 20 \
                    --tilt_lambda_style 2 \
                    --tilt_lambda_text 1 \
                    --text_style_slider 1
                    
done  # Add this missing 'done'

