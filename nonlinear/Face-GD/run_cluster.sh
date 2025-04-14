#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=dm     #Set the job name to "JobExample1"
#SBATCH --time=01:30:00            #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                 #Request 1 task
#SBATCH --ntasks-per-node=1        #Request 1 task/core per node
#SBATCH --mem=32G               #Request 64GB per node
#SBATCH --gres=gpu:a100:1     #Request 1 GPU
#SBATCH --output=logs/style-guid.%j  #Output file name stdout to [JobID]


cd $SCRATCH/semiblind-dps/mpgd-search/nonlinear/Face-GD
ml Miniconda3
ml WebProxy

source activate mpgd

# faceid
python main.py -i wo -s faceid --doc celeba_hq --timesteps 50 --rho_scale 0.015 --stop 100 --batch_size 1 --eta 0.5 --ref_path ./images/294.jpg --repeat 1

# clip
# python main.py -i wo -s face_clip --doc celeba_hq --timesteps 50 --rho_scale 1.5 --seed 0 --stop 100 --batch_size 1 --prompt "a headshot of a person wearing red lipstick" --repeat 1 --repeat_start 500 --repeat_end 200