#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=dm     #Set the job name to "JobExample1"
#SBATCH --time=01:30:00            #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                 #Request 1 task
#SBATCH --ntasks-per-node=1        #Request 1 task/core per node
#SBATCH --mem=32G               #Request 64GB per node
#SBATCH --gres=gpu:a100:1     #Request 1 GPU
#SBATCH --output=logs/test-gram-clip.%j  #Output file name stdout to [JobID]


cd $SCRATCH/semiblind-dps/mpgd-search/nonlinear/SD_style/ldm/models/diffusion/clip
ml Miniconda3
ml WebProxy

source activate mpgd

python base_clip.py