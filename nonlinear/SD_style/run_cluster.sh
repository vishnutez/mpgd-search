#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=dm     #Set the job name to "JobExample1"
#SBATCH --time=01:30:00            #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                 #Request 1 task
#SBATCH --ntasks-per-node=1        #Request 1 task/core per node
#SBATCH --mem=32G               #Request 64GB per node
#SBATCH --gres=gpu:a100:1     #Request 1 GPU
#SBATCH --output=logs/style-guid.%j  #Output file name stdout to [JobID]


cd $SCRATCH/semiblind-dps/mpgd-search/nonlinear/SD_style
ml Miniconda3
ml WebProxy

source activate mpgd

python style.py --ddim_steps 100 --n_iter 1 --H 512 --W 512 --scale 5.0 --rho 15 --tt 1 --prompt "a knight holding his sword" --fixed_code