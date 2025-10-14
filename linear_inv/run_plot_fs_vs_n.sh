#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=plot-fs     #Set the job name to "JobExample1"
#SBATCH --time=00:30:00            #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                 #Request 1 task
#SBATCH --ntasks-per-node=1        #Request 1 task/core per node
#SBATCH --mem=32G               #Request 64GB per node
##SBATCH --gres=gpu:a100:1     #Request 1 GPU
#SBATCH --output=logs/plot-fs.%j  #Output file name stdout to [JobID]


cd $SCRATCH/semiblind-dps/mpgd-search/linear_inv
ml Miniconda3
module load WebProxy

source activate mpgd

python plot_face_similarity_vs_n.py



