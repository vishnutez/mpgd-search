#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=dm-mpgd     #Set the job name to "JobExample1"
#SBATCH --time=00:05:00            #Set the wall clock limit to 0hr and 0min
#SBATCH --ntasks=1                 #Request 1 task
#SBATCH --ntasks-per-node=1        #Request 1 task/core per node
#SBATCH --mem=12G               #Request 32GB per node
#SBATCH --output=logs/get_stats.%j  #Output file name stdout to [JobID]


cd $SCRATCH/semiblind-dps/mpgd-search/linear_inv
ml Miniconda3
module load WebProxy

source activate mpgd

# python get_mean_median.py

python plot_face_similarity_vs_n.py