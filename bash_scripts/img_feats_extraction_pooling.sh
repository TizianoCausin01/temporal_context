#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=30:00:00
#SBATCH --ntasks=5 # number of processes
#SBATCH --cpus-per-task=1
#SBATCH --mem=250G
#SBATCH --account=livingstone       # account name
#SBATCH --partition=priority # partition name
#SBATCH --job-name=img_feats_extraction
#SBATCH --output=/home/tic569/output_jobs/%x.%j.out   # file name will be *job_name*.*job_id*
cd /home/tic569/temporal_context/python_scripts/scripts
module load gcc/14.2.0
module load python/3.13.1
module load openmpi
source ~/virtual_envs/temporal_context/bin/activate
mpiexec python3 run_img_feats_extraction.py --model_name=$1 --num_components=$2 --batch_size=$3 --img_size=$4 --monkey_name=$5 --date=$6 --folder_name=$7 --pkg=$8 --pooling=$9
# example call 
# sbatch img_feats_extraction.sh vit_l_16 1000 1100 384 paul 230204 fewer_occlusion timm 


