#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --ntasks=5 # number of processes
#SBATCH --cpus-per-task=1
#SBATCH --mem=250G
#SBATCH --account=livingstone       # account name
#SBATCH --partition=short # partition name
#SBATCH --job-name=img_feats_extraction
#SBATCH --output=/home/tic569/output_jobs/%x.%j.out   # file name will be *job_name*.*job_id*
cd /home/tic569/temporal_context/python_scripts/scripts
module load gcc/14.2.0
module load python/3.13.1
module load openmpi
source ~/virtual_envs/temporal_context/bin/activate
mpiexec python3 run_img_feats_extraction_pooling.py --model_name=$1 --batch_size=$2 --img_size=$3 --folder_name=$4 --pkg=$5 --pooling=$6
# example call 
# sbatch img_feats_extraction_pooling.sh vit_l_16 1100 384 fewer_occlusion timm mean


