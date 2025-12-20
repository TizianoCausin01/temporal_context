#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --ntasks=10 # number of processes
#SBATCH --cpus-per-task=1
#SBATCH --mem=250G
#SBATCH --account=livingstone       # account name
#SBATCH --partition=priority # partition name
#SBATCH --job-name=ipca
#SBATCH --output=/home/tic569/output_jobs/%x.%j.out   # file name will be *job_name*.*job_id*
cd /home/tic569/temporal_context/python_scripts/scripts
module load gcc/14.2.0
module load python/3.13.1
module load openmpi
source ~/virtual_envs/temporal_context/bin/activate
mpiexec python3 run_torchvision_model.py --model_name=$1 
