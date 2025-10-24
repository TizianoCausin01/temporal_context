#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=1:30:00
#SBATCH --ntasks=1 # number of processes
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --account=livingstone       # account name
#SBATCH --partition=short # partition name
#SBATCH --job-name=preprocesing
#SBATCH --output=/home/tic569/output_jobs/%x.%j.out   # file name will be *job_name*.*job_id*
cd /home/tic569/temporal_context/python_scripts/scripts
module load gcc/14.2.0
module load python/3.13.1
source ~/virtual_envs/temporal_context/bin/activate
python3 loop_load_and_save_data.py --analyses_name $1
