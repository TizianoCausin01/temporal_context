#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=2:30:00
#SBATCH --ntasks=10 # number of processes
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --account=livingstone       # account name
#SBATCH --partition=short # partition name
#SBATCH --job-name=human_face_detection
#SBATCH --output=/home/tic569/output_jobs/%x.%j.out   # file name will be *job_name*.*job_id*
cd /home/tic569/temporal_context/python_scripts/scripts
module load gcc/14.2.0
module load python/3.13.1
module load openmpi
source ~/virtual_envs/temporal_context/bin/activate
mpiexec python3 run_human_face_detection.py
