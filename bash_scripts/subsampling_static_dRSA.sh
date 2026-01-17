#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --ntasks=10 # number of processes
#SBATCH --cpus-per-task=1
#SBATCH --mem=250G
#SBATCH --account=livingstone       # account name
#SBATCH --partition=priority # partition name
#SBATCH --job-name=subsampling_static_dRSA
#SBATCH --output=/home/tic569/output_jobs/%x.%j.out   # file name will be *job_name*.*job_id*
cd /home/tic569/temporal_context/python_scripts/scripts
module load gcc/14.2.0
module load python/3.13.1
module load openmpi
source ~/virtual_envs/temporal_context/bin/activate

#while [[ $# -gt 0 ]]; do
#  case $1 in
#    --monkey_name) MONKEY_NAME="$2"; shift 2 ;;
#    --date) DATE="$2"; shift 2 ;;
#    --brain_area) BRAIN_AREA="$2"; shift 2 ;;
#    --model_name) MODEL_NAME="$2"; shift 2 ;;
#    --img_size) IMG_SIZE="$2"; shift 2 ;;
#    --pooling) POOLING="$2"; shift 2 ;;
#    --pkg) PKG="$2"; shift 2 ;;
#    --RDM_metric) RDM_METRIC="$2"; shift 2 ;;
#    --new_fs) NEW_FS="$2"; shift 2 ;;
#    --max_size) MAX_SIZE="$2"; shift 2 ;;
#    --step_samples) STEP_SAMPLES="$2"; shift 2 ;;
#    --n_iter) N_ITER="$2"; shift 2 ;;
#    *) echo "Unknown argument $1"; exit 1 ;;
#  esac
#done
#
#mpiexec python3 run_subsample_static_drsa.py \
#  --monkey_name "$MONKEY_NAME" \
#  --date "$DATE" \
#  --brain_area "$BRAIN_AREA" \
#  --model_name "$MODEL_NAME" \
#  --img_size "$IMG_SIZE" \
#  --pooling "$POOLING" \
#  --pkg "$PKG" \
#  --RDM_metric "$RDM_METRIC" \
#  --new_fs "$NEW_FS" \
#  --max_size "$MAX_SIZE" \
#  --step_samples "$STEP_SAMPLES" \
#  --n_iter "$N_ITER"


mpiexec python3 run_subsample_static_drsa.py --monkey_name=$1 --date=$2 --brain_area=$3 --model_name=$4 --img_size=$5 --pooling=$6 --pkg=$7 --RDM_metric=$8 --new_fs=$9 --max_size=${10} --step_samples=${11} --n_iter=${12}

# example call 
# sbatch subsampling_static_dRSA.sh paul 230204 AIT vit_l_16 384 mean timm cosine 100 300 100 3 
# args order: 
# --monkey_name=paul --date=230204 --brain_area=AIT --model_name=vit_l_16 --img_size=384 --pooling=mean --pkg=timm --RDM_metric=cosine --new_fs=100 --max_size=300 --step_samples=100 --n_iter=3

