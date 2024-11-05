#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=64gb
#SBATCH -t 7-0:00:00
#SBATCH -o jobs_logs/job.%J.out
#SBATCH -e jobs_logs/job.%J.err


# srun --pty  --nodes=1 --tasks-per-node=32  --cpus-per-task=1  --mem=64G  bash


CONDA_DIR=/share/apps/NYUAD5/miniconda/3-4.11.0
CONDA_ENV=/home/nb3891/.conda/envs/mujoco_env
conda activate $CONDA_ENV


python train.py \
    --epochs 10000 \
    --n_anchors 2 \
    --beta 1 \
    --n_population 1000 \
    --len_trajectory 1024 \
    --batch_size 64 \
    --mode async \
    --actor_hidden_layers 64 64 64 64 \
    --critic_hidden_layers 256 256 256 256 \

