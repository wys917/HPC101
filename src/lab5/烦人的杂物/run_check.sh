#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --partition=V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1                 # 申请1个GPU
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.log

python -u check_env.py