#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --partition=V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1                 # 申请1个GPU
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log

# 设置CUDA环境
source /river/hpc101/2025/lab5/env/bin/activate


python -u main.py 

echo "GPU程序执行完成！"