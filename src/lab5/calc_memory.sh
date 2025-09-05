#!/bin/bash
#SBATCH --job-name=memory_calc
#SBATCH --partition=V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=memory_calc_%j.log

echo "启动理论显存计算..."
source /river/hpc101/2025/lab5/env/bin/activate

python calculate_memory.py

echo "计算完成!"
