#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --partition=V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1                 # 申请1个GPU
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log
echo "--- Final Proof & Capture Job ---"
source /river/hpc101/2025/lab5/env/bin/activate
# 在后台启动你的python程序，让它持续占用GPU
echo "INFO: Starting python script in the background..."
python -u final_gpu_proof.py &

# 等待15秒，确保模型完全加载
echo "INFO: Waiting 15 seconds for model to load..."
sleep 15

# 关键一步：运行nvtop的“非交互式”模式，拍一张快照然后退出
# -b: Batch mode (非交互式)
# -n 1: 只刷新一次就退出
echo "INFO: Taking a snapshot with nvtop from INSIDE the job..."
nvtop -b -n 1

# 清理后台的python进程
echo "INFO: Cleaning up background python process..."
killall python

echo "INFO: Job finished."