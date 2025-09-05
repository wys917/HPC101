#!/bin/bash
#SBATCH --job-name=qwen3_analysis
#SBATCH --partition=V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=qwen3_analysis_%j.log
#SBATCH --error=qwen3_analysis_%j.err

echo "🚀 开始Qwen3模型参数量和显存分析"
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_JOB_NODELIST"
echo "时间: $(date)"
echo "================================"

# 激活环境
source /river/hpc101/2025/lab5/env/bin/activate
source /home/hpc101/h3240103466/miniconda3/etc/profile.d/conda.sh
conda activate myenv

# 显示GPU信息
echo "GPU信息:"
nvidia-smi
echo "================================"

# 运行参数量分析
echo "运行参数量和显存分析..."
cd /home/hpc101/h3240103466/HPC101/src/lab5
python calculate_params.py

echo "================================"
echo "分析完成，时间: $(date)"

echo "开始分析Qwen3模型参数量和显存占用..."
echo "时间: $(date)"
echo "节点: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# 激活环境
module load anaconda3
source activate /share/anaconda3/envs/llm

echo "Python环境信息:"
python --version
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA可用: $(python -c 'import torch; print(torch.cuda.is_available())')"

echo ""
echo "开始显存基线测量..."
nvidia-smi

echo ""
echo "运行模型分析脚本..."
python calculate_model_size.py

echo ""
echo "分析完成!"
echo "结束时间: $(date)"
