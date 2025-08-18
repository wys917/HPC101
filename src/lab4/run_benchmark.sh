#!/bin/bash
#SBATCH --job-name=scaling_test
#SBATCH --partition=M7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:10:00
#SBATCH --output=reports_time/benchmark_%j.out
#SBATCH --error=reports_time/benchmark_%j.err

source /pxe/opt/spack/share/spack/setup-env.sh
# spack load 

mkdir -p reports_time

source /pxe/opt/spack/share/spack/setup-env.sh
spack load intel-oneapi-mpi
spack load intel-oneapi-vtune@2025.0.1 

# 设置线程数
export OMP_NUM_THREADS=16

echo "================================================="
echo "Running benchmark with $OMP_NUM_THREADS threads..."
echo "================================================="

# 使用time命令来计时，它会输出程序运行的真实时间
time ./build/bicgstab data/case_2001.bin