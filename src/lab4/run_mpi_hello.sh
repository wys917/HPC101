#!/bin/bash
#SBATCH --job-name=mpi_hello
#SBATCH --partition=M7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=00:05:00
#SBATCH --output=mpi_hello_%j.out
#SBATCH --error=mpi_hello_%j.err

# 加载 MPI 环境
source /pxe/opt/spack/share/spack/setup-env.sh
spack load intel-oneapi-mpi

# 清除可能冲突的环境变量，让 SLURM 处理
unset I_MPI_PMI_LIBRARY
export I_MPI_HYDRA_BOOTSTRAP=slurm
export I_MPI_PMI=pmi2

echo "Running MPI Hello World with 16 processes..."
echo "=============================================="

# 使用 srun 启动 MPI 程序
srun -n 16 --mpi=pmi2 ./build/mpi_hello

echo "=============================================="
echo "Done!"
