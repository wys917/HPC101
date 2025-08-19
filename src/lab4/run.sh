#!/bin/bash
#SBATCH --job-name=solver
#SBATCH --output=run.out
#SBATCH --error=run.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96  
#SBATCH --time=00:2:00
#SBATCH --partition=solver2

# 对于 sbatch 脚本参数，请阅读 https://slurm.schedmd.com/sbatch.html
# ntasks-per-node 对应每个节点上的进程数
# cpus-per-task 对应每个进程的线程数

# 先加载 spack 再加载编译环境
# e.g. 
#source ${HOME}/spack/share/spack/setup-env.sh
#spack load intel-oneapi-mpi
source /pxe/opt/spack/share/spack/setup-env.sh
spack env activate hpc101-intel

export OMP_NUM_THREADS=48

# 添加 Intel MPI 在 Slurm 上运行所需的环境变量
export I_MPI_PMI_LIBRARY=/slurm/libpmi2.so.0.0.0

# 使用 srun 启动 MPI + OpenMP 混合并行程序
srun --mpi=pmi2 ./build/bicgstab $1