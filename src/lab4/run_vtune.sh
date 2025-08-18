#!/bin/bash
#SBATCH --job-name=solver
#SBATCH --output=run.out
#SBATCH --error=run.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:10:00
#SBATCH --partition=M7

# 对于 sbatch 脚本参数，请阅读 https://slurm.schedmd.com/sbatch.html
# ntasks-per-node 对应每个节点上的进程数
# cpus-per-task 对应每个进程的线程数

# 先加载 spack 再加载编译环境
# e.g.
source /pxe/opt/spack/share/spack/setup-env.sh
spack load intel-oneapi-mpi
spack load intel-oneapi-vtune@2025.0.1 

export OMP_NUM_THREADS=64

# Run BICGSTAB
# 在结果目录名前面加上 reports/
vtune -collect hotspots -result-dir reports/vtune_results_$SLURM_JOB_ID ./build/bicgstab $1
# 把 hotspots 改成 threading !!!
#vtune -collect threading -result-dir reports/vtune_threading_$SLURM_JOB_ID ./build/bicgstab $1
