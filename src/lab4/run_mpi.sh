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
source /pxe/opt/spack/share/spack/setup-env.sh
#spack load intel-oneapi-mpi
spack env activate hpc101-intel
spack load intel-oneapi-vtune@2025.0.1 

export I_MPI_PMI_LIBRARY=/slurm/libpmi2.so.0.0.
# 根据提示，设置 OpenMP 线程数为申请到的逻辑核心数的一半（即物理核心数）
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 2))


# 使用 srun 启动 MPI 程序
srun  --mpi=pmi2 ./build/bicgstab data/case_2001.bi
