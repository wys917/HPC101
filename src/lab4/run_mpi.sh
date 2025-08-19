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

# 清除可能冲突的环境变量，让 SLURM 处理
#unset I_MPI_PMI_LIBRARY
#export I_MPI_HYDRA_BOOTSTRAP=slurm
#export I_MPI_PMI=pmi2
# 添加 Intel MPI 在 Slurm 上运行所需的环境变量
export I_MPI_PMI_LIBRARY=/slurm/libpmi2.so.0.0.0
# 根据提示，设置 OpenMP 线程数为申请到的逻辑核心数的一半（即物理核心数）
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 2))
mkdir -p reports

# 使用 srun 启动 MPI 程序
srun  --mpi=pmi2 ./build/bicgstab data/case_2001.bin


# Run BICGSTAB
# 用srun启动MPI程序！
#echo "Running MPI benchmark with ${SLURM_NTASKS} processes..."
#srun ./build/bicgstab data/case_2001.bin # 必须用srun, 并且提供数据文件
