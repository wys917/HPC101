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

# 根据提示，设置 OpenMP 线程数为申请到的逻辑核心数的一半（即物理核心数）
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 2))
#echo "Running with ${OMP_NUM_THREADS} OpenMP threads..."
echo "Node: $(hostname), Rank: ${SLURM_PROCID}, Threads: ${OMP_NUM_THREADS}"

# Run BICGSTAB
#./build/bicgstab $1

# 使用 vtune 运行程序
# -result-dir r000hs: 将结果保存在名为 r000hs 的文件夹中
#vtune -collect hotspots -result-dir r000hs ./build/bicgstab $1
#vtune -collect uarch-exploration -result-dir r001ue_simd ./build/bicgstab $1
#vtune -collect hotspots -result-dir mpi -- \
#srun --mpi=pmi2 ./build/bicgstab $1

# 添加 Intel MPI 在 Slurm 上运行所需的环境变量
export I_MPI_PMI_LIBRARY=/slurm/libpmi2.so.0.0.0

# 使用 srun 启动 MPI + OpenMP 混合并行程序
srun --mpi=pmi2 ./build/bicgstab $1