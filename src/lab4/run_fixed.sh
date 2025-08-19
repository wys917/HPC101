#!/bin/bash
#SBATCH --job-name=bicgstab_hybrid_correct
#SBATCH --output=run.out
#SBATCH --error=run.err
#SBATCH --ntasks=4              # <-- 只启动4个MPI进程！
#SBATCH --cpus-per-task=4       # <-- 给每个进程分配4个核！
#SBATCH --time=00:02:00
#SBATCH --partition=M7

# 加载环境
source /pxe/opt/spack/share/spack/setup-env.sh
spack load intel-oneapi-mpi

# 清除可能冲突的环境变量
unset I_MPI_PMI_LIBRARY
export I_MPI_HYDRA_BOOTSTRAP=slurm
export I_MPI_PMI=pmi2

# 设置OpenMP环境 - 每个进程内部用4个线程
export OMP_NUM_THREADS=4        # <-- 让每个进程内部用4个线程跑！
export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "=== CORRECT MPI+OpenMP Hybrid BiCGSTAB ==="
echo "MPI processes: 4 (减少通信开销)"
echo "OpenMP threads per process: 4 (胖节点策略)"
echo "Total CPU cores: 16"
echo "Communication participants: 4 (instead of 16!)"
echo "=========================================="

# 运行程序 - 只启动4个进程
time srun -n 4 --mpi=pmi2 ./build/bicgstab data/case_2001.bin
