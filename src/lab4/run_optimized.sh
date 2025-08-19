#!/bin/bash
#SBATCH --job-name=bicgstab_optimized
#SBATCH --output=run.out
#SBATCH --error=run.err
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --partition=M7

# 加载环境
source /pxe/opt/spack/share/spack/setup-env.sh
spack load intel-oneapi-mpi

# 清除可能冲突的环境变量
unset I_MPI_PMI_LIBRARY
export I_MPI_HYDRA_BOOTSTRAP=slurm
export I_MPI_PMI=pmi2

# 设置OpenMP环境
export OMP_NUM_THREADS=2
export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "=== MPI+OpenMP Hybrid BiCGSTAB Optimization ==="
echo "MPI processes: ${SLURM_NTASKS}"
echo "OpenMP threads per process: ${OMP_NUM_THREADS}"
echo "Total CPU cores: $((SLURM_NTASKS * OMP_NUM_THREADS))"
echo "Problem: case_2001 (N=2001)"
echo "==============================================="

# 运行程序
srun -n 16 --mpi=pmi2 ./build/bicgstab data/case_2001.bin
