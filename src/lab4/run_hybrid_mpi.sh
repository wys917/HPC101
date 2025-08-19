#!/bin/bash
#SBATCH --job-name=bicgstab_hybrid
#SBATCH --partition=M7
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=run.out
#SBATCH --error=run.err

# 加载必要的模块
module load mpi/intel-mpi/2021.14

# 设置OpenMP环境变量
export OMP_NUM_THREADS=2
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# 设置Intel MPI环境变量
export I_MPI_PMI=pmi2

# 运行程序
echo "Starting MPI+OpenMP Hybrid BiCGSTAB with $SLURM_NTASKS MPI processes and $OMP_NUM_THREADS OpenMP threads per process"
echo "Total CPU cores: $((SLURM_NTASKS * OMP_NUM_THREADS))"
echo "Problem size: case2001"

srun --mpi=pmi2 ./build/bicgstab data
