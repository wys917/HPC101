#!/bin/bash
#SBATCH --job-name=scaling_test
#SBATCH --partition=M7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=00:10:00
#SBATCH --output=reports_time/benchmark_%j.out
#SBATCH --error=reports_time/benchmark_%j.err

source /pxe/opt/spack/share/spack/setup-env.sh
# spack load 

mkdir -p reports_time

source /pxe/opt/spack/share/spack/setup-env.sh
spack load intel-oneapi-mpi



export OMP_NUM_THREADS=48

# Run BICGSTAB
./build/bicgstab $1

