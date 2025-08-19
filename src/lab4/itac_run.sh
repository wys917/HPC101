#!/bin/bash
#SBATCH --job-name=solver-itac
#SBATCH --output=run.out
#SBATCH --error=run.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=00:2:00
#SBATCH --partition=solver2

source /pxe/opt/spack/share/spack/setup-env.sh
spack env activate hpc101-intel
spack load intel-oneapi-itac

export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 2))
echo "Node: $(hostname), Rank: ${SLURM_PROCID}, Threads: ${OMP_NUM_THREADS}"

export I_MPI_PMI_LIBRARY=/slurm/libpmi2.so.0.0.0

# (FIXED) Add -bootstrap slurm to correctly integrate with the scheduler
# and remove the unnecessary -hosts flag.
mpirun -n ${SLURM_NTASKS} -ppn ${SLURM_NTASKS_PER_NODE} \
       -bootstrap slurm \
       -trace ./build/bicgstab $1

echo "done"