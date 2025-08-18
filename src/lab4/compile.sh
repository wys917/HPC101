#!/bin/bash

# 先加载 spack 再加载编译环境
# e.g.
 source /pxe/opt/spack/share/spack/setup-env.sh
 spack load intel-oneapi-mpi

# ================== 环境验尸代码 ==================
echo "--- Verifying MPI environment ---"
echo "Found mpicc at: $(which mpicc)"
echo "Found mpicxx at: $(which mpicxx)"
echo "---------------------------------"
# ================================================

# 运行你的命令
cmake -B build
cmake --build build
