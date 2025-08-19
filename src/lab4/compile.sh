#!/bin/bash

# 先加载 spack 再加载编译环境
# e.g. 
#source ${HOME}/spack/share/spack/setup-env.sh

source /pxe/opt/spack/share/spack/setup-env.sh
#spack env activate hpc101-intel
spack load intel-oneapi-mpi
# 运行你的命令
cmake -B build
cmake --build build

