cmake_minimum_required(VERSION 3.12)
project(bicgstab C CXX Fortran)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

# GNU MPI compiler wrappers
set(CMAKE_C_COMPILER "mpicc")
set(CMAKE_CXX_COMPILER "mpicxx")
set(CMAKE_Fortran_COMPILER "mpifort")

# Set flags
# GNU compiler flags
set(CMAKE_C_FLAGS "-O2 -g -fopenmp -mavx512f")
set(CMAKE_CXX_FLAGS "-O2 -g -fopenmp") 
set(CMAKE_Fortran_FLAGS "-g")

# Include headers
include_directories(include)
include_directories(${MPI_INCLUDE_PATH})

# Add common source files
file(GLOB COMMON_SOURCES "src/*.cpp")

# Create an object library for common sources
add_library(judger OBJECT ${COMMON_SOURCES})

# If found C implementation, add it to the build
if(EXISTS "${CMAKE_SOURCE_DIR}/src/bicgstab/solver.c")
  add_executable(bicgstab src/bicgstab/solver.c $<TARGET_OBJECTS:judger>)
  # Link OpenMP and MPI libs
  target_link_libraries(bicgstab OpenMP::OpenMP_C MPI::MPI_C)
endif()

# If found Fortran implementation, add it to the build
if(EXISTS "${CMAKE_SOURCE_DIR}/src/bicgstab/solver.f90")
  add_executable(bicgstab-fortran src/bicgstab/solver.f90 $<TARGET_OBJECTS:judger>)
endif()