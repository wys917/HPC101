#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 加上fflush确保立刻输出
    printf("Hello from the REAL rank %d of %d!\n", rank, size);
    fflush(stdout);

    MPI_Finalize();
    return 0;
}