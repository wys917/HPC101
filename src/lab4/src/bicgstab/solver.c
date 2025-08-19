#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>
#include <mpi.h>

int bicgstab(int N_global, double* A_global, double* b_global, double* x_global, int max_iter, double tol) {
    MPI_Comm comm = MPI_COMM_WORLD;
    int world_size, world_rank;
    MPI_Comm_size(comm, &world_size);
    MPI_Comm_rank(comm, &world_rank);

    int N;
    if (world_rank == 0) {
        N = N_global;
    }
    // 1. Broadcast the matrix size N to all processes.
    MPI_Bcast(&N, 1, MPI_INT, 0, comm);

    // 2. All processes calculate the data distribution (counts and displacements).
    int* counts = (int*)malloc(world_size * sizeof(int));
    int* displs = (int*)malloc(world_size * sizeof(int));
    int rows_per_proc = N / world_size;
    int remainder = N % world_size;
    int current_displ = 0;
    for (int i = 0; i < world_size; ++i) {
        counts[i] = rows_per_proc + (i < remainder ? 1 : 0);
        displs[i] = current_displ;
        current_displ += counts[i];
    }
    int local_N = counts[world_rank];
    int offset = displs[world_rank];

    // 3. Allocate memory for local data chunks on each process.
    double* A_local = (double*)malloc(local_N * N * sizeof(double));
    double* b_local = (double*)malloc(local_N * sizeof(double));
    double* x_local = (double*)calloc(local_N, sizeof(double));

    // 4. Rank 0 scatters the global matrix A and vector b to all processes.
    int* counts_A = (int*)malloc(world_size * sizeof(int));
    int* displs_A = (int*)malloc(world_size * sizeof(int));
    for(int i = 0; i < world_size; ++i) {
        counts_A[i] = counts[i] * N;
        displs_A[i] = displs[i] * N;
    }
    MPI_Scatterv(A_global, counts_A, displs_A, MPI_DOUBLE, A_local, local_N * N, MPI_DOUBLE, 0, comm);
    MPI_Scatterv(b_global, counts, displs, MPI_DOUBLE, b_local, local_N, MPI_DOUBLE, 0, comm);
    free(counts_A);
    free(displs_A);

    // --- BiCGSTAB Core Algorithm using Local Data ---
    double* r_local      = (double*)malloc(local_N * sizeof(double));
    double* r_hat_local  = (double*)malloc(local_N * sizeof(double));
    double* p_local      = (double*)malloc(local_N * sizeof(double));
    double* v_local      = (double*)malloc(local_N * sizeof(double));
    double* s_local      = (double*)malloc(local_N * sizeof(double));
    double* h_local      = (double*)malloc(local_N * sizeof(double));
    double* t_local      = (double*)malloc(local_N * sizeof(double));
    double* y_local      = (double*)malloc(local_N * sizeof(double));
    double* z_local      = (double*)malloc(local_N * sizeof(double));
    double* K2_inv_local = (double*)malloc(local_N * sizeof(double));
    double* temp_global_vec = (double*)malloc(N * sizeof(double)); // Re-usable buffer for Allgatherv

    double rho = 1.0, rho_old = 1.0, alpha = 1.0, omega = 1.0, beta = 1.0;
    double tol_squared = tol * tol;

    // Preconditioner (local operation)
    #pragma omp parallel for
    for (int i = 0; i < local_N; i++) K2_inv_local[i] = 1.0 / A_local[i * N + (offset + i)];

    // r0 = b - A*x0
    MPI_Allgatherv(x_local, local_N, MPI_DOUBLE, temp_global_vec, counts, displs, MPI_DOUBLE, comm);
    #pragma omp parallel for
    for (int i = 0; i < local_N; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) sum += A_local[i * N + j] * temp_global_vec[j];
        r_local[i] = b_local[i] - sum;
    }

    memcpy(r_hat_local, r_local, local_N * sizeof(double));
    
    // rho = dot_product(r_hat, r)
    double local_dot = 0.0;
    #pragma omp parallel for reduction(+:local_dot)
    for (int i = 0; i < local_N; i++) local_dot += r_hat_local[i] * r_local[i];
    MPI_Allreduce(&local_dot, &rho, 1, MPI_DOUBLE, MPI_SUM, comm);

    memcpy(p_local, r_local, local_N * sizeof(double));

    int iter;
    for (iter = 1; iter <= max_iter; iter++) {
        if (world_rank == 0 && iter % 1000 == 0) {
            printf("Iteration %d, residual = %e\n", iter, sqrt(rho)); fflush(stdout);
        }

        // y = M_inv * p
        #pragma omp parallel for
        for (int i = 0; i < local_N; i++) y_local[i] = K2_inv_local[i] * p_local[i];

        // v = A*y
        MPI_Allgatherv(y_local, local_N, MPI_DOUBLE, temp_global_vec, counts, displs, MPI_DOUBLE, comm);
        #pragma omp parallel for
        for (int i = 0; i < local_N; i++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++) sum += A_local[i * N + j] * temp_global_vec[j];
            v_local[i] = sum;
        }

        // alpha = rho / dot(r_hat, v)
        local_dot = 0.0;
        #pragma omp parallel for reduction(+:local_dot)
        for (int i = 0; i < local_N; i++) local_dot += r_hat_local[i] * v_local[i];
        double global_dot;
        MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, comm);
        alpha = rho / global_dot;

        // s = r - alpha*v
        #pragma omp parallel for
        for (int i = 0; i < local_N; i++) s_local[i] = r_local[i] - alpha * v_local[i];

        // Check for convergence
        local_dot = 0.0;
        #pragma omp parallel for reduction(+:local_dot)
        for (int i = 0; i < local_N; i++) local_dot += s_local[i] * s_local[i];
        MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, comm);
        if (global_dot < tol_squared) {
            #pragma omp parallel for
            for(int i=0; i < local_N; ++i) x_local[i] = x_local[i] + alpha * y_local[i];
            break;
        }

        // z = M_inv * s
        #pragma omp parallel for
        for (int i = 0; i < local_N; i++) z_local[i] = K2_inv_local[i] * s_local[i];

        // t = A*z
        MPI_Allgatherv(z_local, local_N, MPI_DOUBLE, temp_global_vec, counts, displs, MPI_DOUBLE, comm);
        #pragma omp parallel for
        for (int i = 0; i < local_N; i++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++) sum += A_local[i * N + j] * temp_global_vec[j];
            t_local[i] = sum;
        }

        // omega = dot(t,s) / dot(t,t)
        double local_dot_ts = 0.0, local_dot_tt = 0.0;
        #pragma omp parallel for reduction(+:local_dot_ts, local_dot_tt)
        for (int i = 0; i < local_N; i++) {
            local_dot_ts += t_local[i] * s_local[i];
            local_dot_tt += t_local[i] * t_local[i];
        }
        double global_dots[2];
        MPI_Allreduce(&local_dot_ts, &global_dots[0], 1, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(&local_dot_tt, &global_dots[1], 1, MPI_DOUBLE, MPI_SUM, comm);
        omega = global_dots[0] / global_dots[1];

        // x = x + alpha*y + omega*z
        #pragma omp parallel for
        for (int i = 0; i < local_N; i++) x_local[i] = x_local[i] + alpha * y_local[i] + omega * z_local[i];

        // r = s - omega*t
        #pragma omp parallel for
        for (int i = 0; i < local_N; i++) r_local[i] = s_local[i] - omega * t_local[i];

        // Check for convergence
        local_dot = 0.0;
        #pragma omp parallel for reduction(+:local_dot)
        for (int i = 0; i < local_N; i++) local_dot += r_local[i] * r_local[i];
        MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, comm);
        if (global_dot < tol_squared) break;

        rho_old = rho;
        local_dot = 0.0;
        #pragma omp parallel for reduction(+:local_dot)
        for (int i = 0; i < local_N; i++) local_dot += r_hat_local[i] * r_local[i];
        MPI_Allreduce(&local_dot, &rho, 1, MPI_DOUBLE, MPI_SUM, comm);

        beta = (rho / rho_old) * (alpha / omega);

        // p = r + beta*(p - omega*v)
        #pragma omp parallel for
        for (int i = 0; i < local_N; i++) p_local[i] = r_local[i] + beta * (p_local[i] - omega * v_local[i]);
    }

    // 5. Gather the final result vector x back to rank 0.
    MPI_Gatherv(x_local, local_N, MPI_DOUBLE, x_global, counts, displs, MPI_DOUBLE, 0, comm);

    // 6. Free all allocated memory.
    free(counts); free(displs);
    free(A_local); free(b_local); free(x_local);
    free(r_local); free(r_hat_local); free(p_local); free(v_local); free(s_local);
    free(h_local); free(t_local); free(y_local); free(z_local); free(K2_inv_local);
    free(temp_global_vec);

    if (iter > max_iter) return -1;
    return iter;
}