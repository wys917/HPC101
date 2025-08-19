#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

// 辅助的gemv函数，现在也用OpenMP并行化了
void gemv(double* __restrict y_local, double* __restrict A_local, double* __restrict x, int local_n, int N) {
    #pragma omp parallel for
    for (int i = 0; i < local_n; i++) {
        double sum = 0.0;
        double* A_row = &A_local[i * N];
        for (int j = 0; j < N; j++) {
            sum += A_row[j] * x[j];
        }
        y_local[i] = sum;
    }
}

int bicgstab(int N, double* A, double* b, double* x, int max_iter, double tol) {
    // 全局通信所需的临时向量
    double* y = NULL;
    double* z = NULL;

    // 算法所需的标量
    double rho_old = 1.0, alpha = 1.0, omega = 1.0;
    double rho = 1.0, beta = 1.0;
    double tol_squared = tol * tol;

    // 获取MPI环境信息
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. 广播问题规模 N
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 广播N之后，为全局通信缓冲区分配内存
    y = (double*)calloc(N, sizeof(double));
    z = (double*)calloc(N, sizeof(double));

    // 2. 计算每个进程的工作负载
    int rows_per_proc = N / size;
    int remainder_rows = N % size;
    int local_n;
    int offset;

    if (rank < remainder_rows) {
        local_n = rows_per_proc + 1;
        offset = rank * local_n;
    } else {
        local_n = rows_per_proc;
        offset = rank * local_n + remainder_rows;
    }

    // 3. 为所有本地数据分配内存
    double* A_local = (double*)calloc(local_n * N, sizeof(double));
    double* b_local = (double*)calloc(local_n, sizeof(double));
    double* x_local = (double*)calloc(local_n, sizeof(double));
    double* r_local = (double*)calloc(local_n, sizeof(double));
    double* r_hat_local = (double*)calloc(local_n, sizeof(double));
    double* p_local = (double*)calloc(local_n, sizeof(double));
    double* v_local = (double*)calloc(local_n, sizeof(double));
    double* s_local = (double*)calloc(local_n, sizeof(double));
    double* h_local = (double*)calloc(local_n, sizeof(double));
    double* t_local = (double*)calloc(local_n, sizeof(double));
    double* y_local = (double*)calloc(local_n, sizeof(double));
    double* z_local = (double*)calloc(local_n, sizeof(double));
    double* K2_inv_local = (double*)calloc(local_n, sizeof(double));

    // 4. 准备数据分发清单 (仅rank 0)
    int* sendcounts = NULL;
    int* displs = NULL;

    if (rank == 0) {
        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
    }

    // 5. 分发矩阵A和向量b
    if (rank == 0) {
        int current_offset_rows = 0;
        for (int i = 0; i < size; i++) {
            int rows_for_this_proc = (i < remainder_rows) ? rows_per_proc + 1 : rows_per_proc;
            sendcounts[i] = rows_for_this_proc * N;
            displs[i] = current_offset_rows * N;
            current_offset_rows += rows_for_this_proc;
        }
    }
    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE, A_local, local_n * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int current_offset_rows = 0;
        for (int i = 0; i < size; i++) {
            sendcounts[i] = (i < remainder_rows) ? rows_per_proc + 1 : rows_per_proc;
            displs[i] = current_offset_rows;
            current_offset_rows += sendcounts[i];
        }
    }
    MPI_Scatterv(b, sendcounts, displs, MPI_DOUBLE, b_local, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        free(sendcounts);
        free(displs);
    }
    
    // ========== 关键修复：为非0号进程分配x向量内存 ==========
    if (rank != 0) {
        x = (double*)calloc(N, sizeof(double));
    }
    
    // 6. 广播初始解x，并初始化x_local
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    #pragma omp parallel for
    for (int i = 0; i < local_n; i++) {
        x_local[i] = x[offset + i];
    }
    
    // ======== 算法初始化 ========
    
    // 计算本地预条件子
    #pragma omp parallel for
    for (int i = 0; i < local_n; i++) {
        K2_inv_local[i] = 1.0 / A_local[i * N + (offset + i)];
    }

    // 计算初始残差 r_local = b_local - A_local * x
    double* Ax_local = (double*)calloc(local_n, sizeof(double));
    gemv(Ax_local, A_local, x, local_n, N);
    #pragma omp parallel for
    for (int i = 0; i < local_n; i++) {
        r_local[i] = b_local[i] - Ax_local[i];
    }
    free(Ax_local);

    // r_hat_local = r_local
    memcpy(r_hat_local, r_local, local_n * sizeof(double));

    // rho = (r_hat, r)
    double local_rho = 0.0;
    #pragma omp parallel for reduction(+:local_rho)
    for (int i = 0; i < local_n; i++) {
        local_rho += r_hat_local[i] * r_local[i];
    }
    MPI_Allreduce(&local_rho, &rho, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    // p_local = r_local
    memcpy(p_local, r_local, local_n * sizeof(double));

    // 准备Allgatherv所需的清单 (所有进程都需要)
    int* recvcounts = (int*)malloc(size * sizeof(int));
    int* displs_gather = (int*)malloc(size * sizeof(int));
    int current_offset = 0;
    for (int i = 0; i < size; i++) {
        recvcounts[i] = (i < remainder_rows) ? rows_per_proc + 1 : rows_per_proc;
        displs_gather[i] = current_offset;
        current_offset += recvcounts[i];
    }
    
    // ======== 主迭代循环 ========
    int iter;
    for (iter = 1; iter <= max_iter; iter++) {
        // 1. y_local = K_inv_local * p_local
        #pragma omp parallel for
        for (int i = 0; i < local_n; i++) {
            y_local[i] = K2_inv_local[i] * p_local[i];
        }
        
        // 收集y
        MPI_Allgatherv(y_local, local_n, MPI_DOUBLE, y, recvcounts, displs_gather, MPI_DOUBLE, MPI_COMM_WORLD);

        // 2. v_local = A_local * y
        gemv(v_local, A_local, y, local_n, N);

        // 3. alpha = rho / (r_hat, v)
        double local_dot_rv = 0.0;
        #pragma omp parallel for reduction(+:local_dot_rv)
        for (int i = 0; i < local_n; i++) {
            local_dot_rv += r_hat_local[i] * v_local[i];
        }
        double global_dot_rv;
        MPI_Allreduce(&local_dot_rv, &global_dot_rv, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        alpha = rho / global_dot_rv;

        // 4. h_local = x_local + alpha * y_local
        #pragma omp parallel for
        for (int i = 0; i < local_n; i++) {
            h_local[i] = x_local[i] + alpha * y_local[i];
        }

        // 5. s_local = r_local - alpha * v_local
        #pragma omp parallel for
        for (int i = 0; i < local_n; i++) {
            s_local[i] = r_local[i] - alpha * v_local[i];
        }

        // 6. 第一次收敛检查
        double local_s_norm = 0.0;
        #pragma omp parallel for reduction(+:local_s_norm)
        for (int i = 0; i < local_n; i++) {
            local_s_norm += s_local[i] * s_local[i];
        }
        double global_s_norm;
        MPI_Allreduce(&local_s_norm, &global_s_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        if (global_s_norm < tol_squared) {
            memcpy(x_local, h_local, local_n * sizeof(double));
            break;
        }

        // 7. z_local = K_inv_local * s_local
        #pragma omp parallel for
        for (int i = 0; i < local_n; i++) {
            z_local[i] = K2_inv_local[i] * s_local[i];
        }
        
        // 收集z
        MPI_Allgatherv(z_local, local_n, MPI_DOUBLE, z, recvcounts, displs_gather, MPI_DOUBLE, MPI_COMM_WORLD);

        // 8. t_local = A_local * z
        gemv(t_local, A_local, z, local_n, N);

        // 9. omega = (t,s) / (t,t)
        double local_ts = 0.0, local_tt = 0.0;
        #pragma omp parallel for reduction(+:local_ts,local_tt)
        for (int i = 0; i < local_n; i++) {
            local_ts += t_local[i] * s_local[i];
            local_tt += t_local[i] * t_local[i];
        }
        double global_ts, global_tt;
        MPI_Allreduce(&local_ts, &global_ts, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_tt, &global_tt, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        omega = global_ts / global_tt;

        // 10. x_local = h_local + omega * z_local
        #pragma omp parallel for
        for (int i = 0; i < local_n; i++) {
            x_local[i] = h_local[i] + omega * z_local[i];
        }

        // 11. r_local = s_local - omega * t_local
        #pragma omp parallel for
        for (int i = 0; i < local_n; i++) {
            r_local[i] = s_local[i] - omega * t_local[i];
        }

        // 12. 第二次收敛检查
        double local_r_norm = 0.0;
        #pragma omp parallel for reduction(+:local_r_norm)
        for (int i = 0; i < local_n; i++) {
            local_r_norm += r_local[i] * r_local[i];
        }
        double global_r_norm;
        MPI_Allreduce(&local_r_norm, &global_r_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (global_r_norm < tol_squared) break;

        // 13. rho_new = (r_hat, r_new)
        rho_old = rho;
        double local_rho_new = 0.0;
        #pragma omp parallel for reduction(+:local_rho_new)
        for (int i = 0; i < local_n; i++) {
            local_rho_new += r_hat_local[i] * r_local[i];
        }
        MPI_Allreduce(&local_rho_new, &rho, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // 14. beta = ...
        beta = (rho / rho_old) * (alpha / omega);

        // 15. p_local = r_local + beta * (p_local - omega * v_local)
        #pragma omp parallel for
        for (int i = 0; i < local_n; i++) {
            p_local[i] = r_local[i] + beta * (p_local[i] - omega * v_local[i]);
        }
    }

    // 收集最终的x_local到完整的x向量 (用Gatherv更高效)
    MPI_Gatherv(x_local, local_n, MPI_DOUBLE, x, recvcounts, displs_gather, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // ======== 内存释放 ========
    free(recvcounts);
    free(displs_gather);
    free(y);
    free(z);
    
    free(A_local);
    free(b_local);
    free(x_local);
    free(r_local);
    free(r_hat_local);
    free(p_local);
    free(v_local);
    free(s_local);
    free(h_local);
    free(t_local);
    free(y_local);
    free(z_local);
    free(K2_inv_local);
    
    // ========== 释放为非0号进程分配的x向量内存 ==========
    if (rank != 0) {
        free(x);
    }

    if (iter > max_iter) return -1;
    return iter;
}