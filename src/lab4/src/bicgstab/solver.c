#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h> // OpenMP for parallelization
#include <mpi.h> 

void gemv(double* __restrict y_local, double* __restrict A_local, double* __restrict x, int local_n, int N) {
    // y_local = A_local * x
    // 优化内存访问模式，提高缓存效率
    #pragma omp parallel for
    for (int i = 0; i < local_n; i++) {
        double sum = 0.0;
        double* A_row = &A_local[i * N]; // 获取第i行的指针
        // 内循环现在是连续访问A_row[j]，提高缓存命中率
        for (int j = 0; j < N; j++) {
            sum += A_row[j] * x[j];
        }
        y_local[i] = sum;
    }
}

double dot_product(double* __restrict x, double* __restrict y, int N) {
    // dot product of x and y
    double result = 0.0;

    for (int i = 0; i < N; i++) {
        result += x[i] * y[i];
    }
    return result;
}

void precondition(double* __restrict A, double* __restrict K2_inv, int N) {
    // K2_inv = 1 / diag(A)

    for (int i = 0; i < N; i++) {
        K2_inv[i] = 1.0 / A[i * N + i];
    }
}

void precondition_apply(double* __restrict z, double* __restrict K2_inv, double* __restrict r, int N) {
    // z = K2_inv * r

    for (int i = 0; i < N; i++) {
        z[i] = K2_inv[i] * r[i];
    }
}

int bicgstab(int N, double* A, double* b, double* x, int max_iter, double tol) {
    /**
     * Algorithm: BICGSTAB
     *  r: residual
     *  r_hat: modified residual
     *  p: search direction
     *  K2_inv: preconditioner (We only store the diagonal of K2_inv)
     * Reference: https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
     */
    
    // 声明全局向量（用于 MPI_Allgatherv 的接收缓冲区）
    double* y = NULL;
    double* z = NULL;

    double rho_old = 1, alpha = 1, omega = 1;
    double rho = 1, beta = 1;
    double tol_squared = tol * tol;

    int rank, size;
    // MPI_Init_thread 应该在 main.cpp 里被调用
    // 在这里，我们只获取 rank 和 size
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ================== 你要加的代码在这里 ==================
    // 步骤1: 广播问题规模 N
    // 在调用这个函数时：
    // - 对于 rank 0, N 是从 main 函数传来的正确值。
    // - 对于其他 rank, N 是一个垃圾值。
    // MPI_Bcast 会把 rank 0 的 N 的值，覆盖掉所有其他进程里的 N 的值。
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // =======================================================

    // 在这行代码执行完毕后，村子里的所有进程，从0到size-1，
    // 它们的变量 N 的值就都变成一样的、正确的值了。

    // 广播N后为全局向量分配内存
    y = (double*)calloc(N, sizeof(double));
    z = (double*)calloc(N, sizeof(double));

    // 2. 计算每个进程应该分到多少行
    int rows_per_proc = N / size;
    int remainder_rows = N % size;
    int local_n; // 这是当前进程分到的行数
    int offset;  // 这是当前进程的数据，在原始大矩阵中的起始行号

    if (rank < remainder_rows) {
        // 前面 remainder_rows 个进程，每个多分一行
        local_n = rows_per_proc + 1;
        offset = rank * local_n;
    } else {
        // 剩下的进程，每个分 standard_rows 行
        local_n = rows_per_proc;
        offset = rank * local_n + remainder_rows;
    }

    // 注意，所有向量也要按行进行切分
    double* A_local = NULL;
    double* b_local = NULL;
    double* x_local = NULL; // 我们后面会用x_local来存每个进程负责计算的那部分x
    // ... 其他所有向量r, p, v等等，都需要一个local版本 ...

    // 为本地数据分配内存
    A_local = (double*)calloc(local_n * N, sizeof(double));
    b_local = (double*)calloc(local_n, sizeof(double));
    x_local = (double*)calloc(local_n, sizeof(double));
    
    // 为所有向量创建本地版本
    double* r_local      = (double*)calloc(local_n, sizeof(double));
    double* r_hat_local  = (double*)calloc(local_n, sizeof(double));
    double* p_local      = (double*)calloc(local_n, sizeof(double));
    double* v_local      = (double*)calloc(local_n, sizeof(double));
    double* s_local      = (double*)calloc(local_n, sizeof(double));
    double* h_local      = (double*)calloc(local_n, sizeof(double));
    double* t_local      = (double*)calloc(local_n, sizeof(double));
    double* y_local      = (double*)calloc(local_n, sizeof(double));
    double* z_local      = (double*)calloc(local_n, sizeof(double));
    double* K2_inv_local = (double*)calloc(local_n, sizeof(double));


    // 4. 只有0号进程需要准备发货清单
    // 准备MPI_Scatterv所需的数据分发参数
    // sendcounts: 记录每个进程应该接收多少个元素
    // displs: 记录每个进程的数据在源数组中的起始位置（位移）
    int* sendcounts = NULL;  // 发送数量数组，只有根进程(rank 0)需要分配
    int* displs = NULL;      // 位移数组，只有根进程(rank 0)需要分配

    // 只有根进程(rank 0)需要计算数据分发策略
    if (rank == 0) {
        // 为每个进程分配sendcounts和displs数组
        sendcounts = (int*)malloc(size * sizeof(int));  // size是进程总数
        displs = (int*)malloc(size * sizeof(int));
        
        // 用于追踪当前分配到第几行
        int current_offset_rows = 0;
        
        // 为每个进程计算分配的数据量和起始位置
        for (int i = 0; i < size; i++) {
            // 计算第i个进程应该分配多少行
            // 如果i < remainder_rows，则分配(rows_per_proc + 1)行，否则分配rows_per_proc行
            // 这样可以尽可能平均地分配行数，处理无法整除的情况
            int rows_for_this_proc = (i < remainder_rows) ? rows_per_proc + 1 : rows_per_proc;
            
            // 计算发送给第i个进程的元素数量
            // 由于矩阵A是按行存储的，每行有N个元素，所以总元素数 = 行数 × 列数
            sendcounts[i] = rows_for_this_proc * N;
            
            // 计算第i个进程的数据在原矩阵A中的起始位置（元素索引）
            // current_offset_rows表示前面已经分配的行数
            displs[i] = current_offset_rows * N;
            
            // 更新已分配的行数，为下一个进程做准备
            current_offset_rows += rows_for_this_proc;
        }
    }

    // ========== 第一步：分发矩阵A ==========
    // 使用MPI_Scatterv将矩阵A从根进程分发到所有进程
    // MPI_Scatterv允许发送不同大小的数据块给不同的进程
    MPI_Scatterv(
        A,              // 发送缓冲区：源数据（完整的矩阵A，只有根进程rank 0拥有）
        sendcounts,     // 发送数量数组：每个进程应接收的元素个数（只有根进程需要）
        displs,         // 位移数组：每个进程数据在源数组中的起始位置（只有根进程需要）
        MPI_DOUBLE,     // 发送数据类型：double类型
        A_local,        // 接收缓冲区：每个进程接收数据的本地数组
        local_n * N,    // 接收数量：当前进程期望接收的元素个数
        MPI_DOUBLE,     // 接收数据类型：double类型
        0,              // 根进程：指定哪个进程是数据源（这里是进程0）
        MPI_COMM_WORLD  // 通信器：指定参与通信的进程组（所有进程）
    );

    // ========== 第二步：重新计算向量b的分发参数 ==========
    // 向量b的分发策略与矩阵A不同，因为b是一维向量
    // 需要重新计算sendcounts和displs数组
    if (rank == 0) {
        int current_offset_rows = 0;
        
        // 为每个进程重新计算向量b的分配参数
        for (int i = 0; i < size; i++) {
            // 对于向量b，每个进程接收的元素数量等于分配给它的行数
            // （因为向量b的每个元素对应矩阵A的一行）
            sendcounts[i] = (i < remainder_rows) ? rows_per_proc + 1 : rows_per_proc;
            
            // 向量b中每个进程数据的起始位置就是行偏移量
            // （不需要乘以N，因为b是一维向量）
            displs[i] = current_offset_rows;
            
            // 更新偏移量
            current_offset_rows += sendcounts[i];
        }
    }
    
    // ========== 第三步：分发向量b ==========
    // 使用相同的MPI_Scatterv函数分发向量b
    // 注意：这里使用了重新计算的sendcounts和displs数组
    MPI_Scatterv(
        b,              // 发送缓冲区：源向量b（只有根进程拥有完整的b）
        sendcounts,     // 发送数量数组：每个进程应接收的元素个数
        displs,         // 位移数组：每个进程数据在向量b中的起始位置
        MPI_DOUBLE,     // 发送数据类型
        b_local,        // 接收缓冲区：每个进程的本地向量b片段
        local_n,        // 接收数量：当前进程的向量b片段大小
        MPI_DOUBLE,     // 接收数据类型
        0,              // 根进程
        MPI_COMM_WORLD  // 通信器
    );

    // ========== 第四步：清理内存 ==========
    // 释放sendcounts和displs数组的内存
    // 只有根进程分配了这些数组，所以只有根进程需要释放
    if (rank == 0) {
        free(sendcounts);  // 释放发送数量数组
        free(displs);      // 释放位移数组
    }

    // ========== 第五步：广播初始向量x ==========
    // 使用MPI_Bcast将向量x从根进程广播给所有进程
    // 这是必要的，因为在迭代求解过程中，每个进程都需要完整的x向量
    // 来计算矩阵向量乘法 A_local * x
    MPI_Bcast(
        x,              // 数据缓冲区：要广播的向量x
        N,              // 元素个数：向量x的长度
        MPI_DOUBLE,     // 数据类型
        0,              // 根进程：数据源进程（进程0）
        MPI_COMM_WORLD  // 通信器：所有进程都参与
    );

    // ========== 关键修复：初始化x_local ==========
    // 从完整的x向量中复制当前进程负责的部分到x_local
    for (int i = 0; i < local_n; i++) {
        x_local[i] = x[offset + i];
    }


    
    // Take M_inv as the preconditioner
    // Note that we only use K2_inv (in wikipedia)
    // 计算本地预条件子：K2_inv_local = 1 / diag(A_local)
    for (int i = 0; i < local_n; i++) {
        K2_inv_local[i] = 1.0 / A_local[i * N + (offset + i)];
    }

    // 1. r0 = b - A * x0 (compute local residual)
    // First compute A_local * x into a temporary local array
    double* Ax_local = (double*)calloc(local_n, sizeof(double));
    gemv(Ax_local, A_local, x, local_n, N);

    for (int i = 0; i < local_n; i++) {
        r_local[i] = b_local[i] - Ax_local[i];
    }
    
    free(Ax_local);

    // 2. r_hat_local = r_local (本地操作)
    memmove(r_hat_local, r_local, local_n * sizeof(double));

    // 3. 并行计算 rho = (r_hat_local, r_local)
    double local_rho = 0.0;
    for (int i = 0; i < local_n; i++) {
        local_rho += r_hat_local[i] * r_local[i];
    }
    
 
    MPI_Allreduce(&local_rho, &rho, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // 4. p_0 = r_0 (本地操作)
    memmove(p_local, r_local, local_n * sizeof(double));

    int iter;
    // 在主循环开始前，需要先组合本地向量
    // 创建用于MPI通信的缓冲区 - 所有进程都需要这些数组
    int* recvcounts = (int*)malloc(size * sizeof(int));
    int* displs_gather = (int*)malloc(size * sizeof(int));
    
    // 所有进程都计算相同的 recvcounts 和 displs_gather
    int current_offset = 0;
    for (int i = 0; i < size; i++) {
        recvcounts[i] = (i < remainder_rows) ? rows_per_proc + 1 : rows_per_proc;
        displs_gather[i] = current_offset;
        current_offset += recvcounts[i];
    }
    
    for (iter = 1; iter <= max_iter; iter++) {
        if (iter % 5000 == 0 && rank == 0) {
            // 减少打印频率，从每1000次改为每5000次
            double local_r_print = 0.0;
            for (int i = 0; i < local_n; i++) {
                local_r_print += r_local[i] * r_local[i];
            }
            double global_r_print;
            MPI_Allreduce(&local_r_print, &global_r_print, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            printf("Iteration %d, residul = %e\n", iter, sqrt(global_r_print));
        }

        // 1. 本地计算 y_local
        #pragma omp parallel for
        for (int i = 0; i < local_n; i++) {
            y_local[i] = K2_inv_local[i] * p_local[i];
        }

        MPI_Allgatherv(y_local, local_n, MPI_DOUBLE, y, recvcounts, displs_gather, MPI_DOUBLE, MPI_COMM_WORLD);

        // 2. 本地计算 v_local
        gemv(v_local, A_local, y, local_n, N);

        // 3. alpha计算
        double local_dot_rv = 0.0;
        #pragma omp parallel for reduction(+:local_dot_rv)
        for (int i = 0; i < local_n; i++) {
            local_dot_rv += r_hat_local[i] * v_local[i];
        }
        
        double global_dot_rv;
        MPI_Allreduce(&local_dot_rv, &global_dot_rv, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        alpha = rho / global_dot_rv;

        // 4. h = x_{i-1} + alpha * y
        #pragma omp parallel for
        for (int i = 0; i < local_n; i++) {
            h_local[i] = x_local[i] + alpha * y_local[i];
        }

        // 5. s = r_{i-1} - alpha * v
        #pragma omp parallel for
        for (int i = 0; i < local_n; i++) {
            s_local[i] = r_local[i] - alpha * v_local[i];
        }

        // 6. 收敛性检查
        double local_s_norm = 0.0;
        #pragma omp parallel for reduction(+:local_s_norm)
        for (int i = 0; i < local_n; i++) {
            local_s_norm += s_local[i] * s_local[i];
        }
        
        double global_s_norm;
        MPI_Allreduce(&local_s_norm, &global_s_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        if (global_s_norm < tol_squared) {
            MPI_Allgatherv(h_local, local_n, MPI_DOUBLE, x, recvcounts, displs_gather, MPI_DOUBLE, MPI_COMM_WORLD);
            break;
        }

        // 7. z_local = K2_inv_local * s_local
        #pragma omp parallel for
        for (int i = 0; i < local_n; i++) {
            z_local[i] = K2_inv_local[i] * s_local[i];
        }
        
        MPI_Allgatherv(z_local, local_n, MPI_DOUBLE, z, recvcounts, displs_gather, MPI_DOUBLE, MPI_COMM_WORLD);

        // 8. t_local = A_local * z
        gemv(t_local, A_local, z, local_n, N);

        // 9. omega计算
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

        // 10. x_i = h + omega * z
        #pragma omp parallel for
        for (int i = 0; i < local_n; i++) {
            x_local[i] = h_local[i] + omega * z_local[i];
        }

        // 11. r_i = s - omega * t
        #pragma omp parallel for
        for (int i = 0; i < local_n; i++) {
            r_local[i] = s_local[i] - omega * t_local[i];
        }

        // 12. 第二次收敛性检查
        double local_r_norm = 0.0;
        #pragma omp parallel for reduction(+:local_r_norm)
        for (int i = 0; i < local_n; i++) {
            local_r_norm += r_local[i] * r_local[i];
        }
        
        double global_r_norm;
        MPI_Allreduce(&local_r_norm, &global_r_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        if (global_r_norm < tol_squared) break;

        // 13. rho计算 - 先保存旧值，然后计算新值
        rho_old = rho; // <-- 正确：赋值给已经声明的rho_old
        double local_rho_new = 0.0;
        #pragma omp parallel for reduction(+:local_rho_new)
        for (int i = 0; i < local_n; i++) {
            local_rho_new += r_hat_local[i] * r_local[i];
        }
        
        MPI_Allreduce(&local_rho_new, &rho, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // 14. beta = (rho_i / rho_{i-1}) * (alpha / omega)
        beta = (rho / rho_old) * (alpha / omega);

        // 15. p_i = r_i + beta * (p_{i-1} - omega * v)
        #pragma omp parallel for
        for (int i = 0; i < local_n; i++) {
            p_local[i] = r_local[i] + beta * (p_local[i] - omega * v_local[i]);
        }
    }

    // 收集最终的x_local到完整的x向量
    MPI_Allgatherv(x_local, local_n, MPI_DOUBLE,
                   x, recvcounts, displs_gather, MPI_DOUBLE, MPI_COMM_WORLD);
    
    // 清理MPI通信缓冲区
    free(recvcounts);
    free(displs_gather);

    // 释放全局向量
    free(y);
    free(z);
    
    // 释放本地数组
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

    if (iter >= max_iter)
        return -1;
    else
        return iter;
}
