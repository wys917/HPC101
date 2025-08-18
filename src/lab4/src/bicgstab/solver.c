#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h> // OpenMP for parallelization
#include <mpi.h> 

void gemv(double* __restrict y, double* __restrict A, double* __restrict x, int N) {
    // y = A * x
    for (int i = 0; i < N; i++) {
        y[i] = 0.0;
        for (int j = 0; j < N; j++) {
            y[i] += A[i * N + j] * x[j];
        }
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
    
    double* r      = (double*)calloc(N, sizeof(double));
    double* r_hat  = (double*)calloc(N, sizeof(double));
    double* p      = (double*)calloc(N, sizeof(double));
    double* v      = (double*)calloc(N, sizeof(double));
    double* s      = (double*)calloc(N, sizeof(double));
    double* h      = (double*)calloc(N, sizeof(double));
    double* t      = (double*)calloc(N, sizeof(double));
    double* y      = (double*)calloc(N, sizeof(double));
    double* z      = (double*)calloc(N, sizeof(double));
    double* K2_inv = (double*)calloc(N, sizeof(double));

    double rho_old = 1, alpha = 1, omega = 1;
    double rho = 1, beta = 1;
    double tol_squared = tol * tol;

    int rank, size;
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

    // 验证一下 (这是个好习惯，以后可以删掉):
    if (rank == 1) { // 随便找个非0进程打印一下
        printf("我是进程1, 我收到的 N 是: %d\n", N);
    }

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


    
    // Take M_inv as the preconditioner
    // Note that we only use K2_inv (in wikipedia)
    precondition(A, K2_inv, N);

    // 1. r0 = b - A * x0
    gemv(r, A, x, N);

    for (int i = 0; i < local_n; i++) {
        r_local[i] = b_local[i] - r_local[i];
    }

    // 2. Choose an arbitary vector r_hat that is not orthogonal to r
    // We just take r_hat = r, please do not change this initial value
    memmove(r_hat, r, N * sizeof(double));  // memmove is safer memcpy :)

    // 3. rho_0 = (r_hat, r)
    rho = dot_product(r_hat, r, N);

    // 4. p_0 = r_0
    memmove(p, r, N * sizeof(double));

    int iter;
    for (iter = 1; iter <= max_iter; iter++) {
        if (iter % 1000 == 0) {
            printf("Iteration %d, residul = %e\n", iter, sqrt(dot_product(r, r, N)));
        }

        // 1. y = K2_inv * p (apply preconditioner)
        precondition_apply(y, K2_inv, p, N);

        // 2. v = Ay
        gemv(v, A, y, N);

        // 3. alpha = rho / (r_hat, v)
        alpha = rho / dot_product(r_hat, v, N);

        // 4. h = x_{i-1} + alpha * y
        for (int i = 0; i < local_n; i++) {
            h_local[i] = x_local[i] + alpha * y_local[i];
        }

        // 5. s = r_{i-1} - alpha * v
        for (int i = 0; i < local_n; i++) {
            s_local[i] = r_local[i] - alpha * v_local[i];
        }

        // 6. Is h is accurate enough, then x_i = h and quit
        if (dot_product(s, s, N) < tol_squared) {
            memmove(x, h, N * sizeof(double));
            break;
        }

        // 7. z = K2_inv * s
        precondition_apply(z, K2_inv, s, N);

        // 8. t = Az
        gemv(t, A, z, N);

        // 9. omega = (t, s) / (t, t)
        omega = dot_product(t, s, N) / dot_product(t, t, N);

        // 10. x_i = h + omega * z
        for (int i = 0; i < local_n; i++) {
            x_local[i] = h_local[i] + omega * z_local[i];
        }

        // 11. r_i = s - omega * t
        for (int i = 0; i < local_n; i++) {
            r_local[i] = s_local[i] - omega * t_local[i];
        }

        // 12. If x_i is accurate enough, then quit
        if (dot_product(r, r, N) < tol_squared) break;

        rho_old = rho;
        // 13. rho_i = (r_hat, r)
        rho = dot_product(r_hat, r, N);

        // 14. beta = (rho_i / rho_{i-1}) * (alpha / omega)
        beta = (rho / rho_old) * (alpha / omega);

        // 15. p_i = r_i + beta * (p_{i-1} - omega * v)
        for (int i = 0; i < local_n; i++) {
            p_local[i] = r_local[i] + beta * (p_local[i] - omega * v_local[i]);
        }
    }

    free(r);
    free(r_hat);
    free(p);
    free(v);
    free(s);
    free(h);
    free(t);
    free(y);
    free(z);
    free(K2_inv);
    
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
