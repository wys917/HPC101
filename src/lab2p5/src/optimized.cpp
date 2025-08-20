#include "gemm.h"

// 定义分块大小。4x4是一个常用且高效的选择，
// 它在数据重用和寄存器压力之间取得了很好的平衡。
#define BLOCK_SIZE_M 4
#define BLOCK_SIZE_N 4

/**
 * @brief 进一步优化的矩阵乘法，采用分块和循环展开
 */
void optimized_gemm(uint8_t* A, int8_t* B, int32_t* C, int m, int n, int k) {
    const int K_4 = k * 4;

    // 外层循环：按4x4的块遍历C矩阵
    for (int i_block = 0; i_block < m; i_block += BLOCK_SIZE_M) {
        for (int j_block = 0; j_block < n; j_block += BLOCK_SIZE_N) {

            // 中层循环：在块内逐行计算。一次处理C矩阵的一行（但在块内是4个元素）。
            for (int i = i_block; i < i_block + BLOCK_SIZE_M; i++) {
                
                // --- 循环展开 ---
                // 声明4个累加器，用于同时计算C[i, j_block]到C[i, j_block+3]这4个元素。
                // 优秀的编译器会把这4个累加器直接分配到寄存器中。
                int32_t sum0 = 0;
                int32_t sum1 = 0;
                int32_t sum2 = 0;
                int32_t sum3 = 0;

                // --- 指针优化 ---
                // 使用 restrict 指针告诉编译器这些指针没有别名，让它可以进行更激进的优化。
                // 我们一次性获取当前计算所需的A的一行和B的四行。
                uint8_t* __restrict__ a_row = &A[i * K_4];
                int8_t* __restrict__ b_row0 = &B[(j_block + 0) * K_4];
                int8_t* __restrict__ b_row1 = &B[(j_block + 1) * K_4];
                int8_t* __restrict__ b_row2 = &B[(j_block + 2) * K_4];
                int8_t* __restrict__ b_row3 = &B[(j_block + 3) * K_4];
                
                // --- 内核计算循环 ---
                // 这个循环是性能的关键。它同时为4个输出元素累加结果。
                // 这种结构对编译器的自动矢量化极其友好。
                #pragma clang loop vectorize(enable)
                for (int kk = 0; kk < K_4; kk++) {
                    // 从A加载一次，用于4次计算，最大化数据重用。
                    int32_t a_val = (int32_t)a_row[kk];
                    
                    sum0 += a_val * (int32_t)b_row0[kk];
                    sum1 += a_val * (int32_t)b_row1[kk];
                    sum2 += a_val * (int32_t)b_row2[kk];
                    sum3 += a_val * (int32_t)b_row3[kk];
                }

                // 将计算结果写回C矩阵
                C[i * n + j_block + 0] = sum0;
                C[i * n + j_block + 1] = sum1;
                C[i * n + j_block + 2] = sum2;
                C[i * n + j_block + 3] = sum3;
            }
        }
    }
}