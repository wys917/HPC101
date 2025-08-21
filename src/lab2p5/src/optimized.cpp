#include <cstdint>
#include <cstddef>
#include <vector> // 为了使用 std::vector 来创建临时转置矩阵
#include <riscv_vector.h>

#define IME_BLOCK_M 4
#define IME_BLOCK_N 4
#define IME_BLOCK_K 8

/**
 * @brief 最终修正版的4x4微内核。
 * 该内核基于以下“正确”的假设：
 * 1. B矩阵已经是转置的，所以我们可以高效地加载数据。
 * 2. vmadotus指令本身就是累加操作 (vd += A*B)。
 * 3. K维度是展开后的总长度 k_total。
 */
static inline void micro_kernel_4x4_ime(
    const uint8_t* A, 
    const int8_t* B_T, // 注意：接收的是转置后的B矩阵
    int32_t* C,
    int n,              // C的列数
    int k_total,        // 展开后的K维度
    int i0, 
    int j0
) {
    vint32m2_t vacc;
    size_t vl_32 = __riscv_vsetvl_e32m2(16);
    vacc = __riscv_vmv_v_x_i32m2(0, vl_32);

    for (int kk = 0; kk < k_total; kk += IME_BLOCK_K) {
        size_t vl_64 = __riscv_vsetvl_e64m1(4);

        const uint8_t* a_ptr = &A[i0 * k_total + kk];
        vuint64m1_t va_64 = __riscv_vlse64_v_u64m1(reinterpret_cast<const uint64_t*>(a_ptr), k_total * sizeof(uint8_t), vl_64);

        const int8_t* b_ptr = &B_T[j0 * k_total + kk];
        vint64m1_t vb_64 = __riscv_vlse64_v_i64m1(reinterpret_cast<const int64_t*>(b_ptr), k_total * sizeof(int8_t), vl_64);

        // --- 就是这两行，我之前手滑了 ---
        vuint8m1_t va_8 = __riscv_vreinterpret_v_u64m1_u8m1(va_64);
        vint8m1_t  vb_8 = __riscv_vreinterpret_v_i64m1_i8m1(vb_64);
        // --------------------------------

        asm volatile(
            "vmadotus %[vd], %[vs1], %[vs2]\n\t"
            : [vd] "+vr"(vacc)
            : [vs1] "vr"(va_8), [vs2] "vr"(vb_8)
        );
    }

    int32_t temp_c[16];
    __riscv_vsetvl_e32m2(16);
    __riscv_vse32_v_i32m2(temp_c, vacc, 16);

    for (int row_offset = 0; row_offset < IME_BLOCK_M; ++row_offset) {
        for (int col_offset = 0; col_offset < IME_BLOCK_N; ++col_offset) {
            C[(i0 + row_offset) * n + (j0 + col_offset)] = temp_c[row_offset * IME_BLOCK_N + col_offset];
        }
    }
}

// 最终版的顶层函数
// 假设B已经是转置的，这是正确的顶层函数
void optimized_gemm(uint8_t* A, int8_t* B, int32_t* C, int m, int n, int k) {
    // 1. K维度依然要修正
    const int k_total = k * 4;

    // 2. B矩阵已经是转置的，所以不需要任何操作！
    //    直接把它当成 B_T 来用。

    // 3. 在C矩阵上进行分块计算
    for (int i = 0; i < m; i += IME_BLOCK_M) {
        for (int j = 0; j < n; j += IME_BLOCK_N) {
            // 直接把输入的 B 指针传给微内核，因为它就是我们想要的 B_T
            micro_kernel_4x4_ime(A, B, C, n, k_total, i, j);
        }
    }
}