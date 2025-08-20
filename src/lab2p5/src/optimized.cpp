
#include "gemm.h"
#include <riscv_vector.h>
#include <stddef.h>

#define BLOCK_M 4
#define BLOCK_N 2

// 4x2 微内核：一次计算 4 行 × 2 列；使用“分段向量 + 逐段归约”策略，避免跨段累加时的 vl 混淆。
static inline void micro_kernel_4x2(uint8_t* A, int8_t* B, int32_t* C,
                                    int n, int k, int i0, int j0) {
    const int K4 = k * 4;          // 每行/列真实元素数

    // 8 个标量累加器（行 × 列）
    int32_t s00=0, s01=0,
            s10=0, s11=0,
            s20=0, s21=0,
            s30=0, s31=0;

    // 指向 4 行 A 以及 2 列 B 起始
    uint8_t* a0 = &A[(i0 + 0) * K4];
    uint8_t* a1 = &A[(i0 + 1) * K4];
    uint8_t* a2 = &A[(i0 + 2) * K4];
    uint8_t* a3 = &A[(i0 + 3) * K4];
    int8_t*  b0 = &B[(j0 + 0) * K4];
    int8_t*  b1 = &B[(j0 + 1) * K4];

    for (int kk = 0; kk < K4; ) {
        size_t vl = __riscv_vsetvl_e8m1((size_t)(K4 - kk));

        // 加载 8bit 源
        vuint8m1_t va0 = __riscv_vle8_v_u8m1(a0 + kk, vl);
        vuint8m1_t va1 = __riscv_vle8_v_u8m1(a1 + kk, vl);
        vuint8m1_t va2 = __riscv_vle8_v_u8m1(a2 + kk, vl);
        vuint8m1_t va3 = __riscv_vle8_v_u8m1(a3 + kk, vl);
        vint8m1_t  vb0 = __riscv_vle8_v_i8m1(b0 + kk, vl);
        vint8m1_t  vb1 = __riscv_vle8_v_i8m1(b1 + kk, vl);

        // (uint8 * int8) -> int16  宽乘：使用 vwmulsu_vv （signed * unsigned）
        vint16m2_t p00 = __riscv_vwmulsu_vv_i16m2(vb0, va0, vl);
        vint16m2_t p01 = __riscv_vwmulsu_vv_i16m2(vb1, va0, vl);
        vint16m2_t p10 = __riscv_vwmulsu_vv_i16m2(vb0, va1, vl);
        vint16m2_t p11 = __riscv_vwmulsu_vv_i16m2(vb1, va1, vl);
        vint16m2_t p20 = __riscv_vwmulsu_vv_i16m2(vb0, va2, vl);
        vint16m2_t p21 = __riscv_vwmulsu_vv_i16m2(vb1, va2, vl);
        vint16m2_t p30 = __riscv_vwmulsu_vv_i16m2(vb0, va3, vl);
        vint16m2_t p31 = __riscv_vwmulsu_vv_i16m2(vb1, va3, vl);

        // (int16) 分别宽归约到 int32 标量
        vint32m1_t zero32 = __riscv_vmv_v_x_i32m1(0, 1);
        // 宽归约：把 int16m2 -> int32m1 并做求和
        vint32m1_t r;
        r = __riscv_vwredsum_vs_i16m2_i32m1(p00, zero32, vl); s00 += __riscv_vmv_x_s_i32m1_i32(r);
        r = __riscv_vwredsum_vs_i16m2_i32m1(p01, zero32, vl); s01 += __riscv_vmv_x_s_i32m1_i32(r);
        r = __riscv_vwredsum_vs_i16m2_i32m1(p10, zero32, vl); s10 += __riscv_vmv_x_s_i32m1_i32(r);
        r = __riscv_vwredsum_vs_i16m2_i32m1(p11, zero32, vl); s11 += __riscv_vmv_x_s_i32m1_i32(r);
        r = __riscv_vwredsum_vs_i16m2_i32m1(p20, zero32, vl); s20 += __riscv_vmv_x_s_i32m1_i32(r);
        r = __riscv_vwredsum_vs_i16m2_i32m1(p21, zero32, vl); s21 += __riscv_vmv_x_s_i32m1_i32(r);
        r = __riscv_vwredsum_vs_i16m2_i32m1(p30, zero32, vl); s30 += __riscv_vmv_x_s_i32m1_i32(r);
        r = __riscv_vwredsum_vs_i16m2_i32m1(p31, zero32, vl); s31 += __riscv_vmv_x_s_i32m1_i32(r);

        kk += (int)vl;
    }

    // 写回 (注意边界)
    C[(i0 + 0) * n + (j0 + 0)] = s00;
    C[(i0 + 0) * n + (j0 + 1)] = s01;
    C[(i0 + 1) * n + (j0 + 0)] = s10;
    C[(i0 + 1) * n + (j0 + 1)] = s11;
    C[(i0 + 2) * n + (j0 + 0)] = s20;
    C[(i0 + 2) * n + (j0 + 1)] = s21;
    C[(i0 + 3) * n + (j0 + 0)] = s30;
    C[(i0 + 3) * n + (j0 + 1)] = s31;
}

void optimized_gemm(uint8_t* A, int8_t* B, int32_t* C, int m, int n, int k) {
    for (int i = 0; i < m; i += BLOCK_M) {
        for (int j = 0; j < n; j += BLOCK_N) {
            micro_kernel_4x2(A, B, C, n, k, i, j);
        }
    }
}