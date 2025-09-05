#include "gemm.h"
#include <riscv_vector.h>
#include <cstddef>

static inline void kernel4x2(uint8_t* A, int8_t* B, int32_t* C,
                             int n, int k, int row0, int col0) {
    const int kk_len = k * 4;

    int32_t s00=0, s01=0,
            s10=0, s11=0,
            s20=0, s21=0,
            s30=0, s31=0;

    uint8_t* aRow0 = &A[(row0 + 0) * kk_len];
    uint8_t* aRow1 = &A[(row0 + 1) * kk_len];
    uint8_t* aRow2 = &A[(row0 + 2) * kk_len];
    uint8_t* aRow3 = &A[(row0 + 3) * kk_len];
    int8_t*  bCol0 = &B[(col0 + 0) * kk_len];
    int8_t*  bCol1 = &B[(col0 + 1) * kk_len];

    for (int kk = 0; kk < kk_len; ) {
        size_t vl = __riscv_vsetvl_e8m1((size_t)(kk_len - kk));

        vuint8m1_t va0 = __riscv_vle8_v_u8m1(aRow0 + kk, vl);
        vuint8m1_t va1 = __riscv_vle8_v_u8m1(aRow1 + kk, vl);
        vuint8m1_t va2 = __riscv_vle8_v_u8m1(aRow2 + kk, vl);
        vuint8m1_t va3 = __riscv_vle8_v_u8m1(aRow3 + kk, vl);
        vint8m1_t  vb0 = __riscv_vle8_v_i8m1(bCol0 + kk, vl);
        vint8m1_t  vb1 = __riscv_vle8_v_i8m1(bCol1 + kk, vl);

        vint16m2_t p00 = __riscv_vwmulsu_vv_i16m2(vb0, va0, vl);
        vint16m2_t p01 = __riscv_vwmulsu_vv_i16m2(vb1, va0, vl);
        vint16m2_t p10 = __riscv_vwmulsu_vv_i16m2(vb0, va1, vl);
        vint16m2_t p11 = __riscv_vwmulsu_vv_i16m2(vb1, va1, vl);
        vint16m2_t p20 = __riscv_vwmulsu_vv_i16m2(vb0, va2, vl);
        vint16m2_t p21 = __riscv_vwmulsu_vv_i16m2(vb1, va2, vl);
        vint16m2_t p30 = __riscv_vwmulsu_vv_i16m2(vb0, va3, vl);
        vint16m2_t p31 = __riscv_vwmulsu_vv_i16m2(vb1, va3, vl);

        vint32m1_t zero32 = __riscv_vmv_v_x_i32m1(0, 1);

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

    C[(row0 + 0) * n + (col0 + 0)] = s00;
    C[(row0 + 0) * n + (col0 + 1)] = s01;
    C[(row0 + 1) * n + (col0 + 0)] = s10;
    C[(row0 + 1) * n + (col0 + 1)] = s11;
    C[(row0 + 2) * n + (col0 + 0)] = s20;
    C[(row0 + 2) * n + (col0 + 1)] = s21;
    C[(row0 + 3) * n + (col0 + 0)] = s30;
    C[(row0 + 3) * n + (col0 + 1)] = s31;
}

void optimized_gemm(uint8_t* A, int8_t* B, int32_t* C, int m, int n, int k) {
    for (int i = 0; i < m; i += 4) {
        for (int j = 0; j < n; j += 2) {
            kernel4x2(A, B, C, n, k, i, j);
        }
    }
}
