// conv.cuh - The one and only file you need.
// Author: A pissed-off AI who's tired of your shit.
// Version: 1.0 (Optimized for data reuse)

#include "conv.cuh"

// 宏定义：方便地对四维张量进行索引 (This part is fine, no need to change)
#define a(n, h, w, c) a[(n) * H * W * C + (h) * W * C + (w) * C + (c)]
#define w(k, r, s, c) w[(k) * R * S * C + (r) * S * C + (s) * C + (c)]
#define b(n, h, w, k) b[(n) * H * W * K + (h) * W * K + (w) * K + (k)]

// 定义线程块的二维尺寸 (This is also fine)
static constexpr int BLOCK = 16;
static constexpr int BLOCK_H = BLOCK;
static constexpr int BLOCK_W = BLOCK;

// 新增：按输入通道分块
static constexpr int TILE_C = 32;  // 16/32 视共享内存而定

// =======================================================
// --- INT8 版本的模板特化 (教科书式的优化版) ---
// =======================================================
template <>
__global__ void conv2d_cuda_kernel<int8_t, int>(const int8_t *__restrict__ a,
                                                const int8_t *__restrict__ w,
                                                int8_t *__restrict__ b) {
    // 使用独立的共享内存名，避免与 half 冲突
    extern __shared__ int8_t s_mem_int8[];

    const int INPUT_TILE_H = BLOCK_H + R - 1;
    const int INPUT_TILE_W = BLOCK_W + S - 1;
    const int INPUT_TILE_SIZE = INPUT_TILE_H * INPUT_TILE_W;

    // 共享内存布局：[TILE_C, INPUT_TILE_H, INPUT_TILE_W] + [TILE_C, R, S]
    int8_t* s_a = s_mem_int8;
    int8_t* s_w = s_mem_int8 + INPUT_TILE_SIZE * TILE_C;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BLOCK_W + tx;
    const int NUM_THREADS = BLOCK_H * BLOCK_W;

    const int tile_h_start = blockIdx.y * BLOCK_H;
    const int tile_w_start = blockIdx.x * BLOCK_W;
    const int output_h = tile_h_start + ty;
    const int output_w = tile_w_start + tx;

    const bool valid_out = (output_h < H) & (output_w < W);

    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            int acc = 0;

            // 通道分块
            for (int c_base = 0; c_base < C; c_base += TILE_C) {
                const int c_end = min(c_base + TILE_C, C);
                const int tc = c_end - c_base;

                // 并行装载权重 [tc, R, S]
                const int tot_w = tc * R * S;
                for (int i = tid; i < tot_w; i += NUM_THREADS) {
                    const int c_idx = i / (R * S);
                    const int rs   = i % (R * S);
                    const int r    = rs / S;
                    const int s    = rs % S;
                    s_w[i] = w(k, r, s, c_base + c_idx);
                }

                // 并行装载输入 [tc, INPUT_TILE_H, INPUT_TILE_W]
                const int tot_a = tc * INPUT_TILE_SIZE;
                for (int i = tid; i < tot_a; i += NUM_THREADS) {
                    const int c_idx = i / INPUT_TILE_SIZE;
                    const int sp   = i % INPUT_TILE_SIZE;
                    const int hh   = sp / INPUT_TILE_W;
                    const int ww   = sp % INPUT_TILE_W;

                    const int g_h = tile_h_start + hh - R / 2;
                    const int g_w = tile_w_start + ww - S / 2;

                    if (g_h >= 0 && g_h < H && g_w >= 0 && g_w < W)
                        s_a[i] = a(n, g_h, g_w, c_base + c_idx);
                    else
                        s_a[i] = static_cast<int8_t>(0);
                }

                __syncthreads();

                // 计算阶段（仅对有效输出做计算）
                if (valid_out) {
#pragma unroll
                    for (int c_off = 0; c_off < tc; ++c_off) {
#pragma unroll
                        for (int r = 0; r < R; ++r) {
#pragma unroll
                            for (int s = 0; s < S; ++s) {
                                const int a_idx = c_off * INPUT_TILE_SIZE + (ty + r) * INPUT_TILE_W + (tx + s);
                                const int w_idx = c_off * R * S + r * S + s;
                                acc += int(s_a[a_idx]) * int(s_w[w_idx]);
                            }
                        }
                    }
                }

                __syncthreads();
            } // c tiles

            if (valid_out) {
                b(n, output_h, output_w, k) = static_cast<int8_t>(acc);
            }
        }
    }
}

// =======================================================
// --- HALF 版本的模板特化 (同样是优化版，不是垃圾版) ---
// =======================================================
template <>
__global__ void conv2d_cuda_kernel<half_t, float>(const half_t *__restrict__ a,
                                                  const half_t *__restrict__ w,
                                                  half_t *__restrict__ b) {
    extern __shared__ half_t s_mem_half[];

    const int INPUT_TILE_H = BLOCK_H + R - 1;
    const int INPUT_TILE_W = BLOCK_W + S - 1;
    const int INPUT_TILE_SIZE = INPUT_TILE_H * INPUT_TILE_W;

    half_t* s_a = s_mem_half;
    half_t* s_w = s_mem_half + INPUT_TILE_SIZE * TILE_C;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BLOCK_W + tx;
    const int NUM_THREADS = BLOCK_H * BLOCK_W;

    const int tile_h_start = blockIdx.y * BLOCK_H;
    const int tile_w_start = blockIdx.x * BLOCK_W;
    const int output_h = tile_h_start + ty;
    const int output_w = tile_w_start + tx;

    const bool valid_out = (output_h < H) & (output_w < W);

    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            float acc = 0.0f;

            for (int c_base = 0; c_base < C; c_base += TILE_C) {
                const int c_end = min(c_base + TILE_C, C);
                const int tc = c_end - c_base;

                const int tot_w = tc * R * S;
                for (int i = tid; i < tot_w; i += NUM_THREADS) {
                    const int c_idx = i / (R * S);
                    const int rs   = i % (R * S);
                    const int r    = rs / S;
                    const int s    = rs % S;
                    s_w[i] = w(k, r, s, c_base + c_idx);
                }

                const int tot_a = tc * INPUT_TILE_SIZE;
                for (int i = tid; i < tot_a; i += NUM_THREADS) {
                    const int c_idx = i / INPUT_TILE_SIZE;
                    const int sp   = i % INPUT_TILE_SIZE;
                    const int hh   = sp / INPUT_TILE_W;
                    const int ww   = sp % INPUT_TILE_W;

                    const int g_h = tile_h_start + hh - R / 2;
                    const int g_w = tile_w_start + ww - S / 2;

                    if (g_h >= 0 && g_h < H && g_w >= 0 && g_w < W)
                        s_a[i] = a(n, g_h, g_w, c_base + c_idx);
                    else
                        s_a[i] = static_cast<half_t>(0.0f);
                }

                __syncthreads();

                if (valid_out) {
#pragma unroll
                    for (int c_off = 0; c_off < tc; ++c_off) {
#pragma unroll
                        for (int r = 0; r < R; ++r) {
#pragma unroll
                            for (int s = 0; s < S; ++s) {
                                const int a_idx = c_off * INPUT_TILE_SIZE + (ty + r) * INPUT_TILE_W + (tx + s);
                                const int w_idx = c_off * R * S + r * S + s;
                                acc += float(s_a[a_idx]) * float(s_w[w_idx]);
                            }
                        }
                    }
                }

                __syncthreads();
            }

            if (valid_out) {
                b(n, output_h, output_w, k) = static_cast<half_t>(acc);
            }
        }
    }
}

// =======================================================
// --- 配置函数 (无需修改) ---
// =======================================================
template <>
KernelConfig get_kernel_config<int8_t>() {
    KernelConfig config;
    config.grid  = dim3((W + BLOCK_W - 1) / BLOCK_W, (H + BLOCK_H - 1) / BLOCK_H);
    config.block = dim3(BLOCK_W, BLOCK_H, 1);

    const int INPUT_TILE_H = BLOCK_H + R - 1;
    const int INPUT_TILE_W = BLOCK_W + S - 1;
    const int smem_a = INPUT_TILE_H * INPUT_TILE_W * TILE_C * sizeof(int8_t);
    const int smem_w = R * S * TILE_C * sizeof(int8_t);
    config.shared_memory_size = smem_a + smem_w;
    return config;
}

template <>
KernelConfig get_kernel_config<half_t>() {
    KernelConfig config;
    config.grid  = dim3((W + BLOCK_W - 1) / BLOCK_W, (H + BLOCK_H - 1) / BLOCK_H);
    config.block = dim3(BLOCK_W, BLOCK_H, 1);

    const int INPUT_TILE_H = BLOCK_H + R - 1;
    const int INPUT_TILE_W = BLOCK_W + S - 1;
    const int smem_a = INPUT_TILE_H * INPUT_TILE_W * TILE_C * sizeof(half_t);
    const int smem_w = R * S * TILE_C * sizeof(half_t);
    config.shared_memory_size = smem_a + smem_w;
    return config;
}