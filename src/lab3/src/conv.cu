
#include "conv.cuh"

#define a(n, h, w, c) a[(n) * H * W * C + (h) * W * C + (w) * C + (c)]
#define w(k, r, s, c) w[(k) * R * S * C + (r) * S * C + (s) * C + (c)]
#define b(n, h, w, k) b[(n) * H * W * K + (h) * W * K + (w) * K + (k)]

static constexpr int BLOCK = 16;
static constexpr int BLOCK_H = BLOCK;
static constexpr int BLOCK_W = BLOCK;

// =======================================================
// --- INT8 版本 ---
// =======================================================
template <>
__global__ void conv2d_cuda_kernel<int8_t, int>(const int8_t *__restrict__ a,
                                                const int8_t *__restrict__ w,
                                                int8_t *__restrict__ b) {
    extern __shared__ int8_t s_mem_int8[];                 
    const int INPUT_TILE_SIZE_IN_ELEMENTS = (BLOCK_H + R - 1) * (BLOCK_W + S - 1);
    int8_t* s_a = s_mem_int8;                          
    int8_t* s_w = s_mem_int8 + INPUT_TILE_SIZE_IN_ELEMENTS; 

    // === 1. 坐标与ID计算 ===
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BLOCK_W + tx;
    const int tile_h_start = blockIdx.y * BLOCK_H;
    const int tile_w_start = blockIdx.x * BLOCK_W;
    const int output_h = tile_h_start + ty;
    const int output_w = tile_w_start + tx;
    const int NUM_THREADS = BLOCK_H * BLOCK_W;

    // === 2. 主计算循环 ===
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            int accumulator = 0; 

            for (int c = 0; c < C; ++c) {
                
                const int INPUT_TILE_H = BLOCK_H + R - 1;
                const int INPUT_TILE_W = BLOCK_W + S - 1;
                for (int i = tid; i < INPUT_TILE_SIZE_IN_ELEMENTS; i += NUM_THREADS) {
                    const int smem_h = i / INPUT_TILE_W;
                    const int smem_w = i % INPUT_TILE_W;
                    const int g_ih = tile_h_start + smem_h - R / 2;
                    const int g_iw = tile_w_start + smem_w - S / 2;
                    if (g_ih >= 0 && g_ih < H && g_iw >= 0 && g_iw < W) {
                        s_a[i] = a(n, g_ih, g_iw, c);
                    } else {
                        s_a[i] = static_cast<int8_t>(0);
                    }
                }
                const int WEIGHT_SIZE = R * S;
                for (int i = tid; i < WEIGHT_SIZE; i += NUM_THREADS) {
                    const int r = i / S;        
                    const int s = i % S;     
                    s_w[i] = w(k, r, s, c);
                }

                
                __syncthreads();

           
                if (output_h < H && output_w < W) {
                    for (int r = 0; r < R; ++r) {
                        for (int s = 0; s < S; ++s) {
                            int8_t s_val = s_a[(ty + r) * INPUT_TILE_W + (tx + s)];
                            int8_t w_val = s_w[r * S + s];
                            accumulator += static_cast<int>(s_val) * static_cast<int>(w_val);
                        }
                    }
                }

               
                __syncthreads();

            } 

           
            if (output_h < H && output_w < W) {
                b(n, output_h, output_w, k) = static_cast<int8_t>(accumulator);
            }
        } 
    } 
}

// =======================================================
// --- HALF 版本 ---
// =======================================================
template <>
__global__ void conv2d_cuda_kernel<half_t, float>(const half_t *__restrict__ a,
                                                  const half_t *__restrict__ w,
                                                  half_t *__restrict__ b) {
    
    extern __shared__ half_t s_mem_half[];                
    const int INPUT_TILE_SIZE_IN_ELEMENTS = (BLOCK_H + R - 1) * (BLOCK_W + S - 1);
    half_t* s_a = s_mem_half;                               
    half_t* s_w = s_mem_half + INPUT_TILE_SIZE_IN_ELEMENTS; 

    // === 1. 坐标与ID计算 ===
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BLOCK_W + tx;
    const int tile_h_start = blockIdx.y * BLOCK_H;
    const int tile_w_start = blockIdx.x * BLOCK_W;
    const int output_h = tile_h_start + ty;
    const int output_w = tile_w_start + tx;
    const int NUM_THREADS = BLOCK_H * BLOCK_W;

    // === 2. 主计算循环 ===
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            float accumulator = 0.0f; 

            for (int c = 0; c < C; ++c) {
      
                const int INPUT_TILE_H = BLOCK_H + R - 1;
                const int INPUT_TILE_W = BLOCK_W + S - 1;
                for (int i = tid; i < INPUT_TILE_SIZE_IN_ELEMENTS; i += NUM_THREADS) {
                    const int smem_h = i / INPUT_TILE_W;
                    const int smem_w = i % INPUT_TILE_W;
                    const int g_ih = tile_h_start + smem_h - R / 2;
                    const int g_iw = tile_w_start + smem_w - S / 2;
                    if (g_ih >= 0 && g_ih < H && g_iw >= 0 && g_iw < W) {
                        s_a[i] = a(n, g_ih, g_iw, c);
                    } else {
                        s_a[i] = static_cast<half_t>(0.0f);
                    }
                }
                const int WEIGHT_SIZE = R * S;
                for (int i = tid; i < WEIGHT_SIZE; i += NUM_THREADS) {
                    const int r = i / S;      
                    const int s = i % S;       
                    s_w[i] = w(k, r, s, c);
                }
             
                __syncthreads();

                if (output_h < H && output_w < W) {
                    for (int r = 0; r < R; ++r) {
                        for (int s = 0; s < S; ++s) {
                            half_t s_val = s_a[(ty + r) * INPUT_TILE_W + (tx + s)];
                            half_t w_val = s_w[r * S + s];
                            
                            accumulator += static_cast<float>(s_val) * static_cast<float>(w_val);
                        }
                    }
                }
                
             
                __syncthreads();

            } 

          
            if (output_h < H && output_w < W) {
                b(n, output_h, output_w, k) = static_cast<half_t>(accumulator);
            }
        } 
    } 
}

// =======================================================
// --- 配置函数  ---
// =======================================================
template <>
KernelConfig get_kernel_config<int8_t>() {
    KernelConfig config;
    config.grid = dim3((W + BLOCK_W - 1) / BLOCK_W, (H + BLOCK_H - 1) / BLOCK_H);
    config.block = dim3(BLOCK_W, BLOCK_H, 1);
    const int input_tile_size = (BLOCK_H + R - 1) * (BLOCK_W + S - 1) * sizeof(int8_t);
    const int weight_tile_size = R * S * sizeof(int8_t);
    config.shared_memory_size = input_tile_size + weight_tile_size;
    return config;
}

template <>
KernelConfig get_kernel_config<half_t>() {
    KernelConfig config;
    config.grid = dim3((W + BLOCK_W - 1) / BLOCK_W, (H + BLOCK_H - 1) / BLOCK_H);
    config.block = dim3(BLOCK_W, BLOCK_H, 1);
    const int input_tile_size = (BLOCK_H + R - 1) * (BLOCK_W + S - 1) * sizeof(half_t);
    const int weight_tile_size = R * S * sizeof(half_t);
    config.shared_memory_size = input_tile_size + weight_tile_size;
    return config;
}