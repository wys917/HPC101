#include "conv.cuh"

// 宏定义：方便地对四维张量进行索引
#define a(n, h, w, c) a[(n) * H * W * C + (h) * W * C + (w) * C + (c)]
#define w(k, r, s, c) w[(k) * R * S * C + (r) * S * C + (s) * C + (c)]
#define b(n, h, w, k) b[(n) * H * W * K + (h) * W * K + (w) * K + (k)]

// 定义线程块的二维尺寸
static constexpr int BLOCK = 16;
static constexpr int BLOCK_H = BLOCK;
static constexpr int BLOCK_W = BLOCK;

// =======================================================
// --- INT8 版本的模板特化 ---
// =======================================================
template <>
__global__ void conv2d_cuda_kernel<int8_t, int>(const int8_t *__restrict__ a, const int8_t *__restrict__ w, int8_t *__restrict__ b) {
    // 声明动态共享内存 - 为输入数据和权重数据分配空间
    extern __shared__ int8_t s_mem[];
    int8_t* s_a_int8 = s_mem;
    int8_t* s_w_int8 = s_mem + (BLOCK_H + R - 1) * (BLOCK_W + S - 1);
    
    // === 1. 坐标与ID计算 ===
    const int tx = threadIdx.x; // 线程的x坐标 (0-15)
    const int ty = threadIdx.y; // 线程的y坐标 (0-15)
    const int tid = ty * BLOCK_W + tx; // 线程的唯一ID (0-255)

    // 计算当前线程块(Tile)负责计算的输出区域的左上角像素坐标
    const int tile_h_start = blockIdx.y * BLOCK_H;
    const int tile_w_start = blockIdx.x * BLOCK_W;
    
    // 计算当前线程负责计算的输出像素，在整个输出图像中的全局坐标
    const int output_h = tile_h_start + ty;
    const int output_w = tile_w_start + tx;

    // === 2. 共享内存瓦片(Tile)尺寸定义 ===
    const int INPUT_TILE_H = BLOCK_H + R - 1;
    const int INPUT_TILE_W = BLOCK_W + S - 1;
    const int INPUT_TILE_SIZE = INPUT_TILE_H * INPUT_TILE_W;
    const int NUM_THREADS = BLOCK_H * BLOCK_W;

    // === 3. 主计算循环 - 遍历所有batch ===
    for (int n = 0; n < N; ++n) {
        // 每个线程需要计算它所负责的那个输出像素的所有K个输出通道值
        for (int k = 0; k < K; ++k) {
            // 累加器，用于存储一个输出像素(n, h, w, k)的最终值
            int accumulator = 0;
            
            // 卷积操作需要在所有输入通道C上进行求和
            for (int c = 0; c < C; ++c) {
                
                // --- 4a. 加载权重数据到共享内存 ---
                const int WEIGHT_SIZE = R * S;
                for (int i = tid; i < WEIGHT_SIZE; i += NUM_THREADS) {
                    const int r = i / S;
                    const int s = i % S;
                    s_w_int8[i] = w(k, r, s, c);
                }
                
                // --- 4b. 加载输入数据到共享内存 ---
                for (int i = tid; i < INPUT_TILE_SIZE; i += NUM_THREADS) {
                    // a. 将一维索引i，解码成共享内存中的二维坐标(smem_h, smem_w)
                    const int smem_h = i / INPUT_TILE_W;
                    const int smem_w = i % INPUT_TILE_W;
                    // b. 计算这个共享内存坐标，对应到全局输入张量a中的坐标(g_ih, g_iw)
                    const int g_ih = tile_h_start + smem_h - R / 2;
                    const int g_iw = tile_w_start + smem_w - S / 2;

                    // c. 边界检查。如果计算出的全局坐标在图像范围内，就从全局内存加载
                    if (g_ih >= 0 && g_ih < H && g_iw >= 0 && g_iw < W) {
                        s_a_int8[i] = a(n, g_ih, g_iw, c);
                    } else {
                        // d. 如果超出了边界，说明是padding区域，直接填0
                        s_a_int8[i] = static_cast<int8_t>(0);
                    }
                }
                
                // --- 5. 同步点 1 ---
                __syncthreads();

                // --- 6. 计算阶段：从共享内存读取数据进行计算 ---
                if (output_h < H && output_w < W) {
                    // 遍历卷积核的每一个元素(r, s)
                    for (int r = 0; r < R; ++r) {
                        for (int s = 0; s < S; ++s) {
                            // a. 从共享内存中读取输入值
                            int8_t s_val = s_a_int8[(ty + r) * INPUT_TILE_W + (tx + s)];
                            // b. 从共享内存中读取权重值（性能优化）
                            int8_t w_val = s_w_int8[r * S + s];
                            // c. 执行乘加运算，累加到私有寄存器accumulator中
                            accumulator += static_cast<int>(s_val) * static_cast<int>(w_val);
                        }
                    }
                }
                
                // --- 7. 同步点 2 ---
                __syncthreads();
            }
            
            // --- 8. 写回阶段：将结果从寄存器写回全局内存 ---
            if (output_h < H && output_w < W) {
                // 直接转换，不做饱和处理（与原始实现保持一致）
                b(n, output_h, output_w, k) = static_cast<int8_t>(accumulator);
            }
        }
    }
}

// =======================================================
// --- HALF 版本的模板特化 ---
// =======================================================
template <>
__global__ void conv2d_cuda_kernel<half_t, float>(const half_t *__restrict__ a, const half_t *__restrict__ w, half_t *__restrict__ b) {
    // 声明动态共享内存 - 为输入数据和权重数据分配空间
    extern __shared__ half_t s_mem[];
    half_t* s_a_half = s_mem;
    half_t* s_w_half = s_mem + (BLOCK_H + R - 1) * (BLOCK_W + S - 1);
    
    // === 1. 坐标与ID计算 ===
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BLOCK_W + tx;

    // 计算当前线程块负责的输出区域
    const int tile_h_start = blockIdx.y * BLOCK_H;
    const int tile_w_start = blockIdx.x * BLOCK_W;
    
    const int output_h = tile_h_start + ty;
    const int output_w = tile_w_start + tx;

    // === 2. 共享内存瓦片尺寸定义 ===
    const int INPUT_TILE_H = BLOCK_H + R - 1;
    const int INPUT_TILE_W = BLOCK_W + S - 1;
    const int INPUT_TILE_SIZE = INPUT_TILE_H * INPUT_TILE_W;
    const int NUM_THREADS = BLOCK_H * BLOCK_W;

    // === 3. 主计算循环 - 遍历所有batch ===
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            float accumulator = 0.0f;
            
            for (int c = 0; c < C; ++c) {
                
                // --- 4a. 加载权重数据到共享内存 ---
                const int WEIGHT_SIZE = R * S;
                for (int i = tid; i < WEIGHT_SIZE; i += NUM_THREADS) {
                    const int r = i / S;
                    const int s = i % S;
                    s_w_half[i] = w(k, r, s, c);
                }
                
                // --- 4b. 加载输入数据到共享内存 ---
                for (int i = tid; i < INPUT_TILE_SIZE; i += NUM_THREADS) {
                    const int smem_h = i / INPUT_TILE_W;
                    const int smem_w = i % INPUT_TILE_W;
                    const int g_ih = tile_h_start + smem_h - R / 2;
                    const int g_iw = tile_w_start + smem_w - S / 2;

                    if (g_ih >= 0 && g_ih < H && g_iw >= 0 && g_iw < W) {
                        s_a_half[i] = a(n, g_ih, g_iw, c);
                    } else {
                        s_a_half[i] = static_cast<half_t>(0);
                    }
                }
                
                __syncthreads();

                // --- 6. 计算阶段 ---
                if (output_h < H && output_w < W) {
                    for (int r = 0; r < R; ++r) {
                        for (int s = 0; s < S; ++s) {
                            half_t s_val = s_a_half[(ty + r) * INPUT_TILE_W + (tx + s)];
                            // 从共享内存中读取权重值（性能优化）
                            half_t w_val = s_w_half[r * S + s];
                            // 注意：为了与原始实现保持一致，这里转换为int（虽然accumulator是float）
                            accumulator += static_cast<int>(s_val) * static_cast<int>(w_val);
                        }
                    }
                }
                
                __syncthreads();
            }
            
            // --- 8. 写回阶段 ---
            if (output_h < H && output_w < W) {
                b(n, output_h, output_w, k) = static_cast<half_t>(accumulator);
            }
        }
    }
}

// =======================================================
// --- 配置函数 ---
// =======================================================
template <>
KernelConfig get_kernel_config<int8_t>() {
    KernelConfig config;
    // 注意：这里修正了Grid配置，移除了z维度
    config.grid = dim3((W + BLOCK_W - 1) / BLOCK_W, (H + BLOCK_H - 1) / BLOCK_H);
    config.block = dim3(BLOCK_W, BLOCK_H, 1);
    // 为输入数据和权重数据分配共享内存
    const int input_tile_size = (BLOCK_H + R - 1) * (BLOCK_W + S - 1) * sizeof(int8_t);
    const int weight_tile_size = R * S * sizeof(int8_t);
    config.shared_memory_size = input_tile_size + weight_tile_size;
    return config;
}

template <>
KernelConfig get_kernel_config<half_t>() {
    KernelConfig config;
    // 注意：这里修正了Grid配置，移除了z维度
    config.grid = dim3((W + BLOCK_W - 1) / BLOCK_W, (H + BLOCK_H - 1) / BLOCK_H);
    config.block = dim3(BLOCK_W, BLOCK_H, 1);
    // 为输入数据和权重数据分配共享内存
    const int input_tile_size = (BLOCK_H + R - 1) * (BLOCK_W + S - 1) * sizeof(half_t);
    const int weight_tile_size = R * S * sizeof(half_t);
    config.shared_memory_size = input_tile_size + weight_tile_size;
    return config;
}
