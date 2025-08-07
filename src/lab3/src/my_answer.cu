#include "conv.cuh"

// 宏定义：方便地对四维张量进行索引，避免手动计算时出错。
// a: 输入张量 (N, H, W, C)
// w: 权重张量 (K, R, S, C) - 注意这里的维度顺序
// b: 输出张量 (N, H, W, K)
#define a(n, h, w, c) a[(n) * H * W * C + (h) * W * C + (w) * C + (c)]
#define w(k, r, s, c) w[(k) * R * S * C + (r) * S * C + (s) * C + (c)]
#define b(n, h, w, k) b[(n) * H * W * K + (h) * W * K + (w) * K + (k)]

// 定义线程块的二维尺寸，16x16=256个线程，这是个常见的选择。
static constexpr int BLOCK_H = 16;
static constexpr int BLOCK_W = 16;

// =======================================================
// --- 内部 __device__ 函数 (所有核心逻辑都在这里) ---
// =======================================================
template <typename Element, typename ElementCompute>
__device__ void conv2d_device_logic(const Element *__restrict__ a, const Element *__restrict__ w, Element *__restrict__ b, Element *s_a) {
    // === 1. 坐标与ID计算 ===
    // threadIdx是内置变量，表示当前线程在线程块(Block)内的坐标。
    const int tx = threadIdx.x; // 线程的x坐标 (0-15)
    const int ty = threadIdx.y; // 线程的y坐标 (0-15)
    // 将二维线程坐标转换为一维的线程ID，方便后面在循环中分配任务。
    const int tid = ty * BLOCK_W + tx; // 线程的唯一ID (0-255)

    // blockIdx是内置变量，表示当前线程块在网格(Grid)中的坐标。
    // 这里我们计算出当前线程块(Tile)负责计算的输出区域的左上角像素坐标。
    const int tile_h_start = blockIdx.y * BLOCK_H;
    const int tile_w_start = blockIdx.x * BLOCK_W;
    
    // 计算当前线程负责计算的输出像素，在整个输出图像中的全局坐标。
    const int output_h = tile_h_start + ty;
    const int output_w = tile_w_start + tx;

    // 当前线程处理的图像批次(batch)，由Grid的z维度决定。
    const int n = blockIdx.z;

    // === 2. 共享内存瓦片(Tile)尺寸定义 ===
    // 为了计算一个16x16的输出块，我们需要一个更大的输入块。
    // 例如，如果卷积核是3x3，我们需要在上下左右各多加载1个像素的“Halo”区域。
    // 所以输入瓦片的高度是 16 + 3 - 1 = 18。
    const int INPUT_TILE_H = BLOCK_H + R - 1;
    const int INPUT_TILE_W = BLOCK_W + S - 1;
    // 计算出这个输入瓦片总共有多少个像素，用于后续的加载循环。
    const int INPUT_TILE_SIZE = INPUT_TILE_H * INPUT_TILE_W;
    // 块内的线程总数。
    const int NUM_THREADS = BLOCK_H * BLOCK_W;

    // === 3. 主计算循环 ===
    // 每个线程需要计算它所负责的那个输出像素的所有K个输出通道值。
    for (int k = 0; k < K; ++k) {
        // 累加器，用于存储一个输出像素(n, h, w, k)的最终值。
        // 每次计算新的输出通道k时，必须清零。
        ElementCompute accumulator = 0;
        
        // 卷积操作需要在所有输入通道C上进行求和。
        for (int c = 0; c < C; ++c) {
            
            // --- 4. 加载阶段：将数据从全局内存合作搬运到共享内存 ---
            // 这是一个grid-stride loop。让块内所有256个线程一起合作，
            // 通过循环，把整个 INPUT_TILE_SIZE 大小的瓦片数据全部填满。
            for (int i = tid; i < INPUT_TILE_SIZE; i += NUM_THREADS) {
                // a. 将一维索引i，解码成共享内存中的二维坐标(smem_h, smem_w)。
                const int smem_h = i / INPUT_TILE_W;
                const int smem_w = i % INPUT_TILE_W;
                // b. 计算这个共享内存坐标，对应到全局输入张量a中的坐标(g_ih, g_iw)。
                //    - R/2 和 -S/2 是为了实现'same' padding，让卷积核中心对准像素。
                const int g_ih = tile_h_start + smem_h - R / 2;
                const int g_iw = tile_w_start + smem_w - S / 2;

                // c. 边界检查。如果计算出的全局坐标在图像范围内，就从全局内存加载。
                if (g_ih >= 0 && g_ih < H && g_iw >= 0 && g_iw < W) {
                    s_a[i] = a(n, g_ih, g_iw, c);
                } else {
                    // d. 如果超出了边界，说明是padding区域，直接填0。
                    s_a[i] = static_cast<Element>(0);
                }
            }
            
            // --- 5. 同步点 1 ---
            // **至关重要**！必须保证块内所有256个线程都完成了加载操作，
            // 才能进入下一步的计算。否则有些线程会读到旧的或未初始化的数据。
            __syncthreads();

            // --- 6. 计算阶段：从共享内存读取数据进行计算 ---
            // 再次进行边界检查，确保当前线程负责的输出像素是在图像有效范围内的。
            // 这一步是为了处理图像尺寸不能被BLOCK_H/W整除的情况。
            if (output_h < H && output_w < W) {
                // 遍历卷积核的每一个元素(r, s)
                for (int r = 0; r < R; ++r) {
                    for (int s = 0; s < S; ++s) {
                        // a. 从共享内存中读取输入值。
                        //    注意这里的索引 (ty + r) 和 (tx + s)，它利用了Halo区域的数据。
                        Element s_val = s_a[(ty + r) * INPUT_TILE_W + (tx + s)];
                        // b. 从全局内存中读取权重值。
                        Element w_val = w(k, r, s, c);
                        // c. 执行乘加运算，累加到私有寄存器accumulator中。
                        accumulator += static_cast<ElementCompute>(s_val) * static_cast<ElementCompute>(w_val);
                    }
                }
            }
            
            // --- 7. 同步点 2 ---
            // 同样重要。保证块内所有线程都完成了当前输入通道c的计算，
            // 才能进入下一个c的循环，去覆盖共享内存中的数据。
            __syncthreads();
        }
        
        // --- 8. 写回阶段：将结果从寄存器写回全局内存 ---
        if (output_h < H && output_w < W) {
            // 专门为int8类型做的饱和处理，防止累加结果溢出int8的表示范围。
            if constexpr (std::is_same_v<Element, int8_t>) {
                if (accumulator > 127) accumulator = 127;
                if (accumulator < -128) accumulator = -128;
            }
            // 将计算结果写回到输出张量b的对应位置。
            b(n, output_h, output_w, k) = static_cast<Element>(accumulator);
        }
    }
}


// =======================================================
// --- __global__ 入口函数 (它们是简单的“包装器”) ---
// =======================================================
template <>
__global__ void conv2d_cuda_kernel<int8_t, int>(const int8_t *__restrict__ a, const int8_t *__restrict__ w, int8_t *__restrict__ b) {
    // 声明动态共享内存，并把它作为指针传递给真正的逻辑函数。
    extern __shared__ int8_t s_a_int8[];
    conv2d_device_logic<int8_t, int>(a, w, b, s_a_int8);
}

template <>
__global__ void conv2d_cuda_kernel<half_t, float>(const half_t *__restrict__ a, const half_t *__restrict__ w, half_t *__restrict__ b) {
    extern __shared__ half_t s_a_half[];
    conv2d_device_logic<half_t, float>(a, w, b, s_a_half);
}

// =======================================================
// --- 配置函数 (用于设置启动参数) ---
// =======================================================
template <>
KernelConfig get_kernel_config<int8_t>() {
    KernelConfig config;
    config.grid = dim3((W + BLOCK_W - 1) / BLOCK_W, (H + BLOCK_H - 1) / BLOCK_H, N);
    config.block = dim3(BLOCK_W, BLOCK_H, 1);
    config.shared_memory_size = (BLOCK_H + R - 1) * (BLOCK_W + S - 1) * sizeof(int8_t);
    return config;
}

template <>
KernelConfig get_kernel_config<half_t>() {
    KernelConfig config;
    config.grid = dim3((W + BLOCK_W - 1) / BLOCK_W, (H + BLOCK_H - 1) / BLOCK_H, N);
    config.block = dim3(BLOCK_W, BLOCK_H, 1);
    config.shared_memory_size = (BLOCK_H + R - 1) * (BLOCK_W + S - 1) * sizeof(half_t);
    return config;
}