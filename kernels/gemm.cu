#include "cuda_check.cuh"
#include "kernels.cuh"

#include <cuda_runtime.h>

namespace {

__global__ void gemm_naive_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float accumulator = 0.0f;
        for (int inner = 0; inner < k; ++inner) {
            accumulator += a[row * k + inner] * b[inner * n + col];
        }
        c[row * n + col] = accumulator;
    }
}

constexpr int kTile = 16;

__global__ void gemm_tiled_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float tile_a[kTile][kTile];
    __shared__ float tile_b[kTile][kTile];
    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col = blockIdx.x * kTile + threadIdx.x;
    float accumulator = 0.0f;
    for (int tile = 0; tile < (k + kTile - 1) / kTile; ++tile) {
        const int a_col = tile * kTile + threadIdx.x;
        const int b_row = tile * kTile + threadIdx.y;
        tile_a[threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
        tile_b[threadIdx.y][threadIdx.x] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
        __syncthreads();
        for (int inner = 0; inner < kTile; ++inner) {
            accumulator += tile_a[threadIdx.y][inner] * tile_b[inner][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < m && col < n) c[row * n + col] = accumulator;
}

}  // namespace

namespace gpu_lab::cuda {

void launch_gemm_naive(const float* a, const float* b, float* c, int m, int n, int k,
                       gpu_lab_stream_t stream) {
    const dim3 block(16, 16);
    const dim3 grid(static_cast<unsigned int>((n + 15) / 16),
                    static_cast<unsigned int>((m + 15) / 16));
    gemm_naive_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    GPU_LAB_CUDA_CHECK(cudaGetLastError());
}

void launch_gemm_tiled(const float* a, const float* b, float* c, int m, int n, int k,
                       gpu_lab_stream_t stream) {
    const dim3 block(kTile, kTile);
    const dim3 grid(static_cast<unsigned int>((n + kTile - 1) / kTile),
                    static_cast<unsigned int>((m + kTile - 1) / kTile));
    gemm_tiled_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    GPU_LAB_CUDA_CHECK(cudaGetLastError());
}

}  // namespace gpu_lab::cuda

