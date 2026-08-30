#include "cuda_check.cuh"
#include "kernels.cuh"

#include <cuda_runtime.h>

namespace {

constexpr int kTile = 32;

__global__ void transpose_naive_kernel(const float* input, float* output, std::size_t rows,
                                       std::size_t cols) {
    const std::size_t col = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

__global__ void transpose_tiled_kernel(const float* input, float* output, std::size_t rows,
                                       std::size_t cols) {
    __shared__ float tile[kTile][kTile];
    const std::size_t col = static_cast<std::size_t>(blockIdx.x) * kTile + threadIdx.x;
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * kTile + threadIdx.y;
    if (row < rows && col < cols) {
        tile[threadIdx.y][threadIdx.x] = input[row * cols + col];
    }
    __syncthreads();
    const std::size_t transposed_col = static_cast<std::size_t>(blockIdx.y) * kTile + threadIdx.x;
    const std::size_t transposed_row = static_cast<std::size_t>(blockIdx.x) * kTile + threadIdx.y;
    if (transposed_row < cols && transposed_col < rows) {
        output[transposed_row * rows + transposed_col] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void transpose_tiled_padded_kernel(const float* input, float* output, std::size_t rows,
                                              std::size_t cols) {
    __shared__ float tile[kTile][kTile + 1];
    const std::size_t col = static_cast<std::size_t>(blockIdx.x) * kTile + threadIdx.x;
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * kTile + threadIdx.y;
    if (row < rows && col < cols) {
        tile[threadIdx.y][threadIdx.x] = input[row * cols + col];
    }
    __syncthreads();
    const std::size_t transposed_col = static_cast<std::size_t>(blockIdx.y) * kTile + threadIdx.x;
    const std::size_t transposed_row = static_cast<std::size_t>(blockIdx.x) * kTile + threadIdx.y;
    if (transposed_row < cols && transposed_col < rows) {
        output[transposed_row * rows + transposed_col] = tile[threadIdx.x][threadIdx.y];
    }
}

}  // namespace

namespace gpu_lab::cuda {

void launch_copy_baseline(const float* input, float* output, std::size_t rows, std::size_t cols,
                          gpu_lab_stream_t stream) {
    GPU_LAB_CUDA_CHECK(cudaMemcpyAsync(output, input, rows * cols * sizeof(float),
                                       cudaMemcpyDeviceToDevice, stream));
}

void launch_transpose_naive(const float* input, float* output, std::size_t rows, std::size_t cols,
                            gpu_lab_stream_t stream) {
    const dim3 block(kTile, kTile);
    const dim3 grid(static_cast<unsigned int>((cols + kTile - 1) / kTile),
                    static_cast<unsigned int>((rows + kTile - 1) / kTile));
    transpose_naive_kernel<<<grid, block, 0, stream>>>(input, output, rows, cols);
    GPU_LAB_CUDA_CHECK(cudaGetLastError());
}

void launch_transpose_tiled(const float* input, float* output, std::size_t rows, std::size_t cols,
                            gpu_lab_stream_t stream) {
    const dim3 block(kTile, kTile);
    const dim3 grid(static_cast<unsigned int>((cols + kTile - 1) / kTile),
                    static_cast<unsigned int>((rows + kTile - 1) / kTile));
    transpose_tiled_kernel<<<grid, block, 0, stream>>>(input, output, rows, cols);
    GPU_LAB_CUDA_CHECK(cudaGetLastError());
}

void launch_transpose_tiled_padded(const float* input, float* output, std::size_t rows,
                                   std::size_t cols, gpu_lab_stream_t stream) {
    const dim3 block(kTile, kTile);
    const dim3 grid(static_cast<unsigned int>((cols + kTile - 1) / kTile),
                    static_cast<unsigned int>((rows + kTile - 1) / kTile));
    transpose_tiled_padded_kernel<<<grid, block, 0, stream>>>(input, output, rows, cols);
    GPU_LAB_CUDA_CHECK(cudaGetLastError());
}

}  // namespace gpu_lab::cuda

