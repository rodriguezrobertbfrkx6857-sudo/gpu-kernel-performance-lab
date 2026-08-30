#include "cuda_check.cuh"
#include "kernels.cuh"

#include <algorithm>

namespace {

__global__ void vector_add_kernel(const float* a, const float* b, float* c, std::size_t n) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

}  // namespace

namespace gpu_lab::cuda {

void launch_vector_add(const float* a, const float* b, float* c, std::size_t n,
                       gpu_lab_stream_t stream) {
    if (n == 0) return;
    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = static_cast<unsigned int>(std::min<std::size_t>(
        (n + block_size - 1) / block_size, 2147483647u));
    vector_add_kernel<<<grid_size, block_size, 0, stream>>>(a, b, c, n);
    GPU_LAB_CUDA_CHECK(cudaGetLastError());
}

}  // namespace gpu_lab::cuda
