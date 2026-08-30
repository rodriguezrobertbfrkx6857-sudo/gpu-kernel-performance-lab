#include "cuda_check.cuh"
#include "kernels.cuh"

#include <cuda_runtime.h>

namespace {

constexpr int kBlockSize = 256;

__device__ float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void reduction_naive_kernel(const float* input, float* output, std::size_t n) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < n) {
        atomicAdd(output, input[index]);
    }
}

__global__ void reduction_shared_kernel(const float* input, float* output, std::size_t n) {
    __shared__ float shared[kBlockSize];
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x * 2 + threadIdx.x;
    float value = 0.0f;
    if (index < n) value += input[index];
    if (index + blockDim.x < n) value += input[index + blockDim.x];
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(output, shared[0]);
}

__global__ void reduction_warp_shuffle_kernel(const float* input, float* output, std::size_t n) {
    __shared__ float warp_sums[kBlockSize / 32];
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x * 2 + threadIdx.x;
    float value = 0.0f;
    if (index < n) value += input[index];
    if (index + blockDim.x < n) value += input[index + blockDim.x];
    value = warp_reduce_sum(value);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = (lane < (kBlockSize / 32)) ? warp_sums[lane] : 0.0f;
        value = warp_reduce_sum(value);
        if (lane == 0) atomicAdd(output, value);
    }
}

unsigned int reduction_grid(std::size_t n) {
    return static_cast<unsigned int>((n + (kBlockSize * 2 - 1)) / (kBlockSize * 2));
}

}  // namespace

namespace gpu_lab::cuda {

void launch_reduction_naive(const float* input, float* output, std::size_t n,
                            gpu_lab_stream_t stream) {
    GPU_LAB_CUDA_CHECK(cudaMemsetAsync(output, 0, sizeof(float), stream));
    reduction_naive_kernel<<<static_cast<unsigned int>((n + kBlockSize - 1) / kBlockSize),
                             kBlockSize, 0, stream>>>(input, output, n);
    GPU_LAB_CUDA_CHECK(cudaGetLastError());
}

void launch_reduction_shared(const float* input, float* output, std::size_t n,
                             gpu_lab_stream_t stream) {
    GPU_LAB_CUDA_CHECK(cudaMemsetAsync(output, 0, sizeof(float), stream));
    reduction_shared_kernel<<<reduction_grid(n), kBlockSize, 0, stream>>>(input, output, n);
    GPU_LAB_CUDA_CHECK(cudaGetLastError());
}

void launch_reduction_warp_shuffle(const float* input, float* output, std::size_t n,
                                   gpu_lab_stream_t stream) {
    GPU_LAB_CUDA_CHECK(cudaMemsetAsync(output, 0, sizeof(float), stream));
    reduction_warp_shuffle_kernel<<<reduction_grid(n), kBlockSize, 0, stream>>>(input, output, n);
    GPU_LAB_CUDA_CHECK(cudaGetLastError());
}

}  // namespace gpu_lab::cuda

