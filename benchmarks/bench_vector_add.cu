#include "bench_common.cuh"
#include "kernels.cuh"

#include <cuda_runtime.h>

#include <vector>

int main() {
    constexpr std::size_t n = 1u << 24;
    const std::size_t bytes = n * sizeof(float);
    std::vector<float> host(n, 1.0f);
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    GPU_LAB_CUDA_CHECK(cudaMalloc(&a, bytes));
    GPU_LAB_CUDA_CHECK(cudaMalloc(&b, bytes));
    GPU_LAB_CUDA_CHECK(cudaMalloc(&c, bytes));
    GPU_LAB_CUDA_CHECK(cudaMemcpy(a, host.data(), bytes, cudaMemcpyHostToDevice));
    GPU_LAB_CUDA_CHECK(cudaMemcpy(b, host.data(), bytes, cudaMemcpyHostToDevice));
    const auto summary = gpu_lab::bench::measure_cuda([&] {
        gpu_lab::cuda::launch_vector_add(a, b, c, n);
    });
    gpu_lab::bench::print_summary("vector_add", n, summary, 3 * bytes);
    GPU_LAB_CUDA_CHECK(cudaFree(a));
    GPU_LAB_CUDA_CHECK(cudaFree(b));
    GPU_LAB_CUDA_CHECK(cudaFree(c));
    return 0;
}

