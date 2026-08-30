#include "bench_common.cuh"
#include "kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void run_case(std::size_t n, int iterations) {
    const std::size_t bytes = n * sizeof(float);
    const std::size_t required_bytes = 3 * bytes;
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    GPU_LAB_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (required_bytes > free_bytes / 2) {
        gpu_lab::bench::print_skip("vector_add", "[" + std::to_string(n) + "]",
                                   required_bytes, free_bytes,
                                   "insufficient device memory with safety headroom");
        return;
    }
    std::vector<float> host(n, 1.0f);
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    GPU_LAB_CUDA_CHECK(cudaMalloc(&a, bytes));
    GPU_LAB_CUDA_CHECK(cudaMalloc(&b, bytes));
    GPU_LAB_CUDA_CHECK(cudaMalloc(&c, bytes));
    GPU_LAB_CUDA_CHECK(cudaMemcpy(a, host.data(), bytes, cudaMemcpyHostToDevice));
    GPU_LAB_CUDA_CHECK(cudaMemcpy(b, host.data(), bytes, cudaMemcpyHostToDevice));
    gpu_lab::cuda::launch_vector_add(a, b, c, n);
    GPU_LAB_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> actual(n);
    GPU_LAB_CUDA_CHECK(cudaMemcpy(actual.data(), c, bytes, cudaMemcpyDeviceToHost));
    if (std::any_of(actual.begin(), actual.end(), [](float value) { return value != 2.0f; })) {
        GPU_LAB_CUDA_CHECK(cudaFree(a));
        GPU_LAB_CUDA_CHECK(cudaFree(b));
        GPU_LAB_CUDA_CHECK(cudaFree(c));
        throw std::runtime_error("vector_add correctness check failed before benchmark");
    }
    const auto summary = gpu_lab::bench::measure_cuda([&] {
        gpu_lab::cuda::launch_vector_add(a, b, c, n);
    }, 20, iterations);
    gpu_lab::bench::print_summary("vector_add", "vector_add", "[" + std::to_string(n) + "]",
                                  n, summary, 3 * bytes, 20, iterations);
    GPU_LAB_CUDA_CHECK(cudaFree(a));
    GPU_LAB_CUDA_CHECK(cudaFree(b));
    GPU_LAB_CUDA_CHECK(cudaFree(c));
}

}  // namespace

int main() {
    for (const std::size_t n : {std::size_t{1} << 20, std::size_t{1} << 24}) {
        run_case(n, n >= (std::size_t{1} << 24) ? 50 : 100);
    }
    return 0;
}
