#include "bench_common.cuh"
#include "kernels.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

namespace {

template <typename Launch>
void run_variant(const char* name, Launch&& launch, const float* input, float* output,
                 std::size_t n, int iterations) {
    const auto summary = gpu_lab::bench::measure_cuda(std::forward<Launch>(launch), 20, iterations);
    float result = 0.0f;
    GPU_LAB_CUDA_CHECK(cudaMemcpy(&result, output, sizeof(float), cudaMemcpyDeviceToHost));
    gpu_lab::bench::print_summary(name, n, summary, n * sizeof(float));
    std::cout << "{\"kernel_result\":\"" << name << "\",\"value\":" << result << "}\n";
}

}  // namespace

int main() {
    for (const std::size_t n : {std::size_t{1} << 20, std::size_t{1} << 24,
                                std::size_t{1} << 26}) {
        const std::size_t bytes = n * sizeof(float);
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        GPU_LAB_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (bytes + (8u << 20) > free_bytes / 2) {
            std::cout << "{\"status\":\"SKIPPED_INSUFFICIENT_DEVICE_MEMORY\",\"elements\":"
                      << n << "}\n";
            continue;
        }
        float* input = nullptr;
        float* output = nullptr;
        GPU_LAB_CUDA_CHECK(cudaMalloc(&input, bytes));
        GPU_LAB_CUDA_CHECK(cudaMalloc(&output, sizeof(float)));
        GPU_LAB_CUDA_CHECK(cudaMemset(input, 0, bytes));
        const int iterations = n >= (std::size_t{1} << 26) ? 20 : 100;
        run_variant("reduction_naive", [&] {
            gpu_lab::cuda::launch_reduction_naive(input, output, n);
        }, input, output, n, iterations);
        run_variant("reduction_shared", [&] {
            gpu_lab::cuda::launch_reduction_shared(input, output, n);
        }, input, output, n, iterations);
        run_variant("reduction_warp_shuffle", [&] {
            gpu_lab::cuda::launch_reduction_warp_shuffle(input, output, n);
        }, input, output, n, iterations);
        GPU_LAB_CUDA_CHECK(cudaFree(input));
        GPU_LAB_CUDA_CHECK(cudaFree(output));
    }
    return 0;
}
