#include "bench_common.cuh"
#include "kernels.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

template <typename Launch>
void run_variant(const char* variant, const char* baseline_variant, Launch&& launch,
                 const float* input, float* output,
                 std::size_t n, double expected, int iterations) {
    constexpr int warmup = 20;
    launch();
    GPU_LAB_CUDA_CHECK(cudaDeviceSynchronize());
    float result = 0.0f;
    GPU_LAB_CUDA_CHECK(cudaMemcpy(&result, output, sizeof(float), cudaMemcpyDeviceToHost));
    const double tolerance = 2.0e-2 + 2.0e-5 * std::fabs(expected);
    if (std::fabs(static_cast<double>(result) - expected) > tolerance) {
        throw std::runtime_error(std::string("reduction correctness failed for ") + variant);
    }
    const auto summary = gpu_lab::bench::measure_cuda(std::forward<Launch>(launch), warmup, iterations);
    GPU_LAB_CUDA_CHECK(cudaMemcpy(&result, output, sizeof(float), cudaMemcpyDeviceToHost));
    gpu_lab::bench::print_summary("reduction", variant, "[" + std::to_string(n) + "]", n,
                                  summary, n * sizeof(float), warmup, iterations,
                                  baseline_variant);
    std::cout << "{\"family\":\"reduction\",\"variant\":\"" << variant
              << "\",\"kernel_result\":true,\"value\":" << result
              << ",\"expected\":" << expected << "}\n";
}

}  // namespace

int main() {
    for (const std::size_t n : {std::size_t{1} << 20, std::size_t{1} << 24,
                                std::size_t{1} << 26}) {
        const std::size_t bytes = n * sizeof(float);
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        GPU_LAB_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        const std::size_t required_bytes = bytes + sizeof(float);
        if (required_bytes > free_bytes / 2) {
            gpu_lab::bench::print_skip("reduction", "[" + std::to_string(n) + "]",
                                       required_bytes, free_bytes,
                                       "insufficient device memory with safety headroom");
            continue;
        }
        float* input = nullptr;
        float* output = nullptr;
        std::vector<float> host_input(n);
        for (std::size_t index = 0; index < n; ++index) {
            host_input[index] = std::sin(static_cast<float>(index) * 0.001f);
        }
        const double expected = std::accumulate(host_input.begin(), host_input.end(), 0.0);
        GPU_LAB_CUDA_CHECK(cudaMalloc(&input, bytes));
        GPU_LAB_CUDA_CHECK(cudaMalloc(&output, sizeof(float)));
        GPU_LAB_CUDA_CHECK(cudaMemcpy(input, host_input.data(), bytes, cudaMemcpyHostToDevice));
        const int iterations = n >= (std::size_t{1} << 26) ? 20 : 100;
        run_variant("reduction_naive", nullptr, [&] {
            gpu_lab::cuda::launch_reduction_naive(input, output, n);
        }, input, output, n, expected, iterations);
        run_variant("reduction_shared", "reduction_naive", [&] {
            gpu_lab::cuda::launch_reduction_shared(input, output, n);
        }, input, output, n, expected, iterations);
        run_variant("reduction_warp_shuffle", "reduction_naive", [&] {
            gpu_lab::cuda::launch_reduction_warp_shuffle(input, output, n);
        }, input, output, n, expected, iterations);
        GPU_LAB_CUDA_CHECK(cudaFree(input));
        GPU_LAB_CUDA_CHECK(cudaFree(output));
    }
    return 0;
}
