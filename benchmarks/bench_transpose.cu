#include "bench_common.cuh"
#include "kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

template <typename Launch>
void run_variant(const char* variant, const char* baseline_variant, Launch&& launch,
                 float* output, const std::vector<float>& expected,
                 std::size_t side, std::size_t elements, std::size_t bytes) {
    constexpr int warmup = 20;
    constexpr int iterations = 100;
    launch();
    GPU_LAB_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> actual(expected.size());
    GPU_LAB_CUDA_CHECK(cudaMemcpy(actual.data(), output, bytes, cudaMemcpyDeviceToHost));
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (actual[index] != expected[index]) {
            throw std::runtime_error(std::string("transpose correctness failed for ") + variant);
        }
    }
    const auto summary = gpu_lab::bench::measure_cuda(std::forward<Launch>(launch), warmup,
                                                      iterations);
    gpu_lab::bench::print_summary("transpose", variant,
                                  "[" + std::to_string(side) + "," + std::to_string(side) + "]",
                                  elements, summary, 2 * bytes, warmup, iterations,
                                  baseline_variant);
}

}  // namespace

int main() {
    for (const std::size_t side : {1024u, 2048u, 4096u}) {
        const std::size_t elements = side * side;
        const std::size_t bytes = elements * sizeof(float);
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        GPU_LAB_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        const std::size_t required_bytes = 2 * bytes;
        if (required_bytes > free_bytes / 2) {
            gpu_lab::bench::print_skip(
                "transpose", "[" + std::to_string(side) + "," + std::to_string(side) + "]",
                required_bytes, free_bytes, "insufficient device memory with safety headroom");
            continue;
        }
        float* input = nullptr;
        float* output = nullptr;
        std::vector<float> host_input(elements);
        std::vector<float> expected(elements);
        for (std::size_t row = 0; row < side; ++row) {
            for (std::size_t col = 0; col < side; ++col) {
                const float value = static_cast<float>((row * side + col) % 251) * 0.25f;
                host_input[row * side + col] = value;
                expected[col * side + row] = value;
            }
        }
        GPU_LAB_CUDA_CHECK(cudaMalloc(&input, bytes));
        GPU_LAB_CUDA_CHECK(cudaMalloc(&output, bytes));
        GPU_LAB_CUDA_CHECK(cudaMemcpy(input, host_input.data(), bytes, cudaMemcpyHostToDevice));
        run_variant("copy_baseline", nullptr, [&] {
            gpu_lab::cuda::launch_copy_baseline(input, output, side, side);
        }, output, host_input, side, elements, bytes);
        run_variant("transpose_naive", nullptr, [&] {
            gpu_lab::cuda::launch_transpose_naive(input, output, side, side);
        }, output, expected, side, elements, bytes);
        run_variant("transpose_tiled", "transpose_naive", [&] {
            gpu_lab::cuda::launch_transpose_tiled(input, output, side, side);
        }, output, expected, side, elements, bytes);
        run_variant("transpose_tiled_padded", "transpose_naive", [&] {
            gpu_lab::cuda::launch_transpose_tiled_padded(input, output, side, side);
        }, output, expected, side, elements, bytes);
        GPU_LAB_CUDA_CHECK(cudaFree(input));
        GPU_LAB_CUDA_CHECK(cudaFree(output));
    }
    return 0;
}
