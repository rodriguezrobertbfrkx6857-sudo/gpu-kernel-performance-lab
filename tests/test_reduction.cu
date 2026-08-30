#include "cuda_check.cuh"
#include "kernels.cuh"
#include "test_utils.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

int main() {
    constexpr std::size_t n = 1u << 20;
    const std::size_t bytes = n * sizeof(float);
    std::vector<float> input(n);
    for (std::size_t i = 0; i < n; ++i) input[i] = std::sin(static_cast<float>(i) * 0.001f);
    const double expected = std::accumulate(input.begin(), input.end(), 0.0);
    float* device_input = nullptr;
    float* device_output = nullptr;
    GPU_LAB_CUDA_CHECK(cudaMalloc(&device_input, bytes));
    GPU_LAB_CUDA_CHECK(cudaMalloc(&device_output, sizeof(float)));
    GPU_LAB_CUDA_CHECK(cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice));
    const struct Variant {
        const char* name;
        void (*launch)(const float*, float*, std::size_t, gpu_lab_stream_t);
    } variants[] = {
        {"reduction_naive", gpu_lab::cuda::launch_reduction_naive},
        {"reduction_shared", gpu_lab::cuda::launch_reduction_shared},
        {"reduction_warp_shuffle", gpu_lab::cuda::launch_reduction_warp_shuffle},
    };
    for (const auto& variant : variants) {
        variant.launch(device_input, device_output, n, nullptr);
        GPU_LAB_CUDA_CHECK(cudaDeviceSynchronize());
        float actual = 0.0f;
        GPU_LAB_CUDA_CHECK(cudaMemcpy(&actual, device_output, sizeof(float), cudaMemcpyDeviceToHost));
        gpu_lab::test::require_near(actual, static_cast<float>(expected), 2.0e-2f, 2.0e-5f,
                                    variant.name);
    }
    gpu_lab::test::pass("reduction variants");
    GPU_LAB_CUDA_CHECK(cudaFree(device_input));
    GPU_LAB_CUDA_CHECK(cudaFree(device_output));
    return 0;
}

