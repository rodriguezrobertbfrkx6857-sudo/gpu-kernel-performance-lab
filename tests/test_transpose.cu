#include "cuda_check.cuh"
#include "kernels.cuh"
#include "test_utils.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

int main() {
    constexpr std::size_t rows = 65;
    constexpr std::size_t cols = 97;
    const std::size_t input_bytes = rows * cols * sizeof(float);
    const std::size_t output_bytes = cols * rows * sizeof(float);
    std::vector<float> input(rows * cols), expected(cols * rows), actual(cols * rows);
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i) * 0.5f;
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            expected[col * rows + row] = input[row * cols + col];
        }
    }
    float* device_input = nullptr;
    float* device_output = nullptr;
    GPU_LAB_CUDA_CHECK(cudaMalloc(&device_input, input_bytes));
    GPU_LAB_CUDA_CHECK(cudaMalloc(&device_output, output_bytes));
    GPU_LAB_CUDA_CHECK(cudaMemcpy(device_input, input.data(), input_bytes, cudaMemcpyHostToDevice));
    const struct Variant {
        const char* name;
        void (*launch)(const float*, float*, std::size_t, std::size_t, gpu_lab_stream_t);
    } variants[] = {
        {"transpose_naive", gpu_lab::cuda::launch_transpose_naive},
        {"transpose_tiled", gpu_lab::cuda::launch_transpose_tiled},
        {"transpose_tiled_padded", gpu_lab::cuda::launch_transpose_tiled_padded},
    };
    for (const auto& variant : variants) {
        variant.launch(device_input, device_output, rows, cols, nullptr);
        GPU_LAB_CUDA_CHECK(cudaDeviceSynchronize());
        GPU_LAB_CUDA_CHECK(cudaMemcpy(actual.data(), device_output, output_bytes,
                                      cudaMemcpyDeviceToHost));
        gpu_lab::test::require_array_near(actual.data(), expected.data(), expected.size(),
                                          0.0, 0.0, variant.name);
    }
    gpu_lab::test::pass("transpose variants");
    GPU_LAB_CUDA_CHECK(cudaFree(device_input));
    GPU_LAB_CUDA_CHECK(cudaFree(device_output));
    return 0;
}

