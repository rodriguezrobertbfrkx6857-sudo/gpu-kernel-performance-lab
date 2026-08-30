#include "cuda_check.cuh"
#include "kernels.cuh"
#include "test_utils.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

int main() {
    // A zero-length launch must be a safe no-op rather than a zero-block launch.
    gpu_lab::cuda::launch_vector_add(nullptr, nullptr, nullptr, 0);

    const auto run_case = [](std::size_t n) {
        const std::size_t bytes = n * sizeof(float);
        std::vector<float> a(n), b(n), expected(n), actual(n);
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = static_cast<float>(i) * 0.001f;
            b[i] = static_cast<float>(n - i) * -0.002f;
            expected[i] = a[i] + b[i];
        }
        float* da = nullptr;
        float* db = nullptr;
        float* dc = nullptr;
        GPU_LAB_CUDA_CHECK(cudaMalloc(&da, bytes));
        GPU_LAB_CUDA_CHECK(cudaMalloc(&db, bytes));
        GPU_LAB_CUDA_CHECK(cudaMalloc(&dc, bytes));
        GPU_LAB_CUDA_CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
        GPU_LAB_CUDA_CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
        gpu_lab::cuda::launch_vector_add(da, db, dc, n);
        GPU_LAB_CUDA_CHECK(cudaDeviceSynchronize());
        GPU_LAB_CUDA_CHECK(cudaMemcpy(actual.data(), dc, bytes, cudaMemcpyDeviceToHost));
        gpu_lab::test::require_array_near(actual.data(), expected.data(), n, 1.0e-6, 1.0e-6,
                                          "vector add");
        GPU_LAB_CUDA_CHECK(cudaFree(da));
        GPU_LAB_CUDA_CHECK(cudaFree(db));
        GPU_LAB_CUDA_CHECK(cudaFree(dc));
    };

    run_case(100003);            // non-multiple of the 256-thread block
    run_case((1u << 20) + 17u);  // larger input with a guarded tail
    gpu_lab::test::pass("vector_add zero, irregular, and large inputs");
    return 0;
}
