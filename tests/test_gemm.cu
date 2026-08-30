#include "cuda_check.cuh"
#include "kernels.cuh"
#include "test_utils.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cublas(cublasStatus_t status, const char* expression) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(expression) + " failed with cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

}  // namespace

int main() {
    constexpr int m = 37;
    constexpr int n = 29;
    constexpr int k = 23;
    std::vector<float> a(m * k), b(k * n), expected(m * n), actual(m * n);
    for (int row = 0; row < m; ++row) {
        for (int inner = 0; inner < k; ++inner) a[row * k + inner] = std::sin(row + inner * 0.25f);
    }
    for (int inner = 0; inner < k; ++inner) {
        for (int col = 0; col < n; ++col) b[inner * n + col] = std::cos(inner - col * 0.125f);
    }
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            for (int inner = 0; inner < k; ++inner) {
                expected[row * n + col] += a[row * k + inner] * b[inner * n + col];
            }
        }
    }
    float* da = nullptr;
    float* db = nullptr;
    float* dc = nullptr;
    GPU_LAB_CUDA_CHECK(cudaMalloc(&da, a.size() * sizeof(float)));
    GPU_LAB_CUDA_CHECK(cudaMalloc(&db, b.size() * sizeof(float)));
    GPU_LAB_CUDA_CHECK(cudaMalloc(&dc, expected.size() * sizeof(float)));
    GPU_LAB_CUDA_CHECK(cudaMemcpy(da, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
    GPU_LAB_CUDA_CHECK(cudaMemcpy(db, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));
    const struct Variant {
        const char* name;
        void (*launch)(const float*, const float*, float*, int, int, int, gpu_lab_stream_t);
    } variants[] = {
        {"gemm_naive", gpu_lab::cuda::launch_gemm_naive},
        {"gemm_tiled", gpu_lab::cuda::launch_gemm_tiled},
    };
    for (const auto& variant : variants) {
        variant.launch(da, db, dc, m, n, k, nullptr);
        GPU_LAB_CUDA_CHECK(cudaDeviceSynchronize());
        GPU_LAB_CUDA_CHECK(cudaMemcpy(actual.data(), dc, actual.size() * sizeof(float),
                                      cudaMemcpyDeviceToHost));
        gpu_lab::test::require_array_near(actual.data(), expected.data(), expected.size(),
                                          1.0e-3, 1.0e-3, variant.name);
    }

    cublasHandle_t handle = nullptr;
    check_cublas(cublasCreate(&handle), "cublasCreate");
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // For row-major C = A * B, column-major cuBLAS sees C^T = B^T * A^T.
    check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, db, n, da, k,
                             &beta, dc, n),
                 "cublasSgemm");
    GPU_LAB_CUDA_CHECK(cudaDeviceSynchronize());
    GPU_LAB_CUDA_CHECK(cudaMemcpy(actual.data(), dc, actual.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost));
    gpu_lab::test::require_array_near(actual.data(), expected.data(), expected.size(),
                                      1.0e-3, 1.0e-3, "cublas_sgemm");
    check_cublas(cublasDestroy(handle), "cublasDestroy");
    gpu_lab::test::pass("gemm variants");
    GPU_LAB_CUDA_CHECK(cudaFree(da));
    GPU_LAB_CUDA_CHECK(cudaFree(db));
    GPU_LAB_CUDA_CHECK(cudaFree(dc));
    return 0;
}
