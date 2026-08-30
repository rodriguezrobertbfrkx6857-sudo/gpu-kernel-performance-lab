#include "bench_common.cuh"
#include "kernels.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <iostream>
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
    constexpr int m = 512;
    constexpr int n = 512;
    constexpr int k = 512;
    const std::size_t a_bytes = static_cast<std::size_t>(m) * k * sizeof(float);
    const std::size_t b_bytes = static_cast<std::size_t>(k) * n * sizeof(float);
    const std::size_t c_bytes = static_cast<std::size_t>(m) * n * sizeof(float);
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    GPU_LAB_CUDA_CHECK(cudaMalloc(&a, a_bytes));
    GPU_LAB_CUDA_CHECK(cudaMalloc(&b, b_bytes));
    GPU_LAB_CUDA_CHECK(cudaMalloc(&c, c_bytes));
    GPU_LAB_CUDA_CHECK(cudaMemset(a, 1, a_bytes));
    GPU_LAB_CUDA_CHECK(cudaMemset(b, 1, b_bytes));
    const auto naive = gpu_lab::bench::measure_cuda([&] {
        gpu_lab::cuda::launch_gemm_naive(a, b, c, m, n, k);
    }, 20, 50);
    gpu_lab::bench::print_summary("gemm_naive", static_cast<std::size_t>(m) * n, naive);
    const auto tiled = gpu_lab::bench::measure_cuda([&] {
        gpu_lab::cuda::launch_gemm_tiled(a, b, c, m, n, k);
    }, 20, 50);
    gpu_lab::bench::print_summary("gemm_tiled", static_cast<std::size_t>(m) * n, tiled);

    cublasHandle_t handle = nullptr;
    check_cublas(cublasCreate(&handle), "cublasCreate");
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const auto cublas_summary = gpu_lab::bench::measure_cuda([&] {
        check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                                 b, n, a, k, &beta, c, n), "cublasSgemm");
    }, 20, 50);
    gpu_lab::bench::print_summary("cublas_sgemm", static_cast<std::size_t>(m) * n, cublas_summary);
    check_cublas(cublasDestroy(handle), "cublasDestroy");
    GPU_LAB_CUDA_CHECK(cudaFree(a));
    GPU_LAB_CUDA_CHECK(cudaFree(b));
    GPU_LAB_CUDA_CHECK(cudaFree(c));
    return 0;
}
