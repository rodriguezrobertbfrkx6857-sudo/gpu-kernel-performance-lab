#include "bench_common.cuh"
#include "kernels.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void check_cublas(cublasStatus_t status, const char* expression) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(expression) + " failed with cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

template <typename Launch>
void run_variant(const char* variant, const char* baseline_variant, Launch&& launch, int side,
                 int iterations) {
    constexpr int warmup = 20;
    const auto summary = gpu_lab::bench::measure_cuda(std::forward<Launch>(launch), warmup,
                                                      iterations);
    const std::size_t elements = static_cast<std::size_t>(side) * side;
    const std::string shape = "[" + std::to_string(side) + "," + std::to_string(side) + "," +
                              std::to_string(side) + "]";
    gpu_lab::bench::print_summary("gemm", variant, shape, elements, summary, 0, warmup,
                                  iterations, baseline_variant);
}

bool case_matches(const std::string& profile_case, const char* variant, int side) {
    return profile_case.empty() ||
           profile_case == std::string(variant) + "-" + std::to_string(side);
}

void run_case(int side, int iterations, const std::string& profile_case) {
    const std::size_t matrix_elements = static_cast<std::size_t>(side) * side;
    const std::size_t matrix_bytes = matrix_elements * sizeof(float);
    const std::size_t required_bytes = 3 * matrix_bytes;
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    GPU_LAB_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const std::string shape = "[" + std::to_string(side) + "," + std::to_string(side) + "," +
                              std::to_string(side) + "]";
    if (required_bytes > free_bytes / 2) {
        gpu_lab::bench::print_skip("gemm", shape, required_bytes, free_bytes,
                                   "insufficient device memory with safety headroom");
        return;
    }

    std::vector<float> host_a(matrix_elements, 1.0f);
    std::vector<float> host_b(matrix_elements, 1.0f);
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    GPU_LAB_CUDA_CHECK(cudaMalloc(&a, matrix_bytes));
    GPU_LAB_CUDA_CHECK(cudaMalloc(&b, matrix_bytes));
    GPU_LAB_CUDA_CHECK(cudaMalloc(&c, matrix_bytes));
    GPU_LAB_CUDA_CHECK(cudaMemcpy(a, host_a.data(), matrix_bytes, cudaMemcpyHostToDevice));
    GPU_LAB_CUDA_CHECK(cudaMemcpy(b, host_b.data(), matrix_bytes, cudaMemcpyHostToDevice));

    if (!profile_case.empty()) {
        const std::string suffix = "-" + std::to_string(side);
        if (profile_case.size() <= suffix.size() ||
            profile_case.compare(profile_case.size() - suffix.size(), suffix.size(), suffix) != 0) {
            GPU_LAB_CUDA_CHECK(cudaFree(a));
            GPU_LAB_CUDA_CHECK(cudaFree(b));
            GPU_LAB_CUDA_CHECK(cudaFree(c));
            return;
        }
    }

    if (case_matches(profile_case, "gemm_naive", side)) {
        run_variant("gemm_naive", nullptr, [&] {
            gpu_lab::cuda::launch_gemm_naive(a, b, c, side, side, side);
        }, side, iterations);
    }
    if (case_matches(profile_case, "gemm_tiled", side)) {
        run_variant("gemm_tiled", "gemm_naive", [&] {
            gpu_lab::cuda::launch_gemm_tiled(a, b, c, side, side, side);
        }, side, iterations);
    }

    if (case_matches(profile_case, "cublas_sgemm", side)) {
        cublasHandle_t handle = nullptr;
        check_cublas(cublasCreate(&handle), "cublasCreate");
        const float alpha = 1.0f;
        const float beta = 0.0f;
        // cuBLAS consumes column-major matrices. Passing B as the first operand and A as the
        // second computes the row-major C = A * B into the row-major buffer C.
        run_variant("cublas_sgemm", nullptr, [&] {
            check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, side, side, side, &alpha,
                                     b, side, a, side, &beta, c, side),
                         "cublasSgemm");
        }, side, iterations);
        check_cublas(cublasDestroy(handle), "cublasDestroy");
    }

    GPU_LAB_CUDA_CHECK(cudaFree(a));
    GPU_LAB_CUDA_CHECK(cudaFree(b));
    GPU_LAB_CUDA_CHECK(cudaFree(c));
}

}  // namespace

int main(int argc, char** argv) {
    std::string profile_case;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--profile-case" && index + 1 < argc) {
            profile_case = argv[++index];
        } else if (argument == "--help") {
            std::cout << "usage: bench_gemm [--profile-case variant-side]\n";
            return 0;
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }

    for (const int side : {256, 512, 1024}) {
        run_case(side, side >= 1024 ? 20 : 50, profile_case);
    }
    return 0;
}
