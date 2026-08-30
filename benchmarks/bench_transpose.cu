#include "bench_common.cuh"
#include "kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

namespace {

template <typename Launch>
void run_variant(const char* name, Launch&& launch, std::size_t elements, std::size_t bytes) {
    const auto summary = gpu_lab::bench::measure_cuda(std::forward<Launch>(launch));
    gpu_lab::bench::print_summary(name, elements, summary, 2 * bytes);
}

}  // namespace

int main() {
    for (const std::size_t side : {1024u, 2048u, 4096u}) {
        const std::size_t elements = side * side;
        const std::size_t bytes = elements * sizeof(float);
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        GPU_LAB_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (bytes * 2 > free_bytes / 2) {
            std::cout << "{\"status\":\"SKIPPED_INSUFFICIENT_DEVICE_MEMORY\",\"shape\":["
                      << side << "," << side << "]}\n";
            continue;
        }
        float* input = nullptr;
        float* output = nullptr;
        GPU_LAB_CUDA_CHECK(cudaMalloc(&input, bytes));
        GPU_LAB_CUDA_CHECK(cudaMalloc(&output, bytes));
        GPU_LAB_CUDA_CHECK(cudaMemset(input, 0, bytes));
        run_variant("copy_baseline", [&] {
            gpu_lab::cuda::launch_copy_baseline(input, output, side, side);
        }, elements, bytes);
        run_variant("transpose_naive", [&] {
            gpu_lab::cuda::launch_transpose_naive(input, output, side, side);
        }, elements, bytes);
        run_variant("transpose_tiled", [&] {
            gpu_lab::cuda::launch_transpose_tiled(input, output, side, side);
        }, elements, bytes);
        run_variant("transpose_tiled_padded", [&] {
            gpu_lab::cuda::launch_transpose_tiled_padded(input, output, side, side);
        }, elements, bytes);
        GPU_LAB_CUDA_CHECK(cudaFree(input));
        GPU_LAB_CUDA_CHECK(cudaFree(output));
    }
    return 0;
}
