#pragma once

#include "benchmark_utils.cuh"
#include "cuda_check.cuh"

#include <cuda_runtime.h>

#include <functional>
#include <iostream>
#include <vector>

namespace gpu_lab::bench {

template <typename Launch>
Summary measure_cuda(Launch&& launch, int warmup = 20, int iterations = 100) {
    for (int i = 0; i < warmup; ++i) launch();
    GPU_LAB_CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start{};
    cudaEvent_t stop{};
    GPU_LAB_CUDA_CHECK(cudaEventCreate(&start));
    GPU_LAB_CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<double> samples;
    samples.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        GPU_LAB_CUDA_CHECK(cudaEventRecord(start));
        launch();
        GPU_LAB_CUDA_CHECK(cudaEventRecord(stop));
        GPU_LAB_CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0.0f;
        GPU_LAB_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        samples.push_back(static_cast<double>(milliseconds));
    }
    GPU_LAB_CUDA_CHECK(cudaEventDestroy(start));
    GPU_LAB_CUDA_CHECK(cudaEventDestroy(stop));
    return summarize(std::move(samples));
}

inline void print_summary(const char* kernel, std::size_t elements, const Summary& summary,
                          std::size_t bytes_moved = 0) {
    std::cout << "{\"kernel\":\"" << kernel << "\",\"elements\":" << elements
              << ",\"median_ms\":" << summary.median_ms << ",\"mean_ms\":" << summary.mean_ms
              << ",\"min_ms\":" << summary.min_ms << ",\"std_ms\":" << summary.std_ms;
    if (bytes_moved != 0) {
        std::cout << ",\"effective_bandwidth_gbps\":"
                  << effective_bandwidth_gbps(bytes_moved, summary.median_ms);
    }
    std::cout << "}\n";
}

}  // namespace gpu_lab::bench

