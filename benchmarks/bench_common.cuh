#pragma once

#include "benchmark_utils.cuh"
#include "cuda_check.cuh"

#include <cuda_runtime.h>

#include <functional>
#include <iostream>
#include <string>
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

inline std::string json_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char character : value) {
        switch (character) {
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped += character; break;
        }
    }
    return escaped;
}

inline std::string current_device_name() {
    int device = 0;
    cudaDeviceProp properties{};
    GPU_LAB_CUDA_CHECK(cudaGetDevice(&device));
    GPU_LAB_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    return properties.name;
}

inline void print_summary(const char* family, const char* variant, const std::string& shape_json,
                          std::size_t elements, const Summary& summary,
                          std::size_t bytes_moved, int warmup, int iterations,
                          const char* baseline_variant = nullptr) {
    std::cout << "{\"family\":\"" << family << "\",\"variant\":\"" << variant
              << "\",\"shape\":" << shape_json << ",\"dtype\":\"float32\""
              << ",\"device\":\"" << json_escape(current_device_name()) << "\""
              << ",\"backend\":\"cuda\",\"warmup\":" << warmup
              << ",\"iterations\":" << iterations << ",\"elements\":" << elements
              << ",\"median_ms\":" << summary.median_ms << ",\"mean_ms\":" << summary.mean_ms
              << ",\"min_ms\":" << summary.min_ms << ",\"std_ms\":" << summary.std_ms;
    if (bytes_moved != 0) {
        std::cout << ",\"effective_bandwidth_gbps\":"
                  << effective_bandwidth_gbps(bytes_moved, summary.median_ms);
    } else {
        std::cout << ",\"effective_bandwidth_gbps\":null";
    }
    if (baseline_variant == nullptr) {
        std::cout << ",\"baseline_variant\":null";
    } else {
        std::cout << ",\"baseline_variant\":\"" << baseline_variant << "\"";
    }
    std::cout << ",\"speedup\":null,\"status\":\"BENCHMARKED_CUDA\"}\n";
}

inline void print_skip(const char* family, const std::string& shape_json,
                       std::size_t required_bytes, std::size_t available_bytes,
                       const char* reason) {
    std::cout << "{\"family\":\"" << family << "\",\"variant\":\"all\",\"shape\":"
              << shape_json << ",\"dtype\":\"float32\",\"device\":\""
              << json_escape(current_device_name()) << "\",\"backend\":\"cuda\""
              << ",\"status\":\"SKIPPED\",\"reason\":\"" << reason
              << "\",\"required_bytes\":" << required_bytes
              << ",\"available_bytes\":" << available_bytes << "}\n";
}

}  // namespace gpu_lab::bench
