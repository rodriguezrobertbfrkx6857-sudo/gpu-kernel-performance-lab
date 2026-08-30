#pragma once

#include <stdexcept>
#include <string>

#if defined(__CUDACC__)
#include <cuda_runtime.h>

namespace gpu_lab {
inline void check_cuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA failure at ") + file + ":" + std::to_string(line) +
                                 " for " + expression + ": " + cudaGetErrorString(status));
    }
}
}  // namespace gpu_lab

#define GPU_LAB_CUDA_CHECK(expression) \
    ::gpu_lab::check_cuda((expression), #expression, __FILE__, __LINE__)

#else

#define GPU_LAB_CUDA_CHECK(expression) \
    do { (void)sizeof(expression); } while (false)

#endif

