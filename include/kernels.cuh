#pragma once

#include <cstddef>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
using gpu_lab_stream_t = cudaStream_t;
#else
using gpu_lab_stream_t = void*;
#endif

namespace gpu_lab::cuda {

void launch_vector_add(const float* a, const float* b, float* c, std::size_t n,
                       gpu_lab_stream_t stream = nullptr);

void launch_copy_baseline(const float* input, float* output, std::size_t rows, std::size_t cols,
                          gpu_lab_stream_t stream = nullptr);
void launch_transpose_naive(const float* input, float* output, std::size_t rows, std::size_t cols,
                            gpu_lab_stream_t stream = nullptr);
void launch_transpose_tiled(const float* input, float* output, std::size_t rows, std::size_t cols,
                            gpu_lab_stream_t stream = nullptr);
void launch_transpose_tiled_padded(const float* input, float* output, std::size_t rows,
                                   std::size_t cols, gpu_lab_stream_t stream = nullptr);

void launch_reduction_naive(const float* input, float* output, std::size_t n,
                            gpu_lab_stream_t stream = nullptr);
void launch_reduction_shared(const float* input, float* output, std::size_t n,
                             gpu_lab_stream_t stream = nullptr);
void launch_reduction_warp_shuffle(const float* input, float* output, std::size_t n,
                                   gpu_lab_stream_t stream = nullptr);

void launch_gemm_naive(const float* a, const float* b, float* c, int m, int n, int k,
                       gpu_lab_stream_t stream = nullptr);
void launch_gemm_tiled(const float* a, const float* b, float* c, int m, int n, int k,
                       gpu_lab_stream_t stream = nullptr);

}  // namespace gpu_lab::cuda

