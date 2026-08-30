cmake_minimum_required(VERSION 3.20)
project(gpu_kernel_performance_lab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(gpu_kernels STATIC
        kernels/vector_add.cu
        kernels/transpose.cu
        kernels/reduction.cu
        kernels/gemm.cu
    )
    target_include_directories(gpu_kernels PUBLIC include)
    set_target_properties(gpu_kernels PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )

    foreach(program bench_vector_add bench_transpose bench_reduction bench_gemm)
        add_executable(${program} ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/${program}.cu)
        target_include_directories(${program} PRIVATE include)
        target_link_libraries(${program} PRIVATE gpu_kernels CUDA::cudart)
    endforeach()
    foreach(program test_vector_add test_transpose test_reduction test_gemm)
        add_executable(${program} ${CMAKE_CURRENT_SOURCE_DIR}/tests/${program}.cu)
        target_include_directories(${program} PRIVATE include)
        target_link_libraries(${program} PRIVATE gpu_kernels CUDA::cudart)
    endforeach()

    target_link_libraries(bench_gemm PRIVATE CUDA::cublas)
else()
    message(STATUS "CUDA compiler not found: configuring CPU reference smoke test only")
    add_executable(cpu_smoke_tests tests/host_smoke.cpp)
    target_include_directories(cpu_smoke_tests PRIVATE include)
endif()
