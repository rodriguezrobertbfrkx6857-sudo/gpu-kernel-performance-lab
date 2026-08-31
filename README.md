# GPU Kernel Performance Lab：CUDA 内核性能工程实验

一套以测量为中心的 CUDA 性能工程实验，覆盖内存访问、矩阵转置、归约、Warp 级操作和分块 GEMM。仓库同时保留 CPU 参考路径，因此没有 NVIDIA 硬件时也能运行正确性测试、生成结构化报告，并为后续上机复现实验保留完整 CUDA 源码。

当前提交的报告是在本机 `cpu_only` 模式生成的真实结果，不包含 GPU 时间或 GPU 加速结论。CUDA 数据必须在安装 NVIDIA 驱动和 CUDA Toolkit 的机器上重新测量。

## 实验内容

- 向量加法：设备分配、启动配置和带宽核算。
- 转置：复制基线、朴素全局内存、`tile[32][32]` 以及规避 bank conflict 的 `tile[32][33]` 共享内存版本。
- 归约：交错原子累加、共享内存归约和使用 `__shfl_down_sync` 的 Warp shuffle 归约。
- GEMM：朴素版本、共享内存分块版本，以及 CUDA 构建可用时的 cuBLAS 对照。
- 基准协议：正确性、预热、重复计时、同步、统计摘要和 JSON/Markdown 报告。

## 快速开始

CPU 参考路径只需要 Python 和 NumPy：

```powershell
python scripts/run_tests.py
python scripts/run_all_benchmarks.py
python scripts/generate_report.py
```

基准记录 `median_ms`、`mean_ms`、`min_ms`、`std_ms`、数据类型、设备、后端、预热次数、测量次数和有明确基线的相对加速比。CPU 计时使用自适应次数，并把实际次数写入 JSON。

## CUDA 构建

在具备 CMake、C++ 编译器和 CUDA Toolkit 的机器上：

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

CMake 会自动检测 CUDA。没有 CUDA 编译器时，只构建并注册 `cpu_smoke_tests`，`.cu` 源码仍可检查并等待后续上机。需要显式指定架构时，可使用 `-DGPU_LAB_CUDA_ARCHITECTURES=all` 或具体架构。

## 测量口径

CUDA 使用 CUDA Events 计时，并在停止事件处同步；正确性始终先于预热和性能测量。转置报告读写总字节数除以中位时间得到的有效带宽，GEMM 在可用时与 cuBLAS 对比，不预设自定义内核必须超过厂商库。大规模用例会检查剩余显存，并在安全余量不足时写入结构化跳过记录。

当前机器的结果见 [`results/results.md`](results/results.md)，机器可读数据见 [`results/benchmark_results.json`](results/benchmark_results.json)。由于本机没有 NVIDIA 驱动和 CUDA Toolkit，报告状态为 `NOT BENCHMARKED ON CURRENT HARDWARE`，记录行是 CPU 参考结果。

## 文档与复现

- [`docs/execution_model.md`](docs/execution_model.md)：Grid、Block、Thread、Warp 和 SIMT 执行模型。
- [`docs/memory_optimization.md`](docs/memory_optimization.md)：全局内存合并访问、共享内存和 bank conflict 填充。
- [`docs/profiling_case_study.md`](docs/profiling_case_study.md)：没有 Nsight 时的分析边界。

环境快照由 [`scripts/detect_environment.py`](scripts/detect_environment.py) 生成。每次硬件或编译器变化后重新执行：

```powershell
python scripts/detect_environment.py --output-dir results
python scripts/run_all_benchmarks.py
python scripts/generate_report.py
```

## 限制

- 当前已记录环境没有 NVIDIA GPU、驱动、CUDA Toolkit 或 Nsight。
- CPU 路径用于验证算法行为和基准流程，不能替代 GPU 吞吐数据。
- 归约基线是用于教学和对照的简单原子累加，不作为生产实现推荐。
- 寄存器分块和 Tensor Core 路径不在当前维护范围内，除非有通过正确性测试的实际工作负载支持。
