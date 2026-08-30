# Profiling case study

The intended case study is the tiled GEMM or warp-shuffle reduction. On a CUDA host, build the Release binaries, run the correctness executable first, then collect one focused kernel with Nsight Compute or Nsight Systems. For GEMM, the repository provides a selector such as:

```text
build/Release/bench_gemm.exe --profile-case gemm_tiled-512
```

The questions to record are kernel duration, achieved memory throughput, compute utilization, occupancy, and warp stall reasons.

The current environment audit found no `ncu` or `nsys` executable and no NVIDIA driver, so no profiler session was run for this checkout. The correct status is `Nsight profiling was not available in the current environment.` A future run should store the command, tool version, GPU model, and raw profiler export alongside the generated result instead of transferring any values from another machine.
