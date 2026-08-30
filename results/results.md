# Benchmark Results

This report is generated from `benchmark_results.json`; no timing value is maintained by hand.

- Hardware mode: `cpu_only`
- Timing protocol: CUDA Event for CUDA binaries; perf_counter_ns for CPU fallback
- CUDA defaults: 20 warm-up launches and 100 measured launches when a CUDA binary is available.
- CPU fallback: adaptive warm-up and iteration counts are recorded in the JSON protocol.

| Family | Variant | Shape | Median ms | Mean ms | Min ms | Std ms | Speedup | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| vector_add | vector_add_cpu_numpy | `[1048576]` | 1.631700 | 1.557060 | 1.063200 | 0.360195 | — | BENCHMARKED_CPU_ONLY |
| vector_add | vector_add_cpu_numpy | `[4194304]` | 7.375950 | 7.153390 | 5.600100 | 0.799571 | — | BENCHMARKED_CPU_ONLY |
| transpose | copy_baseline | `[512, 512]` | 0.357750 | 0.362270 | 0.333800 | 0.015551 | — | BENCHMARKED_CPU_ONLY |
| transpose | transpose_naive | `[512, 512]` | 0.424650 | 0.423340 | 0.408900 | 0.008600 | 0.842x | BENCHMARKED_CPU_ONLY |
| transpose | transpose_tiled | `[512, 512]` | 0.839350 | 0.841650 | 0.772900 | 0.032663 | 0.426x | BENCHMARKED_CPU_ONLY |
| transpose | transpose_tiled_padded | `[512, 512]` | 0.953050 | 1.101980 | 0.889700 | 0.294442 | 0.375x | BENCHMARKED_CPU_ONLY |
| transpose | copy_baseline | `[1024, 1024]` | 1.227800 | 1.398030 | 1.119600 | 0.286713 | — | BENCHMARKED_CPU_ONLY |
| transpose | transpose_naive | `[1024, 1024]` | 5.934900 | 6.644410 | 5.556400 | 1.703757 | 0.207x | BENCHMARKED_CPU_ONLY |
| transpose | transpose_tiled | `[1024, 1024]` | 4.830600 | 4.668210 | 3.322000 | 0.637731 | 0.254x | BENCHMARKED_CPU_ONLY |
| transpose | transpose_tiled_padded | `[1024, 1024]` | 6.880800 | 7.126700 | 5.726000 | 1.251507 | 0.178x | BENCHMARKED_CPU_ONLY |
| reduction | reduction_numpy_pairwise | `[1048576]` | 0.840950 | 0.860680 | 0.826800 | 0.044164 | — | BENCHMARKED_CPU_ONLY |
| reduction | reduction_numpy_sum | `[1048576]` | 0.827600 | 0.730790 | 0.483100 | 0.142177 | 1.016x | BENCHMARKED_CPU_ONLY |
| reduction | reduction_numpy_pairwise | `[4194304]` | 3.471000 | 3.199190 | 2.316500 | 0.573857 | — | BENCHMARKED_CPU_ONLY |
| reduction | reduction_numpy_sum | `[4194304]` | 3.532000 | 3.672520 | 2.316800 | 1.157297 | 0.983x | BENCHMARKED_CPU_ONLY |
| reduction | reduction_numpy_pairwise | `[16777216]` | 15.335450 | 15.903300 | 12.131000 | 2.949788 | — | BENCHMARKED_CPU_ONLY |
| reduction | reduction_numpy_sum | `[16777216]` | 16.353200 | 17.794420 | 13.726600 | 3.800696 | 0.938x | BENCHMARKED_CPU_ONLY |
| gemm | gemm_numpy_blas | `[128, 128, 128]` | 0.422500 | 0.444700 | 0.331200 | 0.075459 | — | BENCHMARKED_CPU_ONLY |
| gemm | gemm_blocked_cpu | `[128, 128, 128]` | 0.290900 | 0.483160 | 0.290400 | 0.383521 | 1.452x | BENCHMARKED_CPU_ONLY |
| gemm | gemm_numpy_blas | `[256, 256, 256]` | 0.563000 | 0.569020 | 0.480700 | 0.066741 | — | BENCHMARKED_CPU_ONLY |
| gemm | gemm_blocked_cpu | `[256, 256, 256]` | 8.570700 | 8.651680 | 8.385100 | 0.291957 | 0.066x | BENCHMARKED_CPU_ONLY |

## Environment

- OS: Windows 11
- CPU: Intel64 Family 6 Model 170 Stepping 4, GenuineIntel
- Hardware mode: `cpu_only`

CUDA benchmark status: `NOT BENCHMARKED ON CURRENT HARDWARE`.
The numbers above are CPU reference measurements and must not be interpreted as GPU results.
