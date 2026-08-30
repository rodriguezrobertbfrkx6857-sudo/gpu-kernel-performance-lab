# Benchmark Results

This report is generated from `benchmark_results.json`; no timing value is maintained by hand.

- Hardware mode: `cpu_only`
- Timing protocol: CUDA Event for CUDA binaries; perf_counter_ns for CPU fallback
- CUDA defaults: 20 warm-up launches and 100 measured launches when a CUDA binary is available.
- CPU fallback: adaptive warm-up and iteration counts are recorded in the JSON protocol.
- Speedup rule: declared baseline median divided by candidate median within the same family, shape, and dtype

| Family | Variant | Shape | Backend | Device | Median ms | Mean ms | Min ms | Std ms | Baseline | Speedup | Status |
|---|---|---|---|---|---:|---:|---:|---:|---|---:|---|
| vector_add | vector_add_cpu_numpy | `[1048576]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 1.200250 | 1.239770 | 1.135800 | 0.132635 | — | — | BENCHMARKED_CPU_ONLY |
| vector_add | vector_add_cpu_numpy | `[16777216]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 35.876000 | 36.067720 | 29.405600 | 4.748078 | — | — | BENCHMARKED_CPU_ONLY |
| transpose | copy_baseline | `[1024, 1024]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 1.086750 | 1.244810 | 0.885400 | 0.357894 | — | — | BENCHMARKED_CPU_ONLY |
| transpose | transpose_naive | `[1024, 1024]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 7.654550 | 7.421150 | 4.245000 | 1.420445 | — | — | BENCHMARKED_CPU_ONLY |
| transpose | transpose_tiled | `[1024, 1024]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 3.087950 | 3.046600 | 2.450700 | 0.455552 | transpose_naive | 2.479x | BENCHMARKED_CPU_ONLY |
| transpose | transpose_tiled_padded | `[1024, 1024]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 6.889500 | 7.146920 | 5.602100 | 1.043122 | transpose_naive | 1.111x | BENCHMARKED_CPU_ONLY |
| transpose | copy_baseline | `[2048, 2048]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 4.617450 | 4.711830 | 4.052400 | 0.378188 | — | — | BENCHMARKED_CPU_ONLY |
| transpose | transpose_naive | `[2048, 2048]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 44.915700 | 46.948670 | 37.935900 | 6.687029 | — | — | BENCHMARKED_CPU_ONLY |
| transpose | transpose_tiled | `[2048, 2048]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 18.851500 | 21.584090 | 14.486700 | 6.808668 | transpose_naive | 2.383x | BENCHMARKED_CPU_ONLY |
| transpose | transpose_tiled_padded | `[2048, 2048]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 44.892350 | 45.748410 | 28.304400 | 10.220422 | transpose_naive | 1.001x | BENCHMARKED_CPU_ONLY |
| reduction | reduction_numpy_pairwise | `[1048576]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 1.054150 | 1.041450 | 0.748600 | 0.137562 | — | — | BENCHMARKED_CPU_ONLY |
| reduction | reduction_numpy_sum | `[1048576]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 0.903650 | 0.856410 | 0.464400 | 0.343488 | reduction_numpy_pairwise | 1.167x | BENCHMARKED_CPU_ONLY |
| reduction | reduction_numpy_pairwise | `[16777216]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 13.825450 | 15.814450 | 12.735400 | 3.590931 | — | — | BENCHMARKED_CPU_ONLY |
| reduction | reduction_numpy_sum | `[16777216]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 19.976350 | 20.207760 | 16.010100 | 2.934662 | reduction_numpy_pairwise | 0.692x | BENCHMARKED_CPU_ONLY |
| gemm | gemm_numpy_blas | `[256, 256, 256]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 0.602400 | 0.622900 | 0.578000 | 0.049328 | — | — | BENCHMARKED_CPU_ONLY |
| gemm | gemm_blocked_cpu | `[256, 256, 256]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 6.404200 | 6.269180 | 4.029700 | 1.986015 | gemm_numpy_blas | 0.094x | BENCHMARKED_CPU_ONLY |
| gemm | gemm_numpy_blas | `[512, 512, 512]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 1.372000 | 1.371280 | 1.291300 | 0.046576 | — | — | BENCHMARKED_CPU_ONLY |
| gemm | gemm_blocked_cpu | `[512, 512, 512]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 26.877800 | 33.998500 | 21.732500 | 12.713224 | gemm_numpy_blas | 0.051x | BENCHMARKED_CPU_ONLY |

## Environment

- OS: Windows 11
- CPU: Intel64 Family 6 Model 170 Stepping 4, GenuineIntel
- Hardware mode: `cpu_only`

CUDA benchmark status: `NOT BENCHMARKED ON CURRENT HARDWARE`.
The numbers above are CPU reference measurements and must not be interpreted as GPU results.
