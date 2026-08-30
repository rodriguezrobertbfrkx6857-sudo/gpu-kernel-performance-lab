# Benchmark Results

This report is generated from `benchmark_results.json`; no timing value is maintained by hand.

- Hardware mode: `cpu_only`
- Timing protocol: CUDA Event for CUDA binaries; perf_counter_ns for CPU fallback
- CUDA defaults: 20 warm-up launches and 100 measured launches when a CUDA binary is available.
- CPU fallback: adaptive warm-up and iteration counts are recorded in the JSON protocol.
- Speedup rule: declared baseline median divided by candidate median within the same family, shape, and dtype

| Family | Variant | Shape | Backend | Device | Median ms | Mean ms | Min ms | Std ms | Baseline | Speedup | Status |
|---|---|---|---|---|---:|---:|---:|---:|---|---:|---|
| vector_add | vector_add_cpu_numpy | `[1048576]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 1.876350 | 1.840060 | 1.451400 | 0.190704 | — | — | BENCHMARKED_CPU_ONLY |
| vector_add | vector_add_cpu_numpy | `[16777216]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 27.379700 | 25.783510 | 18.995600 | 3.971900 | — | — | BENCHMARKED_CPU_ONLY |
| transpose | copy_baseline | `[1024, 1024]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 1.523400 | 1.586990 | 1.095600 | 0.244187 | — | — | BENCHMARKED_CPU_ONLY |
| transpose | transpose_naive | `[1024, 1024]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 4.022750 | 4.264430 | 3.761700 | 0.505123 | — | — | BENCHMARKED_CPU_ONLY |
| transpose | transpose_tiled | `[1024, 1024]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 3.471100 | 3.529090 | 2.203000 | 0.794988 | transpose_naive | 1.159x | BENCHMARKED_CPU_ONLY |
| transpose | transpose_tiled_padded | `[1024, 1024]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 8.651700 | 9.025170 | 6.016400 | 1.789837 | transpose_naive | 0.465x | BENCHMARKED_CPU_ONLY |
| transpose | copy_baseline | `[2048, 2048]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 5.083050 | 5.058840 | 3.984400 | 0.911197 | — | — | BENCHMARKED_CPU_ONLY |
| transpose | transpose_naive | `[2048, 2048]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 56.085200 | 60.162990 | 50.381900 | 9.343443 | — | — | BENCHMARKED_CPU_ONLY |
| transpose | transpose_tiled | `[2048, 2048]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 21.781600 | 21.725330 | 13.754800 | 4.610959 | transpose_naive | 2.575x | BENCHMARKED_CPU_ONLY |
| transpose | transpose_tiled_padded | `[2048, 2048]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 39.978500 | 41.234720 | 29.418200 | 8.920866 | transpose_naive | 1.403x | BENCHMARKED_CPU_ONLY |
| reduction | reduction_numpy_pairwise | `[1048576]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 0.716200 | 0.733030 | 0.474600 | 0.249153 | — | — | BENCHMARKED_CPU_ONLY |
| reduction | reduction_numpy_sum | `[1048576]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 0.769850 | 0.759870 | 0.490400 | 0.105155 | reduction_numpy_pairwise | 0.930x | BENCHMARKED_CPU_ONLY |
| reduction | reduction_numpy_pairwise | `[16777216]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 18.419500 | 18.553910 | 15.032900 | 3.202298 | — | — | BENCHMARKED_CPU_ONLY |
| reduction | reduction_numpy_sum | `[16777216]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 16.907900 | 16.485370 | 11.908700 | 3.406998 | reduction_numpy_pairwise | 1.089x | BENCHMARKED_CPU_ONLY |
| gemm | gemm_numpy_blas | `[256, 256, 256]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 0.695800 | 0.701840 | 0.578300 | 0.091879 | — | — | BENCHMARKED_CPU_ONLY |
| gemm | gemm_blocked_cpu | `[256, 256, 256]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 6.647700 | 6.734320 | 6.123800 | 0.439371 | gemm_numpy_blas | 0.105x | BENCHMARKED_CPU_ONLY |
| gemm | gemm_numpy_blas | `[512, 512, 512]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 1.344800 | 1.454100 | 1.235400 | 0.210200 | — | — | BENCHMARKED_CPU_ONLY |
| gemm | gemm_blocked_cpu | `[512, 512, 512]` | numpy | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel | 42.405100 | 46.223380 | 34.439000 | 11.290954 | gemm_numpy_blas | 0.032x | BENCHMARKED_CPU_ONLY |

## Environment

- OS: Windows 11
- CPU: Intel64 Family 6 Model 170 Stepping 4, GenuineIntel
- Hardware mode: `cpu_only`

CUDA benchmark status: `NOT BENCHMARKED ON CURRENT HARDWARE`.
The numbers above are CPU reference measurements and must not be interpreted as GPU results.
