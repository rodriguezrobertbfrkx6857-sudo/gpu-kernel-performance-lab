#!/usr/bin/env python3
"""Run reproducible CPU reference benchmarks when CUDA hardware is absent."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

from detect_environment import write_environment


def _summary(samples: list[float]) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "median_ms": float(np.median(values)),
        "mean_ms": float(np.mean(values)),
        "min_ms": float(np.min(values)),
        "std_ms": float(np.std(values)),
    }


def _measure(function: Callable[[], object], warmup: int, iterations: int) -> tuple[dict[str, float], object]:
    result: object = None
    for _ in range(warmup):
        result = function()
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        result = function()
        samples.append((time.perf_counter_ns() - start) / 1.0e6)
    return _summary(samples), result


def _record(
    records: list[dict],
    *,
    family: str,
    variant: str,
    shape: list[int],
    summary: dict[str, float],
    bytes_moved: int = 0,
    baseline_median: float | None = None,
    notes: str = "CPU reference path",
) -> None:
    record = {
        "family": family,
        "kernel": variant,
        "variant": variant,
        "shape": shape,
        **summary,
        "speedup": (baseline_median / summary["median_ms"] if baseline_median else None),
        "effective_bandwidth_gbps": (
            bytes_moved / (summary["median_ms"] * 1.0e6) if bytes_moved else None
        ),
        "status": "BENCHMARKED_CPU_ONLY",
        "notes": notes,
    }
    records.append(record)


def _transpose_tiled(matrix: np.ndarray, tile: int = 32) -> np.ndarray:
    rows, cols = matrix.shape
    output = np.empty((cols, rows), dtype=matrix.dtype)
    for row in range(0, rows, tile):
        for col in range(0, cols, tile):
            output[col : col + tile, row : row + tile] = matrix[row : row + tile, col : col + tile].T
    return output


def _transpose_tiled_padded(matrix: np.ndarray, tile: int = 32) -> np.ndarray:
    rows, cols = matrix.shape
    output = np.empty((cols, rows), dtype=matrix.dtype)
    for row in range(0, rows, tile):
        for col in range(0, cols, tile):
            block = matrix[row : row + tile, col : col + tile]
            padded = np.empty((tile, tile + 1), dtype=matrix.dtype)
            padded.fill(0)
            padded[: block.shape[0], : block.shape[1]] = block
            output[col : col + block.shape[1], row : row + block.shape[0]] = padded[: block.shape[0], : block.shape[1]].T
    return output


def _cpu_benchmarks(quick: bool = False) -> list[dict]:
    records: list[dict] = []
    warmup = 2 if quick else 3
    iterations = 5 if quick else 10

    for n in ([1 << 20] if quick else [1 << 20, 1 << 22]):
        a = np.linspace(0.0, 1.0, n, dtype=np.float32)
        b = np.linspace(1.0, 0.0, n, dtype=np.float32)
        summary, result = _measure(lambda: np.add(a, b), warmup, iterations)
        if not np.allclose(result, 1.0, rtol=1.0e-6, atol=1.0e-6):
            raise RuntimeError("vector-add correctness check failed")
        _record(records, family="vector_add", variant="vector_add_cpu_numpy", shape=[n], summary=summary,
                bytes_moved=3 * a.nbytes)

    for side in ([512] if quick else [512, 1024]):
        matrix = np.arange(side * side, dtype=np.float32).reshape(side, side)
        expected = matrix.T.copy()
        variants = [
            ("copy_baseline", lambda: np.array(matrix, copy=True)),
            ("transpose_naive", lambda: matrix.T.copy()),
            ("transpose_tiled", lambda: _transpose_tiled(matrix)),
            ("transpose_tiled_padded", lambda: _transpose_tiled_padded(matrix)),
        ]
        baseline_median: float | None = None
        for name, function in variants:
            summary, result = _measure(function, warmup, iterations)
            target = matrix if name == "copy_baseline" else expected
            if not np.array_equal(result, target):
                raise RuntimeError(f"transpose correctness check failed for {name}")
            _record(
                records,
                family="transpose",
                variant=name,
                shape=[side, side],
                summary=summary,
                bytes_moved=2 * matrix.nbytes,
                baseline_median=baseline_median,
                notes="CPU analogue of the CUDA copy/coalescing variants",
            )
            if baseline_median is None:
                baseline_median = summary["median_ms"]

    for exponent in ([20] if quick else [20, 22, 24]):
        n = 1 << exponent
        values = np.sin(np.arange(n, dtype=np.float32) * np.float32(0.001))
        expected = float(np.sum(values, dtype=np.float64))
        variants = [
            ("reduction_numpy_pairwise", lambda: np.add.reduce(values, dtype=np.float64)),
            ("reduction_numpy_sum", lambda: np.sum(values, dtype=np.float64)),
        ]
        baseline_median: float | None = None
        for name, function in variants:
            summary, result = _measure(function, warmup, iterations)
            if not math.isclose(float(result), expected, rel_tol=1.0e-12, abs_tol=1.0e-8):
                raise RuntimeError(f"reduction correctness check failed for {name}")
            _record(
                records,
                family="reduction",
                variant=name,
                shape=[n],
                summary=summary,
                bytes_moved=values.nbytes,
                baseline_median=baseline_median,
                notes="CPU reference; CUDA shared-memory and warp-shuffle sources are in kernels/",
            )
            if baseline_median is None:
                baseline_median = summary["median_ms"]

    for side in ([128] if quick else [128, 256]):
        a = np.linspace(-0.5, 0.5, side * side, dtype=np.float32).reshape(side, side)
        b = np.linspace(0.25, -0.25, side * side, dtype=np.float32).reshape(side, side)
        expected = a @ b
        variants = [
            ("gemm_numpy_blas", lambda: a @ b),
            ("gemm_blocked_cpu", lambda: _blocked_gemm(a, b)),
        ]
        baseline_median: float | None = None
        for name, function in variants:
            summary, result = _measure(function, 1 if quick else 2, 3 if quick else 5)
            if not np.allclose(result, expected, rtol=3.0e-5, atol=3.0e-5):
                raise RuntimeError(f"GEMM correctness check failed for {name}")
            _record(
                records,
                family="gemm",
                variant=name,
                shape=[side, side, side],
                summary=summary,
                baseline_median=baseline_median,
                notes="CPU reference; CUDA naive/tiled/cuBLAS comparison is in the CUDA build path",
            )
            if baseline_median is None:
                baseline_median = summary["median_ms"]
    return records


def _blocked_gemm(a: np.ndarray, b: np.ndarray, tile: int = 32) -> np.ndarray:
    m, k = a.shape
    _, n = b.shape
    output = np.zeros((m, n), dtype=np.float32)
    for row in range(0, m, tile):
        for col in range(0, n, tile):
            for inner in range(0, k, tile):
                output[row : row + tile, col : col + tile] += (
                    a[row : row + tile, inner : inner + tile]
                    @ b[inner : inner + tile, col : col + tile]
                )
    return output


def _cuda_binary_results(root: Path) -> list[dict] | None:
    """Read JSON lines from a CUDA build if the caller built it explicitly."""
    binaries = [
        root / "build" / "bench_vector_add.exe",
        root / "build" / "bench_transpose.exe",
        root / "build" / "bench_reduction.exe",
        root / "build" / "bench_gemm.exe",
    ]
    if not all(path.exists() for path in binaries):
        return None
    records: list[dict] = []
    for binary in binaries:
        completed = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
        family = binary.stem.removeprefix("bench_")
        for line in completed.stdout.splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "median_ms" in item:
                item.setdefault("family", family)
                item.setdefault("variant", item.get("kernel", family))
                item.setdefault("shape", [item.get("elements", 0)])
                item["status"] = "BENCHMARKED_CUDA"
                item["speedup"] = None
                records.append(item)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use smaller CPU workloads for a smoke run")
    parser.add_argument("--force-cpu", action="store_true", help="Run CPU references even when CUDA is detected")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    environment = write_environment(root / "results")
    records = None if args.force_cpu else _cuda_binary_results(root) if environment["cuda_workflow_available"] else None
    mode = "cuda" if records is not None else "cpu_only"
    if records is None:
        records = _cpu_benchmarks(quick=args.quick)
    payload = {
        "schema_version": 1,
        "generated_by": "scripts/run_all_benchmarks.py",
        "hardware_mode": mode,
        "protocol": {
            "cuda_warmup": 20,
            "cuda_iterations": 100,
            "cpu_warmup": 2 if args.quick else 3,
            "cpu_iterations": 5 if args.quick else 10,
            "timer": "CUDA Event for CUDA binaries; perf_counter_ns for CPU fallback",
        },
        "benchmarks": records,
    }
    output = root / "results" / "benchmark_results.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
