#!/usr/bin/env python3
"""Run the CUDA benchmark binaries or an explicitly labelled CPU reference suite."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
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


def _measure(
    function: Callable[[], object], warmup: int, iterations: int
) -> tuple[dict[str, float], object]:
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
    device: str,
    backend: str,
    warmup: int,
    iterations: int,
    bytes_moved: int = 0,
    baseline_variant: str | None = None,
    notes: str,
) -> None:
    records.append(
        {
            "family": family,
            "variant": variant,
            "shape": shape,
            "dtype": "float32",
            "device": device,
            "backend": backend,
            "warmup": warmup,
            "iterations": iterations,
            **summary,
            "effective_bandwidth_gbps": (
                bytes_moved / (summary["median_ms"] * 1.0e6) if bytes_moved else None
            ),
            "baseline_variant": baseline_variant,
            "speedup": None,
            "status": "BENCHMARKED_CPU_ONLY",
            "notes": notes,
        }
    )


def _apply_speedups(records: list[dict]) -> None:
    """Compute only the explicitly declared within-shape baseline comparisons."""
    indexed: dict[tuple[str, tuple[int, ...], str, str], dict] = {}
    for item in records:
        if not isinstance(item.get("median_ms"), (int, float)):
            continue
        key = (
            str(item.get("family")),
            tuple(int(value) for value in item.get("shape", [])),
            str(item.get("dtype", "float32")),
            str(item.get("variant")),
        )
        indexed[key] = item

    for item in records:
        baseline_variant = item.get("baseline_variant")
        if not baseline_variant:
            item["speedup"] = None
            continue
        key = (
            str(item.get("family")),
            tuple(int(value) for value in item.get("shape", [])),
            str(item.get("dtype", "float32")),
            str(baseline_variant),
        )
        baseline = indexed.get(key)
        candidate_ms = item.get("median_ms")
        if baseline is None or not isinstance(candidate_ms, (int, float)) or candidate_ms <= 0:
            item["speedup"] = None
            continue
        item["speedup"] = float(baseline["median_ms"]) / float(candidate_ms)


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
            output[col : col + block.shape[1], row : row + block.shape[0]] = padded[
                : block.shape[0], : block.shape[1]
            ].T
    return output


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


def _cpu_benchmarks(environment: dict, quick: bool = False) -> list[dict]:
    records: list[dict] = []
    device = str(environment["cpu"].get("name") or "CPU reference")
    warmup = 2 if quick else 3
    iterations = 5 if quick else 10

    for n in ([1 << 20] if quick else [1 << 20, 1 << 24]):
        a = np.linspace(0.0, 1.0, n, dtype=np.float32)
        b = np.linspace(1.0, 0.0, n, dtype=np.float32)
        summary, result = _measure(lambda: np.add(a, b), warmup, iterations)
        if not np.allclose(result, 1.0, rtol=1.0e-6, atol=1.0e-6):
            raise RuntimeError("vector-add correctness check failed")
        _record(
            records,
            family="vector_add",
            variant="vector_add_cpu_numpy",
            shape=[n],
            summary=summary,
            device=device,
            backend="numpy",
            warmup=warmup,
            iterations=iterations,
            bytes_moved=3 * a.nbytes,
            notes="CPU reference path; CUDA results require a CUDA-capable NVIDIA host",
        )

    for side in ([512] if quick else [1024, 2048]):
        matrix = np.arange(side * side, dtype=np.float32).reshape(side, side)
        expected = matrix.T.copy()
        variants = [
            ("copy_baseline", lambda: np.array(matrix, copy=True), None),
            ("transpose_naive", lambda: matrix.T.copy(), None),
            ("transpose_tiled", lambda: _transpose_tiled(matrix), "transpose_naive"),
            ("transpose_tiled_padded", lambda: _transpose_tiled_padded(matrix), "transpose_naive"),
        ]
        for name, function, baseline_variant in variants:
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
                device=device,
                backend="numpy",
                warmup=warmup,
                iterations=iterations,
                bytes_moved=2 * matrix.nbytes,
                baseline_variant=baseline_variant,
                notes=(
                    "CPU analogue only; copy is a bandwidth reference and transpose speedups "
                    "are relative to transpose_naive"
                ),
            )

    for exponent in ([20] if quick else [20, 24]):
        n = 1 << exponent
        values = np.sin(np.arange(n, dtype=np.float32) * np.float32(0.001))
        expected = float(np.sum(values, dtype=np.float64))
        variants = [
            ("reduction_numpy_pairwise", lambda: np.add.reduce(values, dtype=np.float64), None),
            ("reduction_numpy_sum", lambda: np.sum(values, dtype=np.float64), "reduction_numpy_pairwise"),
        ]
        for name, function, baseline_variant in variants:
            summary, result = _measure(function, warmup, iterations)
            if not math.isclose(float(result), expected, rel_tol=1.0e-12, abs_tol=1.0e-8):
                raise RuntimeError(f"reduction correctness check failed for {name}")
            _record(
                records,
                family="reduction",
                variant=name,
                shape=[n],
                summary=summary,
                device=device,
                backend="numpy",
                warmup=warmup,
                iterations=iterations,
                bytes_moved=values.nbytes,
                baseline_variant=baseline_variant,
                notes="CPU reference only; CUDA shared-memory and warp-shuffle sources are in kernels/",
            )

    for side in ([128] if quick else [256, 512]):
        a = np.linspace(-0.5, 0.5, side * side, dtype=np.float32).reshape(side, side)
        b = np.linspace(0.25, -0.25, side * side, dtype=np.float32).reshape(side, side)
        expected = a @ b
        variants = [
            ("gemm_numpy_blas", lambda: a @ b, None),
            ("gemm_blocked_cpu", lambda: _blocked_gemm(a, b), "gemm_numpy_blas"),
        ]
        for name, function, baseline_variant in variants:
            summary, result = _measure(function, 1 if quick else 2, 3 if quick else 5)
            if not np.allclose(result, expected, rtol=3.0e-5, atol=3.0e-5):
                raise RuntimeError(f"GEMM correctness check failed for {name}")
            _record(
                records,
                family="gemm",
                variant=name,
                shape=[side, side, side],
                summary=summary,
                device=device,
                backend="numpy",
                warmup=1 if quick else 2,
                iterations=3 if quick else 5,
                baseline_variant=baseline_variant,
                notes="CPU reference only; CUDA naive/tiled/cuBLAS comparison is in the CUDA build path",
            )
    return records


def _find_binary(build_dir: Path, stem: str) -> Path | None:
    candidates = [
        build_dir / stem,
        build_dir / f"{stem}.exe",
        build_dir / "Release" / stem,
        build_dir / "Release" / f"{stem}.exe",
        build_dir / "Debug" / stem,
        build_dir / "Debug" / f"{stem}.exe",
        build_dir / "bin" / stem,
        build_dir / "bin" / f"{stem}.exe",
        build_dir / "bin" / "Release" / f"{stem}.exe",
    ]
    return next((path for path in candidates if path.is_file()), None)


def _cuda_binary_results(root: Path, build_dir: Path) -> list[dict]:
    """Run every CUDA benchmark and parse only records emitted by those binaries."""
    stems = ["bench_vector_add", "bench_transpose", "bench_reduction", "bench_gemm"]
    binaries = {stem: _find_binary(build_dir, stem) for stem in stems}
    missing = [stem for stem, path in binaries.items() if path is None]
    if missing:
        raise RuntimeError(
            "CUDA environment is available but benchmark binaries are missing: "
            + ", ".join(missing)
            + ". Build the Release configuration before running benchmarks."
        )

    records: list[dict] = []
    for stem, binary in binaries.items():
        assert binary is not None
        completed = subprocess.run(
            [str(binary)], capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"{binary} failed with exit code {completed.returncode}: {details}")
        family = stem.removeprefix("bench_")
        for line in completed.stdout.splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            item.setdefault("family", family)
            item.setdefault("backend", "cuda")
            if "median_ms" in item:
                item.setdefault("variant", item.get("kernel", family))
                item.setdefault("dtype", "float32")
                item.setdefault("status", "BENCHMARKED_CUDA")
                item.setdefault("baseline_variant", None)
                item.setdefault("speedup", None)
                records.append(item)
            elif item.get("status") == "SKIPPED":
                item.setdefault("variant", "all")
                records.append(item)
    if not records:
        raise RuntimeError("CUDA benchmark binaries emitted no machine-readable records")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use smaller CPU workloads for a smoke run")
    parser.add_argument("--force-cpu", action="store_true", help="Run CPU references even when CUDA is detected")
    parser.add_argument("--build-dir", type=Path, default=Path("build"), help="CUDA build directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    environment = write_environment(root / "results")

    if args.force_cpu or not environment["cuda_workflow_available"]:
        mode = "cpu_only"
        records = _cpu_benchmarks(environment, quick=args.quick)
    else:
        mode = "cuda"
        build_dir = args.build_dir if args.build_dir.is_absolute() else root / args.build_dir
        records = _cuda_binary_results(root, build_dir)
    _apply_speedups(records)

    payload = {
        "schema_version": 2,
        "generated_by": "scripts/run_all_benchmarks.py",
        "hardware_mode": mode,
        "protocol": {
            "cuda_warmup": 20,
            "cuda_iterations": 100,
            "cpu_warmup": 2 if args.quick else 3,
            "cpu_iterations": 5 if args.quick else 10,
            "timer": "CUDA Event for CUDA binaries; perf_counter_ns for CPU fallback",
            "speedup_rule": "declared baseline median divided by candidate median within the same family, shape, and dtype",
        },
        "benchmarks": records,
    }
    output = root / "results" / "benchmark_results.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
