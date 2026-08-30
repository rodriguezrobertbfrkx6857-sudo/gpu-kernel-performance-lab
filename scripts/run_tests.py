#!/usr/bin/env python3
"""Run CPU reference correctness tests available without CUDA."""

from __future__ import annotations

import math

import numpy as np

from run_all_benchmarks import _blocked_gemm, _transpose_tiled, _transpose_tiled_padded


def main() -> None:
    rng = np.random.default_rng(20260830)
    a = rng.normal(size=100003).astype(np.float32)
    b = rng.normal(size=100003).astype(np.float32)
    np.testing.assert_allclose(np.add(a, b), a + b, rtol=0.0, atol=0.0)
    print("PASS cpu vector_add reference")

    matrix = rng.normal(size=(65, 97)).astype(np.float32)
    expected = matrix.T.copy()
    np.testing.assert_array_equal(_transpose_tiled(matrix), expected)
    np.testing.assert_array_equal(_transpose_tiled_padded(matrix), expected)
    print("PASS cpu transpose tiled and padded references")

    values = rng.normal(size=1 << 20).astype(np.float32)
    actual = float(np.add.reduce(values, dtype=np.float64))
    expected_sum = float(np.sum(values, dtype=np.float64))
    if not math.isclose(actual, expected_sum, rel_tol=1.0e-12, abs_tol=1.0e-8):
        raise AssertionError("reduction reference mismatch")
    print("PASS cpu reduction reference")

    left = rng.normal(size=(37, 23)).astype(np.float32)
    right = rng.normal(size=(23, 29)).astype(np.float32)
    np.testing.assert_allclose(_blocked_gemm(left, right), left @ right, rtol=3.0e-5, atol=3.0e-5)
    print("PASS cpu gemm blocked reference")


if __name__ == "__main__":
    main()

