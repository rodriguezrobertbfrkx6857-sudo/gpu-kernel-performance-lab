#!/usr/bin/env python3
"""Render the benchmark JSON into a concise, generated Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(payload: dict, environment: dict) -> str:
    lines = [
        "# Benchmark Results",
        "",
        "This report is generated from `benchmark_results.json`; no timing value is maintained by hand.",
        "",
        f"- Hardware mode: `{payload['hardware_mode']}`",
        f"- Timing protocol: {payload['protocol']['timer']}",
        "- CUDA defaults: 20 warm-up launches and 100 measured launches when a CUDA binary is available.",
        "- CPU fallback: adaptive warm-up and iteration counts are recorded in the JSON protocol.",
        f"- Speedup rule: {payload['protocol']['speedup_rule']}",
        "",
        "| Family | Variant | Shape | Backend | Device | Median ms | Mean ms | Min ms | Std ms | Baseline | Speedup | Status |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for item in payload["benchmarks"]:
        speedup = "—" if item.get("speedup") is None else f"{item['speedup']:.3f}x"
        baseline = item.get("baseline_variant") or "—"
        if "median_ms" in item:
            median = f"{item['median_ms']:.6f}"
            mean = f"{item['mean_ms']:.6f}"
            minimum = f"{item['min_ms']:.6f}"
            std = f"{item['std_ms']:.6f}"
        else:
            median = mean = minimum = std = "—"
        lines.append(
            f"| {item['family']} | {item['variant']} | `{item['shape']}` | "
            f"{item.get('backend', 'unknown')} | {item.get('device', 'unknown')} | "
            f"{median} | {mean} | {minimum} | {std} | {baseline} | {speedup} | "
            f"{item['status']} |"
        )
    lines.extend([
        "",
        "## Environment",
        "",
        f"- OS: {environment['operating_system']['system']} {environment['operating_system']['release']}",
        f"- CPU: {environment['cpu']['name']}",
        f"- Hardware mode: `{environment['hardware_mode']}`",
    ])
    if environment["hardware_mode"] != "cuda":
        lines.extend([
            "",
            "CUDA benchmark status: `NOT BENCHMARKED ON CURRENT HARDWARE`.",
            "The numbers above are CPU reference measurements and must not be interpreted as GPU results.",
        ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    payload = json.loads((args.results_dir / "benchmark_results.json").read_text(encoding="utf-8"))
    environment = json.loads((args.results_dir / "environment.json").read_text(encoding="utf-8"))
    report = render(payload, environment)
    (args.results_dir / "results.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
