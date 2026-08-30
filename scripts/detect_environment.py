#!/usr/bin/env python3
"""Collect a sanitized, machine-readable environment snapshot."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run(command: list[str], timeout: int = 8) -> tuple[int, str]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, str(exc)
    return completed.returncode, (completed.stdout or completed.stderr).strip()


def _version(command: str, args: list[str] | None = None) -> str | None:
    path = shutil.which(command)
    if path is None:
        return None
    code, output = _run([path, *(args or ["--version"])])
    if code != 0 and not output:
        return None
    first_line = output.splitlines()[0] if output else "available"
    return first_line.replace(str(Path(sys.prefix)), "<python-env>")


def _numpy_version() -> str | None:
    try:
        import numpy

        return numpy.__version__
    except ImportError:
        return None


def _ram_bytes() -> int | None:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong),
                ("available_phys", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong),
                ("available_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_phys)
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def _display_adapters() -> list[dict[str, Any]]:
    if os.name != "nt":
        return []
    command = [
        "powershell",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json -Compress",
    ]
    code, output = _run(command)
    if code != 0 or not output:
        return []
    try:
        value = json.loads(output)
    except json.JSONDecodeError:
        return []
    items = value if isinstance(value, list) else [value]
    return [
        {
            "name": item.get("Name"),
            "adapter_ram_bytes": item.get("AdapterRAM"),
            "driver_version": item.get("DriverVersion"),
        }
        for item in items
        if isinstance(item, dict)
    ]


def _nvidia() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    result: dict[str, Any] = {"available": False, "executable": bool(executable), "devices": []}
    if executable is None:
        result["reason"] = "nvidia-smi was not found"
        return result
    code, output = _run(
        [
            executable,
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if code != 0:
        result["reason"] = output or "nvidia-smi returned a non-zero status"
        return result
    devices = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4:
            devices.append(
                {
                    "name": fields[0],
                    "driver_version": fields[1],
                    "memory_total_mib": fields[2],
                    "compute_capability": fields[3],
                }
            )
    result["available"] = bool(devices)
    result["devices"] = devices
    if not devices:
        result["reason"] = "nvidia-smi returned no CUDA-capable devices"
    return result


def _torch_info() -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {"installed": False}
    try:
        import torch

        return {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "compiled_cuda": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "capabilities": [list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())],
        }
    except Exception as exc:  # pragma: no cover - depends on an external installation
        return {"installed": True, "import_error": repr(exc)}


def collect_environment() -> dict[str, Any]:
    nvidia = _nvidia()
    torch_info = _torch_info()
    cuda_available = bool(nvidia["available"] and shutil.which("nvcc"))
    return {
        "schema_version": 1,
        "captured_by": "scripts/detect_environment.py",
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "cpu": {"name": platform.processor() or platform.machine(), "logical_processors": os.cpu_count()},
        "ram_bytes": _ram_bytes(),
        "display_adapters": _display_adapters(),
        "nvidia": nvidia,
        "cuda_workflow_available": cuda_available,
        "toolchain": {
            "nvcc": _version("nvcc"),
            "cmake": _version("cmake"),
            "cxx": _version("cl") or _version("g++") or _version("gcc"),
            "python": platform.python_version(),
            "pip": _version("pip"),
            "git": _version("git"),
            "github_cli": _version("gh"),
            "nsight_compute": _version("ncu"),
            "nsight_systems": _version("nsys"),
        },
        "github_cli_authenticated": bool(shutil.which("gh") and _run([shutil.which("gh") or "gh", "auth", "status"])[0] == 0),
        "python_packages": {"torch": torch_info, "numpy": _numpy_version()},
        "hardware_mode": "cuda" if cuda_available else "cpu_only",
    }


def render_markdown(environment: dict[str, Any]) -> str:
    operating_system = environment["operating_system"]
    lines = [
        "# Environment Audit",
        "",
        "This file is generated by `scripts/detect_environment.py`.",
        "",
        f"- Mode: `{environment['hardware_mode']}`",
        f"- OS: {operating_system['system']} {operating_system['release']} ({operating_system['machine']})",
        f"- CPU: {environment['cpu']['name']}",
        f"- Logical processors: {environment['cpu']['logical_processors']}",
        f"- RAM bytes: {environment['ram_bytes']}",
        "",
        "## Display adapters",
        "",
    ]
    adapters = environment.get("display_adapters", [])
    if adapters:
        lines.extend(f"- {item['name']} (driver {item['driver_version']})" for item in adapters)
    else:
        lines.append("- No display adapter information was returned by the platform query.")
    lines.extend(["", "## CUDA detection", ""])
    nvidia = environment["nvidia"]
    if nvidia["available"]:
        lines.extend(
            f"- {device['name']}, driver {device['driver_version']}, "
            f"{device['memory_total_mib']} MiB, compute capability {device['compute_capability']}"
            for device in nvidia["devices"]
        )
    else:
        lines.append(f"- No CUDA-capable NVIDIA GPU detected: {nvidia.get('reason', 'unknown reason')}.")
    lines.extend(["", "## Toolchain", ""])
    for name, value in environment["toolchain"].items():
        lines.append(f"- {name}: `{value if value is not None else 'not found'}`")
    torch_info = environment["python_packages"]["torch"]
    lines.extend(["", "## PyTorch", ""])
    if isinstance(torch_info, dict) and torch_info.get("installed"):
        lines.append(f"- Version: `{torch_info.get('version', 'unknown')}`")
        lines.append(f"- `torch.cuda.is_available()`: `{torch_info.get('cuda_available')}`")
        lines.append(f"- Compiled CUDA: `{torch_info.get('compiled_cuda')}`")
    else:
        lines.append("- PyTorch is not installed in the interpreter used for this audit.")
    lines.append("")
    return "\n".join(lines)


def write_environment(output_dir: Path) -> dict[str, Any]:
    environment = collect_environment()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "environment.md").write_text(render_markdown(environment), encoding="utf-8")
    return environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    environment = write_environment(args.output_dir)
    print(json.dumps(environment, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
