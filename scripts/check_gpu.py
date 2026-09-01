#!/usr/bin/env python
"""Prints a quick GPU/CUDA/PyTorch environment report - what SLURM/Jupyter
users on this project should run right after `pip install` to confirm
`torch.cuda.is_available()` actually sees the cluster's GPUs, before
launching scripts/run_experiment.py. See docs/slurm_gpu_setup.md for the
recommended install workflow this is meant to validate.

Deliberately read-only: never installs or changes anything, only reports
what is already present - safe to run repeatedly, on any machine (CPU-only
included), with or without torch installed at all.

Usage:
    python scripts/check_gpu.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def _report_python() -> None:
    _print_header("Python")
    print(f"executable: {sys.executable}")
    print(f"version:    {sys.version.split()[0]}")


def _report_env() -> None:
    _print_header("Environment")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")


def _nvidia_smi_driver_cuda() -> str | None:
    """Parses "CUDA Version: X.Y" out of `nvidia-smi`'s own header line -
    the MAXIMUM CUDA toolkit version this machine's driver supports, NOT
    necessarily the CUDA version anything is actually built against. None
    if `nvidia-smi` isn't on PATH or its output doesn't match."""
    try:
        output = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10, check=False
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"CUDA Version:\s*([\d.]+)", output)
    return match.group(1) if match else None


def _report_nvidia_smi() -> None:
    _print_header("nvidia-smi")
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10, check=False)
    except FileNotFoundError:
        print("nvidia-smi not found on PATH - no NVIDIA driver visible from this shell.")
        return
    except subprocess.TimeoutExpired:
        print("nvidia-smi timed out.")
        return
    if result.returncode != 0:
        print(f"nvidia-smi exited with code {result.returncode}:\n{result.stderr.strip()}")
        return
    print(result.stdout.strip())


def _report_torch() -> None:
    _print_header("PyTorch")
    try:
        import torch
    except ImportError as exc:
        print(f"torch is not importable: {exc}")
        print("Install a CUDA-matched build first - see docs/slurm_gpu_setup.md.")
        return

    torch_cuda_build = getattr(torch.version, "cuda", None)
    print(f"torch.__version__:       {torch.__version__}")
    print(f"torch.version.cuda:      {torch_cuda_build or '<CPU-only build>'}")
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print(f"GPU count: {count}")
        for i in range(count):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
        return

    # CUDA unavailable - try to pin down WHY, specifically the case this
    # script exists for: a driver that supports an older CUDA toolkit than
    # the one this torch build was compiled against (torch built for
    # CUDA 13.0, driver only supports up to CUDA 12.8 -> silently
    # unavailable, no exception raised).
    driver_cuda = _nvidia_smi_driver_cuda()
    if driver_cuda is None:
        print(
            "No NVIDIA driver detected via nvidia-smi (or nvidia-smi is unavailable here) - "
            "either this machine truly has no GPU, or nvidia-smi isn't on PATH in this shell/container."
        )
        return

    if torch_cuda_build is None:
        print(f"Driver supports CUDA {driver_cuda}, but this is a CPU-only torch build (no CUDA support compiled in at all).")
        return

    try:
        driver_major_minor = tuple(int(p) for p in driver_cuda.split(".")[:2])
        torch_major_minor = tuple(int(p) for p in torch_cuda_build.split(".")[:2])
    except ValueError:
        driver_major_minor = torch_major_minor = None

    if driver_major_minor is not None and torch_major_minor is not None and torch_major_minor > driver_major_minor:
        print(
            f"Driver supports CUDA {driver_cuda}, but PyTorch was compiled for CUDA {torch_cuda_build} - "
            "this torch build is too new for this driver, so torch.cuda.is_available() is silently False "
            "(no exception). Install a torch build matching the driver's CUDA version instead - "
            "see docs/slurm_gpu_setup.md."
        )
    else:
        print(
            f"Driver supports CUDA {driver_cuda}, torch was compiled for CUDA {torch_cuda_build}, yet "
            "torch.cuda.is_available() is False for some other reason (no GPU visible to this process, "
            "CUDA_VISIBLE_DEVICES masking it out, a driver/runtime mismatch, etc.) - see docs/slurm_gpu_setup.md."
        )


def main() -> None:
    _report_python()
    _report_env()
    _report_torch()
    _report_nvidia_smi()


if __name__ == "__main__":
    main()
