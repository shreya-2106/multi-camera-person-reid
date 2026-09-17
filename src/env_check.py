"""
Environment checks + CPU/CUDA device detection.

Phase 1 keeps this dependency-light: torch is optional. If torch is not
installed yet, device detection gracefully falls back to CPU and reports
that torch is unavailable, rather than crashing. This lets `main.py` run
end-to-end before heavier ML dependencies (Phase 2+) are installed.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import List

MIN_PYTHON_VERSION = (3, 9)


@dataclass
class DeviceInfo:
    """Result of CPU/CUDA device detection."""

    torch_available: bool
    device: str  # "cuda" or "cpu"
    cuda_available: bool = False
    cuda_device_count: int = 0
    cuda_device_names: List[str] = field(default_factory=list)
    torch_version: str = ""

    def summary(self) -> str:
        if not self.torch_available:
            return "torch not installed -> using CPU (device detection limited)"
        if self.cuda_available:
            names = ", ".join(self.cuda_device_names) or "unknown GPU"
            return (
                f"torch {self.torch_version} | CUDA available "
                f"({self.cuda_device_count} device(s): {names}) -> using '{self.device}'"
            )
        return f"torch {self.torch_version} | CUDA not available -> using '{self.device}'"


@dataclass
class EnvironmentReport:
    """Result of environment sanity checks."""

    python_version: str
    python_ok: bool
    platform_name: str
    missing_packages: List[str] = field(default_factory=list)
    device_info: DeviceInfo = None  # type: ignore[assignment]

    @property
    def ok(self) -> bool:
        return self.python_ok and not self.missing_packages


def check_python_version(min_version=MIN_PYTHON_VERSION) -> bool:
    """Check that the running interpreter meets the minimum version."""
    return sys.version_info[:2] >= min_version


def detect_device(prefer_cuda: bool = True) -> DeviceInfo:
    """
    Detect whether CUDA is available via torch. Falls back cleanly to CPU
    if torch isn't installed (expected during Phase 1, before ML deps land).
    """
    try:
        import torch  # type: ignore
    except ImportError:
        return DeviceInfo(torch_available=False, device="cpu", cuda_available=False)

    cuda_available = bool(torch.cuda.is_available())
    device = "cuda" if (cuda_available and prefer_cuda) else "cpu"
    device_count = torch.cuda.device_count() if cuda_available else 0
    device_names = (
        [torch.cuda.get_device_name(i) for i in range(device_count)]
        if cuda_available
        else []
    )

    return DeviceInfo(
        torch_available=True,
        device=device,
        cuda_available=cuda_available,
        cuda_device_count=device_count,
        cuda_device_names=device_names,
        torch_version=getattr(torch, "__version__", "unknown"),
    )


def check_required_packages(package_names: List[str]) -> List[str]:
    """
    Given a list of pip/import-style package names, return the subset that
    cannot currently be imported. Uses importlib to avoid `pip` subprocess calls.
    """
    import importlib

    missing = []
    for name in package_names:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    return missing


def run_environment_check(
    required_packages: List[str] = None,
) -> EnvironmentReport:
    """
    Run a full environment check: Python version, platform, core importable
    packages, and CPU/CUDA device detection.
    """
    required_packages = required_packages if required_packages is not None else ["yaml"]

    python_ok = check_python_version()
    missing = check_required_packages(required_packages)
    device_info = detect_device()

    return EnvironmentReport(
        python_version=platform.python_version(),
        python_ok=python_ok,
        platform_name=f"{platform.system()} {platform.release()} ({platform.machine()})",
        missing_packages=missing,
        device_info=device_info,
    )
