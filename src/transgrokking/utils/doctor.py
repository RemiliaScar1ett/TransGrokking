"""Environment and target-hardware diagnostics."""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class DoctorReport:
    prefix: str
    expected_prefix: str
    prefix_ok: bool
    python_version: str
    torch_version: str
    torch_cuda_runtime: str | None
    cuda_available: bool
    driver_version: str | None
    device_name: str | None
    total_vram_bytes: int | None
    compute_capability: str | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible diagnostic mapping."""
        return asdict(self)


def _driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else None


def collect_doctor_report(repo_root: str | Path | None = None) -> DoctorReport:
    """Inspect the current prefix and first CUDA device without mutating state."""
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    expected_prefix = (root / "env").resolve()
    actual_prefix = Path(sys.prefix).resolve()
    available = torch.cuda.is_available()
    properties = torch.cuda.get_device_properties(0) if available else None
    return DoctorReport(
        prefix=str(actual_prefix),
        expected_prefix=str(expected_prefix),
        prefix_ok=actual_prefix == expected_prefix,
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        torch_cuda_runtime=torch.version.cuda,
        cuda_available=available,
        driver_version=_driver_version() if available else None,
        device_name=torch.cuda.get_device_name(0) if available else None,
        total_vram_bytes=properties.total_memory if properties is not None else None,
        compute_capability=(
            f"{properties.major}.{properties.minor}" if properties is not None else None
        ),
    )


def _normalized_device(name: str) -> str:
    return " ".join(name.lower().replace("nvidia", "").split())


def validate_doctor_report(
    report: DoctorReport,
    require_cuda: bool = False,
    expected_device: str | None = None,
    expected_vram_gb: float | None = None,
) -> list[str]:
    """Return contract violations for a diagnostic report."""
    errors: list[str] = []
    if not report.prefix_ok:
        errors.append(f"interpreter prefix is {report.prefix}, expected {report.expected_prefix}")
    if require_cuda and not report.cuda_available:
        errors.append("CUDA is required but unavailable")
    if expected_device and report.cuda_available:
        if _normalized_device(report.device_name or "") != _normalized_device(expected_device):
            errors.append(f"GPU is {report.device_name!r}, expected {expected_device!r}")
    if expected_vram_gb is not None and report.cuda_available:
        required = expected_vram_gb * 1_000_000_000 * 0.99
        if report.total_vram_bytes is None or report.total_vram_bytes < required:
            actual = (report.total_vram_bytes or 0) / 1_000_000_000
            errors.append(
                f"GPU VRAM is {actual:.2f} GB, expected approximately {expected_vram_gb} GB"
            )
    return errors
