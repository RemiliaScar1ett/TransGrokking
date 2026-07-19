"""Explicit numerical and random-state controls."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    """Seed all supported RNGs and force the M0 FP32 numerical policy."""
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(deterministic)


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, and Torch RNG state."""
    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state captured by :func:`capture_rng_state`."""
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"])
    if state["torch_cuda_rng_state"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_state"])
