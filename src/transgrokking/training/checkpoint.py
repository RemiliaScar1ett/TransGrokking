"""Versioned training checkpoint save and restore."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from transgrokking.config import ExperimentConfig
from transgrokking.utils.atomic import torch_save
from transgrokking.utils.reproducibility import capture_rng_state, restore_rng_state

CHECKPOINT_SCHEMA_VERSION = 1


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    config: ExperimentConfig,
    split_hash: str,
    global_step: int,
) -> None:
    """Atomically save all state required for exact continuation."""
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None,
        "global_step": global_step,
        "config": config.to_dict(),
        "split_hash": split_hash,
        "optimizer_type": type(optimizer).__name__.lower(),
        **capture_rng_state(),
    }
    torch_save(path, payload)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    config: ExperimentConfig,
    split_hash: str,
    map_location: str | torch.device,
) -> int:
    """Validate and restore a checkpoint, returning its completed update count."""
    payload: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema version")
    if payload.get("config") != config.to_dict():
        raise ValueError("checkpoint configuration does not match requested configuration")
    if payload.get("split_hash") != split_hash:
        raise ValueError("checkpoint split hash does not match generated split")
    if payload.get("optimizer_type") != type(optimizer).__name__.lower():
        raise ValueError("checkpoint optimizer type does not match")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    restore_rng_state(payload)
    return int(payload["global_step"])
