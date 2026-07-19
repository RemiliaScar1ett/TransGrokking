"""Versioned, compatibility-checked training checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from transgrokking.config import ExperimentConfig
from transgrokking.utils.atomic import torch_save
from transgrokking.utils.reproducibility import capture_rng_state, restore_rng_state

CHECKPOINT_SCHEMA_VERSION = 2


def optimizer_group_signature(optimizer: Optimizer) -> list[dict[str, object]]:
    """Return the stable structural fields required for optimizer restoration."""
    return [
        {
            "group_name": group.get("group_name"),
            "parameter_names": list(group.get("parameter_names", [])),
            "weight_decay": group["weight_decay"],
            "learning_rate": group["lr"],
        }
        for group in optimizer.param_groups
    ]


def read_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Read a checkpoint payload without restoring runtime state."""
    return torch.load(path, map_location=map_location, weights_only=False)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    config: ExperimentConfig,
    split_hash: str,
    global_step: int,
) -> None:
    """Atomically save all state required for exact continuation without overwriting."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "optimizer_group_signature": optimizer_group_signature(optimizer),
        "scheduler_state": None,
        "global_step": global_step,
        "config": config.to_dict(),
        "scientific_config_hash": config.scientific_hash(),
        "split_hash": split_hash,
        "optimizer_type": type(optimizer).__name__.lower(),
        **capture_rng_state(),
    }
    torch_save(destination, payload)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    config: ExperimentConfig,
    split_hash: str,
    map_location: str | torch.device,
) -> int:
    """Validate and restore a checkpoint, returning its completed update count."""
    payload = read_checkpoint(path, map_location)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint.schema_version: expected {CHECKPOINT_SCHEMA_VERSION}, "
            f"got {payload.get('schema_version')!r}"
        )
    if payload.get("scientific_config_hash") != config.scientific_hash():
        raise ValueError("checkpoint scientific config hash does not match requested configuration")
    if payload.get("split_hash") != split_hash:
        raise ValueError("checkpoint split hash does not match generated split")
    if payload.get("optimizer_type") != type(optimizer).__name__.lower():
        raise ValueError("checkpoint optimizer type does not match")
    expected_groups = optimizer_group_signature(optimizer)
    if payload.get("optimizer_group_signature") != expected_groups:
        raise ValueError("checkpoint optimizer parameter-group structure does not match")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    restore_rng_state(payload)
    return int(payload["global_step"])
