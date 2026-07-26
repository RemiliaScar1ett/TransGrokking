"""Read-only model loading and full-table evaluation for M2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from transgrokking.config import ExperimentConfig
from transgrokking.data import ModularAdditionData
from transgrokking.metrics.behavior import evaluate_logits, evaluate_model_behavior
from transgrokking.training.checkpoint import load_checkpoint
from transgrokking.training.optimizer import (
    ParameterGrouping,
    build_adamw,
    validate_optimizer_parameter_identity,
)
from transgrokking.training.trainer import build_model


def build_loaded_runtime(
    checkpoint: str | Path,
    config: ExperimentConfig,
    data: ModularAdditionData,
    device: torch.device,
) -> tuple[nn.Module, torch.optim.Optimizer, ParameterGrouping, int]:
    """Build model/optimizer on the target device and restore one checkpoint."""
    model = build_model(config).to(device=device, dtype=torch.float32)
    optimizer, grouping = build_adamw(model, config.optimization)
    validate_optimizer_parameter_identity(model, optimizer)
    step = load_checkpoint(checkpoint, model, optimizer, config, data.split_hash, device)
    return model, optimizer, grouping, step


def full_table_logits(
    model: nn.Module,
    inputs: torch.Tensor,
    modulus: int,
    batch_size: int,
) -> torch.Tensor:
    """Return CPU FP32 logits shaped ``[p,p,p]`` in a-major order."""
    if inputs.shape != (modulus * modulus, 2) or inputs.dtype != torch.long:
        raise ValueError(
            f"inputs: expected [{modulus * modulus},2] torch.long, "
            f"got {tuple(inputs.shape)} {inputs.dtype}"
        )
    if batch_size < 1:
        raise ValueError(f"batch_size: expected >= 1, got {batch_size!r}")
    cpu_inputs = inputs.detach().to("cpu")
    values = torch.arange(modulus, dtype=torch.long)
    expected_a = values.repeat_interleave(modulus)
    expected_b = values.repeat(modulus)
    if not torch.equal(cpu_inputs[:, 0], expected_a) or not torch.equal(
        cpu_inputs[:, 1], expected_b
    ):
        raise ValueError("inputs are not ordered a-major, b-minor")
    device = next(model.parameters()).device
    output = torch.empty((modulus * modulus, modulus), dtype=torch.float32, device="cpu")
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, inputs.shape[0], batch_size):
                stop = min(start + batch_size, inputs.shape[0])
                logits = model(cpu_inputs[start:stop].to(device))[:, -1]
                output[start:stop].copy_(logits.detach().to(device="cpu", dtype=torch.float32))
    finally:
        model.train(was_training)
    return output.reshape(modulus, modulus, modulus)


def behavior_snapshot(
    model: nn.Module,
    data: ModularAdditionData,
    grouping: ParameterGrouping,
    device: torch.device,
    *,
    full_logits: torch.Tensor | None = None,
) -> tuple[dict[str, Any], dict[str, list[int]]]:
    """Recompute committed M1 behavior plus full-table raw behavior."""
    with torch.inference_mode():
        scalars, offsets = evaluate_model_behavior(
            model,
            data.inputs.to(device),
            data.labels.to(device),
            data.train_indices.to(device),
            data.test_indices.to(device),
            int(data.labels.max().item()) + 1,
            grouping,
        )
    if full_logits is None:
        full_logits = full_table_logits(
            model,
            data.inputs,
            int(data.labels.max().item()) + 1,
            data.inputs.shape[0],
        )
    flat = full_logits.reshape(data.inputs.shape[0], -1)
    full = evaluate_logits(flat, data.labels, flat.shape[1])
    scalars.update({f"full_{key}": value for key, value in full.scalars.items()})
    offsets["full"] = full.error_offsets
    return scalars, offsets
