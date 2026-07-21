"""Read-only M1 evaluation of a run checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from transgrokking.config import load_config
from transgrokking.data import generate_modular_addition
from transgrokking.metrics.behavior import evaluate_model_behavior
from transgrokking.training.artifacts import load_manifest
from transgrokking.training.checkpoint import load_checkpoint, read_checkpoint
from transgrokking.training.optimizer import build_adamw, validate_optimizer_parameter_identity
from transgrokking.training.trainer import build_model
from transgrokking.utils.reproducibility import configure_reproducibility


def resolve_checkpoint(run_dir: str | Path, checkpoint: str | None) -> Path:
    """Resolve latest, numeric-step, or manifest-listed checkpoint selection."""
    root = Path(run_dir).resolve()
    entries = load_manifest(root)
    if not entries:
        raise ValueError(f"run has no manifested checkpoints: {root}")
    if checkpoint is None:
        selected = root / "checkpoints" / str(entries[-1]["path"])
    elif checkpoint.isdigit():
        step = int(checkpoint)
        matches = [entry for entry in entries if entry["step"] == step]
        if not matches:
            raise ValueError(f"checkpoint step {step} is not in manifest")
        selected = root / "checkpoints" / str(matches[0]["path"])
    else:
        candidate = Path(checkpoint)
        if not candidate.is_absolute():
            candidate = (
                candidate.resolve() if candidate.exists() else root / "checkpoints" / candidate
            )
        selected = candidate.resolve()
    manifested = {(root / "checkpoints" / str(entry["path"])).resolve() for entry in entries}
    if selected.resolve() not in manifested:
        raise ValueError(f"checkpoint is not listed in run manifest: {selected}")
    return selected.resolve()


def evaluate_run_checkpoint(run_dir: str | Path, checkpoint: str | None = None) -> dict[str, Any]:
    """Recompute one checkpoint's M1 behavior without modifying run artifacts."""
    root = Path(run_dir).resolve()
    config = load_config(root / "config.resolved.yaml")
    selected = resolve_checkpoint(root, checkpoint)
    payload = read_checkpoint(selected)
    split = torch.load(root / "split.pt", map_location="cpu", weights_only=False)
    data = generate_modular_addition(
        config.task.modulus, config.task.train_fraction, config.task.split_seed
    )
    if split.get("split_hash") != data.split_hash or payload.get("split_hash") != data.split_hash:
        raise ValueError("run split artifact, checkpoint, and resolved config do not agree")
    if not torch.equal(split.get("train_indices"), data.train_indices) or not torch.equal(
        split.get("test_indices"), data.test_indices
    ):
        raise ValueError("run split indices do not match deterministic regeneration")
    configure_reproducibility(config.optimization.seed, config.optimization.deterministic)
    device = torch.device(config.optimization.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"configured device {device} is unavailable")
    model = build_model(config).to(device=device, dtype=torch.float32)
    optimizer, grouping = build_adamw(model, config.optimization)
    validate_optimizer_parameter_identity(model, optimizer)
    step = load_checkpoint(selected, model, optimizer, config, data.split_hash, device)
    behavior, offsets = evaluate_model_behavior(
        model,
        data.inputs.to(device),
        data.labels.to(device),
        data.train_indices.to(device),
        data.test_indices.to(device),
        config.task.modulus,
        grouping,
    )
    return {
        "schema_version": 1,
        "run_id": root.name,
        "checkpoint": str(selected),
        "step": step,
        **behavior,
        "error_offsets": offsets,
    }
