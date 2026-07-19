"""CE-only deterministic full-batch training."""

from __future__ import annotations

import hashlib
import json
import subprocess
import traceback
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from transgrokking.config import ExperimentConfig
from transgrokking.data import generate_modular_addition, split_artifact
from transgrokking.models import TransparentTransformer
from transgrokking.training.artifacts import (
    append_scalar,
    create_run_dir,
    write_status,
)
from transgrokking.training.checkpoint import load_checkpoint, save_checkpoint
from transgrokking.utils.atomic import torch_save, write_json, write_yaml
from transgrokking.utils.doctor import (
    collect_doctor_report,
    validate_doctor_report,
)
from transgrokking.utils.reproducibility import configure_reproducibility


def build_model(config: ExperimentConfig) -> TransparentTransformer:
    """Build the M0 transparent Transformer from resolved configuration."""
    model = config.model
    return TransparentTransformer(
        vocab_size=config.task.modulus,
        sequence_length=2,
        d_model=model.d_model,
        n_heads=model.n_heads,
        n_layers=model.n_layers,
        d_mlp=model.d_mlp,
        dropout=model.dropout,
        activation=model.activation,
        norm_first=model.norm_first,
    )


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=5
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def _config_hash(config: ExperimentConfig) -> str:
    encoded = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _evaluate(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(inputs.index_select(0, indices))[:, -1]
        targets = labels.index_select(0, indices)
        loss = F.cross_entropy(logits, targets).item()
        accuracy = (logits.argmax(dim=-1) == targets).float().mean().item()
    return loss, accuracy


def _manifest(run_dir: Path, steps: list[int]) -> None:
    write_json(
        run_dir / "checkpoints" / "manifest.json",
        {
            "schema_version": 1,
            "checkpoints": [
                {"step": step, "path": f"step_{step:06d}.pt"} for step in sorted(set(steps))
            ],
        },
    )


def train(
    config: ExperimentConfig,
    resume_from: str | Path | None = None,
    *,
    stop_after: int | None = None,
) -> Path:
    """Run or resume CE-only full-batch training and return the run directory.

    ``stop_after`` is a test-only interruption boundary expressed as a global step.
    """
    configure_reproducibility(config.optimization.seed, config.optimization.deterministic)
    device = torch.device(config.optimization.device)
    report = collect_doctor_report()
    if config.hardware.formal_run:
        errors = validate_doctor_report(
            report,
            require_cuda=True,
            expected_device=config.hardware.expected_device,
            expected_vram_gb=config.hardware.expected_vram_gb,
        )
        if errors:
            raise RuntimeError("formal-run doctor failed: " + "; ".join(errors))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"configured device {device} is unavailable")

    data = generate_modular_addition(
        config.task.modulus, config.task.train_fraction, config.task.split_seed
    )
    config_hash = _config_hash(config)
    checkpoint_path = Path(resume_from).resolve() if resume_from is not None else None
    run_dir = (
        checkpoint_path.parents[1]
        if checkpoint_path is not None
        else create_run_dir(config.logging.runs_dir, config_hash)
    )
    status_exists = (run_dir / "status.json").exists()
    if checkpoint_path is not None and not status_exists:
        raise ValueError("resume checkpoint is not inside a valid run directory")

    if checkpoint_path is None:
        write_yaml(run_dir / "config.resolved.yaml", config.to_dict())
        torch_save(
            run_dir / "split.pt",
            split_artifact(data, config.task.modulus, config.task.split_seed),
        )
        write_json(
            run_dir / "metadata.json",
            {
                "schema_version": 1,
                "run_id": run_dir.name,
                "git_commit": _git_commit(),
                "config_hash": config_hash,
                "split_hash": data.split_hash,
                "doctor": report.to_dict(),
                "precision": "fp32",
                "allow_tf32": False,
                "use_amp": False,
                "formal_run": config.hardware.formal_run,
            },
        )
    write_status(run_dir, "running", global_step=0)

    model = build_model(config).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    inputs = data.inputs.to(device)
    labels = data.labels.to(device)
    train_indices = data.train_indices.to(device)
    test_indices = data.test_indices.to(device)
    completed_steps: list[int] = []

    try:
        if checkpoint_path is None:
            global_step = 0
            initial = run_dir / "checkpoints" / "step_000000.pt"
            save_checkpoint(initial, model, optimizer, config, data.split_hash, global_step)
            completed_steps.append(0)
            _manifest(run_dir, completed_steps)
        else:
            global_step = load_checkpoint(
                checkpoint_path, model, optimizer, config, data.split_hash, device
            )
            manifest_path = run_dir / "checkpoints" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            completed_steps = [int(item["step"]) for item in manifest["checkpoints"]]
            write_status(
                run_dir, "running", global_step=global_step, resumed_from=str(checkpoint_path)
            )

        while global_step < config.optimization.max_steps:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs.index_select(0, train_indices))[:, -1]
            targets = labels.index_select(0, train_indices)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            global_step += 1

            if (
                global_step % config.logging.eval_interval == 0
                or global_step == config.optimization.max_steps
            ):
                train_loss, train_accuracy = _evaluate(model, inputs, labels, train_indices)
                test_loss, test_accuracy = _evaluate(model, inputs, labels, test_indices)
                append_scalar(
                    run_dir / "metrics" / "scalars.jsonl",
                    {
                        "step": global_step,
                        "train_cross_entropy": train_loss,
                        "test_cross_entropy": test_loss,
                        "train_accuracy": train_accuracy,
                        "test_accuracy": test_accuracy,
                        "congruence_loss": 0.0,
                    },
                )

            checkpoint_due = global_step % config.logging.checkpoint_interval == 0
            final_step = global_step == config.optimization.max_steps
            interrupted_step = stop_after is not None and global_step >= stop_after
            if checkpoint_due or final_step or interrupted_step:
                path = run_dir / "checkpoints" / f"step_{global_step:06d}.pt"
                save_checkpoint(path, model, optimizer, config, data.split_hash, global_step)
                completed_steps.append(global_step)
                _manifest(run_dir, completed_steps)
            if interrupted_step:
                write_status(
                    run_dir, "interrupted", global_step=global_step, reason="test boundary"
                )
                return run_dir

        peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        write_status(
            run_dir,
            "completed",
            global_step=global_step,
            max_memory_allocated=peak_allocated,
            max_memory_reserved=peak_reserved,
        )
        return run_dir
    except KeyboardInterrupt:
        write_status(run_dir, "interrupted", global_step=locals().get("global_step", 0))
        raise
    except Exception as error:
        details: dict[str, Any] = {
            "global_step": locals().get("global_step", 0),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        if isinstance(error, torch.cuda.OutOfMemoryError) and torch.cuda.is_available():
            details["max_memory_allocated"] = torch.cuda.max_memory_allocated()
            details["max_memory_reserved"] = torch.cuda.max_memory_reserved()
        write_status(run_dir, "failed", **details)
        raise
