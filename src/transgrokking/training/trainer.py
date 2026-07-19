"""CE-only deterministic full-batch training and safe resume lifecycle."""

from __future__ import annotations

import hashlib
import json
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F

from transgrokking.config import ExperimentConfig
from transgrokking.data import generate_modular_addition, split_artifact
from transgrokking.metrics.behavior import evaluate_model_behavior
from transgrokking.models import TransparentTransformer
from transgrokking.training.artifacts import (
    add_manifest_checkpoint,
    append_evaluation_artifacts,
    copy_metric_prefix,
    create_run_dir,
    load_manifest,
    reconcile_metric_files,
    scalar_steps,
    update_events,
    write_status,
)
from transgrokking.training.checkpoint import load_checkpoint, read_checkpoint, save_checkpoint
from transgrokking.training.optimizer import build_adamw, optimizer_group_metadata
from transgrokking.utils.atomic import torch_save, write_json, write_yaml
from transgrokking.utils.doctor import collect_doctor_report, validate_doctor_report
from transgrokking.utils.reproducibility import configure_reproducibility

ResumeMode = Literal["auto", "inplace", "branch"]


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
        final_norm=model.final_norm,
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _update_metadata(run_dir: Path, **updates: Any) -> None:
    path = run_dir / "metadata.json"
    metadata = _read_json(path) if path.exists() else {"schema_version": 2, "run_id": run_dir.name}
    metadata.update(updates)
    write_json(path, metadata)


def _save_regular_checkpoint(
    run_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    split_hash: str,
    global_step: int,
) -> Path:
    path = run_dir / "checkpoints" / f"step_{global_step:06d}.pt"
    save_checkpoint(path, model, optimizer, config, split_hash, global_step)
    add_manifest_checkpoint(run_dir, global_step, path)
    return path


def _resume_plan(
    config: ExperimentConfig, checkpoint_path: Path, requested_mode: ResumeMode
) -> tuple[ResumeMode, Path, int, dict[str, Any]]:
    if requested_mode not in {"auto", "inplace", "branch"}:
        raise ValueError(f"resume_mode: expected auto, inplace, or branch, got {requested_mode!r}")
    parent_run = checkpoint_path.parents[1]
    status_path = parent_run / "status.json"
    if not checkpoint_path.is_file() or not status_path.is_file():
        raise ValueError("resume checkpoint is not inside a valid run directory")
    entries = load_manifest(parent_run)
    matching = [entry for entry in entries if entry["path"] == checkpoint_path.name]
    if not matching:
        raise ValueError("resume checkpoint is not listed in its parent manifest")
    payload = read_checkpoint(checkpoint_path)
    parent_step = int(payload["global_step"])
    if payload.get("scientific_config_hash") != config.scientific_hash():
        raise ValueError("checkpoint scientific config hash does not match requested configuration")
    if config.optimization.max_steps <= parent_step:
        raise ValueError(
            "optimization.max_steps must be strictly greater than checkpoint global_step: "
            f"{config.optimization.max_steps} <= {parent_step}"
        )
    status = _read_json(status_path)
    latest = bool(entries) and int(entries[-1]["step"]) == parent_step
    reconcile_metric_files(parent_run)
    scalars = scalar_steps(parent_run / "metrics" / "scalars.jsonl")
    append_safe = not scalars or scalars[-1] <= parent_step
    inplace_allowed = status.get("state") == "interrupted" and latest and append_safe
    if requested_mode == "inplace" and not inplace_allowed:
        raise ValueError(
            "inplace resume requires an interrupted run's latest checkpoint "
            "and no newer scalar step"
        )
    resolved_mode: ResumeMode = (
        "inplace" if requested_mode == "auto" and inplace_allowed else requested_mode
    )
    if resolved_mode == "auto":
        resolved_mode = "branch"
    return resolved_mode, parent_run, parent_step, status


def train(
    config: ExperimentConfig,
    resume_from: str | Path | None = None,
    *,
    resume_mode: ResumeMode = "auto",
    stop_after: int | None = None,
) -> Path:
    """Run or resume training, returning the in-place or child run directory."""
    checkpoint_path = Path(resume_from).resolve() if resume_from is not None else None
    parent_run: Path | None = None
    parent_step: int | None = None
    resolved_mode: ResumeMode | None = None
    if checkpoint_path is not None:
        resolved_mode, parent_run, parent_step, _ = _resume_plan(
            config, checkpoint_path, resume_mode
        )
    elif resume_mode != "auto":
        raise ValueError("resume_mode is only valid with resume_from")

    config_hash = _config_hash(config)
    if resolved_mode == "inplace":
        assert parent_run is not None
        run_dir = parent_run
        write_status(run_dir, "initializing", global_step=parent_step, resume_mode="inplace")
    else:
        run_dir = create_run_dir(config.logging.runs_dir, config_hash)

    model: nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    global_step = parent_step or 0
    device = torch.device(config.optimization.device)
    split_hash: str | None = None
    try:
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

        configure_reproducibility(config.optimization.seed, config.optimization.deterministic)
        data = generate_modular_addition(
            config.task.modulus, config.task.train_fraction, config.task.split_seed
        )
        split_hash = data.split_hash
        write_yaml(run_dir / "config.resolved.yaml", config.to_dict())
        if resolved_mode != "inplace":
            torch_save(
                run_dir / "split.pt",
                split_artifact(data, config.task.modulus, config.task.split_seed),
            )
        elif not (run_dir / "split.pt").is_file():
            raise ValueError("in-place run is missing split.pt")

        parent_fields: dict[str, object] = {}
        if resolved_mode == "branch":
            assert (
                parent_run is not None and checkpoint_path is not None and parent_step is not None
            )
            parent_fields = {
                "parent_run_id": parent_run.name,
                "parent_checkpoint": str(checkpoint_path),
                "parent_global_step": parent_step,
            }
        base_metadata = {
            "schema_version": 2,
            "run_id": run_dir.name,
            "git_commit": _git_commit(),
            "config_hash": config_hash,
            "scientific_config_hash": config.scientific_hash(),
            "scientific_config": config.scientific_dict(),
            "split_hash": split_hash,
            "doctor": report.to_dict(),
            "precision": "fp32",
            "allow_tf32": False,
            "use_amp": False,
            "formal_run": config.hardware.formal_run,
            "resume_mode": resolved_mode,
            "metrics_schema_version": 1,
            "event_definitions": config.to_dict()["events"],
            **parent_fields,
        }
        if resolved_mode == "inplace":
            old_metadata = _read_json(run_dir / "metadata.json")
            history = list(old_metadata.get("resume_history", []))
            history.append(
                {
                    "checkpoint": str(checkpoint_path),
                    "global_step": parent_step,
                    "target_max_steps": config.optimization.max_steps,
                }
            )
            base_metadata["resume_history"] = history
        write_json(run_dir / "metadata.json", base_metadata)

        model = build_model(config)
        optimizer, grouping = build_adamw(model, config.optimization)
        _update_metadata(run_dir, optimizer_parameter_groups=optimizer_group_metadata(optimizer))
        model = model.to(device=device, dtype=torch.float32)
        inputs = data.inputs.to(device)
        labels = data.labels.to(device)
        train_indices = data.train_indices.to(device)
        test_indices = data.test_indices.to(device)

        if checkpoint_path is None:
            global_step = 0
            _save_regular_checkpoint(run_dir, model, optimizer, config, split_hash, global_step)
        else:
            global_step = load_checkpoint(
                checkpoint_path, model, optimizer, config, split_hash, device
            )
            if resolved_mode == "branch":
                _save_regular_checkpoint(run_dir, model, optimizer, config, split_hash, global_step)
                assert parent_run is not None and parent_step is not None
                copy_metric_prefix(
                    parent_run,
                    run_dir,
                    parent_step,
                    config.task.modulus,
                    config.logging.eval_interval,
                    config.events,
                )
            else:
                load_manifest(run_dir)
                reconcile_metric_files(run_dir)
                update_events(
                    run_dir,
                    config.task.modulus,
                    config.logging.eval_interval,
                    config.events,
                    preserve_existing=True,
                )
        if checkpoint_path is None:
            update_events(
                run_dir,
                config.task.modulus,
                config.logging.eval_interval,
                config.events,
                preserve_existing=False,
            )

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        write_status(
            run_dir,
            "running",
            global_step=global_step,
            resumed_from=str(checkpoint_path) if checkpoint_path else None,
            resume_mode=resolved_mode,
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
                behavior, offsets = evaluate_model_behavior(
                    model,
                    inputs,
                    labels,
                    train_indices,
                    test_indices,
                    config.task.modulus,
                    grouping,
                )
                scalar_record = {
                    "schema_version": 1,
                    "step": global_step,
                    **behavior,
                }
                offset_records = [
                    {
                        "schema_version": 1,
                        "step": global_step,
                        "split": split,
                        "modulus": config.task.modulus,
                        "counts": offsets[split],
                    }
                    for split in ("train", "test")
                ]
                append_evaluation_artifacts(
                    run_dir,
                    scalar_record,
                    offset_records,
                    config.task.modulus,
                    config.logging.eval_interval,
                    config.events,
                )

            checkpoint_due = global_step % config.logging.checkpoint_interval == 0
            final_step = global_step == config.optimization.max_steps
            interrupted_step = stop_after is not None and global_step >= stop_after
            if checkpoint_due or final_step or interrupted_step:
                _save_regular_checkpoint(run_dir, model, optimizer, config, split_hash, global_step)
            if interrupted_step:
                write_status(
                    run_dir, "interrupted", global_step=global_step, reason="test boundary"
                )
                return run_dir

        peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        _update_metadata(
            run_dir,
            completed_at=datetime.now(timezone.utc).isoformat(),
            final_global_step=global_step,
            max_memory_allocated=peak_allocated,
            max_memory_reserved=peak_reserved,
        )
        write_status(
            run_dir,
            "completed",
            global_step=global_step,
            max_memory_allocated=peak_allocated,
            max_memory_reserved=peak_reserved,
        )
        return run_dir
    except KeyboardInterrupt:
        emergency: str | None = None
        if model is not None and optimizer is not None and split_hash is not None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            emergency_path = (
                run_dir / "checkpoints" / f"emergency_step_{global_step:06d}_{stamp}.pt"
            )
            save_checkpoint(emergency_path, model, optimizer, config, split_hash, global_step)
            emergency = str(emergency_path)
        write_status(
            run_dir, "interrupted", global_step=global_step, emergency_checkpoint=emergency
        )
        raise
    except Exception as error:
        details: dict[str, Any] = {
            "global_step": global_step,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        if device.type == "cuda" and torch.cuda.is_available():
            details["max_memory_allocated"] = torch.cuda.max_memory_allocated(device)
            details["max_memory_reserved"] = torch.cuda.max_memory_reserved(device)
        write_status(run_dir, "failed", **details)
        if (run_dir / "metadata.json").exists():
            _update_metadata(run_dir, failure=details)
        raise
