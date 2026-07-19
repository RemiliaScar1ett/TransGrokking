"""Strict experiment configuration loading and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass(frozen=True)
class TaskConfig:
    modulus: int
    train_fraction: float
    split_seed: int


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    n_heads: int
    n_layers: int
    d_mlp: int
    dropout: float
    activation: str
    norm_first: bool


@dataclass(frozen=True)
class OptimizationConfig:
    optimizer: str
    learning_rate: float
    weight_decay: float
    max_steps: int
    precision: str
    allow_tf32: bool
    use_amp: bool
    deterministic: bool
    seed: int
    device: str


@dataclass(frozen=True)
class HardwareConfig:
    expected_device: str
    expected_vram_gb: float
    formal_run: bool
    analysis_batch_size: int
    activation_offload: bool


@dataclass(frozen=True)
class LossConfig:
    cross_entropy_weight: float
    congruence_weight: float


@dataclass(frozen=True)
class LoggingConfig:
    eval_interval: int
    checkpoint_interval: int
    optimizer_checkpoint_interval: int
    activation_steps: list[int]
    runs_dir: str


@dataclass(frozen=True)
class ExperimentConfig:
    task: TaskConfig
    model: ModelConfig
    optimization: OptimizationConfig
    hardware: HardwareConfig
    loss: LossConfig
    logging: LoggingConfig

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-serializable resolved configuration."""
        return asdict(self)


T = TypeVar("T")


def _section(cls: type[T], value: Any, name: str) -> T:
    if not isinstance(value, dict):
        raise ValueError(f"config section {name!r} must be a mapping")
    expected = {field.name for field in fields(cls)}
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown or missing:
        raise ValueError(
            f"invalid {name} fields: unknown={sorted(unknown)}, missing={sorted(missing)}"
        )
    return cls(**value)


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    """Parse and strictly validate an experiment configuration mapping."""
    expected = {field.name for field in fields(ExperimentConfig)}
    if set(raw) != expected:
        raise ValueError(
            f"invalid top-level fields: unknown={sorted(set(raw) - expected)}, "
            f"missing={sorted(expected - set(raw))}"
        )
    config = ExperimentConfig(
        task=_section(TaskConfig, raw["task"], "task"),
        model=_section(ModelConfig, raw["model"], "model"),
        optimization=_section(OptimizationConfig, raw["optimization"], "optimization"),
        hardware=_section(HardwareConfig, raw["hardware"], "hardware"),
        loss=_section(LossConfig, raw["loss"], "loss"),
        logging=_section(LoggingConfig, raw["logging"], "logging"),
    )
    validate_config(config)
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a UTF-8 YAML experiment configuration."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return config_from_dict(raw)


def validate_config(config: ExperimentConfig) -> None:
    """Reject settings outside the M0 scientific and engineering contract."""
    task, model, opt, hardware, loss, logging = (
        config.task,
        config.model,
        config.optimization,
        config.hardware,
        config.loss,
        config.logging,
    )
    if task.modulus < 2 or not 0.0 < task.train_fraction < 1.0:
        raise ValueError("modulus must be >=2 and train_fraction must be between 0 and 1")
    if min(model.d_model, model.n_heads, model.n_layers, model.d_mlp) <= 0:
        raise ValueError("model dimensions and layer counts must be positive")
    if model.d_model % model.n_heads:
        raise ValueError("d_model must be divisible by n_heads")
    if not 0.0 <= model.dropout < 1.0 or model.activation not in {"relu", "gelu"}:
        raise ValueError("invalid dropout or activation")
    if opt.optimizer != "adamw" or opt.learning_rate <= 0 or opt.weight_decay < 0:
        raise ValueError("M0 supports AdamW with positive learning rate and nonnegative decay")
    if opt.max_steps < 0 or opt.precision != "fp32" or opt.allow_tf32 or opt.use_amp:
        raise ValueError("M0 requires fp32 with TF32 and AMP disabled")
    if opt.device != "cpu" and not opt.device.startswith("cuda:"):
        raise ValueError("device must be cpu or an explicit cuda index")
    if hardware.formal_run and opt.device != "cuda:0":
        raise ValueError("formal runs must use cuda:0")
    if loss.cross_entropy_weight != 1.0 or loss.congruence_weight != 0.0:
        raise ValueError("M0 is CE-only with weights 1.0 and 0.0")
    if (
        min(
            logging.eval_interval,
            logging.checkpoint_interval,
            logging.optimizer_checkpoint_interval,
            hardware.analysis_batch_size,
        )
        <= 0
    ):
        raise ValueError("logging intervals and analysis batch size must be positive")
