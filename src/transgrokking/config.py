"""Strict experiment configuration and compatibility boundaries."""

from __future__ import annotations

import hashlib
import json
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
    final_norm: bool


@dataclass(frozen=True)
class DecayPolicyConfig:
    matrix_weights: bool
    embeddings: bool
    biases: bool
    layer_norm: bool


@dataclass(frozen=True)
class OptimizationConfig:
    optimizer: str
    learning_rate: float
    weight_decay: float
    decay_policy: DecayPolicyConfig
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

    def scientific_dict(self) -> dict[str, Any]:
        """Return fields that must remain identical when resuming training."""
        optimization = asdict(self.optimization)
        optimization.pop("max_steps")
        hardware = asdict(self.hardware)
        hardware.pop("analysis_batch_size")
        hardware.pop("activation_offload")
        return {
            "task": asdict(self.task),
            "model": asdict(self.model),
            "optimization": optimization,
            "hardware": hardware,
            "loss": asdict(self.loss),
        }

    def scientific_hash(self) -> str:
        """Return a stable SHA-256 digest of scientific configuration fields."""
        payload = json.dumps(self.scientific_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


T = TypeVar("T")


def _section(cls: type[T], value: Any, path: str) -> T:
    if type(value) is not dict:
        raise ValueError(f"{path}: expected mapping, got {value!r}")
    expected = {field.name for field in fields(cls)}
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown or missing:
        raise ValueError(f"{path}: unknown={sorted(unknown)}, missing={sorted(missing)}")
    return cls(**value)


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    """Parse and strictly validate an experiment configuration mapping."""
    if type(raw) is not dict:
        raise ValueError(f"config: expected mapping, got {raw!r}")
    expected = {field.name for field in fields(ExperimentConfig)}
    if set(raw) != expected:
        raise ValueError(
            f"config: unknown={sorted(set(raw) - expected)}, missing={sorted(expected - set(raw))}"
        )
    optimization_raw = raw["optimization"]
    if type(optimization_raw) is not dict:
        raise ValueError(f"optimization: expected mapping, got {optimization_raw!r}")
    optimization = dict(optimization_raw)
    optimization["decay_policy"] = _section(
        DecayPolicyConfig, optimization.get("decay_policy"), "optimization.decay_policy"
    )
    config = ExperimentConfig(
        task=_section(TaskConfig, raw["task"], "task"),
        model=_section(ModelConfig, raw["model"], "model"),
        optimization=_section(OptimizationConfig, optimization, "optimization"),
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
    return config_from_dict(raw)


def _require_type(path: str, value: Any, expected: type | tuple[type, ...]) -> None:
    expected_types = expected if isinstance(expected, tuple) else (expected,)
    if bool in expected_types:
        valid = type(value) is bool
    elif int in expected_types and float in expected_types:
        valid = type(value) in {int, float}
    else:
        valid = type(value) in expected_types
    if not valid:
        names = " or ".join(item.__name__ for item in expected_types)
        raise ValueError(f"{path}: expected {names}, got {value!r}")


def validate_config(config: ExperimentConfig) -> None:
    """Reject values outside the M0 scientific and engineering contract."""
    typed_values: list[tuple[str, Any, type | tuple[type, ...]]] = [
        ("task.modulus", config.task.modulus, int),
        ("task.train_fraction", config.task.train_fraction, (int, float)),
        ("task.split_seed", config.task.split_seed, int),
        ("model.d_model", config.model.d_model, int),
        ("model.n_heads", config.model.n_heads, int),
        ("model.n_layers", config.model.n_layers, int),
        ("model.d_mlp", config.model.d_mlp, int),
        ("model.dropout", config.model.dropout, (int, float)),
        ("model.activation", config.model.activation, str),
        ("model.norm_first", config.model.norm_first, bool),
        ("model.final_norm", config.model.final_norm, bool),
        ("optimization.optimizer", config.optimization.optimizer, str),
        ("optimization.learning_rate", config.optimization.learning_rate, (int, float)),
        ("optimization.weight_decay", config.optimization.weight_decay, (int, float)),
        ("optimization.max_steps", config.optimization.max_steps, int),
        ("optimization.precision", config.optimization.precision, str),
        ("optimization.allow_tf32", config.optimization.allow_tf32, bool),
        ("optimization.use_amp", config.optimization.use_amp, bool),
        ("optimization.deterministic", config.optimization.deterministic, bool),
        ("optimization.seed", config.optimization.seed, int),
        ("optimization.device", config.optimization.device, str),
        ("hardware.expected_device", config.hardware.expected_device, str),
        ("hardware.expected_vram_gb", config.hardware.expected_vram_gb, (int, float)),
        ("hardware.formal_run", config.hardware.formal_run, bool),
        ("hardware.analysis_batch_size", config.hardware.analysis_batch_size, int),
        ("hardware.activation_offload", config.hardware.activation_offload, bool),
        ("loss.cross_entropy_weight", config.loss.cross_entropy_weight, (int, float)),
        ("loss.congruence_weight", config.loss.congruence_weight, (int, float)),
        ("logging.eval_interval", config.logging.eval_interval, int),
        ("logging.checkpoint_interval", config.logging.checkpoint_interval, int),
        ("logging.activation_steps", config.logging.activation_steps, list),
        ("logging.runs_dir", config.logging.runs_dir, str),
    ]
    for field_path, value, expected in typed_values:
        _require_type(field_path, value, expected)
    for field in fields(DecayPolicyConfig):
        value = getattr(config.optimization.decay_policy, field.name)
        _require_type(f"optimization.decay_policy.{field.name}", value, bool)

    task, model, opt, hardware, loss, logging = (
        config.task,
        config.model,
        config.optimization,
        config.hardware,
        config.loss,
        config.logging,
    )
    if task.modulus < 2:
        raise ValueError(f"task.modulus: expected >= 2, got {task.modulus!r}")
    if not 0.0 < task.train_fraction < 1.0:
        raise ValueError(
            f"task.train_fraction: expected 0 < value < 1, got {task.train_fraction!r}"
        )
    for name in ("d_model", "n_heads", "n_layers", "d_mlp"):
        value = getattr(model, name)
        if value <= 0:
            raise ValueError(f"model.{name}: expected > 0, got {value!r}")
    if model.d_model % model.n_heads:
        raise ValueError(
            f"model.d_model: {model.d_model!r} must be divisible by model.n_heads={model.n_heads!r}"
        )
    if not 0.0 <= model.dropout < 1.0:
        raise ValueError(f"model.dropout: expected 0 <= value < 1, got {model.dropout!r}")
    if model.activation not in {"relu", "gelu"}:
        raise ValueError(f"model.activation: unsupported value {model.activation!r}")
    if not model.norm_first:
        raise ValueError(f"model.norm_first: M0 requires true, got {model.norm_first!r}")
    if opt.optimizer != "adamw":
        raise ValueError(f"optimization.optimizer: expected 'adamw', got {opt.optimizer!r}")
    if opt.learning_rate <= 0:
        raise ValueError(f"optimization.learning_rate: expected > 0, got {opt.learning_rate!r}")
    if opt.weight_decay < 0:
        raise ValueError(f"optimization.weight_decay: expected >= 0, got {opt.weight_decay!r}")
    if opt.max_steps < 1:
        raise ValueError(f"optimization.max_steps: expected >= 1, got {opt.max_steps!r}")
    if opt.precision != "fp32" or opt.allow_tf32 or opt.use_amp:
        raise ValueError("optimization: M0 requires fp32 with allow_tf32=false and use_amp=false")
    if opt.device != "cpu" and not opt.device.startswith("cuda:"):
        raise ValueError(
            f"optimization.device: expected 'cpu' or 'cuda:<index>', got {opt.device!r}"
        )
    if hardware.expected_vram_gb <= 0:
        raise ValueError(
            f"hardware.expected_vram_gb: expected > 0, got {hardware.expected_vram_gb!r}"
        )
    if hardware.analysis_batch_size < 1:
        raise ValueError(
            f"hardware.analysis_batch_size: expected >= 1, got {hardware.analysis_batch_size!r}"
        )
    if hardware.formal_run:
        if opt.device != "cuda:0":
            raise ValueError(
                f"optimization.device: formal run requires 'cuda:0', got {opt.device!r}"
            )
        if not hardware.expected_device.strip():
            raise ValueError(
                "hardware.expected_device: formal run requires non-empty value, "
                f"got {hardware.expected_device!r}"
            )
    if loss.cross_entropy_weight != 1.0 or loss.congruence_weight != 0.0:
        raise ValueError(
            "loss: M0 requires cross_entropy_weight=1.0 and congruence_weight=0.0, "
            f"got {asdict(loss)!r}"
        )
    for name in ("eval_interval", "checkpoint_interval"):
        value = getattr(logging, name)
        if value < 1:
            raise ValueError(f"logging.{name}: expected >= 1, got {value!r}")
    if not logging.runs_dir.strip():
        raise ValueError(f"logging.runs_dir: expected non-empty string, got {logging.runs_dir!r}")
    steps = logging.activation_steps
    if any(type(step) is not int or step < 0 for step in steps):
        raise ValueError(f"logging.activation_steps: expected nonnegative integers, got {steps!r}")
    if steps != sorted(set(steps)):
        raise ValueError(f"logging.activation_steps: expected sorted unique values, got {steps!r}")
