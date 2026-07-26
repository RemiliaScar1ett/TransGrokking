"""Strict configuration for read-only M2 function-space analysis."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from transgrokking.utils.atomic import write_yaml

M2_ANALYSIS_SCHEMA_VERSION = 1
_RUN_ID = re.compile(r"[0-9]{8}T[0-9]{12,}Z_[0-9a-f]{8}")


@dataclass(frozen=True)
class M2SourceRunIds:
    """Oldest-to-terminal M1 lineage identities."""

    root: str
    canonical_parent: str
    terminal_child: str


@dataclass(frozen=True)
class M2SourceResultDirs:
    """Frozen Git evidence directories checked before and after analysis."""

    m1_reference: str
    m1_extended: str


@dataclass(frozen=True)
class M2MathTolerances:
    """Scale-aware numerical acceptance thresholds."""

    centering_atol: float
    reconstruction_rtol: float
    orthogonality_normalized: float
    energy_identity_rtol: float
    invariance_rtol: float
    implication_margin_atol: float


@dataclass(frozen=True)
class M2AnalysisConfig:
    """Resolved M2 measurement configuration, independent of scientific config."""

    schema_version: int
    profile: str
    source_run_ids: M2SourceRunIds
    source_result_dirs: M2SourceResultDirs
    device: str
    expected_device: str
    expected_vram_gb: float
    analysis_batch_size: int
    cpu_reduction_dtype: str
    persist_selected_logits: bool
    selected_tensor_steps: list[int]
    selected_tensor_export_limit_mib: int
    math_tolerances: M2MathTolerances
    behavior_validation_atol: float
    behavior_validation_rtol: float
    replay_enabled: bool
    replay_temp_dir: str
    analysis_runs_dir: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def analysis_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
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


def _require(path: str, value: Any, expected: type) -> None:
    valid = type(value) in {int, float} if expected is float else type(value) is expected
    if not valid:
        raise ValueError(f"{path}: expected {expected.__name__}, got {value!r}")


def m2_analysis_config_from_dict(raw: dict[str, Any]) -> M2AnalysisConfig:
    """Strictly parse an M2 analysis mapping."""
    if type(raw) is not dict:
        raise ValueError(f"analysis: expected mapping, got {raw!r}")
    expected = {field.name for field in fields(M2AnalysisConfig)}
    unknown = set(raw) - expected
    missing = expected - set(raw)
    if unknown or missing:
        raise ValueError(f"analysis: unknown={sorted(unknown)}, missing={sorted(missing)}")
    config = M2AnalysisConfig(
        **{
            **raw,
            "source_run_ids": _section(M2SourceRunIds, raw["source_run_ids"], "source_run_ids"),
            "source_result_dirs": _section(
                M2SourceResultDirs, raw["source_result_dirs"], "source_result_dirs"
            ),
            "math_tolerances": _section(
                M2MathTolerances, raw["math_tolerances"], "math_tolerances"
            ),
        }
    )
    validate_m2_analysis_config(config)
    return config


def load_m2_analysis_config(path: str | Path) -> M2AnalysisConfig:
    with Path(path).open(encoding="utf-8") as handle:
        return m2_analysis_config_from_dict(yaml.safe_load(handle))


def dump_m2_analysis_config(config: M2AnalysisConfig, path: str | Path) -> None:
    validate_m2_analysis_config(config)
    write_yaml(path, config.to_dict())


def validate_m2_analysis_config(config: M2AnalysisConfig) -> None:
    """Validate identity, hardware, numerical, replay, and persistence settings."""
    typed: list[tuple[str, Any, type]] = [
        ("schema_version", config.schema_version, int),
        ("profile", config.profile, str),
        ("device", config.device, str),
        ("expected_device", config.expected_device, str),
        ("expected_vram_gb", config.expected_vram_gb, float),
        ("analysis_batch_size", config.analysis_batch_size, int),
        ("cpu_reduction_dtype", config.cpu_reduction_dtype, str),
        ("persist_selected_logits", config.persist_selected_logits, bool),
        ("selected_tensor_steps", config.selected_tensor_steps, list),
        ("selected_tensor_export_limit_mib", config.selected_tensor_export_limit_mib, int),
        ("behavior_validation_atol", config.behavior_validation_atol, float),
        ("behavior_validation_rtol", config.behavior_validation_rtol, float),
        ("replay_enabled", config.replay_enabled, bool),
        ("replay_temp_dir", config.replay_temp_dir, str),
        ("analysis_runs_dir", config.analysis_runs_dir, str),
    ]
    for path, value, expected in typed:
        _require(path, value, expected)
    for section_name, section in (
        ("source_run_ids", config.source_run_ids),
        ("source_result_dirs", config.source_result_dirs),
    ):
        for field in fields(section):
            _require(f"{section_name}.{field.name}", getattr(section, field.name), str)
    for field in fields(config.math_tolerances):
        _require(
            f"math_tolerances.{field.name}",
            getattr(config.math_tolerances, field.name),
            float,
        )

    if config.schema_version != M2_ANALYSIS_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version: expected {M2_ANALYSIS_SCHEMA_VERSION}, got {config.schema_version!r}"
        )
    if config.profile != "m2-function-space":
        raise ValueError(f"profile: expected 'm2-function-space', got {config.profile!r}")
    run_ids = asdict(config.source_run_ids)
    if len(set(run_ids.values())) != 3:
        raise ValueError(f"source_run_ids: expected three distinct runs, got {run_ids!r}")
    for name, value in run_ids.items():
        if _RUN_ID.fullmatch(value) is None:
            raise ValueError(f"source_run_ids.{name}: invalid run id {value!r}")
    for name, value in asdict(config.source_result_dirs).items():
        if not value.strip() or Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError(
                f"source_result_dirs.{name}: expected non-empty repository-relative path, "
                f"got {value!r}"
            )
    expected_source_dirs = {
        "m1_reference": "results/m1_ce_reference",
        "m1_extended": "results/m1_ce_reference_extended",
    }
    if asdict(config.source_result_dirs) != expected_source_dirs:
        raise ValueError(
            "source_result_dirs: formal M2 requires the two frozen M1 evidence directories"
        )
    if config.device != "cuda:0":
        raise ValueError(f"device: formal M2 requires 'cuda:0', got {config.device!r}")
    if config.expected_device != "NVIDIA GeForce RTX 4060 Laptop GPU":
        raise ValueError("expected_device: formal M2 requires NVIDIA GeForce RTX 4060 Laptop GPU")
    if not math.isfinite(float(config.expected_vram_gb)) or config.expected_vram_gb < 8:
        raise ValueError(
            f"expected_vram_gb: expected finite value >= 8, got {config.expected_vram_gb!r}"
        )
    if config.analysis_batch_size < 1:
        raise ValueError(f"analysis_batch_size: expected >= 1, got {config.analysis_batch_size!r}")
    if config.cpu_reduction_dtype != "float64":
        raise ValueError(
            f"cpu_reduction_dtype: expected 'float64', got {config.cpu_reduction_dtype!r}"
        )
    if not config.persist_selected_logits:
        raise ValueError("persist_selected_logits: formal M2 requires true")
    steps = config.selected_tensor_steps
    if any(type(step) is not int or step < 0 or step > 50_000 or step % 50 for step in steps):
        raise ValueError(
            f"selected_tensor_steps: expected 50-step-grid integers in [0, 50000], got {steps!r}"
        )
    if steps != sorted(set(steps)):
        raise ValueError(f"selected_tensor_steps: expected sorted unique values, got {steps!r}")
    if config.selected_tensor_export_limit_mib < 1:
        raise ValueError("selected_tensor_export_limit_mib: expected >= 1")
    for name, value in asdict(config.math_tolerances).items():
        if not math.isfinite(float(value)) or not 0.0 < value <= 1.0:
            raise ValueError(f"math_tolerances.{name}: expected 0 < value <= 1, got {value!r}")
    for name in ("behavior_validation_atol", "behavior_validation_rtol"):
        value = getattr(config, name)
        if not math.isfinite(float(value)) or not 0.0 <= value <= 1.0e-6:
            raise ValueError(f"{name}: expected 0 <= value <= 1e-6, got {value!r}")
    if not config.replay_enabled:
        raise ValueError("replay_enabled: M2-A requires true")
    for name in ("replay_temp_dir", "analysis_runs_dir"):
        value = getattr(config, name)
        if not value.strip() or Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError(f"{name}: expected repository-relative path, got {value!r}")
    if Path(config.analysis_runs_dir).as_posix() != "analysis_runs":
        raise ValueError("analysis_runs_dir: formal M2 requires 'analysis_runs'")
    replay_parts = Path(config.replay_temp_dir).parts
    if not replay_parts or replay_parts[0].lower() != "temp":
        raise ValueError("replay_temp_dir: formal M2 requires a path below temp/")
