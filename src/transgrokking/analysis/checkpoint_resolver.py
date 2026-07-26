"""Read-only checkpoint lineage resolution and semantic training-state identity."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from transgrokking.training.checkpoint import CHECKPOINT_SCHEMA_VERSION, read_checkpoint


class LineageValidationError(ValueError):
    """Raised when manifests do not form the required complete checkpoint grid."""

    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class LineageConflictError(LineageValidationError):
    """Raised when two physical checkpoints claim one step but differ semantically."""


@dataclass(frozen=True)
class SemanticStateComparison:
    """Field-level comparison of two normalized training states."""

    equal: bool
    left_sha256: str
    right_sha256: str
    differing_components: tuple[str, ...]
    detail_differences: tuple[str, ...]


@dataclass(frozen=True)
class PhysicalCheckpoint:
    """One physical checkpoint listed by one run manifest."""

    step: int
    run_id: str
    checkpoint_path: Path
    checkpoint_relative_path: str
    checkpoint_sha256: str
    semantic_state_sha256: str
    source_git_commit: str | None
    scientific_config_hash: str
    split_hash: str
    lineage_depth: int

    def to_record(self) -> dict[str, Any]:
        """Return a portable CSV/JSON record without an absolute path."""
        return {
            "step": self.step,
            "run_id": self.run_id,
            "checkpoint_relative_path": self.checkpoint_relative_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "semantic_state_sha256": self.semantic_state_sha256,
            "source_git_commit": self.source_git_commit,
            "scientific_config_hash": self.scientific_config_hash,
            "split_hash": self.split_hash,
            "lineage_depth": self.lineage_depth,
        }


@dataclass(frozen=True)
class CanonicalCheckpoint:
    """Unique semantic state selected for one absolute global step."""

    step: int
    physical: PhysicalCheckpoint
    alias_count: int

    def to_record(self) -> dict[str, Any]:
        record = self.physical.to_record()
        record["alias_count"] = self.alias_count
        return record


@dataclass(frozen=True)
class AliasCheckpoint:
    """Two physical encodings of a verified-identical branch-anchor state."""

    step: int
    canonical_run_id: str
    alias_run_id: str
    canonical_checkpoint_sha256: str
    alias_checkpoint_sha256: str
    semantic_state_sha256: str
    raw_sha256_equal: bool

    @property
    def alias_group_id(self) -> str:
        """Return a stable identifier shared by physical encodings of this state."""
        return f"step_{self.step:06d}_{self.semantic_state_sha256[:16]}"

    def to_record(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "canonical_run_id": self.canonical_run_id,
            "alias_run_id": self.alias_run_id,
            "canonical_checkpoint_sha256": self.canonical_checkpoint_sha256,
            "alias_checkpoint_sha256": self.alias_checkpoint_sha256,
            "semantic_state_sha256": self.semantic_state_sha256,
            "semantic_state_equal": True,
            "raw_sha256_equal": self.raw_sha256_equal,
            "alias_group_id": self.alias_group_id,
        }


@dataclass(frozen=True)
class LineageSegment:
    """One manifested run segment, including its branch anchor."""

    run_id: str
    run_dir: Path
    start_step: int
    end_step: int
    lineage_depth: int
    parent_run_id: str | None


@dataclass(frozen=True)
class CheckpointLineage:
    """Resolved physical files, canonical states, aliases, and run segments."""

    physical: tuple[PhysicalCheckpoint, ...]
    canonical: tuple[CanonicalCheckpoint, ...]
    aliases: tuple[AliasCheckpoint, ...]
    segments: tuple[LineageSegment, ...]
    regular_interval: int
    expected_start_step: int
    expected_end_step: int

    @property
    def physical_count(self) -> int:
        return len(self.physical)

    @property
    def regular_step_count(self) -> int:
        return len(self.canonical)

    def checkpoint_at(self, step: int) -> CanonicalCheckpoint:
        """Return the unique canonical state at ``step``."""
        for checkpoint in self.canonical:
            if checkpoint.step == step:
                return checkpoint
        raise KeyError(f"checkpoint step is not present in canonical index: {step}")

    def segment_for_replay_target(self, target_step: int) -> LineageSegment:
        """Select the segment whose training path produced a non-anchor target step."""
        if not self.expected_start_step < target_step <= self.expected_end_step:
            raise ValueError(
                f"replay target must be in ({self.expected_start_step}, "
                f"{self.expected_end_step}], got {target_step}"
            )
        for index, segment in enumerate(self.segments):
            if index == 0:
                if segment.start_step < target_step <= segment.end_step:
                    return segment
            elif segment.start_step < target_step <= segment.end_step:
                return segment
        raise KeyError(f"no lineage segment owns replay target step {target_step}")

    def physical_checkpoint(self, run_id: str, step: int) -> PhysicalCheckpoint:
        """Return the manifested file belonging to an explicit run and step."""
        matches = [item for item in self.physical if item.run_id == run_id and item.step == step]
        if len(matches) != 1:
            raise KeyError(
                f"expected one physical checkpoint for {run_id}:{step}, got {len(matches)}"
            )
        return matches[0]


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest without changing the source file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_length(digest: Any, length: int) -> None:
    digest.update(struct.pack(">Q", length))


def _update_digest(digest: Any, value: Any) -> None:
    """Encode supported values with explicit type and length framing."""
    if value is None:
        digest.update(b"N")
    elif type(value) is bool:
        digest.update(b"B1" if value else b"B0")
    elif type(value) is int:
        encoded = str(value).encode("ascii")
        digest.update(b"I")
        _write_length(digest, len(encoded))
        digest.update(encoded)
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"semantic state contains non-finite float: {value!r}")
        digest.update(b"F")
        digest.update(struct.pack(">d", value))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"S")
        _write_length(digest, len(encoded))
        digest.update(encoded)
    elif isinstance(value, bytes):
        digest.update(b"Y")
        _write_length(digest, len(value))
        digest.update(value)
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu").contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        digest.update(b"T")
        _update_digest(digest, str(tensor.dtype))
        _update_digest(digest, list(tensor.shape))
        _write_length(digest, len(raw))
        digest.update(raw)
    elif isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("semantic state does not support object NumPy arrays")
        array = np.ascontiguousarray(value)
        digest.update(b"A")
        _update_digest(digest, array.dtype.str)
        _update_digest(digest, list(array.shape))
        raw = array.tobytes(order="C")
        _write_length(digest, len(raw))
        digest.update(raw)
    elif isinstance(value, np.generic):
        _update_digest(digest, value.item())
    elif isinstance(value, tuple):
        digest.update(b"U")
        _write_length(digest, len(value))
        for item in value:
            _update_digest(digest, item)
    elif isinstance(value, list):
        digest.update(b"L")
        _write_length(digest, len(value))
        for item in value:
            _update_digest(digest, item)
    elif isinstance(value, dict):
        digest.update(b"D")
        encoded_items: list[tuple[bytes, Any]] = []
        for key, item in value.items():
            key_digest = hashlib.sha256()
            _update_digest(key_digest, key)
            encoded_items.append((key_digest.digest(), (key, item)))
        encoded_items.sort(key=lambda pair: pair[0])
        _write_length(digest, len(encoded_items))
        for _, (key, item) in encoded_items:
            _update_digest(digest, key)
            _update_digest(digest, item)
    else:
        raise TypeError(f"unsupported semantic-state value type: {type(value).__name__}")


def _digest_value(value: Any) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, value)
    return digest.hexdigest()


def _normalized_optimizer_state(payload: dict[str, Any]) -> dict[str, Any]:
    optimizer = payload.get("optimizer_state")
    signatures = payload.get("optimizer_group_signature")
    if not isinstance(optimizer, dict) or not isinstance(signatures, list):
        raise ValueError("checkpoint optimizer state or group signature is missing")
    groups = optimizer.get("param_groups")
    states = optimizer.get("state")
    if not isinstance(groups, list) or not isinstance(states, dict):
        raise ValueError("checkpoint optimizer state has invalid groups or state mapping")
    if len(groups) != len(signatures):
        raise ValueError("optimizer groups and parameter-group signature count differ")

    normalized_groups: list[dict[str, Any]] = []
    state_by_name: dict[str, Any] = {}
    seen_ids: set[Any] = set()
    for group, signature in zip(groups, signatures, strict=True):
        if not isinstance(group, dict) or not isinstance(signature, dict):
            raise ValueError("optimizer group and signature entries must be mappings")
        parameter_ids = group.get("params")
        names = signature.get("parameter_names")
        if not isinstance(parameter_ids, list) or not isinstance(names, list):
            raise ValueError("optimizer group params and signature names must be lists")
        if len(parameter_ids) != len(names):
            raise ValueError("optimizer parameter IDs and stable names differ in length")
        if len(set(names)) != len(names):
            raise ValueError("optimizer parameter signature contains duplicate names")
        for parameter_id, name in zip(parameter_ids, names, strict=True):
            if parameter_id in seen_ids:
                raise ValueError(f"optimizer parameter ID appears more than once: {parameter_id}")
            if not isinstance(name, str) or not name:
                raise ValueError(f"optimizer parameter name must be nonempty: {name!r}")
            seen_ids.add(parameter_id)
            state_by_name[name] = states.get(parameter_id, {})
        hyperparameters = {
            key: value for key, value in group.items() if key not in {"params", "parameter_names"}
        }
        normalized_groups.append(
            {
                "group_name": signature.get("group_name"),
                "parameter_names": sorted(names),
                "hyperparameters": hyperparameters,
            }
        )
    unknown_state_ids = set(states) - seen_ids
    if unknown_state_ids:
        raise ValueError(f"optimizer state has unknown parameter IDs: {sorted(unknown_state_ids)}")
    normalized_groups.sort(key=lambda group: str(group["group_name"]))
    return {
        "groups": normalized_groups,
        "state_by_parameter_name": dict(sorted(state_by_name.items())),
    }


def _semantic_components(payload: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "model_state",
        "optimizer_state",
        "optimizer_group_signature",
        "scheduler_state",
        "global_step",
        "scientific_config_hash",
        "split_hash",
        "optimizer_type",
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_state",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"checkpoint is missing semantic state fields: {missing}")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint schema must equal {CHECKPOINT_SCHEMA_VERSION}, "
            f"got {payload['schema_version']!r}"
        )
    model_state = payload["model_state"]
    if not isinstance(model_state, dict):
        raise ValueError("checkpoint model_state must be a mapping")
    cuda_rng = payload["torch_cuda_rng_state"]
    if cuda_rng is not None and not isinstance(cuda_rng, (list, tuple)):
        raise ValueError(
            "checkpoint torch_cuda_rng_state must be a device-ordered sequence or null"
        )
    return {
        "metadata": {
            "schema_version": payload["schema_version"],
            "global_step": payload["global_step"],
            "scientific_config_hash": payload["scientific_config_hash"],
            "split_hash": payload["split_hash"],
            "optimizer_type": payload["optimizer_type"],
            "scheduler_state": payload["scheduler_state"],
        },
        "model_state": dict(sorted(model_state.items())),
        "optimizer_state": _normalized_optimizer_state(payload),
        "optimizer_group_signature": sorted(
            payload["optimizer_group_signature"], key=lambda group: str(group.get("group_name"))
        ),
        "python_rng_state": payload["python_rng_state"],
        "numpy_rng_state": payload["numpy_rng_state"],
        "torch_cpu_rng_state": payload["torch_cpu_rng_state"],
        "torch_cuda_rng_state": list(cuda_rng) if cuda_rng is not None else None,
    }


def semantic_state_sha256(payload: dict[str, Any]) -> str:
    """Hash only the exact training state, excluding execution and analysis metadata."""
    return _digest_value(_semantic_components(payload))


def _detail_differences(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    details: list[str] = []
    left_model = left["model_state"]
    right_model = right["model_state"]
    for name in sorted(set(left_model) | set(right_model)):
        if name not in left_model or name not in right_model:
            details.append(f"model_state.{name}: missing from one state")
        elif _digest_value(left_model[name]) != _digest_value(right_model[name]):
            details.append(f"model_state.{name}: value differs")
    left_opt = left["optimizer_state"]["state_by_parameter_name"]
    right_opt = right["optimizer_state"]["state_by_parameter_name"]
    for name in sorted(set(left_opt) | set(right_opt)):
        if name not in left_opt or name not in right_opt:
            details.append(f"optimizer_state.{name}: missing from one state")
        elif _digest_value(left_opt[name]) != _digest_value(right_opt[name]):
            details.append(f"optimizer_state.{name}: value differs")
    return details


def compare_semantic_states(
    left_payload: dict[str, Any], right_payload: dict[str, Any]
) -> SemanticStateComparison:
    """Compare normalized state components and report stable field-level differences."""
    left = _semantic_components(left_payload)
    right = _semantic_components(right_payload)
    left_hash = _digest_value(left)
    right_hash = _digest_value(right)
    components = tuple(
        name for name in left if _digest_value(left[name]) != _digest_value(right[name])
    )
    details = _detail_differences(left, right)
    for name in components:
        if name not in {"model_state", "optimizer_state"}:
            details.append(f"{name}: value differs")
    return SemanticStateComparison(
        equal=left_hash == right_hash,
        left_sha256=left_hash,
        right_sha256=right_hash,
        differing_components=components,
        detail_differences=tuple(details),
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LineageValidationError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise LineageValidationError(f"JSON artifact must contain an object: {path}")
    return payload


def _ordered_runs(run_dirs: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    metadata = {run.name: (run, _read_json(run / "metadata.json")) for run in run_dirs}
    if len(metadata) != len(run_dirs):
        raise LineageValidationError("lineage run IDs must be unique")
    roots = [item for item in metadata.values() if item[1].get("parent_run_id") is None]
    if len(roots) != 1:
        raise LineageValidationError(f"lineage must contain exactly one root, found {len(roots)}")
    ordered = [roots[0]]
    seen = {roots[0][0].name}
    while len(ordered) < len(run_dirs):
        parent_id = ordered[-1][0].name
        children = [
            item
            for run_id, item in metadata.items()
            if run_id not in seen and item[1].get("parent_run_id") == parent_id
        ]
        if len(children) != 1:
            raise LineageValidationError(
                f"lineage requires one child of {parent_id}, found {len(children)}"
            )
        ordered.append(children[0])
        seen.add(children[0][0].name)
    return ordered


def resolve_checkpoint_lineage(
    run_dirs: list[str | Path] | tuple[str | Path, ...],
    *,
    expected_physical_count: int = 503,
    expected_regular_count: int = 501,
    regular_interval: int = 100,
    expected_start_step: int = 0,
    expected_end_step: int = 50_000,
) -> CheckpointLineage:
    """Resolve a complete read-only lineage and verify semantic branch-anchor aliases."""
    if len(run_dirs) < 1:
        raise ValueError("run_dirs must contain at least one run")
    if regular_interval <= 0:
        raise ValueError(f"regular_interval must be positive, got {regular_interval}")
    resolved_runs = [Path(path).resolve() for path in run_dirs]
    ordered = _ordered_runs(resolved_runs)
    physical: list[PhysicalCheckpoint] = []
    segments: list[LineageSegment] = []
    manifest_counts: dict[str, int] = {}
    manifest_steps: dict[str, list[int]] = {}

    for depth, (run_dir, metadata) in enumerate(ordered):
        if metadata.get("run_id") not in {None, run_dir.name}:
            raise LineageValidationError(f"metadata run_id does not match directory {run_dir.name}")
        manifest = _read_json(run_dir / "checkpoints" / "manifest.json")
        entries = manifest.get("checkpoints")
        if manifest.get("schema_version") != 2 or not isinstance(entries, list) or not entries:
            raise LineageValidationError(f"invalid checkpoint manifest: {run_dir}")
        steps: list[int] = []
        seen_paths: set[Path] = set()
        for entry in entries:
            if not isinstance(entry, dict) or type(entry.get("step")) is not int:
                raise LineageValidationError(f"invalid manifest entry in {run_dir}: {entry!r}")
            step = entry["step"]
            relative = entry.get("path")
            if step < 0 or not isinstance(relative, str) or not relative:
                raise LineageValidationError(f"invalid manifest checkpoint in {run_dir}: {entry!r}")
            checkpoint = (run_dir / "checkpoints" / relative).resolve()
            checkpoint_root = (run_dir / "checkpoints").resolve()
            if checkpoint.parent != checkpoint_root or checkpoint.name.startswith("."):
                raise LineageValidationError(
                    f"manifest checkpoint path escapes source run: {relative}"
                )
            if checkpoint in seen_paths or not checkpoint.is_file():
                raise LineageValidationError(
                    f"duplicate or missing manifested checkpoint: {checkpoint}"
                )
            seen_paths.add(checkpoint)
            payload = read_checkpoint(checkpoint, "cpu")
            if payload.get("global_step") != step:
                raise LineageValidationError(
                    f"manifest/payload global step mismatch for {run_dir.name}:{step}"
                )
            scientific_hash = payload.get("scientific_config_hash")
            split_hash = payload.get("split_hash")
            if not isinstance(scientific_hash, str) or not isinstance(split_hash, str):
                raise LineageValidationError(
                    f"checkpoint hashes are missing for {run_dir.name}:{step}"
                )
            if metadata.get("scientific_config_hash") != scientific_hash:
                raise LineageValidationError(
                    f"metadata/checkpoint scientific hash mismatch for {run_dir.name}:{step}"
                )
            if metadata.get("split_hash") != split_hash:
                raise LineageValidationError(
                    f"metadata/checkpoint split hash mismatch for {run_dir.name}:{step}"
                )
            physical.append(
                PhysicalCheckpoint(
                    step=step,
                    run_id=run_dir.name,
                    checkpoint_path=checkpoint,
                    checkpoint_relative_path=(
                        Path("runs") / run_dir.name / "checkpoints" / checkpoint.name
                    ).as_posix(),
                    checkpoint_sha256=file_sha256(checkpoint),
                    semantic_state_sha256=semantic_state_sha256(payload),
                    source_git_commit=metadata.get("git_commit"),
                    scientific_config_hash=scientific_hash,
                    split_hash=split_hash,
                    lineage_depth=depth,
                )
            )
            steps.append(step)
        if steps != sorted(set(steps)):
            raise LineageValidationError(f"manifest steps must be unique and increasing: {run_dir}")
        if any(
            right - left != regular_interval for left, right in zip(steps, steps[1:], strict=False)
        ):
            raise LineageValidationError(f"manifest has a missing/non-regular step in {run_dir}")
        parent_id = metadata.get("parent_run_id")
        if depth:
            expected_parent = ordered[depth - 1][0].name
            if parent_id != expected_parent:
                raise LineageValidationError(
                    f"lineage parent mismatch: {run_dir.name} expected {expected_parent}, "
                    f"got {parent_id}"
                )
            if metadata.get("parent_global_step") != steps[0]:
                raise LineageValidationError(
                    f"child branch anchor does not match first manifest step: {run_dir.name}"
                )
            previous_steps = manifest_steps[expected_parent]
            if previous_steps[-1] != steps[0]:
                raise LineageValidationError(
                    "parent terminal step and child anchor differ: "
                    f"{expected_parent}->{run_dir.name}"
                )
        segments.append(
            LineageSegment(
                run_id=run_dir.name,
                run_dir=run_dir,
                start_step=steps[0],
                end_step=steps[-1],
                lineage_depth=depth,
                parent_run_id=parent_id,
            )
        )
        manifest_counts[run_dir.name] = len(entries)
        manifest_steps[run_dir.name] = steps

    by_step: dict[int, list[PhysicalCheckpoint]] = {}
    for checkpoint in physical:
        by_step.setdefault(checkpoint.step, []).append(checkpoint)
    canonical: list[CanonicalCheckpoint] = []
    aliases: list[AliasCheckpoint] = []
    for step, items in sorted(by_step.items()):
        items.sort(key=lambda item: item.lineage_depth)
        selected = items[0]
        for alias in items[1:]:
            left_payload = read_checkpoint(selected.checkpoint_path, "cpu")
            right_payload = read_checkpoint(alias.checkpoint_path, "cpu")
            comparison = compare_semantic_states(left_payload, right_payload)
            if not comparison.equal:
                diagnostics = {
                    "step": step,
                    "left_run_id": selected.run_id,
                    "right_run_id": alias.run_id,
                    "comparison": {
                        "left_sha256": comparison.left_sha256,
                        "right_sha256": comparison.right_sha256,
                        "differing_components": list(comparison.differing_components),
                        "detail_differences": list(comparison.detail_differences),
                    },
                }
                raise LineageConflictError(
                    f"semantic branch-anchor conflict at step {step}", diagnostics
                )
            aliases.append(
                AliasCheckpoint(
                    step=step,
                    canonical_run_id=selected.run_id,
                    alias_run_id=alias.run_id,
                    canonical_checkpoint_sha256=selected.checkpoint_sha256,
                    alias_checkpoint_sha256=alias.checkpoint_sha256,
                    semantic_state_sha256=comparison.left_sha256,
                    raw_sha256_equal=(selected.checkpoint_sha256 == alias.checkpoint_sha256),
                )
            )
        canonical.append(
            CanonicalCheckpoint(step=step, physical=selected, alias_count=max(0, len(items) - 1))
        )

    expected_steps = list(range(expected_start_step, expected_end_step + 1, regular_interval))
    actual_steps = [item.step for item in canonical]
    missing_steps = sorted(set(expected_steps) - set(actual_steps))
    extra_steps = sorted(set(actual_steps) - set(expected_steps))
    diagnostics = {
        "manifest_counts": manifest_counts,
        "physical_count": len(physical),
        "expected_physical_count": expected_physical_count,
        "regular_step_count": len(canonical),
        "expected_regular_count": expected_regular_count,
        "missing_steps": missing_steps,
        "extra_steps": extra_steps,
    }
    if (
        len(physical) != expected_physical_count
        or len(canonical) != expected_regular_count
        or actual_steps != expected_steps
    ):
        raise LineageValidationError(
            "checkpoint lineage does not match the required grid", diagnostics
        )

    scientific_hashes = {item.scientific_config_hash for item in physical}
    split_hashes = {item.split_hash for item in physical}
    if len(scientific_hashes) != 1 or len(split_hashes) != 1:
        raise LineageValidationError(
            "checkpoint lineage scientific/split hashes are not constant",
            {
                "scientific_config_hashes": sorted(scientific_hashes),
                "split_hashes": sorted(split_hashes),
            },
        )
    return CheckpointLineage(
        physical=tuple(physical),
        canonical=tuple(canonical),
        aliases=tuple(aliases),
        segments=tuple(segments),
        regular_interval=regular_interval,
        expected_start_step=expected_start_step,
        expected_end_step=expected_end_step,
    )
