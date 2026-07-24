"""Machine-readable audit for a terminal M1-C CE-reference extension."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from transgrokking.config import ExperimentConfig, load_config
from transgrokking.metrics.evaluator import evaluate_run_checkpoint
from transgrokking.metrics.stability import (
    load_measurement_config,
    summarize_stability,
)
from transgrokking.training.artifacts import (
    load_error_offset_records,
    load_manifest,
    load_optimization_records,
    load_scalar_records,
)
from transgrokking.training.checkpoint import CHECKPOINT_SCHEMA_VERSION, read_checkpoint
from transgrokking.utils.atomic import write_json

M1C_AUDIT_SCHEMA_VERSION = 1
FINAL_GLOBAL_STEP = 50_000
DIAGNOSTICS_START_STEP = 20_000
EXPECTED_EVALUATION_INTERVAL = 50
EXPECTED_CHECKPOINT_INTERVAL = 100
EXPECTED_FROZEN_FILE_COUNT = 23
_REGISTERED_MEASUREMENT = Path("configs/analysis/m1c_stability.yaml")
_DEFAULT_FROZEN_MANIFEST = Path("configs/analysis/m1b_frozen_manifest.json")
_OPTIMIZATION_FIELDS = {
    "parameter_tensor_count",
    "parameter_element_count",
    "updated_parameter_tensor_count",
    "updated_parameter_element_count",
    "gradient_l2_total",
    "gradient_l2_decay_group",
    "gradient_l2_no_decay_group",
    "total_update_l2",
    "data_update_l2",
    "decay_update_l2",
    "data_to_decay_ratio",
    "data_decay_cosine",
    "decay_group_total_update_l2",
    "no_decay_group_total_update_l2",
    "adam_first_moment_l2",
    "adam_first_moment_l2_decay_group",
    "adam_first_moment_l2_no_decay_group",
    "adam_second_moment_mean",
    "adam_second_moment_rms",
    "adam_second_moment_max",
}
_REQUIRED_SCALAR_FIELDS = {
    "congruence_loss",
    "train_cross_entropy",
    "test_cross_entropy",
    "train_accuracy",
    "test_accuracy",
    "parameter_norm_total",
    "parameter_group_norm_decay",
    "parameter_group_norm_no_decay",
}
_FORBIDDEN_PATH_TOKENS = {
    "reynolds",
    "fourier",
    "function_space",
    "centered_logits",
    "d_eq",
    "gamma",
    "l_parallel",
    "t_alg",
    "t_dom",
}
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (_repository_root() / candidate).resolve()
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_finite_tree(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if type(value) in {int, float}:
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_is_finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_finite_tree(item) for key, item in value.items())
    return False


def _close(actual: object, expected: object) -> bool:
    if actual is None or expected is None:
        return actual is expected
    if type(actual) in {int, float} and type(expected) in {int, float}:
        return math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-7)
    return actual == expected


def _load_lineage(root: Path) -> list[Path]:
    """Return the complete oldest-to-terminal run lineage."""
    chain = [root]
    seen = {root.resolve()}
    current = root
    while True:
        metadata = _read_json(current / "metadata.json")
        parent_run_id = metadata.get("parent_run_id")
        if parent_run_id is None:
            return list(reversed(chain))
        if not isinstance(parent_run_id, str) or not parent_run_id:
            raise ValueError(f"{current}: invalid parent_run_id {parent_run_id!r}")
        parent = (current.parent / parent_run_id).resolve()
        if parent in seen or not parent.is_dir():
            raise ValueError(f"invalid parent run lineage at {current}")
        chain.append(parent)
        seen.add(parent)
        current = parent


def _same_except_max_steps(
    canonical: ExperimentConfig,
    extension: ExperimentConfig,
) -> bool:
    canonical_dict = canonical.to_dict()
    extension_dict = extension.to_dict()
    canonical_dict["optimization"]["max_steps"] = extension_dict["optimization"]["max_steps"]
    return canonical_dict == extension_dict


def _metadata_group_signature(metadata: dict[str, Any]) -> list[dict[str, object]] | None:
    groups = metadata.get("optimizer_parameter_groups")
    if not isinstance(groups, list):
        return None
    signature: list[dict[str, object]] = []
    seen_names: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            return None
        group_name = group.get("group_name")
        parameter_names = group.get("parameter_names")
        learning_rate = group.get("learning_rate")
        weight_decay = group.get("weight_decay")
        if (
            not isinstance(group_name, str)
            or not isinstance(parameter_names, list)
            or any(not isinstance(name, str) or not name for name in parameter_names)
            or any(name in seen_names for name in parameter_names)
            or type(learning_rate) not in {int, float}
            or type(weight_decay) not in {int, float}
        ):
            return None
        seen_names.update(parameter_names)
        signature.append(
            {
                "group_name": group_name,
                "parameter_names": parameter_names,
                "weight_decay": weight_decay,
                "learning_rate": learning_rate,
            }
        )
    return signature


def _frozen_manifest_check(
    manifest_path: Path,
) -> tuple[bool, dict[str, Any]]:
    errors: list[str] = []
    try:
        manifest = _read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return False, {"manifest": str(manifest_path), "errors": [str(error)]}
    entries = manifest.get("files")
    root_value = manifest.get("root")
    if manifest.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    if not isinstance(entries, list):
        return False, {"manifest": str(manifest_path), "errors": [*errors, "files must be a list"]}
    if len(entries) != EXPECTED_FROZEN_FILE_COUNT:
        errors.append(f"expected {EXPECTED_FROZEN_FILE_COUNT} frozen files, found {len(entries)}")
    if not isinstance(root_value, str) or not root_value:
        return False, {"manifest": str(manifest_path), "errors": [*errors, "root is invalid"]}
    frozen_root = Path(root_value)
    if not frozen_root.is_absolute():
        frozen_root = _repository_root() / frozen_root
    frozen_root = frozen_root.resolve()
    expected_paths: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"files[{index}] is not an object")
            continue
        relative = entry.get("path")
        size = entry.get("size")
        expected_hash = entry.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            errors.append(f"files[{index}].path is invalid")
            continue
        normalized = Path(relative).as_posix()
        if normalized in expected_paths:
            errors.append(f"duplicate frozen path: {normalized}")
            continue
        expected_paths.add(normalized)
        source = (frozen_root / relative).resolve()
        try:
            source.relative_to(frozen_root)
        except ValueError:
            errors.append(f"frozen path escapes root: {relative}")
            continue
        if not source.is_file():
            errors.append(f"missing frozen file: {normalized}")
            continue
        if type(size) is not int or source.stat().st_size != size:
            errors.append(f"frozen file size mismatch: {normalized}")
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or _sha256(source) != expected_hash
        ):
            errors.append(f"frozen file SHA-256 mismatch: {normalized}")
    actual_paths = {
        path.relative_to(frozen_root).as_posix()
        for path in frozen_root.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        errors.append(f"frozen file set mismatch: missing={missing}, extra={extra}")
    return not errors, {
        "manifest": str(manifest_path),
        "root": str(frozen_root),
        "expected_file_count": EXPECTED_FROZEN_FILE_COUNT,
        "actual_file_count": len(actual_paths),
        "errors": errors,
    }


def _event_matches(
    events: dict[str, Any],
    name: str,
    event_step: int,
    detected_at: int,
) -> bool:
    event = events.get(name)
    return (
        isinstance(event, dict)
        and event.get("status") == "reached"
        and event.get("event_step") == event_step
        and event.get("detected_at_evaluation_step") == detected_at
    )


def _offline_evaluation_matches(
    evaluation: dict[str, Any],
    scalar: dict[str, Any],
    final_offsets: list[dict[str, Any]],
) -> bool:
    if evaluation.get("step") != FINAL_GLOBAL_STEP:
        return False
    comparable = {
        key: value
        for key, value in evaluation.items()
        if key in scalar and key not in {"schema_version", "step"}
    }
    if not comparable or not all(_close(scalar[key], value) for key, value in comparable.items()):
        return False
    evaluated_offsets = evaluation.get("error_offsets")
    if not isinstance(evaluated_offsets, dict) or len(final_offsets) != 2:
        return False
    return all(
        evaluated_offsets.get(split) == final_offsets[index].get("counts")
        for index, split in enumerate(("train", "test"))
    )


def _forbidden_artifacts(root: Path) -> list[str]:
    forbidden: set[str] = set()
    for directory_name in ("tensors", "figures"):
        directory = root / directory_name
        if directory.is_dir():
            forbidden.update(
                path.relative_to(root).as_posix() for path in directory.rglob("*") if path.is_file()
            )
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        lowered = relative.lower()
        if any(token in lowered for token in _FORBIDDEN_PATH_TOKENS):
            forbidden.add(relative)
    return sorted(forbidden)


def _checkpoint_checks(
    extension_lineage: list[Path],
    *,
    scientific_config_hash: str,
    split_hash: str,
    group_signature: list[dict[str, object]],
) -> tuple[bool, dict[str, Any]]:
    errors: list[str] = []
    all_steps: set[int] = set()
    checkpoint_count = 0
    for run in extension_lineage:
        metadata = _read_json(run / "metadata.json")
        manifest = load_manifest(run)
        steps = [int(entry["step"]) for entry in manifest]
        anchor = metadata.get("parent_global_step")
        if not steps or type(anchor) is not int or steps[0] != anchor:
            errors.append(f"{run.name}: manifest does not begin at its branch anchor")
        if any(
            current - previous != EXPECTED_CHECKPOINT_INTERVAL
            for previous, current in zip(steps, steps[1:], strict=False)
        ):
            errors.append(f"{run.name}: manifest is not on the 100-step grid")
        all_steps.update(steps)
        for entry in manifest:
            step = int(entry["step"])
            payload = read_checkpoint(run / "checkpoints" / str(entry["path"]))
            checkpoint_count += 1
            if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
                errors.append(f"{run.name}:{step}: checkpoint schema mismatch")
            if payload.get("global_step") != step:
                errors.append(f"{run.name}:{step}: checkpoint global_step mismatch")
            if payload.get("scientific_config_hash") != scientific_config_hash:
                errors.append(f"{run.name}:{step}: scientific config hash mismatch")
            if payload.get("split_hash") != split_hash:
                errors.append(f"{run.name}:{step}: split hash mismatch")
            if payload.get("optimizer_group_signature") != group_signature:
                errors.append(f"{run.name}:{step}: optimizer group signature mismatch")
            if payload.get("optimizer_type") != "adamw":
                errors.append(f"{run.name}:{step}: optimizer type mismatch")
    expected_steps = set(
        range(DIAGNOSTICS_START_STEP, FINAL_GLOBAL_STEP + 1, EXPECTED_CHECKPOINT_INTERVAL)
    )
    if all_steps != expected_steps:
        errors.append("checkpoint lineage does not cover the exact 20000..50000 grid")
    if (
        not extension_lineage
        or int(load_manifest(extension_lineage[-1])[-1]["step"]) != FINAL_GLOBAL_STEP
    ):
        errors.append("terminal manifest does not end at step 50000")
    return not errors, {
        "checkpoint_count_across_lineage": checkpoint_count,
        "unique_checkpoint_steps": len(all_steps),
        "errors": errors,
    }


def _audited_source_hashes(root: Path) -> dict[str, str]:
    """Hash every source file consumed by the reproducible M1-C export."""
    manifest = load_manifest(root)
    if not manifest:
        raise ValueError("terminal run has no manifested checkpoint")
    final_checkpoint = root / "checkpoints" / str(manifest[-1]["path"])
    paths = [
        root / "config.resolved.yaml",
        root / "measurement.resolved.yaml",
        root / "metadata.json",
        root / "status.json",
        root / "metrics" / "scalars.jsonl",
        root / "metrics" / "error_offsets.jsonl",
        root / "metrics" / "events.json",
        root / "metrics" / "stability.json",
        root / "metrics" / "collapse_episodes.json",
        root / "metrics" / "optimization.jsonl",
        root / "checkpoints" / "manifest.json",
        final_checkpoint,
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"audited M1-C source files are missing: {missing}")
    return {path.relative_to(root).as_posix(): _sha256(path) for path in paths}


def audit_m1c_extension(
    run_dir: str | Path,
    frozen_manifest_path: str | Path = _DEFAULT_FROZEN_MANIFEST,
) -> dict[str, Any]:
    """Audit a terminal 50000-step M1-C child and atomically persist the report."""
    root = Path(run_dir).resolve()
    config = load_config(root / "config.resolved.yaml")
    measurement = load_measurement_config(root / "measurement.resolved.yaml")
    registered_measurement = load_measurement_config(
        _resolve_repository_path(_REGISTERED_MEASUREMENT)
    )
    metadata = _read_json(root / "metadata.json")
    status = _read_json(root / "status.json")
    events = _read_json(root / "metrics" / "events.json")
    stability = _read_json(root / "metrics" / "stability.json")
    collapse_episodes = _read_json(root / "metrics" / "collapse_episodes.json")
    scalars_path = root / "metrics" / "scalars.jsonl"
    scalars = load_scalar_records(scalars_path)
    offsets = load_error_offset_records(root / "metrics" / "error_offsets.jsonl")
    optimization = load_optimization_records(root / "metrics" / "optimization.jsonl")
    lineage = _load_lineage(root)

    source = measurement.source
    canonical_matches = [
        index for index, run in enumerate(lineage) if run.name == source.canonical_run_id
    ]
    canonical_index = canonical_matches[0] if len(canonical_matches) == 1 else -1
    canonical = (
        lineage[canonical_index]
        if canonical_index >= 0
        else (root.parent / source.canonical_run_id).resolve()
    )
    extension_lineage = lineage[canonical_index + 1 :] if canonical_index >= 0 else []
    lineage_metadata = [_read_json(path / "metadata.json") for path in extension_lineage]
    canonical_config = (
        load_config(canonical / "config.resolved.yaml") if canonical.is_dir() else None
    )

    scalar_steps = [int(record["step"]) for record in scalars]
    expected_scalar_steps = list(
        range(EXPECTED_EVALUATION_INTERVAL, FINAL_GLOBAL_STEP + 1, EXPECTED_EVALUATION_INTERVAL)
    )
    offset_steps = [int(offsets[index]["step"]) for index in range(0, len(offsets), 2)]
    optimization_steps = [int(record["step"]) for record in optimization]
    expected_optimization_steps = list(
        range(
            DIAGNOSTICS_START_STEP + EXPECTED_EVALUATION_INTERVAL,
            FINAL_GLOBAL_STEP + 1,
            EXPECTED_EVALUATION_INTERVAL,
        )
    )

    terminal_signature = _metadata_group_signature(metadata)
    canonical_metadata = _read_json(canonical / "metadata.json") if canonical.is_dir() else {}
    canonical_signature = _metadata_group_signature(canonical_metadata)
    canonical_prefix_ok = False
    if canonical.is_dir():
        canonical_scalars = load_scalar_records(canonical / "metrics" / "scalars.jsonl")
        canonical_offsets = load_error_offset_records(canonical / "metrics" / "error_offsets.jsonl")
        canonical_prefix_ok = [
            record for record in scalars if int(record["step"]) <= DIAGNOSTICS_START_STEP
        ] == canonical_scalars and [
            record for record in offsets if int(record["step"]) <= DIAGNOSTICS_START_STEP
        ] == canonical_offsets
    groups_valid = (
        terminal_signature is not None
        and terminal_signature == canonical_signature
        and [group["group_name"] for group in terminal_signature] == ["decay", "no_decay"]
        and [group["weight_decay"] for group in terminal_signature]
        == [config.optimization.weight_decay, 0.0]
        and [group["learning_rate"] for group in terminal_signature]
        == [config.optimization.learning_rate, config.optimization.learning_rate]
    )

    parent_links_ok = bool(extension_lineage)
    for parent, _child, child_metadata in zip(
        lineage[canonical_index:-1] if canonical_index >= 0 else [],
        extension_lineage,
        lineage_metadata,
        strict=False,
    ):
        parent_entries = load_manifest(parent)
        parent_checkpoint = Path(str(child_metadata.get("parent_checkpoint", "")))
        manifested = {
            (parent / "checkpoints" / str(entry["path"])).resolve(): int(entry["step"])
            for entry in parent_entries
        }
        parent_links_ok = parent_links_ok and (
            child_metadata.get("parent_run_id") == parent.name
            and parent_checkpoint.resolve() in manifested
            and manifested.get(parent_checkpoint.resolve())
            == child_metadata.get("parent_global_step")
            and bool(parent_entries)
            and int(parent_entries[-1]["step"]) == child_metadata.get("parent_global_step")
        )

    canonical_source_ok = False
    canonical_checkpoint_sha256: str | None = None
    if canonical.is_dir() and canonical_signature is not None:
        canonical_manifest = load_manifest(canonical)
        if canonical_manifest:
            canonical_entry = canonical_manifest[-1]
            canonical_checkpoint = (
                canonical / "checkpoints" / str(canonical_entry["path"])
            ).resolve()
            canonical_payload = read_checkpoint(canonical_checkpoint)
            canonical_checkpoint_sha256 = _sha256(canonical_checkpoint)
            canonical_source_ok = (
                int(canonical_entry["step"]) == source.canonical_checkpoint_step
                and canonical_checkpoint_sha256 == source.canonical_checkpoint_sha256
                and canonical_payload.get("schema_version") == CHECKPOINT_SCHEMA_VERSION
                and canonical_payload.get("global_step") == source.canonical_checkpoint_step
                and canonical_payload.get("scientific_config_hash") == source.scientific_config_hash
                and canonical_payload.get("split_hash") == source.split_hash
                and canonical_payload.get("optimizer_group_signature") == canonical_signature
                and canonical_payload.get("optimizer_type") == "adamw"
                and bool(extension_lineage)
                and Path(str(lineage_metadata[0].get("extension_origin_checkpoint", ""))).resolve()
                == canonical_checkpoint
                and all(
                    Path(str(item.get("extension_origin_checkpoint", ""))).resolve()
                    == canonical_checkpoint
                    for item in lineage_metadata
                )
            )

    config_only_max_steps = (
        canonical_config is not None
        and config.optimization.max_steps == FINAL_GLOBAL_STEP
        and canonical_config.optimization.max_steps == DIAGNOSTICS_START_STEP
        and all(
            _same_except_max_steps(canonical_config, load_config(path / "config.resolved.yaml"))
            and load_config(path / "config.resolved.yaml").optimization.max_steps
            == FINAL_GLOBAL_STEP
            for path in extension_lineage
        )
    )
    measurement_hash = measurement.measurement_hash()
    measurement_identity = (
        measurement.to_dict() == registered_measurement.to_dict()
        and source.canonical_checkpoint_step == DIAGNOSTICS_START_STEP
        and source.eval_interval == EXPECTED_EVALUATION_INTERVAL
        and source.checkpoint_interval == EXPECTED_CHECKPOINT_INTERVAL
        and all(
            item.get("measurement_config_hash") == measurement_hash
            and item.get("extension_origin_run_id") == source.canonical_run_id
            and item.get("extension_origin_step") == DIAGNOSTICS_START_STEP
            and item.get("diagnostics_start_step") == DIAGNOSTICS_START_STEP
            for item in lineage_metadata
        )
    )

    lineage_hashes_ok = (
        source.scientific_config_hash == config.scientific_hash()
        and canonical_metadata.get("scientific_config_hash") == source.scientific_config_hash
        and all(
            item.get("scientific_config_hash") == source.scientific_config_hash
            for item in lineage_metadata
        )
    )
    lineage_split_ok = canonical_metadata.get("split_hash") == source.split_hash and all(
        item.get("split_hash") == source.split_hash for item in lineage_metadata
    )

    checkpoint_ok = False
    checkpoint_details: dict[str, Any] = {"errors": ["optimizer group signature unavailable"]}
    if terminal_signature is not None and extension_lineage:
        checkpoint_ok, checkpoint_details = _checkpoint_checks(
            extension_lineage,
            scientific_config_hash=source.scientific_config_hash,
            split_hash=source.split_hash,
            group_signature=terminal_signature,
        )

    scalar_schema_ok = bool(scalars) and all(
        _REQUIRED_SCALAR_FIELDS.issubset(record)
        and record.get("congruence_loss") == 0.0
        and _is_finite_tree(record)
        for record in scalars
    )
    offsets_ok = offset_steps == expected_scalar_steps and all(
        train.get("modulus") == config.task.modulus and test.get("modulus") == config.task.modulus
        for train, test in zip(offsets[::2], offsets[1::2], strict=True)
    )
    optimization_ok = optimization_steps == expected_optimization_steps and all(
        _OPTIMIZATION_FIELDS.issubset(record) and _is_finite_tree(record) for record in optimization
    )

    frozen = measurement.frozen_events
    frozen_events_ok = (
        events.get("last_evaluated_step") == FINAL_GLOBAL_STEP
        and _event_matches(events, "t_fit", frozen.t_fit, frozen.t_fit_detected_at)
        and _event_matches(events, "t_grok50", frozen.t_grok50, frozen.t_grok50_detected_at)
        and _event_matches(events, "t_grok99", frozen.t_grok99, frozen.t_grok99_detected_at)
    )
    scalar_hash = _sha256(scalars_path)
    expected_stability, expected_episodes = summarize_stability(
        scalars,
        run_id=root.name,
        parent_run_id=metadata.get("parent_run_id"),
        eval_interval=source.eval_interval,
        frozen_events=frozen,
        config=measurement.stability,
        source_scalars_sha256=scalar_hash,
    )
    expected_stability["measurement_config_hash"] = measurement_hash
    expected_episodes["measurement_config_hash"] = measurement_hash
    stability_ok = stability == expected_stability and collapse_episodes == expected_episodes

    final_offsets = offsets[-2:] if len(offsets) >= 2 else []
    evaluation = evaluate_run_checkpoint(root)
    evaluator_ok = _offline_evaluation_matches(evaluation, scalars[-1], final_offsets)
    frozen_ok, frozen_details = _frozen_manifest_check(
        _resolve_repository_path(frozen_manifest_path)
    )
    forbidden_files = _forbidden_artifacts(root)

    status_step = status.get("global_step")
    metadata_step = metadata.get("final_global_step")
    peak_allocated = metadata.get("max_memory_allocated")
    peak_reserved = metadata.get("max_memory_reserved")
    checks = {
        "terminal_completed": status.get("state") == "completed"
        and status_step == FINAL_GLOBAL_STEP
        and metadata_step == FINAL_GLOBAL_STEP,
        "canonical_lineage": len(canonical_matches) == 1
        and canonical_index < len(lineage) - 1
        and parent_links_ok,
        "canonical_source_checkpoint": canonical_source_ok,
        "canonical_metric_prefix": canonical_prefix_ok,
        "config_only_max_steps_changed": config_only_max_steps,
        "measurement_identity": measurement_identity,
        "scientific_config_hash": lineage_hashes_ok,
        "split_hash": lineage_split_ok,
        "optimizer_parameter_groups": groups_valid,
        "formal_gpu_fp32_ce_only": metadata.get("formal_run") is True
        and config.hardware.formal_run
        and config.optimization.device == "cuda:0"
        and metadata.get("doctor", {}).get("device_name") == "NVIDIA GeForce RTX 4060 Laptop GPU"
        and config.optimization.precision == "fp32"
        and not config.optimization.allow_tf32
        and not config.optimization.use_amp
        and config.loss.cross_entropy_weight == 1.0
        and config.loss.congruence_weight == 0.0,
        "scalar_timeline": scalar_steps == expected_scalar_steps and scalar_schema_ok,
        "error_offset_timeline": offsets_ok,
        "optimization_timeline": optimization_ok,
        "frozen_first_events": frozen_events_ok,
        "stability_recomputed": stability_ok,
        "checkpoint_lineage": checkpoint_ok,
        "offline_evaluator": evaluator_ok,
        "peak_vram": type(peak_allocated) is int
        and type(peak_reserved) is int
        and peak_allocated > 0
        and peak_reserved > 0
        and status.get("max_memory_allocated") == peak_allocated
        and status.get("max_memory_reserved") == peak_reserved,
        "git_provenance": bool(lineage_metadata)
        and all(
            isinstance(item.get("git_commit"), str)
            and _GIT_COMMIT_PATTERN.fullmatch(str(item["git_commit"])) is not None
            and item.get("git_worktree_clean") is True
            for item in lineage_metadata
        ),
        "frozen_m1b_results": frozen_ok,
        "no_m2_plus_artifacts": not forbidden_files,
    }
    report = {
        "schema_version": M1C_AUDIT_SCHEMA_VERSION,
        "profile": "m1c-extension",
        "run_id": root.name,
        "canonical_run_id": source.canonical_run_id,
        "canonical_parent_run_id": source.canonical_run_id,
        "lineage": [path.name for path in lineage],
        "extension_lineage": [path.name for path in extension_lineage],
        "scientific_config_hash": source.scientific_config_hash,
        "split_hash": source.split_hash,
        "measurement_config_hash": measurement_hash,
        "canonical_checkpoint_sha256": canonical_checkpoint_sha256,
        "parameter_group_signature": terminal_signature,
        "audited_source_sha256": _audited_source_hashes(root),
        "evaluation_interval": source.eval_interval,
        "checkpoint_interval": source.checkpoint_interval,
        "lineage_git_commits": [
            {
                "run_id": path.name,
                "git_commit": _read_json(path / "metadata.json").get("git_commit"),
            }
            for path in lineage
        ],
        "final_global_step": status_step,
        "evaluation_count": len(scalars),
        "optimization_diagnostic_count": len(optimization),
        "t_fit": frozen.t_fit,
        "t_grok50": frozen.t_grok50,
        "t_grok99": frozen.t_grok99,
        "t_stable99": stability.get("t_stable99"),
        "collapse_count_train": stability.get("collapse_count_train"),
        "collapse_count_test": stability.get("collapse_count_test"),
        "collapse_count_joint": stability.get("collapse_count_joint"),
        "last_collapse_step": stability.get("last_collapse_step"),
        "final_state": stability.get("final_state"),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "checks": checks,
        "checkpoint_details": checkpoint_details,
        "frozen_manifest_details": frozen_details,
        "forbidden_files": forbidden_files,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "passed": all(checks.values()),
    }
    write_json(root / "audit" / "m1c_extension.json", report)
    return report
