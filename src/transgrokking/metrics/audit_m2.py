"""Machine-readable analysis audit for M2-A and M2-B."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from transgrokking.analysis.artifacts import (
    finite_tree,
    inventory_tree,
    repository_root,
    sha256_file,
    write_analysis_status,
)
from transgrokking.analysis.config import load_m2_analysis_config
from transgrokking.metrics.function_space import detect_function_events
from transgrokking.utils.atomic import write_json

M2_ANALYSIS_AUDIT_SCHEMA_VERSION = 1
_FORBIDDEN = (
    "fourier",
    "frequency-line",
    "restricted-frequency",
    "e_line",
    "seed_2",
    "seed_3",
    "wd_grid",
    "congruence",
)
_AUDITED_FILES = (
    "analysis_config.resolved.yaml",
    "provenance.json",
    "status.json",
    "checkpoint_files.csv",
    "checkpoint_index.csv",
    "checkpoint_aliases.csv",
    "context/m1_scalars.jsonl",
    "context/m1_optimization.jsonl",
    "context/collapse_episodes.json",
    "context/events.json",
    "m2a/checkpoint_validation.jsonl",
    "m2a/checkpoint_validation.csv",
    "m2a/episode_state_index.csv",
    "m2a/replay_bridge.jsonl",
    "m2b/function_metrics.jsonl",
    "m2b/function_metrics.csv",
    "m2b/offset_profiles.npz",
    "m2b/function_events.json",
    "m2b/episode_function_deltas.csv",
    "selected_tensors/manifest.json",
)
_REQUIRED_FUNCTION_FIELDS = {
    "schema_version",
    "step",
    "state_source",
    "run_id",
    "checkpoint_sha256",
    "semantic_state_sha256",
    "centered_logit_frobenius",
    "centered_logit_rms",
    "equivariant_energy",
    "residual_energy",
    "D_eq",
    "Gamma",
    "I",
    "Gamma_minus_I",
    "Gamma_over_logit_rms",
    "I_over_logit_rms",
    "Gamma_over_parameter_l2",
    "train_projected_ce",
    "test_projected_ce",
    "full_projected_ce",
    "train_projected_accuracy",
    "test_projected_accuracy",
    "full_projected_accuracy",
    "train_entropy_mean",
    "test_entropy_mean",
    "full_entropy_mean",
    "train_entropy_normalized",
    "test_entropy_normalized",
    "full_entropy_normalized",
    "train_accuracy",
    "test_accuracy",
    "full_accuracy",
    "full_cross_entropy",
    "full_margin_mean",
    "full_margin_min",
    "parameter_norm_total",
    "parameter_group_norm_decay",
    "parameter_group_norm_no_decay",
    "is_regular_grid",
    "state_roles",
    "replay_source_step",
    "replay_updates",
    "full_logits_shape",
    "forward_dtype",
    "reduction_dtype",
    "committed_ce_within_tolerance",
    "committed_behavior_alignment_passed",
    "committed_behavior_alignment_status",
    "batched_predictions_match_committed",
    "committed_reference_recheck_passed",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    if not finite_tree(value):
        raise ValueError(f"{path}: JSON contains non-finite or unsupported values")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        if not finite_tree(value):
            raise ValueError(
                f"{path}:{line_number}: JSON contains non-finite or unsupported values"
            )
        records.append(value)
    return records


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_fields(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        return list(reader.fieldnames), list(reader)


def _csv_matches(value: Any, raw: str | None) -> bool:
    if raw is None:
        return False
    if value is None:
        return raw == ""
    if isinstance(value, bool):
        return raw.lower() == str(value).lower()
    if isinstance(value, int):
        try:
            return int(raw) == value
        except ValueError:
            return False
    if isinstance(value, float):
        try:
            return math.isclose(float(raw), value, rel_tol=1.0e-12, abs_tol=1.0e-12)
        except ValueError:
            return False
    if isinstance(value, (list, dict)):
        try:
            return json.loads(raw) == value
        except json.JSONDecodeError:
            return False
    return raw == str(value)


def _jsonl_csv_equal(json_records: list[dict[str, Any]], csv_path: Path) -> bool:
    fields, rows = _csv_fields(csv_path)
    if not json_records or len(rows) != len(json_records):
        return False
    schema = list(json_records[0])
    if fields != schema or any(list(record) != schema for record in json_records):
        return False
    return all(
        all(_csv_matches(value, row.get(field)) for field, value in record.items())
        for record, row in zip(json_records, rows, strict=True)
    )


def _inventory_sources(config, provenance: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    ids = provenance.get("source_run_ids")
    if not isinstance(ids, dict):
        raise ValueError("provenance.source_run_ids must be a mapping")
    expected_ids = config.to_dict()["source_run_ids"]
    if ids != expected_ids:
        raise ValueError("provenance source run IDs differ from the resolved analysis config")
    run_ids = [expected_ids[name] for name in ("root", "canonical_parent", "terminal_child")]
    repo = repository_root()
    return {
        **{f"runs/{run_id}": inventory_tree(repo / "runs" / str(run_id)) for run_id in run_ids},
        config.source_result_dirs.m1_reference: inventory_tree(
            repo / config.source_result_dirs.m1_reference
        ),
        config.source_result_dirs.m1_extended: inventory_tree(
            repo / config.source_result_dirs.m1_extended
        ),
    }


def _audited_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in _AUDITED_FILES:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"required M2 analysis file is missing: {relative}")
        hashes[relative] = sha256_file(path)
    return hashes


def _source_lineage_state(config) -> dict[str, Any]:
    repo = repository_root()
    ids = config.source_run_ids
    ordered = (ids.root, ids.canonical_parent, ids.terminal_child)
    metadata = {run_id: _read_json(repo / "runs" / run_id / "metadata.json") for run_id in ordered}
    statuses = {run_id: _read_json(repo / "runs" / run_id / "status.json") for run_id in ordered}
    parents = {
        ids.root: None,
        ids.canonical_parent: ids.root,
        ids.terminal_child: ids.canonical_parent,
    }
    parent_steps = {ids.root: None, ids.canonical_parent: 5_000, ids.terminal_child: 20_000}
    valid = all(statuses[run_id].get("state") == "completed" for run_id in ordered)
    valid = valid and all(
        metadata[run_id].get("parent_run_id") == parents[run_id]
        and metadata[run_id].get("parent_global_step") == parent_steps[run_id]
        for run_id in ordered
    )
    scientific_hashes = {str(metadata[run_id].get("scientific_config_hash")) for run_id in ordered}
    split_hashes = {str(metadata[run_id].get("split_hash")) for run_id in ordered}
    valid = valid and len(scientific_hashes) == 1 and len(split_hashes) == 1
    return {
        "valid": valid,
        "run_ids": list(ordered),
        "parents": {run_id: metadata[run_id].get("parent_run_id") for run_id in ordered},
        "parent_steps": {run_id: metadata[run_id].get("parent_global_step") for run_id in ordered},
        "states": {run_id: statuses[run_id].get("state") for run_id in ordered},
        "scientific_config_hashes": sorted(scientific_hashes),
        "split_hashes": sorted(split_hashes),
    }


def _check(name: str, passed: bool, details: Any = None) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details}


def _safe(callable_, fallback: Any = None) -> tuple[Any, str | None]:
    try:
        return callable_(), None
    except Exception as error:  # audit must report all readable failures
        return fallback, f"{type(error).__name__}: {error}"


def _expected_implication_status(value: float, tolerance: float) -> str:
    if value > tolerance:
        return "verified"
    if value < -tolerance:
        return "not_applicable"
    return "numerically_ambiguous"


def _bridge_structure_is_valid(
    bridges: list[dict[str, Any]], expected_replay_steps: set[int], config
) -> bool:
    actual_steps = [row.get("target_step") for row in bridges]

    def expected_owner(step: int) -> str:
        if step <= 5_000:
            return config.source_run_ids.root
        if step <= 20_000:
            return config.source_run_ids.canonical_parent
        return config.source_run_ids.terminal_child

    return (
        len(actual_steps) == len(set(actual_steps))
        and set(actual_steps) == expected_replay_steps
        and all(
            type(row.get("target_step")) is int
            and row.get("source_step") == row["target_step"] - 50
            and row.get("endpoint_step") == row["target_step"] + 50
            and row.get("replay_updates") == 50
            and row.get("run_id") == expected_owner(row["target_step"])
            and row.get("schema_version") == M2_ANALYSIS_AUDIT_SCHEMA_VERSION
            and row.get("source_checkpoint_unchanged") is True
            and row.get("endpoint_checkpoint_unchanged") is True
            and isinstance(row.get("midpoint_repeat_sha256"), list)
            and len(row["midpoint_repeat_sha256"]) == 2
            and len(set(row["midpoint_repeat_sha256"])) == 1
            and all(
                isinstance(value, str) and len(value) == 64
                for value in row["midpoint_repeat_sha256"]
            )
            and isinstance(row.get("endpoint_repeat_sha256"), list)
            and len(row["endpoint_repeat_sha256"]) == 2
            and len(set(row["endpoint_repeat_sha256"])) == 1
            and all(
                isinstance(value, str) and len(value) == 64
                for value in row["endpoint_repeat_sha256"]
            )
            and row.get("endpoint_semantic_equal") is True
            and row.get("midpoint_behavior_repeat_equal") is True
            and row.get("endpoint_behavior_equal") is True
            and row.get("endpoint_replay_behavior_sha256")
            == row.get("endpoint_checkpoint_behavior_sha256")
            and isinstance(row.get("endpoint_replay_behavior_sha256"), str)
            and len(row["endpoint_replay_behavior_sha256"]) == 64
            and isinstance(row.get("endpoint_differing_components"), list)
            and len(row["endpoint_differing_components"]) == 2
            and all(not components for components in row["endpoint_differing_components"])
            for row in bridges
        )
    )


def _lineage_table_errors(
    physical: list[dict[str, str]],
    canonical: list[dict[str, str]],
    aliases: list[dict[str, str]],
    current_inventory: dict[str, list[dict[str, Any]]] | None,
    config,
) -> list[str]:
    """Validate the derived lineage tables against the frozen physical inventory."""
    ids = config.source_run_ids
    expected_by_run = {
        ids.root: list(range(0, 5_001, 100)),
        ids.canonical_parent: list(range(5_000, 20_001, 100)),
        ids.terminal_child: list(range(20_000, 50_001, 100)),
    }
    errors: list[str] = []
    physical_by_key: dict[tuple[str, int], dict[str, str]] = {}
    inventory_by_run: dict[str, dict[str, dict[str, Any]]] = {}
    if current_inventory is not None:
        for run_id in expected_by_run:
            inventory_by_run[run_id] = {
                str(item.get("path")): item
                for item in current_inventory.get(f"runs/{run_id}", [])
                if isinstance(item, dict)
            }
    for row in physical:
        try:
            run_id = row["run_id"]
            step = int(row["step"])
        except (KeyError, TypeError, ValueError):
            errors.append("physical_invalid_step_or_run")
            continue
        key = (run_id, step)
        if key in physical_by_key:
            errors.append(f"physical_duplicate:{run_id}:{step}")
            continue
        physical_by_key[key] = row
        expected_path = f"runs/{run_id}/checkpoints/step_{step:06d}.pt"
        if row.get("checkpoint_relative_path") != expected_path:
            errors.append(f"physical_path:{run_id}:{step}")
        inventory = inventory_by_run.get(run_id, {}).get(f"checkpoints/step_{step:06d}.pt")
        if inventory is None or inventory.get("sha256") != row.get("checkpoint_sha256"):
            errors.append(f"physical_sha:{run_id}:{step}")
    expected_keys = {(run_id, step) for run_id, steps in expected_by_run.items() for step in steps}
    if set(physical_by_key) != expected_keys:
        errors.append("physical_segment_grid")

    def owner(step: int) -> str:
        if step <= 5_000:
            return ids.root
        if step <= 20_000:
            return ids.canonical_parent
        return ids.terminal_child

    canonical_steps: list[int] = []
    for row in canonical:
        try:
            step = int(row["step"])
        except (KeyError, TypeError, ValueError):
            errors.append("canonical_invalid_step")
            continue
        canonical_steps.append(step)
        physical_row = physical_by_key.get((owner(step), step))
        if physical_row is None or any(
            row.get(field) != physical_row.get(field)
            for field in (
                "run_id",
                "checkpoint_relative_path",
                "checkpoint_sha256",
                "semantic_state_sha256",
            )
        ):
            errors.append(f"canonical_source:{step}")
        expected_alias_count = "1" if step in {5_000, 20_000} else "0"
        if row.get("alias_count") != expected_alias_count:
            errors.append(f"canonical_alias_count:{step}")
    if canonical_steps != list(range(0, 50_001, 100)):
        errors.append("canonical_regular_grid")

    expected_alias_runs = {
        5_000: (ids.root, ids.canonical_parent),
        20_000: (ids.canonical_parent, ids.terminal_child),
    }
    if len(aliases) != 2:
        errors.append("alias_count")
    for row in aliases:
        try:
            step = int(row["step"])
        except (KeyError, TypeError, ValueError):
            errors.append("alias_invalid_step")
            continue
        run_pair = expected_alias_runs.get(step)
        if run_pair is None:
            errors.append(f"alias_unexpected_step:{step}")
            continue
        left = physical_by_key.get((run_pair[0], step))
        right = physical_by_key.get((run_pair[1], step))
        if (
            left is None
            or right is None
            or row.get("canonical_run_id") != run_pair[0]
            or row.get("alias_run_id") != run_pair[1]
            or row.get("canonical_checkpoint_sha256") != left.get("checkpoint_sha256")
            or row.get("alias_checkpoint_sha256") != right.get("checkpoint_sha256")
            or row.get("semantic_state_sha256") != left.get("semantic_state_sha256")
            or row.get("semantic_state_sha256") != right.get("semantic_state_sha256")
            or left.get("checkpoint_sha256") == right.get("checkpoint_sha256")
            or row.get("semantic_state_equal", "").lower() not in {"true", "1"}
            or row.get("raw_sha256_equal", "").lower() not in {"false", "0"}
        ):
            errors.append(f"alias_identity:{step}")
    return errors


def _brute_force_gamma_interference(
    centered: np.ndarray, residual: np.ndarray
) -> tuple[float, float]:
    """Independently recompute Gamma and I with explicit modular loops."""
    if centered.shape != residual.shape or centered.ndim != 3:
        raise ValueError("selected centered/residual logits require one shared [p,p,p] shape")
    modulus = centered.shape[0]
    if centered.shape != (modulus, modulus, modulus):
        raise ValueError("selected logits must have equal dimensions")
    profile = np.empty(modulus, dtype=np.float64)
    for offset in range(modulus):
        values: list[float] = []
        for left in range(modulus):
            for right in range(modulus):
                candidate = (left + right + offset) % modulus
                values.append(float(centered[left, right, candidate]))
        profile[offset] = math.fsum(values) / (modulus * modulus)
    gamma = float(profile[0] - max(profile[1:]))
    interference = float("-inf")
    for left in range(modulus):
        for right in range(modulus):
            correct = (left + right) % modulus
            correct_value = float(residual[left, right, correct])
            for candidate in range(modulus):
                if candidate != correct:
                    interference = max(
                        interference,
                        float(residual[left, right, candidate]) - correct_value,
                    )
    return gamma, interference


def audit_m2_analysis(analysis_dir: str | Path) -> dict[str, Any]:
    """Audit completed local M2 computation and atomically persist the result."""
    root = Path(analysis_dir).resolve()
    expected_parent = (repository_root() / "analysis_runs").resolve()
    if root.parent != expected_parent:
        raise ValueError(f"M2 analysis directory must be a direct child of {expected_parent}")
    config = load_m2_analysis_config(root / "analysis_config.resolved.yaml")
    checks: list[dict[str, Any]] = []

    status, error = _safe(lambda: _read_json(root / "status.json"), {})
    checks.append(
        _check(
            "stage_status",
            error is None
            and status.get("m2a_status") == "completed"
            and status.get("m2b_status") == "completed"
            and status.get("analysis_status") == "completed",
            error or status,
        )
    )

    provenance, error = _safe(lambda: _read_json(root / "provenance.json"), {})
    before = provenance.get("source_inventories_before") if error is None else None
    after = provenance.get("source_inventories_after") if error is None else None
    current_inventory, current_inventory_error = _safe(
        lambda: _inventory_sources(config, provenance), None
    )
    checks.append(
        _check(
            "source_inventories_unchanged",
            before is not None
            and before == after
            and current_inventory_error is None
            and current_inventory == before,
            error or current_inventory_error,
        )
    )
    checks.append(
        _check(
            "analysis_identity",
            provenance.get("analysis_id") == root.name
            and status.get("analysis_id") == root.name
            and provenance.get("analysis_config_hash") == config.analysis_hash()
            and status.get("analysis_config_hash") == config.analysis_hash()
            and provenance.get("source_run_ids") == config.to_dict()["source_run_ids"]
            and root.name.endswith(config.analysis_hash()[:8]),
            {
                "analysis_id": provenance.get("analysis_id"),
                "config_hash": provenance.get("analysis_config_hash"),
            },
        )
    )
    source_lineage, source_lineage_error = _safe(lambda: _source_lineage_state(config), {})
    checks.append(
        _check(
            "source_lineage_completed",
            source_lineage_error is None
            and source_lineage.get("valid") is True
            and source_lineage.get("scientific_config_hashes")
            == [provenance.get("scientific_config_hash")]
            and source_lineage.get("split_hashes") == [provenance.get("split_hash")],
            source_lineage_error or source_lineage,
        )
    )
    hardware = provenance.get("hardware", {})
    checks.append(
        _check(
            "formal_hardware_and_peak_vram",
            isinstance(hardware, dict)
            and hardware.get("cuda_available") is True
            and hardware.get("device_name") == config.expected_device
            and type(hardware.get("total_vram_bytes")) is int
            and hardware["total_vram_bytes"] >= int(config.expected_vram_gb * 1024**3 * 0.99)
            and type(status.get("max_memory_allocated")) is int
            and status["max_memory_allocated"] > 0
            and type(status.get("max_memory_reserved")) is int
            and status["max_memory_reserved"] > 0,
            {
                "device_name": hardware.get("device_name") if isinstance(hardware, dict) else None,
                "total_vram_bytes": hardware.get("total_vram_bytes")
                if isinstance(hardware, dict)
                else None,
                "max_memory_allocated": status.get("max_memory_allocated"),
                "max_memory_reserved": status.get("max_memory_reserved"),
            },
        )
    )
    frozen_events, frozen_event_error = _safe(
        lambda: _read_json(root / "context" / "events.json"), {}
    )
    expected_behavior_events = {"t_fit": 100, "t_grok50": 6050, "t_grok99": 7000}
    checks.append(
        _check(
            "frozen_m1_behavior_events",
            frozen_event_error is None
            and provenance.get("behavior_events") == expected_behavior_events
            and all(
                isinstance(frozen_events.get(name), dict)
                and frozen_events[name].get("event_step") == step
                for name, step in expected_behavior_events.items()
            )
            and frozen_events.get("last_evaluated_step") == 50_000,
            frozen_event_error or frozen_events,
        )
    )

    physical, physical_error = _safe(lambda: _read_csv(root / "checkpoint_files.csv"), [])
    canonical, canonical_error = _safe(lambda: _read_csv(root / "checkpoint_index.csv"), [])
    aliases, aliases_error = _safe(lambda: _read_csv(root / "checkpoint_aliases.csv"), [])
    checks.append(
        _check(
            "checkpoint_counts",
            physical_error is None
            and canonical_error is None
            and len(physical) == 503
            and len(canonical) == 501,
            {
                "physical": len(physical),
                "canonical": len(canonical),
                "errors": [physical_error, canonical_error],
            },
        )
    )
    canonical_steps = [int(row["step"]) for row in canonical] if canonical_error is None else []
    physical_paths = [row.get("checkpoint_relative_path") for row in physical]
    physical_counts: dict[str, int] = {}
    for row in physical:
        physical_counts[str(row.get("run_id"))] = physical_counts.get(str(row.get("run_id")), 0) + 1
    expected_counts = {
        config.source_run_ids.root: 51,
        config.source_run_ids.canonical_parent: 151,
        config.source_run_ids.terminal_child: 301,
    }
    checks.append(
        _check(
            "physical_checkpoint_schema",
            len(physical_paths) == len(set(physical_paths))
            and physical_counts == expected_counts
            and all(
                row.get("checkpoint_relative_path", "").startswith("runs/")
                and len(row.get("checkpoint_sha256", "")) == 64
                and len(row.get("semantic_state_sha256", "")) == 64
                and row.get("scientific_config_hash") == provenance.get("scientific_config_hash")
                and row.get("split_hash") == provenance.get("split_hash")
                and bool(row.get("source_git_commit"))
                for row in physical
            ),
            {"manifest_counts": physical_counts, "expected": expected_counts},
        )
    )
    checks.append(
        _check(
            "regular_checkpoint_grid",
            canonical_steps == list(range(0, 50_001, 100)),
            {"first": canonical_steps[:3], "last": canonical_steps[-3:]},
        )
    )
    alias_ok = aliases_error is None and len(aliases) == 2
    if alias_ok:
        alias_ok = {int(row["step"]) for row in aliases} == {5000, 20000} and all(
            row.get("semantic_state_equal", "").lower() in {"true", "1"}
            and row.get("raw_sha256_equal", "").lower() in {"false", "0"}
            and row.get("alias_group_id")
            for row in aliases
        )
    checks.append(_check("branch_anchor_aliases", alias_ok, aliases_error or aliases))
    lineage_table_errors = _lineage_table_errors(
        physical,
        canonical,
        aliases,
        current_inventory if isinstance(current_inventory, dict) else None,
        config,
    )
    checks.append(
        _check(
            "lineage_tables_match_frozen_sources",
            physical_error is None
            and canonical_error is None
            and aliases_error is None
            and not lineage_table_errors,
            lineage_table_errors,
        )
    )

    validations, validation_error = _safe(
        lambda: _read_jsonl(root / "m2a" / "checkpoint_validation.jsonl"), []
    )
    validation_csv, validation_csv_error = _safe(
        lambda: _read_csv(root / "m2a" / "checkpoint_validation.csv"), []
    )
    unresolved = [
        row
        for row in validations
        if row.get("validation_status") != "passed" or row.get("resolution") == "unresolved"
    ]
    validation_detail_errors: list[dict[str, Any]] = []
    for row in validations:
        step = row.get("target_step")
        recomputed = row.get("recomputed_metrics")
        committed = row.get("committed_metrics")
        differences = row.get("absolute_differences")
        if (
            type(step) is not int
            or not isinstance(recomputed, dict)
            or not isinstance(differences, dict)
            or len(str(row.get("checkpoint_sha256", ""))) != 64
            or len(str(row.get("semantic_state_sha256", ""))) != 64
            or row.get("model_code_commit") != provenance.get("implementation_git_commit")
            or not finite_tree(row)
        ):
            validation_detail_errors.append({"step": step, "reason": "record_identity"})
            continue
        for field in (
            "full_cross_entropy",
            "full_accuracy",
            "full_margin_mean",
            "full_margin_min",
            "parameter_norm_total",
            "parameter_group_norm_decay",
            "parameter_group_norm_no_decay",
        ):
            if field not in recomputed:
                validation_detail_errors.append(
                    {"step": step, "reason": f"missing_recomputed_{field}"}
                )
        if step == 0:
            if row.get("committed_available") is not False or committed is not None:
                validation_detail_errors.append({"step": step, "reason": "step0_commit"})
            continue
        if (
            row.get("committed_available") is not True
            or not isinstance(committed, dict)
            or row.get("error_offsets_match") is not True
        ):
            validation_detail_errors.append({"step": step, "reason": "commit_alignment"})
            continue
        for split in ("train", "test"):
            for suffix in ("accuracy", "error_count"):
                field = f"{split}_{suffix}"
                if recomputed.get(field) != committed.get(field) or differences.get(field) != 0.0:
                    validation_detail_errors.append({"step": step, "reason": f"exact_{field}"})
    checks.append(
        _check(
            "m2a_behavior_validation",
            validation_error is None
            and bool(validations)
            and not unresolved
            and not validation_detail_errors,
            validation_error
            or {
                "records": len(validations),
                "failed": len(unresolved),
                "detail_errors": validation_detail_errors[:20],
            },
        )
    )
    checks.append(
        _check(
            "m2a_jsonl_csv_alignment",
            validation_csv_error is None
            and len(validation_csv) == len(validations)
            and [int(row["target_step"]) for row in validation_csv]
            == [int(row["target_step"]) for row in validations]
            and all(
                csv_row.get("validation_status") == json_row.get("validation_status")
                and csv_row.get("semantic_state_sha256") == json_row.get("semantic_state_sha256")
                for csv_row, json_row in zip(validation_csv, validations, strict=True)
            ),
            validation_csv_error,
        )
    )
    roles, roles_error = _safe(lambda: _read_csv(root / "m2a" / "episode_state_index.csv"), [])
    required_episode_ids = {
        *(f"test_{index:03d}" for index in range(1, 11)),
        *(f"joint_{index:03d}" for index in range(1, 11)),
        "train_001",
        "train_015",
        "train_024",
        "train_026",
    }
    represented = {row.get("episode_id") for row in roles}
    terminal_roles = {
        row.get("episode_id") for row in roles if row.get("state_role") == "terminal_unrecovered"
    }
    checks.append(
        _check(
            "required_episode_roles",
            roles_error is None
            and required_episode_ids.issubset(represented)
            and {"test_010", "train_026"}.issubset(terminal_roles),
            roles_error
            or {
                "missing": sorted(required_episode_ids - represented),
                "terminal_unrecovered": sorted(item for item in terminal_roles if item),
            },
        )
    )
    roles_by_episode: dict[str, set[str]] = {}
    for row in roles:
        roles_by_episode.setdefault(str(row.get("episode_id")), set()).add(
            str(row.get("state_role"))
        )
    episode_role_errors: list[str] = []
    recovered_primitive_roles = {
        "pre_collapse",
        "onset",
        "train_trough",
        "train_recovery_start",
        "train_recovery_confirmed",
        "post_recovery",
    }
    for train_id in ("train_001", "train_015", "train_024"):
        if not recovered_primitive_roles.issubset(roles_by_episode.get(train_id, set())):
            episode_role_errors.append(train_id)
    required_terminal_train = {
        "pre_collapse",
        "onset",
        "train_trough",
        "terminal_unrecovered",
    }
    if not required_terminal_train.issubset(roles_by_episode.get("train_026", set())):
        episode_role_errors.append("train_026")
    for index in range(1, 11):
        test_id = f"test_{index:03d}"
        joint_id = f"joint_{index:03d}"
        required_test = {"pre_collapse", "onset", "test_trough"}
        required_joint = {"pre_collapse", "onset", "train_trough", "test_trough"}
        if index == 10:
            required_test.add("terminal_unrecovered")
            required_joint.add("terminal_unrecovered")
        else:
            required_test.update(
                {"test_recovery_start", "test_recovery_confirmed", "post_recovery"}
            )
            required_joint.update(
                {
                    "train_recovery_start",
                    "train_recovery_confirmed",
                    "test_recovery_start",
                    "test_recovery_confirmed",
                    "post_recovery",
                }
            )
        if not required_test.issubset(roles_by_episode.get(test_id, set())):
            episode_role_errors.append(test_id)
        if not required_joint.issubset(roles_by_episode.get(joint_id, set())):
            episode_role_errors.append(joint_id)
    terminal_rows = [row for row in roles if row.get("state_role") == "terminal_unrecovered"]
    if any(
        row.get("episode_status") != "not_recovered"
        or row.get("recovery_start") not in {None, ""}
        or row.get("recovery_confirmed") not in {None, ""}
        or row.get("target_step") != "50000"
        for row in terminal_rows
    ):
        episode_role_errors.append("terminal_unrecovered_fields")
    test_008_steps = {
        int(row["target_step"]) for row in roles if row.get("episode_id") == "test_008"
    }
    if not any(step < 20_000 for step in test_008_steps) or not any(
        step > 20_000 for step in test_008_steps
    ):
        episode_role_errors.append("test_008_cross_lineage")
    validation_steps = {
        int(row["target_step"]) for row in validations if type(row.get("target_step")) is int
    }
    role_steps = {int(row["target_step"]) for row in roles}
    if validation_steps != role_steps:
        episode_role_errors.append("episode_validation_target_alignment")
    checks.append(_check("episode_role_completeness", not episode_role_errors, episode_role_errors))
    bridges, bridge_error = _safe(lambda: _read_jsonl(root / "m2a" / "replay_bridge.jsonl"), [])
    bridge_failed = [row for row in bridges if row.get("validation_status") != "passed"]
    expected_replay_steps = {
        int(row["target_step"])
        for row in validations
        if row.get("resolution") == "deterministic_replay"
    }
    bridge_structure = _bridge_structure_is_valid(bridges, expected_replay_steps, config)
    checks.append(
        _check(
            "deterministic_replay_bridges",
            bridge_error is None and bool(bridges) and not bridge_failed and bridge_structure,
            bridge_error or {"bridges": len(bridges), "failed": len(bridge_failed)},
        )
    )

    metrics, metric_error = _safe(lambda: _read_jsonl(root / "m2b" / "function_metrics.jsonl"), [])
    metric_csv_equal, metric_csv_error = _safe(
        lambda: _jsonl_csv_equal(metrics, root / "m2b" / "function_metrics.csv"), False
    )
    metric_steps = [row.get("step") for row in metrics]
    regular = [row for row in metrics if row.get("is_regular_grid") is True]
    regular_steps = [row.get("step") for row in regular]
    checks.append(
        _check(
            "function_metric_timeline",
            metric_error is None
            and bool(metrics)
            and all(type(step) is int and step >= 0 for step in metric_steps)
            and metric_steps == sorted(set(metric_steps))
            and regular_steps == list(range(0, 50_001, 100))
            and all(finite_tree(row) for row in metrics),
            metric_error
            or {
                "records": len(metrics),
                "regular_records": len(regular),
                "unique": len(set(metric_steps)),
            },
        )
    )
    schema_fields = set(metrics[0]) if metrics else set()
    checks.append(
        _check(
            "function_metric_schema_and_csv",
            metric_csv_error is None
            and metric_csv_equal is True
            and _REQUIRED_FUNCTION_FIELDS.issubset(schema_fields)
            and all(set(row) == schema_fields for row in metrics),
            metric_csv_error
            or {"missing_fields": sorted(_REQUIRED_FUNCTION_FIELDS - schema_fields)},
        )
    )
    tolerance = config.math_tolerances
    invariant_limits = {
        "centering_max_abs": tolerance.centering_atol,
        "offset_profile_sum_abs": tolerance.centering_atol,
        "reconstruction_relative_error": tolerance.reconstruction_rtol,
        "orthogonality_normalized_error": tolerance.orthogonality_normalized,
        "energy_identity_relative_error": tolerance.energy_identity_rtol,
        "projection_idempotence_relative_error": tolerance.reconstruction_rtol,
        "residual_projection_relative_error": tolerance.reconstruction_rtol,
        "group_invariance_relative_error": tolerance.invariance_rtol,
    }
    failures: list[dict[str, Any]] = []
    canonical_by_step = {int(row["step"]): row for row in canonical}
    validation_by_step = {int(row["target_step"]): row for row in validations}
    for row in metrics:
        step = row.get("step")
        if row.get("is_regular_grid") is True:
            source = canonical_by_step.get(int(step)) if type(step) is int else None
            if (
                source is None
                or row.get("state_source") != "exact_checkpoint"
                or row.get("run_id") != source.get("run_id")
                or row.get("checkpoint_sha256") != source.get("checkpoint_sha256")
                or row.get("semantic_state_sha256") != source.get("semantic_state_sha256")
                or row.get("replay_source_step") is not None
                or row.get("replay_updates") != 0
            ):
                failures.append({"step": step, "field": "regular_state_identity"})
        else:
            validation = validation_by_step.get(int(step)) if type(step) is int else None
            if (
                validation is None
                or row.get("state_source") != "deterministic_replay"
                or row.get("run_id") != validation.get("run_id")
                or row.get("checkpoint_sha256") != validation.get("checkpoint_sha256")
                or row.get("semantic_state_sha256") != validation.get("semantic_state_sha256")
                or row.get("replay_source_step") != validation.get("source_checkpoint_step")
                or row.get("replay_updates") != 50
            ):
                failures.append({"step": step, "field": "replay_state_identity"})
        validation = validation_by_step.get(int(step)) if type(step) is int else None
        if validation is not None and row.get("state_roles") != validation.get("state_roles"):
            failures.append({"step": step, "field": "state_roles"})
        if (
            row.get("full_logits_shape") != [97, 97, 97]
            or row.get("forward_dtype") != "float32"
            or row.get("reduction_dtype") != "float64"
        ):
            failures.append(
                {
                    "step": row.get("step"),
                    "field": "full_logits_shape_or_dtype",
                    "shape": row.get("full_logits_shape"),
                    "forward_dtype": row.get("forward_dtype"),
                    "reduction_dtype": row.get("reduction_dtype"),
                }
            )
        for field, limit in invariant_limits.items():
            value = row.get(field)
            if type(value) not in {int, float} or not 0.0 <= float(value) <= limit:
                failures.append(
                    {
                        "step": row.get("step"),
                        "field": field,
                        "value": value,
                        "limit": limit,
                    }
                )
        allowed_statuses = {"verified", "not_applicable", "numerically_ambiguous"}
        for field in ("raw_implication_status", "projected_implication_status"):
            if row.get(field) not in allowed_statuses:
                failures.append({"step": row.get("step"), "field": field, "value": row.get(field)})
        gamma = row.get("Gamma")
        interference = row.get("I")
        if type(gamma) in {int, float} and type(interference) in {int, float}:
            expected_raw = _expected_implication_status(
                float(gamma) - float(interference), tolerance.implication_margin_atol
            )
            expected_projected = _expected_implication_status(
                float(gamma), tolerance.implication_margin_atol
            )
            if row.get("raw_implication_status") != expected_raw:
                failures.append(
                    {
                        "step": row.get("step"),
                        "field": "raw_implication_status",
                        "value": row.get("raw_implication_status"),
                        "expected": expected_raw,
                    }
                )
            if row.get("projected_implication_status") != expected_projected:
                failures.append(
                    {
                        "step": row.get("step"),
                        "field": "projected_implication_status",
                        "value": row.get("projected_implication_status"),
                        "expected": expected_projected,
                    }
                )
        else:
            failures.append({"step": row.get("step"), "field": "Gamma_or_I"})
        d_eq = row.get("D_eq")
        if d_eq is not None and (
            type(d_eq) not in {int, float} or not -1.0e-12 <= float(d_eq) <= 1.0 + 1.0e-12
        ):
            failures.append({"step": row.get("step"), "field": "D_eq", "value": d_eq})
        projected_diffs = [
            row.get("projected_split_ce_max_abs_diff"),
            row.get("projected_split_accuracy_max_abs_diff"),
            row.get("projected_split_margin_max_abs_diff"),
        ]
        if any(
            type(value) not in {int, float} or not 0.0 <= float(value) <= tolerance.invariance_rtol
            for value in projected_diffs
        ):
            failures.append({"step": row.get("step"), "field": "projected_split_diffs"})
        if row.get("projected_split_invariants_passed") is not True:
            failures.append({"step": row.get("step"), "field": "projected_split_invariants_passed"})
        if row.get("raw_implication_status") == "verified" and row.get("full_accuracy") != 1.0:
            failures.append({"step": row.get("step"), "field": "raw_implication"})
        if (
            row.get("projected_implication_status") == "verified"
            and row.get("full_projected_accuracy") != 1.0
        ):
            failures.append({"step": row.get("step"), "field": "projected_implication"})
        if row.get("invariants_passed") is not True:
            failures.append({"step": row.get("step"), "field": "invariants_passed"})
        if row.get("committed_behavior_alignment_passed") is not True:
            failures.append(
                {"step": row.get("step"), "field": "committed_behavior_alignment_passed"}
            )
        committed_diff = row.get("committed_ce_max_abs_diff")
        if row.get("step") == 0:
            if (
                committed_diff is not None
                or row.get("committed_behavior_alignment_status") != "uncommitted_initialization"
                or row.get("batched_predictions_match_committed") is not None
                or row.get("committed_reference_recheck_passed") is not None
                or row.get("committed_ce_within_tolerance") is not True
            ):
                failures.append({"step": row.get("step"), "field": "initialization_alignment"})
        else:
            count_differences = [
                row.get("committed_train_error_count_diff"),
                row.get("committed_test_error_count_diff"),
            ]
            accuracy_differences = [
                row.get("committed_train_accuracy_abs_diff"),
                row.get("committed_test_accuracy_abs_diff"),
            ]
            if (
                type(committed_diff) not in {int, float}
                or not 0.0 <= float(committed_diff) <= config.behavior_validation_atol
                or row.get("committed_ce_within_tolerance") is not True
                or any(type(value) is not int for value in count_differences)
                or any(
                    type(value) not in {int, float} or not math.isfinite(float(value))
                    for value in accuracy_differences
                )
            ):
                failures.append(
                    {
                        "step": row.get("step"),
                        "field": "committed_numeric_alignment",
                        "value": committed_diff,
                    }
                )
            predictions_match = row.get("batched_predictions_match_committed")
            if predictions_match is True:
                if (
                    row.get("committed_behavior_alignment_status") != "prediction_exact"
                    or row.get("committed_reference_recheck_passed") is not None
                    or any(value != 0 for value in count_differences)
                ):
                    failures.append(
                        {"step": row.get("step"), "field": "prediction_exact_alignment"}
                    )
            elif predictions_match is False:
                if (
                    row.get("committed_behavior_alignment_status") != "batch_sensitive_predictions"
                    or row.get("committed_reference_recheck_passed") is not True
                    or not any(value != 0 for value in count_differences)
                ):
                    failures.append({"step": row.get("step"), "field": "batch_sensitive_alignment"})
            else:
                failures.append(
                    {"step": row.get("step"), "field": "batched_predictions_match_committed"}
                )
    checks.append(
        _check("function_space_invariants", not failures and bool(metrics), failures[:20])
    )

    profiles, profile_error = _safe(
        lambda: np.load(root / "m2b" / "offset_profiles.npz", allow_pickle=False), None
    )
    profile_ok = False
    profile_details: Any = profile_error
    if profiles is not None:
        try:
            profile_steps = profiles["steps"]
            values = profiles["offset_profiles"]
            profile_ok = (
                profile_steps.tolist() == metric_steps
                and values.shape == (len(metrics), 97)
                and np.issubdtype(profile_steps.dtype, np.integer)
                and values.dtype == np.float64
                and np.isfinite(values).all()
            )
            profile_details = {"steps": int(profile_steps.size), "shape": list(values.shape)}
        finally:
            profiles.close()
    checks.append(_check("offset_profiles", profile_ok, profile_details))

    events, event_error = _safe(lambda: _read_json(root / "m2b" / "function_events.json"), {})
    rebuilt_events, rebuilt_event_error = _safe(
        lambda: detect_function_events(metrics, expected_interval=100), {}
    )
    checks.append(
        _check(
            "function_events",
            event_error is None
            and rebuilt_event_error is None
            and events == rebuilt_events
            and events.get("summary_grid") == "regular_manifest_checkpoint_grid"
            and events.get("regular_step_count") == 501
            and events.get("event_resolution_steps") == 100
            and finite_tree(events),
            event_error or rebuilt_event_error or events,
        )
    )
    tensor_manifest, tensor_error = _safe(
        lambda: _read_json(root / "selected_tensors" / "manifest.json"), {}
    )
    tensor_entries = tensor_manifest.get("tensors", []) if tensor_error is None else []
    tensor_steps = {row.get("step") for row in tensor_entries if isinstance(row, dict)}
    required_tensor_steps = set(config.selected_tensor_steps)
    for event_name in ("t_alg", "t_dom"):
        event = events.get(event_name, {}) if isinstance(events, dict) else {}
        if type(event.get("event_step")) is int:
            required_tensor_steps.add(int(event["event_step"]))
    tensor_files_ok = True
    metric_by_step = {int(row["step"]): row for row in metrics}
    spot_steps = {0, 50_000}
    spot_steps.update(
        int(events[name]["event_step"])
        for name in ("t_alg", "t_dom")
        if isinstance(events.get(name), dict) and type(events[name].get("event_step")) is int
    )
    spot_differences: list[dict[str, Any]] = []
    for entry in tensor_entries:
        if not isinstance(entry, dict):
            tensor_files_ok = False
            continue
        relative = entry.get("path")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or "\\" in relative
        ):
            tensor_files_ok = False
            continue
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            tensor_files_ok = False
            continue
        if (
            not path.is_file()
            or entry.get("size") != path.stat().st_size
            or entry.get("sha256") != sha256_file(path)
        ):
            tensor_files_ok = False
            continue
        try:
            with np.load(path, allow_pickle=False) as archive:
                required_arrays = {
                    "step",
                    "raw_logits",
                    "centered_logits",
                    "offset_profile",
                    "projected_logits",
                    "residual_logits",
                }
                if not required_arrays.issubset(archive.files):
                    tensor_files_ok = False
                    continue
                stored_step = int(np.asarray(archive["step"]).item())
                raw = np.asarray(archive["raw_logits"])
                centered = np.asarray(archive["centered_logits"])
                profile = np.asarray(archive["offset_profile"])
                projected = np.asarray(archive["projected_logits"])
                residual = np.asarray(archive["residual_logits"])
                if (
                    stored_step != entry.get("step")
                    or raw.shape != (97, 97, 97)
                    or centered.shape != raw.shape
                    or projected.shape != raw.shape
                    or residual.shape != raw.shape
                    or profile.shape != (97,)
                    or raw.dtype != np.float32
                    or any(
                        value.dtype != np.float64
                        for value in (centered, profile, projected, residual)
                    )
                    or not all(
                        np.isfinite(value).all()
                        for value in (raw, centered, profile, projected, residual)
                    )
                ):
                    tensor_files_ok = False
                    continue
                if stored_step in spot_steps:
                    gamma, interference = _brute_force_gamma_interference(centered, residual)
                    metric = metric_by_step[stored_step]
                    spot_differences.append(
                        {
                            "step": stored_step,
                            "Gamma_abs_diff": abs(gamma - float(metric["Gamma"])),
                            "I_abs_diff": abs(interference - float(metric["I"])),
                        }
                    )
        except (OSError, ValueError, KeyError):
            tensor_files_ok = False
    checks.append(
        _check(
            "selected_tensor_manifest",
            tensor_error is None
            and isinstance(tensor_entries, list)
            and bool(tensor_entries)
            and required_tensor_steps.issubset(tensor_steps)
            and tensor_files_ok,
            tensor_error or {"missing_steps": sorted(required_tensor_steps - tensor_steps)},
        )
    )
    checks.append(
        _check(
            "gamma_i_bruteforce_spot_check",
            {row["step"] for row in spot_differences} == spot_steps
            and all(
                row["Gamma_abs_diff"] <= tolerance.invariance_rtol
                and row["I_abs_diff"] <= tolerance.invariance_rtol
                for row in spot_differences
            ),
            spot_differences,
        )
    )
    forbidden = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and any(token in path.relative_to(root).as_posix().lower() for token in _FORBIDDEN)
    ]
    checks.append(_check("no_m3_or_later_artifacts", not forbidden, forbidden))

    passed = all(check["passed"] for check in checks)
    write_analysis_status(
        root,
        analysis_audit_passed=passed,
        export_status="ready" if passed else "blocked",
    )
    audited_hashes, audited_hash_error = _safe(lambda: _audited_hashes(root), {})
    checks.append(_check("audited_source_hashes", audited_hash_error is None, audited_hash_error))
    if audited_hash_error is not None:
        passed = False
        write_analysis_status(root, analysis_audit_passed=False, export_status="blocked")
    result = {
        "schema_version": M2_ANALYSIS_AUDIT_SCHEMA_VERSION,
        "analysis_id": root.name,
        "analysis_config_hash": config.analysis_hash(),
        "passed": passed,
        "checks": checks,
        "audited_source_sha256": audited_hashes,
    }
    write_json(root / "audit" / "m2_analysis.json", result)
    return result
