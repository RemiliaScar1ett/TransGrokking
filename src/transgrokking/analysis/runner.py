"""Orchestration for the gated, read-only M2-A/M2-B formal analysis."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from transgrokking.analysis.artifacts import (
    create_analysis_dir,
    finite_tree,
    inventory_tree,
    repository_relative,
    repository_root,
    sha256_file,
    write_analysis_status,
    write_csv,
)
from transgrokking.analysis.checkpoint_resolver import (
    CheckpointLineage,
    PhysicalCheckpoint,
    read_checkpoint,
    resolve_checkpoint_lineage,
    semantic_state_sha256,
)
from transgrokking.analysis.config import (
    M2AnalysisConfig,
    dump_m2_analysis_config,
)
from transgrokking.analysis.evaluator import behavior_snapshot, full_table_logits
from transgrokking.analysis.replay import (
    EpisodeSelection,
    ReplayBridgeResult,
    replay_checkpoint_bridge,
    select_episode_states,
)
from transgrokking.config import ExperimentConfig, load_config
from transgrokking.data import ModularAdditionData, generate_modular_addition
from transgrokking.metrics.function_space import (
    MathTolerances,
    detect_function_events,
    function_space_metrics,
)
from transgrokking.metrics.norms import parameter_norm_metrics
from transgrokking.training.artifacts import (
    load_error_offset_records,
    load_scalar_records,
)
from transgrokking.training.optimizer import build_adamw
from transgrokking.training.trainer import build_model
from transgrokking.utils.atomic import torch_save, write_json, write_json_lines
from transgrokking.utils.doctor import collect_doctor_report, validate_doctor_report
from transgrokking.utils.reproducibility import configure_reproducibility

M2_ANALYSIS_OUTPUT_SCHEMA_VERSION = 1
_TRAIN_CONTROLS = ("train_001", "train_015", "train_024", "train_026")
_EXTRA_STATE_ROLES: dict[int, tuple[str, ...]] = {
    0: ("protocol:initialization",),
    100: ("event:t_fit",),
    300: ("event:t_fit_detected_at",),
    5050: ("lineage:m1b_first_replay",),
    5850: ("protocol:pre_grokking_instability",),
    6050: ("event:t_grok50",),
    6150: ("event:t_grok50_detected_at",),
    7000: ("event:t_grok99",),
    7100: ("event:t_grok99_detected_at",),
    11000: ("protocol:high_performance_control",),
    20000: ("lineage:m1b_m1c_anchor",),
    20050: ("lineage:m1c_first_replay",),
    50000: ("protocol:terminal",),
}


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root(), text=True
    ).strip()


def _tracked_worktree_clean() -> bool:
    output = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repository_root(),
        text=True,
    )
    return not output.strip()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _inventory_sources(
    run_dirs: dict[str, Path], result_dirs: dict[str, Path]
) -> dict[str, list[dict[str, Any]]]:
    return {
        **{f"runs/{run_id}": inventory_tree(path) for run_id, path in sorted(run_dirs.items())},
        **{
            repository_relative(path): inventory_tree(path)
            for _, path in sorted(result_dirs.items())
        },
    }


def _write_lineage_tables(root: Path, lineage: CheckpointLineage) -> None:
    physical = [item.to_record() for item in lineage.physical]
    canonical = [item.to_record() for item in lineage.canonical]
    aliases = [item.to_record() for item in lineage.aliases]
    write_csv(root / "checkpoint_files.csv", physical, list(physical[0]))
    write_csv(root / "checkpoint_index.csv", canonical, list(canonical[0]))
    write_csv(root / "checkpoint_aliases.csv", aliases, list(aliases[0]))


def _copy_context(terminal_run: Path, output: Path) -> None:
    copies = {
        "m1_scalars.jsonl": terminal_run / "metrics" / "scalars.jsonl",
        "m1_optimization.jsonl": terminal_run / "metrics" / "optimization.jsonl",
        "collapse_episodes.json": terminal_run / "metrics" / "collapse_episodes.json",
        "events.json": terminal_run / "metrics" / "events.json",
    }
    for name, source in copies.items():
        if not source.is_file():
            raise ValueError(f"M2 context source is missing: {source}")
        shutil.copyfile(source, output / "context" / name)


def _validate_source_identity(
    config: M2AnalysisConfig,
    run_dirs: dict[str, Path],
) -> tuple[ExperimentConfig, ModularAdditionData, dict[str, Any]]:
    terminal = run_dirs[config.source_run_ids.terminal_child]
    experiment = load_config(terminal / "config.resolved.yaml")
    if experiment.task.modulus != 97 or experiment.logging.checkpoint_interval != 100:
        raise ValueError("M2 source is not the frozen p=97/checkpoint_interval=100 trajectory")
    if experiment.optimization.max_steps != 50_000:
        raise ValueError("M2 terminal source must have max_steps=50000")
    data = generate_modular_addition(
        experiment.task.modulus,
        experiment.task.train_fraction,
        experiment.task.split_seed,
    )
    scientific_hashes: set[str] = set()
    split_hashes: set[str] = set()
    git_commits: dict[str, str | None] = {}
    for run_id, run_dir in run_dirs.items():
        metadata = _read_json(run_dir / "metadata.json")
        status = _read_json(run_dir / "status.json")
        if status.get("state") != "completed":
            raise ValueError(f"M2 source run is not completed: {run_id}")
        scientific_hashes.add(str(metadata.get("scientific_config_hash")))
        split_hashes.add(str(metadata.get("split_hash")))
        git_commits[run_id] = metadata.get("git_commit")
        split = torch.load(run_dir / "split.pt", map_location="cpu", weights_only=False)
        if split.get("split_hash") != data.split_hash:
            raise ValueError(f"source split hash mismatch: {run_id}")
        if not torch.equal(split.get("train_indices"), data.train_indices) or not torch.equal(
            split.get("test_indices"), data.test_indices
        ):
            raise ValueError(f"source split indices mismatch: {run_id}")
    if scientific_hashes != {experiment.scientific_hash()} or split_hashes != {data.split_hash}:
        raise ValueError("M2 lineage scientific or split hash is not constant")
    return experiment, data, {"lineage_git_commits": git_commits}


def _episode_selection(terminal_run: Path) -> tuple[EpisodeSelection, dict[str, Any]]:
    scalars = load_scalar_records(terminal_run / "metrics" / "scalars.jsonl")
    collapse = _read_json(terminal_run / "metrics" / "collapse_episodes.json")
    selection = select_episode_states(
        collapse,
        [int(row["step"]) for row in scalars],
        selected_train_episode_ids=_TRAIN_CONTROLS,
        extra_state_roles=_EXTRA_STATE_ROLES,
        terminal_step=50_000,
    )
    return selection, collapse


def _physical_state_for_step(
    lineage: CheckpointLineage, step: int
) -> tuple[PhysicalCheckpoint, int | None, int]:
    if step % lineage.regular_interval == 0:
        return lineage.checkpoint_at(step).physical, None, 0
    segment = lineage.segment_for_replay_target(step)
    source_step = step - 50
    return lineage.physical_checkpoint(segment.run_id, source_step), source_step, 50


def _model_from_payload(payload: dict[str, Any], config: ExperimentConfig, device: torch.device):
    model = build_model(config).to(device=device, dtype=torch.float32)
    _, grouping = build_adamw(model, config.optimization)
    model.load_state_dict(payload["model_state"], strict=True)
    return model, grouping


def _committed_offsets(terminal_run: Path) -> dict[int, dict[str, list[int]]]:
    records = load_error_offset_records(terminal_run / "metrics" / "error_offsets.jsonl")
    output: dict[int, dict[str, list[int]]] = {}
    for record in records:
        output.setdefault(int(record["step"]), {})[str(record["split"])] = record["counts"]
    return output


def _metric_differences(
    committed: dict[str, Any], recomputed: dict[str, Any]
) -> tuple[dict[str, float | None], bool]:
    differences: dict[str, float | None] = {}
    exact = True
    for field, expected in committed.items():
        if field in {"schema_version", "step"}:
            continue
        actual = recomputed.get(field)
        if expected is None or actual is None:
            differences[field] = None
            exact = exact and expected is actual
        elif type(expected) in {int, float} and type(actual) in {int, float}:
            differences[field] = abs(float(actual) - float(expected))
        else:
            differences[field] = None
            exact = exact and actual == expected
    return differences, exact


def _behavior_validation_passed(
    committed: dict[str, Any],
    recomputed: dict[str, Any],
    offsets_expected: dict[str, list[int]],
    offsets_actual: dict[str, list[int]],
    config: M2AnalysisConfig,
) -> tuple[bool, dict[str, float | None]]:
    differences, structural_equal = _metric_differences(committed, recomputed)
    passed = structural_equal
    for field, expected in committed.items():
        if field in {"schema_version", "step"}:
            continue
        actual = recomputed.get(field)
        if field.endswith("_accuracy") or field.endswith("_error_count"):
            passed = passed and actual == expected
        elif type(expected) in {int, float} and type(actual) in {int, float}:
            passed = passed and math.isclose(
                float(actual),
                float(expected),
                abs_tol=config.behavior_validation_atol,
                rel_tol=config.behavior_validation_rtol,
            )
        else:
            passed = passed and actual == expected
    passed = passed and all(
        offsets_actual.get(split) == offsets_expected.get(split) for split in ("train", "test")
    )
    return passed, differences


def _bridge_record(result: ReplayBridgeResult, run_id: str) -> dict[str, Any]:
    endpoint_replay_digest = hashlib.sha256(
        json.dumps(
            result.endpoint.behavior,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    endpoint_physical_digest = hashlib.sha256(
        json.dumps(
            result.physical_endpoint_behavior,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": M2_ANALYSIS_OUTPUT_SCHEMA_VERSION,
        "run_id": run_id,
        "source_checkpoint": repository_relative(result.source_checkpoint),
        "endpoint_checkpoint": repository_relative(result.endpoint_checkpoint),
        "source_checkpoint_sha256": result.source_checkpoint_sha256,
        "endpoint_checkpoint_sha256": result.endpoint_checkpoint_sha256,
        "source_checkpoint_unchanged": result.source_checkpoint_unchanged,
        "endpoint_checkpoint_unchanged": result.endpoint_checkpoint_unchanged,
        "source_step": result.source_step,
        "target_step": result.midpoint_step,
        "endpoint_step": result.endpoint_step,
        "replay_updates": result.replay_updates,
        "midpoint_repeat_sha256": list(result.midpoint_repeat_sha256),
        "endpoint_repeat_sha256": list(result.endpoint_repeat_sha256),
        "endpoint_semantic_equal": all(item.equal for item in result.endpoint_comparisons),
        "midpoint_behavior_repeat_equal": True,
        "endpoint_behavior_equal": result.endpoint.behavior == result.physical_endpoint_behavior,
        "endpoint_replay_behavior_sha256": endpoint_replay_digest,
        "endpoint_checkpoint_behavior_sha256": endpoint_physical_digest,
        "endpoint_differing_components": [
            list(item.differing_components) for item in result.endpoint_comparisons
        ],
        "validation_status": "passed",
    }


def _run_replay_bridges(
    root: Path,
    lineage: CheckpointLineage,
    selection: EpisodeSelection,
    experiment: ExperimentConfig,
    config: M2AnalysisConfig,
    scratch: Path | None = None,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    staging = root / "cache" if scratch is None else scratch
    replay_payloads: dict[int, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for target_step in selection.target_steps:
        if target_step % lineage.regular_interval == 0:
            continue
        segment = lineage.segment_for_replay_target(target_step)
        source_step = target_step - 50
        endpoint_step = target_step + 50
        source = lineage.physical_checkpoint(segment.run_id, source_step)
        endpoint = lineage.physical_checkpoint(segment.run_id, endpoint_step)
        bridge = replay_checkpoint_bridge(
            source.checkpoint_path,
            endpoint.checkpoint_path,
            experiment,
            target_step,
            device=config.device,
            repeats=2,
        )
        replay_payloads[target_step] = bridge.midpoint.checkpoint_payload
        cache_path = root / "cache" / f"replay_step_{target_step:06d}.pt"
        staged_cache = staging / f".{cache_path.name}.staged"
        torch_save(staged_cache, bridge.midpoint.checkpoint_payload)
        os.replace(staged_cache, cache_path)
        records.append(_bridge_record(bridge, segment.run_id))
        write_json_lines(root / "m2a" / "replay_bridge.jsonl", records)
    return replay_payloads, records


def _state_payload(
    root: Path,
    lineage: CheckpointLineage,
    step: int,
    replay_payloads: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], PhysicalCheckpoint, str, int | None, int]:
    source, replay_source, replay_updates = _physical_state_for_step(lineage, step)
    if replay_source is None:
        return read_checkpoint(source.checkpoint_path, "cpu"), source, "exact_checkpoint", None, 0
    payload = replay_payloads.get(step)
    if payload is None:
        cache = root / "cache" / f"replay_step_{step:06d}.pt"
        if not cache.is_file():
            raise ValueError(f"verified replay payload is missing for step {step}")
        payload = torch.load(cache, map_location="cpu", weights_only=False)
    return payload, source, "deterministic_replay", replay_source, replay_updates


def _active_test_episode_ids(collapse: dict[str, Any], step: int) -> list[str]:
    active: list[str] = []
    for episode in collapse["test_episodes"]:
        end = episode.get("test_recovery_confirmed_step")
        if end is None:
            end = 50_000
        if int(episode["onset_step"]) <= step <= int(end):
            active.append(str(episode["episode_id"]))
    return active


def _episode_role_rows(
    selection: EpisodeSelection,
    collapse: dict[str, Any],
    lineage: CheckpointLineage,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reference in selection.references:
        _, replay_source, replay_updates = _physical_state_for_step(lineage, reference.target_step)
        row = reference.to_record()
        row.update(
            {
                "resolution": (
                    "exact_checkpoint" if replay_source is None else "deterministic_replay"
                ),
                "replay_source_step": replay_source,
                "replay_updates": replay_updates,
                "active_test_episode_ids": (
                    _active_test_episode_ids(collapse, reference.target_step)
                    if reference.episode_type == "train"
                    else []
                ),
                "recovery_start": None
                if reference.state_role == "terminal_unrecovered"
                else "not_applicable",
                "recovery_confirmed": None
                if reference.state_role == "terminal_unrecovered"
                else "not_applicable",
            }
        )
        rows.append(row)
    for step, roles in _EXTRA_STATE_ROLES.items():
        _, replay_source, replay_updates = _physical_state_for_step(lineage, step)
        for role in roles:
            rows.append(
                {
                    "target_step": step,
                    "episode_id": "protocol",
                    "episode_type": "protocol",
                    "state_role": role.split(":", 1)[-1],
                    "state_role_id": role,
                    "episode_status": "observed",
                    "resolution": (
                        "exact_checkpoint" if replay_source is None else "deterministic_replay"
                    ),
                    "replay_source_step": replay_source,
                    "replay_updates": replay_updates,
                    "active_test_episode_ids": _active_test_episode_ids(collapse, step),
                    "recovery_start": "not_applicable",
                    "recovery_confirmed": "not_applicable",
                }
            )
    return sorted(rows, key=lambda row: (int(row["target_step"]), str(row["state_role_id"])))


def _run_m2a(
    root: Path,
    lineage: CheckpointLineage,
    selection: EpisodeSelection,
    collapse: dict[str, Any],
    experiment: ExperimentConfig,
    data: ModularAdditionData,
    terminal_run: Path,
    config: M2AnalysisConfig,
    model_code_commit: str,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    write_analysis_status(root, m2a_status="running")
    scratch = (repository_root() / config.replay_temp_dir / root.name).resolve()
    scratch.mkdir(parents=True, exist_ok=False)
    try:
        replay_payloads, bridges = _run_replay_bridges(
            root, lineage, selection, experiment, config, scratch=scratch
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    scalar_by_step = {
        int(row["step"]): row
        for row in load_scalar_records(terminal_run / "metrics" / "scalars.jsonl")
    }
    offsets_by_step = _committed_offsets(terminal_run)
    roles_by_step = {state.step: list(state.state_roles) for state in selection.states}
    device = torch.device(config.device)
    records: list[dict[str, Any]] = []
    for step in selection.target_steps:
        payload, source, resolution, replay_source, replay_updates = _state_payload(
            root, lineage, step, replay_payloads
        )
        model, grouping = _model_from_payload(payload, experiment, device)
        logits = full_table_logits(
            model,
            data.inputs,
            experiment.task.modulus,
            config.analysis_batch_size,
        )
        recomputed, actual_offsets = behavior_snapshot(
            model, data, grouping, device, full_logits=logits
        )
        committed = scalar_by_step.get(step)
        if committed is None:
            passed = step == 0
            differences: dict[str, float | None] = {}
        else:
            passed, differences = _behavior_validation_passed(
                committed,
                recomputed,
                offsets_by_step[step],
                actual_offsets,
                config,
            )
        record = {
            "schema_version": M2_ANALYSIS_OUTPUT_SCHEMA_VERSION,
            "target_step": step,
            "episode_ids": sorted({role.split(":", 1)[0] for role in roles_by_step[step]}),
            "state_roles": roles_by_step[step],
            "resolution": resolution,
            "run_id": source.run_id,
            "source_checkpoint_step": source.step,
            "source_checkpoint": source.checkpoint_relative_path,
            "replay_updates": replay_updates,
            "committed_available": committed is not None,
            "recomputed_metrics": recomputed,
            "committed_metrics": committed,
            "absolute_differences": differences,
            "max_absolute_difference": max(
                (value for value in differences.values() if value is not None), default=0.0
            ),
            "error_offsets_match": (
                True
                if committed is None
                else all(
                    actual_offsets.get(split) == offsets_by_step[step].get(split)
                    for split in ("train", "test")
                )
            ),
            "validation_status": "passed" if passed else "failed",
            "checkpoint_sha256": source.checkpoint_sha256,
            "semantic_state_sha256": semantic_state_sha256(payload),
            "model_code_commit": model_code_commit,
        }
        if not finite_tree(record):
            raise ValueError(f"M2-A record contains non-finite values at step {step}")
        records.append(record)
        write_json_lines(root / "m2a" / "checkpoint_validation.jsonl", records)
        del model, logits
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if not passed:
            raise ValueError(f"M2-A behavior validation failed at step {step}")

    csv_records = [
        {
            "target_step": row["target_step"],
            "episode_ids": row["episode_ids"],
            "state_roles": row["state_roles"],
            "resolution": row["resolution"],
            "run_id": row["run_id"],
            "source_checkpoint_step": row["source_checkpoint_step"],
            "replay_updates": row["replay_updates"],
            "committed_available": row["committed_available"],
            "max_absolute_difference": row["max_absolute_difference"],
            "error_offsets_match": row["error_offsets_match"],
            "validation_status": row["validation_status"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "semantic_state_sha256": row["semantic_state_sha256"],
            "model_code_commit": row["model_code_commit"],
        }
        for row in records
    ]
    write_csv(
        root / "m2a" / "checkpoint_validation.csv",
        csv_records,
        list(csv_records[0]),
    )
    role_rows = _episode_role_rows(selection, collapse, lineage)
    write_csv(root / "m2a" / "episode_state_index.csv", role_rows, list(role_rows[0]))
    write_analysis_status(
        root,
        m2a_status="completed",
        m2a_target_count=len(records),
        m2a_replay_target_count=len(bridges),
        m2a_role_reference_count=len(role_rows),
    )
    return replay_payloads, records


def _math_tolerances(config: M2AnalysisConfig) -> MathTolerances:
    values = config.math_tolerances
    return MathTolerances(
        centering_atol=values.centering_atol,
        reconstruction_rtol=values.reconstruction_rtol,
        orthogonality_normalized=values.orthogonality_normalized,
        energy_identity_rtol=values.energy_identity_rtol,
        invariance_rtol=values.invariance_rtol,
        implication_margin_atol=values.implication_margin_atol,
    )


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _save_selected_tensor(
    root: Path,
    step: int,
    raw_logits: torch.Tensor,
    result,
    state_metadata: dict[str, Any],
) -> dict[str, Any]:
    destination = root / "selected_tensors" / f"step_{step:06d}.npz"
    _atomic_npz(
        destination,
        step=np.asarray(step, dtype=np.int64),
        raw_logits=raw_logits.detach().cpu().numpy(),
        centered_logits=result.centered_logits.detach().cpu().numpy(),
        offset_profile=result.offset_profile.detach().cpu().numpy(),
        projected_logits=result.projected_logits.detach().cpu().numpy(),
        residual_logits=result.residual_logits.detach().cpu().numpy(),
    )
    return {
        "step": step,
        "path": destination.relative_to(root).as_posix(),
        "sha256": sha256_file(destination),
        "size": destination.stat().st_size,
        "shape": list(raw_logits.shape),
        "raw_dtype": str(raw_logits.dtype),
        "reduction_dtype": str(result.centered_logits.dtype),
        **state_metadata,
    }


def _function_state_metadata(
    lineage: CheckpointLineage,
    step: int,
    replay_payloads: dict[int, dict[str, Any]],
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, source, resolution, replay_source, replay_updates = _state_payload(
        root, lineage, step, replay_payloads
    )
    metadata = {
        "state_source": resolution,
        "run_id": source.run_id,
        "checkpoint_sha256": source.checkpoint_sha256,
        "semantic_state_sha256": semantic_state_sha256(payload),
        "replay_source_step": replay_source,
        "replay_updates": replay_updates,
    }
    return payload, metadata


def _function_behavior_alignment(
    step: int,
    metrics: dict[str, Any],
    scalar_by_step: dict[int, dict[str, Any]],
    config: M2AnalysisConfig,
) -> tuple[bool, float | None]:
    if step == 0:
        return True, None
    committed = scalar_by_step[step]
    if any(
        metrics[f"{split}_error_count"] != committed[f"{split}_error_count"]
        for split in ("train", "test")
    ):
        return False, None
    ce_differences = [
        abs(float(metrics[f"{split}_cross_entropy"]) - float(committed[f"{split}_cross_entropy"]))
        for split in ("train", "test")
    ]
    passed = all(
        math.isclose(
            float(metrics[f"{split}_{field}"]),
            float(committed[f"{split}_{field}"]),
            abs_tol=config.behavior_validation_atol,
            rel_tol=config.behavior_validation_rtol,
        )
        for split in ("train", "test")
        for field in ("cross_entropy", "accuracy")
    )
    return passed, max(ce_differences)


def _write_function_metrics(root: Path, records: list[dict[str, Any]]) -> None:
    write_json_lines(root / "m2b" / "function_metrics.jsonl", records)
    write_csv(root / "m2b" / "function_metrics.csv", records, list(records[0]))


def _episode_function_deltas(
    role_rows: list[dict[str, str]], records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_step = {int(row["step"]): row for row in records}
    references: dict[str, list[dict[str, str]]] = {}
    for row in role_rows:
        if row["episode_id"] == "protocol":
            continue
        references.setdefault(row["episode_id"], []).append(row)
    fields = (
        "Gamma",
        "I",
        "Gamma_minus_I",
        "D_eq",
        "full_projected_accuracy",
        "full_entropy_normalized",
        "centered_logit_rms",
    )
    output: list[dict[str, Any]] = []
    for episode_id, rows in sorted(references.items()):
        pre = next((item for item in rows if item["state_role"] == "pre_collapse"), None)
        pre_metrics = by_step[int(pre["target_step"])] if pre is not None else None
        for role in sorted(rows, key=lambda item: (int(item["target_step"]), item["state_role"])):
            metrics = by_step[int(role["target_step"])]
            record: dict[str, Any] = {
                "episode_id": episode_id,
                "episode_type": role["episode_type"],
                "state_role": role["state_role"],
                "step": int(role["target_step"]),
                "pre_collapse_step": int(pre["target_step"]) if pre is not None else None,
            }
            for field in fields:
                record[field] = metrics[field]
                record[f"delta_from_pre_{field}"] = (
                    None
                    if pre_metrics is None or metrics[field] is None or pre_metrics[field] is None
                    else float(metrics[field]) - float(pre_metrics[field])
                )
            output.append(record)
    return output


def _run_m2b(
    root: Path,
    lineage: CheckpointLineage,
    selection: EpisodeSelection,
    replay_payloads: dict[int, dict[str, Any]],
    experiment: ExperimentConfig,
    data: ModularAdditionData,
    terminal_run: Path,
    config: M2AnalysisConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    write_analysis_status(root, m2b_status="running")
    roles_by_step = {state.step: list(state.state_roles) for state in selection.states}
    all_steps = sorted({item.step for item in lineage.canonical} | set(selection.target_steps))
    scalar_by_step = {
        int(row["step"]): row
        for row in load_scalar_records(terminal_run / "metrics" / "scalars.jsonl")
    }
    device = torch.device(config.device)
    model = build_model(experiment).to(device=device, dtype=torch.float32)
    _, grouping = build_adamw(model, experiment.optimization)
    tolerances = _math_tolerances(config)
    records: list[dict[str, Any]] = []
    profiles: list[np.ndarray] = []
    tensor_records: dict[int, dict[str, Any]] = {}
    selected_steps = set(config.selected_tensor_steps)

    for step in all_steps:
        payload, state_metadata = _function_state_metadata(lineage, step, replay_payloads, root)
        model.load_state_dict(payload["model_state"], strict=True)
        raw_logits = full_table_logits(
            model,
            data.inputs,
            experiment.task.modulus,
            config.analysis_batch_size,
        )
        norms = parameter_norm_metrics(model, grouping)
        result = function_space_metrics(
            raw_logits,
            data.train_indices,
            data.test_indices,
            parameter_norm_total=float(norms["parameter_norm_total"]),
            tolerances=tolerances,
        )
        metrics: dict[str, Any] = {
            **result.metrics,
            "step": step,
            **state_metadata,
            "is_regular_grid": step % lineage.regular_interval == 0,
            "state_roles": roles_by_step.get(step, []),
            "full_logits_shape": list(raw_logits.shape),
            "forward_dtype": str(raw_logits.dtype).removeprefix("torch."),
            "reduction_dtype": str(result.centered_logits.dtype).removeprefix("torch."),
            **norms,
        }
        alignment_passed, alignment_max_diff = _function_behavior_alignment(
            step, metrics, scalar_by_step, config
        )
        metrics["committed_behavior_alignment_passed"] = alignment_passed
        metrics["committed_ce_max_abs_diff"] = alignment_max_diff
        if not alignment_passed:
            raise ValueError(f"M2-B batched logits disagree with committed behavior at step {step}")
        if not finite_tree(metrics):
            raise ValueError(f"M2-B metric contains non-finite values at step {step}")
        records.append(metrics)
        profiles.append(result.offset_profile.numpy())
        _write_function_metrics(root, records)
        if config.persist_selected_logits and step in selected_steps:
            tensor_records[step] = _save_selected_tensor(
                root,
                step,
                raw_logits,
                result,
                {**state_metadata, "state_roles": roles_by_step.get(step, [])},
            )
        del raw_logits, result
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _atomic_npz(
        root / "m2b" / "offset_profiles.npz",
        steps=np.asarray(all_steps, dtype=np.int64),
        offset_profiles=np.stack(profiles).astype(np.float64, copy=False),
    )
    events = detect_function_events(records, expected_interval=lineage.regular_interval)
    write_json(root / "m2b" / "function_events.json", events)

    for event_name in ("t_alg", "t_dom"):
        event = events[event_name]
        event_step = event.get("event_step")
        if type(event_step) is not int or event_step in tensor_records:
            continue
        payload, state_metadata = _function_state_metadata(
            lineage, event_step, replay_payloads, root
        )
        model.load_state_dict(payload["model_state"], strict=True)
        raw_logits = full_table_logits(
            model,
            data.inputs,
            experiment.task.modulus,
            config.analysis_batch_size,
        )
        norms = parameter_norm_metrics(model, grouping)
        result = function_space_metrics(
            raw_logits,
            data.train_indices,
            data.test_indices,
            parameter_norm_total=float(norms["parameter_norm_total"]),
            tolerances=tolerances,
        )
        tensor_records[event_step] = _save_selected_tensor(
            root,
            event_step,
            raw_logits,
            result,
            {
                **state_metadata,
                "state_roles": [event_name, *roles_by_step.get(event_step, [])],
            },
        )
    tensor_manifest = {
        "schema_version": 1,
        "analysis_id": root.name,
        "export_limit_mib": config.selected_tensor_export_limit_mib,
        "total_size": sum(int(row["size"]) for row in tensor_records.values()),
        "tensors": [tensor_records[step] for step in sorted(tensor_records)],
    }
    write_json(root / "selected_tensors" / "manifest.json", tensor_manifest)

    with (root / "m2a" / "episode_state_index.csv").open(encoding="utf-8", newline="") as handle:
        role_rows = list(csv.DictReader(handle))
    delta_rows = _episode_function_deltas(role_rows, records)
    write_csv(
        root / "m2b" / "episode_function_deltas.csv",
        delta_rows,
        list(delta_rows[0]),
    )
    write_analysis_status(
        root,
        m2b_status="completed",
        function_metric_count=len(records),
        regular_function_metric_count=len(lineage.canonical),
        selected_tensor_count=len(tensor_records),
    )
    return records, events, list(tensor_records.values())


def _provenance(
    root: Path,
    config: M2AnalysisConfig,
    experiment: ExperimentConfig,
    data: ModularAdditionData,
    report: dict[str, Any],
    source_details: dict[str, Any],
    inventories_before: dict[str, list[dict[str, Any]]],
    inventories_after: dict[str, list[dict[str, Any]]] | None,
    lineage: CheckpointLineage,
    selection: EpisodeSelection,
    function_events: dict[str, Any] | None,
) -> dict[str, Any]:
    frozen_results = {
        root_path: {row["path"]: row["sha256"] for row in rows}
        for root_path, rows in inventories_before.items()
        if root_path.startswith("results/")
    }
    return {
        "schema_version": 1,
        "analysis_id": root.name,
        "analysis_git_commit": _git_commit(),
        "implementation_git_commit": _git_commit(),
        "source_git_commit": source_details["lineage_git_commits"][
            config.source_run_ids.terminal_child
        ],
        "tracked_worktree_clean": _tracked_worktree_clean(),
        "analysis_config_hash": config.analysis_hash(),
        "scientific_config_hash": experiment.scientific_hash(),
        "split_hash": data.split_hash,
        "source_run_ids": config.to_dict()["source_run_ids"],
        "source_lineage": [
            config.source_run_ids.root,
            config.source_run_ids.canonical_parent,
            config.source_run_ids.terminal_child,
        ],
        **source_details,
        "source_inventories_before": inventories_before,
        "source_inventories_after": inventories_after,
        "frozen_results_sha256": frozen_results,
        "physical_checkpoint_count": lineage.physical_count,
        "regular_checkpoint_count": lineage.regular_step_count,
        "regular_step_count": lineage.regular_step_count,
        "branch_anchor_alias_count": len(lineage.aliases),
        "episode_role_reference_count": len(selection.references),
        "episode_target_state_count": len(selection.states),
        "replay_target_count": sum(step % 100 != 0 for step in selection.target_steps),
        "replay_count": sum(step % 100 != 0 for step in selection.target_steps),
        "modulus": experiment.task.modulus,
        "behavior_events": {"t_fit": 100, "t_grok50": 6050, "t_grok99": 7000},
        "function_events": function_events,
        "hardware": report,
        "analysis_batch_size": config.analysis_batch_size,
        "reduction_dtype": config.cpu_reduction_dtype,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "smoothing": False,
        "interpolation": False,
        "resampling": False,
    }


def run_m2_analysis(config: M2AnalysisConfig) -> Path:
    """Execute gated M2-A then M2-B without writing to any M1 artifact."""
    configure_reproducibility(seed=1, deterministic=True)
    root = create_analysis_dir(
        repository_root() / config.analysis_runs_dir,
        config.analysis_hash(),
    )
    current_stage = "initializing"
    try:
        dump_m2_analysis_config(config, root / "analysis_config.resolved.yaml")
        write_analysis_status(
            root,
            analysis_id=root.name,
            analysis_config_hash=config.analysis_hash(),
        )
        doctor = collect_doctor_report()
        doctor_errors = validate_doctor_report(
            doctor,
            require_cuda=True,
            expected_device=config.expected_device,
            expected_vram_gb=config.expected_vram_gb,
        )
        if doctor_errors:
            raise RuntimeError("formal M2 doctor failed: " + "; ".join(doctor_errors))
        torch.cuda.reset_peak_memory_stats(torch.device(config.device))
        run_ids = config.source_run_ids
        run_dirs = {
            run_id: repository_root() / "runs" / run_id
            for run_id in (run_ids.root, run_ids.canonical_parent, run_ids.terminal_child)
        }
        result_dirs = {
            "m1_reference": repository_root() / config.source_result_dirs.m1_reference,
            "m1_extended": repository_root() / config.source_result_dirs.m1_extended,
        }
        if any(not path.is_dir() for path in [*run_dirs.values(), *result_dirs.values()]):
            raise ValueError("one or more frozen M1 source directories are missing")
        write_analysis_status(root, analysis_status="running", current_stage="source_preflight")
        inventories_before = _inventory_sources(run_dirs, result_dirs)
        write_json(root / "provenance" / "source_inventory_before.json", inventories_before)
        experiment, data, source_details = _validate_source_identity(config, run_dirs)
        terminal_run = run_dirs[run_ids.terminal_child]
        _copy_context(terminal_run, root)
        lineage = resolve_checkpoint_lineage(
            [run_dirs[run_ids.root], run_dirs[run_ids.canonical_parent], terminal_run],
            expected_physical_count=503,
            expected_regular_count=501,
            regular_interval=100,
            expected_start_step=0,
            expected_end_step=50_000,
        )
        _write_lineage_tables(root, lineage)
        selection, collapse = _episode_selection(terminal_run)
        provisional = _provenance(
            root,
            config,
            experiment,
            data,
            doctor.to_dict(),
            source_details,
            inventories_before,
            None,
            lineage,
            selection,
            None,
        )
        write_json(root / "provenance.json", provisional)

        current_stage = "m2a"
        write_analysis_status(root, current_stage=current_stage)
        replay_payloads, _ = _run_m2a(
            root,
            lineage,
            selection,
            collapse,
            experiment,
            data,
            terminal_run,
            config,
            _git_commit(),
        )

        current_stage = "m2b"
        write_analysis_status(root, current_stage=current_stage)
        _, function_events, _ = _run_m2b(
            root,
            lineage,
            selection,
            replay_payloads,
            experiment,
            data,
            terminal_run,
            config,
        )
        current_stage = "source_postflight"
        write_analysis_status(root, current_stage=current_stage)
        inventories_after = _inventory_sources(run_dirs, result_dirs)
        write_json(root / "provenance" / "source_inventory_after.json", inventories_after)
        if inventories_after != inventories_before:
            raise ValueError("one or more frozen M1 source artifacts changed during M2")
        final_provenance = _provenance(
            root,
            config,
            experiment,
            data,
            doctor.to_dict(),
            source_details,
            inventories_before,
            inventories_after,
            lineage,
            selection,
            function_events,
        )
        write_json(root / "provenance.json", final_provenance)
        peak_allocated = torch.cuda.max_memory_allocated(torch.device(config.device))
        peak_reserved = torch.cuda.max_memory_reserved(torch.device(config.device))
        write_analysis_status(
            root,
            analysis_status="completed",
            current_stage="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            max_memory_allocated=peak_allocated,
            max_memory_reserved=peak_reserved,
        )
        return root
    except KeyboardInterrupt:
        write_analysis_status(
            root,
            analysis_status="interrupted",
            current_stage=current_stage,
        )
        raise
    except Exception as error:
        write_analysis_status(
            root,
            analysis_status="failed",
            current_stage=current_stage,
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise
