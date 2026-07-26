"""Portable, read-only export of audited M2 function-space evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from transgrokking.utils.atomic import write_json

FIGURE_NAMES = (
    "behavior_function_timeline",
    "gamma_interference_timeline",
    "equivariance_timeline",
    "projected_behavior_timeline",
    "logit_scale_entropy_timeline",
    "offset_profile_heatmap",
    "collapse_aligned_function_metrics",
    "state_space_gamma_I_Deq",
    "episode_function_delta_table",
    "m2a_validation_summary",
)

REQUIRED_ANALYSIS_FILES = (
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
    "audit/m2_analysis.json",
    "selected_tensors/manifest.json",
)

REQUIRED_FUNCTION_FIELDS = (
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
    "full_cross_entropy",
    "full_accuracy",
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
)

_FORBIDDEN_ARTIFACT_NAME = re.compile(
    r"(?:fourier|frequency[-_]line|restricted[-_]frequency|(?:^|[-_])e[-_]line(?:[-_.]|$))",
    re.IGNORECASE,
)
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    _require_finite_json(value, str(path))
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL records are forbidden")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            _require_finite_json(record, f"{path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: expected at least one record")
    return records


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        rows = list(reader)
    return list(reader.fieldnames), rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_finite_json(value: Any, location: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{location}: JSON contains a non-finite number")
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_json(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_finite_json(child, f"{location}[{index}]")


def _require_finite_csv(rows: Iterable[dict[str, str]], path: Path) -> None:
    for row_number, row in enumerate(rows, start=2):
        for column, value in row.items():
            if value is None or value.strip() == "":
                continue
            try:
                parsed = float(value)
            except ValueError:
                continue
            if not math.isfinite(parsed):
                raise ValueError(f"{path}:{row_number}:{column}: non-finite CSV value")


def _safe_relative_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"invalid portable relative path: {relative!r}")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"path escapes analysis directory: {relative}") from error
    return candidate


def _verify_analysis_audit(source: Path, audit: dict[str, Any]) -> dict[str, str]:
    if audit.get("passed") is not True:
        raise ValueError("M2 export requires a passing analysis audit")
    hashes = audit.get("audited_source_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("M2 analysis audit lacks audited_source_sha256")
    normalized: dict[str, str] = {}
    for relative, expected in hashes.items():
        if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("M2 analysis audit contains an invalid source hash entry")
        path = _safe_relative_path(source, relative)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"audited M2 analysis source changed: {relative}")
        normalized[Path(relative).as_posix()] = expected
    required_hashed = set(REQUIRED_ANALYSIS_FILES) - {"audit/m2_analysis.json"}
    if not required_hashed.issubset(normalized):
        missing = sorted(required_hashed - set(normalized))
        raise ValueError(f"M2 analysis audit hash set is incomplete: {missing}")
    return normalized


def _validate_function_records(
    json_records: list[dict[str, Any]],
    csv_fields: list[str],
    csv_rows: list[dict[str, str]],
) -> None:
    missing_json = set(REQUIRED_FUNCTION_FIELDS) - set(json_records[0])
    missing_csv = set(REQUIRED_FUNCTION_FIELDS) - set(csv_fields)
    if missing_json or missing_csv:
        raise ValueError(
            "function metrics schema is incomplete: "
            f"JSONL missing={sorted(missing_json)}, CSV missing={sorted(missing_csv)}"
        )
    json_fields = set(json_records[0])
    if any(set(record) != json_fields for record in json_records):
        raise ValueError("function metric JSONL records do not share one schema")
    if set(csv_fields) != json_fields:
        raise ValueError("function metric JSONL/CSV field sets differ")
    steps = [record.get("step") for record in json_records]
    if any(type(step) is not int or step < 0 for step in steps):
        raise ValueError("function metric steps must be non-negative integers")
    if steps != sorted(set(steps)):
        raise ValueError("function metric steps must be unique and strictly increasing")
    if len(csv_rows) != len(json_records):
        raise ValueError("function metric JSONL/CSV row counts differ")
    try:
        csv_steps = [int(row["step"]) for row in csv_rows]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("function metric CSV has an invalid step column") from error
    if csv_steps != steps:
        raise ValueError("function metric JSONL/CSV steps differ")
    for index, (record, csv_row) in enumerate(zip(json_records, csv_rows, strict=True)):
        roles = record.get("state_roles")
        if not isinstance(roles, list) or not all(isinstance(role, str) for role in roles):
            raise ValueError(f"function metric record {index} has invalid state_roles")
        if type(record.get("is_regular_grid")) is not bool:
            raise ValueError(f"function metric record {index} has invalid is_regular_grid")
        for field, expected in record.items():
            if not _csv_value_matches(expected, csv_row.get(field)):
                raise ValueError(
                    f"function metric JSONL/CSV differ at record {index}, field {field}"
                )


def _csv_value_matches(expected: Any, raw: str | None) -> bool:
    if raw is None:
        return False
    if expected is None:
        return raw == ""
    if isinstance(expected, bool):
        return raw.lower() == str(expected).lower()
    if isinstance(expected, int):
        try:
            return int(raw) == expected
        except ValueError:
            return False
    if isinstance(expected, float):
        try:
            parsed = float(raw)
        except ValueError:
            return False
        return math.isfinite(parsed) and parsed == expected
    if isinstance(expected, str):
        return raw == expected
    if isinstance(expected, (list, dict)):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return False
        return parsed == expected
    return False


def _load_offset_profiles(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if "steps" not in archive.files:
            raise ValueError("offset_profiles.npz lacks steps")
        profile_key = next(
            (key for key in ("offset_profiles", "profiles", "g") if key in archive.files),
            None,
        )
        if profile_key is None:
            raise ValueError("offset_profiles.npz lacks offset_profiles")
        steps = np.asarray(archive["steps"])
        profiles = np.asarray(archive[profile_key])
    if steps.ndim != 1 or profiles.ndim != 2 or profiles.shape[0] != steps.shape[0]:
        raise ValueError("offset profile arrays have incompatible shapes")
    if profiles.shape[1] < 2:
        raise ValueError("offset profile modulus must be at least 2")
    if not np.issubdtype(steps.dtype, np.integer):
        raise ValueError("offset profile steps must be integer")
    if list(map(int, steps)) != sorted(set(map(int, steps))):
        raise ValueError("offset profile steps must be unique and strictly increasing")
    if not np.isfinite(profiles).all():
        raise ValueError("offset profiles contain non-finite values")
    return steps.astype(np.int64, copy=False), profiles.astype(np.float64, copy=False)


def _event_step(events: dict[str, Any], name: str) -> int | None:
    value = events.get(name)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, dict):
        step = value.get("event_step")
        if isinstance(step, int) and not isinstance(step, bool):
            return step
    return None


def _behavior_events(provenance: dict[str, Any]) -> dict[str, int | None]:
    values = provenance.get("behavior_events")
    if not isinstance(values, dict):
        values = provenance.get("frozen_events", {})
    if not isinstance(values, dict):
        values = {}
    return {name: _event_step(values, name) for name in ("t_fit", "t_grok50", "t_grok99")}


def _episode_states(path: Path) -> dict[str, dict[str, int]]:
    _, rows = _read_csv(path)
    episodes: dict[str, dict[str, int]] = {}
    for row in rows:
        episode_id = row.get("episode_id")
        role = row.get("state_role") or row.get("role")
        raw_step = row.get("target_step") or row.get("step")
        if not episode_id or not role or raw_step in (None, ""):
            continue
        try:
            step = int(raw_step)
        except ValueError:
            continue
        episodes.setdefault(episode_id, {})[role] = step
    return episodes


def _collapse_spans(
    episode_states: dict[str, dict[str, int]], final_step: int
) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    joint_ids = {episode_id for episode_id in episode_states if episode_id.startswith("joint_")}
    selected_ids = joint_ids or {
        episode_id for episode_id in episode_states if episode_id.startswith("test_")
    }
    for episode_id in sorted(selected_ids):
        roles = episode_states[episode_id]
        onset = roles.get("onset")
        if onset is None:
            continue
        recovery_steps = [
            step for role, step in roles.items() if role.endswith("recovery_confirmed")
        ]
        end = (
            min(recovery_steps) if recovery_steps else roles.get("terminal_unrecovered", final_step)
        )
        spans.append((episode_id, onset, end))
    return spans


def _float_series(
    records: list[dict[str, Any]], field: str, *, allow_none: bool = False
) -> list[float]:
    values: list[float] = []
    for index, record in enumerate(records):
        value = record.get(field)
        if value is None and allow_none:
            values.append(float("nan"))
            continue
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError(f"function metric {field} is not finite at record {index}")
        values.append(float(value))
    return values


def _maximum_absolute_difference(record: dict[str, Any]) -> float:
    maximum = 0.0

    def visit(value: Any) -> None:
        nonlocal maximum
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            maximum = max(maximum, abs(float(value)))

    differences = record.get("absolute_differences")
    if isinstance(differences, dict):
        visit(differences)
    for key, value in record.items():
        if "absolute_difference" in key or key.endswith("_abs_diff"):
            visit(value)
    return maximum


def _plot_all(
    output: Path,
    analysis_id: str,
    records: list[dict[str, Any]],
    offset_steps: np.ndarray,
    offset_profiles: np.ndarray,
    function_events: dict[str, Any],
    provenance: dict[str, Any],
    episode_index_path: Path,
    episode_deltas_path: Path,
    validation_path: Path,
    behavior_path: Path,
    collapse_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output / "figures"
    figure_dir.mkdir()
    steps = [int(record["step"]) for record in records]
    behavior_records = _read_jsonl(behavior_path)
    behavior_steps = [int(record["step"]) for record in behavior_records]
    short_id = analysis_id[-8:]
    episode_states = _episode_states(episode_index_path)
    collapse_spans = _collapse_spans(episode_states, behavior_steps[-1])
    events = {
        **_behavior_events(provenance),
        "t_alg": _event_step(function_events, "t_alg"),
        "t_dom": _event_step(function_events, "t_dom"),
    }

    def decorate(axes, *, zero_line: bool = False, spans: bool = False) -> None:
        axis_list = np.atleast_1d(axes).ravel()
        for axis in axis_list:
            if zero_line:
                axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
            for name, step in events.items():
                if step is not None:
                    axis.axvline(step, linestyle="--", linewidth=0.7, alpha=0.65, label=name)
            if spans:
                for index, (_, start, end) in enumerate(collapse_spans):
                    axis.axvspan(
                        start,
                        end,
                        color="tab:red",
                        alpha=0.07,
                        label="test/joint collapse" if index == 0 else None,
                    )
            axis.set_xlabel("optimizer step")
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                unique = dict(zip(labels, handles, strict=False))
                axis.legend(unique.values(), unique.keys(), fontsize="x-small", ncols=2)

    def finish(figure, name: str, title: str) -> None:
        figure.suptitle(f"{title} — {short_id}")
        figure.tight_layout()
        for suffix in ("png", "svg"):
            figure.savefig(figure_dir / f"{name}.{suffix}", dpi=180)
        plt.close(figure)

    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(
        behavior_steps,
        _float_series(behavior_records, "train_accuracy"),
        label="train accuracy",
    )
    axes[0].plot(
        behavior_steps,
        _float_series(behavior_records, "test_accuracy"),
        label="test accuracy",
    )
    axes[0].set_ylabel("accuracy")
    axes[1].plot(steps, _float_series(records, "Gamma"), label="Gamma")
    axes[1].plot(steps, _float_series(records, "I"), label="I")
    axes[1].plot(steps, _float_series(records, "Gamma_minus_I"), label="Gamma - I")
    axes[1].set_ylabel("logit")
    axes[2].plot(steps, _float_series(records, "D_eq", allow_none=True), label="D_eq")
    axes[2].set_ylabel("fraction")
    decorate(axes, spans=True)
    finish(figure, "behavior_function_timeline", "Behavior and function timeline (raw)")

    figure, axis = plt.subplots(figsize=(11, 5))
    for field, label in (("Gamma", "Gamma"), ("I", "I"), ("Gamma_minus_I", "Gamma - I")):
        axis.plot(steps, _float_series(records, field), label=label)
    axis.set_ylabel("logit")
    decorate(axis, zero_line=True, spans=True)
    finish(figure, "gamma_interference_timeline", "Algorithm margin and interference (raw)")

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(steps, _float_series(records, "D_eq", allow_none=True), label="D_eq")
    axes[0].set_ylabel("D_eq")
    axes[1].plot(steps, _float_series(records, "equivariant_energy"), label="equivariant")
    axes[1].plot(steps, _float_series(records, "residual_energy"), label="residual")
    axes[1].set_ylabel("squared Frobenius")
    decorate(axes, spans=True)
    finish(figure, "equivariance_timeline", "Equivariance decomposition (raw)")

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for split in ("train", "test", "full"):
        axes[0].plot(
            steps,
            _float_series(records, f"{split}_projected_accuracy"),
            label=f"{split} projected accuracy",
        )
        axes[1].plot(
            steps,
            _float_series(records, f"{split}_projected_ce"),
            label=f"{split} projected CE",
        )
    axes[0].set_ylabel("accuracy")
    axes[1].set_ylabel("cross-entropy")
    decorate(axes, spans=True)
    finish(figure, "projected_behavior_timeline", "Projected behavior (raw)")

    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(steps, _float_series(records, "centered_logit_rms"), label="logit RMS")
    axes[0].set_ylabel("RMS")
    for split in ("train", "test", "full"):
        axes[1].plot(
            steps,
            _float_series(records, f"{split}_entropy_normalized"),
            label=f"{split} normalized entropy",
        )
    axes[1].set_ylabel("entropy / log(p)")
    axes[2].plot(
        steps,
        _float_series(records, "Gamma_over_logit_rms", allow_none=True),
        label="Gamma / RMS",
    )
    axes[2].plot(
        steps,
        _float_series(records, "I_over_logit_rms", allow_none=True),
        label="I / RMS",
    )
    axes[2].set_ylabel("normalized logit")
    decorate(axes, spans=True)
    finish(figure, "logit_scale_entropy_timeline", "Logit scale and entropy (raw)")

    figure, axis = plt.subplots(figsize=(12, 6))
    image = axis.pcolormesh(
        offset_steps,
        np.arange(offset_profiles.shape[1]),
        offset_profiles.T,
        shading="nearest",
        cmap="coolwarm",
    )
    axis.set_xlabel("optimizer step")
    axis.set_ylabel("offset d")
    figure.colorbar(image, ax=axis, label="g(d)")
    finish(figure, "offset_profile_heatmap", "Reynolds offset profiles (raw)")

    selected_episode_ids = [
        episode_id
        for episode_id in episode_states
        if episode_id.startswith("test_") or episode_id.startswith("joint_")
    ]
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fields = ("Gamma_minus_I", "D_eq", "test_accuracy")
    for episode_id in selected_episode_ids:
        onset = episode_states[episode_id].get("onset")
        if onset is None:
            continue
        episode_steps = set(episode_states[episode_id].values())
        points = [record for record in records if int(record["step"]) in episode_steps]
        points.sort(key=lambda record: int(record["step"]))
        if not points:
            continue
        relative = [int(record["step"]) - onset for record in points]
        for axis, field in zip(axes, fields, strict=True):
            axis.plot(
                relative,
                _float_series(points, field, allow_none=field == "D_eq"),
                marker="o",
                alpha=0.5,
                label=episode_id,
            )
    for axis, field in zip(axes, fields, strict=True):
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_ylabel(field)
    axes[-1].set_xlabel("step relative to episode onset")
    if selected_episode_ids:
        axes[0].legend(fontsize="xx-small", ncols=4)
    finish(
        figure,
        "collapse_aligned_function_metrics",
        "Collapse-aligned function metrics (no smoothing)",
    )

    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")
    gamma = _float_series(records, "Gamma")
    interference = _float_series(records, "I")
    deq = _float_series(records, "D_eq", allow_none=True)
    finite_state = np.isfinite(gamma) & np.isfinite(interference) & np.isfinite(deq)
    filtered_gamma = np.asarray(gamma)[finite_state]
    filtered_interference = np.asarray(interference)[finite_state]
    filtered_deq = np.asarray(deq)[finite_state]
    filtered_steps = np.asarray(steps)[finite_state]
    if filtered_steps.size == 0:
        raise ValueError("state-space plot has no finite Gamma/I/D_eq state")
    points = axis.scatter(
        filtered_gamma,
        filtered_interference,
        filtered_deq,
        c=filtered_steps,
        cmap="viridis",
        s=12,
    )
    axis.plot(filtered_gamma, filtered_interference, filtered_deq, linewidth=0.5, alpha=0.4)
    axis.set_xlabel("Gamma")
    axis.set_ylabel("I")
    axis.set_zlabel("D_eq")
    figure.colorbar(points, ax=axis, label="optimizer step", shrink=0.65)
    finish(figure, "state_space_gamma_I_Deq", "Function state-space trajectory")

    delta_fields, delta_rows = _read_csv(episode_deltas_path)
    if not delta_rows:
        raise ValueError("episode_function_deltas.csv must contain at least one row")
    label_field = "episode_id" if "episode_id" in delta_fields else delta_fields[0]
    numeric_fields: list[str] = []
    preferred_delta_fields = [
        field
        for field in delta_fields
        if field.startswith("delta_from_pre_") or field.endswith("_delta")
    ]
    candidate_fields = preferred_delta_fields or [
        field for field in delta_fields if field not in {"step", "pre_collapse_step"}
    ]
    for field in candidate_fields:
        if field in {label_field, "state_role", "episode_type"}:
            continue
        try:
            values = [float(row[field]) for row in delta_rows if row.get(field, "") != ""]
        except ValueError:
            continue
        if values:
            numeric_fields.append(field)
    numeric_fields = numeric_fields[:8]
    if not numeric_fields:
        raise ValueError("episode_function_deltas.csv lacks numeric metric columns")
    labels = [
        ":".join(
            part
            for part in (row[label_field], row.get("state_role"))
            if isinstance(part, str) and part
        )
        for row in delta_rows
    ]
    matrix = np.array(
        [
            [float(row[field]) if row.get(field, "") != "" else np.nan for field in numeric_fields]
            for row in delta_rows
        ],
        dtype=np.float64,
    )
    figure, axis = plt.subplots(
        figsize=(max(8, len(numeric_fields) * 1.25), min(18, max(4, len(labels) * 0.25)))
    )
    image = axis.imshow(matrix, aspect="auto", cmap="coolwarm")
    axis.set_xticks(range(len(numeric_fields)), numeric_fields, rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    figure.colorbar(image, ax=axis, label="state delta")
    finish(figure, "episode_function_delta_table", "Episode function deltas")

    validation_records = _read_jsonl(validation_path)
    resolutions = {name: 0 for name in ("exact_checkpoint", "deterministic_replay", "unresolved")}
    maximum_error = 0.0
    for record in validation_records:
        resolution = record.get("resolution")
        if resolution in resolutions:
            resolutions[resolution] += 1
        maximum_error = max(maximum_error, _maximum_absolute_difference(record))
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(list(resolutions), list(resolutions.values()))
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("state count")
    axes[1].bar(["maximum absolute metric difference"], [maximum_error])
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].set_ylabel("absolute difference")
    finish(figure, "m2a_validation_summary", "M2-A validation summary")

    # Parse the frozen episode source as part of figure generation even though the
    # normalized episode-state index controls plotted spans.
    collapse_context = _read_json(collapse_path)
    if not isinstance(collapse_context.get("episodes"), list):
        raise ValueError("context/collapse_episodes.json lacks an episodes list")


def _copy_analysis_files(source: Path, temporary: Path) -> dict[str, str]:
    copied_hashes: dict[str, str] = {}
    for relative in REQUIRED_ANALYSIS_FILES:
        source_path = _safe_relative_path(source, relative)
        if not source_path.is_file():
            raise ValueError(f"M2 analysis is incomplete: missing {relative}")
        target = temporary / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)
        copied_hashes[relative] = _sha256(target)
    return copied_hashes


def _portable_source_path(analysis_id: str, relative: str) -> str:
    return (Path("analysis_runs") / analysis_id / Path(relative)).as_posix()


def _portable_hardware(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    hardware = dict(value)
    for field in ("prefix", "expected_prefix"):
        if field in hardware:
            hardware[field] = "./env"
    return hardware


def _prepare_portable_tensors(source: Path, output: Path, analysis_id: str) -> dict[str, Any]:
    source_manifest_path = source / "selected_tensors/manifest.json"
    manifest = _read_json(source_manifest_path)
    entries = manifest.get("tensors")
    if not isinstance(entries, list):
        raise ValueError("selected tensor manifest lacks a tensors list")
    total_size = sum(int(entry.get("size", -1)) for entry in entries if isinstance(entry, dict))
    if total_size < 0 or total_size != manifest.get("total_size", total_size):
        raise ValueError("selected tensor manifest total_size is inconsistent")
    limit_mib = manifest.get("export_limit_mib")
    if not isinstance(limit_mib, (int, float)) or isinstance(limit_mib, bool) or limit_mib < 0:
        raise ValueError("selected tensor manifest has an invalid export_limit_mib")
    export_tensors = total_size <= float(limit_mib) * 1024 * 1024
    portable_entries: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"selected tensor entry {index} is not an object")
        relative = entry.get("path")
        expected_hash = entry.get("sha256")
        expected_size = entry.get("size")
        if (
            not isinstance(relative, str)
            or not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
        ):
            raise ValueError(f"selected tensor entry {index} has invalid identity fields")
        tensor_source = _safe_relative_path(source, relative)
        if (
            not tensor_source.is_file()
            or tensor_source.stat().st_size != expected_size
            or _sha256(tensor_source) != expected_hash
        ):
            raise ValueError(f"selected tensor source differs from manifest: {relative}")
        portable = dict(entry)
        portable["source_path"] = _portable_source_path(analysis_id, relative)
        portable["exported"] = export_tensors
        if export_tensors:
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tensor_source, target)
            portable["path"] = Path(relative).as_posix()
            portable["storage_scope"] = "portable_export"
        else:
            portable["path"] = portable["source_path"]
            portable["storage_scope"] = "analysis_run_local"
        portable_entries.append(portable)
    portable_manifest = {
        **manifest,
        "schema_version": 1,
        "analysis_id": analysis_id,
        "source_manifest_sha256": _sha256(source_manifest_path),
        "full_tensors_exported": export_tensors,
        "exported_total_size": total_size if export_tensors else 0,
        "tensors": portable_entries,
    }
    _require_no_absolute_strings(portable_manifest, "selected_tensors.manifest")
    write_json(output / "selected_tensors/manifest.json", portable_manifest)
    return portable_manifest


def _portable_provenance(
    source_provenance: dict[str, Any],
    analysis_id: str,
    copied_hashes: dict[str, str],
    function_events: dict[str, Any],
    record_count: int,
) -> dict[str, Any]:
    selected_keys = (
        "source_git_commit",
        "analysis_git_commit",
        "implementation_git_commit",
        "scientific_config_hash",
        "split_hash",
        "analysis_config_hash",
        "source_run_ids",
        "source_lineage",
        "analysis_batch_size",
        "reduction_dtype",
        "modulus",
        "behavior_events",
        "frozen_events",
        "frozen_results_sha256",
        "frozen_results_manifests",
        "physical_checkpoint_count",
        "regular_step_count",
        "replay_count",
    )
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "analysis_id": analysis_id,
        **{key: source_provenance[key] for key in selected_keys if key in source_provenance},
        "function_metric_record_count": record_count,
        "function_events": function_events,
        "source_analysis_files": {
            relative: _portable_source_path(analysis_id, relative)
            for relative in sorted(copied_hashes)
        },
        "source_analysis_sha256": dict(sorted(copied_hashes.items())),
        "figure_files": [
            f"figures/{name}.{suffix}" for name in FIGURE_NAMES for suffix in ("png", "svg")
        ],
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "smoothing": False,
        "interpolation": False,
        "outlier_deletion": False,
        "missing_step_imputation": False,
        "m3_analysis_included": False,
    }
    source_inventories = source_provenance.get("source_inventories_before")
    if isinstance(source_inventories, dict):
        provenance["frozen_results_manifests"] = {
            name: records
            for name, records in source_inventories.items()
            if isinstance(name, str) and name.startswith("results/")
        }
    hardware = _portable_hardware(source_provenance.get("hardware"))
    if hardware is not None:
        provenance["hardware"] = hardware
    _require_portable_paths(provenance)
    return provenance


def _require_portable_paths(provenance: dict[str, Any]) -> None:
    paths = provenance.get("source_analysis_files")
    if not isinstance(paths, dict) or not paths:
        raise ValueError("portable provenance lacks source_analysis_files")
    for value in paths.values():
        if (
            not isinstance(value, str)
            or "\\" in value
            or value.startswith("/")
            or _WINDOWS_ABSOLUTE.match(value)
            or ".." in Path(value).parts
        ):
            raise ValueError(f"non-portable provenance path: {value!r}")
    _require_no_absolute_strings(provenance, "provenance")


def _require_no_absolute_strings(value: Any, location: str) -> None:
    if isinstance(value, str) and (value.startswith("/") or _WINDOWS_ABSOLUTE.match(value)):
        raise ValueError(f"{location}: absolute paths are forbidden in portable provenance")
    if isinstance(value, dict):
        for key, child in value.items():
            _require_no_absolute_strings(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_no_absolute_strings(child, f"{location}[{index}]")


def _write_readme(
    path: Path,
    analysis_id: str,
    function_events: dict[str, Any],
    record_count: int,
) -> None:
    t_alg = _event_step(function_events, "t_alg")
    t_dom = _event_step(function_events, "t_dom")
    content = (
        "# M2 checkpoint validation and function-space evidence\n\n"
        f"- Analysis ID: `{analysis_id}`\n"
        f"- Function states: `{record_count}`\n"
        f"- Regular-grid `t_alg`: `{t_alg if t_alg is not None else 'not reached'}`\n"
        f"- Regular-grid `t_dom`: `{t_dom if t_dom is not None else 'not reached'}`\n"
        "- Analysis audit: `passed`\n"
        "- Portable export audit: `audit/m2_export.json`\n\n"
        "The tables and figures use actual checkpoint or deterministic-replay steps. No "
        "smoothing, interpolation, outlier deletion, or missing-step imputation was applied. "
        "This export contains M2 checkpoint validation, centered-logit and Reynolds-projection "
        "evidence only. It contains no Fourier, representation, circuit, intervention, "
        "multi-seed, weight-decay-grid, or Congruence result.\n"
    )
    path.write_text(content, encoding="utf-8")


def _tree_file_hashes(root: Path, *, exclude: set[str] | None = None) -> dict[str, str]:
    excluded = exclude or set()
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    }


def _valid_frozen_inventories(value: Any, *, require_size: bool = False) -> bool:
    if not isinstance(value, dict) or len(value) < 2:
        return False
    for inventory in value.values():
        if isinstance(inventory, dict):
            entries = [{"path": path, "sha256": sha256} for path, sha256 in inventory.items()]
        elif isinstance(inventory, list):
            entries = inventory
        else:
            return False
        if not entries:
            return False
        for entry in entries:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("path"), str)
                or not isinstance(entry.get("sha256"), str)
                or len(entry["sha256"]) != 64
                or (
                    require_size
                    and (
                        not isinstance(entry.get("size"), int)
                        or isinstance(entry.get("size"), bool)
                        or entry["size"] < 0
                    )
                )
            ):
                return False
    return True


def audit_m2_export(export_dir: str | Path, *, write: bool = False) -> dict[str, Any]:
    """Validate one portable M2 export; optionally atomically write its audit."""
    root = Path(export_dir).resolve()
    errors: list[str] = []
    required = {
        "README.md",
        "provenance.json",
        *REQUIRED_ANALYSIS_FILES,
        *{f"figures/{name}.{suffix}" for name in FIGURE_NAMES for suffix in ("png", "svg")},
    }
    missing = sorted(relative for relative in required if not (root / relative).is_file())
    if missing:
        errors.append(f"missing required files: {missing}")

    provenance: dict[str, Any] = {}
    function_records: list[dict[str, Any]] = []
    try:
        provenance = _read_json(root / "provenance.json")
        _require_portable_paths(provenance)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"invalid provenance: {error}")

    try:
        analysis_audit = _read_json(root / "audit/m2_analysis.json")
        if analysis_audit.get("passed") is not True:
            errors.append("copied analysis audit is not passing")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"invalid analysis audit: {error}")

    try:
        function_records = _read_jsonl(root / "m2b/function_metrics.jsonl")
        fields, rows = _read_csv(root / "m2b/function_metrics.csv")
        _require_finite_csv(rows, root / "m2b/function_metrics.csv")
        _validate_function_records(function_records, fields, rows)
        offset_steps, offset_profiles = _load_offset_profiles(root / "m2b/offset_profiles.npz")
        if list(map(int, offset_steps)) != [int(record["step"]) for record in function_records]:
            errors.append("offset profile and function metric steps differ")
        modulus = provenance.get("modulus")
        if (
            not isinstance(modulus, int)
            or isinstance(modulus, bool)
            or modulus < 2
            or offset_profiles.shape[1] != modulus
        ):
            errors.append("offset profile length differs from valid provenance modulus")
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as error:
        errors.append(f"invalid function artifacts: {error}")

    for path in root.rglob("*.json"):
        try:
            _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"invalid JSON {path.relative_to(root).as_posix()}: {error}")
    for path in root.rglob("*.jsonl"):
        try:
            _read_jsonl(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"invalid JSONL {path.relative_to(root).as_posix()}: {error}")
    for path in root.rglob("*.csv"):
        try:
            _, rows = _read_csv(path)
            _require_finite_csv(rows, path)
        except (OSError, ValueError) as error:
            errors.append(f"invalid CSV {path.relative_to(root).as_posix()}: {error}")

    forbidden = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and _FORBIDDEN_ARTIFACT_NAME.search(path.relative_to(root).as_posix())
    )
    if forbidden:
        errors.append(f"forbidden M3+ artifact names: {forbidden}")
    frozen = provenance.get("frozen_results_sha256")
    frozen_manifests = provenance.get("frozen_results_manifests")
    frozen_valid = _valid_frozen_inventories(frozen) and _valid_frozen_inventories(
        frozen_manifests, require_size=True
    )
    if not frozen_valid:
        errors.append("provenance lacks both frozen M1 result inventories")
    source_hashes = provenance.get("source_analysis_sha256")
    source_copies_valid = isinstance(source_hashes, dict) and bool(source_hashes)
    if source_copies_valid:
        for relative, expected in source_hashes.items():
            if relative in {"provenance.json", "selected_tensors/manifest.json"}:
                continue
            path = _safe_relative_path(root, relative)
            if (
                not isinstance(expected, str)
                or len(expected) != 64
                or not path.is_file()
                or _sha256(path) != expected
            ):
                source_copies_valid = False
                break
    if not source_copies_valid:
        errors.append("portable copies differ from audited analysis source")
    try:
        tensor_manifest = _read_json(root / "selected_tensors/manifest.json")
        tensor_entries = tensor_manifest.get("tensors")
        tensor_manifest_valid = isinstance(tensor_entries, list)
        if not tensor_manifest_valid:
            errors.append("selected tensor manifest lacks a tensors list")
            tensor_entries = []
        for entry in tensor_entries:
            if not isinstance(entry, dict):
                tensor_manifest_valid = False
                continue
            relative = entry.get("path")
            expected_hash = entry.get("sha256")
            expected_size = entry.get("size")
            exported = entry.get("exported")
            if (
                not isinstance(relative, str)
                or not isinstance(expected_hash, str)
                or len(expected_hash) != 64
                or not isinstance(expected_size, int)
                or isinstance(expected_size, bool)
                or type(exported) is not bool
            ):
                tensor_manifest_valid = False
                continue
            if exported:
                path = _safe_relative_path(root, relative)
                tensor_manifest_valid = tensor_manifest_valid and (
                    entry.get("storage_scope") == "portable_export"
                    and path.is_file()
                    and path.stat().st_size == expected_size
                    and _sha256(path) == expected_hash
                )
            else:
                tensor_manifest_valid = tensor_manifest_valid and (
                    entry.get("storage_scope") == "analysis_run_local"
                    and relative.startswith("analysis_runs/")
                )
        _require_no_absolute_strings(tensor_manifest, "selected_tensors.manifest")
        source_manifest_hash = tensor_manifest.get("source_manifest_sha256")
        declared_source_hashes = provenance.get("source_analysis_sha256", {})
        if not isinstance(
            declared_source_hashes, dict
        ) or source_manifest_hash != declared_source_hashes.get("selected_tensors/manifest.json"):
            tensor_manifest_valid = False
        if not tensor_manifest_valid:
            errors.append("selected tensor manifest entries are not portable or complete")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        tensor_manifest_valid = False
        errors.append(f"invalid selected tensor manifest: {error}")

    hashes = _tree_file_hashes(root, exclude={"audit/m2_export.json"})
    analysis_id = provenance.get("analysis_id")
    audit = {
        "schema_version": 1,
        "profile": "m2-portable-export",
        "analysis_id": analysis_id,
        "passed": not errors,
        "checks": {
            "required_files_present": not missing,
            "analysis_audit_passed": not any("analysis audit" in error for error in errors),
            "function_artifacts_valid": not any("function artifacts" in error for error in errors),
            "provenance_portable": not any("provenance" in error for error in errors),
            "frozen_m1_inventories_present": frozen_valid,
            "source_analysis_copies_match": source_copies_valid,
            "selected_tensor_manifest_valid": tensor_manifest_valid,
            "no_m3_plus_artifacts": not forbidden,
            "json_csv_values_finite": not any(
                "invalid JSON" in error or "invalid CSV" in error for error in errors
            ),
        },
        "function_metric_record_count": len(function_records),
        "portable_file_count_excluding_audit": len(hashes),
        "portable_file_sha256_excluding_audit": hashes,
        "errors": errors,
        "audited_at": datetime.now(timezone.utc).isoformat(),
    }
    if write:
        write_json(root / "audit/m2_export.json", audit)
    return audit


def export_m2_results(analysis_dir: str | Path, output_dir: str | Path) -> Path:
    """Atomically export a passing M2 analysis without modifying its source."""
    source = Path(analysis_dir).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite M2 export: {destination}")
    if not source.is_dir():
        raise FileNotFoundError(f"M2 analysis directory does not exist: {source}")
    analysis_audit = _read_json(source / "audit/m2_analysis.json")
    _verify_analysis_audit(source, analysis_audit)
    source_provenance = _read_json(source / "provenance.json")
    analysis_id = source_provenance.get("analysis_id", source.name)
    if not isinstance(analysis_id, str) or not analysis_id:
        raise ValueError("M2 provenance has an invalid analysis_id")
    audited_id = analysis_audit.get("analysis_id")
    if audited_id is not None and audited_id != analysis_id:
        raise ValueError("M2 provenance and analysis audit IDs differ")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        copied_hashes = _copy_analysis_files(source, temporary)
        _prepare_portable_tensors(source, temporary, analysis_id)
        function_records = _read_jsonl(temporary / "m2b/function_metrics.jsonl")
        function_fields, function_rows = _read_csv(temporary / "m2b/function_metrics.csv")
        _require_finite_csv(function_rows, temporary / "m2b/function_metrics.csv")
        _validate_function_records(function_records, function_fields, function_rows)
        offset_steps, offset_profiles = _load_offset_profiles(temporary / "m2b/offset_profiles.npz")
        if list(map(int, offset_steps)) != [int(record["step"]) for record in function_records]:
            raise ValueError("offset profile and function metric steps differ")
        function_events = _read_json(temporary / "m2b/function_events.json")
        provenance = _portable_provenance(
            source_provenance,
            analysis_id,
            copied_hashes,
            function_events,
            len(function_records),
        )
        write_json(temporary / "provenance.json", provenance)
        _write_readme(temporary / "README.md", analysis_id, function_events, len(function_records))
        _plot_all(
            temporary,
            analysis_id,
            function_records,
            offset_steps,
            offset_profiles,
            function_events,
            provenance,
            temporary / "m2a/episode_state_index.csv",
            temporary / "m2b/episode_function_deltas.csv",
            temporary / "m2a/checkpoint_validation.jsonl",
            temporary / "context/m1_scalars.jsonl",
            temporary / "context/collapse_episodes.json",
        )
        export_audit = audit_m2_export(temporary, write=True)
        if export_audit.get("passed") is not True:
            raise ValueError(f"portable M2 export audit failed: {export_audit['errors']}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination
