"""Reproducible, no-smoothing export of audited M1-C evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transgrokking.training.artifacts import (
    load_error_offset_records,
    load_optimization_records,
    load_scalar_records,
)
from transgrokking.utils.atomic import replace_with_retry, write_json

FIGURE_NAMES = (
    "loss_linear",
    "loss_log",
    "accuracy",
    "margin_train",
    "margin_test",
    "parameter_norm_groups",
    "parameter_norm_modules",
    "optimization_update_norms",
    "adam_moments",
    "collapse_timeline",
    "stable_window_timeline",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_audited_source_hashes(source: Path, audit: dict[str, Any]) -> None:
    hashes = audit.get("audited_source_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("M1-C audit lacks audited source SHA-256 values")
    required = {
        "config.resolved.yaml",
        "measurement.resolved.yaml",
        "metadata.json",
        "status.json",
        "metrics/scalars.jsonl",
        "metrics/error_offsets.jsonl",
        "metrics/events.json",
        "metrics/stability.json",
        "metrics/collapse_episodes.json",
        "metrics/optimization.jsonl",
        "checkpoints/manifest.json",
    }
    if not required.issubset(hashes) or not any(
        key.startswith("checkpoints/step_") and key.endswith(".pt") for key in hashes
    ):
        raise ValueError("M1-C audit source hash set is incomplete")
    for relative, expected in hashes.items():
        if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("M1-C audit contains an invalid source hash entry")
        path = (source / relative).resolve()
        try:
            path.relative_to(source)
        except ValueError as error:
            raise ValueError(f"audited source path escapes run directory: {relative}") from error
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"audited source changed after audit: {relative}")


def _columns(records: list[dict[str, Any]], predicate) -> list[str]:
    if not records:
        return []
    return [key for key in records[0] if key != "step" and predicate(key)]


def _write_csv(path: Path, records: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", *columns], extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "step": record["step"],
                    **{
                        column: "" if record.get(column) is None else record.get(column)
                        for column in columns
                    },
                }
            )


def _event_step(events: dict[str, Any], name: str) -> int | None:
    event = events.get(name)
    if not isinstance(event, dict) or event.get("status") != "reached":
        return None
    value = event.get("event_step")
    return int(value) if type(value) is int else None


def _episode_steps(episodes: dict[str, Any]) -> dict[str, list[int]]:
    result = {"onset": [], "trough": [], "recovery": []}
    values = episodes.get("episodes", [])
    if not isinstance(values, list):
        raise ValueError("collapse_episodes.episodes must be a list")
    for episode in values:
        if not isinstance(episode, dict):
            continue
        episode_type = episode.get("episode_type")
        if episode_type not in {"train", "test"}:
            continue
        for source, target in (
            ("onset_step", "onset"),
            ("trough_step", "trough"),
            (f"{episode_type}_recovery_step", "recovery"),
        ):
            value = episode.get(source)
            if type(value) is int:
                result[target].append(value)
    return {key: sorted(set(values)) for key, values in result.items()}


def _plot_all(
    output: Path,
    run_id: str,
    scalars: list[dict[str, Any]],
    optimization: list[dict[str, Any]],
    events: dict[str, Any],
    stability: dict[str, Any],
    episodes: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output / "figures"
    figure_dir.mkdir()
    steps = [int(record["step"]) for record in scalars]
    short_id = run_id[-8:]
    first_events = {name: _event_step(events, name) for name in ("t_fit", "t_grok50", "t_grok99")}
    collapse_steps = _episode_steps(episodes)

    def finish(
        name: str,
        title: str,
        *,
        ylabel: str,
        zero_line: bool = False,
        collapse_annotations: bool = True,
    ) -> None:
        if zero_line:
            plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        for event_name, step in first_events.items():
            if step is not None:
                plt.axvline(step, linestyle="--", linewidth=0.8, label=event_name)
        if collapse_annotations:
            colors = {"onset": "red", "trough": "purple", "recovery": "green"}
            for kind, values in collapse_steps.items():
                for index, step in enumerate(values):
                    plt.axvline(
                        step,
                        color=colors[kind],
                        linestyle=":",
                        linewidth=0.55,
                        alpha=0.35,
                        label=kind if index == 0 else None,
                    )
        plt.xlabel("optimizer step")
        plt.ylabel(ylabel)
        plt.title(f"{title} — {short_id}")
        plt.legend(fontsize="small", ncols=2)
        plt.tight_layout()
        for suffix in ("png", "svg"):
            plt.savefig(figure_dir / f"{name}.{suffix}", dpi=180)
        plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(steps, [row["train_cross_entropy"] for row in scalars], label="train CE")
    plt.plot(steps, [row["test_cross_entropy"] for row in scalars], label="test CE")
    finish("loss_linear", "CE loss (raw)", ylabel="cross-entropy")

    plt.figure(figsize=(10, 5))
    plt.plot(steps, [row["train_cross_entropy"] for row in scalars], label="train CE")
    plt.plot(steps, [row["test_cross_entropy"] for row in scalars], label="test CE")
    plt.yscale("log")
    finish("loss_log", "CE loss (raw, log scale)", ylabel="cross-entropy")

    plt.figure(figsize=(10, 5))
    plt.plot(steps, [row["train_accuracy"] for row in scalars], label="train accuracy")
    plt.plot(steps, [row["test_accuracy"] for row in scalars], label="test accuracy")
    plt.axhline(1.0 / 97.0, color="gray", linestyle=":", label="random 1/97")
    finish("accuracy", "Accuracy (raw)", ylabel="accuracy")

    for split in ("train", "test"):
        plt.figure(figsize=(10, 5))
        columns = [
            key
            for key in scalars[0]
            if key.startswith(f"{split}_margin_")
            and key
            in {
                f"{split}_margin_mean",
                f"{split}_margin_min",
                f"{split}_margin_q05",
                f"{split}_margin_median",
                f"{split}_margin_q95",
            }
        ]
        for column in columns:
            plt.plot(steps, [row[column] for row in scalars], label=column)
        finish(
            f"margin_{split}",
            f"{split.title()} margins (raw)",
            ylabel="logit margin",
            zero_line=True,
        )

    plt.figure(figsize=(10, 5))
    for column in (
        "parameter_norm_total",
        "parameter_group_norm_decay",
        "parameter_group_norm_no_decay",
    ):
        plt.plot(steps, [row[column] for row in scalars], label=column)
    finish("parameter_norm_groups", "Parameter L2 norms", ylabel="L2 / Frobenius norm")

    plt.figure(figsize=(11, 6))
    module_columns = [
        key
        for key in scalars[0]
        if key.startswith("parameter_norm_")
        and key
        not in {
            "parameter_norm_total",
            "parameter_group_norm_decay",
            "parameter_group_norm_no_decay",
            "parameter_norm_final_norm",
        }
    ]
    for column in module_columns:
        values = [row.get(column) for row in scalars]
        if any(value is not None for value in values):
            plt.plot(steps, values, label=column)
    finish("parameter_norm_modules", "Module parameter norms", ylabel="L2 / Frobenius norm")

    opt_steps = [int(row["step"]) for row in optimization]
    plt.figure(figsize=(10, 5))
    for column in ("total_update_l2", "data_update_l2", "decay_update_l2"):
        plt.plot(opt_steps, [row[column] for row in optimization], label=column)
    plt.yscale("log")
    finish(
        "optimization_update_norms",
        "AdamW update norms (extension only)",
        ylabel="L2 / Frobenius norm",
    )

    plt.figure(figsize=(10, 5))
    for column in (
        "adam_first_moment_l2",
        "adam_second_moment_mean",
        "adam_second_moment_rms",
        "adam_second_moment_max",
    ):
        plt.plot(opt_steps, [row[column] for row in optimization], label=column)
    plt.yscale("log")
    finish("adam_moments", "Adam moment summaries (extension only)", ylabel="value")

    plt.figure(figsize=(11, 6))
    plt.plot(steps, [row["train_accuracy"] for row in scalars], label="train accuracy")
    plt.plot(steps, [row["test_accuracy"] for row in scalars], label="test accuracy")
    styles = {"onset": "-", "trough": ":", "recovery": "--"}
    colors = {"onset": "red", "trough": "purple", "recovery": "green"}
    for kind, values in collapse_steps.items():
        for index, step in enumerate(values):
            plt.axvline(
                step,
                color=colors[kind],
                linestyle=styles[kind],
                linewidth=0.7,
                alpha=0.55,
                label=kind if index == 0 else None,
            )
    finish(
        "collapse_timeline",
        "Collapse episodes (raw)",
        ylabel="accuracy",
        collapse_annotations=False,
    )

    plt.figure(figsize=(11, 4))
    above = [1.0 if row["test_accuracy"] >= 0.99 else 0.0 for row in scalars]
    plt.step(steps, above, where="post", label="test accuracy >= 0.99")
    stable = stability.get("t_stable99", {})
    stable_step = (
        stable.get("event_step")
        if isinstance(stable, dict) and stable.get("status") == "reached"
        else None
    )
    if type(stable_step) is int:
        plt.axvline(stable_step, color="green", linewidth=1.2, label="t_stable99")
    else:
        plt.text(0.99, 0.08, "t_stable99: not reached", ha="right", transform=plt.gca().transAxes)
    finish(
        "stable_window_timeline",
        "Stable-window evidence (raw)",
        ylabel="threshold indicator",
    )


def _ensure_expected_files(output: Path) -> None:
    expected = {
        output / "README.md",
        output / "provenance.json",
        output / "config.resolved.yaml",
        output / "measurement.resolved.yaml",
        output / "scalars.jsonl",
        output / "error_offsets.jsonl",
        output / "events.json",
        output / "stability.json",
        output / "collapse_episodes.json",
        output / "optimization.jsonl",
        output / "audit" / "m1c_extension.json",
        output / "loss_curve.csv",
        output / "accuracy_curve.csv",
        output / "margin_curve.csv",
        output / "parameter_norm_curve.csv",
        output / "optimization_curve.csv",
        *{
            output / "figures" / f"{name}.{suffix}"
            for name in FIGURE_NAMES
            for suffix in ("png", "svg")
        },
    }
    missing = sorted(str(path.relative_to(output)) for path in expected if not path.is_file())
    if missing:
        raise ValueError(f"M1-C export is incomplete: {missing}")


def export_m1c_results(run_dir: str | Path, output_dir: str | Path) -> Path:
    """Export an audited M1-C run without modifying source artifacts."""
    source = Path(run_dir).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite M1-C export: {destination}")
    audit_path = source / "audit" / "m1c_extension.json"
    audit = _read_json(audit_path)
    if audit.get("passed") is not True or audit.get("run_id") != source.name:
        raise ValueError("M1-C export requires a passing audit for the selected run")
    _verify_audited_source_hashes(source, audit)

    scalar_path = source / "metrics" / "scalars.jsonl"
    offset_path = source / "metrics" / "error_offsets.jsonl"
    optimization_path = source / "metrics" / "optimization.jsonl"
    scalars = load_scalar_records(scalar_path)
    load_error_offset_records(offset_path)
    optimization = load_optimization_records(optimization_path)
    events = _read_json(source / "metrics" / "events.json")
    stability = _read_json(source / "metrics" / "stability.json")
    episodes = _read_json(source / "metrics" / "collapse_episodes.json")
    metadata = _read_json(source / "metadata.json")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        copies = {
            "config.resolved.yaml": source / "config.resolved.yaml",
            "measurement.resolved.yaml": source / "measurement.resolved.yaml",
            "scalars.jsonl": scalar_path,
            "error_offsets.jsonl": offset_path,
            "events.json": source / "metrics" / "events.json",
            "stability.json": source / "metrics" / "stability.json",
            "collapse_episodes.json": source / "metrics" / "collapse_episodes.json",
            "optimization.jsonl": optimization_path,
            "audit/m1c_extension.json": audit_path,
        }
        for relative, source_path in copies.items():
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target)

        loss_columns = ["train_cross_entropy", "test_cross_entropy"]
        accuracy_columns = ["train_accuracy", "test_accuracy"]
        margin_columns = _columns(
            scalars,
            lambda key: key.startswith("train_margin_") or key.startswith("test_margin_"),
        )
        parameter_columns = _columns(
            scalars,
            lambda key: (
                key.startswith("parameter_norm_") or key.startswith("parameter_group_norm_")
            ),
        )
        optimization_columns = [
            key for key in optimization[0] if key not in {"schema_version", "step"}
        ]
        _write_csv(
            temporary / "loss_curve.csv",
            scalars,
            loss_columns,
        )
        _write_csv(
            temporary / "accuracy_curve.csv",
            scalars,
            accuracy_columns,
        )
        _write_csv(
            temporary / "margin_curve.csv",
            scalars,
            margin_columns,
        )
        _write_csv(
            temporary / "parameter_norm_curve.csv",
            scalars,
            parameter_columns,
        )
        _write_csv(
            temporary / "optimization_curve.csv",
            optimization,
            optimization_columns,
        )
        _plot_all(
            temporary,
            source.name,
            scalars,
            optimization,
            events,
            stability,
            episodes,
        )
        source_hashes = {relative: _sha256(path) for relative, path in sorted(copies.items())}
        source_prefix = Path("runs") / source.name
        provenance = {
            "schema_version": 1,
            "canonical_parent_run_id": audit.get("canonical_parent_run_id"),
            "m1c_child_run_id": source.name,
            "source_git_commit": metadata.get("git_commit"),
            "parent_checkpoint": (
                Path("runs")
                / str(audit.get("canonical_parent_run_id"))
                / "checkpoints"
                / f"step_{int(metadata['diagnostics_start_step']):06d}.pt"
            ).as_posix(),
            "parent_step": metadata.get("diagnostics_start_step"),
            "final_step": int(scalars[-1]["step"]),
            "scientific_config_hash": metadata.get("scientific_config_hash"),
            "split_hash": metadata.get("split_hash"),
            "measurement_config_hash": audit.get("measurement_config_hash"),
            "evaluation_interval": audit.get("evaluation_interval"),
            "checkpoint_interval": audit.get("checkpoint_interval"),
            "lineage_git_commits": audit.get("lineage_git_commits"),
            "parameter_group_signature": audit.get("parameter_group_signature"),
            "scalar_count": len(scalars),
            "optimization_diagnostic_count": len(optimization),
            "source_file_sha256": source_hashes,
            "source_files": {
                "config_resolved": (source_prefix / "config.resolved.yaml").as_posix(),
                "measurement_resolved": (source_prefix / "measurement.resolved.yaml").as_posix(),
                "scalars": (source_prefix / "metrics/scalars.jsonl").as_posix(),
                "error_offsets": (source_prefix / "metrics/error_offsets.jsonl").as_posix(),
                "events": (source_prefix / "metrics/events.json").as_posix(),
                "stability": (source_prefix / "metrics/stability.json").as_posix(),
                "collapse_episodes": (source_prefix / "metrics/collapse_episodes.json").as_posix(),
                "optimization": (source_prefix / "metrics/optimization.jsonl").as_posix(),
                "audit": (source_prefix / "audit/m1c_extension.json").as_posix(),
            },
            "csv_fields": {
                "loss_curve.csv": ["step", *loss_columns],
                "accuracy_curve.csv": ["step", *accuracy_columns],
                "margin_curve.csv": ["step", *margin_columns],
                "parameter_norm_curve.csv": ["step", *parameter_columns],
                "optimization_curve.csv": ["step", *optimization_columns],
            },
            "figure_files": [
                f"figures/{name}.{suffix}" for name in FIGURE_NAMES for suffix in ("png", "svg")
            ],
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "smoothing": False,
            "interpolation": False,
            "outlier_deletion": False,
            "missing_step_imputation": False,
        }
        write_json(temporary / "provenance.json", provenance)
        readme = (
            "# M1-C extended CE-reference evidence\n\n"
            f"- Canonical parent: `{provenance['canonical_parent_run_id']}`\n"
            f"- Terminal child: `{source.name}`\n"
            f"- Final step: `{provenance['final_step']}`\n"
            f"- Final state: `{stability.get('final_state')}`\n"
            f"- Audit passed: `{audit.get('passed')}`\n\n"
            "All curves use the original evaluation steps. No smoothing, interpolation, "
            "outlier deletion, or missing-step imputation was applied. Optimization "
            "diagnostics begin after step 20000. These files support behavior and optimizer "
            "measurement only; M2 mechanism analysis has not been performed.\n"
        )
        (temporary / "README.md").write_text(readme, encoding="utf-8")
        _ensure_expected_files(temporary)
        replace_with_retry(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination
