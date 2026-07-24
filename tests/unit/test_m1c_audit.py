from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import transgrokking.metrics.audit_m1c as audit_module
from transgrokking.config import config_from_dict, load_config
from transgrokking.metrics.stability import dump_measurement_config, load_measurement_config
from transgrokking.training.checkpoint import CHECKPOINT_SCHEMA_VERSION
from transgrokking.utils.atomic import write_json, write_json_lines, write_yaml


def _checkpoint_manifest(run_dir: Path, steps: list[int]) -> None:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    entries = []
    for step in steps:
        name = f"step_{step:06d}.pt"
        (checkpoint_dir / name).write_bytes(b"fixture")
        entries.append({"step": step, "path": name})
    write_json(
        checkpoint_dir / "manifest.json",
        {"schema_version": 2, "checkpoints": entries},
    )


def _groups() -> list[dict[str, Any]]:
    return [
        {
            "group_name": "decay",
            "learning_rate": 0.001,
            "weight_decay": 0.5,
            "parameter_names": ["matrix.weight"],
        },
        {
            "group_name": "no_decay",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "parameter_names": ["matrix.bias"],
        },
    ]


def _signature() -> list[dict[str, Any]]:
    return [
        {
            "group_name": group["group_name"],
            "parameter_names": group["parameter_names"],
            "weight_decay": group["weight_decay"],
            "learning_rate": group["learning_rate"],
        }
        for group in _groups()
    ]


def _optimization_record(step: int) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": 1,
        "step": step,
        "parameter_tensor_count": 2,
        "parameter_element_count": 2,
        "updated_parameter_tensor_count": 2,
        "updated_parameter_element_count": 2,
    }
    for field in (
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
    ):
        record[field] = 0.0
    return record


def _frozen_manifest(tmp_path: Path) -> Path:
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    entries = []
    for index in range(23):
        relative = f"file_{index:02d}.txt"
        path = frozen / relative
        path.write_text(f"frozen-{index}\n", encoding="utf-8")
        entries.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = tmp_path / "frozen_manifest.json"
    write_json(
        manifest,
        {"schema_version": 1, "root": str(frozen), "files": entries},
    )
    return manifest


def _make_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    measurement = load_measurement_config("configs/analysis/m1c_stability.yaml")
    source = measurement.source
    runs = tmp_path / "runs"
    canonical = runs / source.canonical_run_id
    terminal = runs / "terminal_m1c"
    for run in (canonical, terminal):
        (run / "metrics").mkdir(parents=True)
        (run / "tensors").mkdir()
        (run / "figures").mkdir()

    extension = load_config("configs/ce_reference_extend_50000.yaml")
    canonical_raw = extension.to_dict()
    canonical_raw["optimization"]["max_steps"] = 20_000
    canonical_config = config_from_dict(canonical_raw)
    write_yaml(canonical / "config.resolved.yaml", canonical_config.to_dict())
    write_yaml(terminal / "config.resolved.yaml", extension.to_dict())
    dump_measurement_config(measurement, terminal / "measurement.resolved.yaml")

    split_hash = source.split_hash
    write_json(
        canonical / "metadata.json",
        {
            "schema_version": 2,
            "run_id": canonical.name,
            "scientific_config_hash": source.scientific_config_hash,
            "split_hash": split_hash,
            "optimizer_parameter_groups": _groups(),
        },
    )
    canonical_checkpoint = canonical / "checkpoints" / "step_020000.pt"
    write_json(
        terminal / "metadata.json",
        {
            "schema_version": 2,
            "run_id": terminal.name,
            "parent_run_id": canonical.name,
            "parent_checkpoint": str(canonical_checkpoint.resolve()),
            "parent_global_step": 20_000,
            "scientific_config_hash": source.scientific_config_hash,
            "split_hash": split_hash,
            "measurement_config_hash": measurement.measurement_hash(),
            "git_commit": "a" * 40,
            "git_worktree_clean": True,
            "extension_origin_run_id": canonical.name,
            "extension_origin_checkpoint": str(canonical_checkpoint.resolve()),
            "extension_origin_step": 20_000,
            "diagnostics_start_step": 20_000,
            "optimizer_parameter_groups": _groups(),
            "formal_run": True,
            "doctor": {"device_name": "NVIDIA GeForce RTX 4060 Laptop GPU"},
            "final_global_step": 50_000,
            "max_memory_allocated": 1024,
            "max_memory_reserved": 2048,
        },
    )
    write_json(canonical / "status.json", {"state": "completed", "global_step": 20_000})
    write_json(
        terminal / "status.json",
        {
            "state": "completed",
            "global_step": 50_000,
            "max_memory_allocated": 1024,
            "max_memory_reserved": 2048,
        },
    )
    _checkpoint_manifest(canonical, [20_000])
    _checkpoint_manifest(terminal, list(range(20_000, 50_001, 100)))

    scalars = [
        {
            "schema_version": 1,
            "step": step,
            "congruence_loss": 0.0,
            "train_cross_entropy": 0.0,
            "test_cross_entropy": 0.0,
            "train_accuracy": 1.0,
            "test_accuracy": 1.0,
            "parameter_norm_total": 1.0,
            "parameter_group_norm_decay": 1.0,
            "parameter_group_norm_no_decay": 0.0,
        }
        for step in range(50, 50_001, 50)
    ]
    offsets = [
        {
            "schema_version": 1,
            "step": step,
            "split": split,
            "modulus": 97,
            "counts": [0] * 97,
        }
        for step in range(50, 50_001, 50)
        for split in ("train", "test")
    ]
    optimization = [_optimization_record(step) for step in range(20_050, 50_001, 50)]
    scalar_path = terminal / "metrics" / "scalars.jsonl"
    write_json_lines(scalar_path, scalars)
    write_json_lines(terminal / "metrics" / "error_offsets.jsonl", offsets)
    write_json_lines(terminal / "metrics" / "optimization.jsonl", optimization)
    write_json_lines(
        canonical / "metrics" / "scalars.jsonl",
        [record for record in scalars if int(record["step"]) <= 20_000],
    )
    write_json_lines(
        canonical / "metrics" / "error_offsets.jsonl",
        [record for record in offsets if int(record["step"]) <= 20_000],
    )
    events = {
        "schema_version": 1,
        "run_id": terminal.name,
        "modulus": 97,
        "eval_interval": 50,
        "last_evaluated_step": 50_000,
        "t_fit": {
            "status": "reached",
            "event_step": measurement.frozen_events.t_fit,
            "detected_at_evaluation_step": measurement.frozen_events.t_fit_detected_at,
        },
        "t_grok50": {
            "status": "reached",
            "event_step": measurement.frozen_events.t_grok50,
            "detected_at_evaluation_step": measurement.frozen_events.t_grok50_detected_at,
        },
        "t_grok99": {
            "status": "reached",
            "event_step": measurement.frozen_events.t_grok99,
            "detected_at_evaluation_step": measurement.frozen_events.t_grok99_detected_at,
        },
    }
    write_json(terminal / "metrics" / "events.json", events)

    scalar_sha = hashlib.sha256(scalar_path.read_bytes()).hexdigest()
    stability, episodes = audit_module.summarize_stability(
        scalars,
        run_id=terminal.name,
        parent_run_id=canonical.name,
        eval_interval=50,
        frozen_events=measurement.frozen_events,
        config=measurement.stability,
        source_scalars_sha256=scalar_sha,
    )
    stability["measurement_config_hash"] = measurement.measurement_hash()
    episodes["measurement_config_hash"] = measurement.measurement_hash()
    write_json(terminal / "metrics" / "stability.json", stability)
    write_json(terminal / "metrics" / "collapse_episodes.json", episodes)

    def fake_checkpoint(path: str | Path, *_args, **_kwargs) -> dict[str, Any]:
        step = int(Path(path).stem.removeprefix("step_"))
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "global_step": step,
            "scientific_config_hash": source.scientific_config_hash,
            "split_hash": split_hash,
            "optimizer_group_signature": _signature(),
            "optimizer_type": "adamw",
        }

    def fake_evaluate(_run_dir: str | Path) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "step": 50_000,
            **{
                key: value
                for key, value in scalars[-1].items()
                if key not in {"schema_version", "step"}
            },
            "error_offsets": {"train": [0] * 97, "test": [0] * 97},
        }

    real_sha256 = audit_module._sha256

    def fixture_sha256(path: str | Path) -> str:
        candidate = Path(path).resolve()
        if candidate == canonical_checkpoint.resolve():
            return source.canonical_checkpoint_sha256
        return real_sha256(candidate)

    monkeypatch.setattr(audit_module, "_sha256", fixture_sha256)
    monkeypatch.setattr(audit_module, "read_checkpoint", fake_checkpoint)
    monkeypatch.setattr(audit_module, "evaluate_run_checkpoint", fake_evaluate)
    return terminal, _frozen_manifest(tmp_path)


def test_m1c_audit_accepts_complete_terminal_fixture_and_writes_report(
    tmp_path, monkeypatch
) -> None:
    terminal, frozen_manifest = _make_run(tmp_path, monkeypatch)

    report = audit_module.audit_m1c_extension(terminal, frozen_manifest)

    assert report["passed"] is True
    assert report["failed_checks"] == []
    assert report["evaluation_count"] == 1000
    assert report["optimization_diagnostic_count"] == 600
    persisted = json.loads((terminal / "audit" / "m1c_extension.json").read_text(encoding="utf-8"))
    assert persisted == report


def test_m1c_audit_rejects_any_frozen_result_byte_change(tmp_path, monkeypatch) -> None:
    terminal, frozen_manifest = _make_run(tmp_path, monkeypatch)
    manifest = json.loads(frozen_manifest.read_text(encoding="utf-8"))
    frozen_root = Path(manifest["root"])
    (frozen_root / manifest["files"][0]["path"]).write_bytes(b"changed")

    report = audit_module.audit_m1c_extension(terminal, frozen_manifest)

    assert report["passed"] is False
    assert report["checks"]["frozen_m1b_results"] is False
    assert any("frozen file" in error for error in report["frozen_manifest_details"]["errors"])


def test_m1c_audit_rejects_scientific_change_beyond_max_steps(tmp_path, monkeypatch) -> None:
    terminal, frozen_manifest = _make_run(tmp_path, monkeypatch)
    raw = load_config(terminal / "config.resolved.yaml").to_dict()
    raw["optimization"]["learning_rate"] = 0.002
    write_yaml(terminal / "config.resolved.yaml", raw)

    report = audit_module.audit_m1c_extension(terminal, frozen_manifest)

    assert report["passed"] is False
    assert report["checks"]["config_only_max_steps_changed"] is False
    assert report["checks"]["scientific_config_hash"] is False
