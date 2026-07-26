from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from transgrokking.metrics import audit_m2
from transgrokking.reporting import m2
from transgrokking.reporting.m2 import FIGURE_NAMES, audit_m2_export, export_m2_results
from transgrokking.utils.atomic import write_json, write_json_lines


def _function_record(step: int, *, regular: bool = True) -> dict[str, object]:
    return {
        "schema_version": 1,
        "step": step,
        "state_source": "checkpoint" if regular else "deterministic_replay",
        "run_id": "child",
        "checkpoint_sha256": "a" * 64,
        "semantic_state_sha256": "9" * 64,
        "centered_logit_frobenius": 3.0,
        "centered_logit_rms": 0.3,
        "equivariant_energy": 8.0,
        "residual_energy": 1.0,
        "D_eq": 1.0 / 9.0,
        "Gamma": 1.0,
        "I": 0.25,
        "Gamma_minus_I": 0.75,
        "Gamma_over_logit_rms": 10.0 / 3.0,
        "I_over_logit_rms": 5.0 / 6.0,
        "Gamma_over_parameter_l2": 0.2,
        "train_projected_ce": 0.2,
        "test_projected_ce": 0.2,
        "full_projected_ce": 0.2,
        "train_projected_accuracy": 1.0,
        "test_projected_accuracy": 1.0,
        "full_projected_accuracy": 1.0,
        "train_entropy_mean": 0.4,
        "test_entropy_mean": 0.4,
        "full_entropy_mean": 0.4,
        "train_entropy_normalized": 0.1,
        "test_entropy_normalized": 0.1,
        "full_entropy_normalized": 0.1,
        "train_accuracy": 1.0,
        "test_accuracy": 0.9,
        "full_cross_entropy": 0.3,
        "full_accuracy": 0.95,
        "full_margin_mean": 0.5,
        "full_margin_min": -0.1,
        "parameter_norm_total": 5.0,
        "parameter_group_norm_decay": 4.0,
        "parameter_group_norm_no_decay": 3.0,
        "is_regular_grid": regular,
        "state_roles": ["test_001:onset"] if step == 100 else [],
        "replay_source_step": None if regular else step - 50,
        "replay_updates": 0 if regular else 50,
        "full_logits_shape": [3, 3, 3],
        "forward_dtype": "float32",
        "reduction_dtype": "float64",
    }


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            row = dict(record)
            if isinstance(row.get("state_roles"), list):
                row["state_roles"] = json.dumps(row["state_roles"])
            writer.writerow(row)


def _write_passing_analysis_audit(source: Path) -> None:
    hashed = {
        relative: hashlib.sha256((source / relative).read_bytes()).hexdigest()
        for relative in m2.REQUIRED_ANALYSIS_FILES
        if relative != "audit/m2_analysis.json"
    }
    write_json(
        source / "audit/m2_analysis.json",
        {
            "schema_version": 1,
            "analysis_id": "m2-test",
            "passed": True,
            "audited_source_sha256": hashed,
        },
    )


def _analysis(tmp_path: Path) -> Path:
    source = tmp_path / "analysis_runs" / "m2-test"
    (source / "m2a").mkdir(parents=True)
    (source / "m2b").mkdir()
    (source / "audit").mkdir()
    (source / "selected_tensors").mkdir()
    (source / "context").mkdir()
    (source / "analysis_config.resolved.yaml").write_text(
        "schema_version: 1\nanalysis_batch_size: 8\n", encoding="utf-8"
    )
    write_json(
        source / "provenance.json",
        {
            "analysis_id": "m2-test",
            "source_git_commit": "abc123",
            "scientific_config_hash": "b" * 64,
            "split_hash": "c" * 64,
            "analysis_config_hash": "d" * 64,
            "modulus": 3,
            "source_run_ids": ["root", "parent", "child"],
            "hardware": {
                "prefix": "E:\\Workplace\\Trans\\env",
                "expected_prefix": "E:\\Workplace\\Trans\\env",
                "device_name": "NVIDIA GeForce RTX 4060 Laptop GPU",
            },
            "behavior_events": {"t_fit": 100, "t_grok50": 200, "t_grok99": 300},
            "frozen_results_sha256": {
                "results/m1": {"README.md": "e" * 64},
                "results/m1_extended": {"README.md": "f" * 64},
            },
            "source_inventories_before": {
                "results/m1": [{"path": "README.md", "size": 12, "sha256": "e" * 64}],
                "results/m1_extended": [{"path": "README.md", "size": 34, "sha256": "f" * 64}],
            },
        },
    )
    write_json(
        source / "status.json",
        {
            "m2a_status": "completed",
            "m2b_status": "completed",
            "analysis_status": "completed",
            "export_status": "not_started",
        },
    )
    for name in ("checkpoint_files.csv", "checkpoint_index.csv", "checkpoint_aliases.csv"):
        (source / name).write_text("step,run_id\n0,root\n", encoding="utf-8")

    write_json_lines(
        source / "context/m1_scalars.jsonl",
        [
            {"step": 50, "train_accuracy": 1.0, "test_accuracy": 0.8},
            {"step": 100, "train_accuracy": 1.0, "test_accuracy": 0.9},
        ],
    )
    write_json_lines(
        source / "context/m1_optimization.jsonl",
        [{"step": 100, "gradient_l2": 0.2}],
    )
    write_json(
        source / "context/collapse_episodes.json",
        {"schema_version": 1, "episodes": []},
    )
    write_json(
        source / "context/events.json",
        {"t_fit": {"status": "reached", "event_step": 100}},
    )

    validations = [
        {
            "target_step": 0,
            "resolution": "exact_checkpoint",
            "validation_status": "passed",
            "ce_abs_diff": 0.0,
        },
        {
            "target_step": 50,
            "resolution": "deterministic_replay",
            "validation_status": "passed",
            "ce_abs_diff": 1.0e-8,
        },
    ]
    write_json_lines(source / "m2a/checkpoint_validation.jsonl", validations)
    _write_csv(source / "m2a/checkpoint_validation.csv", validations)
    episode_states = [
        {"episode_id": "test_001", "state_role": "onset", "target_step": 100},
        {
            "episode_id": "test_001",
            "state_role": "recovery_confirmed",
            "target_step": 200,
        },
    ]
    _write_csv(source / "m2a/episode_state_index.csv", episode_states)
    write_json_lines(
        source / "m2a/replay_bridge.jsonl",
        [{"target_step": 50, "endpoint_step": 100, "passed": True}],
    )

    records = [_function_record(0), _function_record(50, regular=False), _function_record(100)]
    write_json_lines(source / "m2b/function_metrics.jsonl", records)
    _write_csv(source / "m2b/function_metrics.csv", records)
    np.savez_compressed(
        source / "m2b/offset_profiles.npz",
        steps=np.array([0, 50, 100]),
        offset_profiles=np.array([[0.2, -0.1, -0.1]] * 3),
    )
    write_json(
        source / "m2b/function_events.json",
        {
            "t_alg": {"status": "reached", "event_step": 0},
            "t_dom": {"status": "reached", "event_step": 100},
        },
    )
    deltas = [
        {
            "episode_id": "test_001",
            "Gamma_delta": -0.2,
            "I_delta": 0.1,
            "D_eq_delta": 0.03,
        }
    ]
    _write_csv(source / "m2b/episode_function_deltas.csv", deltas)
    write_json(
        source / "selected_tensors/manifest.json",
        {
            "schema_version": 1,
            "export_limit_mib": 10,
            "total_size": 0,
            "tensors": [],
        },
    )

    _write_passing_analysis_audit(source)
    return source


def _fake_plots(output: Path, *args, **kwargs) -> None:
    del args, kwargs
    (output / "figures").mkdir()
    for name in FIGURE_NAMES:
        for suffix in ("png", "svg"):
            (output / "figures" / f"{name}.{suffix}").write_bytes(b"figure")


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_export_is_atomic_portable_complete_and_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _analysis(tmp_path)
    before = _hash_tree(source)
    monkeypatch.setattr(m2, "_plot_all", _fake_plots)
    output = export_m2_results(source, tmp_path / "results" / "m2_function_space")
    assert output.is_dir()
    assert _hash_tree(source) == before
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["source_analysis_files"]["m2b/function_metrics.csv"] == (
        "analysis_runs/m2-test/m2b/function_metrics.csv"
    )
    assert all("\\" not in value for value in provenance["source_analysis_files"].values())
    assert provenance["smoothing"] is False
    assert provenance["hardware"]["prefix"] == "./env"
    assert provenance["frozen_results_manifests"]["results/m1"][0]["size"] == 12
    audit = json.loads((output / "audit/m2_export.json").read_text(encoding="utf-8"))
    assert audit["passed"] is True
    assert "audit/m2_export.json" not in audit["portable_file_sha256_excluding_audit"]
    for figure in FIGURE_NAMES:
        assert (output / "figures" / f"{figure}.png").is_file()
        assert (output / "figures" / f"{figure}.svg").is_file()
    with pytest.raises(FileExistsError, match="overwrite"):
        export_m2_results(source, output)


def test_publish_failure_removes_temporary_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _analysis(tmp_path)
    monkeypatch.setattr(m2, "_plot_all", _fake_plots)

    def fail_replace(source_path, destination_path):
        raise OSError(f"simulated publish failure: {source_path} -> {destination_path}")

    monkeypatch.setattr(m2, "replace_with_retry", fail_replace)
    destination = tmp_path / "failed-export"
    with pytest.raises(OSError, match="simulated publish failure"):
        export_m2_results(source, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".failed-export.*"))


def test_export_rejects_changed_audited_source(tmp_path: Path) -> None:
    source = _analysis(tmp_path)
    with (source / "m2b/function_metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="source changed"):
        export_m2_results(source, tmp_path / "results")


def test_portable_audit_rejects_nonfinite_csv_and_forbidden_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _analysis(tmp_path)
    monkeypatch.setattr(m2, "_plot_all", _fake_plots)
    output = export_m2_results(source, tmp_path / "results")
    with (output / "m2b/function_metrics.csv").open("a", encoding="utf-8") as handle:
        handle.write("nan\n")
    (output / "forbidden_fourier_metric.json").write_text("{}\n", encoding="utf-8")
    audit = audit_m2_export(output)
    assert audit["passed"] is False
    assert audit["checks"]["no_m3_plus_artifacts"] is False


def test_real_plotter_generates_all_png_svg_pairs(tmp_path: Path) -> None:
    source = _analysis(tmp_path)
    output = export_m2_results(source, tmp_path / "plotted-results")
    for name in FIGURE_NAMES:
        assert (output / "figures" / f"{name}.png").stat().st_size > 0
        assert (output / "figures" / f"{name}.svg").stat().st_size > 0


def test_function_csv_requires_json_encoding_for_state_roles() -> None:
    record = _function_record(100)
    row = {key: "" if value is None else str(value) for key, value in record.items()}
    with pytest.raises(ValueError, match="state_roles"):
        m2._validate_function_records([record], list(record), [row])


def test_validation_difference_summary_reads_nested_and_flat_values() -> None:
    assert m2._maximum_absolute_difference(
        {
            "absolute_differences": {"ce": 1.0e-7, "margin": {"min": -2.0e-7}},
            "norm_abs_diff": 3.0e-7,
        }
    ) == pytest.approx(3.0e-7)


def test_reporting_contract_covers_runner_analysis_audit_files_and_fields() -> None:
    assert set(m2.REQUIRED_ANALYSIS_FILES) - {"audit/m2_analysis.json"} == set(
        audit_m2._AUDITED_FILES
    )
    assert audit_m2._REQUIRED_FUNCTION_FIELDS.issubset(m2.REQUIRED_FUNCTION_FIELDS)


@pytest.mark.parametrize(("limit_mib", "exported"), [(1, True), (0, False)])
def test_selected_tensor_export_obeys_size_limit_and_keeps_portable_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_mib: int,
    exported: bool,
) -> None:
    source = _analysis(tmp_path)
    tensor_path = source / "selected_tensors/step_000100.npz"
    tensor_path.write_bytes(b"selected tensor")
    tensor_hash = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    write_json(
        source / "selected_tensors/manifest.json",
        {
            "schema_version": 1,
            "analysis_id": "m2-test",
            "export_limit_mib": limit_mib,
            "total_size": tensor_path.stat().st_size,
            "tensors": [
                {
                    "step": 100,
                    "path": "selected_tensors/step_000100.npz",
                    "sha256": tensor_hash,
                    "size": tensor_path.stat().st_size,
                    "shape": [3, 3, 3],
                    "raw_dtype": "torch.float32",
                }
            ],
        },
    )
    _write_passing_analysis_audit(source)
    monkeypatch.setattr(m2, "_plot_all", _fake_plots)
    output = export_m2_results(source, tmp_path / f"results-{limit_mib}")
    manifest = json.loads((output / "selected_tensors/manifest.json").read_text(encoding="utf-8"))
    entry = manifest["tensors"][0]
    assert manifest["full_tensors_exported"] is exported
    assert entry["exported"] is exported
    assert entry["source_path"] == ("analysis_runs/m2-test/selected_tensors/step_000100.npz")
    assert (output / "selected_tensors/step_000100.npz").exists() is exported
    assert json.loads((output / "audit/m2_export.json").read_text())["passed"] is True
