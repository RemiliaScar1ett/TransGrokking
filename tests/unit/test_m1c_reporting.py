from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from transgrokking.reporting import m1c
from transgrokking.reporting.m1c import FIGURE_NAMES, export_m1c_results
from transgrokking.utils.atomic import write_json, write_json_lines


def _scalar(step: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "step": step,
        "train_cross_entropy": 1.0 / step,
        "test_cross_entropy": 2.0 / step,
        "train_accuracy": 1.0,
        "test_accuracy": 0.99,
        "train_margin_mean": 1.0,
        "train_margin_min": 0.5,
        "train_margin_q05": 0.6,
        "train_margin_median": 1.0,
        "train_margin_q95": 1.4,
        "test_margin_mean": 0.8,
        "test_margin_min": -0.1,
        "test_margin_q05": 0.1,
        "test_margin_median": 0.8,
        "test_margin_q95": 1.2,
        "parameter_norm_total": 3.0,
        "parameter_group_norm_decay": 2.0,
        "parameter_group_norm_no_decay": 2.0,
        "parameter_norm_token_embedding": 1.0,
        "parameter_norm_final_norm": None,
    }


def _optimization(step: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "step": step,
        "total_update_l2": 0.1,
        "data_update_l2": 0.09,
        "decay_update_l2": 0.01,
        "adam_first_moment_l2": 0.2,
        "adam_second_moment_mean": 0.03,
        "adam_second_moment_rms": 0.04,
        "adam_second_moment_max": 0.05,
    }


def _source(tmp_path: Path) -> Path:
    run = tmp_path / "runs" / "child"
    (run / "metrics").mkdir(parents=True)
    (run / "audit").mkdir()
    (run / "checkpoints").mkdir()
    (run / "config.resolved.yaml").write_text("task: {}\n", encoding="utf-8")
    (run / "measurement.resolved.yaml").write_text(
        "schema_version: 1\nprofile: m1c-extension\n",
        encoding="utf-8",
    )
    scalars = [_scalar(50), _scalar(100)]
    write_json_lines(run / "metrics/scalars.jsonl", scalars)
    offsets = [
        {
            "schema_version": 1,
            "step": step,
            "split": split,
            "modulus": 2,
            "counts": [0, 0],
        }
        for step in (50, 100)
        for split in ("train", "test")
    ]
    write_json_lines(run / "metrics/error_offsets.jsonl", offsets)
    write_json_lines(run / "metrics/optimization.jsonl", [_optimization(100)])
    write_json(
        run / "metrics/events.json",
        {
            "t_fit": {"status": "reached", "event_step": 50},
            "t_grok50": {"status": "reached", "event_step": 50},
            "t_grok99": {"status": "reached", "event_step": 50},
        },
    )
    write_json(
        run / "metrics/stability.json",
        {"t_stable99": {"status": "not_reached"}, "final_state": "recovering"},
    )
    write_json(run / "metrics/collapse_episodes.json", {"episodes": []})
    write_json(
        run / "metadata.json",
        {
            "git_commit": "abc",
            "scientific_config_hash": "a" * 64,
            "split_hash": "b" * 64,
            "extension_origin_checkpoint": "step_000050.pt",
            "diagnostics_start_step": 50,
        },
    )
    write_json(run / "status.json", {"state": "completed", "global_step": 100})
    (run / "checkpoints/step_000100.pt").write_bytes(b"checkpoint")
    write_json(
        run / "checkpoints/manifest.json",
        {
            "schema_version": 2,
            "checkpoints": [{"step": 100, "path": "step_000100.pt"}],
        },
    )
    audited_paths = [
        run / "config.resolved.yaml",
        run / "measurement.resolved.yaml",
        run / "metadata.json",
        run / "status.json",
        run / "metrics/scalars.jsonl",
        run / "metrics/error_offsets.jsonl",
        run / "metrics/events.json",
        run / "metrics/stability.json",
        run / "metrics/collapse_episodes.json",
        run / "metrics/optimization.jsonl",
        run / "checkpoints/manifest.json",
        run / "checkpoints/step_000100.pt",
    ]
    write_json(
        run / "audit/m1c_extension.json",
        {
            "passed": True,
            "run_id": "child",
            "canonical_parent_run_id": "parent",
            "parameter_group_signature": [],
            "audited_source_sha256": {
                path.relative_to(run).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in audited_paths
            },
        },
    )
    return run


def _fake_plots(
    output: Path,
    run_id: str,
    scalars,
    optimization,
    events,
    stability,
    episodes,
) -> None:
    del run_id, scalars, optimization, events, stability, episodes
    (output / "figures").mkdir()
    for name in FIGURE_NAMES:
        for suffix in ("png", "svg"):
            (output / "figures" / f"{name}.{suffix}").write_bytes(b"figure")


def test_export_is_complete_reproducible_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    before = {
        path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*")
        if path.is_file()
    }
    output = export_m1c_results(source, tmp_path / "results")
    assert output.is_dir()
    assert (
        json.loads((output / "provenance.json").read_text(encoding="utf-8"))["smoothing"] is False
    )
    with (output / "margin_curve.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert "train_margin_mean" in rows[0]
    assert "test_margin_mean" in rows[0]
    with (output / "parameter_norm_curve.csv").open(encoding="utf-8", newline="") as handle:
        parameter_rows = list(csv.DictReader(handle))
    assert "parameter_norm_token_embedding" in parameter_rows[0]
    assert "parameter_norm_final_norm" in parameter_rows[0]
    after = {
        path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*")
        if path.is_file()
    }
    assert after == before
    with pytest.raises(FileExistsError, match="overwrite"):
        export_m1c_results(source, output)


def test_export_publish_failure_leaves_no_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    monkeypatch.setattr(m1c, "_plot_all", _fake_plots)

    def fail_replace(source_path, destination_path):
        raise OSError(f"simulated publish failure: {source_path} -> {destination_path}")

    monkeypatch.setattr(m1c.os, "replace", fail_replace)
    output = tmp_path / "failed-results"
    with pytest.raises(OSError, match="simulated"):
        export_m1c_results(source, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".failed-results.*"))


def test_export_rejects_source_changed_after_audit(tmp_path: Path) -> None:
    source = _source(tmp_path)
    with (source / "metrics/scalars.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="changed after audit"):
        export_m1c_results(source, tmp_path / "results")


def test_episode_plot_annotations_use_split_specific_recovery_steps() -> None:
    steps = m1c._episode_steps(
        {
            "episodes": [
                {
                    "episode_type": "train",
                    "onset_step": 100,
                    "trough_step": 150,
                    "train_recovery_step": 200,
                },
                {
                    "episode_type": "test",
                    "onset_step": 110,
                    "trough_step": 160,
                    "test_recovery_step": 210,
                },
                {
                    "episode_type": "joint",
                    "onset_step": 100,
                    "trough_step": 160,
                    "joint_recovery_step": 210,
                },
            ]
        }
    )
    assert steps == {
        "onset": [100, 110],
        "trough": [150, 160],
        "recovery": [200, 210],
    }
