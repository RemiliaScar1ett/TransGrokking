from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
import torch
import yaml

from transgrokking.config import config_from_dict, load_config
from transgrokking.training.artifacts import (
    load_manifest,
    load_optimization_records,
    load_scalar_records,
    scalar_steps,
    update_events,
)
from transgrokking.training.diagnostics import (
    capture_optimization_step,
    finalize_optimization_step,
)
from transgrokking.training.optimizer import (
    build_adamw,
    validate_optimizer_parameter_identity,
)
from transgrokking.training.trainer import build_model, train
from transgrokking.utils.atomic import write_json_lines
from transgrokking.utils.doctor import collect_doctor_report, validate_doctor_report
from transgrokking.utils.reproducibility import configure_reproducibility


def _write_config(tmp_path: Path, *, max_steps: int = 2) -> Path:
    raw = load_config("configs/smoke.yaml").to_dict()
    raw["logging"]["runs_dir"] = str(tmp_path / "runs")
    raw["optimization"]["max_steps"] = max_steps
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _cli(*arguments: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "conda",
            "run",
            "--no-capture-output",
            "--prefix",
            "./env",
            "python",
            "-m",
            "transgrokking.cli",
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def test_real_cli_writes_complete_artifacts_and_branches_completed_run(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    result = _cli("train", "--config", str(config_path))
    assert result.returncode == 0, result.stderr
    parent = Path(result.stdout.strip().splitlines()[-1])
    assert json.loads((parent / "status.json").read_text(encoding="utf-8"))["state"] == "completed"
    assert (parent / "metadata.json").is_file()
    parent_metadata = json.loads((parent / "metadata.json").read_text(encoding="utf-8"))
    assert [group["group_name"] for group in parent_metadata["optimizer_parameter_groups"]] == [
        "decay",
        "no_decay",
    ]
    assert (parent / "split.pt").is_file()
    assert scalar_steps(parent / "metrics" / "scalars.jsonl") == [1, 2]
    assert [entry["step"] for entry in load_manifest(parent)] == [0, 1, 2]
    protected = {
        name: (parent / name).read_bytes()
        for name in (
            "status.json",
            "metrics/scalars.jsonl",
            "metrics/error_offsets.jsonl",
            "metrics/events.json",
        )
    }
    evaluation = _cli("evaluate", "--run-dir", str(parent), "--checkpoint", "1")
    assert evaluation.returncode == 0, evaluation.stderr
    summary = json.loads(evaluation.stdout)
    assert summary["step"] == 1
    assert "train_margin_q01" in summary
    assert set(summary["error_offsets"]) == {"train", "test"}
    for name, contents in protected.items():
        assert (parent / name).read_bytes() == contents

    resumed_path = _write_config(tmp_path, max_steps=3)
    checkpoint = parent / "checkpoints" / "step_000001.pt"
    branch = _cli(
        "train",
        "--config",
        str(resumed_path),
        "--resume-from",
        str(checkpoint),
        "--resume-mode",
        "auto",
    )
    assert branch.returncode == 0, branch.stderr
    child = Path(branch.stdout.strip().splitlines()[-1])
    assert child != parent
    metadata = json.loads((child / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["parent_run_id"] == parent.name
    assert metadata["parent_global_step"] == 1
    assert scalar_steps(child / "metrics/scalars.jsonl") == [1, 2, 3]


def test_real_cli_m1c_measurement_child_preserves_parent(tmp_path: Path) -> None:
    raw = load_config("configs/smoke.yaml").to_dict()
    raw["logging"]["runs_dir"] = str(tmp_path / "runs")
    raw["optimization"]["max_steps"] = 7
    parent_config = config_from_dict(raw)
    parent = train(parent_config)
    scalars_path = parent / "metrics" / "scalars.jsonl"
    scalars = load_scalar_records(scalars_path)
    test_accuracies = [0.0, 0.6, 0.6, 0.6, 1.0, 1.0, 1.0]
    for record, test_accuracy in zip(scalars, test_accuracies, strict=True):
        record["train_accuracy"] = 1.0
        record["test_accuracy"] = test_accuracy
    write_json_lines(scalars_path, scalars)
    update_events(parent, 7, 1, parent_config.events, preserve_existing=False)
    events = json.loads((parent / "metrics/events.json").read_text(encoding="utf-8"))
    assert [events[name]["event_step"] for name in ("t_fit", "t_grok50", "t_grok99")] == [
        1,
        2,
        5,
    ]

    checkpoint = parent / "checkpoints" / "step_000007.pt"
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    metadata = json.loads((parent / "metadata.json").read_text(encoding="utf-8"))
    measurement = {
        "schema_version": 1,
        "profile": "m1c-extension",
        "source": {
            "canonical_run_id": parent.name,
            "canonical_checkpoint_step": 7,
            "canonical_checkpoint_sha256": checkpoint_hash,
            "scientific_config_hash": parent_config.scientific_hash(),
            "split_hash": metadata["split_hash"],
            "eval_interval": 1,
            "checkpoint_interval": 1,
        },
        "frozen_events": {
            "t_fit": 1,
            "t_fit_detected_at": 5,
            "t_grok50": 2,
            "t_grok50_detected_at": 4,
            "t_grok99": 5,
            "t_grok99_detected_at": 7,
        },
        "stability": {
            "stable_accuracy": 0.99,
            "stable_window_intervals": 2,
            "collapse_accuracy": 0.9,
            "train_recovery_accuracy": 0.999,
            "test_recovery_accuracy": 0.99,
            "recovery_consecutive": 2,
            "joint_tolerance_evaluations": 1,
        },
    }
    measurement_path = tmp_path / "measurement.yaml"
    measurement_path.write_text(yaml.safe_dump(measurement, sort_keys=False), encoding="utf-8")
    raw["optimization"]["max_steps"] = 9
    child_path = tmp_path / "child.yaml"
    child_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    protected = {
        path.relative_to(parent).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in parent.rglob("*")
        if path.is_file()
    }

    result = _cli(
        "train",
        "--config",
        str(child_path),
        "--measurement-config",
        str(measurement_path),
        "--resume-from",
        str(checkpoint),
        "--resume-mode",
        "auto",
    )
    assert result.returncode == 0, result.stderr
    child = Path(result.stdout.strip().splitlines()[-1])
    assert child != parent
    assert scalar_steps(child / "metrics/scalars.jsonl") == list(range(1, 10))
    assert [
        record["step"] for record in load_optimization_records(child / "metrics/optimization.jsonl")
    ] == [8, 9]
    assert (child / "metrics/stability.json").is_file()
    assert (child / "metrics/collapse_episodes.json").is_file()
    after = {
        path.relative_to(parent).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in parent.rglob("*")
        if path.is_file()
    }
    assert after == protected


@pytest.mark.cuda
def test_target_gpu_one_update_records_peak_memory(tmp_path: Path) -> None:
    configure_reproducibility(1, True)
    report = collect_doctor_report()
    errors = validate_doctor_report(report, True, "NVIDIA GeForce RTX 4060 Laptop GPU", 8)
    if errors:
        pytest.skip("target RTX 4060 Laptop GPU 8GB unavailable: " + "; ".join(errors))
    raw = load_config("configs/smoke.yaml").to_dict()
    raw["optimization"]["device"] = "cuda:0"
    raw["optimization"]["max_steps"] = 1
    raw["hardware"]["formal_run"] = True
    raw["logging"]["runs_dir"] = str(tmp_path)
    config = config_from_dict(raw)
    torch.cuda.reset_peak_memory_stats("cuda:0")
    model = build_model(config).to("cuda:0", dtype=torch.float32)
    optimizer, grouping = build_adamw(model, config.optimization)
    validate_optimizer_parameter_identity(model, optimizer)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    tokens = torch.tensor([[1, 2], [2, 3]], device="cuda:0")
    targets = torch.tensor([3, 5], device="cuda:0")
    torch.nn.functional.cross_entropy(model(tokens)[:, -1], targets).backward()
    capture = capture_optimization_step(model, optimizer)
    optimizer.step()
    diagnostic = finalize_optimization_step(capture, model, optimizer, step=1)
    changes = [
        (parameter.detach() - before[name]).abs().max()
        for name, parameter in model.named_parameters()
    ]
    assert all(torch.isfinite(change) for change in changes)
    assert any(change.item() > 0 for change in changes)
    assert grouping.decay and grouping.no_decay
    parameter_state = [
        value
        for parameter, state in optimizer.state.items()
        for value in state.values()
        if isinstance(value, torch.Tensor) and value.shape == parameter.shape
    ]
    assert parameter_state and all(value.device.type == "cuda" for value in parameter_state)
    assert all(
        torch.isfinite(value).all()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    assert torch.cuda.max_memory_allocated("cuda:0") > 0
    assert torch.cuda.max_memory_reserved("cuda:0") > 0
    assert diagnostic["step"] == 1
    assert all(
        value is None or not isinstance(value, float) or torch.isfinite(torch.tensor(value))
        for value in diagnostic.values()
    )
    reference = build_model(config).to("cuda:0", dtype=torch.float32)
    reference.load_state_dict(before)
    reference_optimizer, _ = build_adamw(reference, config.optimization)
    reference_optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(reference(tokens)[:, -1], targets).backward()
    reference_optimizer.step()
    assert all(
        torch.equal(parameter, dict(reference.named_parameters())[name])
        for name, parameter in model.named_parameters()
    )
    run_dir = train(config)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert metadata["formal_run"] is True
    assert metadata["max_memory_allocated"] > 0
    assert metadata["max_memory_reserved"] > 0
    assert status["state"] == "completed"
    assert (run_dir / "metrics/error_offsets.jsonl").is_file()
    events = json.loads((run_dir / "metrics/events.json").read_text(encoding="utf-8"))
    assert events["last_evaluated_step"] == 1


@pytest.mark.cuda
def test_fresh_cuda_cli_sets_determinism_before_first_cuda_call(tmp_path: Path) -> None:
    report = collect_doctor_report()
    errors = validate_doctor_report(report, True, "NVIDIA GeForce RTX 4060 Laptop GPU", 8)
    if errors:
        pytest.skip("target RTX 4060 Laptop GPU 8GB unavailable: " + "; ".join(errors))
    raw = load_config("configs/smoke.yaml").to_dict()
    raw["optimization"].update({"device": "cuda:0", "max_steps": 1})
    raw["hardware"]["formal_run"] = True
    raw["logging"]["runs_dir"] = str(tmp_path / "fresh-runs")
    path = tmp_path / "fresh-cuda.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    environment = os.environ.copy()
    environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
    result = _cli("train", "--config", str(path), env=environment)
    assert result.returncode == 0, result.stderr
    run_dir = Path(result.stdout.strip().splitlines()[-1])
    checkpoints = load_manifest(run_dir)
    assert [entry["step"] for entry in checkpoints] == [0, 1]
    initial = torch.load(run_dir / "checkpoints" / "step_000000.pt", map_location="cpu")
    updated = torch.load(run_dir / "checkpoints" / "step_000001.pt", map_location="cpu")
    assert any(
        not torch.equal(initial["model_state"][name], updated["model_state"][name])
        for name in initial["model_state"]
    )
    raw["optimization"]["max_steps"] = 2
    resume_path = tmp_path / "fresh-cuda-resume.yaml"
    resume_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    resumed = _cli(
        "train",
        "--config",
        str(resume_path),
        "--resume-from",
        str(run_dir / "checkpoints" / "step_000001.pt"),
        "--resume-mode",
        "auto",
        env=environment,
    )
    assert resumed.returncode == 0, resumed.stderr
    resumed_run = Path(resumed.stdout.strip().splitlines()[-1])
    assert json.loads((resumed_run / "status.json").read_text(encoding="utf-8"))["global_step"] == 2
