from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import torch
import yaml

from transgrokking.config import config_from_dict, load_config
from transgrokking.training.artifacts import load_manifest, scalar_steps
from transgrokking.training.optimizer import (
    build_adamw,
    validate_optimizer_parameter_identity,
)
from transgrokking.training.trainer import build_model, train
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
    optimizer.step()
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
