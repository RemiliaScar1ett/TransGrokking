from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from transgrokking.config import config_from_dict, load_config
from transgrokking.training.artifacts import load_manifest, scalar_steps
from transgrokking.training.trainer import train
from transgrokking.utils.doctor import collect_doctor_report, validate_doctor_report


def _write_config(tmp_path: Path, *, max_steps: int = 2) -> Path:
    raw = load_config("configs/smoke.yaml").to_dict()
    raw["logging"]["runs_dir"] = str(tmp_path / "runs")
    raw["optimization"]["max_steps"] = max_steps
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _cli(*arguments: str) -> subprocess.CompletedProcess[str]:
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


@pytest.mark.cuda
def test_target_gpu_one_update_records_peak_memory(tmp_path: Path) -> None:
    report = collect_doctor_report()
    errors = validate_doctor_report(report, True, "NVIDIA GeForce RTX 4060 Laptop GPU", 8)
    if errors:
        pytest.skip("target RTX 4060 Laptop GPU 8GB unavailable: " + "; ".join(errors))
    raw = load_config("configs/smoke.yaml").to_dict()
    raw["optimization"]["device"] = "cuda:0"
    raw["optimization"]["max_steps"] = 1
    raw["hardware"]["formal_run"] = True
    raw["logging"]["runs_dir"] = str(tmp_path)
    run_dir = train(config_from_dict(raw))
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert metadata["formal_run"] is True
    assert metadata["max_memory_allocated"] > 0
    assert metadata["max_memory_reserved"] > 0
    assert status["state"] == "completed"
