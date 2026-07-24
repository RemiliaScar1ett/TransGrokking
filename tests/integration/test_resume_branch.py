from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from transgrokking.config import config_from_dict, load_config
from transgrokking.models import TransparentTransformer
from transgrokking.training.artifacts import (
    append_error_offsets,
    load_error_offset_records,
    load_manifest,
    scalar_steps,
)
from transgrokking.training.trainer import _resume_plan, train


def _config(tmp_path: Path, max_steps: int):
    raw = load_config("configs/smoke.yaml").to_dict()
    raw["logging"]["runs_dir"] = str(tmp_path)
    raw["optimization"]["max_steps"] = max_steps
    return config_from_dict(raw)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_interrupted_latest_checkpoint_resumes_inplace_with_higher_limit(tmp_path: Path) -> None:
    initial = _config(tmp_path, 2)
    run_dir = train(initial, stop_after=1)
    checkpoint = run_dir / "checkpoints" / "step_000001.pt"
    before = _digest(checkpoint)
    resumed = train(_config(tmp_path, 3), resume_from=checkpoint)
    assert resumed == run_dir
    assert _digest(checkpoint) == before
    assert scalar_steps(run_dir / "metrics" / "scalars.jsonl") == [1, 2, 3]
    assert [entry["step"] for entry in load_manifest(run_dir)] == [0, 1, 2, 3]
    offsets = load_error_offset_records(run_dir / "metrics/error_offsets.jsonl")
    assert [offsets[index]["step"] for index in range(0, len(offsets), 2)] == [1, 2, 3]
    events = json.loads((run_dir / "metrics/events.json").read_text(encoding="utf-8"))
    assert events["last_evaluated_step"] == 3
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "completed"


def test_instrumented_resume_forces_child_even_when_inplace_would_be_safe(
    tmp_path: Path,
) -> None:
    run_dir = train(_config(tmp_path, 2), stop_after=1)
    checkpoint = run_dir / "checkpoints" / "step_000001.pt"
    mode, *_ = _resume_plan(
        _config(tmp_path, 3),
        checkpoint.resolve(),
        "auto",
        force_branch=True,
    )
    assert mode == "branch"
    with pytest.raises(ValueError, match="must create a child"):
        _resume_plan(
            _config(tmp_path, 3),
            checkpoint.resolve(),
            "inplace",
            force_branch=True,
        )


def test_completed_and_nonlatest_resume_create_traceable_child_runs(tmp_path: Path) -> None:
    parent = train(_config(tmp_path, 3))
    latest = parent / "checkpoints" / "step_000003.pt"
    completed_child = train(_config(tmp_path, 4), resume_from=latest)
    assert completed_child != parent
    metadata = json.loads((completed_child / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["parent_run_id"] == parent.name
    assert Path(metadata["parent_checkpoint"]) == latest
    assert metadata["parent_global_step"] == 3
    assert [entry["step"] for entry in load_manifest(completed_child)] == [3, 4]
    assert scalar_steps(completed_child / "metrics" / "scalars.jsonl") == [1, 2, 3, 4]
    child_offsets = load_error_offset_records(completed_child / "metrics/error_offsets.jsonl")
    assert [child_offsets[index]["step"] for index in range(0, len(child_offsets), 2)] == [
        1,
        2,
        3,
        4,
    ]

    historical = parent / "checkpoints" / "step_000001.pt"
    before = _digest(historical)
    history_child = train(_config(tmp_path, 4), resume_from=historical)
    assert history_child not in {parent, completed_child}
    assert _digest(historical) == before
    history_metadata = json.loads((history_child / "metadata.json").read_text(encoding="utf-8"))
    assert history_metadata["parent_global_step"] == 1
    assert [entry["step"] for entry in load_manifest(history_child)] == [1, 2, 3, 4]
    assert scalar_steps(history_child / "metrics" / "scalars.jsonl") == [1, 2, 3, 4]


def test_scientific_change_is_rejected_and_target_must_increase(tmp_path: Path) -> None:
    parent = train(_config(tmp_path, 2), stop_after=1)
    checkpoint = parent / "checkpoints" / "step_000001.pt"
    raw = _config(tmp_path, 3).to_dict()
    raw["optimization"]["learning_rate"] = 0.002
    with pytest.raises(ValueError, match="scientific config hash"):
        train(config_from_dict(raw), resume_from=checkpoint)
    with pytest.raises(ValueError, match="strictly greater"):
        train(_config(tmp_path, 1), resume_from=checkpoint)


def test_branch_discards_uncommitted_tail_without_mutating_parent(tmp_path: Path) -> None:
    parent = train(_config(tmp_path, 2))
    checkpoint = parent / "checkpoints" / "step_000002.pt"
    offset_path = parent / "metrics" / "error_offsets.jsonl"
    append_error_offsets(
        offset_path,
        [
            {
                "schema_version": 1,
                "step": 3,
                "split": split,
                "modulus": 7,
                "counts": [0] * 7,
            }
            for split in ("train", "test")
        ],
    )
    before = {
        path.relative_to(parent).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in parent.rglob("*")
        if path.is_file()
    }
    child = train(_config(tmp_path, 3), resume_from=checkpoint)
    after = {
        path.relative_to(parent).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in parent.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert child != parent
    assert scalar_steps(child / "metrics" / "scalars.jsonl") == [1, 2, 3]
    child_offsets = load_error_offset_records(child / "metrics/error_offsets.jsonl")
    assert [child_offsets[index]["step"] for index in range(0, len(child_offsets), 2)] == [
        1,
        2,
        3,
    ]


def test_initialization_failure_and_keyboard_interrupt_leave_recoverable_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transgrokking.training.trainer as trainer_module

    def fail_build(config):
        raise RuntimeError("simulated initialization failure")

    monkeypatch.setattr(trainer_module, "build_model", fail_build)
    with pytest.raises(RuntimeError, match="initialization"):
        train(_config(tmp_path, 2))
    failed_run = next(path for path in tmp_path.iterdir() if path.is_dir())
    failed_status = json.loads((failed_run / "status.json").read_text(encoding="utf-8"))
    assert failed_status["state"] == "failed"

    monkeypatch.undo()
    interrupt_root = tmp_path / "interrupt"
    interrupt_root.mkdir()

    def interrupt_forward(self, tokens):
        raise KeyboardInterrupt

    monkeypatch.setattr(TransparentTransformer, "forward", interrupt_forward)
    with pytest.raises(KeyboardInterrupt):
        train(_config(interrupt_root, 2))
    interrupted_run = next(path for path in interrupt_root.iterdir() if path.is_dir())
    interrupted = json.loads((interrupted_run / "status.json").read_text(encoding="utf-8"))
    assert interrupted["state"] == "interrupted"
    emergency = Path(interrupted["emergency_checkpoint"])
    assert emergency.is_file()


def test_interrupt_inside_optimizer_step_is_not_serialized_as_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    def interrupt_step(self, closure=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(torch.optim.AdamW, "step", interrupt_step)
    with pytest.raises(KeyboardInterrupt):
        train(_config(tmp_path, 2))
    run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "interrupted"
    assert status["optimizer_update_started"] is True
    assert status["optimizer_update_completed"] is False
    assert status["emergency_checkpoint"] is None
    assert [entry["step"] for entry in load_manifest(run_dir)] == [0]
