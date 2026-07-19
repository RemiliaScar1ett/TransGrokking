from __future__ import annotations

import json
from pathlib import Path

import pytest

from transgrokking.config import load_config
from transgrokking.training.artifacts import (
    append_error_offsets,
    append_evaluation_artifacts,
    load_error_offset_records,
    load_scalar_records,
    reconcile_metric_files,
)
from transgrokking.utils import atomic


def _offsets(step: int, modulus: int = 3) -> list[dict[str, object]]:
    return [
        {
            "schema_version": 1,
            "step": step,
            "split": split,
            "modulus": modulus,
            "counts": [0] * modulus,
        }
        for split in ("train", "test")
    ]


def _scalar(step: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "step": step,
        "train_accuracy": 0.0,
        "test_accuracy": 0.0,
    }


def test_metric_files_are_schema_valid_atomic_and_reconcilable(tmp_path: Path) -> None:
    (tmp_path / "metrics").mkdir()
    config = load_config("configs/smoke.yaml")
    append_evaluation_artifacts(tmp_path, _scalar(1), _offsets(1), 3, 1, config.events)
    assert [
        record["step"] for record in load_scalar_records(tmp_path / "metrics/scalars.jsonl")
    ] == [1]
    assert len(load_error_offset_records(tmp_path / "metrics/error_offsets.jsonl")) == 2
    events = json.loads((tmp_path / "metrics/events.json").read_text(encoding="utf-8"))
    assert events["schema_version"] == 1
    assert events["last_evaluated_step"] == 1

    append_error_offsets(tmp_path / "metrics/error_offsets.jsonl", _offsets(2))
    reconcile_metric_files(tmp_path)
    assert len(load_error_offset_records(tmp_path / "metrics/error_offsets.jsonl")) == 2


def test_nonfinite_scalar_is_rejected_without_changing_committed_file(tmp_path: Path) -> None:
    (tmp_path / "metrics").mkdir()
    config = load_config("configs/smoke.yaml")
    append_evaluation_artifacts(tmp_path, _scalar(1), _offsets(1), 3, 1, config.events)
    scalar_path = tmp_path / "metrics/scalars.jsonl"
    original = scalar_path.read_bytes()
    invalid = _scalar(2)
    invalid["train_accuracy"] = float("nan")
    with pytest.raises(ValueError):
        append_evaluation_artifacts(tmp_path, invalid, _offsets(2), 3, 1, config.events)
    assert scalar_path.read_bytes() == original
    reconcile_metric_files(tmp_path)


def test_jsonl_replace_failure_preserves_committed_metric_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "metrics").mkdir()
    config = load_config("configs/smoke.yaml")
    append_evaluation_artifacts(tmp_path, _scalar(1), _offsets(1), 3, 1, config.events)
    scalar_path = tmp_path / "metrics/scalars.jsonl"
    offset_path = tmp_path / "metrics/error_offsets.jsonl"
    before = (scalar_path.read_bytes(), offset_path.read_bytes())

    def fail_replace(source: str, target: str | Path) -> None:
        raise OSError("simulated JSONL replace failure")

    monkeypatch.setattr(atomic.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        append_evaluation_artifacts(tmp_path, _scalar(2), _offsets(2), 3, 1, config.events)
    assert (scalar_path.read_bytes(), offset_path.read_bytes()) == before
