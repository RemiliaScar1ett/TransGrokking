from __future__ import annotations

import json
from pathlib import Path

import torch

from transgrokking.config import config_from_dict, load_config
from transgrokking.training.trainer import train


def _final_payload(run_dir: Path, step: int) -> dict[str, object]:
    return torch.load(
        run_dir / "checkpoints" / f"step_{step:06d}.pt",
        map_location="cpu",
        weights_only=False,
    )


def _assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, list):
        assert isinstance(right, list)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_resume_matches_continuous_training(tmp_path) -> None:
    raw = load_config("configs/smoke.yaml").to_dict()
    raw["logging"]["runs_dir"] = str(tmp_path)
    config = config_from_dict(raw)

    continuous_dir = train(config)
    interrupted_dir = train(config, stop_after=1)
    resumed_dir = train(config, resume_from=interrupted_dir / "checkpoints" / "step_000001.pt")
    assert resumed_dir == interrupted_dir

    continuous = _final_payload(continuous_dir, 3)
    resumed = _final_payload(resumed_dir, 3)
    assert continuous["global_step"] == resumed["global_step"] == 3
    for name, value in continuous["model_state"].items():
        assert torch.equal(value, resumed["model_state"][name]), name
    _assert_nested_equal(continuous["optimizer_state"], resumed["optimizer_state"])

    status = json.loads((resumed_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "completed"
    assert (resumed_dir / "config.resolved.yaml").exists()
    assert (resumed_dir / "split.pt").exists()
    assert (resumed_dir / "metrics" / "scalars.jsonl").exists()
