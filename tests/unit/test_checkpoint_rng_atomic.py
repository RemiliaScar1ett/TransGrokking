from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from transgrokking.config import load_config
from transgrokking.training.artifacts import add_manifest_checkpoint
from transgrokking.training.checkpoint import load_checkpoint, save_checkpoint
from transgrokking.training.optimizer import build_adamw
from transgrokking.training.trainer import build_model
from transgrokking.utils import atomic
from transgrokking.utils.reproducibility import configure_reproducibility


def test_checkpoint_restores_python_numpy_and_torch_cpu_rng(tmp_path: Path) -> None:
    config = load_config("configs/smoke.yaml")
    configure_reproducibility(123, True)
    model = build_model(config)
    optimizer, _ = build_adamw(model, config.optimization)
    path = tmp_path / "state.pt"
    save_checkpoint(path, model, optimizer, config, "split", 1)
    expected = (random.random(), np.random.random(), torch.rand(3))
    random.random(), np.random.random(), torch.rand(3)
    assert load_checkpoint(path, model, optimizer, config, "split", "cpu") == 1
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    optimizer.param_groups[0]["group_name"] = "changed"
    with pytest.raises(ValueError, match="parameter-group structure"):
        load_checkpoint(path, model, optimizer, config, "split", "cpu")


@pytest.mark.cuda
def test_checkpoint_restores_torch_cuda_rng(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config = load_config("configs/smoke.yaml")
    configure_reproducibility(456, True)
    model = build_model(config)
    optimizer, _ = build_adamw(model, config.optimization)
    path = tmp_path / "cuda_rng.pt"
    save_checkpoint(path, model, optimizer, config, "split", 1)
    expected = torch.rand(3, device="cuda")
    torch.rand(3, device="cuda")
    load_checkpoint(path, model, optimizer, config, "split", "cpu")
    assert torch.equal(torch.rand(3, device="cuda"), expected)


def test_atomic_replace_failure_preserves_original_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "artifact.json"
    destination.write_text('{"old": true}\n', encoding="utf-8")

    def fail_replace(source: str, target: str | Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(atomic.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        atomic.write_json(destination, {"new": True})
    assert destination.read_text(encoding="utf-8") == '{"old": true}\n'
    assert list(tmp_path.glob(".artifact.json.*")) == []


def test_atomic_checkpoint_replace_failure_preserves_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "step_000001.pt"
    torch.save({"valid": True}, destination)

    def fail_replace(source: str, target: str | Path) -> None:
        raise OSError("simulated checkpoint replace failure")

    monkeypatch.setattr(atomic.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        atomic.torch_save(destination, {"valid": False})
    assert torch.load(destination, weights_only=False) == {"valid": True}
    assert list(tmp_path.glob(".step_000001.pt.*")) == []


def test_manifest_never_references_incomplete_checkpoint(tmp_path: Path) -> None:
    for child in ("checkpoints", "metrics"):
        (tmp_path / child).mkdir()
    incomplete = tmp_path / "checkpoints" / ".step_000001.pt.partial"
    incomplete.write_bytes(b"partial")
    with pytest.raises(ValueError, match="incomplete"):
        add_manifest_checkpoint(tmp_path, 1, incomplete.with_name("step_000001.pt"))
    assert not (tmp_path / "checkpoints" / "manifest.json").exists()
