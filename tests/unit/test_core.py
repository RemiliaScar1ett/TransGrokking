from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from transgrokking.config import config_from_dict, load_config
from transgrokking.data import generate_modular_addition
from transgrokking.models import TransparentTransformer
from transgrokking.training.checkpoint import load_checkpoint, save_checkpoint
from transgrokking.training.trainer import build_model
from transgrokking.utils.doctor import DoctorReport, validate_doctor_report
from transgrokking.utils.reproducibility import configure_reproducibility


def test_data_is_complete_correct_and_reproducible() -> None:
    first = generate_modular_addition(7, 0.5, 42)
    second = generate_modular_addition(7, 0.5, 42)
    assert first.inputs.shape == (49, 2)
    assert torch.unique(first.inputs, dim=0).shape[0] == 49
    assert torch.equal(first.labels, (first.inputs[:, 0] + first.inputs[:, 1]) % 7)
    assert set(first.train_indices.tolist()).isdisjoint(first.test_indices.tolist())
    assert sorted(torch.cat((first.train_indices, first.test_indices)).tolist()) == list(range(49))
    assert first.split_hash == second.split_hash
    assert torch.equal(first.train_indices, second.train_indices)


def test_config_is_strict_and_ce_only() -> None:
    raw = load_config("configs/smoke.yaml").to_dict()
    raw["unknown"] = 1
    with pytest.raises(ValueError, match="top-level"):
        config_from_dict(raw)
    raw = load_config("configs/smoke.yaml").to_dict()
    raw["loss"]["congruence_weight"] = 0.1
    with pytest.raises(ValueError, match="CE-only"):
        config_from_dict(raw)


def test_model_is_causal_cacheable_and_differentiable() -> None:
    torch.manual_seed(3)
    model = TransparentTransformer(7, 2, 16, 2, 1, 32, 0.0, "relu", True)
    tokens = torch.tensor([[1, 2], [3, 4]])
    changed_future = tokens.clone()
    changed_future[:, 1] = (changed_future[:, 1] + 1) % 7
    logits, cache = model.run_with_cache(tokens)
    other_logits = model(changed_future)
    assert logits.shape == (2, 2, 7)
    assert torch.allclose(logits[:, 0], other_logits[:, 0])
    assert cache["blocks.0.attention.head_output"].shape == (2, 2, 2, 16)
    assert cache["blocks.0.mlp.pre"].shape == (2, 2, 32)
    logits[:, -1].sum().backward()
    assert model.token_embedding.weight.grad is not None


def test_doctor_strict_validation_accepts_target_and_rejects_mismatch() -> None:
    report = DoctorReport(
        prefix="E:/Workplace/Trans/env",
        expected_prefix="E:/Workplace/Trans/env",
        prefix_ok=True,
        python_version="3.10.20",
        torch_version="2.2.0",
        torch_cuda_runtime="12.1",
        cuda_available=True,
        driver_version="610.62",
        device_name="NVIDIA GeForce RTX 4060 Laptop GPU",
        total_vram_bytes=8_188_000_000,
        compute_capability="8.9",
    )
    assert not validate_doctor_report(report, True, "NVIDIA GeForce RTX 4060 Laptop GPU", 8)
    mismatch = replace(report, device_name="Other GPU")
    assert validate_doctor_report(mismatch, True, "NVIDIA GeForce RTX 4060 Laptop GPU", 8)


def test_checkpoint_round_trip_and_split_guard(tmp_path) -> None:
    config = load_config("configs/smoke.yaml")
    configure_reproducibility(config.optimization.seed, True)
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    tokens = torch.tensor([[1, 2], [2, 3]])
    model(tokens).sum().backward()
    optimizer.step()
    expected = model(tokens).detach().clone()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, config, "split", 1)
    restored = build_model(config)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.001)
    assert load_checkpoint(path, restored, restored_optimizer, config, "split", "cpu") == 1
    assert torch.equal(expected, restored(tokens))
    assert restored_optimizer.state_dict()["state"]
    with pytest.raises(ValueError, match="split hash"):
        load_checkpoint(path, restored, restored_optimizer, config, "wrong", "cpu")
