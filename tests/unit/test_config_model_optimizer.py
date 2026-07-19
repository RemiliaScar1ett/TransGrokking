from __future__ import annotations

import copy

import pytest
import torch

from transgrokking.config import config_from_dict, load_config
from transgrokking.data import generate_modular_addition
from transgrokking.models import TransparentTransformer
from transgrokking.training.optimizer import build_adamw, classify_parameters
from transgrokking.training.trainer import build_model


def test_modular_addition_data_remains_complete_and_deterministic() -> None:
    first = generate_modular_addition(7, 0.5, 42)
    second = generate_modular_addition(7, 0.5, 42)
    assert first.inputs.shape == (49, 2)
    assert torch.unique(first.inputs, dim=0).shape[0] == 49
    assert torch.equal(first.labels, (first.inputs[:, 0] + first.inputs[:, 1]) % 7)
    assert set(first.train_indices.tolist()).isdisjoint(first.test_indices.tolist())
    assert first.split_hash == second.split_hash


def test_config_hash_separates_scientific_and_execution_fields() -> None:
    config = load_config("configs/smoke.yaml")
    execution = config.to_dict()
    execution["optimization"]["max_steps"] = 10
    execution["logging"]["eval_interval"] = 2
    execution["hardware"]["analysis_batch_size"] = 2
    assert config_from_dict(execution).scientific_hash() == config.scientific_hash()
    scientific = config.to_dict()
    scientific["optimization"]["learning_rate"] = 0.002
    assert config_from_dict(scientific).scientific_hash() != config.scientific_hash()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("task.modulus", True),
        ("model.final_norm", 0),
        ("optimization.max_steps", 0),
        ("hardware.expected_vram_gb", 0),
        ("logging.eval_interval", 0),
        ("logging.activation_steps", [2, 1, 1]),
    ],
)
def test_config_validation_reports_field_path(path: str, value: object) -> None:
    raw = load_config("configs/smoke.yaml").to_dict()
    section, field = path.split(".")
    raw[section][field] = value
    with pytest.raises(ValueError, match=path.replace(".", r"\.")):
        config_from_dict(raw)


def _model(final_norm: bool) -> TransparentTransformer:
    return TransparentTransformer(7, 2, 16, 2, 1, 32, 0.0, "relu", True, final_norm)


def test_final_norm_modes_forward_backward_cache_and_logits() -> None:
    torch.manual_seed(11)
    without_norm = _model(False)
    with_norm = _model(True)
    common = {
        name: value
        for name, value in without_norm.state_dict().items()
        if name in with_norm.state_dict()
    }
    with_norm.load_state_dict(common, strict=False)
    tokens = torch.tensor([[1, 2], [3, 4]])
    raw_logits, raw_cache = without_norm.run_with_cache(tokens)
    norm_logits, norm_cache = with_norm.run_with_cache(tokens)
    assert raw_cache["residual.final"].shape == (2, 2, 16)
    assert "residual.final_normalized" not in raw_cache
    assert norm_cache["residual.final"].shape == (2, 2, 16)
    assert norm_cache["residual.final_normalized"].shape == (2, 2, 16)
    assert not torch.allclose(raw_logits, norm_logits)
    raw_logits.sum().backward()
    norm_logits.sum().backward()
    assert without_norm.unembedding.weight.grad is not None
    assert with_norm.final_norm is not None and with_norm.final_norm.weight.grad is not None
    assert load_config("configs/baseline_ce.yaml").model.final_norm is False


def test_parameter_groups_follow_policy_without_gaps_or_duplicates() -> None:
    config = load_config("configs/smoke.yaml")
    model = build_model(config)
    grouping = classify_parameters(model, config.optimization)
    all_names = set(grouping.decay) | set(grouping.no_decay)
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert all_names == expected
    assert set(grouping.decay).isdisjoint(grouping.no_decay)
    assert "token_embedding.weight" in grouping.decay
    assert "position_embedding.weight" in grouping.decay
    assert "unembedding.weight" in grouping.decay
    assert "unembedding.bias" in grouping.no_decay
    assert "blocks.0.ln1.weight" in grouping.no_decay
    optimizer, _ = build_adamw(model, config.optimization)
    assert [group["group_name"] for group in optimizer.param_groups] == ["decay", "no_decay"]
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.1, 0.0]


def test_policy_can_toggle_embeddings_biases_and_layer_norm() -> None:
    raw = load_config("configs/smoke.yaml").to_dict()
    policy = raw["optimization"]["decay_policy"]
    policy.update({"embeddings": False, "biases": True, "layer_norm": True})
    config = config_from_dict(copy.deepcopy(raw))
    grouping = classify_parameters(build_model(config), config.optimization)
    assert "token_embedding.weight" in grouping.no_decay
    assert "unembedding.bias" in grouping.decay
    assert "blocks.0.ln1.weight" in grouping.decay
