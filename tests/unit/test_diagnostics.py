from __future__ import annotations

import copy
import math
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim import AdamW

from transgrokking.training.diagnostics import (
    capture_optimization_step,
    finalize_optimization_step,
)


class _TwoGroupModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decay_weight = nn.Parameter(torch.tensor([1.0, -2.0]))
        self.no_decay_bias = nn.Parameter(torch.tensor([3.0]))

    def forward(self) -> torch.Tensor:
        return (
            torch.dot(self.decay_weight, torch.tensor([0.5, -0.25])) + self.no_decay_bias[0] * 0.75
        )


def _optimizer(model: _TwoGroupModel, *, with_no_decay_gradient: bool = True) -> AdamW:
    no_decay_parameter = model.no_decay_bias
    if not with_no_decay_gradient:
        no_decay_parameter.requires_grad_(False)
    groups = [
        {
            "params": [model.decay_weight],
            "parameter_names": ["decay_weight"],
            "group_name": "decay",
            "lr": 0.1,
            "weight_decay": 0.2,
        },
    ]
    if with_no_decay_gradient:
        groups.append(
            {
                "params": [model.no_decay_bias],
                "parameter_names": ["no_decay_bias"],
                "group_name": "no_decay",
                "lr": 0.1,
                "weight_decay": 0.0,
            }
        )
    return AdamW(groups, betas=(0.0, 0.0), eps=0.0)


def _clone_optimizer_state(optimizer: AdamW) -> dict[str, object]:
    cloned = copy.deepcopy(optimizer.state_dict())
    for state in cloned["state"].values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.clone()
    return cloned


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


def test_update_decomposition_groups_and_adam_moments_are_analytic() -> None:
    model = _TwoGroupModel()
    optimizer = _optimizer(model)
    model().backward()
    capture = capture_optimization_step(model, optimizer)
    optimizer.step()
    record = finalize_optimization_step(capture, model, optimizer, step=1)

    expected_decay = torch.tensor([-0.02, 0.04], dtype=torch.float64)
    expected_data = torch.tensor([-0.1, 0.1, -0.1], dtype=torch.float64)
    expected_total = torch.tensor([-0.12, 0.14, -0.1], dtype=torch.float64)
    gradient = torch.tensor([0.5, -0.25, 0.75], dtype=torch.float64)
    second_moment = gradient.square()

    assert record["schema_version"] == 1
    assert all(value is None or type(value) in {int, float} for value in record.values())
    assert record["gradient_l2_total"] == pytest.approx(torch.linalg.vector_norm(gradient).item())
    assert record["gradient_l2_decay_group"] == pytest.approx(
        torch.linalg.vector_norm(gradient[:2]).item()
    )
    assert record["gradient_l2_no_decay_group"] == pytest.approx(0.75)
    assert record["decay_update_l2"] == pytest.approx(
        torch.linalg.vector_norm(expected_decay).item()
    )
    assert record["data_update_l2"] == pytest.approx(
        torch.linalg.vector_norm(expected_data).item(), abs=1e-7
    )
    assert record["total_update_l2"] == pytest.approx(
        torch.linalg.vector_norm(expected_total).item(), abs=1e-7
    )
    assert record["data_to_decay_ratio"] == pytest.approx(
        torch.linalg.vector_norm(expected_data).item()
        / torch.linalg.vector_norm(expected_decay).item(),
        abs=2e-6,
    )
    assert record["data_decay_cosine"] == pytest.approx(
        torch.dot(expected_data[:2], expected_decay).item()
        / (
            torch.linalg.vector_norm(expected_data) * torch.linalg.vector_norm(expected_decay)
        ).item(),
        abs=1e-6,
    )
    assert record["adam_first_moment_l2"] == pytest.approx(
        torch.linalg.vector_norm(gradient).item()
    )
    assert record["adam_second_moment_mean"] == pytest.approx(second_moment.mean().item())
    assert record["adam_second_moment_rms"] == pytest.approx(
        torch.sqrt(torch.mean(second_moment.square())).item()
    )
    assert record["adam_second_moment_max"] == pytest.approx(second_moment.max().item())

    assert record["decay_group_total_update_l2"] == pytest.approx(
        torch.linalg.vector_norm(expected_total[:2]).item(), abs=1e-7
    )
    assert record["no_decay_group_total_update_l2"] == pytest.approx(0.1, abs=1e-7)
    assert record["adam_first_moment_l2_decay_group"] == pytest.approx(
        torch.linalg.vector_norm(gradient[:2]).item()
    )
    assert record["adam_first_moment_l2_no_decay_group"] == pytest.approx(0.75)
    reconstructed_total_square = (
        record["data_update_l2"] ** 2
        + record["decay_update_l2"] ** 2
        + 2.0 * record["data_decay_cosine"] * record["data_update_l2"] * record["decay_update_l2"]
    )
    assert reconstructed_total_square == pytest.approx(record["total_update_l2"] ** 2, abs=1e-12)


def test_grad_none_is_not_counted_as_decay_or_updated() -> None:
    class _UnusedParameterModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.used = nn.Parameter(torch.tensor([1.0]))
            self.unused = nn.Parameter(torch.tensor([2.0]))

        def forward(self) -> torch.Tensor:
            return self.used.sum()

    model = _UnusedParameterModel()
    optimizer = AdamW(
        [
            {
                "params": [model.used, model.unused],
                "parameter_names": ["used", "unused"],
                "group_name": "decay",
                "lr": 0.1,
                "weight_decay": 0.5,
            },
            {
                "params": [],
                "parameter_names": [],
                "group_name": "no_decay",
                "lr": 0.1,
                "weight_decay": 0.0,
            },
        ],
        betas=(0.0, 0.0),
        eps=0.0,
    )
    model().backward()
    capture = capture_optimization_step(model, optimizer)
    optimizer.step()
    record = finalize_optimization_step(capture, model, optimizer, step=1)

    assert model.unused.item() == 2.0
    assert record["parameter_tensor_count"] == 2
    assert record["updated_parameter_tensor_count"] == 1
    assert record["updated_parameter_element_count"] == 1
    assert record["decay_update_l2"] == pytest.approx(0.05)


def test_undefined_ratio_cosine_and_moments_are_null() -> None:
    model = _TwoGroupModel()
    optimizer = _optimizer(model)
    capture = capture_optimization_step(model, optimizer)
    record = finalize_optimization_step(capture, model, optimizer, step=1)
    assert record["data_to_decay_ratio"] is None
    assert record["data_decay_cosine"] is None
    assert record["adam_first_moment_l2"] is None
    assert record["adam_second_moment_mean"] is None
    assert record["adam_second_moment_rms"] is None
    assert record["adam_second_moment_max"] is None


def test_capture_and_finalize_have_no_additional_side_effects() -> None:
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    model = _TwoGroupModel()
    optimizer = _optimizer(model)
    model().backward()

    parameter_before_capture = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    gradients_before_capture = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    state_before_capture = _clone_optimizer_state(optimizer)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()

    capture = capture_optimization_step(model, optimizer)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, parameter_before_capture[name])
        assert torch.equal(parameter.grad, gradients_before_capture[name])
    _assert_nested_equal(optimizer.state_dict(), state_before_capture)
    assert random.getstate() == python_rng
    assert np.array_equal(np.random.get_state()[1], numpy_rng[1])
    assert torch.equal(torch.get_rng_state(), torch_rng)

    optimizer.step()
    parameters_after_step = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    gradients_after_step = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    state_after_step = _clone_optimizer_state(optimizer)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()

    record = finalize_optimization_step(capture, model, optimizer, step=1)
    assert record["step"] == 1
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, parameters_after_step[name])
        assert torch.equal(parameter.grad, gradients_after_step[name])
    _assert_nested_equal(optimizer.state_dict(), state_after_step)
    assert random.getstate() == python_rng
    assert np.array_equal(np.random.get_state()[1], numpy_rng[1])
    assert torch.equal(torch.get_rng_state(), torch_rng)


def test_capture_rejects_missing_stable_parameter_names() -> None:
    model = _TwoGroupModel()
    optimizer = _optimizer(model)
    del optimizer.param_groups[0]["parameter_names"]
    with pytest.raises(ValueError, match="parameter_names"):
        capture_optimization_step(model, optimizer)


def test_nonfinite_gradient_is_rejected() -> None:
    model = _TwoGroupModel()
    optimizer = _optimizer(model)
    model.decay_weight.grad = torch.tensor([math.inf, 0.0])
    model.no_decay_bias.grad = torch.tensor([0.0])
    capture = capture_optimization_step(model, optimizer)
    with pytest.raises(ValueError, match="non-finite"):
        finalize_optimization_step(capture, model, optimizer, step=1)
