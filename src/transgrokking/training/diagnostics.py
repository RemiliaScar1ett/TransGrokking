"""Read-only optimization diagnostics aligned with a real AdamW update."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

OPTIMIZATION_DIAGNOSTICS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ParameterStepSnapshot:
    """One named parameter immediately after backward and before optimizer.step()."""

    name: str
    group_name: str
    parameter: nn.Parameter
    value: Tensor
    gradient: Tensor | None
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True)
class OptimizationStepCapture:
    """Immutable CPU snapshots needed to decompose one optimizer update."""

    parameters: tuple[ParameterStepSnapshot, ...]
    group_names: tuple[str, ...]
    optimizer_id: int


@dataclass
class _Accumulator:
    parameter_tensor_count: int = 0
    parameter_element_count: int = 0
    updated_parameter_tensor_count: int = 0
    updated_parameter_element_count: int = 0
    gradient_square_sum: float = 0.0
    total_update_square_sum: float = 0.0
    data_update_square_sum: float = 0.0
    decay_update_square_sum: float = 0.0
    data_decay_dot: float = 0.0
    first_moment_square_sum: float = 0.0
    second_moment_sum: float = 0.0
    second_moment_square_sum: float = 0.0
    second_moment_max: float | None = None
    moment_element_count: int = 0

    def add_parameter(self, snapshot: ParameterStepSnapshot, optimizer: Optimizer) -> None:
        parameter = snapshot.parameter
        after = parameter.detach().to(device="cpu", dtype=torch.float64).clone()
        if after.shape != snapshot.value.shape:
            raise ValueError(
                f"optimization diagnostic parameter shape changed for {snapshot.name}: "
                f"{tuple(snapshot.value.shape)} -> {tuple(after.shape)}"
            )
        total_update = after - snapshot.value
        if snapshot.gradient is None:
            decay_update = torch.zeros_like(total_update)
            data_update = total_update
        else:
            decay_update = snapshot.value * (-snapshot.learning_rate * snapshot.weight_decay)
            data_update = total_update - decay_update
            self.updated_parameter_tensor_count += 1
            self.updated_parameter_element_count += parameter.numel()
            self.gradient_square_sum += _square_sum(snapshot.gradient)

        self.parameter_tensor_count += 1
        self.parameter_element_count += parameter.numel()
        self.total_update_square_sum += _square_sum(total_update)
        self.data_update_square_sum += _square_sum(data_update)
        self.decay_update_square_sum += _square_sum(decay_update)
        self.data_decay_dot += _dot(data_update, decay_update)

        state = optimizer.state.get(parameter)
        if not state:
            return
        first_moment = state.get("exp_avg")
        second_moment = state.get("exp_avg_sq")
        if first_moment is None and second_moment is None:
            return
        if not isinstance(first_moment, Tensor) or not isinstance(second_moment, Tensor):
            raise ValueError(f"optimization diagnostic expected Adam moments for {snapshot.name}")
        first = first_moment.detach().to(device="cpu", dtype=torch.float64)
        second = second_moment.detach().to(device="cpu", dtype=torch.float64)
        if first.shape != parameter.shape or second.shape != parameter.shape:
            raise ValueError(
                f"optimization diagnostic Adam moment shape mismatch for {snapshot.name}"
            )
        self.first_moment_square_sum += _square_sum(first)
        self.second_moment_sum += _sum(second)
        self.second_moment_square_sum += _square_sum(second)
        second_max = _max(second)
        self.second_moment_max = (
            second_max
            if self.second_moment_max is None
            else max(self.second_moment_max, second_max)
        )
        self.moment_element_count += second.numel()

    def record(self) -> dict[str, int | float | None]:
        gradient_l2 = _sqrt(self.gradient_square_sum)
        total_update_l2 = _sqrt(self.total_update_square_sum)
        data_update_l2 = _sqrt(self.data_update_square_sum)
        decay_update_l2 = _sqrt(self.decay_update_square_sum)
        data_decay_ratio = data_update_l2 / decay_update_l2 if decay_update_l2 > 0.0 else None
        denominator = data_update_l2 * decay_update_l2
        data_decay_cosine = self.data_decay_dot / denominator if denominator > 0.0 else None
        if data_decay_cosine is not None:
            data_decay_cosine = min(1.0, max(-1.0, data_decay_cosine))

        if self.moment_element_count:
            second_mean = self.second_moment_sum / self.moment_element_count
            second_rms = _sqrt(self.second_moment_square_sum / self.moment_element_count)
            first_l2: float | None = _sqrt(self.first_moment_square_sum)
            second_max = self.second_moment_max
        else:
            first_l2 = None
            second_mean = None
            second_rms = None
            second_max = None

        record: dict[str, int | float | None] = {
            "parameter_tensor_count": self.parameter_tensor_count,
            "parameter_element_count": self.parameter_element_count,
            "updated_parameter_tensor_count": self.updated_parameter_tensor_count,
            "updated_parameter_element_count": self.updated_parameter_element_count,
            "gradient_l2": gradient_l2,
            "total_update_l2": total_update_l2,
            "data_update_l2": data_update_l2,
            "decay_update_l2": decay_update_l2,
            "data_decay_ratio": data_decay_ratio,
            "data_decay_cosine": data_decay_cosine,
            "adam_first_moment_l2": first_l2,
            "adam_second_moment_mean": second_mean,
            "adam_second_moment_rms": second_rms,
            "adam_second_moment_max": second_max,
        }
        for field, value in record.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"optimization diagnostic produced non-finite {field}: {value!r}")
        return record


def capture_optimization_step(model: nn.Module, optimizer: Optimizer) -> OptimizationStepCapture:
    """Capture values and gradients after backward without mutating training state."""
    names_by_id = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    expected_ids = set(names_by_id)
    seen_ids: set[int] = set()
    snapshots: list[ParameterStepSnapshot] = []
    seen_group_names: set[str] = set()

    for group_index, group in enumerate(optimizer.param_groups):
        group_name = group.get("group_name")
        parameter_names = group.get("parameter_names")
        if not isinstance(group_name, str) or not group_name:
            raise ValueError(f"optimizer group {group_index} lacks a stable non-empty group_name")
        if group_name in seen_group_names:
            raise ValueError(f"optimizer group name appears more than once: {group_name}")
        seen_group_names.add(group_name)
        if not isinstance(parameter_names, (list, tuple)):
            raise ValueError(f"optimizer group {group_name!r} lacks stable parameter_names")
        parameters = group.get("params")
        if not isinstance(parameters, list) or len(parameters) != len(parameter_names):
            raise ValueError(f"optimizer group {group_name!r} parameter_names do not match params")
        learning_rate = _group_float(group, "lr", group_name)
        weight_decay = _group_float(group, "weight_decay", group_name)

        for parameter, recorded_name in zip(parameters, parameter_names, strict=True):
            if not isinstance(parameter, nn.Parameter):
                raise ValueError(f"optimizer group {group_name!r} contains a non-Parameter value")
            parameter_id = id(parameter)
            actual_name = names_by_id.get(parameter_id)
            if actual_name is None:
                raise ValueError(
                    f"optimizer parameter {recorded_name!r} is not a trainable model parameter"
                )
            if recorded_name != actual_name:
                raise ValueError(
                    f"optimizer parameter name mismatch: {recorded_name!r} != {actual_name!r}"
                )
            if parameter_id in seen_ids:
                raise ValueError(f"optimizer parameter appears more than once: {actual_name}")
            seen_ids.add(parameter_id)
            gradient = (
                None
                if parameter.grad is None
                else parameter.grad.detach().to(device="cpu", dtype=torch.float64).clone()
            )
            snapshots.append(
                ParameterStepSnapshot(
                    name=actual_name,
                    group_name=group_name,
                    parameter=parameter,
                    value=parameter.detach().to(device="cpu", dtype=torch.float64).clone(),
                    gradient=gradient,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                )
            )

    if seen_ids != expected_ids:
        missing = sorted(names_by_id[parameter_id] for parameter_id in expected_ids - seen_ids)
        raise ValueError(f"optimizer/model parameter coverage mismatch: missing={missing}")
    return OptimizationStepCapture(
        parameters=tuple(snapshots),
        group_names=tuple(str(group["group_name"]) for group in optimizer.param_groups),
        optimizer_id=id(optimizer),
    )


def finalize_optimization_step(
    capture: OptimizationStepCapture,
    model: nn.Module,
    optimizer: Optimizer,
    *,
    step: int,
) -> dict[str, Any]:
    """Measure the completed update and post-step Adam moments without mutation."""
    if type(step) is not int or step < 1:
        raise ValueError(f"step: expected positive integer, got {step!r}")
    if id(optimizer) != capture.optimizer_id:
        raise ValueError("optimization capture belongs to a different optimizer")
    if set(capture.group_names) != {"decay", "no_decay"}:
        raise ValueError(
            "optimization diagnostics require stable decay and no_decay groups, got "
            f"{list(capture.group_names)!r}"
        )
    current_parameters = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    captured_names = [snapshot.name for snapshot in capture.parameters]
    if len(captured_names) != len(set(captured_names)):
        raise ValueError("optimization capture contains duplicate parameter names")
    if set(captured_names) != set(current_parameters):
        raise ValueError("optimization capture no longer covers the model parameters")

    global_accumulator = _Accumulator()
    group_accumulators = {group_name: _Accumulator() for group_name in capture.group_names}
    for snapshot in capture.parameters:
        if current_parameters[snapshot.name] is not snapshot.parameter:
            raise ValueError(f"optimization capture parameter identity changed for {snapshot.name}")
        accumulator = group_accumulators[snapshot.group_name]
        accumulator.add_parameter(snapshot, optimizer)
        global_accumulator.add_parameter(snapshot, optimizer)

    global_record = global_accumulator.record()
    decay_record = group_accumulators["decay"].record()
    no_decay_record = group_accumulators["no_decay"].record()
    return {
        "schema_version": OPTIMIZATION_DIAGNOSTICS_SCHEMA_VERSION,
        "step": step,
        "parameter_tensor_count": global_record["parameter_tensor_count"],
        "parameter_element_count": global_record["parameter_element_count"],
        "updated_parameter_tensor_count": global_record["updated_parameter_tensor_count"],
        "updated_parameter_element_count": global_record["updated_parameter_element_count"],
        "gradient_l2_total": global_record["gradient_l2"],
        "gradient_l2_decay_group": decay_record["gradient_l2"],
        "gradient_l2_no_decay_group": no_decay_record["gradient_l2"],
        "total_update_l2": global_record["total_update_l2"],
        "data_update_l2": global_record["data_update_l2"],
        "decay_update_l2": global_record["decay_update_l2"],
        "data_to_decay_ratio": global_record["data_decay_ratio"],
        "data_decay_cosine": global_record["data_decay_cosine"],
        "decay_group_total_update_l2": decay_record["total_update_l2"],
        "no_decay_group_total_update_l2": no_decay_record["total_update_l2"],
        "adam_first_moment_l2": global_record["adam_first_moment_l2"],
        "adam_first_moment_l2_decay_group": decay_record["adam_first_moment_l2"],
        "adam_first_moment_l2_no_decay_group": no_decay_record["adam_first_moment_l2"],
        "adam_second_moment_mean": global_record["adam_second_moment_mean"],
        "adam_second_moment_rms": global_record["adam_second_moment_rms"],
        "adam_second_moment_max": global_record["adam_second_moment_max"],
    }


def _group_float(group: dict[str, Any], field: str, group_name: str) -> float:
    value = group.get(field, 0.0)
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"optimizer group {group_name!r} has non-scalar {field}: {value!r}")
        value = value.detach().cpu().item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"optimizer group {group_name!r} has invalid {field}: {value!r}")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"optimizer group {group_name!r} has invalid {field}: {value!r}")
    return result


def _sum(tensor: Tensor) -> float:
    value = float(tensor.sum(dtype=torch.float64).item())
    if not math.isfinite(value):
        raise ValueError("optimization diagnostic encountered a non-finite tensor sum")
    return value


def _square_sum(tensor: Tensor) -> float:
    value = float(torch.sum(tensor * tensor, dtype=torch.float64).item())
    if not math.isfinite(value):
        raise ValueError("optimization diagnostic encountered a non-finite tensor square sum")
    return value


def _dot(left: Tensor, right: Tensor) -> float:
    value = float(torch.sum(left * right, dtype=torch.float64).item())
    if not math.isfinite(value):
        raise ValueError("optimization diagnostic encountered a non-finite tensor dot product")
    return value


def _max(tensor: Tensor) -> float:
    value = float(torch.max(tensor).item())
    if not math.isfinite(value):
        raise ValueError("optimization diagnostic encountered a non-finite tensor maximum")
    return value


def _sqrt(value: float) -> float:
    result = math.sqrt(max(0.0, value))
    if not math.isfinite(result):
        raise ValueError("optimization diagnostic encountered a non-finite norm")
    return result
