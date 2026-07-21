"""Auditable AdamW parameter grouping."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn
from torch.optim import AdamW, Optimizer

from transgrokking.config import OptimizationConfig


@dataclass(frozen=True)
class ParameterGrouping:
    """Stable parameter names assigned to AdamW decay and no-decay groups."""

    decay: tuple[str, ...]
    no_decay: tuple[str, ...]

    def signature(self) -> list[dict[str, object]]:
        """Return a checkpoint-compatible group structure signature."""
        return [
            {"group_name": "decay", "parameter_names": list(self.decay)},
            {"group_name": "no_decay", "parameter_names": list(self.no_decay)},
        ]


def classify_parameters(model: nn.Module, optimization: OptimizationConfig) -> ParameterGrouping:
    """Classify every trainable parameter exactly once using module type and name."""
    policy = optimization.decay_policy
    decisions: dict[str, bool] = {}
    for module_name, module in model.named_modules():
        for local_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            full_name = f"{module_name}.{local_name}" if module_name else local_name
            if full_name in decisions:
                raise ValueError(f"optimizer parameter appears more than once: {full_name}")
            if isinstance(module, nn.Embedding):
                use_decay = policy.embeddings
            elif isinstance(module, nn.LayerNorm):
                use_decay = policy.layer_norm
            elif local_name == "bias":
                use_decay = policy.biases
            elif local_name == "weight" and parameter.ndim >= 2:
                use_decay = policy.matrix_weights
            else:
                raise ValueError(
                    f"optimizer decay policy cannot classify {full_name} with shape "
                    f"{tuple(parameter.shape)}"
                )
            decisions[full_name] = use_decay

    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(decisions) != expected:
        missing = sorted(expected - set(decisions))
        extra = sorted(set(decisions) - expected)
        raise ValueError(f"optimizer parameter coverage mismatch: missing={missing}, extra={extra}")
    decay = tuple(sorted(name for name, enabled in decisions.items() if enabled))
    no_decay = tuple(sorted(name for name, enabled in decisions.items() if not enabled))
    return ParameterGrouping(decay=decay, no_decay=no_decay)


def build_adamw(
    model: nn.Module, optimization: OptimizationConfig
) -> tuple[AdamW, ParameterGrouping]:
    """Build AdamW with stable named decay and no-decay groups."""
    grouping = classify_parameters(model, optimization)
    parameters = dict(model.named_parameters())
    groups = [
        {
            "params": [parameters[name] for name in grouping.decay],
            "lr": optimization.learning_rate,
            "weight_decay": optimization.weight_decay,
            "group_name": "decay",
            "parameter_names": list(grouping.decay),
        },
        {
            "params": [parameters[name] for name in grouping.no_decay],
            "lr": optimization.learning_rate,
            "weight_decay": 0.0,
            "group_name": "no_decay",
            "parameter_names": list(grouping.no_decay),
        },
    ]
    return AdamW(groups), grouping


def validate_optimizer_parameter_identity(model: nn.Module, optimizer: Optimizer) -> None:
    """Require optimizer groups to reference every trainable model parameter exactly once."""
    model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    model_ids = {id(parameter) for parameter in model_parameters}
    optimizer_ids = {id(parameter) for parameter in optimizer_parameters}
    if len(optimizer_parameters) != len(optimizer_ids):
        raise ValueError("optimizer contains duplicate parameter objects")
    if model_ids != optimizer_ids:
        missing = len(model_ids - optimizer_ids)
        extra = len(optimizer_ids - model_ids)
        raise ValueError(
            f"optimizer/model parameter identity mismatch: missing={missing}, extra={extra}"
        )


def optimizer_group_metadata(optimizer: AdamW) -> list[dict[str, object]]:
    """Return group names, hyperparameters, and parameter names for run metadata."""
    return [
        {
            "group_name": group["group_name"],
            "learning_rate": group["lr"],
            "weight_decay": group["weight_decay"],
            "parameter_names": list(group["parameter_names"]),
        }
        for group in optimizer.param_groups
    ]
