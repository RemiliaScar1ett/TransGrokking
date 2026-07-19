"""Non-overlapping FP64 model and optimizer-group parameter norms."""

from __future__ import annotations

from collections import defaultdict

from torch import nn

from transgrokking.training.optimizer import ParameterGrouping


def parameter_module_name(parameter_name: str) -> str:
    """Map one stable model parameter name to one M1 module bucket."""
    if parameter_name.startswith("token_embedding."):
        return "token_embedding"
    if parameter_name.startswith("position_embedding."):
        return "position_embedding"
    if parameter_name.startswith("final_norm."):
        return "final_norm"
    if parameter_name.startswith("unembedding."):
        return "unembedding"
    parts = parameter_name.split(".")
    if len(parts) >= 4 and parts[0] == "blocks" and parts[1].isdigit():
        prefix = f"blocks.{parts[1]}"
        if parts[2] == "attention":
            return f"{prefix}.attention"
        if parts[2] in {"mlp_in", "mlp_out"}:
            return f"{prefix}.mlp"
        if parts[2] in {"ln1", "ln2"}:
            return f"{prefix}.layer_norm"
    raise ValueError(f"parameter module mapping is undefined for {parameter_name!r}")


def _norm(square_sum: float) -> float:
    return square_sum**0.5


def parameter_norm_metrics(
    model: nn.Module, grouping: ParameterGrouping
) -> dict[str, float | None]:
    """Return total, non-overlapping module, and parameter-group L2 norms."""
    parameters = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    grouped_names = set(grouping.decay) | set(grouping.no_decay)
    if grouped_names != set(parameters) or set(grouping.decay) & set(grouping.no_decay):
        raise ValueError("parameter grouping must cover every trainable parameter exactly once")
    module_squares: dict[str, float] = defaultdict(float)
    total_square = 0.0
    for name, parameter in parameters.items():
        square = (
            parameter.detach()
            .to(device="cpu", dtype=parameter.dtype)
            .double()
            .square()
            .sum()
            .item()
        )
        module_squares[parameter_module_name(name)] += square
        total_square += square
    output: dict[str, float | None] = {"parameter_norm_total": _norm(total_square)}
    for module_name in sorted(module_squares):
        output[f"parameter_norm_{module_name}"] = _norm(module_squares[module_name])
    if "parameter_norm_final_norm" not in output:
        output["parameter_norm_final_norm"] = None
    for group_name, names in (("decay", grouping.decay), ("no_decay", grouping.no_decay)):
        square = sum(
            parameters[name].detach().to(device="cpu").double().square().sum().item()
            for name in names
        )
        output[f"parameter_group_norm_{group_name}"] = _norm(square)
    return output
