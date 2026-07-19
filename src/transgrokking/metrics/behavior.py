"""Pure classification metrics and M1 model evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from transgrokking.metrics.norms import parameter_norm_metrics
from transgrokking.training.optimizer import ParameterGrouping

QUANTILES: tuple[tuple[str, float], ...] = (
    ("q01", 0.01),
    ("q05", 0.05),
    ("q25", 0.25),
    ("median", 0.50),
    ("q75", 0.75),
    ("q95", 0.95),
    ("q99", 0.99),
)


@dataclass(frozen=True)
class SplitBehavior:
    """Scalar behavior metrics and error-only cyclic offset counts."""

    scalars: dict[str, float | int]
    error_offsets: list[int]


def classification_margins(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return `z_y - max_{c != y} z_c` for logits `[batch, classes]`."""
    if logits.ndim != 2:
        raise ValueError(f"logits: expected [batch, classes], got {tuple(logits.shape)}")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError(f"labels: expected [{logits.shape[0]}], got {tuple(labels.shape)}")
    if logits.shape[0] < 1 or logits.shape[1] < 2:
        raise ValueError("logits: batch and classes must be nonempty with classes >= 2")
    if labels.dtype != torch.long:
        raise ValueError(f"labels: expected torch.long, got {labels.dtype}")
    if torch.any(labels < 0) or torch.any(labels >= logits.shape[1]):
        raise ValueError("labels: class index outside logits range")
    correct = logits.gather(1, labels[:, None]).squeeze(1)
    incorrect = logits.clone()
    incorrect.scatter_(1, labels[:, None], float("-inf"))
    return correct - incorrect.max(dim=1).values


def margin_summary(margins: torch.Tensor) -> dict[str, float]:
    """Summarize a nonempty margin vector using FP64 reductions."""
    if margins.ndim != 1 or margins.numel() < 1:
        raise ValueError(f"margins: expected nonempty vector, got {tuple(margins.shape)}")
    values = margins.detach().to(dtype=torch.float64)
    if not torch.isfinite(values).all():
        raise ValueError("margins: all values must be finite")
    result = {"mean": values.mean().item(), "min": values.min().item()}
    for name, quantile in QUANTILES:
        result[name] = torch.quantile(values, quantile).item()
    return result


def error_offset_histogram(
    predictions: torch.Tensor, labels: torch.Tensor, modulus: int
) -> list[int]:
    """Count `(prediction-label) mod p` for errors only; bin zero remains zero."""
    if modulus < 2:
        raise ValueError(f"modulus: expected >= 2, got {modulus!r}")
    if predictions.ndim != 1 or labels.ndim != 1 or predictions.shape != labels.shape:
        raise ValueError(
            "predictions and labels: expected equal vectors, got "
            f"{tuple(predictions.shape)} and {tuple(labels.shape)}"
        )
    errors = predictions != labels
    offsets = (predictions[errors] - labels[errors]).remainder(modulus)
    counts = torch.bincount(offsets.to("cpu"), minlength=modulus)
    counts[0] = 0
    return [int(value) for value in counts.tolist()]


def evaluate_logits(logits: torch.Tensor, labels: torch.Tensor, modulus: int) -> SplitBehavior:
    """Compute CE, accuracy, margins, errors, and error offsets for one split."""
    margins = classification_margins(logits, labels)
    predictions = logits.argmax(dim=1)
    errors = predictions != labels
    error_count = int(errors.sum().item())
    batch_size = labels.numel()
    scalars: dict[str, float | int] = {
        "cross_entropy": F.cross_entropy(logits, labels).item(),
        "accuracy": (predictions == labels).to(torch.float32).mean().item(),
        "error_count": error_count,
        "error_rate": error_count / batch_size,
    }
    scalars.update({f"margin_{name}": value for name, value in margin_summary(margins).items()})
    return SplitBehavior(
        scalars=scalars,
        error_offsets=error_offset_histogram(predictions, labels, modulus),
    )


def evaluate_model_behavior(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    test_indices: torch.Tensor,
    modulus: int,
    grouping: ParameterGrouping,
) -> tuple[dict[str, float | int | None], dict[str, list[int]]]:
    """Evaluate train/test behavior without retaining `[p*p,p]` logits."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            results: dict[str, SplitBehavior] = {}
            for split, indices in (("train", train_indices), ("test", test_indices)):
                split_inputs = inputs.index_select(0, indices)
                split_labels = labels.index_select(0, indices)
                logits = model(split_inputs)[:, -1]
                results[split] = evaluate_logits(logits, split_labels, modulus)
                del logits, split_inputs, split_labels
    finally:
        model.train(was_training)
    scalars: dict[str, float | int | None] = {"congruence_loss": 0.0}
    offsets: dict[str, list[int]] = {}
    for split, result in results.items():
        scalars.update({f"{split}_{name}": value for name, value in result.scalars.items()})
        offsets[split] = result.error_offsets
    scalars.update(parameter_norm_metrics(model, grouping))
    return scalars, offsets
