from __future__ import annotations

import pytest
import torch

from transgrokking.metrics.behavior import (
    classification_margins,
    error_offset_histogram,
    evaluate_logits,
    margin_summary,
)


def test_margins_exclude_correct_class_and_cover_positive_zero_negative() -> None:
    logits = torch.tensor([[3.0, 1.0, 2.0], [0.0, 2.0, 2.0], [5.0, 0.0, 1.0]])
    labels = torch.tensor([0, 1, 2])
    margins = classification_margins(logits, labels)
    assert torch.equal(margins, torch.tensor([1.0, 0.0, -4.0]))
    summary = margin_summary(margins)
    values = margins.double()
    assert summary["mean"] == pytest.approx(values.mean().item())
    assert summary["min"] == -4.0
    for name, quantile in (
        ("q01", 0.01),
        ("q05", 0.05),
        ("q25", 0.25),
        ("median", 0.5),
        ("q75", 0.75),
        ("q95", 0.95),
        ("q99", 0.99),
    ):
        assert summary[name] == pytest.approx(torch.quantile(values, quantile).item())


def test_margin_batch_one_and_two_class_boundary() -> None:
    margins = classification_margins(torch.tensor([[1.5, 2.0]]), torch.tensor([0]))
    assert margins.shape == (1,)
    assert margins.item() == -0.5
    assert margin_summary(margins)["median"] == -0.5


def test_error_offsets_use_modulus_and_count_errors_only() -> None:
    predictions = torch.tensor([0, 1, 0, 4])
    labels = torch.tensor([0, 2, 2, 0])
    counts = error_offset_histogram(predictions, labels, 5)
    assert counts == [0, 0, 0, 1, 2]
    assert sum(counts) == 3
    assert error_offset_histogram(torch.tensor([0, 1]), torch.tensor([0, 1]), 2) == [0, 0]


def test_split_behavior_reports_legal_all_correct_histogram() -> None:
    logits = torch.tensor([[3.0, 1.0], [0.0, 2.0]])
    result = evaluate_logits(logits, torch.tensor([0, 1]), 2)
    assert result.scalars["accuracy"] == 1.0
    assert result.scalars["error_count"] == 0
    assert result.scalars["error_rate"] == 0.0
    assert result.error_offsets == [0, 0]
