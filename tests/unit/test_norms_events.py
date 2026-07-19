from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from transgrokking.config import config_from_dict, load_config
from transgrokking.metrics.events import detect_events
from transgrokking.metrics.norms import parameter_norm_metrics
from transgrokking.training.optimizer import build_adamw
from transgrokking.training.trainer import build_model


@pytest.mark.parametrize("final_norm", [False, True])
def test_parameter_norm_modules_and_groups_reconstruct_total(final_norm: bool) -> None:
    raw = load_config("configs/smoke.yaml").to_dict()
    raw["model"]["final_norm"] = final_norm
    config = config_from_dict(raw)
    model = build_model(config)
    optimizer, grouping = build_adamw(model, config.optimization)
    metrics = parameter_norm_metrics(model, grouping)
    module_values = [
        value
        for name, value in metrics.items()
        if name.startswith("parameter_norm_")
        and name != "parameter_norm_total"
        and value is not None
    ]
    assert sum(value**2 for value in module_values) == pytest.approx(
        metrics["parameter_norm_total"] ** 2
    )
    assert metrics["parameter_group_norm_decay"] ** 2 + metrics[
        "parameter_group_norm_no_decay"
    ] ** 2 == pytest.approx(metrics["parameter_norm_total"] ** 2)
    assert (metrics["parameter_norm_final_norm"] is None) is (not final_norm)
    before = dict(metrics)
    first_parameter = next(iter(model.parameters()))
    optimizer.state[first_parameter]["unrelated_optimizer_state"] = torch.full((10,), 1e30)
    assert parameter_norm_metrics(model, grouping) == before


def _records(train: list[float], test: list[float]) -> list[dict[str, float | int]]:
    return [
        {"step": index * 10, "train_accuracy": train_value, "test_accuracy": test_value}
        for index, (train_value, test_value) in enumerate(zip(train, test, strict=True), start=1)
    ]


def test_events_require_consecutive_evaluations_and_keep_first_window() -> None:
    base = load_config("configs/smoke.yaml").events
    config = replace(base, fit_consecutive=3, grok50_consecutive=2, grok99_consecutive=2)
    records = _records(
        [0.999, 0.5, 0.999, 0.999, 0.999, 0.1],
        [0.2, 0.7, 0.8, 0.995, 0.995, 0.1],
    )
    events = detect_events(records, "run", 4, 10, config)
    assert events["t_fit"]["event_step"] == 30
    assert events["t_fit"]["detected_at_evaluation_step"] == 50
    assert events["t_grok50"]["threshold"] == pytest.approx(0.625)
    assert events["t_grok50"]["event_step"] == 20
    assert events["t_grok99"]["event_step"] == 40
    extended = detect_events(
        records + [{"step": 70, "train_accuracy": 1.0, "test_accuracy": 1.0}],
        "run",
        4,
        10,
        replace(config, fit_accuracy=1.0, grok99_accuracy=1.0),
        events,
    )
    assert extended["t_fit"] == events["t_fit"]


def test_event_resume_append_matches_full_recompute_and_rejects_bad_steps() -> None:
    config = replace(load_config("configs/smoke.yaml").events, fit_consecutive=2)
    prefix = _records([0.999], [0.1])
    assert detect_events(prefix, "run", 7, 10, config)["t_fit"]["status"] == "not_reached"
    combined = prefix + [{"step": 20, "train_accuracy": 1.0, "test_accuracy": 0.1}]
    resumed = detect_events(combined, "run", 7, 10, config)
    assert resumed["t_fit"]["event_step"] == 10
    with pytest.raises(ValueError, match="strictly increasing"):
        detect_events([combined[1], combined[0]], "run", 7, 10, config)
    with pytest.raises(ValueError, match="strictly increasing"):
        detect_events([combined[0], combined[0]], "run", 7, 10, config)
