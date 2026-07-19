"""Idempotent M1 behavioral event detection from scalar records."""

from __future__ import annotations

from typing import Any

from transgrokking.config import EventsConfig

EVENTS_SCHEMA_VERSION = 1


def validate_scalar_records(records: list[dict[str, Any]]) -> None:
    """Require unique, strictly increasing integer evaluation steps."""
    steps = [record.get("step") for record in records]
    if any(type(step) is not int for step in steps):
        raise ValueError(f"scalar record steps must be integers: {steps}")
    if any(current <= previous for previous, current in zip(steps, steps[1:], strict=False)):
        raise ValueError(f"scalar record steps must be strictly increasing: {steps}")


def _event(
    records: list[dict[str, Any]],
    field: str,
    threshold: float,
    consecutive: int,
) -> dict[str, Any]:
    for end in range(consecutive - 1, len(records)):
        window = records[end - consecutive + 1 : end + 1]
        if all(float(record[field]) >= threshold for record in window):
            return {
                "status": "reached",
                "event_step": window[0]["step"],
                "threshold": threshold,
                "required_consecutive_evaluations": consecutive,
                "detected_at_evaluation_step": window[-1]["step"],
            }
    return {
        "status": "not_reached",
        "event_step": None,
        "threshold": threshold,
        "required_consecutive_evaluations": consecutive,
        "detected_at_evaluation_step": None,
    }


def detect_events(
    records: list[dict[str, Any]],
    run_id: str,
    modulus: int,
    eval_interval: int,
    config: EventsConfig,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect M1 events, preserving events already reached in the same run."""
    validate_scalar_records(records)
    candidates = {
        "t_fit": _event(records, "train_accuracy", config.fit_accuracy, config.fit_consecutive),
        "t_grok50": _event(
            records,
            "test_accuracy",
            (1.0 + 1.0 / modulus) / 2.0,
            config.grok50_consecutive,
        ),
        "t_grok99": _event(
            records,
            "test_accuracy",
            config.grok99_accuracy,
            config.grok99_consecutive,
        ),
    }
    if existing is not None:
        for name in candidates:
            previous = existing.get(name)
            if isinstance(previous, dict) and previous.get("status") == "reached":
                candidates[name] = previous
    return {
        "schema_version": EVENTS_SCHEMA_VERSION,
        "run_id": run_id,
        "modulus": modulus,
        "eval_interval": eval_interval,
        **candidates,
        "last_evaluated_step": records[-1]["step"] if records else None,
    }
