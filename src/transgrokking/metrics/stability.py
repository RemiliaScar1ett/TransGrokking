"""Pure M1-C stability measurements and strict measurement configuration."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from transgrokking.utils.atomic import write_yaml

STABILITY_SCHEMA_VERSION = 1
MEASUREMENT_SCHEMA_VERSION = 1
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_FINAL_STATES = {
    "high_performance",
    "recovering",
    "collapsed",
    "unstable_but_above_threshold",
}


@dataclass(frozen=True)
class MeasurementSourceConfig:
    """Frozen M1-B source identity for the M1-C extension."""

    canonical_run_id: str
    canonical_checkpoint_step: int
    canonical_checkpoint_sha256: str
    scientific_config_hash: str
    split_hash: str
    eval_interval: int
    checkpoint_interval: int


@dataclass(frozen=True)
class FrozenEventsConfig:
    """First-event steps that M1-C must preserve."""

    t_fit: int
    t_fit_detected_at: int
    t_grok50: int
    t_grok50_detected_at: int
    t_grok99: int
    t_grok99_detected_at: int


@dataclass(frozen=True)
class StabilityConfig:
    """Operational thresholds for M1-C stability measurements."""

    stable_accuracy: float
    stable_window_intervals: int
    collapse_accuracy: float
    train_recovery_accuracy: float
    test_recovery_accuracy: float
    recovery_consecutive: int
    joint_tolerance_evaluations: int


@dataclass(frozen=True)
class MeasurementConfig:
    """Resolved measurement-only M1-C configuration."""

    schema_version: int
    profile: str
    source: MeasurementSourceConfig
    frozen_events: FrozenEventsConfig
    stability: StabilityConfig

    def to_dict(self) -> dict[str, Any]:
        """Return a stable YAML/JSON-serializable representation."""
        return asdict(self)

    def measurement_hash(self) -> str:
        """Return SHA-256 over the complete resolved measurement configuration."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


T = TypeVar("T")


def _strict_section(cls: type[T], value: Any, path: str) -> T:
    if type(value) is not dict:
        raise ValueError(f"{path}: expected mapping, got {value!r}")
    expected = {field.name for field in fields(cls)}
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown or missing:
        raise ValueError(f"{path}: unknown={sorted(unknown)}, missing={sorted(missing)}")
    return cls(**value)


def _require_exact_type(path: str, value: Any, expected: type) -> None:
    if expected is float:
        valid = type(value) in {int, float}
    else:
        valid = type(value) is expected
    if not valid:
        raise ValueError(f"{path}: expected {expected.__name__}, got {value!r}")


def measurement_config_from_dict(raw: dict[str, Any]) -> MeasurementConfig:
    """Strictly parse a measurement sidecar without affecting scientific config."""
    if type(raw) is not dict:
        raise ValueError(f"measurement: expected mapping, got {raw!r}")
    expected = {field.name for field in fields(MeasurementConfig)}
    unknown = set(raw) - expected
    missing = expected - set(raw)
    if unknown or missing:
        raise ValueError(f"measurement: unknown={sorted(unknown)}, missing={sorted(missing)}")
    config = MeasurementConfig(
        schema_version=raw["schema_version"],
        profile=raw["profile"],
        source=_strict_section(MeasurementSourceConfig, raw["source"], "source"),
        frozen_events=_strict_section(FrozenEventsConfig, raw["frozen_events"], "frozen_events"),
        stability=_strict_section(StabilityConfig, raw["stability"], "stability"),
    )
    validate_measurement_config(config)
    return config


def load_measurement_config(path: str | Path) -> MeasurementConfig:
    """Load and strictly validate a UTF-8 M1-C measurement sidecar."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return measurement_config_from_dict(raw)


def dump_measurement_config(config: MeasurementConfig, path: str | Path) -> None:
    """Atomically persist an already validated resolved measurement sidecar."""
    validate_measurement_config(config)
    write_yaml(path, config.to_dict())


def validate_measurement_config(config: MeasurementConfig) -> None:
    """Validate types, identity fields, thresholds, and protocol relationships."""
    typed: list[tuple[str, Any, type]] = [
        ("schema_version", config.schema_version, int),
        ("profile", config.profile, str),
        ("source.canonical_run_id", config.source.canonical_run_id, str),
        (
            "source.canonical_checkpoint_step",
            config.source.canonical_checkpoint_step,
            int,
        ),
        (
            "source.canonical_checkpoint_sha256",
            config.source.canonical_checkpoint_sha256,
            str,
        ),
        ("source.scientific_config_hash", config.source.scientific_config_hash, str),
        ("source.split_hash", config.source.split_hash, str),
        ("source.eval_interval", config.source.eval_interval, int),
        ("source.checkpoint_interval", config.source.checkpoint_interval, int),
        ("frozen_events.t_fit", config.frozen_events.t_fit, int),
        (
            "frozen_events.t_fit_detected_at",
            config.frozen_events.t_fit_detected_at,
            int,
        ),
        ("frozen_events.t_grok50", config.frozen_events.t_grok50, int),
        (
            "frozen_events.t_grok50_detected_at",
            config.frozen_events.t_grok50_detected_at,
            int,
        ),
        ("frozen_events.t_grok99", config.frozen_events.t_grok99, int),
        (
            "frozen_events.t_grok99_detected_at",
            config.frozen_events.t_grok99_detected_at,
            int,
        ),
        ("stability.stable_accuracy", config.stability.stable_accuracy, float),
        (
            "stability.stable_window_intervals",
            config.stability.stable_window_intervals,
            int,
        ),
        ("stability.collapse_accuracy", config.stability.collapse_accuracy, float),
        (
            "stability.train_recovery_accuracy",
            config.stability.train_recovery_accuracy,
            float,
        ),
        (
            "stability.test_recovery_accuracy",
            config.stability.test_recovery_accuracy,
            float,
        ),
        ("stability.recovery_consecutive", config.stability.recovery_consecutive, int),
        (
            "stability.joint_tolerance_evaluations",
            config.stability.joint_tolerance_evaluations,
            int,
        ),
    ]
    for path, value, expected in typed:
        _require_exact_type(path, value, expected)

    if config.schema_version != MEASUREMENT_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version: expected {MEASUREMENT_SCHEMA_VERSION}, got {config.schema_version!r}"
        )
    if config.profile != "m1c-extension":
        raise ValueError(f"profile: expected 'm1c-extension', got {config.profile!r}")
    if not config.source.canonical_run_id.strip():
        raise ValueError(
            "source.canonical_run_id: expected non-empty string, "
            f"got {config.source.canonical_run_id!r}"
        )
    for name in ("canonical_checkpoint_sha256", "scientific_config_hash", "split_hash"):
        value = getattr(config.source, name)
        if _HASH_PATTERN.fullmatch(value) is None:
            raise ValueError(f"source.{name}: expected 64 lowercase hex characters, got {value!r}")
    for name in ("canonical_checkpoint_step", "eval_interval", "checkpoint_interval"):
        value = getattr(config.source, name)
        if value < 1:
            raise ValueError(f"source.{name}: expected >= 1, got {value!r}")
    if config.source.canonical_checkpoint_step % config.source.eval_interval:
        raise ValueError(
            "source.canonical_checkpoint_step: expected a multiple of source.eval_interval, "
            f"got {config.source.canonical_checkpoint_step!r}"
        )
    if config.source.checkpoint_interval % config.source.eval_interval:
        raise ValueError(
            "source.checkpoint_interval: expected a multiple of source.eval_interval, "
            f"got {config.source.checkpoint_interval!r}"
        )

    event_steps = asdict(config.frozen_events)
    for name, value in event_steps.items():
        if value < 0 or value > config.source.canonical_checkpoint_step:
            raise ValueError(
                f"frozen_events.{name}: expected between 0 and "
                f"{config.source.canonical_checkpoint_step}, got {value!r}"
            )
        if value % config.source.eval_interval:
            raise ValueError(
                f"frozen_events.{name}: expected a multiple of source.eval_interval="
                f"{config.source.eval_interval}, got {value!r}"
            )
    if not (
        config.frozen_events.t_fit < config.frozen_events.t_grok50 < config.frozen_events.t_grok99
    ):
        raise ValueError(
            f"frozen_events: expected t_fit < t_grok50 < t_grok99, got {event_steps!r}"
        )
    for name in ("t_fit", "t_grok50", "t_grok99"):
        event_step = getattr(config.frozen_events, name)
        detected_at = getattr(config.frozen_events, f"{name}_detected_at")
        if detected_at < event_step:
            raise ValueError(
                f"frozen_events.{name}_detected_at: expected >= {name}={event_step}, "
                f"got {detected_at!r}"
            )

    stability = config.stability
    for name in (
        "stable_accuracy",
        "collapse_accuracy",
        "train_recovery_accuracy",
        "test_recovery_accuracy",
    ):
        value = float(getattr(stability, name))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"stability.{name}: expected finite value in [0, 1], got {value!r}")
    if stability.stable_window_intervals < 1:
        raise ValueError(
            "stability.stable_window_intervals: expected >= 1, "
            f"got {stability.stable_window_intervals!r}"
        )
    if stability.recovery_consecutive < 1:
        raise ValueError(
            f"stability.recovery_consecutive: expected >= 1, got {stability.recovery_consecutive!r}"
        )
    if stability.joint_tolerance_evaluations < 0:
        raise ValueError(
            "stability.joint_tolerance_evaluations: expected >= 0, "
            f"got {stability.joint_tolerance_evaluations!r}"
        )
    if stability.collapse_accuracy >= min(
        stability.train_recovery_accuracy,
        stability.test_recovery_accuracy,
    ):
        raise ValueError(
            "stability.collapse_accuracy: expected below both recovery thresholds, "
            f"got {stability.collapse_accuracy!r}"
        )


def _validate_records(records: list[dict[str, Any]], eval_interval: int) -> None:
    if type(eval_interval) is not int or eval_interval < 1:
        raise ValueError(f"eval_interval: expected positive integer, got {eval_interval!r}")
    previous: int | None = None
    for index, record in enumerate(records):
        if type(record) is not dict:
            raise ValueError(f"records[{index}]: expected mapping, got {record!r}")
        step = record.get("step")
        if type(step) is not int:
            raise ValueError(f"records[{index}].step: expected integer, got {step!r}")
        if step < 0:
            raise ValueError(f"records[{index}].step: expected >= 0, got {step!r}")
        if previous is not None:
            if step <= previous:
                raise ValueError("record steps must be strictly increasing")
            if step - previous != eval_interval:
                raise ValueError(
                    "record steps must form a complete evaluation grid: "
                    f"{previous!r} -> {step!r}, expected interval {eval_interval!r}"
                )
        for field in ("train_accuracy", "test_accuracy"):
            value = record.get(field)
            if type(value) not in {int, float}:
                raise ValueError(f"records[{index}].{field}: expected finite number, got {value!r}")
            numeric = float(value)
            if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                raise ValueError(
                    f"records[{index}].{field}: expected finite value in [0, 1], got {value!r}"
                )
        previous = step


def _window_dict(records: list[dict[str, Any]], start: int, end: int) -> dict[str, int]:
    return {
        "start_step": int(records[start]["step"]),
        "end_step": int(records[end]["step"]),
        "evaluation_count": end - start + 1,
        "step_span": int(records[end]["step"]) - int(records[start]["step"]),
    }


def detect_stable_window(
    records: list[dict[str, Any]],
    *,
    threshold: float,
    window_intervals: int,
    eval_interval: int,
    start_step: int,
) -> dict[str, Any]:
    """Find the first inclusive stable window spanning ``window_intervals`` gaps."""
    _validate_records(records, eval_interval)
    if type(threshold) not in {int, float} or not math.isfinite(float(threshold)):
        raise ValueError(f"threshold: expected finite number, got {threshold!r}")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"threshold: expected value in [0, 1], got {threshold!r}")
    if type(window_intervals) is not int or window_intervals < 1:
        raise ValueError(f"window_intervals: expected positive integer, got {window_intervals!r}")
    if type(start_step) is not int or start_step < 0:
        raise ValueError(f"start_step: expected nonnegative integer, got {start_step!r}")

    required_records = window_intervals + 1
    for start in range(len(records) - required_records + 1):
        window = records[start : start + required_records]
        if int(window[0]["step"]) < start_step:
            continue
        if all(float(record["test_accuracy"]) >= threshold for record in window):
            return {
                "status": "reached",
                "event_step": int(window[0]["step"]),
                "threshold": float(threshold),
                "required_intervals": window_intervals,
                "required_evaluations": required_records,
                "required_step_span": window_intervals * eval_interval,
                "detected_at_evaluation_step": int(window[-1]["step"]),
            }
    return {
        "status": "not_reached",
        "event_step": None,
        "threshold": float(threshold),
        "required_intervals": window_intervals,
        "required_evaluations": required_records,
        "required_step_span": window_intervals * eval_interval,
        "detected_at_evaluation_step": None,
    }


def _high_window_before(
    records: list[dict[str, Any]],
    onset_index: int,
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, int] | None:
    end = onset_index - 1
    if end < 0 or not predicate(records[end]):
        return None
    start = end
    while start > 0 and predicate(records[start - 1]):
        start -= 1
    return _window_dict(records, start, end)


def _high_window_after(
    records: list[dict[str, Any]],
    start_index: int | None,
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, int] | None:
    if start_index is None or start_index >= len(records) or not predicate(records[start_index]):
        return None
    end = start_index
    while end + 1 < len(records) and predicate(records[end + 1]):
        end += 1
    return _window_dict(records, start_index, end)


def _range_metrics(
    records: list[dict[str, Any]],
    start: int,
    end: int,
    collapse_accuracy: float,
) -> dict[str, Any]:
    train_index = min(range(start, end + 1), key=lambda index: records[index]["train_accuracy"])
    test_index = min(range(start, end + 1), key=lambda index: records[index]["test_accuracy"])
    trough_index = min(
        range(start, end + 1),
        key=lambda index: (
            min(records[index]["train_accuracy"], records[index]["test_accuracy"]),
            index,
        ),
    )
    train_minimum = float(records[train_index]["train_accuracy"])
    test_minimum = float(records[test_index]["test_accuracy"])
    return {
        "trough_step": int(records[trough_index]["step"]),
        "trough_train_accuracy": float(records[trough_index]["train_accuracy"]),
        "trough_test_accuracy": float(records[trough_index]["test_accuracy"]),
        "train_trough_step": int(records[train_index]["step"]),
        "test_trough_step": int(records[test_index]["step"]),
        "train_depth": max(0.0, collapse_accuracy - train_minimum),
        "test_depth": max(0.0, collapse_accuracy - test_minimum),
    }


def _primitive_episodes(
    records: list[dict[str, Any]],
    *,
    split: str,
    first_event_step: int,
    collapse_accuracy: float,
    recovery_accuracy: float,
    recovery_consecutive: int,
    eval_interval: int,
) -> list[dict[str, Any]]:
    accuracy_field = f"{split}_accuracy"
    episodes: list[dict[str, Any]] = []
    index = 0
    while index < len(records):
        record = records[index]
        if (
            int(record["step"]) <= first_event_step
            or float(record[accuracy_field]) >= collapse_accuracy
        ):
            index += 1
            continue

        onset = index
        recovery_start: int | None = None
        recovery_confirmed: int | None = None
        consecutive = 0
        cursor = onset
        while cursor < len(records):
            if float(records[cursor][accuracy_field]) >= recovery_accuracy:
                consecutive += 1
                if consecutive == 1:
                    recovery_start = cursor
                if consecutive == recovery_consecutive:
                    recovery_confirmed = cursor
                    break
            else:
                consecutive = 0
                recovery_start = None
            cursor += 1

        end = recovery_confirmed if recovery_confirmed is not None else len(records) - 1

        def predicate(item: dict[str, Any]) -> bool:
            return float(item[accuracy_field]) >= recovery_accuracy

        recovery_step = (
            int(records[recovery_start]["step"]) if recovery_confirmed is not None else None
        )
        confirmed_step = (
            int(records[recovery_confirmed]["step"]) if recovery_confirmed is not None else None
        )
        episode_number = len(episodes) + 1
        episode = {
            "episode_id": f"{split}_{episode_number:03d}",
            "episode_type": split,
            "onset_step": int(records[onset]["step"]),
            "onset_evaluation_index": int(records[onset]["step"]) // eval_interval,
            **_range_metrics(records, onset, end, collapse_accuracy),
            "train_recovery_step": recovery_step if split == "train" else None,
            "test_recovery_step": recovery_step if split == "test" else None,
            "train_recovery_confirmed_step": confirmed_step if split == "train" else None,
            "test_recovery_confirmed_step": confirmed_step if split == "test" else None,
            "joint_recovery_step": None,
            "joint_recovery_confirmed_step": None,
            "recovery_duration_steps": (
                recovery_step - int(records[onset]["step"]) if recovery_step is not None else None
            ),
            "status": "recovered" if recovery_confirmed is not None else "not_recovered",
            "preceding_high_performance_window": _high_window_before(records, onset, predicate),
            "following_high_performance_window": _high_window_after(
                records,
                recovery_start if recovery_confirmed is not None else None,
                predicate,
            ),
        }
        episodes.append(episode)
        index = recovery_confirmed + 1 if recovery_confirmed is not None else len(records)
    return episodes


def _record_index_by_step(records: list[dict[str, Any]]) -> dict[int, int]:
    return {int(record["step"]): index for index, record in enumerate(records)}


def _pair_joint_episodes(
    records: list[dict[str, Any]],
    train_episodes: list[dict[str, Any]],
    test_episodes: list[dict[str, Any]],
    *,
    tolerance_evaluations: int,
    eval_interval: int,
    collapse_accuracy: float,
    train_recovery_accuracy: float,
    test_recovery_accuracy: float,
) -> list[dict[str, Any]]:
    candidates: list[tuple[int, int, str, str, dict[str, Any], dict[str, Any]]] = []
    for train in train_episodes:
        for test in test_episodes:
            distance = abs(
                int(train["onset_evaluation_index"]) - int(test["onset_evaluation_index"])
            )
            if distance <= tolerance_evaluations:
                candidates.append(
                    (
                        distance,
                        min(int(train["onset_step"]), int(test["onset_step"])),
                        str(train["episode_id"]),
                        str(test["episode_id"]),
                        train,
                        test,
                    )
                )
    candidates.sort(key=lambda item: item[:4])
    used_train: set[str] = set()
    used_test: set[str] = set()
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for _, _, train_id, test_id, train, test in candidates:
        if train_id not in used_train and test_id not in used_test:
            used_train.add(train_id)
            used_test.add(test_id)
            pairs.append((train, test))
    pairs.sort(
        key=lambda pair: (
            min(int(pair[0]["onset_step"]), int(pair[1]["onset_step"])),
            pair[0]["episode_id"],
            pair[1]["episode_id"],
        )
    )

    indices = _record_index_by_step(records)
    joint_episodes: list[dict[str, Any]] = []
    for number, (train, test) in enumerate(pairs, start=1):
        onset_step = min(int(train["onset_step"]), int(test["onset_step"]))
        onset = indices[onset_step]
        train_confirmed = train["train_recovery_confirmed_step"]
        test_confirmed = test["test_recovery_confirmed_step"]
        recovered = train_confirmed is not None and test_confirmed is not None
        confirmed_step = max(train_confirmed, test_confirmed) if recovered else None
        end = indices[int(confirmed_step)] if confirmed_step is not None else len(records) - 1
        train_recovery = train["train_recovery_step"]
        test_recovery = test["test_recovery_step"]
        recovery_step = max(train_recovery, test_recovery) if recovered else None

        def predicate(item: dict[str, Any]) -> bool:
            return (
                float(item["train_accuracy"]) >= train_recovery_accuracy
                and float(item["test_accuracy"]) >= test_recovery_accuracy
            )

        following_index = indices[int(recovery_step)] if recovery_step is not None else None
        joint_episodes.append(
            {
                "episode_id": f"joint_{number:03d}",
                "episode_type": "joint",
                "train_episode_id": train["episode_id"],
                "test_episode_id": test["episode_id"],
                "onset_step": onset_step,
                "onset_evaluation_index": onset_step // eval_interval,
                **_range_metrics(records, onset, end, collapse_accuracy),
                "train_recovery_step": train_recovery,
                "test_recovery_step": test_recovery,
                "train_recovery_confirmed_step": train_confirmed,
                "test_recovery_confirmed_step": test_confirmed,
                "joint_recovery_step": recovery_step,
                "joint_recovery_confirmed_step": confirmed_step,
                "recovery_duration_steps": (
                    int(recovery_step) - onset_step if recovery_step is not None else None
                ),
                "status": "recovered" if recovered else "not_recovered",
                "preceding_high_performance_window": _high_window_before(records, onset, predicate),
                "following_high_performance_window": _high_window_after(
                    records, following_index, predicate
                ),
            }
        )
    return joint_episodes


def detect_collapse_episodes(
    records: list[dict[str, Any]],
    *,
    t_fit_detected_at: int,
    t_grok99_detected_at: int,
    eval_interval: int,
    config: StabilityConfig,
) -> dict[str, Any]:
    """Detect independent train/test episodes and deterministic joint composites."""
    _validate_records(records, eval_interval)
    for name, value in (
        ("t_fit_detected_at", t_fit_detected_at),
        ("t_grok99_detected_at", t_grok99_detected_at),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name}: expected nonnegative integer, got {value!r}")
    train = _primitive_episodes(
        records,
        split="train",
        first_event_step=t_fit_detected_at,
        collapse_accuracy=float(config.collapse_accuracy),
        recovery_accuracy=float(config.train_recovery_accuracy),
        recovery_consecutive=config.recovery_consecutive,
        eval_interval=eval_interval,
    )
    test = _primitive_episodes(
        records,
        split="test",
        first_event_step=t_grok99_detected_at,
        collapse_accuracy=float(config.collapse_accuracy),
        recovery_accuracy=float(config.test_recovery_accuracy),
        recovery_consecutive=config.recovery_consecutive,
        eval_interval=eval_interval,
    )
    joint = _pair_joint_episodes(
        records,
        train,
        test,
        tolerance_evaluations=config.joint_tolerance_evaluations,
        eval_interval=eval_interval,
        collapse_accuracy=float(config.collapse_accuracy),
        train_recovery_accuracy=float(config.train_recovery_accuracy),
        test_recovery_accuracy=float(config.test_recovery_accuracy),
    )
    unified = [*train, *test, *joint]
    type_order = {"train": 0, "test": 1, "joint": 2}
    unified.sort(
        key=lambda episode: (
            int(episode["onset_step"]),
            type_order[str(episode["episode_type"])],
            str(episode["episode_id"]),
        )
    )
    return {
        "schema_version": STABILITY_SCHEMA_VERSION,
        "eval_interval": eval_interval,
        "train_episodes": train,
        "test_episodes": test,
        "joint_episodes": joint,
        "episodes": unified,
    }


def _longest_test_window(
    records: list[dict[str, Any]], threshold: float, eval_interval: int
) -> tuple[int, int]:
    best = 0
    current = 0
    for record in records:
        if float(record["test_accuracy"]) >= threshold:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best, max(0, best - 1) * eval_interval


def _final_state(
    records: list[dict[str, Any]],
    episode_artifact: dict[str, Any],
    *,
    config: StabilityConfig,
    terminal_stable: bool,
) -> str:
    active = [
        episode
        for split in ("train_episodes", "test_episodes")
        for episode in episode_artifact[split]
        if episode["status"] == "not_recovered"
    ]
    final = records[-1]
    if any(
        float(final[f"{episode['episode_type']}_accuracy"]) < config.collapse_accuracy
        for episode in active
    ):
        state = "collapsed"
    elif active:
        state = "recovering"
    elif terminal_stable and float(final["train_accuracy"]) >= config.train_recovery_accuracy:
        state = "high_performance"
    else:
        state = "unstable_but_above_threshold"
    if state not in _FINAL_STATES:
        raise AssertionError(f"unexpected stability final state: {state}")
    return state


def summarize_stability(
    records: list[dict[str, Any]],
    *,
    run_id: str,
    parent_run_id: str | None,
    eval_interval: int,
    frozen_events: FrozenEventsConfig,
    config: StabilityConfig,
    source_scalars_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build complete, deterministic M1-C stability and episode artifacts."""
    _validate_records(records, eval_interval)
    if not records:
        raise ValueError("records: at least one scalar record is required")
    if type(run_id) is not str or not run_id.strip():
        raise ValueError(f"run_id: expected non-empty string, got {run_id!r}")
    if parent_run_id is not None and (type(parent_run_id) is not str or not parent_run_id.strip()):
        raise ValueError(f"parent_run_id: expected non-empty string or null, got {parent_run_id!r}")
    if _HASH_PATTERN.fullmatch(source_scalars_sha256) is None:
        raise ValueError(
            "source_scalars_sha256: expected 64 lowercase hex characters, "
            f"got {source_scalars_sha256!r}"
        )

    stable = detect_stable_window(
        records,
        threshold=float(config.stable_accuracy),
        window_intervals=config.stable_window_intervals,
        eval_interval=eval_interval,
        start_step=frozen_events.t_grok99,
    )
    episodes = detect_collapse_episodes(
        records,
        t_fit_detected_at=frozen_events.t_fit_detected_at,
        t_grok99_detected_at=frozen_events.t_grok99_detected_at,
        eval_interval=eval_interval,
        config=config,
    )
    definitions = {
        "stable_accuracy": float(config.stable_accuracy),
        "stable_window_intervals": config.stable_window_intervals,
        "stable_window_evaluations": config.stable_window_intervals + 1,
        "stable_window_steps": config.stable_window_intervals * eval_interval,
        "collapse_accuracy": float(config.collapse_accuracy),
        "train_recovery_accuracy": float(config.train_recovery_accuracy),
        "test_recovery_accuracy": float(config.test_recovery_accuracy),
        "recovery_consecutive": config.recovery_consecutive,
        "joint_tolerance_evaluations": config.joint_tolerance_evaluations,
        "joint_tolerance_steps": config.joint_tolerance_evaluations * eval_interval,
        "collapse_gate": "first_event_detected_at_evaluation_step",
        "frozen_events": asdict(frozen_events),
    }
    episodes["run_id"] = run_id
    episodes["parent_run_id"] = parent_run_id
    episodes["source_scalars_sha256"] = source_scalars_sha256
    episodes["definitions"] = definitions

    train = episodes["train_episodes"]
    test = episodes["test_episodes"]
    joint = episodes["joint_episodes"]
    all_onsets = [int(episode["onset_step"]) for episode in [*train, *test, *joint]]
    longest_evaluations, longest_steps = _longest_test_window(
        records, float(config.stable_accuracy), eval_interval
    )
    required = config.stable_window_intervals + 1
    terminal_stable = len(records) >= required and all(
        float(record["test_accuracy"]) >= config.stable_accuracy for record in records[-required:]
    )
    summary = {
        "schema_version": STABILITY_SCHEMA_VERSION,
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "eval_interval": eval_interval,
        "final_step": int(records[-1]["step"]),
        "t_fit": frozen_events.t_fit,
        "t_grok50": frozen_events.t_grok50,
        "t_grok99": frozen_events.t_grok99,
        "t_stable99": stable,
        "definitions": definitions,
        "collapse_count_train": len(train),
        "collapse_count_test": len(test),
        "collapse_count_joint": len(joint),
        "last_collapse_step": max(all_onsets) if all_onsets else None,
        "longest_test_window_above_99_evaluations": longest_evaluations,
        "longest_test_window_above_99_steps": longest_steps,
        "fraction_of_evaluations_above_99": sum(
            float(record["test_accuracy"]) >= config.stable_accuracy for record in records
        )
        / len(records),
        "final_train_accuracy": float(records[-1]["train_accuracy"]),
        "final_test_accuracy": float(records[-1]["test_accuracy"]),
        "final_state": _final_state(
            records, episodes, config=config, terminal_stable=terminal_stable
        ),
        "source_scalars_sha256": source_scalars_sha256,
    }
    return summary, episodes
