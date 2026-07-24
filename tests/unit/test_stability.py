from __future__ import annotations

from copy import deepcopy

import pytest

from transgrokking.metrics.stability import (
    FrozenEventsConfig,
    StabilityConfig,
    detect_collapse_episodes,
    detect_stable_window,
    dump_measurement_config,
    load_measurement_config,
    measurement_config_from_dict,
    summarize_stability,
)

EVAL_INTERVAL = 50
SCALARS_HASH = "a" * 64


def _record(step: int, train: float, test: float) -> dict[str, int | float]:
    return {"step": step, "train_accuracy": train, "test_accuracy": test}


def _records(
    train: list[float],
    test: list[float],
    *,
    first_step: int = EVAL_INTERVAL,
) -> list[dict[str, int | float]]:
    return [
        _record(first_step + index * EVAL_INTERVAL, train_value, test_value)
        for index, (train_value, test_value) in enumerate(zip(train, test, strict=True))
    ]


def _stability_config(**updates: float | int) -> StabilityConfig:
    values: dict[str, float | int] = {
        "stable_accuracy": 0.99,
        "stable_window_intervals": 100,
        "collapse_accuracy": 0.9,
        "train_recovery_accuracy": 0.999,
        "test_recovery_accuracy": 0.99,
        "recovery_consecutive": 3,
        "joint_tolerance_evaluations": 1,
    }
    values.update(updates)
    return StabilityConfig(**values)


def _events() -> FrozenEventsConfig:
    return FrozenEventsConfig(
        t_fit=50,
        t_fit_detected_at=50,
        t_grok50=50,
        t_grok50_detected_at=50,
        t_grok99=50,
        t_grok99_detected_at=50,
    )


def test_stable_window_is_100_intervals_and_101_records() -> None:
    only_100 = _records([1.0] * 100, [0.99] * 100)
    not_reached = detect_stable_window(
        only_100,
        threshold=0.99,
        window_intervals=100,
        eval_interval=EVAL_INTERVAL,
        start_step=50,
    )
    assert not_reached["status"] == "not_reached"
    assert not_reached["event_step"] is None

    reached = detect_stable_window(
        [*only_100, _record(5050, 1.0, 0.99)],
        threshold=0.99,
        window_intervals=100,
        eval_interval=EVAL_INTERVAL,
        start_step=50,
    )
    assert reached == {
        "status": "reached",
        "event_step": 50,
        "threshold": 0.99,
        "required_intervals": 100,
        "required_evaluations": 101,
        "required_step_span": 5000,
        "detected_at_evaluation_step": 5050,
    }


@pytest.mark.parametrize(
    "records,match",
    [
        (
            [_record(100, 1.0, 1.0), _record(50, 1.0, 1.0)],
            "strictly increasing",
        ),
        (
            [_record(50, 1.0, 1.0), _record(150, 1.0, 1.0)],
            "complete evaluation grid",
        ),
        (
            [_record(50, 1.0, float("nan"))],
            "finite value",
        ),
    ],
)
def test_stability_rejects_invalid_scalar_timelines(
    records: list[dict[str, int | float]], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        detect_stable_window(
            records,
            threshold=0.99,
            window_intervals=1,
            eval_interval=EVAL_INTERVAL,
            start_step=0,
        )


def test_primitive_recovery_and_joint_composite_are_explicit() -> None:
    records = _records(
        [1.0, 0.8, 0.95, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.95, 0.8, 0.95, 1.0, 1.0, 1.0],
    )
    artifact = detect_collapse_episodes(
        records,
        t_fit_detected_at=50,
        t_grok99_detected_at=50,
        eval_interval=EVAL_INTERVAL,
        config=_stability_config(),
    )
    train = artifact["train_episodes"][0]
    test = artifact["test_episodes"][0]
    joint = artifact["joint_episodes"][0]

    assert train["episode_id"] == "train_001"
    assert train["onset_step"] == 100
    assert train["onset_evaluation_index"] == 2
    assert train["train_recovery_step"] == 200
    assert train["train_recovery_confirmed_step"] == 300
    assert train["test_recovery_step"] is None
    assert train["recovery_duration_steps"] == 100
    assert train["status"] == "recovered"
    assert train["train_trough_step"] == 100
    assert train["test_trough_step"] == 150
    assert train["train_depth"] == pytest.approx(0.1)

    assert test["onset_step"] == 150
    assert test["test_recovery_step"] == 250
    assert test["test_recovery_confirmed_step"] == 350
    assert test["status"] == "recovered"

    assert joint["episode_type"] == "joint"
    assert joint["train_episode_id"] == "train_001"
    assert joint["test_episode_id"] == "test_001"
    assert joint["onset_step"] == 100
    assert joint["joint_recovery_step"] == 250
    assert joint["joint_recovery_confirmed_step"] == 350
    assert joint["recovery_duration_steps"] == 150
    assert joint["status"] == "recovered"
    assert joint["preceding_high_performance_window"]["start_step"] == 50
    assert joint["following_high_performance_window"]["start_step"] == 250


def test_repeated_collapse_after_recovery_creates_new_episode() -> None:
    records = _records(
        [1.0, 0.8, 1.0, 1.0, 1.0, 0.85, 1.0, 1.0, 1.0],
        [1.0] * 9,
    )
    artifact = detect_collapse_episodes(
        records,
        t_fit_detected_at=50,
        t_grok99_detected_at=50,
        eval_interval=EVAL_INTERVAL,
        config=_stability_config(),
    )
    assert [episode["onset_step"] for episode in artifact["train_episodes"]] == [100, 300]
    assert artifact["test_episodes"] == []
    assert artifact["joint_episodes"] == []


def test_unrecovered_episode_keeps_null_recovery_and_is_not_split() -> None:
    records = _records(
        [1.0, 0.8, 1.0, 0.8, 0.95],
        [1.0, 1.0, 1.0, 1.0, 1.0],
    )
    artifact = detect_collapse_episodes(
        records,
        t_fit_detected_at=50,
        t_grok99_detected_at=50,
        eval_interval=EVAL_INTERVAL,
        config=_stability_config(),
    )
    assert len(artifact["train_episodes"]) == 1
    episode = artifact["train_episodes"][0]
    assert episode["status"] == "not_recovered"
    assert episode["train_recovery_step"] is None
    assert episode["train_recovery_confirmed_step"] is None
    assert episode["recovery_duration_steps"] is None


def test_collapse_detection_starts_at_first_event_confirmation() -> None:
    records = _records(
        [1.0, 0.8, 1.0, 1.0, 0.8, 0.8],
        [1.0] * 6,
    )
    artifact = detect_collapse_episodes(
        records,
        t_fit_detected_at=250,
        t_grok99_detected_at=50,
        eval_interval=EVAL_INTERVAL,
        config=_stability_config(),
    )
    assert [episode["onset_step"] for episode in artifact["train_episodes"]] == [300]


@pytest.mark.parametrize(
    ("train", "test", "expected"),
    [
        ([1.0, 0.8], [1.0, 1.0], "collapsed"),
        ([1.0, 0.8, 0.95], [1.0, 1.0, 1.0], "recovering"),
        ([1.0] * 101, [0.99] * 101, "high_performance"),
        ([1.0] * 100, [0.99] * 100, "unstable_but_above_threshold"),
    ],
)
def test_summary_final_state_priority(train: list[float], test: list[float], expected: str) -> None:
    records = _records(train, test)
    summary, episodes = summarize_stability(
        records,
        run_id="child",
        parent_run_id="parent",
        eval_interval=EVAL_INTERVAL,
        frozen_events=_events(),
        config=_stability_config(),
        source_scalars_sha256=SCALARS_HASH,
    )
    assert summary["final_state"] == expected
    assert episodes["run_id"] == "child"
    assert summary["final_step"] == records[-1]["step"]
    assert summary["source_scalars_sha256"] == SCALARS_HASH


def test_summary_longest_window_fraction_and_episode_counts() -> None:
    records = _records(
        [1.0, 0.8, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.8, 1.0, 1.0, 1.0, 0.8],
    )
    summary, episodes = summarize_stability(
        records,
        run_id="child",
        parent_run_id="parent",
        eval_interval=EVAL_INTERVAL,
        frozen_events=_events(),
        config=_stability_config(stable_window_intervals=2),
        source_scalars_sha256=SCALARS_HASH,
    )
    assert summary["collapse_count_train"] == 1
    assert summary["collapse_count_test"] == 2
    assert summary["collapse_count_joint"] == 1
    assert summary["last_collapse_step"] == 300
    assert summary["longest_test_window_above_99_evaluations"] == 3
    assert summary["longest_test_window_above_99_steps"] == 100
    assert summary["fraction_of_evaluations_above_99"] == pytest.approx(4 / 6)
    assert summary["t_stable99"]["event_step"] == 150
    assert len(episodes["joint_episodes"]) == 1
    assert episodes["episodes"] == sorted(
        episodes["episodes"],
        key=lambda episode: (
            episode["onset_step"],
            {"train": 0, "test": 1, "joint": 2}[episode["episode_type"]],
            episode["episode_id"],
        ),
    )
    assert episodes["source_scalars_sha256"] == SCALARS_HASH
    assert episodes["definitions"] == summary["definitions"]


def test_measurement_sidecar_is_strict_hashable_and_round_trips(tmp_path) -> None:
    config = load_measurement_config("configs/analysis/m1c_stability.yaml")
    assert config.profile == "m1c-extension"
    assert config.source.canonical_checkpoint_step == 20000
    assert config.frozen_events.t_grok99 == 7000
    assert config.frozen_events.t_grok99_detected_at == 7100
    assert config.stability.stable_window_intervals == 100
    assert len(config.measurement_hash()) == 64

    path = tmp_path / "measurement.resolved.yaml"
    dump_measurement_config(config, path)
    restored = load_measurement_config(path)
    assert restored == config
    assert restored.measurement_hash() == config.measurement_hash()


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda raw: raw.update({"unknown": True}), "unknown"),
        (
            lambda raw: raw["stability"].update({"stable_window_intervals": True}),
            "stable_window_intervals",
        ),
        (
            lambda raw: raw["source"].update({"scientific_config_hash": "bad"}),
            "64 lowercase hex",
        ),
        (
            lambda raw: raw["stability"].update({"collapse_accuracy": 1.0}),
            "below both recovery",
        ),
    ],
)
def test_measurement_sidecar_rejects_unknown_and_invalid_values(mutate, match: str) -> None:
    raw = load_measurement_config("configs/analysis/m1c_stability.yaml").to_dict()
    candidate = deepcopy(raw)
    mutate(candidate)
    with pytest.raises(ValueError, match=match):
        measurement_config_from_dict(candidate)
