"""Fast mathematical invariant tests for the M2 function-space layer."""

from __future__ import annotations

import json
import math

import pytest
import torch

from transgrokking.metrics.function_space import (
    MathTolerances,
    apply_group_action,
    center_logits,
    detect_function_events,
    explicit_reynolds_projection,
    function_space_metrics,
    invariant_errors,
    offset_profile,
    project_from_offset_profile,
    reynolds_decomposition,
    reynolds_projection,
)


def _split_indices(modulus: int) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.arange(modulus * modulus, dtype=torch.long)
    return indices[::2], indices[1::2]


def _function_metrics(logits: torch.Tensor):
    train, test = _split_indices(logits.shape[0])
    return function_space_metrics(logits, train, test, parameter_norm_total=2.0)


def test_centering_removes_class_mean_and_common_shift() -> None:
    generator = torch.Generator().manual_seed(4)
    logits = torch.randn((5, 5, 5), generator=generator)
    common_shift = torch.randn((5, 5, 1), generator=generator)

    centered = center_logits(logits)

    assert centered.dtype == torch.float64
    assert centered.device.type == "cpu"
    assert torch.allclose(centered.sum(dim=2), torch.zeros((5, 5), dtype=torch.float64))
    torch.testing.assert_close(center_logits(logits + common_shift), centered)


@pytest.mark.parametrize("modulus", [2, 3, 5, 8])
def test_offset_projection_matches_explicit_orbit_average(modulus: int) -> None:
    logits = torch.randn((modulus, modulus, modulus), generator=torch.Generator().manual_seed(8))
    centered = center_logits(logits)

    vectorized = reynolds_projection(centered)
    brute_force = explicit_reynolds_projection(centered)

    torch.testing.assert_close(vectorized, brute_force, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(
        vectorized,
        apply_group_action(vectorized, 1, 0),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    torch.testing.assert_close(
        vectorized,
        apply_group_action(vectorized, 0, 1),
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_reynolds_invariants_and_profile_constraints() -> None:
    logits = torch.randn((5, 5, 5), generator=torch.Generator().manual_seed(17))
    decomposition = reynolds_decomposition(logits)
    errors = invariant_errors(decomposition)

    torch.testing.assert_close(
        decomposition.centered,
        decomposition.projected + decomposition.residual,
    )
    torch.testing.assert_close(
        reynolds_projection(decomposition.projected), decomposition.projected
    )
    torch.testing.assert_close(
        reynolds_projection(decomposition.residual), torch.zeros_like(decomposition.residual)
    )
    assert abs(float(decomposition.offset_profile.sum())) < 1.0e-12
    assert max(errors.values()) < 1.0e-12


def test_single_offset_pattern_has_no_residual_and_positive_gamma() -> None:
    profile = torch.tensor([4.0, -1.0, -1.0, -1.0, -1.0], dtype=torch.float64)
    logits = project_from_offset_profile(profile)
    result = _function_metrics(logits)

    assert result.metrics["Gamma"] == pytest.approx(5.0)
    assert result.metrics["I"] == pytest.approx(0.0)
    assert result.metrics["D_eq"] == pytest.approx(0.0)
    assert result.metrics["full_accuracy"] == 1.0
    assert result.metrics["full_projected_accuracy"] == 1.0
    assert result.metrics["raw_implication_status"] == "verified"
    assert result.metrics["projected_implication_status"] == "verified"


def test_interference_excludes_correct_class_and_matches_brute_force() -> None:
    modulus = 3
    logits = torch.randn((modulus, modulus, modulus), generator=torch.Generator().manual_seed(21))
    result = _function_metrics(logits)
    residual = result.residual_logits
    expected = -math.inf
    for a in range(modulus):
        for b in range(modulus):
            label = (a + b) % modulus
            for candidate in range(modulus):
                if candidate != label:
                    expected = max(
                        expected,
                        float(residual[a, b, candidate] - residual[a, b, label]),
                    )

    assert result.metrics["I"] == pytest.approx(expected)


def test_zero_tensor_uses_null_ratios_and_ambiguous_implications() -> None:
    result = _function_metrics(torch.zeros((2, 2, 2)))

    assert result.metrics["D_eq"] is None
    assert result.metrics["Gamma_over_logit_rms"] is None
    assert result.metrics["I_over_logit_rms"] is None
    assert result.metrics["raw_implication_status"] == "numerically_ambiguous"
    assert result.metrics["projected_implication_status"] == "numerically_ambiguous"
    json.dumps(result.metrics, allow_nan=False)


def test_function_metrics_include_full_behavior_entropy_and_split_invariants() -> None:
    logits = torch.randn((5, 5, 5), generator=torch.Generator().manual_seed(32))
    result = _function_metrics(logits)
    metrics = result.metrics

    for field in (
        "full_cross_entropy",
        "full_accuracy",
        "full_margin_mean",
        "full_margin_min",
        "train_entropy_mean",
        "test_entropy_mean",
        "full_entropy_mean",
        "train_entropy_normalized",
        "test_entropy_normalized",
        "full_entropy_normalized",
    ):
        assert field in metrics
    assert metrics["projected_split_ce_max_abs_diff"] < 1.0e-12
    assert metrics["projected_split_accuracy_max_abs_diff"] == 0.0
    assert metrics["projected_split_margin_max_abs_diff"] < 1.0e-12
    assert metrics["projected_split_invariants_passed"] is True
    assert 0.0 <= float(metrics["full_entropy_normalized"]) <= 1.0
    assert metrics["invariants_passed"] is True
    json.dumps(metrics, allow_nan=False)


def test_offset_profile_vectorized_matches_direct_definition() -> None:
    modulus = 5
    centered = center_logits(
        torch.randn((modulus, modulus, modulus), generator=torch.Generator().manual_seed(44))
    )
    expected = torch.empty(modulus, dtype=torch.float64)
    for offset in range(modulus):
        values = [
            centered[a, b, (a + b + offset) % modulus]
            for a in range(modulus)
            for b in range(modulus)
        ]
        expected[offset] = torch.stack(values).mean()

    torch.testing.assert_close(offset_profile(centered), expected)


def test_math_tolerances_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="centering_atol"):
        MathTolerances(centering_atol=-1.0)
    with pytest.raises(ValueError, match="invariance_rtol"):
        MathTolerances(invariance_rtol=float("nan"))


def test_function_metric_shape_and_index_errors_are_clear() -> None:
    with pytest.raises(ValueError, match=r"expected \[p, p, p\]"):
        center_logits(torch.zeros((2, 2)))
    logits = torch.zeros((3, 3, 3))
    with pytest.raises(ValueError, match="disjoint"):
        function_space_metrics(
            logits,
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([1, 2], dtype=torch.long),
        )
    train, test = _split_indices(3)
    with pytest.raises(ValueError, match="parameter_norm_total"):
        function_space_metrics(logits, train, test, parameter_norm_total=float("nan"))


def _record(step: int, gamma: float, interference: float, regular: bool = True):
    return {
        "step": step,
        "Gamma": gamma,
        "I": interference,
        "is_regular_grid": regular,
    }


def test_function_events_ignore_replay_and_report_crossings_and_intervals() -> None:
    records = [
        _record(0, -1.0, 0.0),
        _record(50, 100.0, -100.0, False),
        _record(100, 1.0, 2.0),
        _record(200, 3.0, 2.0),
        _record(300, -1.0, 0.0),
        _record(400, 2.0, 1.0),
    ]

    events = detect_function_events(records)

    assert events["regular_step_count"] == 5
    assert events["event_resolution_steps"] == 100
    assert events["t_alg"]["event_step"] == 100
    assert events["t_alg"]["previous_regular_step"] == 0
    assert events["t_alg"]["event_interval"] == {
        "lower_exclusive": 0,
        "upper_inclusive": 100,
    }
    assert events["t_dom"]["event_step"] == 200
    assert events["gamma_positive_run_count"] == 2
    assert events["gamma_false_to_true_crossing_count"] == 2
    assert events["gamma_true_to_false_exit_count"] == 1
    assert events["last_gamma_positive_exit_step"] == 300
    assert events["longest_gamma_positive_window_steps"] == 100
    assert events["fraction_gamma_positive"] == pytest.approx(3 / 5)
    assert events["dominance_positive_run_count"] == 2


def test_function_events_step_zero_true_is_run_but_not_crossing() -> None:
    events = detect_function_events(
        [_record(0, 2.0, 1.0), _record(100, -1.0, 0.0), _record(200, 1.0, 2.0)]
    )

    assert events["t_alg"]["event_step"] == 0
    assert events["t_alg"]["previous_regular_step"] is None
    assert events["gamma_positive_run_count"] == 2
    assert events["gamma_false_to_true_crossing_count"] == 1
    assert events["gamma_true_to_false_exit_count"] == 1


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([_record(0, 0.0, 0.0), _record(200, 0.0, 0.0)], "spaced by 100"),
        ([_record(0, 0.0, 0.0), _record(0, 0.0, 0.0)], "strictly increasing"),
        ([_record(0, float("nan"), 0.0)], "expected finite number"),
    ],
)
def test_function_events_reject_gaps_duplicates_and_nonfinite(records, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        detect_function_events(records)
