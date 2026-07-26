"""Pure M2 function-space metrics for modular-addition logits.

All public tensor functions accept logits with shape ``[p, p, p]`` in
``a-major, b-minor, candidate-last`` order.  Reductions are performed on CPU
in FP64 so the functions can be used both by the CUDA analysis runner and by
small CPU invariant tests.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, log
from typing import Any

import torch
from torch.nn import functional as F

from transgrokking.metrics.behavior import classification_margins, margin_summary

FUNCTION_METRICS_SCHEMA_VERSION = 1
FUNCTION_EVENTS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MathTolerances:
    """Scale-aware numerical tolerances used by M2 invariant checks."""

    centering_atol: float = 1.0e-10
    reconstruction_rtol: float = 1.0e-10
    orthogonality_normalized: float = 1.0e-10
    energy_identity_rtol: float = 1.0e-10
    invariance_rtol: float = 1.0e-10
    implication_margin_atol: float = 1.0e-10

    def __post_init__(self) -> None:
        for field_name, value in self.__dict__.items():
            if type(value) not in {float, int} or not isfinite(float(value)) or value < 0:
                raise ValueError(
                    f"math_tolerances.{field_name}: expected finite value >= 0, got {value!r}"
                )


@dataclass(frozen=True)
class ReynoldsDecomposition:
    """Class-centered logits and their equivariant/residual decomposition."""

    centered: torch.Tensor
    offset_profile: torch.Tensor
    projected: torch.Tensor
    residual: torch.Tensor


@dataclass(frozen=True)
class FunctionSpaceResult:
    """Flat JSON-safe metrics plus tensors needed for selected-state export."""

    metrics: dict[str, float | int | bool | str | None]
    offset_profile: torch.Tensor
    centered_logits: torch.Tensor
    projected_logits: torch.Tensor
    residual_logits: torch.Tensor


def _validate_function_tensor(tensor: torch.Tensor, field: str) -> int:
    if tensor.ndim != 3:
        raise ValueError(f"{field}: expected [p, p, p], got {tuple(tensor.shape)}")
    if tensor.shape[0] < 2 or tensor.shape[1:] != tensor.shape[:1] * 2:
        raise ValueError(
            f"{field}: expected equal dimensions with p >= 2, got {tuple(tensor.shape)}"
        )
    if not tensor.is_floating_point():
        raise ValueError(f"{field}: expected floating tensor, got {tensor.dtype}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{field}: all values must be finite")
    return int(tensor.shape[0])


def _fp64_cpu(tensor: torch.Tensor, field: str) -> torch.Tensor:
    _validate_function_tensor(tensor, field)
    return tensor.detach().to(device="cpu", dtype=torch.float64)


def center_logits(logits: torch.Tensor) -> torch.Tensor:
    """Subtract the candidate-class mean independently for every ``(a,b)``."""
    values = _fp64_cpu(logits, "logits")
    return values - values.mean(dim=2, keepdim=True)


def offset_profile(centered_logits: torch.Tensor) -> torch.Tensor:
    """Return ``g(d)=mean[a,b] centered[a,b,a+b+d]`` as a vector ``[p]``."""
    values = _fp64_cpu(centered_logits, "centered_logits")
    modulus = int(values.shape[0])
    indices = torch.arange(modulus, dtype=torch.long)
    base = (indices[:, None] + indices[None, :]).remainder(modulus)
    candidates = (base[:, :, None] + indices[None, None, :]).remainder(modulus)
    return values.gather(2, candidates).mean(dim=(0, 1))


def project_from_offset_profile(profile: torch.Tensor, modulus: int | None = None) -> torch.Tensor:
    """Construct ``z_parallel[a,b,c]=g(c-a-b mod p)`` from a profile ``[p]``."""
    if profile.ndim != 1 or profile.numel() < 2 or not profile.is_floating_point():
        raise ValueError(f"profile: expected floating vector [p] with p >= 2, got {profile.shape}")
    if not torch.isfinite(profile).all():
        raise ValueError("profile: all values must be finite")
    inferred = int(profile.numel())
    if modulus is not None and (type(modulus) is not int or modulus != inferred):
        raise ValueError(f"modulus: expected {inferred}, got {modulus!r}")
    values = profile.detach().to(device="cpu", dtype=torch.float64)
    indices = torch.arange(inferred, dtype=torch.long)
    offsets = (indices[None, None, :] - indices[:, None, None] - indices[None, :, None]).remainder(
        inferred
    )
    return values[offsets]


def reynolds_projection(centered_logits: torch.Tensor) -> torch.Tensor:
    """Apply the modular-addition Reynolds projection to centered logits."""
    values = _fp64_cpu(centered_logits, "centered_logits")
    return project_from_offset_profile(offset_profile(values))


def reynolds_decomposition(logits: torch.Tensor) -> ReynoldsDecomposition:
    """Center raw logits and decompose them into equivariant and residual parts."""
    centered = center_logits(logits)
    profile = offset_profile(centered)
    projected = project_from_offset_profile(profile)
    return ReynoldsDecomposition(
        centered=centered,
        offset_profile=profile,
        projected=projected,
        residual=centered - projected,
    )


def apply_group_action(tensor: torch.Tensor, u: int, v: int) -> torch.Tensor:
    """Return ``f(a+u,b+v,c+u+v)`` for the modular-addition group action."""
    modulus = _validate_function_tensor(tensor, "tensor")
    if type(u) is not int or type(v) is not int:
        raise ValueError(f"u and v: expected integers, got {u!r} and {v!r}")
    return torch.roll(
        tensor,
        shifts=(-u % modulus, -v % modulus, -(u + v) % modulus),
        dims=(0, 1, 2),
    )


def explicit_reynolds_projection(centered_logits: torch.Tensor) -> torch.Tensor:
    """Brute-force orbit average, intended for small-p invariant tests."""
    values = _fp64_cpu(centered_logits, "centered_logits")
    modulus = int(values.shape[0])
    total = torch.zeros_like(values)
    for u in range(modulus):
        for v in range(modulus):
            total.add_(apply_group_action(values, u, v))
    return total / (modulus * modulus)


def _labels(modulus: int) -> torch.Tensor:
    indices = torch.arange(modulus, dtype=torch.long)
    return (indices[:, None] + indices[None, :]).remainder(modulus).reshape(-1)


def _validate_indices(indices: torch.Tensor, name: str, count: int) -> torch.Tensor:
    if indices.ndim != 1 or indices.dtype != torch.long or indices.numel() < 1:
        raise ValueError(f"{name}: expected nonempty torch.long vector, got {indices.shape}")
    values = indices.detach().to("cpu")
    if torch.any(values < 0) or torch.any(values >= count):
        raise ValueError(f"{name}: index outside [0, {count})")
    if torch.unique(values).numel() != values.numel():
        raise ValueError(f"{name}: duplicate indices are not allowed")
    return values


def _split_behavior(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float | int]:
    margins = classification_margins(logits, labels)
    predictions = logits.argmax(dim=1)
    errors = predictions != labels
    output: dict[str, float | int] = {
        "cross_entropy": float(F.cross_entropy(logits, labels).item()),
        "accuracy": float((~errors).to(torch.float64).mean().item()),
        "error_count": int(errors.sum().item()),
        "error_rate": float(errors.to(torch.float64).mean().item()),
    }
    output.update({f"margin_{key}": value for key, value in margin_summary(margins).items()})
    return output


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probabilities = F.log_softmax(logits, dim=1)
    return -(log_probabilities.exp() * log_probabilities).sum(dim=1)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def _norm(tensor: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(tensor).item())


def _max_pairwise_difference(values: Iterable[float]) -> float:
    sequence = list(values)
    return max(sequence) - min(sequence)


def _implication_status(value: float, tolerance: float) -> str:
    if value > tolerance:
        return "verified"
    if value < -tolerance:
        return "not_applicable"
    return "numerically_ambiguous"


def invariant_errors(decomposition: ReynoldsDecomposition) -> dict[str, float]:
    """Return scale-aware errors for the M2 Reynolds invariants."""
    centered = decomposition.centered
    projected = decomposition.projected
    residual = decomposition.residual
    centered_norm = _norm(centered)
    projected_norm = _norm(projected)
    residual_norm = _norm(residual)
    centered_energy = centered_norm**2
    reconstruction = centered - projected - residual
    projected_again = reynolds_projection(projected)
    projected_residual = reynolds_projection(residual)
    generator_errors = [
        _norm(projected - apply_group_action(projected, 1, 0)),
        _norm(projected - apply_group_action(projected, 0, 1)),
    ]
    return {
        "centering_max_abs": float(centered.sum(dim=2).abs().max().item()),
        "offset_profile_sum_abs": float(decomposition.offset_profile.sum().abs().item()),
        "reconstruction_relative_error": _norm(reconstruction) / max(1.0, centered_norm),
        "orthogonality_normalized_error": abs(float(torch.sum(projected * residual).item()))
        / max(1.0, projected_norm * residual_norm),
        "energy_identity_relative_error": abs(
            centered_energy - projected_norm**2 - residual_norm**2
        )
        / max(1.0, centered_energy),
        "projection_idempotence_relative_error": _norm(projected_again - projected)
        / max(1.0, projected_norm),
        "residual_projection_relative_error": _norm(projected_residual) / max(1.0, residual_norm),
        "group_invariance_relative_error": max(generator_errors) / max(1.0, projected_norm),
    }


def _invariants_pass(errors: Mapping[str, float], tolerances: MathTolerances) -> bool:
    return bool(
        errors["centering_max_abs"] <= tolerances.centering_atol
        and errors["offset_profile_sum_abs"] <= tolerances.centering_atol
        and errors["reconstruction_relative_error"] <= tolerances.reconstruction_rtol
        and errors["orthogonality_normalized_error"] <= tolerances.orthogonality_normalized
        and errors["energy_identity_relative_error"] <= tolerances.energy_identity_rtol
        and errors["projection_idempotence_relative_error"] <= tolerances.invariance_rtol
        and errors["residual_projection_relative_error"] <= tolerances.invariance_rtol
        and errors["group_invariance_relative_error"] <= tolerances.invariance_rtol
    )


def function_space_metrics(
    logits: torch.Tensor,
    train_indices: torch.Tensor,
    test_indices: torch.Tensor,
    *,
    parameter_norm_total: float | None = None,
    tolerances: MathTolerances | None = None,
) -> FunctionSpaceResult:
    """Compute the complete scalar M2 function metrics for one model state.

    ``train_indices`` and ``test_indices`` index the flattened ``[p*p]`` input
    table.  Full tensors are returned for optional selected-state persistence;
    callers processing a timeline should release the result after each state.
    """
    selected_tolerances = tolerances or MathTolerances()
    if parameter_norm_total is not None and (
        type(parameter_norm_total) not in {float, int}
        or not isfinite(float(parameter_norm_total))
        or parameter_norm_total < 0
    ):
        raise ValueError(
            "parameter_norm_total: expected a finite value >= 0 or null, "
            f"got {parameter_norm_total!r}"
        )
    decomposition = reynolds_decomposition(logits)
    modulus = int(decomposition.centered.shape[0])
    count = modulus * modulus
    train = _validate_indices(train_indices, "train_indices", count)
    test = _validate_indices(test_indices, "test_indices", count)
    if set(train.tolist()) & set(test.tolist()):
        raise ValueError("train_indices and test_indices must be disjoint")

    labels = _labels(modulus)
    raw = logits.detach().to(device="cpu", dtype=torch.float64).reshape(count, modulus)
    projected = decomposition.projected.reshape(count, modulus)
    residual = decomposition.residual.reshape(count, modulus)
    splits = {
        "train": train,
        "test": test,
        "full": torch.arange(count, dtype=torch.long),
    }

    metrics: dict[str, float | int | bool | str | None] = {
        "schema_version": FUNCTION_METRICS_SCHEMA_VERSION,
        "modulus": modulus,
    }
    raw_behaviors: dict[str, dict[str, float | int]] = {}
    projected_behaviors: dict[str, dict[str, float | int]] = {}
    entropy_means: dict[str, float] = {}
    entropy_values = _entropy(raw)
    for split, indices in splits.items():
        split_labels = labels.index_select(0, indices)
        raw_behavior = _split_behavior(raw.index_select(0, indices), split_labels)
        projected_behavior = _split_behavior(projected.index_select(0, indices), split_labels)
        raw_behaviors[split] = raw_behavior
        projected_behaviors[split] = projected_behavior
        for key, value in raw_behavior.items():
            metrics[f"{split}_{key}"] = value
        for key, value in projected_behavior.items():
            projected_key = "ce" if key == "cross_entropy" else key
            metrics[f"{split}_projected_{projected_key}"] = value
        entropy_mean = float(entropy_values.index_select(0, indices).mean().item())
        entropy_means[split] = entropy_mean
        metrics[f"{split}_entropy_mean"] = entropy_mean
        metrics[f"{split}_entropy_normalized"] = entropy_mean / log(modulus)

    profile = decomposition.offset_profile
    gamma = float((profile[0] - profile[1:].max()).item())
    residual_correct = residual.gather(1, labels[:, None]).squeeze(1)
    residual_incorrect = residual.clone()
    residual_incorrect.scatter_(1, labels[:, None], float("-inf"))
    interference = float((residual_incorrect.max(dim=1).values - residual_correct).max().item())
    centered_norm = _norm(decomposition.centered)
    projected_norm = _norm(decomposition.projected)
    residual_norm = _norm(decomposition.residual)
    logit_rms = centered_norm / (modulus**3) ** 0.5
    errors = invariant_errors(decomposition)
    metrics.update(
        {
            "centered_logit_frobenius": centered_norm,
            "centered_logit_rms": logit_rms,
            "equivariant_energy": projected_norm**2,
            "residual_energy": residual_norm**2,
            "D_eq": _safe_ratio(residual_norm**2, centered_norm**2),
            "Gamma": gamma,
            "I": interference,
            "Gamma_minus_I": gamma - interference,
            "Gamma_over_logit_rms": _safe_ratio(gamma, logit_rms),
            "I_over_logit_rms": _safe_ratio(interference, logit_rms),
            "Gamma_over_parameter_l2": (
                None
                if parameter_norm_total is None
                else _safe_ratio(gamma, float(parameter_norm_total))
            ),
            **errors,
            "invariants_passed": _invariants_pass(errors, selected_tolerances),
        }
    )

    projected_ce_error = _max_pairwise_difference(
        float(projected_behaviors[split]["cross_entropy"]) for split in splits
    )
    projected_accuracy_error = _max_pairwise_difference(
        float(projected_behaviors[split]["accuracy"]) for split in splits
    )
    margin_keys = [key for key in projected_behaviors["full"] if key.startswith("margin_")]
    projected_margin_error = max(
        _max_pairwise_difference(float(projected_behaviors[split][key]) for split in splits)
        for key in margin_keys
    )
    metrics.update(
        {
            "projected_split_ce_max_abs_diff": projected_ce_error,
            "projected_split_accuracy_max_abs_diff": projected_accuracy_error,
            "projected_split_margin_max_abs_diff": projected_margin_error,
            "projected_split_invariants_passed": max(
                projected_ce_error,
                projected_accuracy_error,
                projected_margin_error,
            )
            <= selected_tolerances.invariance_rtol,
        }
    )

    raw_status = _implication_status(
        gamma - interference, selected_tolerances.implication_margin_atol
    )
    projected_status = _implication_status(gamma, selected_tolerances.implication_margin_atol)
    if raw_status == "verified" and raw_behaviors["full"]["accuracy"] != 1.0:
        raise AssertionError("Gamma > I + tolerance but raw full-table accuracy is not 1")
    if projected_status == "verified" and projected_behaviors["full"]["accuracy"] != 1.0:
        raise AssertionError("Gamma > tolerance but projected full-table accuracy is not 1")
    metrics["raw_implication_status"] = raw_status
    metrics["projected_implication_status"] = projected_status

    return FunctionSpaceResult(
        metrics=metrics,
        offset_profile=profile,
        centered_logits=decomposition.centered,
        projected_logits=decomposition.projected,
        residual_logits=decomposition.residual,
    )


def _validate_function_records(
    records: Sequence[Mapping[str, Any]], expected_interval: int
) -> list[Mapping[str, Any]]:
    if type(expected_interval) is not int or expected_interval <= 0:
        raise ValueError(f"expected_interval: expected positive integer, got {expected_interval!r}")
    steps = [record.get("step") for record in records]
    if any(type(step) is not int for step in steps):
        raise ValueError(f"function metric steps must be integers: {steps}")
    if any(current <= previous for previous, current in zip(steps, steps[1:], strict=False)):
        raise ValueError(f"function metric steps must be strictly increasing: {steps}")
    regular = [record for record in records if record.get("is_regular_grid") is True]
    if not regular:
        raise ValueError("function metrics must contain at least one regular-grid record")
    regular_steps = [int(record["step"]) for record in regular]
    gaps = [
        (previous, current)
        for previous, current in zip(regular_steps, regular_steps[1:], strict=False)
        if current - previous != expected_interval
    ]
    if gaps:
        raise ValueError(f"regular-grid steps must be spaced by {expected_interval}: gaps={gaps}")
    for record in regular:
        for field in ("Gamma", "I"):
            value = record.get(field)
            if type(value) not in {float, int} or not isfinite(float(value)):
                raise ValueError(
                    f"step {record['step']} {field}: expected finite number, got {value!r}"
                )
    return regular


def _first_event(
    records: Sequence[Mapping[str, Any]], condition: Sequence[bool], interval: int
) -> dict[str, Any]:
    for index, reached in enumerate(condition):
        if reached:
            event_step = int(records[index]["step"])
            previous = int(records[index - 1]["step"]) if index else None
            return {
                "status": "reached",
                "event_step": event_step,
                "previous_regular_step": previous,
                "event_resolution_steps": interval,
                "event_interval": {
                    "lower_exclusive": previous,
                    "upper_inclusive": event_step,
                },
            }
    return {
        "status": "not_reached",
        "event_step": None,
        "previous_regular_step": None,
        "event_resolution_steps": interval,
        "event_interval": {"lower_exclusive": None, "upper_inclusive": None},
    }


def _condition_summary(
    records: Sequence[Mapping[str, Any]], condition: Sequence[bool]
) -> dict[str, float | int | None]:
    steps = [int(record["step"]) for record in records]
    runs: list[tuple[int, int]] = []
    start: int | None = None
    false_to_true = 0
    true_to_false = 0
    exits: list[int] = []
    for index, value in enumerate(condition):
        previous = condition[index - 1] if index else None
        if value and (previous is False or previous is None):
            start = steps[index]
            if previous is False:
                false_to_true += 1
        if not value and previous is True:
            if start is None:  # pragma: no cover - guarded by transition construction
                raise AssertionError("true run is missing its start")
            runs.append((start, steps[index - 1]))
            start = None
            true_to_false += 1
            exits.append(steps[index])
    if start is not None:
        runs.append((start, steps[-1]))
    longest = max((end - begin for begin, end in runs), default=0)
    return {
        "positive_run_count": len(runs),
        "false_to_true_crossing_count": false_to_true,
        "true_to_false_exit_count": true_to_false,
        "longest_true_window_steps": longest,
        "fraction_true": sum(condition) / len(condition),
        "last_exit_step": exits[-1] if exits else None,
    }


def detect_function_events(
    records: Sequence[Mapping[str, Any]], *, expected_interval: int = 100
) -> dict[str, Any]:
    """Detect first function events and stability summaries on the regular grid only.

    Directed replay records may appear in ``records`` but must set
    ``is_regular_grid=false``.  They never change event estimates or fractions.
    An exit step is the first false regular-grid state after a true run.
    """
    regular = _validate_function_records(records, expected_interval)
    gamma_positive = [float(record["Gamma"]) > 0.0 for record in regular]
    dominant = [float(record["Gamma"]) > float(record["I"]) for record in regular]
    gamma_summary = _condition_summary(regular, gamma_positive)
    dominant_summary = _condition_summary(regular, dominant)
    output = {
        "schema_version": FUNCTION_EVENTS_SCHEMA_VERSION,
        "summary_grid": "regular_manifest_checkpoint_grid",
        "regular_step_count": len(regular),
        "regular_interval_steps": expected_interval,
        "event_resolution_steps": expected_interval,
        "first_regular_step": int(regular[0]["step"]),
        "last_regular_step": int(regular[-1]["step"]),
        "t_alg": _first_event(regular, gamma_positive, expected_interval),
        "t_dom": _first_event(regular, dominant, expected_interval),
        "gamma_positive_run_count": gamma_summary["positive_run_count"],
        "gamma_false_to_true_crossing_count": gamma_summary["false_to_true_crossing_count"],
        "gamma_true_to_false_exit_count": gamma_summary["true_to_false_exit_count"],
        "longest_gamma_positive_window_steps": gamma_summary["longest_true_window_steps"],
        "fraction_gamma_positive": gamma_summary["fraction_true"],
        "last_gamma_positive_exit_step": gamma_summary["last_exit_step"],
        "dominance_positive_run_count": dominant_summary["positive_run_count"],
        "dominance_false_to_true_crossing_count": dominant_summary["false_to_true_crossing_count"],
        "dominance_true_to_false_exit_count": dominant_summary["true_to_false_exit_count"],
        "longest_dominance_window_steps": dominant_summary["longest_true_window_steps"],
        "fraction_dominant": dominant_summary["fraction_true"],
        "last_dominance_exit_step": dominant_summary["last_exit_step"],
    }
    alignment_statuses = [record.get("committed_behavior_alignment_status") for record in records]
    if any(status is not None for status in alignment_statuses):
        allowed = {
            "uncommitted_initialization",
            "prediction_exact",
            "batch_sensitive_predictions",
        }
        if any(status not in allowed for status in alignment_statuses):
            raise ValueError(f"invalid committed behavior alignment status: {alignment_statuses}")
        output["behavior_alignment"] = {
            "state_count": len(records),
            "prediction_exact_state_count": alignment_statuses.count("prediction_exact"),
            "batch_sensitive_prediction_state_count": alignment_statuses.count(
                "batch_sensitive_predictions"
            ),
            "uncommitted_initialization_state_count": alignment_statuses.count(
                "uncommitted_initialization"
            ),
            "regular_batch_sensitive_prediction_state_count": sum(
                record.get("is_regular_grid") is True
                and record.get("committed_behavior_alignment_status")
                == "batch_sensitive_predictions"
                for record in records
            ),
        }
    return output
