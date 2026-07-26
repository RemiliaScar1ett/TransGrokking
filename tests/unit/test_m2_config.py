"""Strict configuration tests for the M2 analysis sidecar."""

from __future__ import annotations

import copy

import pytest
import yaml

from transgrokking.analysis.config import load_m2_analysis_config, m2_analysis_config_from_dict


def _raw() -> dict:
    with open("configs/analysis/m2_function_space.yaml", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    assert isinstance(value, dict)
    return value


def test_m2_analysis_config_is_strict_and_has_independent_hash() -> None:
    config = load_m2_analysis_config("configs/analysis/m2_function_space.yaml")
    assert config.profile == "m2-function-space"
    assert config.analysis_batch_size == 1024
    assert config.cpu_reduction_dtype == "float64"
    assert len(config.analysis_hash()) == 64

    raw = _raw()
    raw["unknown"] = True
    with pytest.raises(ValueError, match="unknown"):
        m2_analysis_config_from_dict(raw)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("analysis_batch_size",), True, "expected int"),
        (("behavior_validation_atol",), 1.1e-6, "expected 0 <= value <= 1e-6"),
        (("replay_enabled",), False, "requires true"),
        (("persist_selected_logits",), False, "requires true"),
        (("math_tolerances", "energy_identity_rtol"), 0.0, "expected 0 < value"),
        (("source_result_dirs", "m1_reference"), "C:/absolute", "repository-relative"),
    ],
)
def test_m2_analysis_config_rejects_invalid_values(path, value, message: str) -> None:
    raw = copy.deepcopy(_raw())
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        m2_analysis_config_from_dict(raw)
