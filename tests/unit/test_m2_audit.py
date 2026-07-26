"""Focused safety tests for the M2 analysis audit boundary."""

from __future__ import annotations

import csv
import json

import pytest

from transgrokking.analysis.artifacts import write_csv
from transgrokking.analysis.config import load_m2_analysis_config
from transgrokking.metrics.audit_m2 import (
    _bridge_structure_is_valid,
    _committed_ce_alignment_valid,
    _expected_implication_status,
    _inventory_sources,
    _lineage_table_errors,
    _read_json,
)


def _bridge(step: int, run_id: str) -> dict[str, object]:
    digest = "a" * 64
    return {
        "schema_version": 1,
        "run_id": run_id,
        "source_step": step - 50,
        "target_step": step,
        "endpoint_step": step + 50,
        "replay_updates": 50,
        "midpoint_repeat_sha256": [digest, digest],
        "endpoint_repeat_sha256": [digest, digest],
        "endpoint_semantic_equal": True,
        "midpoint_behavior_repeat_equal": True,
        "endpoint_behavior_equal": True,
        "endpoint_replay_behavior_sha256": digest,
        "endpoint_checkpoint_behavior_sha256": digest,
        "endpoint_differing_components": [[], []],
        "source_checkpoint_unchanged": True,
        "endpoint_checkpoint_unchanged": True,
        "validation_status": "passed",
    }


def test_bridge_audit_requires_segment_owner_and_unchanged_sources() -> None:
    config = load_m2_analysis_config("configs/analysis/m2_function_space.yaml")
    bridge = _bridge(5050, config.source_run_ids.canonical_parent)
    assert _bridge_structure_is_valid([bridge], {5050}, config)

    for field in ("source_checkpoint_unchanged", "endpoint_checkpoint_unchanged"):
        invalid = dict(bridge)
        invalid[field] = False
        assert not _bridge_structure_is_valid([invalid], {5050}, config)

    wrong_owner = dict(bridge)
    wrong_owner["run_id"] = config.source_run_ids.root
    assert not _bridge_structure_is_valid([wrong_owner], {5050}, config)


def test_implication_status_uses_explicit_numeric_buffer() -> None:
    tolerance = 1.0e-10
    assert _expected_implication_status(2.0e-10, tolerance) == "verified"
    assert _expected_implication_status(-2.0e-10, tolerance) == "not_applicable"
    assert _expected_implication_status(tolerance, tolerance) == "numerically_ambiguous"
    assert _expected_implication_status(-tolerance, tolerance) == "numerically_ambiguous"


def test_committed_ce_alignment_uses_configured_atol_and_rtol() -> None:
    config = load_m2_analysis_config("configs/analysis/m2_function_space.yaml")
    committed = {"train_cross_entropy": 4.5, "test_cross_entropy": 3.0}
    row = {
        "train_cross_entropy": 4.500003,
        "test_cross_entropy": 3.000002,
        "committed_ce_max_abs_diff": 3.0e-6,
        "committed_ce_within_tolerance": True,
    }

    assert _committed_ce_alignment_valid(row, committed, config)
    row["committed_ce_max_abs_diff"] = 2.0e-6
    assert not _committed_ce_alignment_valid(row, committed, config)


def test_source_inventory_rejects_provenance_run_id_substitution() -> None:
    config = load_m2_analysis_config("configs/analysis/m2_function_space.yaml")
    provenance = {"source_run_ids": config.to_dict()["source_run_ids"]}
    provenance["source_run_ids"]["root"] = "../../outside"
    with pytest.raises(ValueError, match="differ from the resolved analysis config"):
        _inventory_sources(config, provenance)


def test_lineage_tables_are_bound_to_frozen_checkpoint_hashes() -> None:
    config = load_m2_analysis_config("configs/analysis/m2_function_space.yaml")
    ids = config.source_run_ids
    segments = {
        ids.root: range(0, 5_001, 100),
        ids.canonical_parent: range(5_000, 20_001, 100),
        ids.terminal_child: range(20_000, 50_001, 100),
    }
    physical = []
    inventories = {}
    for run_index, (run_id, steps) in enumerate(segments.items(), start=10):
        inventory = []
        for step in steps:
            raw_sha = f"{run_index:01x}{step:063x}"[-64:]
            relative = f"runs/{run_id}/checkpoints/step_{step:06d}.pt"
            physical.append(
                {
                    "step": str(step),
                    "run_id": run_id,
                    "checkpoint_relative_path": relative,
                    "checkpoint_sha256": raw_sha,
                    "semantic_state_sha256": f"{step:064x}"[-64:],
                }
            )
            inventory.append(
                {
                    "path": f"checkpoints/step_{step:06d}.pt",
                    "size": 1,
                    "sha256": raw_sha,
                }
            )
        inventories[f"runs/{run_id}"] = inventory

    def owner(step: int) -> str:
        if step <= 5_000:
            return ids.root
        if step <= 20_000:
            return ids.canonical_parent
        return ids.terminal_child

    by_key = {(row["run_id"], int(row["step"])): row for row in physical}
    canonical = []
    for step in range(0, 50_001, 100):
        row = dict(by_key[(owner(step), step)])
        row["alias_count"] = "1" if step in {5_000, 20_000} else "0"
        canonical.append(row)
    aliases = []
    for step, pair in {
        5_000: (ids.root, ids.canonical_parent),
        20_000: (ids.canonical_parent, ids.terminal_child),
    }.items():
        left = by_key[(pair[0], step)]
        right = by_key[(pair[1], step)]
        aliases.append(
            {
                "step": str(step),
                "canonical_run_id": pair[0],
                "alias_run_id": pair[1],
                "canonical_checkpoint_sha256": left["checkpoint_sha256"],
                "alias_checkpoint_sha256": right["checkpoint_sha256"],
                "semantic_state_sha256": left["semantic_state_sha256"],
                "semantic_state_equal": "True",
                "raw_sha256_equal": "False",
            }
        )
    assert not _lineage_table_errors(physical, canonical, aliases, inventories, config)

    inventories[f"runs/{ids.terminal_child}"][-1]["sha256"] = "f" * 64
    errors = _lineage_table_errors(physical, canonical, aliases, inventories, config)
    assert "physical_sha:" in " ".join(errors)


def test_audit_json_reader_rejects_nonfinite_values(tmp_path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text('{"metric": NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        _read_json(path)


def test_atomic_csv_preserves_old_file_and_json_encodes_roles(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    write_csv(path, [{"step": 50, "state_roles": ["test_001:onset"]}], ["step", "state_roles"])
    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert json.loads(row["state_roles"]) == ["test_001:onset"]
    before = path.read_bytes()

    with pytest.raises(ValueError, match="non-finite"):
        write_csv(path, [{"step": 100, "state_roles": [float("nan")]}], ["step", "state_roles"])
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".metrics.csv.*"))
