from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.nn import functional as F

from transgrokking.analysis.checkpoint_resolver import (
    LineageConflictError,
    LineageValidationError,
    compare_semantic_states,
    file_sha256,
    resolve_checkpoint_lineage,
    semantic_state_sha256,
)
from transgrokking.analysis.config import load_m2_analysis_config
from transgrokking.analysis.replay import (
    EpisodeSelection,
    ReplayBridgeResult,
    ReplayState,
    SelectedEpisodeState,
    replay_checkpoint_bridge,
    select_episode_states,
)
from transgrokking.analysis.runner import _run_replay_bridges
from transgrokking.config import ExperimentConfig, load_config
from transgrokking.data import generate_modular_addition
from transgrokking.training.checkpoint import read_checkpoint, save_checkpoint
from transgrokking.training.optimizer import build_adamw
from transgrokking.training.trainer import build_model
from transgrokking.utils.reproducibility import capture_rng_state, configure_reproducibility


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _minimal_payload(step: int, *, execution_max_steps: int) -> dict[str, Any]:
    configure_reproducibility(17, True)
    return {
        "schema_version": 2,
        "model_state": {"weight": torch.tensor([float(step)])},
        "optimizer_state": {"state": {}, "param_groups": []},
        "optimizer_group_signature": [],
        "scheduler_state": None,
        "global_step": step,
        "config": {"optimization": {"max_steps": execution_max_steps}},
        "scientific_config_hash": "a" * 64,
        "split_hash": "b" * 64,
        "optimizer_type": "adamw",
        **capture_rng_state(),
    }


def _write_run(
    root: Path,
    run_id: str,
    steps: list[int],
    *,
    parent_run_id: str | None,
    parent_step: int | None,
    execution_max_steps: int,
) -> Path:
    run = root / run_id
    checkpoint_dir = run / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    entries = []
    for step in steps:
        path = checkpoint_dir / f"step_{step:06d}.pt"
        torch.save(_minimal_payload(step, execution_max_steps=execution_max_steps), path)
        entries.append({"step": step, "path": path.name})
    _write_json(
        run / "metadata.json",
        {
            "schema_version": 2,
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "parent_global_step": parent_step,
            "scientific_config_hash": "a" * 64,
            "split_hash": "b" * 64,
            "git_commit": "c" * 40,
        },
    )
    _write_json(checkpoint_dir / "manifest.json", {"schema_version": 2, "checkpoints": entries})
    return run


def _lineage_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = _write_run(
        tmp_path, "root", [0, 1], parent_run_id=None, parent_step=None, execution_max_steps=1
    )
    child = _write_run(
        tmp_path,
        "child",
        [1, 2],
        parent_run_id="root",
        parent_step=1,
        execution_max_steps=2,
    )
    terminal = _write_run(
        tmp_path,
        "terminal",
        [2, 3],
        parent_run_id="child",
        parent_step=2,
        execution_max_steps=3,
    )
    return root, child, terminal


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def test_semantic_hash_excludes_execution_config_but_detects_training_state() -> None:
    left = _minimal_payload(1, execution_max_steps=1)
    right = _minimal_payload(1, execution_max_steps=50_000)
    assert semantic_state_sha256(left) == semantic_state_sha256(right)
    assert compare_semantic_states(left, right).equal

    right["model_state"]["weight"].add_(1.0)
    comparison = compare_semantic_states(left, right)
    assert not comparison.equal
    assert comparison.differing_components == ("model_state",)
    assert comparison.detail_differences == ("model_state.weight: value differs",)


def test_resolver_verifies_aliases_grid_segment_owners_and_zero_write(tmp_path: Path) -> None:
    runs = _lineage_fixture(tmp_path)
    before = _tree_hashes(tmp_path)
    lineage = resolve_checkpoint_lineage(
        list(reversed(runs)),
        expected_physical_count=6,
        expected_regular_count=4,
        regular_interval=1,
        expected_start_step=0,
        expected_end_step=3,
    )
    assert lineage.physical_count == 6
    assert lineage.regular_step_count == 4
    assert [item.step for item in lineage.canonical] == [0, 1, 2, 3]
    assert [item.physical.run_id for item in lineage.canonical] == [
        "root",
        "root",
        "child",
        "terminal",
    ]
    assert [item.step for item in lineage.aliases] == [1, 2]
    assert all(not item.raw_sha256_equal for item in lineage.aliases)
    assert all(item.to_record()["semantic_state_equal"] for item in lineage.aliases)
    assert all(item.to_record()["alias_group_id"].startswith("step_") for item in lineage.aliases)
    assert lineage.segment_for_replay_target(1).run_id == "root"
    assert lineage.segment_for_replay_target(2).run_id == "child"
    assert lineage.segment_for_replay_target(3).run_id == "terminal"
    assert lineage.checkpoint_at(1).physical.checkpoint_relative_path.startswith(
        "runs/root/checkpoints/"
    )
    assert _tree_hashes(tmp_path) == before


def test_resolver_reports_manifest_counts_and_missing_grid(tmp_path: Path) -> None:
    runs = _lineage_fixture(tmp_path)
    with pytest.raises(LineageValidationError, match="required grid") as captured:
        resolve_checkpoint_lineage(
            runs,
            expected_physical_count=7,
            expected_regular_count=4,
            regular_interval=1,
            expected_start_step=0,
            expected_end_step=3,
        )
    assert captured.value.diagnostics["manifest_counts"] == {
        "root": 2,
        "child": 2,
        "terminal": 2,
    }
    assert captured.value.diagnostics["physical_count"] == 6


def test_resolver_rejects_semantically_different_branch_anchor(tmp_path: Path) -> None:
    runs = _lineage_fixture(tmp_path)
    anchor = runs[1] / "checkpoints" / "step_000001.pt"
    payload = read_checkpoint(anchor)
    payload["model_state"]["weight"].add_(1.0)
    torch.save(payload, anchor)
    with pytest.raises(LineageConflictError, match="step 1") as captured:
        resolve_checkpoint_lineage(
            runs,
            expected_physical_count=6,
            expected_regular_count=4,
            regular_interval=1,
            expected_start_step=0,
            expected_end_step=3,
        )
    assert captured.value.diagnostics["comparison"]["differing_components"] == ["model_state"]


def _primitive(
    episode_id: str,
    split: str,
    onset: int,
    trough: int,
    *,
    recovery_start: int | None,
    recovery_confirmed: int | None,
) -> dict[str, Any]:
    recovered = recovery_start is not None
    return {
        "episode_id": episode_id,
        "episode_type": split,
        "onset_step": onset,
        "train_trough_step": trough,
        "test_trough_step": trough,
        "train_recovery_step": recovery_start if split == "train" else None,
        "test_recovery_step": recovery_start if split == "test" else None,
        "train_recovery_confirmed_step": recovery_confirmed if split == "train" else None,
        "test_recovery_confirmed_step": recovery_confirmed if split == "test" else None,
        "status": "recovered" if recovered else "not_recovered",
    }


def test_episode_selector_keeps_split_troughs_terminal_and_multi_roles() -> None:
    train_only = _primitive(
        "train_001", "train", 100, 100, recovery_start=200, recovery_confirmed=300
    )
    train_joint = _primitive(
        "train_002", "train", 150, 150, recovery_start=200, recovery_confirmed=300
    )
    test_recovered = _primitive(
        "test_001", "test", 150, 200, recovery_start=250, recovery_confirmed=350
    )
    test_unrecovered = _primitive(
        "test_002", "test", 400, 450, recovery_start=None, recovery_confirmed=None
    )
    joint = {
        "episode_id": "joint_001",
        "episode_type": "joint",
        "train_episode_id": "train_002",
        "test_episode_id": "test_001",
        "onset_step": 150,
        "train_trough_step": 150,
        "test_trough_step": 200,
        "train_recovery_step": 200,
        "train_recovery_confirmed_step": 300,
        "test_recovery_step": 250,
        "test_recovery_confirmed_step": 350,
        "joint_recovery_step": 250,
        "joint_recovery_confirmed_step": 350,
        "status": "recovered",
    }
    artifact = {
        "train_episodes": [train_only, train_joint],
        "test_episodes": [test_recovered, test_unrecovered],
        "joint_episodes": [joint],
    }
    selection = select_episode_states(
        artifact,
        [50, 100, 150, 200, 250, 300, 350, 400, 450],
        selected_train_episode_ids=["train_001"],
        extra_state_roles={0: ["initialization"], 500: ["terminal"]},
        terminal_step=500,
    )
    roles = {state.step: set(state.state_roles) for state in selection.states}
    assert "joint_001:train_trough" in roles[150]
    assert "joint_001:test_trough" in roles[200]
    assert "test_002:terminal_unrecovered" in roles[500]
    unrecovered = [item for item in selection.references if item.episode_id == "test_002"]
    assert not any("recovery" in item.state_role for item in unrecovered)
    assert {item.state_role for item in unrecovered} == {
        "pre_collapse",
        "onset",
        "test_trough",
        "terminal_unrecovered",
    }
    with pytest.raises(ValueError, match="referenced by joint"):
        select_episode_states(
            artifact,
            [50, 100, 150, 200, 250, 300, 350, 400, 450],
            selected_train_episode_ids=["train_002"],
        )


def _two_step_checkpoints(tmp_path: Path) -> tuple[ExperimentConfig, Path, Path]:
    original = load_config("configs/smoke.yaml")
    config = replace(original, optimization=replace(original.optimization, max_steps=2))
    configure_reproducibility(config.optimization.seed, config.optimization.deterministic)
    data = generate_modular_addition(
        config.task.modulus, config.task.train_fraction, config.task.split_seed
    )
    model = build_model(config)
    optimizer, _ = build_adamw(model, config.optimization)
    source = tmp_path / "step_000000.pt"
    endpoint = tmp_path / "step_000002.pt"
    save_checkpoint(source, model, optimizer, config, data.split_hash, 0)
    inputs = data.inputs
    labels = data.labels
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs.index_select(0, data.train_indices))[:, -1]
        loss = F.cross_entropy(logits, labels.index_select(0, data.train_indices))
        loss.backward()
        optimizer.step()
    save_checkpoint(endpoint, model, optimizer, config, data.split_hash, 2)
    return config, source, endpoint


def test_replay_is_repeatable_bridges_endpoint_and_never_writes_source(tmp_path: Path) -> None:
    config, source, endpoint = _two_step_checkpoints(tmp_path)
    before = _tree_hashes(tmp_path)
    result = replay_checkpoint_bridge(source, endpoint, config, 1, repeats=2)
    assert result.source_step == 0
    assert result.midpoint.step == 1
    assert result.endpoint.step == 2
    assert result.replay_updates == 1
    assert len(set(result.midpoint_repeat_sha256)) == 1
    assert len(set(result.endpoint_repeat_sha256)) == 1
    assert all(comparison.equal for comparison in result.endpoint_comparisons)
    assert result.source_checkpoint_unchanged
    assert result.endpoint_checkpoint_unchanged
    assert result.endpoint.behavior == result.physical_endpoint_behavior
    assert _tree_hashes(tmp_path) == before


def test_runner_bridges_each_midpoint_with_owning_segment_and_two_repeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_run(
        tmp_path, "root", [0, 100], parent_run_id=None, parent_step=None, execution_max_steps=100
    )
    child = _write_run(
        tmp_path,
        "child",
        [100, 200],
        parent_run_id="root",
        parent_step=100,
        execution_max_steps=200,
    )
    terminal = _write_run(
        tmp_path,
        "terminal",
        [200, 300],
        parent_run_id="child",
        parent_step=200,
        execution_max_steps=300,
    )
    lineage = resolve_checkpoint_lineage(
        [root, child, terminal],
        expected_physical_count=6,
        expected_regular_count=4,
        regular_interval=100,
        expected_start_step=0,
        expected_end_step=300,
    )
    selection = EpisodeSelection(
        references=(),
        states=tuple(
            SelectedEpisodeState(step=step, state_roles=(f"protocol:{step}",))
            for step in (0, 50, 100, 150, 200, 250, 300)
        ),
    )
    calls: list[tuple[str, int, int, int]] = []

    def fake_bridge(
        source_checkpoint: str | Path,
        endpoint_checkpoint: str | Path,
        config: ExperimentConfig,
        midpoint_step: int,
        *,
        device: str | torch.device,
        repeats: int,
    ) -> ReplayBridgeResult:
        source_path = Path(source_checkpoint)
        endpoint_path = Path(endpoint_checkpoint)
        source_step = int(read_checkpoint(source_path)["global_step"])
        endpoint_step = int(read_checkpoint(endpoint_path)["global_step"])
        calls.append((source_path.parent.parent.name, source_step, midpoint_step, repeats))
        midpoint_payload = _minimal_payload(
            midpoint_step, execution_max_steps=config.optimization.max_steps
        )
        endpoint_payload = read_checkpoint(endpoint_path)
        endpoint_comparison = compare_semantic_states(endpoint_payload, endpoint_payload)
        midpoint_state = ReplayState(
            step=midpoint_step,
            semantic_state_sha256=semantic_state_sha256(midpoint_payload),
            checkpoint_payload=midpoint_payload,
            behavior={},
        )
        endpoint_state = ReplayState(
            step=endpoint_step,
            semantic_state_sha256=semantic_state_sha256(endpoint_payload),
            checkpoint_payload=endpoint_payload,
            behavior={},
        )
        return ReplayBridgeResult(
            source_checkpoint=source_path,
            endpoint_checkpoint=endpoint_path,
            source_checkpoint_sha256=file_sha256(source_path),
            endpoint_checkpoint_sha256=file_sha256(endpoint_path),
            source_checkpoint_unchanged=True,
            endpoint_checkpoint_unchanged=True,
            source_step=source_step,
            midpoint_step=midpoint_step,
            endpoint_step=endpoint_step,
            replay_updates=midpoint_step - source_step,
            midpoint=midpoint_state,
            endpoint=endpoint_state,
            physical_endpoint_behavior={},
            midpoint_repeat_sha256=(
                midpoint_state.semantic_state_sha256,
                midpoint_state.semantic_state_sha256,
            ),
            endpoint_repeat_sha256=(
                endpoint_state.semantic_state_sha256,
                endpoint_state.semantic_state_sha256,
            ),
            endpoint_comparisons=(endpoint_comparison, endpoint_comparison),
        )

    monkeypatch.setattr("transgrokking.analysis.runner.replay_checkpoint_bridge", fake_bridge)
    monkeypatch.setattr(
        "transgrokking.analysis.runner.repository_relative", lambda path: Path(path).name
    )
    analysis_root = tmp_path / "analysis"
    (analysis_root / "cache").mkdir(parents=True)
    (analysis_root / "m2a").mkdir()
    experiment = replace(
        load_config("configs/smoke.yaml"),
        optimization=replace(load_config("configs/smoke.yaml").optimization, max_steps=300),
    )
    analysis_config = load_m2_analysis_config("configs/analysis/m2_function_space.yaml")
    payloads, records = _run_replay_bridges(
        analysis_root, lineage, selection, experiment, analysis_config
    )
    assert calls == [
        ("root", 0, 50, 2),
        ("child", 100, 150, 2),
        ("terminal", 200, 250, 2),
    ]
    assert sorted(payloads) == [50, 150, 250]
    assert [record["target_step"] for record in records] == [50, 150, 250]
    assert all(record["source_checkpoint_unchanged"] for record in records)
    assert all(record["endpoint_checkpoint_unchanged"] for record in records)
