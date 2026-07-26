"""Deterministic, source-read-only replay and episode-state selection primitives."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer

from transgrokking.analysis.checkpoint_resolver import (
    SemanticStateComparison,
    compare_semantic_states,
    file_sha256,
    semantic_state_sha256,
)
from transgrokking.config import ExperimentConfig
from transgrokking.data import ModularAdditionData, generate_modular_addition
from transgrokking.metrics.behavior import evaluate_model_behavior
from transgrokking.training.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    load_checkpoint,
    optimizer_group_signature,
    read_checkpoint,
)
from transgrokking.training.optimizer import (
    ParameterGrouping,
    build_adamw,
    validate_optimizer_parameter_identity,
)
from transgrokking.training.trainer import build_model
from transgrokking.utils.reproducibility import (
    capture_rng_state,
    configure_reproducibility,
    restore_rng_state,
)

BehaviorEvaluator = Callable[
    [nn.Module, ModularAdditionData, ParameterGrouping, torch.device], dict[str, Any]
]


class ReplayValidationError(RuntimeError):
    """Raised when replay is not deterministic or cannot bridge to the real endpoint."""


@dataclass(frozen=True)
class EpisodeStateReference:
    """One episode role pointing to one exact evaluation state."""

    target_step: int
    episode_id: str
    episode_type: str
    state_role: str
    episode_status: str

    @property
    def role_id(self) -> str:
        return f"{self.episode_id}:{self.state_role}"

    def to_record(self) -> dict[str, Any]:
        return {
            "target_step": self.target_step,
            "episode_id": self.episode_id,
            "episode_type": self.episode_type,
            "state_role": self.state_role,
            "state_role_id": self.role_id,
            "episode_status": self.episode_status,
        }


@dataclass(frozen=True)
class SelectedEpisodeState:
    """A unique step carrying one or more episode or protocol roles."""

    step: int
    state_roles: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        return {"step": self.step, "state_roles": list(self.state_roles)}


@dataclass(frozen=True)
class EpisodeSelection:
    """Expanded episode references and their unique multi-role target states."""

    references: tuple[EpisodeStateReference, ...]
    states: tuple[SelectedEpisodeState, ...]

    @property
    def target_steps(self) -> tuple[int, ...]:
        return tuple(state.step for state in self.states)


@dataclass(frozen=True)
class ReplayState:
    """One exact in-memory replay state and its recomputed M1 behavior."""

    step: int
    semantic_state_sha256: str
    checkpoint_payload: dict[str, Any]
    behavior: dict[str, Any]


@dataclass(frozen=True)
class ReplayBridgeResult:
    """Midpoint state plus proof that replay reaches the next real checkpoint."""

    source_checkpoint: Path
    endpoint_checkpoint: Path
    source_checkpoint_sha256: str
    endpoint_checkpoint_sha256: str
    source_checkpoint_unchanged: bool
    endpoint_checkpoint_unchanged: bool
    source_step: int
    midpoint_step: int
    endpoint_step: int
    replay_updates: int
    midpoint: ReplayState
    endpoint: ReplayState
    physical_endpoint_behavior: dict[str, Any]
    midpoint_repeat_sha256: tuple[str, ...]
    endpoint_repeat_sha256: tuple[str, ...]
    endpoint_comparisons: tuple[SemanticStateComparison, ...]


def _validate_steps(steps: Iterable[int]) -> list[int]:
    values = list(steps)
    if any(type(step) is not int or step < 0 for step in values):
        raise ValueError(f"scalar steps must be nonnegative integers, got {values!r}")
    if values != sorted(set(values)):
        raise ValueError("scalar steps must be unique and strictly increasing")
    return values


def _reference(episode: dict[str, Any], step: Any, role: str) -> EpisodeStateReference | None:
    if step is None:
        return None
    if type(step) is not int or step < 0:
        raise ValueError(f"{episode.get('episode_id')}.{role} has invalid step {step!r}")
    return EpisodeStateReference(
        target_step=step,
        episode_id=str(episode["episode_id"]),
        episode_type=str(episode["episode_type"]),
        state_role=role,
        episode_status=str(episode["status"]),
    )


def _next_step(steps: list[int], boundary: int) -> int | None:
    return next((step for step in steps if step > boundary), None)


def _previous_step(steps: list[int], boundary: int) -> int | None:
    return next((step for step in reversed(steps) if step < boundary), None)


def _primitive_references(
    episode: dict[str, Any], steps: list[int], terminal_step: int
) -> list[EpisodeStateReference]:
    split = episode.get("episode_type")
    if split not in {"train", "test"}:
        raise ValueError(f"primitive episode has invalid type: {split!r}")
    onset = episode.get("onset_step")
    if type(onset) is not int:
        raise ValueError(f"{episode.get('episode_id')}.onset_step must be an integer")
    fields = [
        (_previous_step(steps, onset), "pre_collapse"),
        (onset, "onset"),
        (episode.get(f"{split}_trough_step"), f"{split}_trough"),
    ]
    recovery_start = episode.get(f"{split}_recovery_step")
    recovery_confirmed = episode.get(f"{split}_recovery_confirmed_step")
    status = episode.get("status")
    if status == "recovered":
        if recovery_start is None or recovery_confirmed is None:
            raise ValueError(f"recovered episode {episode.get('episode_id')} lacks recovery steps")
        fields.extend(
            [
                (recovery_start, f"{split}_recovery_start"),
                (recovery_confirmed, f"{split}_recovery_confirmed"),
                (_next_step(steps, recovery_confirmed), "post_recovery"),
            ]
        )
    elif status == "not_recovered":
        if recovery_start is not None or recovery_confirmed is not None:
            raise ValueError(
                f"unrecovered episode {episode.get('episode_id')} must keep recovery fields null"
            )
        fields.append((terminal_step, "terminal_unrecovered"))
    else:
        raise ValueError(f"episode {episode.get('episode_id')} has invalid status {status!r}")
    return [item for step, role in fields if (item := _reference(episode, step, role))]


def _joint_references(
    episode: dict[str, Any], steps: list[int], terminal_step: int
) -> list[EpisodeStateReference]:
    if episode.get("episode_type") != "joint":
        raise ValueError(f"joint episode has invalid type: {episode.get('episode_type')!r}")
    onset = episode.get("onset_step")
    if type(onset) is not int:
        raise ValueError(f"{episode.get('episode_id')}.onset_step must be an integer")
    fields: list[tuple[Any, str]] = [
        (_previous_step(steps, onset), "pre_collapse"),
        (onset, "onset"),
        (episode.get("train_trough_step"), "train_trough"),
        (episode.get("test_trough_step"), "test_trough"),
    ]
    for split in ("train", "test"):
        start = episode.get(f"{split}_recovery_step")
        confirmed = episode.get(f"{split}_recovery_confirmed_step")
        if start is not None:
            fields.append((start, f"{split}_recovery_start"))
        if confirmed is not None:
            fields.append((confirmed, f"{split}_recovery_confirmed"))
    status = episode.get("status")
    if status == "recovered":
        joint_confirmed = episode.get("joint_recovery_confirmed_step")
        if joint_confirmed is None:
            raise ValueError(
                f"recovered joint episode {episode.get('episode_id')} lacks confirmation"
            )
        fields.append((_next_step(steps, joint_confirmed), "post_recovery"))
    elif status == "not_recovered":
        if (
            episode.get("joint_recovery_step") is not None
            or episode.get("joint_recovery_confirmed_step") is not None
        ):
            raise ValueError(
                f"unrecovered joint episode {episode.get('episode_id')} has joint recovery fields"
            )
        fields.append((terminal_step, "terminal_unrecovered"))
    else:
        raise ValueError(f"episode {episode.get('episode_id')} has invalid status {status!r}")
    return [item for step, role in fields if (item := _reference(episode, step, role))]


def select_episode_states(
    collapse_artifact: Mapping[str, Any],
    scalar_steps: Iterable[int],
    *,
    selected_train_episode_ids: Iterable[str] = (),
    extra_state_roles: Mapping[int, Iterable[str]] | None = None,
    terminal_step: int | None = None,
) -> EpisodeSelection:
    """Expand M1 primitive/composite episodes into exact, multi-role M2 target states."""
    steps = _validate_steps(scalar_steps)
    if not steps:
        raise ValueError("scalar steps must not be empty")
    terminal = steps[-1] if terminal_step is None else terminal_step
    if type(terminal) is not int or terminal < steps[-1]:
        raise ValueError(f"terminal_step must be an integer >= {steps[-1]}, got {terminal!r}")
    train_episodes = collapse_artifact.get("train_episodes")
    test_episodes = collapse_artifact.get("test_episodes")
    joint_episodes = collapse_artifact.get("joint_episodes")
    episode_lists = (train_episodes, test_episodes, joint_episodes)
    if not all(isinstance(value, list) for value in episode_lists):
        raise ValueError("collapse artifact must contain train/test/joint episode lists")
    train_by_id = {episode.get("episode_id"): episode for episode in train_episodes}
    requested_train = tuple(selected_train_episode_ids)
    missing = sorted(set(requested_train) - set(train_by_id))
    if missing:
        raise ValueError(f"selected train episodes do not exist: {missing}")
    joint_train_ids = {episode.get("train_episode_id") for episode in joint_episodes}
    not_train_only = sorted(set(requested_train) & joint_train_ids)
    if not_train_only:
        raise ValueError(
            f"selected train-only episodes are referenced by joint episodes: {not_train_only}"
        )

    references: list[EpisodeStateReference] = []
    for episode in test_episodes:
        references.extend(_primitive_references(episode, steps, terminal))
    for episode in joint_episodes:
        references.extend(_joint_references(episode, steps, terminal))
    for episode_id in requested_train:
        references.extend(_primitive_references(train_by_id[episode_id], steps, terminal))
    references.sort(key=lambda item: (item.target_step, item.episode_id, item.state_role))

    roles_by_step: dict[int, set[str]] = {}
    for reference in references:
        roles_by_step.setdefault(reference.target_step, set()).add(reference.role_id)
    if extra_state_roles is not None:
        for step, roles in extra_state_roles.items():
            if type(step) is not int or step < 0:
                raise ValueError(f"extra target step must be nonnegative, got {step!r}")
            normalized_roles = list(roles)
            if any(not isinstance(role, str) or not role for role in normalized_roles):
                raise ValueError(f"extra state roles at step {step} must be nonempty strings")
            roles_by_step.setdefault(step, set()).update(normalized_roles)
    states = tuple(
        SelectedEpisodeState(step=step, state_roles=tuple(sorted(roles)))
        for step, roles in sorted(roles_by_step.items())
    )
    return EpisodeSelection(references=tuple(references), states=states)


def _clone_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return value


def _runtime_payload(
    model: nn.Module,
    optimizer: Optimizer,
    config: ExperimentConfig,
    split_hash: str,
    step: int,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state": _clone_tree(model.state_dict()),
        "optimizer_state": _clone_tree(optimizer.state_dict()),
        "optimizer_group_signature": optimizer_group_signature(optimizer),
        "scheduler_state": None,
        "global_step": step,
        "scientific_config_hash": config.scientific_hash(),
        "split_hash": split_hash,
        "optimizer_type": type(optimizer).__name__.lower(),
        **_clone_tree(capture_rng_state()),
    }


def _evaluate_behavior(
    model: nn.Module,
    data: ModularAdditionData,
    grouping: ParameterGrouping,
    device: torch.device,
    modulus: int,
) -> dict[str, Any]:
    behavior, offsets = evaluate_model_behavior(
        model,
        data.inputs.to(device),
        data.labels.to(device),
        data.train_indices.to(device),
        data.test_indices.to(device),
        modulus,
        grouping,
    )
    return {"scalars": behavior, "error_offsets": offsets}


def _build_runtime(
    config: ExperimentConfig,
    checkpoint: Path,
    device: torch.device,
    data: ModularAdditionData,
) -> tuple[nn.Module, Optimizer, ParameterGrouping, int]:
    configure_reproducibility(config.optimization.seed, config.optimization.deterministic)
    model = build_model(config).to(device=device, dtype=torch.float32)
    optimizer, grouping = build_adamw(model, config.optimization)
    validate_optimizer_parameter_identity(model, optimizer)
    step = load_checkpoint(checkpoint, model, optimizer, config, data.split_hash, device)
    return model, optimizer, grouping, step


def _one_bridge(
    source: Path,
    config: ExperimentConfig,
    device: torch.device,
    data: ModularAdditionData,
    midpoint_step: int,
    endpoint_step: int,
    evaluator: BehaviorEvaluator | None,
) -> tuple[ReplayState, ReplayState]:
    model, optimizer, grouping, global_step = _build_runtime(config, source, device, data)
    inputs = data.inputs.to(device)
    labels = data.labels.to(device)
    train_indices = data.train_indices.to(device)
    midpoint: ReplayState | None = None
    endpoint: ReplayState | None = None
    while global_step < endpoint_step:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs.index_select(0, train_indices))[:, -1]
        targets = labels.index_select(0, train_indices)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        global_step += 1
        if global_step in {midpoint_step, endpoint_step}:
            behavior = (
                evaluator(model, data, grouping, device)
                if evaluator is not None
                else _evaluate_behavior(model, data, grouping, device, config.task.modulus)
            )
            payload = _runtime_payload(model, optimizer, config, data.split_hash, global_step)
            state = ReplayState(
                step=global_step,
                semantic_state_sha256=semantic_state_sha256(payload),
                checkpoint_payload=payload,
                behavior=behavior,
            )
            if global_step == midpoint_step:
                midpoint = state
            else:
                endpoint = state
        del logits, targets, loss
    if midpoint is None or endpoint is None:
        raise ReplayValidationError("replay did not capture both midpoint and endpoint states")
    return midpoint, endpoint


def _checkpoint_behavior(
    checkpoint: Path,
    config: ExperimentConfig,
    device: torch.device,
    data: ModularAdditionData,
    evaluator: BehaviorEvaluator | None,
) -> dict[str, Any]:
    model, _, grouping, _ = _build_runtime(config, checkpoint, device, data)
    if evaluator is not None:
        return evaluator(model, data, grouping, device)
    return _evaluate_behavior(model, data, grouping, device, config.task.modulus)


def replay_checkpoint_bridge(
    source_checkpoint: str | Path,
    endpoint_checkpoint: str | Path,
    config: ExperimentConfig,
    midpoint_step: int,
    *,
    device: str | torch.device = "cpu",
    behavior_evaluator: BehaviorEvaluator | None = None,
    repeats: int = 2,
) -> ReplayBridgeResult:
    """Replay ``k -> k+50 -> k+100`` and prove identity with the real endpoint.

    No trainer or artifact writer is invoked. The source files are hashed before and after the
    operation, and any mutation causes the bridge to fail.
    """
    if repeats < 2:
        raise ValueError(f"repeats must be >= 2 for independent replay validation, got {repeats}")
    source = Path(source_checkpoint).resolve()
    endpoint_path = Path(endpoint_checkpoint).resolve()
    if not source.is_file() or not endpoint_path.is_file():
        raise ValueError("source and endpoint checkpoints must both exist")
    source_payload = read_checkpoint(source, "cpu")
    physical_endpoint_payload = read_checkpoint(endpoint_path, "cpu")
    source_step = source_payload.get("global_step")
    endpoint_step = physical_endpoint_payload.get("global_step")
    if type(source_step) is not int or type(endpoint_step) is not int:
        raise ValueError("source and endpoint checkpoints require integer global_step")
    if midpoint_step - source_step != endpoint_step - midpoint_step or source_step >= midpoint_step:
        raise ValueError(
            "midpoint must split source-to-endpoint updates equally: "
            f"source={source_step}, midpoint={midpoint_step}, endpoint={endpoint_step}"
        )
    if source_payload.get("scientific_config_hash") != config.scientific_hash() or (
        physical_endpoint_payload.get("scientific_config_hash") != config.scientific_hash()
    ):
        raise ValueError("replay checkpoint scientific config hash mismatch")
    if source_payload.get("split_hash") != physical_endpoint_payload.get("split_hash"):
        raise ValueError("replay source and endpoint split hashes differ")
    source_sha_before = file_sha256(source)
    endpoint_sha_before = file_sha256(endpoint_path)
    data = generate_modular_addition(
        config.task.modulus, config.task.train_fraction, config.task.split_seed
    )
    if data.split_hash != source_payload.get("split_hash"):
        raise ValueError("replay deterministic split does not match checkpoint split hash")
    old_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if config.optimization.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"replay device is unavailable: {target_device}")

    caller_rng = _clone_tree(capture_rng_state())
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    midpoint_states: list[ReplayState] = []
    endpoint_states: list[ReplayState] = []
    comparisons: list[SemanticStateComparison] = []
    source_unchanged = False
    endpoint_unchanged = False
    try:
        physical_behavior = _checkpoint_behavior(
            endpoint_path, config, target_device, data, behavior_evaluator
        )
        for _ in range(repeats):
            midpoint, replay_endpoint = _one_bridge(
                source,
                config,
                target_device,
                data,
                midpoint_step,
                endpoint_step,
                behavior_evaluator,
            )
            comparison = compare_semantic_states(
                replay_endpoint.checkpoint_payload, physical_endpoint_payload
            )
            if not comparison.equal:
                raise ReplayValidationError(
                    "replay endpoint semantic state differs from manifested checkpoint: "
                    f"{comparison.detail_differences}"
                )
            if replay_endpoint.behavior != physical_behavior:
                raise ReplayValidationError(
                    "replay endpoint behavior differs from manifested checkpoint behavior"
                )
            midpoint_states.append(midpoint)
            endpoint_states.append(replay_endpoint)
            comparisons.append(comparison)
        if len({state.semantic_state_sha256 for state in midpoint_states}) != 1:
            raise ReplayValidationError(
                "independent midpoint replays are not semantically identical"
            )
        if len({state.semantic_state_sha256 for state in endpoint_states}) != 1:
            raise ReplayValidationError(
                "independent endpoint replays are not semantically identical"
            )
        if any(state.behavior != midpoint_states[0].behavior for state in midpoint_states[1:]):
            raise ReplayValidationError("independent midpoint replay behavior differs")
    finally:
        restore_rng_state(caller_rng)
        torch.use_deterministic_algorithms(old_deterministic)
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        if old_workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = old_workspace
        source_unchanged = file_sha256(source) == source_sha_before
        endpoint_unchanged = file_sha256(endpoint_path) == endpoint_sha_before
        if not source_unchanged or not endpoint_unchanged:
            raise ReplayValidationError("replay modified a source checkpoint")
    return ReplayBridgeResult(
        source_checkpoint=source,
        endpoint_checkpoint=endpoint_path,
        source_checkpoint_sha256=source_sha_before,
        endpoint_checkpoint_sha256=endpoint_sha_before,
        source_checkpoint_unchanged=source_unchanged,
        endpoint_checkpoint_unchanged=endpoint_unchanged,
        source_step=source_step,
        midpoint_step=midpoint_step,
        endpoint_step=endpoint_step,
        replay_updates=midpoint_step - source_step,
        midpoint=midpoint_states[0],
        endpoint=endpoint_states[0],
        physical_endpoint_behavior=physical_behavior,
        midpoint_repeat_sha256=tuple(state.semantic_state_sha256 for state in midpoint_states),
        endpoint_repeat_sha256=tuple(state.semantic_state_sha256 for state in endpoint_states),
        endpoint_comparisons=tuple(comparisons),
    )
