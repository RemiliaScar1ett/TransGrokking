"""Run-directory lifecycle and validated machine-readable logging."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transgrokking.config import EventsConfig
from transgrokking.metrics.events import detect_events
from transgrokking.utils.atomic import write_json, write_json_lines

METRICS_SCHEMA_VERSION = 1


def create_run_dir(runs_dir: str | Path, config_hash: str) -> Path:
    """Create a unique run directory and immediately mark it initializing."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(runs_dir) / f"{timestamp}_{config_hash[:8]}"
    for child in ("metrics", "checkpoints", "tensors", "figures", "logs"):
        (run_dir / child).mkdir(parents=True, exist_ok=False)
    write_status(run_dir, "initializing", global_step=None)
    return run_dir


def _load_json_lines(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{source}:{line_number}: invalid JSON") from error
        if not isinstance(record, dict):
            raise ValueError(f"{source}:{line_number}: expected JSON object")
        records.append(record)
    return records


def load_scalar_records(path: str | Path) -> list[dict[str, Any]]:
    """Load M1 scalar records and require strictly increasing steps."""
    records = _load_json_lines(path)
    if any(record.get("schema_version") != METRICS_SCHEMA_VERSION for record in records):
        raise ValueError("scalar timeline is not M1 metrics schema version 1")
    steps = [record.get("step") for record in records]
    if any(type(step) is not int for step in steps) or any(
        current <= previous for previous, current in zip(steps, steps[1:], strict=False)
    ):
        raise ValueError(f"scalar steps are not strictly increasing: {steps}")
    return records


def scalar_steps(path: str | Path) -> list[int]:
    """Return validated M1 scalar evaluation steps."""
    return [int(record["step"]) for record in load_scalar_records(path)]


def append_scalar(path: str | Path, record: dict[str, Any]) -> None:
    """Atomically append one finite scalar record at a strictly newer step."""
    destination = Path(path)
    records = load_scalar_records(destination)
    previous = [int(item["step"]) for item in records]
    step = record.get("step")
    if type(step) is not int or (previous and step <= previous[-1]):
        raise ValueError(f"scalar step {step!r} must be greater than {previous[-1:]}")
    if record.get("schema_version") != METRICS_SCHEMA_VERSION:
        raise ValueError("scalar record requires schema_version=1")
    write_json_lines(destination, [*records, record])


def load_error_offset_records(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate paired train/test error-offset records."""
    records = _load_json_lines(path)
    _validate_error_offset_records(records)
    return records


def _validate_error_offset_records(records: list[dict[str, Any]]) -> None:
    if any(record.get("schema_version") != METRICS_SCHEMA_VERSION for record in records):
        raise ValueError("error-offset timeline is not M1 metrics schema version 1")
    if len(records) % 2:
        raise ValueError("error-offset timeline must contain train/test pairs")
    previous_step: int | None = None
    for index in range(0, len(records), 2):
        train, test = records[index : index + 2]
        step = train.get("step")
        if (
            type(step) is not int
            or test.get("step") != step
            or train.get("split") != "train"
            or test.get("split") != "test"
        ):
            raise ValueError("error-offset records must be ordered train/test pairs at one step")
        if previous_step is not None and step <= previous_step:
            raise ValueError("error-offset steps must be strictly increasing")
        for record in (train, test):
            modulus = record.get("modulus")
            counts = record.get("counts")
            if type(modulus) is not int or not isinstance(counts, list) or len(counts) != modulus:
                raise ValueError("error-offset counts length must equal modulus")
            if any(type(count) is not int or count < 0 for count in counts):
                raise ValueError("error-offset counts must be nonnegative integers")
            if counts[0] != 0:
                raise ValueError("error-offset zero bin must be zero")
        previous_step = step


def append_error_offsets(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Atomically append one train/test error-offset pair."""
    if len(records) != 2:
        raise ValueError("exactly two train/test error-offset records are required")
    existing = load_error_offset_records(path)
    candidate = [*existing, *records]
    _validate_error_offset_records(candidate)
    write_json_lines(path, candidate)


def reconcile_metric_files(run_dir: str | Path) -> None:
    """Discard uncommitted offset tails and require committed evaluations to align."""
    root = Path(run_dir) / "metrics"
    scalars = load_scalar_records(root / "scalars.jsonl")
    offsets = load_error_offset_records(root / "error_offsets.jsonl")
    scalar_step_set = {int(record["step"]) for record in scalars}
    committed_offsets = [record for record in offsets if int(record["step"]) in scalar_step_set]
    if committed_offsets != offsets:
        write_json_lines(root / "error_offsets.jsonl", committed_offsets)
        offsets = committed_offsets
    offset_steps = {int(offsets[index]["step"]) for index in range(0, len(offsets), 2)}
    if scalar_step_set != offset_steps:
        raise ValueError(
            f"committed scalar and error-offset steps differ: {scalar_step_set} != {offset_steps}"
        )


def update_events(
    run_dir: str | Path,
    modulus: int,
    eval_interval: int,
    event_config: EventsConfig,
    *,
    preserve_existing: bool,
) -> dict[str, Any]:
    """Recompute and atomically write events from committed scalar records."""
    root = Path(run_dir)
    path = root / "metrics" / "events.json"
    existing = None
    if preserve_existing and path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    events = detect_events(
        load_scalar_records(root / "metrics" / "scalars.jsonl"),
        root.name,
        modulus,
        eval_interval,
        event_config,
        existing,
    )
    write_json(path, events)
    return events


def append_evaluation_artifacts(
    run_dir: str | Path,
    scalar_record: dict[str, Any],
    offset_records: list[dict[str, Any]],
    modulus: int,
    eval_interval: int,
    event_config: EventsConfig,
) -> None:
    """Commit offsets, scalar marker, then idempotently refresh events."""
    root = Path(run_dir) / "metrics"
    append_error_offsets(root / "error_offsets.jsonl", offset_records)
    append_scalar(root / "scalars.jsonl", scalar_record)
    update_events(run_dir, modulus, eval_interval, event_config, preserve_existing=True)


def copy_metric_prefix(
    parent_run: str | Path,
    child_run: str | Path,
    through_step: int,
    modulus: int,
    eval_interval: int,
    event_config: EventsConfig,
) -> None:
    """Copy a committed M1 timeline prefix into a child run."""
    parent_metrics = Path(parent_run) / "metrics"
    scalars = [
        record
        for record in load_scalar_records(parent_metrics / "scalars.jsonl")
        if int(record["step"]) <= through_step
    ]
    offsets = [
        record
        for record in load_error_offset_records(parent_metrics / "error_offsets.jsonl")
        if int(record["step"]) <= through_step
    ]
    child_metrics = Path(child_run) / "metrics"
    if scalars:
        write_json_lines(child_metrics / "scalars.jsonl", scalars)
        write_json_lines(child_metrics / "error_offsets.jsonl", offsets)
    reconcile_metric_files(child_run)
    update_events(
        child_run,
        modulus,
        eval_interval,
        event_config,
        preserve_existing=False,
    )


def load_manifest(
    run_dir: str | Path, *, allow_unreferenced: set[Path] | None = None
) -> list[dict[str, object]]:
    """Load a manifest and verify unique steps and exact checkpoint file agreement."""
    root = Path(run_dir)
    manifest_path = root / "checkpoints" / "manifest.json"
    if not manifest_path.exists():
        return []
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("checkpoints")
    if payload.get("schema_version") != 2 or not isinstance(entries, list):
        raise ValueError("invalid checkpoint manifest schema")
    steps = [entry.get("step") for entry in entries]
    if any(type(step) is not int for step in steps) or len(steps) != len(set(steps)):
        raise ValueError(f"manifest checkpoint steps must be unique integers: {steps}")
    paths = [entry.get("path") for entry in entries]
    if len(paths) != len(set(paths)):
        raise ValueError("manifest checkpoint paths must be unique")
    referenced = {root / "checkpoints" / str(relative) for relative in paths}
    if any(not path.is_file() for path in referenced):
        raise ValueError("manifest references a missing checkpoint file")
    actual = set((root / "checkpoints").glob("step_*.pt"))
    allowed = allow_unreferenced or set()
    if actual - allowed != referenced or not allowed.issubset(actual):
        raise ValueError("checkpoint files and manifest entries do not match")
    return sorted(entries, key=lambda entry: int(entry["step"]))


def add_manifest_checkpoint(run_dir: str | Path, step: int, checkpoint: str | Path) -> None:
    """Atomically add a fully written checkpoint to the manifest."""
    root = Path(run_dir)
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise ValueError(f"cannot manifest incomplete checkpoint: {checkpoint_path}")
    entries = load_manifest(root, allow_unreferenced={checkpoint_path})
    if any(entry["step"] == step for entry in entries):
        raise ValueError(f"manifest already contains checkpoint step {step}")
    relative = checkpoint_path.relative_to(root / "checkpoints").as_posix()
    entries.append({"step": step, "path": relative})
    entries.sort(key=lambda entry: int(entry["step"]))
    write_json(
        root / "checkpoints" / "manifest.json",
        {"schema_version": 2, "checkpoints": entries},
    )


def write_status(run_dir: str | Path, state: str, **details: Any) -> None:
    """Atomically update run lifecycle state."""
    write_json(
        Path(run_dir) / "status.json",
        {"state": state, "updated_at": datetime.now(timezone.utc).isoformat(), **details},
    )
